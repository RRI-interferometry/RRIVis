"""Sky region filter for spatial subsetting of sky models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import healpy as hp
import numpy as np
from astropy.coordinates import Angle, SkyCoord

from ..containers.footprint import (
    DEFAULT_COVERAGE_FOOTPRINT_COORDINATE_FRAME,
    DEFAULT_COVERAGE_FOOTPRINT_NSIDE,
    SkyFootprint,
    _normalize_coordinate_frame,
)
from ..support.healpix_geometry import pixel_solid_angle

logger = logging.getLogger(__name__)


def _healpix_pixel_centers(
    nside: int,
    pixel_indices: np.ndarray,
    *,
    coordinate_frame: str,
    nest: bool = False,
) -> SkyCoord:
    theta, phi = hp.pix2ang(nside, pixel_indices, nest=nest)
    lat_rad = np.pi / 2 - theta
    if coordinate_frame == "galactic":
        return SkyCoord(l=phi, b=lat_rad, unit="rad", frame="galactic")
    return SkyCoord(ra=phi, dec=lat_rad, unit="rad", frame="icrs")


class SkyRegion(ABC):
    """Sky region filter — cone, box, or union of multiple regions.

    Use the ``cone()``, ``box()``, or ``union()`` class methods to construct.

    Examples
    --------
    >>> region = SkyRegion.cone(ra_deg=83.6, dec_deg=22.0, radius_deg=5.0)
    >>> region = SkyRegion.box(
    ...     ra_deg=180.0, dec_deg=-30.0, width_deg=20.0, height_deg=10.0
    ... )
    >>> region = SkyRegion.union(
    ...     [
    ...         SkyRegion.cone(83.6, 22.0, 5.0),
    ...         SkyRegion.box(180.0, -30.0, 20.0, 10.0),
    ...     ]
    ... )
    """

    # ------------------------------------------------------------------
    # Constructors (preserving the original API)
    # ------------------------------------------------------------------

    @classmethod
    def cone(cls, ra_deg: float, dec_deg: float, radius_deg: float) -> ConeRegion:
        """Create a circular cone-search region.

        Parameters
        ----------
        ra_deg : float
            Right ascension of the cone centre (ICRS, degrees).
        dec_deg : float
            Declination of the cone centre (ICRS, degrees).
        radius_deg : float
            Cone radius (degrees, must be > 0).
        """
        return ConeRegion(ra_deg, dec_deg, radius_deg)

    @classmethod
    def box(
        cls,
        ra_deg: float,
        dec_deg: float,
        width_deg: float,
        height_deg: float,
    ) -> BoxRegion:
        """Create a rectangular box region centred at *(ra, dec)*.

        Parameters
        ----------
        ra_deg : float
            Right ascension of the box centre (ICRS, degrees).
        dec_deg : float
            Declination of the box centre (ICRS, degrees).
        width_deg : float
            Full width in the RA direction (degrees, must be > 0).
        height_deg : float
            Full height in the Dec direction (degrees, must be > 0).
        """
        return BoxRegion(ra_deg, dec_deg, width_deg, height_deg)

    @classmethod
    def union(cls, regions: list[SkyRegion]) -> SkyRegion:
        """Create a union of multiple regions.

        A source inside *any* sub-region is included.  Nested unions are
        flattened automatically.

        Parameters
        ----------
        regions : list of SkyRegion
            Two or more regions to combine.
        """
        return UnionRegion.create(regions)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def contains(self, ra_rad: np.ndarray, dec_rad: np.ndarray) -> np.ndarray:
        """Boolean mask — ``True`` for sources inside the region.

        Parameters
        ----------
        ra_rad, dec_rad : np.ndarray
            Source coordinates in radians (ICRS).

        Returns
        -------
        np.ndarray[bool]
        """

    @abstractmethod
    def healpix_mask(
        self,
        nside: int,
        coordinate_frame: str = "icrs",
        *,
        nest: bool = False,
    ) -> np.ndarray:
        """Boolean mask for HEALPix pixels inside the region.

        Parameters
        ----------
        nside : int
            HEALPix NSIDE parameter.
        coordinate_frame : {"icrs", "galactic"}, default "icrs"
            Coordinate frame used by the HEALPix pixel indexing.
        nest : bool, default False
            HEALPix pixel ordering of the returned mask: RING when False,
            NESTED when True. Must match the ordering of the data the mask
            will index.

        Returns
        -------
        np.ndarray[bool]
            Length ``hp.nside2npix(nside)`` boolean mask, indexed in the
            requested ordering.
        """

    @abstractmethod
    def area_sr(self) -> float:
        """Approximate solid angle of the region in steradians."""

    # ------------------------------------------------------------------
    # Helpers used by subclasses / loaders
    # ------------------------------------------------------------------

    def _iter_atomic(self) -> list[SkyRegion]:
        """Return the list of non-union (cone/box) sub-regions."""
        return [self]

    def footprint(
        self,
        *,
        nside: int = DEFAULT_COVERAGE_FOOTPRINT_NSIDE,
        coordinate_frame: str = DEFAULT_COVERAGE_FOOTPRINT_COORDINATE_FRAME,
    ) -> SkyFootprint:
        """Return a sparse HEALPix footprint for this region."""
        return SkyFootprint.from_mask(
            self.healpix_mask(nside, coordinate_frame=coordinate_frame),
            nside=nside,
            coordinate_frame=coordinate_frame,
        )


class ConeRegion(SkyRegion):
    """Circular cone-search region."""

    def __init__(self, ra_deg: float, dec_deg: float, radius_deg: float) -> None:
        if radius_deg <= 0:
            raise ValueError(f"radius_deg must be positive, got {radius_deg}")
        self.center = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs")
        self.radius = Angle(radius_deg, unit="deg")

    def contains(self, ra_rad: np.ndarray, dec_rad: np.ndarray) -> np.ndarray:
        coords = SkyCoord(ra=ra_rad, dec=dec_rad, unit="rad", frame="icrs")
        return np.asarray(coords.separation(self.center) <= self.radius)

    def healpix_mask(
        self,
        nside: int,
        coordinate_frame: str = "icrs",
        *,
        nest: bool = False,
    ) -> np.ndarray:
        npix = hp.nside2npix(nside)
        frame = _normalize_coordinate_frame(coordinate_frame)
        center = self.center.galactic if frame == "galactic" else self.center
        lon_rad = center.spherical.lon.rad
        lat_rad = center.spherical.lat.rad
        vec = hp.ang2vec(np.pi / 2 - lat_rad, lon_rad)
        ipix = hp.query_disc(nside, vec, self.radius.rad, nest=nest)
        mask = np.zeros(npix, dtype=bool)
        mask[ipix] = True
        return mask

    def __repr__(self) -> str:
        return (
            f"SkyRegion.cone(ra={self.center.ra.deg:.4f}\u00b0, "
            f"dec={self.center.dec.deg:.4f}\u00b0, "
            f"radius={self.radius.deg:.4f}\u00b0)"
        )

    def area_sr(self) -> float:
        return float(2.0 * np.pi * (1.0 - np.cos(self.radius.rad)))


class BoxRegion(SkyRegion):
    """Rectangular box region centred at *(ra, dec)*."""

    def __init__(
        self,
        ra_deg: float,
        dec_deg: float,
        width_deg: float,
        height_deg: float,
    ) -> None:
        if width_deg <= 0 or height_deg <= 0:
            raise ValueError(
                f"width_deg and height_deg must be positive, "
                f"got width={width_deg}, height={height_deg}"
            )
        # Out-of-range extents are clamped symmetrically: a width spanning
        # more than the full RA circle saturates at 360 deg, and a height
        # spanning more than pole-to-pole saturates at 180 deg. Both
        # dimensions are handled the same way (clamp, not raise).
        if width_deg > 360:
            width_deg = 360.0
        if height_deg > 180:
            height_deg = 180.0
        self.center = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs")
        self.width = Angle(width_deg, unit="deg")
        self.height = Angle(height_deg, unit="deg")

    def contains(self, ra_rad: np.ndarray, dec_rad: np.ndarray) -> np.ndarray:
        ra_deg = np.degrees(np.mod(ra_rad, 2 * np.pi))
        dec_deg = np.degrees(dec_rad)
        half_w = self.width.deg / 2
        half_h = self.height.deg / 2

        dec_ok = (dec_deg >= self.center.dec.deg - half_h) & (
            dec_deg <= self.center.dec.deg + half_h
        )

        ra_min = self.center.ra.deg - half_w
        ra_max = self.center.ra.deg + half_w

        if ra_min < 0:
            ra_ok = (ra_deg >= ra_min + 360) | (ra_deg <= ra_max)
        elif ra_max >= 360:
            ra_ok = (ra_deg >= ra_min) | (ra_deg <= ra_max - 360)
        else:
            ra_ok = (ra_deg >= ra_min) & (ra_deg <= ra_max)

        return ra_ok & dec_ok

    def healpix_mask(
        self,
        nside: int,
        coordinate_frame: str = "icrs",
        *,
        nest: bool = False,
    ) -> np.ndarray:
        frame = _normalize_coordinate_frame(coordinate_frame)
        npix = hp.nside2npix(nside)
        half_h = self.height.rad / 2
        dec_c = self.center.dec.rad
        mask = np.zeros(npix, dtype=bool)
        if frame == "icrs":
            dec_min = max(dec_c - half_h, -np.pi / 2)
            dec_max = min(dec_c + half_h, np.pi / 2)
            ipix = np.asarray(
                hp.query_strip(
                    nside,
                    np.pi / 2 - dec_max,
                    np.pi / 2 - dec_min,
                    nest=nest,
                ),
                dtype=np.int64,
            )
            if ipix.size == 0:
                return mask
            centers = _healpix_pixel_centers(
                nside,
                ipix,
                coordinate_frame="icrs",
                nest=nest,
            )
            keep = self.contains(centers.ra.rad, centers.dec.rad)
            mask[ipix[keep]] = True
            return mask

        pixel_indices = np.arange(npix, dtype=np.int64)
        galactic = _healpix_pixel_centers(
            nside,
            pixel_indices,
            coordinate_frame="galactic",
            nest=nest,
        )
        icrs = galactic.icrs
        mask[:] = self.contains(icrs.ra.rad, icrs.dec.rad)
        return mask

    def __repr__(self) -> str:
        return (
            f"SkyRegion.box(ra={self.center.ra.deg:.4f}\u00b0, "
            f"dec={self.center.dec.deg:.4f}\u00b0, "
            f"width={self.width.deg:.4f}\u00b0, "
            f"height={self.height.deg:.4f}\u00b0)"
        )

    def area_sr(self) -> float:
        half_h = self.height.rad / 2.0
        dec_c = self.center.dec.rad
        sin_d1 = np.sin(np.clip(dec_c + half_h, -np.pi / 2, np.pi / 2))
        sin_d2 = np.sin(np.clip(dec_c - half_h, -np.pi / 2, np.pi / 2))
        return float(self.width.rad * (sin_d1 - sin_d2))


class UnionRegion(SkyRegion):
    """Union of multiple regions — a source in *any* sub-region is included."""

    def __init__(self, sub_regions: list[SkyRegion]) -> None:
        self._sub_regions = sub_regions

    @classmethod
    def create(cls, regions: list[SkyRegion]) -> SkyRegion:
        """Create a union, flattening nested unions.

        Returns the sole element directly if *regions* has length 1.
        """
        if not regions:
            raise ValueError("union() requires at least one region")
        flat: list[SkyRegion] = []
        for r in regions:
            if isinstance(r, UnionRegion):
                flat.extend(r._sub_regions)
            else:
                flat.append(r)
        if len(flat) == 1:
            return flat[0]
        return cls(flat)

    def contains(self, ra_rad: np.ndarray, dec_rad: np.ndarray) -> np.ndarray:
        mask = np.zeros(len(ra_rad), dtype=bool)
        for sub in self._sub_regions:
            mask |= sub.contains(ra_rad, dec_rad)
        return mask

    def healpix_mask(
        self,
        nside: int,
        coordinate_frame: str = "icrs",
        *,
        nest: bool = False,
    ) -> np.ndarray:
        npix = hp.nside2npix(nside)
        mask = np.zeros(npix, dtype=bool)
        for sub in self._sub_regions:
            mask |= sub.healpix_mask(
                nside, coordinate_frame=coordinate_frame, nest=nest
            )
        return mask

    def _iter_atomic(self) -> list[SkyRegion]:
        return list(self._sub_regions)

    def __repr__(self) -> str:
        n = len(self._sub_regions)
        return f"SkyRegion.union({n} sub-regions)"

    def area_sr(self, *, nside: int = 512) -> float:
        """Approximate the union area by counting HEALPix pixels in the mask.

        Parameters
        ----------
        nside
            HEALPix resolution used for the discretization. Larger ``nside``
            improves accuracy for unions of small regions; smaller ``nside``
            is sufficient for large unions and runs faster. Defaults to 512.
        """
        mask = self.healpix_mask(nside)
        return float(int(mask.sum()) * pixel_solid_angle(nside))
