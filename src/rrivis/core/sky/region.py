# rrivis/core/sky/region.py
"""Sky region filter for spatial subsetting of sky models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
from astropy.coordinates import Angle, SkyCoord

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class SkyRegion:
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

    center: SkyCoord | None = field(default=None, repr=True)
    _shape: str = field(default="cone", repr=True)
    radius: Angle | None = field(default=None, repr=True)
    width: Angle | None = field(default=None, repr=False)
    height: Angle | None = field(default=None, repr=False)
    _sub_regions: list[SkyRegion] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def cone(cls, ra_deg: float, dec_deg: float, radius_deg: float) -> SkyRegion:
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
        if radius_deg <= 0:
            raise ValueError(f"radius_deg must be positive, got {radius_deg}")
        return cls(
            center=SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs"),
            _shape="cone",
            radius=Angle(radius_deg, unit="deg"),
        )

    @classmethod
    def box(
        cls,
        ra_deg: float,
        dec_deg: float,
        width_deg: float,
        height_deg: float,
    ) -> SkyRegion:
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
        if width_deg <= 0 or height_deg <= 0:
            raise ValueError(
                f"width_deg and height_deg must be positive, "
                f"got width={width_deg}, height={height_deg}"
            )
        return cls(
            center=SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs"),
            _shape="box",
            width=Angle(width_deg, unit="deg"),
            height=Angle(height_deg, unit="deg"),
        )

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
        if not regions:
            raise ValueError("union() requires at least one region")
        # Flatten nested unions
        flat: list[SkyRegion] = []
        for r in regions:
            if r._shape == "union" and r._sub_regions is not None:
                flat.extend(r._sub_regions)
            else:
                flat.append(r)
        if len(flat) == 1:
            return flat[0]
        return cls(_shape="union", _sub_regions=flat)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _iter_atomic(self) -> list[SkyRegion]:
        """Return the list of non-union (cone/box) sub-regions."""
        if self._shape == "union" and self._sub_regions is not None:
            return list(self._sub_regions)
        return [self]

    # ------------------------------------------------------------------
    # Point-source filtering
    # ------------------------------------------------------------------

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
        if self._shape == "union":
            mask = np.zeros(len(ra_rad), dtype=bool)
            for sub in self._sub_regions:  # type: ignore[union-attr]
                mask |= sub.contains(ra_rad, dec_rad)
            return mask

        if self._shape == "cone":
            coords = SkyCoord(ra=ra_rad, dec=dec_rad, unit="rad", frame="icrs")
            return np.asarray(coords.separation(self.center) <= self.radius)

        # box
        ra_deg = np.degrees(ra_rad)
        dec_deg = np.degrees(dec_rad)
        half_w = self.width.deg / 2  # type: ignore[union-attr]
        half_h = self.height.deg / 2  # type: ignore[union-attr]

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

    # ------------------------------------------------------------------
    # HEALPix pixel filtering
    # ------------------------------------------------------------------

    def healpix_mask(self, nside: int) -> np.ndarray:
        """Boolean mask for HEALPix pixels inside the region.

        Parameters
        ----------
        nside : int
            HEALPix NSIDE parameter.

        Returns
        -------
        np.ndarray[bool]
            Length ``hp.nside2npix(nside)`` boolean mask.
        """
        npix = hp.nside2npix(nside)

        if self._shape == "union":
            mask = np.zeros(npix, dtype=bool)
            for sub in self._sub_regions:  # type: ignore[union-attr]
                mask |= sub.healpix_mask(nside)
            return mask

        if self._shape == "cone":
            vec = hp.ang2vec(np.pi / 2 - self.center.dec.rad, self.center.ra.rad)
            ipix = hp.query_disc(nside, vec, self.radius.rad)  # type: ignore[union-attr]
        else:
            # box — 4 corner polygon
            half_w = self.width.rad / 2  # type: ignore[union-attr]
            half_h = self.height.rad / 2  # type: ignore[union-attr]
            dec_c = self.center.dec.rad
            ra_c = self.center.ra.rad
            dec_min = max(dec_c - half_h, -np.pi / 2)
            dec_max = min(dec_c + half_h, np.pi / 2)
            corners = np.array(
                [
                    hp.ang2vec(np.pi / 2 - dec_max, ra_c - half_w),
                    hp.ang2vec(np.pi / 2 - dec_max, ra_c + half_w),
                    hp.ang2vec(np.pi / 2 - dec_min, ra_c + half_w),
                    hp.ang2vec(np.pi / 2 - dec_min, ra_c - half_w),
                ]
            )
            ipix = hp.query_polygon(nside, corners)

        mask = np.zeros(npix, dtype=bool)
        mask[ipix] = True
        return mask

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._shape == "cone":
            return (
                f"SkyRegion.cone(ra={self.center.ra.deg:.4f}°, "
                f"dec={self.center.dec.deg:.4f}°, "
                f"radius={self.radius.deg:.4f}°)"  # type: ignore[union-attr]
            )
        if self._shape == "box":
            return (
                f"SkyRegion.box(ra={self.center.ra.deg:.4f}°, "
                f"dec={self.center.dec.deg:.4f}°, "
                f"width={self.width.deg:.4f}°, "  # type: ignore[union-attr]
                f"height={self.height.deg:.4f}°)"  # type: ignore[union-attr]
            )
        n = len(self._sub_regions) if self._sub_regions else 0
        return f"SkyRegion.union({n} sub-regions)"
