# rrivis/core/sky/model.py
"""
Unified SkyModel dataclass (frozen / immutable).

Holds sky data in either point-source (columnar arrays) or multi-frequency
HEALPix format, with bidirectional conversion, combination, and precision
management.  All "mutation" methods return *new* instances via
``dataclasses.replace``.

HEALPix maps are stored as 2-D numpy arrays of shape ``(n_freq, npix)``
(dtype float32 by default) rather than ``dict[float, ndarray]``.  The
accompanying ``_observation_frequencies`` array (shape ``(n_freq,)``)
provides the frequency axis.
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord

from ._data import HealpixData, PointSourceData, SourceArrays, empty_source_arrays
from .constants import BrightnessConversion

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .region import SkyRegion

logger = logging.getLogger(__name__)

# Backward-compat alias for internal use (convert.py imports this)
_empty_source_arrays = empty_source_arrays


# =============================================================================
# Sky Format Enum
# =============================================================================


class SkyFormat(Enum):
    """Sky model representation format."""

    POINT_SOURCES = "point_sources"
    HEALPIX = "healpix_map"


# =============================================================================
# SkyModel Class
# =============================================================================


@dataclass(frozen=True)
class SkyModel:
    """Unified immutable sky model built from typed payloads."""

    point: PointSourceData | None = field(default=None, repr=False)
    healpix: HealpixData | None = field(default=None, repr=False)
    native_representation: SkyFormat = SkyFormat.POINT_SOURCES
    active_representation: SkyFormat | None = None
    reference_frequency: float | None = None
    model_name: str | None = None
    brightness_conversion: BrightnessConversion = BrightnessConversion.PLANCK
    _precision: "PrecisionConfig | None" = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate payload consistency and precision defaults."""
        if self.point is None and self.healpix is None:
            raise ValueError(
                "SkyModel requires at least one payload. "
                "Use SkyModel.empty_sky() for an empty point-source model."
            )

        if not isinstance(self.native_representation, SkyFormat):
            raise TypeError(
                f"native_representation must be a SkyFormat enum, got "
                f"{type(self.native_representation).__name__}: "
                f"{self.native_representation!r}."
            )

        if not isinstance(self.brightness_conversion, BrightnessConversion):
            try:
                object.__setattr__(
                    self,
                    "brightness_conversion",
                    BrightnessConversion(self.brightness_conversion),
                )
            except (ValueError, KeyError):
                raise ValueError(
                    f"brightness_conversion must be 'planck' or 'rayleigh-jeans', "
                    f"got '{self.brightness_conversion}'"
                ) from None

        if self.native_representation == SkyFormat.POINT_SOURCES and self.point is None:
            raise ValueError(
                "native_representation='point_sources' requires a point payload."
            )
        if self.native_representation == SkyFormat.HEALPIX and self.healpix is None:
            raise ValueError(
                "native_representation='healpix_map' requires a HEALPix payload."
            )

        if self._precision is None:
            from rrivis.core.precision import PrecisionConfig

            object.__setattr__(self, "_precision", PrecisionConfig.standard())

        if self.point is not None:
            object.__setattr__(
                self, "point", self._cast_point_data(self.point, self._precision)
            )
        if self.healpix is not None:
            object.__setattr__(
                self, "healpix", self._cast_healpix_data(self.healpix, self._precision)
            )

        active = self.active_representation or self.native_representation
        if active not in self.available_representations:
            raise ValueError(
                f"active_representation={active.value!r} is not available on this model. "
                f"Available: {[r.value for r in sorted(self.available_representations, key=lambda r: r.value)]}"
            )
        object.__setattr__(self, "active_representation", active)

    @classmethod
    def _cast_point_data(
        cls,
        point: PointSourceData | None,
        precision: "PrecisionConfig | None",
    ) -> PointSourceData | None:
        if point is None or precision is None:
            return point
        src_dt = precision.sky_model.get_dtype("source_positions")
        flux_dt = precision.sky_model.get_dtype("flux")
        si_dt = precision.sky_model.get_dtype("spectral_index")
        return PointSourceData(
            ra_rad=np.asarray(point.ra_rad, dtype=src_dt),
            dec_rad=np.asarray(point.dec_rad, dtype=src_dt),
            flux=np.asarray(point.flux, dtype=flux_dt),
            spectral_index=np.asarray(point.spectral_index, dtype=si_dt),
            stokes_q=np.asarray(point.stokes_q, dtype=flux_dt),
            stokes_u=np.asarray(point.stokes_u, dtype=flux_dt),
            stokes_v=np.asarray(point.stokes_v, dtype=flux_dt),
            ref_freq=np.asarray(point.ref_freq, dtype=flux_dt),
            rotation_measure=(
                None
                if point.rotation_measure is None
                else np.asarray(point.rotation_measure, dtype=flux_dt)
            ),
            major_arcsec=(
                None
                if point.major_arcsec is None
                else np.asarray(point.major_arcsec, dtype=src_dt)
            ),
            minor_arcsec=(
                None
                if point.minor_arcsec is None
                else np.asarray(point.minor_arcsec, dtype=src_dt)
            ),
            pa_deg=None
            if point.pa_deg is None
            else np.asarray(point.pa_deg, dtype=src_dt),
            spectral_coeffs=(
                None
                if point.spectral_coeffs is None
                else np.asarray(point.spectral_coeffs, dtype=si_dt)
            ),
        )

    @classmethod
    def _cast_healpix_data(
        cls,
        healpix_data: HealpixData | None,
        precision: "PrecisionConfig | None",
    ) -> HealpixData | None:
        if healpix_data is None or precision is None:
            return healpix_data
        hp_dt = precision.sky_model.get_dtype("healpix_maps")
        flux_dt = precision.sky_model.get_dtype("flux")

        def _cast_map(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            return arr if arr.dtype == hp_dt else arr.astype(hp_dt, copy=False)

        return HealpixData(
            maps=_cast_map(healpix_data.maps),
            nside=healpix_data.nside,
            frequencies=np.asarray(healpix_data.frequencies, dtype=flux_dt),
            q_maps=_cast_map(healpix_data.q_maps),
            u_maps=_cast_map(healpix_data.u_maps),
            v_maps=_cast_map(healpix_data.v_maps),
        )

    @property
    def ps(self) -> "PointSourceData | None":
        """Structured point-source payload."""
        return self.point

    @property
    def hp(self) -> "HealpixData | None":
        """Structured HEALPix payload."""
        return self.healpix

    @property
    def _ps(self) -> "PointSourceData | None":
        return self.point

    @property
    def _hp(self) -> "HealpixData | None":
        return self.healpix

    @property
    def _native_format(self) -> SkyFormat:
        return self.native_representation

    @property
    def _active_mode(self) -> SkyFormat | None:
        return self.active_representation

    @property
    def _ra_rad(self) -> np.ndarray | None:
        return self.point.ra_rad if self.point is not None else None

    @property
    def _dec_rad(self) -> np.ndarray | None:
        return self.point.dec_rad if self.point is not None else None

    @property
    def _flux(self) -> np.ndarray | None:
        return self.point.flux if self.point is not None else None

    @property
    def _spectral_index(self) -> np.ndarray | None:
        return self.point.spectral_index if self.point is not None else None

    @property
    def _stokes_q(self) -> np.ndarray | None:
        return self.point.stokes_q if self.point is not None else None

    @property
    def _stokes_u(self) -> np.ndarray | None:
        return self.point.stokes_u if self.point is not None else None

    @property
    def _stokes_v(self) -> np.ndarray | None:
        return self.point.stokes_v if self.point is not None else None

    @property
    def _rotation_measure(self) -> np.ndarray | None:
        return self.point.rotation_measure if self.point is not None else None

    @property
    def _major_arcsec(self) -> np.ndarray | None:
        return self.point.major_arcsec if self.point is not None else None

    @property
    def _minor_arcsec(self) -> np.ndarray | None:
        return self.point.minor_arcsec if self.point is not None else None

    @property
    def _pa_deg(self) -> np.ndarray | None:
        return self.point.pa_deg if self.point is not None else None

    @property
    def _spectral_coeffs(self) -> np.ndarray | None:
        return self.point.spectral_coeffs if self.point is not None else None

    @property
    def _ref_freq(self) -> np.ndarray | None:
        return self.point.ref_freq if self.point is not None else None

    @property
    def _healpix_maps(self) -> np.ndarray | None:
        return self.healpix.maps if self.healpix is not None else None

    @property
    def _healpix_q_maps(self) -> np.ndarray | None:
        return self.healpix.q_maps if self.healpix is not None else None

    @property
    def _healpix_u_maps(self) -> np.ndarray | None:
        return self.healpix.u_maps if self.healpix is not None else None

    @property
    def _healpix_v_maps(self) -> np.ndarray | None:
        return self.healpix.v_maps if self.healpix is not None else None

    @property
    def _observation_frequencies(self) -> np.ndarray | None:
        return self.healpix.frequencies if self.healpix is not None else None

    @property
    def _healpix_nside(self) -> int | None:
        return self.healpix.nside if self.healpix is not None else None

    # =========================================================================
    # Precision Helpers
    # =========================================================================

    def _source_dtype(self) -> np.dtype:
        """Get dtype for source position arrays (RA/Dec)."""
        return self._precision.sky_model.get_dtype("source_positions")

    def _flux_dtype(self) -> np.dtype:
        """Get dtype for flux and Stokes arrays."""
        return self._precision.sky_model.get_dtype("flux")

    def _spectral_index_dtype(self) -> np.dtype:
        """Get dtype for spectral index arrays."""
        return self._precision.sky_model.get_dtype("spectral_index")

    def _healpix_dtype(self) -> np.dtype:
        """Get dtype for HEALPix brightness temperature maps."""
        return self._precision.sky_model.get_dtype("healpix_maps")

    @staticmethod
    def deg_to_rad_at_precision(
        arr: np.ndarray, precision: "PrecisionConfig | None"
    ) -> np.ndarray:
        """Convert degrees to radians at the precision config's dtype.

        Parameters
        ----------
        arr : np.ndarray
            Array of angles in degrees.
        precision : PrecisionConfig, optional
            Precision configuration object. If None, uses numpy default (float64).

        Returns
        -------
        np.ndarray
            Array of angles in radians, at the dtype set by precision config.
        """
        if precision is None:
            return np.deg2rad(arr)
        src_dt = precision.sky_model.get_dtype("source_positions")
        return np.deg2rad(arr.astype(src_dt, copy=False))

    @staticmethod
    def rad_to_deg_at_precision(
        arr: np.ndarray, precision: "PrecisionConfig | None"
    ) -> np.ndarray:
        """Convert radians to degrees at the precision config's dtype.

        Parameters
        ----------
        arr : np.ndarray
            Array of angles in radians.
        precision : PrecisionConfig, optional
            Precision configuration object. If None, uses numpy default.

        Returns
        -------
        np.ndarray
            Array of angles in degrees, at the dtype set by precision config.
        """
        if precision is None:
            return np.rad2deg(arr)
        src_dt = precision.sky_model.get_dtype("source_positions")
        return np.rad2deg(arr.astype(src_dt, copy=False))

    # Maps precision category names to the field names they govern.
    _PRECISION_CATEGORIES: ClassVar[dict[str, tuple[str, ...]]] = {
        "source_positions": (
            "_ra_rad",
            "_dec_rad",
            "_major_arcsec",
            "_minor_arcsec",
            "_pa_deg",
        ),
        "flux": (
            "_flux",
            "_stokes_q",
            "_stokes_u",
            "_stokes_v",
            "_rotation_measure",
            "_ref_freq",
        ),
        "spectral_index": ("_spectral_index", "_spectral_coeffs"),
        "healpix_maps": (
            "_healpix_maps",
            "_healpix_q_maps",
            "_healpix_u_maps",
            "_healpix_v_maps",
        ),
    }

    # =========================================================================
    # Immutable Replace Helper
    # =========================================================================

    def _replace(self, **changes: Any) -> "SkyModel":
        """Return a new ``SkyModel`` with the given fields replaced."""
        import dataclasses

        precision = changes.pop("_precision", self._precision)
        if precision is None:
            raise ValueError(
                "SkyModel._replace(): _precision must not be None. "
                "This is a bug -- all factory methods should set precision."
            )

        field_changes: dict[str, Any] = {"_precision": precision}

        if "point" in changes:
            field_changes["point"] = self._cast_point_data(
                changes.pop("point"), precision
            )
        elif "_ps" in changes:
            field_changes["point"] = self._cast_point_data(
                changes.pop("_ps"), precision
            )

        if "healpix" in changes:
            field_changes["healpix"] = self._cast_healpix_data(
                changes.pop("healpix"), precision
            )
        elif "_hp" in changes:
            field_changes["healpix"] = self._cast_healpix_data(
                changes.pop("_hp"), precision
            )

        if "_native_format" in changes:
            field_changes["native_representation"] = changes.pop("_native_format")
        if "_active_mode" in changes:
            field_changes["active_representation"] = changes.pop("_active_mode")

        for key in (
            "native_representation",
            "active_representation",
            "reference_frequency",
            "model_name",
            "brightness_conversion",
        ):
            if key in changes:
                field_changes[key] = changes.pop(key)

        if changes:
            unknown = ", ".join(sorted(changes))
            raise TypeError(
                f"SkyModel._replace() received unsupported fields: {unknown}"
            )

        return dataclasses.replace(self, **field_changes)

    # =========================================================================
    # Per-Source Field Helpers
    # =========================================================================

    # Tuple of all per-source 1-D array field names (used by
    # _masked_point_source_kwargs and filter_region).
    _PER_SOURCE_FIELDS: tuple[str, ...] = (
        "_ra_rad",
        "_dec_rad",
        "_flux",
        "_spectral_index",
        "_stokes_q",
        "_stokes_u",
        "_stokes_v",
        "_ref_freq",
        "_rotation_measure",
        "_major_arcsec",
        "_minor_arcsec",
        "_pa_deg",
    )

    def _masked_point_source_data(self, mask: np.ndarray) -> PointSourceData | None:
        """Return point-source payload with a boolean mask applied."""
        if self.point is None:
            return None
        return self.point.masked(mask)

    # =========================================================================
    # Empty Sky / Generic Loader
    # =========================================================================

    @classmethod
    def empty_sky(
        cls,
        model_name: str,
        brightness_conversion: BrightnessConversion = BrightnessConversion.PLANCK,
        precision: "PrecisionConfig | None" = None,
        reference_frequency: float | None = None,
    ) -> "SkyModel":
        """Return an empty point-source SkyModel (zero-length arrays)."""
        from ._factories import create_empty

        return create_empty(
            model_name, brightness_conversion, precision, reference_frequency
        )

    @classmethod
    def from_catalog(cls, name: str, **kwargs: Any) -> "SkyModel":
        """Load a sky model by registered loader name.

        Parameters
        ----------
        name : str
            Loader name (e.g. ``"gleam"``, ``"nvss"``, ``"diffuse_sky"``).
            Use ``list_loaders()`` from ``rrivis.core.sky.registry`` to see all names.
        **kwargs
            Forwarded to the registered loader function.

        Returns
        -------
        SkyModel
        """
        from .registry import get_loader

        return get_loader(name)(**kwargs)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def mode(self) -> SkyFormat:
        """Return the active representation mode for the RIME engine."""
        return self.active_representation or self.native_representation

    @property
    def has_multifreq_maps(self) -> bool:
        """Return True if multi-frequency HEALPix maps are available."""
        return self.healpix is not None

    @property
    def available_representations(self) -> set[SkyFormat]:
        """Return the set of representations populated on this model."""
        result: set[SkyFormat] = set()
        if self.point is not None:
            result.add(SkyFormat.POINT_SOURCES)
        if self.healpix is not None:
            result.add(SkyFormat.HEALPIX)
        return result

    @property
    def representations(self) -> set[SkyFormat]:
        """Backward-compatible alias for available_representations."""
        return self.available_representations

    @property
    def n_frequencies(self) -> int:
        """Return the number of frequency channels (0 if no multi-freq maps)."""
        if self._observation_frequencies is not None:
            return len(self._observation_frequencies)
        return 0

    @property
    def native_format(self) -> SkyFormat:
        """Return the native/original format of this sky model."""
        return self.native_representation

    @property
    def n_sky_elements(self) -> int:
        """Return the count for the active representation.

        For ``point_sources`` mode this is the number of catalog entries.
        For ``healpix_map`` mode this is the number of HEALPix pixels.
        """
        if self.mode == SkyFormat.HEALPIX:
            return self._healpix_maps.shape[1] if self._healpix_maps is not None else 0
        return len(self._ra_rad) if self._ra_rad is not None else 0

    @property
    def n_point_sources(self) -> int:
        """Return the number of point-source catalog entries (0 if none)."""
        return len(self._ra_rad) if self._ra_rad is not None else 0

    @property
    def n_pixels(self) -> int:
        """Return the number of HEALPix pixels (0 if no maps)."""
        return self._healpix_maps.shape[1] if self._healpix_maps is not None else 0

    @property
    def has_polarized_healpix_maps(self) -> bool:
        """Return True if any polarization (Q/U/V) HEALPix maps are populated."""
        return any(
            m is not None
            for m in (self._healpix_q_maps, self._healpix_u_maps, self._healpix_v_maps)
        )

    @property
    def pixel_solid_angle(self) -> float:
        """Solid angle per HEALPix pixel in steradians.

        Raises
        ------
        ValueError
            If no HEALPix maps are available.
        """
        if self._healpix_nside is None:
            raise ValueError(
                "No HEALPix maps available. "
                "Use from_catalog('diffuse_sky', ...) or materialize_healpix() first."
            )
        return 4 * np.pi / hp.nside2npix(self._healpix_nside)

    @property
    def pixel_coords(self) -> SkyCoord:
        """SkyCoord of all HEALPix pixel centers (ICRS, RING ordering).

        Returns
        -------
        SkyCoord
            Coordinates of all pixels, length ``hp.nside2npix(nside)``.

        Raises
        ------
        ValueError
            If no HEALPix maps are available.
        """
        if self._healpix_nside is None:
            raise ValueError(
                "No HEALPix maps available. "
                "Use from_catalog('diffuse_sky', ...) or materialize_healpix() first."
            )
        nside = self._healpix_nside
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        return SkyCoord(ra=phi, dec=np.pi / 2 - theta, unit="rad", frame="icrs")

    @property
    def has_point_sources(self) -> bool:
        """Return True if columnar point-source arrays are populated and non-empty."""
        return self._ra_rad is not None and len(self._ra_rad) > 0

    # =========================================================================
    # Public Read-Only Accessors
    # =========================================================================

    # --- Point-source array accessors (delegate to _ps) ---

    @property
    def ra_rad(self) -> np.ndarray | None:
        """Right ascension in radians, shape ``(N,)``."""
        return self._ps.ra_rad if self._ps else None

    @property
    def dec_rad(self) -> np.ndarray | None:
        """Declination in radians, shape ``(N,)``."""
        return self._ps.dec_rad if self._ps else None

    @property
    def flux(self) -> np.ndarray | None:
        """Reference flux density in Jy, shape ``(N,)``."""
        return self._ps.flux if self._ps else None

    @property
    def spectral_index(self) -> np.ndarray | None:
        """Spectral index, shape ``(N,)``."""
        return self._ps.spectral_index if self._ps else None

    @property
    def stokes_q(self) -> np.ndarray | None:
        """Stokes Q in Jy, shape ``(N,)``."""
        return self._ps.stokes_q if self._ps else None

    @property
    def stokes_u(self) -> np.ndarray | None:
        """Stokes U in Jy, shape ``(N,)``."""
        return self._ps.stokes_u if self._ps else None

    @property
    def stokes_v(self) -> np.ndarray | None:
        """Stokes V in Jy, shape ``(N,)``."""
        return self._ps.stokes_v if self._ps else None

    @property
    def ref_freq(self) -> np.ndarray | None:
        """Per-source reference frequency in Hz, shape ``(N,)``."""
        return self._ps.ref_freq if self._ps else None

    @property
    def rotation_measure(self) -> np.ndarray | None:
        """Rotation measure in rad/m^2, shape ``(N,)``."""
        return self._ps.rotation_measure if self._ps else None

    @property
    def major_arcsec(self) -> np.ndarray | None:
        """FWHM major axis in arcsec, shape ``(N,)``."""
        return self._ps.major_arcsec if self._ps else None

    @property
    def minor_arcsec(self) -> np.ndarray | None:
        """FWHM minor axis in arcsec, shape ``(N,)``."""
        return self._ps.minor_arcsec if self._ps else None

    @property
    def pa_deg(self) -> np.ndarray | None:
        """Position angle in degrees (N through E), shape ``(N,)``."""
        return self._ps.pa_deg if self._ps else None

    @property
    def spectral_coeffs(self) -> np.ndarray | None:
        """Multi-term log-polynomial spectral coefficients, shape ``(N, N_terms)``."""
        return self._ps.spectral_coeffs if self._ps else None

    # --- HEALPix accessors (delegate to _hp) ---

    @property
    def healpix_maps(self) -> np.ndarray | None:
        """Stokes I HEALPix maps, shape ``(n_freq, npix)``."""
        return self._hp.maps if self._hp else None

    @property
    def healpix_q_maps(self) -> np.ndarray | None:
        """Stokes Q HEALPix maps, shape ``(n_freq, npix)``."""
        return self._hp.q_maps if self._hp else None

    @property
    def healpix_u_maps(self) -> np.ndarray | None:
        """Stokes U HEALPix maps, shape ``(n_freq, npix)``."""
        return self._hp.u_maps if self._hp else None

    @property
    def healpix_v_maps(self) -> np.ndarray | None:
        """Stokes V HEALPix maps, shape ``(n_freq, npix)``."""
        return self._hp.v_maps if self._hp else None

    @property
    def healpix_nside(self) -> int | None:
        """HEALPix NSIDE parameter."""
        return self._hp.nside if self._hp else None

    @property
    def observation_frequencies(self) -> np.ndarray | None:
        """Frequency axis for HEALPix maps, shape ``(n_freq,)``."""
        return self._hp.frequencies if self._hp else None

    @property
    def precision(self) -> "PrecisionConfig | None":
        """Precision configuration for this model."""
        return self._precision

    @property
    def plot(self) -> Any:
        """Accessor for plotting methods.

        Returns a ``SkyPlotter`` instance that provides methods like
        ``source_positions()``, ``flux_histogram()``, ``mollview()``, etc.

        Usage::

            sky = SkyModel.from_catalog("gleam", ...)
            fig = sky.plot.source_positions()
            fig = sky.plot.flux_histogram()
            fig = sky.plot("auto")  # dispatcher via __call__
        """
        from .plotter import SkyPlotter

        return SkyPlotter(self)

    # =========================================================================
    # Region Filtering
    # =========================================================================

    def _masked_healpix_data(self, region: "SkyRegion") -> HealpixData | None:
        """Return region-masked HEALPix payload."""
        if self.healpix is None:
            return None
        hp_mask = region.healpix_mask(self._healpix_nside)
        inv_mask = ~hp_mask

        new_maps = self._healpix_maps.copy()
        new_maps[:, inv_mask] = 0.0
        q_maps = None
        u_maps = None
        v_maps = None

        if self._healpix_q_maps is not None:
            q_maps = self._healpix_q_maps.copy()
            q_maps[:, inv_mask] = 0.0
        if self._healpix_u_maps is not None:
            u_maps = self._healpix_u_maps.copy()
            u_maps[:, inv_mask] = 0.0
        if self._healpix_v_maps is not None:
            v_maps = self._healpix_v_maps.copy()
            v_maps[:, inv_mask] = 0.0

        return HealpixData(
            maps=new_maps,
            nside=self._healpix_nside,
            frequencies=self._observation_frequencies,
            q_maps=q_maps,
            u_maps=u_maps,
            v_maps=v_maps,
        )

    def filter_region(self, region: "SkyRegion") -> "SkyModel":
        """Return a new SkyModel containing only sources/pixels within *region*.

        For point-source data, applies a boolean mask to all columnar
        arrays.  For HEALPix data, sets out-of-region pixels to ``0.0``.
        When both representations are present, both are filtered.

        Does **not** mutate ``self`` -- always returns a new instance.

        Parameters
        ----------
        region : SkyRegion
            Sky region to filter to (cone, box, or union).

        Returns
        -------
        SkyModel
            Filtered copy.
        """
        point = self.point
        healpix = self.healpix

        if self.healpix is not None:
            healpix = self._masked_healpix_data(region)

        if self.point is not None:
            mask = region.contains(self._ra_rad, self._dec_rad)
            point = self._masked_point_source_data(mask)

        if point is self.point and healpix is self.healpix:
            return self  # empty model — nothing to filter

        return self._replace(
            point=point,
            healpix=healpix,
            native_representation=self.native_representation,
            active_representation=self.active_representation,
            model_name=self.model_name,
            reference_frequency=self.reference_frequency,
            brightness_conversion=self.brightness_conversion,
            _precision=self._precision,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    # =========================================================================
    # Immutable Conversion Methods
    # =========================================================================

    def with_reference_frequency(self, reference_frequency: float) -> "SkyModel":
        """Return a new SkyModel with the reference frequency changed.

        Parameters
        ----------
        reference_frequency : float
            New reference frequency in Hz.

        Returns
        -------
        SkyModel
            Copy with updated ``reference_frequency``.
        """
        return self._replace(reference_frequency=reference_frequency)

    @staticmethod
    def _coerce_representation(representation: "SkyFormat | str") -> SkyFormat:
        if isinstance(representation, str) and not isinstance(
            representation, SkyFormat
        ):
            try:
                return SkyFormat(representation)
            except ValueError:
                raise ValueError(
                    f"Unknown representation '{representation}'. "
                    f"Supported: SkyFormat.POINT_SOURCES, SkyFormat.HEALPIX."
                ) from None
        return representation

    def activate(self, representation: "SkyFormat | str") -> "SkyModel":
        """Switch the active representation without materializing new data."""
        target = self._coerce_representation(representation)
        if target not in self.available_representations:
            raise ValueError(
                f"Representation '{target.value}' is not populated on this model. "
                f"Available: {[r.value for r in sorted(self.available_representations, key=lambda r: r.value)]}"
            )
        if self.mode == target:
            return self
        return self._replace(active_representation=target)

    def materialize_healpix(
        self,
        nside: int,
        frequencies: np.ndarray | None = None,
        obs_frequency_config: dict[str, Any] | None = None,
        ref_frequency: float | None = None,
        memmap_path: str | None = None,
    ) -> "SkyModel":
        """Materialize a HEALPix payload from the point-source payload.

        The existing point payload is preserved; only the active representation
        changes to HEALPix.
        """
        if self.point is None:
            raise ValueError(
                "No point sources available for conversion. "
                "Load sources first using from_catalog('gleam'), from_catalog('mals'), etc."
            )

        # Resolve frequencies
        if frequencies is not None and obs_frequency_config is not None:
            raise ValueError(
                "Provide either 'frequencies' or 'obs_frequency_config', not both."
            )
        if frequencies is None and obs_frequency_config is not None:
            from rrivis.utils.frequency import parse_frequency_config

            frequencies = parse_frequency_config(obs_frequency_config)
        if frequencies is None:
            raise ValueError(
                "Either 'frequencies' (np.ndarray) or 'obs_frequency_config' "
                "(dict) is required."
            )

        # Resolve reference frequency
        ref_freq = ref_frequency
        if ref_freq is None:
            ref_freq = self.reference_frequency
            if ref_freq is None:
                raise ValueError(
                    "ref_frequency must be provided (no reference_frequency set on "
                    "this SkyModel). Set it via with_reference_frequency() or pass "
                    "ref_frequency explicitly."
                )

        from .convert import point_sources_to_healpix_maps

        # Use per-source ref_freq when available and valid (non-zero) for
        # correct spectral scaling of sources from different catalogs.
        effective_ref_freq: float | np.ndarray = ref_freq
        if self._ref_freq is not None and np.any(self._ref_freq > 0):
            effective_ref_freq = self._ref_freq

        i_maps, q_maps, u_maps, v_maps = point_sources_to_healpix_maps(
            ra_rad=self._ra_rad,
            dec_rad=self._dec_rad,
            flux=self._flux,
            spectral_index=self._spectral_index,
            spectral_coeffs=self._spectral_coeffs,
            stokes_q=self._stokes_q,
            stokes_u=self._stokes_u,
            stokes_v=self._stokes_v,
            rotation_measure=self._rotation_measure,
            nside=nside,
            frequencies=frequencies,
            ref_frequency=effective_ref_freq,
            brightness_conversion=self.brightness_conversion,
            output_dtype=self._healpix_dtype(),
            memmap_path=memmap_path,
        )

        return self._replace(
            healpix=HealpixData(
                maps=i_maps,
                nside=nside,
                frequencies=frequencies,
                q_maps=q_maps,
                u_maps=u_maps,
                v_maps=v_maps,
            ),
            active_representation=SkyFormat.HEALPIX,
        )

    def materialize_point_sources(
        self,
        frequency: float | None = None,
        flux_limit: float = 0.0,
        *,
        lossy: bool = False,
    ) -> "SkyModel":
        """Materialize a point-source payload from the HEALPix payload."""
        if self.point is not None:
            return self.activate(SkyFormat.POINT_SOURCES)

        if self.healpix is None:
            raise ValueError("No HEALPix payload available for conversion.")
        if not lossy:
            raise ValueError(
                "HEALPix-to-point-source conversion is lossy. "
                "Call materialize_point_sources(..., lossy=True) to opt in."
            )

        freq = frequency or self.reference_frequency
        n_freq = (
            len(self._observation_frequencies)
            if self._observation_frequencies is not None
            else 0
        )
        resol_arcmin = float(hp.nside2resol(self._healpix_nside, arcmin=True))
        warnings.warn(
            f"HEALPix-to-point-source conversion is lossy: positions are "
            f"quantized to pixel centers (nside={self._healpix_nside}, "
            f"~{resol_arcmin:.1f}' resolution) and spectral indices are "
            f"fit from {n_freq} channels. Use 'healpix_map' mode for "
            f"full-fidelity diffuse emission.",
            stacklevel=2,
        )

        from .convert import healpix_map_to_point_arrays

        resolve_freq = freq or (
            float(self._observation_frequencies[0])
            if self._observation_frequencies is not None
            else None
        )
        if resolve_freq is None:
            raise ValueError(
                "frequency is required for HEALPix-to-point-source conversion."
            )
        fi = self.resolve_frequency_index(resolve_freq)
        temp_map = self._healpix_maps[fi]
        arrays = healpix_map_to_point_arrays(
            temp_map,
            resolve_freq,
            self.brightness_conversion,
            healpix_q_maps=self._healpix_q_maps,
            healpix_u_maps=self._healpix_u_maps,
            healpix_v_maps=self._healpix_v_maps,
            observation_frequencies=self._observation_frequencies,
            freq_index=fi,
            healpix_maps=self._healpix_maps,
            ref_freq_out=resolve_freq,
        )
        if flux_limit > 0:
            mask = arrays["flux"] >= flux_limit
            arrays = {
                key: (value[mask] if isinstance(value, np.ndarray) else value)
                for key, value in arrays.items()
            }

        return self._replace(
            point=PointSourceData(
                ra_rad=arrays["ra_rad"],
                dec_rad=arrays["dec_rad"],
                flux=arrays["flux"],
                spectral_index=arrays["spectral_index"],
                stokes_q=arrays["stokes_q"],
                stokes_u=arrays["stokes_u"],
                stokes_v=arrays["stokes_v"],
                ref_freq=arrays["ref_freq"],
                rotation_measure=arrays["rotation_measure"],
                major_arcsec=arrays["major_arcsec"],
                minor_arcsec=arrays["minor_arcsec"],
                pa_deg=arrays["pa_deg"],
                spectral_coeffs=arrays["spectral_coeffs"],
            ),
            active_representation=SkyFormat.POINT_SOURCES,
        )

    def as_point_source_arrays(
        self,
        flux_limit: float = 0.0,
        frequency: float | None = None,
    ) -> SourceArrays:
        """Get point-source arrays without performing implicit conversion."""
        if self.point is None:
            hint = (
                " Use materialize_point_sources(frequency=..., lossy=True) first."
                if self.healpix is not None
                else ""
            )
            raise ValueError(f"No point-source payload available.{hint}")

        ra = self._ra_rad
        dec = self._dec_rad
        flux = self._flux
        spectral_index = self._spectral_index
        sq = self._stokes_q
        su = self._stokes_u
        sv = self._stokes_v
        rm = self._rotation_measure
        maj = self._major_arcsec
        minn = self._minor_arcsec
        pa = self._pa_deg
        sc = self._spectral_coeffs

        _ps_ref_freq_val = float(self.reference_frequency or 0.0)

        if ra is None or len(ra) == 0:
            return _empty_source_arrays()

        # Apply flux limit
        if flux_limit > 0 and flux is not None:
            mask = flux >= flux_limit
        else:
            mask = np.ones(len(ra), dtype=bool)

        n = int(mask.sum())
        if n == 0:
            return _empty_source_arrays()

        return {
            "ra_rad": ra[mask],
            "dec_rad": dec[mask],
            "flux": flux[mask],
            "spectral_index": spectral_index[mask]
            if spectral_index is not None
            else np.zeros(n, dtype=np.float64),
            "stokes_q": sq[mask] if sq is not None else np.zeros(n, dtype=np.float64),
            "stokes_u": su[mask] if su is not None else np.zeros(n, dtype=np.float64),
            "stokes_v": sv[mask] if sv is not None else np.zeros(n, dtype=np.float64),
            "ref_freq": self._ref_freq[mask]
            if self._ref_freq is not None
            else np.full(n, _ps_ref_freq_val, dtype=np.float64),
            "rotation_measure": rm[mask] if rm is not None else None,
            "major_arcsec": maj[mask] if maj is not None else None,
            "minor_arcsec": minn[mask] if minn is not None else None,
            "pa_deg": pa[mask] if pa is not None else None,
            "spectral_coeffs": sc[mask] if sc is not None else None,
        }

    # =========================================================================
    # HEALPix Map Accessors (array-indexed)
    # =========================================================================

    def resolve_frequency_index(self, frequency: float) -> int:
        """Resolve a frequency to the nearest index in ``_observation_frequencies``.

        Returns the exact index if present, or the nearest one (with a warning
        if the difference exceeds 1 kHz).

        Parameters
        ----------
        frequency : float
            Frequency in Hz.

        Returns
        -------
        int
            Index into ``_observation_frequencies``.
        """
        if self._observation_frequencies is None:
            raise ValueError("No observation frequencies available.")
        idx = int(np.argmin(np.abs(self._observation_frequencies - frequency)))
        nearest_freq = float(self._observation_frequencies[idx])
        freq_diff_mhz = abs(frequency - nearest_freq) / 1e6
        if freq_diff_mhz > 0.001:
            logger.warning(
                f"Exact frequency {frequency / 1e6:.3f} MHz not found. "
                f"Using nearest: {nearest_freq / 1e6:.3f} MHz "
                f"(diff: {freq_diff_mhz:.3f} MHz)"
            )
        return idx

    def get_map_at_frequency(self, frequency: float) -> np.ndarray:
        """
        Get HEALPix brightness temperature map at a specific frequency.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.

        Returns
        -------
        temp_map : np.ndarray
            Brightness temperature map in Kelvin.

        Raises
        ------
        ValueError
            If no multi-frequency maps are available or frequency not found.

        Notes
        -----
        If the exact frequency is not available, returns the map at the
        nearest available frequency with a warning.

        Examples
        --------
        >>> sky = sky.materialize_healpix(nside=64, frequencies=freqs)
        >>> map_100mhz = sky.get_map_at_frequency(100e6)
        """
        if self._healpix_maps is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Use materialize_healpix() first."
            )
        idx = self.resolve_frequency_index(frequency)
        return self._healpix_maps[idx]

    def get_multifreq_maps(self) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Get all multi-frequency HEALPix maps.

        Returns
        -------
        temp_maps : np.ndarray
            Stokes I brightness temperature maps, shape ``(n_freq, npix)``.
        nside : int
            HEALPix NSIDE parameter.
        frequencies : np.ndarray
            Array of frequencies in Hz.

        Raises
        ------
        ValueError
            If no multi-frequency maps are available.

        Examples
        --------
        >>> maps, nside, freqs = sky.get_multifreq_maps()
        >>> for i, freq in enumerate(freqs):
        ...     print(f"{freq / 1e6:.1f} MHz: max T_b = {maps[i].max():.2f} K")
        """
        if self._healpix_maps is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Use materialize_healpix() first."
            )

        return (self._healpix_maps, self._healpix_nside, self._observation_frequencies)

    def get_stokes_maps_at_frequency(
        self, frequency: float
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Get Stokes I/Q/U/V HEALPix maps at a specific frequency.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.

        Returns
        -------
        I_map : np.ndarray
            Stokes I brightness temperature map (K_RJ).
        Q_map : np.ndarray or None
            Stokes Q map (K_RJ), or None if not available.
        U_map : np.ndarray or None
            Stokes U map (K_RJ), or None if not available.
        V_map : np.ndarray or None
            Stokes V map (K_RJ), or None if not available.

        Raises
        ------
        ValueError
            If no multi-frequency HEALPix maps are available.
        """
        if self._healpix_maps is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Use materialize_healpix() first."
            )
        idx = self.resolve_frequency_index(frequency)
        I_map = self._healpix_maps[idx]
        Q_map = self._healpix_q_maps[idx] if self._healpix_q_maps is not None else None
        U_map = self._healpix_u_maps[idx] if self._healpix_u_maps is not None else None
        V_map = self._healpix_v_maps[idx] if self._healpix_v_maps is not None else None
        return I_map, Q_map, U_map, V_map

    def get_multifreq_stokes_maps(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        int,
        np.ndarray,
    ]:
        """Get all multi-frequency Stokes I/Q/U/V HEALPix maps.

        Returns
        -------
        I_maps : np.ndarray
            Stokes I maps, shape ``(n_freq, npix)``.
        Q_maps : np.ndarray or None
            Stokes Q maps, shape ``(n_freq, npix)``, or None.
        U_maps : np.ndarray or None
            Stokes U maps, shape ``(n_freq, npix)``, or None.
        V_maps : np.ndarray or None
            Stokes V maps, shape ``(n_freq, npix)``, or None.
        nside : int
            HEALPix NSIDE parameter.
        frequencies : np.ndarray
            Array of frequencies in Hz.

        Raises
        ------
        ValueError
            If no multi-frequency maps are available.
        """
        if self._healpix_maps is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Use materialize_healpix() first."
            )
        return (
            self._healpix_maps,
            self._healpix_q_maps,
            self._healpix_u_maps,
            self._healpix_v_maps,
            self._healpix_nside,
            self._observation_frequencies,
        )

    # =========================================================================
    # Frequency Iteration & Memory Management
    # =========================================================================

    def iter_frequency_maps(
        self,
    ):
        """Yield ``(freq_hz, I_map, Q_map, U_map, V_map)`` one channel at a time.

        Useful for memory-efficient processing when the full
        ``(n_freq, npix)`` cube is not needed simultaneously.  The
        visibility engine processes one frequency at a time, so this is
        the natural access pattern.

        Yields
        ------
        freq_hz : float
            Frequency in Hz.
        I_map : np.ndarray
            Stokes I brightness temperature map, shape ``(npix,)``.
        Q_map : np.ndarray or None
            Stokes Q map (K_RJ), or None.
        U_map : np.ndarray or None
            Stokes U map (K_RJ), or None.
        V_map : np.ndarray or None
            Stokes V map (K_RJ), or None.

        Raises
        ------
        ValueError
            If no HEALPix maps are available.
        """
        if self._healpix_maps is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Use materialize_healpix() first."
            )
        for i, freq in enumerate(self._observation_frequencies):
            s_i = self._healpix_maps[i]
            s_q = self._healpix_q_maps[i] if self._healpix_q_maps is not None else None
            s_u = self._healpix_u_maps[i] if self._healpix_u_maps is not None else None
            s_v = self._healpix_v_maps[i] if self._healpix_v_maps is not None else None
            yield float(freq), s_i, s_q, s_u, s_v

    def with_memmap_backing(
        self,
        path: str | None = None,
    ) -> "SkyModel":
        """Return a copy with HEALPix maps backed by memory-mapped files.

        ``np.memmap`` is a subclass of ``np.ndarray``, so it passes all
        existing validation unchanged.  The OS page cache handles memory
        management — only pages being actively read are resident in RAM.

        Parameters
        ----------
        path : str, optional
            Directory for memmap files.  If ``None``, uses a temporary
            directory (cleaned up when the process exits).

        Returns
        -------
        SkyModel
            New instance with memmap-backed HEALPix arrays.

        Raises
        ------
        ValueError
            If no HEALPix maps are available.
        """
        import tempfile

        if self._healpix_maps is None:
            raise ValueError(
                "No HEALPix maps to back with memmap. Use materialize_healpix() first."
            )

        if path is None:
            path = tempfile.mkdtemp(prefix="rrivis_memmap_")

        import os

        def _to_memmap(arr: np.ndarray, name: str) -> np.memmap:
            fpath = os.path.join(path, f"{name}.dat")
            mm = np.memmap(fpath, dtype=arr.dtype, mode="w+", shape=arr.shape)
            mm[:] = arr
            mm.flush()
            # Re-open read-only for immutability
            return np.memmap(fpath, dtype=arr.dtype, mode="r", shape=arr.shape)

        healpix = HealpixData(
            maps=_to_memmap(self._healpix_maps, "i_maps"),
            nside=self._healpix_nside,
            frequencies=self._observation_frequencies,
            q_maps=(
                _to_memmap(self._healpix_q_maps, "q_maps")
                if self._healpix_q_maps is not None
                else None
            ),
            u_maps=(
                _to_memmap(self._healpix_u_maps, "u_maps")
                if self._healpix_u_maps is not None
                else None
            ),
            v_maps=(
                _to_memmap(self._healpix_v_maps, "v_maps")
                if self._healpix_v_maps is not None
                else None
            ),
        )

        return self._replace(healpix=healpix)

    # =========================================================================
    # Serialization (delegates to _serialization.py)
    # =========================================================================

    def _to_pyradiosky(self) -> Any:
        """Convert this SkyModel to a pyradiosky.SkyModel for serialization."""
        from ._serialization import to_pyradiosky

        return to_pyradiosky(self)

    def save(
        self,
        filename: str,
        *,
        clobber: bool = False,
        compression: str | None = "gzip",
    ) -> None:
        """Save this SkyModel to SkyH5 format (HDF5 via pyradiosky)."""
        from ._serialization import save_skyh5

        save_skyh5(self, filename, clobber=clobber, compression=compression)

    @classmethod
    def load(
        cls,
        filename: str,
        *,
        precision: "PrecisionConfig | None" = None,
        **kwargs: Any,
    ) -> "SkyModel":
        """Load a SkyModel from a SkyH5 file."""
        from ._serialization import load_skyh5

        return load_skyh5(filename, precision=precision, **kwargs)

    # =========================================================================
    # Parallel Loading (delegates to _factories.py)
    # =========================================================================

    @classmethod
    def load_parallel(
        cls,
        loaders: list[tuple[str, dict[str, Any]]],
        max_workers: int = 8,
        precision: "PrecisionConfig | None" = None,
        strict: bool = True,
    ) -> list["SkyModel"]:
        """Load multiple sky models in parallel using threads."""
        from ._factories import load_models_parallel

        return load_models_parallel(loaders, max_workers, precision, strict)

    # =========================================================================
    # Factory Methods (no external deps)
    # =========================================================================

    @classmethod
    def from_arrays(
        cls,
        ra_rad: np.ndarray,
        dec_rad: np.ndarray,
        flux: np.ndarray,
        spectral_index: np.ndarray | None = None,
        stokes_q: np.ndarray | None = None,
        stokes_u: np.ndarray | None = None,
        stokes_v: np.ndarray | None = None,
        rotation_measure: np.ndarray | None = None,
        major_arcsec: np.ndarray | None = None,
        minor_arcsec: np.ndarray | None = None,
        pa_deg: np.ndarray | None = None,
        spectral_coeffs: np.ndarray | None = None,
        ref_freq: np.ndarray | None = None,
        model_name: str = "custom",
        reference_frequency: float | None = None,
        brightness_conversion: BrightnessConversion = BrightnessConversion.PLANCK,
        precision: "PrecisionConfig | None" = None,
    ) -> "SkyModel":
        """Create a SkyModel from numpy arrays.

        This is the preferred numpy-native constructor for point-source models.
        """
        from ._factories import create_from_arrays

        return create_from_arrays(
            ra_rad=ra_rad,
            dec_rad=dec_rad,
            flux=flux,
            spectral_index=spectral_index,
            stokes_q=stokes_q,
            stokes_u=stokes_u,
            stokes_v=stokes_v,
            rotation_measure=rotation_measure,
            major_arcsec=major_arcsec,
            minor_arcsec=minor_arcsec,
            pa_deg=pa_deg,
            spectral_coeffs=spectral_coeffs,
            ref_freq=ref_freq,
            model_name=model_name,
            reference_frequency=reference_frequency,
            brightness_conversion=brightness_conversion,
            precision=precision,
        )

    @classmethod
    def from_freq_dict_maps(
        cls,
        i_maps: dict[float, np.ndarray],
        q_maps: dict[float, np.ndarray] | None,
        u_maps: dict[float, np.ndarray] | None,
        v_maps: dict[float, np.ndarray] | None,
        nside: int,
        **kwargs: Any,
    ) -> "SkyModel":
        """Create a SkyModel from frequency-keyed dicts of HEALPix maps.

        Standard constructor for loaders building dict[float, ndarray].
        """
        from ._factories import create_from_freq_dict_maps

        return create_from_freq_dict_maps(
            i_maps, q_maps, u_maps, v_maps, nside, **kwargs
        )

    @classmethod
    def from_test_sources(
        cls,
        num_sources: int = 100,
        flux_range: tuple[float, float] = (1.0, 10.0),
        dec_deg: float = -30.0,
        spectral_index: float = -0.7,
        distribution: str = "uniform",
        seed: int | None = None,
        dec_range_deg: float | None = None,
        brightness_conversion: BrightnessConversion = BrightnessConversion.PLANCK,
        precision: "PrecisionConfig | None" = None,
        polarization_fraction: float = 0.0,
        polarization_angle_deg: float = 0.0,
        stokes_v_fraction: float = 0.0,
    ) -> "SkyModel":
        """Generate synthetic test sources."""
        from ._factories import create_test_sources

        return create_test_sources(
            num_sources=num_sources,
            flux_range=flux_range,
            dec_deg=dec_deg,
            spectral_index=spectral_index,
            distribution=distribution,
            seed=seed,
            dec_range_deg=dec_range_deg,
            brightness_conversion=brightness_conversion,
            precision=precision,
            polarization_fraction=polarization_fraction,
            polarization_angle_deg=polarization_angle_deg,
            stokes_v_fraction=stokes_v_fraction,
        )

    # =========================================================================
    # Listing (delegates to loader modules)
    # =========================================================================

    # =========================================================================
    # Combination
    # =========================================================================

    @classmethod
    def combine(
        cls,
        models: list["SkyModel"],
        **kwargs: Any,
    ) -> "SkyModel":
        """
        Combine multiple sky models into one.

        Delegates to ``combine.combine_models()``.

        Parameters
        ----------
        models : list of SkyModel
            Sky models to combine.
        **kwargs
            Forwarded to ``combine_models()``.  Common keyword arguments:
            ``representation``, ``nside``, ``frequency``,
            ``obs_frequency_config``, ``ref_frequency``,
            ``brightness_conversion``, ``precision``.

        Returns
        -------
        SkyModel
            Combined sky model.
        """
        from .combine import combine_models

        return combine_models(models, **kwargs)

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        """Return a human-readable summary of the sky model.

        Returns
        -------
        str
            Summary string including native format, model name, active mode,
            and available representations.
        """
        native_val = self._native_format.value
        parts: list[str] = [
            f"native='{native_val}'",
            f"model='{self.model_name}'",
            f"mode='{self.mode.value}'",
        ]

        # Point-source info
        if self._ra_rad is not None and len(self._ra_rad) > 0:
            extras = []
            if self._rotation_measure is not None and np.any(
                self._rotation_measure != 0
            ):
                extras.append("RM")
            if self._major_arcsec is not None and np.any(self._major_arcsec > 0):
                n_gauss = int(np.sum(self._major_arcsec > 0))
                extras.append(f"gaussian={n_gauss}")
            if self._spectral_coeffs is not None and self._spectral_coeffs.shape[1] > 1:
                extras.append(f"spectral_terms={self._spectral_coeffs.shape[1]}")
            extra_str = f", {', '.join(extras)}" if extras else ""
            parts.append(f"n_sources={self.n_point_sources}{extra_str}")

        # HEALPix info
        if self._healpix_maps is not None:
            freqs = self._observation_frequencies
            freq_range = (
                f"{freqs[0] / 1e6:.1f}-{freqs[-1] / 1e6:.1f}"
                if len(freqs) > 1
                else f"{freqs[0] / 1e6:.1f}"
            )
            stokes_components = "I"
            n_stokes = 1
            for attr, letter in [
                ("_healpix_q_maps", "Q"),
                ("_healpix_u_maps", "U"),
                ("_healpix_v_maps", "V"),
            ]:
                if getattr(self, attr) is not None:
                    stokes_components += letter
                    n_stokes += 1
            from .discovery import estimate_healpix_memory

            mem_info = estimate_healpix_memory(
                self._healpix_nside,
                len(freqs),
                np.float32,
                n_stokes=n_stokes,
            )
            parts.append(
                f"nside={self._healpix_nside}, n_freq={len(freqs)}, "
                f"freq_range={freq_range}MHz, stokes='{stokes_components}', "
                f"memory={mem_info['total_mb']:.1f}MB"
            )

        return f"SkyModel({', '.join(parts)})"

    # =========================================================================
    # Equality
    # =========================================================================

    # Disable auto-generated __hash__ since we define __eq__ with numpy arrays.
    # SkyModel remains frozen (immutable) but is not hashable.
    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        """Value equality: compare all scalar and array fields.

        Uses ``np.array_equal`` for array comparisons (exact, handles
        None and shape/dtype mismatches).
        """
        if not isinstance(other, SkyModel):
            return NotImplemented

        # Scalar fields
        if (
            self.model_name != other.model_name
            or self.reference_frequency != other.reference_frequency
            or self.brightness_conversion != other.brightness_conversion
            or self._native_format != other._native_format
            or self.mode != other.mode
            or self._healpix_nside != other._healpix_nside
        ):
            return False

        # Per-source 1-D arrays + spectral_coeffs (2-D)
        for field_name in (*self._PER_SOURCE_FIELDS, "_spectral_coeffs"):
            a = getattr(self, field_name)
            b = getattr(other, field_name)
            if a is None and b is None:
                continue
            if a is None or b is None:
                return False
            if not np.array_equal(a, b):
                return False

        # HEALPix arrays
        for hp_field in (
            "_healpix_maps",
            "_healpix_q_maps",
            "_healpix_u_maps",
            "_healpix_v_maps",
            "_observation_frequencies",
        ):
            a = getattr(self, hp_field)
            b = getattr(other, hp_field)
            if a is None and b is None:
                continue
            if a is None or b is None:
                return False
            if not np.array_equal(a, b):
                return False

        return True

    def is_close(
        self, other: "SkyModel", rtol: float = 1e-7, atol: float = 0.0
    ) -> bool:
        """Approximate equality (useful for round-trip and precision testing).

        Parameters
        ----------
        other : SkyModel
            Model to compare against.
        rtol : float, default 1e-7
            Relative tolerance for ``np.allclose``.
        atol : float, default 0.0
            Absolute tolerance for ``np.allclose``.

        Returns
        -------
        bool
        """
        if not isinstance(other, SkyModel):
            return False

        if (
            self.model_name != other.model_name
            or self.reference_frequency != other.reference_frequency
            or self.brightness_conversion != other.brightness_conversion
            or self._native_format != other._native_format
            or self.mode != other.mode
            or self._healpix_nside != other._healpix_nside
        ):
            return False

        all_fields = (
            *self._PER_SOURCE_FIELDS,
            "_spectral_coeffs",
            "_healpix_maps",
            "_healpix_q_maps",
            "_healpix_u_maps",
            "_healpix_v_maps",
            "_observation_frequencies",
        )
        for field_name in all_fields:
            a = getattr(self, field_name)
            b = getattr(other, field_name)
            if a is None and b is None:
                continue
            if a is None or b is None:
                return False
            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                return False

        return True
