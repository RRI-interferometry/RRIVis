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
    """
    Unified sky model with bidirectional conversion (frozen / immutable).

    Can hold sky data in either point source or HEALPix format, and convert
    between them on demand.  This allows uniform treatment of all sky models
    (catalogs, diffuse emission, test sources) regardless of their native format.

    All methods that would previously mutate ``self`` now return a *new*
    ``SkyModel`` instance created via ``dataclasses.replace``.

    Columnar point-source arrays are ``(N_sources,)`` float64.  ``None`` means
    not populated; ``len == 0`` means loaded but no sources passed the filter.

    HEALPix maps are stored as 2-D arrays of shape ``(n_freq, npix)``.  The
    companion ``_observation_frequencies`` array provides the frequency axis.

    Attributes
    ----------
    _ra_rad, _dec_rad : np.ndarray, optional
        Right ascension and declination in radians.
    _flux : np.ndarray, optional
        Reference flux density in Jy.
    _spectral_index : np.ndarray, optional
        Spectral index for power-law extrapolation.
    _stokes_q, _stokes_u, _stokes_v : np.ndarray, optional
        Stokes polarization parameters.
    _rotation_measure : np.ndarray, optional
        Rotation measure in rad/m^2 (per source) for Faraday rotation of Q/U.
    _major_arcsec, _minor_arcsec, _pa_deg : np.ndarray, optional
        Gaussian morphology: FWHM major/minor axes in arcsec, position angle
        in degrees (N through E). Zero values indicate point sources.
    _spectral_coeffs : np.ndarray, optional
        Multi-term log-polynomial spectral coefficients, shape (N, N_terms).
        Column 0 = spectral index (same as ``_spectral_index``).
    _healpix_maps : np.ndarray, optional
        Multi-frequency HEALPix brightness temperature maps, shape ``(n_freq, npix)``.
        One map per observation channel; generated natively by pygdsm for
        diffuse models, or via spectral extrapolation for point source catalogs.
    _healpix_q_maps : np.ndarray, optional
        Stokes Q HEALPix maps, shape ``(n_freq, npix)``.
    _healpix_u_maps : np.ndarray, optional
        Stokes U HEALPix maps, shape ``(n_freq, npix)``.
    _healpix_v_maps : np.ndarray, optional
        Stokes V HEALPix maps, shape ``(n_freq, npix)``.
    _healpix_nside : int, optional
        HEALPix NSIDE parameter.
    _observation_frequencies : np.ndarray, optional
        Frequency axis for HEALPix maps, shape ``(n_freq,)``.
    _native_format : str
        Original format: ``"point_sources"`` or ``"healpix"``.
    reference_frequency : float, optional
        Reference frequency in Hz for spectral extrapolation of point-source models.
    model_name : str, optional
        Name of the sky model.
    brightness_conversion : BrightnessConversion
        Conversion method for T_b / Jy: ``"planck"`` or ``"rayleigh-jeans"``.
    """

    _ra_rad: np.ndarray | None = field(default=None, repr=False)
    _dec_rad: np.ndarray | None = field(default=None, repr=False)
    _flux: np.ndarray | None = field(default=None, repr=False)
    _spectral_index: np.ndarray | None = field(default=None, repr=False)
    _stokes_q: np.ndarray | None = field(default=None, repr=False)
    _stokes_u: np.ndarray | None = field(default=None, repr=False)
    _stokes_v: np.ndarray | None = field(default=None, repr=False)

    _rotation_measure: np.ndarray | None = field(default=None, repr=False)
    """Rotation measure in rad/m^2 (per source). None = no RM data."""

    _major_arcsec: np.ndarray | None = field(default=None, repr=False)
    """FWHM major axis in arcsec (per source). 0 = point source."""
    _minor_arcsec: np.ndarray | None = field(default=None, repr=False)
    """FWHM minor axis in arcsec (per source). 0 = point source."""
    _pa_deg: np.ndarray | None = field(default=None, repr=False)
    """Position angle in degrees, North through East (per source)."""

    _spectral_coeffs: np.ndarray | None = field(default=None, repr=False)
    """Multi-term log-polynomial spectral coefficients, shape (N_sources, N_terms).
    Column 0 = simple spectral index (same as _spectral_index). None = single power law."""

    _ref_freq: np.ndarray | None = field(default=None, repr=False)
    """Per-source reference frequency in Hz, shape (N_sources,).
    Required for point-source models. None for HEALPix-only models."""

    _healpix_maps: np.ndarray | None = None
    _healpix_q_maps: np.ndarray | None = None
    _healpix_u_maps: np.ndarray | None = None
    _healpix_v_maps: np.ndarray | None = None
    _observation_frequencies: np.ndarray | None = None
    _healpix_nside: int | None = None

    _native_format: SkyFormat = SkyFormat.POINT_SOURCES
    _active_mode: SkyFormat | None = None

    reference_frequency: float | None = None
    model_name: str | None = None
    brightness_conversion: BrightnessConversion = BrightnessConversion.PLANCK

    _precision: "PrecisionConfig | None" = field(default=None, repr=False)

    # --- Inner data containers (auto-built from flat fields) ---
    _ps: Any = field(default=None, repr=False)
    """PointSourceData container. Auto-populated from flat fields in __post_init__."""
    _hp: Any = field(default=None, repr=False)
    """HealpixData container. Auto-populated from flat fields in __post_init__."""

    # =========================================================================
    # Post-init Validation
    # =========================================================================

    def __post_init__(self) -> None:
        """Validate field consistency and cast dtypes."""
        # --- Array length consistency ---
        arr_fields = {
            "_ra_rad": self._ra_rad,
            "_dec_rad": self._dec_rad,
            "_flux": self._flux,
            "_spectral_index": self._spectral_index,
        }
        if self._ref_freq is not None:
            arr_fields["_ref_freq"] = self._ref_freq
        lengths = {k: len(v) for k, v in arr_fields.items() if v is not None}
        if lengths:
            unique_lengths = set(lengths.values())
            if len(unique_lengths) > 1:
                raise ValueError(
                    f"Point-source arrays must all have the same length, got: {lengths}"
                )

        # --- Stokes Q/U/V: all-or-none ---
        stokes_set = {
            "_stokes_q": self._stokes_q is not None,
            "_stokes_u": self._stokes_u is not None,
            "_stokes_v": self._stokes_v is not None,
        }
        stokes_present = [k for k, v in stokes_set.items() if v]
        stokes_absent = [k for k, v in stokes_set.items() if not v]
        if stokes_present and stokes_absent:
            raise ValueError(
                f"Stokes Q/U/V must be all set or all None. "
                f"Present: {stokes_present}, absent: {stokes_absent}"
            )

        # --- Morphology fields: all-or-none ---
        morph_set = {
            "_major_arcsec": self._major_arcsec is not None,
            "_minor_arcsec": self._minor_arcsec is not None,
            "_pa_deg": self._pa_deg is not None,
        }
        morph_present = [k for k, v in morph_set.items() if v]
        morph_absent = [k for k, v in morph_set.items() if not v]
        if morph_present and morph_absent:
            raise ValueError(
                f"Morphology fields (major, minor, pa) must be all set or all None. "
                f"Present: {morph_present}, absent: {morph_absent}"
            )

        # --- Optional per-source array length consistency ---
        if lengths:
            n = next(iter(set(lengths.values())))
            optional_arrays = {
                "_stokes_q": self._stokes_q,
                "_stokes_u": self._stokes_u,
                "_stokes_v": self._stokes_v,
                "_rotation_measure": self._rotation_measure,
                "_major_arcsec": self._major_arcsec,
                "_minor_arcsec": self._minor_arcsec,
                "_pa_deg": self._pa_deg,
            }
            for name, arr in optional_arrays.items():
                if arr is not None and len(arr) != n:
                    raise ValueError(
                        f"{name} has length {len(arr)}, expected {n} "
                        f"(must match core point-source arrays)."
                    )
            if (
                self._spectral_coeffs is not None
                and self._spectral_coeffs.shape[0] != n
            ):
                raise ValueError(
                    f"_spectral_coeffs first dimension is {self._spectral_coeffs.shape[0]}, "
                    f"expected {n} (must match core point-source arrays)."
                )

        # --- HEALPix validation ---
        if self._healpix_maps is not None:
            if self._healpix_nside is None:
                raise ValueError(
                    "_healpix_nside is required when _healpix_maps is not None."
                )
            expected_npix = hp.nside2npix(self._healpix_nside)
            if self._healpix_maps.ndim != 2:
                raise ValueError(
                    f"_healpix_maps must be 2-D (n_freq, npix), "
                    f"got shape {self._healpix_maps.shape}"
                )
            if self._healpix_maps.shape[1] != expected_npix:
                raise ValueError(
                    f"_healpix_maps has {self._healpix_maps.shape[1]} pixels per map, "
                    f"expected {expected_npix} for nside={self._healpix_nside}"
                )
            # Validate observation_frequencies exists and matches n_freq
            n_freq = self._healpix_maps.shape[0]
            if self._observation_frequencies is None:
                raise ValueError(
                    "_observation_frequencies is required when _healpix_maps "
                    "is not None."
                )
            if len(self._observation_frequencies) != n_freq:
                raise ValueError(
                    f"_observation_frequencies has {len(self._observation_frequencies)} "
                    f"entries but _healpix_maps has {n_freq} frequency channels. "
                    f"They must match."
                )
            # Validate polarization map shapes
            for name, arr in [
                ("_healpix_q_maps", self._healpix_q_maps),
                ("_healpix_u_maps", self._healpix_u_maps),
                ("_healpix_v_maps", self._healpix_v_maps),
            ]:
                if arr is not None and arr.shape != self._healpix_maps.shape:
                    raise ValueError(
                        f"{name} shape {arr.shape} does not match "
                        f"_healpix_maps shape {self._healpix_maps.shape}"
                    )

        # (_active_mode is now inferred by the mode property when None)

        # --- _native_format validation (strict: enum required, no coercion) ---
        if not isinstance(self._native_format, SkyFormat):
            raise TypeError(
                f"_native_format must be a SkyFormat enum, got "
                f"{type(self._native_format).__name__}: '{self._native_format}'. "
                f"Use SkyFormat.POINT_SOURCES or SkyFormat.HEALPIX."
            )

        # --- brightness_conversion coercion and validation ---
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

        # --- Default precision if not provided ---
        if self._precision is None:
            from rrivis.core.precision import PrecisionConfig

            object.__setattr__(self, "_precision", PrecisionConfig.standard())

        # --- Auto-build inner data containers from flat fields ---
        from ._data import HealpixData, PointSourceData

        if self._ps is None and self._ra_rad is not None:
            ps = PointSourceData(
                ra_rad=self._ra_rad,
                dec_rad=self._dec_rad,
                flux=self._flux,
                spectral_index=self._spectral_index
                if self._spectral_index is not None
                else np.zeros(len(self._ra_rad), dtype=np.float64),
                stokes_q=self._stokes_q
                if self._stokes_q is not None
                else np.zeros(len(self._ra_rad), dtype=np.float64),
                stokes_u=self._stokes_u
                if self._stokes_u is not None
                else np.zeros(len(self._ra_rad), dtype=np.float64),
                stokes_v=self._stokes_v
                if self._stokes_v is not None
                else np.zeros(len(self._ra_rad), dtype=np.float64),
                ref_freq=self._ref_freq
                if self._ref_freq is not None
                else np.full(
                    len(self._ra_rad),
                    self.reference_frequency or 0.0,
                    dtype=np.float64,
                ),
                rotation_measure=self._rotation_measure,
                major_arcsec=self._major_arcsec,
                minor_arcsec=self._minor_arcsec,
                pa_deg=self._pa_deg,
                spectral_coeffs=self._spectral_coeffs,
            )
            object.__setattr__(self, "_ps", ps)

        if self._hp is None and self._healpix_maps is not None:
            hp_data = HealpixData(
                maps=self._healpix_maps,
                nside=self._healpix_nside,
                frequencies=self._observation_frequencies,
                q_maps=self._healpix_q_maps,
                u_maps=self._healpix_u_maps,
                v_maps=self._healpix_v_maps,
            )
            object.__setattr__(self, "_hp", hp_data)

    # =========================================================================
    # Structured Access Properties
    # =========================================================================

    @property
    def ps(self) -> "PointSourceData | None":
        """Structured point-source data container (read-only).

        Returns None if no point-source data is present.
        Use ``has_point_sources`` to check availability.
        """
        return self._ps

    @property
    def hp(self) -> "HealpixData | None":
        """Structured HEALPix data container (read-only).

        Returns None if no HEALPix maps are present.
        Use ``has_multifreq_maps`` to check availability.
        """
        return self._hp

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

    @staticmethod
    def _apply_precision_casts(
        precision: "PrecisionConfig", kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Cast arrays in *kwargs* to precision-appropriate dtypes.

        Operates on a kwargs dict **before** constructing the frozen dataclass,
        so that ``__post_init__`` never needs to mutate fields.

        Parameters
        ----------
        precision : PrecisionConfig
            Precision configuration.
        kwargs : dict
            Constructor kwargs (modified in place and returned).

        Returns
        -------
        dict[str, Any]
        """
        for category, fields in SkyModel._PRECISION_CATEGORIES.items():
            dt = precision.sky_model.get_dtype(category)
            for key in fields:
                arr = kwargs.get(key)
                if arr is not None:
                    kwargs[key] = arr.astype(dt, copy=False)
        return kwargs

    # =========================================================================
    # Immutable Replace Helper
    # =========================================================================

    def _replace(self, **changes: Any) -> "SkyModel":
        """Return a new ``SkyModel`` with the given fields replaced.

        Parameters
        ----------
        **changes
            Keyword arguments matching field names to override.

        Returns
        -------
        SkyModel
            New instance with the specified changes.
        """
        import dataclasses

        # Resolve precision: inherit from self if not overridden.
        # _precision must always be set -- factory methods handle this.
        precision = changes.get("_precision", self._precision)
        if precision is None:
            raise ValueError(
                "SkyModel._replace(): _precision must not be None. "
                "This is a bug -- all factory methods should set precision "
                "via PrecisionConfig.standard() or an explicit config."
            )

        # Apply dtype casts before constructing the frozen dataclass.
        changes = SkyModel._apply_precision_casts(precision, changes)

        # Clear inner containers so __post_init__ rebuilds them from
        # the (potentially updated) flat fields.
        changes.setdefault("_ps", None)
        changes.setdefault("_hp", None)

        return dataclasses.replace(self, **changes)

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

    def _masked_point_source_kwargs(self, mask: np.ndarray) -> dict[str, Any]:
        """Apply a boolean mask to all per-source arrays.

        Returns a kwargs dict suitable for passing to the ``SkyModel``
        constructor.  ``_spectral_coeffs`` (2-D) is handled separately.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask of shape ``(n_sources,)``.

        Returns
        -------
        dict[str, Any]
        """
        kwargs: dict[str, Any] = {}
        for field_name in self._PER_SOURCE_FIELDS:
            arr = getattr(self, field_name)
            kwargs[field_name] = arr[mask] if arr is not None else None
        kwargs["_spectral_coeffs"] = (
            self._spectral_coeffs[mask] if self._spectral_coeffs is not None else None
        )
        return kwargs

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
            Use ``list_loaders()`` from ``_registry`` to see all names.
        **kwargs
            Forwarded to the registered loader function.

        Returns
        -------
        SkyModel
        """
        from ._registry import get_loader

        return get_loader(name)(**kwargs)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def mode(self) -> SkyFormat:
        """Return the active representation mode for the RIME engine.

        When ``_active_mode`` is explicitly set, returns that value.
        Otherwise infers from populated data: HEALPix takes priority
        (since it's the more specific representation).
        """
        if self._active_mode is not None:
            return self._active_mode
        # Infer from data
        if self._healpix_maps is not None:
            return SkyFormat.HEALPIX
        return SkyFormat.POINT_SOURCES

    @property
    def has_multifreq_maps(self) -> bool:
        """Return True if multi-frequency HEALPix maps are available."""
        return self._healpix_maps is not None

    @property
    def representations(self) -> set[SkyFormat]:
        """Return the set of representations currently populated on this model."""
        result: set[SkyFormat] = set()
        if self._ra_rad is not None and len(self._ra_rad) > 0:
            result.add(SkyFormat.POINT_SOURCES)
        if self._healpix_maps is not None:
            result.add(SkyFormat.HEALPIX)
        return result

    @property
    def n_frequencies(self) -> int:
        """Return the number of frequency channels (0 if no multi-freq maps)."""
        if self._observation_frequencies is not None:
            return len(self._observation_frequencies)
        return 0

    @property
    def native_format(self) -> SkyFormat:
        """Return the native/original format of this sky model."""
        return self._native_format

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
                "Use from_catalog('diffuse_sky', ...) or with_healpix_maps() first."
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
                "Use from_catalog('diffuse_sky', ...) or with_healpix_maps() first."
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

    def _masked_healpix_kwargs(self, region: "SkyRegion") -> dict[str, Any]:
        """Return kwargs dict with region-masked HEALPix map copies.

        Zeros out pixels outside the region in copies of all HEALPix arrays.
        """
        hp_mask = region.healpix_mask(self._healpix_nside)
        inv_mask = ~hp_mask

        new_maps = self._healpix_maps.copy()
        new_maps[:, inv_mask] = 0.0
        result: dict[str, Any] = {
            "_healpix_maps": new_maps,
            "_healpix_nside": self._healpix_nside,
            "_observation_frequencies": self._observation_frequencies,
        }

        for field_name, arr in [
            ("_healpix_q_maps", self._healpix_q_maps),
            ("_healpix_u_maps", self._healpix_u_maps),
            ("_healpix_v_maps", self._healpix_v_maps),
        ]:
            if arr is not None:
                new_arr = arr.copy()
                new_arr[:, inv_mask] = 0.0
                result[field_name] = new_arr

        return result

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
        kwargs: dict[str, Any] = {}

        # Filter HEALPix data if present
        if self._healpix_maps is not None:
            kwargs.update(self._masked_healpix_kwargs(region))

        # Filter point-source data if present
        if self.has_point_sources:
            mask = region.contains(self._ra_rad, self._dec_rad)
            kwargs.update(self._masked_point_source_kwargs(mask))

        if not kwargs:
            return self  # empty model — nothing to filter

        kwargs.update(
            _native_format=self._native_format,
            _active_mode=self._active_mode,
            model_name=self.model_name,
            reference_frequency=self.reference_frequency,
            brightness_conversion=self.brightness_conversion,
            _precision=self._precision,
        )
        return SkyModel(**kwargs)

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

    def with_healpix_maps(
        self,
        nside: int,
        frequencies: np.ndarray | None = None,
        obs_frequency_config: dict[str, Any] | None = None,
        ref_frequency: float | None = None,
        memmap_path: str | None = None,
    ) -> "SkyModel":
        """Convert point sources to multi-frequency HEALPix maps (immutable).

        Creates one HEALPix map per frequency channel. Returns a *new*
        ``SkyModel`` with the healpix fields populated; ``self`` is unchanged.

        Parameters
        ----------
        nside : int
            HEALPix NSIDE parameter.
        frequencies : np.ndarray, optional
            Array of frequencies in Hz. Mutually exclusive with
            *obs_frequency_config*.
        obs_frequency_config : dict, optional
            Observation frequency configuration dict.  Parsed via
            ``parse_frequency_config()``.  Mutually exclusive with
            *frequencies*.
            # TODO: accept a Pydantic model for obs_frequency_config
        ref_frequency : float, optional
            Reference frequency for flux values in Hz. Defaults to
            ``self.reference_frequency``.

        Returns
        -------
        SkyModel
            New instance with HEALPix maps populated.

        Raises
        ------
        ValueError
            If no point sources are available, or if neither *frequencies*
            nor *obs_frequency_config* is provided.
        """
        if not self.has_point_sources:
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
            _healpix_maps=i_maps,
            _healpix_q_maps=q_maps,
            _healpix_u_maps=u_maps,
            _healpix_v_maps=v_maps,
            _healpix_nside=nside,
            _observation_frequencies=frequencies,
            _active_mode=SkyFormat.HEALPIX,
        )

    def with_representation(
        self,
        representation: "SkyFormat | str",
        nside: int = 64,
        flux_limit: float = 0.0,
        frequency: float | None = None,
        frequencies: np.ndarray | None = None,
        obs_frequency_config: dict[str, Any] | None = None,
        memmap_path: str | None = None,
    ) -> "SkyModel":
        """Ensure sky model is in the requested representation.

        Returns a new ``SkyModel`` if conversion is needed, or ``self``
        if already in the correct format.  This is the main method to call
        before visibility calculation.

        Parameters
        ----------
        representation : str
            ``"point_sources"`` or ``"healpix_map"``
        nside : int, default=64
            HEALPix NSIDE for healpix_map mode.
        flux_limit : float, default=0.0
            Minimum flux for point_sources mode (used when converting from
            HEALPix).
        frequency : float, optional
            Frequency for conversion.
        frequencies : np.ndarray, optional
            Frequency array in Hz for point→HEALPix conversion.
            Mutually exclusive with *obs_frequency_config*.
        obs_frequency_config : dict, optional
            Observation frequency config dict for point→HEALPix conversion.
            Mutually exclusive with *frequencies*.

        Returns
        -------
        SkyModel
            Instance with the requested representation populated.

        Raises
        ------
        ValueError
            If ``healpix_map`` is requested but no maps exist and no
            frequency information is provided for auto-conversion.
        """
        # Coerce string to SkyFormat enum
        if isinstance(representation, str) and not isinstance(
            representation, SkyFormat
        ):
            try:
                representation = SkyFormat(representation)
            except ValueError:
                raise ValueError(
                    f"Unknown representation '{representation}'. "
                    f"Supported: SkyFormat.POINT_SOURCES, SkyFormat.HEALPIX."
                ) from None

        freq = frequency or self.reference_frequency

        if representation == SkyFormat.POINT_SOURCES:
            # If point-source data is already available, just switch mode
            if self.has_point_sources:
                if self.mode == SkyFormat.POINT_SOURCES:
                    return self
                return self._replace(_active_mode=SkyFormat.POINT_SOURCES)
            # No point-source data — must convert from healpix (lossy)
            if self._healpix_maps is None:
                return self  # empty model

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
            # Apply flux limit
            if flux_limit > 0:
                mask = arrays["flux"] >= flux_limit
                arrays = {k: v[mask] for k, v in arrays.items()}
            return self._replace(
                _ra_rad=arrays["ra_rad"],
                _dec_rad=arrays["dec_rad"],
                _flux=arrays["flux"],
                _spectral_index=arrays["spectral_index"],
                _stokes_q=arrays["stokes_q"],
                _stokes_u=arrays["stokes_u"],
                _stokes_v=arrays["stokes_v"],
                _ref_freq=arrays["ref_freq"],
                _active_mode=SkyFormat.POINT_SOURCES,
            )

        if representation == SkyFormat.HEALPIX:
            # If healpix data is already available, just switch mode
            if self._healpix_maps is not None:
                if self.mode == SkyFormat.HEALPIX:
                    return self
                return self._replace(_active_mode=SkyFormat.HEALPIX)
            # No healpix data — must convert from point sources
            if frequencies is not None or obs_frequency_config is not None:
                return self.with_healpix_maps(
                    nside=nside,
                    frequencies=frequencies,
                    obs_frequency_config=obs_frequency_config,
                    memmap_path=memmap_path,
                )
            raise ValueError(
                "Cannot convert point sources to HEALPix without frequency "
                "information. Pass frequencies=np.array([...]) or "
                "obs_frequency_config={...} to enable auto-conversion."
            )

        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Supported: 'point_sources', 'healpix_map'."
        )

    def as_point_source_arrays(
        self,
        flux_limit: float = 0.0,
        frequency: float | None = None,
    ) -> SourceArrays:
        """Get sky model as a dict of numpy arrays for visibility calculation.

        Pure function that does **not** mutate ``self``.  If the model is
        HEALPix-native and no point-source arrays are populated, a temporary
        conversion is performed without touching the internal state.

        Parameters
        ----------
        flux_limit : float, default=0.0
            Minimum flux in Jy to include.
        frequency : float, optional
            Frequency for T_b to Jy conversion (used to select the channel
            when in HEALPix mode).

        Returns
        -------
        dict
            Keys: ``ra_rad``, ``dec_rad``, ``flux``, ``spectral_index``,
            ``stokes_q``, ``stokes_u``, ``stokes_v``, ``ref_freq`` (float),
            ``rotation_measure``, ``major_arcsec``, ``minor_arcsec``,
            ``pa_deg``, ``spectral_coeffs``.  Array values have shape
            ``(n_sources,)`` (or ``(n_sources, n_terms)`` for
            ``spectral_coeffs``).  Optional fields are ``None`` when
            not populated.
        """
        # Determine which arrays to use
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

        # If point-source arrays are not available, convert from HEALPix
        # by delegating to with_representation (single conversion path).
        if ra is None and self._healpix_maps is not None:
            converted = self.with_representation(
                "point_sources", flux_limit=0.0, frequency=frequency
            )
            return converted.as_point_source_arrays(flux_limit=flux_limit)

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
        >>> sky = sky.with_healpix_maps(nside=64, frequencies=freqs)
        >>> map_100mhz = sky.get_map_at_frequency(100e6)
        """
        if self._healpix_maps is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Use with_healpix_maps() first."
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
                "Use with_healpix_maps() first."
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
                "Use with_healpix_maps() first."
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
                "Use with_healpix_maps() first."
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
                "Use with_healpix_maps() first."
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
                "No HEALPix maps to back with memmap. Use with_healpix_maps() first."
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

        changes: dict[str, Any] = {
            "_healpix_maps": _to_memmap(self._healpix_maps, "i_maps"),
        }
        if self._healpix_q_maps is not None:
            changes["_healpix_q_maps"] = _to_memmap(self._healpix_q_maps, "q_maps")
        if self._healpix_u_maps is not None:
            changes["_healpix_u_maps"] = _to_memmap(self._healpix_u_maps, "u_maps")
        if self._healpix_v_maps is not None:
            changes["_healpix_v_maps"] = _to_memmap(self._healpix_v_maps, "v_maps")

        return self._replace(**changes)

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
