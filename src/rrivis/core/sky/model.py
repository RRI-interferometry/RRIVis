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
from typing import TYPE_CHECKING, Any

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord

from .constants import BrightnessConversion

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .region import SkyRegion

logger = logging.getLogger(__name__)

# =============================================================================
# Mode Constants
# =============================================================================

MODE_POINT_SOURCES = "point_sources"
MODE_HEALPIX = "healpix_multifreq"
NATIVE_POINT_SOURCES = "point_sources"
NATIVE_HEALPIX = "healpix"


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
    _flux_ref : np.ndarray, optional
        Reference flux density in Jy.
    _alpha : np.ndarray, optional
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
        Column 0 = spectral index (same as ``_alpha``).
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
    frequency : float, optional
        Reference frequency in Hz (first channel for multi-freq maps).
    model_name : str, optional
        Name of the sky model.
    brightness_conversion : BrightnessConversion
        Conversion method for T_b / Jy: ``"planck"`` or ``"rayleigh-jeans"``.
    _point_sources_frequency : float, optional
        Frequency used when columnar arrays were populated from a HEALPix map.
        Only meaningful when ``_native_format == "healpix"``; used to
        invalidate the cache when a different frequency is requested.
    """

    _ra_rad: np.ndarray | None = field(default=None, repr=False)
    _dec_rad: np.ndarray | None = field(default=None, repr=False)
    _flux_ref: np.ndarray | None = field(default=None, repr=False)
    _alpha: np.ndarray | None = field(default=None, repr=False)
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
    Column 0 = simple spectral index (same as _alpha). None = single power law."""

    _healpix_maps: np.ndarray | None = None
    _healpix_q_maps: np.ndarray | None = None
    _healpix_u_maps: np.ndarray | None = None
    _healpix_v_maps: np.ndarray | None = None
    _observation_frequencies: np.ndarray | None = None
    _healpix_nside: int | None = None

    _native_format: str = NATIVE_POINT_SOURCES

    frequency: float | None = None
    model_name: str | None = None
    brightness_conversion: BrightnessConversion = "planck"

    _point_sources_frequency: float | None = field(default=None, repr=False)

    _precision: "PrecisionConfig | None" = field(default=None, repr=False)

    _pygdsm_instance: Any = field(default=None, repr=False)

    # =========================================================================
    # Post-init Validation
    # =========================================================================

    def __post_init__(self) -> None:
        """Validate field consistency and cast dtypes."""
        # --- Array length consistency ---
        arr_fields = {
            "_ra_rad": self._ra_rad,
            "_dec_rad": self._dec_rad,
            "_flux_ref": self._flux_ref,
            "_alpha": self._alpha,
        }
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

        # --- _native_format validation ---
        if self._native_format not in (NATIVE_POINT_SOURCES, NATIVE_HEALPIX):
            raise ValueError(
                f"_native_format must be '{NATIVE_POINT_SOURCES}' or "
                f"'{NATIVE_HEALPIX}', got '{self._native_format}'"
            )

        # --- brightness_conversion validation ---
        if self.brightness_conversion not in ("planck", "rayleigh-jeans"):
            raise ValueError(
                f"brightness_conversion must be 'planck' or 'rayleigh-jeans', "
                f"got '{self.brightness_conversion}'"
            )

        # --- Cast dtypes if precision is set ---
        if self._precision is not None:
            self._ensure_dtypes_frozen()

    # =========================================================================
    # Precision Helpers
    # =========================================================================

    def _validate_precision(self) -> None:
        """Raise ValueError if _precision is not set."""
        if self._precision is None:
            raise ValueError(
                "SkyModel requires a PrecisionConfig. Pass precision=PrecisionConfig.standard() "
                "(or .fast(), .precise(), .ultra()) to the factory method. "
                "Example: SkyModel.from_test_sources(num_sources=100, flux_range=(2.0, 8.0), "
                "dec_deg=-30.72, spectral_index=-0.8, precision=PrecisionConfig.standard())"
            )

    def _source_dtype(self) -> np.dtype:
        """Get dtype for source position arrays (RA/Dec)."""
        self._validate_precision()
        return self._precision.sky_model.get_dtype("source_positions")

    def _flux_dtype(self) -> np.dtype:
        """Get dtype for flux and Stokes arrays."""
        self._validate_precision()
        return self._precision.sky_model.get_dtype("flux")

    def _alpha_dtype(self) -> np.dtype:
        """Get dtype for spectral index arrays."""
        self._validate_precision()
        return self._precision.sky_model.get_dtype("spectral_index")

    def _healpix_dtype(self) -> np.dtype:
        """Get dtype for HEALPix brightness temperature maps."""
        self._validate_precision()
        return self._precision.sky_model.get_dtype("healpix_maps")

    @staticmethod
    def _deg_to_rad_at_precision(
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
    def _rad_to_deg_at_precision(
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

    def _ensure_dtypes_frozen(self) -> None:
        """Cast internal arrays to precision-appropriate dtypes.

        Uses ``object.__setattr__`` to work with the frozen dataclass.
        Intermediate calculations (e.g. flux accumulation via ``np.bincount``)
        remain at float64 for numerical correctness; only the final stored
        arrays are cast here.

        Raises
        ------
        ValueError
            If ``_precision`` is None.
        """
        self._validate_precision()

        src_dt = self._source_dtype()
        flux_dt = self._flux_dtype()
        alpha_dt = self._alpha_dtype()
        hp_dt = self._healpix_dtype()

        if self._ra_rad is not None:
            object.__setattr__(self, "_ra_rad", self._ra_rad.astype(src_dt, copy=False))
            object.__setattr__(
                self, "_dec_rad", self._dec_rad.astype(src_dt, copy=False)
            )
        if self._flux_ref is not None:
            object.__setattr__(
                self, "_flux_ref", self._flux_ref.astype(flux_dt, copy=False)
            )
        if self._alpha is not None:
            object.__setattr__(self, "_alpha", self._alpha.astype(alpha_dt, copy=False))
        if self._stokes_q is not None:
            object.__setattr__(
                self, "_stokes_q", self._stokes_q.astype(flux_dt, copy=False)
            )
            object.__setattr__(
                self, "_stokes_u", self._stokes_u.astype(flux_dt, copy=False)
            )
            object.__setattr__(
                self, "_stokes_v", self._stokes_v.astype(flux_dt, copy=False)
            )
        if self._rotation_measure is not None:
            object.__setattr__(
                self,
                "_rotation_measure",
                self._rotation_measure.astype(flux_dt, copy=False),
            )
        if self._major_arcsec is not None:
            object.__setattr__(
                self,
                "_major_arcsec",
                self._major_arcsec.astype(src_dt, copy=False),
            )
            object.__setattr__(
                self,
                "_minor_arcsec",
                self._minor_arcsec.astype(src_dt, copy=False),
            )
            object.__setattr__(self, "_pa_deg", self._pa_deg.astype(src_dt, copy=False))
        if self._spectral_coeffs is not None:
            object.__setattr__(
                self,
                "_spectral_coeffs",
                self._spectral_coeffs.astype(alpha_dt, copy=False),
            )
        if self._healpix_maps is not None:
            object.__setattr__(
                self,
                "_healpix_maps",
                self._healpix_maps.astype(hp_dt, copy=False),
            )
        if self._healpix_q_maps is not None:
            object.__setattr__(
                self,
                "_healpix_q_maps",
                self._healpix_q_maps.astype(hp_dt, copy=False),
            )
        if self._healpix_u_maps is not None:
            object.__setattr__(
                self,
                "_healpix_u_maps",
                self._healpix_u_maps.astype(hp_dt, copy=False),
            )
        if self._healpix_v_maps is not None:
            object.__setattr__(
                self,
                "_healpix_v_maps",
                self._healpix_v_maps.astype(hp_dt, copy=False),
            )

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

        return dataclasses.replace(self, **changes)

    # =========================================================================
    # Empty Sky / Generic Loader
    # =========================================================================

    @classmethod
    def _empty_sky(
        cls,
        model_name: str,
        brightness_conversion: BrightnessConversion = "planck",
        precision: "PrecisionConfig | None" = None,
        frequency: float | None = None,
    ) -> "SkyModel":
        """Return an empty point-source SkyModel (zero-length arrays).

        Parameters
        ----------
        model_name : str
            Name for the model.
        brightness_conversion : BrightnessConversion, default ``"planck"``
            Brightness conversion method.
        precision : PrecisionConfig, optional
            Precision configuration.
        frequency : float, optional
            Reference frequency in Hz.

        Returns
        -------
        SkyModel
        """
        return cls(
            _ra_rad=np.zeros(0, dtype=np.float64),
            _dec_rad=np.zeros(0, dtype=np.float64),
            _flux_ref=np.zeros(0, dtype=np.float64),
            _alpha=np.zeros(0, dtype=np.float64),
            _stokes_q=np.zeros(0, dtype=np.float64),
            _stokes_u=np.zeros(0, dtype=np.float64),
            _stokes_v=np.zeros(0, dtype=np.float64),
            model_name=model_name,
            brightness_conversion=brightness_conversion,
            _precision=precision,
            frequency=frequency,
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
    def mode(self) -> str:
        """Return the current representation mode."""
        if self._healpix_maps is not None:
            return MODE_HEALPIX
        return MODE_POINT_SOURCES

    @property
    def has_multifreq_maps(self) -> bool:
        """Return True if multi-frequency HEALPix maps are available."""
        return self._healpix_maps is not None

    @property
    def n_frequencies(self) -> int:
        """Return the number of frequency channels (0 if no multi-freq maps)."""
        if self._observation_frequencies is not None:
            return len(self._observation_frequencies)
        return 0

    @property
    def native_format(self) -> str:
        """Return the native/original format of this sky model."""
        return self._native_format

    @property
    def pygdsm_model(self) -> Any:
        """Return the retained pygdsm model instance, or None.

        Only populated when ``retain_pygdsm_instance=True`` was passed to
        ``from_diffuse_sky()``. The returned object is the raw pygdsm model
        (e.g. ``GlobalSkyModel``, ``GlobalSkyModel16``, etc.) and exposes
        methods like ``generate()``, ``view()``, ``write_fits()``, and
        attributes like ``generated_map_data``, ``basemap``, ``nside``.

        .. warning::

            Mutating the returned instance (e.g. calling ``generate()`` at a
            new frequency) does **not** update the SkyModel's stored HEALPix
            maps. The instance is provided for read-only / standalone use.

        Returns
        -------
        object or None
            The pygdsm model instance, or None if not retained.
        """
        return self._pygdsm_instance

    @property
    def n_sources(self) -> int:
        """Return the number of sources/pixels."""
        if self._healpix_maps is not None:
            return self._healpix_maps.shape[1]
        return len(self._ra_rad) if self._ra_rad is not None else 0

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
                "Use from_diffuse_sky() or with_healpix_maps() first."
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
                "Use from_diffuse_sky() or with_healpix_maps() first."
            )
        nside = self._healpix_nside
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        return SkyCoord(ra=phi, dec=np.pi / 2 - theta, unit="rad", frame="icrs")

    def _has_point_sources(self) -> bool:
        """Return True if columnar point-source arrays are populated and non-empty."""
        return self._ra_rad is not None and len(self._ra_rad) > 0

    @property
    def plot(self) -> Any:
        """Accessor for plotting methods.

        Returns a ``SkyPlotter`` instance that provides methods like
        ``source_positions()``, ``flux_histogram()``, ``mollview()``, etc.

        Usage::

            sky = SkyModel.from_gleam(...)
            fig = sky.plot.source_positions()
            fig = sky.plot.flux_histogram()
            fig = sky.plot("auto")  # dispatcher via __call__
        """
        from .plotter import SkyPlotter

        return SkyPlotter(self)

    # =========================================================================
    # Region Filtering
    # =========================================================================

    def filter_region(self, region: "SkyRegion") -> "SkyModel":
        """Return a new SkyModel containing only sources/pixels within *region*.

        For point-source models, applies a boolean mask to all columnar
        arrays.  For HEALPix models, sets out-of-region pixels to ``0.0``.

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
        if self._native_format == NATIVE_POINT_SOURCES and self._ra_rad is not None:
            mask = region.contains(self._ra_rad, self._dec_rad)
            return SkyModel(
                _ra_rad=self._ra_rad[mask],
                _dec_rad=self._dec_rad[mask],
                _flux_ref=self._flux_ref[mask] if self._flux_ref is not None else None,
                _alpha=self._alpha[mask] if self._alpha is not None else None,
                _stokes_q=self._stokes_q[mask] if self._stokes_q is not None else None,
                _stokes_u=self._stokes_u[mask] if self._stokes_u is not None else None,
                _stokes_v=self._stokes_v[mask] if self._stokes_v is not None else None,
                _rotation_measure=self._rotation_measure[mask]
                if self._rotation_measure is not None
                else None,
                _major_arcsec=self._major_arcsec[mask]
                if self._major_arcsec is not None
                else None,
                _minor_arcsec=self._minor_arcsec[mask]
                if self._minor_arcsec is not None
                else None,
                _pa_deg=self._pa_deg[mask] if self._pa_deg is not None else None,
                _spectral_coeffs=self._spectral_coeffs[mask]
                if self._spectral_coeffs is not None
                else None,
                _native_format=NATIVE_POINT_SOURCES,
                model_name=self.model_name,
                frequency=self.frequency,
                brightness_conversion=self.brightness_conversion,
                _precision=self._precision,
            )

        if self._healpix_maps is not None:
            hp_mask = region.healpix_mask(self._healpix_nside)
            inv_mask = ~hp_mask

            new_maps = self._healpix_maps.copy()
            new_maps[:, inv_mask] = 0.0

            new_q = None
            new_u = None
            new_v = None
            if self._healpix_q_maps is not None:
                new_q = self._healpix_q_maps.copy()
                new_q[:, inv_mask] = 0.0
            if self._healpix_u_maps is not None:
                new_u = self._healpix_u_maps.copy()
                new_u[:, inv_mask] = 0.0
            if self._healpix_v_maps is not None:
                new_v = self._healpix_v_maps.copy()
                new_v[:, inv_mask] = 0.0

            return SkyModel(
                _healpix_maps=new_maps,
                _healpix_q_maps=new_q,
                _healpix_u_maps=new_u,
                _healpix_v_maps=new_v,
                _healpix_nside=self._healpix_nside,
                _observation_frequencies=self._observation_frequencies,
                _native_format=NATIVE_HEALPIX,
                frequency=self.frequency,
                model_name=self.model_name,
                brightness_conversion=self.brightness_conversion,
                _precision=self._precision,
            )

        # Empty model -- nothing to filter
        return self

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @staticmethod
    def estimate_healpix_memory(
        nside: int,
        n_frequencies: int,
        dtype: np.dtype | type = np.float32,
        n_stokes: int = 1,
    ) -> dict[str, Any]:
        """
        Estimate memory usage for multi-frequency HEALPix maps.

        Parameters
        ----------
        nside : int
            HEALPix NSIDE parameter.
        n_frequencies : int
            Number of frequency channels.
        dtype : np.dtype or type, default=np.float32
            Data type for maps.
        n_stokes : int, default=1
            Number of Stokes components (1 for I-only, 4 for full IQUV).

        Returns
        -------
        dict
            Memory estimation with keys:
            - npix: number of pixels
            - n_freq: number of frequencies
            - n_stokes: number of Stokes components
            - bytes_per_map: bytes for one map
            - total_bytes: total memory in bytes
            - total_mb: total memory in MB
            - total_gb: total memory in GB
            - resolution_arcmin: approximate pixel resolution

        Examples
        --------
        >>> info = SkyModel.estimate_healpix_memory(nside=1024, n_frequencies=20)
        >>> print(f"Memory: {info['total_mb']:.1f} MB")
        Memory: 960.0 MB
        >>> info = SkyModel.estimate_healpix_memory(
        ...     nside=1024, n_frequencies=20, n_stokes=4
        ... )
        >>> print(f"Memory: {info['total_mb']:.1f} MB")
        Memory: 3840.0 MB
        """
        npix = hp.nside2npix(nside)
        bytes_per_value = np.dtype(dtype).itemsize
        bytes_per_map = npix * bytes_per_value
        total_bytes = bytes_per_map * n_frequencies * n_stokes

        # Approximate resolution in arcminutes
        resolution_arcmin = np.sqrt(4 * np.pi / npix) * (180 / np.pi) * 60

        return {
            "npix": npix,
            "n_freq": n_frequencies,
            "n_stokes": n_stokes,
            "bytes_per_map": bytes_per_map,
            "total_bytes": total_bytes,
            "total_mb": total_bytes / 1e6,
            "total_gb": total_bytes / 1e9,
            "resolution_arcmin": resolution_arcmin,
            "dtype": np.dtype(dtype).name,
        }

    # =========================================================================
    # Immutable Conversion Methods
    # =========================================================================

    def with_frequency(self, frequency: float) -> "SkyModel":
        """Return a new SkyModel with the reference frequency changed.

        Parameters
        ----------
        frequency : float
            New reference frequency in Hz.

        Returns
        -------
        SkyModel
            Copy with updated ``frequency``.
        """
        return self._replace(frequency=frequency)

    def with_healpix_maps(
        self,
        nside: int,
        frequencies: np.ndarray | None = None,
        obs_frequency_config: dict[str, Any] | None = None,
        ref_frequency: float | None = None,
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
            ``self.frequency``.

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
        if not self._has_point_sources():
            raise ValueError(
                "No point sources available for conversion. "
                "Load sources first using from_gleam(), from_mals(), etc."
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
            ref_freq = self.frequency
            if ref_freq is None:
                raise ValueError(
                    "ref_frequency must be provided (no frequency set on this SkyModel). "
                    "Set frequency via with_frequency() or pass ref_frequency explicitly."
                )

        from .convert import point_sources_to_healpix_maps

        i_maps, q_maps, u_maps, v_maps = point_sources_to_healpix_maps(
            ra_rad=self._ra_rad,
            dec_rad=self._dec_rad,
            flux_ref=self._flux_ref,
            alpha=self._alpha,
            spectral_coeffs=self._spectral_coeffs,
            stokes_q=self._stokes_q,
            stokes_u=self._stokes_u,
            stokes_v=self._stokes_v,
            rotation_measure=self._rotation_measure,
            nside=nside,
            frequencies=frequencies,
            ref_frequency=ref_freq,
            brightness_conversion=self.brightness_conversion,
            estimate_memory_fn=self.estimate_healpix_memory,
        )

        return self._replace(
            _healpix_maps=i_maps,
            _healpix_q_maps=q_maps,
            _healpix_u_maps=u_maps,
            _healpix_v_maps=v_maps,
            _healpix_nside=nside,
            _observation_frequencies=frequencies,
        )

    def with_representation(
        self,
        representation: str,
        nside: int = 64,
        flux_limit: float = 0.0,
        frequency: float | None = None,
    ) -> "SkyModel":
        """Ensure sky model is in the requested representation.

        Returns a new ``SkyModel`` if conversion is needed, or ``self``
        if already in the correct format.  This is the main method to call
        before visibility calculation.

        Parameters
        ----------
        representation : str
            ``"point_sources"`` or ``"healpix_multifreq"``
        nside : int, default=64
            HEALPix NSIDE for healpix_multifreq mode (unused when already
            in that mode).
        flux_limit : float, default=0.0
            Minimum flux for point_sources mode (used when converting from
            HEALPix).
        frequency : float, optional
            Frequency for conversion.

        Returns
        -------
        SkyModel
            Instance with the requested representation populated.

        Raises
        ------
        ValueError
            If ``healpix_multifreq`` is requested but no maps are available.
        """
        freq = frequency or self.frequency

        if representation == MODE_POINT_SOURCES:
            if self._native_format == NATIVE_HEALPIX and self._healpix_maps is not None:
                if (
                    self._ra_rad is not None
                    and freq is not None
                    and self._point_sources_frequency == freq
                ):
                    return self
                # Convert healpix to point-source arrays
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
                fi = self._resolve_frequency_index(resolve_freq)
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
                )
                # Apply flux limit
                if flux_limit > 0:
                    mask = arrays["flux_ref"] >= flux_limit
                    arrays = {k: v[mask] for k, v in arrays.items()}
                return self._replace(
                    _ra_rad=arrays["ra_rad"],
                    _dec_rad=arrays["dec_rad"],
                    _flux_ref=arrays["flux_ref"],
                    _alpha=arrays["alpha"],
                    _stokes_q=arrays["stokes_q"],
                    _stokes_u=arrays["stokes_u"],
                    _stokes_v=arrays["stokes_v"],
                    _point_sources_frequency=resolve_freq,
                )
            # Already point-source native
            return self

        if representation in ("healpix_map", "healpix_multifreq"):
            if self._healpix_maps is None:
                raise ValueError(
                    "Cannot convert point sources to HEALPix on-the-fly. "
                    "Use with_healpix_maps(nside, frequencies=...) first "
                    "to create multi-frequency HEALPix maps with correct spectral handling."
                )
            return self

        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Supported: 'point_sources', 'healpix_multifreq', 'healpix_map'."
        )

    def as_point_source_dicts(
        self,
        flux_limit: float = 0.0,
        frequency: float | None = None,
    ) -> list[dict[str, Any]]:
        """Get sky model as a list of point-source dictionaries.

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
        list of dict
            Point sources with coords, flux, spectral_index, stokes.
        """
        # Determine which arrays to use
        ra = self._ra_rad
        dec = self._dec_rad
        flux = self._flux_ref
        alpha = self._alpha
        sq = self._stokes_q
        su = self._stokes_u
        sv = self._stokes_v
        rm = self._rotation_measure
        maj = self._major_arcsec
        minn = self._minor_arcsec
        pa = self._pa_deg
        sc = self._spectral_coeffs

        # If point-source arrays are not available, convert from HEALPix
        need_conversion = ra is None
        if (
            not need_conversion
            and self._native_format == NATIVE_HEALPIX
            and frequency is not None
            and self._point_sources_frequency != frequency
        ):
            need_conversion = True

        if need_conversion and self._healpix_maps is not None:
            freq = (
                frequency or self.frequency or float(self._observation_frequencies[0])
            )
            fi = self._resolve_frequency_index(freq)
            temp_map = self._healpix_maps[fi]

            from .convert import healpix_map_to_point_arrays

            arrays = healpix_map_to_point_arrays(
                temp_map,
                freq,
                self.brightness_conversion,
                healpix_q_maps=self._healpix_q_maps,
                healpix_u_maps=self._healpix_u_maps,
                healpix_v_maps=self._healpix_v_maps,
                observation_frequencies=self._observation_frequencies,
                freq_index=fi,
            )
            ra = arrays["ra_rad"]
            dec = arrays["dec_rad"]
            flux = arrays["flux_ref"]
            alpha = arrays["alpha"]
            sq = arrays["stokes_q"]
            su = arrays["stokes_u"]
            sv = arrays["stokes_v"]
            rm = None
            maj = None
            minn = None
            pa = None
            sc = None

        if ra is None or len(ra) == 0:
            return []

        # Apply flux limit
        if flux_limit > 0 and flux is not None:
            mask = flux >= flux_limit
        else:
            mask = np.ones(len(ra), dtype=bool)

        n = int(mask.sum())
        if n == 0:
            return []

        ra_deg = np.rad2deg(ra[mask])
        dec_deg = np.rad2deg(dec[mask])
        coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        flux_m = flux[mask]
        alpha_m = alpha[mask] if alpha is not None else np.zeros(n, dtype=np.float64)
        sq_m = sq[mask] if sq is not None else np.zeros(n, dtype=np.float64)
        su_m = su[mask] if su is not None else np.zeros(n, dtype=np.float64)
        sv_m = sv[mask] if sv is not None else np.zeros(n, dtype=np.float64)
        rm_m = rm[mask] if rm is not None else None
        maj_m = maj[mask] if maj is not None else None
        min_m = minn[mask] if minn is not None else None
        pa_m = pa[mask] if pa is not None else None
        sc_m = sc[mask] if sc is not None else None

        result: list[dict[str, Any]] = []
        for i in range(n):
            d: dict[str, Any] = {
                "coords": coords[i],
                "flux": float(flux_m[i]),
                "spectral_index": float(alpha_m[i]),
                "stokes_q": float(sq_m[i]),
                "stokes_u": float(su_m[i]),
                "stokes_v": float(sv_m[i]),
            }
            if rm_m is not None:
                d["rotation_measure"] = float(rm_m[i])
            if maj_m is not None:
                d["major_arcsec"] = float(maj_m[i])
                d["minor_arcsec"] = float(min_m[i])
                d["pa_deg"] = float(pa_m[i])
            if sc_m is not None:
                d["spectral_coeffs"] = sc_m[i].tolist()
            result.append(d)
        return result

    # =========================================================================
    # HEALPix Map Accessors (array-indexed)
    # =========================================================================

    def _resolve_frequency_index(self, frequency: float) -> int:
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
        idx = self._resolve_frequency_index(frequency)
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
        idx = self._resolve_frequency_index(frequency)
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
    # Serialization (SkyH5 via pyradiosky)
    # =========================================================================

    def _to_pyradiosky(self) -> Any:
        """Convert this SkyModel to a ``pyradiosky.SkyModel`` for serialization.

        Returns
        -------
        pyradiosky.SkyModel

        Raises
        ------
        ValueError
            If the model is empty (no sources/pixels).
        """
        from pyradiosky import SkyModel as PyRadioSkyModel

        if self._native_format == NATIVE_POINT_SOURCES:
            if self._ra_rad is None or len(self._ra_rad) == 0:
                raise ValueError("Cannot save an empty point-source SkyModel.")

            n = len(self._ra_rad)
            prefix = self.model_name or "src"
            names = np.array([f"{prefix}_{i}" for i in range(n)])

            skycoord = SkyCoord(
                ra=self._ra_rad, dec=self._dec_rad, unit="rad", frame="icrs"
            )

            stokes_arr = np.zeros((4, 1, n), dtype=np.float64)
            stokes_arr[0, 0, :] = self._flux_ref
            if self._stokes_q is not None:
                stokes_arr[1, 0, :] = self._stokes_q
            if self._stokes_u is not None:
                stokes_arr[2, 0, :] = self._stokes_u
            if self._stokes_v is not None:
                stokes_arr[3, 0, :] = self._stokes_v

            ref_freq = self.frequency or 1e8
            ref_freq_arr = np.full(n, ref_freq) * u.Hz

            return PyRadioSkyModel(
                name=names,
                skycoord=skycoord,
                stokes=stokes_arr * u.Jy,
                spectral_type="spectral_index",
                spectral_index=self._alpha.copy(),
                reference_frequency=ref_freq_arr,
                component_type="point",
                history=f"RRIVis SkyModel: {self.model_name or 'unknown'}, "
                f"brightness_conversion={self.brightness_conversion}",
            )

        if self._healpix_maps is not None:
            nside = self._healpix_nside
            npix = hp.nside2npix(nside)
            sorted_indices = np.argsort(self._observation_frequencies)
            sorted_freqs = self._observation_frequencies[sorted_indices]
            n_freq = len(sorted_freqs)

            stokes_arr = np.zeros((4, n_freq, npix), dtype=np.float32)
            for out_i, src_i in enumerate(sorted_indices):
                stokes_arr[0, out_i, :] = self._healpix_maps[src_i]
                if self._healpix_q_maps is not None:
                    stokes_arr[1, out_i, :] = self._healpix_q_maps[src_i]
                if self._healpix_u_maps is not None:
                    stokes_arr[2, out_i, :] = self._healpix_u_maps[src_i]
                if self._healpix_v_maps is not None:
                    stokes_arr[3, out_i, :] = self._healpix_v_maps[src_i]

            from astropy.coordinates import ICRS

            return PyRadioSkyModel(
                nside=nside,
                hpx_order="ring",
                hpx_inds=np.arange(npix),
                stokes=stokes_arr * u.K,
                spectral_type="full",
                freq_array=sorted_freqs * u.Hz,
                component_type="healpix",
                frame=ICRS(),
                history=f"RRIVis SkyModel: {self.model_name or 'unknown'}, "
                f"brightness_conversion={self.brightness_conversion}",
            )

        raise ValueError("Cannot save an empty SkyModel (no sources or maps).")

    def save(
        self,
        filename: str,
        *,
        clobber: bool = False,
        compression: str | None = "gzip",
    ) -> None:
        """Save this SkyModel to SkyH5 format (HDF5 via pyradiosky).

        The file can be loaded back with ``SkyModel.load()`` or
        ``SkyModel.from_pyradiosky_file()``.

        .. note::

            SkyH5 does not preserve rotation measure, Gaussian morphology,
            or multi-term spectral coefficients.  Use ``to_bbs()`` for
            lossless export of these fields.

        Parameters
        ----------
        filename : str
            Output file path (typically ``*.skyh5``).
        clobber : bool, default False
            Overwrite an existing file.
        compression : str or None, default "gzip"
            HDF5 compression for data arrays.
        """
        lost = []
        if self._rotation_measure is not None and np.any(self._rotation_measure != 0):
            lost.append("rotation measure")
        if self._major_arcsec is not None and np.any(self._major_arcsec > 0):
            lost.append("Gaussian morphology")
        if self._spectral_coeffs is not None and self._spectral_coeffs.shape[1] > 1:
            lost.append("multi-term spectral coefficients")
        if lost:
            warnings.warn(
                f"SkyH5 format does not preserve: {', '.join(lost)}. "
                "Use to_bbs() for lossless export.",
                UserWarning,
                stacklevel=2,
            )
        psky = self._to_pyradiosky()
        psky.write_skyh5(filename, clobber=clobber, data_compression=compression)
        logger.info(f"SkyModel saved to {filename}")

    @classmethod
    def load(
        cls,
        filename: str,
        *,
        precision: "PrecisionConfig | None" = None,
        **kwargs: Any,
    ) -> "SkyModel":
        """Load a SkyModel from a SkyH5 file.

        Convenience wrapper around ``from_pyradiosky_file()``.

        Parameters
        ----------
        filename : str
            Path to a ``.skyh5`` file.
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.
        **kwargs
            Forwarded to ``from_pyradiosky_file()``.
        """
        from ._loaders_pyradiosky import load_pyradiosky_file

        return load_pyradiosky_file(
            filename, filetype="skyh5", precision=precision, **kwargs
        )

    # =========================================================================
    # Parallel Loading
    # =========================================================================

    @classmethod
    def load_parallel(
        cls,
        loaders: list[tuple[str, dict[str, Any]]],
        max_workers: int = 8,
        precision: "PrecisionConfig | None" = None,
    ) -> list["SkyModel"]:
        """Load multiple sky models in parallel using threads.

        Each loader is a ``(method_name, kwargs)`` tuple identifying a
        factory classmethod on ``SkyModel`` (e.g. ``"from_gleam"``).
        Downloads run concurrently via ``ThreadPoolExecutor``.  Failed
        loaders are logged as warnings and excluded from the result.

        Parameters
        ----------
        loaders : list of (str, dict)
            ``[(method_name, kwargs), ...]``.
            Example: ``[("from_gleam", {"flux_limit": 0.5})]``
        max_workers : int, default 8
            Maximum concurrent threads.
        precision : PrecisionConfig, optional
            Injected into each loader's kwargs if not already present.

        Returns
        -------
        list of SkyModel
            Successfully loaded models.
        """
        import concurrent.futures

        def _load_one(method_name: str, kw: dict) -> "SkyModel":
            from ._registry import get_loader

            return get_loader(method_name)(**kw)

        n = min(len(loaders), max_workers)
        results: list[SkyModel] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            future_to_name: dict[concurrent.futures.Future, str] = {}
            for method_name, kwargs in loaders:
                kw = dict(kwargs)  # shallow copy to avoid mutating caller's dict
                if precision is not None and "precision" not in kw:
                    kw["precision"] = precision
                f = pool.submit(_load_one, method_name, kw)
                future_to_name[f] = method_name

            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    sky = future.result()
                    if sky.n_sources > 0 or sky._healpix_maps is not None:
                        results.append(sky)
                        logger.info(
                            f"Parallel load complete: {name} "
                            f"({sky.n_sources:,} sources)"
                        )
                    else:
                        logger.info(f"Parallel load: {name} returned empty model")
                except Exception as e:
                    logger.warning(f"Parallel load failed for {name}: {e}")

        return results

    # =========================================================================
    # Factory Methods (no external deps)
    # =========================================================================

    @classmethod
    def from_arrays(
        cls,
        ra_rad: np.ndarray,
        dec_rad: np.ndarray,
        flux_ref: np.ndarray,
        alpha: np.ndarray | None = None,
        stokes_q: np.ndarray | None = None,
        stokes_u: np.ndarray | None = None,
        stokes_v: np.ndarray | None = None,
        rotation_measure: np.ndarray | None = None,
        major_arcsec: np.ndarray | None = None,
        minor_arcsec: np.ndarray | None = None,
        pa_deg: np.ndarray | None = None,
        spectral_coeffs: np.ndarray | None = None,
        model_name: str = "custom",
        frequency: float | None = None,
        brightness_conversion: BrightnessConversion = "planck",
        precision: "PrecisionConfig | None" = None,
    ) -> "SkyModel":
        """Create a SkyModel from numpy arrays.

        This is the preferred numpy-native constructor for point-source models.

        Parameters
        ----------
        ra_rad : np.ndarray
            Right ascension in radians, shape ``(N,)``.
        dec_rad : np.ndarray
            Declination in radians, shape ``(N,)``.
        flux_ref : np.ndarray
            Reference flux density in Jy, shape ``(N,)``.
        alpha : np.ndarray, optional
            Spectral index, shape ``(N,)``. Defaults to -0.7 for all sources.
        stokes_q : np.ndarray, optional
            Stokes Q in Jy, shape ``(N,)``. Defaults to 0.
        stokes_u : np.ndarray, optional
            Stokes U in Jy, shape ``(N,)``. Defaults to 0.
        stokes_v : np.ndarray, optional
            Stokes V in Jy, shape ``(N,)``. Defaults to 0.
        rotation_measure : np.ndarray, optional
            Rotation measure in rad/m^2, shape ``(N,)``.
        major_arcsec : np.ndarray, optional
            FWHM major axis in arcsec, shape ``(N,)``.
        minor_arcsec : np.ndarray, optional
            FWHM minor axis in arcsec, shape ``(N,)``.
        pa_deg : np.ndarray, optional
            Position angle in degrees, shape ``(N,)``.
        spectral_coeffs : np.ndarray, optional
            Log-polynomial spectral coefficients, shape ``(N, N_terms)``.
        model_name : str, default ``"custom"``
            Name for the model.
        frequency : float, optional
            Reference frequency in Hz.
        brightness_conversion : BrightnessConversion, default ``"planck"``
            Brightness conversion method.
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.

        Returns
        -------
        SkyModel
        """
        n = len(ra_rad)
        if alpha is None:
            alpha = np.full(n, -0.7, dtype=np.float64)
        if stokes_q is None:
            stokes_q = np.zeros(n, dtype=np.float64)
        if stokes_u is None:
            stokes_u = np.zeros(n, dtype=np.float64)
        if stokes_v is None:
            stokes_v = np.zeros(n, dtype=np.float64)

        return cls(
            _ra_rad=np.asarray(ra_rad, dtype=np.float64),
            _dec_rad=np.asarray(dec_rad, dtype=np.float64),
            _flux_ref=np.asarray(flux_ref, dtype=np.float64),
            _alpha=np.asarray(alpha, dtype=np.float64),
            _stokes_q=np.asarray(stokes_q, dtype=np.float64),
            _stokes_u=np.asarray(stokes_u, dtype=np.float64),
            _stokes_v=np.asarray(stokes_v, dtype=np.float64),
            _rotation_measure=rotation_measure,
            _major_arcsec=major_arcsec,
            _minor_arcsec=minor_arcsec,
            _pa_deg=pa_deg,
            _spectral_coeffs=spectral_coeffs,
            _native_format=NATIVE_POINT_SOURCES,
            model_name=model_name,
            frequency=frequency,
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )

    @classmethod
    def _from_freq_dict_maps(
        cls,
        i_maps: dict[float, np.ndarray],
        q_maps: dict[float, np.ndarray] | None,
        u_maps: dict[float, np.ndarray] | None,
        v_maps: dict[float, np.ndarray] | None,
        nside: int,
        **kwargs: Any,
    ) -> "SkyModel":
        """Create a SkyModel from frequency-keyed dicts of HEALPix maps.

        This is the standard constructor for loaders that build
        ``dict[float, ndarray]`` during generation (pygdsm, pysm3, etc.).
        It converts the dicts to 2-D arrays sorted by frequency.

        Parameters
        ----------
        i_maps : dict[float, np.ndarray]
            Stokes I maps keyed by frequency in Hz.
        q_maps : dict[float, np.ndarray] or None
            Stokes Q maps, or None.
        u_maps : dict[float, np.ndarray] or None
            Stokes U maps, or None.
        v_maps : dict[float, np.ndarray] or None
            Stokes V maps, or None.
        nside : int
            HEALPix NSIDE parameter.
        **kwargs
            Additional keyword arguments passed to the ``SkyModel`` constructor
            (e.g. ``model_name``, ``brightness_conversion``, ``_precision``,
            ``frequency``, ``_native_format``, ``_pygdsm_instance``).

        Returns
        -------
        SkyModel
        """
        sorted_freqs = np.sort(np.array(list(i_maps.keys()), dtype=np.float64))
        i_arr = np.stack([i_maps[f] for f in sorted_freqs])
        q_arr = np.stack([q_maps[f] for f in sorted_freqs]) if q_maps else None
        u_arr = np.stack([u_maps[f] for f in sorted_freqs]) if u_maps else None
        v_arr = np.stack([v_maps[f] for f in sorted_freqs]) if v_maps else None

        return cls(
            _healpix_maps=i_arr,
            _healpix_q_maps=q_arr,
            _healpix_u_maps=u_arr,
            _healpix_v_maps=v_arr,
            _healpix_nside=nside,
            _observation_frequencies=sorted_freqs,
            **kwargs,
        )

    @classmethod
    def from_test_sources(
        cls,
        num_sources: int = 100,
        flux_range: tuple[float, float] | None = None,
        dec_deg: float | None = None,
        spectral_index: float | None = None,
        distribution: str = "uniform",
        seed: int | None = None,
        dec_range_deg: float | None = None,
        brightness_conversion: BrightnessConversion = "planck",
        precision: "PrecisionConfig | None" = None,
        polarization_fraction: float = 0.0,
        polarization_angle_deg: float = 0.0,
        stokes_v_fraction: float = 0.0,
    ) -> "SkyModel":
        """
        Generate synthetic test sources.

        Parameters
        ----------
        num_sources : int, default=100
            Number of sources to generate.
        flux_range : tuple of float, optional
            (min_flux, max_flux) in Jy. Must be provided.
        dec_deg : float, optional
            Declination for all sources (degrees). Must be provided.
            For 'uniform' distribution, all sources share this declination.
            For 'random', used as the center of a declination band.
        spectral_index : float, optional
            Spectral index for all sources. Must be provided.
        distribution : str, default="uniform"
            Source distribution mode:
            - "uniform": evenly spaced in RA, linearly spaced flux, fixed Dec.
            - "random": random RA, Dec (within band around dec_deg), and flux.
        seed : int, optional
            Random seed for reproducibility. Only used when distribution='random'.
        dec_range_deg : float, optional
            Half-width of declination band in degrees (only used when
            distribution='random'). Sources are drawn uniformly from
            [dec_deg - dec_range_deg, dec_deg + dec_range_deg], clamped to
            [-90, 90]. Defaults to 10.0 when distribution='random'.
        brightness_conversion : BrightnessConversion, default="planck"
            Brightness conversion method.
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes. If None, uses float64.
        polarization_fraction : float, default=0.0
            Linear polarization fraction (0 to 1). When > 0, Stokes Q and U
            are computed from flux * polarization_fraction and the given
            polarization angle.
        polarization_angle_deg : float, default=0.0
            Polarization angle in degrees (EVPA). Used with
            ``polarization_fraction`` to compute Q = p*I*cos(2*chi),
            U = p*I*sin(2*chi).
        stokes_v_fraction : float, default=0.0
            Circular polarization fraction (0 to 1). V = v_frac * I.

        Returns
        -------
        SkyModel
            Sky model with test sources.

        Raises
        ------
        ValueError
            If dec_deg, flux_range, or spectral_index is not provided, or if
            distribution is not 'uniform' or 'random'.
        """
        errors = []
        if dec_deg is None:
            errors.append("dec_deg is required for test sources")
        if flux_range is None:
            errors.append(
                "flux_range (flux_min, flux_max) is required for test sources"
            )
        if spectral_index is None:
            errors.append("spectral_index is required for test sources")
        if distribution not in ("uniform", "random"):
            errors.append(
                f"distribution must be 'uniform' or 'random', got '{distribution}'"
            )
        if errors:
            raise ValueError(
                "Missing required test source parameters:\n  - " + "\n  - ".join(errors)
            )

        n = num_sources

        if distribution == "random":
            rng = np.random.default_rng(seed)
            # RA: uniform over full sky
            ra_deg_arr = rng.uniform(0.0, 360.0, size=n)

            # Dec: uniform within band around dec_deg
            half_width = dec_range_deg if dec_range_deg is not None else 10.0
            dec_lo = max(-90.0, dec_deg - half_width)
            dec_hi = min(90.0, dec_deg + half_width)
            dec_deg_arr = rng.uniform(dec_lo, dec_hi, size=n)

            # Flux: uniform between min and max
            flux_arr = rng.uniform(flux_range[0], flux_range[1], size=n)

            logger.debug(
                f"Generated {n} random test sources "
                f"(seed={seed}, dec=[{dec_lo:.1f}, {dec_hi:.1f}]deg)"
            )
        else:
            # Deterministic uniform distribution (original behavior)
            if n == 1:
                ra_deg_arr = np.array([0.0])
                flux_arr = np.array([(flux_range[0] + flux_range[1]) / 2])
            else:
                ra_deg_arr = np.array([(360.0 / n) * i for i in range(n)])
                flux_arr = np.linspace(flux_range[0], flux_range[1], n)

            dec_deg_arr = np.full(n, dec_deg)

            logger.debug(f"Generated {n} uniform test sources")

        # Compute polarization
        if polarization_fraction > 0:
            chi_rad = np.deg2rad(polarization_angle_deg)
            stokes_q_arr = flux_arr * polarization_fraction * np.cos(2.0 * chi_rad)
            stokes_u_arr = flux_arr * polarization_fraction * np.sin(2.0 * chi_rad)
        else:
            stokes_q_arr = np.zeros(n, dtype=np.float64)
            stokes_u_arr = np.zeros(n, dtype=np.float64)

        if stokes_v_fraction > 0:
            stokes_v_arr = flux_arr * stokes_v_fraction
        else:
            stokes_v_arr = np.zeros(n, dtype=np.float64)

        return cls(
            _ra_rad=cls._deg_to_rad_at_precision(ra_deg_arr, precision),
            _dec_rad=cls._deg_to_rad_at_precision(dec_deg_arr, precision),
            _flux_ref=flux_arr.astype(np.float64),
            _alpha=np.full(n, float(spectral_index)),
            _stokes_q=stokes_q_arr,
            _stokes_u=stokes_u_arr,
            _stokes_v=stokes_v_arr,
            _native_format=NATIVE_POINT_SOURCES,
            model_name="test_sources",
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )

    @classmethod
    def from_point_sources(
        cls,
        sources: list[dict[str, Any]],
        model_name: str = "custom",
        brightness_conversion: BrightnessConversion = "planck",
        precision: "PrecisionConfig | None" = None,
    ) -> "SkyModel":
        """
        Create SkyModel from existing point source list.

        Delegates to ``from_arrays()`` after extracting arrays from the
        list of dicts.

        Parameters
        ----------
        sources : list of dict
            Point source list. Each dict must have 'coords' (SkyCoord) and 'flux' (float).
            Optional: 'spectral_index', 'stokes_q', 'stokes_u', 'stokes_v'.
        model_name : str, default="custom"
            Name for the model.
        brightness_conversion : BrightnessConversion, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.

        Returns
        -------
        SkyModel
        """
        n = len(sources)
        if n == 0:
            return cls._empty_sky(
                model_name=model_name,
                brightness_conversion=brightness_conversion,
                precision=precision,
            )

        ra_rad = np.array([s["coords"].ra.rad for s in sources], dtype=np.float64)
        dec_rad = np.array([s["coords"].dec.rad for s in sources], dtype=np.float64)
        flux_ref = np.array([s["flux"] for s in sources], dtype=np.float64)
        alpha = np.array(
            [s.get("spectral_index", -0.7) for s in sources], dtype=np.float64
        )
        stokes_q = np.array([s.get("stokes_q", 0.0) for s in sources], dtype=np.float64)
        stokes_u = np.array([s.get("stokes_u", 0.0) for s in sources], dtype=np.float64)
        stokes_v = np.array([s.get("stokes_v", 0.0) for s in sources], dtype=np.float64)

        # Optional extended fields
        rm = None
        if any("rotation_measure" in s for s in sources):
            rm = np.array(
                [s.get("rotation_measure", 0.0) for s in sources], dtype=np.float64
            )
        major = None
        minor = None
        pa = None
        if any("major_arcsec" in s for s in sources):
            major = np.array(
                [s.get("major_arcsec", 0.0) for s in sources], dtype=np.float64
            )
            minor = np.array(
                [s.get("minor_arcsec", 0.0) for s in sources], dtype=np.float64
            )
            pa = np.array([s.get("pa_deg", 0.0) for s in sources], dtype=np.float64)
        sp_coeffs = None
        if any("spectral_coeffs" in s for s in sources):
            max_terms = max(len(s.get("spectral_coeffs", [])) for s in sources)
            if max_terms > 0:
                sp_coeffs = np.zeros((n, max_terms), dtype=np.float64)
                for i, s in enumerate(sources):
                    coeffs = s.get("spectral_coeffs", [s.get("spectral_index", -0.7)])
                    sp_coeffs[i, : len(coeffs)] = coeffs

        return cls.from_arrays(
            ra_rad=ra_rad,
            dec_rad=dec_rad,
            flux_ref=flux_ref,
            alpha=alpha,
            stokes_q=stokes_q,
            stokes_u=stokes_u,
            stokes_v=stokes_v,
            rotation_measure=rm,
            major_arcsec=major,
            minor_arcsec=minor,
            pa_deg=pa,
            spectral_coeffs=sp_coeffs,
            model_name=model_name,
            brightness_conversion=brightness_conversion,
            precision=precision,
        )

    # =========================================================================
    # Listing (delegates to loader modules)
    # =========================================================================

    @staticmethod
    def list_all_models() -> dict[str, dict[str, str]]:
        """List all available sky models and catalogs with their descriptions.

        Returns
        -------
        dict[str, dict[str, str]]
            Nested mapping: category -> {name: description}.
            Categories: "diffuse", "point_catalogs", "racs".
        """
        from ._loaders_diffuse import list_diffuse_models
        from ._loaders_vizier import list_point_catalogs, list_racs_catalogs

        return {
            "diffuse": list_diffuse_models(),
            "point_catalogs": list_point_catalogs(),
            "racs": list_racs_catalogs(),
        }

    @staticmethod
    def get_catalog_info(catalog_key: str, live: bool = False) -> dict[str, Any]:
        """Get metadata for any supported catalog or model.

        Parameters
        ----------
        catalog_key : str
            Catalog or model identifier (e.g. ``"gleam_egc"``, ``"racs_low"``,
            ``"gsm2008"``).
        live : bool, default=False
            If True, query VizieR/CASDA TAP for live column information.
        """
        from ._loaders_diffuse import get_diffuse_model_info
        from ._loaders_vizier import (
            get_catalog_columns,
            get_point_catalog_metadata,
            get_racs_columns,
            get_racs_metadata,
        )
        from .catalogs import DIFFUSE_MODELS, RACS_CATALOGS, VIZIER_POINT_CATALOGS

        if catalog_key in VIZIER_POINT_CATALOGS:
            return (
                get_catalog_columns(catalog_key)
                if live
                else get_point_catalog_metadata(catalog_key)
            )

        if catalog_key.startswith("racs_"):
            band = catalog_key[5:]
            if band in RACS_CATALOGS:
                return get_racs_columns(band) if live else get_racs_metadata(band)

        if catalog_key in RACS_CATALOGS:
            return (
                get_racs_columns(catalog_key)
                if live
                else get_racs_metadata(catalog_key)
            )

        if catalog_key in DIFFUSE_MODELS:
            return get_diffuse_model_info(catalog_key)

        all_keys = (
            sorted(VIZIER_POINT_CATALOGS.keys())
            + [f"racs_{b}" for b in sorted(RACS_CATALOGS.keys())]
            + sorted(DIFFUSE_MODELS.keys())
        )
        raise ValueError(f"Unknown catalog key '{catalog_key}'. Available: {all_keys}")

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
            Summary string including native format, model name, mode,
            and either source count or HEALPix parameters.
        """
        if self._healpix_maps is not None:
            freqs = self._observation_frequencies
            freq_range = (
                f"{freqs[0] / 1e6:.1f}-{freqs[-1] / 1e6:.1f}"
                if len(freqs) > 1
                else f"{freqs[0] / 1e6:.1f}"
            )
            stokes_components = "I"
            n_stokes = 1
            if self._healpix_q_maps is not None:
                stokes_components += "Q"
                n_stokes += 1
            if self._healpix_u_maps is not None:
                stokes_components += "U"
                n_stokes += 1
            if self._healpix_v_maps is not None:
                stokes_components += "V"
                n_stokes += 1
            mem_info = self.estimate_healpix_memory(
                self._healpix_nside,
                len(freqs),
                np.float32,
                n_stokes=n_stokes,
            )
            return (
                f"SkyModel(native='{self._native_format}', model='{self.model_name}', "
                f"mode='healpix_multifreq', nside={self._healpix_nside}, "
                f"n_freq={len(freqs)}, freq_range={freq_range}MHz, "
                f"stokes='{stokes_components}', "
                f"memory={mem_info['total_mb']:.1f}MB)"
            )
        extras = []
        if self._rotation_measure is not None and np.any(self._rotation_measure != 0):
            extras.append("RM")
        if self._major_arcsec is not None and np.any(self._major_arcsec > 0):
            n_gauss = int(np.sum(self._major_arcsec > 0))
            extras.append(f"gaussian={n_gauss}")
        if self._spectral_coeffs is not None and self._spectral_coeffs.shape[1] > 1:
            extras.append(f"spectral_terms={self._spectral_coeffs.shape[1]}")
        extra_str = f", {', '.join(extras)}" if extras else ""
        return (
            f"SkyModel(native='{self._native_format}', model='{self.model_name}', "
            f"n_sources={self.n_sources}{extra_str})"
        )
