# rrivis/core/sky/model.py
"""
Unified SkyModel dataclass.

Holds sky data in either point-source (columnar arrays) or multi-frequency
HEALPix format, with bidirectional conversion, combination, and precision
management.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord

from ._loaders_diffuse import _DiffuseLoadersMixin
from ._loaders_pyradiosky import _PyradioskyMixin
from ._loaders_vizier import _VizierLoadersMixin
from .constants import (
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SkyModel Class
# =============================================================================


@dataclass
class SkyModel(_VizierLoadersMixin, _DiffuseLoadersMixin, _PyradioskyMixin):
    """
    Unified sky model with bidirectional conversion.

    Can hold sky data in either point source or HEALPix format, and convert
    between them on demand. This allows uniform treatment of all sky models
    (catalogs, diffuse emission, test sources) regardless of their native format.

    Columnar point-source arrays are ``(N_sources,)`` float64. ``None`` means
    not populated; ``len == 0`` means loaded but no sources passed the filter.

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
    _healpix_maps : dict[float, np.ndarray], optional
        Multi-frequency HEALPix brightness temperature maps (freq_hz -> T_b).
        One map per observation channel; generated natively by pygdsm for
        diffuse models, or via spectral extrapolation for point source catalogs.
    _healpix_nside : int, optional
        HEALPix NSIDE parameter.
    _native_format : str
        Original format: ``"point_sources"`` or ``"healpix"``.
    frequency : float, optional
        Reference frequency in Hz (first channel for multi-freq maps).
    model_name : str, optional
        Name of the sky model.
    brightness_conversion : str
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

    _healpix_maps: dict[float, np.ndarray] | None = None
    _healpix_q_maps: dict[float, np.ndarray] | None = None
    _healpix_u_maps: dict[float, np.ndarray] | None = None
    _healpix_v_maps: dict[float, np.ndarray] | None = None
    _observation_frequencies: np.ndarray | None = None
    _healpix_nside: int | None = None

    _native_format: str = "point_sources"

    frequency: float | None = None
    model_name: str | None = None
    brightness_conversion: str = "planck"

    _point_sources_frequency: float | None = field(default=None, repr=False)

    _precision: Any = field(default=None, repr=False)

    _pygdsm_instance: Any = field(default=None, repr=False)

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
    def _deg_to_rad_at_precision(arr: np.ndarray, precision: Any) -> np.ndarray:
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
    def _rad_to_deg_at_precision(arr: np.ndarray, precision: Any) -> np.ndarray:
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

    def _ensure_dtypes(self) -> None:
        """Cast internal arrays to precision-appropriate dtypes.

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
            self._ra_rad = self._ra_rad.astype(src_dt, copy=False)
            self._dec_rad = self._dec_rad.astype(src_dt, copy=False)
        if self._flux_ref is not None:
            self._flux_ref = self._flux_ref.astype(flux_dt, copy=False)
        if self._alpha is not None:
            self._alpha = self._alpha.astype(alpha_dt, copy=False)
        if self._stokes_q is not None:
            self._stokes_q = self._stokes_q.astype(flux_dt, copy=False)
            self._stokes_u = self._stokes_u.astype(flux_dt, copy=False)
            self._stokes_v = self._stokes_v.astype(flux_dt, copy=False)
        if self._healpix_maps is not None:
            self._healpix_maps = {
                f: m.astype(hp_dt, copy=False) for f, m in self._healpix_maps.items()
            }
        if self._healpix_q_maps is not None:
            self._healpix_q_maps = {
                f: m.astype(hp_dt, copy=False) for f, m in self._healpix_q_maps.items()
            }
        if self._healpix_u_maps is not None:
            self._healpix_u_maps = {
                f: m.astype(hp_dt, copy=False) for f, m in self._healpix_u_maps.items()
            }
        if self._healpix_v_maps is not None:
            self._healpix_v_maps = {
                f: m.astype(hp_dt, copy=False) for f, m in self._healpix_v_maps.items()
            }

    @classmethod
    def _empty_sky(
        cls,
        model_name: str,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """Return an empty point-source SkyModel (zero-length arrays)."""
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
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def mode(self) -> str:
        """Return the current representation mode."""
        if self._healpix_maps is not None:
            return "healpix_multifreq"
        return "point_sources"

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
            first_map = next(iter(self._healpix_maps.values()))
            return len(first_map)
        return len(self._ra_rad) if self._ra_rad is not None else 0

    @property
    def has_polarized_healpix_maps(self) -> bool:
        """Return True if any polarization (Q/U/V) HEALPix maps are populated."""
        return any(
            m is not None
            for m in (self._healpix_q_maps, self._healpix_u_maps, self._healpix_v_maps)
        )

    def _has_point_sources(self) -> bool:
        """Return True if columnar point-source arrays are populated and non-empty."""
        return self._ra_rad is not None and len(self._ra_rad) > 0

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @staticmethod
    def _parse_frequency_config(obs_frequency_config: dict[str, Any]) -> np.ndarray:
        """
        Parse observation frequency configuration to array of frequencies in Hz.

        Parameters
        ----------
        obs_frequency_config : dict
            Observation frequency configuration with keys:
            - starting_frequency: float
            - frequency_interval: float (channel width)
            - frequency_bandwidth: float (total bandwidth)
            - frequency_unit: str ("Hz", "kHz", "MHz", "GHz")

        Returns
        -------
        frequencies : np.ndarray
            Array of frequency channel centers in Hz.

        Examples
        --------
        >>> config = {
        ...     "starting_frequency": 100.0,
        ...     "frequency_interval": 1.0,
        ...     "frequency_bandwidth": 20.0,
        ...     "frequency_unit": "MHz",
        ... }
        >>> freqs = SkyModel._parse_frequency_config(config)
        >>> len(freqs)  # 20 channels
        20
        >>> freqs[0] / 1e6  # First channel at 100 MHz
        100.0
        """
        start_freq = obs_frequency_config["starting_frequency"]
        interval = obs_frequency_config["frequency_interval"]
        bandwidth = obs_frequency_config["frequency_bandwidth"]
        unit = obs_frequency_config.get("frequency_unit", "MHz")

        unit_factors = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
        if unit not in unit_factors:
            raise ValueError(
                f"Unknown frequency unit '{unit}'. "
                f"Supported: {list(unit_factors.keys())}"
            )

        unit_factor = unit_factors[unit]
        start_hz = start_freq * unit_factor
        interval_hz = interval * unit_factor
        bandwidth_hz = bandwidth * unit_factor

        n_channels = int(bandwidth_hz / interval_hz)
        if n_channels <= 0:
            raise ValueError(
                f"Invalid frequency config: bandwidth ({bandwidth} {unit}) must be "
                f"greater than interval ({interval} {unit})"
            )

        frequencies = np.array(
            [start_hz + i * interval_hz for i in range(n_channels)], dtype=np.float64
        )

        logger.debug(
            f"Parsed frequency config: {n_channels} channels from "
            f"{start_freq} to {start_freq + bandwidth - interval} {unit}"
        )

        return frequencies

    @staticmethod
    def estimate_healpix_memory(
        nside: int,
        n_frequencies: int,
        dtype: type = np.float32,
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
        dtype : type, default=np.float32
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
            "dtype": dtype.__name__,
        }

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def to_point_sources(
        self, flux_limit: float = 0.0, frequency: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Get sky model as point sources.

        If native format is point sources, returns them directly.
        If in healpix_multifreq mode, converts pixels from the requested
        frequency channel (or first available) to point sources.

        Parameters
        ----------
        flux_limit : float, default=0.0
            Minimum flux in Jy to include.
        frequency : float, optional
            Frequency for T_b to Jy conversion (used to select the channel
            when in healpix_multifreq mode).

        Returns
        -------
        list of dict
            Point sources with coords, flux, spectral_index, stokes.
        """
        # If columnar arrays already populated, build list-of-dicts from them
        if self._ra_rad is not None:
            if flux_limit > 0:
                mask = self._flux_ref >= flux_limit
            else:
                mask = np.ones(len(self._ra_rad), dtype=bool)
            n = int(mask.sum())
            if n == 0:
                return []
            ra_deg = np.rad2deg(self._ra_rad[mask])
            dec_deg = np.rad2deg(self._dec_rad[mask])
            # Bulk SkyCoord construction -- much faster than per-row
            coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
            flux_m = self._flux_ref[mask]
            alpha_m = self._alpha[mask]
            sq_m = self._stokes_q[mask]
            su_m = self._stokes_u[mask]
            sv_m = self._stokes_v[mask]
            return [
                {
                    "coords": coords[i],
                    "flux": float(flux_m[i]),
                    "spectral_index": float(alpha_m[i]),
                    "stokes_q": float(sq_m[i]),
                    "stokes_u": float(su_m[i]),
                    "stokes_v": float(sv_m[i]),
                }
                for i in range(n)
            ]

        # Convert from multi-frequency HEALPix (lazy: populate arrays on first call)
        if self._healpix_maps is not None:
            freq = frequency or self.frequency or next(iter(self._healpix_maps))
            temp_map = self.get_map_at_frequency(freq)
            self._healpix_map_to_arrays(temp_map, freq)
            # Now _ra_rad is populated; recurse to build list-of-dicts
            return self.to_point_sources(flux_limit=flux_limit, frequency=frequency)

        return []

    def _healpix_map_to_arrays(
        self,
        temp_map: np.ndarray,
        frequency: float,
    ) -> None:
        """
        Convert a HEALPix brightness temperature map to columnar arrays.

        Populates _ra_rad, _dec_rad, _flux_ref, _alpha, _stokes_q, _stokes_u,
        _stokes_v in-place. Only positive-temperature pixels are stored
        (no flux_limit filtering here -- apply that in to_point_sources()).

        Parameters
        ----------
        temp_map : np.ndarray
            Brightness temperature map in Kelvin.
        frequency : float
            Frequency in Hz for T_b -> Jy conversion.
        """
        npix = len(temp_map)
        nside = hp.npix2nside(npix)
        omega = 4 * np.pi / npix

        flux_jy = np.zeros(npix, dtype=np.float64)
        pos = temp_map > 0
        if np.any(pos):
            flux_jy[pos] = brightness_temp_to_flux_density(
                temp_map[pos].astype(np.float64),
                frequency,
                omega,
                method=self.brightness_conversion,
            )

        valid_idx = np.where(flux_jy > 0)[0]
        if len(valid_idx) == 0:
            logger.warning("No pixels with positive flux in HEALPix map")
            self._ra_rad = np.zeros(0, dtype=np.float64)
            self._dec_rad = np.zeros(0, dtype=np.float64)
            self._flux_ref = np.zeros(0, dtype=np.float64)
            self._alpha = np.zeros(0, dtype=np.float64)
            self._stokes_q = np.zeros(0, dtype=np.float64)
            self._stokes_u = np.zeros(0, dtype=np.float64)
            self._stokes_v = np.zeros(0, dtype=np.float64)
            return

        theta, phi = hp.pix2ang(nside, valid_idx)
        self._ra_rad = phi  # phi = RA in radians
        self._dec_rad = np.pi / 2 - theta  # colatitude -> declination
        self._flux_ref = flux_jy[valid_idx]
        n = len(valid_idx)
        self._alpha = np.zeros(n, dtype=np.float64)  # no per-pixel alpha in HEALPix

        # Rayleigh-Jeans factor for K_RJ -> Jy (linear, sign-preserving)
        from .constants import C_LIGHT, K_BOLTZMANN

        rj_factor = (2 * K_BOLTZMANN * frequency**2 / C_LIGHT**2) * omega / 1e-26

        # Extract Q/U/V from polarization maps if available
        if self._healpix_q_maps is not None and frequency in self._healpix_q_maps:
            q_map = self._healpix_q_maps[frequency]
            self._stokes_q = q_map[valid_idx].astype(np.float64) * rj_factor
        else:
            self._stokes_q = np.zeros(n, dtype=np.float64)

        if self._healpix_u_maps is not None and frequency in self._healpix_u_maps:
            u_map = self._healpix_u_maps[frequency]
            self._stokes_u = u_map[valid_idx].astype(np.float64) * rj_factor
        else:
            self._stokes_u = np.zeros(n, dtype=np.float64)

        if self._healpix_v_maps is not None and frequency in self._healpix_v_maps:
            v_map = self._healpix_v_maps[frequency]
            self._stokes_v = v_map[valid_idx].astype(np.float64) * rj_factor
        else:
            self._stokes_v = np.zeros(n, dtype=np.float64)

    def to_healpix(
        self, nside: int = 64, frequency: float | None = None
    ) -> tuple[np.ndarray, int, np.ndarray | None]:
        """
        .. deprecated::
            Single-frequency HEALPix maps are no longer supported.
            Use ``to_healpix_for_observation()`` for correct multi-frequency
            conversion that preserves the native pygdsm spectral model.

        Raises
        ------
        ValueError
            Always. Redirect callers to the multi-frequency API.
        """
        raise ValueError(
            "to_healpix() is no longer supported. "
            "Use to_healpix_for_observation(nside, obs_frequency_config) for "
            "correct multi-frequency conversion with proper spectral handling. "
            "For diffuse models, use from_diffuse_sky(frequencies=...) directly."
        )

    def _point_sources_to_healpix_multifreq(
        self, nside: int, frequencies: np.ndarray, ref_frequency: float = 76e6
    ) -> tuple[
        dict[float, np.ndarray],
        dict[float, np.ndarray] | None,
        dict[float, np.ndarray] | None,
        dict[float, np.ndarray] | None,
    ]:
        """
        Convert point sources to multi-frequency HEALPix brightness temperature maps.

        Vectorized implementation: uses np.bincount for O(N_sources) memory per
        frequency channel instead of a Python loop over sources.

        Parameters
        ----------
        nside : int
            HEALPix NSIDE parameter.
        frequencies : np.ndarray
            Array of frequencies in Hz.
        ref_frequency : float, default=76e6
            Reference frequency for flux values (GLEAM standard: 76 MHz).

        Returns
        -------
        i_maps : dict[float, np.ndarray]
            Stokes I brightness temperature maps (K).
        q_maps : dict[float, np.ndarray] or None
            Stokes Q maps (K_RJ), or None if all sources are unpolarized.
        u_maps : dict[float, np.ndarray] or None
            Stokes U maps (K_RJ), or None.
        v_maps : dict[float, np.ndarray] or None
            Stokes V maps (K_RJ), or None.
        """
        if not self._has_point_sources():
            npix = hp.nside2npix(nside)
            empty = {
                float(freq): np.zeros(npix, dtype=np.float32) for freq in frequencies
            }
            return empty, None, None, None

        npix = hp.nside2npix(nside)
        omega_pixel = 4 * np.pi / npix
        n_sources = len(self._ra_rad)
        n_freq = len(frequencies)

        # Check if any source has non-zero polarization
        has_pol = (
            self._stokes_q is not None
            and self._stokes_u is not None
            and self._stokes_v is not None
            and (
                np.any(self._stokes_q != 0)
                or np.any(self._stokes_u != 0)
                or np.any(self._stokes_v != 0)
            )
        )

        n_stokes = 4 if has_pol else 1
        mem_info = self.estimate_healpix_memory(nside, n_freq, np.float32, n_stokes)
        logger.info(
            f"Creating {n_freq} HEALPix maps (nside={nside}, "
            f"stokes={'IQUV' if has_pol else 'I'}): "
            f"~{mem_info['total_mb']:.1f} MB"
        )

        ipix = hp.ang2pix(nside, np.pi / 2 - self._dec_rad, self._ra_rad)

        # Rayleigh-Jeans factor: Jy -> K_RJ (inverse of T->S conversion)
        from .constants import C_LIGHT, K_BOLTZMANN

        i_maps: dict[float, np.ndarray] = {}
        q_maps: dict[float, np.ndarray] = {} if has_pol else None
        u_maps: dict[float, np.ndarray] = {} if has_pol else None
        v_maps: dict[float, np.ndarray] = {} if has_pol else None

        for freq in frequencies:
            scale = (float(freq) / ref_frequency) ** self._alpha
            flux_f = self._flux_ref * scale

            flux_map = np.bincount(ipix, weights=flux_f, minlength=npix)

            temp_out = np.zeros(npix, dtype=np.float32)
            occupied = flux_map > 0
            if np.any(occupied):
                temp_out[occupied] = flux_density_to_brightness_temp(
                    flux_map[occupied],
                    float(freq),
                    omega_pixel,
                    method=self.brightness_conversion,
                ).astype(np.float32)
            i_maps[float(freq)] = temp_out

            if has_pol:
                # Jy -> K_RJ via Rayleigh-Jeans (linear, sign-preserving)
                rj_inv = (
                    C_LIGHT**2 / (2 * K_BOLTZMANN * float(freq) ** 2 * omega_pixel)
                ) * 1e-26

                q_flux = self._stokes_q * scale
                q_map = np.bincount(ipix, weights=q_flux, minlength=npix)
                q_maps[float(freq)] = (q_map * rj_inv).astype(np.float32)

                u_flux = self._stokes_u * scale
                u_map = np.bincount(ipix, weights=u_flux, minlength=npix)
                u_maps[float(freq)] = (u_map * rj_inv).astype(np.float32)

                v_flux = self._stokes_v * scale
                v_map = np.bincount(ipix, weights=v_flux, minlength=npix)
                v_maps[float(freq)] = (v_map * rj_inv).astype(np.float32)

        logger.info(
            f"Converted {n_sources} point sources to {n_freq} HEALPix maps "
            f"({frequencies[0] / 1e6:.1f}-{frequencies[-1] / 1e6:.1f} MHz)"
        )

        return i_maps, q_maps, u_maps, v_maps

    def to_healpix_for_observation(
        self,
        nside: int,
        obs_frequency_config: dict[str, Any],
        ref_frequency: float = 76e6,
    ) -> "SkyModel":
        """
        Convert point sources to multi-frequency HEALPix maps for observation.

        Creates one HEALPix map per frequency channel defined in the observation
        configuration. This is the recommended method for preparing sky models
        for visibility simulation, as it correctly handles spectral extrapolation
        for each source individually.

        Parameters
        ----------
        nside : int
            HEALPix NSIDE parameter.
        obs_frequency_config : dict
            Observation frequency configuration with keys:
            - starting_frequency: float
            - frequency_interval: float (channel width)
            - frequency_bandwidth: float (total bandwidth)
            - frequency_unit: str ("Hz", "kHz", "MHz", "GHz")
        ref_frequency : float, default=76e6
            Reference frequency for flux values in Hz (GLEAM: 76 MHz).

        Returns
        -------
        SkyModel
            Self, with _healpix_maps populated.

        Raises
        ------
        ValueError
            If no point sources are available for conversion.

        Notes
        -----
        Memory usage depends on nside and number of frequency channels:
        - nside=64, 20 channels: ~4 MB
        - nside=256, 100 channels: ~600 MB
        - nside=1024, 20 channels: ~1 GB

        Use `estimate_healpix_memory()` to check before conversion.

        Examples
        --------
        >>> sky = SkyModel.from_gleam(flux_limit=1.0)
        >>> config = {
        ...     "starting_frequency": 100.0,
        ...     "frequency_interval": 1.0,
        ...     "frequency_bandwidth": 20.0,
        ...     "frequency_unit": "MHz",
        ... }
        >>> sky.to_healpix_for_observation(nside=64, obs_frequency_config=config)
        >>> sky.get_map_at_frequency(100e6)  # Get map at 100 MHz
        """
        if not self._has_point_sources():
            raise ValueError(
                "No point sources available for conversion. "
                "Load sources first using from_gleam(), from_mals(), etc."
            )

        frequencies = self._parse_frequency_config(obs_frequency_config)

        i_maps, q_maps, u_maps, v_maps = self._point_sources_to_healpix_multifreq(
            nside, frequencies, ref_frequency
        )
        self._healpix_maps = i_maps
        self._healpix_q_maps = q_maps
        self._healpix_u_maps = u_maps
        self._healpix_v_maps = v_maps
        self._healpix_nside = nside
        self._observation_frequencies = frequencies

        return self

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
        >>> sky.to_healpix_for_observation(nside=64, obs_frequency_config=config)
        >>> map_100mhz = sky.get_map_at_frequency(100e6)
        """
        if self._healpix_maps is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Use to_healpix_for_observation() first."
            )
        return self._healpix_maps[self._resolve_frequency(frequency)]

    def get_multifreq_maps(self) -> tuple[dict[float, np.ndarray], int, np.ndarray]:
        """
        Get all multi-frequency HEALPix maps.

        Returns
        -------
        temp_maps : Dict[float, np.ndarray]
            Dictionary mapping frequency (Hz) to brightness temperature map (K).
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
        >>> for freq in freqs:
        ...     print(f"{freq / 1e6:.1f} MHz: max T_b = {maps[freq].max():.2f} K")
        """
        if self._healpix_maps is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Use to_healpix_for_observation() first."
            )

        return (self._healpix_maps, self._healpix_nside, self._observation_frequencies)

    def _resolve_frequency(self, frequency: float) -> float:
        """Resolve a frequency to the nearest available key in ``_healpix_maps``.

        Returns the exact key if present, or the nearest one (with a warning
        if the difference exceeds 1 kHz).
        """
        if frequency in self._healpix_maps:
            return frequency
        available_freqs = np.array(list(self._healpix_maps.keys()))
        idx = np.argmin(np.abs(available_freqs - frequency))
        nearest_freq = float(available_freqs[idx])
        freq_diff_mhz = abs(frequency - nearest_freq) / 1e6
        if freq_diff_mhz > 0.001:
            logger.warning(
                f"Exact frequency {frequency / 1e6:.3f} MHz not found. "
                f"Using nearest: {nearest_freq / 1e6:.3f} MHz "
                f"(diff: {freq_diff_mhz:.3f} MHz)"
            )
        return nearest_freq

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
                "Use to_healpix_for_observation() first."
            )
        freq_key = self._resolve_frequency(frequency)
        I_map = self._healpix_maps[freq_key]
        Q_map = (
            self._healpix_q_maps[freq_key] if self._healpix_q_maps is not None else None
        )
        U_map = (
            self._healpix_u_maps[freq_key] if self._healpix_u_maps is not None else None
        )
        V_map = (
            self._healpix_v_maps[freq_key] if self._healpix_v_maps is not None else None
        )
        return I_map, Q_map, U_map, V_map

    def get_multifreq_stokes_maps(
        self,
    ) -> tuple[
        dict[float, np.ndarray],
        dict[float, np.ndarray] | None,
        dict[float, np.ndarray] | None,
        dict[float, np.ndarray] | None,
        int,
        np.ndarray,
    ]:
        """Get all multi-frequency Stokes I/Q/U/V HEALPix maps.

        Returns
        -------
        I_maps : dict[float, np.ndarray]
            Stokes I maps keyed by frequency (Hz).
        Q_maps : dict[float, np.ndarray] or None
            Stokes Q maps, or None if not available.
        U_maps : dict[float, np.ndarray] or None
            Stokes U maps, or None if not available.
        V_maps : dict[float, np.ndarray] or None
            Stokes V maps, or None if not available.
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
                "Use to_healpix_for_observation() first."
            )
        return (
            self._healpix_maps,
            self._healpix_q_maps,
            self._healpix_u_maps,
            self._healpix_v_maps,
            self._healpix_nside,
            self._observation_frequencies,
        )

    def get_for_visibility(
        self,
        representation: str,
        nside: int = 64,
        flux_limit: float = 0.0,
        frequency: float | None = None,
    ) -> "SkyModel":
        """
        Ensure sky model is in the requested representation.

        This is the main method to call before visibility calculation.
        It converts the sky model to the requested format if needed.

        Parameters
        ----------
        representation : str
            "point_sources" or "healpix_multifreq"
        nside : int, default=64
            HEALPix NSIDE for healpix_multifreq mode.
        flux_limit : float, default=0.0
            Minimum flux for point_sources mode.
        frequency : float, optional
            Frequency for conversion.

        Returns
        -------
        SkyModel
            Self with the requested representation populated.

        Raises
        ------
        ValueError
            If healpix_multifreq is requested but no observation frequency
            config has been set. Use to_healpix_for_observation() first.
        """
        freq = frequency or self.frequency

        if representation == "point_sources":
            # For HEALPix-native models the cached arrays have flux values
            # baked in at a specific frequency.  If a different frequency is
            # requested, invalidate the cache so conversion reruns.
            if (
                self._native_format == "healpix"
                and self._ra_rad is not None
                and freq is not None
                and self._point_sources_frequency != freq
            ):
                self._ra_rad = None
                self._dec_rad = None
                self._flux_ref = None
                self._alpha = None
                self._stokes_q = None
                self._stokes_u = None
                self._stokes_v = None
                self._point_sources_frequency = None

            if self._ra_rad is None:
                # to_point_sources() populates _ra_rad etc. as a side effect
                self.to_point_sources(flux_limit=flux_limit, frequency=freq)
                if self._native_format == "healpix":
                    self._point_sources_frequency = freq
        elif representation in ("healpix_map", "healpix_multifreq"):
            if self._healpix_maps is None:
                raise ValueError(
                    "Cannot convert point sources to HEALPix on-the-fly. "
                    "Use to_healpix_for_observation(nside, obs_frequency_config) first "
                    "to create multi-frequency HEALPix maps with correct spectral handling."
                )

        return self

    # =========================================================================
    # Factory Methods (no external deps)
    # =========================================================================

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
        brightness_conversion: str = "planck",
        precision: Any = None,
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
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes. If None, uses float64.

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
                f"(seed={seed}, dec=[{dec_lo:.1f}, {dec_hi:.1f}]°)"
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

        sky = cls(
            _ra_rad=cls._deg_to_rad_at_precision(ra_deg_arr, precision),
            _dec_rad=cls._deg_to_rad_at_precision(dec_deg_arr, precision),
            _flux_ref=flux_arr.astype(np.float64),
            _alpha=np.full(n, float(spectral_index)),
            _stokes_q=np.zeros(n, dtype=np.float64),
            _stokes_u=np.zeros(n, dtype=np.float64),
            _stokes_v=np.zeros(n, dtype=np.float64),
            _native_format="point_sources",
            model_name="test_sources",
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )
        sky._ensure_dtypes()
        return sky

    @classmethod
    def from_point_sources(
        cls,
        sources: list[dict[str, Any]],
        model_name: str = "custom",
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Create SkyModel from existing point source list.

        Parameters
        ----------
        sources : list of dict
            Point source list. Each dict must have 'coords' (SkyCoord) and 'flux' (float).
            Optional: 'spectral_index', 'stokes_q', 'stokes_u', 'stokes_v'.
        model_name : str, default="custom"
            Name for the model.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.

        Returns
        -------
        SkyModel
        """
        n = len(sources)
        if n == 0:
            return cls(
                _ra_rad=np.zeros(0, dtype=np.float64),
                _dec_rad=np.zeros(0, dtype=np.float64),
                _flux_ref=np.zeros(0, dtype=np.float64),
                _alpha=np.zeros(0, dtype=np.float64),
                _stokes_q=np.zeros(0, dtype=np.float64),
                _stokes_u=np.zeros(0, dtype=np.float64),
                _stokes_v=np.zeros(0, dtype=np.float64),
                _native_format="point_sources",
                model_name=model_name,
                brightness_conversion=brightness_conversion,
                _precision=precision,
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

        sky = cls(
            _ra_rad=ra_rad,
            _dec_rad=dec_rad,
            _flux_ref=flux_ref,
            _alpha=alpha,
            _stokes_q=stokes_q,
            _stokes_u=stokes_u,
            _stokes_v=stokes_v,
            _native_format="point_sources",
            model_name=model_name,
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )
        sky._ensure_dtypes()
        return sky

    # =========================================================================
    # Listing (delegates to mixins)
    # =========================================================================

    @classmethod
    def list_all_models(cls) -> dict[str, dict[str, str]]:
        """List all available sky models and catalogs with their descriptions.

        Returns
        -------
        dict[str, dict[str, str]]
            Nested mapping: category -> {name: description}.
            Categories: "diffuse", "point_catalogs", "racs".

        Examples
        --------
        >>> all_models = SkyModel.list_all_models()
        >>> for category, models in all_models.items():
        ...     print(f"\\n=== {category} ===")
        ...     for name, desc in models.items():
        ...         print(f"  {name}: {desc[:80]}...")
        """
        return {
            "diffuse": cls.list_diffuse_models(),
            "point_catalogs": cls.list_point_catalogs(),
            "racs": cls.list_racs_catalogs(),
        }

    @classmethod
    def get_catalog_info(cls, catalog_key: str, live: bool = False) -> dict[str, Any]:
        """Get metadata for any supported catalog or model.

        Returns locally-stored metadata by default (no network). Pass
        ``live=True`` to also fetch live column information from VizieR
        or CASDA TAP.

        Parameters
        ----------
        catalog_key : str
            Catalog or model identifier. Accepted values:

            - VizieR catalog key (e.g. ``"gleam_egc"``, ``"nvss"``, ``"tgss"``)
            - RACS band prefixed with ``"racs_"`` (e.g. ``"racs_low"``)
            - Diffuse model name (e.g. ``"gsm2008"``, ``"lfsm"``)
        live : bool, default=False
            If True, query VizieR/CASDA TAP for live column information
            (requires network). If False, return only static metadata.

        Returns
        -------
        dict
            Catalog-specific information. For VizieR catalogs see
            ``get_point_catalog_metadata()`` (offline) or
            ``get_catalog_columns()`` (live). For RACS see
            ``get_racs_metadata()`` / ``get_racs_columns()``. For diffuse
            models see ``get_diffuse_model_info()``.

        Raises
        ------
        ValueError
            If ``catalog_key`` is not recognized.

        Examples
        --------
        >>> info = SkyModel.get_catalog_info("nvss")
        >>> print(info["freq_mhz"])
        1400.0

        >>> info = SkyModel.get_catalog_info("racs_low")
        >>> print(info["freq_mhz"])
        887.5

        >>> info = SkyModel.get_catalog_info("gsm2008")
        >>> print(info["class_name"])
        'GlobalSkyModel'
        """
        from .catalogs import DIFFUSE_MODELS, RACS_CATALOGS, VIZIER_POINT_CATALOGS

        if catalog_key in VIZIER_POINT_CATALOGS:
            if live:
                return cls.get_catalog_columns(catalog_key)
            return cls.get_point_catalog_metadata(catalog_key)

        if catalog_key.startswith("racs_"):
            band = catalog_key[5:]
            if band in RACS_CATALOGS:
                if live:
                    return cls.get_racs_columns(band)
                return cls.get_racs_metadata(band)

        if catalog_key in RACS_CATALOGS:
            if live:
                return cls.get_racs_columns(catalog_key)
            return cls.get_racs_metadata(catalog_key)

        if catalog_key in DIFFUSE_MODELS:
            return cls.get_diffuse_model_info(catalog_key)

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
        representation: str = "point_sources",
        nside: int = 64,
        frequency: float | None = None,
        obs_frequency_config: dict[str, Any] | None = None,
        ref_frequency: float = 76e6,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Combine multiple sky models into one.

        The combination always works by first concatenating all models as point
        sources, preserving each source's individual properties (flux, spectral
        index, coordinates). If healpix_map representation is requested, the
        concatenated sources are then converted to multi-frequency HEALPix maps.

        Parameters
        ----------
        models : list of SkyModel
            Sky models to combine.
        representation : str, default="point_sources"
            Output representation: "point_sources" or "healpix_map".
        nside : int, default=64
            HEALPix NSIDE for healpix_map output.
        frequency : float, optional
            Frequency for HEALPix-to-point-source conversions.
        obs_frequency_config : dict, optional
            Required for healpix_map representation. Observation frequency
            configuration with keys: starting_frequency, frequency_interval,
            frequency_bandwidth, frequency_unit.
        ref_frequency : float, default=76e6
            Reference frequency for spectral extrapolation (Hz).

        Returns
        -------
        SkyModel
            Combined sky model.

        Raises
        ------
        ValueError
            If healpix_map representation is requested without obs_frequency_config.

        Warns
        -----
        UserWarning
            When combining catalog sources (GLEAM, MALS) with diffuse models (GSM),
            as this can result in double-counting of bright sources.

        Notes
        -----
        **How concatenation works**:

        Each model is converted to point sources, then all sources are collected
        into a single list. Each source retains its individual properties:

        - coords: SkyCoord (RA, Dec position)
        - flux: float (Jy at reference frequency)
        - spectral_index: float (power-law exponent)
        - stokes_q, stokes_u, stokes_v: float (polarization)

        No averaging occurs during concatenation - if two sources happen to be
        at similar positions, they remain as separate sources. This preserves
        the correct spectral behavior when later converting to HEALPix maps.

        **Double-counting warning**: Diffuse sky models (GSM, LFSM, Haslam) include
        integrated emission from all sources, including bright ones that also appear
        in catalogs like GLEAM. Naive combination will double-count these sources.

        Examples
        --------
        >>> # Combine as point sources (default)
        >>> gleam = SkyModel.from_gleam(flux_limit=1.0)
        >>> test = SkyModel.from_test_sources(
        ...     num_sources=10,
        ...     flux_range=(2.0, 8.0),
        ...     dec_deg=-30.72,
        ...     spectral_index=-0.8,
        ... )
        >>> combined = SkyModel.combine([gleam, test])

        >>> # Combine and convert to multi-frequency HEALPix
        >>> obs_config = {
        ...     "starting_frequency": 100.0,
        ...     "frequency_interval": 1.0,
        ...     "frequency_bandwidth": 20.0,
        ...     "frequency_unit": "MHz",
        ... }
        >>> combined = SkyModel.combine(
        ...     [gleam, test],
        ...     representation="healpix_map",
        ...     nside=64,
        ...     obs_frequency_config=obs_config,
        ... )
        """
        if not models:
            return cls(
                _ra_rad=np.zeros(0, dtype=np.float64),
                _dec_rad=np.zeros(0, dtype=np.float64),
                _flux_ref=np.zeros(0, dtype=np.float64),
                _alpha=np.zeros(0, dtype=np.float64),
                _stokes_q=np.zeros(0, dtype=np.float64),
                _stokes_u=np.zeros(0, dtype=np.float64),
                _stokes_v=np.zeros(0, dtype=np.float64),
                model_name="combined_empty",
                brightness_conversion=brightness_conversion,
                _precision=precision,
            )

        has_catalog = any(
            m.mode == "point_sources" and m._has_point_sources() for m in models
        )
        has_diffuse = any(m.mode == "healpix_multifreq" for m in models)

        if has_catalog and has_diffuse:
            warnings.warn(
                "Combining catalog sources (GLEAM/MALS) with diffuse models (GSM/LFSM/Haslam) "
                "may result in double-counting of bright sources. Diffuse models already include "
                "integrated emission from bright sources. Consider using only one model type "
                "or implementing source subtraction for accurate results.",
                UserWarning,
                stacklevel=2,
            )

        freq = frequency
        if freq is None:
            for m in models:
                if m.frequency is not None:
                    freq = m.frequency
                    break

        has_healpix_multifreq = any(m.mode == "healpix_multifreq" for m in models)

        # ==================================================================
        # HEALPIX_MAP REPRESENTATION WITH MULTIFREQ MODELS
        # Combine maps element-wise in Jy space (not T_b space), per
        # frequency. T_b addition is nonlinear under Planck:
        # T_b(I1+I2) != T_b(I1) + T_b(I2)
        # ==================================================================
        if representation == "healpix_map" and has_healpix_multifreq:
            ref_model = next(m for m in models if m.mode == "healpix_multifreq")
            _, ref_nside, ref_freqs = ref_model.get_multifreq_maps()

            # Validate all healpix_multifreq models share the same nside and
            # frequency grid before doing element-wise arithmetic.  Mismatched
            # nside causes shape errors; mismatched frequency grids cause silent
            # scientific corruption via nearest-frequency fallback.
            for m in models:
                if m.mode != "healpix_multifreq":
                    continue
                _, m_nside, m_freqs = m.get_multifreq_maps()
                if m_nside != ref_nside:
                    raise ValueError(
                        f"Cannot combine HEALPix models with different nside values: "
                        f"reference has nside={ref_nside}, model '{m.model_name}' has "
                        f"nside={m_nside}. Resample to a common nside with hp.ud_grade() "
                        f"before combining."
                    )
                if not np.array_equal(m_freqs, ref_freqs):
                    raise ValueError(
                        f"Cannot combine HEALPix models with different frequency grids: "
                        f"reference has {len(ref_freqs)} channels "
                        f"({ref_freqs[0] / 1e6:.3f}\u2013{ref_freqs[-1] / 1e6:.3f} MHz), "
                        f"model '{m.model_name}' has {len(m_freqs)} channels "
                        f"({m_freqs[0] / 1e6:.3f}\u2013{m_freqs[-1] / 1e6:.3f} MHz). "
                        f"All models must share the same observation frequency grid."
                    )

            npix = hp.nside2npix(ref_nside)
            omega_pixel = 4 * np.pi / npix

            ps_models_data = []
            for m in models:
                if m.mode == "point_sources" and m._has_point_sources():
                    ipix_m = hp.ang2pix(ref_nside, np.pi / 2 - m._dec_rad, m._ra_rad)
                    ps_models_data.append((ipix_m, m._flux_ref, m._alpha, m))

            # Check if any model has polarized maps
            any_pol = any(
                m.has_polarized_healpix_maps
                for m in models
                if m.mode == "healpix_multifreq"
            ) or any(
                m._stokes_q is not None
                and (
                    np.any(m._stokes_q != 0)
                    or np.any(m._stokes_u != 0)
                    or np.any(m._stokes_v != 0)
                )
                for m in models
                if m.mode == "point_sources" and m._has_point_sources()
            )

            from .constants import C_LIGHT, K_BOLTZMANN

            combined_maps: dict[float, np.ndarray] = {}
            combined_q: dict[float, np.ndarray] = {} if any_pol else None
            combined_u: dict[float, np.ndarray] = {} if any_pol else None
            combined_v: dict[float, np.ndarray] = {} if any_pol else None

            for freq_hz in ref_freqs:
                combined_flux = np.zeros(npix, dtype=np.float64)
                combined_q_flux = np.zeros(npix, dtype=np.float64) if any_pol else None
                combined_u_flux = np.zeros(npix, dtype=np.float64) if any_pol else None
                combined_v_flux = np.zeros(npix, dtype=np.float64) if any_pol else None

                rj_factor = (
                    (2 * K_BOLTZMANN * freq_hz**2 / C_LIGHT**2) * omega_pixel / 1e-26
                )

                for m in models:
                    if m.mode == "healpix_multifreq":
                        t_map = m.get_map_at_frequency(freq_hz).astype(np.float64)
                        pos = t_map > 0
                        if np.any(pos):
                            combined_flux[pos] += brightness_temp_to_flux_density(
                                t_map[pos],
                                freq_hz,
                                omega_pixel,
                                method=brightness_conversion,
                            )

                        if any_pol and m.has_polarized_healpix_maps:
                            if m._healpix_q_maps is not None:
                                q_t = m._healpix_q_maps.get(freq_hz)
                                if q_t is not None:
                                    combined_q_flux += (
                                        q_t.astype(np.float64) * rj_factor
                                    )
                            if m._healpix_u_maps is not None:
                                u_t = m._healpix_u_maps.get(freq_hz)
                                if u_t is not None:
                                    combined_u_flux += (
                                        u_t.astype(np.float64) * rj_factor
                                    )
                            if m._healpix_v_maps is not None:
                                v_t = m._healpix_v_maps.get(freq_hz)
                                if v_t is not None:
                                    combined_v_flux += (
                                        v_t.astype(np.float64) * rj_factor
                                    )

                for ipix_m, flux_ref_m, alpha_m, m_obj in ps_models_data:
                    scale = (float(freq_hz) / ref_frequency) ** alpha_m
                    flux_at_f = flux_ref_m * scale
                    combined_flux += np.bincount(
                        ipix_m, weights=flux_at_f, minlength=npix
                    )

                    if any_pol and m_obj._stokes_q is not None:
                        combined_q_flux += np.bincount(
                            ipix_m, weights=m_obj._stokes_q * scale, minlength=npix
                        )
                        combined_u_flux += np.bincount(
                            ipix_m, weights=m_obj._stokes_u * scale, minlength=npix
                        )
                        combined_v_flux += np.bincount(
                            ipix_m, weights=m_obj._stokes_v * scale, minlength=npix
                        )

                combined_T_b = np.zeros(npix, dtype=np.float64)
                pos_flux = combined_flux > 0
                if np.any(pos_flux):
                    combined_T_b[pos_flux] = flux_density_to_brightness_temp(
                        combined_flux[pos_flux],
                        freq_hz,
                        omega_pixel,
                        method=brightness_conversion,
                    )
                combined_maps[freq_hz] = combined_T_b.astype(np.float32)

                if any_pol:
                    rj_inv = 1.0 / rj_factor if rj_factor != 0 else 0.0
                    combined_q[freq_hz] = (combined_q_flux * rj_inv).astype(np.float32)
                    combined_u[freq_hz] = (combined_u_flux * rj_inv).astype(np.float32)
                    combined_v[freq_hz] = (combined_v_flux * rj_inv).astype(np.float32)

            logger.info(
                f"Combined {len(models)} models into healpix_multifreq "
                f"({len(ref_freqs)} channels, nside={ref_nside}"
                f"{', stokes=IQUV' if any_pol else ''})"
            )

            sky = cls(
                _healpix_maps=combined_maps,
                _healpix_q_maps=combined_q,
                _healpix_u_maps=combined_u,
                _healpix_v_maps=combined_v,
                _healpix_nside=ref_nside,
                _observation_frequencies=ref_freqs,
                _native_format="healpix",
                frequency=float(ref_freqs[0]) if len(ref_freqs) > 0 else freq,
                model_name="combined",
                brightness_conversion=brightness_conversion,
                _precision=precision,
            )
            sky._ensure_dtypes()
            return sky

        # ==================================================================
        # ALL OTHER CASES: Concatenate as point sources first
        # ==================================================================
        all_sources = []
        for m in models:
            sources = m.to_point_sources(frequency=freq)
            all_sources.extend(sources)

        logger.info(f"Combined {len(models)} models: {len(all_sources)} total sources")

        combined = cls.from_point_sources(
            all_sources,
            model_name="combined",
            brightness_conversion=brightness_conversion,
            precision=precision,
        )
        combined.frequency = freq

        if representation == "healpix_map":
            if obs_frequency_config is None:
                raise ValueError(
                    "obs_frequency_config is required for healpix_map representation. "
                    "Provide a dict with: starting_frequency, frequency_interval, "
                    "frequency_bandwidth, frequency_unit."
                )

            combined.to_healpix_for_observation(
                nside=nside,
                obs_frequency_config=obs_frequency_config,
                ref_frequency=ref_frequency,
            )

        return combined

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
                self._healpix_nside, len(freqs), np.float32, n_stokes=n_stokes
            )
            return (
                f"SkyModel(native='{self._native_format}', model='{self.model_name}', "
                f"mode='healpix_multifreq', nside={self._healpix_nside}, "
                f"n_freq={len(freqs)}, freq_range={freq_range}MHz, "
                f"stokes='{stokes_components}', "
                f"memory={mem_info['total_mb']:.1f}MB)"
            )
        return (
            f"SkyModel(native='{self._native_format}', model='{self.model_name}', "
            f"n_sources={self.n_sources})"
        )
