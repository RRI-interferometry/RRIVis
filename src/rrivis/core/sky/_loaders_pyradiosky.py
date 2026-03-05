# rrivis/core/sky/_loaders_pyradiosky.py
"""Pyradiosky file loader mixin for SkyModel."""

import logging
import os
from typing import Any

import astropy.units as u
import healpy as hp
import numpy as np
from healpy.rotator import Rotator
from pyradiosky import SkyModel as PyRadioSkyModel

logger = logging.getLogger(__name__)


class _PyradioskyMixin:
    """Mixin providing pyradiosky file-loading classmethods for SkyModel."""

    @classmethod
    def from_pyradiosky_file(
        cls,
        filename: str,
        filetype: str | None = None,
        flux_limit: float = 0.0,
        reference_frequency_hz: float | None = None,
        brightness_conversion: str = "planck",
        precision: Any = None,
        frequencies: np.ndarray | None = None,
        obs_frequency_config: dict[str, Any] | None = None,
    ) -> "SkyModel":  # noqa: F821
        """
        Load a local sky model file via pyradiosky.

        Supports SkyH5, VOTable, text, and FHD formats (as handled by
        pyradiosky). Both ``component_type='point'`` and
        ``component_type='healpix'`` are supported.

        For HEALPix files, observation frequencies can be provided explicitly
        via ``frequencies`` or ``obs_frequency_config``. If the file has
        ``spectral_type='full'`` or ``'subband'`` and no explicit frequencies
        are given, the file's own frequency array is used.

        Parameters
        ----------
        filename : str
            Path to the sky model file.
        filetype : str, optional
            File format: "skyh5", "votable", "text", "fhd", etc.
            If None, pyradiosky infers from the file extension.
        flux_limit : float, default=0.0
            Minimum Stokes I flux in Jy at the reference frequency.
            Only used for point-source files.
        reference_frequency_hz : float, optional
            Reference frequency for Stokes I extraction (Hz).
            If None, uses the first frequency channel in the file.
            Only used for point-source files.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        frequencies : np.ndarray, optional
            Array of observation frequencies in Hz for HEALPix files.
            Takes precedence over ``obs_frequency_config``.
        obs_frequency_config : dict, optional
            Frequency configuration dict (keys: starting_frequency,
            frequency_interval, frequency_bandwidth, frequency_unit).
            Used for HEALPix files when ``frequencies`` is None.

        Returns
        -------
        SkyModel

        Raises
        ------
        FileNotFoundError
            If ``filename`` does not exist.
        ValueError
            If the file has an unsupported ``component_type``, or if a
            HEALPix file with ``spectral_type='spectral_index'`` or
            ``'flat'`` is loaded without explicit frequencies.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Sky model file not found: {filename}")

        sky = PyRadioSkyModel()
        sky.read(filename, filetype=filetype)

        if sky.component_type == "healpix":
            return cls._load_pyradiosky_healpix(
                sky,
                filename,
                frequencies,
                obs_frequency_config,
                brightness_conversion,
                precision,
            )
        elif sky.component_type != "point":
            raise ValueError(
                f"Unsupported component_type: '{sky.component_type}'. "
                "Only 'point' and 'healpix' are supported."
            )

        ref_freq_hz = reference_frequency_hz
        if ref_freq_hz is None:
            if sky.freq_array is not None and len(sky.freq_array) > 0:
                ref_freq_hz = float(sky.freq_array[0])
            else:
                raise ValueError(
                    "Cannot determine reference frequency. "
                    "Provide reference_frequency_hz explicitly."
                )

        if sky.freq_array is not None and len(sky.freq_array) > 1:
            ref_chan_idx = int(
                np.argmin(np.abs(np.array(sky.freq_array) - ref_freq_hz))
            )
        else:
            ref_chan_idx = 0

        # stokes shape: (4, Nfreqs, Ncomponents) or (4, 1, Ncomponents)
        stokes = sky.stokes
        stokes_i_ref = np.array(stokes[0, ref_chan_idx, :], dtype=np.float64)

        n_stokes = stokes.shape[0]
        stokes_q = (
            np.array(stokes[1, ref_chan_idx, :], dtype=np.float64)
            if n_stokes > 1
            else np.zeros_like(stokes_i_ref)
        )
        stokes_u = (
            np.array(stokes[2, ref_chan_idx, :], dtype=np.float64)
            if n_stokes > 2
            else np.zeros_like(stokes_i_ref)
        )
        stokes_v = (
            np.array(stokes[3, ref_chan_idx, :], dtype=np.float64)
            if n_stokes > 3
            else np.zeros_like(stokes_i_ref)
        )

        if sky.spectral_type == "spectral_index":
            spectral_indices = np.asarray(sky.spectral_index, dtype=np.float64)
        elif sky.spectral_type == "flat":
            spectral_indices = np.zeros(sky.Ncomponents, dtype=np.float64)
        else:
            # "full" or "subband": log power-law fit between first and last channel
            if sky.freq_array is not None and len(sky.freq_array) >= 2:
                s_first = np.array(stokes[0, 0, :], dtype=np.float64)
                s_last = np.array(stokes[0, -1, :], dtype=np.float64)
                freq_first = float(sky.freq_array[0])
                freq_last = float(sky.freq_array[-1])
                spectral_indices = np.zeros(sky.Ncomponents, dtype=np.float64)
                valid = (s_first > 0) & (s_last > 0)
                if np.any(valid):
                    spectral_indices[valid] = np.log(
                        s_first[valid] / s_last[valid]
                    ) / np.log(freq_first / freq_last)
            else:
                spectral_indices = np.zeros(sky.Ncomponents, dtype=np.float64)

        ra_arr = np.array(
            sky.ra.rad if hasattr(sky.ra, "rad") else sky.ra, dtype=np.float64
        )
        dec_arr = np.array(
            sky.dec.rad if hasattr(sky.dec, "rad") else sky.dec, dtype=np.float64
        )

        valid = np.isfinite(stokes_i_ref) & (stokes_i_ref >= flux_limit)
        n = int(valid.sum())

        model_name = f"pyradiosky:{os.path.basename(filename)}"
        logger.info(f"pyradiosky file loaded: {n:,} sources from {filename}")

        if n == 0:
            empty = cls._empty_sky(model_name, brightness_conversion, precision)
            empty.frequency = ref_freq_hz
            return empty

        sky = cls(
            _ra_rad=ra_arr[valid],
            _dec_rad=dec_arr[valid],
            _flux_ref=stokes_i_ref[valid],
            _alpha=spectral_indices[valid],
            _stokes_q=stokes_q[valid],
            _stokes_u=stokes_u[valid],
            _stokes_v=stokes_v[valid],
            _native_format="point_sources",
            model_name=model_name,
            frequency=ref_freq_hz,
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )
        sky._ensure_dtypes()
        return sky

    @classmethod
    def _load_pyradiosky_healpix(
        cls,
        psky: Any,
        filename: str,
        frequencies: np.ndarray | None,
        obs_frequency_config: dict[str, Any] | None,
        brightness_conversion: str,
        precision: Any,
    ) -> "SkyModel":  # noqa: F821
        """
        Load a pyradiosky HEALPix sky model as multi-frequency HEALPix maps.

        This is called internally by ``from_pyradiosky_file()`` when the file
        has ``component_type='healpix'``.

        Parameters
        ----------
        psky : pyradiosky.SkyModel
            Already-read pyradiosky SkyModel with ``component_type='healpix'``.
        filename : str
            Original file path (for logging and model_name).
        frequencies : np.ndarray or None
            Explicit observation frequencies in Hz.
        obs_frequency_config : dict or None
            Frequency configuration dict.
        brightness_conversion : str
            Conversion method: "planck" or "rayleigh-jeans".
        precision : Any
            Precision configuration.

        Returns
        -------
        SkyModel
            Sky model in healpix_multifreq mode.

        Raises
        ------
        ValueError
            If frequencies cannot be determined.
        """
        # --- Determine observation frequencies ---
        if frequencies is not None:
            obs_freqs = np.asarray(frequencies, dtype=np.float64)
        elif obs_frequency_config is not None:
            obs_freqs = cls._parse_frequency_config(obs_frequency_config)
        elif (
            psky.spectral_type in ("full", "subband")
            and psky.freq_array is not None
            and len(psky.freq_array) > 0
        ):
            obs_freqs = np.asarray(psky.freq_array.to(u.Hz).value, dtype=np.float64)
        else:
            raise ValueError(
                f"Cannot determine observation frequencies for HEALPix file "
                f"with spectral_type='{psky.spectral_type}'. "
                "Provide 'frequencies' or 'obs_frequency_config' explicitly."
            )

        n_freq = len(obs_freqs)
        nside = psky.nside

        logger.info(
            f"Loading pyradiosky HEALPix file: {n_freq} frequencies "
            f"({obs_freqs[0] / 1e6:.1f}\u2013{obs_freqs[-1] / 1e6:.1f} MHz), "
            f"nside={nside}, from {filename}"
        )

        # --- Evaluate stokes at observation frequencies ---
        psky_eval = psky.at_frequencies(obs_freqs * u.Hz, inplace=False)

        # --- Convert units to Kelvin if needed ---
        if psky_eval.stokes.unit.is_equivalent(u.Jy / u.sr):
            psky_eval.jansky_to_kelvin()

        # --- Determine coordinate frame and pixel handling ---
        is_galactic = False
        if hasattr(psky_eval, "frame") and psky_eval.frame is not None:
            frame_name = str(psky_eval.frame).lower()
            if "galactic" in frame_name:
                is_galactic = True

        rot = Rotator(coord=["G", "C"]) if is_galactic else None

        # Check for nested ordering
        is_nested = False
        if hasattr(psky_eval, "hpx_order") and psky_eval.hpx_order is not None:
            if psky_eval.hpx_order.lower() == "nested":
                is_nested = True

        # Check for sparse map (partial sky)
        hpx_inds = None
        if hasattr(psky_eval, "hpx_inds") and psky_eval.hpx_inds is not None:
            hpx_inds = np.asarray(psky_eval.hpx_inds)

        npix = hp.nside2npix(nside)

        # --- Extract Stokes I and build full-sky maps ---
        # psky_eval.stokes shape: (4, Nfreqs, Ncomponents)
        stokes_data = np.asarray(psky_eval.stokes.value)

        healpix_maps: dict[float, np.ndarray] = {}
        for i_freq in range(n_freq):
            stokes_i = stokes_data[0, i_freq, :]

            if hpx_inds is not None:
                # Sparse map: fill only specified pixels
                full_map = np.zeros(npix, dtype=np.float64)
                pixel_indices = hpx_inds
                if is_nested:
                    pixel_indices = hp.nest2ring(nside, pixel_indices)
                full_map[pixel_indices] = stokes_i
            else:
                # Full-sky map
                full_map = np.array(stokes_i, dtype=np.float64)
                if is_nested:
                    full_map = hp.reorder(full_map, n2r=True)

            if rot is not None:
                full_map = rot.rotate_map_pixel(full_map)

            healpix_maps[float(obs_freqs[i_freq])] = full_map.astype(np.float32)

        model_name = f"pyradiosky:{os.path.basename(filename)}"
        logger.info(
            f"pyradiosky HEALPix loaded: {npix} pixels \u00d7 {n_freq} frequencies"
        )

        sky_model = cls(
            _healpix_maps=healpix_maps,
            _healpix_nside=nside,
            _observation_frequencies=obs_freqs,
            _native_format="healpix",
            frequency=float(obs_freqs[0]),
            model_name=model_name,
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )
        sky_model._ensure_dtypes()
        return sky_model
