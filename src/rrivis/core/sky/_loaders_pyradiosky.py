# rrivis/core/sky/_loaders_pyradiosky.py
"""Pyradiosky file loader functions for SkyModel."""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal

import astropy.units as u
import healpy as hp
import numpy as np
from healpy.rotator import Rotator
from pyradiosky import SkyModel as PyRadioSkyModel

from rrivis.utils.frequency import parse_frequency_config

from ._data import HealpixData, PointSourceData
from ._precision import get_sky_storage_dtype
from ._registry import register_loader
from .model import SkyFormat

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .region import SkyRegion

logger = logging.getLogger(__name__)


class LossyConversionWarning(UserWarning):
    """Warn when pyradiosky import drops higher-order spectral information."""


@register_loader(
    "pyradiosky_file",
    config_section="pyradiosky",
    use_flag="use_pyradiosky",
    category="file",
    requires_file=True,
    network_service=None,
    aliases=["pyradiosky"],
    config_fields={
        "filename": "filename",
        "filetype": "filetype",
        "flux_limit": "flux_limit",
        "reference_frequency_hz": "reference_frequency_hz",
        "spectral_loss_policy": "spectral_loss_policy",
    },
)
def load_pyradiosky_file(
    filename: str,
    filetype: str | None = None,
    flux_limit: float = 0.0,
    reference_frequency_hz: float | None = None,
    spectral_loss_policy: Literal["warn", "error"] = "warn",
    brightness_conversion: str = "planck",
    precision: PrecisionConfig | None = None,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    region: SkyRegion | None = None,
    memmap_path: str | None = None,
) -> SkyModel:  # noqa: F821
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
    from .model import SkyModel

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Sky model file not found: {filename}")

    sky = PyRadioSkyModel()
    sky.read(filename, filetype=filetype)

    if sky.component_type == "healpix":
        return _load_pyradiosky_healpix(
            sky,
            filename,
            frequencies,
            obs_frequency_config,
            brightness_conversion,
            precision,
            region=region,
            memmap_path=memmap_path,
        )
    elif sky.component_type != "point":
        raise ValueError(
            f"Unsupported component_type: '{sky.component_type}'. "
            "Only 'point' and 'healpix' are supported."
        )

    ref_freq_hz = reference_frequency_hz
    if ref_freq_hz is None:
        if sky.freq_array is not None and len(sky.freq_array) > 0:
            freq_arr = sky.freq_array
            if hasattr(freq_arr, "to_value"):
                ref_freq_hz = float(freq_arr[0].to_value(u.Hz))
            else:
                ref_freq_hz = float(freq_arr[0])
        elif sky.reference_frequency is not None and len(sky.reference_frequency) > 0:
            # spectral_index type: frequency stored per-component
            rf = sky.reference_frequency
            if hasattr(rf, "to_value"):
                ref_freq_hz = float(rf[0].to_value(u.Hz))
            else:
                ref_freq_hz = float(rf[0])
        else:
            raise ValueError(
                "Cannot determine reference frequency. "
                "Provide reference_frequency_hz explicitly."
            )

    if sky.freq_array is not None and len(sky.freq_array) > 1:
        freq_vals = (
            sky.freq_array.to_value(u.Hz)
            if hasattr(sky.freq_array, "to_value")
            else np.asarray(sky.freq_array, dtype=np.float64)
        )
        ref_chan_idx = int(np.argmin(np.abs(freq_vals - ref_freq_hz)))
    else:
        ref_chan_idx = 0

    # stokes shape: (4, Nfreqs, Ncomponents) or (4, 1, Ncomponents)
    stokes = sky.stokes
    if hasattr(stokes, "to_value"):
        stokes = stokes.to_value(u.Jy)
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
        message = (
            "Loading a pyradiosky point file with spectral_type="
            f"{sky.spectral_type!r} collapses the per-channel spectrum to a single "
            "fitted spectral index. Set spectral_loss_policy='error' to reject "
            "this import."
        )
        if spectral_loss_policy == "error":
            raise ValueError(message)
        warnings.warn(message, LossyConversionWarning, stacklevel=2)

        # "full" or "subband": log power-law fit between first and last channel
        if sky.freq_array is not None and len(sky.freq_array) >= 2:
            s_first = np.array(stokes[0, 0, :], dtype=np.float64)
            s_last = np.array(stokes[0, -1, :], dtype=np.float64)
            freq_first = float(
                sky.freq_array[0].to_value(u.Hz)
                if hasattr(sky.freq_array, "to_value")
                else sky.freq_array[0]
            )
            freq_last = float(
                sky.freq_array[-1].to_value(u.Hz)
                if hasattr(sky.freq_array, "to_value")
                else sky.freq_array[-1]
            )
            spectral_indices = np.zeros(sky.Ncomponents, dtype=np.float64)
            valid = (s_first > 0) & (s_last > 0)
            if np.any(valid):
                spectral_indices[valid] = np.log(
                    s_first[valid] / s_last[valid]
                ) / np.log(freq_first / freq_last)
        else:
            spectral_indices = np.zeros(sky.Ncomponents, dtype=np.float64)

    # Build per-source reference frequency array
    per_source_ref_freq = None
    if sky.spectral_type == "spectral_index" and sky.reference_frequency is not None:
        rf = sky.reference_frequency
        per_source_ref_freq = (
            rf.to_value(u.Hz).astype(np.float64)
            if hasattr(rf, "to_value")
            else np.asarray(rf, dtype=np.float64)
        )

    ra_arr = np.array(
        sky.ra.rad if hasattr(sky.ra, "rad") else sky.ra, dtype=np.float64
    )
    dec_arr = np.array(
        sky.dec.rad if hasattr(sky.dec, "rad") else sky.dec, dtype=np.float64
    )
    source_name = None
    if getattr(sky, "name", None) is not None:
        source_name = np.asarray(sky.name)

    source_id = None
    extra_columns: dict[str, np.ndarray] = {}
    extra = getattr(sky, "extra_columns", None)
    if extra is not None and getattr(extra.dtype, "names", None) is not None:
        for name in extra.dtype.names:
            values = np.asarray(extra[name])
            if name == "source_id":
                source_id = values
            else:
                extra_columns[name] = values

    valid = np.isfinite(stokes_i_ref) & (stokes_i_ref >= flux_limit)

    # Client-side region filter
    if region is not None:
        in_region = region.contains(ra_arr, dec_arr)
        valid = valid & in_region

    n = int(valid.sum())

    model_name = f"pyradiosky:{os.path.basename(filename)}"
    logger.info(f"pyradiosky file loaded: {n:,} sources from {filename}")

    if n == 0:
        from ._factories import create_empty

        return create_empty(
            model_name,
            brightness_conversion,
            precision,
            reference_frequency=ref_freq_hz,
        )

    sky_model = SkyModel(
        point=PointSourceData(
            ra_rad=ra_arr[valid],
            dec_rad=dec_arr[valid],
            flux=stokes_i_ref[valid],
            spectral_index=spectral_indices[valid],
            stokes_q=stokes_q[valid],
            stokes_u=stokes_u[valid],
            stokes_v=stokes_v[valid],
            ref_freq=(
                per_source_ref_freq[valid]
                if per_source_ref_freq is not None
                else np.full(n, ref_freq_hz, dtype=np.float64)
            ),
            source_name=source_name[valid] if source_name is not None else None,
            source_id=source_id[valid] if source_id is not None else None,
            extra_columns={
                name: values[valid] for name, values in extra_columns.items()
            },
        ),
        source_format=SkyFormat.POINT_SOURCES,
        model_name=model_name,
        reference_frequency=ref_freq_hz,
        brightness_conversion=brightness_conversion,
        _precision=precision,
    )
    return sky_model


def _load_pyradiosky_healpix(
    psky: Any,
    filename: str,
    frequencies: np.ndarray | None,
    obs_frequency_config: dict[str, Any] | None,
    brightness_conversion: str,
    precision: PrecisionConfig | None,
    region: SkyRegion | None = None,
    memmap_path: str | None = None,
) -> SkyModel:  # noqa: F821
    """
    Load a pyradiosky HEALPix sky model as multi-frequency HEALPix maps.

    This is called internally by ``load_pyradiosky_file()`` when the file
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
        Sky model in healpix_map mode.

    Raises
    ------
    ValueError
        If frequencies cannot be determined.
    """
    from .model import SkyModel

    # --- Determine observation frequencies ---
    if frequencies is not None:
        obs_freqs = np.asarray(frequencies, dtype=np.float64)
    elif obs_frequency_config is not None:
        obs_freqs = parse_frequency_config(obs_frequency_config)
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
    is_sparse = hpx_inds is not None and len(hpx_inds) < npix
    if is_sparse and rot is not None:
        raise NotImplementedError(
            "Sparse pyradiosky HEALPix maps in galactic frame are not supported yet."
        )

    # --- Extract Stokes and build full-sky maps ---
    # psky_eval.stokes shape: (n_stokes, Nfreqs, Ncomponents)
    stokes_data = np.asarray(psky_eval.stokes.value)
    n_stokes_avail = stokes_data.shape[0]
    has_pol = n_stokes_avail >= 3

    if has_pol:
        if brightness_conversion != "rayleigh-jeans":
            logger.info(
                "Using Rayleigh-Jeans conversion (required: polarized K_RJ data)"
            )
        brightness_conversion = "rayleigh-jeans"

    pix = None
    if hpx_inds is not None:
        pix = np.array(hpx_inds, copy=True)
        if is_nested:
            pix = hp.nest2ring(nside, pix)

    if region is not None:
        region_mask = region.healpix_mask(nside)
        if is_sparse and pix is not None:
            keep_mask = region_mask[pix]
            pix = pix[keep_mask]
            stokes_data = stokes_data[:, :, keep_mask]
            logger.info(
                f"Region mask applied: {int(keep_mask.sum())}/{len(keep_mask)} "
                "stored pixels retained"
            )
        else:
            logger.info("Region mask will be applied to the full-sky HEALPix cube.")

    from ._allocation import allocate_cube, ensure_scratch_dir, finalize_cube

    scratch = ensure_scratch_dir(memmap_path) if memmap_path is not None else None
    hp_dtype = get_sky_storage_dtype(precision, "healpix_maps")
    if is_sparse:
        stored_npix = 0 if pix is None else len(pix)
        i_arr = allocate_cube((n_freq, stored_npix), hp_dtype, scratch, "i_maps")
        q_arr = (
            allocate_cube((n_freq, stored_npix), hp_dtype, scratch, "q_maps")
            if has_pol
            else None
        )
        u_arr = (
            allocate_cube((n_freq, stored_npix), hp_dtype, scratch, "u_maps")
            if has_pol
            else None
        )
        v_arr = (
            allocate_cube((n_freq, stored_npix), hp_dtype, scratch, "v_maps")
            if n_stokes_avail >= 4
            else None
        )
    else:

        def _build_full_map(data_1d: np.ndarray) -> np.ndarray:
            """Build a full-sky RING-ordered map from raw Stokes data."""
            if pix is not None:
                full = np.zeros(npix, dtype=np.float64)
                full[pix] = data_1d
                return full
            full = np.array(data_1d, dtype=np.float64)
            if is_nested:
                full = hp.reorder(full, n2r=True)
            return full

        i_arr = allocate_cube((n_freq, npix), hp_dtype, scratch, "i_maps")
        q_arr = (
            allocate_cube((n_freq, npix), hp_dtype, scratch, "q_maps")
            if has_pol
            else None
        )
        u_arr = (
            allocate_cube((n_freq, npix), hp_dtype, scratch, "u_maps")
            if has_pol
            else None
        )
        v_arr = (
            allocate_cube((n_freq, npix), hp_dtype, scratch, "v_maps")
            if n_stokes_avail >= 4
            else None
        )

    for fi in range(n_freq):
        if is_sparse:
            i_map = np.asarray(stokes_data[0, fi, :], dtype=np.float64)
            if has_pol:
                q_map = np.asarray(stokes_data[1, fi, :], dtype=np.float64)
                u_map = np.asarray(stokes_data[2, fi, :], dtype=np.float64)
            if n_stokes_avail >= 4:
                v_map = np.asarray(stokes_data[3, fi, :], dtype=np.float64)

            if pix is not None:
                i_map = i_map[: len(pix)]
                if has_pol:
                    q_map = q_map[: len(pix)]
                    u_map = u_map[: len(pix)]
                if n_stokes_avail >= 4:
                    v_map = v_map[: len(pix)]

            i_arr[fi] = i_map.astype(hp_dtype)
            if has_pol:
                q_arr[fi] = q_map.astype(hp_dtype)
                u_arr[fi] = u_map.astype(hp_dtype)
            if n_stokes_avail >= 4:
                v_arr[fi] = v_map.astype(hp_dtype)
        else:
            i_map = _build_full_map(stokes_data[0, fi, :])

            if has_pol:
                q_map = _build_full_map(stokes_data[1, fi, :])
                u_map = _build_full_map(stokes_data[2, fi, :])

                if rot is not None:
                    iqu = np.array([i_map, q_map, u_map])
                    iqu_rot = rot.rotate_map_alms(iqu)
                    i_map = iqu_rot[0]
                    q_map = iqu_rot[1]
                    u_map = iqu_rot[2]

                i_arr[fi] = i_map.astype(hp_dtype)
                q_arr[fi] = q_map.astype(hp_dtype)
                u_arr[fi] = u_map.astype(hp_dtype)

                if n_stokes_avail >= 4:
                    v_map = _build_full_map(stokes_data[3, fi, :])
                    if rot is not None:
                        v_map = rot.rotate_map_pixel(v_map)
                    v_arr[fi] = v_map.astype(hp_dtype)
            else:
                if rot is not None:
                    i_map = rot.rotate_map_pixel(i_map)
                i_arr[fi] = i_map.astype(hp_dtype)

    if region is not None and not is_sparse:
        mask = region.healpix_mask(nside)
        n_retained = int(mask.sum())
        i_arr[:, ~mask] = 0.0
        for arr in (q_arr, u_arr, v_arr):
            if arr is not None:
                arr[:, ~mask] = 0.0
        logger.info(f"Region mask applied: {n_retained}/{npix} pixels retained")

    model_name = f"pyradiosky:{os.path.basename(filename)}"
    stokes_label = "I"
    if has_pol:
        stokes_label = "IQU" + ("V" if n_stokes_avail >= 4 else "")
    logger.info(
        f"pyradiosky HEALPix loaded: {i_arr.shape[1]} pixels \u00d7 {n_freq} frequencies, "
        f"stokes={stokes_label}"
    )

    # Flush and re-open read-only if memmap-backed.
    i_arr = finalize_cube(i_arr, scratch, "i_maps")
    if q_arr is not None:
        q_arr = finalize_cube(q_arr, scratch, "q_maps")
    if u_arr is not None:
        u_arr = finalize_cube(u_arr, scratch, "u_maps")
    if v_arr is not None:
        v_arr = finalize_cube(v_arr, scratch, "v_maps")

    return SkyModel(
        healpix=HealpixData(
            maps=i_arr,
            nside=nside,
            frequencies=obs_freqs,
            hpx_inds=pix if is_sparse else None,
            q_maps=q_arr,
            u_maps=u_arr,
            v_maps=v_arr,
        ),
        source_format=SkyFormat.HEALPIX,
        model_name=model_name,
        brightness_conversion=brightness_conversion,
        _precision=precision,
    )
