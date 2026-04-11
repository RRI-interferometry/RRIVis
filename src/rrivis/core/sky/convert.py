# rrivis/core/sky/convert.py
"""Bidirectional conversion between point-source and HEALPix representations.

Pure functions that accept and return raw numpy arrays. No SkyModel dependency.
"""

from __future__ import annotations

import logging
from typing import Any

import healpy as hp
import numpy as np

from .constants import (
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
    rayleigh_jeans_factor,
)
from .model import _empty_source_arrays
from .spectral import apply_faraday_rotation, compute_spectral_scale

logger = logging.getLogger(__name__)


def _fit_pixel_spectral_indices(
    healpix_maps: np.ndarray,
    observation_frequencies: np.ndarray,
    valid_idx: np.ndarray,
    ref_frequency: float,
    omega: float,
    brightness_conversion: str,
) -> np.ndarray:
    """Fit per-pixel spectral indices from multi-frequency HEALPix maps.

    Uses vectorized log-log linear regression:
    ``log10(S) = log10(S_ref) + alpha * log10(f / f_ref)``

    Parameters
    ----------
    healpix_maps : np.ndarray
        Stokes I brightness temperature cube, shape ``(n_freq, npix)``.
    observation_frequencies : np.ndarray
        Frequency array in Hz, shape ``(n_freq,)``.
    valid_idx : np.ndarray
        Indices of pixels with positive flux at the reference frequency.
    ref_frequency : float
        Reference frequency in Hz.
    omega : float
        Pixel solid angle in steradians.
    brightness_conversion : str
        ``"planck"`` or ``"rayleigh-jeans"``.

    Returns
    -------
    np.ndarray
        Fitted spectral indices, shape ``(n_valid,)``.
    """
    n_freq = len(observation_frequencies)
    n_valid = len(valid_idx)

    # Convert T_b -> flux density (Jy) at each frequency for valid pixels
    # flux_matrix shape: (n_freq, n_valid)
    flux_matrix = np.zeros((n_freq, n_valid), dtype=np.float64)
    for fi, freq in enumerate(observation_frequencies):
        t_vals = healpix_maps[fi][valid_idx].astype(np.float64)
        pos = t_vals > 0
        if np.any(pos):
            flux_matrix[fi, pos] = brightness_temp_to_flux_density(
                t_vals[pos], float(freq), omega, method=brightness_conversion
            )

    # Build mask: pixel must have positive flux at >=2 frequencies to fit
    positive_mask = flux_matrix > 0  # (n_freq, n_valid)
    n_positive = positive_mask.sum(axis=0)  # (n_valid,)
    fittable = n_positive >= 2

    alpha = np.zeros(n_valid, dtype=np.float64)

    if not np.any(fittable):
        return alpha

    # Vectorized log-log linear regression for fittable pixels
    log_ratio = np.log10(observation_frequencies / ref_frequency)  # (n_freq,)

    # For pixels where all frequencies are valid, use the fast path
    all_valid = n_positive == n_freq
    if np.any(all_valid):
        log_S = np.log10(flux_matrix[:, all_valid])  # (n_freq, n_all_valid)
        x = log_ratio
        N = n_freq
        sum_x = np.sum(x)
        sum_x2 = np.sum(x**2)
        sum_y = np.sum(log_S, axis=0)  # (n_all_valid,)
        sum_xy = np.sum(x[:, None] * log_S, axis=0)  # (n_all_valid,)
        denom = N * sum_x2 - sum_x**2
        if abs(denom) > 1e-30:
            alpha[all_valid] = (N * sum_xy - sum_x * sum_y) / denom

    # For pixels with partial frequency coverage, fit individually
    partial = fittable & ~all_valid
    if np.any(partial):
        partial_idx = np.where(partial)[0]
        for pi in partial_idx:
            mask = positive_mask[:, pi]
            x = log_ratio[mask]
            y = np.log10(flux_matrix[mask, pi])
            N = len(x)
            sum_x = np.sum(x)
            sum_x2 = np.sum(x**2)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            denom = N * sum_x2 - sum_x**2
            if abs(denom) > 1e-30:
                alpha[pi] = (N * sum_xy - sum_x * sum_y) / denom

    # Clamp non-finite values
    bad = ~np.isfinite(alpha)
    if np.any(bad):
        alpha[bad] = 0.0
        logger.warning(
            "Clamped %d non-finite fitted spectral indices to 0.",
            int(bad.sum()),
        )

    return alpha


def healpix_map_to_point_arrays(
    temp_map: np.ndarray,
    frequency: float,
    brightness_conversion: str,
    healpix_q_maps: np.ndarray | None = None,
    healpix_u_maps: np.ndarray | None = None,
    healpix_v_maps: np.ndarray | None = None,
    observation_frequencies: np.ndarray | None = None,
    freq_index: int | None = None,
    healpix_maps: np.ndarray | None = None,
    ref_freq_out: float | None = None,
) -> dict[str, np.ndarray]:
    """Convert a HEALPix brightness temperature map to columnar point-source arrays.

    Only positive-temperature pixels are stored (no flux_limit filtering
    here -- apply that in the caller).

    When ``healpix_maps`` (full multi-frequency Stokes I cube) and
    ``observation_frequencies`` (with ≥2 entries) are provided, a per-pixel
    spectral index is fitted via log-log linear regression of flux density
    vs. frequency.  Otherwise all pixels receive ``alpha=0``.

    Parameters
    ----------
    temp_map : np.ndarray
        Brightness temperature map in Kelvin (reference frequency slice).
    frequency : float
        Frequency in Hz for T_b -> Jy conversion (reference frequency).
    brightness_conversion : str
        Conversion method: ``"planck"`` or ``"rayleigh-jeans"``.
    healpix_q_maps : np.ndarray or None
        Stokes Q maps, shape ``(n_freq, npix)`` or None.
    healpix_u_maps : np.ndarray or None
        Stokes U maps, shape ``(n_freq, npix)`` or None.
    healpix_v_maps : np.ndarray or None
        Stokes V maps, shape ``(n_freq, npix)`` or None.
    observation_frequencies : np.ndarray or None
        Frequency array in Hz, used to find freq_index if not given.
    freq_index : int or None
        Index into the (n_freq, npix) polarization arrays. If None and
        ``observation_frequencies`` is provided, the nearest index is found.
    healpix_maps : np.ndarray or None
        Full Stokes I brightness temperature cube, shape ``(n_freq, npix)``.
        When provided together with ``observation_frequencies`` (≥2 channels),
        enables per-pixel spectral index fitting.
    ref_freq_out : float or None
        Reference frequency stored in the output ``ref_freq`` array.
        Defaults to ``frequency`` if not given.

    Returns
    -------
    dict
        TypedDict with keys: ``"ra_rad"``, ``"dec_rad"``, ``"flux"``,
        ``"spectral_index"``, ``"ref_freq"``, ``"stokes_q"``, ``"stokes_u"``,
        ``"stokes_v"``.
        All arrays have shape ``(N,)`` where N is the number of valid pixels.
    """
    npix = len(temp_map)
    nside = hp.npix2nside(npix)
    omega = 4 * np.pi / npix
    ref_freq_val = ref_freq_out if ref_freq_out is not None else frequency

    flux_jy = np.zeros(npix, dtype=np.float64)
    pos = temp_map > 0
    if np.any(pos):
        flux_jy[pos] = brightness_temp_to_flux_density(
            temp_map[pos].astype(np.float64),
            frequency,
            omega,
            method=brightness_conversion,
        )

    valid_idx = np.where(flux_jy > 0)[0]
    if len(valid_idx) == 0:
        logger.warning("No pixels with positive flux in HEALPix map")
        return _empty_source_arrays()

    theta, phi = hp.pix2ang(nside, valid_idx)
    ra_rad = phi  # phi = RA in radians
    dec_rad = np.pi / 2 - theta  # colatitude -> declination
    flux_ref = flux_jy[valid_idx]
    n = len(valid_idx)

    # --- Per-pixel spectral index fitting ---
    can_fit = (
        healpix_maps is not None
        and observation_frequencies is not None
        and len(observation_frequencies) >= 2
    )

    if can_fit:
        alpha = _fit_pixel_spectral_indices(
            healpix_maps,
            observation_frequencies,
            valid_idx,
            frequency,
            omega,
            brightness_conversion,
        )
        logger.info(
            "Fitted per-pixel spectral indices from %d frequency channels "
            "(%.1f\u2013%.1f MHz). Median alpha=%.3f.",
            len(observation_frequencies),
            observation_frequencies[0] / 1e6,
            observation_frequencies[-1] / 1e6,
            float(np.median(alpha)),
        )
    else:
        alpha = np.zeros(n, dtype=np.float64)
        n_freq = (
            len(observation_frequencies) if observation_frequencies is not None else 0
        )
        if n_freq <= 1:
            logger.warning(
                "Only %d frequency channel available \u2014 cannot fit spectral "
                "index. All pixels assigned alpha=0 (flat spectrum). For "
                "accurate multi-frequency results, use 'healpix_map' "
                "representation directly.",
                max(n_freq, 1),
            )
        else:
            logger.warning(
                "HEALPix-to-point-source conversion assigns alpha=0 (flat "
                "spectrum) to all pixels. For accurate multi-frequency "
                "results, use 'healpix_map' representation directly.",
            )

    # Position quantization warning
    resol_arcmin = np.degrees(hp.nside2resol(nside)) * 60
    logger.warning(
        "HEALPix-to-point-source conversion: source positions are quantized "
        "to pixel centers (nside=%d, angular resolution ~%.2f arcmin). "
        "Sub-pixel positions from the original catalog are lost.",
        nside,
        resol_arcmin,
    )

    # Stokes Q/U/V use Rayleigh-Jeans (linear, sign-preserving) rather than
    # Planck because: (a) Q/U/V values can be negative, which Planck cannot
    # handle; (b) at the small amplitudes typical of polarized emission, the
    # RJ approximation is adequate. Stokes I above uses the user's chosen
    # brightness_conversion (Planck by default) for the non-linear T_b -> Jy
    # conversion of always-positive intensity values.
    rj_factor = rayleigh_jeans_factor(frequency, omega)

    # Resolve freq_index if polarization maps are array-indexed
    fi = freq_index
    if fi is None and observation_frequencies is not None:
        fi = int(np.argmin(np.abs(observation_frequencies - frequency)))

    # Extract Q/U/V from polarization maps if available
    if healpix_q_maps is not None and fi is not None:
        stokes_q = healpix_q_maps[fi][valid_idx].astype(np.float64) * rj_factor
    else:
        stokes_q = np.zeros(n, dtype=np.float64)

    if healpix_u_maps is not None and fi is not None:
        stokes_u = healpix_u_maps[fi][valid_idx].astype(np.float64) * rj_factor
    else:
        stokes_u = np.zeros(n, dtype=np.float64)

    if healpix_v_maps is not None and fi is not None:
        stokes_v = healpix_v_maps[fi][valid_idx].astype(np.float64) * rj_factor
    else:
        stokes_v = np.zeros(n, dtype=np.float64)

    return {
        "ra_rad": ra_rad,
        "dec_rad": dec_rad,
        "flux": flux_ref,
        "spectral_index": alpha,
        "ref_freq": np.full(n, ref_freq_val, dtype=np.float64),
        "stokes_q": stokes_q,
        "stokes_u": stokes_u,
        "stokes_v": stokes_v,
        "rotation_measure": None,
        "major_arcsec": None,
        "minor_arcsec": None,
        "pa_deg": None,
        "spectral_coeffs": None,
    }


def bin_sources_to_flux(
    ipix: np.ndarray,
    flux: np.ndarray,
    spectral_index: np.ndarray,
    spectral_coeffs: np.ndarray | None,
    freq: float,
    ref_frequency: float | np.ndarray,
    npix: int,
) -> np.ndarray:
    """Bin point sources into a HEALPix flux density map at a given frequency.

    Computes the spectral scaling factor for each source and accumulates
    the scaled flux into HEALPix pixels via ``np.bincount``.

    Parameters
    ----------
    ipix : np.ndarray
        HEALPix pixel index for each source, shape ``(N_sources,)``.
    flux : np.ndarray
        Reference flux density in Jy, shape ``(N_sources,)``.
    spectral_index : np.ndarray
        Spectral index, shape ``(N_sources,)``.
    spectral_coeffs : np.ndarray or None
        Log-polynomial coefficients, shape ``(N_sources, N_terms)``.
    freq : float
        Observation frequency in Hz.
    ref_frequency : float or np.ndarray
        Reference frequency in Hz (scalar or per-source).
    npix : int
        Total number of HEALPix pixels.

    Returns
    -------
    np.ndarray
        Flux density map in Jy, shape ``(npix,)``.
    """
    scale = compute_spectral_scale(spectral_index, spectral_coeffs, freq, ref_frequency)
    flux_f = flux * scale
    return np.bincount(ipix, weights=flux_f, minlength=npix)


def point_sources_to_healpix_maps(
    ra_rad: np.ndarray,
    dec_rad: np.ndarray,
    flux: np.ndarray,
    spectral_index: np.ndarray,
    spectral_coeffs: np.ndarray | None,
    stokes_q: np.ndarray | None,
    stokes_u: np.ndarray | None,
    stokes_v: np.ndarray | None,
    rotation_measure: np.ndarray | None,
    nside: int,
    frequencies: np.ndarray,
    ref_frequency: float | np.ndarray,
    brightness_conversion: str,
    estimate_memory_fn: Any = None,
    output_dtype: np.dtype = np.float32,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Convert point sources to multi-frequency HEALPix brightness temperature maps.

    Vectorized implementation: uses ``np.bincount`` for O(N_sources) memory per
    frequency channel instead of a Python loop over sources.

    Parameters
    ----------
    ra_rad, dec_rad : np.ndarray
        Source coordinates in radians, shape ``(N_sources,)``.
    flux : np.ndarray
        Reference flux density in Jy, shape ``(N_sources,)``.
    spectral_index : np.ndarray
        Spectral index, shape ``(N_sources,)``.
    spectral_coeffs : np.ndarray or None
        Log-polynomial coefficients, shape ``(N_sources, N_terms)``.
    stokes_q, stokes_u, stokes_v : np.ndarray or None
        Stokes polarization, shape ``(N_sources,)``.
    rotation_measure : np.ndarray or None
        Rotation measure in rad/m^2, shape ``(N_sources,)``.
    nside : int
        HEALPix NSIDE parameter.
    frequencies : np.ndarray
        Array of frequencies in Hz.
    ref_frequency : float or np.ndarray
        Reference frequency in Hz. Can be a scalar (shared by all sources)
        or a per-source array of shape ``(N_sources,)`` for correct spectral
        scaling of sources from different catalogs.
    brightness_conversion : str
        ``"planck"`` or ``"rayleigh-jeans"``.
    estimate_memory_fn : callable, optional
        Function ``(nside, n_freq, dtype, n_stokes) -> dict`` for logging.
    output_dtype : np.dtype, default np.float32
        Dtype for output HEALPix arrays. Use ``precision.sky_model.get_dtype("healpix_maps")``
        to respect the user's precision configuration.

    Returns
    -------
    i_maps : np.ndarray
        Stokes I brightness temperature maps, shape ``(n_freq, npix)``.
    q_maps : np.ndarray or None
        Stokes Q maps (K_RJ), shape ``(n_freq, npix)``, or None.
    u_maps : np.ndarray or None
        Stokes U maps (K_RJ), shape ``(n_freq, npix)``, or None.
    v_maps : np.ndarray or None
        Stokes V maps (K_RJ), shape ``(n_freq, npix)``, or None.
    """
    npix = hp.nside2npix(nside)
    n_freq = len(frequencies)
    n_sources = len(ra_rad)

    if n_sources == 0:
        empty = np.zeros((n_freq, npix), dtype=output_dtype)
        return empty, None, None, None

    omega_pixel = 4 * np.pi / npix

    # Check if any source has non-zero polarization
    has_pol = (
        stokes_q is not None
        and stokes_u is not None
        and stokes_v is not None
        and (np.any(stokes_q != 0) or np.any(stokes_u != 0) or np.any(stokes_v != 0))
    )

    n_stokes = 4 if has_pol else 1
    if estimate_memory_fn is not None:
        mem_info = estimate_memory_fn(nside, n_freq, output_dtype, n_stokes)
        logger.info(
            f"Creating {n_freq} HEALPix maps (nside={nside}, "
            f"stokes={'IQUV' if has_pol else 'I'}): "
            f"~{mem_info['total_mb']:.1f} MB"
        )

    ipix = hp.ang2pix(nside, np.pi / 2 - dec_rad, ra_rad)

    # Detect pixel collisions (multiple sources in one pixel)
    _unique_pixels, _counts = np.unique(ipix, return_counts=True)
    _multi = _counts > 1
    if np.any(_multi):
        n_collisions = int(np.sum(_multi))
        n_merged = int(np.sum(_counts[_multi]))
        logger.warning(
            "HEALPix pixelization: %d sources were merged into %d pixels "
            "(out of %d total sources). Individual source identities and "
            "per-source spectral indices are irreversibly combined. "
            "Increase nside (currently %d) to reduce merging.",
            n_merged,
            n_collisions,
            n_sources,
            nside,
        )

    i_arr = np.zeros((n_freq, npix), dtype=output_dtype)
    q_arr = np.zeros((n_freq, npix), dtype=output_dtype) if has_pol else None
    u_arr = np.zeros((n_freq, npix), dtype=output_dtype) if has_pol else None
    v_arr = np.zeros((n_freq, npix), dtype=output_dtype) if has_pol else None

    for fi, freq in enumerate(frequencies):
        scale = compute_spectral_scale(
            spectral_index, spectral_coeffs, float(freq), ref_frequency
        )

        flux_map = bin_sources_to_flux(
            ipix,
            flux,
            spectral_index,
            spectral_coeffs,
            float(freq),
            ref_frequency,
            npix,
        )

        temp_out = np.zeros(npix, dtype=output_dtype)
        occupied = flux_map > 0
        if np.any(occupied):
            temp_out[occupied] = flux_density_to_brightness_temp(
                flux_map[occupied],
                float(freq),
                omega_pixel,
                method=brightness_conversion,
            ).astype(output_dtype)
        i_arr[fi] = temp_out

        if has_pol:
            # Jy -> K_RJ via Rayleigh-Jeans (linear, sign-preserving)
            rj_inv = 1.0 / rayleigh_jeans_factor(float(freq), omega_pixel)

            q_flux, u_flux = apply_faraday_rotation(
                stokes_q,
                stokes_u,
                rotation_measure,
                float(freq),
                ref_frequency,
                scale,
            )
            q_map = np.bincount(ipix, weights=q_flux, minlength=npix)
            q_arr[fi] = (q_map * rj_inv).astype(output_dtype)

            u_map = np.bincount(ipix, weights=u_flux, minlength=npix)
            u_arr[fi] = (u_map * rj_inv).astype(output_dtype)

            v_flux = stokes_v * scale
            v_map = np.bincount(ipix, weights=v_flux, minlength=npix)
            v_arr[fi] = (v_map * rj_inv).astype(output_dtype)

    logger.info(
        f"Converted {n_sources} point sources to {n_freq} HEALPix maps "
        f"({frequencies[0] / 1e6:.1f}-{frequencies[-1] / 1e6:.1f} MHz)"
    )

    return i_arr, q_arr, u_arr, v_arr
