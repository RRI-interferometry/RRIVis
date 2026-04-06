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
    C_LIGHT,
    K_BOLTZMANN,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
)
from .spectral import apply_faraday_rotation, compute_spectral_scale

logger = logging.getLogger(__name__)


def healpix_map_to_point_arrays(
    temp_map: np.ndarray,
    frequency: float,
    brightness_conversion: str,
    healpix_q_maps: np.ndarray | None = None,
    healpix_u_maps: np.ndarray | None = None,
    healpix_v_maps: np.ndarray | None = None,
    observation_frequencies: np.ndarray | None = None,
    freq_index: int | None = None,
) -> dict[str, np.ndarray]:
    """Convert a HEALPix brightness temperature map to columnar point-source arrays.

    Only positive-temperature pixels are stored (no flux_limit filtering
    here -- apply that in the caller).

    Parameters
    ----------
    temp_map : np.ndarray
        Brightness temperature map in Kelvin.
    frequency : float
        Frequency in Hz for T_b -> Jy conversion.
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

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``"ra_rad"``, ``"dec_rad"``, ``"flux_ref"``, ``"alpha"``,
        ``"stokes_q"``, ``"stokes_u"``, ``"stokes_v"``.
        All arrays have shape ``(N,)`` where N is the number of valid pixels.
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
            method=brightness_conversion,
        )

    valid_idx = np.where(flux_jy > 0)[0]
    if len(valid_idx) == 0:
        logger.warning("No pixels with positive flux in HEALPix map")
        z = np.zeros(0, dtype=np.float64)
        return {
            "ra_rad": z.copy(),
            "dec_rad": z.copy(),
            "flux_ref": z.copy(),
            "alpha": z.copy(),
            "stokes_q": z.copy(),
            "stokes_u": z.copy(),
            "stokes_v": z.copy(),
        }

    theta, phi = hp.pix2ang(nside, valid_idx)
    ra_rad = phi  # phi = RA in radians
    dec_rad = np.pi / 2 - theta  # colatitude -> declination
    flux_ref = flux_jy[valid_idx]
    n = len(valid_idx)
    alpha = np.zeros(n, dtype=np.float64)  # no per-pixel alpha in HEALPix

    # Stokes Q/U/V use Rayleigh-Jeans (linear, sign-preserving) rather than
    # Planck because: (a) Q/U/V values can be negative, which Planck cannot
    # handle; (b) at the small amplitudes typical of polarized emission, the
    # RJ approximation is adequate. Stokes I above uses the user's chosen
    # brightness_conversion (Planck by default) for the non-linear T_b -> Jy
    # conversion of always-positive intensity values.
    rj_factor = (2 * K_BOLTZMANN * frequency**2 / C_LIGHT**2) * omega / 1e-26

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
        "flux_ref": flux_ref,
        "alpha": alpha,
        "stokes_q": stokes_q,
        "stokes_u": stokes_u,
        "stokes_v": stokes_v,
    }


def point_sources_to_healpix_maps(
    ra_rad: np.ndarray,
    dec_rad: np.ndarray,
    flux_ref: np.ndarray,
    alpha: np.ndarray,
    spectral_coeffs: np.ndarray | None,
    stokes_q: np.ndarray | None,
    stokes_u: np.ndarray | None,
    stokes_v: np.ndarray | None,
    rotation_measure: np.ndarray | None,
    nside: int,
    frequencies: np.ndarray,
    ref_frequency: float,
    brightness_conversion: str,
    estimate_memory_fn: Any = None,
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
    flux_ref : np.ndarray
        Reference flux density in Jy, shape ``(N_sources,)``.
    alpha : np.ndarray
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
    ref_frequency : float
        Reference frequency in Hz.
    brightness_conversion : str
        ``"planck"`` or ``"rayleigh-jeans"``.
    estimate_memory_fn : callable, optional
        Function ``(nside, n_freq, dtype, n_stokes) -> dict`` for logging.

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
        empty = np.zeros((n_freq, npix), dtype=np.float32)
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
        mem_info = estimate_memory_fn(nside, n_freq, np.float32, n_stokes)
        logger.info(
            f"Creating {n_freq} HEALPix maps (nside={nside}, "
            f"stokes={'IQUV' if has_pol else 'I'}): "
            f"~{mem_info['total_mb']:.1f} MB"
        )

    ipix = hp.ang2pix(nside, np.pi / 2 - dec_rad, ra_rad)

    i_arr = np.zeros((n_freq, npix), dtype=np.float32)
    q_arr = np.zeros((n_freq, npix), dtype=np.float32) if has_pol else None
    u_arr = np.zeros((n_freq, npix), dtype=np.float32) if has_pol else None
    v_arr = np.zeros((n_freq, npix), dtype=np.float32) if has_pol else None

    for fi, freq in enumerate(frequencies):
        scale = compute_spectral_scale(
            alpha, spectral_coeffs, float(freq), ref_frequency
        )
        flux_f = flux_ref * scale

        flux_map = np.bincount(ipix, weights=flux_f, minlength=npix)

        temp_out = np.zeros(npix, dtype=np.float32)
        occupied = flux_map > 0
        if np.any(occupied):
            temp_out[occupied] = flux_density_to_brightness_temp(
                flux_map[occupied],
                float(freq),
                omega_pixel,
                method=brightness_conversion,
            ).astype(np.float32)
        i_arr[fi] = temp_out

        if has_pol:
            # Jy -> K_RJ via Rayleigh-Jeans (linear, sign-preserving)
            rj_inv = (
                C_LIGHT**2 / (2 * K_BOLTZMANN * float(freq) ** 2 * omega_pixel)
            ) * 1e-26

            q_flux, u_flux = apply_faraday_rotation(
                stokes_q,
                stokes_u,
                rotation_measure,
                float(freq),
                ref_frequency,
                scale,
            )
            q_map = np.bincount(ipix, weights=q_flux, minlength=npix)
            q_arr[fi] = (q_map * rj_inv).astype(np.float32)

            u_map = np.bincount(ipix, weights=u_flux, minlength=npix)
            u_arr[fi] = (u_map * rj_inv).astype(np.float32)

            v_flux = stokes_v * scale
            v_map = np.bincount(ipix, weights=v_flux, minlength=npix)
            v_arr[fi] = (v_map * rj_inv).astype(np.float32)

    logger.info(
        f"Converted {n_sources} point sources to {n_freq} HEALPix maps "
        f"({frequencies[0] / 1e6:.1f}-{frequencies[-1] / 1e6:.1f} MHz)"
    )

    return i_arr, q_arr, u_arr, v_arr
