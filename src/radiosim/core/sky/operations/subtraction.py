"""Bright-source subtraction operations.

Remazeilles-style Gaussian fitting and inpainting helpers live here so the
general operations module stays focused on representation conversion and
small SkyModel transformations.
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
import os
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np

from ..containers import SourceSubtractionStatus
from ..containers.constants import (
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
    pixel_solid_angle,
)

if TYPE_CHECKING:
    from ..containers.model import SkyModel

logger = logging.getLogger(__name__)


# =============================================================================
# Bright-source subtraction (Remazeilles 2015 style)
# =============================================================================

# Auto-detection uses a permissive peak-pixel threshold of
# ``detection_peak_fraction × flux_limit_jy`` so that resolved sources whose
# brightest pixel holds only a fraction of their integrated flux still enter
# the candidate list; the integrated fitted flux is then compared to the full
# threshold before subtraction.  0.2 mirrors the Remazeilles 2015 §3 setup.
_DEFAULT_DETECTION_PEAK_FRACTION: float = 0.2

# Cost-warning threshold: subtract_bright_sources runs one
# scipy.optimize.curve_fit per candidate × per channel.  At ~5 ms / fit
# this caps the warned regime to ≲ 5 s worth of fitting; users hitting
# thousands of fits get a heads-up so they can cap with ``max_sources``.
_SUBTRACT_FIT_COUNT_WARN_THRESHOLD: int = 1000

# Stop criteria for the harmonic-space inpaint loop.  80 iterations is more
# than sufficient for an ℓ_max ≈ 2·nside band-limited fill at the rtol below;
# tighten only if the inpainted pixels visibly diverge.
_DEFAULT_INPAINT_MAX_ITERATIONS: int = 80
_DEFAULT_INPAINT_RTOL: float = 1e-3

# FWHM = 2·sqrt(2·ln 2)·σ ≈ 2.3548·σ for a Gaussian.  Kept exact to four
# decimal places — matches the literature convention used downstream.
_GAUSSIAN_FWHM_TO_SIGMA: float = 2.3548


def _detect_local_maxima_above(
    flux_per_pixel_jy: np.ndarray,
    nside: int,
    flux_limit_jy: float,
) -> np.ndarray:
    """Return the HEALPix pixel indices of local maxima above ``flux_limit_jy``.

    A local maximum is a pixel whose Stokes-I flux is strictly greater than
    each of its eight HEALPix neighbours (as returned by
    :func:`healpy.get_all_neighbours`).  Non-existent neighbours at the poles
    (``-1`` sentinel) are skipped by replacing their flux with ``-inf``.
    """
    candidates = np.flatnonzero(flux_per_pixel_jy > flux_limit_jy)
    if candidates.size == 0:
        return candidates

    neighbours = hp.get_all_neighbours(nside, candidates)  # shape (8, N)
    safe_idx = np.where(neighbours >= 0, neighbours, 0)
    neighbour_flux = flux_per_pixel_jy[safe_idx]
    neighbour_flux = np.where(neighbours >= 0, neighbour_flux, -np.inf)

    centre_flux = flux_per_pixel_jy[candidates]
    is_local_max = np.all(centre_flux[np.newaxis, :] > neighbour_flux, axis=0)
    return candidates[is_local_max]


def _gnomonic_patch_coords(
    center_pix: int,
    patch_pix: np.ndarray,
    nside: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gnomonic (tangent-plane) ``(x, y)`` coordinates in radians for each
    pixel in ``patch_pix`` relative to the tangent point at ``center_pix``.
    """
    theta0, phi0 = hp.pix2ang(nside, center_pix)
    theta, phi = hp.pix2ang(nside, patch_pix)

    lat0 = np.pi / 2.0 - theta0
    lat = np.pi / 2.0 - theta
    dlon = phi - phi0

    cos_c = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(dlon)
    cos_c = np.where(cos_c <= 1e-12, 1e-12, cos_c)
    x = np.cos(lat) * np.sin(dlon) / cos_c
    y = (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon)) / cos_c
    return x, y


def _fit_elliptical_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sigma_init_rad: float,
) -> tuple[np.ndarray, bool]:
    """Fit an elliptical 2-D Gaussian + planar baseline to a tangent-plane patch.

    Model::

        x' =  (x - x0) cos(pa) + (y - y0) sin(pa)
        y' = -(x - x0) sin(pa) + (y - y0) cos(pa)
        f(x, y) = A · exp(-½ · (x'²/σ_M² + y'²/σ_m²))
                  + bx·x + by·y + c

    Matches the elliptical-Gaussian morphology used in Remazeilles,
    Dickinson & Banday (2015) §3.  σ_M / σ_m are the major/minor 1-σ
    widths; ``pa`` is the position angle of the major axis from the +x
    axis (radians, wrapped into ``[-π/2, π/2]`` by symmetry).

    Returns
    -------
    params : np.ndarray
        ``[A, x0, y0, sigma_major, sigma_minor, pa_rad, bx, by, c]``.
    ok : bool
        ``False`` when the fit failed or produced non-physical values.
    """
    from scipy.optimize import curve_fit

    def _model(xy, amp, x0, y0, sigma_M, sigma_m, pa, bx, by, c):
        xx, yy = xy
        cos_pa = np.cos(pa)
        sin_pa = np.sin(pa)
        xr = (xx - x0) * cos_pa + (yy - y0) * sin_pa
        yr = -(xx - x0) * sin_pa + (yy - y0) * cos_pa
        return (
            amp * np.exp(-0.5 * (xr**2 / sigma_M**2 + yr**2 / sigma_m**2))
            + bx * xx
            + by * yy
            + c
        )

    peak_idx = int(np.argmax(z))
    amp_init = float(z[peak_idx] - np.median(z))
    amp_init = amp_init if amp_init > 0 else float(z[peak_idx])
    p0 = [
        amp_init,
        float(x[peak_idx]),
        float(y[peak_idx]),
        float(sigma_init_rad),  # sigma_major
        float(sigma_init_rad),  # sigma_minor
        0.0,  # pa_rad
        0.0,
        0.0,
        float(np.median(z)),
    ]

    lower = [
        0.0,
        float(x.min()),
        float(y.min()),
        sigma_init_rad * 0.2,
        sigma_init_rad * 0.2,
        -np.pi / 2.0,
        -np.inf,
        -np.inf,
        -np.inf,
    ]
    upper = [
        np.inf,
        float(x.max()),
        float(y.max()),
        sigma_init_rad * 5.0,
        sigma_init_rad * 5.0,
        np.pi / 2.0,
        np.inf,
        np.inf,
        np.inf,
    ]

    try:
        popt, _ = curve_fit(
            _model,
            (x, y),
            z,
            p0=p0,
            bounds=(lower, upper),
            maxfev=1000,
        )
    except (RuntimeError, ValueError):
        return np.asarray(p0), False

    amp_fit = popt[0]
    sigma_M_fit = popt[3]
    sigma_m_fit = popt[4]
    sigma_min = min(sigma_M_fit, sigma_m_fit)
    # Reject degenerate fits (negative amp, or either axis collapsed below
    # the resolution floor that would imply an unresolved spike-on-noise).
    if amp_fit <= 0.0 or sigma_min <= sigma_init_rad * 0.25:
        return popt, False

    # Canonicalise so popt[3] is the major axis (larger σ) and popt[5] is
    # measured from that axis.  Without this, the fitter is free to swap
    # axes and shift the angle by π/2, which is fine for the model but
    # confusing for downstream consumers that read sigma_major directly.
    if sigma_m_fit > sigma_M_fit:
        popt[3], popt[4] = sigma_m_fit, sigma_M_fit
        pa_new = popt[5] + np.pi / 2.0
        # Re-wrap into [-π/2, π/2].
        pa_new = (pa_new + np.pi / 2.0) % np.pi - np.pi / 2.0
        popt[5] = pa_new
    return popt, True


def _evaluate_elliptical_gaussian(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    amp, x0, y0, sigma_M, sigma_m, pa, _bx, _by, _c = params
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    xr = (x - x0) * cos_pa + (y - y0) * sin_pa
    yr = -(x - x0) * sin_pa + (y - y0) * cos_pa
    return amp * np.exp(-0.5 * (xr**2 / sigma_M**2 + yr**2 / sigma_m**2))


def _fit_multifreq_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    z_per_channel: np.ndarray,
    sigma_init_rad: float,
) -> tuple[dict[str, np.ndarray | float], bool]:
    """Joint multi-frequency elliptical-Gaussian + planar-baseline fit.

    Geometry parameters ``(x0, y0, sigma_major, sigma_minor, pa)`` are
    shared across every channel; amplitude ``A_i`` and planar baseline
    ``(bx_i, by_i, c_i)`` are per-channel. For steep-spectrum sources
    σ and (x0, y0) move only weakly with frequency, so a single
    multi-frequency fit is more stable in low-SNR channels and ~``N_freq``×
    cheaper than running one per-channel ``curve_fit`` per candidate.

    Parameters
    ----------
    x, y : np.ndarray, shape (N,)
        Tangent-plane coordinates of every patch pixel (radians).
    z_per_channel : np.ndarray, shape (N_freq, N)
        Flux on the same patch pixels at each channel.
    sigma_init_rad : float
        Initial σ guess (single-pixel resolution scale, radians).

    Returns
    -------
    fit : dict
        Keys: ``"x0"``, ``"y0"``, ``"sigma_major"``, ``"sigma_minor"``,
        ``"pa"`` (scalars); ``"amplitudes"`` (shape ``(N_freq,)``);
        ``"baselines"`` (shape ``(N_freq, 3)``: ``[bx, by, c]`` per row).
        Always returned — populated with seed values if ``ok`` is False.
    ok : bool
        ``False`` when the joint fit failed or produced non-physical
        geometry.
    """
    from scipy.optimize import least_squares

    z_per_channel = np.asarray(z_per_channel, dtype=np.float64)
    if z_per_channel.ndim != 2:
        raise ValueError(
            "_fit_multifreq_gaussian: z_per_channel must be (N_freq, N), "
            f"got shape {z_per_channel.shape}."
        )
    n_freq, n_pix = z_per_channel.shape
    if n_pix != x.size:
        raise ValueError(
            "_fit_multifreq_gaussian: patch pixel count mismatch: "
            f"x={x.size}, z_per_channel rows={n_pix}."
        )

    # Initialise geometry from the brightest channel — Remazeilles 2015 §3
    # uses the same one-channel seed in their bootstrap.
    bright_ch = int(np.argmax(np.ptp(z_per_channel, axis=1)))
    seed_z = z_per_channel[bright_ch]
    peak_idx = int(np.argmax(seed_z))
    medians = np.median(z_per_channel, axis=1)
    amps_init = np.maximum(z_per_channel.max(axis=1) - medians, np.abs(medians) * 1e-3)
    x0_init = float(x[peak_idx])
    y0_init = float(y[peak_idx])

    # Pack: [x0, y0, sigma_M, sigma_m, pa, A_0, ..., A_{n-1},
    #        bx_0, ..., bx_{n-1}, by_0, ..., by_{n-1}, c_0, ..., c_{n-1}]
    # Named slices defined once so every pack/unpack site shares the same
    # offsets (a mismatched ``5 + k*n_freq`` was a silent-wrong-fit footgun).
    n_params = 5 + 4 * n_freq
    AMP = slice(5, 5 + n_freq)
    BX = slice(5 + n_freq, 5 + 2 * n_freq)
    BY = slice(5 + 2 * n_freq, 5 + 3 * n_freq)
    C = slice(5 + 3 * n_freq, 5 + 4 * n_freq)

    def _baseline_rows(params: np.ndarray) -> np.ndarray:
        """``(n_freq, 3)`` array of ``[bx, by, c]`` per channel."""
        return np.column_stack([params[BX], params[BY], params[C]])

    p0 = np.zeros(n_params, dtype=np.float64)
    p0[0] = x0_init
    p0[1] = y0_init
    p0[2] = sigma_init_rad
    p0[3] = sigma_init_rad
    p0[4] = 0.0
    p0[AMP] = amps_init
    # bx, by start at 0; c starts at the per-channel median.
    p0[C] = medians

    lower = np.full(n_params, -np.inf)
    upper = np.full(n_params, np.inf)
    lower[0] = float(x.min())
    upper[0] = float(x.max())
    lower[1] = float(y.min())
    upper[1] = float(y.max())
    lower[2] = sigma_init_rad * 0.2
    upper[2] = sigma_init_rad * 5.0
    lower[3] = sigma_init_rad * 0.2
    upper[3] = sigma_init_rad * 5.0
    lower[4] = -np.pi / 2.0
    upper[4] = np.pi / 2.0
    lower[AMP] = 0.0  # amplitudes non-negative

    # Cache geometry-dependent terms once per evaluation; per-channel work is O(N).
    def _residuals(params: np.ndarray) -> np.ndarray:
        x0, y0, sigma_M, sigma_m, pa = params[:5]
        amps = params[AMP]
        bxs = params[BX]
        bys = params[BY]
        cs = params[C]
        cos_pa = np.cos(pa)
        sin_pa = np.sin(pa)
        xr = (x - x0) * cos_pa + (y - y0) * sin_pa
        yr = -(x - x0) * sin_pa + (y - y0) * cos_pa
        gauss_shape = np.exp(-0.5 * (xr**2 / sigma_M**2 + yr**2 / sigma_m**2))
        # Vectorise across channels: model_per_channel = amps[:, None] * gauss + ...
        model = (
            amps[:, None] * gauss_shape[None, :]
            + bxs[:, None] * x[None, :]
            + bys[:, None] * y[None, :]
            + cs[:, None]
        )
        return (model - z_per_channel).ravel()

    seed_dict: dict[str, np.ndarray | float] = {
        "x0": float(p0[0]),
        "y0": float(p0[1]),
        "sigma_major": float(p0[2]),
        "sigma_minor": float(p0[3]),
        "pa": float(p0[4]),
        "amplitudes": p0[AMP].copy(),
        "baselines": _baseline_rows(p0),
    }

    try:
        result = least_squares(
            _residuals,
            p0,
            bounds=(lower, upper),
            max_nfev=2000,
        )
    except (RuntimeError, ValueError):
        return seed_dict, False

    if not result.success:
        return seed_dict, False

    popt = result.x
    sigma_M_fit = float(popt[2])
    sigma_m_fit = float(popt[3])
    pa_fit = float(popt[4])
    amps_fit = popt[AMP]

    # Reject geometry collapses below the resolution floor (matches the
    # single-channel fit's guard at sigma_min <= sigma_init * 0.25).
    if min(sigma_M_fit, sigma_m_fit) <= sigma_init_rad * 0.25:
        return seed_dict, False
    # At least one channel must have a positive integrated flux for the
    # fit to be useful.
    if not np.any(amps_fit > 0):
        return seed_dict, False

    # Canonicalise so sigma_major is the larger axis (matches single-channel
    # fitter convention so downstream consumers can read sigma_major directly).
    if sigma_m_fit > sigma_M_fit:
        sigma_M_fit, sigma_m_fit = sigma_m_fit, sigma_M_fit
        pa_fit = (pa_fit + np.pi / 2.0 + np.pi / 2.0) % np.pi - np.pi / 2.0

    return {
        "x0": float(popt[0]),
        "y0": float(popt[1]),
        "sigma_major": sigma_M_fit,
        "sigma_minor": sigma_m_fit,
        "pa": pa_fit,
        "amplitudes": amps_fit.copy(),
        "baselines": _baseline_rows(popt),
    }, True


def _multifreq_fit_to_channel_params(
    fit: dict[str, np.ndarray | float], channel_index: int
) -> np.ndarray:
    """Assemble the 9-vector channel-specific param array used by
    :func:`_evaluate_elliptical_gaussian` from a multi-freq fit dict.
    """
    amps = np.asarray(fit["amplitudes"])
    baselines = np.asarray(fit["baselines"])
    return np.asarray(
        [
            float(amps[channel_index]),
            float(fit["x0"]),
            float(fit["y0"]),
            float(fit["sigma_major"]),
            float(fit["sigma_minor"]),
            float(fit["pa"]),
            float(baselines[channel_index, 0]),
            float(baselines[channel_index, 1]),
            float(baselines[channel_index, 2]),
        ]
    )


def _inpaint_by_alm(
    maps: np.ndarray,
    nside: int,
    mask_pixels: np.ndarray,
    max_iterations: int,
    rtol: float,
) -> np.ndarray:
    """Iteratively fill ``mask_pixels`` by harmonic-space smoothing.

    At each iteration the current map is round-tripped through
    :func:`healpy.map2alm` and :func:`alm2map` — which implicitly
    band-limits at ``ℓ_max = 2·nside`` — and the *masked* pixels are
    replaced by the smoothed values.  Unmasked pixels stay untouched.
    """
    if mask_pixels.size == 0:
        return maps

    working = maps.astype(np.float64, copy=True)
    for fi in range(working.shape[0]):
        chan = working[fi]
        unmasked = np.ones_like(chan, dtype=bool)
        unmasked[mask_pixels] = False
        baseline = float(np.mean(chan[unmasked])) if np.any(unmasked) else 0.0
        chan[mask_pixels] = baseline

        ref_scale = max(abs(chan).max(), 1e-30)
        for _ in range(max_iterations):
            alm = hp.map2alm(chan, lmax=None, use_weights=False, iter=1)
            smoothed = hp.alm2map(alm, nside=nside)
            delta = np.abs(smoothed[mask_pixels] - chan[mask_pixels]).max()
            chan[mask_pixels] = smoothed[mask_pixels]
            if delta <= rtol * ref_scale:
                break
        working[fi] = chan

    return working.astype(maps.dtype, copy=False)


def _select_subtraction_candidates(
    sky: SkyModel,
    *,
    frequency_hz: float,
    flux_limit_jy: float,
    catalog: SkyModel | None,
    detection_peak_fraction: float,
    max_sources: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the HEALPix pixel indices to attempt subtraction at.

    Two modes:

    * ``catalog`` supplied: scale each catalog source's flux to
      ``frequency_hz`` via its declared spectral index, keep those above
      ``flux_limit_jy``, and project to pixel indices.
    * ``catalog is None``: convert the reference-frequency Stokes I map to
      Jy/pixel and call :func:`_detect_local_maxima_above` with a permissive
      peak threshold of ``detection_peak_fraction × flux_limit_jy``.

    Returns ``(candidate_pix, flux_per_pixel_jy)``.  ``flux_per_pixel_jy``
    is the reference-frequency Jy map and is used by ``max_sources`` to
    rank candidates.
    """
    nside = sky.healpix.nside
    pixel_area_sr = pixel_solid_angle(nside)

    idx = sky.healpix.resolve_frequency_index(frequency_hz)
    reference_map_k = np.asarray(sky.healpix.maps[idx], dtype=np.float64)
    use_rj = sky.brightness_conversion.value == "rayleigh-jeans" or np.any(
        reference_map_k <= 0.0
    )
    conv_method = "rayleigh-jeans" if use_rj else "planck"
    flux_per_pixel_jy = brightness_temp_to_flux_density(
        reference_map_k,
        frequency=float(frequency_hz),
        solid_angle=pixel_area_sr,
        method=conv_method,
    )

    if catalog is not None and catalog.point is not None and not catalog.point.is_empty:
        cat_ref = (
            catalog.point.ref_freq.astype(np.float64)
            if catalog.point.ref_freq.size
            else np.zeros(catalog.point.n_sources, dtype=np.float64)
        )
        alpha = catalog.point.spectral_index.astype(np.float64)
        scaled_flux = catalog.point.flux.astype(np.float64).copy()
        valid_ref = cat_ref > 0.0
        if np.any(valid_ref):
            scaled_flux[valid_ref] *= (
                float(frequency_hz) / cat_ref[valid_ref]
            ) ** alpha[valid_ref]
        keep = scaled_flux >= flux_limit_jy
        if not np.any(keep):
            candidate_pix = np.asarray([], dtype=np.int64)
        else:
            theta = np.pi / 2.0 - catalog.point.dec_rad[keep].astype(np.float64)
            phi = catalog.point.ra_rad[keep].astype(np.float64)
            candidate_pix = np.unique(hp.ang2pix(nside, theta, phi))
    else:
        detection_threshold_jy = max(
            flux_limit_jy * float(detection_peak_fraction), 0.0
        )
        candidate_pix = _detect_local_maxima_above(
            flux_per_pixel_jy, nside, detection_threshold_jy
        )

    if max_sources is not None and candidate_pix.size > max_sources:
        order = np.argsort(-flux_per_pixel_jy[candidate_pix])
        candidate_pix = candidate_pix[order[:max_sources]]

    return candidate_pix, flux_per_pixel_jy


def _fit_one_candidate_multifreq(
    center: int,
    flux_per_channel_jy: np.ndarray,
    *,
    nside: int,
    patch_radius: float,
    sigma_init: float,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | float], bool]
    | None
):
    """One candidate's patch + multi-frequency joint Gaussian fit.

    Returns ``(patch_indices, px, py, fit_dict, ok)`` or ``None`` when the
    patch is too small to fit. Pure read of ``flux_per_channel_jy``, so
    multiple candidates can be fit in parallel safely.
    """
    patch = hp.query_disc(
        nside, hp.pix2vec(nside, center), patch_radius, inclusive=True
    )
    if patch.size < 8:
        return None
    px, py = _gnomonic_patch_coords(int(center), patch, nside)
    pz_per_channel = flux_per_channel_jy[:, patch]
    fit, ok = _fit_multifreq_gaussian(px, py, pz_per_channel, sigma_init_rad=sigma_init)
    return patch, px, py, fit, ok


def _apply_multifreq_fit_to_buffers(
    candidate_fit: (
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | float], bool]
        | None
    ),
    *,
    flux_per_channel_jy: np.ndarray,
    pixel_area_sr: float,
    inpaint_mask_sigma: float,
    flux_limit_jy: float,
    catalog_present: bool,
    inpaint_mask: set[int],
) -> tuple[int, int]:
    """Apply one multi-freq fit to every channel buffer.

    Subtracts the per-channel Gaussian + planar baseline using the
    shared geometry and the channel-specific amplitude. Updates
    ``inpaint_mask`` once (geometry is shared across channels).

    Returns ``(n_channel_ok, n_channel_failed)`` summed across channels.
    The integrated-flux gate is applied per channel using that channel's
    fitted amplitude — a candidate may pass the gate at one frequency
    and be skipped at another.
    """
    if candidate_fit is None:
        return 0, 0
    patch, px, py, fit, ok = candidate_fit
    if not ok:
        return 0, flux_per_channel_jy.shape[0]

    sigma_M_fit = float(fit["sigma_major"])
    sigma_m_fit = float(fit["sigma_minor"])
    amps = np.asarray(fit["amplitudes"])
    n_freq = flux_per_channel_jy.shape[0]
    n_ok = 0
    n_failed = 0

    integrated_flux_per_channel = (
        amps * 2.0 * np.pi * sigma_M_fit * sigma_m_fit / pixel_area_sr
    )

    for ch in range(n_freq):
        if not catalog_present and integrated_flux_per_channel[ch] < flux_limit_jy:
            n_failed += 1
            continue
        params = _multifreq_fit_to_channel_params(fit, ch)
        model_vals = _evaluate_elliptical_gaussian(params, px, py)
        flux_per_channel_jy[ch, patch] -= model_vals
        n_ok += 1

    # Inpaint mask uses shared geometry — same pixels for every channel.
    x0_fit = float(fit["x0"])
    y0_fit = float(fit["y0"])
    pa_fit = float(fit["pa"])
    cos_pa = np.cos(pa_fit)
    sin_pa = np.sin(pa_fit)
    xr = (px - x0_fit) * cos_pa + (py - y0_fit) * sin_pa
    yr = -(px - x0_fit) * sin_pa + (py - y0_fit) * cos_pa
    ellipse_r2 = (xr / sigma_M_fit) ** 2 + (yr / sigma_m_fit) ** 2
    core_mask = ellipse_r2 <= inpaint_mask_sigma**2
    for p in patch[core_mask]:
        inpaint_mask.add(int(p))

    return n_ok, n_failed


def _fit_and_subtract_per_channel(
    sky: SkyModel,
    candidate_pix: np.ndarray,
    *,
    patch_radius: float,
    sigma_init: float,
    inpaint_mask_sigma: float,
    flux_limit_jy: float,
    catalog_present: bool,
    parallel: bool = False,
) -> tuple[np.ndarray, set[int], int, int]:
    """Multi-frequency joint Gaussian fit per candidate, then per-channel subtract.

    For each candidate, geometry parameters ``(x0, y0, sigma_M, sigma_m, pa)``
    are fit jointly across every frequency channel; amplitudes and planar
    baselines are per-channel. For steep-spectrum sources σ and the
    centroid move only weakly with frequency, so a single multi-freq fit
    is more stable in low-SNR channels and ~``N_freq``× cheaper than the
    older one-curve-fit-per-(candidate, channel) loop.

    Subtractions are applied per channel using the channel-specific
    amplitude. The integrated-flux gate (auto-detect mode only) is also
    per channel — a candidate that drops below ``flux_limit_jy`` at one
    frequency stays untouched there but can still be subtracted at the
    frequencies where it remains bright.

    When ``parallel=True``, the per-candidate joint fits run on a
    :class:`ThreadPoolExecutor` (scipy MINPACK / pure-numpy model
    release the GIL). Subtractions are applied serially in the input
    ``candidate_pix`` order regardless of ``parallel``, so the two modes
    produce bit-identical results — ``parallel`` is purely a wall-clock
    speedup, never a semantic switch.

    Returns ``(new_maps_K, inpaint_mask_pixels, n_fits_ok, n_fits_failed)``.
    ``n_fits_ok`` / ``n_fits_failed`` count per-(candidate, channel)
    application outcomes so the totals are comparable to the old
    per-channel implementation.
    """
    nside = sky.healpix.nside
    pixel_area_sr = pixel_solid_angle(nside)
    new_maps = np.array(sky.healpix.maps, dtype=np.float64, copy=True)
    n_freq = new_maps.shape[0]
    cube_freqs = sky.healpix.frequencies
    inpaint_mask: set[int] = set()
    n_fits_ok = 0
    n_fits_failed = 0

    # Pre-convert the entire cube K → Jy so the joint fit sees one
    # consistent flux array. Per-channel rj/planck choice is made once
    # per channel based on whether that channel has any non-positive
    # pixels (matches the original per-channel logic).
    flux_per_channel_jy = np.empty((n_freq, new_maps.shape[1]), dtype=np.float64)
    rj_per_channel = np.zeros(n_freq, dtype=bool)
    for fi in range(n_freq):
        rj_per_channel[fi] = sky.brightness_conversion.value == "rayleigh-jeans" or (
            bool(np.any(new_maps[fi] <= 0.0))
        )
        method_ch = "rayleigh-jeans" if rj_per_channel[fi] else "planck"
        flux_per_channel_jy[fi] = brightness_temp_to_flux_density(
            new_maps[fi].copy(),
            frequency=float(cube_freqs[fi]),
            solid_angle=pixel_area_sr,
            method=method_ch,
        )

    if parallel:
        # Parallel fit phase: every candidate reads the same buffer.
        n_threads = min(os.cpu_count() or 4, 8)
        fit_call = functools.partial(
            _fit_one_candidate_multifreq,
            flux_per_channel_jy=flux_per_channel_jy,
            nside=nside,
            patch_radius=patch_radius,
            sigma_init=sigma_init,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as pool:
            fit_results = list(pool.map(fit_call, candidate_pix.tolist()))
    else:
        fit_results = [
            _fit_one_candidate_multifreq(
                int(center),
                flux_per_channel_jy,
                nside=nside,
                patch_radius=patch_radius,
                sigma_init=sigma_init,
            )
            for center in candidate_pix
        ]

    # Apply subtractions in deterministic candidate order. Each fit
    # touches one patch across every channel.
    for fit in fit_results:
        ok_delta, failed_delta = _apply_multifreq_fit_to_buffers(
            fit,
            flux_per_channel_jy=flux_per_channel_jy,
            pixel_area_sr=pixel_area_sr,
            inpaint_mask_sigma=inpaint_mask_sigma,
            flux_limit_jy=flux_limit_jy,
            catalog_present=catalog_present,
            inpaint_mask=inpaint_mask,
        )
        n_fits_ok += ok_delta
        n_fits_failed += failed_delta

    # Convert each modified channel back to brightness temperature.
    for fi in range(n_freq):
        flux_positive = flux_per_channel_jy[fi].copy()
        flux_positive[flux_positive <= 0] = np.finfo(np.float64).tiny
        method_back = "rayleigh-jeans" if rj_per_channel[fi] else "planck"
        new_maps[fi] = flux_density_to_brightness_temp(
            flux_positive,
            frequency=float(cube_freqs[fi]),
            solid_angle=pixel_area_sr,
            method=method_back,
        )

    return new_maps, inpaint_mask, n_fits_ok, n_fits_failed


def subtract_bright_sources(
    sky: SkyModel,
    *,
    flux_limit_jy: float,
    frequency_hz: float,
    catalog: SkyModel | None = None,
    patch_radius_rad: float | None = None,
    inpaint_mask_sigma: float = 3.0,
    inpaint_max_iterations: int = _DEFAULT_INPAINT_MAX_ITERATIONS,
    inpaint_rtol: float = _DEFAULT_INPAINT_RTOL,
    max_sources: int | None = None,
    detection_peak_fraction: float = _DEFAULT_DETECTION_PEAK_FRACTION,
    parallel: bool = False,
) -> SkyModel:
    """Remove bright point sources from a HEALPix diffuse map.

    Implements the Remazeilles, Dickinson & Banday (2015) methodology for
    preparing a source-subtracted all-sky template that can be combined
    with an independent point-source catalog without double-counting:

    1. **Detection.**  Either use the positions in a supplied ``catalog``
       (flux-scaled to ``frequency_hz`` and filtered at ``flux_limit_jy``),
       or auto-detect local maxima in the map above ``flux_limit_jy``.
    2. **Fit.**  For each candidate, project a small tangent-plane patch
       and fit a 2-D *elliptical* Gaussian (independent major/minor σ
       and a position angle) + planar baseline, matching the morphology
       fit by Remazeilles 2015.
    3. **Subtract.**  Evaluate the fitted Gaussian on the patch and
       subtract from the working Stokes-I cube (per frequency).
    4. **Inpaint.**  Pixels within ``inpaint_mask_sigma × σ_fit`` of each
       source centre are re-estimated via iterative harmonic-space
       round-trip smoothing (healpy analogue of the minimum-curvature spline
       used in Remazeilles 2015).
    5. **Provenance.**  Advertise ``source_subtraction=ABOVE_THRESHOLD``.

    Parameters
    ----------
    sky
        Input model.  Must carry a dense HEALPix payload.
    flux_limit_jy
        Flux-density threshold (Jy) above which sources are removed.
    frequency_hz
        Frequency at which the threshold is interpreted (used to convert
        between Kelvin and Jy and to scale a supplied catalog's fluxes).
    catalog
        Optional point-source model providing candidate positions.  When
        given, local-maximum detection is skipped and only catalog sources
        whose flux at ``frequency_hz`` (power-law scaled) is above the
        threshold are subtracted.
    patch_radius_rad
        Radius of the tangent-plane fitting patch in radians.  Defaults to
        ``3 × hp.nside2resol(nside)`` — three native pixels.
    inpaint_mask_sigma
        Pixels within ``inpaint_mask_sigma × σ_fit`` of a source centre are
        inpainted.
    inpaint_max_iterations, inpaint_rtol
        Stop criteria for the harmonic-space inpaint loop.
    max_sources
        Cap on the number of sources processed.  The brightest are kept.
    detection_peak_fraction
        Auto-detection uses a permissive *peak-pixel* threshold of
        ``detection_peak_fraction × flux_limit_jy`` to find candidates; the
        integrated fitted flux is then compared to ``flux_limit_jy`` before
        subtraction.  This captures resolved sources whose peak pixel holds
        only a fraction of their total flux.  Default 0.2.

    Returns
    -------
    SkyModel
        A new model with the HEALPix cube updated and provenance advertising
        the subtraction.

    Raises
    ------
    ValueError
        If the input does not carry a dense HEALPix payload.

    Notes
    -----
    **Frequency dependence of the candidate set.**  Detection — and supplied-
    catalog filtering — is performed at ``frequency_hz`` only.  Once the
    candidate list is fixed at that single reference frequency, the
    Gaussian-fit-and-subtract loop is repeated independently at every
    frequency in the cube.  Two consequences follow:

    * A source above ``flux_limit_jy`` at ``frequency_hz`` is subtracted
      at *all* output frequencies, even if its scaled flux drops below the
      threshold elsewhere in the band.
    * A source below the threshold at ``frequency_hz`` but above it at
      other frequencies is *missed entirely* — it never enters the
      candidate list and is never subtracted.

    This is the same simplification used in Remazeilles, Dickinson &
    Banday (2015) §3 and is acceptable for the steep, smoothly varying
    sources that dominate diffuse-template residuals.  Spectrally
    unusual sources (GPS, IPS, transients, sources crossing the
    threshold within the band) require either per-frequency repetition
    of this routine or a multi-frequency catalog evaluated at each
    output frequency.

    **Cost.** This routine runs one joint multi-frequency
    ``scipy.optimize.least_squares`` call per candidate (geometry shared
    across channels, amplitudes per-channel) — i.e. ``O(N_candidates)``
    nonlinear fits, not the ``O(N_candidates × N_freq)`` cost of the
    older per-channel implementation.  Cap the candidate list with
    ``max_sources=...`` or pre-filter with a higher ``flux_limit_jy``
    when many candidates are expected; a logger warning is emitted when
    the candidate count exceeds 1000.

    **Parallelism (``parallel``).** ``parallel=True`` runs the
    per-candidate joint fits on a :class:`ThreadPoolExecutor`
    (``min(os.cpu_count(), 8)`` threads); scipy MINPACK and the pure-numpy
    model evaluator both release the GIL.  Subtractions are applied
    serially in the input candidate order regardless of ``parallel``, so
    the two modes produce **bit-identical results** — ``parallel`` is
    purely a wall-clock speedup.
    """
    if sky.healpix is None:
        raise ValueError(
            "subtract_bright_sources requires a HEALPix payload; got a "
            "point-only model.  Use materialize_healpix_model(...) first."
        )
    sky.healpix.require_dense("subtract_bright_sources")

    # The fitting geometry and the alm-based inpainting (map2alm) assume RING
    # ordering. Normalize NESTED inputs to RING for the duration, then restore
    # the caller's ordering on the result.
    restore_ordering = sky.healpix.ordering
    if sky.healpix.is_nested:
        sky = sky.replace(healpix=sky.healpix.reordered("ring"))

    nside = sky.healpix.nside
    sigma_init = hp.nside2resol(nside) / _GAUSSIAN_FWHM_TO_SIGMA
    patch_radius = (
        float(patch_radius_rad)
        if patch_radius_rad is not None
        else 3.0 * hp.nside2resol(nside)
    )

    candidate_pix, _ = _select_subtraction_candidates(
        sky,
        frequency_hz=frequency_hz,
        flux_limit_jy=flux_limit_jy,
        catalog=catalog,
        detection_peak_fraction=detection_peak_fraction,
        max_sources=max_sources,
    )

    n_freq_total = int(sky.healpix.frequencies.size)
    if int(candidate_pix.size) > _SUBTRACT_FIT_COUNT_WARN_THRESHOLD:
        logger.warning(
            "subtract_bright_sources: about to run %d joint multi-frequency "
            "scipy.optimize.least_squares fits (one per candidate; geometry "
            "is shared across %d channel(s), amplitudes are per-channel). "
            "Pass max_sources=... to cap the candidate list, or filter by "
            "flux_limit_jy first.",
            int(candidate_pix.size),
            n_freq_total,
        )

    if candidate_pix.size == 0:
        new_prov = sky.provenance.replace(
            source_subtraction=SourceSubtractionStatus.ABOVE_THRESHOLD,
            source_subtraction_threshold_jy=float(flux_limit_jy),
            source_subtraction_freq_hz=float(frequency_hz),
            source_subtraction_method="gaussian_fit_inpaint",
        )
        return sky.replace(provenance=new_prov)

    new_maps, inpaint_mask, n_fits_ok, n_fits_failed = _fit_and_subtract_per_channel(
        sky,
        candidate_pix,
        patch_radius=patch_radius,
        sigma_init=sigma_init,
        inpaint_mask_sigma=inpaint_mask_sigma,
        flux_limit_jy=flux_limit_jy,
        catalog_present=catalog is not None,
        parallel=parallel,
    )

    if inpaint_mask:
        mask_pixels = np.asarray(sorted(inpaint_mask), dtype=np.int64)
        new_maps = _inpaint_by_alm(
            new_maps,
            nside=nside,
            mask_pixels=mask_pixels,
            max_iterations=inpaint_max_iterations,
            rtol=inpaint_rtol,
        )

    new_healpix = sky.healpix.replace(
        maps=new_maps.astype(sky.healpix.maps.dtype, copy=False),
    )
    if restore_ordering != new_healpix.ordering:
        new_healpix = new_healpix.reordered(restore_ordering)

    old_prov = sky.provenance
    new_monopole = None
    if old_prov.monopole_k is not None:
        new_monopole = float(np.mean(new_maps[0]))
    new_prov = old_prov.replace(
        monopole_k=new_monopole,
        source_subtraction=SourceSubtractionStatus.ABOVE_THRESHOLD,
        source_subtraction_threshold_jy=float(flux_limit_jy),
        source_subtraction_freq_hz=float(frequency_hz),
        source_subtraction_method="gaussian_fit_inpaint",
        notes=((old_prov.notes + " + ") if old_prov.notes else "")
        + f"subtracted>{flux_limit_jy:g}Jy@{frequency_hz / 1e6:.1f}MHz "
        + f"(ok={n_fits_ok}/{n_fits_ok + n_fits_failed})",
    )
    return sky.replace(healpix=new_healpix, provenance=new_prov)
