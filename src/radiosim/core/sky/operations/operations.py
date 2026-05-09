"""Functional sky-model operations.

These helpers keep mutation-free transformations and memory-management
operations outside ``SkyModel`` itself.
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
import os
import tempfile
import warnings
from typing import TYPE_CHECKING, Any

import healpy as hp
import numpy as np

from radiosim.utils.frequency import parse_frequency_config

from ..containers.constants import (
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
)
from ..containers.data import (
    HealpixData,
    MonopoleConvention,
    PointSourceData,
    SourceSubtractionStatus,
)
from ..containers.spectral import per_source_reference_frequencies
from .convert import healpix_map_to_point_arrays, point_sources_to_healpix_maps

if TYPE_CHECKING:
    from ..containers.model import SkyModel

logger = logging.getLogger(__name__)


def materialize_healpix_model(
    sky: SkyModel,
    *,
    nside: int,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    ref_frequency: float | None = None,
    memmap_path: str | None = None,
    clear_other: bool = False,
) -> SkyModel:
    """Materialize a HEALPix payload from a point-source payload.

    By default the result is a hybrid model carrying both the original
    point payload and the new HEALPix payload. Pass ``clear_other=True``
    to drop the source point payload (pure point→HEALPix conversion).

    Stokes I uses ``sky.brightness_conversion``.  Stokes Q/U/V use
    ``sky.polarization_brightness_conversion`` (defaults to Rayleigh-Jeans
    because polarized brightness can be negative; the Planck inverse is
    undefined for non-positive arguments).  Set the policy on the model
    itself via ``sky.replace(polarization_brightness_conversion="planck")``
    when you need Stokes-I-style non-linear conversion for polarised maps.
    """
    if sky.point is None:
        raise ValueError(
            "No point sources available for conversion. "
            "Load a point-source model first, for example with "
            "radiosim.core.sky.loaders.load_gleam()."
        )

    if frequencies is not None and obs_frequency_config is not None:
        raise ValueError(
            "Provide either 'frequencies' or 'obs_frequency_config', not both."
        )
    if frequencies is None and obs_frequency_config is not None:
        frequencies = parse_frequency_config(obs_frequency_config)
    if frequencies is None:
        raise ValueError(
            "Either 'frequencies' (np.ndarray) or 'obs_frequency_config' "
            "(dict) is required."
        )

    effective_ref_freq = per_source_reference_frequencies(
        sky.point,
        model_reference_frequency=sky.reference_frequency,
        fallback=ref_frequency,
    )
    if not np.any(effective_ref_freq > 0):
        raise ValueError(
            "ref_frequency must be provided when this SkyModel has no "
            "per-source ref_freq values and no reference_frequency. "
            "Set it via with_reference_frequency() or pass ref_frequency "
            "explicitly."
        )

    spectrum = sky.point.spectrum
    pol_method = sky.polarization_brightness_conversion.value
    i_maps, q_maps, u_maps, v_maps, collision_stats = point_sources_to_healpix_maps(
        ra_rad=sky.point.ra_rad,
        dec_rad=sky.point.dec_rad,
        flux=sky.point.flux,
        spectral_index=sky.point.spectral_index,
        spectral_coeffs=sky.point.spectral_coeffs,
        stokes_q=sky.point.stokes_q,
        stokes_u=sky.point.stokes_u,
        stokes_v=sky.point.stokes_v,
        rotation_measure=sky.point.rotation_measure,
        nside=nside,
        frequencies=frequencies,
        ref_frequency=effective_ref_freq,
        brightness_conversion=sky.brightness_conversion,
        coordinate_frame="icrs",
        output_dtype=sky._healpix_dtype(),
        memmap_path=memmap_path,
        per_channel_flux=spectrum.flux if spectrum is not None else None,
        per_channel_stokes_q=spectrum.stokes_q if spectrum is not None else None,
        per_channel_stokes_u=spectrum.stokes_u if spectrum is not None else None,
        per_channel_stokes_v=spectrum.stokes_v if spectrum is not None else None,
        channel_frequencies=spectrum.frequencies if spectrum is not None else None,
        polarization_brightness_conversion=pol_method,
    )

    new_healpix = HealpixData(
        maps=i_maps,
        nside=nside,
        frequencies=frequencies,
        coordinate_frame="icrs",
        q_maps=q_maps,
        u_maps=u_maps,
        v_maps=v_maps,
        i_brightness_conversion=sky.brightness_conversion.value,
        q_brightness_conversion=pol_method,
        u_brightness_conversion=pol_method,
        v_brightness_conversion=pol_method,
    )

    # Record pixel collisions in provenance.notes so a downstream consumer
    # can detect — programmatically — that per-source identities were
    # merged into individual pixels (and therefore that per-source spectral
    # indices, source IDs, etc. cannot be recovered).
    new_prov = sky.provenance
    n_merged = collision_stats.get("n_merged", 0)
    n_sources = collision_stats.get("n_sources", 0)
    if n_merged > 0 and n_sources > 0:
        pct = n_merged / n_sources
        note = f"pixel_collisions={n_merged}/{n_sources} ({pct:.2%}) at nside={nside}"
        merged_notes = (
            f"{sky.provenance.notes}; {note}" if sky.provenance.notes else note
        )
        new_prov = sky.provenance.replace(notes=merged_notes)

    if clear_other:
        return sky.replace(point=None, healpix=new_healpix, provenance=new_prov)
    return sky.replace(healpix=new_healpix, provenance=new_prov)


def materialize_point_sources_model(
    sky: SkyModel,
    frequency: float | None = None,
    flux_limit: float = 0.0,
    *,
    lossy: bool = False,
    clear_other: bool = False,
) -> SkyModel:
    """Materialize a point-source payload from a HEALPix payload.

    By default the result is a hybrid model carrying both the original
    HEALPix payload and the new point payload. Pass ``clear_other=True``
    to drop the source HEALPix payload (pure HEALPix→point conversion).

    Stokes I uses ``sky.brightness_conversion``.  Stokes Q/U/V use
    ``sky.polarization_brightness_conversion`` (defaults to Rayleigh-Jeans
    because polarized brightness can be negative).  Set the policy on the
    model itself via
    ``sky.replace(polarization_brightness_conversion="planck")`` when you
    need Stokes-I-style non-linear conversion for polarised maps.
    """
    if sky.point is not None:
        return sky

    if sky.healpix is None:
        raise ValueError("No HEALPix payload available for conversion.")
    if not lossy:
        raise ValueError(
            "HEALPix-to-point-source conversion is lossy. "
            "Call materialize_point_sources_model(..., lossy=True) to opt in."
        )

    freq = frequency or sky.reference_frequency
    healpix = sky.healpix.to_dense() if sky.healpix.is_sparse else sky.healpix
    n_freq = len(healpix.frequencies)
    resol_arcmin = float(hp.nside2resol(healpix.nside, arcmin=True))
    warnings.warn(
        f"HEALPix-to-point-source conversion is lossy: positions are "
        f"quantized to pixel centers (nside={healpix.nside}, "
        f"~{resol_arcmin:.1f}' resolution) and spectral indices are "
        f"fit from {n_freq} channels. Use 'healpix_map' mode for "
        f"full-fidelity diffuse emission.",
        stacklevel=2,
    )

    resolve_freq = freq or float(healpix.frequencies[0])
    if resolve_freq is None:
        raise ValueError(
            "frequency is required for HEALPix-to-point-source conversion."
        )
    fi = healpix.resolve_frequency_index(resolve_freq)
    temp_map = healpix.maps[fi]
    arrays = healpix_map_to_point_arrays(
        temp_map,
        resolve_freq,
        sky.brightness_conversion,
        healpix_q_maps=healpix.q_maps,
        healpix_u_maps=healpix.u_maps,
        healpix_v_maps=healpix.v_maps,
        observation_frequencies=healpix.frequencies,
        freq_index=fi,
        healpix_maps=healpix.maps,
        coordinate_frame=healpix.coordinate_frame,
        ref_freq_out=resolve_freq,
        polarization_brightness_conversion=(
            sky.polarization_brightness_conversion.value
        ),
        warn=False,
    )
    if flux_limit > 0:
        mask = arrays["flux"] >= flux_limit
        arrays = {
            key: (value[mask] if isinstance(value, np.ndarray) else value)
            for key, value in arrays.items()
        }

    new_point = PointSourceData(
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
    )

    # Update provenance: the new point catalog has *no* sub-pixel positional
    # information.  Its effective angular resolution is the HEALPix pixel
    # size at the nside used during conversion.  Without this update,
    # downstream code that trusts ``angular_resolution_rad`` would still see
    # the original diffuse template's resolution band, which no longer
    # describes the quantized point catalog we just produced.
    pixel_resolution_rad = float(hp.nside2resol(healpix.nside))
    new_prov = sky.provenance.replace(
        angular_resolution_rad=(pixel_resolution_rad, float(np.pi)),
    )

    if clear_other:
        return sky.replace(point=new_point, healpix=None, provenance=new_prov)
    return sky.replace(point=new_point, provenance=new_prov)


def with_memmap_backing(
    sky: SkyModel,
    path: str | None = None,
) -> SkyModel:
    """Return a copy with HEALPix maps backed by memory-mapped files."""
    if sky.healpix is None:
        raise ValueError(
            "No HEALPix maps to back with memmap. Materialize a HEALPix payload first."
        )

    if path is None:
        path = tempfile.mkdtemp(prefix="radiosim_memmap_")

    os.makedirs(path, exist_ok=True)

    def _to_memmap(arr: np.ndarray, name: str) -> np.memmap:
        fpath = os.path.join(path, f"{name}.dat")
        mm = np.memmap(fpath, dtype=arr.dtype, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        return np.memmap(fpath, dtype=arr.dtype, mode="r", shape=arr.shape)

    healpix = HealpixData(
        maps=_to_memmap(sky.healpix.maps, "i_maps"),
        nside=sky.healpix.nside,
        frequencies=sky.healpix.frequencies,
        channel_widths_hz=sky.healpix.channel_widths_hz,
        coordinate_frame=sky.healpix.coordinate_frame,
        hpx_inds=sky.healpix.hpx_inds,
        q_maps=(
            _to_memmap(sky.healpix.q_maps, "q_maps")
            if sky.healpix.q_maps is not None
            else None
        ),
        u_maps=(
            _to_memmap(sky.healpix.u_maps, "u_maps")
            if sky.healpix.u_maps is not None
            else None
        ),
        v_maps=(
            _to_memmap(sky.healpix.v_maps, "v_maps")
            if sky.healpix.v_maps is not None
            else None
        ),
        i_unit=sky.healpix.i_unit,
        q_unit=sky.healpix.q_unit,
        u_unit=sky.healpix.u_unit,
        v_unit=sky.healpix.v_unit,
        i_brightness_conversion=sky.healpix.i_brightness_conversion,
        q_brightness_conversion=sky.healpix.q_brightness_conversion,
        u_brightness_conversion=sky.healpix.u_brightness_conversion,
        v_brightness_conversion=sky.healpix.v_brightness_conversion,
    )

    return sky.replace(healpix=healpix)


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
    pixel_area_sr = 4.0 * np.pi / hp.nside2npix(nside)

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


def _fit_one_candidate(
    center: int,
    flux_ch_jy: np.ndarray,
    *,
    nside: int,
    patch_radius: float,
    sigma_init: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | None:
    """Compute the patch geometry + Gaussian fit for one candidate pixel.

    Pure read of ``flux_ch_jy``; safe to call concurrently for distinct
    candidates. Returns ``None`` for patches that are too small to fit
    (matches the ``patch.size < 8`` skip in the sequential path).

    Returns ``(patch, px, py, params, ok)``.
    """
    patch = hp.query_disc(
        nside, hp.pix2vec(nside, center), patch_radius, inclusive=True
    )
    if patch.size < 8:
        return None
    px, py = _gnomonic_patch_coords(int(center), patch, nside)
    pz = flux_ch_jy[patch]
    params, ok = _fit_elliptical_gaussian(px, py, pz, sigma_init_rad=sigma_init)
    return patch, px, py, params, ok


def _apply_fit_to_buffer(
    fit: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | None,
    *,
    flux_ch_jy: np.ndarray,
    pixel_area_sr: float,
    inpaint_mask_sigma: float,
    flux_limit_jy: float,
    catalog_present: bool,
    inpaint_mask: set[int],
) -> tuple[int, int]:
    """Apply one Gaussian fit to ``flux_ch_jy`` and update the inpaint mask.

    Subtracts the fitted model on the patch, accumulates pixel indices
    within ``inpaint_mask_sigma × σ_fit`` of the centre into
    ``inpaint_mask``, and returns ``(n_ok_delta, n_failed_delta)`` for the
    caller to tally.

    No-op + ``(0, 0)`` when the fit was skipped (``patch.size < 8``).
    No-op + ``(0, 1)`` when curve_fit failed or the integrated flux fell
    below ``flux_limit_jy`` in auto-detect mode.
    """
    if fit is None:
        return 0, 0
    patch, px, py, params, ok = fit
    if not ok:
        return 0, 1
    # Reject fits whose integrated flux is below the catalog-style
    # threshold so we don't subtract noise-level bumps.  Integral of
    # A·exp(-½(x'²/σ_M² + y'²/σ_m²)) over the plane is 2π·A·σ_M·σ_m;
    # divide by pixel area to convert from Jy/sr back to a Jy total.
    amp_fit = float(params[0])
    sigma_M_fit = float(params[3])
    sigma_m_fit = float(params[4])
    integrated_flux_jy = (
        amp_fit * 2.0 * np.pi * sigma_M_fit * sigma_m_fit / pixel_area_sr
    )
    if not catalog_present and integrated_flux_jy < flux_limit_jy:
        return 0, 1

    model_vals = _evaluate_elliptical_gaussian(params, px, py)
    flux_ch_jy[patch] -= model_vals

    x0_fit, y0_fit, pa_fit = params[1], params[2], params[5]
    cos_pa = np.cos(pa_fit)
    sin_pa = np.sin(pa_fit)
    xr = (px - x0_fit) * cos_pa + (py - y0_fit) * sin_pa
    yr = -(px - x0_fit) * sin_pa + (py - y0_fit) * cos_pa
    ellipse_r2 = (xr / sigma_M_fit) ** 2 + (yr / sigma_m_fit) ** 2
    core_mask = ellipse_r2 <= inpaint_mask_sigma**2
    for p in patch[core_mask]:
        inpaint_mask.add(int(p))
    return 1, 0


def _fit_and_subtract_per_channel(
    sky: SkyModel,
    candidate_pix: np.ndarray,
    *,
    patch_radius: float,
    sigma_init: float,
    inpaint_mask_sigma: float,
    flux_limit_jy: float,
    catalog_present: bool,
    n_workers: int = 1,
) -> tuple[np.ndarray, set[int], int, int]:
    """Run the per-channel Gaussian-fit-and-subtract loop.

    For each channel: convert the cube slice to Jy, fit an elliptical
    Gaussian + planar baseline at every candidate pixel, reject fits whose
    integrated flux lies below ``flux_limit_jy`` (auto-detect mode only),
    subtract the model on the patch, and accumulate the inpaint mask
    (pixels within ``inpaint_mask_sigma × σ_fit`` of each centre).
    Converts each modified channel back to brightness temperature.

    When ``n_workers > 1``, the per-candidate fits within a channel run in
    parallel via :class:`concurrent.futures.ThreadPoolExecutor`.  Every
    parallel fit sees the same starting flux buffer (no inter-candidate
    coupling); subtractions are then applied serially in the input
    ``candidate_pix`` order, so the result is deterministic for any fixed
    candidate set.  Sequential mode (``n_workers == 1``) preserves the
    historical contract where a candidate's fit sees the partially-
    subtracted flux of any previously-processed candidate within the same
    channel; the two modes therefore agree only when candidate patches do
    not overlap (the common case at typical bright-source separations).

    Returns ``(new_maps_K, inpaint_mask_pixels, n_fits_ok, n_fits_failed)``.
    """
    nside = sky.healpix.nside
    pixel_area_sr = 4.0 * np.pi / hp.nside2npix(nside)
    new_maps = np.array(sky.healpix.maps, dtype=np.float64, copy=True)
    n_freq = new_maps.shape[0]
    cube_freqs = sky.healpix.frequencies
    inpaint_mask: set[int] = set()
    n_fits_ok = 0
    n_fits_failed = 0

    for fi in range(n_freq):
        freq_ch = float(cube_freqs[fi])
        use_rj_ch = sky.brightness_conversion.value == "rayleigh-jeans" or np.any(
            new_maps[fi] <= 0.0
        )
        method_ch = "rayleigh-jeans" if use_rj_ch else "planck"
        flux_ch_jy = brightness_temp_to_flux_density(
            new_maps[fi].copy(),
            frequency=freq_ch,
            solid_angle=pixel_area_sr,
            method=method_ch,
        )

        if n_workers > 1:
            # Parallel fit phase: every candidate reads the same buffer.
            # ``functools.partial`` binds the shared kwargs once; ``pool.map``
            # passes each candidate index in as the leading positional arg.
            fit_call = functools.partial(
                _fit_one_candidate,
                flux_ch_jy=flux_ch_jy,
                nside=nside,
                patch_radius=patch_radius,
                sigma_init=sigma_init,
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                fit_results = list(pool.map(fit_call, candidate_pix.tolist()))
        else:
            # Sequential fit phase: applied inline so the next fit sees
            # residuals from prior subtractions in the same channel.
            fit_results = []
            for center in candidate_pix:
                fit = _fit_one_candidate(
                    int(center),
                    flux_ch_jy,
                    nside=nside,
                    patch_radius=patch_radius,
                    sigma_init=sigma_init,
                )
                fit_results.append(fit)
                ok_delta, failed_delta = _apply_fit_to_buffer(
                    fit,
                    flux_ch_jy=flux_ch_jy,
                    pixel_area_sr=pixel_area_sr,
                    inpaint_mask_sigma=inpaint_mask_sigma,
                    flux_limit_jy=flux_limit_jy,
                    catalog_present=catalog_present,
                    inpaint_mask=inpaint_mask,
                )
                n_fits_ok += ok_delta
                n_fits_failed += failed_delta

        if n_workers > 1:
            # Serial subtraction in deterministic candidate order.
            for fit in fit_results:
                ok_delta, failed_delta = _apply_fit_to_buffer(
                    fit,
                    flux_ch_jy=flux_ch_jy,
                    pixel_area_sr=pixel_area_sr,
                    inpaint_mask_sigma=inpaint_mask_sigma,
                    flux_limit_jy=flux_limit_jy,
                    catalog_present=catalog_present,
                    inpaint_mask=inpaint_mask,
                )
                n_fits_ok += ok_delta
                n_fits_failed += failed_delta

        flux_positive = flux_ch_jy.copy()
        flux_positive[flux_positive <= 0] = np.finfo(np.float64).tiny
        method_back = "rayleigh-jeans" if use_rj_ch else "planck"
        new_maps[fi] = flux_density_to_brightness_temp(
            flux_positive,
            frequency=freq_ch,
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
    n_workers: int = 1,
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

    **Cost.** This routine runs one ``scipy.optimize.curve_fit`` call per
    candidate per frequency channel — i.e. ``O(N_candidates × N_freq)``
    nonlinear fits.  At dense HEALPix with broad bands this can stretch
    into minutes-to-hours.  Cap the candidate list with ``max_sources=...``
    or pre-filter with a higher ``flux_limit_jy`` when many candidates are
    expected; a logger warning is emitted when the predicted fit count
    exceeds 1000.

    **Parallelism (``n_workers``).** When ``n_workers > 1``, per-candidate
    fits within each channel run on a :class:`ThreadPoolExecutor`.  ``scipy``
    MINPACK and the pure-numpy model evaluator both release the GIL during
    the actual minimisation, so each thread runs effectively in parallel
    on CPU-bound work.  Subtractions are applied serially in the input
    candidate order after the parallel fits return, so results are
    deterministic for any fixed candidate set.  Sequential mode
    (``n_workers == 1``, default) preserves the historical contract where
    a candidate's fit sees the partially-subtracted flux of any
    previously-processed candidate within the same channel; the two modes
    therefore agree numerically only when candidate patches do not overlap
    (the common case at typical bright-source separations).
    """
    if sky.healpix is None:
        raise ValueError(
            "subtract_bright_sources requires a HEALPix payload; got a "
            "point-only model.  Use materialize_healpix_model(...) first."
        )
    sky.healpix.require_dense("subtract_bright_sources")

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
    n_fits_total = int(candidate_pix.size) * n_freq_total
    if n_fits_total > _SUBTRACT_FIT_COUNT_WARN_THRESHOLD:
        logger.warning(
            "subtract_bright_sources: about to run %d × %d = %d "
            "scipy.optimize.curve_fit calls (one per candidate per "
            "channel) — this is O(N_candidates × N_freq) and can be "
            "minutes-to-hours at large nside / wide bands. Pass "
            "max_sources=... to cap the candidate list, or filter by "
            "flux_limit_jy first.",
            int(candidate_pix.size),
            n_freq_total,
            n_fits_total,
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
        n_workers=int(n_workers),
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

    new_healpix = HealpixData(
        maps=new_maps.astype(sky.healpix.maps.dtype, copy=False),
        nside=sky.healpix.nside,
        frequencies=sky.healpix.frequencies,
        channel_widths_hz=sky.healpix.channel_widths_hz,
        coordinate_frame=sky.healpix.coordinate_frame,
        hpx_inds=sky.healpix.hpx_inds,
        q_maps=sky.healpix.q_maps,
        u_maps=sky.healpix.u_maps,
        v_maps=sky.healpix.v_maps,
        i_unit=sky.healpix.i_unit,
        q_unit=sky.healpix.q_unit,
        u_unit=sky.healpix.u_unit,
        v_unit=sky.healpix.v_unit,
        i_brightness_conversion=sky.healpix.i_brightness_conversion,
        q_brightness_conversion=sky.healpix.q_brightness_conversion,
        u_brightness_conversion=sky.healpix.u_brightness_conversion,
        v_brightness_conversion=sky.healpix.v_brightness_conversion,
    )

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


# =============================================================================
# Linear-polarisation diagnostics
# =============================================================================


def compute_linear_polarization(
    sky: SkyModel,
    *,
    frequency: float | None = None,
) -> dict[str, np.ndarray]:
    """Derive ``(P, χ, P/|I|)`` from a SkyModel's Stokes Q/U.

    For a HEALPix payload, returns dense maps shaped ``(npix,)`` when
    ``frequency`` is given (the closest channel is selected) or
    ``(n_freq, npix)`` when ``frequency=None``.  For a point-source
    payload, returns ``(n_sources,)`` arrays — Q/U here are intrinsic
    Stokes parameters, no per-frequency scaling is applied.

    Parameters
    ----------
    sky
        Sky model carrying Stokes Q and U.  ``ValueError`` is raised if
        either is absent.
    frequency
        Optional frequency (Hz) at which to slice a HEALPix payload.
        Ignored for point-source payloads.

    Returns
    -------
    dict
        Keys:

        - ``"P"`` : ``sqrt(Q² + U²)`` (linear polarisation amplitude).
        - ``"chi_deg"`` : ``0.5 · atan2(U, Q)`` in degrees, range
          ``(-90°, 90°]``.
        - ``"frac_pol"`` : ``P / |I|`` (fractional linear polarisation).
          ``nan`` where ``I = 0``.

    Raises
    ------
    ValueError
        If neither payload carries Q and U.
    """
    if sky.healpix is not None:
        if sky.healpix.q_maps is None or sky.healpix.u_maps is None:
            raise ValueError(
                "compute_linear_polarization requires Stokes Q and U HEALPix "
                "maps; the input has none.  Load a polarised template (e.g. "
                "PySM3 with synchrotron) or supply Q/U arrays explicitly."
            )
        if frequency is None:
            i_maps = sky.healpix.maps
            q_maps = sky.healpix.q_maps
            u_maps = sky.healpix.u_maps
        else:
            idx = sky.healpix.resolve_frequency_index(float(frequency))
            i_maps = sky.healpix.maps[idx]
            q_maps = sky.healpix.q_maps[idx]
            u_maps = sky.healpix.u_maps[idx]
        return _linear_pol_arrays(i_maps, q_maps, u_maps)

    if sky.point is not None:
        if sky.point.stokes_q is None or sky.point.stokes_u is None:
            raise ValueError(
                "compute_linear_polarization requires Stokes Q and U "
                "components on the point payload; got neither."
            )
        return _linear_pol_arrays(
            sky.point.flux,
            sky.point.stokes_q,
            sky.point.stokes_u,
        )

    raise ValueError("SkyModel carries no payload; cannot derive polarisation.")


def _linear_pol_arrays(
    i: np.ndarray,
    q: np.ndarray,
    u: np.ndarray,
) -> dict[str, np.ndarray]:
    q_arr = np.asarray(q, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    i_arr = np.asarray(i, dtype=float)
    p = np.hypot(q_arr, u_arr)
    chi_rad = 0.5 * np.arctan2(u_arr, q_arr)
    chi_deg = np.degrees(chi_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_pol = p / np.abs(i_arr)
    frac_pol = np.where(i_arr == 0.0, np.nan, frac_pol)
    return {"P": p, "chi_deg": chi_deg, "frac_pol": frac_pol}


# =============================================================================
# Monopole bookkeeping operations
# =============================================================================


def _coerce_monopole_convention(
    convention: MonopoleConvention | str,
) -> MonopoleConvention:
    if isinstance(convention, MonopoleConvention):
        return convention
    return MonopoleConvention(convention)


def with_monopole(
    sky: SkyModel,
    value_k: float,
    convention: MonopoleConvention | str = MonopoleConvention.ABSOLUTE_NO_CMB,
) -> SkyModel:
    """Return a new :class:`SkyModel` with ``value_k`` added to the sky monopole.

    For HEALPix payloads, ``value_k`` is added uniformly to every pixel of
    the Stokes-I cube (Q/U/V are zero-mean by construction and are not
    touched).  For pure point-source payloads the map arrays are unchanged —
    only the provenance is updated to advertise the new monopole.

    Parameters
    ----------
    sky
        Input sky model.
    value_k
        Brightness-temperature monopole to add, in Kelvin.
    convention
        Monopole convention to declare on the returned model.  Use
        :class:`MonopoleConvention.ABSOLUTE_WITH_CMB` when re-adding the CMB,
        :class:`MonopoleConvention.ABSOLUTE_NO_CMB` otherwise.

    Returns
    -------
    SkyModel
        A new model with the DC level shifted and provenance updated.
    """
    convention = _coerce_monopole_convention(convention)
    if np.ndim(value_k) != 0:
        raise TypeError(
            "with_monopole(value_k=...) must be a scalar (a uniform DC shift "
            "applied to every pixel of the Stokes-I cube); received an "
            f"array-like with shape {np.asarray(value_k).shape}.  Did you "
            "intend to pass a per-channel mean?  with_monopole only supports "
            "a single full-sky scalar."
        )
    value_k = float(value_k)
    if sky.provenance.is_partial_sky:
        raise ValueError(
            "with_monopole requires a full-sky model; partial-sky products do "
            "not have a well-defined global monopole."
        )

    old_prov = sky.provenance
    old_monopole = old_prov.monopole_k
    new_monopole = old_monopole + value_k if old_monopole is not None else value_k
    new_prov = old_prov.replace(
        monopole_convention=convention,
        monopole_k=new_monopole,
    )

    if sky.healpix is None:
        return sky.replace(provenance=new_prov)

    new_maps = sky.healpix.maps + np.asarray(value_k, dtype=sky.healpix.maps.dtype)
    new_healpix = HealpixData(
        maps=new_maps,
        nside=sky.healpix.nside,
        frequencies=sky.healpix.frequencies,
        channel_widths_hz=sky.healpix.channel_widths_hz,
        coordinate_frame=sky.healpix.coordinate_frame,
        hpx_inds=sky.healpix.hpx_inds,
        q_maps=sky.healpix.q_maps,
        u_maps=sky.healpix.u_maps,
        v_maps=sky.healpix.v_maps,
        i_unit=sky.healpix.i_unit,
        q_unit=sky.healpix.q_unit,
        u_unit=sky.healpix.u_unit,
        v_unit=sky.healpix.v_unit,
        i_brightness_conversion=sky.healpix.i_brightness_conversion,
        q_brightness_conversion=sky.healpix.q_brightness_conversion,
        u_brightness_conversion=sky.healpix.u_brightness_conversion,
        v_brightness_conversion=sky.healpix.v_brightness_conversion,
    )
    return sky.replace(healpix=new_healpix, provenance=new_prov)


def with_monopole_subtracted(sky: SkyModel) -> SkyModel:
    """Return a new :class:`SkyModel` with the per-frequency Stokes-I mean removed.

    For HEALPix payloads, the pixel-weighted mean of each frequency channel is
    subtracted from the Stokes-I cube (Q/U/V channels are left untouched —
    they are already mean-zero by construction).  For pure point-source
    payloads only the provenance is updated (no array modification).  In
    both cases the returned model's ``provenance.monopole_convention`` becomes
    :class:`MonopoleConvention.MEAN_SUBTRACTED` and ``monopole_k`` is set to 0.

    Raises
    ------
    ValueError
        If the input model already has ``monopole_convention = MEAN_SUBTRACTED``
        (idempotent subtraction on an already-zero-mean sky is a user error).
    """
    if sky.provenance.monopole_convention is MonopoleConvention.MEAN_SUBTRACTED:
        raise ValueError(
            "SkyModel is already mean-subtracted "
            "(provenance.monopole_convention=MEAN_SUBTRACTED); "
            "with_monopole_subtracted would subtract the mean twice."
        )
    if sky.provenance.is_partial_sky:
        raise ValueError(
            "with_monopole_subtracted requires a full-sky model; partial-sky "
            "products do not have a well-defined global monopole."
        )

    new_prov = sky.provenance.replace(
        monopole_convention=MonopoleConvention.MEAN_SUBTRACTED,
        monopole_k=0.0,
    )

    if sky.healpix is None:
        return sky.replace(provenance=new_prov)

    maps = sky.healpix.maps
    # Per-channel pixel-area-weighted mean: pixels are equal-area on the HEALPix
    # grid so a plain mean over stored pixels is the correct solid-angle average.
    means = maps.mean(axis=1, keepdims=True)
    new_maps = maps - means.astype(maps.dtype)

    new_healpix = HealpixData(
        maps=new_maps,
        nside=sky.healpix.nside,
        frequencies=sky.healpix.frequencies,
        channel_widths_hz=sky.healpix.channel_widths_hz,
        coordinate_frame=sky.healpix.coordinate_frame,
        hpx_inds=sky.healpix.hpx_inds,
        q_maps=sky.healpix.q_maps,
        u_maps=sky.healpix.u_maps,
        v_maps=sky.healpix.v_maps,
        i_unit=sky.healpix.i_unit,
        q_unit=sky.healpix.q_unit,
        u_unit=sky.healpix.u_unit,
        v_unit=sky.healpix.v_unit,
        i_brightness_conversion=sky.healpix.i_brightness_conversion,
        q_brightness_conversion=sky.healpix.q_brightness_conversion,
        u_brightness_conversion=sky.healpix.u_brightness_conversion,
        v_brightness_conversion=sky.healpix.v_brightness_conversion,
    )
    return sky.replace(healpix=new_healpix, provenance=new_prov)
