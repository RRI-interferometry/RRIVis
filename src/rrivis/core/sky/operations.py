"""Functional sky-model operations.

These helpers keep mutation-free transformations and memory-management
operations outside ``SkyModel`` itself.
"""

from __future__ import annotations

import tempfile
import warnings
from typing import TYPE_CHECKING, Any

import healpy as hp
import numpy as np

from ._data import (
    HealpixData,
    MonopoleConvention,
    PointSourceData,
    SkyProvenance,
    SourceSubtractionStatus,
)

if TYPE_CHECKING:
    from .model import SkyModel


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
    """
    if sky.point is None:
        raise ValueError(
            "No point sources available for conversion. "
            "Load a point-source model first, for example with "
            "rrivis.core.sky.loaders.load_gleam()."
        )

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

    from .convert import point_sources_to_healpix_maps

    effective_ref_freq: float | np.ndarray | None = None
    if sky.point.ref_freq is not None and np.any(sky.point.ref_freq > 0):
        effective_ref_freq = sky.point.ref_freq
    else:
        effective_ref_freq = ref_frequency or sky.reference_frequency
        if effective_ref_freq is None:
            raise ValueError(
                "ref_frequency must be provided when this SkyModel has no "
                "per-source ref_freq values and no reference_frequency. "
                "Set it via with_reference_frequency() or pass ref_frequency "
                "explicitly."
            )

    spectrum = sky.point.spectrum
    i_maps, q_maps, u_maps, v_maps = point_sources_to_healpix_maps(
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
    )
    if clear_other:
        return sky.replace(point=None, healpix=new_healpix)
    return sky.replace(healpix=new_healpix)


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

    from .convert import healpix_map_to_point_arrays

    resolve_freq = freq or float(healpix.frequencies[0])
    if resolve_freq is None:
        raise ValueError(
            "frequency is required for HEALPix-to-point-source conversion."
        )
    fi = sky.resolve_frequency_index(resolve_freq)
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
    if clear_other:
        return sky.replace(point=new_point, healpix=None)
    return sky.replace(point=new_point)


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
        path = tempfile.mkdtemp(prefix="rrivis_memmap_")

    import os

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


def _fit_symmetric_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sigma_init_rad: float,
) -> tuple[np.ndarray, bool]:
    """Fit ``A · exp(-½·r²/σ²) + b_x·x + b_y·y + c`` to the patch.

    Returns ``(params, ok)`` where ``params`` is
    ``[A, x0, y0, sigma, bx, by, c]``; ``ok`` is False if the fit failed or
    produced non-physical values.
    """
    from scipy.optimize import curve_fit

    def _model(xy, amp, x0, y0, sigma, bx, by, c):
        xx, yy = xy
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        return amp * np.exp(-0.5 * r2 / sigma**2) + bx * xx + by * yy + c

    peak_idx = int(np.argmax(z))
    amp_init = float(z[peak_idx] - np.median(z))
    amp_init = amp_init if amp_init > 0 else float(z[peak_idx])
    p0 = [
        amp_init,
        float(x[peak_idx]),
        float(y[peak_idx]),
        float(sigma_init_rad),
        0.0,
        0.0,
        float(np.median(z)),
    ]

    lower = [
        0.0,
        float(x.min()),
        float(y.min()),
        sigma_init_rad * 0.2,
        -np.inf,
        -np.inf,
        -np.inf,
    ]
    upper = [
        np.inf,
        float(x.max()),
        float(y.max()),
        sigma_init_rad * 5.0,
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
            maxfev=500,
        )
    except (RuntimeError, ValueError):
        return np.asarray(p0), False

    sigma_fit = popt[3]
    amp_fit = popt[0]
    if amp_fit <= 0.0 or sigma_fit <= sigma_init_rad * 0.25:
        return popt, False
    return popt, True


def _evaluate_symmetric_gaussian(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    amp, x0, y0, sigma, _bx, _by, _c = params
    return amp * np.exp(-0.5 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma**2)


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


def subtract_bright_sources(
    sky: SkyModel,
    *,
    flux_limit_jy: float,
    frequency_hz: float,
    catalog: SkyModel | None = None,
    patch_radius_rad: float | None = None,
    inpaint_mask_sigma: float = 3.0,
    inpaint_max_iterations: int = 80,
    inpaint_rtol: float = 1e-3,
    max_sources: int | None = None,
    detection_peak_fraction: float = 0.2,
) -> SkyModel:
    """Remove bright point sources from a HEALPix diffuse map.

    Implements the Remazeilles, Dickinson & Banday (2015) methodology for
    preparing a source-subtracted all-sky template that can be combined
    with an independent point-source catalog without double-counting:

    1. **Detection.**  Either use the positions in a supplied ``catalog``
       (flux-scaled to ``frequency_hz`` and filtered at ``flux_limit_jy``),
       or auto-detect local maxima in the map above ``flux_limit_jy``.
    2. **Fit.**  For each candidate, project a small tangent-plane patch
       and fit a 2D symmetric Gaussian + planar baseline.
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
    """
    if sky.healpix is None:
        raise ValueError(
            "subtract_bright_sources requires a HEALPix payload; got a "
            "point-only model.  Use materialize_healpix_model(...) first."
        )
    if sky.healpix.is_sparse:
        raise ValueError(
            "subtract_bright_sources requires a dense HEALPix cube; call "
            "sky.healpix.to_dense() (or re-materialize) before subtraction."
        )

    from .constants import (
        brightness_temp_to_flux_density,
        flux_density_to_brightness_temp,
    )

    nside = sky.healpix.nside
    pixel_area_sr = 4.0 * np.pi / hp.nside2npix(nside)
    default_patch_rad = 3.0 * hp.nside2resol(nside)
    patch_radius = (
        float(patch_radius_rad) if patch_radius_rad is not None else default_patch_rad
    )
    sigma_init = hp.nside2resol(nside) / 2.355

    # Candidate positions at the target subtraction frequency.
    idx = sky.resolve_frequency_index(frequency_hz)
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

    if candidate_pix.size == 0:
        old_prov = sky.provenance
        new_prov = SkyProvenance(
            flux_completeness_jy=old_prov.flux_completeness_jy,
            flux_completeness_freq_hz=old_prov.flux_completeness_freq_hz,
            angular_resolution_rad=old_prov.angular_resolution_rad,
            sky_coverage=old_prov.sky_coverage,
            coverage_fraction=old_prov.coverage_fraction,
            coverage_footprint=old_prov.coverage_footprint,
            monopole_convention=old_prov.monopole_convention,
            monopole_k=old_prov.monopole_k,
            source_subtraction=SourceSubtractionStatus.ABOVE_THRESHOLD,
            source_subtraction_threshold_jy=float(flux_limit_jy),
            source_subtraction_freq_hz=float(frequency_hz),
            source_subtraction_method="gaussian_fit_inpaint",
            notes=old_prov.notes,
        )
        return sky.replace(provenance=new_prov)

    if max_sources is not None and candidate_pix.size > max_sources:
        order = np.argsort(-flux_per_pixel_jy[candidate_pix])
        candidate_pix = candidate_pix[order[:max_sources]]

    # Per-channel fit + subtract loop.
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

        for center in candidate_pix:
            patch = hp.query_disc(
                nside, hp.pix2vec(nside, center), patch_radius, inclusive=True
            )
            if patch.size < 8:
                continue

            px, py = _gnomonic_patch_coords(int(center), patch, nside)
            pz = flux_ch_jy[patch]
            params, ok = _fit_symmetric_gaussian(px, py, pz, sigma_init_rad=sigma_init)
            if not ok:
                n_fits_failed += 1
                continue

            # Reject fits whose integrated flux is below the catalog-style
            # threshold so we don't subtract noise-level bumps.
            amp_fit = float(params[0])
            sigma_fit_sq = float(params[3]) ** 2
            integrated_flux_jy = amp_fit * 2.0 * np.pi * sigma_fit_sq / pixel_area_sr
            if catalog is None and integrated_flux_jy < flux_limit_jy:
                n_fits_failed += 1
                continue
            n_fits_ok += 1

            model_vals = _evaluate_symmetric_gaussian(params, px, py)
            flux_ch_jy[patch] -= model_vals

            sigma_fit = params[3]
            x0_fit, y0_fit = params[1], params[2]
            r2 = (px - x0_fit) ** 2 + (py - y0_fit) ** 2
            core_mask = r2 <= (inpaint_mask_sigma * sigma_fit) ** 2
            for p in patch[core_mask]:
                inpaint_mask.add(int(p))

        flux_positive = flux_ch_jy.copy()
        flux_positive[flux_positive <= 0] = np.finfo(np.float64).tiny
        method_back = "rayleigh-jeans" if use_rj_ch else "planck"
        new_maps[fi] = flux_density_to_brightness_temp(
            flux_positive,
            frequency=freq_ch,
            solid_angle=pixel_area_sr,
            method=method_back,
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
    new_prov = SkyProvenance(
        flux_completeness_jy=old_prov.flux_completeness_jy,
        flux_completeness_freq_hz=old_prov.flux_completeness_freq_hz,
        angular_resolution_rad=old_prov.angular_resolution_rad,
        sky_coverage=old_prov.sky_coverage,
        coverage_fraction=old_prov.coverage_fraction,
        coverage_footprint=old_prov.coverage_footprint,
        monopole_convention=old_prov.monopole_convention,
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
            idx = sky.resolve_frequency_index(float(frequency))
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
    value_k = float(value_k)
    if sky.provenance.is_partial_sky:
        raise ValueError(
            "with_monopole requires a full-sky model; partial-sky products do "
            "not have a well-defined global monopole."
        )

    old_prov = sky.provenance
    old_monopole = old_prov.monopole_k
    new_monopole = old_monopole + value_k if old_monopole is not None else value_k
    new_prov = SkyProvenance(
        flux_completeness_jy=old_prov.flux_completeness_jy,
        flux_completeness_freq_hz=old_prov.flux_completeness_freq_hz,
        angular_resolution_rad=old_prov.angular_resolution_rad,
        sky_coverage=old_prov.sky_coverage,
        coverage_fraction=old_prov.coverage_fraction,
        coverage_footprint=old_prov.coverage_footprint,
        monopole_convention=convention,
        monopole_k=new_monopole,
        source_subtraction=old_prov.source_subtraction,
        source_subtraction_threshold_jy=old_prov.source_subtraction_threshold_jy,
        source_subtraction_freq_hz=old_prov.source_subtraction_freq_hz,
        source_subtraction_method=old_prov.source_subtraction_method,
        notes=old_prov.notes,
    )

    if sky.healpix is None:
        return sky.replace(provenance=new_prov)

    new_maps = sky.healpix.maps + np.asarray(value_k, dtype=sky.healpix.maps.dtype)
    new_healpix = HealpixData(
        maps=new_maps,
        nside=sky.healpix.nside,
        frequencies=sky.healpix.frequencies,
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

    old_prov = sky.provenance
    new_prov = SkyProvenance(
        flux_completeness_jy=old_prov.flux_completeness_jy,
        flux_completeness_freq_hz=old_prov.flux_completeness_freq_hz,
        angular_resolution_rad=old_prov.angular_resolution_rad,
        sky_coverage=old_prov.sky_coverage,
        coverage_fraction=old_prov.coverage_fraction,
        coverage_footprint=old_prov.coverage_footprint,
        monopole_convention=MonopoleConvention.MEAN_SUBTRACTED,
        monopole_k=0.0,
        source_subtraction=old_prov.source_subtraction,
        source_subtraction_threshold_jy=old_prov.source_subtraction_threshold_jy,
        source_subtraction_freq_hz=old_prov.source_subtraction_freq_hz,
        source_subtraction_method=old_prov.source_subtraction_method,
        notes=old_prov.notes,
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
