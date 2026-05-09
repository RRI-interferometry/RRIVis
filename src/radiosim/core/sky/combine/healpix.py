# radiosim/core/sky/_combine_healpix.py
"""HEALPix map combination — RJ fast path and Planck round-trip path."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

import healpy as hp
import numpy as np

from ..containers.constants import (
    BrightnessConversion,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
    rayleigh_jeans_factor,
)
from ..containers.data import PointSourceData, PointSpectrum
from ..containers.spectral import (
    evaluate_point_flux_at_freq,
    per_source_reference_frequencies,
)
from ..support.allocation import allocate_cube, ensure_scratch_dir, finalize_cube
from .regrid import (
    _format_healpix_freq_grid,
    _point_source_healpix_indices,
    _resolve_common_healpix_frame,
)

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from ..containers.model import SkyModel


def _point_contributions_at_freq(
    ipix: np.ndarray,
    point: PointSourceData,
    spectrum: PointSpectrum | None,
    ref_freq: float | np.ndarray,
    freq_hz: float,
    npix: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin a point-source model's I/Q/U/V into HEALPix flux-density (Jy) maps.

    Uses the per-channel ``PointSpectrum`` table when populated (lossless
    nearest-channel lookup); otherwise falls back to power-law extrapolation
    plus Faraday rotation on the reference Stokes arrays.  Mirrors the
    behaviour of :func:`evaluate_point_flux_at_freq`, then accumulates by
    pixel via :func:`np.bincount`.
    """
    i_f, q_f, u_f, v_f = evaluate_point_flux_at_freq(
        stokes_i=point.flux,
        stokes_q=point.stokes_q,
        stokes_u=point.stokes_u,
        stokes_v=point.stokes_v,
        spectral_index=point.spectral_index,
        spectral_coeffs=point.spectral_coeffs,
        ref_freq=ref_freq,
        rotation_measure=point.rotation_measure,
        per_channel_flux=spectrum.flux if spectrum is not None else None,
        per_channel_stokes_q=(
            spectrum.stokes_q
            if spectrum is not None and spectrum.stokes_q is not None
            else None
        ),
        per_channel_stokes_u=(
            spectrum.stokes_u
            if spectrum is not None and spectrum.stokes_u is not None
            else None
        ),
        per_channel_stokes_v=(
            spectrum.stokes_v
            if spectrum is not None and spectrum.stokes_v is not None
            else None
        ),
        channel_frequencies=spectrum.frequencies if spectrum is not None else None,
        freq=freq_hz,
    )
    return (
        np.bincount(ipix, weights=i_f, minlength=npix),
        np.bincount(ipix, weights=q_f, minlength=npix),
        np.bincount(ipix, weights=u_f, minlength=npix),
        np.bincount(ipix, weights=v_f, minlength=npix),
    )


logger = logging.getLogger(__name__)


class CombineHealpixData(TypedDict):
    """Return type for combine_healpix."""

    healpix_maps: np.ndarray
    healpix_q_maps: np.ndarray | None
    healpix_u_maps: np.ndarray | None
    healpix_v_maps: np.ndarray | None
    healpix_nside: int
    observation_frequencies: np.ndarray
    channel_widths_hz: np.ndarray | None
    coordinate_frame: str
    reference_frequency: float | None


def _resolve_combined_channel_widths(
    healpix_models: list[SkyModel],
) -> np.ndarray | None:
    """Resolve ``channel_widths_hz`` for a HEALPix combine.

    All HEALPix-bearing inputs must agree: either every input declares the
    same widths array (positional equality on the already-validated common
    frequency grid), or none declares one. A partial declaration is an
    asymmetric-metadata error and raises.
    """
    widths_seen: list[np.ndarray | None] = [
        m.healpix.channel_widths_hz for m in healpix_models if m.healpix is not None
    ]
    if not widths_seen:
        return None
    have = [w for w in widths_seen if w is not None]
    if not have:
        return None
    if len(have) != len(widths_seen):
        raise ValueError(
            "Cannot combine HEALPix models with asymmetric channel_widths_hz: "
            "some inputs declare per-channel bandwidths and others do not. "
            "Either declare widths on every input or on none of them."
        )
    reference = have[0]
    for widths in have[1:]:
        if not np.array_equal(widths, reference):
            raise ValueError(
                "Cannot combine HEALPix models with mismatched channel_widths_hz. "
                "Frequency grids agree but per-channel bandwidth declarations do "
                "not; align them before combining."
            )
    return reference


def combine_healpix(
    models: list[SkyModel],
    ref_nside: int,
    ref_freqs: np.ndarray,
    ref_frequency: float | None,
    brightness_conversion: str = "planck",
    precision: PrecisionConfig | None = None,
    memmap_path: str | None = None,
) -> CombineHealpixData:
    """Combine models by element-wise addition in Jy space per frequency channel.

    All ``healpix_map`` models must share the same nside and frequency
    grid.  Point-source models are binned into the same grid via
    ``np.bincount``.  Stokes I is converted T_b -> Jy -> T_b so that
    addition is physically correct under both Planck and Rayleigh-Jeans
    conversions.

    Parameters
    ----------
    models : list of SkyModel
        Models to combine.
    ref_nside : int
        Common HEALPix nside (from first ``healpix_map`` model).
    ref_freqs : np.ndarray
        Common frequency grid in Hz, shape ``(n_freq,)``.
    ref_frequency : float or None
        Reference frequency for spectral extrapolation of point sources.
    brightness_conversion : str, default ``"planck"``
        Brightness conversion method.
    precision : PrecisionConfig, optional
        Precision configuration.  Controls the output dtype of HEALPix
        arrays via ``precision.sky_model.get_dtype("healpix_maps")``.

    Returns
    -------
    dict
        Raw data dict with keys: ``_healpix_maps`` (``np.ndarray`` of
        shape ``(n_freq, npix)``), ``_healpix_q_maps``,
        ``_healpix_u_maps``, ``_healpix_v_maps`` (same shape or
        ``None``), ``_healpix_nside`` (int),
        ``_observation_frequencies`` (``np.ndarray``), ``frequency``
        (``float | None``).

    Raises
    ------
    ValueError
        If HEALPix models have mismatched nside or frequency grids.
    """
    # Validate all healpix_map models share the same nside and
    # frequency grid before doing element-wise arithmetic.
    healpix_models = [m for m in models if m.healpix is not None]
    coordinate_frame = _resolve_common_healpix_frame(healpix_models)
    point_only_models = [
        m
        for m in models
        if m.healpix is None and m.point is not None and not m.point.is_empty
    ]

    for m in healpix_models:
        if m.healpix is None:
            continue
        m_nside = m.healpix.nside
        m_freqs = m.healpix.frequencies
        if m_nside != ref_nside:
            raise ValueError(
                f"Cannot combine HEALPix models with different nside values: "
                f"reference has nside={ref_nside}, model '{m.model_name}' has "
                f"nside={m_nside}. Regrid one of the models first with "
                f"`regrid_healpix_model(model, nside=...)` before combining."
            )
        if not np.array_equal(m_freqs, ref_freqs):
            raise ValueError(
                f"Cannot combine HEALPix models with different frequency grids: "
                f"reference has {_format_healpix_freq_grid(ref_freqs)}, "
                f"model '{m.model_name}' has {_format_healpix_freq_grid(m_freqs)}. "
                "Frequency interpolation is not implemented; align the grids "
                "exactly before combining."
            )

    channel_widths_hz = _resolve_combined_channel_widths(healpix_models)

    npix = hp.nside2npix(ref_nside)
    n_freq = len(ref_freqs)
    omega_pixel = 4 * np.pi / npix

    # Collect point-source data for pixel-binning. Each entry carries the
    # pre-computed pixel indices, a stable handle to the PointSourceData, the
    # optional PointSpectrum table, and the per-source reference frequency
    # (with fallback to the model-level / call-level frequency).
    ps_models_data: list[
        tuple[np.ndarray, PointSourceData, PointSpectrum | None, np.ndarray]
    ] = []
    for m in point_only_models:
        if m.has_point_sources and m.point is not None:
            ipix_m = _point_source_healpix_indices(
                m.point,
                ref_nside,
                coordinate_frame=coordinate_frame,
            )
            ps_ref_freq = per_source_reference_frequencies(
                m.point,
                model_reference_frequency=m.reference_frequency,
                fallback=ref_frequency,
            )
            ps_models_data.append((ipix_m, m.point, m.point.spectrum, ps_ref_freq))

    # Check if any model has polarized maps. A populated PointSpectrum.stokes_*
    # axis is enough to require IQUV output even when the reference Stokes
    # arrays are all zero.
    any_pol = any(m.has_polarized_healpix_maps for m in healpix_models) or any(
        m.point is not None
        and (
            np.any(m.point.stokes_q != 0)
            or np.any(m.point.stokes_u != 0)
            or np.any(m.point.stokes_v != 0)
            or (
                m.point.spectrum is not None
                and (
                    m.point.spectrum.stokes_q is not None
                    or m.point.spectrum.stokes_u is not None
                    or m.point.spectrum.stokes_v is not None
                )
            )
        )
        for m in point_only_models
    )

    # Resolve output dtype from precision config
    hp_dtype = (
        precision.sky_model.get_dtype("healpix_maps")
        if precision is not None
        else np.float32
    )

    scratch = ensure_scratch_dir(memmap_path) if memmap_path is not None else None
    combined_I = allocate_cube((n_freq, npix), hp_dtype, scratch, "i_maps")
    combined_Q: np.ndarray | None = (
        allocate_cube((n_freq, npix), hp_dtype, scratch, "q_maps") if any_pol else None
    )
    combined_U: np.ndarray | None = (
        allocate_cube((n_freq, npix), hp_dtype, scratch, "u_maps") if any_pol else None
    )
    combined_V: np.ndarray | None = (
        allocate_cube((n_freq, npix), hp_dtype, scratch, "v_maps") if any_pol else None
    )

    # Determine if we can use the RJ fast path (T_b linearly additive)
    is_rj = brightness_conversion == BrightnessConversion.RAYLEIGH_JEANS or (
        isinstance(brightness_conversion, str)
        and brightness_conversion == "rayleigh-jeans"
    )

    for freq_idx, freq_hz in enumerate(ref_freqs):
        rj_factor = rayleigh_jeans_factor(freq_hz, omega_pixel)

        if is_rj:
            # --- RJ fast path: T_b is linearly additive ---
            combined_T_b = np.zeros(npix, dtype=np.float64)

            # Add healpix T_b maps directly
            for m in healpix_models:
                if m.healpix is not None:
                    pixel_indices = m.healpix.pixel_indices
                    combined_T_b[pixel_indices] += m.healpix.maps[freq_idx].astype(
                        np.float64
                    )

            # Polarization buffers allocated up-front so the I and Q/U/V
            # contributions for a given point model are produced in a single
            # call to _point_contributions_at_freq.
            combined_q_T = np.zeros(npix, dtype=np.float64) if any_pol else None
            combined_u_T = np.zeros(npix, dtype=np.float64) if any_pol else None
            combined_v_T = np.zeros(npix, dtype=np.float64) if any_pol else None

            # Add point-source contributions (flux → T_b via RJ factor),
            # using PointSpectrum per-channel lookup when available.
            rj_inv = 1.0 / rj_factor if rj_factor != 0 else 0.0
            for ipix_m, point_m, spectrum_m, ps_ref_freq in ps_models_data:
                i_map, q_map, u_map, v_map = _point_contributions_at_freq(
                    ipix_m,
                    point_m,
                    spectrum_m,
                    ps_ref_freq,
                    float(freq_hz),
                    npix,
                )
                # Jy → K_RJ: divide by RJ factor
                combined_T_b += i_map * rj_inv
                if any_pol:
                    combined_q_T += q_map * rj_inv
                    combined_u_T += u_map * rj_inv
                    combined_v_T += v_map * rj_inv

            combined_I[freq_idx] = combined_T_b.astype(hp_dtype)

            # Add polarized HEALPix contributions (already in K_RJ).
            if any_pol:
                for m in healpix_models:
                    if m.has_polarized_healpix_maps:
                        pixel_indices = m.healpix.pixel_indices
                        if m.healpix.q_maps is not None:
                            combined_q_T[pixel_indices] += m.healpix.q_maps[
                                freq_idx
                            ].astype(np.float64)
                        if m.healpix.u_maps is not None:
                            combined_u_T[pixel_indices] += m.healpix.u_maps[
                                freq_idx
                            ].astype(np.float64)
                        if m.healpix.v_maps is not None:
                            combined_v_T[pixel_indices] += m.healpix.v_maps[
                                freq_idx
                            ].astype(np.float64)

                combined_Q[freq_idx] = combined_q_T.astype(hp_dtype)
                combined_U[freq_idx] = combined_u_T.astype(hp_dtype)
                combined_V[freq_idx] = combined_v_T.astype(hp_dtype)

        else:
            # --- Planck path: must round-trip through Jy (non-linear) ---
            combined_flux = np.zeros(npix, dtype=np.float64)
            combined_q_flux = np.zeros(npix, dtype=np.float64) if any_pol else None
            combined_u_flux = np.zeros(npix, dtype=np.float64) if any_pol else None
            combined_v_flux = np.zeros(npix, dtype=np.float64) if any_pol else None

            # Add healpix_map models
            for m in healpix_models:
                if m.healpix is not None:
                    t_map = m.healpix.maps[freq_idx].astype(np.float64)
                    pixel_indices = m.healpix.pixel_indices
                    pos = t_map > 0
                    if np.any(pos):
                        combined_flux[pixel_indices[pos]] += (
                            brightness_temp_to_flux_density(
                                t_map[pos],
                                freq_hz,
                                omega_pixel,
                                method=brightness_conversion,
                            )
                        )

                    if any_pol and m.has_polarized_healpix_maps:
                        if m.healpix.q_maps is not None:
                            q_t = m.healpix.q_maps[freq_idx]
                            if q_t is not None:
                                combined_q_flux[pixel_indices] += (
                                    q_t.astype(np.float64) * rj_factor
                                )
                        if m.healpix.u_maps is not None:
                            u_t = m.healpix.u_maps[freq_idx]
                            if u_t is not None:
                                combined_u_flux[pixel_indices] += (
                                    u_t.astype(np.float64) * rj_factor
                                )
                        if m.healpix.v_maps is not None:
                            v_t = m.healpix.v_maps[freq_idx]
                            if v_t is not None:
                                combined_v_flux[pixel_indices] += (
                                    v_t.astype(np.float64) * rj_factor
                                )

            # Add point-source models via bincount, using PointSpectrum
            # per-channel lookup when available.
            for ipix_m, point_m, spectrum_m, ps_ref_freq in ps_models_data:
                i_map, q_map, u_map, v_map = _point_contributions_at_freq(
                    ipix_m,
                    point_m,
                    spectrum_m,
                    ps_ref_freq,
                    float(freq_hz),
                    npix,
                )
                combined_flux += i_map
                if any_pol:
                    combined_q_flux += q_map
                    combined_u_flux += u_map
                    combined_v_flux += v_map

            # Convert combined flux back to brightness temperature
            combined_T_b = np.zeros(npix, dtype=np.float64)
            pos_flux = combined_flux > 0
            if np.any(pos_flux):
                combined_T_b[pos_flux] = flux_density_to_brightness_temp(
                    combined_flux[pos_flux],
                    freq_hz,
                    omega_pixel,
                    method=brightness_conversion,
                )
            combined_I[freq_idx] = combined_T_b.astype(hp_dtype)

            if any_pol:
                rj_inv = 1.0 / rj_factor if rj_factor != 0 else 0.0
                combined_Q[freq_idx] = (combined_q_flux * rj_inv).astype(hp_dtype)
                combined_U[freq_idx] = (combined_u_flux * rj_inv).astype(hp_dtype)
                combined_V[freq_idx] = (combined_v_flux * rj_inv).astype(hp_dtype)

    logger.info(
        f"Combined {len(models)} models into healpix_map "
        f"({n_freq} channels, nside={ref_nside}"
        f"{', stokes=IQUV' if any_pol else ''})"
    )

    # Flush and re-open read-only if memmap-backed.
    combined_I = finalize_cube(combined_I, scratch, "i_maps")
    if combined_Q is not None:
        combined_Q = finalize_cube(combined_Q, scratch, "q_maps")
    if combined_U is not None:
        combined_U = finalize_cube(combined_U, scratch, "u_maps")
    if combined_V is not None:
        combined_V = finalize_cube(combined_V, scratch, "v_maps")

    return {
        "healpix_maps": combined_I,
        "healpix_q_maps": combined_Q,
        "healpix_u_maps": combined_U,
        "healpix_v_maps": combined_V,
        "healpix_nside": ref_nside,
        "observation_frequencies": ref_freqs,
        "channel_widths_hz": channel_widths_hz,
        "coordinate_frame": coordinate_frame,
        "reference_frequency": None,
    }
