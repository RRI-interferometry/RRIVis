"""HEALPix map combination — one accumulation loop, two brightness strategies.

Both the Rayleigh-Jeans (linear, T_b additive) and Planck (non-linear, must
round-trip through Jy) cases share a single per-channel accumulation loop. The
only thing that differs between them is *how a diffuse Stokes-I map enters the
accumulator*, *how the accumulator is read back as brightness temperature*, and
the working space (T_b vs Jy). Those three operations are bundled into a
:class:`_BrightnessStrategy`; the loop is written once.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import healpy as hp
import numpy as np

from ..containers import PointSourceData, PointSpectrum
from ..containers.constants import (
    BrightnessConversion,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
    rayleigh_jeans_factor,
)
from ..containers.spectral import (
    evaluate_point_flux_at_freq,
    per_source_reference_frequencies,
)
from ..support.allocation import allocate_cube, ensure_scratch_dir, finalize_cube
from ..support.backend_helpers import maybe_asarray
from ..support.healpix_geometry import pixel_solid_angle
from ..support.precision import get_sky_storage_dtype
from .regrid import (
    _format_healpix_freq_grid,
    _point_source_healpix_indices,
    _resolve_common_healpix_frame,
)

if TYPE_CHECKING:
    from radiosim.backends import ArrayBackend
    from radiosim.core.precision import PrecisionConfig

    from ..containers.model import SkyModel


def _point_contributions_at_freq(
    ipix: np.ndarray,
    point: PointSourceData,
    spectrum: PointSpectrum | None,
    ref_freq: float | np.ndarray,
    freq_hz: float,
    npix: int,
    backend: ArrayBackend | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin a point-source model's I/Q/U/V into HEALPix flux-density (Jy) maps.

    Uses the per-channel ``PointSpectrum`` table when populated (lossless
    nearest-channel lookup); otherwise falls back to power-law extrapolation
    plus Faraday rotation on the reference Stokes arrays.  Mirrors the
    behaviour of :func:`evaluate_point_flux_at_freq`, then accumulates by
    pixel via :func:`np.bincount`.
    """
    xp = np if backend is None else backend.xp
    stokes_i = maybe_asarray(backend, point.flux)
    stokes_q = maybe_asarray(backend, point.stokes_q)
    stokes_u = maybe_asarray(backend, point.stokes_u)
    stokes_v = maybe_asarray(backend, point.stokes_v)
    spectral_index = maybe_asarray(backend, point.spectral_index)
    spectral_coeffs = maybe_asarray(backend, point.spectral_coeffs)
    ref_freq_eval = maybe_asarray(backend, ref_freq)
    rotation_measure = (
        point.polarization.rotation_measure if point.polarization is not None else None
    )
    rotation_measure = maybe_asarray(backend, rotation_measure)
    per_channel_flux = maybe_asarray(
        backend, None if spectrum is None else spectrum.flux
    )
    per_channel_stokes_q = maybe_asarray(
        backend, spectrum.stokes_q if spectrum is not None else None
    )
    per_channel_stokes_u = maybe_asarray(
        backend, spectrum.stokes_u if spectrum is not None else None
    )
    per_channel_stokes_v = maybe_asarray(
        backend, spectrum.stokes_v if spectrum is not None else None
    )
    i_f, q_f, u_f, v_f = evaluate_point_flux_at_freq(
        stokes_i=stokes_i,
        stokes_q=stokes_q,
        stokes_u=stokes_u,
        stokes_v=stokes_v,
        spectral_index=spectral_index,
        spectral_coeffs=spectral_coeffs,
        ref_freq=ref_freq_eval,
        rotation_measure=rotation_measure,
        per_channel_flux=per_channel_flux,
        per_channel_stokes_q=per_channel_stokes_q,
        per_channel_stokes_u=per_channel_stokes_u,
        per_channel_stokes_v=per_channel_stokes_v,
        channel_frequencies=spectrum.frequencies if spectrum is not None else None,
        freq=freq_hz,
        xp=xp,
    )
    bincount = np.bincount if backend is None else backend.bincount
    return (
        bincount(ipix, weights=i_f, minlength=npix),
        bincount(ipix, weights=q_f, minlength=npix),
        bincount(ipix, weights=u_f, minlength=npix),
        bincount(ipix, weights=v_f, minlength=npix),
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
    coordinate_frame: str
    reference_frequency: float | None


@dataclass(frozen=True)
class _BrightnessStrategy:
    """Per-channel accumulation policy for one brightness convention.

    Both the Stokes-I and the polarization accumulators work in *brightness
    temperature* (Kelvin) for the Rayleigh-Jeans case (T_b is linearly additive)
    and in *flux density* (Jy) for the Planck case (Stokes I must round-trip
    through Jy because the T_b ↔ Jy map is non-linear; polarization shares the
    same Jy working space). Keeping the RJ path entirely in Kelvin makes its
    output bit-exact (no ``rj_factor * 1/rj_factor`` round-trip noise).

    Attributes
    ----------
    diffuse_i_to_accum : callable
        Map a diffuse Stokes-I brightness-temperature array (K) onto the I
        accumulator's working units, given ``(t_map, freq_hz, omega_pixel,
        rj_factor)``.
    point_i_scale : callable
        Given the RJ factor, return the multiplier applied to a point source's
        binned Stokes-I flux (Jy) before adding it to the I accumulator
        (``1`` for the Jy-space Planck path, ``1/rj_factor`` for the K-space
        RJ path).
    accum_i_to_tb : callable
        Convert the finished I accumulator back to brightness temperature (K),
        given ``(accum, freq_hz, omega_pixel, rj_factor)``.
    diffuse_pol_scale : callable
        Given the RJ factor, return the multiplier applied to a diffuse
        polarized map (K_RJ) before adding it to the pol accumulator (``1`` for
        the K-space RJ path, ``rj_factor`` for the Jy-space Planck path).
    point_pol_scale : callable
        Given the RJ factor, return the multiplier applied to a point source's
        binned polarized flux (Jy) before adding it to the pol accumulator
        (``1/rj_factor`` for the K-space RJ path, ``1`` for the Jy-space Planck
        path).
    pol_accum_to_k : callable
        Given the RJ factor, return the multiplier converting the finished pol
        accumulator back to K_RJ (``1`` for the K-space RJ path, ``1/rj_factor``
        for the Jy-space Planck path).
    """

    diffuse_i_to_accum: Callable[[np.ndarray, float, float, float], np.ndarray]
    point_i_scale: Callable[[float], float]
    accum_i_to_tb: Callable[[np.ndarray, float, float, float], np.ndarray]
    diffuse_pol_scale: Callable[[float], float]
    point_pol_scale: Callable[[float], float]
    pol_accum_to_k: Callable[[float], float]


def _rj_inv(rj_factor: float) -> float:
    return 1.0 / rj_factor if rj_factor != 0 else 0.0


def _make_strategy(brightness_conversion: str) -> _BrightnessStrategy:
    """Return the accumulation strategy for the requested conversion."""
    is_rj = brightness_conversion == BrightnessConversion.RAYLEIGH_JEANS or (
        isinstance(brightness_conversion, str)
        and brightness_conversion == "rayleigh-jeans"
    )
    if is_rj:
        # RJ: accumulate directly in T_b (Kelvin). Diffuse maps pass through;
        # point flux (Jy) is divided by the RJ factor; the accumulator already
        # *is* brightness temperature.
        return _BrightnessStrategy(
            diffuse_i_to_accum=lambda t_map, freq, omega, rj: t_map,
            point_i_scale=_rj_inv,
            accum_i_to_tb=lambda accum, freq, omega, rj: accum,
            # Polarization stays in K_RJ: diffuse maps pass through, point flux
            # (Jy) is divided by the RJ factor, output is already K.
            diffuse_pol_scale=lambda rj: 1.0,
            point_pol_scale=_rj_inv,
            pol_accum_to_k=lambda rj: 1.0,
        )

    # Planck: accumulate in Jy. Diffuse maps convert T_b → Jy (positive pixels
    # only); point flux (Jy) is added unscaled; the accumulator converts Jy → T_b
    # (positive pixels only) for output.
    def _diffuse_to_jy(
        t_map: np.ndarray, freq: float, omega: float, rj: float
    ) -> np.ndarray:
        out = np.zeros_like(t_map)
        pos = t_map > 0
        if np.any(pos):
            out[pos] = brightness_temp_to_flux_density(
                t_map[pos], freq, omega, method=brightness_conversion
            )
        return out

    def _jy_to_tb(
        accum: np.ndarray, freq: float, omega: float, rj: float
    ) -> np.ndarray:
        out = np.zeros_like(accum)
        pos = accum > 0
        if np.any(pos):
            out[pos] = flux_density_to_brightness_temp(
                accum[pos], freq, omega, method=brightness_conversion
            )
        return out

    return _BrightnessStrategy(
        diffuse_i_to_accum=_diffuse_to_jy,
        point_i_scale=lambda rj: 1.0,
        accum_i_to_tb=_jy_to_tb,
        # Polarization works in Jy: diffuse maps (K_RJ) convert via the RJ
        # factor, point flux (Jy) is unscaled, output divides back to K_RJ.
        diffuse_pol_scale=lambda rj: rj,
        point_pol_scale=lambda rj: 1.0,
        pol_accum_to_k=_rj_inv,
    )


def combine_healpix(
    models: list[SkyModel],
    ref_nside: int,
    ref_freqs: np.ndarray,
    ref_frequency: float | None,
    brightness_conversion: str = "planck",
    precision: PrecisionConfig | None = None,
    memmap_path: str | None = None,
    backend: ArrayBackend | None = None,
) -> CombineHealpixData:
    """Combine models by element-wise addition in Jy space per frequency channel.

    All ``healpix_map`` models must share the same nside and frequency
    grid.  Point-source models are binned into the same grid via
    ``np.bincount``.  Stokes I is accumulated either directly in brightness
    temperature (Rayleigh-Jeans, linear) or via a Jy round-trip (Planck,
    non-linear) so that addition is physically correct under both conventions.

    Parameters
    ----------
    models : list of SkyModel
        Models to combine.
    ref_nside : int
        Common HEALPix nside (from first ``healpix_map`` model).
    ref_freqs : np.ndarray
        Common frequency grid in Hz, shape ``(n_freq,)``.
    ref_frequency : float or None
        Reference frequency for spectral extrapolation of point sources; also
        surfaced in the returned ``reference_frequency``.
    brightness_conversion : str, default ``"planck"``
        Brightness conversion method.
    precision : PrecisionConfig, optional
        Precision configuration.  Controls the output dtype of HEALPix
        arrays via the shared ``get_sky_storage_dtype`` helper.

    Returns
    -------
    dict
        Raw data dict with keys: ``healpix_maps`` (``np.ndarray`` of shape
        ``(n_freq, npix)``), ``healpix_q_maps``, ``healpix_u_maps``,
        ``healpix_v_maps`` (same shape or ``None``), ``healpix_nside`` (int),
        ``observation_frequencies`` (``np.ndarray``), ``coordinate_frame``
        (str), and ``reference_frequency`` (``float | None``).

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

    npix = hp.nside2npix(ref_nside)
    n_freq = len(ref_freqs)
    omega_pixel = pixel_solid_angle(ref_nside)

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

    # Resolve output dtype from precision config (single source of truth).
    hp_dtype = get_sky_storage_dtype(precision, "healpix_maps")

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

    strategy = _make_strategy(brightness_conversion)

    for freq_idx, freq_hz in enumerate(ref_freqs):
        freq = float(freq_hz)
        rj_factor = rayleigh_jeans_factor(freq, omega_pixel)
        point_i_scale = strategy.point_i_scale(rj_factor)
        diffuse_pol_scale = strategy.diffuse_pol_scale(rj_factor)
        point_pol_scale = strategy.point_pol_scale(rj_factor)

        # The accumulators work in the strategy's units: Kelvin for the RJ path
        # (bit-exact, no round-trip) and Jy for the Planck path.
        accum_i = np.zeros(npix, dtype=np.float64)
        accum_q = np.zeros(npix, dtype=np.float64) if any_pol else None
        accum_u = np.zeros(npix, dtype=np.float64) if any_pol else None
        accum_v = np.zeros(npix, dtype=np.float64) if any_pol else None

        # --- Diffuse HEALPix maps ---
        for m in healpix_models:
            if m.healpix is None:
                continue
            pixel_indices = m.healpix.pixel_indices
            t_map = m.healpix.maps[freq_idx].astype(np.float64)
            accum_i[pixel_indices] += strategy.diffuse_i_to_accum(
                t_map, freq, omega_pixel, rj_factor
            )
            if any_pol and m.has_polarized_healpix_maps:
                # Polarized diffuse maps are in K_RJ; the strategy scales them
                # into the pol accumulator's working units (K for RJ, Jy for
                # Planck).
                if m.healpix.q_maps is not None:
                    accum_q[pixel_indices] += (
                        m.healpix.q_maps[freq_idx].astype(np.float64)
                        * diffuse_pol_scale
                    )
                if m.healpix.u_maps is not None:
                    accum_u[pixel_indices] += (
                        m.healpix.u_maps[freq_idx].astype(np.float64)
                        * diffuse_pol_scale
                    )
                if m.healpix.v_maps is not None:
                    accum_v[pixel_indices] += (
                        m.healpix.v_maps[freq_idx].astype(np.float64)
                        * diffuse_pol_scale
                    )

        # --- Point-source contributions (binned flux in Jy) ---
        for ipix_m, point_m, spectrum_m, ps_ref_freq in ps_models_data:
            i_map, q_map, u_map, v_map = _point_contributions_at_freq(
                ipix_m,
                point_m,
                spectrum_m,
                ps_ref_freq,
                freq,
                npix,
                backend=backend,
            )
            if backend is not None:
                i_map = backend.to_numpy(i_map)
                q_map = backend.to_numpy(q_map)
                u_map = backend.to_numpy(u_map)
                v_map = backend.to_numpy(v_map)
            accum_i += i_map * point_i_scale
            if any_pol:
                accum_q += q_map * point_pol_scale
                accum_u += u_map * point_pol_scale
                accum_v += v_map * point_pol_scale

        # --- Read back Stokes I as brightness temperature ---
        combined_I[freq_idx] = strategy.accum_i_to_tb(
            accum_i, freq, omega_pixel, rj_factor
        ).astype(hp_dtype)

        # --- Polarization: accumulator → K_RJ (identity for RJ, ÷rj for Planck)
        if any_pol:
            pol_to_k = strategy.pol_accum_to_k(rj_factor)
            combined_Q[freq_idx] = (accum_q * pol_to_k).astype(hp_dtype)
            combined_U[freq_idx] = (accum_u * pol_to_k).astype(hp_dtype)
            combined_V[freq_idx] = (accum_v * pol_to_k).astype(hp_dtype)

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
        "coordinate_frame": coordinate_frame,
        "reference_frequency": ref_frequency,
    }
