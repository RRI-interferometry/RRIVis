"""Spectral scaling and Faraday rotation helpers for sky models.

These functions are used by both the sky-model conversion code and the
visibility calculation engine (``visibility.py``). The visibility hot
loop calls :func:`evaluate_point_flux_at_freq` once per
``(timestep × frequency)`` step, so each function accepts an ``xp``
keyword (default ``numpy``) that lets a JAX or Numba backend dispatch
the array math on-device. Data-dependent control flow has been
rewritten as branchless masks so the bodies are JAX-traceable.

``np.<name>`` is still used for dtype literals and for warning side
effects (warnings stay numpy-only — they fire once during tracing on
CPU, not on every per-frequency call).

Spectral representation introspection (which of power-law / log-polynomial /
per-channel is populated on a :class:`~.point.PointSourceData` payload) lives
on the container: :attr:`~.point.PointSourceData.populated_spectral_fields` and
:meth:`~.point.PointSourceData.assert_single_spectral_representation`. These
helpers only evaluate flux at runtime and do not enforce representation
exclusivity.
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from .constants import C_LIGHT

if TYPE_CHECKING:
    from .point import PointSourceData

logger = logging.getLogger(__name__)


def per_source_reference_frequencies(
    point: PointSourceData,
    *,
    model_reference_frequency: float | None = None,
    fallback: float | None = None,
) -> np.ndarray:
    """Resolve a per-source reference-frequency array of shape ``(n_sources,)``.

    Resolution order, first match wins:

    1. ``point.ref_freq`` cast to float64 — when it is not None and contains
       at least one positive value.
    2. ``model_reference_frequency`` if truthy (broadcast to every source).
    3. ``fallback`` if truthy.
    4. ``0.0`` — downstream spectral code (``compute_spectral_scale``,
       ``apply_faraday_rotation``) treats values <= 0 as "no extrapolation".

    Numpy-only by design — runs once per simulation start, never in the
    visibility hot loop.
    """
    if point.ref_freq is not None and np.any(point.ref_freq > 0):
        return point.ref_freq.astype(np.float64, copy=False)
    fill = model_reference_frequency or fallback or 0.0
    return np.full(point.n_sources, float(fill), dtype=np.float64)


def compute_spectral_scale(
    alpha: np.ndarray,
    spectral_coeffs: np.ndarray | None,
    freq: float,
    ref_freq: float | np.ndarray,
    *,
    xp: ModuleType = np,
) -> np.ndarray:
    """Compute frequency scaling factor for each source.

    Uses log-polynomial when *spectral_coeffs* has >1 term, else simple
    power law ``(freq / ref_freq) ** alpha``.

    The log-polynomial form follows the standard radio convention
    (e.g. Remazeilles, Dickinson & Banday 2015 eqn. 7):

    .. math::

       S(\\nu) = S_0 \\cdot 10^{\\sum_k \\alpha_k (\\log_{10} r)^{k+1}},
       \\quad r = \\nu / \\nu_0.

    Parameters
    ----------
    alpha : np.ndarray
        Simple spectral index array, shape ``(N,)``.
    spectral_coeffs : np.ndarray or None
        Log-polynomial coefficients, shape ``(N, N_terms)``.  Column 0 is
        the simple spectral index.  ``None`` => use *alpha* only.
    freq : float
        Observation frequency in Hz.
    ref_freq : float or np.ndarray
        Reference frequency in Hz. Can be a scalar (shared by all sources)
        or a per-source array of shape ``(N,)``.
    xp : module
        Array namespace (numpy by default; pass ``backend.xp`` from the
        visibility hot path so JAX/Numba arrays stay on-device).

    Returns
    -------
    np.ndarray
        Multiplicative scaling factor, shape ``(N,)``.

    Notes
    -----
    Branchless: invalid ``ref_freq`` (zero, negative, or NaN — e.g. test
    sources or mixed-catalog models where some sources lack ``ref_freq``)
    is replaced by 1.0 in the ratio so the scale factor for those
    sources is 1.0 (no spectral extrapolation). Always evaluating the
    ``xp.where`` adds one mask op per call but keeps the function
    JAX-traceable.
    """
    valid = ref_freq > 0
    safe_ref = xp.where(valid, ref_freq, 1.0)
    ratio = xp.where(valid, freq / safe_ref, 1.0)
    if spectral_coeffs is not None and spectral_coeffs.shape[1] > 1:
        log_ratio = xp.log10(ratio)
        log_scale = xp.zeros(alpha.shape, dtype=alpha.dtype)
        for k in range(spectral_coeffs.shape[1]):
            log_scale = log_scale + spectral_coeffs[:, k] * log_ratio ** (k + 1)
        return 10.0**log_scale
    return ratio**alpha


def apply_faraday_rotation(
    q_ref: np.ndarray,
    u_ref: np.ndarray,
    rm: np.ndarray | None,
    freq: float,
    ref_freq: float | np.ndarray,
    spectral_scale: np.ndarray,
    *,
    xp: ModuleType = np,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply spectral scaling and Faraday rotation to Q/U arrays.

    When *rm* is ``None``, this reduces to simple power-law scaling.
    When *rm* is provided, the rotation is always evaluated (a zero-RM
    source incurs an identity rotation — branchless, JAX-traceable).

    Parameters
    ----------
    q_ref, u_ref : np.ndarray
        Stokes Q, U at the reference frequency (Jy), shape ``(N,)``.
    rm : np.ndarray or None
        Rotation measure in rad/m^2 per source.
    freq : float
        Observation frequency in Hz.
    ref_freq : float or np.ndarray
        Reference frequency in Hz. Scalar or per-source array ``(N,)``.
    spectral_scale : np.ndarray
        Pre-computed ``(freq / ref_freq) ** alpha`` (or log-poly), shape ``(N,)``.
    xp : module
        Array namespace (numpy by default).

    Returns
    -------
    q_out, u_out : np.ndarray
        Scaled (and optionally Faraday-rotated) Stokes Q, U.
    """
    q_scaled = q_ref * spectral_scale
    u_scaled = u_ref * spectral_scale
    if rm is None:
        return q_scaled, u_scaled
    # Guard against invalid ref_freq (zero, negative, NaN) which would
    # produce inf/nan in (C_LIGHT / ref_freq)**2. Substitute freq there
    # so that delta_chi = 0 (identity rotation) for those sources.
    valid = ref_freq > 0
    safe_ref_freq = xp.where(valid, ref_freq, freq)
    delta_chi = rm * ((C_LIGHT / freq) ** 2 - (C_LIGHT / safe_ref_freq) ** 2)
    cos2 = xp.cos(2.0 * delta_chi)
    sin2 = xp.sin(2.0 * delta_chi)
    q_out = q_scaled * cos2 - u_scaled * sin2
    u_out = q_scaled * sin2 + u_scaled * cos2
    return q_out, u_out


def nearest_channel_index(channel_frequencies: np.ndarray, freq: float) -> int:
    """Return the index of the channel nearest to ``freq`` (ties broken low-side).

    Silent variant. Use :func:`nearest_channel_index_with_warning` for
    user-facing entry points where an off-grid frequency request is
    surprising and should be flagged.
    """
    return int(np.argmin(np.abs(channel_frequencies - freq)))


def nearest_channel_index_with_warning(
    channel_frequencies: np.ndarray,
    freq: float,
    *,
    label: str | None = None,
) -> int:
    """Return the nearest-channel index and log a warning when ``freq`` is
    far from every grid point.

    Off-grid threshold is ``max(1 kHz, 0.1 * median(|diff|))`` of the channel
    grid spacing. This stays numpy-only — it fires once per simulation
    start, not once per (timestep, frequency).
    """
    freqs = np.asarray(channel_frequencies)
    idx = int(np.argmin(np.abs(freqs - freq)))
    nearest_freq = float(freqs[idx])
    diff_hz = abs(freq - nearest_freq)
    if freqs.size > 1:
        spacing_hz = float(np.median(np.diff(np.sort(freqs))))
        warn_threshold_hz = max(1_000.0, 0.1 * spacing_hz)
    else:
        warn_threshold_hz = 1_000.0
    if diff_hz > warn_threshold_hz:
        prefix = f"{label}: " if label else ""
        logger.warning(
            "%snearest-channel lookup is off-grid: requested %.6f MHz, "
            "nearest channel %.6f MHz (Δ=%.3f kHz).",
            prefix,
            freq / 1e6,
            nearest_freq / 1e6,
            diff_hz / 1e3,
        )
    return idx


def evaluate_point_flux_at_freq(
    stokes_i: np.ndarray,
    stokes_q: np.ndarray,
    stokes_u: np.ndarray,
    stokes_v: np.ndarray,
    spectral_index: np.ndarray,
    spectral_coeffs: np.ndarray | None,
    ref_freq: float | np.ndarray,
    rotation_measure: np.ndarray | None,
    per_channel_flux: np.ndarray | None,
    per_channel_stokes_q: np.ndarray | None,
    per_channel_stokes_u: np.ndarray | None,
    per_channel_stokes_v: np.ndarray | None,
    channel_frequencies: np.ndarray | None,
    freq: float,
    *,
    xp: ModuleType = np,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Stokes I/Q/U/V at a single observation frequency.

    If ``per_channel_flux`` and ``channel_frequencies`` are populated, use
    nearest-channel lookup (lossless for sampled frequencies). Per-channel
    Q/U are already observed values at that frequency, so Faraday rotation
    is **not** reapplied on the per-channel path.

    Otherwise, fall back to the spectral-index / log-polynomial
    extrapolation path, with Faraday rotation applied to Q/U.

    Parameters
    ----------
    stokes_i, stokes_q, stokes_u, stokes_v : np.ndarray
        Reference-frequency Stokes arrays, shape ``(N,)``.
    spectral_index, spectral_coeffs : np.ndarray, np.ndarray or None
        Spectral extrapolation inputs for the fallback path.
    ref_freq : float or np.ndarray
        Reference frequency (Hz).
    rotation_measure : np.ndarray or None
        Per-source rotation measure in rad/m^2 (fallback path only).
    per_channel_flux : np.ndarray or None
        Shape ``(n_channel, N)``. When present, selects the short-circuit path.
    per_channel_stokes_q/u/v : np.ndarray or None
        Per-channel polarization tables.
    channel_frequencies : np.ndarray or None
        Channel grid matching the first axis of ``per_channel_flux``.
    freq : float
        Observation frequency (Hz).
    xp : module
        Array namespace forwarded to the math primitives.

    Returns
    -------
    (I, Q, U, V) : tuple of arrays in the ``xp`` namespace.
    """
    if per_channel_flux is not None and channel_frequencies is not None:
        idx = nearest_channel_index_with_warning(
            channel_frequencies, freq, label="per-channel flux"
        )
        i_out = per_channel_flux[idx]
        if per_channel_stokes_q is not None and per_channel_stokes_u is not None:
            q_out = per_channel_stokes_q[idx]
            u_out = per_channel_stokes_u[idx]
        else:
            q_out = stokes_q
            u_out = stokes_u
        if per_channel_stokes_v is not None:
            v_out = per_channel_stokes_v[idx]
        else:
            v_out = stokes_v
        return i_out, q_out, u_out, v_out

    scale = compute_spectral_scale(
        spectral_index, spectral_coeffs, freq, ref_freq, xp=xp
    )
    i_out = stokes_i * scale
    q_out, u_out = apply_faraday_rotation(
        stokes_q, stokes_u, rotation_measure, freq, ref_freq, scale, xp=xp
    )
    v_out = stokes_v * scale
    return i_out, q_out, u_out, v_out
