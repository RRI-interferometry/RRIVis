# rrivis/core/sky/spectral.py
"""Spectral scaling and Faraday rotation helpers for sky models.

These functions are used by both the sky model conversion code and the
visibility calculation engine (``visibility.py``).
"""

from __future__ import annotations

import numpy as np

from .constants import C_LIGHT


def compute_spectral_scale(
    alpha: np.ndarray,
    spectral_coeffs: np.ndarray | None,
    freq: float,
    ref_freq: float | np.ndarray,
) -> np.ndarray:
    """Compute frequency scaling factor for each source.

    Uses log-polynomial when *spectral_coeffs* has >1 term, else simple
    power law ``(freq / ref_freq) ** alpha``.

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

    Returns
    -------
    np.ndarray
        Multiplicative scaling factor, shape ``(N,)``.
    """
    ratio = freq / ref_freq
    # Guard against invalid ref_freq (zero, negative, or NaN — e.g. test
    # sources or mixed-catalog models where some sources lack ref_freq).
    # Scale factor = 1.0 for those sources (no spectral extrapolation).
    if isinstance(ref_freq, np.ndarray):
        bad = ~(ref_freq > 0)  # catches <= 0 and NaN
        if np.any(bad):
            ratio = np.where(bad, 1.0, ratio)
    elif not (ref_freq > 0):  # catches <= 0 and NaN
        return np.ones_like(alpha)
    if spectral_coeffs is not None and spectral_coeffs.shape[1] > 1:
        log_ratio = np.log10(ratio)
        exponent = np.zeros(len(alpha), dtype=np.float64)
        for k in range(spectral_coeffs.shape[1]):
            exponent += spectral_coeffs[:, k] * log_ratio**k
        return ratio**exponent
    return ratio**alpha


def apply_faraday_rotation(
    q_ref: np.ndarray,
    u_ref: np.ndarray,
    rm: np.ndarray | None,
    freq: float,
    ref_freq: float | np.ndarray,
    spectral_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply spectral scaling and Faraday rotation to Q/U arrays.

    When *rm* is ``None`` or all-zero, this reduces to simple power-law
    scaling (identical to the pre-RM behaviour).

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

    Returns
    -------
    q_out, u_out : np.ndarray
        Scaled (and optionally Faraday-rotated) Stokes Q, U.
    """
    q_scaled = q_ref * spectral_scale
    u_scaled = u_ref * spectral_scale
    if rm is not None and np.any(rm != 0):
        # Guard against invalid ref_freq (zero, negative, NaN) which would
        # produce inf/nan in (C_LIGHT / ref_freq)**2.  For those sources
        # substitute freq so that delta_chi = 0 (no Faraday rotation).
        safe_ref_freq = ref_freq
        if isinstance(ref_freq, np.ndarray):
            bad = ~(ref_freq > 0)  # catches <= 0 and NaN
            if np.any(bad):
                safe_ref_freq = np.where(bad, freq, ref_freq)
        elif not (ref_freq > 0):
            return q_scaled, u_scaled
        delta_chi = rm * ((C_LIGHT / freq) ** 2 - (C_LIGHT / safe_ref_freq) ** 2)
        cos2 = np.cos(2.0 * delta_chi)
        sin2 = np.sin(2.0 * delta_chi)
        q_out = q_scaled * cos2 - u_scaled * sin2
        u_out = q_scaled * sin2 + u_scaled * cos2
        return q_out, u_out
    return q_scaled, u_scaled


def nearest_channel_index(channel_frequencies: np.ndarray, freq: float) -> int:
    """Return the index of the channel nearest to ``freq`` (ties broken low-side).

    Parameters
    ----------
    channel_frequencies : np.ndarray
        Strictly ascending channel frequency array in Hz.
    freq : float
        Observation frequency in Hz.

    Returns
    -------
    int
        Index of the channel closest to ``freq``.
    """
    return int(np.argmin(np.abs(channel_frequencies - freq)))


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
        Reference-frequency Stokes arrays, shape ``(N,)``. Used on the
        extrapolation path and also as the Q/U/V fallback when
        per-channel Q/U/V tables are absent.
    spectral_index, spectral_coeffs : np.ndarray, np.ndarray or None
        Spectral extrapolation inputs for the fallback path.
    ref_freq : float or np.ndarray
        Reference frequency (Hz) for spectral extrapolation and Faraday rotation.
    rotation_measure : np.ndarray or None
        Per-source rotation measure in rad/m^2 (fallback path only).
    per_channel_flux : np.ndarray or None
        Shape ``(n_channel, N)``. When present, selects the short-circuit path.
    per_channel_stokes_q/u/v : np.ndarray or None
        Per-channel polarization tables. Q and U are paired; V independent.
    channel_frequencies : np.ndarray or None
        Channel grid matching the first axis of ``per_channel_flux``.
    freq : float
        Observation frequency (Hz).

    Returns
    -------
    (I, Q, U, V) : tuple of np.ndarray
        Stokes values at the observation frequency, shape ``(N,)`` each.
    """
    if per_channel_flux is not None and channel_frequencies is not None:
        idx = nearest_channel_index(channel_frequencies, freq)
        i_out = per_channel_flux[idx].astype(np.float64, copy=False)
        if per_channel_stokes_q is not None and per_channel_stokes_u is not None:
            q_out = per_channel_stokes_q[idx].astype(np.float64, copy=False)
            u_out = per_channel_stokes_u[idx].astype(np.float64, copy=False)
        else:
            q_out = stokes_q.astype(np.float64, copy=False)
            u_out = stokes_u.astype(np.float64, copy=False)
        if per_channel_stokes_v is not None:
            v_out = per_channel_stokes_v[idx].astype(np.float64, copy=False)
        else:
            v_out = stokes_v.astype(np.float64, copy=False)
        return i_out, q_out, u_out, v_out

    scale = compute_spectral_scale(spectral_index, spectral_coeffs, freq, ref_freq)
    i_out = stokes_i * scale
    q_out, u_out = apply_faraday_rotation(
        stokes_q, stokes_u, rotation_measure, freq, ref_freq, scale
    )
    v_out = stokes_v * scale
    return i_out, q_out, u_out, v_out
