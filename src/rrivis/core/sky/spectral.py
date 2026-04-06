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
    ref_freq: float,
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
    freq, ref_freq : float
        Observation and reference frequencies in Hz.

    Returns
    -------
    np.ndarray
        Multiplicative scaling factor, shape ``(N,)``.
    """
    ratio = freq / ref_freq
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
    ref_freq: float,
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
    freq, ref_freq : float
        Observation and reference frequencies in Hz.
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
        delta_chi = rm * ((C_LIGHT / freq) ** 2 - (C_LIGHT / ref_freq) ** 2)
        cos2 = np.cos(2.0 * delta_chi)
        sin2 = np.sin(2.0 * delta_chi)
        q_out = q_scaled * cos2 - u_scaled * sin2
        u_out = q_scaled * sin2 + u_scaled * cos2
        return q_out, u_out
    return q_scaled, u_scaled
