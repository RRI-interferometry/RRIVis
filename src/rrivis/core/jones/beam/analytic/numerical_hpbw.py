"""Diagnostic HPBW finder for beam pattern visualization.

This is a **diagnostic/visualization utility** — it is NOT used in RIME
simulation computations. The RIME evaluates the full beam pattern directly
at each source position; HPBW is never used as an intermediate quantity.

Given any beam voltage pattern function, this module numerically evaluates
the pattern on a fine angular grid and interpolates to find the half-power
beam width (HPBW). This is useful for annotating beam plots and inspecting
beam patterns that lack a closed-form HPBW expression (e.g., tapered
aperture patterns, measured beams).

The algorithm works by:

1. Sampling the voltage pattern E(theta) over a fine grid.
2. Taking |E(theta)| to handle patterns with negative sidelobes.
3. Finding the first angle where |E| drops to 1/sqrt(2) of the peak
   (the voltage half-power point).
4. Returning HPBW = 2 * theta_half_power.

Examples
--------
>>> import numpy as np
>>> from rrivis.core.jones.beam.taper import uniform_taper
>>> from rrivis.core.jones.beam.numerical_hpbw import compute_hpbw_numerical
>>> hpbw = compute_hpbw_numerical(uniform_taper, 150e6, 14.0)
>>> hpbw.shape
(1,)
"""

from collections.abc import Callable

import numpy as np

_C: float = 299_792_458.0
"""Speed of light in m/s."""


def compute_hpbw_numerical(
    beam_func: Callable,
    freq_hz: float | np.ndarray,
    diameter: float,
    search_range_rad: float = 0.5,
    n_samples: int = 10000,
    **beam_kwargs,
) -> np.ndarray:
    """Find HPBW by evaluating beam pattern and interpolating to half-power.

    This function is for diagnostics and plot annotations only. The RIME
    simulation evaluates the full beam pattern directly at each source
    position — HPBW is never used as an intermediate computational quantity.

    Parameters
    ----------
    beam_func : callable
        Beam pattern function that takes u_beam as first argument and
        returns voltage pattern. May accept additional kwargs (e.g., edge_taper_dB).
    freq_hz : float or array_like
        Frequency(ies) in Hz.
    diameter : float
        Antenna diameter in meters.
    search_range_rad : float
        Maximum angle to search in radians.
    n_samples : int
        Number of sample points in the theta grid.
    **beam_kwargs
        Additional keyword arguments passed to beam_func (e.g., edge_taper_dB).

    Returns
    -------
    ndarray
        HPBW in radians, one per frequency.

    Algorithm
    ---------
    1. Create theta grid from 0 to search_range_rad
    2. For each frequency, compute u_beam = D * sin(theta) * nu / c
    3. Evaluate beam_func(u_beam, **beam_kwargs) to get voltage pattern
    4. Find theta where |E(theta)| = 1/sqrt(2) (voltage half-power) via np.interp
    5. HPBW = 2 * that theta
    """
    freqs = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    theta = np.linspace(0.0, search_range_rad, n_samples)
    sin_theta = np.sin(theta)

    half_power_level = 1.0 / np.sqrt(2.0)
    hpbw_values = np.empty(freqs.shape[0], dtype=np.float64)

    for i, nu in enumerate(freqs):
        u_beam = diameter * sin_theta * nu / _C
        voltage = beam_func(u_beam, **beam_kwargs)
        voltage_abs = np.abs(voltage)

        # Interpolate to find the first crossing of the half-power level.
        # np.interp expects xp to be increasing, so we use the decreasing
        # |E(theta)| from peak (1.0) down, mapping voltage -> theta.
        # We need the first crossing, so we only look at the main lobe
        # (before the first null / first rise after a null).

        # Find the first index where |E| drops below half-power
        below_mask = voltage_abs < half_power_level
        if np.any(below_mask):
            first_below = np.argmax(below_mask)
            # Interpolate between the sample just above and the sample at/below
            # Use the segment [0, first_below] for interpolation
            segment_theta = theta[: first_below + 1]
            segment_voltage = voltage_abs[: first_below + 1]
            # voltage is decreasing in this segment, so flip for np.interp
            # which requires xp increasing
            theta_half = np.interp(
                half_power_level,
                segment_voltage[::-1],
                segment_theta[::-1],
            )
        else:
            # Half-power point not found in search range; use the endpoint
            theta_half = search_range_rad

        hpbw_values[i] = 2.0 * theta_half

    return hpbw_values


__all__ = [
    "compute_hpbw_numerical",
]
