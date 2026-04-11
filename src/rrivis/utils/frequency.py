# rrivis/utils/frequency.py
"""Frequency configuration parsing utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def parse_frequency_config(obs_frequency_config: dict[str, Any]) -> np.ndarray:
    """Parse observation frequency configuration to array of frequencies in Hz.

    Parameters
    ----------
    obs_frequency_config : dict
        Observation frequency configuration with keys:
        - starting_frequency: float
        - frequency_interval: float (channel width)
        - frequency_bandwidth: float (total bandwidth)
        - frequency_unit: str ("Hz", "kHz", "MHz", "GHz")

        If the dict contains a ``frequencies_hz`` key (a list or array of
        frequencies in Hz), that array is returned directly.  This is used
        by the programmatic ``Simulator(frequencies=[...])`` API to
        preserve non-uniform channel spacing.

    Returns
    -------
    frequencies : np.ndarray
        Array of frequency channel centers in Hz.

    Examples
    --------
    >>> config = {
    ...     "starting_frequency": 100.0,
    ...     "frequency_interval": 1.0,
    ...     "frequency_bandwidth": 20.0,
    ...     "frequency_unit": "MHz",
    ... }
    >>> freqs = parse_frequency_config(config)
    >>> len(freqs)  # 21 channels (inclusive of both endpoints)
    21
    >>> freqs[0] / 1e6  # First channel at 100 MHz
    100.0
    """
    # Fast path: raw frequency array provided by the programmatic API
    if "frequencies_hz" in obs_frequency_config:
        freqs = np.asarray(obs_frequency_config["frequencies_hz"], dtype=np.float64)
        logger.debug(f"Using raw frequency array: {len(freqs)} channels")
        return freqs

    start_freq = obs_frequency_config["starting_frequency"]
    interval = obs_frequency_config["frequency_interval"]
    bandwidth = obs_frequency_config["frequency_bandwidth"]
    unit = obs_frequency_config.get("frequency_unit", "MHz")

    unit_factors = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
    if unit not in unit_factors:
        raise ValueError(
            f"Unknown frequency unit '{unit}'. Supported: {list(unit_factors.keys())}"
        )

    unit_factor = unit_factors[unit]
    start_hz = start_freq * unit_factor
    interval_hz = interval * unit_factor
    bandwidth_hz = bandwidth * unit_factor

    if bandwidth_hz < interval_hz:
        raise ValueError(
            f"Invalid frequency config: bandwidth ({bandwidth} {unit}) must be "
            f"greater than or equal to interval ({interval} {unit})"
        )

    n_channels = max(1, int(bandwidth_hz / interval_hz)) + 1
    frequencies = np.linspace(start_hz, start_hz + bandwidth_hz, n_channels)

    logger.debug(
        f"Parsed frequency config: {n_channels} channels from "
        f"{start_freq} to {start_freq + bandwidth} {unit}"
    )

    return frequencies
