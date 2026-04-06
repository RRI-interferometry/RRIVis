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
    >>> len(freqs)  # 20 channels
    20
    >>> freqs[0] / 1e6  # First channel at 100 MHz
    100.0
    """
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

    n_channels = int(bandwidth_hz / interval_hz)
    if n_channels <= 0:
        raise ValueError(
            f"Invalid frequency config: bandwidth ({bandwidth} {unit}) must be "
            f"greater than interval ({interval} {unit})"
        )

    frequencies = np.array(
        [start_hz + i * interval_hz for i in range(n_channels)], dtype=np.float64
    )

    logger.debug(
        f"Parsed frequency config: {n_channels} channels from "
        f"{start_freq} to {start_freq + bandwidth - interval} {unit}"
    )

    return frequencies
