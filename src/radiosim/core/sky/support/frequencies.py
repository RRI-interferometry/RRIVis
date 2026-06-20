"""Frequency-config resolution helper.

Consolidates the ``frequencies`` vs ``obs_frequency_config`` resolution
that was duplicated across ``loaders/diffuse.py`` and
``loaders/pyradiosky.py`` (spec item B6). The shared helper returns an
ascending float64 Hz array and rejects ambiguous/missing inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from radiosim.utils.frequency import parse_frequency_config


def resolve_frequency_config(
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Resolve observation frequencies to an ascending float64 Hz array.

    Exactly one of ``frequencies`` / ``obs_frequency_config`` must be
    provided. An explicit ``frequencies`` array is cast to float64; an
    ``obs_frequency_config`` dict is expanded via
    :func:`radiosim.utils.frequency.parse_frequency_config`. The result is
    sorted ascending.

    Parameters
    ----------
    frequencies : np.ndarray or None, optional
        Explicit observation frequencies in Hz.
    obs_frequency_config : dict or None, optional
        Frequency configuration dict (keys: ``starting_frequency``,
        ``frequency_interval``, ``frequency_bandwidth``, ``frequency_unit``;
        or a raw ``frequencies_hz`` array).

    Returns
    -------
    np.ndarray
        Ascending float64 array of frequencies in Hz.

    Raises
    ------
    ValueError
        If both or neither argument is provided.
    """
    if (frequencies is None) == (obs_frequency_config is None):
        raise ValueError(
            "Provide exactly one of 'frequencies' or 'obs_frequency_config' "
            "(got both or neither)."
        )

    if frequencies is not None:
        resolved = np.asarray(frequencies, dtype=np.float64)
    else:
        resolved = np.asarray(
            parse_frequency_config(obs_frequency_config), dtype=np.float64
        )

    return np.sort(resolved)
