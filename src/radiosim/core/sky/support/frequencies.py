"""Explicit observation-frequency validation for lower-level sky APIs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..containers._shared import validate_frequency_axis


def validate_observation_frequencies(
    frequencies: Sequence[float] | np.ndarray,
    *,
    label: str = "observation frequencies",
) -> np.ndarray:
    """Copy and validate an explicit ordered frequency sequence in Hz.

    Parameters
    ----------
    frequencies : sequence of float or np.ndarray
        Explicit observation frequencies in Hz.
    label : str, optional
        Boundary-specific label used in validation errors.

    Returns
    -------
    np.ndarray
        A caller-independent, strictly ascending float64 array in Hz.
    """
    resolved = np.array(frequencies, dtype=np.float64, copy=True)
    return validate_frequency_axis(
        resolved,
        label=label,
        ascending=True,
    )
