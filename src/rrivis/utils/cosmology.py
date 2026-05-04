"""21 cm cosmology helpers for EoR analysis.

Pure numpy + matplotlib; no rrivis imports. Used by EoR notebooks (cube
analysis, multipole evolution) to map between observation frequency and
hyperfine-line redshift, and to add a secondary redshift axis to frequency
plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

F_21CM_HZ: float = 1_420_405_751.768
"""Rest-frame frequency of the HI 21 cm hyperfine transition, in Hz."""


def frequency_to_redshift_21cm(
    freq_hz: np.ndarray | float,
) -> np.ndarray | float:
    """Map an observation frequency to the implied 21 cm redshift.

    ``z = (ν_rest / ν_obs) - 1``.  Frequencies must be positive; non-positive
    values raise ``ValueError`` because the redshift would be undefined.
    """
    arr = np.asarray(freq_hz, dtype=float)
    if np.any(arr <= 0):
        raise ValueError("frequency must be positive to compute a 21 cm redshift.")
    z = F_21CM_HZ / arr - 1.0
    if np.isscalar(freq_hz):
        return float(z)
    return z


def redshift_to_frequency_21cm(
    z: np.ndarray | float,
) -> np.ndarray | float:
    """Map a 21 cm redshift to its observation frequency in Hz.

    ``ν_obs = ν_rest / (1 + z)``.  Redshifts must satisfy ``z > -1``; values
    at or below ``-1`` raise ``ValueError``.
    """
    arr = np.asarray(z, dtype=float)
    if np.any(arr <= -1.0):
        raise ValueError("redshift must satisfy z > -1.")
    freq = F_21CM_HZ / (1.0 + arr)
    if np.isscalar(z):
        return float(freq)
    return freq


def add_redshift_secondary_axis(
    ax: Axes,
    *,
    axis: Literal["x", "y"] = "x",
    label: str = r"Redshift  $z$",
) -> Axes:
    """Attach a secondary 21 cm-redshift axis to a frequency plot.

    The host axis is assumed to carry frequency in Hz.  Returns the new
    twin axis so the caller can further style it (tick formatter, etc.).
    """
    if axis == "x":
        secondary = ax.secondary_xaxis(
            "top",
            functions=(frequency_to_redshift_21cm, redshift_to_frequency_21cm),
        )
        secondary.set_xlabel(label)
    elif axis == "y":
        secondary = ax.secondary_yaxis(
            "right",
            functions=(frequency_to_redshift_21cm, redshift_to_frequency_21cm),
        )
        secondary.set_ylabel(label)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}.")
    return secondary


__all__ = [
    "F_21CM_HZ",
    "frequency_to_redshift_21cm",
    "redshift_to_frequency_21cm",
    "add_redshift_secondary_axis",
]
