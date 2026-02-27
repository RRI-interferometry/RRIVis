"""
Bandpass Jones term (B) for frequency-dependent instrumental response.

Stub implementation: returns identity matrix. TODO: implement properly.
"""

from typing import Any, Optional
import numpy as np

from .base import JonesTerm


class BandpassJones(JonesTerm):
    """Stub: Bandpass Jones term. TODO: implement properly.

    Parameters
    ----------
    n_antennas : int
        Number of antennas in the array.
    frequencies : np.ndarray
        Frequency channels in Hz, shape (n_freq,).
    bandpass_gains : np.ndarray, optional
        Pre-computed bandpass gains (not used in stub).
    """

    def __init__(
        self,
        n_antennas: int,
        frequencies: np.ndarray,
        bandpass_gains: Optional[np.ndarray] = None
    ):
        self.n_antennas = n_antennas
        self.frequencies = np.asarray(frequencies)

    @property
    def name(self) -> str:
        return "B"

    @property
    def is_direction_dependent(self) -> bool:
        return False

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs
    ) -> Any:
        """Compute bandpass Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class PolynomialBandpassJones(BandpassJones):
    """Stub: Bandpass model using polynomial representation. TODO: implement properly."""

    def __init__(
        self,
        n_antennas: int,
        frequencies: np.ndarray,
        **kwargs
    ):
        super().__init__(n_antennas, frequencies)


class SplineBandpassJones(BandpassJones):
    """Stub: Bandpass model using cubic spline interpolation. TODO: implement properly."""

    def __init__(
        self,
        n_antennas: int,
        frequencies: np.ndarray,
        **kwargs
    ):
        super().__init__(n_antennas, frequencies)


class RFIFlaggedBandpassJones(BandpassJones):
    """Stub: Bandpass with RFI flagging support. TODO: implement properly."""

    def __init__(
        self,
        n_antennas: int,
        frequencies: np.ndarray,
        **kwargs
    ):
        super().__init__(n_antennas, frequencies)
