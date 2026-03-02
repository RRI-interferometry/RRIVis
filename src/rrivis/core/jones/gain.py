"""Complex gain Jones term (G matrix).

Stub implementation: returns identity matrix. TODO: implement properly.
"""

from typing import Any, Dict, Optional
import numpy as np

from rrivis.core.jones.base import JonesTerm


class GainJones(JonesTerm):
    """Stub: Complex electronic gains Jones matrix. TODO: implement properly.

    Args:
        n_antennas: Number of antennas (used if gains is None)
        gain_sigma: Standard deviation for random gain perturbations
        gains: Pre-computed gain values (complex). If None, uses ideal gains.
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        n_antennas: int = 1,
        gain_sigma: float = 0.0,
        gains: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        """Initialize gain Jones term (stub)."""
        self.n_antennas = n_antennas
        self.gain_sigma = gain_sigma
        self._seed = seed

    @property
    def name(self) -> str:
        return "G"

    @property
    def is_direction_dependent(self) -> bool:
        return False

    @property
    def is_frequency_dependent(self) -> bool:
        return False

    def is_diagonal(self) -> bool:
        return True

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: Optional[int],
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs
    ) -> Any:
        """Compute gain Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class TimeVariableGainJones(GainJones):
    """Stub: Gains that vary smoothly with time. TODO: implement properly."""

    def __init__(
        self,
        n_antennas: int,
        n_times: int,
        **kwargs
    ):
        """Initialize time-variable gains (stub)."""
        super().__init__(n_antennas=n_antennas)


class ElevationGainJones(GainJones):
    """Stub: Elevation-dependent antenna gain (GAINCURVE). TODO: implement properly.

    Antenna gain is often a polynomial function of elevation angle:
        g(el) = c₀ + c₁·el + c₂·el² + ...

    This term applies antenna-specific gain curves.
    """

    def __init__(
        self,
        n_antennas: int = 1,
        gain_curve_coeffs: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Initialize elevation gain Jones term (stub)."""
        super().__init__(n_antennas=n_antennas)
        self.gain_curve_coeffs = np.asarray(gain_curve_coeffs) if gain_curve_coeffs is not None else np.array([])