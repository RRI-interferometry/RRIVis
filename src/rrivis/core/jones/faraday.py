"""Faraday rotation Jones term (F matrix).

Faraday rotation from magnetized plasma along the line of sight.
The rotation angle is proportional to RM·λ²:

    φ_F = RM · λ²

Stub implementation: returns identity matrix. TODO: implement properly.
"""

from typing import Any, Optional
import numpy as np

from rrivis.core.jones.base import JonesTerm


class FaradayRotationJones(JonesTerm):
    """Stub: Faraday rotation Jones matrix. TODO: implement properly.

    Parameters
    ----------
    rotation_measure : float, optional
        Rotation measure in rad/m² (ignored in stub)
    frequencies : np.ndarray, optional
        Observation frequencies in Hz (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(
        self,
        rotation_measure: Optional[float] = None,
        frequencies: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Initialize Faraday rotation Jones term (stub)."""
        self.rotation_measure = rotation_measure
        self.frequencies = np.asarray(frequencies) if frequencies is not None else np.array([])

    @property
    def name(self) -> str:
        return "F"

    @property
    def is_direction_dependent(self) -> bool:
        return True

    def is_unitary(self) -> bool:
        return True  # Rotation is unitary

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: Optional[int],
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs
    ) -> Any:
        """Compute Faraday rotation Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class DifferentialFaradayJones(FaradayRotationJones):
    """Stub: Differential Faraday rotation between antennas. TODO: implement properly."""

    def __init__(
        self,
        n_antennas: int,
        n_sources: int,
        frequencies: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Initialize differential Faraday rotation Jones term (stub)."""
        super().__init__(frequencies=frequencies)
        self.n_antennas = n_antennas
        self.n_sources = n_sources
