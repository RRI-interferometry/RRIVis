"""Cross-hand and frequency-dependent leakage Jones terms.

Cross-hand effects arise from coupling between orthogonal polarization channels.
The cross-hand phase (X) represents static phase offset between X and Y,
while cross-hand delay (KCROSS) represents time-varying cross-hand effects.
Frequency-dependent leakage (DF) models leakage that varies with frequency.

Stub implementation: returns identity matrix. TODO: implement properly.
"""

from typing import Any, Optional
import numpy as np

from rrivis.core.jones.base import JonesTerm


class CrosshandPhaseJones(JonesTerm):
    """Stub: Cross-hand phase offset Jones matrix. TODO: implement properly.

    Parameters
    ----------
    phase_offset : float, optional
        Cross-hand phase offset in radians (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(
        self,
        phase_offset: float = 0.0,
        **kwargs
    ):
        """Initialize cross-hand phase Jones term (stub)."""
        self.phase_offset = phase_offset

    @property
    def name(self) -> str:
        return "X"

    @property
    def is_direction_dependent(self) -> bool:
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
        """Compute cross-hand phase Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class CrosshandDelayJones(JonesTerm):
    """Stub: Cross-hand delay Jones matrix. TODO: implement properly.

    Parameters
    ----------
    delay : float, optional
        Cross-hand delay in seconds (ignored in stub)
    frequencies : np.ndarray, optional
        Observation frequencies in Hz (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(
        self,
        delay: float = 0.0,
        frequencies: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Initialize cross-hand delay Jones term (stub)."""
        self.delay = delay
        self.frequencies = np.asarray(frequencies) if frequencies is not None else np.array([])

    @property
    def name(self) -> str:
        return "Kx"

    @property
    def is_direction_dependent(self) -> bool:
        return False

    @property
    def is_frequency_dependent(self) -> bool:
        return True

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
        """Compute cross-hand delay Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class FrequencyDependentLeakageJones(JonesTerm):
    """Stub: Frequency-dependent polarization leakage Jones matrix. TODO: implement properly.

    Parameters
    ----------
    n_antennas : int, optional
        Number of antennas (ignored in stub)
    frequencies : np.ndarray, optional
        Observation frequencies in Hz (ignored in stub)
    d_terms : np.ndarray, optional
        Leakage coefficients (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(
        self,
        n_antennas: int = 1,
        frequencies: Optional[np.ndarray] = None,
        d_terms: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Initialize frequency-dependent leakage Jones term (stub)."""
        self.n_antennas = n_antennas
        self.frequencies = np.asarray(frequencies) if frequencies is not None else np.array([])
        self.d_terms = np.asarray(d_terms) if d_terms is not None else np.array([])

    @property
    def name(self) -> str:
        return "DF"

    @property
    def is_direction_dependent(self) -> bool:
        return False

    @property
    def is_frequency_dependent(self) -> bool:
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
        """Compute frequency-dependent leakage Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)
