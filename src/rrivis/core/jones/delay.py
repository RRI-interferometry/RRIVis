"""Delay-based Jones terms (electronic delays, cable reflections, fringe-fitting).

The delay term (K_delay) represents instrumental delay offsets:

    K_delay = exp(-2πi·ν·τ) * I

Cable reflections model reflections in the RF path, while fringe-fitting
corrections adjust delays and rates for VLBI applications.

Stub implementation: returns identity matrix. TODO: implement properly.
"""

from typing import Any, Optional
import numpy as np

from rrivis.core.jones.base import JonesTerm


class DelayJones(JonesTerm):
    """Stub: Electronic delay Jones matrix. TODO: implement properly.

    Parameters
    ----------
    n_antennas : int, optional
        Number of antennas (ignored in stub)
    delays : np.ndarray, optional
        Per-antenna delays in seconds (ignored in stub)
    frequencies : np.ndarray, optional
        Observation frequencies in Hz (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(
        self,
        n_antennas: int = 1,
        delays: Optional[np.ndarray] = None,
        frequencies: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Initialize delay Jones term (stub)."""
        self.n_antennas = n_antennas
        self.delays = np.asarray(delays) if delays is not None else np.array([])
        self.frequencies = np.asarray(frequencies) if frequencies is not None else np.array([])

    @property
    def name(self) -> str:
        return "Kd"

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
        """Compute delay Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class CableReflectionJones(JonesTerm):
    """Stub: Cable reflection Jones matrix. TODO: implement properly.

    Parameters
    ----------
    n_antennas : int, optional
        Number of antennas (ignored in stub)
    reflection_coeff : float, optional
        Reflection coefficient (ignored in stub)
    cable_delay : float, optional
        Cable reflection delay in seconds (ignored in stub)
    frequencies : np.ndarray, optional
        Observation frequencies in Hz (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(
        self,
        n_antennas: int = 1,
        reflection_coeff: Optional[float] = None,
        cable_delay: Optional[float] = None,
        frequencies: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Initialize cable reflection Jones term (stub)."""
        self.n_antennas = n_antennas
        self.reflection_coeff = reflection_coeff
        self.cable_delay = cable_delay
        self.frequencies = np.asarray(frequencies) if frequencies is not None else np.array([])

    @property
    def name(self) -> str:
        return "Rc"

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
        """Compute cable reflection Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class FringeFitJones(JonesTerm):
    """Stub: VLBI fringe-fitting (delay/rate correction) Jones matrix. TODO: implement properly.

    Parameters
    ----------
    n_antennas : int, optional
        Number of antennas (ignored in stub)
    delays : np.ndarray, optional
        Per-antenna delays in seconds (ignored in stub)
    rates : np.ndarray, optional
        Per-antenna fringe rates (ignored in stub)
    phases : np.ndarray, optional
        Per-antenna phases (ignored in stub)
    frequencies : np.ndarray, optional
        Observation frequencies in Hz (ignored in stub)
    times : np.ndarray, optional
        Observation times (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(
        self,
        n_antennas: int = 1,
        delays: Optional[np.ndarray] = None,
        rates: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
        frequencies: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Initialize fringe-fitting Jones term (stub)."""
        self.n_antennas = n_antennas
        self.delays = np.asarray(delays) if delays is not None else np.array([])
        self.rates = np.asarray(rates) if rates is not None else np.array([])
        self.phases = np.asarray(phases) if phases is not None else np.array([])
        self.frequencies = np.asarray(frequencies) if frequencies is not None else np.array([])
        self.times = np.asarray(times) if times is not None else np.array([])

    @property
    def name(self) -> str:
        return "ff"

    @property
    def is_direction_dependent(self) -> bool:
        return False

    @property
    def is_frequency_dependent(self) -> bool:
        return True

    @property
    def is_time_dependent(self) -> bool:
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
        """Compute fringe-fitting Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)
