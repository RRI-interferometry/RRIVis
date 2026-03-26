"""Element beam and array factor Jones terms.

The element beam (E_element) represents the radiation pattern of a single
antenna element. The array factor (a) describes the combined effect of
multiple elements. The differential beam (ΔE) captures residual beam
errors between antennas.

Stub implementation: returns identity matrix. TODO: implement properly.
"""

from typing import Any

import numpy as np

from rrivis.core.jones.base import JonesTerm


class ElementBeamJones(JonesTerm):
    """Stub: Element beam pattern Jones matrix. TODO: implement properly.

    Parameters
    ----------
    n_antennas : int, optional
        Number of antennas (ignored in stub)
    source_positions : np.ndarray, optional
        Source positions (ignored in stub)
    frequencies : np.ndarray, optional
        Observation frequencies in Hz (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(
        self,
        n_antennas: int = 1,
        source_positions: np.ndarray | None = None,
        frequencies: np.ndarray | None = None,
        **kwargs,
    ):
        """Initialize element beam Jones term (stub)."""
        self.n_antennas = n_antennas
        self.source_positions = (
            np.asarray(source_positions)
            if source_positions is not None
            else np.array([])
        )
        self.frequencies = (
            np.asarray(frequencies) if frequencies is not None else np.array([])
        )

    @property
    def name(self) -> str:
        return "Ee"

    @property
    def is_direction_dependent(self) -> bool:
        return True

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int | None,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs,
    ) -> Any:
        """Compute element beam Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class ArrayFactorJones(JonesTerm):
    """Stub: Array factor (mutual coupling) Jones matrix. TODO: implement properly.

    Parameters
    ----------
    n_antennas : int, optional
        Number of antennas (ignored in stub)
    n_elements : int, optional
        Number of elements per antenna (ignored in stub)
    frequencies : np.ndarray, optional
        Observation frequencies in Hz (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(
        self,
        n_antennas: int = 1,
        n_elements: int = 1,
        frequencies: np.ndarray | None = None,
        **kwargs,
    ):
        """Initialize array factor Jones term (stub)."""
        self.n_antennas = n_antennas
        self.n_elements = n_elements
        self.frequencies = (
            np.asarray(frequencies) if frequencies is not None else np.array([])
        )

    @property
    def name(self) -> str:
        return "a"

    @property
    def is_direction_dependent(self) -> bool:
        return True

    def is_scalar(self) -> bool:
        return True  # Array factor is typically scalar

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int | None,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs,
    ) -> Any:
        """Compute array factor Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class DifferentialBeamJones(JonesTerm):
    """Stub: Differential beam residuals between antennas. TODO: implement properly.

    Parameters
    ----------
    n_antennas : int, optional
        Number of antennas (ignored in stub)
    n_sources : int, optional
        Number of sources (ignored in stub)
    frequencies : np.ndarray, optional
        Observation frequencies in Hz (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(
        self,
        n_antennas: int = 1,
        n_sources: int = 1,
        frequencies: np.ndarray | None = None,
        **kwargs,
    ):
        """Initialize differential beam Jones term (stub)."""
        self.n_antennas = n_antennas
        self.n_sources = n_sources
        self.frequencies = (
            np.asarray(frequencies) if frequencies is not None else np.array([])
        )

    @property
    def name(self) -> str:
        return "dE"

    @property
    def is_direction_dependent(self) -> bool:
        return True

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int | None,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs,
    ) -> Any:
        """Compute differential beam Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)
