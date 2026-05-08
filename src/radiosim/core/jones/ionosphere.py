"""
Ionosphere Jones term (Z) for ionospheric propagation effects.

Stub implementation: returns identity matrix. TODO: implement properly.
"""

from typing import Any

import numpy as np

from .base import JonesTerm


class IonosphereJones(JonesTerm):
    """Stub: Ionospheric propagation effects Jones term. TODO: implement properly.

    Parameters
    ----------
    tec : np.ndarray, optional
        Total Electron Content array (not used in stub).
    frequencies : np.ndarray, optional
        Observation frequencies in Hz.
    include_faraday : bool, optional
        Whether to include Faraday rotation (stub ignores this).
    include_delay : bool, optional
        Whether to include dispersive delay (stub ignores this).
    **kwargs : dict
        Additional parameters (ignored).
    """

    def __init__(
        self,
        tec: np.ndarray | None = None,
        frequencies: np.ndarray | None = None,
        include_faraday: bool = True,
        include_delay: bool = True,
        **kwargs,
    ):
        self.frequencies = (
            np.asarray(frequencies) if frequencies is not None else np.array([])
        )
        self.include_faraday = include_faraday
        self.include_delay = include_delay

    @property
    def name(self) -> str:
        return "Z"

    @property
    def is_direction_dependent(self) -> bool:
        return True

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs,
    ) -> Any:
        """Compute ionospheric Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class TurbulentIonosphereJones(IonosphereJones):
    """Stub: Turbulent ionosphere model. TODO: implement properly."""

    def __init__(
        self, n_antennas: int, n_sources: int, frequencies: np.ndarray, **kwargs
    ):
        super().__init__(frequencies=frequencies)
        self.n_antennas = n_antennas
        self.n_sources = n_sources


class GPSIonosphereJones(IonosphereJones):
    """Stub: GPS-based ionosphere model. TODO: implement properly."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
