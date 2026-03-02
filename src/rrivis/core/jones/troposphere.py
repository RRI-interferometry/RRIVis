"""
Troposphere Jones term (T) for atmospheric propagation effects.

Stub implementation: returns identity matrix. TODO: implement properly.
"""

from typing import Any, Optional
import numpy as np

from .base import JonesTerm


class TroposphereJones(JonesTerm):
    """Stub: Tropospheric propagation effects Jones term. TODO: implement properly.

    Parameters
    ----------
    n_antennas : int, optional
        Number of antennas.
    frequencies : np.ndarray, optional
        Observation frequencies in Hz.
    elevations : np.ndarray, optional
        Source elevations in radians (not used in stub).
    **kwargs : dict
        Additional parameters (ignored).
    """

    def __init__(
        self,
        n_antennas: int = 1,
        frequencies: Optional[np.ndarray] = None,
        elevations: Optional[np.ndarray] = None,
        **kwargs
    ):
        self.n_antennas = n_antennas
        self.frequencies = np.asarray(frequencies) if frequencies is not None else np.array([])
        self.elevations = elevations

    @property
    def name(self) -> str:
        return "T"

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
        **kwargs
    ) -> Any:
        """Compute troposphere Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class SaastamoinenTroposphereJones(TroposphereJones):
    """Stub: Saastamoinen troposphere model. TODO: implement properly."""

    def __init__(
        self,
        n_antennas: int,
        n_sources: int,
        frequencies: np.ndarray,
        antenna_heights: Optional[np.ndarray] = None,
        **kwargs
    ):
        super().__init__(n_antennas=n_antennas, frequencies=frequencies)
        self.n_sources = n_sources


class TurbulentTroposphereJones(TroposphereJones):
    """Stub: Turbulent troposphere model. TODO: implement properly."""

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)


class TroposphericOpacityJones(TroposphereJones):
    """Stub: Tropospheric opacity correction (TOPAC). TODO: implement properly.

    The opacity term corrects for atmospheric absorption:
        J_opacity = exp(-τ / sin(el)) * I

    where τ is the zenith opacity and el is the source elevation.
    """

    def __init__(
        self,
        n_antennas: int = 1,
        frequencies: Optional[np.ndarray] = None,
        zenith_opacity: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Initialize tropospheric opacity Jones term (stub)."""
        super().__init__(n_antennas=n_antennas, frequencies=frequencies)
        self.zenith_opacity = np.asarray(zenith_opacity) if zenith_opacity is not None else np.array([])
