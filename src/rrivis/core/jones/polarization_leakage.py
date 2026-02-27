"""
Polarization Leakage Jones term (D) for instrumental polarization.

Stub implementation: returns identity matrix. TODO: implement properly.
"""

from typing import Any, Optional
import numpy as np

from .base import JonesTerm


class PolarizationLeakageJones(JonesTerm):
    """Stub: Polarization leakage (D-term) calibration. TODO: implement properly.

    Parameters
    ----------
    n_antennas : int
        Number of antennas in the array.
    d_terms : np.ndarray, optional
        Complex leakage terms (not used in stub).
    **kwargs : dict
        Additional parameters (ignored).
    """

    def __init__(
        self,
        n_antennas: int,
        d_terms: Optional[np.ndarray] = None,
        **kwargs
    ):
        self.n_antennas = n_antennas

    @property
    def name(self) -> str:
        return "D"

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
        """Compute polarization leakage Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class IXRLeakageJones(PolarizationLeakageJones):
    """Stub: Polarization leakage with IXR model. TODO: implement properly."""

    def __init__(
        self,
        n_antennas: int,
        target_ixr: float = 1000.0,
        **kwargs
    ):
        super().__init__(n_antennas)
        self.target_ixr = target_ixr

    def get_ixr(self, antenna_idx: Optional[int] = None) -> float:
        """Stub: return target IXR."""
        return self.target_ixr if antenna_idx is None else np.full(self.n_antennas, self.target_ixr)


class MuellerLeakageJones(PolarizationLeakageJones):
    """Stub: D-terms derived from Mueller matrix formalism. TODO: implement properly."""

    def __init__(
        self,
        n_antennas: int,
        **kwargs
    ):
        super().__init__(n_antennas)


class BeamSquintLeakageJones(PolarizationLeakageJones):
    """Stub: Direction-dependent D-terms from beam squint. TODO: implement properly."""

    def __init__(
        self,
        n_antennas: int,
        **kwargs
    ):
        super().__init__(n_antennas)
