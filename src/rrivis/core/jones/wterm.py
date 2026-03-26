"""Non-coplanar baseline (W-term) Jones terms.

The W-term corrects for the non-coplanar nature of baselines in wide-field
radio interferometry. For a baseline with w-coordinate and source position lmn:

    K_W = exp(-2πi·w·(n-1)) * I

The wide-field polarimetric projection term T^(xy) applies coordinate
transformations for polarimetric imaging.

Stub implementation: returns identity matrix. TODO: implement properly.
"""

from typing import Any

import numpy as np

from rrivis.core.jones.base import JonesTerm


class WPhaseJones(JonesTerm):
    """Stub: W-phase non-coplanar correction Jones matrix. TODO: implement properly.

    Parameters
    ----------
    source_lmn : np.ndarray, optional
        Source direction cosines (ignored in stub)
    wavelengths : np.ndarray, optional
        Wavelengths in meters (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(
        self,
        source_lmn: np.ndarray | None = None,
        wavelengths: np.ndarray | None = None,
        **kwargs,
    ):
        """Initialize W-phase Jones term (stub)."""
        self.source_lmn = (
            np.asarray(source_lmn) if source_lmn is not None else np.array([])
        )
        self.wavelengths = (
            np.asarray(wavelengths) if wavelengths is not None else np.array([])
        )

    @property
    def name(self) -> str:
        return "W"

    @property
    def is_direction_dependent(self) -> bool:
        return True

    def is_scalar(self) -> bool:
        return True  # Phase term is scalar

    def is_unitary(self) -> bool:
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
        """Compute W-phase Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class WProjectionJones(WPhaseJones):
    """Stub: W-projection imaging kernel for wide-field corrections. TODO: implement properly."""

    def __init__(
        self,
        n_antennas: int,
        source_lmn: np.ndarray | None = None,
        wavelengths: np.ndarray | None = None,
        **kwargs,
    ):
        """Initialize W-projection Jones term (stub)."""
        super().__init__(source_lmn=source_lmn, wavelengths=wavelengths)
        self.n_antennas = n_antennas


class WidefieldPolarimetricJones(JonesTerm):
    """Stub: Wide-field polarimetric projection T^(xy). TODO: implement properly.

    Parameters
    ----------
    source_lmn : np.ndarray, optional
        Source direction cosines (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(self, source_lmn: np.ndarray | None = None, **kwargs):
        """Initialize wide-field polarimetric Jones term (stub)."""
        self.source_lmn = (
            np.asarray(source_lmn) if source_lmn is not None else np.array([])
        )

    @property
    def name(self) -> str:
        return "Txy"

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
        """Compute wide-field polarimetric Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)
