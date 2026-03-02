"""Receptor configuration and basis transform Jones terms (C, H matrices).

The C term describes the feed configuration (linear vs circular polarization):

    C = [[1, 0], [0, 1]]  for linear
    C = [[1, i], [i, 1]] / √2  for circular

The H term is a generic basis transformation between polarization bases.

Stub implementation: returns identity matrix. TODO: implement properly.
"""

from typing import Any, Optional
import numpy as np

from rrivis.core.jones.base import JonesTerm


class ReceptorConfigJones(JonesTerm):
    """Stub: Receptor configuration (feed type) Jones matrix. TODO: implement properly.

    Parameters
    ----------
    feed_type : str, optional
        Feed type: "linear", "circular", or "custom" (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(
        self,
        feed_type: str = "linear",
        **kwargs
    ):
        """Initialize receptor configuration Jones term (stub)."""
        self.feed_type = feed_type

    @property
    def name(self) -> str:
        return "C"

    @property
    def is_direction_dependent(self) -> bool:
        return False

    def is_unitary(self) -> bool:
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
        """Compute receptor configuration Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class BasisTransformJones(JonesTerm):
    """Stub: Polarization basis transformation matrix. TODO: implement properly.

    Parameters
    ----------
    from_basis : str, optional
        Source basis: "linear", "circular" (ignored in stub)
    to_basis : str, optional
        Target basis: "linear", "circular" (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(
        self,
        from_basis: str = "linear",
        to_basis: str = "circular",
        **kwargs
    ):
        """Initialize basis transform Jones term (stub)."""
        self.from_basis = from_basis
        self.to_basis = to_basis

    @property
    def name(self) -> str:
        return "H"

    @property
    def is_direction_dependent(self) -> bool:
        return False

    def is_unitary(self) -> bool:
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
        """Compute basis transform Jones matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)
