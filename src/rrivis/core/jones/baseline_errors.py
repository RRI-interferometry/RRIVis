"""Baseline-dependent Jones terms (closure errors, smearing).

Some RIME terms are baseline-dependent (not per-antenna) and cannot be
represented as per-antenna Jones matrices. These include:

- M_pq: Baseline multiplicative closure errors (error closure)
- Q_spq: Time/bandwidth smearing decorrelation factors

These terms apply to visibilities via Hadamard (element-wise) multiplication
rather than the standard matrix chain. They require a separate base class
JonesBaselineTerm that is NOT a subclass of JonesTerm.

Stub implementation: returns identity matrix. TODO: implement properly.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class JonesBaselineTerm(ABC):
    """Abstract base for per-BASELINE (not per-antenna) RIME terms.

    These terms cannot be added to JonesChain (which expects per-antenna terms).
    They apply to visibilities directly via element-wise multiplication:

        V_pq_corrected = M_pq ⊙ V_pq_original

    where ⊙ denotes Hadamard (element-wise) product.

    This is a separate abstraction from JonesTerm because baseline-dependent
    effects fundamentally differ from antenna effects.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name identifier (e.g., 'M', 'Q')."""
        pass

    @property
    @abstractmethod
    def is_direction_dependent(self) -> bool:
        """True if effect varies across the sky (DDE)."""
        pass

    @abstractmethod
    def compute_baseline_term(
        self,
        antenna_p: int,
        antenna_q: int,
        source_idx: int | None,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs,
    ) -> Any:
        """Compute 2x2 multiplicative correction for baseline V_pq.

        Args:
            antenna_p: Index of antenna p
            antenna_q: Index of antenna q
            source_idx: Source index (None for DI effects)
            freq_idx: Frequency channel index
            time_idx: Time sample index
            backend: ArrayBackend instance
            **kwargs: Effect-specific parameters

        Returns:
            Complex 2x2 array on the backend device
            Shape: (2, 2) in linear polarization basis [X, Y]
        """
        pass


class BaselineMultiplicativeJones(JonesBaselineTerm):
    """Stub: Per-baseline multiplicative closure error M_pq. TODO: implement properly.

    Closure errors from baseline-specific instrumental effects (e.g., correlator
    non-linearity, baseline-dependent gain variation).

    Parameters
    ----------
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(self, **kwargs):
        """Initialize baseline multiplicative error Jones term (stub)."""
        pass

    @property
    def name(self) -> str:
        return "M"

    @property
    def is_direction_dependent(self) -> bool:
        return False

    def compute_baseline_term(
        self,
        antenna_p: int,
        antenna_q: int,
        source_idx: int | None,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs,
    ) -> Any:
        """Compute baseline multiplicative error (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class SmearingFactorJones(JonesBaselineTerm):
    """Stub: Time/bandwidth smearing decorrelation Q_spq. TODO: implement properly.

    Time smearing from source motion during integration time, and bandwidth
    smearing from frequency spread across a channel. Both reduce visibility
    amplitude (decorrelation).

    Parameters
    ----------
    time_smearing : bool, optional
        Include time smearing correction (ignored in stub)
    bandwidth_smearing : bool, optional
        Include bandwidth smearing correction (ignored in stub)
    **kwargs : dict
        Additional parameters (ignored)
    """

    def __init__(
        self, time_smearing: bool = True, bandwidth_smearing: bool = True, **kwargs
    ):
        """Initialize smearing factor Jones term (stub)."""
        self.time_smearing = time_smearing
        self.bandwidth_smearing = bandwidth_smearing

    @property
    def name(self) -> str:
        return "Q"

    @property
    def is_direction_dependent(self) -> bool:
        return True

    def compute_baseline_term(
        self,
        antenna_p: int,
        antenna_q: int,
        source_idx: int | None,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs,
    ) -> Any:
        """Compute smearing decorrelation (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)
