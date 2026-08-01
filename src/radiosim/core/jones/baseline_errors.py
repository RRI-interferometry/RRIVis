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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from radiosim.core.jones.directions import DirectionBatch


class JonesBaselineTerm(ABC):
    """Abstract base for per-BASELINE (not per-antenna) RIME terms.

    These terms cannot be added to JonesChain (which expects per-antenna terms,
    and rejects these by type). They apply to visibilities directly via
    element-wise multiplication:

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

    def compute_baseline_factor(
        self,
        *,
        baseline_idx: int,
        antenna_p: int,
        antenna_q: int,
        directions: "DirectionBatch",
        frequency_hz: float,
        freq_idx: int,
        time_mjd: float,
        time_idx: int,
        backend: Any,
        dtype: Any,
    ) -> Any:
        """Return this term's Hadamard factor for one baseline.

        The batched counterpart of :class:`~radiosim.core.jones.base.JonesTerm`'s
        ``compute_jones_batch`` (``Tier7JonesSciencePlan.md`` Section 13.2), for
        the same reason: ``Q``'s smearing factor is direction-dependent, and one
        Python call per direction cannot carry a HEALPix pixel batch.

        Parameters
        ----------
        baseline_idx : int
            Row of this baseline in the resolved baseline selection.
        antenna_p, antenna_q : int
            Antenna rows of the pair, in the solver instrument view.
        directions : DirectionBatch
            The directions for this ``(time, frequency)`` step.
        frequency_hz, time_mjd : float
            Physical frequency and time.
        freq_idx, time_idx : int
            The corresponding grid indices.
        backend : ArrayBackend
            The backend to compute through.
        dtype : dtype
            The resolved complex dtype, passed in and never chosen by the term.

        Returns
        -------
        array
            ``(n_dir, 2, 2)`` for a direction-dependent factor (``Q``) or
            ``(1, 2, 2)`` for a direction-independent one (``M``), to be applied
            elementwise -- never composed into the matrix chain.

        Notes
        -----
        Concrete-and-raising rather than ``@abstractmethod`` for the same bounded
        reason as ``JonesTerm.compute_jones_batch``: the ``M`` and ``Q`` identity
        stubs below are Tier 7H's to replace, and an abstract declaration here
        would make them impossible to instantiate and silently break the 7A
        characterization pins that 7H owns.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement compute_baseline_factor; "
            "every baseline-dependent term must implement the direction-batched "
            "contract (Tier7JonesSciencePlan.md Section 13.2)."
        )


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
