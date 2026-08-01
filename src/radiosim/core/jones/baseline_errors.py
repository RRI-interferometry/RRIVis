"""Baseline-dependent Jones terms (closure errors, smearing).

Some RIME terms are baseline-dependent (not per-antenna) and cannot be
represented as per-antenna Jones matrices. These include:

- M_pq: Baseline multiplicative closure errors (error closure)
- Q_spq: Time/bandwidth smearing decorrelation factors

These terms apply to visibilities via Hadamard (element-wise) multiplication
rather than the standard matrix chain. They require a separate base class
JonesBaselineTerm that is NOT a subclass of JonesTerm.

Both are planned, not implemented: Tier 7H implements them on the Hadamard path,
``Q`` folded into the compiled kernel's existing ``envelope`` argument and ``M``
applied to the kernel's ``(B, 2, 2)`` output, with the kernel signature
unchanged (``Tier7JonesSciencePlan.md`` Section 15, invariant I16).

References
----------
Bridle & Schwab (1999), in *Synthesis Imaging in Radio Astronomy II*, ASP Conf.
Ser. 180, 371 -- the time and bandwidth smearing expressions ``Q`` implements.
Thompson, Moran & Swenson (2017), 3rd ed., Section 10.4 -- closure relations,
which ``M`` breaks and every per-antenna term preserves (invariant I11).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

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

    @property
    def term_status(self) -> str:
        """``"implemented"`` if this term's physics exists, else ``"planned"``.

        The baseline-path counterpart of
        :attr:`~radiosim.core.jones.base.JonesTerm.term_status`, with the same
        contract and the same honest default; see that docstring for why the
        default is ``"planned"``.  Both ``M`` and ``Q`` are ``"planned"`` until
        Tier 7H.
        """
        return "planned"

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
        reason as ``JonesTerm.compute_jones_batch``: ``M`` and ``Q`` below are
        both ``term_status == "planned"`` until Tier 7H, and an abstract
        declaration here would make them impossible to instantiate.  It becomes
        ``@abstractmethod`` in that slice.

        Raising rather than returning an identity is the point: a Hadamard
        factor that is ``1`` everywhere is indistinguishable from no term at all.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement compute_baseline_factor; "
            "every baseline-dependent term must implement the direction-batched "
            "contract (Tier7JonesSciencePlan.md Section 13.2)."
        )


class BaselineMultiplicativeJones(JonesBaselineTerm):
    """Per-baseline multiplicative closure error ``M_pq`` (planned; Tier 7H).

    The canonical baseline-Hadamard error: a per-baseline complex factor that
    cannot be written as a product of two per-antenna gains, and therefore is
    the one term in the package that genuinely breaks closure.  Invariant I11 is
    the proof of the distinction -- an enabled ``G`` with arbitrary per-antenna
    phases leaves the closure phase invariant, an enabled ``M`` changes it by
    the predicted amount.

    ``term_status`` is ``"planned"``: constructing it is allowed, evaluating it
    raises.  It takes no parameters, for the same reason
    :class:`~radiosim.core.jones.gain.GainJones` takes none.
    """

    @property
    def name(self) -> str:
        return "M"

    @property
    def is_direction_dependent(self) -> bool:
        return False


class SmearingFactorJones(JonesBaselineTerm):
    """Time and bandwidth smearing decorrelation ``Q_spq`` (planned; Tier 7H).

    Time smearing from source motion during an integration and bandwidth
    smearing from the frequency spread across a channel both reduce the
    visibility amplitude and neither changes its phase, so ``Q`` is bounded by
    ``0 < Q <= 1``, is exactly ``1`` at the phase centre, and is
    direction-dependent (invariant I12).

    ``term_status`` is ``"planned"``: constructing it is allowed, evaluating it
    raises.  It takes no parameters, for the same reason
    :class:`~radiosim.core.jones.gain.GainJones` takes none.
    """

    @property
    def name(self) -> str:
        return "Q"

    @property
    def is_direction_dependent(self) -> bool:
        return True
