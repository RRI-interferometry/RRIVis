"""Abstract base class for Jones matrix terms.

Each Jones term represents one physical effect in the signal propagation chain.
Terms combine multiplicatively: J_total = J_n @ J_{n-1} @ ... @ J_1
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radiosim.core.jones.directions import DirectionBatch


class JonesTerm(ABC):
    """Abstract base class for Jones matrix terms.

    Each term represents one physical effect in the signal propagation chain.
    Terms combine multiplicatively to form the total Jones matrix:

        J_total = J_n @ J_{n-1} @ ... @ J_1

    The order matters because matrix multiplication is non-commutative.

    Every ``JonesTerm`` subclass in this package is one of the terms below, and
    each carries a ``term_status`` saying whether its physics exists yet.  There
    are no others: ``Tier7JonesSciencePlan.md`` Section 9.1 deleted the
    twenty-six classes that were identity scaffolds for effects RadioSim has no
    plan to model, so a name that is here is a term that either works or refuses
    to run.

    The canonical chain, sky → correlator:

    - K  (``geometric_phase()``)    : Geometric phase delay (DDE, scalar, unitary).
         Not a ``JonesTerm``: the phase is per *baseline*, so the solvers apply
         it separately from the per-antenna chain
         (``Tier7JonesSciencePlan.md`` Section 13.3, defect D6).
    - Z  (IonosphereJones)          : Ionospheric TEC phase + Faraday rotation (DDE)
    - T  (TroposphereJones)         : Tropospheric delay and opacity (DDE)
    - P  (ParallacticAngleJones)    : Parallactic angle / field rotation (DDE).
         Sky-side of ``C`` and ``E`` since Tier 7F: a field rotation acts on the
         incoming field before the receptor sees it, so ``C P`` is the receptor
         at ``chi + psi`` (``Tier7JonesSciencePlan.md`` Section 12.1)
    - E  (canonical beam adapter)   : Primary beam voltage pattern (DDE) -- the
         private solver-owned adapter over ``BeamSystem``, not an exported class
    - C  (ReceptorConfigJones)      : Receptor basis and static feed rotation (DIE, unitary)
    - D  (PolarizationLeakageJones) : Polarization leakage D-terms (DIE)
    - B  (BandpassJones)            : Frequency-dependent bandpass (DIE, diagonal)
    - G  (GainJones)                : Complex electronic gains (DIE, diagonal)
    - H  (BasisTransformJones)      : Reporting-basis transform (DIE, unitary)

    Instrumental terms outside the canonical eight:

    - Kd (DelayJones)               : Instrumental delay offset; exp(-2πi·ν·τ) (DIE, diagonal)
    - Rc (CableReflectionJones)     : RF cable reflection ripple (DIE, diagonal)
    - X  (CrosshandJones)           : Cross-hand phase and delay (DIE, diagonal)

    Baseline-dependent terms (NOT subclasses of JonesTerm, use JonesBaselineTerm):

    - M  (BaselineMultiplicativeJones): Per-baseline closure errors (Hadamard product)
    - Q  (SmearingFactorJones)       : Time/bandwidth smearing decorrelation (Hadamard product, DDE)

    Example:
        >>> class MyJonesTerm(JonesTerm):
        ...     @property
        ...     def name(self) -> str:
        ...         return "My"
        ...
        ...     @property
        ...     def is_direction_dependent(self) -> bool:
        ...         return False
        ...
        ...     def compute_jones_batch(self, *, antenna_idx, dtype, backend, **_):
        ...         # Direction-independent: one (1, 2, 2) factor that broadcasts.
        ...         return backend.batch_eye((1,), 2, dtype=dtype)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name identifier for this term.

        Core names follow Smirnov (2011): 'K', 'Z', 'T', 'E', 'P', 'D', 'G', 'B'.
        Instrumental names beyond the core eight: 'C', 'H', 'Kd', 'Rc', 'X'.
        Baseline names (JonesBaselineTerm): 'M', 'Q'.
        """
        pass

    @property
    @abstractmethod
    def is_direction_dependent(self) -> bool:
        """True if effect varies across the sky (DDE), False for DIE.

        Direction-dependent (DDE, True):
            K, Z, T, E, P, Q
        Direction-independent (DIE, False):
            G, B, D, C, H, Kd, Rc, X, M
        """
        pass

    @property
    def term_status(self) -> str:
        """``"implemented"`` if this term's physics exists, else ``"planned"``.

        Exactly two values, and the default is the honest one.
        ``Tier7JonesSciencePlan.md`` Section 23 gives ``"implemented"`` as the
        value every exported term reaches by 7K, and Section 37 criterion 2 is
        the assertion that no ``"planned"`` survives the tier.  Defaulting to it
        *here* would be a lie on every term that does not yet run -- the same
        vacuous-``True`` failure mode invariant I2 exists to prevent, one level
        up.  So the base class declares ``"planned"``, each term overrides it in
        the slice that implements it (Section 31 step 5), and invariant I20
        checks the correspondence both ways: an ``"implemented"`` term must not
        be an identity for all inputs, and a ``"planned"`` term must not be
        evaluable at all.

        After Tier 7G every ``JonesTerm`` in this package is
        ``"implemented"``: ``Z`` and ``T`` were the last two, which is why
        ``compute_jones_batch`` below could become ``@abstractmethod`` in the
        same slice.  The two names still declaring ``"planned"`` are ``M`` and
        ``Q``, which are :class:`~radiosim.core.jones.baseline_errors.JonesBaselineTerm`
        and inherit this property's counterpart there; Tier 7H implements them.

        A ``"planned"`` term exists as a name, a chain position and a documented
        physical effect.  It is not a silent identity: it inherits the raising
        ``compute_jones_batch`` below, declares no capability flag it cannot
        support, and accepts no parameter it would discard.
        """
        return "planned"

    @property
    def is_baseline_dependent(self) -> bool:
        """True if effect depends on baseline (rare).

        Most Jones terms are per-antenna. Only exotic effects like
        baseline-dependent correlator errors would return True.

        Default: False
        """
        return False

    @property
    def is_time_dependent(self) -> bool:
        """True if effect varies with time.

        Time-dependent (True): G, P, Z, T
        Typically static (False): B, D, K, C, H, X, Kd, Rc

        Default: False (override if time-variable)
        """
        return False

    @property
    def is_frequency_dependent(self) -> bool:
        """True if effect varies with frequency.

        Frequency-dependent (True):
            B, K, E, Z, T, Kd, Rc, X (cross-hand delay)
        Frequency-independent (False):
            G (constant gains), P, D, C, H

        Default: True (most effects are chromatic)
        """
        return True

    @abstractmethod
    def compute_jones_batch(
        self,
        *,
        antenna_idx: int,
        directions: "DirectionBatch",
        frequency_hz: float,
        freq_idx: int,
        time_mjd: float,
        time_idx: int,
        backend: Any,
        dtype: Any,
    ) -> Any:
        """Return this term's Jones matrices for one antenna over one batch.

        This is *the* evaluation contract (``Tier7JonesSciencePlan.md``
        Section 13.2).  It replaced the scalar ``compute_jones(source_idx: int)``
        contract and its Python-loop ``compute_jones_all_sources`` default, which
        could not carry a HEALPix-scale direction batch: one Python call per
        pixel is why the diffuse solver bypassed the chain entirely (defects D4,
        D5).

        Parameters
        ----------
        antenna_idx : int
            Antenna row in the solver instrument view.
        directions : DirectionBatch
            Every sky direction for this ``(time, frequency)`` step, host-side
            and immutable.
        frequency_hz, time_mjd : float
            Physical frequency and time, not only indices, so a term needs no
            constructor-time copy of the observation grids to know where it is
            being evaluated.
        freq_idx, time_idx : int
            The corresponding grid indices.
        backend : ArrayBackend
            The backend to compute through.  A term must use backend primitives
            and ``backend.xp`` only: no ``float()`` on a traced array, no Python
            ``if`` on an array value, no ``.item()`` (Section 17.2).
        dtype : dtype
            The resolved complex dtype for this term.  It is passed in, never
            chosen by the term: that is what closes the ``np.complex128``
            hard-codes of defects D8 and D9.

        Returns
        -------
        array
            Complex, shape ``(n_dir, 2, 2)`` for a direction-dependent term or
            ``(1, 2, 2)`` for a direction-independent one, in the backend's own
            array domain and in ``dtype``.  A ``(1, 2, 2)`` return broadcasts
            against ``(n_dir, 2, 2)`` and is the **required** form for a DIE
            term: materialising ``n_dir`` identical copies would multiply the
            chain's memory by the direction count for no reason.  Invariant I3
            tests exactly this.

        Notes
        -----
        This method is ``@abstractmethod`` as of Tier 7G, the slice that
        implemented the last two planned ``JonesTerm`` subclasses (``Z`` and
        ``T``).  Until then it was concrete-and-raising for one bounded reason:
        an abstract declaration would have made every still-planned term
        impossible to instantiate, and a term that cannot be constructed cannot
        be named in a chain-order test either.  Every exported ``JonesTerm`` now
        implements this method, so the declaration costs nothing and the
        contract is enforced by the type system rather than by a runtime raise.

        The body is kept, and still raises.  ``@abstractmethod`` stops a
        *subclass* that forgets the contract from being instantiated; the body
        is what a caller gets if one reaches this method anyway -- through
        ``super()``, or through a subclass that declares the method and defers
        to it.  Raising rather than returning an identity is the whole point: a
        term that returns ``I2`` for every input is indistinguishable from no
        term at all, which is the ``SCI-001`` defect this contract closes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement compute_jones_batch; every "
            "Jones term must implement the direction-batched evaluation contract "
            "(Tier7JonesSciencePlan.md Section 13.2)."
        )

    def is_diagonal(self) -> bool:
        """True if Jones matrix is always diagonal (optimization hint).

        Diagonal matrices can be combined more efficiently.

        Declaring ``True`` here is a claim about numbers, not a hint that goes
        unchecked: ``tests/unit/test_jones/test_term_contract.py`` sweeps every
        term over its parameter space and asserts each declared property
        numerically, and requires a witness for each declared ``False``
        (invariant I2, the structural fix for defect D10 -- terms used to claim
        unitarity and scalarity about a matrix that was the 2x2 identity, which
        is trivially both).

        A ``"planned"`` term declares no flag at all: invariant I2's sweep
        cannot verify a claim about a matrix that cannot be computed, so each
        term slice adds its flags together with its physics.

        Diagonal: G, B, T (a scalar times ``I2``), Kd, Rc, X
        Non-diagonal: E, P, D, C, H, and Z whenever its Faraday rotation is
        non-zero -- a real rotation is diagonal only at angle zero

        Default: False
        """
        return False

    def is_scalar(self) -> bool:
        """True if Jones matrix is scalar (proportional to identity).

        Scalar matrices commute with everything and simplify the chain.

        Scalar: K (the geometric phase, which is why it is a function and not a
        term at all), T (delay and opacity are both scalars times ``I2``), Z
        without Faraday rotation, and C or H whenever their parameters make them
        exactly ``I2``.
        Non-scalar: all others.

        Default: False
        """
        return False

    def is_unitary(self) -> bool:
        """True if Jones matrix is unitary (J @ J^H = I).

        Unitary matrices preserve power (pure rotation/phase).

        Unitary: K, P, C, H, and Z -- whose dispersive phase and Faraday
        rotation are each unitary, so their product is too: the ionosphere
        delays and rotates the field without absorbing it.
        Non-unitary: G (amplitude errors), E (beam attenuation), D, B, and T
        whenever its opacity is enabled, because an absorbing atmosphere really
        does remove power.

        Default: False
        """
        return False

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary for this Jones term.

        Used for serialization, logging, and reproducibility.

        Returns:
            Dictionary with term configuration
        """
        return {
            "name": self.name,
            "term_status": self.term_status,
            "is_direction_dependent": self.is_direction_dependent,
            "is_time_dependent": self.is_time_dependent,
            "is_frequency_dependent": self.is_frequency_dependent,
            "is_diagonal": self.is_diagonal(),
            "is_scalar": self.is_scalar(),
            "is_unitary": self.is_unitary(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
