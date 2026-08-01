"""Jones chain manager for combining multiple Jones terms.

The JonesChain class manages the ordered chain of per-antenna Jones matrices and
composes them over one direction batch at a time.  It does not contract
baselines: that is the compiled kernel's job (``core/contraction.py``), and the
duplicate per-source contraction this module used to carry had no production
caller (``Tier7JonesSciencePlan.md`` Section 13.3).
"""

from typing import TYPE_CHECKING, Any

from radiosim.core.jones.base import JonesTerm
from radiosim.core.jones.baseline_errors import JonesBaselineTerm

if TYPE_CHECKING:
    from radiosim.core.jones.directions import DirectionBatch


class JonesChain:
    """Manages the ordered chain of per-antenna Jones matrices.

    The chain composes ``terms[0] @ terms[1] @ ... @ terms[-1]``: terms are
    stored left-to-right and iterated in reverse, so the **last** term added is
    the sky-side factor applied first to the incoming signal and the **first**
    term added is the correlator-side factor applied last.

    The canonical factorization, leftmost nearest the correlator
    (``Tier5ReceptorFeedPlan.md`` Section 19.1):

        J_total = H @ G @ B @ D @ P @ C @ E @ T @ Z   (K applied separately)

    so the canonical add order is H, G, B, D, P, C, E, T, Z.  ``H`` is leftmost
    because the reporting-basis change happens at the correlator; ``C`` sits
    between the sky-side direction-dependent terms (``E``, ``T``, ``Z``) and the
    electronics-side direction-independent ones (``D``, ``G``, ``B``), because
    leakage and gains are defined in the receptor's own basis.  ``K`` is applied
    separately as a scalar phase because it needs baseline coordinates.

    The full designed chain, once every term Tier 7 implements is present
    (``Tier7JonesSciencePlan.md`` Section 20.12), is the same factorization with
    the three additional diagonal calibration terms at their designed positions:

        J_total = H @ G @ B @ Rc @ Kd @ X @ D @ P @ C @ E @ T @ Z

    ``Rc``, ``Kd`` and ``X`` sit among ``G`` and ``B`` because all five are
    diagonal in the same basis and therefore commute: their *mutual* order is
    fixed by the plan so that the chain has one shape, one provenance string and
    one test, and is explicitly a convention rather than a physical claim.  The
    placements that *are* physical -- ``D`` correlator-side of ``C``, ``E``
    between ``C`` and the atmosphere, ``T`` and ``Z`` sky-side of everything --
    are unchanged.  ``K`` is applied separately by the solver because it is
    per-baseline; ``M`` and ``Q`` are not chain terms at all (they multiply
    finished visibilities elementwise).

    Order matters: ``C`` and ``H`` are the first factors RadioSim composes that
    do not commute with their neighbours.

    Notes:
        Only ``JonesTerm`` subclasses may be added here, and ``add_term``
        enforces it.  Baseline-dependent terms (``JonesBaselineTerm`` --
        currently M and Q) operate on finished visibilities via Hadamard
        multiplication and are applied by the solver, not by this chain.

    Example:
        >>> from radiosim.backends import get_backend
        >>> backend = get_backend("numpy")
        >>> chain = JonesChain(backend)
        >>>
        >>> # Add terms correlator-side first; the last added is applied first
        >>> chain.add_term(basis_transform_term)  # H
        >>> chain.add_term(receptor_config_term)  # C
        >>> chain.add_term(primary_beam_term)  # E, supplied by the solver
        >>>
        >>> # Compute this antenna's Jones over one whole direction batch
        >>> J = chain.compute_antenna_jones_batch(
        ...     antenna_idx=0,
        ...     directions=direction_batch,
        ...     frequency_hz=1.5e8,
        ...     freq_idx=0,
        ...     time_mjd=60000.0,
        ...     time_idx=0,
        ...     dtype=complex_dtype,
        ... )
    """

    def __init__(self, backend: Any):
        """Initialize Jones chain.

        Args:
            backend: ArrayBackend instance for device placement
        """
        self.backend = backend
        self.terms: list[JonesTerm] = []

    def add_term(self, term: JonesTerm, position: str = "append") -> None:
        """Add a per-antenna Jones term to the chain.

        Args:
            term: JonesTerm instance to add.  Must be a subclass of
                ``JonesTerm``.  ``JonesBaselineTerm`` instances (M, Q)
                cannot be added here — they must be applied separately
                via Hadamard multiplication on the finished visibility.
            position: Where to insert the term:
                - "append"  : Add to end / correlator side (default)
                - "prepend" : Add to beginning / sky side
                - int       : Insert at specific index

        Raises:
            TypeError: ``term`` is a ``JonesBaselineTerm`` (M, Q), or is not a
                ``JonesTerm`` at all.  Before Tier 7B this method performed no
                check at all, contradicting its own class docstring: a baseline
                term was accepted and then failed with an ``AttributeError``
                deep inside evaluation rather than with a typed rejection at the
                point of the mistake (defect D7).

        Example:
            >>> chain.add_term(gain_jones)  # append (correlator side)
            >>> chain.add_term(troposphere_jones, position="prepend")  # sky side
            >>> chain.add_term(leakage_jones, position=2)  # at index 2
        """
        if isinstance(term, JonesBaselineTerm):
            raise TypeError(
                f"{type(term).__name__} is a JonesBaselineTerm and cannot be added "
                "to a JonesChain: baseline-dependent terms multiply finished "
                "visibilities elementwise and are applied by the solver, not "
                "composed into the per-antenna matrix chain."
            )
        if not isinstance(term, JonesTerm):
            raise TypeError(
                f"{type(term).__name__} is not a JonesTerm; only JonesTerm "
                "subclasses may be added to a JonesChain."
            )
        if position == "append":
            self.terms.append(term)
        elif position == "prepend":
            self.terms.insert(0, term)
        elif isinstance(position, int):
            self.terms.insert(position, term)
        else:
            raise ValueError(
                f"Invalid position '{position}'. "
                f"Use 'append', 'prepend', or integer index."
            )

    def remove_term(self, name: str) -> bool:
        """Remove Jones term by name.

        Args:
            name: Short name of term to remove (e.g., 'K', 'E', 'G', 'F',
                  'Kd', 'ff', 'X', 'DF', 'GAINCURVE', etc.)

        Returns:
            True if term was found and removed, False otherwise
        """
        for i, term in enumerate(self.terms):
            if term.name == name:
                del self.terms[i]
                return True
        return False

    def get_term(self, name: str) -> JonesTerm | None:
        """Get Jones term by name.

        Args:
            name: Short name of term (e.g., 'K', 'E', 'G', 'F',
                  'Kd', 'ff', 'X', 'DF', 'GAINCURVE', etc.)

        Returns:
            JonesTerm instance if found, None otherwise
        """
        for term in self.terms:
            if term.name == name:
                return term
        return None

    def has_term(self, name: str) -> bool:
        """Check if chain contains a term with given name.

        Args:
            name: Name of term

        Returns:
            True if term exists in chain
        """
        return self.get_term(name) is not None

    def compute_antenna_jones_batch(
        self,
        *,
        antenna_idx: int,
        directions: "DirectionBatch",
        frequency_hz: float,
        freq_idx: int,
        time_mjd: float,
        time_idx: int,
        dtype: Any,
    ) -> Any:
        """Compose every term for one antenna over one direction batch.

        Applies the terms in reverse storage order, so the **last** term added
        is the sky-side factor applied first and the **first** term added is the
        correlator-side factor applied last::

            J_total = terms[0] @ terms[1] @ ... @ terms[-1]

        Parameters
        ----------
        antenna_idx : int
            Antenna row in the solver instrument view.
        directions : DirectionBatch
            The directions for this ``(time, frequency)`` step.
        frequency_hz, time_mjd : float
            Physical frequency and time.
        freq_idx, time_idx : int
            The corresponding grid indices.
        dtype : dtype
            The resolved complex dtype.  It seeds the identity and is handed to
            every term, so the composed product honours ``PrecisionConfig``
            instead of the two literal ``np.complex128`` seeds this method used
            to carry (defect D8).

        Returns
        -------
        array
            ``(n_dir, 2, 2)`` when any term is direction-dependent, otherwise
            ``(1, 2, 2)``.  The seed is ``(1, 2, 2)``, so a chain of purely
            direction-independent terms stays ``(1, 2, 2)`` all the way through
            and broadcasts once, at the end, against the direction-dependent
            factors -- never ``n_dir`` copies of one constant matrix.
        """
        J_total = self.backend.batch_eye((1,), 2, dtype=dtype)

        if not self.terms:
            return J_total

        for term in reversed(self.terms):
            J_term = term.compute_jones_batch(
                antenna_idx=antenna_idx,
                directions=directions,
                frequency_hz=frequency_hz,
                freq_idx=freq_idx,
                time_mjd=time_mjd,
                time_idx=time_idx,
                backend=self.backend,
                dtype=dtype,
            )
            J_total = self.backend.matmul(J_term, J_total)

        return J_total

    def get_enabled_effects(self) -> dict[str, dict[str, Any]]:
        """Get list of enabled Jones effects with metadata.

        Returns:
            Dictionary mapping effect name to properties
        """
        return {
            term.name: {
                "direction_dependent": term.is_direction_dependent,
                "time_dependent": term.is_time_dependent,
                "frequency_dependent": term.is_frequency_dependent,
                "diagonal": term.is_diagonal(),
                "scalar": term.is_scalar(),
            }
            for term in self.terms
        }

    def get_config(self) -> dict[str, Any]:
        """Get full chain configuration.

        Returns:
            Dictionary with chain configuration
        """
        return {
            "num_terms": len(self.terms),
            "term_order": [term.name for term in self.terms],
            "terms": {term.name: term.get_config() for term in self.terms},
        }

    def clear(self) -> None:
        """Remove all terms from chain."""
        self.terms.clear()

    def __len__(self) -> int:
        """Number of Jones terms in chain."""
        return len(self.terms)

    def __repr__(self) -> str:
        """String representation of chain."""
        if not self.terms:
            return "JonesChain(empty)"
        term_names = [term.name for term in self.terms]
        return f"JonesChain({' @ '.join(term_names)})"

    def __iter__(self):
        """Iterate over terms."""
        return iter(self.terms)
