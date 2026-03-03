"""Jones chain manager for combining multiple Jones terms.

The JonesChain class manages the ordered chain of Jones matrices and provides
methods to compute the total Jones matrix and baseline visibilities.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from rrivis.core.jones.base import JonesTerm


class JonesChain:
    """Manages the ordered chain of per-antenna Jones matrices.

    The Jones chain represents the multiplicative sequence of instrumental
    and propagation effects applied to the signal, sky → correlator:

        J_total = B @ G @ D @ P @ E @ T @ Z @ K   (core 8 terms)

    Extended terms can be inserted at any position, e.g.:

        J_total = B @ G @ GAINCURVE @ X @ DF @ D @ P @ C @ E @ Ee @ a @ dE
                  @ F @ T @ Z @ W @ K @ Kd @ Rc

    Terms are stored left-to-right and iterated in reverse so that the
    rightmost (sky-side) term is applied first to the incoming signal.

    Notes:
        Only ``JonesTerm`` subclasses may be added here.  Baseline-dependent
        terms (``JonesBaselineTerm`` — currently M and Q) operate on
        visibilities via Hadamard multiplication and must be applied
        separately *after* ``compute_baseline_visibility``.

    Example:
        >>> from rrivis.backends import get_backend
        >>> backend = get_backend("numpy")
        >>> chain = JonesChain(backend)
        >>>
        >>> # Add Jones terms (sky → correlator order, rightmost first)
        >>> chain.add_term(GeometricPhaseJones(...))   # K
        >>> chain.add_term(BeamJones(...))             # E
        >>> chain.add_term(GainJones(...))             # G
        >>>
        >>> # Compute total Jones matrix for antenna 0, source 5
        >>> J = chain.compute_antenna_jones(
        ...     antenna_idx=0, source_idx=5,
        ...     freq_idx=10, time_idx=0
        ... )
    """

    def __init__(self, backend: Any):
        """Initialize Jones chain.

        Args:
            backend: ArrayBackend instance for device placement
        """
        self.backend = backend
        self.terms: List[JonesTerm] = []

    def add_term(
        self,
        term: JonesTerm,
        position: str = "append"
    ) -> None:
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

        Example:
            >>> chain.add_term(gain_jones)                    # append (correlator side)
            >>> chain.add_term(geometric_jones, position="prepend")  # sky side
            >>> chain.add_term(faraday_jones, position=2)     # at index 2
        """
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

    def get_term(self, name: str) -> Optional[JonesTerm]:
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

    def compute_antenna_jones(
        self,
        antenna_idx: int,
        source_idx: Optional[int],
        freq_idx: int,
        time_idx: int,
        **kwargs
    ) -> Any:
        """Compute total Jones matrix for one antenna-source pair.

        This multiplies all Jones terms in order:
            J_total = J_n @ J_{n-1} @ ... @ J_1

        Args:
            antenna_idx: Antenna index
            source_idx: Source index (None for DI-only chains)
            freq_idx: Frequency channel index
            time_idx: Time sample index
            **kwargs: Additional parameters passed to each term

        Returns:
            Complex 2x2 Jones matrix on backend device
        """
        xp = self.backend.xp

        # Start with identity matrix
        J_total = xp.eye(2, dtype=np.complex128)

        if not self.terms:
            return J_total

        # Apply terms in reverse order (rightmost applied first)
        # This matches the physical order: sky -> correlator
        for term in reversed(self.terms):
            # Compute Jones matrix for this term
            if term.is_direction_dependent:
                if source_idx is None:
                    raise ValueError(
                        f"Term '{term.name}' is direction-dependent "
                        f"but source_idx is None"
                    )
                J_term = term.compute_jones(
                    antenna_idx, source_idx, freq_idx, time_idx,
                    self.backend, **kwargs
                )
            else:
                # Direction-independent: source index not needed
                J_term = term.compute_jones(
                    antenna_idx, None, freq_idx, time_idx,
                    self.backend, **kwargs
                )

            # Multiply: J_total = J_term @ J_total
            J_total = self.backend.matmul(J_term, J_total)

        return J_total

    def compute_antenna_jones_all_sources(
        self,
        antenna_idx: int,
        n_sources: int,
        freq_idx: int,
        time_idx: int,
        **kwargs,
    ) -> Any:
        """Compute total Jones matrix for one antenna, all sources.

        Multiplies all terms in the chain for all sources at once:
            J_total[s] = J_n[s] @ J_{n-1}[s] @ ... @ J_1[s]

        For direction-independent terms, the single (2, 2) matrix is
        broadcast across all sources.

        Args:
            antenna_idx: Antenna index
            n_sources: Number of sources
            freq_idx: Frequency channel index
            time_idx: Time sample index
            **kwargs: Additional parameters passed to each term

        Returns:
            Complex array of shape (n_sources, 2, 2)
        """
        xp = self.backend.xp

        # Start with batch identity: (n_sources, 2, 2)
        J_total = xp.zeros((n_sources, 2, 2), dtype=np.complex128)
        J_total[:, 0, 0] = 1.0
        J_total[:, 1, 1] = 1.0

        if not self.terms:
            return J_total

        # Apply terms in reverse order (rightmost applied first)
        for term in reversed(self.terms):
            if term.is_direction_dependent:
                J_term = term.compute_jones_all_sources(
                    antenna_idx, n_sources, freq_idx, time_idx,
                    self.backend, **kwargs,
                )
            else:
                # Direction-independent: compute once and broadcast
                J_single = term.compute_jones(
                    antenna_idx, None, freq_idx, time_idx,
                    self.backend, **kwargs,
                )
                J_term = xp.broadcast_to(
                    J_single[np.newaxis], (n_sources, 2, 2)
                ).copy()

            # Batched matmul: J_total = J_term @ J_total
            J_total = J_term @ J_total

        return J_total

    def compute_baseline_visibility(
        self,
        antenna_p: int,
        antenna_q: int,
        source_idx: int,
        freq_idx: int,
        time_idx: int,
        coherency_matrix: Any,
        **kwargs
    ) -> Any:
        """Compute visibility contribution from one source for one baseline.

        Implements the per-antenna RIME:

            V_pq = J_p @ C @ J_q^H

        Note:
            This method only applies ``JonesTerm`` (per-antenna) effects.
            Baseline-dependent corrections (M, Q from ``JonesBaselineTerm``)
            must be applied *after* this call via Hadamard multiplication:

                V_pq = M_pq ⊙ Q_spq ⊙ (J_p @ C @ J_q^H)

        Args:
            antenna_p: First antenna index
            antenna_q: Second antenna index
            source_idx: Source index
            freq_idx: Frequency index
            time_idx: Time index
            coherency_matrix: Source coherency matrix (2x2 complex)
            **kwargs: Additional parameters forwarded to each term's
                      ``compute_jones`` (e.g., ``baseline_uvw`` for the K term)

        Returns:
            Complex 2x2 visibility matrix
        """
        # Compute Jones matrices for both antennas
        J_p = self.compute_antenna_jones(
            antenna_p, source_idx, freq_idx, time_idx, **kwargs
        )

        J_q = self.compute_antenna_jones(
            antenna_q, source_idx, freq_idx, time_idx, **kwargs
        )

        # RIME: V = J_p @ C @ J_q^H
        temp = self.backend.matmul(J_p, coherency_matrix)
        V = self.backend.matmul(temp, self.backend.conjugate_transpose(J_q))

        return V

    def get_enabled_effects(self) -> Dict[str, Dict[str, Any]]:
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

    def get_config(self) -> Dict[str, Any]:
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
