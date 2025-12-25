"""Abstract base class for Jones matrix terms.

Each Jones term represents one physical effect in the signal propagation chain.
Terms combine multiplicatively: J_total = J_n @ J_{n-1} @ ... @ J_1
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class JonesTerm(ABC):
    """Abstract base class for Jones matrix terms.

    Each term represents one physical effect in the signal propagation chain.
    Terms combine multiplicatively to form the total Jones matrix:

        J_total = J_n @ J_{n-1} @ ... @ J_1

    The order matters because matrix multiplication is non-commutative.

    Standard Jones terms (Smirnov 2011):
    - K: Geometric phase delay
    - Z: Ionosphere (Faraday rotation)
    - T: Troposphere (atmospheric delay)
    - E: Primary beam (antenna pattern)
    - P: Parallactic angle (feed rotation)
    - D: Polarization leakage
    - G: Complex gains
    - B: Bandpass

    Example:
        >>> class MyJonesTerm(JonesTerm):
        ...     @property
        ...     def name(self) -> str:
        ...         return "X"
        ...
        ...     @property
        ...     def is_direction_dependent(self) -> bool:
        ...         return False
        ...
        ...     def compute_jones(self, antenna_idx, source_idx, freq_idx,
        ...                       time_idx, backend, **kwargs):
        ...         return backend.eye(2, dtype=complex)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name identifier (e.g., 'K', 'E', 'G', 'B', 'Z', 'T', 'P', 'D').

        Standard names follow Smirnov (2011) convention.
        """
        pass

    @property
    @abstractmethod
    def is_direction_dependent(self) -> bool:
        """True if effect varies across the sky (DDE).

        Direction-dependent effects (DDE):
        - True: Beam (E), Ionosphere (Z), Troposphere (T), Phase (K)
        - False: Gains (G), Bandpass (B), Leakage (D), Parallactic (P)
        """
        pass

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

        Time-dependent effects:
        - True: Gains (G), Parallactic (P), Ionosphere (Z)
        - False: Bandpass (B), Leakage (D) [typically]

        Default: False (override if time-variable)
        """
        return False

    @property
    def is_frequency_dependent(self) -> bool:
        """True if effect varies with frequency.

        Frequency-dependent effects:
        - True: Bandpass (B), Ionosphere (Z), Phase (K), Beam (E)
        - False: Some constant gains (G)

        Default: True (most effects are chromatic)
        """
        return True

    @abstractmethod
    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: Optional[int],
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs
    ) -> Any:
        """Compute 2x2 Jones matrix for this effect.

        Args:
            antenna_idx: Antenna index (0 to N_ant-1)
            source_idx: Source index (0 to N_src-1), None for DI effects
            freq_idx: Frequency channel index
            time_idx: Time sample index
            backend: ArrayBackend instance (for device placement)
            **kwargs: Effect-specific parameters

        Returns:
            Complex 2x2 array on the backend device
            Shape: (2, 2) in linear polarization basis [X, Y]

        Notes:
            - For direction-independent effects, source_idx may be None
            - Result must be on the backend device (CPU/GPU)
            - Use backend.xp for array operations
        """
        pass

    def is_diagonal(self) -> bool:
        """True if Jones matrix is always diagonal (optimization hint).

        Diagonal matrices commute with each other and can be
        combined more efficiently.

        Diagonal Jones terms:
        - Gains (G), Bandpass (B), simple Troposphere (T)

        Non-diagonal:
        - Beam (E), Ionosphere (Z), Parallactic (P), Leakage (D)

        Default: False
        """
        return False

    def is_scalar(self) -> bool:
        """True if Jones matrix is scalar (proportional to identity).

        Scalar matrices commute with everything and simplify
        the matrix chain significantly.

        Scalar Jones terms:
        - Geometric phase (K) for unpolarized sources

        Default: False
        """
        return False

    def is_unitary(self) -> bool:
        """True if Jones matrix is unitary (J @ J^H = I).

        Unitary matrices preserve power and are energy-conserving.

        Unitary Jones terms:
        - Parallactic (P), Ionosphere (Z), Geometric phase (K)

        Non-unitary:
        - Gains (G), Beam (E), Leakage (D)

        Default: False
        """
        return False

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for this Jones term.

        Used for serialization, logging, and reproducibility.

        Returns:
            Dictionary with term configuration
        """
        return {
            "name": self.name,
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
