"""Abstract base class for Jones matrix terms.

Each Jones term represents one physical effect in the signal propagation chain.
Terms combine multiplicatively: J_total = J_n @ J_{n-1} @ ... @ J_1
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class JonesTerm(ABC):
    """Abstract base class for Jones matrix terms.

    Each term represents one physical effect in the signal propagation chain.
    Terms combine multiplicatively to form the total Jones matrix:

        J_total = J_n @ J_{n-1} @ ... @ J_1

    The order matters because matrix multiplication is non-commutative.

    Core Jones terms (Smirnov 2011), sky → correlator:
    - K  (GeometricPhaseJones)      : Geometric phase delay (DDE, scalar, unitary)
    - Z  (IonosphereJones, ...)     : Ionospheric Faraday rotation + TEC phase (DDE)
    - T  (TroposphereJones, ...)    : Tropospheric delay / opacity (DDE)
    - E  (BeamJones, ...)           : Primary beam voltage pattern (DDE)
    - P  (ParallacticAngleJones, ...): Parallactic angle / feed rotation (DIE)
    - D  (PolarizationLeakageJones, ...): Polarization leakage D-terms (DIE)
    - G  (GainJones, ...)           : Complex electronic gains (DIE, diagonal)
    - B  (BandpassJones, ...)       : Frequency-dependent bandpass (DIE, diagonal)

    Extended terms beyond the core 8:
    - F  (FaradayRotationJones, DifferentialFaradayJones)
         : Faraday rotation from magnetised ISM; φ = RM·λ² (DDE, unitary)
    - W  (WPhaseJones, WProjectionJones)
         : Non-coplanar baseline w-phase correction (DDE, scalar, unitary)
    - Txy (WidefieldPolarimetricJones)
         : Wide-field polarimetric projection for non-coplanar arrays (DDE)
    - C  (ReceptorConfigJones)      : Feed receptor configuration (linear/circular) (DIE, unitary)
    - H  (BasisTransformJones)      : Polarization basis transformation (DIE, unitary)
    - Ee (ElementBeamJones)         : Single-element beam pattern (DDE)
    - a  (ArrayFactorJones)         : Phased-array factor / mutual coupling (DDE, scalar)
    - dE (DifferentialBeamJones)    : Per-antenna differential beam residuals (DDE)
    - Kd (DelayJones)               : Instrumental delay offset; exp(-2πi·ν·τ) (DIE, diagonal)
    - Rc (CableReflectionJones)     : RF cable reflection errors (DIE, diagonal)
    - ff (FringeFitJones)           : VLBI fringe-fitting delay/rate correction (DIE, diagonal)
    - X  (CrosshandPhaseJones)      : Static cross-hand phase offset (DIE, diagonal)
    - Kx (CrosshandDelayJones)      : Time-varying cross-hand delay (DIE, diagonal)
    - DF (FrequencyDependentLeakageJones): Frequency-dependent D-terms (DIE)
    - GAINCURVE (ElevationGainJones): Elevation-dependent gain curve polynomial (DIE, diagonal)

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
        ...     def compute_jones(
        ...         self, antenna_idx, source_idx, freq_idx, time_idx, backend, **kwargs
        ...     ):
        ...         return backend.xp.eye(2, dtype=complex)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name identifier for this term.

        Core names follow Smirnov (2011): 'K', 'Z', 'T', 'E', 'P', 'D', 'G', 'B'.
        Extended names: 'F', 'W', 'Txy', 'C', 'H', 'Ee', 'a', 'dE',
                        'Kd', 'Rc', 'ff', 'X', 'Kx', 'DF', 'GAINCURVE'.
        Baseline names (JonesBaselineTerm): 'M', 'Q'.
        """
        pass

    @property
    @abstractmethod
    def is_direction_dependent(self) -> bool:
        """True if effect varies across the sky (DDE), False for DIE.

        Direction-dependent (DDE, True):
            K, Z, T, E, F, W, Txy, Ee, a, dE, Q
        Direction-independent (DIE, False):
            G, B, D, P, C, H, Kd, Rc, ff, X, Kx, DF, GAINCURVE, M
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

        Time-dependent (True): G, P, Z, T, F, Kx, ff (fringe rate)
        Typically static (False): B, D, K, C, H, X, Kd, Rc, DF

        Default: False (override if time-variable)
        """
        return False

    @property
    def is_frequency_dependent(self) -> bool:
        """True if effect varies with frequency.

        Frequency-dependent (True):
            B, K, E, Z, T, F, W, Ee, a, dE, Kd, Rc, ff, Kx, DF
        Frequency-independent (False):
            G (constant gains), P, D, C, H, X, GAINCURVE

        Default: True (most effects are chromatic)
        """
        return True

    @abstractmethod
    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int | None,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs,
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

        Diagonal matrices can be combined more efficiently.

        Diagonal: G, B, T (simple delay), Kd, Rc, ff, X, Kx, GAINCURVE
        Non-diagonal: E, Z, P, D, F, W, Txy, C, H, Ee, a, dE, DF

        Default: False
        """
        return False

    def is_scalar(self) -> bool:
        """True if Jones matrix is scalar (proportional to identity).

        Scalar matrices commute with everything and simplify the chain.

        Scalar: K, W (w-phase), a (array factor)
        Non-scalar: all others

        Default: False
        """
        return False

    def is_unitary(self) -> bool:
        """True if Jones matrix is unitary (J @ J^H = I).

        Unitary matrices preserve power (pure rotation/phase).

        Unitary: K, W, F, P, Z (Faraday rotation), C, H
        Non-unitary: G (amplitude errors), E (beam attenuation), D, B, T

        Default: False
        """
        return False

    def compute_jones_all_sources(
        self,
        antenna_idx: int,
        n_sources: int,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs,
    ) -> Any:
        """Compute Jones matrices for all sources at once.

        Default implementation: loops over sources calling compute_jones().
        Subclasses should override for true vectorization.

        Args:
            antenna_idx: Antenna index (0 to N_ant-1)
            n_sources: Number of sources
            freq_idx: Frequency channel index
            time_idx: Time sample index
            backend: ArrayBackend instance
            **kwargs: Effect-specific parameters

        Returns:
            Complex array of shape (n_sources, 2, 2)
        """
        xp = backend.xp
        result = xp.zeros((n_sources, 2, 2), dtype=np.complex128)
        for s in range(n_sources):
            result[s] = self.compute_jones(
                antenna_idx, s, freq_idx, time_idx, backend, **kwargs
            )
        return result

    def get_config(self) -> dict[str, Any]:
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
