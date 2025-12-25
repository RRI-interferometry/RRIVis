"""
Polarization Leakage Jones term (D) for instrumental polarization.

The D-Jones term models the leakage of signal between orthogonal
polarizations due to imperfect feed alignment, cross-coupling, and
other instrumental effects. This is a critical calibration term for
polarimetric observations.

The D-matrix has the form:
    D = [[1,    d_p],
         [d_q,  1  ]]

where d_p and d_q are the complex leakage terms (typically |d| < 0.1).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from abc import abstractmethod

from .base import JonesTerm


class PolarizationLeakageJones(JonesTerm):
    """
    Polarization leakage (D-term) calibration.

    Models the leakage between orthogonal polarizations due to
    non-ideal feed response.

    Parameters
    ----------
    n_antennas : int
        Number of antennas in the array.
    d_terms : np.ndarray, optional
        Complex leakage terms with shape (n_antennas, 2).
        d_terms[:, 0] is d_p (leakage from Y to X)
        d_terms[:, 1] is d_q (leakage from X to Y)
        If None, zero leakage is assumed.
    frequency_dependent : bool
        If True, d_terms has shape (n_antennas, n_freq, 2).
    frequencies : np.ndarray, optional
        Frequencies in Hz, required if frequency_dependent=True.
    """

    def __init__(
        self,
        n_antennas: int,
        d_terms: Optional[np.ndarray] = None,
        frequency_dependent: bool = False,
        frequencies: Optional[np.ndarray] = None
    ):
        self.n_antennas = n_antennas
        self.frequency_dependent = frequency_dependent

        if frequency_dependent:
            if frequencies is None:
                raise ValueError("frequencies required when frequency_dependent=True")
            self.frequencies = np.asarray(frequencies)
            self.n_freq = len(self.frequencies)

            if d_terms is None:
                self.d_terms = np.zeros((n_antennas, self.n_freq, 2), dtype=np.complex128)
            else:
                self.d_terms = np.asarray(d_terms, dtype=np.complex128)
                if self.d_terms.shape != (n_antennas, self.n_freq, 2):
                    raise ValueError(
                        f"d_terms shape {self.d_terms.shape} doesn't match "
                        f"expected ({n_antennas}, {self.n_freq}, 2)"
                    )
        else:
            self.frequencies = frequencies
            self.n_freq = len(frequencies) if frequencies is not None else None

            if d_terms is None:
                self.d_terms = np.zeros((n_antennas, 2), dtype=np.complex128)
            else:
                self.d_terms = np.asarray(d_terms, dtype=np.complex128)
                if self.d_terms.shape != (n_antennas, 2):
                    raise ValueError(
                        f"d_terms shape {self.d_terms.shape} doesn't match "
                        f"expected ({n_antennas}, 2)"
                    )

    @property
    def name(self) -> str:
        return "D"

    @property
    def is_direction_dependent(self) -> bool:
        # Primary D-terms are direction-independent
        # (though there can be direction-dependent leakage from beam squint)
        return False

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs
    ) -> Any:
        """
        Compute polarization leakage Jones matrix.

        The D-matrix is:
            D = [[1,    d_p],
                 [d_q,  1  ]]

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        source_idx : int
            Source index (not used, D is direction-independent).
        freq_idx : int
            Frequency channel index.
        time_idx : int
            Time index (not used for basic D-terms).
        backend : ArrayBackend
            Compute backend.

        Returns
        -------
        jones : array
            2x2 complex Jones matrix for polarization leakage.
        """
        xp = backend.xp

        if self.frequency_dependent:
            d_p = self.d_terms[antenna_idx, freq_idx, 0]
            d_q = self.d_terms[antenna_idx, freq_idx, 1]
        else:
            d_p = self.d_terms[antenna_idx, 0]
            d_q = self.d_terms[antenna_idx, 1]

        # Construct D-matrix
        d_matrix = np.array([
            [1.0 + 0j, d_p],
            [d_q, 1.0 + 0j]
        ], dtype=np.complex128)

        return backend.asarray(d_matrix)

    def set_d_terms(
        self,
        antenna_idx: int,
        d_p: complex,
        d_q: complex,
        freq_idx: Optional[int] = None
    ) -> None:
        """
        Set D-terms for an antenna.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        d_p : complex
            Leakage from Y to X polarization.
        d_q : complex
            Leakage from X to Y polarization.
        freq_idx : int, optional
            Frequency index (required if frequency_dependent=True).
        """
        if self.frequency_dependent:
            if freq_idx is None:
                raise ValueError("freq_idx required for frequency-dependent D-terms")
            self.d_terms[antenna_idx, freq_idx, 0] = d_p
            self.d_terms[antenna_idx, freq_idx, 1] = d_q
        else:
            self.d_terms[antenna_idx, 0] = d_p
            self.d_terms[antenna_idx, 1] = d_q

    def get_leakage_amplitude(self, antenna_idx: int) -> Tuple[float, float]:
        """
        Get leakage amplitudes for an antenna.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.

        Returns
        -------
        amp_p : float
            Amplitude of d_p (averaged over frequency if applicable).
        amp_q : float
            Amplitude of d_q (averaged over frequency if applicable).
        """
        if self.frequency_dependent:
            amp_p = np.mean(np.abs(self.d_terms[antenna_idx, :, 0]))
            amp_q = np.mean(np.abs(self.d_terms[antenna_idx, :, 1]))
        else:
            amp_p = np.abs(self.d_terms[antenna_idx, 0])
            amp_q = np.abs(self.d_terms[antenna_idx, 1])
        return amp_p, amp_q

    def get_average_leakage(self) -> float:
        """
        Get average leakage amplitude across all antennas.

        Returns
        -------
        avg_leakage : float
            Average |d| value.
        """
        return np.mean(np.abs(self.d_terms))


class IXRLeakageJones(PolarizationLeakageJones):
    """
    Polarization leakage with Intrinsic Cross-polarization Ratio (IXR) model.

    The IXR quantifies the quality of polarimetric calibration and is
    related to the D-terms by:
        IXR = (1 + |d|^2) / (2|d|)

    Higher IXR indicates better polarization purity.

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    target_ixr : float
        Target IXR value (typically > 100 for good polarimetry).
    d_phase_p : np.ndarray, optional
        Phases for d_p terms in radians, shape (n_antennas,).
    d_phase_q : np.ndarray, optional
        Phases for d_q terms in radians, shape (n_antennas,).
    """

    def __init__(
        self,
        n_antennas: int,
        target_ixr: float = 1000.0,
        d_phase_p: Optional[np.ndarray] = None,
        d_phase_q: Optional[np.ndarray] = None
    ):
        # Calculate |d| from IXR
        # IXR = (1 + |d|^2) / (2|d|)
        # 2*IXR*|d| = 1 + |d|^2
        # |d|^2 - 2*IXR*|d| + 1 = 0
        # |d| = IXR - sqrt(IXR^2 - 1)  (taking the smaller root)
        d_amplitude = target_ixr - np.sqrt(target_ixr**2 - 1)

        self.target_ixr = target_ixr

        # Random phases if not specified
        if d_phase_p is None:
            d_phase_p = np.random.uniform(0, 2 * np.pi, n_antennas)
        if d_phase_q is None:
            d_phase_q = np.random.uniform(0, 2 * np.pi, n_antennas)

        # Construct complex D-terms
        d_terms = np.zeros((n_antennas, 2), dtype=np.complex128)
        d_terms[:, 0] = d_amplitude * np.exp(1j * d_phase_p)
        d_terms[:, 1] = d_amplitude * np.exp(1j * d_phase_q)

        super().__init__(n_antennas, d_terms)

    def get_ixr(self, antenna_idx: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Calculate IXR for antenna(s).

        Parameters
        ----------
        antenna_idx : int, optional
            Antenna index. If None, returns IXR for all antennas.

        Returns
        -------
        ixr : float or np.ndarray
            Intrinsic cross-polarization ratio.
        """
        if antenna_idx is not None:
            d_avg = np.mean(np.abs(self.d_terms[antenna_idx]))
            return (1 + d_avg**2) / (2 * d_avg) if d_avg > 0 else np.inf
        else:
            d_avg = np.mean(np.abs(self.d_terms), axis=1)
            ixr = np.where(d_avg > 0, (1 + d_avg**2) / (2 * d_avg), np.inf)
            return ixr


class MuellerLeakageJones(PolarizationLeakageJones):
    """
    D-terms derived from Mueller matrix formalism.

    Useful for converting between Jones and Mueller representations
    of instrumental polarization.

    The Mueller matrix M and Jones matrix J are related by:
        M = A (J ⊗ J*) A^(-1)

    where A is the transformation matrix and ⊗ is the Kronecker product.

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    mueller_matrices : np.ndarray, optional
        Mueller matrices with shape (n_antennas, 4, 4).
    """

    def __init__(
        self,
        n_antennas: int,
        mueller_matrices: Optional[np.ndarray] = None
    ):
        if mueller_matrices is not None:
            # Extract D-terms from Mueller matrices
            d_terms = self._mueller_to_dterms(mueller_matrices)
        else:
            d_terms = None

        super().__init__(n_antennas, d_terms)

        self.mueller_matrices = mueller_matrices

    def _mueller_to_dterms(self, mueller: np.ndarray) -> np.ndarray:
        """
        Extract D-terms from Mueller matrices.

        For small leakage, the Mueller matrix has approximate form:
            M ≈ [[1,    0,     0,    0   ],
                 [0,    1,     Re(d), Im(d)],
                 [0,    -Re(d), 1,    0   ],
                 [0,    Im(d),  0,    1   ]]

        Parameters
        ----------
        mueller : np.ndarray
            Mueller matrices with shape (n_antennas, 4, 4).

        Returns
        -------
        d_terms : np.ndarray
            Complex leakage terms with shape (n_antennas, 2).
        """
        n_antennas = mueller.shape[0]
        d_terms = np.zeros((n_antennas, 2), dtype=np.complex128)

        for ant in range(n_antennas):
            M = mueller[ant]
            # Extract from M[1,2] and M[1,3] (Q->U and Q->V leakage)
            d_p = M[1, 2] + 1j * M[1, 3]
            # Extract from M[2,1] and M[3,1] (U->Q and V->Q leakage)
            d_q = -M[2, 1] + 1j * M[3, 1]

            d_terms[ant, 0] = d_p
            d_terms[ant, 1] = d_q

        return d_terms

    def to_mueller(self, antenna_idx: int) -> np.ndarray:
        """
        Convert Jones D-matrix to Mueller matrix.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.

        Returns
        -------
        mueller : np.ndarray
            4x4 Mueller matrix.
        """
        d_p = self.d_terms[antenna_idx, 0]
        d_q = self.d_terms[antenna_idx, 1]

        # First-order approximation for small d
        M = np.eye(4, dtype=float)
        M[1, 2] = np.real(d_p)
        M[1, 3] = np.imag(d_p)
        M[2, 1] = -np.real(d_q)
        M[3, 1] = np.imag(d_q)

        return M


class BeamSquintLeakageJones(PolarizationLeakageJones):
    """
    Direction-dependent D-terms from beam squint.

    Beam squint causes the two polarization beams to point in slightly
    different directions, leading to direction-dependent polarization
    leakage. This is particularly important for off-axis sources.

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    squint_x : np.ndarray
        X-pol beam pointing offset in radians, shape (n_antennas, 2).
        Each row is [delta_l, delta_m].
    squint_y : np.ndarray
        Y-pol beam pointing offset in radians, shape (n_antennas, 2).
    beam_width : float
        FWHM beam width in radians (for leakage calculation).
    """

    def __init__(
        self,
        n_antennas: int,
        squint_x: np.ndarray,
        squint_y: np.ndarray,
        beam_width: float
    ):
        self.squint_x = np.asarray(squint_x)
        self.squint_y = np.asarray(squint_y)
        self.beam_width = beam_width

        # Differential squint
        self.squint_diff = self.squint_x - self.squint_y

        # Base D-terms (at beam center)
        d_terms = np.zeros((n_antennas, 2), dtype=np.complex128)
        super().__init__(n_antennas, d_terms)

        # Override direction dependence
        self._is_direction_dependent = True

    @property
    def is_direction_dependent(self) -> bool:
        return self._is_direction_dependent

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        source_lm: Optional[np.ndarray] = None,
        **kwargs
    ) -> Any:
        """
        Compute direction-dependent D-matrix.

        The leakage increases with distance from beam center as:
            d(l,m) ≈ d_0 + α * (l * Δl + m * Δm) / θ_beam

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        source_idx : int
            Source index.
        freq_idx : int
            Frequency channel index.
        time_idx : int
            Time index.
        backend : ArrayBackend
            Compute backend.
        source_lm : np.ndarray, optional
            Source direction cosines (l, m).

        Returns
        -------
        jones : array
            2x2 complex Jones matrix.
        """
        xp = backend.xp

        if source_lm is None:
            # No direction info, use base D-terms
            return super().compute_jones(
                antenna_idx, source_idx, freq_idx, time_idx, backend
            )

        # Calculate direction-dependent leakage
        l, m = source_lm[0], source_lm[1]
        dl, dm = self.squint_diff[antenna_idx]

        # Leakage proportional to beam gradient at source position
        sigma = self.beam_width / (2 * np.sqrt(2 * np.log(2)))  # Gaussian sigma
        gradient_factor = (l * dl + m * dm) / sigma**2

        # D-terms increase away from beam center
        d_p = self.d_terms[antenna_idx, 0] + 0.1 * gradient_factor * (1 + 1j)
        d_q = self.d_terms[antenna_idx, 1] + 0.1 * gradient_factor * (1 - 1j)

        d_matrix = np.array([
            [1.0 + 0j, d_p],
            [d_q, 1.0 + 0j]
        ], dtype=np.complex128)

        return backend.asarray(d_matrix)
