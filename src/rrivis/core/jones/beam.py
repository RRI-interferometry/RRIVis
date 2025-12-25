"""Primary beam Jones term (E matrix).

The E term represents the primary beam voltage pattern of the antenna.
It is direction-dependent and generally a full 2x2 matrix due to
cross-polarization coupling and beam squint.
"""

from typing import Any, Callable, Dict, Optional
import numpy as np

from rrivis.core.jones.base import JonesTerm


class BeamJones(JonesTerm):
    """Primary beam voltage pattern Jones matrix.

    E = [[E_xx, E_xy],
         [E_yx, E_yy]]

    The beam pattern describes how antenna sensitivity varies across the sky.
    This is generally a full 2x2 matrix (non-diagonal) due to:
    - Cross-polarization coupling
    - Beam squint (different patterns for X and Y polarizations)
    - Asymmetric feeds

    Args:
        beam_model: Callable that returns 2x2 Jones matrix.
                   Signature: (antenna_idx, zenith_angle, azimuth, frequency, **kw) -> (2, 2)
        source_altaz: Source coordinates in alt-az, shape (N_sources, 2) [alt, az] in radians
        frequencies: Observation frequencies in Hz, shape (N_freq,)
    """

    def __init__(
        self,
        beam_model: Callable,
        source_altaz: np.ndarray,
        frequencies: np.ndarray,
    ):
        """Initialize beam Jones term.

        Args:
            beam_model: Function computing beam Jones matrix
            source_altaz: Array of source alt-az coordinates (N_sources, 2) in radians
            frequencies: Frequencies in Hz (N_freq,)
        """
        self.beam_model = beam_model
        self.source_altaz = np.asarray(source_altaz)
        self.frequencies = np.asarray(frequencies)

    @property
    def name(self) -> str:
        return "E"

    @property
    def is_direction_dependent(self) -> bool:
        return True  # Beam pattern varies across sky

    @property
    def is_time_dependent(self) -> bool:
        return True  # For alt-az, source position in beam frame changes

    @property
    def is_frequency_dependent(self) -> bool:
        return True  # Beam pattern scales with frequency

    def is_diagonal(self) -> bool:
        return False  # Generally has cross-pol terms

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: Optional[int],
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs
    ) -> Any:
        """Compute primary beam Jones matrix.

        Args:
            antenna_idx: Antenna index
            source_idx: Source index
            freq_idx: Frequency index
            time_idx: Time index
            backend: Array backend
            **kwargs: Additional parameters passed to beam_model

        Returns:
            2x2 complex Jones matrix
        """
        # Get source coordinates
        alt, az = self.source_altaz[source_idx]
        zenith_angle = np.pi / 2 - alt  # Convert altitude to zenith angle

        # Get frequency
        frequency = self.frequencies[freq_idx]

        # Compute beam Jones matrix using provided model
        E = self.beam_model(
            antenna_idx=antenna_idx,
            zenith_angle=zenith_angle,
            azimuth=az,
            frequency=frequency,
            time_idx=time_idx,
            **kwargs
        )

        # Convert to backend array
        return backend.asarray(E, dtype=np.complex128)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "n_sources": len(self.source_altaz),
            "n_frequencies": len(self.frequencies),
            "beam_model": self.beam_model.__name__
            if hasattr(self.beam_model, '__name__')
            else str(type(self.beam_model)),
        })
        return config


class AnalyticBeamJones(BeamJones):
    """Beam Jones term using analytic beam patterns.

    Uses Gaussian, Airy, or other analytic beam models.
    Assumes diagonal beam (no cross-polarization).
    """

    def __init__(
        self,
        source_altaz: np.ndarray,
        frequencies: np.ndarray,
        hpbw_radians: float,
        beam_type: str = "gaussian",
    ):
        """Initialize analytic beam.

        Args:
            source_altaz: Source alt-az coordinates (N_sources, 2) in radians
            frequencies: Frequencies in Hz (N_freq,)
            hpbw_radians: Half-power beam width in radians
            beam_type: 'gaussian', 'airy', or 'cosine'
        """
        self.hpbw_radians = hpbw_radians
        self.beam_type = beam_type
        self._sigma = hpbw_radians / (2 * np.sqrt(2 * np.log(2)))

        # Create beam model function
        def beam_model(antenna_idx, zenith_angle, azimuth, frequency, time_idx, **kw):
            return self._compute_analytic_beam(zenith_angle)

        super().__init__(beam_model, source_altaz, frequencies)

    def _compute_analytic_beam(self, zenith_angle: float) -> np.ndarray:
        """Compute analytic beam pattern.

        Args:
            zenith_angle: Angle from zenith in radians

        Returns:
            2x2 diagonal Jones matrix
        """
        if self.beam_type == "gaussian":
            # Gaussian beam: A(θ) = exp(-(θ/σ)²)
            amplitude = np.exp(-(zenith_angle / self._sigma) ** 2)
        elif self.beam_type == "cosine":
            # Cosine beam: A(θ) = cos(θ)
            amplitude = np.cos(zenith_angle) if zenith_angle < np.pi/2 else 0.0
        elif self.beam_type == "airy":
            # Simplified Airy pattern approximation
            if zenith_angle < 1e-10:
                amplitude = 1.0
            else:
                x = 2 * np.pi * zenith_angle / self.hpbw_radians
                amplitude = (2 * np.sinc(x / np.pi)) ** 2
        else:
            amplitude = 1.0

        # Diagonal beam (no cross-pol)
        return np.array([
            [amplitude, 0],
            [0, amplitude],
        ], dtype=np.complex128)

    def is_diagonal(self) -> bool:
        return True  # Analytic beams are diagonal

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hpbw_radians": self.hpbw_radians,
            "beam_type": self.beam_type,
        })
        return config
