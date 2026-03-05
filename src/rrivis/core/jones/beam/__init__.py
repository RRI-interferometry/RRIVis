"""Primary beam Jones term (E matrix) and beam pattern models.

This sub-package consolidates:
- Analytic beam patterns (``analytic``) — Gaussian, Airy, cosine, exponential
- FITS-based beam file handling (``fits``) — UVBeam interpolation, BeamManager
- Jones matrix wrappers (``BeamJones``, ``AnalyticBeamJones``) for RIME integration

The E term represents the primary beam voltage pattern of the antenna.
It is direction-dependent and generally a full 2x2 matrix due to
cross-polarization coupling and beam squint.

Examples
--------
>>> from rrivis.core.jones.beam import BeamJones, AnalyticBeamJones
>>> from rrivis.core.jones.beam import gaussian_A_theta_EBeam, BeamManager
>>> from rrivis.core.jones.beam.analytic import AntennaType
>>> from rrivis.core.jones.beam.fits import BeamFITSHandler
"""

from typing import Any, Callable, Dict, Optional
import numpy as np

from rrivis.core.jones.base import JonesTerm

# Re-export analytic beam patterns
from rrivis.core.jones.beam.analytic import (
    AntennaType,
    BeamPatternType,
    HPBW_FUNCTIONS,
    BEAM_PATTERN_FUNCTIONS,
    gaussian_A_theta_EBeam,
    airy_disk_pattern,
    cosine_tapered_pattern,
    exponential_tapered_pattern,
    calculate_gaussian_beam_area_EBeam,
    calculate_airy_beam_area,
    calculate_cosine_beam_area,
    calculate_exponential_beam_area,
    calculate_hpbw_for_antenna_type,
    get_hpbw_function,
    get_beam_pattern_function,
    calculate_beam_pattern,
    convert_angle_for_display,
)

# Re-export FITS beam handling
from rrivis.core.jones.beam.fits import (
    astropy_az_to_uvbeam_az,
    BeamFITSHandler,
    BeamManager,
)


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
    """Beam Jones term using analytic beam patterns from ``analytic.py``.

    Delegates to the proper beam pattern functions:
    - ``gaussian_A_theta_EBeam`` for Gaussian beams
    - ``airy_disk_pattern`` for Airy disk (requires ``wavelength`` and ``diameter``)
    - ``cosine_tapered_pattern`` for cosine-tapered beams
    - ``exponential_tapered_pattern`` for exponential-tapered beams

    Assumes diagonal beam (no cross-polarization).

    Args:
        source_altaz: Source alt-az coordinates (N_sources, 2) in radians
        frequencies: Frequencies in Hz (N_freq,)
        hpbw_radians: Half-power beam width in radians
        beam_type: 'gaussian', 'airy', 'cosine', or 'exponential'
        wavelength: Observation wavelength in meters (required for 'airy')
        diameter: Antenna diameter in meters (required for 'airy')
        taper_exponent: Cosine taper exponent (for 'cosine', default 1.0)
        taper_dB: Exponential taper in dB (for 'exponential', default 10.0)
    """

    def __init__(
        self,
        source_altaz: np.ndarray,
        frequencies: np.ndarray,
        hpbw_radians: float,
        beam_type: str = "gaussian",
        wavelength: Optional[float] = None,
        diameter: Optional[float] = None,
        taper_exponent: float = 1.0,
        taper_dB: float = 10.0,
    ):
        """Initialize analytic beam.

        Args:
            source_altaz: Source alt-az coordinates (N_sources, 2) in radians
            frequencies: Frequencies in Hz (N_freq,)
            hpbw_radians: Half-power beam width in radians
            beam_type: 'gaussian', 'airy', 'cosine', or 'exponential'
            wavelength: Observation wavelength in meters (required for 'airy')
            diameter: Antenna diameter in meters (required for 'airy')
            taper_exponent: Cosine taper exponent (for 'cosine', default 1.0)
            taper_dB: Exponential taper in dB (for 'exponential', default 10.0)
        """
        self.hpbw_radians = hpbw_radians
        self.beam_type = beam_type
        self.wavelength = wavelength
        self.diameter = diameter
        self.taper_exponent = taper_exponent
        self.taper_dB = taper_dB

        # Create beam model function
        def beam_model(antenna_idx, zenith_angle, azimuth, frequency, time_idx, **kw):
            return self._compute_analytic_beam(zenith_angle)

        super().__init__(beam_model, source_altaz, frequencies)

    def _compute_analytic_beam(self, zenith_angle) -> np.ndarray:
        """Compute analytic beam pattern using functions from ``analytic.py``.

        Args:
            zenith_angle: Angle from zenith in radians (scalar or array)

        Returns:
            2x2 diagonal Jones matrix (2, 2) for scalar,
            or (n_sources, 2, 2) for array input
        """
        # Detect scalar input before pattern functions may wrap it
        input_is_scalar = np.ndim(zenith_angle) == 0

        if self.beam_type == "uniform":
            amplitude = np.ones_like(np.asarray(zenith_angle, dtype=float))
        elif self.beam_type == "gaussian":
            amplitude = gaussian_A_theta_EBeam(zenith_angle, self.hpbw_radians)
        elif self.beam_type == "airy":
            if self.wavelength is not None and self.diameter is not None:
                amplitude = airy_disk_pattern(
                    zenith_angle, self.wavelength, self.diameter
                )
            else:
                # Fallback approximation when wavelength/diameter not provided
                za = np.asarray(zenith_angle, dtype=float)
                amplitude = np.where(
                    np.abs(za) < 1e-10,
                    1.0,
                    (2 * np.sinc(2 * np.pi * za / (self.hpbw_radians * np.pi))) ** 2,
                )
        elif self.beam_type == "cosine":
            amplitude = cosine_tapered_pattern(
                zenith_angle, self.hpbw_radians,
                taper_exponent=self.taper_exponent,
            )
        elif self.beam_type == "exponential":
            amplitude = exponential_tapered_pattern(
                zenith_angle, self.hpbw_radians,
                taper_dB=self.taper_dB,
            )
        else:
            amplitude = np.ones_like(np.asarray(zenith_angle, dtype=float))

        amplitude = np.asarray(amplitude).ravel()

        if input_is_scalar:
            # Scalar: return (2, 2)
            a = float(amplitude[0]) if amplitude.size > 0 else 1.0
            return np.array([
                [a, 0],
                [0, a],
            ], dtype=np.complex128)
        else:
            # Array: return (n_sources, 2, 2)
            n = amplitude.shape[0]
            jones = np.zeros((n, 2, 2), dtype=np.complex128)
            jones[:, 0, 0] = amplitude
            jones[:, 1, 1] = amplitude
            return jones

    def compute_jones_all_sources(
        self,
        antenna_idx: int,
        n_sources: int,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs,
    ) -> Any:
        """Compute analytic beam Jones for all sources at once.

        Returns (n_sources, 2, 2) diagonal matrices.
        """
        # Get all zenith angles at once
        alts = self.source_altaz[:n_sources, 0]
        zenith_angles = np.pi / 2 - alts

        # _compute_analytic_beam now handles arrays
        return backend.asarray(
            self._compute_analytic_beam(zenith_angles), dtype=np.complex128,
        )

    def is_diagonal(self) -> bool:
        return True  # Analytic beams are diagonal

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hpbw_radians": self.hpbw_radians,
            "beam_type": self.beam_type,
        })
        if self.wavelength is not None:
            config["wavelength"] = self.wavelength
        if self.diameter is not None:
            config["diameter"] = self.diameter
        return config


class FITSBeamJones(BeamJones):
    """Beam Jones term using FITS beam files via BeamManager.

    Wraps a BeamManager instance to provide FITS-based beam responses
    within the JonesChain framework. Falls back to identity if the
    BeamManager returns None.

    Args:
        beam_manager: BeamManager instance for FITS beam interpolation
        source_altaz: Source coordinates in alt-az, shape (N_sources, 2) [alt, az] in radians
        frequencies: Observation frequencies in Hz, shape (N_freq,)
    """

    def __init__(
        self,
        beam_manager: Any,
        source_altaz: np.ndarray,
        frequencies: np.ndarray,
    ):
        self._beam_manager = beam_manager

        def beam_model(
            antenna_idx: int,
            zenith_angle: float,
            azimuth: float,
            frequency: float,
            time_idx: int,
            **kw: Any,
        ) -> np.ndarray:
            alt = np.pi / 2 - zenith_angle
            antenna_number = kw.get("antenna_number", antenna_idx)
            jones = self._beam_manager.get_jones_matrix(
                antenna_number=antenna_number,
                alt_rad=alt,
                az_rad=azimuth,
                freq_hz=frequency,
                location=None,
                time=None,
            )
            if jones is None:
                return np.eye(2, dtype=np.complex128)
            return jones

        super().__init__(beam_model, source_altaz, frequencies)

    def compute_jones_all_sources(
        self,
        antenna_idx: int,
        n_sources: int,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs: Any,
    ) -> Any:
        """Vectorized FITS beam lookup for all sources.

        Args:
            antenna_idx: Antenna index
            n_sources: Number of sources
            freq_idx: Frequency index
            time_idx: Time index
            backend: Array backend
            **kwargs: Must include 'antenna_number' for BeamManager lookup

        Returns:
            Jones matrices, shape (n_sources, 2, 2)
        """
        alts = self.source_altaz[:n_sources, 0]
        azs = self.source_altaz[:n_sources, 1]
        frequency = self.frequencies[freq_idx]
        antenna_number = kwargs.get("antenna_number", antenna_idx)

        jones = self._beam_manager.get_jones_matrix(
            antenna_number=antenna_number,
            alt_rad=alts,
            az_rad=azs,
            freq_hz=frequency,
            location=None,
            time=None,
        )

        if jones is None:
            result = np.zeros((n_sources, 2, 2), dtype=np.complex128)
            result[:, 0, 0] = 1.0
            result[:, 1, 1] = 1.0
            return backend.asarray(result, dtype=np.complex128)

        if jones.ndim == 2:
            jones = np.tile(jones, (n_sources, 1, 1))
        return backend.asarray(jones, dtype=np.complex128)

    def is_diagonal(self) -> bool:
        return False  # FITS beams can have cross-pol

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["beam_source"] = "fits"
        return config


__all__ = [
    # Jones matrix classes
    "BeamJones",
    "AnalyticBeamJones",
    "FITSBeamJones",
    # Analytic beam patterns
    "AntennaType",
    "BeamPatternType",
    "HPBW_FUNCTIONS",
    "BEAM_PATTERN_FUNCTIONS",
    "gaussian_A_theta_EBeam",
    "airy_disk_pattern",
    "cosine_tapered_pattern",
    "exponential_tapered_pattern",
    "calculate_gaussian_beam_area_EBeam",
    "calculate_airy_beam_area",
    "calculate_cosine_beam_area",
    "calculate_exponential_beam_area",
    "calculate_hpbw_for_antenna_type",
    "get_hpbw_function",
    "get_beam_pattern_function",
    "calculate_beam_pattern",
    "convert_angle_for_display",
    # FITS beam handling
    "astropy_az_to_uvbeam_az",
    "BeamFITSHandler",
    "BeamManager",
]
