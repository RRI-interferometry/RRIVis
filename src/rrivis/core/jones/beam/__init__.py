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

from collections.abc import Callable
from typing import Any

import numpy as np

from rrivis.core.jones.base import JonesTerm

# Re-export analytic beam patterns
from rrivis.core.jones.beam.analytic import (
    BEAM_PATTERN_FUNCTIONS,
    HPBW_FUNCTIONS,
    AntennaType,
    BeamPatternType,
    airy_disk_pattern,
    calculate_airy_beam_area,
    calculate_beam_pattern,
    calculate_cosine_beam_area,
    calculate_exponential_beam_area,
    calculate_gaussian_beam_area_EBeam,
    calculate_hpbw_for_antenna_type,
    convert_angle_for_display,
    cosine_tapered_pattern,
    exponential_tapered_pattern,
    gaussian_A_theta_EBeam,
    get_beam_pattern_function,
    get_hpbw_function,
    short_dipole_jones,
)

# Re-export FITS beam handling
from rrivis.core.jones.beam.fits import (
    BeamFITSHandler,
    BeamManager,
    astropy_az_to_uvbeam_az,
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
        source_idx: int | None,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs,
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
            **kwargs,
        )

        # Convert to backend array
        return backend.asarray(E, dtype=np.complex128)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "n_sources": len(self.source_altaz),
                "n_frequencies": len(self.frequencies),
                "beam_model": self.beam_model.__name__
                if hasattr(self.beam_model, "__name__")
                else str(type(self.beam_model)),
            }
        )
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
        wavelength: float | None = None,
        diameter: float | None = None,
        taper_exponent: float = 1.0,
        taper_dB: float = 10.0,
        hpbw_per_antenna: dict[Any, float] | None = None,
        beam_type_per_antenna: dict[Any, str] | None = None,
    ):
        """Initialize analytic beam.

        Args:
            source_altaz: Source alt-az coordinates (N_sources, 2) in radians
            frequencies: Frequencies in Hz (N_freq,)
            hpbw_radians: Half-power beam width in radians (default for all antennas)
            beam_type: 'gaussian', 'airy', 'cosine', or 'exponential' (default)
            wavelength: Observation wavelength in meters (required for 'airy')
            diameter: Antenna diameter in meters (required for 'airy')
            taper_exponent: Cosine taper exponent (for 'cosine', default 1.0)
            taper_dB: Exponential taper in dB (for 'exponential', default 10.0)
            hpbw_per_antenna: Per-antenna HPBW map {ant_number: hpbw_rad}.
                Falls back to hpbw_radians when None or missing.
            beam_type_per_antenna: Per-antenna beam type map {ant_number: str}.
                Falls back to beam_type when None or missing.
        """
        self.hpbw_radians = hpbw_radians
        self.beam_type = beam_type
        self.wavelength = wavelength
        self.diameter = diameter
        self.taper_exponent = taper_exponent
        self.taper_dB = taper_dB
        self.hpbw_per_antenna = hpbw_per_antenna
        self.beam_type_per_antenna = beam_type_per_antenna

        # Create beam model function
        def beam_model(antenna_idx, zenith_angle, azimuth, frequency, time_idx, **kw):
            ant_num = kw.get("antenna_number", antenna_idx)
            hpbw = self._get_hpbw_for_antenna(ant_num)
            btype = self._get_beam_type_for_antenna(ant_num)
            return self._compute_analytic_beam(
                zenith_angle, azimuth=azimuth, hpbw=hpbw, beam_type=btype
            )

        super().__init__(beam_model, source_altaz, frequencies)

    def _get_hpbw_for_antenna(self, ant_num: Any) -> float:
        """Get HPBW for a specific antenna, falling back to default."""
        if self.hpbw_per_antenna is not None and ant_num in self.hpbw_per_antenna:
            return self.hpbw_per_antenna[ant_num]
        return self.hpbw_radians

    def _get_beam_type_for_antenna(self, ant_num: Any) -> str:
        """Get beam type for a specific antenna, falling back to default."""
        if (
            self.beam_type_per_antenna is not None
            and ant_num in self.beam_type_per_antenna
        ):
            return self.beam_type_per_antenna[ant_num]
        return self.beam_type

    def _compute_analytic_beam(
        self,
        zenith_angle,
        azimuth=None,
        hpbw: float | None = None,
        beam_type: str | None = None,
    ) -> np.ndarray:
        """Compute analytic beam pattern using functions from ``analytic.py``.

        Args:
            zenith_angle: Angle from zenith in radians (scalar or array)
            azimuth: Azimuth angle in radians (required for 'short_dipole')
            hpbw: Override HPBW in radians (defaults to self.hpbw_radians)
            beam_type: Override beam type (defaults to self.beam_type)

        Returns:
            2x2 Jones matrix (2, 2) for scalar,
            or (n_sources, 2, 2) for array input.
            Diagonal for standard beams, full 2x2 for 'short_dipole'.
        """
        btype = beam_type if beam_type is not None else self.beam_type
        hpbw_rad = hpbw if hpbw is not None else self.hpbw_radians

        # Short dipole returns full 2x2 non-diagonal Jones
        if btype == "short_dipole":
            if azimuth is None:
                raise ValueError("short_dipole beam requires azimuth angles")
            from rrivis.core.jones.beam.analytic import short_dipole_jones

            return short_dipole_jones(zenith_angle, azimuth)

        # Detect scalar input before pattern functions may wrap it
        input_is_scalar = np.ndim(zenith_angle) == 0

        if btype == "uniform":
            amplitude = np.ones_like(np.asarray(zenith_angle, dtype=float))
        elif btype == "gaussian":
            amplitude = gaussian_A_theta_EBeam(zenith_angle, hpbw_rad)
        elif btype == "airy":
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
                    (2 * np.sinc(2 * np.pi * za / (hpbw_rad * np.pi))) ** 2,
                )
        elif btype == "cosine":
            amplitude = cosine_tapered_pattern(
                zenith_angle,
                hpbw_rad,
                taper_exponent=self.taper_exponent,
            )
        elif btype == "exponential":
            amplitude = exponential_tapered_pattern(
                zenith_angle,
                hpbw_rad,
                taper_dB=self.taper_dB,
            )
        else:
            amplitude = np.ones_like(np.asarray(zenith_angle, dtype=float))

        amplitude = np.asarray(amplitude).ravel()

        if input_is_scalar:
            # Scalar: return (2, 2)
            a = float(amplitude[0]) if amplitude.size > 0 else 1.0
            return np.array(
                [
                    [a, 0],
                    [0, a],
                ],
                dtype=np.complex128,
            )
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

        Returns (n_sources, 2, 2) matrices (diagonal for standard beams,
        full 2x2 for non-diagonal beam types like 'short_dipole').
        """
        alts = self.source_altaz[:n_sources, 0]
        azs = self.source_altaz[:n_sources, 1]
        zenith_angles = np.pi / 2 - alts

        ant_num = kwargs.get("antenna_number", antenna_idx)
        hpbw = self._get_hpbw_for_antenna(ant_num)
        btype = self._get_beam_type_for_antenna(ant_num)

        return backend.asarray(
            self._compute_analytic_beam(
                zenith_angles, azimuth=azs, hpbw=hpbw, beam_type=btype
            ),
            dtype=np.complex128,
        )

    def is_diagonal(self) -> bool:
        # Non-diagonal for short_dipole; also check per-antenna types
        if self.beam_type == "short_dipole":
            return False
        if self.beam_type_per_antenna:
            for btype in self.beam_type_per_antenna.values():
                if btype == "short_dipole":
                    return False
        return True

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "hpbw_radians": self.hpbw_radians,
                "beam_type": self.beam_type,
            }
        )
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

    def get_config(self) -> dict[str, Any]:
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
    "short_dipole_jones",
    # FITS beam handling
    "astropy_az_to_uvbeam_az",
    "BeamFITSHandler",
    "BeamManager",
]
