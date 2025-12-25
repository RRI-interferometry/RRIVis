"""
Ionosphere Jones term (Z) for ionospheric propagation effects.

The Z-Jones term models the effects of the Earth's ionosphere on
radio wave propagation, including:
- Faraday rotation (rotation of polarization plane)
- Dispersive delay (frequency-dependent path delay)
- Phase rotation
- Scintillation (amplitude/phase fluctuations)

These effects are direction-dependent and vary with:
- Total Electron Content (TEC) along the line of sight
- Earth's magnetic field geometry
- Frequency (effects scale as ν^(-2))
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from abc import abstractmethod

from .base import JonesTerm


# Physical constants
ELECTRON_CHARGE = 1.602176634e-19  # C
ELECTRON_MASS = 9.1093837015e-31  # kg
SPEED_OF_LIGHT = 299792458.0  # m/s
EPSILON_0 = 8.8541878128e-12  # F/m


class IonosphereJones(JonesTerm):
    """
    Ionospheric propagation effects Jones term.

    Models Faraday rotation and dispersive phase due to ionospheric TEC.

    The ionospheric rotation matrix has the form:
        Z = exp(i*φ_ion) * [[cos(RM*λ²), -sin(RM*λ²)],
                            [sin(RM*λ²),  cos(RM*λ²)]]

    where RM is the Rotation Measure and λ is the wavelength.

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    n_sources : int
        Number of sources.
    frequencies : np.ndarray
        Observation frequencies in Hz.
    tec_values : np.ndarray, optional
        Total Electron Content in TECU (10^16 electrons/m²),
        shape (n_antennas, n_sources, n_times) or (n_sources, n_times).
    rotation_measure : np.ndarray, optional
        Rotation Measure in rad/m², shape same as tec_values.
    times : np.ndarray, optional
        Observation times for time-varying ionosphere.
    """

    def __init__(
        self,
        n_antennas: int,
        n_sources: int,
        frequencies: np.ndarray,
        tec_values: Optional[np.ndarray] = None,
        rotation_measure: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None
    ):
        self.n_antennas = n_antennas
        self.n_sources = n_sources
        self.frequencies = np.asarray(frequencies)
        self.n_freq = len(self.frequencies)
        self.wavelengths = SPEED_OF_LIGHT / self.frequencies

        self.times = times if times is not None else np.array([0.0])
        self.n_times = len(self.times)

        # TEC values (default: zero ionosphere)
        if tec_values is None:
            self.tec_values = np.zeros((n_antennas, n_sources, self.n_times))
        else:
            tec_values = np.asarray(tec_values)
            if tec_values.ndim == 2:
                # Same TEC for all antennas
                self.tec_values = np.broadcast_to(
                    tec_values[np.newaxis, :, :],
                    (n_antennas, n_sources, self.n_times)
                ).copy()
            else:
                self.tec_values = tec_values

        # Rotation Measure (default: zero)
        if rotation_measure is None:
            self.rotation_measure = np.zeros((n_antennas, n_sources, self.n_times))
        else:
            rotation_measure = np.asarray(rotation_measure)
            if rotation_measure.ndim == 2:
                self.rotation_measure = np.broadcast_to(
                    rotation_measure[np.newaxis, :, :],
                    (n_antennas, n_sources, self.n_times)
                ).copy()
            else:
                self.rotation_measure = rotation_measure

        # Precompute phase constant
        # Phase = -8.448e9 * TEC / ν² (in radians, TEC in TECU, ν in Hz)
        self.phase_constant = -8.448e9

    @property
    def name(self) -> str:
        return "Z"

    @property
    def is_direction_dependent(self) -> bool:
        return True  # TEC varies with direction

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
        Compute ionospheric Jones matrix.

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

        Returns
        -------
        jones : array
            2x2 complex Jones matrix.
        """
        xp = backend.xp

        tec = self.tec_values[antenna_idx, source_idx, time_idx]
        rm = self.rotation_measure[antenna_idx, source_idx, time_idx]
        freq = self.frequencies[freq_idx]
        wavelength = self.wavelengths[freq_idx]

        # Dispersive phase
        phi_ion = self.phase_constant * tec / (freq * freq)

        # Faraday rotation angle
        theta_faraday = rm * wavelength * wavelength

        # Construct Jones matrix
        cos_theta = np.cos(theta_faraday)
        sin_theta = np.sin(theta_faraday)
        phase_factor = np.exp(1j * phi_ion)

        z_matrix = phase_factor * np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ], dtype=np.complex128)

        return backend.asarray(z_matrix)

    def set_tec(
        self,
        tec_values: np.ndarray,
        antenna_idx: Optional[int] = None,
        source_idx: Optional[int] = None,
        time_idx: Optional[int] = None
    ) -> None:
        """
        Set TEC values.

        Parameters
        ----------
        tec_values : np.ndarray
            TEC values in TECU.
        antenna_idx : int, optional
            Antenna index. If None, sets for all antennas.
        source_idx : int, optional
            Source index. If None, sets for all sources.
        time_idx : int, optional
            Time index. If None, sets for all times.
        """
        if antenna_idx is None and source_idx is None and time_idx is None:
            self.tec_values = np.asarray(tec_values)
        else:
            ant_slice = antenna_idx if antenna_idx is not None else slice(None)
            src_slice = source_idx if source_idx is not None else slice(None)
            time_slice = time_idx if time_idx is not None else slice(None)
            self.tec_values[ant_slice, src_slice, time_slice] = tec_values

    def set_rotation_measure(
        self,
        rm_values: np.ndarray,
        antenna_idx: Optional[int] = None,
        source_idx: Optional[int] = None,
        time_idx: Optional[int] = None
    ) -> None:
        """
        Set Rotation Measure values.

        Parameters
        ----------
        rm_values : np.ndarray
            RM values in rad/m².
        antenna_idx : int, optional
            Antenna index.
        source_idx : int, optional
            Source index.
        time_idx : int, optional
            Time index.
        """
        if antenna_idx is None and source_idx is None and time_idx is None:
            self.rotation_measure = np.asarray(rm_values)
        else:
            ant_slice = antenna_idx if antenna_idx is not None else slice(None)
            src_slice = source_idx if source_idx is not None else slice(None)
            time_slice = time_idx if time_idx is not None else slice(None)
            self.rotation_measure[ant_slice, src_slice, time_slice] = rm_values

    def get_dispersive_delay(
        self,
        antenna_idx: int,
        source_idx: int,
        time_idx: int,
        freq_idx: int
    ) -> float:
        """
        Get dispersive delay due to ionosphere.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        source_idx : int
            Source index.
        time_idx : int
            Time index.
        freq_idx : int
            Frequency index.

        Returns
        -------
        delay : float
            Dispersive delay in seconds.
        """
        tec = self.tec_values[antenna_idx, source_idx, time_idx]
        freq = self.frequencies[freq_idx]

        # Delay = 1.34e-7 * TEC / ν² (TEC in TECU, ν in Hz, result in seconds)
        delay = 1.34e-7 * tec / (freq * freq)
        return delay

    def get_faraday_rotation(
        self,
        antenna_idx: int,
        source_idx: int,
        time_idx: int,
        freq_idx: int
    ) -> float:
        """
        Get Faraday rotation angle.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        source_idx : int
            Source index.
        time_idx : int
            Time index.
        freq_idx : int
            Frequency index.

        Returns
        -------
        angle : float
            Faraday rotation angle in radians.
        """
        rm = self.rotation_measure[antenna_idx, source_idx, time_idx]
        wavelength = self.wavelengths[freq_idx]
        return rm * wavelength * wavelength


class TurbulentIonosphereJones(IonosphereJones):
    """
    Ionosphere with Kolmogorov turbulence model.

    Models ionospheric phase fluctuations using a Kolmogorov power
    spectrum, appropriate for scintillation effects.

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    n_sources : int
        Number of sources.
    frequencies : np.ndarray
        Observation frequencies in Hz.
    r_diff : float
        Diffractive scale in meters (typical: 1-10 km).
    r_ref : float
        Refractive scale in meters (typical: 10-100 km).
    mean_tec : float
        Mean TEC in TECU.
    mean_rm : float
        Mean Rotation Measure in rad/m².
    times : np.ndarray, optional
        Observation times.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_antennas: int,
        n_sources: int,
        frequencies: np.ndarray,
        r_diff: float = 5000.0,
        r_ref: float = 50000.0,
        mean_tec: float = 10.0,
        mean_rm: float = 1.0,
        times: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ):
        self.r_diff = r_diff
        self.r_ref = r_ref
        self.mean_tec = mean_tec
        self.mean_rm = mean_rm

        if seed is not None:
            np.random.seed(seed)

        times = times if times is not None else np.array([0.0])
        n_times = len(times)

        # Generate turbulent TEC fluctuations
        tec_fluctuations = self._generate_turbulent_field(
            n_antennas, n_sources, n_times
        )
        tec_values = mean_tec + tec_fluctuations

        # RM correlated with TEC
        rm_fluctuations = tec_fluctuations * (mean_rm / mean_tec)
        rotation_measure = mean_rm + rm_fluctuations

        super().__init__(
            n_antennas, n_sources, frequencies,
            tec_values=tec_values,
            rotation_measure=rotation_measure,
            times=times
        )

    def _generate_turbulent_field(
        self,
        n_antennas: int,
        n_sources: int,
        n_times: int
    ) -> np.ndarray:
        """
        Generate Kolmogorov-like turbulent TEC field.

        The structure function for Kolmogorov turbulence is:
            D(r) = (r / r_diff)^(5/3)

        Parameters
        ----------
        n_antennas : int
            Number of antennas.
        n_sources : int
            Number of sources.
        n_times : int
            Number of time samples.

        Returns
        -------
        fluctuations : np.ndarray
            TEC fluctuations relative to mean.
        """
        # Simplified model: random phases with Kolmogorov-like statistics
        # True implementation would use proper spatial structure

        # RMS fluctuations (typical: 0.1-1 TECU for quiet ionosphere)
        tec_rms = 0.1 * self.mean_tec

        # Generate correlated Gaussian fluctuations
        fluctuations = np.random.normal(0, tec_rms, (n_antennas, n_sources, n_times))

        return fluctuations

    def get_scintillation_index(
        self,
        antenna_idx: int,
        source_idx: int
    ) -> float:
        """
        Estimate scintillation index S4.

        S4 = RMS(I) / <I> is the normalized intensity fluctuation.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        source_idx : int
            Source index.

        Returns
        -------
        s4 : float
            Scintillation index.
        """
        tec_series = self.tec_values[antenna_idx, source_idx, :]
        tec_var = np.var(tec_series)
        tec_mean = np.mean(tec_series)

        # Approximate S4 from TEC variance
        # S4² ≈ (TEC_var / TEC_mean)² for weak scintillation
        if tec_mean > 0:
            return np.sqrt(tec_var) / tec_mean
        return 0.0


class GPSIonosphereJones(IonosphereJones):
    """
    Ionosphere model using GPS-derived TEC maps.

    Interpolates TEC from global ionospheric maps (GIM) or local
    GPS measurements.

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    n_sources : int
        Number of sources.
    frequencies : np.ndarray
        Observation frequencies in Hz.
    tec_map : np.ndarray
        TEC map values with shape (n_lat, n_lon, n_times).
    lat_grid : np.ndarray
        Latitude grid in radians.
    lon_grid : np.ndarray
        Longitude grid in radians.
    antenna_positions : np.ndarray
        Antenna (lat, lon) in radians, shape (n_antennas, 2).
    source_positions : np.ndarray
        Source (az, el) in radians, shape (n_sources, 2).
    pierce_point_height : float
        Ionospheric pierce point height in meters (default: 350 km).
    times : np.ndarray, optional
        Observation times.
    """

    def __init__(
        self,
        n_antennas: int,
        n_sources: int,
        frequencies: np.ndarray,
        tec_map: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        antenna_positions: np.ndarray,
        source_positions: np.ndarray,
        pierce_point_height: float = 350e3,
        times: Optional[np.ndarray] = None
    ):
        self.tec_map = np.asarray(tec_map)
        self.lat_grid = np.asarray(lat_grid)
        self.lon_grid = np.asarray(lon_grid)
        self.antenna_positions = np.asarray(antenna_positions)
        self.source_positions = np.asarray(source_positions)
        self.pierce_point_height = pierce_point_height

        times = times if times is not None else np.arange(tec_map.shape[2])
        n_times = len(times)

        # Interpolate TEC values at pierce points
        tec_values = self._interpolate_tec(n_antennas, n_sources, n_times)

        # Estimate RM from TEC (simplified model)
        # RM ≈ 2.6e-13 * B_parallel * TEC
        b_parallel = 30e-6  # Typical parallel B field in Tesla
        rotation_measure = 2.6e-13 * b_parallel * tec_values * 1e16  # TECU to electrons/m²

        super().__init__(
            n_antennas, n_sources, frequencies,
            tec_values=tec_values,
            rotation_measure=rotation_measure,
            times=times
        )

    def _interpolate_tec(
        self,
        n_antennas: int,
        n_sources: int,
        n_times: int
    ) -> np.ndarray:
        """
        Interpolate TEC at ionospheric pierce points.

        Parameters
        ----------
        n_antennas : int
            Number of antennas.
        n_sources : int
            Number of sources.
        n_times : int
            Number of time samples.

        Returns
        -------
        tec : np.ndarray
            Interpolated TEC values.
        """
        from scipy.interpolate import RegularGridInterpolator

        tec_values = np.zeros((n_antennas, n_sources, n_times))

        for t in range(n_times):
            # Create interpolator for this time
            interpolator = RegularGridInterpolator(
                (self.lat_grid, self.lon_grid),
                self.tec_map[:, :, t],
                method='linear',
                bounds_error=False,
                fill_value=None
            )

            for ant in range(n_antennas):
                ant_lat, ant_lon = self.antenna_positions[ant]

                for src in range(n_sources):
                    az, el = self.source_positions[src]

                    # Calculate pierce point
                    pp_lat, pp_lon = self._compute_pierce_point(
                        ant_lat, ant_lon, az, el
                    )

                    # Interpolate TEC
                    tec_values[ant, src, t] = interpolator([pp_lat, pp_lon])[0]

        return tec_values

    def _compute_pierce_point(
        self,
        lat: float,
        lon: float,
        az: float,
        el: float
    ) -> Tuple[float, float]:
        """
        Compute ionospheric pierce point location.

        Parameters
        ----------
        lat : float
            Observer latitude in radians.
        lon : float
            Observer longitude in radians.
        az : float
            Azimuth in radians.
        el : float
            Elevation in radians.

        Returns
        -------
        pp_lat : float
            Pierce point latitude in radians.
        pp_lon : float
            Pierce point longitude in radians.
        """
        # Earth radius
        R_earth = 6.371e6

        # Angular distance to pierce point
        psi = np.pi / 2 - el - np.arcsin(
            R_earth / (R_earth + self.pierce_point_height) * np.cos(el)
        )

        # Pierce point position
        pp_lat = np.arcsin(
            np.sin(lat) * np.cos(psi) +
            np.cos(lat) * np.sin(psi) * np.cos(az)
        )

        pp_lon = lon + np.arcsin(
            np.sin(psi) * np.sin(az) / np.cos(pp_lat)
        )

        return pp_lat, pp_lon

    def get_slant_factor(self, elevation: float) -> float:
        """
        Get mapping function for slant TEC.

        TEC_slant = TEC_vertical * M(el)

        Parameters
        ----------
        elevation : float
            Source elevation in radians.

        Returns
        -------
        m : float
            Mapping function value.
        """
        # Simple mapping function (thin shell model)
        R_earth = 6.371e6
        r_ratio = R_earth / (R_earth + self.pierce_point_height)

        cos_zenith_pp = np.sqrt(1 - (r_ratio * np.cos(elevation))**2)

        return 1.0 / cos_zenith_pp
