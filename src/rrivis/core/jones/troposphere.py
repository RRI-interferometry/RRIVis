"""
Troposphere Jones term (T) for atmospheric propagation effects.

The T-Jones term models the effects of the neutral atmosphere
(troposphere) on radio wave propagation, including:
- Path delay (dry + wet components)
- Amplitude attenuation (mainly water vapor)
- Phase fluctuations (turbulence)
- Refraction (pointing offset)

Unlike ionospheric effects, tropospheric effects are generally:
- Non-dispersive (frequency-independent delay at low frequencies)
- Direction-dependent (elevation-dependent path length)
- Time-varying (weather, turbulence)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from abc import abstractmethod

from .base import JonesTerm


# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s


class TroposphereJones(JonesTerm):
    """
    Tropospheric propagation effects Jones term.

    Models the delay and attenuation due to the neutral atmosphere.
    At frequencies below ~15 GHz, the delay is nearly non-dispersive.

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    n_sources : int
        Number of sources.
    frequencies : np.ndarray
        Observation frequencies in Hz.
    zenith_delay : np.ndarray, optional
        Zenith delay in meters, shape (n_antennas,) or (n_antennas, n_times).
    zenith_opacity : np.ndarray, optional
        Zenith opacity (dimensionless), shape same as zenith_delay.
    source_elevations : np.ndarray, optional
        Source elevations in radians, shape (n_sources,).
    times : np.ndarray, optional
        Observation times.
    """

    def __init__(
        self,
        n_antennas: int,
        n_sources: int,
        frequencies: np.ndarray,
        zenith_delay: Optional[np.ndarray] = None,
        zenith_opacity: Optional[np.ndarray] = None,
        source_elevations: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None
    ):
        self.n_antennas = n_antennas
        self.n_sources = n_sources
        self.frequencies = np.asarray(frequencies)
        self.n_freq = len(self.frequencies)

        self.times = times if times is not None else np.array([0.0])
        self.n_times = len(self.times)

        # Zenith delay (default: typical sea level value ~2.3 m)
        if zenith_delay is None:
            self.zenith_delay = 2.3 * np.ones((n_antennas, self.n_times))
        else:
            zenith_delay = np.asarray(zenith_delay)
            if zenith_delay.ndim == 1:
                self.zenith_delay = np.tile(zenith_delay[:, np.newaxis], (1, self.n_times))
            else:
                self.zenith_delay = zenith_delay

        # Zenith opacity (default: near-zero for low frequencies)
        if zenith_opacity is None:
            self.zenith_opacity = 0.01 * np.ones((n_antennas, self.n_times))
        else:
            zenith_opacity = np.asarray(zenith_opacity)
            if zenith_opacity.ndim == 1:
                self.zenith_opacity = np.tile(zenith_opacity[:, np.newaxis], (1, self.n_times))
            else:
                self.zenith_opacity = zenith_opacity

        # Source elevations
        if source_elevations is None:
            self.source_elevations = np.full(n_sources, np.pi / 4)  # 45 degrees
        else:
            self.source_elevations = np.asarray(source_elevations)

    @property
    def name(self) -> str:
        return "T"

    @property
    def is_direction_dependent(self) -> bool:
        return True  # Depends on source elevation

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
        Compute tropospheric Jones matrix.

        The troposphere introduces a scalar complex gain:
            T = A * exp(i * φ) * I

        where A is attenuation, φ is phase delay, and I is identity.

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
            2x2 complex Jones matrix (scalar * identity).
        """
        xp = backend.xp

        elevation = self.source_elevations[source_idx]
        freq = self.frequencies[freq_idx]

        # Get zenith values
        delay_zenith = self.zenith_delay[antenna_idx, time_idx]
        opacity_zenith = self.zenith_opacity[antenna_idx, time_idx]

        # Apply mapping function (airmass approximation)
        airmass = self._mapping_function(elevation)

        # Slant delay and opacity
        delay_slant = delay_zenith * airmass
        opacity_slant = opacity_zenith * airmass

        # Phase from delay (φ = 2π * ν * τ)
        phase = 2 * np.pi * freq * delay_slant / SPEED_OF_LIGHT

        # Attenuation (A = exp(-τ))
        attenuation = np.exp(-opacity_slant)

        # Construct Jones matrix (scalar * identity)
        complex_gain = attenuation * np.exp(1j * phase)
        t_matrix = complex_gain * np.eye(2, dtype=np.complex128)

        return backend.asarray(t_matrix)

    def _mapping_function(self, elevation: float) -> float:
        """
        Compute mapping function (airmass approximation).

        Uses the Niell mapping function for better accuracy at low elevations.

        Parameters
        ----------
        elevation : float
            Source elevation in radians.

        Returns
        -------
        m : float
            Mapping function value (airmass).
        """
        # Simple airmass approximation
        # m = 1 / sin(el) with correction for Earth curvature
        sin_el = np.sin(elevation)

        if sin_el < 0.01:
            # Very low elevation, limit to avoid infinity
            sin_el = 0.01

        # Marini approximation (better at low elevations)
        a = 0.00143
        b = 0.0445
        c = 0.00035
        m = (1 + a / (1 + b / (1 + c))) / (sin_el + a / (sin_el + b / (sin_el + c)))

        return m

    def set_zenith_delay(
        self,
        delay: np.ndarray,
        antenna_idx: Optional[int] = None,
        time_idx: Optional[int] = None
    ) -> None:
        """
        Set zenith delay values.

        Parameters
        ----------
        delay : np.ndarray
            Zenith delay in meters.
        antenna_idx : int, optional
            Antenna index. If None, sets for all.
        time_idx : int, optional
            Time index. If None, sets for all.
        """
        ant_slice = antenna_idx if antenna_idx is not None else slice(None)
        time_slice = time_idx if time_idx is not None else slice(None)
        self.zenith_delay[ant_slice, time_slice] = delay

    def set_zenith_opacity(
        self,
        opacity: np.ndarray,
        antenna_idx: Optional[int] = None,
        time_idx: Optional[int] = None
    ) -> None:
        """
        Set zenith opacity values.

        Parameters
        ----------
        opacity : np.ndarray
            Zenith opacity (dimensionless).
        antenna_idx : int, optional
            Antenna index.
        time_idx : int, optional
            Time index.
        """
        ant_slice = antenna_idx if antenna_idx is not None else slice(None)
        time_slice = time_idx if time_idx is not None else slice(None)
        self.zenith_opacity[ant_slice, time_slice] = opacity

    def get_path_delay(
        self,
        antenna_idx: int,
        source_idx: int,
        time_idx: int
    ) -> float:
        """
        Get total path delay in seconds.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        source_idx : int
            Source index.
        time_idx : int
            Time index.

        Returns
        -------
        delay : float
            Path delay in seconds.
        """
        elevation = self.source_elevations[source_idx]
        delay_zenith = self.zenith_delay[antenna_idx, time_idx]
        airmass = self._mapping_function(elevation)

        return delay_zenith * airmass / SPEED_OF_LIGHT

    def get_attenuation_db(
        self,
        antenna_idx: int,
        source_idx: int,
        time_idx: int
    ) -> float:
        """
        Get atmospheric attenuation in dB.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        source_idx : int
            Source index.
        time_idx : int
            Time index.

        Returns
        -------
        attenuation : float
            Attenuation in dB (positive value = loss).
        """
        elevation = self.source_elevations[source_idx]
        opacity_zenith = self.zenith_opacity[antenna_idx, time_idx]
        airmass = self._mapping_function(elevation)

        opacity_slant = opacity_zenith * airmass
        return 10 * np.log10(np.exp(opacity_slant))


class SaastamoinenTroposphereJones(TroposphereJones):
    """
    Troposphere model using Saastamoinen zenith delay model.

    Computes zenith delay from meteorological parameters (pressure,
    temperature, humidity).

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    n_sources : int
        Number of sources.
    frequencies : np.ndarray
        Observation frequencies in Hz.
    antenna_heights : np.ndarray
        Antenna heights above sea level in meters, shape (n_antennas,).
    pressure : np.ndarray, optional
        Surface pressure in hPa, shape (n_antennas,) or (n_antennas, n_times).
    temperature : np.ndarray, optional
        Surface temperature in Kelvin.
    humidity : np.ndarray, optional
        Relative humidity (0-1).
    source_elevations : np.ndarray, optional
        Source elevations in radians.
    times : np.ndarray, optional
        Observation times.
    """

    def __init__(
        self,
        n_antennas: int,
        n_sources: int,
        frequencies: np.ndarray,
        antenna_heights: np.ndarray,
        pressure: Optional[np.ndarray] = None,
        temperature: Optional[np.ndarray] = None,
        humidity: Optional[np.ndarray] = None,
        source_elevations: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None
    ):
        self.antenna_heights = np.asarray(antenna_heights)

        times = times if times is not None else np.array([0.0])
        n_times = len(times)

        # Default meteorological values
        if pressure is None:
            # Standard pressure adjusted for altitude
            p0 = 1013.25  # hPa at sea level
            pressure = p0 * np.exp(-self.antenna_heights / 8500)
        pressure = np.asarray(pressure)
        if pressure.ndim == 1:
            # Expand to (n_antennas, n_times)
            pressure = np.tile(pressure[:, np.newaxis], (1, n_times))
        elif pressure.shape[1] != n_times:
            pressure = np.tile(pressure, (1, n_times))

        if temperature is None:
            # Standard temperature adjusted for altitude
            T0 = 288.15  # K at sea level
            temperature = T0 - 0.0065 * self.antenna_heights
        temperature = np.asarray(temperature)
        if temperature.ndim == 1:
            temperature = np.tile(temperature[:, np.newaxis], (1, n_times))
        elif temperature.shape[1] != n_times:
            temperature = np.tile(temperature, (1, n_times))

        if humidity is None:
            humidity = 0.5 * np.ones((n_antennas, n_times))
        humidity = np.asarray(humidity)
        if humidity.ndim == 1:
            humidity = np.tile(humidity[:, np.newaxis], (1, n_times))
        elif humidity.shape[1] != n_times:
            humidity = np.tile(humidity, (1, n_times))

        self.pressure = pressure
        self.temperature = temperature
        self.humidity = humidity
        self.frequencies = np.asarray(frequencies)  # Store for _compute_opacity

        # Compute zenith delays using Saastamoinen model
        zenith_delay = self._compute_saastamoinen_delay()

        # Compute opacity (simplified model for low frequencies)
        zenith_opacity = self._compute_opacity()

        super().__init__(
            n_antennas, n_sources, frequencies,
            zenith_delay=zenith_delay,
            zenith_opacity=zenith_opacity,
            source_elevations=source_elevations,
            times=times
        )

    def _compute_saastamoinen_delay(self) -> np.ndarray:
        """
        Compute zenith delay using Saastamoinen model.

        The total zenith delay has dry and wet components:
            ZTD = ZHD + ZWD

        Returns
        -------
        ztd : np.ndarray
            Total zenith delay in meters.
        """
        n_antennas = len(self.antenna_heights)
        n_times = self.pressure.shape[1]
        ztd = np.zeros((n_antennas, n_times))

        for ant in range(n_antennas):
            for t in range(n_times):
                P = self.pressure[ant, t]  # hPa
                T = self.temperature[ant, t]  # K
                RH = self.humidity[ant, t]  # 0-1
                h = self.antenna_heights[ant]  # m

                # Saturation vapor pressure (Magnus formula)
                T_c = T - 273.15  # Celsius
                e_sat = 6.1078 * np.exp(17.27 * T_c / (T_c + 237.3))  # hPa
                e = RH * e_sat  # Partial water vapor pressure

                # Zenith Hydrostatic Delay (dry)
                # ZHD = 0.0022768 * P / (1 - 0.00266 * cos(2φ) - 0.00028 * h)
                # Simplified for latitude 45°
                zhd = 0.0022768 * P / (1 - 0.00028 * h / 1000)

                # Zenith Wet Delay
                # ZWD = 0.002277 * (1255/T + 0.05) * e
                zwd = 0.002277 * (1255 / T + 0.05) * e

                ztd[ant, t] = zhd + zwd

        return ztd

    def _compute_opacity(self) -> np.ndarray:
        """
        Compute zenith opacity from meteorological data.

        At low frequencies, opacity is mainly due to water vapor.

        Returns
        -------
        tau : np.ndarray
            Zenith opacity (dimensionless).
        """
        n_antennas = len(self.antenna_heights)
        n_times = self.pressure.shape[1]

        # At low frequencies (< 10 GHz), opacity is very small
        # At higher frequencies, need proper atmospheric model
        freq_ghz = self.frequencies.mean() / 1e9

        tau = np.zeros((n_antennas, n_times))

        for ant in range(n_antennas):
            for t in range(n_times):
                T = self.temperature[ant, t]
                RH = self.humidity[ant, t]

                # Simplified opacity model
                # τ ≈ 0.01 * (ν/20 GHz)² * (pwv/20 mm)
                T_c = T - 273.15
                e_sat = 6.1078 * np.exp(17.27 * T_c / (T_c + 237.3))
                pwv = 0.1 * RH * e_sat  # Precipitable water vapor (mm, approximate)

                tau[ant, t] = 0.01 * (freq_ghz / 20)**2 * (pwv / 20)

        return tau

    def update_meteorology(
        self,
        pressure: Optional[np.ndarray] = None,
        temperature: Optional[np.ndarray] = None,
        humidity: Optional[np.ndarray] = None,
        antenna_idx: Optional[int] = None,
        time_idx: Optional[int] = None
    ) -> None:
        """
        Update meteorological parameters and recompute delays.

        Parameters
        ----------
        pressure : np.ndarray, optional
            New pressure values in hPa.
        temperature : np.ndarray, optional
            New temperature values in K.
        humidity : np.ndarray, optional
            New humidity values (0-1).
        antenna_idx : int, optional
            Antenna index to update.
        time_idx : int, optional
            Time index to update.
        """
        ant_slice = antenna_idx if antenna_idx is not None else slice(None)
        time_slice = time_idx if time_idx is not None else slice(None)

        if pressure is not None:
            self.pressure[ant_slice, time_slice] = pressure
        if temperature is not None:
            self.temperature[ant_slice, time_slice] = temperature
        if humidity is not None:
            self.humidity[ant_slice, time_slice] = humidity

        # Recompute delays
        self.zenith_delay = self._compute_saastamoinen_delay()
        self.zenith_opacity = self._compute_opacity()


class TurbulentTroposphereJones(TroposphereJones):
    """
    Troposphere with Kolmogorov turbulence for phase fluctuations.

    Models phase fluctuations from tropospheric turbulence, which
    limits coherence time at high frequencies.

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    n_sources : int
        Number of sources.
    frequencies : np.ndarray
        Observation frequencies in Hz.
    source_elevations : np.ndarray
        Source elevations in radians.
    times : np.ndarray
        Observation times (used to generate time series).
    C_n2 : float
        Structure constant for refractive index (typical: 1e-14 m^(-2/3)).
    wind_speed : float
        Wind speed in m/s (affects temporal fluctuations).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_antennas: int,
        n_sources: int,
        frequencies: np.ndarray,
        source_elevations: np.ndarray,
        times: np.ndarray,
        C_n2: float = 1e-14,
        wind_speed: float = 10.0,
        seed: Optional[int] = None
    ):
        self.C_n2 = C_n2
        self.wind_speed = wind_speed

        if seed is not None:
            np.random.seed(seed)

        n_times = len(times)

        # Generate turbulent delay fluctuations
        zenith_delay = self._generate_turbulent_delay(n_antennas, n_times)

        # Small opacity fluctuations
        zenith_opacity = 0.01 * (1 + 0.1 * np.random.randn(n_antennas, n_times))
        zenith_opacity = np.clip(zenith_opacity, 0, 0.5)

        super().__init__(
            n_antennas, n_sources, frequencies,
            zenith_delay=zenith_delay,
            zenith_opacity=zenith_opacity,
            source_elevations=source_elevations,
            times=times
        )

    def _generate_turbulent_delay(
        self,
        n_antennas: int,
        n_times: int
    ) -> np.ndarray:
        """
        Generate turbulent delay fluctuations.

        Uses Kolmogorov power spectrum to generate realistic
        tropospheric phase fluctuations.

        Parameters
        ----------
        n_antennas : int
            Number of antennas.
        n_times : int
            Number of time samples.

        Returns
        -------
        delay : np.ndarray
            Zenith delay with fluctuations in meters.
        """
        # Mean zenith delay
        mean_delay = 2.3  # meters

        # RMS delay fluctuation (typical: 1-10 mm)
        delay_rms = 0.005  # 5 mm RMS

        # Generate correlated noise with Kolmogorov spectrum
        # For simplicity, use AR(1) process
        delay = np.zeros((n_antennas, n_times))

        # Correlation time from wind speed (frozen turbulence)
        # τ_corr ≈ L_outer / v_wind
        correlation_coeff = 0.99

        for ant in range(n_antennas):
            # AR(1) process
            delay[ant, 0] = mean_delay + delay_rms * np.random.randn()
            for t in range(1, n_times):
                innovation = delay_rms * np.sqrt(1 - correlation_coeff**2) * np.random.randn()
                delay[ant, t] = mean_delay + correlation_coeff * (delay[ant, t-1] - mean_delay) + innovation

        return delay

    def get_phase_rms(self, freq_idx: int) -> float:
        """
        Get RMS phase fluctuation at given frequency.

        Parameters
        ----------
        freq_idx : int
            Frequency index.

        Returns
        -------
        rms : float
            RMS phase in radians.
        """
        delay_rms = np.std(self.zenith_delay)
        freq = self.frequencies[freq_idx]

        # Phase RMS = 2π * ν * delay_rms / c
        return 2 * np.pi * freq * delay_rms / SPEED_OF_LIGHT

    def get_coherence_time(self, freq_idx: int, threshold: float = 1.0) -> float:
        """
        Estimate coherence time at given frequency.

        Coherence time is when RMS phase reaches threshold (typically 1 radian).

        Parameters
        ----------
        freq_idx : int
            Frequency index.
        threshold : float
            Phase threshold in radians (default: 1).

        Returns
        -------
        t_coh : float
            Coherence time in seconds.
        """
        phase_rms = self.get_phase_rms(freq_idx)

        if len(self.times) < 2:
            return np.inf

        # Time sampling
        dt = (self.times[1] - self.times[0]) * 86400  # MJD to seconds

        # Estimate from phase variance growth
        # Simplified: assume linear variance growth
        t_coh = threshold / phase_rms * dt

        return t_coh
