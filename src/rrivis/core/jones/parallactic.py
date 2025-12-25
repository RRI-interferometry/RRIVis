"""
Parallactic Angle Jones term (P) for feed rotation.

The P-Jones term accounts for the rotation of the polarization basis
as a source moves across the sky. This is essential for:
- Alt-az mounted telescopes (parallactic angle changes with hour angle)
- Comparing observations at different times
- Polarimetric calibration

The parallactic angle χ is the angle between the local vertical and
the direction to the celestial pole, measured at the source position.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from abc import abstractmethod

from .base import JonesTerm


class ParallacticAngleJones(JonesTerm):
    """
    Parallactic angle rotation Jones term.

    For an alt-az mounted antenna, the feed orientation rotates with
    respect to the sky as the source tracks. The P-matrix is a pure
    rotation:
        P = [[cos(χ),  -sin(χ)],
             [sin(χ),   cos(χ)]]

    Parameters
    ----------
    antenna_positions : np.ndarray
        Antenna positions with shape (n_antennas, 3) in ECEF or
        local ENU coordinates.
    antenna_latitudes : np.ndarray
        Geodetic latitudes of antennas in radians, shape (n_antennas,).
    source_positions : np.ndarray
        Source positions with shape (n_sources, 2) as (RA, Dec) in radians.
    times : np.ndarray
        Observation times as MJD or Unix timestamps, shape (n_times,).
    mount_type : str
        Antenna mount type: 'altaz', 'equatorial', or 'xy'.
        For equatorial mounts, P is identity.
    feed_angle_offset : np.ndarray, optional
        Fixed feed angle offset in radians, shape (n_antennas,).
    """

    def __init__(
        self,
        antenna_latitudes: np.ndarray,
        source_positions: np.ndarray,
        times: np.ndarray,
        mount_type: str = "altaz",
        feed_angle_offset: Optional[np.ndarray] = None
    ):
        self.antenna_latitudes = np.asarray(antenna_latitudes)
        self.n_antennas = len(self.antenna_latitudes)

        self.source_positions = np.asarray(source_positions)
        if self.source_positions.ndim == 1:
            self.source_positions = self.source_positions.reshape(1, -1)
        self.n_sources = self.source_positions.shape[0]

        self.times = np.asarray(times)
        self.n_times = len(self.times)

        self.mount_type = mount_type.lower()
        if self.mount_type not in ("altaz", "equatorial", "xy"):
            raise ValueError(f"Unknown mount type: {mount_type}")

        if feed_angle_offset is None:
            self.feed_angle_offset = np.zeros(self.n_antennas)
        else:
            self.feed_angle_offset = np.asarray(feed_angle_offset)

        # Precompute parallactic angles if possible
        self._parallactic_angles = None
        self._precompute_angles()

    @property
    def name(self) -> str:
        return "P"

    @property
    def is_direction_dependent(self) -> bool:
        return True  # Depends on source position

    def _precompute_angles(self) -> None:
        """Precompute parallactic angles for all combinations."""
        if self.mount_type == "equatorial":
            # No parallactic angle rotation for equatorial mount
            self._parallactic_angles = np.zeros(
                (self.n_antennas, self.n_sources, self.n_times)
            )
            return

        # For alt-az mounts, compute parallactic angles
        self._parallactic_angles = np.zeros(
            (self.n_antennas, self.n_sources, self.n_times)
        )

        for ant in range(self.n_antennas):
            lat = self.antenna_latitudes[ant]
            offset = self.feed_angle_offset[ant]

            for src in range(self.n_sources):
                ra, dec = self.source_positions[src]

                for t_idx, t in enumerate(self.times):
                    # Calculate hour angle
                    lst = self._get_lst(t, 0.0)  # Assuming longitude = 0 for now
                    ha = lst - ra

                    # Parallactic angle formula
                    chi = self._compute_parallactic_angle(ha, dec, lat)
                    self._parallactic_angles[ant, src, t_idx] = chi + offset

    def _get_lst(self, mjd: float, longitude: float) -> float:
        """
        Calculate Local Sidereal Time.

        Parameters
        ----------
        mjd : float
            Modified Julian Date.
        longitude : float
            Observer longitude in radians (east positive).

        Returns
        -------
        lst : float
            Local sidereal time in radians.
        """
        # Approximate LST calculation
        # Days since J2000.0
        d = mjd - 51544.5
        # Greenwich Mean Sidereal Time in hours
        gmst = 18.697374558 + 24.06570982441908 * d
        gmst = gmst % 24.0
        # Convert to radians and add longitude
        lst = (gmst / 24.0 * 2 * np.pi + longitude) % (2 * np.pi)
        return lst

    def _compute_parallactic_angle(
        self,
        hour_angle: float,
        declination: float,
        latitude: float
    ) -> float:
        """
        Compute parallactic angle.

        Parameters
        ----------
        hour_angle : float
            Hour angle in radians.
        declination : float
            Source declination in radians.
        latitude : float
            Observer latitude in radians.

        Returns
        -------
        chi : float
            Parallactic angle in radians.
        """
        # Parallactic angle formula
        sin_ha = np.sin(hour_angle)
        cos_ha = np.cos(hour_angle)
        sin_dec = np.sin(declination)
        cos_dec = np.cos(declination)
        sin_lat = np.sin(latitude)
        cos_lat = np.cos(latitude)

        # tan(χ) = sin(HA) / (cos(δ) * tan(φ) - sin(δ) * cos(HA))
        numerator = sin_ha * cos_lat
        denominator = cos_dec * sin_lat - sin_dec * cos_lat * cos_ha

        chi = np.arctan2(numerator, denominator)
        return chi

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
        Compute parallactic angle rotation matrix.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        source_idx : int
            Source index.
        freq_idx : int
            Frequency channel index (not used for P).
        time_idx : int
            Time index.
        backend : ArrayBackend
            Compute backend.

        Returns
        -------
        jones : array
            2x2 complex rotation matrix.
        """
        xp = backend.xp

        chi = self._parallactic_angles[antenna_idx, source_idx, time_idx]

        # Rotation matrix
        cos_chi = np.cos(chi)
        sin_chi = np.sin(chi)

        p_matrix = np.array([
            [cos_chi, -sin_chi],
            [sin_chi, cos_chi]
        ], dtype=np.complex128)

        return backend.asarray(p_matrix)

    def get_parallactic_angle(
        self,
        antenna_idx: int,
        source_idx: int,
        time_idx: int
    ) -> float:
        """
        Get parallactic angle for specific configuration.

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
        chi : float
            Parallactic angle in radians.
        """
        return self._parallactic_angles[antenna_idx, source_idx, time_idx]

    def get_rotation_rate(
        self,
        antenna_idx: int,
        source_idx: int,
        time_idx: int
    ) -> float:
        """
        Get rate of change of parallactic angle.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        source_idx : int
            Source index.
        time_idx : int
            Time index (will use adjacent points).

        Returns
        -------
        rate : float
            Rotation rate in radians/second.
        """
        if time_idx == 0:
            t0, t1 = 0, 1
        elif time_idx >= self.n_times - 1:
            t0, t1 = self.n_times - 2, self.n_times - 1
        else:
            t0, t1 = time_idx - 1, time_idx + 1

        chi0 = self._parallactic_angles[antenna_idx, source_idx, t0]
        chi1 = self._parallactic_angles[antenna_idx, source_idx, t1]
        dt = (self.times[t1] - self.times[t0]) * 86400  # MJD to seconds

        # Handle angle wrapping
        dchi = chi1 - chi0
        if dchi > np.pi:
            dchi -= 2 * np.pi
        elif dchi < -np.pi:
            dchi += 2 * np.pi

        return dchi / dt if dt > 0 else 0.0


class FieldRotationJones(ParallacticAngleJones):
    """
    Extended parallactic angle including field rotation effects.

    For wide-field imaging, the parallactic angle varies across the
    field of view. This class handles direction-dependent rotation.

    Parameters
    ----------
    antenna_latitudes : np.ndarray
        Antenna latitudes in radians.
    phase_center : np.ndarray
        Phase center (RA, Dec) in radians.
    field_offsets : np.ndarray
        Source offsets from phase center (dl, dm) in radians.
    times : np.ndarray
        Observation times.
    mount_type : str
        Mount type.
    """

    def __init__(
        self,
        antenna_latitudes: np.ndarray,
        phase_center: np.ndarray,
        field_offsets: np.ndarray,
        times: np.ndarray,
        mount_type: str = "altaz"
    ):
        self.phase_center = np.asarray(phase_center)
        self.field_offsets = np.asarray(field_offsets)

        # Convert field offsets to absolute positions
        ra0, dec0 = self.phase_center
        source_positions = np.zeros((len(field_offsets), 2))

        for i, (dl, dm) in enumerate(field_offsets):
            # Approximate offset to absolute position
            dra = dl / np.cos(dec0)
            ddec = dm
            source_positions[i] = [ra0 + dra, dec0 + ddec]

        super().__init__(
            antenna_latitudes, source_positions, times, mount_type
        )

    def get_differential_rotation(
        self,
        antenna_idx: int,
        source_idx: int,
        time_idx: int
    ) -> float:
        """
        Get rotation relative to phase center.

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
        delta_chi : float
            Differential rotation in radians.
        """
        chi_source = self.get_parallactic_angle(antenna_idx, source_idx, time_idx)
        chi_center = self.get_parallactic_angle(antenna_idx, 0, time_idx)
        return chi_source - chi_center


class VLBIFeedRotationJones(ParallacticAngleJones):
    """
    Feed rotation for VLBI with heterogeneous antenna networks.

    Different antennas may have different mount types and feed
    orientations. This class handles mixed arrays.

    Parameters
    ----------
    antenna_info : List[Dict]
        List of antenna configuration dictionaries with keys:
        - 'latitude': float, antenna latitude in radians
        - 'mount_type': str, 'altaz', 'equatorial', or 'xy'
        - 'feed_offset': float, feed angle offset in radians
        - 'name': str, optional antenna name
    source_positions : np.ndarray
        Source positions (RA, Dec) in radians.
    times : np.ndarray
        Observation times.
    """

    def __init__(
        self,
        antenna_info: List[Dict],
        source_positions: np.ndarray,
        times: np.ndarray
    ):
        self.antenna_info = antenna_info

        # Extract parameters
        latitudes = np.array([a['latitude'] for a in antenna_info])
        offsets = np.array([a.get('feed_offset', 0.0) for a in antenna_info])
        self.mount_types = [a.get('mount_type', 'altaz') for a in antenna_info]

        super().__init__(
            latitudes, source_positions, times,
            mount_type='altaz',  # Will be overridden per-antenna
            feed_angle_offset=offsets
        )

    def _precompute_angles(self) -> None:
        """Precompute parallactic angles respecting per-antenna mount types."""
        self._parallactic_angles = np.zeros(
            (self.n_antennas, self.n_sources, self.n_times)
        )

        for ant in range(self.n_antennas):
            mount = self.mount_types[ant]

            if mount == "equatorial":
                # No rotation for equatorial mount
                continue

            lat = self.antenna_latitudes[ant]
            offset = self.feed_angle_offset[ant]

            for src in range(self.n_sources):
                ra, dec = self.source_positions[src]

                for t_idx, t in enumerate(self.times):
                    lst = self._get_lst(t, 0.0)
                    ha = lst - ra

                    chi = self._compute_parallactic_angle(ha, dec, lat)
                    self._parallactic_angles[ant, src, t_idx] = chi + offset

    def get_baseline_rotation(
        self,
        ant_p: int,
        ant_q: int,
        source_idx: int,
        time_idx: int
    ) -> float:
        """
        Get differential parallactic angle for a baseline.

        For VLBI, the effective rotation on a baseline is the
        difference of parallactic angles.

        Parameters
        ----------
        ant_p : int
            First antenna index.
        ant_q : int
            Second antenna index.
        source_idx : int
            Source index.
        time_idx : int
            Time index.

        Returns
        -------
        delta_chi : float
            Differential parallactic angle in radians.
        """
        chi_p = self.get_parallactic_angle(ant_p, source_idx, time_idx)
        chi_q = self.get_parallactic_angle(ant_q, source_idx, time_idx)
        return chi_p - chi_q
