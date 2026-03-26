"""
Parallactic Angle Jones term (P) for feed rotation.

Stub implementation: returns identity matrix. TODO: implement properly.
"""

from typing import Any

import numpy as np

from .base import JonesTerm


class ParallacticAngleJones(JonesTerm):
    """Stub: Parallactic angle rotation Jones term. TODO: implement properly.

    Parameters
    ----------
    antenna_latitudes : np.ndarray
        Geodetic latitudes of antennas in radians.
    source_positions : np.ndarray
        Source positions with shape (n_sources, 2) as (RA, Dec) in radians.
    times : np.ndarray
        Observation times.
    mount_type : str
        Antenna mount type: 'altaz', 'equatorial', or 'xy'.
    feed_angle_offset : np.ndarray, optional
        Fixed feed angle offset in radians.
    """

    def __init__(
        self,
        antenna_latitudes: np.ndarray,
        source_positions: np.ndarray,
        times: np.ndarray,
        mount_type: str = "altaz",
        feed_angle_offset: np.ndarray | None = None,
    ):
        self.antenna_latitudes = np.asarray(antenna_latitudes)
        self.n_antennas = len(self.antenna_latitudes)

        self.source_positions = np.asarray(source_positions)
        if self.source_positions.ndim == 1:
            self.source_positions = self.source_positions.reshape(1, -1)

        self.times = np.asarray(times)
        self.mount_type = mount_type.lower()

    @property
    def name(self) -> str:
        return "P"

    @property
    def is_direction_dependent(self) -> bool:
        return True

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs,
    ) -> Any:
        """Compute parallactic angle rotation matrix (stub returns identity)."""
        xp = backend.xp
        return xp.eye(2, dtype=np.complex128)


class FieldRotationJones(ParallacticAngleJones):
    """Stub: Extended parallactic angle including field rotation effects. TODO: implement properly."""

    def __init__(self, antenna_latitudes: np.ndarray, **kwargs):
        super().__init__(antenna_latitudes, np.array([[0.0, 0.0]]), np.array([0.0]))


class VLBIFeedRotationJones(ParallacticAngleJones):
    """Stub: Feed rotation for VLBI with heterogeneous antenna networks. TODO: implement properly."""

    def __init__(
        self,
        antenna_info,
        source_positions: np.ndarray,
        times: np.ndarray,
    ):
        latitudes = np.array([a.get("latitude", 0.0) for a in antenna_info])
        super().__init__(latitudes, source_positions, times)
