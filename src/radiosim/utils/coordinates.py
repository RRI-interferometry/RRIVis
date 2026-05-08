"""Coordinate-system helpers for RA/Dec, sidereal time, contour extraction,
and polyline wrapping.

Pure numpy (with a lazy ``matplotlib.pyplot`` import inside
``extract_contour_segments``).  No radiosim imports — safe to use from any
layer without creating cycles.
"""

from __future__ import annotations

import numpy as np

SIDEREAL_DAY_SECONDS = 86164.0905
SIDEREAL_DEG_PER_SECOND = 360.0 / SIDEREAL_DAY_SECONDS


def normalize_ra_deg(ra_deg: np.ndarray | float) -> np.ndarray | float:
    """Normalise right ascension to ``[-180, 180]`` degrees."""
    arr = np.asarray(ra_deg, dtype=float)
    norm = ((arr + 180.0) % 360.0) - 180.0
    if np.isscalar(ra_deg):
        return float(norm)
    return norm


def wrap_ra_deg(ra_deg: np.ndarray | float) -> np.ndarray | float:
    """Wrap right ascension to ``[0, 360)`` degrees."""
    arr = np.asarray(ra_deg, dtype=float) % 360.0
    if np.isscalar(ra_deg):
        return float(arr)
    return arr


def ra_deg_to_sidereal_hours(ra_deg: np.ndarray | float) -> np.ndarray | float:
    """Convert wrapped RA to sidereal hours in ``[-12, 12]``."""
    hours = normalize_ra_deg(ra_deg) / 15.0
    if np.isscalar(ra_deg):
        return float(hours)
    return hours


def axis_from_ra_deg(ra_deg: np.ndarray | float, x_axis: str) -> np.ndarray | float:
    """Map RA degrees to a plot x-axis."""
    if x_axis == "ra":
        return normalize_ra_deg(ra_deg)
    if x_axis == "lst":
        return ra_deg_to_sidereal_hours(ra_deg)
    raise ValueError(f"Unknown x_axis={x_axis!r}")


def circular_interval_width_deg(start_deg: float, end_deg: float) -> float:
    """Angular width from start to end on a wrapped circle."""
    width = wrap_ra_deg(end_deg) - wrap_ra_deg(start_deg)
    if width < 0:
        width += 360.0
    return float(width)


def unwrap_interval_end_deg(start_deg: float, end_deg: float) -> float:
    """Return an end angle unwrapped to be >= start."""
    width = circular_interval_width_deg(start_deg, end_deg)
    return float(start_deg + width)


def grid_extent(axis_values: np.ndarray) -> tuple[float, float]:
    """Return ``(start, width)`` using cell edges for a uniform 1-D grid."""
    if len(axis_values) == 1:
        return float(axis_values[0] - 0.5), 1.0
    step = float(axis_values[1] - axis_values[0])
    start = float(axis_values[0] - step / 2.0)
    width = float(step * len(axis_values))
    return start, width


def angular_separation_deg(
    ra1_deg: np.ndarray | float,
    dec1_deg: np.ndarray | float,
    ra2_deg: np.ndarray | float,
    dec2_deg: np.ndarray | float,
) -> np.ndarray:
    """Great-circle separation in degrees."""
    ra1 = np.deg2rad(ra1_deg)
    dec1 = np.deg2rad(dec1_deg)
    ra2 = np.deg2rad(ra2_deg)
    dec2 = np.deg2rad(dec2_deg)
    delta_ra = np.angle(np.exp(1j * (ra1 - ra2)))
    cos_sep = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(
        delta_ra
    )
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_sep))


def radec_to_za_az(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    zenith_ra_deg: float,
    zenith_dec_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert RA/Dec coordinates to zenith angle and azimuth."""
    dec_z = np.deg2rad(zenith_dec_deg)
    dec = np.deg2rad(dec_deg)
    # Wrap (ra - zenith_ra) into (-π, π] radians via angle(exp(i·φ)).
    # The previous implementation applied an extra np.deg2rad to the output,
    # which shrank the delta to (delta_deg · π/180)·π/180 radians and
    # collapsed the projected beam into a declination-only stripe.
    delta_ra = np.angle(np.exp(1j * np.deg2rad(ra_deg - zenith_ra_deg)))

    cos_za = np.sin(dec_z) * np.sin(dec) + np.cos(dec_z) * np.cos(dec) * np.cos(
        delta_ra
    )
    cos_za = np.clip(cos_za, -1.0, 1.0)
    za_rad = np.arccos(cos_za)

    sin_za = np.sin(za_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_az = np.cos(dec) * np.sin(delta_ra) / sin_za
        cos_az = (np.sin(dec) - np.sin(dec_z) * cos_za) / (np.cos(dec_z) * sin_za)

    sin_az = np.clip(sin_az, -1.0, 1.0)
    cos_az = np.clip(cos_az, -1.0, 1.0)
    az_rad = np.arctan2(sin_az, cos_az) % (2.0 * np.pi)
    az_rad[za_rad < 1e-6] = 0.0
    return za_rad, az_rad


def extract_contour_segments(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    levels: list[float],
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Extract contour segments from a regular 2-D grid."""
    import matplotlib.pyplot as plt

    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    fig, ax = plt.subplots()
    contours: list[tuple[np.ndarray, ...]] = []
    try:
        for level in levels:
            cs = ax.contour(x_mesh, y_mesh, values, levels=[level])
            segments: list[np.ndarray] = []
            if hasattr(cs, "get_paths"):
                for path in cs.get_paths():
                    if len(path.vertices) > 1:
                        segments.append(path.vertices.copy())
            elif hasattr(cs, "collections"):
                for collection in cs.collections:
                    for path in collection.get_paths():
                        if len(path.vertices) > 1:
                            segments.append(path.vertices.copy())
            contours.append(tuple(segments))
    finally:
        plt.close(fig)
    return tuple(contours)


def split_wrapped_path(
    x_values: np.ndarray,
    y_values: np.ndarray,
    boundary: float,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Split a polyline when it jumps across a wrapped axis."""
    if len(x_values) <= 1:
        return ((x_values, y_values),)

    jumps = np.where(np.abs(np.diff(x_values)) > boundary)[0]
    if len(jumps) == 0:
        return ((x_values, y_values),)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for jump in jumps:
        stop = jump + 1
        if stop - start > 1:
            segments.append((x_values[start:stop], y_values[start:stop]))
        start = stop
    if len(x_values) - start > 1:
        segments.append((x_values[start:], y_values[start:]))
    return tuple(segments)
