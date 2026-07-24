"""Beam sky projection — transform beam power from Az/ZA to RA/Dec.

Projects an antenna beam power pattern onto equatorial (RA/Dec) coordinates
for visualization. Given a beam power function in native Az/ZA coordinates
and a zenith position (RA, Dec), this module computes the beam power on a
regular RA/Dec grid using spherical trigonometry.

This is a general-purpose utility — not tied to any specific telescope.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

logger = logging.getLogger(__name__)


class _RGBAOverlay(TypedDict):
    image: np.ndarray
    x: float
    y: float
    dw: float
    dh: float
    ra_center: float
    dec_center: float


def _owned_read_only_array(
    value: np.ndarray,
    *,
    field_name: str,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    """Return a detached C-contiguous, non-writeable ndarray."""
    if type(value) is not np.ndarray:
        raise TypeError(f"{field_name} must be an exact ndarray")
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class BeamSkyProjection:
    """Result of projecting a beam pattern onto RA/Dec coordinates.

    Attributes
    ----------
    ra_grid_deg : ndarray
        1-D RA axis in degrees.
    dec_grid_deg : ndarray
        1-D Dec axis in degrees.
    power_db : ndarray
        2-D beam power in dB, shape ``(len(dec_grid_deg), len(ra_grid_deg))``.
        NaN where the beam is outside ``max_za_deg``.
    zenith_ra_deg : float
        RA of the zenith (beam centre) in degrees.
    zenith_dec_deg : float
        Dec of the zenith (beam centre) in degrees.
    max_za_deg : float
        Maximum zenith angle included in the projection.
    """

    ra_grid_deg: np.ndarray
    dec_grid_deg: np.ndarray
    power_db: np.ndarray
    zenith_ra_deg: float
    zenith_dec_deg: float
    max_za_deg: float

    def __post_init__(self) -> None:
        ra = _owned_read_only_array(
            self.ra_grid_deg,
            field_name="ra_grid_deg",
            dtype=np.float64,
        )
        dec = _owned_read_only_array(
            self.dec_grid_deg,
            field_name="dec_grid_deg",
            dtype=np.float64,
        )
        power = _owned_read_only_array(
            self.power_db,
            field_name="power_db",
            dtype=np.float64,
        )
        if ra.ndim != 1 or dec.ndim != 1:
            raise ValueError("beam projection axes must be one-dimensional")
        if (
            len(ra) == 0
            or len(dec) == 0
            or not np.all(np.isfinite(ra))
            or not np.all(np.isfinite(dec))
        ):
            raise ValueError("beam projection axes must be nonempty and finite")
        if len(ra) > 1 and np.any(np.diff(ra) <= 0.0):
            raise ValueError("ra_grid_deg must be strictly increasing")
        if len(dec) > 1 and np.any(np.diff(dec) <= 0.0):
            raise ValueError("dec_grid_deg must be strictly increasing")
        if power.shape != (len(dec), len(ra)):
            raise ValueError(
                "beam projection power_db shape must match (dec_grid, ra_grid)"
            )
        if np.any(np.isinf(power)):
            raise ValueError("beam projection power_db must not contain infinities")
        for name in ("zenith_ra_deg", "zenith_dec_deg", "max_za_deg"):
            value = getattr(self, name)
            if type(value) is not float or not np.isfinite(value):
                raise TypeError(f"{name} must be an exact finite float")
        if self.max_za_deg <= 0.0 or self.max_za_deg > 90.0:
            raise ValueError("max_za_deg must be in (0, 90]")
        object.__setattr__(self, "ra_grid_deg", ra)
        object.__setattr__(self, "dec_grid_deg", dec)
        object.__setattr__(self, "power_db", power)

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, eq=False)
class BeamContour:
    """One immutable beam-contour level and its detached path segments."""

    level_db: float
    segments: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if type(self.level_db) is not float or not np.isfinite(self.level_db):
            raise TypeError("level_db must be an exact finite float")
        if type(self.segments) is not tuple:
            raise TypeError("segments must be an exact tuple")
        copied: list[np.ndarray] = []
        for index, segment in enumerate(self.segments):
            if type(segment) is not np.ndarray:
                raise TypeError(f"segments[{index}] must be an exact ndarray")
            owned = _owned_read_only_array(
                segment,
                field_name=f"segments[{index}]",
                dtype=np.float64,
            )
            if owned.ndim != 2 or owned.shape[1] != 2:
                raise ValueError(f"segments[{index}] must have shape (N, 2)")
            if not np.all(np.isfinite(owned)):
                raise ValueError(f"segments[{index}] must be finite")
            copied.append(owned)
        object.__setattr__(self, "segments", tuple(copied))

    __hash__ = None  # type: ignore[assignment]


def compute_beam_power_on_radec_grid(
    beam_power_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    zenith_ra_deg: float,
    zenith_dec_deg: float,
    max_za_deg: float = 90.0,
    ra_resolution_deg: float = 0.5,
    dec_resolution_deg: float = 0.5,
) -> BeamSkyProjection:
    """Project a beam power pattern onto an RA/Dec grid.

    Parameters
    ----------
    beam_power_func : callable
        ``(za_rad, az_rad) -> power`` where *za_rad* and *az_rad* are
        2-D arrays (same shape as the RA/Dec meshgrid) and the return is
        a normalised linear power pattern (peak = 1) of the same shape.
        Azimuth convention: North = 0, East = pi/2.
    zenith_ra_deg : float
        Right ascension of the zenith in degrees.
    zenith_dec_deg : float
        Declination of the zenith in degrees.
    max_za_deg : float
        Maximum zenith angle to include (degrees, default 90).
    ra_resolution_deg : float
        RA grid spacing in degrees.
    dec_resolution_deg : float
        Dec grid spacing in degrees.

    Returns
    -------
    BeamSkyProjection
    """
    # RA extent corrected for cos(dec) foreshortening
    cos_dec = np.cos(np.deg2rad(zenith_dec_deg))
    ra_extent = max_za_deg / max(cos_dec, 0.01)  # guard near poles

    ra_min = zenith_ra_deg - ra_extent
    ra_max = zenith_ra_deg + ra_extent
    dec_min = max(zenith_dec_deg - max_za_deg, -90.0)
    dec_max = min(zenith_dec_deg + max_za_deg, 90.0)

    ra_grid = np.arange(ra_min, ra_max + ra_resolution_deg, ra_resolution_deg)
    dec_grid = np.arange(dec_min, dec_max + dec_resolution_deg, dec_resolution_deg)

    RA, DEC = np.meshgrid(ra_grid, dec_grid)

    # --- RA/Dec  →  ZA/Az via spherical trigonometry ---
    dec_z_rad = np.deg2rad(zenith_dec_deg)
    dec_rad = np.deg2rad(DEC)
    delta_ra_rad = np.deg2rad(RA - zenith_ra_deg)

    # Zenith angle
    cos_za = np.sin(dec_z_rad) * np.sin(dec_rad) + np.cos(dec_z_rad) * np.cos(
        dec_rad
    ) * np.cos(delta_ra_rad)
    cos_za = np.clip(cos_za, -1.0, 1.0)
    za_rad = np.arccos(cos_za)

    # Azimuth (N = 0, E = pi/2)
    sin_za = np.sin(za_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_az = np.cos(dec_rad) * np.sin(delta_ra_rad) / sin_za
        cos_az = (np.sin(dec_rad) - np.sin(dec_z_rad) * cos_za) / (
            np.cos(dec_z_rad) * sin_za
        )

    sin_az = np.clip(sin_az, -1.0, 1.0)
    cos_az = np.clip(cos_az, -1.0, 1.0)

    az_rad = np.arctan2(sin_az, cos_az) % (2.0 * np.pi)

    # Handle exact zenith (ZA ≈ 0)
    zenith_mask = za_rad < 1e-6
    az_rad[zenith_mask] = 0.0

    # --- Evaluate beam power ---
    power = beam_power_func(za_rad, az_rad)

    # Mask beyond max_za
    power[za_rad > np.deg2rad(max_za_deg)] = np.nan

    # Convert to dB (with floor to avoid -inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        power_db = 10.0 * np.log10(np.where(np.isnan(power), np.nan, power + 1e-30))
    power_db[np.isnan(power)] = np.nan

    return BeamSkyProjection(
        ra_grid_deg=np.asarray(ra_grid),
        dec_grid_deg=np.asarray(dec_grid),
        power_db=np.asarray(power_db),
        zenith_ra_deg=float(zenith_ra_deg),
        zenith_dec_deg=float(zenith_dec_deg),
        max_za_deg=float(max_za_deg),
    )


def create_rgba_overlay(
    projection: BeamSkyProjection,
    cmap: str = "RdBu_r",
    vmin_db: float = -40.0,
    vmax_db: float = 0.0,
    alpha_scale: float = 0.7,
) -> _RGBAOverlay:
    """Create a uint32 RGBA image for ``Bokeh.image_rgba()``.

    Parameters
    ----------
    projection : BeamSkyProjection
        Output of :func:`compute_beam_power_on_radec_grid`.
    cmap : str
        Matplotlib colormap name.
    vmin_db, vmax_db : float
        Colour-scale limits in dB.
    alpha_scale : float
        Maximum alpha (0–1) for the overlay.

    Returns
    -------
    dict
        Keys ``image`` (uint32 2-D), ``x``, ``y``, ``dw``, ``dh``,
        ``ra_center``, ``dec_center`` — ready for Bokeh.
    """
    import matplotlib.pyplot as plt

    power_db = projection.power_db
    colormap = plt.get_cmap(cmap)

    # Normalise to [0, 1] for the colormap
    normed = np.clip((power_db - vmin_db) / (vmax_db - vmin_db), 0.0, 1.0)
    rgba = colormap(normed)  # (H, W, 4) float in [0, 1]

    # Alpha proportional to normalised power; NaN → fully transparent
    alpha = normed * alpha_scale
    alpha[np.isnan(power_db)] = 0.0
    rgba[:, :, 3] = alpha

    # Pack into uint32:  R | G<<8 | B<<16 | A<<24
    rgba_u8 = (rgba * 255).astype(np.uint8)
    img = np.empty(rgba_u8.shape[:2], dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape(*rgba_u8.shape[:2], 4)
    view[:, :, 0] = rgba_u8[:, :, 0]
    view[:, :, 1] = rgba_u8[:, :, 1]
    view[:, :, 2] = rgba_u8[:, :, 2]
    view[:, :, 3] = rgba_u8[:, :, 3]

    ra = projection.ra_grid_deg
    dec = projection.dec_grid_deg

    return {
        "image": img,
        "x": float(ra[0]),
        "y": float(dec[0]),
        "dw": float(ra[-1] - ra[0]),
        "dh": float(dec[-1] - dec[0]),
        "ra_center": projection.zenith_ra_deg,
        "dec_center": projection.zenith_dec_deg,
    }


def extract_contours(
    projection: BeamSkyProjection,
    levels_db: list[float] | None = None,
) -> tuple[BeamContour, ...]:
    """Extract contour paths from a beam sky projection.

    Parameters
    ----------
    projection : BeamSkyProjection
        Output of :func:`compute_beam_power_on_radec_grid`.
    levels_db : list of float, optional
        dB levels at which to extract contours.  Default ``[-3, -10]``.

    Returns
    -------
    tuple of BeamContour
        One immutable model per requested level.
    """
    import contourpy

    if levels_db is None:
        levels_db = [-3.0, -10.0]

    results: list[BeamContour] = []
    generator = contourpy.contour_generator(
        x=projection.ra_grid_deg,
        y=projection.dec_grid_deg,
        z=projection.power_db,
        name="serial",
        line_type="Separate",
    )
    for level in levels_db:
        segments = [
            np.array(path, dtype=np.float64, copy=True, order="C")
            for path in generator.lines(float(level))
            if len(path) > 1
        ]
        results.append(
            BeamContour(
                level_db=float(level),
                segments=tuple(segments),
            )
        )
    return tuple(results)


__all__ = [
    "BeamSkyProjection",
    "BeamContour",
    "compute_beam_power_on_radec_grid",
    "create_rgba_overlay",
    "extract_contours",
]
