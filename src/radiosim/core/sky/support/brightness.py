"""Pure brightness-temperature and Stokes unit conversions for loaders.

Keeps flux-density ↔ brightness-temperature physics out of I/O, TAP, and
reprojection code in :mod:`radiosim.core.sky.loaders`.
"""

from __future__ import annotations

import numpy as np

from ..containers.constants import (
    flux_density_to_brightness_temp,
    rayleigh_jeans_factor,
)


def healpix_flux_row_to_brightness_temp(
    hp_map: np.ndarray,
    *,
    freq_hz: float,
    omega_pixel: float,
    is_stokes_i: bool,
    is_flux_unit: bool,
    brightness_conversion: str,
) -> np.ndarray:
    """Convert one HEALPix-projected FITS Stokes row to brightness temperature."""
    if not is_flux_unit:
        return hp_map

    flux_map = hp_map * omega_pixel
    if is_stokes_i:
        pos = flux_map > 0
        temp_map = np.zeros_like(hp_map)
        if np.any(pos):
            temp_map[pos] = flux_density_to_brightness_temp(
                flux_map[pos],
                freq_hz,
                omega_pixel,
                method=brightness_conversion,
            )
        return temp_map

    return flux_map / rayleigh_jeans_factor(freq_hz, omega_pixel)


def skyh5_stokes_slice_to_kelvin(
    stokes_slice: np.ndarray,
    *,
    unit: str,
    freq_hz: float,
) -> np.ndarray:
    """Convert a skyh5 Stokes slice from Jy/sr to K_RJ when required."""
    if unit != "Jy / sr":
        return stokes_slice
    return flux_density_to_brightness_temp(
        stokes_slice,
        freq_hz,
        1.0,
        method="rayleigh-jeans",
    )
