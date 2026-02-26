# rrivis/core/sky/constants.py
"""Physical constants and unit-conversion helpers for sky models."""

import warnings

import numpy as np


# =============================================================================
# Physical Constants
# =============================================================================

K_BOLTZMANN = 1.380649e-23  # Boltzmann constant (J/K)
C_LIGHT = 299792458  # Speed of light (m/s)
H_PLANCK = 6.62607015e-34  # Planck constant (J·s)


def brightness_temp_to_flux_density(
    temperature: np.ndarray,
    frequency: float,
    solid_angle: float,
    method: str = "planck",
) -> np.ndarray:
    """
    Convert brightness temperature to flux density in Jy.

    Parameters
    ----------
    temperature : np.ndarray
        Brightness temperature in Kelvin.
    frequency : float
        Frequency in Hz.
    solid_angle : float
        Solid angle in steradians.
    method : str, default="planck"
        Conversion method: "planck" (exact) or "rayleigh-jeans" (approximation).

    Returns
    -------
    np.ndarray
        Flux density in Jy.
    """
    temperature = np.asarray(temperature, dtype=np.float64)
    if method == "rayleigh-jeans":
        # S = (2 k_B T ν²/c²) × Ω / 1e-26
        return (2 * K_BOLTZMANN * temperature * frequency**2 / C_LIGHT**2) * solid_angle / 1e-26

    # Planck-exact: S = (2hν³/c²) / expm1(hν/kT) × Ω / 1e-26
    if np.any(temperature <= 0):
        bad = np.where(temperature <= 0)[0]
        raise ValueError(
            f"brightness_temp_to_flux_density: temperature must be strictly positive, "
            f"but found {len(bad)} pixel(s) with T ≤ 0 "
            f"(min value: {temperature[bad].min():.6g} K). "
            f"Filter zero/negative pixels before calling this function."
        )
    x = H_PLANCK * frequency / (K_BOLTZMANN * temperature)
    intensity = (2 * H_PLANCK * frequency**3 / C_LIGHT**2) / np.expm1(x)
    return intensity * solid_angle / 1e-26


def flux_density_to_brightness_temp(
    flux_jy: np.ndarray,
    frequency: float,
    solid_angle: float,
    method: str = "planck",
) -> np.ndarray:
    """
    Convert flux density in Jy to brightness temperature in Kelvin.

    Parameters
    ----------
    flux_jy : np.ndarray
        Flux density in Jy.
    frequency : float
        Frequency in Hz.
    solid_angle : float
        Solid angle in steradians.
    method : str, default="planck"
        Conversion method: "planck" (exact) or "rayleigh-jeans" (approximation).

    Returns
    -------
    np.ndarray
        Brightness temperature in Kelvin.
    """
    flux_jy = np.asarray(flux_jy, dtype=np.float64)
    if method == "rayleigh-jeans":
        # T = S c² / (2 k_B ν² Ω) × 1e-26
        return flux_jy * C_LIGHT**2 / (2 * K_BOLTZMANN * frequency**2 * solid_angle) * 1e-26

    # Planck-exact: T = hν / (k ln(1 + 2hν³/(c² I_ν)))
    # where I_ν = S × 1e-26 / Ω
    if np.any(flux_jy <= 0):
        bad = np.where(flux_jy <= 0)[0]
        raise ValueError(
            f"flux_density_to_brightness_temp: flux density must be strictly positive, "
            f"but found {len(bad)} pixel(s) with S ≤ 0 "
            f"(min value: {flux_jy[bad].min():.6g} Jy). "
            f"Filter zero/negative pixels before calling this function."
        )
    I_nu = flux_jy * 1e-26 / solid_angle
    ratio = 2 * H_PLANCK * frequency**3 / (C_LIGHT**2 * I_nu)
    return H_PLANCK * frequency / (K_BOLTZMANN * np.log1p(ratio))
