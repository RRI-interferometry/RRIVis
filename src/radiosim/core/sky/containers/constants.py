"""Physical constants and unit-conversion helpers for sky models."""

from __future__ import annotations

from enum import Enum

import numpy as np

# =============================================================================
# Enums
# =============================================================================


class BrightnessConversion(str, Enum):
    """Brightness temperature conversion method.

    Inherits from ``str`` so that enum members can be passed directly to
    functions with ``method: str`` parameters (e.g. ``method=BrightnessConversion.PLANCK``
    works because the member IS the string ``"planck"``).
    """

    PLANCK = "planck"
    RAYLEIGH_JEANS = "rayleigh-jeans"


class SpectralType(str, Enum):
    """Active spectral model for a point-source payload.

    The representations are layered, not mutually exclusive (``spectral_index``
    is term 0 of ``spectral_coeffs``); this enum names which one *drives*
    flux evaluation, using the same precedence as the materialization path:
    a per-channel ``spectrum`` wins over a log-polynomial, which wins over a
    bare power law.

    Introspection and exclusivity
    -----------------------------
    To inspect *all* populated representations (not just the driving one) use
    :attr:`PointSourceData.populated_spectral_fields`, which returns the
    ``frozenset`` of :class:`SpectralType` members actually carried by a
    payload. Construction permits co-population (the layered design is
    intentional); callers that contractually require a single representation
    opt in to the exclusivity gate via
    :meth:`PointSourceData.assert_single_spectral_representation`, which raises
    when more than one *higher-order* representation (log-polynomial /
    per-channel) is populated. The bare power law is always allowed.
    """

    POWER_LAW = "power_law"
    LOG_POLYNOMIAL = "log_polynomial"
    PER_CHANNEL = "per_channel"


# =============================================================================
# Physical Constants
# =============================================================================

K_BOLTZMANN = 1.380649e-23  # Boltzmann constant (J/K)
C_LIGHT = 299792458  # Speed of light (m/s)
H_PLANCK = 6.62607015e-34  # Planck constant (J·s)

#: Canonical synchrotron spectral index used as the default scaling exponent
#: for flux-completeness/threshold extrapolation across frequency. A single
#: source of truth so the sharp accept/reject boundaries in the disjointness
#: checks and the realistic-foreground recipe cannot drift apart.
SYNCHROTRON_SPECTRAL_INDEX = -0.7

#: Default flux floor (Jy) for the bright point-source catalog layer of
#: ``realistic_foreground_sky``. This is a low-frequency foreground recipe
#: convention, not a measured constant; override it per array and band as
#: needed.
DEFAULT_BRIGHT_CATALOG_FLUX_MIN_JY = 2.0

#: Default ``(mean, sigma)`` for per-source spectral-index draws in the Poisson
#: sub-threshold confusion layer. The steeper mean reflects the conventional
#: faint, synchrotron-dominated low-frequency source population assumption.
DEFAULT_CONFUSION_SPECTRAL_INDEX_DIST = (-0.8, 0.2)


def pixel_solid_angle(nside: int) -> float:
    """Return the HEALPix pixel solid angle in steradians: ``4π / npix``.

    Single definition shared by every consumer instead of re-deriving
    ``4 * np.pi / hp.nside2npix(nside)`` at each call site.
    """
    import healpy as hp

    return float(4.0 * np.pi / hp.nside2npix(int(nside)))


def brightness_temp_to_flux_density(
    temperature: np.ndarray,
    frequency: float,
    solid_angle: float,
    method: str = "planck",
    dtype: np.dtype | type = np.float64,
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
    dtype : np.dtype or type, default=np.float64
        Working precision for the conversion.  The Planck formula involves
        ``expm1(hν/kT)`` which can lose precision in float32; use float64
        or higher for accurate results.

    Returns
    -------
    np.ndarray
        Flux density in Jy.
    """
    temperature = np.asarray(temperature, dtype=dtype)
    if method == "rayleigh-jeans":
        # S = (2 k_B T ν²/c²) × Ω / 1e-26
        return (
            (2 * K_BOLTZMANN * temperature * frequency**2 / C_LIGHT**2)
            * solid_angle
            / 1e-26
        )

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
    dtype: np.dtype | type = np.float64,
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
    dtype : np.dtype or type, default=np.float64
        Working precision for the conversion.

    Returns
    -------
    np.ndarray
        Brightness temperature in Kelvin.
    """
    flux_jy = np.asarray(flux_jy, dtype=dtype)
    if method == "rayleigh-jeans":
        # T = S c² / (2 k_B ν² Ω) × 1e-26
        return (
            flux_jy
            * C_LIGHT**2
            / (2 * K_BOLTZMANN * frequency**2 * solid_angle)
            * 1e-26
        )

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


def rayleigh_jeans_factor(frequency: float, solid_angle: float) -> float:
    """Rayleigh-Jeans conversion factor between brightness temperature and flux density.

    Returns ``(2 * k_B * freq^2 / c^2) * omega / 1e-26``.

    Multiply brightness temperature (K) by this factor to get flux density (Jy).
    Divide flux density (Jy) by this factor to get brightness temperature (K).

    Parameters
    ----------
    frequency : float
        Frequency in Hz.
    solid_angle : float
        Solid angle in steradians (e.g. per-pixel solid angle).

    Returns
    -------
    float
        Conversion factor in Jy/K.
    """
    return (2 * K_BOLTZMANN * frequency**2 / C_LIGHT**2) * solid_angle / 1e-26
