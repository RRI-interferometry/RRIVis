"""Aperture shape far-field pattern functions for radio astronomy beam modeling.

This module provides voltage (E-field) pattern functions for common aperture
geometries used in radio interferometry. All pattern functions take a
normalized angle ``u_beam = D * sin(theta) / lambda`` as input and return
the voltage response normalized to 1.0 at ``u_beam = 0``.

Supported aperture shapes:

- **Circular** (Airy pattern): ``E(u) = 2 * J1(pi*u) / (pi*u)``
- **Rectangular** (sinc pattern): ``E(ux, uy) = sinc(ux) * sinc(uy)``
- **Elliptical** (direction-dependent Airy): Airy with effective diameter
  that varies with azimuth angle phi.

References
----------
.. [1] Balanis, C.A. "Antenna Theory: Analysis and Design" (4th ed., 2016)
       Chapter 12: Aperture Antennas
.. [2] Born, M. & Wolf, E. "Principles of Optics" (7th ed., 1999)
       Chapter 8: Elements of the Theory of Diffraction
"""

from collections.abc import Callable

import numpy as np
from scipy.special import j1

# Speed of light in vacuum (m/s)
_C = 299_792_458.0


def compute_u_beam(
    theta: np.ndarray,
    diameter: float,
    freq_hz: float,
) -> np.ndarray:
    """Compute the normalized beam angle u_beam = D * sin(theta) * freq / c.

    This dimensionless quantity is the natural coordinate for aperture
    diffraction patterns, equal to ``D * sin(theta) / lambda``.

    Parameters
    ----------
    theta : np.ndarray
        Angle from boresight in radians.
    diameter : float
        Aperture diameter in meters.
    freq_hz : float
        Observing frequency in Hz.

    Returns
    -------
    np.ndarray
        Normalized beam angle, same shape as ``theta``.
    """
    return diameter * np.sin(theta) * freq_hz / _C


def airy_voltage_pattern(u_beam: np.ndarray) -> np.ndarray:
    """Voltage pattern for a uniformly illuminated circular aperture (Airy).

    Computes ``E(u) = 2 * J1(pi * u) / (pi * u)`` where ``u_beam`` is the
    normalized angle ``D * sin(theta) / lambda``. At ``u_beam = 0`` the
    pattern evaluates to 1.0 via L'Hopital's rule.

    Parameters
    ----------
    u_beam : np.ndarray
        Normalized beam angle ``D * sin(theta) / lambda``.

    Returns
    -------
    np.ndarray
        Voltage pattern values, same shape as ``u_beam``.
    """
    u_beam = np.asarray(u_beam, dtype=np.float64)
    arg = np.pi * u_beam
    # 2 * J1(pi*u) / (pi*u), with E(0) = 1.0 by L'Hopital
    return np.where(
        arg == 0.0,
        1.0,
        2.0 * j1(arg) / arg,
    )


def sinc_voltage_pattern(
    ux_beam: np.ndarray,
    uy_beam: np.ndarray,
) -> np.ndarray:
    """Voltage pattern for a uniformly illuminated rectangular aperture.

    Computes ``E(ux, uy) = sinc(ux) * sinc(uy)`` using NumPy's ``sinc``
    which evaluates ``sin(pi*x) / (pi*x)``.

    Parameters
    ----------
    ux_beam : np.ndarray
        Normalized beam angle along x-axis, ``Dx * sin(theta_x) / lambda``.
    uy_beam : np.ndarray
        Normalized beam angle along y-axis, ``Dy * sin(theta_y) / lambda``.

    Returns
    -------
    np.ndarray
        Voltage pattern values, broadcastable shape of ``ux_beam`` and ``uy_beam``.
    """
    return np.sinc(ux_beam) * np.sinc(uy_beam)


def elliptical_airy_voltage_pattern(
    theta: np.ndarray,
    phi: np.ndarray,
    diameter_x: float,
    diameter_y: float,
    wavelength: float,
) -> np.ndarray:
    """Voltage pattern for an elliptical aperture (direction-dependent Airy).

    The effective diameter varies with azimuth angle phi:

        D_eff(phi) = 1 / sqrt((cos(phi)/Dx)**2 + (sin(phi)/Dy)**2)

    The pattern is then computed as ``airy_voltage_pattern(D_eff * sin(theta) / lambda)``.

    When ``diameter_x == diameter_y``, this reduces to the standard circular
    Airy pattern.

    Parameters
    ----------
    theta : np.ndarray
        Angle from boresight in radians.
    phi : np.ndarray
        Azimuth angle in radians, same shape as ``theta``.
    diameter_x : float
        Aperture diameter along x-axis in meters.
    diameter_y : float
        Aperture diameter along y-axis in meters.
    wavelength : float
        Observing wavelength in meters.

    Returns
    -------
    np.ndarray
        Voltage pattern values, same shape as ``theta``.
    """
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    d_eff = 1.0 / np.sqrt((cos_phi / diameter_x) ** 2 + (sin_phi / diameter_y) ** 2)
    u_beam = d_eff * np.sin(theta) / wavelength
    return airy_voltage_pattern(u_beam)


APERTURE_SHAPES: dict[str, Callable] = {
    "circular": airy_voltage_pattern,
    "rectangular": sinc_voltage_pattern,
    "elliptical": elliptical_airy_voltage_pattern,
}
"""Registry mapping aperture shape names to their voltage pattern functions.

- ``"circular"`` -> :func:`airy_voltage_pattern`
- ``"rectangular"`` -> :func:`sinc_voltage_pattern`
- ``"elliptical"`` -> :func:`elliptical_airy_voltage_pattern`
"""


__all__ = [
    "compute_u_beam",
    "airy_voltage_pattern",
    "sinc_voltage_pattern",
    "elliptical_airy_voltage_pattern",
    "APERTURE_SHAPES",
]
