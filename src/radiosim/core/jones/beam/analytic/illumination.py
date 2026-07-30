"""Aperture illumination patterns for radio astronomy beam modeling.

This module provides voltage-level aperture illumination patterns and pure
reflector-geometry functions.

These identifiers belong to the illumination vocabulary of the beam subsystem
and describe how a reflector aperture is illuminated.  They are unrelated to
the receiving-receptor model in :mod:`radiosim.core.receptor`, which owns the
``receptor``, ``feed``, ``basis``, and ``feed_rotation`` vocabulary.

Supported illumination models:

- **Corrugated horn**: ``E(theta) = cos^q(theta)``
- **Open waveguide**: Separate E-plane and H-plane patterns
- **Dipole over ground plane**: ``E(theta) = cos(theta) * sin(2*pi*h*cos(theta))``

Supported reflector geometries:

- **Prime-focus**: ``theta_illumination = 2 * arctan(rho / (2*F))``
- **Cassegrain**: ``theta_illumination = 2 * arctan(rho / (2*M*F))``

References
----------
.. [1] Balanis, C.A. "Antenna Theory: Analysis and Design" (4th ed., 2016)
       Chapter 15: Reflector Antennas
.. [2] Goldsmith, P.F. "Quasioptical Systems" (1998)
       Chapter 7: Gaussian Beam Coupling to Feeds
.. [3] Stutzman, W.L. & Thiele, G.A. "Antenna Theory and Design" (3rd ed., 2012)
       Chapter 9: Aperture Antennas
.. [4] Rahmat-Samii, Y. & Imbriale, W.A. "Handbook of Reflector Antennas and
       Feed Systems" (2013) Volume I: Theory and Design
"""

import numpy as np

# ---------------------------------------------------------------------------
# Illumination pattern functions
# ---------------------------------------------------------------------------


def corrugated_horn_illumination(
    theta_illumination: np.ndarray,
    q: float = 1.15,
) -> np.ndarray:
    """Corrugated horn illumination pattern: ``E(theta) = cos^q(theta)``.

    A good approximation for corrugated (scalar) horns used on most
    modern radio telescopes. The parameter *q* controls the pattern
    taper; typical values range from 1.0 to 1.3.

    Parameters
    ----------
    theta_illumination : np.ndarray
        Illumination angle in radians measured from the feed axis.
    q : float, optional
        Cosine exponent controlling pattern rolloff (default 1.15).

    Returns
    -------
    np.ndarray
        Voltage pattern values, same shape as ``theta_illumination``.
    """
    theta_illumination = np.asarray(theta_illumination, dtype=np.float64)
    return np.cos(theta_illumination) ** q


def open_waveguide_illumination(
    theta_illumination: np.ndarray,
    b_over_lambda: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Open-ended rectangular waveguide illumination pattern.

    Returns separate E-plane and H-plane voltage patterns.

    - E-plane: ``cos(theta)``
    - H-plane: ``cos(pi * b * sin(theta) / lambda) / (1 - (2*b*sin(theta)/lambda)^2)``

    The H-plane expression has a removable singularity at
    ``2*b*sin(theta)/lambda = +/-1`` where the limit is ``pi/4``.

    Parameters
    ----------
    theta_illumination : np.ndarray
        Illumination angle in radians measured from the feed axis.
    b_over_lambda : float, optional
        Waveguide broad-wall dimension in wavelengths (default 0.7).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(E_plane, H_plane)`` voltage patterns, each same shape as
        ``theta_illumination``.
    """
    theta_illumination = np.asarray(theta_illumination, dtype=np.float64)
    e_plane = np.cos(theta_illumination)

    sin_theta = np.sin(theta_illumination)
    x = b_over_lambda * sin_theta  # b * sin(theta) / lambda
    denom = 1.0 - (2.0 * x) ** 2

    # Evaluate cos(pi * x) / denom, handling singularity via L'Hopital
    singular = np.abs(denom) < 1e-12
    safe_denom = np.where(singular, 1.0, denom)
    h_plane = np.where(
        singular,
        np.pi / 4.0,
        np.cos(np.pi * x) / safe_denom,
    )

    return e_plane, h_plane


def dipole_ground_plane_illumination(
    theta_illumination: np.ndarray,
    height_wavelengths: float = 0.25,
) -> np.ndarray:
    """Dipole over ground-plane illumination pattern.

    Computes ``E(theta) = cos(theta) * sin(2*pi*h*cos(theta))``, where
    *h* is the dipole height above the ground plane in wavelengths and
    *theta* is measured from the feed axis (boresight).

    Parameters
    ----------
    theta_illumination : np.ndarray
        Illumination angle in radians measured from the feed axis.
    height_wavelengths : float, optional
        Dipole height above the ground plane in wavelengths (default 0.25).

    Returns
    -------
    np.ndarray
        Voltage pattern values, same shape as ``theta_illumination``.
    """
    theta_illumination = np.asarray(theta_illumination, dtype=np.float64)
    return np.cos(theta_illumination) * np.sin(
        2.0 * np.pi * height_wavelengths * np.cos(theta_illumination)
    )


# ---------------------------------------------------------------------------
# Reflector geometry functions
# ---------------------------------------------------------------------------


def prime_focus_angle(
    rho: np.ndarray,
    focal_length: float,
) -> np.ndarray:
    """Illumination angle for a prime-focus reflector.

    Converts aperture radial position *rho* to the corresponding
    illumination angle via ``theta_illumination = 2 * arctan(rho / (2*F))``.

    Parameters
    ----------
    rho : np.ndarray
        Radial distance from aperture centre in metres.
    focal_length : float
        Focal length of the reflector in metres.

    Returns
    -------
    np.ndarray
        Illumination angle in radians, same shape as ``rho``.
    """
    rho = np.asarray(rho, dtype=np.float64)
    return 2.0 * np.arctan(rho / (2.0 * focal_length))


def cassegrain_angle(
    rho: np.ndarray,
    focal_length: float,
    magnification: float = 1.0,
) -> np.ndarray:
    """Illumination angle for a Cassegrain reflector.

    The effective focal length is ``F_eq = M * F``, so
    ``theta_illumination = 2 * arctan(rho / (2 * M * F))``.

    Parameters
    ----------
    rho : np.ndarray
        Radial distance from aperture centre in metres.
    focal_length : float
        Primary reflector focal length in metres.
    magnification : float, optional
        Cassegrain magnification factor (default 1.0).

    Returns
    -------
    np.ndarray
        Illumination angle in radians, same shape as ``rho``.
    """
    rho = np.asarray(rho, dtype=np.float64)
    f_eq = magnification * focal_length
    return 2.0 * np.arctan(rho / (2.0 * f_eq))


def compute_edge_angle(
    dish_diameter: float,
    focal_length: float,
    reflector_type: str = "prime_focus",
    magnification: float = 1.0,
) -> float:
    """Illumination angle at the edge of the dish.

    - Prime-focus: ``theta_edge = 2 * arctan(D / (4*F))``
    - Cassegrain: ``theta_edge = 2 * arctan(D / (4*M*F))``

    Parameters
    ----------
    dish_diameter : float
        Dish diameter in metres.
    focal_length : float
        Focal length in metres.
    reflector_type : str, optional
        ``"prime_focus"`` or ``"cassegrain"`` (default ``"prime_focus"``).
    magnification : float, optional
        Cassegrain magnification factor (default 1.0, ignored for prime-focus).

    Returns
    -------
    float
        Edge illumination angle in radians.

    Raises
    ------
    ValueError
        If *reflector_type* is not recognised.
    """
    if reflector_type == "prime_focus":
        return float(2.0 * np.arctan(dish_diameter / (4.0 * focal_length)))
    elif reflector_type == "cassegrain":
        f_eq = magnification * focal_length
        return float(2.0 * np.arctan(dish_diameter / (4.0 * f_eq)))
    else:
        raise ValueError(
            f"Unknown reflector_type '{reflector_type}'. "
            "Expected 'prime_focus' or 'cassegrain'."
        )


__all__ = [
    "corrugated_horn_illumination",
    "open_waveguide_illumination",
    "dipole_ground_plane_illumination",
    "prime_focus_angle",
    "cassegrain_angle",
    "compute_edge_angle",
]
