"""Feed pattern models for radio astronomy beam modeling.

This module provides voltage-level feed illumination patterns and reflector
geometry functions that convert between aperture position and feed angle.
The feed-to-illumination conversion accounts for the reflector geometry
(prime-focus or Cassegrain).

Supported feed models:

- **Corrugated horn**: ``E(theta) = cos^q(theta)``
- **Open waveguide**: Separate E-plane and H-plane patterns
- **Dipole over ground plane**: ``E(theta) = cos(theta) * sin(2*pi*h*cos(theta))``

Supported reflector geometries:

- **Prime-focus**: ``theta_feed = 2 * arctan(rho / (2*F))``
- **Cassegrain**: ``theta_feed = 2 * arctan(rho / (2*M*F))``

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

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.special import j0

# ---------------------------------------------------------------------------
# Feed pattern functions
# ---------------------------------------------------------------------------


def corrugated_horn_pattern(
    theta_feed: np.ndarray,
    q: float = 1.15,
) -> np.ndarray:
    """Corrugated horn feed pattern: ``E(theta) = cos^q(theta)``.

    A good approximation for corrugated (scalar) horns used on most
    modern radio telescopes. The parameter *q* controls the pattern
    taper; typical values range from 1.0 to 1.3.

    Parameters
    ----------
    theta_feed : np.ndarray
        Feed angle in radians measured from the feed axis.
    q : float, optional
        Cosine exponent controlling pattern rolloff (default 1.15).

    Returns
    -------
    np.ndarray
        Voltage pattern values, same shape as ``theta_feed``.
    """
    theta_feed = np.asarray(theta_feed, dtype=np.float64)
    return np.cos(theta_feed) ** q


def open_waveguide_pattern(
    theta_feed: np.ndarray,
    b_over_lambda: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Open-ended rectangular waveguide feed pattern.

    Returns separate E-plane and H-plane voltage patterns.

    - E-plane: ``cos(theta)``
    - H-plane: ``cos(pi * b * sin(theta) / lambda) / (1 - (2*b*sin(theta)/lambda)^2)``

    The H-plane expression has a removable singularity at
    ``2*b*sin(theta)/lambda = +/-1`` where the limit is ``pi/4``.

    Parameters
    ----------
    theta_feed : np.ndarray
        Feed angle in radians measured from the feed axis.
    b_over_lambda : float, optional
        Waveguide broad-wall dimension in wavelengths (default 0.7).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(E_plane, H_plane)`` voltage patterns, each same shape as
        ``theta_feed``.
    """
    theta_feed = np.asarray(theta_feed, dtype=np.float64)
    e_plane = np.cos(theta_feed)

    sin_theta = np.sin(theta_feed)
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


def dipole_ground_plane_pattern(
    theta_feed: np.ndarray,
    height_wavelengths: float = 0.25,
) -> np.ndarray:
    """Dipole over ground-plane feed pattern.

    Computes ``E(theta) = cos(theta) * sin(2*pi*h*cos(theta))``, where
    *h* is the dipole height above the ground plane in wavelengths and
    *theta* is measured from the feed axis (boresight).

    Parameters
    ----------
    theta_feed : np.ndarray
        Feed angle in radians measured from the feed axis.
    height_wavelengths : float, optional
        Dipole height above the ground plane in wavelengths (default 0.25).

    Returns
    -------
    np.ndarray
        Voltage pattern values, same shape as ``theta_feed``.
    """
    theta_feed = np.asarray(theta_feed, dtype=np.float64)
    return np.cos(theta_feed) * np.sin(
        2.0 * np.pi * height_wavelengths * np.cos(theta_feed)
    )


# ---------------------------------------------------------------------------
# Reflector geometry functions
# ---------------------------------------------------------------------------


def prime_focus_angle(
    rho: np.ndarray,
    focal_length: float,
) -> np.ndarray:
    """Feed angle for a prime-focus reflector.

    Converts aperture radial position *rho* to the corresponding feed
    angle via ``theta_feed = 2 * arctan(rho / (2*F))``.

    Parameters
    ----------
    rho : np.ndarray
        Radial distance from aperture centre in metres.
    focal_length : float
        Focal length of the reflector in metres.

    Returns
    -------
    np.ndarray
        Feed angle in radians, same shape as ``rho``.
    """
    rho = np.asarray(rho, dtype=np.float64)
    return 2.0 * np.arctan(rho / (2.0 * focal_length))


def cassegrain_angle(
    rho: np.ndarray,
    focal_length: float,
    magnification: float = 1.0,
) -> np.ndarray:
    """Feed angle for a Cassegrain reflector.

    The effective focal length is ``F_eq = M * F``, so
    ``theta_feed = 2 * arctan(rho / (2 * M * F))``.

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
        Feed angle in radians, same shape as ``rho``.
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
    """Feed angle at the edge of the dish.

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
        Edge feed angle in radians.

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


# ---------------------------------------------------------------------------
# Feed-to-illumination bridge
# ---------------------------------------------------------------------------


def compute_edge_taper_from_feed(
    feed_model: str,
    dish_diameter: float,
    focal_length: float,
    feed_params: dict[str, Any] | None = None,
    reflector_type: str = "prime_focus",
    magnification: float = 1.0,
) -> float:
    """Compute the effective edge taper (in dB) from a feed pattern.

    Evaluates the feed voltage pattern at the dish edge angle and
    converts to decibels: ``taper_dB = -20 * log10(E(theta_edge))``.

    Parameters
    ----------
    feed_model : str
        Name of the feed model (key in :data:`FEED_MODELS`).
    dish_diameter : float
        Dish diameter in metres.
    focal_length : float
        Focal length in metres.
    feed_params : dict[str, Any] or None, optional
        Extra keyword arguments forwarded to the feed pattern function.
    reflector_type : str, optional
        ``"prime_focus"`` or ``"cassegrain"`` (default ``"prime_focus"``).
    magnification : float, optional
        Cassegrain magnification (default 1.0).

    Returns
    -------
    float
        Edge taper in dB (positive value means the edge is weaker).

    Raises
    ------
    KeyError
        If *feed_model* is not found in :data:`FEED_MODELS`.
    """
    if feed_model not in FEED_MODELS:
        raise KeyError(
            f"Unknown feed model '{feed_model}'. Available: {list(FEED_MODELS.keys())}"
        )

    theta_edge = compute_edge_angle(
        dish_diameter, focal_length, reflector_type, magnification
    )
    params = dict(feed_params) if feed_params is not None else {}
    # Remove keys not accepted by pattern functions
    params.pop("focal_ratio", None)
    pattern_func = FEED_MODELS[feed_model]
    result = pattern_func(np.array([theta_edge]), **params)

    # For open_waveguide, result is a tuple; use the geometric mean
    if isinstance(result, tuple):
        e_plane, h_plane = result
        e_edge = float(np.sqrt(np.abs(e_plane[0]) * np.abs(h_plane[0])))
    else:
        e_edge = float(np.abs(result[0]))

    if e_edge <= 0.0:
        return float("inf")
    return float(-20.0 * np.log10(e_edge))


def feed_to_taper(
    rho_over_a: np.ndarray,
    focal_length: float,
    aperture_radius: float,
    feed_model: str,
    feed_params: dict[str, Any] | None = None,
    reflector_type: str = "prime_focus",
    magnification: float = 1.0,
) -> np.ndarray:
    """Convert aperture position to illumination weight via feed pattern.

    Maps normalised radial position ``rho/a`` through the reflector
    geometry to a feed angle, evaluates the feed pattern, and returns
    the illumination weight.

    Parameters
    ----------
    rho_over_a : np.ndarray
        Normalised radial position in the aperture (0 at centre, 1 at edge).
    focal_length : float
        Reflector focal length in metres.
    aperture_radius : float
        Physical aperture radius in metres.
    feed_model : str
        Name of the feed model (key in :data:`FEED_MODELS`).
    feed_params : dict[str, Any] or None, optional
        Extra keyword arguments for the feed pattern function.
    reflector_type : str, optional
        ``"prime_focus"`` or ``"cassegrain"`` (default ``"prime_focus"``).
    magnification : float, optional
        Cassegrain magnification (default 1.0).

    Returns
    -------
    np.ndarray
        Illumination weights, same shape as ``rho_over_a``.

    Raises
    ------
    KeyError
        If *feed_model* is not found in :data:`FEED_MODELS`.
    """
    if feed_model not in FEED_MODELS:
        raise KeyError(
            f"Unknown feed model '{feed_model}'. Available: {list(FEED_MODELS.keys())}"
        )

    rho_over_a = np.asarray(rho_over_a, dtype=np.float64)
    rho = rho_over_a * aperture_radius

    angle_func = REFLECTOR_TYPES[reflector_type]
    if reflector_type == "cassegrain":
        theta_feed = angle_func(rho, focal_length, magnification)
    else:
        theta_feed = angle_func(rho, focal_length)

    params = dict(feed_params) if feed_params is not None else {}
    params.pop("focal_ratio", None)
    pattern_func = FEED_MODELS[feed_model]
    result = pattern_func(theta_feed, **params)

    # For open_waveguide, return geometric mean of E- and H-plane
    if isinstance(result, tuple):
        e_plane, h_plane = result
        return np.sqrt(np.abs(e_plane) * np.abs(h_plane))

    return np.asarray(result, dtype=np.float64)


def feed_to_farfield_numerical(
    feed_model: str,
    feed_params: dict[str, Any] | None = None,
    focal_length: float = 5.0,
    aperture_radius: float = 7.0,
    u_beam: np.ndarray | None = None,
    n_radial: int = 256,
    reflector_type: str = "prime_focus",
    magnification: float = 1.0,
) -> np.ndarray:
    """Numerically compute far-field pattern via Hankel transform of feed illumination.

    Steps:

    1. Sample the illumination ``g(rho) = feed_pattern(angle_func(rho, F))``
       on a uniform radial grid from 0 to ``aperture_radius``.
    2. Hankel transform:
       ``E(u) = integral_0^a g(rho) * J0(2*pi*u*rho/D) * rho d(rho)``
       using the trapezoidal rule (D = 2*aperture_radius).
    3. Normalise so ``E(0) = 1.0``.

    Parameters
    ----------
    feed_model : str
        Name of the feed model (key in :data:`FEED_MODELS`).
    feed_params : dict[str, Any] or None, optional
        Extra keyword arguments for the feed pattern function.
    focal_length : float, optional
        Reflector focal length in metres (default 5.0).
    aperture_radius : float, optional
        Physical aperture radius in metres (default 7.0).
    u_beam : np.ndarray or None, optional
        Normalised beam angle ``D * sin(theta) / lambda`` at which to
        evaluate the far field.  If ``None``, a default grid of 512
        points from 0 to 5 is used.
    n_radial : int, optional
        Number of radial sample points for the integration (default 256).
    reflector_type : str, optional
        ``"prime_focus"`` or ``"cassegrain"`` (default ``"prime_focus"``).
    magnification : float, optional
        Cassegrain magnification (default 1.0).

    Returns
    -------
    np.ndarray
        Far-field voltage pattern values at the requested ``u_beam``
        positions, normalised so that ``E(0) = 1.0``.

    Raises
    ------
    KeyError
        If *feed_model* is not found in :data:`FEED_MODELS`.
    """
    if feed_model not in FEED_MODELS:
        raise KeyError(
            f"Unknown feed model '{feed_model}'. Available: {list(FEED_MODELS.keys())}"
        )

    if u_beam is None:
        u_beam = np.linspace(0.0, 5.0, 512)
    u_beam = np.asarray(u_beam, dtype=np.float64)

    diameter = 2.0 * aperture_radius
    rho = np.linspace(0.0, aperture_radius, n_radial)

    # Step 1: sample illumination
    angle_func = REFLECTOR_TYPES[reflector_type]
    if reflector_type == "cassegrain":
        theta_feed = angle_func(rho, focal_length, magnification)
    else:
        theta_feed = angle_func(rho, focal_length)

    params = dict(feed_params) if feed_params is not None else {}
    params.pop("focal_ratio", None)
    pattern_func = FEED_MODELS[feed_model]
    result = pattern_func(theta_feed, **params)

    if isinstance(result, tuple):
        e_plane, h_plane = result
        g = np.sqrt(np.abs(e_plane) * np.abs(h_plane))
    else:
        g = np.asarray(result, dtype=np.float64)

    # Step 2: Hankel transform via trapezoidal rule
    # E(u) = integral_0^a g(rho) * J0(2*pi*u*rho/D) * rho d(rho)
    e_farfield = np.empty_like(u_beam)
    for i, u in enumerate(u_beam):
        integrand = g * j0(2.0 * np.pi * u * rho / diameter) * rho
        _trapz = getattr(np, "trapezoid", np.trapz)
        e_farfield[i] = _trapz(integrand, rho)

    # Step 3: normalise E(0) = 1.0
    e0 = e_farfield[0] if np.abs(e_farfield[0]) > 0.0 else 1.0
    e_farfield /= e0

    return e_farfield


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

FEED_MODELS: dict[str, Callable] = {
    "corrugated_horn": corrugated_horn_pattern,
    "open_waveguide": open_waveguide_pattern,
    "dipole_ground_plane": dipole_ground_plane_pattern,
}
"""Registry mapping feed model names to their pattern functions.

- ``"corrugated_horn"`` -> :func:`corrugated_horn_pattern`
- ``"open_waveguide"`` -> :func:`open_waveguide_pattern`
- ``"dipole_ground_plane"`` -> :func:`dipole_ground_plane_pattern`
"""

REFLECTOR_TYPES: dict[str, Callable] = {
    "prime_focus": prime_focus_angle,
    "cassegrain": cassegrain_angle,
}
"""Registry mapping reflector type names to their angle functions.

- ``"prime_focus"`` -> :func:`prime_focus_angle`
- ``"cassegrain"`` -> :func:`cassegrain_angle`
"""


__all__ = [
    "corrugated_horn_pattern",
    "open_waveguide_pattern",
    "dipole_ground_plane_pattern",
    "prime_focus_angle",
    "cassegrain_angle",
    "compute_edge_angle",
    "compute_edge_taper_from_feed",
    "feed_to_taper",
    "feed_to_farfield_numerical",
    "FEED_MODELS",
    "REFLECTOR_TYPES",
]
