"""Illumination taper functions for circular aperture beam modeling.

This module provides voltage (E-field) pattern functions for circular apertures
with various illumination tapers applied. Each function takes the normalized
aperture coordinate ``u_beam = D * sin(theta) / lambda`` and returns the
far-field voltage pattern, normalized to 1.0 at ``u_beam = 0``.

The general pedestal-on-taper formula for a circular aperture is::

    E(u) = C * [2 J_1(pi u) / (pi u)]
         + (1 - C) * [2^n * n! * J_{n+1}(pi u) / (pi u)^{n+1}]

where ``C = 10^(-edge_taper_dB / 20)`` is the pedestal level at the aperture
edge, and ``n`` is the taper order (0 = uniform, 1 = parabolic, 2 = parabolic
squared).

References
----------
.. [1] Balanis, C.A. "Antenna Theory: Analysis and Design" (4th ed., 2016)
       Chapter 12: Aperture Antennas
.. [2] Stutzman, W.L. & Thiele, G.A. "Antenna Theory and Design" (3rd ed., 2012)
       Chapter 9: Aperture Antennas
.. [3] Goldsmith, P.F. "Quasioptical Systems" (1998)
       Chapter 6: Gaussian Beam Optics

Examples
--------
>>> import numpy as np
>>> from rrivis.core.jones.beam.taper import uniform_taper, parabolic_taper

Uniform illumination (Airy pattern):

>>> u = np.array([0.0, 0.5, 1.0])
>>> response = uniform_taper(u)
>>> response[0]
1.0

Parabolic taper with 10 dB edge illumination:

>>> response = parabolic_taper(u, edge_taper_dB=10.0)
>>> response[0]
1.0
"""

from collections.abc import Callable

import numpy as np
from scipy.special import j1, jv


def uniform_taper(u_beam: np.ndarray | float) -> np.ndarray:
    """Uniform illumination pattern (Airy pattern) for a circular aperture.

    This is the far-field voltage pattern of a uniformly illuminated circular
    aperture, also known as the Airy pattern:

        E(u) = 2 J_1(pi u) / (pi u)

    Parameters
    ----------
    u_beam : float or array_like
        Normalized aperture coordinate: ``D * sin(theta) / lambda``.

    Returns
    -------
    ndarray
        Voltage pattern normalized to 1.0 at ``u_beam = 0``.

    Notes
    -----
    At ``u_beam = 0`` the function returns 1.0 (the limiting value of
    ``2 J_1(x) / x`` as ``x -> 0``).
    """
    u = np.asarray(u_beam, dtype=np.float64)
    x = np.pi * u
    result = np.ones_like(x)
    mask = x != 0
    result[mask] = 2.0 * j1(x[mask]) / x[mask]
    return result


def gaussian_taper_pattern(
    u_beam: np.ndarray | float,
    edge_taper_dB: float = 10.0,
) -> np.ndarray:
    """Far-field voltage pattern for a Gaussian-tapered circular aperture.

    The aperture illumination is modeled as a pedestal (uniform) component
    plus a Gaussian taper::

        g(r) = C + (1 - C) * exp(-alpha * r^2)

    where ``C = 10^(-edge_taper_dB / 20)`` is the pedestal level and
    ``alpha = edge_taper_dB * ln(10) / 20``.

    The far-field pattern is approximated as a weighted blend of the uniform
    Airy pattern and a Gaussian envelope.

    Parameters
    ----------
    u_beam : float or array_like
        Normalized aperture coordinate: ``D * sin(theta) / lambda``.
    edge_taper_dB : float, optional
        Edge illumination taper in dB (default 10.0). Higher values produce
        wider main lobes with lower sidelobes.

    Returns
    -------
    ndarray
        Voltage pattern normalized to 1.0 at ``u_beam = 0``.
    """
    u = np.asarray(u_beam, dtype=np.float64)
    C = 10.0 ** (-edge_taper_dB / 20.0)
    alpha = edge_taper_dB * np.log(10.0) / 20.0

    # Airy (uniform) component
    x = np.pi * u
    airy = np.ones_like(x)
    mask = x != 0
    airy[mask] = 2.0 * j1(x[mask]) / x[mask]

    # Gaussian envelope component
    gauss_term = np.exp(-alpha * u**2)

    result = C * airy + (1.0 - C) * gauss_term
    # Normalize to 1 at u=0 (C + (1 - C) = 1, but keep explicit for clarity)
    norm = C + (1.0 - C)
    result /= norm
    return result


def parabolic_taper(
    u_beam: np.ndarray | float,
    edge_taper_dB: float = 10.0,
) -> np.ndarray:
    """Far-field voltage pattern for a parabolic-tapered circular aperture.

    Uses the n=1 pedestal-on-taper formula::

        E(u) = C * [2 J_1(pi u) / (pi u)]
             + (1 - C) * [8 J_2(pi u) / (pi u)^2]

    where ``C = 10^(-edge_taper_dB / 20)``.

    Parameters
    ----------
    u_beam : float or array_like
        Normalized aperture coordinate: ``D * sin(theta) / lambda``.
    edge_taper_dB : float, optional
        Edge illumination taper in dB (default 10.0).

    Returns
    -------
    ndarray
        Voltage pattern normalized to 1.0 at ``u_beam = 0``.

    Notes
    -----
    At ``u_beam = 0``, the limiting values are:

    - ``2 J_1(x) / x -> 1``
    - ``8 J_2(x) / x^2 -> 1``

    so E(0) = C + (1 - C) = 1.
    """
    u = np.asarray(u_beam, dtype=np.float64)
    C = 10.0 ** (-edge_taper_dB / 20.0)
    x = np.pi * u

    # Uniform (Airy) component: 2 J_1(x) / x
    airy = np.ones_like(x)
    mask = x != 0
    airy[mask] = 2.0 * j1(x[mask]) / x[mask]

    # Parabolic taper component: 8 J_2(x) / x^2
    para = np.ones_like(x)
    para[mask] = 8.0 * jv(2, x[mask]) / x[mask] ** 2

    result = C * airy + (1.0 - C) * para
    return result


def parabolic_squared_taper(
    u_beam: np.ndarray | float,
    edge_taper_dB: float = 10.0,
) -> np.ndarray:
    """Far-field voltage pattern for a parabolic-squared-tapered circular aperture.

    Uses the n=2 pedestal-on-taper formula::

        E(u) = C * [2 J_1(pi u) / (pi u)]
             + (1 - C) * [48 J_3(pi u) / (pi u)^3]

    where ``C = 10^(-edge_taper_dB / 20)``.

    Parameters
    ----------
    u_beam : float or array_like
        Normalized aperture coordinate: ``D * sin(theta) / lambda``.
    edge_taper_dB : float, optional
        Edge illumination taper in dB (default 10.0).

    Returns
    -------
    ndarray
        Voltage pattern normalized to 1.0 at ``u_beam = 0``.

    Notes
    -----
    At ``u_beam = 0``, the limiting values are:

    - ``2 J_1(x) / x -> 1``
    - ``48 J_3(x) / x^3 -> 1``

    so E(0) = C + (1 - C) = 1.
    """
    u = np.asarray(u_beam, dtype=np.float64)
    C = 10.0 ** (-edge_taper_dB / 20.0)
    x = np.pi * u

    # Uniform (Airy) component: 2 J_1(x) / x
    airy = np.ones_like(x)
    mask = x != 0
    airy[mask] = 2.0 * j1(x[mask]) / x[mask]

    # Parabolic-squared taper component: 48 J_3(x) / x^3
    para2 = np.ones_like(x)
    para2[mask] = 48.0 * jv(3, x[mask]) / x[mask] ** 3

    result = C * airy + (1.0 - C) * para2
    return result


def cosine_taper(u_beam: np.ndarray | float) -> np.ndarray:
    """Far-field voltage pattern for a cosine-tapered circular aperture.

    The cosine illumination taper produces the far-field pattern::

        E(u) = cos(pi u / 2) / (1 - u^2)

    Parameters
    ----------
    u_beam : float or array_like
        Normalized aperture coordinate: ``D * sin(theta) / lambda``.

    Returns
    -------
    ndarray
        Voltage pattern normalized to 1.0 at ``u_beam = 0``.

    Notes
    -----
    At ``u_beam = 0``, cos(0)/(1-0) = 1.0.

    At ``u_beam = +/-1``, both numerator and denominator vanish. By
    L'Hopital's rule, the limiting value is ``pi / 4``.
    """
    u = np.asarray(u_beam, dtype=np.float64)
    result = np.empty_like(u)

    # Handle u = 0 -> 1.0
    zero_mask = u == 0.0
    # Handle u = +/-1 singularity -> pi/4 (L'Hopital)
    one_mask = np.abs(np.abs(u) - 1.0) < 1e-12
    # Regular points
    regular_mask = ~zero_mask & ~one_mask

    result[zero_mask] = 1.0
    result[one_mask] = np.pi / 4.0
    result[regular_mask] = np.cos(np.pi * u[regular_mask] / 2.0) / (
        1.0 - u[regular_mask] ** 2
    )

    return result


TAPER_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "uniform": uniform_taper,
    "gaussian": gaussian_taper_pattern,
    "parabolic": parabolic_taper,
    "parabolic_squared": parabolic_squared_taper,
    "cosine": cosine_taper,
}
"""Registry mapping taper name strings to their voltage pattern functions."""


__all__ = [
    "uniform_taper",
    "gaussian_taper_pattern",
    "parabolic_taper",
    "parabolic_squared_taper",
    "cosine_taper",
    "TAPER_FUNCTIONS",
]
