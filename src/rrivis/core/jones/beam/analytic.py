# rrivis/core/jones/beam/analytic.py
"""Composed aperture beam model for the E-Jones (primary beam) in the RIME.

This module provides :func:`compute_aperture_beam`, which combines aperture
shape, illumination taper, and optional feed model to produce a diagonal 2x2
Jones matrix per sky direction.

References
----------
.. [1] Balanis, C.A. "Antenna Theory: Analysis and Design" (4th ed., 2016)
       Chapter 12: Aperture Antennas
.. [2] Condon, J.J. & Ransom, S.M. "Essential Radio Astronomy" (2016)
       Chapter 3: Radio Telescopes and Radiometers
.. [3] Thompson, Moran, Swenson "Interferometry and Synthesis in Radio Astronomy"
       (3rd ed., 2017) Chapter 7: Design of the Receiving System
.. [4] Smirnov, O.M. "Revisiting the RIME I" (2011) A&A 527, A106
       Section 2.3: The E-Jones (primary beam)
"""

import numpy as np

# Speed of light in vacuum (m/s)
_C = 299_792_458.0


def compute_aperture_beam(
    theta: np.ndarray,
    phi: np.ndarray | None,
    frequency: float,
    diameter: float,
    aperture_shape: str = "circular",
    taper: str = "gaussian",
    edge_taper_dB: float = 10.0,
    feed_model: str = "none",
    feed_params: dict | None = None,
    feed_computation: str = "analytical",
    reflector_type: str = "prime_focus",
    magnification: float = 1.0,
    aperture_params: dict | None = None,
) -> np.ndarray:
    """Compute the composed beam Jones matrix.

    Combines aperture shape, illumination taper, optional feed model,
    to produce a diagonal 2x2 Jones matrix per sky direction.
    The computation proceeds in three steps:

    1. **Feed model** (if enabled): derive effective ``edge_taper_dB``
       analytically, or compute the far-field pattern numerically via
       Hankel transform.
    2. **Co-pol voltage**: evaluate the aperture+taper pattern at
       ``u_beam = D * sin(theta) * freq / c``.
    3. **Jones matrix**: assemble diagonal ``[[co, 0], [0, co]]``.

    Parameters
    ----------
    theta : ndarray
        Zenith angles in radians (scalar or array).
    phi : ndarray or None
        Azimuth angles in radians. Required for cross-pol and
        elliptical aperture; defaults to zero if ``None``.
    frequency : float
        Frequency in Hz.
    diameter : float
        Antenna diameter in metres.
    aperture_shape : str
        Aperture geometry: ``'circular'``, ``'rectangular'``,
        ``'elliptical'``.
    taper : str
        Illumination taper: ``'uniform'``, ``'gaussian'``,
        ``'parabolic'``, ``'parabolic_squared'``, ``'cosine'``.
    edge_taper_dB : float
        Edge taper in dB for tapers that accept it.
    feed_model : str
        Feed pattern model: ``'none'``, ``'corrugated_horn'``,
        ``'open_waveguide'``, ``'dipole_ground_plane'``.
    feed_params : dict or None
        Feed-specific parameters (``q``, ``b_over_lambda``,
        ``height_wavelengths``, ``focal_ratio``).
    feed_computation : str
        ``'analytical'`` (derive edge taper from feed edge
        illumination) or ``'numerical'`` (Hankel transform).
    reflector_type : str
        ``'prime_focus'`` or ``'cassegrain'``.
    magnification : float
        Cassegrain magnification ``M = (e+1)/(e-1)``.
    aperture_params : dict or None
        Aperture-specific parameters (``length_x``/``length_y`` for
        rectangular, ``diameter_x``/``diameter_y`` for elliptical).

    Returns
    -------
    ndarray
        Jones matrix, shape ``(2, 2)`` for scalar input or
        ``(N, 2, 2)`` for array input, dtype ``complex128``.
    """
    from rrivis.core.jones.beam.aperture import compute_u_beam
    from rrivis.core.jones.beam.taper import TAPER_FUNCTIONS

    if feed_params is None:
        feed_params = {}
    if aperture_params is None:
        aperture_params = {}

    input_is_scalar = np.ndim(theta) == 0
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))

    # Step 1: If feed_model != "none", compute effective edge_taper_dB
    effective_edge_taper = edge_taper_dB
    if feed_model != "none":
        from rrivis.core.jones.beam.feed import (
            compute_edge_taper_from_feed,
            feed_to_farfield_numerical,
        )

        focal_ratio = feed_params.get("focal_ratio", 0.4)
        focal_length = focal_ratio * diameter

        if feed_computation == "numerical":
            # Path B: numerical Hankel transform
            u_beam = compute_u_beam(theta, diameter, frequency)
            co_pol = feed_to_farfield_numerical(
                feed_model=feed_model,
                feed_params=feed_params,
                focal_length=focal_length,
                aperture_radius=diameter / 2,
                u_beam=u_beam,
                reflector_type=reflector_type,
                magnification=magnification,
            )
        else:
            # Path A: analytical bridge — derive effective edge taper
            effective_edge_taper = compute_edge_taper_from_feed(
                feed_model=feed_model,
                dish_diameter=diameter,
                focal_length=focal_length,
                feed_params=feed_params,
                reflector_type=reflector_type,
                magnification=magnification,
            )

    # Step 2: Compute co-pol voltage pattern
    if feed_model != "none" and feed_computation == "numerical":
        # Already computed above
        pass
    elif aperture_shape == "rectangular":
        from rrivis.core.jones.beam.aperture import sinc_voltage_pattern

        length_x = aperture_params.get("length_x", diameter)
        length_y = aperture_params.get("length_y", diameter)
        wavelength = _C / frequency
        phi_arr = phi if phi is not None else np.zeros_like(theta)
        # Decompose theta into x and y components using phi
        theta_x = theta * np.cos(phi_arr)
        theta_y = theta * np.sin(phi_arr)
        ux = length_x * np.sin(theta_x) / wavelength
        uy = length_y * np.sin(theta_y) / wavelength
        co_pol = sinc_voltage_pattern(ux, uy)
    elif aperture_shape == "elliptical":
        from rrivis.core.jones.beam.aperture import elliptical_airy_voltage_pattern

        diameter_x = aperture_params.get("diameter_x", diameter)
        diameter_y = aperture_params.get("diameter_y", diameter)
        wavelength = _C / frequency
        phi_arr = phi if phi is not None else np.zeros_like(theta)
        co_pol = elliptical_airy_voltage_pattern(
            theta, phi_arr, diameter_x, diameter_y, wavelength
        )
    else:
        # Circular aperture with taper
        u_beam = compute_u_beam(theta, diameter, frequency)
        taper_func = TAPER_FUNCTIONS.get(taper)
        if taper_func is None:
            raise ValueError(
                f"Unknown taper type: '{taper}'. "
                f"Available: {', '.join(TAPER_FUNCTIONS.keys())}"
            )
        # Tapers that accept edge_taper_dB
        if taper in ("gaussian", "parabolic", "parabolic_squared"):
            co_pol = taper_func(u_beam, edge_taper_dB=effective_edge_taper)
        else:
            co_pol = taper_func(u_beam)

    # Step 3: Build diagonal Jones matrix: [[co, 0], [0, co]]
    if input_is_scalar:
        jones = np.zeros((2, 2), dtype=np.complex128)
        jones[0, 0] = co_pol[0] if co_pol.ndim > 0 else co_pol
        jones[1, 1] = co_pol[0] if co_pol.ndim > 0 else co_pol
        return jones
    jones = np.zeros((len(co_pol), 2, 2), dtype=np.complex128)
    jones[:, 0, 0] = co_pol
    jones[:, 1, 1] = co_pol
    return jones


__all__ = [
    "compute_aperture_beam",
]
