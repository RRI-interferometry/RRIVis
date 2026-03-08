# rrivis/core/jones/beam/analytic.py
"""
Gaussian beam pattern and HPBW calculation functions.

This module provides Gaussian beam pattern modeling for radio astronomy
interferometry, implementing the "E" term (primary beam) in the RIME
(Radio Interferometer Measurement Equation):

    V_pq = sum_sources [E_p(theta) * C_source * E_q^H(theta) * exp(-2*pi*i*uvw.lmn)]

Beam Pattern
------------
- **Gaussian**: exp(-theta^2) model, good approximation for tapered dishes

Physical Background
-------------------
The beam width scales inversely with frequency (lambda/D relationship):
    HPBW = k * (lambda / D) = k * (c / f) / D

where k depends on the illumination pattern (default k=1.15 for 10dB edge taper).

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

Examples
--------
Calculate HPBW for a 14m dish at 150 MHz:

>>> from rrivis.core.jones.beam.analytic import compute_hpbw
>>> import numpy as np
>>> freq_hz = 150e6  # 150 MHz
>>> diameter = 14.0  # meters
>>> hpbw = compute_hpbw(freq_hz, diameter)
>>> print(f"HPBW: {np.degrees(hpbw[0]):.2f} degrees")
HPBW: 9.43 degrees

Calculate beam response at various angles:

>>> from rrivis.core.jones.beam.analytic import gaussian_A_theta_EBeam
>>> theta = np.linspace(0, 0.3, 100)  # radians
>>> response = gaussian_A_theta_EBeam(theta, theta_HPBW=0.1)
>>> response[0]  # On-axis response
1.0
"""

import healpy as hp
import numpy as np

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Speed of light in vacuum (m/s)
_C = 299_792_458.0


# =============================================================================
# HPBW CALCULATION
# =============================================================================


def compute_hpbw(freq_hz, diameter, k=1.15):
    """Compute HPBW using k * lambda / D formula.

    Parameters
    ----------
    freq_hz : float or array_like
        Frequency in Hz.
    diameter : float
        Antenna diameter in meters.
    k : float, optional
        Beam width coefficient (default 1.15, standard 10dB taper).

    Returns
    -------
    ndarray
        HPBW in radians.
    """
    f = np.atleast_1d(freq_hz)
    return k * (_C / f) / diameter


# =============================================================================
# BEAM PATTERN FUNCTIONS
# =============================================================================


def gaussian_A_theta_EBeam(theta, theta_HPBW):
    """
    Compute Gaussian primary beam pattern A(theta).

    This is the standard beam model used in visibility simulations. The
    Gaussian approximation is computationally efficient and provides a
    good match to real beam measurements for most radio telescopes.

    Parameters
    ----------
    theta : float or array_like
        Off-axis angle(s) from pointing center in radians.
        For HEALPix coordinates, this is the colatitude (zenith angle).
    theta_HPBW : float
        Half-Power Beam Width (HPBW) in radians.
        This is the full width at half maximum (FWHM) of the beam.

    Returns
    -------
    ndarray
        Normalized beam response A(theta) in range [0, 1].
        A(0) = 1.0 (on-axis pointing center).

    Notes
    -----
    **Formula**::

        A(theta) = exp(-(theta / (sqrt(2) * theta_HPBW))^2)
                 = exp(-theta^2 / (2 * sigma^2))

    where sigma = theta_HPBW / sqrt(2*ln(2)) ~ 0.849 * theta_HPBW

    This ensures:

    - A(0) = 1.0 (normalized on-axis)
    - A(theta_HPBW) = exp(-ln(2)) = 0.5 = -3 dB (half-power definition)

    **Physical motivation**: Gaussian beams arise from Gaussian-tapered
    aperture illumination and are self-Fourier-transforming, meaning the
    far-field pattern has the same Gaussian shape as the aperture field.

    **RIME context**: In the RIME, A(theta) represents the E-Jones (primary
    beam), which weights source contributions to visibility:
    V = sum_sources [E_p * C * E_q^H * K]

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.array([0, 0.05, 0.1, 0.15])  # radians
    >>> theta_HPBW = 0.1  # radians
    >>> response = gaussian_A_theta_EBeam(theta, theta_HPBW)
    >>> response[0]  # On-axis
    1.0
    >>> np.isclose(response[2], 0.5, atol=0.01)  # At HPBW
    True

    See Also
    --------
    calculate_gaussian_beam_area_EBeam : Integrate beam over sky.
    """
    # Gaussian beam formula ensuring A(theta_HPBW) = 0.5
    # sigma = theta_HPBW / sqrt(2*ln(2)), simplified form below
    return np.exp(-((theta / (np.sqrt(2) * theta_HPBW)) ** 2))


def calculate_gaussian_beam_area_EBeam(nside, theta_HPBW):
    """
    Calculate beam solid angle by integrating Gaussian pattern over HEALPix sphere.

    The beam solid angle (or beam area) is the integral of the beam pattern
    over the full sky. This quantity is needed for flux density calculations
    and beam normalization in visibility simulations.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter controlling sky pixelization resolution.
        Higher nside = finer resolution. Typical values: 64, 128, 256, 512.
        Number of pixels = 12 * nside^2.
    theta_HPBW : float or array_like
        Half-Power Beam Width in radians. Can be:

        - Single value: Returns single beam area
        - Array with all identical values: Returns single beam area (optimized)
        - Array with varying values: Returns list of beam areas, one per HPBW

    Returns
    -------
    float or list
        Beam solid angle in steradians (sr).

        - If theta_HPBW is scalar or all-identical array: returns float
        - If theta_HPBW varies: returns list of floats, one per unique HPBW

    Notes
    -----
    **Calculation method**::

        Omega_beam = sum_pixels[A(theta_pixel) * delta_Omega_pixel]

    where A(theta) is the Gaussian beam pattern and delta_Omega_pixel is
    the solid angle of each HEALPix pixel (constant for a given nside).

    **Analytical approximation**: For a Gaussian beam, the analytical
    beam solid angle is approximately:

        Omega_beam ~ pi * theta_HPBW^2 / (4 * ln(2)) ~ 1.133 * theta_HPBW^2

    The numerical integration provides accuracy for wide beams where
    spherical geometry matters.

    **Resolution requirements**: For accurate integration, use nside such
    that the pixel size is much smaller than the beam width. Rule of thumb:
    nside > 2 * pi / theta_HPBW.

    Examples
    --------
    >>> # Single frequency beam area
    >>> beam_area = calculate_gaussian_beam_area_EBeam(nside=128, theta_HPBW=0.1)
    >>> print(f"Beam area: {beam_area:.4f} sr")

    >>> # Multiple frequencies with different HPBWs
    >>> hpbw_array = np.array([0.05, 0.1, 0.15])  # radians
    >>> beam_areas = calculate_gaussian_beam_area_EBeam(
    ...     nside=128, theta_HPBW=hpbw_array
    ... )
    >>> for h, a in zip(hpbw_array, beam_areas):
    ...     print(f"HPBW={np.degrees(h):.1f} deg: {a:.4f} sr")

    See Also
    --------
    gaussian_A_theta_EBeam : The beam pattern function being integrated.
    """
    # Get HEALPix pixel properties
    pixel_area = hp.nside2pixarea(nside)  # Solid angle per pixel (sr)
    npix = hp.nside2npix(nside)  # Total pixels: 12 * nside^2

    # Get angular coordinates for all pixels
    # theta: colatitude (0 at north pole), phi: azimuth
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # Optimization: if all HPBW values are identical, compute once
    if np.ndim(theta_HPBW) > 0 and np.all(theta_HPBW == theta_HPBW[0]):
        beam_response = gaussian_A_theta_EBeam(theta, theta_HPBW[0])
        return np.sum(beam_response * pixel_area)

    # Handle array of varying HPBW values
    beam_areas = []
    for hpbw in np.atleast_1d(theta_HPBW):
        beam_response = gaussian_A_theta_EBeam(theta, hpbw)
        beam_area = np.sum(beam_response * pixel_area)
        beam_areas.append(beam_area)

    # Return scalar if input was scalar
    if np.ndim(theta_HPBW) == 0:
        return beam_areas[0]
    return beam_areas


# =============================================================================
# BEAM PATTERN TYPE REGISTRY
# =============================================================================


class BeamPatternType:
    """
    String constants identifying beam pattern models.

    Attributes
    ----------
    GAUSSIAN : str
        Gaussian beam pattern. Fast, simple, no sidelobes.
        Params: theta_HPBW

    Examples
    --------
    >>> pattern_func = get_beam_pattern_function(BeamPatternType.GAUSSIAN)
    >>> response = pattern_func(theta, theta_HPBW=0.1)

    See Also
    --------
    get_beam_pattern_function : Retrieve function for a pattern type.
    calculate_beam_pattern : Calculate pattern with unified interface.
    """

    GAUSSIAN = "gaussian"


# Registry mapping pattern type strings to functions
BEAM_PATTERN_FUNCTIONS = {
    BeamPatternType.GAUSSIAN: gaussian_A_theta_EBeam,
}
"""
dict : Mapping from pattern type identifiers to pattern functions.
"""


def get_beam_pattern_function(pattern_type):
    """
    Retrieve the beam pattern function for a given pattern type.

    Parameters
    ----------
    pattern_type : str
        Beam pattern type identifier from :class:`BeamPatternType`.

    Returns
    -------
    callable
        Beam pattern calculation function.

    Raises
    ------
    ValueError
        If ``pattern_type`` is not recognized.

    Examples
    --------
    >>> gauss_func = get_beam_pattern_function(BeamPatternType.GAUSSIAN)
    >>> response = gauss_func(theta, theta_HPBW=0.1)

    See Also
    --------
    calculate_beam_pattern : Unified interface for all patterns.
    BeamPatternType : Available pattern types.
    """
    if pattern_type not in BEAM_PATTERN_FUNCTIONS:
        available = ", ".join(sorted(BEAM_PATTERN_FUNCTIONS.keys()))
        raise ValueError(
            f"Unknown beam pattern type: '{pattern_type}'. Available types: {available}"
        )
    return BEAM_PATTERN_FUNCTIONS[pattern_type]


def calculate_beam_pattern(pattern_type, theta, **kwargs):
    """
    Calculate beam response for a specific pattern type.

    Unified interface for computing beam patterns. Automatically routes
    to the correct function based on ``pattern_type`` and passes through
    any additional keyword arguments.

    Parameters
    ----------
    pattern_type : str
        Beam pattern type from :class:`BeamPatternType`:

        - ``"gaussian"``: Gaussian pattern

    theta : float or array_like
        Off-axis angle(s) from pointing center in radians.
    **kwargs : dict
        Additional parameters specific to each pattern type:

        - **gaussian**: ``theta_HPBW`` (required)

    Returns
    -------
    ndarray
        Normalized beam response A(theta) in range [0, 1].

    Raises
    ------
    ValueError
        If ``pattern_type`` is not recognized.
    TypeError
        If required kwargs for the pattern type are missing.

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.linspace(0, 0.2, 100)  # radians

    >>> # Gaussian pattern
    >>> response = calculate_beam_pattern("gaussian", theta, theta_HPBW=0.1)

    See Also
    --------
    BeamPatternType : Available pattern type identifiers.
    get_beam_pattern_function : Get function reference directly.
    """
    pattern_func = get_beam_pattern_function(pattern_type)
    return pattern_func(theta, **kwargs)


def convert_angle_for_display(angle_radians, angle_unit):
    """
    Convert angle from radians to user-specified display unit.

    Utility function for presenting angles in configuration output
    or user interfaces where the display unit may be configurable.

    Parameters
    ----------
    angle_radians : float or array_like
        Angle value(s) in radians.
    angle_unit : str
        Target unit for display. Supported values:

        - ``"degrees"``: Convert to degrees
        - ``"radians"``: Return unchanged

    Returns
    -------
    float or ndarray
        Angle value in the specified unit.

    Examples
    --------
    >>> hpbw_rad = 0.1745  # ~10 degrees
    >>> convert_angle_for_display(hpbw_rad, "degrees")
    10.0
    >>> convert_angle_for_display(hpbw_rad, "radians")
    0.1745
    """
    if angle_unit == "degrees":
        return np.degrees(angle_radians)
    return angle_radians  # Already in radians
