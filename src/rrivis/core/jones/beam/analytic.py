# rrivis/core/beams.py
"""
Beam pattern and Half-Power Beam Width (HPBW) calculation functions.

This module provides comprehensive support for modeling antenna beam patterns
used in radio astronomy interferometry. It implements both:

1. **HPBW Calculations**: Frequency-dependent beam width for various antenna types
2. **Beam Pattern Models**: Spatial response functions A(theta) for visibility weighting

The primary beam pattern A(theta) determines how strongly a source at angle theta
from the pointing center contributes to the measured visibility. This is the "E" term
in the RIME (Radio Interferometer Measurement Equation):

    V_pq = sum_sources [E_p(theta) * C_source * E_q^H(theta) * exp(-2*pi*i*uvw.lmn)]

Supported Antenna Types
-----------------------
- **Parabolic dishes**: Uniform, cosine, Gaussian, 10dB/20dB edge taper
- **Spherical reflectors**: FAST/Arecibo-type with aberration corrections
- **Phased arrays**: Electronic beamforming (MWA, LOFAR tiles)
- **Dipoles**: Short, half-wave, and folded configurations (HERA, MWA elements)

Beam Pattern Models
-------------------
- **Gaussian**: Simple exp(-theta^2) model, good approximation for tapered dishes
- **Airy disk**: Exact diffraction pattern for uniform circular aperture
- **Cosine-tapered**: Models Cassegrain and offset-fed reflector illumination
- **Exponential**: Gaussian-like with configurable edge taper

Physical Background
-------------------
The beam width scales inversely with frequency (lambda/D relationship):
    HPBW ≈ k * (lambda / D) = k * (c / f) / D

where k depends on the illumination pattern:
- k = 1.02 for uniform illumination (theoretical minimum, high sidelobes)
- k = 1.15 for 10dB edge taper (standard choice, good sidelobe suppression)
- k = 1.22 for diffraction-limited approximation (commonly quoted)

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

>>> from rrivis.core.jones.beam import calculate_hpbw_for_antenna_type, AntennaType
>>> import numpy as np
>>> freq_hz = 150e6  # 150 MHz
>>> diameter = 14.0  # meters
>>> hpbw = calculate_hpbw_for_antenna_type(
...     AntennaType.PARABOLIC_10DB, freq_hz, diameter
... )
>>> print(f"HPBW: {np.degrees(hpbw):.2f} degrees")
HPBW: 9.43 degrees

Calculate beam response at various angles:

>>> from rrivis.core.jones.beam import gaussian_A_theta_EBeam
>>> theta = np.linspace(0, 0.3, 100)  # radians
>>> response = gaussian_A_theta_EBeam(theta, theta_HPBW=0.1)
>>> response[0]  # On-axis response
1.0
"""

from math import pi

import healpy as hp
import numpy as np
from scipy.special import j1  # First-order Bessel function for Airy pattern

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Speed of light in vacuum (m/s)
_C = 299_792_458.0


# =============================================================================
# ANTENNA TYPE CONSTANTS
# =============================================================================


class AntennaType:
    """
    String constants identifying antenna types for HPBW calculations.

    These identifiers are used with :func:`get_hpbw_function` and
    :func:`calculate_hpbw_for_antenna_type` to select the appropriate
    beam width formula based on physical antenna design and illumination pattern.

    The HPBW formula is: HPBW = k * (lambda / D), where k varies by type.

    Attributes
    ----------
    PARABOLIC_UNIFORM : str
        Uniform illumination (k=1.02). Theoretical max efficiency, high sidelobes.
    PARABOLIC_COSINE : str
        Cosine taper (k=1.10). Good efficiency/sidelobe compromise.
    PARABOLIC_GAUSSIAN : str
        Gaussian taper (k=1.18). Very low sidelobes, clean beam.
    PARABOLIC_10DB : str
        10 dB edge taper (k=1.15). **Standard for most radio telescopes.**
    PARABOLIC_20DB : str
        20 dB edge taper (k=1.25). Ultra-low sidelobes.
    SPHERICAL_UNIFORM : str
        Spherical reflector, uniform (k=1.05). FAST/Arecibo-type.
    SPHERICAL_GAUSSIAN : str
        Spherical reflector, Gaussian taper (k=1.20).
    PHASED_ARRAY : str
        Phased array (k=1.10). MWA, LOFAR, SKA-Low tiles.
    DIPOLE_SHORT : str
        Short dipole. Fixed 90 deg HPBW (frequency-independent).
    DIPOLE_HALFWAVE : str
        Half-wave dipole. Fixed 78 deg HPBW in E-plane.
    DIPOLE_FOLDED : str
        Folded dipole. Same as half-wave, higher impedance.

    Examples
    --------
    >>> from rrivis.core.jones.beam import AntennaType, get_hpbw_function
    >>> hpbw_func = get_hpbw_function(AntennaType.PARABOLIC_10DB)
    >>> hpbw_radians = hpbw_func(150e6, 14.0)  # 150 MHz, 14m dish

    See Also
    --------
    get_hpbw_function : Retrieve HPBW function for a given type.
    calculate_hpbw_for_antenna_type : Direct HPBW calculation.
    """

    # Parabolic dish antennas - various illumination tapers
    # Trade-off: taper reduces efficiency but suppresses sidelobes
    PARABOLIC_UNIFORM = "parabolic_uniform"  # k=1.02, 100% eff, -17.6 dB SL
    PARABOLIC_COSINE = "parabolic_cosine"  # k=1.10, ~81% eff, -23 dB SL
    PARABOLIC_GAUSSIAN = "parabolic_gaussian"  # k=1.18, ~75% eff, -40 dB SL
    PARABOLIC_10DB = "parabolic_10db_taper"  # k=1.15, ~75% eff, -30 dB SL
    PARABOLIC_20DB = "parabolic_20db_taper"  # k=1.25, ~62% eff, -42 dB SL

    # Spherical reflector antennas (FAST, former Arecibo)
    # Broader beams due to spherical aberration
    SPHERICAL_UNIFORM = "spherical_uniform"  # k=1.05
    SPHERICAL_GAUSSIAN = "spherical_gaussian"  # k=1.20

    # Phased array antennas (MWA tiles, LOFAR stations, SKA-Low)
    # D_eff = array extent, electronic beamforming
    PHASED_ARRAY = "phased_array"  # k=1.10

    # Dipole antennas (HERA, MWA elements, LOFAR LBA)
    # Frequency-independent HPBW (no aperture)
    DIPOLE_SHORT = "dipole_short"  # 90 deg constant
    DIPOLE_HALFWAVE = "dipole_halfwave"  # 78 deg E-plane
    DIPOLE_FOLDED = "dipole_folded"  # 78 deg, higher Z


# =============================================================================
# ANTENNA TYPE-SPECIFIC HPBW FUNCTIONS
# =============================================================================
#
# These functions implement the standard HPBW formula with antenna-specific
# coefficients that account for different illumination patterns:
#
#     HPBW = k * (lambda / D) = k * (c / f) / D
#
# The coefficient k is determined by the Fourier transform of the aperture
# illumination pattern. Stronger tapering (lower illumination at dish edges)
# increases k, producing a broader beam but lower sidelobes.
#
# Reference: Balanis "Antenna Theory", Table 12.2 for various tapers.


def _lambda_over_D(frequencies_hz, diameter, k):
    """
    Core HPBW calculation: k * (lambda / D).

    This is the fundamental diffraction formula relating beam width to
    wavelength and aperture size. All antenna-specific HPBW functions
    call this helper with their appropriate k coefficient.

    Parameters
    ----------
    frequencies_hz : float or array_like
        Observation frequency or frequencies in Hz.
    diameter : float
        Antenna diameter or effective aperture in meters.
        For phased arrays, this is N_elements * element_spacing.
    k : float
        Illumination-dependent coefficient. Typical values:
        - 1.02: Uniform illumination (theoretical minimum)
        - 1.10: Cosine or phased array taper
        - 1.15: 10 dB edge taper (standard)
        - 1.22: Rayleigh criterion (first null)
        - 1.25: 20 dB edge taper

    Returns
    -------
    ndarray
        HPBW values in radians.

    Notes
    -----
    This is the small-angle approximation where sin(theta) ≈ theta.
    Valid for HPBW < ~30 degrees, which covers most dish antennas.
    For wide-field dipoles, use the constant HPBW functions instead.
    """
    f = np.atleast_1d(frequencies_hz)
    # lambda = c / f  (wavelength in meters)
    wavelengths = _C / f
    # HPBW = k * (lambda / D)  (in radians, small-angle approximation)
    return k * (wavelengths / diameter)


# -----------------------------------------------------------------------------
# PARABOLIC DISH ANTENNAS
# -----------------------------------------------------------------------------
# Parabolic reflectors are the most common antenna type in radio astronomy.
# The feed illumination pattern determines the aperture efficiency and
# sidelobe levels. Stronger taper = wider beam, lower sidelobes.


def hpbw_parabolic_uniform(frequencies_hz, dish_diameter):
    """
    HPBW for parabolic dish with uniform aperture illumination.

    Calculates the Half-Power Beam Width using k=1.02, which corresponds
    to a uniformly illuminated circular aperture with no edge taper.

    Parameters
    ----------
    frequencies_hz : float or array_like
        Observation frequency or frequencies in Hz.
    dish_diameter : float
        Physical dish diameter in meters.

    Returns
    -------
    ndarray
        HPBW values in radians.

    Notes
    -----
    **Formula**: HPBW = 1.02 * (lambda / D)

    **Characteristics**:

    - Aperture efficiency: ~100% (theoretical maximum)
    - First sidelobe level: -17.6 dB (highest among tapers)
    - Beam shape: Narrowest possible for given aperture

    This is the theoretical limit for a circular aperture. Real feeds
    cannot achieve perfectly uniform illumination, so this model is
    primarily useful for theoretical comparisons.

    The Airy pattern from this illumination has nulls at:
    theta_null = 1.22, 2.23, 3.24, ... * (lambda / D)

    See Also
    --------
    hpbw_parabolic_10db_taper : Standard choice for real telescopes.
    airy_disk_pattern : Full beam pattern for uniform illumination.
    """
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.02)


def hpbw_parabolic_cosine(frequencies_hz, dish_diameter):
    """
    HPBW for parabolic dish with cosine illumination taper.

    Calculates the Half-Power Beam Width using k=1.10, which corresponds
    to a cosine-tapered aperture illumination pattern.

    Parameters
    ----------
    frequencies_hz : float or array_like
        Observation frequency or frequencies in Hz.
    dish_diameter : float
        Physical dish diameter in meters.

    Returns
    -------
    ndarray
        HPBW values in radians.

    Notes
    -----
    **Formula**: HPBW = 1.10 * (lambda / D)

    **Characteristics**:

    - Aperture efficiency: ~81%
    - First sidelobe level: -23 dB
    - Common for Cassegrain and offset-fed reflectors

    The cosine taper provides a good balance between efficiency and
    sidelobe suppression. The illumination drops to zero at the dish edge:
    I(r) = cos(pi * r / R) for r <= R

    See Also
    --------
    cosine_tapered_pattern : Full beam pattern with cosine taper.
    """
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.10)


def hpbw_parabolic_gaussian(frequencies_hz, dish_diameter):
    """
    HPBW for parabolic dish with Gaussian illumination taper.

    Calculates the Half-Power Beam Width using k=1.18, which corresponds
    to a Gaussian-tapered aperture illumination pattern.

    Parameters
    ----------
    frequencies_hz : float or array_like
        Observation frequency or frequencies in Hz.
    dish_diameter : float
        Physical dish diameter in meters.

    Returns
    -------
    ndarray
        HPBW values in radians.

    Notes
    -----
    **Formula**: HPBW = 1.18 * (lambda / D)

    **Characteristics**:

    - Aperture efficiency: ~75%
    - First sidelobe level: < -40 dB (excellent suppression)
    - Smooth, symmetric beam with no sharp features

    Gaussian illumination produces the cleanest beam with the lowest
    sidelobes, at the cost of reduced efficiency. Used when sidelobe
    contamination from bright sources must be minimized.

    The main beam itself is also Gaussian: A(theta) = exp(-theta^2 / sigma^2)

    See Also
    --------
    gaussian_A_theta_EBeam : Gaussian beam pattern model.
    """
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.18)


def hpbw_parabolic_10db_taper(frequencies_hz, dish_diameter):
    """
    HPBW for parabolic dish with 10 dB edge taper.

    Calculates the Half-Power Beam Width using k=1.15, the **standard
    choice for most radio telescopes** including VLA, ALMA, and MeerKAT.

    Parameters
    ----------
    frequencies_hz : float or array_like
        Observation frequency or frequencies in Hz.
    dish_diameter : float
        Physical dish diameter in meters.

    Returns
    -------
    ndarray
        HPBW values in radians.

    Notes
    -----
    **Formula**: HPBW = 1.15 * (lambda / D)

    **Characteristics**:

    - Aperture efficiency: ~75%
    - First sidelobe level: -30 dB (good suppression)
    - **Recommended default** for most simulations

    A 10 dB edge taper means the illumination at the dish edge is 10 dB
    (factor of 10 in power) below the center. This provides the best
    trade-off between sensitivity and sidelobe contamination.

    Examples
    --------
    >>> freq = 1.4e9  # 1.4 GHz (L-band)
    >>> diameter = 25.0  # VLA dish size
    >>> hpbw = hpbw_parabolic_10db_taper(freq, diameter)
    >>> print(f"VLA L-band HPBW: {np.degrees(hpbw):.2f} arcmin * 60")

    See Also
    --------
    AntennaType.PARABOLIC_10DB : Constant for this antenna type.
    exponential_tapered_pattern : Full beam pattern with dB taper.
    """
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.15)


def hpbw_parabolic_20db_taper(frequencies_hz, dish_diameter):
    """
    HPBW for parabolic dish with 20 dB edge taper.

    Calculates the Half-Power Beam Width using k=1.25, for antennas
    requiring ultra-low sidelobe levels.

    Parameters
    ----------
    frequencies_hz : float or array_like
        Observation frequency or frequencies in Hz.
    dish_diameter : float
        Physical dish diameter in meters.

    Returns
    -------
    ndarray
        HPBW values in radians.

    Notes
    -----
    **Formula**: HPBW = 1.25 * (lambda / D)

    **Characteristics**:

    - Aperture efficiency: ~62% (significant reduction)
    - First sidelobe level: -42 dB (excellent suppression)
    - Broadest beam among parabolic tapers

    A 20 dB edge taper is used when sidelobe contamination must be
    minimized, such as when observing near very bright sources (Sun,
    Cas A) or for precision polarimetry where leakage is critical.

    The reduced efficiency means longer integration times are needed
    to achieve the same sensitivity as a 10 dB taper.

    See Also
    --------
    hpbw_parabolic_10db_taper : Standard choice with better efficiency.
    """
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.25)


# -----------------------------------------------------------------------------
# SPHERICAL REFLECTOR ANTENNAS (FAST, former Arecibo)
# -----------------------------------------------------------------------------
# Spherical reflectors have a fixed reflecting surface (unlike steerable
# parabolic dishes). The beam is formed by moving the feed, and spherical
# aberration slightly broadens the beam compared to parabolic equivalents.
# D_eff is the *illuminated* diameter, not the full sphere diameter.


def hpbw_spherical_uniform(frequencies_hz, diameter_effective):
    """
    HPBW for spherical reflector with uniform illumination.

    Calculates the Half-Power Beam Width using k=1.05 for spherical
    reflector geometry with uniform aperture illumination.

    Parameters
    ----------
    frequencies_hz : float or array_like
        Observation frequency or frequencies in Hz.
    diameter_effective : float
        Effective illuminated diameter in meters. For FAST, this is
        ~300m (illuminated) out of 500m total diameter.

    Returns
    -------
    ndarray
        HPBW values in radians.

    Notes
    -----
    **Formula**: HPBW = 1.05 * (lambda / D_eff)

    Spherical reflectors have slightly broader beams than parabolic dishes
    of the same illuminated diameter due to spherical aberration. The
    aberration causes phase errors across the aperture that cannot be
    perfectly corrected.

    For FAST (Five-hundred-meter Aperture Spherical Telescope):

    - Total diameter: 500m
    - Illuminated diameter: ~300m
    - Use D_eff = 300m for HPBW calculations

    See Also
    --------
    hpbw_spherical_gaussian : More realistic model with taper.
    """
    return _lambda_over_D(frequencies_hz, diameter_effective, 1.05)


def hpbw_spherical_gaussian(frequencies_hz, diameter_effective):
    """
    HPBW for spherical reflector with Gaussian illumination taper.

    Calculates the Half-Power Beam Width using k=1.20 for spherical
    reflector geometry with Gaussian-tapered illumination.

    Parameters
    ----------
    frequencies_hz : float or array_like
        Observation frequency or frequencies in Hz.
    diameter_effective : float
        Effective illuminated diameter in meters.

    Returns
    -------
    ndarray
        HPBW values in radians.

    Notes
    -----
    **Formula**: HPBW = 1.20 * (lambda / D_eff)

    This is a more realistic model for FAST-type telescopes with active
    surface panels, combining the effects of:

    - Gaussian illumination taper from the feed
    - Spherical aberration from the reflector geometry
    - Active surface corrections

    The broader k=1.20 coefficient (vs 1.18 for parabolic Gaussian)
    accounts for residual aberration after active correction.

    See Also
    --------
    hpbw_spherical_uniform : Uniform illumination model.
    """
    return _lambda_over_D(frequencies_hz, diameter_effective, 1.20)


# -----------------------------------------------------------------------------
# PHASED ARRAY ANTENNAS (MWA, LOFAR, SKA-Low)
# -----------------------------------------------------------------------------
# Phased arrays form beams electronically by adjusting element phases.
# The "effective diameter" is the physical extent of the array, typically
# D_eff = N * d where N is number of elements and d is element spacing.
# Most arrays use lambda/2 spacing to avoid grating lobes.


def hpbw_phased_array(frequencies_hz, diameter_effective):
    """
    HPBW for phased array antenna tile or station.

    Calculates the Half-Power Beam Width using k=1.10 for electronically
    steered phased array systems like MWA tiles and LOFAR stations.

    Parameters
    ----------
    frequencies_hz : float or array_like
        Observation frequency or frequencies in Hz.
    diameter_effective : float
        Effective array aperture in meters, typically:
        D_eff = N_elements * element_spacing

        For MWA 4x4 tile with 1.1m spacing: D_eff ≈ 4.4m
        For LOFAR HBA station: D_eff ≈ 30m

    Returns
    -------
    ndarray
        HPBW values in radians.

    Notes
    -----
    **Formula**: HPBW = 1.10 * (lambda / D_eff)

    Phased arrays have unique properties:

    - Electronic beam steering (no mechanical movement)
    - Beam broadening when steered off-zenith: HPBW / cos(zenith_angle)
    - Frequency-dependent element spacing in wavelengths
    - Grating lobes if spacing > lambda/2

    The k=1.10 coefficient assumes uniform element weighting. Tapered
    weighting (for sidelobe control) would increase k.

    **Beam steering effect**: When pointing at zenith angle theta_z,
    the effective aperture is reduced by cos(theta_z), broadening the beam.

    Examples
    --------
    >>> # MWA tile at 150 MHz
    >>> freq = 150e6
    >>> d_eff = 4.4  # 4x4 tile, 1.1m spacing
    >>> hpbw = hpbw_phased_array(freq, d_eff)
    >>> print(f"MWA tile HPBW: {np.degrees(hpbw):.1f} degrees")
    MWA tile HPBW: 27.5 degrees

    See Also
    --------
    AntennaType.PHASED_ARRAY : Constant for this antenna type.
    """
    return _lambda_over_D(frequencies_hz, diameter_effective, 1.10)


# -----------------------------------------------------------------------------
# DIPOLE ANTENNAS (HERA, MWA elements, LOFAR LBA)
# -----------------------------------------------------------------------------
# Dipole antennas have no aperture in the traditional sense, so their HPBW
# is frequency-independent and determined by the dipole radiation pattern.
# The "diameter" parameter is ignored but kept for API consistency.
#
# Short dipole (L << lambda): sin^2(theta) pattern, 90 deg HPBW
# Half-wave dipole (L = lambda/2): more directive, 78 deg HPBW


def hpbw_dipole_short(frequencies_hz, _unused_diameter):
    """
    HPBW for short dipole antenna (L << lambda).

    Returns a constant HPBW of 90 degrees (pi/2 radians), independent
    of frequency. Short dipoles are used in wide-field arrays like HERA.

    Parameters
    ----------
    frequencies_hz : float or array_like
        Observation frequency or frequencies in Hz.
        **Not used** - HPBW is frequency-independent for dipoles.
    _unused_diameter : float
        Ignored. Present for API consistency with dish functions.

    Returns
    -------
    ndarray
        HPBW values in radians (constant pi/2 = 90 degrees).

    Notes
    -----
    **Formula**: HPBW = pi/2 radians = 90 degrees (constant)

    Short dipole characteristics:

    - **E-plane**: Figure-8 pattern, sin^2(theta), HPBW = 90 degrees
    - **H-plane**: Omnidirectional (no directivity)
    - No frequency dependence (unlike aperture antennas)
    - Used by: HERA, MWA (individual elements), LOFAR LBA

    The wide beam makes dipoles ideal for all-sky surveys and 21cm
    cosmology where the entire visible sky must be observed.

    The radiation pattern is: P(theta) = sin^2(theta)
    where theta is measured from the dipole axis.

    See Also
    --------
    hpbw_dipole_halfwave : Slightly more directive half-wave dipole.
    """
    # Return constant array matching input shape
    return np.full_like(np.atleast_1d(frequencies_hz), pi / 2, dtype=float)


def hpbw_dipole_halfwave(frequencies_hz, _unused_diameter):
    """
    HPBW for half-wave dipole antenna (L = lambda/2).

    Returns a constant HPBW of 78 degrees in the E-plane, independent
    of frequency. This is the most common resonant dipole configuration.

    Parameters
    ----------
    frequencies_hz : float or array_like
        Observation frequency or frequencies in Hz.
        **Not used** - HPBW is frequency-independent for dipoles.
    _unused_diameter : float
        Ignored. Present for API consistency with dish functions.

    Returns
    -------
    ndarray
        HPBW values in radians (constant 78 degrees = 1.36 rad).

    Notes
    -----
    **Formula**: HPBW = 78 degrees = 1.36 radians (constant)

    Half-wave dipole characteristics:

    - **E-plane**: HPBW = 78 degrees (slightly more directive than short dipole)
    - **H-plane**: Omnidirectional
    - Radiation resistance: ~73 Ohms (good impedance match)
    - Length: L = lambda/2 = c / (2*f)

    The radiation pattern is more complex than sin^2(theta):
    P(theta) = [cos(pi/2 * cos(theta)) / sin(theta)]^2

    See Also
    --------
    hpbw_dipole_short : Wider-beam short dipole.
    hpbw_dipole_folded : Same pattern, different impedance.
    """
    # 78 degrees in radians = 78 * pi / 180
    return np.full_like(np.atleast_1d(frequencies_hz), 78.0 * pi / 180.0, dtype=float)


def hpbw_dipole_folded(frequencies_hz, _unused_diameter):
    """
    HPBW for folded dipole antenna.

    Returns the same HPBW as a half-wave dipole (78 degrees). The folded
    configuration affects impedance but not the radiation pattern.

    Parameters
    ----------
    frequencies_hz : float or array_like
        Observation frequency or frequencies in Hz.
        **Not used** - HPBW is frequency-independent for dipoles.
    _unused_diameter : float
        Ignored. Present for API consistency with dish functions.

    Returns
    -------
    ndarray
        HPBW values in radians (constant 78 degrees = 1.36 rad).

    Notes
    -----
    **Formula**: HPBW = 78 degrees (same as half-wave dipole)

    Folded dipole characteristics:

    - **Radiation pattern**: Identical to half-wave dipole
    - **Impedance**: ~280-300 Ohms (4x higher than simple dipole)
    - **Bandwidth**: Wider than simple half-wave dipole
    - Common in Yagi arrays and broadband applications

    The higher impedance makes folded dipoles easier to match to
    balanced transmission lines (300 Ohm twin-lead).

    See Also
    --------
    hpbw_dipole_halfwave : Simple half-wave dipole.
    """
    return hpbw_dipole_halfwave(frequencies_hz, _unused_diameter)


# -----------------------------------------------------------------------------
# DEGREE VERSIONS (for user convenience)
# -----------------------------------------------------------------------------
# These wrapper functions return HPBW in degrees instead of radians.
# Useful for display purposes and configuration files where degrees
# are more intuitive for users.


def hpbw_parabolic_uniform_deg(frequencies_hz, dish_diameter):
    """HPBW for parabolic uniform illumination, in degrees."""
    return np.degrees(hpbw_parabolic_uniform(frequencies_hz, dish_diameter))


def hpbw_parabolic_cosine_deg(frequencies_hz, dish_diameter):
    """HPBW for parabolic cosine taper, in degrees."""
    return np.degrees(hpbw_parabolic_cosine(frequencies_hz, dish_diameter))


def hpbw_parabolic_gaussian_deg(frequencies_hz, dish_diameter):
    """HPBW for parabolic Gaussian taper, in degrees."""
    return np.degrees(hpbw_parabolic_gaussian(frequencies_hz, dish_diameter))


def hpbw_parabolic_10db_taper_deg(frequencies_hz, dish_diameter):
    """HPBW for parabolic 10 dB edge taper, in degrees."""
    return np.degrees(hpbw_parabolic_10db_taper(frequencies_hz, dish_diameter))


def hpbw_parabolic_20db_taper_deg(frequencies_hz, dish_diameter):
    """HPBW for parabolic 20 dB edge taper, in degrees."""
    return np.degrees(hpbw_parabolic_20db_taper(frequencies_hz, dish_diameter))


def hpbw_spherical_uniform_deg(frequencies_hz, diameter_effective):
    """HPBW for spherical uniform illumination, in degrees."""
    return np.degrees(hpbw_spherical_uniform(frequencies_hz, diameter_effective))


def hpbw_spherical_gaussian_deg(frequencies_hz, diameter_effective):
    """HPBW for spherical Gaussian taper, in degrees."""
    return np.degrees(hpbw_spherical_gaussian(frequencies_hz, diameter_effective))


def hpbw_phased_array_deg(frequencies_hz, diameter_effective):
    """HPBW for phased array, in degrees."""
    return np.degrees(hpbw_phased_array(frequencies_hz, diameter_effective))


def hpbw_dipole_short_deg(frequencies_hz, _unused_diameter):
    """HPBW for short dipole, in degrees (constant 90 deg)."""
    return np.degrees(hpbw_dipole_short(frequencies_hz, _unused_diameter))


def hpbw_dipole_halfwave_deg(frequencies_hz, _unused_diameter):
    """HPBW for half-wave dipole, in degrees (constant 78 deg)."""
    return np.degrees(hpbw_dipole_halfwave(frequencies_hz, _unused_diameter))


def hpbw_dipole_folded_deg(frequencies_hz, _unused_diameter):
    """HPBW for folded dipole, in degrees (constant 78 deg)."""
    return np.degrees(hpbw_dipole_folded(frequencies_hz, _unused_diameter))


# =============================================================================
# HPBW FUNCTION REGISTRY
# =============================================================================
#
# This registry maps AntennaType string identifiers to their corresponding
# HPBW calculation functions. Use get_hpbw_function() or
# calculate_hpbw_for_antenna_type() to access functions by type name.

HPBW_FUNCTIONS = {
    # Parabolic dish antennas (returns radians)
    AntennaType.PARABOLIC_UNIFORM: hpbw_parabolic_uniform,
    AntennaType.PARABOLIC_COSINE: hpbw_parabolic_cosine,
    AntennaType.PARABOLIC_GAUSSIAN: hpbw_parabolic_gaussian,
    AntennaType.PARABOLIC_10DB: hpbw_parabolic_10db_taper,
    AntennaType.PARABOLIC_20DB: hpbw_parabolic_20db_taper,
    # Spherical reflector antennas (returns radians)
    AntennaType.SPHERICAL_UNIFORM: hpbw_spherical_uniform,
    AntennaType.SPHERICAL_GAUSSIAN: hpbw_spherical_gaussian,
    # Phased array antennas (returns radians)
    AntennaType.PHASED_ARRAY: hpbw_phased_array,
    # Dipole antennas (returns radians, frequency-independent)
    AntennaType.DIPOLE_SHORT: hpbw_dipole_short,
    AntennaType.DIPOLE_HALFWAVE: hpbw_dipole_halfwave,
    AntennaType.DIPOLE_FOLDED: hpbw_dipole_folded,
}
"""
dict : Mapping from antenna type identifiers to HPBW functions.

All functions in this registry have the signature:
    func(frequencies_hz, diameter) -> ndarray (radians)

The diameter parameter is ignored for dipole types but required
for API consistency.
"""


def get_hpbw_function(antenna_type):
    """
    Retrieve the HPBW calculation function for a given antenna type.

    This function provides access to the antenna-specific HPBW calculations
    through the string-based AntennaType identifiers. Useful when the antenna
    type is determined at runtime (e.g., from configuration files).

    Parameters
    ----------
    antenna_type : str
        Antenna type identifier from :class:`AntennaType`, e.g.,
        ``AntennaType.PARABOLIC_10DB`` or ``"parabolic_10db_taper"``.

    Returns
    -------
    callable
        HPBW calculation function with signature:
        ``func(frequencies_hz, diameter) -> ndarray``
        Returns HPBW in radians.

    Raises
    ------
    ValueError
        If ``antenna_type`` is not a recognized type in :data:`HPBW_FUNCTIONS`.

    Examples
    --------
    >>> hpbw_func = get_hpbw_function(AntennaType.PARABOLIC_10DB)
    >>> hpbw = hpbw_func(150e6, 14.0)
    >>> print(f"HPBW: {np.degrees(hpbw):.2f} degrees")

    >>> # Using string directly
    >>> hpbw_func = get_hpbw_function("parabolic_10db_taper")

    See Also
    --------
    calculate_hpbw_for_antenna_type : Convenience function that combines lookup and call.
    AntennaType : Available antenna type identifiers.
    HPBW_FUNCTIONS : The underlying registry dictionary.
    """
    if antenna_type not in HPBW_FUNCTIONS:
        available = ", ".join(sorted(HPBW_FUNCTIONS.keys()))
        raise ValueError(
            f"Unknown antenna type: '{antenna_type}'. Available types: {available}"
        )
    return HPBW_FUNCTIONS[antenna_type]


def calculate_hpbw_for_antenna_type(antenna_type, frequencies_hz, diameter):
    """
    Calculate HPBW for a specific antenna type.

    Convenience function that combines :func:`get_hpbw_function` lookup
    with immediate evaluation. This is the recommended high-level interface
    for antenna-specific HPBW calculations.

    Parameters
    ----------
    antenna_type : str
        Antenna type identifier from :class:`AntennaType`.
    frequencies_hz : float or array_like
        Observation frequency or frequencies in Hz.
    diameter : float
        Antenna diameter or effective aperture in meters.
        Ignored for dipole types but required for API consistency.

    Returns
    -------
    ndarray
        HPBW values in radians, one per input frequency.

    Examples
    --------
    >>> from rrivis.core.jones.beam import AntennaType, calculate_hpbw_for_antenna_type
    >>> import numpy as np

    >>> # Single frequency
    >>> hpbw = calculate_hpbw_for_antenna_type(AntennaType.PARABOLIC_10DB, 150e6, 14.0)
    >>> print(f"HPBW at 150 MHz: {np.degrees(hpbw)[0]:.2f} degrees")
    HPBW at 150 MHz: 9.43 degrees

    >>> # Multiple frequencies
    >>> freqs = np.array([100e6, 150e6, 200e6])
    >>> hpbw = calculate_hpbw_for_antenna_type(AntennaType.PARABOLIC_10DB, freqs, 14.0)
    >>> for f, h in zip(freqs, hpbw):
    ...     print(f"{f / 1e6:.0f} MHz: {np.degrees(h):.2f} deg")
    100 MHz: 14.14 deg
    150 MHz: 9.43 deg
    200 MHz: 7.07 deg

    See Also
    --------
    get_hpbw_function : Get function reference for repeated calls.
    AntennaType : Available antenna type identifiers.
    """
    hpbw_func = get_hpbw_function(antenna_type)
    return hpbw_func(frequencies_hz, diameter)


# =============================================================================
# BEAM PATTERN MODELS
# =============================================================================
#
# These functions compute the spatial response A(theta) of the primary beam
# as a function of off-axis angle. The beam response weights source contributions
# in visibility calculations:
#
#     V_pq = sum_sources [A(theta) * S * exp(-2*pi*i*uvw.lmn)]
#
# All patterns are normalized to unity on-axis: A(0) = 1.0
#
# The choice of pattern model affects:
# - Sidelobe levels (important for bright source contamination)
# - Main beam shape (affects source flux recovery)
# - Computational cost (Gaussian is cheapest, Airy most expensive)


def airy_disk_pattern(theta, wavelength, diameter):
    """
    Compute Airy disk diffraction pattern for uniform circular aperture.

    The Airy pattern is the exact Fraunhofer diffraction solution for a
    uniformly illuminated circular aperture. It provides the most physically
    accurate beam model for dishes with uniform feed illumination.

    Parameters
    ----------
    theta : float or array_like
        Off-axis angle(s) from pointing center in radians.
    wavelength : float
        Observation wavelength in meters (lambda = c / f).
    diameter : float
        Antenna diameter in meters.

    Returns
    -------
    ndarray
        Normalized beam response A(theta) in range [0, 1].
        A(0) = 1.0 (on-axis).

    Notes
    -----
    **Formula**::

        A(theta) = [2 * J1(x) / x]^2

    where::

        x = pi * D * sin(theta) / lambda

    and J1 is the first-order Bessel function of the first kind.

    **Characteristics**:

    - First null: theta_null = 1.22 * lambda / D (Rayleigh criterion)
    - First sidelobe: -17.6 dB at theta ≈ 1.63 * lambda / D
    - Second null: theta = 2.23 * lambda / D
    - Infinite series of diminishing sidelobes

    The Airy pattern is more computationally expensive than Gaussian but
    provides realistic sidelobe structure. Use when sidelobe contamination
    from bright sources is a concern.

    **Numerical handling**: Uses L'Hopital's rule at x=0 where J1(x)/x -> 0.5,
    giving A(0) = [2 * 0.5]^2 = 1.0.

    References
    ----------
    .. [1] Born & Wolf, "Principles of Optics", Chapter 8
    .. [2] Balanis, "Antenna Theory", Section 12.2

    Examples
    --------
    >>> theta = np.linspace(0, 0.1, 100)  # radians
    >>> wavelength = 2.0  # 150 MHz
    >>> response = airy_disk_pattern(theta, wavelength, diameter=14.0)
    >>> response[0]  # On-axis
    1.0
    """
    theta = np.atleast_1d(theta)

    # Compute the Airy pattern argument: x = pi * D * sin(theta) / lambda
    # For small angles, sin(theta) ≈ theta, but we use sin() for accuracy
    x = np.pi * diameter * np.sin(theta) / wavelength

    # Handle x=0 case (on-axis) where J1(x)/x -> 0.5 by L'Hopital's rule
    # lim_{x->0} J1(x)/x = 0.5, so [2 * J1(x)/x]^2 -> 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            np.abs(x) < 1e-10,
            1.0,  # On-axis value (avoids 0/0)
            (2 * j1(x) / x) ** 2,  # Standard Airy formula
        )

    return result


def cosine_tapered_pattern(theta, theta_HPBW, taper_exponent=1.0):
    """
    Compute cosine-tapered beam pattern.

    Models the far-field pattern from a cosine-tapered aperture illumination,
    common in Cassegrain and offset-fed reflector systems.

    Parameters
    ----------
    theta : float or array_like
        Off-axis angle(s) from pointing center in radians.
    theta_HPBW : float
        Half-Power Beam Width in radians.
    taper_exponent : float, optional
        Cosine exponent n in cos^n formula. Default is 1.0.

        - n=1.0: Standard cosine, ~81% efficiency, -23 dB sidelobes
        - n=2.0: Cosine-squared, ~67% efficiency, -32 dB sidelobes
        - n=3.0: Cosine-cubed, ~55% efficiency, very low sidelobes

    Returns
    -------
    ndarray
        Normalized beam response A(theta) in range [0, 1].
        Returns 0.0 outside the beam edge (hard cutoff).

    Notes
    -----
    **Formula**::

        A(theta) = cos^n(pi/2 * theta / theta_edge)  for |theta| < theta_edge
                 = 0                                  otherwise

    where theta_edge ≈ 1.4 * theta_HPBW (approximate conversion).

    Unlike Gaussian or Airy patterns, the cosine pattern has a hard cutoff
    at the beam edge. This makes it computationally efficient for simulations
    where sources outside the main beam can be ignored.

    **Physical interpretation**: The exponent n controls the illumination
    taper strength. Higher n means more aggressive edge tapering, which
    reduces sidelobes but also reduces aperture efficiency.

    Examples
    --------
    >>> theta = np.linspace(0, 0.2, 100)
    >>> response = cosine_tapered_pattern(theta, theta_HPBW=0.1, taper_exponent=2.0)
    >>> response[0]  # On-axis
    1.0
    """
    theta = np.atleast_1d(theta)

    # Convert HPBW to beam edge angle
    # For cosine pattern, the relationship is approximately theta_edge ≈ 1.4 * HPBW
    # This ensures the -3dB point falls at theta_HPBW
    theta_edge = 1.4 * theta_HPBW

    # Cosine^n taper with hard cutoff at beam edge
    # The pi/2 factor maps theta_edge to the cosine zero-crossing
    result = np.where(
        np.abs(theta) < theta_edge,
        np.cos(np.pi / 2 * theta / theta_edge) ** taper_exponent,
        0.0,  # Hard cutoff outside beam edge
    )

    return result


def exponential_tapered_pattern(theta, theta_HPBW, taper_dB=10.0):
    """
    Compute exponential (Gaussian-like) tapered beam pattern.

    Models the far-field pattern from exponentially-tapered aperture
    illumination, characteristic of feed horns and Gaussian-tapered dishes.

    Parameters
    ----------
    theta : float or array_like
        Off-axis angle(s) from pointing center in radians.
    theta_HPBW : float
        Half-Power Beam Width in radians.
    taper_dB : float, optional
        Edge taper parameter in dB. Default is 10.0.
        **Note**: This parameter is currently unused in the calculation;
        the pattern is defined purely by theta_HPBW.

    Returns
    -------
    ndarray
        Normalized beam response A(theta) in range [0, 1].
        A(0) = 1.0 (on-axis).

    Notes
    -----
    **Formula**::

        A(theta) = exp(-alpha * theta^2)

    where::

        alpha = ln(2) / theta_HPBW ^ 2

    This ensures A(theta_HPBW) = 0.5 = -3 dB (half-power definition).

    **Comparison to Gaussian**:
    This is equivalent to :func:`gaussian_A_theta_EBeam` with a different
    parameterization. The Gaussian width sigma relates to HPBW as:
    sigma = theta_HPBW / sqrt(2*ln(2)) ≈ theta_HPBW / 1.177

    Examples
    --------
    >>> theta = np.linspace(0, 0.3, 100)
    >>> response = exponential_tapered_pattern(theta, theta_HPBW=0.1)
    >>> response[0]  # On-axis
    1.0
    >>> np.isclose(response[np.argmin(np.abs(theta - 0.1))], 0.5, atol=0.01)
    True
    """
    theta = np.atleast_1d(theta)

    # Compute alpha to give -3 dB (half power) at theta = theta_HPBW
    # A(theta_HPBW) = exp(-alpha * theta_HPBW^2) = 0.5
    # -> alpha = ln(2) / theta_HPBW^2
    alpha = np.log(2) / theta_HPBW**2

    # Gaussian beam pattern
    result = np.exp(-alpha * theta**2)

    return result


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

    where sigma = theta_HPBW / sqrt(2*ln(2)) ≈ 0.849 * theta_HPBW

    This ensures:

    - A(0) = 1.0 (normalized on-axis)
    - A(theta_HPBW) = exp(-ln(2)) = 0.5 = -3 dB (half-power definition)

    **Physical motivation**: Gaussian beams arise from Gaussian-tapered
    aperture illumination and are self-Fourier-transforming, meaning the
    far-field pattern has the same Gaussian shape as the aperture field.

    **Comparison to Airy pattern**: The Gaussian has no sidelobes (decays
    monotonically), while real beams have sidelobe structure. Use Airy
    pattern when sidelobe effects matter.

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
    exponential_tapered_pattern : Alternative parameterization.
    airy_disk_pattern : Physically accurate pattern with sidelobes.
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

        Omega_beam ≈ pi * theta_HPBW^2 / (4 * ln(2)) ≈ 1.133 * theta_HPBW^2

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
    calculate_airy_beam_area : Beam area for Airy pattern.
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


def calculate_airy_beam_area(nside, wavelength, diameter):
    """
    Calculate beam solid angle for Airy disk pattern via HEALPix integration.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter (resolution). Use nside >= 128 for accuracy.
    wavelength : float
        Observation wavelength in meters.
    diameter : float
        Antenna diameter in meters.

    Returns
    -------
    float
        Beam solid angle in steradians (sr).

    Notes
    -----
    The Airy pattern includes sidelobes, so the integrated beam area is
    slightly larger than a Gaussian with the same HPBW. The sidelobe
    contribution is approximately 16% of the main beam.

    See Also
    --------
    airy_disk_pattern : The beam pattern function being integrated.
    calculate_gaussian_beam_area_EBeam : Gaussian equivalent.
    """
    pixel_area = hp.nside2pixarea(nside)
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    beam_response = airy_disk_pattern(theta, wavelength, diameter)
    return np.sum(beam_response * pixel_area)


def calculate_cosine_beam_area(nside, theta_HPBW, taper_exponent=1.0):
    """
    Calculate beam solid angle for cosine-tapered pattern via HEALPix integration.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter (resolution).
    theta_HPBW : float
        Half-Power Beam Width in radians.
    taper_exponent : float, optional
        Cosine exponent n in cos^n formula. Default is 1.0.

    Returns
    -------
    float
        Beam solid angle in steradians (sr).

    Notes
    -----
    The cosine pattern has a hard cutoff at the beam edge, so the beam
    area is well-defined and finite. This makes it useful for simulations
    where computational efficiency is important.

    See Also
    --------
    cosine_tapered_pattern : The beam pattern function being integrated.
    """
    pixel_area = hp.nside2pixarea(nside)
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    beam_response = cosine_tapered_pattern(theta, theta_HPBW, taper_exponent)
    return np.sum(beam_response * pixel_area)


def calculate_exponential_beam_area(nside, theta_HPBW, taper_dB=10.0):
    """
    Calculate beam solid angle for exponential-tapered pattern via HEALPix integration.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter (resolution).
    theta_HPBW : float
        Half-Power Beam Width in radians.
    taper_dB : float, optional
        Edge taper in dB. Default is 10.0. (Currently unused in calculation.)

    Returns
    -------
    float
        Beam solid angle in steradians (sr).

    See Also
    --------
    exponential_tapered_pattern : The beam pattern function being integrated.
    calculate_gaussian_beam_area_EBeam : Equivalent for Gaussian pattern.
    """
    pixel_area = hp.nside2pixarea(nside)
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    beam_response = exponential_tapered_pattern(theta, theta_HPBW, taper_dB)
    return np.sum(beam_response * pixel_area)


# =============================================================================
# BEAM PATTERN TYPE REGISTRY
# =============================================================================
#
# This registry maps beam pattern type identifiers to their corresponding
# pattern functions. Use get_beam_pattern_function() or calculate_beam_pattern()
# to access patterns by type name.
#
# Note: Different patterns have different function signatures:
# - Gaussian/Exponential: func(theta, theta_HPBW, ...)
# - Airy: func(theta, wavelength, diameter)
# - Cosine: func(theta, theta_HPBW, taper_exponent)


class BeamPatternType:
    """
    String constants identifying beam pattern models.

    These identifiers are used with :func:`get_beam_pattern_function` and
    :func:`calculate_beam_pattern` to select the appropriate beam model.

    Attributes
    ----------
    GAUSSIAN : str
        Gaussian beam pattern. Fast, simple, no sidelobes.
        Params: theta_HPBW
    AIRY : str
        Airy disk diffraction pattern. Physically accurate, has sidelobes.
        Params: wavelength, diameter
    COSINE : str
        Cosine-tapered pattern with hard cutoff at beam edge.
        Params: theta_HPBW, taper_exponent
    EXPONENTIAL : str
        Exponential (Gaussian-like) pattern.
        Params: theta_HPBW, taper_dB

    Examples
    --------
    >>> pattern_func = get_beam_pattern_function(BeamPatternType.GAUSSIAN)
    >>> response = pattern_func(theta, theta_HPBW=0.1)

    See Also
    --------
    get_beam_pattern_function : Retrieve function for a pattern type.
    calculate_beam_pattern : Calculate pattern with unified interface.
    """

    GAUSSIAN = "gaussian"  # Simple Gaussian, fastest computation
    AIRY = "airy"  # Airy disk, includes sidelobes
    COSINE = "cosine"  # Cosine taper, hard edge cutoff
    EXPONENTIAL = "exponential"  # Exponential/Gaussian-like taper


# Registry mapping pattern type strings to functions
BEAM_PATTERN_FUNCTIONS = {
    BeamPatternType.GAUSSIAN: gaussian_A_theta_EBeam,
    BeamPatternType.AIRY: airy_disk_pattern,
    BeamPatternType.COSINE: cosine_tapered_pattern,
    BeamPatternType.EXPONENTIAL: exponential_tapered_pattern,
}
"""
dict : Mapping from pattern type identifiers to pattern functions.

**Warning**: Different patterns have different function signatures.
Use :func:`calculate_beam_pattern` for a unified interface, or check
the individual function docstrings for required parameters.
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
        Beam pattern calculation function. **Note**: Different patterns
        have different signatures - see individual function docstrings.

    Raises
    ------
    ValueError
        If ``pattern_type`` is not recognized.

    Examples
    --------
    >>> gauss_func = get_beam_pattern_function(BeamPatternType.GAUSSIAN)
    >>> response = gauss_func(theta, theta_HPBW=0.1)

    >>> airy_func = get_beam_pattern_function("airy")
    >>> response = airy_func(theta, wavelength=2.0, diameter=14.0)

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
        - ``"airy"``: Airy disk diffraction pattern
        - ``"cosine"``: Cosine-tapered pattern
        - ``"exponential"``: Exponential taper pattern

    theta : float or array_like
        Off-axis angle(s) from pointing center in radians.
    **kwargs : dict
        Additional parameters specific to each pattern type:

        - **gaussian**: ``theta_HPBW`` (required)
        - **airy**: ``wavelength``, ``diameter`` (both required)
        - **cosine**: ``theta_HPBW`` (required), ``taper_exponent`` (optional)
        - **exponential**: ``theta_HPBW`` (required), ``taper_dB`` (optional)

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

    >>> # Airy disk pattern
    >>> response = calculate_beam_pattern("airy", theta, wavelength=2.0, diameter=14.0)

    >>> # Cosine pattern with custom exponent
    >>> response = calculate_beam_pattern(
    ...     "cosine", theta, theta_HPBW=0.1, taper_exponent=2.0
    ... )

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


def short_dipole_jones(zenith_angle, azimuth):
    """Short dipole antenna Jones matrix (full 2x2, non-diagonal).

    Computes the Jones matrix for a short (Hertzian) dipole antenna
    aligned along the x-axis. This is the simplest polarized beam model
    with genuine cross-polarization coupling.

    The Jones matrix maps sky basis vectors (theta-hat, phi-hat) to
    feed responses::

        J = [[cos(az) * cos(za), -sin(az)], [sin(az) * cos(za), cos(az)]]

    Parameters
    ----------
    zenith_angle : float or ndarray
        Zenith angle(s) in radians.
    azimuth : float or ndarray
        Azimuth angle(s) in radians.

    Returns
    -------
    jones : ndarray
        Complex Jones matrices.
        Shape (2, 2) for scalar inputs, (N, 2, 2) for array inputs.

    References
    ----------
    .. [1] Balanis, "Antenna Theory" (4th ed.), Chapter 4: Linear Wire Antennas
    """
    za = np.asarray(zenith_angle, dtype=np.float64)
    az = np.asarray(azimuth, dtype=np.float64)
    input_is_scalar = za.ndim == 0

    za = np.atleast_1d(za)
    az = np.atleast_1d(az)

    cos_za = np.cos(za)
    cos_az = np.cos(az)
    sin_az = np.sin(az)

    n = za.shape[0]
    jones = np.zeros((n, 2, 2), dtype=np.complex128)
    jones[:, 0, 0] = cos_az * cos_za
    jones[:, 0, 1] = -sin_az
    jones[:, 1, 0] = sin_az * cos_za
    jones[:, 1, 1] = cos_az

    return jones.squeeze() if input_is_scalar else jones
