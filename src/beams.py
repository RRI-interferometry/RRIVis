# beams.py
"""
Beam pattern and HPBW calculation functions for various antenna types.

This module provides comprehensive support for different antenna types used in
radio astronomy, including parabolic dishes, spherical reflectors, phased arrays,
and dipoles, with various illumination taper options.

References:
    - Antenna Theory: Analysis and Design (Balanis)
    - Essential Radio Astronomy (Condon & Ransom)
    - Interferometry and Synthesis in Radio Astronomy (Thompson, Moran, Swenson)
"""

import numpy as np
from math import pi
from scipy.special import j1  # Bessel function for Airy pattern

# Speed of light in m/s
_C = 299_792_458.0
import healpy as hp


# =============================================================================
# ANTENNA TYPE CONSTANTS
# =============================================================================

class AntennaType:
    """Antenna type identifiers."""
    PARABOLIC_UNIFORM = "parabolic_uniform"
    PARABOLIC_COSINE = "parabolic_cosine"
    PARABOLIC_GAUSSIAN = "parabolic_gaussian"
    PARABOLIC_10DB = "parabolic_10db_taper"
    PARABOLIC_20DB = "parabolic_20db_taper"
    SPHERICAL_UNIFORM = "spherical_uniform"
    SPHERICAL_GAUSSIAN = "spherical_gaussian"
    PHASED_ARRAY = "phased_array"
    DIPOLE_SHORT = "dipole_short"
    DIPOLE_HALFWAVE = "dipole_halfwave"
    DIPOLE_FOLDED = "dipole_folded"


# =============================================================================
# LEGACY FUNCTIONS (for backward compatibility)
# =============================================================================

def calculate_hpbw_radians(frequencies_hz, dish_diameter=14.0, fixed_hpbw_radians=None):
    """
    Calculate the Half Power Beam Width (HPBW) in radians for an array of frequencies.

    Legacy function - uses generic k=1.22 approximation.
    For antenna-type specific calculations, use the hpbw_* functions below.

    Parameters:
    - frequencies_hz (float or ndarray): Frequency or array of frequencies in Hz.
    - dish_diameter (float): Diameter of the dish in meters. Default is 14.0 m.
    - fixed_hpbw_radians (float, optional): If provided, sets the HPBW to this value (in radians) for all frequencies.

    Returns:
    - ndarray: HPBW values in radians for each frequency.
    """
    if fixed_hpbw_radians is not None:
        return np.full_like(
            np.atleast_1d(frequencies_hz), fixed_hpbw_radians, dtype=float
        )

    f = np.atleast_1d(frequencies_hz)
    wavelengths = _C / f
    # Default approximation (diffraction-limited circular aperture)
    hpbw_radians = 1.22 * (wavelengths / dish_diameter)
    return hpbw_radians


def calculate_hpbw(frequencies, dish_diameter=14.0, fixed_hpbw=None):
    """
    Calculate the Half Power Beam Width (HPBW) for an array of frequencies.

    Legacy function - uses generic k=1.22 approximation.
    For antenna-type specific calculations, use the hpbw_* functions below.

    Notes:
    - Internally computes in radians, returns in degrees as user-facing value.
    - If fixed_hpbw is provided (degrees), returns that constant for all channels.

    Parameters:
    - frequencies (float or ndarray): Frequency or array in Hz.
    - dish_diameter (float): Diameter in meters.
    - fixed_hpbw (float or None): Constant HPBW to use in degrees.

    Returns:
    - ndarray of HPBW in degrees for each frequency.
    """
    if fixed_hpbw is not None:
        return np.full_like(np.atleast_1d(frequencies), float(fixed_hpbw), dtype=float)

    f = np.atleast_1d(frequencies)
    wavelengths = _C / f
    hpbw_rad = 1.22 * (wavelengths / float(dish_diameter))
    return np.degrees(hpbw_rad)


# =============================================================================
# ANTENNA TYPE-SPECIFIC HPBW FUNCTIONS
# =============================================================================

def _lambda_over_D(frequencies_hz, diameter, k):
    """
    Helper function for HPBW calculation: k * (lambda / D).

    Parameters:
    - frequencies_hz (float or ndarray): Frequency in Hz
    - diameter (float): Antenna diameter in meters
    - k (float): Coefficient depending on antenna type and illumination

    Returns:
    - ndarray: HPBW values in radians
    """
    f = np.atleast_1d(frequencies_hz)
    wavelengths = _C / f
    return k * (wavelengths / diameter)


# --- PARABOLIC DISH ANTENNAS ---

def hpbw_parabolic_uniform(frequencies_hz, dish_diameter):
    """
    HPBW for parabolic dish with uniform illumination.

    Formula: HPBW = 1.02 * (lambda / D)

    Characteristics:
    - Highest aperture efficiency (100% illumination)
    - High sidelobes (-17.6 dB first sidelobe)
    - Not realistic for actual feeds

    Parameters:
    - frequencies_hz (float or ndarray): Frequency in Hz
    - dish_diameter (float): Dish diameter in meters

    Returns:
    - ndarray: HPBW values in radians
    """
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.02)


def hpbw_parabolic_cosine(frequencies_hz, dish_diameter):
    """
    HPBW for parabolic dish with cosine illumination taper.

    Formula: HPBW = 1.10 * (lambda / D)

    Characteristics:
    - Good compromise between efficiency and sidelobes
    - ~81% aperture efficiency
    - ~-23 dB first sidelobe

    Parameters:
    - frequencies_hz (float or ndarray): Frequency in Hz
    - dish_diameter (float): Dish diameter in meters

    Returns:
    - ndarray: HPBW values in radians
    """
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.10)


def hpbw_parabolic_gaussian(frequencies_hz, dish_diameter):
    """
    HPBW for parabolic dish with Gaussian illumination taper.

    Formula: HPBW = 1.18 * (lambda / D)

    Characteristics:
    - Very low sidelobes (-40+ dB)
    - ~75% aperture efficiency
    - Smooth rolloff

    Parameters:
    - frequencies_hz (float or ndarray): Frequency in Hz
    - dish_diameter (float): Dish diameter in meters

    Returns:
    - ndarray: HPBW values in radians
    """
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.18)


def hpbw_parabolic_10db_taper(frequencies_hz, dish_diameter):
    """
    HPBW for parabolic dish with 10 dB edge taper.

    Formula: HPBW = 1.15 * (lambda / D)

    Characteristics:
    - Standard choice for most radio telescopes
    - ~75% aperture efficiency
    - ~-30 dB first sidelobe
    - Good balance of gain and low sidelobes

    Parameters:
    - frequencies_hz (float or ndarray): Frequency in Hz
    - dish_diameter (float): Dish diameter in meters

    Returns:
    - ndarray: HPBW values in radians
    """
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.15)


def hpbw_parabolic_20db_taper(frequencies_hz, dish_diameter):
    """
    HPBW for parabolic dish with 20 dB edge taper.

    Formula: HPBW = 1.25 * (lambda / D)

    Characteristics:
    - Lowest sidelobes (-42 dB)
    - ~62% aperture efficiency
    - Used when sidelobe control is critical

    Parameters:
    - frequencies_hz (float or ndarray): Frequency in Hz
    - dish_diameter (float): Dish diameter in meters

    Returns:
    - ndarray: HPBW values in radians
    """
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.25)


# --- SPHERICAL REFLECTOR ANTENNAS (Arecibo/FAST type) ---

def hpbw_spherical_uniform(frequencies_hz, diameter_effective):
    """
    HPBW for spherical reflector with uniform illumination.

    Formula: HPBW = 1.05 * (lambda / D_eff)

    Characteristics:
    - Slightly broader than parabolic due to spherical aberration
    - D_eff is the illuminated diameter, not total sphere size

    Parameters:
    - frequencies_hz (float or ndarray): Frequency in Hz
    - diameter_effective (float): Effective illuminated diameter in meters

    Returns:
    - ndarray: HPBW values in radians
    """
    return _lambda_over_D(frequencies_hz, diameter_effective, 1.05)


def hpbw_spherical_gaussian(frequencies_hz, diameter_effective):
    """
    HPBW for spherical reflector with Gaussian illumination taper.

    Formula: HPBW = 1.20 * (lambda / D_eff)

    Characteristics:
    - More realistic for FAST-type active surface telescopes
    - Accounts for taper and spherical aberration

    Parameters:
    - frequencies_hz (float or ndarray): Frequency in Hz
    - diameter_effective (float): Effective illuminated diameter in meters

    Returns:
    - ndarray: HPBW values in radians
    """
    return _lambda_over_D(frequencies_hz, diameter_effective, 1.20)


# --- PHASED ARRAY ANTENNAS ---

def hpbw_phased_array(frequencies_hz, diameter_effective):
    """
    HPBW for phased array antenna with uniform element spacing.

    Formula: HPBW = 1.10 * (lambda / D_eff)

    Characteristics:
    - D_eff = N * d, where N is number of elements and d is spacing
    - Assumes lambda/2 element spacing
    - Electronic beamforming

    Parameters:
    - frequencies_hz (float or ndarray): Frequency in Hz
    - diameter_effective (float): Effective array aperture in meters

    Returns:
    - ndarray: HPBW values in radians
    """
    return _lambda_over_D(frequencies_hz, diameter_effective, 1.10)


# --- DIPOLE ANTENNAS ---

def hpbw_dipole_short(frequencies_hz, _unused_diameter):
    """
    HPBW for short dipole antenna (L << lambda).

    Formula: HPBW = pi/2 radians = 90 degrees (CONSTANT)

    Characteristics:
    - Frequency-independent HPBW
    - Omnidirectional in H-plane
    - Figure-8 pattern in E-plane with 90° HPBW
    - Used in: HERA, MWA, LOFAR (approximately)

    Parameters:
    - frequencies_hz (float or ndarray): Frequency in Hz (not used)
    - _unused_diameter: Ignored parameter for API consistency

    Returns:
    - ndarray: HPBW values in radians (constant pi/2)
    """
    return np.full_like(np.atleast_1d(frequencies_hz), pi / 2, dtype=float)


def hpbw_dipole_halfwave(frequencies_hz, _unused_diameter):
    """
    HPBW for half-wave dipole antenna (L = lambda/2).

    Formula: HPBW_E-plane = 78 degrees (CONSTANT in E-plane)

    Characteristics:
    - E-plane: ~78° HPBW
    - H-plane: Omnidirectional (360°)
    - Most common resonant dipole
    - Radiation resistance ~73 Ohms

    Parameters:
    - frequencies_hz (float or ndarray): Frequency in Hz
    - _unused_diameter: Ignored parameter for API consistency

    Returns:
    - ndarray: HPBW values in radians (constant 78° in E-plane)
    """
    return np.full_like(np.atleast_1d(frequencies_hz), 78.0 * pi / 180.0, dtype=float)


def hpbw_dipole_folded(frequencies_hz, _unused_diameter):
    """
    HPBW for folded dipole antenna.

    Formula: Same as half-wave dipole = 78 degrees

    Characteristics:
    - Same radiation pattern as half-wave dipole
    - Higher impedance (~280-300 Ohms vs 73 Ohms)
    - Better bandwidth than simple dipole

    Parameters:
    - frequencies_hz (float or ndarray): Frequency in Hz
    - _unused_diameter: Ignored parameter for API consistency

    Returns:
    - ndarray: HPBW values in radians (constant 78° in E-plane)
    """
    return hpbw_dipole_halfwave(frequencies_hz, _unused_diameter)


# --- DEGREE VERSIONS (for user convenience) ---

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
    """HPBW for parabolic 10dB edge taper, in degrees."""
    return np.degrees(hpbw_parabolic_10db_taper(frequencies_hz, dish_diameter))


def hpbw_parabolic_20db_taper_deg(frequencies_hz, dish_diameter):
    """HPBW for parabolic 20dB edge taper, in degrees."""
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
    """HPBW for short dipole, in degrees."""
    return np.degrees(hpbw_dipole_short(frequencies_hz, _unused_diameter))


def hpbw_dipole_halfwave_deg(frequencies_hz, _unused_diameter):
    """HPBW for half-wave dipole, in degrees."""
    return np.degrees(hpbw_dipole_halfwave(frequencies_hz, _unused_diameter))


def hpbw_dipole_folded_deg(frequencies_hz, _unused_diameter):
    """HPBW for folded dipole, in degrees."""
    return np.degrees(hpbw_dipole_folded(frequencies_hz, _unused_diameter))


# --- HPBW FUNCTION REGISTRY ---

HPBW_FUNCTIONS = {
    # Parabolic types (radians)
    AntennaType.PARABOLIC_UNIFORM: hpbw_parabolic_uniform,
    AntennaType.PARABOLIC_COSINE: hpbw_parabolic_cosine,
    AntennaType.PARABOLIC_GAUSSIAN: hpbw_parabolic_gaussian,
    AntennaType.PARABOLIC_10DB: hpbw_parabolic_10db_taper,
    AntennaType.PARABOLIC_20DB: hpbw_parabolic_20db_taper,
    # Spherical types (radians)
    AntennaType.SPHERICAL_UNIFORM: hpbw_spherical_uniform,
    AntennaType.SPHERICAL_GAUSSIAN: hpbw_spherical_gaussian,
    # Phased array (radians)
    AntennaType.PHASED_ARRAY: hpbw_phased_array,
    # Dipole types (radians)
    AntennaType.DIPOLE_SHORT: hpbw_dipole_short,
    AntennaType.DIPOLE_HALFWAVE: hpbw_dipole_halfwave,
    AntennaType.DIPOLE_FOLDED: hpbw_dipole_folded,
}


def get_hpbw_function(antenna_type):
    """
    Get the HPBW calculation function for a given antenna type.

    Parameters:
    - antenna_type (str): Antenna type identifier

    Returns:
    - function: HPBW calculation function (returns radians)

    Raises:
    - ValueError: If antenna type is not recognized
    """
    if antenna_type not in HPBW_FUNCTIONS:
        available = ', '.join(HPBW_FUNCTIONS.keys())
        raise ValueError(
            f"Unknown antenna type: '{antenna_type}'. "
            f"Available types: {available}"
        )
    return HPBW_FUNCTIONS[antenna_type]


def calculate_hpbw_for_antenna_type(antenna_type, frequencies_hz, diameter):
    """
    Calculate HPBW for a specific antenna type.

    Parameters:
    - antenna_type (str): Antenna type identifier (from AntennaType class)
    - frequencies_hz (float or ndarray): Frequency or array of frequencies in Hz
    - diameter (float): Antenna diameter/effective aperture in meters

    Returns:
    - ndarray: HPBW values in radians for each frequency

    Example:
    >>> from beams import AntennaType, calculate_hpbw_for_antenna_type
    >>> freqs = np.array([100e6, 150e6, 200e6])
    >>> hpbw = calculate_hpbw_for_antenna_type(
    ...     AntennaType.PARABOLIC_10DB, freqs, 14.0
    ... )
    """
    hpbw_func = get_hpbw_function(antenna_type)
    return hpbw_func(frequencies_hz, diameter)


# =============================================================================
# BEAM PATTERN MODELS
# =============================================================================

def airy_disk_pattern(theta, wavelength, diameter):
    """
    Calculate Airy disk pattern for uniform circular aperture.

    This is the classical diffraction pattern for a uniformly illuminated
    circular aperture (parabolic dish with uniform feed illumination).

    Formula: A(theta) = [2*J1(x)/x]^2
    where x = pi * D * sin(theta) / lambda
    and J1 is the first-order Bessel function of the first kind.

    Characteristics:
    - First null at theta = 1.22 * lambda / D
    - First sidelobe at -17.6 dB
    - More accurate than Gaussian for uniform illumination

    Parameters:
    - theta (float or ndarray): Off-axis angle in radians
    - wavelength (float): Wavelength in meters
    - diameter (float): Antenna diameter in meters

    Returns:
    - ndarray: Normalized beam response A(theta) (0 to 1)

    References:
    - Born & Wolf, "Principles of Optics"
    - Balanis, "Antenna Theory"
    """
    theta = np.atleast_1d(theta)
    x = np.pi * diameter * np.sin(theta) / wavelength

    # Handle x=0 case (on-axis) where J1(x)/x -> 0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            np.abs(x) < 1e-10,
            1.0,  # On-axis value
            (2 * j1(x) / x) ** 2  # Off-axis Airy pattern
        )

    return result


def cosine_tapered_pattern(theta, theta_HPBW, taper_exponent=1.0):
    """
    Calculate cosine-tapered beam pattern.

    Formula: A(theta) = cos^n(theta / theta_edge)
    where theta_edge is the edge of the beam and n is the taper exponent.

    This models feeds with cosine-on-a-pedestal illumination, common in
    Cassegrain and offset-fed reflectors.

    Characteristics:
    - taper_exponent = 1.0: cosine taper (~81% efficiency, -23 dB sidelobes)
    - taper_exponent = 2.0: cosine-squared taper (~67% efficiency, -32 dB sidelobes)
    - taper_exponent = 3.0: strong taper (~55% efficiency, very low sidelobes)

    Parameters:
    - theta (float or ndarray): Off-axis angle in radians
    - theta_HPBW (float): Half Power Beam Width in radians
    - taper_exponent (float): Cosine exponent (default 1.0)

    Returns:
    - ndarray: Normalized beam response A(theta) (0 to 1)
    """
    theta = np.atleast_1d(theta)
    # Convert HPBW to edge angle (approximate for cosine pattern)
    # For cosine taper, theta_edge ≈ 1.4 * theta_HPBW
    theta_edge = 1.4 * theta_HPBW

    # Cosine taper with cutoff at edge
    result = np.where(
        np.abs(theta) < theta_edge,
        np.cos(theta / theta_edge) ** taper_exponent,
        0.0  # Zero outside edge
    )

    return result


def exponential_tapered_pattern(theta, theta_HPBW, taper_dB=10.0):
    """
    Calculate exponential-tapered beam pattern (Gaussian-like).

    Formula: A(theta) = exp(-alpha * theta^2)
    where alpha is chosen to match the specified edge taper in dB.

    This approximates feed horns and Gaussian illumination patterns.

    Parameters:
    - theta (float or ndarray): Off-axis angle in radians
    - theta_HPBW (float): Half Power Beam Width in radians
    - taper_dB (float): Edge taper in dB at theta_edge (default 10 dB)

    Returns:
    - ndarray: Normalized beam response A(theta) (0 to 1)
    """
    theta = np.atleast_1d(theta)
    # Alpha chosen to give -3 dB at HPBW
    alpha = np.log(2) / theta_HPBW**2
    result = np.exp(-alpha * theta**2)

    return result


def gaussian_A_theta_EBeam(theta, theta_HPBW):
    """
    Calculate the Gaussian primary beam pattern A(theta).

    Parameters:
    theta (ndarray): Zenith angles in radians.
    theta_HPBW (float): Half Power Beam Width (HPBW) in radians.

    Returns:
    ndarray: The primary beam pattern A(theta) evaluated at theta.
    """
    return np.exp(-((theta / (np.sqrt(2) * theta_HPBW)) ** 2))


def calculate_gaussian_beam_area_EBeam(nside, theta_HPBW):
    """
    Calculate the beam solid angle (beam area) by summing the Gaussian beam response over all HEALPix pixels.

    Parameters:
    nside (int): The nside parameter defining the resolution of the HEALPix map.
    theta_HPBW (float or ndarray): Half Power Beam Width (HPBW) in radians. Can be a single value or an array.

    Returns:
    float or list: Beam solid angle in steradians. Returns a single value if all HPBW values are identical,
    otherwise returns a list of beam solid angles for each unique HPBW.
    """

    pixel_area = hp.nside2pixarea(nside)  # Solid angle of each pixel in steradians
    npix = hp.nside2npix(nside)  # Total number of pixels in HEALPix map
    theta, phi = hp.pix2ang(
        nside, np.arange(npix)
    )  # Get angular coordinates (theta, phi)

    # Handle single or identical HPBW case
    if np.ndim(theta_HPBW) > 0 and np.all(theta_HPBW == theta_HPBW[0]):
        beam_response = gaussian_A_theta_EBeam(theta, theta_HPBW[0])
        return np.sum(beam_response * pixel_area)  # Single beam solid angle
    else:
        # Array of unique HPBW values
        beam_areas = []  # Initialize list to store beam areas
        for hpbw in theta_HPBW:
            beam_response = gaussian_A_theta_EBeam(
                theta, hpbw
            )  # Calculate beam response for this HPBW
            beam_area = np.sum(beam_response * pixel_area)  # Compute beam solid angle
            beam_areas.append(beam_area)  # Append to the result list

        return beam_areas


def calculate_airy_beam_area(nside, wavelength, diameter):
    """
    Calculate the beam solid angle for Airy disk pattern.

    Parameters:
    - nside (int): HEALPix nside parameter
    - wavelength (float): Wavelength in meters
    - diameter (float): Antenna diameter in meters

    Returns:
    - float: Beam solid angle in steradians
    """
    pixel_area = hp.nside2pixarea(nside)
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    beam_response = airy_disk_pattern(theta, wavelength, diameter)
    return np.sum(beam_response * pixel_area)


def calculate_cosine_beam_area(nside, theta_HPBW, taper_exponent=1.0):
    """
    Calculate the beam solid angle for cosine-tapered pattern.

    Parameters:
    - nside (int): HEALPix nside parameter
    - theta_HPBW (float): Half Power Beam Width in radians
    - taper_exponent (float): Cosine exponent (default 1.0)

    Returns:
    - float: Beam solid angle in steradians
    """
    pixel_area = hp.nside2pixarea(nside)
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    beam_response = cosine_tapered_pattern(theta, theta_HPBW, taper_exponent)
    return np.sum(beam_response * pixel_area)


def calculate_exponential_beam_area(nside, theta_HPBW, taper_dB=10.0):
    """
    Calculate the beam solid angle for exponential-tapered pattern.

    Parameters:
    - nside (int): HEALPix nside parameter
    - theta_HPBW (float): Half Power Beam Width in radians
    - taper_dB (float): Edge taper in dB (default 10 dB)

    Returns:
    - float: Beam solid angle in steradians
    """
    pixel_area = hp.nside2pixarea(nside)
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    beam_response = exponential_tapered_pattern(theta, theta_HPBW, taper_dB)
    return np.sum(beam_response * pixel_area)


# =============================================================================
# BEAM PATTERN TYPE REGISTRY
# =============================================================================

class BeamPatternType:
    """Beam pattern model identifiers."""
    GAUSSIAN = "gaussian"
    AIRY = "airy"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"


BEAM_PATTERN_FUNCTIONS = {
    BeamPatternType.GAUSSIAN: gaussian_A_theta_EBeam,
    BeamPatternType.AIRY: airy_disk_pattern,
    BeamPatternType.COSINE: cosine_tapered_pattern,
    BeamPatternType.EXPONENTIAL: exponential_tapered_pattern,
}


def get_beam_pattern_function(pattern_type):
    """
    Get the beam pattern calculation function for a given pattern type.

    Parameters:
    - pattern_type (str): Beam pattern type identifier

    Returns:
    - function: Beam pattern calculation function

    Raises:
    - ValueError: If pattern type is not recognized
    """
    if pattern_type not in BEAM_PATTERN_FUNCTIONS:
        available = ', '.join(BEAM_PATTERN_FUNCTIONS.keys())
        raise ValueError(
            f"Unknown beam pattern type: '{pattern_type}'. "
            f"Available types: {available}"
        )
    return BEAM_PATTERN_FUNCTIONS[pattern_type]


def calculate_beam_pattern(pattern_type, theta, **kwargs):
    """
    Calculate beam pattern for a specific pattern type.

    Parameters:
    - pattern_type (str): Beam pattern type (from BeamPatternType class)
    - theta (float or ndarray): Off-axis angle in radians
    - **kwargs: Additional parameters specific to each pattern type
        - For 'gaussian': theta_HPBW
        - For 'airy': wavelength, diameter
        - For 'cosine': theta_HPBW, taper_exponent (optional)
        - For 'exponential': theta_HPBW, taper_dB (optional)

    Returns:
    - ndarray: Beam response values (0 to 1)

    Example:
    >>> theta = np.linspace(0, 0.1, 100)
    >>> # Gaussian pattern
    >>> pattern = calculate_beam_pattern('gaussian', theta, theta_HPBW=0.02)
    >>> # Airy pattern
    >>> pattern = calculate_beam_pattern('airy', theta, wavelength=2.0, diameter=14.0)
    """
    pattern_func = get_beam_pattern_function(pattern_type)
    return pattern_func(theta, **kwargs)


def convert_angle_for_display(angle_radians, angle_unit):
    """Convert angle from radians to the user's preferred unit.

    Args:
        angle_radians: Angle value in radians
        angle_unit: Target unit ('degrees' or 'radians')

    Returns:
        Angle value in the specified unit
    """
    if angle_unit == "degrees":
        return np.degrees(angle_radians)
    return angle_radians  # Already in radians
