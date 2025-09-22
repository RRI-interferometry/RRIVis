# beams.py

import numpy as np
from math import pi

# Speed of light in m/s
_C = 299_792_458.0
import healpy as hp


def calculate_hpbw_radians(frequencies_hz, dish_diameter=14.0, fixed_hpbw_radians=None):
    """
    Calculate the Half Power Beam Width (HPBW) in radians for an array of frequencies.

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


# --- HPBW models (radians) for different antenna types and illumination tapers ---
# The formulas below provide common approximations of HPBW ≈ k * (lambda / D) in radians.
# The coefficient k depends on aperture type and illumination. These are pragmatic defaults
# and can be refined with measured/analytic models as needed.


def _lambda_over_D(frequencies_hz, dish_diameter, k):
    f = np.atleast_1d(frequencies_hz)
    return k * ((_C / f) / dish_diameter)


# Parabolic dish
def hpbw_parabolic_uniform(frequencies_hz, dish_diameter):
    # Uniform illumination, approximate HPBW ~ 1.02 * lambda / D (radians)
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.02)


def hpbw_parabolic_uniform_deg(frequencies_hz, dish_diameter):
    return np.degrees(hpbw_parabolic_uniform(frequencies_hz, dish_diameter))


def hpbw_parabolic_cosine(frequencies_hz, dish_diameter):
    # Cosine taper often broadens main lobe slightly; use ~1.10
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.10)


def hpbw_parabolic_cosine_deg(frequencies_hz, dish_diameter):
    return np.degrees(hpbw_parabolic_cosine(frequencies_hz, dish_diameter))


def hpbw_parabolic_gaussian(frequencies_hz, dish_diameter):
    # Gaussian taper broadening; use ~1.18 as a typical value
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.18)


def hpbw_parabolic_gaussian_deg(frequencies_hz, dish_diameter):
    return np.degrees(hpbw_parabolic_gaussian(frequencies_hz, dish_diameter))


def hpbw_parabolic_taper10db(frequencies_hz, dish_diameter):
    # 10 dB taper example; broaden relative to uniform
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.15)


def hpbw_parabolic_taper10db_deg(frequencies_hz, dish_diameter):
    return np.degrees(hpbw_parabolic_taper10db(frequencies_hz, dish_diameter))


def hpbw_parabolic_taper20db(frequencies_hz, dish_diameter):
    # 20 dB taper example; broader still
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.25)


def hpbw_parabolic_taper20db_deg(frequencies_hz, dish_diameter):
    return np.degrees(hpbw_parabolic_taper20db(frequencies_hz, dish_diameter))


# Spherical reflector (use parabolic approximations as baseline unless refined)
def hpbw_spherical_uniform(frequencies_hz, dish_diameter):
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.05)


def hpbw_spherical_uniform_deg(frequencies_hz, dish_diameter):
    return np.degrees(hpbw_spherical_uniform(frequencies_hz, dish_diameter))


def hpbw_spherical_gaussian(frequencies_hz, dish_diameter):
    return _lambda_over_D(frequencies_hz, dish_diameter, 1.20)


def hpbw_spherical_gaussian_deg(frequencies_hz, dish_diameter):
    return np.degrees(hpbw_spherical_gaussian(frequencies_hz, dish_diameter))


# Phased-array (tile) – effective diameter may differ; keep lambda/D form with adjustable k
def hpbw_phased_array_uniform(frequencies_hz, diameter_effective):
    return _lambda_over_D(frequencies_hz, diameter_effective, 1.10)


def hpbw_phased_array_uniform_deg(frequencies_hz, diameter_effective):
    return np.degrees(hpbw_phased_array_uniform(frequencies_hz, diameter_effective))


# Dipole (broad beam); for completeness, return a large, nearly constant HPBW using pattern model
def hpbw_dipole_broad(frequencies_hz, _unused):
    # Ideal short dipole HPBW ~ 90 deg (pi/2 rad). Return constant in radians.
    return np.full_like(np.atleast_1d(frequencies_hz), (pi / 2), dtype=float)


def hpbw_dipole_broad_deg(frequencies_hz, _unused):
    return np.full_like(np.atleast_1d(frequencies_hz), 90.0, dtype=float)


def analytic_A_theta():
    """
    Placeholder for a custom analytic beam pattern.
    """
    return None


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
