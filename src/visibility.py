# src/visibility.py

import numpy as np
from astropy.coordinates import AltAz, SkyCoord
import astropy.units as au
from beams import gaussian_A_theta_EBeam, convert_angle_for_display
import healpy as hp

### Visibility Calculation Function ###


def calculate_visibility(
    antennas,
    baselines,
    sources,
    spectral_indices,
    location,
    obstime,
    wavelengths,
    freqs,
    hpbw_per_antenna,
    nside,
):
    """
    Optimized calculation of complex visibility for each baseline at different frequencies.

    Parameters:
    antennas (dict): Dictionary of antenna positions.
    baselines (dict): Dictionary of baselines between antennas.
    sources (list): List of source dictionaries containing 'coords', 'flux', and 'spectral_index'.
    location (EarthLocation): Observer's geographical location.
    obstime (Time): Observation time.
    wavelengths (Quantity): Wavelength array corresponding to the frequencies.
    freqs (ndarray): Frequency array in Hz.
    hpbw_per_antenna (dict): Maps antenna number -> array of HPBW values in radians per frequency.

    Returns:
    dict: Dictionary of visibilities for each baseline as arrays over frequencies.
    """
    # Initialize visibilities dictionary with zeros for each baseline and frequency
    visibilities = {
        key: np.zeros(len(wavelengths), dtype=complex) for key in baselines.keys()
    }

    # Handle the case where there are no sources
    if not sources:
        return visibilities

    # Convert source list to arrays for vectorized operations
    source_coords = SkyCoord([s["coords"] for s in sources])
    source_fluxes = np.array([s["flux"] for s in sources])
    source_spectral_indices = np.array([s["spectral_index"] for s in sources])

    # Reference frequency for spectral index calculation is 76 MHz
    reference_freq = 76e6  # in Hz

    # Transform source coordinates to AltAz frame at the observation time and location
    altaz = source_coords.transform_to(AltAz(obstime=obstime, location=location))
    az = altaz.az.rad  # Azimuth in radians
    alt = altaz.alt.rad  # Altitude in radians

    # Filter out sources below the horizon (altitude <= 0)
    above_horizon = alt > 0
    az = az[above_horizon]
    alt = alt[above_horizon]
    source_fluxes = source_fluxes[above_horizon]
    source_spectral_indices = source_spectral_indices[above_horizon]

    # Calculate zenith angle theta
    theta = np.pi / 2 - alt
    # Direction cosines for the sources
    l = np.cos(alt) * np.sin(az)
    m = np.cos(alt) * np.cos(az)
    n = np.sin(alt)

    # Loop over each frequency and wavelength
    for i, (wavelength, freq) in enumerate(zip(wavelengths, freqs)):
        # Adjust fluxes by the spectral indices at the current frequency
        flux_adj = source_fluxes * (freq / reference_freq) ** source_spectral_indices

        # Loop over each baseline
        for (ant1, ant2), baseline in baselines.items():
            # Get the HPBW for this frequency for both antennas (already in radians)
            hpbw_ant1_rad = hpbw_per_antenna[ant1][i]
            hpbw_ant2_rad = hpbw_per_antenna[ant2][i]

            # Calculate beam patterns for both antennas
            A_theta_ant1 = gaussian_A_theta_EBeam(theta, hpbw_ant1_rad)
            A_theta_ant2 = gaussian_A_theta_EBeam(theta, hpbw_ant2_rad)

            # For interferometry, multiply the beam patterns
            A_theta_combined = A_theta_ant1 * A_theta_ant2

            # Convert baseline vector to units of wavelength
            u, v, w = np.array(baseline["BaselineVector"]) / wavelength.value

            # Projected baseline component in the direction of the sources
            b_dot_s = u * l + v * m + w * n  # Array over sources

            # Compute the phase term e^{-2πi * (b ⋅ s)}
            phase = np.exp(-2j * np.pi * b_dot_s)

            # Sum over all sources to get the visibility for this baseline and frequency
            visibility = np.sum(flux_adj * A_theta_combined * phase)

            # Store the visibility
            visibilities[(ant1, ant2)][i] = visibility
    # # Calculate total memory usage
    # total_memory_bytes = sys.getsizeof(visibilities) + sum(
    #     sys.getsizeof(key) + sys.getsizeof(value) + value.nbytes for key, value in visibilities.items()
    # )
    # total_memory_mb = total_memory_bytes / (1024 * 1024)
    # print(f"Total memory used by visibilities: {total_memory_mb:.4f} MB")

    return visibilities


### Common Function to Calculate Modulus and Phase ###


def calculate_modulus_phase(visibilities):
    """
    Calculate the modulus (amplitude) and phase of visibilities.

    Parameters:
    visibilities (dict): Dictionary of visibilities for each baseline.

    Returns:
    tuple: Two dictionaries,
           - moduli: Dictionary of amplitudes of visibilities for each baseline.
           - phases: Dictionary of phases (in radians) of visibilities for each baseline.
    """
    moduli = {key: np.abs(val) for key, val in visibilities.items()}
    phases = {key: np.angle(val) for key, val in visibilities.items()}
    return moduli, phases


def calculate_visibility_with_healpix_and_alpha(
    antennas,
    baselines,
    sources,
    spectral_indices,
    location,
    obstime,
    wavelengths,
    freqs,
    theta_HPBW,
    nside,
):
    """
    Calculate complex visibility using HEALPix map with spectral indices.

    Parameters:
    Same as before, but with additional:
    healpix_alpha_map (ndarray): HEALPix map of spectral indices.
    ref_freq (float): Reference frequency in Hz.

    Returns:
    dict: Dictionary of visibilities for each baseline as arrays over frequencies.
    """
    visibilities = {
        key: np.zeros(len(wavelengths), dtype=complex) for key in baselines.keys()
    }

    # Get HEALPix pixel centers in theta, phi
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    ref_freq = 76e6  # in Hz
    # Convert to RA, Dec
    ra = np.degrees(phi)
    dec = 90 - np.degrees(theta)
    source_coords = SkyCoord(ra=ra * au.deg, dec=dec * au.deg, frame="icrs")

    # Transform source coordinates to AltAz at observation time
    altaz = source_coords.transform_to(AltAz(obstime=obstime, location=location))
    az = altaz.az.rad
    alt = altaz.alt.rad

    # Filter pixels above the horizon
    above_horizon = alt > 0
    az = az[above_horizon]
    alt = alt[above_horizon]
    fluxes = sources[above_horizon]
    alphas = spectral_indices[above_horizon]

    # Extrapolate fluxes to the current frequency
    extrapolated_fluxes = np.array(
        [
            flux * (freq / ref_freq) ** alpha
            for freq in freqs
            for flux, alpha in zip(fluxes, alphas)
        ]
    ).reshape(
        len(freqs), -1
    )  # Reshape to (freqs, sources)

    # Calculate direction cosines
    theta = np.pi / 2 - alt
    l = np.cos(alt) * np.sin(az)
    m = np.cos(alt) * np.cos(az)
    n = np.sin(alt)

    # Gaussian primary beam pattern
    A_theta = gaussian_A_theta_EBeam(theta, theta_HPBW)

    # Loop over each frequency
    for i, (wavelength, freq) in enumerate(zip(wavelengths, freqs)):
        for (ant1, ant2), baseline in baselines.items():
            u, v, w = np.array(baseline) / wavelength.value
            b_dot_s = u * l + v * m + w * n
            phase = np.exp(-2j * np.pi * b_dot_s)

            visibility = np.sum(extrapolated_fluxes[i] * A_theta * phase)
            visibilities[(ant1, ant2)][i] = visibility

    return visibilities


def convert_phase_for_display(phase_radians, angle_unit):
    """Convert phase from radians to the user's preferred unit for display.

    Args:
        phase_radians: Phase value in radians
        angle_unit: Target unit ('degrees' or 'radians')

    Returns:
        Phase value in the specified unit
    """
    return convert_angle_for_display(phase_radians, angle_unit)
