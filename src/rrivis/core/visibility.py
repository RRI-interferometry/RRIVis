# rrivis/core/visibility.py
"""
Visibility calculation using the Radio Interferometer Measurement Equation (RIME).

Implements full polarization with 2×2 Jones matrices and coherency matrices.
Supports both beam FITS files and analytic beam patterns.
"""

import numpy as np
from astropy.coordinates import AltAz, SkyCoord
import astropy.units as au
import logging

# Import polarization utilities
from rrivis.core.polarization import (
    stokes_to_coherency,
    apply_jones_matrices,
    visibility_to_correlations
)

# Import analytic beam patterns for fallback
from rrivis.core.beams import (
    gaussian_A_theta_EBeam,
    airy_disk_pattern,
    cosine_tapered_pattern,
    exponential_tapered_pattern,
)


logger = logging.getLogger(__name__)


def calculate_visibility(
    antennas,
    baselines,
    sources,
    location,
    obstime,
    wavelengths,
    freqs,
    hpbw_per_antenna,
    beam_manager=None,
    beam_pattern_per_antenna=None,
    beam_pattern_params=None,
    return_correlations=True,
):
    """
    Calculate complex visibility using full polarization (RIME).

    Implements: V_ij = Σ_sources E_i(θ,φ) @ C_source @ E_j^H(θ,φ) @ exp[-2πi(uvw·lmn)]

    Parameters:
    -----------
    antennas : dict
        Dictionary of antenna positions and properties.
        Keys: antenna numbers, Values: dicts with "Name", "Position", etc.
    baselines : dict
        Dictionary of baselines between antennas.
        Keys: (ant1, ant2) tuples, Values: dicts with "BaselineVector"
    sources : list
        List of source dicts with 'coords', 'flux', 'spectral_index',
        'stokes_q', 'stokes_u', 'stokes_v'.
    location : EarthLocation
        Observer's geographical location.
    obstime : Time
        Observation time.
    wavelengths : Quantity
        Wavelength array corresponding to frequencies (with units).
    freqs : ndarray
        Frequency array in Hz.
    hpbw_per_antenna : dict
        Maps antenna number -> array of HPBW values in radians per frequency.
        Used for analytic beam fallback.
    beam_manager : BeamManager, optional
        BeamManager instance for getting Jones matrices from beam FITS files.
        If None or returns None, falls back to analytic beams.
    beam_pattern_per_antenna : dict, optional
        Maps antenna number -> beam pattern type string for analytic beams.
        Supported: 'gaussian', 'airy', 'cosine', 'exponential'.
        Defaults to 'gaussian'.
    beam_pattern_params : dict, optional
        Additional parameters for analytic beam patterns.
    return_correlations : bool, optional
        If True, extract and return correlation products (XX, XY, YX, YY, I).
        If False, return raw 2×2 visibility matrices.

    Returns:
    --------
    dict: Dictionary of visibilities for each baseline.
        If return_correlations=True:
            Keys: (ant1, ant2) tuples
            Values: dict with keys "XX", "XY", "YX", "YY", "I", each an array over frequencies
        If return_correlations=False:
            Keys: (ant1, ant2) tuples
            Values: ndarray of shape (N_freq, 2, 2) - visibility matrices
    """
    # Set defaults for beam pattern configuration
    if beam_pattern_per_antenna is None:
        beam_pattern_per_antenna = {}
    if beam_pattern_params is None:
        beam_pattern_params = {
            'cosine_taper_exponent': 1.0,
            'exponential_taper_dB': 10.0,
        }

    # Reference frequency for spectral index calculation (76 MHz)
    reference_freq = 76e6  # Hz

    # Initialize visibilities dictionary
    # Each baseline gets a (N_freq, 2, 2) array for visibility matrices
    n_freq = len(wavelengths)
    visibilities_matrices = {
        key: np.zeros((n_freq, 2, 2), dtype=complex) for key in baselines.keys()
    }

    # Handle empty source list
    if not sources:
        if return_correlations:
            return {key: _extract_correlations(val) for key, val in visibilities_matrices.items()}
        return visibilities_matrices

    # Convert source list to arrays for vectorized operations
    source_coords = SkyCoord([s["coords"] for s in sources])
    source_stokes_I = np.array([s["flux"] for s in sources])
    source_stokes_Q = np.array([s.get("stokes_q", 0.0) for s in sources])
    source_stokes_U = np.array([s.get("stokes_u", 0.0) for s in sources])
    source_stokes_V = np.array([s.get("stokes_v", 0.0) for s in sources])
    source_spectral_indices = np.array([s["spectral_index"] for s in sources])

    # Transform source coordinates to AltAz frame
    altaz = source_coords.transform_to(AltAz(obstime=obstime, location=location))
    az_rad = altaz.az.rad
    alt_rad = altaz.alt.rad

    # Filter out sources below the horizon
    above_horizon = alt_rad > 0
    if not np.any(above_horizon):
        logger.info("No sources above horizon")
        if return_correlations:
            return {key: _extract_correlations(val) for key, val in visibilities_matrices.items()}
        return visibilities_matrices

    # Apply horizon filter
    az_rad = az_rad[above_horizon]
    alt_rad = alt_rad[above_horizon]
    source_stokes_I = source_stokes_I[above_horizon]
    source_stokes_Q = source_stokes_Q[above_horizon]
    source_stokes_U = source_stokes_U[above_horizon]
    source_stokes_V = source_stokes_V[above_horizon]
    source_spectral_indices = source_spectral_indices[above_horizon]

    n_sources = len(alt_rad)
    logger.info(f"Computing visibilities for {n_sources} sources above horizon")

    # Calculate direction cosines (l, m, n)
    # l = sin(θ)sin(φ), m = sin(θ)cos(φ), n = cos(θ) where θ=zenith angle, φ=azimuth
    zenith_angle = np.pi / 2 - alt_rad
    l = np.cos(alt_rad) * np.sin(az_rad)
    m = np.cos(alt_rad) * np.cos(az_rad)
    n = np.sin(alt_rad)

    # Loop over each frequency
    for freq_idx, (wavelength, freq) in enumerate(zip(wavelengths, freqs)):
        # Scale source fluxes by spectral index
        I_scaled = source_stokes_I * (freq / reference_freq) ** source_spectral_indices
        Q_scaled = source_stokes_Q * (freq / reference_freq) ** source_spectral_indices
        U_scaled = source_stokes_U * (freq / reference_freq) ** source_spectral_indices
        V_scaled = source_stokes_V * (freq / reference_freq) ** source_spectral_indices

        # Convert Stokes parameters to coherency matrices for all sources
        # Shape: (n_sources, 2, 2)
        coherency_matrices = np.array([
            stokes_to_coherency(I, Q, U, V)
            for I, Q, U, V in zip(I_scaled, Q_scaled, U_scaled, V_scaled)
        ])

        # Loop over each baseline
        for (ant1, ant2), baseline in baselines.items():
            # Get antenna names
            ant1_name = antennas[ant1]["Name"]
            ant2_name = antennas[ant2]["Name"]

            # Get Jones matrices for both antennas
            # Try beam FITS first, then fall back to analytic
            E1_all = _get_jones_matrices_for_sources(
                beam_manager, ant1, ant1_name, alt_rad, az_rad, freq,
                hpbw_per_antenna, beam_pattern_per_antenna, beam_pattern_params,
                antennas, wavelength.value, freq_idx
            )  # Shape: (n_sources, 2, 2)

            E2_all = _get_jones_matrices_for_sources(
                beam_manager, ant2, ant2_name, alt_rad, az_rad, freq,
                hpbw_per_antenna, beam_pattern_per_antenna, beam_pattern_params,
                antennas, wavelength.value, freq_idx
            )  # Shape: (n_sources, 2, 2)

            # Calculate geometric phase term for all sources
            # Convert baseline to wavelength units
            u, v, w = np.array(baseline["BaselineVector"]) / wavelength.value

            # Projected baseline component: b·s = u*l + v*m + w*n
            b_dot_s = u * l + v * m + w * n  # Shape: (n_sources,)

            # Phase term: exp(-2πi * b·s)
            phase = np.exp(-2j * np.pi * b_dot_s)  # Shape: (n_sources,)

            # Apply RIME for each source and sum
            # V_ij = Σ_sources E_i @ C @ E_j^H @ exp(-2πi * b·s)
            visibility_matrix = np.zeros((2, 2), dtype=complex)

            for src_idx in range(n_sources):
                E_i = E1_all[src_idx]  # (2, 2)
                C = coherency_matrices[src_idx]  # (2, 2)
                E_j = E2_all[src_idx]  # (2, 2)
                phase_factor = phase[src_idx]  # scalar

                # RIME: V = E_i @ C @ E_j^H
                V_src = apply_jones_matrices(E_i, C, E_j)  # (2, 2)

                # Apply geometric phase and accumulate
                visibility_matrix += V_src * phase_factor

            # Store the visibility matrix for this baseline and frequency
            visibilities_matrices[(ant1, ant2)][freq_idx] = visibility_matrix

    # Convert to correlation products if requested
    if return_correlations:
        visibilities_correlations = {}
        for baseline_key, vis_matrix_array in visibilities_matrices.items():
            visibilities_correlations[baseline_key] = _extract_correlations(vis_matrix_array)
        return visibilities_correlations

    return visibilities_matrices


def _get_jones_matrices_for_sources(
    beam_manager, ant_num, ant_name, alt_rad, az_rad, freq,
    hpbw_per_antenna, beam_pattern_per_antenna, beam_pattern_params,
    antennas, wavelength_value, freq_idx
):
    """
    Get Jones matrices for all sources for a given antenna.

    Tries beam FITS first via beam_manager, falls back to analytic beam.

    Parameters:
    -----------
    beam_manager : BeamManager or None
        BeamManager instance
    ant_num : int
        Antenna number
    ant_name : str
        Antenna name
    alt_rad : ndarray
        Altitudes in radians for all sources (n_sources,)
    az_rad : ndarray
        Azimuths in radians for all sources (n_sources,)
    freq : float
        Frequency in Hz
    hpbw_per_antenna : dict
        HPBW data for analytic fallback
    beam_pattern_per_antenna : dict
        Beam pattern types for analytic fallback
    beam_pattern_params : dict
        Beam pattern parameters for analytic fallback
    antennas : dict
        Antenna metadata
    wavelength_value : float
        Wavelength in meters
    freq_idx : int
        Frequency index

    Returns:
    --------
    ndarray: Jones matrices, shape (n_sources, 2, 2)
    """
    n_sources = len(alt_rad)

    # Try beam FITS via BeamManager
    if beam_manager is not None:
        try:
            jones_matrices = beam_manager.get_jones_matrix(
                antenna_name=ant_name,
                alt_rad=alt_rad,
                az_rad=az_rad,
                freq_hz=freq,
                reuse_spline=True
            )

            if jones_matrices is not None:
                # beam_manager returns either (2, 2) for single source or (n_sources, 2, 2)
                if jones_matrices.ndim == 2:
                    # Single source, tile to match n_sources
                    jones_matrices = np.tile(jones_matrices, (n_sources, 1, 1))
                return jones_matrices

        except Exception as e:
            logger.warning(f"Beam FITS query failed for antenna {ant_name}, falling back to analytic: {e}")

    # Fall back to analytic beam (scalar amplitude, convert to diagonal Jones matrix)
    zenith_angle = np.pi / 2 - alt_rad
    hpbw_rad = hpbw_per_antenna[ant_num][freq_idx]

    # Get beam pattern type
    pattern_type = beam_pattern_per_antenna.get(ant_num, 'gaussian')

    # Calculate scalar beam amplitude for all sources
    if pattern_type == 'airy':
        diameter = antennas[ant_num].get('diameter', 12.0)  # Default 12m
        A_theta = airy_disk_pattern(zenith_angle, wavelength_value, diameter)
    elif pattern_type == 'cosine':
        taper_exp = beam_pattern_params.get('cosine_taper_exponent', 1.0)
        A_theta = cosine_tapered_pattern(zenith_angle, hpbw_rad, taper_exponent=taper_exp)
    elif pattern_type == 'exponential':
        taper_db = beam_pattern_params.get('exponential_taper_dB', 10.0)
        A_theta = exponential_tapered_pattern(zenith_angle, hpbw_rad, taper_dB=taper_db)
    else:
        # Default to Gaussian
        A_theta = gaussian_A_theta_EBeam(zenith_angle, hpbw_rad)

    # Convert scalar amplitude to diagonal Jones matrix (unpolarized beam response)
    # Jones matrix: [[A, 0], [0, A]] for each source
    jones_matrices = np.zeros((n_sources, 2, 2), dtype=complex)
    jones_matrices[:, 0, 0] = A_theta
    jones_matrices[:, 1, 1] = A_theta

    return jones_matrices


def _extract_correlations(vis_matrix_array):
    """
    Extract correlation products from visibility matrix array.

    Parameters:
    -----------
    vis_matrix_array : ndarray
        Array of visibility matrices, shape (N_freq, 2, 2)

    Returns:
    --------
    dict: Dictionary with keys "XX", "XY", "YX", "YY", "I"
        Each value is an array of shape (N_freq,)
    """
    correlations = visibility_to_correlations(vis_matrix_array)
    return correlations


### Legacy Functions for Backward Compatibility ###


def calculate_modulus_phase(visibilities):
    """
    Calculate the modulus (amplitude) and phase of visibilities.

    Works with both old scalar format and new correlation dict format.

    Parameters:
    -----------
    visibilities : dict
        Dictionary of visibilities for each baseline.
        Can be scalar complex arrays (old format) or dicts of correlations (new format).

    Returns:
    --------
    tuple: (moduli, phases)
        moduli: Dictionary of amplitudes
        phases: Dictionary of phases in radians
    """
    moduli = {}
    phases = {}

    for key, val in visibilities.items():
        if isinstance(val, dict):
            # New format: dict of correlations
            # Use Stokes I for amplitude/phase
            moduli[key] = np.abs(val["I"])
            phases[key] = np.angle(val["I"])
        else:
            # Old format: scalar complex array
            moduli[key] = np.abs(val)
            phases[key] = np.angle(val)

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

    NOTE: This function is retained for backward compatibility but does not
    support full polarization. Use calculate_visibility() for polarized calculations.

    Parameters:
    -----------
    antennas : dict
        Antenna positions
    baselines : dict
        Baseline vectors
    sources : ndarray
        HEALPix flux map
    spectral_indices : ndarray
        HEALPix spectral index map
    location : EarthLocation
        Observer location
    obstime : Time
        Observation time
    wavelengths : Quantity
        Wavelengths
    freqs : ndarray
        Frequencies in Hz
    theta_HPBW : float
        HPBW in radians
    nside : int
        HEALPix NSIDE parameter

    Returns:
    --------
    dict: Dictionary of scalar visibilities (Stokes I only)
    """
    import healpy as hp

    visibilities = {
        key: np.zeros(len(wavelengths), dtype=complex) for key in baselines.keys()
    }

    # Get HEALPix pixel centers
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    ref_freq = 76e6  # Hz

    # Convert to RA, Dec
    ra = np.degrees(phi)
    dec = 90 - np.degrees(theta)
    source_coords = SkyCoord(ra=ra * au.deg, dec=dec * au.deg, frame="icrs")

    # Transform to AltAz
    altaz = source_coords.transform_to(AltAz(obstime=obstime, location=location))
    az = altaz.az.rad
    alt = altaz.alt.rad

    # Filter above horizon
    above_horizon = alt > 0
    az = az[above_horizon]
    alt = alt[above_horizon]
    fluxes = sources[above_horizon]
    alphas = spectral_indices[above_horizon]

    # Extrapolate fluxes
    extrapolated_fluxes = np.array([
        flux * (freq / ref_freq) ** alpha
        for freq in freqs
        for flux, alpha in zip(fluxes, alphas)
    ]).reshape(len(freqs), -1)

    # Direction cosines
    zenith_theta = np.pi / 2 - alt
    l = np.cos(alt) * np.sin(az)
    m = np.cos(alt) * np.cos(az)
    n = np.sin(alt)

    # Gaussian beam
    A_theta = gaussian_A_theta_EBeam(zenith_theta, theta_HPBW)

    # Loop over frequencies
    for i, (wavelength, freq) in enumerate(zip(wavelengths, freqs)):
        for (ant1, ant2), baseline in baselines.items():
            u, v, w = np.array(baseline["BaselineVector"]) / wavelength.value
            b_dot_s = u * l + v * m + w * n
            phase = np.exp(-2j * np.pi * b_dot_s)

            visibility = np.sum(extrapolated_fluxes[i] * A_theta * phase)
            visibilities[(ant1, ant2)][i] = visibility

    return visibilities


def convert_phase_for_display(phase_radians, angle_unit):
    """
    Convert phase from radians to the user's preferred unit for display.

    Parameters:
    -----------
    phase_radians : ndarray or float
        Phase value(s) in radians
    angle_unit : str
        Target unit ('degrees' or 'radians')

    Returns:
    --------
    Phase value(s) in the specified unit
    """
    from beams import convert_angle_for_display
    return convert_angle_for_display(phase_radians, angle_unit)
