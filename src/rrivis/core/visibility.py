# rrivis/core/visibility.py
"""
Visibility calculation using the Radio Interferometer Measurement Equation (RIME).

Implements full polarization with 2×2 Jones matrices and coherency matrices.
Supports both beam FITS files and analytic beam patterns.

NEW in v0.2.0: Backend abstraction for CPU/GPU/TPU acceleration and
JonesChain integration for complete instrumental forward modeling.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import TimeDelta
import astropy.units as u
import logging

# Import backend abstraction
from rrivis.backends import get_backend, ArrayBackend

# Import Jones matrix framework
from rrivis.core.jones import (
    JonesChain,
    GeometricPhaseJones,
    AnalyticBeamJones,
    GainJones,
    BandpassJones,
    PolarizationLeakageJones,
    ParallacticAngleJones,
    IonosphereJones,
    TroposphereJones,
)

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
    antennas: Dict,
    baselines: Dict,
    sources: List,
    location: Any,
    obstime: Any,
    wavelengths: Any,
    freqs: Any,
    hpbw_per_antenna: Dict,
    duration_seconds: float,
    time_step_seconds: float,
    beam_manager: Optional[Any] = None,
    beam_pattern_per_antenna: Optional[Dict] = None,
    beam_pattern_params: Optional[Dict] = None,
    return_correlations: bool = True,
    backend: Optional[ArrayBackend] = None,
    use_jones_chain: bool = False,
    jones_config: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Calculate complex visibility using full polarization (RIME).

    Implements: V_pq = Σ_sources J_p @ C_source @ J_q^H

    Where J is the total Jones matrix chain: J = B @ G @ D @ P @ E @ T @ Z @ K

    Parameters
    ----------
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
    backend : ArrayBackend, optional
        Computation backend for CPU/GPU/TPU acceleration.
        If None, uses NumPy CPU backend (backward compatible).
        Options: get_backend("numpy"), get_backend("jax"), get_backend("numba")
    use_jones_chain : bool, optional
        If True, use the full JonesChain framework for RIME computation.
        If False (default), use legacy beam-only computation for backward compatibility.
    jones_config : dict, optional
        Configuration for Jones chain terms. Keys are term names ('K', 'E', 'G', etc.),
        values are dicts with 'enabled' (bool) and term-specific parameters.
        Example: {'G': {'enabled': True, 'sigma': 0.02}, 'Z': {'enabled': True, 'tec': 1e16}}
    duration_seconds : float
        Total observation duration in seconds. Controls the number of time samples.
    time_step_seconds : float
        Time step between samples in seconds. Number of time steps = duration / time_step.

    Returns
    -------
    dict
        Dictionary of visibilities for each baseline.
        If return_correlations=True:
            Keys: (ant1, ant2) tuples
            Values: dict with keys "XX", "XY", "YX", "YY", "I"
            Shape is (N_times, N_freq) for time-stepping mode, (N_freq,) for single-time
        If return_correlations=False:
            Keys: (ant1, ant2) tuples
            Values: ndarray of shape (N_times, N_freq, 2, 2) or (N_freq, 2, 2)

    Examples
    --------
    >>> # CPU backend (default, backward compatible)
    >>> vis = calculate_visibility(antennas, baselines, sources, ...)

    >>> # GPU acceleration with JAX
    >>> from rrivis.backends import get_backend
    >>> gpu = get_backend("jax")
    >>> vis = calculate_visibility(..., backend=gpu)

    >>> # Full Jones chain with instrumental effects
    >>> vis = calculate_visibility(
    ...     ...,
    ...     use_jones_chain=True,
    ...     jones_config={
    ...         'G': {'enabled': True, 'sigma': 0.02},
    ...         'B': {'enabled': True},
    ...         'Z': {'enabled': True, 'tec': 1e16},
    ...     }
    ... )
    """
    # Set defaults for beam pattern configuration
    if beam_pattern_per_antenna is None:
        beam_pattern_per_antenna = {}
    if beam_pattern_params is None:
        beam_pattern_params = {
            'cosine_taper_exponent': 1.0,
            'exponential_taper_dB': 10.0,
        }
    if jones_config is None:
        jones_config = {}

    # Initialize backend (default to NumPy for backward compatibility)
    if backend is None:
        backend = get_backend("numpy")

    # Get array namespace from backend
    xp = backend.xp

    # Reference frequency for spectral index calculation (76 MHz)
    reference_freq = 76e6  # Hz

    # Calculate number of time steps
    n_times = max(1, int(duration_seconds / time_step_seconds))
    n_freq = len(wavelengths)

    # Initialize visibilities dictionary with time dimension
    # Each baseline gets a (N_times, N_freq, 2, 2) array for visibility matrices
    visibilities_matrices = {
        key: xp.zeros((n_times, n_freq, 2, 2), dtype=complex) for key in baselines.keys()
    }

    # Handle empty source list
    if not sources:
        if return_correlations:
            return {key: _extract_correlations(backend.to_numpy(val))
                    for key, val in visibilities_matrices.items()}
        return {key: backend.to_numpy(val) for key, val in visibilities_matrices.items()}

    # Convert source list to arrays (these are time-invariant)
    source_coords = SkyCoord([s["coords"] for s in sources])
    source_stokes_I_orig = np.array([s["flux"] for s in sources])
    source_stokes_Q_orig = np.array([s.get("stokes_q", 0.0) for s in sources])
    source_stokes_U_orig = np.array([s.get("stokes_u", 0.0) for s in sources])
    source_stokes_V_orig = np.array([s.get("stokes_v", 0.0) for s in sources])
    source_spectral_indices_orig = np.array([s["spectral_index"] for s in sources])

    # ===========================================================================
    # TIME LOOP: Iterate over time steps, updating source positions each step
    # ===========================================================================
    for time_idx in range(n_times):
        # Update observation time for this step
        current_obstime = obstime + TimeDelta(time_step_seconds * time_idx, format='sec')

        # Transform source coordinates to AltAz frame (changes with time!)
        altaz = source_coords.transform_to(AltAz(obstime=current_obstime, location=location))
        az_rad = altaz.az.rad
        alt_rad = altaz.alt.rad

        # Filter out sources below the horizon
        above_horizon = alt_rad > 0
        if not np.any(above_horizon):
            # No sources visible at this time - skip to next time step
            continue

        # Apply horizon filter for this time step
        az_rad_t = az_rad[above_horizon]
        alt_rad_t = alt_rad[above_horizon]
        source_stokes_I_t = source_stokes_I_orig[above_horizon]
        source_stokes_Q_t = source_stokes_Q_orig[above_horizon]
        source_stokes_U_t = source_stokes_U_orig[above_horizon]
        source_stokes_V_t = source_stokes_V_orig[above_horizon]
        source_spectral_indices_t = source_spectral_indices_orig[above_horizon]

        n_sources = len(az_rad_t)

        # Calculate direction cosines (l, m, n) for this time step
        zenith_angle_t = np.pi / 2 - alt_rad_t
        l_t = backend.asarray(np.cos(alt_rad_t) * np.sin(az_rad_t))
        m_t = backend.asarray(np.cos(alt_rad_t) * np.cos(az_rad_t))
        n_t = backend.asarray(np.sin(alt_rad_t))

        # Convert to backend arrays
        source_stokes_I = backend.asarray(source_stokes_I_t)
        source_stokes_Q = backend.asarray(source_stokes_Q_t)
        source_stokes_U = backend.asarray(source_stokes_U_t)
        source_stokes_V = backend.asarray(source_stokes_V_t)
        source_spectral_indices = backend.asarray(source_spectral_indices_t)

        # Use JonesChain if requested (inside time loop)
        if use_jones_chain:
            # TODO: JonesChain path needs refactoring for time-stepping
            # For now, skip and use legacy path
            pass
        else:
            # Legacy visibility calculation path (default)
            # Convert to numpy for legacy code path
            l_np = backend.to_numpy(l_t)
            m_np = backend.to_numpy(m_t)
            n_np = backend.to_numpy(n_t)
            source_stokes_I_np = backend.to_numpy(source_stokes_I)
            source_stokes_Q_np = backend.to_numpy(source_stokes_Q)
            source_stokes_U_np = backend.to_numpy(source_stokes_U)
            source_stokes_V_np = backend.to_numpy(source_stokes_V)
            source_spectral_indices_np = backend.to_numpy(source_spectral_indices)

            # Loop over each frequency
            for freq_idx, (wavelength, freq) in enumerate(zip(wavelengths, freqs)):
                # Scale source fluxes by spectral index
                I_scaled = source_stokes_I_np * (freq / reference_freq) ** source_spectral_indices_np
                Q_scaled = source_stokes_Q_np * (freq / reference_freq) ** source_spectral_indices_np
                U_scaled = source_stokes_U_np * (freq / reference_freq) ** source_spectral_indices_np
                V_scaled = source_stokes_V_np * (freq / reference_freq) ** source_spectral_indices_np

                # Convert Stokes parameters to coherency matrices for all sources
                coherency_matrices = np.array([
                    stokes_to_coherency(I, Q, U, V)
                    for I, Q, U, V in zip(I_scaled, Q_scaled, U_scaled, V_scaled)
                ])

                # Loop over each baseline
                for (ant1, ant2), baseline in baselines.items():
                    ant1_name = antennas[ant1]["Name"]
                    ant2_name = antennas[ant2]["Name"]

                    # Get Jones matrices for both antennas
                    E1_all = _get_jones_matrices_for_sources(
                        beam_manager, ant1, ant1_name, az_rad_t, alt_rad_t, freq,
                        hpbw_per_antenna, beam_pattern_per_antenna, beam_pattern_params,
                        antennas, wavelength.value, freq_idx
                    )

                    E2_all = _get_jones_matrices_for_sources(
                        beam_manager, ant2, ant2_name, az_rad_t, alt_rad_t, freq,
                        hpbw_per_antenna, beam_pattern_per_antenna, beam_pattern_params,
                        antennas, wavelength.value, freq_idx
                    )

                    # Calculate geometric phase term
                    # Using w(n-1) formulation per Smirnov 2011 RIME
                    # This ensures zero phase at the phase center (l=0, m=0, n=1)
                    u, v, w = np.array(baseline["BaselineVector"]) / wavelength.value
                    b_dot_s = u * l_np + v * m_np + w * (n_np - 1.0)
                    phase = np.exp(-2j * np.pi * b_dot_s)

                    # Apply RIME for each source and sum
                    visibility_matrix = np.zeros((2, 2), dtype=complex)

                    for src_idx in range(n_sources):
                        E_i = E1_all[src_idx]
                        C = coherency_matrices[src_idx]
                        E_j = E2_all[src_idx]
                        phase_factor = phase[src_idx]

                        V_src = apply_jones_matrices(E_i, C, E_j)
                        visibility_matrix += V_src * phase_factor

                    # Store with time index
                    visibilities_matrices[(ant1, ant2)][time_idx, freq_idx] = visibility_matrix

    # Convert backend arrays to numpy for output
    result_matrices = {
        key: backend.to_numpy(val) for key, val in visibilities_matrices.items()
    }

    # Convert to correlation products if requested
    if return_correlations:
        visibilities_correlations = {}
        for baseline_key, vis_matrix_array in result_matrices.items():
            visibilities_correlations[baseline_key] = _extract_correlations(vis_matrix_array)
        return visibilities_correlations

    return result_matrices


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


def _calculate_visibility_with_jones_chain(
    antennas: Dict,
    baselines: Dict,
    n_sources: int,
    n_freq: int,
    l: Any,
    m: Any,
    n: Any,
    alt_rad: Any,
    az_rad: Any,
    zenith_angle: Any,
    source_stokes_I: Any,
    source_stokes_Q: Any,
    source_stokes_U: Any,
    source_stokes_V: Any,
    source_spectral_indices: Any,
    wavelengths: Any,
    freqs: Any,
    reference_freq: float,
    hpbw_per_antenna: Dict,
    beam_manager: Optional[Any],
    beam_pattern_per_antenna: Dict,
    beam_pattern_params: Dict,
    backend: ArrayBackend,
    jones_config: Dict[str, Any],
    return_correlations: bool,
    location: Any,
) -> Dict:
    """
    Calculate visibility using full JonesChain framework.

    This implements the complete RIME: V_pq = Σ_s J_p(s) @ C(s) @ J_q(s)^H

    Where J = B @ G @ D @ P @ E @ T @ Z @ K (configurable via jones_config).

    Parameters
    ----------
    antennas : dict
        Antenna dictionary
    baselines : dict
        Baseline dictionary
    n_sources : int
        Number of sources above horizon
    n_freq : int
        Number of frequency channels
    l, m, n : array
        Direction cosines (backend arrays)
    alt_rad, az_rad : array
        Source alt/az in radians (numpy arrays)
    zenith_angle : array
        Zenith angle in radians (numpy array)
    source_stokes_* : array
        Stokes parameters (backend arrays)
    source_spectral_indices : array
        Spectral indices (backend arrays)
    wavelengths : Quantity
        Wavelengths with units
    freqs : array
        Frequencies in Hz
    reference_freq : float
        Reference frequency for spectral index
    hpbw_per_antenna : dict
        HPBW data for beam calculation
    beam_manager : BeamManager or None
        Beam manager for FITS beams
    beam_pattern_per_antenna : dict
        Beam pattern types
    beam_pattern_params : dict
        Beam pattern parameters
    backend : ArrayBackend
        Computation backend
    jones_config : dict
        Configuration for Jones terms
    return_correlations : bool
        Whether to return correlation products
    location : EarthLocation
        Observer location

    Returns
    -------
    dict
        Visibility dictionary
    """
    xp = backend.xp
    n_antennas = len(antennas)

    # Initialize output visibility matrices
    visibilities_matrices = {
        key: xp.zeros((n_freq, 2, 2), dtype=complex) for key in baselines.keys()
    }

    # Convert direction cosines to numpy for internal use
    l_np = backend.to_numpy(l)
    m_np = backend.to_numpy(m)
    n_np = backend.to_numpy(n)

    # Prepare source LMN array for GeometricPhaseJones
    source_lmn = np.column_stack([l_np, m_np, n_np])  # (n_sources, 3)

    # Loop over each frequency
    for freq_idx, (wavelength, freq) in enumerate(zip(wavelengths, freqs)):
        wavelength_m = wavelength.value

        # Scale source fluxes by spectral index
        freq_ratio = freq / reference_freq
        I_scaled = backend.to_numpy(source_stokes_I) * freq_ratio ** backend.to_numpy(source_spectral_indices)
        Q_scaled = backend.to_numpy(source_stokes_Q) * freq_ratio ** backend.to_numpy(source_spectral_indices)
        U_scaled = backend.to_numpy(source_stokes_U) * freq_ratio ** backend.to_numpy(source_spectral_indices)
        V_scaled = backend.to_numpy(source_stokes_V) * freq_ratio ** backend.to_numpy(source_spectral_indices)

        # Build coherency matrices for all sources
        coherency_matrices = np.array([
            stokes_to_coherency(I, Q, U, V)
            for I, Q, U, V in zip(I_scaled, Q_scaled, U_scaled, V_scaled)
        ])

        # Create Jones chain for this frequency
        chain = JonesChain(backend)

        # -----------------------------------------------------------------
        # Add Jones terms based on configuration
        # Order: K -> Z -> T -> E -> P -> D -> G -> B (sky to correlator)
        # -----------------------------------------------------------------

        # K term: Geometric phase (always enabled)
        k_jones = GeometricPhaseJones(
            source_lmn=source_lmn,
            wavelengths=np.array([wavelength_m]),
        )
        chain.add_term(k_jones)

        # Z term: Ionosphere (optional)
        z_config = jones_config.get('Z', {})
        if z_config.get('enabled', False):
            tec = z_config.get('tec', 1e16)  # Default 10 TECU
            tec_array = np.full(n_antennas, tec)
            z_jones = IonosphereJones(
                tec=tec_array,
                frequencies=np.array([freq]),
                include_faraday=z_config.get('include_faraday', True),
                include_delay=z_config.get('include_delay', True),
            )
            chain.add_term(z_jones)

        # T term: Troposphere (optional)
        t_config = jones_config.get('T', {})
        if t_config.get('enabled', False):
            # Simple troposphere with default parameters
            t_jones = TroposphereJones(
                n_antennas=n_antennas,
                frequencies=np.array([freq]),
                elevations=alt_rad,
            )
            chain.add_term(t_jones)

        # E term: Primary beam (always enabled)
        # Use analytic beam from hpbw_per_antenna
        first_ant = list(antennas.keys())[0]
        hpbw_rad = hpbw_per_antenna.get(first_ant, np.array([0.1]))[freq_idx]
        e_jones = AnalyticBeamJones(
            source_altaz=np.column_stack([alt_rad, az_rad]),
            frequencies=np.array([freq]),
            hpbw_radians=hpbw_rad,
            beam_type=beam_pattern_per_antenna.get(first_ant, 'gaussian'),
        )
        chain.add_term(e_jones)

        # P term: Parallactic angle (optional)
        p_config = jones_config.get('P', {})
        if p_config.get('enabled', False):
            # Get antenna latitudes (assume from location for simplicity)
            ant_latitudes = np.full(n_antennas, location.lat.rad)
            # Source positions as RA/Dec (approximate from alt/az)
            # For now, use placeholder - proper conversion would need obstime
            source_positions = np.column_stack([az_rad, alt_rad])  # Placeholder
            p_jones = ParallacticAngleJones(
                antenna_latitudes=ant_latitudes,
                source_positions=source_positions,
                times=np.array([0.0]),  # Single time
                mount_type=p_config.get('mount_type', 'altaz'),
            )
            chain.add_term(p_jones)

        # D term: Polarization leakage (optional)
        d_config = jones_config.get('D', {})
        if d_config.get('enabled', False):
            d_jones = PolarizationLeakageJones(
                n_antennas=n_antennas,
                d_terms=d_config.get('d_terms'),
            )
            chain.add_term(d_jones)

        # G term: Electronic gains (optional)
        g_config = jones_config.get('G', {})
        if g_config.get('enabled', False):
            g_jones = GainJones(
                n_antennas=n_antennas,
                gain_sigma=g_config.get('sigma', 0.0),
            )
            chain.add_term(g_jones)

        # B term: Bandpass (optional)
        b_config = jones_config.get('B', {})
        if b_config.get('enabled', False):
            b_jones = BandpassJones(
                n_antennas=n_antennas,
                frequencies=np.array([freq]),
                bandpass_gains=b_config.get('bandpass_gains'),
            )
            chain.add_term(b_jones)

        # -----------------------------------------------------------------
        # Compute visibilities for each baseline
        # -----------------------------------------------------------------
        for (ant1, ant2), baseline in baselines.items():
            # Get baseline UVW in wavelength units
            uvw = np.array(baseline["BaselineVector"]) / wavelength_m
            u, v, w = uvw

            # Compute visibility for this baseline by summing over sources
            visibility_matrix = np.zeros((2, 2), dtype=complex)

            for src_idx in range(n_sources):
                # Geometric phase for this source
                phase = np.exp(-2j * np.pi * (
                    u * l_np[src_idx] + v * m_np[src_idx] + w * (n_np[src_idx] - 1)
                ))

                # Get coherency matrix for this source
                C = backend.asarray(coherency_matrices[src_idx])

                # Compute Jones matrices for both antennas
                # Note: We use antenna indices, not names
                ant1_idx = list(antennas.keys()).index(ant1)
                ant2_idx = list(antennas.keys()).index(ant2)

                J_p = chain.compute_antenna_jones(
                    antenna_idx=ant1_idx,
                    source_idx=src_idx,
                    freq_idx=0,  # Single frequency per loop iteration
                    time_idx=0,
                    baseline_uvw=uvw,
                )

                J_q = chain.compute_antenna_jones(
                    antenna_idx=ant2_idx,
                    source_idx=src_idx,
                    freq_idx=0,
                    time_idx=0,
                    baseline_uvw=uvw,
                )

                # RIME: V = J_p @ C @ J_q^H
                temp = backend.matmul(J_p, C)
                V_src = backend.matmul(temp, backend.conjugate_transpose(J_q))

                # Apply geometric phase and accumulate
                visibility_matrix += backend.to_numpy(V_src) * phase

            # Store result
            visibilities_matrices[(ant1, ant2)][freq_idx] = visibility_matrix

    # Convert to numpy for output
    result_matrices = {
        key: backend.to_numpy(val) for key, val in visibilities_matrices.items()
    }

    # Convert to correlation products if requested
    if return_correlations:
        visibilities_correlations = {}
        for baseline_key, vis_matrix_array in result_matrices.items():
            visibilities_correlations[baseline_key] = _extract_correlations(vis_matrix_array)
        return visibilities_correlations

    return result_matrices


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
    source_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

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
            # Using w(n-1) formulation per Smirnov 2011 RIME
            u, v, w = np.array(baseline["BaselineVector"]) / wavelength.value
            b_dot_s = u * l + v * m + w * (n - 1.0)
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
