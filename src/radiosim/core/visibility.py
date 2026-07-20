# radiosim/core/visibility.py
"""
Visibility calculation using the Radio Interferometer Measurement Equation (RIME).

Implements the direct-sum Jones/coherency calculation used by the high-level
analytic-beam path. Backend selection is explicit, but this module does not
claim complete accelerator coverage for the full simulation workflow.
"""

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radiosim.core.instrument_adapters import SolverInstrumentView
    from radiosim.core.sky.containers.model import SourceArrays

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import TimeDelta

# Import backend abstraction
from radiosim.backends import ArrayBackend, get_backend
from radiosim.core.instrument_adapters import InstrumentAdapterInvariantError

# Import Jones matrix framework
from radiosim.core.jones import (
    AnalyticBeamJones,
    BandpassJones,
    FITSBeamJones,
    GainJones,
    IonosphereJones,
    JonesChain,
    ParallacticAngleJones,
    PolarizationLeakageJones,
    TroposphereJones,
)

# Import polarization utilities
from radiosim.core.polarization import (
    stokes_to_coherency,
    visibility_to_correlations,
)

logger = logging.getLogger(__name__)


class _ResolvedInstrumentAnalyticBeamJones(AnalyticBeamJones):
    """Analytic beam whose antenna diameters are exact resolved identities."""

    def __init__(
        self,
        *,
        diameters_by_antenna: Mapping[int, float],
        **kwargs: Any,
    ) -> None:
        if not diameters_by_antenna:
            raise ValueError("diameters_by_antenna must not be empty")
        exact_diameters = MappingProxyType(
            {number: float(value) for number, value in diameters_by_antenna.items()}
        )
        self._resolved_diameters_by_antenna = exact_diameters

        # The base class retains a scalar parameter for standalone homogeneous
        # beam use. This private integration adapter overrides every diameter
        # lookup, so the scalar is deliberately identity-neutral and unreachable.
        super().__init__(
            diameter=1.0,
            diameter_per_antenna=dict(exact_diameters),
            **kwargs,
        )

    def _get_diameter_for_antenna(self, ant_num: Any) -> float:
        try:
            return self._resolved_diameters_by_antenna[ant_num]
        except KeyError as exc:
            raise InstrumentAdapterInvariantError(
                f"antenna number {ant_num!r} is absent from the resolved diameter map"
            ) from exc


def calculate_visibility(
    instrument: "SolverInstrumentView",
    source_arrays: "SourceArrays",
    location: Any,
    obstime: Any,
    wavelengths: Any,
    freqs: Any,
    duration_seconds: float,
    time_step_seconds: float,
    beam_manager: Any | None = None,
    return_correlations: bool = True,
    backend: ArrayBackend | None = None,
    jones_config: dict[str, Any] | None = None,
) -> dict:
    """
    Calculate complex visibility using full polarization (RIME).

    Implements: V_pq = Σ_sources J_p @ C_source @ J_q^H

    Where J is the total Jones matrix chain: J = B @ G @ D @ P @ E @ T @ Z @ K

    Parameters
    ----------
    instrument : SolverInstrumentView
        Owned canonical antenna values and selected baseline geometry.
    source_arrays : dict
        Dict of source arrays from ``SkyModel.as_point_source_arrays()``.
        Keys: ``ra_rad``, ``dec_rad``, ``flux``, ``spectral_index``,
        ``stokes_q``, ``stokes_u``, ``stokes_v``, ``ref_freq``,
        ``rotation_measure``, ``major_arcsec``, ``minor_arcsec``,
        ``pa_deg``, ``spectral_coeffs``.
    location : EarthLocation
        Observer's geographical location.
    obstime : Time
        Observation time.
    wavelengths : Quantity
        Wavelength array corresponding to frequencies (with units).
    freqs : ndarray
        Frequency array in Hz.
    beam_manager : BeamManager, optional
        Internal low-level beam adapter. The public high-level configuration
        currently resolves only analytic beams.
    return_correlations : bool, optional
        If True, extract and return correlation products (XX, XY, YX, YY, I).
        If False, return raw 2×2 visibility matrices.
    backend : ArrayBackend, optional
        Array backend used by supported kernels. If omitted, uses NumPy.
        Options: get_backend("numpy"), get_backend("jax"), get_backend("numba")
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
        Numeric antenna pairs map to correlation dictionaries when
        ``return_correlations`` is true, or to visibility matrices otherwise.
        Correlation arrays have shape ``(N_times, N_freq)``. Matrix arrays have
        shape ``(N_times, N_freq, 2, 2)``.

    Examples
    --------
    >>> # Deterministic NumPy default
    >>> vis = calculate_visibility(instrument, source_arrays, ...)

    >>> # Explicit optional backend
    >>> from radiosim.backends import get_backend
    >>> optional_backend = get_backend("jax")
    >>> vis = calculate_visibility(..., backend=optional_backend)
    """
    if jones_config is None:
        jones_config = {}

    from radiosim.core.instrument_adapters import SolverInstrumentView

    if type(instrument) is not SolverInstrumentView:
        raise TypeError("instrument must be a SolverInstrumentView")

    # Initialize backend (default to NumPy for backward compatibility)
    if backend is None:
        backend = get_backend("numpy")

    # Get array namespace from backend
    xp = backend.xp

    # Extract arrays from source_arrays dict
    _ra_rad = source_arrays["ra_rad"]
    _dec_rad = source_arrays["dec_rad"]
    _ref_freq = source_arrays["ref_freq"]

    # Calculate number of time steps
    n_times = max(1, int(duration_seconds / time_step_seconds))
    n_freq = len(wavelengths)

    # Initialize visibilities dictionary with time dimension
    # Each baseline gets a (N_times, N_freq, 2, 2) array for visibility matrices
    visibilities_matrices = {
        key: backend.zeros_complex((n_times, n_freq, 2, 2))
        for key in instrument.selected_pairs
    }

    # Handle empty source arrays
    if len(_ra_rad) == 0:
        if return_correlations:
            return {
                key: _extract_correlations(backend.to_numpy(val))
                for key, val in visibilities_matrices.items()
            }
        return {
            key: backend.to_numpy(val) for key, val in visibilities_matrices.items()
        }

    # Build SkyCoord from RA/Dec arrays (time-invariant)
    source_coords = SkyCoord(
        ra=np.rad2deg(_ra_rad) * u.deg,
        dec=np.rad2deg(_dec_rad) * u.deg,
        frame="icrs",
    )
    if isinstance(_ref_freq, np.ndarray):
        source_ref_freq_orig = _ref_freq.astype(np.float64, copy=False)
    else:
        source_ref_freq_orig = np.full(len(_ra_rad), _ref_freq, dtype=np.float64)
    source_stokes_I_orig = source_arrays["flux"]
    source_stokes_Q_orig = source_arrays["stokes_q"]
    source_stokes_U_orig = source_arrays["stokes_u"]
    source_stokes_V_orig = source_arrays["stokes_v"]
    source_spectral_indices_orig = source_arrays["spectral_index"]
    source_rm_orig = (
        source_arrays["rotation_measure"]
        if source_arrays["rotation_measure"] is not None
        else np.zeros(len(_ra_rad), dtype=np.float64)
    )

    # Multi-term spectral coefficients
    source_spectral_coeffs_orig = source_arrays["spectral_coeffs"]

    # Per-channel Stokes tables (lossless multi-frequency spectrum).
    # When populated, evaluate_point_flux_at_freq short-circuits to
    # nearest-channel lookup instead of spectral-index extrapolation.
    source_per_channel_flux_orig = source_arrays["per_channel_flux"]
    source_per_channel_q_orig = source_arrays["per_channel_stokes_q"]
    source_per_channel_u_orig = source_arrays["per_channel_stokes_u"]
    source_per_channel_v_orig = source_arrays["per_channel_stokes_v"]
    source_channel_frequencies = source_arrays["channel_frequencies"]

    # Gaussian morphology
    _maj = source_arrays["major_arcsec"]
    _minn = source_arrays["minor_arcsec"]
    _pa = source_arrays["pa_deg"]
    source_major_orig = _maj if _maj is not None else np.zeros(len(_ra_rad))
    source_minor_orig = _minn if _minn is not None else np.zeros(len(_ra_rad))
    source_pa_orig = _pa if _pa is not None else np.zeros(len(_ra_rad))
    has_gaussians = np.any(source_major_orig > 0)

    # Pre-compute Gaussian (a, b, c) quadratic form coefficients
    if has_gaussians:
        _maj_rad = np.deg2rad(source_major_orig / 3600.0)
        _min_rad = np.deg2rad(source_minor_orig / 3600.0)
        _pa_rad = np.deg2rad(source_pa_orig)
        _K_maj = (np.pi**2 * _maj_rad**2) / (4.0 * np.log(2))
        _K_min = (np.pi**2 * _min_rad**2) / (4.0 * np.log(2))
        gauss_a_orig = np.cos(_pa_rad) ** 2 * _K_min + np.sin(_pa_rad) ** 2 * _K_maj
        gauss_b_orig = 0.5 * np.sin(2 * _pa_rad) * (_K_maj - _K_min)
        gauss_c_orig = np.sin(_pa_rad) ** 2 * _K_min + np.cos(_pa_rad) ** 2 * _K_maj

    # ===========================================================================
    # TIME LOOP: Iterate over time steps, updating source positions each step
    # ===========================================================================
    for time_idx in range(n_times):
        # Update observation time for this step
        current_obstime = obstime + TimeDelta(
            time_step_seconds * time_idx, format="sec"
        )

        # Transform source coordinates to AltAz frame (changes with time!)
        altaz = source_coords.transform_to(
            AltAz(obstime=current_obstime, location=location)
        )
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
        source_ref_freq_t = source_ref_freq_orig[above_horizon]
        source_rm_t = source_rm_orig[above_horizon]
        source_spectral_coeffs_t = (
            source_spectral_coeffs_orig[above_horizon]
            if source_spectral_coeffs_orig is not None
            else None
        )
        source_per_channel_flux_t = (
            source_per_channel_flux_orig[:, above_horizon]
            if source_per_channel_flux_orig is not None
            else None
        )
        source_per_channel_q_t = (
            source_per_channel_q_orig[:, above_horizon]
            if source_per_channel_q_orig is not None
            else None
        )
        source_per_channel_u_t = (
            source_per_channel_u_orig[:, above_horizon]
            if source_per_channel_u_orig is not None
            else None
        )
        source_per_channel_v_t = (
            source_per_channel_v_orig[:, above_horizon]
            if source_per_channel_v_orig is not None
            else None
        )
        if has_gaussians:
            gauss_a_t = gauss_a_orig[above_horizon]
            gauss_b_t = gauss_b_orig[above_horizon]
            gauss_c_t = gauss_c_orig[above_horizon]

        n_sources = len(az_rad_t)

        # Calculate direction cosines (l, m, n) for this time step
        l_np = np.cos(alt_rad_t) * np.sin(az_rad_t)
        m_np = np.cos(alt_rad_t) * np.cos(az_rad_t)
        n_np = np.sin(alt_rad_t)
        l_dir = backend.asarray(l_np, dtype=backend.default_real_dtype)
        m_dir = backend.asarray(m_np, dtype=backend.default_real_dtype)
        n_dir = backend.asarray(n_np, dtype=backend.default_real_dtype)
        source_stokes_I_t = backend.asarray(
            source_stokes_I_t, dtype=backend.default_real_dtype
        )
        source_stokes_Q_t = backend.asarray(
            source_stokes_Q_t, dtype=backend.default_real_dtype
        )
        source_stokes_U_t = backend.asarray(
            source_stokes_U_t, dtype=backend.default_real_dtype
        )
        source_stokes_V_t = backend.asarray(
            source_stokes_V_t, dtype=backend.default_real_dtype
        )
        source_spectral_indices_t = backend.asarray(
            source_spectral_indices_t, dtype=backend.default_real_dtype
        )
        source_ref_freq_t = backend.asarray(
            source_ref_freq_t, dtype=backend.default_real_dtype
        )
        source_rm_t = backend.asarray(source_rm_t, dtype=backend.default_real_dtype)
        if source_spectral_coeffs_t is not None:
            source_spectral_coeffs_t = backend.asarray(
                source_spectral_coeffs_t, dtype=backend.default_real_dtype
            )
        if source_per_channel_flux_t is not None:
            source_per_channel_flux_t = backend.asarray(
                source_per_channel_flux_t, dtype=backend.default_real_dtype
            )
        if source_per_channel_q_t is not None:
            source_per_channel_q_t = backend.asarray(
                source_per_channel_q_t, dtype=backend.default_real_dtype
            )
        if source_per_channel_u_t is not None:
            source_per_channel_u_t = backend.asarray(
                source_per_channel_u_t, dtype=backend.default_real_dtype
            )
        if source_per_channel_v_t is not None:
            source_per_channel_v_t = backend.asarray(
                source_per_channel_v_t, dtype=backend.default_real_dtype
            )
        if has_gaussians:
            gauss_a_t = backend.asarray(gauss_a_t, dtype=backend.default_real_dtype)
            gauss_b_t = backend.asarray(gauss_b_t, dtype=backend.default_real_dtype)
            gauss_c_t = backend.asarray(gauss_c_t, dtype=backend.default_real_dtype)

        for freq_idx, (wavelength, freq) in enumerate(
            zip(wavelengths, freqs, strict=True)
        ):
            # Resolve Stokes at this observation frequency. Short-circuits to
            # nearest-channel lookup when per_channel_flux is populated;
            # otherwise applies spectral-index extrapolation + Faraday rotation.
            from radiosim.core.sky.containers.spectral import (
                evaluate_point_flux_at_freq,
            )

            I_scaled, Q_scaled, U_scaled, V_scaled = evaluate_point_flux_at_freq(
                source_stokes_I_t,
                source_stokes_Q_t,
                source_stokes_U_t,
                source_stokes_V_t,
                source_spectral_indices_t,
                source_spectral_coeffs_t,
                source_ref_freq_t,
                source_rm_t,
                source_per_channel_flux_t,
                source_per_channel_q_t,
                source_per_channel_u_t,
                source_per_channel_v_t,
                source_channel_frequencies,
                freq,
                xp=xp,
            )

            # Coherency matrices: (n_sources, 2, 2)
            coherency_matrices = stokes_to_coherency(
                I_scaled, Q_scaled, U_scaled, V_scaled, xp=xp
            )

            is_unpolarized = bool(
                backend.to_numpy(
                    xp.all((Q_scaled == 0) & (U_scaled == 0) & (V_scaled == 0))
                )
            )

            # Build JonesChain (without K — K is applied separately)
            chain = _build_jones_chain(
                backend,
                jones_config,
                instrument,
                alt_rad_t,
                az_rad_t,
                freq,
                freq_idx,
                n_sources,
                location,
                time_idx,
                beam_manager=beam_manager,
            )

            # Per-antenna Jones cache: compute chain once per antenna
            jones_antenna_cache = {}
            for ant_num in {a for pair in instrument.selected_pairs for a in pair}:
                ant_idx = instrument.row_for_number(ant_num)
                jones_antenna_cache[ant_num] = chain.compute_antenna_jones_all_sources(
                    antenna_idx=ant_idx,
                    n_sources=n_sources,
                    freq_idx=0,  # single freq per loop iteration
                    time_idx=0,
                    antenna_number=ant_num,
                )

            # Compute visibilities per baseline
            for (ant1, ant2), baseline_vector in zip(
                instrument.selected_pairs,
                instrument.baseline_vectors_enu_m,
                strict=True,
            ):
                J_p = jones_antenna_cache[ant1]  # (n_sources, 2, 2)
                J_q = jones_antenna_cache[ant2]

                # Geometric phase (K) applied separately
                bl_u, bl_v, bl_w = (
                    backend.asarray(
                        baseline_vector,
                        dtype=backend.default_real_dtype,
                    )
                    / wavelength.value
                )
                b_dot_s = bl_u * l_dir + bl_v * m_dir + bl_w * (n_dir - 1.0)
                phase = backend.exp(-2j * np.pi * b_dot_s)

                # Gaussian envelope: scalar attenuation per source
                if has_gaussians:
                    envelope = backend.exp(
                        -(
                            gauss_a_t * bl_u**2
                            + 2 * gauss_b_t * bl_u * bl_v
                            + gauss_c_t * bl_v**2
                        )
                    )
                else:
                    envelope = 1.0

                # Vectorized RIME: V = sum_s phase_s * J_p[s] @ C[s] @ J_q_H[s]
                J_q_H = backend.conjugate_transpose(J_q)

                if is_unpolarized:
                    V_all = backend.matmul(J_p, J_q_H)
                    V_all = V_all * (I_scaled * phase * envelope / 2.0)[:, None, None]
                else:
                    V_all = backend.matmul(
                        backend.matmul(J_p, coherency_matrices), J_q_H
                    )
                    V_all = V_all * (phase * envelope)[:, None, None]

                visibility_matrix = backend.sum(V_all, axis=0)
                visibilities_matrices[(ant1, ant2)] = backend.set_at(
                    visibilities_matrices[(ant1, ant2)],
                    (time_idx, freq_idx),
                    visibility_matrix,
                )

    # Convert backend arrays to numpy for output
    result_matrices = {
        key: backend.to_numpy(val) for key, val in visibilities_matrices.items()
    }

    # Convert to correlation products if requested
    if return_correlations:
        visibilities_correlations = {}
        for baseline_key, vis_matrix_array in result_matrices.items():
            visibilities_correlations[baseline_key] = _extract_correlations(
                vis_matrix_array
            )
        return visibilities_correlations

    return result_matrices


def _build_jones_chain(
    backend,
    jones_config,
    instrument,
    alt_rad,
    az_rad,
    freq,
    freq_idx,
    n_sources,
    location,
    time_idx,
    beam_manager=None,
):
    """Build a JonesChain with configured terms (K excluded).

    K is excluded because it requires baseline coordinates and is applied
    separately as a scalar phase multiplication for efficiency.

    Parameters
    ----------
    backend : ArrayBackend
        Computation backend.
    jones_config : dict
        Configuration for Jones chain terms.
    instrument : SolverInstrumentView
        Owned canonical antenna values.
    alt_rad, az_rad : ndarray
        Source altitudes/azimuths in radians.
    freq : float
        Frequency in Hz.
    freq_idx : int
        Frequency index.
    n_sources : int
        Number of sources.
    location : EarthLocation
        Observer location.
    time_idx : int
        Time step index.
    beam_manager : BeamManager, optional
        Internal low-level beam adapter. High-level configuration currently
        supplies only the analytic path.

    Returns
    -------
    JonesChain
        Chain with E (and optionally Z, T, P, D, G, B) terms.
    """
    n_antennas = len(instrument.antenna_numbers)
    chain = JonesChain(backend)

    # Z term: Ionosphere (optional)
    z_config = jones_config.get("Z", {})
    if z_config.get("enabled", False):
        tec = z_config.get("tec", 1e16)
        tec_array = np.full(n_antennas, tec)
        z_jones = IonosphereJones(
            tec=tec_array,
            frequencies=np.array([freq]),
            include_faraday=z_config.get("include_faraday", True),
            include_delay=z_config.get("include_delay", True),
        )
        chain.add_term(z_jones)

    # T term: Troposphere (optional)
    t_config = jones_config.get("T", {})
    if t_config.get("enabled", False):
        t_jones = TroposphereJones(
            n_antennas=n_antennas,
            frequencies=np.array([freq]),
            elevations=alt_rad,
        )
        chain.add_term(t_jones)

    # E term: Primary beam (always enabled)
    # Use FITS beam if beam_manager is available and not in analytic mode
    if (
        beam_manager is not None
        and getattr(beam_manager, "mode", "analytic") != "analytic"
    ):
        e_jones = FITSBeamJones(
            beam_manager=beam_manager,
            source_altaz=np.column_stack([alt_rad, az_rad]),
            frequencies=np.array([freq]),
        )
    else:
        beam_cfg = jones_config.get("beam", {})

        diameter_map = {
            number: float(instrument.diameters_m[index])
            for index, number in enumerate(instrument.antenna_numbers)
        }

        e_jones = _ResolvedInstrumentAnalyticBeamJones(
            source_altaz=np.column_stack([alt_rad, az_rad]),
            frequencies=np.array([freq]),
            aperture_shape=beam_cfg.get("aperture_shape", "circular"),
            taper=beam_cfg.get("taper", "gaussian"),
            edge_taper_dB=beam_cfg.get("edge_taper_dB", 10.0),
            feed_model=beam_cfg.get("feed_model", "none"),
            feed_computation=beam_cfg.get("feed_computation", "analytical"),
            feed_params=beam_cfg.get("feed_params"),
            reflector_type=beam_cfg.get("reflector_type", "prime_focus"),
            magnification=beam_cfg.get("magnification", 1.0),
            diameters_by_antenna=diameter_map,
            aperture_params=beam_cfg.get("aperture_params"),
        )
    chain.add_term(e_jones)

    # P term: Parallactic angle (optional)
    p_config = jones_config.get("P", {})
    if p_config.get("enabled", False):
        ant_latitudes = np.full(n_antennas, location.lat.rad)
        source_positions = np.column_stack([az_rad, alt_rad])
        p_jones = ParallacticAngleJones(
            antenna_latitudes=ant_latitudes,
            source_positions=source_positions,
            times=np.array([0.0]),
            mount_type=p_config.get("mount_type", "altaz"),
        )
        chain.add_term(p_jones)

    # D term: Polarization leakage (optional)
    d_config = jones_config.get("D", {})
    if d_config.get("enabled", False):
        d_jones = PolarizationLeakageJones(
            n_antennas=n_antennas,
            d_terms=d_config.get("d_terms"),
        )
        chain.add_term(d_jones)

    # G term: Electronic gains (optional)
    g_config = jones_config.get("G", {})
    if g_config.get("enabled", False):
        g_jones = GainJones(
            n_antennas=n_antennas,
            gain_sigma=g_config.get("sigma", 0.0),
        )
        chain.add_term(g_jones)

    # B term: Bandpass (optional)
    b_config = jones_config.get("B", {})
    if b_config.get("enabled", False):
        b_jones = BandpassJones(
            n_antennas=n_antennas,
            frequencies=np.array([freq]),
            bandpass_gains=b_config.get("bandpass_gains"),
        )
        chain.add_term(b_jones)

    return chain


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
