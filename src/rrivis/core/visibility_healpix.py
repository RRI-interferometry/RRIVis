# rrivis/core/visibility_healpix.py
"""
HEALPix-based visibility calculation for diffuse emission.

This module provides visibility calculation that works directly with HEALPix
brightness temperature maps, avoiding the inefficiency of converting each
pixel to a point source.

The key advantages over the point source approach:
1. Works in brightness temperature (K) - more physical for diffuse emission
2. No per-pixel Jy conversion overhead
3. Pixel coordinates pre-computed and cached
4. Single Rayleigh-Jeans conversion factor applied at the end

References
----------
- healvis: https://github.com/rasg-affiliates/healvis
- pyuvsim: https://github.com/RadioAstronomySoftwareGroup/pyuvsim
"""

import logging
from typing import Any

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import AltAz
from astropy.time import TimeDelta

from rrivis.backends import ArrayBackend, get_backend
from rrivis.core.polarization import stokes_to_coherency
from rrivis.core.sky import (
    C_LIGHT,
    K_BOLTZMANN,
    SkyModel,
    brightness_temp_to_flux_density,
)

logger = logging.getLogger(__name__)


def _compute_beam_power_pattern(
    zenith_angles: np.ndarray,
    diameter: float,
    frequency: float,
    beam_manager: Any | None = None,
    antenna_number: Any | None = None,
    azimuth: np.ndarray | None = None,
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
    """Compute scalar beam power pattern B^2 for HEALPix pixels.

    For FITS beams: B^2 = 0.5 * (|J_00|^2 + |J_01|^2 + |J_10|^2 + |J_11|^2)
    For analytic beams: Uses compute_aperture_beam to get Jones, then
    B^2 = 0.5 * sum(|J_ij|^2).

    Parameters
    ----------
    zenith_angles : ndarray
        Zenith angles in radians, shape (N,).
    diameter : float
        Antenna diameter in meters.
    frequency : float
        Frequency in Hz.
    beam_manager : BeamManager, optional
        BeamManager for FITS beam lookup.
    antenna_number : int, optional
        Antenna number for beam_manager lookup.
    azimuth : ndarray, optional
        Azimuth angles in radians, shape (N,).
    aperture_shape : str
        Aperture geometry type.
    taper : str
        Illumination taper function.
    edge_taper_dB : float
        Edge taper in dB.
    feed_model : str
        Feed pattern model.
    feed_params : dict, optional
        Feed-specific parameters.
    feed_computation : str
        'analytical' or 'numerical'.
    reflector_type : str
        Reflector geometry type.
    magnification : float
        Cassegrain magnification.
    aperture_params : dict, optional
        Aperture-specific parameters.

    Returns
    -------
    power_pattern : ndarray
        Beam power pattern, shape (N,). Values in [0, 1].
    """
    if beam_manager is not None and antenna_number is not None:
        jones = beam_manager.get_jones_matrix(
            antenna_number=antenna_number,
            alt_rad=np.pi / 2 - zenith_angles,
            az_rad=azimuth if azimuth is not None else np.zeros_like(zenith_angles),
            freq_hz=frequency,
            location=None,
            time=None,
        )
        if jones is not None:
            # Power pattern = sum of |J_ij|^2 / 2  (average over polarizations)
            return 0.5 * np.sum(np.abs(jones) ** 2, axis=(-2, -1))

    # Aperture-based analytic beam
    from rrivis.core.jones.beam.analytic.composed import compute_aperture_beam

    jones = compute_aperture_beam(
        theta=zenith_angles,
        phi=azimuth,
        frequency=frequency,
        diameter=diameter,
        aperture_shape=aperture_shape,
        taper=taper,
        edge_taper_dB=edge_taper_dB,
        feed_model=feed_model,
        feed_params=feed_params,
        feed_computation=feed_computation,
        reflector_type=reflector_type,
        magnification=magnification,
        aperture_params=aperture_params,
    )
    return 0.5 * np.sum(np.abs(jones) ** 2, axis=(-2, -1))


def _compute_beam_jones_matrix(
    zenith_angles: np.ndarray,
    diameter: float,
    frequency: float,
    beam_manager: Any | None = None,
    antenna_number: Any | None = None,
    azimuth: np.ndarray | None = None,
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
    """Compute 2x2 Jones beam matrix for HEALPix pixels.

    Same inputs as ``_compute_beam_power_pattern``, but returns the full
    Jones matrices instead of collapsing to a scalar power pattern.

    Returns
    -------
    jones : ndarray
        Complex Jones matrices, shape ``(N, 2, 2)``.
    """
    if beam_manager is not None and antenna_number is not None:
        jones = beam_manager.get_jones_matrix(
            antenna_number=antenna_number,
            alt_rad=np.pi / 2 - zenith_angles,
            az_rad=azimuth if azimuth is not None else np.zeros_like(zenith_angles),
            freq_hz=frequency,
            location=None,
            time=None,
        )
        if jones is not None:
            return jones

    from rrivis.core.jones.beam.analytic.composed import compute_aperture_beam

    return compute_aperture_beam(
        theta=zenith_angles,
        phi=azimuth,
        frequency=frequency,
        diameter=diameter,
        aperture_shape=aperture_shape,
        taper=taper,
        edge_taper_dB=edge_taper_dB,
        feed_model=feed_model,
        feed_params=feed_params,
        feed_computation=feed_computation,
        reflector_type=reflector_type,
        magnification=magnification,
        aperture_params=aperture_params,
    )


def compute_beam_squared_integral(
    beam_power: np.ndarray, pixel_solid_angle: float
) -> float:
    """Compute the beam squared integral (beam solid angle).

    Omega_pp = sum(B^2 * Omega_pix)

    Useful for power spectrum normalization.

    Parameters
    ----------
    beam_power : ndarray
        Beam power pattern B^2, shape (N_pixels,).
    pixel_solid_angle : float
        Solid angle per pixel in steradians.

    Returns
    -------
    float
        Beam squared integral in steradians.
    """
    return float(np.sum(beam_power * pixel_solid_angle))


def calculate_visibility_healpix(
    sky_model: SkyModel,
    antennas: dict,
    baselines: dict,
    location: Any,
    obstime: Any,
    wavelengths: Any,
    freqs: Any,
    duration_seconds: float,
    time_step_seconds: float,
    beam_manager: Any | None = None,
    backend: ArrayBackend | None = None,
    output_units: str = "Jy",
    beam_config: dict | None = None,
    include_polarization: bool = False,
) -> dict:
    """
    Calculate visibility directly from HEALPix brightness temperature map.

    This function computes visibilities using the direct sum over HEALPix pixels,
    working in brightness temperature and applying the Rayleigh-Jeans conversion
    at the end.

    **Scalar mode** (default):

    ``V(b, ν) = (2kν²/c²) × Ω_pixel × Σ_pixels T(p) × exp(-2πi b·ŝ(p) / λ)``

    **Polarized mode** (``include_polarization=True``):

    ``V_pq(ν) = Σ_pixels J_p(p) @ C(p) @ J_q^H(p) × exp(-2πi b·ŝ(p) / λ)``

    where C(p) is the 2×2 coherency matrix built from Stokes I/Q/U/V per pixel.

    Parameters
    ----------
    sky_model : SkyModel
        Sky model in HEALPix mode (brightness temperature in K).
    antennas : dict
        Dictionary of antenna positions and properties.
    baselines : dict
        Dictionary of baselines between antennas.
    location : EarthLocation
        Observer's geographical location.
    obstime : Time
        Observation start time.
    wavelengths : Quantity
        Wavelength array with units.
    freqs : ndarray
        Frequency array in Hz.
    duration_seconds : float
        Total observation duration in seconds.
    time_step_seconds : float
        Time step for integration in seconds.
    beam_manager : BeamManager, optional
        Beam pattern manager for FITS-based beams.
    backend : ArrayBackend, optional
        Computation backend (CPU/GPU).
    output_units : str, default="Jy"
        Output units: "Jy" (convert to Jansky) or "K.sr" (keep temperature ×
        solid angle). In polarized mode, always "Jy".
    beam_config : dict, optional
        Beam configuration with keys: aperture_shape, taper, edge_taper_dB,
        feed_model, feed_params, feed_computation, reflector_type, magnification,
        aperture_params.
    include_polarization : bool, default=False
        If True and sky model has polarized HEALPix maps, compute full 2×2
        visibility matrices using the RIME with Jones beam matrices. Output
        shape becomes ``(n_baselines, n_times, n_freqs, 2, 2)``.

    Returns
    -------
    dict
        Dictionary containing:
        - visibilities: Complex visibility array. Scalar mode:
          ``(n_baselines, n_times, n_freqs)``.
          Polarized mode: ``(n_baselines, n_times, n_freqs, 2, 2)``.
        - times: Time array
        - frequencies: Frequency array
        - baselines: Baseline info
        - metadata: Additional information
    """
    from rrivis.core.sky.model import SkyFormat

    if sky_model.mode != SkyFormat.HEALPIX:
        raise ValueError(
            "sky_model must be in healpix_map mode. "
            "Call materialize_healpix() first (for point-source catalogs) "
            "or use from_catalog('diffuse_sky', frequencies=...) (for diffuse models)."
        )

    # Determine if we should use the polarized path
    use_polarization = include_polarization and sky_model.has_polarized_healpix_maps

    # Get backend
    if backend is None:
        backend = get_backend("numpy")

    # Get multi-frequency map metadata
    _, nside, _ = sky_model.get_multifreq_maps()
    npix = hp.nside2npix(nside)
    omega_pixel = sky_model.pixel_solid_angle
    pixel_coords = sky_model.pixel_coords

    pol_label = "polarized (2x2 RIME)" if use_polarization else "scalar"
    logger.info(
        f"HEALPix visibility calculation: nside={nside}, {npix} pixels, {pol_label}"
    )
    logger.info(
        f"Pixel solid angle: {omega_pixel:.6f} sr ({np.degrees(np.sqrt(omega_pixel)):.3f}\u00b0)"
    )

    # Setup time steps
    n_times = int(np.ceil(duration_seconds / time_step_seconds))
    times = np.arange(n_times) * time_step_seconds

    # Setup baseline info
    baseline_keys = list(baselines.keys())
    n_baselines = len(baseline_keys)
    n_freqs = len(freqs)

    # Pre-compute baseline vectors in local ENU
    baseline_vectors = np.array(
        [baselines[bl]["BaselineVector"] for bl in baseline_keys]
    )

    # Initialize output array
    if use_polarization:
        visibilities = np.zeros(
            (n_baselines, n_times, n_freqs, 2, 2), dtype=np.complex128
        )
    else:
        visibilities = np.zeros((n_baselines, n_times, n_freqs), dtype=np.complex128)

    logger.info(
        f"Computing visibilities: {n_times} times \u00d7 {n_freqs} freqs "
        f"\u00d7 {n_baselines} baselines"
    )

    # ==========================================================================
    # TIME LOOP
    # ==========================================================================
    for time_idx in range(n_times):
        current_obstime = obstime + TimeDelta(
            time_step_seconds * time_idx, format="sec"
        )

        # Transform pixel coordinates to AltAz
        altaz = pixel_coords.transform_to(
            AltAz(obstime=current_obstime, location=location)
        )
        az_rad = altaz.az.rad
        alt_rad = altaz.alt.rad

        # Filter pixels above horizon
        above_horizon = alt_rad > 0
        if not np.any(above_horizon):
            continue

        n_visible = np.sum(above_horizon)

        # Get visible pixel indices and geometry
        az_vis = az_rad[above_horizon]
        alt_vis = alt_rad[above_horizon]

        # Compute direction cosines (dir_l, dir_m, dir_n) in local ENU frame
        # dir_l = East, dir_m = North, dir_n = Up (zenith)
        dir_l = np.cos(alt_vis) * np.sin(az_vis)
        dir_m = np.cos(alt_vis) * np.cos(az_vis)
        dir_n = np.sin(alt_vis)

        # Zenith angles for beam computation
        za_vis = np.pi / 2 - alt_vis

        # Collect unique antenna numbers
        ant_nums = set()
        for ant1, ant2 in baselines:
            ant_nums.add(ant1)
            ant_nums.add(ant2)

        has_beam_manager = (
            beam_manager is not None
            and getattr(beam_manager, "mode", "analytic") != "analytic"
        )
        bcfg = beam_config or {}

        # ======================================================================
        # FREQUENCY LOOP
        # ======================================================================
        for freq_idx, (wavelength, freq) in enumerate(
            zip(wavelengths, freqs, strict=False)
        ):
            wavelength_m = wavelength.to(u.m).value

            if use_polarization:
                # ----- POLARIZED PATH -----
                # Get all Stokes maps at this frequency
                I_map, Q_map, U_map, V_map = sky_model.get_stokes_maps_at_frequency(
                    freq
                )

                I_vis = I_map[above_horizon].astype(np.float64)
                Q_vis = (
                    Q_map[above_horizon].astype(np.float64)
                    if Q_map is not None
                    else np.zeros_like(I_vis)
                )
                U_vis = (
                    U_map[above_horizon].astype(np.float64)
                    if U_map is not None
                    else np.zeros_like(I_vis)
                )
                V_vis = (
                    V_map[above_horizon].astype(np.float64)
                    if V_map is not None
                    else np.zeros_like(I_vis)
                )

                # Stokes I: respect brightness_conversion (Planck or RJ)
                conversion = getattr(sky_model, "brightness_conversion", "planck")
                if conversion == "rayleigh-jeans":
                    rj_factor_I = (
                        (2 * K_BOLTZMANN * freq**2 / C_LIGHT**2) * omega_pixel * 1e26
                    )
                    I_jy = I_vis * rj_factor_I
                else:
                    I_jy = np.zeros(len(I_vis))
                    pos = I_vis > 0
                    if np.any(pos):
                        I_jy[pos] = brightness_temp_to_flux_density(
                            I_vis[pos].astype(np.float64),
                            freq,
                            omega_pixel,
                            method="planck",
                        )

                # Stokes Q/U/V: always RJ (can be negative, RJ is linear)
                rj_factor_pol = (
                    (2 * K_BOLTZMANN * freq**2 / C_LIGHT**2) * omega_pixel * 1e26
                )
                Q_jy = Q_vis * rj_factor_pol
                U_jy = U_vis * rj_factor_pol
                V_jy = V_vis * rj_factor_pol

                # Build per-pixel coherency matrices: (n_visible, 2, 2)
                coherency = stokes_to_coherency(I_jy, Q_jy, U_jy, V_jy)

                # Compute per-antenna Jones matrices
                jones_cache: dict[Any, np.ndarray] = {}
                for ant_num in ant_nums:
                    ant_diameter = antennas.get(ant_num, {}).get("diameter", 14.0)
                    jones_cache[ant_num] = _compute_beam_jones_matrix(
                        zenith_angles=za_vis,
                        diameter=ant_diameter,
                        frequency=freq,
                        beam_manager=beam_manager if has_beam_manager else None,
                        antenna_number=ant_num,
                        azimuth=az_vis,
                        aperture_shape=bcfg.get("aperture_shape", "circular"),
                        taper=bcfg.get("taper", "gaussian"),
                        edge_taper_dB=bcfg.get("edge_taper_dB", 10.0),
                        feed_model=bcfg.get("feed_model", "none"),
                        feed_params=bcfg.get("feed_params"),
                        feed_computation=bcfg.get("feed_computation", "analytical"),
                        reflector_type=bcfg.get("reflector_type", "prime_focus"),
                        magnification=bcfg.get("magnification", 1.0),
                        aperture_params=bcfg.get("aperture_params"),
                    )

                # Compute visibility for each baseline
                # V_pq = Σ_pix phase_pix * J_p @ C_pix @ J_q^H
                for bl_idx, ((ant1, ant2), bl_vec) in enumerate(
                    zip(baseline_keys, baseline_vectors, strict=False)
                ):
                    bl_u, bl_v, bl_w = bl_vec / wavelength_m
                    delay = bl_u * dir_l + bl_v * dir_m + bl_w * (dir_n - 1.0)
                    phase = np.exp(-2j * np.pi * delay)

                    J_p = jones_cache[ant1]  # (n_vis, 2, 2)
                    J_q_H = np.conj(
                        np.swapaxes(jones_cache[ant2], -2, -1)
                    )  # (n_vis, 2, 2)

                    # V_all: (n_vis, 2, 2) = J_p @ C @ J_q^H * phase
                    V_all = J_p @ coherency @ J_q_H
                    V_all = V_all * phase[:, np.newaxis, np.newaxis]

                    visibilities[bl_idx, time_idx, freq_idx] = V_all.sum(axis=0)

            else:
                # ----- SCALAR PATH (unchanged) -----
                full_temp_map = sky_model.get_map_at_frequency(freq)
                temp_vis = full_temp_map[above_horizon]

                if output_units == "Jy":
                    conversion = getattr(sky_model, "brightness_conversion", "planck")
                    if conversion == "rayleigh-jeans":
                        rj_factor = (
                            (2 * K_BOLTZMANN * freq**2 / C_LIGHT**2)
                            * omega_pixel
                            * 1e26
                        )
                        signal = temp_vis * rj_factor
                    else:
                        pos = temp_vis > 0
                        signal = np.zeros(len(temp_vis))
                        if np.any(pos):
                            signal[pos] = brightness_temp_to_flux_density(
                                temp_vis[pos].astype(np.float64),
                                freq,
                                omega_pixel,
                                method="planck",
                            )
                else:
                    signal = temp_vis * omega_pixel

                # Compute per-antenna beam power patterns for this frequency
                beam_patterns: dict[Any, np.ndarray] = {}
                for ant_num in ant_nums:
                    ant_diameter = antennas.get(ant_num, {}).get("diameter", 14.0)
                    beam_patterns[ant_num] = _compute_beam_power_pattern(
                        zenith_angles=za_vis,
                        diameter=ant_diameter,
                        frequency=freq,
                        beam_manager=beam_manager if has_beam_manager else None,
                        antenna_number=ant_num,
                        azimuth=az_vis,
                        aperture_shape=bcfg.get("aperture_shape", "circular"),
                        taper=bcfg.get("taper", "gaussian"),
                        edge_taper_dB=bcfg.get("edge_taper_dB", 10.0),
                        feed_model=bcfg.get("feed_model", "none"),
                        feed_params=bcfg.get("feed_params"),
                        feed_computation=bcfg.get("feed_computation", "analytical"),
                        reflector_type=bcfg.get("reflector_type", "prime_focus"),
                        magnification=bcfg.get("magnification", 1.0),
                        aperture_params=bcfg.get("aperture_params"),
                    )

                # V = Σ B_pq * signal × exp(-2πi (ul + vm + w(n-1)))
                for bl_idx, ((ant1, ant2), bl_vec) in enumerate(
                    zip(baseline_keys, baseline_vectors, strict=False)
                ):
                    bl_u, bl_v, bl_w = bl_vec / wavelength_m
                    delay = bl_u * dir_l + bl_v * dir_m + bl_w * (dir_n - 1.0)
                    phase = np.exp(-2j * np.pi * delay)

                    B_pq = np.sqrt(beam_patterns[ant1] * beam_patterns[ant2])
                    beamed_signal = signal * B_pq

                    vis = np.sum(beamed_signal * phase)
                    visibilities[bl_idx, time_idx, freq_idx] = vis

        if time_idx % 10 == 0 or time_idx == n_times - 1:
            logger.debug(
                f"Time step {time_idx + 1}/{n_times}: {n_visible} pixels visible"
            )

    # Prepare output
    result = {
        "visibilities": visibilities,
        "times": times,
        "frequencies": freqs,
        "baseline_keys": baseline_keys,
        "n_baselines": n_baselines,
        "n_times": n_times,
        "n_freqs": n_freqs,
        "output_units": "Jy" if use_polarization else output_units,
        "polarized": use_polarization,
        "metadata": {
            "model": sky_model.model_name,
            "nside": nside,
            "n_pixels": npix,
            "pixel_solid_angle_sr": omega_pixel,
            "n_frequencies": n_freqs,
            "stokes": "IQUV" if use_polarization else "I",
        },
    }

    logger.info(
        f"HEALPix visibility calculation complete. "
        f"Output units: {'Jy' if use_polarization else output_units}, "
        f"mode: {pol_label}"
    )

    return result


def rayleigh_jeans_factor(frequency: float, solid_angle: float) -> float:
    """
    Compute the Rayleigh-Jeans conversion factor from T_b to Jy.

    S [Jy] = T [K] × factor

    Parameters
    ----------
    frequency : float
        Frequency in Hz.
    solid_angle : float
        Solid angle in steradians.

    Returns
    -------
    float
        Conversion factor (multiply temperature by this to get Jy).
    """
    return (2 * K_BOLTZMANN * (frequency**2) / (C_LIGHT**2)) * solid_angle * 1e26
