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
from rrivis.core.sky import (
    C_LIGHT,
    K_BOLTZMANN,
    SkyModel,
    brightness_temp_to_flux_density,
)

logger = logging.getLogger(__name__)


def _compute_beam_power_pattern(
    zenith_angles: np.ndarray,
    hpbw_rad: float,
    beam_type: str = "gaussian",
    beam_manager: Any | None = None,
    antenna_number: Any | None = None,
    azimuth: np.ndarray | None = None,
    frequency: float | None = None,
) -> np.ndarray:
    """Compute scalar beam power pattern B^2 for HEALPix pixels.

    For analytic beams: B^2 = pattern_function(za, hpbw)^2
    For FITS beams: B^2 = 0.5 * (|J_00|^2 + |J_01|^2 + |J_10|^2 + |J_11|^2)

    Parameters
    ----------
    zenith_angles : ndarray
        Zenith angles in radians, shape (N,).
    hpbw_rad : float
        Half-power beam width in radians (for analytic beams).
    beam_type : str
        Analytic beam type: 'gaussian', 'airy', 'cosine', 'exponential'.
    beam_manager : BeamManager, optional
        BeamManager for FITS beam lookup.
    antenna_number : int, optional
        Antenna number for beam_manager lookup.
    azimuth : ndarray, optional
        Azimuth angles in radians, shape (N,). Required for FITS beams.
    frequency : float, optional
        Frequency in Hz. Required for FITS beams and 'airy' beam.

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

    # Analytic beam fallback
    from rrivis.core.jones.beam.analytic import (
        cosine_tapered_pattern,
        exponential_tapered_pattern,
        gaussian_A_theta_EBeam,
    )

    if beam_type == "gaussian":
        voltage = gaussian_A_theta_EBeam(zenith_angles, hpbw_rad)
    elif beam_type == "cosine":
        voltage = cosine_tapered_pattern(zenith_angles, hpbw_rad)
    elif beam_type == "exponential":
        voltage = exponential_tapered_pattern(zenith_angles, hpbw_rad)
    else:
        voltage = gaussian_A_theta_EBeam(zenith_angles, hpbw_rad)

    return np.asarray(voltage, dtype=np.float64) ** 2


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
    hpbw_per_antenna: dict | None = None,
    beam_manager: Any | None = None,
    beam_pattern_per_antenna: dict | None = None,
    backend: ArrayBackend | None = None,
    output_units: str = "Jy",
) -> dict:
    """
    Calculate visibility directly from HEALPix brightness temperature map.

    This function computes visibilities using the direct sum over HEALPix pixels,
    working in brightness temperature and applying the Rayleigh-Jeans conversion
    at the end.

    V(b, ν) = (2kν²/c²) × Ω_pixel × Σ_pixels T(p) × exp(-2πi b·ŝ(p) / λ)

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
    hpbw_per_antenna : dict, optional
        Half-power beam width per antenna (for beam attenuation).
    beam_manager : BeamManager, optional
        Beam pattern manager for FITS-based beams.
    beam_pattern_per_antenna : dict, optional
        Maps antenna number -> beam pattern type string ('gaussian', 'airy', etc.).
    backend : ArrayBackend, optional
        Computation backend (CPU/GPU).
    output_units : str, default="Jy"
        Output units: "Jy" (convert to Jansky) or "K.sr" (keep temperature × solid angle).

    Returns
    -------
    dict
        Dictionary containing:
        - visibilities: Complex visibility array (n_baselines, n_times, n_freqs)
        - times: Time array
        - frequencies: Frequency array
        - baselines: Baseline info
        - metadata: Additional information
    """
    if sky_model.mode != "healpix_multifreq":
        raise ValueError(
            "sky_model must be in healpix_multifreq mode. "
            "Call to_healpix_for_observation() first (for point-source catalogs) "
            "or use from_diffuse_sky(frequencies=...) (for diffuse models)."
        )

    # Get backend
    if backend is None:
        backend = get_backend("numpy")

    # Get multi-frequency map metadata
    _, nside, _ = sky_model.get_multifreq_maps()
    npix = hp.nside2npix(nside)
    omega_pixel = sky_model.pixel_solid_angle
    pixel_coords = sky_model.pixel_coords

    logger.info(f"HEALPix visibility calculation: nside={nside}, {npix} pixels")
    logger.info(
        f"Pixel solid angle: {omega_pixel:.6f} sr ({np.degrees(np.sqrt(omega_pixel)):.3f}°)"
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
    visibilities = np.zeros((n_baselines, n_times, n_freqs), dtype=np.complex128)

    logger.info(
        f"Computing visibilities: {n_times} times × {n_freqs} freqs × {n_baselines} baselines"
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

        # Compute per-antenna beam power patterns
        ant_nums = set()
        for ant1, ant2 in baselines:
            ant_nums.add(ant1)
            ant_nums.add(ant2)

        beam_patterns: dict[Any, np.ndarray] = {}
        has_beam_manager = (
            beam_manager is not None
            and getattr(beam_manager, "mode", "analytic") != "analytic"
        )

        # ======================================================================
        # FREQUENCY LOOP
        # ======================================================================
        for freq_idx, (wavelength, freq) in enumerate(
            zip(wavelengths, freqs, strict=False)
        ):
            wavelength_m = wavelength.to(u.m).value

            # Look up the pre-computed native map for this frequency channel
            full_temp_map = sky_model.get_map_at_frequency(freq)
            temp_vis = full_temp_map[above_horizon]

            # Convert to output units before baseline loop
            # (Planck conversion is nonlinear in T, so must be per-pixel)
            if output_units == "Jy":
                conversion = getattr(sky_model, "brightness_conversion", "planck")
                if conversion == "rayleigh-jeans":
                    rj_factor = (
                        (2 * K_BOLTZMANN * freq**2 / C_LIGHT**2) * omega_pixel * 1e26
                    )
                    signal = temp_vis * rj_factor
                else:
                    # Handle zero-temperature pixels gracefully
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
            for ant_num in ant_nums:
                if hpbw_per_antenna is not None:
                    hpbw_rad = hpbw_per_antenna.get(ant_num, np.array([0.1]))[freq_idx]
                else:
                    hpbw_rad = 0.1
                btype = (
                    beam_pattern_per_antenna.get(ant_num, "gaussian")
                    if beam_pattern_per_antenna
                    else "gaussian"
                )
                beam_patterns[ant_num] = _compute_beam_power_pattern(
                    zenith_angles=za_vis,
                    hpbw_rad=hpbw_rad,
                    beam_type=btype,
                    beam_manager=beam_manager if has_beam_manager else None,
                    antenna_number=ant_num,
                    azimuth=az_vis,
                    frequency=freq,
                )

            # Compute visibility for each baseline
            # V = Σ B_pq * signal × exp(-2πi (ul + vm + w(n-1)))
            # Using w(n-1) formulation per Smirnov 2011 RIME
            for bl_idx, ((ant1, ant2), bl_vec) in enumerate(
                zip(baseline_keys, baseline_vectors, strict=False)
            ):
                # Baseline in wavelengths
                bl_u, bl_v, bl_w = bl_vec / wavelength_m

                # Geometric delay with w(n-1) correction
                # This ensures zero phase at phase center (l=0, m=0, n=1)
                delay = bl_u * dir_l + bl_v * dir_m + bl_w * (dir_n - 1.0)

                # Phase factor
                phase = np.exp(-2j * np.pi * delay)

                # Apply beam: geometric mean of antenna power patterns
                B_pq = np.sqrt(beam_patterns[ant1] * beam_patterns[ant2])
                beamed_signal = signal * B_pq

                # Sum over all visible pixels
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
        "output_units": output_units,
        "metadata": {
            "model": sky_model.model_name,
            "nside": nside,
            "n_pixels": npix,
            "pixel_solid_angle_sr": omega_pixel,
            "n_frequencies": n_freqs,
        },
    }

    logger.info(
        f"HEALPix visibility calculation complete. Output units: {output_units}"
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
