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

from typing import Any, Dict, Optional, Tuple, Union
import logging

import numpy as np
import healpy as hp
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import TimeDelta
import astropy.units as u

from rrivis.backends import get_backend, ArrayBackend
from rrivis.core.sky_model import SkyModel, K_BOLTZMANN, C_LIGHT


logger = logging.getLogger(__name__)


def calculate_visibility_healpix(
    sky_model: SkyModel,
    antennas: Dict,
    baselines: Dict,
    location: Any,
    obstime: Any,
    wavelengths: Any,
    freqs: Any,
    duration_seconds: float,
    time_step_seconds: float,
    hpbw_per_antenna: Optional[Dict] = None,
    beam_manager: Optional[Any] = None,
    backend: Optional[ArrayBackend] = None,
    output_units: str = "Jy",
) -> Dict:
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
    if sky_model.mode != "healpix":
        raise ValueError("sky_model must be in HEALPix mode. Use calculate_visibility() for point sources.")

    # Get backend
    if backend is None:
        backend = get_backend("numpy")

    # Get HEALPix data
    temp_map, nside, spec_idx_map = sky_model.get_healpix_data()
    npix = len(temp_map)
    omega_pixel = sky_model.pixel_solid_angle
    pixel_coords = sky_model.pixel_coords

    logger.info(f"HEALPix visibility calculation: nside={nside}, {npix} pixels")
    logger.info(f"Pixel solid angle: {omega_pixel:.6f} sr ({np.degrees(np.sqrt(omega_pixel)):.3f}°)")

    # Reference frequency for spectral scaling
    reference_freq = sky_model.frequency if sky_model.frequency else freqs[0]

    # Setup time steps
    n_times = int(np.ceil(duration_seconds / time_step_seconds))
    times = np.arange(n_times) * time_step_seconds

    # Setup baseline info
    baseline_keys = list(baselines.keys())
    n_baselines = len(baseline_keys)
    n_freqs = len(freqs)

    # Pre-compute baseline vectors in local ENU
    baseline_vectors = np.array([
        baselines[bl]["BaselineVector"] for bl in baseline_keys
    ])

    # Initialize output array
    visibilities = np.zeros((n_baselines, n_times, n_freqs), dtype=np.complex128)

    # Pre-compute spectral indices (default to 0 if not available)
    if spec_idx_map is not None:
        spectral_indices = spec_idx_map
    else:
        spectral_indices = np.zeros(npix)

    logger.info(f"Computing visibilities: {n_times} times × {n_freqs} freqs × {n_baselines} baselines")

    # ==========================================================================
    # TIME LOOP
    # ==========================================================================
    for time_idx in range(n_times):
        current_obstime = obstime + TimeDelta(time_step_seconds * time_idx, format='sec')

        # Transform pixel coordinates to AltAz
        altaz = pixel_coords.transform_to(AltAz(obstime=current_obstime, location=location))
        az_rad = altaz.az.rad
        alt_rad = altaz.alt.rad

        # Filter pixels above horizon
        above_horizon = alt_rad > 0
        if not np.any(above_horizon):
            continue

        n_visible = np.sum(above_horizon)

        # Get visible pixel data
        az_vis = az_rad[above_horizon]
        alt_vis = alt_rad[above_horizon]
        temp_vis = temp_map[above_horizon]
        spec_idx_vis = spectral_indices[above_horizon]

        # Compute direction cosines (l, m, n) in local ENU frame
        # l = East, m = North, n = Up (zenith)
        l = np.cos(alt_vis) * np.sin(az_vis)
        m = np.cos(alt_vis) * np.cos(az_vis)
        n = np.sin(alt_vis)

        # Stack direction cosines for vectorized computation
        direction_vectors = np.column_stack([l, m, n])  # (n_visible, 3)

        # ======================================================================
        # FREQUENCY LOOP
        # ======================================================================
        for freq_idx, (wavelength, freq) in enumerate(zip(wavelengths, freqs)):
            wavelength_m = wavelength.to(u.m).value

            # Scale temperature by spectral index (T ∝ ν^β, where β = α - 2)
            # Since we store flux spectral index α, temperature index β = α - 2
            beta = spec_idx_vis - 2.0
            temp_scaled = temp_vis * (freq / reference_freq) ** beta

            # Compute visibility for each baseline
            # V = Σ T × exp(-2πi (b·ŝ) / λ)
            for bl_idx, (bl_key, bl_vec) in enumerate(zip(baseline_keys, baseline_vectors)):
                # Geometric delay: b · ŝ / λ (in wavelengths)
                # bl_vec is in meters, ŝ is unit direction vector
                delay = np.dot(direction_vectors, bl_vec) / wavelength_m

                # Phase factor
                phase = np.exp(-2j * np.pi * delay)

                # Sum over all visible pixels
                # This gives visibility in K (temperature units)
                vis_temp = np.sum(temp_scaled * phase)

                visibilities[bl_idx, time_idx, freq_idx] = vis_temp

            # Convert to output units
            if output_units == "Jy":
                # Apply Rayleigh-Jeans factor: S = (2kTν²/c²) × Ω × 10^26
                rj_factor = (2 * K_BOLTZMANN * (freq ** 2) / (C_LIGHT ** 2)) * omega_pixel * 1e26
                visibilities[:, time_idx, freq_idx] *= rj_factor

        if time_idx % 10 == 0 or time_idx == n_times - 1:
            logger.debug(f"Time step {time_idx + 1}/{n_times}: {n_visible} pixels visible")

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
            "reference_freq_hz": reference_freq,
        }
    }

    logger.info(f"HEALPix visibility calculation complete. Output units: {output_units}")

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
    return (2 * K_BOLTZMANN * (frequency ** 2) / (C_LIGHT ** 2)) * solid_angle * 1e26
