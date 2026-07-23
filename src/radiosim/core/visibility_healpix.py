# radiosim/core/visibility_healpix.py
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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radiosim.core.instrument_adapters import SolverInstrumentView

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz
from astropy.time import TimeDelta

from radiosim.backends import ArrayBackend, get_backend
from radiosim.core.beam import BeamSystem
from radiosim.core.instrument import AntennaId
from radiosim.core.instrument_adapters import InstrumentAdapterInvariantError
from radiosim.core.polarization import stokes_to_coherency
from radiosim.core.sky import (
    SkyModel,
    brightness_temp_to_flux_density,
    rayleigh_jeans_factor,
)

logger = logging.getLogger(__name__)


def _canonical_antenna_id(
    instrument: "SolverInstrumentView",
    antenna_number: int,
) -> AntennaId:
    """Resolve one exact canonical antenna identity from the solver view."""
    row = instrument.row_for_number(antenna_number)
    try:
        name = instrument.antenna_names[row]
    except IndexError as exc:
        raise InstrumentAdapterInvariantError(
            f"antenna row {row} has no canonical name in the solver instrument view"
        ) from exc
    return AntennaId(antenna_number, name)


def _evaluate_beam_batch_by_antenna(
    *,
    beam_system: BeamSystem,
    instrument: "SolverInstrumentView",
    antenna_numbers: tuple[int, ...],
    altitude_rad: np.ndarray,
    azimuth_rad: np.ndarray,
    frequency_hz: float,
    time_mjd: float,
    backend: ArrayBackend,
) -> dict[int, Any]:
    """Evaluate every selected handler once and share it by canonical ID."""
    handler_by_antenna = dict(beam_system.state.assignment_handler_ids)
    handler_cache: dict[str, Any] = {}
    result: dict[int, Any] = {}
    for antenna_number in antenna_numbers:
        antenna_id = _canonical_antenna_id(instrument, antenna_number)
        try:
            handler_id = handler_by_antenna[antenna_id]
        except KeyError as exc:
            raise InstrumentAdapterInvariantError(
                "BeamSystem assignment state does not cover solver antenna "
                f"number={antenna_id.number}, name={antenna_id.name!r}"
            ) from exc
        if handler_id not in handler_cache:
            handler_cache[handler_id] = beam_system.evaluate_jones(
                antenna_id,
                altitude_rad=np.array(
                    altitude_rad,
                    dtype=np.float64,
                    copy=True,
                    order="C",
                ),
                azimuth_rad=np.array(
                    azimuth_rad,
                    dtype=np.float64,
                    copy=True,
                    order="C",
                ),
                frequency_hz=float(frequency_hz),
                time_mjd=float(time_mjd),
                backend=backend,
            )
        result[antenna_number] = handler_cache[handler_id]
    return result


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
    instrument: "SolverInstrumentView",
    beam_system: BeamSystem,
    location: Any,
    obstime: Any,
    wavelengths: Any,
    freqs: Any,
    duration_seconds: float,
    time_step_seconds: float,
    output_units: str = "Jy",
    include_polarization: bool = False,
    backend: ArrayBackend | None = None,
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
    instrument : SolverInstrumentView
        Owned canonical antenna values and selected baseline geometry.
    beam_system : BeamSystem
        Canonical per-antenna beam evaluator.
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
    output_units : str, default="Jy"
        Output units: "Jy" (convert to Jansky) or "K.sr" (keep temperature ×
        solid angle). In polarized mode, always "Jy".
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
    if sky_model.healpix is None:
        raise ValueError(
            "sky_model must contain a HEALPix payload. "
            "Materialize a HEALPix payload first (for point-source catalogs) "
            "or load a diffuse HEALPix model with frequencies=...."
        )
    from radiosim.core.instrument_adapters import SolverInstrumentView

    if type(instrument) is not SolverInstrumentView:
        raise TypeError("instrument must be a SolverInstrumentView")
    if type(beam_system) is not BeamSystem:
        raise TypeError("beam_system must be an exact BeamSystem")
    if backend is None:
        backend = get_backend("numpy")
    xp = backend.xp

    # Determine if we should use the polarized path
    use_polarization = include_polarization and sky_model.has_polarized_healpix_maps

    # Get multi-frequency map metadata
    nside = sky_model.healpix.nside
    omega_pixel = sky_model.healpix.pixel_solid_angle
    pixel_coords = sky_model.healpix.pixel_coords
    n_pixels = len(pixel_coords)

    pol_label = "polarized (2x2 RIME)" if use_polarization else "scalar"
    logger.info(
        f"HEALPix visibility calculation: nside={nside}, {n_pixels} pixels, {pol_label}"
    )
    logger.info(
        f"Pixel solid angle: {omega_pixel:.6f} sr ({np.degrees(np.sqrt(omega_pixel)):.3f}\u00b0)"
    )

    # Setup time steps
    n_times = int(np.ceil(duration_seconds / time_step_seconds))
    times = np.arange(n_times) * time_step_seconds

    # Setup baseline info
    baseline_keys = instrument.selected_pairs
    n_baselines = len(baseline_keys)
    n_freqs = len(freqs)

    # Pre-compute baseline vectors in local ENU
    baseline_vectors = backend.asarray(
        instrument.baseline_vectors_enu_m,
        dtype=backend.default_real_dtype,
    )

    # Initialize output array
    if use_polarization:
        visibilities = backend.zeros_complex((n_baselines, n_times, n_freqs, 2, 2))
    else:
        visibilities = backend.zeros_complex((n_baselines, n_times, n_freqs))

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
        dir_l_xp = backend.asarray(dir_l, dtype=backend.default_real_dtype)
        dir_m_xp = backend.asarray(dir_m, dtype=backend.default_real_dtype)
        dir_n_xp = backend.asarray(dir_n, dtype=backend.default_real_dtype)

        # Collect selected antenna numbers in canonical instrument order.
        selected_numbers = {number for pair in baseline_keys for number in pair}
        ant_nums = tuple(
            number
            for number in instrument.antenna_numbers
            if number in selected_numbers
        )

        # ======================================================================
        # FREQUENCY LOOP
        # ======================================================================
        for freq_idx, (wavelength, freq) in enumerate(
            zip(wavelengths, freqs, strict=True)
        ):
            wavelength_m = wavelength.to(u.m).value
            jones_cache = _evaluate_beam_batch_by_antenna(
                beam_system=beam_system,
                instrument=instrument,
                antenna_numbers=ant_nums,
                altitude_rad=alt_vis,
                azimuth_rad=az_vis,
                frequency_hz=float(freq),
                time_mjd=float(current_obstime.mjd),
                backend=backend,
            )

            if use_polarization:
                # ----- POLARIZED PATH -----
                # Get all Stokes maps at this frequency
                I_map, Q_map, U_map, V_map = (
                    sky_model.healpix.get_stokes_maps_at_frequency(freq)
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
                    rj_factor_I = rayleigh_jeans_factor(freq, omega_pixel)
                    I_jy = (
                        backend.asarray(I_vis, dtype=backend.default_real_dtype)
                        * rj_factor_I
                    )
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
                    I_jy = backend.asarray(I_jy, dtype=backend.default_real_dtype)

                # Stokes Q/U/V: always RJ (can be negative, RJ is linear)
                rj_factor_pol = rayleigh_jeans_factor(freq, omega_pixel)
                Q_jy = backend.asarray(Q_vis, dtype=backend.default_real_dtype) * (
                    rj_factor_pol
                )
                U_jy = backend.asarray(U_vis, dtype=backend.default_real_dtype) * (
                    rj_factor_pol
                )
                V_jy = backend.asarray(V_vis, dtype=backend.default_real_dtype) * (
                    rj_factor_pol
                )

                # Build per-pixel coherency matrices: (n_visible, 2, 2)
                coherency = stokes_to_coherency(I_jy, Q_jy, U_jy, V_jy, xp=xp)

                # Compute visibility for each baseline
                # V_pq = Σ_pix phase_pix * J_p @ C_pix @ J_q^H
                for bl_idx, ((ant1, ant2), bl_vec) in enumerate(
                    zip(baseline_keys, baseline_vectors, strict=True)
                ):
                    bl_u, bl_v, bl_w = bl_vec / wavelength_m
                    delay = bl_u * dir_l_xp + bl_v * dir_m_xp + bl_w * (dir_n_xp - 1.0)
                    phase = backend.exp(-2j * np.pi * delay)

                    J_p = jones_cache[ant1]  # (n_vis, 2, 2)
                    J_q_H = backend.conjugate_transpose(jones_cache[ant2])

                    # V_all: (n_vis, 2, 2) = J_p @ C @ J_q^H * phase
                    V_all = backend.matmul(backend.matmul(J_p, coherency), J_q_H)
                    V_all = V_all * phase[:, None, None]

                    visibilities = backend.set_at(
                        visibilities,
                        (bl_idx, time_idx, freq_idx),
                        backend.sum(V_all, axis=0),
                    )

            else:
                # ----- I-ONLY PATH -----
                full_temp_map = sky_model.healpix.get_map_at_frequency(freq)
                temp_vis = full_temp_map[above_horizon]

                if output_units == "Jy":
                    conversion = getattr(sky_model, "brightness_conversion", "planck")
                    if conversion == "rayleigh-jeans":
                        rj_factor = rayleigh_jeans_factor(freq, omega_pixel)
                        signal = (
                            backend.asarray(temp_vis, dtype=backend.default_real_dtype)
                            * rj_factor
                        )
                    else:
                        pos = temp_vis > 0
                        signal_np = np.zeros(len(temp_vis))
                        if np.any(pos):
                            signal_np[pos] = brightness_temp_to_flux_density(
                                temp_vis[pos].astype(np.float64),
                                freq,
                                omega_pixel,
                                method="planck",
                            )
                        signal = backend.asarray(
                            signal_np, dtype=backend.default_real_dtype
                        )
                else:
                    signal = (
                        backend.asarray(temp_vis, dtype=backend.default_real_dtype)
                        * omega_pixel
                    )

                # Construct C=(I/2)I2, then apply the complete matrix RIME.
                coherency = backend.batch_eye(
                    (len(temp_vis),),
                    2,
                    dtype=backend.default_complex_dtype,
                )
                coherency = coherency * (signal / 2.0)[:, None, None]
                for bl_idx, ((ant1, ant2), bl_vec) in enumerate(
                    zip(baseline_keys, baseline_vectors, strict=True)
                ):
                    bl_u, bl_v, bl_w = bl_vec / wavelength_m
                    delay = bl_u * dir_l_xp + bl_v * dir_m_xp + bl_w * (dir_n_xp - 1.0)
                    phase = backend.exp(-2j * np.pi * delay)
                    J_p = jones_cache[ant1]
                    J_q_H = backend.conjugate_transpose(jones_cache[ant2])
                    V_all = backend.matmul(
                        backend.matmul(J_p, coherency),
                        J_q_H,
                    )
                    V_all = V_all * phase[:, None, None]
                    matrix = backend.sum(V_all, axis=0)
                    vis = matrix[0, 0] + matrix[1, 1]
                    visibilities = backend.set_at(
                        visibilities, (bl_idx, time_idx, freq_idx), vis
                    )

        if time_idx % 10 == 0 or time_idx == n_times - 1:
            logger.debug(
                f"Time step {time_idx + 1}/{n_times}: {n_visible} pixels visible"
            )

    # Prepare output
    result = {
        "visibilities": backend.to_numpy(visibilities),
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
            "n_pixels": n_pixels,
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
