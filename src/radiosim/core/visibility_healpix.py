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
    from radiosim.core.time_grid import ObservationTimeGrid

import numpy as np
from astropy.coordinates import AltAz

from radiosim.backends import ArrayBackend
from radiosim.core.beam import BeamSystem
from radiosim.core.instrument import AntennaId
from radiosim.core.instrument_adapters import InstrumentAdapterInvariantError
from radiosim.core.jones.receptor import basis_transform_matrix, receptor_matrix
from radiosim.core.polarization import stokes_to_coherency
from radiosim.core.receptor import ResolvedReceptorSet
from radiosim.core.runtime_config import ResolvedSolverExecutionConfig
from radiosim.core.sky import (
    SkyModel,
    brightness_temp_to_flux_density,
    rayleigh_jeans_factor,
)
from radiosim.core.sky.containers.constants import C_LIGHT
from radiosim.core.solver_partition import (
    SERIAL_SOLVER_EXECUTION,
    execute_time_blocks,
    require_solver_execution,
)

logger = logging.getLogger(__name__)


def _require_backend(backend: object) -> ArrayBackend:
    if not isinstance(backend, ArrayBackend):
        raise TypeError("backend must be an ArrayBackend")
    return backend


def _require_frequencies(frequencies: object) -> np.ndarray:
    if type(frequencies) is not np.ndarray:
        raise TypeError("frequencies must be an exact numpy.ndarray")
    if (
        frequencies.dtype != np.dtype("float64")
        or frequencies.ndim != 1
        or frequencies.size == 0
        or not np.all(np.isfinite(frequencies))
        or not np.all(frequencies > 0.0)
        or not np.all(np.diff(frequencies) > 0.0)
    ):
        raise ValueError(
            "frequencies must be a nonempty, finite, positive, strictly "
            "increasing float64 array"
        )
    return frequencies


def _require_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")
    return value


def _require_receptors(receptors: object) -> ResolvedReceptorSet:
    if type(receptors) is not ResolvedReceptorSet:
        raise TypeError(
            "receptors must be an exact ResolvedReceptorSet from resolve_receptors()"
        )
    return receptors


def _receptor_transforms(
    *,
    receptors: ResolvedReceptorSet,
    instrument: "SolverInstrumentView",
    antenna_numbers: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Return the constant ``H_p @ C_p`` for each selected antenna.

    The HEALPix path builds no Jones chain; it evaluates the beam directly.
    ``C`` and ``H`` are direction, time, and frequency
    independent, so left-multiplying the per-antenna beam Jones by the single
    constant product is exactly the canonical chain restricted to the terms
    this path carries (``Tier5ReceptorFeedPlan.md`` Section 19.3).
    """
    transforms: dict[int, np.ndarray] = {}
    for antenna_number in antenna_numbers:
        antenna_id = _canonical_antenna_id(instrument, antenna_number)
        try:
            receptor = receptors.receptor_by_antenna[antenna_id]
        except KeyError as exc:
            raise InstrumentAdapterInvariantError(
                "the resolved receptor set does not cover solver antenna "
                f"number={antenna_id.number}, name={antenna_id.name!r}"
            ) from exc
        transforms[antenna_number] = basis_transform_matrix(
            receptor.basis,
            receptors.output_basis,
        ) @ receptor_matrix(receptor.basis, receptor.feed_rotation_rad)
    return transforms


def _host_preprocess_time_step(
    pixel_coords: Any,
    obstime: Any,
    location: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Named host-preprocessing stage for one time step.

    ``Tier6HybridRuntimePlan.md`` Section 13.3 requires this stage to be named
    rather than left implicit: it is the boundary below which the solver runs on
    the host regardless of the selected backend, because it calls into astropy's
    ``ICRS`` -> ``AltAz`` machinery, which has no backend-agnostic form. Its
    outputs -- the horizon mask and the two visible-direction angle arrays --
    are host ``float64`` arrays that the caller hands to ``backend.asarray``
    exactly once per time step.

    Returns
    -------
    tuple of ndarray
        ``(above_horizon, altitude_rad_visible, azimuth_rad_visible)``.
    """
    altaz = pixel_coords.transform_to(AltAz(obstime=obstime, location=location))
    az_rad = altaz.az.rad
    alt_rad = altaz.alt.rad
    above_horizon = alt_rad > 0
    return above_horizon, alt_rad[above_horizon], az_rad[above_horizon]


def _host_direction_cosines(
    altitude_rad: np.ndarray,
    azimuth_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direction cosines ``(l, m, n)`` in the local ENU frame, on the host.

    The second half of the Section 13.3 host-preprocessing stage: a fixed-cost
    trigonometric transform of the astropy output, evaluated once per time step
    and then transferred once. ``l`` is East, ``m`` is North, ``n`` is Up.
    """
    return (
        np.cos(altitude_rad) * np.sin(azimuth_rad),
        np.cos(altitude_rad) * np.cos(azimuth_rad),
        np.sin(altitude_rad),
    )


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
    receptor_transforms: dict[int, np.ndarray],
) -> dict[int, Any]:
    """Evaluate every selected handler once and share it by canonical ID.

    The shared beam evaluation is deduplicated by handler, but the receptor
    factor is per antenna, so ``H_p @ C_p`` is applied after the lookup.
    """
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
        beam_jones = handler_cache[handler_id]
        transform = backend.asarray(
            receptor_transforms[antenna_number],
            dtype=beam_jones.dtype,
        )
        result[antenna_number] = backend.matmul(transform, beam_jones)
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
    time_grid: "ObservationTimeGrid",
    frequencies: Any,
    backend: ArrayBackend,
    receptors: ResolvedReceptorSet,
    output_units: str = "Jy",
    include_polarization: bool = False,
    solver_execution: ResolvedSolverExecutionConfig = SERIAL_SOLVER_EXECUTION,
) -> Any:
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
    time_grid : ObservationTimeGrid
        Exact canonical UTC sample-center grid.
    frequencies : ndarray
        Canonical frequency centers in Hz.
    receptors : ResolvedReceptorSet
        Canonical resolved receptor inventory from ``resolve_receptors()``.
        Every antenna's beam Jones is left-multiplied by the constant
        ``H_p @ C_p``, in both the polarized and the scalar path, so cross-hand
        outputs are zero in the reported basis rather than by assumption.
    output_units : str, default="Jy"
        Output units: "Jy" (convert to Jansky) or "K.sr" (keep temperature ×
        solid angle). In polarized mode, always "Jy".
    include_polarization : bool, default=False
        If True and sky model has polarized HEALPix maps, compute full 2×2
        visibility matrices using the RIME with Jones beam matrices. Output
        The returned cube retains the same canonical shape in both paths.
    solver_execution : ResolvedSolverExecutionConfig, optional
        Resolved solver worker policy. ``workers=1`` (the default) is the exact
        serial path; ``workers=N`` distributes contiguous time blocks over a
        thread pool of ``N`` threads and reassembles them in time order, which
        is bit-identical to the serial result for every ``N``
        (``Tier6HybridRuntimePlan.md`` Sections 11.3, 11.5).

    Returns
    -------
    backend array
        Receptor visibility cube with shape ``(T, B, F, 2, 2)``.
    """
    from radiosim.core.instrument_adapters import SolverInstrumentView
    from radiosim.core.time_grid import ObservationTimeGrid

    if type(instrument) is not SolverInstrumentView:
        raise TypeError("instrument must be a SolverInstrumentView")
    if type(beam_system) is not BeamSystem:
        raise TypeError("beam_system must be an exact BeamSystem")
    if type(time_grid) is not ObservationTimeGrid:
        raise TypeError("time_grid must be an exact ObservationTimeGrid")
    backend = _require_backend(backend)
    frequencies = _require_frequencies(frequencies)
    if type(output_units) is not str or output_units not in {"Jy", "K.sr"}:
        raise ValueError("output_units must be 'Jy' or 'K.sr'")
    include_polarization = _require_bool(
        include_polarization,
        field_name="include_polarization",
    )
    receptors = _require_receptors(receptors)
    solver_execution = require_solver_execution(solver_execution)
    if sky_model.healpix is None:
        raise ValueError(
            "sky_model must contain a HEALPix payload. "
            "Materialize a HEALPix payload first (for point-source catalogs) "
            "or load a diffuse HEALPix model with frequencies=...."
        )
    xp = backend.xp
    output_complex_dtype = backend.get_complex_dtype("output")

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

    n_times = len(time_grid)
    sample_times = time_grid.as_astropy()

    # Setup baseline info
    selected_pairs = instrument.selected_pairs
    n_baselines = len(selected_pairs)
    n_freqs = len(frequencies)

    # Pre-compute baseline vectors in local ENU
    baseline_vectors = backend.asarray(
        instrument.baseline_vectors_enu_m,
        dtype=backend.default_real_dtype,
    )

    logger.info(
        f"Computing visibilities: {n_times} times \u00d7 {n_freqs} freqs "
        f"\u00d7 {n_baselines} baselines"
    )

    # Any degenerate axis leaves nothing to assemble.
    if n_times == 0 or n_baselines == 0 or n_freqs == 0:
        return backend.zeros_complex(
            (n_times, n_baselines, n_freqs, 2, 2),
            dtype=output_complex_dtype,
        )

    # ``C`` and ``H`` are direction, time, and frequency independent, so the
    # constant ``H_p @ C_p`` product is built once for the whole call rather
    # than rebuilt inside the time loop (defect D12, Section 13.2).
    selected_numbers = {number for pair in selected_pairs for number in pair}
    ant_nums = tuple(
        number for number in instrument.antenna_numbers if number in selected_numbers
    )
    receptor_transforms = _receptor_transforms(
        receptors=receptors,
        instrument=instrument,
        antenna_numbers=ant_nums,
    )

    # ==========================================================================
    # TIME LOOP
    #
    # Output accumulation follows Section 13.3 of Tier6HybridRuntimePlan.md:
    # one (B, 2, 2) block per (time, frequency), one (B, F, 2, 2) block per
    # time, and exactly one (T, B, F, 2, 2) assembly per call.
    #
    # Every time step reads only run-constant inputs and produces its own
    # independent block, so contiguous time ranges are exactly the unit of
    # solver worker parallelism (Section 11.3/11.4).
    # ==========================================================================
    def _time_block(time_idx: int, empty_block: list[Any]) -> Any:
        """Compute the ``(B, F, 2, 2)`` output block for one time index."""
        current_obstime = sample_times[time_idx]

        # ---- host preprocessing (named stage; see _host_preprocess_time_step)
        above_horizon, alt_vis, az_vis = _host_preprocess_time_step(
            pixel_coords,
            current_obstime,
            location,
        )
        if not np.any(above_horizon):
            # Contribute an exactly zero block so the time axis keeps its slot.
            if not empty_block:
                empty_block.append(
                    backend.zeros_complex(
                        (n_baselines, n_freqs, 2, 2),
                        dtype=output_complex_dtype,
                    )
                )
            return empty_block[0]

        n_visible = np.sum(above_horizon)

        dir_l, dir_m, dir_n = _host_direction_cosines(alt_vis, az_vis)
        dir_l_xp = backend.asarray(dir_l, dtype=backend.default_real_dtype)
        dir_m_xp = backend.asarray(dir_m, dtype=backend.default_real_dtype)
        dir_n_xp = backend.asarray(dir_n, dtype=backend.default_real_dtype)

        # ======================================================================
        # FREQUENCY LOOP
        # ======================================================================
        freq_blocks: list[Any] = []
        for freq in frequencies:
            wavelength_m = float(C_LIGHT) / float(freq)
            jones_cache = _evaluate_beam_batch_by_antenna(
                beam_system=beam_system,
                instrument=instrument,
                antenna_numbers=ant_nums,
                altitude_rad=alt_vis,
                azimuth_rad=az_vis,
                frequency_hz=float(freq),
                time_mjd=float(current_obstime.mjd),
                backend=backend,
                receptor_transforms=receptor_transforms,
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
                baseline_matrices: list[Any] = []
                for (ant1, ant2), bl_vec in zip(
                    selected_pairs, baseline_vectors, strict=True
                ):
                    bl_u, bl_v, bl_w = bl_vec / wavelength_m
                    delay = bl_u * dir_l_xp + bl_v * dir_m_xp + bl_w * (dir_n_xp - 1.0)
                    phase = backend.exp(-2j * np.pi * delay)

                    J_p = jones_cache[ant1]  # (n_vis, 2, 2)
                    J_q_H = backend.conjugate_transpose(jones_cache[ant2])

                    # V_all: (n_vis, 2, 2) = J_p @ C @ J_q^H * phase
                    V_all = backend.matmul(backend.matmul(J_p, coherency), J_q_H)
                    V_all = V_all * phase[:, None, None]

                    baseline_matrices.append(
                        backend.asarray(
                            backend.sum(V_all, axis=0),
                            dtype=output_complex_dtype,
                        )
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
                baseline_matrices = []
                for (ant1, ant2), bl_vec in zip(
                    selected_pairs, baseline_vectors, strict=True
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
                    matrix = backend.asarray(
                        backend.sum(V_all, axis=0),
                        dtype=output_complex_dtype,
                    )
                    baseline_matrices.append(matrix)

            # One (B, 2, 2) block for all baselines at this (time, frequency).
            freq_blocks.append(backend.stack(baseline_matrices, axis=0))

        if time_idx % 10 == 0 or time_idx == n_times - 1:
            logger.debug(
                f"Time step {time_idx + 1}/{n_times}: {n_visible} pixels visible"
            )

        # One (B, F, 2, 2) block for this time step.
        return backend.stack(freq_blocks, axis=1)

    def _time_range(start: int, stop: int) -> list[Any]:
        """Compute one contiguous worker share of the time axis, in order."""
        empty_block: list[Any] = []
        return [_time_block(time_idx, empty_block) for time_idx in range(start, stop)]

    time_blocks = execute_time_blocks(
        _time_range,
        n_times=n_times,
        solver_execution=solver_execution,
        thread_name_prefix="radiosim-healpix-solver",
    )

    logger.info(
        f"HEALPix visibility calculation complete. "
        f"Output units: {'Jy' if use_polarization else output_units}, "
        f"mode: {pol_label}"
    )

    # One (T, B, F, 2, 2) cube, assembled in a single operation.
    return backend.stack(time_blocks, axis=0)
