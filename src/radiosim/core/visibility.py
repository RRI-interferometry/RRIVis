# radiosim/core/visibility.py
"""
Visibility calculation using the Radio Interferometer Measurement Equation (RIME).

Implements the direct-sum Jones/coherency calculation used by the high-level
analytic-beam path. Backend selection is explicit, but this module does not
claim complete accelerator coverage for the full simulation workflow.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radiosim.core.instrument_adapters import SolverInstrumentView
    from radiosim.core.sky.containers.model import SourceArrays
    from radiosim.core.time_grid import ObservationTimeGrid

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from typing_extensions import override

# Import backend abstraction
from radiosim.backends import ArrayBackend
from radiosim.core.beam import BeamSystem
from radiosim.core.contraction import baseline_contraction_for
from radiosim.core.instrument import AntennaId
from radiosim.core.instrument_adapters import InstrumentAdapterInvariantError

# Import Jones matrix framework
from radiosim.core.jones import (
    BandpassJones,
    GainJones,
    IonosphereJones,
    JonesChain,
    JonesTerm,
    ParallacticAngleJones,
    PolarizationLeakageJones,
    TroposphereJones,
)
from radiosim.core.jones.directions import DirectionBatch
from radiosim.core.jones.evaluate import evaluate_antenna_jones
from radiosim.core.jones.geometric import geometric_phase, uvw_in_wavelengths
from radiosim.core.jones.receptor import BasisTransformJones, ReceptorConfigJones

# Import polarization utilities
from radiosim.core.polarization import (
    stokes_to_coherency,
)
from radiosim.core.receptor import (
    ResolvedReceptorSet,
    UnsupportedFeedGeometryError,
)
from radiosim.core.runtime_config import ResolvedSolverExecutionConfig
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


def _require_receptors(receptors: object) -> ResolvedReceptorSet:
    if type(receptors) is not ResolvedReceptorSet:
        raise TypeError(
            "receptors must be an exact ResolvedReceptorSet from resolve_receptors()"
        )
    return receptors


def _reject_parallactic_rotation(
    jones_config: Any,
    receptors: ResolvedReceptorSet,
) -> None:
    """Reject an enabled ``P`` term combined with a rotated receptor.

    ``ParallacticAngleJones`` is a Tier 7 identity stub and Tier 5 accepts only
    a static topocentric feed rotation, so composing the two would silently
    omit the time-dependent part of the receptor orientation
    (``Tier5ReceptorFeedPlan.md`` Sections 12.3 and 27).
    """
    if not jones_config.get("P", {}).get("enabled", False):
        return
    if any(
        receptor.feed_rotation_rad != 0.0
        for receptor in receptors.receptor_by_antenna.values()
    ):
        raise UnsupportedFeedGeometryError(
            "a non-zero feed_rotation_deg cannot be combined with an enabled "
            "parallactic-angle term until Tier 7 implements it."
        )


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


def _host_preprocess_time_step(
    source_coords: Any,
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
    altaz = source_coords.transform_to(AltAz(obstime=obstime, location=location))
    az_rad = altaz.az.rad
    alt_rad = altaz.alt.rad
    above_horizon = alt_rad > 0
    return above_horizon, alt_rad[above_horizon], az_rad[above_horizon]


def _host_local_sidereal_time_rad(obstime: Any, location: Any) -> float:
    """Local apparent sidereal time, in radians, for one time step.

    The third piece of the Section 13.3 host-preprocessing stage, added with the
    direction-batched contract: the equatorial half of
    :class:`~radiosim.core.jones.directions.DirectionBatch` needs an hour angle,
    and an hour angle needs a sidereal time.  One astropy call per time step,
    beside the ``ICRS -> AltAz`` transform of every direction that already
    happens there.
    """
    return float(obstime.sidereal_time("apparent", longitude=location.lon).rad)


def _host_direction_cosines(
    altitude_rad: np.ndarray,
    azimuth_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direction cosines ``(l, m, n)`` in the local ENU frame, on the host.

    The second half of the Section 13.3 host-preprocessing stage: a fixed-cost
    trigonometric transform of the astropy output, evaluated once per time step
    and then transferred once.
    """
    return (
        np.cos(altitude_rad) * np.sin(azimuth_rad),
        np.cos(altitude_rad) * np.cos(azimuth_rad),
        np.sin(altitude_rad),
    )


class _ResolvedBeamJones(JonesTerm):
    """Private E-Jones adapter over one canonical :class:`BeamSystem`.

    One adapter per ``(time, frequency)`` step, shared by both solvers, holding
    the per-handler evaluation cache that the HEALPix path used to keep in the
    solver: a beam handler is evaluated once per direction batch no matter how
    many antennas are assigned to it, and the per-antenna receptor factors are
    applied afterwards by the ``C`` and ``H`` chain terms rather than folded in
    here.  Keeping the cache inside the adapter -- whose lifetime is exactly one
    ``(time, frequency)`` step -- is also what keeps it thread-safe under
    ``execution.workers > 1``: solver workers own disjoint time ranges and
    therefore disjoint adapters.
    """

    def __init__(
        self,
        *,
        beam_system: BeamSystem,
        instrument: "SolverInstrumentView",
        altitude_rad: np.ndarray,
        azimuth_rad: np.ndarray,
        frequency_hz: float,
        time_mjd: float,
    ) -> None:
        from radiosim.core.instrument_adapters import SolverInstrumentView

        if type(beam_system) is not BeamSystem:
            raise TypeError("beam_system must be an exact BeamSystem")
        if type(instrument) is not SolverInstrumentView:
            raise TypeError("instrument must be a SolverInstrumentView")
        altitude = np.array(altitude_rad, dtype=np.float64, copy=True, order="C")
        azimuth = np.array(azimuth_rad, dtype=np.float64, copy=True, order="C")
        if altitude.ndim != 1 or azimuth.ndim != 1 or altitude.shape != azimuth.shape:
            raise ValueError(
                "altitude_rad and azimuth_rad must be equal-shape one-dimensional "
                "arrays"
            )
        altitude.setflags(write=False)
        azimuth.setflags(write=False)
        self._beam_system = beam_system
        self._instrument = instrument
        self._altitude_rad = altitude
        self._azimuth_rad = azimuth
        self._frequency_hz = float(frequency_hz)
        self._time_mjd = float(time_mjd)
        self._handler_by_antenna = dict(beam_system.state.assignment_handler_ids)
        self._handler_cache: dict[str, Any] = {}

    @property
    @override
    def name(self) -> str:
        return "E"

    @property
    @override
    def is_direction_dependent(self) -> bool:
        return True

    def _antenna_id(self, antenna_idx: int) -> AntennaId:
        if type(antenna_idx) is not int:
            raise InstrumentAdapterInvariantError("antenna_idx must be an integer")
        try:
            number = self._instrument.antenna_numbers[antenna_idx]
            name = self._instrument.antenna_names[antenna_idx]
        except IndexError as exc:
            raise InstrumentAdapterInvariantError(
                f"antenna row {antenna_idx} is absent from the solver instrument view"
            ) from exc
        return AntennaId(number, name)

    @override
    def compute_jones_batch(
        self,
        *,
        antenna_idx: int,
        directions: DirectionBatch,
        frequency_hz: float,
        freq_idx: int,
        time_mjd: float,
        time_idx: int,
        backend: Any,
        dtype: Any,
    ) -> Any:
        """Return this antenna's ``(n_dir, 2, 2)`` beam response.

        The batch, frequency and time are checked against the ones this adapter
        was resolved for rather than silently preferred one way or the other: the
        adapter carries the exact host arrays ``BeamSystem`` was going to be
        asked about, and a caller evaluating it against a different step is a
        solver bug, not something to paper over.
        """
        if type(directions) is not DirectionBatch:
            raise TypeError("directions must be an exact DirectionBatch")
        if directions.n_dir != self._altitude_rad.size:
            raise InstrumentAdapterInvariantError(
                "Jones direction count does not match the resolved beam direction batch"
            )
        if float(frequency_hz) != self._frequency_hz:
            raise InstrumentAdapterInvariantError(
                "Jones frequency does not match the resolved beam frequency"
            )
        if float(time_mjd) != self._time_mjd:
            raise InstrumentAdapterInvariantError(
                "Jones time does not match the resolved beam time"
            )
        canonical = self._antenna_id(antenna_idx)
        try:
            handler_id = self._handler_by_antenna[canonical]
        except KeyError as exc:
            raise InstrumentAdapterInvariantError(
                "BeamSystem assignment state does not cover solver antenna "
                f"number={canonical.number}, name={canonical.name!r}"
            ) from exc
        if handler_id not in self._handler_cache:
            self._handler_cache[handler_id] = self._beam_system.evaluate_jones(
                canonical,
                altitude_rad=np.array(
                    self._altitude_rad,
                    dtype=np.float64,
                    copy=True,
                    order="C",
                ),
                azimuth_rad=np.array(
                    self._azimuth_rad,
                    dtype=np.float64,
                    copy=True,
                    order="C",
                ),
                frequency_hz=self._frequency_hz,
                time_mjd=self._time_mjd,
                backend=backend,
            )
        return backend.asarray(self._handler_cache[handler_id], dtype=dtype)


def _resolved_receptor_terms(
    *,
    instrument: "SolverInstrumentView",
    receptors: ResolvedReceptorSet,
) -> tuple[BasisTransformJones, ReceptorConfigJones]:
    """Build the two run-constant receptor terms ``(H, C)`` once per call.

    ``C`` and ``H`` depend on the resolved receptor set and the instrument and on
    nothing else -- not on direction, time, or frequency -- so their matrices are
    resolved once above the time loop and the same two term objects are reused by
    every ``(time, frequency)`` chain.  That preserves the Tier 6D property for
    the HEALPix path (the constant receptor factor is not rebuilt per time
    sample) and extends it to the point path, which used to reconstruct both
    terms inside the frequency loop.

    The terms are immutable after construction and their evaluation is pure, so
    sharing them across solver worker threads is safe.
    """
    return (
        BasisTransformJones(receptors=receptors, instrument=instrument),
        ReceptorConfigJones(receptors=receptors, instrument=instrument),
    )


def calculate_visibility(
    instrument: "SolverInstrumentView",
    beam_system: BeamSystem,
    source_arrays: "SourceArrays",
    location: Any,
    time_grid: "ObservationTimeGrid",
    frequencies: Any,
    backend: ArrayBackend,
    receptors: ResolvedReceptorSet,
    jones_config: dict[str, Any] | None = None,
    solver_execution: ResolvedSolverExecutionConfig = SERIAL_SOLVER_EXECUTION,
) -> Any:
    """
    Calculate complex visibility using full polarization (RIME).

    Implements: V_pq = Σ_sources J_p @ C_source @ J_q^H

    Where J is the canonical Jones chain J = H @ G @ B @ D @ P @ C @ E @ T @ Z,
    with K applied separately as a scalar phase.

    Parameters
    ----------
    instrument : SolverInstrumentView
        Owned canonical antenna values and selected baseline geometry.
    beam_system : BeamSystem
        Canonical per-antenna beam evaluator.
    source_arrays : dict
        Dict of source arrays from ``SkyModel.as_point_source_arrays()``.
        Keys: ``ra_rad``, ``dec_rad``, ``flux``, ``spectral_index``,
        ``stokes_q``, ``stokes_u``, ``stokes_v``, ``ref_freq``,
        ``rotation_measure``, ``major_arcsec``, ``minor_arcsec``,
        ``pa_deg``, ``spectral_coeffs``.
    location : EarthLocation
        Observer's geographical location.
    time_grid : ObservationTimeGrid
        Exact canonical UTC sample-center grid.
    frequencies : ndarray
        Canonical frequency centers in Hz.
    backend : ArrayBackend
        Explicit array backend used by supported kernels.
        Options: get_backend("numpy"), get_backend("jax"), get_backend("dask")
    receptors : ResolvedReceptorSet
        Canonical resolved receptor inventory from ``resolve_receptors()``.
        Supplies the per-antenna receptor term ``C`` and basis transform ``H``.
        The default configuration (linear feeds, zero rotation, ``auto``) makes
        both terms exactly the identity.
    jones_config : dict, optional
        Configuration for Jones chain terms. Keys are term names ('K', 'E', 'G', etc.),
        values are dicts with 'enabled' (bool) and term-specific parameters.
        Example: {'G': {'enabled': True, 'sigma': 0.02}, 'Z': {'enabled': True, 'tec': 1e16}}
    solver_execution : ResolvedSolverExecutionConfig, optional
        Resolved solver worker policy. ``workers=1`` (the default) is the exact
        serial path; ``workers=N`` distributes contiguous time blocks over a
        thread pool of ``N`` threads and reassembles them in time order, which
        is bit-identical to the serial result for every ``N``
        (``Tier6HybridRuntimePlan.md`` Sections 11.3, 11.5).

    Returns
    -------
    backend array
        Receptor visibility cube with shape ``(T, B, F, 2, 2)`` in exact
        time-grid, selected-baseline, frequency, receptor-row, receptor-column
        order.

    Examples
    --------
    >>> # Explicit backend
    >>> from radiosim.backends import get_backend
    >>> selected_backend = get_backend("jax")
    >>> vis = calculate_visibility(..., backend=selected_backend)
    """
    if jones_config is None:
        jones_config = {}
    elif type(jones_config) is not dict:
        raise TypeError("jones_config must be a dict or None")
    if "beam" in jones_config:
        raise TypeError(
            "jones_config must not contain a beam entry; pass beam_system directly"
        )

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
    receptors = _require_receptors(receptors)
    solver_execution = require_solver_execution(solver_execution)
    _reject_parallactic_rotation(jones_config, receptors)

    # Get array namespace from backend
    xp = backend.xp
    output_complex_dtype = backend.get_complex_dtype("output")

    # One resolved complex dtype for the whole Jones chain, from the precision
    # model rather than from a literal (defects D8 and D9).  The accumulation
    # precision is the right source: the chain product *is* an accumulation of
    # matrix factors, and it is what the identity seed and every term must agree
    # on.  Both shipped presets whose accumulation is ``float64`` -- ``standard``
    # and ``fast`` -- resolve ``complex128`` here, which is exactly what the
    # removed literals said.
    jones_complex_dtype = backend.get_complex_dtype("accumulation")

    # Extract arrays from source_arrays dict
    _ra_rad = source_arrays["ra_rad"]
    _dec_rad = source_arrays["dec_rad"]
    _ref_freq = source_arrays["ref_freq"]

    n_times = len(time_grid)
    n_baselines = len(instrument.selected_pairs)
    n_freq = len(frequencies)
    sample_times = time_grid.as_astropy()

    # The one compiled kernel, built once per call rather than per step: a fresh
    # closure on every step would defeat the compilation cache entirely
    # (``Tier6HybridRuntimePlan.md`` Section 13.6).
    contraction = baseline_contraction_for(backend)
    baseline_vectors = backend.asarray(
        instrument.baseline_vectors_enu_m,
        dtype=backend.default_real_dtype,
    )

    # Run-constant Jones inputs, resolved once above the time loop: the two
    # receptor terms and the antenna-row view of the selected baseline pairs.
    # Rows, not numbers, because that is what every chain term indexes by.
    receptor_terms = _resolved_receptor_terms(
        instrument=instrument,
        receptors=receptors,
    )
    selected_row_pairs = tuple(
        (instrument.row_for_number(ant1), instrument.row_for_number(ant2))
        for ant1, ant2 in instrument.selected_pairs
    )
    _selected_rows = {row for pair in selected_row_pairs for row in pair}
    selected_rows = tuple(
        row for row in range(len(instrument.antenna_numbers)) if row in _selected_rows
    )

    def _zero_cube() -> Any:
        return backend.zeros_complex(
            (n_times, n_baselines, n_freq, 2, 2),
            dtype=output_complex_dtype,
        )

    # Handle empty source arrays, and any degenerate axis: there is nothing to
    # assemble, so return the canonical zero cube directly.
    if len(_ra_rad) == 0 or n_times == 0 or n_baselines == 0 or n_freq == 0:
        return _zero_cube()

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
    has_polarized_sources = any(
        value is not None and bool(np.any(np.asarray(value) != 0))
        for value in (
            source_stokes_Q_orig,
            source_stokes_U_orig,
            source_stokes_V_orig,
            source_per_channel_q_orig,
            source_per_channel_u_orig,
            source_per_channel_v_orig,
        )
    )

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
    #
    # Output accumulation follows Section 13.3 of Tier6HybridRuntimePlan.md:
    # one (B, 2, 2) block per (time, frequency), one (B, F, 2, 2) block per
    # time, and exactly one (T, B, F, 2, 2) assembly per call. Nothing is
    # written into a pre-allocated cube, so an immutable-array backend performs
    # one assembly instead of T*B*F whole-cube functional copies.
    #
    # Because every time step reads only run-constant inputs and produces its
    # own independent block, contiguous ranges of time indices are exactly the
    # unit of solver worker parallelism (Section 11.3/11.4). ``_time_block``
    # below is that unit; ``execute_time_blocks`` runs it inline for
    # ``workers=1`` and on a thread pool otherwise, with ordered reassembly.
    # ===========================================================================
    def _time_block(time_idx: int, empty_block: list[Any]) -> Any:
        """Compute the ``(B, F, 2, 2)`` output block for one time index."""
        current_obstime = sample_times[time_idx]

        # ---- host preprocessing (named stage; see _host_preprocess_time_step)
        above_horizon, alt_rad_t, az_rad_t = _host_preprocess_time_step(
            source_coords,
            current_obstime,
            location,
        )
        if not np.any(above_horizon):
            # No sources visible at this time - contribute an exactly zero block
            # so the time axis keeps its slot in the single final assembly.
            if not empty_block:
                empty_block.append(
                    backend.zeros_complex(
                        (n_baselines, n_freq, 2, 2),
                        dtype=output_complex_dtype,
                    )
                )
            return empty_block[0]

        # Apply horizon filter for this time step
        source_stokes_I_t = source_stokes_I_orig[above_horizon]
        source_stokes_Q_t = source_stokes_Q_orig[above_horizon]
        source_stokes_U_t = source_stokes_U_orig[above_horizon]
        source_stokes_V_t = source_stokes_V_orig[above_horizon]
        is_unpolarized = not has_polarized_sources
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
        l_np, m_np, n_np = _host_direction_cosines(alt_rad_t, az_rad_t)

        # The one direction batch every Jones term sees at this time step,
        # carrying both the horizontal and the equatorial description of the
        # same directions (Section 13.2).
        directions = DirectionBatch.from_horizontal(
            alt_rad=alt_rad_t,
            az_rad=az_rad_t,
            dir_l=l_np,
            dir_m=m_np,
            dir_n=n_np,
            latitude_rad=float(location.lat.rad),
            local_sidereal_time_rad=_host_local_sidereal_time_rad(
                current_obstime,
                location,
            ),
        )
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

        freq_blocks: list[Any] = []
        for freq_idx, freq in enumerate(frequencies):
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
                time_mjd=float(current_obstime.mjd),
                beam_system=beam_system,
                receptors=receptors,
                receptor_terms=receptor_terms,
            )

            # One chain evaluation per selected antenna, through the one
            # evaluator both solvers share (Section 14, defect D4).
            jones_by_row = evaluate_antenna_jones(
                chain=chain,
                antenna_rows=selected_rows,
                directions=directions,
                frequency_hz=float(freq),
                freq_idx=freq_idx,
                time_mjd=float(current_obstime.mjd),
                time_idx=time_idx,
                backend=backend,
                dtype=jones_complex_dtype,
            )

            # Per-baseline antenna Jones batches: (B, n_sources, 2, 2). These
            # go through the backend's array namespace rather than through
            # ``ArrayBackend.stack``, deliberately: ``stack`` is documented as
            # the solvers' one *accumulation* primitive (Section 13.3), and this
            # is input batching for the kernel, not output accumulation. Keeping
            # the two apart is what lets the accumulation invariants stay
            # countable.
            J_p = xp.stack(
                [jones_by_row[row_p] for row_p, _ in selected_row_pairs],
                axis=0,
            )
            J_q = xp.stack(
                [jones_by_row[row_q] for _, row_q in selected_row_pairs],
                axis=0,
            )

            # Geometric phase (K), applied separately from the chain because it
            # is per-baseline, and batched over baselines: (B, n_sources).
            uvw_wavelengths = uvw_in_wavelengths(
                baseline_vectors_m=baseline_vectors,
                wavelength_m=float(C_LIGHT) / float(freq),
            )
            bl_u = uvw_wavelengths[:, 0:1]
            bl_v = uvw_wavelengths[:, 1:2]
            phase = geometric_phase(
                uvw_wavelengths=uvw_wavelengths,
                dir_l=l_dir,
                dir_m=m_dir,
                dir_n=n_dir,
                backend=backend,
            )

            # Gaussian envelope: scalar attenuation per (baseline, source)
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

            # The one compiled kernel (Section 13.6): one (B, 2, 2) block for
            # all baselines at this (time, frequency).
            block = contraction(
                J_p,
                J_q,
                None if is_unpolarized else coherency_matrices,
                phase,
                envelope,
                I_scaled if is_unpolarized else None,
            )
            freq_blocks.append(backend.asarray(block, dtype=output_complex_dtype))

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
        thread_name_prefix="radiosim-point-solver",
    )

    # One (T, B, F, 2, 2) cube, assembled in a single operation.
    return backend.stack(time_blocks, axis=0)


def _build_jones_chain(
    backend: ArrayBackend,
    jones_config: Any,
    instrument: "SolverInstrumentView",
    alt_rad: Any,
    az_rad: Any,
    freq: Any,
    freq_idx: int,
    n_sources: int,
    location: Any,
    time_mjd: Any,
    beam_system: BeamSystem,
    receptors: ResolvedReceptorSet,
    receptor_terms: tuple[BasisTransformJones, ReceptorConfigJones] | None = None,
) -> JonesChain:
    """Build a JonesChain in the canonical Section 19.1 order (K excluded).

    Terms are added correlator-side first, because ``JonesChain`` composes
    ``terms[0] @ ... @ terms[-1]``::

        J = H @ G @ B @ D @ P @ C @ E @ T @ Z

    ``H``, ``C``, and ``E`` are always present; every other term is optional.
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
    time_mjd : float
        Current observation time in MJD.
    beam_system : BeamSystem
        Canonical per-antenna beam evaluator.
    receptors : ResolvedReceptorSet
        Canonical resolved receptor inventory supplying the C and H terms.
    receptor_terms : tuple, optional
        The ``(H, C)`` terms already built by :func:`_resolved_receptor_terms`
        for this call.  They are run-constant, so a solver builds them once above
        its time loop and passes them here rather than paying for two term
        constructions per ``(time, frequency)`` step.  ``None`` builds them.

    Returns
    -------
    JonesChain
        Chain with H, C, and E (and optionally G, B, D, P, T, Z) terms, added
        in the canonical Section 19.1 order.
    """
    n_antennas = len(instrument.antenna_numbers)
    receptors = _require_receptors(receptors)
    _reject_parallactic_rotation(jones_config, receptors)
    if receptor_terms is None:
        receptor_terms = _resolved_receptor_terms(
            instrument=instrument,
            receptors=receptors,
        )
    basis_transform_term, receptor_config_term = receptor_terms
    chain = JonesChain(backend)

    # H term: reporting-basis transform, leftmost because the correlator
    # performs it (always enabled; exactly I2 when every antenna's native
    # basis already is the resolved output basis).
    chain.add_term(basis_transform_term)

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

    # D term: Polarization leakage (optional)
    d_config = jones_config.get("D", {})
    if d_config.get("enabled", False):
        d_jones = PolarizationLeakageJones(
            n_antennas=n_antennas,
            d_terms=d_config.get("d_terms"),
        )
        chain.add_term(d_jones)

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

    # C term: receptor configuration, between the electronics-side DIEs and
    # the sky-side DDEs (always enabled; exactly I2 for a linear receptor with
    # zero rotation).
    chain.add_term(receptor_config_term)

    # E term: one exact canonical BeamSystem adapter (always enabled).
    e_jones = _ResolvedBeamJones(
        beam_system=beam_system,
        instrument=instrument,
        altitude_rad=alt_rad,
        azimuth_rad=az_rad,
        frequency_hz=float(freq),
        time_mjd=float(time_mjd),
    )
    chain.add_term(e_jones)

    # T term: Troposphere (optional)
    t_config = jones_config.get("T", {})
    if t_config.get("enabled", False):
        t_jones = TroposphereJones(
            n_antennas=n_antennas,
            frequencies=np.array([freq]),
            elevations=alt_rad,
        )
        chain.add_term(t_jones)

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

    return chain
