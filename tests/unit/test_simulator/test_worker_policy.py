"""Worker policy: typed in Tier 6B, effective for the loader in 6C, solver in 6E.

Tier 6B owned configuration and resolution only -- both worker knobs were typed,
resolved, clamped, and recorded while no loader driver or solver read them.  Tier
6C makes the *loader* half effective (``Tier6HybridRuntimePlan.md`` Sections 11.2,
16.1, 32.3) and flips the interim pin that recorded the hard-coded pool size.
Tier 6E makes the *solver* half effective (Sections 11.3, 11.5, 12.1, 32.5):
``execution.solver.workers`` now drives a thread pool over contiguous time
blocks, ``run(n_workers=...)`` is gone, and the 6B interim pin below is flipped
from "nothing reads the resolved count" to "both solvers do".

Tier 6E evidence record -- Q2 reconfirmation
============================================

Plan Section 41 Q2 asks whether the FITS beam handlers and the ``BeamSystem``
are safe to share across solver threads.  Tier 6A answered it provisionally with
a synthetic probe -- four threads calling ``BeamSystem.evaluate_jones`` directly
over 64 cases -- and its independent acceptance ruled that **6E must reconfirm
under its own real workload** rather than inherit the answer.

Reconfirmed here, on ``osx-arm64`` (Apple M1 Max, macOS 26.5.2), pyuvdata 3.2.1,
in both locked environments (py311 / py312), by
``test_tier6e_q2_shared_fits_and_analytic_beams_are_thread_safe``: a ``mixed``
beam configuration -- antenna 0 on an analytic circular-aperture beam, antenna 1
on a shared ``_LoadedFITSHandler`` wrapping pyuvdata ``UVBeam.interp`` -- driven
through the *real* ``calculate_visibility`` solver over 8 time samples and 2
frequencies at 2, 4 and 8 threads, compared against the serial run of byte-
identical inputs.  Result: **max absolute deviation 0.0 and byte-identical cubes
at every worker count**, i.e. 3/3 worker counts clean.  The companion
``test_tier6e_point_solver_...``/``test_tier6e_healpix_solver_...`` matrices
extend the same comparison to both solvers in both polarization states, 12
further serial-vs-parallel comparisons, all byte-identical.

The Q2 fallback -- giving each worker its own handler instance -- therefore does
**not** fire, and ``core/beam/fits.py`` / ``core/beam/runtime.py`` are correctly
absent from 6E's Section 33 file list.  As in 6A, this is positive evidence from
one platform and one pyuvdata version, not a proof: a future divergence should
reopen Q2 rather than be worked around.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from radiosim.api import Simulator
from radiosim.core.sky.operations.parallel import LoaderExecutionRecord
from radiosim.utils.network import set_offline_policy
from tests.fixtures.beamfits import write_scalar_efield_beamfits
from tests.fixtures.configs import valid_config_mapping

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Eight time samples: enough for the Section 27 W3/W4 worker counts to produce
# genuinely different partitions (2, 4 and 8 blocks) instead of clamping.
_MULTI_TIME_OBS = {
    "start_time": "2025-01-01T00:00:00",
    "duration_seconds": 8.0,
    "time_step_seconds": 1.0,
}


@pytest.fixture(autouse=True)
def _never_leak_an_offline_policy():
    """A forced-offline run must not silence a later test's network gate."""
    set_offline_policy(False)
    yield
    set_offline_policy(False)


def _run(tmp_path: Path, **overrides: object):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return Simulator.from_mapping(
        valid_config_mapping(tmp_path, **overrides),
        base_dir=tmp_path,
    ).run(progress=False)


def test_tier6b_resolved_worker_policy_appears_in_the_result_snapshot(tmp_path):
    result = _run(
        tmp_path,
        execution={
            "sky_loading": {"max_workers": 4, "executor": "thread"},
            "solver": {"workers": 2},
        },
    )
    execution = result.resolved_config["execution"]

    assert execution["sky_loading"]["max_workers"] == 4
    assert execution["sky_loading"]["executor"] == "thread"
    assert execution["solver"]["workers"] == 2
    assert execution["solver"]["executor"] == "thread"


def test_tier6b_auto_loader_policy_is_resolved_before_the_result_is_recorded(tmp_path):
    result = _run(tmp_path)
    sky_loading = result.resolved_config["execution"]["sky_loading"]

    assert sky_loading["max_workers"] == min(1, os.cpu_count() or 1, 8)
    assert sky_loading["max_workers"] is not None
    assert sky_loading["executor"] == "auto"


def test_tier6b_resolved_worker_policy_appears_in_the_summary_json(tmp_path):
    result = _run(
        tmp_path,
        execution={
            "sky_loading": {"max_workers": 6, "executor": "process"},
            "solver": {"workers": 1},
        },
    )
    write_result_summary_json = importlib.import_module(
        "radiosim.io.summary_json"
    ).write_result_summary_json
    target = tmp_path / "result.summary.json"

    write_result_summary_json(result, target)
    document = json.loads(target.read_text(encoding="utf-8"))
    execution = document["resolved_config"]["execution"]

    assert execution["sky_loading"] == {"max_workers": 6, "executor": "process"}
    assert execution["solver"] == {"workers": 1, "executor": "thread"}


def test_tier6b_worker_policy_never_changes_the_scientific_fingerprint(tmp_path):
    serial = _run(tmp_path)
    parallel = _run(
        tmp_path,
        execution={
            "sky_loading": {"max_workers": 4, "executor": "thread"},
            "solver": {"workers": 2},
        },
    )

    assert serial.scientific_sha256 == parallel.scientific_sha256
    assert serial.provenance_sha256 != parallel.provenance_sha256


def test_tier6c_loader_driver_receives_the_resolved_worker_count(tmp_path, monkeypatch):
    """Flipped by Tier 6C from the 6B interim pin.

    The 6B pin asserted ``observed == [8]``: the loader driver was handed a
    literal while the typed policy sat unread.  Tier 6C removed the literal
    (plan Section 11.2), so the driver now receives exactly the resolved value.
    """
    parallel = importlib.import_module("radiosim.core.sky.operations.parallel")
    observed: list[object] = []
    real_loader = parallel.load_models_parallel

    def recording_loader(requests, *args, **kwargs):
        observed.append(kwargs.get("max_workers"))
        return real_loader(requests, *args, **kwargs)

    monkeypatch.setattr(parallel, "load_models_parallel", recording_loader)
    simulator = Simulator.from_mapping(
        valid_config_mapping(
            tmp_path,
            execution={"sky_loading": {"max_workers": 2}},
        ),
        base_dir=tmp_path,
    )
    simulator.setup()

    assert observed == [2]
    assert simulator.config.execution.sky_loading.max_workers == 2
    assert simulator._resolved.execution.sky_loading.max_workers == 2


def test_tier6c_no_worker_literal_survives_in_the_simulator():
    """Section 27 W7 -- the api layer names no pool size of its own."""
    import re

    source = (_REPO_ROOT / "src" / "radiosim" / "api" / "simulator.py").read_text(
        encoding="utf-8"
    )

    assert "max_workers=8" not in source
    assert re.search(r"max_workers\s*=\s*\d", source) is None
    assert "sky_loading.max_workers" in source


def test_tier6c_the_loader_execution_record_is_observable_on_the_simulator(tmp_path):
    """Section 11.2 -- the driver reports what it actually did."""
    simulator = Simulator.from_mapping(
        valid_config_mapping(
            tmp_path,
            execution={"sky_loading": {"max_workers": 3, "executor": "thread"}},
        ),
        base_dir=tmp_path,
    )
    assert simulator.loader_execution is None

    simulator.setup()
    record = simulator.loader_execution

    assert record is not None
    assert record.requested_executor == "thread"
    assert record.actual_executor == "thread"
    assert record.max_workers == 3
    assert record.degraded_reason is None


def test_tier6c_auto_loader_policy_records_the_registry_choice(tmp_path):
    """Section 27 W2 -- the *actual* executor is recorded, not just the request."""
    result = _run(tmp_path)
    record = LoaderExecutionRecord.from_history(result.history)

    assert record is not None
    assert record.requested_executor == "auto"
    # The default fixture requests one ``test_sources`` loader, a synthetic
    # (GIL-bound) category, so the registry recommends a process pool.
    assert record.actual_executor == "process"
    assert record.max_workers == min(1, os.cpu_count() or 1, 8)
    assert record.degraded_reason is None


def test_tier6c_the_summary_json_execution_block_reports_the_policy(tmp_path):
    """Section 19 / Section 27 W2 -- resolved policy *and* executed record."""
    result = _run(
        tmp_path,
        execution={
            "sky_loading": {"max_workers": 2, "executor": "thread"},
            "solver": {"workers": 1},
        },
    )
    write_result_summary_json = importlib.import_module(
        "radiosim.io.summary_json"
    ).write_result_summary_json
    target = tmp_path / "execution.summary.json"

    write_result_summary_json(result, target)
    document = json.loads(target.read_text(encoding="utf-8"))
    execution = document["execution"]

    assert execution["sky_loading"] == {"max_workers": 2, "executor": "thread"}
    assert execution["solver"] == {"workers": 1, "executor": "thread"}
    assert execution["loader"] == {
        "requested_executor": "thread",
        "actual_executor": "thread",
        "max_workers": 2,
        "degraded_reason": None,
    }
    # The shared fixture forces offline, and the executed policy says so.
    assert execution["offline"] is True


@pytest.mark.parametrize("max_workers", [1, 2, 4, 8])
@pytest.mark.parametrize("executor", ["thread", "process"])
def test_tier6c_loader_worker_invariance(tmp_path, max_workers, executor):
    """Section 27 W1 / invariant S7 -- the loader policy is scientifically inert.

    Reference and varied runs share one ``tmp_path`` because the antenna-layout
    path is part of the instrument identity, and therefore of the scientific
    digest; only the worker policy may differ between the two runs.
    """
    reference = _run(tmp_path)
    varied = _run(
        tmp_path,
        execution={"sky_loading": {"max_workers": max_workers, "executor": executor}},
    )

    assert varied.scientific_sha256 == reference.scientific_sha256
    assert np.array_equal(varied.visibilities, reference.visibilities)


def test_tier6c_offline_policy_is_installed_before_any_loader_runs(
    tmp_path, monkeypatch
):
    """Section 16.1 / Section 20.1 step 6 precedes step 7."""
    network = importlib.import_module("radiosim.utils.network")
    parallel = importlib.import_module("radiosim.core.sky.operations.parallel")
    events: list[str] = []

    real_set = network.set_offline_policy
    real_loader = parallel.load_models_parallel

    def recording_set(offline):
        events.append(f"offline:{offline}")
        return real_set(offline)

    def recording_loader(*args, **kwargs):
        events.append("load")
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(network, "set_offline_policy", recording_set)
    monkeypatch.setattr(parallel, "load_models_parallel", recording_loader)

    Simulator.from_mapping(
        valid_config_mapping(tmp_path, execution={"offline": True}),
        base_dir=tmp_path,
    ).setup()

    assert events == ["offline:True", "load"]
    network.set_offline_policy(False)


def test_tier6e_both_solvers_consume_the_resolved_solver_execution_policy():
    """Flipped by Tier 6E from the 6B interim pin.

    The 6B pin asserted the opposite -- that no solver named a pool or read the
    resolved worker count -- because 6B typed the policy and deliberately left
    it unread.  Tier 6E makes it effective: both solvers take an exact
    ``ResolvedSolverExecutionConfig`` and hand their time partition to the one
    shared pool driver in ``core/solver_partition.py`` (plan Section 11.3).
    """
    import inspect

    from radiosim.core import solver_partition, visibility, visibility_healpix
    from radiosim.core.runtime_config import ResolvedSolverExecutionConfig
    from radiosim.simulator import rime

    for module in (visibility, visibility_healpix, rime):
        source = inspect.getsource(module)
        assert "solver_execution" in source
        # The pool itself lives in exactly one module, not in each solver.
        assert "ThreadPoolExecutor" not in source

    assert "ThreadPoolExecutor" in inspect.getsource(solver_partition)
    for function in (
        visibility.calculate_visibility,
        visibility_healpix.calculate_visibility_healpix,
        rime.RIMESimulator.calculate_visibilities,
    ):
        parameter = inspect.signature(function).parameters["solver_execution"]
        assert parameter.default == ResolvedSolverExecutionConfig(
            workers=1, executor="thread"
        )


def test_tier6e_the_simulator_hands_the_resolved_policy_to_the_solver(
    tmp_path, monkeypatch
):
    """Section 12.1 -- one centrally resolved source, no ``run()`` argument."""
    from radiosim.simulator.rime import RIMESimulator

    observed: list[Any] = []
    real = RIMESimulator.calculate_visibilities

    def recording(self, *args: Any, **kwargs: Any):
        observed.append(kwargs["solver_execution"])
        return real(self, *args, **kwargs)

    monkeypatch.setattr(RIMESimulator, "calculate_visibilities", recording)
    result = _run(
        tmp_path,
        obs_time=_MULTI_TIME_OBS,
        execution={"solver": {"workers": 3}},
    )

    assert [(policy.workers, policy.executor) for policy in observed] == [(3, "thread")]
    assert result.resolved_config["execution"]["solver"]["workers"] == 3


def test_tier6e_run_no_longer_accepts_n_workers(tmp_path):
    """Section 27 E9 / Section 12.3 -- plain ``TypeError`` naming the parameter."""
    import inspect

    simulator = Simulator.from_mapping(
        valid_config_mapping(tmp_path), base_dir=tmp_path
    )

    with pytest.raises(TypeError, match="n_workers"):
        simulator.run(n_workers=1)  # type: ignore[call-arg]

    signature = inspect.signature(Simulator.run)
    assert list(signature.parameters) == ["self", "progress"]
    assert signature.parameters["progress"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("workers", [1, 2, 3, 4])
def test_tier6e_solver_worker_invariance_end_to_end(tmp_path, workers):
    """Section 27 W3 / invariant S6 -- the solver policy is scientifically inert.

    Reference and varied runs share one ``tmp_path`` because the antenna-layout
    path is part of the instrument identity, and therefore of the scientific
    digest; only the worker policy may differ between the two runs.
    """
    reference = _run(tmp_path, obs_time=_MULTI_TIME_OBS)
    varied = _run(
        tmp_path,
        obs_time=_MULTI_TIME_OBS,
        execution={"solver": {"workers": workers}},
    )

    assert varied.scientific_sha256 == reference.scientific_sha256
    assert np.array_equal(varied.visibilities, reference.visibilities)


def _thread_tracing_driver(observed: set[str]):
    """Wrap the real pool driver so the executing thread of each block is seen."""
    from radiosim.core import solver_partition

    real = solver_partition.execute_time_blocks

    def driver(compute_block, **kwargs: Any):
        def traced(start: int, stop: int):
            observed.add(threading.current_thread().name)
            return compute_block(start, stop)

        return real(traced, **kwargs)

    return driver


def test_tier6e_solver_workers_actually_use_more_than_one_thread(tmp_path, monkeypatch):
    """Section 27 W4 -- with ``workers=4`` and 8 time samples, the knob is real.

    A four-party barrier is the deterministic form of "more than one distinct
    worker thread executes time blocks": if the policy were a no-op, or if the
    pool were smaller than the requested worker count, the barrier could never
    be satisfied and the run would fail with ``BrokenBarrierError`` instead of
    silently passing on a thread count that happened to be reused.
    """
    from radiosim.core import solver_partition

    real = solver_partition.execute_time_blocks
    observed: set[str] = set()
    barrier = threading.Barrier(4, timeout=60.0)

    def driver(compute_block, **kwargs: Any):
        def traced(start: int, stop: int):
            observed.add(threading.current_thread().name)
            barrier.wait()
            return compute_block(start, stop)

        return real(traced, **kwargs)

    monkeypatch.setattr("radiosim.core.visibility.execute_time_blocks", driver)
    _run(
        tmp_path,
        obs_time=_MULTI_TIME_OBS,
        execution={"solver": {"workers": 4}},
    )

    assert len(observed) == 4
    assert all(name.startswith("radiosim-point-solver") for name in observed)
    assert threading.current_thread().name not in observed


def test_tier6e_one_worker_never_leaves_the_calling_thread(tmp_path, monkeypatch):
    """``workers=1`` is the exact serial path: no pool, no thread hop."""
    observed: set[str] = set()
    monkeypatch.setattr(
        "radiosim.core.visibility.execute_time_blocks",
        _thread_tracing_driver(observed),
    )

    _run(tmp_path, obs_time=_MULTI_TIME_OBS)

    assert observed == {threading.current_thread().name}


def test_tier6e_a_failing_time_block_propagates_without_partial_results(
    tmp_path, monkeypatch
):
    """Section 20 -- one worker's exception surfaces; nothing partial is built."""
    from radiosim.core import solver_partition

    real = solver_partition.execute_time_blocks
    finished: list[tuple[int, int]] = []

    class _Boom(RuntimeError):
        pass

    def driver(compute_block, **kwargs: Any):
        def traced(start: int, stop: int):
            if start == 4:
                raise _Boom("time block failed")
            produced = compute_block(start, stop)
            finished.append((start, stop))
            return produced

        return real(traced, **kwargs)

    monkeypatch.setattr("radiosim.core.visibility.execute_time_blocks", driver)
    simulator = Simulator.from_mapping(
        valid_config_mapping(
            tmp_path,
            obs_time=_MULTI_TIME_OBS,
            execution={"solver": {"workers": 4}},
        ),
        base_dir=tmp_path,
    )

    with pytest.raises(_Boom, match="time block failed"):
        simulator.run(progress=False)

    # The failure is the worker's own exception, not a pool wrapper, and no
    # result was published from the blocks that did finish.
    assert simulator.result is None
    assert (4, 6) not in finished


# =========================================================================
# Tier 6E -- solver-level serial-vs-parallel bit-identity, both solvers
#
# Section 11.5 promises bit-identity for *any* worker count, and Section 11.4
# explains why it is structural: each time index writes a disjoint output block,
# so no reduction is repartitioned.  These tests assert the promise directly at
# the solver boundary, where a whole-run digest cannot hide behind result
# post-processing.
# =========================================================================


def _solver_inputs(tmp_path: Path, **overrides: object):
    """Resolve the instrument view, beam system and receptors for a fixture run."""
    from radiosim.core.instrument_adapters import SolverInstrumentView

    simulator = Simulator.from_mapping(
        valid_config_mapping(tmp_path, **overrides),
        base_dir=tmp_path,
    )
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    return (
        SolverInstrumentView.from_state(simulator._instrument_state),
        simulator.beam_system,
        simulator.receptors,
    )


_SOLVER_FREQS = np.array([100e6, 101e6], dtype=np.float64)


def _eight_sample_time_grid():
    from radiosim.core.time_grid import build_observation_time_grid

    return build_observation_time_grid(
        start_time="2025-01-01T00:00:00",
        duration_seconds=8.0,
        cadence_seconds=1.0,
    )


def _point_source_arrays(*, polarized: bool) -> dict[str, Any]:
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    location = EarthLocation.from_geodetic(
        21.4283 * u.deg, -30.72152 * u.deg, 1073.0 * u.m
    )
    lst = Time("2025-01-01T00:00:00").sidereal_time("apparent", longitude=location.lon)
    zeros = np.zeros(2, dtype=np.float64)
    return {
        "ra_rad": np.array([lst.rad, lst.rad + 0.01], dtype=np.float64),
        "dec_rad": np.array([-0.536, -0.526], dtype=np.float64),
        "flux": np.array([2.0, 1.0], dtype=np.float64),
        "spectral_index": np.array([-0.7, -0.8], dtype=np.float64),
        "stokes_q": np.array([0.2, 0.0]) if polarized else zeros.copy(),
        "stokes_u": np.array([0.0, 0.1]) if polarized else zeros.copy(),
        "stokes_v": np.array([0.05, 0.0]) if polarized else zeros.copy(),
        "ref_freq": np.full(2, 100e6, dtype=np.float64),
        "rotation_measure": zeros.copy(),
        "spectral_coeffs": None,
        "per_channel_flux": None,
        "per_channel_stokes_q": None,
        "per_channel_stokes_u": None,
        "per_channel_stokes_v": None,
        "channel_frequencies": None,
        "major_arcsec": zeros.copy(),
        "minor_arcsec": zeros.copy(),
        "pa_deg": zeros.copy(),
    }


def _healpix_model(*, polarized: bool):
    from radiosim.core.precision import PrecisionConfig
    from radiosim.core.sky import HealpixData, SkyModel

    maps = np.linspace(1.0, 2.0, 12, dtype=np.float64)
    maps = np.vstack([maps, maps * 1.1])
    return SkyModel(
        healpix=HealpixData(
            maps=maps,
            nside=1,
            frequencies=_SOLVER_FREQS,
            coordinate_frame="icrs",
            q_maps=np.full_like(maps, 0.1) if polarized else None,
            u_maps=np.full_like(maps, 0.05) if polarized else None,
            v_maps=np.full_like(maps, 0.02) if polarized else None,
        ),
        model_name="tier6e-worker-invariance",
        brightness_conversion="rayleigh-jeans",
        precision=PrecisionConfig.standard(),
    )


def _solver_location():
    from astropy import units as u
    from astropy.coordinates import EarthLocation

    return EarthLocation.from_geodetic(21.4283 * u.deg, -30.72152 * u.deg, 1073.0 * u.m)


def _resolved_solver_execution(workers: int):
    from radiosim.core.runtime_config import ResolvedSolverExecutionConfig

    return ResolvedSolverExecutionConfig(workers=workers, executor="thread")


@pytest.mark.parametrize("workers", [2, 4, 8])
@pytest.mark.parametrize("polarized", [True, False])
def test_tier6e_point_solver_is_bit_identical_under_workers(
    tmp_path, workers, polarized
):
    """Section 11.5 -- the point solver's cube does not depend on worker count."""
    from radiosim.backends import get_backend
    from radiosim.core.visibility import calculate_visibility

    instrument, beam_system, receptors = _solver_inputs(tmp_path)
    common = {
        "instrument": instrument,
        "beam_system": beam_system,
        "source_arrays": _point_source_arrays(polarized=polarized),
        "location": _solver_location(),
        "time_grid": _eight_sample_time_grid(),
        "frequencies": _SOLVER_FREQS,
        "backend": get_backend("numpy"),
        "receptors": receptors,
    }

    serial = np.asarray(
        calculate_visibility(**common, solver_execution=_resolved_solver_execution(1))
    )
    parallel = np.asarray(
        calculate_visibility(
            **common, solver_execution=_resolved_solver_execution(workers)
        )
    )

    assert parallel.shape == serial.shape == (8, 3, 2, 2, 2)
    assert parallel.dtype == serial.dtype
    assert parallel.tobytes() == serial.tobytes()


@pytest.mark.parametrize("workers", [2, 4, 8])
@pytest.mark.parametrize("polarized", [True, False])
def test_tier6e_healpix_solver_is_bit_identical_under_workers(
    tmp_path, workers, polarized
):
    """Section 11.5 -- the HEALPix solver's cube does not depend on worker count."""
    from radiosim.backends import get_backend
    from radiosim.core.visibility_healpix import calculate_visibility_healpix

    instrument, beam_system, receptors = _solver_inputs(tmp_path)
    common = {
        "instrument": instrument,
        "beam_system": beam_system,
        "location": _solver_location(),
        "time_grid": _eight_sample_time_grid(),
        "frequencies": _SOLVER_FREQS,
        "backend": get_backend("numpy"),
        "receptors": receptors,
        "include_polarization": polarized,
    }
    sky = _healpix_model(polarized=polarized)

    serial = np.asarray(
        calculate_visibility_healpix(
            sky, **common, solver_execution=_resolved_solver_execution(1)
        )
    )
    parallel = np.asarray(
        calculate_visibility_healpix(
            sky, **common, solver_execution=_resolved_solver_execution(workers)
        )
    )

    assert parallel.shape == serial.shape == (8, 3, 2, 2, 2)
    assert parallel.dtype == serial.dtype
    assert parallel.tobytes() == serial.tobytes()


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_tier6e_q2_shared_fits_and_analytic_beams_are_thread_safe(tmp_path, workers):
    """Plan Section 41 Q2, reconfirmed under Tier 6E's own solver workload.

    Tier 6A answered Q2 provisionally with a synthetic four-thread probe over
    ``BeamSystem.evaluate_jones``; its acceptance ruled that 6E must reconfirm
    under the workload it actually ships.  This is that reconfirmation: a
    ``mixed`` beam configuration -- one antenna on an analytic beam, one on a
    shared ``_LoadedFITSHandler`` wrapping pyuvdata ``UVBeam.interp`` -- driven
    through the real solver at 2, 4 and 8 threads over 8 time samples, compared
    byte for byte against the serial run of the identical inputs.
    """
    from radiosim.backends import get_backend
    from radiosim.core.visibility import calculate_visibility

    beam_path = write_scalar_efield_beamfits(tmp_path).path
    beams = {
        "mode": "mixed",
        "analytic_model": {
            "kind": "circular_aperture",
            "taper": {"kind": "uniform"},
        },
        "assignments": [
            {"antenna": {"kind": "number", "number": 0}, "beam": {"kind": "analytic"}},
            {
                "antenna": {"kind": "number", "number": 1},
                "beam": {"kind": "fits", "path": beam_path.name},
            },
        ],
    }
    instrument, beam_system, receptors = _solver_inputs(tmp_path, beams=beams)
    handler_ids = set(dict(beam_system.state.assignment_handler_ids).values())
    assert len(handler_ids) == 2

    common = {
        "instrument": instrument,
        "beam_system": beam_system,
        "source_arrays": _point_source_arrays(polarized=True),
        "location": _solver_location(),
        "time_grid": _eight_sample_time_grid(),
        "frequencies": _SOLVER_FREQS,
        "backend": get_backend("numpy"),
        "receptors": receptors,
    }

    serial = np.asarray(
        calculate_visibility(**common, solver_execution=_resolved_solver_execution(1))
    )
    parallel = np.asarray(
        calculate_visibility(
            **common, solver_execution=_resolved_solver_execution(workers)
        )
    )

    assert np.max(np.abs(parallel - serial)) == 0.0
    assert parallel.tobytes() == serial.tobytes()
