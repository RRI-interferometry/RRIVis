"""Contracts for immutable Tier 4 result models and fingerprints."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version

import numpy as np
import pytest

from radiosim.api.simulator import Simulator
from radiosim.backends import get_backend
from radiosim.backends.numpy_backend import NumPyBackend
from radiosim.core.phase_center import PhaseCenter
from radiosim.core.result import (
    BackendResultProvenance,
    InvalidResultError,
    LoadedSimulationResult,
    ResultPerformance,
    ResultShapeError,
    SimulationResult,
    SolverResultProvenance,
    build_loaded_simulation_result,
    build_simulation_result,
)


def _mapping(tmp_path):
    layout = tmp_path / "antennas.txt"
    layout.write_text(
        "Name Number BeamID E N U Diameter\nA0 0 0 0 0 0 14\nA1 1 0 10 0 0 14\n",
        encoding="utf-8",
    )
    return {
        "instrument": {
            "source": {
                "kind": "layout_file",
                "path": str(layout),
                "format": "radiosim",
                "telescope_name": "Result Array",
            },
            "location": {
                "longitude_deg": 21.0,
                "latitude_deg": -30.0,
                "height_m": 1000.0,
            },
        },
        "baseline_selection": {"correlations": "cross"},
        "beams": {
            "mode": "analytic",
            "model": {
                "kind": "circular_aperture",
                "taper": {"kind": "gaussian", "edge_taper_db": 10.0},
            },
        },
        "obs_time": {
            "start_time": "2025-01-01T00:00:00",
            "duration_seconds": 2.0,
            "time_step_seconds": 1.0,
        },
        "obs_frequency": {
            "mode": "explicit",
            "channel_frequencies_hz": [100e6, 101e6],
            "channel_widths_hz": [1e6, 1e6],
        },
        "sky_model": {
            "sources": [{"kind": "test_sources", "num_sources": 1, "seed": 1}]
        },
        "execution": {"backend": "numpy", "offline": True},
    }


def _parts(tmp_path, *, dtype="complex128"):
    simulator = Simulator.from_mapping(_mapping(tmp_path), base_dir=tmp_path)
    simulator._ensure_instrument_state()
    simulator._ensure_beam_system()
    backend = get_backend("numpy")
    provenance = BackendResultProvenance(
        requested_backend="numpy",
        actual_backend=backend.name,
        requested_precision={"output": dtype},
        actual_precision={"output": dtype},
        result_dtype=dtype,
    )
    solver = SolverResultProvenance(
        solver="rime",
        sky_representation="point_sources",
        convention="radiosim.rime-zenith-drift.v1",
        execution_path="polarized",
    )
    performance = ResultPerformance(
        setup_seconds=1.0,
        solver_seconds=2.0,
        result_construction_seconds=0.5,
        host_transfer_seconds=0.25,
        total_seconds=3.75,
    )
    cube = np.arange(2 * 1 * 2 * 4, dtype=np.float64).reshape(2, 1, 2, 2, 2)
    receptor = cube.astype(dtype)
    receptor += 1j * receptor
    return simulator, backend, provenance, solver, performance, receptor


def _build(tmp_path, *, dtype="complex128"):
    simulator, backend, provenance, solver, performance, receptor = _parts(
        tmp_path,
        dtype=dtype,
    )
    result = build_simulation_result(
        receptor_visibilities=receptor,
        backend=backend,
        time_grid=simulator.config.observation.time_grid,
        frequencies_hz=simulator.config.frequency.channel_frequencies_hz,
        channel_widths_hz=simulator.config.frequency.channel_widths_hz,
        instrument=simulator.instrument,
        selection=simulator._instrument_state.selection,
        beam_state=simulator.beam_state,
        phase_center=PhaseCenter(),
        backend_provenance=provenance,
        solver_provenance=solver,
        resolved_config=simulator.config.to_json_safe(),
        configuration_provenance=None,
        performance=performance,
        history=("simulated",),
    )
    return result, receptor


class _CountingBackend(NumPyBackend):
    def __init__(self):
        super().__init__()
        self.transfer_count = 0

    def to_numpy(self, arr):
        self.transfer_count += 1
        return super().to_numpy(arr)


def test_result_factory_records_its_own_host_transfer_timing(tmp_path):
    simulator, _backend, provenance, solver, performance, receptor = _parts(tmp_path)
    backend = _CountingBackend()

    result = build_simulation_result(
        receptor_visibilities=receptor,
        backend=backend,
        time_grid=simulator.config.observation.time_grid,
        frequencies_hz=simulator.config.frequency.channel_frequencies_hz,
        channel_widths_hz=simulator.config.frequency.channel_widths_hz,
        instrument=simulator.instrument,
        selection=simulator._instrument_state.selection,
        beam_state=simulator.beam_state,
        phase_center=PhaseCenter(),
        backend_provenance=provenance,
        solver_provenance=solver,
        resolved_config=simulator.config.to_json_safe(),
        configuration_provenance=None,
        performance=performance,
    )

    assert backend.transfer_count == 1
    assert result.performance.host_transfer_seconds >= 0.0


def test_result_factory_timing_owns_transfer_and_extends_total_through_hashing(
    tmp_path,
    monkeypatch,
):
    import radiosim.core.result as result_module

    simulator, _backend, provenance, solver, performance, receptor = _parts(tmp_path)
    backend = _CountingBackend()
    ticks = iter((10.0, 12.0, 15.0, 19.0))
    monkeypatch.setattr(result_module.time, "perf_counter", lambda: next(ticks))

    result = build_simulation_result(
        receptor_visibilities=receptor,
        backend=backend,
        time_grid=simulator.config.observation.time_grid,
        frequencies_hz=simulator.config.frequency.channel_frequencies_hz,
        channel_widths_hz=simulator.config.frequency.channel_widths_hz,
        instrument=simulator.instrument,
        selection=simulator._instrument_state.selection,
        beam_state=simulator.beam_state,
        phase_center=PhaseCenter(),
        backend_provenance=provenance,
        solver_provenance=solver,
        resolved_config=simulator.config.to_json_safe(),
        configuration_provenance=None,
        performance=performance,
    )

    assert backend.transfer_count == 1
    assert result.performance.host_transfer_seconds == 3.0
    assert result.performance.result_construction_seconds == 6.0
    assert result.performance.total_seconds == performance.total_seconds + 9.0


def _json_tree(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _json_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_tree(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _oracle_tag(digest, tag, payload):
    encoded_tag = tag.encode("utf-8")
    digest.update(len(encoded_tag).to_bytes(8, "little"))
    digest.update(encoded_tag)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _oracle_json(digest, tag, value):
    payload = json.dumps(
        _json_tree(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    _oracle_tag(digest, tag, payload)


def _oracle_array(digest, tag, value):
    dtype = value.dtype.newbyteorder("<")
    canonical = np.array(value, dtype=dtype, order="C", copy=True, subok=False)
    _oracle_json(
        digest,
        f"{tag}.metadata",
        {"dtype": dtype.str, "shape": list(value.shape)},
    )
    _oracle_tag(digest, f"{tag}.data", canonical.tobytes(order="C"))


def _independent_fingerprints(
    result,
    *,
    instrument_snapshot=None,
    selection_snapshot=None,
    beam_snapshot=None,
    backend_snapshot=None,
    solver_snapshot=None,
):
    if instrument_snapshot is None:
        instrument_snapshot = result.instrument.to_snapshot()
    if selection_snapshot is None:
        selection_snapshot = result.selection.to_snapshot()
    if beam_snapshot is None:
        beam_snapshot = result.beam_state.to_snapshot()
    if backend_snapshot is None:
        backend_snapshot = result.backend.to_snapshot()
    if solver_snapshot is None:
        solver_snapshot = result.solver.to_snapshot()

    scientific = hashlib.sha256()
    _oracle_json(scientific, "schema", "radiosim.result.v1")
    for tag, array in (
        ("visibilities", result.visibilities),
        ("flags", result.flags),
        ("weights", result.weights),
        ("time.utc_jd1", result.time_grid.utc_jd1),
        ("time.utc_jd2", result.time_grid.utc_jd2),
        (
            "time.integration_time_seconds",
            result.time_grid.integration_time_seconds,
        ),
        ("frequency_hz", result.frequencies_hz),
        ("channel_width_hz", result.channel_widths_hz),
    ):
        _oracle_array(scientific, tag, array)
    for tag, value in (
        ("correlations", result.correlations),
        ("polarization_basis", result.polarization_basis),
        ("instrument", instrument_snapshot),
        ("selection", selection_snapshot),
        ("beam", beam_snapshot),
        ("phase_center", result.phase_center.to_snapshot()),
        ("solver", solver_snapshot),
    ):
        _oracle_json(scientific, tag, value)
    scientific_hex = scientific.hexdigest()

    provenance = hashlib.sha256()
    for tag, value in (
        ("scientific_sha256", scientific_hex),
        ("backend", backend_snapshot),
        ("resolved_config", result.resolved_config),
        ("configuration_provenance", result.configuration_provenance),
        ("package_version", version("radiosim")),
        ("history", result.history),
    ):
        _oracle_json(provenance, tag, value)
    return scientific_hex, provenance.hexdigest()


def _loaded_result_arguments(
    result,
    *,
    instrument_snapshot=None,
    selection_snapshot=None,
    beam_snapshot=None,
    backend_snapshot=None,
    solver_snapshot=None,
):
    if instrument_snapshot is None:
        instrument_snapshot = result.instrument.to_snapshot()
    if selection_snapshot is None:
        selection_snapshot = result.selection.to_snapshot()
    if beam_snapshot is None:
        beam_snapshot = result.beam_state.to_snapshot()
    if backend_snapshot is None:
        backend_snapshot = result.backend.to_snapshot()
    if solver_snapshot is None:
        solver_snapshot = result.solver.to_snapshot()
    scientific, provenance = _independent_fingerprints(
        result,
        instrument_snapshot=instrument_snapshot,
        selection_snapshot=selection_snapshot,
        beam_snapshot=beam_snapshot,
        backend_snapshot=backend_snapshot,
        solver_snapshot=solver_snapshot,
    )
    return {
        "visibilities": result.visibilities,
        "flags": result.flags,
        "weights": result.weights,
        "time_grid": result.time_grid,
        "frequencies_hz": result.frequencies_hz,
        "channel_widths_hz": result.channel_widths_hz,
        "correlations": result.correlations,
        "phase_center": result.phase_center,
        "instrument_snapshot": instrument_snapshot,
        "selection_snapshot": selection_snapshot,
        "beam_snapshot": beam_snapshot,
        "backend_snapshot": backend_snapshot,
        "solver_snapshot": solver_snapshot,
        "resolved_config_snapshot": result.resolved_config,
        "configuration_provenance_snapshot": result.configuration_provenance,
        "performance_snapshot": result.performance.to_snapshot(),
        "history": result.history,
        "expected_scientific_sha256": scientific,
        "expected_provenance_sha256": provenance,
    }


def test_result_factory_flattens_correlations_once_and_hardens_all_arrays(tmp_path):
    result, receptor = _build(tmp_path)

    assert type(result) is SimulationResult
    assert result.visibilities.shape == (2, 1, 2, 4)
    assert np.array_equal(result.visibilities[..., 0], receptor[..., 0, 0])
    assert np.array_equal(result.visibilities[..., 1], receptor[..., 0, 1])
    assert np.array_equal(result.visibilities[..., 2], receptor[..., 1, 0])
    assert np.array_equal(result.visibilities[..., 3], receptor[..., 1, 1])
    assert result.correlations == ("XX", "XY", "YX", "YY")
    assert result.polarization_basis == "linear_xy"
    assert result.flags.dtype == np.dtype("bool")
    assert result.weights.dtype == np.dtype("float64")

    for value in (
        result.visibilities,
        result.flags,
        result.weights,
        result.frequencies_hz,
        result.channel_widths_hz,
    ):
        assert type(value) is np.ndarray
        assert value.flags.c_contiguous
        assert value.flags.writeable is False
        with pytest.raises(ValueError):
            value.setflags(write=True)

    receptor[...] = 99
    assert not np.all(result.visibilities == 99)
    assert result.stokes_i() is not result.stokes_i()


def test_result_factory_crosses_the_backend_transfer_boundary_exactly_once(tmp_path):
    simulator, _, provenance, solver, performance, receptor = _parts(tmp_path)
    backend = _CountingBackend()

    build_simulation_result(
        receptor_visibilities=receptor,
        backend=backend,
        time_grid=simulator.config.observation.time_grid,
        frequencies_hz=simulator.config.frequency.channel_frequencies_hz,
        channel_widths_hz=simulator.config.frequency.channel_widths_hz,
        instrument=simulator.instrument,
        selection=simulator._instrument_state.selection,
        beam_state=simulator.beam_state,
        phase_center=PhaseCenter(),
        backend_provenance=provenance,
        solver_provenance=solver,
        resolved_config=simulator.config.to_json_safe(),
        configuration_provenance=None,
        performance=performance,
    )

    assert backend.transfer_count == 1


def test_result_fingerprints_are_stable_and_loaded_state_verifies_them(tmp_path):
    result, _ = _build(tmp_path)
    loaded = build_loaded_simulation_result(
        visibilities=result.visibilities,
        flags=result.flags,
        weights=result.weights,
        time_grid=result.time_grid,
        frequencies_hz=result.frequencies_hz,
        channel_widths_hz=result.channel_widths_hz,
        correlations=result.correlations,
        phase_center=result.phase_center,
        instrument_snapshot=result.instrument.to_snapshot(),
        selection_snapshot=result.selection.to_snapshot(),
        beam_snapshot=result.beam_state.to_snapshot(),
        backend_snapshot=result.backend.to_snapshot(),
        solver_snapshot=result.solver.to_snapshot(),
        resolved_config_snapshot=result.resolved_config,
        configuration_provenance_snapshot=result.configuration_provenance,
        performance_snapshot=result.performance.to_snapshot(),
        history=result.history,
        expected_scientific_sha256=result.scientific_sha256,
        expected_provenance_sha256=result.provenance_sha256,
    )

    assert type(loaded) is LoadedSimulationResult
    assert result.scientifically_equal(loaded)
    assert loaded.scientifically_equal(result)
    assert loaded.scientific_sha256 == result.scientific_sha256
    assert loaded.provenance_sha256 == result.provenance_sha256
    assert type(loaded.performance) is ResultPerformance
    assert loaded.performance == result.performance
    assert len(result.to_summary_snapshot()["array_summaries"]) == 3
    assert _independent_fingerprints(result) == (
        result.scientific_sha256,
        result.provenance_sha256,
    )

    changed = np.array(result.visibilities, copy=True)
    changed[0, 0, 0, 0] += 1
    with pytest.raises(InvalidResultError, match="scientific"):
        build_loaded_simulation_result(
            visibilities=changed,
            flags=result.flags,
            weights=result.weights,
            time_grid=result.time_grid,
            frequencies_hz=result.frequencies_hz,
            channel_widths_hz=result.channel_widths_hz,
            correlations=result.correlations,
            phase_center=result.phase_center,
            instrument_snapshot=result.instrument.to_snapshot(),
            selection_snapshot=result.selection.to_snapshot(),
            beam_snapshot=result.beam_state.to_snapshot(),
            backend_snapshot=result.backend.to_snapshot(),
            solver_snapshot=result.solver.to_snapshot(),
            resolved_config_snapshot=result.resolved_config,
            configuration_provenance_snapshot=None,
            performance_snapshot=result.performance.to_snapshot(),
            history=result.history,
            expected_scientific_sha256=result.scientific_sha256,
            expected_provenance_sha256=result.provenance_sha256,
        )


def test_result_fingerprints_exclude_performance_workflow_and_output_paths(tmp_path):
    simulator, backend, provenance, solver, performance, receptor = _parts(tmp_path)
    common = {
        "receptor_visibilities": receptor,
        "backend": backend,
        "time_grid": simulator.config.observation.time_grid,
        "frequencies_hz": simulator.config.frequency.channel_frequencies_hz,
        "channel_widths_hz": simulator.config.frequency.channel_widths_hz,
        "instrument": simulator.instrument,
        "selection": simulator._instrument_state.selection,
        "beam_state": simulator.beam_state,
        "phase_center": PhaseCenter(),
        "backend_provenance": provenance,
        "solver_provenance": solver,
        "history": ("simulated",),
    }
    first = build_simulation_result(
        **common,
        resolved_config={
            **simulator.config.to_json_safe(),
            "workflow": {"output_dir": "/tmp/first"},
        },
        configuration_provenance={
            "input_snapshot": {
                "execution": {"backend": "numpy"},
                "workflow": {"output_dir": "/tmp/first"},
            },
            "override_origins": {
                "execution.backend": "document",
                "workflow.output_dir": "document",
            },
            "path_resolutions": {
                "workflow.output_dir": {"resolved_path": "/tmp/first"}
            },
        },
        performance=performance,
    )
    second = build_simulation_result(
        **common,
        resolved_config={
            **simulator.config.to_json_safe(),
            "workflow": {"output_dir": "/tmp/second"},
        },
        configuration_provenance={
            "input_snapshot": {
                "execution": {"backend": "numpy"},
                "workflow": {"output_dir": "/tmp/second"},
            },
            "override_origins": {
                "execution.backend": "document",
                "workflow.output_dir": "override",
            },
            "path_resolutions": {
                "workflow.output_dir": {"resolved_path": "/tmp/second"}
            },
        },
        performance=ResultPerformance(
            setup_seconds=2.0,
            solver_seconds=3.0,
            result_construction_seconds=1.0,
            host_transfer_seconds=0.5,
            total_seconds=6.5,
        ),
    )

    assert first.scientific_sha256 == second.scientific_sha256
    assert first.provenance_sha256 == second.provenance_sha256
    assert "workflow" not in first.resolved_config
    assert "workflow" not in first.configuration_provenance["input_snapshot"]
    assert (
        "workflow.output_dir" not in first.configuration_provenance["override_origins"]
    )
    assert (
        "workflow.output_dir" not in first.configuration_provenance["path_resolutions"]
    )


def test_loaded_result_rejects_nonboolean_flags_instead_of_coercing(tmp_path):
    result, _ = _build(tmp_path)

    with pytest.raises(InvalidResultError, match="flags must use bool dtype"):
        build_loaded_simulation_result(
            visibilities=result.visibilities,
            flags=np.zeros(result.flags.shape, dtype=np.uint8),
            weights=result.weights,
            time_grid=result.time_grid,
            frequencies_hz=result.frequencies_hz,
            channel_widths_hz=result.channel_widths_hz,
            correlations=result.correlations,
            phase_center=result.phase_center,
            instrument_snapshot=result.instrument.to_snapshot(),
            selection_snapshot=result.selection.to_snapshot(),
            beam_snapshot=result.beam_state.to_snapshot(),
            backend_snapshot=result.backend.to_snapshot(),
            solver_snapshot=result.solver.to_snapshot(),
            resolved_config_snapshot=result.resolved_config,
            configuration_provenance_snapshot=result.configuration_provenance,
            performance_snapshot=result.performance.to_snapshot(),
            history=result.history,
            expected_scientific_sha256=result.scientific_sha256,
            expected_provenance_sha256=result.provenance_sha256,
        )


def test_loaded_result_rejects_self_consistent_invalid_identity_snapshots(tmp_path):
    result, _ = _build(tmp_path)

    invalid_selection = _json_tree(result.selection.to_snapshot())
    invalid_selection["selected_ids"] = [[0, 99]]
    with pytest.raises(InvalidResultError, match="selection"):
        build_loaded_simulation_result(
            **_loaded_result_arguments(
                result,
                selection_snapshot=invalid_selection,
            )
        )

    wrong_count = _json_tree(result.selection.to_snapshot())
    wrong_count["selected_ids"] = []
    with pytest.raises(ResultShapeError, match="selection"):
        build_loaded_simulation_result(
            **_loaded_result_arguments(
                result,
                selection_snapshot=wrong_count,
            )
        )

    wrong_backend = _json_tree(result.backend.to_snapshot())
    wrong_backend["result_dtype"] = "complex64"
    with pytest.raises(InvalidResultError, match="backend"):
        build_loaded_simulation_result(
            **_loaded_result_arguments(
                result,
                backend_snapshot=wrong_backend,
            )
        )

    wrong_beam = _json_tree(result.beam_state.to_snapshot())
    wrong_beam["resolved"]["instrument_fingerprint"] = "0" * 64
    with pytest.raises(InvalidResultError, match="beam"):
        build_loaded_simulation_result(
            **_loaded_result_arguments(
                result,
                beam_snapshot=wrong_beam,
            )
        )

    wrong_solver = _json_tree(result.solver.to_snapshot())
    wrong_solver["convention"] = "invalid"
    with pytest.raises(InvalidResultError, match="solver"):
        build_loaded_simulation_result(
            **_loaded_result_arguments(
                result,
                solver_snapshot=wrong_solver,
            )
        )


def test_result_models_are_identity_equal_unhashable_and_not_directly_constructible(
    tmp_path,
):
    result, _ = _build(tmp_path)
    same, _ = _build(tmp_path)

    assert result == result
    assert result != same
    assert result.scientifically_equal(same)
    with pytest.raises(TypeError):
        hash(result)
    with pytest.raises(TypeError):
        SimulationResult()
    with pytest.raises(TypeError):

        class MutableResult(SimulationResult):
            pass


def test_result_factory_rejects_shapes_nonfinite_coordinates_and_model_subclasses(
    tmp_path,
):
    simulator, backend, provenance, solver, performance, receptor = _parts(tmp_path)
    common = {
        "backend": backend,
        "time_grid": simulator.config.observation.time_grid,
        "frequencies_hz": simulator.config.frequency.channel_frequencies_hz,
        "channel_widths_hz": simulator.config.frequency.channel_widths_hz,
        "instrument": simulator.instrument,
        "selection": simulator._instrument_state.selection,
        "beam_state": simulator.beam_state,
        "phase_center": PhaseCenter(),
        "backend_provenance": provenance,
        "solver_provenance": solver,
        "resolved_config": simulator.config.to_json_safe(),
        "configuration_provenance": None,
        "performance": performance,
    }

    with pytest.raises(ResultShapeError):
        build_simulation_result(
            receptor_visibilities=receptor[..., 0],
            **common,
        )

    bad = receptor.copy()
    bad[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(InvalidResultError):
        build_simulation_result(receptor_visibilities=bad, **common)

    with pytest.raises(TypeError):

        class MutablePerformance(ResultPerformance):
            pass
