"""Contracts for immutable Tier 4 result models and fingerprints."""

from __future__ import annotations

import hashlib
import inspect
import json
from importlib.metadata import version

import numpy as np
import pytest

from radiosim.api.simulator import Simulator
from radiosim.backends import get_backend
from radiosim.backends.numpy_backend import NumPyBackend
from radiosim.core.phase_center import PhaseCenter
from radiosim.core.polarization_basis import (
    CORRELATION_LABELS,
    parallel_hand_indices,
)
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


def _mapping(tmp_path, *, receptors=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    layout = tmp_path / "antennas.txt"
    layout.write_text(
        "Name Number BeamID E N U Diameter\nA0 0 0 0 0 0 14\nA1 1 0 10 0 0 14\n",
        encoding="utf-8",
    )
    extra = {} if receptors is None else {"receptors": receptors}
    return {
        **extra,
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


def _parts(tmp_path, *, dtype="complex128", receptors=None):
    simulator = Simulator.from_mapping(
        _mapping(tmp_path, receptors=receptors),
        base_dir=tmp_path,
    )
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
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


def _build(tmp_path, *, dtype="complex128", receptors=None):
    simulator, backend, provenance, solver, performance, receptor = _parts(
        tmp_path,
        dtype=dtype,
        receptors=receptors,
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
        receptors=simulator.receptors,
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
        receptors=simulator.receptors,
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
        receptors=simulator.receptors,
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


def _independent_receptor_entry(result):
    """Re-derive the hashed receptor entry without reading production code."""
    receptors = result.receptors
    snapshot = receptors if isinstance(receptors, dict) else receptors.to_snapshot()
    snapshot = _json_tree(snapshot)
    return {
        "schema_version": snapshot["schema_version"],
        "output_basis": snapshot["output_basis"],
        "receptor_sha256": snapshot["receptor_sha256"],
        "receptors": [
            {
                "antenna_number": row["antenna_number"],
                "antenna_name": row["antenna_name"],
                "basis": row["basis"],
                "feed_rotation_rad": row["feed_rotation_rad"],
                "feed_angle_rad": list(row["feed_angle_rad"]),
            }
            for row in snapshot["receptors"]
        ],
    }


def _independent_instrument_entry(snapshot):
    """Re-derive the hashed instrument entry without reading production code.

    The scientific digest covers only the transport-free scientific facts of
    the instrument: the resolved values, the field-source labels explaining
    them, and ``instrument_sha256``.  Source paths and locators, raw source
    hashes, dependency versions, registry policy, pre-override diameters,
    per-antenna source records, and location transport diagnostics stay out.
    """
    snapshot = _json_tree(snapshot)
    return {
        "schema_version": snapshot["schema_version"],
        "instrument_sha256": snapshot["instrument_sha256"],
        "name": snapshot["name"],
        "source": {
            "telescope_name_source": snapshot["source"]["telescope_name_source"],
        },
        "location": {
            key: snapshot["location"][key]
            for key in (
                "longitude_deg",
                "latitude_deg",
                "height_m",
                "itrs_xyz_m",
                "source",
                "location_source",
            )
        },
        "antennas": [
            {
                "number": antenna["number"],
                "name": antenna["name"],
                "position_enu_m": antenna["position_enu_m"],
                "diameter_m": antenna["diameter_m"],
                "mount_type": antenna["mount_type"],
                "beam_id": antenna["beam_id"],
                "provenance": {
                    key: antenna["provenance"][key]
                    for key in (
                        "identity_source",
                        "position_source",
                        "diameter_source",
                        "mount_source",
                        "beam_id_source",
                    )
                },
            }
            for antenna in snapshot["antennas"]
        ],
    }


_BEAM_TRANSPORT_KEYS = {
    "path",
    "resolved_path",
    "path_provenance_key",
    "definition_fingerprint",
    "assignment_fingerprint",
    "state_fingerprint",
    "loaded_fingerprint",
}


def _independent_beam_entry(value):
    """Re-derive the hashed beam entry: drop filesystem-transport keys."""
    if isinstance(value, dict) or hasattr(value, "items"):
        return {
            key: _independent_beam_entry(item)
            for key, item in value.items()
            if key not in _BEAM_TRANSPORT_KEYS
        }
    if isinstance(value, (tuple, list)):
        return [_independent_beam_entry(item) for item in value]
    return value


def _independent_fingerprints(
    result,
    *,
    instrument_snapshot=None,
    selection_snapshot=None,
    beam_snapshot=None,
    backend_snapshot=None,
    solver_snapshot=None,
    receptor_entry=None,
):
    if receptor_entry is None:
        receptor_entry = _independent_receptor_entry(result)
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
        ("receptor", receptor_entry),
        ("instrument", _independent_instrument_entry(instrument_snapshot)),
        ("selection", selection_snapshot),
        ("beam", _independent_beam_entry(beam_snapshot)),
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
    receptors_snapshot=None,
    correlations=None,
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
    if receptors_snapshot is None:
        receptors_snapshot = result.receptors.to_snapshot()
    if correlations is None:
        correlations = result.correlations
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
        "correlations": correlations,
        "phase_center": result.phase_center,
        "instrument_snapshot": instrument_snapshot,
        "selection_snapshot": selection_snapshot,
        "beam_snapshot": beam_snapshot,
        "receptors_snapshot": receptors_snapshot,
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
        receptors=simulator.receptors,
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
        receptors_snapshot=result.receptors.to_snapshot(),
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
            receptors_snapshot=result.receptors.to_snapshot(),
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
        "receptors": simulator.receptors,
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


def test_scientific_fingerprint_is_independent_of_source_checkout_location(tmp_path):
    """Equal science hashes equally regardless of where sources live on disk.

    The two builds are identical except for the directory holding the antenna
    layout file, mimicking the same commit checked out at two filesystem
    locations.  The absolute path stays visible in the stored snapshot and
    keeps contributing to ``provenance_sha256`` through the resolved
    configuration; only the scientific digest ignores it.
    """
    first, _ = _build(tmp_path / "checkout_a")
    second, _ = _build(tmp_path / "checkout_b" / "nested")

    first_reference = first.instrument.provenance.source_reference
    second_reference = second.instrument.provenance.source_reference
    assert first_reference != second_reference
    assert first.instrument.to_snapshot()["source"]["reference"] == first_reference
    assert first.scientific_sha256 == second.scientific_sha256
    assert first.scientifically_equal(second)
    assert second.scientifically_equal(first)
    assert first.provenance_sha256 != second.provenance_sha256


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
            receptors_snapshot=result.receptors.to_snapshot(),
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
        "receptors": simulator.receptors,
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


# ---------------------------------------------------------------------------
# Tier 5E: data-driven correlation coordinates
# ---------------------------------------------------------------------------

CIRCULAR = {"default": {"basis": "circular"}}


def test_correlation_coordinates_follow_the_resolved_output_basis(tmp_path):
    linear, _ = _build(tmp_path)
    circular, _ = _build(tmp_path, receptors=CIRCULAR)

    assert linear.receptors.output_basis == "linear_xy"
    assert linear.correlations == ("XX", "XY", "YX", "YY")
    assert linear.polarization_basis == "linear_xy"

    assert circular.receptors.output_basis == "circular_rl"
    assert circular.correlations == ("RR", "RL", "LR", "LL")
    assert circular.polarization_basis == "circular_rl"

    assert linear.correlations is CORRELATION_LABELS["linear_xy"]
    assert circular.correlations is CORRELATION_LABELS["circular_rl"]


def test_result_correlation_labels_come_from_the_shared_table_only(tmp_path):
    """No literal correlation tuple survives in ``core/result.py`` (defect D4)."""
    import radiosim.core.result as result_module

    source = inspect.getsource(result_module)
    assert "radiosim.core.polarization_basis" in source
    assert not hasattr(result_module, "_CORRELATIONS")
    assert '("XX", "XY", "YX", "YY")' not in source
    assert '("RR", "RL", "LR", "LL")' not in source
    assert 'polarization_basis="linear_xy"' not in source


def test_stokes_i_derives_its_indices_from_the_correlation_labels(tmp_path):
    circular, receptor = _build(tmp_path, receptors=CIRCULAR)

    np.testing.assert_array_equal(
        circular.stokes_i(),
        circular.visibilities[..., 0] + circular.visibilities[..., 3],
    )
    assert parallel_hand_indices(circular.correlations) == (0, 3)

    # The indices are derived, not assumed: a corrupted label axis is rejected
    # instead of silently summing indices 0 and 3.
    object.__setattr__(circular, "correlations", ("XX", "YY", "XY", "YX"))
    with pytest.raises(ValueError, match="accepted correlation coordinate set"):
        circular.stokes_i()


def test_stokes_i_recovers_total_intensity_in_both_bases(tmp_path):
    """S11: ``stokes_i()`` is the parallel-hand sum in either basis."""
    linear = Simulator.from_mapping(_mapping(tmp_path / "l"), base_dir=tmp_path / "l")
    circular_dir = tmp_path / "c"
    circular = Simulator.from_mapping(
        _mapping(circular_dir, receptors=CIRCULAR),
        base_dir=circular_dir,
    )
    linear_result = linear.run(progress=False)
    circular_result = circular.run(progress=False)

    assert linear_result.correlations == ("XX", "XY", "YX", "YY")
    assert circular_result.correlations == ("RR", "RL", "LR", "LL")
    np.testing.assert_allclose(
        circular_result.stokes_i(),
        linear_result.stokes_i(),
        rtol=1e-10,
        atol=1e-12,
    )


def test_scientific_fingerprint_records_the_basis_and_the_receptor_state(tmp_path):
    """S14 and Section 23: basis and receptors enter the scientific hash only.

    All three results are built in one directory, so the receptor configuration
    is the *only* difference between them: the instrument snapshot records the
    layout path, which would otherwise vary.
    """
    linear, _ = _build(tmp_path)
    circular, _ = _build(tmp_path, receptors=CIRCULAR)
    rotated, _ = _build(
        tmp_path,
        receptors={"default": {"basis": "linear", "feed_rotation_deg": 30.0}},
    )

    assert linear.scientific_sha256 != circular.scientific_sha256
    assert linear.scientific_sha256 != rotated.scientific_sha256
    assert circular.scientific_sha256 != rotated.scientific_sha256

    # instrument_sha256 is unchanged: receptors are a sibling of the instrument.
    fingerprints = {
        result.instrument.provenance.instrument_sha256
        for result in (linear, circular, rotated)
    }
    assert len(fingerprints) == 1

    for result in (linear, circular, rotated):
        assert _independent_fingerprints(result) == (
            result.scientific_sha256,
            result.provenance_sha256,
        )


def test_scientific_fingerprint_is_stable_for_an_identical_receptor_set(tmp_path):
    first, _ = _build(tmp_path, receptors=CIRCULAR)
    second, _ = _build(
        tmp_path,
        receptors={"default": {"basis": "circular"}, "output_basis": "circular"},
    )

    assert first.receptors.provenance.receptor_sha256 == (
        second.receptors.provenance.receptor_sha256
    )
    assert first.scientific_sha256 == second.scientific_sha256


def test_summary_snapshot_reports_a_bounded_receptor_block(tmp_path):
    circular, _ = _build(tmp_path, receptors=CIRCULAR)

    snapshot = circular.to_summary_snapshot()

    assert snapshot["correlations"] == ["RR", "RL", "LR", "LL"]
    assert snapshot["polarization_basis"] == "circular_rl"
    assert snapshot["receptor"] == {
        "output_basis": "circular_rl",
        "receptor_sha256": circular.receptors.provenance.receptor_sha256,
        "native_basis_counts": {"linear": 0, "circular": 2},
        "antenna_count": 2,
    }
    json.dumps(snapshot, allow_nan=False)


def test_result_factory_rejects_receptors_from_another_instrument(tmp_path):
    other = tmp_path / "other"
    mapping = _mapping(other)
    layout = other / "antennas.txt"
    layout.write_text(
        "Name Number BeamID E N U Diameter\nB0 7 0 0 0 0 14\nB1 8 0 10 0 0 14\n",
        encoding="utf-8",
    )
    foreign = Simulator.from_mapping(mapping, base_dir=other)
    foreign._ensure_instrument_state()
    foreign._ensure_receptor_set()

    simulator, backend, provenance, solver, performance, receptor = _parts(tmp_path)
    with pytest.raises(InvalidResultError, match="receptors do not belong"):
        build_simulation_result(
            receptor_visibilities=receptor,
            backend=backend,
            time_grid=simulator.config.observation.time_grid,
            frequencies_hz=simulator.config.frequency.channel_frequencies_hz,
            channel_widths_hz=simulator.config.frequency.channel_widths_hz,
            instrument=simulator.instrument,
            selection=simulator._instrument_state.selection,
            beam_state=simulator.beam_state,
            receptors=foreign.receptors,
            phase_center=PhaseCenter(),
            backend_provenance=provenance,
            solver_provenance=solver,
            resolved_config=simulator.config.to_json_safe(),
            configuration_provenance=None,
            performance=performance,
        )


def test_loaded_result_round_trips_both_accepted_correlation_tuples(tmp_path):
    for receptors, labels, basis in (
        (None, ("XX", "XY", "YX", "YY"), "linear_xy"),
        (CIRCULAR, ("RR", "RL", "LR", "LL"), "circular_rl"),
    ):
        result, _ = _build(tmp_path, receptors=receptors)
        loaded = build_loaded_simulation_result(**_loaded_result_arguments(result))

        assert type(loaded) is LoadedSimulationResult
        assert loaded.correlations == labels
        assert loaded.polarization_basis == basis
        assert loaded.receptors["output_basis"] == basis
        assert loaded.receptors["receptor_sha256"] == (
            result.receptors.provenance.receptor_sha256
        )
        assert set(loaded.receptors) == {
            "schema_version",
            "output_basis",
            "receptor_sha256",
            "receptors",
        }
        assert [row["antenna_number"] for row in loaded.receptors["receptors"]] == [
            antenna.id.number for antenna in result.instrument.antennas
        ]
        assert loaded.scientific_sha256 == result.scientific_sha256
        assert loaded.scientifically_equal(result)
        np.testing.assert_array_equal(loaded.stokes_i(), result.stokes_i())


@pytest.mark.parametrize(
    "correlations",
    [
        ("XX", "YY", "XY", "YX"),
        ("RR", "LL", "RL", "LR"),
        ("XX", "XY", "YX", "RR"),
        ("XX", "XY", "YX"),
        ("I", "Q", "U", "V"),
    ],
)
def test_loaded_result_rejects_every_unaccepted_correlation_axis(
    tmp_path,
    correlations,
):
    result, _ = _build(tmp_path)

    with pytest.raises(InvalidResultError) as caught:
        build_loaded_simulation_result(
            **_loaded_result_arguments(result, correlations=correlations)
        )

    message = str(caught.value)
    assert "('XX', 'XY', 'YX', 'YY')" in message
    assert "('RR', 'RL', 'LR', 'LL')" in message


def test_loaded_result_rejects_a_receptor_snapshot_that_contradicts_the_axis(tmp_path):
    result, _ = _build(tmp_path)
    snapshot = _json_tree(result.receptors.to_snapshot())
    snapshot["output_basis"] = "circular_rl"

    with pytest.raises(InvalidResultError, match="receptor"):
        build_loaded_simulation_result(
            **_loaded_result_arguments(result, receptors_snapshot=snapshot)
        )


def test_loaded_result_rejects_receptor_rows_outside_the_instrument(tmp_path):
    result, _ = _build(tmp_path)
    snapshot = _json_tree(result.receptors.to_snapshot())
    snapshot["receptors"][1]["antenna_number"] = 91

    with pytest.raises(InvalidResultError, match="receptor"):
        build_loaded_simulation_result(
            **_loaded_result_arguments(result, receptors_snapshot=snapshot)
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("basis", "elliptical"),
        ("feed_rotation_rad", float("nan")),
        ("feed_angle_rad", [0.0]),
    ],
)
def test_loaded_result_rejects_malformed_receptor_rows(tmp_path, key, value):
    result, _ = _build(tmp_path)
    snapshot = _json_tree(result.receptors.to_snapshot())
    snapshot["receptors"][0][key] = value

    with pytest.raises(InvalidResultError, match="receptor"):
        build_loaded_simulation_result(
            **_loaded_result_arguments(result, receptors_snapshot=snapshot)
        )


def test_the_receptor_snapshot_schema_version_matches_the_receptor_module():
    import radiosim.core.receptor as receptor_module
    import radiosim.core.result as result_module

    assert (
        result_module._RECEPTOR_SCHEMA_VERSION
        == receptor_module._RECEPTOR_SCHEMA_VERSION
    )
