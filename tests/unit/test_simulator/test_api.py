"""Tests for the resolved-only public Simulator construction contract."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import webbrowser
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import healpy as hp
import numpy as np
import pytest
from pydantic import ValidationError

from radiosim.api import Simulator
from radiosim.core.precision import PrecisionConfig
from radiosim.io.config import ExecutionConfig, PrecisionInput, RadioSimConfig
from radiosim.io.config_resolution import (
    ConfigPathError,
    ConfigSchemaError,
    ConfigSemanticError,
    ConfigSourceError,
    SimulationOverrides,
)
from radiosim.io.instrument_config import (
    BaselineSelectionConfig,
    InstrumentConfig,
)
from tests.fixtures.beamfits import write_scalar_efield_beamfits
from tests.fixtures.configs import (
    resolved_config,
    valid_config_mapping,
    write_config_yaml,
)


def _explicit_data(tmp_path: Path, **section_overrides: object) -> dict[str, object]:
    data = valid_config_mapping(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [100e6, 101.25e6, 109e6],
            "channel_widths_hz": [1e6, 1e6, 1e6],
        },
        **section_overrides,
    )
    return data


def _from_parameters(
    tmp_path: Path,
    data: dict[str, object],
    *,
    frequencies: object | None = None,
    execution: object | None = None,
    overrides: SimulationOverrides | None = None,
) -> Simulator:
    instrument_data = data["instrument"]
    observation = data["obs_time"]
    frequency = data["obs_frequency"]
    assert isinstance(instrument_data, dict)
    assert isinstance(observation, dict)
    assert isinstance(frequency, dict)
    channels = (
        frequency["channel_frequencies_hz"] if frequencies is None else frequencies
    )
    return Simulator.from_parameters(
        instrument=InstrumentConfig.model_validate(instrument_data),
        baseline_selection=BaselineSelectionConfig.model_validate(
            data["baseline_selection"]
        ),
        channel_frequencies_hz=channels,
        channel_widths_hz=frequency["channel_widths_hz"],
        start_time=observation["start_time"],
        duration_seconds=observation["duration_seconds"],
        time_step_seconds=observation["time_step_seconds"],
        sky_model=data["sky_model"],
        beams=data["beams"],
        visibility=data["visibility"],
        execution=data["execution"] if execution is None else execution,
        base_dir=tmp_path,
        overrides=overrides,
    )


def test_simulator_constructor_accepts_only_resolved_runtime(tmp_path):
    bundle = resolved_config(tmp_path)

    simulator = Simulator(bundle.runtime)

    assert simulator.config is bundle.runtime
    assert simulator.provenance is None
    assert simulator._backend_name == "numpy"
    assert simulator._precision == PrecisionConfig.standard()

    for invalid in (
        {},
        bundle,
        RadioSimConfig.model_validate(valid_config_mapping(tmp_path)),
    ):
        with pytest.raises(TypeError, match="ResolvedSimulationConfig"):
            Simulator(invalid)


def test_public_constructor_signatures_are_disjoint_and_explicit():
    init_parameters = inspect.signature(Simulator.__init__).parameters
    assert list(init_parameters) == ["self", "resolved"]
    assert init_parameters["resolved"].kind is inspect.Parameter.POSITIONAL_ONLY

    expected = {
        "from_yaml": ["path", "overrides"],
        "from_config": ["config", "base_dir", "overrides"],
        "from_mapping": ["data", "base_dir", "overrides"],
        "from_parameters": [
            "instrument",
            "baseline_selection",
            "channel_frequencies_hz",
            "channel_widths_hz",
            "start_time",
            "duration_seconds",
            "time_step_seconds",
            "sky_model",
            "beams",
            "visibility",
            "execution",
            "base_dir",
            "overrides",
        ],
    }
    for method_name, names in expected.items():
        parameters = inspect.signature(getattr(Simulator, method_name)).parameters
        assert list(parameters) == names
        if method_name == "from_yaml":
            assert parameters["path"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            keyword_only = names[1:]
        else:
            keyword_only = names if method_name == "from_parameters" else names[1:]
        assert all(
            parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            for name in keyword_only
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"config": {}},
        {"antenna_layout": "antennas.txt"},
        {"frequencies": [100.0]},
        {"backend": "numpy"},
        {"precision": "fast"},
        {"simulator": "rime"},
        {"sky_model": "test"},
    ],
)
def test_old_multi_purpose_constructor_is_rejected(kwargs):
    with pytest.raises(TypeError):
        Simulator(**kwargs)


def test_disjoint_classmethods_reject_wrong_input_kinds(tmp_path):
    data = valid_config_mapping(tmp_path)
    model = RadioSimConfig.model_validate(data)
    path = write_config_yaml(tmp_path, data)

    with pytest.raises(TypeError, match="RadioSimConfig"):
        Simulator.from_config(data)
    with pytest.raises(TypeError, match="RadioSimConfig"):
        Simulator.from_config(path)
    with pytest.raises(TypeError, match="Mapping"):
        Simulator.from_mapping(model)
    with pytest.raises((TypeError, ConfigPathError)):
        Simulator.from_yaml(data)


def test_all_public_construction_paths_have_equivalent_runtime_meaning(tmp_path):
    data = _explicit_data(tmp_path)
    path = write_config_yaml(tmp_path, data)
    model = RadioSimConfig.model_validate(data)

    yaml_simulator = Simulator.from_yaml(path)
    model_simulator = Simulator.from_config(model, base_dir=tmp_path)
    mapping_simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    parameter_simulator = _from_parameters(tmp_path, data)
    direct_simulator = Simulator(mapping_simulator.config)

    assert (
        yaml_simulator.config
        == model_simulator.config
        == mapping_simulator.config
        == parameter_simulator.config
        == direct_simulator.config
    )
    assert yaml_simulator.provenance.source.kind == "yaml"
    assert model_simulator.provenance.source.kind == "model"
    assert mapping_simulator.provenance.source.kind == "mapping"
    assert parameter_simulator.provenance.source.kind == "parameters"
    assert direct_simulator.provenance is None
    assert "workflow" not in yaml_simulator.provenance.input_snapshot
    assert not yaml_simulator.provenance.workflow_origins


def test_from_yaml_honors_yaml_parent_base_without_repository_cwd(
    tmp_path, monkeypatch
):
    data = valid_config_mapping(tmp_path)
    data["instrument"]["source"]["path"] = "antennas.txt"
    config_path = write_config_yaml(tmp_path, data)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    simulator = Simulator.from_yaml(config_path)

    assert (
        simulator.config.instrument.source.path == (tmp_path / "antennas.txt").resolve()
    )


def test_mapping_and_model_relative_paths_require_an_explicit_base(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["instrument"]["source"]["path"] = "antennas.txt"
    model = RadioSimConfig.model_validate(data)

    with pytest.raises((ConfigPathError, ConfigSourceError)):
        Simulator.from_mapping(data)
    with pytest.raises((ConfigPathError, ConfigSourceError)):
        Simulator.from_config(model)


def test_from_parameters_honors_explicit_base_dir(tmp_path, monkeypatch):
    data = _explicit_data(tmp_path)
    data["instrument"]["source"]["path"] = "antennas.txt"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    simulator = _from_parameters(tmp_path, data)

    assert (
        simulator.config.instrument.source.path == (tmp_path / "antennas.txt").resolve()
    )


def test_backend_and_precision_are_identical_across_python_entry_points(tmp_path):
    data = _explicit_data(
        tmp_path,
        execution={"backend": "jax", "precision": {"preset": "standard"}},
    )
    model = RadioSimConfig.model_validate(data)
    path = write_config_yaml(tmp_path, data)
    overrides = SimulationOverrides(
        backend="auto",
        precision=PrecisionInput(preset="fast"),
    )

    simulators = (
        Simulator.from_yaml(path, overrides=overrides),
        Simulator.from_config(model, base_dir=tmp_path, overrides=overrides),
        Simulator.from_mapping(data, base_dir=tmp_path, overrides=overrides),
        _from_parameters(tmp_path, data, overrides=overrides),
    )

    for simulator in simulators:
        assert simulator._backend_name == "auto"
        assert simulator.config.execution.backend_strategy == "auto"
        assert simulator._precision == PrecisionConfig.fast()
        assert simulator.precision == PrecisionConfig.fast()


def test_none_preserves_document_values_and_auto_is_a_real_override(tmp_path):
    data = _explicit_data(
        tmp_path,
        execution={"backend": "numpy", "precision": {"preset": "precise"}},
    )

    document = Simulator.from_mapping(
        data,
        base_dir=tmp_path,
        overrides=SimulationOverrides(backend=None, precision=None),
    )
    overridden = Simulator.from_mapping(
        data,
        base_dir=tmp_path,
        overrides=SimulationOverrides(
            backend="auto", precision=PrecisionInput(preset="fast")
        ),
    )

    assert document._backend_name == "numpy"
    assert document._precision == PrecisionConfig.precise()
    assert overridden._backend_name == "auto"
    assert overridden._precision == PrecisionConfig.fast()
    assert overridden._precision != document._precision


def test_from_parameters_preserves_nonuniform_numpy_frequencies_without_aliasing(
    tmp_path,
):
    data = _explicit_data(tmp_path)
    caller_frequencies = np.array([100e6, 101.25e6, 109e6])

    simulator = _from_parameters(
        tmp_path,
        data,
        frequencies=caller_frequencies,
    )
    caller_frequencies[1] = 999e6

    assert simulator.config.frequency.source_mode == "explicit"
    assert simulator.config.frequency.channel_frequencies_hz == (
        100e6,
        101.25e6,
        109e6,
    )
    runtime_array = simulator.config.frequency.as_numpy()
    runtime_array[0] = 1.0
    assert simulator.config.frequency.channel_frequencies_hz[0] == 100e6


def test_from_parameters_accepts_typed_preset_execution_config(tmp_path):
    data = _explicit_data(tmp_path)
    execution = ExecutionConfig(
        backend="numpy",
        precision=PrecisionInput(preset="standard"),
        offline=True,
    )

    simulator = _from_parameters(tmp_path, data, execution=execution)

    assert simulator.config.execution.backend_strategy == "numpy"
    assert simulator.config.execution.precision == PrecisionConfig.standard()
    assert simulator.config.execution.offline is True


def test_from_mapping_does_not_retain_caller_owned_nested_data(tmp_path):
    caller = _explicit_data(tmp_path)
    original_sources = caller["sky_model"]["sources"]
    original_frequencies = caller["obs_frequency"]["channel_frequencies_hz"]

    simulator = Simulator.from_mapping(caller, base_dir=tmp_path)
    original_sources.append({"kind": "test_sources"})
    original_frequencies[1] = 999e6
    caller["execution"]["offline"] = False

    assert len(simulator.config.sky_model.sources) == 1
    assert simulator.config.frequency.channel_frequencies_hz[1] == 101.25e6
    assert simulator.config.execution.offline is True


def test_resolved_runtime_and_classmethod_provenance_are_immutable(tmp_path):
    simulator = Simulator.from_mapping(
        _explicit_data(tmp_path),
        base_dir=tmp_path,
    )

    with pytest.raises(ValidationError, match="frozen"):
        simulator.config.instrument.location.latitude_deg = 0.0
    with pytest.raises(FrozenInstanceError):
        simulator.provenance.schema_version = 2


def test_workflow_is_absent_from_runtime_state(tmp_path):
    data = _explicit_data(
        tmp_path,
        workflow={
            "result_filename": "workflow-name",
            "result_format": "summary_json",
            "save_results": True,
            "plot_results": True,
            "open_plots_in_browser": True,
            "save_log": True,
        },
    )

    simulator = Simulator.from_mapping(data, base_dir=tmp_path)

    assert not hasattr(simulator.config, "workflow")
    assert "workflow" not in simulator.__dict__


def test_construction_crosses_no_runtime_output_plot_or_browser_boundary(
    tmp_path, monkeypatch
):
    data = _explicit_data(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Simulator construction crossed a runtime side-effect boundary")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(webbrowser, "open", forbidden)
    device_module = importlib.import_module("radiosim.utils.device")
    backends_module = importlib.import_module("radiosim.backends")
    instrument_resolution_module = importlib.import_module(
        "radiosim.core.instrument_resolution"
    )
    network_module = importlib.import_module("radiosim.utils.network")
    parallel_module = importlib.import_module("radiosim.core.sky.operations.parallel")
    monkeypatch.setattr(device_module, "get_device_resources", forbidden)
    monkeypatch.setattr(backends_module, "get_backend", forbidden)
    monkeypatch.setattr(
        instrument_resolution_module,
        "resolve_instrument",
        forbidden,
    )
    monkeypatch.setattr(network_module, "get_network_status", forbidden)
    monkeypatch.setattr(parallel_module, "load_models_parallel", forbidden)

    simulator = Simulator.from_mapping(data, base_dir=tmp_path)

    assert simulator.device_resources is None
    assert simulator.result is None


def test_invalid_mapping_fails_before_device_backend_network_or_loader(
    tmp_path, monkeypatch
):
    data = _explicit_data(
        tmp_path,
        obs_time={"duration_seconds": 1.0, "time_step_seconds": 2.0},
    )

    def forbidden(*args, **kwargs):
        pytest.fail("invalid configuration reached a runtime boundary")

    monkeypatch.setattr("radiosim.utils.device.get_device_resources", forbidden)
    monkeypatch.setattr("radiosim.backends.get_backend", forbidden)
    monkeypatch.setattr("radiosim.utils.network.get_network_status", forbidden)
    monkeypatch.setattr(
        "radiosim.core.sky.operations.parallel.load_models_parallel", forbidden
    )

    with pytest.raises(ConfigSemanticError):
        Simulator.from_mapping(data, base_dir=tmp_path)


def test_schema_errors_are_reported_by_from_mapping(tmp_path):
    data = _explicit_data(tmp_path)
    data["instrument"]["location"]["latitude_deg"] = float("inf")

    with pytest.raises(ConfigSchemaError, match="instrument.location.latitude_deg"):
        Simulator.from_mapping(data, base_dir=tmp_path)


@pytest.mark.parametrize("entry_point", ["yaml", "model", "mapping", "parameters"])
def test_fits_runtime_activates_from_every_document_entry_point(
    tmp_path,
    entry_point,
):
    beam_path = write_scalar_efield_beamfits(tmp_path).path
    data = _explicit_data(
        tmp_path,
        beams={
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": beam_path.name},
        },
    )

    if entry_point == "yaml":
        simulator = Simulator.from_yaml(write_config_yaml(tmp_path, data))
    elif entry_point == "model":
        simulator = Simulator.from_config(
            RadioSimConfig.model_validate(data),
            base_dir=tmp_path,
        )
    elif entry_point == "mapping":
        simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    else:
        simulator = _from_parameters(tmp_path, data)

    simulator.setup()

    assert simulator.beam_system.state is simulator.beam_state
    assert simulator._backend is not None
    assert simulator._is_setup is True
    assert not hasattr(simulator, "_beam_config")
    assert not hasattr(simulator, "_beam_manager")


@pytest.mark.parametrize(
    "model",
    [
        {
            "kind": "rectangular_aperture",
            "north_length_m": 14.0,
            "east_length_m": 12.0,
        },
        {
            "kind": "elliptical_aperture",
            "north_diameter_m": 14.0,
            "east_diameter_m": 12.0,
        },
        {
            "kind": "analytical_illumination",
            "illumination": {"kind": "corrugated_horn"},
        },
        {
            "kind": "numerical_illumination",
            "illumination": {"kind": "open_waveguide"},
        },
    ],
)
def test_accepted_analytic_variants_activate_canonical_runtime(
    tmp_path,
    model,
):
    data = _explicit_data(
        tmp_path,
        beams={"mode": "analytic", "model": model},
    )

    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator.setup()

    assert simulator.beam_system.state is simulator.beam_state
    assert simulator.beam_state.handlers[0].kind == "analytic"
    assert simulator._backend is not None
    assert simulator._is_setup is True


def test_beam_properties_fail_without_triggering_resolution_or_side_effects(
    tmp_path,
    monkeypatch,
):
    simulator = Simulator.from_mapping(_explicit_data(tmp_path), base_dir=tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("beam property access initiated work")

    monkeypatch.setattr(
        "radiosim.core.instrument_resolution.resolve_instrument",
        forbidden,
    )
    monkeypatch.setattr("radiosim.utils.device.get_device_resources", forbidden)
    monkeypatch.setattr("radiosim.backends.get_backend", forbidden)
    monkeypatch.setattr("radiosim.utils.network.get_network_status", forbidden)
    monkeypatch.setattr(
        "radiosim.core.sky.operations.parallel.load_models_parallel",
        forbidden,
    )

    for property_name in ("beam_system", "beam_state"):
        with pytest.raises(
            RuntimeError,
            match="^Beam resolution has not completed$",
        ):
            getattr(simulator, property_name)

    assert simulator._instrument_state is None
    assert simulator._beam_system is None


@pytest.mark.parametrize(
    ("taper", "expected", "edge"),
    [
        ({"kind": "uniform"}, "uniform", None),
        ({"kind": "gaussian", "edge_taper_db": 11.0}, "gaussian", 11.0),
        ({"kind": "parabolic", "edge_taper_db": 12.0}, "parabolic", 12.0),
        (
            {"kind": "parabolic_squared", "edge_taper_db": 13.0},
            "parabolic_squared",
            13.0,
        ),
        ({"kind": "cosine"}, "cosine", None),
    ],
)
def test_direct_circular_runtime_consumes_every_authored_taper_field(
    tmp_path,
    taper,
    expected,
    edge,
):
    data = _explicit_data(
        tmp_path,
        beams={
            "mode": "analytic",
            "model": {"kind": "circular_aperture", "taper": taper},
        },
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)

    simulator.setup()

    model = simulator.beam_state.resolved.assignments[0].definition.model
    assert model.kind == "circular_aperture"
    assert model.taper.kind == expected
    if edge is not None:
        assert model.taper.edge_taper_db == edge
    assert not hasattr(simulator, "_beam_config")


def test_setup_uses_resolved_backend_precision_frequency_and_runtime_fields(tmp_path):
    data = _explicit_data(tmp_path)
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)

    simulator.setup()

    assert simulator._backend.name == "numpy-cpu"
    assert simulator._backend.precision == PrecisionConfig.standard()
    assert simulator._simulator.name == "rime"
    np.testing.assert_array_equal(
        simulator._frequencies_hz,
        np.array([100e6, 101.25e6, 109e6]),
    )
    assert simulator._frequencies_hz.flags.owndata


def test_setup_resolves_receptors_between_instrument_and_beam_state(tmp_path):
    simulator = Simulator.from_mapping(_explicit_data(tmp_path), base_dir=tmp_path)

    with pytest.raises(RuntimeError, match="^Receptor resolution has not completed$"):
        _ = simulator.receptors

    simulator.setup()

    assert simulator.receptors.output_basis == "linear_xy"
    assert len(simulator.receptors.receptor_by_antenna) == len(simulator.antennas)
    assert len(simulator.receptors.provenance.receptor_sha256) == 64


def test_receptor_resolution_is_idempotent_and_retained(tmp_path):
    simulator = Simulator.from_mapping(_explicit_data(tmp_path), base_dir=tmp_path)

    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    first = simulator.receptors
    simulator._ensure_receptor_set()

    assert simulator.receptors is first

    simulator.setup()

    assert simulator.receptors is first


def test_non_default_receptor_configuration_resolves_through_setup(tmp_path):
    data = _explicit_data(tmp_path)
    data["receptors"] = {
        "default": {"basis": "circular", "feed_rotation_deg": 30.0},
        "output_basis": "circular",
    }
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)

    simulator.setup()

    resolved = simulator.receptors
    assert resolved.output_basis == "circular_rl"
    for receptor in resolved.receptor_by_antenna.values():
        assert receptor.basis == "circular"
        assert receptor.feed_array == ("r", "l")


def test_receptor_failure_precedes_beam_load_and_leaves_no_runtime_state(
    tmp_path,
    monkeypatch,
):
    from radiosim.core.receptor import ReceptorAssignmentError

    data = _explicit_data(tmp_path)
    data["receptors"] = {
        "overrides": [
            {"antenna": {"kind": "number", "number": 91}, "basis": "circular"}
        ]
    }
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("receptor resolution failure initiated later setup work")

    monkeypatch.setattr("radiosim.core.beam.load_beam_system", forbidden)
    monkeypatch.setattr("radiosim.core.beam.resolve_beam_assignments", forbidden)
    monkeypatch.setattr("radiosim.backends.get_backend", forbidden)
    monkeypatch.setattr("radiosim.utils.device.get_device_resources", forbidden)
    monkeypatch.setattr("radiosim.utils.network.get_network_status", forbidden)

    with pytest.raises(ReceptorAssignmentError):
        simulator.setup()

    assert simulator._receptor_set is None
    assert simulator._beam_system is None
    assert simulator._backend is None
    assert simulator._is_setup is False
    assert not (tmp_path / "output").exists()


def test_run_hands_the_resolved_receptor_set_to_the_point_solver(
    tmp_path,
    monkeypatch,
):
    simulator = Simulator.from_mapping(_explicit_data(tmp_path), base_dir=tmp_path)
    simulator.setup()
    captured: dict[str, object] = {}
    original = simulator._simulator.calculate_visibilities

    def record(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(simulator._simulator, "calculate_visibilities", record)
    simulator.run(progress=False)

    assert captured["receptors"] is simulator.receptors


def test_run_hands_the_resolved_receptor_set_to_the_healpix_solver(
    tmp_path,
    monkeypatch,
):
    data = _explicit_data(
        tmp_path,
        visibility={
            "calculation_type": "direct_sum",
            "sky_representation": "healpix_map",
        },
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator.setup()
    captured: dict[str, object] = {}
    healpix_module = importlib.import_module("radiosim.core.visibility_healpix")
    original = healpix_module.calculate_visibility_healpix

    def record(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(healpix_module, "calculate_visibility_healpix", record)
    simulator.run(progress=False)

    assert captured["receptors"] is simulator.receptors


def test_a_circular_receptor_configuration_changes_the_published_visibilities(
    tmp_path,
):
    """A configured basis reaches the result, and Tier 5E labels it honestly."""
    sources = [
        {
            "kind": "test_sources",
            "representation": "point_sources",
            "num_sources": 40,
            "distribution": "uniform",
            "seed": 5,
            "dec_deg": -30.0,
            "stokes_v_fraction": 0.9,
        }
    ]
    results = {}
    for basis in ("linear", "circular"):
        base_dir = tmp_path / basis
        base_dir.mkdir()
        data = _explicit_data(base_dir, sky_model={"sources": sources})
        data["receptors"] = {"default": {"basis": basis}}
        simulator = Simulator.from_mapping(data, base_dir=base_dir)
        results[basis] = simulator.run(progress=False)

    linear = np.asarray(results["linear"].visibilities)
    circular = np.asarray(results["circular"].visibilities)

    assert np.max(np.abs(linear)) > 0.0
    assert not np.allclose(linear, circular)
    # Stokes V lands in the cross hands of a linear array and in the parallel
    # hands of a circular one.
    assert np.max(np.abs(circular[..., 1])) < 1e-12
    assert np.max(np.abs(linear[..., 1])) > 1e-6
    # Total intensity is basis independent (Section 18.6).
    np.testing.assert_allclose(
        circular[..., 0] + circular[..., 3],
        linear[..., 0] + linear[..., 3],
        rtol=1e-10,
        atol=1e-12,
    )
    # FLIPPED BY: Tier 5E.  The correlation coordinates are now derived from
    # the resolved receptor output basis at every construction site, so a
    # circular run no longer publishes linear labels.
    assert results["linear"].correlations == ("XX", "XY", "YX", "YY")
    assert results["linear"].polarization_basis == "linear_xy"
    assert results["circular"].correlations == ("RR", "RL", "LR", "LL")
    assert results["circular"].polarization_basis == "circular_rl"
    for basis, result in results.items():
        assert result.receptors.output_basis == result.polarization_basis
        assert result.receptors.native_basis_counts[basis] == len(
            result.instrument.antennas
        )
    assert results["linear"].scientific_sha256 != results["circular"].scientific_sha256


def test_observability_resolves_receptors_before_beam_work(tmp_path):
    simulator = Simulator.from_mapping(_explicit_data(tmp_path), base_dir=tmp_path)

    simulator.plan_observability(channel_index=0)

    assert simulator.receptors.output_basis == "linear_xy"


def test_setup_passes_resolved_glob_matches_without_re_globbing(tmp_path, monkeypatch):
    for name in ("b.skyh5", "a.skyh5"):
        (tmp_path / name).touch()
    data = _explicit_data(
        tmp_path,
        sky_model={"sources": [{"kind": "skyh5_multifile", "file_glob": "*.skyh5"}]},
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    captured: dict[str, object] = {}

    class CapturedLoaderRequests(Exception):
        pass

    def capture_requests(requests, **kwargs):
        captured["requests"] = requests
        raise CapturedLoaderRequests

    monkeypatch.setattr(
        "radiosim.core.sky.operations.parallel.load_models_parallel",
        capture_requests,
    )

    with pytest.raises(CapturedLoaderRequests):
        simulator.setup()

    [(kind, loader_kwargs)] = captured["requests"]
    assert kind == "skyh5_multifile"
    assert "file_glob" not in loader_kwargs
    assert loader_kwargs["filenames"] == [
        str((tmp_path / "a.skyh5").resolve()),
        str((tmp_path / "b.skyh5").resolve()),
    ]


def test_result_metadata_uses_json_safe_scientific_snapshot_without_workflow(
    tmp_path,
):
    data = _explicit_data(
        tmp_path,
        workflow={"result_filename": "workflow-name", "save_results": True},
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)

    result = simulator.run(progress=False)

    assert result.instrument is simulator.instrument
    assert result.selection is simulator._instrument_state.selection
    assert result.beam_state is simulator.beam_state
    assert result.backend.requested_backend == "numpy"
    assert result.backend.actual_backend == "numpy-cpu"
    assert result.backend.requested_precision == result.backend.actual_precision
    assert result.resolved_config["frequency"]["channel_frequencies_hz"] == (
        100e6,
        101.25e6,
        109e6,
    )
    assert "workflow" not in result.resolved_config
    json.dumps(result.to_summary_snapshot(), allow_nan=False)

    later = simulator.run(progress=False)
    assert later is simulator.result
    assert later is not result
    assert later.beam_state is simulator.beam_state


def test_save_rejects_absent_result_before_writer_or_filesystem_work(
    tmp_path, monkeypatch
):
    from radiosim.core.result import ResultUnavailableError

    data = _explicit_data(
        tmp_path,
        workflow={
            "result_filename": "workflow-name",
            "result_format": "summary_json",
        },
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("save crossed a side-effect boundary")

    monkeypatch.setattr(Path, "mkdir", forbidden)

    with pytest.raises(ResultUnavailableError, match="no successfully published"):
        simulator.save(tmp_path / "saved")


@pytest.mark.parametrize(
    ("format_name", "module_name", "writer_name", "extension"),
    [
        ("HDF5", "radiosim.io.hdf5", "write_result_hdf5", ".h5"),
        (
            "SUMMARY_JSON",
            "radiosim.io.summary_json",
            "write_result_summary_json",
            ".summary.json",
        ),
        ("MS", "radiosim.io.measurement_set", "write_measurement_set", ".ms"),
        ("UVFITS", "radiosim.io.uvfits", "write_uvfits", ".uvfits"),
    ],
)
def test_save_dispatches_exact_typed_format_to_final_artifact(
    tmp_path,
    monkeypatch,
    format_name,
    module_name,
    writer_name,
    extension,
):
    from radiosim.io.result_format import ResultFormat
    from tests.unit.test_core.test_result import _build

    result, _ = _build(tmp_path)
    simulator = object.__new__(Simulator)
    simulator._result = result
    calls = []

    def writer(observed_result, path, *, overwrite):
        calls.append((observed_result, path, overwrite))
        return Path(path)

    monkeypatch.setattr(importlib.import_module(module_name), writer_name, writer)
    selected = getattr(ResultFormat, format_name)
    target = simulator.save(
        tmp_path / "artifact",
        format=selected,
        overwrite=True,
    )

    assert target == tmp_path / f"artifact{extension}"
    assert calls == [(result, target, True)]


def test_save_default_is_hdf5_and_rejects_strings_before_writer_import(
    tmp_path, monkeypatch
):
    from radiosim.io.result_format import ResultFormat
    from tests.unit.test_core.test_result import _build

    result, _ = _build(tmp_path)
    simulator = object.__new__(Simulator)
    simulator._result = result
    calls = []

    def writer(observed_result, path, *, overwrite):
        calls.append((observed_result, path, overwrite))
        return Path(path)

    monkeypatch.setattr("radiosim.io.hdf5.write_result_hdf5", writer)
    assert simulator.save(tmp_path / "default") == tmp_path / "default.h5"
    assert calls == [(result, tmp_path / "default.h5", False)]

    with pytest.raises(TypeError, match="ResultFormat"):
        simulator.save(tmp_path / "invalid", format="hdf5")
    assert not (tmp_path / "invalid.h5").exists()
    assert ResultFormat.HDF5.value == "hdf5"


def test_run_rejects_ignored_worker_control_before_setup(tmp_path):
    simulator = Simulator(resolved_config(tmp_path).runtime)

    with pytest.raises(NotImplementedError, match=r"run\(n_workers"):
        simulator.run(n_workers=2)

    assert simulator.device_resources is None


def test_api_local_examples_advertise_only_the_new_construction_contract():
    module = importlib.import_module("radiosim.api.simulator")
    source = inspect.getsource(module)

    assert "Simulator.from_yaml" in source
    assert "Simulator.from_mapping" in source
    assert 'Simulator.from_config("config.yaml")' not in source
    assert "Simulator(config=" not in source
    assert "frequencies=[" not in source


def test_repr_reports_requested_backend_before_setup(tmp_path):
    simulator = Simulator.from_mapping(
        _explicit_data(tmp_path, execution={"backend": "auto"}),
        base_dir=tmp_path,
    )

    assert "configured" in repr(simulator)
    assert "backend=auto" in repr(simulator)


def _beam_mode_input(tmp_path: Path, mode: str) -> dict[str, object]:
    if mode == "analytic":
        return {
            "mode": "analytic",
            "model": {
                "kind": "circular_aperture",
                "taper": {"kind": "uniform"},
            },
        }
    beam_path = write_scalar_efield_beamfits(tmp_path).path
    fits = {"kind": "fits", "path": beam_path.name}
    if mode == "shared_fits":
        return {"mode": "shared_fits", "beam": fits}
    if mode == "per_antenna_fits":
        return {
            "mode": "per_antenna_fits",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": fits,
                },
                {
                    "antenna": {"kind": "number", "number": 1},
                    "beam": fits,
                },
            ],
        }
    assert mode == "mixed"
    return {
        "mode": "mixed",
        "analytic_model": {
            "kind": "circular_aperture",
            "taper": {"kind": "uniform"},
        },
        "assignments": [
            {
                "antenna": {"kind": "number", "number": 0},
                "beam": {"kind": "analytic"},
            },
            {
                "antenna": {"kind": "number", "number": 1},
                "beam": fits,
            },
        ],
    }


@pytest.mark.parametrize(
    "mode",
    ("analytic", "shared_fits", "per_antenna_fits", "mixed"),
)
def test_point_results_include_fresh_beam_resolution_for_every_mode(
    tmp_path,
    mode,
):
    simulator = Simulator.from_mapping(
        _explicit_data(tmp_path, beams=_beam_mode_input(tmp_path, mode)),
        base_dir=tmp_path,
    )

    first = simulator.run(progress=False)
    assert first.beam_state is simulator.beam_state
    first_snapshot = first.beam_state.to_snapshot()
    assert "reference_antenna" not in first_snapshot

    second = simulator.run(progress=False)
    assert second.beam_state is simulator.beam_state
    assert second.beam_state.to_snapshot() is not first_snapshot


def test_healpix_results_include_fresh_beam_resolution(tmp_path):
    data = _explicit_data(
        tmp_path,
        visibility={
            "calculation_type": "direct_sum",
            "sky_representation": "healpix_map",
        },
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)

    results = simulator.run(progress=False)

    assert simulator._sky_model.healpix is not None
    assert results.beam_state is simulator.beam_state
    json.dumps(results.to_summary_snapshot(), allow_nan=False)


def test_failed_solver_publishes_no_partial_result_metadata(tmp_path, monkeypatch):
    simulator = Simulator.from_mapping(_explicit_data(tmp_path), base_dir=tmp_path)
    simulator.setup()

    def fail_solver(*args, **kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(
        simulator._simulator,
        "calculate_visibilities",
        fail_solver,
    )

    with pytest.raises(RuntimeError, match="solver failed"):
        simulator.run(progress=False)

    assert simulator.result is None


def test_sampling_derivation_precedes_device_backend_network_and_sky(
    tmp_path,
    monkeypatch,
):
    simulator = Simulator.from_mapping(_explicit_data(tmp_path), base_dir=tmp_path)
    events: list[str] = []
    healpix_module = importlib.import_module("radiosim.utils.healpix")
    original_device = importlib.import_module(
        "radiosim.utils.device"
    ).get_device_resources

    def derive(**kwargs):
        events.append("derive")
        assert kwargs["selected_baselines"] == simulator.baselines
        assert kwargs["beam_state"] is simulator.beam_state
        assert kwargs["observation_frequencies_hz"] == (
            simulator.config.frequency.channel_frequencies_hz
        )
        nside = kwargs["actual_nside"]
        pixel = float(hp.nside2resol(nside))
        return SimpleNamespace(
            actual_nside=nside,
            recommended_nside=nside,
            actual_pixel_scale_rad=pixel,
            product_feature_scale_rad=pixel * 10.0,
            pixel_limit_rad=pixel * 2.0,
            baseline_ant1=simulator.baselines[0].ant1,
            baseline_ant2=simulator.baselines[0].ant2,
            frequency_hz=simulator.config.frequency.channel_frequencies_hz[0],
            handler_id_p="p",
            handler_id_q="q",
            metric_kind="analytic_aperture_support",
            safety_factor=5,
        )

    def device():
        events.append("device")
        return original_device()

    monkeypatch.setattr(
        healpix_module,
        "derive_beam_sampling_requirement",
        derive,
        raising=False,
    )
    monkeypatch.setattr("radiosim.utils.device.get_device_resources", device)
    monkeypatch.setattr(
        "radiosim.backends.get_backend",
        lambda *args, **kwargs: (
            events.append("backend")
            or importlib.import_module("radiosim.backends.numpy_backend").NumPyBackend(
                precision=kwargs["precision"]
            )
        ),
    )
    original_network = importlib.import_module(
        "radiosim.utils.network"
    ).get_network_status

    def network(*args, **kwargs):
        events.append("network")
        return original_network(*args, **kwargs)

    monkeypatch.setattr("radiosim.utils.network.get_network_status", network)
    original_load = importlib.import_module(
        "radiosim.core.sky.operations.parallel"
    ).load_models_parallel

    def load_sky(*args, **kwargs):
        events.append("sky")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(
        "radiosim.core.sky.operations.parallel.load_models_parallel",
        load_sky,
    )

    simulator.setup()

    assert events[0] == "derive"
    assert events.index("derive") < events.index("device")
    assert events.index("derive") < events.index("backend")
    assert events.index("derive") < events.index("network")
    assert events.index("derive") < events.index("sky")


def test_invalid_sampling_derivation_retries_beam_construction_before_side_effects(
    tmp_path,
    monkeypatch,
):
    from radiosim.core.beam import BeamSamplingDerivationError

    simulator = Simulator.from_mapping(_explicit_data(tmp_path), base_dir=tmp_path)
    beam_module = importlib.import_module("radiosim.core.beam")
    healpix_module = importlib.import_module("radiosim.utils.healpix")
    original_load_beam_system = beam_module.load_beam_system
    load_count = 0

    def load_beam_system(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        return original_load_beam_system(*args, **kwargs)

    def invalid_derivation(**kwargs):
        raise BeamSamplingDerivationError("invalid canonical sampling state")

    def forbidden(*args, **kwargs):
        pytest.fail("sampling failure crossed a runtime side-effect boundary")

    monkeypatch.setattr(beam_module, "load_beam_system", load_beam_system)
    monkeypatch.setattr(
        healpix_module,
        "derive_beam_sampling_requirement",
        invalid_derivation,
        raising=False,
    )
    monkeypatch.setattr("radiosim.utils.device.get_device_resources", forbidden)
    monkeypatch.setattr("radiosim.backends.get_backend", forbidden)
    monkeypatch.setattr("radiosim.utils.network.get_network_status", forbidden)
    monkeypatch.setattr(
        "radiosim.core.sky.operations.parallel.load_models_parallel",
        forbidden,
    )

    for expected_count in (1, 2):
        with pytest.raises(
            BeamSamplingDerivationError,
            match="invalid canonical sampling state",
        ):
            simulator.setup()
        assert simulator._instrument_state is not None
        assert simulator._beam_system is None
        assert simulator.device_resources is None
        assert load_count == expected_count


def test_coarse_pre_sky_warning_is_exact_ordered_and_never_mutates_nside(
    tmp_path,
    monkeypatch,
    caplog,
):
    data = _explicit_data(
        tmp_path,
        beams={
            "mode": "analytic",
            "model": {
                "kind": "rectangular_aperture",
                "north_length_m": 100.0,
                "east_length_m": 80.0,
            },
        },
        visibility={
            "calculation_type": "direct_sum",
            "sky_representation": "healpix_map",
        },
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    events: list[str] = []
    simulator_module = importlib.import_module("radiosim.api.simulator")
    original_warning = simulator_module.logger.warning
    original_device = importlib.import_module(
        "radiosim.utils.device"
    ).get_device_resources

    def warning(*args, **kwargs):
        events.append("warning")
        return original_warning(*args, **kwargs)

    def device():
        events.append("device")
        return original_device()

    monkeypatch.setattr(simulator_module.logger, "warning", warning)
    monkeypatch.setattr("radiosim.utils.device.get_device_resources", device)

    with caplog.at_level(logging.WARNING, logger="radiosim.api.simulator"):
        simulator.setup()

    requirement = importlib.import_module(
        "radiosim.utils.healpix"
    ).derive_beam_sampling_requirement(
        selected_baselines=simulator.baselines,
        beam_state=simulator.beam_state,
        observation_frequencies_hz=(simulator.config.frequency.channel_frequencies_hz),
        actual_nside=64,
    )
    expected = (
        f"HEALPix nside=64 has pixel scale "
        f"{requirement.actual_pixel_scale_rad:.6g} rad, above the Tier 3 "
        f"beam-product limit {requirement.pixel_limit_rad:.6g} rad "
        f"(smallest feature {requirement.product_feature_scale_rad:.6g} rad, "
        f"safety factor 5, baseline "
        f"{requirement.baseline_ant1.number}:{requirement.baseline_ant1.name}-"
        f"{requirement.baseline_ant2.number}:{requirement.baseline_ant2.name}, "
        f"frequency {requirement.frequency_hz:.6g} Hz). Use at least "
        f"nside={requirement.recommended_nside}; the requested NSIDE is unchanged."
    )
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "radiosim.api.simulator" and record.levelno == logging.WARNING
    ]

    assert messages == [expected]
    assert events.index("warning") < events.index("device")
    assert simulator._sky_model.healpix.nside == 64


def test_post_sky_warning_uses_actual_loaded_nside_without_mutation(
    tmp_path,
    monkeypatch,
    caplog,
):
    data = _explicit_data(
        tmp_path,
        visibility={
            "calculation_type": "direct_sum",
            "sky_representation": "healpix_map",
        },
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    pipeline = importlib.import_module("radiosim.core.sky.combine.pipeline")
    original_prepare = pipeline.prepare_sky_model
    requested_nsides: list[int] = []

    def prepare_with_actual_nside(*args, **kwargs):
        requested_nsides.append(kwargs["nside"])
        kwargs["nside"] = 32
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(pipeline, "prepare_sky_model", prepare_with_actual_nside)

    with caplog.at_level(logging.WARNING, logger="radiosim.api.simulator"):
        simulator.setup()

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "radiosim.api.simulator" and record.levelno == logging.WARNING
    ]
    assert requested_nsides == [64]
    assert simulator._sky_model.healpix.nside == 32
    assert len(messages) == 1
    assert messages[0].startswith("HEALPix nside=32 has pixel scale ")
    assert messages[0].endswith("the requested NSIDE is unchanged.")


def test_point_only_setup_derives_once_and_emits_no_nside_warning(
    tmp_path,
    monkeypatch,
    caplog,
):
    simulator = Simulator.from_mapping(_explicit_data(tmp_path), base_dir=tmp_path)
    healpix_module = importlib.import_module("radiosim.utils.healpix")
    original_derive = healpix_module.derive_beam_sampling_requirement
    actual_nsides: list[int] = []

    def derive(**kwargs):
        actual_nsides.append(kwargs["actual_nside"])
        return original_derive(**kwargs)

    monkeypatch.setattr(
        healpix_module,
        "derive_beam_sampling_requirement",
        derive,
    )

    with caplog.at_level(logging.WARNING, logger="radiosim.api.simulator"):
        simulator.setup()

    assert actual_nsides == [64]
    assert simulator._sky_model.healpix is None
    assert not any(
        record.name == "radiosim.api.simulator" and record.levelno == logging.WARNING
        for record in caplog.records
    )


def test_approximate_fwhm_advisor_is_absent_from_simulator_source() -> None:
    source = inspect.getsource(importlib.import_module("radiosim.api.simulator"))

    assert "beam_fwhm_rad" not in source
    assert "nside_safety_factor" not in source
    assert "1.22" not in source
    assert "d_min" not in source
    assert "lam_max" not in source
