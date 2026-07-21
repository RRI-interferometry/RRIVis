"""Tests for the resolved-only public Simulator construction contract."""

from __future__ import annotations

import importlib
import inspect
import json
import re
import webbrowser
from dataclasses import FrozenInstanceError
from pathlib import Path

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
    UnsupportedConfigError,
)
from radiosim.io.instrument_config import (
    BaselineSelectionConfig,
    InstrumentConfig,
)
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
            "result_format": "json",
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
    assert simulator.results is None


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
def test_tier3b_fits_runtime_guards_apply_to_every_document_entry_point(
    tmp_path, monkeypatch, entry_point
):
    beam_path = tmp_path / "pending.beamfits"
    beam_path.touch()
    data = _explicit_data(
        tmp_path,
        beams={
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": beam_path.name},
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Tier 0 guard reached device detection")

    monkeypatch.setattr("radiosim.utils.device.get_device_resources", forbidden)

    with pytest.raises(
        UnsupportedConfigError,
        match=re.escape("beam_runtime_fits_pending"),
    ) as exc_info:
        if entry_point == "yaml":
            Simulator.from_yaml(write_config_yaml(tmp_path, data))
        elif entry_point == "model":
            Simulator.from_config(
                RadioSimConfig.model_validate(data),
                base_dir=tmp_path,
            )
        elif entry_point == "mapping":
            Simulator.from_mapping(data, base_dir=tmp_path)
        else:
            _from_parameters(tmp_path, data)

    message = exc_info.value.issues[0].message
    assert "later Tier 3 slice" in message
    assert "Tier 3C" not in message


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
def test_pending_analytic_variants_fail_before_device_backend_network_and_sky(
    tmp_path, monkeypatch, model
):
    data = _explicit_data(
        tmp_path,
        beams={"mode": "analytic", "model": model},
    )

    def forbidden(*args, **kwargs):
        pytest.fail("pending beam mode crossed a runtime side-effect boundary")

    monkeypatch.setattr("radiosim.utils.device.get_device_resources", forbidden)
    monkeypatch.setattr("radiosim.backends.get_backend", forbidden)
    monkeypatch.setattr("radiosim.utils.network.get_network_status", forbidden)
    monkeypatch.setattr(
        "radiosim.core.sky.operations.parallel.load_models_parallel", forbidden
    )

    with pytest.raises(UnsupportedConfigError) as exc_info:
        Simulator.from_mapping(data, base_dir=tmp_path)

    assert [issue.code for issue in exc_info.value.issues] == [
        "beam_runtime_analytic_variant_pending"
    ]
    assert exc_info.value.issues[0].path == "beams.model.kind"


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
def test_direct_circular_projection_consumes_every_authored_taper_field(
    tmp_path, taper, expected, edge
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

    assert simulator._beam_config["aperture_shape"] == "circular"
    assert simulator._beam_config["taper"] == expected
    if edge is None:
        assert "edge_taper_dB" not in simulator._beam_config
    else:
        assert simulator._beam_config["edge_taper_dB"] == edge


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

    results = simulator.run(progress=False)
    metadata = results["metadata"]

    assert results["antennas"] is simulator.antennas
    assert results["baselines"] is simulator.baselines
    assert metadata["requested_backend"] == "numpy"
    assert metadata["backend"] == "numpy-cpu"
    assert metadata["requested_precision"] == metadata["precision"]
    assert metadata["config"]["frequency"]["channel_frequencies_hz"] == [
        100e6,
        101.25e6,
        109e6,
    ]
    assert "workflow" not in metadata["config"]
    resolution = metadata["instrument_resolution"]
    assert tuple(resolution) == (
        "schema_version",
        "instrument_sha256",
        "name",
        "source",
        "location",
        "antennas",
        "baseline_selection",
    )
    assert resolution["instrument_sha256"] == (
        simulator.instrument.provenance.instrument_sha256
    )
    assert resolution["baseline_selection"]["selected_ids"] == [
        [baseline.ant1.number, baseline.ant2.number] for baseline in simulator.baselines
    ]
    json.dumps(metadata, allow_nan=False)

    resolution["antennas"][0]["name"] = "mutated snapshot"
    assert simulator.antennas[0].id.name != "mutated snapshot"


def test_save_uses_only_explicit_output_choices_not_workflow(tmp_path, monkeypatch):
    data = _explicit_data(
        tmp_path,
        workflow={"result_filename": "workflow-name", "result_format": "json"},
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator._results = {
        "visibilities": {},
        "frequencies": np.array([100e6]),
        "baselines": {},
        "metadata": {},
    }
    captured: dict[str, object] = {}

    def record_writer(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("radiosim.io.writers.save_visibilities_hdf5", record_writer)

    output = simulator.save(tmp_path / "saved")

    assert output.name == "visibilities.h5"
    assert captured["output_path"] == output


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
