"""Contract tests for divergent current configuration entry points."""

from __future__ import annotations

import builtins
import socket
import webbrowser
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from radiosim.api import Simulator
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.registry.facade import SkyLoaderRegistry
from radiosim.io.config import (
    ExplicitFrequencyConfig,
    PrecisionInput,
    RadioSimConfig,
    TestSourcesConfig,
    collect_semantic_issues,
    load_config,
)
from radiosim.io.config_resolution import (
    ConfigOverrideError,
    ConfigParseError,
    ConfigPathError,
    ConfigSchemaError,
    ConfigSemanticError,
    ConfigSourceError,
    ConfigurationSource,
    InstrumentSourcePathOverride,
    SimulationOverrides,
    UnsupportedConfigError,
    WorkflowOverrides,
    resolve_config,
)
from radiosim.io.instrument_config import InstrumentLocationConfig
from tests.fixtures.configs import valid_config_mapping, write_config_yaml


class _DeviceSentinel:
    def summary(self) -> str:
        return "Tier 1A sentinel"


def test_tier1b_semantic_validation_is_pure_and_explicit(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["obs_time"]["duration_seconds"] = 1.0
    data["obs_time"]["time_step_seconds"] = 2.0

    config = RadioSimConfig.model_validate(data)
    issues = collect_semantic_issues(config)

    assert [issue.path for issue in issues] == ["obs_time.time_step_seconds"]


def test_from_yaml_runs_common_path_preflight(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["instrument"]["source"]["path"] = "missing.txt"
    config_path = write_config_yaml(tmp_path, data)

    with pytest.raises(ConfigPathError, match="instrument.source.path"):
        Simulator.from_yaml(config_path)


def test_from_mapping_runs_schema_validation(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["instrument"]["location"]["latitude_deg"] = float("inf")

    with pytest.raises(ConfigSchemaError, match="instrument.location.latitude_deg"):
        Simulator.from_mapping(data, base_dir=tmp_path)


def test_invalid_mapping_fails_before_device_detection(tmp_path, monkeypatch):
    calls: list[str] = []

    def detect_device() -> _DeviceSentinel:
        calls.append("device")
        return _DeviceSentinel()

    monkeypatch.setattr(
        "radiosim.utils.device.get_device_resources",
        detect_device,
    )
    data = valid_config_mapping(tmp_path)
    data["obs_time"]["duration_seconds"] = -1.0

    with pytest.raises(ConfigSchemaError, match="obs_time.duration_seconds"):
        Simulator.from_mapping(data, base_dir=tmp_path)

    assert calls == []


def test_from_config_rejects_semantic_errors_before_construction(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["obs_time"]["duration_seconds"] = 1.0
    data["obs_time"]["time_step_seconds"] = 2.0
    config = RadioSimConfig.model_validate(data)

    with pytest.raises(ConfigSemanticError, match="obs_time.time_step_seconds"):
        Simulator.from_config(config, base_dir=tmp_path)


def test_from_mapping_rejects_invalid_input_at_public_boundary(tmp_path):
    with pytest.raises(ConfigSchemaError, match="location"):
        Simulator.from_mapping({"location": {"lat": 999.0}}, base_dir=tmp_path)


def test_equivalent_yaml_mapping_and_model_resolve_identically(tmp_path):
    data = valid_config_mapping(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [100e6, 101.25e6, 109e6],
        },
        execution={"backend": "jax", "offline": True},
        workflow={"output_dir": "document-output"},
    )
    data["instrument"]["source"]["path"] = "antennas.txt"
    config_path = write_config_yaml(tmp_path, data)
    model = RadioSimConfig.model_validate(data)
    overrides = SimulationOverrides(
        backend="auto",
        offline=False,
        precision=PrecisionInput(preset="fast"),
    )
    workflow_overrides = WorkflowOverrides(output_dir=tmp_path / "override-output")

    yaml_bundle = load_config(
        config_path,
        overrides=overrides,
        workflow_overrides=workflow_overrides,
    )
    mapping_bundle = resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
        overrides=overrides,
        workflow_overrides=workflow_overrides,
    )
    model_bundle = resolve_config(
        model,
        source=ConfigurationSource.for_model(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
        overrides=overrides,
        workflow_overrides=workflow_overrides,
    )

    assert yaml_bundle.runtime == mapping_bundle.runtime == model_bundle.runtime
    assert yaml_bundle.workflow == mapping_bundle.workflow == model_bundle.workflow
    assert yaml_bundle.runtime.frequency.channel_frequencies_hz == (
        100e6,
        101.25e6,
        109e6,
    )
    assert yaml_bundle.runtime.execution.backend_strategy == "auto"
    assert yaml_bundle.runtime.execution.offline is False
    assert yaml_bundle.runtime.execution.precision == PrecisionConfig.fast()
    assert yaml_bundle.workflow.output_dir == tmp_path / "override-output"
    assert {
        bundle.provenance.source.kind
        for bundle in (
            yaml_bundle,
            mapping_bundle,
            model_bundle,
        )
    } == {"yaml", "mapping", "model"}


def test_typed_model_normalization_preserves_defaulted_union_discriminators(
    tmp_path,
):
    config = RadioSimConfig.model_validate(valid_config_mapping(tmp_path))
    config = config.model_copy(
        update={
            "sky_model": config.sky_model.model_copy(
                update={"sources": (TestSourcesConfig(),)}
            ),
            "obs_frequency": ExplicitFrequencyConfig(
                channel_frequencies_hz=(100e6, 101.25e6, 109e6)
            ),
        }
    )

    bundle = resolve_config(
        config,
        source=ConfigurationSource.for_model(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )

    assert bundle.runtime.frequency.channel_frequencies_hz == (
        100e6,
        101.25e6,
        109e6,
    )
    assert bundle.runtime.sky_model.sources[0].kind == "test_sources"


def test_timestamp_normalization_is_shared_by_all_source_types(tmp_path):
    timestamp = datetime(
        2025,
        1,
        2,
        12,
        34,
        56,
        123456,
        tzinfo=timezone(timedelta(hours=5, minutes=30)),
    )
    datetime_data = valid_config_mapping(tmp_path)
    datetime_data["obs_time"]["start_time"] = timestamp
    string_data = valid_config_mapping(tmp_path)
    string_data["obs_time"]["start_time"] = timestamp.isoformat()
    datetime_path = write_config_yaml(
        tmp_path,
        datetime_data,
        name="datetime.yaml",
    )
    string_path = write_config_yaml(tmp_path, string_data, name="string.yaml")
    model = RadioSimConfig.model_validate(string_data)
    mapping_source = ConfigurationSource.for_mapping(
        base_dir=tmp_path,
        invocation_dir=tmp_path,
    )
    model_source = ConfigurationSource.for_model(
        base_dir=tmp_path,
        invocation_dir=tmp_path,
    )

    bundles = (
        load_config(datetime_path),
        load_config(string_path),
        resolve_config(datetime_data, source=mapping_source),
        resolve_config(model, source=model_source),
    )

    normalized = {bundle.runtime.observation.start_time_iso for bundle in bundles}
    assert normalized == {"2025-01-02T07:04:56.123"}
    assert (
        bundles[2].provenance.input_snapshot["obs_time"]["start_time"]
        == "2025-01-02T07:04:56.123456"
    )
    assert datetime_data["obs_time"]["start_time"] is timestamp


def test_invalid_timestamp_values_use_schema_or_semantic_errors(tmp_path):
    source = ConfigurationSource.for_mapping(
        base_dir=tmp_path,
        invocation_dir=tmp_path,
    )
    schema_data = valid_config_mapping(tmp_path)
    schema_data["obs_time"]["start_time"] = {"not": "a scalar"}
    semantic_data = valid_config_mapping(tmp_path)
    semantic_data["obs_time"]["start_time"] = "not-a-time"

    with pytest.raises(ConfigSchemaError):
        resolve_config(schema_data, source=source)
    with pytest.raises(ConfigSemanticError):
        resolve_config(semantic_data, source=source)


def test_yaml_syntax_root_and_empty_document_errors_are_typed(tmp_path, capsys):
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("antenna_layout: [\n")
    sequence = tmp_path / "sequence.yaml"
    sequence.write_text("- one\n- two\n")
    empty = tmp_path / "empty.yaml"
    empty.write_text("")

    with pytest.raises(ConfigParseError) as syntax_error:
        load_config(invalid)
    with pytest.raises(ConfigParseError) as root_error:
        load_config(sequence)
    with pytest.raises(ConfigSchemaError) as empty_error:
        load_config(empty)

    assert [issue.code for issue in syntax_error.value.issues] == ["yaml_syntax_error"]
    assert [issue.code for issue in root_error.value.issues] == [
        "yaml_root_not_mapping"
    ]
    assert {issue.path for issue in empty_error.value.issues} >= {
        "instrument",
        "obs_frequency",
        "obs_time",
        "sky_model",
    }
    assert capsys.readouterr() == ("", "")


def test_unsupported_source_shapes_raise_ordered_parse_issues_without_mutation(
    tmp_path,
):
    data = valid_config_mapping(tmp_path)
    first = object()
    second = {"unordered"}
    data["first_bad"] = first
    data["second_bad"] = second
    source = ConfigurationSource.for_mapping(
        base_dir=tmp_path,
        invocation_dir=tmp_path,
    )

    with pytest.raises(ConfigParseError) as exc_info:
        resolve_config(data, source=source)

    assert [(issue.path, issue.code) for issue in exc_info.value.issues] == [
        ("first_bad", "unsupported_source_value"),
        ("second_bad", "unsupported_source_value"),
    ]
    assert data["first_bad"] is first
    assert data["second_bad"] is second


def test_semantic_errors_fail_before_device_detection(tmp_path, monkeypatch):
    calls: list[str] = []

    def detect_device() -> _DeviceSentinel:
        calls.append("device")
        return _DeviceSentinel()

    monkeypatch.setattr(
        "radiosim.utils.device.get_device_resources",
        detect_device,
    )
    data = valid_config_mapping(tmp_path)
    data["obs_time"]["duration_seconds"] = 1.0
    data["obs_time"]["time_step_seconds"] = 2.0

    with pytest.raises(ConfigSemanticError):
        Simulator.from_mapping(data, base_dir=tmp_path)

    assert calls == []


def test_yaml_source_path_is_observable_without_repository_cwd(tmp_path, monkeypatch):
    data = valid_config_mapping(tmp_path)
    data["instrument"]["source"]["path"] = "antennas.txt"
    config_path = write_config_yaml(tmp_path, data)
    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    simulator = Simulator.from_yaml(config_path)

    assert (
        simulator.config.instrument.source.path == (tmp_path / "antennas.txt").resolve()
    )


def test_configuration_source_is_frozen_and_normalizes_yaml_base(tmp_path):
    config_path = write_config_yaml(tmp_path)

    source = ConfigurationSource.for_yaml(
        config_path,
        invocation_dir=tmp_path,
        label="fixture yaml",
    )

    assert source.config_path == config_path.resolve()
    assert source.base_dir == tmp_path.resolve()
    assert source.invocation_dir == tmp_path.resolve()
    assert source.label == "fixture yaml"
    with pytest.raises(FrozenInstanceError):
        source.base_dir = Path("elsewhere")


@pytest.mark.parametrize(
    "kwargs, code",
    [
        ({"kind": "yaml"}, "missing_yaml_config_path"),
        (
            {"kind": "mapping", "config_path": "config.yaml"},
            "config_path_not_allowed",
        ),
        ({"kind": "model", "label": "   "}, "empty_source_label"),
    ],
)
def test_invalid_source_combinations_are_typed(tmp_path, kwargs, code):
    if kwargs.get("config_path"):
        (tmp_path / "config.yaml").touch()
    kwargs.setdefault("invocation_dir", tmp_path)

    with pytest.raises(ConfigSourceError) as exc_info:
        ConfigurationSource(**kwargs)

    assert code in {issue.code for issue in exc_info.value.issues}


def test_mapping_relative_document_paths_require_explicit_base(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["instrument"]["source"]["path"] = "antennas.txt"

    with pytest.raises(ConfigSourceError) as exc_info:
        resolve_config(
            data,
            source=ConfigurationSource.for_mapping(invocation_dir=tmp_path),
        )

    assert any(
        issue.path == "instrument.source.path"
        and issue.code == "relative_path_requires_base_dir"
        for issue in exc_info.value.issues
    )


def test_absolute_mapping_needs_no_base_when_workflow_is_also_absolute(tmp_path):
    data = valid_config_mapping(tmp_path)

    bundle = resolve_config(
        data,
        source=ConfigurationSource.for_mapping(invocation_dir=tmp_path),
    )

    assert bundle.runtime.instrument.source.path.is_absolute()
    assert bundle.workflow.output_dir.is_absolute()


def test_invocation_directory_is_captured_once_for_overrides(tmp_path, monkeypatch):
    yaml_dir = tmp_path / "yaml"
    invocation_dir = tmp_path / "invocation"
    later_dir = tmp_path / "later"
    for directory in (yaml_dir, invocation_dir, later_dir):
        directory.mkdir()
    data = valid_config_mapping(yaml_dir)
    override_file = invocation_dir / "override.txt"
    override_file.write_text((yaml_dir / "antennas.txt").read_text())
    source = ConfigurationSource.for_yaml(
        write_config_yaml(yaml_dir, data),
        invocation_dir=invocation_dir,
    )
    monkeypatch.chdir(later_dir)

    bundle = resolve_config(
        data,
        source=source,
        overrides=SimulationOverrides(
            instrument_source=InstrumentSourcePathOverride(path="override.txt")
        ),
        workflow_overrides=WorkflowOverrides(output_dir="override-output"),
    )

    assert bundle.runtime.instrument.source.path == override_file
    assert bundle.workflow.output_dir == invocation_dir / "override-output"
    assert not bundle.workflow.output_dir.exists()


def test_instrument_source_path_override_preserves_every_other_source_fact(tmp_path):
    data = valid_config_mapping(tmp_path)
    original = RadioSimConfig.model_validate(data).instrument
    override_path = tmp_path / "replacement.txt"
    override_path.write_text((tmp_path / "antennas.txt").read_text())

    bundle = resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
        overrides=SimulationOverrides(
            instrument_source=InstrumentSourcePathOverride(path=override_path)
        ),
    )

    assert bundle.runtime.instrument.source.path == override_path
    assert bundle.runtime.instrument.source.format == original.source.format
    assert (
        bundle.runtime.instrument.source.telescope_name
        == original.source.telescope_name
    )
    assert bundle.runtime.instrument.location == original.location
    assert bundle.runtime.instrument.default_diameter_m == original.default_diameter_m
    assert bundle.runtime.instrument.diameter_overrides == original.diameter_overrides


def test_instrument_source_path_override_rejects_known_telescope_before_loading(
    tmp_path,
):
    data = valid_config_mapping(tmp_path)
    data["instrument"]["source"] = {
        "kind": "known_telescope",
        "name": "HERA",
        "registry_policy": "offline",
    }

    with pytest.raises(
        ConfigOverrideError,
        match="overrides.instrument_source.path.*known-telescope",
    ):
        resolve_config(
            data,
            source=ConfigurationSource.for_mapping(
                base_dir=tmp_path,
                invocation_dir=tmp_path,
            ),
            overrides=SimulationOverrides(
                instrument_source=InstrumentSourcePathOverride(
                    path=tmp_path / "replacement.txt"
                )
            ),
        )


@pytest.mark.parametrize(
    ("execution_offline", "registry_policy", "valid"),
    [
        (True, "offline", True),
        (False, "offline", True),
        (False, "allow_network", True),
        (True, "allow_network", False),
    ],
)
def test_execution_and_known_source_network_policy_matrix_is_pure(
    tmp_path,
    execution_offline,
    registry_policy,
    valid,
):
    data = valid_config_mapping(
        tmp_path,
        execution={"backend": "numpy", "offline": execution_offline},
    )
    data["instrument"]["source"] = {
        "kind": "known_telescope",
        "name": "HERA",
        "registry_policy": registry_policy,
    }
    source = ConfigurationSource.for_mapping(
        base_dir=tmp_path,
        invocation_dir=tmp_path,
    )

    if not valid:
        with pytest.raises(
            ConfigSemanticError,
            match="registry_policy='allow_network' requires execution.offline=false",
        ):
            resolve_config(data, source=source)
        return

    bundle = resolve_config(data, source=source)
    assert bundle.runtime.execution.offline is execution_offline
    assert bundle.runtime.instrument.source.registry_policy == registry_policy


def test_public_path_override_names_the_instrument_source_not_legacy_layout():
    import radiosim.io.config_resolution as config_resolution
    from radiosim import io

    assert "instrument_source" in SimulationOverrides.model_fields
    assert "antenna_layout" not in SimulationOverrides.model_fields
    assert hasattr(config_resolution, "InstrumentSourcePathOverride")
    assert io.InstrumentSourcePathOverride is (
        config_resolution.InstrumentSourcePathOverride
    )
    assert "InstrumentSourcePathOverride" in config_resolution.__all__
    assert "AntennaLayoutOverride" not in config_resolution.__all__


def test_complete_override_matrix_replaces_only_named_logical_values(tmp_path):
    data = valid_config_mapping(
        tmp_path,
        execution={"backend": "jax", "offline": True},
    )
    original = RadioSimConfig.model_validate(data)
    overrides = SimulationOverrides(
        backend="auto",
        precision=PrecisionInput(preset="fast"),
        offline=False,
        obs_frequency=ExplicitFrequencyConfig(channel_frequencies_hz=[120e6, 123.5e6]),
        location=InstrumentLocationConfig(
            longitude_deg=2.0,
            latitude_deg=1.0,
            height_m=3.0,
        ),
        start_time="2025-02-03T04:05:06",
        simulator="rime",
    )

    bundle = resolve_config(
        original,
        source=ConfigurationSource.for_model(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
        overrides=overrides,
    )

    assert bundle.runtime.execution.backend_strategy == "auto"
    assert bundle.runtime.execution.offline is False
    assert bundle.runtime.execution.precision == PrecisionConfig.fast()
    assert bundle.runtime.frequency.channel_frequencies_hz == (120e6, 123.5e6)
    assert bundle.runtime.instrument.location.latitude_deg == 1.0
    assert bundle.runtime.instrument.location.longitude_deg == 2.0
    assert bundle.runtime.instrument.location.height_m == 3.0
    assert bundle.runtime.observation.start_time_iso.startswith("2025-02-03T04:05:06")
    assert bundle.runtime.observation.duration_seconds == 2.0
    assert bundle.provenance.override_origins["execution.backend"] == "override"
    assert bundle.provenance.override_origins["execution.offline"] == "override"
    assert bundle.provenance.override_origins["instrument.location"] == "override"
    assert original.execution.backend == "jax"
    assert original.execution.offline is True
    assert overrides.offline is False


def test_none_means_no_override_and_runtime_precision_is_accepted(tmp_path):
    data = valid_config_mapping(tmp_path, execution={"backend": "jax"})

    bundle = resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
        overrides=SimulationOverrides(
            backend=None,
            precision=PrecisionConfig.fast(),
        ),
    )

    assert bundle.runtime.execution.backend_strategy == "jax"
    assert bundle.runtime.execution.precision == PrecisionConfig.fast()


def test_schema_semantic_unsupported_and_path_errors_are_distinct(tmp_path):
    source = ConfigurationSource.for_mapping(
        base_dir=tmp_path,
        invocation_dir=tmp_path,
    )
    schema_data = valid_config_mapping(tmp_path)
    schema_data["instrument"]["location"]["latitude_deg"] = float("inf")
    with pytest.raises(ConfigSchemaError):
        resolve_config(schema_data, source=source)

    semantic_data = valid_config_mapping(
        tmp_path,
        obs_time={"duration_seconds": 1.0, "time_step_seconds": 2.0},
    )
    with pytest.raises(ConfigSemanticError):
        resolve_config(semantic_data, source=source)

    unsupported_data = valid_config_mapping(
        tmp_path,
        visibility={"calculation_type": "spherical_harmonic"},
    )
    with pytest.raises(UnsupportedConfigError) as exc_info:
        resolve_config(unsupported_data, source=source)
    assert "spherical_harmonic_unsupported" in {
        issue.code for issue in exc_info.value.issues
    }

    path_data = valid_config_mapping(tmp_path)
    path_data["instrument"]["source"]["path"] = "missing.txt"
    with pytest.raises(ConfigPathError):
        resolve_config(path_data, source=source)


def test_override_models_are_strict_frozen_and_do_not_partially_merge():
    with pytest.raises(ValidationError):
        SimulationOverrides.model_validate({"offline": False, "unknown": True})
    with pytest.raises(ValidationError, match="frozen"):
        override = SimulationOverrides(offline=False)
        override.offline = True


def test_resolution_has_no_runtime_or_external_side_effects(tmp_path, monkeypatch):
    data = valid_config_mapping(tmp_path)
    source = ConfigurationSource.for_mapping(
        base_dir=tmp_path,
        invocation_dir=tmp_path,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("configuration resolution crossed its side-effect boundary")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(SkyLoaderRegistry, "loader", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(webbrowser, "open", forbidden)
    monkeypatch.setattr(
        "radiosim.utils.device.get_device_resources",
        forbidden,
    )
    with monkeypatch.context() as content_guard:
        content_guard.setattr(builtins, "open", forbidden)
        content_guard.setattr(Path, "open", forbidden)
        content_guard.setattr(Path, "read_bytes", forbidden)
        content_guard.setattr(Path, "read_text", forbidden)
        bundle = resolve_config(data, source=source)

    assert bundle.runtime.execution.backend_strategy == "numpy"
    assert not bundle.workflow.output_dir.exists()


@pytest.mark.parametrize(
    "beams",
    [
        {"mode": "analytic"},
        {"mode": "shared_fits", "beam": {"kind": "fits", "path": "beam.fits"}},
        {
            "mode": "per_antenna_fits",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": {"kind": "fits", "path": "beam.fits"},
                }
            ],
        },
        {
            "mode": "mixed",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": {"kind": "analytic"},
                }
            ],
        },
    ],
)
def test_resolution_accepts_every_final_beam_mode_without_loading_uvbeam(
    tmp_path, monkeypatch, beams
):
    (tmp_path / "beam.fits").touch()
    data = valid_config_mapping(tmp_path, beams=beams)

    def forbidden(*args, **kwargs):
        pytest.fail("configuration resolution loaded BeamFITS content")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    bundle = resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )

    assert bundle.runtime.beams.mode == beams["mode"]
