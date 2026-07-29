"""Tests for the Tier 1B strict immutable input configuration contract."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from radiosim.io.config import (
    CliWorkflowConfig,
    CustomRegisteredSourceConfig,
    ExecutionConfig,
    ExplicitFrequencyConfig,
    FrozenDict,
    PrecisionInput,
    RadioSimConfig,
    StrictFrozenModel,
    TestSourcesConfig,
    VisibilityConfig,
    collect_config_issues,
    collect_schema_issues,
    collect_semantic_issues,
    collect_unsupported_issues,
    dump_config,
    load_config,
    parse_sky_source_config,
)
from radiosim.io.config_resolution import ConfigurationSource, resolve_config
from tests.fixtures.configs import (
    resolved_config,
    valid_config_mapping,
    valid_input_config,
    write_config_yaml,
)


def _all_strict_model_types() -> set[type[StrictFrozenModel]]:
    found: set[type[StrictFrozenModel]] = set()
    pending = list(StrictFrozenModel.__subclasses__())
    while pending:
        model = pending.pop()
        if model in found:
            continue
        found.add(model)
        pending.extend(model.__subclasses__())
    return found


def test_load_config_requires_complete_tagged_target_shape(tmp_path):
    data = valid_config_mapping(
        tmp_path,
        sky_model={"sources": [{"kind": "gleam", "flux_limit": 1.5}]},
    )

    bundle = load_config(write_config_yaml(tmp_path, data))

    source = bundle.runtime.sky_model.sources[0]
    assert source.kind == "gleam"
    assert source.options["catalog"] == "gleam_egc"


def test_parse_sky_source_config_does_not_mutate_test_source_defaults():
    source = parse_sky_source_config(
        {"kind": "test_sources", "representation": "healpix_map"}
    )

    assert isinstance(source, TestSourcesConfig)
    assert source.representation == "healpix_map"
    assert source.nside is None
    kind, kwargs = source.to_loader_request()
    assert kind == "test_sources"
    assert "nside" not in kwargs


def test_registered_alias_uses_explicit_strict_options_envelope():
    source = parse_sky_source_config({"kind": "gsm2016", "options": {"nside": 128}})

    assert isinstance(source, CustomRegisteredSourceConfig)
    kind, kwargs = source.to_loader_request()
    assert kind == "diffuse_sky"
    assert kwargs["model"] == "gsm2016"
    assert kwargs["nside"] == 128


def test_registered_catalog_options_are_typed_and_frozen():
    source = parse_sky_source_config(
        {"kind": "nvss", "options": {"max_rows": 5, "flux_limit": 0.5}}
    )

    assert isinstance(source.options, FrozenDict)
    kind, kwargs = source.to_loader_request()
    assert kind == "nvss"
    assert kwargs["max_rows"] == 5
    assert kwargs["flux_limit"] == 0.5
    with pytest.raises(TypeError, match="immutable"):
        source.options["max_rows"] = 6


@pytest.mark.parametrize(
    "source",
    [
        {"kind": "nvss", "max_rows": 5},
        {"kind": "nvss", "options": {"unknown": True}},
        {"kind": "nvss", "options": {"max_rows": "many"}},
    ],
)
def test_registered_sources_reject_direct_unknown_or_wrong_typed_options(source):
    with pytest.raises(ValidationError):
        parse_sky_source_config(source)


def test_registered_unknown_option_reports_full_indexed_path(tmp_path):
    data = valid_config_mapping(
        tmp_path,
        sky_sources=[{"kind": "nvss", "options": {"unknown": True}}],
    )

    issues = collect_schema_issues(data)

    assert [issue.path for issue in issues] == ["sky_model.sources[0].options.unknown"]


def test_racs_does_not_expose_silent_allow_full_catalog_field():
    with pytest.raises(ValidationError, match="allow_full_catalog"):
        parse_sky_source_config({"kind": "racs", "allow_full_catalog": True})


def test_source_region_and_brightness_override_global_context():
    source = parse_sky_source_config(
        {
            "kind": "gleam",
            "brightness_conversion": "rayleigh-jeans",
            "region": {
                "shape": "cone",
                "center_ra_deg": 180.0,
                "center_dec_deg": 0.0,
                "radius_deg": 5.0,
            },
        }
    )

    kind, kwargs = source.to_loader_request(
        region="global_region", brightness_conversion="planck"
    )

    assert kind == "gleam"
    assert kwargs["brightness_conversion"] == "rayleigh-jeans"
    assert kwargs["region"] != "global_region"


def test_visibility_and_top_level_defaults_match_target(tmp_path):
    data = valid_config_mapping(tmp_path)
    for section in (
        "beams",
        "baseline_selection",
        "visibility",
        "execution",
        "workflow",
    ):
        data.pop(section, None)

    config = RadioSimConfig.model_validate(data)

    assert VisibilityConfig().sky_representation == "point_sources"
    assert config.beams.mode == "analytic"
    assert config.beams.model.kind == "circular_aperture"
    assert config.beams.model.taper.kind == "gaussian"
    assert config.beams.model.taper.edge_taper_db == 10.0
    assert config.baseline_selection.correlations == "all"
    assert config.execution.backend == "numpy"
    assert config.execution.precision.preset == "standard"
    assert config.execution.simulator == "rime"
    assert config.workflow == CliWorkflowConfig()
    assert collect_config_issues(config) == ()


def test_valid_input_builder_uses_final_top_level_shape(tmp_path):
    config = valid_input_config(tmp_path)

    assert tuple(config.model_fields) == (
        "instrument",
        "beams",
        "baseline_selection",
        "sky_model",
        "obs_time",
        "obs_frequency",
        "visibility",
        "execution",
        "workflow",
    )
    assert config.execution.offline is True
    assert config.workflow.save_results is False
    assert collect_config_issues(config) == ()


@pytest.mark.parametrize(
    "missing",
    ["instrument", "sky_model", "obs_time", "obs_frequency"],
)
def test_required_scientific_sections_fail_when_absent(tmp_path, missing):
    data = valid_config_mapping(tmp_path)
    data.pop(missing)

    with pytest.raises(ValidationError) as exc_info:
        RadioSimConfig.model_validate(data)

    assert any(error["loc"] == (missing,) for error in exc_info.value.errors())


def test_every_concrete_input_model_is_strict_and_frozen():
    models = _all_strict_model_types()
    assert RadioSimConfig in models
    assert CustomRegisteredSourceConfig in models
    assert PrecisionInput in models
    assert len(models) >= 35

    for model in models:
        assert model.model_config["extra"] == "forbid", model.__name__
        assert model.model_config["frozen"] is True, model.__name__
        with pytest.raises(ValidationError) as exc_info:
            model.model_validate({"tier1b_unknown": True})
        assert any(
            error["type"] == "extra_forbidden" for error in exc_info.value.errors()
        ), model.__name__


def test_unknown_top_level_field_is_rejected(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["tier1b_unknown"] = True

    with pytest.raises(ValidationError, match="tier1b_unknown"):
        RadioSimConfig.model_validate(data)


@pytest.mark.parametrize(
    "section",
    [
        "instrument",
        "beams",
        "obs_time",
        "workflow",
        "execution",
    ],
)
def test_unknown_nested_field_is_rejected_for_representative_sections(
    tmp_path, section
):
    data = valid_config_mapping(tmp_path)
    data[section]["tier1b_unknown"] = True

    with pytest.raises(ValidationError, match="tier1b_unknown"):
        RadioSimConfig.model_validate(data)


def test_unknown_builtin_source_and_precision_leaf_are_rejected(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["sky_model"]["sources"][0]["typo"] = True
    data["execution"]["precision"] = {
        "coordinates": {"antenna_positions": "float64", "typo": "float32"}
    }

    issues = collect_schema_issues(data)

    assert {issue.path for issue in issues} == {
        "execution.precision.coordinates.typo",
        "sky_model.sources[0].test_sources.typo",
    }


@pytest.mark.parametrize(
    ("path", "value", "hint_fragment"),
    [
        ("telescope", {"telescope_name": "Old"}, "instrument.source"),
        ("antenna_layout", {}, "instrument.source"),
        ("location", {}, "instrument.location"),
        ("feeds", {}, "Tier 5"),
        ("compute", {"backend": "numpy"}, "execution"),
        ("precision", {"preset": "fast"}, "execution.precision"),
        ("simulators", {"name": "rime"}, "execution.simulator"),
        ("output", {}, "workflow"),
    ],
)
def test_removed_top_level_sections_have_migration_hints(
    tmp_path, path, value, hint_fragment
):
    data = valid_config_mapping(tmp_path)
    data[path] = value

    issue = next(item for item in collect_schema_issues(data) if item.path == path)

    assert issue.code == "removed_field"
    assert hint_fragment in issue.hint


@pytest.mark.parametrize(
    ("section", "field", "hint_fragment"),
    [
        ("instrument.location", "ra", "Phase-center"),
        ("instrument.location", "dec", "Phase-center"),
        ("beams", "use_beam_file", "shared_fits"),
        ("beams", "all_beam_response", "beams.mode"),
    ],
)
def test_removed_nested_fields_have_actionable_messages(
    tmp_path, section, field, hint_fragment
):
    data = valid_config_mapping(tmp_path)
    target = data
    for part in section.split("."):
        target = target[part]
    target[field] = 1.0

    issue = next(
        item
        for item in collect_schema_issues(data)
        if item.path == f"{section}.{field}"
    )

    assert issue.code == "removed_field"
    assert hint_fragment in issue.hint


@pytest.mark.parametrize("legacy_mode", ["analytic", "fits", "mixed", "shared"])
def test_removed_beam_mode_field_has_actionable_message(tmp_path, legacy_mode):
    data = valid_config_mapping(tmp_path)
    data["beams"]["beam_mode"] = legacy_mode

    issue = next(
        item for item in collect_schema_issues(data) if item.path == "beams.beam_mode"
    )

    assert issue.code == "removed_field"
    assert issue.message.startswith("removed in Tier 3;")
    assert "beams.mode" in issue.hint


def test_close_unknown_field_gets_single_did_you_mean_hint(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["workflow"]["result_filenam"] = "visibilities"

    issue = next(
        item
        for item in collect_schema_issues(data)
        if item.path == "workflow.result_filenam"
    )

    assert issue.hint == "Did you mean 'result_filename'?"


def test_legacy_nested_sky_sections_keep_direct_migration_message(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["sky_model"]["gleam"] = {"use_gleam": True}

    issues = collect_schema_issues(data)

    assert any(
        "Rewrite each enabled section as an entry under sky_model.sources"
        in issue.message
        for issue in issues
    )


def test_schema_reports_multiple_errors_without_partial_model(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["instrument"]["default_diameter_m"] = -1.0
    data["instrument"]["location"]["latitude_deg"] = float("inf")
    data["workflow"]["plotting_backend"] = "browser"

    issues = collect_schema_issues(data)

    assert len(issues) == 3
    assert [issue.path for issue in issues] == sorted(issue.path for issue in issues)


def test_deep_immutability_rejects_nested_assignment_and_mapping_mutation(tmp_path):
    config = valid_input_config(
        tmp_path,
        instrument={
            "diameter_overrides": [
                {"antenna": {"kind": "number", "number": 0}, "diameter_m": 14.0}
            ]
        },
        beams={
            "mode": "mixed",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": {"kind": "analytic"},
                }
            ],
        },
    )

    with pytest.raises(ValidationError, match="frozen"):
        config.execution.backend = "jax"
    with pytest.raises(ValidationError, match="frozen"):
        config.execution.precision.coordinates.uvw = "float32"
    with pytest.raises(TypeError):
        config.instrument.diameter_overrides[0] = None
    with pytest.raises(TypeError):
        config.beams.assignments[0] = None
    with pytest.raises(ValidationError, match="frozen"):
        config.beams.assignments[0].antenna.number = 1


def test_beam_input_has_no_mutable_mapping_surface(tmp_path):
    config = valid_input_config(
        tmp_path,
        beams={
            "mode": "mixed",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": {"kind": "analytic"},
                }
            ],
        },
    )

    assert type(config.beams.assignments) is tuple
    assert not isinstance(config.beams.assignments, Mapping)
    assert not isinstance(config.beams.assignments[0], Mapping)


def test_caller_owned_containers_are_copied(tmp_path):
    data = valid_config_mapping(tmp_path)
    diameters = [{"antenna": {"kind": "number", "number": 0}, "diameter_m": 14.0}]
    options = {"max_rows": 5}
    source_entries = [{"kind": "nvss", "options": options}]
    precision = {"coordinates": {"uvw": "float32"}}
    beam_assignments = [
        {
            "antenna": {"kind": "number", "number": 0},
            "beam": {"kind": "analytic"},
        }
    ]
    data["instrument"]["diameter_overrides"] = diameters
    data["sky_model"]["sources"] = source_entries
    data["execution"]["precision"] = precision
    data["beams"] = {"mode": "mixed", "assignments": beam_assignments}

    config = RadioSimConfig.model_validate(data)
    source_entries.append({"kind": "test_sources"})
    diameters[0]["diameter_m"] = 99.0
    options["max_rows"] = 6
    precision["coordinates"]["uvw"] = "float64"
    beam_assignments[0]["antenna"]["number"] = 99

    assert len(config.sky_model.sources) == 1
    assert config.instrument.diameter_overrides[0].diameter_m == 14.0
    assert config.sky_model.sources[0].options["max_rows"] == 5
    assert config.execution.precision.coordinates.uvw == "float32"
    assert config.beams.assignments[0].antenna.number == 0


def test_json_and_yaml_serialization_use_ordinary_lists_and_mappings(tmp_path):
    config = valid_input_config(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [100e6, 101.5e6, 108e6],
            "channel_widths_hz": [1e6, 2e6, 3e6],
        },
    )
    output = tmp_path / "roundtrip.yaml"

    dumped = config.model_dump(mode="json")
    dump_config(config, output)
    yaml_data = yaml.safe_load(output.read_text())

    assert dumped["obs_frequency"]["channel_frequencies_hz"] == [
        100e6,
        101.5e6,
        108e6,
    ]
    assert yaml_data["obs_frequency"]["channel_frequencies_hz"] == [
        100e6,
        101.5e6,
        108e6,
    ]
    assert (
        RadioSimConfig.model_validate(yaml_data).obs_frequency == config.obs_frequency
    )


def test_dump_config_rejects_root_model_subclasses_with_extra_fields(tmp_path):
    class ExtendedRadioSimConfig(RadioSimConfig):
        surprise: str = "must not be serialized"

    config = ExtendedRadioSimConfig.model_validate(valid_config_mapping(tmp_path))

    with pytest.raises(TypeError, match="only RadioSimConfig"):
        dump_config(config, tmp_path / "subclass.yaml")


def test_standard_model_dump_mapping_round_trips_preset_precision(tmp_path):
    config = valid_input_config(tmp_path)
    dumped = config.model_dump(mode="json")

    assert dumped["execution"]["precision"] == {"preset": "standard"}
    reparsed = RadioSimConfig.model_validate(dumped)
    bundle = resolve_config(
        dumped,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )

    assert reparsed.execution.precision.preset == "standard"
    assert bundle.runtime.execution.precision.default == "float64"


@pytest.mark.parametrize(
    ("frequency", "expected"),
    [
        (
            {
                "mode": "grid",
                "starting_frequency": 100.0,
                "frequency_interval": 1.0,
                "frequency_bandwidth": 2.0,
                "channel_width": 1.0,
                "frequency_unit": "MHz",
            },
            (100e6, 101e6, 102e6),
        ),
        (
            {
                "mode": "explicit",
                "channel_frequencies_hz": [100e6],
                "channel_widths_hz": [1e6],
            },
            (100e6,),
        ),
        (
            {
                "mode": "explicit",
                "channel_frequencies_hz": [100e6, 101.25e6, 109e6],
                "channel_widths_hz": [1e6, 1e6, 1e6],
            },
            (100e6, 101.25e6, 109e6),
        ),
    ],
)
def test_dump_load_round_trip_preserves_frequency_meaning(
    tmp_path,
    frequency,
    expected,
):
    config = valid_input_config(tmp_path, frequency=frequency)
    output = tmp_path / "round-trip.yaml"

    dump_config(config, output)
    bundle = load_config(output)

    assert bundle.runtime.frequency.channel_frequencies_hz == expected


def test_dump_config_preserves_paths_maps_order_precision_and_workflow(tmp_path):
    data = valid_config_mapping(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [100e6, 101.25e6, 109e6],
            "channel_widths_hz": [1e6, 1e6, 1e6],
        },
        sky_sources=[
            {"kind": "test_sources", "num_sources": 1},
            {"kind": "nvss", "options": {"max_rows": 5, "flux_limit": 0.5}},
        ],
        workflow={
            "output_dir": str(tmp_path / "custom-output"),
            "collision_policy": "replace",
        },
    )
    data["execution"]["precision"] = {
        "default": "float32",
        "coordinates": {"uvw": "float32"},
        "accumulation": "float64",
        "output": "float32",
    }
    config = RadioSimConfig.model_validate(data)
    before = config.model_dump(mode="json")
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"

    dump_config(config, first)
    dump_config(config, second)

    assert first.read_text() == second.read_text()
    document = yaml.safe_load(first.read_text())
    assert document["instrument"]["source"]["path"] == str(
        config.instrument.source.path
    )
    assert document["obs_frequency"]["channel_frequencies_hz"] == [
        100e6,
        101.25e6,
        109e6,
    ]
    assert [source["kind"] for source in document["sky_model"]["sources"]] == [
        "test_sources",
        "nvss",
    ]
    assert document["sky_model"]["sources"][1]["options"] == {
        "max_rows": 5,
        "flux_limit": 0.5,
    }
    assert document["execution"]["precision"]["coordinates"]["uvw"] == "float32"
    assert document["workflow"]["collision_policy"] == "replace"
    assert "runtime" not in document
    assert "provenance" not in document
    assert config.model_dump(mode="json") == before

    bundle = load_config(first)
    assert [source.kind for source in bundle.runtime.sky_model.sources] == [
        "test_sources",
        "nvss",
    ]
    assert bundle.runtime.execution.precision.coordinates.uvw == "float32"
    assert bundle.workflow.output_dir == tmp_path / "custom-output"


def test_dump_config_serializes_default_workflow_fields(tmp_path):
    data = valid_config_mapping(tmp_path)
    data.pop("workflow")
    config = RadioSimConfig.model_validate(data)
    output = tmp_path / "defaults.yaml"

    dump_config(config, output)

    workflow = yaml.safe_load(output.read_text())["workflow"]
    assert workflow["output_dir"] == "output"
    assert workflow["result_filename"] == "visibilities"
    assert workflow["save_results"] is False
    bundle = load_config(output)
    assert bundle.workflow.output_dir == tmp_path / "output"
    assert bundle.workflow.result_filename == "visibilities"
    assert bundle.workflow.save_results is False


def test_dump_config_preserves_defaulted_union_discriminators(tmp_path):
    config = valid_input_config(tmp_path)
    config = config.model_copy(
        update={
            "sky_model": config.sky_model.model_copy(
                update={"sources": (TestSourcesConfig(),)}
            ),
            "obs_frequency": ExplicitFrequencyConfig(
                channel_frequencies_hz=(100e6, 101.25e6, 109e6),
                channel_widths_hz=(1e6, 1e6, 1e6),
            ),
        }
    )
    output = tmp_path / "programmatic-defaults.yaml"

    dump_config(config, output)

    document = yaml.safe_load(output.read_text())
    assert document["sky_model"]["sources"][0]["kind"] == "test_sources"
    assert document["obs_frequency"]["mode"] == "explicit"
    bundle = load_config(output)
    assert bundle.runtime.frequency.channel_frequencies_hz == (
        100e6,
        101.25e6,
        109e6,
    )
    assert bundle.runtime.sky_model.sources[0].kind == "test_sources"


def test_dump_config_requires_input_model_and_existing_parent(tmp_path):
    config = valid_input_config(tmp_path)

    with pytest.raises(TypeError, match="only RadioSimConfig"):
        dump_config(resolved_config(tmp_path), tmp_path / "resolved.yaml")
    with pytest.raises(TypeError, match="only RadioSimConfig"):
        dump_config(config.model_dump(), tmp_path / "mapping.yaml")
    with pytest.raises(FileNotFoundError, match="parent does not exist"):
        dump_config(config, tmp_path / "missing" / "config.yaml")


def test_dump_config_resolves_relative_destination_from_invocation_cwd(
    tmp_path,
    monkeypatch,
):
    config = valid_input_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    dump_config(config, "relative.yaml")

    assert (tmp_path / "relative.yaml").is_file()


def test_legacy_config_methods_and_resolver_alias_are_not_defined():
    import radiosim.io as public_io
    import radiosim.io.config_resolution as resolution

    for name in ("from_yaml", "to_yaml", "to_dict", "validate"):
        assert name not in RadioSimConfig.__dict__
    assert not hasattr(resolution, "resolve_configuration")
    assert not hasattr(public_io, "resolve_configuration")
    assert not hasattr(public_io, "save_config_yaml")
    for name in (
        "load_config",
        "resolve_config",
        "dump_config",
        "ConfigParseError",
        "ResolvedConfiguration",
    ):
        assert name in public_io.__all__


def test_workflow_field_constraints_have_no_side_effects(tmp_path):
    output = tmp_path / "must-not-exist"
    data = valid_config_mapping(tmp_path, workflow={"output_dir": str(output)})

    config = RadioSimConfig.model_validate(data)

    assert config.workflow.output_dir == output
    assert not output.exists()
    for patch in (
        {"run_subdir": "../escape"},
        {"result_filename": "result.h5"},
        {"result_format": "HDF5"},
        {"plotting_backend": "plotly"},
        {"visibility_phase_unit": "gradians"},
        {"visibility_phase_unit": ""},
        {"visibility_phase_unit": None},
    ):
        with pytest.raises(ValidationError):
            CliWorkflowConfig.model_validate(patch)


def test_tier4g_visibility_phase_unit_replaces_the_removed_plot_fields():
    workflow = CliWorkflowConfig()

    assert workflow.visibility_phase_unit == "radians"
    assert CliWorkflowConfig(visibility_phase_unit="degrees").visibility_phase_unit == (
        "degrees"
    )
    assert "angle_unit" not in CliWorkflowConfig.model_fields
    assert "sky_model_frequency_hz" not in CliWorkflowConfig.model_fields
    for field_name, expected in (
        (
            "angle_unit",
            "workflow.angle_unit: removed before v1.0; "
            "use workflow.visibility_phase_unit",
        ),
        (
            "sky_model_frequency_hz",
            "workflow.sky_model_frequency_hz: removed before v1.0; "
            "no Tier 4 sky renderer consumes it",
        ),
    ):
        with pytest.raises(ValidationError) as error:
            CliWorkflowConfig.model_validate({field_name: "degrees"})
        assert expected in str(error.value)


def test_tier4g_removed_plot_fields_are_schema_errors_in_a_full_document(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["workflow"]["angle_unit"] = "degrees"

    issues = collect_schema_issues(data)

    assert any(
        "use workflow.visibility_phase_unit" in issue.message for issue in issues
    )
    with pytest.raises(ValidationError):
        RadioSimConfig.model_validate(data)


def test_semantic_collector_aggregates_stably_without_mutating_config(tmp_path):
    config = valid_input_config(
        tmp_path,
        obs_time={"duration_seconds": 1.0, "time_step_seconds": 2.0},
        execution={
            "backend": "jax",
            "precision": {"preset": "ultra", "output": "float128"},
        },
        workflow={"collision_policy": "error"},
    )
    before = config.model_dump(mode="json")

    first = collect_semantic_issues(config)
    second = collect_semantic_issues(config)

    assert first == second
    assert tuple((item.path, item.code) for item in first) == tuple(
        sorted((item.path, item.code) for item in first)
    )
    assert {
        "execution.precision",
        "execution.precision.preset.ultra",
        "obs_time.time_step_seconds",
    } <= {item.path for item in first}
    assert config.model_dump(mode="json") == before


def test_unsupported_collector_accepts_final_beam_modes_for_runtime(tmp_path):
    config = valid_input_config(
        tmp_path,
        beams={
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": "missing.fits"},
        },
        visibility={"calculation_type": "spherical_harmonic"},
        workflow={
            "result_format": "uvfits",
            "collision_policy": "suffix",
            "visibility_phase_unit": "degrees",
        },
    )

    issues = collect_unsupported_issues(config)
    paths = {issue.path for issue in issues}

    assert {"visibility.calculation_type"} <= paths
    assert not any(path.startswith("beams.") for path in paths)
    assert not any(path.startswith("workflow.") for path in paths)
    assert all(issue.stage == "unsupported" for issue in issues)


def test_precision_default_and_declared_backend_values_are_input_only():
    assert ExecutionConfig().backend == "numpy"
    assert ExecutionConfig(backend="auto").backend == "auto"
    assert ExecutionConfig(backend="jax").backend == "jax"
    assert ExecutionConfig(backend="numba").backend == "numba"
    with pytest.raises(ValidationError):
        ExecutionConfig(backend=None)


def test_explicit_numpy_container_is_owned_by_model():
    values = np.array([100e6, 101.5e6, 108e6])
    widths = np.array([1e6, 2e6, 3e6])
    frequency = ExplicitFrequencyConfig(
        channel_frequencies_hz=values,
        channel_widths_hz=widths,
    )
    values[1] = 999e6
    widths[1] = 999e6

    assert frequency.channel_frequencies_hz == (100e6, 101.5e6, 108e6)
    assert frequency.channel_widths_hz == (1e6, 2e6, 3e6)
