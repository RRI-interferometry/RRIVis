"""Characterization tests for current configuration path handling."""

from __future__ import annotations

from pathlib import Path

import pytest

import radiosim.core.sky.registry.core as registry_core
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.registry import loader_registry
from radiosim.io.config import RadioSimConfig, load_config
from radiosim.io.config_resolution import (
    ConfigPathError,
    ConfigSemanticError,
    ConfigSourceError,
    ConfigurationSource,
    resolve_config,
)
from tests.fixtures.configs import valid_config_mapping, write_config_yaml


def test_load_config_resolves_antenna_path_from_yaml_parent(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["antenna_layout"]["antenna_positions_file"] = "antennas.txt"
    config_path = write_config_yaml(tmp_path, data)

    bundle = load_config(config_path)

    assert (
        bundle.runtime.antenna_layout.antenna_positions_file
        == (tmp_path / "antennas.txt").resolve()
    )
    assert bundle.runtime.antenna_layout.antenna_positions_file.is_absolute()


def test_mapping_and_model_keep_relative_antenna_path_current_behavior(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["antenna_layout"]["antenna_positions_file"] = "antennas.txt"

    model = RadioSimConfig.model_validate(data)

    assert data["antenna_layout"]["antenna_positions_file"] == "antennas.txt"
    assert model.antenna_layout.antenna_positions_file == Path("antennas.txt")
    assert not Path(model.antenna_layout.antenna_positions_file).is_absolute()


def test_load_config_resolves_non_antenna_paths_from_yaml_parent(tmp_path):
    (tmp_path / "sky.skyh5").touch()

    data = valid_config_mapping(tmp_path)
    data["sky_model"]["sources"] = [
        {"kind": "pyradiosky_file", "filename": "sky.skyh5"}
    ]
    data["workflow"]["output_dir"] = "relative-output"
    config_path = write_config_yaml(tmp_path, data)

    bundle = load_config(config_path)

    assert (
        bundle.runtime.sky_model.sources[0].options["filename"]
        == (tmp_path / "sky.skyh5").resolve()
    )
    assert bundle.workflow.output_dir == (tmp_path / "relative-output").resolve()


def test_load_config_resolves_relative_config_path_from_captured_cwd(
    tmp_path,
    monkeypatch,
):
    data = valid_config_mapping(tmp_path)
    data["antenna_layout"]["antenna_positions_file"] = "antennas.txt"
    data["workflow"]["output_dir"] = "relative-output"
    config_path = write_config_yaml(tmp_path, data)
    monkeypatch.chdir(tmp_path)

    bundle = load_config(config_path.name)

    assert bundle.provenance.source.config_path == config_path.resolve()
    assert bundle.provenance.source.invocation_dir == tmp_path.resolve()
    assert (
        bundle.runtime.antenna_layout.antenna_positions_file
        == (tmp_path / "antennas.txt").resolve()
    )
    assert bundle.workflow.output_dir == (tmp_path / "relative-output").resolve()


def test_load_config_rejects_environment_syntax_and_non_file_source(tmp_path):
    with pytest.raises(ConfigSourceError) as environment_error:
        load_config("$CONFIG_ROOT/config.yaml")
    with pytest.raises(ConfigSourceError) as type_error:
        load_config(tmp_path)

    assert environment_error.value.issues[0].code == "environment_path_syntax"
    assert type_error.value.issues[0].code == "invalid_yaml_config_path"


def test_check_input_paths_false_skips_only_existence_and_type_checks(tmp_path):
    data = valid_config_mapping(
        tmp_path,
        antenna_layout={"antenna_positions_file": "missing-antennas.txt"},
        sky_sources=[{"kind": "pyradiosky_file", "filename": "missing-sky.skyh5"}],
        workflow={"output_dir": "missing-output"},
    )
    config_path = write_config_yaml(tmp_path, data)

    bundle = load_config(config_path, check_input_paths=False)

    assert (
        bundle.runtime.antenna_layout.antenna_positions_file
        == (tmp_path / "missing-antennas.txt").resolve()
    )
    assert (
        bundle.runtime.sky_model.sources[0].options["filename"]
        == (tmp_path / "missing-sky.skyh5").resolve()
    )
    assert bundle.workflow.output_dir == (tmp_path / "missing-output").resolve()
    assert not bundle.workflow.output_dir.exists()
    with pytest.raises(ConfigPathError):
        load_config(config_path, check_input_paths=True)

    bundle.workflow.output_dir.write_text("not a directory")
    with pytest.raises(ConfigPathError) as output_error:
        load_config(config_path, check_input_paths=False)
    assert [issue.code for issue in output_error.value.issues] == [
        "output_path_wrong_type"
    ]

    environment_data = valid_config_mapping(
        tmp_path,
        antenna_layout={"antenna_positions_file": "$DATA/antennas.txt"},
    )
    environment_path = write_config_yaml(
        tmp_path,
        environment_data,
        name="environment.yaml",
    )
    with pytest.raises(ConfigPathError, match="environment-variable syntax"):
        load_config(environment_path, check_input_paths=False)

    semantic_data = valid_config_mapping(
        tmp_path,
        obs_time={"duration_seconds": 1.0, "time_step_seconds": 2.0},
    )
    semantic_path = write_config_yaml(
        tmp_path,
        semantic_data,
        name="semantic.yaml",
    )
    with pytest.raises(ConfigSemanticError):
        load_config(semantic_path, check_input_paths=False)


def test_source_aware_relative_antenna_path_parity(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["antenna_layout"]["antenna_positions_file"] = "antennas.txt"
    config_path = write_config_yaml(tmp_path, data)

    mapping_model = RadioSimConfig.model_validate(data)
    yaml_bundle = resolve_config(
        data,
        source=ConfigurationSource.for_yaml(
            config_path,
            invocation_dir=tmp_path,
        ),
    )
    mapping_bundle = resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )
    model_bundle = resolve_config(
        mapping_model,
        source=ConfigurationSource.for_model(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )

    expected = (tmp_path / "antennas.txt").resolve()
    assert (
        yaml_bundle.runtime.antenna_layout.antenna_positions_file
        == mapping_bundle.runtime.antenna_layout.antenna_positions_file
        == model_bundle.runtime.antenna_layout.antenna_positions_file
        == expected
    )


def test_yaml_resolves_every_input_and_workflow_path_from_yaml_parent(tmp_path):
    for filename in ("sky.skyh5",):
        (tmp_path / filename).touch()

    data = valid_config_mapping(tmp_path)
    data["antenna_layout"]["antenna_positions_file"] = "antennas.txt"
    data["sky_model"]["sources"] = [
        {"kind": "pyradiosky_file", "filename": "sky.skyh5"}
    ]
    data["workflow"]["output_dir"] = "relative-output"
    config_path = write_config_yaml(tmp_path, data)
    bundle = resolve_config(
        data,
        source=ConfigurationSource.for_yaml(
            config_path,
            invocation_dir=tmp_path,
        ),
    )

    assert (
        bundle.runtime.antenna_layout.antenna_positions_file
        == (tmp_path / "antennas.txt").resolve()
    )
    assert (
        bundle.runtime.sky_model.sources[0].options["filename"]
        == (tmp_path / "sky.skyh5").resolve()
    )
    assert bundle.workflow.output_dir == (tmp_path / "relative-output").resolve()
    assert not bundle.workflow.output_dir.exists()
    assert {
        "configuration_source.config_path",
        "antenna_layout.antenna_positions_file",
        "sky_model.sources[0].filename",
        "workflow.output_dir",
    } <= set(bundle.provenance.path_resolutions)
    assert (
        bundle.provenance.path_resolutions["sky_model.sources[0].filename"].base
        == tmp_path.resolve()
    )


def test_parameter_paths_use_captured_invocation_directory(tmp_path, monkeypatch):
    invocation_dir = tmp_path / "invocation"
    later_dir = tmp_path / "later"
    invocation_dir.mkdir()
    later_dir.mkdir()
    data = valid_config_mapping(invocation_dir)
    data["antenna_layout"]["antenna_positions_file"] = "antennas.txt"
    data["workflow"]["output_dir"] = "relative-output"
    source = ConfigurationSource.for_parameters(invocation_dir=invocation_dir)
    monkeypatch.chdir(later_dir)

    bundle = resolve_config(data, source=source)

    assert (
        bundle.runtime.antenna_layout.antenna_positions_file
        == (invocation_dir / "antennas.txt").resolve()
    )
    assert bundle.workflow.output_dir == invocation_dir / "relative-output"


@pytest.mark.parametrize("kind", ["bbs", "fits_image", "pyradiosky_file"])
def test_builtin_scalar_file_sources_use_registry_path_metadata(tmp_path, kind):
    input_file = tmp_path / f"{kind}.dat"
    input_file.touch()
    data = valid_config_mapping(
        tmp_path,
        sky_sources=[{"kind": kind, "filename": input_file.name}],
    )

    bundle = resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )

    assert bundle.runtime.sky_model.sources[0].options["filename"] == input_file


def test_measurement_set_requires_directory_and_other_formats_require_file(tmp_path):
    ms_dir = tmp_path / "layout.ms"
    ms_dir.mkdir()
    source = ConfigurationSource.for_mapping(
        base_dir=tmp_path,
        invocation_dir=tmp_path,
    )
    data = valid_config_mapping(
        tmp_path,
        antenna_layout={
            "antenna_positions_file": "layout.ms",
            "antenna_file_format": "measurement_set",
        },
    )
    bundle = resolve_config(data, source=source)
    assert bundle.runtime.antenna_layout.antenna_positions_file == ms_dir.resolve()

    data["antenna_layout"]["antenna_file_format"] = "radiosim"
    with pytest.raises(ConfigPathError, match="expected a regular file"):
        resolve_config(data, source=source)


def test_path_expands_home_rejects_environment_syntax_and_tracks_symlink(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    home.mkdir()
    target = home / "target.txt"
    target.write_text("layout")
    symlink = home / "link.txt"
    symlink.symlink_to(target)
    source = ConfigurationSource.for_mapping(
        base_dir=tmp_path,
        invocation_dir=tmp_path,
    )
    monkeypatch.setenv("HOME", str(home))
    data = valid_config_mapping(
        tmp_path,
        antenna_layout={"antenna_positions_file": "~/link.txt"},
    )

    bundle = resolve_config(data, source=source)

    record = bundle.provenance.path_resolutions["antenna_layout.antenna_positions_file"]
    assert record.original == "~/link.txt"
    assert record.user_path == symlink.absolute()
    assert record.resolved == target.resolve()
    assert bundle.runtime.antenna_layout.antenna_positions_file == target.resolve()

    data["antenna_layout"]["antenna_positions_file"] = "$DATA/layout.txt"
    with pytest.raises(ConfigPathError) as exc_info:
        resolve_config(data, source=source)
    assert exc_info.value.issues[0].code == "environment_path_syntax"
    assert "explicit path" in exc_info.value.issues[0].hint


def test_skyh5_glob_is_sorted_and_file_list_preserves_input_order(tmp_path):
    for name in ("b.skyh5", "a.skyh5"):
        (tmp_path / name).touch()
    source = ConfigurationSource.for_mapping(
        base_dir=tmp_path,
        invocation_dir=tmp_path,
    )
    glob_data = valid_config_mapping(
        tmp_path,
        sky_sources=[{"kind": "skyh5_multifile", "file_glob": "*.skyh5"}],
    )

    glob_bundle = resolve_config(glob_data, source=source)

    assert glob_bundle.runtime.sky_model.sources[0].options["file_glob"] == (
        (tmp_path / "a.skyh5").resolve(),
        (tmp_path / "b.skyh5").resolve(),
    )

    list_data = valid_config_mapping(
        tmp_path,
        sky_sources=[
            {
                "kind": "skyh5_multifile",
                "filenames": ["b.skyh5", "a.skyh5"],
            }
        ],
    )
    list_bundle = resolve_config(list_data, source=source)
    assert list_bundle.runtime.sky_model.sources[0].options["filenames"] == (
        (tmp_path / "b.skyh5").resolve(),
        (tmp_path / "a.skyh5").resolve(),
    )


def test_zero_match_glob_and_multiple_missing_files_are_stably_ordered(tmp_path):
    data = valid_config_mapping(
        tmp_path,
        antenna_layout={"antenna_positions_file": "missing-antennas.txt"},
        sky_sources=[{"kind": "skyh5_multifile", "file_glob": "missing-*.skyh5"}],
    )

    with pytest.raises(ConfigPathError) as exc_info:
        resolve_config(
            data,
            source=ConfigurationSource.for_mapping(
                base_dir=tmp_path,
                invocation_dir=tmp_path,
            ),
        )

    assert [(issue.path, issue.code) for issue in exc_info.value.issues] == [
        (
            "antenna_layout.antenna_positions_file",
            "input_path_missing",
        ),
        ("sky_model.sources[0].file_glob", "glob_no_regular_files"),
    ]


def test_registered_loader_path_metadata_resolves_without_calling_loader(tmp_path):
    calls: list[str] = []

    @loader_registry.register_loader(
        "_tier1c_file_loader",
        requires_file=True,
        config_fields=["filename"],
        path_options={"filename": "file"},
    )
    def _loader(filename: str, *, precision: PrecisionConfig):
        calls.append(filename)
        raise AssertionError("resolver must not execute a loader")

    try:
        input_file = tmp_path / "custom.dat"
        input_file.touch()
        data = valid_config_mapping(
            tmp_path,
            sky_sources=[
                {
                    "kind": "_tier1c_file_loader",
                    "options": {"filename": "custom.dat"},
                }
            ],
        )
        bundle = resolve_config(
            data,
            source=ConfigurationSource.for_mapping(
                base_dir=tmp_path,
                invocation_dir=tmp_path,
            ),
        )

        request = bundle.runtime.sky_model.sources[0]
        assert request.kind == "_tier1c_file_loader"
        assert request.options["filename"] == input_file.resolve()
        assert calls == []
    finally:
        registry_core._REGISTRY.unregister("_tier1c_file_loader")
