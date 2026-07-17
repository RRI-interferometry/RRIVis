"""Target-contract tests for ``radiosim validate``."""

from __future__ import annotations

import importlib
import webbrowser
from pathlib import Path

import pytest
from click.testing import CliRunner

from radiosim.cli.main import cli
from tests.fixtures.configs import valid_config_mapping, write_config_yaml


def test_validate_uses_resolver_and_never_crosses_runtime_side_effects(
    tmp_path, monkeypatch, recording_simulator
):
    output_dir = tmp_path / "must-not-be-created"
    data = valid_config_mapping(
        tmp_path,
        workflow={
            "output_dir": str(output_dir),
            "save_results": True,
            "plot_results": True,
            "open_plots_in_browser": True,
            "save_log": True,
        },
    )
    config_path = write_config_yaml(tmp_path, data)

    def forbidden_boundary(*args, **kwargs):
        pytest.fail("validate reached a runtime side-effect boundary")

    device_module = importlib.import_module("radiosim.utils.device")
    network_module = importlib.import_module("radiosim.utils.network")
    antenna_module = importlib.import_module("radiosim.core.antenna")
    parallel_module = importlib.import_module("radiosim.core.sky.operations.parallel")
    monkeypatch.setattr(device_module, "get_device_resources", forbidden_boundary)
    monkeypatch.setattr(network_module, "get_network_status", forbidden_boundary)
    monkeypatch.setattr(antenna_module, "read_antenna_positions", forbidden_boundary)
    monkeypatch.setattr(parallel_module, "load_models_parallel", forbidden_boundary)
    monkeypatch.setattr(webbrowser, "open", forbidden_boundary)

    result = CliRunner().invoke(cli, ["validate", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "Configuration is valid" in result.output
    assert recording_simulator.instances == []
    assert not output_dir.exists()


def test_validate_prints_useful_resolved_summary(tmp_path, recording_simulator):
    config_path = write_config_yaml(tmp_path, valid_config_mapping(tmp_path))

    result = CliRunner().invoke(cli, ["validate", str(config_path)])

    assert result.exit_code == 0, result.output
    assert f"Resolved config path: {config_path}" in result.output
    assert f"Document base: {tmp_path}" in result.output
    assert "Backend strategy: numpy" in result.output
    assert "Precision: standard" in result.output
    assert "Frequency channels: 3" in result.output
    assert "Frequency minimum (Hz): 100000000" in result.output
    assert "Frequency maximum (Hz): 102000000" in result.output
    assert "Scientific input paths: 1" in result.output
    assert recording_simulator.instances == []


def test_validate_source_error_uses_common_renderer_not_click_path_rejection(
    tmp_path, recording_simulator
):
    missing = tmp_path / "missing.yaml"

    result = CliRunner().invoke(cli, ["validate", str(missing)])

    assert result.exit_code == 1
    assert "Configuration invalid" in result.output
    assert "source.config_path" in result.output
    assert "Invalid value for 'CONFIG'" not in result.output
    assert recording_simulator.instances == []


def test_validate_preserves_parse_error_classification(tmp_path, recording_simulator):
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("antenna_layout: [\n")

    result = CliRunner().invoke(cli, ["validate", str(config_path)])

    assert result.exit_code == 1
    assert "Configuration invalid" in result.output
    assert "could not parse YAML" in result.output
    assert "Traceback" not in result.output
    assert recording_simulator.instances == []


def test_repository_sample_validates_without_runtime_construction(recording_simulator):
    repository_root = Path(__file__).resolve().parents[3]
    config_path = repository_root / "configs" / "config.yaml"

    result = CliRunner().invoke(cli, ["validate", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "Frequency channels: 101" in result.output
    assert "Scientific input paths: 1" in result.output
    assert recording_simulator.instances == []
