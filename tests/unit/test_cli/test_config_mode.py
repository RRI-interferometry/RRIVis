"""Target-contract tests for resolved config-mode CLI orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from radiosim.api import Simulator as RealSimulator
from radiosim.cli.main import cli, run_config_mode
from radiosim.io.config import load_config
from radiosim.io.config_resolution import SimulationOverrides, WorkflowOverrides
from tests.fixtures.configs import valid_config_mapping, write_config_yaml


def _invoke_config(runner: CliRunner, config_path: Path, *args: str):
    return runner.invoke(cli, ["--config", str(config_path), *args])


def _issue_lines(output: str) -> list[str]:
    return sorted(
        line.strip()
        for line in output.splitlines()
        if any(
            marker in line
            for marker in (
                "instrument.",
                "obs_time.",
                "workflow.",
                "beams.",
                "configuration_source.",
            )
        )
    )


def test_config_mode_constructs_simulator_from_resolved_runtime_only(
    tmp_path, recording_simulator
):
    config_path = write_config_yaml(tmp_path)

    result = _invoke_config(CliRunner(), config_path)

    assert result.exit_code == 0, result.output
    assert len(recording_simulator.instances) == 1
    simulator = recording_simulator.instances[0]
    assert simulator.ran is True
    assert simulator.config.execution.offline is True
    assert not hasattr(simulator.config, "workflow")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("document_backend", ["numpy", "numba"])
def test_cli_backend_default_is_none_and_preserves_document(
    tmp_path, recording_simulator, document_backend
):
    config_path = write_config_yaml(
        tmp_path,
        valid_config_mapping(tmp_path, execution={"backend": document_backend}),
    )

    result = _invoke_config(CliRunner(), config_path)

    assert result.exit_code == 0, result.output
    assert recording_simulator.instances[0].config.execution.backend_strategy == (
        document_backend
    )


@pytest.mark.parametrize("override", ["auto", "numpy", "jax", "numba"])
def test_explicit_cli_backend_is_a_frozen_override(
    tmp_path, recording_simulator, override
):
    config_path = write_config_yaml(
        tmp_path,
        valid_config_mapping(tmp_path, execution={"backend": "numpy"}),
    )

    result = _invoke_config(CliRunner(), config_path, "--backend", override)

    assert result.exit_code == 0, result.output
    assert (
        recording_simulator.instances[0].config.execution.backend_strategy == override
    )


@pytest.mark.parametrize(
    ("document", "flag", "expected"),
    [
        (False, (), False),
        (False, ("--offline",), True),
        (True, (), True),
        (True, ("--online",), False),
    ],
)
def test_cli_offline_online_and_no_override_matrix(
    tmp_path, recording_simulator, document, flag, expected
):
    config_path = write_config_yaml(
        tmp_path,
        valid_config_mapping(tmp_path, execution={"offline": document}),
    )

    result = _invoke_config(CliRunner(), config_path, *flag)

    assert result.exit_code == 0, result.output
    assert recording_simulator.instances[0].config.execution.offline is expected


def test_relative_antenna_and_output_overrides_use_invocation_cwd(
    tmp_path, monkeypatch, recording_simulator
):
    document_dir = tmp_path / "document"
    invocation_dir = tmp_path / "invocation"
    document_dir.mkdir()
    invocation_dir.mkdir()
    config_path = write_config_yaml(
        document_dir,
        valid_config_mapping(document_dir, workflow={"save_results": True}),
    )
    override_antenna = invocation_dir / "override.txt"
    override_antenna.write_text((document_dir / "antennas.txt").read_text())
    monkeypatch.chdir(invocation_dir)

    result = _invoke_config(
        CliRunner(),
        config_path,
        "--antenna-file",
        "override.txt",
        "--sim-data-dir",
        "workflow-output",
    )

    assert result.exit_code == 0, result.output
    simulator = recording_simulator.instances[0]
    assert simulator.config.instrument.source.path == override_antenna
    assert not hasattr(simulator.config, "output_dir")
    assert (invocation_dir / "workflow-output" / "run").exists()


def test_config_mode_builds_frozen_overrides_once_without_mutating_bundle(
    tmp_path, monkeypatch, recording_simulator
):
    config_path = write_config_yaml(tmp_path)
    original = load_config(config_path)
    original_runtime = original.runtime.to_json_safe()
    original_workflow = original.workflow.model_dump(mode="json")
    calls: list[tuple[SimulationOverrides, WorkflowOverrides, bool]] = []
    real_load = load_config

    def spy(path, *, overrides, workflow_overrides, check_input_paths):
        calls.append((overrides, workflow_overrides, check_input_paths))
        return real_load(
            path,
            overrides=overrides,
            workflow_overrides=workflow_overrides,
            check_input_paths=check_input_paths,
        )

    monkeypatch.setattr("radiosim.io.config.load_config", spy)

    exit_code = run_config_mode(
        config_flag=str(config_path),
        antenna_file=None,
        sim_data_dir="override-output",
        backend="auto",
        verbose=0,
        quiet=True,
        offline=False,
    )

    assert exit_code == 0
    assert len(calls) == 1
    simulation, workflow, check_paths = calls[0]
    assert isinstance(simulation, SimulationOverrides)
    assert isinstance(workflow, WorkflowOverrides)
    assert simulation.backend == "auto"
    assert simulation.offline is False
    assert workflow.output_dir == Path("override-output")
    assert check_paths is True
    assert original.runtime.to_json_safe() == original_runtime
    assert original.workflow.model_dump(mode="json") == original_workflow


def test_invalid_antenna_override_fails_after_application_before_simulator(
    tmp_path, recording_simulator
):
    output_dir = tmp_path / "must-not-be-created"
    config_path = write_config_yaml(
        tmp_path,
        valid_config_mapping(
            tmp_path,
            workflow={"output_dir": str(output_dir), "save_log": True},
        ),
    )

    result = _invoke_config(
        CliRunner(), config_path, "--antenna-file", "missing-override.txt"
    )

    assert result.exit_code == 1
    assert "instrument.source.path" in result.output
    assert "does not exist" in result.output
    assert recording_simulator.instances == []
    assert not output_dir.exists()


def test_config_mode_renders_override_error_hierarchy(
    tmp_path, monkeypatch, recording_simulator
):
    from radiosim.io.config import ConfigIssue
    from radiosim.io.config_resolution import ConfigOverrideError

    config_path = write_config_yaml(tmp_path)

    def reject_override(*args, **kwargs):
        raise ConfigOverrideError(
            [
                ConfigIssue(
                    "overrides.backend",
                    "invalid_backend_override",
                    "invalid override",
                    stage="override",
                    category="override",
                )
            ]
        )

    monkeypatch.setattr("radiosim.io.config.load_config", reject_override)

    result = _invoke_config(CliRunner(), config_path, "--backend", "auto")

    assert result.exit_code == 1
    assert "Configuration invalid for config mode (override)" in result.output
    assert "overrides.backend: invalid override" in result.output
    assert recording_simulator.instances == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda data: data["instrument"].update({"default_diameter_m": "wide"}),
            "instrument.default_diameter_m",
        ),
        (
            lambda data: data["obs_time"].update(
                {"duration_seconds": 1.0, "time_step_seconds": 2.0}
            ),
            "obs_time.time_step_seconds",
        ),
        (
            lambda data: data["workflow"].update({"result_format": "uvfits"}),
            "workflow.result_format",
        ),
        (
            lambda data: data["instrument"]["source"].update({"path": "missing.txt"}),
            "instrument.source.path",
        ),
    ],
)
def test_config_mode_renders_typed_schema_semantic_unsupported_and_path_issues(
    tmp_path, recording_simulator, mutate, expected
):
    data = valid_config_mapping(tmp_path)
    mutate(data)
    config_path = write_config_yaml(tmp_path, data, name="invalid.yaml")

    result = _invoke_config(CliRunner(), config_path)

    assert result.exit_code == 1
    assert "Configuration invalid" in result.output
    assert expected in result.output
    assert recording_simulator.instances == []
    assert not (tmp_path / "output").exists()


def test_config_mode_schema_renderer_keeps_all_issues_without_partial_model(
    tmp_path, recording_simulator
):
    data = valid_config_mapping(tmp_path)
    data["instrument"]["default_diameter_m"] = "wide"
    data["beams"]["mode"] = "unknown"
    data["obs_time"]["duration_seconds"] = -1.0
    config_path = write_config_yaml(tmp_path, data, name="invalid.yaml")

    result = _invoke_config(CliRunner(), config_path)

    assert result.exit_code == 1
    assert "3 issue(s)" in result.output
    assert "instrument.default_diameter_m" in result.output
    assert "beams" in result.output
    assert "obs_time.duration_seconds" in result.output
    assert "obs_time.time_step_seconds" not in result.output
    assert recording_simulator.instances == []


def test_config_mode_and_validate_render_equivalent_issue_content(
    tmp_path, recording_simulator
):
    data = valid_config_mapping(tmp_path)
    data["obs_time"].update({"duration_seconds": 1.0, "time_step_seconds": 2.0})
    config_path = write_config_yaml(tmp_path, data, name="invalid.yaml")
    runner = CliRunner()

    config_result = _invoke_config(runner, config_path)
    validate_result = runner.invoke(cli, ["validate", str(config_path)])

    assert config_result.exit_code == validate_result.exit_code == 1
    assert _issue_lines(config_result.output) == _issue_lines(validate_result.output)


def test_validate_accepts_pending_fits_schema_but_config_mode_rejects_runtime(
    tmp_path, recording_simulator
):
    beam_path = tmp_path / "pending.beamfits"
    beam_path.touch()
    data = valid_config_mapping(
        tmp_path,
        beams={
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": beam_path.name},
        },
    )
    config_path = write_config_yaml(tmp_path, data)
    runner = CliRunner()

    validate_result = runner.invoke(cli, ["validate", str(config_path)])
    config_result = _invoke_config(runner, config_path)

    assert validate_result.exit_code == 0, validate_result.output
    assert "Configuration is valid" in validate_result.output
    assert config_result.exit_code == 1
    assert isinstance(config_result.exception, Exception)
    assert "beam_runtime_fits_pending" in str(config_result.exception)
    assert recording_simulator.instances == []


def test_config_mode_and_python_api_consume_same_backend_and_precision(
    tmp_path, recording_simulator
):
    config_path = write_config_yaml(
        tmp_path,
        valid_config_mapping(
            tmp_path,
            execution={"backend": "numpy", "precision": {"preset": "fast"}},
        ),
    )
    api_simulator = RealSimulator.from_yaml(config_path)

    result = _invoke_config(CliRunner(), config_path)

    assert result.exit_code == 0, result.output
    cli_runtime = recording_simulator.instances[-1].config
    assert (
        cli_runtime.execution.backend_strategy
        == api_simulator.config.execution.backend_strategy
    )
    assert cli_runtime.execution.precision == api_simulator.config.execution.precision


def test_workflow_forwards_save_and_plot_policy_after_run(
    tmp_path, recording_simulator
):
    data = valid_config_mapping(
        tmp_path,
        workflow={
            "run_subdir": "chosen-run",
            "result_filename": "science",
            "result_format": "json",
            "save_results": True,
            "overwrite": True,
            "plot_results": True,
            "open_plots_in_browser": True,
            "plotting_backend": "matplotlib",
        },
    )
    config_path = write_config_yaml(tmp_path, data)

    result = _invoke_config(CliRunner(), config_path)

    assert result.exit_code == 0, result.output
    simulator = recording_simulator.instances[0]
    output_dir = tmp_path / "output" / "chosen-run"
    assert simulator.save_calls == [
        (
            (output_dir,),
            {"format": "json", "overwrite": True, "filename": "science"},
        )
    ]
    assert simulator.plot_calls == [
        (
            (),
            {
                "plot_type": "all",
                "output_dir": output_dir,
                "backend": "matplotlib",
                "show": True,
                "overwrite": True,
            },
        )
    ]
    assert output_dir.exists()


def test_existing_output_conflict_aborts_without_overwrite(
    tmp_path, recording_simulator
):
    output_dir = tmp_path / "output" / "run"
    output_dir.mkdir(parents=True)
    (output_dir / "existing.txt").write_text("keep")
    data = valid_config_mapping(
        tmp_path,
        workflow={"save_results": True, "overwrite": False},
    )
    config_path = write_config_yaml(tmp_path, data)

    result = CliRunner().invoke(
        cli,
        ["--config", str(config_path)],
        input="n\n",
    )

    assert result.exit_code == 0
    assert recording_simulator.instances[0].ran is True
    assert recording_simulator.instances[0].save_calls == []
    assert (output_dir / "existing.txt").read_text() == "keep"


def test_generated_run_subdir_is_deterministic_from_resolved_science(
    tmp_path, recording_simulator
):
    from radiosim.cli.workflow import deterministic_run_subdir
    from radiosim.io.config import load_config

    data = valid_config_mapping(
        tmp_path,
        workflow={
            "run_subdir": None,
            "save_results": True,
            "overwrite": True,
            "skip_overwrite_confirmation": True,
        },
    )
    config_path = write_config_yaml(tmp_path, data)
    expected = deterministic_run_subdir(load_config(config_path).runtime)

    first = _invoke_config(CliRunner(), config_path)
    second = _invoke_config(CliRunner(), config_path)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert (tmp_path / "output" / expected).is_dir()


def test_root_help_exposes_tri_state_options_without_implicit_backend_default():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "--backend [auto|numpy|jax|numba]" in result.output
    assert "--offline / --online" in result.output
    backend_line = next(
        line for line in result.output.splitlines() if "--backend" in line
    )
    assert "default" not in backend_line.lower()
