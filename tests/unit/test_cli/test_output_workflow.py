"""Tier 4F owned workflow transaction and collision-policy contracts."""

from __future__ import annotations

import importlib
import json
import logging
import webbrowser
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from radiosim.cli.main import cli
from radiosim.cli.workflow import (
    NonInteractivePromptError,
    ensure_result_workflow_available,
    preflight_cli_workflow,
    run_cli_workflow,
)
from radiosim.io.config import CliWorkflowConfig, RadioSimConfig
from radiosim.io.result_errors import (
    AtomicWriteError,
    OutputCollisionError,
    OverwriteRefusedError,
    PartialCleanupError,
    UnsafeOutputDirectoryError,
)
from radiosim.io.workflow_artifacts import (
    validate_owned_run_directory,
    write_workflow_manifest,
)
from tests.fixtures.configs import (
    resolved_config,
    valid_config_mapping,
    write_config_yaml,
    write_minimal_antenna_file,
)


def _owned_run(path: Path, *, content: bytes = b"owned") -> Path:
    path.mkdir(parents=True)
    artifact = path / "old-result.h5"
    artifact.write_bytes(content)
    write_workflow_manifest(path, [artifact])
    return path


def test_workflow_artifact_module_owns_manifest_and_config_writing():
    module = importlib.import_module("radiosim.io.workflow_artifacts")

    assert callable(module.write_resolved_config_artifact)
    assert callable(module.write_workflow_manifest)
    assert callable(module.validate_owned_run_directory)


@pytest.mark.parametrize("policy", ["error", "replace", "suffix", "prompt"])
def test_cli_workflow_accepts_exact_collision_policy_values(policy):
    workflow = CliWorkflowConfig(collision_policy=policy)

    assert workflow.collision_policy == policy


@pytest.mark.parametrize(
    ("field", "guidance"),
    [
        (
            "overwrite",
            "workflow.overwrite: removed before v1.0; use workflow.collision_policy",
        ),
        (
            "skip_overwrite_confirmation",
            "workflow.skip_overwrite_confirmation: removed before v1.0; "
            "use collision_policy=replace",
        ),
        (
            "prompt_for_output_suffix",
            "workflow.prompt_for_output_suffix: removed before v1.0; "
            "use collision_policy=suffix",
        ),
    ],
)
def test_removed_workflow_fields_fail_with_exact_migration_guidance(
    tmp_path, field, guidance
):
    data = valid_config_mapping(tmp_path)
    data["workflow"][field] = True

    with pytest.raises(ValidationError, match=guidance):
        RadioSimConfig.model_validate(data)


def test_legacy_json_format_fails_with_exact_migration_guidance(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["workflow"]["result_format"] = "json"

    with pytest.raises(
        ValidationError,
        match=(
            "workflow.result_format=json: removed before v1.0; use summary_json or hdf5"
        ),
    ):
        RadioSimConfig.model_validate(data)


def test_save_workflow_is_available_while_plot_workflow_remains_rejected():
    ensure_result_workflow_available(save_results=True, plot_results=False)

    with pytest.raises(Exception, match="plot"):
        ensure_result_workflow_available(save_results=False, plot_results=True)


def test_direct_simulate_uses_exact_final_target_and_never_prompts(
    tmp_path, recording_simulator, monkeypatch
):
    antenna_path = write_minimal_antenna_file(tmp_path)
    target = tmp_path / "direct-result"

    def forbidden_prompt(*args, **kwargs):
        pytest.fail("direct simulate attempted to prompt")

    monkeypatch.setattr("click.prompt", forbidden_prompt)
    result = CliRunner().invoke(
        cli,
        [
            "simulate",
            "--antenna-layout",
            str(antenna_path),
            "--telescope-name",
            "CLI Array",
            "--default-diameter-m",
            "14",
            "--frequencies",
            "100,101.5",
            "--channel-widths-mhz",
            "1,0.5",
            "--sky-model",
            "test",
            "--output",
            str(target),
            "--format",
            "summary_json",
            "--backend",
            "numpy",
            "--latitude",
            "-30.72152",
            "--longitude",
            "21.4283",
            "--height",
            "1073",
            "--start-time",
            "2025-01-01T00:00:00",
        ],
    )

    assert result.exit_code == 0, result.output
    simulator = recording_simulator.instances[0]
    assert len(simulator.save_calls) == 1
    args, kwargs = simulator.save_calls[0]
    assert Path(args[0]) == target
    assert kwargs == {
        "format": importlib.import_module(
            "radiosim.io.result_format"
        ).ResultFormat.SUMMARY_JSON,
        "overwrite": False,
    }


def test_config_mode_plot_preflight_remains_before_runtime(
    tmp_path, recording_simulator
):
    data = valid_config_mapping(tmp_path, workflow={"plot_results": True})
    config_path = write_config_yaml(tmp_path, data)

    result = CliRunner().invoke(cli, ["--config", str(config_path)])

    assert result.exit_code == 1
    assert "plot" in result.output
    assert recording_simulator.instances == []


def test_manifest_rejects_duplicate_keys_traversal_and_unlisted_content(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    artifact = run / "result.h5"
    artifact.write_bytes(b"result")
    manifest = run / "manifest.json"
    manifest.write_text(
        '{"schema":"radiosim.workflow-manifest.v1",'
        '"schema":"radiosim.workflow-manifest.v1","artifacts":[]}',
        encoding="utf-8",
    )
    with pytest.raises(UnsafeOutputDirectoryError, match="strict bounded JSON"):
        validate_owned_run_directory(run)

    manifest.write_text(
        json.dumps(
            {
                "schema": "radiosim.workflow-manifest.v1",
                "artifacts": [
                    {
                        "kind": "file",
                        "path": "../result.h5",
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(UnsafeOutputDirectoryError, match="unsafe"):
        validate_owned_run_directory(run)

    manifest.unlink()
    write_workflow_manifest(run, [artifact])
    (run / "hostile.txt").write_text("not owned", encoding="utf-8")
    with pytest.raises(UnsafeOutputDirectoryError, match="unlisted"):
        validate_owned_run_directory(run)


def test_manifest_rejects_decoder_recursion_as_typed_hostile_input(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    nesting = 2_000
    (run / "manifest.json").write_text(
        '{"schema":"radiosim.workflow-manifest.v1","artifacts":'
        + "[" * nesting
        + "0"
        + "]" * nesting
        + "}",
        encoding="utf-8",
    )

    with pytest.raises(UnsafeOutputDirectoryError, match="strict bounded JSON"):
        validate_owned_run_directory(run)


def test_error_replace_and_suffix_preflight_do_not_mutate_targets(tmp_path):
    bundle = resolved_config(tmp_path)
    base = tmp_path / "runs" / "run"
    absent = CliWorkflowConfig(
        output_dir=base.parent,
        run_subdir=base.name,
        save_log=True,
        collision_policy="error",
    )
    plan = preflight_cli_workflow(absent, runtime=bundle.runtime)
    assert plan.target == base
    assert plan.publish_mode == "no_clobber"
    assert not base.parent.exists()

    _owned_run(base)
    with pytest.raises(OverwriteRefusedError):
        preflight_cli_workflow(absent, runtime=bundle.runtime)
    replace = absent.model_copy(update={"collision_policy": "replace"})
    replace_plan = preflight_cli_workflow(replace, runtime=bundle.runtime)
    assert replace_plan.target == base
    assert replace_plan.publish_mode == "exchange"

    suffix = absent.model_copy(update={"collision_policy": "suffix"})
    (base.parent / "run-001").mkdir()
    (base.parent / "run-003").mkdir()
    suffix_plan = preflight_cli_workflow(suffix, runtime=bundle.runtime)
    assert suffix_plan.target == base.parent / "run-002"
    assert (base / "old-result.h5").read_bytes() == b"owned"


def test_suffix_policy_exhaustion_is_bounded_and_noninteractive(tmp_path):
    bundle = resolved_config(tmp_path)
    base = _owned_run(tmp_path / "runs" / "run")
    for index in range(1, 1000):
        (base.parent / f"run-{index:03d}").mkdir()
    workflow = CliWorkflowConfig(
        output_dir=base.parent,
        run_subdir=base.name,
        save_log=True,
        collision_policy="suffix",
    )

    with pytest.raises(OutputCollisionError, match="-001 through -999"):
        preflight_cli_workflow(workflow, runtime=bundle.runtime)


def test_suffix_policy_can_select_final_bounded_candidate(tmp_path):
    bundle = resolved_config(tmp_path)
    base = _owned_run(tmp_path / "runs" / "run")
    for index in range(1, 999):
        (base.parent / f"run-{index:03d}").mkdir()
    workflow = CliWorkflowConfig(
        output_dir=base.parent,
        run_subdir=base.name,
        save_log=True,
        collision_policy="suffix",
    )

    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)

    assert plan.target == base.parent / "run-999"
    assert plan.publish_mode == "no_clobber"


def test_prompt_policy_checks_tty_once_before_runtime_or_mutation(
    tmp_path, monkeypatch
):
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run")
    workflow = CliWorkflowConfig(
        output_dir=target.parent,
        run_subdir=target.name,
        save_log=True,
        collision_policy="prompt",
    )
    prompts: list[str] = []

    monkeypatch.setattr(
        "click.get_text_stream",
        lambda name: SimpleNamespace(isatty=lambda: False),
    )
    monkeypatch.setattr(
        "click.confirm",
        lambda *args, **kwargs: pytest.fail("non-TTY attempted to prompt"),
    )
    with pytest.raises(NonInteractivePromptError):
        preflight_cli_workflow(workflow, runtime=bundle.runtime)
    assert (target / "old-result.h5").read_bytes() == b"owned"

    monkeypatch.setattr(
        "click.get_text_stream",
        lambda name: SimpleNamespace(isatty=lambda: True),
    )

    def decline(message, **kwargs):
        prompts.append(message)
        return False

    monkeypatch.setattr("click.confirm", decline)
    declined = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    assert declined.declined is True
    assert prompts == [f"Replace owned workflow run {target}?"]
    assert (target / "old-result.h5").read_bytes() == b"owned"

    prompts.clear()

    def accept(message, **kwargs):
        prompts.append(message)
        return True

    monkeypatch.setattr("click.confirm", accept)
    accepted = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    assert accepted.declined is False
    assert accepted.publish_mode == "exchange"
    assert prompts == [f"Replace owned workflow run {target}?"]
    assert (target / "old-result.h5").read_bytes() == b"owned"


def test_writer_failure_rolls_back_staging_and_preserves_owned_run(tmp_path):
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run", content=b"old bytes")
    workflow = CliWorkflowConfig(
        output_dir=target.parent,
        run_subdir=target.name,
        save_results=True,
        result_format="hdf5",
        collision_policy="replace",
    )
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)

    def fail_save(path, **kwargs):
        Path(path).with_suffix(".h5").write_bytes(b"partial")
        raise RuntimeError("injected writer failure")

    simulator = SimpleNamespace(save=fail_save)
    with pytest.raises(Exception, match="workflow transaction failed"):
        run_cli_workflow(
            simulator,
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    assert (target / "old-result.h5").read_bytes() == b"old bytes"
    assert not any(path.name.startswith(".run.") for path in target.parent.iterdir())


@pytest.mark.parametrize(
    ("failure_point", "failure_message"),
    [
        ("logger", "logger could not be initialized"),
        ("logger_close", "logger failed to flush or close"),
        ("config", "workflow transaction failed"),
        ("manifest", "workflow transaction failed"),
        ("publish", "injected publication failure"),
    ],
)
def test_logger_manifest_and_publish_failures_preserve_owned_run(
    tmp_path,
    monkeypatch,
    failure_point,
    failure_message,
):
    workflow_module = importlib.import_module("radiosim.cli.workflow")
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run", content=b"old bytes")
    workflow = CliWorkflowConfig(
        output_dir=target.parent,
        run_subdir=target.name,
        save_log=True,
        collision_policy="replace",
    )
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)

    if failure_point == "logger":
        monkeypatch.setattr(
            logging,
            "FileHandler",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("injected logger failure")
            ),
        )
    elif failure_point == "logger_close":
        original_close = logging.FileHandler.close

        def fail_after_close(handler):
            original_close(handler)
            raise OSError("injected logger close failure")

        monkeypatch.setattr(logging.FileHandler, "close", fail_after_close)
    elif failure_point == "config":
        monkeypatch.setattr(
            workflow_module,
            "write_resolved_config_artifact",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("injected config artifact failure")
            ),
        )
    elif failure_point == "manifest":
        monkeypatch.setattr(
            workflow_module,
            "write_workflow_manifest",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("injected manifest failure")
            ),
        )
    else:
        monkeypatch.setattr(
            workflow_module,
            "exchange_directories",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AtomicWriteError("injected publication failure")
            ),
        )

    with pytest.raises(Exception, match=failure_message):
        run_cli_workflow(
            SimpleNamespace(),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    assert (target / "old-result.h5").read_bytes() == b"old bytes"
    assert not any(path.name.startswith(".run.") for path in target.parent.iterdir())


def test_cleanup_failure_reports_exact_residual_path(tmp_path, monkeypatch):
    workflow_module = importlib.import_module("radiosim.cli.workflow")
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run", content=b"old bytes")
    workflow = CliWorkflowConfig(
        output_dir=target.parent,
        run_subdir=target.name,
        save_results=True,
        result_format="hdf5",
        collision_policy="replace",
    )
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)

    def fail_save(path, **kwargs):
        raise RuntimeError("injected writer failure")

    def fail_cleanup(path):
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(workflow_module, "remove_temporary_directory", fail_cleanup)
    with pytest.raises(PartialCleanupError) as caught:
        run_cli_workflow(
            SimpleNamespace(save=fail_save),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    residual = caught.value.residual_path
    assert residual.parent == target.parent
    assert residual.name.startswith(".run.")
    assert residual.is_dir()
    assert (target / "old-result.h5").read_bytes() == b"old bytes"

    monkeypatch.undo()
    workflow_module.remove_temporary_directory(residual)


def test_successful_logging_closes_handler_before_publication(tmp_path):
    bundle = resolved_config(tmp_path)
    target = tmp_path / "runs" / "run"
    workflow = CliWorkflowConfig(
        output_dir=target.parent,
        run_subdir=target.name,
        save_log=True,
        collision_policy="error",
    )
    before = tuple(logging.getLogger().handlers)

    artifacts = run_cli_workflow(
        SimpleNamespace(),
        workflow,
        runtime=bundle.runtime,
        provenance=bundle.provenance,
    )

    assert tuple(logging.getLogger().handlers) == before
    assert target / "simulation.log" in artifacts
    assert validate_owned_run_directory(target).run_directory == target
    renamed = target.with_name("renamed")
    target.rename(renamed)
    assert (renamed / "simulation.log").is_file()


def test_plot_preflight_never_opens_browser(tmp_path, monkeypatch):
    bundle = resolved_config(tmp_path)
    workflow = CliWorkflowConfig(
        output_dir=tmp_path / "runs",
        run_subdir="run",
        plot_results=True,
        open_plots_in_browser=True,
    )
    monkeypatch.setattr(
        webbrowser,
        "open",
        lambda *args, **kwargs: pytest.fail("Tier 4F opened a browser"),
    )

    with pytest.raises(Exception, match="plot"):
        preflight_cli_workflow(workflow, runtime=bundle.runtime)

    assert not (tmp_path / "runs").exists()


def test_concurrent_target_creation_never_overwrites_racing_directory(tmp_path):
    bundle = resolved_config(tmp_path)
    target = tmp_path / "runs" / "run"
    workflow = CliWorkflowConfig(
        output_dir=target.parent,
        run_subdir=target.name,
        save_results=True,
        result_format="hdf5",
        collision_policy="error",
    )
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    target.mkdir(parents=True)
    (target / "racing.txt").write_text("keep", encoding="utf-8")

    def save(path, **kwargs):
        result_path = Path(path).with_suffix(".h5")
        result_path.write_bytes(b"new")
        return result_path

    with pytest.raises(OutputCollisionError, match="concurrently"):
        run_cli_workflow(
            SimpleNamespace(save=save),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    assert (target / "racing.txt").read_text(encoding="utf-8") == "keep"
    assert not any(path.name.startswith(".run.") for path in target.parent.iterdir())
