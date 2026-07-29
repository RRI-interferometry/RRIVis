"""Tier 4F owned workflow transaction and collision-policy contracts."""

from __future__ import annotations

import importlib
import json
import logging
import os
import webbrowser
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from radiosim.cli.main import cli
from radiosim.cli.workflow import (
    NonInteractivePromptError,
    WorkflowExecutionPlan,
    WorkflowOutputError,
    preflight_cli_workflow,
    run_cli_workflow,
)
from radiosim.io.atomic_paths import remove_temporary_directory
from radiosim.io.config import CliWorkflowConfig, RadioSimConfig
from radiosim.io.result_errors import (
    AtomicWriteError,
    AtomicWriteUnsupportedError,
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


def _replacement_workflow(target: Path) -> CliWorkflowConfig:
    return CliWorkflowConfig(
        output_dir=target.parent,
        run_subdir=target.name,
        save_log=True,
        collision_policy="replace",
    )


def _assert_new_run_is_published(target: Path, artifacts: tuple[Path, ...]) -> None:
    owned = validate_owned_run_directory(target)

    assert owned.run_directory == target
    assert {path.name for path in artifacts} == {
        "manifest.json",
        "resolved-config.yaml",
        "simulation.log",
    }
    assert all(path.parent == target and path.exists() for path in artifacts)


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


def test_tier4g_removes_the_dedicated_plot_availability_preflight():
    import radiosim.cli.workflow as workflow_module

    assert not hasattr(workflow_module, "ensure_result_workflow_available")
    assert "ensure_result_workflow_available" not in workflow_module.__all__


def test_plot_contract_preflight_precedes_every_directory_decision(tmp_path):
    from radiosim.visualization.errors import ResultPlotContractError

    bundle = resolved_config(tmp_path)
    runs = tmp_path / "runs"
    workflow = CliWorkflowConfig(
        output_dir=runs,
        run_subdir="run",
        plot_results=True,
        plotting_backend="matplotlib",
    )

    with pytest.raises(ResultPlotContractError, match="bokeh"):
        preflight_cli_workflow(workflow, runtime=bundle.runtime)

    assert not runs.exists()
    accepted = preflight_cli_workflow(
        workflow.model_copy(update={"plotting_backend": "bokeh"}),
        runtime=bundle.runtime,
    )
    assert accepted.enabled is True
    assert accepted.target == runs / "run"
    assert not runs.exists()


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


def test_config_mode_plot_contract_failure_precedes_runtime(
    tmp_path, recording_simulator
):
    data = valid_config_mapping(
        tmp_path,
        workflow={"plot_results": True, "plotting_backend": "matplotlib"},
    )
    config_path = write_config_yaml(tmp_path, data)

    result = CliRunner().invoke(cli, ["--config", str(config_path)])

    assert result.exit_code == 1
    assert "bokeh" in result.output
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

    def fail_cleanup(path, *args, **kwargs):
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(workflow_module, "remove_directory_by_identity", fail_cleanup)
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
    remove_temporary_directory(residual)


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


def _staging_plotter(target: Path, staged: list[Path], recorded: list[dict]):
    """Return a renderer double that writes only into the staged run directory."""

    def plot(**kwargs):
        recorded.append(dict(kwargs))
        directory = Path(kwargs["output_dir"])
        assert directory != target
        written = []
        for name in ("antenna_layout.html", "visibility-phase-lsts.html"):
            path = directory / name
            path.write_text("<html></html>", encoding="utf-8")
            written.append(path)
        staged.extend(written)
        return tuple(written)

    return plot


def test_workflow_plots_into_staging_and_opens_published_paths_last(
    tmp_path, monkeypatch
):
    bundle = resolved_config(tmp_path)
    target = tmp_path / "runs" / "run"
    workflow = CliWorkflowConfig(
        output_dir=target.parent,
        run_subdir=target.name,
        plot_results=True,
        open_plots_in_browser=True,
        visibility_phase_unit="degrees",
    )
    staged: list[Path] = []
    recorded: list[dict] = []
    opened: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        webbrowser,
        "open",
        lambda url, *args, **kwargs: opened.append((url, target.is_dir())),
    )

    artifacts = run_cli_workflow(
        SimpleNamespace(plot=_staging_plotter(target, staged, recorded)),
        workflow,
        runtime=bundle.runtime,
        provenance=bundle.provenance,
    )

    assert len(recorded) == 1
    assert recorded[0]["show"] is False
    assert recorded[0]["backend"] == "bokeh"
    assert recorded[0]["overwrite"] is False
    assert recorded[0]["visibility_phase_unit"] == "degrees"
    assert [path.name for path in artifacts] == [
        "manifest.json",
        "resolved-config.yaml",
        "antenna_layout.html",
        "visibility-phase-lsts.html",
    ]
    assert all(path.parent == target and path.is_file() for path in artifacts)
    assert not any(path.exists() for path in staged)
    assert [url for url, _ in opened] == [
        (target / "antenna_layout.html").as_uri(),
        (target / "visibility-phase-lsts.html").as_uri(),
    ]
    assert all(existed for _url, existed in opened)
    assert validate_owned_run_directory(target).run_directory == target


def test_workflow_never_opens_a_browser_when_the_flag_is_false(tmp_path, monkeypatch):
    bundle = resolved_config(tmp_path)
    target = tmp_path / "runs" / "run"
    workflow = CliWorkflowConfig(
        output_dir=target.parent,
        run_subdir=target.name,
        plot_results=True,
        open_plots_in_browser=False,
    )
    monkeypatch.setattr(
        webbrowser,
        "open",
        lambda *args, **kwargs: pytest.fail("workflow opened an unrequested browser"),
    )

    artifacts = run_cli_workflow(
        SimpleNamespace(plot=_staging_plotter(target, [], [])),
        workflow,
        runtime=bundle.runtime,
        provenance=bundle.provenance,
    )

    assert target / "visibility-phase-lsts.html" in artifacts


def test_workflow_renderer_failure_leaves_no_published_run(tmp_path, monkeypatch):
    bundle = resolved_config(tmp_path)
    target = tmp_path / "runs" / "run"
    workflow = CliWorkflowConfig(
        output_dir=target.parent,
        run_subdir=target.name,
        plot_results=True,
        open_plots_in_browser=True,
    )
    monkeypatch.setattr(
        webbrowser,
        "open",
        lambda *args, **kwargs: pytest.fail("browser opened after a render failure"),
    )

    def failing_plot(**_kwargs):
        raise RuntimeError("controlled renderer failure")

    with pytest.raises(AtomicWriteError):
        run_cli_workflow(
            SimpleNamespace(plot=failing_plot),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
        )

    assert not target.exists()
    assert list(target.parent.iterdir()) == []


def test_workflow_renderer_escaping_staging_is_rejected(tmp_path, monkeypatch):
    bundle = resolved_config(tmp_path)
    target = tmp_path / "runs" / "run"
    escape = tmp_path / "escape.html"
    workflow = CliWorkflowConfig(
        output_dir=target.parent,
        run_subdir=target.name,
        plot_results=True,
    )

    def escaping_plot(**_kwargs):
        escape.write_text("<html></html>", encoding="utf-8")
        return (escape,)

    with pytest.raises(WorkflowOutputError, match="outside the staged run"):
        run_cli_workflow(
            SimpleNamespace(plot=escaping_plot),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
        )

    assert not target.exists()


def test_workflow_browser_failure_keeps_the_published_run(tmp_path, monkeypatch):
    bundle = resolved_config(tmp_path)
    target = tmp_path / "runs" / "run"
    workflow = CliWorkflowConfig(
        output_dir=target.parent,
        run_subdir=target.name,
        plot_results=True,
        open_plots_in_browser=True,
    )

    def failing_open(*_args, **_kwargs):
        raise OSError("no browser available")

    monkeypatch.setattr(webbrowser, "open", failing_open)

    artifacts = run_cli_workflow(
        SimpleNamespace(plot=_staging_plotter(target, [], [])),
        workflow,
        runtime=bundle.runtime,
        provenance=bundle.provenance,
    )

    assert (target / "visibility-phase-lsts.html").is_file()
    assert validate_owned_run_directory(target).run_directory == target
    assert target / "antenna_layout.html" in artifacts


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


def test_replacement_race_preserves_unknown_exchanged_out_directory(
    tmp_path,
    monkeypatch,
):
    workflow_module = importlib.import_module("radiosim.cli.workflow")
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run", content=b"validated owned")
    workflow = _replacement_workflow(target)
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    original_exchange = workflow_module.exchange_directories
    original_cleanup = workflow_module.remove_directory_by_identity
    validated_aside = target.with_name("validated-owned-aside")
    sentinel = b"unique unknown user bytes"
    exchanged_out: list[Path] = []
    cleanup_calls: list[Path] = []

    def race_then_exchange(staging, final, parent_fd):
        final.rename(validated_aside)
        final.mkdir()
        (final / "unknown.bin").write_bytes(sentinel)
        original_exchange(staging, final, parent_fd)
        exchanged_out.append(staging)
        assert (staging / "unknown.bin").read_bytes() == sentinel

    def record_cleanup(path, parent_fd, **kwargs):
        cleanup_calls.append(path)
        original_cleanup(path, parent_fd, **kwargs)

    monkeypatch.setattr(workflow_module, "exchange_directories", race_then_exchange)
    monkeypatch.setattr(
        workflow_module,
        "remove_directory_by_identity",
        record_cleanup,
    )

    with pytest.raises(PartialCleanupError) as caught:
        run_cli_workflow(
            SimpleNamespace(),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    assert exchanged_out == [caught.value.residual_path]
    residual = caught.value.residual_path
    assert residual.is_dir()
    assert (residual / "unknown.bin").read_bytes() == sentinel
    assert cleanup_calls == []
    assert (validated_aside / "old-result.h5").read_bytes() == b"validated owned"
    _assert_new_run_is_published(
        target,
        (
            target / "manifest.json",
            target / "simulation.log",
            target / "resolved-config.yaml",
        ),
    )
    assert (
        sum(
            (candidate / "unknown.bin").read_bytes() == sentinel
            for candidate in target.parent.iterdir()
            if (candidate / "unknown.bin").is_file()
        )
        == 1
    )


def test_replacement_race_rejects_valid_owned_directory_with_wrong_identity(
    tmp_path,
    monkeypatch,
):
    workflow_module = importlib.import_module("radiosim.cli.workflow")
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run", content=b"validated owned")
    workflow = _replacement_workflow(target)
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    original_exchange = workflow_module.exchange_directories
    validated_aside = target.with_name("validated-owned-aside")
    exchanged_out: list[Path] = []

    def race_then_exchange(staging, final, parent_fd):
        final.rename(validated_aside)
        _owned_run(final, content=b"forged but valid owned run")
        original_exchange(staging, final, parent_fd)
        exchanged_out.append(staging)

    monkeypatch.setattr(workflow_module, "exchange_directories", race_then_exchange)
    monkeypatch.setattr(
        workflow_module,
        "remove_directory_by_identity",
        lambda *args, **kwargs: pytest.fail(
            "wrong-identity residual reached recursive cleanup"
        ),
    )

    with pytest.raises(PartialCleanupError) as caught:
        run_cli_workflow(
            SimpleNamespace(),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    assert exchanged_out == [caught.value.residual_path]
    residual = caught.value.residual_path
    assert validate_owned_run_directory(residual).run_directory == residual
    assert (residual / "old-result.h5").read_bytes() == (b"forged but valid owned run")
    assert validate_owned_run_directory(target).run_directory == target


@pytest.mark.parametrize("initial_contract", ["owned", "empty"])
def test_post_exchange_content_change_preserves_exact_residual(
    tmp_path,
    monkeypatch,
    initial_contract,
):
    workflow_module = importlib.import_module("radiosim.cli.workflow")
    bundle = resolved_config(tmp_path)
    target = tmp_path / "runs" / "run"
    if initial_contract == "owned":
        _owned_run(target, content=b"validated owned")
    else:
        target.mkdir(parents=True)
    workflow = _replacement_workflow(target)
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    original_exchange = workflow_module.exchange_directories
    exchanged_out: list[Path] = []

    def exchange_then_change_contents(staging, final, parent_fd):
        original_exchange(staging, final, parent_fd)
        exchanged_out.append(staging)
        (staging / "unknown-after-exchange.bin").write_bytes(
            b"post-exchange unknown bytes"
        )

    monkeypatch.setattr(
        workflow_module,
        "exchange_directories",
        exchange_then_change_contents,
    )
    monkeypatch.setattr(
        workflow_module,
        "remove_directory_by_identity",
        lambda *args, **kwargs: pytest.fail(
            "changed residual reached recursive cleanup"
        ),
    )

    with pytest.raises(PartialCleanupError) as caught:
        run_cli_workflow(
            SimpleNamespace(),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    assert exchanged_out == [caught.value.residual_path]
    residual = caught.value.residual_path
    assert (residual / "unknown-after-exchange.bin").read_bytes() == (
        b"post-exchange unknown bytes"
    )
    assert validate_owned_run_directory(target).run_directory == target


def test_swap_after_post_exchange_validation_never_reaches_recursive_cleanup(
    tmp_path,
    monkeypatch,
):
    workflow_module = importlib.import_module("radiosim.cli.workflow")
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run", content=b"validated owned")
    workflow = _replacement_workflow(target)
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    original_revalidate = workflow_module._revalidate_replacement_directory
    original_cleanup = workflow_module.remove_directory_by_identity
    validated_aside = target.with_name("validated-owned-residual-aside")
    sentinel = b"unknown bytes after post-exchange validation"
    cleanup_calls: list[Path] = []
    swapped = False

    def revalidate_then_swap(path, expected, *, phase):
        nonlocal swapped
        original_revalidate(path, expected, phase=phase)
        if phase == "after exchange" and not swapped:
            swapped = True
            path.rename(validated_aside)
            path.mkdir()
            (path / "unknown.bin").write_bytes(sentinel)

    def record_cleanup(path, parent_fd, **kwargs):
        cleanup_calls.append(path)
        original_cleanup(path, parent_fd, **kwargs)

    monkeypatch.setattr(
        workflow_module,
        "_revalidate_replacement_directory",
        revalidate_then_swap,
    )
    monkeypatch.setattr(
        workflow_module,
        "remove_directory_by_identity",
        record_cleanup,
    )

    with pytest.raises(PartialCleanupError) as caught:
        run_cli_workflow(
            SimpleNamespace(),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    residual = caught.value.residual_path
    assert (residual / "unknown.bin").read_bytes() == sentinel
    assert cleanup_calls == [residual]
    assert (validated_aside / "old-result.h5").read_bytes() == b"validated owned"
    assert validate_owned_run_directory(target).run_directory == target


def test_post_exchange_cleanup_failure_retains_exact_old_run_residual(
    tmp_path,
    monkeypatch,
):
    workflow_module = importlib.import_module("radiosim.cli.workflow")
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run", content=b"validated owned")
    workflow = _replacement_workflow(target)
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    cleanup_attempts: list[Path] = []

    def fail_cleanup(path, *args, **kwargs):
        cleanup_attempts.append(path)
        raise OSError("injected post-exchange cleanup failure")

    monkeypatch.setattr(workflow_module, "remove_directory_by_identity", fail_cleanup)

    with pytest.raises(PartialCleanupError) as caught:
        run_cli_workflow(
            SimpleNamespace(),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    assert cleanup_attempts == [caught.value.residual_path]
    residual = caught.value.residual_path
    assert (residual / "old-result.h5").read_bytes() == b"validated owned"
    assert validate_owned_run_directory(residual).run_directory == residual
    assert validate_owned_run_directory(target).run_directory == target


@pytest.mark.parametrize(
    "replacement_kind",
    ["missing", "symlink", "file", "directory", "owned_directory", "fifo"],
)
def test_changed_target_before_exchange_is_preserved_without_publication(
    tmp_path,
    monkeypatch,
    replacement_kind,
):
    workflow_module = importlib.import_module("radiosim.cli.workflow")
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run", content=b"validated owned")
    workflow = _replacement_workflow(target)
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    validated_aside = target.with_name("validated-owned-aside")
    original_path_status = workflow_module._path_status
    changed = False

    def replace_target_before_status(path):
        nonlocal changed
        if path == target and not changed:
            changed = True
            target.rename(validated_aside)
            if replacement_kind == "symlink":
                target.symlink_to(validated_aside, target_is_directory=True)
            elif replacement_kind == "file":
                target.write_bytes(b"changed regular file")
            elif replacement_kind == "directory":
                target.mkdir()
                (target / "unknown.bin").write_bytes(b"changed directory")
            elif replacement_kind == "owned_directory":
                _owned_run(target, content=b"changed valid owned directory")
            elif replacement_kind == "fifo":
                os.mkfifo(target)
        return original_path_status(path)

    monkeypatch.setattr(workflow_module, "_path_status", replace_target_before_status)
    monkeypatch.setattr(
        workflow_module,
        "exchange_directories",
        lambda *args, **kwargs: pytest.fail("changed target reached exchange"),
    )

    with pytest.raises(OutputCollisionError):
        run_cli_workflow(
            SimpleNamespace(),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    assert (validated_aside / "old-result.h5").read_bytes() == b"validated owned"
    if replacement_kind == "missing":
        assert not target.exists() and not target.is_symlink()
    elif replacement_kind == "symlink":
        assert target.is_symlink()
    elif replacement_kind == "file":
        assert target.read_bytes() == b"changed regular file"
    elif replacement_kind == "directory":
        assert (target / "unknown.bin").read_bytes() == b"changed directory"
    elif replacement_kind == "owned_directory":
        assert (target / "old-result.h5").read_bytes() == (
            b"changed valid owned directory"
        )
    else:
        assert target.exists()
    assert not any(path.name.startswith(".run.") for path in target.parent.iterdir())


def test_empty_directory_replacement_captures_identity_and_publishes_normally(
    tmp_path,
):
    bundle = resolved_config(tmp_path)
    target = tmp_path / "runs" / "run"
    target.mkdir(parents=True)
    workflow = _replacement_workflow(target)

    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)

    assert plan.publish_mode == "exchange"
    assert plan._replacement_identity is not None
    artifacts = run_cli_workflow(
        SimpleNamespace(),
        workflow,
        runtime=bundle.runtime,
        provenance=bundle.provenance,
        plan=plan,
    )
    _assert_new_run_is_published(target, artifacts)
    assert not any(path.name.startswith(".run.") for path in target.parent.iterdir())


def test_empty_directory_replacement_race_preserves_unknown_directory(
    tmp_path,
    monkeypatch,
):
    workflow_module = importlib.import_module("radiosim.cli.workflow")
    bundle = resolved_config(tmp_path)
    target = tmp_path / "runs" / "run"
    target.mkdir(parents=True)
    workflow = _replacement_workflow(target)
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    original_exchange = workflow_module.exchange_directories
    validated_aside = target.with_name("validated-empty-aside")
    sentinel = b"unknown bytes replacing empty directory"

    def race_then_exchange(staging, final, parent_fd):
        final.rename(validated_aside)
        final.mkdir()
        (final / "unknown.bin").write_bytes(sentinel)
        original_exchange(staging, final, parent_fd)

    monkeypatch.setattr(workflow_module, "exchange_directories", race_then_exchange)

    with pytest.raises(PartialCleanupError) as caught:
        run_cli_workflow(
            SimpleNamespace(),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    residual = caught.value.residual_path
    assert (residual / "unknown.bin").read_bytes() == sentinel
    assert validate_owned_run_directory(target).run_directory == target


def test_exchange_plan_without_captured_identity_fails_before_any_mutation(
    tmp_path,
    monkeypatch,
):
    workflow_module = importlib.import_module("radiosim.cli.workflow")
    bundle = resolved_config(tmp_path)
    target = tmp_path / "must-not-exist" / "run"
    workflow = _replacement_workflow(target)
    incomplete = WorkflowExecutionPlan(True, False, target, "exchange")

    monkeypatch.setattr(
        workflow_module,
        "write_resolved_config_artifact",
        lambda *args, **kwargs: pytest.fail("incomplete plan reached writer work"),
    )
    monkeypatch.setattr(
        workflow_module,
        "exchange_directories",
        lambda *args, **kwargs: pytest.fail("incomplete plan reached exchange"),
    )

    with pytest.raises(WorkflowOutputError, match="identity"):
        run_cli_workflow(
            SimpleNamespace(),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=incomplete,
        )

    assert not target.parent.exists()


def test_normal_replacement_removes_only_exact_validated_owned_run(tmp_path):
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run", content=b"old bytes")
    sibling = target.parent / "sibling"
    sibling.mkdir()
    (sibling / "keep.bin").write_bytes(b"keep sibling")
    workflow = _replacement_workflow(target)
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    before_handlers = tuple(logging.getLogger().handlers)

    artifacts = run_cli_workflow(
        SimpleNamespace(),
        workflow,
        runtime=bundle.runtime,
        provenance=bundle.provenance,
        plan=plan,
    )

    _assert_new_run_is_published(target, artifacts)
    assert not (target / "old-result.h5").exists()
    assert (sibling / "keep.bin").read_bytes() == b"keep sibling"
    assert tuple(logging.getLogger().handlers) == before_handlers
    assert not any(path.name.startswith(".run.") for path in target.parent.iterdir())


def test_cleanup_swap_after_final_validation_preserves_unknown_residual(
    tmp_path,
    monkeypatch,
):
    """Never delete an unknown directory swapped in after validation."""
    workflow_module = importlib.import_module("radiosim.cli.workflow")
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run", content=b"validated owned")
    workflow = _replacement_workflow(target)
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    original_cleanup = workflow_module.remove_directory_by_identity
    validated_aside = target.with_name("validated-owned-after-final-check")
    sentinel = b"unknown bytes swapped after final validation"
    cleanup_calls: list[Path] = []

    def swap_then_cleanup(path, parent_fd, **kwargs):
        cleanup_calls.append(path)
        path.rename(validated_aside)
        path.mkdir()
        (path / "unknown.bin").write_bytes(sentinel)
        original_cleanup(path, parent_fd, **kwargs)

    monkeypatch.setattr(
        workflow_module,
        "remove_directory_by_identity",
        swap_then_cleanup,
    )

    with pytest.raises(PartialCleanupError) as caught:
        run_cli_workflow(
            SimpleNamespace(),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    assert caught.value.residual_path == cleanup_calls[0]
    residual = caught.value.residual_path
    assert (residual / "unknown.bin").read_bytes() == sentinel
    assert (validated_aside / "old-result.h5").read_bytes() == b"validated owned"
    assert validate_owned_run_directory(target).run_directory == target


def test_cleanup_swap_after_descriptor_open_preserves_unknown_residual(
    tmp_path,
    monkeypatch,
):
    """Traversal stays pinned when the residual pathname is replaced post-open."""
    atomic_paths = importlib.import_module("radiosim.io.atomic_paths")
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run", content=b"validated owned")
    workflow = _replacement_workflow(target)
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    original_contents = atomic_paths._remove_pinned_directory_contents
    validated_aside = target.with_name("validated-owned-after-descriptor-open")
    sentinel = b"unknown bytes after descriptor open"
    swapped = False

    def swap_after_open(descriptor, *, path):
        nonlocal swapped
        if not swapped:
            swapped = True
            path.rename(validated_aside)
            path.mkdir()
            (path / "unknown.bin").write_bytes(sentinel)
        original_contents(descriptor, path=path)

    monkeypatch.setattr(
        atomic_paths,
        "_remove_pinned_directory_contents",
        swap_after_open,
    )

    with pytest.raises(PartialCleanupError) as caught:
        run_cli_workflow(
            SimpleNamespace(),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    residual = caught.value.residual_path
    assert residual.is_dir()
    assert (residual / "unknown.bin").read_bytes() == sentinel
    assert not (validated_aside / "unknown.bin").exists()
    assert validate_owned_run_directory(target).run_directory == target


def test_cleanup_swap_before_top_level_removal_preserves_unknown_residual(
    tmp_path,
    monkeypatch,
):
    """The final parent-relative identity inspection precedes top-level rmdir."""
    atomic_paths = importlib.import_module("radiosim.io.atomic_paths")
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run", content=b"validated owned")
    workflow = _replacement_workflow(target)
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    original_contents = atomic_paths._remove_pinned_directory_contents
    validated_aside = target.with_name("validated-owned-before-top-level-rmdir")
    sentinel = b"unknown bytes before top-level removal"
    swapped = False

    def cleanup_then_swap(descriptor, *, path):
        nonlocal swapped
        original_contents(descriptor, path=path)
        if not swapped:
            swapped = True
            path.rename(validated_aside)
            path.mkdir()
            (path / "unknown.bin").write_bytes(sentinel)

    monkeypatch.setattr(
        atomic_paths,
        "_remove_pinned_directory_contents",
        cleanup_then_swap,
    )

    with pytest.raises(PartialCleanupError) as caught:
        run_cli_workflow(
            SimpleNamespace(),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    residual = caught.value.residual_path
    assert (residual / "unknown.bin").read_bytes() == sentinel
    assert validate_owned_run_directory(target).run_directory == target


@pytest.mark.parametrize(
    "replacement_kind",
    [
        "missing",
        "symlink",
        "file",
        "fifo",
        "empty_directory",
        "nonempty_directory",
        "valid_owned_directory",
    ],
)
def test_cleanup_boundary_target_kind_attacks_preserve_unknown_residual(
    tmp_path,
    monkeypatch,
    replacement_kind,
):
    """Every non-identical cleanup target fails before child traversal."""
    workflow_module = importlib.import_module("radiosim.cli.workflow")
    atomic_paths = importlib.import_module("radiosim.io.atomic_paths")
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run", content=b"validated owned")
    workflow = _replacement_workflow(target)
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    original_cleanup = workflow_module.remove_directory_by_identity
    validated_aside = target.with_name(f"validated-owned-{replacement_kind}")
    unknown_target = target.with_name(f"unknown-{replacement_kind}")
    sentinel = f"unknown {replacement_kind} bytes".encode()
    cleanup_paths: list[Path] = []

    def swap_then_cleanup(path, parent_fd, **kwargs):
        cleanup_paths.append(path)
        path.rename(validated_aside)
        if replacement_kind == "missing":
            pass
        elif replacement_kind == "symlink":
            unknown_target.mkdir()
            (unknown_target / "unknown.bin").write_bytes(sentinel)
            path.symlink_to(unknown_target, target_is_directory=True)
        elif replacement_kind == "file":
            path.write_bytes(sentinel)
        elif replacement_kind == "fifo":
            os.mkfifo(path)
        elif replacement_kind == "empty_directory":
            path.mkdir()
        elif replacement_kind == "nonempty_directory":
            path.mkdir()
            (path / "unknown.bin").write_bytes(sentinel)
        else:
            _owned_run(path, content=sentinel)
        original_cleanup(path, parent_fd, **kwargs)

    monkeypatch.setattr(
        workflow_module,
        "remove_directory_by_identity",
        swap_then_cleanup,
    )
    monkeypatch.setattr(
        atomic_paths,
        "_remove_pinned_directory_contents",
        lambda *args, **kwargs: pytest.fail(
            "changed cleanup target reached child enumeration"
        ),
    )

    with pytest.raises(PartialCleanupError) as caught:
        run_cli_workflow(
            SimpleNamespace(),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    residual = caught.value.residual_path
    assert residual == cleanup_paths[0]
    assert (validated_aside / "old-result.h5").read_bytes() == b"validated owned"
    if replacement_kind == "missing":
        assert not residual.exists()
    elif replacement_kind == "symlink":
        assert residual.is_symlink()
        assert (unknown_target / "unknown.bin").read_bytes() == sentinel
    elif replacement_kind == "file":
        assert residual.read_bytes() == sentinel
    elif replacement_kind == "fifo":
        assert residual.exists()
    elif replacement_kind == "empty_directory":
        assert residual.is_dir() and not any(residual.iterdir())
    elif replacement_kind == "nonempty_directory":
        assert (residual / "unknown.bin").read_bytes() == sentinel
    else:
        assert validate_owned_run_directory(residual).run_directory == residual
        assert (residual / "old-result.h5").read_bytes() == sentinel
    assert validate_owned_run_directory(target).run_directory == target


def test_prepublication_staging_swap_preserves_unknown_residual(tmp_path, monkeypatch):
    """Writer rollback never recursively deletes a substituted staging path."""
    workflow_module = importlib.import_module("radiosim.cli.workflow")
    bundle = resolved_config(tmp_path)
    target = _owned_run(tmp_path / "runs" / "run", content=b"validated owned")
    workflow = CliWorkflowConfig(
        output_dir=target.parent,
        run_subdir=target.name,
        save_results=True,
        result_format="hdf5",
        collision_policy="replace",
    )
    plan = preflight_cli_workflow(workflow, runtime=bundle.runtime)
    original_cleanup = workflow_module.remove_directory_by_identity
    sentinel = b"unknown bytes replacing failed staging"
    validated_aside: list[Path] = []

    def fail_save(path, **kwargs):
        Path(path).with_suffix(".h5").write_bytes(b"partial")
        raise RuntimeError("injected writer failure")

    def swap_then_cleanup(path, parent_fd, **kwargs):
        aside = path.with_name(f"{path.name}.validated")
        validated_aside.append(aside)
        path.rename(aside)
        path.mkdir()
        (path / "unknown.bin").write_bytes(sentinel)
        original_cleanup(path, parent_fd, **kwargs)

    monkeypatch.setattr(
        workflow_module,
        "remove_directory_by_identity",
        swap_then_cleanup,
    )

    with pytest.raises(PartialCleanupError) as caught:
        run_cli_workflow(
            SimpleNamespace(save=fail_save),
            workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            plan=plan,
        )

    residual = caught.value.residual_path
    assert (residual / "unknown.bin").read_bytes() == sentinel
    assert validated_aside[0].is_dir()
    assert (target / "old-result.h5").read_bytes() == b"validated owned"


def test_pinned_cleanup_removes_nested_tree_without_following_symlink(tmp_path):
    """A legitimate tree is removed only through its verified descriptors."""
    atomic_paths = importlib.import_module("radiosim.io.atomic_paths")
    parent = tmp_path / "parent"
    candidate = parent / "candidate"
    nested = candidate / "nested" / "deeper"
    nested.mkdir(parents=True)
    (candidate / "top.bin").write_bytes(b"top")
    (nested / "deep.bin").write_bytes(b"deep")
    (candidate / "empty").mkdir()
    external = tmp_path / "external.bin"
    external.write_bytes(b"do not follow")
    (candidate / "external-link").symlink_to(external)
    for index in range(32):
        (candidate / f"small-{index:02d}.bin").write_bytes(b"x")

    parent_fd = atomic_paths.open_parent_directory(parent, create=False)
    try:
        identity = atomic_paths.capture_directory_identity(candidate, parent_fd)
        atomic_paths.remove_directory_by_identity(
            candidate,
            parent_fd,
            expected_st_dev=identity.st_dev,
            expected_st_ino=identity.st_ino,
            expected_file_type=identity.file_type,
        )
    finally:
        os.close(parent_fd)

    assert not candidate.exists()
    assert external.read_bytes() == b"do not follow"


def test_pinned_cleanup_rejects_special_nested_entry(tmp_path):
    """Unexpected special entries are left in a recoverable residual tree."""
    atomic_paths = importlib.import_module("radiosim.io.atomic_paths")
    parent = tmp_path / "parent"
    candidate = parent / "candidate"
    candidate.mkdir(parents=True)
    fifo = candidate / "unsafe.fifo"
    os.mkfifo(fifo)

    parent_fd = atomic_paths.open_parent_directory(parent, create=False)
    try:
        identity = atomic_paths.capture_directory_identity(candidate, parent_fd)
        with pytest.raises(AtomicWriteError, match="unsafe entry"):
            atomic_paths.remove_directory_by_identity(
                candidate,
                parent_fd,
                expected_st_dev=identity.st_dev,
                expected_st_ino=identity.st_ino,
                expected_file_type=identity.file_type,
            )
    finally:
        os.close(parent_fd)

    assert fifo.exists()


def test_pinned_cleanup_fails_closed_without_no_follow_support(tmp_path, monkeypatch):
    """Platforms without no-follow descriptor semantics never fall back."""
    atomic_paths = importlib.import_module("radiosim.io.atomic_paths")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    parent_fd = atomic_paths.open_parent_directory(tmp_path, create=False)
    try:
        monkeypatch.setattr(atomic_paths, "_NOFOLLOW", 0)
        with pytest.raises(AtomicWriteUnsupportedError, match="no-follow"):
            atomic_paths.capture_directory_identity(candidate, parent_fd)
    finally:
        os.close(parent_fd)
