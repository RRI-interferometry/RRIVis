"""Resolved CLI error rendering, validation summaries, and output workflow."""

from __future__ import annotations

import logging
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from radiosim.core.precision import PrecisionConfig
from radiosim.io.atomic_paths import (
    capture_directory_identity,
    create_sibling_temporary_directory,
    exchange_directories,
    fsync_directory,
    open_parent_directory,
    publish_directory_no_clobber,
    remove_directory_by_identity,
)
from radiosim.io.config_resolution import ConfigResolutionError
from radiosim.io.result_errors import (
    AtomicWriteError,
    OutputCollisionError,
    OverwriteRefusedError,
    PartialCleanupError,
    ResultIOError,
    UnsafeOutputDirectoryError,
)
from radiosim.io.result_format import ResultFormat, require_result_dependencies
from radiosim.io.workflow_artifacts import (
    validate_owned_run_directory,
    write_resolved_config_artifact,
    write_workflow_manifest,
)

if TYPE_CHECKING:
    from radiosim.api.simulator import Simulator
    from radiosim.core.runtime_config import (
        ConfigurationProvenance,
        ResolvedConfiguration,
        ResolvedSimulationConfig,
    )
    from radiosim.io.config import CliWorkflowConfig


logger = logging.getLogger(__name__)

_ERROR_LABELS = {
    "ConfigSourceError": "source",
    "ConfigParseError": "parse",
    "ConfigSchemaError": "schema",
    "ConfigOverrideError": "override",
    "ConfigSemanticError": "semantic",
    "UnsupportedConfigError": "unsupported feature",
    "ConfigPathError": "path",
    "ConfigResolutionError": "resolution",
}


class WorkflowOutputError(RuntimeError):
    """A requested CLI result workflow violates the orchestration contract."""


class NonInteractivePromptError(WorkflowOutputError):
    """Prompt collision policy was requested without an interactive TTY."""


@dataclass(frozen=True, slots=True)
class _ReplacementDirectoryIdentity:
    """Exact directory identity and content contract approved for replacement."""

    st_dev: int
    st_ino: int
    file_type: int
    content_contract: str


@dataclass(frozen=True, slots=True)
class WorkflowExecutionPlan:
    """Read-only preflight decision consumed only after simulation succeeds."""

    enabled: bool
    declined: bool
    target: Path | None
    publish_mode: str | None
    _replacement_identity: _ReplacementDirectoryIdentity | None = None

    def validate_for_execution(self) -> None:
        """Reject internally inconsistent plans before workflow mutation."""
        if self.publish_mode == "exchange":
            if type(self._replacement_identity) is not _ReplacementDirectoryIdentity:
                raise WorkflowOutputError(
                    "workflow exchange plan lacks a valid replacement identity"
                )
        elif self._replacement_identity is not None:
            raise WorkflowOutputError(
                "workflow non-exchange plan unexpectedly contains a "
                "replacement identity"
            )

    def revalidate_replacement(self, path: Path, *, phase: str) -> None:
        """Revalidate the private replacement identity without exposing it."""
        identity = self._replacement_identity
        if type(identity) is not _ReplacementDirectoryIdentity:
            raise WorkflowOutputError(
                "workflow exchange plan lacks a valid replacement identity"
            )
        _revalidate_replacement_directory(path, identity, phase=phase)

    def remove_replacement(self, path: Path, parent_fd: int) -> None:
        """Remove the exchange residual through its captured identity only."""
        identity = self._replacement_identity
        if type(identity) is not _ReplacementDirectoryIdentity:
            raise WorkflowOutputError(
                "workflow exchange plan lacks a valid replacement identity"
            )
        remove_directory_by_identity(
            path,
            parent_fd,
            expected_st_dev=identity.st_dev,
            expected_st_ino=identity.st_ino,
            expected_file_type=identity.file_type,
        )


def render_workflow_error(error: RuntimeError, *, command: str) -> None:
    """Render one typed workflow failure without traceback."""
    click.echo(f"Workflow failed for {command} — {error}")


def render_configuration_error(
    error: ConfigResolutionError,
    *,
    command: str,
) -> None:
    """Render every production issue in stable resolver order."""
    category = _ERROR_LABELS.get(type(error).__name__, "resolution")
    click.echo(
        f"Configuration invalid for {command} ({category}) — "
        f"{len(error.issues)} issue(s)"
    )
    for index, issue in enumerate(error.issues, 1):
        click.echo(f"  [{index}] {issue.render()}")


def _precision_summary(config: PrecisionConfig) -> str:
    for name in ("standard", "fast", "precise", "ultra"):
        if config == getattr(PrecisionConfig, name)():
            return name
    return (
        f"custom (default={config.default}, accumulation={config.accumulation}, "
        f"output={config.output})"
    )


def _format_hz(value: float) -> str:
    return format(value, ".15g")


def render_resolved_summary(bundle: ResolvedConfiguration) -> None:
    """Print a useful validation summary without runtime construction."""
    source = bundle.provenance.source
    frequencies = bundle.runtime.frequency.channel_frequencies_hz
    scientific_paths = sum(
        not key.startswith(("workflow.", "configuration_source."))
        for key in bundle.provenance.path_resolutions
    )
    click.echo("Configuration is valid")
    click.echo(f"  Source: {source.label}")
    click.echo(f"  Resolved config path: {source.config_path}")
    click.echo(f"  Document base: {source.document_base}")
    click.echo(f"  Backend strategy: {bundle.runtime.execution.backend_strategy}")
    click.echo(f"  Precision: {_precision_summary(bundle.runtime.execution.precision)}")
    click.echo(f"  Frequency channels: {len(frequencies)}")
    click.echo(f"  Frequency minimum (Hz): {_format_hz(min(frequencies))}")
    click.echo(f"  Frequency maximum (Hz): {_format_hz(max(frequencies))}")
    click.echo(f"  Scientific input paths: {scientific_paths}")


def _safe_fragment(value: str) -> str:
    fragment = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return fragment or "simulation"


def deterministic_run_subdir(runtime: ResolvedSimulationConfig) -> str:
    """Derive one safe stable run name from resolved scientific state."""
    frequencies = runtime.frequency.channel_frequencies_hz
    start = _safe_fragment(runtime.observation.start_time_iso)
    source = runtime.instrument.source
    telescope = _safe_fragment(
        source.name
        if source.kind == "known_telescope"
        else (source.telescope_name or "simulation")
    )
    return (
        f"{telescope}_{_format_hz(frequencies[0])}-"
        f"{_format_hz(frequencies[-1])}Hz_{len(frequencies)}channels_"
        f"{start}_{_format_hz(runtime.observation.duration_seconds)}s"
    )


def _artifact_mapping(
    runtime: ResolvedSimulationConfig,
    workflow: CliWorkflowConfig,
    provenance: ConfigurationProvenance,
) -> dict[str, Any]:
    """Create a newly owned YAML-safe reproducibility artifact."""
    return {
        "schema_version": 1,
        "scientific_runtime": runtime.to_json_safe(),
        "workflow": workflow.model_dump(mode="json"),
        "provenance": provenance.to_json_safe(),
    }


def _path_status(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise UnsafeOutputDirectoryError(
            f"could not inspect workflow output target: {path}"
        ) from exc


def _validate_target_ancestors(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:-1]:
        current = current / component
        status = _path_status(current)
        if status is None:
            continue
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise UnsafeOutputDirectoryError(
                f"workflow output ancestor is unsafe: {current}"
            )


def _is_empty_directory(path: Path) -> bool:
    status = _path_status(path)
    if status is None:
        return False
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        return False
    try:
        return next(path.iterdir(), None) is None
    except OSError as exc:
        raise UnsafeOutputDirectoryError(
            f"could not inspect workflow output directory: {path}"
        ) from exc


def _replacement_identity(
    path: Path,
    status: os.stat_result,
    *,
    content_contract: str,
) -> _ReplacementDirectoryIdentity:
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise UnsafeOutputDirectoryError(
            f"workflow replacement target is not a safe directory: {path}"
        )
    if content_contract not in {"empty", "owned"}:
        raise WorkflowOutputError(
            f"unknown workflow replacement content contract: {content_contract}"
        )
    return _ReplacementDirectoryIdentity(
        st_dev=status.st_dev,
        st_ino=status.st_ino,
        file_type=stat.S_IFMT(status.st_mode),
        content_contract=content_contract,
    )


def _require_replacement_identity(
    path: Path,
    expected: _ReplacementDirectoryIdentity,
    *,
    phase: str,
) -> os.stat_result:
    status = _path_status(path)
    if status is None:
        raise OutputCollisionError(
            f"workflow replacement target disappeared {phase}: {path}"
        )
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise OutputCollisionError(
            f"workflow replacement target changed kind {phase}: {path}"
        )
    observed = (
        status.st_dev,
        status.st_ino,
        stat.S_IFMT(status.st_mode),
    )
    authorized = (
        expected.st_dev,
        expected.st_ino,
        expected.file_type,
    )
    if observed != authorized:
        raise OutputCollisionError(
            f"workflow replacement target changed identity {phase}: {path}"
        )
    return status


def _revalidate_replacement_directory(
    path: Path,
    expected: _ReplacementDirectoryIdentity,
    *,
    phase: str,
) -> None:
    _ = _require_replacement_identity(path, expected, phase=phase)
    if expected.content_contract == "empty":
        if not _is_empty_directory(path):
            raise UnsafeOutputDirectoryError(
                f"workflow replacement target is no longer empty {phase}: {path}"
            )
    elif expected.content_contract == "owned":
        _ = validate_owned_run_directory(path)
    else:
        raise WorkflowOutputError(
            "workflow replacement identity has an invalid content contract"
        )
    _ = _require_replacement_identity(path, expected, phase=phase)


def _fsync_published_parent_or_note(
    parent_fd: int,
    error: PartialCleanupError,
) -> None:
    try:
        fsync_directory(parent_fd)
    except Exception as sync_error:
        error.add_note(f"published parent fsync failure: {sync_error!r}")


def _validate_format_preflight(
    workflow: CliWorkflowConfig,
    runtime: ResolvedSimulationConfig,
) -> None:
    if not workflow.save_results:
        return
    result_format = workflow.result_format
    if type(result_format) is not ResultFormat:
        raise TypeError("workflow.result_format must resolve to ResultFormat")
    require_result_dependencies(result_format)
    output_precision = runtime.execution.precision.output
    if result_format in {ResultFormat.MS, ResultFormat.UVFITS} and str(
        output_precision
    ) in {"complex256", "float128"}:
        raise WorkflowOutputError(
            f"{result_format.value} cannot represent the configured output precision"
        )


def _validate_plot_preflight(workflow: CliWorkflowConfig) -> None:
    """Reject an unrenderable plot request before any directory or run work."""
    if not workflow.plot_results:
        return
    from radiosim.visualization.errors import ResultPlotContractError

    if workflow.plotting_backend != "bokeh":
        raise ResultPlotContractError(
            "only the bokeh result renderer is implemented; "
            f"workflow.plotting_backend={workflow.plotting_backend!r}"
        )
    if workflow.visibility_phase_unit not in ("radians", "degrees"):
        raise ResultPlotContractError(
            "workflow.visibility_phase_unit must be 'radians' or 'degrees'; "
            f"received {workflow.visibility_phase_unit!r}"
        )


def _open_published_plots(paths: tuple[Path, ...]) -> None:
    """Open published plots last; a browser failure never unpublishes data."""
    import webbrowser

    from radiosim.visualization.errors import ResultBrowserError

    for path in paths:
        try:
            webbrowser.open(path.as_uri())
        except Exception as exc:
            failure = ResultBrowserError(
                f"published plot could not be opened in a browser: {path}"
            )
            failure.__cause__ = exc
            logger.error(
                f"{type(failure).__name__}: {failure} "
                f"(published output is unaffected; cause {exc!r})"
            )


def _suffix_target(base: Path) -> Path:
    for index in range(1, 1000):
        candidate = base.with_name(f"{base.name}-{index:03d}")
        if _path_status(candidate) is None:
            return candidate
    raise OutputCollisionError(
        f"workflow suffix policy exhausted candidates -001 through -999: {base}"
    )


def preflight_cli_workflow(
    workflow: CliWorkflowConfig,
    *,
    runtime: ResolvedSimulationConfig,
) -> WorkflowExecutionPlan:
    """Resolve collision policy and prompt outcome without filesystem mutation."""
    _validate_plot_preflight(workflow)
    any_output = workflow.save_results or workflow.plot_results or workflow.save_log
    if not any_output:
        return WorkflowExecutionPlan(False, False, None, None)
    _validate_format_preflight(workflow, runtime)
    run_subdir = workflow.run_subdir or deterministic_run_subdir(runtime)
    target = Path(os.path.abspath(os.path.normpath(workflow.output_dir / run_subdir)))
    _validate_target_ancestors(target)
    status = _path_status(target)
    if status is None:
        return WorkflowExecutionPlan(True, False, target, "no_clobber")
    if _is_empty_directory(target):
        identity = _replacement_identity(
            target,
            status,
            content_contract="empty",
        )
        _revalidate_replacement_directory(
            target,
            identity,
            phase="during preflight",
        )
        return WorkflowExecutionPlan(
            True,
            False,
            target,
            "exchange",
            identity,
        )
    if workflow.collision_policy == "suffix":
        return WorkflowExecutionPlan(
            True,
            False,
            _suffix_target(target),
            "no_clobber",
        )
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise UnsafeOutputDirectoryError(
            f"workflow output target is not a safe directory: {target}"
        )
    try:
        _ = validate_owned_run_directory(target)
    except UnsafeOutputDirectoryError:
        raise
    identity = _replacement_identity(
        target,
        status,
        content_contract="owned",
    )
    _revalidate_replacement_directory(
        target,
        identity,
        phase="during preflight",
    )
    if workflow.collision_policy == "error":
        raise OverwriteRefusedError(f"owned workflow run already exists: {target}")
    if workflow.collision_policy == "prompt":
        stream = click.get_text_stream("stdin")
        if not stream.isatty():
            raise NonInteractivePromptError(
                f"collision policy prompt requires a TTY for owned run: {target}"
            )
        if not click.confirm(
            f"Replace owned workflow run {target}?",
            default=False,
            show_default=True,
        ):
            return WorkflowExecutionPlan(False, True, target, None)
    return WorkflowExecutionPlan(
        True,
        False,
        target,
        "exchange",
        identity,
    )


def _close_file_handler(handler: logging.FileHandler | None) -> None:
    if handler is None:
        return
    root = logging.getLogger()
    root.removeHandler(handler)
    try:
        handler.flush()
        handler.close()
    except Exception as exc:
        raise AtomicWriteError("workflow file logger failed to flush or close") from exc


def _fsync_staged_tree(staging: Path) -> None:
    for path in sorted(staging.rglob("*"), key=lambda item: item.as_posix()):
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode):
            raise UnsafeOutputDirectoryError(
                f"staged workflow contains a symbolic link: {path}"
            )
        if stat.S_ISREG(status.st_mode):
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif not stat.S_ISDIR(status.st_mode):
            raise UnsafeOutputDirectoryError(
                f"staged workflow contains a special file: {path}"
            )
    for directory in sorted(
        (path for path in staging.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        descriptor = open_parent_directory(directory, create=False)
        try:
            fsync_directory(descriptor)
        finally:
            os.close(descriptor)
    descriptor = open_parent_directory(staging, create=False)
    try:
        fsync_directory(descriptor)
    finally:
        os.close(descriptor)


def run_cli_workflow(
    simulator: Simulator,
    workflow: CliWorkflowConfig,
    *,
    runtime: ResolvedSimulationConfig,
    provenance: ConfigurationProvenance,
    verbose: int = 0,
    plan: WorkflowExecutionPlan | None = None,
) -> tuple[Path, ...]:
    """Build, verify, and atomically publish one complete owned run."""
    execution_plan = plan or preflight_cli_workflow(workflow, runtime=runtime)
    if execution_plan.declined or not execution_plan.enabled:
        return ()
    if execution_plan.target is None or execution_plan.publish_mode is None:
        raise WorkflowOutputError("workflow preflight produced an incomplete plan")
    execution_plan.validate_for_execution()

    from radiosim.utils.logging import (
        print_info,
    )

    target = execution_plan.target
    parent_fd: int | None = None
    staging: Path | None = None
    staging_identity: tuple[int, int, int] | None = None
    handler: logging.FileHandler | None = None
    published = False
    artifact_names: list[str] = []
    plot_names: list[str] = []
    try:
        parent_fd = open_parent_directory(target.parent, create=True)
        staging = create_sibling_temporary_directory(target, parent_fd)
        captured_staging = capture_directory_identity(staging, parent_fd)
        staging_identity = (
            captured_staging.st_dev,
            captured_staging.st_ino,
            captured_staging.file_type,
        )
        if workflow.save_log:
            log_path = staging / "simulation.log"
            try:
                handler = logging.FileHandler(log_path, encoding="utf-8", delay=False)
                handler.setLevel(logging.DEBUG if verbose >= 2 else logging.INFO)
                logging.getLogger().addHandler(handler)
            except Exception as exc:
                raise AtomicWriteError(
                    "workflow file logger could not be initialized in staging"
                ) from exc
            artifact_names.append(log_path.name)
            print_info(f"Logging to staged run for: {target / log_path.name}")
        config_path = write_resolved_config_artifact(
            _artifact_mapping(runtime, workflow, provenance),
            staging / "resolved-config.yaml",
        )
        artifact_names.append(config_path.name)
        if workflow.save_results:
            result_path = simulator.save(
                staging / workflow.result_filename,
                format=workflow.result_format,
                overwrite=False,
            )
            artifact_names.append(result_path.name)
        if workflow.plot_results:
            rendered = simulator.plot(
                plot_type="all",
                output_dir=staging,
                backend=workflow.plotting_backend,
                show=False,
                overwrite=False,
                visibility_phase_unit=workflow.visibility_phase_unit,
            )
            if type(rendered) is not tuple or not rendered:
                raise WorkflowOutputError(
                    "the result renderer declared no staged workflow plot files"
                )
            for candidate in rendered:
                plot_path = Path(candidate)
                if plot_path.parent != staging:
                    raise WorkflowOutputError(
                        "the result renderer wrote outside the staged run "
                        f"directory: {plot_path}"
                    )
                artifact_names.append(plot_path.name)
                plot_names.append(plot_path.name)
        _close_file_handler(handler)
        handler = None
        artifact_paths = tuple(staging / name for name in artifact_names)
        _ = write_workflow_manifest(staging, artifact_paths)
        _fsync_staged_tree(staging)
        if execution_plan.publish_mode == "no_clobber":
            publish_directory_no_clobber(staging, target, parent_fd)
            staging = None
        elif execution_plan.publish_mode == "exchange":
            execution_plan.revalidate_replacement(
                target,
                phase="before exchange",
            )
            exchange_directories(staging, target, parent_fd)
            published = True
            old_run = staging
            try:
                execution_plan.revalidate_replacement(
                    old_run,
                    phase="after exchange",
                )
            except Exception as exc:
                residual_error = PartialCleanupError(old_run)
                _fsync_published_parent_or_note(parent_fd, residual_error)
                staging = None
                raise residual_error from exc
            try:
                execution_plan.remove_replacement(old_run, parent_fd)
            except Exception as exc:
                residual_error = PartialCleanupError(old_run)
                _fsync_published_parent_or_note(parent_fd, residual_error)
                staging = None
                raise residual_error from exc
            staging = None
        else:
            raise WorkflowOutputError(
                f"unknown workflow publication mode: {execution_plan.publish_mode}"
            )
        published = True
        fsync_directory(parent_fd)
        final_names = ["manifest.json", *artifact_names]
        final_paths = tuple(target / name for name in final_names)
        if plot_names and workflow.open_plots_in_browser:
            _open_published_plots(tuple(target / name for name in plot_names))
        return final_paths
    except PartialCleanupError:
        raise
    except Exception as exc:
        try:
            _close_file_handler(handler)
        except Exception as close_error:
            exc.add_note(f"logger cleanup failure: {close_error!r}")
        if staging is not None and staging_identity is not None:
            assert parent_fd is not None
            try:
                remove_directory_by_identity(
                    staging,
                    parent_fd,
                    expected_st_dev=staging_identity[0],
                    expected_st_ino=staging_identity[1],
                    expected_file_type=staging_identity[2],
                )
            except Exception as cleanup_error:
                if _path_status(staging) is not None:
                    error = PartialCleanupError(staging)
                    error.add_note(f"cleanup failure: {cleanup_error!r}")
                    raise error from exc
        if isinstance(exc, (ResultIOError, WorkflowOutputError)):
            raise
        if published:
            raise AtomicWriteError(
                f"workflow run published with a post-publication error: {target}"
            ) from exc
        raise AtomicWriteError(
            f"workflow transaction failed before publication: {target}"
        ) from exc
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


__all__ = [
    "deterministic_run_subdir",
    "NonInteractivePromptError",
    "preflight_cli_workflow",
    "render_configuration_error",
    "render_resolved_summary",
    "render_workflow_error",
    "run_cli_workflow",
    "WorkflowExecutionPlan",
    "WorkflowOutputError",
]
