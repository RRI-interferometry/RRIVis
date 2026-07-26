"""Resolved CLI error rendering, validation summaries, and output workflow."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from radiosim.core.precision import PrecisionConfig
from radiosim.io.config_resolution import ConfigResolutionError

if TYPE_CHECKING:
    from radiosim.api.simulator import Simulator
    from radiosim.core.runtime_config import (
        ConfigurationProvenance,
        ResolvedConfiguration,
        ResolvedSimulationConfig,
    )
    from radiosim.io.config import CliWorkflowConfig


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
    """A requested CLI result workflow is unavailable in the current slice."""


def ensure_result_workflow_available(
    *,
    save_results: bool,
    plot_results: bool,
) -> None:
    """Reject unavailable Tier 4C result work before runtime construction."""
    requested: list[str] = []
    if save_results:
        requested.append("saving")
    if plot_results:
        requested.append("plotting")
    if requested:
        subject = "result " + " and ".join(requested)
        verb = "are" if len(requested) > 1 else "is"
        raise WorkflowOutputError(
            f"{subject} {verb} temporarily unavailable in Tier 4C; saving requires the "
            "planned output workflow and plotting requires the canonical result "
            "renderer"
        )


def render_workflow_error(error: WorkflowOutputError, *, command: str) -> None:
    """Render a normal unsupported intermediate workflow without traceback."""
    click.echo(f"Workflow unavailable for {command} — {error}")


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


def run_cli_workflow(
    simulator: Simulator,
    workflow: CliWorkflowConfig,
    *,
    runtime: ResolvedSimulationConfig,
    provenance: ConfigurationProvenance,
    verbose: int = 0,
) -> tuple[Path, ...]:
    """Execute resolved output policy after a successful simulation run."""
    ensure_result_workflow_available(
        save_results=workflow.save_results,
        plot_results=workflow.plot_results,
    )

    from radiosim.utils.logging import (
        print_info,
        print_warning,
        setup_logging,
    )

    any_output = workflow.save_results or workflow.plot_results or workflow.save_log
    if not any_output:
        return ()

    run_subdir = workflow.run_subdir or deterministic_run_subdir(runtime)
    output_dir = workflow.output_dir / run_subdir
    overwrite = workflow.overwrite

    if output_dir.exists() and any(output_dir.iterdir()):
        if workflow.skip_overwrite_confirmation:
            print_warning(
                "Output folder exists (confirmation skipped); existing files may "
                "be overwritten."
            )
        else:
            print_warning(f"Output folder already exists: {output_dir}")
            answer = (
                click.prompt(
                    "Overwrite existing files? [y/N]",
                    default="n",
                    show_default=False,
                )
                .strip()
                .lower()
            )
            if answer not in {"y", "yes"}:
                print_warning("Aborted. No files were modified.")
                return ()
            overwrite = True

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    if workflow.save_log:
        log_file = output_dir / "simulation.log"
        setup_logging(
            level=logging.DEBUG if verbose >= 2 else logging.INFO,
            log_file=str(log_file),
        )
        saved_files.append(log_file)
        print_info(f"Logging to: {log_file}")

    from radiosim.io.writers import save_config_yaml

    artifact_path = output_dir / "resolved-config.yaml"
    save_config_yaml(
        _artifact_mapping(runtime, workflow, provenance),
        artifact_path,
    )
    saved_files.append(artifact_path)

    if workflow.save_results:
        data_path = simulator.save(
            output_dir,
            format=workflow.result_format,
            overwrite=overwrite,
            filename=workflow.result_filename,
        )
        if data_path:
            saved_files.append(Path(data_path))

    if workflow.plot_results:
        plot_paths = simulator.plot(
            plot_type="all",
            output_dir=output_dir,
            backend=workflow.plotting_backend,
            show=workflow.open_plots_in_browser,
            overwrite=overwrite,
        )
        saved_files.extend(Path(path) for path in plot_paths or ())

    return tuple(saved_files)


__all__ = [
    "deterministic_run_subdir",
    "ensure_result_workflow_available",
    "render_configuration_error",
    "render_resolved_summary",
    "render_workflow_error",
    "run_cli_workflow",
    "WorkflowOutputError",
]
