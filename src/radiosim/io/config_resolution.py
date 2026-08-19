"""Source-aware Tier 1 configuration loading and resolution.

This module transforms a strict frozen input document into one deeply
immutable runtime/workflow/provenance bundle.  Source normalization is shared
by YAML, mapping, and typed-model inputs.  Resolution performs no backend or
device construction, network work, scientific file reads, loader execution,
output creation, plotting, or browser interaction.
"""

from __future__ import annotations

import glob
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import ValidationError, field_validator

import radiosim.core.beam.models as resolved_beams
import radiosim.io.beam_config as beam_input
from radiosim.core.precision import PrecisionConfig
from radiosim.core.runtime_config import (
    ConfigurationProvenance,
    FrozenMapping,
    PathResolutionProvenance,
    ResolvedConfiguration,
    ResolvedExecutionConfig,
    ResolvedFrequencyConfig,
    ResolvedObservationConfig,
    ResolvedSimulationConfig,
    ResolvedSkyLoadingConfig,
    ResolvedSkyModelConfig,
    ResolvedSkySourceRequest,
    ResolvedSolverExecutionConfig,
    ValueOrigin,
    freeze_runtime_value,
    json_safe_mapping,
)
from radiosim.core.sky.registry import loader_registry
from radiosim.core.time_grid import build_observation_time_grid
from radiosim.io.config import (
    CliWorkflowConfig,
    ConfigIssue,
    CustomRegisteredSourceConfig,
    ExplicitFrequencyConfig,
    ObsFrequencyConfig,
    PrecisionInput,
    RadioSimConfig,
    RealisticForegroundSourceConfig,
    SkyLoadingConfig,
    SkySourceConfig,
    SolverExecutionConfig,
    StrictFrozenModel,
    collect_schema_issues,
    collect_semantic_issues,
    collect_unsupported_issues,
    schema_issues_from_validation_error,
)
from radiosim.io.instrument_config import (
    InstrumentLocationConfig,
    KnownTelescopeSourceConfig,
    LayoutFileSourceConfig,
)

SourceKind = Literal["yaml", "mapping", "model", "parameters"]
BackendStrategy = Literal["auto", "numpy", "jax", "dask"]
PrecisionPreset = Literal["standard", "fast", "precise", "ultra"]

_STAGE_ORDER = {
    "source": 0,
    "schema": 1,
    "override": 2,
    "semantic": 3,
    "unsupported": 4,
    "path": 5,
    "resolution": 6,
}
_ENVIRONMENT_PATH = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")
_FREQUENCY_UNIT_TO_HZ = {
    "Hz": 1.0,
    "kHz": 1e3,
    "MHz": 1e6,
    "GHz": 1e9,
}


def _order_issues(issues: Sequence[ConfigIssue]) -> tuple[ConfigIssue, ...]:
    return tuple(
        sorted(
            issues,
            key=lambda issue: (
                _STAGE_ORDER[issue.stage],
                issue.path,
                issue.code,
            ),
        )
    )


class ConfigResolutionError(ValueError):
    """Base class for every typed configuration-resolution failure."""

    def __init__(self, issues: Sequence[ConfigIssue]) -> None:
        ordered = _order_issues(issues)
        if not ordered:
            raise ValueError("configuration errors require at least one issue")
        self.issues = ordered
        super().__init__("\n".join(issue.render() for issue in ordered))


class ConfigSourceError(ConfigResolutionError):
    """Configuration source context is invalid or incomplete."""


class ConfigParseError(ConfigResolutionError):
    """Source acquisition or source-native normalization failed."""


class ConfigSchemaError(ConfigResolutionError):
    """Strict input-schema validation failed."""


class ConfigOverrideError(ConfigResolutionError):
    """An override object or replacement operation is invalid."""


class ConfigSemanticError(ConfigResolutionError):
    """Cross-field scientific or workflow validation failed."""


class UnsupportedConfigError(ConfigResolutionError):
    """The document requests a declared but later-tier feature."""


class ConfigPathError(ConfigResolutionError):
    """Path normalization, existence, type, or glob validation failed."""


def _source_issue(
    path: str,
    code: str,
    message: str,
    hint: str | None = None,
) -> ConfigIssue:
    return ConfigIssue(
        path,
        code,
        message,
        hint,
        stage="source",
        category="source",
    )


def _parse_issue(
    path: str,
    code: str,
    message: str,
    hint: str | None = None,
) -> ConfigIssue:
    return ConfigIssue(
        path,
        code,
        message,
        hint,
        stage="source",
        category="source",
    )


def _normalize_context_path(value: str | Path, invocation_dir: Path) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = invocation_dir / raw
    return raw.resolve(strict=False)


@dataclass(frozen=True, slots=True)
class ConfigurationSource:
    """Immutable source kind and the two bases used during resolution."""

    kind: SourceKind
    base_dir: Path | None = None
    config_path: Path | None = None
    invocation_dir: Path = field(default_factory=Path.cwd)
    label: str | None = None
    original_base_dir: str | None = field(init=False, default=None, repr=False)
    original_config_path: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        issues: list[ConfigIssue] = []
        if self.kind not in {"yaml", "mapping", "model", "parameters"}:
            issues.append(
                _source_issue(
                    "source.kind",
                    "invalid_source_kind",
                    f"unknown configuration source kind {self.kind!r}",
                    "Use 'yaml', 'mapping', 'model', or 'parameters'.",
                )
            )
        invocation = Path(self.invocation_dir).expanduser()
        if not invocation.is_absolute():
            invocation = (Path.cwd() / invocation).resolve(strict=False)
        else:
            invocation = invocation.resolve(strict=False)
        if not invocation.exists() or not invocation.is_dir():
            issues.append(
                _source_issue(
                    "source.invocation_dir",
                    "invalid_invocation_directory",
                    f"invocation_dir is not an existing directory: {invocation}",
                )
            )
        object.__setattr__(self, "invocation_dir", invocation)

        original_base = None if self.base_dir is None else str(self.base_dir)
        original_config = None if self.config_path is None else str(self.config_path)
        for path, original in (
            ("source.base_dir", original_base),
            ("source.config_path", original_config),
        ):
            if original is not None and _ENVIRONMENT_PATH.search(original):
                issues.append(
                    _source_issue(
                        path,
                        "environment_path_syntax",
                        "environment-variable syntax is not allowed in source paths",
                        "Expand the variable before creating ConfigurationSource.",
                    )
                )
        object.__setattr__(self, "original_base_dir", original_base)
        object.__setattr__(self, "original_config_path", original_config)

        normalized_config: Path | None = None
        if self.config_path is not None:
            normalized_config = _normalize_context_path(
                self.config_path,
                invocation,
            )
        normalized_base: Path | None = None
        if self.base_dir is not None:
            normalized_base = _normalize_context_path(self.base_dir, invocation)

        if self.kind == "yaml":
            if normalized_config is None:
                issues.append(
                    _source_issue(
                        "source.config_path",
                        "missing_yaml_config_path",
                        "YAML sources require config_path",
                    )
                )
            else:
                derived_base = normalized_config.parent
                if normalized_base is not None and normalized_base != derived_base:
                    issues.append(
                        _source_issue(
                            "source.base_dir",
                            "yaml_base_mismatch",
                            "YAML base_dir must be the resolved parent of config_path",
                            f"Use {derived_base} or omit base_dir.",
                        )
                    )
                normalized_base = derived_base
                if not normalized_config.exists() or not normalized_config.is_file():
                    issues.append(
                        _source_issue(
                            "source.config_path",
                            "invalid_yaml_config_path",
                            "config_path must be an existing regular file",
                        )
                    )
        elif normalized_config is not None:
            issues.append(
                _source_issue(
                    "source.config_path",
                    "config_path_not_allowed",
                    f"config_path is not valid for source kind {self.kind!r}",
                )
            )

        object.__setattr__(self, "config_path", normalized_config)
        object.__setattr__(self, "base_dir", normalized_base)
        label = self.label
        if label is None:
            subject = normalized_config or normalized_base or invocation
            label = f"{self.kind}:{subject}"
        elif not label.strip():
            issues.append(
                _source_issue(
                    "source.label",
                    "empty_source_label",
                    "source label must be nonempty",
                )
            )
        object.__setattr__(self, "label", label.strip())
        if issues:
            raise ConfigSourceError(issues)

    @property
    def document_base(self) -> Path | None:
        if self.kind == "parameters":
            return self.base_dir or self.invocation_dir
        return self.base_dir

    @classmethod
    def for_yaml(
        cls,
        config_path: str | Path,
        *,
        invocation_dir: str | Path | None = None,
        label: str | None = None,
    ) -> ConfigurationSource:
        return cls(
            kind="yaml",
            config_path=Path(config_path),
            invocation_dir=(
                Path.cwd() if invocation_dir is None else Path(invocation_dir)
            ),
            label=label,
        )

    @classmethod
    def for_mapping(
        cls,
        *,
        base_dir: str | Path | None = None,
        invocation_dir: str | Path | None = None,
        label: str | None = None,
    ) -> ConfigurationSource:
        return cls(
            kind="mapping",
            base_dir=None if base_dir is None else Path(base_dir),
            invocation_dir=(
                Path.cwd() if invocation_dir is None else Path(invocation_dir)
            ),
            label=label,
        )

    @classmethod
    def for_model(
        cls,
        *,
        base_dir: str | Path | None = None,
        invocation_dir: str | Path | None = None,
        label: str | None = None,
    ) -> ConfigurationSource:
        return cls(
            kind="model",
            base_dir=None if base_dir is None else Path(base_dir),
            invocation_dir=(
                Path.cwd() if invocation_dir is None else Path(invocation_dir)
            ),
            label=label,
        )

    @classmethod
    def for_parameters(
        cls,
        *,
        base_dir: str | Path | None = None,
        invocation_dir: str | Path | None = None,
        label: str | None = None,
    ) -> ConfigurationSource:
        return cls(
            kind="parameters",
            base_dir=None if base_dir is None else Path(base_dir),
            invocation_dir=(
                Path.cwd() if invocation_dir is None else Path(invocation_dir)
            ),
            label=label,
        )


class InstrumentSourcePathOverride(StrictFrozenModel):
    """Path-only replacement for an existing layout-file instrument source."""

    path: Path

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("path must be nonempty")
        return value


class SimulationOverrides(StrictFrozenModel):
    """Explicit complete-value replacements for runtime-owned values."""

    backend: BackendStrategy | None = None
    precision: PrecisionInput | PrecisionConfig | PrecisionPreset | None = None
    offline: bool | None = None
    instrument_source: InstrumentSourcePathOverride | None = None
    obs_frequency: ObsFrequencyConfig | None = None
    location: InstrumentLocationConfig | None = None
    start_time: str | None = None
    simulator: Literal["rime"] | None = None

    @field_validator("start_time")
    @classmethod
    def validate_start_time(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("start_time must be nonempty")
        return stripped


class WorkflowOverrides(StrictFrozenModel):
    """CLI-only path replacements that cannot enter runtime state."""

    output_dir: Path | None = None

    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("output_dir must be nonempty")
        return value


def _path_issue(
    path: str,
    code: str,
    message: str,
    hint: str | None = None,
) -> ConfigIssue:
    return ConfigIssue(
        path,
        code,
        message,
        hint,
        stage="path",
        category="path",
    )


class _PathResolver:
    def __init__(
        self,
        source: ConfigurationSource,
        *,
        check_input_paths: bool,
    ) -> None:
        self.source = source
        self.check_input_paths = check_input_paths
        self.source_issues: list[ConfigIssue] = []
        self.path_issues: list[ConfigIssue] = []
        self.records: dict[str, PathResolutionProvenance] = {}
        if source.config_path is not None:
            original = source.original_config_path or str(source.config_path)
            raw = Path(original).expanduser()
            user_path = raw if raw.is_absolute() else source.invocation_dir / raw
            self.records["configuration_source.config_path"] = PathResolutionProvenance(
                logical_path="configuration_source.config_path",
                original=original,
                base=source.invocation_dir,
                user_path=Path(os.path.abspath(user_path)),
                resolved=source.config_path,
                origin="document",
                kind="file",
            )

    def _base_for(self, origin: ValueOrigin) -> Path | None:
        if origin == "override":
            return self.source.invocation_dir
        return self.source.document_base

    def _prepare(
        self,
        value: str | Path,
        *,
        logical_path: str,
        origin: ValueOrigin,
    ) -> tuple[str, Path, Path | None] | None:
        original = str(value)
        if _ENVIRONMENT_PATH.search(original):
            self.path_issues.append(
                _path_issue(
                    logical_path,
                    "environment_path_syntax",
                    "environment-variable syntax is not allowed in config paths",
                    "Expand the variable in the calling environment and pass the explicit path.",
                )
            )
            return None
        expanded = Path(original).expanduser()
        selected_base = self._base_for(origin)
        if not expanded.is_absolute():
            if selected_base is None:
                self.source_issues.append(
                    _source_issue(
                        logical_path,
                        "relative_path_requires_base_dir",
                        "relative document path requires an explicit source base_dir",
                        "Pass base_dir for mapping/model input or use an absolute path.",
                    )
                )
                return None
            expanded = selected_base / expanded
        user_path = Path(os.path.abspath(expanded))
        return original, user_path, selected_base

    def path(
        self,
        value: str | Path,
        *,
        logical_path: str,
        origin: ValueOrigin,
        expected: Literal["file", "directory", "output_directory"],
    ) -> Path:
        prepared = self._prepare(
            value,
            logical_path=logical_path,
            origin=origin,
        )
        if prepared is None:
            return Path(value)
        original, user_path, selected_base = prepared
        resolved = user_path.resolve(strict=False)
        self.records[logical_path] = PathResolutionProvenance(
            logical_path=logical_path,
            original=original,
            base=selected_base,
            user_path=user_path,
            resolved=resolved,
            origin=origin,
            kind=expected,
        )
        if expected == "output_directory":
            if resolved.exists() and not resolved.is_dir():
                self.path_issues.append(
                    _path_issue(
                        logical_path,
                        "output_path_wrong_type",
                        f"output_dir exists but is not a directory: {resolved}",
                    )
                )
            return resolved
        if not self.check_input_paths:
            return resolved
        if expected == "file":
            if not resolved.exists():
                self.path_issues.append(
                    _path_issue(
                        logical_path,
                        "input_path_missing",
                        f"required input file does not exist: {resolved}",
                    )
                )
            elif not resolved.is_file():
                self.path_issues.append(
                    _path_issue(
                        logical_path,
                        "input_path_wrong_type",
                        f"expected a regular file, got: {resolved}",
                    )
                )
            else:
                mode = resolved.stat().st_mode
                readable_bits = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
                try:
                    if mode & readable_bits == 0:
                        raise PermissionError("no read permission bits are set")
                    descriptor = os.open(resolved, os.O_RDONLY)
                    os.close(descriptor)
                except OSError as error:
                    self.path_issues.append(
                        _path_issue(
                            logical_path,
                            "input_path_unreadable",
                            f"required input file is not readable: {resolved} ({error})",
                        )
                    )
        elif expected == "directory":
            if not resolved.exists():
                self.path_issues.append(
                    _path_issue(
                        logical_path,
                        "input_path_missing",
                        f"required input directory does not exist: {resolved}",
                    )
                )
            elif not resolved.is_dir():
                self.path_issues.append(
                    _path_issue(
                        logical_path,
                        "input_path_wrong_type",
                        f"expected a directory, got: {resolved}",
                    )
                )
        return resolved

    def file_list(
        self,
        values: Sequence[str | Path],
        *,
        logical_path: str,
        origin: ValueOrigin,
    ) -> tuple[Path, ...]:
        return tuple(
            self.path(
                value,
                logical_path=f"{logical_path}[{index}]",
                origin=origin,
                expected="file",
            )
            for index, value in enumerate(values)
        )

    def glob(
        self,
        value: str | Path,
        *,
        logical_path: str,
        origin: ValueOrigin,
    ) -> tuple[Path, ...]:
        prepared = self._prepare(
            value,
            logical_path=logical_path,
            origin=origin,
        )
        if prepared is None:
            return ()
        original, user_path, selected_base = prepared
        matches = tuple(
            sorted(
                {
                    Path(match).resolve(strict=False)
                    for match in glob.glob(str(user_path))
                    if Path(match).is_file()
                },
                key=str,
            )
        )
        self.records[logical_path] = PathResolutionProvenance(
            logical_path=logical_path,
            original=original,
            base=selected_base,
            user_path=user_path,
            resolved=matches,
            origin=origin,
            kind="glob",
        )
        if self.check_input_paths and not matches:
            self.path_issues.append(
                _path_issue(
                    logical_path,
                    "glob_no_regular_files",
                    f"glob matched no regular files: {user_path}",
                )
            )
        return matches


def _document_origin(model: Any, field_name: str) -> ValueOrigin:
    return "document" if field_name in model.model_fields_set else "default"


_AUTO_LOADER_WORKER_CEILING = 8


def _resolve_sky_loading(
    config: SkyLoadingConfig,
    *,
    request_count: int,
) -> ResolvedSkyLoadingConfig:
    """Resolve `max_workers` to an integer without executing any loader.

    ``None`` means auto and becomes ``min(requests, cpu_count, 8)``: the auto
    ceiling preserves the pre-Tier-6 hard-coded pool size instead of changing
    performance while the value becomes configurable and recorded.
    """
    configured = config.max_workers
    if configured is not None:
        resolved = configured
    else:
        resolved = min(
            max(request_count, 1),
            os.cpu_count() or 1,
            _AUTO_LOADER_WORKER_CEILING,
        )
    return ResolvedSkyLoadingConfig(
        max_workers=max(resolved, 1),
        executor=config.executor,
    )


def _resolve_solver_execution(
    config: SolverExecutionConfig,
    *,
    time_sample_count: int,
) -> ResolvedSolverExecutionConfig:
    """Clamp solver workers to the time samples a partition can cover.

    The pre-clamp request stays visible in
    :attr:`ConfigurationProvenance.input_snapshot`, whose ``execution`` block is
    the validated input document, so no requested value is lost by the clamp.
    """
    return ResolvedSolverExecutionConfig(
        workers=min(config.workers, max(time_sample_count, 1)),
        executor=config.executor,
    )


def _precision_input(value: Any) -> PrecisionInput:
    if isinstance(value, PrecisionInput):
        if value.preset is not None and not value.has_preset_custom_contradiction:
            return PrecisionInput(preset=value.preset)
        return PrecisionInput.model_validate(value.model_dump(mode="python"))
    if isinstance(value, PrecisionConfig):
        return PrecisionInput.model_validate(value.model_dump(mode="python"))
    if isinstance(value, str):
        return PrecisionInput(preset=cast(PrecisionPreset, value))
    raise TypeError("precision override must be a complete precision value")


def _apply_overrides(
    config: RadioSimConfig,
    simulation: SimulationOverrides | None,
    workflow: WorkflowOverrides | None,
) -> tuple[RadioSimConfig, dict[str, ValueOrigin]]:
    origins: dict[str, ValueOrigin] = {
        "instrument.source.path": "document",
        "execution.backend": _document_origin(config.execution, "backend"),
        "execution.precision": _document_origin(config.execution, "precision"),
        "execution.offline": _document_origin(config.execution, "offline"),
        "execution.simulator": _document_origin(config.execution, "simulator"),
        "execution.sky_loading.max_workers": _document_origin(
            config.execution.sky_loading, "max_workers"
        ),
        "execution.sky_loading.executor": _document_origin(
            config.execution.sky_loading, "executor"
        ),
        "execution.solver.workers": _document_origin(
            config.execution.solver, "workers"
        ),
        "execution.solver.executor": _document_origin(
            config.execution.solver, "executor"
        ),
        "instrument.location": "document",
        "obs_frequency": "document",
        "obs_time.start_time": "document",
        "workflow.output_dir": _document_origin(config.workflow, "output_dir"),
    }
    simulation = simulation or SimulationOverrides()
    workflow = workflow or WorkflowOverrides()
    execution_updates: dict[str, Any] = {}
    if simulation.backend is not None:
        execution_updates["backend"] = simulation.backend
        origins["execution.backend"] = "override"
    if simulation.precision is not None:
        execution_updates["precision"] = _precision_input(simulation.precision)
        origins["execution.precision"] = "override"
    if simulation.offline is not None:
        execution_updates["offline"] = simulation.offline
        origins["execution.offline"] = "override"
    if simulation.simulator is not None:
        execution_updates["simulator"] = simulation.simulator
        origins["execution.simulator"] = "override"
    execution = config.execution.model_copy(update=execution_updates)

    instrument = config.instrument
    if simulation.instrument_source is not None:
        if isinstance(instrument.source, KnownTelescopeSourceConfig):
            raise ConfigOverrideError(
                [
                    ConfigIssue(
                        "overrides.instrument_source.path",
                        "layout_path_override_requires_layout_source",
                        "an instrument source path cannot replace a known-telescope source",
                        "Select a layout_file source in the document before overriding its path.",
                        stage="override",
                        category="override",
                    )
                ]
            )
        source = instrument.source.model_copy(
            update={"path": simulation.instrument_source.path}
        )
        instrument = instrument.model_copy(update={"source": source})
        origins["instrument.source.path"] = "override"

    obs_frequency = config.obs_frequency
    if simulation.obs_frequency is not None:
        obs_frequency = simulation.obs_frequency
        origins["obs_frequency"] = "override"
    if simulation.location is not None:
        instrument = instrument.model_copy(update={"location": simulation.location})
        origins["instrument.location"] = "override"
    obs_time = config.obs_time
    if simulation.start_time is not None:
        obs_time = obs_time.model_copy(update={"start_time": simulation.start_time})
        origins["obs_time.start_time"] = "override"
    workflow_config = config.workflow
    if workflow.output_dir is not None:
        workflow_config = workflow_config.model_copy(
            update={"output_dir": workflow.output_dir}
        )
        origins["workflow.output_dir"] = "override"
    candidate = config.model_copy(
        update={
            "instrument": instrument,
            "execution": execution,
            "obs_frequency": obs_frequency,
            "obs_time": obs_time,
            "workflow": workflow_config,
        }
    )
    return candidate, origins


def _schema_model(
    config: object,
) -> RadioSimConfig:
    if isinstance(config, RadioSimConfig):
        return config
    if not isinstance(config, Mapping):
        raise ConfigSchemaError(
            [
                ConfigIssue(
                    "",
                    "invalid_config_input",
                    "config must be a RadioSimConfig or Python mapping",
                    stage="schema",
                    category="schema",
                )
            ]
        )
    try:
        mapping = cast(Mapping[str, object], config)
        issues = collect_schema_issues(mapping)
        if issues:
            raise ConfigSchemaError(issues)
        return RadioSimConfig.model_validate(dict(mapping))
    except ConfigSchemaError:
        raise
    except ValidationError as error:
        raise ConfigSchemaError(schema_issues_from_validation_error(error)) from error


def _resolve_frequency(config: ObsFrequencyConfig) -> ResolvedFrequencyConfig:
    if isinstance(config, ExplicitFrequencyConfig):
        return ResolvedFrequencyConfig(
            channel_frequencies_hz=config.channel_frequencies_hz,
            channel_widths_hz=config.channel_widths_hz,
            source_mode="explicit",
        )
    factor = _FREQUENCY_UNIT_TO_HZ[config.frequency_unit]
    start_hz = config.starting_frequency * factor
    interval_hz = config.frequency_interval * factor
    n_intervals = round(config.frequency_bandwidth / config.frequency_interval)
    channels = tuple(start_hz + index * interval_hz for index in range(n_intervals + 1))
    return ResolvedFrequencyConfig(
        channel_frequencies_hz=channels,
        channel_widths_hz=(config.channel_width * factor,) * len(channels),
        source_mode="grid",
    )


def _source_common_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return freeze_runtime_value(value.model_dump(mode="python"))
    return freeze_runtime_value(value)


def _resolve_path_options(
    options: dict[str, Any],
    *,
    definition: Any,
    path_prefix: str,
    resolver: _PathResolver,
) -> dict[str, Any]:
    resolved = dict(options)
    for loader_arg, path_kind in definition.path_options.items():
        value = resolved.get(loader_arg)
        if value is None:
            continue
        logical_path = f"{path_prefix}.{loader_arg}"
        if path_kind == "file":
            resolved[loader_arg] = resolver.path(
                value,
                logical_path=logical_path,
                origin="document",
                expected="file",
            )
        elif path_kind == "file_list":
            resolved[loader_arg] = resolver.file_list(
                cast(Sequence[str | Path], value),
                logical_path=logical_path,
                origin="document",
            )
        else:
            resolved[loader_arg] = resolver.glob(
                value,
                logical_path=logical_path,
                origin="document",
            )
    return resolved


def _resolve_nested_registered_options(
    kind: str,
    options: Mapping[str, Any] | None,
    *,
    path_prefix: str,
    resolver: _PathResolver,
) -> FrozenMapping | None:
    if options is None:
        return None
    definition = loader_registry.definition(kind)
    source_to_loader = {
        source_name: loader_arg
        for loader_arg, source_name in definition.config_fields.items()
    }
    loader_options = {source_to_loader[name]: value for name, value in options.items()}
    resolved = _resolve_path_options(
        loader_options,
        definition=definition,
        path_prefix=path_prefix,
        resolver=resolver,
    )
    return FrozenMapping(resolved)


def _resolve_sky_source(
    source: SkySourceConfig,
    *,
    index: int,
    resolver: _PathResolver,
) -> ResolvedSkySourceRequest:
    canonical, alias_defaults = loader_registry.resolve_request(source.kind, {})
    definition = loader_registry.definition(canonical)
    base_path = f"sky_model.sources[{index}]"
    if isinstance(source, CustomRegisteredSourceConfig):
        source_to_loader = {
            source_name: loader_arg
            for loader_arg, source_name in definition.config_fields.items()
        }
        explicit = {
            source_to_loader[name]: value for name, value in source.options.items()
        }
    else:
        dumped = source.model_dump(mode="python")
        explicit = {
            name: value
            for name, value in dumped.items()
            if name
            not in {
                "kind",
                "region",
                "brightness_conversion",
                "provenance_override",
            }
        }
    options = dict(alias_defaults)
    options.update(explicit)
    options = _resolve_path_options(
        options,
        definition=definition,
        path_prefix=base_path,
        resolver=resolver,
    )
    if isinstance(source, RealisticForegroundSourceConfig):
        diffuse_options = _resolve_nested_registered_options(
            source.diffuse,
            source.diffuse_kwargs,
            path_prefix=f"{base_path}.diffuse_kwargs",
            resolver=resolver,
        )
        catalog_options = _resolve_nested_registered_options(
            source.bright_catalogs,
            source.bright_catalog_kwargs,
            path_prefix=f"{base_path}.bright_catalog_kwargs",
            resolver=resolver,
        )
        if diffuse_options is not None:
            options["diffuse_kwargs"] = diffuse_options
        if catalog_options is not None:
            options["bright_catalog_kwargs"] = catalog_options
    return ResolvedSkySourceRequest(
        kind=canonical,
        options=FrozenMapping(options),
        region=_source_common_value(source.region),
        brightness_conversion=source.brightness_conversion,
        provenance_override=_source_common_value(source.provenance_override),
    )


def _resolve_direct_taper(
    taper: beam_input.DirectTaperConfig,
) -> resolved_beams.ResolvedDirectTaper:
    if isinstance(taper, beam_input.UniformTaperConfig):
        return resolved_beams.ResolvedUniformTaper("uniform")
    if isinstance(taper, beam_input.GaussianTaperConfig):
        return resolved_beams.ResolvedGaussianTaper("gaussian", taper.edge_taper_db)
    if isinstance(taper, beam_input.ParabolicTaperConfig):
        return resolved_beams.ResolvedParabolicTaper("parabolic", taper.edge_taper_db)
    if isinstance(taper, beam_input.ParabolicSquaredTaperConfig):
        return resolved_beams.ResolvedParabolicSquaredTaper(
            "parabolic_squared", taper.edge_taper_db
        )
    if isinstance(taper, beam_input.CosineTaperConfig):
        return resolved_beams.ResolvedCosineTaper("cosine")
    raise TypeError(f"unsupported direct taper type {type(taper).__name__}")


def _resolve_derived_taper(
    taper: beam_input.FeedDerivedTaperConfig,
) -> resolved_beams.ResolvedDerivedTaper:
    if isinstance(taper, beam_input.DerivedGaussianTaperConfig):
        return resolved_beams.ResolvedDerivedGaussianTaper("gaussian")
    if isinstance(taper, beam_input.DerivedParabolicTaperConfig):
        return resolved_beams.ResolvedDerivedParabolicTaper("parabolic")
    if isinstance(taper, beam_input.DerivedParabolicSquaredTaperConfig):
        return resolved_beams.ResolvedDerivedParabolicSquaredTaper("parabolic_squared")
    raise TypeError(f"unsupported derived taper type {type(taper).__name__}")


def _resolve_illumination(
    illumination: beam_input.IlluminationConfig,
) -> resolved_beams.ResolvedIllumination:
    if isinstance(illumination, beam_input.CorrugatedHornIlluminationConfig):
        return resolved_beams.ResolvedCorrugatedHornIllumination(
            "corrugated_horn",
            illumination.focal_ratio,
            illumination.q,
        )
    if isinstance(illumination, beam_input.OpenWaveguideIlluminationConfig):
        return resolved_beams.ResolvedOpenWaveguideIllumination(
            "open_waveguide",
            illumination.focal_ratio,
            illumination.b_over_lambda,
        )
    if isinstance(illumination, beam_input.DipoleGroundPlaneIlluminationConfig):
        return resolved_beams.ResolvedDipoleGroundPlaneIllumination(
            "dipole_ground_plane",
            illumination.focal_ratio,
            illumination.height_wavelengths,
        )
    raise TypeError(f"unsupported illumination type {type(illumination).__name__}")


def _resolve_reflector(
    reflector: beam_input.ReflectorConfig,
) -> resolved_beams.ResolvedReflector:
    if isinstance(reflector, beam_input.PrimeFocusReflectorConfig):
        return resolved_beams.ResolvedPrimeFocusReflector("prime_focus")
    if isinstance(reflector, beam_input.CassegrainReflectorConfig):
        return resolved_beams.ResolvedCassegrainReflector(
            "cassegrain", reflector.magnification
        )
    raise TypeError(f"unsupported reflector type {type(reflector).__name__}")


def _resolve_analytic_model(
    model: beam_input.AnalyticBeamModelConfig,
) -> resolved_beams.ResolvedAnalyticBeamModel:
    if isinstance(model, beam_input.CircularApertureBeamModelConfig):
        return resolved_beams.ResolvedCircularApertureBeamModel(
            "circular_aperture", _resolve_direct_taper(model.taper)
        )
    if isinstance(model, beam_input.RectangularApertureBeamModelConfig):
        return resolved_beams.ResolvedRectangularApertureBeamModel(
            "rectangular_aperture",
            model.north_length_m,
            model.east_length_m,
        )
    if isinstance(model, beam_input.EllipticalApertureBeamModelConfig):
        return resolved_beams.ResolvedEllipticalApertureBeamModel(
            "elliptical_aperture",
            model.north_diameter_m,
            model.east_diameter_m,
        )
    if isinstance(model, beam_input.AnalyticalIlluminationBeamModelConfig):
        return resolved_beams.ResolvedAnalyticalIlluminationBeamModel(
            "analytical_illumination",
            _resolve_illumination(model.illumination),
            _resolve_derived_taper(model.taper_profile),
            _resolve_reflector(model.reflector),
        )
    if isinstance(model, beam_input.NumericalIlluminationBeamModelConfig):
        return resolved_beams.ResolvedNumericalIlluminationBeamModel(
            "numerical_illumination",
            _resolve_illumination(model.illumination),
            _resolve_reflector(model.reflector),
            256,
        )
    raise TypeError(f"unsupported analytic beam model {type(model).__name__}")


def _resolve_analytic_definition(
    model: beam_input.AnalyticBeamModelConfig,
) -> resolved_beams.ResolvedAnalyticBeamDefinition:
    resolved_model = _resolve_analytic_model(model)
    return resolved_beams.ResolvedAnalyticBeamDefinition(
        "analytic",
        resolved_model,
        resolved_beams._definition_fingerprint("analytic", resolved_model),
    )


def _resolve_fits_definition(
    source: beam_input.FITSBeamSourceConfig,
    *,
    logical_path: str,
    resolver: _PathResolver,
) -> resolved_beams.ResolvedFITSBeamDefinition:
    path = resolver.path(
        source.path,
        logical_path=logical_path,
        origin="document",
        expected="file",
    )
    # A missing document base is already recorded as a source issue.  Keep the
    # temporary value constructible so the resolver can raise that typed issue.
    if not path.is_absolute():
        path = (resolver.source.invocation_dir / path).resolve(strict=False)
    fingerprint_payload = {
        "normalization": source.normalization,
        "angular_interpolation": source.angular_interpolation,
        "frequency_interpolation": source.frequency_interpolation,
    }
    return resolved_beams.ResolvedFITSBeamDefinition(
        "fits",
        path,
        source.normalization,
        source.angular_interpolation,
        source.frequency_interpolation,
        logical_path,
        resolved_beams._definition_fingerprint("fits", fingerprint_payload),
    )


def _resolve_pointing_offset(
    offset: beam_input.PointingOffsetConfig | beam_input.AntennaPointingOffsetConfig,
) -> resolved_beams.ResolvedPointingOffset | None:
    """Convert one authored offset to radians, or to ``None`` if it is inert.

    An exactly-zero offset resolves to absence rather than to a stored zero, so
    that a configuration authoring one is bit-identical -- cube, assignment
    fingerprint and ``scientific_sha256`` alike -- to one authoring nothing.
    """
    if offset.azimuth_offset_deg == 0.0 and offset.elevation_offset_deg == 0.0:
        return None
    return resolved_beams.ResolvedPointingOffset(
        math.radians(offset.azimuth_offset_deg),
        math.radians(offset.elevation_offset_deg),
    )


def _resolve_error_beam_diagnostic(
    diagnostic: beam_input.RuzeErrorBeamDiagnosticConfig | None,
) -> resolved_beams.ResolvedRuzePowerDiagnostic | None:
    """Convert one authored nested ensemble-power declaration."""
    if diagnostic is None:
        return None
    return resolved_beams.ResolvedRuzePowerDiagnostic(
        diagnostic.kind,
        diagnostic.correlation_length_m,
    )


def _resolve_surface_error(
    surface: beam_input.SurfaceErrorConfig | beam_input.AntennaSurfaceErrorConfig,
) -> resolved_beams.ResolvedSurfaceError | None:
    """Convert one authored surface RMS, or ``None`` if it is exactly zero."""
    if surface.rms_surface_error_m == 0.0:
        return None
    return resolved_beams.ResolvedSurfaceError(
        surface.rms_surface_error_m,
        _resolve_error_beam_diagnostic(surface.error_beam_diagnostic),
    )


def _resolve_aperture_physics(
    aperture: beam_input.AperturePhysicsConfig | None,
) -> resolved_beams.ResolvedAperturePhysics | None:
    """Resolve the authored ``beams.aperture_physics`` block.

    Domain, identity, and duplicate rejections have already been collected as
    ``ConfigSemanticError`` issues before resolution runs, so this conversion
    only has to carry exact authored values across and let the resolved
    dataclasses re-assert their own invariants.
    """
    if aperture is None:
        return None
    blockage = None
    if aperture.blockage is not None:
        blockage = resolved_beams.ResolvedApertureBlockage(
            aperture.blockage.central_diameter_ratio,
            tuple(
                resolved_beams.ResolvedSupportLeg(
                    leg.position_angle_deg,
                    leg.width_m,
                )
                for leg in aperture.blockage.support_legs
            ),
        )
    zernike = None
    if aperture.zernike_surface is not None:
        zernike = resolved_beams.ResolvedZernikeSurface(
            aperture.zernike_surface.convention,
            tuple(
                resolved_beams.ResolvedZernikeMode(
                    mode.n,
                    mode.m,
                    mode.surface_height_coefficient_m,
                )
                for mode in aperture.zernike_surface.modes
            ),
        )
    return resolved_beams.ResolvedAperturePhysics(
        aperture.normalization,
        blockage,
        zernike,
    )


def _resolve_beam_pointing(
    pointing: beam_input.BeamPointingConfig | None,
) -> resolved_beams.ResolvedBeamPointing | None:
    if pointing is None:
        return None
    return resolved_beams.ResolvedBeamPointing(
        None
        if pointing.default is None
        else _resolve_pointing_offset(pointing.default),
        tuple(
            resolved_beams.ResolvedAntennaPointingOffset(
                entry.antenna,
                _resolve_pointing_offset(entry),
            )
            for entry in pointing.per_antenna
        ),
    )


def _resolve_beam_surface_error(
    surface_error: beam_input.BeamSurfaceErrorConfig | None,
) -> resolved_beams.ResolvedBeamSurfaceError | None:
    if surface_error is None:
        return None
    return resolved_beams.ResolvedBeamSurfaceError(
        None
        if surface_error.default is None
        else _resolve_surface_error(surface_error.default),
        tuple(
            resolved_beams.ResolvedAntennaSurfaceError(
                entry.antenna,
                _resolve_surface_error(entry),
            )
            for entry in surface_error.per_antenna
        ),
    )


def _resolve_squint_record(
    record: beam_input.SquintRecordConfig | beam_input.AntennaSquintConfig,
) -> resolved_beams.ResolvedSquintRecord:
    """Carry one authored squint record across exactly as authored.

    Domain and identity rejections have already been collected as
    ``ConfigSemanticError`` issues before resolution runs, and Section 4.1.1
    rules that the mechanical position angle is never wrapped for the author, so
    this conversion changes no value.
    """
    return resolved_beams.ResolvedSquintRecord(
        record.convention,
        record.reference_frequency_hz,
        record.per_feed_offset_deg_at_reference,
        record.mechanical_feed_position_angle_deg,
        record.positive_native_feed,
    )


def _resolve_beam_squint(
    squint: beam_input.BeamSquintConfig | None,
) -> resolved_beams.ResolvedBeamSquint | None:
    if squint is None:
        return None
    return resolved_beams.ResolvedBeamSquint(
        None if squint.default is None else _resolve_squint_record(squint.default),
        tuple(
            resolved_beams.ResolvedAntennaSquint(
                entry.antenna,
                _resolve_squint_record(entry),
            )
            for entry in squint.per_antenna
        ),
    )


def _resolve_beam_input(
    beams: beam_input.BeamsConfig,
    resolver: _PathResolver,
) -> resolved_beams.ResolvedBeamsInput:
    pointing = _resolve_beam_pointing(beams.pointing)
    surface_error = _resolve_beam_surface_error(beams.surface_error)
    aperture_physics = _resolve_aperture_physics(beams.aperture_physics)
    squint = _resolve_beam_squint(beams.squint)
    if isinstance(beams, beam_input.AnalyticBeamsConfig):
        return resolved_beams.ResolvedAnalyticBeamsInput(
            "analytic",
            _resolve_analytic_definition(beams.model),
            pointing,
            surface_error,
            aperture_physics,
            squint,
        )
    if isinstance(beams, beam_input.SharedFITSBeamsConfig):
        return resolved_beams.ResolvedSharedFITSBeamsInput(
            "shared_fits",
            _resolve_fits_definition(
                beams.beam,
                logical_path="beams.beam.path",
                resolver=resolver,
            ),
            pointing,
            surface_error,
            aperture_physics,
            squint,
        )
    if isinstance(beams, beam_input.PerAntennaFITSBeamsConfig):
        assignments = tuple(
            resolved_beams.ResolvedFITSBeamAssignmentInput(
                assignment.antenna,
                _resolve_fits_definition(
                    assignment.beam,
                    logical_path=f"beams.assignments[{index}].beam.path",
                    resolver=resolver,
                ),
            )
            for index, assignment in enumerate(beams.assignments)
        )
        return resolved_beams.ResolvedPerAntennaFITSBeamsInput(
            "per_antenna_fits",
            assignments,
            pointing,
            surface_error,
            aperture_physics,
            squint,
        )
    if isinstance(beams, beam_input.MixedBeamsConfig):
        mixed_assignments = tuple(
            resolved_beams.ResolvedMixedBeamAssignmentInput(
                assignment.antenna,
                resolved_beams.ResolvedAnalyticBeamChoice("analytic")
                if isinstance(assignment.beam, beam_input.AnalyticBeamChoiceConfig)
                else _resolve_fits_definition(
                    assignment.beam,
                    logical_path=f"beams.assignments[{index}].beam.path",
                    resolver=resolver,
                ),
            )
            for index, assignment in enumerate(beams.assignments)
        )
        return resolved_beams.ResolvedMixedBeamsInput(
            "mixed",
            _resolve_analytic_definition(beams.analytic_model),
            mixed_assignments,
            pointing,
            surface_error,
            aperture_physics,
            squint,
        )
    raise TypeError(f"unsupported beams mode {type(beams).__name__}")


def _normalize_start_time(value: str) -> str:
    time_module: Any = import_module("astropy.time")
    return str(time_module.Time(value).utc.isot)


def _resolution_issue(path: str, code: str, message: str) -> ConfigIssue:
    return ConfigIssue(
        path,
        code,
        message,
        stage="resolution",
        category="resolution",
    )


def _validated_source(value: object) -> ConfigurationSource:
    if isinstance(value, ConfigurationSource):
        return value
    raise ConfigSourceError(
        [
            _source_issue(
                "source",
                "invalid_source_object",
                "source must be a ConfigurationSource",
            )
        ]
    )


def _validated_simulation_overrides(
    value: object,
) -> SimulationOverrides | None:
    if value is None or isinstance(value, SimulationOverrides):
        return value
    raise ConfigOverrideError(
        [
            ConfigIssue(
                "overrides",
                "invalid_simulation_overrides",
                "overrides must be a frozen SimulationOverrides model",
                stage="override",
                category="override",
            )
        ]
    )


def _validated_workflow_overrides(
    value: object,
) -> WorkflowOverrides | None:
    if value is None or isinstance(value, WorkflowOverrides):
        return value
    raise ConfigOverrideError(
        [
            ConfigIssue(
                "workflow_overrides",
                "invalid_workflow_overrides",
                "workflow_overrides must be a frozen WorkflowOverrides model",
                stage="override",
                category="override",
            )
        ]
    )


def _child_source_path(path: str, item: str | int) -> str:
    if isinstance(item, int):
        return f"{path}[{item}]" if path else f"[{item}]"
    return f"{path}.{item}" if path else item


def _normalize_source_value(
    value: object,
    *,
    path: str,
    issues: list[ConfigIssue],
) -> object:
    """Copy one source-native value into the shared validation representation."""
    if isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return _normalize_source_value(value.value, path=path, issues=issues)
    if isinstance(value, Path):
        return Path(value)
    if isinstance(value, np.generic):
        return _normalize_source_value(value.item(), path=path, issues=issues)
    if isinstance(value, str):
        if path == "obs_time.start_time":
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
            else:
                if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                    return parsed.astimezone(UTC).replace(tzinfo=None).isoformat()
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                issues.append(
                    _parse_issue(
                        path,
                        "non_string_mapping_key",
                        "configuration mapping keys must be strings",
                        f"Replace key {key!r} with a string field name.",
                    )
                )
                continue
            normalized[key] = _normalize_source_value(
                item,
                path=_child_source_path(path, key),
                issues=issues,
            )
        return normalized
    if isinstance(value, np.ndarray):
        return _normalize_source_value(value.tolist(), path=path, issues=issues)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        return [
            _normalize_source_value(
                item,
                path=_child_source_path(path, index),
                issues=issues,
            )
            for index, item in enumerate(value)
        ]
    issues.append(
        _parse_issue(
            path,
            "unsupported_source_value",
            f"unsupported source value of type {type(value).__name__}",
            "Use YAML/JSON scalar values, paths, mappings, or ordered sequences.",
        )
    )
    return None


def _normalized_source_mapping(
    config: RadioSimConfig | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(config, RadioSimConfig):
        source_data = config.model_dump(mode="python", exclude_unset=True)
        if "beams" in config.model_fields_set:
            source_data["beams"] = config.beams.model_dump(mode="python")
        frequency = cast(dict[str, object], source_data["obs_frequency"])
        frequency.setdefault("mode", config.obs_frequency.mode)
        sky_model = cast(dict[str, object], source_data["sky_model"])
        dumped_sources = cast(list[dict[str, object]], sky_model["sources"])
        for dumped_source, model_source in zip(
            dumped_sources,
            config.sky_model.sources,
            strict=True,
        ):
            dumped_source.setdefault("kind", model_source.kind)
    elif isinstance(config, Mapping):
        source_data = dict(config)
    else:
        raise ConfigParseError(
            [
                _parse_issue(
                    "",
                    "invalid_config_source_value",
                    "config must be a RadioSimConfig or mapping",
                )
            ]
        )

    issues: list[ConfigIssue] = []
    normalized = _normalize_source_value(source_data, path="", issues=issues)
    if issues:
        raise ConfigParseError(issues)
    if not isinstance(normalized, dict):
        raise ConfigParseError(
            [
                _parse_issue(
                    "",
                    "config_root_not_mapping",
                    "configuration source must contain a top-level mapping",
                )
            ]
        )
    return cast(dict[str, object], normalized)


def resolve_config(
    config: RadioSimConfig | Mapping[str, object],
    *,
    source: ConfigurationSource,
    overrides: SimulationOverrides | None = None,
    workflow_overrides: WorkflowOverrides | None = None,
    check_input_paths: bool = True,
) -> ResolvedConfiguration:
    """Resolve a normalized model or copied mapping into one configuration."""
    source = _validated_source(source)
    overrides = _validated_simulation_overrides(overrides)
    workflow_overrides = _validated_workflow_overrides(workflow_overrides)
    model = _schema_model(_normalized_source_mapping(config))
    input_snapshot = json_safe_mapping(model.model_dump(mode="json"))
    try:
        candidate, origins = _apply_overrides(
            model,
            overrides,
            workflow_overrides,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise ConfigOverrideError(
            [
                ConfigIssue(
                    "overrides",
                    "override_application_failed",
                    str(error),
                    stage="override",
                    category="override",
                )
            ]
        ) from error

    semantic = collect_semantic_issues(candidate)
    unsupported = collect_unsupported_issues(candidate)
    if semantic:
        raise ConfigSemanticError((*semantic, *unsupported))
    if unsupported:
        raise UnsupportedConfigError(unsupported)

    path_resolver = _PathResolver(
        source,
        check_input_paths=check_input_paths,
    )
    instrument = candidate.instrument
    if isinstance(instrument.source, LayoutFileSourceConfig):
        antenna_expected: Literal["file", "directory"] = (
            "directory" if instrument.source.format == "measurement_set" else "file"
        )
        antenna_path = path_resolver.path(
            instrument.source.path,
            logical_path="instrument.source.path",
            origin=origins["instrument.source.path"],
            expected=antenna_expected,
        )
        instrument = instrument.model_copy(
            update={
                "source": instrument.source.model_copy(update={"path": antenna_path})
            }
        )
    beam_config = _resolve_beam_input(candidate.beams, path_resolver)
    sources = tuple(
        _resolve_sky_source(source_config, index=index, resolver=path_resolver)
        for index, source_config in enumerate(candidate.sky_model.sources)
    )
    output_dir = path_resolver.path(
        candidate.workflow.output_dir,
        logical_path="workflow.output_dir",
        origin=origins["workflow.output_dir"],
        expected="output_directory",
    )
    if path_resolver.source_issues:
        raise ConfigSourceError(path_resolver.source_issues)
    if path_resolver.path_issues:
        raise ConfigPathError(path_resolver.path_issues)

    try:
        precision = candidate.execution.precision.to_precision_config()
        precision = PrecisionConfig.model_validate(precision.model_dump(mode="python"))
        frequency = _resolve_frequency(candidate.obs_frequency)
        provenance = ConfigurationProvenance(
            source=source,
            input_snapshot=input_snapshot,
            override_origins=FrozenMapping(origins),
            path_resolutions=FrozenMapping(path_resolver.records),
        )
        workflow_data = candidate.workflow.model_dump(mode="python")
        workflow_data["output_dir"] = output_dir
        resolved_workflow = CliWorkflowConfig.model_validate(workflow_data)
        time_grid = build_observation_time_grid(
            start_time=_normalize_start_time(candidate.obs_time.start_time),
            duration_seconds=candidate.obs_time.duration_seconds,
            cadence_seconds=candidate.obs_time.time_step_seconds,
        )
        runtime = ResolvedSimulationConfig(
            instrument=instrument,
            beams=beam_config,
            baseline_selection=candidate.baseline_selection,
            receptors=candidate.receptors,
            jones=candidate.jones,
            sky_model=ResolvedSkyModelConfig(
                sources=sources,
                flux_unit=candidate.sky_model.flux_unit,
                brightness_conversion=candidate.sky_model.brightness_conversion,
                mixed_model_policy=candidate.sky_model.mixed_model_policy,
                assume_disjoint=candidate.sky_model.assume_disjoint,
                region=_source_common_value(candidate.sky_model.region),
            ),
            observation=ResolvedObservationConfig(time_grid=time_grid),
            frequency=frequency,
            visibility=FrozenMapping(candidate.visibility.model_dump(mode="python")),
            execution=ResolvedExecutionConfig(
                backend_strategy=candidate.execution.backend,
                precision=precision,
                simulator=candidate.execution.simulator,
                offline=candidate.execution.offline,
                sky_loading=_resolve_sky_loading(
                    candidate.execution.sky_loading,
                    request_count=len(sources),
                ),
                solver=_resolve_solver_execution(
                    candidate.execution.solver,
                    time_sample_count=len(time_grid),
                ),
            ),
        )
        return ResolvedConfiguration(
            runtime=runtime,
            workflow=resolved_workflow,
            provenance=provenance,
        )
    except ConfigResolutionError:
        raise
    except Exception as error:
        raise ConfigResolutionError(
            [
                _resolution_issue(
                    "",
                    "runtime_resolution_failed",
                    str(error),
                )
            ]
        ) from error


__all__ = [
    "ConfigOverrideError",
    "ConfigParseError",
    "ConfigPathError",
    "ConfigResolutionError",
    "ConfigSchemaError",
    "ConfigSemanticError",
    "ConfigSourceError",
    "ConfigurationSource",
    "InstrumentSourcePathOverride",
    "SimulationOverrides",
    "UnsupportedConfigError",
    "WorkflowOverrides",
    "resolve_config",
]
