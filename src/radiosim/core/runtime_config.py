"""Deeply immutable runtime configuration values.

The input schema in :mod:`radiosim.io.config` owns user coercion.  This module
owns the post-resolution values consumed by later runtime tiers: absolute
paths, immutable containers, exact frequency samples, frozen precision, and
versioned provenance.  It deliberately has no Simulator or loader dependency.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

import numpy as np
from pydantic import BaseModel
from typing_extensions import override

from radiosim.core.beam.models import (
    ResolvedAnalyticBeamsInput,
    ResolvedBeamsInput,
    ResolvedMixedBeamsInput,
    ResolvedPerAntennaFITSBeamsInput,
    ResolvedSharedFITSBeamsInput,
)
from radiosim.core.precision import PrecisionConfig

if TYPE_CHECKING:
    from radiosim.core.time_grid import ObservationTimeGrid
    from radiosim.io.config import CliWorkflowConfig
    from radiosim.io.config_resolution import ConfigurationSource
    from radiosim.io.instrument_config import (
        BaselineSelectionConfig,
        InstrumentConfig,
    )
    from radiosim.io.jones_config import JonesConfig
    from radiosim.io.receptor_config import ReceptorsConfig

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
ValueOrigin = Literal["default", "document", "override"]
PathValueKind = Literal["file", "directory", "output_directory", "glob"]


class FrozenMapping(Mapping[str, Any]):
    """A recursively copy-owning dictionary with no mutation surface."""

    __slots__ = ("_data",)
    _data: Mapping[str, Any]

    def __init__(
        self,
        value: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        copied: dict[str, Any] = dict(value or {})
        copied.update(kwargs)
        frozen: dict[str, Any] = {
            str(key): freeze_runtime_value(item) for key, item in copied.items()
        }
        object.__setattr__(self, "_data", MappingProxyType(frozen))

    @override
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._data)!r})"

    @override
    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise TypeError("FrozenMapping is immutable")

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        raise TypeError("FrozenMapping is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self) -> FrozenMapping:
        return FrozenMapping(self)

    def __deepcopy__(self, memo: dict[int, object]) -> FrozenMapping:
        return FrozenMapping(self)


def freeze_runtime_value(value: Any) -> Any:
    """Recursively copy containers without retaining caller-owned state."""
    if isinstance(value, np.ndarray):
        return tuple(freeze_runtime_value(item) for item in value.tolist())
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return FrozenMapping(
            {str(key): freeze_runtime_value(item) for key, item in mapping.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_runtime_value(item) for item in cast(Sequence[Any], value))
    return value


def _json_safe_value(value: Any) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON-safe runtime values must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return cast(JsonScalar, value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_safe_value(value.value)
    if isinstance(value, np.generic):
        return _json_safe_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {str(key): _json_safe_value(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in cast(Sequence[Any], value)]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, BaseModel):
        return _json_safe_value(value.model_dump(mode="json"))
    raise TypeError(f"value of type {type(value).__name__} is not JSON-safe")


def json_safe_mapping(value: Mapping[str, Any]) -> FrozenMapping:
    """Return a copy-owned immutable mapping containing JSON-safe values."""
    converted = _json_safe_value(value)
    if not isinstance(converted, dict):
        raise TypeError("expected a mapping")
    return FrozenMapping(cast(dict[str, Any], converted))


def _require_absolute(path: Path | None, field_name: str) -> None:
    if path is not None and not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute")


@dataclass(frozen=True, slots=True)
class ResolvedObservationConfig:
    """Resolved observation timing."""

    time_grid: ObservationTimeGrid

    def __post_init__(self) -> None:
        from radiosim.core.time_grid import ObservationTimeGrid

        if type(self.time_grid) is not ObservationTimeGrid:
            raise TypeError("time_grid must be an ObservationTimeGrid")

    @property
    def start_time_iso(self) -> str:
        """Forward the canonical start for uncutover Tier 4B consumers."""
        return self.time_grid.start_time_iso

    @property
    def duration_seconds(self) -> float:
        """Forward the configured duration for uncutover Tier 4B consumers."""
        return self.time_grid.duration_seconds

    @property
    def time_step_seconds(self) -> float:
        """Forward canonical cadence for uncutover Tier 4B consumers."""
        return self.time_grid.cadence_seconds


@dataclass(frozen=True, slots=True)
class ResolvedFrequencyConfig:
    """Exact immutable channel samples expressed in Hz."""

    channel_frequencies_hz: tuple[float, ...]
    channel_widths_hz: tuple[float, ...]
    source_mode: Literal["grid", "explicit"]

    def __post_init__(self) -> None:
        if type(self.source_mode) is not str or self.source_mode not in {
            "grid",
            "explicit",
        }:
            raise TypeError("source_mode must be 'grid' or 'explicit'")
        if any(
            isinstance(value, (bool, np.bool_))
            for value in (*self.channel_frequencies_hz, *self.channel_widths_hz)
        ):
            raise TypeError("frequency centers and widths cannot be boolean")
        try:
            copied = tuple(float(value) for value in self.channel_frequencies_hz)
            widths = tuple(float(value) for value in self.channel_widths_hz)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("frequency centers and widths must be numeric") from exc
        if not copied:
            raise ValueError("channel_frequencies_hz must be nonempty")
        if any(not math.isfinite(value) or value <= 0.0 for value in copied):
            raise ValueError("channel frequencies must be finite and positive")
        if any(right <= left for left, right in zip(copied, copied[1:], strict=False)):
            raise ValueError("channel frequencies must be strictly increasing")
        if len(widths) != len(copied):
            raise ValueError("channel widths must match channel frequencies")
        if any(not math.isfinite(value) or value <= 0.0 for value in widths):
            raise ValueError("channel widths must be finite and positive")
        object.__setattr__(self, "channel_frequencies_hz", copied)
        object.__setattr__(self, "channel_widths_hz", widths)

    def as_numpy(self) -> np.ndarray:
        """Return a newly owned float64 array on every call."""
        return np.array(self.channel_frequencies_hz, dtype=np.float64, copy=True)

    def widths_as_numpy(self) -> np.ndarray:
        """Return newly owned float64 channel widths on every call."""
        return np.array(self.channel_widths_hz, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class ResolvedSkySourceRequest:
    """One canonical loader request without executing or retaining a loader."""

    kind: str
    options: FrozenMapping
    region: Any = None
    brightness_conversion: str | None = None
    provenance_override: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", FrozenMapping(self.options))
        object.__setattr__(self, "region", freeze_runtime_value(self.region))
        object.__setattr__(
            self,
            "provenance_override",
            freeze_runtime_value(self.provenance_override),
        )


@dataclass(frozen=True, slots=True)
class ResolvedSkyModelConfig:
    """Resolved immutable sky-model policy and canonical source requests."""

    sources: tuple[ResolvedSkySourceRequest, ...]
    flux_unit: str
    brightness_conversion: str
    mixed_model_policy: str
    assume_disjoint: bool
    region: Any = None

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        if any(type(source) is not ResolvedSkySourceRequest for source in sources):
            raise TypeError("sources must contain only ResolvedSkySourceRequest values")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "region", freeze_runtime_value(self.region))

    @property
    def requests(self) -> tuple[ResolvedSkySourceRequest, ...]:
        """Alias exposing that resolved sources are loader requests."""
        return self.sources


def _require_worker_count(value: Any, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ResolvedSkyLoadingConfig:
    """Loader-side concurrency policy with `max_workers` already resolved."""

    max_workers: int
    executor: Literal["auto", "thread", "process"]

    def __post_init__(self) -> None:
        _require_worker_count(self.max_workers, "max_workers")
        if self.executor not in {"auto", "thread", "process"}:
            raise ValueError("executor must be 'auto', 'thread', or 'process'")


@dataclass(frozen=True, slots=True)
class ResolvedSolverExecutionConfig:
    """Solver-side concurrency policy with `workers` already clamped."""

    workers: int
    executor: Literal["thread"]

    def __post_init__(self) -> None:
        _require_worker_count(self.workers, "workers")
        if self.executor != "thread":
            raise ValueError("executor must be 'thread'")


@dataclass(frozen=True, slots=True)
class ResolvedExecutionConfig:
    """Requested backend strategy, frozen precision, and worker policy."""

    backend_strategy: Literal["auto", "numpy", "jax", "dask"]
    precision: PrecisionConfig
    #: ``docs/development/sci004_mmode_design.md`` Section 2: the accepted
    #: values are exactly the simulator registry keys.
    simulator: Literal["rime", "mmode"]
    offline: bool
    sky_loading: ResolvedSkyLoadingConfig
    solver: ResolvedSolverExecutionConfig
    #: The resolved ``execution.mmode`` block.  ``None`` -- the only value a
    #: direct run may carry -- means the document declared no m-mode block, and
    #: Section 8 makes an absent block never change a direct run.
    mmode: Any = None

    def __post_init__(self) -> None:
        from radiosim.io.config import MModeExecutionConfig

        if self.backend_strategy not in {"auto", "numpy", "jax", "dask"}:
            raise ValueError("backend_strategy is not supported")
        if type(self.precision) is not PrecisionConfig:
            raise TypeError("precision must be a PrecisionConfig")
        if self.simulator not in {"rime", "mmode"}:
            raise ValueError("simulator must be 'rime' or 'mmode'")
        if self.mmode is not None and type(self.mmode) is not MModeExecutionConfig:
            raise TypeError("mmode must be a MModeExecutionConfig or None")
        if (self.mmode is None) != (self.simulator != "mmode"):
            raise ValueError(
                "execution.mmode is required with simulator='mmode' and "
                "forbidden otherwise"
            )
        if type(self.offline) is not bool:
            raise TypeError("offline must be a boolean")
        if type(self.sky_loading) is not ResolvedSkyLoadingConfig:
            raise TypeError("sky_loading must be a ResolvedSkyLoadingConfig")
        if type(self.solver) is not ResolvedSolverExecutionConfig:
            raise TypeError("solver must be a ResolvedSolverExecutionConfig")

    @property
    def backend(self) -> Literal["auto", "numpy", "jax", "dask"]:
        return self.backend_strategy


@dataclass(frozen=True, slots=True)
class PathResolutionProvenance:
    """Original, lexical-user, and resolved-target information for one path."""

    logical_path: str
    original: str
    base: Path | None
    user_path: Path
    resolved: Path | tuple[Path, ...]
    origin: ValueOrigin
    kind: PathValueKind

    def __post_init__(self) -> None:
        _require_absolute(self.base, "base")
        _require_absolute(self.user_path, "user_path")
        resolved = self.resolved
        if isinstance(resolved, tuple):
            copied = tuple(resolved)
            for path in copied:
                _require_absolute(path, "resolved path")
            object.__setattr__(self, "resolved", copied)
        else:
            _require_absolute(resolved, "resolved path")


@dataclass(frozen=True, slots=True)
class ConfigurationProvenance:
    """Versioned source, input, override, and path provenance."""

    source: ConfigurationSource
    input_snapshot: FrozenMapping
    override_origins: FrozenMapping
    path_resolutions: FrozenMapping
    schema_version: Literal[1] = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_snapshot",
            json_safe_mapping(self.input_snapshot),
        )
        object.__setattr__(
            self,
            "override_origins",
            FrozenMapping(self.override_origins),
        )
        object.__setattr__(
            self,
            "path_resolutions",
            FrozenMapping(self.path_resolutions),
        )

    @property
    def runtime_origins(self) -> FrozenMapping:
        return FrozenMapping(
            {
                key: value
                for key, value in self.override_origins.items()
                if not key.startswith("workflow.")
            }
        )

    @property
    def workflow_origins(self) -> FrozenMapping:
        return FrozenMapping(
            {
                key: value
                for key, value in self.override_origins.items()
                if key.startswith("workflow.")
            }
        )

    def to_json_safe(self) -> dict[str, JsonValue]:
        """Return a newly owned JSON-safe primitive tree."""
        result = _json_safe_value(self)
        if not isinstance(result, dict):
            raise TypeError("provenance did not serialize to a mapping")
        return cast(dict[str, JsonValue], result)

    def runtime_only(self) -> ConfigurationProvenance:
        """Return scientific/runtime provenance with all workflow state removed."""
        return ConfigurationProvenance(
            source=self.source,
            input_snapshot=FrozenMapping(
                {
                    key: value
                    for key, value in self.input_snapshot.items()
                    if key != "workflow"
                }
            ),
            override_origins=self.runtime_origins,
            path_resolutions=FrozenMapping(
                {
                    key: value
                    for key, value in self.path_resolutions.items()
                    if not key.startswith("workflow.")
                }
            ),
            schema_version=self.schema_version,
        )


def _default_receptors_config() -> ReceptorsConfig:
    """Return the default receptor input without importing I/O at module load."""
    from radiosim.io.receptor_config import ReceptorsConfig

    return ReceptorsConfig()


@dataclass(frozen=True, slots=True)
class ResolvedSimulationConfig:
    """The complete scientific/runtime configuration with no workflow state."""

    instrument: InstrumentConfig
    beams: ResolvedBeamsInput
    baseline_selection: BaselineSelectionConfig
    sky_model: ResolvedSkyModelConfig
    observation: ResolvedObservationConfig
    frequency: ResolvedFrequencyConfig
    visibility: FrozenMapping
    execution: ResolvedExecutionConfig
    #: The validated ``obs_time`` input, carried through resolution untouched so
    #: a consumer can read which of the two Section 3.2 variants was declared.
    #: ``ResolvedObservationConfig`` keeps the canonical UTC sample grid either
    #: way; this record is what distinguishes an m-mode full-sidereal cycle from
    #: a UTC-uniform interval that happens to have the same sample count.
    obs_time: Any = None
    receptors: ReceptorsConfig = field(default_factory=_default_receptors_config)
    #: The Tier 7 ``jones:`` section, carried through resolution untouched.
    #: ``None`` means the document selected the current empty optional-term
    #: inventory.  An authored empty section can reach this record, then Jones
    #: resolution rejects it under R2 before simulation setup completes.
    jones: JonesConfig | None = None

    def __post_init__(self) -> None:
        from radiosim.io.instrument_config import (
            BaselineSelectionConfig,
            InstrumentConfig,
        )
        from radiosim.io.jones_config import JonesConfig
        from radiosim.io.receptor_config import ReceptorsConfig

        if type(self.instrument) is not InstrumentConfig:
            raise TypeError("instrument must be an InstrumentConfig")
        if type(self.baseline_selection) is not BaselineSelectionConfig:
            raise TypeError("baseline_selection must be a BaselineSelectionConfig")
        if type(self.receptors) is not ReceptorsConfig:
            raise TypeError("receptors must be a ReceptorsConfig")
        if self.jones is not None and type(self.jones) is not JonesConfig:
            raise TypeError("jones must be a JonesConfig or None")
        if type(self.beams) not in (
            ResolvedAnalyticBeamsInput,
            ResolvedSharedFITSBeamsInput,
            ResolvedPerAntennaFITSBeamsInput,
            ResolvedMixedBeamsInput,
        ):
            raise TypeError("beams must be an exact ResolvedBeamsInput")
        for field_name, expected_type in (
            ("sky_model", ResolvedSkyModelConfig),
            ("observation", ResolvedObservationConfig),
            ("frequency", ResolvedFrequencyConfig),
            ("execution", ResolvedExecutionConfig),
        ):
            if type(getattr(self, field_name)) is not expected_type:
                raise TypeError(f"{field_name} must be a {expected_type.__name__}")
        if not isinstance(self.visibility, Mapping):
            raise TypeError("visibility must be a mapping")
        object.__setattr__(self, "visibility", FrozenMapping(self.visibility))

    def to_json_safe(self) -> dict[str, JsonValue]:
        """Return a newly owned JSON-safe scientific configuration snapshot."""
        result = _json_safe_value(self)
        if not isinstance(result, dict):
            raise TypeError("resolved simulation config did not serialize to a mapping")
        return cast(dict[str, JsonValue], result)


@dataclass(frozen=True, slots=True)
class ResolvedConfiguration:
    """Resolved runtime, separate CLI workflow, and shared provenance."""

    runtime: ResolvedSimulationConfig
    workflow: CliWorkflowConfig
    provenance: ConfigurationProvenance


__all__ = [
    "ConfigurationProvenance",
    "FrozenMapping",
    "JsonValue",
    "PathResolutionProvenance",
    "ResolvedConfiguration",
    "ResolvedExecutionConfig",
    "ResolvedFrequencyConfig",
    "ResolvedObservationConfig",
    "ResolvedSimulationConfig",
    "ResolvedSkyLoadingConfig",
    "ResolvedSkyModelConfig",
    "ResolvedSkySourceRequest",
    "ResolvedSolverExecutionConfig",
    "ValueOrigin",
    "freeze_runtime_value",
    "json_safe_mapping",
]
