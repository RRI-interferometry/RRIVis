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

from radiosim.core.precision import PrecisionConfig

if TYPE_CHECKING:
    from radiosim.io.config import CliWorkflowConfig
    from radiosim.io.config_resolution import ConfigurationSource
    from radiosim.io.instrument_config import (
        BaselineSelectionConfig,
        InstrumentConfig,
    )

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
class ResolvedTelescopeConfig:
    """Resolved telescope identity and disabled future-source flags."""

    telescope_name: str
    use_pyuvdata_telescope: bool = False
    use_pyuvdata_location: bool = False
    use_pyuvdata_antennas: bool = False
    use_pyuvdata_diameters: bool = False


@dataclass(frozen=True, slots=True)
class ResolvedAntennaLayoutConfig:
    """Resolved antenna-layout path and current uniform-diameter contract."""

    antenna_positions_file: Path
    antenna_file_format: str
    all_antenna_diameter: float
    use_different_diameters: bool = False
    diameters: FrozenMapping = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        _require_absolute(
            self.antenna_positions_file,
            "antenna_positions_file",
        )
        object.__setattr__(self, "diameters", FrozenMapping(self.diameters))


@dataclass(frozen=True, slots=True)
class ResolvedBeamsConfig:
    """Resolved beam input without activating deferred FITS behavior."""

    beam_mode: str
    per_antenna: bool
    beam_file: Path | None
    antenna_beam_map: FrozenMapping
    beam_za_max_deg: float | None
    beam_za_buffer_deg: float | None
    beam_freq_buffer_hz: float | None
    beam_peak_normalize: bool
    beam_interp_function: str | None
    aperture_shape: str
    taper: str
    edge_taper_dB: float
    feed_model: str
    feed_computation: str
    feed_params: FrozenMapping
    reflector_type: str
    magnification: float
    aperture_params: FrozenMapping

    def __post_init__(self) -> None:
        _require_absolute(self.beam_file, "beam_file")
        object.__setattr__(
            self,
            "antenna_beam_map",
            FrozenMapping(self.antenna_beam_map),
        )
        object.__setattr__(self, "feed_params", FrozenMapping(self.feed_params))
        object.__setattr__(
            self,
            "aperture_params",
            FrozenMapping(self.aperture_params),
        )


@dataclass(frozen=True, slots=True)
class ResolvedLocationConfig:
    """Resolved observatory location with unit-explicit field names."""

    lat_deg: float
    lon_deg: float
    height_m: float


@dataclass(frozen=True, slots=True)
class ResolvedObservationConfig:
    """Resolved observation timing."""

    start_time_iso: str
    duration_seconds: float
    time_step_seconds: float


@dataclass(frozen=True, slots=True)
class ResolvedFrequencyConfig:
    """Exact immutable channel samples expressed in Hz."""

    channel_frequencies_hz: tuple[float, ...]
    source_mode: Literal["grid", "explicit"]

    def __post_init__(self) -> None:
        copied = tuple(float(value) for value in self.channel_frequencies_hz)
        if not copied:
            raise ValueError("channel_frequencies_hz must be nonempty")
        if any(not math.isfinite(value) or value <= 0.0 for value in copied):
            raise ValueError("channel frequencies must be finite and positive")
        object.__setattr__(self, "channel_frequencies_hz", copied)

    def as_numpy(self) -> np.ndarray:
        """Return a newly owned float64 array on every call."""
        return np.array(self.channel_frequencies_hz, dtype=np.float64, copy=True)


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
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "region", freeze_runtime_value(self.region))

    @property
    def requests(self) -> tuple[ResolvedSkySourceRequest, ...]:
        """Alias exposing that resolved sources are loader requests."""
        return self.sources


@dataclass(frozen=True, slots=True)
class ResolvedExecutionConfig:
    """Requested backend strategy and frozen pre-backend precision."""

    backend_strategy: Literal["auto", "numpy", "jax", "numba"]
    precision: PrecisionConfig
    simulator: Literal["rime"]
    offline: bool

    @property
    def backend(self) -> Literal["auto", "numpy", "jax", "numba"]:
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


@dataclass(frozen=True, slots=True)
class ResolvedSimulationConfig:
    """The complete scientific/runtime configuration with no workflow state."""

    instrument: InstrumentConfig
    beams: ResolvedBeamsConfig
    baseline_selection: BaselineSelectionConfig
    sky_model: ResolvedSkyModelConfig
    observation: ResolvedObservationConfig
    frequency: ResolvedFrequencyConfig
    visibility: FrozenMapping
    execution: ResolvedExecutionConfig

    def __post_init__(self) -> None:
        from radiosim.io.instrument_config import (
            BaselineSelectionConfig,
            InstrumentConfig,
        )

        if type(self.instrument) is not InstrumentConfig:
            raise TypeError("instrument must be an InstrumentConfig")
        if type(self.baseline_selection) is not BaselineSelectionConfig:
            raise TypeError("baseline_selection must be a BaselineSelectionConfig")
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
    "ResolvedAntennaLayoutConfig",
    "ResolvedBeamsConfig",
    "ResolvedConfiguration",
    "ResolvedExecutionConfig",
    "ResolvedFrequencyConfig",
    "ResolvedLocationConfig",
    "ResolvedObservationConfig",
    "ResolvedSimulationConfig",
    "ResolvedSkyModelConfig",
    "ResolvedSkySourceRequest",
    "ResolvedTelescopeConfig",
    "ValueOrigin",
    "freeze_runtime_value",
    "json_safe_mapping",
]
