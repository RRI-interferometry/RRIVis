"""Strict, immutable user-input configuration models for RadioSim.

This module owns the Tier 1 input document and its public YAML acquisition and
serialization boundaries.  It validates and copies user-authored values and
exposes pure semantic and unsupported-feature issue collectors.  It does not
select a backend or device, execute sky loaders, or create output directories.
"""

from __future__ import annotations

import difflib
import inspect
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NoReturn,
    TypeVar,
    cast,
    get_type_hints,
)

import numpy as np
import yaml
from pydantic import (
    Field,
    PlainSerializer,
    SerializeAsAny,
    TypeAdapter,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import override

from radiosim.core.precision import PrecisionLevel
from radiosim.core.sky.containers import (
    MonopoleConvention,
    SkyCoverage,
    SourceSubtractionStatus,
)
from radiosim.core.sky.containers.constants import (
    DEFAULT_BRIGHT_CATALOG_FLUX_MIN_JY,
    DEFAULT_CONFUSION_SPECTRAL_INDEX_DIST,
)
from radiosim.io.model_base import StrictFrozenModel
from radiosim.io.result_format import ResultFormat

if TYPE_CHECKING:
    from radiosim.core.runtime_config import ResolvedConfiguration
    from radiosim.io.config_resolution import (
        SimulationOverrides,
        WorkflowOverrides,
    )

DEFAULT_SKY_REPRESENTATION: Literal["point_sources"] = "point_sources"

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class FrozenDict(Mapping[str, Any]):
    """Copy-owning mapping with no mutable ``dict`` base-class escape hatch."""

    __slots__ = ("_data",)

    _data: Mapping[str, Any]

    def __init__(self, value: Mapping[str, Any] | None = None) -> None:
        copied = {key: _freeze_value(item) for key, item in dict(value or {}).items()}
        object.__setattr__(self, "_data", MappingProxyType(copied))

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
        return repr(dict(self.items()))

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        other_mapping = cast(Mapping[Any, Any], other)
        return dict(self.items()) == dict(other_mapping.items())

    @override
    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise TypeError("FrozenDict is immutable")

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        raise TypeError("FrozenDict is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    setdefault = _immutable
    update = _immutable

    def popitem(self) -> NoReturn:
        raise TypeError("FrozenDict is immutable")

    def __ior__(self, value: object) -> NoReturn:
        raise TypeError("FrozenDict is immutable")

    def __copy__(self) -> FrozenDict:
        return FrozenDict({key: _freeze_value(item) for key, item in self.items()})

    def __deepcopy__(self, memo: dict[int, object]) -> FrozenDict:
        return FrozenDict({key: _freeze_value(item) for key, item in self.items()})


_MapValue = TypeVar("_MapValue")


def _serialize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return an ordinary recursively owned mapping for Pydantic serialization."""
    serialized: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            serialized[key] = _serialize_mapping(cast(Mapping[str, Any], item))
        else:
            serialized[key] = item
    return serialized


SerializableMapping = Annotated[
    Mapping[str, _MapValue],
    PlainSerializer(_serialize_mapping, return_type=dict[str, Any]),
]


def _freeze_value(value: Any) -> Any:
    """Recursively copy mutable input containers into immutable containers."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return FrozenDict({key: _freeze_value(item) for key, item in mapping.items()})
    if isinstance(value, np.ndarray):
        return tuple(_freeze_value(item) for item in cast(list[Any], value.tolist()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in cast(Sequence[Any], value))
    return value


def _freeze_dict(value: Mapping[str, Any]) -> FrozenDict:
    return FrozenDict({str(key): _freeze_value(item) for key, item in value.items()})


IssueStage = Literal[
    "source",
    "schema",
    "override",
    "semantic",
    "unsupported",
    "path",
    "resolution",
]
IssueCategory = Literal[
    "source",
    "schema",
    "override",
    "scientific",
    "workflow",
    "unsupported",
    "path",
    "resolution",
]


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    """One immutable, renderable configuration issue."""

    path: str
    code: str
    message: str
    hint: str | None = None
    stage: IssueStage = "semantic"
    category: IssueCategory = "scientific"

    def render(self) -> str:
        text = f"{self.path}: {self.message}" if self.path else self.message
        if self.hint:
            text += f" Hint: {self.hint}"
        return text


_STAGE_ORDER: dict[IssueStage, int] = {
    "source": 0,
    "schema": 1,
    "override": 2,
    "semantic": 3,
    "unsupported": 4,
    "path": 5,
    "resolution": 6,
}


def _ordered_issues(issues: Sequence[ConfigIssue]) -> tuple[ConfigIssue, ...]:
    return tuple(
        sorted(
            issues,
            key=lambda issue: (_STAGE_ORDER[issue.stage], issue.path, issue.code),
        )
    )


def _nonblank(value: str, *, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be nonempty")
    return stripped


def _nonempty_path_input(value: Any, *, field_name: str) -> Any:
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field_name} must be a nonempty path")
    return value


def _finite_number_map(
    value: Mapping[str, float],
    *,
    positive: bool = False,
) -> FrozenDict:
    copied: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = _nonblank(str(raw_key), field_name="mapping key")
        if isinstance(raw_value, (bool, np.bool_)):
            raise ValueError(f"{key!r} must be numeric, not boolean")
        numeric = float(raw_value)
        if not math.isfinite(numeric):
            raise ValueError(f"{key!r} must be finite")
        if positive and numeric <= 0.0:
            raise ValueError(f"{key!r} must be > 0")
        copied[key] = numeric
    return _freeze_dict(copied)


def _is_valid_nside(value: int) -> bool:
    return not isinstance(value, bool) and value > 0 and value & (value - 1) == 0


def _validate_nside(value: int) -> int:
    if not _is_valid_nside(value):
        raise ValueError("NSIDE must be a positive power of two")
    return value


class SkyRegionEntryConfig(StrictFrozenModel):
    """One immutable cone or box region input."""

    shape: Literal["cone", "box"] = "cone"
    center_ra_deg: Annotated[float, Field(ge=0.0, lt=360.0, allow_inf_nan=False)]
    center_dec_deg: Annotated[float, Field(ge=-90.0, le=90.0, allow_inf_nan=False)]
    radius_deg: (
        Annotated[float, Field(gt=0.0, le=180.0, allow_inf_nan=False)] | None
    ) = None
    width_deg: Annotated[float, Field(gt=0.0, le=360.0, allow_inf_nan=False)] | None = (
        None
    )
    height_deg: (
        Annotated[float, Field(gt=0.0, le=180.0, allow_inf_nan=False)] | None
    ) = None


SkyRegionInput = SkyRegionEntryConfig | tuple[SkyRegionEntryConfig, ...] | None


def _copy_region_sequence(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(cast(list[Any], value))
    return value


def build_sky_region(
    config: SkyRegionEntryConfig
    | tuple[SkyRegionEntryConfig, ...]
    | list[SkyRegionEntryConfig]
    | dict[str, Any]
    | list[dict[str, Any]]
    | None,
) -> Any:
    """Build a runtime ``SkyRegion`` from already validated config input."""
    if config is None:
        return None

    from radiosim.core.sky.operations.region import SkyRegion

    def _build_one(entry: SkyRegionEntryConfig | dict[str, Any]) -> Any:
        if not isinstance(entry, SkyRegionEntryConfig):
            entry = SkyRegionEntryConfig.model_validate(entry)
        if entry.shape == "cone":
            assert entry.radius_deg is not None
            return SkyRegion.cone(
                entry.center_ra_deg,
                entry.center_dec_deg,
                entry.radius_deg,
            )
        assert entry.width_deg is not None
        assert entry.height_deg is not None
        return SkyRegion.box(
            entry.center_ra_deg,
            entry.center_dec_deg,
            entry.width_deg,
            entry.height_deg,
        )

    if isinstance(config, (list, tuple)):
        return SkyRegion.union([_build_one(entry) for entry in config])
    return _build_one(config)


class SkyFootprintInput(StrictFrozenModel):
    """Strict immutable HEALPix support footprint."""

    nside: int
    hpx_inds: tuple[int, ...]
    coordinate_frame: Literal["icrs", "galactic"] = "icrs"

    @field_validator("nside")
    @classmethod
    def validate_nside(cls, value: int) -> int:
        return _validate_nside(value)

    @field_validator("hpx_inds", mode="before")
    @classmethod
    def copy_indices(cls, value: Any) -> tuple[int, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            if not isinstance(value, np.ndarray):
                raise ValueError("hpx_inds must be a one-dimensional sequence")
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError("hpx_inds must be one-dimensional")
            items: list[Any] = cast(list[Any], value.tolist())
        else:
            items = list(cast(Sequence[Any], value))
        copied: list[int] = []
        for item in items:
            if isinstance(item, (bool, np.bool_)) or not isinstance(
                item, (int, np.integer)
            ):
                raise ValueError("hpx_inds values must be integers")
            copied.append(int(cast(Any, item)))
        return tuple(copied)

    @model_validator(mode="after")
    def validate_indices(self) -> SkyFootprintInput:
        if not self.hpx_inds:
            raise ValueError("hpx_inds must be nonempty")
        if len(set(self.hpx_inds)) != len(self.hpx_inds):
            raise ValueError("hpx_inds must be unique")
        upper = 12 * self.nside * self.nside
        if any(index < 0 or index >= upper for index in self.hpx_inds):
            raise ValueError(f"hpx_inds must lie in [0, {upper})")
        return self


FinitePair = tuple[FiniteFloat, FiniteFloat]


class SkyProvenanceInput(StrictFrozenModel):
    """Strict user-authored sky provenance metadata."""

    flux_completeness_jy: FinitePair | None = None
    flux_completeness_freq_hz: PositiveFiniteFloat | None = None
    angular_resolution_rad: FinitePair | None = None
    sky_coverage: SkyCoverage = SkyCoverage.UNKNOWN
    coverage_fraction: (
        Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)] | None
    ) = None
    coverage_footprint: SkyFootprintInput | None = None
    monopole_convention: MonopoleConvention = MonopoleConvention.UNKNOWN
    monopole_k: FiniteFloat | None = None
    source_subtraction: SourceSubtractionStatus = SourceSubtractionStatus.UNKNOWN
    source_subtraction_threshold_jy: NonNegativeFiniteFloat | None = None
    source_subtraction_freq_hz: PositiveFiniteFloat | None = None
    source_subtraction_method: str | None = None
    notes: str | None = None
    rng_seed: int | None = None

    @field_validator("source_subtraction_method")
    @classmethod
    def validate_method(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _nonblank(value, field_name="source_subtraction_method")


@dataclass(frozen=True, slots=True)
class SkyLoaderRequestContext:
    """Resolved global context used to build one loader request."""

    flux_multiplier: float = 1.0
    region: Any = None
    brightness_conversion: Literal["planck", "rayleigh-jeans"] | None = None
    frequencies: np.ndarray | None = None
    memmap_path: str | None = None


class SkySourceConfig(StrictFrozenModel):
    """Strict base type for one entry in ``sky_model.sources``."""

    kind: str
    region: SkyRegionInput = None
    brightness_conversion: Literal["planck", "rayleigh-jeans"] | None = None
    provenance_override: SkyProvenanceInput | None = None

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        return _nonblank(value, field_name="kind")

    @field_validator("region", mode="before")
    @classmethod
    def copy_region(cls, value: Any) -> Any:
        return _copy_region_sequence(value)

    def to_loader_request(
        self,
        *,
        flux_multiplier: float = 1.0,
        region: Any = None,
        brightness_conversion: Literal["planck", "rayleigh-jeans"] | None = None,
        frequencies: Sequence[float] | np.ndarray | None = None,
        memmap_path: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build a resolved loader request without executing the loader."""
        from radiosim.core.sky.registry import loader_registry
        from radiosim.core.sky.support.frequencies import (
            validate_observation_frequencies,
        )

        resolved_frequencies = (
            None
            if frequencies is None
            else validate_observation_frequencies(
                frequencies,
                label="SkySourceConfig.to_loader_request frequencies",
            )
        )

        context = SkyLoaderRequestContext(
            flux_multiplier=flux_multiplier,
            region=region,
            brightness_conversion=brightness_conversion,
            frequencies=resolved_frequencies,
            memmap_path=memmap_path,
        )
        raw_kind, explicit_kwargs = self._build_loader_request(context)
        return loader_registry.resolve_request(raw_kind, explicit_kwargs)

    def _scaled_flux(
        self,
        value: float | None,
        context: SkyLoaderRequestContext,
    ) -> float | None:
        return None if value is None else value * context.flux_multiplier

    def _common_kwargs(
        self,
        context: SkyLoaderRequestContext,
        *,
        include_frequency_context: bool = False,
        include_memmap: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        brightness_conversion = self.brightness_conversion
        if brightness_conversion is None:
            brightness_conversion = context.brightness_conversion
        if brightness_conversion is not None:
            kwargs["brightness_conversion"] = brightness_conversion

        region = (
            build_sky_region(self.region) if self.region is not None else context.region
        )
        if region is not None:
            kwargs["region"] = region

        if include_frequency_context:
            if context.frequencies is not None:
                kwargs["frequencies"] = context.frequencies
        if include_memmap and context.memmap_path is not None:
            kwargs["memmap_path"] = context.memmap_path

        if self.provenance_override is not None:
            from radiosim.core.sky.containers import SkyFootprint, SkyProvenance

            provenance = self.provenance_override.model_dump()
            footprint = self.provenance_override.coverage_footprint
            if footprint is not None:
                provenance["coverage_footprint"] = SkyFootprint(
                    nside=footprint.nside,
                    hpx_inds=np.asarray(footprint.hpx_inds, dtype=np.int64),
                    coordinate_frame=footprint.coordinate_frame,
                )
            kwargs["provenance"] = SkyProvenance(**provenance)
        return kwargs

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError


class DiffuseSkySourceConfig(SkySourceConfig):
    kind: Literal["diffuse_sky"] = "diffuse_sky"
    model: str = "gsm2008"
    nside: int = 64
    include_cmb: bool | None = None
    basemap: str | None = None
    interpolation: str | None = None

    @field_validator("nside")
    @classmethod
    def validate_nside(cls, value: int) -> int:
        return _validate_nside(value)

    @override
    def _build_loader_request(
        self, context: SkyLoaderRequestContext
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(
            context, include_frequency_context=True, include_memmap=True
        )
        kwargs.update({"model": self.model, "nside": self.nside})
        for name in ("include_cmb", "basemap", "interpolation"):
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value
        return self.kind, kwargs


class Pysm3SourceConfig(SkySourceConfig):
    kind: Literal["pysm3"] = "pysm3"
    components: str | tuple[str, ...] = "s1"
    nside: int = 64
    include_polarization: bool = False

    @field_validator("components", mode="before")
    @classmethod
    def copy_components(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("components must be nonempty")
            return tuple(cast(Sequence[Any], value))
        return value

    @field_validator("components")
    @classmethod
    def validate_components(cls, value: str | tuple[str, ...]) -> Any:
        if isinstance(value, str):
            return _nonblank(value, field_name="components")
        if not value:
            raise ValueError("components must be nonempty")
        return tuple(_nonblank(item, field_name="component") for item in value)

    @field_validator("nside")
    @classmethod
    def validate_nside(cls, value: int) -> int:
        return _validate_nside(value)

    @override
    def _build_loader_request(
        self, context: SkyLoaderRequestContext
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(
            context, include_frequency_context=True, include_memmap=True
        )
        kwargs.update(
            {
                "components": self.components,
                "nside": self.nside,
                "include_polarization": self.include_polarization,
            }
        )
        return self.kind, kwargs


class PointCatalogSourceConfig(SkySourceConfig):
    flux_limit: NonNegativeFiniteFloat | None = None
    max_rows: Annotated[int, Field(ge=1)] | None = None

    def _build_catalog_kwargs(self, context: SkyLoaderRequestContext) -> dict[str, Any]:
        kwargs = self._common_kwargs(context)
        if self.flux_limit is not None:
            kwargs["flux_limit"] = self._scaled_flux(self.flux_limit, context)
        if self.max_rows is not None:
            kwargs["max_rows"] = self.max_rows
        return kwargs


class FullCatalogPointSourceConfig(PointCatalogSourceConfig):
    allow_full_catalog: bool = False

    @override
    def _build_catalog_kwargs(self, context: SkyLoaderRequestContext) -> dict[str, Any]:
        kwargs = super()._build_catalog_kwargs(context)
        kwargs["allow_full_catalog"] = self.allow_full_catalog
        return kwargs


class PyradioskyFileSourceConfig(SkySourceConfig):
    kind: Literal["pyradiosky_file"] = "pyradiosky_file"
    filename: Path
    filetype: str | None = None
    flux_limit: NonNegativeFiniteFloat | None = None
    reference_frequency_hz: PositiveFiniteFloat | None = None
    spectral_loss_policy: Literal["warn", "error"] = "warn"

    @field_validator("filename", mode="before")
    @classmethod
    def validate_filename(cls, value: Any) -> Any:
        return _nonempty_path_input(value, field_name="filename")

    @override
    def _build_loader_request(
        self, context: SkyLoaderRequestContext
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(
            context, include_frequency_context=True, include_memmap=True
        )
        kwargs["filename"] = str(self.filename)
        if self.filetype is not None:
            kwargs["filetype"] = self.filetype
        if self.flux_limit is not None:
            kwargs["flux_limit"] = self._scaled_flux(self.flux_limit, context)
        if self.reference_frequency_hz is not None:
            kwargs["reference_frequency_hz"] = self.reference_frequency_hz
        kwargs["spectral_loss_policy"] = self.spectral_loss_policy
        return self.kind, kwargs


class BbsSourceConfig(SkySourceConfig):
    kind: Literal["bbs"] = "bbs"
    filename: Path
    flux_limit: NonNegativeFiniteFloat | None = None

    @field_validator("filename", mode="before")
    @classmethod
    def validate_filename(cls, value: Any) -> Any:
        return _nonempty_path_input(value, field_name="filename")

    @override
    def _build_loader_request(
        self, context: SkyLoaderRequestContext
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(context)
        kwargs["filename"] = str(self.filename)
        if self.flux_limit is not None:
            kwargs["flux_limit"] = self._scaled_flux(self.flux_limit, context)
        return self.kind, kwargs


class Skyh5MultifileSourceConfig(SkySourceConfig):
    kind: Literal["skyh5_multifile"] = "skyh5_multifile"
    file_glob: str | None = None
    filenames: tuple[Path, ...] | None = None
    reference_frequency_hz: PositiveFiniteFloat | None = None

    @field_validator("filenames", mode="before")
    @classmethod
    def copy_filenames(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            items = cast(Sequence[Any], value)
            for item in items:
                _nonempty_path_input(item, field_name="filenames entry")
            return tuple(items)
        return value

    @field_validator("file_glob")
    @classmethod
    def validate_file_glob(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _nonblank(value, field_name="file_glob")

    @override
    def _build_loader_request(
        self, context: SkyLoaderRequestContext
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(
            context, include_frequency_context=True, include_memmap=True
        )
        if self.file_glob is not None:
            kwargs["file_glob"] = self.file_glob
        elif self.filenames is not None:
            kwargs["filenames"] = [str(path) for path in self.filenames]
        if self.reference_frequency_hz is not None:
            kwargs["reference_frequency_hz"] = self.reference_frequency_hz
        return self.kind, kwargs


class FitsImageSourceConfig(SkySourceConfig):
    kind: Literal["fits_image"] = "fits_image"
    filename: Path
    nside: int = 128

    @field_validator("filename", mode="before")
    @classmethod
    def validate_filename(cls, value: Any) -> Any:
        return _nonempty_path_input(value, field_name="filename")

    @field_validator("nside")
    @classmethod
    def validate_nside(cls, value: int) -> int:
        return _validate_nside(value)

    @override
    def _build_loader_request(
        self, context: SkyLoaderRequestContext
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(
            context, include_frequency_context=True, include_memmap=True
        )
        kwargs.update({"filename": str(self.filename), "nside": self.nside})
        return self.kind, kwargs


class TestSourcesConfig(SkySourceConfig):
    kind: Literal["test_sources"] = "test_sources"
    representation: Literal["point_sources", "healpix_map"] = "point_sources"
    num_sources: Annotated[int, Field(ge=1)] = 100
    distribution: Literal["uniform", "random"] = "uniform"
    seed: int | None = None
    flux_min: NonNegativeFiniteFloat | None = None
    flux_max: NonNegativeFiniteFloat | None = None
    dec_deg: Annotated[float, Field(ge=-90.0, le=90.0, allow_inf_nan=False)] | None = (
        None
    )
    dec_range_deg: (
        Annotated[float, Field(ge=0.0, le=90.0, allow_inf_nan=False)] | None
    ) = None
    spectral_index: FiniteFloat | None = None
    polarization_fraction: Annotated[
        float, Field(ge=0.0, le=1.0, allow_inf_nan=False)
    ] = 0.0
    polarization_angle_deg: FiniteFloat = 0.0
    stokes_v_fraction: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)] = (
        0.0
    )
    nside: int | None = None

    @field_validator("nside")
    @classmethod
    def validate_nside(cls, value: int | None) -> int | None:
        return None if value is None else _validate_nside(value)

    @override
    def _build_loader_request(
        self, context: SkyLoaderRequestContext
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(
            context, include_frequency_context=True, include_memmap=True
        )
        kwargs.update(
            {
                "representation": self.representation,
                "num_sources": self.num_sources,
                "distribution": self.distribution,
                "polarization_fraction": self.polarization_fraction,
                "polarization_angle_deg": self.polarization_angle_deg,
                "stokes_v_fraction": self.stokes_v_fraction,
            }
        )
        for name in (
            "seed",
            "flux_min",
            "flux_max",
            "dec_deg",
            "dec_range_deg",
            "spectral_index",
            "nside",
        ):
            value = getattr(self, name)
            if value is not None:
                if name in {"flux_min", "flux_max"}:
                    value = self._scaled_flux(value, context)
                kwargs[name] = value
        return self.kind, kwargs


def _validate_catalog_option(name: str, value: Any) -> Any:
    if name == "allow_full_catalog":
        if not isinstance(value, bool):
            raise ValueError("allow_full_catalog must be a boolean")
        return value
    if name == "max_rows":
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("max_rows must be a positive integer")
        return value
    if name == "flux_limit":
        if isinstance(value, bool):
            raise ValueError("flux_limit must be a nonnegative finite number")
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError("flux_limit must be a nonnegative finite number")
        return result
    raise AssertionError(name)


def _validate_poisson_option(name: str, value: Any) -> Any:
    if name in {"flux_range_jy", "spectral_index_dist"}:
        if isinstance(value, (str, bytes)) or len(value) != 2:
            raise ValueError(f"{name} must be a two-value sequence")
        pair = tuple(float(item) for item in value)
        if not all(math.isfinite(item) for item in pair):
            raise ValueError(f"{name} values must be finite")
        return pair
    if name in {"reference_frequency", "area_sr"}:
        if value is None and name == "area_sr":
            return None
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric")
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return result
    if name == "nside":
        return _validate_nside(value)
    if name == "seed":
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError("seed must be an integer or null")
        return value
    if name == "representation":
        if value not in {"point_sources", "healpix_map"}:
            raise ValueError("representation must be point_sources or healpix_map")
        return value
    if name == "dn_ds":
        if not isinstance(value, str) or not value.strip():
            raise ValueError("dn_ds must be a nonempty registered preset string")
        return value.strip()
    raise AssertionError(name)


def _validate_registered_options(
    kind: str,
    options: Mapping[str, Any] | None,
    *,
    field_name: str = "options",
) -> FrozenDict:
    """Validate option names and annotated values without executing a loader."""
    from radiosim.core.sky.registry import loader_registry

    definition = loader_registry.definition(kind)
    provided = dict(options or {})
    source_to_loader = {
        source_name: loader_name
        for loader_name, source_name in definition.config_fields.items()
    }
    unknown = sorted(set(provided) - set(source_to_loader))
    if unknown:
        raise ValidationError.from_exception_data(
            "RegisteredLoaderOptions",
            [
                {
                    "type": "extra_forbidden",
                    "loc": (field_name, name),
                    "input": provided[name],
                }
                for name in unknown
            ],
        )

    try:
        annotations = get_type_hints(definition.loader)
    except (NameError, TypeError):
        annotations = {}
    signature = inspect.signature(definition.loader)
    validated: dict[str, Any] = {}
    for source_name, value in provided.items():
        if source_name in {"flux_limit", "max_rows", "allow_full_catalog"}:
            validated[source_name] = _validate_catalog_option(source_name, value)
            continue
        if source_name == "nside":
            validated[source_name] = _validate_nside(value)
            continue
        if source_name in {"include_cmb", "include_polarization"}:
            if not isinstance(value, bool):
                raise ValueError(f"{source_name} must be a boolean")
            validated[source_name] = value
            continue
        if source_name in {"model", "basemap", "interpolation", "components"}:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{source_name} must be a nonempty string")
            validated[source_name] = value.strip()
            continue
        if definition.name == "poisson_confusion":
            validated[source_name] = _validate_poisson_option(source_name, value)
            continue
        loader_name = source_to_loader[source_name]
        annotation = annotations.get(
            loader_name, signature.parameters[loader_name].annotation
        )
        if annotation in {inspect.Parameter.empty, Any} or isinstance(annotation, str):
            raise ValueError(
                f"loader {kind!r} option {source_name!r} lacks a usable type annotation"
            )
        try:
            validated[source_name] = TypeAdapter(annotation).validate_python(value)
        except ValidationError as exc:
            raise ValueError(
                f"option {source_name!r} has an invalid value: {exc.errors()[0]['msg']}"
            ) from exc
    return _freeze_dict(validated)


class RealisticForegroundSourceConfig(SkySourceConfig):
    kind: Literal["realistic_foreground"] = "realistic_foreground"
    diffuse: str = "haslam"
    diffuse_kwargs: SerializableMapping[Any] | None = None
    bright_catalogs: str = "gleam"
    bright_catalog_kwargs: SerializableMapping[Any] | None = None
    bright_catalog_flux_min_jy: NonNegativeFiniteFloat = (
        DEFAULT_BRIGHT_CATALOG_FLUX_MIN_JY
    )
    confusion_flux_range_jy: FinitePair | None = None
    confusion_dn_ds: str = "franzen2019_gleam_154mhz"
    confusion_spectral_index_dist: FinitePair = DEFAULT_CONFUSION_SPECTRAL_INDEX_DIST
    nside: int = 128
    include_cmb: bool = False
    seed: int | None = None
    mixed_model_policy: Literal["error", "warn", "allow"] = "error"

    @field_validator("nside")
    @classmethod
    def validate_nside(cls, value: int) -> int:
        return _validate_nside(value)

    @field_validator("diffuse_kwargs", "bright_catalog_kwargs")
    @classmethod
    def freeze_nested_options(
        cls, value: Mapping[str, Any] | None
    ) -> FrozenDict | None:
        return None if value is None else _freeze_dict(value)

    @model_validator(mode="before")
    @classmethod
    def validate_nested_options(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        copied: dict[str, Any] = dict(cast(Mapping[str, Any], data))
        diffuse = cast(str, copied.get("diffuse", "haslam"))
        if copied.get("diffuse_kwargs") is not None:
            copied["diffuse_kwargs"] = _validate_registered_options(
                diffuse,
                copied["diffuse_kwargs"],
                field_name="diffuse_kwargs",
            )
        bright_catalogs = cast(str, copied.get("bright_catalogs", "gleam"))
        if copied.get("bright_catalog_kwargs") is not None:
            copied["bright_catalog_kwargs"] = _validate_registered_options(
                bright_catalogs,
                copied["bright_catalog_kwargs"],
                field_name="bright_catalog_kwargs",
            )
        return copied

    @override
    def _build_loader_request(
        self, context: SkyLoaderRequestContext
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(
            context, include_frequency_context=True, include_memmap=True
        )
        kwargs.update(
            {
                "diffuse": self.diffuse,
                "bright_catalogs": self.bright_catalogs,
                "bright_catalog_flux_min_jy": self.bright_catalog_flux_min_jy,
                "confusion_flux_range_jy": self.confusion_flux_range_jy,
                "confusion_dn_ds": self.confusion_dn_ds,
                "confusion_spectral_index_dist": self.confusion_spectral_index_dist,
                "nside": self.nside,
                "include_cmb": self.include_cmb,
                "seed": self.seed,
                "mixed_model_policy": self.mixed_model_policy,
            }
        )
        if self.diffuse_kwargs is not None:
            kwargs["diffuse_kwargs"] = dict(self.diffuse_kwargs)
        if self.bright_catalog_kwargs is not None:
            kwargs["bright_catalog_kwargs"] = dict(self.bright_catalog_kwargs)
        return self.kind, kwargs


class GleamSourceConfig(FullCatalogPointSourceConfig):
    kind: Literal["gleam"] = "gleam"
    flux_limit: NonNegativeFiniteFloat = 1.0
    catalog: str = "gleam_egc"

    @override
    def _build_loader_request(
        self, context: SkyLoaderRequestContext
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._build_catalog_kwargs(context)
        kwargs["catalog"] = self.catalog
        return self.kind, kwargs


class MalsSourceConfig(FullCatalogPointSourceConfig):
    kind: Literal["mals"] = "mals"
    flux_limit: NonNegativeFiniteFloat = 1.0
    release: str = "dr2"

    @override
    def _build_loader_request(
        self, context: SkyLoaderRequestContext
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._build_catalog_kwargs(context)
        kwargs["release"] = self.release
        return self.kind, kwargs


class LotssSourceConfig(FullCatalogPointSourceConfig):
    kind: Literal["lotss"] = "lotss"
    flux_limit: NonNegativeFiniteFloat = 0.001
    release: str = "dr2"

    @override
    def _build_loader_request(
        self, context: SkyLoaderRequestContext
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._build_catalog_kwargs(context)
        kwargs["release"] = self.release
        return self.kind, kwargs


class RacsSourceConfig(PointCatalogSourceConfig):
    kind: Literal["racs"] = "racs"
    band: str = "low"
    flux_limit: NonNegativeFiniteFloat = 1.0
    max_rows: Annotated[int, Field(ge=1)] = 1_000_000

    @override
    def _build_loader_request(
        self, context: SkyLoaderRequestContext
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._build_catalog_kwargs(context)
        kwargs.update({"band": self.band, "max_rows": self.max_rows})
        return self.kind, kwargs


class CustomRegisteredSourceConfig(SkySourceConfig):
    """Strict option envelope for a non-built-in registered loader."""

    options: SerializableMapping[Any] = Field(default_factory=FrozenDict)

    @model_validator(mode="before")
    @classmethod
    def validate_options(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        copied: dict[str, Any] = dict(cast(Mapping[str, Any], data))
        kind = copied.get("kind")
        if isinstance(kind, str):
            copied["options"] = _validate_registered_options(
                kind, copied.get("options")
            )
        return copied

    @field_validator("options")
    @classmethod
    def freeze_options(cls, value: Mapping[str, Any]) -> FrozenDict:
        return _freeze_dict(value)

    @override
    def _build_loader_request(
        self, context: SkyLoaderRequestContext
    ) -> tuple[str, dict[str, Any]]:
        from radiosim.core.sky.registry import loader_registry

        definition = loader_registry.definition(self.kind)
        kwargs = self._common_kwargs(
            context, include_frequency_context=definition.supports_healpix_map
        )
        flux_fields = {"flux_limit", "flux_min", "flux_max"}
        for loader_arg, source_field in definition.config_fields.items():
            if source_field not in self.options:
                continue
            value = self.options[source_field]
            if source_field in flux_fields:
                value = self._scaled_flux(value, context)
            kwargs[loader_arg] = value
        return self.kind, kwargs


_SKY_SOURCE_CONFIG_UNION = Annotated[
    GleamSourceConfig
    | MalsSourceConfig
    | LotssSourceConfig
    | RacsSourceConfig
    | DiffuseSkySourceConfig
    | Pysm3SourceConfig
    | PyradioskyFileSourceConfig
    | Skyh5MultifileSourceConfig
    | BbsSourceConfig
    | FitsImageSourceConfig
    | TestSourcesConfig
    | RealisticForegroundSourceConfig,
    Field(discriminator="kind"),
]
_SKY_SOURCE_CONFIG_ADAPTER: TypeAdapter[Any] = TypeAdapter(_SKY_SOURCE_CONFIG_UNION)
_BUILTIN_SKY_SOURCE_KINDS = frozenset(
    {
        "gleam",
        "mals",
        "lotss",
        "racs",
        "diffuse_sky",
        "pysm3",
        "pyradiosky_file",
        "skyh5_multifile",
        "bbs",
        "fits_image",
        "test_sources",
        "realistic_foreground",
    }
)


def parse_sky_source_config(data: Any) -> SkySourceConfig:
    """Parse one strict tagged source specification."""
    if not isinstance(data, Mapping):
        raise TypeError("sky_model.sources entries must be objects with a 'kind' field")
    copied: dict[str, Any] = dict(cast(Mapping[str, Any], data))
    if copied.get("kind") in _BUILTIN_SKY_SOURCE_KINDS:
        return cast(
            SkySourceConfig,
            _SKY_SOURCE_CONFIG_ADAPTER.validate_python(copied),
        )
    return CustomRegisteredSourceConfig.model_validate(copied)


_LEGACY_SKY_MODEL_SECTIONS = frozenset(
    {
        "bbs",
        "fits_image",
        "gleam",
        "gleam_healpix",
        "gsm_healpix",
        "lotss",
        "mals",
        "nvss",
        "pyradiosky",
        "pysm3",
        "racs",
        "sumss",
        "test_sources",
        "test_sources_healpix",
        "tgss",
        "three_c",
        "vlass",
        "vlssr",
        "wenss",
    }
)


class SkyModelConfig(StrictFrozenModel):
    """Strict immutable sky-model input."""

    sources: tuple[SerializeAsAny[SkySourceConfig], ...] = Field(min_length=1)
    flux_unit: Literal["Jy", "mJy", "uJy"] = "Jy"
    brightness_conversion: Literal["planck", "rayleigh-jeans"] = "planck"
    mixed_model_policy: Literal["error", "warn", "allow"] = "error"
    assume_disjoint: bool = False
    region: SkyRegionInput = None

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_sections(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        mapping = cast(Mapping[str, Any], data)
        legacy_sections = sorted(set(mapping) & _LEGACY_SKY_MODEL_SECTIONS)
        if legacy_sections:
            sections = ", ".join(legacy_sections)
            raise ValueError(
                "sky_model now uses only a 'sources' list. "
                f"Legacy nested section(s) are no longer accepted: {sections}. "
                "Rewrite each enabled section as an entry under sky_model.sources."
            )
        return mapping

    @field_validator("sources", mode="before")
    @classmethod
    def parse_source_specs(cls, sources: Any) -> Any:
        if isinstance(sources, (str, bytes)) or not isinstance(sources, (list, tuple)):
            return sources
        parsed: list[SkySourceConfig] = []
        errors: list[Any] = []
        for index, source in enumerate(cast(Sequence[Any], sources)):
            if isinstance(source, SkySourceConfig):
                parsed.append(source)
                continue
            try:
                parsed.append(parse_sky_source_config(source))
            except ValidationError as error:
                for item in error.errors(include_url=False):
                    item["loc"] = (index, *item.get("loc", ()))
                    errors.append(item)
        if errors:
            raise ValidationError.from_exception_data("SkyModelConfig", errors)
        return tuple(parsed)

    @field_validator("region", mode="before")
    @classmethod
    def copy_region(cls, value: Any) -> Any:
        return _copy_region_sequence(value)


class ObsTimeConfig(StrictFrozenModel):
    """Required observation start, duration, and cadence."""

    start_time: str
    duration_seconds: PositiveFiniteFloat
    time_step_seconds: PositiveFiniteFloat

    @field_validator("start_time")
    @classmethod
    def validate_start_time(cls, value: str) -> str:
        return _nonblank(value, field_name="start_time")


class FrequencyGridConfig(StrictFrozenModel):
    """Uniform frequency grid input."""

    mode: Literal["grid"] = "grid"
    starting_frequency: PositiveFiniteFloat
    frequency_interval: PositiveFiniteFloat
    frequency_bandwidth: PositiveFiniteFloat
    channel_width: PositiveFiniteFloat
    frequency_unit: Literal["Hz", "kHz", "MHz", "GHz"] = "MHz"

    @model_validator(mode="after")
    def validate_integral_interval_count(self) -> FrequencyGridConfig:
        ratio = self.frequency_bandwidth / self.frequency_interval
        nearest = round(ratio)
        if not math.isclose(ratio, nearest, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(
                "frequency_bandwidth/frequency_interval must be an integer "
                "within relative tolerance 1e-12"
            )
        return self

    @property
    def n_channels(self) -> int:
        return round(self.frequency_bandwidth / self.frequency_interval) + 1


class ExplicitFrequencyConfig(StrictFrozenModel):
    """Immutable explicit channel frequencies, always expressed in Hz."""

    mode: Literal["explicit"] = "explicit"
    channel_frequencies_hz: tuple[float, ...]
    channel_widths_hz: tuple[float, ...]

    @field_validator("channel_frequencies_hz", mode="before")
    @classmethod
    def copy_and_validate_shape(cls, value: Any) -> tuple[float, ...]:
        if isinstance(value, (str, bytes, Mapping)):
            raise ValueError(
                "channel_frequencies_hz must be a non-string one-dimensional sequence"
            )
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError("channel_frequencies_hz must be one-dimensional")
            items = value.tolist()
        else:
            try:
                items = list(value)
            except TypeError as exc:
                raise ValueError(
                    "channel_frequencies_hz must be a one-dimensional sequence"
                ) from exc
        if not items:
            raise ValueError("channel_frequencies_hz must be nonempty")
        copied: list[float] = []
        for item in items:
            if isinstance(item, (bool, np.bool_)):
                raise ValueError("channel frequencies cannot be boolean")
            if isinstance(item, (str, bytes, Mapping, Sequence)):
                raise ValueError("channel_frequencies_hz must be one-dimensional")
            try:
                frequency = float(item)
            except (TypeError, ValueError) as exc:
                raise ValueError("channel frequencies must be numeric") from exc
            if not math.isfinite(frequency) or frequency <= 0.0:
                raise ValueError("channel frequencies must be finite and positive")
            copied.append(frequency)
        if any(right <= left for left, right in zip(copied, copied[1:], strict=False)):
            raise ValueError("channel frequencies must be strictly increasing")
        return tuple(copied)

    @field_validator("channel_widths_hz", mode="before")
    @classmethod
    def copy_and_validate_widths(cls, value: Any) -> tuple[float, ...]:
        if isinstance(value, (str, bytes, Mapping)):
            raise ValueError(
                "channel_widths_hz must be a non-string one-dimensional sequence"
            )
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError("channel_widths_hz must be one-dimensional")
            items = value.tolist()
        else:
            try:
                items = list(value)
            except TypeError as exc:
                raise ValueError(
                    "channel_widths_hz must be a one-dimensional sequence"
                ) from exc
        if not items:
            raise ValueError("channel_widths_hz must be nonempty")
        copied: list[float] = []
        for item in items:
            if isinstance(item, (bool, np.bool_)):
                raise ValueError("channel widths cannot be boolean")
            if isinstance(item, (str, bytes, Mapping, Sequence)):
                raise ValueError("channel_widths_hz must be one-dimensional")
            try:
                width = float(item)
            except (TypeError, ValueError) as exc:
                raise ValueError("channel widths must be numeric") from exc
            if not math.isfinite(width) or width <= 0.0:
                raise ValueError("channel widths must be finite and positive")
            copied.append(width)
        return tuple(copied)

    @model_validator(mode="after")
    def validate_coordinate_lengths(self) -> ExplicitFrequencyConfig:
        if len(self.channel_widths_hz) != len(self.channel_frequencies_hz):
            raise ValueError(
                "channel_widths_hz must match channel_frequencies_hz length"
            )
        return self


ObsFrequencyConfig = Annotated[
    FrequencyGridConfig | ExplicitFrequencyConfig,
    Field(discriminator="mode"),
]


class VisibilityConfig(StrictFrozenModel):
    """Visibility calculation input.

    There is no ``calculation_type`` field.  It was removed before v1.0
    (``Tier7JonesSciencePlan.md`` Section 33.2, defect D13): it validated two
    values, no module in ``src/radiosim`` ever read it, and the strategy it
    appeared to select is chosen by ``execution.simulator``, whose accepted
    values equal the simulator registry keys by an asserted invariant (I15).
    A document that still sets it is rejected with removed-field guidance (R1).
    """

    sky_representation: Literal["point_sources", "healpix_map", "hybrid"] = (
        DEFAULT_SKY_REPRESENTATION
    )
    allow_lossy_point_materialization: bool = False
    #: Opt in to folding point sources into the HEALPix grid under
    #: ``sky_representation: healpix_map``.  Rasterization quantizes source
    #: positions to pixel centers, so Tier 6F made it explicit rather than
    #: silent (``Tier6HybridRuntimePlan.md`` Sections 8.2, 18.3).  Prefer
    #: ``sky_representation: hybrid``, which sums both components losslessly.
    allow_lossy_point_rasterization: bool = False


class CoordinatePrecisionInput(StrictFrozenModel):
    antenna_positions: PrecisionLevel = "float64"
    source_positions: PrecisionLevel = "float64"
    direction_cosines: PrecisionLevel = "float64"
    uvw: PrecisionLevel = "float64"


class JonesPrecisionInput(StrictFrozenModel):
    """Per-Jones-term precision input.

    One field per term letter, including the three Tier 7D added -- ``C``, ``H``
    and the extended calibration terms -- so that no term in the canonical chain
    is without a declared precision (defect D15).
    """

    geometric_phase: PrecisionLevel = "float64"
    beam: PrecisionLevel = "float64"
    ionosphere: PrecisionLevel = "float64"
    troposphere: PrecisionLevel = "float64"
    parallactic: PrecisionLevel = "float64"
    gain: PrecisionLevel = "float64"
    bandpass: PrecisionLevel = "float64"
    polarization_leakage: PrecisionLevel = "float64"
    receptor_config: PrecisionLevel = "float64"
    basis_transform: PrecisionLevel = "float64"
    crosshand: PrecisionLevel = "float64"
    delay: PrecisionLevel = "float64"
    cable_reflection: PrecisionLevel = "float64"
    baseline_multiplicative: PrecisionLevel = "float64"
    smearing: PrecisionLevel = "float64"


class SkyModelPrecisionInput(StrictFrozenModel):
    source_positions: PrecisionLevel = "float64"
    flux: PrecisionLevel = "float64"
    spectral_index: PrecisionLevel = "float64"
    healpix_maps: PrecisionLevel = "float32"


class PrecisionInput(StrictFrozenModel):
    """Preset or complete custom precision input."""

    preset: Literal["standard", "fast", "precise", "ultra"] | None = None
    default: PrecisionLevel = "float64"
    coordinates: CoordinatePrecisionInput = Field(
        default_factory=CoordinatePrecisionInput
    )
    jones: JonesPrecisionInput = Field(default_factory=JonesPrecisionInput)
    sky_model: SkyModelPrecisionInput = Field(default_factory=SkyModelPrecisionInput)
    accumulation: PrecisionLevel = "float64"
    output: PrecisionLevel = "float64"

    @property
    def has_preset_custom_contradiction(self) -> bool:
        return self.preset is not None and bool(self.model_fields_set - {"preset"})

    def float128_paths(self) -> tuple[str, ...]:
        paths: list[str] = []
        if self.preset in {"precise", "ultra"}:
            paths.append(f"preset.{self.preset}")
        for field in ("default", "accumulation", "output"):
            if getattr(self, field) == "float128":
                paths.append(field)
        for group, nested in (
            ("coordinates", self.coordinates),
            ("jones", self.jones),
            ("sky_model", self.sky_model),
        ):
            for field, value in nested.model_dump().items():
                if value == "float128":
                    paths.append(f"{group}.{field}")
        return tuple(paths)

    def to_precision_config(self):
        """Convert a non-contradictory input into frozen runtime precision."""
        from radiosim.core.precision import (
            CoordinatePrecision,
            JonesPrecision,
            PrecisionConfig,
            SkyModelPrecision,
        )

        if self.has_preset_custom_contradiction:
            raise ValueError(
                "precision preset and custom leaves are mutually exclusive"
            )
        if self.preset is not None:
            return {
                "standard": PrecisionConfig.standard,
                "fast": PrecisionConfig.fast,
                "precise": PrecisionConfig.precise,
                "ultra": PrecisionConfig.ultra,
            }[self.preset]()
        return PrecisionConfig(
            default=self.default,
            coordinates=CoordinatePrecision(**self.coordinates.model_dump()),
            jones=JonesPrecision(**self.jones.model_dump()),
            sky_model=SkyModelPrecision(**self.sky_model.model_dump()),
            accumulation=self.accumulation,
            output=self.output,
        )


_SKY_LOADING_MAX_WORKERS_GUIDANCE = (
    "execution.sky_loading.max_workers must be a positive integer or null "
    "(null means auto)."
)
_SOLVER_WORKERS_GUIDANCE = "execution.solver.workers must be a positive integer."
_SOLVER_PROCESS_EXECUTOR_GUIDANCE = (
    "execution.solver.executor=process: unsupported; the solver closure holds "
    "beam handlers and astropy objects that cannot cross a process boundary. "
    "Use execution.solver.executor=thread."
)
_REMOVED_EXECUTION_N_WORKERS_GUIDANCE = (
    "execution.n_workers: not a field; use execution.sky_loading.max_workers "
    "for sky-loader concurrency or execution.solver.workers for solver "
    "concurrency."
)
_REMOVED_EXECUTION_NUMBA_BACKEND_GUIDANCE = (
    "execution.backend=numba: removed before v1.0; the backend never compiled "
    "any kernel. Use execution.backend=dask for the NumPy/Dask backend or "
    "execution.backend=numpy."
)


def _positive_worker_count(value: Any, *, guidance: str) -> int:
    """Return one strictly positive non-boolean integer worker count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(guidance)
    return value


class SkyLoadingConfig(StrictFrozenModel):
    """Loader-side concurrency policy for sky-model acquisition."""

    max_workers: int | None = None
    executor: Literal["auto", "thread", "process"] = "auto"

    @field_validator("max_workers", mode="before")
    @classmethod
    def validate_max_workers(cls, value: Any) -> Any:
        if value is None:
            return None
        return _positive_worker_count(
            value,
            guidance=_SKY_LOADING_MAX_WORKERS_GUIDANCE,
        )


class SolverExecutionConfig(StrictFrozenModel):
    """Solver-side concurrency policy for visibility computation."""

    workers: int = 1
    executor: Literal["thread"] = "thread"

    @field_validator("workers", mode="before")
    @classmethod
    def validate_workers(cls, value: Any) -> Any:
        return _positive_worker_count(value, guidance=_SOLVER_WORKERS_GUIDANCE)

    @field_validator("executor", mode="before")
    @classmethod
    def reject_process_executor(cls, value: Any) -> Any:
        if value == "process":
            raise ValueError(_SOLVER_PROCESS_EXECUTOR_GUIDANCE)
        return value


class ExecutionConfig(StrictFrozenModel):
    """Declared execution strategy; no backend construction occurs here."""

    backend: Literal["auto", "numpy", "jax", "dask"] = "numpy"
    precision: PrecisionInput = Field(
        default_factory=lambda: PrecisionInput(preset="standard")
    )
    simulator: Literal["rime"] = "rime"
    offline: bool = False
    sky_loading: SkyLoadingConfig = Field(default_factory=SkyLoadingConfig)
    solver: SolverExecutionConfig = Field(default_factory=SolverExecutionConfig)

    @model_validator(mode="before")
    @classmethod
    def reject_removed_worker_field(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        if "n_workers" in cast(Mapping[str, object], value):
            raise ValueError(_REMOVED_EXECUTION_N_WORKERS_GUIDANCE)
        return value

    @model_validator(mode="before")
    @classmethod
    def reject_removed_numba_backend(cls, value: Any) -> Any:
        """Name the removed backend explicitly instead of listing literals.

        Without this the strict literal set would reject ``numba`` with a
        generic Pydantic enumeration error that never mentions ``dask``
        (``Tier6HybridRuntimePlan.md`` Section 18.3).
        """
        if not isinstance(value, Mapping):
            return value
        if cast(Mapping[str, object], value).get("backend") == "numba":
            raise ValueError(_REMOVED_EXECUTION_NUMBA_BACKEND_GUIDANCE)
        return value

    @field_serializer("precision")
    def serialize_precision(self, value: PrecisionInput, info: Any) -> dict[str, Any]:
        if value.preset is not None and not value.has_preset_custom_contradiction:
            return {"preset": value.preset}
        return value.model_dump(mode=info.mode)


class CliWorkflowConfig(StrictFrozenModel):
    """CLI-only workflow policy kept out of scientific runtime state."""

    output_dir: Path = Path("output")
    run_subdir: str | None = None
    result_filename: str = "visibilities"
    result_format: ResultFormat = ResultFormat.HDF5
    save_results: bool = False
    collision_policy: Literal["error", "replace", "suffix", "prompt"] = "error"
    plot_results: bool = False
    open_plots_in_browser: bool = False
    plotting_backend: Literal["bokeh", "matplotlib"] = "bokeh"
    save_log: bool = False
    visibility_phase_unit: Literal["radians", "degrees"] = "radians"

    @model_validator(mode="before")
    @classmethod
    def reject_removed_output_policy(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        mapping = cast(Mapping[str, object], value)
        guidance = {
            "overwrite": (
                "workflow.overwrite: removed before v1.0; use workflow.collision_policy"
            ),
            "skip_overwrite_confirmation": (
                "workflow.skip_overwrite_confirmation: removed before v1.0; "
                "use collision_policy=replace"
            ),
            "prompt_for_output_suffix": (
                "workflow.prompt_for_output_suffix: removed before v1.0; "
                "use collision_policy=suffix"
            ),
            "angle_unit": (
                "workflow.angle_unit: removed before v1.0; "
                "use workflow.visibility_phase_unit"
            ),
            "sky_model_frequency_hz": (
                "workflow.sky_model_frequency_hz: removed before v1.0; "
                "no Tier 4 sky renderer consumes it"
            ),
        }
        for field_name in (
            "overwrite",
            "skip_overwrite_confirmation",
            "prompt_for_output_suffix",
            "angle_unit",
            "sky_model_frequency_hz",
        ):
            if field_name in mapping:
                raise ValueError(guidance[field_name])
        if mapping.get("result_format") == "json":
            raise ValueError(
                "workflow.result_format=json: removed before v1.0; "
                "use summary_json or hdf5"
            )
        return mapping

    @field_validator("run_subdir")
    @classmethod
    def validate_run_subdir(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = _nonblank(value, field_name="run_subdir")
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("run_subdir must be one safe path component")
        if Path(value).is_absolute():
            raise ValueError("run_subdir must be one safe path component")
        return value

    @field_validator("result_filename")
    @classmethod
    def validate_result_filename(cls, value: str) -> str:
        value = _nonblank(value, field_name="result_filename")
        if value in {".", ".."} or "/" in value or "\\" in value or Path(value).suffix:
            raise ValueError("result_filename must be a safe filename stem")
        return value


from radiosim.io.beam_config import (  # noqa: E402, I001
    AnalyticBeamsConfig,
    BeamsConfig as _BeamsConfig,
    MixedBeamsConfig,
    PerAntennaFITSBeamsConfig,
    SharedFITSBeamsConfig,
)
from radiosim.io.instrument_config import (  # noqa: E402
    BaselineSelectionConfig,
    InstrumentConfig,
    InstrumentLocationConfig,
    KnownTelescopeSourceConfig,
    LayoutFileSourceConfig,
)
from radiosim.io.jones_config import (  # noqa: E402
    BandpassTermConfig,
    GainTermConfig,
    JonesConfig,
)
from radiosim.io.receptor_config import (  # noqa: E402
    ReceptorDefinitionConfig,
    ReceptorsConfig,
)

_REMOVED_BEAM_FIELD_GUIDANCE: dict[str, str] = {
    "beam_mode": (
        "Use beams.mode=analytic, shared_fits, per_antenna_fits, or mixed and "
        "select the corresponding complete tagged shape."
    ),
    "per_antenna": (
        "Use beams.mode=per_antenna_fits with beams.assignments[].antenna and "
        "beams.assignments[].beam."
    ),
    "beam_file": (
        "Use beams.mode=shared_fits with beams.beam.kind=fits and beams.beam.path, "
        "or beams.assignments[].beam.path."
    ),
    "antenna_beam_map": (
        "Use ordered beams.assignments[] entries with tagged .antenna and .beam values."
    ),
    "beam_za_max_deg": (
        "Tier 3 requires full visible-hemisphere coverage; beams.model has no "
        "partial angular-read field."
    ),
    "beam_za_buffer_deg": (
        "Tier 3 loads full angular axes; beams.model has no ZA buffer field."
    ),
    "beam_freq_buffer_hz": (
        "Tier 3 loads the full frequency axis; beams.beam has no frequency "
        "buffer field."
    ),
    "beam_peak_normalize": (
        "Use beams.beam.normalization=peak as a BeamFITS file requirement; runtime "
        "normalization is not performed."
    ),
    "beam_interp_function": (
        "Use beams.beam.angular_interpolation=bilinear and "
        "beams.beam.frequency_interpolation=cubic or linear."
    ),
    "aperture_shape": (
        "Use beams.mode=analytic and beams.model.kind=circular_aperture, "
        "rectangular_aperture, or elliptical_aperture."
    ),
    "taper": (
        "Use beams.model.taper.kind for circular_aperture or "
        "beams.model.taper_profile.kind for analytical_illumination; the old "
        "implementation ignored this value; select an active Tier 3 model."
    ),
    "edge_taper_dB": (
        "Use beams.model.taper.edge_taper_db on Gaussian, parabolic, or "
        "parabolic-squared direct circular tapers; the old implementation ignored "
        "this value; select an active Tier 3 model."
    ),
    "feed_model": (
        "Use beams.model.kind=analytical_illumination or numerical_illumination "
        "and beams.model.illumination.kind."
    ),
    "feed_computation": (
        "Use beams.model.kind=analytical_illumination or numerical_illumination; "
        "the old implementation ignored this value; select an active Tier 3 model."
    ),
    "feed_params": (
        "Use typed beams.model.illumination.focal_ratio and the selected q, "
        "b_over_lambda, or height_wavelengths field; the old implementation "
        "ignored this value; select an active Tier 3 model."
    ),
    "reflector_type": (
        "Use beams.model.reflector.kind only under analytical_illumination or "
        "numerical_illumination; the old implementation ignored this value; select "
        "an active Tier 3 model."
    ),
    "magnification": (
        "Use beams.model.reflector.kind=cassegrain with "
        "beams.model.reflector.magnification greater than 1; the old implementation "
        "ignored this value; select an active Tier 3 model."
    ),
    "aperture_params": (
        "Use beams.model.north_length_m/east_length_m for rectangular_aperture or "
        "north_diameter_m/east_diameter_m for elliptical_aperture; the old "
        "implementation ignored this value; select an active Tier 3 model."
    ),
    "use_beam_file": "Use beams.mode=shared_fits and beams.beam.path.",
    "use_different_beams": (
        "Use beams.mode=per_antenna_fits or mixed with beams.assignments[]."
    ),
    "beam_file_path": "Use beams.beam.path or beams.assignments[].beam.path.",
    "beam_files": "Use ordered beams.assignments[].beam tagged FITS sources.",
    "beams_per_antenna": (
        "Use ordered beams.assignments[].antenna and beams.assignments[].beam."
    ),
    "default_beam_id": (
        "Tier 3 has no default assignment; use a complete beams.assignments[] list."
    ),
    "beam_freq_interp": "Use beams.beam.frequency_interpolation=cubic or linear.",
    "beam_freq_buffer_mhz": (
        "Tier 3 loads the full frequency axis; beams.beam has no frequency "
        "buffer field."
    ),
    "all_beam_response": "Use beams.mode and its complete tagged model or source.",
    "beam_assignment": "Use ordered beams.assignments[] tagged entries.",
}


def _legacy_beam_fields(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    beams = value.get("beams")
    if not isinstance(beams, Mapping):
        return ()
    return tuple(sorted(set(beams) & set(_REMOVED_BEAM_FIELD_GUIDANCE)))


def _legacy_beam_error_text(fields: Sequence[str]) -> str:
    return "\n".join(
        f"beams.{field}: removed in Tier 3; {_REMOVED_BEAM_FIELD_GUIDANCE[field]}"
        for field in fields
    )


class RadioSimConfig(StrictFrozenModel):
    """Complete strict and deeply immutable user-authored document."""

    instrument: InstrumentConfig
    beams: _BeamsConfig = Field(default_factory=AnalyticBeamsConfig)
    baseline_selection: BaselineSelectionConfig = Field(
        default_factory=BaselineSelectionConfig
    )
    receptors: ReceptorsConfig = Field(default_factory=ReceptorsConfig)
    #: The Tier 7 Jones-term section.  ``None`` -- not an empty model -- is the
    #: default, because an absent section and a present-but-empty one are
    #: different statements: the first configures nothing and is the historical
    #: forward model bit for bit, and the second is rejected as R2.  A
    #: ``default_factory`` would collapse the two.
    jones: JonesConfig | None = None
    sky_model: SkyModelConfig
    obs_time: ObsTimeConfig
    obs_frequency: ObsFrequencyConfig
    visibility: VisibilityConfig = Field(default_factory=VisibilityConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    workflow: CliWorkflowConfig = Field(default_factory=CliWorkflowConfig)

    @model_validator(mode="before")
    @classmethod
    def reject_removed_beam_fields(cls, value: Any) -> Any:
        fields = _legacy_beam_fields(value)
        if fields:
            raise ValueError(_legacy_beam_error_text(fields))
        return value

    @field_validator("beams")
    @classmethod
    def require_exact_beams_model(cls, value: _BeamsConfig) -> _BeamsConfig:
        if type(value) not in (
            AnalyticBeamsConfig,
            SharedFITSBeamsConfig,
            PerAntennaFITSBeamsConfig,
            MixedBeamsConfig,
        ):
            raise ValueError("beams must be an exact Tier 3 beam mode model")
        return value


def _region_semantic_issues(
    region: SkyRegionInput,
    path: str,
) -> list[ConfigIssue]:
    if region is None:
        return []
    entries = region if isinstance(region, tuple) else (region,)
    issues: list[ConfigIssue] = []
    for index, entry in enumerate(entries):
        entry_path = f"{path}[{index}]" if isinstance(region, tuple) else path
        if entry.shape == "cone" and entry.radius_deg is None:
            issues.append(
                ConfigIssue(
                    f"{entry_path}.radius_deg",
                    "region_missing_radius",
                    "radius_deg is required when shape='cone'",
                )
            )
        if entry.shape == "box":
            for field in ("width_deg", "height_deg"):
                if getattr(entry, field) is None:
                    issues.append(
                        ConfigIssue(
                            f"{entry_path}.{field}",
                            "region_missing_dimension",
                            f"{field} is required when shape='box'",
                        )
                    )
    return issues


def _provenance_semantic_issues(
    provenance: SkyProvenanceInput | None,
    path: str,
) -> list[ConfigIssue]:
    if provenance is None:
        return []
    issues: list[ConfigIssue] = []
    completeness = provenance.flux_completeness_jy
    if completeness is not None:
        if completeness[0] < 0.0 or completeness[0] > completeness[1]:
            issues.append(
                ConfigIssue(
                    f"{path}.flux_completeness_jy",
                    "invalid_flux_completeness_range",
                    "must satisfy 0 <= minimum <= maximum",
                )
            )
        if provenance.flux_completeness_freq_hz is None:
            issues.append(
                ConfigIssue(
                    f"{path}.flux_completeness_freq_hz",
                    "missing_flux_completeness_frequency",
                    "is required when flux_completeness_jy is present",
                )
            )
    angular = provenance.angular_resolution_rad
    if angular is not None and not (0.0 <= angular[0] <= angular[1] <= math.pi):
        issues.append(
            ConfigIssue(
                f"{path}.angular_resolution_rad",
                "invalid_angular_resolution_range",
                "must satisfy 0 <= minimum <= maximum <= pi",
            )
        )
    threshold = provenance.source_subtraction_threshold_jy
    if (
        provenance.source_subtraction == SourceSubtractionStatus.ABOVE_THRESHOLD
        and threshold is None
    ):
        issues.append(
            ConfigIssue(
                f"{path}.source_subtraction_threshold_jy",
                "missing_subtraction_threshold",
                "is required for source_subtraction='above_threshold'",
            )
        )
    if threshold is not None and provenance.source_subtraction_freq_hz is None:
        issues.append(
            ConfigIssue(
                f"{path}.source_subtraction_freq_hz",
                "missing_subtraction_frequency",
                "is required when source_subtraction_threshold_jy is present",
            )
        )
    if (
        threshold is not None
        and provenance.source_subtraction == SourceSubtractionStatus.NONE
    ):
        issues.append(
            ConfigIssue(
                f"{path}.source_subtraction",
                "subtraction_threshold_contradiction",
                "cannot be 'none' when a subtraction threshold is present",
            )
        )
    footprint = provenance.coverage_footprint
    if footprint is not None:
        fraction = len(footprint.hpx_inds) / (12 * footprint.nside**2)
        implied = (
            SkyCoverage.FULL_SKY
            if math.isclose(fraction, 1.0)
            else SkyCoverage.PARTIAL_SKY
        )
        if provenance.sky_coverage not in {SkyCoverage.UNKNOWN, implied}:
            issues.append(
                ConfigIssue(
                    f"{path}.sky_coverage",
                    "footprint_coverage_contradiction",
                    f"coverage_footprint implies {implied.value!r}",
                )
            )
        if provenance.coverage_fraction is not None and not math.isclose(
            provenance.coverage_fraction, fraction
        ):
            issues.append(
                ConfigIssue(
                    f"{path}.coverage_fraction",
                    "footprint_fraction_contradiction",
                    "is inconsistent with coverage_footprint",
                )
            )
    return issues


def collect_semantic_issues(config: RadioSimConfig) -> tuple[ConfigIssue, ...]:
    """Collect every pure cross-field issue from a valid input model."""
    issues: list[ConfigIssue] = []
    if config.obs_time.time_step_seconds > config.obs_time.duration_seconds:
        issues.append(
            ConfigIssue(
                "obs_time.time_step_seconds",
                "cadence_exceeds_duration",
                "must be <= obs_time.duration_seconds",
            )
        )
    try:
        from astropy.time import Time

        _ = Time(config.obs_time.start_time)
    except Exception:
        issues.append(
            ConfigIssue(
                "obs_time.start_time",
                "invalid_start_time",
                "must be an Astropy-parseable ISO time",
            )
        )

    precision = config.execution.precision
    if precision.has_preset_custom_contradiction:
        issues.append(
            ConfigIssue(
                "execution.precision",
                "preset_custom_contradiction",
                "preset and explicit precision leaves are mutually exclusive",
            )
        )
    if config.execution.backend in {"jax", "dask"}:
        for field in precision.float128_paths():
            issues.append(
                ConfigIssue(
                    f"execution.precision.{field}",
                    "backend_precision_incompatible",
                    f"float128 is not supported by explicit {config.execution.backend!r}",
                )
            )

    source = config.instrument.source
    if (
        isinstance(source, KnownTelescopeSourceConfig)
        and source.registry_policy == "allow_network"
        and config.execution.offline
    ):
        issues.append(
            ConfigIssue(
                "instrument.source.registry_policy",
                "registry_policy_conflicts_with_offline_execution",
                "registry_policy='allow_network' requires execution.offline=false",
                "Use registry_policy='offline' or permit application network access.",
            )
        )

    issues.extend(_region_semantic_issues(config.sky_model.region, "sky_model.region"))
    for index, source in enumerate(config.sky_model.sources):
        base = f"sky_model.sources[{index}]"
        issues.extend(_region_semantic_issues(source.region, f"{base}.region"))
        issues.extend(
            _provenance_semantic_issues(
                source.provenance_override, f"{base}.provenance_override"
            )
        )
        if isinstance(source, Skyh5MultifileSourceConfig):
            if (source.file_glob is None) == (source.filenames is None):
                issues.append(
                    ConfigIssue(
                        base,
                        "skyh5_source_contradiction",
                        "specify exactly one of file_glob or filenames",
                    )
                )
            elif source.filenames == ():
                issues.append(
                    ConfigIssue(
                        f"{base}.filenames",
                        "empty_skyh5_filenames",
                        "must be nonempty",
                    )
                )
        if isinstance(source, TestSourcesConfig):
            if (
                source.flux_min is not None
                and source.flux_max is not None
                and source.flux_min > source.flux_max
            ):
                issues.append(
                    ConfigIssue(
                        f"{base}.flux_max",
                        "invalid_test_source_flux_range",
                        "must be >= flux_min",
                    )
                )
        if isinstance(source, RealisticForegroundSourceConfig):
            pair = source.confusion_flux_range_jy
            if pair is not None and not (0.0 <= pair[0] < pair[1]):
                issues.append(
                    ConfigIssue(
                        f"{base}.confusion_flux_range_jy",
                        "invalid_confusion_flux_range",
                        "must satisfy 0 <= minimum < maximum",
                    )
                )
            if source.confusion_spectral_index_dist[1] < 0.0:
                issues.append(
                    ConfigIssue(
                        f"{base}.confusion_spectral_index_dist",
                        "invalid_spectral_index_sigma",
                        "sigma must be >= 0",
                    )
                )
    return _ordered_issues(issues)


def collect_unsupported_issues(config: RadioSimConfig) -> tuple[ConfigIssue, ...]:
    """Collect every declared setting not implemented by the current runtime.

    Empty since Tier 7C.  Its one entry rejected
    ``visibility.calculation_type: spherical_harmonic``, and that field was
    removed rather than implemented (``Tier7JonesSciencePlan.md`` Section 33.2):
    a value nothing read is not a capability gap, and m-mode / spherical-harmonic
    transform solvers are register row ``SCI-004``, not a config value away.  The
    function and its ``unsupported`` issue stage stay because they are part of
    the Tier 1 validator contract that ``collect_config_issues`` composes, and a
    later capability gap declares itself here.
    """
    issues: list[ConfigIssue] = []
    return _ordered_issues(issues)


def collect_config_issues(config: RadioSimConfig) -> tuple[ConfigIssue, ...]:
    """Return semantic and unsupported issues in stable stage/path/code order."""
    return _ordered_issues(
        (*collect_semantic_issues(config), *collect_unsupported_issues(config))
    )


_REMOVED_FIELD_GUIDANCE: dict[str, tuple[str, str]] = {
    "telescope": (
        "top-level 'telescope' was removed by the Tier 2 instrument cutover",
        "Use 'instrument.source' for telescope identity and source selection.",
    ),
    "telescope_name": (
        "top-level 'telescope_name' was removed by the Tier 2 instrument cutover",
        "Use 'instrument.source.telescope_name' for local layouts or 'instrument.source.name' for a known telescope.",
    ),
    "antenna_layout": (
        "top-level 'antenna_layout' was removed by the Tier 2 instrument cutover",
        "Use 'instrument.source' with kind='layout_file'.",
    ),
    "location": (
        "top-level 'location' was removed by the Tier 2 instrument cutover",
        "Use 'instrument.location' with longitude_deg, latitude_deg, and height_m.",
    ),
    "feeds": (
        "top-level 'feeds' was replaced by the Tier 5 receptor model",
        "Use the 'receptors' section with 'default.basis', 'default.feed_rotation_deg', and 'output_basis'.",
    ),
    "receptors.default.feed_type": (
        "removed before v1.0; use 'basis'",
        "Set receptors.default.basis to 'linear' or 'circular'.",
    ),
    "receptors.default.n_feeds": (
        "removed before v1.0; every antenna has exactly two feeds",
        "Single-feed and multi-feed antennas are rejected until Tier 7 implements them.",
    ),
    "receptors.default.feed_angle_deg": (
        "removed before v1.0; use 'feed_rotation_deg'",
        "feed_rotation_deg is an offset from the nominal orientation for the selected basis.",
    ),
    "all_antenna_diameter": (
        "all_antenna_diameter was removed by the Tier 2 instrument cutover",
        "Use 'instrument.default_diameter_m'.",
    ),
    "use_different_diameters": (
        "use_different_diameters was removed by the Tier 2 instrument cutover",
        "Use typed entries under 'instrument.diameter_overrides'.",
    ),
    "diameters": (
        "the legacy diameters mapping was removed by the Tier 2 instrument cutover",
        "Use typed entries under 'instrument.diameter_overrides'.",
    ),
    "compute": (
        "top-level 'compute' was removed",
        "Move backend/offline values to 'execution'.",
    ),
    "precision": (
        "top-level 'precision' was removed",
        "Move it to 'execution.precision'.",
    ),
    "simulators": (
        "top-level 'simulators' was removed",
        "Use 'execution.simulator: rime'; cross-check behavior remains unsupported.",
    ),
    "output": (
        "top-level 'output' was removed",
        "Move CLI-only output policy to 'workflow'.",
    ),
    "visibility.calculation_type": (
        "visibility.calculation_type was removed before v1.0; the solver "
        "strategy is selected by 'execution.simulator' (currently only 'rime').",
        "Delete the key; 'execution.simulator: rime' already selects the "
        "direct-sum RIME solver, and it is the only accepted value.",
    ),
    "workflow.overwrite": (
        "workflow.overwrite: removed before v1.0; use workflow.collision_policy",
        "Use collision_policy=error, replace, suffix, or prompt.",
    ),
    "workflow.skip_overwrite_confirmation": (
        "workflow.skip_overwrite_confirmation: removed before v1.0; "
        "use collision_policy=replace",
        "Use collision_policy=replace.",
    ),
    "workflow.prompt_for_output_suffix": (
        "workflow.prompt_for_output_suffix: removed before v1.0; "
        "use collision_policy=suffix",
        "Use collision_policy=suffix.",
    ),
    "antenna_layout.fixed_HPBW": (
        "fixed_HPBW was removed because it had no live runtime reader",
        "Configure an analytic beam under 'beams'.",
    ),
    "instrument.location.ra": (
        "instrument.location.ra is not a geodetic location field",
        "Phase-center configuration belongs to the future Tier 4 result contract.",
    ),
    "instrument.location.dec": (
        "instrument.location.dec is not a geodetic location field",
        "Phase-center configuration belongs to the future Tier 4 result contract.",
    ),
}

for _removed_instrument_field in (
    "use_pyuvdata_telescope",
    "use_pyuvdata_location",
    "use_pyuvdata_antennas",
    "use_pyuvdata_diameters",
):
    _REMOVED_FIELD_GUIDANCE[_removed_instrument_field] = (
        f"{_removed_instrument_field} was removed by the Tier 2 instrument cutover",
        "Select exactly one 'instrument.source.kind' instead of combining boolean field sources.",
    )

for _removed_baseline_field in (
    "use_autocorrelations",
    "use_crosscorrelations",
    "only_selective_baseline_length",
    "selective_baseline_lengths",
    "selective_baseline_tolerance_meters",
    "trim_by_angle_ranges",
    "selective_angle_ranges_deg",
):
    _REMOVED_FIELD_GUIDANCE[f"baseline_selection.{_removed_baseline_field}"] = (
        f"baseline_selection.{_removed_baseline_field} was removed by the Tier 2 baseline cutover",
        "Use correlations, a typed length_filter, and azimuth_ranges_deg.",
    )

_KNOWN_FIELDS_BY_PARENT: dict[str, tuple[str, ...]] = {
    "": tuple(RadioSimConfig.model_fields),
    "instrument": tuple(InstrumentConfig.model_fields),
    "instrument.source": tuple(
        sorted(
            set(LayoutFileSourceConfig.model_fields)
            | set(KnownTelescopeSourceConfig.model_fields)
        )
    ),
    "instrument.location": tuple(InstrumentLocationConfig.model_fields),
    "beams": ("mode", "model", "beam", "assignments", "analytic_model"),
    "baseline_selection": tuple(BaselineSelectionConfig.model_fields),
    "receptors": tuple(ReceptorsConfig.model_fields),
    "receptors.default": tuple(ReceptorDefinitionConfig.model_fields),
    "jones": tuple(JonesConfig.model_fields),
    "jones.G": tuple(GainTermConfig.model_fields),
    "jones.B": tuple(BandpassTermConfig.model_fields),
    "sky_model": tuple(SkyModelConfig.model_fields),
    "obs_time": tuple(ObsTimeConfig.model_fields),
    "workflow": tuple(CliWorkflowConfig.model_fields),
    "execution": tuple(ExecutionConfig.model_fields),
    "execution.precision": tuple(PrecisionInput.model_fields),
    "visibility": tuple(VisibilityConfig.model_fields),
}


_BEAM_UNION_BRANCH_TAGS = frozenset(
    {
        "analytic",
        "shared_fits",
        "per_antenna_fits",
        "mixed",
        "fits",
        "uniform",
        "gaussian",
        "parabolic",
        "parabolic_squared",
        "cosine",
        "corrugated_horn",
        "open_waveguide",
        "dipole_ground_plane",
        "prime_focus",
        "cassegrain",
        "circular_aperture",
        "rectangular_aperture",
        "elliptical_aperture",
        "analytical_illumination",
        "numerical_illumination",
    }
)


def _logical_schema_location(
    location: Sequence[str | int],
) -> tuple[str | int, ...]:
    """Remove Pydantic discriminator branch labels from logical issue paths."""
    if not location or location[0] != "beams":
        return tuple(location)
    return tuple(
        item
        for index, item in enumerate(location)
        if index == 0 or not (isinstance(item, str) and item in _BEAM_UNION_BRANCH_TAGS)
    )


def _removed_beam_schema_issues(data: Mapping[str, Any]) -> tuple[ConfigIssue, ...]:
    return tuple(
        ConfigIssue(
            f"beams.{field}",
            "removed_field",
            "removed in Tier 3; old flat and BeamManager inputs are not accepted",
            _REMOVED_BEAM_FIELD_GUIDANCE[field],
            stage="schema",
            category="schema",
        )
        for field in _legacy_beam_fields(data)
    )


def _removed_workflow_schema_issues(
    data: Mapping[str, Any],
) -> tuple[ConfigIssue, ...]:
    workflow = data.get("workflow")
    if not isinstance(workflow, Mapping):
        return ()
    workflow_mapping = cast(Mapping[str, object], workflow)
    messages = {
        "overwrite": (
            "workflow.overwrite: removed before v1.0; use workflow.collision_policy"
        ),
        "skip_overwrite_confirmation": (
            "workflow.skip_overwrite_confirmation: removed before v1.0; "
            "use collision_policy=replace"
        ),
        "prompt_for_output_suffix": (
            "workflow.prompt_for_output_suffix: removed before v1.0; "
            "use collision_policy=suffix"
        ),
    }
    issues = [
        ConfigIssue(
            f"workflow.{field_name}",
            "removed_field",
            message,
            stage="schema",
            category="schema",
        )
        for field_name, message in messages.items()
        if field_name in workflow_mapping
    ]
    if workflow_mapping.get("result_format") == "json":
        issues.append(
            ConfigIssue(
                "workflow.result_format",
                "removed_value",
                "workflow.result_format=json: removed before v1.0; "
                "use summary_json or hdf5",
                stage="schema",
                category="schema",
            )
        )
    return _ordered_issues(issues)


_RECEPTOR_BASIS_GUIDANCE = (
    "input should be 'linear' or 'circular'",
    "Tier 5 supports exactly two receptor bases; elliptical and mixed-feed "
    "receptors are Tier 7.",
)
_RECEPTOR_OUTPUT_BASIS_GUIDANCE = (
    "input should be 'auto', 'linear' or 'circular'",
    "Use 'auto' for a homogeneous array; name a basis explicitly for a mixed array.",
)


def _receptor_literal_guidance(path: str) -> tuple[str, str] | None:
    """Return the exact Tier 5 message and hint for one receptor literal path."""
    if path == "receptors.output_basis":
        return _RECEPTOR_OUTPUT_BASIS_GUIDANCE
    if path == "receptors.default.basis":
        return _RECEPTOR_BASIS_GUIDANCE
    if path.startswith("receptors.overrides[") and path.endswith("].basis"):
        return _RECEPTOR_BASIS_GUIDANCE
    return None


def _dotted_path(location: Sequence[str | int]) -> str:
    path = ""
    for item in location:
        if isinstance(item, int):
            path += f"[{item}]"
        else:
            path += f".{item}" if path else item
    return path


def schema_issues_from_validation_error(
    error: ValidationError,
) -> tuple[ConfigIssue, ...]:
    """Convert Pydantic failures into actionable immutable schema issues."""
    issues: list[ConfigIssue] = []
    for item in error.errors(include_url=False):
        location = _logical_schema_location(tuple(item.get("loc", ())))
        path = _dotted_path(location)
        code = str(item.get("type", "schema_error"))
        message = str(item.get("msg", "invalid value"))
        hint: str | None = None
        if (
            path == "beams.beam_mode"
            and code == "literal_error"
            and item.get("input") in {"shared", "per_antenna"}
        ):
            message = f"legacy beam_mode={item['input']!r} is not accepted"
            hint = (
                "Use the strict tagged beams.mode values: analytic, shared_fits, "
                "per_antenna_fits, or mixed."
            )
            code = "removed_value"
        if code == "literal_error":
            receptor_guidance = _receptor_literal_guidance(path)
            if receptor_guidance is not None:
                message, hint = receptor_guidance
        if code == "extra_forbidden":
            message = "unknown or removed field"
            if path in _REMOVED_FIELD_GUIDANCE:
                message, hint = _REMOVED_FIELD_GUIDANCE[path]
                code = "removed_field"
            else:
                field = str(location[-1]) if location else ""
                parent = _dotted_path(location[:-1])
                normalized_parent = parent
                if ".sources[" in normalized_parent:
                    normalized_parent = ""
                candidates = _KNOWN_FIELDS_BY_PARENT.get(normalized_parent, ())
                matches = difflib.get_close_matches(field, candidates, n=2, cutoff=0.72)
                if len(matches) == 1:
                    hint = f"Did you mean '{matches[0]}'?"
        if message.startswith("Value error, "):
            message = message.removeprefix("Value error, ")
        issues.append(
            ConfigIssue(
                path,
                code,
                message,
                hint,
                stage="schema",
                category="schema",
            )
        )
    return _ordered_issues(issues)


def collect_schema_issues(data: Mapping[str, Any]) -> tuple[ConfigIssue, ...]:
    """Validate a complete document without constructing a partial model."""
    removed_beams = _removed_beam_schema_issues(data)
    removed_workflow = _removed_workflow_schema_issues(data)
    validation_data: Mapping[str, Any] = data
    if removed_beams or removed_workflow:
        validation_copy = dict(data)
        beams = data.get("beams")
        if isinstance(beams, Mapping):
            beam_mapping = cast(Mapping[str, object], beams)
            sanitized_beams = {
                key: value
                for key, value in beam_mapping.items()
                if key not in _REMOVED_BEAM_FIELD_GUIDANCE
            }
            if "mode" not in sanitized_beams:
                sanitized_beams = {"mode": "analytic"}
            validation_copy["beams"] = sanitized_beams
        workflow = data.get("workflow")
        if isinstance(workflow, Mapping):
            workflow_mapping = cast(Mapping[str, object], workflow)
            sanitized_workflow = {
                key: value
                for key, value in workflow_mapping.items()
                if key
                not in {
                    "overwrite",
                    "skip_overwrite_confirmation",
                    "prompt_for_output_suffix",
                }
            }
            if sanitized_workflow.get("result_format") == "json":
                sanitized_workflow["result_format"] = "hdf5"
            validation_copy["workflow"] = sanitized_workflow
        validation_data = validation_copy
    try:
        _ = RadioSimConfig.model_validate(dict(validation_data))
    except ValidationError as error:
        return _ordered_issues(
            (
                *removed_beams,
                *removed_workflow,
                *schema_issues_from_validation_error(error),
            )
        )
    return _ordered_issues((*removed_beams, *removed_workflow))


_ENVIRONMENT_PATH = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")


def _yaml_parse_issue(
    path: Path,
    code: str,
    message: str,
    hint: str | None = None,
) -> ConfigIssue:
    return ConfigIssue(
        "configuration_source.config_path",
        code,
        f"{message}: {path}",
        hint,
        stage="source",
        category="source",
    )


def load_config(
    path: str | Path,
    *,
    overrides: SimulationOverrides | None = None,
    workflow_overrides: WorkflowOverrides | None = None,
    check_input_paths: bool = True,
) -> ResolvedConfiguration:
    """Load YAML and resolve it through the shared configuration pipeline."""
    from radiosim.io.config_resolution import (
        ConfigParseError,
        ConfigurationSource,
        resolve_config,
    )

    invocation_dir = Path.cwd().resolve(strict=False)
    source = ConfigurationSource.for_yaml(
        path,
        invocation_dir=invocation_dir,
        label=f"yaml:{path}",
    )
    config_path = source.config_path
    assert config_path is not None
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded: object = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        problem = getattr(error, "problem", None) or "invalid YAML syntax"
        location = ""
        if mark is not None:
            location = f" at line {mark.line + 1}, column {mark.column + 1}"
        raise ConfigParseError(
            [
                _yaml_parse_issue(
                    config_path,
                    "yaml_syntax_error",
                    f"could not parse YAML{location} ({problem})",
                    "Correct the YAML syntax and load the file again.",
                )
            ]
        ) from error
    except UnicodeError as error:
        raise ConfigParseError(
            [
                _yaml_parse_issue(
                    config_path,
                    "config_decode_error",
                    "configuration file is not valid UTF-8 text",
                )
            ]
        ) from error
    except OSError as error:
        detail = error.strerror or type(error).__name__
        raise ConfigParseError(
            [
                _yaml_parse_issue(
                    config_path,
                    "config_read_error",
                    f"could not read configuration file ({detail})",
                )
            ]
        ) from error

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ConfigParseError(
            [
                _yaml_parse_issue(
                    config_path,
                    "yaml_root_not_mapping",
                    "configuration YAML root must be a mapping",
                    "Use top-level sections such as instrument, sky_model, and obs_time.",
                )
            ]
        )
    return resolve_config(
        cast(Mapping[str, object], loaded),
        source=source,
        overrides=overrides,
        workflow_overrides=workflow_overrides,
        check_input_paths=check_input_paths,
    )


def _require_input_config(config: object) -> RadioSimConfig:
    if type(config) is not RadioSimConfig:
        raise TypeError("dump_config accepts only RadioSimConfig input models")
    return config


def dump_config(config: RadioSimConfig, path: str | Path) -> None:
    """Atomically serialize one user-input model as deterministic safe YAML."""
    from radiosim.io.config_resolution import ConfigPathError

    input_config = _require_input_config(config)

    invocation_dir = Path.cwd().resolve(strict=False)
    original = str(path)
    if _ENVIRONMENT_PATH.search(original):
        raise ConfigPathError(
            [
                ConfigIssue(
                    "path",
                    "environment_path_syntax",
                    "environment-variable syntax is not allowed in dump paths",
                    "Expand the variable before calling dump_config.",
                    stage="path",
                    category="path",
                )
            ]
        )
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = invocation_dir / destination
    destination = destination.resolve(strict=False)
    parent = destination.parent
    if not parent.exists():
        raise FileNotFoundError(
            f"dump_config destination parent does not exist: {parent}"
        )
    if not parent.is_dir():
        raise NotADirectoryError(
            f"dump_config destination parent is not a directory: {parent}"
        )
    if destination.exists() and not destination.is_file():
        raise IsADirectoryError(
            f"dump_config destination is not a regular file: {destination}"
        )

    document = input_config.model_dump(mode="json")
    serialized = yaml.safe_dump(
        document,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            _ = stream.write(serialized)
        _ = temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def create_default_config(output_path: str | Path) -> None:
    """Write a target-shape template containing explicit scientific placeholders."""
    template = {
        "instrument": {
            "source": {
                "kind": "layout_file",
                "path": "antenna_layout.txt",
                "format": "radiosim",
                "telescope_name": "Example Array",
            },
            "location": {
                "longitude_deg": 0.0,
                "latitude_deg": 0.0,
                "height_m": 0.0,
            },
            "default_diameter_m": 14.0,
        },
        "baseline_selection": {"correlations": "all"},
        "sky_model": {"sources": [{"kind": "test_sources"}]},
        "obs_time": {
            "start_time": "2025-01-01T00:00:00",
            "duration_seconds": 1.0,
            "time_step_seconds": 1.0,
        },
        "obs_frequency": {
            "mode": "explicit",
            "channel_frequencies_hz": [100_000_000.0],
            "channel_widths_hz": [1_000_000.0],
        },
        "execution": {
            "backend": "numpy",
            "precision": {"preset": "standard"},
            "simulator": "rime",
            "offline": False,
        },
        "workflow": {
            "output_dir": "output",
            "result_format": "hdf5",
            "collision_policy": "error",
            "save_results": False,
            "plot_results": False,
            "open_plots_in_browser": False,
            "save_log": False,
            "visibility_phase_unit": "radians",
        },
    }
    with Path(output_path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(template, stream, default_flow_style=False, sort_keys=False)


__all__ = [
    "BaselineSelectionConfig",
    "BbsSourceConfig",
    "CliWorkflowConfig",
    "ConfigIssue",
    "CoordinatePrecisionInput",
    "CustomRegisteredSourceConfig",
    "DiffuseSkySourceConfig",
    "ExecutionConfig",
    "ExplicitFrequencyConfig",
    "FitsImageSourceConfig",
    "FrequencyGridConfig",
    "FrozenDict",
    "GleamSourceConfig",
    "JonesPrecisionInput",
    "InstrumentConfig",
    "InstrumentLocationConfig",
    "LotssSourceConfig",
    "MalsSourceConfig",
    "ObsFrequencyConfig",
    "ObsTimeConfig",
    "PrecisionInput",
    "PyradioskyFileSourceConfig",
    "Pysm3SourceConfig",
    "RacsSourceConfig",
    "RadioSimConfig",
    "RealisticForegroundSourceConfig",
    "SkyFootprintInput",
    "SkyLoadingConfig",
    "SkyModelConfig",
    "SkyModelPrecisionInput",
    "SkyProvenanceInput",
    "SkyRegionEntryConfig",
    "SkySourceConfig",
    "Skyh5MultifileSourceConfig",
    "SolverExecutionConfig",
    "StrictFrozenModel",
    "TestSourcesConfig",
    "VisibilityConfig",
    "build_sky_region",
    "collect_config_issues",
    "collect_schema_issues",
    "collect_semantic_issues",
    "collect_unsupported_issues",
    "create_default_config",
    "dump_config",
    "load_config",
    "parse_sky_source_config",
    "schema_issues_from_validation_error",
]
