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
    BaseModel,
    ConfigDict,
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


class StrictFrozenModel(BaseModel):
    """Shared base for every concrete user-input model."""

    model_config = ConfigDict(extra="forbid", frozen=True)


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


class TelescopeConfig(StrictFrozenModel):
    """Telescope identity and currently unsupported pyuvdata opt-ins."""

    telescope_name: str = "Unknown"
    use_pyuvdata_telescope: bool = False
    use_pyuvdata_location: bool = False
    use_pyuvdata_antennas: bool = False
    use_pyuvdata_diameters: bool = False

    @field_validator("telescope_name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _nonblank(value, field_name="telescope_name")


AntennaFileFormat = Literal[
    "radiosim", "casa", "measurement_set", "uvfits", "mwa", "pyuvdata"
]


class AntennaLayoutConfig(StrictFrozenModel):
    """Required antenna-layout input and deferred heterogeneous settings."""

    antenna_positions_file: Path
    antenna_file_format: AntennaFileFormat
    all_antenna_diameter: PositiveFiniteFloat
    use_different_diameters: bool = False
    diameters: SerializableMapping[PositiveFiniteFloat] = Field(
        default_factory=FrozenDict
    )

    @field_validator("antenna_positions_file", mode="before")
    @classmethod
    def validate_antenna_path(cls, value: Any) -> Any:
        return _nonempty_path_input(value, field_name="antenna_positions_file")

    @field_validator("diameters")
    @classmethod
    def freeze_diameters(cls, value: Mapping[str, PositiveFiniteFloat]) -> FrozenDict:
        return _finite_number_map(value, positive=True)


class FeedsConfig(StrictFrozenModel):
    """Deferred receptor configuration retained for explicit rejection."""

    use_polarized_feeds: bool = False
    polarization_type: str = ""
    use_different_polarization_type: bool = False
    polarization_per_antenna: SerializableMapping[str] = Field(
        default_factory=FrozenDict
    )
    use_different_feed_types: bool = False
    all_feed_type: str = ""
    feed_types_per_antenna: SerializableMapping[str] = Field(default_factory=FrozenDict)

    @field_validator("polarization_per_antenna", "feed_types_per_antenna")
    @classmethod
    def freeze_string_maps(cls, value: Mapping[str, str]) -> FrozenDict:
        return _freeze_dict(value)


class BeamsConfig(StrictFrozenModel):
    """Analytic beam input plus explicitly unsupported future FITS controls."""

    beam_mode: Literal["analytic", "fits", "mixed"] = "analytic"
    per_antenna: bool = False
    beam_file: Path | None = None
    antenna_beam_map: SerializableMapping[Path | Literal["analytic"]] = Field(
        default_factory=FrozenDict
    )
    beam_za_max_deg: FiniteFloat | None = None
    beam_za_buffer_deg: NonNegativeFiniteFloat | None = None
    beam_freq_buffer_hz: NonNegativeFiniteFloat | None = None
    beam_peak_normalize: bool = True
    beam_interp_function: str | None = None
    aperture_shape: Literal["circular", "rectangular", "elliptical"] = "circular"
    taper: Literal[
        "uniform", "gaussian", "parabolic", "parabolic_squared", "cosine"
    ] = "gaussian"
    edge_taper_dB: NonNegativeFiniteFloat = 10.0
    feed_model: Literal[
        "none", "corrugated_horn", "open_waveguide", "dipole_ground_plane"
    ] = "none"
    feed_computation: Literal["analytical", "numerical"] = "analytical"
    feed_params: SerializableMapping[FiniteFloat] = Field(default_factory=FrozenDict)
    reflector_type: Literal["prime_focus", "cassegrain"] = "prime_focus"
    magnification: PositiveFiniteFloat = 1.0
    aperture_params: SerializableMapping[PositiveFiniteFloat] = Field(
        default_factory=FrozenDict
    )

    @field_validator("beam_file", mode="before")
    @classmethod
    def validate_beam_file(cls, value: Any) -> Any:
        if value is None:
            return None
        return _nonempty_path_input(value, field_name="beam_file")

    @field_validator("beam_interp_function")
    @classmethod
    def validate_interp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _nonblank(value, field_name="beam_interp_function")

    @field_validator("antenna_beam_map")
    @classmethod
    def freeze_beam_map(
        cls, value: Mapping[str, Path | Literal["analytic"]]
    ) -> FrozenDict:
        copied: dict[str, Path | Literal["analytic"]] = {}
        for raw_key, beam in value.items():
            key = _nonblank(str(raw_key), field_name="antenna_beam_map key")
            if isinstance(beam, Path) and str(beam) in {"", "."}:
                raise ValueError(f"antenna_beam_map[{key!r}] must be nonempty")
            copied[key] = beam
        return _freeze_dict(copied)

    @field_validator("feed_params")
    @classmethod
    def validate_feed_params(cls, value: Mapping[str, FiniteFloat]) -> FrozenDict:
        allowed = {"q", "b_over_lambda", "height_wavelengths", "focal_ratio"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown feed parameter key(s): {unknown}")
        return _finite_number_map(value)

    @field_validator("aperture_params")
    @classmethod
    def validate_aperture_params(
        cls, value: Mapping[str, PositiveFiniteFloat]
    ) -> FrozenDict:
        allowed = {"length_x", "length_y", "diameter_x", "diameter_y"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown aperture parameter key(s): {unknown}")
        return _finite_number_map(value, positive=True)


class BaselineSelectionConfig(StrictFrozenModel):
    """Deferred baseline selection with defaults matching generated baselines."""

    use_autocorrelations: bool = True
    use_crosscorrelations: bool = True
    only_selective_baseline_length: bool = False
    selective_baseline_lengths: tuple[PositiveFiniteFloat, ...] = ()
    selective_baseline_tolerance_meters: NonNegativeFiniteFloat = 0.5
    trim_by_angle_ranges: bool = False
    selective_angle_ranges_deg: tuple[tuple[FiniteFloat, FiniteFloat], ...] = ()


class LocationConfig(StrictFrozenModel):
    """Required finite observatory location."""

    lat: Annotated[float, Field(ge=-90.0, le=90.0, allow_inf_nan=False)]
    lon: Annotated[float, Field(ge=-180.0, le=180.0, allow_inf_nan=False)]
    height: NonNegativeFiniteFloat


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


ObsFrequencyConfig = Annotated[
    FrequencyGridConfig | ExplicitFrequencyConfig,
    Field(discriminator="mode"),
]


class VisibilityConfig(StrictFrozenModel):
    """Visibility calculation input."""

    calculation_type: Literal["direct_sum", "spherical_harmonic"] = "direct_sum"
    sky_representation: Literal["point_sources", "healpix_map"] = (
        DEFAULT_SKY_REPRESENTATION
    )
    allow_lossy_point_materialization: bool = False


class CoordinatePrecisionInput(StrictFrozenModel):
    antenna_positions: PrecisionLevel = "float64"
    source_positions: PrecisionLevel = "float64"
    direction_cosines: PrecisionLevel = "float64"
    uvw: PrecisionLevel = "float64"


class JonesPrecisionInput(StrictFrozenModel):
    geometric_phase: PrecisionLevel = "float64"
    beam: PrecisionLevel = "float64"
    ionosphere: PrecisionLevel = "float64"
    troposphere: PrecisionLevel = "float64"
    parallactic: PrecisionLevel = "float64"
    gain: PrecisionLevel = "float64"
    bandpass: PrecisionLevel = "float64"
    polarization_leakage: PrecisionLevel = "float64"


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


class ExecutionConfig(StrictFrozenModel):
    """Declared execution strategy; no backend construction occurs here."""

    backend: Literal["auto", "numpy", "jax", "numba"] = "numpy"
    precision: PrecisionInput = Field(
        default_factory=lambda: PrecisionInput(preset="standard")
    )
    simulator: Literal["rime"] = "rime"
    offline: bool = False

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
    result_format: Literal["hdf5", "json", "ms", "uvfits"] = "hdf5"
    save_results: bool = False
    overwrite: bool = False
    skip_overwrite_confirmation: bool = False
    prompt_for_output_suffix: bool = False
    plot_results: bool = False
    open_plots_in_browser: bool = False
    plotting_backend: Literal["bokeh", "matplotlib"] = "bokeh"
    save_log: bool = False
    angle_unit: Literal["degrees", "radians", ""] = ""
    sky_model_frequency_hz: PositiveFiniteFloat | None = None

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


class RadioSimConfig(StrictFrozenModel):
    """Complete strict and deeply immutable user-authored document."""

    telescope: TelescopeConfig = Field(default_factory=TelescopeConfig)
    antenna_layout: AntennaLayoutConfig
    feeds: FeedsConfig = Field(default_factory=FeedsConfig)
    beams: BeamsConfig = Field(default_factory=BeamsConfig)
    baseline_selection: BaselineSelectionConfig = Field(
        default_factory=BaselineSelectionConfig
    )
    location: LocationConfig
    sky_model: SkyModelConfig
    obs_time: ObsTimeConfig
    obs_frequency: ObsFrequencyConfig
    visibility: VisibilityConfig = Field(default_factory=VisibilityConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    workflow: CliWorkflowConfig = Field(default_factory=CliWorkflowConfig)


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

    beams = config.beams
    expected_aperture_keys: set[str] = {
        "circular": set(),
        "rectangular": {"length_x", "length_y"},
        "elliptical": {"diameter_x", "diameter_y"},
    }[beams.aperture_shape]
    missing_aperture = sorted(expected_aperture_keys - set(beams.aperture_params))
    unexpected_aperture = sorted(set(beams.aperture_params) - expected_aperture_keys)
    for field in missing_aperture:
        issues.append(
            ConfigIssue(
                f"beams.aperture_params.{field}",
                "missing_aperture_parameter",
                f"is required for aperture_shape={beams.aperture_shape!r}",
            )
        )
    for field in unexpected_aperture:
        issues.append(
            ConfigIssue(
                f"beams.aperture_params.{field}",
                "inapplicable_aperture_parameter",
                f"is not valid for aperture_shape={beams.aperture_shape!r}",
            )
        )
    expected_feed_keys: set[str] = {
        "none": set(),
        "corrugated_horn": {"focal_ratio", "q"},
        "open_waveguide": {"focal_ratio", "b_over_lambda"},
        "dipole_ground_plane": {"focal_ratio", "height_wavelengths"},
    }[beams.feed_model]
    if beams.feed_model != "none" and "focal_ratio" not in beams.feed_params:
        issues.append(
            ConfigIssue(
                "beams.feed_params.focal_ratio",
                "missing_focal_ratio",
                "is required when feed_model is not 'none'",
            )
        )
    for field in sorted(set(beams.feed_params) - expected_feed_keys):
        issues.append(
            ConfigIssue(
                f"beams.feed_params.{field}",
                "inapplicable_feed_parameter",
                f"is not valid for feed_model={beams.feed_model!r}",
            )
        )
    for field, value in beams.feed_params.items():
        if value <= 0.0:
            issues.append(
                ConfigIssue(
                    f"beams.feed_params.{field}",
                    "nonpositive_feed_parameter",
                    "must be > 0",
                )
            )
    if beams.reflector_type == "cassegrain" and beams.magnification <= 1.0:
        issues.append(
            ConfigIssue(
                "beams.magnification",
                "invalid_cassegrain_magnification",
                "must be > 1 for reflector_type='cassegrain'",
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
    if config.execution.backend in {"jax", "numba"}:
        for field in precision.float128_paths():
            issues.append(
                ConfigIssue(
                    f"execution.precision.{field}",
                    "backend_precision_incompatible",
                    f"float128 is not supported by explicit {config.execution.backend!r}",
                )
            )

    if config.workflow.skip_overwrite_confirmation and not config.workflow.overwrite:
        issues.append(
            ConfigIssue(
                "workflow.skip_overwrite_confirmation",
                "overwrite_confirmation_contradiction",
                "requires workflow.overwrite=true",
                category="workflow",
            )
        )
    if (
        not config.baseline_selection.use_autocorrelations
        and not config.baseline_selection.use_crosscorrelations
    ):
        issues.append(
            ConfigIssue(
                "baseline_selection",
                "empty_baseline_selection",
                "autocorrelations and crosscorrelations cannot both be disabled",
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
    """Collect every declared setting not implemented by the current runtime."""
    issues: list[ConfigIssue] = []

    def add(path: str, code: str, message: str, hint: str | None = None) -> None:
        issues.append(
            ConfigIssue(
                path,
                code,
                message,
                hint,
                stage="unsupported",
                category="unsupported",
            )
        )

    for field in (
        "use_pyuvdata_telescope",
        "use_pyuvdata_location",
        "use_pyuvdata_antennas",
        "use_pyuvdata_diameters",
    ):
        if getattr(config.telescope, field):
            add(
                f"telescope.{field}",
                "pyuvdata_telescope_unsupported",
                "pyuvdata telescope opt-ins are not implemented until Tier 2",
            )
    if config.antenna_layout.use_different_diameters:
        add(
            "antenna_layout.use_different_diameters",
            "heterogeneous_diameters_unsupported",
            "per-antenna diameters are not implemented until Tier 2",
        )
    if config.antenna_layout.diameters:
        add(
            "antenna_layout.diameters",
            "heterogeneous_diameters_unsupported",
            "per-antenna diameters are not implemented until Tier 2",
        )

    beams = config.beams
    if beams.beam_mode != "analytic":
        add(
            "beams.beam_mode",
            "fits_beams_unsupported",
            "only analytic beams are implemented; FITS/mixed modes belong to Tier 3",
        )
    beam_unsupported = {
        "per_antenna": beams.per_antenna,
        "beam_file": beams.beam_file is not None,
        "antenna_beam_map": bool(beams.antenna_beam_map),
        "beam_za_max_deg": beams.beam_za_max_deg is not None,
        "beam_za_buffer_deg": beams.beam_za_buffer_deg is not None,
        "beam_freq_buffer_hz": beams.beam_freq_buffer_hz is not None,
        "beam_peak_normalize": not beams.beam_peak_normalize,
        "beam_interp_function": beams.beam_interp_function is not None,
    }
    for field, enabled in beam_unsupported.items():
        if enabled:
            add(
                f"beams.{field}",
                "fits_beam_control_unsupported",
                "this FITS/per-antenna beam control is not implemented until Tier 3",
            )

    feeds = config.feeds
    feed_nondefaults = {
        "use_polarized_feeds": feeds.use_polarized_feeds,
        "polarization_type": bool(feeds.polarization_type),
        "use_different_polarization_type": feeds.use_different_polarization_type,
        "polarization_per_antenna": bool(feeds.polarization_per_antenna),
        "use_different_feed_types": feeds.use_different_feed_types,
        "all_feed_type": bool(feeds.all_feed_type),
        "feed_types_per_antenna": bool(feeds.feed_types_per_antenna),
    }
    for field, enabled in feed_nondefaults.items():
        if enabled:
            add(
                f"feeds.{field}",
                "receptor_config_unsupported",
                "top-level receptor/feed physics is not implemented until Tier 5",
            )

    baseline = config.baseline_selection
    baseline_nondefaults = {
        "use_autocorrelations": not baseline.use_autocorrelations,
        "use_crosscorrelations": not baseline.use_crosscorrelations,
        "only_selective_baseline_length": baseline.only_selective_baseline_length,
        "selective_baseline_lengths": bool(baseline.selective_baseline_lengths),
        "selective_baseline_tolerance_meters": not math.isclose(
            baseline.selective_baseline_tolerance_meters, 0.5
        ),
        "trim_by_angle_ranges": baseline.trim_by_angle_ranges,
        "selective_angle_ranges_deg": bool(baseline.selective_angle_ranges_deg),
    }
    for field, enabled in baseline_nondefaults.items():
        if enabled:
            add(
                f"baseline_selection.{field}",
                "baseline_selection_unsupported",
                "baseline selection changes are not implemented until Tier 2",
            )

    if config.visibility.calculation_type == "spherical_harmonic":
        add(
            "visibility.calculation_type",
            "spherical_harmonic_unsupported",
            "spherical-harmonic calculation is not implemented until Tier 7",
        )
    workflow = config.workflow
    if workflow.result_format == "uvfits":
        add(
            "workflow.result_format",
            "uvfits_unsupported",
            "UVFITS workflow output is not implemented until Tier 4",
        )
    if workflow.prompt_for_output_suffix:
        add(
            "workflow.prompt_for_output_suffix",
            "output_suffix_prompt_unsupported",
            "suffix prompting is not implemented until Tier 4",
        )
    if workflow.angle_unit:
        add(
            "workflow.angle_unit",
            "angle_unit_unsupported",
            "workflow angle-unit control is not implemented until Tier 4",
        )
    if workflow.sky_model_frequency_hz is not None:
        add(
            "workflow.sky_model_frequency_hz",
            "sky_model_frequency_unsupported",
            "workflow sky-model frequency control is not implemented until Tier 4",
        )
    return _ordered_issues(issues)


def collect_config_issues(config: RadioSimConfig) -> tuple[ConfigIssue, ...]:
    """Return semantic and unsupported issues in stable stage/path/code order."""
    return _ordered_issues(
        (*collect_semantic_issues(config), *collect_unsupported_issues(config))
    )


_REMOVED_FIELD_GUIDANCE: dict[str, tuple[str, str]] = {
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
    "antenna_layout.fixed_HPBW": (
        "fixed_HPBW was removed because it had no live runtime reader",
        "Configure an analytic beam under 'beams'.",
    ),
    "location.ra": (
        "location.ra was removed from the simulation input",
        "Phase-center configuration belongs to the future Tier 4 result contract.",
    ),
    "location.dec": (
        "location.dec was removed from the simulation input",
        "Phase-center configuration belongs to the future Tier 4 result contract.",
    ),
    "telescope.name": (
        "telescope.name is not part of the RadioSim input schema",
        "Did you mean 'telescope_name'?",
    ),
    "telescope.location": (
        "nested telescope.location is a stale example shape",
        "Use the required top-level 'location' section.",
    ),
    "antenna_layout.file": (
        "antenna_layout.file is a stale example field",
        "Did you mean 'antenna_positions_file'?",
    ),
    "antenna_layout.format": (
        "antenna_layout.format is a stale example field",
        "Did you mean 'antenna_file_format'?",
    ),
}

for _legacy_beam_field in (
    "use_beam_file",
    "use_different_beams",
    "beam_file_path",
    "beam_files",
    "beams_per_antenna",
    "default_beam_id",
    "beam_freq_interp",
    "beam_freq_buffer_mhz",
    "all_beam_response",
    "beam_assignment",
):
    _REMOVED_FIELD_GUIDANCE[f"beams.{_legacy_beam_field}"] = (
        f"beams.{_legacy_beam_field} is a removed legacy beam field",
        "Use only the strict BeamsConfig fields; no BeamManager compatibility keys are accepted.",
    )

_KNOWN_FIELDS_BY_PARENT: dict[str, tuple[str, ...]] = {
    "": tuple(RadioSimConfig.model_fields),
    "telescope": tuple(TelescopeConfig.model_fields),
    "antenna_layout": tuple(AntennaLayoutConfig.model_fields),
    "feeds": tuple(FeedsConfig.model_fields),
    "beams": tuple(BeamsConfig.model_fields),
    "baseline_selection": tuple(BaselineSelectionConfig.model_fields),
    "location": tuple(LocationConfig.model_fields),
    "sky_model": tuple(SkyModelConfig.model_fields),
    "obs_time": tuple(ObsTimeConfig.model_fields),
    "workflow": tuple(CliWorkflowConfig.model_fields),
    "execution": tuple(ExecutionConfig.model_fields),
    "execution.precision": tuple(PrecisionInput.model_fields),
    "visibility": tuple(VisibilityConfig.model_fields),
}


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
        location = tuple(item.get("loc", ()))
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
                "Use 'analytic' for the current shared analytic-beam input; "
                "FITS/per-antenna behavior remains unsupported until Tier 3."
            )
            code = "removed_value"
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
    try:
        _ = RadioSimConfig.model_validate(dict(data))
    except ValidationError as error:
        return schema_issues_from_validation_error(error)
    return ()


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
                    "Use top-level section names such as antenna_layout and location.",
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
    if not isinstance(config, RadioSimConfig):
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
        "antenna_layout": {
            "antenna_positions_file": "antenna_layout.txt",
            "antenna_file_format": "radiosim",
            "all_antenna_diameter": 14.0,
        },
        "location": {"lat": 0.0, "lon": 0.0, "height": 0.0},
        "sky_model": {"sources": [{"kind": "test_sources"}]},
        "obs_time": {
            "start_time": "2025-01-01T00:00:00",
            "duration_seconds": 1.0,
            "time_step_seconds": 1.0,
        },
        "obs_frequency": {
            "mode": "explicit",
            "channel_frequencies_hz": [100_000_000.0],
        },
        "execution": {
            "backend": "numpy",
            "precision": {"preset": "standard"},
            "simulator": "rime",
            "offline": False,
        },
        "workflow": {
            "output_dir": "output",
            "save_results": False,
            "plot_results": False,
            "open_plots_in_browser": False,
            "save_log": False,
        },
    }
    with Path(output_path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(template, stream, default_flow_style=False, sort_keys=False)


__all__ = [
    "AntennaFileFormat",
    "AntennaLayoutConfig",
    "BaselineSelectionConfig",
    "BeamsConfig",
    "BbsSourceConfig",
    "CliWorkflowConfig",
    "ConfigIssue",
    "CoordinatePrecisionInput",
    "CustomRegisteredSourceConfig",
    "DiffuseSkySourceConfig",
    "ExecutionConfig",
    "ExplicitFrequencyConfig",
    "FeedsConfig",
    "FitsImageSourceConfig",
    "FrequencyGridConfig",
    "FrozenDict",
    "GleamSourceConfig",
    "JonesPrecisionInput",
    "LocationConfig",
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
    "SkyModelConfig",
    "SkyModelPrecisionInput",
    "SkyProvenanceInput",
    "SkyRegionEntryConfig",
    "SkySourceConfig",
    "Skyh5MultifileSourceConfig",
    "StrictFrozenModel",
    "TelescopeConfig",
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
