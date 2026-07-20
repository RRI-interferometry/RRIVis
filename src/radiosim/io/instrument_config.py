"""Strict, frozen instrument and baseline-selection input models.

These models define the active top-level instrument contract used by configuration,
resolution, CLI, and Simulator entry points.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from radiosim.io.config import StrictFrozenModel

_ENVIRONMENT_PATH = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")
_LOCAL_LAYOUT_FORMATS = frozenset({"radiosim", "casa_loc", "mwa_metafits"})
_MAX_ANTENNA_NUMBER = 2_147_483_647

_StrictFiniteFloat = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False),
]
_StrictPositiveFiniteFloat = Annotated[
    float,
    Field(strict=True, gt=0.0, allow_inf_nan=False),
]
_StrictNonNegativeFiniteFloat = Annotated[
    float,
    Field(strict=True, ge=0.0, allow_inf_nan=False),
]
_StrictAntennaNumber = Annotated[
    int,
    Field(strict=True, ge=0, le=_MAX_ANTENNA_NUMBER),
]
_StrictAxialAzimuth = Annotated[
    float,
    Field(strict=True, ge=0.0, lt=180.0, allow_inf_nan=False),
]


def _normalize_name(value: Any, *, field_name: str) -> str:
    """Return a stripped, NFC-normalized, case-preserving identity."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized:
        raise ValueError(f"{field_name} must be nonblank")
    return normalized


def _validate_layout_path(value: Any) -> Any:
    """Validate path syntax without expanding, resolving, or reading it."""
    if isinstance(value, (str, Path)):
        raw = str(value)
        if not raw.strip():
            raise ValueError("path must be a nonempty path")
        if _ENVIRONMENT_PATH.search(raw):
            raise ValueError("environment-variable syntax is not allowed in path")
    return value


class LayoutFileSourceConfig(StrictFrozenModel):
    """One retained local file or dataset instrument source.

    Parameters
    ----------
    kind
        Discriminator for a layout-file source.
    path
        Unresolved user-authored path. No filesystem access occurs here.
    format
        Exact retained source-format contract.
    telescope_name
        Explicit normalized instrument identity where required by the format.
    """

    kind: Literal["layout_file"] = "layout_file"
    path: Path
    format: Literal[
        "radiosim",
        "casa_loc",
        "measurement_set",
        "uvfits",
        "mwa_metafits",
    ]
    telescope_name: str | None = None

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, value: Any) -> Any:
        return _validate_layout_path(value)

    @field_validator("telescope_name", mode="before")
    @classmethod
    def normalize_telescope_name(cls, value: Any) -> str | None:
        if value is None:
            return None
        return _normalize_name(value, field_name="telescope_name")

    @model_validator(mode="after")
    def require_local_telescope_name(self) -> Self:
        if self.format in _LOCAL_LAYOUT_FORMATS and self.telescope_name is None:
            raise ValueError(
                f"telescope_name is required for layout format {self.format!r}"
            )
        return self


class KnownTelescopeSourceConfig(StrictFrozenModel):
    """A structurally valid request for a known telescope.

    Parameters
    ----------
    kind
        Discriminator for a known-telescope source.
    name
        Requested normalized telescope name. Registry lookup occurs later.
    registry_policy
        Whether the future loader must remain offline or may use the network.
    """

    kind: Literal["known_telescope"] = "known_telescope"
    name: str
    registry_policy: Literal["offline", "allow_network"] = "offline"

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: Any) -> str:
        return _normalize_name(value, field_name="name")


InstrumentSourceConfig = Annotated[
    LayoutFileSourceConfig | KnownTelescopeSourceConfig,
    Field(discriminator="kind"),
]


class InstrumentLocationConfig(StrictFrozenModel):
    """Optional explicit finite Earth-location input.

    Parameters
    ----------
    longitude_deg
        Geodetic longitude in degrees.
    latitude_deg
        Geodetic latitude in degrees.
    height_m
        Finite height in metres; negative values are valid.
    """

    longitude_deg: _StrictFiniteFloat
    latitude_deg: _StrictFiniteFloat
    height_m: _StrictFiniteFloat


class AntennaNumberReference(StrictFrozenModel):
    """Tagged reference to an antenna by its canonical integer number."""

    kind: Literal["number"] = "number"
    number: _StrictAntennaNumber


class AntennaNameReference(StrictFrozenModel):
    """Tagged reference to an antenna by its normalized case-sensitive name."""

    kind: Literal["name"] = "name"
    name: str

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: Any) -> str:
        return _normalize_name(value, field_name="name")


AntennaReference = Annotated[
    AntennaNumberReference | AntennaNameReference,
    Field(discriminator="kind"),
]


class AntennaDiameterOverrideConfig(StrictFrozenModel):
    """One strict positive diameter override for a tagged antenna reference."""

    antenna: AntennaReference
    diameter_m: _StrictPositiveFiniteFloat


class InstrumentConfig(StrictFrozenModel):
    """Frozen Tier 2 instrument input before any source resolution.

    Parameters
    ----------
    source
        Exactly one discriminated source of positions and identities.
    location
        Optional explicit location, required for local-layout formats.
    default_diameter_m
        Optional strict positive fallback diameter.
    diameter_overrides
        Ordered, immutable tagged per-antenna diameter overrides.
    """

    source: InstrumentSourceConfig
    location: InstrumentLocationConfig | None = None
    default_diameter_m: _StrictPositiveFiniteFloat | None = None
    diameter_overrides: tuple[AntennaDiameterOverrideConfig, ...] = ()

    @model_validator(mode="after")
    def require_local_layout_location(self) -> Self:
        if (
            isinstance(self.source, LayoutFileSourceConfig)
            and self.source.format in _LOCAL_LAYOUT_FORMATS
            and self.location is None
        ):
            raise ValueError(
                f"location is required for layout format {self.source.format!r}"
            )
        return self


class LengthTargetsConfig(StrictFrozenModel):
    """Ordered exact baseline-length targets with a matching tolerance."""

    mode: Literal["targets"] = "targets"
    targets_m: tuple[_StrictNonNegativeFiniteFloat, ...] = Field(min_length=1)
    tolerance_m: _StrictNonNegativeFiniteFloat

    @model_validator(mode="after")
    def reject_duplicate_targets(self) -> Self:
        if len(set(self.targets_m)) != len(self.targets_m):
            raise ValueError("targets_m must not contain exact duplicates")
        return self


class LengthRangeConfig(StrictFrozenModel):
    """One inclusive nonnegative baseline-length range in metres."""

    min_m: _StrictNonNegativeFiniteFloat
    max_m: _StrictNonNegativeFiniteFloat

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.max_m < self.min_m:
            raise ValueError("max_m must be greater than or equal to min_m")
        return self


class LengthRangesConfig(StrictFrozenModel):
    """Ordered nonempty union of baseline-length ranges."""

    mode: Literal["ranges"] = "ranges"
    ranges_m: tuple[LengthRangeConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_ranges(self) -> Self:
        pairs = tuple((item.min_m, item.max_m) for item in self.ranges_m)
        if len(set(pairs)) != len(pairs):
            raise ValueError("ranges_m must not contain exact duplicates")
        return self


LengthFilterConfig = Annotated[
    LengthTargetsConfig | LengthRangesConfig,
    Field(discriminator="mode"),
]


class AzimuthRangeConfig(StrictFrozenModel):
    """One normal or wrapped axial-azimuth range on ``[0, 180)`` degrees."""

    start_deg: _StrictAxialAzimuth
    end_deg: _StrictAxialAzimuth

    @model_validator(mode="after")
    def reject_equal_endpoints(self) -> Self:
        if self.start_deg == self.end_deg:
            raise ValueError("start_deg and end_deg must differ")
        return self


class BaselineSelectionConfig(StrictFrozenModel):
    """Frozen Tier 2 baseline-selection criteria without execution behavior.

    Parameters
    ----------
    correlations
        Select all, cross-only, or auto-only correlations.
    length_filter
        Optional discriminated targets or ranges criteria.
    azimuth_ranges_deg
        Ordered axial-azimuth ranges; omission represents the full half-circle.
    """

    correlations: Literal["all", "cross", "auto"] = "all"
    length_filter: LengthFilterConfig | None = None
    azimuth_ranges_deg: tuple[AzimuthRangeConfig, ...] = ()

    @model_validator(mode="after")
    def reject_duplicate_azimuth_ranges(self) -> Self:
        pairs = tuple(
            (item.start_deg, item.end_deg) for item in self.azimuth_ranges_deg
        )
        if len(set(pairs)) != len(pairs):
            raise ValueError("azimuth_ranges_deg must not contain exact duplicates")
        return self


__all__ = [
    "AntennaDiameterOverrideConfig",
    "AntennaNameReference",
    "AntennaNumberReference",
    "AntennaReference",
    "AzimuthRangeConfig",
    "BaselineSelectionConfig",
    "InstrumentConfig",
    "InstrumentLocationConfig",
    "InstrumentSourceConfig",
    "KnownTelescopeSourceConfig",
    "LayoutFileSourceConfig",
    "LengthFilterConfig",
    "LengthRangeConfig",
    "LengthRangesConfig",
    "LengthTargetsConfig",
]
