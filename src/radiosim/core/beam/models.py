"""Immutable source-resolved beam definitions and mode inputs for Tier 3B."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from radiosim.io.instrument_config import AntennaReference

_SCHEMA_VERSION = "tier3-beam-v1"
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _snapshot_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        dataclass_value = cast(Any, value)
        return {
            field.name: _snapshot_value(getattr(dataclass_value, field.name))
            for field in fields(dataclass_value)
        }
    if isinstance(value, tuple):
        return [_snapshot_value(item) for item in cast(tuple[Any, ...], value)]
    if isinstance(value, list):
        return [_snapshot_value(item) for item in cast(list[Any], value)]
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        return {str(key): _snapshot_value(item) for key, item in mapping.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("resolved beam snapshots require finite floats")
        return value
    if isinstance(value, BaseModel):
        return _snapshot_value(value.model_dump(mode="json"))
    raise TypeError(f"value of type {type(value).__name__} is not JSON-safe")


def _canonical_value(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("beam fingerprints require finite floats")
        return value.hex().lower()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        dataclass_value = cast(Any, value)
        return {
            field.name: _canonical_value(getattr(dataclass_value, field.name))
            for field in fields(dataclass_value)
        }
    if isinstance(value, (tuple, list)):
        sequence = cast(tuple[Any, ...] | list[Any], value)
        return [_canonical_value(item) for item in sequence]
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        return {str(key): _canonical_value(item) for key, item in mapping.items()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python"))
    raise TypeError(f"value of type {type(value).__name__} cannot be fingerprinted")


def _definition_fingerprint(kind: str, payload: Any) -> str:
    canonical = _canonical_value(
        {
            "schema_version": _SCHEMA_VERSION,
            "kind": kind,
            "definition": payload,
        }
    )
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_literal(value: Any, expected: str, field_name: str) -> None:
    if type(value) is not str or value != expected:
        raise ValueError(f"{field_name} must be {expected!r}")


def _require_float(
    value: Any,
    field_name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite float")
    if positive and value <= 0.0:
        raise ValueError(f"{field_name} must be > 0")
    if nonnegative and value < 0.0:
        raise ValueError(f"{field_name} must be >= 0")


def _require_exact(value: Any, allowed: tuple[type[Any], ...], field_name: str) -> None:
    if type(value) not in allowed:
        names = ", ".join(item.__name__ for item in allowed)
        raise TypeError(f"{field_name} must be an exact {names} value")


def _require_fingerprint(value: Any, field_name: str) -> None:
    if type(value) is not str or _FINGERPRINT.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _copy_exact_tuple(
    value: Any,
    allowed: tuple[type[Any], ...],
    field_name: str,
) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be an exact tuple")
    source = cast(tuple[Any, ...], value)
    copied: tuple[Any, ...] = tuple(item for item in source)
    if not copied:
        raise ValueError(f"{field_name} must be nonempty")
    for item in copied:
        _require_exact(item, allowed, f"{field_name} item")
    return copied


def _require_absolute_path(value: Any, field_name: str) -> None:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{field_name} must be an absolute normalized Path")


def _require_antenna_reference(value: Any, field_name: str) -> None:
    from radiosim.io.instrument_config import (
        AntennaNameReference,
        AntennaNumberReference,
    )

    _require_exact(value, (AntennaNumberReference, AntennaNameReference), field_name)


class _ResolvedValue:
    """Shared detached snapshot behavior and final-class enforcement."""

    __slots__ = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if any(
            base is not _ResolvedValue and issubclass(base, _ResolvedValue)
            for base in cls.__bases__
        ):
            raise TypeError("resolved beam dataclasses do not support subclassing")

    def to_snapshot(self) -> dict[str, Any]:
        """Return a detached JSON-safe ordinary mapping."""
        snapshot = _snapshot_value(self)
        if not isinstance(snapshot, dict):
            raise TypeError("resolved beam value did not serialize to a mapping")
        return cast(dict[str, Any], snapshot)


@dataclass(frozen=True, slots=True)
class ResolvedUniformTaper(_ResolvedValue):
    kind: Literal["uniform"]

    def __post_init__(self) -> None:
        _require_literal(self.kind, "uniform", "kind")


@dataclass(frozen=True, slots=True)
class ResolvedGaussianTaper(_ResolvedValue):
    kind: Literal["gaussian"]
    edge_taper_db: float

    def __post_init__(self) -> None:
        _require_literal(self.kind, "gaussian", "kind")
        _require_float(self.edge_taper_db, "edge_taper_db", nonnegative=True)


@dataclass(frozen=True, slots=True)
class ResolvedParabolicTaper(_ResolvedValue):
    kind: Literal["parabolic"]
    edge_taper_db: float

    def __post_init__(self) -> None:
        _require_literal(self.kind, "parabolic", "kind")
        _require_float(self.edge_taper_db, "edge_taper_db", nonnegative=True)


@dataclass(frozen=True, slots=True)
class ResolvedParabolicSquaredTaper(_ResolvedValue):
    kind: Literal["parabolic_squared"]
    edge_taper_db: float

    def __post_init__(self) -> None:
        _require_literal(self.kind, "parabolic_squared", "kind")
        _require_float(self.edge_taper_db, "edge_taper_db", nonnegative=True)


@dataclass(frozen=True, slots=True)
class ResolvedCosineTaper(_ResolvedValue):
    kind: Literal["cosine"]

    def __post_init__(self) -> None:
        _require_literal(self.kind, "cosine", "kind")


@dataclass(frozen=True, slots=True)
class ResolvedDerivedGaussianTaper(_ResolvedValue):
    kind: Literal["gaussian"]

    def __post_init__(self) -> None:
        _require_literal(self.kind, "gaussian", "kind")


@dataclass(frozen=True, slots=True)
class ResolvedDerivedParabolicTaper(_ResolvedValue):
    kind: Literal["parabolic"]

    def __post_init__(self) -> None:
        _require_literal(self.kind, "parabolic", "kind")


@dataclass(frozen=True, slots=True)
class ResolvedDerivedParabolicSquaredTaper(_ResolvedValue):
    kind: Literal["parabolic_squared"]

    def __post_init__(self) -> None:
        _require_literal(self.kind, "parabolic_squared", "kind")


ResolvedDirectTaper = (
    ResolvedUniformTaper
    | ResolvedGaussianTaper
    | ResolvedParabolicTaper
    | ResolvedParabolicSquaredTaper
    | ResolvedCosineTaper
)
ResolvedDerivedTaper = (
    ResolvedDerivedGaussianTaper
    | ResolvedDerivedParabolicTaper
    | ResolvedDerivedParabolicSquaredTaper
)


@dataclass(frozen=True, slots=True)
class ResolvedCorrugatedHornIllumination(_ResolvedValue):
    kind: Literal["corrugated_horn"]
    focal_ratio: float
    q: float

    def __post_init__(self) -> None:
        _require_literal(self.kind, "corrugated_horn", "kind")
        _require_float(self.focal_ratio, "focal_ratio", positive=True)
        _require_float(self.q, "q", positive=True)


@dataclass(frozen=True, slots=True)
class ResolvedOpenWaveguideIllumination(_ResolvedValue):
    kind: Literal["open_waveguide"]
    focal_ratio: float
    b_over_lambda: float

    def __post_init__(self) -> None:
        _require_literal(self.kind, "open_waveguide", "kind")
        _require_float(self.focal_ratio, "focal_ratio", positive=True)
        _require_float(self.b_over_lambda, "b_over_lambda", positive=True)


@dataclass(frozen=True, slots=True)
class ResolvedDipoleGroundPlaneIllumination(_ResolvedValue):
    kind: Literal["dipole_ground_plane"]
    focal_ratio: float
    height_wavelengths: float

    def __post_init__(self) -> None:
        _require_literal(self.kind, "dipole_ground_plane", "kind")
        _require_float(self.focal_ratio, "focal_ratio", positive=True)
        _require_float(self.height_wavelengths, "height_wavelengths", positive=True)


ResolvedIllumination = (
    ResolvedCorrugatedHornIllumination
    | ResolvedOpenWaveguideIllumination
    | ResolvedDipoleGroundPlaneIllumination
)


@dataclass(frozen=True, slots=True)
class ResolvedPrimeFocusReflector(_ResolvedValue):
    kind: Literal["prime_focus"]

    def __post_init__(self) -> None:
        _require_literal(self.kind, "prime_focus", "kind")


@dataclass(frozen=True, slots=True)
class ResolvedCassegrainReflector(_ResolvedValue):
    kind: Literal["cassegrain"]
    magnification: float

    def __post_init__(self) -> None:
        _require_literal(self.kind, "cassegrain", "kind")
        _require_float(self.magnification, "magnification", positive=True)
        if self.magnification <= 1.0:
            raise ValueError("magnification must be > 1")


ResolvedReflector = ResolvedPrimeFocusReflector | ResolvedCassegrainReflector

_DIRECT_TAPER_TYPES = (
    ResolvedUniformTaper,
    ResolvedGaussianTaper,
    ResolvedParabolicTaper,
    ResolvedParabolicSquaredTaper,
    ResolvedCosineTaper,
)
_DERIVED_TAPER_TYPES = (
    ResolvedDerivedGaussianTaper,
    ResolvedDerivedParabolicTaper,
    ResolvedDerivedParabolicSquaredTaper,
)
_ILLUMINATION_TYPES = (
    ResolvedCorrugatedHornIllumination,
    ResolvedOpenWaveguideIllumination,
    ResolvedDipoleGroundPlaneIllumination,
)
_REFLECTOR_TYPES = (ResolvedPrimeFocusReflector, ResolvedCassegrainReflector)


@dataclass(frozen=True, slots=True)
class ResolvedCircularApertureBeamModel(_ResolvedValue):
    kind: Literal["circular_aperture"]
    taper: ResolvedDirectTaper

    def __post_init__(self) -> None:
        _require_literal(self.kind, "circular_aperture", "kind")
        _require_exact(self.taper, _DIRECT_TAPER_TYPES, "taper")


@dataclass(frozen=True, slots=True)
class ResolvedRectangularApertureBeamModel(_ResolvedValue):
    kind: Literal["rectangular_aperture"]
    north_length_m: float
    east_length_m: float

    def __post_init__(self) -> None:
        _require_literal(self.kind, "rectangular_aperture", "kind")
        _require_float(self.north_length_m, "north_length_m", positive=True)
        _require_float(self.east_length_m, "east_length_m", positive=True)


@dataclass(frozen=True, slots=True)
class ResolvedEllipticalApertureBeamModel(_ResolvedValue):
    kind: Literal["elliptical_aperture"]
    north_diameter_m: float
    east_diameter_m: float

    def __post_init__(self) -> None:
        _require_literal(self.kind, "elliptical_aperture", "kind")
        _require_float(self.north_diameter_m, "north_diameter_m", positive=True)
        _require_float(self.east_diameter_m, "east_diameter_m", positive=True)


@dataclass(frozen=True, slots=True)
class ResolvedAnalyticalIlluminationBeamModel(_ResolvedValue):
    kind: Literal["analytical_illumination"]
    illumination: ResolvedIllumination
    taper_profile: ResolvedDerivedTaper
    reflector: ResolvedReflector

    def __post_init__(self) -> None:
        _require_literal(self.kind, "analytical_illumination", "kind")
        _require_exact(self.illumination, _ILLUMINATION_TYPES, "illumination")
        _require_exact(self.taper_profile, _DERIVED_TAPER_TYPES, "taper_profile")
        _require_exact(self.reflector, _REFLECTOR_TYPES, "reflector")


@dataclass(frozen=True, slots=True)
class ResolvedNumericalIlluminationBeamModel(_ResolvedValue):
    kind: Literal["numerical_illumination"]
    illumination: ResolvedIllumination
    reflector: ResolvedReflector
    n_radial: Literal[256]

    def __post_init__(self) -> None:
        _require_literal(self.kind, "numerical_illumination", "kind")
        _require_exact(self.illumination, _ILLUMINATION_TYPES, "illumination")
        _require_exact(self.reflector, _REFLECTOR_TYPES, "reflector")
        if type(self.n_radial) is not int or self.n_radial != 256:
            raise ValueError("n_radial must be exactly 256")


ResolvedAnalyticBeamModel = (
    ResolvedCircularApertureBeamModel
    | ResolvedRectangularApertureBeamModel
    | ResolvedEllipticalApertureBeamModel
    | ResolvedAnalyticalIlluminationBeamModel
    | ResolvedNumericalIlluminationBeamModel
)
_ANALYTIC_MODEL_TYPES = (
    ResolvedCircularApertureBeamModel,
    ResolvedRectangularApertureBeamModel,
    ResolvedEllipticalApertureBeamModel,
    ResolvedAnalyticalIlluminationBeamModel,
    ResolvedNumericalIlluminationBeamModel,
)


@dataclass(frozen=True, slots=True)
class ResolvedAnalyticBeamDefinition(_ResolvedValue):
    kind: Literal["analytic"]
    model: ResolvedAnalyticBeamModel
    definition_fingerprint: str

    def __post_init__(self) -> None:
        _require_literal(self.kind, "analytic", "kind")
        _require_exact(self.model, _ANALYTIC_MODEL_TYPES, "model")
        _require_fingerprint(self.definition_fingerprint, "definition_fingerprint")
        expected = _definition_fingerprint("analytic", self.model)
        if self.definition_fingerprint != expected:
            raise ValueError("definition_fingerprint does not match analytic model")


@dataclass(frozen=True, slots=True)
class ResolvedFITSBeamDefinition(_ResolvedValue):
    kind: Literal["fits"]
    path: Path
    normalization: Literal["peak"]
    angular_interpolation: Literal["bilinear"]
    frequency_interpolation: Literal["cubic", "linear"]
    path_provenance_key: str
    definition_fingerprint: str

    def __post_init__(self) -> None:
        _require_literal(self.kind, "fits", "kind")
        _require_absolute_path(self.path, "path")
        _require_literal(self.normalization, "peak", "normalization")
        _require_literal(
            self.angular_interpolation, "bilinear", "angular_interpolation"
        )
        if self.frequency_interpolation not in {"cubic", "linear"}:
            raise ValueError("frequency_interpolation must be 'cubic' or 'linear'")
        if type(self.path_provenance_key) is not str or not self.path_provenance_key:
            raise ValueError("path_provenance_key must be nonempty")
        _require_fingerprint(self.definition_fingerprint, "definition_fingerprint")
        payload = {
            "path": self.path,
            "normalization": self.normalization,
            "angular_interpolation": self.angular_interpolation,
            "frequency_interpolation": self.frequency_interpolation,
        }
        expected = _definition_fingerprint("fits", payload)
        if self.definition_fingerprint != expected:
            raise ValueError("definition_fingerprint does not match FITS definition")


@dataclass(frozen=True, slots=True)
class ResolvedAnalyticBeamChoice(_ResolvedValue):
    kind: Literal["analytic"]

    def __post_init__(self) -> None:
        _require_literal(self.kind, "analytic", "kind")


@dataclass(frozen=True, slots=True)
class ResolvedFITSBeamAssignmentInput(_ResolvedValue):
    antenna: AntennaReference
    beam: ResolvedFITSBeamDefinition

    def __post_init__(self) -> None:
        _require_antenna_reference(self.antenna, "antenna")
        _require_exact(self.beam, (ResolvedFITSBeamDefinition,), "beam")


@dataclass(frozen=True, slots=True)
class ResolvedMixedBeamAssignmentInput(_ResolvedValue):
    antenna: AntennaReference
    beam: ResolvedAnalyticBeamChoice | ResolvedFITSBeamDefinition

    def __post_init__(self) -> None:
        _require_antenna_reference(self.antenna, "antenna")
        _require_exact(
            self.beam,
            (ResolvedAnalyticBeamChoice, ResolvedFITSBeamDefinition),
            "beam",
        )


@dataclass(frozen=True, slots=True)
class ResolvedAnalyticBeamsInput(_ResolvedValue):
    mode: Literal["analytic"]
    model: ResolvedAnalyticBeamDefinition

    def __post_init__(self) -> None:
        _require_literal(self.mode, "analytic", "mode")
        _require_exact(self.model, (ResolvedAnalyticBeamDefinition,), "model")


@dataclass(frozen=True, slots=True)
class ResolvedSharedFITSBeamsInput(_ResolvedValue):
    mode: Literal["shared_fits"]
    beam: ResolvedFITSBeamDefinition

    def __post_init__(self) -> None:
        _require_literal(self.mode, "shared_fits", "mode")
        _require_exact(self.beam, (ResolvedFITSBeamDefinition,), "beam")


@dataclass(frozen=True, slots=True)
class ResolvedPerAntennaFITSBeamsInput(_ResolvedValue):
    mode: Literal["per_antenna_fits"]
    assignments: tuple[ResolvedFITSBeamAssignmentInput, ...]

    def __post_init__(self) -> None:
        _require_literal(self.mode, "per_antenna_fits", "mode")
        copied = _copy_exact_tuple(
            self.assignments,
            (ResolvedFITSBeamAssignmentInput,),
            "assignments",
        )
        object.__setattr__(self, "assignments", copied)


@dataclass(frozen=True, slots=True)
class ResolvedMixedBeamsInput(_ResolvedValue):
    mode: Literal["mixed"]
    analytic_model: ResolvedAnalyticBeamDefinition
    assignments: tuple[ResolvedMixedBeamAssignmentInput, ...]

    def __post_init__(self) -> None:
        _require_literal(self.mode, "mixed", "mode")
        _require_exact(
            self.analytic_model,
            (ResolvedAnalyticBeamDefinition,),
            "analytic_model",
        )
        copied = _copy_exact_tuple(
            self.assignments,
            (ResolvedMixedBeamAssignmentInput,),
            "assignments",
        )
        object.__setattr__(self, "assignments", copied)


ResolvedBeamsInput = (
    ResolvedAnalyticBeamsInput
    | ResolvedSharedFITSBeamsInput
    | ResolvedPerAntennaFITSBeamsInput
    | ResolvedMixedBeamsInput
)

_RESOLVED_BEAMS_INPUT_TYPES = (
    ResolvedAnalyticBeamsInput,
    ResolvedSharedFITSBeamsInput,
    ResolvedPerAntennaFITSBeamsInput,
    ResolvedMixedBeamsInput,
)


__all__ = [
    "ResolvedAnalyticBeamChoice",
    "ResolvedAnalyticBeamDefinition",
    "ResolvedAnalyticBeamModel",
    "ResolvedAnalyticBeamsInput",
    "ResolvedAnalyticalIlluminationBeamModel",
    "ResolvedBeamsInput",
    "ResolvedCassegrainReflector",
    "ResolvedCircularApertureBeamModel",
    "ResolvedCorrugatedHornIllumination",
    "ResolvedCosineTaper",
    "ResolvedDerivedGaussianTaper",
    "ResolvedDerivedParabolicSquaredTaper",
    "ResolvedDerivedParabolicTaper",
    "ResolvedDerivedTaper",
    "ResolvedDipoleGroundPlaneIllumination",
    "ResolvedDirectTaper",
    "ResolvedEllipticalApertureBeamModel",
    "ResolvedFITSBeamAssignmentInput",
    "ResolvedFITSBeamDefinition",
    "ResolvedGaussianTaper",
    "ResolvedIllumination",
    "ResolvedMixedBeamAssignmentInput",
    "ResolvedMixedBeamsInput",
    "ResolvedNumericalIlluminationBeamModel",
    "ResolvedOpenWaveguideIllumination",
    "ResolvedParabolicSquaredTaper",
    "ResolvedParabolicTaper",
    "ResolvedPerAntennaFITSBeamsInput",
    "ResolvedPrimeFocusReflector",
    "ResolvedRectangularApertureBeamModel",
    "ResolvedReflector",
    "ResolvedSharedFITSBeamsInput",
    "ResolvedUniformTaper",
]
