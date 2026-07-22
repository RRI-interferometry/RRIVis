"""Immutable source-resolved beam definitions and mode inputs for Tier 3B."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel

from radiosim.core.instrument import AntennaId

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
    if type(value) is not type(Path()):
        raise TypeError(f"{field_name} must be an exact Path value")
    if not value.is_absolute():
        raise ValueError(f"{field_name} must be an absolute normalized Path")
    if value != value.resolve(strict=False):
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
        if type(self.frequency_interpolation) is not str or (
            self.frequency_interpolation not in {"cubic", "linear"}
        ):
            raise ValueError("frequency_interpolation must be 'cubic' or 'linear'")
        if (
            type(self.path_provenance_key) is not str
            or not self.path_provenance_key.strip()
        ):
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

_BEAM_DEFINITION_TYPES = (
    ResolvedAnalyticBeamDefinition,
    ResolvedFITSBeamDefinition,
)
_ASSIGNMENT_SOURCES = frozenset({"analytic_mode", "shared_mode", "explicit_assignment"})
_STATE_MODES = frozenset({"analytic", "shared_fits", "per_antenna_fits", "mixed"})


def _require_normalized_string(value: Any, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be an exact string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized:
        raise ValueError(f"{field_name} must be nonblank")
    if normalized != value:
        raise ValueError(f"{field_name} must already be stripped and NFC-normalized")
    return value


def _require_exact_integer(
    value: Any,
    field_name: str,
    *,
    positive: bool = False,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an exact integer")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return value


def _copy_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be an exact tuple")
    source = cast(tuple[Any, ...], value)
    copied = tuple(cast(str, item) for item in source)
    if not copied:
        raise ValueError(f"{field_name} must be nonempty")
    for index, item in enumerate(copied):
        _ = _require_normalized_string(item, f"{field_name}[{index}]")
    return copied


def _copy_integer_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be an exact tuple")
    source = cast(tuple[Any, ...], value)
    copied = tuple(cast(int, item) for item in source)
    if not copied:
        raise ValueError(f"{field_name} must be nonempty")
    for index, item in enumerate(copied):
        _ = _require_exact_integer(
            item,
            f"{field_name}[{index}]",
            positive=True,
        )
    return copied


@dataclass(frozen=True, slots=True)
class BeamFileProvenance(_ResolvedValue):
    """Detached immutable provenance for one validated BeamFITS transport."""

    resolved_path: Path
    size_bytes: int
    sha256: str
    pyuvdata_version: str
    beam_type: str
    antenna_type: str
    pixel_coordinate_system: str
    mount_type: str
    data_normalization: str
    feed_array: tuple[str, ...]
    x_orientation: str
    data_shape: tuple[int, ...]
    native_dtype: str
    frequency_min_hz: float
    frequency_max_hz: float
    frequency_count: int
    azimuth_step_rad: float
    zenith_angle_step_rad: float
    zenith_angle_max_rad: float
    basis_tolerance: float
    scalar_absolute_tolerance: float
    scalar_relative_tolerance: float
    normalization_absolute_tolerance: float

    def __post_init__(self) -> None:
        _require_absolute_path(self.resolved_path, "resolved_path")
        _ = _require_exact_integer(self.size_bytes, "size_bytes")
        _require_fingerprint(self.sha256, "sha256")
        for field_name in (
            "pyuvdata_version",
            "beam_type",
            "antenna_type",
            "pixel_coordinate_system",
            "mount_type",
            "data_normalization",
            "x_orientation",
            "native_dtype",
        ):
            _ = _require_normalized_string(getattr(self, field_name), field_name)

        if self.pyuvdata_version != "3.2.1":
            raise ValueError("pyuvdata_version must be the pinned '3.2.1' contract")
        expected_metadata = {
            "beam_type": "efield",
            "antenna_type": "simple",
            "pixel_coordinate_system": "az_za",
            "mount_type": "fixed",
            "data_normalization": "peak",
            "x_orientation": "east",
        }
        for field_name, expected in expected_metadata.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} must be {expected!r}")

        feed_array = _copy_string_tuple(self.feed_array, "feed_array")
        if feed_array != ("x", "y"):
            raise ValueError("feed_array must be exactly ('x', 'y')")
        data_shape = _copy_integer_tuple(self.data_shape, "data_shape")
        if len(data_shape) != 5 or data_shape[:2] != (2, 2):
            raise ValueError("data_shape must be exactly (2, 2, Nfreq, Nza, Naz)")
        if self.native_dtype not in {"complex64", "complex128"}:
            raise ValueError("native_dtype must be 'complex64' or 'complex128'")

        _require_float(self.frequency_min_hz, "frequency_min_hz", positive=True)
        _require_float(self.frequency_max_hz, "frequency_max_hz", positive=True)
        frequency_count = _require_exact_integer(
            self.frequency_count,
            "frequency_count",
            positive=True,
        )
        if self.frequency_min_hz > self.frequency_max_hz:
            raise ValueError("frequency bounds must be ordered")
        if frequency_count != data_shape[2]:
            raise ValueError("frequency_count must equal data_shape[2]")

        for field_name in (
            "azimuth_step_rad",
            "zenith_angle_step_rad",
            "zenith_angle_max_rad",
            "basis_tolerance",
            "scalar_absolute_tolerance",
            "scalar_relative_tolerance",
            "normalization_absolute_tolerance",
        ):
            _require_float(getattr(self, field_name), field_name, positive=True)
        if self.zenith_angle_max_rad < math.pi / 2.0 - 1e-10:
            raise ValueError("zenith_angle_max_rad must cover the visible hemisphere")
        if self.basis_tolerance != 1e-12:
            raise ValueError("basis_tolerance must be exactly 1e-12")

        object.__setattr__(self, "feed_array", feed_array)
        object.__setattr__(self, "data_shape", data_shape)


@dataclass(frozen=True, slots=True)
class LoadedBeamHandlerState(_ResolvedValue):
    """Detached immutable state for one standalone validated beam handler."""

    handler_id: str
    kind: Literal["fits"]
    definition_fingerprint: str
    scientific_fingerprint: str
    file: BeamFileProvenance
    voltage_feature_scale_by_frequency: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        _require_literal(self.kind, "fits", "kind")
        _require_fingerprint(self.definition_fingerprint, "definition_fingerprint")
        _require_fingerprint(self.scientific_fingerprint, "scientific_fingerprint")
        _require_exact(self.file, (BeamFileProvenance,), "file")
        self.file.__post_init__()
        expected_prefix = f"-{self.scientific_fingerprint[:12]}"
        if (
            type(self.handler_id) is not str
            or re.fullmatch(r"beam-[0-9]{4}-[0-9a-f]{12}", self.handler_id) is None
            or not self.handler_id.endswith(expected_prefix)
        ):
            raise ValueError(
                "handler_id must be beam-{ordinal:04d}-{scientific_fingerprint[:12]}"
            )
        values = self.voltage_feature_scale_by_frequency
        if type(values) is not tuple or not values:
            raise ValueError(
                "voltage_feature_scale_by_frequency must be a nonempty exact tuple"
            )
        copied: list[tuple[float, float]] = []
        previous_frequency: float | None = None
        for index, pair in enumerate(values):
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError(
                    "voltage_feature_scale_by_frequency items must be exact pairs"
                )
            frequency_hz, scale_rad = pair
            _require_float(
                frequency_hz,
                f"voltage_feature_scale_by_frequency[{index}][0]",
                positive=True,
            )
            _require_float(
                scale_rad,
                f"voltage_feature_scale_by_frequency[{index}][1]",
                positive=True,
            )
            if previous_frequency is not None and frequency_hz <= previous_frequency:
                raise ValueError(
                    "voltage feature-scale frequencies must be strictly increasing"
                )
            previous_frequency = frequency_hz
            copied.append((frequency_hz, scale_rad))
        object.__setattr__(self, "voltage_feature_scale_by_frequency", tuple(copied))


def _canonical_digest(payload: dict[str, Any]) -> str:
    canonical = _canonical_value(payload)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _copy_antenna_id(value: Any, field_name: str) -> AntennaId:
    _require_exact(value, (AntennaId,), field_name)
    canonical = cast(AntennaId, value)
    return AntennaId(canonical.number, canonical.name)


def _require_canonical_authored_name(value: Any, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be an exact string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized:
        raise ValueError(f"{field_name} must be nonblank")
    if normalized != value:
        raise ValueError(f"{field_name} must already be stripped and NFC-normalized")
    return value


@dataclass(frozen=True, slots=True)
class BeamAssignmentProvenance(_ResolvedValue):
    source: Literal["analytic_mode", "shared_mode", "explicit_assignment"]
    input_index: int | None
    authored_reference_kind: Literal["number", "name"] | None
    authored_reference_value: int | str | None
    canonical_antenna: AntennaId

    def __post_init__(self) -> None:
        if type(self.source) is not str or self.source not in _ASSIGNMENT_SOURCES:
            raise ValueError(
                "source must be 'analytic_mode', 'shared_mode', or "
                "'explicit_assignment'"
            )
        canonical_antenna = _copy_antenna_id(
            self.canonical_antenna,
            "canonical_antenna",
        )
        if self.source in {"analytic_mode", "shared_mode"}:
            if self.input_index is not None:
                raise ValueError(f"{self.source} requires input_index=None")
            if self.authored_reference_kind is not None:
                raise ValueError(f"{self.source} requires authored_reference_kind=None")
            if self.authored_reference_value is not None:
                raise ValueError(
                    f"{self.source} requires authored_reference_value=None"
                )
        else:
            if type(self.input_index) is not int or self.input_index < 0:
                raise ValueError(
                    "explicit_assignment requires a nonnegative exact input_index"
                )
            if type(self.authored_reference_kind) is not str or (
                self.authored_reference_kind not in {"number", "name"}
            ):
                raise ValueError(
                    "explicit_assignment requires reference kind 'number' or 'name'"
                )
            if self.authored_reference_kind == "number":
                if (
                    type(self.authored_reference_value) is not int
                    or self.authored_reference_value < 0
                ):
                    raise ValueError(
                        "number references require a nonnegative exact integer value"
                    )
                if self.authored_reference_value != canonical_antenna.number:
                    raise ValueError("number reference must identify canonical_antenna")
            else:
                authored_name = _require_canonical_authored_name(
                    self.authored_reference_value,
                    "authored_reference_value",
                )
                if authored_name != canonical_antenna.name:
                    raise ValueError("name reference must identify canonical_antenna")
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "canonical_antenna", canonical_antenna)


def _effective_assignment_dimensions(
    definition: ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition,
    antenna_diameter_m: float,
) -> dict[str, Any] | None:
    if type(definition) is ResolvedFITSBeamDefinition:
        return None
    model = cast(ResolvedAnalyticBeamDefinition, definition).model
    if type(model) is ResolvedCircularApertureBeamModel:
        return {"kind": "circular", "diameter_m": antenna_diameter_m}
    if type(model) is ResolvedRectangularApertureBeamModel:
        return {
            "kind": "rectangular",
            "north_length_m": model.north_length_m,
            "east_length_m": model.east_length_m,
        }
    if type(model) is ResolvedEllipticalApertureBeamModel:
        return {
            "kind": "elliptical",
            "north_diameter_m": model.north_diameter_m,
            "east_diameter_m": model.east_diameter_m,
        }
    if type(model) in {
        ResolvedAnalyticalIlluminationBeamModel,
        ResolvedNumericalIlluminationBeamModel,
    }:
        return {"kind": "circular", "diameter_m": antenna_diameter_m}
    raise TypeError("definition contains an unsupported analytic beam model")


def _assignment_fingerprint(
    antenna_id: AntennaId,
    antenna_diameter_m: float,
    definition: ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition,
) -> str:
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "resolved_beam_assignment",
        "canonical_antenna": {
            "number": antenna_id.number,
            "name": antenna_id.name,
        },
        "definition_fingerprint": definition.definition_fingerprint,
    }
    dimensions = _effective_assignment_dimensions(definition, antenna_diameter_m)
    if dimensions is not None:
        payload["effective_dimensions"] = dimensions
    return _canonical_digest(payload)


@dataclass(frozen=True, slots=True)
class ResolvedBeamAssignment(_ResolvedValue):
    antenna_id: AntennaId
    antenna_diameter_m: float
    definition: ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition
    provenance: BeamAssignmentProvenance
    assignment_fingerprint: str

    def __post_init__(self) -> None:
        antenna_id = _copy_antenna_id(self.antenna_id, "antenna_id")
        _require_float(
            self.antenna_diameter_m,
            "antenna_diameter_m",
            positive=True,
        )
        _require_exact(self.definition, _BEAM_DEFINITION_TYPES, "definition")
        _require_exact(
            self.provenance,
            (BeamAssignmentProvenance,),
            "provenance",
        )
        self.definition.__post_init__()
        self.provenance.__post_init__()
        if self.provenance.canonical_antenna != antenna_id:
            raise ValueError("provenance.canonical_antenna must equal antenna_id")
        _require_fingerprint(self.assignment_fingerprint, "assignment_fingerprint")
        expected = _assignment_fingerprint(
            antenna_id,
            self.antenna_diameter_m,
            self.definition,
        )
        if self.assignment_fingerprint != expected:
            raise ValueError(
                "assignment_fingerprint does not match canonical assignment science"
            )
        object.__setattr__(self, "antenna_id", antenna_id)


def _create_resolved_beam_assignment(  # pyright: ignore[reportUnusedFunction]
    *,
    antenna_id: AntennaId,
    antenna_diameter_m: float,
    definition: ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition,
    provenance: BeamAssignmentProvenance,
) -> ResolvedBeamAssignment:
    fingerprint = _assignment_fingerprint(
        antenna_id,
        antenna_diameter_m,
        definition,
    )
    return ResolvedBeamAssignment(
        antenna_id=antenna_id,
        antenna_diameter_m=antenna_diameter_m,
        definition=definition,
        provenance=provenance,
        assignment_fingerprint=fingerprint,
    )


def _deduplicated_definitions(
    assignments: tuple[ResolvedBeamAssignment, ...],
) -> tuple[ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition, ...]:
    seen: set[str] = set()
    unique: list[ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition] = []
    for assignment in assignments:
        fingerprint = assignment.definition.definition_fingerprint
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(assignment.definition)
    return tuple(unique)


def _state_fingerprint(
    mode: str,
    instrument_fingerprint: str,
    assignments: tuple[ResolvedBeamAssignment, ...],
    unique_definitions: tuple[
        ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition,
        ...,
    ],
) -> str:
    return _canonical_digest(
        {
            "schema_version": _SCHEMA_VERSION,
            "kind": "resolved_beam_state",
            "mode": mode,
            "instrument_fingerprint": instrument_fingerprint,
            "assignments": [
                assignment.assignment_fingerprint for assignment in assignments
            ],
            "unique_definitions": [
                definition.definition_fingerprint for definition in unique_definitions
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class ResolvedBeamState(_ResolvedValue):
    mode: Literal["analytic", "shared_fits", "per_antenna_fits", "mixed"]
    instrument_fingerprint: str
    assignments: tuple[ResolvedBeamAssignment, ...]
    unique_definitions: tuple[
        ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition,
        ...,
    ]
    state_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in _STATE_MODES:
            raise ValueError(
                "mode must be 'analytic', 'shared_fits', 'per_antenna_fits', or 'mixed'"
            )
        _require_fingerprint(self.instrument_fingerprint, "instrument_fingerprint")
        assignments = cast(
            tuple[ResolvedBeamAssignment, ...],
            _copy_exact_tuple(
                self.assignments,
                (ResolvedBeamAssignment,),
                "assignments",
            ),
        )
        unique_definitions = cast(
            tuple[ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition, ...],
            _copy_exact_tuple(
                self.unique_definitions,
                _BEAM_DEFINITION_TYPES,
                "unique_definitions",
            ),
        )
        for assignment in assignments:
            assignment.__post_init__()
        for definition in unique_definitions:
            definition.__post_init__()

        antenna_ids = tuple(assignment.antenna_id for assignment in assignments)
        numbers = tuple(antenna.number for antenna in antenna_ids)
        names = tuple(antenna.name for antenna in antenna_ids)
        if (
            len(set(antenna_ids)) != len(antenna_ids)
            or len(set(numbers)) != len(numbers)
            or len(set(names)) != len(names)
        ):
            raise ValueError("assignments contain a duplicate canonical antenna")
        if numbers != tuple(sorted(numbers)):
            raise ValueError("assignments must use canonical instrument order")

        expected_source = {
            "analytic": "analytic_mode",
            "shared_fits": "shared_mode",
            "per_antenna_fits": "explicit_assignment",
            "mixed": "explicit_assignment",
        }[self.mode]
        if any(
            assignment.provenance.source != expected_source
            for assignment in assignments
        ):
            raise ValueError(
                f"{self.mode} mode requires {expected_source!r} assignment provenance"
            )

        expected_unique = _deduplicated_definitions(assignments)
        if len(unique_definitions) != len(expected_unique) or any(
            actual is not expected
            for actual, expected in zip(
                unique_definitions, expected_unique, strict=True
            )
        ):
            raise ValueError(
                "unique_definitions must retain the first canonical assignment "
                "definition for each definition fingerprint"
            )

        if self.mode == "analytic" and (
            len(unique_definitions) != 1
            or any(
                type(assignment.definition) is not ResolvedAnalyticBeamDefinition
                for assignment in assignments
            )
        ):
            raise ValueError("analytic mode requires one analytic definition")
        if self.mode == "shared_fits" and (
            len(unique_definitions) != 1
            or any(
                type(assignment.definition) is not ResolvedFITSBeamDefinition
                for assignment in assignments
            )
        ):
            raise ValueError("shared_fits mode requires one FITS definition")
        if self.mode == "per_antenna_fits" and any(
            type(assignment.definition) is not ResolvedFITSBeamDefinition
            for assignment in assignments
        ):
            raise ValueError("per_antenna_fits mode requires FITS definitions")
        if (
            self.mode == "mixed"
            and len(
                {
                    definition.definition_fingerprint
                    for definition in unique_definitions
                    if type(definition) is ResolvedAnalyticBeamDefinition
                }
            )
            > 1
        ):
            raise ValueError("mixed mode requires at most one analytic definition")

        _require_fingerprint(self.state_fingerprint, "state_fingerprint")
        expected_fingerprint = _state_fingerprint(
            self.mode,
            self.instrument_fingerprint,
            assignments,
            unique_definitions,
        )
        if self.state_fingerprint != expected_fingerprint:
            raise ValueError(
                "state_fingerprint does not match canonical beam state science"
            )

        object.__setattr__(self, "mode", str(self.mode))
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "unique_definitions", unique_definitions)


def _create_resolved_beam_state(  # pyright: ignore[reportUnusedFunction]
    *,
    mode: Literal["analytic", "shared_fits", "per_antenna_fits", "mixed"],
    instrument_fingerprint: str,
    assignments: tuple[ResolvedBeamAssignment, ...],
    unique_definitions: tuple[
        ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition,
        ...,
    ],
) -> ResolvedBeamState:
    fingerprint = _state_fingerprint(
        mode,
        instrument_fingerprint,
        assignments,
        unique_definitions,
    )
    return ResolvedBeamState(
        mode=mode,
        instrument_fingerprint=instrument_fingerprint,
        assignments=assignments,
        unique_definitions=unique_definitions,
        state_fingerprint=fingerprint,
    )


__all__ = [
    "BeamAssignmentProvenance",
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
    "ResolvedBeamAssignment",
    "ResolvedBeamState",
    "ResolvedReflector",
    "ResolvedSharedFITSBeamsInput",
    "ResolvedUniformTaper",
]
