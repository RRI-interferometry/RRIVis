"""Immutable source-resolved beam definitions and mode inputs for Tier 3B."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel

from radiosim.core.instrument import AntennaId

if TYPE_CHECKING:
    from radiosim.io.instrument_config import AntennaReference

_SCHEMA_VERSION = "tier3-beam-v1"
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")


def _optional_block_fields(dataclass_value: Any) -> tuple[str, ...]:
    """Return the field names an *absent* optional block occupies.

    A resolved field that both defaults to ``None`` and *is* ``None`` describes
    science the configuration never authored, so it is omitted from snapshots
    and canonical payloads rather than serialized as a null. That is what keeps
    equivalent current runs with no ``beams.pointing`` and no
    ``beams.surface_error`` block free of an optional-beam snapshot distinction,
    including in ``scientific_sha256``. A required optional field, such as
    ``BeamAssignmentProvenance.input_index``, has no ``None`` default and is
    always serialized.
    """
    return tuple(
        field.name
        for field in fields(dataclass_value)
        if field.default is None and getattr(dataclass_value, field.name) is None
    )


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _snapshot_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        dataclass_value = cast(Any, value)
        omitted = _optional_block_fields(dataclass_value)
        return {
            field.name: _snapshot_value(getattr(dataclass_value, field.name))
            for field in fields(dataclass_value)
            if field.name not in omitted
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
        omitted = _optional_block_fields(dataclass_value)
        return {
            field.name: _canonical_value(getattr(dataclass_value, field.name))
            for field in fields(dataclass_value)
            if field.name not in omitted
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
    normalization: Literal["peak", "uvbeam_peak_common_v1"]
    angular_interpolation: Literal["bilinear"]
    frequency_interpolation: Literal["cubic", "linear"]
    path_provenance_key: str
    definition_fingerprint: str

    def __post_init__(self) -> None:
        _require_literal(self.kind, "fits", "kind")
        _require_absolute_path(self.path, "path")
        # SCI-005 Stage 3 (Section 5.1.1): the second literal selects the
        # full-efield accepted subset of the same file, and nothing else.
        if type(self.normalization) is not str or self.normalization not in {
            "peak",
            "uvbeam_peak_common_v1",
        }:
            raise ValueError("normalization must be 'peak' or 'uvbeam_peak_common_v1'")
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
        # The fingerprint binds only the load settings, never the path: the
        # path is filesystem transport, and hashing it would make every
        # downstream fingerprint differ between checkouts of the same science.
        # File content is bound at load time by the handler's
        # scientific_fingerprint; pre-load identity keys that must distinguish
        # distinct files compare the stored path field directly.
        payload = {
            "normalization": self.normalization,
            "angular_interpolation": self.angular_interpolation,
            "frequency_interpolation": self.frequency_interpolation,
        }
        expected = _definition_fingerprint("fits", payload)
        if self.definition_fingerprint != expected:
            raise ValueError("definition_fingerprint does not match FITS definition")


@dataclass(frozen=True, slots=True)
class ResolvedPointingOffset(_ResolvedValue):
    """One antenna's deterministic mount mispointing (Tier 7I, Section 19.2).

    The offset is a fixed rotation of the antenna's beam frame relative to the
    topocentric horizontal frame, composed as the two encoder errors of an
    alt-az mount: a rotation about the local vertical that increases azimuth by
    ``azimuth_offset_rad`` (North through East), then a tilt of
    ``elevation_offset_rad`` carrying the boresight away from the zenith. For
    RadioSim's zenith-pointed beams the composed boresight lands at topocentric
    azimuth ``azimuth_offset_rad`` and zenith angle ``elevation_offset_rad``.

    An offset of exactly zero is not representable: it resolves to ``None``, so
    that a configuration authoring one is bit-identical to one authoring nothing
    all the way down to ``assignment_fingerprint``.
    """

    azimuth_offset_rad: float
    elevation_offset_rad: float

    def __post_init__(self) -> None:
        _require_float(self.azimuth_offset_rad, "azimuth_offset_rad")
        _require_float(self.elevation_offset_rad, "elevation_offset_rad")
        if abs(self.elevation_offset_rad) > math.pi / 2.0:
            raise ValueError("elevation_offset_rad must lie in [-pi/2, pi/2]")
        if self.azimuth_offset_rad == 0.0 and self.elevation_offset_rad == 0.0:
            raise ValueError(
                "an inert pointing offset resolves to None, never to a stored zero"
            )


@dataclass(frozen=True, slots=True)
class ResolvedRuzePowerDiagnostic(_ResolvedValue):
    """One antenna's resolved ``error_beam_diagnostic`` declaration.

    ``docs/development/sci005_beam_physics_plan.md`` Section 3.4: the literal
    ``gaussian_covariance_power`` declares a real, zero-mean, jointly Gaussian,
    second-order stationary aperture-equivalent surface-error field whose
    covariance is ``sigma_h^2 exp[-(|Delta|/L)^2]``, so ``correlation_length_m``
    is that field's one-over-e correlation length.  The declaration selects an
    ensemble-power diagnostic; it never creates a deterministic error-beam
    voltage, and it never enters a cross-correlation Jones matrix.
    """

    kind: Literal["gaussian_covariance_power"]
    correlation_length_m: float

    def __post_init__(self) -> None:
        _require_literal(self.kind, "gaussian_covariance_power", "kind")
        _require_float(
            self.correlation_length_m,
            "correlation_length_m",
            positive=True,
        )


@dataclass(frozen=True, slots=True)
class ResolvedSurfaceError(_ResolvedValue):
    """One antenna's Ruze random-surface RMS, in metres.

    The power efficiency is ``eta_s = exp(-(4 pi sigma / lambda)^2)`` (Ruze
    1966); the factor applied to the *voltage* beam is its square root, so that
    a baseline of two antennas sharing this ``sigma`` loses exactly ``eta_s`` of
    power. A zero RMS resolves to ``None`` for the same reason a zero pointing
    offset does.

    ``error_beam_diagnostic`` is the optional SCI-005 Stage-1 nested
    ensemble-power declaration.  It leaves the coherent voltage meaning of
    ``rms_surface_error_m`` exactly as accepted.
    """

    rms_surface_error_m: float
    error_beam_diagnostic: ResolvedRuzePowerDiagnostic | None = None

    def __post_init__(self) -> None:
        _require_float(
            self.rms_surface_error_m,
            "rms_surface_error_m",
            positive=True,
        )
        if self.error_beam_diagnostic is not None:
            _require_exact(
                self.error_beam_diagnostic,
                (ResolvedRuzePowerDiagnostic,),
                "error_beam_diagnostic",
            )
            self.error_beam_diagnostic.__post_init__()


@dataclass(frozen=True, slots=True)
class ResolvedAntennaPointingOffset(_ResolvedValue):
    """One authored per-antenna pointing override.

    ``offset=None`` is the authored *exact zero*: it suppresses the array-wide
    default for this antenna rather than being absent from the list.
    """

    antenna: AntennaReference
    offset: ResolvedPointingOffset | None

    def __post_init__(self) -> None:
        _require_antenna_reference(self.antenna, "antenna")
        if self.offset is not None:
            _require_exact(self.offset, (ResolvedPointingOffset,), "offset")
            self.offset.__post_init__()


@dataclass(frozen=True, slots=True)
class ResolvedAntennaSurfaceError(_ResolvedValue):
    """One authored per-antenna surface-error override."""

    antenna: AntennaReference
    surface_error: ResolvedSurfaceError | None

    def __post_init__(self) -> None:
        _require_antenna_reference(self.antenna, "antenna")
        if self.surface_error is not None:
            _require_exact(
                self.surface_error,
                (ResolvedSurfaceError,),
                "surface_error",
            )
            self.surface_error.__post_init__()

    @property
    def error_beam_diagnostic(self) -> ResolvedRuzePowerDiagnostic | None:
        """Return this override's nested diagnostic, if any.

        A read-only view of ``surface_error.error_beam_diagnostic`` rather than
        a second stored field, so the override cannot carry a diagnostic that
        disagrees with the surface error it belongs to and no new key enters the
        canonical snapshot.
        """
        if self.surface_error is None:
            return None
        return self.surface_error.error_beam_diagnostic


def _copy_override_tuple(
    value: Any,
    allowed: tuple[type[Any], ...],
    field_name: str,
) -> tuple[Any, ...]:
    """Copy a possibly empty exact override tuple."""
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be an exact tuple")
    copied: tuple[Any, ...] = tuple(item for item in cast(tuple[Any, ...], value))
    for item in copied:
        _require_exact(item, allowed, f"{field_name} item")
        item.__post_init__()
    return copied


@dataclass(frozen=True, slots=True)
class ResolvedBeamPointing(_ResolvedValue):
    """The authored ``beams.pointing`` block, resolved but not yet assigned."""

    default: ResolvedPointingOffset | None
    per_antenna: tuple[ResolvedAntennaPointingOffset, ...]

    def __post_init__(self) -> None:
        if self.default is not None:
            _require_exact(self.default, (ResolvedPointingOffset,), "default")
            self.default.__post_init__()
        per_antenna = cast(
            tuple[ResolvedAntennaPointingOffset, ...],
            _copy_override_tuple(
                self.per_antenna,
                (ResolvedAntennaPointingOffset,),
                "per_antenna",
            ),
        )
        if self.default is None and not any(
            item.offset is not None for item in per_antenna
        ):
            raise ValueError(
                "a resolved pointing block must carry at least one non-zero offset"
            )
        object.__setattr__(self, "per_antenna", per_antenna)


@dataclass(frozen=True, slots=True)
class ResolvedBeamSurfaceError(_ResolvedValue):
    """The authored ``beams.surface_error`` block, resolved but not assigned."""

    default: ResolvedSurfaceError | None
    per_antenna: tuple[ResolvedAntennaSurfaceError, ...]

    def __post_init__(self) -> None:
        if self.default is not None:
            _require_exact(self.default, (ResolvedSurfaceError,), "default")
            self.default.__post_init__()
        per_antenna = cast(
            tuple[ResolvedAntennaSurfaceError, ...],
            _copy_override_tuple(
                self.per_antenna,
                (ResolvedAntennaSurfaceError,),
                "per_antenna",
            ),
        )
        if self.default is None and not any(
            item.surface_error is not None for item in per_antenna
        ):
            raise ValueError(
                "a resolved surface-error block must carry at least one non-zero RMS"
            )
        object.__setattr__(self, "per_antenna", per_antenna)


#: Section 4.1's five accepted mount literals, with ``None`` already resolved to
#: ``fixed`` by beam-assignment resolution.
SquintMountType = Literal[
    "alt-az",
    "equatorial",
    "fixed",
    "alt-az+nasmyth-l",
    "alt-az+nasmyth-r",
]

#: Section 4.1's four accepted native feed labels.
NativeFeedLabel = Literal["x", "y", "r", "l"]

_SQUINT_MOUNT_TYPES: tuple[str, ...] = (
    "alt-az",
    "equatorial",
    "fixed",
    "alt-az+nasmyth-l",
    "alt-az+nasmyth-r",
)
_NATIVE_FEED_LABELS: tuple[str, ...] = ("x", "y", "r", "l")


def _require_member(value: Any, allowed: tuple[str, ...], field_name: str) -> None:
    if type(value) is not str or value not in allowed:
        raise ValueError(f"{field_name} must be one of {allowed!r}")


def _require_squint_fields(record: Any) -> None:
    """Validate the five authored squint values every resolved record carries."""
    _require_literal(record.convention, "cotton_uson_exact_v1", "convention")
    _require_float(
        record.reference_frequency_hz,
        "reference_frequency_hz",
        positive=True,
    )
    _require_float(
        record.per_feed_offset_deg_at_reference,
        "per_feed_offset_deg_at_reference",
        positive=True,
    )
    if record.per_feed_offset_deg_at_reference >= 90.0:
        raise ValueError("per_feed_offset_deg_at_reference must lie in (0, 90)")
    _require_float(
        record.mechanical_feed_position_angle_deg,
        "mechanical_feed_position_angle_deg",
    )
    if not -180.0 < record.mechanical_feed_position_angle_deg <= 180.0:
        raise ValueError("mechanical_feed_position_angle_deg must lie in (-180, 180]")
    _require_member(
        record.positive_native_feed,
        _NATIVE_FEED_LABELS,
        "positive_native_feed",
    )


@dataclass(frozen=True, slots=True)
class ResolvedSquintRecord(_ResolvedValue):
    """One authored ``beams.squint`` record, resolved but not yet assigned.

    ``docs/development/sci005_beam_physics_plan.md`` Section 4.1: the nominal
    pointing is the midpoint of the two native feeds, ``delta_ref`` is the
    displacement of *one* hand, and the mechanical position angle is the
    physical off-axis feed direction in the antenna beam frame, measured North
    through East.  The mount literal is not here because it belongs to the
    instrument and is captured only when the record is assigned.
    """

    convention: Literal["cotton_uson_exact_v1"]
    reference_frequency_hz: float
    per_feed_offset_deg_at_reference: float
    mechanical_feed_position_angle_deg: float
    positive_native_feed: NativeFeedLabel

    def __post_init__(self) -> None:
        _require_squint_fields(self)


@dataclass(frozen=True, slots=True)
class ResolvedAntennaSquint(_ResolvedValue):
    """One authored per-antenna squint override.

    Section 4.1.1 grants no suppression form in v1, so ``squint`` is always a
    real record: an antenna that must not squint is simply not named.
    """

    antenna: AntennaReference
    squint: ResolvedSquintRecord

    def __post_init__(self) -> None:
        _require_antenna_reference(self.antenna, "antenna")
        _require_exact(self.squint, (ResolvedSquintRecord,), "squint")
        self.squint.__post_init__()

    @property
    def convention(self) -> str:
        """This override's authored convention literal."""
        return self.squint.convention

    @property
    def reference_frequency_hz(self) -> float:
        """This override's authored reference frequency, in Hz."""
        return self.squint.reference_frequency_hz

    @property
    def per_feed_offset_deg_at_reference(self) -> float:
        """This override's authored one-hand displacement, in degrees."""
        return self.squint.per_feed_offset_deg_at_reference

    @property
    def mechanical_feed_position_angle_deg(self) -> float:
        """This override's authored mechanical feed position angle."""
        return self.squint.mechanical_feed_position_angle_deg

    @property
    def positive_native_feed(self) -> str:
        """The native feed label carrying the positive displacement."""
        return self.squint.positive_native_feed


@dataclass(frozen=True, slots=True)
class ResolvedBeamSquint(_ResolvedValue):
    """The authored ``beams.squint`` block, resolved but not yet assigned."""

    default: ResolvedSquintRecord | None
    per_antenna: tuple[ResolvedAntennaSquint, ...]

    def __post_init__(self) -> None:
        if self.default is not None:
            _require_exact(self.default, (ResolvedSquintRecord,), "default")
            self.default.__post_init__()
        per_antenna = cast(
            tuple[ResolvedAntennaSquint, ...],
            _copy_override_tuple(
                self.per_antenna,
                (ResolvedAntennaSquint,),
                "per_antenna",
            ),
        )
        if self.default is None and not per_antenna:
            raise ValueError("a resolved squint block must carry at least one record")
        object.__setattr__(self, "per_antenna", per_antenna)


@dataclass(frozen=True, slots=True)
class ResolvedSquint(_ResolvedValue):
    """One antenna's assigned native-feed squint (Section 4.2.1).

    The five authored values plus the antenna's resolved mount literal, which
    fixes the field-rotation factors ``(eta_p, nu_p)`` the feed ray follows:
    ``beta_feed = wrap(beta_mechanical + eta_p psi_p + nu_p alt_p)`` at the
    antenna's resolved boresight.  An instrument source carrying no mount
    metadata resolves to ``fixed``, which is that value's accepted reading.
    """

    convention: Literal["cotton_uson_exact_v1"]
    reference_frequency_hz: float
    per_feed_offset_deg_at_reference: float
    mechanical_feed_position_angle_deg: float
    positive_native_feed: NativeFeedLabel
    mount_type: SquintMountType

    def __post_init__(self) -> None:
        _require_squint_fields(self)
        _require_member(self.mount_type, _SQUINT_MOUNT_TYPES, "mount_type")


def _require_squint_block(value: Any) -> None:
    if value is not None:
        _require_exact(value, (ResolvedBeamSquint,), "squint")
        value.__post_init__()


ZERNIKE_MAX_RADIAL_ORDER = 32
"""Section 3.3's v1 radial-order computation bound.

It is a bound on what this version evaluates, not a statement that higher
physical modes do not exist.
"""


@dataclass(frozen=True, slots=True)
class ResolvedSupportLeg(_ResolvedValue):
    """One resolved support leg (Section 3.2).

    ``position_angle_deg`` has already been checked against its canonical
    ``(-180, 180]`` interval, and ``width_m`` is the physical strip width in
    metres.  The leg is the closed outward half-strip from the edge of the
    central shadow to the ideal pupil edge; two legs 180 degrees apart describe
    a structure crossing the dish.
    """

    position_angle_deg: float
    width_m: float

    def __post_init__(self) -> None:
        _require_float(self.position_angle_deg, "position_angle_deg")
        if not (-180.0 < self.position_angle_deg <= 180.0):
            raise ValueError("position_angle_deg must lie in (-180, 180]")
        _require_float(self.width_m, "width_m", positive=True)


@dataclass(frozen=True, slots=True)
class ResolvedApertureBlockage(_ResolvedValue):
    """The resolved central shadow and its support legs (Section 3.2)."""

    central_diameter_ratio: float
    support_legs: tuple[ResolvedSupportLeg, ...]

    def __post_init__(self) -> None:
        _require_float(self.central_diameter_ratio, "central_diameter_ratio")
        if not (0.0 < self.central_diameter_ratio < 1.0):
            raise ValueError("central_diameter_ratio must satisfy 0 < epsilon < 1")
        if type(self.support_legs) is not tuple:
            raise TypeError("support_legs must be an exact tuple")
        legs = tuple(cast(tuple[Any, ...], self.support_legs))
        seen: set[float] = set()
        for leg in legs:
            _require_exact(leg, (ResolvedSupportLeg,), "support_legs item")
            leg.__post_init__()
            if leg.position_angle_deg in seen:
                raise ValueError("support_legs must have unique resolved angles")
            seen.add(leg.position_angle_deg)
        object.__setattr__(self, "support_legs", legs)


@dataclass(frozen=True, slots=True)
class ResolvedZernikeMode(_ResolvedValue):
    """One resolved real unit-RMS disk Zernike mode (Section 3.3)."""

    n: int
    m: int
    surface_height_coefficient_m: float

    def __post_init__(self) -> None:
        if type(self.n) is not int or type(self.m) is not int:
            raise TypeError("n and m must be exact Python integers")
        if not (0 <= self.n <= ZERNIKE_MAX_RADIAL_ORDER):
            raise ValueError("n must satisfy 0 <= n <= 32")
        if abs(self.m) > self.n or (self.n - abs(self.m)) % 2 != 0:
            raise ValueError("m must satisfy |m| <= n with n - |m| even")
        if (self.n, self.m) in {(0, 0), (1, -1), (1, 1)}:
            raise ValueError("piston and tip/tilt are owned by delay and pointing")
        _require_float(
            self.surface_height_coefficient_m,
            "surface_height_coefficient_m",
        )


@dataclass(frozen=True, slots=True)
class ResolvedZernikeSurface(_ResolvedValue):
    """The resolved deterministic surface-height map (Section 3.3).

    ``modes`` is sorted by ``(n, m)`` so the fingerprint is stable; sorting a
    sum of orthogonal basis functions does not change the exact mathematical
    sum.
    """

    convention: Literal["radiosim.real_unit_rms_disk_surface_height.v1"]
    modes: tuple[ResolvedZernikeMode, ...]

    def __post_init__(self) -> None:
        _require_literal(
            self.convention,
            "radiosim.real_unit_rms_disk_surface_height.v1",
            "convention",
        )
        modes = cast(
            tuple[ResolvedZernikeMode, ...],
            _copy_exact_tuple(self.modes, (ResolvedZernikeMode,), "modes"),
        )
        seen: set[tuple[int, int]] = set()
        for mode in modes:
            mode.__post_init__()
            if (mode.n, mode.m) in seen:
                raise ValueError("modes must have unique (n, m) index pairs")
            seen.add((mode.n, mode.m))
        if not any(mode.surface_height_coefficient_m != 0.0 for mode in modes):
            raise ValueError("an all-zero Zernike block is an exact identity")
        object.__setattr__(
            self,
            "modes",
            tuple(sorted(modes, key=lambda mode: (mode.n, mode.m))),
        )


@dataclass(frozen=True, slots=True)
class ResolvedAperturePhysics(_ResolvedValue):
    """The resolved array-wide ``beams.aperture_physics`` block (Section 3.1).

    At least one effective child is required: a parent whose children together
    resolve to the identity is rejected rather than accepted and discarded.
    """

    normalization: Literal["unmodified_ideal_aperture_v1"]
    blockage: ResolvedApertureBlockage | None = None
    zernike_surface: ResolvedZernikeSurface | None = None

    def __post_init__(self) -> None:
        _require_literal(
            self.normalization,
            "unmodified_ideal_aperture_v1",
            "normalization",
        )
        if self.blockage is not None:
            _require_exact(self.blockage, (ResolvedApertureBlockage,), "blockage")
            self.blockage.__post_init__()
        if self.zernike_surface is not None:
            _require_exact(
                self.zernike_surface,
                (ResolvedZernikeSurface,),
                "zernike_surface",
            )
            self.zernike_surface.__post_init__()
        if self.blockage is None and self.zernike_surface is None:
            raise ValueError(
                "aperture physics requires at least one effective child block"
            )


def _require_aperture_physics(value: Any) -> None:
    if value is not None:
        _require_exact(value, (ResolvedAperturePhysics,), "aperture_physics")
        value.__post_init__()


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


def _require_mount_blocks(value: Any) -> None:
    """Validate the two mode-independent mount blocks every input carries.

    Pointing offsets and surface errors describe the mount and the dish, not the
    beam model, so they are available in all four modes rather than only where
    the configuration happens to be per-antenna already.
    """
    pointing, surface_error = value
    if pointing is not None:
        _require_exact(pointing, (ResolvedBeamPointing,), "pointing")
        pointing.__post_init__()
    if surface_error is not None:
        _require_exact(surface_error, (ResolvedBeamSurfaceError,), "surface_error")
        surface_error.__post_init__()


@dataclass(frozen=True, slots=True)
class ResolvedAnalyticBeamsInput(_ResolvedValue):
    mode: Literal["analytic"]
    model: ResolvedAnalyticBeamDefinition
    pointing: ResolvedBeamPointing | None = None
    surface_error: ResolvedBeamSurfaceError | None = None
    aperture_physics: ResolvedAperturePhysics | None = None
    squint: ResolvedBeamSquint | None = None

    def __post_init__(self) -> None:
        _require_literal(self.mode, "analytic", "mode")
        _require_exact(self.model, (ResolvedAnalyticBeamDefinition,), "model")
        _require_mount_blocks((self.pointing, self.surface_error))
        _require_aperture_physics(self.aperture_physics)
        _require_squint_block(self.squint)


@dataclass(frozen=True, slots=True)
class ResolvedSharedFITSBeamsInput(_ResolvedValue):
    mode: Literal["shared_fits"]
    beam: ResolvedFITSBeamDefinition
    pointing: ResolvedBeamPointing | None = None
    surface_error: ResolvedBeamSurfaceError | None = None
    aperture_physics: ResolvedAperturePhysics | None = None
    squint: ResolvedBeamSquint | None = None

    def __post_init__(self) -> None:
        _require_literal(self.mode, "shared_fits", "mode")
        _require_exact(self.beam, (ResolvedFITSBeamDefinition,), "beam")
        _require_mount_blocks((self.pointing, self.surface_error))
        _require_aperture_physics(self.aperture_physics)
        _require_squint_block(self.squint)


@dataclass(frozen=True, slots=True)
class ResolvedPerAntennaFITSBeamsInput(_ResolvedValue):
    mode: Literal["per_antenna_fits"]
    assignments: tuple[ResolvedFITSBeamAssignmentInput, ...]
    pointing: ResolvedBeamPointing | None = None
    surface_error: ResolvedBeamSurfaceError | None = None
    aperture_physics: ResolvedAperturePhysics | None = None
    squint: ResolvedBeamSquint | None = None

    def __post_init__(self) -> None:
        _require_literal(self.mode, "per_antenna_fits", "mode")
        copied = _copy_exact_tuple(
            self.assignments,
            (ResolvedFITSBeamAssignmentInput,),
            "assignments",
        )
        _require_mount_blocks((self.pointing, self.surface_error))
        _require_aperture_physics(self.aperture_physics)
        _require_squint_block(self.squint)
        object.__setattr__(self, "assignments", copied)


@dataclass(frozen=True, slots=True)
class ResolvedMixedBeamsInput(_ResolvedValue):
    mode: Literal["mixed"]
    analytic_model: ResolvedAnalyticBeamDefinition
    assignments: tuple[ResolvedMixedBeamAssignmentInput, ...]
    pointing: ResolvedBeamPointing | None = None
    surface_error: ResolvedBeamSurfaceError | None = None
    aperture_physics: ResolvedAperturePhysics | None = None
    squint: ResolvedBeamSquint | None = None

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
        _require_mount_blocks((self.pointing, self.surface_error))
        _require_aperture_physics(self.aperture_physics)
        _require_squint_block(self.squint)
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
    """Detached immutable provenance for one validated BeamFITS transport.

    ``docs/development/sci005_beam_physics_plan.md`` Section 5.2.1 appends the
    last seven fields for the Stage-3 full-efield accepted subset.  Every one is
    annotated ``<type> | None = None`` and left ``None`` on the accepted scalar
    ``peak`` path, which is what keeps :func:`_optional_block_fields` omitting
    them from both ``to_snapshot`` and the canonical fingerprint payload: a
    ``peak`` document's beam snapshot, scientific digest, HDF5
    ``provenance/beam_json``, and result bytes stay byte-identical.
    """

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
    x_orientation: str | None
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
    accepted_subset_version: str | None = None
    radiosim_normalization: str | None = None
    resolved_feed_array: tuple[str, str] | None = None
    derived_x_orientation_verdict: str | None = None
    basis_vector_convention: str | None = None
    factorization_convention: str | None = None
    stored_grid_peak_by_frequency: tuple[tuple[float, float], ...] | None = None

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
            "native_dtype",
        ):
            _ = _require_normalized_string(getattr(self, field_name), field_name)
        # ``get_x_orientation_from_feeds`` legitimately returns ``None`` for a
        # rotated linear receptor and for a circular receptor whose static
        # rotation is neither 0 nor pi/2 (Section 5.1.1 item 7), so the legacy
        # field is nullable while the scalar path still stores exactly "east".
        if self.x_orientation is not None:
            _ = _require_normalized_string(self.x_orientation, "x_orientation")
            if self.x_orientation not in {"east", "north"}:
                raise ValueError("x_orientation must be 'east', 'north', or None")

        if self.pyuvdata_version != "3.2.1":
            raise ValueError("pyuvdata_version must be the pinned '3.2.1' contract")
        expected_metadata = {
            "beam_type": "efield",
            "antenna_type": "simple",
            "pixel_coordinate_system": "az_za",
            "mount_type": "fixed",
            "data_normalization": "peak",
        }
        for field_name, expected in expected_metadata.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} must be {expected!r}")

        feed_array = _copy_string_tuple(self.feed_array, "feed_array")
        if feed_array not in {("x", "y"), ("r", "l")}:
            raise ValueError("feed_array must be exactly ('x', 'y') or ('r', 'l')")
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

        peaks = self._validated_stage3_fields()

        object.__setattr__(self, "feed_array", feed_array)
        object.__setattr__(self, "data_shape", data_shape)
        if peaks is not None:
            object.__setattr__(self, "stored_grid_peak_by_frequency", peaks)

    def _validated_stage3_fields(self) -> tuple[tuple[float, float], ...] | None:
        """Validate Section 5.2.1's appended full-efield record, all or none."""
        stage3 = (
            "accepted_subset_version",
            "radiosim_normalization",
            "resolved_feed_array",
            "derived_x_orientation_verdict",
            "basis_vector_convention",
            "factorization_convention",
            "stored_grid_peak_by_frequency",
        )
        present = tuple(name for name in stage3 if getattr(self, name) is not None)
        if not present:
            return None
        if len(present) != len(stage3):
            raise ValueError(
                "the full-efield provenance fields are all present or all None"
            )
        _require_literal(
            self.accepted_subset_version,
            "sci005-stage3-full-efield-v1",
            "accepted_subset_version",
        )
        _require_literal(
            self.radiosim_normalization,
            "uvbeam_peak_common_v1",
            "radiosim_normalization",
        )
        resolved_feeds = _copy_string_tuple(
            self.resolved_feed_array,
            "resolved_feed_array",
        )
        if resolved_feeds not in {("x", "y"), ("r", "l")}:
            raise ValueError(
                "resolved_feed_array must be exactly ('x', 'y') or ('r', 'l')"
            )
        verdict = _require_normalized_string(
            self.derived_x_orientation_verdict,
            "derived_x_orientation_verdict",
        )
        if verdict not in {"east", "north", "none"}:
            raise ValueError(
                "derived_x_orientation_verdict must be 'east', 'north', or 'none'"
            )
        _require_literal(
            self.basis_vector_convention,
            "uvbeam_theta_phi_chain_tangent_v1",
            "basis_vector_convention",
        )
        _require_literal(
            self.factorization_convention,
            "receptor_conjugated_native_efield_v1",
            "factorization_convention",
        )
        values = self.stored_grid_peak_by_frequency
        if type(values) is not tuple or not values:
            raise ValueError(
                "stored_grid_peak_by_frequency must be a nonempty exact tuple"
            )
        copied: list[tuple[float, float]] = []
        previous: float | None = None
        for index, pair in enumerate(cast(tuple[Any, ...], values)):
            if type(pair) is not tuple or len(cast(tuple[Any, ...], pair)) != 2:
                raise TypeError(
                    "stored_grid_peak_by_frequency items must be exact pairs"
                )
            frequency_hz, observed_peak = cast(tuple[Any, Any], pair)
            _require_float(
                frequency_hz,
                f"stored_grid_peak_by_frequency[{index}][0]",
                positive=True,
            )
            _require_float(
                observed_peak,
                f"stored_grid_peak_by_frequency[{index}][1]",
                positive=True,
            )
            if previous is not None and frequency_hz <= previous:
                raise ValueError(
                    "stored_grid_peak_by_frequency must be strictly increasing"
                )
            previous = cast(float, frequency_hz)
            copied.append((cast(float, frequency_hz), cast(float, observed_peak)))
        return tuple(copied)


@dataclass(frozen=True, slots=True)
class LoadedBeamHandlerState(_ResolvedValue):
    """Detached immutable state for one standalone validated beam handler."""

    handler_id: str
    kind: Literal["analytic", "fits"]
    definition_fingerprint: str
    scientific_fingerprint: str
    file: BeamFileProvenance | None
    voltage_feature_scale_by_frequency: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in {"analytic", "fits"}:
            raise ValueError("kind must be 'analytic' or 'fits'")
        _require_fingerprint(self.definition_fingerprint, "definition_fingerprint")
        _require_fingerprint(self.scientific_fingerprint, "scientific_fingerprint")
        if self.kind == "fits":
            _require_exact(self.file, (BeamFileProvenance,), "file")
            cast(BeamFileProvenance, self.file).__post_init__()
        elif self.file is not None:
            raise ValueError("analytic handlers require file=None")
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


def _aperture_physics_payload(aperture: ResolvedAperturePhysics) -> dict[str, Any]:
    """Return the canonical scientific payload for one resolved aperture block.

    Section 3.1 requires the profile-set convention literal to enter the
    scientific fingerprint whenever a Stage-1 feature is explicit, and Section 2
    requires convention-version literals to be fingerprinted alongside the
    resolved physical parameters.  Paths and scheduling choices never appear
    here; only physics and convention versions do.
    """
    payload: dict[str, Any] = {
        "normalization": aperture.normalization,
        "pupil_profile_set": "radiosim.circular_stage1_pupil_profiles.v1",
        "aperture_axes": "north_east_azimuth_north_through_east_v1",
        "aperture_method": "boundary_fitted_polar_gauss_legendre_v1",
    }
    if aperture.blockage is not None:
        payload["blockage"] = {
            "support_mask": "radiosim.central_disk_outward_half_strip_ne.v1",
            "central_diameter_ratio": aperture.blockage.central_diameter_ratio,
            "support_legs": [
                {
                    "position_angle_deg": leg.position_angle_deg,
                    "width_m": leg.width_m,
                }
                for leg in aperture.blockage.support_legs
            ],
        }
    if aperture.zernike_surface is not None:
        payload["zernike_surface"] = {
            "convention": aperture.zernike_surface.convention,
            "modes": [
                {
                    "n": mode.n,
                    "m": mode.m,
                    "surface_height_coefficient_m": (mode.surface_height_coefficient_m),
                }
                for mode in aperture.zernike_surface.modes
            ],
        }
    return payload


def _surface_error_payload(surface_error: ResolvedSurfaceError) -> dict[str, Any]:
    """Return the canonical payload for one resolved surface error.

    An antenna with no nested diagnostic reproduces its pre-SCI-005 payload
    exactly, so configuring the diagnostic -- and only configuring it -- changes
    the scientific fingerprint.
    """
    payload: dict[str, Any] = {
        "rms_surface_error_m": surface_error.rms_surface_error_m,
    }
    diagnostic = surface_error.error_beam_diagnostic
    if diagnostic is not None:
        payload["error_beam_diagnostic"] = {
            "kind": diagnostic.kind,
            "correlation_length_m": diagnostic.correlation_length_m,
            "covariance_convention": ("gaussian_one_over_e_surface_covariance_v1"),
            "method": "poisson_paired_pupil_separation_v1",
        }
    return payload


def _squint_payload(squint: ResolvedSquint) -> dict[str, Any]:
    """Return the canonical scientific payload for one resolved squint record.

    ``docs/development/sci005_beam_physics_plan.md`` Section 4.2.1 freezes the
    key set: the six resolved field values plus the three convention literals
    that fix which displacement, which frame composition, and which
    factorization the numbers mean.  An antenna without squint contributes no
    key at all, which is what keeps every pre-Stage-2 assignment identity
    byte-identical.
    """
    return {
        "convention": squint.convention,
        "reference_frequency_hz": squint.reference_frequency_hz,
        "per_feed_offset_deg_at_reference": (squint.per_feed_offset_deg_at_reference),
        "mechanical_feed_position_angle_deg": (
            squint.mechanical_feed_position_angle_deg
        ),
        "positive_native_feed": squint.positive_native_feed,
        "mount_type": squint.mount_type,
        "direction_convention": "feed_ray_plus_half_pi_north_through_east_v1",
        "frame_convention": "pointing_then_squint_great_circle_v1",
        "factorization_convention": "receptor_conjugated_native_diagonal_v1",
    }


def _assignment_fingerprint(
    antenna_id: AntennaId,
    antenna_diameter_m: float,
    definition: ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition,
    pointing: ResolvedPointingOffset | None = None,
    surface_error: ResolvedSurfaceError | None = None,
    aperture_physics: ResolvedAperturePhysics | None = None,
    squint: ResolvedSquint | None = None,
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
    # The two Tier 7I keys are added only when the science is present, so an
    # assignment with neither reproduces its pre-7I digest exactly.  An inert
    # value cannot reach here: it resolves to ``None`` (Section 19.2).
    if pointing is not None:
        payload["pointing"] = {
            "azimuth_offset_rad": pointing.azimuth_offset_rad,
            "elevation_offset_rad": pointing.elevation_offset_rad,
        }
    if surface_error is not None:
        payload["surface_error"] = _surface_error_payload(surface_error)
    # Stage-1 aperture physics is array-wide but reaches the response through
    # every assignment, so it is fingerprinted here beside the mount science and
    # is absent -- byte for byte -- when the block is absent.
    if aperture_physics is not None:
        payload["aperture_physics"] = _aperture_physics_payload(aperture_physics)
    # Stage-2 squint is per-antenna state like pointing and surface error: it
    # enters the assignment identity only when the antenna actually carries it.
    if squint is not None:
        payload["squint"] = _squint_payload(squint)
    return _canonical_digest(payload)


@dataclass(frozen=True, slots=True)
class ResolvedBeamAssignment(_ResolvedValue):
    antenna_id: AntennaId
    antenna_diameter_m: float
    definition: ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition
    provenance: BeamAssignmentProvenance
    assignment_fingerprint: str
    pointing: ResolvedPointingOffset | None = None
    surface_error: ResolvedSurfaceError | None = None
    aperture_physics: ResolvedAperturePhysics | None = None
    squint: ResolvedSquint | None = None

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
        if self.pointing is not None:
            _require_exact(self.pointing, (ResolvedPointingOffset,), "pointing")
            self.pointing.__post_init__()
        if self.surface_error is not None:
            _require_exact(
                self.surface_error,
                (ResolvedSurfaceError,),
                "surface_error",
            )
            self.surface_error.__post_init__()
        _require_aperture_physics(self.aperture_physics)
        if self.squint is not None:
            _require_exact(self.squint, (ResolvedSquint,), "squint")
            self.squint.__post_init__()
        if self.provenance.canonical_antenna != antenna_id:
            raise ValueError("provenance.canonical_antenna must equal antenna_id")
        _require_fingerprint(self.assignment_fingerprint, "assignment_fingerprint")
        expected = _assignment_fingerprint(
            antenna_id,
            self.antenna_diameter_m,
            self.definition,
            self.pointing,
            self.surface_error,
            self.aperture_physics,
            self.squint,
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
    pointing: ResolvedPointingOffset | None = None,
    surface_error: ResolvedSurfaceError | None = None,
    aperture_physics: ResolvedAperturePhysics | None = None,
    squint: ResolvedSquint | None = None,
) -> ResolvedBeamAssignment:
    fingerprint = _assignment_fingerprint(
        antenna_id,
        antenna_diameter_m,
        definition,
        pointing,
        surface_error,
        aperture_physics,
        squint,
    )
    return ResolvedBeamAssignment(
        antenna_id=antenna_id,
        antenna_diameter_m=antenna_diameter_m,
        definition=definition,
        provenance=provenance,
        assignment_fingerprint=fingerprint,
        pointing=pointing,
        surface_error=surface_error,
        aperture_physics=aperture_physics,
        squint=squint,
    )


def _definition_identity_key(
    definition: ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition,
) -> tuple[str, ...]:
    """Return the exact pre-load identity key for one resolved definition.

    FITS definition fingerprints bind only the load settings, so the resolved
    path must join the key to keep two distinct files with identical settings
    distinct until load time binds their content.
    """
    if type(definition) is ResolvedFITSBeamDefinition:
        return (
            "fits",
            definition.definition_fingerprint,
            definition.path.as_posix(),
        )
    return ("analytic", definition.definition_fingerprint)


def _deduplicated_definitions(
    assignments: tuple[ResolvedBeamAssignment, ...],
) -> tuple[ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition, ...]:
    seen: set[tuple[str, ...]] = set()
    unique: list[ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition] = []
    for assignment in assignments:
        key = _definition_identity_key(assignment.definition)
        if key not in seen:
            seen.add(key)
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
                "definition for each distinct definition identity"
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


def _loaded_state_fingerprint(
    resolved: ResolvedBeamState,
    handlers: tuple[LoadedBeamHandlerState, ...],
    assignment_handler_ids: tuple[tuple[AntennaId, str], ...],
) -> str:
    by_id = {handler.handler_id: handler for handler in handlers}
    return _canonical_digest(
        {
            "schema_version": _SCHEMA_VERSION,
            "kind": "loaded_beam_state",
            "resolved_state_fingerprint": resolved.state_fingerprint,
            "handlers": [
                {
                    "kind": handler.kind,
                    "definition_fingerprint": handler.definition_fingerprint,
                    "scientific_fingerprint": handler.scientific_fingerprint,
                    "voltage_feature_scale_by_frequency": (
                        handler.voltage_feature_scale_by_frequency
                    ),
                }
                for handler in handlers
            ],
            "assignments": [
                {
                    "canonical_antenna": {
                        "number": antenna_id.number,
                        "name": antenna_id.name,
                    },
                    "definition_fingerprint": by_id[handler_id].definition_fingerprint,
                    "scientific_fingerprint": by_id[handler_id].scientific_fingerprint,
                }
                for antenna_id, handler_id in assignment_handler_ids
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class LoadedBeamState(_ResolvedValue):
    """Immutable public snapshot for one completely loaded beam system."""

    resolved: ResolvedBeamState
    handlers: tuple[LoadedBeamHandlerState, ...]
    assignment_handler_ids: tuple[tuple[AntennaId, str], ...]
    loaded_fingerprint: str

    def __post_init__(self) -> None:
        _require_exact(self.resolved, (ResolvedBeamState,), "resolved")
        self.resolved.__post_init__()
        resolved = replace(self.resolved)

        if type(self.handlers) is not tuple or not self.handlers:
            raise ValueError("handlers must be a nonempty exact tuple")
        copied_handlers: list[LoadedBeamHandlerState] = []
        handler_ids: set[str] = set()
        frequency_axis: tuple[float, ...] | None = None
        for index, handler in enumerate(self.handlers):
            _require_exact(
                handler,
                (LoadedBeamHandlerState,),
                f"handlers[{index}]",
            )
            handler.__post_init__()
            copied_file = replace(handler.file) if handler.file is not None else None
            copied = replace(handler, file=copied_file)
            if copied.handler_id in handler_ids:
                raise ValueError("handler_id values must be unique")
            if int(copied.handler_id[5:9]) != index:
                raise ValueError(
                    "handler_id ordinal must match canonical handler order"
                )
            copied_frequency_axis = tuple(
                frequency_hz
                for frequency_hz, _scale_rad in (
                    copied.voltage_feature_scale_by_frequency
                )
            )
            if frequency_axis is None:
                frequency_axis = copied_frequency_axis
            elif copied_frequency_axis != frequency_axis:
                raise ValueError(
                    "loaded handlers must use identical ordered frequency axes"
                )
            handler_ids.add(copied.handler_id)
            copied_handlers.append(copied)
        handlers = tuple(copied_handlers)
        handlers_by_id = {handler.handler_id: handler for handler in handlers}

        if type(self.assignment_handler_ids) is not tuple:
            raise TypeError("assignment_handler_ids must be an exact tuple")
        if len(self.assignment_handler_ids) != len(resolved.assignments):
            raise ValueError(
                "assignment_handler_ids must cover every resolved assignment"
            )
        copied_assignments: list[tuple[AntennaId, str]] = []
        first_used_handler_ids: list[str] = []
        seen_antennas: set[AntennaId] = set()
        for index, pair in enumerate(self.assignment_handler_ids):
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("assignment_handler_ids items must be exact pairs")
            antenna_id, handler_id = pair
            copied_antenna = _copy_antenna_id(
                antenna_id,
                f"assignment_handler_ids[{index}][0]",
            )
            if type(handler_id) is not str or handler_id not in handlers_by_id:
                raise ValueError(
                    "assignment_handler_ids must reference a loaded handler_id"
                )
            expected_assignment = resolved.assignments[index]
            if copied_antenna != expected_assignment.antenna_id:
                raise ValueError(
                    "assignment_handler_ids must follow canonical assignment order"
                )
            if copied_antenna in seen_antennas:
                raise ValueError("assignment_handler_ids antenna values must be unique")
            seen_antennas.add(copied_antenna)
            handler = handlers_by_id[handler_id]
            if handler.definition_fingerprint != (
                expected_assignment.definition.definition_fingerprint
            ):
                raise ValueError(
                    "loaded handler definition does not match resolved assignment"
                )
            if handler.kind != expected_assignment.definition.kind:
                raise ValueError(
                    "loaded handler kind does not match resolved assignment"
                )
            if handler.kind == "fits":
                handler_file = cast(BeamFileProvenance, handler.file)
                fits_definition = cast(
                    ResolvedFITSBeamDefinition,
                    expected_assignment.definition,
                )
                if handler_file.resolved_path != fits_definition.path:
                    raise ValueError(
                        "loaded handler file path does not match resolved assignment"
                    )
            if handler_id not in first_used_handler_ids:
                first_used_handler_ids.append(handler_id)
            copied_assignments.append((copied_antenna, handler_id))
        assignment_handler_ids = tuple(copied_assignments)
        if tuple(first_used_handler_ids) != tuple(
            handler.handler_id for handler in handlers
        ):
            raise ValueError(
                "handlers must be ordered by first canonical assignment use"
            )

        _require_fingerprint(self.loaded_fingerprint, "loaded_fingerprint")
        expected = _loaded_state_fingerprint(
            resolved,
            handlers,
            assignment_handler_ids,
        )
        if self.loaded_fingerprint != expected:
            raise ValueError(
                "loaded_fingerprint does not match canonical loaded beam state"
            )

        object.__setattr__(self, "resolved", resolved)
        object.__setattr__(self, "handlers", handlers)
        object.__setattr__(
            self,
            "assignment_handler_ids",
            assignment_handler_ids,
        )


def _create_loaded_beam_state(  # pyright: ignore[reportUnusedFunction]
    *,
    resolved: ResolvedBeamState,
    handlers: tuple[LoadedBeamHandlerState, ...],
    assignment_handler_ids: tuple[tuple[AntennaId, str], ...],
) -> LoadedBeamState:
    fingerprint = _loaded_state_fingerprint(
        resolved,
        handlers,
        assignment_handler_ids,
    )
    return LoadedBeamState(
        resolved=resolved,
        handlers=handlers,
        assignment_handler_ids=assignment_handler_ids,
        loaded_fingerprint=fingerprint,
    )


__all__ = [
    "ZERNIKE_MAX_RADIAL_ORDER",
    "BeamAssignmentProvenance",
    "ResolvedAntennaPointingOffset",
    "ResolvedAntennaSurfaceError",
    "ResolvedApertureBlockage",
    "ResolvedAperturePhysics",
    "ResolvedRuzePowerDiagnostic",
    "ResolvedSupportLeg",
    "ResolvedZernikeMode",
    "ResolvedZernikeSurface",
    "ResolvedBeamPointing",
    "ResolvedBeamSurfaceError",
    "ResolvedPointingOffset",
    "ResolvedSurfaceError",
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
    "LoadedBeamState",
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
