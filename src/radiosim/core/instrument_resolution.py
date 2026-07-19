"""Tier 2D instrument-source normalization and coordinate staging.

This module deliberately stops at a diameter-incomplete, immutable staging
inventory.  Diameter precedence, final instrument construction, baselines, and
Simulator integration belong to later Tier 2 slices.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt

from radiosim.core.instrument import AntennaFieldSource, ResolvedEarthLocation
from radiosim.io.instrument_config import InstrumentConfig

if TYPE_CHECKING:
    from radiosim.io.instrument_sources import (
        DatasetTelescopeLoader,
        InternetConfiguration,
        KnownTelescopeLoader,
        LoadedInstrumentSource,
        ModuleImporter,
        SourceLocationFacts,
    )

_MAX_ANTENNA_NUMBER = 2_147_483_647
_LOCATION_MATCH_THRESHOLD_M = 1.0
PositionFrame = Literal["enu", "relative_ecef"]


class InstrumentResolutionError(ValueError):
    """Base class for Tier 2 instrument-resolution failures."""


class InstrumentSourceError(InstrumentResolutionError):
    """The selected source could not provide coherent instrument metadata."""


class InstrumentFormatError(InstrumentSourceError):
    """A retained source format is malformed or structurally incoherent."""


class TelescopeNotFoundError(InstrumentSourceError):
    """A requested known telescope is absent from pyuvdata/Astropy metadata."""


class OptionalInstrumentDependencyError(InstrumentSourceError):
    """A dependency needed by the selected source is unavailable."""


class InstrumentLocationMismatchError(InstrumentResolutionError):
    """Explicit and embedded Earth locations describe different references."""


class AntennaIdentifierError(InstrumentResolutionError):
    """An antenna number, name, or repeated-record identity is invalid."""


class DuplicateAntennaError(AntennaIdentifierError):
    """Two source records share a canonical name or number."""


class CoordinateFrameError(InstrumentResolutionError):
    """A source coordinate frame is unsupported or conversion failed."""


class InvalidAntennaPositionError(InstrumentResolutionError):
    """An antenna position is malformed or non-finite."""


class EmptyInstrumentError(InstrumentResolutionError):
    """The selected source contains no antennas."""


class DiameterResolutionError(InstrumentResolutionError):
    """A present source diameter is malformed, non-finite, or non-positive."""


class _Quantity(Protocol):
    def to_value(self, unit: str) -> object: ...


class _EarthLocationValue(Protocol):
    x: _Quantity
    y: _Quantity
    z: _Quantity
    lon: _Quantity
    lat: _Quantity
    height: _Quantity


class _EarthLocationType(Protocol):
    def from_geodetic(
        self, longitude: float, latitude: float, height: float
    ) -> _EarthLocationValue: ...

    def from_geocentric(
        self, x: float, y: float, z: float, *, unit: str
    ) -> _EarthLocationValue: ...


class _CoordinatesModule(Protocol):
    EarthLocation: _EarthLocationType


class EcefToEnu(Protocol):
    """Public-coordinate-converter seam used by deterministic tests."""

    def __call__(
        self,
        absolute_ecef_m: npt.NDArray[np.float64],
        *,
        center_loc: object,
    ) -> object: ...


class _AntennaIdentity(Protocol):
    @property
    def number(self) -> int: ...

    @property
    def name(self) -> str: ...

    @property
    def source_record(self) -> str: ...


class _PyuvdataUtilsModule(Protocol):
    ENU_from_ECEF: EcefToEnu


@dataclass(frozen=True, slots=True)
class StagedAntenna:
    """One normalized antenna before diameter precedence is applied."""

    number: int
    name: str
    position_enu_m: tuple[float, float, float]
    source_diameter_m: float | None
    mount_type: str | None
    beam_id: int | str | None
    number_source: AntennaFieldSource
    name_source: AntennaFieldSource
    position_source: AntennaFieldSource
    diameter_source: AntennaFieldSource | None
    mount_source: AntennaFieldSource | None
    beam_id_source: AntennaFieldSource | None
    source_record: str


@dataclass(frozen=True, slots=True)
class StagedInstrumentProvenance:
    """Stable source and location facts retained for Tier 2E."""

    source_kind: str
    source_reference: str
    source_format: str | None
    registry_policy: str | None
    explicit_telescope_name: str | None
    embedded_telescope_name: str | None
    embedded_location_itrs_xyz_m: tuple[float, float, float] | None
    explicit_location_itrs_xyz_m: tuple[float, float, float] | None
    location_separation_m: float | None
    pyuvdata_version: str | None
    source_sha256: str | None


@dataclass(frozen=True, slots=True)
class StagedInstrument:
    """Owned, deterministic, diameter-incomplete Tier 2D result."""

    location: ResolvedEarthLocation
    antennas: tuple[StagedAntenna, ...]
    provenance: StagedInstrumentProvenance


def normalize_antenna_number(value: object, *, reference: str) -> int:
    """Copy one semantically integral source value to a bounded built-in int."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise AntennaIdentifierError(
            f"{reference}: antenna number must be an integer in "
            f"0..{_MAX_ANTENNA_NUMBER}"
        )
    number = int(value)
    if not 0 <= number <= _MAX_ANTENNA_NUMBER:
        raise AntennaIdentifierError(
            f"{reference}: antenna number {number} is outside 0..{_MAX_ANTENNA_NUMBER}"
        )
    return number


def normalize_identity(value: object, *, reference: str, label: str) -> str:
    """Normalize a source identity using the accepted NFC/case-sensitive rule."""
    if not isinstance(value, str):
        raise AntennaIdentifierError(f"{reference}: {label} must be a string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized:
        raise AntennaIdentifierError(f"{reference}: {label} must be nonblank")
    return str(normalized)


def normalize_mount(value: object | None, *, reference: str) -> str | None:
    """Normalize optional inert mount metadata."""
    if value is None:
        return None
    try:
        return normalize_identity(value, reference=reference, label="mount type")
    except AntennaIdentifierError as error:
        raise InstrumentFormatError(str(error)) from error


def normalize_beam_id(value: object | None, *, reference: str) -> int | str | None:
    """Normalize an optional inert RadioSim BeamID."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise InstrumentFormatError(
            f"{reference}: BeamID must be a nonblank string or integer"
        )
    if isinstance(value, Integral):
        return int(value)
    if not isinstance(value, str):
        raise InstrumentFormatError(
            f"{reference}: BeamID must be a nonblank string or integer"
        )
    normalized = unicodedata.normalize("NFC", value.strip())
    if normalized in {"", "''", '""'}:
        raise InstrumentFormatError(
            f"{reference}: BeamID must be a nonblank string or integer"
        )
    return str(normalized)


def normalize_finite_float(
    value: object,
    *,
    reference: str,
    label: str,
    error_type: type[InstrumentResolutionError] = InvalidAntennaPositionError,
) -> float:
    """Copy one real source value to a finite float64-derived built-in float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise error_type(f"{reference}: {label} must be a real number")
    try:
        normalized = float(np.float64(value))
    except (OverflowError, TypeError, ValueError) as error:
        raise error_type(
            f"{reference}: {label} must be representable as float64"
        ) from error
    if not math.isfinite(normalized):
        raise error_type(f"{reference}: {label} must be finite")
    return 0.0 if normalized == 0.0 else normalized


def normalize_position(
    value: object,
    *,
    reference: str,
) -> tuple[float, float, float]:
    """Validate and own one exact three-component position."""
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise InvalidAntennaPositionError(
            f"{reference}: antenna position must contain exactly three values"
        )
    try:
        copied = tuple(cast(Iterable[object], value))
    except TypeError as error:
        raise InvalidAntennaPositionError(
            f"{reference}: antenna position must contain exactly three values"
        ) from error
    if len(copied) != 3:
        raise InvalidAntennaPositionError(
            f"{reference}: antenna position must contain exactly three values"
        )
    result = tuple(
        normalize_finite_float(
            component,
            reference=reference,
            label=f"position component {index}",
        )
        for index, component in enumerate(copied)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        norm = float(np.linalg.norm(np.asarray(result, dtype=np.float64)))
    if not math.isfinite(norm):
        raise InvalidAntennaPositionError(
            f"{reference}: antenna position has a non-finite derived norm"
        )
    return (result[0], result[1], result[2])


def normalize_source_diameter(
    value: object | None,
    *,
    reference: str,
) -> float | None:
    """Validate a present source diameter without filling missing values."""
    if value is None:
        return None
    diameter = normalize_finite_float(
        value,
        reference=reference,
        label="source diameter",
        error_type=DiameterResolutionError,
    )
    if diameter <= 0.0:
        raise DiameterResolutionError(
            f"{reference}: source diameter must be finite and positive"
        )
    return diameter


def validate_unique_antennas(
    antennas: Iterable[_AntennaIdentity],
    *,
    source_reference: str,
) -> None:
    """Reject duplicate canonical names and numbers in one selected source."""
    numbers: dict[int, str] = {}
    names: dict[str, str] = {}
    count = 0
    for antenna in antennas:
        count += 1
        number = antenna.number
        name = antenna.name
        record = antenna.source_record
        prior_number = numbers.get(number)
        if prior_number is not None:
            raise DuplicateAntennaError(
                f"{source_reference}: duplicate antenna number {number} at "
                f"{record}; first seen at {prior_number}"
            )
        prior_name = names.get(name)
        if prior_name is not None:
            raise DuplicateAntennaError(
                f"{source_reference}: duplicate antenna name {name!r} at "
                f"{record}; first seen at {prior_name}"
            )
        numbers[number] = record
        names[name] = record
    if count == 0:
        raise EmptyInstrumentError(
            f"{source_reference}: selected instrument source contains zero antennas"
        )


def _coordinates_module(
    module_importer: Callable[[str], object] = import_module,
    *,
    reference: str,
) -> _CoordinatesModule:
    try:
        return cast(_CoordinatesModule, module_importer("astropy.coordinates"))
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalInstrumentDependencyError(
            f"{reference}: instrument coordinates require astropy; "
            "install RadioSim dependencies"
        ) from error


def earth_location_facts(
    value: object,
    *,
    reference: str,
    module_importer: Callable[[str], object] = import_module,
) -> tuple[float, float, float, tuple[float, float, float]]:
    """Copy exact geodetic and ITRS facts from an EarthLocation only."""
    coordinates = _coordinates_module(module_importer, reference=reference)
    earth_location_type = cast(type[object], coordinates.EarthLocation)
    if not isinstance(value, earth_location_type):
        raise CoordinateFrameError(
            f"{reference}: embedded location must be astropy.coordinates.EarthLocation"
        )
    location = cast(_EarthLocationValue, value)
    longitude = normalize_finite_float(
        location.lon.to_value("deg"),
        reference=reference,
        label="longitude",
        error_type=CoordinateFrameError,
    )
    latitude = normalize_finite_float(
        location.lat.to_value("deg"),
        reference=reference,
        label="latitude",
        error_type=CoordinateFrameError,
    )
    height = normalize_finite_float(
        location.height.to_value("m"),
        reference=reference,
        label="height",
        error_type=CoordinateFrameError,
    )
    xyz = normalize_position(
        (
            location.x.to_value("m"),
            location.y.to_value("m"),
            location.z.to_value("m"),
        ),
        reference=f"{reference} ITRS",
    )
    return longitude, latitude, height, xyz


def _explicit_location(
    config: InstrumentConfig,
    *,
    module_importer: Callable[[str], object],
) -> tuple[object, tuple[float, float, float, tuple[float, float, float]]] | None:
    if config.location is None:
        return None
    coordinates = _coordinates_module(
        module_importer,
        reference="instrument.location",
    )
    try:
        location = coordinates.EarthLocation.from_geodetic(
            config.location.longitude_deg,
            config.location.latitude_deg,
            config.location.height_m,
        )
    except Exception as error:
        raise CoordinateFrameError(
            "instrument.location: could not construct the explicit Earth location"
        ) from error
    facts = earth_location_facts(
        location,
        reference="instrument.location",
        module_importer=module_importer,
    )
    return location, facts


def _location_from_facts(
    facts: SourceLocationFacts,
    *,
    module_importer: Callable[[str], object],
) -> object:
    coordinates = _coordinates_module(
        module_importer,
        reference=facts.reference,
    )
    try:
        return coordinates.EarthLocation.from_geocentric(
            facts.itrs_xyz_m[0],
            facts.itrs_xyz_m[1],
            facts.itrs_xyz_m[2],
            unit="m",
        )
    except Exception as error:
        raise CoordinateFrameError(
            f"{facts.reference}: could not reconstruct the embedded Earth location"
        ) from error


def _default_enu_from_ecef(
    *,
    module_importer: Callable[[str], object],
    source_reference: str,
) -> EcefToEnu:
    try:
        utils = cast(_PyuvdataUtilsModule, module_importer("pyuvdata.utils"))
        converter = utils.ENU_from_ECEF
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalInstrumentDependencyError(
            f"{source_reference}: relative-ECEF instrument coordinates require "
            "pyuvdata; install pyuvdata"
        ) from error
    if not callable(converter):
        raise OptionalInstrumentDependencyError(
            f"{source_reference}: pyuvdata.utils.ENU_from_ECEF is unavailable "
            "for instrument coordinates"
        )
    return converter


def _field_source(loaded: LoadedInstrumentSource) -> AntennaFieldSource:
    if loaded.source_kind == "known_telescope":
        return AntennaFieldSource.KNOWN_TELESCOPE
    if loaded.position_frame == "relative_ecef":
        return AntennaFieldSource.EMBEDDED_DATASET
    return AntennaFieldSource.LAYOUT_FILE


def resolve_instrument_source(
    config: InstrumentConfig,
    *,
    dataset_loader: DatasetTelescopeLoader | None = None,
    known_telescope_loader: KnownTelescopeLoader | None = None,
    module_importer: ModuleImporter = import_module,
    internet_config: InternetConfiguration | None = None,
    pyuvdata_version: str | None = None,
    enu_from_ecef: EcefToEnu | None = None,
) -> StagedInstrument:
    """Resolve one typed source into deterministic diameter-incomplete ENU state."""
    # Imported here so source adapters can depend on this module's stable error
    # taxonomy without creating an import cycle.
    from radiosim.io.instrument_sources import load_instrument_source

    loaded = load_instrument_source(
        config.source,
        dataset_loader=dataset_loader,
        known_telescope_loader=known_telescope_loader,
        module_importer=module_importer,
        internet_config=internet_config,
        pyuvdata_version=pyuvdata_version,
    )
    explicit = _explicit_location(config, module_importer=module_importer)
    embedded = loaded.embedded_location
    separation: float | None = None

    if explicit is None and embedded is None:
        raise CoordinateFrameError(
            f"{loaded.source_reference}: selected source has no Earth location"
        )
    if explicit is not None:
        canonical_object, explicit_facts = explicit
        canonical_xyz = explicit_facts[3]
        location_source = AntennaFieldSource.EXPLICIT_CONFIG
        location_reference = "instrument.location"
        longitude, latitude, height = explicit_facts[:3]
    else:
        assert embedded is not None
        canonical_object = _location_from_facts(
            embedded,
            module_importer=module_importer,
        )
        canonical_xyz = embedded.itrs_xyz_m
        location_source = _field_source(loaded)
        location_reference = embedded.reference
        longitude = embedded.longitude_deg
        latitude = embedded.latitude_deg
        height = embedded.height_m

    explicit_xyz = explicit[1][3] if explicit is not None else None
    embedded_xyz = embedded.itrs_xyz_m if embedded is not None else None
    if explicit_xyz is not None and embedded_xyz is not None:
        difference = np.asarray(explicit_xyz, dtype=np.float64) - np.asarray(
            embedded_xyz, dtype=np.float64
        )
        separation = float(np.linalg.norm(difference))
        if not math.isfinite(separation):
            raise CoordinateFrameError(
                f"{loaded.source_reference}: location separation is non-finite"
            )
        if separation > _LOCATION_MATCH_THRESHOLD_M:
            raise InstrumentLocationMismatchError(
                f"{loaded.source_reference}: explicit and embedded locations differ "
                f"by {separation:.9g} m, exceeding the 1.0 m threshold"
            )

    resolved_location = ResolvedEarthLocation(
        longitude_deg=longitude,
        latitude_deg=latitude,
        height_m=height,
        itrs_xyz_m=canonical_xyz,
        source=location_source,
        reference=location_reference,
    )

    positions: tuple[tuple[float, float, float], ...]
    if loaded.position_frame == "enu":
        positions = tuple(antenna.position_m for antenna in loaded.antennas)
    else:
        if embedded is None:
            raise CoordinateFrameError(
                f"{loaded.source_reference}: relative-ECEF positions require an "
                "embedded Earth location"
            )
        relative = np.asarray(
            [antenna.position_m for antenna in loaded.antennas], dtype=np.float64
        )
        absolute = relative + np.asarray(embedded.itrs_xyz_m, dtype=np.float64)
        if not np.isfinite(absolute).all():
            raise InvalidAntennaPositionError(
                f"{loaded.source_reference}: absolute ECEF antenna positions "
                "must be finite"
            )
        converter = enu_from_ecef or _default_enu_from_ecef(
            module_importer=module_importer,
            source_reference=loaded.source_reference,
        )
        try:
            converted = converter(absolute, center_loc=canonical_object)
            converted_array = np.asarray(converted)
        except Exception as error:
            raise CoordinateFrameError(
                f"{loaded.source_reference}: public ECEF-to-ENU conversion failed"
            ) from error
        if converted_array.shape != (len(loaded.antennas), 3):
            raise CoordinateFrameError(
                f"{loaded.source_reference}: ECEF-to-ENU conversion returned "
                "an incoherent shape"
            )
        positions = tuple(
            normalize_position(
                converted_array[index],
                reference=loaded.antennas[index].source_record,
            )
            for index in range(len(loaded.antennas))
        )

    selected_source = _field_source(loaded)
    staged_antennas = tuple(
        StagedAntenna(
            number=antenna.number,
            name=antenna.name,
            position_enu_m=positions[index],
            source_diameter_m=antenna.source_diameter_m,
            mount_type=antenna.mount_type,
            beam_id=antenna.beam_id,
            number_source=(
                AntennaFieldSource.GENERATED
                if antenna.number_generated
                else selected_source
            ),
            name_source=(
                AntennaFieldSource.GENERATED
                if antenna.name_generated
                else selected_source
            ),
            position_source=selected_source,
            diameter_source=(
                selected_source if antenna.source_diameter_m is not None else None
            ),
            mount_source=(selected_source if antenna.mount_type is not None else None),
            beam_id_source=(selected_source if antenna.beam_id is not None else None),
            source_record=antenna.source_record,
        )
        for index, antenna in enumerate(loaded.antennas)
    )
    return StagedInstrument(
        location=resolved_location,
        antennas=staged_antennas,
        provenance=StagedInstrumentProvenance(
            source_kind=loaded.source_kind,
            source_reference=loaded.source_reference,
            source_format=loaded.source_format,
            registry_policy=loaded.registry_policy,
            explicit_telescope_name=loaded.explicit_telescope_name,
            embedded_telescope_name=loaded.embedded_telescope_name,
            embedded_location_itrs_xyz_m=embedded_xyz,
            explicit_location_itrs_xyz_m=explicit_xyz,
            location_separation_m=separation,
            pyuvdata_version=loaded.pyuvdata_version,
            source_sha256=loaded.source_sha256,
        ),
    )


__all__ = [
    "AntennaIdentifierError",
    "CoordinateFrameError",
    "DiameterResolutionError",
    "DuplicateAntennaError",
    "EcefToEnu",
    "EmptyInstrumentError",
    "InstrumentFormatError",
    "InstrumentLocationMismatchError",
    "InstrumentResolutionError",
    "InstrumentSourceError",
    "InvalidAntennaPositionError",
    "OptionalInstrumentDependencyError",
    "StagedAntenna",
    "StagedInstrument",
    "StagedInstrumentProvenance",
    "TelescopeNotFoundError",
    "resolve_instrument_source",
]
