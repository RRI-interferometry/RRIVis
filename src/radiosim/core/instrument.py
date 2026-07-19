"""Immutable canonical antenna and instrument value models.

This module owns already-resolved scientific values only.  It performs no source
loading, coordinate conversion, precedence resolution, baseline construction, or
runtime integration.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, TypeVar, cast

_MAX_ANTENNA_NUMBER = 2_147_483_647
_INSTRUMENT_SCHEMA_VERSION = "radiosim.instrument.v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_T = TypeVar("_T")


def _require_instance(
    value: object,
    expected_type: type[_T],
    *,
    field_name: str,
) -> _T:
    if type(value) is not expected_type:
        raise TypeError(f"{field_name} must be a {expected_type.__name__}")
    return cast(_T, value)


def _normalize_nonblank_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized:
        raise ValueError(f"{field_name} must be nonblank")
    return str(normalized)


def _normalize_optional_string(
    value: object | None,
    *,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    return _normalize_nonblank_string(value, field_name=field_name)


def _normalize_integer(
    value: object,
    *,
    field_name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}")
    return normalized


def _normalize_finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be representable as a float") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    if normalized == 0.0:
        return 0.0
    return normalized


def _normalize_positive_float(value: object, *, field_name: str) -> float:
    normalized = _normalize_finite_float(value, field_name=field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _normalize_nonnegative_float(value: object, *, field_name: str) -> float:
    normalized = _normalize_finite_float(value, field_name=field_name)
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be nonnegative")
    return normalized


def _copy_three_floats(
    value: object,
    *,
    field_name: str,
) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{field_name} must be a one-dimensional array-like value")
    try:
        copied: tuple[object, ...] = tuple(cast(Iterable[object], value))
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be a one-dimensional array-like value"
        ) from exc
    if len(copied) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    normalized = tuple(
        _normalize_finite_float(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(copied)
    )
    return (normalized[0], normalized[1], normalized[2])


class AntennaFieldSource(StrEnum):
    """Stable vocabulary describing the origin of a resolved antenna field."""

    EXPLICIT_CONFIG = "explicit_config"
    EXPLICIT_OVERRIDE = "explicit_override"
    LAYOUT_FILE = "layout_file"
    EMBEDDED_DATASET = "embedded_dataset"
    KNOWN_TELESCOPE = "known_telescope"
    GENERATED = "generated"
    CONFIG_DEFAULT = "config_default"


def _require_field_source(
    value: object,
    *,
    field_name: str,
) -> AntennaFieldSource:
    if not isinstance(value, AntennaFieldSource):
        raise TypeError(f"{field_name} must be an AntennaFieldSource")
    return value


def _require_optional_field_source(
    value: object | None,
    *,
    field_name: str,
) -> AntennaFieldSource | None:
    if value is None:
        return None
    return _require_field_source(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class AntennaId:
    """Canonical case-sensitive antenna identity."""

    number: int
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "number",
            _normalize_integer(
                self.number,
                field_name="number",
                minimum=0,
                maximum=_MAX_ANTENNA_NUMBER,
            ),
        )
        object.__setattr__(
            self,
            "name",
            _normalize_nonblank_string(self.name, field_name="name"),
        )


@dataclass(frozen=True, slots=True)
class ResolvedEarthLocation:
    """Canonical Earth location and its already-computed ITRS coordinates."""

    longitude_deg: float
    latitude_deg: float
    height_m: float
    itrs_xyz_m: tuple[float, float, float]
    source: AntennaFieldSource
    reference: str

    def __post_init__(self) -> None:
        longitude = _normalize_finite_float(
            self.longitude_deg,
            field_name="longitude_deg",
        )
        if not -180.0 <= longitude < 180.0:
            longitude = ((longitude + 180.0) % 360.0) - 180.0
        if longitude == 0.0:
            longitude = 0.0
        latitude = _normalize_finite_float(
            self.latitude_deg,
            field_name="latitude_deg",
        )
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("latitude_deg must be in [-90, 90]")

        object.__setattr__(self, "longitude_deg", longitude)
        object.__setattr__(self, "latitude_deg", latitude)
        object.__setattr__(
            self,
            "height_m",
            _normalize_finite_float(self.height_m, field_name="height_m"),
        )
        object.__setattr__(
            self,
            "itrs_xyz_m",
            _copy_three_floats(self.itrs_xyz_m, field_name="itrs_xyz_m"),
        )
        object.__setattr__(
            self,
            "source",
            _require_field_source(self.source, field_name="source"),
        )
        object.__setattr__(
            self,
            "reference",
            _normalize_nonblank_string(self.reference, field_name="reference"),
        )


@dataclass(frozen=True, slots=True)
class AntennaProvenance:
    """Field-level source facts for one resolved antenna."""

    identity_source: AntennaFieldSource
    position_source: AntennaFieldSource
    diameter_source: AntennaFieldSource
    source_diameter_m: float | None
    mount_source: AntennaFieldSource | None
    beam_id_source: AntennaFieldSource | None
    source_record: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "identity_source",
            _require_field_source(
                self.identity_source,
                field_name="identity_source",
            ),
        )
        object.__setattr__(
            self,
            "position_source",
            _require_field_source(
                self.position_source,
                field_name="position_source",
            ),
        )
        object.__setattr__(
            self,
            "diameter_source",
            _require_field_source(
                self.diameter_source,
                field_name="diameter_source",
            ),
        )
        if self.source_diameter_m is not None:
            object.__setattr__(
                self,
                "source_diameter_m",
                _normalize_positive_float(
                    self.source_diameter_m,
                    field_name="source_diameter_m",
                ),
            )
        object.__setattr__(
            self,
            "mount_source",
            _require_optional_field_source(
                self.mount_source,
                field_name="mount_source",
            ),
        )
        object.__setattr__(
            self,
            "beam_id_source",
            _require_optional_field_source(
                self.beam_id_source,
                field_name="beam_id_source",
            ),
        )
        object.__setattr__(
            self,
            "source_record",
            _normalize_nonblank_string(
                self.source_record,
                field_name="source_record",
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedAntenna:
    """Canonical antenna position, diameter, inert metadata, and provenance."""

    id: AntennaId
    position_enu_m: tuple[float, float, float]
    diameter_m: float
    mount_type: str | None
    beam_id: int | str | None
    provenance: AntennaProvenance

    def __post_init__(self) -> None:
        canonical_id = _require_instance(self.id, AntennaId, field_name="id")
        canonical_provenance = _require_instance(
            self.provenance,
            AntennaProvenance,
            field_name="provenance",
        )
        object.__setattr__(self, "id", canonical_id)
        object.__setattr__(self, "provenance", canonical_provenance)

        object.__setattr__(
            self,
            "position_enu_m",
            _copy_three_floats(self.position_enu_m, field_name="position_enu_m"),
        )
        object.__setattr__(
            self,
            "diameter_m",
            _normalize_positive_float(self.diameter_m, field_name="diameter_m"),
        )
        object.__setattr__(
            self,
            "mount_type",
            _normalize_optional_string(self.mount_type, field_name="mount_type"),
        )

        beam_id = self.beam_id
        if beam_id is None:
            normalized_beam_id: int | str | None = None
        elif isinstance(beam_id, str):
            normalized_beam_id = _normalize_nonblank_string(
                beam_id,
                field_name="beam_id",
            )
        else:
            normalized_beam_id = _normalize_integer(
                beam_id,
                field_name="beam_id",
            )
        object.__setattr__(self, "beam_id", normalized_beam_id)


def _normalize_sha256(value: object, *, field_name: str) -> str:
    normalized = _normalize_nonblank_string(value, field_name=field_name)
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            f"{field_name} must be a lowercase 64-character hexadecimal SHA-256"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class InstrumentProvenance:
    """Versioned source and fingerprint facts for a resolved instrument."""

    schema_version: str
    source_kind: str
    source_reference: str
    source_format: str | None
    registry_policy: str | None
    telescope_name_source: AntennaFieldSource
    location_source: AntennaFieldSource
    source_location_itrs_xyz_m: tuple[float, float, float] | None
    location_separation_m: float | None
    pyuvdata_version: str | None
    source_sha256: str | None
    instrument_sha256: str

    def __post_init__(self) -> None:
        schema_version = _normalize_nonblank_string(
            self.schema_version,
            field_name="schema_version",
        )
        if schema_version != _INSTRUMENT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_INSTRUMENT_SCHEMA_VERSION!r}")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(
            self,
            "source_kind",
            _normalize_nonblank_string(self.source_kind, field_name="source_kind"),
        )
        object.__setattr__(
            self,
            "source_reference",
            _normalize_nonblank_string(
                self.source_reference,
                field_name="source_reference",
            ),
        )
        object.__setattr__(
            self,
            "source_format",
            _normalize_optional_string(
                self.source_format,
                field_name="source_format",
            ),
        )
        object.__setattr__(
            self,
            "registry_policy",
            _normalize_optional_string(
                self.registry_policy,
                field_name="registry_policy",
            ),
        )
        object.__setattr__(
            self,
            "telescope_name_source",
            _require_field_source(
                self.telescope_name_source,
                field_name="telescope_name_source",
            ),
        )
        object.__setattr__(
            self,
            "location_source",
            _require_field_source(
                self.location_source,
                field_name="location_source",
            ),
        )
        if self.source_location_itrs_xyz_m is not None:
            object.__setattr__(
                self,
                "source_location_itrs_xyz_m",
                _copy_three_floats(
                    self.source_location_itrs_xyz_m,
                    field_name="source_location_itrs_xyz_m",
                ),
            )
        if self.location_separation_m is not None:
            object.__setattr__(
                self,
                "location_separation_m",
                _normalize_nonnegative_float(
                    self.location_separation_m,
                    field_name="location_separation_m",
                ),
            )
        object.__setattr__(
            self,
            "pyuvdata_version",
            _normalize_optional_string(
                self.pyuvdata_version,
                field_name="pyuvdata_version",
            ),
        )
        if self.source_sha256 is not None:
            object.__setattr__(
                self,
                "source_sha256",
                _normalize_sha256(
                    self.source_sha256,
                    field_name="source_sha256",
                ),
            )
        object.__setattr__(
            self,
            "instrument_sha256",
            _normalize_sha256(
                self.instrument_sha256,
                field_name="instrument_sha256",
            ),
        )


@dataclass(frozen=True, slots=True)
class _InstrumentIndexes:
    by_number: Mapping[int, ResolvedAntenna]
    by_name: Mapping[str, ResolvedAntenna]


def _build_instrument_indexes(
    antennas: Iterable[object],
) -> _InstrumentIndexes:
    """Build fresh immutable identity indexes for canonical antenna objects."""
    number_index: dict[int, ResolvedAntenna] = {}
    name_index: dict[str, ResolvedAntenna] = {}
    for value in antennas:
        antenna = _require_instance(
            value,
            ResolvedAntenna,
            field_name="instrument inventory item",
        )
        if antenna.id.number in number_index:
            raise ValueError(f"duplicate antenna number {antenna.id.number}")
        if antenna.id.name in name_index:
            raise ValueError(f"duplicate antenna name {antenna.id.name!r}")
        number_index[antenna.id.number] = antenna
        name_index[antenna.id.name] = antenna
    return _InstrumentIndexes(
        by_number=MappingProxyType(number_index),
        by_name=MappingProxyType(name_index),
    )


def _canonicalize_antennas(value: object) -> tuple[ResolvedAntenna, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("antennas must be an iterable of ResolvedAntenna values")
    try:
        copied: tuple[object, ...] = tuple(cast(Iterable[object], value))
    except TypeError as exc:
        raise TypeError(
            "antennas must be an iterable of ResolvedAntenna values"
        ) from exc
    if not copied:
        raise ValueError("antennas must contain at least one antenna")
    if any(type(item) is not ResolvedAntenna for item in copied):
        raise TypeError("antennas must contain only ResolvedAntenna values")
    typed = tuple(cast(ResolvedAntenna, item) for item in copied)
    ordered = tuple(sorted(typed, key=lambda antenna: antenna.id.number))
    _ = _build_instrument_indexes(ordered)
    return ordered


def _canonical_instrument_fingerprint_payload(
    name: str,
    location: ResolvedEarthLocation,
    antennas: Iterable[ResolvedAntenna],
    *,
    telescope_name_source: AntennaFieldSource,
    location_source: AntennaFieldSource,
) -> dict[str, Any]:
    """Return the exact path-independent canonical instrument hash payload.

    The payload contains resolved scientific values and the field-source labels that
    explain them.  Transport facts such as source paths, source-record locators, raw
    source hashes, dependency versions, registry policy, and pre-override source
    diameters are excluded.  Those facts remain available in explicit provenance and
    snapshots without making equivalent resolved inventories hash differently.
    """
    canonical_name = _normalize_nonblank_string(name, field_name="name")
    canonical_location = _require_instance(
        location,
        ResolvedEarthLocation,
        field_name="location",
    )
    canonical_antennas = _canonicalize_antennas(antennas)
    canonical_name_source = _require_field_source(
        telescope_name_source,
        field_name="telescope_name_source",
    )
    canonical_location_source = _require_field_source(
        location_source,
        field_name="location_source",
    )

    return {
        "schema_version": _INSTRUMENT_SCHEMA_VERSION,
        "name": canonical_name,
        "location": {
            "longitude_deg": canonical_location.longitude_deg,
            "latitude_deg": canonical_location.latitude_deg,
            "height_m": canonical_location.height_m,
            "itrs_xyz_m": list(canonical_location.itrs_xyz_m),
            "source": canonical_location.source.value,
            "provenance": {
                "location_source": canonical_location_source.value,
            },
        },
        "antennas": [
            {
                "number": antenna.id.number,
                "name": antenna.id.name,
                "position_enu_m": list(antenna.position_enu_m),
                "diameter_m": antenna.diameter_m,
                "mount_type": antenna.mount_type,
                "beam_id": antenna.beam_id,
                "provenance": {
                    "identity_source": antenna.provenance.identity_source.value,
                    "position_source": antenna.provenance.position_source.value,
                    "diameter_source": antenna.provenance.diameter_source.value,
                    "mount_source": (
                        antenna.provenance.mount_source.value
                        if antenna.provenance.mount_source is not None
                        else None
                    ),
                    "beam_id_source": (
                        antenna.provenance.beam_id_source.value
                        if antenna.provenance.beam_id_source is not None
                        else None
                    ),
                },
            }
            for antenna in canonical_antennas
        ],
        "provenance": {
            "telescope_name_source": canonical_name_source.value,
        },
    }


def _compute_instrument_sha256(
    name: str,
    location: ResolvedEarthLocation,
    antennas: Iterable[ResolvedAntenna],
    *,
    telescope_name_source: AntennaFieldSource,
    location_source: AntennaFieldSource,
) -> str:
    """Compute SHA-256 over the canonical UTF-8 JSON instrument payload."""
    payload = _canonical_instrument_fingerprint_payload(
        name,
        location,
        antennas,
        telescope_name_source=telescope_name_source,
        location_source=location_source,
    )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ResolvedInstrument:
    """Canonical ordered instrument inventory and its validated provenance."""

    name: str
    location: ResolvedEarthLocation
    antennas: tuple[ResolvedAntenna, ...]
    provenance: InstrumentProvenance

    def __post_init__(self) -> None:
        canonical_name = _normalize_nonblank_string(self.name, field_name="name")
        canonical_location = _require_instance(
            self.location,
            ResolvedEarthLocation,
            field_name="location",
        )
        canonical_provenance = _require_instance(
            self.provenance,
            InstrumentProvenance,
            field_name="provenance",
        )
        canonical_antennas = _canonicalize_antennas(self.antennas)
        expected_sha256 = _compute_instrument_sha256(
            canonical_name,
            canonical_location,
            canonical_antennas,
            telescope_name_source=canonical_provenance.telescope_name_source,
            location_source=canonical_provenance.location_source,
        )
        if canonical_provenance.instrument_sha256 != expected_sha256:
            raise ValueError(
                "provenance.instrument_sha256 does not match canonical instrument "
                "content"
            )

        object.__setattr__(self, "name", canonical_name)
        object.__setattr__(self, "antennas", canonical_antennas)

    def to_snapshot(self) -> dict[str, Any]:
        """Return a fresh deterministic JSON-safe instrument snapshot."""
        provenance = self.provenance
        source_location = provenance.source_location_itrs_xyz_m
        return {
            "schema_version": provenance.schema_version,
            "instrument_sha256": provenance.instrument_sha256,
            "name": self.name,
            "source": {
                "kind": provenance.source_kind,
                "reference": provenance.source_reference,
                "format": provenance.source_format,
                "registry_policy": provenance.registry_policy,
                "source_sha256": provenance.source_sha256,
                "pyuvdata_version": provenance.pyuvdata_version,
                "telescope_name_source": provenance.telescope_name_source.value,
            },
            "location": {
                "longitude_deg": self.location.longitude_deg,
                "latitude_deg": self.location.latitude_deg,
                "height_m": self.location.height_m,
                "itrs_xyz_m": list(self.location.itrs_xyz_m),
                "source": self.location.source.value,
                "reference": self.location.reference,
                "location_source": provenance.location_source.value,
                "source_location_itrs_xyz_m": (
                    list(source_location) if source_location is not None else None
                ),
                "separation_m": provenance.location_separation_m,
            },
            "antennas": [
                {
                    "number": antenna.id.number,
                    "name": antenna.id.name,
                    "position_enu_m": list(antenna.position_enu_m),
                    "diameter_m": antenna.diameter_m,
                    "source_diameter_m": antenna.provenance.source_diameter_m,
                    "mount_type": antenna.mount_type,
                    "beam_id": antenna.beam_id,
                    "provenance": {
                        "identity_source": (antenna.provenance.identity_source.value),
                        "position_source": (antenna.provenance.position_source.value),
                        "diameter_source": (antenna.provenance.diameter_source.value),
                        "mount_source": (
                            antenna.provenance.mount_source.value
                            if antenna.provenance.mount_source is not None
                            else None
                        ),
                        "beam_id_source": (
                            antenna.provenance.beam_id_source.value
                            if antenna.provenance.beam_id_source is not None
                            else None
                        ),
                        "source_record": antenna.provenance.source_record,
                    },
                }
                for antenna in self.antennas
            ],
        }


__all__ = [
    "AntennaId",
    "AntennaFieldSource",
    "ResolvedEarthLocation",
    "AntennaProvenance",
    "ResolvedAntenna",
    "InstrumentProvenance",
    "ResolvedInstrument",
]
