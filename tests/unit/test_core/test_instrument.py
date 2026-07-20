"""Tier 2C immutable canonical instrument-model contract tests."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

import radiosim
import radiosim.core as core
import radiosim.core.instrument as instrument_module
from radiosim.core.instrument import (
    AntennaFieldSource,
    AntennaId,
    AntennaProvenance,
    InstrumentProvenance,
    ResolvedAntenna,
    ResolvedEarthLocation,
    ResolvedInstrument,
    _build_instrument_indexes,
    _canonical_instrument_fingerprint_payload,
    _compute_instrument_sha256,
    _create_resolved_instrument,
)

PUBLIC_MODEL_NAMES = (
    "AntennaId",
    "AntennaFieldSource",
    "ResolvedEarthLocation",
    "AntennaProvenance",
    "ResolvedAntenna",
    "InstrumentProvenance",
    "ResolvedInstrument",
)

EXPECTED_FIELDS = {
    AntennaId: ("number", "name"),
    ResolvedEarthLocation: (
        "longitude_deg",
        "latitude_deg",
        "height_m",
        "itrs_xyz_m",
        "source",
        "reference",
    ),
    AntennaProvenance: (
        "identity_source",
        "position_source",
        "diameter_source",
        "source_diameter_m",
        "mount_source",
        "beam_id_source",
        "source_record",
    ),
    ResolvedAntenna: (
        "id",
        "position_enu_m",
        "diameter_m",
        "mount_type",
        "beam_id",
        "provenance",
    ),
    InstrumentProvenance: (
        "schema_version",
        "source_kind",
        "source_reference",
        "source_format",
        "registry_policy",
        "telescope_name_source",
        "location_source",
        "source_location_itrs_xyz_m",
        "location_separation_m",
        "pyuvdata_version",
        "source_sha256",
        "instrument_sha256",
    ),
    ResolvedInstrument: ("name", "location", "antennas", "provenance"),
}

REFERENCE_PAYLOAD = {
    "schema_version": "radiosim.instrument.v1",
    "name": "Array Å",
    "location": {
        "longitude_deg": -180.0,
        "latitude_deg": 45.0,
        "height_m": -12.5,
        "itrs_xyz_m": [1.0, 2.0, 3.0],
        "source": "explicit_config",
        "provenance": {"location_source": "explicit_config"},
    },
    "antennas": [
        {
            "number": 0,
            "name": "ANT Å",
            "position_enu_m": [0.0, 0.0, 0.0],
            "diameter_m": 12.0,
            "mount_type": "fixed",
            "beam_id": "beam α",
            "provenance": {
                "identity_source": "layout_file",
                "position_source": "layout_file",
                "diameter_source": "explicit_override",
                "mount_source": "layout_file",
                "beam_id_source": "layout_file",
            },
        },
        {
            "number": 5,
            "name": "ANT5",
            "position_enu_m": [10.5, -2.0, 3.0],
            "diameter_m": 15.0,
            "mount_type": None,
            "beam_id": 7,
            "provenance": {
                "identity_source": "layout_file",
                "position_source": "layout_file",
                "diameter_source": "layout_file",
                "mount_source": None,
                "beam_id_source": None,
            },
        },
    ],
    "provenance": {"telescope_name_source": "explicit_config"},
}
REFERENCE_JSON = json.dumps(
    REFERENCE_PAYLOAD,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
)
REFERENCE_SHA256 = "c57da5979e17852d23c51f15ba6006dac4536ff8b0c44aab3f9caeefbc6cdbf6"


def _antenna_provenance(
    *,
    identity_source: AntennaFieldSource = AntennaFieldSource.LAYOUT_FILE,
    position_source: AntennaFieldSource = AntennaFieldSource.LAYOUT_FILE,
    diameter_source: AntennaFieldSource = AntennaFieldSource.LAYOUT_FILE,
    source_diameter_m: float | None = 15.0,
    mount_source: AntennaFieldSource | None = None,
    beam_id_source: AntennaFieldSource | None = None,
    source_record: str = "row:1",
) -> AntennaProvenance:
    return AntennaProvenance(
        identity_source=identity_source,
        position_source=position_source,
        diameter_source=diameter_source,
        source_diameter_m=source_diameter_m,
        mount_source=mount_source,
        beam_id_source=beam_id_source,
        source_record=source_record,
    )


def _antenna(
    number: int,
    name: str,
    *,
    position: Any = (0.0, 0.0, 0.0),
    diameter_m: float = 15.0,
    mount_type: str | None = None,
    beam_id: int | str | None = None,
    provenance: AntennaProvenance | None = None,
) -> ResolvedAntenna:
    return ResolvedAntenna(
        id=AntennaId(number, name),
        position_enu_m=position,
        diameter_m=diameter_m,
        mount_type=mount_type,
        beam_id=beam_id,
        provenance=provenance or _antenna_provenance(),
    )


def _reference_parts() -> tuple[
    ResolvedEarthLocation,
    tuple[ResolvedAntenna, ResolvedAntenna],
]:
    location = ResolvedEarthLocation(
        longitude_deg=180.0,
        latitude_deg=45.0,
        height_m=-12.5,
        itrs_xyz_m=np.array([1.0, 2.0, 3.0]),
        source=AntennaFieldSource.EXPLICIT_CONFIG,
        reference=" config:location ",
    )
    ant0 = _antenna(
        0,
        " ANT A\N{COMBINING RING ABOVE} ",
        diameter_m=12.0,
        mount_type=" fixed ",
        beam_id=" beam α ",
        provenance=_antenna_provenance(
            diameter_source=AntennaFieldSource.EXPLICIT_OVERRIDE,
            source_diameter_m=14.0,
            mount_source=AntennaFieldSource.LAYOUT_FILE,
            beam_id_source=AntennaFieldSource.LAYOUT_FILE,
            source_record=" row:2 ",
        ),
    )
    ant5 = _antenna(
        5,
        "ANT5",
        position=(10.5, -2.0, 3.0),
        diameter_m=15.0,
        beam_id=np.int64(7),
    )
    return location, (ant5, ant0)


def _instrument_provenance(
    instrument_sha256: str,
    *,
    source_reference: str = "/tmp/reference-layout",
    telescope_name_source: AntennaFieldSource = AntennaFieldSource.EXPLICIT_CONFIG,
    location_source: AntennaFieldSource = AntennaFieldSource.EXPLICIT_CONFIG,
) -> InstrumentProvenance:
    return InstrumentProvenance(
        schema_version="radiosim.instrument.v1",
        source_kind="fixture",
        source_reference=source_reference,
        source_format="radiosim",
        registry_policy="offline",
        telescope_name_source=telescope_name_source,
        location_source=location_source,
        source_location_itrs_xyz_m=[1.0, 2.0, 3.0],
        location_separation_m=0.25,
        pyuvdata_version="3.2.1",
        source_sha256="a" * 64,
        instrument_sha256=instrument_sha256,
    )


def _resolved_instrument(
    *,
    name: str = " Array A\N{COMBINING RING ABOVE} ",
    location: ResolvedEarthLocation | None = None,
    antennas: Any = None,
    source_reference: str = "/tmp/reference-layout",
    telescope_name_source: AntennaFieldSource = AntennaFieldSource.EXPLICIT_CONFIG,
    location_source: AntennaFieldSource = AntennaFieldSource.EXPLICIT_CONFIG,
) -> ResolvedInstrument:
    reference_location, reference_antennas = _reference_parts()
    resolved_location = location or reference_location
    resolved_antennas = reference_antennas if antennas is None else antennas
    fingerprint = _compute_instrument_sha256(
        name,
        resolved_location,
        resolved_antennas,
        telescope_name_source=telescope_name_source,
        location_source=location_source,
    )
    return ResolvedInstrument(
        name=name,
        location=resolved_location,
        antennas=resolved_antennas,
        provenance=_instrument_provenance(
            fingerprint,
            source_reference=source_reference,
            telescope_name_source=telescope_name_source,
            location_source=location_source,
        ),
    )


def _assert_json_primitive_tree(value: Any) -> None:
    assert not is_dataclass(value)
    assert not isinstance(
        value,
        (Enum, tuple, MappingProxyType, np.ndarray, np.generic, Path),
    )
    if isinstance(value, dict):
        assert type(value) is dict
        for key, item in value.items():
            assert type(key) is str
            _assert_json_primitive_tree(item)
    elif isinstance(value, list):
        assert type(value) is list
        for item in value:
            _assert_json_primitive_tree(item)
    else:
        assert value is None or type(value) in {str, int, float, bool}


@pytest.mark.parametrize("model_type,field_names", EXPECTED_FIELDS.items())
def test_public_dataclasses_have_exact_frozen_slotted_fields(model_type, field_names):
    assert is_dataclass(model_type)
    assert tuple(item.name for item in fields(model_type)) == field_names
    assert "__slots__" in model_type.__dict__
    assert "__dict__" not in model_type.__dict__


@pytest.mark.parametrize(
    "value,expected",
    [(0, 0), (2_147_483_647, 2_147_483_647), (np.int64(7), 7)],
)
def test_antenna_id_accepts_exact_number_boundaries_and_owns_builtin_int(
    value, expected
):
    antenna_id = AntennaId(value, "ANT")

    assert antenna_id.number == expected
    assert type(antenna_id.number) is int


@pytest.mark.parametrize(
    "value",
    [True, False, -1, 2_147_483_648, 1.0, 1.5, np.float64(2.0), "1", None],
)
def test_antenna_id_rejects_invalid_numbers(value):
    with pytest.raises((TypeError, ValueError)):
        AntennaId(value, "ANT")


def test_antenna_id_normalizes_name_without_case_folding():
    composed = AntennaId(3, "  Ånt-01  ")
    decomposed = AntennaId(3, "A\N{COMBINING RING ABOVE}nt-01")

    assert composed.name == "Ånt-01"
    assert type(composed.name) is str
    assert composed == decomposed
    assert hash(composed) == hash(decomposed)
    assert AntennaId(3, "ÅNT-01") != composed
    assert AntennaId(4, "007").name == "007"


@pytest.mark.parametrize("name", ["", "   ", 4, None])
def test_antenna_id_rejects_blank_or_nonstring_name(name):
    with pytest.raises((TypeError, ValueError)):
        AntennaId(0, name)


def test_field_source_enum_is_exact_stable_string_vocabulary():
    assert [member.value for member in AntennaFieldSource] == [
        "explicit_config",
        "explicit_override",
        "layout_file",
        "embedded_dataset",
        "known_telescope",
        "generated",
        "config_default",
    ]
    assert len(AntennaFieldSource.__members__) == 7
    assert str(AntennaFieldSource.LAYOUT_FILE) == "layout_file"
    assert json.dumps(AntennaFieldSource.LAYOUT_FILE) == '"layout_file"'
    assert hash(AntennaFieldSource.LAYOUT_FILE) == hash("layout_file")


@pytest.mark.parametrize(
    "longitude,expected",
    [
        (-180.0, -180.0),
        (180.0, -180.0),
        (540.0, -180.0),
        (-540.0, -180.0),
        (360.0, 0.0),
        (-0.0, 0.0),
        (179.999, 179.999),
    ],
)
def test_resolved_location_normalizes_longitude_to_canonical_interval(
    longitude, expected
):
    location = ResolvedEarthLocation(
        longitude,
        0.0,
        -5.0,
        (1.0, 2.0, 3.0),
        AntennaFieldSource.EXPLICIT_CONFIG,
        "config",
    )

    assert location.longitude_deg == expected
    assert -180.0 <= location.longitude_deg < 180.0
    assert math.copysign(1.0, location.longitude_deg) == 1.0 or expected != 0.0


@pytest.mark.parametrize("latitude", [-90.0, 90.0])
def test_resolved_location_accepts_latitude_boundaries(latitude):
    location = ResolvedEarthLocation(
        0.0,
        latitude,
        -100.0,
        (1.0, 2.0, 3.0),
        AntennaFieldSource.EMBEDDED_DATASET,
        "dataset",
    )

    assert location.latitude_deg == latitude
    assert location.height_m == -100.0


@pytest.mark.parametrize("latitude", [-90.0001, 90.0001])
def test_resolved_location_rejects_latitude_outside_boundaries(latitude):
    with pytest.raises(ValueError):
        ResolvedEarthLocation(
            0.0,
            latitude,
            0.0,
            (1.0, 2.0, 3.0),
            AntennaFieldSource.EXPLICIT_CONFIG,
            "config",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("longitude_deg", math.nan),
        ("longitude_deg", math.inf),
        ("latitude_deg", -math.inf),
        ("height_m", math.nan),
    ],
)
def test_resolved_location_rejects_nonfinite_scalar_fields(field, value):
    kwargs = {
        "longitude_deg": 0.0,
        "latitude_deg": 0.0,
        "height_m": 0.0,
        "itrs_xyz_m": (1.0, 2.0, 3.0),
        "source": AntennaFieldSource.EXPLICIT_CONFIG,
        "reference": "config",
    }
    kwargs[field] = value

    with pytest.raises((TypeError, ValueError)):
        ResolvedEarthLocation(**kwargs)


@pytest.mark.parametrize(
    "itrs",
    [
        (),
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
        (1.0, math.nan, 3.0),
        {0: 1.0, 1: 2.0, 2: 3.0},
        "123",
    ],
)
def test_resolved_location_requires_exact_finite_three_component_itrs(itrs):
    with pytest.raises((TypeError, ValueError)):
        ResolvedEarthLocation(
            0.0,
            0.0,
            0.0,
            itrs,
            AntennaFieldSource.EXPLICIT_CONFIG,
            "config",
        )


def test_resolved_location_copies_arrays_and_tuples_and_normalizes_reference():
    array = np.array([-0.0, np.float32(2.5), 3])
    tuple_value = (-0.0, 2.5, 3.0)
    from_array = ResolvedEarthLocation(
        np.float64(-0.0),
        np.float32(1.0),
        np.int64(-2),
        array,
        AntennaFieldSource.EXPLICIT_CONFIG,
        "  re\N{COMBINING ACUTE ACCENT}fe\N{COMBINING ACUTE ACCENT}rence  ",
    )
    from_tuple = ResolvedEarthLocation(
        0.0,
        1.0,
        -2.0,
        tuple_value,
        AntennaFieldSource.EXPLICIT_CONFIG,
        "référence",
    )
    array[:] = 99.0

    assert from_array == from_tuple
    assert from_array.itrs_xyz_m == (0.0, 2.5, 3.0)
    assert from_array.itrs_xyz_m is not tuple_value
    assert all(type(item) is float for item in from_array.itrs_xyz_m)
    assert all(math.copysign(1.0, item) == 1.0 for item in from_array.itrs_xyz_m)
    assert from_array.reference == "référence"


def test_resolved_location_rejects_invalid_source_and_reference():
    with pytest.raises((TypeError, ValueError)):
        ResolvedEarthLocation(0, 0, 0, (1, 2, 3), "layout_file", "config")
    with pytest.raises((TypeError, ValueError)):
        ResolvedEarthLocation(
            0,
            0,
            0,
            (1, 2, 3),
            AntennaFieldSource.LAYOUT_FILE,
            "   ",
        )


def test_antenna_provenance_normalizes_all_values_and_is_hashable():
    provenance = AntennaProvenance(
        AntennaFieldSource.GENERATED,
        AntennaFieldSource.LAYOUT_FILE,
        AntennaFieldSource.EXPLICIT_OVERRIDE,
        np.float64(14.0),
        AntennaFieldSource.KNOWN_TELESCOPE,
        AntennaFieldSource.LAYOUT_FILE,
        "  row:A\N{COMBINING RING ABOVE}  ",
    )

    assert provenance.source_diameter_m == 14.0
    assert type(provenance.source_diameter_m) is float
    assert provenance.source_record == "row:Å"
    assert isinstance(hash(provenance), int)


@pytest.mark.parametrize(
    "source_diameter",
    [0.0, -1.0, math.nan, math.inf, True, "14"],
)
def test_antenna_provenance_rejects_invalid_present_source_diameter(
    source_diameter,
):
    with pytest.raises((TypeError, ValueError)):
        _antenna_provenance(source_diameter_m=source_diameter)


def test_antenna_provenance_accepts_missing_optionals():
    provenance = _antenna_provenance(
        source_diameter_m=None,
        mount_source=None,
        beam_id_source=None,
    )

    assert provenance.source_diameter_m is None
    assert provenance.mount_source is None
    assert provenance.beam_id_source is None


@pytest.mark.parametrize(
    "field",
    ["identity_source", "position_source", "diameter_source", "mount_source"],
)
def test_antenna_provenance_rejects_invalid_source_labels(field):
    kwargs = {
        "identity_source": AntennaFieldSource.LAYOUT_FILE,
        "position_source": AntennaFieldSource.LAYOUT_FILE,
        "diameter_source": AntennaFieldSource.LAYOUT_FILE,
        "source_diameter_m": None,
        "mount_source": None,
        "beam_id_source": None,
        "source_record": "row:1",
    }
    kwargs[field] = "layout_file"

    with pytest.raises((TypeError, ValueError)):
        AntennaProvenance(**kwargs)


def test_antenna_provenance_rejects_blank_source_record():
    with pytest.raises((TypeError, ValueError)):
        _antenna_provenance(source_record="   ")


def test_resolved_antenna_copies_enu_and_normalizes_inert_metadata():
    position = np.array([-0.0, np.float32(2.5), 3])
    antenna = _antenna(
        4,
        "ANT4",
        position=position,
        diameter_m=np.float64(12.5),
        mount_type="  Alt-Az  ",
        beam_id=" beam A\N{COMBINING RING ABOVE} ",
    )
    position[:] = 100.0

    assert antenna.position_enu_m == (0.0, 2.5, 3.0)
    assert all(type(item) is float for item in antenna.position_enu_m)
    assert antenna.diameter_m == 12.5
    assert type(antenna.diameter_m) is float
    assert antenna.mount_type == "Alt-Az"
    assert antenna.beam_id == "beam Å"
    assert isinstance(hash(antenna), int)


@pytest.mark.parametrize(
    "position",
    [
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
        (1.0, math.inf, 3.0),
        {"east": 1.0, "north": 2.0, "up": 3.0},
        "123",
    ],
)
def test_resolved_antenna_requires_exact_finite_enu(position):
    with pytest.raises((TypeError, ValueError)):
        _antenna(1, "ANT1", position=position)


@pytest.mark.parametrize("diameter", [0, -1, math.nan, math.inf, True, "14"])
def test_resolved_antenna_requires_positive_finite_diameter(diameter):
    with pytest.raises((TypeError, ValueError)):
        _antenna(1, "ANT1", diameter_m=diameter)


def test_resolved_antenna_has_no_hidden_diameter_default_or_feed_fields():
    with pytest.raises(TypeError):
        ResolvedAntenna(
            id=AntennaId(1, "ANT1"),
            position_enu_m=(0.0, 0.0, 0.0),
            mount_type=None,
            beam_id=None,
            provenance=_antenna_provenance(),
        )
    assert "feed" not in {item.name for item in fields(ResolvedAntenna)}
    assert not hasattr(_antenna(1, "ANT1"), "feed")


@pytest.mark.parametrize("mount", ["", "   ", 2, False])
def test_resolved_antenna_rejects_invalid_present_mount(mount):
    with pytest.raises((TypeError, ValueError)):
        _antenna(1, "ANT1", mount_type=mount)


@pytest.mark.parametrize("beam_id", [True, False, "", "   ", 2.5, object()])
def test_resolved_antenna_rejects_invalid_beam_id(beam_id):
    with pytest.raises((TypeError, ValueError)):
        _antenna(1, "ANT1", beam_id=beam_id)


def test_resolved_antenna_accepts_none_and_builtin_integer_beam_id():
    missing = _antenna(1, "ANT1", beam_id=None)
    numbered = _antenna(2, "ANT2", beam_id=np.int64(9))

    assert missing.beam_id is None
    assert numbered.beam_id == 9
    assert type(numbered.beam_id) is int


def test_resolved_antenna_requires_canonical_nested_types():
    with pytest.raises((TypeError, ValueError)):
        ResolvedAntenna(
            id=(1, "ANT1"),
            position_enu_m=(0.0, 0.0, 0.0),
            diameter_m=14.0,
            mount_type=None,
            beam_id=None,
            provenance=_antenna_provenance(),
        )
    with pytest.raises((TypeError, ValueError)):
        ResolvedAntenna(
            id=AntennaId(1, "ANT1"),
            position_enu_m=(0.0, 0.0, 0.0),
            diameter_m=14.0,
            mount_type=None,
            beam_id=None,
            provenance={},
        )


def test_instrument_provenance_normalizes_and_copy_owns_values():
    source_location = (-0.0, np.float32(2.5), np.int64(3))
    provenance = InstrumentProvenance(
        schema_version=" radiosim.instrument.v1 ",
        source_kind=" fixture ",
        source_reference=" /tmp/A\N{COMBINING RING ABOVE} ",
        source_format=" radiosim ",
        registry_policy=" offline ",
        telescope_name_source=AntennaFieldSource.EXPLICIT_CONFIG,
        location_source=AntennaFieldSource.EXPLICIT_CONFIG,
        source_location_itrs_xyz_m=source_location,
        location_separation_m=np.float64(-0.0),
        pyuvdata_version=" 3.2.1 ",
        source_sha256="a" * 64,
        instrument_sha256="b" * 64,
    )

    assert provenance.schema_version == "radiosim.instrument.v1"
    assert provenance.source_kind == "fixture"
    assert provenance.source_reference == "/tmp/Å"
    assert provenance.source_format == "radiosim"
    assert provenance.registry_policy == "offline"
    assert provenance.source_location_itrs_xyz_m == (0.0, 2.5, 3.0)
    assert provenance.source_location_itrs_xyz_m is not source_location
    assert provenance.location_separation_m == 0.0
    assert math.copysign(1.0, provenance.location_separation_m) == 1.0
    assert provenance.pyuvdata_version == "3.2.1"
    assert isinstance(hash(provenance), int)


def test_instrument_provenance_accepts_all_missing_optional_values():
    provenance = InstrumentProvenance(
        "radiosim.instrument.v1",
        "known_telescope",
        "HERA",
        None,
        None,
        AntennaFieldSource.KNOWN_TELESCOPE,
        AntennaFieldSource.KNOWN_TELESCOPE,
        None,
        None,
        None,
        None,
        "b" * 64,
    )

    assert provenance.source_format is None
    assert provenance.registry_policy is None
    assert provenance.source_location_itrs_xyz_m is None
    assert provenance.location_separation_m is None
    assert provenance.pyuvdata_version is None
    assert provenance.source_sha256 is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "radiosim.instrument.v2"),
        ("source_kind", "  "),
        ("source_reference", Path("layout.txt")),
        ("source_format", ""),
        ("registry_policy", "   "),
        ("telescope_name_source", "explicit_config"),
        ("location_source", "explicit_config"),
        ("location_separation_m", -0.001),
        ("location_separation_m", math.inf),
        ("source_sha256", "A" * 64),
        ("source_sha256", "a" * 63),
        ("source_sha256", "g" * 64),
        ("instrument_sha256", "B" * 64),
        ("instrument_sha256", "b" * 65),
        ("instrument_sha256", "z" * 64),
    ],
)
def test_instrument_provenance_rejects_invalid_values(field, value):
    kwargs = {
        "schema_version": "radiosim.instrument.v1",
        "source_kind": "fixture",
        "source_reference": "layout",
        "source_format": None,
        "registry_policy": None,
        "telescope_name_source": AntennaFieldSource.EXPLICIT_CONFIG,
        "location_source": AntennaFieldSource.EXPLICIT_CONFIG,
        "source_location_itrs_xyz_m": None,
        "location_separation_m": None,
        "pyuvdata_version": None,
        "source_sha256": None,
        "instrument_sha256": "b" * 64,
    }
    kwargs[field] = value

    with pytest.raises((TypeError, ValueError)):
        InstrumentProvenance(**kwargs)


@pytest.mark.parametrize(
    "source_location",
    [
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
        (1.0, math.nan, 3.0),
        {"x": 1.0, "y": 2.0, "z": 3.0},
    ],
)
def test_instrument_provenance_rejects_invalid_source_location(source_location):
    kwargs = _instrument_provenance("b" * 64)
    values = {item.name: getattr(kwargs, item.name) for item in fields(kwargs)}
    values["source_location_itrs_xyz_m"] = source_location

    with pytest.raises((TypeError, ValueError)):
        InstrumentProvenance(**values)


def test_instrument_inventory_is_nonempty_sorted_copy_owned_and_hashable():
    location, reversed_antennas = _reference_parts()
    caller_list = list(reversed_antennas)
    instrument = _resolved_instrument(location=location, antennas=caller_list)
    caller_list.clear()

    assert tuple(antenna.id.number for antenna in instrument.antennas) == (0, 5)
    assert instrument.antennas[0] is reversed_antennas[1]
    assert instrument.antennas[1] is reversed_antennas[0]
    assert len(instrument.antennas) == 2
    assert isinstance(hash(instrument), int)


def test_instrument_copies_caller_tuple_even_when_already_ordered():
    location, reversed_antennas = _reference_parts()
    ordered = tuple(sorted(reversed_antennas, key=lambda antenna: antenna.id.number))
    instrument = _resolved_instrument(location=location, antennas=ordered)

    assert instrument.antennas == ordered
    assert instrument.antennas is not ordered


def test_instrument_rejects_empty_inventory():
    location, _ = _reference_parts()
    with pytest.raises(ValueError):
        _resolved_instrument(location=location, antennas=())


@pytest.mark.parametrize("duplicate", ["number", "name"])
def test_instrument_rejects_duplicate_number_or_normalized_name(duplicate):
    location, _ = _reference_parts()
    if duplicate == "number":
        antennas = (_antenna(1, "A"), _antenna(1, "B"))
    else:
        antennas = (_antenna(1, " Å "), _antenna(2, "A\N{COMBINING RING ABOVE}"))

    with pytest.raises(ValueError):
        _resolved_instrument(location=location, antennas=antennas)


def test_instrument_names_are_case_sensitive_for_uniqueness():
    location, _ = _reference_parts()
    antennas = (_antenna(2, "ant"), _antenna(1, "ANT"))
    instrument = _resolved_instrument(location=location, antennas=antennas)

    assert tuple(item.id.name for item in instrument.antennas) == ("ANT", "ant")


def test_instrument_order_variations_compare_and_hash_equally():
    location, reversed_antennas = _reference_parts()
    first = _resolved_instrument(location=location, antennas=reversed_antennas)
    second = _resolved_instrument(
        location=location,
        antennas=tuple(reversed(reversed_antennas)),
    )

    assert first == second
    assert hash(first) == hash(second)


def test_instrument_rejects_invalid_nested_types_and_hash_mismatch():
    location, antennas = _reference_parts()
    valid_hash = _compute_instrument_sha256(
        "Array Å",
        location,
        antennas,
        telescope_name_source=AntennaFieldSource.EXPLICIT_CONFIG,
        location_source=AntennaFieldSource.EXPLICIT_CONFIG,
    )
    provenance = _instrument_provenance(valid_hash)

    with pytest.raises((TypeError, ValueError)):
        ResolvedInstrument("Array Å", object(), antennas, provenance)
    with pytest.raises((TypeError, ValueError)):
        ResolvedInstrument("Array Å", location, (object(),), provenance)
    with pytest.raises((TypeError, ValueError)):
        ResolvedInstrument("Array Å", location, antennas, object())
    with pytest.raises(ValueError):
        ResolvedInstrument(
            "Array Å",
            location,
            antennas,
            replace(provenance, instrument_sha256="0" * 64),
        )


def test_nested_models_reject_caller_mutable_subclasses():
    def mutable_copy(value):
        def mutable_setattr(self, name, replacement):
            object.__setattr__(self, name, replacement)

        mutable_type = type(
            f"Mutable{type(value).__name__}",
            (type(value),),
            {"__setattr__": mutable_setattr},
        )
        return mutable_type(*(getattr(value, item.name) for item in fields(value)))

    antenna = _antenna(1, "ANT1")
    mutable_id = mutable_copy(antenna.id)
    mutable_id.name = "changed after construction"
    assert mutable_id.name == "changed after construction"

    with pytest.raises(TypeError):
        replace(antenna, id=mutable_copy(antenna.id))
    with pytest.raises(TypeError):
        replace(antenna, provenance=mutable_copy(antenna.provenance))

    instrument = _resolved_instrument()
    with pytest.raises(TypeError):
        replace(instrument, location=mutable_copy(instrument.location))
    with pytest.raises(TypeError):
        replace(
            instrument,
            antennas=(
                mutable_copy(instrument.antennas[0]),
                instrument.antennas[1],
            ),
        )
    with pytest.raises(TypeError):
        replace(instrument, provenance=mutable_copy(instrument.provenance))


def test_all_public_models_are_frozen():
    location, antennas = _reference_parts()
    values = [
        antennas[0].id,
        location,
        antennas[0].provenance,
        antennas[0],
        _instrument_provenance(REFERENCE_SHA256),
        _resolved_instrument(),
    ]

    for value in values:
        first_field = fields(value)[0].name
        with pytest.raises(FrozenInstanceError):
            setattr(value, first_field, getattr(value, first_field))


def test_private_indexes_are_fresh_read_only_and_reference_canonical_objects():
    instrument = _resolved_instrument()
    indexes = _build_instrument_indexes(instrument.antennas)

    assert isinstance(indexes.by_number, MappingProxyType)
    assert isinstance(indexes.by_name, MappingProxyType)
    assert tuple(indexes.by_number) == (0, 5)
    assert tuple(indexes.by_name) == ("ANT Å", "ANT5")
    assert indexes.by_number[0] is instrument.antennas[0]
    assert indexes.by_name["ANT5"] is instrument.antennas[1]
    with pytest.raises(TypeError):
        indexes.by_number[9] = instrument.antennas[0]
    with pytest.raises(TypeError):
        del indexes.by_name["ANT5"]


@pytest.mark.parametrize(
    "antennas",
    [
        (_antenna(1, "A"), _antenna(1, "B")),
        (_antenna(1, "A"), _antenna(2, "A")),
    ],
)
def test_private_index_builder_rejects_duplicates(antennas):
    with pytest.raises(ValueError):
        _build_instrument_indexes(antennas)


def test_reference_payload_and_hash_are_independently_fixed():
    assert hashlib.sha256(REFERENCE_JSON.encode("utf-8")).hexdigest() == (
        REFERENCE_SHA256
    )
    location, antennas = _reference_parts()

    payload = _canonical_instrument_fingerprint_payload(
        " Array A\N{COMBINING RING ABOVE} ",
        location,
        antennas,
        telescope_name_source=AntennaFieldSource.EXPLICIT_CONFIG,
        location_source=AntennaFieldSource.EXPLICIT_CONFIG,
    )
    fingerprint = _compute_instrument_sha256(
        " Array A\N{COMBINING RING ABOVE} ",
        location,
        antennas,
        telescope_name_source=AntennaFieldSource.EXPLICIT_CONFIG,
        location_source=AntennaFieldSource.EXPLICIT_CONFIG,
    )

    assert payload == REFERENCE_PAYLOAD
    assert fingerprint == REFERENCE_SHA256
    assert len(fingerprint) == 64
    assert fingerprint == fingerprint.lower()


def test_private_factory_reuses_canonical_hash_and_returns_exact_public_model():
    location, antennas = _reference_parts()

    instrument = _create_resolved_instrument(
        name=" Array A\N{COMBINING RING ABOVE} ",
        location=location,
        antennas=antennas,
        source_kind="fixture",
        source_reference="/tmp/reference-layout",
        source_format="radiosim",
        registry_policy="offline",
        telescope_name_source=AntennaFieldSource.EXPLICIT_CONFIG,
        location_source=AntennaFieldSource.EXPLICIT_CONFIG,
        source_location_itrs_xyz_m=(1.0, 2.0, 3.0),
        location_separation_m=0.25,
        pyuvdata_version="3.2.1",
        source_sha256="a" * 64,
    )

    assert type(instrument) is ResolvedInstrument
    assert instrument == _resolved_instrument()
    assert instrument.provenance.instrument_sha256 == REFERENCE_SHA256
    assert "_create_resolved_instrument" not in instrument_module.__all__


def test_fingerprint_is_repeatable_order_unicode_and_negative_zero_independent():
    location, antennas = _reference_parts()
    negative_zero_antennas = tuple(
        replace(
            antenna,
            position_enu_m=tuple(
                -0.0 if value == 0.0 else value for value in antenna.position_enu_m
            ),
        )
        for antenna in antennas
    )
    negative_zero_location = replace(
        location,
        longitude_deg=-0.0,
        itrs_xyz_m=(-0.0, 2.0, 3.0),
    )
    zero_location = replace(
        location,
        longitude_deg=0.0,
        itrs_xyz_m=(0.0, 2.0, 3.0),
    )
    kwargs = {
        "telescope_name_source": AntennaFieldSource.EXPLICIT_CONFIG,
        "location_source": AntennaFieldSource.EXPLICIT_CONFIG,
    }

    first = _compute_instrument_sha256(
        "Array A\N{COMBINING RING ABOVE}",
        zero_location,
        antennas,
        **kwargs,
    )
    second = _compute_instrument_sha256(
        "Array Å",
        negative_zero_location,
        tuple(reversed(negative_zero_antennas)),
        **kwargs,
    )

    assert first == second
    assert first == _compute_instrument_sha256(
        "Array Å", zero_location, antennas, **kwargs
    )


@pytest.mark.parametrize(
    "change",
    ["position", "diameter", "identity", "mount", "beam_id", "field_source"],
)
def test_fingerprint_changes_with_canonical_inventory_content(change):
    location, antennas = _reference_parts()
    canonical = tuple(sorted(antennas, key=lambda antenna: antenna.id.number))
    changed = list(canonical)
    if change == "position":
        changed[0] = replace(changed[0], position_enu_m=(0.0, 1.0, 0.0))
    elif change == "diameter":
        changed[0] = replace(changed[0], diameter_m=12.5)
    elif change == "identity":
        changed[0] = replace(changed[0], id=AntennaId(0, "OTHER"))
    elif change == "mount":
        changed[0] = replace(changed[0], mount_type="equatorial")
    elif change == "beam_id":
        changed[0] = replace(changed[0], beam_id="other")
    else:
        changed[0] = replace(
            changed[0],
            provenance=replace(
                changed[0].provenance,
                position_source=AntennaFieldSource.EMBEDDED_DATASET,
            ),
        )
    kwargs = {
        "telescope_name_source": AntennaFieldSource.EXPLICIT_CONFIG,
        "location_source": AntennaFieldSource.EXPLICIT_CONFIG,
    }

    assert _compute_instrument_sha256("Array Å", location, canonical, **kwargs) != (
        _compute_instrument_sha256("Array Å", location, changed, **kwargs)
    )


def test_fingerprint_changes_with_instrument_level_field_source_labels():
    location, antennas = _reference_parts()
    first = _compute_instrument_sha256(
        "Array Å",
        location,
        antennas,
        telescope_name_source=AntennaFieldSource.EXPLICIT_CONFIG,
        location_source=AntennaFieldSource.EXPLICIT_CONFIG,
    )
    changed_name_source = _compute_instrument_sha256(
        "Array Å",
        location,
        antennas,
        telescope_name_source=AntennaFieldSource.KNOWN_TELESCOPE,
        location_source=AntennaFieldSource.EXPLICIT_CONFIG,
    )
    changed_location_source = _compute_instrument_sha256(
        "Array Å",
        location,
        antennas,
        telescope_name_source=AntennaFieldSource.EXPLICIT_CONFIG,
        location_source=AntennaFieldSource.EMBEDDED_DATASET,
    )

    assert len({first, changed_name_source, changed_location_source}) == 3


def test_fingerprint_excludes_transport_path_and_location_reference():
    location, antennas = _reference_parts()
    other_reference = replace(location, reference="/different/temporary/path")
    kwargs = {
        "telescope_name_source": AntennaFieldSource.EXPLICIT_CONFIG,
        "location_source": AntennaFieldSource.EXPLICIT_CONFIG,
    }

    assert _compute_instrument_sha256("Array Å", location, antennas, **kwargs) == (
        _compute_instrument_sha256(
            "Array Å",
            other_reference,
            antennas,
            **kwargs,
        )
    )
    assert _resolved_instrument(
        source_reference="/tmp/one"
    ).provenance.instrument_sha256 == (
        _resolved_instrument(source_reference="/tmp/two").provenance.instrument_sha256
    )


def test_snapshot_has_exact_shape_and_every_provenance_field():
    instrument = _resolved_instrument()
    snapshot = instrument.to_snapshot()

    assert snapshot == {
        "schema_version": "radiosim.instrument.v1",
        "instrument_sha256": REFERENCE_SHA256,
        "name": "Array Å",
        "source": {
            "kind": "fixture",
            "reference": "/tmp/reference-layout",
            "format": "radiosim",
            "registry_policy": "offline",
            "source_sha256": "a" * 64,
            "pyuvdata_version": "3.2.1",
            "telescope_name_source": "explicit_config",
        },
        "location": {
            "longitude_deg": -180.0,
            "latitude_deg": 45.0,
            "height_m": -12.5,
            "itrs_xyz_m": [1.0, 2.0, 3.0],
            "source": "explicit_config",
            "reference": "config:location",
            "location_source": "explicit_config",
            "source_location_itrs_xyz_m": [1.0, 2.0, 3.0],
            "separation_m": 0.25,
        },
        "antennas": [
            {
                "number": 0,
                "name": "ANT Å",
                "position_enu_m": [0.0, 0.0, 0.0],
                "diameter_m": 12.0,
                "source_diameter_m": 14.0,
                "mount_type": "fixed",
                "beam_id": "beam α",
                "provenance": {
                    "identity_source": "layout_file",
                    "position_source": "layout_file",
                    "diameter_source": "explicit_override",
                    "mount_source": "layout_file",
                    "beam_id_source": "layout_file",
                    "source_record": "row:2",
                },
            },
            {
                "number": 5,
                "name": "ANT5",
                "position_enu_m": [10.5, -2.0, 3.0],
                "diameter_m": 15.0,
                "source_diameter_m": 15.0,
                "mount_type": None,
                "beam_id": 7,
                "provenance": {
                    "identity_source": "layout_file",
                    "position_source": "layout_file",
                    "diameter_source": "layout_file",
                    "mount_source": None,
                    "beam_id_source": None,
                    "source_record": "row:1",
                },
            },
        ],
    }
    assert "baseline_selection" not in snapshot
    assert "instrument_resolution" not in snapshot
    assert snapshot["instrument_sha256"] == instrument.provenance.instrument_sha256


def test_snapshot_is_fresh_recursively_json_safe_and_model_independent():
    instrument = _resolved_instrument()
    first = instrument.to_snapshot()
    second = instrument.to_snapshot()

    assert first == second
    assert first is not second
    assert first["location"] is not second["location"]
    assert first["antennas"] is not second["antennas"]
    _assert_json_primitive_tree(first)
    json.dumps(first, allow_nan=False, ensure_ascii=False)

    first["location"]["itrs_xyz_m"][0] = 999.0
    first["antennas"][0]["position_enu_m"][0] = 999.0
    first["antennas"].append({"number": 99})

    assert instrument.location.itrs_xyz_m == (1.0, 2.0, 3.0)
    assert instrument.antennas[0].position_enu_m == (0.0, 0.0, 0.0)
    assert len(instrument.antennas) == 2
    assert second == instrument.to_snapshot()


def test_public_exports_are_exact_and_share_object_identity():
    assert tuple(instrument_module.__all__) == PUBLIC_MODEL_NAMES + (
        "ResolvedBaseline",
        "BaselineSelectionCriteriaSnapshot",
        "BaselineSelectionProvenance",
        "ResolvedBaselineSelection",
    )
    for name in PUBLIC_MODEL_NAMES:
        direct = getattr(instrument_module, name)
        assert getattr(core, name) is direct
        assert getattr(radiosim, name) is direct
        assert name in core.__all__
        assert name in radiosim.__all__

    for removed in ("read_antenna_positions", "generate_baselines"):
        assert not hasattr(core, removed)
        assert removed not in core.__all__
        assert not hasattr(radiosim, removed)
        assert removed not in radiosim.__all__


def test_private_helpers_are_not_exported():
    private_names = {
        "_InstrumentIndexes",
        "_build_instrument_indexes",
        "_canonical_instrument_fingerprint_payload",
        "_compute_instrument_sha256",
        "_create_resolved_instrument",
    }
    for name in private_names:
        assert name not in instrument_module.__all__
        assert name not in core.__all__
        assert name not in radiosim.__all__


def test_model_module_is_lightweight_and_contains_no_resolution_surface():
    source = Path(instrument_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    class_names: set[str] = set()
    function_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.ClassDef):
            class_names.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.add(node.name)

    assert imported_roots <= {
        "__future__",
        "collections",
        "dataclasses",
        "enum",
        "hashlib",
        "json",
        "math",
        "numbers",
        "re",
        "types",
        "typing",
        "unicodedata",
    }
    assert class_names.isdisjoint(
        {
            "ResolvedInstrumentState",
        }
    )
    assert function_names.isdisjoint(
        {
            "read_antenna_positions",
            "generate_baselines",
            "select_baselines",
            "load_instrument",
            "resolve_instrument",
            "write_measurement_set",
        }
    )


def test_public_state_contains_only_models_enums_and_builtin_immutable_values():
    instrument = _resolved_instrument()

    assert type(instrument.name) is str
    assert type(instrument.antennas) is tuple
    assert all(type(value) is float for value in instrument.location.itrs_xyz_m)
    for antenna in instrument.antennas:
        assert type(antenna.id.number) is int
        assert type(antenna.id.name) is str
        assert type(antenna.position_enu_m) is tuple
        assert all(type(value) is float for value in antenna.position_enu_m)
        assert type(antenna.diameter_m) is float
        assert not isinstance(antenna.beam_id, np.generic)
        assert not isinstance(antenna.provenance.source_diameter_m, np.generic)
    assert not isinstance(instrument.provenance.source_location_itrs_xyz_m, np.ndarray)
    assert not isinstance(instrument.provenance.source_reference, Path)
    assert not isinstance(instrument.antennas, Mapping)
