"""Tier 2E metadata merge and complete-diameter resolution contract tests."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from pydantic import ValidationError

import radiosim
import radiosim.core as core
import radiosim.core.instrument_resolution as resolution_module
from radiosim.core.instrument import (
    AntennaFieldSource,
    AntennaId,
    AntennaProvenance,
    InstrumentProvenance,
    ResolvedAntenna,
    ResolvedEarthLocation,
    ResolvedInstrument,
)
from radiosim.core.instrument_resolution import (
    DiameterResolutionError,
    DuplicateAntennaError,
    EmptyInstrumentError,
    InstrumentLocationMismatchError,
    InstrumentSourceError,
    StagedInstrument,
    UnknownDiameterOverrideError,
    _finalize_staged_instrument,
    _index_diameter_overrides,
    _resolve_final_diameters,
    _resolve_instrument_identity,
    resolve_instrument,
    resolve_instrument_source,
)
from radiosim.io.instrument_config import (
    AntennaDiameterOverrideConfig,
    AntennaNameReference,
    AntennaNumberReference,
    InstrumentConfig,
    InstrumentLocationConfig,
    KnownTelescopeSourceConfig,
    LayoutFileSourceConfig,
)


class FakeTelescope:
    """Mutable dependency object for deterministic dataset/registry tests."""

    def __init__(
        self,
        *,
        name: object = "Embedded Array",
        location: object | None = None,
        names: object | None = None,
        numbers: object | None = None,
        positions: object | None = None,
        diameters: object | None = None,
        mounts: object | None = None,
    ) -> None:
        self.name = name
        self.location = (
            EarthLocation.from_geodetic(
                116.67 * u.deg,
                -26.70 * u.deg,
                377.0 * u.m,
            )
            if location is None
            else location
        )
        self.antenna_names = np.array(["A1", "A0"]) if names is None else names
        self.antenna_numbers = (
            np.array([1, 0], dtype=np.int64) if numbers is None else numbers
        )
        self.antenna_positions = (
            np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            if positions is None
            else positions
        )
        self.antenna_diameters = diameters
        self.mount_type = mounts
        self.feeds = np.array(["x", "y"])


class FakeInternetConfig:
    def __init__(self) -> None:
        self.allow_internet = True

    @contextmanager
    def set_temp(self, attribute: str, value: object) -> Iterator[None]:
        assert attribute == "allow_internet"
        previous = self.allow_internet
        self.allow_internet = value
        try:
            yield
        finally:
            self.allow_internet = previous


def _location_config(location: EarthLocation | None = None) -> InstrumentLocationConfig:
    if location is None:
        location = EarthLocation.from_geodetic(
            116.67 * u.deg,
            -26.70 * u.deg,
            377.0 * u.m,
        )
    geodetic = location.to_geodetic()
    return InstrumentLocationConfig(
        longitude_deg=float(geodetic.lon.to_value(u.deg)),
        latitude_deg=float(geodetic.lat.to_value(u.deg)),
        height_m=float(geodetic.height.to_value(u.m)),
    )


def _write_radiosim(
    path: Path,
    *,
    rows: str = "A1 1 10 0 0\nA0 0 0 0 0\n",
    header: str = "Name Number E N U",
) -> None:
    path.write_text(f"{header}\n{rows}", encoding="utf-8")


def _local_config(
    path: Path,
    *,
    name: str = "Local Array",
    source_format: str = "radiosim",
    default: float | None = None,
    overrides: tuple[AntennaDiameterOverrideConfig, ...] = (),
) -> InstrumentConfig:
    return InstrumentConfig(
        source=LayoutFileSourceConfig(
            path=path,
            format=source_format,
            telescope_name=name,
        ),
        location=_location_config(),
        default_diameter_m=default,
        diameter_overrides=overrides,
    )


def _dataset_config(
    path: Path,
    *,
    source_format: str = "uvfits",
    explicit_name: str | None = None,
    location: InstrumentLocationConfig | None = None,
    default: float | None = 14.0,
    overrides: tuple[AntennaDiameterOverrideConfig, ...] = (),
) -> InstrumentConfig:
    return InstrumentConfig(
        source=LayoutFileSourceConfig(
            path=path,
            format=source_format,
            telescope_name=explicit_name,
        ),
        location=location,
        default_diameter_m=default,
        diameter_overrides=overrides,
    )


def _known_config(
    *,
    name: str = "Requested Array",
    default: float | None = 14.0,
    overrides: tuple[AntennaDiameterOverrideConfig, ...] = (),
) -> InstrumentConfig:
    return InstrumentConfig(
        source=KnownTelescopeSourceConfig(
            name=name,
            registry_policy="allow_network",
        ),
        default_diameter_m=default,
        diameter_overrides=overrides,
    )


def _number_override(number: int, diameter: float) -> AntennaDiameterOverrideConfig:
    return AntennaDiameterOverrideConfig(
        antenna=AntennaNumberReference(number=number),
        diameter_m=diameter,
    )


def _name_override(name: str, diameter: float) -> AntennaDiameterOverrideConfig:
    return AntennaDiameterOverrideConfig(
        antenna=AntennaNameReference(name=name),
        diameter_m=diameter,
    )


def _resolve_local(path: Path, **updates: Any) -> ResolvedInstrument:
    return resolve_instrument(_local_config(path, **updates))


def _resolve_dataset(
    path: Path,
    telescope: FakeTelescope,
    **updates: Any,
) -> ResolvedInstrument:
    return resolve_instrument(
        _dataset_config(path, **updates),
        dataset_loader=lambda _path, _format: telescope,
        pyuvdata_version="3.2.1",
    )


def _resolve_known(telescope: FakeTelescope, **updates: Any) -> ResolvedInstrument:
    return resolve_instrument(
        _known_config(**updates),
        known_telescope_loader=lambda _name: telescope,
        pyuvdata_version="3.2.1",
    )


def _staged_local(
    path: Path, **updates: Any
) -> tuple[InstrumentConfig, StagedInstrument]:
    config = _local_config(path, **updates)
    return config, resolve_instrument_source(config)


def _assert_public_tree_is_canonical(value: object) -> None:
    if isinstance(value, tuple):
        for item in value:
            _assert_public_tree_is_canonical(item)
        return
    if isinstance(
        value,
        (
            AntennaId,
            AntennaProvenance,
            InstrumentProvenance,
            ResolvedAntenna,
            ResolvedEarthLocation,
            ResolvedInstrument,
        ),
    ):
        for field in fields(value):
            _assert_public_tree_is_canonical(getattr(value, field.name))
        return
    assert value is None or isinstance(value, (str, int, float, AntennaFieldSource))
    assert not isinstance(value, (np.ndarray, np.generic, Path, Mapping, list, dict))


def test_local_explicit_identity_is_final_and_records_explicit_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)

    instrument = _resolve_local(
        path, name="  Local A\N{COMBINING RING ABOVE}rray  ", default=12.0
    )

    assert instrument.name == "Local Årray"
    assert instrument.provenance.telescope_name_source is (
        AntennaFieldSource.EXPLICIT_CONFIG
    )


def test_identity_helper_resolves_local_known_and_dataset_rules(tmp_path: Path) -> None:
    local_path = tmp_path / "layout.txt"
    _write_radiosim(local_path)
    local_config, local_staged = _staged_local(local_path, default=12.0)
    assert _resolve_instrument_identity(local_config, local_staged) == (
        "Local Array",
        AntennaFieldSource.EXPLICIT_CONFIG,
    )

    dataset_path = tmp_path / "array.uvfits"
    dataset_path.write_bytes(b"dataset")
    dataset_config = _dataset_config(dataset_path, explicit_name=None)
    dataset_staged = resolve_instrument_source(
        dataset_config,
        dataset_loader=lambda _path, _format: FakeTelescope(name="Embedded"),
        pyuvdata_version="3.2.1",
    )
    assert _resolve_instrument_identity(dataset_config, dataset_staged) == (
        "Embedded",
        AntennaFieldSource.EMBEDDED_DATASET,
    )

    known_config = _known_config(name="Requested")
    known_staged = resolve_instrument_source(
        known_config,
        known_telescope_loader=lambda _name: FakeTelescope(name="Ignored"),
        pyuvdata_version="3.2.1",
    )
    assert _resolve_instrument_identity(known_config, known_staged) == (
        "Requested",
        AntennaFieldSource.KNOWN_TELESCOPE,
    )


def test_known_requested_identity_wins_over_dependency_metadata() -> None:
    instrument = _resolve_known(FakeTelescope(name="Dependency Name"), name="Requested")

    assert instrument.name == "Requested"
    assert instrument.provenance.source_reference == "Requested"
    assert instrument.provenance.telescope_name_source is (
        AntennaFieldSource.KNOWN_TELESCOPE
    )


@pytest.mark.parametrize(
    ("explicit", "embedded", "expected", "winner"),
    [
        (None, "Embedded", "Embedded", AntennaFieldSource.EMBEDDED_DATASET),
        ("Explicit", None, "Explicit", AntennaFieldSource.EXPLICIT_CONFIG),
        ("Same", "Same", "Same", AntennaFieldSource.EXPLICIT_CONFIG),
        (
            "  A\N{COMBINING RING ABOVE}rray  ",
            "Årray",
            "Årray",
            AntennaFieldSource.EXPLICIT_CONFIG,
        ),
    ],
)
def test_dataset_identity_precedence(
    tmp_path: Path,
    explicit: str | None,
    embedded: str | None,
    expected: str,
    winner: AntennaFieldSource,
) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"same-science")

    instrument = _resolve_dataset(
        path,
        FakeTelescope(name=embedded),
        explicit_name=explicit,
    )

    assert instrument.name == expected
    assert instrument.provenance.telescope_name_source is winner


@pytest.mark.parametrize(
    ("explicit", "embedded"),
    [("Array", "array"), ("Explicit", "Different")],
)
def test_dataset_identity_mismatch_is_explicit_source_error(
    tmp_path: Path,
    explicit: str,
    embedded: str,
) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"dataset")

    with pytest.raises(InstrumentSourceError, match="identity|name"):
        _resolve_dataset(
            path,
            FakeTelescope(name=embedded),
            explicit_name=explicit,
        )


def test_dataset_missing_both_identity_sources_fails(tmp_path: Path) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"dataset")

    with pytest.raises(InstrumentSourceError, match="identity|name"):
        _resolve_dataset(path, FakeTelescope(name=None), explicit_name=None)


def test_explicit_only_location_and_provenance_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)

    instrument = _resolve_local(path, default=12.0)

    assert instrument.location.source is AntennaFieldSource.EXPLICIT_CONFIG
    assert instrument.provenance.location_source is AntennaFieldSource.EXPLICIT_CONFIG
    assert instrument.provenance.source_location_itrs_xyz_m is None
    assert instrument.provenance.location_separation_m is None


def test_embedded_only_location_and_exact_source_itrs_are_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"dataset")
    telescope = FakeTelescope()

    instrument = _resolve_dataset(path, telescope)
    source_xyz = tuple(
        float(value.to_value(u.m))
        for value in (telescope.location.x, telescope.location.y, telescope.location.z)
    )

    assert instrument.location.source is AntennaFieldSource.EMBEDDED_DATASET
    assert instrument.provenance.location_source is (
        AntennaFieldSource.EMBEDDED_DATASET
    )
    assert instrument.provenance.source_location_itrs_xyz_m == source_xyz
    assert instrument.provenance.location_separation_m is None


def test_matching_explicit_and_embedded_location_facts_are_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"dataset")
    telescope = FakeTelescope()

    instrument = _resolve_dataset(
        path,
        telescope,
        location=_location_config(telescope.location),
    )

    assert instrument.location.source is AntennaFieldSource.EXPLICIT_CONFIG
    assert instrument.provenance.location_source is AntennaFieldSource.EXPLICIT_CONFIG
    assert instrument.provenance.source_location_itrs_xyz_m is not None
    assert instrument.provenance.location_separation_m == pytest.approx(0.0, abs=1e-6)


def test_finalization_reuses_staged_location_without_coordinate_work(
    tmp_path: Path,
) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)
    config, staged = _staged_local(path, default=12.0)

    instrument = _finalize_staged_instrument(config, staged)

    assert instrument.location is staged.location


def test_location_mismatch_error_passes_through_final_resolver(tmp_path: Path) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"dataset")
    embedded = EarthLocation.from_geodetic(116.67, -26.70, 377.0)
    explicit = EarthLocation.from_geocentric(
        embedded.x.to_value(u.m) + 2.0,
        embedded.y.to_value(u.m),
        embedded.z.to_value(u.m),
        unit=u.m,
    )

    with pytest.raises(InstrumentLocationMismatchError):
        _resolve_dataset(
            path,
            FakeTelescope(location=embedded),
            location=_location_config(explicit),
        )


@pytest.mark.parametrize(
    ("row", "expected_name"),
    [
        ("1 2 3 12 STATION ANTENNA", "ANTENNA"),
        ("1 2 3 12 STATION", "STATION"),
        ("1 2 3", "ANT000"),
    ],
)
def test_casa_generated_identity_provenance(
    tmp_path: Path,
    row: str,
    expected_name: str,
) -> None:
    path = tmp_path / "array.cfg"
    path.write_text(f"#coordsys=LOC\n{row}\n", encoding="utf-8")

    instrument = resolve_instrument(
        _local_config(path, source_format="casa_loc", default=14.0)
    )

    assert instrument.antennas[0].id == AntennaId(0, expected_name)
    assert instrument.antennas[0].provenance.identity_source is (
        AntennaFieldSource.GENERATED
    )


def test_non_casa_identity_is_never_marked_generated(tmp_path: Path) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)

    instrument = _resolve_local(path, default=12.0)

    assert all(
        antenna.provenance.identity_source is AntennaFieldSource.LAYOUT_FILE
        for antenna in instrument.antennas
    )


def test_final_inventory_is_canonical_and_one_antenna_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "one.txt"
    _write_radiosim(path, rows="ONLY 7 1 2 3\n")

    instrument = _resolve_local(path, default=11.0)

    assert instrument.antennas == (
        ResolvedAntenna(
            id=AntennaId(7, "ONLY"),
            position_enu_m=(1.0, 2.0, 3.0),
            diameter_m=11.0,
            mount_type=None,
            beam_id=None,
            provenance=instrument.antennas[0].provenance,
        ),
    )


def test_zero_and_duplicate_source_failures_pass_through(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    _write_radiosim(empty, rows="")
    with pytest.raises(EmptyInstrumentError):
        _resolve_local(empty, default=14.0)

    duplicate = tmp_path / "duplicate.txt"
    _write_radiosim(duplicate, rows="A 0 0 0 0\nB 0 1 0 0\n")
    with pytest.raises(DuplicateAntennaError):
        _resolve_local(duplicate, default=14.0)


def test_source_diameters_complete_inventory_without_default(tmp_path: Path) -> None:
    path = tmp_path / "source.txt"
    _write_radiosim(
        path,
        header="Name Number E N U Diameter",
        rows="A1 1 10 0 0 11\nA0 0 0 0 0 10\n",
    )

    instrument = _resolve_local(path)

    assert tuple(antenna.diameter_m for antenna in instrument.antennas) == (10.0, 11.0)
    assert all(
        antenna.provenance.diameter_source is AntennaFieldSource.LAYOUT_FILE
        for antenna in instrument.antennas
    )


def test_default_completes_all_missing_source_diameters(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"
    _write_radiosim(path)

    instrument = _resolve_local(path, default=13.0)

    assert tuple(antenna.diameter_m for antenna in instrument.antennas) == (13.0, 13.0)
    assert all(
        antenna.provenance.diameter_source is AntennaFieldSource.CONFIG_DEFAULT
        for antenna in instrument.antennas
    )
    assert all(
        antenna.provenance.source_diameter_m is None for antenna in instrument.antennas
    )


def test_partial_row_diameters_use_source_before_default(tmp_path: Path) -> None:
    path = tmp_path / "partial.cfg"
    path.write_text(
        "#coordsys=ENU\n0 0 0\n1 0 0 21 STATION\n",
        encoding="utf-8",
    )

    instrument = resolve_instrument(
        _local_config(path, source_format="casa_loc", default=17.0)
    )

    assert tuple(antenna.diameter_m for antenna in instrument.antennas) == (17.0, 21.0)
    assert tuple(
        antenna.provenance.diameter_source for antenna in instrument.antennas
    ) == (AntennaFieldSource.CONFIG_DEFAULT, AntennaFieldSource.LAYOUT_FILE)


def test_diameter_helper_applies_override_source_default_precedence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.cfg"
    path.write_text(
        "#coordsys=ENU\n0 0 0\n1 0 0 21 STATION\n2 0 0\n",
        encoding="utf-8",
    )
    config = _local_config(
        path,
        source_format="casa_loc",
        default=17.0,
        overrides=(_number_override(2, 25.0),),
    )
    staged = resolve_instrument_source(config)

    resolved = _resolve_final_diameters(config, staged)

    assert resolved == {
        0: (17.0, AntennaFieldSource.CONFIG_DEFAULT),
        1: (21.0, AntennaFieldSource.LAYOUT_FILE),
        2: (25.0, AntennaFieldSource.EXPLICIT_OVERRIDE),
    }


def test_mixed_exact_overrides_produce_heterogeneous_complete_inventory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)
    overrides = (_number_override(0, 12.0), _name_override("A1", 18.0))

    instrument = _resolve_local(path, default=14.0, overrides=overrides)

    assert tuple(antenna.diameter_m for antenna in instrument.antennas) == (12.0, 18.0)
    assert all(
        antenna.provenance.diameter_source is AntennaFieldSource.EXPLICIT_OVERRIDE
        for antenna in instrument.antennas
    )


def test_override_index_helper_uses_exact_namespaces_and_fresh_storage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)
    config, staged = _staged_local(
        path,
        default=14.0,
        overrides=(_number_override(0, 12.0), _name_override("A1", 18.0)),
    )

    first = _index_diameter_overrides(config, staged)
    second = _index_diameter_overrides(config, staged)

    assert first == {0: 12.0, 1: 18.0}
    assert first is not second
    first[0] = 99.0
    assert second == {0: 12.0, 1: 18.0}


@pytest.mark.parametrize(
    "overrides",
    [
        (_number_override(99, 12.0),),
        (_name_override("missing", 12.0),),
        (_number_override(0, 12.0), _name_override("A0", 12.0)),
    ],
)
def test_override_index_helper_rejects_unknown_and_duplicate_targets(
    tmp_path: Path,
    overrides: tuple[AntennaDiameterOverrideConfig, ...],
) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)
    config, staged = _staged_local(path, default=14.0, overrides=overrides)

    with pytest.raises(UnknownDiameterOverrideError):
        _index_diameter_overrides(config, staged)


def test_override_wins_source_and_retains_source_diameter(tmp_path: Path) -> None:
    path = tmp_path / "source.txt"
    _write_radiosim(
        path,
        header="Name Number E N U Diameter",
        rows="A0 0 0 0 0 10\nA1 1 1 0 0 11\n",
    )

    instrument = _resolve_local(
        path,
        default=99.0,
        overrides=(_name_override("A0", 25.0),),
    )

    assert tuple(antenna.diameter_m for antenna in instrument.antennas) == (25.0, 11.0)
    assert instrument.antennas[0].provenance.source_diameter_m == 10.0
    assert instrument.antennas[0].provenance.diameter_source is (
        AntennaFieldSource.EXPLICIT_OVERRIDE
    )
    assert instrument.antennas[1].provenance.diameter_source is (
        AntennaFieldSource.LAYOUT_FILE
    )


def test_override_wins_configured_default(tmp_path: Path) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)

    instrument = _resolve_local(
        path,
        default=14.0,
        overrides=(_number_override(0, 22.0),),
    )

    assert tuple(antenna.diameter_m for antenna in instrument.antennas) == (22.0, 14.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ((_number_override(99, 12.0),), "number 99"),
        ((_name_override("missing", 12.0),), "name 'missing'"),
        ((_name_override("a0", 12.0),), "name 'a0'"),
        ((_name_override("1", 12.0),), "name '1'"),
        (
            (_number_override(0, 12.0), _number_override(0, 12.0)),
            "multiple diameter overrides",
        ),
        (
            (_name_override("A0", 12.0), _name_override("A0", 12.0)),
            "multiple diameter overrides",
        ),
        (
            (_number_override(0, 12.0), _name_override("A0", 12.0)),
            "multiple diameter overrides",
        ),
    ],
)
def test_unknown_and_duplicate_override_targets_are_stable_errors(
    tmp_path: Path,
    overrides: tuple[AntennaDiameterOverrideConfig, ...],
    message: str,
) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)

    with pytest.raises(UnknownDiameterOverrideError, match=message):
        _resolve_local(path, default=14.0, overrides=overrides)


def test_override_error_order_follows_config_tuple_and_leaves_input_untouched(
    tmp_path: Path,
) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)
    config = _local_config(
        path,
        default=14.0,
        overrides=(_name_override("first-missing", 12.0), _number_override(99, 13.0)),
    )
    before = config.model_dump(mode="json")

    with pytest.raises(UnknownDiameterOverrideError, match="first-missing"):
        resolve_instrument(config)

    assert config.model_dump(mode="json") == before


def test_missing_diameters_are_aggregated_in_canonical_order(tmp_path: Path) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path, rows="THREE 3 3 0 0\nONE 1 1 0 0\nTWO 2 2 0 0\n")

    with pytest.raises(DiameterResolutionError) as raised:
        _resolve_local(path)

    message = str(raised.value)
    assert "1/'ONE'" in message
    assert "2/'TWO'" in message
    assert "3/'THREE'" in message
    assert (
        message.index("1/'ONE'") < message.index("2/'TWO'") < message.index("3/'THREE'")
    )


def test_one_missing_diameter_has_no_hidden_fallback(tmp_path: Path) -> None:
    path = tmp_path / "partial.cfg"
    path.write_text("#coordsys=ENU\n0 0 0 21 A\n1 0 0\n", encoding="utf-8")

    with pytest.raises(DiameterResolutionError, match="1/'ANT001'"):
        resolve_instrument(_local_config(path, source_format="casa_loc"))


@pytest.mark.parametrize("value", [math.nan, math.inf, 0.0, -1.0, True, "12"])
def test_invalid_present_staged_diameter_never_falls_through(
    tmp_path: Path,
    value: object,
) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)
    config, staged = _staged_local(path, default=14.0)
    invalid = replace(
        staged.antennas[0],
        source_diameter_m=value,  # type: ignore[arg-type]
        diameter_source=AntennaFieldSource.LAYOUT_FILE,
    )
    staged = replace(staged, antennas=(invalid, *staged.antennas[1:]))

    with pytest.raises(DiameterResolutionError):
        _finalize_staged_instrument(config, staged)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("default_diameter_m", True),
        ("default_diameter_m", "14"),
        ("default_diameter_m", math.nan),
        ("default_diameter_m", math.inf),
        ("default_diameter_m", 0.0),
        ("default_diameter_m", -1.0),
        ("override", True),
        ("override", "14"),
        ("override", math.nan),
        ("override", math.inf),
        ("override", 0.0),
        ("override", -1.0),
    ],
)
def test_invalid_configured_diameters_fail_in_strict_schema(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)
    data: dict[str, Any] = {
        "source": {
            "kind": "layout_file",
            "path": path,
            "format": "radiosim",
            "telescope_name": "Array",
        },
        "location": {
            "longitude_deg": 116.67,
            "latitude_deg": -26.70,
            "height_m": 377.0,
        },
    }
    if field == "default_diameter_m":
        data[field] = value
    else:
        data["diameter_overrides"] = [
            {
                "antenna": {"kind": "number", "number": 0},
                "diameter_m": value,
            }
        ]

    with pytest.raises(ValidationError):
        InstrumentConfig.model_validate(data)


def test_every_final_antenna_and_instrument_provenance_field_is_populated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(
        path,
        header="Name Number BeamID E N U Diameter",
        rows="A1 1 beam-1 10 0 0 11\nA0 0 7 0 0 0 10\n",
    )

    instrument = _resolve_local(
        path,
        overrides=(_number_override(0, 12.0),),
    )
    provenance = instrument.provenance

    assert provenance == InstrumentProvenance(
        schema_version="radiosim.instrument.v1",
        source_kind="layout_file",
        source_reference=str(path.resolve()),
        source_format="radiosim",
        registry_policy=None,
        telescope_name_source=AntennaFieldSource.EXPLICIT_CONFIG,
        location_source=AntennaFieldSource.EXPLICIT_CONFIG,
        source_location_itrs_xyz_m=None,
        location_separation_m=None,
        pyuvdata_version=None,
        source_sha256=provenance.source_sha256,
        instrument_sha256=provenance.instrument_sha256,
    )
    assert provenance.source_sha256 is not None
    first = instrument.antennas[0]
    assert first.provenance == AntennaProvenance(
        identity_source=AntennaFieldSource.LAYOUT_FILE,
        position_source=AntennaFieldSource.LAYOUT_FILE,
        diameter_source=AntennaFieldSource.EXPLICIT_OVERRIDE,
        source_diameter_m=10.0,
        mount_source=None,
        beam_id_source=AntennaFieldSource.LAYOUT_FILE,
        source_record="line 3",
    )
    assert first.beam_id == 7


@pytest.mark.parametrize("source_format", ["measurement_set", "uvfits"])
def test_dataset_transport_provenance_and_mount_are_retained(
    tmp_path: Path,
    source_format: str,
) -> None:
    path = tmp_path / (
        "array.ms" if source_format == "measurement_set" else "array.uvfits"
    )
    if source_format == "measurement_set":
        path.mkdir()
    else:
        path.write_bytes(b"uvfits")
    telescope = FakeTelescope(
        diameters=np.array([11.0, 10.0]),
        mounts=np.array(["alt-az", "fixed"]),
    )

    instrument = _resolve_dataset(
        path,
        telescope,
        source_format=source_format,
        default=None,
    )

    assert instrument.provenance.pyuvdata_version == "3.2.1"
    assert instrument.provenance.source_sha256 is (
        None
        if source_format == "measurement_set"
        else instrument.provenance.source_sha256
    )
    if source_format == "uvfits":
        assert instrument.provenance.source_sha256 is not None
    assert instrument.antennas[0].mount_type == "fixed"
    assert instrument.antennas[0].provenance.mount_source is (
        AntennaFieldSource.EMBEDDED_DATASET
    )


def test_known_registry_provenance_and_absent_source_sha_are_retained() -> None:
    instrument = _resolve_known(
        FakeTelescope(diameters=np.array([11.0, 10.0])),
        default=None,
    )

    assert instrument.provenance.source_format is None
    assert instrument.provenance.registry_policy == "allow_network"
    assert instrument.provenance.pyuvdata_version == "3.2.1"
    assert instrument.provenance.source_sha256 is None


def test_snapshot_and_fingerprint_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)
    config = _local_config(path, default=14.0)

    first = resolve_instrument(config)
    second = resolve_instrument(config)

    assert first == second
    assert first is not second
    assert first.antennas is not second.antennas
    assert first.to_snapshot() == second.to_snapshot()
    assert first.provenance.instrument_sha256 == second.provenance.instrument_sha256


def test_source_and_override_input_order_do_not_change_final_science(
    tmp_path: Path,
) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path, rows="A1 1 1 0 0\nA0 0 0 0 0\n")
    first_config = _local_config(
        path,
        overrides=(_number_override(0, 12.0), _name_override("A1", 13.0)),
    )
    second_config = _local_config(
        path,
        overrides=(_name_override("A1", 13.0), _number_override(0, 12.0)),
    )
    staged = resolve_instrument_source(first_config)
    first = _finalize_staged_instrument(first_config, staged)
    second = _finalize_staged_instrument(
        second_config,
        replace(staged, antennas=tuple(reversed(staged.antennas))),
    )

    assert first == second
    assert first.to_snapshot() == second.to_snapshot()
    assert first.provenance.instrument_sha256 == second.provenance.instrument_sha256


def test_transport_path_difference_does_not_change_instrument_fingerprint(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "one" / "layout.txt"
    second_path = tmp_path / "two" / "layout.txt"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    _write_radiosim(first_path)
    _write_radiosim(second_path)

    first = _resolve_local(first_path, default=14.0)
    second = _resolve_local(second_path, default=14.0)

    assert first.provenance.source_reference != second.provenance.source_reference
    assert first.provenance.instrument_sha256 == second.provenance.instrument_sha256


def test_diameter_and_winning_source_label_change_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)
    default = _resolve_local(path, default=14.0)
    changed_value = _resolve_local(path, default=15.0)
    same_value_override = _resolve_local(
        path,
        default=14.0,
        overrides=(_number_override(0, 14.0), _number_override(1, 14.0)),
    )

    assert (
        default.provenance.instrument_sha256
        != changed_value.provenance.instrument_sha256
    )
    assert (
        default.provenance.instrument_sha256
        != same_value_override.provenance.instrument_sha256
    )


def test_telescope_name_winner_label_changes_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"dataset")
    telescope = FakeTelescope(name="Same")

    embedded = _resolve_dataset(path, telescope, explicit_name=None)
    explicit = _resolve_dataset(path, telescope, explicit_name="Same")

    assert embedded.name == explicit.name
    assert embedded.antennas == explicit.antennas
    assert (
        embedded.provenance.instrument_sha256 != explicit.provenance.instrument_sha256
    )


def test_resolved_instrument_constructor_validates_factory_hash(tmp_path: Path) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)
    instrument = _resolve_local(path, default=14.0)

    with pytest.raises(ValueError, match="instrument_sha256"):
        replace(
            instrument,
            provenance=replace(instrument.provenance, instrument_sha256="0" * 64),
        )


def test_config_override_tuple_and_staging_are_not_mutated(tmp_path: Path) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)
    overrides = (_number_override(0, 12.0),)
    config = _local_config(path, default=14.0, overrides=overrides)
    staged = resolve_instrument_source(config)
    config_before = deepcopy(config.model_dump(mode="json"))
    staged_before = staged

    instrument = _finalize_staged_instrument(config, staged)

    assert config.model_dump(mode="json") == config_before
    assert config.diameter_overrides == overrides
    assert staged == staged_before
    assert instrument.location is staged.location


def test_dependency_arrays_can_change_after_resolution_without_affecting_result(
    tmp_path: Path,
) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"dataset")
    names = np.array(["A1", "A0"])
    numbers = np.array([1, 0], dtype=np.int64)
    positions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    diameters = np.array([11.0, 10.0])
    telescope = FakeTelescope(
        names=names,
        numbers=numbers,
        positions=positions,
        diameters=diameters,
    )

    instrument = _resolve_dataset(path, telescope, default=None)
    before = instrument.to_snapshot()
    names[:] = "changed"
    numbers[:] = 99
    positions[:] = 999.0
    diameters[:] = 999.0

    assert instrument.to_snapshot() == before


def test_public_final_state_is_frozen_hashable_and_canonical(tmp_path: Path) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)
    instrument = _resolve_local(path, default=14.0)

    assert isinstance(hash(instrument), int)
    _assert_public_tree_is_canonical(instrument)
    with pytest.raises(FrozenInstanceError):
        instrument.name = "changed"  # type: ignore[misc]


def test_error_paths_leave_staging_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)
    config, staged = _staged_local(path)
    before = staged

    with pytest.raises(DiameterResolutionError):
        _finalize_staged_instrument(config, staged)

    assert staged == before


def test_error_hierarchy_and_internal_export_surface_are_exact() -> None:
    assert issubclass(UnknownDiameterOverrideError, DiameterResolutionError)
    assert "UnknownDiameterOverrideError" in resolution_module.__all__
    assert "resolve_instrument" in resolution_module.__all__
    assert "_finalize_staged_instrument" not in resolution_module.__all__
    for name in ("UnknownDiameterOverrideError", "resolve_instrument"):
        assert name not in core.__all__
        assert name not in radiosim.__all__


def test_resolution_module_does_not_duplicate_the_fingerprint_algorithm() -> None:
    source = Path(resolution_module.__file__).read_text(encoding="utf-8")

    assert "import hashlib" not in source
    assert "import json" not in source
    assert "_compute_instrument_sha256" not in source


def test_final_resolver_accepts_only_typed_instrument_config(tmp_path: Path) -> None:
    path = tmp_path / "layout.txt"
    _write_radiosim(path)

    with pytest.raises(TypeError, match="InstrumentConfig"):
        resolve_instrument(  # type: ignore[arg-type]
            {
                "source": {
                    "kind": "layout_file",
                    "path": path,
                    "format": "radiosim",
                    "telescope_name": "Array",
                }
            }
        )
