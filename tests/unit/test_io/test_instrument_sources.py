"""Contract tests for strict Tier 2D instrument source adapters."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.io import fits

from radiosim.core.instrument_resolution import (
    AntennaIdentifierError,
    DiameterResolutionError,
    DuplicateAntennaError,
    EmptyInstrumentError,
    InstrumentFormatError,
    InstrumentSourceError,
    InvalidAntennaPositionError,
    OptionalInstrumentDependencyError,
    TelescopeNotFoundError,
)
from radiosim.io.instrument_config import (
    KnownTelescopeSourceConfig,
    LayoutFileSourceConfig,
)
from radiosim.io.instrument_sources import load_instrument_source


@dataclass
class FakeTelescope:
    """Small mutable dependency object used to prove source-boundary copying."""

    name: object = "Embedded Array"
    location: object = EarthLocation.from_geodetic(116.67, -26.70, 377.0)
    antenna_names: object = None
    antenna_numbers: object = None
    antenna_positions: object = None
    antenna_diameters: object = None
    mount_type: object = None
    feeds: object = None
    feed_array: object = None
    feed_angle: object = None

    def __post_init__(self) -> None:
        if self.antenna_names is None:
            self.antenna_names = np.array(["A1", "A0"])
        if self.antenna_numbers is None:
            self.antenna_numbers = np.array([1, 0], dtype=np.int64)
        if self.antenna_positions is None:
            self.antenna_positions = np.array(
                [[10.0, 20.0, 30.0], [0.0, 0.0, 0.0]], dtype=np.float64
            )


class RecordingDatasetLoader:
    def __init__(self, telescope: object) -> None:
        self.telescope = telescope
        self.calls: list[tuple[Path, str]] = []

    def __call__(self, path: Path, source_format: str) -> object:
        self.calls.append((path, source_format))
        return self.telescope


class RecordingKnownLoader:
    def __init__(self, telescope: object | None = None) -> None:
        self.telescope = telescope or FakeTelescope()
        self.calls: list[str] = []

    def __call__(self, name: str) -> object:
        self.calls.append(name)
        return self.telescope


class FakeInternetConfig:
    def __init__(self, value: object) -> None:
        self.allow_internet = value
        self.observed: list[object] = []

    @contextmanager
    def set_temp(self, attribute: str, value: object) -> Iterator[None]:
        assert attribute == "allow_internet"
        previous = self.allow_internet
        self.allow_internet = value
        self.observed.append(value)
        try:
            yield
        finally:
            self.allow_internet = previous


def _layout(path: Path, source_format: str) -> LayoutFileSourceConfig:
    return LayoutFileSourceConfig(
        path=path,
        format=source_format,
        telescope_name=(
            "Local Array"
            if source_format in {"radiosim", "casa_loc", "mwa_metafits"}
            else None
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_metafits(
    path: Path,
    *,
    names: list[object] | None = None,
    numbers: list[object] | None = None,
    east: list[float] | None = None,
    north: list[float] | None = None,
    height: list[float] | None = None,
) -> None:
    names = names if names is not None else ["Tile001", "Tile001", "Tile000"]
    numbers = numbers if numbers is not None else [1, 1, 0]
    east = east if east is not None else [10.0, 10.0, 0.0]
    north = north if north is not None else [20.0, 20.0, 0.0]
    height = height if height is not None else [3.0, 3.0, 1.0]
    columns = [
        fits.Column(name="TileName", format="16A", array=np.asarray(names)),
        fits.Column(name="Antenna", format="J", array=np.asarray(numbers)),
        fits.Column(name="East", format="D", array=np.asarray(east)),
        fits.Column(name="North", format="D", array=np.asarray(north)),
        fits.Column(name="Height", format="D", array=np.asarray(height)),
    ]
    fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="TILEDATA")]
    ).writeto(path)


@pytest.mark.parametrize(
    ("header", "row", "beam_id", "diameter"),
    [
        ("Name Number E N U", "A0 0 1 2 3", None, None),
        ("Name Number BeamID E N U", "A0 0 beam-a 1 2 3", "beam-a", None),
        ("Name Number E N U Diameter", "A0 0 1 2 3 12.5", None, 12.5),
        ("Name Number BeamID E N U Diameter", "A0 0 7 1 2 3 14", 7, 14.0),
    ],
)
def test_radiosim_valid_header_variants(
    tmp_path: Path,
    header: str,
    row: str,
    beam_id: int | str | None,
    diameter: float | None,
) -> None:
    path = tmp_path / "array.txt"
    path.write_text(f"# preface\n{header}\n\n{row}\n", encoding="utf-8")

    loaded = load_instrument_source(_layout(path, "radiosim"))

    assert loaded.antennas[0].beam_id == beam_id
    assert loaded.antennas[0].source_diameter_m == diameter
    assert loaded.antennas[0].source_record == "line 4"
    assert loaded.source_sha256 == _sha256(path)


def test_radiosim_normalizes_sorts_and_owns_values(tmp_path: Path) -> None:
    path = tmp_path / "array.txt"
    path.write_text(
        "Name Number E N U\nB 2 -0 2 3\n  # ignored\nA 0 4 5 6\n",
        encoding="utf-8",
    )
    source = _layout(path, "radiosim")

    loaded = load_instrument_source(source)

    assert tuple(item.number for item in loaded.antennas) == (0, 2)
    assert loaded.antennas[1].position_m == (0.0, 2.0, 3.0)
    assert loaded.source_reference == str(path.resolve())
    assert source.path == path


@pytest.mark.parametrize(
    "text",
    [
        "name Number E N U\nA 0 1 2 3\n",
        "Name Number E N U Extra\nA 0 1 2 3 x\n",
        "Name Number E N U\nA nope 1 2 3\n",
        "Name Number E N U Diameter\nA 0 1 2 3 nope\n",
        "Name Number BeamID E N U\nA 0 '' 1 2 3\n",
        "Name Number E N U\nA 0 1 2\n",
        "Name Number E N U\nA 0 1 2 3 extra\n",
    ],
)
def test_radiosim_rejects_incoherent_content(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises((InstrumentFormatError, AntennaIdentifierError)):
        load_instrument_source(_layout(path, "radiosim"))


@pytest.mark.parametrize(
    "rows",
    [
        "A 0 1 2 3\nB 0 4 5 6\n",
        " A\u030a 0 1 2 3\n\u00c5 1 4 5 6\n",
    ],
)
def test_radiosim_rejects_duplicate_normalized_identity(
    tmp_path: Path, rows: str
) -> None:
    path = tmp_path / "duplicate.txt"
    path.write_text("Name Number E N U\n" + rows, encoding="utf-8")
    with pytest.raises(DuplicateAntennaError):
        load_instrument_source(_layout(path, "radiosim"))


@pytest.mark.parametrize("text", ["", "# comment only\n", "Name Number E N U\n"])
def test_radiosim_rejects_empty_inventory(tmp_path: Path, text: str) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(EmptyInstrumentError):
        load_instrument_source(_layout(path, "radiosim"))


@pytest.mark.parametrize("frame", ["LOC", "ENU"])
def test_casa_loc_accepts_only_local_frames(tmp_path: Path, frame: str) -> None:
    path = tmp_path / "array.cfg"
    path.write_text(
        f"#coordsys={frame}\n1 2 3\n4 5 6 12 STATION\n7 8 9 14 STATION2 ANTENNA2\n",
        encoding="utf-8",
    )

    loaded = load_instrument_source(_layout(path, "casa_loc"))

    assert [item.number for item in loaded.antennas] == [0, 1, 2]
    assert [item.name for item in loaded.antennas] == ["ANT000", "STATION", "ANTENNA2"]
    assert loaded.antennas[0].name_generated is True
    assert loaded.antennas[0].number_generated is True
    assert loaded.antennas[1].source_diameter_m == 12.0
    assert loaded.antennas[2].source_record == "data row 2 (line 4)"


@pytest.mark.parametrize(
    "header", ["", "#coordsys=XYZ\n", "#coordsys=ITRF\n", "#coordsys=WGS84\n"]
)
def test_casa_loc_rejects_missing_or_nonlocal_frame(
    tmp_path: Path, header: str
) -> None:
    path = tmp_path / "bad.cfg"
    path.write_text(header + "1 2 3\n", encoding="utf-8")
    with pytest.raises(InstrumentFormatError, match="coordinate"):
        load_instrument_source(_layout(path, "casa_loc"))


@pytest.mark.parametrize(
    "row",
    ["1 2", "1 2 nope", "1 2 3 nope", "1 2 3 12 station ant extra"],
)
def test_casa_loc_rejects_every_malformed_row(tmp_path: Path, row: str) -> None:
    path = tmp_path / "bad.cfg"
    path.write_text(f"#coordsys=LOC\n{row}\n", encoding="utf-8")
    with pytest.raises((InstrumentFormatError, DiameterResolutionError)):
        load_instrument_source(_layout(path, "casa_loc"))


def test_casa_loc_rejects_duplicate_generated_or_explicit_name(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.cfg"
    path.write_text("#coordsys=ENU\n1 2 3 12 S A\n4 5 6 12 S A\n", encoding="utf-8")
    with pytest.raises(DuplicateAntennaError):
        load_instrument_source(_layout(path, "casa_loc"))


def test_casa_loc_rejects_empty_inventory(tmp_path: Path) -> None:
    path = tmp_path / "empty.cfg"
    path.write_text("#coordsys=LOC\n# no rows\n", encoding="utf-8")
    with pytest.raises(EmptyInstrumentError):
        load_instrument_source(_layout(path, "casa_loc"))


def test_mwa_metafits_loads_deduplicates_and_drops_metadata(tmp_path: Path) -> None:
    path = tmp_path / "obs.metafits"
    _write_metafits(path)

    loaded = load_instrument_source(_layout(path, "mwa_metafits"))

    assert [(item.number, item.name) for item in loaded.antennas] == [
        (0, "Tile000"),
        (1, "Tile001"),
    ]
    assert loaded.antennas[1].position_m == (10.0, 20.0, 3.0)
    assert loaded.antennas[1].source_record == "TILEDATA rows 0,1"
    assert loaded.antennas[1].source_diameter_m is None
    assert loaded.antennas[1].mount_type is None
    assert loaded.antennas[1].beam_id is None
    assert loaded.source_sha256 == _sha256(path)


def test_mwa_metafits_rejects_conflicting_polarization_rows(tmp_path: Path) -> None:
    path = tmp_path / "conflict.metafits"
    _write_metafits(path, east=[10.0, 11.0, 0.0])
    with pytest.raises(AntennaIdentifierError, match="conflict"):
        load_instrument_source(_layout(path, "mwa_metafits"))


def test_mwa_metafits_rejects_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "missing.metafits"
    columns = [
        fits.Column(name="TileName", format="16A", array=np.array(["A"])),
        fits.Column(name="Antenna", format="J", array=np.array([0])),
        fits.Column(name="East", format="D", array=np.array([0.0])),
        fits.Column(name="North", format="D", array=np.array([0.0])),
    ]
    fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="TILEDATA")]
    ).writeto(path)
    with pytest.raises(InstrumentFormatError, match="Height"):
        load_instrument_source(_layout(path, "mwa_metafits"))


@pytest.mark.parametrize(
    ("names", "numbers", "east"),
    [([""], [0], [0.0]), (["A"], [-1], [0.0]), (["A"], [0], [float("nan")])],
)
def test_mwa_metafits_rejects_invalid_row(
    tmp_path: Path,
    names: list[object],
    numbers: list[object],
    east: list[float],
) -> None:
    path = tmp_path / "bad.metafits"
    _write_metafits(
        path,
        names=names,
        numbers=numbers,
        east=east,
        north=[0.0],
        height=[0.0],
    )
    with pytest.raises((AntennaIdentifierError, InvalidAntennaPositionError)):
        load_instrument_source(_layout(path, "mwa_metafits"))


def test_mwa_metafits_rejects_empty_or_missing_tiledata(tmp_path: Path) -> None:
    empty = tmp_path / "empty.metafits"
    _write_metafits(empty, names=[], numbers=[], east=[], north=[], height=[])
    with pytest.raises(EmptyInstrumentError):
        load_instrument_source(_layout(empty, "mwa_metafits"))

    missing = tmp_path / "missing.metafits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(missing)
    with pytest.raises(InstrumentFormatError, match="TILEDATA"):
        load_instrument_source(_layout(missing, "mwa_metafits"))


@pytest.mark.parametrize("source_format", ["measurement_set", "uvfits"])
def test_dataset_dispatch_is_metadata_only_and_never_uses_known_loader(
    tmp_path: Path, source_format: str
) -> None:
    path = tmp_path / (
        "array.ms" if source_format == "measurement_set" else "array.uvfits"
    )
    if source_format == "measurement_set":
        path.mkdir()
    else:
        path.write_bytes(b"uvfits-placeholder")
    dataset_loader = RecordingDatasetLoader(FakeTelescope())

    def forbidden_known_loader(name: str) -> object:
        raise AssertionError(f"known loader called for {name}")

    loaded = load_instrument_source(
        _layout(path, source_format),
        dataset_loader=dataset_loader,
        known_telescope_loader=forbidden_known_loader,
        pyuvdata_version="3.2.1",
    )

    assert dataset_loader.calls == [(path.resolve(), source_format)]
    assert loaded.position_frame == "relative_ecef"
    assert loaded.pyuvdata_version == "3.2.1"
    assert loaded.source_sha256 == (
        None if source_format == "measurement_set" else _sha256(path)
    )


def test_production_dataset_adapter_requests_read_data_false(tmp_path: Path) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"placeholder")
    calls: list[tuple[Path, bool]] = []
    telescope = FakeTelescope()

    class FakeUVData:
        def __init__(self) -> None:
            self.telescope: object | None = None

        def read(self, read_path: Path, *, read_data: bool) -> None:
            calls.append((Path(read_path), read_data))
            self.telescope = telescope

    module = SimpleNamespace(UVData=FakeUVData, __version__="3.2.1")

    loaded = load_instrument_source(
        _layout(path, "uvfits"),
        module_importer=lambda name: module
        if name == "pyuvdata"
        else import_module(name),
    )

    assert calls == [(path.resolve(), False)]
    assert loaded.embedded_telescope_name == "Embedded Array"


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("antenna_names", np.array(["A"])),
        ("antenna_numbers", np.array([0])),
        ("antenna_positions", np.zeros((1, 3))),
        ("antenna_positions", np.zeros((2, 2))),
    ],
)
def test_dataset_rejects_parallel_length_or_shape_mismatch(
    tmp_path: Path, attribute: str, value: object
) -> None:
    path = tmp_path / "array.ms"
    path.mkdir()
    telescope = FakeTelescope()
    setattr(telescope, attribute, value)
    with pytest.raises(InstrumentFormatError):
        load_instrument_source(
            _layout(path, "measurement_set"),
            dataset_loader=RecordingDatasetLoader(telescope),
            pyuvdata_version="3.2.1",
        )


def test_dataset_rejects_empty_and_duplicate_inventory(tmp_path: Path) -> None:
    path = tmp_path / "array.ms"
    path.mkdir()
    empty = FakeTelescope(
        antenna_names=np.array([]),
        antenna_numbers=np.array([], dtype=np.int64),
        antenna_positions=np.empty((0, 3)),
    )
    with pytest.raises(EmptyInstrumentError):
        load_instrument_source(
            _layout(path, "measurement_set"),
            dataset_loader=RecordingDatasetLoader(empty),
            pyuvdata_version="3.2.1",
        )

    duplicate = FakeTelescope(antenna_names=np.array(["A", "A"]))
    with pytest.raises(DuplicateAntennaError):
        load_instrument_source(
            _layout(path, "measurement_set"),
            dataset_loader=RecordingDatasetLoader(duplicate),
            pyuvdata_version="3.2.1",
        )


@pytest.mark.parametrize(
    "numbers",
    [
        np.array([True, False]),
        np.array([1.0, 0.0]),
        np.array(["1", "0"]),
        np.array([-1, 0]),
        np.array([2_147_483_648, 0]),
    ],
)
def test_dataset_rejects_noncanonical_antenna_numbers(
    tmp_path: Path, numbers: np.ndarray
) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"placeholder")
    telescope = FakeTelescope(antenna_numbers=numbers)
    with pytest.raises(AntennaIdentifierError):
        load_instrument_source(
            _layout(path, "uvfits"),
            dataset_loader=RecordingDatasetLoader(telescope),
            pyuvdata_version="3.2.1",
        )


@pytest.mark.parametrize(
    "names",
    [np.array(["", "A"]), np.array([" ", "A"]), np.array([1, 2])],
)
def test_dataset_rejects_noncanonical_antenna_names(
    tmp_path: Path, names: np.ndarray
) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"placeholder")
    telescope = FakeTelescope(antenna_names=names)
    with pytest.raises(AntennaIdentifierError):
        load_instrument_source(
            _layout(path, "uvfits"),
            dataset_loader=RecordingDatasetLoader(telescope),
            pyuvdata_version="3.2.1",
        )


def test_dataset_copies_numpy_identifiers_to_builtins(tmp_path: Path) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"placeholder")
    telescope = FakeTelescope(
        antenna_names=np.array([np.str_(" A1 "), np.str_("A0")]),
        antenna_numbers=np.array([np.int64(1), np.int64(0)]),
    )
    loaded = load_instrument_source(
        _layout(path, "uvfits"),
        dataset_loader=RecordingDatasetLoader(telescope),
        pyuvdata_version="3.2.1",
    )
    assert [(item.number, item.name) for item in loaded.antennas] == [
        (0, "A0"),
        (1, "A1"),
    ]
    assert all(type(item.number) is int for item in loaded.antennas)
    assert all(type(item.name) is str for item in loaded.antennas)


@pytest.mark.parametrize(
    "diameters",
    [
        np.array([12.0]),
        np.array([12.0, np.nan]),
        np.array([12.0, np.inf]),
        np.array([12.0, 0.0]),
        np.array([12.0, -1.0]),
    ],
)
def test_dataset_rejects_invalid_dense_diameters(
    tmp_path: Path, diameters: np.ndarray
) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"placeholder")
    telescope = FakeTelescope(antenna_diameters=diameters)
    with pytest.raises(DiameterResolutionError):
        load_instrument_source(
            _layout(path, "uvfits"),
            dataset_loader=RecordingDatasetLoader(telescope),
            pyuvdata_version="3.2.1",
        )


@pytest.mark.parametrize(
    ("mount", "expected"),
    [
        (" fixed ", ("fixed", "fixed")),
        (np.array(["alt-az", "fixed"]), ("fixed", "alt-az")),
        (None, (None, None)),
    ],
)
def test_dataset_normalizes_mounts_and_drops_feeds(
    tmp_path: Path, mount: object, expected: tuple[str | None, ...]
) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"placeholder")
    telescope = FakeTelescope(
        antenna_diameters=np.array([12.0, 14.0]),
        mount_type=mount,
        feeds=np.array(["x", "y"]),
        feed_array=np.array([["x"], ["y"]]),
        feed_angle=np.array([0.0, 1.0]),
    )
    loaded = load_instrument_source(
        _layout(path, "uvfits"),
        dataset_loader=RecordingDatasetLoader(telescope),
        pyuvdata_version="3.2.1",
    )

    assert tuple(item.mount_type for item in loaded.antennas) == expected
    assert tuple(item.source_diameter_m for item in loaded.antennas) == (14.0, 12.0)
    assert not hasattr(loaded.antennas[0], "feeds")


def test_dataset_none_diameters_are_preserved_as_missing(tmp_path: Path) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"placeholder")
    loaded = load_instrument_source(
        _layout(path, "uvfits"),
        dataset_loader=RecordingDatasetLoader(FakeTelescope()),
        pyuvdata_version="3.2.1",
    )
    assert tuple(item.source_diameter_m for item in loaded.antennas) == (None, None)


def test_dataset_dependency_failure_is_stable_and_chained(tmp_path: Path) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"placeholder")
    cause = RuntimeError("backend details")

    def failing_loader(path: Path, source_format: str) -> object:
        raise cause

    with pytest.raises(InstrumentSourceError, match="uvfits") as raised:
        load_instrument_source(
            _layout(path, "uvfits"),
            dataset_loader=failing_loader,
            pyuvdata_version="3.2.1",
        )
    assert raised.value.__cause__ is cause
    assert "backend details" not in str(raised.value)


def test_dataset_dependency_metadata_property_failure_is_chained(
    tmp_path: Path,
) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"placeholder")
    cause = RuntimeError("property internals")

    class BrokenTelescope:
        @property
        def antenna_names(self) -> object:
            raise cause

    with pytest.raises(InstrumentSourceError, match="normalization") as raised:
        load_instrument_source(
            _layout(path, "uvfits"),
            dataset_loader=RecordingDatasetLoader(BrokenTelescope()),
            pyuvdata_version="3.2.1",
        )
    assert raised.value.__cause__ is cause
    assert "property internals" not in str(raised.value)


@pytest.mark.parametrize("wrong_type", ["measurement_set", "uvfits"])
def test_dataset_path_type_is_strict(tmp_path: Path, wrong_type: str) -> None:
    path = tmp_path / "source"
    if wrong_type == "measurement_set":
        path.write_text("not a directory", encoding="utf-8")
    else:
        path.mkdir()
    with pytest.raises(InstrumentSourceError):
        load_instrument_source(_layout(path, wrong_type))


def test_known_telescope_uses_injected_loader_without_enumeration() -> None:
    loader = RecordingKnownLoader()
    source = KnownTelescopeSourceConfig(name=" HERA ", registry_policy="allow_network")
    config = FakeInternetConfig(True)

    loaded = load_instrument_source(
        source,
        known_telescope_loader=loader,
        internet_config=config,
        pyuvdata_version="3.2.1",
    )

    assert loader.calls == ["HERA"]
    assert loaded.source_reference == "HERA"
    assert loaded.registry_policy == "allow_network"
    assert loaded.pyuvdata_version == "3.2.1"
    assert loaded.source_sha256 is None
    assert config.observed == []


def test_known_telescope_maps_only_known_absence_condition() -> None:
    absence = ValueError("name not in astropy_sites or known_telescopes_dict")

    def missing(name: str) -> object:
        raise absence

    with pytest.raises(TelescopeNotFoundError, match="Missing") as raised:
        load_instrument_source(
            KnownTelescopeSourceConfig(name="Missing"),
            known_telescope_loader=missing,
            internet_config=FakeInternetConfig(True),
            pyuvdata_version="3.2.1",
        )
    assert raised.value.__cause__ is absence


@pytest.mark.parametrize(
    "cause", [ValueError("different validation"), RuntimeError("boom")]
)
def test_known_telescope_other_failures_are_source_errors(cause: Exception) -> None:
    def failing(name: str) -> object:
        raise cause

    with pytest.raises(InstrumentSourceError) as raised:
        load_instrument_source(
            KnownTelescopeSourceConfig(name="HERA"),
            known_telescope_loader=failing,
            internet_config=FakeInternetConfig(True),
            pyuvdata_version="3.2.1",
        )
    assert not isinstance(raised.value, TelescopeNotFoundError)
    assert raised.value.__cause__ is cause


def test_known_telescope_missing_pyuvdata_is_actionable() -> None:
    def missing_import(name: str) -> object:
        raise ModuleNotFoundError(name)

    with pytest.raises(OptionalInstrumentDependencyError) as raised:
        load_instrument_source(
            KnownTelescopeSourceConfig(name="HERA", registry_policy="allow_network"),
            module_importer=missing_import,
        )
    assert "known_telescope" in str(raised.value)
    assert "pyuvdata" in str(raised.value)
    assert isinstance(raised.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize("initial", [True, False, "sentinel"])
def test_offline_guard_restores_exact_value_after_success(initial: object) -> None:
    config = FakeInternetConfig(initial)

    def loader(name: str) -> object:
        assert config.allow_internet is False
        return FakeTelescope()

    load_instrument_source(
        KnownTelescopeSourceConfig(name="HERA"),
        known_telescope_loader=loader,
        internet_config=config,
        pyuvdata_version="3.2.1",
    )
    assert config.allow_internet is initial


def test_offline_guard_restores_after_failure_and_is_reentrant() -> None:
    config = FakeInternetConfig(True)
    depth = 0

    def nested(name: str) -> object:
        nonlocal depth
        assert config.allow_internet is False
        if depth == 0:
            depth += 1
            load_instrument_source(
                KnownTelescopeSourceConfig(name="Nested"),
                known_telescope_loader=nested,
                internet_config=config,
                pyuvdata_version="3.2.1",
            )
            raise RuntimeError("outer failure")
        return FakeTelescope()

    with pytest.raises(InstrumentSourceError) as raised:
        load_instrument_source(
            KnownTelescopeSourceConfig(name="Outer"),
            known_telescope_loader=nested,
            internet_config=config,
            pyuvdata_version="3.2.1",
        )
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert config.allow_internet is True


def test_offline_guard_serializes_radio_sim_calls() -> None:
    config = FakeInternetConfig(True)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    failures: list[BaseException] = []

    def first_loader(name: str) -> object:
        first_entered.set()
        if not release_first.wait(timeout=5):
            raise AssertionError("test did not release first loader")
        return FakeTelescope()

    def second_loader(name: str) -> object:
        second_entered.set()
        return FakeTelescope()

    def invoke(name: str, loader: Any) -> None:
        try:
            load_instrument_source(
                KnownTelescopeSourceConfig(name=name),
                known_telescope_loader=loader,
                internet_config=config,
                pyuvdata_version="3.2.1",
            )
        except BaseException as error:  # pragma: no cover - asserted below
            failures.append(error)

    first = threading.Thread(target=invoke, args=("First", first_loader))
    second = threading.Thread(target=invoke, args=("Second", second_loader))
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert second_entered.is_set()
    assert failures == []
    assert config.allow_internet is True


def test_missing_optional_dependency_has_actionable_guidance(tmp_path: Path) -> None:
    path = tmp_path / "array.ms"
    path.mkdir()

    def missing_import(name: str) -> object:
        raise ModuleNotFoundError(name)

    with pytest.raises(OptionalInstrumentDependencyError) as raised:
        load_instrument_source(
            _layout(path, "measurement_set"), module_importer=missing_import
        )
    message = str(raised.value)
    assert "measurement_set" in message
    assert "pyuvdata" in message
    assert "radiosim[ms]" in message
    assert isinstance(raised.value.__cause__, ModuleNotFoundError)
