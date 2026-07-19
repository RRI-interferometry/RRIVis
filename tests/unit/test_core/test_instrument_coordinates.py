"""Contract tests for Tier 2D instrument coordinate normalization."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from pyuvdata.utils import ECEF_from_ENU, ENU_from_ECEF

from radiosim.core.instrument import AntennaFieldSource
from radiosim.core.instrument_resolution import (
    CoordinateFrameError,
    InstrumentFormatError,
    InstrumentLocationMismatchError,
    InvalidAntennaPositionError,
    resolve_instrument_source,
)
from radiosim.io.instrument_config import (
    InstrumentConfig,
    InstrumentLocationConfig,
    KnownTelescopeSourceConfig,
    LayoutFileSourceConfig,
)


class FakeTelescope:
    def __init__(
        self,
        *,
        location: object,
        positions: np.ndarray,
        name: object = "Embedded",
    ) -> None:
        self.name = name
        self.location = location
        self.antenna_names = np.array(["A0", "A1"])
        self.antenna_numbers = np.array([0, 1], dtype=np.int64)
        self.antenna_positions = positions
        self.antenna_diameters = None
        self.mount_type = None
        self.feeds = np.array(["x", "y"])


def _location_config(location: EarthLocation) -> InstrumentLocationConfig:
    geodetic = location.to_geodetic()
    return InstrumentLocationConfig(
        longitude_deg=float(geodetic.lon.to_value(u.deg)),
        latitude_deg=float(geodetic.lat.to_value(u.deg)),
        height_m=float(geodetic.height.to_value(u.m)),
    )


def _local_config(path: Path, source_format: str) -> InstrumentConfig:
    return InstrumentConfig(
        source=LayoutFileSourceConfig(
            path=path,
            format=source_format,
            telescope_name="Local Array",
        ),
        location=InstrumentLocationConfig(
            longitude_deg=116.67,
            latitude_deg=-26.70,
            height_m=377.0,
        ),
    )


def _dataset_config(
    path: Path,
    source_format: str,
    *,
    explicit_location: EarthLocation | None = None,
) -> InstrumentConfig:
    return InstrumentConfig(
        source=LayoutFileSourceConfig(path=path, format=source_format),
        location=(
            _location_config(explicit_location)
            if explicit_location is not None
            else None
        ),
    )


@pytest.mark.parametrize("source_format", ["radiosim", "casa_loc"])
def test_local_text_positions_are_preserved_exactly(
    tmp_path: Path, source_format: str
) -> None:
    path = tmp_path / "layout.txt"
    if source_format == "radiosim":
        path.write_text("Name Number E N U\nA0 0 -0 2.5 3\n", encoding="utf-8")
    else:
        path.write_text("#coordsys=LOC\n-0 2.5 3\n", encoding="utf-8")

    staged = resolve_instrument_source(_local_config(path, source_format))

    assert staged.antennas[0].position_enu_m == (0.0, 2.5, 3.0)
    assert all(type(value) is float for value in staged.antennas[0].position_enu_m)
    assert type(staged.antennas) is tuple
    assert staged.antennas[0].position_source is AntennaFieldSource.LAYOUT_FILE


def test_mwa_local_enu_and_altitude_are_preserved(tmp_path: Path) -> None:
    from astropy.io import fits

    path = tmp_path / "array.metafits"
    columns = [
        fits.Column(name="TileName", format="8A", array=np.array(["Tile0"])),
        fits.Column(name="Antenna", format="J", array=np.array([0])),
        fits.Column(name="East", format="D", array=np.array([1.0])),
        fits.Column(name="North", format="D", array=np.array([2.0])),
        fits.Column(name="Height", format="D", array=np.array([8.25])),
    ]
    fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="TILEDATA")]
    ).writeto(path)

    staged = resolve_instrument_source(_local_config(path, "mwa_metafits"))

    assert staged.antennas[0].position_enu_m == (1.0, 2.0, 8.25)


def test_staging_is_frozen_and_diameter_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "layout.txt"
    path.write_text("Name Number E N U\nA0 0 1 2 3\n", encoding="utf-8")
    staged = resolve_instrument_source(_local_config(path, "radiosim"))

    assert staged.antennas[0].source_diameter_m is None
    assert not hasattr(staged, "instrument_sha256")
    with pytest.raises(FrozenInstanceError):
        staged.antennas[0].name = "changed"  # type: ignore[misc]


def test_explicit_location_is_constructed_and_longitude_is_canonical(
    tmp_path: Path,
) -> None:
    path = tmp_path / "layout.txt"
    path.write_text("Name Number E N U\nA0 0 0 0 0\n", encoding="utf-8")
    config = InstrumentConfig(
        source=LayoutFileSourceConfig(
            path=path,
            format="radiosim",
            telescope_name="Local",
        ),
        location=InstrumentLocationConfig(
            longitude_deg=190.0,
            latitude_deg=-26.7,
            height_m=377.0,
        ),
    )
    staged = resolve_instrument_source(config)

    assert staged.location.longitude_deg == pytest.approx(-170.0)
    assert staged.location.latitude_deg == pytest.approx(-26.7)
    assert staged.location.height_m == pytest.approx(377.0)
    assert staged.location.source is AntennaFieldSource.EXPLICIT_CONFIG
    assert staged.provenance.embedded_location_itrs_xyz_m is None
    assert staged.provenance.explicit_location_itrs_xyz_m == staged.location.itrs_xyz_m


@pytest.mark.parametrize(
    "positions",
    [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]]),
        np.array([[1.0, 2.0, np.inf], [3.0, 4.0, 5.0]]),
        np.array([[1e308, 1e308, 1e308], [3.0, 4.0, 5.0]]),
    ],
)
def test_wrong_or_nonfinite_relative_positions_are_rejected(
    tmp_path: Path, positions: np.ndarray
) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"placeholder")
    telescope = FakeTelescope(
        location=EarthLocation.from_geodetic(116.67, -26.70, 377.0),
        positions=positions,
    )
    with pytest.raises(
        (
            InstrumentFormatError,
            InvalidAntennaPositionError,
            CoordinateFrameError,
        )
    ):
        resolve_instrument_source(
            _dataset_config(path, "uvfits"),
            dataset_loader=lambda path, source_format: telescope,
            pyuvdata_version="3.2.1",
        )


@pytest.mark.parametrize("source_format", ["measurement_set", "uvfits"])
def test_relative_ecef_sources_match_public_conversion_and_round_trip(
    tmp_path: Path, source_format: str
) -> None:
    path = tmp_path / (
        "array.ms" if source_format == "measurement_set" else "array.uvfits"
    )
    if source_format == "measurement_set":
        path.mkdir()
    else:
        path.write_bytes(b"placeholder")
    embedded = EarthLocation.from_geodetic(116.67, -26.70, 377.0)
    relative_ecef = np.array([[0.0, 0.0, 0.0], [11.0, -7.0, 4.0]])
    telescope = FakeTelescope(location=embedded, positions=relative_ecef.copy())

    staged = resolve_instrument_source(
        _dataset_config(path, source_format),
        dataset_loader=lambda path, source_format: telescope,
        pyuvdata_version="3.2.1",
    )

    center_xyz = np.array(
        [embedded.x.to_value(u.m), embedded.y.to_value(u.m), embedded.z.to_value(u.m)]
    )
    absolute = center_xyz + relative_ecef
    expected = ENU_from_ECEF(absolute, center_loc=embedded)
    actual = np.asarray([item.position_enu_m for item in staged.antennas])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
    round_trip = ECEF_from_ENU(actual, center_loc=embedded)
    np.testing.assert_allclose(round_trip, absolute, rtol=0.0, atol=1e-6)
    assert staged.provenance.embedded_location_itrs_xyz_m == tuple(center_xyz)


def test_known_telescope_relative_ecef_conversion() -> None:
    embedded = EarthLocation.from_geodetic(21.4439, -30.7130, 1086.6)
    relative = np.array([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
    telescope = FakeTelescope(location=embedded, positions=relative)

    staged = resolve_instrument_source(
        InstrumentConfig(source=KnownTelescopeSourceConfig(name="Known")),
        known_telescope_loader=lambda name: telescope,
        pyuvdata_version="3.2.1",
    )

    assert staged.provenance.source_reference == "Known"
    assert staged.antennas[0].position_source is AntennaFieldSource.KNOWN_TELESCOPE
    assert staged.location.source is AntennaFieldSource.KNOWN_TELESCOPE
    center_xyz = np.array(
        [embedded.x.to_value(u.m), embedded.y.to_value(u.m), embedded.z.to_value(u.m)]
    )
    expected = ENU_from_ECEF(center_xyz + relative, center_loc=embedded)
    np.testing.assert_allclose(
        staged.antennas[1].position_enu_m, expected[1], rtol=0.0, atol=1e-6
    )


@pytest.mark.parametrize("offset_m", [0.0, 0.5, 1.0])
def test_explicit_embedded_location_within_threshold_uses_explicit(
    tmp_path: Path, offset_m: float
) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"placeholder")
    embedded = EarthLocation.from_geodetic(116.67, -26.70, 377.0)
    explicit = EarthLocation.from_geocentric(
        embedded.x.to_value(u.m) + offset_m,
        embedded.y.to_value(u.m),
        embedded.z.to_value(u.m),
        unit=u.m,
    )
    telescope = FakeTelescope(location=embedded, positions=np.zeros((2, 3)))

    staged = resolve_instrument_source(
        _dataset_config(path, "uvfits", explicit_location=explicit),
        dataset_loader=lambda path, source_format: telescope,
        pyuvdata_version="3.2.1",
    )

    assert staged.location.source is AntennaFieldSource.EXPLICIT_CONFIG
    assert staged.provenance.location_separation_m == pytest.approx(offset_m, abs=1e-6)
    assert staged.provenance.explicit_location_itrs_xyz_m == staged.location.itrs_xyz_m
    expected_origin = ENU_from_ECEF(
        np.array([[embedded.x.value, embedded.y.value, embedded.z.value]]),
        center_loc=explicit,
    )[0]
    np.testing.assert_allclose(
        staged.antennas[0].position_enu_m, expected_origin, rtol=0.0, atol=1e-6
    )


def test_explicit_embedded_location_over_threshold_fails(tmp_path: Path) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"placeholder")
    embedded = EarthLocation.from_geodetic(116.67, -26.70, 377.0)
    explicit = EarthLocation.from_geocentric(
        embedded.x.to_value(u.m) + 1.0001,
        embedded.y.to_value(u.m),
        embedded.z.to_value(u.m),
        unit=u.m,
    )
    telescope = FakeTelescope(location=embedded, positions=np.zeros((2, 3)))
    with pytest.raises(InstrumentLocationMismatchError, match="1.0 m"):
        resolve_instrument_source(
            _dataset_config(path, "uvfits", explicit_location=explicit),
            dataset_loader=lambda path, source_format: telescope,
            pyuvdata_version="3.2.1",
        )


def test_non_earth_location_is_rejected() -> None:
    telescope = FakeTelescope(location=object(), positions=np.zeros((2, 3)))
    with pytest.raises(CoordinateFrameError, match="EarthLocation"):
        resolve_instrument_source(
            InstrumentConfig(source=KnownTelescopeSourceConfig(name="Known")),
            known_telescope_loader=lambda name: telescope,
            pyuvdata_version="3.2.1",
        )


def test_conversion_failure_is_mapped_and_chained(tmp_path: Path) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"placeholder")
    telescope = FakeTelescope(
        location=EarthLocation.from_geodetic(116.67, -26.70, 377.0),
        positions=np.zeros((2, 3)),
    )
    cause = RuntimeError("converter internals")

    def failing_converter(absolute_ecef_m: np.ndarray, *, center_loc: object) -> object:
        raise cause

    with pytest.raises(CoordinateFrameError) as raised:
        resolve_instrument_source(
            _dataset_config(path, "uvfits"),
            dataset_loader=lambda path, source_format: telescope,
            pyuvdata_version="3.2.1",
            enu_from_ecef=failing_converter,
        )
    assert raised.value.__cause__ is cause
    assert "converter internals" not in str(raised.value)


def test_dependency_arrays_are_copied_before_staging(tmp_path: Path) -> None:
    path = tmp_path / "array.uvfits"
    path.write_bytes(b"placeholder")
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    names = np.array(["A0", "A1"])
    telescope = FakeTelescope(
        location=EarthLocation.from_geodetic(116.67, -26.70, 377.0),
        positions=positions,
    )
    telescope.antenna_names = names
    staged = resolve_instrument_source(
        _dataset_config(path, "uvfits"),
        dataset_loader=lambda path, source_format: telescope,
        pyuvdata_version="3.2.1",
    )
    before = staged.antennas

    positions[:] = 999.0
    names[:] = "changed"

    assert staged.antennas == before
    assert staged.antennas[0].name == "A0"
    assert all(type(value) is float for value in staged.location.itrs_xyz_m)
