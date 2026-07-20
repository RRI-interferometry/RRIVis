"""Legacy reader characterization and canonical antenna consumer contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

import radiosim.core.antenna as antenna_module
from radiosim.api import Simulator
from radiosim.core.antenna import (
    format_antenna_data,
    read_antenna_positions,
    read_casa_format,
    read_measurement_set,
    read_mwa_format,
    read_pyuvdata_format,
    read_radiosim_format,
    read_uvfits,
)
from tests.fixtures.configs import resolved_config


def test_radiosim_parser_characterizes_optional_columns_and_mutable_records(tmp_path):
    layout = tmp_path / "native.txt"
    layout.write_text(
        "# current native layout\n"
        "Name Number BeamID E N U Diameter\n"
        "INTEGER 4 12 1 2 3 14\n"
        "STRING 5 beam-x 4 5 6 15.5\n"
        "MISSING 6 none 7 8 9\n"
        "MALFORMED 7 7 10 11 12 not-a-number\n"
    )

    antennas = read_radiosim_format(layout)

    assert list(antennas) == [4, 5, 6, 7]
    assert antennas[4] == {
        "Name": "INTEGER",
        "Number": 4,
        "BeamID": 12,
        "Position": (1.0, 2.0, 3.0),
        "diameter": 14.0,
    }
    assert antennas[5]["BeamID"] == "beam-x"
    assert antennas[5]["diameter"] == 15.5
    assert "diameter" not in antennas[6]
    assert "diameter" not in antennas[7]

    # The nested dictionary is the current mutable public shape.
    antennas[4]["Position"] = (99.0, 2.0, 3.0)
    assert antennas[4]["Position"][0] == 99.0


def test_radiosim_parser_characterizes_layout_without_beamid(tmp_path):
    layout = tmp_path / "native-no-beam.txt"
    layout.write_text("Name Number E N U Diameter\nANT0 0 1 2 3 11\nANT1 1 4 5 6\n")

    antennas = read_radiosim_format(layout)

    assert antennas[0]["BeamID"] is None
    assert antennas[0]["Position"] == (1.0, 2.0, 3.0)
    assert antennas[0]["diameter"] == 11.0
    assert "diameter" not in antennas[1]


def test_radiosim_duplicate_number_silently_overwrites_legacy_record(tmp_path):
    """Undesirable current behavior: the later duplicate silently wins."""
    layout = tmp_path / "duplicate-native.txt"
    layout.write_text(
        "Name Number E N U Diameter\n"
        "FIRST_TWO 2 1 0 0 10\n"
        "ONLY_ONE 1 2 0 0 11\n"
        "LAST_TWO 2 3 0 0 12\n"
    )

    antennas = read_radiosim_format(layout)

    # Replacing a dict value does not move the original insertion slot.
    assert list(antennas) == [2, 1]
    assert antennas[2]["Name"] == "LAST_TWO"
    assert antennas[2]["Position"] == (3.0, 0.0, 0.0)


@pytest.mark.parametrize("coordsys", ["LOC", "ENU", "XYZ"])
def test_casa_parser_currently_passes_recognized_coordinates_through(
    tmp_path, coordsys
):
    layout = tmp_path / f"{coordsys.lower()}.cfg"
    layout.write_text(f"#coordsys={coordsys}\n1 2 3 12 STATION ANTENNA\n")

    antennas = read_casa_format(layout)

    # XYZ is intentionally characterized here even though treating it as ENU is
    # scientifically wrong and scheduled for removal.
    assert antennas[0]["Position"] == (1.0, 2.0, 3.0)


def test_casa_parser_characterizes_generated_identity_and_ignored_invalid_rows(
    tmp_path,
):
    layout = tmp_path / "legacy-casa.cfg"
    layout.write_text(
        "#observatory=IGNORED\n"
        "#coordsys=LOC\n"
        "0 1 2\n"
        "3 4 5 12 STATION\n"
        "6 7 8 13 STATION2 ANTENNA2\n"
        "incomplete row\n"
        "not numeric coordinates\n"
        "9 10 11 not-a-diameter STATION3 ANTENNA3\n"
    )

    antennas = read_casa_format(layout)

    assert list(antennas) == [0, 1, 2, 3]
    assert antennas[0]["Name"] == "A000"
    assert antennas[1]["Name"] == "STATION"
    assert antennas[1]["diameter"] == 12.0
    assert antennas[2]["Name"] == "ANTENNA2"
    assert antennas[2]["diameter"] == 13.0
    assert antennas[3]["Name"] == "ANTENNA3"
    assert "diameter" not in antennas[3]
    assert all(ant["BeamID"] is None for ant in antennas.values())


@pytest.mark.parametrize(
    ("reader", "label"),
    [(read_measurement_set, "measurement_set"), (read_uvfits, "uvfits")],
)
def test_dataset_readers_characterize_full_read_and_legacy_array_mapping(
    tmp_path, monkeypatch, reader, label
):
    read_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeUVData:
        def __init__(self):
            self.antenna_names = np.array(["ANT-A", "ANT-B"])
            self.antenna_numbers = np.array([7, 3])
            # These are dependency-relative ECEF values. Legacy RadioSim copies
            # them directly into its untyped Position tuple.
            self.antenna_positions = np.array(
                [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]
            )
            self.antenna_diameters = np.array([12.0, 0.0])
            self.telescope = SimpleNamespace(
                location=object(),
                mount_type=np.array(["fixed", "alt-az"]),
                feed_array=np.array(["x", "y"]),
            )

        def read(self, *args, **kwargs):
            read_calls.append((args, kwargs))

    monkeypatch.setattr(antenna_module, "UVData", FakeUVData)
    monkeypatch.setattr(antenna_module, "PYUVDATA_AVAILABLE", True)
    source = tmp_path / label

    antennas = reader(source)

    assert read_calls == [((source,), {})]
    assert list(antennas) == [7, 3]
    assert antennas == {
        7: {
            "Name": "ANT-A",
            "Number": 7,
            "BeamID": None,
            "Position": (100.0, 200.0, 300.0),
            "diameter": 12.0,
        },
        3: {
            "Name": "ANT-B",
            "Number": 3,
            "BeamID": None,
            "Position": (400.0, 500.0, 600.0),
        },
    }
    assert all("mount_type" not in ant for ant in antennas.values())
    assert all("feed_array" not in ant for ant in antennas.values())
    assert all("location" not in ant for ant in antennas.values())


@pytest.mark.parametrize("reader", [read_measurement_set, read_uvfits])
def test_dataset_readers_silently_truncate_mismatched_parallel_arrays(
    tmp_path, monkeypatch, reader
):
    """Undesirable legacy behavior: shortest dependency array controls length."""

    class FakeUVData:
        antenna_names = np.array(["ANT-A", "ANT-B"])
        antenna_numbers = np.array([10, 20])
        antenna_positions = np.array([[1.0, 2.0, 3.0]])
        antenna_diameters = np.array([11.0, 22.0])

        def read(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(antenna_module, "UVData", FakeUVData)
    monkeypatch.setattr(antenna_module, "PYUVDATA_AVAILABLE", True)

    antennas = reader(tmp_path / "dataset")

    assert list(antennas) == [10]
    assert antennas[10]["Name"] == "ANT-A"


@pytest.mark.parametrize("reader", [read_measurement_set, read_uvfits])
def test_dataset_readers_wrap_current_none_diameter_failure(
    tmp_path, monkeypatch, reader
):
    class FakeUVData:
        antenna_names = np.array(["ANT-A"])
        antenna_numbers = np.array([1])
        antenna_positions = np.array([[1.0, 2.0, 3.0]])
        antenna_diameters = None

        def read(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(antenna_module, "UVData", FakeUVData)
    monkeypatch.setattr(antenna_module, "PYUVDATA_AVAILABLE", True)

    with pytest.raises(ValueError) as exc_info:
        reader(tmp_path / "dataset")

    assert isinstance(exc_info.value.__cause__, TypeError)


@pytest.mark.parametrize("reader", [read_measurement_set, read_uvfits])
def test_dataset_readers_wrap_dependency_read_errors(tmp_path, monkeypatch, reader):
    dependency_error = RuntimeError("dependency failed")

    class FakeUVData:
        def read(self, *_args, **_kwargs):
            raise dependency_error

    monkeypatch.setattr(antenna_module, "UVData", FakeUVData)
    monkeypatch.setattr(antenna_module, "PYUVDATA_AVAILABLE", True)

    with pytest.raises(ValueError) as exc_info:
        reader(tmp_path / "dataset")

    assert exc_info.value.__cause__ is dependency_error


def test_mwa_parser_characterizes_first_duplicate_row_wins_without_validation(
    tmp_path,
):
    path = tmp_path / "metafits.fits"
    columns = [
        fits.Column(name="Antenna", format="J", array=np.array([7, 7, 8])),
        fits.Column(
            name="TileName",
            format="8A",
            array=np.array(["Tile007", "Tile007", "Tile008"]),
        ),
        fits.Column(name="East", format="D", array=np.array([1.0, 99.0, 4.0])),
        fits.Column(name="North", format="D", array=np.array([2.0, 98.0, 5.0])),
        fits.Column(name="Height", format="D", array=np.array([3.0, 97.0, 6.0])),
        fits.Column(name="Pol", format="1A", array=np.array(["X", "Y", "X"])),
    ]
    tile_data = fits.BinTableHDU.from_columns(columns, name="TILEDATA")
    fits.HDUList([fits.PrimaryHDU(), tile_data]).writeto(path)

    antennas = read_mwa_format(path)

    assert list(antennas) == [7, 8]
    assert antennas[7] == {
        "Name": "Tile007",
        "Number": 7,
        "BeamID": None,
        "Position": (1.0, 2.0, 3.0),
    }
    assert "Pol" not in antennas[7]


def test_pyuvdata_text_parser_characterizes_ambiguous_three_column_contract(
    tmp_path,
):
    path = tmp_path / "positions.txt"
    path.write_text(
        "# no frame or location metadata\n"
        "\n"
        "1 2 3 ignored extra columns\n"
        "not numeric 4\n"
        "4 5\n"
        "6 7 8\n"
    )

    antennas = read_pyuvdata_format(path)

    assert antennas == {
        0: {
            "Name": "ANT000",
            "Number": 0,
            "BeamID": None,
            "Position": (1.0, 2.0, 3.0),
        },
        1: {
            "Name": "ANT001",
            "Number": 1,
            "BeamID": None,
            "Position": (6.0, 7.0, 8.0),
        },
    }


@pytest.mark.parametrize(
    ("format_type", "reader_name"),
    [
        ("radiosim", "read_radiosim_format"),
        ("casa", "read_casa_format"),
        ("measurement_set", "read_measurement_set"),
        ("uvfits", "read_uvfits"),
        ("mwa", "read_mwa_format"),
        ("pyuvdata", "read_pyuvdata_format"),
    ],
)
def test_dispatcher_routes_all_six_current_identifiers(
    tmp_path, monkeypatch, format_type, reader_name
):
    path = tmp_path / "source"
    path.touch()
    calls: list[Path] = []
    expected = {
        9: {
            "Name": "ROUTED",
            "Number": 9,
            "BeamID": None,
            "Position": (0.0, 0.0, 0.0),
        }
    }

    def fake_reader(source_path):
        calls.append(source_path)
        return expected

    monkeypatch.setattr(antenna_module, reader_name, fake_reader)

    assert read_antenna_positions(path, format_type=format_type.upper()) is expected
    assert calls == [path]


def test_dispatcher_rejects_unsupported_format(tmp_path):
    path = tmp_path / "source"
    path.touch()

    with pytest.raises(ValueError, match="Unsupported antenna file format"):
        read_antenna_positions(path, format_type="future-format")


def test_formatter_sorts_outer_keys_but_preserves_nested_numbers_and_mutability():
    antennas = {
        8: {
            "Name": "OUTER-EIGHT",
            "Number": 80,
            "BeamID": 7,
            "Position": (8.0, 0.0, 0.0),
            "diameter": 25.0,
        },
        2: {
            "Name": "OUTER-TWO",
            "Number": 20,
            "BeamID": None,
            "Position": (2.0, 0.0, 0.0),
        },
    }

    formatted = format_antenna_data(antennas)

    assert formatted["names"].tolist() == ["OUTER-TWO", "OUTER-EIGHT"]
    assert formatted["numbers"].tolist() == [20, 80]
    np.testing.assert_array_equal(
        formatted["positions_m"], [[2.0, 0.0, 0.0], [8.0, 0.0, 0.0]]
    )
    assert formatted["beam_ids"].tolist() == [None, "7"]
    assert np.isnan(formatted["diameters"][0])
    assert formatted["diameters"][1] == 25.0

    assert formatted["positions_m"].flags.writeable
    formatted["positions_m"][0, 0] = 123.0
    assert formatted["positions_m"][0, 0] == 123.0


def test_layout_visualizers_consume_canonical_name_number_position_and_diameter(
    tmp_path, monkeypatch
):
    from radiosim.visualization.bokeh_plots import (
        plot_antenna_layout,
        plot_antenna_layout_3d_plotly,
    )

    bundle = resolved_config(tmp_path)
    bundle.runtime.instrument.source.path.write_text(
        "Name Number BeamID E N U Diameter\nVISIBLE 50 0 1.0 2.0 3.0 4.0\n",
        encoding="utf-8",
    )
    simulator = Simulator(bundle.runtime)
    simulator._ensure_instrument_state()
    antennas = simulator.antennas

    figure_2d = plot_antenna_layout(antennas, open_in_browser=False)
    source_data = figure_2d.renderers[0].data_source.data
    assert source_data == {
        "E": [1.0],
        "N": [2.0],
        "Number": ["50"],
        "Name": ["VISIBLE"],
    }

    import plotly.io as pio

    captured: dict[str, object] = {}

    def capture_html(figure, **_kwargs):
        captured["figure"] = figure
        return "<div>captured</div>"

    monkeypatch.setattr(pio, "to_html", capture_html)
    output = plot_antenna_layout_3d_plotly(
        antennas,
        save_simulation_data=True,
        folder_path=str(tmp_path),
        open_in_browser=False,
    )

    assert Path(output) == tmp_path / "antenna_layout_3d.html"
    ring_traces = [
        trace
        for trace in captured["figure"].data
        if getattr(trace, "mode", None) == "lines"
        and getattr(trace.line, "color", None) == "#1f2a44"
    ]
    assert len(ring_traces) == 1
    ring_x = [value for value in ring_traces[0].x if value is not None]
    assert min(ring_x) == pytest.approx(-1.0)
    assert max(ring_x) == pytest.approx(3.0)


def test_simulator_resolution_preserves_source_diameters_and_immutable_state(tmp_path):
    bundle = resolved_config(
        tmp_path,
        instrument={"default_diameter_m": 23.0},
    )
    bundle.runtime.instrument.source.path.write_text(
        "Name Number BeamID E N U Diameter\n"
        "SOURCE0 0 0 0.0 0.0 0.0 5.0\n"
        "SOURCE1 1 0 2.0 0.0 0.0 6.0\n",
        encoding="utf-8",
    )
    simulator = Simulator(bundle.runtime)

    simulator._ensure_instrument_state()

    assert simulator.antennas is simulator.instrument.antennas
    assert tuple(antenna.diameter_m for antenna in simulator.antennas) == (5.0, 6.0)
    assert simulator.baselines[1].vector_enu_m == (2.0, 0.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        simulator.antennas[0].id.name = "MUTATED"
    with pytest.raises(TypeError):
        simulator.baselines[1].vector_enu_m[0] = 99.0


def test_current_results_retain_exact_immutable_canonical_aliases(tmp_path):
    simulator = Simulator(resolved_config(tmp_path).runtime)

    results = simulator.run(progress=False)

    assert results["antennas"] is simulator.antennas
    assert results["baselines"] is simulator.baselines
    with pytest.raises(FrozenInstanceError):
        simulator.antennas[0].id.name = "CHANGED-AFTER-RUN"
    with pytest.raises(TypeError):
        simulator.baselines[1].vector_enu_m[0] = 321.0
    assert results["antennas"][0].id.name == "ANT0"
    assert results["baselines"][1].vector_enu_m == (14.0, 0.0, 0.0)
