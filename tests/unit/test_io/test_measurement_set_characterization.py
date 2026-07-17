"""Characterization tests for legacy Measurement Set and observability adapters.

The writer assertions describe current assumptions that the Tier 2 replacement
will remove. No real Measurement Set or casacore operation is performed.
"""

from __future__ import annotations

from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c
from astropy.coordinates import EarthLocation
from astropy.time import Time

import radiosim.io.measurement_set as measurement_set_module
from radiosim.api import Simulator
from radiosim.core.observability import ObservabilityPlanner
from radiosim.io.measurement_set import write_ms
from tests.fixtures.configs import resolved_config


class _FakeUVDataResult:
    def __init__(self):
        self.uvw_array = np.full((1, 3), -999.0)
        self.check_calls = 0
        self.write_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def check(self):
        self.check_calls += 1

    def write_ms(self, *args, **kwargs):
        self.write_calls.append((args, kwargs))


def _install_writer_fakes(monkeypatch):
    captured: dict[str, object] = {}
    fake_uvd = _FakeUVDataResult()
    telescope = object()

    def telescope_new(**kwargs):
        captured["telescope"] = kwargs
        return telescope

    def uvdata_new(**kwargs):
        captured["uvdata"] = kwargs
        return fake_uvd

    monkeypatch.setattr(measurement_set_module, "_check_ms_dependencies", lambda: None)
    monkeypatch.setattr(
        measurement_set_module,
        "Telescope",
        SimpleNamespace(new=telescope_new),
    )
    monkeypatch.setattr(
        measurement_set_module,
        "UVData",
        SimpleNamespace(new=uvdata_new),
    )
    return captured, fake_uvd, telescope


def test_write_ms_characterizes_identity_diameter_position_and_dispatch_assumptions(
    tmp_path, monkeypatch
):
    captured, fake_uvd, telescope = _install_writer_fakes(monkeypatch)
    location = EarthLocation.from_geodetic(
        lon=0.0 * u.deg,
        lat=0.0 * u.deg,
        height=0.0 * u.m,
    )
    antennas = {
        "late": {
            "Name": "SOURCE-NAME-NINE",
            "Number": 9,
            "Position": (4.0, 5.0, 6.0),
            "diameter": 25.0,
        },
        "early": {
            "Name": "SOURCE-NAME-TWO",
            "Number": 2,
            "Position": (1.0, 2.0, 3.0),
        },
    }
    output = tmp_path / "characterized.ms"

    returned = write_ms(
        output_path=output,
        visibilities={(2, 9): np.array([1.0 + 2.0j])},
        frequencies=np.array([100e6]),
        antennas=antennas,
        baselines={(2, 9): {"BaselineVector": np.array([9.0, 8.0, 7.0])}},
        location=location,
        obstime=Time("2024-01-01T00:00:00"),
        telescope_name="LegacyScope",
        instrument_name="LegacyInstrument",
        phase_center_ra=30.0,
        phase_center_dec=-20.0,
        integration_time=2.5,
    )

    assert returned == output
    telescope_args = captured["telescope"]
    assert telescope_args["name"] == "LegacyScope"
    assert telescope_args["instrument"] == "LegacyInstrument"
    assert telescope_args["location"] is location
    assert telescope_args["antenna_names"] == ["ANT002", "ANT009"]
    assert telescope_args["antenna_diameters"] == [14.0, 25.0]
    assert list(telescope_args["antenna_positions"]) == [2, 9]
    # At latitude=longitude=0, the writer's ENU-to-relative-ECEF rotation is
    # (E, N, U) -> (U, E, N).
    np.testing.assert_array_equal(
        telescope_args["antenna_positions"][2],
        [3.0, 1.0, 2.0],
    )
    np.testing.assert_array_equal(
        telescope_args["antenna_positions"][9],
        [6.0, 4.0, 5.0],
    )
    assert "antenna_numbers" not in telescope_args
    assert "update_from_known" not in telescope_args

    uvdata_args = captured["uvdata"]
    assert uvdata_args["telescope"] is telescope
    assert uvdata_args["antpairs"] == [(2, 9)]
    assert uvdata_args["polarization_array"] == ["XX"]
    assert uvdata_args["do_blt_outer"] is True
    assert uvdata_args["time_axis_faster_than_bls"] is False
    assert "update_telescope_from_known" not in uvdata_args

    # Legacy RadioSim manually replaces any dependency-derived UVW.
    np.testing.assert_array_equal(fake_uvd.uvw_array, [[9.0, 8.0, 7.0]])
    assert fake_uvd.phase_center_ra == pytest.approx(np.pi / 6.0)
    assert fake_uvd.phase_center_dec == pytest.approx(np.deg2rad(-20.0))
    np.testing.assert_array_equal(fake_uvd.integration_time, [2.5])
    np.testing.assert_array_equal(fake_uvd.channel_width, [1e6])
    assert fake_uvd.check_calls == 1
    assert fake_uvd.write_calls == [
        ((str(output),), {"clobber": False, "force_phase": True})
    ]
    assert not output.exists()


@pytest.mark.parametrize(
    ("baselines", "expected_uvw"),
    [
        (
            {(1, 2): {"BaselineVector": np.array([8.0, 7.0, 6.0])}},
            [8.0, 7.0, 6.0],
        ),
        ({}, [3.0, 4.0, 0.0]),
        ({(1, 2): {}}, [0.0, 0.0, 0.0]),
    ],
)
def test_write_ms_characterizes_baseline_vector_and_two_distinct_fallback_paths(
    tmp_path, monkeypatch, baselines, expected_uvw
):
    _captured, fake_uvd, _telescope = _install_writer_fakes(monkeypatch)
    antennas = {
        "one": {"Number": 1, "Position": (0.0, 0.0, 0.0)},
        "two": {"Number": 2, "Position": (3.0, 4.0, 0.0)},
    }

    write_ms(
        output_path=tmp_path / "fallback.ms",
        visibilities={(1, 2): np.array([1.0 + 0.0j])},
        frequencies=np.array([100e6]),
        antennas=antennas,
        baselines=baselines,
        location=EarthLocation.from_geodetic(0.0, 0.0, 0.0),
        obstime=Time("2024-01-01T00:00:00"),
    )

    np.testing.assert_array_equal(fake_uvd.uvw_array[0], expected_uvw)


@pytest.mark.parametrize(
    ("configured_diameter", "effective_diameter"),
    [(None, 14.0), (0.0, 14.0), (28.0, 28.0)],
)
def test_observability_field_radius_characterizes_truthy_14m_fallback(
    configured_diameter, effective_diameter
):
    planner = ObservabilityPlanner(
        frequency_mhz=150.0,
        beam_diameter_m=configured_diameter,
    )

    radius = planner._resolve_field_radius_deg()

    wavelength_m = c.value / 150e6
    expected = np.degrees(1.22 * wavelength_m / effective_diameter) / 2.0
    assert radius == pytest.approx(expected)


def test_simulator_observability_uses_one_configured_scalar_before_setup(
    tmp_path, monkeypatch
):
    captured: dict[str, object] = {}

    class FakePlanner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def build(self):
            return object()

    class FakeRenderer:
        def __init__(self, _plan, **_kwargs):
            pass

        def create_plot(self):
            return "layout"

        def save(self, *_args, **_kwargs):
            raise AssertionError("save should not be called")

    monkeypatch.setattr(
        "radiosim.core.observability.ObservabilityPlanner",
        FakePlanner,
    )
    monkeypatch.setattr(
        "radiosim.visualization.observability.ObservabilityBokehRenderer",
        FakeRenderer,
    )
    simulator = Simulator(
        resolved_config(
            tmp_path,
            antenna_layout={"all_antenna_diameter": 31.0},
        ).runtime
    )

    layout = simulator.plot_observability(
        lst_start_hours=1.0,
        lst_end_hours=2.0,
        open_in_browser=False,
    )

    assert layout == "layout"
    assert simulator.antennas is None
    assert captured["beam_diameter_m"] == 31.0
    assert isinstance(captured["beam_diameter_m"], float)
