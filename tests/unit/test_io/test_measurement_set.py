"""Canonical Measurement Set writer contracts for Tier 2G."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.time import Time
from pyuvdata.utils import ENU_from_ECEF

import radiosim.io.measurement_set as measurement_set_module
from radiosim.api import Simulator
from radiosim.io.measurement_set import write_ms
from radiosim.io.writers import load_visibilities_hdf5, save_visibilities_hdf5
from tests.fixtures.configs import valid_config_mapping


class _FakeUVData:
    def __init__(self):
        object.__setattr__(self, "check_calls", 0)
        object.__setattr__(self, "write_calls", [])

    def __setattr__(self, name, value):
        if name == "uvw_array":
            raise AssertionError("writer must not assign UVW directly")
        object.__setattr__(self, name, value)

    def set_uvws_from_antenna_positions(self):
        raise AssertionError("writer must not redundantly recompute UVW")

    def check(self):
        self.check_calls += 1

    def write_ms(self, *args, **kwargs):
        self.write_calls.append((args, kwargs))


def _canonical_state(
    tmp_path,
    *,
    maximum_number: bool = False,
    correlations: str = "cross",
):
    data = valid_config_mapping(
        tmp_path,
        baseline_selection={"correlations": correlations},
    )
    second = 2_147_483_647 if maximum_number else 9
    (tmp_path / "antennas.txt").write_text(
        "Name Number BeamID E N U Diameter\n"
        "ALPHA 2 0 1.0 2.0 3.0 12.0\n"
        f"OMEGA {second} 0 4.0 5.0 6.0 25.0\n",
        encoding="utf-8",
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    return simulator._instrument_state


def _install_fakes(monkeypatch):
    captured: dict[str, object] = {}
    fake_uvd = _FakeUVData()
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


def test_write_ms_preserves_canonical_identity_geometry_and_registry_policy(
    tmp_path,
    monkeypatch,
):
    state = _canonical_state(tmp_path, maximum_number=True)
    captured, fake_uvd, telescope = _install_fakes(monkeypatch)
    pair = state.selection.provenance.selected_ids[0]
    before_instrument = state.instrument.to_snapshot()
    before_selection = state.selection.to_snapshot()
    output = tmp_path / "canonical.ms"

    returned = write_ms(
        output_path=output,
        visibilities={pair: {"XX": np.array([[1.0 + 2.0j]])}},
        frequencies=np.array([100e6]),
        instrument=state.instrument,
        selection=state.selection,
        obstime=Time("2024-01-01T00:00:00"),
        integration_time=2.5,
    )

    assert returned == output
    telescope_args = captured["telescope"]
    assert telescope_args["name"] == state.instrument.name
    assert telescope_args["antenna_names"] == ["ALPHA", "OMEGA"]
    assert telescope_args["antenna_numbers"] == [2, 2_147_483_647]
    assert telescope_args["antenna_diameters"] == [12.0, 25.0]
    assert telescope_args["update_from_known"] is False

    relative_ecef = telescope_args["antenna_positions"]
    center = np.asarray(state.instrument.location.itrs_xyz_m)
    absolute_ecef = np.stack(
        [relative_ecef[number] + center for number in telescope_args["antenna_numbers"]]
    )
    round_trip = ENU_from_ECEF(
        absolute_ecef,
        center_loc=telescope_args["location"],
    )
    np.testing.assert_allclose(
        round_trip,
        [antenna.position_enu_m for antenna in state.instrument.antennas],
        atol=1e-6,
    )

    uvdata_args = captured["uvdata"]
    assert uvdata_args["telescope"] is telescope
    assert uvdata_args["antpairs"] == [pair]
    assert uvdata_args["update_telescope_from_known"] is False
    assert fake_uvd.check_calls == 1
    assert fake_uvd.write_calls == [
        ((str(output),), {"clobber": False, "force_phase": True})
    ]
    assert state.instrument.to_snapshot() == before_instrument
    assert state.selection.to_snapshot() == before_selection


def test_write_ms_uses_selected_pairs_only_and_canonical_order(tmp_path, monkeypatch):
    state = _canonical_state(tmp_path)
    captured, _fake_uvd, _telescope = _install_fakes(monkeypatch)
    pair = state.selection.provenance.selected_ids[0]

    write_ms(
        output_path=tmp_path / "selected.ms",
        visibilities={pair: np.array([1.0 + 0.0j])},
        frequencies=np.array([100e6]),
        instrument=state.instrument,
        selection=state.selection,
        obstime=Time("2024-01-01T00:00:00"),
    )

    assert captured["uvdata"]["antpairs"] == list(
        state.selection.provenance.selected_ids
    )


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0 + 0.0j]),
        np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
        np.array([[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]]),
    ],
)
def test_write_ms_rejects_frequency_or_time_shape_mismatch(
    tmp_path,
    monkeypatch,
    values,
):
    state = _canonical_state(tmp_path)
    _install_fakes(monkeypatch)
    pair = state.selection.provenance.selected_ids[0]

    with pytest.raises(ValueError, match="shape"):
        write_ms(
            output_path=tmp_path / "invalid-shape.ms",
            visibilities={pair: {"XX": values}},
            frequencies=np.array([100e6, 101e6]),
            instrument=state.instrument,
            selection=state.selection,
            obstime=Time("2024-01-01T00:00:00"),
            polarizations=["XX"],
        )


def test_write_ms_rejects_inconsistent_polarization_shapes(tmp_path, monkeypatch):
    state = _canonical_state(tmp_path)
    _install_fakes(monkeypatch)
    pair = state.selection.provenance.selected_ids[0]

    with pytest.raises(ValueError, match="shape"):
        write_ms(
            output_path=tmp_path / "invalid-polarization-shape.ms",
            visibilities={
                pair: {
                    "XX": np.ones((1, 2), dtype=np.complex128),
                    "YY": np.ones((1, 1), dtype=np.complex128),
                }
            },
            frequencies=np.array([100e6, 101e6]),
            instrument=state.instrument,
            selection=state.selection,
            obstime=Time("2024-01-01T00:00:00"),
            polarizations=["XX", "YY"],
        )


def test_write_ms_aligns_time_major_rows_with_local_pyuvdata_contract(
    tmp_path,
    monkeypatch,
):
    state = _canonical_state(tmp_path, correlations="all")
    captured, fake_uvd, _telescope = _install_fakes(monkeypatch)
    pairs = state.selection.provenance.selected_ids
    visibilities = {
        pair: np.array(
            [
                [complex(10 * pair_index + 1), complex(10 * pair_index + 2)],
                [complex(10 * pair_index + 3), complex(10 * pair_index + 4)],
            ]
        )
        for pair_index, pair in enumerate(pairs)
    }

    write_ms(
        output_path=tmp_path / "ordered.ms",
        visibilities=visibilities,
        frequencies=np.array([100e6, 101e6]),
        instrument=state.instrument,
        selection=state.selection,
        obstime=Time(["2024-01-01T00:00:00", "2024-01-01T00:00:01"]),
        polarizations=["XX"],
    )

    assert captured["uvdata"]["time_axis_faster_than_bls"] is False
    assert captured["uvdata"]["antpairs"] == list(pairs)
    np.testing.assert_array_equal(
        fake_uvd.data_array[:, :, 0],
        np.array(
            [
                visibilities[pair][time_index]
                for time_index in range(2)
                for pair in pairs
            ]
        ),
    )


def test_hdf5_round_trip_preserves_nested_instrument_resolution_metadata(tmp_path):
    output = tmp_path / "metadata.h5"
    metadata = {
        "instrument_resolution": {
            "schema_version": "radiosim.instrument.v1",
            "selected_ids": [[0, 1]],
        }
    }

    save_visibilities_hdf5(
        output_path=output,
        visibilities={(0, 1): [np.array([1.0 + 0.0j])]},
        frequencies=np.array([100e6]),
        time_points_mjd=np.array([60_000.0]),
        metadata=metadata,
    )

    loaded = load_visibilities_hdf5(output)
    assert (
        loaded["metadata"]["instrument_resolution"] == metadata["instrument_resolution"]
    )


def test_hdf5_rejects_nonfinite_nested_metadata_before_creating_output(tmp_path):
    output = tmp_path / "nonfinite" / "metadata.h5"

    with pytest.raises(ValueError):
        save_visibilities_hdf5(
            output_path=output,
            visibilities={(0, 1): [np.array([1.0 + 0.0j])]},
            frequencies=np.array([100e6]),
            time_points_mjd=np.array([60_000.0]),
            metadata={"instrument_resolution": {"diameter_m": float("nan")}},
        )

    assert not output.exists()
    assert not output.parent.exists()
