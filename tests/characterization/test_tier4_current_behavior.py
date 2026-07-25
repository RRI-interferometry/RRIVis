"""Characterize the pre-Tier-4 RadioSim result and output behavior."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import EarthLocation
from astropy.time import Time

from radiosim.api import Simulator
from radiosim.backends import get_backend
from radiosim.core.instrument_adapters import SolverInstrumentView
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.containers.healpix import HealpixData
from radiosim.core.sky.containers.model import SkyModel
from radiosim.core.visibility import calculate_visibility
from radiosim.core.visibility_healpix import calculate_visibility_healpix
from radiosim.io.config import (
    RadioSimConfig,
    collect_unsupported_issues,
)
from radiosim.io.writers import load_visibilities_hdf5
from tests.fixtures.configs import valid_config_mapping

FREQUENCIES_HZ = np.array([100_000_000.0], dtype=np.float64)
WAVELENGTHS = np.array([c.value / FREQUENCIES_HZ[0]], dtype=np.float64) * u.m
LOCATION = EarthLocation.from_geodetic(
    21.4283 * u.deg,
    -30.72152 * u.deg,
    1073.0 * u.m,
)
START = Time("2025-01-01T00:00:00")


def _solver_components(
    tmp_path: Path,
) -> tuple[SolverInstrumentView, object]:
    simulator = Simulator.from_mapping(
        valid_config_mapping(
            tmp_path,
            frequency={
                "mode": "explicit",
                "channel_frequencies_hz": FREQUENCIES_HZ.tolist(),
            },
        ),
        base_dir=tmp_path,
    )
    simulator._ensure_instrument_state()
    simulator._ensure_beam_system()
    return (
        SolverInstrumentView.from_state(simulator._instrument_state),
        simulator.beam_system,
    )


def _empty_point_sources() -> dict[str, np.ndarray]:
    empty = np.array([], dtype=np.float64)
    return {
        "ra_rad": empty,
        "dec_rad": empty,
        "ref_freq": empty,
    }


def _tiny_healpix_model() -> SkyModel:
    nside = 1
    return SkyModel(
        healpix=HealpixData(
            maps=np.zeros((1, 12), dtype=np.float64),
            nside=nside,
            frequencies=FREQUENCIES_HZ,
            coordinate_frame="icrs",
        ),
        model_name="tier4-current-time-characterization",
        brightness_conversion="rayleigh-jeans",
        precision=PrecisionConfig.standard(),
    )


@pytest.mark.parametrize(
    ("duration", "cadence", "point_count", "healpix_count"),
    [
        (2.5, 1.0, 2, 3),
        (3.0, 1.0, 3, 3),
        (1.0, 1.0, 1, 1),
    ],
)
def test_current_point_and_healpix_time_counts_disagree_only_when_nondivisible(
    tmp_path: Path,
    duration: float,
    cadence: float,
    point_count: int,
    healpix_count: int,
) -> None:
    """Characterizes current counts; this is not the target time contract."""
    instrument, beam_system = _solver_components(tmp_path)
    backend = get_backend("numpy")

    point = calculate_visibility(
        instrument=instrument,
        beam_system=beam_system,
        source_arrays=_empty_point_sources(),
        location=LOCATION,
        obstime=START,
        wavelengths=WAVELENGTHS,
        freqs=FREQUENCIES_HZ,
        duration_seconds=duration,
        time_step_seconds=cadence,
        backend=backend,
    )
    healpix = calculate_visibility_healpix(
        sky_model=_tiny_healpix_model(),
        instrument=instrument,
        beam_system=beam_system,
        location=LOCATION,
        obstime=START,
        wavelengths=WAVELENGTHS,
        freqs=FREQUENCIES_HZ,
        duration_seconds=duration,
        time_step_seconds=cadence,
        backend=backend,
    )

    pair = instrument.selected_pairs[0]
    assert point[pair]["XX"].shape == (point_count, 1)
    assert healpix["visibilities"].shape[1:] == (healpix_count, 1)
    np.testing.assert_array_equal(
        healpix["times"],
        np.arange(healpix_count, dtype=np.float64) * cadence,
    )


def _controlled_simulator(
    tmp_path: Path,
    *,
    duration: float = 2.5,
    cadence: float = 1.0,
) -> tuple[Simulator, tuple[int, int]]:
    data = valid_config_mapping(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [100e6, 101e6],
        },
        obs_time={
            "start_time": START.isot,
            "duration_seconds": duration,
            "time_step_seconds": cadence,
        },
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    pair = simulator._instrument_state.selection.provenance.selected_ids[0]
    samples = np.arange(6, dtype=np.float32).reshape(3, 2)
    simulator._obstime = START
    simulator._frequencies_hz = np.array([100e6, 101e6], dtype=np.float64)
    simulator._results = {
        "visibilities": {
            pair: {
                "I": (samples + 1j * (samples + 1)).astype(np.complex64),
                "XX": (samples + 10 + 2j).astype(np.complex64),
                "XY": (samples + 20 + 3j).astype(np.complex64),
                "YX": (samples + 30 + 4j).astype(np.complex64),
                "YY": (samples + 40 + 5j).astype(np.complex64),
            }
        },
        "frequencies": simulator._frequencies_hz,
        "baselines": simulator.baselines,
        "antennas": simulator.antennas,
        "obstime": START,
        "metadata": {
            "version": simulator.version,
            "nested": {"purpose": "current-characterization"},
        },
    }
    return simulator, pair


def test_current_save_and_plot_reconstruct_floor_like_axis_from_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Characterizes independent writer/plot axes, not a desired contract."""
    simulator, _pair = _controlled_simulator(tmp_path)
    captured: dict[str, np.ndarray] = {}

    def record_hdf5(**kwargs: object) -> None:
        captured["save"] = np.asarray(kwargs["time_points_mjd"])

    def record_visibility(**kwargs: object) -> None:
        captured["plot"] = np.asarray(kwargs["mjd_time_points"])

    def no_op(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "radiosim.io.writers.save_visibilities_hdf5",
        record_hdf5,
    )
    monkeypatch.setattr(
        "radiosim.visualization.bokeh_plots.plot_visibility",
        record_visibility,
    )
    monkeypatch.setattr(
        "radiosim.visualization.bokeh_plots.plot_antenna_layout",
        no_op,
    )
    monkeypatch.setattr(
        "radiosim.visualization.bokeh_plots.plot_antenna_layout_3d_plotly",
        no_op,
    )
    monkeypatch.setattr(
        "radiosim.visualization.bokeh_plots.plot_heatmaps",
        no_op,
    )
    monkeypatch.setattr(
        "radiosim.visualization.bokeh_plots.plot_modulus_vs_frequency",
        no_op,
    )

    simulator.save(tmp_path / "save", format="hdf5")
    simulator.plot(plot_type="all", output_dir=tmp_path / "plots", show=False)

    expected = START.mjd + np.arange(2) / 86400.0
    np.testing.assert_allclose(captured["save"], expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(captured["plot"], expected, rtol=0.0, atol=1e-12)
    assert simulator.results["visibilities"][_pair]["I"].shape[0] == 3


def test_current_run_and_results_expose_one_mutable_nested_alias(
    tmp_path: Path,
) -> None:
    """Characterizes the current mutable result lifecycle, not the target."""
    data = valid_config_mapping(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [100e6],
        },
        obs_time={
            "duration_seconds": 1.0,
            "time_step_seconds": 1.0,
        },
        sky_model={
            "sources": [
                {
                    "kind": "test_sources",
                    "representation": "point_sources",
                    "num_sources": 1,
                    "seed": 4,
                }
            ]
        },
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)

    returned = simulator.run(progress=False)
    assert returned is simulator.results
    assert isinstance(returned["visibilities"], dict)
    pair = next(iter(returned["visibilities"]))
    assert set(returned["visibilities"][pair]) == {"XX", "XY", "YX", "YY", "I"}

    original = returned["visibilities"][pair]["XX"][0, 0]
    returned["visibilities"][pair]["XX"][0, 0] = original + (7.0 + 3.0j)
    assert simulator.results["visibilities"][pair]["XX"][0, 0] != original
    assert (
        simulator.results["visibilities"][pair]["XX"]
        is returned["visibilities"][pair]["XX"]
    )


def test_current_hdf5_is_lossy_promoting_unversioned_and_direct(
    tmp_path: Path,
) -> None:
    """Characterizes current HDF5 loss; it does not endorse the format."""
    simulator, first_pair = _controlled_simulator(tmp_path)
    second_pair = simulator._instrument_state.selection.provenance.selected_ids[1]
    first_products = simulator._results["visibilities"][first_pair]
    second_products = {
        key: value + np.complex64(100 + 10j)
        for key, value in first_products.items()
        if key != "I"
    }
    simulator._results["visibilities"][second_pair] = second_products

    output_dir = tmp_path / "direct"
    output = simulator.save(output_dir, format="hdf5", filename="legacy")

    assert output == output_dir / "legacy.h5"
    assert sorted(path.name for path in output_dir.iterdir()) == ["legacy.h5"]
    with h5py.File(output, "r") as handle:
        first_group = handle[f"baseline_{first_pair}"]
        second_group = handle[f"baseline_{second_pair}"]
        assert set(first_group) == {"complex_visibility"}
        assert set(second_group) == {"complex_visibility"}
        first_stored = first_group["complex_visibility"][:]
        second_stored = second_group["complex_visibility"][:]
        assert first_stored.dtype == np.dtype(np.complex128)
        assert second_stored.dtype == np.dtype(np.complex128)
        np.testing.assert_array_equal(first_stored, first_products["I"])
        np.testing.assert_array_equal(second_stored, second_products["XX"])
        assert "schema_name" not in handle.attrs
        assert "schema_version" not in handle.attrs
        assert str(handle.attrs["nested"]).startswith("__radiosim_json__:")
        assert f"baseline_{first_pair}" in handle
        assert first_products["XY"].tobytes() not in first_stored.tobytes()
        assert second_products["YY"].tobytes() not in second_stored.tobytes()


def test_current_json_is_an_ambiguous_nonreconstructable_summary(
    tmp_path: Path,
) -> None:
    """Characterizes current JSON omission; this is not a result contract."""
    simulator, _pair = _controlled_simulator(tmp_path)

    output = simulator.save(tmp_path / "json", format="json", filename="legacy")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output.name == "legacy.json"
    assert set(payload) == {"metadata", "frequencies", "n_baselines"}
    assert payload["frequencies"] == [100e6, 101e6]
    assert payload["n_baselines"] == len(simulator.baselines)
    assert "visibilities" not in payload
    assert "times" not in payload
    assert "channel_widths" not in payload
    assert "flags" not in payload
    assert "weights" not in payload
    assert set(payload) != set(simulator.results)


def test_legacy_hdf5_reader_evaluates_arithmetic_baseline_text(
    tmp_path: Path,
) -> None:
    """Characterizes unsafe legacy parsing without executing side effects."""
    path = tmp_path / "legacy-eval.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("baseline_(1 + 1, 3 * 2)")
        group.create_dataset(
            "complex_visibility",
            data=np.ones((1, 1), dtype=np.complex128),
        )

    loaded = load_visibilities_hdf5(path)
    assert set(loaded["visibilities"]) == {(2, 6)}

    # Temporary characterization: Tier 4H removes this direct eval call.
    tree = ast.parse(inspect.getsource(load_visibilities_hdf5))
    direct_eval_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "eval"
    ]
    assert len(direct_eval_calls) == 1


def test_current_save_creates_directory_before_unknown_format_error(
    tmp_path: Path,
) -> None:
    """Characterizes current side-effect ordering; it is not future policy."""
    simulator, _pair = _controlled_simulator(tmp_path)
    absent = tmp_path / "created-before-error"
    assert not absent.exists()

    with pytest.raises(ValueError, match="Unknown format"):
        simulator.save(absent, format="unknown")

    assert absent.is_dir()
    assert list(absent.iterdir()) == []


def test_current_high_level_ms_passes_scalar_time_for_multitime_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Characterizes the scalar-time MS mismatch, not the target mapping."""
    from radiosim.io.measurement_set import write_ms as active_write_ms

    simulator, pair = _controlled_simulator(tmp_path)
    captured: dict[str, object] = {}

    def record_ms(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("radiosim.io.measurement_set.write_ms", record_ms)
    output = simulator.save(tmp_path / "ms", format="ms")

    assert output.name == "simulation.ms"
    assert captured["obstime"] is START
    assert captured["obstime"].isscalar
    assert captured["visibilities"][pair]["XX"].shape[0] == 3
    assert "times" not in captured
    assert "integration_time" not in captured
    assert captured["selection"].provenance.selected_ids == (
        simulator._instrument_state.selection.provenance.selected_ids
    )

    parameters = inspect.signature(active_write_ms).parameters
    assert "obstime" in parameters
    assert "integration_time" in parameters
    assert "time_points" not in parameters


def test_current_uvfits_is_rejected_by_config_and_direct_save(
    tmp_path: Path,
) -> None:
    """Characterizes both active UVFITS rejection surfaces."""
    data = valid_config_mapping(tmp_path, workflow={"result_format": "uvfits"})
    config = RadioSimConfig.model_validate(data)
    issues = collect_unsupported_issues(config)
    uvfits = [
        issue
        for issue in issues
        if issue.path == "workflow.result_format" and issue.code == "uvfits_unsupported"
    ]
    assert len(uvfits) == 1
    assert uvfits[0].stage == "unsupported"

    fixture_dir = tmp_path / "direct-fixture"
    fixture_dir.mkdir()
    simulator, _pair = _controlled_simulator(fixture_dir)
    output_dir = tmp_path / "direct-uvfits"
    output_dir.mkdir()
    with pytest.raises(ValueError, match="Unknown format: uvfits"):
        simulator.save(output_dir, format="uvfits")
    assert list(output_dir.iterdir()) == []
