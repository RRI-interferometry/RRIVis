"""Characterize the active Tier 4C result and output boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation

from radiosim.api import Simulator
from radiosim.backends import get_backend
from radiosim.core.instrument_adapters import SolverInstrumentView
from radiosim.core.precision import PrecisionConfig
from radiosim.core.result import ResultUnavailableError, SimulationResult
from radiosim.core.sky.containers.healpix import HealpixData
from radiosim.core.sky.containers.model import SkyModel
from radiosim.core.time_grid import build_observation_time_grid
from radiosim.core.visibility import calculate_visibility
from radiosim.core.visibility_healpix import calculate_visibility_healpix
from radiosim.io.config import RadioSimConfig, collect_unsupported_issues
from radiosim.io.result_format import ResultFormat
from tests.fixtures.configs import valid_config_mapping

FREQUENCIES_HZ = np.array([100_000_000.0], dtype=np.float64)
LOCATION = EarthLocation.from_geodetic(
    21.4283 * u.deg,
    -30.72152 * u.deg,
    1073.0 * u.m,
)
START_ISO = "2025-01-01T00:00:00"


def _solver_components(
    tmp_path: Path,
) -> tuple[SolverInstrumentView, object, object]:
    simulator = Simulator.from_mapping(
        valid_config_mapping(
            tmp_path,
            frequency={
                "mode": "explicit",
                "channel_frequencies_hz": FREQUENCIES_HZ.tolist(),
                "channel_widths_hz": [1e6],
            },
        ),
        base_dir=tmp_path,
    )
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    return (
        SolverInstrumentView.from_state(simulator._instrument_state),
        simulator.beam_system,
        simulator.receptors,
    )


def _empty_point_sources() -> dict[str, np.ndarray]:
    empty = np.array([], dtype=np.float64)
    return {
        "ra_rad": empty,
        "dec_rad": empty,
        "ref_freq": empty,
    }


def _tiny_healpix_model() -> SkyModel:
    return SkyModel(
        healpix=HealpixData(
            maps=np.zeros((1, 12), dtype=np.float64),
            nside=1,
            frequencies=FREQUENCIES_HZ,
            coordinate_frame="icrs",
        ),
        model_name="tier4c-time-characterization",
        brightness_conversion="rayleigh-jeans",
        precision=PrecisionConfig.standard(),
    )


@pytest.mark.parametrize(
    ("duration", "cadence", "expected_count"),
    [
        (2.5, 1.0, 3),
        (3.0, 1.0, 3),
        (1.0, 1.0, 1),
    ],
)
def test_point_and_healpix_use_the_same_canonical_time_count(
    tmp_path: Path,
    duration: float,
    cadence: float,
    expected_count: int,
) -> None:
    instrument, beam_system, receptors = _solver_components(tmp_path)
    backend = get_backend("numpy")
    time_grid = build_observation_time_grid(
        start_time=START_ISO,
        duration_seconds=duration,
        cadence_seconds=cadence,
    )

    point = calculate_visibility(
        instrument=instrument,
        beam_system=beam_system,
        source_arrays=_empty_point_sources(),
        location=LOCATION,
        time_grid=time_grid,
        frequencies=FREQUENCIES_HZ,
        backend=backend,
        receptors=receptors,
    )
    healpix = calculate_visibility_healpix(
        sky_model=_tiny_healpix_model(),
        instrument=instrument,
        beam_system=beam_system,
        location=LOCATION,
        time_grid=time_grid,
        frequencies=FREQUENCIES_HZ,
        backend=backend,
        receptors=receptors,
    )

    expected_shape = (
        expected_count,
        len(instrument.selected_pairs),
        len(FREQUENCIES_HZ),
        2,
        2,
    )
    assert point.shape == expected_shape
    assert healpix.shape == expected_shape


def _canonical_simulator(tmp_path: Path) -> Simulator:
    data = valid_config_mapping(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [100e6, 101e6],
            "channel_widths_hz": [1e6, 0.5e6],
        },
        obs_time={
            "start_time": START_ISO,
            "duration_seconds": 2.5,
            "time_step_seconds": 1.0,
        },
    )
    return Simulator.from_mapping(data, base_dir=tmp_path)


def test_run_publishes_one_immutable_canonical_result(tmp_path: Path) -> None:
    simulator = _canonical_simulator(tmp_path)

    returned = simulator.run(progress=False)

    assert type(returned) is SimulationResult
    assert returned is simulator.result
    assert not hasattr(simulator, "results")
    assert returned.visibilities.shape == (3, 3, 2, 4)
    assert returned.correlations == ("XX", "XY", "YX", "YY")
    assert returned.visibilities.flags.writeable is False
    assert returned.flags.flags.writeable is False
    assert returned.weights.flags.writeable is False
    np.testing.assert_array_equal(
        returned.stokes_i(),
        returned.visibilities[..., 0] + returned.visibilities[..., 3],
    )


@pytest.mark.parametrize("operation", ["save", "plot"])
@pytest.mark.parametrize("run_first", [False, True])
def test_save_and_plot_both_require_one_published_result(
    tmp_path: Path,
    operation: str,
    run_first: bool,
) -> None:
    simulator = _canonical_simulator(tmp_path)
    if run_first:
        simulator.run(progress=False)
    output = tmp_path / operation

    if run_first:
        if operation == "save":
            target = output.with_suffix(".h5")
            assert simulator.save(target, format=ResultFormat.HDF5) == target
            assert target.is_file()
            return
        published = simulator.plot(
            plot_type="visibility",
            output_dir=output,
            show=False,
        )
        assert [path.name for path in published] == ["visibility-phase-lsts.html"]
        assert published[0].is_file()
        return

    with pytest.raises(ResultUnavailableError):
        if operation == "save":
            simulator.save(output.with_suffix(".h5"), format=ResultFormat.HDF5)
        else:
            simulator.plot(output_dir=output, show=False)

    assert not output.exists()


def test_uvfits_config_and_direct_save_are_available(
    tmp_path: Path,
) -> None:
    data = valid_config_mapping(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [100e6, 101e6],
            "channel_widths_hz": [1e6, 1e6],
        },
        workflow={"result_format": "uvfits"},
    )
    config = RadioSimConfig.model_validate(data)
    issues = collect_unsupported_issues(config)
    uvfits = [
        issue
        for issue in issues
        if issue.path == "workflow.result_format" and issue.code == "uvfits_unsupported"
    ]
    assert uvfits == []

    fixture_dir = tmp_path / "direct-fixture"
    fixture_dir.mkdir()
    simulator = Simulator.from_mapping(data, base_dir=fixture_dir)
    simulator.run(progress=False)
    output = tmp_path / "direct.uvfits"

    assert simulator.save(output, format=ResultFormat.UVFITS) == output
    assert output.is_file()
