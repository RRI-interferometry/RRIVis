"""Contracts for the canonical Tier 4 observation time grid."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy.time import Time

from radiosim.core.result import InvalidTimeGridError, TimeGridLimitError
from radiosim.core.time_grid import (
    MAX_TIME_SAMPLES,
    ObservationTimeGrid,
    build_observation_time_grid,
)


def _grid(*, duration: float = 3.0, cadence: float = 1.0) -> ObservationTimeGrid:
    return build_observation_time_grid(
        start_time="2025-01-01T00:00:00",
        duration_seconds=duration,
        cadence_seconds=cadence,
    )


@pytest.mark.parametrize(
    ("duration", "cadence", "expected"),
    [
        (1.0, 1.0, (0.0,)),
        (3.0, 1.0, (0.0, 1.0, 2.0)),
        (2.5, 1.0, (0.0, 1.0, 2.0)),
        (2.2, 1.0, (0.0, 1.0, 2.0)),
        (1.0, 0.4, (0.0, 0.4, 0.8)),
    ],
)
def test_time_grid_uses_half_open_integer_index_centers(
    duration,
    cadence,
    expected,
):
    grid = _grid(duration=duration, cadence=cadence)
    offsets = (grid.as_astropy() - grid.as_astropy()[0]).to_value("s")

    assert type(grid) is ObservationTimeGrid
    assert grid.schema_version == "radiosim.time-grid.v1"
    assert grid.interval_semantics == "half_open_sample_centers"
    assert tuple(offsets) == pytest.approx(expected, abs=2e-9)
    assert tuple(grid.integration_time_seconds) == (cadence,) * len(expected)


def test_time_grid_normalizes_only_values_inside_the_specified_tolerance():
    epsilon = np.finfo(np.float64).eps

    inside = _grid(duration=3.0 * (1.0 + 8.0 * epsilon))
    outside = _grid(duration=3.0 * (1.0 + 64.0 * epsilon))

    assert len(inside) == 3
    assert len(outside) == 4


def test_time_grid_owns_bytes_backed_immutable_coordinates_and_returns_copies():
    grid = _grid()
    for coordinate in (
        grid.utc_jd1,
        grid.utc_jd2,
        grid.integration_time_seconds,
    ):
        assert type(coordinate) is np.ndarray
        assert coordinate.dtype == np.dtype("float64")
        assert coordinate.flags.c_contiguous
        assert coordinate.flags.writeable is False
        with pytest.raises(ValueError):
            coordinate.setflags(write=True)

    jd = grid.to_jd()
    mjd = grid.to_mjd()
    astropy_time = grid.as_astropy()
    jd[0] = 0.0
    mjd[0] = 0.0
    astropy_time[0] = Time("2025-01-02")

    assert grid.to_jd()[0] != 0.0
    assert grid.to_mjd()[0] != 0.0
    assert grid.start_time_iso.startswith("2025-01-01T00:00:00")


def test_time_grid_preserves_two_part_utc_across_a_leap_second():
    grid = build_observation_time_grid(
        start_time="2016-12-31T23:59:59",
        duration_seconds=3.0,
        cadence_seconds=1.0,
    )
    rendered = tuple(grid.as_astropy().isot)

    assert rendered == (
        "2016-12-31T23:59:59.000",
        "2016-12-31T23:59:60.000",
        "2017-01-01T00:00:00.000",
    )
    assert np.array_equal(
        grid.to_jd(),
        np.asarray(grid.as_astropy().jd, dtype=np.float64),
    )
    assert np.any(grid.utc_jd2 != 0.0)


def test_time_grid_rejects_subclasses_and_invalid_inputs():
    with pytest.raises(TypeError, match="factory"):
        ObservationTimeGrid()

    with pytest.raises(TypeError):

        class MutableTimeGrid(ObservationTimeGrid):
            pass

    for field, value in (
        ("duration_seconds", math.nan),
        ("duration_seconds", 0.0),
        ("cadence_seconds", math.inf),
        ("cadence_seconds", 0.0),
    ):
        arguments = {
            "start_time": "2025-01-01",
            "duration_seconds": 1.0,
            "cadence_seconds": 1.0,
        }
        arguments[field] = value
        with pytest.raises(InvalidTimeGridError):
            build_observation_time_grid(**arguments)

    with pytest.raises(InvalidTimeGridError):
        build_observation_time_grid(
            start_time="not-a-time",
            duration_seconds=1.0,
            cadence_seconds=1.0,
        )


def test_time_grid_checks_the_limit_before_allocation():
    with pytest.raises(TimeGridLimitError) as caught:
        build_observation_time_grid(
            start_time="2025-01-01",
            duration_seconds=float(MAX_TIME_SAMPLES + 1),
            cadence_seconds=1.0,
        )

    assert caught.value.requested_count == MAX_TIME_SAMPLES + 1
    assert caught.value.limit == MAX_TIME_SAMPLES
