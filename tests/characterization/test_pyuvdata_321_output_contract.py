"""Characterize pinned pyuvdata 3.2.1 standard-output behavior."""

from __future__ import annotations

import inspect
import shutil
import sys
import warnings
from pathlib import Path

import astropy
import casacore
import h5py
import numpy as np
import pytest
import pyuvdata
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import GCRS, EarthLocation
from astropy.time import Time
from pyuvdata import Telescope, UVData
from pyuvdata.utils import ECEF_from_ENU
from pyuvdata.uvdata.ms import MS
from pyuvdata.uvdata.uvfits import UVFITS

FREQUENCIES_HZ = np.array([100_000_000.0, 101_500_000.0], dtype=np.float64)
CHANNEL_WIDTHS_HZ = np.array([1_500_000.0, 1_500_000.0], dtype=np.float64)
TIMES = Time(["2025-01-01T00:00:00", "2025-01-01T00:00:02"])
ANTENNA_PAIRS = [(0, 0), (0, 2)]
CANONICAL_CODES = np.array([-5, -7, -8, -6], dtype=np.int64)
FILE_CODES = np.array([-5, -6, -7, -8], dtype=np.int64)
UNCALIBRATED_UNIT_WARNING = (
    "Writing in the MS file that the units of the data are uncalib, although "
    "some CASA process will ignore this and assume the units are all in Jy "
    "(or may not know how to handle data in these units)."
)
NUMPY_WHERE_WITHOUT_OUT_WARNING = (
    "'where' used without 'out', expect unitialized memory in output. "
    "If this is intentional, use out=None."
)


def _location_and_relative_ecef() -> tuple[EarthLocation, np.ndarray]:
    location = EarthLocation.from_geodetic(
        21.4 * u.deg,
        -30.7 * u.deg,
        1000.0 * u.m,
    )
    positions_enu = np.array(
        [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 30.0, 0.0]],
        dtype=np.float64,
    )
    center = np.array(
        [coordinate.to_value(u.m) for coordinate in location.geocentric],
        dtype=np.float64,
    )
    relative_ecef = ECEF_from_ENU(positions_enu, center_loc=location) - center
    return location, relative_ecef


def _telescope() -> Telescope:
    location, relative_ecef = _location_and_relative_ecef()
    return Telescope.new(
        name="Tier4DependencyArray",
        instrument="Tier4DependencyArray",
        location=location,
        antenna_positions={number: relative_ecef[number] for number in range(3)},
        antenna_names=["A0", "A1", "A2"],
        antenna_numbers=[0, 1, 2],
        antenna_diameters=[14.0, 15.0, 16.0],
        x_orientation="east",
        feeds=["x", "y"],
        mount_type="fixed",
        update_from_known=False,
    )


def _deterministic_data(
    dtype: type[np.complexfloating],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (4, 2, 4)
    real = np.arange(1, np.prod(shape) + 1, dtype=np.float64).reshape(shape)
    imaginary = (real * 0.123456789) + 0.03125
    data = (real + 1j * imaginary).astype(dtype)

    # Parallel-hand autos must be real. Cross-hand autos are conjugate pairs.
    for row in (0, 2):
        data[row, :, 0] = data[row, :, 0].real
        data[row, :, 3] = data[row, :, 3].real
        data[row, :, 2] = np.conj(data[row, :, 1])

    flags = np.zeros(shape, dtype=bool)
    flags[3, 1, 2] = True
    nsamples = np.full(shape, 2.5, dtype=np.float64)
    nsamples[1, 0, 0] = 0.5
    return data, flags, nsamples


def _new_uvdata(
    dtype: type[np.complexfloating] = np.complex64,
    *,
    frequencies: np.ndarray = FREQUENCIES_HZ,
    channel_widths: np.ndarray = CHANNEL_WIDTHS_HZ,
    metadata_only: bool = False,
    vis_units: str = "Jy",
) -> UVData:
    data, flags, nsamples = _deterministic_data(dtype)
    kwargs: dict[str, object] = {}
    if not metadata_only:
        kwargs.update(
            data_array=data,
            flag_array=flags,
            nsample_array=nsamples,
        )
    return UVData.new(
        freq_array=np.asarray(frequencies, dtype=np.float64),
        polarization_array=["xx", "xy", "yx", "yy"],
        times=TIMES.jd,
        antpairs=ANTENNA_PAIRS,
        telescope=_telescope(),
        do_blt_outer=True,
        time_axis_faster_than_bls=False,
        update_telescope_from_known=False,
        integration_time=2.0,
        channel_width=np.asarray(channel_widths, dtype=np.float64),
        history="Offline deterministic Tier 4 dependency characterization.",
        vis_units=vis_units,
        **kwargs,
    )


def _normalize_and_project(uvdata: UVData) -> tuple[np.ndarray, np.ndarray]:
    original_data = np.array(uvdata.data_array, copy=True)
    original_uvw = np.array(uvdata.uvw_array, copy=True)
    uvdata.polarization_array = np.asarray(
        uvdata.polarization_array,
        dtype=np.int64,
    )
    uvdata.phase_to_time(TIMES[0])
    return original_data, original_uvw


def _catalog(uvdata: UVData) -> dict[str, object]:
    assert len(uvdata.phase_center_catalog) == 1
    return next(iter(uvdata.phase_center_catalog.values()))


def _classify_warnings(
    captured: list[warnings.WarningMessage],
) -> set[str]:
    known = {
        (UserWarning, UNCALIBRATED_UNIT_WARNING): "uncalibrated-unit",
        (UserWarning, NUMPY_WHERE_WITHOUT_OUT_WARNING): "numpy-where-without-out",
    }
    categories: set[str] = set()
    unknown: list[str] = []
    for warning in captured:
        message = str(warning.message)
        category = known.get((warning.category, message))
        if category is not None:
            categories.add(category)
        else:
            unknown.append(f"{warning.category.__name__}: {message}")
    assert unknown == [], f"Unclassified warnings: {unknown}"
    return categories


def _write_ms_with_classified_warnings(
    uvdata: UVData,
    path: Path,
    *,
    clobber: bool = False,
) -> set[str]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            uvdata.write_ms(
                str(path),
                clobber=clobber,
                force_phase=False,
            )
        finally:
            categories = _classify_warnings(captured)
    return categories


def _write_uvfits_with_classified_warnings(
    uvdata: UVData,
    path: Path,
) -> set[str]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            uvdata.write_uvfits(str(path), force_phase=False)
        finally:
            categories = _classify_warnings(captured)
    return categories


def _expected_ms_warning_categories(*, uncalibrated: bool = False) -> set[str]:
    categories = {"uncalibrated-unit"} if uncalibrated else set()
    if sys.version_info[:2] == (3, 12):
        categories.add("numpy-where-without-out")
    return categories


def _assert_common_round_trip(expected: UVData, actual: UVData) -> None:
    np.testing.assert_array_equal(actual.ant_1_array, expected.ant_1_array)
    np.testing.assert_array_equal(actual.ant_2_array, expected.ant_2_array)
    np.testing.assert_allclose(actual.time_array, expected.time_array, atol=5e-10)
    np.testing.assert_allclose(
        actual.integration_time,
        expected.integration_time,
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        actual.freq_array,
        expected.freq_array,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        actual.channel_width,
        expected.channel_width,
        rtol=0.0,
        atol=1e-9,
    )
    assert actual.get_antpairs() == ANTENNA_PAIRS
    assert actual.telescope.antenna_numbers.tolist() == [0, 1, 2]
    assert list(actual.telescope.antenna_names) == ["A0", "A1", "A2"]
    np.testing.assert_allclose(actual.uvw_array, expected.uvw_array, atol=1e-6)
    assert _catalog(actual)["cat_type"] == "sidereal"
    assert _catalog(actual)["cat_frame"] == "icrs"
    np.testing.assert_allclose(
        [
            _catalog(actual)["cat_lon"],
            _catalog(actual)["cat_lat"],
        ],
        [
            _catalog(expected)["cat_lon"],
            _catalog(expected)["cat_lat"],
        ],
        rtol=0.0,
        atol=1e-12,
    )

    expected_codes = np.asarray(expected.polarization_array)
    actual_codes = np.asarray(actual.polarization_array)
    assert actual_codes.tolist() == FILE_CODES.tolist()
    for code in FILE_CODES:
        expected_index = int(np.flatnonzero(expected_codes == code)[0])
        actual_index = int(np.flatnonzero(actual_codes == code)[0])
        np.testing.assert_array_equal(
            actual.flag_array[:, :, actual_index],
            expected.flag_array[:, :, expected_index],
        )
        np.testing.assert_allclose(
            actual.nsample_array[:, :, actual_index],
            expected.nsample_array[:, :, expected_index],
            rtol=5e-6,
            atol=1e-7,
        )


def _actual_data_in_canonical_order(actual: UVData) -> np.ndarray:
    indices = [
        int(np.flatnonzero(np.asarray(actual.polarization_array) == code)[0])
        for code in CANONICAL_CODES
    ]
    return actual.data_array[:, :, indices]


def test_dependency_identity_and_public_signatures_match_locked_matrix() -> None:
    """Characterizes exact installed identities without network metadata."""
    expected = {
        (3, 11): {
            "python": "3.11.13",
            "numpy": "2.3.2",
            "astropy": "7.1.0",
            "casacore": "3.7.1",
            "h5py": "3.14.0",
        },
        (3, 12): {
            "python": "3.12.13",
            "numpy": "2.4.6",
            "astropy": "8.0.1",
            "casacore": "3.8.1",
            "h5py": "3.16.0",
        },
    }[sys.version_info[:2]]
    assert pyuvdata.__version__ == "3.2.1"
    assert ".".join(map(str, sys.version_info[:3])) == expected["python"]
    assert np.__version__ == expected["numpy"]
    assert astropy.__version__ == expected["astropy"]
    assert casacore.__version__ == expected["casacore"]
    assert h5py.__version__ == expected["h5py"]

    telescope_parameters = inspect.signature(Telescope.new).parameters
    assert {"location", "antenna_positions", "update_from_known"} <= set(
        telescope_parameters
    )
    phase_parameters = inspect.signature(UVData.phase_to_time).parameters
    assert {"self", "time", "phase_frame", "use_ant_pos"} <= set(phase_parameters)
    assert "clobber" in inspect.signature(MS.write_ms).parameters
    assert "force_phase" in inspect.signature(MS.write_ms).parameters
    assert "clobber" not in inspect.signature(UVFITS.write_uvfits).parameters
    assert "force_phase" in inspect.signature(UVFITS.write_uvfits).parameters


def test_dependency_uvdata_new_preserves_order_identity_and_coordinates() -> None:
    """Characterizes UVData.new ordering and initial unprojected state."""
    uvdata = _new_uvdata()

    assert isinstance(uvdata.polarization_array, list)
    assert uvdata.polarization_array == CANONICAL_CODES.tolist()
    np.testing.assert_array_equal(uvdata.ant_1_array, [0, 0, 0, 0])
    np.testing.assert_array_equal(uvdata.ant_2_array, [0, 2, 0, 2])
    np.testing.assert_allclose(
        uvdata.time_array,
        np.repeat(TIMES.jd, 2),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(uvdata.integration_time, np.full(4, 2.0))
    np.testing.assert_array_equal(uvdata.freq_array, FREQUENCIES_HZ)
    np.testing.assert_array_equal(uvdata.channel_width, CHANNEL_WIDTHS_HZ)
    assert uvdata.get_antpairs() == ANTENNA_PAIRS
    assert uvdata.telescope.antenna_numbers.tolist() == [0, 1, 2]
    assert uvdata.telescope.antenna_names.tolist() == ["A0", "A1", "A2"]
    assert np.all(np.isfinite(uvdata.uvw_array))
    assert _catalog(uvdata)["cat_type"] == "unprojected"
    assert uvdata.flag_array[3, 1, 2]
    assert uvdata.nsample_array[1, 0, 0] == 0.5
    assert uvdata.check() is True


def _independent_projected_uvw(uvdata: UVData) -> np.ndarray:
    location, relative_ecef = _location_and_relative_ecef()
    catalog = _catalog(uvdata)
    ra = float(catalog["cat_lon"])
    dec = float(catalog["cat_lat"])
    u_axis = np.array([-np.sin(ra), np.cos(ra), 0.0])
    v_axis = np.array(
        [
            -np.sin(dec) * np.cos(ra),
            -np.sin(dec) * np.sin(ra),
            np.cos(dec),
        ]
    )
    w_axis = np.array(
        [
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ]
    )
    center = np.array(
        [coordinate.to_value(u.m) for coordinate in location.geocentric],
        dtype=np.float64,
    )
    rows: list[np.ndarray] = []
    for sample_time in TIMES:
        for antenna1, antenna2 in ANTENNA_PAIRS:
            gcrs_positions = []
            for antenna in (antenna1, antenna2):
                xyz = center + relative_ecef[antenna]
                antenna_location = EarthLocation.from_geocentric(*xyz, unit="m")
                gcrs = antenna_location.get_itrs(obstime=sample_time).transform_to(
                    GCRS(obstime=sample_time)
                )
                gcrs_positions.append(gcrs.cartesian.xyz.to_value(u.m))
            baseline = gcrs_positions[1] - gcrs_positions[0]
            rows.append(
                np.array(
                    [
                        np.dot(u_axis, baseline),
                        np.dot(v_axis, baseline),
                        np.dot(w_axis, baseline),
                    ]
                )
            )
    return np.stack(rows)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_dependency_explicit_phase_projection_matches_independent_calculation(
    tmp_path: Path,
    dtype: type[np.complexfloating],
) -> None:
    """Characterizes explicit projection; hidden force phasing is not target."""
    unprojected = _new_uvdata(dtype)
    with pytest.raises(ValueError, match="unprojected|phase"):
        unprojected.write_ms(
            str(tmp_path / "unprojected.ms"),
            force_phase=False,
        )
    with pytest.raises(ValueError, match="unprojected|phase"):
        unprojected.write_uvfits(
            str(tmp_path / "unprojected.uvfits"),
            force_phase=False,
        )
    assert not (tmp_path / "unprojected.ms").exists()
    assert not (tmp_path / "unprojected.uvfits").exists()

    original_data, original_uvw = _normalize_and_project(unprojected)
    assert _catalog(unprojected)["cat_type"] == "sidereal"
    assert _catalog(unprojected)["cat_frame"] == "icrs"
    assert unprojected.check() is True

    # GCRS/ICRS tangent projection independently reproduces UVW to a
    # millimetre; the bound covers Astropy apparent-frame convention details.
    np.testing.assert_allclose(
        unprojected.uvw_array,
        _independent_projected_uvw(unprojected),
        rtol=0.0,
        atol=1e-3,
    )
    delta_w = unprojected.uvw_array[:, 2] - original_uvw[:, 2]
    expected_phase = np.exp(
        -2j
        * np.pi
        * delta_w[:, np.newaxis, np.newaxis]
        * FREQUENCIES_HZ[np.newaxis, :, np.newaxis]
        / c.value
    )
    phase_tolerance = 2e-6 if dtype is np.complex64 else 5e-8
    np.testing.assert_allclose(
        unprojected.data_array,
        original_data * expected_phase,
        rtol=phase_tolerance,
        atol=phase_tolerance,
    )

    _write_ms_with_classified_warnings(
        unprojected,
        tmp_path / f"projected-{np.dtype(dtype).name}.ms",
    )


def test_dependency_polarization_list_affects_only_uvfits_until_normalized(
    tmp_path: Path,
) -> None:
    """Characterizes the asymmetric MS and UVFITS list-write boundaries."""
    ms_data = _new_uvdata()
    assert isinstance(ms_data.polarization_array, list)
    ms_data.phase_to_time(TIMES[0])
    ms_path = tmp_path / "list-polarizations.ms"
    categories = _write_ms_with_classified_warnings(ms_data, ms_path)
    assert categories == _expected_ms_warning_categories()
    assert ms_path.is_dir()
    assert isinstance(ms_data.polarization_array, list)
    shutil.rmtree(ms_path)

    uvdata = _new_uvdata()
    assert isinstance(uvdata.polarization_array, list)
    uvdata.phase_to_time(TIMES[0])

    with pytest.raises(TypeError, match="integer scalar"):
        uvdata.write_uvfits(
            str(tmp_path / "list-polarizations.uvfits"),
            force_phase=False,
        )

    uvdata.polarization_array = np.asarray(
        uvdata.polarization_array,
        dtype=np.int64,
    )
    assert isinstance(uvdata.polarization_array, np.ndarray)
    np.testing.assert_array_equal(uvdata.polarization_array, CANONICAL_CODES)
    _write_uvfits_with_classified_warnings(
        uvdata,
        tmp_path / "normalized-polarizations.uvfits",
    )

    malformed = uvdata.copy()
    malformed.polarization_array = np.array([-5, -7, -8], dtype=np.int64)
    with pytest.raises(ValueError, match="polarization_array"):
        malformed.check()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_dependency_measurement_set_round_trip_characterizes_dtype(
    tmp_path: Path,
    dtype: type[np.complexfloating],
) -> None:
    """Characterizes current MS precision; complex128 is observably lossy."""
    expected = _new_uvdata(dtype)
    _normalize_and_project(expected)
    path = tmp_path / f"round-trip-{np.dtype(dtype).name}.ms"

    categories = _write_ms_with_classified_warnings(expected, path)
    assert categories == _expected_ms_warning_categories()
    actual = UVData()
    actual.read_ms(str(path))
    _assert_common_round_trip(expected, actual)
    assert actual.data_array.dtype == np.dtype(np.complex64)

    canonical_actual = _actual_data_in_canonical_order(actual)
    expected_stored = expected.data_array.astype(np.complex64)
    error = np.abs(canonical_actual - expected_stored)
    assert np.all(np.isfinite(error))
    bound = (
        4
        * np.finfo(np.float32).eps
        * max(1.0, float(np.max(np.abs(expected.data_array))))
    )
    assert float(np.max(error)) <= bound
    if dtype is np.complex64:
        np.testing.assert_array_equal(canonical_actual, expected.data_array)
    else:
        assert expected.data_array.dtype == np.dtype(np.complex128)
        assert actual.data_array.dtype != expected.data_array.dtype

    shutil.rmtree(path)
    assert not path.exists()


def test_dependency_measurement_set_collision_replacement_and_unit_warning(
    tmp_path: Path,
) -> None:
    """Characterizes clobber behavior, warning class, and closed handles."""
    first = _new_uvdata(vis_units="uncalib")
    _normalize_and_project(first)
    path = tmp_path / "collision.ms"

    categories = _write_ms_with_classified_warnings(first, path)
    assert categories == _expected_ms_warning_categories(uncalibrated=True)
    with pytest.raises(OSError):
        _write_ms_with_classified_warnings(first, path, clobber=False)

    replacement = first.copy()
    replacement.data_array = replacement.data_array + np.float32(50.0)
    categories = _write_ms_with_classified_warnings(replacement, path, clobber=True)
    assert categories == _expected_ms_warning_categories(uncalibrated=True)
    actual = UVData()
    actual.read_ms(str(path))
    np.testing.assert_allclose(
        _actual_data_in_canonical_order(actual),
        replacement.data_array.astype(np.complex64),
        rtol=5e-6,
        atol=1e-7,
    )
    shutil.rmtree(path)
    assert not path.exists()

    fabricated = warnings.WarningMessage(
        UserWarning("somewhere arbitrary output changed"),
        UserWarning,
        "probe.py",
        1,
    )
    with pytest.raises(AssertionError, match="Unclassified warnings"):
        _classify_warnings([fabricated])

    wrong_category = warnings.WarningMessage(
        RuntimeWarning(NUMPY_WHERE_WITHOUT_OUT_WARNING),
        RuntimeWarning,
        "probe.py",
        1,
    )
    with pytest.raises(AssertionError, match="Unclassified warnings"):
        _classify_warnings([wrong_category])


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_dependency_uvfits_round_trip_preserves_supported_dtype(
    tmp_path: Path,
    dtype: type[np.complexfloating],
) -> None:
    """Characterizes supported UVFITS round trips, not replacement policy."""
    expected = _new_uvdata(dtype)
    _normalize_and_project(expected)
    path = tmp_path / f"round-trip-{np.dtype(dtype).name}.uvfits"

    categories = _write_uvfits_with_classified_warnings(expected, path)
    assert categories == set()
    actual = UVData()
    actual.read_uvfits(str(path))
    _assert_common_round_trip(expected, actual)
    assert actual.data_array.dtype == np.dtype(dtype)
    tolerance = 2e-6 if dtype is np.complex64 else 5e-13
    np.testing.assert_allclose(
        _actual_data_in_canonical_order(actual),
        expected.data_array,
        rtol=tolerance,
        atol=tolerance,
    )
    path.unlink()
    assert not path.exists()


def test_dependency_uvfits_has_no_clobber_keyword(tmp_path: Path) -> None:
    """Characterizes why future replacement belongs to RadioSim publication."""
    uvdata = _new_uvdata()
    _normalize_and_project(uvdata)
    with pytest.raises(TypeError, match="clobber"):
        uvdata.write_uvfits(
            str(tmp_path / "no-clobber.uvfits"),
            force_phase=False,
            clobber=False,
        )
    assert not (tmp_path / "no-clobber.uvfits").exists()


def test_dependency_uvfits_rejects_metadata_only_and_invalid_polarizations(
    tmp_path: Path,
) -> None:
    """Characterizes pinned UVFITS representability failures."""
    metadata_only = _new_uvdata(metadata_only=True)
    metadata_only.polarization_array = np.asarray(
        metadata_only.polarization_array,
        dtype=np.int64,
    )
    metadata_only.phase_to_time(TIMES[0])
    with pytest.raises(ValueError, match="data"):
        metadata_only.write_uvfits(
            str(tmp_path / "metadata-only.uvfits"),
            force_phase=False,
        )

    malformed = _new_uvdata()
    _normalize_and_project(malformed)
    malformed.data_array[[0, 2]] = malformed.data_array[[0, 2]].real
    malformed.polarization_array = np.array([-5, -7, -8, -12], dtype=np.int64)
    with pytest.raises(ValueError, match="polarization|evenly spaced|not conform"):
        malformed.write_uvfits(
            str(tmp_path / "malformed-polarization.uvfits"),
            force_phase=False,
        )


def test_dependency_uvfits_requires_spacing_equal_to_width(
    tmp_path: Path,
) -> None:
    """Characterizes the pinned equality rule, not a weaker overlap rule."""
    unequal = _new_uvdata(
        frequencies=np.array([100e6, 101e6], dtype=np.float64),
        channel_widths=np.array([2e6, 2e6], dtype=np.float64),
    )
    _normalize_and_project(unequal)
    with pytest.raises(ValueError, match="spacing|channel width"):
        unequal.write_uvfits(
            str(tmp_path / "spacing-less-than-width.uvfits"),
            force_phase=False,
        )

    malformed_width = _new_uvdata()
    malformed_width.channel_width = np.array([1.5e6], dtype=np.float64)
    with pytest.raises(ValueError, match="channel_width"):
        malformed_width.check()
