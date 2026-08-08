"""SCI-006 acceptance across the result and standard-output boundaries."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
from pyuvdata import UVData

from radiosim.io.hdf5 import load_result_hdf5, write_result_hdf5
from radiosim.io.measurement_set import read_measurement_set, write_measurement_set
from radiosim.io.standard_visibility import project_simulation_result
from radiosim.io.uvfits import read_uvfits, write_uvfits
from tests.unit.test_io.test_standard_visibility import build_standard_result


def _assert_corrected_linear_products(values: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(values, expected, rtol=5e-6, atol=1e-7)


def test_corrected_east_x_products_and_metadata_survive_every_output(
    tmp_path: Path,
) -> None:
    """Result, HDF5, UVFITS, and MS preserve values, labels, feeds, and angles."""
    # I=2, Q=0.6, U=0.4, V=-0.2 in canonical sky (North, East).  SCI-006's
    # east-X reporting matrix gives [[(I-Q)/2, (U-iV)/2], ...].
    corrected = np.array(
        [[0.7, 0.2 + 0.1j], [0.2 - 0.1j, 1.3]],
        dtype=np.complex128,
    )
    expected_flat = corrected.reshape(4)
    result = build_standard_result(tmp_path / "result", receptor_matrix=corrected)
    expected_result = np.broadcast_to(expected_flat, result.visibilities.shape)
    antenna_count = len(result.instrument.antennas)
    expected_feeds = [["x", "y"]] * antenna_count
    expected_angles = np.tile([math.pi / 2.0, 0.0], (antenna_count, 1))

    assert result.correlations == ("XX", "XY", "YX", "YY")
    assert result.polarization_basis == "linear_xy"
    np.testing.assert_allclose(result.visibilities, expected_result, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result.visibilities[..., 0] - result.visibilities[..., 3],
        -0.6,
        rtol=0.0,
        atol=2e-16,
    )
    for receptor in result.receptors.receptor_by_antenna.values():
        assert receptor.feed_array == ("x", "y")
        assert receptor.feed_angle_rad == (math.pi / 2.0, 0.0)

    hdf5_path = write_result_hdf5(result, tmp_path / "sci006.h5")
    loaded_hdf5 = load_result_hdf5(hdf5_path)
    _assert_corrected_linear_products(loaded_hdf5.visibilities, expected_result)
    np.testing.assert_allclose(
        loaded_hdf5.visibilities[..., 0] - loaded_hdf5.visibilities[..., 3],
        -0.6,
        rtol=0.0,
        atol=2e-16,
    )
    assert loaded_hdf5.correlations == result.correlations
    with h5py.File(hdf5_path, "r") as handle:
        np.testing.assert_array_equal(
            handle["coordinates/correlation/labels"][:],
            np.array([b"XX", b"XY", b"YX", b"YY"]),
        )
        np.testing.assert_allclose(
            handle["receptors/feed_angle_rad"][:],
            expected_angles,
            rtol=0.0,
            atol=0.0,
        )

    uvfits_path = write_uvfits(result, tmp_path / "sci006.uvfits")
    loaded_uvfits = read_uvfits(uvfits_path)
    expected_uvfits = project_simulation_result(result, format="uvfits").data
    _assert_corrected_linear_products(
        loaded_uvfits.visibilities, expected_uvfits.visibilities
    )
    assert loaded_uvfits.correlations == result.correlations
    raw_uvfits = UVData()
    raw_uvfits.read_uvfits(str(uvfits_path))
    assert raw_uvfits.telescope.feed_array.tolist() == expected_feeds
    np.testing.assert_allclose(
        raw_uvfits.telescope.feed_angle,
        expected_angles,
        rtol=0.0,
        atol=5e-8,
    )

    ms_path = write_measurement_set(result, tmp_path / "sci006.ms")
    loaded_ms = read_measurement_set(ms_path)
    expected_ms = project_simulation_result(result, format="ms").data
    _assert_corrected_linear_products(loaded_ms.visibilities, expected_ms.visibilities)
    assert loaded_ms.correlations == result.correlations
    raw_ms = UVData()
    raw_ms.read_ms(str(ms_path))
    assert raw_ms.telescope.feed_array.tolist() == expected_feeds
    np.testing.assert_allclose(
        raw_ms.telescope.feed_angle,
        expected_angles,
        rtol=0.0,
        atol=1e-9,
    )
