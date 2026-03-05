# tests/unit/test_io/test_writers.py
"""
Unit tests for I/O writers (HDF5, YAML).

Tests the save and load functions for visibility data and configurations.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from rrivis.io.writers import (
    load_visibilities_hdf5,
    save_config_yaml,
    save_visibilities_hdf5,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_visibilities():
    """Generate sample visibility data for testing."""
    np.random.seed(42)
    n_times = 5
    n_freqs = 10

    visibilities = {
        (0, 0): [
            np.random.randn(n_freqs) + 1j * np.random.randn(n_freqs)
            for _ in range(n_times)
        ],
        (0, 1): [
            np.random.randn(n_freqs) + 1j * np.random.randn(n_freqs)
            for _ in range(n_times)
        ],
        (1, 1): [
            np.random.randn(n_freqs) + 1j * np.random.randn(n_freqs)
            for _ in range(n_times)
        ],
    }

    return visibilities


@pytest.fixture
def sample_frequencies():
    """Generate sample frequency array."""
    return np.linspace(100e6, 200e6, 10)  # 100-200 MHz


@pytest.fixture
def sample_time_points():
    """Generate sample time points array."""
    return np.linspace(59000.0, 59001.0, 5)  # MJD


@pytest.fixture
def sample_metadata():
    """Generate sample metadata dictionary."""
    return {
        "version": "0.2.0",
        "backend": "numpy",
        "n_antennas": 3,
        "n_baselines": 3,
    }


@pytest.fixture
def sample_config():
    """Generate sample configuration dictionary."""
    return {
        "telescope": {
            "telescope_name": "HERA",
            "use_pyuvdata_telescope": False,
        },
        "antenna_layout": {
            "all_antenna_diameter": 14.0,
            "antenna_file_format": "rrivis",
        },
        "obs_frequency": {
            "starting_frequency": 100.0,
            "frequency_bandwidth": 50.0,
            "frequency_unit": "MHz",
        },
    }


# =============================================================================
# HDF5 Writer Tests
# =============================================================================


class TestSaveVisibilitiesHDF5:
    """Test save_visibilities_hdf5 function."""

    def test_creates_file(
        self, tmp_path, sample_visibilities, sample_frequencies, sample_time_points
    ):
        """Test that HDF5 file is created."""
        output_path = tmp_path / "test_vis.h5"

        result = save_visibilities_hdf5(
            output_path,
            sample_visibilities,
            sample_frequencies,
            sample_time_points,
        )

        assert Path(result).exists()
        assert output_path.exists()

    def test_creates_parent_directories(
        self, tmp_path, sample_visibilities, sample_frequencies, sample_time_points
    ):
        """Test that parent directories are created."""
        output_path = tmp_path / "subdir" / "nested" / "test_vis.h5"

        save_visibilities_hdf5(
            output_path,
            sample_visibilities,
            sample_frequencies,
            sample_time_points,
        )

        assert output_path.exists()

    def test_stores_baselines(
        self, tmp_path, sample_visibilities, sample_frequencies, sample_time_points
    ):
        """Test that baseline groups are created."""
        output_path = tmp_path / "test_vis.h5"

        save_visibilities_hdf5(
            output_path,
            sample_visibilities,
            sample_frequencies,
            sample_time_points,
        )

        with h5py.File(output_path, "r") as h5file:
            baseline_groups = [k for k in h5file.keys() if k.startswith("baseline_")]
            assert len(baseline_groups) == 3

    def test_stores_visibility_data(
        self, tmp_path, sample_visibilities, sample_frequencies, sample_time_points
    ):
        """Test that visibility data is stored correctly."""
        output_path = tmp_path / "test_vis.h5"

        save_visibilities_hdf5(
            output_path,
            sample_visibilities,
            sample_frequencies,
            sample_time_points,
        )

        with h5py.File(output_path, "r") as h5file:
            for key in h5file.keys():
                if key.startswith("baseline_"):
                    assert "complex_visibility" in h5file[key]
                    vis_data = h5file[key]["complex_visibility"][:]
                    assert vis_data.dtype == np.complex128

    def test_stores_frequencies(
        self, tmp_path, sample_visibilities, sample_frequencies, sample_time_points
    ):
        """Test that frequencies are stored."""
        output_path = tmp_path / "test_vis.h5"

        save_visibilities_hdf5(
            output_path,
            sample_visibilities,
            sample_frequencies,
            sample_time_points,
        )

        with h5py.File(output_path, "r") as h5file:
            assert "frequencies" in h5file
            np.testing.assert_allclose(h5file["frequencies"][:], sample_frequencies)

    def test_stores_time_points(
        self, tmp_path, sample_visibilities, sample_frequencies, sample_time_points
    ):
        """Test that time points are stored."""
        output_path = tmp_path / "test_vis.h5"

        save_visibilities_hdf5(
            output_path,
            sample_visibilities,
            sample_frequencies,
            sample_time_points,
        )

        with h5py.File(output_path, "r") as h5file:
            assert "time_points_mjd" in h5file
            np.testing.assert_allclose(h5file["time_points_mjd"][:], sample_time_points)

    def test_stores_metadata(
        self,
        tmp_path,
        sample_visibilities,
        sample_frequencies,
        sample_time_points,
        sample_metadata,
    ):
        """Test that metadata is stored as attributes."""
        output_path = tmp_path / "test_vis.h5"

        save_visibilities_hdf5(
            output_path,
            sample_visibilities,
            sample_frequencies,
            sample_time_points,
            metadata=sample_metadata,
        )

        with h5py.File(output_path, "r") as h5file:
            for key, value in sample_metadata.items():
                assert key in h5file.attrs
                assert h5file.attrs[key] == value

    def test_handles_empty_metadata(
        self, tmp_path, sample_visibilities, sample_frequencies, sample_time_points
    ):
        """Test handling of None metadata."""
        output_path = tmp_path / "test_vis.h5"

        # Should not raise error
        save_visibilities_hdf5(
            output_path,
            sample_visibilities,
            sample_frequencies,
            sample_time_points,
            metadata=None,
        )

        assert output_path.exists()


# =============================================================================
# HDF5 Reader Tests
# =============================================================================


class TestLoadVisibilitiesHDF5:
    """Test load_visibilities_hdf5 function."""

    def test_loads_visibilities(
        self, tmp_path, sample_visibilities, sample_frequencies, sample_time_points
    ):
        """Test loading visibility data."""
        output_path = tmp_path / "test_vis.h5"

        save_visibilities_hdf5(
            output_path,
            sample_visibilities,
            sample_frequencies,
            sample_time_points,
        )

        result = load_visibilities_hdf5(output_path)

        assert "visibilities" in result
        assert len(result["visibilities"]) == len(sample_visibilities)

    def test_loads_frequencies(
        self, tmp_path, sample_visibilities, sample_frequencies, sample_time_points
    ):
        """Test loading frequency data."""
        output_path = tmp_path / "test_vis.h5"

        save_visibilities_hdf5(
            output_path,
            sample_visibilities,
            sample_frequencies,
            sample_time_points,
        )

        result = load_visibilities_hdf5(output_path)

        assert "frequencies" in result
        np.testing.assert_allclose(result["frequencies"], sample_frequencies)

    def test_loads_time_points(
        self, tmp_path, sample_visibilities, sample_frequencies, sample_time_points
    ):
        """Test loading time points."""
        output_path = tmp_path / "test_vis.h5"

        save_visibilities_hdf5(
            output_path,
            sample_visibilities,
            sample_frequencies,
            sample_time_points,
        )

        result = load_visibilities_hdf5(output_path)

        assert "time_points_mjd" in result
        np.testing.assert_allclose(result["time_points_mjd"], sample_time_points)

    def test_loads_metadata(
        self,
        tmp_path,
        sample_visibilities,
        sample_frequencies,
        sample_time_points,
        sample_metadata,
    ):
        """Test loading metadata."""
        output_path = tmp_path / "test_vis.h5"

        save_visibilities_hdf5(
            output_path,
            sample_visibilities,
            sample_frequencies,
            sample_time_points,
            metadata=sample_metadata,
        )

        result = load_visibilities_hdf5(output_path)

        assert "metadata" in result
        for key in sample_metadata:
            assert key in result["metadata"]

    def test_roundtrip_visibility_data(
        self, tmp_path, sample_visibilities, sample_frequencies, sample_time_points
    ):
        """Test that visibility data survives save/load roundtrip."""
        output_path = tmp_path / "test_vis.h5"

        save_visibilities_hdf5(
            output_path,
            sample_visibilities,
            sample_frequencies,
            sample_time_points,
        )

        result = load_visibilities_hdf5(output_path)

        # Check that data matches
        for baseline, expected_vis_list in sample_visibilities.items():
            expected_array = np.stack(expected_vis_list)
            loaded_array = result["visibilities"][baseline]
            np.testing.assert_allclose(loaded_array, expected_array)


# =============================================================================
# YAML Writer Tests
# =============================================================================


class TestSaveConfigYAML:
    """Test save_config_yaml function."""

    def test_creates_file(self, tmp_path, sample_config):
        """Test that YAML file is created."""
        output_path = tmp_path / "test_config.yaml"

        result = save_config_yaml(sample_config, output_path)

        assert Path(result).exists()
        assert output_path.exists()

    def test_creates_parent_directories(self, tmp_path, sample_config):
        """Test that parent directories are created."""
        output_path = tmp_path / "subdir" / "nested" / "config.yaml"

        save_config_yaml(sample_config, output_path)

        assert output_path.exists()

    def test_valid_yaml_output(self, tmp_path, sample_config):
        """Test that output is valid YAML."""
        output_path = tmp_path / "test_config.yaml"

        save_config_yaml(sample_config, output_path)

        # Should be parseable as YAML
        with open(output_path) as f:
            loaded = yaml.safe_load(f)

        assert isinstance(loaded, dict)

    def test_roundtrip_config(self, tmp_path, sample_config):
        """Test that config survives save/load roundtrip."""
        output_path = tmp_path / "test_config.yaml"

        save_config_yaml(sample_config, output_path)

        with open(output_path) as f:
            loaded = yaml.safe_load(f)

        # Deep comparison
        assert (
            loaded["telescope"]["telescope_name"]
            == sample_config["telescope"]["telescope_name"]
        )
        assert (
            loaded["antenna_layout"]["all_antenna_diameter"]
            == sample_config["antenna_layout"]["all_antenna_diameter"]
        )
        assert (
            loaded["obs_frequency"]["starting_frequency"]
            == sample_config["obs_frequency"]["starting_frequency"]
        )

    def test_handles_nested_config(self, tmp_path):
        """Test handling of deeply nested configurations."""
        nested_config = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": 42,
                        "list": [1, 2, 3],
                    }
                }
            }
        }

        output_path = tmp_path / "nested_config.yaml"
        save_config_yaml(nested_config, output_path)

        with open(output_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["level1"]["level2"]["level3"]["value"] == 42
        assert loaded["level1"]["level2"]["level3"]["list"] == [1, 2, 3]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_visibilities(self, tmp_path, sample_frequencies, sample_time_points):
        """Test handling of empty visibility dictionary."""
        output_path = tmp_path / "empty_vis.h5"

        save_visibilities_hdf5(
            output_path,
            {},  # Empty visibilities
            sample_frequencies,
            sample_time_points,
        )

        result = load_visibilities_hdf5(output_path)
        assert len(result["visibilities"]) == 0

    def test_single_baseline(self, tmp_path, sample_frequencies, sample_time_points):
        """Test with single baseline."""
        np.random.seed(42)
        single_vis = {
            (0, 1): [np.random.randn(10) + 1j * np.random.randn(10) for _ in range(5)]
        }

        output_path = tmp_path / "single_bl.h5"

        save_visibilities_hdf5(
            output_path,
            single_vis,
            sample_frequencies,
            sample_time_points,
        )

        result = load_visibilities_hdf5(output_path)
        assert len(result["visibilities"]) == 1

    def test_large_array_handling(self, tmp_path):
        """Test handling of larger arrays."""
        np.random.seed(42)
        n_baselines = 50
        n_times = 10
        n_freqs = 100

        large_vis = {}
        for i in range(n_baselines):
            large_vis[(i, i + 1)] = [
                np.random.randn(n_freqs) + 1j * np.random.randn(n_freqs)
                for _ in range(n_times)
            ]

        frequencies = np.linspace(100e6, 200e6, n_freqs)
        time_points = np.linspace(59000.0, 59001.0, n_times)

        output_path = tmp_path / "large_vis.h5"

        save_visibilities_hdf5(output_path, large_vis, frequencies, time_points)

        result = load_visibilities_hdf5(output_path)
        assert len(result["visibilities"]) == n_baselines

    def test_special_characters_in_path(self, tmp_path):
        """Test handling of paths with special characters."""
        config = {"key": "value"}
        output_path = tmp_path / "test config (1).yaml"

        save_config_yaml(config, output_path)
        assert output_path.exists()
