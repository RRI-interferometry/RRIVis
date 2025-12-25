"""Tests for Measurement Set I/O functionality.

These tests verify the MS reading and writing capabilities.
Tests are skipped if python-casacore is not installed.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as u


# Check for MS dependencies
try:
    from rrivis.io.measurement_set import (
        write_ms,
        read_ms,
        ms_info,
        PYUVDATA_AVAILABLE,
        CASACORE_AVAILABLE,
    )
    MS_AVAILABLE = PYUVDATA_AVAILABLE and CASACORE_AVAILABLE
except ImportError:
    MS_AVAILABLE = False

# Skip all tests in this module if MS dependencies not available
pytestmark = pytest.mark.skipif(
    not MS_AVAILABLE,
    reason="python-casacore not available. Install with: pip install python-casacore"
)


@pytest.fixture
def sample_visibilities():
    """Create sample visibility data for testing."""
    # 3 antennas -> 6 baselines (including autos)
    visibilities = {}
    n_freqs = 10

    for ant1 in range(3):
        for ant2 in range(ant1, 3):
            # Create complex visibility data
            vis_xx = np.random.randn(n_freqs) + 1j * np.random.randn(n_freqs)
            vis_yy = np.random.randn(n_freqs) + 1j * np.random.randn(n_freqs)
            vis_xy = np.random.randn(n_freqs) * 0.1 + 1j * np.random.randn(n_freqs) * 0.1
            vis_yx = np.random.randn(n_freqs) * 0.1 + 1j * np.random.randn(n_freqs) * 0.1

            visibilities[(ant1, ant2)] = {
                "XX": vis_xx,
                "XY": vis_xy,
                "YX": vis_yx,
                "YY": vis_yy,
            }

    return visibilities


@pytest.fixture
def sample_antennas():
    """Create sample antenna data for testing."""
    return {
        "0000": {"Number": 0, "Position": [0.0, 0.0, 0.0], "diameter": 14.0},
        "0001": {"Number": 1, "Position": [10.0, 0.0, 0.0], "diameter": 14.0},
        "0002": {"Number": 2, "Position": [0.0, 10.0, 0.0], "diameter": 14.0},
    }


@pytest.fixture
def sample_baselines(sample_antennas):
    """Create sample baseline data for testing."""
    baselines = {}
    ant_nums = sorted([ant["Number"] for ant in sample_antennas.values()])

    for i, ant1 in enumerate(ant_nums):
        for ant2 in ant_nums[i:]:
            pos1 = np.array(sample_antennas[f"{ant1:04d}"]["Position"])
            pos2 = np.array(sample_antennas[f"{ant2:04d}"]["Position"])
            baseline_vector = pos2 - pos1

            baselines[(ant1, ant2)] = {
                "BaselineVector": baseline_vector,
                "Length": np.linalg.norm(baseline_vector),
            }

    return baselines


@pytest.fixture
def sample_frequencies():
    """Create sample frequency array for testing."""
    return np.linspace(100e6, 200e6, 10)  # 100-200 MHz, 10 channels


@pytest.fixture
def sample_location():
    """Create sample observatory location for testing."""
    # HERA location
    return EarthLocation(
        lat=-30.72152777777791 * u.deg,
        lon=21.428305555555557 * u.deg,
        height=1073.0 * u.m,
    )


@pytest.fixture
def sample_obstime():
    """Create sample observation time for testing."""
    return Time("2025-01-01T00:00:00", scale="utc")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    # Cleanup
    shutil.rmtree(tmpdir)


class TestWriteMS:
    """Tests for write_ms function."""

    def test_write_ms_basic(
        self,
        temp_dir,
        sample_visibilities,
        sample_frequencies,
        sample_antennas,
        sample_baselines,
        sample_location,
        sample_obstime,
    ):
        """Test basic MS writing functionality."""
        output_path = temp_dir / "test.ms"

        result = write_ms(
            output_path=output_path,
            visibilities=sample_visibilities,
            frequencies=sample_frequencies,
            antennas=sample_antennas,
            baselines=sample_baselines,
            location=sample_location,
            obstime=sample_obstime,
        )

        assert result.exists()
        assert result.is_dir()  # MS is a directory
        assert (result / "table.dat").exists()  # Main MS table

    def test_write_ms_with_telescope_name(
        self,
        temp_dir,
        sample_visibilities,
        sample_frequencies,
        sample_antennas,
        sample_baselines,
        sample_location,
        sample_obstime,
    ):
        """Test MS writing with custom telescope name."""
        output_path = temp_dir / "test_hera.ms"

        write_ms(
            output_path=output_path,
            visibilities=sample_visibilities,
            frequencies=sample_frequencies,
            antennas=sample_antennas,
            baselines=sample_baselines,
            location=sample_location,
            obstime=sample_obstime,
            telescope_name="HERA",
        )

        assert output_path.exists()

        # Verify telescope name in metadata
        info = ms_info(output_path)
        assert info["telescope_name"] == "HERA"

    def test_write_ms_overwrite(
        self,
        temp_dir,
        sample_visibilities,
        sample_frequencies,
        sample_antennas,
        sample_baselines,
        sample_location,
        sample_obstime,
    ):
        """Test MS overwrite behavior."""
        output_path = temp_dir / "test_overwrite.ms"

        # First write
        write_ms(
            output_path=output_path,
            visibilities=sample_visibilities,
            frequencies=sample_frequencies,
            antennas=sample_antennas,
            baselines=sample_baselines,
            location=sample_location,
            obstime=sample_obstime,
        )

        # Second write without overwrite should fail
        with pytest.raises(FileExistsError):
            write_ms(
                output_path=output_path,
                visibilities=sample_visibilities,
                frequencies=sample_frequencies,
                antennas=sample_antennas,
                baselines=sample_baselines,
                location=sample_location,
                obstime=sample_obstime,
                overwrite=False,
            )

        # Second write with overwrite should succeed
        write_ms(
            output_path=output_path,
            visibilities=sample_visibilities,
            frequencies=sample_frequencies,
            antennas=sample_antennas,
            baselines=sample_baselines,
            location=sample_location,
            obstime=sample_obstime,
            overwrite=True,
        )

        assert output_path.exists()


class TestReadMS:
    """Tests for read_ms function."""

    def test_read_ms_basic(
        self,
        temp_dir,
        sample_visibilities,
        sample_frequencies,
        sample_antennas,
        sample_baselines,
        sample_location,
        sample_obstime,
    ):
        """Test basic MS reading functionality."""
        output_path = temp_dir / "test_read.ms"

        # Write MS first
        write_ms(
            output_path=output_path,
            visibilities=sample_visibilities,
            frequencies=sample_frequencies,
            antennas=sample_antennas,
            baselines=sample_baselines,
            location=sample_location,
            obstime=sample_obstime,
        )

        # Read it back
        data = read_ms(output_path)

        assert "visibilities" in data
        assert "frequencies" in data
        assert "times" in data
        assert "ant_1_array" in data
        assert "ant_2_array" in data
        assert "polarizations" in data

    def test_read_ms_frequency_preservation(
        self,
        temp_dir,
        sample_visibilities,
        sample_frequencies,
        sample_antennas,
        sample_baselines,
        sample_location,
        sample_obstime,
    ):
        """Test that frequencies are preserved in round-trip."""
        output_path = temp_dir / "test_freq.ms"

        write_ms(
            output_path=output_path,
            visibilities=sample_visibilities,
            frequencies=sample_frequencies,
            antennas=sample_antennas,
            baselines=sample_baselines,
            location=sample_location,
            obstime=sample_obstime,
        )

        data = read_ms(output_path)

        np.testing.assert_allclose(
            data["frequencies"],
            sample_frequencies,
            rtol=1e-6,
        )

    def test_read_ms_nonexistent(self, temp_dir):
        """Test reading nonexistent MS raises error."""
        with pytest.raises(FileNotFoundError):
            read_ms(temp_dir / "nonexistent.ms")


class TestMSInfo:
    """Tests for ms_info function."""

    def test_ms_info_basic(
        self,
        temp_dir,
        sample_visibilities,
        sample_frequencies,
        sample_antennas,
        sample_baselines,
        sample_location,
        sample_obstime,
    ):
        """Test MS info extraction."""
        output_path = temp_dir / "test_info.ms"

        write_ms(
            output_path=output_path,
            visibilities=sample_visibilities,
            frequencies=sample_frequencies,
            antennas=sample_antennas,
            baselines=sample_baselines,
            location=sample_location,
            obstime=sample_obstime,
        )

        info = ms_info(output_path)

        assert info["n_antennas"] == 3
        assert info["n_channels"] == 10
        assert info["n_polarizations"] == 4  # XX, XY, YX, YY
        assert "frequencies" in info
        assert "time_range" in info

    def test_ms_info_nonexistent(self, temp_dir):
        """Test info for nonexistent MS raises error."""
        with pytest.raises(FileNotFoundError):
            ms_info(temp_dir / "nonexistent.ms")


class TestMSRoundTrip:
    """Integration tests for MS round-trip."""

    def test_full_round_trip(
        self,
        temp_dir,
        sample_visibilities,
        sample_frequencies,
        sample_antennas,
        sample_baselines,
        sample_location,
        sample_obstime,
    ):
        """Test full write-read round trip preserves data."""
        output_path = temp_dir / "test_roundtrip.ms"

        # Write
        write_ms(
            output_path=output_path,
            visibilities=sample_visibilities,
            frequencies=sample_frequencies,
            antennas=sample_antennas,
            baselines=sample_baselines,
            location=sample_location,
            obstime=sample_obstime,
        )

        # Read
        data = read_ms(output_path)

        # Verify structure
        assert data["n_frequencies"] == len(sample_frequencies)
        assert data["n_antennas"] == len(sample_antennas)
        assert data["n_baselines"] == len(sample_baselines)

    def test_stokes_i_visibility(
        self,
        temp_dir,
        sample_frequencies,
        sample_antennas,
        sample_baselines,
        sample_location,
        sample_obstime,
    ):
        """Test writing Stokes I visibilities."""
        # Create Stokes I only visibilities
        visibilities = {}
        n_freqs = 10

        for ant1 in range(3):
            for ant2 in range(ant1, 3):
                vis_i = np.random.randn(n_freqs) + 1j * np.random.randn(n_freqs)
                visibilities[(ant1, ant2)] = {"I": vis_i}

        output_path = temp_dir / "test_stokes.ms"

        write_ms(
            output_path=output_path,
            visibilities=visibilities,
            frequencies=sample_frequencies,
            antennas=sample_antennas,
            baselines=sample_baselines,
            location=sample_location,
            obstime=sample_obstime,
        )

        assert output_path.exists()


class TestMSImportFallback:
    """Test import behavior when dependencies are missing."""

    def test_io_module_exposes_ms_available(self):
        """Test that io module exposes MS_AVAILABLE flag."""
        from rrivis import io

        assert hasattr(io, "MS_AVAILABLE")
        assert isinstance(io.MS_AVAILABLE, bool)

    def test_io_module_has_write_ms(self):
        """Test that io module has write_ms function."""
        from rrivis import io

        assert hasattr(io, "write_ms")
        assert callable(io.write_ms)
