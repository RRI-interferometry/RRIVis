# tests/unit/test_core/test_antenna.py
"""
Unit tests for antenna position reading functionality.

Uses pytest tmp_path fixtures for creating temporary test files,
avoiding the need for complex mocking of file system operations.
"""

import pytest

from rrivis.core.antenna import read_antenna_positions


class TestRRIvisFormat:
    """Test RRIvis native format reading."""

    @pytest.fixture
    def rrivis_file_content(self):
        """Sample RRIvis format file content with BeamID column."""
        return """Name  Number   BeamID   E          N          U
HH136      136        0  -156.5976     2.9439    -0.1819
HH140      140        0   -98.1662     3.1671    -0.3008
HH121      121        0   -90.8139    -9.4618    -0.1707
"""

    @pytest.fixture
    def rrivis_file(self, tmp_path, rrivis_file_content):
        """Create a temporary RRIvis format file."""
        file_path = tmp_path / "antennas.txt"
        file_path.write_text(rrivis_file_content)
        return file_path

    def test_read_rrivis_format_correct(self, rrivis_file):
        """Test reading valid RRIvis format file."""
        antennas = read_antenna_positions(str(rrivis_file), "rrivis")

        expected = {
            136: {
                "Name": "HH136",
                "Number": 136,
                "BeamID": 0,
                "Position": (-156.5976, 2.9439, -0.1819),
            },
            140: {
                "Name": "HH140",
                "Number": 140,
                "BeamID": 0,
                "Position": (-98.1662, 3.1671, -0.3008),
            },
            121: {
                "Name": "HH121",
                "Number": 121,
                "BeamID": 0,
                "Position": (-90.8139, -9.4618, -0.1707),
            },
        }

        assert antennas == expected, (
            "Parsed antenna data does not match expected values"
        )

    def test_read_rrivis_default_format(self, rrivis_file):
        """Test that default format is 'rrivis'."""
        antennas = read_antenna_positions(str(rrivis_file))
        assert len(antennas) == 3
        assert 136 in antennas

    def test_rrivis_without_beamid(self, tmp_path):
        """Test RRIvis format without BeamID column."""
        content = """Name  Number   E          N          U
ANT0       0   100.0     200.0      3.0
ANT1       1   150.0     250.0      3.5
"""
        file_path = tmp_path / "no_beamid.txt"
        file_path.write_text(content)

        antennas = read_antenna_positions(str(file_path), "rrivis")

        assert len(antennas) == 2
        assert antennas[0]["BeamID"] is None
        assert antennas[0]["Position"] == (100.0, 200.0, 3.0)

    def test_rrivis_with_diameter(self, tmp_path):
        """Test RRIvis format with Diameter column."""
        content = """Name  Number   E          N          U       Diameter
ANT0       0   100.0     200.0      3.0      14.0
ANT1       1   150.0     250.0      3.5      14.0
"""
        file_path = tmp_path / "with_diameter.txt"
        file_path.write_text(content)

        antennas = read_antenna_positions(str(file_path), "rrivis")

        assert len(antennas) == 2
        assert antennas[0]["diameter"] == 14.0
        assert antennas[1]["diameter"] == 14.0

    def test_rrivis_with_comments(self, tmp_path):
        """Test RRIvis format with comment lines."""
        content = """# This is a comment
Name  Number   E          N          U
# Another comment
ANT0       0   100.0     200.0      3.0
ANT1       1   150.0     250.0      3.5
"""
        file_path = tmp_path / "with_comments.txt"
        file_path.write_text(content)

        antennas = read_antenna_positions(str(file_path), "rrivis")

        assert len(antennas) == 2

    def test_rrivis_string_beamid(self, tmp_path):
        """Test RRIvis format with string BeamID."""
        content = """Name  Number   BeamID      E          N          U
ANT0       0   beam_a   100.0     200.0      3.0
ANT1       1   beam_b   150.0     250.0      3.5
"""
        file_path = tmp_path / "string_beamid.txt"
        file_path.write_text(content)

        antennas = read_antenna_positions(str(file_path), "rrivis")

        assert antennas[0]["BeamID"] == "beam_a"
        assert antennas[1]["BeamID"] == "beam_b"


class TestCASAFormat:
    """Test CASA .cfg format reading."""

    @pytest.fixture
    def casa_file_content(self):
        """Sample CASA format file content."""
        return """#observatory=ALMA
#COFA=-67.75,-23.02
#coordsys=LOC (local tangent plane)
# x             y               z             diam  station  ant
-5.850273514   -125.9985379    -1.590364043   12.   A058     DA41
-19.90369337    52.82680653    -1.892119601   12.   A023     DA42
"""

    @pytest.fixture
    def casa_file(self, tmp_path, casa_file_content):
        """Create a temporary CASA format file."""
        file_path = tmp_path / "array.cfg"
        file_path.write_text(casa_file_content)
        return file_path

    def test_read_casa_format(self, casa_file):
        """Test reading CASA format file."""
        antennas = read_antenna_positions(str(casa_file), "casa")

        assert len(antennas) == 2
        assert 0 in antennas
        assert antennas[0]["Name"] == "DA41"
        assert antennas[0]["diameter"] == 12.0

    def test_casa_positions(self, casa_file):
        """Test CASA format position parsing."""
        antennas = read_antenna_positions(str(casa_file), "casa")

        # First antenna position
        pos = antennas[0]["Position"]
        assert len(pos) == 3
        assert abs(pos[0] - (-5.850273514)) < 1e-6
        assert abs(pos[1] - (-125.9985379)) < 1e-6

    def test_casa_without_optional_columns(self, tmp_path):
        """Test CASA format with only coordinates."""
        content = """# Simple CASA format
100.0  200.0  3.0
150.0  250.0  3.5
"""
        file_path = tmp_path / "simple.cfg"
        file_path.write_text(content)

        antennas = read_antenna_positions(str(file_path), "casa")

        assert len(antennas) == 2
        assert antennas[0]["Position"] == (100.0, 200.0, 3.0)


class TestPyUVDataFormat:
    """Test simple x y z coordinate format reading."""

    @pytest.fixture
    def pyuvdata_file_content(self):
        """Sample simple coordinate file content."""
        return """# Simple coordinates
-156.5976     2.9439    -0.1819
-98.1662     3.1671    -0.3008
"""

    @pytest.fixture
    def pyuvdata_file(self, tmp_path, pyuvdata_file_content):
        """Create a temporary pyuvdata format file."""
        file_path = tmp_path / "coords.txt"
        file_path.write_text(pyuvdata_file_content)
        return file_path

    def test_read_pyuvdata_format(self, pyuvdata_file):
        """Test reading simple coordinate file."""
        antennas = read_antenna_positions(str(pyuvdata_file), "pyuvdata")

        assert len(antennas) == 2
        assert 0 in antennas
        assert antennas[0]["Name"] == "ANT000"
        assert antennas[0]["Position"] == (-156.5976, 2.9439, -0.1819)

    def test_pyuvdata_auto_naming(self, pyuvdata_file):
        """Test auto-generated antenna names."""
        antennas = read_antenna_positions(str(pyuvdata_file), "pyuvdata")

        assert antennas[0]["Name"] == "ANT000"
        assert antennas[1]["Name"] == "ANT001"

    def test_pyuvdata_with_extra_columns(self, tmp_path):
        """Test that extra columns are ignored."""
        content = """100.0  200.0  3.0  extra  columns  here
150.0  250.0  3.5  more  stuff
"""
        file_path = tmp_path / "extra_cols.txt"
        file_path.write_text(content)

        antennas = read_antenna_positions(str(file_path), "pyuvdata")

        assert len(antennas) == 2
        assert antennas[0]["Position"] == (100.0, 200.0, 3.0)


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_file_not_found(self):
        """Test error raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_antenna_positions("/nonexistent/path/file.txt", "rrivis")

    def test_no_file_path_provided(self):
        """Test error raised for empty file path."""
        with pytest.raises(ValueError, match="file path is not provided"):
            read_antenna_positions("", "rrivis")

    def test_unsupported_format(self, tmp_path):
        """Test error raised for unsupported format type."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("dummy content")

        with pytest.raises(ValueError, match="Unsupported antenna file format"):
            read_antenna_positions(str(file_path), "unsupported_format")

    def test_empty_file(self, tmp_path):
        """Test error raised for empty file."""
        file_path = tmp_path / "empty.txt"
        file_path.write_text("")

        with pytest.raises(ValueError, match="No valid antenna data"):
            read_antenna_positions(str(file_path), "rrivis")

    def test_malformed_rrivis_file(self, tmp_path):
        """Test error raised for malformed RRIvis file (missing columns)."""
        content = """Name  Number   BeamID   E          N          U
HH136      136        0  -156.5976     2.9439
"""
        file_path = tmp_path / "malformed.txt"
        file_path.write_text(content)

        with pytest.raises(ValueError, match="expected at least"):
            read_antenna_positions(str(file_path), "rrivis")

    def test_invalid_data_in_rrivis(self, tmp_path):
        """Test error raised for invalid data in RRIvis file."""
        content = """Name  Number   E          N          U
ANT0       0   100.0     200.0      3.0
INVALID LINE HERE
ANT1       1   150.0     250.0      3.5
"""
        file_path = tmp_path / "invalid.txt"
        file_path.write_text(content)

        with pytest.raises(ValueError):
            read_antenna_positions(str(file_path), "rrivis")


class TestReturnFormats:
    """Test different return format options."""

    @pytest.fixture
    def sample_file(self, tmp_path):
        """Create a sample antenna file."""
        content = """Name  Number   E          N          U       Diameter
ANT0       0   100.0     200.0      3.0      14.0
ANT1       1   150.0     250.0      3.5      14.0
ANT2       2   200.0     300.0      4.0      14.0
"""
        file_path = tmp_path / "sample.txt"
        file_path.write_text(content)
        return file_path

    def test_dict_format(self, sample_file):
        """Test default dict return format."""
        antennas = read_antenna_positions(str(sample_file), return_format="dict")

        assert isinstance(antennas, dict)
        assert len(antennas) == 3
        assert isinstance(antennas[0], dict)
        assert "Position" in antennas[0]

    def test_arrays_format(self, sample_file):
        """Test arrays return format."""
        antennas = read_antenna_positions(str(sample_file), return_format="arrays")

        assert isinstance(antennas, dict)
        assert "names" in antennas
        assert "numbers" in antennas
        assert "positions_m" in antennas

        # Check array shapes
        assert len(antennas["names"]) == 3
        assert len(antennas["numbers"]) == 3
        assert antennas["positions_m"].shape == (3, 3)

    def test_arrays_format_with_diameter(self, sample_file):
        """Test arrays format includes diameters when present."""
        antennas = read_antenna_positions(str(sample_file), return_format="arrays")

        assert antennas["diameters"] is not None
        assert len(antennas["diameters"]) == 3
        assert all(d == 14.0 for d in antennas["diameters"])

    def test_invalid_return_format(self, sample_file):
        """Test error for invalid return format."""
        with pytest.raises(ValueError, match="Invalid return_format"):
            read_antenna_positions(str(sample_file), return_format="invalid")


class TestFormatAntennaData:
    """Test the format_antenna_data utility function."""

    def test_format_empty_dict(self):
        """Test error for empty antenna dictionary."""
        from rrivis.core.antenna import format_antenna_data

        with pytest.raises(ValueError, match="Empty antenna dictionary"):
            format_antenna_data({})

    def test_format_preserves_order(self, tmp_path):
        """Test that formatting preserves antenna number order."""
        content = """Name  Number   E          N          U
ANT2       2   200.0     300.0      4.0
ANT0       0   100.0     200.0      3.0
ANT1       1   150.0     250.0      3.5
"""
        file_path = tmp_path / "unordered.txt"
        file_path.write_text(content)

        antennas = read_antenna_positions(str(file_path), return_format="arrays")

        # Should be sorted by antenna number
        assert list(antennas["numbers"]) == [0, 1, 2]
        assert list(antennas["names"]) == ["ANT0", "ANT1", "ANT2"]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_antenna(self, tmp_path):
        """Test file with single antenna."""
        content = """Name  Number   E          N          U
ANT0       0   100.0     200.0      3.0
"""
        file_path = tmp_path / "single.txt"
        file_path.write_text(content)

        antennas = read_antenna_positions(str(file_path), "rrivis")

        assert len(antennas) == 1
        assert 0 in antennas

    def test_large_coordinates(self, tmp_path):
        """Test handling of large coordinate values."""
        content = """Name  Number   E              N              U
ANT0       0   1000000.123    2000000.456    500.789
"""
        file_path = tmp_path / "large.txt"
        file_path.write_text(content)

        antennas = read_antenna_positions(str(file_path), "rrivis")

        pos = antennas[0]["Position"]
        assert abs(pos[0] - 1000000.123) < 1e-3
        assert abs(pos[1] - 2000000.456) < 1e-3

    def test_negative_coordinates(self, tmp_path):
        """Test handling of negative coordinates."""
        content = """Name  Number   E          N          U
ANT0       0   -100.0    -200.0     -3.0
"""
        file_path = tmp_path / "negative.txt"
        file_path.write_text(content)

        antennas = read_antenna_positions(str(file_path), "rrivis")

        assert antennas[0]["Position"] == (-100.0, -200.0, -3.0)

    def test_scientific_notation(self, tmp_path):
        """Test handling of scientific notation in coordinates."""
        content = """Name  Number   E          N          U
ANT0       0   1.5e2     -2.0e-1    3.0e0
"""
        file_path = tmp_path / "scientific.txt"
        file_path.write_text(content)

        antennas = read_antenna_positions(str(file_path), "rrivis")

        pos = antennas[0]["Position"]
        assert abs(pos[0] - 150.0) < 1e-6
        assert abs(pos[1] - (-0.2)) < 1e-6
        assert abs(pos[2] - 3.0) < 1e-6

    def test_whitespace_handling(self, tmp_path):
        """Test handling of various whitespace."""
        content = """Name  Number   E          N          U
ANT0       0     100.0      200.0       3.0
ANT1       1   150.0   250.0    3.5
"""
        file_path = tmp_path / "whitespace.txt"
        file_path.write_text(content)

        antennas = read_antenna_positions(str(file_path), "rrivis")

        assert len(antennas) == 2
        assert antennas[0]["Position"] == (100.0, 200.0, 3.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
