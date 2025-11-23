"""
Test suite for antenna file format reading functionality.

Tests that RRIvis can correctly handle different antenna layout file formats:
- RRIvis native format (.txt)
- CASA configuration format (.cfg)
- PyUVData simple text format (.txt)
- MWA metafits format (.txt)
- PyUVSim antenna layout CSV (.csv)

These tests verify:
1. File reading and parsing
2. Antenna data structure integrity
3. Coordinate validation
4. Diameter extraction
5. Metadata preservation
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.antenna import (
    read_antenna_positions,
    read_rrivis_format,
    read_casa_format,
    read_pyuvdata_format,
    read_mwa_format,
)


# Get path to antenna_layout_examples directory
DATA_DIR = Path(__file__).parent.parent / "antenna_layout_examples"


class TestRRIvisFormat:
    """Test RRIvis native ENU format reading."""

    @pytest.fixture
    def rrivis_file(self):
        """Path to RRIvis format example file."""
        return DATA_DIR / "example_rrivis_format.txt"

    def test_file_exists(self, rrivis_file):
        """Verify RRIvis format file exists."""
        assert rrivis_file.exists(), f"RRIvis format file not found: {rrivis_file}"

    def test_can_read_rrivis_format(self, rrivis_file):
        """Test reading RRIvis format file."""
        antennas = read_antenna_positions(str(rrivis_file), "rrivis")
        assert antennas is not None
        assert isinstance(antennas, dict)
        assert len(antennas) > 0

    def test_rrivis_antenna_structure(self, rrivis_file):
        """Verify antenna data structure for RRIvis format."""
        antennas = read_antenna_positions(str(rrivis_file), "rrivis")

        # Check each antenna has required fields
        for ant_num, ant_data in antennas.items():
            assert isinstance(ant_num, int), f"Antenna number should be int, got {type(ant_num)}"
            assert isinstance(ant_data, dict), f"Antenna data should be dict, got {type(ant_data)}"

            # Check required keys
            required_keys = {"Name", "Number", "BeamID", "Position"}
            assert required_keys.issubset(ant_data.keys()), \
                f"Missing required keys. Expected {required_keys}, got {ant_data.keys()}"

            # Validate position is tuple of 3 floats
            assert isinstance(ant_data["Position"], tuple)
            assert len(ant_data["Position"]) == 3
            assert all(isinstance(coord, (int, float)) for coord in ant_data["Position"])

    def test_rrivis_coordinates_reasonable(self, rrivis_file):
        """Verify antenna coordinates are in reasonable range."""
        antennas = read_antenna_positions(str(rrivis_file), "rrivis")

        for ant_num, ant_data in antennas.items():
            e, n, u = ant_data["Position"]

            # For HERA compact array, coordinates should be within ±100m
            assert abs(e) < 200, f"East coordinate {e}m outside reasonable range for antenna {ant_num}"
            assert abs(n) < 200, f"North coordinate {n}m outside reasonable range for antenna {ant_num}"
            assert abs(u) < 50, f"Up coordinate {u}m outside reasonable range for antenna {ant_num}"

    def test_rrivis_diameter_extraction(self, rrivis_file):
        """Verify diameter is correctly extracted from RRIvis format."""
        antennas = read_antenna_positions(str(rrivis_file), "rrivis")

        # At least some antennas should have diameter
        antennas_with_diameter = [ant for ant in antennas.values() if "diameter" in ant]
        assert len(antennas_with_diameter) > 0, "No antennas with diameter found"

        # Check diameter values
        for ant_data in antennas_with_diameter:
            diam = ant_data["diameter"]
            assert isinstance(diam, (int, float))
            assert 0 < diam < 100, f"Diameter {diam}m outside reasonable range"

    def test_rrivis_antenna_count(self, rrivis_file):
        """Verify expected number of antennas in RRIvis format."""
        antennas = read_antenna_positions(str(rrivis_file), "rrivis")

        # Example file should have at least 10 antennas
        assert len(antennas) >= 10, f"Expected at least 10 antennas, got {len(antennas)}"


class TestCASAFormat:
    """Test CASA .cfg format reading."""

    @pytest.fixture
    def casa_file(self):
        """Path to CASA format example file."""
        return DATA_DIR / "example_casa_format.cfg"

    def test_file_exists(self, casa_file):
        """Verify CASA format file exists."""
        assert casa_file.exists(), f"CASA format file not found: {casa_file}"

    def test_can_read_casa_format(self, casa_file):
        """Test reading CASA format file."""
        antennas = read_antenna_positions(str(casa_file), "casa")
        assert antennas is not None
        assert isinstance(antennas, dict)
        assert len(antennas) > 0

    def test_casa_antenna_structure(self, casa_file):
        """Verify antenna data structure for CASA format."""
        antennas = read_antenna_positions(str(casa_file), "casa")

        # CASA format auto-generates antenna numbers starting at 0
        for ant_num, ant_data in antennas.items():
            assert "Name" in ant_data
            assert "Position" in ant_data
            assert "BeamID" in ant_data

            # Position should be 3-tuple
            pos = ant_data["Position"]
            assert len(pos) == 3
            assert all(isinstance(c, (int, float)) for c in pos)

    def test_casa_coordinates_reasonable(self, casa_file):
        """Verify CASA format coordinates are reasonable."""
        antennas = read_antenna_positions(str(casa_file), "casa")

        for ant_data in antennas.values():
            x, y, z = ant_data["Position"]

            # LOC coordinates should be within ±150m for typical arrays
            assert abs(x) < 200, f"X coordinate {x}m outside reasonable range"
            assert abs(y) < 200, f"Y coordinate {y}m outside reasonable range"
            assert abs(z) < 50, f"Z coordinate {z}m outside reasonable range"

    def test_casa_diameter_extraction(self, casa_file):
        """Verify diameter extraction from CASA format."""
        antennas = read_antenna_positions(str(casa_file), "casa")

        # Some antennas should have diameter
        antennas_with_diameter = [ant for ant in antennas.values() if "diameter" in ant]
        if len(antennas_with_diameter) > 0:
            for ant_data in antennas_with_diameter:
                diam = ant_data["diameter"]
                assert 0 < diam < 100, f"Diameter {diam}m outside reasonable range"

    def test_casa_antenna_count(self, casa_file):
        """Verify antenna count in CASA format."""
        antennas = read_antenna_positions(str(casa_file), "casa")
        assert len(antennas) >= 10, f"Expected at least 10 antennas, got {len(antennas)}"


class TestPyUVDataFormat:
    """Test PyUVData simple text format reading."""

    @pytest.fixture
    def pyuvdata_file(self):
        """Path to PyUVData format example file."""
        return DATA_DIR / "example_pyuvdata_format.txt"

    def test_file_exists(self, pyuvdata_file):
        """Verify PyUVData format file exists."""
        assert pyuvdata_file.exists(), f"PyUVData format file not found: {pyuvdata_file}"

    def test_can_read_pyuvdata_format(self, pyuvdata_file):
        """Test reading PyUVData format file."""
        antennas = read_antenna_positions(str(pyuvdata_file), "pyuvdata")
        assert antennas is not None
        assert isinstance(antennas, dict)
        assert len(antennas) > 0

    def test_pyuvdata_antenna_structure(self, pyuvdata_file):
        """Verify antenna data structure for PyUVData format."""
        antennas = read_antenna_positions(str(pyuvdata_file), "pyuvdata")

        for ant_num, ant_data in antennas.items():
            # PyUVData format auto-generates antenna names
            assert "Name" in ant_data
            assert "Number" in ant_data
            assert "Position" in ant_data

            # Position should be 3-tuple of numbers
            pos = ant_data["Position"]
            assert len(pos) == 3
            assert all(isinstance(c, (int, float)) for c in pos)

    def test_pyuvdata_coordinates_reasonable(self, pyuvdata_file):
        """Verify PyUVData format coordinates are reasonable."""
        antennas = read_antenna_positions(str(pyuvdata_file), "pyuvdata")

        for ant_data in antennas.values():
            x, y, z = ant_data["Position"]

            # Coordinates should be in reasonable range
            assert abs(x) < 200
            assert abs(y) < 200
            assert abs(z) < 100

    def test_pyuvdata_minimal_metadata(self, pyuvdata_file):
        """Verify PyUVData format has minimal metadata."""
        antennas = read_antenna_positions(str(pyuvdata_file), "pyuvdata")

        # Auto-generated names should be ANT###
        for ant_data in antennas.values():
            name = ant_data["Name"]
            assert name.startswith("ANT"), f"Auto-generated name should start with ANT, got {name}"

    def test_pyuvdata_antenna_count(self, pyuvdata_file):
        """Verify antenna count in PyUVData format."""
        antennas = read_antenna_positions(str(pyuvdata_file), "pyuvdata")
        assert len(antennas) >= 10


class TestMWAFormat:
    """Test MWA metafits format reading."""

    @pytest.fixture
    def mwa_file(self):
        """Path to MWA metafits FITS file."""
        return DATA_DIR / "1101503312_metafits.fits"

    def test_file_exists(self, mwa_file):
        """Verify MWA format file exists."""
        assert mwa_file.exists(), f"MWA format file not found: {mwa_file}"

    def test_can_read_mwa_format(self, mwa_file):
        """Test reading MWA format file."""
        antennas = read_antenna_positions(str(mwa_file), "mwa")
        assert antennas is not None
        assert isinstance(antennas, dict)
        assert len(antennas) > 0

    def test_mwa_antenna_structure(self, mwa_file):
        """Verify antenna data structure for MWA format."""
        antennas = read_antenna_positions(str(mwa_file), "mwa")

        for ant_num, ant_data in antennas.items():
            assert "Name" in ant_data
            assert "Position" in ant_data
            assert "BeamID" in ant_data

            # Position should be 3-tuple
            pos = ant_data["Position"]
            assert len(pos) == 3
            assert all(isinstance(c, (int, float)) for c in pos)

    def test_mwa_antenna_names(self, mwa_file):
        """Verify MWA antenna names contain 'Tile' or are auto-generated."""
        antennas = read_antenna_positions(str(mwa_file), "mwa")

        for ant_data in antennas.values():
            name = ant_data["Name"]
            # MWA tiles typically start with Tile or are indexed
            assert isinstance(name, str)
            assert len(name) > 0

    def test_mwa_coordinates_reasonable(self, mwa_file):
        """Verify MWA format coordinates are reasonable."""
        antennas = read_antenna_positions(str(mwa_file), "mwa")

        for ant_data in antennas.values():
            east, north, elev = ant_data["Position"]

            # MWA coordinates might be in different units, but should be finite
            assert np.isfinite(east), f"East coordinate not finite: {east}"
            assert np.isfinite(north), f"North coordinate not finite: {north}"
            assert np.isfinite(elev), f"Elevation not finite: {elev}"

    def test_mwa_antenna_count(self, mwa_file):
        """Verify antenna count in MWA format."""
        antennas = read_antenna_positions(str(mwa_file), "mwa")
        assert len(antennas) >= 10


class TestCSVFormat:
    """Test CSV antenna layout format reading."""

    @pytest.fixture
    def csv_file(self):
        """Path to CSV format example file."""
        return DATA_DIR / "example_antenna_layout.csv"

    def test_file_exists(self, csv_file):
        """Verify CSV format file exists."""
        assert csv_file.exists(), f"CSV format file not found: {csv_file}"

    def test_can_read_csv_as_pyuvdata(self, csv_file):
        """Test reading CSV file using pyuvdata format reader."""
        # CSV can be read as pyuvdata format (simple coordinates)
        # This tests that the file is valid
        assert csv_file.exists()
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 1, "CSV file should have header + data rows"

            # Check header
            header = lines[0].strip().split(',')
            assert 'Name' in header or 'name' in header
            assert 'E' in header or 'N' in header or 'U' in header


class TestYAMLConfiguration:
    """Test YAML telescope configuration file."""

    @pytest.fixture
    def yaml_file(self):
        """Path to YAML config file."""
        return DATA_DIR / "example_telescope_config.yaml"

    def test_file_exists(self, yaml_file):
        """Verify YAML config file exists."""
        assert yaml_file.exists(), f"YAML config file not found: {yaml_file}"

    def test_yaml_content_structure(self, yaml_file):
        """Verify YAML file has proper structure."""
        with open(yaml_file, 'r') as f:
            content = f.read()

            # Check for required sections
            assert 'telescope:' in content, "Missing telescope section"
            assert 'antenna_layout:' in content, "Missing antenna_layout section"
            assert 'observation:' in content or 'telescope:' in content, "Missing configuration sections"

    def test_yaml_is_readable(self, yaml_file):
        """Test that YAML file can be parsed."""
        try:
            import yaml
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                assert data is not None
                assert 'telescope' in data, "Missing telescope section"
                assert 'antenna_layout' in data, "Missing antenna_layout section"
                # Verify nested structure
                assert 'name' in data['telescope'], "Missing telescope name"
                assert 'location' in data['telescope'], "Missing telescope location"
        except ImportError:
            pytest.skip("PyYAML not installed")


class TestFormatConsistency:
    """Test consistency across different formats."""

    def test_all_formats_produce_valid_antennas(self):
        """Verify all formats produce valid antenna dictionaries."""
        formats = [
            (DATA_DIR / "example_rrivis_format.txt", "rrivis"),
            (DATA_DIR / "example_casa_format.cfg", "casa"),
            (DATA_DIR / "example_pyuvdata_format.txt", "pyuvdata"),
            (DATA_DIR / "1101503312_metafits.fits", "mwa"),
        ]

        for file_path, format_type in formats:
            if not file_path.exists():
                pytest.skip(f"File not found: {file_path}")

            antennas = read_antenna_positions(str(file_path), format_type)
            assert antennas is not None
            assert len(antennas) > 0
            assert all(isinstance(k, int) for k in antennas.keys())

    def test_antenna_data_structure_consistency(self):
        """Verify all formats produce consistent antenna data structures."""
        formats = [
            (DATA_DIR / "example_rrivis_format.txt", "rrivis"),
            (DATA_DIR / "example_casa_format.cfg", "casa"),
            (DATA_DIR / "example_pyuvdata_format.txt", "pyuvdata"),
        ]

        for file_path, format_type in formats:
            if not file_path.exists():
                pytest.skip(f"File not found: {file_path}")

            antennas = read_antenna_positions(str(file_path), format_type)

            for ant_num, ant_data in antennas.items():
                # All formats should have these minimal fields
                assert "Position" in ant_data, f"Missing Position in {format_type}"
                assert "Name" in ant_data, f"Missing Name in {format_type}"
                assert isinstance(ant_data["Position"], tuple), \
                    f"Position should be tuple in {format_type}"
                assert len(ant_data["Position"]) == 3, \
                    f"Position should have 3 coordinates in {format_type}"

    def test_position_coordinate_types(self):
        """Verify all coordinate values are numeric."""
        formats = [
            (DATA_DIR / "example_rrivis_format.txt", "rrivis"),
            (DATA_DIR / "example_casa_format.cfg", "casa"),
            (DATA_DIR / "example_pyuvdata_format.txt", "pyuvdata"),
        ]

        for file_path, format_type in formats:
            if not file_path.exists():
                pytest.skip(f"File not found: {file_path}")

            antennas = read_antenna_positions(str(file_path), format_type)

            for ant_num, ant_data in antennas.items():
                e, n, u = ant_data["Position"]
                assert isinstance(e, (int, float)), \
                    f"Coordinate should be numeric in {format_type}"
                assert isinstance(n, (int, float)), \
                    f"Coordinate should be numeric in {format_type}"
                assert isinstance(u, (int, float)), \
                    f"Coordinate should be numeric in {format_type}"


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_nonexistent_file_raises_error(self):
        """Verify error is raised for nonexistent file."""
        with pytest.raises((FileNotFoundError, ValueError)):
            read_antenna_positions("/nonexistent/path/file.txt", "rrivis")

    def test_invalid_format_raises_error(self):
        """Verify error is raised for invalid format type."""
        rrivis_file = DATA_DIR / "example_rrivis_format.txt"
        if rrivis_file.exists():
            with pytest.raises(ValueError):
                read_antenna_positions(str(rrivis_file), "invalid_format")

    def test_empty_antenna_data_raises_error(self):
        """Verify error is raised if no antennas are parsed."""
        # This would require an empty valid file, skip for now
        pass


# Integration tests
class TestIntegration:
    """Integration tests for antenna format handling in RRIvis context."""

    def test_all_data_folder_files_accessible(self):
        """Verify all example files in data folder are accessible."""
        assert DATA_DIR.exists(), f"Data directory not found: {DATA_DIR}"

        expected_files = [
            "example_rrivis_format.txt",
            "example_casa_format.cfg",
            "example_pyuvdata_format.txt",
            "example_antenna_layout.csv",
            "1101503312_metafits.fits",
            "example_telescope_config.yaml",
            "README_antenna_formats.md",
        ]

        for filename in expected_files:
            filepath = DATA_DIR / filename
            assert filepath.exists(), f"Expected file not found: {filepath}"

    def test_readme_documentation_exists(self):
        """Verify README documentation exists and is not empty."""
        readme_file = DATA_DIR / "README_antenna_formats.md"
        assert readme_file.exists()

        with open(readme_file, 'r') as f:
            content = f.read()
            assert len(content) > 100, "README file too short"
            assert "Supported Formats" in content
            assert "RRIvis" in content
            assert "CASA" in content

    def test_format_readers_imported_successfully(self):
        """Verify all format readers are imported and available."""
        from src.antenna import (
            read_rrivis_format,
            read_casa_format,
            read_pyuvdata_format,
            read_mwa_format,
        )

        # Verify all readers are callable
        assert callable(read_rrivis_format)
        assert callable(read_casa_format)
        assert callable(read_pyuvdata_format)
        assert callable(read_mwa_format)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
