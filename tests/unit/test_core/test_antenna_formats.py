"""
Pytest tests for antenna format reading functionality.
Tests different antenna file formats are properly loaded.
"""

import pytest
from pathlib import Path
from rrivis.core.antenna import read_antenna_positions

# Get the antenna_layout_examples directory
DATA_DIR = Path(__file__).parent.parent.parent.parent / "antenna_layout_examples"


@pytest.mark.parametrize("format_name,filename,min_expected", [
    ("rrivis", "example_rrivis_format.txt", 10),
    ("casa", "example_casa_format.cfg", 10),
    ("pyuvdata", "example_pyuvdata_format.txt", 10),
    ("mwa", "1101503312_metafits.fits", 100),  # MWA has 128 tiles
])
def test_format_loading(format_name, filename, min_expected):
    """Test loading different antenna file formats."""
    file_path = DATA_DIR / filename

    # Skip if file doesn't exist
    if not file_path.exists():
        pytest.skip(f"Example file not found: {file_path}")

    # Load the antennas
    antennas = read_antenna_positions(str(file_path), format_name)

    # Verify we got antennas
    assert antennas is not None, f"Failed to load {format_name} format"
    assert isinstance(antennas, dict), f"Expected dict, got {type(antennas)}"
    assert len(antennas) > 0, f"No antennas loaded from {format_name} format"

    # Check minimum count
    assert len(antennas) >= min_expected, \
        f"Expected at least {min_expected} antennas, got {len(antennas)}"

    # Verify antenna structure
    first_ant = next(iter(antennas.values()))
    assert "Name" in first_ant, "Antenna missing 'Name' field"
    assert "Position" in first_ant, "Antenna missing 'Position' field"
    assert isinstance(first_ant["Position"], tuple), "Position should be a tuple"
    assert len(first_ant["Position"]) == 3, "Position should have 3 coordinates"