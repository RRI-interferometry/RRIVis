# src/antenna.py
import sys
import os
import re
import numpy as np
from astropy.io import fits
from astropy.table import Table
import warnings
from pathlib import Path

# Try to import pyuvdata for MS and UVFITS support
try:
    from pyuvdata import UVData
    PYUVDATA_AVAILABLE = True
except ImportError:
    PYUVDATA_AVAILABLE = False
    warnings.warn("pyuvdata not available. MS and UVFITS format support disabled.")

# Try to import astropy for coordinate conversions
try:
    from astropy.coordinates import EarthLocation, AltAz, ICRS
    from astropy import units as u
    from astropy.time import Time
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    warnings.warn("astropy not fully available. Some coordinate conversions disabled.")


def _supports_color() -> bool:
    try:
        return hasattr(sys.__stdout__, "isatty") and sys.__stdout__.isatty()
    except Exception:
        return False


_TTY = _supports_color()
_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"


def _c(text: str, style: str) -> str:
    if not _TTY:
        return text
    return f"{style}{text}{_RESET}"


def _convert_coordinates_to_enu(positions, coordsys, reference_location=None):
    """
    Convert various coordinate systems to ENU (East-North-Up).

    Parameters:
    -----------
    positions : array-like
        Antenna positions in the specified coordinate system
    coordsys : str
        Coordinate system: 'ENU', 'XYZ', 'ITRF', 'LOC', 'ENH'
    reference_location : tuple, optional
        Reference location (lat, lon, height) for coordinate conversions

    Returns:
    --------
    np.ndarray
        Positions converted to ENU coordinates in meters
    """
    positions = np.array(positions)
    if coordsys.upper() == 'ENU':
        return positions
    elif coordsys.upper() in ['XYZ', 'ITRF']:
        # Convert from ITRF/XYZ to ENU
        if reference_location is None:
            warnings.warn("No reference location provided for ITRF->ENU conversion")
            # Default to HERA coordinates
            reference_location = (-30.72152777777791, 21.428305555555557, 1073.0)

        lat, lon, height = reference_location
        if ASTROPY_AVAILABLE:
            # Use astropy for conversion
            from astropy.coordinates import spherical_to_cartesian

            # Create rotation matrix for ITRF to ENU conversion
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)

            # Rotation matrix from ITRF to ENU
            R = np.array([
                [-np.sin(lon_rad), np.cos(lon_rad), 0],
                [-np.sin(lat_rad)*np.cos(lon_rad), -np.sin(lat_rad)*np.sin(lon_rad), np.cos(lat_rad)],
                [np.cos(lat_rad)*np.cos(lon_rad), np.cos(lat_rad)*np.sin(lon_rad), np.sin(lat_rad)]
            ])

            # Subtract reference position and rotate
            ref_xyz = spherical_to_cartesian(height + 6371000, np.pi/2 - lat_rad, lon_rad)[0]
            enu_positions = []
            for pos in positions:
                rel_pos = pos - ref_xyz
                enu = R @ rel_pos
                enu_positions.append(enu)

            return np.array(enu_positions)
        else:
            warnings.warn("astropy not available, returning original coordinates")
            return positions
    else:
        warnings.warn(f"Unknown coordinate system {coordsys}, returning original positions")
        return positions


def read_rrivis_format(file_path):
    """
    Read RRIvis format antenna files.

    Format: Name  Number  BeamID  E  N  U  [Diameter]
    """
    antennas = {}
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Detect header and optional Diameter column index
    header_idx = None
    diameter_col_idx = None
    for idx, line in enumerate(lines):
        # Skip empty lines and comment lines when looking for header
        if line.strip() and not line.strip().startswith('#'):
            header_idx = idx
            header_tokens = line.strip().split()
            for j, tok in enumerate(header_tokens):
                if tok.lower() == "diameter":
                    diameter_col_idx = j
                    break
            break

    for i, line in enumerate(lines):
        # Skip the header line, empty lines, or comment lines
        if i == header_idx or not line.strip() or line.strip().startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) < 6:
            raise ValueError(f"Invalid antenna position in line {i+1}: {line}")

        # Extract metadata and positions
        try:
            name = parts[0]
            number = int(parts[1])
            beam_id = int(parts[2])
            e, n, u = map(float, parts[3:6])
            ant = {
                "Name": name,
                "Number": number,
                "BeamID": beam_id,
                "Position": (e, n, u),
            }
            # Optional diameter if header specified a Diameter column and value present
            if diameter_col_idx is not None and len(parts) > diameter_col_idx:
                try:
                    ant["diameter"] = float(parts[diameter_col_idx])
                except Exception:
                    pass
            antennas[number] = ant
        except ValueError:
            raise ValueError(f"Could not parse data in line {i+1}: {line}")

    return antennas


def read_casa_format(file_path):
    """
    Read CASA .cfg format antenna files.

    Format:
    #observatory=ALMA
    #COFA=-67.75,-23.02
    #coordsys=LOC (local tangent plane)
    # x             y               z             diam  station  ant
    -5.850273514   -125.9985379    -1.590364043   12.   A058     DA41
    """
    antennas = {}

    # Parse header information
    coordsys = 'LOC'  # Default
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract coordinate system from header
    for line in lines:
        if line.startswith('#coordsys='):
            coordsys_str = line.split('=')[1].strip()
            if 'LOC' in coordsys_str or 'ENU' in coordsys_str:
                coordsys = 'ENU'
            elif 'XYZ' in coordsys_str:
                coordsys = 'XYZ'
            else:
                coordsys = 'LOC'  # Default to local

    # Parse antenna data
    ant_idx = 0
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue

        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                x, y, z = map(float, parts[:3])
                diameter = float(parts[3]) if len(parts) > 3 and parts[3].replace('.', '').isdigit() else None
                station = parts[4] if len(parts) > 4 else f"A{ant_idx:03d}"
                name = parts[5] if len(parts) > 5 else station

                # Convert to ENU if needed
                if coordsys == 'ENU':
                    e, n, u = x, y, z
                else:
                    # For other coordinate systems, we'd need more complex conversion
                    # For now, assume LOC coordinates are ENU-like
                    e, n, u = x, y, z

                ant = {
                    "Name": name,
                    "Number": ant_idx,
                    "BeamID": 0,  # Default beam ID
                    "Position": (e, n, u),
                }
                if diameter is not None:
                    ant["diameter"] = diameter

                antennas[ant_idx] = ant
                ant_idx += 1

            except ValueError:
                continue  # Skip lines that can't be parsed

    return antennas


def read_measurement_set(file_path):
    """
    Read antenna positions from CASA Measurement Set files.

    Requires pyuvdata to be available.
    """
    if not PYUVDATA_AVAILABLE:
        raise ImportError("pyuvdata is required for Measurement Set support. Install with: pip install pyuvdata")

    try:
        uv = UVData()
        uv.read(file_path)

        antennas = {}
        ant_names = uv.antenna_names
        ant_numbers = uv.antenna_numbers
        ant_positions = uv.antenna_positions  # in ENU coordinates relative to array center
        antenna_diameters = uv.antenna_diameter

        for i, (name, number, pos, diam) in enumerate(zip(ant_names, ant_numbers, ant_positions, antenna_diameters)):
            ant = {
                "Name": name,
                "Number": int(number),
                "BeamID": 0,  # Default beam ID
                "Position": tuple(pos),  # Already in ENU coordinates
            }
            if diam > 0:
                ant["diameter"] = float(diam)

            antennas[int(number)] = ant

        return antennas

    except Exception as e:
        raise ValueError(f"Failed to read Measurement Set '{file_path}': {e}")


def read_uvfits(file_path):
    """
    Read antenna positions from UVFITS files.

    Requires pyuvdata to be available.
    """
    if not PYUVDATA_AVAILABLE:
        raise ImportError("pyuvdata is required for UVFITS support. Install with: pip install pyuvdata")

    try:
        uv = UVData()
        uv.read(file_path)

        antennas = {}
        ant_names = uv.antenna_names
        ant_numbers = uv.antenna_numbers
        ant_positions = uv.antenna_positions  # in ENU coordinates
        antenna_diameters = uv.antenna_diameter

        for i, (name, number, pos, diam) in enumerate(zip(ant_names, ant_numbers, ant_positions, antenna_diameters)):
            ant = {
                "Name": name,
                "Number": int(number),
                "BeamID": 0,  # Default beam ID
                "Position": tuple(pos),
            }
            if diam > 0:
                ant["diameter"] = float(diam)

            antennas[int(number)] = ant

        return antennas

    except Exception as e:
        raise ValueError(f"Failed to read UVFITS file '{file_path}': {e}")


def read_mwa_format(file_path):
    """
    Read MWA metafits format antenna files.

    Format: coordinates (m) East North Elevation x y z
    Tile011MWA 4.0 m +116.40.09.5 -26.32.48.3 -150.1601 264.8352 376.9023
    """
    antennas = {}

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Look for coordinate header
    coordsys = 'ENU'  # Default
    for line in lines:
        if 'coordinates' in line.lower() and 'east' in line.lower():
            coordsys = 'ENU'
            break

    ant_idx = 0
    for line in lines:
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split()

            # Try to identify MWA format
            if len(parts) >= 7 and any(coord in line.upper() for coord in ['EAST', 'NORTH', 'ELEVATION']):
                try:
                    # Skip to the coordinate values
                    coord_start = 0
                    for i, part in enumerate(parts):
                        try:
                            float(part)
                            coord_start = i
                            break
                        except ValueError:
                            continue

                    if coord_start + 3 <= len(parts):
                        east, north, elevation = map(float, parts[coord_start:coord_start+3])

                        # Extract name and diameter if available
                        name = parts[0] if parts[0] not in ['coordinates', 'm'] else f"Tile{ant_idx:03d}"
                        diameter = None

                        ant = {
                            "Name": name,
                            "Number": ant_idx,
                            "BeamID": 0,
                            "Position": (east, north, elevation),
                        }
                        if diameter is not None:
                            ant["diameter"] = diameter

                        antennas[ant_idx] = ant
                        ant_idx += 1

                except ValueError:
                    continue

    return antennas


def read_pyuvdata_format(file_path):
    """
    Read antenna positions from numpy array format or simple coordinate files.

    Expected format: Simple text file with x y z coordinates per line
    """
    antennas = {}

    try:
        # Try to read as simple text file first
        with open(file_path, "r") as f:
            lines = f.readlines()

        ant_idx = 0
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        x, y, z = map(float, parts[:3])

                        ant = {
                            "Name": f"ANT{ant_idx:03d}",
                            "Number": ant_idx,
                            "BeamID": 0,
                            "Position": (x, y, z),
                        }

                        antennas[ant_idx] = ant
                        ant_idx += 1

                    except ValueError:
                        continue

    except Exception as e:
        raise ValueError(f"Failed to read pyuvdata format file '{file_path}': {e}")

    return antennas


def read_antenna_positions(file_path, format_type="rrivis"):
    """
    Reads antenna positions and metadata from various file formats.

    Parameters:
    -----------
    file_path : str
        Path to the antenna position file.
    format_type : str
        Format of the antenna file. Options:
        - "rrivis" (RRIvis ENU format)
        - "casa" (CASA .cfg files)
        - "measurement_set" (MS format)
        - "uvfits" (UVFITS format)
        - "mwa" (MWA metafits)
        - "pyuvdata" (numpy arrays/simple text)

    Returns:
    --------
    dict: Dictionary with antenna indices as keys and dictionaries of metadata and positions as values.

    Raises:
    -------
    ValueError: If the file is empty, has invalid data, or the file format is incorrect.
    FileNotFoundError: If the file path does not exist.
    ImportError: If required dependencies are missing for specific formats.
    """

    # Check if file path is provided
    if not file_path:
        raise ValueError("Antenna positions file path is not provided.")

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # Determine file extension for automatic format detection if needed
    file_ext = Path(file_path).suffix.lower()

    # Parse based on format type
    try:
        if format_type.lower() == "rrivis":
            antennas = read_rrivis_format(file_path)
        elif format_type.lower() == "casa":
            antennas = read_casa_format(file_path)
        elif format_type.lower() == "measurement_set":
            antennas = read_measurement_set(file_path)
        elif format_type.lower() == "uvfits":
            antennas = read_uvfits(file_path)
        elif format_type.lower() == "mwa":
            antennas = read_mwa_format(file_path)
        elif format_type.lower() == "pyuvdata":
            antennas = read_pyuvdata_format(file_path)
        else:
            raise ValueError(f"Unsupported antenna file format: {format_type}. "
                           f"Supported formats: rrivis, casa, measurement_set, uvfits, mwa, pyuvdata")

        if not antennas:
            raise ValueError("No valid antenna data found in file.")

        # Debug output
        print("")
        print(_c("Antenna metadata and positions:", _BOLD + _CYAN))
        for idx, data in antennas.items():
            print(f"{_c(f'Antenna {idx}:', _BOLD + _CYAN)} {data}")

        # Calculate total memory usage in MB
        total_memory_bytes = sys.getsizeof(antennas) + sum(
            sys.getsizeof(value) + sum(sys.getsizeof(v) for v in value.values())
            for value in antennas.values()
        )
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        print(
            f"{_c('Total memory used by antennas:', _BOLD + _CYAN)} {total_memory_mb:.4f} MB"
        )

        return antennas

    except Exception as e:
        raise ValueError(f"Failed to read antenna positions file '{file_path}' with format '{format_type}': {e}")


# Backward compatibility function
def read_antenna_positions_legacy(file_path):
    """
    Legacy function for backward compatibility.
    Uses the original RRIvis format.
    """
    return read_antenna_positions(file_path, "rrivis")