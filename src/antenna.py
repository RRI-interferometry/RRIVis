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

    Format (BeamID is optional):
    - With BeamID:    Name  Number  BeamID  E  N  U  [Diameter]
    - Without BeamID: Name  Number  E  N  U  [Diameter]

    Returns:
    --------
    dict: Antenna data with optional BeamID field
    """
    antennas = {}
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Detect header and optional column indices
    header_idx = None
    has_beamid_col = False
    diameter_col_idx = None

    for idx, line in enumerate(lines):
        # Skip empty lines and comment lines when looking for header
        if line.strip() and not line.strip().startswith('#'):
            header_idx = idx
            header_tokens = line.strip().split()

            # Check for BeamID column
            for j, tok in enumerate(header_tokens):
                if tok.lower() == "beamid":
                    has_beamid_col = True
                if tok.lower() == "diameter":
                    diameter_col_idx = j
            break

    # Determine expected minimum columns based on header
    min_cols = 6 if has_beamid_col else 5  # With/without BeamID

    for i, line in enumerate(lines):
        # Skip the header line, empty lines, or comment lines
        if i == header_idx or not line.strip() or line.strip().startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) < min_cols:
            raise ValueError(f"Invalid antenna position in line {i+1}: expected at least {min_cols} columns, got {len(parts)}")

        # Extract metadata and positions
        try:
            name = parts[0]
            number = int(parts[1])

            if has_beamid_col:
                # Format: Name Number BeamID E N U [Diameter]
                # Try to convert BeamID to int, fallback to string for non-numeric IDs
                try:
                    beam_id = int(parts[2])
                except ValueError:
                    beam_id = parts[2]
                e, n, u = map(float, parts[3:6])
            else:
                # Format: Name Number E N U [Diameter]
                beam_id = None
                e, n, u = map(float, parts[2:5])

            ant = {
                "Name": name,
                "Number": number,
                "BeamID": beam_id,  # Can be string or None
                "Position": (e, n, u),
            }

            # Optional diameter if header specified a Diameter column and value present
            if diameter_col_idx is not None and len(parts) > diameter_col_idx:
                try:
                    ant["diameter"] = float(parts[diameter_col_idx])
                except Exception:
                    pass

            antennas[number] = ant

        except ValueError as e:
            raise ValueError(f"Could not parse data in line {i+1}: {line}. Error: {e}")

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
                    "BeamID": None,  # No beam ID info in CASA format
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
                "BeamID": None,  # No beam ID info in Measurement Set
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
                "BeamID": None,  # No beam ID info in UVFITS
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
    Read MWA metafits FITS file format.

    MWA metafits files are FITS files containing antenna positions in ENU coordinates.
    The TILEDATA table has 256 rows (2 per tile for X and Y polarization).
    We deduplicate to get unique tile positions.
    """
    try:
        from astropy.io import fits
    except ImportError:
        raise ImportError("astropy is required to read MWA metafits FITS files. Install with: pip install astropy")

    antennas = {}

    try:
        # Open the FITS file
        with fits.open(file_path) as hdul:
            # Get the TILEDATA table
            if 'TILEDATA' not in hdul:
                raise ValueError("TILEDATA table not found in MWA metafits file")

            tile_table = hdul['TILEDATA'].data

            # Deduplicate tiles (each tile has 2 rows for X and Y polarization)
            seen_tiles = {}

            for row in tile_table:
                tile_name = row['TileName'].strip()
                antenna_num = int(row['Antenna'])

                # Skip if we've already processed this tile
                if tile_name in seen_tiles:
                    continue

                # Get ENU coordinates (note: metafits has East, North, Height)
                east = float(row['East'])
                north = float(row['North'])
                height = float(row['Height'])

                ant = {
                    "Name": tile_name,
                    "Number": antenna_num,
                    "BeamID": None,  # MWA doesn't use BeamID in this context
                    "Position": (east, north, height),
                }

                # Use antenna number as key for consistency with other formats
                antennas[antenna_num] = ant
                seen_tiles[tile_name] = antenna_num

        return antennas

    except Exception as e:
        raise ValueError(f"Failed to read MWA metafits file: {e}")


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
                            "BeamID": None,  # No beam ID info in simple format
                            "Position": (x, y, z),
                        }

                        antennas[ant_idx] = ant
                        ant_idx += 1

                    except ValueError:
                        continue

    except Exception as e:
        raise ValueError(f"Failed to read pyuvdata format file '{file_path}': {e}")

    return antennas


def format_antenna_data(antennas_dict):
    """
    Convert antenna dict to array format for BeamManager.

    Parameters:
    -----------
    antennas_dict : dict
        Dictionary with antenna numbers as keys and metadata dicts as values.
        Format: {0: {"Name": "ant1", "Number": 0, "BeamID": "beam1", "Position": (e, n, u)}, ...}

    Returns:
    --------
    dict with keys:
        - "names": np.ndarray of antenna names
        - "numbers": np.ndarray of antenna numbers
        - "positions_m": np.ndarray of positions (N_ant, 3)
        - "beam_ids": np.ndarray of beam IDs or None if no beam IDs present
        - "diameters": np.ndarray of diameters or None if no diameters present
    """
    if not antennas_dict:
        raise ValueError("Empty antenna dictionary provided")

    # Sort by antenna number for consistent ordering
    sorted_items = sorted(antennas_dict.items(), key=lambda x: x[0])

    # Extract arrays
    names = []
    numbers = []
    positions = []
    beam_ids = []
    diameters = []
    has_beam_ids = False
    has_diameters = False

    for ant_num, ant_data in sorted_items:
        names.append(ant_data["Name"])
        numbers.append(ant_data["Number"])
        positions.append(ant_data["Position"])

        # Check for BeamID
        beam_id = ant_data.get("BeamID")
        if beam_id is not None:
            has_beam_ids = True
            beam_ids.append(str(beam_id))  # Convert to string for consistency
        else:
            beam_ids.append(None)

        # Check for diameter
        diameter = ant_data.get("diameter")
        if diameter is not None:
            has_diameters = True
            diameters.append(diameter)
        else:
            diameters.append(None)

    # Convert to numpy arrays
    formatted_data = {
        "names": np.array(names, dtype=str),
        "numbers": np.array(numbers, dtype=int),
        "positions_m": np.array(positions, dtype=float),
        "beam_ids": np.array(beam_ids, dtype=str) if has_beam_ids else None,
        "diameters": np.array(diameters, dtype=float) if has_diameters else None,
    }

    return formatted_data


def read_antenna_positions(file_path, format_type="rrivis", return_format="dict"):
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
    return_format : str, optional
        Output format. Options:
        - "dict" (default): Legacy dict format {ant_num: {"Name": ..., "Position": ...}}
        - "arrays": Array format for BeamManager {"names": ndarray, "positions_m": ndarray, ...}

    Returns:
    --------
    dict: Dictionary with antenna data in the specified format.

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

        # Debug output (use dict format for printing)
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

        # Convert to requested format
        if return_format == "arrays":
            return format_antenna_data(antennas)
        elif return_format == "dict":
            return antennas
        else:
            raise ValueError(f"Invalid return_format: {return_format}. Use 'dict' or 'arrays'.")

    except Exception as e:
        raise ValueError(f"Failed to read antenna positions file '{file_path}' with format '{format_type}': {e}")


# Backward compatibility function
def read_antenna_positions_legacy(file_path):
    """
    Legacy function for backward compatibility.
    Uses the original RRIvis format.
    """
    return read_antenna_positions(file_path, "rrivis")