# rrivis/core/antenna.py
"""
Antenna position reading and coordinate conversion utilities.

This module provides functionality for reading antenna positions from various
file formats commonly used in radio astronomy, and converting between different
coordinate systems. All positions are internally converted to ENU (East-North-Up)
coordinates for consistency throughout RRIvis.

Supported File Formats
----------------------
- **rrivis**: Native RRIvis text format with optional BeamID and diameter columns
- **casa**: CASA configuration files (.cfg) used by ALMA, VLA, etc.
- **measurement_set**: CASA Measurement Set format (requires pyuvdata)
- **uvfits**: UVFITS format (requires pyuvdata)
- **mwa**: MWA metafits FITS files with TILEDATA extension
- **pyuvdata**: Simple x, y, z text format

Coordinate Systems
------------------
- **ENU**: East-North-Up local tangent plane (default output)
- **ITRF/XYZ**: Earth-Centered Earth-Fixed geocentric coordinates
- **LOC**: Local tangent plane (treated as ENU-like)

Output Data Structure
---------------------
Each antenna is represented as a dictionary with the following keys:

- ``Name`` (str): Human-readable antenna identifier
- ``Number`` (int): Numeric antenna index
- ``BeamID`` (str, int, or None): Beam pattern identifier for per-antenna beams
- ``Position`` (tuple): (East, North, Up) coordinates in meters
- ``diameter`` (float, optional): Antenna diameter in meters

Examples
--------
Basic usage with RRIvis format:

>>> from rrivis.core.antenna import read_antenna_positions
>>> antennas = read_antenna_positions("antennas.txt")
>>> print(f"Loaded {len(antennas)} antennas")
>>> print(antennas[0]["Position"])  # (E, N, U) in meters

Reading CASA configuration file:

>>> antennas = read_antenna_positions("alma.cfg", format_type="casa")

Getting array format for BeamManager:

>>> antennas = read_antenna_positions("antennas.txt", return_format="arrays")
>>> print(antennas["positions_m"].shape)  # (N_antennas, 3)

See Also
--------
rrivis.core.baseline : Baseline generation from antenna pairs
rrivis.core.beams : Beam pattern calculations using antenna properties

References
----------
.. [1] CASA Configuration File Format:
       https://casaguides.nrao.edu/index.php/Antenna_Configurations
.. [2] pyuvdata Documentation:
       https://pyuvdata.readthedocs.io/
.. [3] MWA Metafits Format:
       https://wiki.mwatelescope.org/display/MP/MWA+metafits+file+format
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# =============================================================================
# OPTIONAL DEPENDENCY IMPORTS
# =============================================================================

# pyuvdata is required for Measurement Set and UVFITS support
try:
    from pyuvdata import UVData
    PYUVDATA_AVAILABLE = True
except ImportError:
    PYUVDATA_AVAILABLE = False
    warnings.warn(
        "pyuvdata not available. Measurement Set and UVFITS format support disabled. "
        "Install with: pip install pyuvdata"
    )

# astropy is required for coordinate conversions and FITS file handling
try:
    from astropy.coordinates import EarthLocation
    from astropy import units as u
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    warnings.warn(
        "astropy not fully available. Some coordinate conversions and MWA format disabled. "
        "Install with: pip install astropy"
    )


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Default reference location for ITRF->ENU conversion (HERA site, South Africa)
DEFAULT_REFERENCE_LOCATION = (-30.72152777777791, 21.428305555555557, 1073.0)

# Supported file format identifiers
SUPPORTED_FORMATS = ["rrivis", "casa", "measurement_set", "uvfits", "mwa", "pyuvdata"]


# =============================================================================
# COORDINATE CONVERSION UTILITIES
# =============================================================================

def _convert_coordinates_to_enu(
    positions: np.ndarray,
    coordsys: str,
    reference_location: Optional[Tuple[float, float, float]] = None
) -> np.ndarray:
    """
    Convert antenna positions from various coordinate systems to ENU.

    ENU (East-North-Up) is a local tangent plane coordinate system centered
    at a reference location on Earth's surface. This is the standard coordinate
    system used internally by RRIvis for all antenna position calculations.

    Parameters
    ----------
    positions : np.ndarray
        Antenna positions in the source coordinate system.
        Shape: (N_antennas, 3) or (3,) for a single antenna.
    coordsys : str
        Source coordinate system identifier. Supported values:
        - 'ENU': Already in ENU, no conversion needed
        - 'XYZ' or 'ITRF': Earth-Centered Earth-Fixed geocentric coordinates
        - 'LOC': Local tangent plane (assumed ENU-like)
    reference_location : tuple of float, optional
        Reference location for coordinate transformation as (latitude, longitude, height)
        where latitude and longitude are in degrees and height is in meters.
        If None, defaults to HERA site coordinates.

    Returns
    -------
    np.ndarray
        Positions converted to ENU coordinates in meters.
        Shape matches input shape.

    Notes
    -----
    The ITRF to ENU transformation uses a rotation matrix derived from the
    reference location's latitude and longitude:

    .. math::

        R = \\begin{bmatrix}
            -\\sin(\\lambda) & \\cos(\\lambda) & 0 \\\\
            -\\sin(\\phi)\\cos(\\lambda) & -\\sin(\\phi)\\sin(\\lambda) & \\cos(\\phi) \\\\
            \\cos(\\phi)\\cos(\\lambda) & \\cos(\\phi)\\sin(\\lambda) & \\sin(\\phi)
        \\end{bmatrix}

    where :math:`\\phi` is latitude and :math:`\\lambda` is longitude.

    Warnings
    --------
    If astropy is not available, ITRF coordinates are returned unchanged
    with a warning.

    Examples
    --------
    >>> positions_itrf = np.array([[5000000, 2000000, -3000000]])
    >>> positions_enu = _convert_coordinates_to_enu(
    ...     positions_itrf, 'ITRF', reference_location=(-30.7, 21.4, 1000.0)
    ... )
    """
    positions = np.array(positions)
    coordsys_upper = coordsys.upper()

    # ENU coordinates need no conversion
    if coordsys_upper == 'ENU':
        return positions

    # Handle ITRF/XYZ geocentric coordinates
    if coordsys_upper in ['XYZ', 'ITRF']:
        if reference_location is None:
            warnings.warn(
                "No reference location provided for ITRF->ENU conversion. "
                f"Using default HERA site: {DEFAULT_REFERENCE_LOCATION}"
            )
            reference_location = DEFAULT_REFERENCE_LOCATION

        lat, lon, height = reference_location

        if not ASTROPY_AVAILABLE:
            warnings.warn(
                "astropy not available for ITRF->ENU conversion. "
                "Returning original coordinates unchanged."
            )
            return positions

        # Import here to avoid issues if astropy is partially available
        from astropy.coordinates import spherical_to_cartesian

        # Convert reference location to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # Build rotation matrix from ITRF to ENU
        # This matrix rotates geocentric XYZ to local East-North-Up
        sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
        sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)

        rotation_matrix = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ])

        # Calculate reference position in ITRF coordinates
        # Using Earth radius + height for approximate geocentric distance
        earth_radius = 6371000.0  # meters
        ref_xyz = spherical_to_cartesian(
            height + earth_radius,
            np.pi / 2 - lat_rad,  # co-latitude
            lon_rad
        )[0]

        # Convert each position: subtract reference, then rotate
        enu_positions = []
        for pos in np.atleast_2d(positions):
            relative_pos = pos - ref_xyz
            enu = rotation_matrix @ relative_pos
            enu_positions.append(enu)

        return np.array(enu_positions).squeeze()

    # LOC and unknown coordinate systems: assume ENU-like
    if coordsys_upper != 'LOC':
        warnings.warn(
            f"Unknown coordinate system '{coordsys}'. "
            "Assuming positions are already in ENU-like coordinates."
        )

    return positions


# =============================================================================
# FORMAT-SPECIFIC READERS
# =============================================================================

def read_rrivis_format(file_path: Union[str, Path]) -> Dict[int, Dict[str, Any]]:
    """
    Read antenna positions from RRIvis native text format.

    The RRIvis format is a flexible whitespace-separated text format with
    optional columns for BeamID and antenna diameter.

    Parameters
    ----------
    file_path : str or Path
        Path to the antenna positions file.

    Returns
    -------
    dict
        Dictionary mapping antenna numbers to antenna data dictionaries.
        Each antenna dict contains: Name, Number, BeamID, Position, and
        optionally diameter.

    Raises
    ------
    ValueError
        If the file contains invalid data or has fewer columns than expected.

    Notes
    -----
    **File Format:**

    The file should have a header row followed by data rows. Comments starting
    with '#' are ignored. Two column layouts are supported:

    *Without BeamID (5+ columns):*

    .. code-block:: text

        Name  Number  E        N        U        [Diameter]
        ANT0  0       100.5    -50.2    3.1      14.0
        ANT1  1       200.3    -30.1    2.8

    *With BeamID (6+ columns):*

    .. code-block:: text

        Name  Number  BeamID  E        N        U        [Diameter]
        ANT0  0       beam_a  100.5    -50.2    3.1      14.0
        ANT1  1       beam_b  200.3    -30.1    2.8

    The presence of a 'BeamID' column is auto-detected from the header.
    Diameter column is optional and extracted if present in header.

    Examples
    --------
    >>> antennas = read_rrivis_format("antenna_layout.txt")
    >>> print(antennas[0])
    {'Name': 'ANT0', 'Number': 0, 'BeamID': None, 'Position': (100.5, -50.2, 3.1)}
    """
    antennas = {}

    with open(file_path, "r") as f:
        lines = f.readlines()

    # --- Phase 1: Parse header to determine column layout ---
    header_idx = None
    has_beamid_col = False
    diameter_col_idx = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        # Skip empty lines and comments when looking for header
        if stripped and not stripped.startswith('#'):
            header_idx = idx
            header_tokens = stripped.split()

            # Detect optional columns from header
            for col_idx, token in enumerate(header_tokens):
                token_lower = token.lower()
                if token_lower == "beamid":
                    has_beamid_col = True
                if token_lower == "diameter":
                    diameter_col_idx = col_idx
            break

    # Calculate minimum required columns based on detected format
    # With BeamID: Name, Number, BeamID, E, N, U = 6 columns
    # Without BeamID: Name, Number, E, N, U = 5 columns
    min_cols = 6 if has_beamid_col else 5

    # --- Phase 2: Parse data rows ---
    for line_num, line in enumerate(lines):
        stripped = line.strip()

        # Skip header row, empty lines, and comments
        if line_num == header_idx or not stripped or stripped.startswith('#'):
            continue

        parts = stripped.split()

        # Validate column count
        if len(parts) < min_cols:
            raise ValueError(
                f"Invalid antenna position in line {line_num + 1}: "
                f"expected at least {min_cols} columns, got {len(parts)}. "
                f"Line content: '{stripped}'"
            )

        try:
            # Extract common fields
            name = parts[0]
            number = int(parts[1])

            # Extract position and optional BeamID based on format
            if has_beamid_col:
                # Format: Name Number BeamID E N U [Diameter]
                beam_id_raw = parts[2]
                # Try to convert BeamID to int, keep as string otherwise
                try:
                    beam_id = int(beam_id_raw)
                except ValueError:
                    beam_id = beam_id_raw
                e, n, u = float(parts[3]), float(parts[4]), float(parts[5])
            else:
                # Format: Name Number E N U [Diameter]
                beam_id = None
                e, n, u = float(parts[2]), float(parts[3]), float(parts[4])

            # Build antenna dictionary
            ant = {
                "Name": name,
                "Number": number,
                "BeamID": beam_id,
                "Position": (e, n, u),
            }

            # Extract optional diameter if column exists and value is present
            if diameter_col_idx is not None and len(parts) > diameter_col_idx:
                try:
                    ant["diameter"] = float(parts[diameter_col_idx])
                except (ValueError, IndexError):
                    pass  # Diameter is optional, skip if not parseable

            antennas[number] = ant

        except ValueError as e:
            raise ValueError(
                f"Could not parse data in line {line_num + 1}: '{stripped}'. "
                f"Error: {e}"
            )

    return antennas


def read_casa_format(file_path: Union[str, Path]) -> Dict[int, Dict[str, Any]]:
    """
    Read antenna positions from CASA configuration files (.cfg).

    CASA configuration files are used by ALMA, VLA, and other observatories
    to define antenna array layouts. The format includes header comments
    with metadata and whitespace-separated position data.

    Parameters
    ----------
    file_path : str or Path
        Path to the CASA configuration file.

    Returns
    -------
    dict
        Dictionary mapping antenna numbers to antenna data dictionaries.

    Notes
    -----
    **File Format:**

    .. code-block:: text

        #observatory=ALMA
        #COFA=-67.75,-23.02
        #coordsys=LOC
        # x             y               z             diam  station  ant
        -5.850273514   -125.9985379    -1.590364043   12.   A058     DA41
        10.123456      200.789012      0.5            12.   A059     DA42

    **Header Comments:**

    - ``#observatory=``: Observatory name (informational)
    - ``#COFA=``: Center of array coordinates (not currently used)
    - ``#coordsys=``: Coordinate system (LOC, ENU, or XYZ)

    **Data Columns:**

    1. X/East coordinate (meters)
    2. Y/North coordinate (meters)
    3. Z/Up coordinate (meters)
    4. Diameter (meters, optional)
    5. Station name (optional)
    6. Antenna name (optional)

    Examples
    --------
    >>> antennas = read_casa_format("alma.cfg")
    >>> print(f"Loaded {len(antennas)} antennas")
    """
    antennas = {}

    with open(file_path, "r") as f:
        lines = f.readlines()

    # --- Parse header for coordinate system ---
    coordsys = 'LOC'  # Default to local tangent plane

    for line in lines:
        if line.startswith('#coordsys='):
            coordsys_str = line.split('=')[1].strip().upper()
            if 'LOC' in coordsys_str or 'ENU' in coordsys_str:
                coordsys = 'ENU'
            elif 'XYZ' in coordsys_str:
                coordsys = 'XYZ'
            # Otherwise keep default 'LOC'

    # --- Parse antenna data rows ---
    ant_idx = 0

    for line in lines:
        stripped = line.strip()

        # Skip comments and empty lines
        if stripped.startswith('#') or not stripped:
            continue

        parts = stripped.split()

        # Need at least x, y, z coordinates
        if len(parts) < 3:
            continue

        try:
            # Parse coordinates
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])

            # Parse optional diameter (column 4)
            # Check if it looks like a number (handles "12." format)
            diameter = None
            if len(parts) > 3:
                diameter_str = parts[3].replace('.', '', 1)
                if diameter_str.replace('-', '').isdigit():
                    diameter = float(parts[3])

            # Parse optional station and antenna names
            station = parts[4] if len(parts) > 4 else f"A{ant_idx:03d}"
            name = parts[5] if len(parts) > 5 else station

            # Convert coordinates to ENU
            # For LOC/ENU coordinate systems, positions are already usable
            if coordsys in ['ENU', 'LOC']:
                e, n, u = x, y, z
            else:
                # For XYZ/ITRF, would need proper conversion
                # Currently treating as ENU-like (may need enhancement)
                e, n, u = x, y, z

            # Build antenna dictionary
            ant = {
                "Name": name,
                "Number": ant_idx,
                "BeamID": None,  # CASA format doesn't include beam IDs
                "Position": (e, n, u),
            }

            if diameter is not None:
                ant["diameter"] = diameter

            antennas[ant_idx] = ant
            ant_idx += 1

        except ValueError:
            # Skip lines that can't be parsed as antenna data
            continue

    return antennas


def read_measurement_set(file_path: Union[str, Path]) -> Dict[int, Dict[str, Any]]:
    """
    Read antenna positions from a CASA Measurement Set.

    Measurement Sets are the standard data format for radio interferometry
    observations. This function extracts antenna metadata from the ANTENNA
    subtable using pyuvdata.

    Parameters
    ----------
    file_path : str or Path
        Path to the Measurement Set directory (typically ending in .ms).

    Returns
    -------
    dict
        Dictionary mapping antenna numbers to antenna data dictionaries.

    Raises
    ------
    ImportError
        If pyuvdata is not installed.
    ValueError
        If the Measurement Set cannot be read or contains no antenna data.

    Notes
    -----
    This function requires pyuvdata to be installed. Install with:

    .. code-block:: bash

        pip install pyuvdata

    The antenna positions are extracted from the UVData object's
    ``antenna_positions`` attribute, which provides positions in ENU
    coordinates relative to the array center.

    Examples
    --------
    >>> antennas = read_measurement_set("observation.ms")
    >>> print(antennas[0]["Name"])
    'DA41'
    """
    if not PYUVDATA_AVAILABLE:
        raise ImportError(
            "pyuvdata is required for Measurement Set support. "
            "Install with: pip install pyuvdata"
        )

    try:
        # Read the Measurement Set using pyuvdata
        uv = UVData()
        uv.read(file_path)

        antennas = {}

        # Extract antenna metadata arrays
        ant_names = uv.antenna_names
        ant_numbers = uv.antenna_numbers
        ant_positions = uv.antenna_positions  # ENU relative to array center
        ant_diameters = uv.antenna_diameters if hasattr(uv, 'antenna_diameters') else [0] * len(ant_names)

        # Build antenna dictionaries
        for name, number, pos, diam in zip(ant_names, ant_numbers, ant_positions, ant_diameters):
            ant = {
                "Name": str(name),
                "Number": int(number),
                "BeamID": None,  # MS format doesn't include beam IDs
                "Position": tuple(pos),
            }

            if diam > 0:
                ant["diameter"] = float(diam)

            antennas[int(number)] = ant

        return antennas

    except Exception as e:
        raise ValueError(f"Failed to read Measurement Set '{file_path}': {e}")


def read_uvfits(file_path: Union[str, Path]) -> Dict[int, Dict[str, Any]]:
    """
    Read antenna positions from a UVFITS file.

    UVFITS is a FITS-based format for storing visibility data and metadata.
    This function extracts antenna information from the AN (antenna) table
    using pyuvdata.

    Parameters
    ----------
    file_path : str or Path
        Path to the UVFITS file.

    Returns
    -------
    dict
        Dictionary mapping antenna numbers to antenna data dictionaries.

    Raises
    ------
    ImportError
        If pyuvdata is not installed.
    ValueError
        If the UVFITS file cannot be read or contains no antenna data.

    Notes
    -----
    This function requires pyuvdata to be installed. Install with:

    .. code-block:: bash

        pip install pyuvdata

    See Also
    --------
    read_measurement_set : Similar function for CASA MS format.

    Examples
    --------
    >>> antennas = read_uvfits("observation.uvfits")
    """
    if not PYUVDATA_AVAILABLE:
        raise ImportError(
            "pyuvdata is required for UVFITS support. "
            "Install with: pip install pyuvdata"
        )

    try:
        # Read the UVFITS file using pyuvdata
        uv = UVData()
        uv.read(file_path)

        antennas = {}

        # Extract antenna metadata
        ant_names = uv.antenna_names
        ant_numbers = uv.antenna_numbers
        ant_positions = uv.antenna_positions
        ant_diameters = uv.antenna_diameters if hasattr(uv, 'antenna_diameters') else [0] * len(ant_names)

        # Build antenna dictionaries
        for name, number, pos, diam in zip(ant_names, ant_numbers, ant_positions, ant_diameters):
            ant = {
                "Name": str(name),
                "Number": int(number),
                "BeamID": None,
                "Position": tuple(pos),
            }

            if diam > 0:
                ant["diameter"] = float(diam)

            antennas[int(number)] = ant

        return antennas

    except Exception as e:
        raise ValueError(f"Failed to read UVFITS file '{file_path}': {e}")


def read_mwa_format(file_path: Union[str, Path]) -> Dict[int, Dict[str, Any]]:
    """
    Read antenna positions from an MWA metafits file.

    MWA (Murchison Widefield Array) metafits files are FITS files containing
    observation metadata including tile (antenna) positions in the TILEDATA
    extension.

    Parameters
    ----------
    file_path : str or Path
        Path to the MWA metafits FITS file.

    Returns
    -------
    dict
        Dictionary mapping antenna numbers to antenna data dictionaries.

    Raises
    ------
    ImportError
        If astropy is not installed.
    ValueError
        If the file is not a valid MWA metafits file or lacks TILEDATA.

    Notes
    -----
    MWA metafits files contain 256 rows in the TILEDATA table (2 per tile
    for X and Y polarization). This function deduplicates to return 128
    unique tile positions.

    The TILEDATA columns used are:

    - ``TileName``: Tile identifier string
    - ``Antenna``: Numeric antenna/tile index
    - ``East``, ``North``, ``Height``: ENU coordinates in meters

    Examples
    --------
    >>> antennas = read_mwa_format("1234567890_metafits.fits")
    >>> print(f"Loaded {len(antennas)} MWA tiles")
    Loaded 128 MWA tiles
    """
    if not ASTROPY_AVAILABLE:
        raise ImportError(
            "astropy is required to read MWA metafits FITS files. "
            "Install with: pip install astropy"
        )

    antennas = {}

    try:
        with fits.open(file_path) as hdul:
            # Verify TILEDATA extension exists
            if 'TILEDATA' not in hdul:
                raise ValueError(
                    f"TILEDATA extension not found in '{file_path}'. "
                    "This may not be a valid MWA metafits file."
                )

            tile_table = hdul['TILEDATA'].data

            # Track processed tiles to handle X/Y polarization deduplication
            seen_tiles = {}

            for row in tile_table:
                tile_name = row['TileName'].strip()
                antenna_num = int(row['Antenna'])

                # Skip duplicate entries (X and Y polarization rows)
                if tile_name in seen_tiles:
                    continue

                # Extract ENU coordinates
                east = float(row['East'])
                north = float(row['North'])
                height = float(row['Height'])

                ant = {
                    "Name": tile_name,
                    "Number": antenna_num,
                    "BeamID": None,  # MWA uses separate beam models
                    "Position": (east, north, height),
                }

                antennas[antenna_num] = ant
                seen_tiles[tile_name] = antenna_num

        return antennas

    except KeyError as e:
        raise ValueError(f"Missing required column in TILEDATA: {e}")
    except Exception as e:
        raise ValueError(f"Failed to read MWA metafits file '{file_path}': {e}")


def read_pyuvdata_format(file_path: Union[str, Path]) -> Dict[int, Dict[str, Any]]:
    """
    Read antenna positions from a simple text coordinate file.

    This format supports basic x, y, z coordinate files with one antenna
    per line. It's useful for quick testing or custom antenna layouts.

    Parameters
    ----------
    file_path : str or Path
        Path to the coordinate file.

    Returns
    -------
    dict
        Dictionary mapping antenna numbers to antenna data dictionaries.
        Antenna names are auto-generated as ANT000, ANT001, etc.

    Notes
    -----
    **File Format:**

    Simple whitespace-separated x, y, z coordinates, one per line.
    Lines starting with '#' are treated as comments and ignored.

    .. code-block:: text

        # Optional comment
        100.5  -50.2  3.1
        200.3  -30.1  2.8
        150.0   10.5  4.2

    Additional columns beyond the first three are ignored.

    Examples
    --------
    >>> antennas = read_pyuvdata_format("positions.txt")
    >>> print(antennas[0]["Name"])
    'ANT000'
    """
    antennas = {}

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        ant_idx = 0

        for line in lines:
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue

            parts = stripped.split()

            # Need at least x, y, z coordinates
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])

                    ant = {
                        "Name": f"ANT{ant_idx:03d}",
                        "Number": ant_idx,
                        "BeamID": None,
                        "Position": (x, y, z),
                    }

                    antennas[ant_idx] = ant
                    ant_idx += 1

                except ValueError:
                    # Skip lines with non-numeric data
                    continue

    except Exception as e:
        raise ValueError(f"Failed to read coordinate file '{file_path}': {e}")

    return antennas


# =============================================================================
# DATA FORMAT CONVERSION
# =============================================================================

def format_antenna_data(antennas_dict: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert antenna dictionary to NumPy array format.

    This function transforms the dictionary-based antenna representation
    into a format with NumPy arrays, which is more efficient for numerical
    operations and required by components like BeamManager.

    Parameters
    ----------
    antennas_dict : dict
        Dictionary mapping antenna numbers to antenna metadata dictionaries.
        Each antenna dict must contain: Name, Number, Position.
        Optional keys: BeamID, diameter.

    Returns
    -------
    dict
        Dictionary with NumPy arrays:

        - ``names`` (ndarray of str): Antenna name strings, shape (N_ant,)
        - ``numbers`` (ndarray of int): Antenna numbers, shape (N_ant,)
        - ``positions_m`` (ndarray of float): ENU positions, shape (N_ant, 3)
        - ``beam_ids`` (ndarray of str or None): Beam IDs if any antenna has one
        - ``diameters`` (ndarray of float or None): Diameters if any antenna has one

    Raises
    ------
    ValueError
        If the input dictionary is empty.

    Notes
    -----
    The output arrays are sorted by antenna number for consistent ordering.
    This is important for baseline indexing and correlation with other data.

    If any antenna has a BeamID or diameter, the corresponding array is
    created for all antennas (with None values where not specified).

    Examples
    --------
    >>> antennas = read_antenna_positions("layout.txt")
    >>> arrays = format_antenna_data(antennas)
    >>> print(arrays["positions_m"].shape)
    (64, 3)
    >>> print(arrays["names"][:3])
    ['ANT000' 'ANT001' 'ANT002']
    """
    if not antennas_dict:
        raise ValueError("Empty antenna dictionary provided")

    # Sort by antenna number for consistent ordering
    sorted_items = sorted(antennas_dict.items(), key=lambda x: x[0])

    # Initialize collection lists
    names = []
    numbers = []
    positions = []
    beam_ids = []
    diameters = []

    # Track which optional fields are present
    has_beam_ids = False
    has_diameters = False

    # Extract data from each antenna
    for ant_num, ant_data in sorted_items:
        names.append(ant_data["Name"])
        numbers.append(ant_data["Number"])
        positions.append(ant_data["Position"])

        # Handle optional BeamID
        beam_id = ant_data.get("BeamID")
        if beam_id is not None:
            has_beam_ids = True
            beam_ids.append(str(beam_id))
        else:
            beam_ids.append(None)

        # Handle optional diameter
        diameter = ant_data.get("diameter")
        if diameter is not None:
            has_diameters = True
            diameters.append(diameter)
        else:
            diameters.append(None)

    # Convert to NumPy arrays
    formatted_data = {
        "names": np.array(names, dtype=str),
        "numbers": np.array(numbers, dtype=int),
        "positions_m": np.array(positions, dtype=float),
        "beam_ids": np.array(beam_ids, dtype=object) if has_beam_ids else None,
        "diameters": np.array(diameters, dtype=float) if has_diameters else None,
    }

    return formatted_data


# =============================================================================
# MAIN PUBLIC API
# =============================================================================

def read_antenna_positions(
    file_path: Union[str, Path],
    format_type: str = "rrivis",
    return_format: str = "dict",
    verbose: bool = False
) -> Union[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """
    Read antenna positions from various file formats.

    This is the main entry point for loading antenna layout data into RRIvis.
    It supports multiple file formats and can return data in either dictionary
    or array format.

    Parameters
    ----------
    file_path : str or Path
        Path to the antenna positions file.
    format_type : str, default="rrivis"
        File format identifier. Supported values:

        - ``"rrivis"``: Native RRIvis text format (default)
        - ``"casa"``: CASA configuration files (.cfg)
        - ``"measurement_set"``: CASA Measurement Set (.ms)
        - ``"uvfits"``: UVFITS format
        - ``"mwa"``: MWA metafits FITS files
        - ``"pyuvdata"``: Simple x, y, z text format

    return_format : str, default="dict"
        Output format:

        - ``"dict"``: Dictionary mapping antenna numbers to metadata dicts
        - ``"arrays"``: NumPy arrays suitable for BeamManager

    verbose : bool, default=False
        If True, log debug information about loaded antennas and memory usage.

    Returns
    -------
    dict
        Antenna data in the requested format.

        If ``return_format="dict"``:
            ``{ant_num: {"Name": str, "Number": int, "BeamID": any,
            "Position": tuple, "diameter": float}, ...}``

        If ``return_format="arrays"``:
            ``{"names": ndarray, "numbers": ndarray, "positions_m": ndarray,
            "beam_ids": ndarray or None, "diameters": ndarray or None}``

    Raises
    ------
    ValueError
        If the file is empty, contains invalid data, or format is unsupported.
    FileNotFoundError
        If the specified file does not exist.
    ImportError
        If required dependencies are missing for the specified format.

    See Also
    --------
    read_rrivis_format : Details on RRIvis format
    read_casa_format : Details on CASA format
    read_measurement_set : Details on MS format
    format_antenna_data : Convert dict to array format

    Examples
    --------
    Load antennas from RRIvis format (default):

    >>> antennas = read_antenna_positions("antennas.txt")
    >>> print(f"Loaded {len(antennas)} antennas")
    >>> print(antennas[0]["Position"])

    Load from CASA configuration:

    >>> antennas = read_antenna_positions("alma.cfg", format_type="casa")

    Get array format for numerical operations:

    >>> arrays = read_antenna_positions("antennas.txt", return_format="arrays")
    >>> positions = arrays["positions_m"]  # Shape: (N_ant, 3)

    Load from Measurement Set:

    >>> antennas = read_antenna_positions("obs.ms", format_type="measurement_set")
    """
    # --- Input validation ---
    if not file_path:
        raise ValueError("Antenna positions file path is not provided.")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    format_lower = format_type.lower()
    if format_lower not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported antenna file format: '{format_type}'. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    # --- Dispatch to format-specific reader ---
    format_readers = {
        "rrivis": read_rrivis_format,
        "casa": read_casa_format,
        "measurement_set": read_measurement_set,
        "uvfits": read_uvfits,
        "mwa": read_mwa_format,
        "pyuvdata": read_pyuvdata_format,
    }

    try:
        antennas = format_readers[format_lower](file_path)

        if not antennas:
            raise ValueError("No valid antenna data found in file.")

        # --- Optional verbose logging ---
        if verbose:
            logger.debug(f"Loaded {len(antennas)} antennas from '{file_path}'")
            for idx, data in antennas.items():
                logger.debug(f"  Antenna {idx}: {data['Name']} at {data['Position']}")

            # Calculate approximate memory usage
            total_bytes = sys.getsizeof(antennas)
            for value in antennas.values():
                total_bytes += sys.getsizeof(value)
                total_bytes += sum(sys.getsizeof(v) for v in value.values())
            logger.debug(f"  Memory usage: {total_bytes / 1024:.2f} KB")

        # --- Convert to requested output format ---
        if return_format == "arrays":
            return format_antenna_data(antennas)
        elif return_format == "dict":
            return antennas
        else:
            raise ValueError(
                f"Invalid return_format: '{return_format}'. "
                "Use 'dict' or 'arrays'."
            )

    except (ImportError, ValueError, FileNotFoundError):
        # Re-raise these specific exceptions without wrapping
        raise
    except Exception as e:
        raise ValueError(
            f"Failed to read antenna positions from '{file_path}' "
            f"with format '{format_type}': {e}"
        )
