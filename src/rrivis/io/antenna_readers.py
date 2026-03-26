# rrivis/io/antenna_readers.py
"""Antenna file format readers.

This module provides functions for reading antenna layout files in various formats:
- RRIvis native format (.txt)
- CASA configuration format (.cfg)
- PyUVData simple text format (.txt)
- MWA metafits format (.fits)
- Measurement Set format (.ms)
- UVFITS format (.uvfits)

Note: The actual implementation is in rrivis.core.antenna.
This module re-exports those functions for cleaner I/O organization.
"""

# Re-export antenna reading functions from core module
from rrivis.core.antenna import (
    read_antenna_positions,
    read_casa_format,
    read_mwa_format,
    read_pyuvdata_format,
    read_rrivis_format,
)

__all__ = [
    "read_antenna_positions",
    "read_rrivis_format",
    "read_casa_format",
    "read_pyuvdata_format",
    "read_mwa_format",
]
