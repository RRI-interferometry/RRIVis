"""Input/Output modules for RadioSim.

This module handles configuration loading, data reading/writing,
and file format conversions.

Submodules
----------
config
    Pydantic-based configuration management.
writers
    Data output writers (HDF5, YAML).
antenna_readers
    Antenna layout file readers.
measurement_set
    CASA Measurement Set I/O (requires python-casacore).
"""

# Configuration management
# Antenna file readers
from radiosim.io.antenna_readers import (
    read_antenna_positions,
    read_casa_format,
    read_mwa_format,
    read_pyuvdata_format,
    read_radiosim_format,
)
from radiosim.io.config import (
    AntennaLayoutConfig,
    BeamsConfig,
    OutputConfig,
    RadioSimConfig,
    SkyModelConfig,
    create_default_config,
    load_config,
)

# Data writers
from radiosim.io.writers import (
    load_visibilities_hdf5,
    save_config_yaml,
    save_visibilities_hdf5,
)

# Measurement Set I/O (optional - requires python-casacore)
try:
    from radiosim.io.measurement_set import (
        CASACORE_AVAILABLE,
        DASKMS_AVAILABLE,
        PYUVDATA_AVAILABLE,
        ms_info,
        read_ms,
        read_ms_dask,
        write_ms,
    )

    MS_AVAILABLE = PYUVDATA_AVAILABLE and CASACORE_AVAILABLE
except ImportError:
    MS_AVAILABLE = False
    PYUVDATA_AVAILABLE = False
    CASACORE_AVAILABLE = False
    DASKMS_AVAILABLE = False

    def write_ms(*args, **kwargs):
        raise ImportError(
            "Measurement Set support not available. Install with:\n"
            "  pip install radiosim[ms]"
        )

    def read_ms(*args, **kwargs):
        raise ImportError(
            "Measurement Set support not available. Install with:\n"
            "  pip install radiosim[ms]"
        )

    def read_ms_dask(*args, **kwargs):
        raise ImportError(
            "Measurement Set support not available. Install with:\n"
            "  pip install dask-ms"
        )

    def ms_info(*args, **kwargs):
        raise ImportError(
            "Measurement Set support not available. Install with:\n"
            "  pip install radiosim[ms]"
        )


__all__ = [
    # Configuration
    "RadioSimConfig",
    "load_config",
    "create_default_config",
    "AntennaLayoutConfig",
    "BeamsConfig",
    "SkyModelConfig",
    "OutputConfig",
    # Writers
    "save_visibilities_hdf5",
    "save_config_yaml",
    "load_visibilities_hdf5",
    # Antenna readers
    "read_antenna_positions",
    "read_radiosim_format",
    "read_casa_format",
    "read_pyuvdata_format",
    "read_mwa_format",
    # Measurement Set I/O
    "write_ms",
    "read_ms",
    "read_ms_dask",
    "ms_info",
    "MS_AVAILABLE",
    "PYUVDATA_AVAILABLE",
    "CASACORE_AVAILABLE",
    "DASKMS_AVAILABLE",
]
