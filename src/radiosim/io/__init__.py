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

from collections.abc import Callable
from typing import Any, NoReturn

# Resolved configuration values
from radiosim.core.runtime_config import (
    ConfigurationProvenance,
    PathResolutionProvenance,
    ResolvedAntennaLayoutConfig,
    ResolvedBeamsConfig,
    ResolvedConfiguration,
    ResolvedExecutionConfig,
    ResolvedFrequencyConfig,
    ResolvedLocationConfig,
    ResolvedObservationConfig,
    ResolvedSimulationConfig,
    ResolvedSkyModelConfig,
    ResolvedSkySourceRequest,
    ResolvedTelescopeConfig,
)

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
    CliWorkflowConfig,
    ConfigIssue,
    ExecutionConfig,
    ExplicitFrequencyConfig,
    FrequencyGridConfig,
    PrecisionInput,
    RadioSimConfig,
    SkyModelConfig,
    create_default_config,
    dump_config,
    load_config,
)
from radiosim.io.config_resolution import (
    AntennaLayoutOverride,
    ConfigOverrideError,
    ConfigParseError,
    ConfigPathError,
    ConfigResolutionError,
    ConfigSchemaError,
    ConfigSemanticError,
    ConfigSourceError,
    ConfigurationSource,
    SimulationOverrides,
    UnsupportedConfigError,
    WorkflowOverrides,
    resolve_config,
)

# Data writers
from radiosim.io.writers import (
    load_visibilities_hdf5,
    save_visibilities_hdf5,
)

# Measurement Set I/O (optional - requires python-casacore)
write_ms: Callable[..., Any]
read_ms: Callable[..., Any]
read_ms_dask: Callable[..., Any]
ms_info: Callable[..., Any]

try:
    from radiosim.io.measurement_set import (
        CASACORE_AVAILABLE as _casacore_available,
    )
    from radiosim.io.measurement_set import (
        DASKMS_AVAILABLE as _daskms_available,
    )
    from radiosim.io.measurement_set import (
        PYUVDATA_AVAILABLE as _pyuvdata_available,
    )
    from radiosim.io.measurement_set import (
        ms_info as _ms_info,
    )
    from radiosim.io.measurement_set import (
        read_ms as _read_ms,
    )
    from radiosim.io.measurement_set import (
        read_ms_dask as _read_ms_dask,
    )
    from radiosim.io.measurement_set import (
        write_ms as _write_ms,
    )

    write_ms = _write_ms
    read_ms = _read_ms
    read_ms_dask = _read_ms_dask
    ms_info = _ms_info
except ImportError:
    _pyuvdata_available = False
    _casacore_available = False
    _daskms_available = False

    def _write_ms_unavailable(*args: Any, **kwargs: Any) -> NoReturn:
        raise ImportError(
            "Measurement Set support not available. Install with:\n"
            "  pip install radiosim[ms]"
        )

    def _read_ms_unavailable(*args: Any, **kwargs: Any) -> NoReturn:
        raise ImportError(
            "Measurement Set support not available. Install with:\n"
            "  pip install radiosim[ms]"
        )

    def _read_ms_dask_unavailable(*args: Any, **kwargs: Any) -> NoReturn:
        raise ImportError(
            "Measurement Set support not available. Install with:\n"
            "  pip install dask-ms"
        )

    def _ms_info_unavailable(*args: Any, **kwargs: Any) -> NoReturn:
        raise ImportError(
            "Measurement Set support not available. Install with:\n"
            "  pip install radiosim[ms]"
        )

    write_ms = _write_ms_unavailable
    read_ms = _read_ms_unavailable
    read_ms_dask = _read_ms_dask_unavailable
    ms_info = _ms_info_unavailable

PYUVDATA_AVAILABLE = bool(_pyuvdata_available)
CASACORE_AVAILABLE = bool(_casacore_available)
DASKMS_AVAILABLE = bool(_daskms_available)
MS_AVAILABLE = PYUVDATA_AVAILABLE and CASACORE_AVAILABLE


__all__ = [
    # Configuration
    "RadioSimConfig",
    "load_config",
    "resolve_config",
    "dump_config",
    "create_default_config",
    "AntennaLayoutConfig",
    "BeamsConfig",
    "SkyModelConfig",
    "FrequencyGridConfig",
    "ExplicitFrequencyConfig",
    "ExecutionConfig",
    "PrecisionInput",
    "CliWorkflowConfig",
    "ConfigurationSource",
    "AntennaLayoutOverride",
    "SimulationOverrides",
    "WorkflowOverrides",
    "ConfigIssue",
    "ConfigResolutionError",
    "ConfigSourceError",
    "ConfigParseError",
    "ConfigSchemaError",
    "ConfigOverrideError",
    "ConfigSemanticError",
    "UnsupportedConfigError",
    "ConfigPathError",
    "ConfigurationProvenance",
    "PathResolutionProvenance",
    "ResolvedTelescopeConfig",
    "ResolvedAntennaLayoutConfig",
    "ResolvedBeamsConfig",
    "ResolvedLocationConfig",
    "ResolvedObservationConfig",
    "ResolvedFrequencyConfig",
    "ResolvedSkySourceRequest",
    "ResolvedSkyModelConfig",
    "ResolvedExecutionConfig",
    "ResolvedSimulationConfig",
    "ResolvedConfiguration",
    # Writers
    "save_visibilities_hdf5",
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
