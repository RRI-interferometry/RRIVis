"""Input/Output modules for RadioSim.

This module handles configuration loading, data reading/writing,
and file format conversions.

Submodules
----------
config
    Pydantic-based configuration management.
writers
    Data output writers (HDF5, YAML).
measurement_set
    CASA Measurement Set I/O (requires python-casacore).
"""

# Import order is deliberate: beam_config subclasses StrictFrozenModel from
# config, while config completes the bidirectional type boundary lazily.
# ruff: noqa: I001

from importlib.util import find_spec
from typing import Any

# Resolved configuration values
from radiosim.core.runtime_config import (
    ConfigurationProvenance,
    PathResolutionProvenance,
    ResolvedConfiguration,
    ResolvedExecutionConfig,
    ResolvedFrequencyConfig,
    ResolvedObservationConfig,
    ResolvedSimulationConfig,
    ResolvedSkyModelConfig,
    ResolvedSkySourceRequest,
)

# Configuration management
from radiosim.io.config import (
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
from radiosim.io.beam_config import BeamsConfig
from radiosim.io.config_resolution import (
    ConfigOverrideError,
    ConfigParseError,
    ConfigPathError,
    ConfigResolutionError,
    ConfigSchemaError,
    ConfigSemanticError,
    ConfigSourceError,
    ConfigurationSource,
    InstrumentSourcePathOverride,
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


# Measurement Set imports stay lazy so configuration-only imports do not load
# pyuvdata or other optional scientific I/O dependencies.
def write_ms(*args: Any, **kwargs: Any) -> Any:
    from radiosim.io.measurement_set import write_ms as implementation

    return implementation(*args, **kwargs)


def read_ms(*args: Any, **kwargs: Any) -> Any:
    from radiosim.io.measurement_set import read_ms as implementation

    return implementation(*args, **kwargs)


def read_ms_dask(*args: Any, **kwargs: Any) -> Any:
    from radiosim.io.measurement_set import read_ms_dask as implementation

    return implementation(*args, **kwargs)


def ms_info(*args: Any, **kwargs: Any) -> Any:
    from radiosim.io.measurement_set import ms_info as implementation

    return implementation(*args, **kwargs)


PYUVDATA_AVAILABLE = find_spec("pyuvdata") is not None
CASACORE_AVAILABLE = find_spec("casacore") is not None
DASKMS_AVAILABLE = find_spec("daskms") is not None
MS_AVAILABLE = PYUVDATA_AVAILABLE and CASACORE_AVAILABLE


__all__ = [
    # Configuration
    "RadioSimConfig",
    "load_config",
    "resolve_config",
    "dump_config",
    "create_default_config",
    "BeamsConfig",
    "SkyModelConfig",
    "FrequencyGridConfig",
    "ExplicitFrequencyConfig",
    "ExecutionConfig",
    "PrecisionInput",
    "CliWorkflowConfig",
    "ConfigurationSource",
    "InstrumentSourcePathOverride",
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
