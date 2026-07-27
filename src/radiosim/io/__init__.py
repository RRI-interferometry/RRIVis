"""Input/Output modules for RadioSim.

This module handles configuration loading, data reading/writing,
and file format conversions.

Submodules
----------
config
    Pydantic-based configuration management.
writers
    Resolved-configuration YAML artifact writer.
hdf5
    Versioned canonical-result HDF5 I/O.
measurement_set
    CASA Measurement Set I/O (requires python-casacore).
"""

from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    from radiosim.io.beam_config import BeamsConfig
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
    from radiosim.io.hdf5 import (
        HDF5ReadLimits,
        load_result_hdf5,
        write_result_hdf5,
    )


_LAZY_EXPORTS = {
    "BeamsConfig": ("radiosim.io.beam_config", "BeamsConfig"),
    "RadioSimConfig": ("radiosim.io.config", "RadioSimConfig"),
    "load_config": ("radiosim.io.config", "load_config"),
    "dump_config": ("radiosim.io.config", "dump_config"),
    "create_default_config": ("radiosim.io.config", "create_default_config"),
    "SkyModelConfig": ("radiosim.io.config", "SkyModelConfig"),
    "FrequencyGridConfig": ("radiosim.io.config", "FrequencyGridConfig"),
    "ExplicitFrequencyConfig": ("radiosim.io.config", "ExplicitFrequencyConfig"),
    "ExecutionConfig": ("radiosim.io.config", "ExecutionConfig"),
    "PrecisionInput": ("radiosim.io.config", "PrecisionInput"),
    "CliWorkflowConfig": ("radiosim.io.config", "CliWorkflowConfig"),
    "ConfigIssue": ("radiosim.io.config", "ConfigIssue"),
    "resolve_config": ("radiosim.io.config_resolution", "resolve_config"),
    "ConfigurationSource": (
        "radiosim.io.config_resolution",
        "ConfigurationSource",
    ),
    "InstrumentSourcePathOverride": (
        "radiosim.io.config_resolution",
        "InstrumentSourcePathOverride",
    ),
    "SimulationOverrides": (
        "radiosim.io.config_resolution",
        "SimulationOverrides",
    ),
    "WorkflowOverrides": ("radiosim.io.config_resolution", "WorkflowOverrides"),
    "ConfigResolutionError": (
        "radiosim.io.config_resolution",
        "ConfigResolutionError",
    ),
    "ConfigSourceError": ("radiosim.io.config_resolution", "ConfigSourceError"),
    "ConfigParseError": ("radiosim.io.config_resolution", "ConfigParseError"),
    "ConfigSchemaError": ("radiosim.io.config_resolution", "ConfigSchemaError"),
    "ConfigOverrideError": (
        "radiosim.io.config_resolution",
        "ConfigOverrideError",
    ),
    "ConfigSemanticError": (
        "radiosim.io.config_resolution",
        "ConfigSemanticError",
    ),
    "UnsupportedConfigError": (
        "radiosim.io.config_resolution",
        "UnsupportedConfigError",
    ),
    "ConfigPathError": ("radiosim.io.config_resolution", "ConfigPathError"),
    "ConfigurationProvenance": (
        "radiosim.core.runtime_config",
        "ConfigurationProvenance",
    ),
    "PathResolutionProvenance": (
        "radiosim.core.runtime_config",
        "PathResolutionProvenance",
    ),
    "ResolvedObservationConfig": (
        "radiosim.core.runtime_config",
        "ResolvedObservationConfig",
    ),
    "ResolvedFrequencyConfig": (
        "radiosim.core.runtime_config",
        "ResolvedFrequencyConfig",
    ),
    "ResolvedSkySourceRequest": (
        "radiosim.core.runtime_config",
        "ResolvedSkySourceRequest",
    ),
    "ResolvedSkyModelConfig": (
        "radiosim.core.runtime_config",
        "ResolvedSkyModelConfig",
    ),
    "ResolvedExecutionConfig": (
        "radiosim.core.runtime_config",
        "ResolvedExecutionConfig",
    ),
    "ResolvedSimulationConfig": (
        "radiosim.core.runtime_config",
        "ResolvedSimulationConfig",
    ),
    "ResolvedConfiguration": (
        "radiosim.core.runtime_config",
        "ResolvedConfiguration",
    ),
    "HDF5ReadLimits": (
        "radiosim.io.hdf5",
        "HDF5ReadLimits",
    ),
    "write_result_hdf5": (
        "radiosim.io.hdf5",
        "write_result_hdf5",
    ),
    "load_result_hdf5": (
        "radiosim.io.hdf5",
        "load_result_hdf5",
    ),
}


def __getattr__(name: str) -> object:
    """Load public I/O objects only when they are first accessed."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public I/O objects in interactive discovery."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


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
    # Canonical HDF5 result I/O
    "HDF5ReadLimits",
    "write_result_hdf5",
    "load_result_hdf5",
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
