"""Input/Output modules for RadioSim.

This module handles configuration loading, data reading/writing,
and file format conversions.

Submodules
----------
config
    Pydantic-based configuration management.
workflow_artifacts
    Resolved-configuration YAML artifact writer for staged CLI runs.
hdf5
    Versioned canonical-result HDF5 I/O.
summary_json
    Bounded ``radiosim.result-summary`` JSON writer.
standard_visibility
    Canonical-to-standard visibility projection shared by MS and UVFITS.
measurement_set
    CASA Measurement Set I/O (requires python-casacore).
uvfits
    UVFITS I/O (requires pyuvdata and Astropy FITS).
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radiosim.core.runtime_config import (
        ConfigurationProvenance,
        PathResolutionProvenance,
        ResolvedConfiguration,
        ResolvedExecutionConfig,
        ResolvedFrequencyConfig,
        ResolvedObservationConfig,
        ResolvedSimulationConfig,
        ResolvedSkyLoadingConfig,
        ResolvedSkyModelConfig,
        ResolvedSkySourceRequest,
        ResolvedSolverExecutionConfig,
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
        SkyLoadingConfig,
        SkyModelConfig,
        SolverExecutionConfig,
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
    from radiosim.io.measurement_set import (
        read_measurement_set,
        write_measurement_set,
    )
    from radiosim.io.receptor_config import (
        ReceptorDefinitionConfig,
        ReceptorOverrideConfig,
        ReceptorsConfig,
    )
    from radiosim.io.result_format import ResultFormat
    from radiosim.io.standard_visibility import StandardVisibilityData
    from radiosim.io.summary_json import write_result_summary_json
    from radiosim.io.uvfits import read_uvfits, write_uvfits


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
    "SkyLoadingConfig": ("radiosim.io.config", "SkyLoadingConfig"),
    "SolverExecutionConfig": ("radiosim.io.config", "SolverExecutionConfig"),
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
    "ResolvedSkyLoadingConfig": (
        "radiosim.core.runtime_config",
        "ResolvedSkyLoadingConfig",
    ),
    "ResolvedSolverExecutionConfig": (
        "radiosim.core.runtime_config",
        "ResolvedSolverExecutionConfig",
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
    "ReceptorsConfig": ("radiosim.io.receptor_config", "ReceptorsConfig"),
    "ReceptorDefinitionConfig": (
        "radiosim.io.receptor_config",
        "ReceptorDefinitionConfig",
    ),
    "ReceptorOverrideConfig": (
        "radiosim.io.receptor_config",
        "ReceptorOverrideConfig",
    ),
    "ResultFormat": ("radiosim.io.result_format", "ResultFormat"),
    "write_result_summary_json": (
        "radiosim.io.summary_json",
        "write_result_summary_json",
    ),
    "StandardVisibilityData": (
        "radiosim.io.standard_visibility",
        "StandardVisibilityData",
    ),
    "write_measurement_set": (
        "radiosim.io.measurement_set",
        "write_measurement_set",
    ),
    "read_measurement_set": (
        "radiosim.io.measurement_set",
        "read_measurement_set",
    ),
    "write_uvfits": (
        "radiosim.io.uvfits",
        "write_uvfits",
    ),
    "read_uvfits": (
        "radiosim.io.uvfits",
        "read_uvfits",
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


__all__ = [
    # Configuration
    "RadioSimConfig",
    "load_config",
    "resolve_config",
    "dump_config",
    "create_default_config",
    "BeamsConfig",
    "ReceptorsConfig",
    "ReceptorDefinitionConfig",
    "ReceptorOverrideConfig",
    "SkyModelConfig",
    "FrequencyGridConfig",
    "ExplicitFrequencyConfig",
    "ExecutionConfig",
    "SkyLoadingConfig",
    "SolverExecutionConfig",
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
    "ResolvedSkyLoadingConfig",
    "ResolvedSolverExecutionConfig",
    "ResolvedSimulationConfig",
    "ResolvedConfiguration",
    # Canonical HDF5 result I/O
    "HDF5ReadLimits",
    "write_result_hdf5",
    "load_result_hdf5",
    "ResultFormat",
    "write_result_summary_json",
    # Standard visibility projection and exchange formats
    "StandardVisibilityData",
    "write_measurement_set",
    "read_measurement_set",
    "write_uvfits",
    "read_uvfits",
]
