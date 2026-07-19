"""Core computation modules for RadioSim.

This module contains the fundamental building blocks for radio interferometry
visibility simulation including antenna handling, baseline generation,
beam patterns, source models, and the RIME visibility calculation.
"""

from radiosim.core.antenna import read_antenna_positions
from radiosim.core.baseline import generate_baselines
from radiosim.core.instrument import (
    AntennaFieldSource,
    AntennaId,
    AntennaProvenance,
    BaselineSelectionCriteriaSnapshot,
    BaselineSelectionProvenance,
    InstrumentProvenance,
    ResolvedAntenna,
    ResolvedBaseline,
    ResolvedBaselineSelection,
    ResolvedEarthLocation,
    ResolvedInstrument,
)
from radiosim.core.jones.beam import BeamManager
from radiosim.core.observation import get_location_and_time
from radiosim.core.polarization import (
    apply_jones_matrices,
    stokes_to_coherency,
    visibility_to_correlations,
)
from radiosim.core.precision import (
    COMPLEX256_AVAILABLE,
    FLOAT128_AVAILABLE,
    CoordinatePrecision,
    JonesPrecision,
    PrecisionConfig,
    PrecisionLevel,
    get_complex_dtype,
    get_real_dtype,
    resolve_precision,
)
from radiosim.core.runtime_config import (
    ConfigurationProvenance,
    FrozenMapping,
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
from radiosim.core.sky import (
    C_LIGHT,
    H_PLANCK,
    K_BOLTZMANN,
    SkyModel,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
)
from radiosim.core.visibility import calculate_visibility
from radiosim.core.visibility_healpix import calculate_visibility_healpix

__all__ = [
    # Antenna
    "read_antenna_positions",
    # Baseline
    "generate_baselines",
    # Canonical instrument models
    "AntennaId",
    "AntennaFieldSource",
    "ResolvedEarthLocation",
    "AntennaProvenance",
    "ResolvedAntenna",
    "InstrumentProvenance",
    "ResolvedInstrument",
    "ResolvedBaseline",
    "BaselineSelectionCriteriaSnapshot",
    "BaselineSelectionProvenance",
    "ResolvedBaselineSelection",
    # Beams
    "BeamManager",
    # Observation
    "get_location_and_time",
    # Polarization
    "stokes_to_coherency",
    "apply_jones_matrices",
    "visibility_to_correlations",
    # Sky Model (unified)
    "SkyModel",
    "GSMObserver08",
    "K_BOLTZMANN",
    "C_LIGHT",
    "H_PLANCK",
    "brightness_temp_to_flux_density",
    "flux_density_to_brightness_temp",
    # Visibility
    "calculate_visibility",
    "calculate_visibility_healpix",
    # Precision
    "PrecisionConfig",
    "PrecisionLevel",
    "CoordinatePrecision",
    "JonesPrecision",
    "resolve_precision",
    "get_real_dtype",
    "get_complex_dtype",
    "FLOAT128_AVAILABLE",
    "COMPLEX256_AVAILABLE",
    # Resolved configuration
    "FrozenMapping",
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
]


def __getattr__(name: str):
    if name == "GSMObserver08":
        from pygdsm import GSMObserver08

        return GSMObserver08
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
