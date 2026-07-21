"""Core computation modules for RadioSim.

This module contains the fundamental building blocks for radio interferometry
visibility simulation including antenna handling, baseline generation,
beam patterns, source models, and the RIME visibility calculation.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from radiosim.core.beam import (
    ResolvedAnalyticalIlluminationBeamModel,
    ResolvedAnalyticBeamChoice,
    ResolvedAnalyticBeamDefinition,
    ResolvedAnalyticBeamModel,
    ResolvedAnalyticBeamsInput,
    ResolvedBeamsInput,
    ResolvedCassegrainReflector,
    ResolvedCircularApertureBeamModel,
    ResolvedCorrugatedHornIllumination,
    ResolvedCosineTaper,
    ResolvedDerivedGaussianTaper,
    ResolvedDerivedParabolicSquaredTaper,
    ResolvedDerivedParabolicTaper,
    ResolvedDerivedTaper,
    ResolvedDipoleGroundPlaneIllumination,
    ResolvedDirectTaper,
    ResolvedEllipticalApertureBeamModel,
    ResolvedFITSBeamAssignmentInput,
    ResolvedFITSBeamDefinition,
    ResolvedGaussianTaper,
    ResolvedIllumination,
    ResolvedMixedBeamAssignmentInput,
    ResolvedMixedBeamsInput,
    ResolvedNumericalIlluminationBeamModel,
    ResolvedOpenWaveguideIllumination,
    ResolvedParabolicSquaredTaper,
    ResolvedParabolicTaper,
    ResolvedPerAntennaFITSBeamsInput,
    ResolvedPrimeFocusReflector,
    ResolvedRectangularApertureBeamModel,
    ResolvedReflector,
    ResolvedSharedFITSBeamsInput,
    ResolvedUniformTaper,
)
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
    ResolvedConfiguration,
    ResolvedExecutionConfig,
    ResolvedFrequencyConfig,
    ResolvedObservationConfig,
    ResolvedSimulationConfig,
    ResolvedSkyModelConfig,
    ResolvedSkySourceRequest,
)

if TYPE_CHECKING:
    from radiosim.core.sky import (
        C_LIGHT,
        H_PLANCK,
        K_BOLTZMANN,
        SkyModel,
        brightness_temp_to_flux_density,
        flux_density_to_brightness_temp,
    )
    from radiosim.core.visibility_healpix import calculate_visibility_healpix


_LAZY_EXPORTS = {
    "SkyModel": ("radiosim.core.sky", "SkyModel"),
    "K_BOLTZMANN": ("radiosim.core.sky", "K_BOLTZMANN"),
    "C_LIGHT": ("radiosim.core.sky", "C_LIGHT"),
    "H_PLANCK": ("radiosim.core.sky", "H_PLANCK"),
    "brightness_temp_to_flux_density": (
        "radiosim.core.sky",
        "brightness_temp_to_flux_density",
    ),
    "flux_density_to_brightness_temp": (
        "radiosim.core.sky",
        "flux_density_to_brightness_temp",
    ),
    "calculate_visibility_healpix": (
        "radiosim.core.visibility_healpix",
        "calculate_visibility_healpix",
    ),
}


def calculate_visibility(*args, **kwargs):
    """Lazily dispatch to the point-source visibility implementation."""
    from radiosim.core.visibility import calculate_visibility as implementation

    return implementation(*args, **kwargs)


__all__ = [
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
    "ResolvedAnalyticBeamChoice",
    "ResolvedAnalyticBeamDefinition",
    "ResolvedAnalyticBeamModel",
    "ResolvedAnalyticBeamsInput",
    "ResolvedAnalyticalIlluminationBeamModel",
    "ResolvedBeamsInput",
    "ResolvedCassegrainReflector",
    "ResolvedCircularApertureBeamModel",
    "ResolvedCorrugatedHornIllumination",
    "ResolvedCosineTaper",
    "ResolvedDerivedGaussianTaper",
    "ResolvedDerivedParabolicSquaredTaper",
    "ResolvedDerivedParabolicTaper",
    "ResolvedDerivedTaper",
    "ResolvedDipoleGroundPlaneIllumination",
    "ResolvedDirectTaper",
    "ResolvedEllipticalApertureBeamModel",
    "ResolvedFITSBeamAssignmentInput",
    "ResolvedFITSBeamDefinition",
    "ResolvedGaussianTaper",
    "ResolvedIllumination",
    "ResolvedMixedBeamAssignmentInput",
    "ResolvedMixedBeamsInput",
    "ResolvedNumericalIlluminationBeamModel",
    "ResolvedOpenWaveguideIllumination",
    "ResolvedParabolicSquaredTaper",
    "ResolvedParabolicTaper",
    "ResolvedPerAntennaFITSBeamsInput",
    "ResolvedPrimeFocusReflector",
    "ResolvedRectangularApertureBeamModel",
    "ResolvedReflector",
    "ResolvedSharedFITSBeamsInput",
    "ResolvedUniformTaper",
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
    "ResolvedObservationConfig",
    "ResolvedFrequencyConfig",
    "ResolvedSkySourceRequest",
    "ResolvedSkyModelConfig",
    "ResolvedExecutionConfig",
    "ResolvedSimulationConfig",
    "ResolvedConfiguration",
]


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attribute_name)
        globals()[name] = value
        return value
    if name == "BeamManager":
        from radiosim.core.jones.beam import BeamManager

        globals()[name] = BeamManager
        return BeamManager
    if name == "GSMObserver08":
        from pygdsm import GSMObserver08

        globals()[name] = GSMObserver08
        return GSMObserver08
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy public API objects in interactive discovery."""
    return sorted(
        set(globals()) | set(_LAZY_EXPORTS) | {"BeamManager", "GSMObserver08"}
    )
