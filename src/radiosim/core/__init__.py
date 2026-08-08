"""Core computation modules for RadioSim.

This module contains the fundamental building blocks for radio interferometry
visibility simulation including antenna handling, baseline generation,
beam patterns, source models, and the RIME visibility calculation.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from radiosim.core.beam import (
    BeamAngularDomainError,
    BeamAssignmentError,
    BeamAssignmentProvenance,
    BeamDependencyError,
    BeamDisplayNormalizationError,
    BeamError,
    BeamEvaluationError,
    BeamFileChangedError,
    BeamFileReadError,
    BeamFrequencyDomainError,
    BeamLoadError,
    BeamNormalizationError,
    BeamSamplingDerivationError,
    DuplicateBeamAssignmentError,
    IncompleteBeamAssignmentError,
    InconsistentBeamAssignmentError,
    LoadedBeamState,
    NonFiniteBeamResponseError,
    ResolvedAnalyticalIlluminationBeamModel,
    ResolvedAnalyticBeamChoice,
    ResolvedAnalyticBeamDefinition,
    ResolvedAnalyticBeamModel,
    ResolvedAnalyticBeamsInput,
    ResolvedAntennaPointingOffset,
    ResolvedAntennaSurfaceError,
    ResolvedBeamAssignment,
    ResolvedBeamPointing,
    ResolvedBeamsInput,
    ResolvedBeamState,
    ResolvedBeamSurfaceError,
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
    ResolvedPointingOffset,
    ResolvedPrimeFocusReflector,
    ResolvedRectangularApertureBeamModel,
    ResolvedReflector,
    ResolvedSharedFITSBeamsInput,
    ResolvedSurfaceError,
    ResolvedUniformTaper,
    UnknownBeamAntennaError,
    UnsupportedBeamBasisError,
    UnsupportedBeamCoordinateError,
    UnsupportedBeamFeedError,
    UnsupportedBeamMetadataError,
    UnsupportedBeamPrecisionError,
    UnsupportedBeamTypeError,
    resolve_beam_assignments,
)
from radiosim.core.hybrid import (
    HYBRID_COMPONENT_NAMES,
    HybridSkyError,
    HybridSolveOutcome,
    SolvedComponent,
    check_representation_compatibility,
    component_names_for_representation,
    solve_sky,
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
from radiosim.core.phase_center import PhaseCenter
from radiosim.core.polarization import (
    apply_jones_matrices,
    stokes_to_coherency,
)
from radiosim.core.polarization_basis import (
    CORRELATION_LABELS,
    SKY_NORTH_EAST_TO_CIRCULAR_RL,
    SKY_NORTH_EAST_TO_LINEAR_XY,
    PolarizationBasis,
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
from radiosim.core.receptor import (
    AmbiguousOutputBasisError,
    InvalidReceptorConfigError,
    ReceptorAssignmentError,
    ReceptorError,
    ReceptorProvenance,
    ResolvedReceptor,
    ResolvedReceptorSet,
    UnsupportedBasisTransformError,
    UnsupportedFeedGeometryError,
    UnsupportedReceptorBasisError,
    resolve_receptors,
)
from radiosim.core.result import (
    BackendResultProvenance,
    InvalidPhaseCenterError,
    InvalidResultError,
    InvalidTimeGridError,
    LoadedSimulationResult,
    ResultCoordinateError,
    ResultError,
    ResultPerformance,
    ResultShapeError,
    ResultUnavailableError,
    SimulationResult,
    SolverResultProvenance,
    TimeGridLimitError,
    build_loaded_simulation_result,
    build_simulation_result,
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
from radiosim.core.solver_partition import (
    SolverPartitionError,
    partition_time_axis,
    validate_time_partition,
)
from radiosim.core.time_grid import (
    MAX_TIME_SAMPLES,
    ObservationTimeGrid,
    build_observation_time_grid,
)

if TYPE_CHECKING:
    from radiosim.core.beam.runtime import BeamSystem, load_beam_system
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
    "BeamSystem": ("radiosim.core.beam.runtime", "BeamSystem"),
    "load_beam_system": ("radiosim.core.beam.runtime", "load_beam_system"),
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
    # Canonical receptor models
    "ResolvedReceptor",
    "ResolvedReceptorSet",
    "ReceptorProvenance",
    "resolve_receptors",
    "ReceptorError",
    "InvalidReceptorConfigError",
    "UnsupportedReceptorBasisError",
    "UnsupportedFeedGeometryError",
    "AmbiguousOutputBasisError",
    "UnsupportedBasisTransformError",
    "ReceptorAssignmentError",
    # Beams
    "BeamSystem",
    "LoadedBeamState",
    "load_beam_system",
    "BeamError",
    "BeamAssignmentError",
    "UnknownBeamAntennaError",
    "DuplicateBeamAssignmentError",
    "IncompleteBeamAssignmentError",
    "InconsistentBeamAssignmentError",
    "BeamLoadError",
    "BeamDependencyError",
    "BeamFileReadError",
    "BeamFileChangedError",
    "UnsupportedBeamMetadataError",
    "UnsupportedBeamTypeError",
    "UnsupportedBeamFeedError",
    "UnsupportedBeamBasisError",
    "UnsupportedBeamCoordinateError",
    "BeamNormalizationError",
    "UnsupportedBeamPrecisionError",
    "BeamSamplingDerivationError",
    "BeamEvaluationError",
    "BeamFrequencyDomainError",
    "BeamAngularDomainError",
    "NonFiniteBeamResponseError",
    "BeamDisplayNormalizationError",
    "BeamAssignmentProvenance",
    "ResolvedBeamAssignment",
    "ResolvedBeamState",
    "resolve_beam_assignments",
    "ResolvedAnalyticBeamChoice",
    "ResolvedAnalyticBeamDefinition",
    "ResolvedAnalyticBeamModel",
    "ResolvedAnalyticBeamsInput",
    "ResolvedAnalyticalIlluminationBeamModel",
    "ResolvedBeamsInput",
    "ResolvedCassegrainReflector",
    "ResolvedCircularApertureBeamModel",
    "ResolvedCorrugatedHornIllumination",
    "ResolvedAntennaPointingOffset",
    "ResolvedAntennaSurfaceError",
    "ResolvedBeamPointing",
    "ResolvedBeamSurfaceError",
    "ResolvedCosineTaper",
    "ResolvedPointingOffset",
    "ResolvedSurfaceError",
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
    "ObservationTimeGrid",
    "build_observation_time_grid",
    "MAX_TIME_SAMPLES",
    "PhaseCenter",
    # Results
    "SimulationResult",
    "LoadedSimulationResult",
    "BackendResultProvenance",
    "SolverResultProvenance",
    "ResultPerformance",
    "build_simulation_result",
    "build_loaded_simulation_result",
    "ResultError",
    "ResultUnavailableError",
    "InvalidResultError",
    "ResultShapeError",
    "ResultCoordinateError",
    "InvalidPhaseCenterError",
    "InvalidTimeGridError",
    "TimeGridLimitError",
    # Polarization
    "stokes_to_coherency",
    "apply_jones_matrices",
    # Polarization basis
    "PolarizationBasis",
    "CORRELATION_LABELS",
    "SKY_NORTH_EAST_TO_LINEAR_XY",
    "SKY_NORTH_EAST_TO_CIRCULAR_RL",
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
    # Solver worker partition
    "SolverPartitionError",
    "partition_time_axis",
    "validate_time_partition",
    # Hybrid solve mode
    "HYBRID_COMPONENT_NAMES",
    "HybridSkyError",
    "HybridSolveOutcome",
    "SolvedComponent",
    "check_representation_compatibility",
    "component_names_for_representation",
    "solve_sky",
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
    if name == "GSMObserver08":
        from pygdsm import GSMObserver08

        globals()[name] = GSMObserver08
        return GSMObserver08
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy public API objects in interactive discovery."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | {"GSMObserver08"})
