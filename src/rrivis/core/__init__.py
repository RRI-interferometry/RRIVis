"""Core computation modules for RRIvis.

This module contains the fundamental building blocks for radio interferometry
visibility simulation including antenna handling, baseline generation,
beam patterns, source models, and the RIME visibility calculation.
"""

from pygdsm import GSMObserver08

from rrivis.core.antenna import read_antenna_positions
from rrivis.core.baseline import generate_baselines
from rrivis.core.jones.beam import (
    AntennaType,
    BeamManager,
    BeamPatternType,
    airy_disk_pattern,
    calculate_airy_beam_area,
    calculate_gaussian_beam_area_EBeam,
    calculate_hpbw_for_antenna_type,
    gaussian_A_theta_EBeam,
    get_hpbw_function,
)
from rrivis.core.observation import get_location_and_time
from rrivis.core.polarization import (
    apply_jones_matrices,
    stokes_to_coherency,
    visibility_to_correlations,
)
from rrivis.core.precision import (
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
from rrivis.core.sky import (
    C_LIGHT,
    DIFFUSE_MODELS,
    H_PLANCK,
    K_BOLTZMANN,
    VIZIER_POINT_CATALOGS,
    SkyModel,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
)
from rrivis.core.visibility import calculate_visibility
from rrivis.core.visibility_healpix import calculate_visibility_healpix

__all__ = [
    # Antenna
    "read_antenna_positions",
    # Baseline
    "generate_baselines",
    # Beams
    "gaussian_A_theta_EBeam",
    "airy_disk_pattern",
    "calculate_gaussian_beam_area_EBeam",
    "calculate_airy_beam_area",
    "calculate_hpbw_for_antenna_type",
    "get_hpbw_function",
    "AntennaType",
    "BeamPatternType",
    # Beam file
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
    "VIZIER_POINT_CATALOGS",
    "DIFFUSE_MODELS",
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
]
