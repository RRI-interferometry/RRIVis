"""Core computation modules for RRIvis.

This module contains the fundamental building blocks for radio interferometry
visibility simulation including antenna handling, baseline generation,
beam patterns, source models, and the RIME visibility calculation.
"""

from rrivis.core.antenna import read_antenna_positions
from rrivis.core.baseline import generate_baselines
from rrivis.core.beams import (
    gaussian_A_theta_EBeam,
    airy_disk_pattern,
    calculate_hpbw,
    calculate_gaussian_beam_area_EBeam,
    calculate_airy_beam_area,
    calculate_hpbw_radians,
    AntennaType,
    BeamPatternType,
)
from rrivis.core.beam_file import BeamManager
from rrivis.core.observation import get_location_and_time
from rrivis.core.polarization import (
    stokes_to_coherency,
    apply_jones_matrices,
    visibility_to_correlations,
)
from rrivis.core.source import (
    get_sources,
    generate_test_sources,
    load_gleam,
    load_gsm2008,
)
from rrivis.core.visibility import calculate_visibility

__all__ = [
    # Antenna
    "read_antenna_positions",
    # Baseline
    "generate_baselines",
    # Beams
    "gaussian_A_theta_EBeam",
    "airy_disk_pattern",
    "calculate_hpbw",
    "calculate_gaussian_beam_area_EBeam",
    "calculate_airy_beam_area",
    "calculate_hpbw_radians",
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
    # Source
    "get_sources",
    "generate_test_sources",
    "load_gleam",
    "load_gsm2008",
    # Visibility
    "calculate_visibility",
]
