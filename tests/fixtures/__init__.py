# tests/fixtures/__init__.py
"""
Test fixtures for RRIvis.

This module provides reusable test data for antenna layouts, sky models,
and configurations.
"""

from .sample_antenna_layouts import (
    get_simple_two_antenna_layout,
    get_three_antenna_layout,
    get_linear_array_layout,
    get_hexagonal_layout,
    get_random_layout,
    get_baselines_for_antennas,
    LAYOUTS,
)

from .sample_sky_models import (
    get_single_bright_source,
    get_multiple_unpolarized_sources,
    get_polarized_sources,
    get_random_sources,
    get_grid_sources,
    get_point_source_at_phase_center,
    get_two_point_sources_symmetric,
    SKY_MODELS,
)

from .sample_configs import (
    get_minimal_config,
    get_full_config,
    get_jones_enabled_config,
    get_gleam_config,
    get_gsm_config,
    get_gpu_config,
    get_numba_config,
    write_config_to_file,
    create_test_antenna_file,
    CONFIGS,
)

__all__ = [
    # Antenna layouts
    "get_simple_two_antenna_layout",
    "get_three_antenna_layout",
    "get_linear_array_layout",
    "get_hexagonal_layout",
    "get_random_layout",
    "get_baselines_for_antennas",
    "LAYOUTS",
    # Sky models
    "get_single_bright_source",
    "get_multiple_unpolarized_sources",
    "get_polarized_sources",
    "get_random_sources",
    "get_grid_sources",
    "get_point_source_at_phase_center",
    "get_two_point_sources_symmetric",
    "SKY_MODELS",
    # Configs
    "get_minimal_config",
    "get_full_config",
    "get_jones_enabled_config",
    "get_gleam_config",
    "get_gsm_config",
    "get_gpu_config",
    "get_numba_config",
    "write_config_to_file",
    "create_test_antenna_file",
    "CONFIGS",
]
