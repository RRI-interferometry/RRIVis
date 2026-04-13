# rrivis/core/sky/__init__.py
"""Unified sky model package for RRIVis."""

from ._data import HealpixData, PointSourceData, SourceArrays
from ._factories import create_empty, create_from_arrays, create_test_sources
from ._loaders_bbs import write_bbs
from ._serialization import load_skyh5, save_skyh5, to_pyradiosky
from .combine import combine_models
from .constants import (
    C_LIGHT,
    H_PLANCK,
    K_BOLTZMANN,
    BrightnessConversion,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
    rayleigh_jeans_factor,
)
from .discovery import estimate_healpix_memory, get_catalog_info, list_all_models
from .model import (
    SkyFormat,
    SkyModel,
)
from .operations import (
    materialize_healpix_model,
    materialize_point_sources_model,
    with_memmap_backing,
)
from .plotter import SkyPlotter
from .region import BoxRegion, ConeRegion, SkyRegion, UnionRegion
from .spectral import apply_faraday_rotation, compute_spectral_scale

__all__ = [
    "SkyModel",
    "SkyPlotter",
    "SkyRegion",
    "create_empty",
    "create_from_arrays",
    "create_test_sources",
    "combine_models",
    "materialize_healpix_model",
    "materialize_point_sources_model",
    "with_memmap_backing",
    "to_pyradiosky",
    "save_skyh5",
    "load_skyh5",
    "write_bbs",
    "ConeRegion",
    "BoxRegion",
    "UnionRegion",
    "K_BOLTZMANN",
    "C_LIGHT",
    "H_PLANCK",
    "BrightnessConversion",
    "brightness_temp_to_flux_density",
    "flux_density_to_brightness_temp",
    "compute_spectral_scale",
    "apply_faraday_rotation",
    "SkyFormat",
    "SourceArrays",
    "PointSourceData",
    "HealpixData",
    "estimate_healpix_memory",
    "list_all_models",
    "get_catalog_info",
    "rayleigh_jeans_factor",
]
