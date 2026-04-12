# rrivis/core/sky/__init__.py
"""Unified sky model package for RRIVis."""

from ._data import HealpixData, PointSourceData, SourceArrays
from .catalogs import DiffuseModelEntry, RacsCatalogEntry, VizierCatalogEntry
from .constants import (
    C_LIGHT,
    H_PLANCK,
    K_BOLTZMANN,
    BrightnessConversion,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
    rayleigh_jeans_factor,
)
from .convert import bin_sources_to_flux
from .discovery import estimate_healpix_memory, get_catalog_info, list_all_models
from .model import (
    SkyFormat,
    SkyModel,
)
from .plotter import SkyPlotter
from .region import BoxRegion, ConeRegion, SkyRegion, UnionRegion
from .spectral import apply_faraday_rotation, compute_spectral_scale

__all__ = [
    "SkyModel",
    "SkyPlotter",
    "SkyRegion",
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
    "bin_sources_to_flux",
    "DiffuseModelEntry",
    "VizierCatalogEntry",
    "RacsCatalogEntry",
]
