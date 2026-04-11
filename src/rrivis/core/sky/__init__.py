# rrivis/core/sky/__init__.py
"""Unified sky model package for RRIVis."""

# Ensure all loader modules are imported so their @register_loader decorators run.
from . import _loaders_bbs as _loaders_bbs  # noqa: F401
from . import _loaders_diffuse as _loaders_diffuse  # noqa: F401
from . import _loaders_fits as _loaders_fits  # noqa: F401
from . import _loaders_pyradiosky as _loaders_pyradiosky  # noqa: F401
from . import _loaders_vizier as _loaders_vizier  # noqa: F401
from ._registry import (
    build_alias_map,
    build_loader_kwargs,
    build_network_services_map,
    build_sky_model_map,
    get_loader,
    get_loader_meta,
    list_loaders,
    register_loader,
)
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
    SourceArrays,
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
    "register_loader",
    "get_loader",
    "get_loader_meta",
    "list_loaders",
    "build_network_services_map",
    "build_alias_map",
    "build_loader_kwargs",
    "compute_spectral_scale",
    "apply_faraday_rotation",
    "SkyFormat",
    "SourceArrays",
    "estimate_healpix_memory",
    "list_all_models",
    "get_catalog_info",
    "build_sky_model_map",
    "rayleigh_jeans_factor",
    "bin_sources_to_flux",
    "DiffuseModelEntry",
    "VizierCatalogEntry",
    "RacsCatalogEntry",
]
