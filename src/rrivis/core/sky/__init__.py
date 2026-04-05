# rrivis/core/sky/__init__.py
"""Unified sky model package for RRIVis."""

# Ensure all loader modules are imported so their @register_loader decorators run.
from . import _loaders_bbs as _loaders_bbs  # noqa: F401
from . import _loaders_diffuse as _loaders_diffuse  # noqa: F401
from . import _loaders_fits as _loaders_fits  # noqa: F401
from . import _loaders_pyradiosky as _loaders_pyradiosky  # noqa: F401
from . import _loaders_vizier as _loaders_vizier  # noqa: F401
from ._registry import get_loader, list_loaders, register_loader
from .constants import (
    C_LIGHT,
    H_PLANCK,
    K_BOLTZMANN,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
)
from .model import SkyModel
from .region import BoxRegion, ConeRegion, SkyRegion, UnionRegion

__all__ = [
    "SkyModel",
    "SkyRegion",
    "ConeRegion",
    "BoxRegion",
    "UnionRegion",
    "K_BOLTZMANN",
    "C_LIGHT",
    "H_PLANCK",
    "brightness_temp_to_flux_density",
    "flux_density_to_brightness_temp",
    "register_loader",
    "get_loader",
    "list_loaders",
]
