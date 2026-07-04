"""Sky-loader registry: public facade, definitions, and catalog metadata."""

from __future__ import annotations

from .catalogs import (
    CASDA_TAP_URL,
    DIFFUSE_MODELS,
    RACS_CATALOGS,
    VIZIER_POINT_CATALOGS,
    DiffuseModelEntry,
    RacsCatalogEntry,
    VizierCatalogEntry,
    load_catalog_footprint_asset,
)
from .facade import (
    LoaderCategory,
    LoaderDefinition,
    LoaderOutputMode,
    LoaderRepresentation,
    ResolvedLoader,
    SkyLoaderRegistry,
    loader_registry,
)

__all__ = [
    "CASDA_TAP_URL",
    "DIFFUSE_MODELS",
    "LoaderCategory",
    "LoaderDefinition",
    "LoaderOutputMode",
    "LoaderRepresentation",
    "RACS_CATALOGS",
    "ResolvedLoader",
    "SkyLoaderRegistry",
    "VIZIER_POINT_CATALOGS",
    "DiffuseModelEntry",
    "RacsCatalogEntry",
    "VizierCatalogEntry",
    "load_catalog_footprint_asset",
    "loader_registry",
]
