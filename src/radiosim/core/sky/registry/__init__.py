"""Sky-loader registry: public facade, definitions, and catalog metadata."""

from __future__ import annotations

from .catalogs import (
    CASDA_TAP_URL,
    DIFFUSE_MODELS,
    DIFFUSE_SKY_LOADER_REGISTRY,
    RACS_CATALOGS,
    VIZIER_POINT_CATALOGS,
    DiffuseModelEntry,
    RacsCatalogEntry,
    VizierCatalogEntry,
    build_diffuse_sky_aliases,
    build_racs_aliases,
    build_vizier_family_aliases,
    diffuse_sky_loader_registration,
    load_catalog_footprint_asset,
    racs_loader_registration,
    vizier_family_loader_registration,
    vizier_simple_loader_registration,
)
from .facade import (
    LoaderCategory,
    LoaderDefinition,
    LoaderOutputMode,
    LoaderPathKind,
    LoaderRepresentation,
    ResolvedLoader,
    SkyLoaderRegistry,
    loader_registry,
)

__all__ = [
    "CASDA_TAP_URL",
    "DIFFUSE_MODELS",
    "DIFFUSE_SKY_LOADER_REGISTRY",
    "LoaderCategory",
    "LoaderDefinition",
    "LoaderOutputMode",
    "LoaderPathKind",
    "LoaderRepresentation",
    "RACS_CATALOGS",
    "ResolvedLoader",
    "SkyLoaderRegistry",
    "VIZIER_POINT_CATALOGS",
    "DiffuseModelEntry",
    "RacsCatalogEntry",
    "VizierCatalogEntry",
    "build_diffuse_sky_aliases",
    "build_racs_aliases",
    "build_vizier_family_aliases",
    "diffuse_sky_loader_registration",
    "load_catalog_footprint_asset",
    "loader_registry",
    "racs_loader_registration",
    "vizier_family_loader_registration",
    "vizier_simple_loader_registration",
]
