"""Sky-loader registry: public facade, definitions, and catalog metadata."""

from __future__ import annotations

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
    "LoaderCategory",
    "LoaderDefinition",
    "LoaderOutputMode",
    "LoaderRepresentation",
    "ResolvedLoader",
    "SkyLoaderRegistry",
    "loader_registry",
]
