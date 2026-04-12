"""Public registry surface for sky-model loaders.

Advanced code that needs to inspect or extend the loader registry should
import from this module rather than the private ``_registry`` implementation.
"""

from ._registry import (
    LoaderDefinition,
    build_alias_map,
    build_network_services_map,
    build_sky_model_map,
    ensure_default_loaders_registered,
    get_loader,
    get_loader_definition,
    get_loader_meta,
    list_loader_definitions,
    list_loaders,
    register_loader,
    resolve_loader_name,
)

__all__ = [
    "LoaderDefinition",
    "register_loader",
    "get_loader",
    "get_loader_meta",
    "get_loader_definition",
    "resolve_loader_name",
    "list_loaders",
    "list_loader_definitions",
    "build_sky_model_map",
    "build_network_services_map",
    "build_alias_map",
    "ensure_default_loaders_registered",
]
