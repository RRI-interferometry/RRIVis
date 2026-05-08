"""Public registry surface for sky-model loaders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import _registry as _backend
from ._registry import (
    LoaderCategory,
    LoaderDefinition,
    LoaderOutputMode,
    LoaderRepresentation,
)


class SkyLoaderRegistry:
    """Public facade around the built-in sky-loader registry."""

    def register(
        self,
        name: str,
        *,
        config_section: str | None = None,
        use_flag: str | None = None,
        representations: tuple[LoaderRepresentation, ...] | None = None,
        category: LoaderCategory = "catalog",
        requires_file: bool = False,
        network_service: str | None = None,
        aliases: (
            list[str] | tuple[str, ...] | dict[str, dict[str, Any] | None] | None
        ) = None,
        config_fields: dict[str, str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a loader function."""

        return _backend.register_loader(
            name,
            config_section=config_section,
            use_flag=use_flag,
            representations=representations,
            category=category,
            requires_file=requires_file,
            network_service=network_service,
            aliases=aliases,
            config_fields=config_fields,
        )

    def loader(self, name: str) -> Callable[..., Any]:
        """Return a loader function by canonical name or alias."""
        return _backend.get_loader(name)

    def definition(self, name: str) -> LoaderDefinition:
        """Return loader metadata by canonical name or alias."""
        return _backend.get_loader_definition(name)

    def meta(self, name: str) -> dict[str, Any]:
        """Return a serializable metadata dict for one loader."""
        return _backend.loader_metadata(name)

    def resolve_name(self, name: str) -> str:
        """Resolve a canonical loader name from an alias or canonical name."""
        return _backend.resolve_loader_name(name)

    def resolve_request(
        self,
        name: str,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Resolve a loader request and merge alias-bound default kwargs."""
        return _backend.resolve_loader_request(name, kwargs)

    def names(self) -> list[str]:
        """Return registered canonical loader names."""
        return _backend.list_loaders()

    def definitions(self) -> list[LoaderDefinition]:
        """Return registered loader definitions."""
        return _backend.list_loader_definitions()

    def aliases(self) -> dict[str, str]:
        """Return alias -> canonical loader mappings."""
        _backend.ensure_default_loaders_registered()
        return _backend._REGISTRY.alias_map()

    def alias_defaults(self) -> dict[str, dict[str, Any]]:
        """Return alias-bound default kwargs."""
        _backend.ensure_default_loaders_registered()
        return _backend._REGISTRY.alias_defaults_map()

    def network_services(self) -> dict[str, str]:
        """Return loader name -> required network service."""
        result: dict[str, str] = {}
        for definition in self.definitions():
            if definition.network_service is not None:
                result[definition.name] = definition.network_service
        return result


loader_registry = SkyLoaderRegistry()


__all__ = [
    "LoaderCategory",
    "LoaderDefinition",
    "LoaderOutputMode",
    "LoaderRepresentation",
    "SkyLoaderRegistry",
    "loader_registry",
]
