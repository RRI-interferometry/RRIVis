"""Public registry surface for sky-model loaders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import core as _backend
from .core import (
    LoaderCategory,
    LoaderDefinition,
    LoaderOutputMode,
    LoaderRepresentation,
    ResolvedLoader,
)


class SkyLoaderRegistry:
    """Public facade around the built-in sky-loader registry.

    Two distinct accessors return the underlying loader callable:

    * :meth:`loader` — returns the **bare canonical** function. Alias-bound
      default kwargs are NOT applied; the caller supplies every argument.
    * :meth:`resolve_callable` — returns a :class:`ResolvedLoader` callable
      that merges any alias-bound defaults under the caller's kwargs.

    Use :meth:`loader` when you want a stable function reference (e.g. to
    re-export a catalog-specific loader). Use :meth:`resolve_callable`
    when you accept a name from configuration and need alias defaults
    (``"gsm"`` → ``load_diffuse_sky(model="gsm2008")``) applied
    automatically.
    """

    def register_loader(
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
        """Return a decorator that registers a loader function."""

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
        """Return the bare canonical loader function (no alias defaults applied)."""
        return _backend._get_canonical_loader(name)

    def resolve_callable(self, name: str) -> ResolvedLoader:
        """Return an alias-resolved callable that merges alias defaults on invocation."""
        return _backend._get_resolved_loader(name)

    def definition(self, name: str) -> LoaderDefinition:
        """Return loader metadata by canonical name or alias."""
        return _backend._get_loader_definition(name)

    def meta(self, name: str) -> dict[str, Any]:
        """Return a serializable metadata dict for one loader."""
        return _backend._loader_metadata(name)

    def resolve_name(self, name: str) -> str:
        """Resolve a canonical loader name from an alias or canonical name."""
        return _backend._resolve_loader_name(name)

    def resolve_request(
        self,
        name: str,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Resolve a loader request and merge alias-bound default kwargs."""
        return _backend._resolve_loader_request(name, kwargs)

    def names(self) -> list[str]:
        """Return registered canonical loader names."""
        return _backend._list_loaders()

    def definitions(self) -> list[LoaderDefinition]:
        """Return registered loader definitions."""
        return _backend._list_loader_definitions()

    def aliases(self) -> dict[str, str]:
        """Return alias -> canonical loader mappings."""
        return _backend._alias_map()

    def alias_defaults(self) -> dict[str, dict[str, Any]]:
        """Return alias-bound default kwargs."""
        return _backend._alias_defaults_map()

    def network_services(self) -> dict[str, str]:
        """Return loader name -> required network service."""
        result: dict[str, str] = {}
        for definition in self.definitions():
            if definition.network_service is not None:
                result[definition.name] = definition.network_service
        return result

    def ensure_default_loaders_registered(self) -> None:
        """Force-import every built-in loader module (idempotent)."""
        _backend._ensure_default_loaders_registered()

    def unregister(self, name: str) -> None:
        """Remove a loader and its aliases (intended for tests).

        Production code should not need this — the built-in loaders are
        permanent.
        """
        _backend._unregister_loader(name)


loader_registry = SkyLoaderRegistry()


__all__ = [
    "LoaderCategory",
    "LoaderDefinition",
    "LoaderOutputMode",
    "LoaderRepresentation",
    "ResolvedLoader",
    "SkyLoaderRegistry",
    "loader_registry",
]
