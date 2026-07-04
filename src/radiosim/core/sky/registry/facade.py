"""Public registry surface for sky-model loaders."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import core as _registry_core
from .core import (
    _ALL_REPRESENTATIONS,
    _REGISTRY,
    _REPRESENTATION_TO_OUTPUT_MODE,
    LoaderCategory,
    LoaderDefinition,
    LoaderOutputMode,
    LoaderRepresentation,
    ResolvedLoader,
    _ensure_default_loaders_registered,
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
        config_fields: Mapping[str, str] | Sequence[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator that registers a loader function."""

        return _registry_core.register_loader(
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
        _ensure_default_loaders_registered()
        canonical = _REGISTRY.resolve_name(name)
        return _REGISTRY.get_loader(canonical)

    def resolve_callable(self, name: str) -> ResolvedLoader:
        """Return an alias-resolved callable that merges alias defaults on invocation."""
        _ensure_default_loaders_registered()
        canonical, defaults = _REGISTRY.resolve_request(name)
        return ResolvedLoader(
            canonical_name=canonical,
            definition=_REGISTRY.get_definition(canonical),
            alias_defaults=defaults,
        )

    def definition(self, name: str) -> LoaderDefinition:
        """Return loader metadata by canonical name or alias."""
        _ensure_default_loaders_registered()
        return _REGISTRY.get_definition(name)

    def meta(self, name: str) -> dict[str, Any]:
        """Return a serializable metadata dict for one loader."""
        _ensure_default_loaders_registered()
        canonical, defaults = _REGISTRY.resolve_request(name, {})
        definition = _REGISTRY.get_definition(canonical)
        meta = definition.meta_dict()
        representation = defaults.get("representation")
        if representation in _ALL_REPRESENTATIONS:
            meta["representations"] = [representation]
            meta["output_mode"] = _REPRESENTATION_TO_OUTPUT_MODE[
                frozenset({representation})
            ]
        return meta

    def resolve_name(self, name: str) -> str:
        """Resolve a canonical loader name from an alias or canonical name."""
        _ensure_default_loaders_registered()
        return _REGISTRY.resolve_name(name)

    def resolve_request(
        self,
        name: str,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Resolve a loader request and merge alias-bound default kwargs."""
        _ensure_default_loaders_registered()
        return _REGISTRY.resolve_request(name, kwargs)

    def names(self) -> list[str]:
        """Return registered canonical loader names."""
        _ensure_default_loaders_registered()
        return _REGISTRY.list_loaders()

    def definitions(self) -> list[LoaderDefinition]:
        """Return registered loader definitions."""
        _ensure_default_loaders_registered()
        return _REGISTRY.definitions()

    def aliases(self) -> dict[str, str]:
        """Return alias -> canonical loader mappings."""
        _ensure_default_loaders_registered()
        return _REGISTRY.alias_map()

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
    "ResolvedLoader",
    "SkyLoaderRegistry",
    "loader_registry",
]
