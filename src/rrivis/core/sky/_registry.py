"""Central loader registry for sky-model sources."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal

LoaderCategory = Literal["catalog", "diffuse", "synthetic", "file"]
LoaderRepresentation = Literal["point_sources", "healpix_map"]
LoaderOutputMode = Literal["point_only", "healpix_only", "polymorphic"]

_ALL_REPRESENTATIONS: tuple[LoaderRepresentation, ...] = (
    "point_sources",
    "healpix_map",
)
_REPRESENTATION_TO_OUTPUT_MODE: dict[
    frozenset[LoaderRepresentation], LoaderOutputMode
] = {
    frozenset({"point_sources"}): "point_only",
    frozenset({"healpix_map"}): "healpix_only",
    frozenset(_ALL_REPRESENTATIONS): "polymorphic",
}


def _normalize_representations(
    representations: (Sequence[LoaderRepresentation] | None) = None,
    *,
    loader: Callable[..., Any] | None = None,
    name: str | None = None,
    config_fields: Mapping[str, str] | None = None,
) -> tuple[LoaderRepresentation, ...]:
    """Normalize explicit or inferred representation hints into one tuple."""

    if representations is not None:
        ordered = tuple(dict.fromkeys(representations))
        if not ordered:
            raise ValueError("Loader representations cannot be empty")
        invalid = [rep for rep in ordered if rep not in _ALL_REPRESENTATIONS]
        if invalid:
            raise ValueError(
                f"Unknown loader representations: {invalid}. "
                f"Known: {list(_ALL_REPRESENTATIONS)}"
            )
        return ordered

    param_names: set[str] = set()
    if loader is not None:
        try:
            param_names = set(inspect.signature(loader).parameters)
        except (TypeError, ValueError):
            param_names = set()

    if "representation" in param_names or (
        config_fields is not None and "representation" in config_fields
    ):
        return _ALL_REPRESENTATIONS

    if name == "pyradiosky_file":
        return _ALL_REPRESENTATIONS

    return ("point_sources",)


@dataclass(frozen=True)
class LoaderDefinition:
    """Metadata describing a registered sky loader."""

    name: str
    loader: Callable[..., Any]
    config_section: str | None = None
    use_flag: str | None = None
    representations: tuple[LoaderRepresentation, ...] = ("point_sources",)
    category: LoaderCategory = "catalog"
    requires_file: bool = False
    network_service: str | None = None
    aliases: tuple[str, ...] = ()
    alias_defaults: dict[str, dict[str, Any]] = field(default_factory=dict)
    config_fields: dict[str, str] = field(default_factory=dict)

    @property
    def supports_point_sources(self) -> bool:
        return "point_sources" in self.representations

    @property
    def supports_healpix_map(self) -> bool:
        return "healpix_map" in self.representations

    @property
    def output_mode(self) -> LoaderOutputMode:
        return _REPRESENTATION_TO_OUTPUT_MODE.get(
            frozenset(self.representations), "polymorphic"
        )

    @property
    def primary_representation(self) -> LoaderRepresentation:
        return self.representations[0]

    def meta_dict(self) -> dict[str, Any]:
        return {
            "config_section": self.config_section or self.name,
            "use_flag": self.use_flag or f"use_{self.name}",
            "representations": list(self.representations),
            "representation": self.primary_representation,
            "output_mode": self.output_mode,
            "primary_representation": self.primary_representation,
            "supports_point_sources": self.supports_point_sources,
            "supports_healpix_map": self.supports_healpix_map,
            "capabilities": {
                "supports_point_sources": self.supports_point_sources,
                "supports_healpix_map": self.supports_healpix_map,
                "output_mode": self.output_mode,
            },
            "category": self.category,
            "requires_file": self.requires_file,
            "network_service": self.network_service,
            "aliases": list(self.aliases),
            "alias_defaults": {
                alias: dict(defaults) for alias, defaults in self.alias_defaults.items()
            },
            "config_fields": dict(self.config_fields),
        }


class LoaderRegistry:
    """Mutable registry of sky loaders plus their metadata."""

    def __init__(self) -> None:
        self._loaders: dict[str, Callable[..., Any]] = {}
        self._definitions: dict[str, LoaderDefinition] = {}
        self._aliases: dict[str, str] = {}
        self._alias_defaults: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _normalize_aliases(
        aliases: (
            list[str] | tuple[str, ...] | Mapping[str, Mapping[str, Any] | None] | None
        ),
    ) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
        if aliases is None:
            return (), {}
        if isinstance(aliases, Mapping):
            names = tuple(aliases.keys())
            defaults = {
                alias: dict(default or {}) for alias, default in aliases.items()
            }
            return names, defaults
        names = tuple(aliases)
        return names, {alias: {} for alias in names}

    def register(
        self,
        name: str,
        loader: Callable[..., Any],
        *,
        config_section: str | None = None,
        use_flag: str | None = None,
        representations: Sequence[LoaderRepresentation] | None = None,
        category: LoaderCategory = "catalog",
        requires_file: bool = False,
        network_service: str | None = None,
        aliases: (
            list[str] | tuple[str, ...] | Mapping[str, Mapping[str, Any] | None] | None
        ) = None,
        config_fields: dict[str, str] | None = None,
    ) -> Callable[..., Any]:
        alias_names, alias_defaults = self._normalize_aliases(aliases)
        normalized_representations = _normalize_representations(
            representations,
            loader=loader,
            name=name,
            config_fields=config_fields,
        )
        definition = LoaderDefinition(
            name=name,
            loader=loader,
            config_section=config_section,
            use_flag=use_flag,
            representations=normalized_representations,
            category=category,
            requires_file=requires_file,
            network_service=network_service,
            aliases=alias_names,
            alias_defaults=alias_defaults,
            config_fields=dict(config_fields or {}),
        )
        self._loaders[name] = loader
        self._definitions[name] = definition
        for alias in definition.aliases:
            self._aliases[alias] = name
            self._alias_defaults[alias] = dict(definition.alias_defaults.get(alias, {}))
        return loader

    def resolve_name(self, name: str) -> str:
        if name in self._definitions:
            return name
        if name in self._aliases:
            return self._aliases[name]
        raise ValueError(
            f"Unknown sky model loader '{name}'. Available: {sorted(self._definitions)}"
        )

    def get_loader(self, name: str) -> Callable[..., Any]:
        return self._definitions[self.resolve_name(name)].loader

    def get_definition(self, name: str) -> LoaderDefinition:
        return self._definitions[self.resolve_name(name)]

    def resolve_request(
        self, name: str, kwargs: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Resolve a loader name or alias and merge alias-bound defaults."""
        if name in self._definitions:
            return name, dict(kwargs or {})
        if name in self._aliases:
            canonical = self._aliases[name]
            merged = dict(self._alias_defaults.get(name, {}))
            merged.update(kwargs or {})
            return canonical, merged
        raise ValueError(
            f"Unknown sky model loader '{name}'. Available: {sorted(self._definitions)}"
        )

    def list_loaders(self) -> list[str]:
        return sorted(self._definitions)

    def alias_map(self) -> dict[str, str]:
        return dict(self._aliases)

    def alias_defaults_map(self) -> dict[str, dict[str, Any]]:
        return {
            alias: dict(defaults) for alias, defaults in self._alias_defaults.items()
        }

    def definitions(self) -> list[LoaderDefinition]:
        return [self._definitions[name] for name in self.list_loaders()]


_REGISTRY = LoaderRegistry()
_LOADERS = _REGISTRY._loaders
_LOADER_META: dict[str, dict[str, Any]] = {}
_DEFAULT_LOADER_MODULES = (
    "rrivis.core.sky._loaders_bbs",
    "rrivis.core.sky._loaders_diffuse",
    "rrivis.core.sky._loaders_fits",
    "rrivis.core.sky._loaders_pyradiosky",
    "rrivis.core.sky._loaders_synthetic",
    "rrivis.core.sky._loaders_vizier",
)
_DEFAULT_LOADERS_IMPORTED = False


def ensure_default_loaders_registered() -> None:
    """Import built-in loader modules exactly once."""
    global _DEFAULT_LOADERS_IMPORTED
    if _DEFAULT_LOADERS_IMPORTED:
        return
    for module_name in _DEFAULT_LOADER_MODULES:
        import_module(module_name)
    _DEFAULT_LOADERS_IMPORTED = True


def _sync_meta_cache() -> None:
    _LOADER_META.clear()
    for definition in _REGISTRY.definitions():
        _LOADER_META[definition.name] = definition.meta_dict()


def register_loader(
    name: str,
    *,
    config_section: str | None = None,
    use_flag: str | None = None,
    representations: Sequence[LoaderRepresentation] | None = None,
    category: LoaderCategory = "catalog",
    requires_file: bool = False,
    network_service: str | None = None,
    aliases: (
        list[str] | tuple[str, ...] | Mapping[str, Mapping[str, Any] | None] | None
    ) = None,
    config_fields: dict[str, str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator used by loader modules to register themselves."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY.register(
            name,
            func,
            config_section=config_section,
            use_flag=use_flag,
            representations=representations,
            category=category,
            requires_file=requires_file,
            network_service=network_service,
            aliases=aliases,
            config_fields=config_fields,
        )
        _sync_meta_cache()
        return func

    return decorator


def get_loader(name: str) -> Callable[..., Any]:
    ensure_default_loaders_registered()
    canonical, defaults = _REGISTRY.resolve_request(name)
    loader = _REGISTRY.get_loader(canonical)
    if not defaults:
        return loader

    def _loader_with_alias_defaults(**kwargs: Any) -> Any:
        merged = dict(defaults)
        merged.update(kwargs)
        return loader(**merged)

    _loader_with_alias_defaults.__name__ = getattr(loader, "__name__", canonical)
    _loader_with_alias_defaults.__qualname__ = getattr(
        loader, "__qualname__", _loader_with_alias_defaults.__name__
    )
    _loader_with_alias_defaults.__doc__ = getattr(loader, "__doc__", None)
    return _loader_with_alias_defaults


def get_loader_definition(name: str) -> LoaderDefinition:
    ensure_default_loaders_registered()
    return _REGISTRY.get_definition(name)


def resolve_loader_name(name: str) -> str:
    ensure_default_loaders_registered()
    return _REGISTRY.resolve_name(name)


def resolve_loader_request(
    name: str,
    kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    ensure_default_loaders_registered()
    return _REGISTRY.resolve_request(name, kwargs)


def list_loader_definitions() -> list[LoaderDefinition]:
    ensure_default_loaders_registered()
    return _REGISTRY.definitions()


def loader_metadata(name: str) -> dict[str, Any]:
    """Return resolved metadata for a loader or alias.

    Alias-bound defaults are applied when they affect the representation
    capability view, so callers see the actual loader request they asked for.
    """

    ensure_default_loaders_registered()
    canonical, defaults = _REGISTRY.resolve_request(name, {})
    definition = _REGISTRY.get_definition(canonical)
    meta = definition.meta_dict()
    representation = defaults.get("representation")
    if representation in _ALL_REPRESENTATIONS:
        meta["representations"] = [representation]
        meta["representation"] = representation
        meta["primary_representation"] = representation
        meta["supports_point_sources"] = representation == "point_sources"
        meta["supports_healpix_map"] = representation == "healpix_map"
        meta["capabilities"] = {
            "supports_point_sources": meta["supports_point_sources"],
            "supports_healpix_map": meta["supports_healpix_map"],
            "output_mode": (
                "point_only" if representation == "point_sources" else "healpix_only"
            ),
        }
        meta["output_mode"] = meta["capabilities"]["output_mode"]
    return meta


def list_loaders() -> list[str]:
    ensure_default_loaders_registered()
    return _REGISTRY.list_loaders()
