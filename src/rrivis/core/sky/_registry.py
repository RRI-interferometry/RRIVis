"""Central loader registry for sky-model sources."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class LoaderDefinition:
    """Metadata describing a registered sky loader."""

    name: str
    loader: Callable[..., Any]
    config_section: str | None = None
    use_flag: str | None = None
    is_healpix: bool = False
    requires_file: bool = False
    network_service: str | None = None
    aliases: tuple[str, ...] = ()
    config_fields: dict[str, str] = field(default_factory=dict)

    def meta_dict(self) -> dict[str, Any]:
        return {
            "config_section": self.config_section or self.name,
            "use_flag": self.use_flag or f"use_{self.name}",
            "is_healpix": self.is_healpix,
            "requires_file": self.requires_file,
            "network_service": self.network_service,
            "aliases": list(self.aliases),
            "config_fields": dict(self.config_fields),
        }


class LoaderRegistry:
    """Mutable registry of sky loaders plus their metadata."""

    def __init__(self) -> None:
        self._loaders: dict[str, Callable[..., Any]] = {}
        self._definitions: dict[str, LoaderDefinition] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        loader: Callable[..., Any],
        *,
        config_section: str | None = None,
        use_flag: str | None = None,
        is_healpix: bool = False,
        requires_file: bool = False,
        network_service: str | None = None,
        aliases: list[str] | tuple[str, ...] | None = None,
        config_fields: dict[str, str] | None = None,
    ) -> Callable[..., Any]:
        definition = LoaderDefinition(
            name=name,
            loader=loader,
            config_section=config_section,
            use_flag=use_flag,
            is_healpix=is_healpix,
            requires_file=requires_file,
            network_service=network_service,
            aliases=tuple(aliases or ()),
            config_fields=dict(config_fields or {}),
        )
        self._loaders[name] = loader
        self._definitions[name] = definition
        for alias in definition.aliases:
            self._aliases[alias] = name
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

    def list_loaders(self) -> list[str]:
        return sorted(self._definitions)

    def alias_map(self) -> dict[str, str]:
        return dict(self._aliases)

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
    is_healpix: bool = False,
    requires_file: bool = False,
    network_service: str | None = None,
    aliases: list[str] | None = None,
    config_fields: dict[str, str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator used by loader modules to register themselves."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY.register(
            name,
            func,
            config_section=config_section,
            use_flag=use_flag,
            is_healpix=is_healpix,
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
    return _REGISTRY.get_loader(name)


def get_loader_meta(name: str) -> dict[str, Any]:
    ensure_default_loaders_registered()
    return _REGISTRY.get_definition(name).meta_dict()


def get_loader_definition(name: str) -> LoaderDefinition:
    ensure_default_loaders_registered()
    return _REGISTRY.get_definition(name)


def resolve_loader_name(name: str) -> str:
    ensure_default_loaders_registered()
    return _REGISTRY.resolve_name(name)


def list_loader_definitions() -> list[LoaderDefinition]:
    ensure_default_loaders_registered()
    return _REGISTRY.definitions()


def list_loaders() -> list[str]:
    ensure_default_loaders_registered()
    return _REGISTRY.list_loaders()


def build_sky_model_map() -> dict[str, tuple[str, str, bool]]:
    """Legacy helper for programmatic Simulator sugar."""
    ensure_default_loaders_registered()
    result: dict[str, tuple[str, str, bool]] = {}
    for definition in _REGISTRY.definitions():
        if definition.requires_file:
            continue
        meta = definition.meta_dict()
        result[definition.name] = (
            meta["config_section"],
            meta["use_flag"],
            meta["is_healpix"],
        )
    return result


def build_network_services_map() -> dict[str, str]:
    """Return kind -> required network service for source-aware configs."""
    ensure_default_loaders_registered()
    result: dict[str, str] = {}
    for definition in _REGISTRY.definitions():
        if definition.network_service is not None:
            result[definition.name] = definition.network_service
    return result


def build_alias_map() -> dict[str, str]:
    ensure_default_loaders_registered()
    return _REGISTRY.alias_map()
