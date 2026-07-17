"""Central loader registry for sky-model sources."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
from typing import Any, Literal

LoaderCategory = Literal["catalog", "diffuse", "synthetic", "file"]
LoaderRepresentation = Literal["point_sources", "healpix_map"]
LoaderOutputMode = Literal["point_only", "healpix_only", "polymorphic"]
LoaderPathKind = Literal["file", "file_list", "glob"]

_BUILTIN_PATH_OPTIONS: Mapping[tuple[str, str], Mapping[str, LoaderPathKind]] = (
    MappingProxyType(
        {
            ("radiosim.core.sky.loaders.bbs", "bbs"): MappingProxyType(
                {"filename": "file"}
            ),
            ("radiosim.core.sky.loaders.fits", "fits_image"): MappingProxyType(
                {"filename": "file"}
            ),
            (
                "radiosim.core.sky.loaders.pyradiosky",
                "pyradiosky_file",
            ): MappingProxyType({"filename": "file"}),
            (
                "radiosim.core.sky.loaders.skyh5_multifile",
                "skyh5_multifile",
            ): MappingProxyType({"file_glob": "glob", "filenames": "file_list"}),
        }
    )
)

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

    return ("point_sources",)


def _normalize_config_fields(
    config_fields: Mapping[str, str] | Sequence[str] | None,
) -> dict[str, str]:
    """Normalize config field metadata to loader-kwarg mapping form.

    ``config_fields`` accepts either an explicit ``{loader_kwarg: config_key}``
    mapping or a **list shorthand**: a sequence of loader argument names where
    each name is also the YAML config key (``["filename", "nside"]`` →
    ``{"filename": "filename", "nside": "nside"}``). Use the list form when
    the mapping is identity to reduce registration boilerplate.
    """

    if config_fields is None:
        return {}
    if isinstance(config_fields, Mapping):
        return dict(config_fields)
    return {field: field for field in config_fields}


def _assert_config_fields_match_signature(
    name: str,
    loader: Callable[..., Any],
    config_fields: Mapping[str, str] | None,
) -> None:
    """Assert every ``config_fields`` key is an accepted loader argument.

    The keys of ``config_fields`` are loader argument names that
    :mod:`radiosim.io.config` forwards as keyword arguments. A typo there
    would silently drop a config field, so we fail loudly at registration
    time instead. Loaders that accept ``**kwargs`` accept any name and are
    therefore exempt.
    """
    if not config_fields:
        return
    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError):
        return
    parameters = signature.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return
    accepted = {
        p.name
        for p in parameters
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    unknown = sorted(key for key in config_fields if key not in accepted)
    if unknown:
        raise ValueError(
            f"Loader '{name}' declares config_fields {unknown} that are not "
            f"parameters of {getattr(loader, '__qualname__', loader)!r}. "
            f"Accepted parameters: {sorted(accepted)}."
        )


def _normalize_path_options(
    name: str,
    *,
    requires_file: bool,
    config_fields: Mapping[str, str],
    path_options: Mapping[str, LoaderPathKind] | None,
) -> Mapping[str, LoaderPathKind]:
    """Validate explicit loader-argument path semantics."""
    normalized: dict[str, LoaderPathKind] = dict(path_options or {})
    invalid_kinds = sorted(
        (key, value)
        for key, value in normalized.items()
        if value not in {"file", "file_list", "glob"}
    )
    if invalid_kinds:
        raise ValueError(
            f"Loader {name!r} declares invalid path_options: {invalid_kinds}. "
            "Allowed kinds are 'file', 'file_list', and 'glob'."
        )
    unknown = sorted(set(normalized) - set(config_fields))
    if unknown:
        raise ValueError(
            f"Loader {name!r} declares path_options {unknown} that are not "
            "declared loader arguments in config_fields."
        )
    if requires_file and not normalized:
        raise ValueError(
            f"Loader {name!r} sets requires_file=True but declares no "
            "path_options; file semantics must be explicit."
        )
    return MappingProxyType(normalized)


def _empty_alias_defaults() -> dict[str, dict[str, Any]]:
    return {}


def _empty_config_fields() -> dict[str, str]:
    return {}


def _empty_path_options() -> Mapping[str, LoaderPathKind]:
    return MappingProxyType({})


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
    alias_defaults: dict[str, dict[str, Any]] = field(
        default_factory=_empty_alias_defaults
    )
    config_fields: dict[str, str] = field(default_factory=_empty_config_fields)
    path_options: Mapping[str, LoaderPathKind] = field(
        default_factory=_empty_path_options
    )

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
        """Serializable metadata with a single canonical representation set.

        ``representations`` is the one source of truth for what this loader
        can emit; ``output_mode`` is its derived label. Callers that want the
        boolean ``supports_*`` views or the ``primary_representation`` read the
        corresponding :class:`LoaderDefinition` properties rather than relying
        on denormalized copies in this dict.
        """
        return {
            "config_section": self.config_section or self.name,
            "use_flag": self.use_flag or f"use_{self.name}",
            "representations": list(self.representations),
            "output_mode": self.output_mode,
            "category": self.category,
            "requires_file": self.requires_file,
            "network_service": self.network_service,
            "aliases": list(self.aliases),
            "alias_defaults": {
                alias: dict(defaults) for alias, defaults in self.alias_defaults.items()
            },
            "config_fields": dict(self.config_fields),
            "path_options": dict(self.path_options),
        }


class LoaderRegistry:
    """Mutable registry of sky loaders plus their metadata."""

    def __init__(self) -> None:
        self._loaders: dict[str, Callable[..., Any]] = {}
        self._definitions: dict[str, LoaderDefinition] = {}
        self._aliases: dict[str, str] = {}
        self._alias_defaults: dict[str, dict[str, Any]] = {}
        # Guards the four mutable dicts against concurrent register/unregister
        # (the advertised parallel-loading path can register from threads).
        self._lock = threading.Lock()

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
        config_fields: Mapping[str, str] | Sequence[str] | None = None,
        path_options: Mapping[str, LoaderPathKind] | None = None,
    ) -> Callable[..., Any]:
        alias_names, alias_defaults = self._normalize_aliases(aliases)
        normalized_config_fields = _normalize_config_fields(config_fields)
        _assert_config_fields_match_signature(name, loader, normalized_config_fields)
        normalized_representations = _normalize_representations(
            representations,
            loader=loader,
            config_fields=normalized_config_fields,
        )
        declared_path_options = path_options
        if declared_path_options is None:
            declared_path_options = _BUILTIN_PATH_OPTIONS.get((loader.__module__, name))
        normalized_path_options = _normalize_path_options(
            name,
            requires_file=requires_file,
            config_fields=normalized_config_fields,
            path_options=declared_path_options,
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
            config_fields=normalized_config_fields,
            path_options=normalized_path_options,
        )
        with self._lock:
            self._loaders[name] = loader
            self._definitions[name] = definition
            for alias in definition.aliases:
                self._aliases[alias] = name
                self._alias_defaults[alias] = dict(
                    definition.alias_defaults.get(alias, {})
                )
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

    def definitions(self) -> list[LoaderDefinition]:
        return [self._definitions[name] for name in self.list_loaders()]

    def unregister(self, name: str) -> None:
        """Remove a loader, its definition, and any aliases pointing at it.

        Intended for test cleanup; production code should not call this.
        """
        canonical = self.resolve_name(name)
        with self._lock:
            _ = self._loaders.pop(canonical, None)
            self._definitions.pop(canonical, None)
            for alias in [a for a, c in self._aliases.items() if c == canonical]:
                self._aliases.pop(alias, None)
                self._alias_defaults.pop(alias, None)


@dataclass(frozen=True)
class ResolvedLoader:
    """Alias-resolved invocation of a registered sky loader.

    Calling an instance applies the loader's alias-bound default kwargs
    (if any) before forwarding to the canonical function. Inspect
    ``canonical_name`` / ``alias_defaults`` to see what was resolved.
    """

    canonical_name: str
    definition: LoaderDefinition
    alias_defaults: dict[str, Any]

    def __call__(self, **kwargs: Any) -> Any:
        if not self.alias_defaults:
            return self.definition.loader(**kwargs)
        merged = dict(self.alias_defaults)
        merged.update(kwargs)
        return self.definition.loader(**merged)


_REGISTRY = LoaderRegistry()
_DEFAULT_LOADER_MODULES = (
    "radiosim.core.sky.loaders.bbs",
    "radiosim.core.sky.loaders.diffuse",
    "radiosim.core.sky.loaders.fits",
    "radiosim.core.sky.loaders.pyradiosky",
    "radiosim.core.sky.loaders.skyh5_multifile",
    "radiosim.core.sky.loaders.synthetic",
    "radiosim.core.sky.loaders.vizier",
    "radiosim.core.sky.recipes.realistic_foreground",
)
_DEFAULT_LOADERS_IMPORTED = False
_DEFAULT_LOADERS_LOCK = threading.Lock()


def _ensure_default_loaders_registered() -> None:
    """Import built-in loader modules exactly once (thread-safe)."""
    global _DEFAULT_LOADERS_IMPORTED
    if _DEFAULT_LOADERS_IMPORTED:
        return
    with _DEFAULT_LOADERS_LOCK:
        if _DEFAULT_LOADERS_IMPORTED:
            return
        for module_name in _DEFAULT_LOADER_MODULES:
            import_module(module_name)
        _DEFAULT_LOADERS_IMPORTED = True


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
    config_fields: Mapping[str, str] | Sequence[str] | None = None,
    path_options: Mapping[str, LoaderPathKind] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator used by loader modules to register themselves.

    ``config_fields`` may be a mapping or a **list shorthand** of loader
    argument names (each name doubles as the YAML config key). See
    :func:`_normalize_config_fields`.

    Internal helper — call ``loader_registry.register_loader(...)`` from the
    public facade rather than importing this directly.
    """

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
            path_options=path_options,
        )
        return func

    return decorator
