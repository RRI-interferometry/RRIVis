"""Loader registry for SkyModel factory methods."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_LOADERS: dict[str, Callable[..., Any]] = {}
_LOADER_META: dict[str, dict[str, Any]] = {}


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
) -> Callable:
    """Decorator: register a sky model loader function by name.

    Parameters
    ----------
    name : str
        Registry name for the loader.
    config_section : str, optional
        Corresponding section name in ``SkyModelConfig``.  Defaults to *name*.
    use_flag : str, optional
        Boolean flag field in the config section.  Defaults to ``use_{name}``.
    is_healpix : bool
        Whether this loader produces HEALPix output.
    requires_file : bool
        Whether this loader requires a ``filename`` config field.
    network_service : str, optional
        Network service this loader depends on (e.g. ``"vizier"``,
        ``"casda"``, ``"pygdsm_data"``, ``"pysm3_data"``).  ``None`` for
        local-only loaders.
    aliases : list of str, optional
        Alternative names that map to this loader (e.g. ``["gsm", "gsm2008"]``
        for the ``diffuse_sky`` loader).
    config_fields : dict[str, str], optional
        Maps config field names to loader kwarg names.  Used by
        ``build_loader_kwargs()`` to generically extract kwargs from a
        config section dict.  Example: ``{"gleam_catalogue": "catalog"}``
        means ``config["gleam_catalogue"]`` is passed as ``catalog=`` to
        the loader.
    """

    def decorator(func: Callable) -> Callable:
        _LOADERS[name] = func
        _LOADER_META[name] = {
            "config_section": config_section or name,
            "use_flag": use_flag or f"use_{name}",
            "is_healpix": is_healpix,
            "requires_file": requires_file,
            "network_service": network_service,
            "aliases": aliases or [],
            "config_fields": config_fields or {},
        }
        return func

    return decorator


def get_loader(name: str) -> Callable:
    """Get a registered loader by name or alias.

    Resolves aliases (e.g. ``"gsm"`` -> ``"diffuse_sky"``) before lookup.
    """
    if name in _LOADERS:
        return _LOADERS[name]
    # Check aliases
    for canonical, meta in _LOADER_META.items():
        if name in meta.get("aliases", []):
            return _LOADERS[canonical]
    raise ValueError(
        f"Unknown sky model loader '{name}'. Available: {sorted(_LOADERS.keys())}"
    )


def get_loader_meta(name: str) -> dict[str, Any]:
    """Get metadata for a registered loader by name or alias.

    Resolves aliases (e.g. ``"gsm"`` -> ``"diffuse_sky"``) before lookup.
    """
    if name in _LOADER_META:
        return _LOADER_META[name]
    # Check aliases
    for canonical, meta in _LOADER_META.items():
        if name in meta.get("aliases", []):
            return _LOADER_META[canonical]
    raise ValueError(
        f"No metadata for loader '{name}'. Available: {sorted(_LOADER_META.keys())}"
    )


def list_loaders() -> list[str]:
    """List registered loader names."""
    return sorted(_LOADERS.keys())


def build_sky_model_map() -> dict[str, tuple[str, str, bool]]:
    """Build the sky model name -> (config_section, use_flag, is_healpix) mapping.

    Generates the mapping from registered loader metadata, excluding
    loaders that require a file path (bbs, fits_image, pyradiosky_file).

    Returns
    -------
    dict[str, tuple[str, str, bool]]
        Mapping of loader name to ``(config_section, use_flag, is_healpix)``.
    """
    result: dict[str, tuple[str, str, bool]] = {}
    for name, meta in _LOADER_META.items():
        if meta.get("requires_file", False):
            continue
        result[name] = (meta["config_section"], meta["use_flag"], meta["is_healpix"])
    return result


def build_network_services_map() -> dict[tuple[str, str], str]:
    """Build the (config_section, use_flag) -> network_service mapping.

    Generates a mapping suitable for network connectivity checks from
    registered loader metadata.  Only loaders with a non-None
    ``network_service`` are included.

    Returns
    -------
    dict[tuple[str, str], str]
        Mapping of ``(config_section, use_flag)`` to service name.
    """
    result: dict[tuple[str, str], str] = {}
    for meta in _LOADER_META.values():
        service = meta.get("network_service")
        if service is not None:
            result[(meta["config_section"], meta["use_flag"])] = service
    return result


def build_alias_map() -> dict[str, str]:
    """Build a mapping of alias names to canonical loader names.

    Returns
    -------
    dict[str, str]
        Mapping of alias to canonical loader name.
    """
    result: dict[str, str] = {}
    for name, meta in _LOADER_META.items():
        for alias in meta.get("aliases", []):
            result[alias] = name
    return result


def build_loader_kwargs(
    name: str,
    config_section: dict[str, Any],
    flux_multiplier: float = 1.0,
    region: Any = None,
    obs_freq_config: dict[str, Any] | None = None,
    brightness_conversion: str | None = None,
) -> dict[str, Any]:
    """Build loader kwargs from a config section using registry metadata.

    Extracts keyword arguments for a registered loader from the
    corresponding config section dict.  Handles flux unit conversion
    and passes ``obs_frequency_config`` for HEALPix loaders.

    Parameters
    ----------
    name : str
        Registered loader name.
    config_section : dict
        The config dict for this loader's section (e.g. ``config["gleam"]``).
    flux_multiplier : float, default 1.0
        Multiplier for ``flux_limit`` values (e.g. 1e-3 for mJy -> Jy).
    region : SkyRegion or None
        Sky region filter to pass to the loader.
    obs_freq_config : dict or None
        Observation frequency config, passed to HEALPix loaders.
    brightness_conversion : str or None
        Brightness conversion method (``"planck"`` or ``"rayleigh-jeans"``).
        When ``None``, loaders use their own default.

    Returns
    -------
    dict[str, Any]
        Keyword arguments ready to pass to the loader function.
    """
    if name not in _LOADER_META:
        raise ValueError(f"Unknown loader '{name}'.")

    meta = _LOADER_META[name]
    kwargs: dict[str, Any] = {}

    if region is not None:
        kwargs["region"] = region

    if brightness_conversion is not None:
        kwargs["brightness_conversion"] = brightness_conversion

    for config_key, loader_kwarg in meta.get("config_fields", {}).items():
        if config_key in config_section:
            val = config_section[config_key]
            if loader_kwarg == "flux_limit":
                val = val * flux_multiplier
            kwargs[loader_kwarg] = val

    if meta.get("is_healpix") and obs_freq_config is not None:
        kwargs["obs_frequency_config"] = obs_freq_config

    return kwargs
