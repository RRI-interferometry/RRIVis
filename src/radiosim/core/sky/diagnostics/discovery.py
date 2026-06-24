"""Catalog discovery and memory estimation utilities.

Module-level functions for listing available sky models, querying catalog
metadata, and estimating HEALPix memory usage.  These were formerly static
methods on :class:`SkyModel` but have no dependency on instance state.
"""

from __future__ import annotations

from typing import Any

import healpy as hp
import numpy as np

from ..support.healpix_geometry import pixel_solid_angle


def estimate_healpix_memory(
    nside: int,
    n_frequencies: int,
    dtype: np.dtype | type = np.float32,
    n_stokes: int = 1,
) -> dict[str, Any]:
    """
    Estimate memory usage for multi-frequency HEALPix maps.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE parameter.
    n_frequencies : int
        Number of frequency channels.
    dtype : np.dtype or type, default=np.float32
        Data type for maps.
    n_stokes : int, default=1
        Number of Stokes components (1 for I-only, 4 for full IQUV).

    Returns
    -------
    dict
        Memory estimation with keys:
        - npix: number of pixels
        - n_freq: number of frequencies
        - n_stokes: number of Stokes components
        - bytes_per_map: bytes for one map
        - total_bytes: total memory in bytes
        - total_mb: total memory in MB
        - total_gb: total memory in GB
        - resolution_arcmin: approximate pixel resolution

    Examples
    --------
    >>> info = estimate_healpix_memory(nside=1024, n_frequencies=20)
    >>> print(f"Memory: {info['total_mb']:.1f} MB")
    Memory: 960.0 MB
    >>> info = estimate_healpix_memory(nside=1024, n_frequencies=20, n_stokes=4)
    >>> print(f"Memory: {info['total_mb']:.1f} MB")
    Memory: 3840.0 MB
    """
    npix = hp.nside2npix(nside)
    bytes_per_value = np.dtype(dtype).itemsize
    bytes_per_map = npix * bytes_per_value
    total_bytes = bytes_per_map * n_frequencies * n_stokes

    # Approximate resolution in arcminutes
    resolution_arcmin = np.sqrt(pixel_solid_angle(nside)) * (180 / np.pi) * 60

    return {
        "npix": npix,
        "n_freq": n_frequencies,
        "n_stokes": n_stokes,
        "bytes_per_map": bytes_per_map,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / 1e6,
        "total_gb": total_bytes / 1e9,
        "resolution_arcmin": resolution_arcmin,
        "dtype": np.dtype(dtype).name,
    }


def list_all_models() -> dict[str, dict[str, str]]:
    """List all available sky models and catalogs with their descriptions.

    Returns
    -------
    dict[str, dict[str, str]]
        Nested mapping: category -> {name: description}.
        Categories: "diffuse", "point_catalogs", "racs".
    """
    from ..registry import loader_registry

    groups: dict[str, dict[str, str]] = {
        "diffuse": {},
        "point_catalogs": {},
        "synthetic": {},
        "file": {},
    }
    for definition in loader_registry.definitions():
        if definition.category == "catalog":
            group = "point_catalogs"
        else:
            group = definition.category
        doc = (definition.loader.__doc__ or "").strip().splitlines()
        description = doc[0].strip() if doc else definition.name
        groups.setdefault(group, {})[definition.name] = description
        for alias in definition.aliases:
            groups[group][alias] = f"Alias for {definition.name}"
    return {key: dict(sorted(value.items())) for key, value in groups.items()}


def get_catalog_info(catalog_key: str, live: bool = False) -> dict[str, Any]:
    """Resolve a catalog/model identifier to its registry metadata.

    Every supported identifier — canonical loader name (``"gleam"``,
    ``"diffuse_sky"``), alias (``"gsm2016"``, ``"racs_low"``,
    ``"gleam_egc"``, ``"lotss_dr1"``, ``"mals_dr2"``) — resolves through
    the loader registry. Sub-catalog keys are registered as
    alias-with-bound-defaults (e.g. ``gleam_egc`` →
    ``gleam(catalog="gleam_egc")``); RACS bands likewise.

    Parameters
    ----------
    catalog_key : str
        Catalog or model identifier.
    live : bool, default=False
        If True, augment the registry metadata with live VizieR / CASDA
        column information for VizieR catalogs and RACS bands. Network
        I/O — falls back to cached metadata silently on failure.
    """
    from ..loaders.diffuse import get_diffuse_model_info
    from ..registry import loader_registry
    from ..registry.catalogs import DIFFUSE_MODELS

    try:
        loader_name, resolved_kwargs = loader_registry.resolve_request(catalog_key, {})
    except ValueError as exc:
        raise ValueError(
            f"Unknown catalog key '{catalog_key}'. Use loader_registry.names() "
            f"or loader_registry.aliases() to list valid identifiers."
        ) from exc

    definition = loader_registry.definition(loader_name)
    meta = loader_registry.meta(catalog_key)
    reps = meta["representations"]
    info: dict[str, Any] = {
        "name": catalog_key,
        "loader": definition.name,
        "resolved_loader": loader_name,
        "resolved_kwargs": dict(resolved_kwargs),
        "category": definition.category,
        "representation": reps[0],
        "representations": reps,
        "output_mode": meta["output_mode"],
        "primary_representation": reps[0],
        "supports_point_sources": "point_sources" in reps,
        "supports_healpix_map": "healpix_map" in reps,
        "network_service": definition.network_service,
        "requires_file": definition.requires_file,
        "aliases": list(definition.aliases),
        "config_fields": dict(definition.config_fields),
    }
    model_name = resolved_kwargs.get("model")
    if isinstance(model_name, str) and model_name in DIFFUSE_MODELS:
        info["diffuse_model"] = model_name
        info["diffuse_model_info"] = get_diffuse_model_info(model_name)

    if live:
        info.update(_fetch_live_columns(loader_name, resolved_kwargs))
    return info


def _fetch_live_columns(
    loader_name: str, resolved_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Augment registry metadata with live VizieR / CASDA column info.

    Best-effort: returns an empty dict if the loader has no live source
    or the network call fails. Pulled out of ``get_catalog_info`` so the
    primary path stays a single registry lookup.
    """
    from ..loaders.vizier import (
        get_catalog_columns,
        get_racs_columns,
    )

    try:
        if loader_name in ("gleam", "mals", "lotss"):
            sub_key = resolved_kwargs.get("catalog") or resolved_kwargs.get("release")
            if sub_key is None:
                return {}
            if loader_name in ("mals", "lotss"):
                sub_key = f"{loader_name}_{sub_key.lower()}"
            return {"live_columns": get_catalog_columns(sub_key)}
        if loader_name == "racs":
            band = resolved_kwargs.get("band")
            if band is None:
                return {}
            return {"live_columns": get_racs_columns(band)}
    except Exception:
        return {}
    return {}
