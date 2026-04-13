# rrivis/core/sky/discovery.py
"""Catalog discovery and memory estimation utilities.

Module-level functions for listing available sky models, querying catalog
metadata, and estimating HEALPix memory usage.  These were formerly static
methods on :class:`SkyModel` but have no dependency on instance state.
"""

from __future__ import annotations

from typing import Any

import healpy as hp
import numpy as np


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
    resolution_arcmin = np.sqrt(4 * np.pi / npix) * (180 / np.pi) * 60

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
    from .registry import loader_registry

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
    """Get metadata for any supported catalog or model.

    Parameters
    ----------
    catalog_key : str
        Catalog or model identifier (e.g. ``"gleam_egc"``, ``"racs_low"``,
        ``"gsm2008"``).
    live : bool, default=False
        If True, query VizieR/CASDA TAP for live column information.
    """
    from ._loaders_diffuse import get_diffuse_model_info
    from ._loaders_vizier import (
        get_catalog_columns,
        get_point_catalog_metadata,
        get_racs_columns,
        get_racs_metadata,
    )
    from .catalogs import DIFFUSE_MODELS, RACS_CATALOGS, VIZIER_POINT_CATALOGS
    from .registry import loader_registry

    try:
        loader_name, _ = loader_registry.resolve_request(catalog_key, {})
        definition = loader_registry.definition(loader_name)
        meta = loader_registry.meta(catalog_key)
        return {
            "name": catalog_key,
            "loader": definition.name,
            "category": definition.category,
            "representation": meta["representation"],
            "representations": meta["representations"],
            "output_mode": meta["output_mode"],
            "primary_representation": meta["primary_representation"],
            "supports_point_sources": meta["supports_point_sources"],
            "supports_healpix_map": meta["supports_healpix_map"],
            "network_service": definition.network_service,
            "requires_file": definition.requires_file,
            "aliases": list(definition.aliases),
            "config_fields": dict(definition.config_fields),
        }
    except ValueError:
        pass

    if catalog_key in VIZIER_POINT_CATALOGS:
        return (
            get_catalog_columns(catalog_key)
            if live
            else get_point_catalog_metadata(catalog_key)
        )

    if catalog_key.startswith("racs_"):
        band = catalog_key[5:]
        if band in RACS_CATALOGS:
            return get_racs_columns(band) if live else get_racs_metadata(band)

    if catalog_key in RACS_CATALOGS:
        return get_racs_columns(catalog_key) if live else get_racs_metadata(catalog_key)

    if catalog_key in DIFFUSE_MODELS:
        return get_diffuse_model_info(catalog_key)

    all_keys = (
        sorted(VIZIER_POINT_CATALOGS.keys())
        + [f"racs_{b}" for b in sorted(RACS_CATALOGS.keys())]
        + sorted(DIFFUSE_MODELS.keys())
    )
    raise ValueError(f"Unknown catalog key '{catalog_key}'. Available: {all_keys}")
