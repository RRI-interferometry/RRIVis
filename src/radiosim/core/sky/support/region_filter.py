"""Shared client-side point-source region filtering helpers.

Region filtering convention
---------------------------
Loaders follow a **dual convention** depending on data source:

**Server-side pre-filter (network catalogs).** VizieR and RACS loaders push
spatial constraints into the remote query (``query_region`` / ADQL ``CONTAINS``)
to reduce download volume. A **client-side trim** via
:func:`apply_point_region_filter` still runs afterwards so the final arrays
match RadioSim :class:`~radiosim.core.sky.operations.region.SkyRegion` semantics
exactly (union logic, coordinate-frame handling, and any server approximation
gaps).

**Client-side post-build (file and synthetic loaders).** FITS, BBS, skyh5,
pyradiosky, and synthetic loaders read or generate full tables/maps first, then
apply :func:`apply_point_region_filter` (point paths) or HEALPix mask cropping
(diffuse paths) once columnar arrays or maps exist.

Use :meth:`~radiosim.core.sky.containers.model.SkyModel.filter_region` to apply
the same region semantics to an already-built model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..operations.region import SkyRegion


def apply_point_region_filter(
    arrays: Mapping[str, np.ndarray | None],
    region: SkyRegion | None,
    *,
    ra_key: str = "ra_rad",
    dec_key: str = "dec_rad",
) -> dict[str, np.ndarray | None]:
    """Return point-source arrays masked to ``region``.

    Parameters
    ----------
    arrays
        Columnar arrays keyed by name. Every array whose length matches the
        coordinate columns is masked in lockstep.
    region
        Sky region to apply. When ``None``, returns a shallow copy unchanged.
    ra_key, dec_key
        Keys for right-ascension and declination columns in radians.
    """
    result = dict(arrays)
    ra = result.get(ra_key)
    dec = result.get(dec_key)
    if region is None or ra is None or dec is None:
        return result

    mask = np.asarray(region.contains(ra, dec), dtype=bool)
    for key, values in result.items():
        if values is not None and len(values) == len(mask):
            result[key] = values[mask]
    return result


def deduplicate_union_point_sources(
    arrays: Mapping[str, np.ndarray | None],
    *,
    source_id_key: str = "source_id",
    source_name_key: str = "source_name",
    ra_key: str = "ra_rad",
    dec_key: str = "dec_rad",
    coord_decimals: int = 8,
) -> dict[str, np.ndarray | None]:
    """Deduplicate sources that appear in multiple union sub-regions.

    Prefer ``source_id``, then ``source_name``, then rounded ``(ra, dec)``
    coordinate pairs as the uniqueness key. Returns arrays indexed by the
    first occurrence of each key.
    """
    result = dict(arrays)
    ra = result.get(ra_key)
    if ra is None or len(ra) == 0:
        return result

    source_id = result.get(source_id_key)
    source_name = result.get(source_name_key)
    dec = result.get(dec_key)
    unique_idx = None
    if source_id is not None and np.all(source_id != ""):
        _, unique_idx = np.unique(source_id, return_index=True)
    elif source_name is not None and np.all(source_name != ""):
        _, unique_idx = np.unique(source_name, return_index=True)
    elif dec is not None:
        coords_key = np.round(np.column_stack([ra, dec]), decimals=coord_decimals)
        _, unique_idx = np.unique(coords_key, axis=0, return_index=True)
    if unique_idx is None:
        return result

    unique_idx = np.sort(unique_idx)
    for key, values in result.items():
        if values is not None and len(values) == len(ra):
            result[key] = values[unique_idx]
    return result
