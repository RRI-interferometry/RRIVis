"""Shared client-side point-source region filtering helpers."""

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

    Network catalog loaders can use server-side spatial constraints first, but
    still apply this local trim to enforce exact RadioSim region semantics. File
    and synthetic point-source paths apply the same client-side convention after
    reading or generating their data.
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
