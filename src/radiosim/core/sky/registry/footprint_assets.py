"""Filesystem loading of packaged catalog footprint assets.

Kept separate from :mod:`registry.catalogs` so the catalog *metadata*
module stays pure data (frozen Pydantic entries) with no filesystem IO.
"""

from __future__ import annotations

from functools import cache
from importlib import resources

import numpy as np

from ..containers import SkyFootprint

# Package-relative location of the packaged NPZ footprint assets.
_FOOTPRINT_RESOURCE_PATH = ("core", "sky", "data", "footprints")


@cache
def load_catalog_footprint_asset(asset_name: str) -> SkyFootprint:
    """Load a packaged catalog footprint asset by file name."""
    resource = resources.files("radiosim")
    for part in _FOOTPRINT_RESOURCE_PATH:
        resource = resource.joinpath(part)
    resource = resource.joinpath(asset_name)
    with resource.open("rb") as handle, np.load(handle, allow_pickle=False) as payload:
        nside = int(payload["nside"])
        coordinate_frame = str(np.asarray(payload["coordinate_frame"]).item())
        hpx_inds = np.asarray(payload["hpx_inds"], dtype=np.int64)
    return SkyFootprint(
        nside=nside,
        coordinate_frame=coordinate_frame,
        hpx_inds=hpx_inds,
    )
