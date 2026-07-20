"""Internal support helpers for the sky-model stack.

These modules consolidate logic shared across loaders, combine, operations,
and containers. They are not part of the top-level :mod:`radiosim.core.sky`
public surface, but selective entry points are re-exported here for tests and
advanced callers that need the same primitives without reaching into private
module paths.
"""

from __future__ import annotations

from ..containers._shared import validate_frequency_axis
from . import healpix_geometry
from .frequencies import validate_observation_frequencies
from .healpix_geometry import pixel_solid_angle
from .precision import get_sky_storage_dtype, require_precision
from .region_filter import apply_point_region_filter


def __getattr__(name: str):
    if name == "point_source_data_from_mapping":
        from .point_builder import point_source_data_from_mapping

        return point_source_data_from_mapping
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "healpix_geometry",
    "apply_point_region_filter",
    "get_sky_storage_dtype",
    "pixel_solid_angle",
    "point_source_data_from_mapping",
    "require_precision",
    "validate_observation_frequencies",
    "validate_frequency_axis",
]
