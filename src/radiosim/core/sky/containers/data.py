"""Backwards-compatibility re-export shim.

Historically every container dataclass lived in this single 1700-LOC
module. They now live in dedicated files under ``containers/``:

* :mod:`._shared`     — ``_FROZEN_NDARRAY_CONFIG``, ``_arrays_equal``
* :mod:`.footprint`   — coverage / monopole / subtraction enums and ``SkyFootprint``
* :mod:`.provenance`  — ``SkyProvenance``
* :mod:`.point`       — ``PointSpectrum``, ``PointSourceData``, ``SourceArrays``
* :mod:`.healpix`     — ``HealpixData``

Existing ``from radiosim.core.sky.containers.data import …`` imports
keep working through this shim. New code should import from the
dedicated module or from :mod:`radiosim.core.sky.containers`.
"""

from __future__ import annotations

from ._shared import _FROZEN_NDARRAY_CONFIG, _arrays_equal
from .footprint import (
    DEFAULT_COVERAGE_FOOTPRINT_COORDINATE_FRAME,
    DEFAULT_COVERAGE_FOOTPRINT_NSIDE,
    MonopoleConvention,
    SkyCoverage,
    SkyFootprint,
    SourceSubtractionStatus,
    _normalize_coordinate_frame,
)
from .healpix import HealpixData
from .point import (
    PointMetadata,
    PointMorphology,
    PointPolarization,
    PointSourceData,
    PointSpectrum,
    SourceArrays,
    empty_source_arrays,
)
from .provenance import SkyProvenance

__all__ = [
    "DEFAULT_COVERAGE_FOOTPRINT_COORDINATE_FRAME",
    "DEFAULT_COVERAGE_FOOTPRINT_NSIDE",
    "HealpixData",
    "MonopoleConvention",
    "PointMetadata",
    "PointMorphology",
    "PointPolarization",
    "PointSourceData",
    "PointSpectrum",
    "SkyCoverage",
    "SkyFootprint",
    "SkyProvenance",
    "SourceArrays",
    "SourceSubtractionStatus",
    "_FROZEN_NDARRAY_CONFIG",
    "_arrays_equal",
    "_normalize_coordinate_frame",
    "empty_source_arrays",
]
