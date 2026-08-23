"""Frozen container dataclasses for sky payloads.

Public surface — re-exports every container so callers don't need to
know which sub-module each class lives in. The dedicated modules
(``footprint``, ``provenance``, ``point``, ``healpix``) are still
importable directly when stricter dependency hygiene is wanted.
"""

from .footprint import (
    DEFAULT_COVERAGE_FOOTPRINT_COORDINATE_FRAME,
    DEFAULT_COVERAGE_FOOTPRINT_NSIDE,
    MonopoleConvention,
    SkyCoverage,
    SkyFootprint,
    SourceSubtractionStatus,
)
from .healpix import HealpixData
from .point import (
    PointMetadata,
    PointMorphology,
    PointPolarization,
    PointSourceData,
    PointSpectrum,
    SourceArrays,
    TangentPolarizationFrame,
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
    "TangentPolarizationFrame",
    "SourceArrays",
    "SourceSubtractionStatus",
    "empty_source_arrays",
]
