"""Mutation-free sky-model operations.

Groups the functional transforms, factory constructors, spatial region
filters, and conversion/subtraction helpers that operate on
:class:`~radiosim.core.sky.SkyModel` without mutating stored payloads.

The parallel-loading machinery lives in :mod:`.parallel`; the linear
polarisation diagnostic now lives in
:mod:`radiosim.core.sky.diagnostics.polarization`.
"""

from __future__ import annotations

from .convert import (
    bin_per_channel_flux,
    bin_scaled_flux,
    healpix_map_to_point_arrays,
    point_sources_to_healpix_maps,
)
from .factories import (
    create_empty,
    create_from_arrays,
    create_from_freq_dict_maps,
    create_test_sources,
)
from .operations import (
    materialize_healpix_model,
    materialize_point_sources_model,
    with_memmap_backing,
    with_monopole,
    with_monopole_subtracted,
)
from .parallel import (
    SkyLoadAggregateError,
    SkyLoadError,
    load_models_parallel,
    recommend_executor_for_loaders,
)
from .region import BoxRegion, ConeRegion, SkyRegion, UnionRegion
from .subtraction import subtract_bright_sources

__all__ = [
    # factories
    "create_empty",
    "create_from_arrays",
    "create_from_freq_dict_maps",
    "create_test_sources",
    # transforms
    "materialize_healpix_model",
    "materialize_point_sources_model",
    "with_memmap_backing",
    "with_monopole",
    "with_monopole_subtracted",
    # conversion
    "bin_per_channel_flux",
    "bin_scaled_flux",
    "healpix_map_to_point_arrays",
    "point_sources_to_healpix_maps",
    # regions
    "SkyRegion",
    "ConeRegion",
    "BoxRegion",
    "UnionRegion",
    # subtraction
    "subtract_bright_sources",
    # parallel loading
    "SkyLoadError",
    "SkyLoadAggregateError",
    "load_models_parallel",
    "recommend_executor_for_loaders",
]
