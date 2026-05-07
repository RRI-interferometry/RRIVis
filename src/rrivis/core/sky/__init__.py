# rrivis/core/sky/__init__.py
"""Unified sky model package for RRIVis.

Use :func:`prepare_sky_model` as the canonical user entry point for combining
and materialising sky models.  The lower-level ``_combine_models`` engine
lives in :mod:`rrivis.core.sky.combine` and is intended for advanced /
internal use.

Sparse :class:`HealpixData` is the canonical form for partial-sky inputs and
propagates losslessly through load → combine → simulate.  Operations that
genuinely need a full-sky array (plotting, harmonic regridding, lightcurves,
observability projections, bright-source subtraction) raise — densify
explicitly with ``sky.replace(healpix=sky.healpix.to_dense())`` rather than
relying on implicit densification.
"""

from ._data import (
    HealpixData,
    MonopoleConvention,
    PointSourceData,
    PointSpectrum,
    SkyCoverage,
    SkyFootprint,
    SkyProvenance,
    SourceArrays,
    SourceSubtractionStatus,
)
from ._factories import create_empty, create_from_arrays, create_test_sources
from ._loaders_bbs import write_bbs
from ._serialization import load_skyh5, save_skyh5, to_pyradiosky
from .combine import regrid_healpix_model
from .constants import (
    C_LIGHT,
    H_PLANCK,
    K_BOLTZMANN,
    BrightnessConversion,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
    rayleigh_jeans_factor,
)
from .discovery import estimate_healpix_memory, get_catalog_info, list_all_models
from .loaders import (
    load_3c,
    load_bbs,
    load_diffuse_sky,
    load_fits_image,
    load_gleam,
    load_lotss,
    load_mals,
    load_nvss,
    load_pyradiosky_file,
    load_pysm3,
    load_racs,
    load_sumss,
    load_test_sources,
    load_tgss,
    load_vlass,
    load_vlssr,
    load_wenss,
)
from .model import (
    SkyFormat,
    SkyModel,
)
from .operations import (
    compute_linear_polarization,
    materialize_healpix_model,
    materialize_point_sources_model,
    subtract_bright_sources,
    with_memmap_backing,
    with_monopole,
    with_monopole_subtracted,
)
from .pipeline import prepare_sky_model
from .plotter import SkyPlotter
from .recipes import realistic_foreground_sky
from .region import BoxRegion, ConeRegion, SkyRegion, UnionRegion
from .spectral import apply_faraday_rotation, compute_spectral_scale

__all__ = [
    "SkyModel",
    "SkyPlotter",
    "SkyRegion",
    "create_empty",
    "create_from_arrays",
    "create_test_sources",
    "compute_linear_polarization",
    "materialize_healpix_model",
    "materialize_point_sources_model",
    "subtract_bright_sources",
    "with_memmap_backing",
    "with_monopole",
    "with_monopole_subtracted",
    "to_pyradiosky",
    "save_skyh5",
    "load_skyh5",
    "write_bbs",
    "ConeRegion",
    "BoxRegion",
    "UnionRegion",
    "realistic_foreground_sky",
    "K_BOLTZMANN",
    "C_LIGHT",
    "H_PLANCK",
    "BrightnessConversion",
    "brightness_temp_to_flux_density",
    "flux_density_to_brightness_temp",
    "compute_spectral_scale",
    "apply_faraday_rotation",
    "SkyFormat",
    "SourceArrays",
    "PointSourceData",
    "PointSpectrum",
    "HealpixData",
    "SkyProvenance",
    "SkyCoverage",
    "SkyFootprint",
    "MonopoleConvention",
    "SourceSubtractionStatus",
    "estimate_healpix_memory",
    "list_all_models",
    "get_catalog_info",
    "rayleigh_jeans_factor",
    # Loaders (typed re-exports)
    "load_gleam",
    "load_3c",
    "load_mals",
    "load_lotss",
    "load_nvss",
    "load_racs",
    "load_sumss",
    "load_tgss",
    "load_vlass",
    "load_vlssr",
    "load_wenss",
    "load_diffuse_sky",
    "load_pysm3",
    "load_fits_image",
    "load_pyradiosky_file",
    "load_bbs",
    "load_test_sources",
    # Orchestration
    "prepare_sky_model",
    "regrid_healpix_model",
]
