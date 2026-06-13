# radiosim/core/sky/__init__.py
"""Unified sky model package for RadioSim.

Use :func:`prepare_sky_model` as the canonical user entry point for combining
and materialising sky models.  The lower-level ``_combine_models`` engine
lives in :mod:`radiosim.core.sky.combine` and is intended for advanced /
internal use.

Sparse :class:`HealpixData` is the canonical form for partial-sky inputs and
propagates losslessly through load → combine → simulate.  Operations that
genuinely need a full-sky array (plotting, harmonic regridding, lightcurves,
observability projections, bright-source subtraction) raise — densify
explicitly with ``sky.replace(healpix=sky.healpix.to_dense())`` rather than
relying on implicit densification.
"""

from .combine.engine import regrid_healpix_model
from .combine.options import PrepareSkyOptions
from .combine.pipeline import prepare_sky_model
from .containers import (
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
from .containers.constants import (
    C_LIGHT,
    H_PLANCK,
    K_BOLTZMANN,
    BrightnessConversion,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
    rayleigh_jeans_factor,
)
from .containers.model import (
    SkyFormat,
    SkyModel,
)
from .containers.spectral import apply_faraday_rotation, compute_spectral_scale
from .diagnostics.discovery import (
    estimate_healpix_memory,
    get_catalog_info,
    list_all_models,
)
from .io.serialization import load_skyh5, save_skyh5, to_pyradiosky
from .loaders import (
    load_3c,
    load_bbs,
    load_diffuse_sky,
    load_fits_image,
    load_gleam,
    load_lotss,
    load_mals,
    load_nvss,
    load_poisson_confusion,
    load_pyradiosky_file,
    load_pysm3,
    load_racs,
    load_skyh5_multifile,
    load_sumss,
    load_test_sources,
    load_tgss,
    load_vlass,
    load_vlssr,
    load_wenss,
)
from .loaders.bbs import write_bbs
from .operations.factories import create_empty, create_from_arrays, create_test_sources
from .operations.operations import (
    compute_linear_polarization,
    materialize_healpix_model,
    materialize_point_sources_model,
    with_memmap_backing,
    with_monopole,
    with_monopole_subtracted,
)
from .operations.region import BoxRegion, ConeRegion, SkyRegion, UnionRegion
from .operations.subtraction import subtract_bright_sources
from .recipes.realistic_foreground import realistic_foreground_sky

__all__ = [
    "SkyModel",
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
    "load_poisson_confusion",
    "load_skyh5_multifile",
    # Orchestration
    "PrepareSkyOptions",
    "prepare_sky_model",
    "regrid_healpix_model",
]
