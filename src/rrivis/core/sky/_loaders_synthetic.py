"""Synthetic sky-model loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._registry import register_loader

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .model import SkyModel
    from .region import SkyRegion


@register_loader(
    "test_sources",
    config_section="test_sources",
    use_flag="use_test_sources",
    category="synthetic",
    aliases={"test": {}, "test_healpix": {"representation": "healpix_map"}},
    config_fields={
        "num_sources": "num_sources",
        "distribution": "distribution",
        "seed": "seed",
        "flux_min": "flux_min",
        "flux_max": "flux_max",
        "dec_deg": "dec_deg",
        "dec_range_deg": "dec_range_deg",
        "spectral_index": "spectral_index",
        "representation": "representation",
        "nside": "nside",
        "polarization_fraction": "polarization_fraction",
        "polarization_angle_deg": "polarization_angle_deg",
        "stokes_v_fraction": "stokes_v_fraction",
    },
)
def load_test_sources(
    num_sources: int = 100,
    flux_min: float | None = None,
    flux_max: float | None = None,
    dec_deg: float = -30.0,
    spectral_index: float = -0.7,
    distribution: str = "uniform",
    seed: int | None = None,
    dec_range_deg: float | None = None,
    representation: str = "point_sources",
    nside: int = 64,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    reference_frequency: float | None = None,
    brightness_conversion: str = "planck",
    *,
    precision: PrecisionConfig,
    polarization_fraction: float = 0.0,
    polarization_angle_deg: float = 0.0,
    stokes_v_fraction: float = 0.0,
    region: SkyRegion | None = None,
    memmap_path: str | None = None,
) -> SkyModel:
    """Generate synthetic test sources in point or HEALPix form."""
    from rrivis.utils.frequency import parse_frequency_config

    from ._factories import create_test_sources
    from .model import SkyFormat, SkyModel
    from .operations import materialize_healpix_model

    flux_range = (
        (flux_min, flux_max)
        if flux_min is not None and flux_max is not None
        else (1.0, 10.0)
    )
    sky = create_test_sources(
        num_sources=num_sources,
        flux_range=flux_range,
        dec_deg=dec_deg,
        spectral_index=spectral_index,
        distribution=distribution,
        seed=seed,
        dec_range_deg=dec_range_deg,
        brightness_conversion=brightness_conversion,
        precision=precision,
        polarization_fraction=polarization_fraction,
        polarization_angle_deg=polarization_angle_deg,
        stokes_v_fraction=stokes_v_fraction,
    )

    if region is not None:
        sky = sky.filter_region(region)

    target = SkyModel._coerce_representation(representation)
    if target == SkyFormat.HEALPIX:
        if frequencies is None and obs_frequency_config is not None:
            frequencies = parse_frequency_config(obs_frequency_config)
        ref_frequency = reference_frequency or (
            float(frequencies[0])
            if frequencies is not None and len(frequencies) > 0
            else None
        )
        if ref_frequency is not None:
            sky = sky.with_reference_frequency(ref_frequency)
        sky = materialize_healpix_model(
            sky,
            nside=nside,
            frequencies=frequencies,
            obs_frequency_config=obs_frequency_config,
            ref_frequency=ref_frequency,
            memmap_path=memmap_path,
        )

    return sky
