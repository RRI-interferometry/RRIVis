"""Sky-model orchestration helpers for consumers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .combine import (
    MixedModelPolicy,
    _resolve_requested_healpix_frequencies,
    _validate_requested_healpix_grid,
    combine_models,
)
from .model import SkyFormat, SkyModel
from .operations import materialize_healpix_model, materialize_point_sources_model


def prepare_sky_model(
    models: list[SkyModel],
    *,
    representation: SkyFormat | str,
    nside: int | None = None,
    frequencies: np.ndarray | None = None,
    frequency: float | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    allow_lossy: bool = False,
    mixed_model_policy: MixedModelPolicy = "error",
    brightness_conversion: Any = None,
    precision: Any = None,
    memmap_path: str | None = None,
) -> SkyModel:
    """Combine and materialize sky models for an explicit representation."""
    target = SkyModel._coerce_representation(representation)
    if not models:
        raise ValueError("prepare_sky_model requires at least one input model.")

    requested_freqs = _resolve_requested_healpix_frequencies(
        frequencies,
        obs_frequency_config,
    )

    if len(models) == 1:
        sky = models[0]
    else:
        sky = combine_models(
            models,
            representation=target,
            nside=nside,
            frequency=frequency,
            frequencies=requested_freqs,
            obs_frequency_config=None,
            brightness_conversion=brightness_conversion,
            allow_lossy_point_materialization=allow_lossy,
            mixed_model_policy=mixed_model_policy,
            precision=precision,
            memmap_path=memmap_path,
        )

    if target == SkyFormat.HEALPIX:
        if sky.healpix is not None:
            _validate_requested_healpix_grid([sky], nside, requested_freqs)
            return sky
        return materialize_healpix_model(
            sky,
            nside=64 if nside is None else nside,
            frequencies=requested_freqs,
            ref_frequency=frequency,
            memmap_path=memmap_path,
        )

    if sky.point is not None:
        return sky
    if not allow_lossy:
        raise ValueError(
            "Requested point-source sky representation for a HEALPix-only model. "
            "Set allow_lossy=True to opt in to lossy HEALPix-to-point conversion."
        )
    return materialize_point_sources_model(sky, frequency=frequency, lossy=True)
