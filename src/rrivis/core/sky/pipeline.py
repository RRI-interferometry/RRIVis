"""Sky-model orchestration helpers for consumers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .combine import (
    MixedModelPolicy,
    _resolve_requested_healpix_frequencies,
    _validate_requested_healpix_grid,
    combine_models,
)
from .model import SkyFormat, SkyModel, _coerce_format
from .operations import materialize_healpix_model, materialize_point_sources_model

logger = logging.getLogger(__name__)


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
    beam_fwhm_rad: float | None = None,
    nside_safety_factor: float = 5.0,
) -> SkyModel:
    """Combine and materialize sky models for an explicit representation.

    Parameters
    ----------
    beam_fwhm_rad
        Optional primary-beam FWHM (radians).  When provided together with a
        ``HEALPIX`` representation request, an advisory warning is logged
        if the chosen ``nside`` gives a pixel size larger than
        ``beam_fwhm_rad / nside_safety_factor`` (the "five pixels across
        the beam" rule of thumb).  Purely advisory — never raises.
    nside_safety_factor
        Target ratio of ``beam_fwhm`` to pixel scale for the advisor.
        Defaults to 5.
    """
    target = _coerce_format(representation)
    if not models:
        raise ValueError("prepare_sky_model requires at least one input model.")

    requested_freqs = _resolve_requested_healpix_frequencies(
        frequencies,
        obs_frequency_config,
    )

    # Beam-aware nside advisor: warn (advisory only, never raise) when the
    # user-chosen nside is too coarse relative to the primary-beam FWHM.
    if target == SkyFormat.HEALPIX and beam_fwhm_rad is not None and nside is not None:
        from rrivis.utils.healpix import pixel_too_coarse, recommend_nside_for_beam

        if pixel_too_coarse(
            int(nside), float(beam_fwhm_rad), safety_factor=nside_safety_factor
        ):
            suggested = recommend_nside_for_beam(
                float(beam_fwhm_rad), safety_factor=nside_safety_factor
            )
            logger.warning(
                "prepare_sky_model: chosen nside=%d gives a pixel scale that "
                "exceeds beam_fwhm/%g (beam FWHM=%.3g rad). Consider "
                "nside=%d for at least %g pixels across the beam.",
                int(nside),
                nside_safety_factor,
                float(beam_fwhm_rad),
                suggested,
                nside_safety_factor,
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
