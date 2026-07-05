"""Sky-model orchestration helpers for consumers.

``prepare_sky_model`` is the canonical user entry point for combining and
materializing sky models.  It wraps the internal ``_combine_models`` engine
with a beam-aware ``nside`` advisor and consistent materialization defaults.
"""

from __future__ import annotations

import logging
from typing import Any

from ..containers.model import SkyFormat, SkyModel
from ..operations.operations import (
    materialize_healpix_model,
    materialize_point_sources_model,
)
from ..support.precision import resolve_combine_precision
from .engine import (
    _combine_models,
    resolve_target_representation,
)
from .options import PrepareSkyOptions
from .regrid import (
    _resolve_requested_healpix_frequencies,
    _validate_requested_healpix_grid,
)

logger = logging.getLogger(__name__)


def prepare_sky_model(
    models: list[SkyModel],
    *,
    options: PrepareSkyOptions | None = None,
    **overrides: Any,
) -> SkyModel:
    """Combine and materialize sky models into a usable representation.

    This is the canonical user entry point.  It validates inputs, runs the
    physical-disjointness check (see :attr:`PrepareSkyOptions.assume_disjoint`
    for a narrow escape that skips only double-count rules), combines the
    models, and materializes the result into the requested ``representation``.

    Two equivalent calling styles:

    1. Build a :class:`PrepareSkyOptions` once (and serialise it for run
       reproducibility) and pass it via ``options=...``.
    2. Pass any combination of options-fields directly as keyword
       arguments. They are applied as overrides on top of the supplied
       (or default) options object.

    Cross-field rules — e.g. ``frequencies`` xor ``obs_frequency_config``
    — are validated at the :class:`PrepareSkyOptions` constructor before
    any combine work runs.

    See :class:`PrepareSkyOptions` for the full field catalogue.
    """
    if not models:
        raise ValueError("prepare_sky_model requires at least one input model.")

    base = options if options is not None else PrepareSkyOptions()
    opts = base.merged(**overrides) if overrides else base

    representation = opts.representation
    nside = opts.nside
    frequencies = opts.frequencies
    frequency = opts.frequency
    obs_frequency_config = opts.obs_frequency_config
    allow_lossy = opts.allow_lossy
    mixed_model_policy = opts.mixed_model_policy
    assume_disjoint = opts.assume_disjoint
    brightness_conversion = opts.brightness_conversion
    precision = resolve_combine_precision(opts.precision, models)
    backend = opts.backend
    memmap_path = opts.memmap_path
    beam_fwhm_rad = opts.beam_fwhm_rad
    nside_safety_factor = opts.nside_safety_factor
    subtraction_scaling_alpha = opts.subtraction_scaling_alpha

    # Single source of truth for hybrid auto-detection. ``target`` is None
    # when no explicit format is requested AND inputs span both
    # representations (the hybrid-output signal); otherwise it is the
    # concrete ``SkyFormat`` to materialize/preserve.
    target = resolve_target_representation(models, representation)

    requested_freqs = _resolve_requested_healpix_frequencies(
        frequencies,
        obs_frequency_config,
    )

    # Beam-aware nside advisor: warn (advisory only, never raise) when the
    # user-chosen nside is too coarse relative to the primary-beam FWHM.
    # Runs before the single-model fast path so passing-through one model
    # still gets the advisory.
    if (
        (target == SkyFormat.HEALPIX or target is None)
        and beam_fwhm_rad is not None
        and nside is not None
    ):
        from radiosim.utils.healpix import pixel_too_coarse, recommend_nside_for_beam

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

    # Single-model fast path: when no combine is needed and the caller didn't
    # request a specific representation, return the model unchanged.
    if len(models) == 1 and target is None:
        return models[0]

    if len(models) == 1:
        sky = models[0]
    else:
        sky = _combine_models(
            models,
            representation=target,
            _target_resolved=True,
            nside=nside,
            frequency=frequency,
            frequencies=requested_freqs,
            obs_frequency_config=None,
            brightness_conversion=brightness_conversion,
            allow_lossy_point_materialization=allow_lossy,
            mixed_model_policy=mixed_model_policy,
            assume_disjoint=assume_disjoint,
            precision=precision,
            backend=backend,
            memmap_path=memmap_path,
            subtraction_scaling_alpha=subtraction_scaling_alpha,
        )

    # When target is None (auto-detect), accept whatever combine produced —
    # _combine_models already preserves hybrids when inputs span formats.
    if target is None:
        return sky

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
            clear_other=True,
            backend=backend,
        )

    if sky.point is not None:
        return sky
    if not allow_lossy:
        raise ValueError(
            "Requested point-source sky representation for a HEALPix-only model. "
            "Set allow_lossy=True to opt in to lossy HEALPix-to-point conversion."
        )
    return materialize_point_sources_model(
        sky, frequency=frequency, lossy=True, clear_other=True, backend=backend
    )
