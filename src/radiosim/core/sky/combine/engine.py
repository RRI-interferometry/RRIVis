# radiosim/core/sky/combine.py
"""Sky-model combination engine (internal).

User code should call :func:`radiosim.core.sky.prepare_sky_model` — the
canonical entry point — which wraps this module with the beam-aware
``nside`` advisor and consistent materialization defaults. The engine
function ``_combine_models`` exported here is intentionally underscored
and intended for tests and advanced internal callers; it is **not** part
of the public surface (notice it is absent from ``__all__``).

The arithmetic, regridding, disjointness, and provenance reduction code
each live in their own module:

- ``_combine_concat`` — point-source array concatenation + extra-column helpers
- ``_combine_regrid`` — HEALPix nside regridding + grid validators
- ``_combine_healpix`` — element-wise HEALPix map combination
- ``_combine_disjointness`` — physical disjointness rules and policy enforcement
- ``_combine_provenance`` — ``SkyProvenance`` reduction across input models

Only ``_combine_models`` constructs a :class:`SkyModel` (via a late import
to avoid the circular dependency through ``_factories``).  The internal
helpers return raw data dicts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ..containers import (
    HealpixData,
    PointSourceData,
    SkyCoverage,
)
from ..containers.constants import BrightnessConversion
from ..containers.model import SkyFormat, SkyModel
from ..operations.factories import create_empty
from .concat import concat_point_sources
from .disjointness import (
    MixedModelPolicy,
    check_physical_disjointness,
    classify_model,
    resolve_brightness_conversion,
    resolve_combination_params,
)
from .healpix import CombineHealpixData, combine_healpix
from .provenance import merge_provenance
from .regrid import (
    _resolve_requested_healpix_frequencies,
    _validate_requested_healpix_grid,
    regrid_healpix_model,
)

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

logger = logging.getLogger(__name__)


__all__ = [
    "CombineHealpixData",
    "MixedModelPolicy",
    "combine_healpix",
    "concat_point_sources",
    "regrid_healpix_model",
    "resolve_target_representation",
]


# =============================================================================
# Target-representation resolution (shared by prepare_sky_model and
# _combine_models)
# =============================================================================


def resolve_target_representation(
    models: list[SkyModel],
    requested: SkyFormat | str | None,
) -> SkyFormat | None:
    """Resolve the output representation for a combine/materialize request.

    * If ``requested`` is provided, it wins (coerced to :class:`SkyFormat`).
    * Otherwise the inputs decide: returns ``SkyFormat.HEALPIX`` or
      ``SkyFormat.POINT_SOURCES`` when every populated input shares a single
      representation; returns ``None`` to signal that hybrid output should
      be preserved (any input is hybrid, or inputs span both types).

    Single source of truth for the hybrid auto-detection that previously
    lived in both :func:`prepare_sky_model` and :func:`_combine_models`.
    """
    if requested is not None:
        if isinstance(requested, SkyFormat):
            return requested
        return SkyFormat(requested)
    if not models:
        return None
    classifications = [classify_model(m) for m in models]
    hybrid_set = frozenset({SkyFormat.POINT_SOURCES, SkyFormat.HEALPIX})
    any_hybrid_input = any(c == hybrid_set for c in classifications)
    has_point = any(SkyFormat.POINT_SOURCES in c for c in classifications)
    has_healpix = any(SkyFormat.HEALPIX in c for c in classifications)
    if any_hybrid_input or (has_point and has_healpix):
        return None
    if has_healpix:
        return SkyFormat.HEALPIX
    if has_point:
        return SkyFormat.POINT_SOURCES
    return None


# =============================================================================
# Combination strategy implementations (private)
# =============================================================================


def _combine_as_healpix_merge(
    models: list[SkyModel],
    ref_frequency: float | None,
    brightness_conversion: BrightnessConversion,
    precision: PrecisionConfig | None,
    *,
    nside: int | None = None,
    frequencies: np.ndarray | None = None,
    memmap_path: str | None = None,
) -> SkyModel:
    """Combine models into a HEALPix cube via Jy-space addition.

    When at least one input carries HEALPix maps, the first such model
    fixes the output grid (``nside`` and ``frequencies`` arguments are
    rejected by ``_validate_requested_healpix_grid`` upstream).  When
    every input is point-only, the user-supplied ``nside`` (default 64)
    and ``frequencies`` define the output grid; ``frequencies`` is
    required in that case.
    """
    healpix_inputs = [m for m in models if m.healpix is not None]
    if healpix_inputs:
        ref_nside = healpix_inputs[0].healpix.nside
        ref_freqs = healpix_inputs[0].healpix.frequencies
    else:
        if frequencies is None:
            raise ValueError(
                "healpix_map output requires 'frequencies' or "
                "'obs_frequency_config' when no input model already carries "
                "HEALPix maps."
            )
        ref_nside = 64 if nside is None else nside
        ref_freqs = np.asarray(frequencies, dtype=np.float64)

    data = combine_healpix(
        models,
        ref_nside=ref_nside,
        ref_freqs=ref_freqs,
        ref_frequency=ref_frequency,
        brightness_conversion=brightness_conversion,
        precision=precision,
        memmap_path=memmap_path,
    )

    provenance = merge_provenance(models)
    # Prefer the measured monopole of the assembled HEALPix cube over the
    # merged per-layer sum.  This keeps ``monopole_k`` populated even when
    # some contributors didn't declare it (e.g. point catalogs).
    if (
        provenance.monopole_k is None
        and provenance.sky_coverage == SkyCoverage.FULL_SKY
        and data["healpix_maps"].shape[0] > 0
    ):
        measured = float(np.mean(data["healpix_maps"][0]))
        provenance = provenance.replace(monopole_k=measured)
    return SkyModel(
        healpix=HealpixData(
            maps=data["healpix_maps"],
            nside=data["healpix_nside"],
            frequencies=data["observation_frequencies"],
            coordinate_frame=data["coordinate_frame"],
            q_maps=data["healpix_q_maps"],
            u_maps=data["healpix_u_maps"],
            v_maps=data["healpix_v_maps"],
            i_brightness_conversion=brightness_conversion.value,
        ),
        reference_frequency=data["reference_frequency"],
        model_name="combined",
        brightness_conversion=brightness_conversion,
        provenance=provenance,
        precision=precision,
    )


def _combine_as_point_sources(
    models: list[SkyModel],
    frequency: float | None,
    brightness_conversion: BrightnessConversion,
    precision: PrecisionConfig | None,
    allow_lossy_point_materialization: bool,
) -> SkyModel:
    """Combine models by concatenating point-source arrays."""
    data = concat_point_sources(
        models,
        reference_frequency=frequency,
        brightness_conversion=brightness_conversion,
        precision=precision,
        allow_lossy_point_materialization=allow_lossy_point_materialization,
    )

    provenance = merge_provenance(models)
    return SkyModel(
        point=PointSourceData(
            ra_rad=data["ra_rad"],
            dec_rad=data["dec_rad"],
            flux=data["flux"],
            spectral_index=data["spectral_index"],
            stokes_q=data["stokes_q"],
            stokes_u=data["stokes_u"],
            stokes_v=data["stokes_v"],
            ref_freq=data["ref_freq"],
            rotation_measure=data["rotation_measure"],
            major_arcsec=data["major_arcsec"],
            minor_arcsec=data["minor_arcsec"],
            pa_deg=data["pa_deg"],
            spectral_coeffs=data["spectral_coeffs"],
            source_name=data["source_name"],
            source_id=data["source_id"],
            extra_columns=data["extra_columns"],
        ),
        model_name="combined",
        reference_frequency=data["reference_frequency"],
        brightness_conversion=brightness_conversion,
        provenance=provenance,
        precision=precision,
    )


def _combine_as_hybrid(
    models: list[SkyModel],
    *,
    freq: float | None,
    ref_freq: float | None,
    brightness_conversion: BrightnessConversion,
    precision: PrecisionConfig | None,
    memmap_path: str | None,
) -> SkyModel:
    """Build a hybrid SkyModel preserving both point and HEALPix payloads.

    Models with point payloads contribute to the point pile; models with
    HEALPix payloads contribute to the HEALPix pile. Each pile is reduced
    independently — no lossy point↔HEALPix conversion happens here.
    """
    point_models = [m for m in models if SkyFormat.POINT_SOURCES in classify_model(m)]
    healpix_models = [m for m in models if SkyFormat.HEALPIX in classify_model(m)]

    point_payload: PointSourceData | None = None
    if point_models:
        data = concat_point_sources(
            point_models,
            reference_frequency=freq,
            brightness_conversion=brightness_conversion,
            precision=precision,
            allow_lossy_point_materialization=False,
        )
        if data["ra_rad"].size:
            point_payload = PointSourceData(
                ra_rad=data["ra_rad"],
                dec_rad=data["dec_rad"],
                flux=data["flux"],
                spectral_index=data["spectral_index"],
                stokes_q=data["stokes_q"],
                stokes_u=data["stokes_u"],
                stokes_v=data["stokes_v"],
                ref_freq=data["ref_freq"],
                rotation_measure=data["rotation_measure"],
                major_arcsec=data["major_arcsec"],
                minor_arcsec=data["minor_arcsec"],
                pa_deg=data["pa_deg"],
                spectral_coeffs=data["spectral_coeffs"],
                source_name=data["source_name"],
                source_id=data["source_id"],
                extra_columns=data["extra_columns"],
            )
            point_ref_freq = data["reference_frequency"]
        else:
            point_ref_freq = None
    else:
        point_ref_freq = None

    healpix_payload: HealpixData | None = None
    if healpix_models:
        # Reuse _combine_as_healpix_merge to get a HEALPix-only SkyModel,
        # then borrow its healpix payload.
        merged = _combine_as_healpix_merge(
            healpix_models,
            ref_freq,
            brightness_conversion,
            precision,
            memmap_path=memmap_path,
        )
        healpix_payload = merged.healpix

    if point_payload is None and healpix_payload is None:
        return create_empty(
            model_name="combined_empty",
            brightness_conversion=brightness_conversion,
            precision=precision,
        )

    provenance = merge_provenance(models)
    return SkyModel(
        point=point_payload,
        healpix=healpix_payload,
        model_name="combined",
        reference_frequency=point_ref_freq if point_ref_freq is not None else ref_freq,
        brightness_conversion=brightness_conversion,
        provenance=provenance,
        precision=precision,
    )


# =============================================================================
# Internal building block: _combine_models
# =============================================================================
#
# This is the low-level combination engine.  User code should call
# :func:`radiosim.core.sky.prepare_sky_model` instead — that wrapper adds the
# beam-aware nside advisor and a single, documented entry-point.  Tests and
# advanced internals may import ``_combine_models`` directly from this module.


def _combine_models(
    models: list[SkyModel],
    representation: SkyFormat | str | None = None,
    nside: int | None = None,
    frequency: float | None = None,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    ref_frequency: float | None = None,
    brightness_conversion: BrightnessConversion | str | None = None,
    allow_lossy_point_materialization: bool = False,
    mixed_model_policy: MixedModelPolicy = "error",
    precision: PrecisionConfig | None = None,
    memmap_path: str | None = None,
) -> SkyModel:
    """Combine multiple sky models into one (internal building block).

    Dispatches to the appropriate combination strategy based on the input
    models and requested representation.  See module docstring for details.

    User code should prefer :func:`radiosim.core.sky.prepare_sky_model`, which
    wraps this function with a beam-aware ``nside`` advisor and consistent
    materialization defaults.

    Parameters
    ----------
    models : list of SkyModel
        Sky models to combine.
    representation : str or None, default None
        Output representation: ``"point_sources"`` or ``"healpix_map"``.
        When ``None``, auto-detects from input models.
    nside : int, optional
        HEALPix NSIDE for ``healpix_map`` output.
    frequency : float, optional
        Frequency for HEALPix-to-point-source conversions.
    frequencies : np.ndarray, optional
        Frequency array for point-to-HEALPix conversion when no input
        model already carries HEALPix maps.
    obs_frequency_config : dict, optional
        Frequency config fallback for point-to-HEALPix conversion.
    ref_frequency : float, optional
        Reference frequency for spectral extrapolation (Hz).
    brightness_conversion : str or BrightnessConversion, optional
        Output brightness conversion method. When omitted, all inputs must
        already agree; otherwise an explicit target is required.
    allow_lossy_point_materialization : bool, default False
        Allow lossy HEALPix-to-point conversion when point-source output
        is requested.
    mixed_model_policy : {"error", "warn", "allow"}, default "error"
        Policy for combining point catalogs with diffuse HEALPix models.
    precision : PrecisionConfig, optional
        Precision configuration for the combined model.
    memmap_path : str or None, optional
        If given, stream the combined HEALPix cube to memory-mapped files
        at this directory (created if needed) rather than allocating it
        in RAM.  Only affects HEALPix output paths.

    Returns
    -------
    SkyModel
        Combined sky model.
    """
    if not models:
        return create_empty(
            model_name="combined_empty",
            brightness_conversion=(
                BrightnessConversion.PLANCK
                if brightness_conversion is None
                else BrightnessConversion(brightness_conversion)
            ),
            precision=precision,
        )

    brightness_conversion = resolve_brightness_conversion(models, brightness_conversion)
    check_physical_disjointness(models, mixed_model_policy)
    requested_freqs = _resolve_requested_healpix_frequencies(
        frequencies, obs_frequency_config
    )

    target = resolve_target_representation(models, representation)
    if target is None:
        # Hybrid output — at least one input is hybrid or inputs span types.
        return _combine_as_hybrid(
            models,
            freq=frequency,
            ref_freq=ref_frequency,
            brightness_conversion=brightness_conversion,
            precision=precision,
            memmap_path=memmap_path,
        )

    representation, freq, ref_freq = resolve_combination_params(
        models, target, frequency, ref_frequency
    )

    has_healpix_map = any(m.healpix is not None for m in models)

    if representation == SkyFormat.HEALPIX and has_healpix_map:
        _validate_requested_healpix_grid(models, nside, requested_freqs)

    # All HEALPix-output paths route through _combine_as_healpix_merge,
    # whether or not any input already carries maps.  The merge path keeps
    # each contributor distinct (so per-model PointSpectrum tables propagate
    # losslessly), unlike the older concat-then-materialize fallback.
    if representation == SkyFormat.HEALPIX:
        return _combine_as_healpix_merge(
            models,
            ref_freq,
            brightness_conversion,
            precision,
            nside=nside,
            frequencies=requested_freqs,
            memmap_path=memmap_path,
        )

    # Point-source output: concatenate as point sources.
    return _combine_as_point_sources(
        models,
        freq,
        brightness_conversion,
        precision,
        allow_lossy_point_materialization,
    )
