"""Provenance merge logic for combined sky models.

Extracted from ``combine.py`` so the provenance reduction rules can evolve
independently of the arithmetic and disjointness layers.  ``merge_provenance``
is decomposed into one small reducer per provenance field, each of which takes
the list of input models and returns the combined value(s) for its field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..containers import (
    MonopoleConvention,
    SkyCoverage,
    SkyProvenance,
    SourceSubtractionStatus,
)

if TYPE_CHECKING:
    from ..containers.footprint import SkyFootprint
    from ..containers.model import SkyModel


def _merge_monopole_convention(models: list[SkyModel]) -> MonopoleConvention:
    """Reduce ``monopole_convention`` across inputs.

    UNKNOWN inputs are ignored; among declared conventions MEAN_SUBTRACTED
    wins, then ABSOLUTE_WITH_CMB, else ABSOLUTE_NO_CMB.  An all-UNKNOWN set
    downgrades to UNKNOWN.  (``_check_monopole_consistency`` has already
    rejected numerically-incompatible mixes upstream.)
    """
    conventions = {
        m.provenance.monopole_convention
        for m in models
        if m.provenance.monopole_convention != MonopoleConvention.UNKNOWN
    }
    if not conventions:
        return MonopoleConvention.UNKNOWN
    if MonopoleConvention.MEAN_SUBTRACTED in conventions:
        return MonopoleConvention.MEAN_SUBTRACTED
    if MonopoleConvention.ABSOLUTE_WITH_CMB in conventions:
        return MonopoleConvention.ABSOLUTE_WITH_CMB
    return MonopoleConvention.ABSOLUTE_NO_CMB


def _merge_coverage(
    models: list[SkyModel],
) -> tuple[SkyCoverage, float | None, SkyFootprint | None]:
    """Reduce sky coverage to ``(coverage, fraction, footprint)``.

    Any FULL_SKY input ⇒ FULL_SKY (fraction 1.0).  Any UNKNOWN ⇒ UNKNOWN.
    Otherwise the partial footprints are unioned; a union that turns out
    full-sky is reported as FULL_SKY with no footprint.
    """
    coverages = [m.provenance.sky_coverage for m in models]
    if any(c == SkyCoverage.FULL_SKY for c in coverages):
        return SkyCoverage.FULL_SKY, 1.0, None
    if any(c == SkyCoverage.UNKNOWN for c in coverages):
        return SkyCoverage.UNKNOWN, None, None

    footprints = [m.provenance.coverage_footprint for m in models]
    if any(footprint is None for footprint in footprints):
        return SkyCoverage.UNKNOWN, None, None
    try:
        union = footprints[0].union(*footprints[1:])
    except (TypeError, ValueError):
        return SkyCoverage.UNKNOWN, None, None
    if union.is_full_sky:
        return SkyCoverage.FULL_SKY, 1.0, None
    return SkyCoverage.PARTIAL_SKY, union.coverage_fraction, union


def _merge_monopole_k(
    models: list[SkyModel],
    combined_coverage: SkyCoverage,
) -> tuple[float | None, str | None]:
    """Reduce ``monopole_k`` to ``(value, drop_reason)``.

    Naively summing monopoles is only correct when each layer contributes a
    *disjoint* DC level.  Two ABSOLUTE_WITH_CMB inputs both carry the CMB
    monopole (~2.725 K) and would double-count it; an UNKNOWN convention next
    to absolutes could already include the absolute level we're about to add.
    In either case the algebraic sum is dropped (``value=None``) with a reason
    so ``_combine_as_healpix_merge`` can fall back to the measured monopole of
    the assembled cube.
    """
    monopoles = [m.provenance.monopole_k for m in models]
    if combined_coverage != SkyCoverage.FULL_SKY or not all(
        m is not None for m in monopoles
    ):
        return None, None

    n_with_cmb = sum(
        1
        for m in models
        if m.provenance.monopole_convention == MonopoleConvention.ABSOLUTE_WITH_CMB
    )
    n_unknown = sum(
        1
        for m in models
        if m.provenance.monopole_convention == MonopoleConvention.UNKNOWN
    )
    n_absolute = sum(
        1
        for m in models
        if m.provenance.monopole_convention
        in (
            MonopoleConvention.ABSOLUTE_WITH_CMB,
            MonopoleConvention.ABSOLUTE_NO_CMB,
        )
    )
    if n_with_cmb > 1:
        return None, (
            "monopole_k dropped: multiple ABSOLUTE_WITH_CMB inputs would "
            "double-count the CMB"
        )
    if n_unknown and n_absolute:
        return None, (
            "monopole_k dropped: UNKNOWN monopole_convention alongside "
            "absolute inputs cannot be summed safely"
        )
    return float(sum(monopoles)), None


def _merge_angular_resolution(
    models: list[SkyModel],
) -> tuple[float, float] | None:
    """Reduce ``angular_resolution_rad`` to the tightest common range.

    Lower bound is the min of the per-input lower bounds; upper bound is the
    min of the upper bounds (the combined model is only as accurate at large
    scales as its loosest contributor).  ``None`` if any input is UNKNOWN.
    """
    angular_ranges = [
        m.provenance.angular_resolution_rad
        for m in models
        if m.provenance.angular_resolution_rad is not None
    ]
    if len(angular_ranges) != len(models) or not angular_ranges:
        return None
    lo = min(r[0] for r in angular_ranges)
    hi = min(r[1] for r in angular_ranges)
    return (lo, hi)


def _merge_source_subtraction(
    models: list[SkyModel],
) -> tuple[SourceSubtractionStatus, float | None, float | None, str | None]:
    """Reduce source-subtraction to ``(status, threshold, freq, method)``.

    Promoted to ALL only when every input is ALL; ABOVE_THRESHOLD / NONE only
    when homogeneous; else UNKNOWN.  Threshold + frequency + method carry
    through only when all contributors share them exactly in the homogeneous
    ABOVE_THRESHOLD case.
    """
    statuses = {m.provenance.source_subtraction for m in models}
    if statuses == {SourceSubtractionStatus.ALL}:
        status = SourceSubtractionStatus.ALL
    elif statuses == {SourceSubtractionStatus.ABOVE_THRESHOLD}:
        status = SourceSubtractionStatus.ABOVE_THRESHOLD
    elif statuses == {SourceSubtractionStatus.NONE}:
        status = SourceSubtractionStatus.NONE
    else:
        status = SourceSubtractionStatus.UNKNOWN

    thresholds = {m.provenance.source_subtraction_threshold_jy for m in models}
    freqs = {m.provenance.source_subtraction_freq_hz for m in models}
    methods = {m.provenance.source_subtraction_method for m in models}
    threshold = (
        next(iter(thresholds))
        if status == SourceSubtractionStatus.ABOVE_THRESHOLD and len(thresholds) == 1
        else None
    )
    freq = next(iter(freqs)) if threshold is not None and len(freqs) == 1 else None
    method = (
        next(iter(methods)) if threshold is not None and len(methods) == 1 else None
    )
    return status, threshold, freq, method


def _merge_notes(
    models: list[SkyModel],
    extra_note: str | None,
) -> str | None:
    """Concatenate contributor notes, appending an optional ``extra_note``."""
    note_parts = [m.provenance.notes for m in models if m.provenance.notes]
    if extra_note is not None:
        note_parts.append(extra_note)
    return " + ".join(note_parts) if note_parts else None


def merge_provenance(models: list[SkyModel]) -> SkyProvenance:
    """Merge provenance across combined input models.

    Each field is reduced by a dedicated helper (see the per-field reducers in
    this module).  ``flux_completeness_*`` is dropped to ``None`` after
    combination because composite support is no longer a single band.
    """
    combined_convention = _merge_monopole_convention(models)
    (
        combined_coverage,
        combined_coverage_fraction,
        combined_coverage_footprint,
    ) = _merge_coverage(models)
    combined_monopole_k, monopole_drop_reason = _merge_monopole_k(
        models, combined_coverage
    )
    combined_angular = _merge_angular_resolution(models)
    (
        combined_status,
        combined_threshold,
        combined_freq,
        combined_method,
    ) = _merge_source_subtraction(models)
    combined_notes = _merge_notes(models, monopole_drop_reason)

    return SkyProvenance(
        flux_completeness_jy=None,
        flux_completeness_freq_hz=None,
        angular_resolution_rad=combined_angular,
        sky_coverage=combined_coverage,
        coverage_fraction=combined_coverage_fraction,
        coverage_footprint=combined_coverage_footprint,
        monopole_convention=combined_convention,
        monopole_k=combined_monopole_k,
        source_subtraction=combined_status,
        source_subtraction_threshold_jy=combined_threshold,
        source_subtraction_freq_hz=combined_freq,
        source_subtraction_method=combined_method,
        notes=combined_notes,
    )
