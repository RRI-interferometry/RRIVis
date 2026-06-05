# radiosim/core/sky/_combine_provenance.py
"""Provenance merge logic for combined sky models.

Extracted from ``combine.py`` so the provenance reduction rules can evolve
independently of the arithmetic and disjointness layers.
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
    from ..containers.model import SkyModel


def merge_provenance(models: list[SkyModel]) -> SkyProvenance:
    """Merge provenance across combined input models.

    Rules:
    - ``monopole_convention``: must match across all declared inputs
      (``_check_monopole_consistency`` has already enforced this); UNKNOWN is
      tolerated but downgrades the combined convention to UNKNOWN.
    - ``monopole_k``: summed when every input declares a value; otherwise None.
    - ``angular_resolution_rad``: tightest lower bound (min of min) across
      inputs; upper bound is the tightest upper bound (min of max) since the
      combined model is only as accurate at large scales as its loosest
      contributor.  None if any input is UNKNOWN.
    - ``flux_completeness_jy``: None after combination (composite support is
      no longer a single band).
    - ``source_subtraction``: promoted to ALL only when every input is ALL;
      NONE only when every input is NONE; else UNKNOWN.  Threshold + frequency
      + method carry through only in the homogeneous ALL / ABOVE_THRESHOLD case.
    - ``notes``: concatenated from contributors.
    """
    conventions = {
        m.provenance.monopole_convention
        for m in models
        if m.provenance.monopole_convention != MonopoleConvention.UNKNOWN
    }
    if not conventions:
        combined_convention = MonopoleConvention.UNKNOWN
    elif MonopoleConvention.MEAN_SUBTRACTED in conventions:
        combined_convention = MonopoleConvention.MEAN_SUBTRACTED
    elif MonopoleConvention.ABSOLUTE_WITH_CMB in conventions:
        combined_convention = MonopoleConvention.ABSOLUTE_WITH_CMB
    else:
        combined_convention = MonopoleConvention.ABSOLUTE_NO_CMB

    combined_coverage_footprint = None
    coverages = [m.provenance.sky_coverage for m in models]
    if any(c == SkyCoverage.FULL_SKY for c in coverages):
        combined_coverage = SkyCoverage.FULL_SKY
        combined_coverage_fraction = 1.0
    elif any(c == SkyCoverage.UNKNOWN for c in coverages):
        combined_coverage = SkyCoverage.UNKNOWN
        combined_coverage_fraction = None
    else:
        footprints = [m.provenance.coverage_footprint for m in models]
        if any(footprint is None for footprint in footprints):
            combined_coverage = SkyCoverage.UNKNOWN
            combined_coverage_fraction = None
        else:
            try:
                combined_coverage_footprint = footprints[0].union(*footprints[1:])
            except (TypeError, ValueError):
                combined_coverage = SkyCoverage.UNKNOWN
                combined_coverage_fraction = None
            else:
                if combined_coverage_footprint.is_full_sky:
                    combined_coverage = SkyCoverage.FULL_SKY
                    combined_coverage_fraction = 1.0
                    combined_coverage_footprint = None
                else:
                    combined_coverage = SkyCoverage.PARTIAL_SKY
                    combined_coverage_fraction = (
                        combined_coverage_footprint.coverage_fraction
                    )

    monopoles = [m.provenance.monopole_k for m in models]
    combined_monopole_k: float | None = None
    monopole_drop_reason: str | None = None
    if combined_coverage == SkyCoverage.FULL_SKY and all(
        m is not None for m in monopoles
    ):
        # Naively summing monopoles is only correct when each layer
        # contributes a *disjoint* DC level.  Two ABSOLUTE_WITH_CMB
        # inputs both carry the CMB monopole (~2.725 K) and would
        # double-count it; an UNKNOWN convention next to absolutes
        # could already include the absolute level we're about to add.
        # In either case, drop the algebraic sum and let
        # _combine_as_healpix_merge fall back to the measured monopole
        # of the assembled cube.
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
            monopole_drop_reason = (
                "monopole_k dropped: multiple ABSOLUTE_WITH_CMB inputs would "
                "double-count the CMB"
            )
        elif n_unknown and n_absolute:
            monopole_drop_reason = (
                "monopole_k dropped: UNKNOWN monopole_convention alongside "
                "absolute inputs cannot be summed safely"
            )
        else:
            combined_monopole_k = float(sum(monopoles))

    angular_ranges = [
        m.provenance.angular_resolution_rad
        for m in models
        if m.provenance.angular_resolution_rad is not None
    ]
    if len(angular_ranges) == len(models) and angular_ranges:
        lo = min(r[0] for r in angular_ranges)
        hi = min(r[1] for r in angular_ranges)
        combined_angular: tuple[float, float] | None = (lo, hi)
    else:
        combined_angular = None

    statuses = {m.provenance.source_subtraction for m in models}
    if statuses == {SourceSubtractionStatus.ALL}:
        combined_status = SourceSubtractionStatus.ALL
    elif statuses == {SourceSubtractionStatus.ABOVE_THRESHOLD}:
        combined_status = SourceSubtractionStatus.ABOVE_THRESHOLD
    elif statuses == {SourceSubtractionStatus.NONE}:
        combined_status = SourceSubtractionStatus.NONE
    else:
        combined_status = SourceSubtractionStatus.UNKNOWN

    # Threshold carries through only when all contributors share it exactly.
    thresholds = {m.provenance.source_subtraction_threshold_jy for m in models}
    freqs = {m.provenance.source_subtraction_freq_hz for m in models}
    methods = {m.provenance.source_subtraction_method for m in models}
    combined_threshold = (
        next(iter(thresholds))
        if combined_status in (SourceSubtractionStatus.ABOVE_THRESHOLD,)
        and len(thresholds) == 1
        else None
    )
    combined_freq = (
        next(iter(freqs))
        if combined_threshold is not None and len(freqs) == 1
        else None
    )
    combined_method = (
        next(iter(methods))
        if combined_threshold is not None and len(methods) == 1
        else None
    )

    note_parts = [m.provenance.notes for m in models if m.provenance.notes]
    if monopole_drop_reason is not None:
        note_parts.append(monopole_drop_reason)
    combined_notes = " + ".join(note_parts) if note_parts else None

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
