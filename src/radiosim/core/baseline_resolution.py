"""Pure canonical baseline generation and exact scientific selection."""

from __future__ import annotations

import json
import math
from typing import cast

from radiosim.core.instrument import (
    BaselineSelectionCriteriaSnapshot,
    BaselineSelectionProvenance,
    ResolvedBaseline,
    ResolvedBaselineSelection,
    ResolvedInstrument,
)
from radiosim.core.instrument_resolution import InstrumentResolutionError
from radiosim.io.instrument_config import (
    BaselineSelectionConfig,
    LengthRangesConfig,
    LengthTargetsConfig,
)

_COINCIDENT_ANTENNA_THRESHOLD_M = 1e-9
_LENGTH_BOUNDARY_ALLOWANCE_M = 1e-9
_AZIMUTH_BOUNDARY_ALLOWANCE_DEG = 1e-12
_BASELINE_SELECTION_SCHEMA_VERSION = "radiosim.baseline-selection.v1"


class BaselineGenerationError(InstrumentResolutionError):
    """Canonical baseline generation encountered an invalid numeric state."""


class CoincidentAntennaError(BaselineGenerationError):
    """Distinct antennas are coincident within the canonical threshold."""


class BaselineSelectionError(InstrumentResolutionError):
    """Canonical baseline selection encountered an invalid runtime state."""


class EmptyBaselineSelectionError(BaselineSelectionError):
    """Valid normalized criteria selected no canonical baselines."""


def _require_resolved_instrument(value: object) -> ResolvedInstrument:
    if type(value) is not ResolvedInstrument:
        raise TypeError("instrument must be a ResolvedInstrument")
    return value


def _generate_cross_baseline(
    instrument: ResolvedInstrument,
    first_index: int,
    second_index: int,
) -> ResolvedBaseline:
    ant1 = instrument.antennas[first_index]
    ant2 = instrument.antennas[second_index]
    pair = (ant1.id.number, ant2.id.number)
    try:
        vector = tuple(
            float(second - first)
            for first, second in zip(
                ant1.position_enu_m,
                ant2.position_enu_m,
                strict=True,
            )
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise BaselineGenerationError(
            f"baseline pair {pair}: ENU subtraction failed"
        ) from exc

    if len(vector) != 3 or any(not math.isfinite(value) for value in vector):
        raise BaselineGenerationError(
            f"baseline pair {pair}: ENU subtraction produced a nonfinite component"
        )
    try:
        length = math.hypot(*vector)
    except (OverflowError, TypeError, ValueError) as exc:
        raise BaselineGenerationError(
            f"baseline pair {pair}: Euclidean norm calculation failed"
        ) from exc
    if not math.isfinite(length):
        raise BaselineGenerationError(
            f"baseline pair {pair}: Euclidean norm is nonfinite"
        )
    if length <= _COINCIDENT_ANTENNA_THRESHOLD_M:
        raise CoincidentAntennaError(
            "distinct antennas "
            f"{ant1.id.number}/{ant1.id.name!r} and "
            f"{ant2.id.number}/{ant2.id.name!r} are separated by {length!r} m, "
            f"at or below {_COINCIDENT_ANTENNA_THRESHOLD_M!r} m"
        )

    azimuth = math.degrees(math.atan2(vector[0], vector[1])) % 180.0
    if azimuth == 0.0:
        azimuth = 0.0
    try:
        return ResolvedBaseline(
            ant1=ant1.id,
            ant2=ant2.id,
            vector_enu_m=(vector[0], vector[1], vector[2]),
            length_m=length,
            is_autocorrelation=False,
            azimuth_deg=azimuth,
        )
    except (TypeError, ValueError) as exc:
        raise BaselineGenerationError(
            f"baseline pair {pair}: generated state is inconsistent"
        ) from exc


def generate_resolved_baselines(
    instrument: ResolvedInstrument,
) -> tuple[ResolvedBaseline, ...]:
    """Generate every canonical auto and cross baseline for an instrument.

    Parameters
    ----------
    instrument
        Complete canonical instrument whose antennas are already sorted by number.

    Returns
    -------
    tuple of ResolvedBaseline
        All pairs in lexicographic numeric order, including autocorrelations.

    Raises
    ------
    BaselineGenerationError
        If subtraction, norm, or constructed baseline state is invalid.
    CoincidentAntennaError
        If distinct antennas are separated by at most ``1e-9`` metres.
    """
    canonical_instrument = _require_resolved_instrument(instrument)
    generated: list[ResolvedBaseline] = []
    for first_index, ant1 in enumerate(canonical_instrument.antennas):
        for second_index in range(first_index, len(canonical_instrument.antennas)):
            if first_index == second_index:
                generated.append(
                    ResolvedBaseline(
                        ant1=ant1.id,
                        ant2=ant1.id,
                        vector_enu_m=(0.0, 0.0, 0.0),
                        length_m=0.0,
                        is_autocorrelation=True,
                        azimuth_deg=None,
                    )
                )
            else:
                generated.append(
                    _generate_cross_baseline(
                        canonical_instrument,
                        first_index,
                        second_index,
                    )
                )

    expected_count = (
        len(canonical_instrument.antennas)
        * (len(canonical_instrument.antennas) + 1)
        // 2
    )
    if len(generated) != expected_count:
        raise BaselineGenerationError(
            "canonical baseline generation produced an inconsistent pair count"
        )
    return tuple(baseline for baseline in generated)


def _criteria_from_config(
    config: BaselineSelectionConfig,
) -> BaselineSelectionCriteriaSnapshot:
    length_filter = config.length_filter
    if length_filter is None:
        length_mode = None
        length_targets: tuple[float, ...] = ()
        length_tolerance = None
        length_ranges: tuple[tuple[float, float], ...] = ()
    elif type(length_filter) is LengthTargetsConfig:
        length_mode = "targets"
        length_targets = tuple(float(value) for value in length_filter.targets_m)
        length_tolerance = float(length_filter.tolerance_m)
        length_ranges = ()
    elif type(length_filter) is LengthRangesConfig:
        length_mode = "ranges"
        length_targets = ()
        length_tolerance = None
        length_ranges = tuple(
            (float(item.min_m), float(item.max_m)) for item in length_filter.ranges_m
        )
    else:
        raise BaselineSelectionError(
            "config.length_filter is not an exact canonical length filter"
        )

    try:
        return BaselineSelectionCriteriaSnapshot(
            correlations=config.correlations,
            length_mode=length_mode,
            length_targets_m=length_targets,
            length_tolerance_m=length_tolerance,
            length_ranges_m=length_ranges,
            azimuth_ranges_deg=tuple(
                (float(item.start_deg), float(item.end_deg))
                for item in config.azimuth_ranges_deg
            ),
        )
    except (TypeError, ValueError) as exc:
        raise BaselineSelectionError(
            "config could not produce canonical baseline-selection criteria"
        ) from exc


def _validate_generated_inventory(
    baselines: object,
    instrument: ResolvedInstrument,
) -> tuple[ResolvedBaseline, ...]:
    if type(baselines) is not tuple:
        raise TypeError("baselines must be a tuple of ResolvedBaseline values")
    untyped_baselines = cast(tuple[object, ...], baselines)
    if any(type(item) is not ResolvedBaseline for item in untyped_baselines):
        raise TypeError("baselines must contain only ResolvedBaseline values")
    canonical_baselines = tuple(
        cast(ResolvedBaseline, item) for item in untyped_baselines
    )
    expected = generate_resolved_baselines(instrument)
    if canonical_baselines != expected:
        raise BaselineSelectionError(
            "baselines do not match the complete canonical instrument inventory"
        )
    return tuple(baseline for baseline in canonical_baselines)


def _matches_length(
    baseline: ResolvedBaseline,
    criteria: BaselineSelectionCriteriaSnapshot,
) -> bool:
    if criteria.length_mode is None:
        return True
    if criteria.length_mode == "targets":
        tolerance = criteria.length_tolerance_m
        if tolerance is None:
            raise BaselineSelectionError(
                "normalized target criteria are missing a tolerance"
            )
        return any(
            abs(baseline.length_m - target) <= tolerance + _LENGTH_BOUNDARY_ALLOWANCE_M
            for target in criteria.length_targets_m
        )
    if criteria.length_mode == "ranges":
        return any(
            minimum - _LENGTH_BOUNDARY_ALLOWANCE_M
            <= baseline.length_m
            <= maximum + _LENGTH_BOUNDARY_ALLOWANCE_M
            for minimum, maximum in criteria.length_ranges_m
        )
    raise BaselineSelectionError("normalized criteria contain an unknown length mode")


def _matches_azimuth(
    angle: float,
    ranges: tuple[tuple[float, float], ...],
) -> bool:
    for start, end in ranges:
        if start < end:
            in_closed_range = start <= angle <= end
        else:
            in_closed_range = angle >= start or angle <= end
        start_distance = abs(angle - start)
        end_distance = abs(angle - end)
        within_boundary_allowance = (
            min(start_distance, 180.0 - start_distance)
            <= _AZIMUTH_BOUNDARY_ALLOWANCE_DEG
            or min(end_distance, 180.0 - end_distance)
            <= _AZIMUTH_BOUNDARY_ALLOWANCE_DEG
        )
        if in_closed_range or within_boundary_allowance:
            return True
    return False


def select_resolved_baselines(
    baselines: tuple[ResolvedBaseline, ...],
    *,
    instrument: ResolvedInstrument,
    config: BaselineSelectionConfig,
) -> ResolvedBaselineSelection:
    """Apply exact stable baseline filters and freeze selection provenance.

    Parameters
    ----------
    baselines
        Complete tuple returned by :func:`generate_resolved_baselines`.
    instrument
        Canonical instrument that owns the baseline identities and fingerprint.
    config
        Exact frozen Tier 2 baseline-selection input.

    Returns
    -------
    ResolvedBaselineSelection
        Nonempty stable selection and its normalized provenance.

    Raises
    ------
    BaselineSelectionError
        If runtime inputs contradict the canonical generation contract.
    EmptyBaselineSelectionError
        If valid criteria leave no selected baselines.
    """
    canonical_instrument = _require_resolved_instrument(instrument)
    if type(config) is not BaselineSelectionConfig:
        raise TypeError("config must be a BaselineSelectionConfig")
    generated = _validate_generated_inventory(baselines, canonical_instrument)
    criteria = _criteria_from_config(config)

    after_correlation = tuple(
        baseline
        for baseline in generated
        if criteria.correlations == "all"
        or (criteria.correlations == "auto" and baseline.is_autocorrelation)
        or (criteria.correlations == "cross" and not baseline.is_autocorrelation)
    )
    after_length = tuple(
        baseline
        for baseline in after_correlation
        if _matches_length(baseline, criteria)
    )

    if criteria.azimuth_ranges_deg:
        exempt_auto_count = sum(
            baseline.is_autocorrelation for baseline in after_length
        )
        after_azimuth = tuple(
            baseline
            for baseline in after_length
            if baseline.is_autocorrelation
            or (
                baseline.azimuth_deg is not None
                and _matches_azimuth(
                    baseline.azimuth_deg,
                    criteria.azimuth_ranges_deg,
                )
            )
        )
    else:
        exempt_auto_count = 0
        after_azimuth = tuple(baseline for baseline in after_length)

    if not after_azimuth:
        normalized = json.dumps(
            criteria.to_snapshot(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        raise EmptyBaselineSelectionError(
            f"baseline selection is empty for normalized criteria {normalized}"
        )

    selected_ids = tuple(
        (baseline.ant1.number, baseline.ant2.number) for baseline in after_azimuth
    )
    provenance = BaselineSelectionProvenance(
        schema_version=_BASELINE_SELECTION_SCHEMA_VERSION,
        instrument_sha256=canonical_instrument.provenance.instrument_sha256,
        criteria=criteria,
        generated_count=len(generated),
        after_correlation_count=len(after_correlation),
        after_length_count=len(after_length),
        after_azimuth_count=len(after_azimuth),
        azimuth_exempt_auto_count=exempt_auto_count,
        selected_ids=selected_ids,
    )
    return ResolvedBaselineSelection(
        baselines=after_azimuth,
        provenance=provenance,
    )


__all__ = [
    "BaselineGenerationError",
    "CoincidentAntennaError",
    "BaselineSelectionError",
    "EmptyBaselineSelectionError",
    "generate_resolved_baselines",
    "select_resolved_baselines",
]
