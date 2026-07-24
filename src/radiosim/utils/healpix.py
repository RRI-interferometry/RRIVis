"""Conservative HEALPix sampling advice for canonical beam products.

The pure recommendation helper converts an angular pixel limit into the
smallest retained power-of-two NSIDE. The Tier 3 derivation uses immutable
loaded handler feature scales for every selected baseline and exact observation
frequency. It does not evaluate beams, reload files, or mutate sky state.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, cast

import numpy as np

from radiosim.core.beam.errors import BeamSamplingDerivationError
from radiosim.core.beam.models import LoadedBeamHandlerState, LoadedBeamState
from radiosim.core.instrument import AntennaId, ResolvedBaseline

_MAX_NSIDE = 65536
_MAX_VALID_NSIDE = 1 << 29
_SAFETY_FACTOR = 5
_SamplingMetric = Literal[
    "analytic_aperture_support",
    "native_grid_representation_bound",
]
_HandlerKind = Literal["analytic", "fits"]

__all__ = [
    "BeamSamplingRequirement",
    "derive_beam_sampling_requirement",
    "recommend_nside_for_angular_scale",
]


def _valid_nside(value: object) -> bool:
    return (
        type(value) is int
        and value > 0
        and value & (value - 1) == 0
        and value <= _MAX_VALID_NSIDE
    )


def _pixel_scale_rad(nside: int) -> float:
    """Return the exact square-root HEALPix pixel-area scale in radians."""
    return math.sqrt(math.pi / (3.0 * nside * nside))


def _positive_finite_float(value: object, field_name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field_name} must be an exact positive finite float")
    return value


def recommend_nside_for_angular_scale(target_angular_scale_rad: object) -> int:
    """Return the smallest retained power-of-two NSIDE satisfying a pixel limit.

    Parameters
    ----------
    target_angular_scale_rad
        Positive finite maximum HEALPix pixel resolution in radians.

    Returns
    -------
    int
        The smallest power-of-two NSIDE whose
        :func:`healpy.nside2resol` value is no larger than the target.

    Raises
    ------
    ValueError
        If the target is malformed, non-finite, non-positive, or would require
        an NSIDE larger than 65536.
    """
    if isinstance(target_angular_scale_rad, (bool, np.bool_)) or not isinstance(
        target_angular_scale_rad,
        Real,
    ):
        raise ValueError(
            "target_angular_scale_rad must be a positive finite real number, "
            f"got {target_angular_scale_rad!r}."
        )
    target = float(target_angular_scale_rad)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError(
            "target_angular_scale_rad must be a positive finite real number, "
            f"got {target_angular_scale_rad!r}."
        )

    nside = 1
    while nside < _MAX_NSIDE and _pixel_scale_rad(nside) > target:
        nside <<= 1
    if _pixel_scale_rad(nside) > target:
        raise ValueError(
            "target_angular_scale_rad requires an NSIDE larger than the retained "
            f"maximum {_MAX_NSIDE}."
        )
    return int(nside)


@dataclass(frozen=True, slots=True)
class BeamSamplingRequirement:
    """Detached immutable provenance for one HEALPix sampling decision.

    ``product_feature_scale_rad`` is the smallest selected baseline-product
    voltage feature scale. ``pixel_limit_rad`` applies the exact Tier 3
    engineering safety factor of five. A FITS-involving limiting product uses
    ``native_grid_representation_bound``; this describes the loaded grid
    representation and is not a physical beam bandwidth or measured FWHM.
    """

    actual_nside: int
    recommended_nside: int
    actual_pixel_scale_rad: float
    product_feature_scale_rad: float
    pixel_limit_rad: float
    baseline_ant1: AntennaId
    baseline_ant2: AntennaId
    frequency_hz: float
    handler_id_p: str
    handler_id_q: str
    handler_kind_p: _HandlerKind
    handler_kind_q: _HandlerKind
    metric_kind: _SamplingMetric
    safety_factor: Literal[5]

    def __post_init__(self) -> None:
        if not _valid_nside(self.actual_nside):
            raise ValueError("actual_nside must be a strict valid HEALPix NSIDE")
        if not _valid_nside(self.recommended_nside):
            raise ValueError("recommended_nside must be a strict valid HEALPix NSIDE")
        if self.recommended_nside > _MAX_NSIDE:
            raise ValueError(f"recommended_nside must not exceed {_MAX_NSIDE}")

        actual_pixel = _positive_finite_float(
            self.actual_pixel_scale_rad,
            "actual_pixel_scale_rad",
        )
        feature = _positive_finite_float(
            self.product_feature_scale_rad,
            "product_feature_scale_rad",
        )
        limit = _positive_finite_float(self.pixel_limit_rad, "pixel_limit_rad")
        frequency = _positive_finite_float(self.frequency_hz, "frequency_hz")
        expected_pixel = _pixel_scale_rad(self.actual_nside)
        if actual_pixel != expected_pixel:
            raise ValueError(
                "actual_pixel_scale_rad must equal healpy.nside2resol(actual_nside)"
            )
        if limit != feature / float(_SAFETY_FACTOR):
            raise ValueError("pixel_limit_rad must equal product_feature_scale_rad / 5")
        if self.recommended_nside != recommend_nside_for_angular_scale(limit):
            raise ValueError(
                "recommended_nside must be the smallest retained NSIDE satisfying "
                "pixel_limit_rad"
            )

        if type(self.baseline_ant1) is not AntennaId:
            raise TypeError("baseline_ant1 must be an exact AntennaId")
        if type(self.baseline_ant2) is not AntennaId:
            raise TypeError("baseline_ant2 must be an exact AntennaId")
        ant1 = AntennaId(self.baseline_ant1.number, self.baseline_ant1.name)
        ant2 = AntennaId(self.baseline_ant2.number, self.baseline_ant2.name)
        if ant1.number > ant2.number:
            raise ValueError("baseline endpoint ordering must be canonical")

        for field_name in ("handler_id_p", "handler_id_q"):
            handler_id = getattr(self, field_name)
            if (
                type(handler_id) is not str
                or re.fullmatch(r"beam-[0-9]{4}-[0-9a-f]{12}", handler_id) is None
            ):
                raise ValueError(
                    f"{field_name} must be a canonical loaded beam handler ID"
                )
        for field_name in ("handler_kind_p", "handler_kind_q"):
            handler_kind = getattr(self, field_name)
            if type(handler_kind) is not str or handler_kind not in {
                "analytic",
                "fits",
            }:
                raise ValueError(f"{field_name} must be 'analytic' or 'fits'")
        if type(self.metric_kind) is not str or self.metric_kind not in {
            "analytic_aperture_support",
            "native_grid_representation_bound",
        }:
            raise ValueError(
                "metric_kind must be 'analytic_aperture_support' or "
                "'native_grid_representation_bound'"
            )
        expected_metric = (
            "native_grid_representation_bound"
            if "fits" in {self.handler_kind_p, self.handler_kind_q}
            else "analytic_aperture_support"
        )
        if self.metric_kind != expected_metric:
            raise ValueError("metric_kind must match the exact endpoint handler kinds")
        if type(self.safety_factor) is not int or self.safety_factor != 5:
            raise ValueError("safety_factor must be the exact integer 5")

        object.__setattr__(self, "baseline_ant1", ant1)
        object.__setattr__(self, "baseline_ant2", ant2)
        object.__setattr__(self, "frequency_hz", frequency)

    def to_snapshot(self) -> dict[str, Any]:
        """Return a detached JSON-safe scalar and identity snapshot."""
        return {
            "actual_nside": self.actual_nside,
            "recommended_nside": self.recommended_nside,
            "actual_pixel_scale_rad": self.actual_pixel_scale_rad,
            "product_feature_scale_rad": self.product_feature_scale_rad,
            "pixel_limit_rad": self.pixel_limit_rad,
            "baseline_ant1": {
                "number": self.baseline_ant1.number,
                "name": self.baseline_ant1.name,
            },
            "baseline_ant2": {
                "number": self.baseline_ant2.number,
                "name": self.baseline_ant2.name,
            },
            "frequency_hz": self.frequency_hz,
            "handler_id_p": self.handler_id_p,
            "handler_id_q": self.handler_id_q,
            "handler_kind_p": self.handler_kind_p,
            "handler_kind_q": self.handler_kind_q,
            "metric_kind": self.metric_kind,
            "safety_factor": self.safety_factor,
        }


def _validated_frequencies(value: object) -> tuple[float, ...]:
    if type(value) is not tuple or not value:
        raise BeamSamplingDerivationError(
            "observation frequencies must be a nonempty exact tuple."
        )
    frequencies: list[float] = []
    previous: float | None = None
    frequency_values = cast(tuple[object, ...], value)
    for index, frequency_value in enumerate(frequency_values):
        if (
            type(frequency_value) is not float
            or not math.isfinite(frequency_value)
            or frequency_value <= 0.0
        ):
            raise BeamSamplingDerivationError(
                "observation frequency "
                f"at index {index} must be an exact positive finite float."
            )
        frequency = frequency_value
        if previous is not None and frequency <= previous:
            raise BeamSamplingDerivationError(
                "observation frequencies must be strictly increasing."
            )
        frequencies.append(frequency)
        previous = frequency
    return tuple(frequencies)


def _handler_scale_lookup(
    handler: LoadedBeamHandlerState,
) -> dict[float, float]:
    values = handler.voltage_feature_scale_by_frequency
    if type(values) is not tuple or not values:
        raise BeamSamplingDerivationError(
            f"handler {handler.handler_id!r} has no voltage feature scales."
        )
    scales: dict[float, float] = {}
    for index, pair in enumerate(values):
        if type(pair) is not tuple or len(pair) != 2:
            raise BeamSamplingDerivationError(
                f"handler {handler.handler_id!r} feature scale item {index} "
                "is not an exact frequency/scale pair."
            )
        frequency_hz, scale_rad = pair
        if (
            type(frequency_hz) is not float
            or not math.isfinite(frequency_hz)
            or frequency_hz <= 0.0
        ):
            raise BeamSamplingDerivationError(
                f"handler {handler.handler_id!r} has an invalid feature-scale "
                f"frequency {frequency_hz!r}."
            )
        if (
            type(scale_rad) is not float
            or not math.isfinite(scale_rad)
            or scale_rad <= 0.0
        ):
            raise BeamSamplingDerivationError(
                f"handler {handler.handler_id!r} has invalid feature scale "
                f"{scale_rad!r} at frequency {frequency_hz!r}."
            )
        if frequency_hz in scales:
            raise BeamSamplingDerivationError(
                f"handler {handler.handler_id!r} has ambiguous duplicate feature "
                f"scale frequency {frequency_hz!r}."
            )
        scales[frequency_hz] = scale_rad
    return scales


def derive_beam_sampling_requirement(
    *,
    selected_baselines: object,
    beam_state: object,
    observation_frequencies_hz: object,
    actual_nside: object,
) -> BeamSamplingRequirement:
    """Derive canonical advice from loaded handlers and selected baselines.

    Ties retain stable selected-baseline order followed by exact observation
    frequency order. Any invalid, incomplete, ambiguous, non-finite, or
    unmatched canonical state raises :class:`BeamSamplingDerivationError`.
    """
    if type(selected_baselines) is not tuple or not selected_baselines:
        raise BeamSamplingDerivationError(
            "selected baseline domain must be a nonempty exact tuple."
        )
    baseline_values = cast(tuple[object, ...], selected_baselines)
    if any(type(baseline) is not ResolvedBaseline for baseline in baseline_values):
        raise BeamSamplingDerivationError(
            "selected baseline domain must contain exact ResolvedBaseline values."
        )
    baselines: list[ResolvedBaseline] = []
    seen_baselines: set[ResolvedBaseline] = set()
    for index, baseline_value in enumerate(baseline_values):
        baseline = cast(ResolvedBaseline, baseline_value)
        try:
            validated_baseline = deepcopy(baseline)
            validated_baseline.__post_init__()
        except (AttributeError, TypeError, ValueError) as exc:
            raise BeamSamplingDerivationError(
                f"selected baseline at index {index} failed canonical validation."
            ) from exc
        if validated_baseline in seen_baselines:
            raise BeamSamplingDerivationError(
                f"selected baseline at index {index} is a duplicate."
            )
        seen_baselines.add(validated_baseline)
        baselines.append(validated_baseline)
    if type(beam_state) is not LoadedBeamState:
        raise BeamSamplingDerivationError(
            "beam_state must be an exact LoadedBeamState."
        )
    try:
        validated_beam_state = deepcopy(beam_state)
        validated_beam_state.__post_init__()
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise BeamSamplingDerivationError(
            "beam_state has invalid handler feature scale or canonical relationship."
        ) from exc
    beam_state = validated_beam_state
    frequencies = _validated_frequencies(observation_frequencies_hz)
    if not _valid_nside(actual_nside):
        raise BeamSamplingDerivationError(
            "actual_nside must be a strict valid HEALPix NSIDE."
        )
    actual_nside_value = cast(int, actual_nside)

    handlers_value = beam_state.handlers
    if type(handlers_value) is not tuple or not handlers_value:
        raise BeamSamplingDerivationError(
            "loaded beam state must contain at least one exact handler."
        )
    handlers_by_id: dict[str, LoadedBeamHandlerState] = {}
    scale_by_handler: dict[str, dict[float, float]] = {}
    for index, handler in enumerate(handlers_value):
        if type(handler) is not LoadedBeamHandlerState:
            raise BeamSamplingDerivationError(
                f"loaded handler at index {index} is not exact."
            )
        if (
            type(handler.handler_id) is not str
            or not handler.handler_id
            or handler.handler_id in handlers_by_id
        ):
            raise BeamSamplingDerivationError(
                f"loaded handler identity at index {index} is invalid or ambiguous."
            )
        if type(handler.kind) is not str or handler.kind not in {"analytic", "fits"}:
            raise BeamSamplingDerivationError(
                f"handler {handler.handler_id!r} has an invalid kind."
            )
        handlers_by_id[handler.handler_id] = handler
        scale_by_handler[handler.handler_id] = _handler_scale_lookup(handler)

    assignment_value = beam_state.assignment_handler_ids
    if type(assignment_value) is not tuple or not assignment_value:
        raise BeamSamplingDerivationError(
            "loaded beam assignment mapping must be a nonempty exact tuple."
        )
    assignment_by_antenna: dict[AntennaId, str] = {}
    for index, pair in enumerate(assignment_value):
        if type(pair) is not tuple or len(pair) != 2:
            raise BeamSamplingDerivationError(
                f"beam assignment at index {index} is not an exact pair."
            )
        antenna_id, handler_id = pair
        if type(antenna_id) is not AntennaId:
            raise BeamSamplingDerivationError(
                f"beam assignment at index {index} has a noncanonical antenna."
            )
        canonical = AntennaId(antenna_id.number, antenna_id.name)
        if canonical in assignment_by_antenna:
            raise BeamSamplingDerivationError(
                f"beam assignment for {canonical!r} is ambiguous."
            )
        if type(handler_id) is not str or handler_id not in handlers_by_id:
            raise BeamSamplingDerivationError(
                f"beam assignment for {canonical!r} references an unknown handler."
            )
        assignment_by_antenna[canonical] = handler_id

    limiting: (
        tuple[
            float,
            ResolvedBaseline,
            float,
            str,
            str,
            _SamplingMetric,
        ]
        | None
    ) = None
    for baseline in baselines:
        handler_id_p = assignment_by_antenna.get(baseline.ant1)
        handler_id_q = assignment_by_antenna.get(baseline.ant2)
        if handler_id_p is None or handler_id_q is None:
            missing = baseline.ant1 if handler_id_p is None else baseline.ant2
            raise BeamSamplingDerivationError(
                "selected baseline endpoint has no exact loaded beam assignment: "
                f"number={missing.number}, name={missing.name!r}."
            )
        handler_p = handlers_by_id[handler_id_p]
        handler_q = handlers_by_id[handler_id_q]
        scales_p = scale_by_handler[handler_id_p]
        scales_q = scale_by_handler[handler_id_q]
        metric: _SamplingMetric = (
            "native_grid_representation_bound"
            if "fits" in {handler_p.kind, handler_q.kind}
            else "analytic_aperture_support"
        )
        for frequency_hz in frequencies:
            scale_p = scales_p.get(frequency_hz)
            scale_q = scales_q.get(frequency_hz)
            if scale_p is None or scale_q is None:
                missing_handler = handler_id_p if scale_p is None else handler_id_q
                raise BeamSamplingDerivationError(
                    f"handler {missing_handler!r} has no exact feature scale for "
                    f"observation frequency {frequency_hz!r} Hz."
                )
            try:
                product_scale = 1.0 / (1.0 / scale_p + 1.0 / scale_q)
            except OverflowError as exc:
                raise BeamSamplingDerivationError(
                    "baseline-product feature scale overflowed for "
                    f"baseline {baseline.ant1.number}-{baseline.ant2.number} at "
                    f"{frequency_hz!r} Hz."
                ) from exc
            if not math.isfinite(product_scale) or product_scale <= 0.0:
                raise BeamSamplingDerivationError(
                    "baseline-product feature scale is not positive and finite for "
                    f"baseline {baseline.ant1.number}-{baseline.ant2.number} at "
                    f"{frequency_hz!r} Hz."
                )
            if limiting is None or product_scale < limiting[0]:
                limiting = (
                    product_scale,
                    baseline,
                    frequency_hz,
                    handler_id_p,
                    handler_id_q,
                    metric,
                )

    if limiting is None:
        raise BeamSamplingDerivationError(
            "selected baseline and observation frequency domain is empty."
        )
    feature, baseline, frequency, handler_p, handler_q, metric = limiting
    limit = feature / float(_SAFETY_FACTOR)
    try:
        recommended = recommend_nside_for_angular_scale(limit)
    except ValueError as exc:
        raise BeamSamplingDerivationError(
            "derived beam-product pixel limit cannot be represented by the retained "
            "HEALPix NSIDE range."
        ) from exc
    return BeamSamplingRequirement(
        actual_nside=actual_nside_value,
        recommended_nside=recommended,
        actual_pixel_scale_rad=_pixel_scale_rad(actual_nside_value),
        product_feature_scale_rad=float(feature),
        pixel_limit_rad=float(limit),
        baseline_ant1=baseline.ant1,
        baseline_ant2=baseline.ant2,
        frequency_hz=float(frequency),
        handler_id_p=handler_p,
        handler_id_q=handler_q,
        handler_kind_p=handlers_by_id[handler_p].kind,
        handler_kind_q=handlers_by_id[handler_q].kind,
        metric_kind=metric,
        safety_factor=_SAFETY_FACTOR,
    )
