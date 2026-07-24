"""Tier 3H.1 canonical selected-baseline beam sampling tests."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import healpy as hp
import pytest

from radiosim.core.beam import BeamSamplingDerivationError
from radiosim.core.beam import models as beam_models
from radiosim.core.instrument import AntennaId, ResolvedBaseline

FREQUENCIES = (100_000_000.0, 200_000_000.0)


def _healpix_module():
    return importlib.import_module("radiosim.utils.healpix")


def _analytic_definition():
    model = beam_models.ResolvedCircularApertureBeamModel(
        "circular_aperture",
        beam_models.ResolvedUniformTaper("uniform"),
    )
    return beam_models.ResolvedAnalyticBeamDefinition(
        "analytic",
        model,
        beam_models._definition_fingerprint("analytic", model),
    )


def _fits_definition(key: str):
    path = Path(f"/tmp/radiosim-tier3h1-{key}.beamfits").resolve(strict=False)
    payload = {
        "path": path,
        "normalization": "peak",
        "angular_interpolation": "bilinear",
        "frequency_interpolation": "linear",
    }
    return beam_models.ResolvedFITSBeamDefinition(
        "fits",
        path,
        "peak",
        "bilinear",
        "linear",
        f"beams.assignments[{key}].beam.path",
        beam_models._definition_fingerprint("fits", payload),
    )


def _file_provenance(key: str) -> beam_models.BeamFileProvenance:
    return beam_models.BeamFileProvenance(
        resolved_path=Path(f"/tmp/radiosim-tier3h1-{key}.beamfits").resolve(
            strict=False
        ),
        size_bytes=1,
        sha256=hashlib.sha256(f"file:{key}".encode()).hexdigest(),
        pyuvdata_version="3.2.1",
        beam_type="efield",
        antenna_type="simple",
        pixel_coordinate_system="az_za",
        mount_type="fixed",
        data_normalization="peak",
        feed_array=("x", "y"),
        x_orientation="east",
        data_shape=(2, 2, 2, 5, 8),
        native_dtype="complex128",
        frequency_min_hz=FREQUENCIES[0],
        frequency_max_hz=FREQUENCIES[-1],
        frequency_count=2,
        azimuth_step_rad=math.pi / 4.0,
        zenith_angle_step_rad=math.pi / 8.0,
        zenith_angle_max_rad=math.pi / 2.0,
        basis_tolerance=1e-12,
        scalar_absolute_tolerance=1e-12,
        scalar_relative_tolerance=1e-10,
        normalization_absolute_tolerance=1e-8,
    )


def _beam_state(
    scales_by_antenna: tuple[tuple[float, float], ...],
    *,
    kinds: tuple[str, ...] | None = None,
    handler_keys: tuple[str, ...] | None = None,
):
    count = len(scales_by_antenna)
    kinds = kinds or tuple("analytic" for _ in range(count))
    handler_keys = handler_keys or tuple(str(index) for index in range(count))
    assert len(kinds) == count
    assert len(handler_keys) == count

    if all(kind == "analytic" for kind in kinds):
        mode = "analytic"
        provenance_source = "analytic_mode"
    elif all(kind == "fits" for kind in kinds):
        mode = "shared_fits" if len(set(handler_keys)) == 1 else "per_antenna_fits"
        provenance_source = (
            "shared_mode" if mode == "shared_fits" else "explicit_assignment"
        )
    else:
        mode = "mixed"
        provenance_source = "explicit_assignment"

    analytic_definition = _analytic_definition()
    definition_by_key: dict[str, Any] = {}
    for key, kind in zip(handler_keys, kinds, strict=True):
        definition_by_key.setdefault(
            key,
            analytic_definition if kind == "analytic" else _fits_definition(key),
        )

    assignments = []
    antenna_ids = tuple(AntennaId(index, f"ANT{index}") for index in range(count))
    for index, (antenna_id, key) in enumerate(
        zip(antenna_ids, handler_keys, strict=True)
    ):
        explicit = provenance_source == "explicit_assignment"
        provenance = beam_models.BeamAssignmentProvenance(
            source=provenance_source,
            input_index=index if explicit else None,
            authored_reference_kind="number" if explicit else None,
            authored_reference_value=index if explicit else None,
            canonical_antenna=antenna_id,
        )
        assignments.append(
            beam_models._create_resolved_beam_assignment(
                antenna_id=antenna_id,
                antenna_diameter_m=14.0,
                definition=definition_by_key[key],
                provenance=provenance,
            )
        )
    unique_definitions = []
    for assignment in assignments:
        if not any(
            definition.definition_fingerprint
            == assignment.definition.definition_fingerprint
            for definition in unique_definitions
        ):
            unique_definitions.append(assignment.definition)
    resolved = beam_models._create_resolved_beam_state(
        mode=mode,
        instrument_fingerprint="a" * 64,
        assignments=tuple(assignments),
        unique_definitions=tuple(unique_definitions),
    )

    first_index_by_key: dict[str, int] = {}
    handlers = []
    for key, kind in zip(handler_keys, kinds, strict=True):
        if key in first_index_by_key:
            continue
        ordinal = len(handlers)
        first_index_by_key[key] = ordinal
        scientific = hashlib.sha256(f"science:{key}".encode()).hexdigest()
        source_index = handler_keys.index(key)
        handlers.append(
            beam_models.LoadedBeamHandlerState(
                handler_id=f"beam-{ordinal:04d}-{scientific[:12]}",
                kind=kind,
                definition_fingerprint=definition_by_key[key].definition_fingerprint,
                scientific_fingerprint=scientific,
                file=_file_provenance(key) if kind == "fits" else None,
                voltage_feature_scale_by_frequency=tuple(
                    (frequency, float(scale))
                    for frequency, scale in zip(
                        FREQUENCIES,
                        scales_by_antenna[source_index],
                        strict=True,
                    )
                ),
            )
        )
    assignment_handler_ids = tuple(
        (
            antenna_id,
            handlers[first_index_by_key[key]].handler_id,
        )
        for antenna_id, key in zip(antenna_ids, handler_keys, strict=True)
    )
    return beam_models._create_loaded_beam_state(
        resolved=resolved,
        handlers=tuple(handlers),
        assignment_handler_ids=assignment_handler_ids,
    )


def _baseline(
    p: int,
    q: int,
    *,
    names: tuple[str, str] | None = None,
) -> ResolvedBaseline:
    p_name, q_name = names or (f"ANT{p}", f"ANT{q}")
    ant1 = AntennaId(p, p_name)
    ant2 = AntennaId(q, q_name)
    if p == q:
        return ResolvedBaseline(
            ant1,
            ant2,
            (0.0, 0.0, 0.0),
            0.0,
            True,
            None,
        )
    distance = float(q - p)
    return ResolvedBaseline(
        ant1,
        ant2,
        (distance, 0.0, 0.0),
        distance,
        False,
        90.0,
    )


def _derive(
    state,
    baselines: tuple[ResolvedBaseline, ...],
    *,
    frequencies: tuple[float, ...] = FREQUENCIES,
    actual_nside: int = 64,
):
    return _healpix_module().derive_beam_sampling_requirement(
        selected_baselines=baselines,
        beam_state=state,
        observation_frequencies_hz=frequencies,
        actual_nside=actual_nside,
    )


def test_homogeneous_cross_uses_baseline_product_harmonic_scale() -> None:
    state = _beam_state(((2.0, 1.0), (2.0, 1.0)), handler_keys=("a", "a"))

    requirement = _derive(state, (_baseline(0, 1),))

    assert requirement.product_feature_scale_rad == 0.5
    assert requirement.frequency_hz == FREQUENCIES[1]
    assert requirement.baseline_ant1 == AntennaId(0, "ANT0")
    assert requirement.baseline_ant2 == AntennaId(1, "ANT1")
    assert requirement.handler_id_p == requirement.handler_id_q
    assert requirement.handler_kind_p == "analytic"
    assert requirement.handler_kind_q == "analytic"
    assert requirement.metric_kind == "analytic_aperture_support"


def test_heterogeneous_analytic_uses_both_endpoint_scales() -> None:
    state = _beam_state(((1.0, 1.0), (2.0, 2.0)))

    requirement = _derive(state, (_baseline(0, 1),))

    assert requirement.product_feature_scale_rad == pytest.approx(2.0 / 3.0)


def test_highest_frequency_tightens_analytic_sampling() -> None:
    state = _beam_state(((4.0, 1.0), (2.0, 0.5)))

    requirement = _derive(state, (_baseline(0, 1),))

    assert requirement.frequency_hz == FREQUENCIES[1]
    assert requirement.product_feature_scale_rad == pytest.approx(1.0 / 3.0)


def test_only_selected_baselines_can_limit_sampling() -> None:
    state = _beam_state(((2.0, 2.0), (2.0, 2.0), (0.01, 0.01)))

    requirement = _derive(state, (_baseline(0, 1),))

    assert requirement.product_feature_scale_rad == 1.0
    assert requirement.baseline_ant2 == AntennaId(1, "ANT1")


def test_auto_only_selection_evaluates_every_selected_auto() -> None:
    state = _beam_state(((2.0, 2.0), (0.5, 0.5)))

    requirement = _derive(state, (_baseline(0, 0), _baseline(1, 1)))

    assert requirement.product_feature_scale_rad == 0.25
    assert requirement.baseline_ant1 == requirement.baseline_ant2
    assert requirement.baseline_ant1 == AntennaId(1, "ANT1")


def test_mixed_auto_and_cross_selection_uses_same_formula() -> None:
    state = _beam_state(((1.0, 1.0), (4.0, 4.0)))

    requirement = _derive(state, (_baseline(0, 0), _baseline(0, 1)))

    assert requirement.product_feature_scale_rad == 0.5
    assert requirement.baseline_ant1 == requirement.baseline_ant2


def test_shared_fits_uses_native_grid_representation_metric() -> None:
    state = _beam_state(
        ((0.6, 0.6), (0.6, 0.6)),
        kinds=("fits", "fits"),
        handler_keys=("shared", "shared"),
    )

    requirement = _derive(state, (_baseline(0, 1),))

    assert requirement.product_feature_scale_rad == pytest.approx(0.3)
    assert requirement.metric_kind == "native_grid_representation_bound"
    assert requirement.handler_id_p == requirement.handler_id_q
    assert requirement.handler_kind_p == "fits"
    assert requirement.handler_kind_q == "fits"


def test_distinct_fits_handlers_use_both_native_scales() -> None:
    state = _beam_state(
        ((0.6, 0.6), (0.3, 0.3)),
        kinds=("fits", "fits"),
    )

    requirement = _derive(state, (_baseline(0, 1),))

    assert requirement.product_feature_scale_rad == pytest.approx(0.2)
    assert requirement.handler_id_p != requirement.handler_id_q
    assert requirement.metric_kind == "native_grid_representation_bound"


def test_mixed_analytic_fits_product_retains_fits_metric() -> None:
    state = _beam_state(
        ((1.0, 0.5), (0.6, 0.6)),
        kinds=("analytic", "fits"),
    )

    requirement = _derive(state, (_baseline(0, 1),))

    assert requirement.frequency_hz == FREQUENCIES[1]
    assert requirement.product_feature_scale_rad == pytest.approx(3.0 / 11.0)
    assert requirement.handler_kind_p == "analytic"
    assert requirement.handler_kind_q == "fits"
    assert requirement.metric_kind == "native_grid_representation_bound"


def test_equal_scale_tie_uses_selected_baseline_then_frequency_order() -> None:
    state = _beam_state(((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)))
    baselines = (_baseline(0, 1), _baseline(0, 2))

    requirement = _derive(state, baselines)

    assert requirement.baseline_ant1 == baselines[0].ant1
    assert requirement.baseline_ant2 == baselines[0].ant2
    assert requirement.frequency_hz == FREQUENCIES[0]


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (lambda state: (), "selected baseline"),
        (lambda state: (_baseline(0, 2),), "assignment"),
    ),
)
def test_empty_or_unmatched_selected_domain_raises(
    mutator,
    match: str,
) -> None:
    state = _beam_state(((1.0, 1.0), (1.0, 1.0)))

    with pytest.raises(BeamSamplingDerivationError, match=match):
        _derive(state, mutator(state))


def test_exact_frequency_lookup_has_no_nearest_fallback() -> None:
    state = _beam_state(((1.0, 1.0), (1.0, 1.0)))

    with pytest.raises(BeamSamplingDerivationError, match="frequency"):
        _derive(
            state,
            (_baseline(0, 1),),
            frequencies=(150_000_000.0,),
        )


def test_duplicate_selected_baseline_raises_typed_error() -> None:
    state = _beam_state(((1.0, 1.0), (1.0, 1.0)))
    baseline = _baseline(0, 1)

    with pytest.raises(BeamSamplingDerivationError, match="duplicate"):
        _derive(state, (baseline, baseline))


def test_noncanonical_selected_baseline_raises_typed_error() -> None:
    state = _beam_state(((1.0, 1.0), (1.0, 1.0)))
    canonical = _baseline(0, 1)
    forged = object.__new__(ResolvedBaseline)
    object.__setattr__(forged, "ant1", canonical.ant2)
    object.__setattr__(forged, "ant2", canonical.ant1)
    object.__setattr__(forged, "vector_enu_m", canonical.vector_enu_m)
    object.__setattr__(forged, "length_m", canonical.length_m)
    object.__setattr__(forged, "is_autocorrelation", canonical.is_autocorrelation)
    object.__setattr__(forged, "azimuth_deg", canonical.azimuth_deg)

    with pytest.raises(BeamSamplingDerivationError, match="baseline"):
        _derive(state, (forged,))


def test_nonincreasing_handler_scale_axis_raises_typed_error() -> None:
    state = _beam_state(((1.0, 1.0), (1.0, 1.0)))
    handler = state.handlers[0]
    object.__setattr__(
        handler,
        "voltage_feature_scale_by_frequency",
        tuple(reversed(handler.voltage_feature_scale_by_frequency)),
    )

    with pytest.raises(BeamSamplingDerivationError, match="beam_state"):
        _derive(state, (_baseline(0, 1),))


def test_loaded_assignment_order_disagreement_raises_typed_error() -> None:
    state = _beam_state(((1.0, 1.0), (1.0, 1.0)))
    object.__setattr__(
        state,
        "assignment_handler_ids",
        tuple(reversed(state.assignment_handler_ids)),
    )

    with pytest.raises(BeamSamplingDerivationError, match="beam_state"):
        _derive(state, (_baseline(0, 1),))


@pytest.mark.parametrize("invalid_scale", (0.0, -1.0, float("nan"), float("inf")))
def test_invalid_loaded_feature_scale_raises_typed_error(
    invalid_scale: float,
) -> None:
    state = _beam_state(((1.0, 1.0), (1.0, 1.0)))
    handler = state.handlers[0]
    object.__setattr__(
        handler,
        "voltage_feature_scale_by_frequency",
        ((FREQUENCIES[0], invalid_scale), (FREQUENCIES[1], 1.0)),
    )

    with pytest.raises(BeamSamplingDerivationError, match="feature scale"):
        _derive(state, (_baseline(0, 1),))


def test_forged_handler_identity_raises_typed_error() -> None:
    state = _beam_state(((1.0, 1.0), (1.0, 1.0)))
    forged = tuple(
        (antenna_id, "beam-9999-forgedvalue0")
        for antenna_id, _handler_id in state.assignment_handler_ids
    )
    object.__setattr__(state, "assignment_handler_ids", forged)

    with pytest.raises(BeamSamplingDerivationError, match="handler"):
        _derive(state, (_baseline(0, 1),))


@pytest.mark.parametrize(
    ("baselines", "frequencies", "actual_nside"),
    (
        ([_baseline(0, 1)], FREQUENCIES, 64),
        ((_baseline(0, 1),), list(FREQUENCIES), 64),
        ((_baseline(0, 1),), FREQUENCIES, True),
        ((_baseline(0, 1),), FREQUENCIES, 3),
    ),
)
def test_malformed_derivation_inputs_raise_typed_error(
    baselines: object,
    frequencies: object,
    actual_nside: object,
) -> None:
    state = _beam_state(((1.0, 1.0), (1.0, 1.0)))

    with pytest.raises(BeamSamplingDerivationError):
        _healpix_module().derive_beam_sampling_requirement(
            selected_baselines=baselines,
            beam_state=state,
            observation_frequencies_hz=frequencies,
            actual_nside=actual_nside,
        )


def test_requirement_is_frozen_owned_hashable_and_json_safe() -> None:
    state = _beam_state(((1.0, 1.0), (1.0, 1.0)))
    baseline = _baseline(0, 1)

    requirement = _derive(state, (baseline,))

    assert type(requirement.baseline_ant1) is AntennaId
    assert type(requirement.baseline_ant2) is AntennaId
    assert requirement.baseline_ant1 is not baseline.ant1
    assert requirement.baseline_ant2 is not baseline.ant2
    assert requirement.actual_pixel_scale_rad == float(hp.nside2resol(64))
    assert requirement.pixel_limit_rad == (requirement.product_feature_scale_rad / 5.0)
    assert requirement.safety_factor == 5
    assert hp.nside2resol(requirement.recommended_nside) <= (
        requirement.pixel_limit_rad
    )
    hash(requirement)
    json.dumps(requirement.to_snapshot(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        requirement.actual_nside = 128


def test_requirement_rejects_inconsistent_public_state() -> None:
    state = _beam_state(((1.0, 1.0), (1.0, 1.0)))
    requirement = _derive(state, (_baseline(0, 1),))

    for changes in (
        {"actual_nside": 3},
        {"recommended_nside": 3},
        {"actual_pixel_scale_rad": requirement.actual_pixel_scale_rad * 2.0},
        {"pixel_limit_rad": requirement.pixel_limit_rad * 2.0},
        {"product_feature_scale_rad": float("nan")},
        {"handler_id_p": ""},
        {"handler_id_p": "forged-handler"},
        {"handler_kind_p": "unknown"},
        {"metric_kind": "physical_bandwidth"},
        {"safety_factor": 4},
        {
            "handler_kind_p": "fits",
            "metric_kind": "analytic_aperture_support",
        },
        {
            "handler_kind_p": "analytic",
            "handler_kind_q": "analytic",
            "metric_kind": "native_grid_representation_bound",
        },
        {
            "baseline_ant1": AntennaId(2, "ANT2"),
            "baseline_ant2": AntennaId(1, "ANT1"),
        },
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(requirement, **changes)
