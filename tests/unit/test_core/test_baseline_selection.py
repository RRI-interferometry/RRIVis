"""Tier 2F exact baseline selection and provenance contract tests."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, fields
from typing import Any

import numpy as np
import pytest

from radiosim.core.baseline_resolution import (
    BaselineSelectionError,
    EmptyBaselineSelectionError,
    generate_resolved_baselines,
    select_resolved_baselines,
)
from radiosim.core.instrument import (
    AntennaFieldSource,
    AntennaId,
    AntennaProvenance,
    BaselineSelectionCriteriaSnapshot,
    BaselineSelectionProvenance,
    ResolvedAntenna,
    ResolvedBaseline,
    ResolvedBaselineSelection,
    ResolvedEarthLocation,
    _create_resolved_instrument,
)
from radiosim.io.instrument_config import (
    AzimuthRangeConfig,
    BaselineSelectionConfig,
    LengthRangeConfig,
    LengthRangesConfig,
    LengthTargetsConfig,
)


def _antenna(number: int, position: tuple[float, float, float]) -> ResolvedAntenna:
    return ResolvedAntenna(
        id=AntennaId(number, f"A{number}"),
        position_enu_m=position,
        diameter_m=12.0,
        mount_type=None,
        beam_id=None,
        provenance=AntennaProvenance(
            identity_source=AntennaFieldSource.LAYOUT_FILE,
            position_source=AntennaFieldSource.LAYOUT_FILE,
            diameter_source=AntennaFieldSource.CONFIG_DEFAULT,
            source_diameter_m=None,
            mount_source=None,
            beam_id_source=None,
            source_record=f"row:{number}",
        ),
    )


def _instrument(*positions: tuple[int, tuple[float, float, float]]):
    if not positions:
        positions = (
            (0, (0.0, 0.0, 0.0)),
            (1, (0.0, 10.0, 0.0)),
            (2, (10.0, 0.0, 0.0)),
        )
    location = ResolvedEarthLocation(
        longitude_deg=0.0,
        latitude_deg=0.0,
        height_m=0.0,
        itrs_xyz_m=(1.0, 2.0, 3.0),
        source=AntennaFieldSource.EXPLICIT_CONFIG,
        reference="fixture:location",
    )
    return _create_resolved_instrument(
        name="Fixture Array",
        location=location,
        antennas=tuple(_antenna(number, position) for number, position in positions),
        source_kind="fixture",
        source_reference="fixture:layout",
        source_format="radiosim",
        registry_policy=None,
        telescope_name_source=AntennaFieldSource.EXPLICIT_CONFIG,
        location_source=AntennaFieldSource.EXPLICIT_CONFIG,
        source_location_itrs_xyz_m=None,
        location_separation_m=None,
        pyuvdata_version=None,
        source_sha256=None,
    )


def _select(instrument, config: BaselineSelectionConfig):
    baselines = generate_resolved_baselines(instrument)
    return select_resolved_baselines(
        baselines,
        instrument=instrument,
        config=config,
    )


def _two_antenna_at_angle(angle_deg: float, *, length_m: float = 10.0):
    radians = math.radians(angle_deg)
    east = length_m * math.sin(radians)
    north = length_m * math.cos(radians)
    return _instrument((0, (0.0, 0.0, 0.0)), (1, (east, north, 0.0)))


def _criteria(**updates: Any) -> BaselineSelectionCriteriaSnapshot:
    values = {
        "correlations": "all",
        "length_mode": None,
        "length_targets_m": (),
        "length_tolerance_m": None,
        "length_ranges_m": (),
        "azimuth_ranges_deg": (),
    }
    values.update(updates)
    return BaselineSelectionCriteriaSnapshot(**values)


def _provenance(**updates: Any) -> BaselineSelectionProvenance:
    values = {
        "schema_version": "radiosim.baseline-selection.v1",
        "instrument_sha256": "a" * 64,
        "criteria": _criteria(),
        "generated_count": 1,
        "after_correlation_count": 1,
        "after_length_count": 1,
        "after_azimuth_count": 1,
        "azimuth_exempt_auto_count": 0,
        "selected_ids": ((0, 0),),
    }
    values.update(updates)
    return BaselineSelectionProvenance(**values)


def _auto(number: int = 0) -> ResolvedBaseline:
    antenna_id = AntennaId(number, f"A{number}")
    return ResolvedBaseline(
        antenna_id,
        antenna_id,
        (0.0, 0.0, 0.0),
        0.0,
        True,
        None,
    )


def test_default_selection_retains_all_pairs_and_records_every_stage():
    instrument = _instrument()
    selection = _select(instrument, BaselineSelectionConfig())

    assert tuple(
        (baseline.ant1.number, baseline.ant2.number) for baseline in selection.baselines
    ) == ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    assert selection.provenance.criteria == _criteria()
    assert selection.provenance.generated_count == 6
    assert selection.provenance.after_correlation_count == 6
    assert selection.provenance.after_length_count == 6
    assert selection.provenance.after_azimuth_count == 6
    assert selection.provenance.azimuth_exempt_auto_count == 0
    assert selection.provenance.selected_ids == (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    )
    assert selection.provenance.instrument_sha256 == (
        instrument.provenance.instrument_sha256
    )


@pytest.mark.parametrize(
    ("correlations", "expected"),
    [
        ("cross", ((0, 1), (0, 2), (1, 2))),
        ("auto", ((0, 0), (1, 1), (2, 2))),
    ],
)
def test_correlation_modes_are_exact_and_stable(correlations, expected):
    selection = _select(
        _instrument(),
        BaselineSelectionConfig(correlations=correlations),
    )

    assert selection.provenance.selected_ids == expected
    assert selection.provenance.after_correlation_count == 3


@pytest.mark.parametrize(
    ("target", "tolerance", "selected"),
    [
        (10.0, 0.0, True),
        (10.5 - 0.5e-9, 0.5, True),
        (10.5 + 0.5e-9, 0.5, True),
        (10.5 + 2e-9, 0.5, False),
    ],
)
def test_target_filter_exact_tolerance_allowance_and_just_outside(
    target, tolerance, selected
):
    instrument = _instrument((0, (0.0, 0.0, 0.0)), (1, (10.0, 0.0, 0.0)))
    config = BaselineSelectionConfig(
        correlations="cross",
        length_filter=LengthTargetsConfig(
            targets_m=(target,),
            tolerance_m=tolerance,
        ),
    )

    if selected:
        assert _select(instrument, config).provenance.selected_ids == ((0, 1),)
    else:
        with pytest.raises(EmptyBaselineSelectionError):
            _select(instrument, config)


def test_target_union_zero_and_correlation_interaction():
    instrument = _instrument()
    config = BaselineSelectionConfig(
        correlations="all",
        length_filter=LengthTargetsConfig(
            targets_m=(math.sqrt(200.0), 0.0),
            tolerance_m=0.0,
        ),
    )

    selection = _select(instrument, config)

    assert selection.provenance.selected_ids == (
        (0, 0),
        (1, 1),
        (1, 2),
        (2, 2),
    )
    assert selection.provenance.after_length_count == 4

    crosses = _select(instrument, config.model_copy(update={"correlations": "cross"}))
    assert crosses.provenance.selected_ids == ((1, 2),)


@pytest.mark.parametrize(
    ("minimum", "maximum", "selected"),
    [
        (10.0, 10.0, True),
        (10.0 + 0.5e-9, 20.0, True),
        (0.0, 10.0 - 0.5e-9, True),
        (10.0 + 2e-9, 20.0, False),
        (0.0, 10.0 - 2e-9, False),
    ],
)
def test_range_filter_boundaries_allowance_and_just_outside(minimum, maximum, selected):
    instrument = _instrument((0, (0.0, 0.0, 0.0)), (1, (10.0, 0.0, 0.0)))
    config = BaselineSelectionConfig(
        correlations="cross",
        length_filter=LengthRangesConfig(
            ranges_m=(LengthRangeConfig(min_m=minimum, max_m=maximum),)
        ),
    )

    if selected:
        assert _select(instrument, config).provenance.selected_ids == ((0, 1),)
    else:
        with pytest.raises(EmptyBaselineSelectionError):
            _select(instrument, config)


def test_range_union_overlap_zero_and_correlation_interaction():
    instrument = _instrument()
    config = BaselineSelectionConfig(
        length_filter=LengthRangesConfig(
            ranges_m=(
                LengthRangeConfig(min_m=9.0, max_m=11.0),
                LengthRangeConfig(min_m=0.0, max_m=10.0),
                LengthRangeConfig(min_m=9.5, max_m=15.0),
            )
        )
    )

    selection = _select(instrument, config)

    assert selection.provenance.selected_ids == (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    )
    assert selection.provenance.criteria.length_ranges_m == (
        (0.0, 10.0),
        (9.0, 11.0),
        (9.5, 15.0),
    )
    autos = _select(instrument, config.model_copy(update={"correlations": "auto"}))
    assert autos.provenance.selected_ids == ((0, 0), (1, 1), (2, 2))


@pytest.mark.parametrize(
    ("angle", "start", "end"),
    [
        (0.0, 0.0, 10.0),
        (90.0, 80.0, 100.0),
        (0.0, 170.0, 10.0),
        (175.0, 170.0, 10.0),
        (30.0, 30.0, 60.0),
        (60.0, 30.0, 60.0),
    ],
)
def test_normal_wrapped_and_closed_azimuth_ranges(angle, start, end):
    selection = _select(
        _two_antenna_at_angle(angle),
        BaselineSelectionConfig(
            correlations="cross",
            azimuth_ranges_deg=(AzimuthRangeConfig(start_deg=start, end_deg=end),),
        ),
    )

    assert selection.provenance.selected_ids == ((0, 1),)


@pytest.mark.parametrize(
    ("offset", "selected"),
    [(0.5e-12, True), (2e-12, False)],
)
def test_azimuth_boundary_allowance_and_just_outside(offset, selected):
    instrument = _two_antenna_at_angle(30.0)
    actual = generate_resolved_baselines(instrument)[1].azimuth_deg
    assert actual is not None
    config = BaselineSelectionConfig(
        correlations="cross",
        azimuth_ranges_deg=(
            AzimuthRangeConfig(start_deg=actual + offset, end_deg=60.0),
        ),
    )

    if selected:
        assert _select(instrument, config).provenance.selected_ids == ((0, 1),)
    else:
        with pytest.raises(EmptyBaselineSelectionError):
            _select(instrument, config)


@pytest.mark.parametrize(
    ("angle", "start", "end", "selected"),
    [
        (180.0 - 0.5e-12, 0.0, 10.0, True),
        (180.0 - 2e-12, 0.0, 10.0, False),
        (0.0, 170.0, 180.0 - 0.5e-12, True),
        (0.0, 170.0, 180.0 - 2e-12, False),
    ],
)
def test_azimuth_boundary_allowance_is_continuous_across_axial_seam(
    angle, start, end, selected
):
    instrument = _two_antenna_at_angle(angle)
    config = BaselineSelectionConfig(
        correlations="cross",
        azimuth_ranges_deg=(AzimuthRangeConfig(start_deg=start, end_deg=end),),
    )

    if selected:
        assert _select(instrument, config).provenance.selected_ids == ((0, 1),)
    else:
        with pytest.raises(EmptyBaselineSelectionError):
            _select(instrument, config)


def test_azimuth_range_union_and_category_intersection_preserve_order():
    instrument = _instrument(
        (0, (0.0, 0.0, 0.0)),
        (1, (0.0, 10.0, 0.0)),
        (2, (10.0, 0.0, 0.0)),
        (3, (0.0, -20.0, 0.0)),
    )
    config = BaselineSelectionConfig(
        correlations="cross",
        length_filter=LengthRangesConfig(
            ranges_m=(LengthRangeConfig(min_m=9.0, max_m=11.0),)
        ),
        azimuth_ranges_deg=(
            AzimuthRangeConfig(start_deg=80.0, end_deg=100.0),
            AzimuthRangeConfig(start_deg=170.0, end_deg=10.0),
        ),
    )

    selection = _select(instrument, config)

    assert selection.provenance.selected_ids == ((0, 1), (0, 2))
    assert selection.provenance.generated_count == 10
    assert selection.provenance.after_correlation_count == 6
    assert selection.provenance.after_length_count == 2
    assert selection.provenance.after_azimuth_count == 2


def test_opposite_vectors_and_number_assignment_select_the_same_axial_orientation():
    forward = _instrument((1, (0.0, 0.0, 0.0)), (9, (3.0, 4.0, 0.0)))
    reverse = _instrument((1, (3.0, 4.0, 0.0)), (9, (0.0, 0.0, 0.0)))
    angle = generate_resolved_baselines(forward)[1].azimuth_deg
    assert angle is not None
    config = BaselineSelectionConfig(
        correlations="cross",
        azimuth_ranges_deg=(
            AzimuthRangeConfig(start_deg=angle - 1.0, end_deg=angle + 1.0),
        ),
    )

    assert _select(forward, config).provenance.selected_ids == ((1, 9),)
    assert _select(reverse, config).provenance.selected_ids == ((1, 9),)


def test_autos_survive_active_azimuth_filter_and_exemption_is_explicit():
    selection = _select(
        _instrument(),
        BaselineSelectionConfig(
            correlations="all",
            azimuth_ranges_deg=(AzimuthRangeConfig(start_deg=80.0, end_deg=100.0),),
        ),
    )

    assert selection.provenance.selected_ids == (
        (0, 0),
        (0, 2),
        (1, 1),
        (2, 2),
    )
    assert selection.provenance.azimuth_exempt_auto_count == 3
    assert selection.provenance.after_length_count == 6
    assert selection.provenance.after_azimuth_count == 4


def test_normalized_criteria_sort_equivalent_caller_order_without_merging():
    instrument = _instrument()
    first = BaselineSelectionConfig(
        length_filter=LengthTargetsConfig(targets_m=(14.0, 0.0, 10.0), tolerance_m=0.5),
        azimuth_ranges_deg=(
            AzimuthRangeConfig(start_deg=170.0, end_deg=10.0),
            AzimuthRangeConfig(start_deg=20.0, end_deg=40.0),
        ),
    )
    second = BaselineSelectionConfig(
        length_filter=LengthTargetsConfig(targets_m=(10.0, 14.0, 0.0), tolerance_m=0.5),
        azimuth_ranges_deg=(
            AzimuthRangeConfig(start_deg=20.0, end_deg=40.0),
            AzimuthRangeConfig(start_deg=170.0, end_deg=10.0),
        ),
    )

    first_selection = _select(instrument, first)
    second_selection = _select(instrument, second)

    assert first_selection == second_selection
    assert hash(first_selection) == hash(second_selection)
    assert first_selection.provenance.criteria.length_targets_m == (0.0, 10.0, 14.0)
    assert first_selection.provenance.criteria.azimuth_ranges_deg == (
        (20.0, 40.0),
        (170.0, 10.0),
    )


def test_empty_selection_error_includes_stable_normalized_criteria():
    instrument = _instrument()
    config = BaselineSelectionConfig(
        correlations="cross",
        length_filter=LengthTargetsConfig(targets_m=(999.0, 998.0), tolerance_m=0.0),
    )

    with pytest.raises(EmptyBaselineSelectionError) as exc_info:
        _select(instrument, config)

    message = str(exc_info.value)
    assert '"correlations":"cross"' in message
    assert '"length_targets_m":[998.0,999.0]' in message


def test_selector_rejects_mapping_config_and_mutable_config_subclass():
    instrument = _instrument()
    baselines = generate_resolved_baselines(instrument)

    class MutableSelectionConfig(BaselineSelectionConfig):
        pass

    with pytest.raises(TypeError):
        select_resolved_baselines(
            baselines,
            instrument=instrument,
            config={"correlations": "all"},
        )
    with pytest.raises(TypeError):
        select_resolved_baselines(
            baselines,
            instrument=instrument,
            config=MutableSelectionConfig(),
        )


def test_selector_rejects_noncanonical_or_instrument_mismatched_baselines():
    instrument = _instrument()
    baselines = generate_resolved_baselines(instrument)

    with pytest.raises((TypeError, BaselineSelectionError)):
        select_resolved_baselines(
            list(baselines),
            instrument=instrument,
            config=BaselineSelectionConfig(),
        )
    with pytest.raises(BaselineSelectionError):
        select_resolved_baselines(
            baselines[:-1],
            instrument=instrument,
            config=BaselineSelectionConfig(),
        )
    other = _instrument(
        (0, (0.0, 0.0, 0.0)),
        (1, (0.0, 10.0, 0.0)),
        (2, (20.0, 0.0, 0.0)),
    )
    with pytest.raises(BaselineSelectionError):
        select_resolved_baselines(
            baselines,
            instrument=other,
            config=BaselineSelectionConfig(),
        )


def test_selector_does_not_mutate_inputs_and_returns_repeatable_new_values():
    instrument = _instrument()
    baselines = generate_resolved_baselines(instrument)
    config = BaselineSelectionConfig(correlations="cross")
    instrument_before = instrument.to_snapshot()
    config_before = config.model_dump(mode="json")

    first = select_resolved_baselines(baselines, instrument=instrument, config=config)
    second = select_resolved_baselines(baselines, instrument=instrument, config=config)

    assert first == second
    assert first is not second
    assert first.baselines is not second.baselines
    assert instrument.to_snapshot() == instrument_before
    assert config.model_dump(mode="json") == config_before
    assert baselines == generate_resolved_baselines(instrument)


def test_criteria_snapshot_normalizes_builtin_values_and_owns_mutable_inputs():
    targets = [np.float64(20.0), 10]
    azimuths = [[170, 10], [20.0, 40.0]]
    snapshot = _criteria(
        correlations="cross",
        length_mode="targets",
        length_targets_m=targets,
        length_tolerance_m=np.float64(0.5),
        azimuth_ranges_deg=azimuths,
    )
    targets.clear()
    azimuths[0][0] = 1.0

    assert snapshot.length_targets_m == (10.0, 20.0)
    assert snapshot.length_tolerance_m == 0.5
    assert snapshot.azimuth_ranges_deg == ((20.0, 40.0), (170.0, 10.0))
    assert type(snapshot.length_targets_m) is tuple
    assert all(type(value) is float for value in snapshot.length_targets_m)
    assert isinstance(hash(snapshot), int)
    with pytest.raises(FrozenInstanceError):
        snapshot.correlations = "auto"


@pytest.mark.parametrize(
    "updates",
    [
        {"correlations": "both"},
        {"length_mode": "unknown"},
        {"length_targets_m": (1.0,)},
        {"length_tolerance_m": 0.0},
        {"length_ranges_m": ((0.0, 1.0),)},
        {"length_mode": "targets", "length_targets_m": (), "length_tolerance_m": 0.0},
        {
            "length_mode": "targets",
            "length_targets_m": (1.0,),
            "length_tolerance_m": None,
        },
        {
            "length_mode": "targets",
            "length_targets_m": (1.0,),
            "length_tolerance_m": -1.0,
        },
        {
            "length_mode": "targets",
            "length_targets_m": (1.0, 1.0),
            "length_tolerance_m": 0.0,
        },
        {
            "length_mode": "targets",
            "length_targets_m": (1.0,),
            "length_tolerance_m": 0.0,
            "length_ranges_m": ((0.0, 1.0),),
        },
        {"length_mode": "ranges", "length_ranges_m": ()},
        {"length_mode": "ranges", "length_ranges_m": ((2.0, 1.0),)},
        {"length_mode": "ranges", "length_ranges_m": ((0.0, math.inf),)},
        {
            "length_mode": "ranges",
            "length_ranges_m": ((0.0, 1.0),),
            "length_targets_m": (1.0,),
        },
        {"azimuth_ranges_deg": ((0.0, 0.0),)},
        {"azimuth_ranges_deg": ((-1.0, 10.0),)},
        {"azimuth_ranges_deg": ((170.0, 10.0), (170.0, 10.0))},
    ],
)
def test_criteria_snapshot_rejects_malformed_or_contradictory_states(updates):
    with pytest.raises((TypeError, ValueError)):
        _criteria(**updates)


def test_provenance_normalizes_counts_and_selected_ids_to_builtin_values():
    provenance = _provenance(
        criteria=_criteria(
            correlations="cross",
            length_mode="targets",
            length_targets_m=(1.0,),
            length_tolerance_m=0.0,
        ),
        generated_count=np.int64(3),
        after_correlation_count=np.int64(1),
        after_length_count=np.int64(1),
        after_azimuth_count=np.int64(1),
        selected_ids=[[np.int64(1), np.int64(2)]],
    )

    assert provenance.selected_ids == ((1, 2),)
    assert type(provenance.selected_ids[0][0]) is int
    assert type(provenance.generated_count) is int
    assert isinstance(hash(provenance), int)


@pytest.mark.parametrize(
    "updates",
    [
        {"schema_version": "radiosim.baseline-selection.v2"},
        {"instrument_sha256": "A" * 64},
        {"instrument_sha256": "a" * 63},
        {"criteria": object()},
        {"generated_count": -1},
        {"generated_count": True},
        {"generated_count": 0, "after_correlation_count": 1},
        {"after_correlation_count": 0, "after_length_count": 1},
        {"after_length_count": 0, "after_azimuth_count": 1},
        {"after_azimuth_count": 0},
        {"azimuth_exempt_auto_count": 2},
        {"selected_ids": ((1, 0),)},
        {"selected_ids": ((1, 2), (0, 0)), "after_azimuth_count": 2},
        {"selected_ids": ((0, 0), (0, 0)), "after_azimuth_count": 2},
        {"selected_ids": ((0, True),)},
        {
            "criteria": _criteria(azimuth_ranges_deg=((0.0, 10.0),)),
            "azimuth_exempt_auto_count": 2,
        },
    ],
)
def test_provenance_rejects_invalid_schema_counts_and_ids(updates):
    with pytest.raises((TypeError, ValueError)):
        _provenance(**updates)


def test_provenance_rejects_exempt_count_when_no_azimuth_filter_is_active():
    with pytest.raises(ValueError):
        _provenance(azimuth_exempt_auto_count=1)


def test_provenance_rejects_stage_counts_that_contradict_inactive_filters():
    with pytest.raises(ValueError):
        _provenance(
            generated_count=3,
            after_correlation_count=3,
            after_length_count=2,
            after_azimuth_count=2,
        )
    with pytest.raises(ValueError):
        _provenance(
            generated_count=3,
            after_correlation_count=3,
            after_length_count=3,
            after_azimuth_count=2,
        )


def test_provenance_rejects_more_exempt_autos_than_survive_azimuth_filter():
    with pytest.raises(ValueError):
        _provenance(
            criteria=_criteria(azimuth_ranges_deg=((0.0, 10.0),)),
            generated_count=3,
            after_correlation_count=3,
            after_length_count=3,
            after_azimuth_count=1,
            azimuth_exempt_auto_count=2,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {
            "generated_count": 3,
            "after_correlation_count": 2,
        },
        {
            "criteria": _criteria(correlations="auto"),
            "selected_ids": ((0, 1),),
        },
        {
            "criteria": _criteria(correlations="cross"),
            "selected_ids": ((0, 0),),
        },
        {
            "criteria": _criteria(azimuth_ranges_deg=((0.0, 10.0),)),
            "selected_ids": ((0, 0),),
            "azimuth_exempt_auto_count": 0,
        },
    ],
)
def test_provenance_rejects_correlation_and_auto_exemption_contradictions(updates):
    with pytest.raises(ValueError):
        _provenance(**updates)


@pytest.mark.parametrize(
    "updates",
    [
        {
            "generated_count": 2,
            "after_correlation_count": 2,
            "after_length_count": 2,
            "after_azimuth_count": 2,
            "selected_ids": ((0, 0), (1, 1)),
        },
        {
            "criteria": _criteria(correlations="auto"),
            "generated_count": 6,
            "after_correlation_count": 2,
            "after_length_count": 2,
            "after_azimuth_count": 2,
            "selected_ids": ((0, 0), (1, 1)),
        },
        {
            "criteria": _criteria(correlations="cross"),
            "generated_count": 6,
            "after_correlation_count": 2,
            "after_length_count": 2,
            "after_azimuth_count": 2,
            "selected_ids": ((0, 1), (0, 2)),
        },
        {
            "selected_ids": ((0, 1),),
        },
    ],
)
def test_provenance_rejects_nontriangular_or_impossible_correlation_counts(updates):
    with pytest.raises(ValueError):
        _provenance(**updates)


def test_selection_constructor_rejects_baselines_that_contradict_active_criteria():
    cross = ResolvedBaseline(
        AntennaId(0, "A0"),
        AntennaId(1, "A1"),
        (10.0, 0.0, 0.0),
        10.0,
        False,
        90.0,
    )
    target_criteria = _criteria(
        correlations="cross",
        length_mode="targets",
        length_targets_m=(5.0,),
        length_tolerance_m=0.0,
    )
    with pytest.raises(ValueError):
        ResolvedBaselineSelection(
            (cross,),
            _provenance(
                criteria=target_criteria,
                generated_count=3,
                selected_ids=((0, 1),),
            ),
        )

    range_criteria = _criteria(
        correlations="cross",
        length_mode="ranges",
        length_ranges_m=((5.0, 9.0),),
    )
    with pytest.raises(ValueError):
        ResolvedBaselineSelection(
            (cross,),
            _provenance(
                criteria=range_criteria,
                generated_count=3,
                selected_ids=((0, 1),),
            ),
        )

    azimuth_criteria = _criteria(
        correlations="cross",
        azimuth_ranges_deg=((0.0, 10.0),),
    )
    with pytest.raises(ValueError):
        ResolvedBaselineSelection(
            (cross,),
            _provenance(
                criteria=azimuth_criteria,
                generated_count=3,
                selected_ids=((0, 1),),
            ),
        )


def test_selection_constructor_keeps_auto_exemption_under_active_azimuth_filter():
    criteria = _criteria(
        correlations="auto",
        azimuth_ranges_deg=((80.0, 100.0),),
    )
    provenance = _provenance(
        criteria=criteria,
        azimuth_exempt_auto_count=1,
    )

    assert ResolvedBaselineSelection((_auto(),), provenance).baselines == (_auto(),)


def test_selection_constructor_enforces_nonempty_canonical_order_and_provenance():
    auto0 = _auto(0)
    auto1 = _auto(1)
    provenance = _provenance(
        criteria=_criteria(correlations="auto"),
        generated_count=3,
        after_correlation_count=2,
        after_length_count=2,
        after_azimuth_count=2,
        selected_ids=((0, 0), (1, 1)),
    )
    caller = [auto0, auto1]
    selection = ResolvedBaselineSelection(caller, provenance)
    caller.clear()

    assert selection.baselines == (auto0, auto1)
    assert type(selection.baselines) is tuple
    assert isinstance(hash(selection), int)
    with pytest.raises(FrozenInstanceError):
        selection.provenance = provenance

    with pytest.raises(ValueError):
        ResolvedBaselineSelection(
            (),
            _provenance(
                criteria=_criteria(
                    correlations="cross",
                    azimuth_ranges_deg=((0.0, 10.0),),
                ),
                generated_count=3,
                after_correlation_count=1,
                after_length_count=1,
                after_azimuth_count=0,
                selected_ids=(),
            ),
        )
    with pytest.raises((TypeError, ValueError)):
        ResolvedBaselineSelection((object(),), _provenance())
    with pytest.raises(ValueError):
        ResolvedBaselineSelection((auto1, auto0), provenance)
    with pytest.raises(ValueError):
        ResolvedBaselineSelection((auto0, auto1), _provenance())


def test_nested_models_reject_mutable_subclasses():
    def mutable_setattr(self, name, value):
        object.__setattr__(self, name, value)

    mutable_criteria_type = type(
        "MutableCriteria",
        (BaselineSelectionCriteriaSnapshot,),
        {"__setattr__": mutable_setattr},
    )
    mutable_criteria = mutable_criteria_type("all", None, (), None, (), ())
    with pytest.raises(TypeError):
        _provenance(criteria=mutable_criteria)

    valid = _provenance()
    mutable_provenance_type = type(
        "MutableProvenance",
        (BaselineSelectionProvenance,),
        {"__setattr__": mutable_setattr},
    )
    mutable_provenance = mutable_provenance_type(
        *(getattr(valid, item.name) for item in fields(valid))
    )
    with pytest.raises(TypeError):
        ResolvedBaselineSelection((_auto(),), mutable_provenance)


def test_selection_snapshot_is_exact_fresh_detached_and_json_safe():
    selection = _select(
        _instrument(),
        BaselineSelectionConfig(
            correlations="cross",
            length_filter=LengthTargetsConfig(targets_m=(14.0, 10.0), tolerance_m=0.5),
            azimuth_ranges_deg=(
                AzimuthRangeConfig(start_deg=170.0, end_deg=10.0),
                AzimuthRangeConfig(start_deg=20.0, end_deg=100.0),
            ),
        ),
    )

    first = selection.to_snapshot()
    second = selection.to_snapshot()

    assert first == second
    assert first is not second
    assert first["criteria"] is not second["criteria"]
    assert first["selected_ids"] is not second["selected_ids"]
    assert first == {
        "schema_version": "radiosim.baseline-selection.v1",
        "criteria": {
            "correlations": "cross",
            "length_mode": "targets",
            "length_targets_m": [10.0, 14.0],
            "length_tolerance_m": 0.5,
            "length_ranges_m": [],
            "azimuth_ranges_deg": [[20.0, 100.0], [170.0, 10.0]],
        },
        "generated_count": 6,
        "after_correlation_count": 3,
        "after_length_count": 3,
        "after_azimuth_count": 2,
        "azimuth_exempt_auto_count": 0,
        "selected_ids": [[0, 1], [0, 2]],
    }
    json.dumps(first, allow_nan=False, sort_keys=True)
    first["criteria"]["length_targets_m"].append(999.0)
    first["selected_ids"][0][0] = 999
    assert second == selection.to_snapshot()


def test_direct_snapshot_range_mode_sorts_without_merging():
    snapshot = _criteria(
        correlations="auto",
        length_mode="ranges",
        length_ranges_m=((20, 30), (0, 10), (5, 25)),
        azimuth_ranges_deg=((170, 10), (20, 40)),
    )

    assert snapshot.length_ranges_m == (
        (0.0, 10.0),
        (5.0, 25.0),
        (20.0, 30.0),
    )
    assert snapshot.azimuth_ranges_deg == ((20.0, 40.0), (170.0, 10.0))


def test_no_filter_stage_counts_remain_equal_and_no_auto_exemption_is_recorded():
    selection = _select(
        _instrument(),
        BaselineSelectionConfig(correlations="auto"),
    )

    provenance = selection.provenance
    assert provenance.after_correlation_count == 3
    assert provenance.after_length_count == 3
    assert provenance.after_azimuth_count == 3
    assert provenance.azimuth_exempt_auto_count == 0
