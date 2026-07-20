"""Contracts for owned canonical instrument runtime adapters."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from radiosim.api import Simulator
from radiosim.core.instrument import (
    BaselineSelectionCriteriaSnapshot,
    ResolvedBaselineSelection,
)
from radiosim.core.instrument_adapters import (
    InstrumentAdapterInvariantError,
    ResolvedInstrumentState,
    SolverInstrumentView,
)
from tests.fixtures.configs import valid_config_mapping


def _state(tmp_path):
    data = valid_config_mapping(
        tmp_path,
        baseline_selection={"correlations": "cross"},
    )
    lines = (tmp_path / "antennas.txt").read_text().splitlines()
    lines[-1] = lines[-1].removesuffix("14.0") + "25.0"
    (tmp_path / "antennas.txt").write_text("\n".join(lines) + "\n")
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    return simulator._instrument_state


def test_resolved_state_owns_exact_immutable_indexes(tmp_path):
    state = _state(tmp_path)

    assert isinstance(state.by_number, MappingProxyType)
    assert isinstance(state.by_name, MappingProxyType)
    assert state.by_number[0] is state.instrument.antennas[0]
    assert state.by_name["ANT1"] is state.instrument.antennas[1]
    with pytest.raises(TypeError):
        state.by_number[2] = state.instrument.antennas[0]


@pytest.mark.parametrize("inventory_case", ["empty", "missing", "reordered"])
def test_resolved_state_rejects_noncanonical_complete_inventory(
    tmp_path,
    inventory_case,
):
    state = _state(tmp_path)
    if inventory_case == "empty":
        all_baselines = ()
    elif inventory_case == "missing":
        all_baselines = tuple(
            baseline
            for baseline in state.all_baselines
            if (baseline.ant1.number, baseline.ant2.number) != (0, 1)
        )
    else:
        all_baselines = tuple(reversed(state.all_baselines))

    with pytest.raises(ValueError, match="complete canonical instrument inventory"):
        ResolvedInstrumentState(
            instrument=state.instrument,
            all_baselines=all_baselines,
            selection=state.selection,
        )


def test_resolved_state_rejects_selection_count_for_another_inventory(tmp_path):
    state = _state(tmp_path)
    criteria = BaselineSelectionCriteriaSnapshot(
        correlations="cross",
        length_mode="ranges",
        length_targets_m=(),
        length_tolerance_m=None,
        length_ranges_m=((0.0, 100.0),),
        azimuth_ranges_deg=(),
    )
    provenance = replace(
        state.selection.provenance,
        criteria=criteria,
        generated_count=6,
        after_correlation_count=3,
        after_length_count=1,
        after_azimuth_count=1,
    )
    selection = ResolvedBaselineSelection(
        baselines=state.selection.baselines,
        provenance=provenance,
    )

    with pytest.raises(ValueError, match="generated_count"):
        ResolvedInstrumentState(
            instrument=state.instrument,
            all_baselines=state.all_baselines,
            selection=selection,
        )


def test_solver_view_is_fresh_read_only_float64_and_c_contiguous(tmp_path):
    state = _state(tmp_path)

    first = SolverInstrumentView.from_state(state)
    second = SolverInstrumentView.from_state(state)

    assert first.antenna_numbers == (0, 1)
    assert first.antenna_names == ("ANT0", "ANT1")
    assert first.selected_pairs == ((0, 1),)
    np.testing.assert_array_equal(first.positions_enu_m, [[0, 0, 0], [14, 0, 0]])
    np.testing.assert_array_equal(first.diameters_m, [14, 25])
    np.testing.assert_array_equal(first.baseline_vectors_enu_m, [[14, 0, 0]])

    for array in (
        first.positions_enu_m,
        first.diameters_m,
        first.baseline_vectors_enu_m,
    ):
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
        assert array.flags.owndata
        assert array.flags.writeable is False
    assert not np.shares_memory(first.positions_enu_m, second.positions_enu_m)
    assert not np.shares_memory(first.diameters_m, second.diameters_m)
    assert not np.shares_memory(
        first.baseline_vectors_enu_m,
        second.baseline_vectors_enu_m,
    )


def test_solver_view_missing_identity_is_an_invariant_error(tmp_path):
    view = SolverInstrumentView.from_state(_state(tmp_path))

    with pytest.raises(
        InstrumentAdapterInvariantError,
        match="antenna number 999 is absent",
    ):
        view.row_for_number(999)


def test_solver_view_direct_construction_copy_owns_and_freezes_arrays():
    positions = np.array([[0.0, 0.0, 0.0], [14.0, 0.0, 0.0]])
    diameters = np.array([12.0, 25.0])
    vectors = np.array([[14.0, 0.0, 0.0]])
    row_index = {0: 0, 1: 1}

    view = SolverInstrumentView(
        antenna_numbers=(0, 1),
        antenna_names=("ANT0", "ANT1"),
        positions_enu_m=positions,
        diameters_m=diameters,
        row_index_by_number=row_index,
        selected_pairs=((0, 1),),
        baseline_vectors_enu_m=vectors,
    )
    positions[0, 0] = 99.0
    diameters[0] = 99.0
    vectors[0, 0] = 99.0
    row_index[0] = 1

    np.testing.assert_array_equal(view.positions_enu_m, [[0, 0, 0], [14, 0, 0]])
    np.testing.assert_array_equal(view.diameters_m, [12, 25])
    np.testing.assert_array_equal(view.baseline_vectors_enu_m, [[14, 0, 0]])
    assert view.row_index_by_number[0] == 0
    for array in (
        view.positions_enu_m,
        view.diameters_m,
        view.baseline_vectors_enu_m,
    ):
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
        assert array.flags.owndata
        assert array.flags.writeable is False


def test_solver_view_direct_construction_rejects_inconsistent_geometry():
    with pytest.raises(ValueError, match="baseline vectors"):
        SolverInstrumentView(
            antenna_numbers=(0, 1),
            antenna_names=("ANT0", "ANT1"),
            positions_enu_m=np.array([[0.0, 0.0, 0.0], [14.0, 0.0, 0.0]]),
            diameters_m=np.array([12.0, 25.0]),
            row_index_by_number={0: 0, 1: 1},
            selected_pairs=((0, 1),),
            baseline_vectors_enu_m=np.array([[13.0, 0.0, 0.0]]),
        )
