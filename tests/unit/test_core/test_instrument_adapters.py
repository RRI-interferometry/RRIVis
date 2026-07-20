"""Contracts for owned canonical instrument runtime adapters."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from radiosim.api import Simulator
from radiosim.core.instrument_adapters import (
    InstrumentAdapterInvariantError,
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
