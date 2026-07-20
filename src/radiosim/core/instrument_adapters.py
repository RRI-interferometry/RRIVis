"""Owned runtime state and narrow adapters for canonical instruments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import numpy as np
import numpy.typing as npt

from radiosim.core.instrument import (
    ResolvedAntenna,
    ResolvedBaseline,
    ResolvedBaselineSelection,
    ResolvedInstrument,
)


@dataclass(frozen=True, slots=True)
class ResolvedInstrumentState:
    """One atomically assignable canonical instrument and baseline state."""

    instrument: ResolvedInstrument
    all_baselines: tuple[ResolvedBaseline, ...]
    selection: ResolvedBaselineSelection
    by_number: Mapping[int, ResolvedAntenna] = field(init=False)
    by_name: Mapping[str, ResolvedAntenna] = field(init=False)

    def __post_init__(self) -> None:
        if type(self.instrument) is not ResolvedInstrument:
            raise TypeError("instrument must be a ResolvedInstrument")
        if type(self.all_baselines) is not tuple or any(
            type(item) is not ResolvedBaseline for item in self.all_baselines
        ):
            raise TypeError("all_baselines must be a tuple of ResolvedBaseline values")
        if type(self.selection) is not ResolvedBaselineSelection:
            raise TypeError("selection must be a ResolvedBaselineSelection")
        if self.selection.provenance.instrument_sha256 != (
            self.instrument.provenance.instrument_sha256
        ):
            raise ValueError("selection does not belong to instrument")

        from radiosim.core.baseline_resolution import generate_resolved_baselines

        expected_baselines = generate_resolved_baselines(self.instrument)
        if self.all_baselines != expected_baselines:
            raise ValueError(
                "all_baselines must equal the complete canonical instrument inventory"
            )
        if self.selection.provenance.generated_count != len(expected_baselines):
            raise ValueError(
                "selection generated_count does not match the complete canonical "
                "instrument inventory"
            )
        expected_by_pair = {
            (baseline.ant1.number, baseline.ant2.number): baseline
            for baseline in expected_baselines
        }
        if any(
            expected_by_pair.get((baseline.ant1.number, baseline.ant2.number))
            != baseline
            for baseline in self.selection.baselines
        ):
            raise ValueError(
                "selected baselines must belong to the complete canonical "
                "instrument inventory"
            )

        by_number = {antenna.id.number: antenna for antenna in self.instrument.antennas}
        by_name = {antenna.id.name: antenna for antenna in self.instrument.antennas}
        if len(by_number) != len(self.instrument.antennas):
            raise ValueError("instrument contains duplicate antenna numbers")
        if len(by_name) != len(self.instrument.antennas):
            raise ValueError("instrument contains duplicate antenna names")
        object.__setattr__(self, "all_baselines", tuple(self.all_baselines))
        object.__setattr__(self, "by_number", MappingProxyType(dict(by_number)))
        object.__setattr__(self, "by_name", MappingProxyType(dict(by_name)))

    @property
    def baselines(self) -> tuple[ResolvedBaseline, ...]:
        """Return the exact selected canonical baseline tuple."""
        return self.selection.baselines


class InstrumentAdapterInvariantError(RuntimeError):
    """Canonical state could not be represented without losing identity."""


@dataclass(frozen=True, slots=True)
class SolverInstrumentView:
    """Fresh read-only numeric values consumed by visibility solvers."""

    antenna_numbers: tuple[int, ...]
    antenna_names: tuple[str, ...]
    positions_enu_m: npt.NDArray[np.float64]
    diameters_m: npt.NDArray[np.float64]
    row_index_by_number: Mapping[int, int]
    selected_pairs: tuple[tuple[int, int], ...]
    baseline_vectors_enu_m: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if type(self.antenna_numbers) is not tuple or any(
            type(number) is not int for number in self.antenna_numbers
        ):
            raise TypeError("antenna_numbers must be a tuple of integers")
        if not self.antenna_numbers:
            raise ValueError("antenna_numbers must be nonempty")
        if len(set(self.antenna_numbers)) != len(self.antenna_numbers):
            raise ValueError("antenna_numbers must be unique")
        if type(self.antenna_names) is not tuple or any(
            type(name) is not str for name in self.antenna_names
        ):
            raise TypeError("antenna_names must be a tuple of strings")
        if len(self.antenna_names) != len(self.antenna_numbers):
            raise ValueError("antenna names and numbers must have equal lengths")
        if len(set(self.antenna_names)) != len(self.antenna_names):
            raise ValueError("antenna_names must be unique")

        row_index_value = cast(object, self.row_index_by_number)
        if not isinstance(row_index_value, Mapping):
            raise TypeError("row_index_by_number must be a mapping")
        row_index = cast(Mapping[int, int], row_index_value)
        expected_index = {
            number: index for index, number in enumerate(self.antenna_numbers)
        }
        if dict(row_index) != expected_index:
            raise ValueError("row_index_by_number must exactly index antenna_numbers")

        if type(self.selected_pairs) is not tuple or any(
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(number) is not int for number in pair)
            for pair in self.selected_pairs
        ):
            raise TypeError("selected_pairs must be a tuple of integer pairs")
        if not self.selected_pairs:
            raise ValueError("selected_pairs must be nonempty")
        if len(set(self.selected_pairs)) != len(self.selected_pairs):
            raise ValueError("selected_pairs must be unique")
        if self.selected_pairs != tuple(sorted(self.selected_pairs)):
            raise ValueError("selected_pairs must use canonical stable order")
        for ant1, ant2 in self.selected_pairs:
            if ant1 > ant2:
                raise ValueError("selected_pairs must use canonical numeric order")
            if ant1 not in expected_index or ant2 not in expected_index:
                missing = ant1 if ant1 not in expected_index else ant2
                raise InstrumentAdapterInvariantError(
                    f"selected antenna number {missing} is absent from instrument"
                )

        positions = np.array(
            self.positions_enu_m,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        diameters = np.array(
            self.diameters_m,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        vectors = np.array(
            self.baseline_vectors_enu_m,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if positions.shape != (len(self.antenna_numbers), 3):
            raise ValueError("positions_enu_m must have shape (n_antennas, 3)")
        if diameters.shape != (len(self.antenna_numbers),):
            raise ValueError("diameters_m must have shape (n_antennas,)")
        if vectors.shape != (len(self.selected_pairs), 3):
            raise ValueError("baseline vectors must have shape (n_baselines, 3)")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions_enu_m must contain only finite values")
        if not np.all(np.isfinite(diameters)) or np.any(diameters <= 0.0):
            raise ValueError("diameters_m must contain finite positive values")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("baseline vectors must contain only finite values")
        expected_vectors = np.array(
            [
                positions[expected_index[ant2]] - positions[expected_index[ant1]]
                for ant1, ant2 in self.selected_pairs
            ],
            dtype=np.float64,
            order="C",
        )
        if not np.array_equal(vectors, expected_vectors):
            raise ValueError(
                "baseline vectors must exactly equal position(ant2)-position(ant1)"
            )

        for array in (positions, diameters, vectors):
            array.setflags(write=False)
        object.__setattr__(self, "positions_enu_m", positions)
        object.__setattr__(self, "diameters_m", diameters)
        object.__setattr__(self, "baseline_vectors_enu_m", vectors)
        object.__setattr__(
            self,
            "row_index_by_number",
            MappingProxyType(dict(expected_index)),
        )

    @classmethod
    def from_state(cls, state: ResolvedInstrumentState) -> SolverInstrumentView:
        """Allocate one solver-owned lossless view of canonical state."""
        if type(state) is not ResolvedInstrumentState:
            raise TypeError("state must be a ResolvedInstrumentState")

        antennas = state.instrument.antennas
        baselines = state.selection.baselines
        numbers = tuple(antenna.id.number for antenna in antennas)
        names = tuple(antenna.id.name for antenna in antennas)
        row_index = {number: index for index, number in enumerate(numbers)}
        if len(row_index) != len(numbers):
            raise InstrumentAdapterInvariantError(
                "canonical antenna numbers are not unique"
            )

        positions = np.array(
            [antenna.position_enu_m for antenna in antennas],
            dtype=np.float64,
            order="C",
            copy=True,
        )
        diameters = np.array(
            [antenna.diameter_m for antenna in antennas],
            dtype=np.float64,
            order="C",
            copy=True,
        )
        pairs = tuple(
            (baseline.ant1.number, baseline.ant2.number) for baseline in baselines
        )
        vectors = np.array(
            [baseline.vector_enu_m for baseline in baselines],
            dtype=np.float64,
            order="C",
            copy=True,
        )
        for pair in pairs:
            for number in pair:
                if number not in row_index:
                    raise InstrumentAdapterInvariantError(
                        f"selected antenna number {number} is absent from instrument"
                    )
        return cls(
            antenna_numbers=numbers,
            antenna_names=names,
            positions_enu_m=positions,
            diameters_m=diameters,
            row_index_by_number=MappingProxyType(dict(row_index)),
            selected_pairs=pairs,
            baseline_vectors_enu_m=vectors,
        )

    def row_for_number(self, number: int) -> int:
        """Return an antenna row or raise a focused internal invariant error."""
        try:
            return self.row_index_by_number[number]
        except KeyError as error:
            raise InstrumentAdapterInvariantError(
                f"antenna number {number} is absent from solver instrument view"
            ) from error


__all__ = [
    "InstrumentAdapterInvariantError",
    "ResolvedInstrumentState",
    "SolverInstrumentView",
]
