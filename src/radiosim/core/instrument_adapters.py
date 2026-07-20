"""Owned runtime state and narrow adapters for canonical instruments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

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
        for array in (positions, diameters, vectors):
            array.setflags(write=False)

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
