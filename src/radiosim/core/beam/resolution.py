"""Canonical standalone beam-assignment resolution for Tier 3C."""

from __future__ import annotations

from typing import Literal, cast

from radiosim.core.beam.errors import (
    DuplicateBeamAssignmentError,
    IncompleteBeamAssignmentError,
    UnknownBeamAntennaError,
)
from radiosim.core.beam.models import (
    _RESOLVED_BEAMS_INPUT_TYPES,  # pyright: ignore[reportPrivateUsage]
    BeamAssignmentProvenance,
    ResolvedAnalyticBeamChoice,
    ResolvedAnalyticBeamDefinition,
    ResolvedAnalyticBeamsInput,
    ResolvedBeamAssignment,
    ResolvedBeamsInput,
    ResolvedBeamState,
    ResolvedFITSBeamAssignmentInput,
    ResolvedFITSBeamDefinition,
    ResolvedMixedBeamAssignmentInput,
    ResolvedMixedBeamsInput,
    ResolvedPerAntennaFITSBeamsInput,
    ResolvedSharedFITSBeamsInput,
    _create_resolved_beam_assignment,  # pyright: ignore[reportPrivateUsage]
    _create_resolved_beam_state,  # pyright: ignore[reportPrivateUsage]
    _deduplicated_definitions,  # pyright: ignore[reportPrivateUsage]
)
from radiosim.core.instrument import AntennaId, ResolvedAntenna, ResolvedInstrument
from radiosim.io.instrument_config import (
    AntennaNameReference,
    AntennaNumberReference,
    AntennaReference,
)

_ExplicitInput = ResolvedFITSBeamAssignmentInput | ResolvedMixedBeamAssignmentInput


def _copy_antenna_id(antenna: ResolvedAntenna) -> AntennaId:
    return AntennaId(antenna.id.number, antenna.id.name)


def _reference_value(
    reference: AntennaReference,
) -> tuple[Literal["number", "name"], int | str]:
    if type(reference) is AntennaNumberReference:
        return "number", reference.number
    if type(reference) is AntennaNameReference:
        return "name", reference.name
    raise TypeError("assignment antenna must be an exact Tier 2 AntennaReference")


def _shown_reference(reference: AntennaReference) -> str:
    kind, value = _reference_value(reference)
    return str(value) if kind == "number" else repr(value)


def _definition_for_explicit_input(
    config: ResolvedPerAntennaFITSBeamsInput | ResolvedMixedBeamsInput,
    item: _ExplicitInput,
) -> ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition:
    if type(item) is ResolvedFITSBeamAssignmentInput:
        return item.beam
    choice = cast(ResolvedMixedBeamAssignmentInput, item).beam
    if type(choice) is ResolvedAnalyticBeamChoice:
        return cast(ResolvedMixedBeamsInput, config).analytic_model
    return cast(ResolvedFITSBeamDefinition, choice)


def _explicit_assignments(
    config: ResolvedPerAntennaFITSBeamsInput | ResolvedMixedBeamsInput,
    instrument: ResolvedInstrument,
) -> tuple[ResolvedBeamAssignment, ...]:
    by_number = {antenna.id.number: antenna for antenna in instrument.antennas}
    by_name = {antenna.id.name: antenna for antenna in instrument.antennas}
    resolved: list[tuple[int, _ExplicitInput, ResolvedAntenna]] = []
    unknown_messages: list[str] = []

    for index, item in enumerate(config.assignments):
        reference = item.antenna
        _kind, _value = _reference_value(reference)
        if type(reference) is AntennaNumberReference:
            antenna = by_number.get(reference.number)
        else:
            antenna = by_name.get(cast(AntennaNameReference, reference).name)
        if antenna is None:
            unknown_messages.append(
                f"beams.assignments[{index}].antenna="
                f"{_shown_reference(reference)}: no canonical antenna matches; "
                "use an exact Tier 2 number or case-sensitive canonical name."
            )
        else:
            resolved.append((index, item, antenna))

    if unknown_messages:
        raise UnknownBeamAntennaError("\n".join(unknown_messages))

    first_index: dict[AntennaId, int] = {}
    resolved_by_number: dict[int, tuple[int, _ExplicitInput, ResolvedAntenna]] = {}
    for index, item, antenna in resolved:
        canonical_id = antenna.id
        prior = first_index.get(canonical_id)
        if prior is not None:
            raise DuplicateBeamAssignmentError(
                f"beams.assignments[{index}].antenna="
                f"{_shown_reference(item.antenna)}: canonical antenna "
                f"number={canonical_id.number}, name={canonical_id.name!r} was "
                f"already assigned at index {prior}."
            )
        first_index[canonical_id] = index
        resolved_by_number[canonical_id.number] = (index, item, antenna)

    missing = tuple(
        antenna
        for antenna in instrument.antennas
        if antenna.id.number not in resolved_by_number
    )
    if missing:
        rendered = ", ".join(
            f"{antenna.id.number}:{antenna.id.name}" for antenna in missing
        )
        raise IncompleteBeamAssignmentError(
            f"beams.assignments: missing canonical antennas [{rendered}]; every "
            "antenna requires one explicit assignment and no default is supported."
        )

    assignments: list[ResolvedBeamAssignment] = []
    for canonical_antenna in instrument.antennas:
        index, item, antenna = resolved_by_number[canonical_antenna.id.number]
        reference_kind, reference_value = _reference_value(item.antenna)
        antenna_id = _copy_antenna_id(antenna)
        provenance = BeamAssignmentProvenance(
            source="explicit_assignment",
            input_index=index,
            authored_reference_kind=reference_kind,
            authored_reference_value=reference_value,
            canonical_antenna=antenna_id,
        )
        assignments.append(
            _create_resolved_beam_assignment(
                antenna_id=antenna_id,
                antenna_diameter_m=antenna.diameter_m,
                definition=_definition_for_explicit_input(config, item),
                provenance=provenance,
            )
        )
    return tuple(assignments)


def _uniform_assignments(
    instrument: ResolvedInstrument,
    definition: ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition,
    *,
    source: Literal["analytic_mode", "shared_mode"],
) -> tuple[ResolvedBeamAssignment, ...]:
    assignments: list[ResolvedBeamAssignment] = []
    for antenna in instrument.antennas:
        antenna_id = _copy_antenna_id(antenna)
        provenance = BeamAssignmentProvenance(
            source=source,
            input_index=None,
            authored_reference_kind=None,
            authored_reference_value=None,
            canonical_antenna=antenna_id,
        )
        assignments.append(
            _create_resolved_beam_assignment(
                antenna_id=antenna_id,
                antenna_diameter_m=antenna.diameter_m,
                definition=definition,
                provenance=provenance,
            )
        )
    return tuple(assignments)


def resolve_beam_assignments(
    config: ResolvedBeamsInput,
    instrument: ResolvedInstrument,
) -> ResolvedBeamState:
    """Resolve one exact beam input against one canonical Tier 2 instrument.

    This function performs identity lookup and immutable state construction only. It
    does not resolve paths, open FITS files, import beam dependencies, or initialize
    any runtime service.
    """
    if type(config) not in _RESOLVED_BEAMS_INPUT_TYPES:
        raise TypeError("config must be an exact supported ResolvedBeamsInput")
    if type(instrument) is not ResolvedInstrument:
        raise TypeError("instrument must be an exact ResolvedInstrument")

    if type(config) is ResolvedAnalyticBeamsInput:
        assignments = _uniform_assignments(
            instrument,
            config.model,
            source="analytic_mode",
        )
    elif type(config) is ResolvedSharedFITSBeamsInput:
        assignments = _uniform_assignments(
            instrument,
            config.beam,
            source="shared_mode",
        )
    else:
        assignments = _explicit_assignments(
            cast(
                ResolvedPerAntennaFITSBeamsInput | ResolvedMixedBeamsInput,
                config,
            ),
            instrument,
        )

    unique_definitions = _deduplicated_definitions(assignments)
    return _create_resolved_beam_state(
        mode=config.mode,
        instrument_fingerprint=instrument.provenance.instrument_sha256,
        assignments=assignments,
        unique_definitions=unique_definitions,
    )


__all__ = ["resolve_beam_assignments"]
