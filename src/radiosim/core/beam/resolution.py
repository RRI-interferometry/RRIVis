"""Canonical standalone beam-assignment resolution for Tier 3C."""

from __future__ import annotations

from typing import Any, Literal, cast

from radiosim.core.beam.errors import (
    DuplicateBeamAssignmentError,
    IncompleteBeamAssignmentError,
    InvalidBeamGeometryError,
    UnknownBeamAntennaError,
)
from radiosim.core.beam.models import (
    _RESOLVED_BEAMS_INPUT_TYPES,  # pyright: ignore[reportPrivateUsage]
    BeamAssignmentProvenance,
    ResolvedAnalyticBeamChoice,
    ResolvedAnalyticBeamDefinition,
    ResolvedAnalyticBeamsInput,
    ResolvedAperturePhysics,
    ResolvedBeamAssignment,
    ResolvedBeamsInput,
    ResolvedBeamState,
    ResolvedFITSBeamAssignmentInput,
    ResolvedFITSBeamDefinition,
    ResolvedMixedBeamAssignmentInput,
    ResolvedMixedBeamsInput,
    ResolvedPerAntennaFITSBeamsInput,
    ResolvedPointingOffset,
    ResolvedSharedFITSBeamsInput,
    ResolvedSquint,
    ResolvedSquintRecord,
    ResolvedSurfaceError,
    SquintMountType,
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


def _mount_override_map(
    entries: tuple[Any, ...],
    value_field: str,
    *,
    logical_path: str,
    instrument: ResolvedInstrument,
) -> dict[AntennaId, Any]:
    """Resolve one ordered per-antenna mount override list against the array.

    The two Tier 7I blocks reuse the Tier 3C assignment discipline verbatim: an
    unknown reference collects into one ``UnknownBeamAntennaError`` naming every
    bad entry, and a repeated canonical antenna is a
    ``DuplicateBeamAssignmentError`` naming the index that already claimed it.
    Completeness is *not* required -- unlike a beam assignment, an absent entry
    means "take the array-wide default", which is a real and common intent.
    """
    by_number = {antenna.id.number: antenna for antenna in instrument.antennas}
    by_name = {antenna.id.name: antenna for antenna in instrument.antennas}
    unknown_messages: list[str] = []
    resolved: list[tuple[int, ResolvedAntenna, Any]] = []

    for index, entry in enumerate(entries):
        reference = entry.antenna
        _kind, _value = _reference_value(reference)
        if type(reference) is AntennaNumberReference:
            antenna = by_number.get(reference.number)
        else:
            antenna = by_name.get(cast(AntennaNameReference, reference).name)
        if antenna is None:
            unknown_messages.append(
                f"{logical_path}[{index}].antenna="
                f"{_shown_reference(reference)}: no canonical antenna matches; "
                "use an exact Tier 2 number or case-sensitive canonical name."
            )
        else:
            resolved.append((index, antenna, getattr(entry, value_field)))

    if unknown_messages:
        raise UnknownBeamAntennaError("\n".join(unknown_messages))

    first_index: dict[AntennaId, int] = {}
    values: dict[AntennaId, Any] = {}
    for index, antenna, value in resolved:
        canonical_id = antenna.id
        prior = first_index.get(canonical_id)
        if prior is not None:
            raise DuplicateBeamAssignmentError(
                f"{logical_path}[{index}].antenna="
                f"{_shown_reference(entries[index].antenna)}: canonical antenna "
                f"number={canonical_id.number}, name={canonical_id.name!r} was "
                f"already assigned at index {prior}."
            )
        first_index[canonical_id] = index
        values[canonical_id] = value
    return values


def _mount_science(
    config: ResolvedBeamsInput,
    instrument: ResolvedInstrument,
) -> tuple[
    dict[AntennaId, ResolvedPointingOffset | None],
    dict[AntennaId, ResolvedSurfaceError | None],
    dict[AntennaId, ResolvedSquint | None],
]:
    """Return the per-antenna pointing, surface-error, and squint values.

    ``docs/development/sci005_beam_physics_plan.md`` Section 4.1.1: squint
    resolves through the same accepted default-then-override map the two Tier 7I
    blocks use, so an unknown reference is the existing typed
    ``UnknownBeamAntennaError`` and a repeated canonical antenna the existing
    typed ``DuplicateBeamAssignmentError``.  Resolution is also where each
    squinting antenna's mount literal is captured, because the mount belongs to
    the instrument and ``None`` retains its accepted ``fixed`` reading.
    """
    pointing_by_antenna: dict[AntennaId, ResolvedPointingOffset | None] = {}
    surface_by_antenna: dict[AntennaId, ResolvedSurfaceError | None] = {}
    squint_by_antenna: dict[AntennaId, ResolvedSquint | None] = {}

    pointing = config.pointing
    if pointing is not None:
        overrides = _mount_override_map(
            pointing.per_antenna,
            "offset",
            logical_path="beams.pointing.per_antenna",
            instrument=instrument,
        )
        for antenna in instrument.antennas:
            pointing_by_antenna[antenna.id] = overrides.get(
                antenna.id,
                pointing.default,
            )

    surface_error = config.surface_error
    if surface_error is not None:
        overrides = _mount_override_map(
            surface_error.per_antenna,
            "surface_error",
            logical_path="beams.surface_error.per_antenna",
            instrument=instrument,
        )
        for antenna in instrument.antennas:
            surface_by_antenna[antenna.id] = overrides.get(
                antenna.id,
                surface_error.default,
            )

    squint = config.squint
    if squint is not None:
        overrides = _mount_override_map(
            squint.per_antenna,
            "squint",
            logical_path="beams.squint.per_antenna",
            instrument=instrument,
        )
        for antenna in instrument.antennas:
            record = cast(
                ResolvedSquintRecord | None,
                overrides.get(antenna.id, squint.default),
            )
            squint_by_antenna[antenna.id] = (
                None
                if record is None
                else ResolvedSquint(
                    record.convention,
                    record.reference_frequency_hz,
                    record.per_feed_offset_deg_at_reference,
                    record.mechanical_feed_position_angle_deg,
                    record.positive_native_feed,
                    cast(
                        SquintMountType,
                        "fixed" if antenna.mount_type is None else antenna.mount_type,
                    ),
                )
            )

    return pointing_by_antenna, surface_by_antenna, squint_by_antenna


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
    pointing_by_antenna: dict[AntennaId, ResolvedPointingOffset | None],
    surface_by_antenna: dict[AntennaId, ResolvedSurfaceError | None],
    squint_by_antenna: dict[AntennaId, ResolvedSquint | None],
    aperture_physics: ResolvedAperturePhysics | None,
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
                pointing=pointing_by_antenna.get(antenna.id),
                surface_error=surface_by_antenna.get(antenna.id),
                aperture_physics=aperture_physics,
                squint=squint_by_antenna.get(antenna.id),
            )
        )
    return tuple(assignments)


def _uniform_assignments(
    instrument: ResolvedInstrument,
    definition: ResolvedAnalyticBeamDefinition | ResolvedFITSBeamDefinition,
    *,
    source: Literal["analytic_mode", "shared_mode"],
    pointing_by_antenna: dict[AntennaId, ResolvedPointingOffset | None],
    surface_by_antenna: dict[AntennaId, ResolvedSurfaceError | None],
    squint_by_antenna: dict[AntennaId, ResolvedSquint | None],
    aperture_physics: ResolvedAperturePhysics | None,
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
                pointing=pointing_by_antenna.get(antenna.id),
                surface_error=surface_by_antenna.get(antenna.id),
                aperture_physics=aperture_physics,
                squint=squint_by_antenna.get(antenna.id),
            )
        )
    return tuple(assignments)


def _require_representable_support_geometry(
    aperture_physics: ResolvedAperturePhysics | None,
    instrument: ResolvedInstrument,
) -> None:
    """Reject a support leg wider than an assigned antenna's resolved aperture.

    ``docs/development/sci005_beam_physics_plan.md`` Section 3.2 rules that this
    check belongs here rather than to document validation, because per-antenna
    diameters exist only after instrument resolution. The comparison is per
    assigned antenna, so a leg that is too wide for one dish in a heterogeneous
    array is named against exactly that dish. A leg *exactly* as wide as the
    resolved aperture is degenerate but representable and is not rejected: the
    rule is "wider than", and the mask's own boundaries are closed sets.
    """
    if aperture_physics is None or aperture_physics.blockage is None:
        return
    for antenna in instrument.antennas:
        diameter_m = antenna.diameter_m
        for leg in aperture_physics.blockage.support_legs:
            if leg.width_m > diameter_m:
                raise InvalidBeamGeometryError(
                    "beams.aperture_physics.blockage.support_legs: the leg at "
                    f"position_angle_deg={leg.position_angle_deg} has authored "
                    f"width_m={leg.width_m}, which is wider than the resolved "
                    f"aperture diameter {diameter_m} m of canonical antenna "
                    f"number={antenna.id.number}, name={antenna.id.name!r}; a "
                    "support leg cannot be wider than the dish it crosses."
                )


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

    pointing_by_antenna, surface_by_antenna, squint_by_antenna = _mount_science(
        config, instrument
    )
    aperture_physics = config.aperture_physics
    _require_representable_support_geometry(aperture_physics, instrument)

    if type(config) is ResolvedAnalyticBeamsInput:
        assignments = _uniform_assignments(
            instrument,
            config.model,
            source="analytic_mode",
            pointing_by_antenna=pointing_by_antenna,
            surface_by_antenna=surface_by_antenna,
            squint_by_antenna=squint_by_antenna,
            aperture_physics=aperture_physics,
        )
    elif type(config) is ResolvedSharedFITSBeamsInput:
        assignments = _uniform_assignments(
            instrument,
            config.beam,
            source="shared_mode",
            pointing_by_antenna=pointing_by_antenna,
            surface_by_antenna=surface_by_antenna,
            squint_by_antenna=squint_by_antenna,
            aperture_physics=aperture_physics,
        )
    else:
        assignments = _explicit_assignments(
            cast(
                ResolvedPerAntennaFITSBeamsInput | ResolvedMixedBeamsInput,
                config,
            ),
            instrument,
            pointing_by_antenna,
            surface_by_antenna,
            squint_by_antenna,
            aperture_physics,
        )

    unique_definitions = _deduplicated_definitions(assignments)
    return _create_resolved_beam_state(
        mode=config.mode,
        instrument_fingerprint=instrument.provenance.instrument_sha256,
        assignments=assignments,
        unique_definitions=unique_definitions,
    )


__all__ = ["resolve_beam_assignments"]
