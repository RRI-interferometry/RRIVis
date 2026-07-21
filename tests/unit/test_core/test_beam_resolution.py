"""Tier 3C immutable canonical beam-assignment resolution contract tests."""

from __future__ import annotations

import builtins
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from radiosim.core.instrument import (
    AntennaFieldSource,
    AntennaId,
    AntennaProvenance,
    ResolvedAntenna,
    ResolvedEarthLocation,
    ResolvedInstrument,
    _create_resolved_instrument,
)
from radiosim.io.instrument_config import AntennaNameReference, AntennaNumberReference

ASSIGNMENT_FIXED_SHA256 = (
    "407b9df278595aebfa1aae895558a73385b9bd7f9f49dbaaf8e00c6bdd480f3c"
)
STATE_FIXED_SHA256 = "9e158b191471edf730921f3641d2fe1eb2d8d902775be1665ee1590f01d083eb"


def _models():
    return importlib.import_module("radiosim.core.beam.models")


def _resolution():
    return importlib.import_module("radiosim.core.beam.resolution")


def _errors():
    return importlib.import_module("radiosim.core.beam.errors")


def _provenance(number: int) -> AntennaProvenance:
    return AntennaProvenance(
        identity_source=AntennaFieldSource.LAYOUT_FILE,
        position_source=AntennaFieldSource.LAYOUT_FILE,
        diameter_source=AntennaFieldSource.LAYOUT_FILE,
        source_diameter_m=float(10 + number),
        mount_source=None,
        beam_id_source=AntennaFieldSource.LAYOUT_FILE,
        source_record=f"row:{number}",
    )


def _instrument(
    entries=((0, "ANT0", 10.0, "shared"), (1, "ANT1", 11.0, "shared")),
) -> ResolvedInstrument:
    antennas = tuple(
        ResolvedAntenna(
            id=AntennaId(number, name),
            position_enu_m=(float(number), 0.0, 0.0),
            diameter_m=diameter,
            mount_type=None,
            beam_id=beam_id,
            provenance=_provenance(number),
        )
        for number, name, diameter, beam_id in entries
    )
    location = ResolvedEarthLocation(
        longitude_deg=0.0,
        latitude_deg=0.0,
        height_m=0.0,
        itrs_xyz_m=(1.0, 2.0, 3.0),
        source=AntennaFieldSource.EXPLICIT_CONFIG,
        reference="fixture",
    )
    return _create_resolved_instrument(
        name="Array",
        location=location,
        antennas=antennas,
        source_kind="fixture",
        source_reference="fixture",
        source_format="radiosim",
        registry_policy=None,
        telescope_name_source=AntennaFieldSource.EXPLICIT_CONFIG,
        location_source=AntennaFieldSource.EXPLICIT_CONFIG,
        source_location_itrs_xyz_m=None,
        location_separation_m=None,
        pyuvdata_version=None,
        source_sha256=None,
    )


def _analytic_definition(kind="circular_aperture", **values):
    models = _models()
    if kind == "circular_aperture":
        model = models.ResolvedCircularApertureBeamModel(
            "circular_aperture", models.ResolvedGaussianTaper("gaussian", 10.0)
        )
    elif kind == "rectangular_aperture":
        model = models.ResolvedRectangularApertureBeamModel(
            "rectangular_aperture",
            values.get("north_length_m", 14.0),
            values.get("east_length_m", 12.0),
        )
    elif kind == "elliptical_aperture":
        model = models.ResolvedEllipticalApertureBeamModel(
            "elliptical_aperture",
            values.get("north_diameter_m", 14.0),
            values.get("east_diameter_m", 12.0),
        )
    elif kind == "analytical_illumination":
        model = models.ResolvedAnalyticalIlluminationBeamModel(
            "analytical_illumination",
            models.ResolvedCorrugatedHornIllumination("corrugated_horn", 0.4, 1.15),
            models.ResolvedDerivedGaussianTaper("gaussian"),
            models.ResolvedPrimeFocusReflector("prime_focus"),
        )
    elif kind == "numerical_illumination":
        model = models.ResolvedNumericalIlluminationBeamModel(
            "numerical_illumination",
            models.ResolvedOpenWaveguideIllumination("open_waveguide", 0.4, 0.7),
            models.ResolvedPrimeFocusReflector("prime_focus"),
            256,
        )
    else:  # pragma: no cover - test helper guard
        raise AssertionError(kind)
    return models.ResolvedAnalyticBeamDefinition(
        "analytic",
        model,
        models._definition_fingerprint("analytic", model),
    )


def _fits_definition(
    path: Path,
    *,
    provenance_key="beams.beam.path",
    frequency_interpolation="cubic",
):
    models = _models()
    normalized = path.resolve(strict=False)
    payload = {
        "path": normalized,
        "normalization": "peak",
        "angular_interpolation": "bilinear",
        "frequency_interpolation": frequency_interpolation,
    }
    return models.ResolvedFITSBeamDefinition(
        "fits",
        normalized,
        "peak",
        "bilinear",
        frequency_interpolation,
        provenance_key,
        models._definition_fingerprint("fits", payload),
    )


def _analytic_input(definition=None):
    models = _models()
    return models.ResolvedAnalyticBeamsInput(
        "analytic", definition or _analytic_definition()
    )


def _per_fits_input(items):
    models = _models()
    return models.ResolvedPerAntennaFITSBeamsInput(
        "per_antenna_fits",
        tuple(
            models.ResolvedFITSBeamAssignmentInput(reference, definition)
            for reference, definition in items
        ),
    )


def _mixed_input(items, definition=None):
    models = _models()
    return models.ResolvedMixedBeamsInput(
        "mixed",
        definition or _analytic_definition(),
        tuple(
            models.ResolvedMixedBeamAssignmentInput(reference, choice)
            for reference, choice in items
        ),
    )


def test_complete_public_error_hierarchy_and_exports_are_exact():
    errors = _errors()
    beam = importlib.import_module("radiosim.core.beam")
    core = importlib.import_module("radiosim.core")
    root = importlib.import_module("radiosim")
    expected = {
        "BeamError": RuntimeError,
        "BeamAssignmentError": errors.BeamError,
        "UnknownBeamAntennaError": errors.BeamAssignmentError,
        "DuplicateBeamAssignmentError": errors.BeamAssignmentError,
        "IncompleteBeamAssignmentError": errors.BeamAssignmentError,
        "InconsistentBeamAssignmentError": errors.BeamAssignmentError,
        "BeamLoadError": errors.BeamError,
        "BeamDependencyError": errors.BeamLoadError,
        "BeamFileReadError": errors.BeamLoadError,
        "BeamFileChangedError": errors.BeamLoadError,
        "UnsupportedBeamMetadataError": errors.BeamLoadError,
        "UnsupportedBeamTypeError": errors.UnsupportedBeamMetadataError,
        "UnsupportedBeamFeedError": errors.UnsupportedBeamMetadataError,
        "UnsupportedBeamBasisError": errors.UnsupportedBeamMetadataError,
        "UnsupportedBeamCoordinateError": errors.UnsupportedBeamMetadataError,
        "BeamNormalizationError": errors.BeamLoadError,
        "UnsupportedBeamPrecisionError": errors.BeamLoadError,
        "BeamSamplingDerivationError": errors.BeamLoadError,
        "BeamEvaluationError": errors.BeamError,
        "BeamFrequencyDomainError": errors.BeamEvaluationError,
        "BeamAngularDomainError": errors.BeamEvaluationError,
        "NonFiniteBeamResponseError": errors.BeamEvaluationError,
        "BeamDisplayNormalizationError": errors.BeamEvaluationError,
    }

    assert tuple(errors.__all__) == tuple(expected)
    for name, direct_base in expected.items():
        value = getattr(errors, name)
        assert value.__bases__ == (direct_base,)
        assert getattr(beam, name) is value
        assert getattr(core, name) is value
        assert name in beam.__all__
        assert name in core.__all__
        assert not hasattr(root, name)


def test_resolver_and_assignment_models_have_public_identity_only_at_core_boundaries():
    models = _models()
    resolution = _resolution()
    beam = importlib.import_module("radiosim.core.beam")
    core = importlib.import_module("radiosim.core")
    root = importlib.import_module("radiosim")

    for name in (
        "BeamAssignmentProvenance",
        "ResolvedBeamAssignment",
        "ResolvedBeamState",
    ):
        assert getattr(beam, name) is getattr(models, name)
        assert getattr(core, name) is getattr(models, name)
        assert not hasattr(root, name)
    assert beam.resolve_beam_assignments is resolution.resolve_beam_assignments
    assert core.resolve_beam_assignments is resolution.resolve_beam_assignments
    assert not hasattr(root, "resolve_beam_assignments")


@pytest.mark.parametrize("bad", [{}, object(), None])
def test_resolver_rejects_wrong_config_type(bad):
    with pytest.raises(TypeError, match="config"):
        _resolution().resolve_beam_assignments(bad, _instrument())


@pytest.mark.parametrize("bad", [{}, object(), None])
def test_resolver_rejects_wrong_instrument_type(bad):
    with pytest.raises(TypeError, match="instrument"):
        _resolution().resolve_beam_assignments(_analytic_input(), bad)


def test_resolver_rejects_hostile_instrument_subclass():
    instrument = _instrument()

    class HostileInstrument(ResolvedInstrument):
        pass

    hostile = HostileInstrument(
        instrument.name,
        instrument.location,
        instrument.antennas,
        instrument.provenance,
    )
    with pytest.raises(TypeError, match="instrument"):
        _resolution().resolve_beam_assignments(_analytic_input(), hostile)


def test_analytic_mode_assigns_canonical_order_heterogeneous_diameters():
    instrument = _instrument(((5, "ANT5", 25.0, 7), (2, "ANT2", 12.0, 7)))
    config = _analytic_input()

    state = _resolution().resolve_beam_assignments(config, instrument)

    assert state.mode == "analytic"
    assert tuple(item.antenna_id.number for item in state.assignments) == (2, 5)
    assert tuple(item.antenna_diameter_m for item in state.assignments) == (12.0, 25.0)
    assert all(item.definition is config.model for item in state.assignments)
    assert all(item.provenance.source == "analytic_mode" for item in state.assignments)
    assert all(item.provenance.input_index is None for item in state.assignments)
    assert state.unique_definitions == (config.model,)


def test_shared_fits_assigns_without_opening_and_diameter_is_scientifically_inert(
    tmp_path: Path, monkeypatch
):
    models = _models()
    definition = _fits_definition(tmp_path / "does-not-exist.beamfits")
    config = models.ResolvedSharedFITSBeamsInput("shared_fits", definition)
    first_instrument = _instrument(
        ((0, "ANT0", 10.0, "same"), (1, "ANT1", 11.0, "same"))
    )
    second_instrument = _instrument(
        ((0, "ANT0", 20.0, "same"), (1, "ANT1", 21.0, "same"))
    )
    monkeypatch.setattr(
        builtins,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("FITS opened")),
    )

    first = _resolution().resolve_beam_assignments(config, first_instrument)
    second = _resolution().resolve_beam_assignments(config, second_instrument)

    assert first.mode == "shared_fits"
    assert first.unique_definitions == (definition,)
    assert all(item.provenance.source == "shared_mode" for item in first.assignments)
    assert tuple(item.antenna_diameter_m for item in first.assignments) == (10.0, 11.0)
    assert [item.assignment_fingerprint for item in first.assignments] == [
        item.assignment_fingerprint for item in second.assignments
    ]


def test_per_antenna_number_and_case_sensitive_name_lookup_reorders_canonically(
    tmp_path: Path,
):
    first = _fits_definition(tmp_path / "first.beamfits", provenance_key="first")
    second = _fits_definition(tmp_path / "second.beamfits", provenance_key="second")
    config = _per_fits_input(
        (
            (AntennaNameReference(name="ANT1"), second),
            (AntennaNumberReference(number=0), first),
        )
    )

    state = _resolution().resolve_beam_assignments(config, _instrument())

    assert tuple(item.antenna_id.number for item in state.assignments) == (0, 1)
    assert tuple(item.definition for item in state.assignments) == (first, second)
    assert tuple(item.provenance.input_index for item in state.assignments) == (1, 0)
    assert state.assignments[0].provenance.authored_reference_kind == "number"
    assert state.assignments[1].provenance.authored_reference_kind == "name"
    assert state.unique_definitions == (first, second)


@pytest.mark.parametrize(
    "reference",
    [AntennaNumberReference(number=99), AntennaNameReference(name="missing")],
)
def test_unknown_number_or_name_uses_typed_fixed_error(tmp_path: Path, reference):
    errors = _errors()
    config = _per_fits_input(((reference, _fits_definition(tmp_path / "a")),))

    with pytest.raises(errors.UnknownBeamAntennaError) as raised:
        _resolution().resolve_beam_assignments(config, _instrument())

    message = str(raised.value)
    assert "beams.assignments[0].antenna=" in message
    assert "no canonical antenna matches" in message
    assert "case-sensitive canonical name" in message


def test_multiple_unknowns_are_aggregated_in_authored_order(tmp_path: Path):
    errors = _errors()
    definition = _fits_definition(tmp_path / "a")
    config = _per_fits_input(
        (
            (AntennaNameReference(name="first"), definition),
            (AntennaNumberReference(number=99), definition),
            (AntennaNameReference(name="third"), definition),
        )
    )

    with pytest.raises(errors.UnknownBeamAntennaError) as raised:
        _resolution().resolve_beam_assignments(config, _instrument())

    message = str(raised.value)
    assert message.count("no canonical antenna matches") == 3
    assert message.index("assignments[0]") < message.index("assignments[1]")
    assert message.index("assignments[1]") < message.index("assignments[2]")


@pytest.mark.parametrize(
    "reference",
    [AntennaNameReference(name="ant0"), AntennaNameReference(name="0")],
)
def test_case_only_and_numeric_looking_names_are_not_guessed(tmp_path: Path, reference):
    config = _per_fits_input(((reference, _fits_definition(tmp_path / "a.beamfits")),))

    with pytest.raises(_errors().UnknownBeamAntennaError):
        _resolution().resolve_beam_assignments(config, _instrument())


@pytest.mark.parametrize(
    "references",
    [
        (AntennaNumberReference(number=0), AntennaNumberReference(number=0)),
        (AntennaNameReference(name="ANT0"), AntennaNameReference(name="ANT0")),
        (AntennaNumberReference(number=0), AntennaNameReference(name="ANT0")),
    ],
)
def test_duplicate_number_name_and_mixed_references_fail_on_later_index(
    tmp_path: Path, references
):
    definition = _fits_definition(tmp_path / "a")
    config = _per_fits_input(tuple((item, definition) for item in references))

    with pytest.raises(_errors().DuplicateBeamAssignmentError) as raised:
        _resolution().resolve_beam_assignments(config, _instrument())

    assert "beams.assignments[1].antenna=" in str(raised.value)
    assert "number=0, name='ANT0' was already assigned at index 0" in str(raised.value)


def test_unknown_collection_precedes_duplicate_detection(tmp_path: Path):
    definition = _fits_definition(tmp_path / "a")
    config = _per_fits_input(
        (
            (AntennaNumberReference(number=0), definition),
            (AntennaNumberReference(number=0), definition),
            (AntennaNameReference(name="missing"), definition),
        )
    )

    with pytest.raises(_errors().UnknownBeamAntennaError) as raised:
        _resolution().resolve_beam_assignments(config, _instrument())

    assert "assignments[2]" in str(raised.value)
    assert "already assigned" not in str(raised.value)


def test_missing_antennas_are_complete_and_in_canonical_order(tmp_path: Path):
    instrument = _instrument(
        ((5, "ANT5", 10.0, 1), (1, "ANT1", 10.0, 1), (3, "ANT3", 10.0, 1))
    )
    config = _per_fits_input(
        ((AntennaNumberReference(number=3), _fits_definition(tmp_path / "a")),)
    )

    with pytest.raises(_errors().IncompleteBeamAssignmentError) as raised:
        _resolution().resolve_beam_assignments(config, instrument)

    assert str(raised.value) == (
        "beams.assignments: missing canonical antennas [1:ANT1, 5:ANT5]; every "
        "antenna requires one explicit assignment and no default is supported."
    )


def test_mixed_mode_supports_analytic_fits_and_all_analytic_explicit_coverage(
    tmp_path: Path,
):
    models = _models()
    definition = _analytic_definition()
    fits = _fits_definition(tmp_path / "a")
    mixed = _mixed_input(
        (
            (
                AntennaNumberReference(number=0),
                models.ResolvedAnalyticBeamChoice("analytic"),
            ),
            (AntennaNameReference(name="ANT1"), fits),
        ),
        definition,
    )
    all_analytic = _mixed_input(
        (
            (
                AntennaNameReference(name="ANT1"),
                models.ResolvedAnalyticBeamChoice("analytic"),
            ),
            (
                AntennaNameReference(name="ANT0"),
                models.ResolvedAnalyticBeamChoice("analytic"),
            ),
        ),
        definition,
    )

    mixed_state = _resolution().resolve_beam_assignments(mixed, _instrument())
    analytic_state = _resolution().resolve_beam_assignments(all_analytic, _instrument())

    assert tuple(type(item.definition) for item in mixed_state.assignments) == (
        models.ResolvedAnalyticBeamDefinition,
        models.ResolvedFITSBeamDefinition,
    )
    assert mixed_state.unique_definitions == (definition, fits)
    assert analytic_state.unique_definitions == (definition,)
    assert all(
        item.provenance.source == "explicit_assignment"
        for item in analytic_state.assignments
    )


def test_repeated_definition_dedup_retains_first_canonical_definition_object(
    tmp_path: Path,
):
    first = _fits_definition(tmp_path / "same", provenance_key="authored[0]")
    equal_science = _fits_definition(tmp_path / "same", provenance_key="authored[1]")
    assert first.definition_fingerprint == equal_science.definition_fingerprint
    assert first != equal_science
    config = _per_fits_input(
        (
            (AntennaNumberReference(number=1), equal_science),
            (AntennaNumberReference(number=0), first),
        )
    )

    state = _resolution().resolve_beam_assignments(config, _instrument())

    assert state.unique_definitions == (first,)
    assert state.unique_definitions[0] is state.assignments[0].definition
    assert state.assignments[1].definition is equal_science


def test_unique_definition_order_is_first_canonical_assignment_not_authored_order(
    tmp_path: Path,
):
    first = _fits_definition(tmp_path / "first")
    second = _fits_definition(tmp_path / "second")
    config = _per_fits_input(
        (
            (AntennaNumberReference(number=1), second),
            (AntennaNumberReference(number=0), first),
        )
    )

    state = _resolution().resolve_beam_assignments(config, _instrument())

    assert state.unique_definitions == (first, second)


def test_authored_order_and_reference_form_do_not_change_scientific_state(
    tmp_path: Path,
):
    first = _fits_definition(tmp_path / "first")
    second = _fits_definition(tmp_path / "second")
    numbered = _per_fits_input(
        (
            (AntennaNumberReference(number=0), first),
            (AntennaNumberReference(number=1), second),
        )
    )
    named_reversed = _per_fits_input(
        (
            (AntennaNameReference(name="ANT1"), second),
            (AntennaNameReference(name="ANT0"), first),
        )
    )

    left = _resolution().resolve_beam_assignments(numbered, _instrument())
    right = _resolution().resolve_beam_assignments(named_reversed, _instrument())

    assert [item.assignment_fingerprint for item in left.assignments] == [
        item.assignment_fingerprint for item in right.assignments
    ]
    assert left.state_fingerprint == right.state_fingerprint
    assert left.to_snapshot() != right.to_snapshot()


@pytest.mark.parametrize(
    "kind",
    ["circular_aperture", "analytical_illumination", "numerical_illumination"],
)
def test_active_circular_dimensions_use_each_canonical_antenna_diameter(kind):
    state = _resolution().resolve_beam_assignments(
        _analytic_input(_analytic_definition(kind)), _instrument()
    )

    assert state.assignments[0].assignment_fingerprint != (
        state.assignments[1].assignment_fingerprint
    )
    assert tuple(item.antenna_diameter_m for item in state.assignments) == (10.0, 11.0)


@pytest.mark.parametrize("kind", ["rectangular_aperture", "elliptical_aperture"])
def test_configured_two_axis_dimensions_make_instrument_diameter_inert(kind):
    definition = _analytic_definition(kind)
    first = _resolution().resolve_beam_assignments(
        _analytic_input(definition),
        _instrument(((0, "ANT0", 10.0, 1), (1, "ANT1", 11.0, 1))),
    )
    second = _resolution().resolve_beam_assignments(
        _analytic_input(definition),
        _instrument(((0, "ANT0", 20.0, 1), (1, "ANT1", 21.0, 1))),
    )

    assert [item.assignment_fingerprint for item in first.assignments] == [
        item.assignment_fingerprint for item in second.assignments
    ]


def test_one_ulp_active_dimension_and_fits_preload_options_change_identity(
    tmp_path: Path,
):
    base = _resolution().resolve_beam_assignments(_analytic_input(), _instrument())
    changed_instrument = _instrument(
        (
            (0, "ANT0", 10.0 + math.ulp(10.0), "shared"),
            (1, "ANT1", 11.0, "shared"),
        )
    )
    changed = _resolution().resolve_beam_assignments(
        _analytic_input(), changed_instrument
    )
    assert base.assignments[0].assignment_fingerprint != (
        changed.assignments[0].assignment_fingerprint
    )

    models = _models()
    cubic = _fits_definition(tmp_path / "same", frequency_interpolation="cubic")
    linear = _fits_definition(tmp_path / "same", frequency_interpolation="linear")
    other_path = _fits_definition(tmp_path / "other", frequency_interpolation="cubic")
    states = [
        _resolution().resolve_beam_assignments(
            models.ResolvedSharedFITSBeamsInput("shared_fits", item), _instrument()
        )
        for item in (cubic, linear, other_path)
    ]
    assert len({state.assignments[0].assignment_fingerprint for state in states}) == 3


def test_native_beam_id_is_inert_for_lookup_coverage_and_assignment():
    models = _models()
    config = _analytic_input()
    duplicated = _instrument(((0, "ANT0", 10.0, "same"), (1, "ANT1", 11.0, "same")))
    distinct = _instrument(((0, "ANT0", 10.0, "one"), (1, "ANT1", 11.0, "two")))

    left = _resolution().resolve_beam_assignments(config, duplicated)
    right = _resolution().resolve_beam_assignments(config, distinct)

    assert [item.assignment_fingerprint for item in left.assignments] == [
        item.assignment_fingerprint for item in right.assignments
    ]
    assert all(item.definition is config.model for item in left.assignments)

    incomplete = _per_fits_input(
        ((AntennaNumberReference(number=0), _fits_definition(Path("/tmp/a"))),)
    )
    with pytest.raises(_errors().IncompleteBeamAssignmentError):
        _resolution().resolve_beam_assignments(incomplete, duplicated)

    mixed = _mixed_input(
        (
            (
                AntennaNumberReference(number=0),
                models.ResolvedAnalyticBeamChoice("analytic"),
            ),
            (
                AntennaNumberReference(number=1),
                models.ResolvedAnalyticBeamChoice("analytic"),
            ),
        )
    )
    assert (
        len(_resolution().resolve_beam_assignments(mixed, duplicated).assignments) == 2
    )


def test_repeated_calls_are_equal_independently_owned_and_inputs_unchanged(tmp_path):
    definition = _fits_definition(tmp_path / "a")
    config = _per_fits_input(
        (
            (AntennaNumberReference(number=0), definition),
            (AntennaNumberReference(number=1), definition),
        )
    )
    instrument = _instrument()
    config_before = config.to_snapshot()
    instrument_before = instrument.to_snapshot()

    first = _resolution().resolve_beam_assignments(config, instrument)
    second = _resolution().resolve_beam_assignments(config, instrument)

    assert first == second
    assert first is not second
    assert first.assignments is not second.assignments
    assert first.unique_definitions is not second.unique_definitions
    assert all(
        a is not b for a, b in zip(first.assignments, second.assignments, strict=False)
    )
    assert all(
        a.provenance is not b.provenance
        for a, b in zip(first.assignments, second.assignments, strict=False)
    )
    assert config.to_snapshot() == config_before
    assert instrument.to_snapshot() == instrument_before


def test_fixed_assignment_and_state_fingerprints_match_independent_payload():
    models = _models()
    definition = _analytic_definition()
    antenna = AntennaId(0, "ANT0")
    provenance = models.BeamAssignmentProvenance(
        "analytic_mode", None, None, None, antenna
    )
    assignment = models._create_resolved_beam_assignment(
        antenna_id=antenna,
        antenna_diameter_m=10.0,
        definition=definition,
        provenance=provenance,
    )
    state = models._create_resolved_beam_state(
        mode="analytic",
        instrument_fingerprint="1" * 64,
        assignments=(assignment,),
        unique_definitions=(definition,),
    )

    independent_assignment_payload = {
        "schema_version": "tier3-beam-v1",
        "kind": "resolved_beam_assignment",
        "canonical_antenna": {"number": 0, "name": "ANT0"},
        "definition_fingerprint": (
            "f79b164f5ae8b08fcc728f5f6a25bd14e5566312230eb668c14b7f5765b628d4"
        ),
        "effective_dimensions": {
            "kind": "circular",
            "diameter_m": float.hex(10.0).lower(),
        },
    }
    independent_state_payload = {
        "schema_version": "tier3-beam-v1",
        "kind": "resolved_beam_state",
        "mode": "analytic",
        "instrument_fingerprint": "1" * 64,
        "assignments": [ASSIGNMENT_FIXED_SHA256],
        "unique_definitions": [definition.definition_fingerprint],
    }
    import hashlib

    def digest(payload):
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    assert digest(independent_assignment_payload) == ASSIGNMENT_FIXED_SHA256
    assert digest(independent_state_payload) == STATE_FIXED_SHA256
    assert assignment.assignment_fingerprint == ASSIGNMENT_FIXED_SHA256
    assert state.state_fingerprint == STATE_FIXED_SHA256


def test_resolution_import_and_execution_are_dependency_light_in_fresh_process():
    script = r"""
import builtins
import json
import socket
import sys
from pathlib import Path
from radiosim.core.beam.models import (
    ResolvedAnalyticBeamDefinition,
    ResolvedAnalyticBeamsInput,
    ResolvedCircularApertureBeamModel,
    ResolvedGaussianTaper,
    _definition_fingerprint,
)
from radiosim.core.beam.resolution import resolve_beam_assignments
from radiosim.core.instrument import (
    AntennaFieldSource, AntennaId, AntennaProvenance, ResolvedAntenna,
    ResolvedEarthLocation, _create_resolved_instrument,
)
model = ResolvedCircularApertureBeamModel(
    "circular_aperture", ResolvedGaussianTaper("gaussian", 10.0)
)
definition = ResolvedAnalyticBeamDefinition(
    "analytic", model, _definition_fingerprint("analytic", model)
)
antenna = ResolvedAntenna(
    AntennaId(0, "ANT0"), (0.0, 0.0, 0.0), 10.0, None, "inert",
    AntennaProvenance(
        AntennaFieldSource.LAYOUT_FILE, AntennaFieldSource.LAYOUT_FILE,
        AntennaFieldSource.LAYOUT_FILE, 10.0, None,
        AntennaFieldSource.LAYOUT_FILE, "row:0"
    ),
)
location = ResolvedEarthLocation(
    0.0, 0.0, 0.0, (1.0, 2.0, 3.0),
    AntennaFieldSource.EXPLICIT_CONFIG, "fixture"
)
instrument = _create_resolved_instrument(
    name="Array", location=location, antennas=(antenna,), source_kind="fixture",
    source_reference="fixture", source_format="radiosim", registry_policy=None,
    telescope_name_source=AntennaFieldSource.EXPLICIT_CONFIG,
    location_source=AntennaFieldSource.EXPLICIT_CONFIG,
    source_location_itrs_xyz_m=None, location_separation_m=None,
    pyuvdata_version=None, source_sha256=None,
)
builtins.open = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("open"))
socket.socket = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network"))
state = resolve_beam_assignments(ResolvedAnalyticBeamsInput("analytic", definition), instrument)
forbidden = sorted(
    name for name in sys.modules
    if name.split('.')[0] in {
        'pyuvdata', 'jax', 'matplotlib', 'bokeh', 'selenium', 'webbrowser'
    }
    or name.startswith('radiosim.core.observability')
    or name.startswith('radiosim.api.simulator')
    or name.startswith('radiosim.core.visibility')
    or 'beam.runtime' in name
)
print(json.dumps({"forbidden": forbidden, "count": len(state.assignments)}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"forbidden": [], "count": 1}
