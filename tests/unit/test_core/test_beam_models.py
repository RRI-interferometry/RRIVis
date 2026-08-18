"""Contract tests for immutable source-resolved Tier 3B beam models."""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import numpy as np
import pytest

from radiosim.io.config_resolution import ConfigurationSource, resolve_config
from tests.fixtures.configs import valid_config_mapping


def _models():
    return importlib.import_module("radiosim.core.beam.models")


def _resolve(tmp_path: Path, beams: dict[str, object]):
    data = valid_config_mapping(tmp_path, beams=beams)
    return resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )


def test_resolved_types_are_exported_only_from_core_boundaries():
    models = _models()
    beam = importlib.import_module("radiosim.core.beam")
    core = importlib.import_module("radiosim.core")
    root = importlib.import_module("radiosim")

    # SCI-005 Stage-1 added its resolved aperture types and the Zernike bound to
    # the beam boundary only.  ``src/radiosim/core/__init__.py`` is not on the
    # Stage-1 writable list, and a top-level re-export is a convenience rather
    # than an obligation, so these names stop at ``radiosim.core.beam`` exactly
    # as the Tier 3D handler/provenance types below do.
    beam_only = {
        "ZERNIKE_MAX_RADIAL_ORDER",
        "ResolvedApertureBlockage",
        "ResolvedAperturePhysics",
        "ResolvedRuzePowerDiagnostic",
        "ResolvedSupportLeg",
        "ResolvedZernikeMode",
        "ResolvedZernikeSurface",
    }
    assert beam_only <= set(models.__all__)
    for name in models.__all__:
        direct = getattr(models, name)
        assert getattr(beam, name) is direct
        assert name in beam.__all__
        assert not hasattr(root, name)
        if name in beam_only:
            assert not hasattr(core, name)
            assert name not in core.__all__
            continue
        assert getattr(core, name) is direct
        assert name in core.__all__
    for tier3c_name in (
        "BeamAssignmentProvenance",
        "ResolvedBeamAssignment",
        "ResolvedBeamState",
    ):
        direct = getattr(models, tier3c_name)
        assert getattr(beam, tier3c_name) is direct
        assert getattr(core, tier3c_name) is direct
        assert tier3c_name in beam.__all__
        assert tier3c_name in core.__all__
        assert not hasattr(root, tier3c_name)
    for tier3d_name in (
        "BeamFileProvenance",
        "LoadedBeamHandlerState",
    ):
        direct = getattr(models, tier3d_name)
        assert getattr(beam, tier3d_name) is direct
        assert tier3d_name in beam.__all__
        assert not hasattr(core, tier3d_name)
        assert not hasattr(root, tier3d_name)
    assert beam.LoadedBeamState is models.LoadedBeamState
    assert core.LoadedBeamState is models.LoadedBeamState
    assert beam.BeamSystem is core.BeamSystem
    assert beam.load_beam_system is core.load_beam_system
    for tier3e_name in ("BeamSystem", "LoadedBeamState", "load_beam_system"):
        assert tier3e_name in beam.__all__
        assert tier3e_name in core.__all__
        assert not hasattr(root, tier3e_name)


def test_beam_schema_imports_do_not_load_heavy_or_later_tier_modules():
    script = """
import json
import sys
import radiosim.io.beam_config
import radiosim.core.beam.models
forbidden = sorted(
    name for name in sys.modules
    if name.split('.')[0] in {
        'pyuvdata', 'jax', 'matplotlib', 'bokeh', 'selenium', 'webbrowser'
    }
    or name.startswith('radiosim.core.observability')
)
print(json.dumps(forbidden))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_resolved_leaf_field_order_is_exact():
    models = _models()
    expected = {
        "ResolvedGaussianTaper": ("kind", "edge_taper_db"),
        "ResolvedCorrugatedHornIllumination": ("kind", "focal_ratio", "q"),
        "ResolvedCassegrainReflector": ("kind", "magnification"),
        "ResolvedCircularApertureBeamModel": ("kind", "taper"),
        "ResolvedNumericalIlluminationBeamModel": (
            "kind",
            "illumination",
            "reflector",
            "n_radial",
        ),
        "ResolvedAnalyticBeamDefinition": (
            "kind",
            "model",
            "definition_fingerprint",
        ),
        "ResolvedFITSBeamDefinition": (
            "kind",
            "path",
            "normalization",
            "angular_interpolation",
            "frequency_interpolation",
            "path_provenance_key",
            "definition_fingerprint",
        ),
        "ResolvedMixedBeamsInput": (
            "mode",
            "analytic_model",
            "assignments",
            "pointing",
            "surface_error",
            "aperture_physics",
        ),
        "ResolvedPointingOffset": (
            "azimuth_offset_rad",
            "elevation_offset_rad",
        ),
        "ResolvedSurfaceError": (
            "rms_surface_error_m",
            "error_beam_diagnostic",
        ),
        "ResolvedRuzePowerDiagnostic": ("kind", "correlation_length_m"),
        "ResolvedSupportLeg": ("position_angle_deg", "width_m"),
        "ResolvedApertureBlockage": (
            "central_diameter_ratio",
            "support_legs",
        ),
        "ResolvedZernikeMode": ("n", "m", "surface_height_coefficient_m"),
        "ResolvedZernikeSurface": ("convention", "modes"),
        "ResolvedAperturePhysics": (
            "normalization",
            "blockage",
            "zernike_surface",
        ),
        "ResolvedAntennaPointingOffset": ("antenna", "offset"),
        "ResolvedAntennaSurfaceError": ("antenna", "surface_error"),
        "ResolvedBeamPointing": ("default", "per_antenna"),
        "ResolvedBeamSurfaceError": ("default", "per_antenna"),
        "BeamAssignmentProvenance": (
            "source",
            "input_index",
            "authored_reference_kind",
            "authored_reference_value",
            "canonical_antenna",
        ),
        "ResolvedBeamAssignment": (
            "antenna_id",
            "antenna_diameter_m",
            "definition",
            "provenance",
            "assignment_fingerprint",
            "pointing",
            "surface_error",
            "aperture_physics",
        ),
        "ResolvedBeamState": (
            "mode",
            "instrument_fingerprint",
            "assignments",
            "unique_definitions",
            "state_fingerprint",
        ),
        "LoadedBeamHandlerState": (
            "handler_id",
            "kind",
            "definition_fingerprint",
            "scientific_fingerprint",
            "file",
            "voltage_feature_scale_by_frequency",
        ),
        "LoadedBeamState": (
            "resolved",
            "handlers",
            "assignment_handler_ids",
            "loaded_fingerprint",
        ),
    }

    for name, names in expected.items():
        assert tuple(field.name for field in fields(getattr(models, name))) == names


def test_resolved_leaves_are_frozen_hashable_and_snapshot_detached():
    models = _models()
    taper = models.ResolvedGaussianTaper("gaussian", 10.0)
    model = models.ResolvedCircularApertureBeamModel("circular_aperture", taper)

    assert hash(taper)
    assert hash(model)
    snapshot = model.to_snapshot()
    snapshot["taper"]["edge_taper_db"] = 99.0
    assert model.taper.edge_taper_db == 10.0
    assert json.loads(json.dumps(model.to_snapshot())) == model.to_snapshot()
    assert not isinstance(model, (Mapping, Sequence))
    with pytest.raises(FrozenInstanceError):
        taper.edge_taper_db = 1.0


def test_resolved_classes_reject_subclassing_and_nested_subclass_values():
    models = _models()

    with pytest.raises(TypeError):

        class MutableTaper(models.ResolvedGaussianTaper):
            pass

    class FakeTaper:
        kind = "gaussian"
        edge_taper_db = 10.0

    with pytest.raises(TypeError, match="taper"):
        models.ResolvedCircularApertureBeamModel("circular_aperture", FakeTaper())


@pytest.mark.parametrize(
    "beams",
    [
        {"mode": "analytic"},
        {
            "mode": "analytic",
            "model": {
                "kind": "rectangular_aperture",
                "north_length_m": 14.0,
                "east_length_m": 12.0,
            },
        },
        {
            "mode": "analytic",
            "model": {
                "kind": "analytical_illumination",
                "illumination": {"kind": "dipole_ground_plane"},
                "taper_profile": {"kind": "parabolic_squared"},
                "reflector": {"kind": "cassegrain", "magnification": 2.0},
            },
        },
        {
            "mode": "analytic",
            "model": {
                "kind": "numerical_illumination",
                "illumination": {"kind": "open_waveguide"},
            },
        },
    ],
)
def test_analytic_resolution_is_deterministic_and_json_safe(tmp_path, beams):
    first = _resolve(tmp_path, beams).runtime.beams
    second = _resolve(tmp_path, beams).runtime.beams

    assert first == second
    assert first.model.definition_fingerprint == second.model.definition_fingerprint
    assert len(first.model.definition_fingerprint) == 64
    assert first.to_snapshot()["mode"] == "analytic"
    if first.model.model.kind == "numerical_illumination":
        assert first.model.model.n_radial == 256


def test_analytic_fingerprint_uses_exact_float_and_is_key_order_independent(tmp_path):
    left = {
        "mode": "analytic",
        "model": {
            "kind": "circular_aperture",
            "taper": {"kind": "gaussian", "edge_taper_db": 10.0},
        },
    }
    right = {
        "model": {
            "taper": {"edge_taper_db": 10.0, "kind": "gaussian"},
            "kind": "circular_aperture",
        },
        "mode": "analytic",
    }
    changed = {
        "mode": "analytic",
        "model": {
            "kind": "circular_aperture",
            "taper": {
                "kind": "gaussian",
                "edge_taper_db": float.fromhex((10.0).hex()) + math.ulp(10.0),
            },
        },
    }

    left_id = _resolve(tmp_path, left).runtime.beams.model.definition_fingerprint
    right_id = _resolve(tmp_path, right).runtime.beams.model.definition_fingerprint
    changed_id = _resolve(tmp_path, changed).runtime.beams.model.definition_fingerprint

    assert left_id == right_id
    assert changed_id != left_id


def test_analytic_fingerprint_matches_independent_canonical_digest(tmp_path):
    resolved = _resolve(tmp_path, {"mode": "analytic"}).runtime.beams.model

    assert resolved.definition_fingerprint == (
        "f79b164f5ae8b08fcc728f5f6a25bd14e5566312230eb668c14b7f5765b628d4"
    )


def test_resolved_fits_definition_rejects_noncanonical_paths_and_strings():
    models = _models()

    def definition(path, *, frequency_interpolation="cubic"):
        payload = {
            "normalization": "peak",
            "angular_interpolation": "bilinear",
            "frequency_interpolation": frequency_interpolation,
        }
        return models.ResolvedFITSBeamDefinition(
            "fits",
            path,
            "peak",
            "bilinear",
            frequency_interpolation,
            "beams.beam.path",
            models._definition_fingerprint("fits", payload),
        )

    with pytest.raises(ValueError, match="normalized"):
        definition(Path("/tmp/a/../beam.fits"))

    path_type = type(Path())

    class PathSubclass(path_type):
        pass

    with pytest.raises((TypeError, ValueError), match="Path"):
        definition(PathSubclass("/tmp/beam.fits"))

    class StringSubclass(str):
        pass

    normalized = Path("/tmp/beam.fits").resolve(strict=False)
    with pytest.raises((TypeError, ValueError), match="frequency_interpolation"):
        definition(
            normalized,
            frequency_interpolation=StringSubclass("cubic"),
        )


@pytest.mark.parametrize("provenance_key", ["", "   "])
def test_resolved_fits_definition_rejects_blank_provenance_keys(provenance_key):
    models = _models()
    path = Path("/tmp/beam.fits").resolve(strict=False)
    payload = {
        "normalization": "peak",
        "angular_interpolation": "bilinear",
        "frequency_interpolation": "cubic",
    }

    with pytest.raises(ValueError, match="path_provenance_key"):
        models.ResolvedFITSBeamDefinition(
            "fits",
            path,
            "peak",
            "bilinear",
            "cubic",
            provenance_key,
            models._definition_fingerprint("fits", payload),
        )


def test_fits_definitions_use_normalized_path_options_and_logical_keys(tmp_path):
    models = _models()
    first_path = tmp_path / "first.beamfits"
    second_path = tmp_path / "second.beamfits"
    first_path.touch()
    second_path.touch()
    bundle = _resolve(
        tmp_path,
        {
            "mode": "per_antenna_fits",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 1},
                    "beam": {"kind": "fits", "path": first_path.name},
                },
                {
                    "antenna": {"kind": "name", "name": "ANT0"},
                    "beam": {
                        "kind": "fits",
                        "path": second_path.name,
                        "frequency_interpolation": "linear",
                    },
                },
            ],
        },
    )
    resolved = bundle.runtime.beams

    assert type(resolved) is models.ResolvedPerAntennaFITSBeamsInput
    assert [item.antenna.kind for item in resolved.assignments] == ["number", "name"]
    assert resolved.assignments[0].beam.path == first_path.resolve()
    assert resolved.assignments[0].beam.path_provenance_key == (
        "beams.assignments[0].beam.path"
    )
    assert resolved.assignments[1].beam.path_provenance_key == (
        "beams.assignments[1].beam.path"
    )
    assert resolved.assignments[0].beam.definition_fingerprint != (
        resolved.assignments[1].beam.definition_fingerprint
    )
    assert set(bundle.provenance.path_resolutions) >= {
        "beams.assignments[0].beam.path",
        "beams.assignments[1].beam.path",
    }


def test_fits_fingerprint_is_path_independent_and_matches_canonical_digest(tmp_path):
    first_path = tmp_path / "one" / "beam.beamfits"
    second_path = tmp_path / "two" / "beam.beamfits"
    for path in (first_path, second_path):
        path.parent.mkdir()
        path.touch()
    resolved = _resolve(
        tmp_path,
        {
            "mode": "per_antenna_fits",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 1},
                    "beam": {"kind": "fits", "path": "one/beam.beamfits"},
                },
                {
                    "antenna": {"kind": "name", "name": "ANT0"},
                    "beam": {"kind": "fits", "path": "two/beam.beamfits"},
                },
            ],
        },
    ).runtime.beams

    first, second = (item.beam for item in resolved.assignments)
    assert first.path != second.path
    assert first.definition_fingerprint == second.definition_fingerprint
    assert first.definition_fingerprint == (
        "e6a3cdc9c545cba8c0e8bda69405483ab70232071103b5286dd4fa5bee5c5926"
    )


def test_shared_and_mixed_source_resolution_do_not_create_fake_choices(tmp_path):
    models = _models()
    beam_path = tmp_path / "beam.fits"
    beam_path.touch()
    shared = _resolve(
        tmp_path,
        {"mode": "shared_fits", "beam": {"kind": "fits", "path": beam_path.name}},
    ).runtime.beams
    mixed = _resolve(
        tmp_path,
        {
            "mode": "mixed",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": {"kind": "analytic"},
                },
                {
                    "antenna": {"kind": "number", "number": 1},
                    "beam": {"kind": "fits", "path": beam_path.name},
                },
            ],
        },
    ).runtime.beams

    assert type(shared) is models.ResolvedSharedFITSBeamsInput
    assert shared.beam.path_provenance_key == "beams.beam.path"
    assert type(mixed) is models.ResolvedMixedBeamsInput
    assert type(mixed.assignments[0].beam) is models.ResolvedAnalyticBeamChoice
    assert not hasattr(mixed.assignments[0].beam, "path")
    assert mixed.assignments[1].beam.path_provenance_key == (
        "beams.assignments[1].beam.path"
    )


def test_resolved_assignment_tuple_is_owned_and_rejects_non_tuple(tmp_path):
    models = _models()
    beam_path = tmp_path / "beam.fits"
    beam_path.touch()
    resolved = _resolve(
        tmp_path,
        {
            "mode": "per_antenna_fits",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": {"kind": "fits", "path": beam_path.name},
                }
            ],
        },
    ).runtime.beams
    original = resolved.assignments

    rebuilt = models.ResolvedPerAntennaFITSBeamsInput("per_antenna_fits", original)

    assert rebuilt.assignments == original
    assert rebuilt.assignments is not original
    with pytest.raises(TypeError, match="tuple"):
        models.ResolvedPerAntennaFITSBeamsInput("per_antenna_fits", list(original))


def test_resolved_beams_config_is_fully_absent():
    runtime = importlib.import_module("radiosim.core.runtime_config")
    core = importlib.import_module("radiosim.core")
    public_io = importlib.import_module("radiosim.io")

    for module in (runtime, core, public_io):
        assert not hasattr(module, "ResolvedBeamsConfig")
        assert "ResolvedBeamsConfig" not in module.__all__


def _valid_assignment_models(tmp_path: Path):
    models = _models()
    resolved = _resolve(tmp_path, {"mode": "analytic"}).runtime.beams
    antenna = importlib.import_module("radiosim.core.instrument").AntennaId(0, "ANT0")
    provenance = models.BeamAssignmentProvenance(
        "analytic_mode",
        None,
        None,
        None,
        antenna,
    )
    assignment = models._create_resolved_beam_assignment(
        antenna_id=antenna,
        antenna_diameter_m=10.0,
        definition=resolved.model,
        provenance=provenance,
    )
    state = models._create_resolved_beam_state(
        mode="analytic",
        instrument_fingerprint="1" * 64,
        assignments=(assignment,),
        unique_definitions=(resolved.model,),
    )
    return models, provenance, assignment, state


def test_tier3c_assignment_models_are_frozen_slotted_hashable_and_detached(tmp_path):
    models, provenance, assignment, state = _valid_assignment_models(tmp_path)

    for value in (provenance, assignment, state):
        assert "__slots__" in type(value).__dict__
        assert "__dict__" not in type(value).__dict__
        assert isinstance(hash(value), int)
        assert not isinstance(value, (Mapping, Sequence))
        with pytest.raises(FrozenInstanceError):
            setattr(value, fields(value)[0].name, None)

    snapshot = state.to_snapshot()
    assert json.loads(json.dumps(snapshot)) == snapshot
    assert snapshot["assignments"][0]["provenance"]["source"] == "analytic_mode"
    snapshot["assignments"][0]["provenance"]["canonical_antenna"]["name"] = "changed"
    assert state.assignments[0].antenna_id.name == "ANT0"
    assert state.to_snapshot()["assignments"][0]["antenna_id"]["name"] == "ANT0"

    for class_name in (
        "BeamAssignmentProvenance",
        "ResolvedBeamAssignment",
        "ResolvedBeamState",
    ):
        with pytest.raises(TypeError):
            type(f"Hostile{class_name}", (getattr(models, class_name),), {})


@pytest.mark.parametrize(
    "values",
    [
        ("analytic_mode", 0, None, None),
        ("analytic_mode", None, "number", 0),
        ("shared_mode", None, None, "ANT0"),
        ("explicit_assignment", None, "number", 0),
        ("explicit_assignment", True, "number", 0),
        ("explicit_assignment", -1, "number", 0),
        ("explicit_assignment", 0, "number", True),
        ("explicit_assignment", 0, "number", -1),
        ("explicit_assignment", 0, "number", "0"),
        ("explicit_assignment", 0, "number", 1),
        ("explicit_assignment", 0, "name", 0),
        ("explicit_assignment", 0, "name", "OTHER"),
        ("explicit_assignment", 0, "name", " ANT0 "),
        ("explicit_assignment", 0, "name", "A\N{COMBINING RING ABOVE}"),
    ],
)
def test_assignment_provenance_rejects_inconsistent_or_noncanonical_values(values):
    models = _models()
    antenna = importlib.import_module("radiosim.core.instrument").AntennaId(0, "ANT0")

    with pytest.raises((TypeError, ValueError)):
        models.BeamAssignmentProvenance(*values, antenna)


def test_assignment_provenance_accepts_exact_explicit_number_and_name():
    models = _models()
    antenna = importlib.import_module("radiosim.core.instrument").AntennaId(0, "ANT0")

    numbered = models.BeamAssignmentProvenance(
        "explicit_assignment", 0, "number", 0, antenna
    )
    named = models.BeamAssignmentProvenance(
        "explicit_assignment", 1, "name", "ANT0", antenna
    )

    assert numbered.authored_reference_value == 0
    assert type(numbered.authored_reference_value) is int
    assert named.authored_reference_value == "ANT0"
    assert type(named.authored_reference_value) is str


@pytest.mark.parametrize(
    "diameter",
    [10, True, math.nan, math.inf, -math.inf, 0.0, -1.0],
)
def test_resolved_assignment_rejects_nonexact_invalid_diameter(tmp_path, diameter):
    _, _, assignment, _ = _valid_assignment_models(tmp_path)

    with pytest.raises((TypeError, ValueError)):
        replace(assignment, antenna_diameter_m=diameter)


def test_resolved_assignment_rejects_numpy_float_and_forged_nested_values(tmp_path):
    models, provenance, assignment, _ = _valid_assignment_models(tmp_path)

    with pytest.raises((TypeError, ValueError)):
        replace(assignment, antenna_diameter_m=np.float64(10.0))
    with pytest.raises((TypeError, ValueError)):
        replace(assignment, definition=object())
    with pytest.raises((TypeError, ValueError)):
        replace(assignment, provenance=object())

    other = importlib.import_module("radiosim.core.instrument").AntennaId(1, "ANT1")
    with pytest.raises(ValueError, match="canonical_antenna"):
        replace(
            assignment,
            provenance=replace(provenance, canonical_antenna=other),
        )


@pytest.mark.parametrize("fingerprint", ["A" * 64, "a" * 63, "g" * 64])
def test_assignment_and_state_reject_malformed_or_recomputed_fingerprints(
    tmp_path, fingerprint
):
    _, _, assignment, state = _valid_assignment_models(tmp_path)

    with pytest.raises(ValueError):
        replace(assignment, assignment_fingerprint=fingerprint)
    with pytest.raises(ValueError):
        replace(state, state_fingerprint=fingerprint)
    with pytest.raises(ValueError, match="does not match"):
        replace(assignment, assignment_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        replace(state, state_fingerprint="0" * 64)


def test_state_requires_exact_owned_nonempty_tuples_and_unique_antennas(tmp_path):
    models, _, assignment, state = _valid_assignment_models(tmp_path)

    rebuilt = models._create_resolved_beam_state(
        mode=state.mode,
        instrument_fingerprint=state.instrument_fingerprint,
        assignments=state.assignments,
        unique_definitions=state.unique_definitions,
    )
    assert rebuilt.assignments == state.assignments
    assert rebuilt.assignments is not state.assignments
    assert rebuilt.unique_definitions == state.unique_definitions
    assert rebuilt.unique_definitions is not state.unique_definitions

    with pytest.raises(TypeError, match="tuple"):
        replace(state, assignments=list(state.assignments))
    with pytest.raises(TypeError, match="tuple"):
        replace(state, unique_definitions=list(state.unique_definitions))
    with pytest.raises(ValueError, match="nonempty"):
        replace(state, assignments=())
    with pytest.raises(ValueError, match="nonempty"):
        replace(state, unique_definitions=())
    with pytest.raises(ValueError, match="duplicate"):
        models._create_resolved_beam_state(
            mode="analytic",
            instrument_fingerprint="1" * 64,
            assignments=(assignment, assignment),
            unique_definitions=state.unique_definitions,
        )


def test_state_rejects_wrong_assignment_order_and_unique_definition_membership(
    tmp_path,
):
    models, _, first, state = _valid_assignment_models(tmp_path)
    antenna_type = importlib.import_module("radiosim.core.instrument").AntennaId
    second_id = antenna_type(2, "ANT2")
    second_provenance = models.BeamAssignmentProvenance(
        "analytic_mode", None, None, None, second_id
    )
    second = models._create_resolved_beam_assignment(
        antenna_id=second_id,
        antenna_diameter_m=10.0,
        definition=first.definition,
        provenance=second_provenance,
    )

    with pytest.raises(ValueError, match="canonical"):
        models._create_resolved_beam_state(
            mode="analytic",
            instrument_fingerprint="1" * 64,
            assignments=(second, first),
            unique_definitions=state.unique_definitions,
        )
    with pytest.raises((TypeError, ValueError), match="unique_definitions"):
        replace(state, unique_definitions=(object(),))


def test_state_rejects_assignment_provenance_source_inconsistent_with_mode(tmp_path):
    models, _, assignment, _ = _valid_assignment_models(tmp_path)
    explicit = models.BeamAssignmentProvenance(
        "explicit_assignment",
        0,
        "number",
        assignment.antenna_id.number,
        assignment.antenna_id,
    )
    forged = models._create_resolved_beam_assignment(
        antenna_id=assignment.antenna_id,
        antenna_diameter_m=assignment.antenna_diameter_m,
        definition=assignment.definition,
        provenance=explicit,
    )

    with pytest.raises(ValueError, match="provenance"):
        models._create_resolved_beam_state(
            mode="analytic",
            instrument_fingerprint="1" * 64,
            assignments=(forged,),
            unique_definitions=(forged.definition,),
        )


def test_assignment_revalidates_nested_definition_fingerprint(tmp_path):
    models, provenance, assignment, _ = _valid_assignment_models(tmp_path)
    rectangular = models.ResolvedRectangularApertureBeamModel(
        "rectangular_aperture",
        14.0,
        12.0,
    )

    stale_definition = replace(assignment.definition)
    object.__setattr__(stale_definition, "model", rectangular)
    with pytest.raises(ValueError, match="definition_fingerprint"):
        models._create_resolved_beam_assignment(
            antenna_id=assignment.antenna_id,
            antenna_diameter_m=assignment.antenna_diameter_m,
            definition=stale_definition,
            provenance=provenance,
        )


def test_state_revalidates_nested_assignment_fingerprint(tmp_path):
    models, _, assignment, state = _valid_assignment_models(tmp_path)
    stale_assignment = replace(assignment)
    object.__setattr__(stale_assignment, "antenna_diameter_m", 20.0)
    with pytest.raises(ValueError, match="assignment_fingerprint"):
        models._create_resolved_beam_state(
            mode=state.mode,
            instrument_fingerprint=state.instrument_fingerprint,
            assignments=(stale_assignment,),
            unique_definitions=state.unique_definitions,
        )


def test_state_rejects_duplicate_canonical_name_with_different_numbers(tmp_path):
    models, _, first, state = _valid_assignment_models(tmp_path)
    duplicate_name_id = importlib.import_module("radiosim.core.instrument").AntennaId(
        1, first.antenna_id.name
    )
    duplicate_name = models._create_resolved_beam_assignment(
        antenna_id=duplicate_name_id,
        antenna_diameter_m=first.antenna_diameter_m,
        definition=first.definition,
        provenance=models.BeamAssignmentProvenance(
            "analytic_mode",
            None,
            None,
            None,
            duplicate_name_id,
        ),
    )

    with pytest.raises(ValueError, match="duplicate canonical antenna"):
        models._create_resolved_beam_state(
            mode="analytic",
            instrument_fingerprint=state.instrument_fingerprint,
            assignments=(first, duplicate_name),
            unique_definitions=state.unique_definitions,
        )


def test_mixed_state_rejects_multiple_analytic_definitions(tmp_path):
    models, _, first, state = _valid_assignment_models(tmp_path)
    second_definition = _resolve(
        tmp_path,
        {
            "mode": "analytic",
            "model": {
                "kind": "rectangular_aperture",
                "north_length_m": 14.0,
                "east_length_m": 12.0,
            },
        },
    ).runtime.beams.model
    second_id = importlib.import_module("radiosim.core.instrument").AntennaId(
        1,
        "ANT1",
    )
    explicit_first = models._create_resolved_beam_assignment(
        antenna_id=first.antenna_id,
        antenna_diameter_m=first.antenna_diameter_m,
        definition=first.definition,
        provenance=models.BeamAssignmentProvenance(
            "explicit_assignment",
            0,
            "number",
            first.antenna_id.number,
            first.antenna_id,
        ),
    )
    explicit_second = models._create_resolved_beam_assignment(
        antenna_id=second_id,
        antenna_diameter_m=10.0,
        definition=second_definition,
        provenance=models.BeamAssignmentProvenance(
            "explicit_assignment",
            1,
            "number",
            second_id.number,
            second_id,
        ),
    )

    with pytest.raises(ValueError, match="one analytic definition"):
        models._create_resolved_beam_state(
            mode="mixed",
            instrument_fingerprint=state.instrument_fingerprint,
            assignments=(explicit_first, explicit_second),
            unique_definitions=(first.definition, second_definition),
        )
