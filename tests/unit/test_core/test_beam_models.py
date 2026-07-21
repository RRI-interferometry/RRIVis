"""Contract tests for immutable source-resolved Tier 3B beam models."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

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

    for name in models.__all__:
        direct = getattr(models, name)
        assert getattr(beam, name) is direct
        assert getattr(core, name) is direct
        assert name in beam.__all__
        assert name in core.__all__
        assert not hasattr(root, name)
    for future_name in (
        "BeamSystem",
        "ResolvedBeamAssignment",
        "ResolvedBeamState",
        "LoadedBeamState",
        "BeamFileProvenance",
    ):
        assert not hasattr(beam, future_name)
        assert not hasattr(core, future_name)


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
        "ResolvedMixedBeamsInput": ("mode", "analytic_model", "assignments"),
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
