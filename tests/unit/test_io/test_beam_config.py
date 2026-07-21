"""Contract tests for the final strict Tier 3 beam input union."""

from __future__ import annotations

import importlib
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from radiosim.io.config import RadioSimConfig, collect_schema_issues
from tests.fixtures.configs import valid_config_mapping


def _beam_config():
    return importlib.import_module("radiosim.io.beam_config")


def _adapter(name: str) -> TypeAdapter[Any]:
    return TypeAdapter(getattr(_beam_config(), name))


def _analytic(model: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {"mode": "analytic"}
    if model is not None:
        data["model"] = model
    return data


def test_beams_union_and_default_are_public_with_exact_identity():
    module = _beam_config()
    public_io = importlib.import_module("radiosim.io")

    assert public_io.BeamsConfig is module.BeamsConfig
    assert "BeamsConfig" in public_io.__all__
    assert not hasattr(importlib.import_module("radiosim"), "BeamsConfig")


def test_omitted_beams_defaults_to_direct_circular_gaussian_ten_db(tmp_path):
    data = valid_config_mapping(tmp_path)
    data.pop("beams")

    beams = RadioSimConfig.model_validate(data).beams

    assert beams.mode == "analytic"
    assert beams.model.kind == "circular_aperture"
    assert beams.model.taper.kind == "gaussian"
    assert beams.model.taper.edge_taper_db == 10.0


@pytest.mark.parametrize(
    ("name", "data", "expected"),
    [
        ("DirectTaperConfig", {"kind": "uniform"}, "UniformTaperConfig"),
        (
            "DirectTaperConfig",
            {"kind": "gaussian"},
            "GaussianTaperConfig",
        ),
        (
            "DirectTaperConfig",
            {"kind": "parabolic"},
            "ParabolicTaperConfig",
        ),
        (
            "DirectTaperConfig",
            {"kind": "parabolic_squared"},
            "ParabolicSquaredTaperConfig",
        ),
        ("DirectTaperConfig", {"kind": "cosine"}, "CosineTaperConfig"),
        (
            "FeedDerivedTaperConfig",
            {"kind": "gaussian"},
            "DerivedGaussianTaperConfig",
        ),
        (
            "FeedDerivedTaperConfig",
            {"kind": "parabolic"},
            "DerivedParabolicTaperConfig",
        ),
        (
            "FeedDerivedTaperConfig",
            {"kind": "parabolic_squared"},
            "DerivedParabolicSquaredTaperConfig",
        ),
        (
            "IlluminationConfig",
            {"kind": "corrugated_horn"},
            "CorrugatedHornIlluminationConfig",
        ),
        (
            "IlluminationConfig",
            {"kind": "open_waveguide"},
            "OpenWaveguideIlluminationConfig",
        ),
        (
            "IlluminationConfig",
            {"kind": "dipole_ground_plane"},
            "DipoleGroundPlaneIlluminationConfig",
        ),
        (
            "ReflectorConfig",
            {"kind": "prime_focus"},
            "PrimeFocusReflectorConfig",
        ),
        (
            "ReflectorConfig",
            {"kind": "cassegrain", "magnification": 2.0},
            "CassegrainReflectorConfig",
        ),
    ],
)
def test_leaf_unions_are_discriminated(name, data, expected):
    value = _adapter(name).validate_python(data)

    assert type(value).__name__ == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ({"kind": "circular_aperture"}, "CircularApertureBeamModelConfig"),
        (
            {
                "kind": "rectangular_aperture",
                "north_length_m": 12.0,
                "east_length_m": 10.0,
            },
            "RectangularApertureBeamModelConfig",
        ),
        (
            {
                "kind": "elliptical_aperture",
                "north_diameter_m": 14.0,
                "east_diameter_m": 12.0,
            },
            "EllipticalApertureBeamModelConfig",
        ),
        (
            {
                "kind": "analytical_illumination",
                "illumination": {"kind": "corrugated_horn"},
            },
            "AnalyticalIlluminationBeamModelConfig",
        ),
        (
            {
                "kind": "numerical_illumination",
                "illumination": {"kind": "open_waveguide"},
            },
            "NumericalIlluminationBeamModelConfig",
        ),
    ],
)
def test_all_analytic_model_variants_parse_with_exact_defaults(model, expected):
    value = _adapter("AnalyticBeamModelConfig").validate_python(model)

    assert type(value).__name__ == expected
    if value.kind == "analytical_illumination":
        assert value.taper_profile.kind == "gaussian"
        assert value.reflector.kind == "prime_focus"
    if value.kind == "numerical_illumination":
        assert value.reflector.kind == "prime_focus"
        assert not hasattr(value, "taper")


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"mode": "analytic"}, "AnalyticBeamsConfig"),
        (
            {"mode": "shared_fits", "beam": {"kind": "fits", "path": "a.fits"}},
            "SharedFITSBeamsConfig",
        ),
        (
            {
                "mode": "per_antenna_fits",
                "assignments": [
                    {
                        "antenna": {"kind": "number", "number": 0},
                        "beam": {"kind": "fits", "path": "a.fits"},
                    }
                ],
            },
            "PerAntennaFITSBeamsConfig",
        ),
        (
            {
                "mode": "mixed",
                "assignments": [
                    {
                        "antenna": {"kind": "name", "name": "ANT0"},
                        "beam": {"kind": "analytic"},
                    }
                ],
            },
            "MixedBeamsConfig",
        ),
    ],
)
def test_all_four_modes_parse_and_preserve_nested_discriminators(data, expected):
    value = _adapter("BeamsConfig").validate_python(data)
    dumped = value.model_dump(mode="json")
    reloaded = _adapter("BeamsConfig").validate_python(dumped)

    assert type(value).__name__ == expected
    assert reloaded == value
    assert dumped["mode"] == data["mode"]


def test_fits_source_has_only_fixed_normalization_and_interpolation_contract():
    module = _beam_config()
    source = module.FITSBeamSourceConfig(path="beam.fits")

    assert source.kind == "fits"
    assert source.path == Path("beam.fits")
    assert source.normalization == "peak"
    assert source.angular_interpolation == "bilinear"
    assert source.frequency_interpolation == "cubic"
    for field, value in (
        ("normalization", "none"),
        ("angular_interpolation", "spline"),
        ("frequency_interpolation", "nearest"),
        ("beam_za_max_deg", 90.0),
    ):
        with pytest.raises(ValidationError):
            module.FITSBeamSourceConfig.model_validate(
                {"path": "beam.fits", field: value}
            )


@pytest.mark.parametrize("path", ["", "   ", "$DATA/beam.fits", "${DATA}/b.fits"])
def test_fits_paths_reject_blank_and_environment_syntax(path):
    with pytest.raises(ValidationError, match="path|environment"):
        _beam_config().FITSBeamSourceConfig(path=path)


@pytest.mark.parametrize("value", [True, False, "1.0", math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    ("model", "field"),
    [
        ("GaussianTaperConfig", "edge_taper_db"),
        ("CorrugatedHornIlluminationConfig", "focal_ratio"),
        ("CorrugatedHornIlluminationConfig", "q"),
        ("CassegrainReflectorConfig", "magnification"),
        ("RectangularApertureBeamModelConfig", "north_length_m"),
    ],
)
def test_all_numeric_beam_fields_are_strict_and_finite(model, field, value):
    cls = getattr(_beam_config(), model)
    required: dict[str, Any] = {}
    if model == "CassegrainReflectorConfig":
        required[field] = value
    elif model == "RectangularApertureBeamModelConfig":
        required.update({"north_length_m": 1.0, "east_length_m": 1.0, field: value})
    else:
        required[field] = value

    with pytest.raises(ValidationError):
        cls.model_validate(required)


@pytest.mark.parametrize("value", [-1.0])
def test_edge_taper_is_nonnegative(value):
    with pytest.raises(ValidationError):
        _beam_config().GaussianTaperConfig(edge_taper_db=value)


@pytest.mark.parametrize("value", [0.0, -1.0, 1.0])
def test_cassegrain_magnification_must_be_strictly_greater_than_one(value):
    with pytest.raises(ValidationError):
        _beam_config().CassegrainReflectorConfig(magnification=value)


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "uniform", "edge_taper_db": 10.0},
        {"kind": "cosine", "edge_taper_db": 10.0},
        {"kind": "gaussian", "edge_taper_db": 10.0, "ignored": True},
    ],
)
def test_direct_taper_rejects_ignored_and_unknown_fields(data):
    with pytest.raises(ValidationError):
        _adapter("DirectTaperConfig").validate_python(data)


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "gaussian", "edge_taper_db": 10.0},
        {"kind": "parabolic", "edge_taper_db": 10.0},
        {"kind": "parabolic_squared", "edge_taper_db": 10.0},
    ],
)
def test_feed_derived_taper_rejects_authored_edge(data):
    with pytest.raises(ValidationError):
        _adapter("FeedDerivedTaperConfig").validate_python(data)


@pytest.mark.parametrize(
    "model",
    [
        {"kind": "circular_aperture", "north_length_m": 12.0},
        {
            "kind": "rectangular_aperture",
            "north_length_m": 12.0,
            "east_length_m": 10.0,
            "taper": {"kind": "uniform"},
        },
        {
            "kind": "elliptical_aperture",
            "north_diameter_m": 12.0,
            "east_diameter_m": 10.0,
            "reflector": {"kind": "prime_focus"},
        },
        {
            "kind": "analytical_illumination",
            "illumination": {"kind": "corrugated_horn"},
            "taper": {"kind": "uniform"},
        },
        {
            "kind": "numerical_illumination",
            "illumination": {"kind": "corrugated_horn"},
            "taper_profile": {"kind": "gaussian"},
        },
    ],
)
def test_analytic_models_reject_every_inapplicable_field_combination(model):
    with pytest.raises(ValidationError):
        _adapter("AnalyticBeamModelConfig").validate_python(model)


@pytest.mark.parametrize("mode", ["per_antenna_fits", "mixed"])
def test_assignment_modes_require_nonempty_ordered_assignments(mode):
    with pytest.raises(ValidationError):
        _adapter("BeamsConfig").validate_python({"mode": mode, "assignments": []})


def test_assignment_input_is_copied_frozen_and_order_preserving():
    assignments = [
        {
            "antenna": {"kind": "number", "number": 1},
            "beam": {"kind": "fits", "path": "b.fits"},
        },
        {
            "antenna": {"kind": "name", "name": "ANT0"},
            "beam": {"kind": "fits", "path": "a.fits"},
        },
    ]
    value = _adapter("BeamsConfig").validate_python(
        {"mode": "per_antenna_fits", "assignments": assignments}
    )
    assignments.reverse()
    assignments[0]["antenna"]["name"] = "MUTATED"

    assert type(value.assignments) is tuple
    assert value.assignments[0].antenna.number == 1
    assert value.assignments[1].antenna.name == "ANT0"
    with pytest.raises(ValidationError, match="frozen"):
        value.mode = "mixed"
    with pytest.raises((FrozenInstanceError, TypeError)):
        value.assignments[0] = value.assignments[1]


def test_hostile_model_subclasses_cannot_survive_in_accepted_beam_state():
    module = _beam_config()

    for parent in (
        module.GaussianTaperConfig,
        module.FITSBeamSourceConfig,
        module.FITSBeamAssignmentConfig,
    ):
        with pytest.raises(TypeError, match="do not support subclassing"):
            type(f"Mutable{parent.__name__}", (parent,), {})


OLD_FIELDS = (
    "beam_mode",
    "per_antenna",
    "beam_file",
    "antenna_beam_map",
    "beam_za_max_deg",
    "beam_za_buffer_deg",
    "beam_freq_buffer_hz",
    "beam_peak_normalize",
    "beam_interp_function",
    "aperture_shape",
    "taper",
    "edge_taper_dB",
    "feed_model",
    "feed_computation",
    "feed_params",
    "reflector_type",
    "magnification",
    "aperture_params",
    "use_beam_file",
    "use_different_beams",
    "beam_file_path",
    "beam_files",
    "beams_per_antenna",
    "default_beam_id",
    "beam_freq_interp",
    "beam_freq_buffer_mhz",
)


@pytest.mark.parametrize("field", OLD_FIELDS)
def test_every_old_field_has_direct_tier3_migration_guidance(tmp_path, field):
    data = valid_config_mapping(tmp_path)
    data["beams"][field] = True

    issue = next(
        item for item in collect_schema_issues(data) if item.path == f"beams.{field}"
    )

    assert issue.code == "removed_field"
    assert issue.message.startswith("removed in Tier 3;")
    assert issue.hint
    assert "beams." in issue.hint


def test_old_ignored_combinations_name_the_historical_noop(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["beams"]["edge_taper_dB"] = 11.0

    issue = next(
        item
        for item in collect_schema_issues(data)
        if item.path == "beams.edge_taper_dB"
    )

    assert (
        "the old implementation ignored this value; select an active Tier 3 model"
        in issue.hint
    )


def test_radio_config_rejects_old_fields_with_actionable_text(tmp_path):
    data = valid_config_mapping(tmp_path)
    data["beams"] = {"beam_mode": "analytic", "aperture_shape": "circular"}

    with pytest.raises(ValidationError) as exc_info:
        RadioSimConfig.model_validate(data)

    rendered = str(exc_info.value)
    assert "beams.beam_mode: removed in Tier 3;" in rendered
    assert "beams.aperture_shape: removed in Tier 3;" in rendered
