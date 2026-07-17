"""Tier 1B strict/frozen precision input and runtime tests."""

from __future__ import annotations

import builtins

import pytest
from pydantic import ValidationError

from radiosim.core.precision import (
    CoordinatePrecision,
    JonesPrecision,
    PrecisionConfig,
    SkyModelPrecision,
)
from radiosim.io.config import (
    ExecutionConfig,
    PrecisionInput,
    RadioSimConfig,
    collect_semantic_issues,
)
from tests.fixtures.configs import valid_config_mapping


@pytest.mark.parametrize(
    "model",
    [
        CoordinatePrecision(),
        JonesPrecision(),
        SkyModelPrecision(),
        PrecisionConfig(),
    ],
)
def test_runtime_precision_models_are_strict_and_frozen(model):
    model_type = type(model)
    assert model_type.model_config["extra"] == "forbid"
    assert model_type.model_config["frozen"] is True
    with pytest.raises(ValidationError, match="extra"):
        model_type.model_validate({"unknown": "float64"})


def test_runtime_precision_tree_rejects_top_and_nested_assignment():
    precision = PrecisionConfig.standard()

    with pytest.raises(ValidationError, match="frozen"):
        precision.default = "float32"
    with pytest.raises(ValidationError, match="frozen"):
        precision.coordinates.uvw = "float32"
    with pytest.raises(ValidationError, match="frozen"):
        precision.jones.beam = "float32"
    with pytest.raises(ValidationError, match="frozen"):
        precision.sky_model.flux = "float32"


def test_runtime_precision_presets_keep_existing_numerical_policy():
    standard = PrecisionConfig.standard()
    fast = PrecisionConfig.fast()
    precise = PrecisionConfig.precise()
    ultra = PrecisionConfig.ultra()

    assert standard.default == "float64"
    assert standard.sky_model.healpix_maps == "float32"
    assert fast.default == "float32"
    assert fast.jones.geometric_phase == "float64"
    assert fast.accumulation == "float64"
    assert precise.jones.geometric_phase == "float128"
    assert precise.accumulation == "float128"
    assert ultra.default == "float128"
    assert ultra.sky_model.healpix_maps == "float64"


def test_frozen_runtime_with_overrides_returns_new_tree():
    original = PrecisionConfig.fast()

    changed = original.with_overrides(output="float64", jones={"beam": "float64"})

    assert changed is not original
    assert changed.output == "float64"
    assert changed.jones.beam == "float64"
    assert original.output == "float32"
    assert original.jones.beam == "float32"


def test_input_precision_tree_is_strict_and_frozen():
    precision = PrecisionInput(coordinates={"uvw": "float32"})

    with pytest.raises(ValidationError, match="extra"):
        PrecisionInput(coordinates={"uvw": "float32", "typo": "float64"})
    with pytest.raises(ValidationError, match="frozen"):
        precision.accumulation = "float32"
    with pytest.raises(ValidationError, match="frozen"):
        precision.coordinates.uvw = "float64"


def test_standard_input_default_resolves_to_frozen_runtime_precision():
    execution = ExecutionConfig()

    runtime = execution.precision.to_precision_config()

    assert execution.precision.preset == "standard"
    assert runtime == PrecisionConfig.standard()
    with pytest.raises(ValidationError, match="frozen"):
        runtime.output = "float32"


def test_preset_and_custom_leaves_are_semantic_contradiction(tmp_path):
    data = valid_config_mapping(
        tmp_path,
        execution={
            "precision": {"preset": "fast", "output": "float32"},
        },
    )
    config = RadioSimConfig.model_validate(data)

    issues = collect_semantic_issues(config)

    assert any(issue.code == "preset_custom_contradiction" for issue in issues)
    with pytest.raises(ValueError, match="mutually exclusive"):
        config.execution.precision.to_precision_config()


@pytest.mark.parametrize("backend", ["jax", "numba"])
def test_explicit_jax_or_numba_float128_is_semantic_issue(tmp_path, backend):
    data = valid_config_mapping(
        tmp_path,
        execution={
            "backend": backend,
            "precision": {"coordinates": {"uvw": "float128"}},
        },
    )
    config = RadioSimConfig.model_validate(data)

    issues = collect_semantic_issues(config)

    assert any(
        issue.path == "execution.precision.coordinates.uvw"
        and issue.code == "backend_precision_incompatible"
        for issue in issues
    )


def test_precision_validation_does_not_import_optional_backends(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if (
            name == "jax"
            or name.startswith("jax.")
            or name == "numba"
            or name.startswith("numba.")
        ):
            pytest.fail(f"precision validation imported optional backend {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    config = RadioSimConfig.model_validate(
        valid_config_mapping(
            tmp_path,
            execution={"backend": "jax", "precision": {"preset": "precise"}},
        )
    )

    assert any(
        issue.code == "backend_precision_incompatible"
        for issue in collect_semantic_issues(config)
    )
