"""Pure Tier 1C backend-strategy and precision resolution tests."""

from __future__ import annotations

import builtins
import warnings

import pytest

import radiosim.backends as backend_module
from radiosim.backends import get_backend
from radiosim.backends.base import BackendNotAvailableError
from radiosim.core.precision import PrecisionConfig, SkyModelPrecision
from radiosim.io.config import PrecisionInput
from radiosim.io.config_resolution import (
    ConfigSemanticError,
    ConfigurationSource,
    SimulationOverrides,
    resolve_config,
)
from tests.fixtures.configs import valid_config_mapping


@pytest.mark.parametrize("backend", ["numpy", "jax", "numba", "auto"])
def test_requested_backend_strategy_is_preserved_without_construction(
    tmp_path, backend
):
    data = valid_config_mapping(tmp_path, execution={"backend": backend})

    bundle = resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )

    assert bundle.runtime.execution.backend_strategy == backend
    assert bundle.runtime.execution.backend == backend


def test_execution_omission_resolves_standard_precision(tmp_path):
    data = valid_config_mapping(tmp_path)
    data.pop("execution")

    bundle = resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )

    assert bundle.runtime.execution.backend_strategy == "numpy"
    assert bundle.runtime.execution.precision == PrecisionConfig.standard()


@pytest.mark.parametrize("preset", ["standard", "fast", "precise", "ultra"])
def test_precision_presets_resolve_deterministically(tmp_path, preset):
    data = valid_config_mapping(
        tmp_path,
        execution={"backend": "numpy", "precision": {"preset": preset}},
    )

    bundle = resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )

    expected = getattr(PrecisionConfig, preset)()
    assert bundle.runtime.execution.precision == expected


def test_complete_precision_override_replaces_document_tree(tmp_path):
    data = valid_config_mapping(
        tmp_path,
        execution={"precision": {"preset": "fast"}},
    )

    bundle = resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
        overrides=SimulationOverrides(precision=PrecisionInput(preset="standard")),
    )

    assert bundle.runtime.execution.precision == PrecisionConfig.standard()
    assert bundle.provenance.override_origins["execution.precision"] == "override"


@pytest.mark.parametrize("backend", ["jax", "numba"])
def test_incompatible_precision_override_fails_before_optional_import(
    tmp_path, monkeypatch, backend
):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == backend or name.startswith(f"{backend}."):
            pytest.fail(f"resolution imported optional backend {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    data = valid_config_mapping(tmp_path)

    with pytest.raises(ConfigSemanticError) as exc_info:
        resolve_config(
            data,
            source=ConfigurationSource.for_mapping(
                base_dir=tmp_path,
                invocation_dir=tmp_path,
            ),
            overrides=SimulationOverrides(
                backend=backend,
                precision=PrecisionInput(coordinates={"uvw": "float128"}),
            ),
        )

    assert any(
        issue.code == "backend_precision_incompatible"
        for issue in exc_info.value.issues
    )


@pytest.mark.parametrize(
    "precision",
    [
        PrecisionConfig.precise(),
        PrecisionConfig(sky_model=SkyModelPrecision(flux="float128")),
    ],
)
def test_auto_backend_never_silently_downgrades_float128(monkeypatch, precision):
    monkeypatch.setattr(backend_module, "NUMBA_AVAILABLE", True)
    monkeypatch.setattr(backend_module, "is_cuda_available", lambda: False)

    with warnings.catch_warnings(record=True) as caught:
        try:
            backend = get_backend("auto", precision=precision)
        except BackendNotAvailableError:
            backend = None
        if backend is not None:
            if precision.sky_model.flux == "float128":
                _ = backend.get_real_dtype("sky_model", "flux")
            else:
                _ = backend.get_complex_dtype("jones", "geometric_phase")

    assert not caught
    if backend is not None:
        assert backend.name == "numpy-cpu"
        assert backend.precision == precision


@pytest.mark.parametrize("backend_name", ["jax", "numba"])
def test_explicit_backend_factory_rejects_float128_without_warning(backend_name):
    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises(BackendNotAvailableError, match="requested precision"):
            get_backend(backend_name, precision=PrecisionConfig.precise())

    assert not caught


def test_explicit_numpy_factory_never_silently_downgrades_float128():
    precision = PrecisionConfig.precise()

    with warnings.catch_warnings(record=True) as caught:
        try:
            backend = get_backend("numpy", precision=precision)
        except BackendNotAvailableError:
            backend = None

    assert not caught
    if backend is not None:
        assert backend.name == "numpy-cpu"
        assert backend.precision == precision
