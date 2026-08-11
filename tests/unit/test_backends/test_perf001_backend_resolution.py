"""PERF-001 P-c backend-resolution acceptance tests."""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

import radiosim.backends as backend_module
import radiosim.backends.jax_backend as jax_backend_module
from radiosim.backends import NumPyBackend, get_backend, list_backends
from radiosim.backends.base import BackendNotAvailableError
from radiosim.backends.jax_backend import JAXBackend


def test_auto_is_deterministic_numpy_without_importing_jax() -> None:
    """The default path is selection, not implicit accelerator discovery."""
    code = """
import sys
from radiosim.backends import NumPyBackend, get_backend

assert "jax" not in sys.modules
assert "jaxlib" not in sys.modules
backend = get_backend("auto")
assert isinstance(backend, NumPyBackend)
assert backend.name == "numpy-cpu"
assert "jax" not in sys.modules
assert "jaxlib" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_auto_ignores_installed_jax_and_never_constructs_it(monkeypatch) -> None:
    class ForbiddenBackend:
        def __init__(self, *args, **kwargs):
            pytest.fail("auto constructed an optional JAX backend")

    monkeypatch.setattr(backend_module, "JAX_AVAILABLE", True)
    monkeypatch.setattr(backend_module, "JAXBackend", ForbiddenBackend)

    assert isinstance(get_backend("auto"), NumPyBackend)


def test_generic_jax_request_passes_the_runtime_default_device(monkeypatch) -> None:
    observed: list[object] = []

    class BackendSpy:
        def __init__(self, *, device, precision):
            observed.append((device, precision))

    monkeypatch.setattr(backend_module, "JAX_AVAILABLE", True)
    monkeypatch.setattr(backend_module, "JAXBackend", BackendSpy)

    result = get_backend("jax")

    assert isinstance(result, BackendSpy)
    assert observed[0][0] is None


def test_runtime_default_jax_device_uses_the_unqualified_runtime_query(
    monkeypatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    runtime_device = SimpleNamespace(platform="gpu", id=0)

    class FakeConfig:
        @staticmethod
        def update(name: str, enabled: bool) -> None:
            assert (name, enabled) == ("jax_enable_x64", True)

    def devices(*args):
        calls.append(args)
        return [runtime_device]

    fake_jax = SimpleNamespace(config=FakeConfig(), devices=devices)
    monkeypatch.setattr(jax_backend_module, "jax", fake_jax)
    monkeypatch.setattr(jax_backend_module, "jnp", SimpleNamespace())
    monkeypatch.setattr(jax_backend_module, "_load_jax", lambda: True)

    backend = JAXBackend(device=None)

    assert backend.device is runtime_device
    assert calls == [()]


@pytest.mark.parametrize("device", ["cpu", "gpu", "tpu"])
def test_explicit_jax_device_is_strict_and_preserves_runtime_failure(
    monkeypatch,
    device,
) -> None:
    cpu_device = SimpleNamespace(platform="cpu", id=0)
    failure = RuntimeError(f"{device} runtime is unavailable")
    calls: list[object] = []

    class FakeConfig:
        @staticmethod
        def update(name: str, enabled: bool) -> None:
            assert (name, enabled) == ("jax_enable_x64", True)

    def devices(requested=None):
        calls.append(requested)
        if requested == device:
            raise failure
        if requested == "cpu":
            return [cpu_device]
        raise AssertionError(f"unexpected device query {requested!r}")

    fake_jax = SimpleNamespace(config=FakeConfig(), devices=devices)
    monkeypatch.setattr(jax_backend_module, "jax", fake_jax)
    monkeypatch.setattr(jax_backend_module, "jnp", SimpleNamespace())
    monkeypatch.setattr(jax_backend_module, "_load_jax", lambda: True)

    with pytest.raises(BackendNotAvailableError, match=device) as exc_info:
        JAXBackend(device=device)

    assert exc_info.value.__cause__ is failure
    assert calls == [device]


@pytest.mark.parametrize(
    ("backend_name", "kwargs", "required_device"),
    [
        ("gpu", {}, "gpu"),
        ("tpu", {}, "tpu"),
        ("jax", {"device": "cpu"}, "cpu"),
        ("jax", {"device": "gpu"}, "gpu"),
        ("jax", {"device": "tpu"}, "tpu"),
    ],
)
def test_factory_preserves_strict_device_runtime_failure(
    monkeypatch,
    backend_name,
    kwargs,
    required_device,
) -> None:
    failure = RuntimeError(f"{required_device} plugin failed")

    class FakeConfig:
        @staticmethod
        def update(name: str, enabled: bool) -> None:
            assert (name, enabled) == ("jax_enable_x64", True)

    def devices(device=None):
        assert device == required_device
        raise failure

    fake_jax = SimpleNamespace(config=FakeConfig(), devices=devices)
    monkeypatch.setattr(backend_module, "JAX_AVAILABLE", True)
    monkeypatch.setattr(jax_backend_module, "jax", fake_jax)
    monkeypatch.setattr(jax_backend_module, "jnp", SimpleNamespace())
    monkeypatch.setattr(jax_backend_module, "_load_jax", lambda: True)

    with pytest.raises(
        BackendNotAvailableError,
        match=required_device,
    ) as exc_info:
        get_backend(backend_name, **kwargs)

    assert exc_info.value.__cause__ is failure


@pytest.mark.parametrize("returned_devices", [[], [SimpleNamespace(platform="cpu")]])
def test_strict_gpu_result_never_falls_back_or_accepts_the_wrong_platform(
    monkeypatch,
    returned_devices,
) -> None:
    calls: list[object] = []

    class FakeConfig:
        @staticmethod
        def update(name: str, enabled: bool) -> None:
            assert (name, enabled) == ("jax_enable_x64", True)

    def devices(device=None):
        calls.append(device)
        return returned_devices

    fake_jax = SimpleNamespace(config=FakeConfig(), devices=devices)
    monkeypatch.setattr(jax_backend_module, "jax", fake_jax)
    monkeypatch.setattr(jax_backend_module, "jnp", SimpleNamespace())
    monkeypatch.setattr(jax_backend_module, "_load_jax", lambda: True)

    with pytest.raises(BackendNotAvailableError, match="gpu"):
        JAXBackend(device="gpu")

    assert calls == ["gpu"]


def test_runtime_default_rejects_an_unreportable_device_platform(monkeypatch) -> None:
    class FakeConfig:
        @staticmethod
        def update(name: str, enabled: bool) -> None:
            assert (name, enabled) == ("jax_enable_x64", True)

    fake_jax = SimpleNamespace(
        config=FakeConfig(),
        devices=lambda: [SimpleNamespace(platform="unknown-plugin")],
    )
    monkeypatch.setattr(jax_backend_module, "jax", fake_jax)
    monkeypatch.setattr(jax_backend_module, "jnp", SimpleNamespace())
    monkeypatch.setattr(jax_backend_module, "_load_jax", lambda: True)

    with pytest.raises(BackendNotAvailableError, match="unsupported device platform"):
        JAXBackend(device=None)


@pytest.mark.parametrize(("broken", "working"), [("gpu", "tpu"), ("tpu", "gpu")])
def test_discovery_isolates_gpu_and_tpu_runtime_queries(
    monkeypatch,
    broken,
    working,
) -> None:
    fake_jax = ModuleType("jax")

    def devices(device=None):
        if device == broken:
            raise RuntimeError(f"{broken} plugin is broken")
        if device == working:
            return [SimpleNamespace(platform=working)]
        return [SimpleNamespace(platform="cpu")]

    fake_jax.devices = devices  # type: ignore[attr-defined]
    monkeypatch.setattr(backend_module, "JAX_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    availability = list_backends()

    assert availability["jax"] is True
    assert availability[f"jax_{broken}"] is False
    assert availability[f"jax_{working}"] is True


def test_broken_jax_import_is_reported_as_a_typed_backend_failure(
    monkeypatch,
) -> None:
    failure = RuntimeError("incompatible jaxlib")

    def broken_runtime() -> bool:
        raise failure

    monkeypatch.setattr(jax_backend_module, "_load_jax", broken_runtime)

    with pytest.raises(BackendNotAvailableError, match="JAX runtime") as exc_info:
        JAXBackend(device=None)

    assert exc_info.value.__cause__ is failure


def test_broken_jax_x64_initialization_is_a_typed_backend_failure(
    monkeypatch,
) -> None:
    failure = RuntimeError("configuration is frozen")

    class BrokenConfig:
        @staticmethod
        def update(name: str, enabled: bool) -> None:
            raise failure

    monkeypatch.setattr(
        jax_backend_module, "jax", SimpleNamespace(config=BrokenConfig())
    )
    monkeypatch.setattr(jax_backend_module, "jnp", SimpleNamespace())
    monkeypatch.setattr(jax_backend_module, "_load_jax", lambda: True)

    with pytest.raises(BackendNotAvailableError, match="x64") as exc_info:
        JAXBackend(device=None)

    assert exc_info.value.__cause__ is failure
