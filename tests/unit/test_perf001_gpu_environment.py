"""PERF-001 P-e GPU-environment and strict-preflight contracts.

These tests exercise infrastructure only.  They never skip for missing hardware,
never create a benchmark record, and never turn a mocked preflight into GPU
evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools" / "wp7_gpu_preflight.py"
CPU_ENVIRONMENT_PACKAGE_DIGESTS = {
    "default": "5953d45ac40bf62e8f0f4d3e0bde218eabea0fcf12df9d8e81e639ecd4fb2af6",
    "py312": "a197dc7baefafe6fa87a71526198f2c1768bfe42d9d13c21efaff7afec66b065",
    "crossval": "96a12488773be0662bc7babc38bcbc281060d0f9fd2d0cf8046ca16daf4b8072",
}


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("wp7_gpu_preflight", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    return _load_tool()


def _package_identity_digest(lock: dict[str, Any], environment: str) -> str:
    packages = lock["environments"][environment]["packages"]
    identity = {
        platform_name: sorted(next(iter(reference.items())) for reference in refs)
        for platform_name, refs in packages.items()
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def test_gpu_manifest_is_linux_only_isolated_and_non_gating() -> None:
    manifest = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
    feature = manifest["feature"]["jax-gpu"]
    assert feature["platforms"] == ["linux-64"]
    assert feature["pypi-dependencies"] == {
        "jax": {"version": "==0.10.2", "extras": ["cuda13"]}
    }
    assert "jax-cpu" not in manifest["environments"]["gpu"]["features"]
    assert manifest["environments"]["gpu"] == {
        "features": ["py311", "jax-gpu"],
        "solve-group": "gpu-py311",
    }
    assert manifest["tasks"]["bench"] == (
        "python -m pytest tests/performance/ -m performance"
    )
    assert feature["tasks"]["gpu-preflight"] == ("python tools/wp7_gpu_preflight.py")
    assert feature["tasks"]["bench-gpu"] == {
        "cmd": "python -m pytest tests/performance/ -m performance",
        "depends-on": ["gpu-preflight"],
        "env": {
            "RADIOSIM_BENCHMARK_BACKENDS": "numpy,gpu",
            "RADIOSIM_REQUIRE_ACCELERATOR": "gpu",
        },
    }


def test_gpu_lock_adds_only_a_linux_cuda_stack() -> None:
    lock = yaml.safe_load((ROOT / "pixi.lock").read_text(encoding="utf-8"))
    assert set(lock["environments"]["gpu"]["packages"]) == {"linux-64"}
    references = lock["environments"]["gpu"]["packages"]["linux-64"]
    filenames = {
        next(iter(reference.values())).rsplit("/", maxsplit=1)[-1].replace("_", "-")
        for reference in references
    }
    for distribution in (
        "jax",
        "jaxlib",
        "jax-cuda13-plugin",
        "jax-cuda13-pjrt",
    ):
        matches = [
            filename
            for filename in filenames
            if filename.startswith(f"{distribution}-0.10.2-")
        ]
        assert len(matches) == 1, (distribution, matches)
    assert not any(
        reference.get("conda", "").rsplit("/", maxsplit=1)[-1].startswith("jaxlib-")
        for reference in references
    )


def test_gpu_lock_does_not_move_any_cpu_environment_package_identity() -> None:
    lock = yaml.safe_load((ROOT / "pixi.lock").read_text(encoding="utf-8"))
    observed = {
        environment: _package_identity_digest(lock, environment)
        for environment in CPU_ENVIRONMENT_PACKAGE_DIGESTS
    }
    assert observed == CPU_ENVIRONMENT_PACKAGE_DIGESTS


class _ScenarioRunner:
    def __init__(self) -> None:
        self.dirty = False
        self.status_returncode = 0
        self.revision_returncode = 0
        self.revision_stdout = "a" * 40 + "\n"
        self.lock_returncode = 0
        self.query_returncode = 0
        self.summary_returncode = 0
        self.query_stdout = (
            "0, GPU-secret-uuid, NVIDIA H100 PCIe, 580.82.07, 9.0, "
            "81559, 1024, 80535, 00000000:01:00.0\n"
        )
        self.summary_stdout = (
            "NVIDIA-SMI 580.82.07  Driver Version: 580.82.07  CUDA Version: 13.0\n"
        )

    def __call__(self, command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        if command[:2] == ("git", "status"):
            stdout = " M pixi.toml\n" if self.dirty else ""
            return subprocess.CompletedProcess(
                command,
                self.status_returncode,
                stdout,
                "status failed" if self.status_returncode else "",
            )
        if command[:3] == ("git", "rev-parse", "HEAD"):
            return subprocess.CompletedProcess(
                command,
                self.revision_returncode,
                self.revision_stdout,
                "revision failed" if self.revision_returncode else "",
            )
        if "lock" in command and "--check" in command:
            return subprocess.CompletedProcess(
                command,
                self.lock_returncode,
                "",
                "lock is stale" if self.lock_returncode else "",
            )
        if (
            command
            and command[0] == "nvidia-smi"
            and any(item.startswith("--query-gpu=") for item in command)
        ):
            return subprocess.CompletedProcess(
                command,
                self.query_returncode,
                self.query_stdout,
                "query failed" if self.query_returncode else "",
            )
        if command == ("nvidia-smi",):
            return subprocess.CompletedProcess(
                command,
                self.summary_returncode,
                self.summary_stdout,
                "summary failed" if self.summary_returncode else "",
            )
        raise AssertionError(f"unexpected command: {command}")


def _valid_probe(tool: ModuleType, **updates: Any) -> object:
    values: dict[str, object] = {
        "default_backend": "gpu",
        "device_platform": "gpu",
        "device_kind": "NVIDIA H100 PCIe",
        "device_id": 0,
        "visible_device_count": 1,
        "x64_enabled": True,
        "input_dtype": "complex128",
        "output_dtype": "complex128",
        "compiled": True,
        "synchronized": True,
        "reference_scale": 32.0,
        "max_absolute_deviation": 1e-14,
        "max_relative_deviation": 1e-15,
        "tolerance_rtol": 1e-12,
        "tolerance_atol": 3.2e-11,
        "within_tolerance": True,
    }
    values.update(updates)
    return tool.JaxProbe(**values)


def _dependencies(
    tool: ModuleType,
    *,
    runner: _ScenarioRunner | None = None,
    environment_updates: dict[str, str] | None = None,
    versions: dict[str, str] | None = None,
    probe_updates: dict[str, object] | None = None,
) -> object:
    scenario = runner or _ScenarioRunner()
    environ = {
        "PIXI_ENVIRONMENT_NAME": "gpu",
        "PIXI_PROJECT_ROOT": str(ROOT),
        "PIXI_PROJECT_MANIFEST": str(ROOT / "pixi.toml"),
        "PIXI_EXE": "/usr/bin/pixi",
        "CUDA_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    environ.update(environment_updates or {})
    expected_versions = {
        "jax": "0.10.2",
        "jaxlib": "0.10.2",
        "jax-cuda13-plugin": "0.10.2",
        "jax-cuda13-pjrt": "0.10.2",
        "nvidia-cuda-runtime": "13.3.29",
    }
    if versions is not None:
        expected_versions = versions

    def distribution_version(name: str) -> str:
        if name not in expected_versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return expected_versions[name]

    return tool.PreflightDependencies(
        repository_root=ROOT,
        environ=environ,
        system=lambda: "Linux",
        machine=lambda: "x86_64",
        command_runner=scenario,
        distribution_version=distribution_version,
        jax_probe=lambda: _valid_probe(tool, **(probe_updates or {})),
    )


def test_valid_mocked_preflight_is_complete_and_never_leaks_uuid(
    tool: ModuleType,
) -> None:
    report = tool.run_preflight(_dependencies(tool))

    assert report["schema_version"] == "radiosim.perf001.gpu_preflight.v1"
    assert report["source"]["commit"] == "a" * 40
    assert report["pixi"]["environment"] == "gpu"
    assert report["accelerator"] == {
        "vendor": "NVIDIA",
        "model": "NVIDIA H100 PCIe",
        "runtime": "CUDA 13.3.29",
        "driver_version": "580.82.07",
        "compute_capability": "9.0",
        "total_memory_bytes": 81559 * 1024 * 1024,
        "used_memory_bytes": 1024 * 1024 * 1024,
        "free_memory_bytes": 80535 * 1024 * 1024,
        "pci_bus_id": "00000000:01:00.0",
        "device_uuid_sha256": (
            "bd76180a0ddebe6f46433d346b697eebe053ac41d3b09e658c6c55fbf95cde77"
        ),
        "jax_device_id": 0,
        "jax_device_kind": "NVIDIA H100 PCIe",
        "jax_device_platform": "gpu",
        "visible_device_count": 1,
    }
    assert report["wheels"] == {
        "jax": "0.10.2",
        "jaxlib": "0.10.2",
        "jax-cuda13-plugin": "0.10.2",
        "jax-cuda13-pjrt": "0.10.2",
    }
    assert report["allocator_environment"] == {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    assert report["smoke"]["compiled"] is True
    assert report["smoke"]["synchronized"] is True
    serialized = json.dumps(report, sort_keys=True)
    assert "GPU-secret-uuid" not in serialized


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("system", lambda: "Darwin", "Linux"),
        ("machine", lambda: "aarch64", "x86_64"),
    ],
)
def test_platform_rejections_are_fatal(
    tool: ModuleType,
    field: str,
    replacement: object,
    message: str,
) -> None:
    dependencies = replace(_dependencies(tool), **{field: replacement})
    with pytest.raises(tool.PreflightError, match=message):
        tool.run_preflight(dependencies)


@pytest.mark.parametrize(
    ("environment_updates", "message"),
    [
        ({"PIXI_ENVIRONMENT_NAME": "default"}, "gpu Pixi environment"),
        ({"PIXI_PROJECT_ROOT": "/tmp/not-radiosim"}, "PIXI_PROJECT_ROOT"),
        ({"PIXI_PROJECT_MANIFEST": "/tmp/pixi.toml"}, "PIXI_PROJECT_MANIFEST"),
        ({"LD_LIBRARY_PATH": "/opt/cuda/lib64"}, "LD_LIBRARY_PATH"),
        ({"CUDA_VISIBLE_DEVICES": "0,1"}, "exactly one GPU"),
        ({"CUDA_VISIBLE_DEVICES": "99"}, "selected GPU"),
    ],
)
def test_environment_rejections_are_fatal(
    tool: ModuleType,
    environment_updates: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(tool.PreflightError, match=message):
        tool.run_preflight(_dependencies(tool, environment_updates=environment_updates))


@pytest.mark.parametrize(
    ("missing_key", "message"),
    [
        ("PIXI_PROJECT_ROOT", "PIXI_PROJECT_ROOT"),
        ("PIXI_PROJECT_MANIFEST", "PIXI_PROJECT_MANIFEST"),
        ("PIXI_EXE", "PIXI_EXE"),
    ],
)
def test_missing_pixi_identity_is_fatal(
    tool: ModuleType, missing_key: str, message: str
) -> None:
    dependencies = _dependencies(tool)
    environment = dict(dependencies.environ)
    environment.pop(missing_key)
    with pytest.raises(tool.PreflightError, match=message):
        tool.run_preflight(replace(dependencies, environ=environment))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dirty", "clean source"),
        ("status", "git status"),
        ("revision", "git source identity"),
        ("invalid-revision", "full commit SHA"),
        ("lock", "current pixi.lock"),
        ("query", "nvidia-smi GPU query"),
        ("summary", "nvidia-smi summary"),
    ],
)
def test_command_failures_are_fatal(
    tool: ModuleType, mutation: str, message: str
) -> None:
    runner = _ScenarioRunner()
    if mutation == "dirty":
        runner.dirty = True
    elif mutation == "status":
        runner.status_returncode = 1
    elif mutation == "revision":
        runner.revision_returncode = 1
    elif mutation == "invalid-revision":
        runner.revision_stdout = "short\n"
    elif mutation == "lock":
        runner.lock_returncode = 1
    elif mutation == "query":
        runner.query_returncode = 1
    else:
        runner.summary_returncode = 1

    with pytest.raises(tool.PreflightError, match=message):
        tool.run_preflight(_dependencies(tool, runner=runner))


@pytest.mark.parametrize(
    ("query_stdout", "summary_stdout", "message"),
    [
        ("", "CUDA Version: 13.0", "at least one NVIDIA GPU"),
        ("not,csv", "CUDA Version: 13.0", "nine fields"),
        (
            "0, GPU-id, Test, 579.99, 9.0, 100, 1, 99, 0000:01:00.0",
            "CUDA Version: 13.0",
            "driver.*580",
        ),
        (
            "0, GPU-id, Test, 580.00, 7.0, 100, 1, 99, 0000:01:00.0",
            "CUDA Version: 13.0",
            "compute capability.*7.5",
        ),
        (
            "0, GPU-id, Test, 580.00, 9.0, unknown, 1, 99, 0000:01:00.0",
            "CUDA Version: 13.0",
            "numeric memory",
        ),
        (
            "0, GPU-id, Test, 580.00, 9.0, 100, 1, 99, 0000:01:00.0",
            "no runtime here",
            "CUDA runtime",
        ),
        (
            "0, GPU-id, Test, 580.00, 9.0, 100, 1, 99, 0000:01:00.0",
            "CUDA Version: 12.9",
            "CUDA runtime.*13",
        ),
    ],
)
def test_nvidia_inventory_rejections_are_fatal(
    tool: ModuleType,
    query_stdout: str,
    summary_stdout: str,
    message: str,
) -> None:
    runner = _ScenarioRunner()
    runner.query_stdout = query_stdout
    runner.summary_stdout = summary_stdout
    with pytest.raises(tool.PreflightError, match=message):
        tool.run_preflight(_dependencies(tool, runner=runner))


@pytest.mark.parametrize(
    ("query_stdout", "message"),
    [
        (
            "x, GPU-id, Test, 580.00, 9.0, 100, 1, 99, 0000:01:00.0",
            "index, driver, and compute capability",
        ),
        (
            "0, , Test, 580.00, 9.0, 100, 1, 99, 0000:01:00.0",
            "incomplete GPU identity",
        ),
        (
            "0, GPU-id, Test, 580.00, nan, 100, 1, 99, 0000:01:00.0",
            "must be finite",
        ),
        (
            "0, GPU-id, Test, 580.00, 9.0, 100, -1, 101, 0000:01:00.0",
            "must be non-negative",
        ),
        (
            "0, GPU-id, Test, 580.00, 9.0, 100, 2, 99, 0000:01:00.0",
            "inconsistent numeric memory",
        ),
    ],
)
def test_malformed_nvidia_fields_are_fatal(
    tool: ModuleType, query_stdout: str, message: str
) -> None:
    runner = _ScenarioRunner()
    runner.query_stdout = query_stdout
    with pytest.raises(tool.PreflightError, match=message):
        tool.run_preflight(_dependencies(tool, runner=runner))


def test_ambiguous_unmasked_inventory_is_rejected(tool: ModuleType) -> None:
    runner = _ScenarioRunner()
    runner.query_stdout += (
        "1, GPU-another-secret, NVIDIA H100 PCIe, 580.82.07, 9.0, "
        "81559, 0, 81559, 00000000:02:00.0\n"
    )
    dependencies = _dependencies(tool, runner=runner)
    environment = dict(dependencies.environ)
    environment.pop("CUDA_VISIBLE_DEVICES")
    with pytest.raises(tool.PreflightError, match="identify the selected GPU"):
        tool.run_preflight(replace(dependencies, environ=environment))


def test_only_the_explicitly_selected_gpu_must_meet_hardware_minima(
    tool: ModuleType,
) -> None:
    runner = _ScenarioRunner()
    runner.query_stdout += (
        "1, GPU-unselected-old, NVIDIA T4, 580.82.07, 7.0, "
        "16384, 0, 16384, 00000000:02:00.0\n"
    )
    report = tool.run_preflight(_dependencies(tool, runner=runner))
    assert report["accelerator"]["model"] == "NVIDIA H100 PCIe"


def test_selected_gpu_can_be_named_by_uuid_without_leaking_it(tool: ModuleType) -> None:
    report = tool.run_preflight(
        _dependencies(
            tool,
            environment_updates={"CUDA_VISIBLE_DEVICES": "GPU-secret-uuid"},
        )
    )
    assert "GPU-secret-uuid" not in json.dumps(report)


def test_missing_or_wrong_cuda_wheel_is_rejected(tool: ModuleType) -> None:
    missing = {
        "jax": "0.10.2",
        "jaxlib": "0.10.2",
        "jax-cuda13-plugin": "0.10.2",
        "nvidia-cuda-runtime": "13.3.29",
    }
    with pytest.raises(tool.PreflightError, match="jax-cuda13-pjrt.*installed"):
        tool.run_preflight(_dependencies(tool, versions=missing))

    wrong = dict(missing, **{"jax-cuda13-pjrt": "0.10.2"})
    wrong["jaxlib"] = "0.10.1"
    with pytest.raises(tool.PreflightError, match="jaxlib.*0.10.2"):
        tool.run_preflight(_dependencies(tool, versions=wrong))


def test_missing_or_wrong_cuda_runtime_wheel_is_rejected(tool: ModuleType) -> None:
    versions = {
        "jax": "0.10.2",
        "jaxlib": "0.10.2",
        "jax-cuda13-plugin": "0.10.2",
        "jax-cuda13-pjrt": "0.10.2",
    }
    with pytest.raises(tool.PreflightError, match="nvidia-cuda-runtime.*installed"):
        tool.run_preflight(_dependencies(tool, versions=versions))

    versions["nvidia-cuda-runtime"] = "12.9.0"
    with pytest.raises(tool.PreflightError, match="runtime wheel.*13"):
        tool.run_preflight(_dependencies(tool, versions=versions))

    versions["nvidia-cuda-runtime"] = "not-a-version"
    with pytest.raises(tool.PreflightError, match="version must be numeric"):
        tool.run_preflight(_dependencies(tool, versions=versions))


@pytest.mark.parametrize(
    ("probe_updates", "message"),
    [
        ({"visible_device_count": 0}, "exactly one JAX-visible GPU"),
        ({"visible_device_count": 2}, "exactly one JAX-visible GPU"),
        ({"visible_device_count": True}, "exactly one JAX-visible GPU"),
        ({"default_backend": "cpu"}, "CPU fallback"),
        ({"device_platform": "cpu"}, "device platform.*gpu"),
        ({"device_kind": ""}, "device kind"),
        ({"device_id": -1}, "device id"),
        ({"x64_enabled": False}, "x64"),
        ({"input_dtype": "complex64"}, "complex128 input"),
        ({"output_dtype": "complex64"}, "complex128 output"),
        ({"compiled": False}, "compiled"),
        ({"synchronized": False}, "synchronized"),
        ({"within_tolerance": False}, "existing NumPy tolerance"),
        ({"tolerance_rtol": 1e-6}, "rtol"),
        ({"tolerance_atol": 1e-6}, "atol"),
        ({"reference_scale": float("nan")}, "reference_scale"),
        ({"max_absolute_deviation": -1.0}, "max_absolute_deviation"),
        ({"max_relative_deviation": float("inf")}, "max_relative_deviation"),
    ],
)
def test_jax_probe_rejections_are_fatal(
    tool: ModuleType, probe_updates: dict[str, object], message: str
) -> None:
    with pytest.raises(tool.PreflightError, match=message):
        tool.run_preflight(_dependencies(tool, probe_updates=probe_updates))


def test_complex128_smoke_really_compiles_and_synchronizes_on_jax_cpu(
    tool: ModuleType,
) -> None:
    """Exercise the real smoke helper without pretending CPU is GPU evidence."""
    import jax

    probe = tool._execute_complex128_smoke(jax, jax.devices("cpu")[0])
    assert probe.input_dtype == "complex128"
    assert probe.output_dtype == "complex128"
    assert probe.compiled is True
    assert probe.synchronized is True
    assert probe.within_tolerance is True


def test_default_jax_probe_turns_initialization_failure_into_preflight_error(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_import(name: str) -> object:
        assert name == "jax"
        raise ImportError("broken JAX")

    monkeypatch.setattr(tool.importlib, "import_module", fail_import)
    with pytest.raises(tool.PreflightError, match="could not initialize"):
        tool._probe_jax()


def test_default_jax_probe_turns_smoke_failure_into_preflight_error(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(jax_enable_x64=False)

    def update(name: str, value: bool) -> None:
        assert name == "jax_enable_x64"
        config.jax_enable_x64 = value

    config.update = update
    device = SimpleNamespace(platform="gpu", device_kind="GPU", id=0)
    fake_jax = SimpleNamespace(
        config=config,
        devices=lambda platform: [device] if platform == "gpu" else [],
    )
    monkeypatch.setattr(
        tool.importlib,
        "import_module",
        lambda name: fake_jax if name == "jax" else None,
    )

    def fail_smoke(jax_module: object, selected_device: object) -> object:
        assert jax_module is fake_jax
        assert selected_device is device
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(tool, "_execute_complex128_smoke", fail_smoke)
    with pytest.raises(tool.PreflightError, match="could not compile and synchronize"):
        tool._probe_jax()


def test_command_launch_failure_is_a_preflight_error(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_run(*args: object, **kwargs: object) -> object:
        raise OSError("missing executable")

    monkeypatch.setattr(tool.subprocess, "run", fail_run)
    with pytest.raises(tool.PreflightError, match="command could not run"):
        tool._run_command(("missing",))


def test_missing_lockfile_is_a_preflight_error(
    tool: ModuleType, tmp_path: Path
) -> None:
    dependencies = replace(_dependencies(tool), repository_root=tmp_path)
    with pytest.raises(tool.PreflightError, match="pixi.lock could not be read"):
        tool._check_source(dependencies)


def test_cli_failure_is_nonzero_and_explicit(
    tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> dict[str, object]:
        raise tool.PreflightError("no supported GPU")

    monkeypatch.setattr(tool, "run_preflight", fail)
    assert tool.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "GPU preflight failed: no supported GPU" in captured.err
