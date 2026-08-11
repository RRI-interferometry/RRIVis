"""Strict, non-gating GPU preflight for the PERF-001 P-e environment.

This command emits machine-readable facts only after every prerequisite passes.
It does not write a benchmark artifact and its success is not accelerator
evidence.  Missing or incompatible hardware is a failure, never a skip.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "radiosim.perf001.gpu_preflight.v1"
EXPECTED_WHEEL_VERSIONS = {
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "jax-cuda13-plugin": "0.10.2",
    "jax-cuda13-pjrt": "0.10.2",
}
CUDA_RUNTIME_DISTRIBUTION = "nvidia-cuda-runtime"
MINIMUM_DRIVER_MAJOR = 580
MINIMUM_COMPUTE_CAPABILITY = 7.5
MINIMUM_CUDA_RUNTIME_MAJOR = 13
PARITY_RTOL = 1e-12
PARITY_ATOL_SCALE = 1e-12
MIB = 1024 * 1024

_NVIDIA_QUERY = (
    "index,uuid,name,driver_version,compute_cap,memory.total,memory.used,"
    "memory.free,pci.bus_id"
)
_ALLOCATOR_ENVIRONMENT_KEYS = (
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "TF_GPU_ALLOCATOR",
)


class PreflightError(RuntimeError):
    """A required property of the PERF-001 GPU host did not hold."""


@dataclass(frozen=True, slots=True)
class NvidiaGpu:
    """One physical NVIDIA GPU inventory row (raw UUID is never serialized)."""

    index: int
    uuid: str
    model: str
    driver_version: str
    compute_capability: str
    total_memory_bytes: int
    used_memory_bytes: int
    free_memory_bytes: int
    pci_bus_id: str


@dataclass(frozen=True, slots=True)
class JaxProbe:
    """JAX selection, precision, compilation, and correctness observations."""

    default_backend: str
    device_platform: str
    device_kind: str
    device_id: int
    visible_device_count: int
    x64_enabled: bool
    input_dtype: str
    output_dtype: str
    compiled: bool
    synchronized: bool
    reference_scale: float
    max_absolute_deviation: float
    max_relative_deviation: float
    tolerance_rtol: float
    tolerance_atol: float
    within_tolerance: bool


@dataclass(frozen=True, slots=True)
class JaxSmoke:
    """Result of the compiled complex128 contraction itself."""

    input_dtype: str
    output_dtype: str
    compiled: bool
    synchronized: bool
    reference_scale: float
    max_absolute_deviation: float
    max_relative_deviation: float
    tolerance_rtol: float
    tolerance_atol: float
    within_tolerance: bool


CommandRunner = Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class PreflightDependencies:
    """Injectable host boundaries used by unit tests and the real CLI."""

    repository_root: Path
    environ: Mapping[str, str]
    system: Callable[[], str]
    machine: Callable[[], str]
    command_runner: CommandRunner
    distribution_version: Callable[[str], str]
    jax_probe: Callable[[], JaxProbe]


def _run_command(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PreflightError(f"command could not run: {command[0]}") from error


def _default_dependencies() -> PreflightDependencies:
    return PreflightDependencies(
        repository_root=REPOSITORY_ROOT,
        environ=os.environ,
        system=platform.system,
        machine=platform.machine,
        command_runner=_run_command,
        distribution_version=importlib.metadata.version,
        jax_probe=_probe_jax,
    )


def _require_success(
    completed: subprocess.CompletedProcess[str], *, operation: str
) -> str:
    if completed.returncode != 0:
        raise PreflightError(f"{operation} failed")
    return completed.stdout


def _check_platform(dependencies: PreflightDependencies) -> dict[str, str]:
    system = dependencies.system()
    machine = dependencies.machine()
    if system != "Linux":
        raise PreflightError(f"GPU preflight requires Linux, found {system!r}")
    if machine != "x86_64":
        raise PreflightError(
            f"GPU preflight requires Linux x86_64, found machine {machine!r}"
        )
    return {"system": system, "machine": machine}


def _check_pixi_environment(dependencies: PreflightDependencies) -> dict[str, str]:
    environ = dependencies.environ
    root = dependencies.repository_root.resolve()
    manifest = root / "pixi.toml"
    if environ.get("PIXI_ENVIRONMENT_NAME") != "gpu":
        raise PreflightError("preflight must run inside the gpu Pixi environment")

    try:
        selected_root = Path(environ["PIXI_PROJECT_ROOT"]).resolve()
    except KeyError as error:
        raise PreflightError("PIXI_PROJECT_ROOT is not set") from error
    if selected_root != root:
        raise PreflightError("PIXI_PROJECT_ROOT does not name this checkout")

    try:
        selected_manifest = Path(environ["PIXI_PROJECT_MANIFEST"]).resolve()
    except KeyError as error:
        raise PreflightError("PIXI_PROJECT_MANIFEST is not set") from error
    if selected_manifest != manifest:
        raise PreflightError("PIXI_PROJECT_MANIFEST does not name pixi.toml")

    if environ.get("LD_LIBRARY_PATH"):
        raise PreflightError(
            "LD_LIBRARY_PATH must be unset for the self-contained CUDA wheel"
        )

    pixi_executable = environ.get("PIXI_EXE")
    if not pixi_executable:
        raise PreflightError("PIXI_EXE is not set by the gpu Pixi environment")
    lock_check = dependencies.command_runner(
        (
            pixi_executable,
            "lock",
            "--check",
            "--no-install",
            "--manifest-path",
            str(manifest),
        )
    )
    _require_success(lock_check, operation="current pixi.lock check")
    return {"environment": "gpu", "manifest": "pixi.toml"}


def _check_source(dependencies: PreflightDependencies) -> dict[str, object]:
    status = dependencies.command_runner(
        ("git", "status", "--porcelain=v1", "--untracked-files=all")
    )
    status_text = _require_success(status, operation="git status")
    if status_text.strip():
        raise PreflightError("GPU preflight requires an exact clean source checkout")

    revision = dependencies.command_runner(("git", "rev-parse", "HEAD"))
    commit = _require_success(revision, operation="git source identity").strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise PreflightError("git source identity is not a full commit SHA")

    lock_path = dependencies.repository_root / "pixi.lock"
    try:
        lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    except OSError as error:
        raise PreflightError("pixi.lock could not be read") from error
    return {"commit": commit, "lock_sha256": lock_sha256, "clean": True}


def _parse_nonnegative_int(value: str, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise PreflightError("nvidia-smi must report numeric memory values") from error
    if parsed < 0:
        raise PreflightError(f"nvidia-smi {field_name} must be non-negative")
    return parsed


def _parse_nvidia_inventory(stdout: str) -> tuple[NvidiaGpu, ...]:
    rows = list(csv.reader(stdout.splitlines(), skipinitialspace=True))
    if not rows:
        raise PreflightError("nvidia-smi must report at least one NVIDIA GPU")

    inventory: list[NvidiaGpu] = []
    for row in rows:
        if len(row) != 9:
            raise PreflightError("each nvidia-smi GPU row must contain nine fields")
        index_text, uuid, model, driver, capability, total, used, free, pci = (
            item.strip() for item in row
        )
        try:
            index = int(index_text)
            driver_major = int(driver.split(".", maxsplit=1)[0])
            capability_value = float(capability)
        except ValueError as error:
            raise PreflightError(
                "nvidia-smi index, driver, and compute capability must be numeric"
            ) from error
        if index < 0 or not uuid or not model or not pci:
            raise PreflightError("nvidia-smi returned incomplete GPU identity")
        if driver_major < 0 or not math.isfinite(capability_value):
            raise PreflightError(
                "nvidia-smi driver and compute capability must be finite"
            )

        total_mib = _parse_nonnegative_int(total, field_name="memory.total")
        used_mib = _parse_nonnegative_int(used, field_name="memory.used")
        free_mib = _parse_nonnegative_int(free, field_name="memory.free")
        if total_mib <= 0 or used_mib + free_mib > total_mib:
            raise PreflightError("nvidia-smi returned inconsistent numeric memory")
        inventory.append(
            NvidiaGpu(
                index=index,
                uuid=uuid,
                model=model,
                driver_version=driver,
                compute_capability=capability,
                total_memory_bytes=total_mib * MIB,
                used_memory_bytes=used_mib * MIB,
                free_memory_bytes=free_mib * MIB,
                pci_bus_id=pci,
            )
        )
    return tuple(inventory)


def _query_nvidia(
    dependencies: PreflightDependencies,
) -> tuple[NvidiaGpu, ...]:
    query = dependencies.command_runner(
        (
            "nvidia-smi",
            f"--query-gpu={_NVIDIA_QUERY}",
            "--format=csv,noheader,nounits",
        )
    )
    query_stdout = _require_success(query, operation="nvidia-smi GPU query")
    inventory = _parse_nvidia_inventory(query_stdout)

    summary = dependencies.command_runner(("nvidia-smi",))
    summary_stdout = _require_success(summary, operation="nvidia-smi summary")
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)", summary_stdout)
    if match is None:
        raise PreflightError("nvidia-smi did not report a CUDA runtime")
    runtime_version = match.group(1)
    if int(runtime_version.split(".", maxsplit=1)[0]) < MINIMUM_CUDA_RUNTIME_MAJOR:
        raise PreflightError(
            f"CUDA runtime must be at least {MINIMUM_CUDA_RUNTIME_MAJOR}"
        )
    return inventory


def _select_gpu(
    inventory: tuple[NvidiaGpu, ...], environ: Mapping[str, str]
) -> NvidiaGpu:
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        if len(inventory) != 1:
            raise PreflightError(
                "cannot identify the selected GPU from an unmasked multi-GPU host"
            )
        return inventory[0]

    selectors = [item.strip() for item in visible.split(",") if item.strip()]
    if len(selectors) != 1:
        raise PreflightError("CUDA_VISIBLE_DEVICES must select exactly one GPU")
    selector = selectors[0]
    matches = [
        gpu for gpu in inventory if selector == str(gpu.index) or selector == gpu.uuid
    ]
    if len(matches) != 1:
        raise PreflightError("CUDA_VISIBLE_DEVICES does not identify a selected GPU")
    return matches[0]


def _validate_selected_gpu(gpu: NvidiaGpu) -> None:
    driver_major = int(gpu.driver_version.split(".", maxsplit=1)[0])
    if driver_major < MINIMUM_DRIVER_MAJOR:
        raise PreflightError(f"NVIDIA driver must be at least {MINIMUM_DRIVER_MAJOR}")
    if float(gpu.compute_capability) < MINIMUM_COMPUTE_CAPABILITY:
        raise PreflightError(
            f"NVIDIA compute capability must be at least {MINIMUM_COMPUTE_CAPABILITY}"
        )


def _check_wheels(dependencies: PreflightDependencies) -> dict[str, str]:
    observed: dict[str, str] = {}
    for distribution, expected in EXPECTED_WHEEL_VERSIONS.items():
        try:
            version = dependencies.distribution_version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise PreflightError(
                f"{distribution} {expected} must be installed"
            ) from error
        if version != expected:
            raise PreflightError(
                f"{distribution} must be exactly {expected}, found {version}"
            )
        observed[distribution] = version
    return observed


def _check_cuda_runtime(dependencies: PreflightDependencies) -> str:
    try:
        version = dependencies.distribution_version(CUDA_RUNTIME_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as error:
        raise PreflightError(
            f"{CUDA_RUNTIME_DISTRIBUTION} must be installed"
        ) from error
    try:
        major = int(version.split(".", maxsplit=1)[0])
    except ValueError as error:
        raise PreflightError("CUDA runtime wheel version must be numeric") from error
    if major != MINIMUM_CUDA_RUNTIME_MAJOR:
        raise PreflightError(
            f"CUDA runtime wheel must be major {MINIMUM_CUDA_RUNTIME_MAJOR}, "
            f"found {version}"
        )
    return f"CUDA {version}"


def _execute_complex128_smoke(jax_module: Any, device: Any) -> JaxSmoke:
    """Compile and synchronize one deterministic complex128 source contraction."""
    numpy = importlib.import_module("numpy")
    jax_module.config.update("jax_enable_x64", True)
    jnp = jax_module.numpy

    phase_numpy = numpy.asarray(
        [
            [1.0 + 2.0j, -0.5 + 0.25j, 2.0 - 1.0j],
            [-1.0 + 0.5j, 0.75 - 1.25j, 1.5 + 0.0j],
        ],
        dtype=numpy.complex128,
    )
    coherency_numpy = numpy.asarray(
        [
            [[1.0 + 0.0j, 2.0 - 1.0j], [0.5 + 0.5j, -2.0j], [3.0, -1.0]],
            [[-0.5j, 1.0], [2.0 + 1.0j, 0.25], [-3.0j, 0.5 - 0.5j]],
        ],
        dtype=numpy.complex128,
    )
    reference = numpy.einsum("bs,bsp->bp", phase_numpy, coherency_numpy)
    phase = jax_module.device_put(jnp.asarray(phase_numpy), device=device)
    coherency = jax_module.device_put(jnp.asarray(coherency_numpy), device=device)

    def contraction(lhs: Any, rhs: Any) -> Any:
        return jnp.einsum("bs,bsp->bp", lhs, rhs)

    executable = jax_module.jit(contraction).lower(phase, coherency).compile()
    candidate = executable(phase, coherency)
    candidate.block_until_ready()
    output_devices = candidate.devices()
    if output_devices != {device}:
        raise PreflightError("compiled smoke did not execute on the selected device")
    observed = numpy.asarray(jax_module.device_get(candidate))

    difference = numpy.abs(observed - reference)
    reference_magnitude = numpy.abs(reference)
    reference_scale = float(max(1.0, float(numpy.max(reference_magnitude))))
    tolerance_atol = PARITY_ATOL_SCALE * reference_scale
    allowed = tolerance_atol + PARITY_RTOL * reference_magnitude
    with numpy.errstate(divide="ignore", invalid="ignore"):
        relative = numpy.where(
            reference_magnitude > 0.0,
            difference / reference_magnitude,
            difference,
        )
    return JaxSmoke(
        input_dtype=str(phase.dtype),
        output_dtype=str(candidate.dtype),
        compiled=True,
        synchronized=True,
        reference_scale=reference_scale,
        max_absolute_deviation=float(numpy.max(difference)),
        max_relative_deviation=float(numpy.max(relative)),
        tolerance_rtol=PARITY_RTOL,
        tolerance_atol=tolerance_atol,
        within_tolerance=bool(numpy.all(difference <= allowed)),
    )


def _probe_jax() -> JaxProbe:
    try:
        jax = importlib.import_module("jax")
        jax.config.update("jax_enable_x64", True)
        devices = tuple(jax.devices("gpu"))
    except Exception as error:
        raise PreflightError("JAX could not initialize the selected GPU") from error
    if len(devices) != 1:
        raise PreflightError("preflight requires exactly one JAX-visible GPU")
    device = devices[0]
    try:
        smoke = _execute_complex128_smoke(jax, device)
        return JaxProbe(
            default_backend=str(jax.default_backend()),
            device_platform=str(device.platform),
            device_kind=str(device.device_kind),
            device_id=int(device.id),
            visible_device_count=len(devices),
            x64_enabled=bool(jax.config.jax_enable_x64),
            **{
                field: getattr(smoke, field)
                for field in (
                    "input_dtype",
                    "output_dtype",
                    "compiled",
                    "synchronized",
                    "reference_scale",
                    "max_absolute_deviation",
                    "max_relative_deviation",
                    "tolerance_rtol",
                    "tolerance_atol",
                    "within_tolerance",
                )
            },
        )
    except PreflightError:
        raise
    except Exception as error:
        raise PreflightError(
            "JAX complex128 contraction could not compile and synchronize"
        ) from error


def _validate_jax_probe(probe: JaxProbe) -> None:
    if type(probe.visible_device_count) is not int or probe.visible_device_count != 1:
        raise PreflightError("preflight requires exactly one JAX-visible GPU")
    if probe.default_backend != "gpu":
        raise PreflightError("JAX selected a CPU fallback instead of the GPU")
    if probe.device_platform != "gpu":
        raise PreflightError("JAX device platform must equal gpu")
    if not probe.device_kind:
        raise PreflightError("JAX device kind is empty")
    if type(probe.device_id) is not int or probe.device_id < 0:
        raise PreflightError("JAX device id must be a non-negative integer")
    if probe.x64_enabled is not True:
        raise PreflightError("JAX x64 must be enabled")
    if probe.input_dtype != "complex128":
        raise PreflightError("smoke must retain complex128 input")
    if probe.output_dtype != "complex128":
        raise PreflightError("smoke must produce complex128 output")
    if probe.compiled is not True:
        raise PreflightError("smoke must be explicitly compiled")
    if probe.synchronized is not True:
        raise PreflightError("smoke must be synchronized on its output")
    for name in (
        "reference_scale",
        "max_absolute_deviation",
        "max_relative_deviation",
    ):
        value = getattr(probe, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise PreflightError(f"smoke {name} must be finite and non-negative")
    if probe.tolerance_rtol != PARITY_RTOL:
        raise PreflightError("smoke rtol differs from the existing predicate")
    expected_atol = PARITY_ATOL_SCALE * max(1.0, probe.reference_scale)
    if probe.tolerance_atol != expected_atol:
        raise PreflightError("smoke atol differs from the existing predicate")
    if probe.within_tolerance is not True:
        raise PreflightError("smoke failed the existing NumPy tolerance")


def _allocator_environment(environ: Mapping[str, str]) -> dict[str, str]:
    return {
        key: environ[key] for key in _ALLOCATOR_ENVIRONMENT_KEYS if environ.get(key)
    }


def run_preflight(
    dependencies: PreflightDependencies | None = None,
) -> dict[str, object]:
    """Run every strict prerequisite and return a non-evidentiary fact report."""
    selected_dependencies = dependencies or _default_dependencies()
    host = _check_platform(selected_dependencies)
    pixi = _check_pixi_environment(selected_dependencies)
    source = _check_source(selected_dependencies)
    inventory = _query_nvidia(selected_dependencies)
    selected_gpu = _select_gpu(inventory, selected_dependencies.environ)
    _validate_selected_gpu(selected_gpu)
    wheels = _check_wheels(selected_dependencies)
    runtime = _check_cuda_runtime(selected_dependencies)
    probe = selected_dependencies.jax_probe()
    _validate_jax_probe(probe)

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "GPU readiness only; not PERF-001 evidence",
        "source": source,
        "pixi": pixi,
        "host": host,
        "accelerator": {
            "vendor": "NVIDIA",
            "model": selected_gpu.model,
            "runtime": runtime,
            "driver_version": selected_gpu.driver_version,
            "compute_capability": selected_gpu.compute_capability,
            "total_memory_bytes": selected_gpu.total_memory_bytes,
            "used_memory_bytes": selected_gpu.used_memory_bytes,
            "free_memory_bytes": selected_gpu.free_memory_bytes,
            "pci_bus_id": selected_gpu.pci_bus_id,
            "device_uuid_sha256": hashlib.sha256(
                selected_gpu.uuid.encode("utf-8")
            ).hexdigest(),
            "jax_device_id": probe.device_id,
            "jax_device_kind": probe.device_kind,
            "jax_device_platform": probe.device_platform,
            "visible_device_count": probe.visible_device_count,
        },
        "wheels": wheels,
        "allocator_environment": _allocator_environment(selected_dependencies.environ),
        "smoke": {
            "x64_enabled": probe.x64_enabled,
            "input_dtype": probe.input_dtype,
            "output_dtype": probe.output_dtype,
            "compiled": probe.compiled,
            "synchronized": probe.synchronized,
            "reference_scale": probe.reference_scale,
            "max_absolute_deviation": probe.max_absolute_deviation,
            "max_relative_deviation": probe.max_relative_deviation,
            "tolerance_rtol": probe.tolerance_rtol,
            "tolerance_atol": probe.tolerance_atol,
            "within_tolerance": probe.within_tolerance,
        },
    }


def main() -> int:
    """Print the strict report, or one actionable failure, and return status."""
    try:
        report = run_preflight()
    except PreflightError as error:
        print(f"GPU preflight failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
