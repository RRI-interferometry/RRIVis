#!/usr/bin/env python3
# pyright: reportMissingTypeStubs=false, reportPrivateUsage=false
"""Generate or strictly validate the retained PERF-001 clean-CPU document.

Importing this module loads only the Python standard library.  ``validate``
keeps that boundary.  ``generate`` performs all clean-source and Pixi checks
before lazily importing RadioSim, NumPy, Astropy, Dask, or JAX-backed code.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import re
import stat
import statistics
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.parser import Parser
from pathlib import Path
from typing import Any, NoReturn, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL_RELATIVE_PATH = Path("tools/wp7_perf001_cpu_evidence.py")
REFERENCE_DIRECTORY = Path("output/benchmarks/reference/perf001")
SOURCE_SNAPSHOT_PATHS = (
    "src/radiosim",
    TOOL_RELATIVE_PATH.as_posix(),
    "pixi.toml",
    "pixi.lock",
    "pyproject.toml",
)
WORKER_RESULT_PREFIX = "RADIOSIM_PERF001_CPU_WORKER="
ACCEPTANCE_CERTIFICATE_SCHEMA = "radiosim.perf001.cpu_acceptance_certificate.v1"
ACCEPTANCE_VERDICT = "CPU_ACCEPTED_P_E_HARDWARE_GATED"
HARNESS_RELATIVE_PATH = Path("src/radiosim/benchmarks/harness.py")
RECORD_RELATIVE_PATH = Path("src/radiosim/benchmarks/record.py")
PERF001_MEMO_RELATIVE_PATH = Path("docs/development/perf001_runtime_mitigations.md")
LIVE_PLAN_RELATIVE_PATH = Path("PostTier8RemediationPlan.md")
FIX_RELATIVE_PATH = Path("Fix.md")
EVIDENCE_REPRODUCTION_SENTINEL = "<!-- PERF001_E_REPRODUCTION_SENTINEL_V1 -->"
ACCEPTANCE_MEMO_STATUS_SENTINEL = "<!-- PERF001_A_MEMO_STATUS_SENTINEL_V1 -->"
ACCEPTANCE_PLAN_STATUS_SENTINEL = (
    "| WP-7 | P-a…P-d implementation and runtime readiness landed; exact-SHA CI "
    "green; retained CPU evidence and whole CPU-slice independent acceptance "
    "pending; P-e infrastructure authorized, evidence blocked on Q4; PERF-001 "
    "remains ROADMAP |"
)
ACCEPTANCE_STATUS_LINE = (
    "CPU ACCEPTED; P-e hardware-gated. PERF-001 remains ROADMAP; supports_gpu "
    "remains false; no accelerator evidence or claim is accepted."
)
ACCEPTANCE_PLAN_STATUS_ROW = f"| WP-7 | {ACCEPTANCE_STATUS_LINE} |"
ACCEPTED_SOURCE_MEMO_SHA256 = (
    "663521aaf5933c57987ab8ae2976477e19347cb8dd6b8de2395e63ee3c3dbdf4"
)
ACCEPTED_SOURCE_PLAN_SHA256 = (
    "be22567a3f2c296e1aa3f3a21554f791020539a143399adbcea4dddac90f1b5f"
)
FIX_PERF001_ROADMAP_ROW = (
    "| PERF-001 | ROADMAP | Accelerator (GPU/TPU) performance remains "
    "undemonstrated: the time and frequency axes are host-side Python loops, "
    "astropy coordinate transforms / horizon masking / Planck conversion / "
    "pyuvdata beam interpolation are host-side by design, the locked JAX build "
    "is CPU-only, and measured JAX-CPU is slower than NumPy on every "
    "benchmarked workload (`output/benchmarks/reference/`). Filed 2026-07-31 at "
    "Tier 6J re-run acceptance per §41 Q4, as the successor to the "
    "accelerator-performance remainder of `RUN-004`; requires GPU/TPU hardware "
    "this environment does not have | post-Tier-7, hardware-gated |"
)
FIX_PERF001_ROADMAP_ROW_SHA256 = (
    "9306bf612ed4856f6e0d822ad62d814bc54a9def9cee80f0ea50f87938d944bc"
)
EVIDENCE_FIXED_DIFF_PATHS = frozenset(
    {HARNESS_RELATIVE_PATH.as_posix(), PERF001_MEMO_RELATIVE_PATH.as_posix()}
)
ACCEPTANCE_DIFF_PATHS = frozenset(
    {PERF001_MEMO_RELATIVE_PATH.as_posix(), LIVE_PLAN_RELATIVE_PATH.as_posix()}
)
_SNAPSHOT_WORKER_BOOTSTRAP = r"""
import json
import os
import subprocess
import sys
import sysconfig
import types
from pathlib import Path

snapshot_root = Path(sys.argv[1]).resolve(strict=True)
repository_root = Path(sys.argv[2]).resolve(strict=True)
approved_source_sha = sys.argv[3]
recorded_at_utc = sys.argv[4]
tool_path = snapshot_root / "tools/wp7_perf001_cpu_evidence.py"
tool_bytes = tool_path.read_bytes()
committed_tool = subprocess.run(
    ["git", "show", f"{approved_source_sha}:tools/wp7_perf001_cpu_evidence.py"],
    cwd=repository_root,
    check=True,
    capture_output=True,
).stdout
if tool_bytes != committed_tool:
    raise RuntimeError("snapshot worker tool bytes differ from approved S")
runtime_paths = [snapshot_root / "src"]
for existing in tuple(sys.path):
    if existing and Path(existing).resolve() not in {
        repository_root,
        repository_root / "src",
    }:
        candidate = Path(existing).resolve()
        if candidate not in runtime_paths:
            runtime_paths.append(candidate)
for scheme_name in ("purelib", "platlib"):
    candidate = Path(sysconfig.get_path(scheme_name)).resolve(strict=True)
    if candidate not in runtime_paths:
        runtime_paths.append(candidate)
sys.path[:] = [str(path) for path in runtime_paths]
module_name = "wp7_perf001_authenticated_snapshot_worker"
module = types.ModuleType(module_name)
module.__file__ = str(tool_path)
module.__package__ = None
sys.modules[module_name] = module
exec(compile(tool_bytes, str(tool_path), "exec"), module.__dict__)
summary = module._worker_generate(
    repository_root=repository_root,
    snapshot_root=snapshot_root,
    approved_source_sha=approved_source_sha,
    recorded_at_utc=recorded_at_utc,
)
print("RADIOSIM_PERF001_CPU_WORKER=" + json.dumps(summary, sort_keys=True))
"""

PIXI_MANIFEST_SHA256 = (
    "17faff8d6b3f0a37f08b68f600945436fa2ce9b9c97bd6131c025ea12deff899"
)
PIXI_LOCK_SHA256 = "3a8dc9136be95e9ccaff6cb03ad4d7aaf50e54eafa112cb24726a98b40b3fd17"
CPU_ENVIRONMENT_PACKAGE_SHA256 = {
    "default": "5953d45ac40bf62e8f0f4d3e0bde218eabea0fcf12df9d8e81e639ecd4fb2af6",
    "py312": "a197dc7baefafe6fa87a71526198f2c1768bfe42d9d13c21efaff7afec66b065",
}

DOCUMENT_SCHEMA = "radiosim.benchmark.perf001.v1"
PROVENANCE_SCHEMA = "radiosim.benchmark.perf001.provenance.v1"
WORKLOAD_SCHEMA = "radiosim.benchmark.perf001.workload.v2"
MEMORY_SCHEMA = "radiosim.benchmark.perf001.memory_scaling.v2"
SOLVER_MEMORY_SCHEMA = "radiosim.benchmark.perf001.solver_memory.v1"
RETRACING_SCHEMA = "radiosim.benchmark.perf001.retracing.v2"
BACKEND_SCHEMA = "radiosim.benchmark.perf001.backend_resolution.v1"
CONTROL_SCHEMA = "radiosim.perf001.control.backend_resolution.v1"
TARGET_KERNEL_PAIRS = 131072

CPU_BACKENDS = ("numpy", "jax", "dask")
CPU_WORKLOADS = (
    "point_unpolarized_1time_2freq",
    "point_polarized_2times",
    "point_gaussian_morphology",
    "healpix_scalar",
    "healpix_polarized",
    "hybrid_point_plus_healpix",
    "heterogeneous_receptor_bases",
    "point_scaled_4096_sources_4times",
)
CPU_CANONICAL_INPUT_IDENTITIES = {
    "workload:point_unpolarized_1time_2freq": "314ed5f1d7aab50d7fe06c7edbcf982b3b8e1cb59006fafce419766cd4ab9073",
    "workload:point_polarized_2times": "2c6e7ff0921ea65b12c9e054274dfff6a28a2d46b1b8d4c18e66cd330eb870a3",
    "workload:point_gaussian_morphology": "109d28b9e32659720356cb471598f271bdf824fb0db01b833e5150b0116eb931",
    "workload:healpix_scalar": "89907785f42dedd6dfe28364f188c6a7a6202e7527bfecdb880112c53b713b11",
    "workload:healpix_polarized": "11be16918ba4fb9ef5454555e094b98f068b460383cc4e83139f033ea5b4549c",
    "workload:hybrid_point_plus_healpix": "e64a04d2c9489ce030000b2d31726e8652f409e99a9da5b284e406b669662e0a",
    "workload:heterogeneous_receptor_bases": "a7247a38af8b07135fcd2f92ab747635039096fd9bcb4a4f0d4289bdd13d1887",
    "workload:point_scaled_4096_sources_4times": "98af9359065e69823b2dd32f4d4fce93897e0e33dd7d808c950bbeed9bbd9b44",
    "memory:p-a-memory-b100-s100-v1": "eb1860c5f7617c8fe621c0fb3f521157ae451160259a0313972a7455c4a22ff4",
    "memory:p-a-memory-b200-s200-v1": "92819addedf4bf3609f0390dcfca8cb1656e18d2f6ea43f903a277263e155dd6",
    "memory:p-a-memory-b400-s400-v1": "6f6804e130a570f4d18d08ee762a506a751943f0ef0df46cbcb788b1725cb0fb",
    "memory:p-a-memory-b800-s800-v1": "3558b8f3d72771bade10b639109a4a094635684de06a8c42ea3a9784190043bd",
    "solver-memory:point": "50b9cbcd620e5bf7b6d8fe09cf0b7fcca21db0250ee64d7f4a864a9ad7434774",
    "solver-memory:healpix": "e9d2c2f2890af95aa32b48c56b9c8224396efb549315290ef60fadcd9a421b65",
    "retracing:synthetic_wrapper": "8f55ec9114bca7955b49f7263b963a22f18ed3ad68cab6c7e2ac851de2945eb1",
    "retracing:point": "2e9ac389c6357d6ca94d43c052f68b282acc5ab40cf6c0fc757f9b9dbf889940",
    "retracing:healpix": "8aede744badd0183523da8a54d50c77685cec1dd419e97b5181461d8675c867e",
    "backend-resolution:get_backend_auto": "da7e72f9fb6896123598763dcbd5b8b3f5513361592a17468d6edef7327dd27e",
    "backend-resolution:get_device_resources_default": "10b92794ca2e53f9a7ac86208d4b5cb9c3325bc87e6139b946fcd8636a5d84d0",
    "backend-resolution:simulator_setup_auto": "91a3d5bbedf35e5a2b832660d1412f2595d469752eac8358c9de1e7854e2a9f9",
}
WORKLOAD_DIMENSIONS = {
    "point_unpolarized_1time_2freq": (2, 3, 2, 0, 1, 2, "point_sources"),
    "point_polarized_2times": (2, 3, 2, 0, 2, 2, "point_sources"),
    "point_gaussian_morphology": (2, 3, 2, 0, 2, 2, "point_sources"),
    "healpix_scalar": (2, 3, 0, 12, 2, 2, "healpix_map"),
    "healpix_polarized": (2, 3, 0, 12, 2, 2, "healpix_map"),
    "hybrid_point_plus_healpix": (2, 3, 2, 12, 2, 2, "hybrid"),
    "heterogeneous_receptor_bases": (2, 3, 2, 0, 2, 2, "point_sources"),
    "point_scaled_4096_sources_4times": (
        2,
        3,
        4096,
        0,
        4,
        2,
        "point_sources",
    ),
}
MEMORY_FIXTURES = ((100, 100), (200, 200), (400, 400), (800, 800))

CANONICAL_ANTENNA_LAYOUT_BYTES = (
    b"Name Number BeamID E N U Diameter\n"
    b"ANT0 0 0 0.0 0.0 0.0 14.0\n"
    b"ANT1 1 0 14.0 0.0 0.0 14.0\n"
)
CANONICAL_LOCATION = {
    "longitude_deg": 21.4283,
    "latitude_deg": -30.72152,
    "height_m": 1073.0,
}
CANONICAL_BEAM_CONFIGURATION = {
    "mode": "analytic",
    "model": {
        "kind": "circular_aperture",
        "taper": {"kind": "gaussian", "edge_taper_db": 10.0},
    },
}
CANONICAL_HOMOGENEOUS_RECEPTOR_CONFIGURATION: dict[str, object] = {
    "default": {"basis": "linear", "feed_rotation_deg": 0.0},
    "overrides": [],
    "output_basis": "linear",
}
CANONICAL_HETEROGENEOUS_RECEPTOR_CONFIGURATION: dict[str, object] = {
    "default": {"basis": "linear", "feed_rotation_deg": 0.0},
    "overrides": [
        {
            "antenna": {"kind": "number", "number": 1},
            "basis": "circular",
        }
    ],
    "output_basis": "linear",
}

_LOWER_HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_FILENAME = re.compile(
    r"(?P<stamp>[0-9]{8}T[0-9]{6}Z)-"
    r"(?P<system>[a-z0-9]+)-"
    r"(?P<machine>[a-z0-9]+(?:[._-][a-z0-9]+)*)\.json\Z"
)

DOCUMENT_FIELDS = (
    "schema_version",
    "workload_benchmarks",
    "memory_scaling",
    "solver_memory",
    "retracing",
    "backend_resolution",
)
PROVENANCE_FIELDS = (
    "schema_version",
    "recorded_at_utc",
    "radiosim_version",
    "git_sha",
    "working_tree_clean",
    "platform",
    "machine",
    "cpu_model",
    "cpu_count_logical",
    "python_version",
    "numpy_version",
    "jax_version",
    "jaxlib_version",
    "dask_version",
    "pixi_environment",
    "pixi_lock_sha256",
)
CONTEXT_FIELDS = (
    "backend_requested",
    "backend_actual",
    "backend_version",
    "device_kind",
    "compilation_used",
    "precision_preset",
    "precision_default",
    "precision_accumulation",
    "precision_output",
    "result_dtype",
    "policy_id",
    "input_identity_sha256",
    "measurement_limitations",
)
WORKLOAD_FIELDS = (
    "schema_version",
    "provenance",
    "context",
    "accelerator",
    "device_memory",
    "workload",
    "n_antennas",
    "n_baselines",
    "n_point_sources",
    "n_healpix_pixels",
    "n_times",
    "n_frequencies",
    "sky_representation",
    "solver_workers",
    "loader_max_workers",
    "setup_seconds",
    "compile_seconds",
    "steady_state_median_seconds",
    "steady_state_min_seconds",
    "steady_state_max_seconds",
    "steady_state_iterations",
    "host_transfer_seconds",
    "peak_host_bytes",
    "host_memory_method",
    "reference_backend",
    "max_absolute_deviation",
    "max_relative_deviation",
    "tolerance_rtol",
    "tolerance_atol",
    "within_tolerance",
    "unmeasured",
    "notes",
)
MEMORY_FIELDS = (
    "schema_version",
    "provenance",
    "context",
    "comparison_id",
    "implementation_state",
    "measurement_scope",
    "allocator",
    "includes_backend_native_allocations",
    "inputs_preallocated",
    "includes_solver_input_construction",
    "includes_output_reassembly",
    "logical_n_baselines",
    "logical_n_sources",
    "logical_pair_count",
    "kernel_n_sources",
    "target_kernel_pairs",
    "kernel_baseline_chunks",
    "kernel_pair_counts",
    "max_kernel_pair_count",
    "synthetic_input_bytes_excluded",
    "peak_host_bytes",
    "notes",
)
SOLVER_MEMORY_FIELDS = (
    "schema_version",
    "provenance",
    "context",
    "comparison_id",
    "implementation_state",
    "measurement_scope",
    "allocator",
    "includes_backend_native_allocations",
    "includes_simulator_setup",
    "includes_solver_input_construction",
    "includes_output_assembly",
    "solver",
    "sky_representation",
    "logical_n_baselines",
    "logical_source_counts",
    "kernel_source_counts",
    "n_times",
    "n_frequencies",
    "target_kernel_pairs",
    "bucket_policy",
    "peak_host_bytes",
    "notes",
)
SIGNATURE_FIELDS = (
    "jones_p_shape",
    "jones_q_shape",
    "coherency_shape",
    "phase_shape",
    "envelope_shape",
    "stokes_i_shape",
    "jones_p_dtype",
    "jones_q_dtype",
    "coherency_dtype",
    "phase_dtype",
    "envelope_dtype",
    "stokes_i_dtype",
    "call_count",
    "first_call_seconds",
    "minimum_repeat_call_seconds",
)
RETRACING_FIELDS = (
    "schema_version",
    "provenance",
    "context",
    "comparison_id",
    "implementation_state",
    "measurement_scope",
    "solver",
    "sky_representation",
    "bucket_policy",
    "padding_location",
    "logical_source_counts",
    "kernel_source_counts",
    "distinct_logical_source_counts",
    "distinct_kernel_source_counts",
    "observed_signatures",
    "distinct_signature_count",
    "leaf_call_count",
    "scope_step_seconds",
    "scope_total_seconds",
    "max_first_to_repeat_ratio",
    "retrace_overhead_seconds",
    "notes",
)
BACKEND_FIELDS = (
    "schema_version",
    "provenance",
    "context",
    "comparison_id",
    "implementation_state",
    "operation",
    "requested_backend",
    "resolved_backend",
    "discovery_policy",
    "fresh_process_samples",
    "cold_seconds",
    "minimum_seconds",
    "median_seconds",
    "maximum_seconds",
    "jax_distribution_installed",
    "jax_in_sys_modules_before",
    "jax_in_sys_modules_after",
    "jaxlib_in_sys_modules_before",
    "jaxlib_in_sys_modules_after",
    "notes",
)


class CpuEvidenceError(RuntimeError):
    """A clean-source, document, or publication contract did not hold."""


@dataclass(frozen=True, slots=True)
class PreflightDependencies:
    """Injectable source/Pixi boundaries used by unit tests and generation."""

    repository_root: Path
    cwd: Path
    environ: Mapping[str, str]
    prefix: Path
    executable: Path
    run_command: Callable[[tuple[str, ...], Path], subprocess.CompletedProcess[str]]
    package_identity_check: Callable[[PreflightDependencies], None]


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Authenticated regular-file inventory materialized from exact source S."""

    root: Path
    entries: tuple[tuple[str, str, int, str], ...]
    manifest_sha256: str


def _run_command(
    command: tuple[str, ...], cwd: Path
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CpuEvidenceError(f"command could not run: {command[0]}") from error


def _default_dependencies() -> PreflightDependencies:
    return PreflightDependencies(
        repository_root=REPOSITORY_ROOT,
        cwd=Path.cwd(),
        environ=os.environ,
        prefix=Path(sys.prefix),
        executable=Path(sys.executable),
        run_command=_run_command,
        package_identity_check=_require_cpu_package_identity,
    )


def _command_stdout(
    dependencies: PreflightDependencies,
    command: tuple[str, ...],
    *,
    operation: str,
) -> str:
    completed = dependencies.run_command(command, dependencies.repository_root)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CpuEvidenceError(f"{operation} failed: {detail}")
    return completed.stdout


def _sha256_path(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise CpuEvidenceError(f"required file is unreadable: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise CpuEvidenceError(f"required path must be a regular non-symlink: {path}")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise CpuEvidenceError(f"required file is unreadable: {path}") from error


def _lexical_absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


_PIXI_LOCK_PLATFORMS = ("linux-64", "osx-64", "osx-arm64")


def _parse_pixi_list(raw: str, *, platform_name: str) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, CpuEvidenceError) as error:
        raise CpuEvidenceError(
            f"Pixi package list for {platform_name} is invalid JSON"
        ) from error
    if type(decoded) is not list:
        raise CpuEvidenceError(f"Pixi package list for {platform_name} is not an array")
    rows = cast(list[object], decoded)
    parsed: list[dict[str, Any]] = []
    for index, value in enumerate(rows):
        if type(value) is not dict:
            raise CpuEvidenceError(
                f"Pixi package list {platform_name}[{index}] is not an object"
            )
        row = cast(dict[str, Any], value)
        for field in ("kind", "name", "url"):
            if type(row.get(field)) is not str or not row[field]:
                raise CpuEvidenceError(
                    f"Pixi package list {platform_name}[{index}].{field} is invalid"
                )
        if row["kind"] not in {"conda", "pypi"}:
            raise CpuEvidenceError(
                f"Pixi package list {platform_name}[{index}].kind is unsupported"
            )
        if row["kind"] == "conda":
            _ = _digest(
                row.get("sha256"),
                location=f"Pixi package list {platform_name}[{index}].sha256",
            )
        elif row["url"] != "./":
            _ = _string(
                row.get("version"),
                location=f"Pixi package list {platform_name}[{index}].version",
            )
        parsed.append(row)
    lock_keys = [(cast(str, row["kind"]), cast(str, row["url"])) for row in parsed]
    if len(set(lock_keys)) != len(lock_keys):
        raise CpuEvidenceError(f"Pixi package list for {platform_name} has duplicates")
    return parsed


def _locked_environment_digest(
    rows_by_platform: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    payload = {
        platform_name: sorted(
            (cast(str, row["kind"]), cast(str, row["url"]))
            for row in rows_by_platform[platform_name]
        )
        for platform_name in _PIXI_LOCK_PLATFORMS
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _host_pixi_platform() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "osx-arm64"
    if system == "darwin" and machine in {"x86_64", "amd64"}:
        return "osx-64"
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return "linux-64"
    raise CpuEvidenceError(
        f"default CPU evidence has no locked package platform for {system}/{machine}"
    )


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise CpuEvidenceError(f"{label} is unreadable") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise CpuEvidenceError(f"{label} must be a regular non-symlink file")
    try:
        return path.read_bytes()
    except OSError as error:
        raise CpuEvidenceError(f"{label} is unreadable") from error


def _strict_json_bytes(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, CpuEvidenceError) as error:
        raise CpuEvidenceError(f"{label} is not strict UTF-8 JSON") from error


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _project_version(root: Path) -> str:
    raw = _regular_file_bytes(root / "pyproject.toml", label="pyproject.toml")
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise CpuEvidenceError("pyproject.toml is invalid") from error
    project = document.get("project")
    if type(project) is not dict:
        raise CpuEvidenceError("pyproject.toml has no project table")
    version = cast(dict[str, object], project).get("version")
    return _string(version, location="pyproject.project.version")


def _expected_live_inventories(
    rows: Sequence[Mapping[str, Any]],
    *,
    root: Path,
) -> tuple[set[tuple[str, str]], set[tuple[str, str, bool, str | None]]]:
    conda: set[tuple[str, str]] = set()
    pypi: set[tuple[str, str, bool, str | None]] = set()
    project_version = _project_version(root)
    for row in rows:
        if row["kind"] == "conda":
            pair = (cast(str, row["url"]), cast(str, row["sha256"]))
            if pair in conda:
                raise CpuEvidenceError("locked Conda inventory has duplicate artifacts")
            conda.add(pair)
            continue
        name = _normalized_distribution_name(cast(str, row["name"]))
        editable = row["url"] == "./" or "editable = true" in str(
            row.get("requested_spec", "")
        )
        version = project_version if row["url"] == "./" else cast(str, row["version"])
        source = root.resolve().as_uri() if editable else None
        item = (name, version, editable, source)
        if item in pypi:
            raise CpuEvidenceError("locked PyPI inventory has duplicate distributions")
        pypi.add(item)
    return conda, pypi


def _require_live_environment_matches(
    *,
    root: Path,
    prefix: Path,
    locked_rows: Sequence[Mapping[str, Any]],
) -> None:
    conda_meta = prefix / "conda-meta"
    try:
        meta_mode = conda_meta.lstat().st_mode
    except OSError as error:
        raise CpuEvidenceError("live Pixi conda-meta is unreadable") from error
    if stat.S_ISLNK(meta_mode) or not stat.S_ISDIR(meta_mode):
        raise CpuEvidenceError("live Pixi conda-meta must be a non-symlink directory")
    expected_conda, expected_pypi = _expected_live_inventories(locked_rows, root=root)
    actual_conda: set[tuple[str, str]] = set()
    conda_owned_dist_info: set[str] = set()
    try:
        metadata_paths = sorted(conda_meta.glob("*.json"))
    except OSError as error:
        raise CpuEvidenceError("live Pixi conda-meta cannot be enumerated") from error
    if not metadata_paths:
        raise CpuEvidenceError("live Pixi conda-meta is empty")
    dist_info_pattern = re.compile(
        r"lib/python[^/]+/site-packages/([^/]+\.dist-info)(?:/.*)?\Z"
    )
    for path in metadata_paths:
        record = _strict_json_bytes(
            _regular_file_bytes(path, label=f"Conda record {path.name}"),
            label=f"Conda record {path.name}",
        )
        if type(record) is not dict:
            raise CpuEvidenceError(f"Conda record {path.name} is not an object")
        mapping = cast(dict[str, object], record)
        url = _string(mapping.get("url"), location=f"Conda record {path.name}.url")
        digest = _digest(
            mapping.get("sha256"), location=f"Conda record {path.name}.sha256"
        )
        pair = (url, digest)
        if pair in actual_conda:
            raise CpuEvidenceError("live Conda inventory has duplicate artifacts")
        actual_conda.add(pair)
        files = mapping.get("files")
        if type(files) is not list:
            raise CpuEvidenceError(f"Conda record {path.name}.files is not an array")
        for item in cast(list[object], files):
            file_name = _string(item, location=f"Conda record {path.name}.files")
            match = dist_info_pattern.fullmatch(file_name)
            if match is not None:
                conda_owned_dist_info.add(match.group(1))
    if actual_conda != expected_conda:
        raise CpuEvidenceError("live Conda package inventory differs from Pixi lock")

    site_packages_candidates = tuple(
        path
        for path in (prefix / "lib").glob("python*/site-packages")
        if path.is_dir()
        and stat.S_ISDIR(path.parent.lstat().st_mode)
        and not stat.S_ISLNK(path.parent.lstat().st_mode)
    )
    if len(site_packages_candidates) != 1:
        raise CpuEvidenceError("live Pixi prefix must have one Python site-packages")
    site_packages = site_packages_candidates[0]
    actual_pypi: set[tuple[str, str, bool, str | None]] = set()
    for dist_info in sorted(site_packages.glob("*.dist-info")):
        try:
            mode = dist_info.lstat().st_mode
        except OSError as error:
            raise CpuEvidenceError("live dist-info is unreadable") from error
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise CpuEvidenceError("live dist-info must be a non-symlink directory")
        if dist_info.name in conda_owned_dist_info:
            continue
        metadata_raw = _regular_file_bytes(
            dist_info / "METADATA", label=f"{dist_info.name}/METADATA"
        )
        try:
            metadata = Parser().parsestr(metadata_raw.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise CpuEvidenceError(f"{dist_info.name}/METADATA is not UTF-8") from error
        name = metadata.get("Name")
        version = metadata.get("Version")
        normalized_name = _normalized_distribution_name(
            _string(name, location=f"{dist_info.name}.Name")
        )
        version_text = _string(version, location=f"{dist_info.name}.Version")
        installer = _regular_file_bytes(
            dist_info / "INSTALLER", label=f"{dist_info.name}/INSTALLER"
        )
        if installer.decode("utf-8").strip() != "uv-pixi":
            raise CpuEvidenceError(f"{dist_info.name} was not installed by uv-pixi")
        direct_url = dist_info / "direct_url.json"
        editable = False
        source: str | None = None
        if direct_url.exists():
            direct = _strict_json_bytes(
                _regular_file_bytes(
                    direct_url, label=f"{dist_info.name}/direct_url.json"
                ),
                label=f"{dist_info.name}/direct_url.json",
            )
            if type(direct) is not dict:
                raise CpuEvidenceError(
                    f"{dist_info.name}/direct_url.json is not an object"
                )
            direct_mapping = cast(dict[str, object], direct)
            source = _string(
                direct_mapping.get("url"), location=f"{dist_info.name}.direct_url"
            )
            directory_info = direct_mapping.get("dir_info")
            if type(directory_info) is dict:
                editable_value = cast(dict[str, object], directory_info).get("editable")
                if editable_value is not None:
                    editable = _boolean(
                        editable_value, location=f"{dist_info.name}.editable"
                    )
        item = (normalized_name, version_text, editable, source if editable else None)
        if item in actual_pypi:
            raise CpuEvidenceError("live PyPI inventory has duplicate distributions")
        actual_pypi.add(item)
    if actual_pypi != expected_pypi:
        raise CpuEvidenceError("live PyPI package inventory differs from Pixi lock")


def _require_cpu_package_identity(dependencies: PreflightDependencies) -> None:
    root = dependencies.repository_root
    manifest = root / "pixi.toml"
    pixi_exe = dependencies.environ.get("PIXI_EXE")
    if not pixi_exe:
        raise CpuEvidenceError("PIXI_EXE is required for package authentication")
    rows_by_platform: dict[str, list[dict[str, Any]]] = {}
    for platform_name in _PIXI_LOCK_PLATFORMS:
        raw = _command_stdout(
            dependencies,
            (
                pixi_exe,
                "list",
                "--manifest-path",
                str(manifest),
                "--environment",
                "default",
                "--platform",
                platform_name,
                "--json",
                "--no-install",
                "--locked",
            ),
            operation=f"Pixi locked package identity ({platform_name})",
        )
        rows_by_platform[platform_name] = _parse_pixi_list(
            raw, platform_name=platform_name
        )
    observed = _locked_environment_digest(rows_by_platform)
    expected = CPU_ENVIRONMENT_PACKAGE_SHA256["default"]
    if observed != expected:
        raise CpuEvidenceError(
            "default Pixi locked package identity differs from the authorized digest"
        )
    _require_live_environment_matches(
        root=root,
        prefix=dependencies.prefix,
        locked_rows=rows_by_platform[_host_pixi_platform()],
    )


def _require_empty_reference_manifests(root: Path) -> None:
    source = root / "src/radiosim/benchmarks/harness.py"
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, SyntaxError) as error:
        raise CpuEvidenceError(
            "benchmark harness manifest source is unreadable"
        ) from error
    observed: dict[str, bool] = {}
    required = {"PERF001_REFERENCE_SHA256", "PERF001_REFERENCE_SOURCE_SHA"}
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id in required:
            observed[node.target.id] = isinstance(node.value, ast.Dict) and not (
                node.value.keys or node.value.values
            )
    if observed != dict.fromkeys(required, True):
        raise CpuEvidenceError(
            "clean source S requires both retained PERF-001 manifests to be empty"
        )


def _require_empty_reference_namespace(root: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(root, flags)
    except OSError as error:
        raise CpuEvidenceError("PERF-001 reference namespace is unreadable") from error
    try:
        for component in REFERENCE_DIRECTORY.parts:
            try:
                mode = os.stat(
                    component,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                ).st_mode
            except FileNotFoundError:
                return
            except OSError as error:
                raise CpuEvidenceError(
                    "PERF-001 reference namespace has an unreadable component"
                ) from error
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise CpuEvidenceError(
                    "PERF-001 reference namespace has a symlink component or "
                    "non-directory component"
                )
            try:
                child_descriptor = os.open(
                    component,
                    flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as error:
                raise CpuEvidenceError(
                    "PERF-001 reference namespace has an unreadable or symlink component"
                ) from error
            os.close(directory_descriptor)
            directory_descriptor = child_descriptor
        try:
            entries = os.listdir(directory_descriptor)
        except OSError as error:
            raise CpuEvidenceError(
                "PERF-001 reference namespace is unreadable"
            ) from error
        if entries:
            raise CpuEvidenceError(
                "PERF-001 reference namespace must be empty before generation"
            )
    finally:
        os.close(directory_descriptor)


def preflight_generation(
    approved_source_sha: str,
    *,
    dependencies: PreflightDependencies | None = None,
) -> str:
    """Fail before runtime imports unless source and Pixi bind exactly to S."""
    if (
        _LOWER_HEX_40.fullmatch(approved_source_sha) is None
        or approved_source_sha == "0" * 40
    ):
        raise CpuEvidenceError("approved source SHA must be nonzero lowercase 40-hex")
    selected = dependencies or _default_dependencies()
    root = _lexical_absolute(selected.repository_root)
    if root != selected.repository_root:
        raise CpuEvidenceError("repository root must be an exact absolute lexical path")
    if _lexical_absolute(selected.cwd) != root:
        raise CpuEvidenceError("generation must run from the repository root")
    tool_path = root / TOOL_RELATIVE_PATH
    _ = _sha256_path(tool_path)

    top_level = _command_stdout(
        selected,
        ("git", "rev-parse", "--show-toplevel"),
        operation="Git top-level check",
    ).strip()
    if _lexical_absolute(top_level) != root:
        raise CpuEvidenceError("Git top-level does not match repository root")
    head = _command_stdout(
        selected,
        ("git", "rev-parse", "HEAD"),
        operation="Git source identity",
    ).strip()
    if head != approved_source_sha:
        raise CpuEvidenceError("HEAD does not equal the approved source SHA S")
    status_output = _command_stdout(
        selected,
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        operation="Git cleanliness check",
    )
    if status_output:
        raise CpuEvidenceError("generation requires an exactly clean worktree")
    tracked = _command_stdout(
        selected,
        ("git", "ls-files", "--", REFERENCE_DIRECTORY.as_posix()),
        operation="tracked PERF-001 namespace check",
    )
    if tracked.strip():
        raise CpuEvidenceError("source S must not track a PERF-001 JSON artifact")

    environ = selected.environ
    if environ.get("PIXI_ENVIRONMENT_NAME") != "default":
        raise CpuEvidenceError("generation is authorized only in Pixi default")
    if _lexical_absolute(environ.get("PIXI_PROJECT_ROOT", "")) != root:
        raise CpuEvidenceError("PIXI_PROJECT_ROOT does not bind to this checkout")
    manifest = root / "pixi.toml"
    lock = root / "pixi.lock"
    if _lexical_absolute(environ.get("PIXI_PROJECT_MANIFEST", "")) != manifest:
        raise CpuEvidenceError("PIXI_PROJECT_MANIFEST does not bind to pixi.toml")
    if _sha256_path(manifest) != PIXI_MANIFEST_SHA256:
        raise CpuEvidenceError("pixi.toml digest differs from the authorized source")
    if _sha256_path(lock) != PIXI_LOCK_SHA256:
        raise CpuEvidenceError("pixi.lock digest differs from the authorized source")

    expected_prefix = root / ".pixi/envs/default"
    declared_prefix = environ.get("CONDA_PREFIX")
    if not declared_prefix:
        raise CpuEvidenceError("CONDA_PREFIX is required")
    try:
        actual_prefix = selected.prefix.resolve(strict=True)
        conda_prefix = Path(declared_prefix).resolve(strict=True)
    except OSError as error:
        raise CpuEvidenceError("Pixi interpreter prefix is unreadable") from error
    if actual_prefix != expected_prefix or conda_prefix != expected_prefix:
        raise CpuEvidenceError("interpreter and CONDA_PREFIX must be Pixi default")
    try:
        executable = selected.executable.resolve(strict=True)
    except OSError as error:
        raise CpuEvidenceError("active Python executable is unreadable") from error
    if expected_prefix not in executable.parents:
        raise CpuEvidenceError("active Python executable is outside Pixi default")
    executable_mode = executable.stat().st_mode
    if not stat.S_ISREG(executable_mode) or not os.access(executable, os.X_OK):
        raise CpuEvidenceError("active Python executable is not regular/executable")

    pixi_exe_text = environ.get("PIXI_EXE")
    if not pixi_exe_text:
        raise CpuEvidenceError("PIXI_EXE is required")
    pixi_exe = Path(pixi_exe_text)
    try:
        pixi_mode = pixi_exe.stat().st_mode
    except OSError as error:
        raise CpuEvidenceError("PIXI_EXE is unreadable") from error
    if not stat.S_ISREG(pixi_mode) or not os.access(pixi_exe, os.X_OK):
        raise CpuEvidenceError("PIXI_EXE must be a regular executable")
    _ = _command_stdout(
        selected,
        (
            str(pixi_exe),
            "lock",
            "--check",
            "--no-install",
            "--manifest-path",
            str(manifest),
        ),
        operation="Pixi lock check",
    )
    selected.package_identity_check(selected)
    _require_empty_reference_manifests(root)
    _require_empty_reference_namespace(root)
    return head


def _git_snapshot_tree(
    repository_root: Path,
    approved_source_sha: str,
) -> tuple[str, dict[str, tuple[str, str]]]:
    try:
        object_format = subprocess.run(
            ["git", "rev-parse", "--show-object-format"],
            cwd=repository_root,
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        raw_tree = subprocess.run(
            [
                "git",
                "ls-tree",
                "-rz",
                "--full-tree",
                approved_source_sha,
                "--",
                *SOURCE_SNAPSHOT_PATHS,
            ],
            cwd=repository_root,
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise CpuEvidenceError("approved source tree is unreadable") from error
    if object_format not in {"sha1", "sha256"}:
        raise CpuEvidenceError("Git source tree uses an unsupported object format")
    expected: dict[str, tuple[str, str]] = {}
    for raw_entry in raw_tree.split(b"\0"):
        if not raw_entry:
            continue
        try:
            header, raw_path = raw_entry.split(b"\t", 1)
            raw_mode, raw_type, raw_oid = header.split(b" ", 2)
            relative = raw_path.decode("utf-8")
            mode = raw_mode.decode("ascii")
            object_type = raw_type.decode("ascii")
            oid = raw_oid.decode("ascii")
        except (UnicodeDecodeError, ValueError) as error:
            raise CpuEvidenceError(
                "approved source tree has a malformed entry"
            ) from error
        if (
            object_type != "blob"
            or mode not in {"100644", "100755"}
            or not relative
            or relative in expected
        ):
            raise CpuEvidenceError("approved source tree contains an unsafe entry")
        expected[relative] = (mode, oid)
    if not expected or TOOL_RELATIVE_PATH.as_posix() not in expected:
        raise CpuEvidenceError("approved source tree is missing the CPU evidence tool")
    return object_format, expected


def _git_blob_oid(raw: bytes, *, object_format: str) -> str:
    hasher = hashlib.new(object_format)
    hasher.update(f"blob {len(raw)}\0".encode("ascii"))
    hasher.update(raw)
    return hasher.hexdigest()


def _authenticate_source_snapshot(
    *,
    repository_root: Path,
    snapshot_root: Path,
    approved_source_sha: str,
) -> SourceSnapshot:
    object_format, expected = _git_snapshot_tree(repository_root, approved_source_sha)
    root = snapshot_root.resolve(strict=True)
    actual_paths: set[str] = set()
    entries: list[tuple[str, str, int, str]] = []
    try:
        candidates = sorted(root.rglob("*"))
    except OSError as error:
        raise CpuEvidenceError("source snapshot cannot be enumerated") from error
    for path in candidates:
        try:
            mode_bits = path.lstat().st_mode
        except OSError as error:
            raise CpuEvidenceError(
                "source snapshot contains an unreadable entry"
            ) from error
        if stat.S_ISLNK(mode_bits):
            raise CpuEvidenceError("source snapshot contains a symlink")
        if stat.S_ISDIR(mode_bits):
            continue
        if not stat.S_ISREG(mode_bits):
            raise CpuEvidenceError("source snapshot contains a non-regular file")
        relative = path.relative_to(root).as_posix()
        actual_paths.add(relative)
        expected_entry = expected.get(relative)
        if expected_entry is None:
            raise CpuEvidenceError("source snapshot contains an uncommitted file")
        expected_mode, expected_oid = expected_entry
        actual_mode = "100755" if mode_bits & 0o111 else "100644"
        if actual_mode != expected_mode:
            raise CpuEvidenceError("source snapshot file mode differs from approved S")
        raw = _regular_file_bytes(path, label=f"source snapshot {relative}")
        if _git_blob_oid(raw, object_format=object_format) != expected_oid:
            raise CpuEvidenceError("source snapshot bytes differ from approved S")
        entries.append(
            (relative, actual_mode, len(raw), hashlib.sha256(raw).hexdigest())
        )
    if actual_paths != set(expected):
        raise CpuEvidenceError("source snapshot file inventory differs from approved S")
    canonical = json.dumps(
        entries,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return SourceSnapshot(
        root=root,
        entries=tuple(entries),
        manifest_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def _safe_extract_source_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(mode=0o700)
    seen: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                relative = Path(member.name)
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or member.name in seen
                    or not (member.isdir() or member.isreg())
                ):
                    raise CpuEvidenceError(
                        "Git source archive contains an unsafe entry"
                    )
                seen.add(member.name)
                target = destination.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise CpuEvidenceError(
                        "Git source archive contains an unreadable file"
                    )
                with target.open("xb") as stream:
                    while chunk := extracted.read(1024 * 1024):
                        _ = stream.write(chunk)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
    except (OSError, tarfile.TarError) as error:
        raise CpuEvidenceError("Git source archive could not be extracted") from error


def _export_source_snapshot(
    *,
    repository_root: Path,
    approved_source_sha: str,
    workspace: Path,
) -> SourceSnapshot:
    archive_path = workspace / "source.tar"
    snapshot_root = workspace / "source"
    try:
        completed = subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                f"--output={archive_path}",
                approved_source_sha,
                "--",
                *SOURCE_SNAPSHOT_PATHS,
            ],
            cwd=repository_root,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CpuEvidenceError(
            "approved source snapshot could not be exported"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CpuEvidenceError(
            f"approved source snapshot could not be exported: {detail}"
        )
    _safe_extract_source_archive(archive_path, snapshot_root)
    try:
        archive_path.unlink()
    except OSError as error:
        raise CpuEvidenceError(
            "temporary source archive could not be removed"
        ) from error
    snapshot = _authenticate_source_snapshot(
        repository_root=repository_root,
        snapshot_root=snapshot_root,
        approved_source_sha=approved_source_sha,
    )
    _seal_source_snapshot(snapshot)
    return snapshot


def _seal_source_snapshot(snapshot: SourceSnapshot) -> None:
    for relative, mode, _, _ in snapshot.entries:
        (snapshot.root / relative).chmod(0o555 if mode == "100755" else 0o444)
    directories = sorted(
        (path for path in snapshot.root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        directory.chmod(0o555)
    snapshot.root.chmod(0o555)


def _unseal_source_snapshot(snapshot_root: Path) -> None:
    if not snapshot_root.exists():
        return
    try:
        snapshot_root.chmod(0o700)
        paths = sorted(snapshot_root.rglob("*"), key=lambda path: len(path.parts))
        for path in paths:
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                path.chmod(0o700)
            elif stat.S_ISREG(mode):
                path.chmod(0o600)
    except OSError as error:
        raise CpuEvidenceError(
            "temporary source snapshot permissions cannot be restored"
        ) from error


def _verify_source_snapshot(
    snapshot: SourceSnapshot,
    *,
    repository_root: Path,
    approved_source_sha: str,
) -> None:
    observed = _authenticate_source_snapshot(
        repository_root=repository_root,
        snapshot_root=snapshot.root,
        approved_source_sha=approved_source_sha,
    )
    if observed.entries != snapshot.entries or (
        observed.manifest_sha256 != snapshot.manifest_sha256
    ):
        raise CpuEvidenceError(
            "authenticated source snapshot changed during measurement"
        )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CpuEvidenceError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> NoReturn:
    raise CpuEvidenceError(f"non-finite JSON number is forbidden: {token}")


def _mapping(value: object, fields: Sequence[str], *, location: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise CpuEvidenceError(f"{location} must be a JSON object")
    result = cast(dict[str, Any], value)
    if set(result) != set(fields):
        missing = sorted(set(fields) - set(result))
        extra = sorted(set(result) - set(fields))
        raise CpuEvidenceError(
            f"{location} has wrong fields; missing={missing}, extra={extra}"
        )
    return result


def _array(value: object, *, location: str) -> list[Any]:
    if type(value) is not list:
        raise CpuEvidenceError(f"{location} must be an ordered JSON array")
    return cast(list[Any], value)


def _string(value: object, *, location: str) -> str:
    if type(value) is not str or not value:
        raise CpuEvidenceError(f"{location} must be a nonempty JSON string")
    return value


def _integer(value: object, *, location: str, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise CpuEvidenceError(f"{location} must be a {qualifier} JSON integer")
    return value


def _number(value: object, *, location: str) -> float:
    if type(value) not in (int, float):
        raise CpuEvidenceError(f"{location} must be a finite nonnegative number")
    number = float(cast(int | float, value))
    if not math.isfinite(number) or number < 0:
        raise CpuEvidenceError(f"{location} must be a finite nonnegative number")
    return number


def _digest(value: object, *, location: str, git: bool = False) -> str:
    text = _string(value, location=location)
    pattern = _LOWER_HEX_40 if git else _LOWER_HEX_64
    if pattern.fullmatch(text) is None or (git and text == "0" * 40):
        raise CpuEvidenceError(f"{location} is not a canonical digest")
    return text


def _context(value: object, *, location: str) -> dict[str, Any]:
    context = _mapping(value, CONTEXT_FIELDS, location=location)
    for name in (
        "backend_requested",
        "backend_actual",
        "backend_version",
        "device_kind",
        "precision_preset",
        "precision_default",
        "precision_accumulation",
        "precision_output",
        "result_dtype",
        "policy_id",
    ):
        _ = _string(context[name], location=f"{location}.{name}")
    if type(context["compilation_used"]) is not bool:
        raise CpuEvidenceError(f"{location}.compilation_used must be boolean")
    _ = _digest(context["input_identity_sha256"], location=f"{location}.identity")
    limitations = _array(
        context["measurement_limitations"],
        location=f"{location}.measurement_limitations",
    )
    for index, limitation in enumerate(limitations):
        _ = _string(limitation, location=f"{location}.measurement_limitations[{index}]")
    return context


def _provenance(value: object, *, location: str) -> dict[str, Any]:
    provenance = _mapping(value, PROVENANCE_FIELDS, location=location)
    if provenance["schema_version"] != PROVENANCE_SCHEMA:
        raise CpuEvidenceError(f"{location}.schema_version is wrong")
    _ = _digest(provenance["git_sha"], location=f"{location}.git_sha", git=True)
    _ = _digest(provenance["pixi_lock_sha256"], location=f"{location}.pixi_lock")
    if provenance["working_tree_clean"] is not True:
        raise CpuEvidenceError(f"{location} does not name a clean source")
    _ = _integer(
        provenance["cpu_count_logical"],
        location=f"{location}.cpu_count",
        positive=True,
    )
    for name in PROVENANCE_FIELDS:
        if name not in {
            "working_tree_clean",
            "cpu_count_logical",
            "git_sha",
            "pixi_lock_sha256",
        }:
            text = _string(provenance[name], location=f"{location}.{name}")
            if text == "unknown":
                raise CpuEvidenceError(f"{location}.{name} must not be unknown")
    recorded = cast(str, provenance["recorded_at_utc"])
    if not (recorded.endswith("Z") or recorded.endswith("+00:00")):
        raise CpuEvidenceError(f"{location}.recorded_at_utc must name explicit UTC")
    try:
        parsed = datetime.fromisoformat(
            recorded.removesuffix("Z") + ("+00:00" if recorded.endswith("Z") else "")
        )
    except ValueError as error:
        raise CpuEvidenceError(
            f"{location}.recorded_at_utc must be ISO-8601"
        ) from error
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise CpuEvidenceError(f"{location}.recorded_at_utc must be normalized UTC")
    return provenance


def _common_row(
    value: object,
    fields: Sequence[str],
    schema: str,
    *,
    location: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    row = _mapping(value, fields, location=location)
    if row["schema_version"] != schema:
        raise CpuEvidenceError(f"{location}.schema_version is wrong")
    provenance = _provenance(row["provenance"], location=f"{location}.provenance")
    context = _context(row["context"], location=f"{location}.context")
    _validate_context_provenance(provenance, context, location=location)
    return row, provenance, context


def _validate_context_provenance(
    provenance: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    location: str,
) -> None:
    actual = cast(str, context["backend_actual"])
    expected: object | None = None
    if actual.startswith("jax-"):
        expected = provenance["jax_version"]
    elif actual.startswith("numpy-"):
        expected = provenance["numpy_version"]
    elif actual.startswith("dask-"):
        expected = provenance["dask_version"]
    if expected is not None and context["backend_version"] != expected:
        raise CpuEvidenceError(
            f"{location}.context.backend_version differs from provenance"
        )


def _boolean(value: object, *, location: str) -> bool:
    if type(value) is not bool:
        raise CpuEvidenceError(f"{location} must be a JSON boolean")
    return value


def _integer_array(
    value: object,
    *,
    location: str,
    allow_empty: bool = False,
) -> list[int]:
    values = _array(value, location=location)
    if not values and not allow_empty:
        raise CpuEvidenceError(f"{location} must be a nonempty integer array")
    return [
        _integer(item, location=f"{location}[{index}]")
        for index, item in enumerate(values)
    ]


def _number_array(value: object, *, location: str) -> list[float]:
    values = _array(value, location=location)
    if not values:
        raise CpuEvidenceError(f"{location} must be a nonempty number array")
    return [
        _number(item, location=f"{location}[{index}]")
        for index, item in enumerate(values)
    ]


def _string_array(value: object, *, location: str) -> list[str]:
    values = _array(value, location=location)
    return [
        _string(item, location=f"{location}[{index}]")
        for index, item in enumerate(values)
    ]


def _positive_number(value: object, *, location: str) -> float:
    number = _number(value, location=location)
    if number == 0.0:
        raise CpuEvidenceError(f"{location} must be a positive JSON number")
    return number


def _power_of_two_bucket(count: int) -> int:
    return 0 if count == 0 else 1 << (count - 1).bit_length()


def _require_equal_fields(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    fields: Sequence[str],
    *,
    location: str,
) -> None:
    for name in fields:
        if first[name] != second[name]:
            raise CpuEvidenceError(f"{location} pair field {name} must match")


def _require_pair_identity(
    first: tuple[Mapping[str, Any], Mapping[str, Any]],
    second: tuple[Mapping[str, Any], Mapping[str, Any]],
    *,
    row_fields: Sequence[str],
    location: str,
) -> None:
    first_row, first_context = first
    second_row, second_context = second
    if (
        first_context["input_identity_sha256"]
        != second_context["input_identity_sha256"]
    ):
        raise CpuEvidenceError(f"{location} pair input identities must match")
    _require_equal_fields(
        first_context,
        second_context,
        (
            "backend_requested",
            "backend_actual",
            "backend_version",
            "device_kind",
            "compilation_used",
            "precision_preset",
            "precision_default",
            "precision_accumulation",
            "precision_output",
            "result_dtype",
            "measurement_limitations",
        ),
        location=f"{location}.context",
    )
    _require_equal_fields(
        first_row,
        second_row,
        row_fields,
        location=location,
    )


def _require_cpu_numeric_context(
    context: Mapping[str, Any],
    *,
    location: str,
) -> None:
    expected = {
        "precision_preset": "explicit",
        "precision_default": "float64",
        "precision_accumulation": "float64",
        "precision_output": "float64",
        "result_dtype": "complex128",
    }
    if any(context[name] != value for name, value in expected.items()):
        raise CpuEvidenceError(
            f"{location} must use explicit float64/complex128 precision"
        )


def _validate_workload_row(
    row: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    location: str,
) -> None:
    if row["accelerator"] is not None or row["device_memory"] is not None:
        raise CpuEvidenceError("CPU workload accelerator/device_memory must be null")
    for name in (
        "workload",
        "sky_representation",
        "host_memory_method",
        "reference_backend",
        "notes",
    ):
        _ = _string(row[name], location=f"{location}.{name}")
    for name in (
        "n_antennas",
        "n_baselines",
        "n_point_sources",
        "n_healpix_pixels",
        "solver_workers",
        "loader_max_workers",
        "peak_host_bytes",
    ):
        _ = _integer(row[name], location=f"{location}.{name}")
    for name in ("n_times", "n_frequencies", "steady_state_iterations"):
        _ = _integer(row[name], location=f"{location}.{name}", positive=True)
    if cast(int, row["steady_state_iterations"]) < 5:
        raise CpuEvidenceError("workload steady-state sample count must be at least 5")
    timings = {
        name: _number(row[name], location=f"{location}.{name}")
        for name in (
            "setup_seconds",
            "compile_seconds",
            "steady_state_median_seconds",
            "steady_state_min_seconds",
            "steady_state_max_seconds",
            "host_transfer_seconds",
            "max_absolute_deviation",
            "max_relative_deviation",
            "tolerance_rtol",
            "tolerance_atol",
        )
    }
    if not (
        timings["steady_state_min_seconds"]
        <= timings["steady_state_median_seconds"]
        <= timings["steady_state_max_seconds"]
    ):
        raise CpuEvidenceError("workload median must lie between minimum and maximum")
    if row["reference_backend"] != "numpy":
        raise CpuEvidenceError("workload correctness reference must be NumPy")
    _ = _boolean(row["within_tolerance"], location=f"{location}.within_tolerance")
    unmeasured = _string_array(row["unmeasured"], location=f"{location}.unmeasured")
    if context["device_kind"] != "cpu":
        raise CpuEvidenceError("CPU workload context must identify a CPU")
    if "gpu" not in unmeasured:
        raise CpuEvidenceError("CPU workload must state GPU was unmeasured")


def _validate_memory_row(
    row: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    location: str,
) -> None:
    for name in ("comparison_id", "implementation_state", "allocator", "notes"):
        _ = _string(row[name], location=f"{location}.{name}")
    if (
        row["measurement_scope"]
        != "contraction_wrapper_python_heap_including_output_assembly"
    ):
        raise CpuEvidenceError("P-a memory measurement scope is wrong")
    expected_facts = {
        "allocator": "python_heap_tracemalloc",
        "includes_backend_native_allocations": False,
        "inputs_preallocated": True,
        "includes_solver_input_construction": False,
        "includes_output_reassembly": True,
    }
    for name, expected in expected_facts.items():
        if name != "allocator":
            _ = _boolean(row[name], location=f"{location}.{name}")
        if row[name] != expected:
            raise CpuEvidenceError(f"{location}.{name} has the wrong scope value")
    integers = {
        name: _integer(row[name], location=f"{location}.{name}")
        for name in (
            "logical_n_baselines",
            "logical_n_sources",
            "logical_pair_count",
            "kernel_n_sources",
            "max_kernel_pair_count",
            "synthetic_input_bytes_excluded",
            "peak_host_bytes",
        )
    }
    baselines = integers["logical_n_baselines"]
    sources = integers["logical_n_sources"]
    kernel_sources = integers["kernel_n_sources"]
    if integers["logical_pair_count"] != baselines * sources:
        raise CpuEvidenceError("P-a logical pair count is inconsistent")
    if kernel_sources != sources:
        raise CpuEvidenceError("P-a kernel source count differs from logical count")
    chunks = _integer_array(
        row["kernel_baseline_chunks"], location=f"{location}.kernel_baseline_chunks"
    )
    pair_counts = _integer_array(
        row["kernel_pair_counts"], location=f"{location}.kernel_pair_counts"
    )
    if sum(chunks) != baselines or (baselines > 0 and 0 in chunks):
        raise CpuEvidenceError("P-a baseline chunks do not cover the logical axis")
    if pair_counts != [chunk * kernel_sources for chunk in chunks]:
        raise CpuEvidenceError("P-a kernel pair counts are inconsistent")
    if integers["max_kernel_pair_count"] != max(pair_counts):
        raise CpuEvidenceError("P-a maximum kernel pair count is inconsistent")
    if row["target_kernel_pairs"] is not None:
        _ = _integer(
            row["target_kernel_pairs"],
            location=f"{location}.target_kernel_pairs",
            positive=True,
        )
    state = row["implementation_state"]
    if state == "unchunked_reference":
        expected_policy = "unbounded_reference_v1"
        if row["target_kernel_pairs"] is not None or chunks != [baselines]:
            raise CpuEvidenceError("P-a reference chunk policy is wrong")
    elif state == "chunked_production":
        expected_policy = "target_kernel_pairs_131072_v1"
        if row["target_kernel_pairs"] != TARGET_KERNEL_PAIRS:
            raise CpuEvidenceError("P-a production target is wrong")
        if baselines == 0 or kernel_sources == 0:
            expected_chunks = [baselines]
        else:
            chunk_size = max(
                1,
                min(baselines, TARGET_KERNEL_PAIRS // kernel_sources),
            )
            expected_chunks = [
                min(chunk_size, baselines - start)
                for start in range(0, baselines, chunk_size)
            ]
        if chunks != expected_chunks:
            raise CpuEvidenceError("P-a production chunk sequence is wrong")
        if integers["max_kernel_pair_count"] > max(TARGET_KERNEL_PAIRS, kernel_sources):
            raise CpuEvidenceError("P-a production kernel pair limit was exceeded")
    else:
        raise CpuEvidenceError("P-a implementation state is wrong")
    if context["policy_id"] != expected_policy:
        raise CpuEvidenceError("P-a context policy differs from implementation state")


def _validate_solver_memory_row(
    row: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    location: str,
) -> None:
    for name in (
        "comparison_id",
        "implementation_state",
        "allocator",
        "solver",
        "sky_representation",
        "bucket_policy",
        "notes",
    ):
        _ = _string(row[name], location=f"{location}.{name}")
    if (
        row["measurement_scope"]
        != "direct_solver_step_python_heap_including_input_construction_and_output_assembly"
    ):
        raise CpuEvidenceError("solver-memory measurement scope is wrong")
    expected_facts = {
        "allocator": "python_heap_tracemalloc",
        "includes_backend_native_allocations": False,
        "includes_simulator_setup": False,
        "includes_solver_input_construction": True,
        "includes_output_assembly": True,
    }
    for name, expected in expected_facts.items():
        if name != "allocator":
            _ = _boolean(row[name], location=f"{location}.{name}")
        if row[name] != expected:
            raise CpuEvidenceError(f"{location}.{name} has the wrong scope value")
    expected_representation = {"point": "point_sources", "healpix": "healpix"}.get(
        cast(str, row["solver"])
    )
    if row["sky_representation"] != expected_representation:
        raise CpuEvidenceError("solver-memory solver/sky representation is wrong")
    baselines = _integer(
        row["logical_n_baselines"], location=f"{location}.logical_n_baselines"
    )
    del baselines
    logical_counts = _integer_array(
        row["logical_source_counts"], location=f"{location}.logical_source_counts"
    )
    kernel_counts = _integer_array(
        row["kernel_source_counts"], location=f"{location}.kernel_source_counts"
    )
    if len(logical_counts) != len(kernel_counts):
        raise CpuEvidenceError("solver-memory source-count lengths differ")
    n_times = _integer(row["n_times"], location=f"{location}.n_times", positive=True)
    n_frequencies = _integer(
        row["n_frequencies"], location=f"{location}.n_frequencies", positive=True
    )
    if len(logical_counts) != n_times * n_frequencies:
        raise CpuEvidenceError("solver-memory logical source-count length is wrong")
    target_kernel_pairs = _integer(
        row["target_kernel_pairs"],
        location=f"{location}.target_kernel_pairs",
        positive=True,
    )
    if target_kernel_pairs != TARGET_KERNEL_PAIRS:
        raise CpuEvidenceError("solver-memory target kernel-pair count is wrong")
    state = row["implementation_state"]
    if state == "unbucketed_reference":
        expected_policy = "identity_reference_v1"
        expected_counts = logical_counts
    elif state == "bucketed_production":
        expected_policy = "pow2_compiled_v1"
        expected_counts = [_power_of_two_bucket(count) for count in logical_counts]
    else:
        raise CpuEvidenceError("solver-memory implementation state is wrong")
    if (
        row["bucket_policy"] != expected_policy
        or context["policy_id"] != expected_policy
    ):
        raise CpuEvidenceError("solver-memory bucket policy is wrong")
    if kernel_counts != expected_counts:
        raise CpuEvidenceError("solver-memory kernel source counts are wrong")
    _ = _integer(row["peak_host_bytes"], location=f"{location}.peak_host_bytes")


def _validate_signature(value: object, *, location: str) -> dict[str, Any]:
    signature = _mapping(value, SIGNATURE_FIELDS, location=location)
    for operand in (
        "jones_p",
        "jones_q",
        "coherency",
        "phase",
        "envelope",
        "stokes_i",
    ):
        shape_value = signature[f"{operand}_shape"]
        dtype_value = signature[f"{operand}_dtype"]
        if (shape_value is None) != (dtype_value is None):
            raise CpuEvidenceError(
                f"{location}.{operand} shape/dtype nullability differs"
            )
        if shape_value is not None:
            _ = _integer_array(
                shape_value,
                location=f"{location}.{operand}_shape",
                allow_empty=True,
            )
            _ = _string(dtype_value, location=f"{location}.{operand}_dtype")
    for operand in ("jones_p", "jones_q", "phase"):
        if signature[f"{operand}_shape"] is None:
            raise CpuEvidenceError(f"{location}.{operand} is mandatory")
    if (signature["coherency_shape"] is None) == (signature["stokes_i_shape"] is None):
        raise CpuEvidenceError(f"{location} must have exactly one signal operand")
    jones_p = cast(list[int], signature["jones_p_shape"])
    jones_q = cast(list[int], signature["jones_q_shape"])
    phase = cast(list[int], signature["phase_shape"])
    if len(jones_p) != 4 or jones_p[-2:] != [2, 2]:
        raise CpuEvidenceError(f"{location}.jones_p_shape is not (B,S,2,2)")
    if jones_q != jones_p or phase != jones_p[:2]:
        raise CpuEvidenceError(f"{location} Jones/phase shapes are inconsistent")
    source_count = jones_p[1]
    if signature["coherency_shape"] is not None and signature["coherency_shape"] != [
        source_count,
        2,
        2,
    ]:
        raise CpuEvidenceError(f"{location}.coherency_shape is wrong")
    if signature["stokes_i_shape"] is not None and signature["stokes_i_shape"] != [
        source_count
    ]:
        raise CpuEvidenceError(f"{location}.stokes_i_shape is wrong")
    if signature["envelope_shape"] not in (None, [], jones_p[:2]):
        raise CpuEvidenceError(f"{location}.envelope_shape is wrong")
    call_count = _integer(
        signature["call_count"], location=f"{location}.call_count", positive=True
    )
    if call_count < 2:
        raise CpuEvidenceError(f"{location}.call_count must include a repeat")
    _ = _number(
        signature["first_call_seconds"], location=f"{location}.first_call_seconds"
    )
    _ = _positive_number(
        signature["minimum_repeat_call_seconds"],
        location=f"{location}.minimum_repeat_call_seconds",
    )
    return signature


def _signature_key(signature: Mapping[str, Any]) -> tuple[object, ...]:
    key: list[object] = []
    for operand in (
        "jones_p",
        "jones_q",
        "coherency",
        "phase",
        "envelope",
        "stokes_i",
    ):
        shape = signature[f"{operand}_shape"]
        key.extend(
            (None if shape is None else tuple(shape), signature[f"{operand}_dtype"])
        )
    return tuple(key)


def _validate_retracing_row(
    row: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    location: str,
) -> None:
    for name in (
        "comparison_id",
        "implementation_state",
        "measurement_scope",
        "solver",
        "sky_representation",
        "bucket_policy",
        "padding_location",
        "notes",
    ):
        _ = _string(row[name], location=f"{location}.{name}")
    expected_representation = {
        "synthetic_wrapper": "synthetic_contraction",
        "point": "point_sources",
        "healpix": "healpix",
    }.get(cast(str, row["solver"]))
    if row["sky_representation"] != expected_representation:
        raise CpuEvidenceError("retracing solver/sky representation is wrong")
    state = row["implementation_state"]
    if state == "unbucketed_reference":
        expected_policy = "identity_reference_v1"
        expected_padding = "none"
    elif state == "bucketed_production":
        expected_policy = "pow2_compiled_v1"
        expected_padding = "early_host"
    else:
        raise CpuEvidenceError("retracing implementation state is wrong")
    if (
        row["bucket_policy"] != expected_policy
        or context["policy_id"] != expected_policy
    ):
        raise CpuEvidenceError("retracing bucket policy is wrong")
    if row["padding_location"] != expected_padding:
        raise CpuEvidenceError("retracing padding location is wrong")
    logical_counts = _integer_array(
        row["logical_source_counts"], location=f"{location}.logical_source_counts"
    )
    kernel_counts = _integer_array(
        row["kernel_source_counts"], location=f"{location}.kernel_source_counts"
    )
    if len(logical_counts) != len(kernel_counts):
        raise CpuEvidenceError("retracing source-count lengths differ")
    expected_counts = (
        logical_counts
        if state == "unbucketed_reference"
        else [_power_of_two_bucket(count) for count in logical_counts]
    )
    if kernel_counts != expected_counts:
        raise CpuEvidenceError("retracing kernel source counts are wrong")
    distinct_logical = _integer(
        row["distinct_logical_source_counts"],
        location=f"{location}.distinct_logical_source_counts",
        positive=True,
    )
    distinct_kernel = _integer(
        row["distinct_kernel_source_counts"],
        location=f"{location}.distinct_kernel_source_counts",
        positive=True,
    )
    if distinct_logical != len(set(logical_counts)) or distinct_kernel != len(
        set(kernel_counts)
    ):
        raise CpuEvidenceError("retracing distinct source-count summary is wrong")
    signatures = [
        _validate_signature(value, location=f"{location}.observed_signatures[{index}]")
        for index, value in enumerate(
            _array(
                row["observed_signatures"], location=f"{location}.observed_signatures"
            )
        )
    ]
    if len({_signature_key(item) for item in signatures}) != len(signatures):
        raise CpuEvidenceError("retracing signatures are not unique")
    positive_counts = [count for count in kernel_counts if count > 0]
    signature_counts = {
        cast(list[int], signature["jones_p_shape"])[1] for signature in signatures
    }
    if signature_counts != set(positive_counts):
        raise CpuEvidenceError("retracing signatures do not cover kernel source counts")
    for source_count in set(positive_counts):
        observed_calls = sum(
            cast(int, signature["call_count"])
            for signature in signatures
            if cast(list[int], signature["jones_p_shape"])[1] == source_count
        )
        if observed_calls < positive_counts.count(source_count):
            raise CpuEvidenceError("retracing signature calls undercount leaf steps")
    distinct_signatures = _integer(
        row["distinct_signature_count"],
        location=f"{location}.distinct_signature_count",
    )
    if distinct_signatures != len(signatures):
        raise CpuEvidenceError("retracing distinct-signature count is wrong")
    leaf_calls = _integer(
        row["leaf_call_count"], location=f"{location}.leaf_call_count"
    )
    if leaf_calls != sum(cast(int, item["call_count"]) for item in signatures):
        raise CpuEvidenceError("retracing leaf-call count is wrong")
    step_seconds = _number_array(
        row["scope_step_seconds"], location=f"{location}.scope_step_seconds"
    )
    if len(step_seconds) != len(logical_counts):
        raise CpuEvidenceError("retracing step timings do not cover logical steps")
    total = _number(
        row["scope_total_seconds"], location=f"{location}.scope_total_seconds"
    )
    if not math.isclose(total, sum(step_seconds), rel_tol=1e-12, abs_tol=1e-15):
        raise CpuEvidenceError("retracing total timing is not derived")
    ratio = max(
        (
            float(item["first_call_seconds"])
            / float(item["minimum_repeat_call_seconds"])
            for item in signatures
        ),
        default=0.0,
    )
    measured_ratio = _number(
        row["max_first_to_repeat_ratio"],
        location=f"{location}.max_first_to_repeat_ratio",
    )
    if not math.isclose(measured_ratio, ratio, rel_tol=1e-12, abs_tol=1e-15):
        raise CpuEvidenceError("retracing ratio is not derived")
    derived_overhead = sum(
        max(
            0.0,
            float(item["first_call_seconds"])
            - float(item["minimum_repeat_call_seconds"]),
        )
        for item in signatures
    )
    overhead = _number(
        row["retrace_overhead_seconds"],
        location=f"{location}.retrace_overhead_seconds",
    )
    if not math.isclose(overhead, derived_overhead, rel_tol=1e-12, abs_tol=1e-15):
        raise CpuEvidenceError("retracing overhead is not derived")
    if overhead > total:
        raise CpuEvidenceError("retracing overhead exceeds scope total")


def _validate_backend_row(
    row: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    location: str,
) -> tuple[float, ...]:
    for name in (
        "comparison_id",
        "implementation_state",
        "operation",
        "requested_backend",
        "resolved_backend",
        "discovery_policy",
        "notes",
    ):
        _ = _string(row[name], location=f"{location}.{name}")
    if context["backend_requested"] != row["requested_backend"]:
        raise CpuEvidenceError("P-c context request differs from measured request")
    if context["backend_actual"] != row["resolved_backend"]:
        raise CpuEvidenceError("P-c context backend differs from resolved backend")
    samples = _integer(
        row["fresh_process_samples"],
        location=f"{location}.fresh_process_samples",
        positive=True,
    )
    cold_seconds = tuple(
        _number_array(row["cold_seconds"], location=f"{location}.cold_seconds")
    )
    if samples != len(cold_seconds):
        raise CpuEvidenceError("P-c sample count is inconsistent")
    return cold_seconds


def _validate_document_mapping(
    decoded: object,
    *,
    approved_source_sha: str,
    filename: str,
) -> dict[str, Any]:
    document = _mapping(decoded, DOCUMENT_FIELDS, location="document")
    if document["schema_version"] != DOCUMENT_SCHEMA:
        raise CpuEvidenceError("document schema_version is wrong")
    collections = {
        name: _array(document[name], location=f"document.{name}")
        for name in DOCUMENT_FIELDS[1:]
    }
    expected_counts = {
        "workload_benchmarks": 24,
        "memory_scaling": 8,
        "solver_memory": 4,
        "retracing": 6,
        "backend_resolution": 3,
    }
    for name, count in expected_counts.items():
        if len(collections[name]) != count:
            raise CpuEvidenceError(f"document.{name} must contain exactly {count} rows")

    provenances: list[dict[str, Any]] = []
    workload_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, value in enumerate(collections["workload_benchmarks"]):
        row, provenance, context = _common_row(
            value,
            WORKLOAD_FIELDS,
            WORKLOAD_SCHEMA,
            location=f"workload[{index}]",
        )
        provenances.append(provenance)
        workload_rows.append((row, context))
        _validate_workload_row(row, context, location=f"workload[{index}]")
        for name in (
            "n_antennas",
            "n_baselines",
            "n_point_sources",
            "n_healpix_pixels",
            "solver_workers",
            "loader_max_workers",
            "peak_host_bytes",
        ):
            _ = _integer(row[name], location=f"workload[{index}].{name}")
        for name in ("n_times", "n_frequencies", "steady_state_iterations"):
            _ = _integer(row[name], location=f"workload[{index}].{name}", positive=True)
        if row["steady_state_iterations"] < 5:
            raise CpuEvidenceError(
                "workload steady-state sample count must be at least 5"
            )
        for name in (
            "setup_seconds",
            "compile_seconds",
            "steady_state_median_seconds",
            "steady_state_min_seconds",
            "steady_state_max_seconds",
            "host_transfer_seconds",
            "max_absolute_deviation",
            "max_relative_deviation",
            "tolerance_rtol",
            "tolerance_atol",
        ):
            _ = _number(row[name], location=f"workload[{index}].{name}")
        if row["within_tolerance"] is not True or row["reference_backend"] != "numpy":
            raise CpuEvidenceError("workload correctness contract failed")
        if "gpu" not in _array(row["unmeasured"], location="workload.unmeasured"):
            raise CpuEvidenceError("CPU workload must state GPU was unmeasured")

    expected_workloads = tuple(
        (workload, backend) for workload in CPU_WORKLOADS for backend in CPU_BACKENDS
    )
    if (
        tuple(
            (row["workload"], context["backend_requested"])
            for row, context in workload_rows
        )
        != expected_workloads
    ):
        raise CpuEvidenceError("workload matrix order or membership is wrong")
    identities: list[str] = []
    for index, workload in enumerate(CPU_WORKLOADS):
        group = workload_rows[index * 3 : index * 3 + 3]
        first, first_context = group[0]
        dimensions = (
            first["n_antennas"],
            first["n_baselines"],
            first["n_point_sources"],
            first["n_healpix_pixels"],
            first["n_times"],
            first["n_frequencies"],
            first["sky_representation"],
        )
        if dimensions != WORKLOAD_DIMENSIONS[workload]:
            raise CpuEvidenceError(f"workload dimensions are wrong for {workload}")
        identity = first_context["input_identity_sha256"]
        identities.append(identity)
        if identity != CPU_CANONICAL_INPUT_IDENTITIES[f"workload:{workload}"]:
            raise CpuEvidenceError(
                f"workload input identity is not canonical for {workload}"
            )
        for backend, (row, context) in zip(CPU_BACKENDS, group, strict=True):
            if context["input_identity_sha256"] != identity:
                raise CpuEvidenceError("matched workload identities differ")
            _require_equal_fields(
                first,
                row,
                (
                    "n_antennas",
                    "n_baselines",
                    "n_point_sources",
                    "n_healpix_pixels",
                    "n_times",
                    "n_frequencies",
                    "sky_representation",
                    "solver_workers",
                    "loader_max_workers",
                ),
                location=f"workload {workload}",
            )
            expected_actual = {
                "numpy": "numpy-cpu",
                "jax": "jax-cpu-cpu",
                "dask": "dask-cpu",
            }[backend]
            if (
                context["backend_actual"] != expected_actual
                or context["device_kind"] != "cpu"
                or context["compilation_used"] is not (backend == "jax")
                or context["precision_preset"] != "standard"
                or context["precision_default"] != "float64"
                or context["precision_accumulation"] != "float64"
                or context["precision_output"] != "float64"
                or context["result_dtype"] != "complex128"
                or context["policy_id"] != "cpu_workload_matrix_v1"
                or row["solver_workers"] != 1
                or row["loader_max_workers"] != 0
            ):
                raise CpuEvidenceError(
                    f"workload backend context is wrong for {workload}"
                )
            if backend in {"numpy", "dask"} and (
                row["max_absolute_deviation"] != 0.0
                or row["max_relative_deviation"] != 0.0
            ):
                raise CpuEvidenceError("NumPy and Dask rows must be byte-identical")
    if len(set(identities)) != len(CPU_WORKLOADS):
        raise CpuEvidenceError("workload fixture identities must be distinct")

    memory_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, value in enumerate(collections["memory_scaling"]):
        row, provenance, context = _common_row(
            value, MEMORY_FIELDS, MEMORY_SCHEMA, location=f"memory[{index}]"
        )
        provenances.append(provenance)
        memory_rows.append((row, context))
        _validate_memory_row(row, context, location=f"memory[{index}]")
    for index, (baselines, sources) in enumerate(MEMORY_FIXTURES):
        (reference, reference_context), (production, production_context) = memory_rows[
            index * 2 : index * 2 + 2
        ]
        comparison = f"p-a-memory-b{baselines}-s{sources}-v1"
        if (
            reference["implementation_state"] != "unchunked_reference"
            or production["implementation_state"] != "chunked_production"
            or reference["comparison_id"] != comparison
            or production["comparison_id"] != comparison
        ):
            raise CpuEvidenceError("P-a memory pair order or comparison ID is wrong")
        for row in (reference, production):
            if (
                row["logical_n_baselines"] != baselines
                or row["logical_n_sources"] != sources
                or row["logical_pair_count"] != baselines * sources
            ):
                raise CpuEvidenceError("P-a memory fixture dimensions are wrong")
        for context in (reference_context, production_context):
            if (
                context["input_identity_sha256"]
                != CPU_CANONICAL_INPUT_IDENTITIES[f"memory:{comparison}"]
            ):
                raise CpuEvidenceError("P-a memory input identity is not canonical")
            if (
                context["backend_requested"] != "numpy"
                or context["backend_actual"] != "numpy-cpu"
                or context["backend_version"] != provenances[0]["numpy_version"]
                or context["device_kind"] != "cpu"
                or context["compilation_used"] is not False
            ):
                raise CpuEvidenceError("P-a memory rows must use NumPy CPU")
            _require_cpu_numeric_context(context, location="P-a memory")
        _require_pair_identity(
            (reference, reference_context),
            (production, production_context),
            row_fields=(
                "logical_n_baselines",
                "logical_n_sources",
                "logical_pair_count",
                "kernel_n_sources",
                "synthetic_input_bytes_excluded",
            ),
            location="memory_scaling",
        )
        if baselines * sources > TARGET_KERNEL_PAIRS and (
            production["peak_host_bytes"] >= reference["peak_host_bytes"]
        ):
            raise CpuEvidenceError("large P-a production peak is not lower")

    solver_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, value in enumerate(collections["solver_memory"]):
        row, provenance, context = _common_row(
            value,
            SOLVER_MEMORY_FIELDS,
            SOLVER_MEMORY_SCHEMA,
            location=f"solver_memory[{index}]",
        )
        provenances.append(provenance)
        solver_rows.append((row, context))
        _validate_solver_memory_row(row, context, location=f"solver_memory[{index}]")
    expected_solver = tuple(
        (solver, state)
        for solver in ("point", "healpix")
        for state in ("unbucketed_reference", "bucketed_production")
    )
    if (
        tuple((row["solver"], row["implementation_state"]) for row, _ in solver_rows)
        != expected_solver
    ):
        raise CpuEvidenceError("solver-memory order or membership is wrong")
    for index, solver in enumerate(("point", "healpix")):
        comparison = f"p-b-solver-memory-{solver}-v1"
        pair = solver_rows[index * 2 : index * 2 + 2]
        if any(row["comparison_id"] != comparison for row, _ in pair):
            raise CpuEvidenceError("solver-memory comparison ID is wrong")
        if any(
            row["logical_n_baselines"] != 3
            or row["logical_source_counts"] != [3]
            or row["n_times"] != 1
            or row["n_frequencies"] != 1
            for row, _ in pair
        ):
            raise CpuEvidenceError("solver-memory fixture dimensions are wrong")
        if any(
            context["input_identity_sha256"]
            != CPU_CANONICAL_INPUT_IDENTITIES[f"solver-memory:{solver}"]
            for _, context in pair
        ):
            raise CpuEvidenceError("solver-memory input identity is not canonical")
        _require_pair_identity(
            pair[0],
            pair[1],
            row_fields=(
                "solver",
                "sky_representation",
                "logical_n_baselines",
                "logical_source_counts",
                "n_times",
                "n_frequencies",
                "target_kernel_pairs",
            ),
            location="solver_memory",
        )
    for row, context in solver_rows:
        if (
            context["backend_requested"] != "jax"
            or context["backend_actual"] != "jax-cpu-cpu"
            or context["backend_version"] != row["provenance"]["jax_version"]
            or context["device_kind"] != "cpu"
            or context["compilation_used"] is not True
        ):
            raise CpuEvidenceError("solver-memory rows must use JAX CPU")
        _require_cpu_numeric_context(context, location="solver-memory")
    if len({solver_rows[index][1]["input_identity_sha256"] for index in (0, 2)}) != 2:
        raise CpuEvidenceError("point and HEALPix solver identities must differ")

    retracing_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, value in enumerate(collections["retracing"]):
        row, provenance, context = _common_row(
            value,
            RETRACING_FIELDS,
            RETRACING_SCHEMA,
            location=f"retracing[{index}]",
        )
        provenances.append(provenance)
        _validate_retracing_row(row, context, location=f"retracing[{index}]")
        retracing_rows.append((row, context))
    expected_retracing = tuple(
        (solver, state)
        for solver in ("synthetic_wrapper", "point", "healpix")
        for state in ("unbucketed_reference", "bucketed_production")
    )
    if (
        tuple((row["solver"], row["implementation_state"]) for row, _ in retracing_rows)
        != expected_retracing
    ):
        raise CpuEvidenceError("retracing order or membership is wrong")
    for index, solver in enumerate(("synthetic_wrapper", "point", "healpix")):
        (reference, reference_context), (production, production_context) = (
            retracing_rows[index * 2 : index * 2 + 2]
        )
        comparison = f"p-b-retracing-{solver.replace('_', '-')}-v1"
        expected_scope = {
            "synthetic_wrapper": "complete_synthetic_contraction_wrapper_step",
            "point": "complete_point_solver_step",
            "healpix": "complete_healpix_solver_step",
        }[solver]
        expected_logical_counts = [3, 4, 5, 8, 3, 4, 5, 8]
        if (
            reference["comparison_id"] != comparison
            or production["comparison_id"] != comparison
            or reference["measurement_scope"] != expected_scope
            or production["measurement_scope"] != expected_scope
            or reference["logical_source_counts"] != expected_logical_counts
            or production["logical_source_counts"] != expected_logical_counts
            or reference["leaf_call_count"] != len(expected_logical_counts)
            or production["leaf_call_count"] != len(expected_logical_counts)
        ):
            raise CpuEvidenceError("retracing fixture is not canonical")
        expected_identity = CPU_CANONICAL_INPUT_IDENTITIES[f"retracing:{solver}"]
        if (
            reference_context["input_identity_sha256"] != expected_identity
            or production_context["input_identity_sha256"] != expected_identity
        ):
            raise CpuEvidenceError("retracing input identity is not canonical")
        _require_pair_identity(
            (reference, reference_context),
            (production, production_context),
            row_fields=(
                "measurement_scope",
                "solver",
                "sky_representation",
                "logical_source_counts",
            ),
            location="retracing",
        )
        if (
            production["distinct_signature_count"]
            >= reference["distinct_signature_count"]
        ):
            raise CpuEvidenceError("production retracing did not reduce signatures")
        if (
            production["retrace_overhead_seconds"]
            >= reference["retrace_overhead_seconds"]
        ):
            raise CpuEvidenceError("production retracing overhead is not lower")
        for row, context in (
            (reference, reference_context),
            (production, production_context),
        ):
            if (
                context["backend_requested"] != "jax"
                or context["backend_actual"] != "jax-cpu-cpu"
                or context["backend_version"] != row["provenance"]["jax_version"]
                or context["device_kind"] != "cpu"
                or context["compilation_used"] is not True
            ):
                raise CpuEvidenceError("retracing rows must use JAX CPU")
            signatures = cast(list[dict[str, Any]], row["observed_signatures"])
            if any(
                cast(list[int], observation["jones_p_shape"])[0] != 3
                for observation in signatures
            ):
                raise CpuEvidenceError("retracing fixture baseline count is wrong")
            _require_cpu_numeric_context(context, location="retracing")
    if (
        len({retracing_rows[index][1]["input_identity_sha256"] for index in (0, 2, 4)})
        != 3
    ):
        raise CpuEvidenceError("synthetic, point, and HEALPix identities must differ")

    backend_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, value in enumerate(collections["backend_resolution"]):
        row, provenance, context = _common_row(
            value, BACKEND_FIELDS, BACKEND_SCHEMA, location=f"backend[{index}]"
        )
        provenances.append(provenance)
        backend_rows.append((row, context))
        _ = _validate_backend_row(row, context, location=f"backend[{index}]")
    backend_contracts = (
        (
            "get_backend_auto",
            "auto",
            "numpy-cpu",
            "p-c-get-backend-auto-v1",
            "deterministic_auto_numpy_v1",
            "numpy_version",
            "cpu",
        ),
        (
            "get_device_resources_default",
            "default",
            "radiosim-device-resources",
            "p-c-get-device-resources-default-v1",
            "platform_device_discovery_v1",
            "radiosim_version",
            "host",
        ),
        (
            "simulator_setup_auto",
            "auto",
            "numpy-cpu",
            "p-c-simulator-setup-auto-v1",
            "deterministic_auto_numpy_v1",
            "numpy_version",
            "cpu",
        ),
    )
    for (row, context), contract in zip(backend_rows, backend_contracts, strict=True):
        operation, requested, resolved, comparison, policy, version_field, device = (
            contract
        )
        if (
            row["operation"] != operation
            or row["requested_backend"] != requested
            or row["resolved_backend"] != resolved
            or row["comparison_id"] != comparison
            or row["implementation_state"] != "production"
            or row["discovery_policy"] != "no_optional_backend_imports"
            or context["policy_id"] != policy
            or context["backend_version"] != row["provenance"][version_field]
            or context["device_kind"] != device
            or context["compilation_used"] is not False
            or row["jax_distribution_installed"] is not True
        ):
            raise CpuEvidenceError("backend-resolution contract is wrong")
        if any(
            context[name] != "not-applicable"
            for name in (
                "precision_preset",
                "precision_default",
                "precision_accumulation",
                "precision_output",
                "result_dtype",
            )
        ):
            raise CpuEvidenceError("P-c result/precision fields are not applicable")
        if any(
            row[name] is not False
            for name in (
                "jax_in_sys_modules_before",
                "jax_in_sys_modules_after",
                "jaxlib_in_sys_modules_before",
                "jaxlib_in_sys_modules_after",
            )
        ):
            raise CpuEvidenceError("P-c imported JAX/JAXlib")
        if (
            context["input_identity_sha256"]
            != CPU_CANONICAL_INPUT_IDENTITIES[f"backend-resolution:{operation}"]
        ):
            raise CpuEvidenceError("P-c control identity is not canonical")
        measured = _validate_backend_row(
            row,
            context,
            location=f"backend.{operation}",
        )
        summaries = (
            (
                _number(row["minimum_seconds"], location="backend.minimum_seconds"),
                min(measured),
            ),
            (
                _number(row["median_seconds"], location="backend.median_seconds"),
                statistics.median(measured),
            ),
            (
                _number(row["maximum_seconds"], location="backend.maximum_seconds"),
                max(measured),
            ),
        )
        if any(
            not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-15)
            for actual, expected in summaries
        ):
            raise CpuEvidenceError("P-c timing summaries are not derived")
    if len({context["input_identity_sha256"] for _, context in backend_rows}) != 3:
        raise CpuEvidenceError("P-c control identities must be distinct")

    if not provenances or any(item != provenances[0] for item in provenances[1:]):
        raise CpuEvidenceError("every one of the 45 rows must share provenance")
    provenance = provenances[0]
    if provenance["git_sha"] != approved_source_sha:
        raise CpuEvidenceError("artifact provenance does not name approved S")
    if provenance["pixi_environment"] != "default":
        raise CpuEvidenceError("CPU evidence must be generated in Pixi default")
    if provenance["pixi_lock_sha256"] != PIXI_LOCK_SHA256:
        raise CpuEvidenceError("artifact provenance has the wrong lock digest")

    match = _FILENAME.fullmatch(filename)
    if match is None:
        raise CpuEvidenceError("artifact filename is not canonical")
    try:
        recorded = datetime.fromisoformat(
            str(provenance["recorded_at_utc"]).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise CpuEvidenceError("artifact recorded_at_utc is invalid") from error
    if recorded.utcoffset() != UTC.utcoffset(recorded):
        raise CpuEvidenceError("artifact recorded_at_utc is not normalized to UTC")
    if match.group("stamp") != recorded.strftime("%Y%m%dT%H%M%SZ"):
        raise CpuEvidenceError("artifact filename timestamp differs from provenance")
    platform_tag = str(provenance["platform"]).split("-", 1)[0].lower()
    expected_system = "darwin" if platform_tag == "macos" else platform_tag
    if match.group("system") != expected_system:
        raise CpuEvidenceError("artifact filename system differs from provenance")
    if match.group("machine") != str(provenance["machine"]).lower():
        raise CpuEvidenceError("artifact filename machine differs from provenance")
    return document


def _canonical_artifact_relative(input_path: str) -> Path:
    relative = Path(input_path)
    if relative.is_absolute() or relative.parent != REFERENCE_DIRECTORY:
        raise CpuEvidenceError("input must be one repository-relative direct child")
    if relative.as_posix() != input_path or _FILENAME.fullmatch(relative.name) is None:
        raise CpuEvidenceError("input path or filename is not canonical")
    return relative


def _read_artifact_snapshot(
    input_path: str,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[Path, bytes]:
    """Open and read one canonical regular direct-child artifact exactly once."""
    relative = _canonical_artifact_relative(input_path)
    root = _lexical_absolute(repository_root)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(root, flags)
    except OSError as error:
        raise CpuEvidenceError(
            "artifact repository root is not a readable non-symlink directory"
        ) from error
    for component in REFERENCE_DIRECTORY.parts:
        try:
            child_descriptor = os.open(
                component,
                flags,
                dir_fd=directory_descriptor,
            )
        except OSError as error:
            os.close(directory_descriptor)
            raise CpuEvidenceError(
                "artifact namespace has an unreadable or symlink component"
            ) from error
        os.close(directory_descriptor)
        directory_descriptor = child_descriptor
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                relative.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
        except OSError as error:
            raise CpuEvidenceError(
                "artifact is not a readable non-symlink file"
            ) from error
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CpuEvidenceError("artifact is not a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            raw = stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_descriptor)
    return relative, raw


def _validate_artifact_bytes(
    raw: bytes,
    *,
    approved_source_sha: str,
    artifact_sha256: str,
    filename: str,
) -> tuple[dict[str, Any], str]:
    """Hash and strictly validate the exact bytes returned by one file read."""
    if _LOWER_HEX_40.fullmatch(approved_source_sha) is None:
        raise CpuEvidenceError("approved source SHA must be lowercase 40-hex")
    if _LOWER_HEX_64.fullmatch(artifact_sha256) is None:
        raise CpuEvidenceError("artifact SHA-256 must be lowercase 64-hex")
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != artifact_sha256:
        raise CpuEvidenceError(
            "artifact byte digest does not match the approved digest"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CpuEvidenceError("artifact must be strict UTF-8") from error
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise CpuEvidenceError("artifact is invalid JSON") from error
    document = _validate_document_mapping(
        decoded,
        approved_source_sha=approved_source_sha,
        filename=filename,
    )
    return document, actual_digest


def load_and_validate_artifact(
    input_path: str,
    *,
    approved_source_sha: str,
    artifact_sha256: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], bytes, str]:
    """Read, hash, and validate one canonical direct-child artifact snapshot."""
    relative, raw = _read_artifact_snapshot(
        input_path,
        repository_root=repository_root,
    )
    document, actual_digest = _validate_artifact_bytes(
        raw,
        approved_source_sha=approved_source_sha,
        artifact_sha256=artifact_sha256,
        filename=relative.name,
    )
    return document, raw, actual_digest


def _authenticate_cli_evidence_edge(
    *,
    input_path: str,
    approved_source_sha: str,
    raw: bytes,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    """Prove that validated bytes are committed in direct successor E of S."""
    relative = _canonical_artifact_relative(input_path)
    if (
        _LOWER_HEX_40.fullmatch(approved_source_sha) is None
        or approved_source_sha == "0" * 40
    ):
        raise CpuEvidenceError("approved source SHA must be nonzero lowercase 40-hex")
    try:
        root = repository_root.resolve(strict=True)
    except OSError as error:
        raise CpuEvidenceError("evidence repository root is unreadable") from error
    if _sha256_path(root / "pixi.toml") != PIXI_MANIFEST_SHA256:
        raise CpuEvidenceError("evidence checkout pixi.toml is not authorized")
    if _sha256_path(root / "pixi.lock") != PIXI_LOCK_SHA256:
        raise CpuEvidenceError("evidence checkout pixi.lock is not authorized")

    def status() -> str:
        try:
            return subprocess.run(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                cwd=root,
                capture_output=True,
                check=True,
                text=True,
                timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise CpuEvidenceError(
                "validation requires a readable clean evidence checkout"
            ) from error

    if status():
        raise CpuEvidenceError("validation requires a clean evidence checkout")
    try:
        committed_raw = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=root,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise CpuEvidenceError(
            "artifact is not readable from committed HEAD"
        ) from error
    if committed_raw != raw:
        raise CpuEvidenceError("artifact bytes differ from committed HEAD")
    try:
        _ = subprocess.run(
            ["git", "cat-file", "-e", f"{approved_source_sha}^{{commit}}"],
            cwd=root,
            capture_output=True,
            check=True,
            timeout=30,
        )
        evidence_parent = subprocess.run(
            ["git", "rev-parse", "HEAD^"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise CpuEvidenceError(
            "artifact does not name a valid generating commit and evidence-successor edge"
        ) from error
    if evidence_parent != approved_source_sha:
        raise CpuEvidenceError(
            "approved source is not the direct parent of the evidence commit"
        )
    try:
        source_artifact = subprocess.run(
            [
                "git",
                "cat-file",
                "-e",
                f"{approved_source_sha}:{relative.as_posix()}",
            ],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=30,
        )
        immutable_generator = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                approved_source_sha,
                "HEAD",
                "--",
                TOOL_RELATIVE_PATH.as_posix(),
                "pixi.toml",
                "pixi.lock",
            ],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CpuEvidenceError(
            "generating source edge could not be authenticated"
        ) from error
    if source_artifact.returncode == 0:
        raise CpuEvidenceError("artifact was already present in generating source S")
    if source_artifact.returncode != 128:
        raise CpuEvidenceError("generating source artifact absence cannot be proven")
    if immutable_generator.returncode != 0:
        raise CpuEvidenceError(
            "CPU generator and Pixi lock must be unchanged across S..E"
        )
    if status():
        raise CpuEvidenceError("validation requires a clean evidence checkout")
    try:
        committed_recheck = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=root,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise CpuEvidenceError("committed artifact could not be rechecked") from error
    if committed_recheck != raw:
        raise CpuEvidenceError("artifact bytes changed during authentication")


def _git_bytes(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    operation: str,
) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise CpuEvidenceError(f"{operation} failed") from error


def _git_text(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    operation: str,
) -> str:
    try:
        return _git_bytes(
            repository_root,
            arguments,
            operation=operation,
        ).decode("utf-8")
    except UnicodeDecodeError as error:
        raise CpuEvidenceError(f"{operation} did not return UTF-8") from error


def _require_commit(repository_root: Path, commit: str, *, label: str) -> str:
    if _LOWER_HEX_40.fullmatch(commit) is None or commit == "0" * 40:
        raise CpuEvidenceError(f"{label} must be nonzero lowercase 40-hex")
    resolved = _git_text(
        repository_root,
        ["rev-parse", "--verify", f"{commit}^{{commit}}"],
        operation=f"{label} commit authentication",
    ).strip()
    if resolved != commit:
        raise CpuEvidenceError(f"{label} does not resolve to the exact commit")
    return resolved


def _single_parent(repository_root: Path, commit: str, *, label: str) -> str:
    ancestry = _git_text(
        repository_root,
        ["rev-list", "--parents", "-n", "1", commit],
        operation=f"{label} parent authentication",
    ).split()
    if len(ancestry) != 2 or ancestry[0] != commit:
        raise CpuEvidenceError(f"{label} must be a non-merge direct-child commit")
    parent = ancestry[1]
    if _LOWER_HEX_40.fullmatch(parent) is None:
        raise CpuEvidenceError(f"{label} parent is not canonical")
    return parent


def _git_diff_paths(repository_root: Path, before: str, after: str) -> tuple[str, ...]:
    raw = _git_bytes(
        repository_root,
        ["diff", "--name-only", "-z", before, after, "--"],
        operation=f"{before}..{after} diff authentication",
    )
    paths: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            path = item.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CpuEvidenceError("certificate diff path is not UTF-8") from error
        if not path or Path(path).is_absolute() or path in paths:
            raise CpuEvidenceError("certificate diff contains a noncanonical path")
        paths.append(path)
    return tuple(sorted(paths))


def _reference_manifests_from_source(
    raw: bytes,
) -> tuple[dict[str, str], dict[str, str]]:
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=HARNESS_RELATIVE_PATH.as_posix())
    except (UnicodeDecodeError, SyntaxError) as error:
        raise CpuEvidenceError(
            "committed benchmark harness is not valid UTF-8 Python"
        ) from error
    targets = {"PERF001_REFERENCE_SHA256", "PERF001_REFERENCE_SOURCE_SHA"}
    values: dict[str, dict[str, str]] = {}
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        if not isinstance(target, ast.Name) or target.id not in targets:
            continue
        if target.id in values or value is None:
            raise CpuEvidenceError(
                "committed reference manifest is assigned more than once"
            )
        if not isinstance(value, ast.Dict) or len(value.keys) != len(value.values):
            raise CpuEvidenceError("committed reference manifest is not literal")
        entries: list[tuple[str, str]] = []
        for key_node, value_node in zip(value.keys, value.values, strict=True):
            if (
                not isinstance(key_node, ast.Constant)
                or type(key_node.value) is not str
                or not isinstance(value_node, ast.Constant)
                or type(value_node.value) is not str
            ):
                raise CpuEvidenceError("committed reference manifest has invalid types")
            entries.append((key_node.value, value_node.value))
        if len({key for key, _ in entries}) != len(entries):
            raise CpuEvidenceError("committed reference manifest has duplicate keys")
        decoded = dict(entries)
        if any(not key or not item for key, item in decoded.items()):
            raise CpuEvidenceError("committed reference manifest has invalid types")
        values[target.id] = decoded
    if set(values) != targets:
        raise CpuEvidenceError("committed benchmark harness lacks reference manifests")
    return values["PERF001_REFERENCE_SHA256"], values["PERF001_REFERENCE_SOURCE_SHA"]


def _normalized_reference_manifest_source(raw: bytes) -> bytes:
    """Normalize only the two retained-map RHS spans for S/E byte comparison."""
    _ = _reference_manifests_from_source(raw)
    try:
        tree = ast.parse(
            raw.decode("utf-8"),
            filename=HARNESS_RELATIVE_PATH.as_posix(),
        )
    except (UnicodeDecodeError, SyntaxError) as error:  # pragma: no cover
        raise CpuEvidenceError("committed benchmark harness is invalid") from error
    targets = {"PERF001_REFERENCE_SHA256", "PERF001_REFERENCE_SOURCE_SHA"}
    nodes: dict[str, ast.expr] = {}
    for statement in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
        if isinstance(target, ast.Name) and target.id in targets and value is not None:
            nodes[target.id] = value
    if set(nodes) != targets:
        raise CpuEvidenceError("committed benchmark harness lacks reference spans")
    line_offsets = [0]
    for line in raw.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))
    spans: list[tuple[int, int, str]] = []
    for name, node in nodes.items():
        if node.end_lineno is None or node.end_col_offset is None:
            raise CpuEvidenceError("committed reference manifest span is incomplete")
        try:
            start = line_offsets[node.lineno - 1] + node.col_offset
            end = line_offsets[node.end_lineno - 1] + node.end_col_offset
        except IndexError as error:  # pragma: no cover - ast/raw invariant
            raise CpuEvidenceError(
                "committed reference manifest span is outside source bytes"
            ) from error
        if start < 0 or end <= start or end > len(raw):
            raise CpuEvidenceError("committed reference manifest span is invalid")
        spans.append((start, end, name))
    normalized = bytearray(raw)
    for start, end, name in sorted(spans, reverse=True):
        normalized[start:end] = f"<{name}:LITERAL_MAP>".encode("ascii")
    return bytes(normalized)


def _require_reference_manifest_only_change(
    source_harness: bytes,
    evidence_harness: bytes,
) -> None:
    if _normalized_reference_manifest_source(
        source_harness
    ) != _normalized_reference_manifest_source(evidence_harness):
        raise CpuEvidenceError(
            "S..E harness bytes may change only the two literal reference-map RHS spans"
        )


def _replace_unique_sentinel(
    raw: bytes,
    *,
    sentinel: str,
    replacement: str,
    label: str,
) -> bytes:
    marker = sentinel.encode("utf-8")
    if raw.count(marker) != 1:
        raise CpuEvidenceError(f"{label} sentinel must occur exactly once")
    return raw.replace(marker, replacement.encode("utf-8"), 1)


def _expected_evidence_memo(
    source_memo: bytes,
    *,
    source_sha: str,
    artifact_sha256: str,
    artifact_path: str,
) -> bytes:
    if _LOWER_HEX_40.fullmatch(source_sha) is None or source_sha == "0" * 40:
        raise CpuEvidenceError("evidence memo source SHA is not canonical")
    _ = _digest(artifact_sha256, location="evidence memo artifact_sha256")
    _ = _canonical_artifact_relative(artifact_path)
    replacement = (
        "```console\n"
        "pixi run python tools/wp7_perf001_cpu_evidence.py generate "
        f"--approved-source-sha {source_sha}\n"
        "pixi run python tools/wp7_perf001_cpu_evidence.py validate "
        f"--approved-source-sha {source_sha} "
        f"--artifact-sha256 {artifact_sha256} --input {artifact_path}\n"
        "```"
    )
    return _replace_unique_sentinel(
        source_memo,
        sentinel=EVIDENCE_REPRODUCTION_SENTINEL,
        replacement=replacement,
        label="PERF-001 evidence reproduction",
    )


def _expected_acceptance_memo(evidence_memo: bytes) -> bytes:
    return _replace_unique_sentinel(
        evidence_memo,
        sentinel=ACCEPTANCE_MEMO_STATUS_SENTINEL,
        replacement=ACCEPTANCE_STATUS_LINE,
        label="PERF-001 memo acceptance",
    )


def _expected_acceptance_plan(evidence_plan: bytes) -> bytes:
    return _replace_unique_sentinel(
        evidence_plan,
        sentinel=ACCEPTANCE_PLAN_STATUS_SENTINEL,
        replacement=ACCEPTANCE_PLAN_STATUS_ROW,
        label="PERF-001 plan acceptance",
    )


def _authenticate_acceptance_document_transforms(
    *,
    source_memo: bytes,
    source_plan: bytes,
    evidence_memo: bytes,
    evidence_plan: bytes,
    acceptance_memo: bytes,
    acceptance_plan: bytes,
    descendant_memo: bytes,
    descendant_plan: bytes,
    source_sha: str,
    artifact_sha256: str,
    artifact_path: str,
) -> None:
    expected_evidence_memo = _expected_evidence_memo(
        source_memo,
        source_sha=source_sha,
        artifact_sha256=artifact_sha256,
        artifact_path=artifact_path,
    )
    expected_acceptance_memo = _expected_acceptance_memo(expected_evidence_memo)
    expected_acceptance_plan = _expected_acceptance_plan(source_plan)
    if (
        evidence_memo != expected_evidence_memo
        or evidence_plan != source_plan
        or acceptance_memo != expected_acceptance_memo
        or acceptance_plan != expected_acceptance_plan
        or descendant_memo != acceptance_memo
        or descendant_plan != acceptance_plan
    ):
        raise CpuEvidenceError(
            "PERF-001 status documents are not the exact byte transformation"
        )


def _require_fix_perf001_roadmap_row(raw: bytes, *, label: str) -> None:
    row_bytes = (FIX_PERF001_ROADMAP_ROW + "\n").encode("utf-8")
    if (
        hashlib.sha256(row_bytes).hexdigest() != FIX_PERF001_ROADMAP_ROW_SHA256
        or raw.count(row_bytes) != 1
    ):
        raise CpuEvidenceError(
            f"{label} Fix.md lacks the exact unique ROADMAP row for PERF-001"
        )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise CpuEvidenceError(f"{label} Fix.md is not UTF-8") from error
    perf001_rows = [
        line for line in lines if re.match(r"^\|\s*`?PERF-001`?\s*\|", line) is not None
    ]
    if perf001_rows != [FIX_PERF001_ROADMAP_ROW]:
        raise CpuEvidenceError(
            f"{label} Fix.md lacks the exact unique ROADMAP row for PERF-001"
        )


def verify_accepted_cpu_certificate(
    *,
    acceptance_commit: str,
    descendant: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Authenticate the exact S -> E -> A CPU-acceptance edge from descendant D."""
    try:
        root = repository_root.resolve(strict=True)
    except OSError as error:
        raise CpuEvidenceError("acceptance repository root is unreadable") from error
    top_level = _git_text(
        root,
        ["rev-parse", "--show-toplevel"],
        operation="acceptance Git top-level authentication",
    ).strip()
    if _lexical_absolute(top_level) != root:
        raise CpuEvidenceError("acceptance repository root is not the Git top-level")
    acceptance = _require_commit(root, acceptance_commit, label="acceptance A")
    descendant_commit = _require_commit(root, descendant, label="descendant D")
    head = _git_text(
        root,
        ["rev-parse", "HEAD"],
        operation="acceptance HEAD authentication",
    ).strip()
    if head != descendant_commit:
        raise CpuEvidenceError("acceptance verification requires HEAD == descendant D")
    status = _git_text(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        operation="acceptance worktree authentication",
    )
    if status:
        raise CpuEvidenceError("acceptance verification requires a clean checkout")
    evidence = _single_parent(root, acceptance, label="acceptance A")
    source = _single_parent(root, evidence, label="evidence E")
    try:
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", acceptance, descendant_commit],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CpuEvidenceError(
            "acceptance descendant ancestry could not be proven"
        ) from error
    if ancestry.returncode != 0:
        raise CpuEvidenceError("acceptance A is not an ancestor of descendant D")

    evidence_paths = _git_diff_paths(root, source, evidence)
    artifact_paths = tuple(
        path
        for path in evidence_paths
        if Path(path).parent == REFERENCE_DIRECTORY
        and _FILENAME.fullmatch(Path(path).name) is not None
    )
    if len(artifact_paths) != 1 or set(evidence_paths) != set(
        EVIDENCE_FIXED_DIFF_PATHS | {artifact_paths[0]}
    ):
        raise CpuEvidenceError(
            "S..E must contain exactly one CPU JSON, its harness pins, and memo instructions"
        )
    artifact_path = artifact_paths[0]
    acceptance_paths = _git_diff_paths(root, evidence, acceptance)
    if set(acceptance_paths) != set(ACCEPTANCE_DIFF_PATHS):
        raise CpuEvidenceError(
            "E..A must contain exactly the PERF-001 memo and live-plan status edits"
        )

    source_harness = _git_bytes(
        root,
        ["show", f"{source}:{HARNESS_RELATIVE_PATH.as_posix()}"],
        operation="source harness authentication",
    )
    evidence_harness = _git_bytes(
        root,
        ["show", f"{evidence}:{HARNESS_RELATIVE_PATH.as_posix()}"],
        operation="evidence harness authentication",
    )
    source_digests, source_sources = _reference_manifests_from_source(source_harness)
    if source_digests or source_sources:
        raise CpuEvidenceError("source S retained reference manifests must be empty")
    _require_reference_manifest_only_change(source_harness, evidence_harness)
    raw = _git_bytes(
        root,
        ["show", f"{evidence}:{artifact_path}"],
        operation="evidence artifact authentication",
    )
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    _document, observed_digest = _validate_artifact_bytes(
        raw,
        approved_source_sha=source,
        artifact_sha256=artifact_sha256,
        filename=Path(artifact_path).name,
    )
    evidence_digests, evidence_sources = _reference_manifests_from_source(
        evidence_harness
    )
    if evidence_digests != {artifact_path: observed_digest} or evidence_sources != {
        artifact_path: source
    }:
        raise CpuEvidenceError(
            "evidence E harness pins do not exactly bind the artifact"
        )
    try:
        source_artifact = subprocess.run(
            ["git", "cat-file", "-e", f"{source}:{artifact_path}"],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CpuEvidenceError(
            "source artifact absence could not be checked"
        ) from error
    if source_artifact.returncode == 0:
        raise CpuEvidenceError("evidence artifact was already present in source S")
    if source_artifact.returncode != 128:
        raise CpuEvidenceError("source artifact absence could not be proven")
    for commit, label in (
        (acceptance, "acceptance A"),
        (descendant_commit, "descendant D"),
    ):
        if (
            _git_bytes(
                root,
                ["show", f"{commit}:{artifact_path}"],
                operation=f"{label} artifact authentication",
            )
            != raw
        ):
            raise CpuEvidenceError(f"{label} changed the accepted artifact bytes")

    protected_paths = (
        TOOL_RELATIVE_PATH,
        RECORD_RELATIVE_PATH,
        Path("pixi.toml"),
        Path("pixi.lock"),
    )
    protected_bytes: dict[str, bytes] = {}
    for path in protected_paths:
        relative = path.as_posix()
        source_bytes = _git_bytes(
            root,
            ["show", f"{source}:{relative}"],
            operation=f"source {relative} authentication",
        )
        for commit, label in (
            (evidence, "evidence E"),
            (acceptance, "acceptance A"),
            (descendant_commit, "descendant D"),
        ):
            if (
                _git_bytes(
                    root,
                    ["show", f"{commit}:{relative}"],
                    operation=f"{label} {relative} authentication",
                )
                != source_bytes
            ):
                raise CpuEvidenceError(
                    f"protected acceptance source changed after S: {relative}"
                )
        protected_bytes[relative] = source_bytes
    if hashlib.sha256(protected_bytes["pixi.toml"]).hexdigest() != PIXI_MANIFEST_SHA256:
        raise CpuEvidenceError("accepted pixi.toml digest is not authorized")
    if hashlib.sha256(protected_bytes["pixi.lock"]).hexdigest() != PIXI_LOCK_SHA256:
        raise CpuEvidenceError("accepted pixi.lock digest is not authorized")
    current_tool = _regular_file_bytes(
        Path(__file__),
        label="executing CPU acceptance tool",
    )
    if current_tool != protected_bytes[TOOL_RELATIVE_PATH.as_posix()]:
        raise CpuEvidenceError("executing CPU acceptance tool differs from committed D")
    acceptance_harness = _git_bytes(
        root,
        ["show", f"{acceptance}:{HARNESS_RELATIVE_PATH.as_posix()}"],
        operation="acceptance harness authentication",
    )
    descendant_harness = _git_bytes(
        root,
        ["show", f"{descendant_commit}:{HARNESS_RELATIVE_PATH.as_posix()}"],
        operation="descendant harness authentication",
    )
    if acceptance_harness != evidence_harness or descendant_harness != evidence_harness:
        raise CpuEvidenceError("accepted harness pins changed after evidence E")

    source_memo = _git_bytes(
        root,
        ["show", f"{source}:{PERF001_MEMO_RELATIVE_PATH.as_posix()}"],
        operation="source memo authentication",
    )
    source_plan = _git_bytes(
        root,
        ["show", f"{source}:{LIVE_PLAN_RELATIVE_PATH.as_posix()}"],
        operation="source live-plan authentication",
    )
    if hashlib.sha256(source_memo).hexdigest() != ACCEPTED_SOURCE_MEMO_SHA256:
        raise CpuEvidenceError("source S PERF-001 memo bytes are not authorized")
    if hashlib.sha256(source_plan).hexdigest() != ACCEPTED_SOURCE_PLAN_SHA256:
        raise CpuEvidenceError("source S live-plan bytes are not authorized")
    evidence_memo = _git_bytes(
        root,
        ["show", f"{evidence}:{PERF001_MEMO_RELATIVE_PATH.as_posix()}"],
        operation="evidence memo authentication",
    )
    evidence_plan = _git_bytes(
        root,
        ["show", f"{evidence}:{LIVE_PLAN_RELATIVE_PATH.as_posix()}"],
        operation="evidence live-plan authentication",
    )
    acceptance_memo = _git_bytes(
        root,
        ["show", f"{acceptance}:{PERF001_MEMO_RELATIVE_PATH.as_posix()}"],
        operation="acceptance memo authentication",
    )
    acceptance_plan = _git_bytes(
        root,
        ["show", f"{acceptance}:{LIVE_PLAN_RELATIVE_PATH.as_posix()}"],
        operation="acceptance live-plan authentication",
    )
    descendant_memo = _git_bytes(
        root,
        ["show", f"{descendant_commit}:{PERF001_MEMO_RELATIVE_PATH.as_posix()}"],
        operation="descendant memo authentication",
    )
    descendant_plan = _git_bytes(
        root,
        ["show", f"{descendant_commit}:{LIVE_PLAN_RELATIVE_PATH.as_posix()}"],
        operation="descendant live-plan authentication",
    )
    _authenticate_acceptance_document_transforms(
        source_memo=source_memo,
        source_plan=source_plan,
        evidence_memo=evidence_memo,
        evidence_plan=evidence_plan,
        acceptance_memo=acceptance_memo,
        acceptance_plan=acceptance_plan,
        descendant_memo=descendant_memo,
        descendant_plan=descendant_plan,
        source_sha=source,
        artifact_sha256=artifact_sha256,
        artifact_path=artifact_path,
    )
    for commit, label in (
        (source, "source S"),
        (evidence, "evidence E"),
        (acceptance, "acceptance A"),
        (descendant_commit, "descendant D"),
    ):
        _require_fix_perf001_roadmap_row(
            _git_bytes(
                root,
                ["show", f"{commit}:{FIX_RELATIVE_PATH.as_posix()}"],
                operation=f"{label} Fix.md authentication",
            ),
            label=label,
        )

    return {
        "schema_version": ACCEPTANCE_CERTIFICATE_SCHEMA,
        "acceptance_commit": acceptance,
        "evidence_commit": evidence,
        "generating_source_sha": source,
        "descendant_commit": descendant_commit,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "cpu_evidence_tool_sha256": hashlib.sha256(
            protected_bytes[TOOL_RELATIVE_PATH.as_posix()]
        ).hexdigest(),
        "production_record_validator_sha256": hashlib.sha256(
            protected_bytes[RECORD_RELATIVE_PATH.as_posix()]
        ).hexdigest(),
        "production_harness_sha256": hashlib.sha256(evidence_harness).hexdigest(),
        "pixi_manifest_sha256": PIXI_MANIFEST_SHA256,
        "pixi_lock_sha256": PIXI_LOCK_SHA256,
        "evidence_diff_paths": list(evidence_paths),
        "acceptance_diff_paths": list(acceptance_paths),
        "verdict": ACCEPTANCE_VERDICT,
        "passed": True,
    }


def _loaded_runtime_binding(root: Path) -> None:
    import radiosim
    import radiosim.benchmarks.harness as harness
    import radiosim.benchmarks.record as record

    expected = {
        "radiosim": root / "src/radiosim/__init__.py",
        "harness": root / "src/radiosim/benchmarks/harness.py",
        "record": root / "src/radiosim/benchmarks/record.py",
        "tool": root / TOOL_RELATIVE_PATH,
    }
    actual = {
        "radiosim": Path(radiosim.__file__),
        "harness": Path(harness.__file__),
        "record": Path(record.__file__),
        "tool": Path(__file__),
    }
    for name, path in actual.items():
        try:
            resolved = path.resolve(strict=True)
            expected_resolved = expected[name].resolve(strict=True)
        except OSError as error:
            raise CpuEvidenceError(
                f"loaded {name} source path is unreadable"
            ) from error
        if resolved != expected_resolved:
            raise CpuEvidenceError(f"loaded {name} source is outside approved checkout")


def _verify_loaded_snapshot_modules(snapshot: SourceSnapshot) -> None:
    expected = {relative: digest for relative, _, _, digest in snapshot.entries}
    source_directory = snapshot.root / "src"
    checked = 0
    for name, module in tuple(sys.modules.items()):
        if name != "radiosim" and not name.startswith("radiosim."):
            continue
        raw_path = getattr(module, "__file__", None)
        if type(raw_path) is not str or not raw_path:
            raise CpuEvidenceError(
                f"loaded RadioSim module {name!r} has no source file"
            )
        try:
            path = Path(raw_path).resolve(strict=True)
        except OSError as error:
            raise CpuEvidenceError(
                f"loaded RadioSim module {name!r} source is unreadable"
            ) from error
        if not path.is_relative_to(source_directory):
            raise CpuEvidenceError(
                f"loaded RadioSim module {name!r} escaped the source snapshot"
            )
        module_spec = getattr(module, "__spec__", None)
        raw_origin = getattr(module_spec, "origin", None)
        if type(raw_origin) is not str or not raw_origin:
            raise CpuEvidenceError(
                f"loaded RadioSim module {name!r} has no import-spec origin"
            )
        try:
            origin = Path(raw_origin).resolve(strict=True)
        except OSError as error:
            raise CpuEvidenceError(
                f"loaded RadioSim module {name!r} import origin is unreadable"
            ) from error
        if origin != path:
            raise CpuEvidenceError(
                f"loaded RadioSim module {name!r} import origin differs from __file__"
            )
        relative = path.relative_to(snapshot.root).as_posix()
        expected_digest = expected.get(relative)
        if expected_digest is None or _sha256_path(path) != expected_digest:
            raise CpuEvidenceError(
                f"loaded RadioSim module {name!r} bytes differ from approved S"
            )
        checked += 1
    if checked == 0:
        raise CpuEvidenceError("snapshot worker loaded no RadioSim modules")
    tool_path = snapshot.root / TOOL_RELATIVE_PATH
    if _sha256_path(tool_path) != expected[TOOL_RELATIVE_PATH.as_posix()]:
        raise CpuEvidenceError("loaded CPU evidence tool bytes differ from approved S")


def assemble_document(
    *,
    workload_benchmarks: Sequence[Any],
    memory_scaling: Sequence[Any],
    solver_memory: Sequence[Any],
    retracing: Sequence[Any],
    backend_resolution: Sequence[Any],
) -> Any:
    """Assemble the complete in-memory CPU document without publishing it."""
    from radiosim.benchmarks import assemble_perf001_cpu_evidence_document

    return assemble_perf001_cpu_evidence_document(
        workload_benchmarks=workload_benchmarks,
        memory_scaling=memory_scaling,
        solver_memory=solver_memory,
        retracing=retracing,
        backend_resolution=backend_resolution,
    )


def _minimal_configuration(base_dir: Path) -> dict[str, Any]:
    antenna_path = base_dir / "antennas.txt"
    _ = antenna_path.write_bytes(CANONICAL_ANTENNA_LAYOUT_BYTES)
    return {
        "instrument": {
            "source": {
                "kind": "layout_file",
                "path": "antennas.txt",
                "format": "radiosim",
                "telescope_name": "Perf001CpuArray",
            },
            "location": dict(CANONICAL_LOCATION),
            "default_diameter_m": 14.0,
        },
        "baseline_selection": {"correlations": "all"},
        "beams": json.loads(json.dumps(CANONICAL_BEAM_CONFIGURATION)),
        "receptors": json.loads(
            json.dumps(CANONICAL_HOMOGENEOUS_RECEPTOR_CONFIGURATION)
        ),
        "obs_time": {
            "start_time": "2025-01-01T00:00:00",
            "duration_seconds": 2.0,
            "time_step_seconds": 1.0,
        },
        "obs_frequency": {
            "mode": "grid",
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 2.0,
            "channel_width": 1.0,
            "frequency_unit": "MHz",
        },
        "sky_model": {
            "flux_unit": "Jy",
            "sources": [
                {
                    "kind": "test_sources",
                    "representation": "point_sources",
                    "num_sources": 2,
                    "distribution": "uniform",
                    "seed": 1,
                }
            ],
        },
        "visibility": {"sky_representation": "point_sources"},
        "execution": {
            "backend": "numpy",
            "offline": True,
            "precision": {"preset": "standard"},
            "simulator": "rime",
        },
        "workflow": {
            "output_dir": "output",
            "run_subdir": "run",
            "result_filename": "visibilities",
            "result_format": "hdf5",
            "collision_policy": "error",
            "save_results": False,
            "plot_results": False,
            "open_plots_in_browser": False,
            "save_log": False,
        },
    }


def _canonical_point_source_arrays(
    np: Any,
    *,
    lst_rad: float,
    count: int,
    polarized: bool,
    gaussian: bool,
    seed: int = 20260811,
    spread: bool = False,
) -> dict[str, Any]:
    """Return the production-owned canonical PERF-001 point fixture arrays."""
    if spread:
        rng = np.random.default_rng(seed)
        ra = lst_rad + rng.uniform(-0.05, 0.05, count)
        dec = -0.536 + rng.uniform(-0.05, 0.05, count)
        flux = rng.uniform(0.5, 5.0, count)
        spectral_index = np.full(count, -0.7, dtype=np.float64)
    else:
        ra = lst_rad + np.arange(count, dtype=np.float64) * 0.01
        dec = -0.536 + np.arange(count, dtype=np.float64) * 0.01
        flux = np.linspace(2.0, 1.0, count, dtype=np.float64)
        spectral_index = np.linspace(-0.7, -0.8, count, dtype=np.float64)
    zeros = np.zeros(count, dtype=np.float64)
    q = zeros.copy()
    u_stokes = zeros.copy()
    v = zeros.copy()
    if polarized and count:
        q[0] = 0.2
        u_stokes[min(1, count - 1)] = 0.1
        v[0] = 0.05
    return {
        "ra_rad": np.asarray(ra, dtype=np.float64),
        "dec_rad": np.asarray(dec, dtype=np.float64),
        "flux": np.asarray(flux, dtype=np.float64),
        "spectral_index": spectral_index,
        "stokes_q": q,
        "stokes_u": u_stokes,
        "stokes_v": v,
        "ref_freq": np.full(count, 100e6, dtype=np.float64),
        "rotation_measure": zeros.copy(),
        "spectral_coeffs": None,
        "per_channel_flux": None,
        "per_channel_stokes_q": None,
        "per_channel_stokes_u": None,
        "per_channel_stokes_v": None,
        "channel_frequencies": None,
        "major_arcsec": (
            np.full(count, 120.0, dtype=np.float64) if gaussian else zeros.copy()
        ),
        "minor_arcsec": (
            np.full(count, 60.0, dtype=np.float64) if gaussian else zeros.copy()
        ),
        "pa_deg": (
            np.full(count, 30.0, dtype=np.float64) if gaussian else zeros.copy()
        ),
    }


def _measure_document(
    provenance: Any,
    fixture_root: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    loaded_source_root: Path | None = None,
) -> Any:
    """Measure all 45 rows from production code and canonical real fixtures."""
    # Heavy and optional imports are deliberately confined below preflight.
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation
    from astropy.time import Time

    from radiosim.api import Simulator
    from radiosim.backends import (
        get_backend,  # pyright: ignore[reportUnknownVariableType]
    )
    from radiosim.benchmarks import (
        WorkloadShape,
        build_perf001_workload_record,
        measure_perf001_backend_resolution,
        measure_perf001_memory_scaling_pair,
        measure_perf001_solver_memory_pair,
        measure_perf001_solver_retracing_pair,
        measure_perf001_synthetic_retracing_pair,
        time_backend_call,
    )
    from radiosim.core.instrument_adapters import SolverInstrumentView
    from radiosim.core.precision import PrecisionConfig
    from radiosim.core.sky import (
        BrightnessConversion,
        HealpixData,
        SkyModel,
        SourceArrays,
    )
    from radiosim.core.time_grid import build_observation_time_grid
    from radiosim.core.visibility import _calculate_visibility, calculate_visibility
    from radiosim.core.visibility_healpix import (
        _calculate_visibility_healpix,
        calculate_visibility_healpix,
    )

    configuration = _minimal_configuration(fixture_root)
    layout_sha256 = hashlib.sha256(CANONICAL_ANTENNA_LAYOUT_BYTES).hexdigest()
    get_backend_runtime = cast(Callable[..., Any], get_backend)
    units = cast(Any, u)
    earth_location_type = cast(Any, EarthLocation)
    altaz_type = cast(Any, AltAz)
    time_type = cast(Any, Time)

    def components(*, heterogeneous: bool = False) -> tuple[Any, Any, Any]:
        selected = json.loads(json.dumps(configuration))
        if heterogeneous:
            selected["receptors"] = json.loads(
                json.dumps(CANONICAL_HETEROGENEOUS_RECEPTOR_CONFIGURATION)
            )
        simulator = Simulator.from_mapping(selected, base_dir=fixture_root)
        simulator._ensure_instrument_state()
        simulator._ensure_receptor_set()
        simulator._ensure_beam_system()
        instrument_state = simulator._instrument_state
        if instrument_state is None:
            raise CpuEvidenceError("canonical simulator did not resolve an instrument")
        return (
            SolverInstrumentView.from_state(instrument_state),
            simulator.beam_system,
            simulator.receptors,
        )

    homogeneous = components()
    heterogeneous = components(heterogeneous=True)
    location: Any = earth_location_type.from_geodetic(
        21.4283 * units.deg, -30.72152 * units.deg, 1073.0 * units.m
    )
    obstime: Any = time_type("2025-01-01T00:00:00")
    start_time = str(obstime.isot)
    frequencies = np.array([100e6, 101e6], dtype=np.float64)
    time_grid = build_observation_time_grid(
        start_time=start_time, duration_seconds=2.0, cadence_seconds=1.0
    )
    single_time_grid = build_observation_time_grid(
        start_time=start_time, duration_seconds=1.0, cadence_seconds=1.0
    )
    scaled_time_grid = build_observation_time_grid(
        start_time=start_time, duration_seconds=4.0, cadence_seconds=1.0
    )
    lst_rad = float(obstime.sidereal_time("apparent", longitude=location.lon).rad)

    def point_sources(
        count: int,
        *,
        polarized: bool,
        gaussian: bool,
        seed: int = 20260811,
        spread: bool = False,
    ) -> SourceArrays:
        return cast(
            SourceArrays,
            _canonical_point_source_arrays(
                np,
                lst_rad=lst_rad,
                count=count,
                polarized=polarized,
                gaussian=gaussian,
                seed=seed,
                spread=spread,
            ),
        )

    def healpix_model(*, polarized: bool) -> SkyModel:
        maps = np.linspace(1.0, 2.0, 12, dtype=np.float64)
        maps = np.vstack([maps, maps * 1.1])
        return SkyModel(
            healpix=HealpixData(
                maps=maps,
                nside=1,
                frequencies=frequencies,
                coordinate_frame="icrs",
                q_maps=np.full_like(maps, 0.1) if polarized else None,
                u_maps=np.full_like(maps, 0.05) if polarized else None,
                v_maps=np.full_like(maps, 0.02) if polarized else None,
            ),
            model_name="perf001-workload-healpix-v1",
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
            precision=PrecisionConfig.standard(),
        )

    def healpix_metadata(model: SkyModel, *, storage: str) -> dict[str, Any]:
        payload = model.healpix
        if payload is None:
            raise CpuEvidenceError("canonical HEALPix fixture has no payload")
        return {
            "storage": storage,
            "nside": payload.nside,
            "ordering": payload.ordering,
            "coordinate_frame": payload.coordinate_frame,
            "i_unit": payload.i_unit,
            "q_unit": payload.q_unit,
            "u_unit": payload.u_unit,
            "v_unit": payload.v_unit,
            "i_brightness_conversion": payload.i_brightness_conversion,
            "q_brightness_conversion": payload.q_brightness_conversion,
            "u_brightness_conversion": payload.u_brightness_conversion,
            "v_brightness_conversion": payload.v_brightness_conversion,
            "sky_brightness_conversion": model.brightness_conversion.value,
            "polarized": any(
                getattr(payload, name) is not None
                for name in ("q_maps", "u_maps", "v_maps")
            ),
        }

    def scientific_runtime_manifest(
        selected: Any,
        *,
        heterogeneous_receptors: bool,
    ) -> dict[str, Any]:
        _, beam_system, receptors = selected
        receptor_configuration = (
            CANONICAL_HETEROGENEOUS_RECEPTOR_CONFIGURATION
            if heterogeneous_receptors
            else CANONICAL_HOMOGENEOUS_RECEPTOR_CONFIGURATION
        )
        selected_configuration = json.loads(json.dumps(configuration))
        selected_configuration["receptors"] = json.loads(
            json.dumps(receptor_configuration)
        )
        return {
            "configuration": selected_configuration,
            "antenna_layout_sha256": layout_sha256,
            "location_geodetic": dict(CANONICAL_LOCATION),
            "beam_configuration": json.loads(json.dumps(CANONICAL_BEAM_CONFIGURATION)),
            "beam_loaded_fingerprint": beam_system.state.loaded_fingerprint,
            "receptor_configuration": json.loads(json.dumps(receptor_configuration)),
            "resolved_receptors": receptors.to_snapshot(),
        }

    point_unpolarized = point_sources(2, polarized=False, gaussian=False)
    point_polarized = point_sources(2, polarized=True, gaussian=False)
    point_gaussian = point_sources(2, polarized=True, gaussian=True)
    point_scaled = point_sources(
        4096,
        polarized=False,
        gaussian=False,
        seed=20260731,
        spread=True,
    )
    healpix_scalar = healpix_model(polarized=False)
    healpix_polarized = healpix_model(polarized=True)

    def named_arrays(prefix: str, value: Any) -> list[tuple[str, Any]]:
        if isinstance(value, dict):
            mapping = cast(dict[str, object], value)
            return [
                (f"{prefix}.{name}", array)
                for name, array in mapping.items()
                if isinstance(array, np.ndarray)
            ]
        payload = value.healpix
        if payload is None:
            raise CpuEvidenceError("canonical HEALPix workload has no payload")
        arrays = [
            (f"{prefix}.maps", payload.maps),
            (f"{prefix}.frequencies", payload.frequencies),
            (f"{prefix}.pixel_indices", payload.pixel_indices),
        ]
        for name in ("q_maps", "u_maps", "v_maps"):
            array = getattr(payload, name)
            if array is not None:
                arrays.append((f"{prefix}.{name}", array))
        return arrays

    def common_identity(instrument: Any, grid: Any) -> list[tuple[str, Any]]:
        return [
            ("baseline_vectors_enu_m", instrument.baseline_vectors_enu_m),
            ("time_mjd", np.asarray(grid.to_mjd(), dtype=np.float64)),
            ("frequencies_hz", frequencies),
        ]

    workload_definitions: dict[str, tuple[Any, Any, Any, Any]] = {}

    def point_run(
        sources: SourceArrays, grid: Any, selected: Any
    ) -> Callable[[Any], Any]:
        instrument, beam_system, receptors = selected
        return lambda backend: calculate_visibility(
            instrument=instrument,
            beam_system=beam_system,
            source_arrays=sources,
            location=location,
            time_grid=grid,
            frequencies=frequencies,
            backend=backend,
            receptors=receptors,
        )

    def healpix_run(
        model: SkyModel, selected: Any, polarized: bool
    ) -> Callable[[Any], Any]:
        instrument, beam_system, receptors = selected
        return lambda backend: calculate_visibility_healpix(
            model,
            instrument=instrument,
            beam_system=beam_system,
            location=location,
            time_grid=time_grid,
            frequencies=frequencies,
            backend=backend,
            receptors=receptors,
            include_polarization=polarized,
        )

    workload_definitions["point_unpolarized_1time_2freq"] = (
        point_run(point_unpolarized, single_time_grid, homogeneous),
        homogeneous,
        common_identity(homogeneous[0], single_time_grid)
        + named_arrays("point", point_unpolarized),
        {
            "point_fixture": "unpolarized_v1",
            "coordinate_frame": "icrs",
            "spectral_model": "power_law_about_per_source_reference_frequency",
        },
    )
    workload_definitions["point_polarized_2times"] = (
        point_run(point_polarized, time_grid, homogeneous),
        homogeneous,
        common_identity(homogeneous[0], time_grid)
        + named_arrays("point", point_polarized),
        {
            "point_fixture": "polarized_v1",
            "coordinate_frame": "icrs",
            "spectral_model": "power_law_about_per_source_reference_frequency",
        },
    )
    workload_definitions["point_gaussian_morphology"] = (
        point_run(point_gaussian, time_grid, homogeneous),
        homogeneous,
        common_identity(homogeneous[0], time_grid)
        + named_arrays("point", point_gaussian),
        {
            "point_fixture": "polarized_gaussian_v1",
            "coordinate_frame": "icrs",
            "spectral_model": "power_law_about_per_source_reference_frequency",
            "morphology_units": {"major": "arcsec", "minor": "arcsec", "pa": "deg"},
        },
    )
    workload_definitions["healpix_scalar"] = (
        healpix_run(healpix_scalar, homogeneous, False),
        homogeneous,
        common_identity(homogeneous[0], time_grid)
        + named_arrays("healpix", healpix_scalar),
        {
            "healpix_fixture": "dense_nside1_scalar_v1",
            "healpix_metadata": healpix_metadata(healpix_scalar, storage="dense"),
        },
    )
    workload_definitions["healpix_polarized"] = (
        healpix_run(healpix_polarized, homogeneous, True),
        homogeneous,
        common_identity(homogeneous[0], time_grid)
        + named_arrays("healpix", healpix_polarized),
        {
            "healpix_fixture": "dense_nside1_polarized_v1",
            "healpix_metadata": healpix_metadata(healpix_polarized, storage="dense"),
        },
    )

    point_hybrid_run = point_run(point_polarized, time_grid, homogeneous)
    healpix_hybrid_run = healpix_run(healpix_polarized, homogeneous, True)

    def hybrid_run(backend: Any) -> Any:
        return backend.add(point_hybrid_run(backend), healpix_hybrid_run(backend))

    workload_definitions["hybrid_point_plus_healpix"] = (
        hybrid_run,
        homogeneous,
        common_identity(homogeneous[0], time_grid)
        + named_arrays("point", point_polarized)
        + named_arrays("healpix", healpix_polarized),
        {
            "hybrid_fixture": "point_plus_dense_healpix_v1",
            "point_coordinate_frame": "icrs",
            "healpix_metadata": healpix_metadata(healpix_polarized, storage="dense"),
        },
    )
    workload_definitions["heterogeneous_receptor_bases"] = (
        point_run(point_polarized, time_grid, heterogeneous),
        heterogeneous,
        common_identity(heterogeneous[0], time_grid)
        + named_arrays("point", point_polarized),
        {
            "point_fixture": "polarized_v1",
            "coordinate_frame": "icrs",
            "spectral_model": "power_law_about_per_source_reference_frequency",
            "receptors": "heterogeneous_v1",
        },
    )
    workload_definitions["point_scaled_4096_sources_4times"] = (
        point_run(point_scaled, scaled_time_grid, homogeneous),
        homogeneous,
        common_identity(homogeneous[0], scaled_time_grid)
        + named_arrays("point", point_scaled),
        {
            "point_fixture": "scaled_4096_seed_20260731_v1",
            "coordinate_frame": "icrs",
            "spectral_model": "power_law_about_per_source_reference_frequency",
        },
    )

    backend_objects = {
        "numpy": get_backend_runtime("numpy"),
        "jax": get_backend_runtime("jax", device="cpu"),
        "dask": get_backend_runtime("dask", mode="cpu"),
    }
    workload_rows: list[Any] = []
    for workload in CPU_WORKLOADS:
        run, selected_components, logical_inputs, fixture_detail = workload_definitions[
            workload
        ]
        manifest = {
            "schema_version": "radiosim.perf001.fixture.cpu_workload.v1",
            "workload": workload,
            "fixture": fixture_detail,
            "scientific_runtime": scientific_runtime_manifest(
                selected_components,
                heterogeneous_receptors=(workload == "heterogeneous_receptor_bases"),
            ),
        }
        reference = None
        dimensions = WORKLOAD_DIMENSIONS[workload]
        for backend_name in CPU_BACKENDS:
            backend = backend_objects[backend_name]
            timing = time_backend_call(
                lambda backend=backend, run=run: run(backend), backend=backend
            )
            if reference is None:
                reference = timing.host_result
            workload_rows.append(
                build_perf001_workload_record(
                    provenance=provenance,
                    backend=backend,
                    requested=backend_name,
                    shape=WorkloadShape(
                        workload=workload,
                        n_antennas=dimensions[0],
                        n_baselines=dimensions[1],
                        n_point_sources=dimensions[2],
                        n_healpix_pixels=dimensions[3],
                        n_times=dimensions[4],
                        n_frequencies=dimensions[5],
                        sky_representation=dimensions[6],
                        solver_workers=1,
                        loader_max_workers=0,
                    ),
                    timing=timing,
                    numpy_reference=reference,
                    fixture_manifest=manifest,
                    logical_inputs=logical_inputs,
                    notes=(
                        "Canonical retained CPU matrix; timing is observational "
                        "and carries no absolute threshold."
                    ),
                )
            )

    memory_rows = tuple(
        row
        for baselines, sources in MEMORY_FIXTURES
        for row in measure_perf001_memory_scaling_pair(
            backend_objects["numpy"],
            provenance=provenance,
            n_baselines=baselines,
            n_sources=sources,
            comparison_id=f"p-a-memory-b{baselines}-s{sources}-v1",
        )
    )

    jax_backend = backend_objects["jax"]
    synthetic_retracing = measure_perf001_synthetic_retracing_pair(
        jax_backend,
        provenance=provenance,
        source_counts=(3, 4, 5, 8, 3, 4, 5, 8),
        n_baselines=3,
        comparison_id="p-b-retracing-synthetic-wrapper-v1",
    )

    # Real point fixtures use genuine Astropy ICRS -> AltAz transforms.  Real
    # sparse HEALPix fixtures select actual above-horizon pixel centres from a
    # genuine HealpixData payload and production coordinate objects throughout.
    counts = (3, 4, 5, 8, 3, 4, 5, 8)
    point_by_count = {
        count: point_sources(count, polarized=True, gaussian=True)
        for count in sorted(set(counts))
    }
    dense_probe = HealpixData(
        maps=np.ones((1, 192), dtype=np.float64),
        nside=4,
        frequencies=frequencies[:1],
        coordinate_frame="icrs",
    )
    altaz = dense_probe.pixel_coords.transform_to(
        altaz_type(obstime=obstime, location=location)
    )
    visible_indices = dense_probe.pixel_indices[np.asarray(altaz.alt.rad) > 0]
    if visible_indices.size < max(counts):
        raise CpuEvidenceError("canonical real HEALPix fixture lacks visible pixels")
    healpix_by_count: dict[int, SkyModel] = {}
    for count in sorted(set(counts)):
        values = np.linspace(1.0, 2.0, count, dtype=np.float64)[None, :]
        payload = HealpixData(
            maps=values,
            q_maps=np.full_like(values, 0.1),
            u_maps=np.full_like(values, 0.05),
            v_maps=np.full_like(values, 0.02),
            nside=4,
            hpx_inds=np.asarray(visible_indices[:count], dtype=np.int64),
            frequencies=frequencies[:1],
            coordinate_frame="icrs",
        )
        healpix_by_count[count] = SkyModel(
            healpix=payload,
            model_name=f"perf001-real-healpix-{count}-v1",
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
            precision=PrecisionConfig.standard(),
        )

    instrument, beam_system, receptors = homogeneous

    def point_step(policy: str, index: int) -> Any:
        return _calculate_visibility(
            instrument=instrument,
            beam_system=beam_system,
            source_arrays=point_by_count[counts[index]],
            location=location,
            time_grid=single_time_grid,
            frequencies=frequencies[:1],
            backend=jax_backend,
            receptors=receptors,
            _source_bucket_policy=policy,
        )

    def healpix_step(policy: str, index: int) -> Any:
        return _calculate_visibility_healpix(
            sky_model=healpix_by_count[counts[index]],
            instrument=instrument,
            beam_system=beam_system,
            location=location,
            time_grid=single_time_grid,
            frequencies=frequencies[:1],
            backend=jax_backend,
            receptors=receptors,
            include_polarization=True,
            _source_bucket_policy=policy,
        )

    base_logical_inputs = [
        ("baseline_vectors_enu_m", instrument.baseline_vectors_enu_m),
        ("time_mjd", np.asarray(single_time_grid.to_mjd(), dtype=np.float64)),
        ("frequencies_hz", frequencies[:1]),
    ]
    point_logical_inputs = base_logical_inputs + [
        (f"sources_{count}.{name}", array)
        for count, source in sorted(point_by_count.items())
        for name, array in source.items()
        if isinstance(array, np.ndarray)
    ]
    healpix_logical_inputs = list(base_logical_inputs)
    for count, model in sorted(healpix_by_count.items()):
        payload = model.healpix
        if payload is None:
            raise CpuEvidenceError("canonical real HEALPix solver has no payload")
        healpix_logical_inputs.extend(
            (f"sources_{count}.{name}", array)
            for name, array in (
                ("maps", payload.maps),
                ("q_maps", payload.q_maps),
                ("u_maps", payload.u_maps),
                ("v_maps", payload.v_maps),
                ("hpx_inds", payload.pixel_indices),
            )
        )

    real_rows: dict[str, tuple[Any, Any, Any, Any]] = {}
    for solver, inputs, step in (
        ("point", point_logical_inputs, point_step),
        ("healpix", healpix_logical_inputs, healpix_step),
    ):
        manifest = {
            "schema_version": "radiosim.perf001.fixture.real_solver.v1",
            "solver": solver,
            "logical_source_counts": list(counts),
            "astropy_horizon_transform": True,
            "scientific_runtime": scientific_runtime_manifest(
                homogeneous,
                heterogeneous_receptors=False,
            ),
            "point_coordinate_frame": "icrs" if solver == "point" else None,
            "point_spectral_model": (
                "power_law_about_per_source_reference_frequency"
                if solver == "point"
                else None
            ),
            "healpix_metadata": (
                {
                    **healpix_metadata(
                        healpix_by_count[min(healpix_by_count)], storage="sparse"
                    ),
                    "visible_pixel_selection": {
                        "altitude_predicate": "astropy_altaz_altitude_rad_gt_zero",
                        "obstime_isot": start_time,
                        "location_geodetic": dict(CANONICAL_LOCATION),
                        "selection_order": "ascending_ring_pixel_index",
                    },
                }
                if solver == "healpix"
                else None
            ),
        }
        retracing_pair = measure_perf001_solver_retracing_pair(
            jax_backend,
            provenance=provenance,
            solver=solver,
            logical_source_counts=counts,
            fixture_manifest=manifest,
            logical_inputs=inputs,
            run_solver_step=step,
            comparison_id=f"p-b-retracing-{solver}-v1",
        )
        memory_manifest = {**manifest, "logical_source_counts": [3]}
        memory_inputs = base_logical_inputs + [
            item for item in inputs if item[0].startswith("sources_3.")
        ]
        memory_pair = measure_perf001_solver_memory_pair(
            jax_backend,
            provenance=provenance,
            solver=solver,
            logical_n_baselines=3,
            logical_source_counts=(3,),
            n_times=1,
            n_frequencies=1,
            fixture_manifest=memory_manifest,
            logical_inputs=memory_inputs,
            run_solver=lambda policy, step=step: step(policy, 0),
            comparison_id=f"p-b-solver-memory-{solver}-v1",
        )
        real_rows[solver] = (*memory_pair, *retracing_pair)

    solver_memory_rows = (
        real_rows["point"][0],
        real_rows["point"][1],
        real_rows["healpix"][0],
        real_rows["healpix"][1],
    )
    retracing_rows = (
        *synthetic_retracing,
        real_rows["point"][2],
        real_rows["point"][3],
        real_rows["healpix"][2],
        real_rows["healpix"][3],
    )

    control_manifests: tuple[dict[str, Any], ...] = (
        {
            "schema_version": CONTROL_SCHEMA,
            "operation": "get_backend_auto",
            "requested_backend": "auto",
        },
        {
            "schema_version": CONTROL_SCHEMA,
            "operation": "get_device_resources_default",
            "requested_backend": "default",
        },
        {
            "schema_version": CONTROL_SCHEMA,
            "operation": "simulator_setup_auto",
            "requested_backend": "auto",
            "fixture": "canonical_minimal_simulator_v1",
            "configuration": configuration,
            "antenna_layout_sha256": layout_sha256,
        },
    )
    backend_rows = tuple(
        measure_perf001_backend_resolution(
            provenance=provenance,
            operation=cast(str, manifest["operation"]),
            control_manifest=manifest,
            repository_root=repository_root,
            loaded_source_root=loaded_source_root,
            simulator_configuration=(
                configuration
                if manifest["operation"] == "simulator_setup_auto"
                else None
            ),
            simulator_base_dir=(
                fixture_root
                if manifest["operation"] == "simulator_setup_auto"
                else None
            ),
        )
        for manifest in control_manifests
    )

    return assemble_document(
        workload_benchmarks=workload_rows,
        memory_scaling=memory_rows,
        solver_memory=solver_memory_rows,
        retracing=retracing_rows,
        backend_resolution=backend_rows,
    )


def _worker_generate(  # pyright: ignore[reportUnusedFunction]
    *,
    repository_root: Path,
    snapshot_root: Path,
    approved_source_sha: str,
    recorded_at_utc: str,
) -> dict[str, Any]:
    """Measure, publish, and authenticate entirely from exact snapshot S."""
    root = repository_root.resolve(strict=True)
    source_root = snapshot_root.resolve(strict=True)
    try:
        captured = datetime.fromisoformat(recorded_at_utc)
    except ValueError as error:
        raise CpuEvidenceError("snapshot worker timestamp is invalid") from error
    if captured.utcoffset() != UTC.utcoffset(captured) or captured.microsecond != 0:
        raise CpuEvidenceError("snapshot worker timestamp must be whole-second UTC")
    snapshot = _authenticate_source_snapshot(
        repository_root=root,
        snapshot_root=source_root,
        approved_source_sha=approved_source_sha,
    )
    dependencies = PreflightDependencies(
        repository_root=root,
        cwd=Path.cwd(),
        environ=os.environ,
        prefix=Path(sys.prefix),
        executable=Path(sys.executable),
        run_command=_run_command,
        package_identity_check=_require_cpu_package_identity,
    )
    _ = preflight_generation(approved_source_sha, dependencies=dependencies)

    from radiosim.benchmarks import (
        benchmark_filename,
        describe_perf001_provenance,
        parse_perf001_evidence_document,
        validate_perf001_cpu_evidence_document,
        write_perf001_cpu_evidence_document,
    )

    _loaded_runtime_binding(source_root)
    _verify_loaded_snapshot_modules(snapshot)
    provenance = describe_perf001_provenance(
        repository_root=root,
        pixi_environment="default",
        recorded_at=captured,
        loaded_source_root=source_root,
    )
    if provenance.git_sha != approved_source_sha:
        raise CpuEvidenceError("captured provenance does not bind to approved S")
    with tempfile.TemporaryDirectory(prefix="radiosim-perf001-fixture-") as temporary:
        document = _measure_document(
            provenance,
            Path(temporary),
            repository_root=root,
            loaded_source_root=source_root,
        )
    _verify_source_snapshot(
        snapshot,
        repository_root=root,
        approved_source_sha=approved_source_sha,
    )
    _verify_loaded_snapshot_modules(snapshot)
    _ = preflight_generation(approved_source_sha, dependencies=dependencies)
    filename = benchmark_filename(captured)
    destination = write_perf001_cpu_evidence_document(
        document,
        filename=filename,
        repository_root=root,
        loaded_source_root=source_root,
    )
    try:
        relative = destination.relative_to(root).as_posix()
        artifact_relative, raw = _read_artifact_snapshot(
            relative,
            repository_root=root,
        )
        document_mapping, digest = _validate_artifact_bytes(
            raw,
            approved_source_sha=approved_source_sha,
            artifact_sha256=hashlib.sha256(raw).hexdigest(),
            filename=artifact_relative.name,
        )
        if not document_mapping:
            raise CpuEvidenceError("published artifact document is empty")
        production_document = parse_perf001_evidence_document(raw)
        validate_perf001_cpu_evidence_document(production_document)
        _verify_source_snapshot(
            snapshot,
            repository_root=root,
            approved_source_sha=approved_source_sha,
        )
        _verify_loaded_snapshot_modules(snapshot)
    except (
        OSError,
        UnicodeDecodeError,
        CpuEvidenceError,
        ValueError,
    ) as error:
        raise CpuEvidenceError(
            "published artifact is retained but post-write authentication failed"
        ) from error
    return {
        "artifact_path": relative,
        "artifact_sha256": digest,
        "generating_source_sha": approved_source_sha,
        "row_count": 45,
        "passed": True,
    }


def _run_snapshot_worker(
    *,
    dependencies: PreflightDependencies,
    snapshot: SourceSnapshot,
    approved_source_sha: str,
    captured: datetime,
) -> dict[str, Any]:
    environment = dict(dependencies.environ)
    _ = environment.pop("PYTHONPATH", None)
    _ = environment.pop("PYTHONHOME", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            [
                str(dependencies.executable),
                "-I",
                "-S",
                "-B",
                "-c",
                _SNAPSHOT_WORKER_BOOTSTRAP,
                str(snapshot.root),
                str(dependencies.repository_root),
                approved_source_sha,
                captured.isoformat(timespec="seconds"),
            ],
            cwd=dependencies.repository_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CpuEvidenceError("authenticated source worker could not run") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CpuEvidenceError(
            f"authenticated source worker exited {completed.returncode}: {detail}"
        )
    output_lines = [line for line in completed.stdout.splitlines() if line]
    if len(output_lines) != 1 or not output_lines[0].startswith(WORKER_RESULT_PREFIX):
        raise CpuEvidenceError(
            "authenticated source worker emitted a noncanonical result"
        )
    try:
        decoded = json.loads(
            output_lines[0].removeprefix(WORKER_RESULT_PREFIX),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, CpuEvidenceError) as error:
        raise CpuEvidenceError(
            "authenticated source worker emitted invalid JSON"
        ) from error
    expected_fields = {
        "artifact_path",
        "artifact_sha256",
        "generating_source_sha",
        "row_count",
        "passed",
    }
    if type(decoded) is not dict:
        raise CpuEvidenceError("authenticated source worker result is incomplete")
    summary = cast(dict[str, Any], decoded)
    if set(summary) != expected_fields:
        raise CpuEvidenceError("authenticated source worker result is incomplete")
    artifact_path = _string(summary["artifact_path"], location="worker.artifact_path")
    expected_filename = (
        f"{captured.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{platform.system().lower()}-{platform.machine().lower()}.json"
    )
    expected_artifact_path = (REFERENCE_DIRECTORY / expected_filename).as_posix()
    if (
        artifact_path != expected_artifact_path
        or Path(artifact_path).parent != REFERENCE_DIRECTORY
        or _FILENAME.fullmatch(Path(artifact_path).name) is None
        or summary["generating_source_sha"] != approved_source_sha
        or _digest(summary["artifact_sha256"], location="worker.artifact_sha256")
        != summary["artifact_sha256"]
        or summary["row_count"] != 45
        or type(summary["row_count"]) is not int
        or summary["passed"] is not True
    ):
        raise CpuEvidenceError("authenticated source worker result failed validation")
    return summary


def generate(
    approved_source_sha: str,
    *,
    dependencies: PreflightDependencies | None = None,
    moment: datetime | None = None,
) -> dict[str, Any]:
    """Export exact S and delegate every measurement/write to an isolated worker."""
    selected = dependencies or _default_dependencies()
    _ = preflight_generation(approved_source_sha, dependencies=selected)
    captured = (moment or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    with tempfile.TemporaryDirectory(prefix="radiosim-perf001-source-") as temporary:
        workspace = Path(temporary)
        snapshot: SourceSnapshot | None = None
        try:
            snapshot = _export_source_snapshot(
                repository_root=selected.repository_root,
                approved_source_sha=approved_source_sha,
                workspace=workspace,
            )
            return _run_snapshot_worker(
                dependencies=selected,
                snapshot=snapshot,
                approved_source_sha=approved_source_sha,
                captured=captured,
            )
        finally:
            if snapshot is not None:
                _unseal_source_snapshot(snapshot.root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    _ = generate_parser.add_argument("--approved-source-sha", required=True)
    validate_parser = subparsers.add_parser("validate")
    _ = validate_parser.add_argument("--approved-source-sha", required=True)
    _ = validate_parser.add_argument("--artifact-sha256", required=True)
    _ = validate_parser.add_argument("--input", required=True)
    accepted_parser = subparsers.add_parser("verify-accepted")
    _ = accepted_parser.add_argument("--acceptance-commit", required=True)
    _ = accepted_parser.add_argument("--descendant", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "generate":
            summary = generate(arguments.approved_source_sha)
        elif arguments.command == "validate":
            document, raw, digest = load_and_validate_artifact(
                arguments.input,
                approved_source_sha=arguments.approved_source_sha,
                artifact_sha256=arguments.artifact_sha256,
                repository_root=REPOSITORY_ROOT,
            )
            _authenticate_cli_evidence_edge(
                input_path=arguments.input,
                approved_source_sha=arguments.approved_source_sha,
                raw=raw,
                repository_root=REPOSITORY_ROOT,
            )
            summary = {
                "artifact_path": arguments.input,
                "artifact_sha256": digest,
                "generating_source_sha": arguments.approved_source_sha,
                "row_count": sum(len(document[name]) for name in DOCUMENT_FIELDS[1:]),
                "passed": True,
            }
        else:
            summary = verify_accepted_cpu_certificate(
                acceptance_commit=arguments.acceptance_commit,
                descendant=arguments.descendant,
                repository_root=REPOSITORY_ROOT,
            )
    except CpuEvidenceError as error:
        print(f"PERF-001 CPU evidence error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
