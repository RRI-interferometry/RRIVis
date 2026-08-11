"""The reproducible benchmark harness of ``Tier6HybridRuntimePlan.md`` Section 22.

The harness lives in ``src/`` rather than in the test tree for one reason: it has
to be trustworthy even when the benchmarks themselves are not running. Section
22.1 puts it here so the fast suite can unit-test the record schema and the
timing discipline, while the actual measurements stay behind the
``performance``/``slow`` markers that CI never selects.

Timing discipline (Section 22.2), implemented literally:

- the **first** call of any backend is reported as ``setup_seconds`` and never
  enters the steady-state statistics, because JAX compiles on first call;
- ``compile_seconds`` is ``max(0, setup - steady-state median)``, reported as its
  own field. For a backend whose ``compilation_used`` is ``False`` this is
  first-call excess (import, cache warm-up, first-touch page faults), not
  compilation -- the field is only readable next to ``compilation_used``, which
  is why both are mandatory;
- steady state is the median of at least five iterations, with the min and max
  recorded, because one sample is not a measurement;
- every timed call ends with ``backend.synchronize(result)`` before the clock
  stops. Without it a JAX measurement times dispatch, not work (defect D13);
- host transfer is timed around ``backend.to_numpy`` alone, which is the single
  transfer point;
- peak host memory is a ``tracemalloc`` peak taken in a **separate**, untimed
  iteration, plus ``backend.memory_info()``. Tracing allocations perturbs timings
  by a large factor, so measuring both in one pass would corrupt the numbers this
  harness exists to produce;
- every record states its correctness delta against the NumPy reference for the
  identical workload.

No function here asserts a threshold, and no benchmark number is ever compared
against a hard-coded time. The performance tests assert *completeness and
correctness*; speed is reported, never gated.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import secrets
import stat
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from radiosim.benchmarks.record import (
    BENCHMARK_SCHEMA_VERSION,
    MEMORY_SCALING_SCHEMA_VERSION,
    PERF001_MEMORY_SCALING_SCHEMA_VERSION,
    PERF001_PROVENANCE_SCHEMA_VERSION,
    PERF001_RETRACING_SCHEMA_VERSION,
    PERF001_SOLVER_MEMORY_SCHEMA_VERSION,
    PERF001_TARGET_KERNEL_PAIRS,
    RETRACING_SCHEMA_VERSION,
    BenchmarkRecord,
    BenchmarkRecordError,
    ContractionSignatureObservation,
    MeasurementContext,
    MemoryScalingRecord,
    MemoryScalingRecordV2,
    Perf001EvidenceDocument,
    Perf001Provenance,
    RetracingRecord,
    RetracingRecordV2,
    SolverMemoryRecord,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard only
    from radiosim.backends.base import ArrayBackend
    from radiosim.core.precision import PrecisionConfig

__all__ = [
    "DEFAULT_STEADY_STATE_ITERATIONS",
    "PERF001_REFERENCE_SHA256",
    "BackendFacts",
    "BenchmarkBackendSelection",
    "Deviation",
    "EnvironmentFacts",
    "Perf001ReferenceAuthentication",
    "TimingMeasurement",
    "WorkloadShape",
    "authenticate_perf001_references",
    "benchmark_filename",
    "benchmark_backend_selection",
    "benchmark_output_directory",
    "build_record",
    "compare_to_reference",
    "describe_backend",
    "describe_environment",
    "describe_perf001_provenance",
    "measure_perf001_memory_scaling_pair",
    "measure_perf001_solver_memory_pair",
    "measure_perf001_solver_retracing_pair",
    "measure_perf001_synthetic_retracing_pair",
    "measure_kernel_memory_scaling",
    "measure_retracing",
    "perf001_input_identity_sha256",
    "perf001_reference_output_directory",
    "time_backend_call",
    "verify_perf001_provenance_binding",
    "verify_required_benchmark_accelerator",
    "write_perf001_evidence_document",
]

#: Section 22.2: "the median of at least 5 iterations".
DEFAULT_STEADY_STATE_ITERATIONS = 5

#: Section 13.5 tolerance for float64 accumulation.
FLOAT64_RTOL = 1e-12
FLOAT64_ATOL_SCALE = 1e-12

#: Exact digests of every committed namespaced PERF-001 reference.  The CPU
#: evidence-successor commit adds its path and digest here in the same commit as
#: the JSON file.  Keeping the mapping empty before that commit is intentional:
#: authentication compares it with ``git ls-files`` and therefore still fails
#: if an unlisted record appears.
PERF001_REFERENCE_SHA256: dict[str, str] = {}

_PERF001_REFERENCE_RELATIVE_DIRECTORY = Path("output/benchmarks/reference/perf001")
_LOWER_HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_PIXI_ENVIRONMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_PERF001_REFERENCE_FILENAME = re.compile(
    r"(?P<stamp>[0-9]{8}T[0-9]{6}Z)-"
    r"(?P<host>[a-z0-9]+(?:[._-][a-z0-9]+)*)\.json\Z"
)

_DEFAULT_BENCHMARK_BACKENDS = ("numpy", "jax", "dask")
_BENCHMARK_BACKEND_REQUESTS = frozenset({"numpy", "jax", "dask", "gpu"})
_BENCHMARK_ACCELERATORS = frozenset({"gpu"})


@dataclass(frozen=True, slots=True)
class BenchmarkBackendSelection:
    """Strict backend matrix selected for one benchmark invocation."""

    backend_requests: tuple[str, ...]
    required_accelerator: str | None


def benchmark_backend_selection(
    environment: Mapping[str, str] | None = None,
) -> BenchmarkBackendSelection:
    """Parse the two opt-in benchmark environment controls fail-closed.

    With neither variable present, this returns the historical CPU matrix
    ``("numpy", "jax", "dask")`` exactly.  Accelerator readiness uses
    ``RADIOSIM_BENCHMARK_BACKENDS=numpy,gpu`` and
    ``RADIOSIM_REQUIRE_ACCELERATOR=gpu``; NumPy must remain first because it is
    the correctness reference for every workload object.
    """
    values = os.environ if environment is None else environment
    raw_backends = values.get("RADIOSIM_BENCHMARK_BACKENDS")
    if raw_backends is None:
        backends = _DEFAULT_BENCHMARK_BACKENDS
    else:
        if not raw_backends or raw_backends.strip() != raw_backends:
            raise BenchmarkRecordError(
                "RADIOSIM_BENCHMARK_BACKENDS must be a non-empty canonical "
                "comma-separated list"
            )
        backends = tuple(raw_backends.split(","))
        if any(not item or item.strip() != item for item in backends):
            raise BenchmarkRecordError(
                "RADIOSIM_BENCHMARK_BACKENDS contains an empty or padded token"
            )
    unknown = sorted(set(backends) - _BENCHMARK_BACKEND_REQUESTS)
    if unknown:
        raise BenchmarkRecordError(
            "RADIOSIM_BENCHMARK_BACKENDS contains unsupported request(s): "
            + ", ".join(unknown)
        )
    if len(set(backends)) != len(backends):
        raise BenchmarkRecordError(
            "RADIOSIM_BENCHMARK_BACKENDS must not contain duplicate requests"
        )
    if not backends or backends[0] != "numpy":
        raise BenchmarkRecordError(
            "RADIOSIM_BENCHMARK_BACKENDS must keep numpy first as the reference"
        )

    required = values.get("RADIOSIM_REQUIRE_ACCELERATOR")
    if required == "":
        raise BenchmarkRecordError(
            "RADIOSIM_REQUIRE_ACCELERATOR may be omitted but not empty"
        )
    if required is not None and required not in _BENCHMARK_ACCELERATORS:
        raise BenchmarkRecordError(
            "PERF-001 RADIOSIM_REQUIRE_ACCELERATOR currently supports only 'gpu'"
        )
    if required is not None and required not in backends:
        raise BenchmarkRecordError(
            f"required accelerator {required!r} is absent from "
            "RADIOSIM_BENCHMARK_BACKENDS"
        )
    return BenchmarkBackendSelection(
        backend_requests=backends,
        required_accelerator=required,
    )


def verify_required_benchmark_accelerator(
    backend: ArrayBackend,
    *,
    requested: str,
    required_accelerator: str | None,
) -> None:
    """Reject a required accelerator request that resolved to another device."""
    if required_accelerator is None or requested != required_accelerator:
        return
    actual = str(backend.device_kind).lower()
    if actual != required_accelerator:
        raise BenchmarkRecordError(
            f"benchmark request {requested!r} required a real "
            f"{required_accelerator}, but resolved device_kind={actual!r}"
        )


# =========================================================================
# Environment and backend description
# =========================================================================


@dataclass(frozen=True, slots=True)
class EnvironmentFacts:
    """What machine and what checkout produced a record."""

    recorded_at_utc: str
    radiosim_version: str
    git_sha: str
    platform: str
    cpu_model: str
    cpu_count_logical: int


def _git_sha(repository_root: Path | None = None) -> str:
    """Return the current commit, suffixed ``-dirty`` when the tree is modified.

    A benchmark taken against an unrecorded working tree is not reproducible, so
    the dirty state travels with the number rather than being dropped.
    """
    root = repository_root or Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not sha:
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _cpu_model() -> str:
    """Return a human-readable CPU description, never an empty string."""
    if platform.system() == "Darwin":
        try:
            brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout.strip()
            if brand:
                return brand
        except (OSError, subprocess.SubprocessError):
            pass
    elif platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


def describe_environment() -> EnvironmentFacts:
    """Capture the hardware and checkout facts every record must carry."""
    from radiosim.__about__ import __version__

    return EnvironmentFacts(
        recorded_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        radiosim_version=__version__,
        git_sha=_git_sha(),
        platform=platform.platform(),
        cpu_model=_cpu_model(),
        cpu_count_logical=os.cpu_count() or 1,
    )


# =========================================================================
# PERF-001 source, fixture, and retained-reference identity
# =========================================================================


def _canonical_json_bytes(value: object, *, field_name: str) -> bytes:
    """Return strict canonical JSON, rejecting values JSON would coerce."""

    def require_json(item: object, location: str) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise BenchmarkRecordError(
                    f"PERF-001 {location} must contain only finite JSON numbers"
                )
            return
        if type(item) is dict:
            mapping = cast(dict[object, object], item)
            for key, nested in mapping.items():
                if type(key) is not str or not key:
                    raise BenchmarkRecordError(
                        f"PERF-001 {location} keys must be non-empty JSON strings"
                    )
                require_json(nested, f"{location}.{key}")
            return
        if type(item) in (list, tuple):
            sequence = cast(list[object] | tuple[object, ...], item)
            for index, nested in enumerate(sequence):
                require_json(nested, f"{location}[{index}]")
            return
        raise BenchmarkRecordError(
            f"PERF-001 {location} contains non-JSON value {type(item).__name__}"
        )

    require_json(value, field_name)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _update_length_framed(hasher: Any, payload: bytes) -> None:
    """Hash one unambiguous byte string using an unsigned 64-bit length."""
    hasher.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    hasher.update(payload)


def perf001_input_identity_sha256(
    fixture_manifest: Mapping[str, object],
    logical_inputs: Sequence[tuple[str, Any]],
) -> str:
    """Authenticate one logical scientific fixture deterministically.

    The digest is domain-separated and length-framed.  It covers canonical JSON
    for the versioned fixture manifest, followed in caller-supplied order by
    each input's name, C-order shape, NumPy dtype string, and C-contiguous raw
    bytes.  Memory layout alone is not scientific identity, so equivalent
    Fortran- and C-contiguous views produce the same digest; changing array
    order, shape, dtype, or any value does not.
    """
    manifest = dict(fixture_manifest)
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not str or not schema_version:
        raise BenchmarkRecordError(
            "PERF-001 fixture_manifest.schema_version must be a non-empty string"
        )
    if not logical_inputs:
        raise BenchmarkRecordError(
            "PERF-001 logical_inputs must contain at least one scientific input"
        )

    digest = sha256()
    _update_length_framed(digest, b"radiosim.perf001.input_identity.v1")
    _update_length_framed(
        digest,
        _canonical_json_bytes(manifest, field_name="fixture_manifest"),
    )
    seen_names: set[str] = set()
    for index, item in enumerate(logical_inputs):
        if type(item) is not tuple or len(item) != 2:
            raise BenchmarkRecordError(
                "PERF-001 logical_inputs entries must be exact (name, array) tuples"
            )
        name, values = item
        if type(name) is not str or not name:
            raise BenchmarkRecordError(
                f"PERF-001 logical_inputs[{index}] name must be non-empty"
            )
        if name in seen_names:
            raise BenchmarkRecordError(
                "PERF-001 logical input names must be unique and ordered"
            )
        seen_names.add(name)
        array = np.asarray(values)
        if array.dtype.hasobject:
            raise BenchmarkRecordError(
                f"PERF-001 logical input {name!r} has an object dtype with "
                "non-canonical pointer bytes"
            )
        if array.dtype.fields is not None:
            raise BenchmarkRecordError(
                f"PERF-001 logical input {name!r} has a structured dtype; "
                "name fields separately"
            )
        contiguous = np.ascontiguousarray(array)
        metadata = {
            "name": name,
            "shape": list(contiguous.shape),
            "dtype": contiguous.dtype.str,
        }
        _update_length_framed(
            digest,
            _canonical_json_bytes(metadata, field_name=f"logical_inputs[{index}]"),
        )
        _update_length_framed(digest, contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _clean_git_sha(repository_root: Path) -> str:
    """Return a clean exact HEAD SHA or fail closed."""
    try:
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise BenchmarkRecordError(
            "PERF-001 provenance requires a readable live Git checkout"
        ) from error
    if Path(top_level).resolve() != repository_root.resolve():
        raise BenchmarkRecordError(
            "PERF-001 repository_root must be the exact Git top level"
        )
    if _LOWER_HEX_40.fullmatch(git_sha) is None:
        raise BenchmarkRecordError(
            "PERF-001 provenance requires an exact lowercase 40-hex Git SHA"
        )
    if dirty:
        raise BenchmarkRecordError(
            "PERF-001 evidence generation requires a clean working tree"
        )
    return git_sha


def _installed_distribution_version(distribution: str) -> str:
    """Read package metadata without importing the distribution."""
    try:
        return distribution_version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _require_loaded_runtime_binding(
    repository_root: Path,
    environment_prefix: Path,
) -> None:
    """Bind loaded source and executable to the selected checkout and Pixi env."""
    import radiosim

    expected_package = repository_root / "src/radiosim/__init__.py"
    expected_harness = repository_root / "src/radiosim/benchmarks/harness.py"
    loaded_files = (
        ("package", getattr(radiosim, "__file__", None), expected_package),
        ("harness", __file__, expected_harness),
    )
    for label, raw_loaded, expected in loaded_files:
        if not raw_loaded:
            raise BenchmarkRecordError(
                f"PERF-001 loaded RadioSim {label} has no source file"
            )
        try:
            loaded = Path(raw_loaded).resolve(strict=True)
        except OSError as error:
            raise BenchmarkRecordError(
                f"PERF-001 loaded RadioSim {label} source is unreadable"
            ) from error
        # ``expected`` deliberately remains lexical. If a source path inside
        # the checkout is itself a symlink, ``loaded`` resolves away and fails.
        if loaded != expected or not loaded.is_file():
            raise BenchmarkRecordError(
                f"PERF-001 loaded RadioSim {label} does not come from repository_root"
            )

    try:
        executable = Path(sys.executable).resolve(strict=True)
    except OSError as error:
        raise BenchmarkRecordError(
            "PERF-001 active Python executable is unreadable"
        ) from error
    if (
        not executable.is_file()
        or not os.access(executable, os.X_OK)
        or not executable.is_relative_to(environment_prefix)
    ):
        raise BenchmarkRecordError(
            "PERF-001 active Python executable is not inside the selected "
            "Pixi environment under repository_root"
        )


def _require_pixi_environment(repository_root: Path, expected: str | None) -> str:
    """Bind generation to the actual named Pixi environment and project."""
    actual = os.environ.get("PIXI_ENVIRONMENT_NAME")
    if not actual or _PIXI_ENVIRONMENT_NAME.fullmatch(actual) is None:
        raise BenchmarkRecordError(
            "PERF-001 evidence generation must run inside a canonical named "
            "Pixi environment"
        )
    if expected is not None and actual != expected:
        raise BenchmarkRecordError(
            f"PERF-001 expected Pixi environment {expected!r}, got {actual!r}"
        )
    project_root = os.environ.get("PIXI_PROJECT_ROOT")
    if (
        project_root is None
        or Path(project_root).resolve() != repository_root.resolve()
    ):
        raise BenchmarkRecordError(
            "PERF-001 PIXI_PROJECT_ROOT does not identify repository_root"
        )
    if not (repository_root / "pixi.toml").is_file():
        raise BenchmarkRecordError(
            "PERF-001 repository_root does not contain the Pixi manifest"
        )
    declared_prefix = os.environ.get("CONDA_PREFIX")
    if not declared_prefix:
        raise BenchmarkRecordError(
            "PERF-001 evidence generation requires Pixi's CONDA_PREFIX"
        )
    expected_prefix = repository_root / ".pixi" / "envs" / actual
    try:
        interpreter_prefix = Path(sys.prefix).resolve(strict=True)
        declared_environment_prefix = Path(declared_prefix).resolve(strict=True)
    except OSError as error:
        raise BenchmarkRecordError(
            "PERF-001 active interpreter prefix is unreadable"
        ) from error
    # ``expected_prefix`` remains lexical so a symlinked environment component
    # cannot escape the checkout while still comparing equal after resolution.
    if (
        declared_environment_prefix != interpreter_prefix
        or interpreter_prefix != expected_prefix
        or not interpreter_prefix.is_dir()
    ):
        raise BenchmarkRecordError(
            "PERF-001 active interpreter prefix does not match the declared "
            f"Pixi environment {actual!r} under repository_root"
        )
    _require_loaded_runtime_binding(repository_root, expected_prefix)
    return actual


def describe_perf001_provenance(
    *,
    repository_root: Path | None = None,
    pixi_environment: str | None = None,
    recorded_at: datetime | None = None,
) -> Perf001Provenance:
    """Capture strict clean-source provenance for a PERF-001 measurement."""
    from radiosim.__about__ import __version__

    root = (repository_root or Path(__file__).resolve().parents[3]).resolve()
    git_sha = _clean_git_sha(root)
    environment = _require_pixi_environment(root, pixi_environment)
    lock_path = root / "pixi.lock"
    if not lock_path.is_file():
        raise BenchmarkRecordError("PERF-001 provenance requires repository pixi.lock")
    moment = recorded_at or datetime.now(UTC)
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise BenchmarkRecordError("PERF-001 recorded_at must be timezone-aware")
    recorded_at_utc = moment.astimezone(UTC).isoformat(timespec="seconds")
    return Perf001Provenance.create(
        schema_version=PERF001_PROVENANCE_SCHEMA_VERSION,
        recorded_at_utc=recorded_at_utc,
        radiosim_version=__version__,
        git_sha=git_sha,
        working_tree_clean=True,
        platform=platform.platform(),
        machine=platform.machine() or "unreported-machine",
        cpu_model=_cpu_model(),
        cpu_count_logical=os.cpu_count() or 1,
        python_version=platform.python_version(),
        numpy_version=_installed_distribution_version("numpy"),
        jax_version=_installed_distribution_version("jax"),
        jaxlib_version=_installed_distribution_version("jaxlib"),
        dask_version=_installed_distribution_version("dask"),
        pixi_environment=environment,
        pixi_lock_sha256=sha256(lock_path.read_bytes()).hexdigest(),
    )


def verify_perf001_provenance_binding(
    provenance: Perf001Provenance,
    *,
    repository_root: Path | None = None,
) -> None:
    """Fail unless ``provenance`` still names the live clean source and lock."""
    if type(provenance) is not Perf001Provenance:
        raise TypeError("provenance must be an exact Perf001Provenance")
    root = (repository_root or Path(__file__).resolve().parents[3]).resolve()
    live_git_sha = _clean_git_sha(root)
    if provenance.git_sha != live_git_sha:
        raise BenchmarkRecordError(
            "PERF-001 provenance git_sha does not match live clean HEAD"
        )
    live_environment = _require_pixi_environment(
        root,
        provenance.pixi_environment,
    )
    if provenance.pixi_environment != live_environment:  # pragma: no cover
        raise BenchmarkRecordError("PERF-001 Pixi environment binding changed")
    lock_path = root / "pixi.lock"
    if not lock_path.is_file():
        raise BenchmarkRecordError("PERF-001 provenance requires repository pixi.lock")
    live_lock_digest = sha256(lock_path.read_bytes()).hexdigest()
    if provenance.pixi_lock_sha256 != live_lock_digest:
        raise BenchmarkRecordError(
            "PERF-001 provenance pixi_lock_sha256 does not match live pixi.lock"
        )


def perf001_reference_output_directory(
    repository_root: Path | None = None,
) -> Path:
    """Return the only permitted retained PERF-001 evidence directory."""
    root = (repository_root or Path(__file__).resolve().parents[3]).resolve()
    return root / _PERF001_REFERENCE_RELATIVE_DIRECTORY


def _open_perf001_reference_directory(
    repository_root: Path,
    *,
    create: bool,
) -> int:
    """Open the exact evidence directory without following any component link."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(repository_root, flags)
    except OSError as error:
        raise BenchmarkRecordError(
            "PERF-001 repository_root must be a readable non-symlink directory"
        ) from error
    try:
        for component in _PERF001_REFERENCE_RELATIVE_DIRECTORY.parts:
            if create:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise BenchmarkRecordError(
                    "PERF-001 evidence destination contains a symlink, is "
                    f"missing, or is not a directory: {component!r}"
                ) from error
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):  # pragma: no cover
            raise BenchmarkRecordError(
                "PERF-001 evidence destination is not a directory"
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _canonical_perf001_reference_filename(filename: str) -> str:
    """Return one exact ``<UTC>-<host>.json`` reference filename."""
    match = _PERF001_REFERENCE_FILENAME.fullmatch(filename)
    if match is None:
        raise BenchmarkRecordError(
            "PERF-001 reference filename must be canonical "
            "YYYYMMDDTHHMMSSZ-<lowercase-host>.json"
        )
    try:
        datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%SZ")
    except ValueError as error:
        raise BenchmarkRecordError(
            "PERF-001 reference filename must contain a valid canonical UTC timestamp"
        ) from error
    return filename


def _perf001_document_provenance(
    document: Perf001EvidenceDocument,
) -> Perf001Provenance:
    all_rows = (
        document.workload_benchmarks
        + document.memory_scaling
        + document.solver_memory
        + document.retracing
        + document.backend_resolution
    )
    # Strict document validation requires all five collections to be nonempty,
    # and also proves every row has the same provenance.
    return all_rows[0].provenance


def write_perf001_evidence_document(
    document: Perf001EvidenceDocument,
    *,
    filename: str,
    repository_root: Path | None = None,
    directory: Path | None = None,
) -> Path:
    """Write one strict PERF-001 document, failing closed on stale source state.

    The destination is fixed to the namespaced retained-reference directory,
    the filename is derived from the document's UTC timestamp and current host
    tag, and exclusive creation forbids overwriting evidence.
    """
    if type(document) is not Perf001EvidenceDocument:
        raise TypeError("document must be an exact Perf001EvidenceDocument")
    if type(filename) is not str or not filename:
        raise ValueError("filename must be a non-empty string")
    root = (repository_root or Path(__file__).resolve().parents[3]).resolve()
    expected_directory = perf001_reference_output_directory(root)
    destination_directory = (
        Path(directory) if directory is not None else expected_directory
    )
    lexical_destination = Path(os.path.abspath(os.fspath(destination_directory)))
    if lexical_destination != expected_directory:
        raise BenchmarkRecordError(
            f"PERF-001 evidence output directory must be exactly {expected_directory}"
        )
    provenance = _perf001_document_provenance(document)
    verify_perf001_provenance_binding(provenance, repository_root=root)
    recorded_at = datetime.fromisoformat(
        provenance.recorded_at_utc.replace("Z", "+00:00")
    )
    expected_filename = benchmark_filename(recorded_at)
    if filename != expected_filename:
        raise BenchmarkRecordError(
            "PERF-001 evidence filename must match its UTC provenance and host: "
            f"expected {expected_filename!r}"
        )
    _canonical_perf001_reference_filename(filename)
    serialized = (
        json.dumps(
            document.to_json_safe(),
            allow_nan=False,
            indent=2,
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")
    destination = expected_directory / filename
    directory_descriptor = _open_perf001_reference_directory(root, create=True)
    temporary_name: str | None = None
    try:
        for _ in range(100):
            candidate = f".{filename}.{secrets.token_hex(12)}.tmp"
            try:
                temporary_descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory_descriptor,
                )
            except FileExistsError:  # pragma: no cover - random collision
                continue
            temporary_name = candidate
            break
        else:  # pragma: no cover - cryptographically implausible
            raise BenchmarkRecordError(
                "PERF-001 could not allocate an exclusive temporary file"
            )

        try:
            stream = os.fdopen(temporary_descriptor, "wb")
            with stream:
                _ = stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise BenchmarkRecordError(
                f"PERF-001 evidence could not be published atomically: {destination}"
            ) from error

        try:
            # A same-directory hard link publishes already-complete bytes in
            # one atomic namespace operation and cannot clobber an existing
            # retained artifact.
            os.link(
                temporary_name,
                filename,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise BenchmarkRecordError(
                f"PERF-001 evidence target already exists: {destination}"
            ) from error
        except OSError as error:
            raise BenchmarkRecordError(
                f"PERF-001 evidence could not be published atomically: {destination}"
            ) from error

        # Remove the second temporary name before flushing the directory. The
        # directory fsync therefore makes both publication and cleanup durable.
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        temporary_name = None
        try:
            os.fsync(directory_descriptor)
        except OSError as error:
            # The final name already refers to a complete file. Retaining it is
            # safer than an unverifiable rollback after a durability failure;
            # no-overwrite semantics force a human to inspect before retrying.
            raise BenchmarkRecordError(
                "PERF-001 evidence publication reached the complete final "
                f"target but directory durability failed: {destination}"
            ) from error
    except BenchmarkRecordError:
        raise
    except OSError as error:
        raise BenchmarkRecordError(
            f"PERF-001 evidence could not be published atomically: {destination}"
        ) from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
        os.close(directory_descriptor)
    return destination


@dataclass(frozen=True, slots=True)
class Perf001ReferenceAuthentication:
    """One exact tracked PERF-001 path and its authenticated byte digest."""

    relative_path: str
    path: Path
    sha256: str


def _tracked_perf001_reference_paths(repository_root: Path) -> tuple[str, ...]:
    """Enumerate namespaced PERF-001 JSON tracked by Git, in exact path order."""
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--",
                _PERF001_REFERENCE_RELATIVE_DIRECTORY.as_posix(),
            ],
            cwd=repository_root,
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BenchmarkRecordError(
            "PERF-001 reference authentication requires git ls-files"
        ) from error
    paths: list[str] = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        try:
            relative = raw_path.decode("utf-8")
        except UnicodeDecodeError as error:
            raise BenchmarkRecordError(
                "PERF-001 tracked reference paths must be UTF-8"
            ) from error
        path = Path(relative)
        if (
            path.parent != _PERF001_REFERENCE_RELATIVE_DIRECTORY
            or path.suffix != ".json"
        ):
            raise BenchmarkRecordError(
                "PERF-001 tracked references must be direct lowercase-.json "
                "children of output/benchmarks/reference/perf001: "
                f"{relative}"
            )
        _canonical_perf001_reference_filename(path.name)
        paths.append(path.as_posix())
    return tuple(sorted(paths))


def authenticate_perf001_references(
    *,
    repository_root: Path | None = None,
    expected_sha256: Mapping[str, str] | None = None,
) -> tuple[Perf001ReferenceAuthentication, ...]:
    """Authenticate every tracked PERF-001 record by exact path and SHA-256.

    The tracked set and expected set must be identical.  This deliberately has
    no "first JSON" route: adding, removing, renaming, or changing any retained
    record fails acceptance until its exact digest manifest is updated.
    """
    root = (repository_root or Path(__file__).resolve().parents[3]).resolve()
    expected = dict(
        PERF001_REFERENCE_SHA256 if expected_sha256 is None else expected_sha256
    )
    tracked = _tracked_perf001_reference_paths(root)
    tracked_set = set(tracked)
    expected_set = set(expected)
    unlisted = sorted(tracked_set - expected_set)
    if unlisted:
        raise BenchmarkRecordError(
            "PERF-001 reference digest manifest has unlisted tracked file(s): "
            + ", ".join(unlisted)
        )
    untracked = sorted(expected_set - tracked_set)
    if untracked:
        raise BenchmarkRecordError(
            "PERF-001 reference digest manifest names expected but untracked "
            "file(s): " + ", ".join(untracked)
        )

    if not tracked:
        return ()

    directory_descriptor = _open_perf001_reference_directory(root, create=False)
    authenticated: list[Perf001ReferenceAuthentication] = []
    try:
        for relative in tracked:
            expected_digest = expected[relative]
            if _LOWER_HEX_64.fullmatch(expected_digest) is None:
                raise BenchmarkRecordError(
                    f"PERF-001 reference {relative} has an invalid expected SHA-256"
                )
            path = root / relative
            try:
                descriptor = os.open(
                    path.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_descriptor,
                )
            except OSError as error:
                raise BenchmarkRecordError(
                    "PERF-001 tracked reference must be a regular non-symlink "
                    f"file: {relative}"
                ) from error
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise BenchmarkRecordError(
                        "PERF-001 tracked reference must be a regular "
                        f"non-symlink file: {relative}"
                    )
                with os.fdopen(descriptor, "rb") as stream:
                    descriptor = -1
                    actual_digest = sha256(stream.read()).hexdigest()
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            if actual_digest != expected_digest:
                raise BenchmarkRecordError(
                    f"PERF-001 reference digest mismatch for {relative}: "
                    f"expected {expected_digest}, got {actual_digest}"
                )
            authenticated.append(
                Perf001ReferenceAuthentication(
                    relative_path=relative,
                    path=path,
                    sha256=actual_digest,
                )
            )
    finally:
        os.close(directory_descriptor)
    return tuple(authenticated)


@dataclass(frozen=True, slots=True)
class BackendFacts:
    """What actually executed, as opposed to what was asked for."""

    backend_requested: str
    backend_actual: str
    backend_version: str
    device_kind: str
    compilation_used: bool
    accelerator: str
    accelerator_driver: str | None
    memory_info: dict[str, object]
    unmeasured: tuple[str, ...]


def _backend_version(backend: ArrayBackend) -> str:
    """Return the version of the library the backend actually calls."""
    name = backend.name.lower()
    try:
        if name.startswith("jax"):
            import jax

            return str(jax.__version__)
        if name.startswith("dask"):
            import dask

            return str(dask.__version__)
    except Exception:  # pragma: no cover - a backend that imported cannot vanish
        return "unknown"
    return str(np.__version__)


def describe_backend(backend: ArrayBackend, *, requested: str) -> BackendFacts:
    """Describe ``backend`` for a record, including what it did **not** measure.

    ``accelerator`` is derived from the backend's own ``device_kind``, never from
    the requested name: asking for JAX on a CPU-only build must not produce a
    record that reads like an accelerator run. When no accelerator is present,
    ``unmeasured`` says so explicitly.
    """
    device_kind = str(backend.device_kind).lower()
    if device_kind in {"", "cpu"}:
        accelerator = "none"
        accelerator_driver = None
        unmeasured = ("gpu", "tpu", "distributed")
    else:
        accelerator = device_kind
        info = backend.get_device_info()
        accelerator_driver = str(
            info.get("device_kind") or info.get("vendor") or info.get("platform") or ""
        )
        unmeasured = ("distributed",)
    return BackendFacts(
        backend_requested=requested,
        backend_actual=backend.name,
        backend_version=_backend_version(backend),
        device_kind=device_kind or "cpu",
        compilation_used=bool(backend.supports_compilation),
        accelerator=accelerator,
        accelerator_driver=accelerator_driver,
        memory_info=dict(backend.memory_info()),
        unmeasured=unmeasured,
    )


# =========================================================================
# Workload description
# =========================================================================


@dataclass(frozen=True, slots=True)
class WorkloadShape:
    """The problem dimensions a record must state to be reproducible."""

    workload: str
    n_antennas: int
    n_baselines: int
    n_point_sources: int
    n_healpix_pixels: int
    n_times: int
    n_frequencies: int
    sky_representation: str
    solver_workers: int
    loader_max_workers: int


# =========================================================================
# Timing
# =========================================================================


@dataclass(frozen=True, slots=True)
class TimingMeasurement:
    """One backend's timing profile for one workload."""

    setup_seconds: float
    compile_seconds: float
    steady_state_median_seconds: float
    steady_state_min_seconds: float
    steady_state_max_seconds: float
    steady_state_iterations: int
    host_transfer_seconds: float
    peak_host_bytes: int
    host_result: np.ndarray


def time_backend_call(
    call: Callable[[], Any],
    *,
    backend: ArrayBackend,
    iterations: int = DEFAULT_STEADY_STATE_ITERATIONS,
) -> TimingMeasurement:
    """Measure ``call`` under the Section 22.2 discipline.

    Parameters
    ----------
    call
        A zero-argument callable returning the backend-domain result. It is
        invoked ``iterations + 2`` times: once for setup, ``iterations`` times
        for steady state, and once more under ``tracemalloc`` for peak memory.
    backend
        The backend that produced the result. Used for ``synchronize`` before
        every clock stop and for the single ``to_numpy`` transfer.
    iterations
        Steady-state sample count. Section 22.2 requires at least five.

    Returns
    -------
    TimingMeasurement
        Setup, compile, steady-state, transfer, and peak-memory figures, plus
        the host-side result so the caller can compare it against the reference.
    """
    if iterations < DEFAULT_STEADY_STATE_ITERATIONS:
        raise ValueError(
            "Section 22.2 requires at least "
            f"{DEFAULT_STEADY_STATE_ITERATIONS} steady-state iterations; "
            f"got {iterations}."
        )

    start = time.perf_counter()
    result = call()
    backend.synchronize(result)
    setup_seconds = time.perf_counter() - start

    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = call()
        backend.synchronize(result)
        samples.append(time.perf_counter() - start)

    median = statistics.median(samples)

    # Peak host memory is measured in its own untimed iteration: tracemalloc
    # perturbs wall clock by a large factor, and a record that reported both
    # from one pass would be reporting a traced runtime as a steady state.
    tracemalloc.start()
    traced = call()
    backend.synchronize(traced)
    peak_host_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    start = time.perf_counter()
    host_result = np.asarray(backend.to_numpy(result))
    host_transfer_seconds = time.perf_counter() - start

    return TimingMeasurement(
        setup_seconds=setup_seconds,
        compile_seconds=max(0.0, setup_seconds - median),
        steady_state_median_seconds=median,
        steady_state_min_seconds=min(samples),
        steady_state_max_seconds=max(samples),
        steady_state_iterations=iterations,
        host_transfer_seconds=host_transfer_seconds,
        peak_host_bytes=int(peak_host_bytes),
        host_result=host_result,
    )


# =========================================================================
# Correctness
# =========================================================================


@dataclass(frozen=True, slots=True)
class Deviation:
    """How far a candidate result sits from the NumPy reference."""

    reference_backend: str
    max_absolute_deviation: float
    max_relative_deviation: float
    tolerance_rtol: float
    tolerance_atol: float
    within_tolerance: bool


def compare_to_reference(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    reference_backend: str = "numpy",
    rtol: float = FLOAT64_RTOL,
    atol_scale: float = FLOAT64_ATOL_SCALE,
) -> Deviation:
    """Apply the Section 13.5 predicate and report the measured deviation.

    ``|V_candidate - V_reference| <= atol + rtol * |V_reference|`` with
    ``atol = atol_scale * max(1, max|V_reference|)``.
    """
    if candidate.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: candidate {candidate.shape} vs "
            f"reference {reference.shape}"
        )
    scale = max(1.0, float(np.max(np.abs(reference))))
    atol = atol_scale * scale
    difference = np.abs(candidate - reference)
    allowed = atol + rtol * np.abs(reference)
    denominator = np.where(np.abs(reference) > 0, np.abs(reference), np.inf)
    return Deviation(
        reference_backend=reference_backend,
        max_absolute_deviation=float(np.max(difference)) if difference.size else 0.0,
        max_relative_deviation=(
            float(np.max(difference / denominator)) if difference.size else 0.0
        ),
        tolerance_rtol=rtol,
        tolerance_atol=atol,
        within_tolerance=bool(np.all(difference <= allowed)),
    )


# =========================================================================
# Record assembly
# =========================================================================


def build_record(
    *,
    environment: EnvironmentFacts,
    backend_facts: BackendFacts,
    shape: WorkloadShape,
    timing: TimingMeasurement,
    deviation: Deviation,
    precision: PrecisionConfig,
    precision_preset: str | None,
) -> BenchmarkRecord:
    """Assemble one complete Section 23 record from measured parts."""
    return BenchmarkRecord.create(
        schema_version=BENCHMARK_SCHEMA_VERSION,
        recorded_at_utc=environment.recorded_at_utc,
        radiosim_version=environment.radiosim_version,
        git_sha=environment.git_sha,
        platform=environment.platform,
        cpu_model=environment.cpu_model,
        cpu_count_logical=environment.cpu_count_logical,
        accelerator=backend_facts.accelerator,
        accelerator_driver=backend_facts.accelerator_driver,
        backend_requested=backend_facts.backend_requested,
        backend_actual=backend_facts.backend_actual,
        backend_version=backend_facts.backend_version,
        device_kind=backend_facts.device_kind,
        compilation_used=backend_facts.compilation_used,
        precision_preset=precision_preset,
        precision_default=precision.default,
        precision_accumulation=precision.accumulation,
        precision_output=precision.output,
        result_dtype=str(timing.host_result.dtype),
        workload=shape.workload,
        n_antennas=shape.n_antennas,
        n_baselines=shape.n_baselines,
        n_point_sources=shape.n_point_sources,
        n_healpix_pixels=shape.n_healpix_pixels,
        n_times=shape.n_times,
        n_frequencies=shape.n_frequencies,
        sky_representation=shape.sky_representation,
        solver_workers=shape.solver_workers,
        loader_max_workers=shape.loader_max_workers,
        setup_seconds=timing.setup_seconds,
        compile_seconds=timing.compile_seconds,
        steady_state_median_seconds=timing.steady_state_median_seconds,
        steady_state_min_seconds=timing.steady_state_min_seconds,
        steady_state_max_seconds=timing.steady_state_max_seconds,
        steady_state_iterations=timing.steady_state_iterations,
        host_transfer_seconds=timing.host_transfer_seconds,
        peak_host_bytes=timing.peak_host_bytes,
        backend_memory_info=backend_facts.memory_info,
        reference_backend=deviation.reference_backend,
        max_absolute_deviation=deviation.max_absolute_deviation,
        max_relative_deviation=deviation.max_relative_deviation,
        tolerance_rtol=deviation.tolerance_rtol,
        tolerance_atol=deviation.tolerance_atol,
        within_tolerance=deviation.within_tolerance,
        unmeasured=backend_facts.unmeasured,
    )


# =========================================================================
# Output location
# =========================================================================


def benchmark_output_directory(repository_root: Path | None = None) -> Path:
    """Return ``output/benchmarks/`` for this checkout (Section 22.1)."""
    root = repository_root or Path(__file__).resolve().parents[3]
    return root / "output" / "benchmarks"


def benchmark_filename(when: datetime | None = None) -> str:
    """Return ``<UTC timestamp>-<host tag>.json`` (Section 22.1).

    The host tag is the operating system and machine architecture, not the
    machine's name: a committed record should describe the class of host that
    produced it without carrying somebody's hostname into version control.
    """
    moment = when or datetime.now(UTC)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    host_tag = f"{platform.system().lower()}-{platform.machine().lower()}"
    return f"{stamp}-{host_tag}.json"


# =========================================================================
# PERF-001 P-a/P-b measurement infrastructure
# =========================================================================


def _precision_values(backend: ArrayBackend) -> tuple[str, str, str, str]:
    """Return the preset label and exact three common precision fields."""
    from radiosim.core.precision import PrecisionConfig

    precision = backend.precision or PrecisionConfig.standard()
    preset = "standard" if backend.precision is None else "explicit"
    return preset, precision.default, precision.accumulation, precision.output


def _perf001_backend_version(backend: ArrayBackend) -> str:
    """Read the executing backend's distribution version without imports."""
    distribution = {
        "numpy": "numpy",
        "jax": "jax",
        "dask": "dask",
    }.get(backend.backend_type)
    if distribution is None:
        raise BenchmarkRecordError(
            f"PERF-001 does not know the distribution for {backend.name!r}"
        )
    resolved = _installed_distribution_version(distribution)
    if resolved == "not-installed":
        raise BenchmarkRecordError(
            f"PERF-001 backend {backend.name!r} has no installed distribution metadata"
        )
    return resolved


def _perf001_context(
    backend: ArrayBackend,
    *,
    backend_requested: str,
    policy_id: str,
    input_identity: str,
    result_dtype: str,
    measurement_limitations: tuple[str, ...],
) -> MeasurementContext:
    preset, default, accumulation, output = _precision_values(backend)
    return MeasurementContext.create(
        backend_requested=backend_requested,
        backend_actual=backend.name,
        backend_version=_perf001_backend_version(backend),
        device_kind=str(backend.device_kind),
        compilation_used=bool(backend.supports_compilation),
        precision_preset=preset,
        precision_default=default,
        precision_accumulation=accumulation,
        precision_output=output,
        result_dtype=result_dtype,
        policy_id=policy_id,
        input_identity_sha256=input_identity,
        measurement_limitations=measurement_limitations,
    )


def _require_synthetic_shape(n_baselines: int, n_sources: int) -> None:
    if type(n_baselines) is not int or n_baselines < 0:
        raise ValueError("n_baselines must be a nonnegative integer")
    if type(n_sources) is not int or n_sources < 0:
        raise ValueError("n_sources must be a nonnegative integer")


def _synthetic_contraction_inputs(
    *,
    n_baselines: int,
    n_sources: int,
    dtype: Any,
    seed: int,
    polarized: bool,
) -> tuple[tuple[Any, ...], tuple[tuple[str, np.ndarray], ...]]:
    """Build deterministic host inputs and their canonical identity sequence."""
    _require_synthetic_shape(n_baselines, n_sources)
    complex_dtype = np.dtype(dtype)
    if complex_dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("dtype must be complex64 or complex128")
    real_dtype = np.dtype(np.float32 if complex_dtype.itemsize == 8 else np.float64)
    rng = np.random.default_rng(seed)
    jones_shape = (n_baselines, n_sources, 2, 2)
    jones_p = np.asarray(
        rng.standard_normal(jones_shape) + 1j * rng.standard_normal(jones_shape),
        dtype=complex_dtype,
    )
    jones_q = np.asarray(
        rng.standard_normal(jones_shape) + 1j * rng.standard_normal(jones_shape),
        dtype=complex_dtype,
    )
    phase = np.asarray(
        np.exp(1j * rng.standard_normal((n_baselines, n_sources))),
        dtype=complex_dtype,
    )
    envelope = np.asarray(
        0.5 + rng.random((n_baselines, n_sources)),
        dtype=real_dtype,
    )
    if polarized:
        coherency = np.asarray(
            rng.standard_normal((n_sources, 2, 2))
            + 1j * rng.standard_normal((n_sources, 2, 2)),
            dtype=complex_dtype,
        )
        stokes_i = None
        identity_inputs = (
            ("jones_p", jones_p),
            ("jones_q", jones_q),
            ("coherency", coherency),
            ("phase", phase),
            ("envelope", envelope),
        )
    else:
        coherency = None
        stokes_i = np.asarray(0.5 + rng.random(n_sources), dtype=real_dtype)
        identity_inputs = (
            ("jones_p", jones_p),
            ("jones_q", jones_q),
            ("stokes_i", stokes_i),
            ("phase", phase),
            ("envelope", envelope),
        )
    return (
        jones_p,
        jones_q,
        coherency,
        phase,
        envelope,
        stokes_i,
    ), identity_inputs


def _backend_inputs(backend: ArrayBackend, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    """Transfer each present host operand once, before a measured scope."""
    return tuple(None if item is None else backend.asarray(item) for item in inputs)


def _traced_call(call: Callable[[], Any], backend: ArrayBackend) -> tuple[Any, int]:
    """Run one untimed Python-heap peak measurement."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        result = call()
        backend.synchronize(result)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return result, int(peak)


class _LeafInvocationObserver:
    """Observe actual six-operand leaf calls outside any compiled function."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def wrap(self, leaf: Callable[..., Any]) -> Callable[..., Any]:
        def observed(*arguments: Any) -> Any:
            if len(arguments) != 6:
                raise BenchmarkRecordError(
                    "PERF-001 leaf schedule observation requires six operands"
                )
            jones_p, jones_q, coherency, phase, envelope, stokes_i = arguments
            jones_shape = tuple(int(value) for value in np.shape(jones_p))
            if len(jones_shape) != 4 or jones_shape[-2:] != (2, 2):
                raise BenchmarkRecordError(
                    "PERF-001 observed jones_p must have shape (B, S, 2, 2)"
                )
            n_baselines, n_sources = jones_shape[:2]
            if tuple(int(value) for value in np.shape(jones_q)) != jones_shape:
                raise BenchmarkRecordError(
                    "PERF-001 observed Jones operand shapes do not match"
                )
            if tuple(int(value) for value in np.shape(phase)) != (
                n_baselines,
                n_sources,
            ):
                raise BenchmarkRecordError(
                    "PERF-001 observed phase shape does not match Jones axes"
                )
            envelope_shape = tuple(int(value) for value in np.shape(envelope))
            if envelope_shape not in ((), (n_baselines, n_sources)):
                raise BenchmarkRecordError(
                    "PERF-001 observed envelope shape does not match Jones axes"
                )
            if (coherency is None) is (stokes_i is None):
                raise BenchmarkRecordError(
                    "PERF-001 observed leaf requires exactly one source signal"
                )
            signal_shape = tuple(
                int(value)
                for value in np.shape(coherency if coherency is not None else stokes_i)
            )
            expected_signal_shape = (
                (n_sources, 2, 2) if coherency is not None else (n_sources,)
            )
            if signal_shape != expected_signal_shape:
                raise BenchmarkRecordError(
                    "PERF-001 observed source-signal shape does not match Jones axes"
                )
            self.calls.append((n_baselines, n_sources))
            return leaf(*arguments)

        return observed


class _ObservedCompileBackend:
    """Delegate a backend while exposing every scheduled leaf invocation."""

    def __init__(
        self,
        backend: ArrayBackend,
        observer: _LeafInvocationObserver,
    ) -> None:
        self._backend = backend
        self._observer = observer

    @property
    def supports_compilation(self) -> bool:
        # Force the scheduler through ``compile`` even for NumPy/Dask. Their
        # compile method is the documented identity, so execution is unchanged
        # while the observer still sits outside the leaf.
        return True

    def compile(self, function: Callable[..., Any]) -> Callable[..., Any]:
        # The production contraction factory owns the package's sole compile
        # call site.  This adapter only delegates that already-observed call to
        # the wrapped backend; keeping the bound compiler separate also avoids
        # presenting the evidence harness as a second kernel compile boundary.
        compiler = self._backend.compile
        compiled = (
            compiler(function) if self._backend.supports_compilation else function
        )
        return self._observer.wrap(compiled)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


def _observe_memory_leaf_schedule(
    factory: Callable[..., Any],
    backend: ArrayBackend,
    *,
    target_kernel_pairs: int | None,
    inputs: tuple[Any, ...],
) -> tuple[Any, tuple[int, ...], int, tuple[int, ...]]:
    """Run one untimed matched call and return its observed B and pair sequence."""
    observer = _LeafInvocationObserver()
    observed_backend = _ObservedCompileBackend(backend, observer)
    wrapper = factory(
        cast("ArrayBackend", observed_backend),
        target_kernel_pairs=target_kernel_pairs,
    )
    result = wrapper(*inputs)
    backend.synchronize(result)
    if not observer.calls:
        raise BenchmarkRecordError(
            "PERF-001 memory measurement observed no contraction leaf calls"
        )
    chunks = tuple(n_baselines for n_baselines, _ in observer.calls)
    source_counts = tuple(n_sources for _, n_sources in observer.calls)
    if len(set(source_counts)) != 1:
        raise BenchmarkRecordError(
            "PERF-001 memory measurement observed inconsistent leaf source axes"
        )
    pair_counts = tuple(
        n_baselines * n_sources for n_baselines, n_sources in observer.calls
    )
    return result, chunks, source_counts[0], pair_counts


def _require_matched_results(
    backend: ArrayBackend,
    reference: Any,
    candidate: Any,
    *,
    scope: str,
) -> None:
    reference_host = np.asarray(backend.to_numpy(reference))
    candidate_host = np.asarray(backend.to_numpy(candidate))
    if reference_host.dtype != candidate_host.dtype:
        raise BenchmarkRecordError(
            f"PERF-001 {scope} changed result dtype from "
            f"{reference_host.dtype} to {candidate_host.dtype}"
        )
    if backend.supports_compilation:
        deviation = compare_to_reference(reference_host, candidate_host)
        if not deviation.within_tolerance:
            raise BenchmarkRecordError(
                f"PERF-001 {scope} exceeds the accepted JAX parity predicate"
            )
    elif not (
        np.array_equal(reference_host, candidate_host)
        and reference_host.tobytes(order="C") == candidate_host.tobytes(order="C")
    ):
        raise BenchmarkRecordError(f"PERF-001 {scope} changed NumPy/Dask result bytes")


def measure_perf001_memory_scaling_pair(
    backend: ArrayBackend,
    *,
    provenance: Perf001Provenance,
    n_baselines: int,
    n_sources: int,
    comparison_id: str,
    dtype: Any = np.complex128,
    seed: int = 20260811,
    polarized: bool = True,
    backend_requested: str | None = None,
) -> tuple[MemoryScalingRecordV2, MemoryScalingRecordV2]:
    """Measure matched unbounded and production P-a wrapper-memory rows.

    Inputs are built and transferred before tracing.  The measured scope starts
    at the private scheduling wrapper and therefore includes retained chunk
    outputs and final assembly while excluding fixture construction and backend-
    native allocation, exactly as the strict schema states.
    """
    from radiosim.core.contraction import (
        _baseline_contraction_for_policy,  # pyright: ignore[reportPrivateUsage]
    )

    if type(provenance) is not Perf001Provenance:
        raise TypeError("provenance must be an exact Perf001Provenance")
    _require_synthetic_shape(n_baselines, n_sources)
    if type(comparison_id) is not str or not comparison_id:
        raise ValueError("comparison_id must be a non-empty string")
    host_inputs, identity_inputs = _synthetic_contraction_inputs(
        n_baselines=n_baselines,
        n_sources=n_sources,
        dtype=dtype,
        seed=seed,
        polarized=polarized,
    )
    manifest = {
        "schema_version": "radiosim.perf001.fixture.contraction_memory.v1",
        "seed": seed,
        "logical_n_baselines": n_baselines,
        "logical_n_sources": n_sources,
        "dtype": np.dtype(dtype).str,
        "signal_path": "coherency" if polarized else "stokes_i",
        "envelope": "array",
    }
    input_identity = perf001_input_identity_sha256(manifest, identity_inputs)
    transferred = _backend_inputs(backend, host_inputs)
    synthetic_input_bytes = sum(array.nbytes for _, array in identity_inputs)
    requested = backend_requested or backend.backend_type
    limitations = (
        "tracemalloc excludes backend-native and device allocations",
        "fixture inputs are preallocated outside the traced scope",
    )

    measured: list[
        tuple[Any, int, str, int | None, tuple[int, ...], int, tuple[int, ...]]
    ] = []
    for _state, target, policy_id in (
        (
            "unchunked_reference",
            None,
            "unbounded_reference_v1",
        ),
        (
            "chunked_production",
            PERF001_TARGET_KERNEL_PAIRS,
            "target_kernel_pairs_131072_v1",
        ),
    ):
        wrapper = _baseline_contraction_for_policy(
            backend,
            target_kernel_pairs=target,
        )
        result, peak = _traced_call(
            lambda wrapper=wrapper: wrapper(*transferred),
            backend,
        )
        observed_result, chunks, kernel_n_sources, pair_counts = (
            _observe_memory_leaf_schedule(
                _baseline_contraction_for_policy,
                backend,
                target_kernel_pairs=target,
                inputs=transferred,
            )
        )
        _require_matched_results(
            backend,
            result,
            observed_result,
            scope=f"{_state} observed leaf schedule",
        )
        if sum(chunks) != n_baselines:
            raise BenchmarkRecordError(
                "PERF-001 observed leaf baseline chunks do not sum to the "
                "logical baseline count"
            )
        if kernel_n_sources != n_sources:
            raise BenchmarkRecordError(
                "PERF-001 observed leaf source axis does not match the logical "
                "source count"
            )
        measured.append(
            (
                result,
                peak,
                policy_id,
                target,
                chunks,
                kernel_n_sources,
                pair_counts,
            )
        )

    _require_matched_results(
        backend,
        measured[0][0],
        measured[1][0],
        scope="baseline contraction chunk policy",
    )
    result_dtype = str(np.asarray(backend.to_numpy(measured[0][0])).dtype)
    rows: list[MemoryScalingRecordV2] = []
    for state, measurement in zip(
        ("unchunked_reference", "chunked_production"), measured, strict=True
    ):
        _, peak, policy_id, target, chunks, kernel_n_sources, pair_counts = measurement
        rows.append(
            MemoryScalingRecordV2.create(
                schema_version=PERF001_MEMORY_SCALING_SCHEMA_VERSION,
                provenance=provenance,
                context=_perf001_context(
                    backend,
                    backend_requested=requested,
                    policy_id=policy_id,
                    input_identity=input_identity,
                    result_dtype=result_dtype,
                    measurement_limitations=limitations,
                ),
                comparison_id=comparison_id,
                implementation_state=state,
                measurement_scope=(
                    "contraction_wrapper_python_heap_including_output_assembly"
                ),
                allocator="python_heap_tracemalloc",
                includes_backend_native_allocations=False,
                inputs_preallocated=True,
                includes_solver_input_construction=False,
                includes_output_reassembly=True,
                logical_n_baselines=n_baselines,
                logical_n_sources=n_sources,
                logical_pair_count=n_baselines * n_sources,
                kernel_n_sources=kernel_n_sources,
                target_kernel_pairs=target,
                kernel_baseline_chunks=chunks,
                kernel_pair_counts=pair_counts,
                max_kernel_pair_count=max(pair_counts),
                synthetic_input_bytes_excluded=synthetic_input_bytes,
                peak_host_bytes=peak,
                notes=(
                    "Matched deterministic contraction fixture. The recorded "
                    "leaf pair counts mechanize the source-dependent bound; "
                    "the wrapper peak still includes baseline-dependent output "
                    "retention and assembly."
                ),
            )
        )
    return rows[0], rows[1]


def _operand_shape_dtype(value: Any) -> tuple[tuple[int, ...] | None, str | None]:
    if value is None:
        return None, None
    shape = tuple(int(dimension) for dimension in np.shape(value))
    dtype = str(np.dtype(getattr(value, "dtype", np.asarray(value).dtype)))
    return shape, dtype


@dataclass(slots=True)
class _SignatureTiming:
    values: tuple[tuple[int, ...] | None | str, ...]
    call_count: int
    first_call_seconds: float
    minimum_repeat_call_seconds: float | None


class _ContractionCompileSpy:
    """Observe complete six-operand leaf signatures without JAX cache APIs."""

    def __init__(self, backend: ArrayBackend):
        self._backend = backend
        self._ordered: list[_SignatureTiming] = []
        self._by_key: dict[
            tuple[tuple[int, ...] | None | str, ...], _SignatureTiming
        ] = {}

    def wrap(self, compiled: Callable[..., Any]) -> Callable[..., Any]:
        def observed(*arguments: Any) -> Any:
            if len(arguments) != 6:
                raise BenchmarkRecordError(
                    "PERF-001 compile spy requires the complete six-operand leaf"
                )
            key_parts: list[tuple[int, ...] | None | str] = []
            for argument in arguments:
                shape, dtype = _operand_shape_dtype(argument)
                key_parts.extend((shape, dtype))
            key = tuple(key_parts)
            started = time.perf_counter()
            result = compiled(*arguments)
            self._backend.synchronize(result)
            elapsed = time.perf_counter() - started
            timing = self._by_key.get(key)
            if timing is None:
                timing = _SignatureTiming(
                    values=key,
                    call_count=1,
                    first_call_seconds=elapsed,
                    minimum_repeat_call_seconds=None,
                )
                self._by_key[key] = timing
                self._ordered.append(timing)
            else:
                timing.call_count += 1
                if timing.minimum_repeat_call_seconds is None:
                    timing.minimum_repeat_call_seconds = elapsed
                else:
                    timing.minimum_repeat_call_seconds = min(
                        timing.minimum_repeat_call_seconds,
                        elapsed,
                    )
            return result

        return observed

    def snapshot_call_count(self) -> int:
        """Return the exact number of leaf invocations observed so far."""
        return sum(timing.call_count for timing in self._ordered)

    def observations(self) -> tuple[ContractionSignatureObservation, ...]:
        rows: list[ContractionSignatureObservation] = []
        for timing in self._ordered:
            if timing.minimum_repeat_call_seconds is None:
                raise BenchmarkRecordError(
                    "PERF-001 retracing fixture must repeat every complete leaf "
                    "signature at least once"
                )
            if timing.minimum_repeat_call_seconds <= 0.0:
                raise BenchmarkRecordError(
                    "PERF-001 repeat-call clock resolution must be positive"
                )
            values = timing.values
            rows.append(
                ContractionSignatureObservation.create(
                    jones_p_shape=values[0],
                    jones_q_shape=values[2],
                    coherency_shape=values[4],
                    phase_shape=values[6],
                    envelope_shape=values[8],
                    stokes_i_shape=values[10],
                    jones_p_dtype=values[1],
                    jones_q_dtype=values[3],
                    coherency_dtype=values[5],
                    phase_dtype=values[7],
                    envelope_dtype=values[9],
                    stokes_i_dtype=values[11],
                    call_count=timing.call_count,
                    first_call_seconds=timing.first_call_seconds,
                    minimum_repeat_call_seconds=timing.minimum_repeat_call_seconds,
                )
            )
        return tuple(rows)


def _compile_spied_contraction(
    backend: ArrayBackend,
) -> tuple[Callable[..., Any], _ContractionCompileSpy]:
    """Build the real production wrapper with its one compile site observed."""
    from radiosim.core.contraction import (
        _baseline_contraction_for_policy,  # pyright: ignore[reportPrivateUsage]
    )

    observer = _ContractionCompileSpy(backend)
    original_compile = backend.compile

    def observed_compile(function: Callable[..., Any]) -> Callable[..., Any]:
        return observer.wrap(original_compile(function))

    had_instance_attribute = "compile" in getattr(backend, "__dict__", {})
    prior_instance_attribute = getattr(backend, "__dict__", {}).get("compile")
    backend.compile = observed_compile  # type: ignore[method-assign]
    try:
        wrapper = _baseline_contraction_for_policy(
            backend,
            target_kernel_pairs=PERF001_TARGET_KERNEL_PAIRS,
        )
    finally:
        if had_instance_attribute:
            backend.compile = prior_instance_attribute  # type: ignore[method-assign]
        else:
            del backend.compile
    return wrapper, observer


def _pad_synthetic_inputs(
    inputs: tuple[Any, ...],
    *,
    kernel_n_sources: int,
) -> tuple[Any, ...]:
    """Append repeated finite operands and exact-zero signal dummy rows."""
    jones_p, jones_q, coherency, phase, envelope, stokes_i = inputs
    logical_n_sources = int(jones_p.shape[1])
    padding = kernel_n_sources - logical_n_sources
    if padding == 0:
        return inputs

    def repeat(values: np.ndarray, *, axis: int) -> np.ndarray:
        first = np.take(values, [0], axis=axis)
        return np.concatenate((values, np.repeat(first, padding, axis=axis)), axis=axis)

    jones_p = repeat(jones_p, axis=1)
    jones_q = repeat(jones_q, axis=1)
    phase = repeat(phase, axis=1)
    envelope = repeat(envelope, axis=1)
    if coherency is not None:
        coherency = np.concatenate(
            (
                coherency,
                np.zeros((padding, 2, 2), dtype=coherency.dtype),
            )
        )
    else:
        stokes_i = np.concatenate((stokes_i, np.zeros(padding, dtype=stokes_i.dtype)))
    return jones_p, jones_q, coherency, phase, envelope, stokes_i


def _synthetic_retracing_row(
    backend: ArrayBackend,
    *,
    provenance: Perf001Provenance,
    logical_counts: tuple[int, ...],
    n_baselines: int,
    comparison_id: str,
    input_identity: str,
    logical_by_count: Mapping[int, tuple[Any, ...]],
    production: bool,
    backend_requested: str,
) -> tuple[RetracingRecordV2, tuple[np.ndarray, ...]]:
    policy = "pow2_compiled_v1" if production else "identity_reference_v1"
    state = "bucketed_production" if production else "unbucketed_reference"
    kernel_counts = tuple(
        0 if count == 0 else (1 << (count - 1).bit_length() if production else count)
        for count in logical_counts
    )
    wrapper, observer = _compile_spied_contraction(backend)
    step_seconds: list[float] = []
    outputs: list[np.ndarray] = []
    for logical_count, kernel_count in zip(logical_counts, kernel_counts, strict=True):
        started = time.perf_counter()
        if logical_count == 0:
            result = backend.zeros_complex((n_baselines, 2, 2))
        else:
            host_inputs = logical_by_count[logical_count]
            if production:
                host_inputs = _pad_synthetic_inputs(
                    host_inputs,
                    kernel_n_sources=kernel_count,
                )
            result = wrapper(*_backend_inputs(backend, host_inputs))
        backend.synchronize(result)
        step_seconds.append(time.perf_counter() - started)
        outputs.append(np.asarray(backend.to_numpy(result)))

    observations = observer.observations()
    maximum_ratio = max(
        (
            observation.first_call_seconds / observation.minimum_repeat_call_seconds
            for observation in observations
        ),
        default=0.0,
    )
    overhead = sum(
        max(
            0.0,
            observation.first_call_seconds - observation.minimum_repeat_call_seconds,
        )
        for observation in observations
    )
    result_dtype = str(outputs[0].dtype)
    record = RetracingRecordV2.create(
        schema_version=PERF001_RETRACING_SCHEMA_VERSION,
        provenance=provenance,
        context=_perf001_context(
            backend,
            backend_requested=backend_requested,
            policy_id=policy,
            input_identity=input_identity,
            result_dtype=result_dtype,
            measurement_limitations=(
                "compile-spy synchronization is included in complete wrapper steps",
                "timing values are evidence and are not acceptance thresholds",
            ),
        ),
        comparison_id=comparison_id,
        implementation_state=state,
        measurement_scope="complete_synthetic_contraction_wrapper_step",
        solver="synthetic_wrapper",
        sky_representation="synthetic_contraction",
        bucket_policy=policy,
        padding_location="early_host" if production else "none",
        logical_source_counts=logical_counts,
        kernel_source_counts=kernel_counts,
        distinct_logical_source_counts=len(set(logical_counts)),
        distinct_kernel_source_counts=len(set(kernel_counts)),
        observed_signatures=observations,
        distinct_signature_count=len(observations),
        leaf_call_count=sum(item.call_count for item in observations),
        scope_step_seconds=tuple(step_seconds),
        scope_total_seconds=sum(step_seconds),
        max_first_to_repeat_ratio=maximum_ratio,
        retrace_overhead_seconds=overhead,
        notes=(
            "Complete six-operand signatures are observed at the production "
            "compile boundary; no private JAX cache state is inspected."
        ),
    )
    return record, tuple(outputs)


def measure_perf001_synthetic_retracing_pair(
    backend: ArrayBackend,
    *,
    provenance: Perf001Provenance,
    source_counts: Sequence[int],
    n_baselines: int,
    comparison_id: str,
    dtype: Any = np.complex128,
    seed: int = 20260811,
    polarized: bool = True,
    backend_requested: str = "jax",
) -> tuple[RetracingRecordV2, RetracingRecordV2]:
    """Measure matched P-b retracing rows through the real wrapper/leaf seam."""
    if not backend.supports_compilation or not backend.name.startswith("jax-"):
        raise BenchmarkRecordError(
            "PERF-001 retracing evidence requires a concrete compiling JAX backend"
        )
    if type(provenance) is not Perf001Provenance:
        raise TypeError("provenance must be an exact Perf001Provenance")
    if type(n_baselines) is not int or n_baselines <= 0:
        raise ValueError("n_baselines must be a positive integer")
    logical_counts = tuple(source_counts)
    if not logical_counts:
        raise ValueError("source_counts must be non-empty")
    if any(type(count) is not int or count < 0 for count in logical_counts):
        raise ValueError("source_counts must contain nonnegative integers")
    if type(comparison_id) is not str or not comparison_id:
        raise ValueError("comparison_id must be a non-empty string")

    logical_by_count: dict[int, tuple[Any, ...]] = {}
    identity_inputs: list[tuple[str, np.ndarray]] = []
    for count in logical_counts:
        if count == 0 or count in logical_by_count:
            continue
        inputs, named = _synthetic_contraction_inputs(
            n_baselines=n_baselines,
            n_sources=count,
            dtype=dtype,
            seed=seed + count,
            polarized=polarized,
        )
        logical_by_count[count] = inputs
        for name, array in named:
            identity_inputs.append((f"sources_{count}.{name}", array))
    if not identity_inputs:
        # An all-zero-visible fixture has no scientific leaf input and cannot
        # produce the schema-required signature observations.
        raise ValueError("source_counts must include at least one positive count")
    manifest = {
        "schema_version": "radiosim.perf001.fixture.synthetic_retracing.v1",
        "seed": seed,
        "logical_source_counts": list(logical_counts),
        "logical_n_baselines": n_baselines,
        "dtype": np.dtype(dtype).str,
        "signal_path": "coherency" if polarized else "stokes_i",
    }
    input_identity = perf001_input_identity_sha256(manifest, tuple(identity_inputs))
    reference, reference_outputs = _synthetic_retracing_row(
        backend,
        provenance=provenance,
        logical_counts=logical_counts,
        n_baselines=n_baselines,
        comparison_id=comparison_id,
        input_identity=input_identity,
        logical_by_count=logical_by_count,
        production=False,
        backend_requested=backend_requested,
    )
    production, production_outputs = _synthetic_retracing_row(
        backend,
        provenance=provenance,
        logical_counts=logical_counts,
        n_baselines=n_baselines,
        comparison_id=comparison_id,
        input_identity=input_identity,
        logical_by_count=logical_by_count,
        production=True,
        backend_requested=backend_requested,
    )
    for index, (reference_output, production_output) in enumerate(
        zip(reference_outputs, production_outputs, strict=True)
    ):
        _require_matched_results(
            backend,
            reference_output,
            production_output,
            scope=f"synthetic source bucket step {index}",
        )
    return reference, production


def _solver_module(solver: str) -> Any:
    """Return the production module that owns one private solver seam."""
    if solver == "point":
        import radiosim.core.visibility as solver_module

        return solver_module
    if solver == "healpix":
        import radiosim.core.visibility_healpix as solver_module

        return solver_module
    raise ValueError("solver must be 'point' or 'healpix'")


def _install_solver_contraction(
    solver: str,
    backend: ArrayBackend,
    wrapper: Callable[..., Any],
) -> tuple[Any, Callable[..., Any]]:
    """Install one persistent wrapper at a private solver factory seam."""
    module = _solver_module(solver)
    original_factory = module.baseline_contraction_for

    def evidence_factory(resolved_backend: ArrayBackend) -> Callable[..., Any]:
        if resolved_backend is not backend:
            raise BenchmarkRecordError(
                "PERF-001 solver runner changed backend inside the measured scope"
            )
        return wrapper

    module.baseline_contraction_for = evidence_factory
    return module, original_factory


def _restore_solver_contraction(
    module: Any, original_factory: Callable[..., Any]
) -> None:
    module.baseline_contraction_for = original_factory


def _require_solver_dimensions(
    *,
    logical_n_baselines: int,
    logical_source_counts: Sequence[int],
    n_times: int,
    n_frequencies: int,
) -> tuple[int, ...]:
    if type(logical_n_baselines) is not int or logical_n_baselines < 0:
        raise ValueError("logical_n_baselines must be a nonnegative integer")
    if type(n_times) is not int or n_times <= 0:
        raise ValueError("n_times must be a positive integer")
    if type(n_frequencies) is not int or n_frequencies <= 0:
        raise ValueError("n_frequencies must be a positive integer")
    counts = tuple(logical_source_counts)
    if len(counts) != n_times * n_frequencies:
        raise ValueError(
            "logical_source_counts must contain exactly n_times * n_frequencies entries"
        )
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("logical_source_counts must contain nonnegative integers")
    return counts


def _kernel_source_counts(
    logical_counts: tuple[int, ...], *, production: bool
) -> tuple[int, ...]:
    return tuple(
        0 if count == 0 else (1 << (count - 1).bit_length() if production else count)
        for count in logical_counts
    )


def _require_observed_source_counts(
    observations: tuple[ContractionSignatureObservation, ...],
    expected_counts: tuple[int, ...],
    *,
    scope: str,
) -> None:
    observed = {
        int(observation.jones_p_shape[1])
        for observation in observations
        if observation.jones_p_shape is not None
    }
    expected = {count for count in expected_counts if count > 0}
    if observed != expected:
        raise BenchmarkRecordError(
            f"PERF-001 {scope} observed kernel source counts {sorted(observed)}, "
            f"expected {sorted(expected)} from the selected bucket policy"
        )


def measure_perf001_solver_memory_pair(
    backend: ArrayBackend,
    *,
    provenance: Perf001Provenance,
    solver: str,
    logical_n_baselines: int,
    logical_source_counts: Sequence[int],
    n_times: int,
    n_frequencies: int,
    fixture_manifest: Mapping[str, object],
    logical_inputs: Sequence[tuple[str, Any]],
    run_solver: Callable[[str], Any],
    comparison_id: str,
    backend_requested: str = "jax",
) -> tuple[SolverMemoryRecord, SolverMemoryRecord]:
    """Measure matched direct-solver P-b Python-heap peaks.

    ``run_solver`` must call the private point or HEALPix solver with the policy
    string it receives.  Instrument, beam, sky, and other public fixture objects
    are prepared by the caller; the callback boundary begins at the private
    complete solver and therefore includes horizon selection, host bucketing,
    Jones/phase construction, contraction, and output assembly.  A persistent
    production contraction wrapper is installed at the solver's existing
    private factory seam so an untimed warm-up and the measured call share the
    same compiled leaf.
    """
    if not backend.supports_compilation or not backend.name.startswith("jax-"):
        raise BenchmarkRecordError(
            "PERF-001 solver-memory evidence requires a concrete compiling JAX backend"
        )
    if type(provenance) is not Perf001Provenance:
        raise TypeError("provenance must be an exact Perf001Provenance")
    logical_counts = _require_solver_dimensions(
        logical_n_baselines=logical_n_baselines,
        logical_source_counts=logical_source_counts,
        n_times=n_times,
        n_frequencies=n_frequencies,
    )
    if type(comparison_id) is not str or not comparison_id:
        raise ValueError("comparison_id must be a non-empty string")
    input_identity = perf001_input_identity_sha256(
        fixture_manifest,
        logical_inputs,
    )
    sky_representation = "point_sources" if solver == "point" else "healpix"
    limitations = (
        "tracemalloc excludes backend-native and device allocations",
        "fixture and public-API setup are outside the measured solver callback",
        "an untimed call warms the persistent contraction leaf before tracing",
    )
    measured: list[tuple[str, str, tuple[int, ...], Any, int]] = []
    for production in (False, True):
        policy = "pow2_compiled_v1" if production else "identity_reference_v1"
        state = "bucketed_production" if production else "unbucketed_reference"
        kernel_counts = _kernel_source_counts(
            logical_counts,
            production=production,
        )
        wrapper, observer = _compile_spied_contraction(backend)
        module, original_factory = _install_solver_contraction(
            solver,
            backend,
            wrapper,
        )
        try:
            warm = run_solver(policy)
            backend.synchronize(warm)
            result, peak = _traced_call(
                lambda policy=policy: run_solver(policy),
                backend,
            )
        finally:
            _restore_solver_contraction(module, original_factory)
        observations = observer.observations()
        _require_observed_source_counts(
            observations,
            kernel_counts,
            scope=f"{solver} solver-memory {state}",
        )
        measured.append((state, policy, kernel_counts, result, peak))

    _require_matched_results(
        backend,
        measured[0][3],
        measured[1][3],
        scope=f"complete {solver} solver source bucket policy",
    )
    result_dtype = str(np.asarray(backend.to_numpy(measured[0][3])).dtype)
    rows: list[SolverMemoryRecord] = []
    for state, policy, kernel_counts, _, peak in measured:
        rows.append(
            SolverMemoryRecord.create(
                schema_version=PERF001_SOLVER_MEMORY_SCHEMA_VERSION,
                provenance=provenance,
                context=_perf001_context(
                    backend,
                    backend_requested=backend_requested,
                    policy_id=policy,
                    input_identity=input_identity,
                    result_dtype=result_dtype,
                    measurement_limitations=limitations,
                ),
                comparison_id=comparison_id,
                implementation_state=state,
                measurement_scope=(
                    "direct_solver_step_python_heap_including_input_construction_"
                    "and_output_assembly"
                ),
                allocator="python_heap_tracemalloc",
                includes_backend_native_allocations=False,
                includes_simulator_setup=False,
                includes_solver_input_construction=True,
                includes_output_assembly=True,
                solver=solver,
                sky_representation=sky_representation,
                logical_n_baselines=logical_n_baselines,
                logical_source_counts=logical_counts,
                kernel_source_counts=kernel_counts,
                n_times=n_times,
                n_frequencies=n_frequencies,
                target_kernel_pairs=PERF001_TARGET_KERNEL_PAIRS,
                bucket_policy=policy,
                peak_host_bytes=peak,
                notes=(
                    "Direct private solver call with complete host preprocessing, "
                    "Jones/phase construction, contraction, and output assembly."
                ),
            )
        )
    return rows[0], rows[1]


def _solver_retracing_row(
    backend: ArrayBackend,
    *,
    provenance: Perf001Provenance,
    solver: str,
    logical_counts: tuple[int, ...],
    comparison_id: str,
    input_identity: str,
    run_solver_step: Callable[[str, int], Any],
    production: bool,
    backend_requested: str,
    leaf_call_delta_observer: Callable[[str, int, int, int], None] | None,
) -> tuple[RetracingRecordV2, tuple[np.ndarray, ...]]:
    policy = "pow2_compiled_v1" if production else "identity_reference_v1"
    state = "bucketed_production" if production else "unbucketed_reference"
    kernel_counts = _kernel_source_counts(logical_counts, production=production)
    wrapper, observer = _compile_spied_contraction(backend)
    module, original_factory = _install_solver_contraction(solver, backend, wrapper)
    outputs: list[np.ndarray] = []
    step_seconds: list[float] = []
    try:
        for step_index, logical_count in enumerate(logical_counts):
            calls_before = observer.snapshot_call_count()
            started = time.perf_counter()
            result = run_solver_step(policy, step_index)
            backend.synchronize(result)
            step_seconds.append(time.perf_counter() - started)
            outputs.append(np.asarray(backend.to_numpy(result)))
            leaf_call_delta = observer.snapshot_call_count() - calls_before
            if logical_count == 0 and leaf_call_delta != 0:
                raise BenchmarkRecordError(
                    f"PERF-001 {solver} {state} zero-visible step {step_index} "
                    f"reached the contraction leaf {leaf_call_delta} time(s)"
                )
            if logical_count > 0 and leaf_call_delta <= 0:
                raise BenchmarkRecordError(
                    f"PERF-001 {solver} {state} visible-source step {step_index} "
                    "did not reach the contraction leaf"
                )
            if leaf_call_delta_observer is not None:
                leaf_call_delta_observer(
                    state,
                    step_index,
                    logical_count,
                    leaf_call_delta,
                )
    finally:
        _restore_solver_contraction(module, original_factory)
    observations = observer.observations()
    _require_observed_source_counts(
        observations,
        kernel_counts,
        scope=f"{solver} retracing {state}",
    )
    maximum_ratio = max(
        (
            observation.first_call_seconds / observation.minimum_repeat_call_seconds
            for observation in observations
        ),
        default=0.0,
    )
    overhead = sum(
        max(
            0.0,
            observation.first_call_seconds - observation.minimum_repeat_call_seconds,
        )
        for observation in observations
    )
    record = RetracingRecordV2.create(
        schema_version=PERF001_RETRACING_SCHEMA_VERSION,
        provenance=provenance,
        context=_perf001_context(
            backend,
            backend_requested=backend_requested,
            policy_id=policy,
            input_identity=input_identity,
            result_dtype=str(outputs[0].dtype),
            measurement_limitations=(
                "one complete private solver call is timed per logical step",
                "compile-spy synchronization is included in solver-step timings",
                "timing values are evidence and are not acceptance thresholds",
            ),
        ),
        comparison_id=comparison_id,
        implementation_state=state,
        measurement_scope=f"complete_{solver}_solver_step",
        solver=solver,
        sky_representation="point_sources" if solver == "point" else "healpix",
        bucket_policy=policy,
        padding_location="early_host" if production else "none",
        logical_source_counts=logical_counts,
        kernel_source_counts=kernel_counts,
        distinct_logical_source_counts=len(set(logical_counts)),
        distinct_kernel_source_counts=len(set(kernel_counts)),
        observed_signatures=observations,
        distinct_signature_count=len(observations),
        leaf_call_count=sum(item.call_count for item in observations),
        scope_step_seconds=tuple(step_seconds),
        scope_total_seconds=sum(step_seconds),
        max_first_to_repeat_ratio=maximum_ratio,
        retrace_overhead_seconds=overhead,
        notes=(
            "Private complete-solver source-bucket control with a persistent "
            "production contraction leaf; no JAX cache internals are inspected."
        ),
    )
    return record, tuple(outputs)


def measure_perf001_solver_retracing_pair(
    backend: ArrayBackend,
    *,
    provenance: Perf001Provenance,
    solver: str,
    logical_source_counts: Sequence[int],
    fixture_manifest: Mapping[str, object],
    logical_inputs: Sequence[tuple[str, Any]],
    run_solver_step: Callable[[str, int], Any],
    comparison_id: str,
    backend_requested: str = "jax",
    _leaf_call_delta_observer: (Callable[[str, int, int, int], None] | None) = None,
) -> tuple[RetracingRecordV2, RetracingRecordV2]:
    """Measure matched P-b retracing through complete private solver calls.

    The caller supplies one complete point or HEALPix private-solver invocation
    per logical step.  The harness supplies only the documented private bucket
    policy and temporarily installs one persistent production contraction
    wrapper at the solver module's existing factory seam.  Thus host work stays
    in scope and leaf signatures come from the real six-operand compile boundary.
    """
    if not backend.supports_compilation or not backend.name.startswith("jax-"):
        raise BenchmarkRecordError(
            "PERF-001 solver retracing requires a concrete compiling JAX backend"
        )
    if type(provenance) is not Perf001Provenance:
        raise TypeError("provenance must be an exact Perf001Provenance")
    _ = _solver_module(solver)
    logical_counts = tuple(logical_source_counts)
    if not logical_counts:
        raise ValueError("logical_source_counts must be non-empty")
    if any(type(count) is not int or count < 0 for count in logical_counts):
        raise ValueError("logical_source_counts must contain nonnegative integers")
    if type(comparison_id) is not str or not comparison_id:
        raise ValueError("comparison_id must be a non-empty string")
    input_identity = perf001_input_identity_sha256(
        fixture_manifest,
        logical_inputs,
    )
    reference, reference_outputs = _solver_retracing_row(
        backend,
        provenance=provenance,
        solver=solver,
        logical_counts=logical_counts,
        comparison_id=comparison_id,
        input_identity=input_identity,
        run_solver_step=run_solver_step,
        production=False,
        backend_requested=backend_requested,
        leaf_call_delta_observer=_leaf_call_delta_observer,
    )
    production, production_outputs = _solver_retracing_row(
        backend,
        provenance=provenance,
        solver=solver,
        logical_counts=logical_counts,
        comparison_id=comparison_id,
        input_identity=input_identity,
        run_solver_step=run_solver_step,
        production=True,
        backend_requested=backend_requested,
        leaf_call_delta_observer=_leaf_call_delta_observer,
    )
    for index, (reference_output, production_output) in enumerate(
        zip(reference_outputs, production_outputs, strict=True)
    ):
        _require_matched_results(
            backend,
            reference_output,
            production_output,
            scope=f"complete {solver} source bucket step {index}",
        )
    return reference, production


# =========================================================================
# The two Tier 6H acceptance obligations
# =========================================================================


def measure_retracing(
    backend: ArrayBackend,
    *,
    source_counts: Sequence[int],
    n_baselines: int = 6,
    dtype: Any = np.complex128,
) -> RetracingRecord:
    """Measure recompilation cost when the kernel's source axis changes size.

    Both solvers mask by ``above_horizon`` per time step, so within one run the
    compiled contraction can be handed a different source count at every step.
    Section 22.2's timing loop repeats an *identical* call and therefore cannot
    see this. This measurement walks ``source_counts`` in order -- the way a real
    observation walks a time axis whose visible-source count rises and falls --
    and counts how many times the kernel body was actually entered.

    What is measured is *cost*, not an internal trace counter: the kernel used is
    the production one from :func:`radiosim.core.contraction.baseline_contraction_for`
    -- Section 13.6 authorizes exactly one compiled kernel and this harness does
    not create a second one -- and retracing shows up as the excess of the first
    call at a given source count over a later call at that same source count.
    On a compiling backend that excess is a recompilation; on a non-compiling one
    it is ordinary cache warming, and the ratio between the two backends is the
    result.

    Parameters
    ----------
    backend
        The backend under test. ``supports_compilation`` decides whether the
        kernel is compiled, exactly as ``core/contraction.py`` decides it.
    source_counts
        The per-step source counts, in order. Every count must appear at least
        twice for its retrace cost to be separable from its steady-state cost.
    n_baselines
        Baseline count held fixed across steps, as in a real run.
    dtype
        Complex dtype for the synthetic Jones and coherency inputs.

    Returns
    -------
    RetracingRecord
        Distinct shape count, first- and repeat-call time per source count, the
        worst first-to-repeat ratio, and the total time attributable to
        retracing.
    """
    from radiosim.core.contraction import baseline_contraction_for

    compiled = baseline_contraction_for(backend)

    def inputs(n_sources: int) -> tuple[Any, ...]:
        rng = np.random.default_rng(20260731 + n_sources)
        shape = (n_baselines, n_sources, 2, 2)
        jones_p = backend.asarray(
            (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
        )
        jones_q = backend.asarray(
            (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
        )
        coherency = backend.asarray(
            (
                rng.standard_normal((n_sources, 2, 2))
                + 1j * rng.standard_normal((n_sources, 2, 2))
            ).astype(dtype)
        )
        phase = backend.asarray(
            np.exp(1j * rng.standard_normal((n_baselines, n_sources))).astype(dtype)
        )
        envelope = backend.asarray(np.ones((n_baselines, n_sources), dtype=np.float64))
        return jones_p, jones_q, coherency, phase, envelope

    first_seconds: dict[str, float] = {}
    repeat_seconds: dict[str, float] = {}
    total_seconds = 0.0
    for n_sources in source_counts:
        arguments = inputs(n_sources)
        start = time.perf_counter()
        result = compiled(*arguments, None)
        backend.synchronize(result)
        elapsed = time.perf_counter() - start
        total_seconds += elapsed
        key = str(n_sources)
        if key in first_seconds:
            repeat_seconds[key] = min(repeat_seconds.get(key, elapsed), elapsed)
        else:
            first_seconds[key] = elapsed

    # A repeat call at a shape already seen costs steady-state time; the excess
    # of the first call over it is what a retrace cost.
    retrace_overhead = 0.0
    worst_ratio = 0.0
    for key, first in first_seconds.items():
        steady = repeat_seconds.get(key)
        if steady is None:
            continue
        retrace_overhead += max(0.0, first - steady)
        if steady > 0.0:
            worst_ratio = max(worst_ratio, first / steady)

    return RetracingRecord.create(
        schema_version=RETRACING_SCHEMA_VERSION,
        recorded_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        backend_actual=backend.name,
        compilation_used=bool(backend.supports_compilation),
        source_counts=tuple(int(count) for count in source_counts),
        distinct_source_counts=len(set(source_counts)),
        steps=len(tuple(source_counts)),
        first_call_seconds_by_source_count=first_seconds,
        repeat_call_seconds_by_source_count=repeat_seconds,
        max_first_to_repeat_ratio=worst_ratio,
        total_seconds=total_seconds,
        retrace_overhead_seconds=retrace_overhead,
        notes=(
            "Section 13.6 calls the kernel 'shape-stable within a run'. Both "
            "solvers mask sources by above_horizon per time step, so the source "
            "axis can change size step to step. Under a compiling backend each "
            "newly seen source count costs a recompilation before it costs "
            "arithmetic; the first-to-repeat ratio is that cost."
        ),
    )


def measure_kernel_memory_scaling(
    backend: ArrayBackend,
    *,
    n_baselines: int,
    n_sources: int,
    dtype: Any = np.complex128,
) -> MemoryScalingRecord:
    """Measure the compiled kernel's ``(B, S, 2, 2)`` working set.

    ``core/contraction.py`` materializes two ``(B, S, 2, 2)`` antenna-Jones
    batches plus a ``(B, S, 2, 2)`` product per ``(time, frequency)`` step, so
    peak memory is ``O(baselines x sources)`` where the per-baseline Python loop
    it replaced was ``O(sources)``. No Section 13.4 workload and no shipped
    configuration exceeds fifteen baselines, so correctness tests cannot see
    this. The measured slope is what turns the hazard into a bounded, tracked
    fact.

    The inputs are excluded from the traced region: what is measured is the
    kernel's own working set, not the cost of constructing its arguments.
    """
    from radiosim.core.contraction import (
        _baseline_contraction_for_policy,  # pyright: ignore[reportPrivateUsage]
    )

    # This frozen v1 measurement predates the production chunk policy and its
    # retained records describe the unbounded leaf.  Keep measuring that exact
    # historical control; PERF-001 v2 owns the paired production comparison.
    kernel = _baseline_contraction_for_policy(backend, target_kernel_pairs=None)
    rng = np.random.default_rng(20260731)
    shape = (n_baselines, n_sources, 2, 2)
    jones_p = backend.asarray(
        (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    )
    jones_q = backend.asarray(
        (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    )
    coherency = backend.asarray(
        (
            rng.standard_normal((n_sources, 2, 2))
            + 1j * rng.standard_normal((n_sources, 2, 2))
        ).astype(dtype)
    )
    phase = backend.asarray(
        np.exp(1j * rng.standard_normal((n_baselines, n_sources))).astype(dtype)
    )
    envelope = backend.asarray(np.ones((n_baselines, n_sources), dtype=np.float64))

    tracemalloc.start()
    tracemalloc.reset_peak()
    result = kernel(jones_p, jones_q, coherency, phase, envelope, None)
    backend.synchronize(result)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    pair_count = n_baselines * n_sources
    return MemoryScalingRecord.create(
        schema_version=MEMORY_SCALING_SCHEMA_VERSION,
        recorded_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        backend_actual=backend.name,
        n_baselines=n_baselines,
        n_sources=n_sources,
        pair_count=pair_count,
        peak_host_bytes=int(peak),
        bytes_per_pair=float(peak) / pair_count if pair_count else 0.0,
        notes=(
            "Peak traced host allocation for one baseline_contraction call, "
            "inputs excluded. The known one-line mitigation, if this ever needs "
            "bounding, is to chunk the baseline axis inside "
            "baseline_contraction_for."
        ),
    )
