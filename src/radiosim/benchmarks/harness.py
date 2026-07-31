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

import os
import platform
import statistics
import subprocess
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from radiosim.benchmarks.record import (
    BENCHMARK_SCHEMA_VERSION,
    MEMORY_SCALING_SCHEMA_VERSION,
    RETRACING_SCHEMA_VERSION,
    BenchmarkRecord,
    MemoryScalingRecord,
    RetracingRecord,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard only
    from radiosim.backends.base import ArrayBackend
    from radiosim.core.precision import PrecisionConfig

__all__ = [
    "DEFAULT_STEADY_STATE_ITERATIONS",
    "BackendFacts",
    "Deviation",
    "EnvironmentFacts",
    "TimingMeasurement",
    "WorkloadShape",
    "benchmark_filename",
    "benchmark_output_directory",
    "build_record",
    "compare_to_reference",
    "describe_backend",
    "describe_environment",
    "measure_kernel_memory_scaling",
    "measure_retracing",
    "time_backend_call",
]

#: Section 22.2: "the median of at least 5 iterations".
DEFAULT_STEADY_STATE_ITERATIONS = 5

#: Section 13.5 tolerance for float64 accumulation.
FLOAT64_RTOL = 1e-12
FLOAT64_ATOL_SCALE = 1e-12


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
    from radiosim.core.contraction import baseline_contraction_for

    kernel = baseline_contraction_for(backend)
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
