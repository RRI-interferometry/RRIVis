"""Reproducible backend benchmarks (``Tier6HybridRuntimePlan.md`` Sections 22-23).

RadioSim publishes no speed claim without a record. This package is the record:
:mod:`radiosim.benchmarks.record` defines the schema, which is complete or it is
an error, and :mod:`radiosim.benchmarks.harness` defines the measurement
discipline that fills it in.

The benchmarks themselves live in ``tests/performance/`` behind the
``performance`` and ``slow`` markers, so they never gate CI. Run them with::

    pixi run bench

Output lands in ``output/benchmarks/<UTC timestamp>-<host tag>.json``.

Importing this package pulls in NumPy and the standard library only; no backend,
no solver, and no optional dependency is imported at module load, so the
package-level import laziness the rest of RadioSim maintains is preserved.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from radiosim.benchmarks.harness import (
    DEFAULT_STEADY_STATE_ITERATIONS,
    PERF001_REFERENCE_SHA256,
    PERF001_REFERENCE_SOURCE_SHA,
    BackendFacts,
    BenchmarkBackendSelection,
    Deviation,
    EnvironmentFacts,
    Perf001ReferenceAuthentication,
    TimingMeasurement,
    WorkloadShape,
    assemble_perf001_cpu_evidence_document,
    authenticate_perf001_references,
    benchmark_backend_selection,
    benchmark_filename,
    benchmark_output_directory,
    build_perf001_workload_record,
    build_record,
    compare_to_reference,
    describe_backend,
    describe_environment,
    describe_perf001_provenance,
    measure_kernel_memory_scaling,
    measure_perf001_backend_resolution,
    measure_perf001_memory_scaling_pair,
    measure_perf001_solver_memory_pair,
    measure_perf001_solver_retracing_pair,
    measure_perf001_synthetic_retracing_pair,
    measure_retracing,
    perf001_control_identity_sha256,
    perf001_input_identity_sha256,
    perf001_reference_output_directory,
    time_backend_call,
    verify_perf001_provenance_binding,
    verify_required_benchmark_accelerator,
    write_perf001_cpu_evidence_document,
    write_perf001_evidence_document,
)
from radiosim.benchmarks.record import (
    BENCHMARK_SCHEMA_VERSION,
    MEMORY_SCALING_SCHEMA_VERSION,
    PERF001_BACKEND_RESOLUTION_SCHEMA_VERSION,
    PERF001_CPU_BACKENDS,
    PERF001_CPU_CANONICAL_INPUT_IDENTITIES,
    PERF001_CPU_WORKLOADS,
    PERF001_MEMORY_SCALING_SCHEMA_VERSION,
    PERF001_PROVENANCE_SCHEMA_VERSION,
    PERF001_RETRACING_SCHEMA_VERSION,
    PERF001_SCHEMA_VERSION,
    PERF001_SOLVER_MEMORY_SCHEMA_VERSION,
    PERF001_TARGET_KERNEL_PAIRS,
    PERF001_WORKLOAD_SCHEMA_VERSION,
    RETRACING_SCHEMA_VERSION,
    AcceleratorFacts,
    BackendResolutionRecord,
    BenchmarkDocument,
    BenchmarkRecord,
    BenchmarkRecordError,
    ContractionSignatureObservation,
    DeviceMemoryMeasurement,
    MeasurementContext,
    MemoryScalingRecord,
    MemoryScalingRecordV2,
    Perf001EvidenceDocument,
    Perf001Provenance,
    RetracingRecord,
    RetracingRecordV2,
    SolverMemoryRecord,
    WorkloadBenchmarkRecordV2,
    load_perf001_evidence_document,
    parse_perf001_evidence_document,
    records_are_complete,
    validate_perf001_cpu_evidence_document,
    write_benchmark_document,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "DEFAULT_STEADY_STATE_ITERATIONS",
    "MEMORY_SCALING_SCHEMA_VERSION",
    "PERF001_BACKEND_RESOLUTION_SCHEMA_VERSION",
    "PERF001_CPU_BACKENDS",
    "PERF001_CPU_CANONICAL_INPUT_IDENTITIES",
    "PERF001_CPU_WORKLOADS",
    "PERF001_MEMORY_SCALING_SCHEMA_VERSION",
    "PERF001_PROVENANCE_SCHEMA_VERSION",
    "PERF001_RETRACING_SCHEMA_VERSION",
    "PERF001_SCHEMA_VERSION",
    "PERF001_SOLVER_MEMORY_SCHEMA_VERSION",
    "PERF001_TARGET_KERNEL_PAIRS",
    "PERF001_WORKLOAD_SCHEMA_VERSION",
    "PERF001_REFERENCE_SHA256",
    "PERF001_REFERENCE_SOURCE_SHA",
    "RETRACING_SCHEMA_VERSION",
    "AcceleratorFacts",
    "BackendFacts",
    "BenchmarkBackendSelection",
    "BackendResolutionRecord",
    "BenchmarkDocument",
    "BenchmarkRecord",
    "BenchmarkRecordError",
    "ContractionSignatureObservation",
    "Deviation",
    "DeviceMemoryMeasurement",
    "EnvironmentFacts",
    "MeasurementContext",
    "MemoryScalingRecord",
    "MemoryScalingRecordV2",
    "Perf001EvidenceDocument",
    "Perf001Provenance",
    "Perf001ReferenceAuthentication",
    "RetracingRecord",
    "SCI004_BACKEND_PREDICATE_ID",
    "SCI004_BACKENDS",
    "SCI004_BENCHMARK_SCHEMA_VERSION",
    "SCI004_DIRECT_PREDICATE_ID",
    "SCI004_FIXTURE_IDS",
    "SCI004_PROVENANCE_SCHEMA_VERSION",
    "SCI004_SKY_REPRESENTATIONS",
    "SCI004_TOP_LEVEL_KEYS",
    "SCI004_WORKLOAD_INVENTORY",
    "SCI004_WORKLOAD_KEYS",
    "Sci004WorkloadIdentity",
    "sci004_claims_not_licensed",
    "sci004_reference_output_directory",
    "RetracingRecordV2",
    "SolverMemoryRecord",
    "TimingMeasurement",
    "WorkloadShape",
    "WorkloadBenchmarkRecordV2",
    "assemble_perf001_cpu_evidence_document",
    "benchmark_filename",
    "benchmark_output_directory",
    "authenticate_perf001_references",
    "benchmark_backend_selection",
    "build_record",
    "build_perf001_workload_record",
    "compare_to_reference",
    "describe_backend",
    "describe_environment",
    "describe_perf001_provenance",
    "measure_perf001_memory_scaling_pair",
    "measure_perf001_backend_resolution",
    "measure_perf001_solver_memory_pair",
    "measure_perf001_solver_retracing_pair",
    "measure_perf001_synthetic_retracing_pair",
    "measure_kernel_memory_scaling",
    "measure_retracing",
    "perf001_control_identity_sha256",
    "perf001_input_identity_sha256",
    "perf001_reference_output_directory",
    "records_are_complete",
    "load_perf001_evidence_document",
    "parse_perf001_evidence_document",
    "time_backend_call",
    "verify_perf001_provenance_binding",
    "verify_required_benchmark_accelerator",
    "write_benchmark_document",
    "write_perf001_evidence_document",
    "write_perf001_cpu_evidence_document",
    "validate_perf001_cpu_evidence_document",
]


# ---------------------------------------------------------------------------
# SCI-004 Section 11: the non-gating m-mode benchmark record surface
# ---------------------------------------------------------------------------

#: ``docs/development/sci004_mmode_design.md`` Section 11's exact top-level and
#: provenance schema literals.  The record "deliberately defines its own schema
#: rather than extending the accepted ``radiosim.benchmark.perf001.v1``
#: inventory: every SCI-004 row must join a frame certificate, scientific
#: identity, deterministic block schedule, and direct/backend comparison that the
#: PERF-001 record has no analogue for, and each schema remains governed by its
#: own strict validator."
SCI004_BENCHMARK_SCHEMA_VERSION: Final = "radiosim.benchmark.sci004.v1"
SCI004_PROVENANCE_SCHEMA_VERSION: Final = "radiosim.benchmark.sci004.provenance.v1"

#: Section 11's exact top-level key set.
SCI004_TOP_LEVEL_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "provenance",
    "workloads",
)

#: Section 11's fixed fixture and backend axes, in record order.
SCI004_FIXTURE_IDS: Final[tuple[str, ...]] = (
    "mmode_point_full_stokes",
    "mmode_healpix_full_stokes",
    "mmode_hybrid_full_stokes",
)
SCI004_BACKENDS: Final[tuple[str, ...]] = ("numpy", "jax", "dask")

#: The sky representation each fixture group carries.
SCI004_SKY_REPRESENTATIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "mmode_point_full_stokes": "point",
        "mmode_healpix_full_stokes": "healpix",
        "mmode_hybrid_full_stokes": "hybrid",
    }
)

#: Section 11's exact ordered workload-row key set.
SCI004_WORKLOAD_KEYS: Final[tuple[str, ...]] = (
    "workload_id",
    "comparison_group_id",
    "fixture_id",
    "input_identity_sha256",
    "frame_certificate_sha256",
    "scientific_sha256",
    "result_cube_sha256",
    "source_sha",
    "working_tree_clean",
    "backend",
    "backend_runtime",
    "device_kind",
    "precision",
    "accumulation_dtype",
    "result_dtype",
    "workers",
    "n_antennas",
    "n_baselines",
    "n_frequencies",
    "sidereal_samples",
    "lmax",
    "mmax",
    "quadrature_nside",
    "n_point_sources",
    "n_healpix_pixels",
    "sky_representation",
    "working_memory_bytes",
    "resolved_block_dimensions",
    "timings",
    "memory",
    "direct_comparison",
    "backend_comparison",
    "claims_not_licensed",
)

#: Section 11's two comparison predicate literals.
SCI004_DIRECT_PREDICATE_ID: Final = "sci004_two_tier_direct.v3"
SCI004_BACKEND_PREDICATE_ID: Final = "sci004_backend_complex128.v1"

#: Section 11's exact lexicographically sorted per-row claim array.  "A record is
#: evidence only of these nine measured CPU rows.  Timing values never gate CI
#: and license neither a speedup nor a memory/accelerator advantage."
_SCI004_CLAIMS_NOT_LICENSED: Final[tuple[str, ...]] = (
    "general_speedup",
    "gpu_or_accelerator_support",
    "perf001_evidence_or_closure",
    "performance_regression_gate",
    "unmeasured_workloads",
)


def sci004_claims_not_licensed() -> tuple[str, ...]:
    """Return Section 11's sorted, unique per-row claim array."""
    return _SCI004_CLAIMS_NOT_LICENSED


@dataclass(frozen=True, slots=True)
class Sci004WorkloadIdentity:
    """One row of Section 11's fixed nine-row ``v1`` inventory."""

    fixture_id: str
    backend: str
    workload_id: str
    comparison_group_id: str
    sky_representation: str
    device_kind: str = "cpu"
    precision: str = "standard"
    accumulation_dtype: str = "complex128"
    result_dtype: str = "complex128"


def _build_sci004_inventory() -> tuple[Sci004WorkloadIdentity, ...]:
    """Return the exact Cartesian product Section 11 fixes, in record order."""
    rows: list[Sci004WorkloadIdentity] = []
    for fixture in SCI004_FIXTURE_IDS:
        for backend in SCI004_BACKENDS:
            rows.append(
                Sci004WorkloadIdentity(
                    fixture_id=fixture,
                    backend=backend,
                    workload_id=f"{fixture}:{backend}:standard",
                    comparison_group_id=fixture,
                    sky_representation=SCI004_SKY_REPRESENTATIONS[fixture],
                )
            )
    return tuple(rows)


#: Section 11: "the array has exactly nine rows", fixture-major then backend.
SCI004_WORKLOAD_INVENTORY: Final[tuple[Sci004WorkloadIdentity, ...]] = (
    _build_sci004_inventory()
)


def sci004_reference_output_directory(repository_root: Path | None = None) -> Path:
    """Return Section 11's fixed ``output/benchmarks/reference/sci004`` directory.

    The records that live there are non-gating: Section 11 keeps them out of CI
    entirely, and ``PERF-001`` continues to govern every performance statement.
    """
    root = repository_root or Path(__file__).resolve().parents[3]
    return root / "output" / "benchmarks" / "reference" / "sci004"
