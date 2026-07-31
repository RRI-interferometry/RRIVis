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

from radiosim.benchmarks.harness import (
    DEFAULT_STEADY_STATE_ITERATIONS,
    BackendFacts,
    Deviation,
    EnvironmentFacts,
    TimingMeasurement,
    WorkloadShape,
    benchmark_filename,
    benchmark_output_directory,
    build_record,
    compare_to_reference,
    describe_backend,
    describe_environment,
    measure_kernel_memory_scaling,
    measure_retracing,
    time_backend_call,
)
from radiosim.benchmarks.record import (
    BENCHMARK_SCHEMA_VERSION,
    MEMORY_SCALING_SCHEMA_VERSION,
    RETRACING_SCHEMA_VERSION,
    BenchmarkDocument,
    BenchmarkRecord,
    BenchmarkRecordError,
    MemoryScalingRecord,
    RetracingRecord,
    records_are_complete,
    write_benchmark_document,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "DEFAULT_STEADY_STATE_ITERATIONS",
    "MEMORY_SCALING_SCHEMA_VERSION",
    "RETRACING_SCHEMA_VERSION",
    "BackendFacts",
    "BenchmarkDocument",
    "BenchmarkRecord",
    "BenchmarkRecordError",
    "Deviation",
    "EnvironmentFacts",
    "MemoryScalingRecord",
    "RetracingRecord",
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
    "records_are_complete",
    "time_backend_call",
    "write_benchmark_document",
]
