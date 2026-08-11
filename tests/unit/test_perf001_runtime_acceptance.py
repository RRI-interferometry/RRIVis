"""Strict, standalone PERF-001 v2 evidence contracts.

The historical Tier 6 benchmark document is frozen.  PERF-001 therefore uses
new row types whose validation is deliberately structural: malformed evidence
must fail before it can be written or reviewed.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from radiosim.benchmarks import (
    PERF001_BACKEND_RESOLUTION_SCHEMA_VERSION,
    PERF001_MEMORY_SCALING_SCHEMA_VERSION,
    PERF001_PROVENANCE_SCHEMA_VERSION,
    PERF001_RETRACING_SCHEMA_VERSION,
    PERF001_SCHEMA_VERSION,
    PERF001_SOLVER_MEMORY_SCHEMA_VERSION,
    PERF001_TARGET_KERNEL_PAIRS,
    PERF001_WORKLOAD_SCHEMA_VERSION,
    AcceleratorFacts,
    BackendResolutionRecord,
    BenchmarkRecordError,
    ContractionSignatureObservation,
    DeviceMemoryMeasurement,
    MeasurementContext,
    MemoryScalingRecordV2,
    Perf001EvidenceDocument,
    Perf001Provenance,
    RetracingRecordV2,
    SolverMemoryRecord,
    WorkloadBenchmarkRecordV2,
    benchmark_filename,
    write_perf001_evidence_document,
)


def _provenance(**updates: Any) -> Perf001Provenance:
    values: dict[str, Any] = {
        "schema_version": PERF001_PROVENANCE_SCHEMA_VERSION,
        "recorded_at_utc": "2026-08-11T00:00:00+00:00",
        "radiosim_version": "0.2.0",
        "git_sha": "a" * 40,
        "working_tree_clean": True,
        "platform": "macOS-15-arm64",
        "machine": "arm64",
        "cpu_model": "Apple M1 Max",
        "cpu_count_logical": 10,
        "python_version": "3.11.13",
        "numpy_version": "2.3.2",
        "jax_version": "0.10.2",
        "jaxlib_version": "0.10.2",
        "dask_version": "2025.7.0",
        "pixi_environment": "default",
        "pixi_lock_sha256": "b" * 64,
    }
    values.update(updates)
    return Perf001Provenance.create(**values)


def _context(
    *,
    policy_id: str,
    identity: str = "c" * 64,
    backend: str = "numpy",
    actual_backend: str | None = None,
    device_kind: str = "cpu",
) -> MeasurementContext:
    resolved_backend = actual_backend or {
        "numpy": "numpy-cpu",
        "jax": "jax-cpu-cpu",
        "gpu": "jax-gpu-gpu",
    }.get(backend, backend)
    return MeasurementContext.create(
        backend_requested=backend,
        backend_actual=resolved_backend,
        backend_version="2.3.2" if resolved_backend.startswith("numpy") else "0.10.2",
        device_kind=device_kind,
        compilation_used=resolved_backend == "gpu"
        or resolved_backend.startswith("jax"),
        precision_preset="standard",
        precision_default="float64",
        precision_accumulation="float64",
        precision_output="float64",
        result_dtype="complex128",
        policy_id=policy_id,
        input_identity_sha256=identity,
        measurement_limitations=("tracemalloc excludes native allocations",),
    )


def _memory_row(
    state: str, *, provenance: Perf001Provenance | None = None
) -> MemoryScalingRecordV2:
    production = state == "chunked_production"
    return MemoryScalingRecordV2.create(
        schema_version=PERF001_MEMORY_SCALING_SCHEMA_VERSION,
        provenance=provenance or _provenance(),
        context=_context(
            policy_id="target_kernel_pairs_131072_v1"
            if production
            else "unbounded_reference_v1"
        ),
        comparison_id="memory-001",
        implementation_state=state,
        measurement_scope=("contraction_wrapper_python_heap_including_output_assembly"),
        allocator="python_heap_tracemalloc",
        includes_backend_native_allocations=False,
        inputs_preallocated=True,
        includes_solver_input_construction=False,
        includes_output_reassembly=True,
        logical_n_baselines=4,
        logical_n_sources=3,
        logical_pair_count=12,
        kernel_n_sources=3,
        target_kernel_pairs=PERF001_TARGET_KERNEL_PAIRS if production else None,
        kernel_baseline_chunks=(4,),
        kernel_pair_counts=(12,),
        max_kernel_pair_count=12,
        synthetic_input_bytes_excluded=4096,
        peak_host_bytes=8192 if production else 16384,
        notes="matched synthetic contraction fixture",
    )


def _solver_memory_row(
    state: str,
    *,
    provenance: Perf001Provenance | None = None,
    solver: str = "point",
    sky_representation: str = "point_sources",
) -> SolverMemoryRecord:
    production = state == "bucketed_production"
    return SolverMemoryRecord.create(
        schema_version=PERF001_SOLVER_MEMORY_SCHEMA_VERSION,
        provenance=provenance or _provenance(),
        context=_context(
            policy_id="pow2_compiled_v1" if production else "identity_reference_v1",
            backend="jax",
        ),
        comparison_id=f"solver-memory-{solver}-001",
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
        logical_n_baselines=4,
        logical_source_counts=(3, 5),
        kernel_source_counts=(4, 8) if production else (3, 5),
        n_times=1,
        n_frequencies=2,
        target_kernel_pairs=PERF001_TARGET_KERNEL_PAIRS,
        bucket_policy="pow2_compiled_v1" if production else "identity_reference_v1",
        peak_host_bytes=32768,
        notes="direct solver steps",
    )


def _signature(n_sources: int, *, call_count: int = 2):
    return ContractionSignatureObservation.create(
        jones_p_shape=(4, n_sources, 2, 2),
        jones_q_shape=(4, n_sources, 2, 2),
        coherency_shape=(n_sources, 2, 2),
        phase_shape=(4, n_sources),
        envelope_shape=None,
        stokes_i_shape=None,
        jones_p_dtype="complex128",
        jones_q_dtype="complex128",
        coherency_dtype="complex128",
        phase_dtype="complex128",
        envelope_dtype=None,
        stokes_i_dtype=None,
        call_count=call_count,
        first_call_seconds=0.02,
        minimum_repeat_call_seconds=0.01,
    )


def _retrace_row(
    state: str,
    *,
    provenance: Perf001Provenance | None = None,
    solver: str = "point",
    sky_representation: str = "point_sources",
) -> RetracingRecordV2:
    production = state == "bucketed_production"
    logical_counts = (3, 4, 5, 8, 3, 4, 5, 8)
    kernel_counts = (4, 4, 8, 8, 4, 4, 8, 8) if production else logical_counts
    signatures = (
        (_signature(4, call_count=4), _signature(8, call_count=4))
        if production
        else tuple(_signature(count) for count in (3, 4, 5, 8))
    )
    step_seconds = (0.01,) * len(logical_counts)
    return RetracingRecordV2.create(
        schema_version=PERF001_RETRACING_SCHEMA_VERSION,
        provenance=provenance or _provenance(),
        context=_context(
            policy_id="pow2_compiled_v1" if production else "identity_reference_v1",
            backend="jax",
        ),
        comparison_id=f"retrace-{solver}-001",
        implementation_state=state,
        measurement_scope=f"complete_{solver}_solver_step",
        solver=solver,
        sky_representation=sky_representation,
        bucket_policy="pow2_compiled_v1" if production else "identity_reference_v1",
        padding_location="early_host" if production else "none",
        logical_source_counts=logical_counts,
        kernel_source_counts=kernel_counts,
        distinct_logical_source_counts=4,
        distinct_kernel_source_counts=2 if production else 4,
        observed_signatures=signatures,
        distinct_signature_count=len(signatures),
        leaf_call_count=8,
        scope_step_seconds=step_seconds,
        scope_total_seconds=sum(step_seconds),
        max_first_to_repeat_ratio=2.0,
        retrace_overhead_seconds=0.02 if production else 0.04,
        notes="compile-spy observations",
    )


def _backend_resolution(
    *,
    provenance: Perf001Provenance | None = None,
    operation: str = "get_backend_auto",
    requested_backend: str = "auto",
    resolved_backend: str = "numpy-cpu",
    comparison_id: str = "auto-direct-001",
) -> BackendResolutionRecord:
    return BackendResolutionRecord.create(
        schema_version=PERF001_BACKEND_RESOLUTION_SCHEMA_VERSION,
        provenance=provenance or _provenance(),
        context=_context(
            policy_id="deterministic_auto_numpy_v1",
            backend=requested_backend,
            actual_backend=resolved_backend,
        ),
        comparison_id=comparison_id,
        implementation_state="production",
        operation=operation,
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        discovery_policy="no_optional_backend_imports",
        fresh_process_samples=3,
        cold_seconds=(0.10, 0.20, 0.15),
        minimum_seconds=0.10,
        median_seconds=0.15,
        maximum_seconds=0.20,
        jax_distribution_installed=True,
        jax_in_sys_modules_before=False,
        jax_in_sys_modules_after=False,
        jaxlib_in_sys_modules_before=False,
        jaxlib_in_sys_modules_after=False,
        notes="fresh subprocesses",
    )


def _accelerator(**updates: Any) -> AcceleratorFacts:
    values: dict[str, Any] = {
        "vendor": "NVIDIA",
        "model": "Example GPU",
        "runtime": "CUDA 13",
        "driver_version": "580.00",
        "compute_capability": "8.0",
        "total_memory_bytes": 16_000_000_000,
        "pci_bus_id": "0000:01:00.0",
        "device_uuid_sha256": "d" * 64,
        "jax_device_id": 0,
        "jax_device_kind": "gpu",
        "visible_device_count": 1,
        "wheel_versions": {
            "jax": "0.10.2",
            "jaxlib": "0.10.2",
            "jax-cuda13-plugin": "0.10.2",
            "jax-cuda13-pjrt": "0.10.2",
        },
        "allocator_environment": {},
    }
    values.update(updates)
    return AcceleratorFacts.create(**values)


def _device_memory(**updates: Any) -> DeviceMemoryMeasurement:
    values: dict[str, Any] = {
        "method": "nvidia-smi process sampler",
        "sampling_scope": "dedicated untimed workload iteration",
        "sample_interval_seconds": 0.01,
        "sample_count": 10,
        "total_bytes": 16_000_000_000,
        "used_bytes_before": 1_000,
        "free_bytes_before": 15_999_999_000,
        "used_bytes_after_setup": 2_000,
        "free_bytes_after_setup": 15_999_998_000,
        "peak_observed_used_bytes": 4_000,
        "used_bytes_after_transfer": 3_000,
        "free_bytes_after_transfer": 15_999_997_000,
        "raw_jax_memory_stats": None,
        "limitations": "sampled peak, not allocator-exact",
    }
    values.update(updates)
    return DeviceMemoryMeasurement.create(**values)


def _workload(
    *,
    provenance: Perf001Provenance | None = None,
    context: MeasurementContext | None = None,
    accelerator: AcceleratorFacts | None = None,
    device_memory: DeviceMemoryMeasurement | None = None,
) -> WorkloadBenchmarkRecordV2:
    return WorkloadBenchmarkRecordV2.create(
        schema_version=PERF001_WORKLOAD_SCHEMA_VERSION,
        provenance=provenance or _provenance(),
        context=context or _context(policy_id="cpu_production_v1"),
        accelerator=accelerator,
        device_memory=device_memory,
        workload="point_polarized_2times",
        n_antennas=5,
        n_baselines=10,
        n_point_sources=2,
        n_healpix_pixels=0,
        n_times=2,
        n_frequencies=2,
        sky_representation="point_sources",
        solver_workers=1,
        loader_max_workers=1,
        setup_seconds=0.5,
        compile_seconds=0.4,
        steady_state_median_seconds=0.1,
        steady_state_min_seconds=0.09,
        steady_state_max_seconds=0.12,
        steady_state_iterations=5,
        host_transfer_seconds=0.001,
        peak_host_bytes=1024,
        host_memory_method="python_heap_tracemalloc",
        reference_backend="numpy",
        max_absolute_deviation=0.0,
        max_relative_deviation=0.0,
        tolerance_rtol=1e-12,
        tolerance_atol=1e-12,
        within_tolerance=True,
        unmeasured=("gpu", "tpu", "distributed"),
        notes="complete CPU workload",
    )


def _document() -> Perf001EvidenceDocument:
    provenance = _provenance()
    return Perf001EvidenceDocument.create(
        schema_version=PERF001_SCHEMA_VERSION,
        workload_benchmarks=(_workload(provenance=provenance),),
        memory_scaling=(
            _memory_row("unchunked_reference", provenance=provenance),
            _memory_row("chunked_production", provenance=provenance),
        ),
        solver_memory=(
            _solver_memory_row("unbucketed_reference", provenance=provenance),
            _solver_memory_row("bucketed_production", provenance=provenance),
            _solver_memory_row(
                "unbucketed_reference",
                provenance=provenance,
                solver="healpix",
                sky_representation="healpix",
            ),
            _solver_memory_row(
                "bucketed_production",
                provenance=provenance,
                solver="healpix",
                sky_representation="healpix",
            ),
        ),
        retracing=(
            _retrace_row(
                "unbucketed_reference",
                provenance=provenance,
                solver="synthetic_wrapper",
                sky_representation="synthetic_contraction",
            ),
            _retrace_row(
                "bucketed_production",
                provenance=provenance,
                solver="synthetic_wrapper",
                sky_representation="synthetic_contraction",
            ),
            _retrace_row("unbucketed_reference", provenance=provenance),
            _retrace_row("bucketed_production", provenance=provenance),
            _retrace_row(
                "unbucketed_reference",
                provenance=provenance,
                solver="healpix",
                sky_representation="healpix",
            ),
            _retrace_row(
                "bucketed_production",
                provenance=provenance,
                solver="healpix",
                sky_representation="healpix",
            ),
        ),
        backend_resolution=(
            _backend_resolution(provenance=provenance),
            _backend_resolution(
                provenance=provenance,
                operation="get_device_resources_default",
                requested_backend="default",
                resolved_backend="platform-tools",
                comparison_id="device-resources-default-001",
            ),
            _backend_resolution(
                provenance=provenance,
                operation="simulator_setup_auto",
                comparison_id="simulator-setup-auto-001",
            ),
        ),
    )


def test_perf001_types_have_the_exact_documented_fields() -> None:
    expected = {
        Perf001EvidenceDocument: (
            "schema_version",
            "workload_benchmarks",
            "memory_scaling",
            "solver_memory",
            "retracing",
            "backend_resolution",
        ),
        Perf001Provenance: (
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
        ),
        MeasurementContext: (
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
        ),
        MemoryScalingRecordV2: (
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
        ),
        SolverMemoryRecord: (
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
        ),
        ContractionSignatureObservation: (
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
        ),
        RetracingRecordV2: (
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
        ),
        BackendResolutionRecord: (
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
        ),
        AcceleratorFacts: (
            "vendor",
            "model",
            "runtime",
            "driver_version",
            "compute_capability",
            "total_memory_bytes",
            "pci_bus_id",
            "device_uuid_sha256",
            "jax_device_id",
            "jax_device_kind",
            "visible_device_count",
            "wheel_versions",
            "allocator_environment",
        ),
        DeviceMemoryMeasurement: (
            "method",
            "sampling_scope",
            "sample_interval_seconds",
            "sample_count",
            "total_bytes",
            "used_bytes_before",
            "free_bytes_before",
            "used_bytes_after_setup",
            "free_bytes_after_setup",
            "peak_observed_used_bytes",
            "used_bytes_after_transfer",
            "free_bytes_after_transfer",
            "raw_jax_memory_stats",
            "limitations",
        ),
        WorkloadBenchmarkRecordV2: (
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
        ),
    }
    for record_type, names in expected.items():
        assert tuple(field.name for field in fields(record_type)) == names


def test_every_perf001_create_rejects_missing_and_unknown_fields() -> None:
    instances = (
        _provenance(),
        _context(policy_id="policy"),
        _memory_row("unchunked_reference"),
        _solver_memory_row("unbucketed_reference"),
        _signature(3),
        _retrace_row("unbucketed_reference"),
        _backend_resolution(),
        _accelerator(),
        _device_memory(),
        _workload(),
        _document(),
    )
    for instance in instances:
        values = {
            field.name: getattr(instance, field.name) for field in fields(instance)
        }
        missing = next(iter(values))
        del values[missing]
        with pytest.raises(BenchmarkRecordError, match=missing):
            type(instance).create(**values)
        values[missing] = getattr(instance, missing)
        values["unknown_evidence"] = "must fail"
        with pytest.raises(BenchmarkRecordError, match="unknown_evidence"):
            type(instance).create(**values)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"schema_version": "wrong"}, "schema_version"),
        ({"git_sha": "A" * 40}, "git_sha"),
        ({"git_sha": "0" * 40}, "git_sha"),
        ({"working_tree_clean": False}, "working_tree_clean"),
        ({"cpu_count_logical": True}, "cpu_count_logical"),
        ({"pixi_lock_sha256": "B" * 64}, "pixi_lock_sha256"),
    ],
)
def test_provenance_rejects_unknown_dirty_or_non_json_values(
    updates: dict[str, Any], match: str
) -> None:
    with pytest.raises(BenchmarkRecordError, match=match):
        _provenance(**updates)


def test_provenance_schema_is_pure_and_generation_must_bind_live_digests() -> None:
    """The Phase 7.6 generator, not this value type, verifies checkout bytes."""
    provenance = _provenance(git_sha="1" * 40, pixi_lock_sha256="2" * 64)

    assert provenance.git_sha == "1" * 40
    assert provenance.pixi_lock_sha256 == "2" * 64


def test_context_requires_a_lowercase_input_identity_and_ordered_limitations() -> None:
    with pytest.raises(BenchmarkRecordError, match="input_identity_sha256"):
        _context(policy_id="policy", identity="C" * 64)
    with pytest.raises(BenchmarkRecordError, match="measurement_limitations"):
        replace(
            _context(policy_id="policy"),
            measurement_limitations={"unordered"},
        )


def test_memory_scaling_validates_derived_counts_and_implementation_contract() -> None:
    with pytest.raises(BenchmarkRecordError, match="logical_pair_count"):
        replace(_memory_row("unchunked_reference"), logical_pair_count=13)
    with pytest.raises(BenchmarkRecordError, match="kernel_baseline_chunks"):
        replace(_memory_row("chunked_production"), kernel_baseline_chunks=(3,))
    with pytest.raises(BenchmarkRecordError, match="kernel_pair_counts"):
        replace(_memory_row("chunked_production"), kernel_pair_counts=(11,))
    with pytest.raises(BenchmarkRecordError, match="target_kernel_pairs"):
        replace(_memory_row("chunked_production"), target_kernel_pairs=None)
    with pytest.raises(BenchmarkRecordError, match="context.policy_id"):
        replace(
            _memory_row("chunked_production"),
            context=_context(policy_id="unbounded_reference_v1"),
        )
    with pytest.raises(BenchmarkRecordError, match="inputs_preallocated"):
        replace(_memory_row("chunked_production"), inputs_preallocated=False)
    with pytest.raises(BenchmarkRecordError, match="stable production chunk"):
        replace(
            _memory_row("chunked_production"),
            kernel_baseline_chunks=(2, 2),
            kernel_pair_counts=(6, 6),
            max_kernel_pair_count=6,
        )


def test_signature_requires_shape_dtype_nullability_to_be_paired() -> None:
    with pytest.raises(BenchmarkRecordError, match="envelope"):
        replace(_signature(3), envelope_dtype="float64")
    with pytest.raises(BenchmarkRecordError, match="jones_p"):
        replace(_signature(3), jones_p_shape=None)


def test_solver_memory_scope_flags_cannot_contradict_the_measurement() -> None:
    with pytest.raises(BenchmarkRecordError, match="includes_simulator_setup"):
        replace(
            _solver_memory_row("bucketed_production"),
            includes_simulator_setup=True,
        )
    with pytest.raises(BenchmarkRecordError, match="measurement_scope"):
        replace(
            _solver_memory_row("bucketed_production"),
            measurement_scope="whole_public_simulator",
        )


def test_retracing_validates_shape_counts_leaf_calls_and_scope_total() -> None:
    record = _retrace_row("bucketed_production")
    with pytest.raises(BenchmarkRecordError, match="distinct_kernel_source_counts"):
        replace(record, distinct_kernel_source_counts=3)
    with pytest.raises(BenchmarkRecordError, match="leaf_call_count"):
        replace(record, leaf_call_count=7)
    with pytest.raises(BenchmarkRecordError, match="scope_total_seconds"):
        replace(record, scope_total_seconds=1.0)
    with pytest.raises(BenchmarkRecordError, match="retrace_overhead_seconds"):
        replace(record, retrace_overhead_seconds=0.03)
    with pytest.raises(BenchmarkRecordError, match="padding_location"):
        replace(record, padding_location="late_jax")
    undercounted = tuple(
        replace(observation, call_count=2) for observation in record.observed_signatures
    )
    with pytest.raises(BenchmarkRecordError, match="logical leaf steps"):
        replace(record, observed_signatures=undercounted, leaf_call_count=4)


def test_zero_visible_retracing_steps_never_claim_a_leaf_signature() -> None:
    reference = _retrace_row("unbucketed_reference")
    zero_step_values = {
        "logical_source_counts": (0, 0),
        "kernel_source_counts": (0, 0),
        "distinct_logical_source_counts": 1,
        "distinct_kernel_source_counts": 1,
        "scope_step_seconds": (0.01, 0.01),
        "scope_total_seconds": 0.02,
    }
    with pytest.raises(BenchmarkRecordError, match="zero-visible"):
        replace(
            reference,
            **zero_step_values,
            observed_signatures=(_signature(0),),
            distinct_signature_count=1,
            leaf_call_count=2,
            max_first_to_repeat_ratio=2.0,
            retrace_overhead_seconds=0.01,
        )

    zero_visible = replace(
        reference,
        **zero_step_values,
        observed_signatures=(),
        distinct_signature_count=0,
        leaf_call_count=0,
        max_first_to_repeat_ratio=0.0,
        retrace_overhead_seconds=0.0,
    )
    assert zero_visible.observed_signatures == ()
    assert zero_visible.leaf_call_count == 0


def test_backend_resolution_derives_sample_count_and_summary_statistics() -> None:
    record = _backend_resolution()
    with pytest.raises(BenchmarkRecordError, match="fresh_process_samples"):
        replace(record, fresh_process_samples=4)
    with pytest.raises(BenchmarkRecordError, match="median_seconds"):
        replace(record, median_seconds=0.19)


def test_workload_cpu_and_gpu_fields_are_paired() -> None:
    with pytest.raises(BenchmarkRecordError, match="accelerator"):
        _workload(accelerator=_accelerator())
    with pytest.raises(BenchmarkRecordError, match="steady_state_iterations"):
        replace(_workload(), steady_state_iterations=4)

    gpu = replace(
        _workload(),
        context=_context(
            policy_id="gpu_production_v1",
            backend="gpu",
            actual_backend="jax-gpu-gpu",
            device_kind="gpu",
        ),
        accelerator=_accelerator(),
        device_memory=_device_memory(),
        unmeasured=("tpu", "distributed"),
    )
    assert gpu.accelerator is not None
    assert gpu.device_memory is not None

    with pytest.raises(BenchmarkRecordError, match="total_bytes"):
        replace(gpu, device_memory=_device_memory(total_bytes=15_000_000_000))
    for impossible_actual, backend_version in (
        ("numpy-cpu", "2.3.2"),
        ("jax-cpu-cpu", "0.10.2"),
        ("jax-tpu-tpu", "0.10.2"),
    ):
        with pytest.raises(BenchmarkRecordError, match="context.backend_actual"):
            replace(
                gpu,
                context=replace(
                    gpu.context,
                    backend_actual=impossible_actual,
                    backend_version=backend_version,
                ),
            )


def test_device_memory_raw_stats_are_strict_recursive_json() -> None:
    with pytest.raises(BenchmarkRecordError, match="finite"):
        _device_memory(raw_jax_memory_stats={"nested": {"peak": float("nan")}})
    with pytest.raises(BenchmarkRecordError, match="keys"):
        _device_memory(raw_jax_memory_stats={"nested": {1: "not-a-string-key"}})


def test_document_rejects_heterogeneous_provenance_and_incomplete_pairs() -> None:
    document = _document()
    with pytest.raises(BenchmarkRecordError, match="heterogeneous provenance"):
        replace(
            document,
            workload_benchmarks=(
                _workload(provenance=_provenance(recorded_at_utc="2026-08-12T00:00Z")),
            ),
        )
    with pytest.raises(BenchmarkRecordError, match="memory_scaling.*pair"):
        replace(document, memory_scaling=document.memory_scaling[:1])
    with pytest.raises(BenchmarkRecordError, match="workload_benchmarks"):
        replace(document, workload_benchmarks=())
    with pytest.raises(BenchmarkRecordError, match="required operation"):
        replace(document, backend_resolution=document.backend_resolution[:1])
    with pytest.raises(BenchmarkRecordError, match="required real solver paths"):
        replace(document, solver_memory=document.solver_memory[:2])
    with pytest.raises(BenchmarkRecordError, match="synthetic and real solver paths"):
        replace(document, retracing=document.retracing[2:4])


def test_document_pairs_require_identical_logical_input_identity() -> None:
    document = _document()
    production = document.memory_scaling[1]
    mismatched = replace(
        production,
        context=replace(production.context, input_identity_sha256="e" * 64),
    )
    with pytest.raises(BenchmarkRecordError, match="input_identity_sha256"):
        replace(document, memory_scaling=(document.memory_scaling[0], mismatched))

    mismatched_backend = replace(
        production,
        context=replace(
            production.context,
            backend_requested="jax",
            backend_actual="jax",
            backend_version="0.10.2",
            compilation_used=True,
        ),
    )
    with pytest.raises(BenchmarkRecordError, match="context field backend_requested"):
        replace(
            document,
            memory_scaling=(document.memory_scaling[0], mismatched_backend),
        )


def test_document_serializes_recursively_to_exact_json_arrays() -> None:
    payload = _document().to_json_safe()

    assert tuple(payload) == (
        "schema_version",
        "workload_benchmarks",
        "memory_scaling",
        "solver_memory",
        "retracing",
        "backend_resolution",
    )
    assert payload["schema_version"] == PERF001_SCHEMA_VERSION
    assert isinstance(payload["memory_scaling"], list)
    assert payload["memory_scaling"][0]["kernel_baseline_chunks"] == [4]
    assert payload["retracing"][0]["observed_signatures"][0]["envelope_shape"] is None


def test_perf001_writer_uses_only_the_exact_namespaced_non_overwriting_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import radiosim.benchmarks.harness as harness

    monkeypatch.setattr(
        harness,
        "verify_perf001_provenance_binding",
        lambda _provenance, *, repository_root: None,
    )
    document = _document()
    filename = benchmark_filename(datetime(2026, 8, 11, tzinfo=UTC))

    with pytest.raises(BenchmarkRecordError, match="output directory"):
        write_perf001_evidence_document(
            document,
            filename=filename,
            repository_root=tmp_path,
            directory=tmp_path / "output/benchmarks/reference",
        )

    written = write_perf001_evidence_document(
        document,
        filename=filename,
        repository_root=tmp_path,
    )
    assert written.parent == tmp_path / "output/benchmarks/reference/perf001"
    assert json.loads(written.read_text(encoding="utf-8")) == (document.to_json_safe())

    with pytest.raises(BenchmarkRecordError, match="already exists"):
        write_perf001_evidence_document(
            document,
            filename=filename,
            repository_root=tmp_path,
        )


def test_perf001_writer_never_leaves_a_partial_final_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import radiosim.benchmarks.harness as harness

    monkeypatch.setattr(
        harness,
        "verify_perf001_provenance_binding",
        lambda _provenance, *, repository_root: None,
    )

    def fail_after_write(_descriptor: int) -> None:
        raise OSError("injected durability failure")

    monkeypatch.setattr(harness.os, "fsync", fail_after_write)
    filename = benchmark_filename(datetime(2026, 8, 11, tzinfo=UTC))
    expected_directory = tmp_path / "output/benchmarks/reference/perf001"
    expected_target = expected_directory / filename

    with pytest.raises(BenchmarkRecordError, match="published atomically"):
        write_perf001_evidence_document(
            _document(),
            filename=filename,
            repository_root=tmp_path,
        )

    assert not expected_target.exists()
    assert list(expected_directory.iterdir()) == []


@pytest.mark.parametrize(
    "component",
    [
        "output",
        "output/benchmarks",
        "output/benchmarks/reference",
        "output/benchmarks/reference/perf001",
    ],
)
def test_perf001_writer_rejects_symlinked_destination_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
) -> None:
    import radiosim.benchmarks.harness as harness

    monkeypatch.setattr(
        harness,
        "verify_perf001_provenance_binding",
        lambda _provenance, *, repository_root: None,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = tmp_path / component
    symlink.parent.mkdir(parents=True, exist_ok=True)
    symlink.symlink_to(outside, target_is_directory=True)

    with pytest.raises(BenchmarkRecordError, match="symlink"):
        write_perf001_evidence_document(
            _document(),
            filename=benchmark_filename(datetime(2026, 8, 11, tzinfo=UTC)),
            repository_root=tmp_path,
        )

    assert list(outside.iterdir()) == []


def test_perf001_writer_flushes_file_then_unlinks_temp_then_flushes_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import radiosim.benchmarks.harness as harness

    monkeypatch.setattr(
        harness,
        "verify_perf001_provenance_binding",
        lambda _provenance, *, repository_root: None,
    )
    events: list[str] = []
    real_fsync = harness.os.fsync
    real_link = harness.os.link
    real_unlink = harness.os.unlink

    def observed_fsync(descriptor: int) -> None:
        kind = (
            "directory_fsync"
            if stat.S_ISDIR(os.fstat(descriptor).st_mode)
            else "file_fsync"
        )
        events.append(kind)
        real_fsync(descriptor)

    def observed_link(*args: object, **kwargs: object) -> None:
        events.append("link")
        real_link(*args, **kwargs)

    def observed_unlink(*args: object, **kwargs: object) -> None:
        events.append("temp_unlink")
        real_unlink(*args, **kwargs)

    monkeypatch.setattr(harness.os, "fsync", observed_fsync)
    monkeypatch.setattr(harness.os, "link", observed_link)
    monkeypatch.setattr(harness.os, "unlink", observed_unlink)
    write_perf001_evidence_document(
        _document(),
        filename=benchmark_filename(datetime(2026, 8, 11, tzinfo=UTC)),
        repository_root=tmp_path,
    )

    assert events == ["file_fsync", "link", "temp_unlink", "directory_fsync"]


def test_perf001_writer_cleans_temp_if_hard_link_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import radiosim.benchmarks.harness as harness

    monkeypatch.setattr(
        harness,
        "verify_perf001_provenance_binding",
        lambda _provenance, *, repository_root: None,
    )

    def fail_link(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected link failure")

    monkeypatch.setattr(harness.os, "link", fail_link)
    filename = benchmark_filename(datetime(2026, 8, 11, tzinfo=UTC))
    expected_directory = tmp_path / "output/benchmarks/reference/perf001"

    with pytest.raises(BenchmarkRecordError, match="published atomically"):
        write_perf001_evidence_document(
            _document(),
            filename=filename,
            repository_root=tmp_path,
        )

    assert list(expected_directory.iterdir()) == []


def test_perf001_writer_reports_directory_fsync_failure_after_temp_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import radiosim.benchmarks.harness as harness

    monkeypatch.setattr(
        harness,
        "verify_perf001_provenance_binding",
        lambda _provenance, *, repository_root: None,
    )
    real_fsync = harness.os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory durability failure")
        real_fsync(descriptor)

    monkeypatch.setattr(harness.os, "fsync", fail_directory_fsync)
    filename = benchmark_filename(datetime(2026, 8, 11, tzinfo=UTC))
    expected_directory = tmp_path / "output/benchmarks/reference/perf001"
    expected_target = expected_directory / filename

    with pytest.raises(BenchmarkRecordError, match="directory durability"):
        write_perf001_evidence_document(
            _document(),
            filename=filename,
            repository_root=tmp_path,
        )

    assert expected_target.is_file()
    assert not any(path.name.endswith(".tmp") for path in expected_directory.iterdir())
