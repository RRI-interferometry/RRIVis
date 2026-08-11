"""Section 27 rows P1-P3: the benchmark record schema and the harness itself.

These tests are deliberately **unmarked**. Section 22.1 puts the harness in
``src/`` precisely so that its schema and its timing discipline are covered by
the fast suite, which always runs, while the measurements it drives stay behind
the ``performance``/``slow`` markers that CI never selects. A harness that is
only exercised by the benchmarks would be trusted exactly when it is least
observed.

- **P1** -- a record missing any mandatory field raises ``BenchmarkRecordError``.
- **P2** -- every Tier 6 record has ``accelerator == "none"`` and lists ``"gpu"``
  in ``unmeasured``. Tier 6 exercises no accelerator (Section 4), so a record
  that read otherwise would be the exact failure mode the tier exists to remove.
- **P3** -- the harness produces a valid record for a tiny workload on the NumPy
  backend.
"""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.benchmarks import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkDocument,
    BenchmarkRecord,
    BenchmarkRecordError,
    MemoryScalingRecord,
    RetracingRecord,
    WorkloadShape,
    benchmark_filename,
    benchmark_output_directory,
    build_record,
    compare_to_reference,
    describe_backend,
    describe_environment,
    measure_kernel_memory_scaling,
    measure_retracing,
    records_are_complete,
    time_backend_call,
    write_benchmark_document,
)
from radiosim.core.contraction import baseline_contraction_for
from radiosim.core.precision import PrecisionConfig


def _tiny_kernel_call(backend: Any):
    """A tiny but real backend workload: one baseline-batched contraction."""
    kernel = baseline_contraction_for(backend)
    rng = np.random.default_rng(6109)
    shape = (3, 4, 2, 2)
    jones_p = backend.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )
    jones_q = backend.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )
    coherency = backend.asarray(
        rng.standard_normal((4, 2, 2)) + 1j * rng.standard_normal((4, 2, 2))
    )
    phase = backend.asarray(np.exp(1j * rng.standard_normal((3, 4))))
    envelope = backend.asarray(np.ones((3, 4)))

    def call():
        return kernel(jones_p, jones_q, coherency, phase, envelope, None)

    return call


def _complete_values(**overrides: Any) -> dict[str, Any]:
    """A full, honest Section 23 field mapping, for mutation by the tests."""
    values: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "recorded_at_utc": "2026-07-31T00:00:00+00:00",
        "radiosim_version": "0.2.0",
        "git_sha": "0" * 40,
        "platform": "macOS-15-arm64",
        "cpu_model": "Apple M1 Max",
        "cpu_count_logical": 10,
        "accelerator": "none",
        "accelerator_driver": None,
        "backend_requested": "numpy",
        "backend_actual": "numpy",
        "backend_version": "2.0.0",
        "device_kind": "cpu",
        "compilation_used": False,
        "precision_preset": "standard",
        "precision_default": "float64",
        "precision_accumulation": "float64",
        "precision_output": "float64",
        "result_dtype": "complex128",
        "workload": "point_polarized_2times",
        "n_antennas": 5,
        "n_baselines": 10,
        "n_point_sources": 2,
        "n_healpix_pixels": 0,
        "n_times": 2,
        "n_frequencies": 2,
        "sky_representation": "point_sources",
        "solver_workers": 1,
        "loader_max_workers": 1,
        "setup_seconds": 0.5,
        "compile_seconds": 0.4,
        "steady_state_median_seconds": 0.1,
        "steady_state_min_seconds": 0.09,
        "steady_state_max_seconds": 0.12,
        "steady_state_iterations": 5,
        "host_transfer_seconds": 0.001,
        "peak_host_bytes": 1024,
        "backend_memory_info": {"backend": "numpy"},
        "reference_backend": "numpy",
        "max_absolute_deviation": 0.0,
        "max_relative_deviation": 0.0,
        "tolerance_rtol": 1e-12,
        "tolerance_atol": 1e-12,
        "within_tolerance": True,
        "unmeasured": ("gpu", "tpu", "distributed"),
    }
    values.update(overrides)
    return values


# =========================================================================
# P1 -- record completeness
# =========================================================================


def test_p1_a_record_missing_any_mandatory_field_is_rejected() -> None:
    """Every single Section 23 field is mandatory, one at a time."""
    for declared in fields(BenchmarkRecord):
        values = _complete_values()
        del values[declared.name]
        with pytest.raises(BenchmarkRecordError) as raised:
            BenchmarkRecord.create(**values)
        assert declared.name in str(raised.value)


def test_p1_a_none_in_a_non_nullable_field_is_rejected() -> None:
    """A field that was not measured is a gap, not a record."""
    with pytest.raises(BenchmarkRecordError, match="peak_host_bytes"):
        BenchmarkRecord.create(**_complete_values(peak_host_bytes=None))
    with pytest.raises(BenchmarkRecordError, match="compile_seconds"):
        BenchmarkRecord.create(**_complete_values(compile_seconds=None))


def test_p1_the_two_nullable_fields_stay_nullable() -> None:
    """Section 23 declares exactly two fields nullable; both must construct."""
    record = BenchmarkRecord.create(
        **_complete_values(accelerator_driver=None, precision_preset=None)
    )
    assert record.accelerator_driver is None
    assert record.precision_preset is None


def test_p1_an_unknown_field_is_rejected() -> None:
    """A record may not carry a field the schema never declared."""
    with pytest.raises(BenchmarkRecordError, match="speedup_multiplier"):
        BenchmarkRecord.create(**_complete_values(speedup_multiplier=12.0))


def test_p1_a_record_missing_a_field_is_never_silently_completed() -> None:
    """``records_are_complete`` checks a collection the caller did not build."""
    assert records_are_complete([BenchmarkRecord.create(**_complete_values())])


# =========================================================================
# P2 -- record honesty
# =========================================================================


def test_p2_a_no_accelerator_record_must_state_that_gpu_was_unmeasured() -> None:
    """Absence of an accelerator run is stated, never inferred from silence."""
    with pytest.raises(BenchmarkRecordError, match="unmeasured"):
        BenchmarkRecord.create(**_complete_values(unmeasured=("distributed",)))


def test_p2_an_accelerator_claim_requires_a_hardware_description() -> None:
    """Section 23: an accelerator claim without hardware is an acceptance failure."""
    with pytest.raises(BenchmarkRecordError, match="accelerator_driver"):
        BenchmarkRecord.create(
            **_complete_values(accelerator="gpu", accelerator_driver=None)
        )


@pytest.mark.parametrize("requested", ["numpy", "dask", "jax"])
def test_p2_every_backend_on_this_host_describes_itself_as_no_accelerator(
    requested: str,
) -> None:
    """Section 4: Tier 6 measures CPU only, on every backend it ships."""
    backend = (
        get_backend("jax", device="cpu")
        if requested == "jax"
        else get_backend(requested)
    )
    facts = describe_backend(backend, requested=requested)

    assert facts.accelerator == "none"
    assert facts.accelerator_driver is None
    assert facts.device_kind == "cpu"
    assert "gpu" in facts.unmeasured
    assert "tpu" in facts.unmeasured
    assert "distributed" in facts.unmeasured


def test_p2_only_the_jax_backend_reports_compilation() -> None:
    """``compilation_used`` reports what happened, not what was requested."""
    assert (
        describe_backend(get_backend("numpy"), requested="numpy").compilation_used
        is False
    )
    assert (
        describe_backend(get_backend("dask"), requested="dask").compilation_used
        is False
    )
    assert (
        describe_backend(
            get_backend("jax", device="cpu"), requested="jax"
        ).compilation_used
        is True
    )


# =========================================================================
# P3 -- harness determinism on a tiny NumPy workload
# =========================================================================


def test_p3_the_harness_produces_a_complete_record_for_a_tiny_numpy_workload(
    tmp_path,
) -> None:
    """The whole harness path, end to end, in the fast suite."""
    backend = get_backend("numpy")
    call = _tiny_kernel_call(backend)

    timing = time_backend_call(call, backend=backend)
    reference = timing.host_result
    deviation = compare_to_reference(reference, timing.host_result)
    record = build_record(
        environment=describe_environment(),
        backend_facts=describe_backend(backend, requested="numpy"),
        shape=WorkloadShape(
            workload="tiny_contraction",
            n_antennas=3,
            n_baselines=3,
            n_point_sources=4,
            n_healpix_pixels=0,
            n_times=1,
            n_frequencies=1,
            sky_representation="point_sources",
            solver_workers=1,
            loader_max_workers=1,
        ),
        timing=timing,
        deviation=deviation,
        precision=PrecisionConfig.standard(),
        precision_preset="standard",
    )

    assert record.schema_version == BENCHMARK_SCHEMA_VERSION
    assert record.within_tolerance is True
    assert record.max_absolute_deviation == 0.0
    assert record.steady_state_iterations >= 5
    assert record.steady_state_min_seconds <= record.steady_state_median_seconds
    assert record.steady_state_median_seconds <= record.steady_state_max_seconds
    assert record.compile_seconds >= 0.0
    assert record.peak_host_bytes > 0
    assert record.result_dtype == "complex128"
    assert record.git_sha != ""

    document = BenchmarkDocument(records=(record,), retracing=(), memory_scaling=())
    written = write_benchmark_document(
        document, directory=tmp_path, filename="record.json"
    )
    loaded = json.loads(written.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert set(loaded["records"][0]) == {
        declared.name for declared in fields(BenchmarkRecord)
    }
    assert loaded["records"][0]["unmeasured"] == ["gpu", "tpu", "distributed"]


def test_p3_the_harness_never_asserts_a_time_threshold() -> None:
    """Section 22.1: no benchmark number is ever hard-coded into an assertion."""
    timing_fields = {
        "setup_seconds",
        "compile_seconds",
        "steady_state_median_seconds",
        "steady_state_min_seconds",
        "steady_state_max_seconds",
        "host_transfer_seconds",
    }
    declared = {field.name for field in fields(BenchmarkRecord)}

    assert timing_fields <= declared


def test_p3_the_retracing_measurement_separates_first_and_repeat_calls() -> None:
    """The Tier 6H obligation: a changing source count is a changing shape.

    On the NumPy backend nothing is compiled, so a first call at a new source
    count costs the same as a repeat call at that count. That is the control
    measurement the JAX record is read against.
    """
    backend = get_backend("numpy")
    record = measure_retracing(backend, source_counts=(3, 2, 3, 2), n_baselines=2)

    assert isinstance(record, RetracingRecord)
    assert record.compilation_used is False
    assert record.steps == 4
    assert record.distinct_source_counts == 2
    assert set(record.first_call_seconds_by_source_count) == {"2", "3"}
    assert set(record.repeat_call_seconds_by_source_count) == {"2", "3"}
    assert record.max_first_to_repeat_ratio > 0.0
    assert record.total_seconds > 0.0


def test_p3_the_memory_scaling_measurement_reports_bytes_per_pair() -> None:
    """The Tier 6H obligation: the kernel working set is O(baselines x sources)."""
    backend = get_backend("numpy")
    small = measure_kernel_memory_scaling(backend, n_baselines=8, n_sources=8)
    large = measure_kernel_memory_scaling(backend, n_baselines=64, n_sources=64)

    assert isinstance(small, MemoryScalingRecord)
    assert small.pair_count == 64
    assert large.pair_count == 4096
    assert large.peak_host_bytes > small.peak_host_bytes
    assert large.bytes_per_pair > 0.0


def test_p3_benchmark_output_goes_to_the_documented_location() -> None:
    """Section 22.1: ``output/benchmarks/<UTC timestamp>-<host tag>.json``."""
    directory = benchmark_output_directory()

    assert directory.name == "benchmarks"
    assert directory.parent.name == "output"

    filename = benchmark_filename(datetime(2026, 7, 31, 12, 0, 0, tzinfo=UTC))

    assert filename.startswith("20260731T120000Z-")
    assert filename.endswith(".json")


def test_frozen_v1_reference_record_remains_byte_identical() -> None:
    """WP-7 adds new types; it may not rewrite the retained Tier 6 artifact."""
    reference = (
        Path(__file__).parents[3]
        / "output/benchmarks/reference/20260731T104303Z-darwin-arm64.json"
    )

    assert sha256(reference.read_bytes()).hexdigest() == (
        "00a02edd98903254e1f5f04569e88def0fff5ff239fbff40f2f5f34c5dc8b225"
    )
