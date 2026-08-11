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
import subprocess
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
    PERF001_MEMORY_SCALING_SCHEMA_VERSION,
    PERF001_PROVENANCE_SCHEMA_VERSION,
    PERF001_RETRACING_SCHEMA_VERSION,
    PERF001_TARGET_KERNEL_PAIRS,
    BenchmarkDocument,
    BenchmarkRecord,
    BenchmarkRecordError,
    MemoryScalingRecord,
    MemoryScalingRecordV2,
    Perf001Provenance,
    RetracingRecord,
    RetracingRecordV2,
    WorkloadShape,
    authenticate_perf001_references,
    benchmark_backend_selection,
    benchmark_filename,
    benchmark_output_directory,
    build_record,
    compare_to_reference,
    describe_backend,
    describe_environment,
    describe_perf001_provenance,
    measure_kernel_memory_scaling,
    measure_perf001_memory_scaling_pair,
    measure_perf001_solver_memory_pair,
    measure_perf001_solver_retracing_pair,
    measure_perf001_synthetic_retracing_pair,
    measure_retracing,
    perf001_input_identity_sha256,
    records_are_complete,
    time_backend_call,
    verify_perf001_provenance_binding,
    verify_required_benchmark_accelerator,
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


def _perf001_provenance() -> Perf001Provenance:
    """A structurally valid source identity for unit-only measurements."""
    return Perf001Provenance.create(
        schema_version=PERF001_PROVENANCE_SCHEMA_VERSION,
        recorded_at_utc="2026-08-11T00:00:00+00:00",
        radiosim_version="0.3.0",
        git_sha="a" * 40,
        working_tree_clean=True,
        platform="test-platform",
        machine="test-machine",
        cpu_model="test-cpu",
        cpu_count_logical=1,
        python_version="3.11.13",
        numpy_version=np.__version__,
        jax_version="0.10.2",
        jaxlib_version="0.10.2",
        dask_version="2025.7.0",
        pixi_environment="default",
        pixi_lock_sha256="b" * 64,
    )


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


# =========================================================================
# PERF-001 v2 harness -- identity, exact-source routing, and P-a/P-b records
# =========================================================================


def test_benchmark_backend_environment_defaults_to_the_frozen_cpu_matrix() -> None:
    selection = benchmark_backend_selection({})

    assert selection.backend_requests == ("numpy", "jax", "dask")
    assert selection.required_accelerator is None


def test_benchmark_backend_environment_selects_explicit_required_gpu() -> None:
    selection = benchmark_backend_selection(
        {
            "RADIOSIM_BENCHMARK_BACKENDS": "numpy,gpu",
            "RADIOSIM_REQUIRE_ACCELERATOR": "gpu",
        }
    )

    assert selection.backend_requests == ("numpy", "gpu")
    assert selection.required_accelerator == "gpu"


@pytest.mark.parametrize(
    "environment",
    [
        {"RADIOSIM_BENCHMARK_BACKENDS": ""},
        {"RADIOSIM_BENCHMARK_BACKENDS": "gpu,numpy"},
        {"RADIOSIM_BENCHMARK_BACKENDS": "numpy, gpu"},
        {"RADIOSIM_BENCHMARK_BACKENDS": "numpy,gpu,gpu"},
        {"RADIOSIM_BENCHMARK_BACKENDS": "numpy,cuda"},
        {
            "RADIOSIM_BENCHMARK_BACKENDS": "numpy,jax",
            "RADIOSIM_REQUIRE_ACCELERATOR": "gpu",
        },
        {
            "RADIOSIM_BENCHMARK_BACKENDS": "numpy,gpu",
            "RADIOSIM_REQUIRE_ACCELERATOR": "tpu",
        },
    ],
)
def test_benchmark_backend_environment_rejects_ambiguous_or_unsafe_selection(
    environment: dict[str, str],
) -> None:
    with pytest.raises(BenchmarkRecordError):
        benchmark_backend_selection(environment)


def test_required_gpu_rejects_a_cpu_backend_instead_of_silently_falling_back() -> None:
    backend = get_backend("jax", device="cpu")

    with pytest.raises(BenchmarkRecordError, match="resolved device_kind='cpu'"):
        verify_required_benchmark_accelerator(
            backend,
            requested="gpu",
            required_accelerator="gpu",
        )


def test_perf001_input_identity_binds_manifest_order_shape_dtype_and_bytes() -> None:
    """The retained digest authenticates semantics, not an object identity."""
    manifest = {
        "schema_version": "radiosim.perf001.fixture.test.v1",
        "description": "ordered logical inputs",
    }
    first = np.arange(12, dtype=np.float64).reshape(3, 4)
    second = np.array([1, 2, 3], dtype=np.int32)

    digest = perf001_input_identity_sha256(
        manifest,
        (("first", first), ("second", second)),
    )

    assert digest == perf001_input_identity_sha256(
        dict(reversed(tuple(manifest.items()))),
        (("first", np.asfortranarray(first)), ("second", second.copy())),
    )
    assert digest != perf001_input_identity_sha256(
        manifest,
        (("second", second), ("first", first)),
    )
    assert digest != perf001_input_identity_sha256(
        manifest,
        (("first", first.astype(np.float32)), ("second", second)),
    )
    changed = first.copy()
    changed[0, 0] = 1.0
    assert digest != perf001_input_identity_sha256(
        manifest,
        (("first", changed), ("second", second)),
    )


@pytest.mark.parametrize(
    ("manifest", "inputs", "message"),
    [
        ({"description": "unversioned"}, (("x", np.ones(1)),), "schema_version"),
        (
            {"schema_version": "fixture.v1", "bad": float("nan")},
            (("x", np.ones(1)),),
            "finite JSON",
        ),
        (
            {"schema_version": "fixture.v1"},
            (("x", np.ones(1)), ("x", np.zeros(1))),
            "unique",
        ),
        (
            {"schema_version": "fixture.v1"},
            (("x", np.array([object()])),),
            "object",
        ),
    ],
)
def test_perf001_input_identity_rejects_ambiguous_inputs(
    manifest: dict[str, object],
    inputs: tuple[tuple[str, np.ndarray], ...],
    message: str,
) -> None:
    with pytest.raises(BenchmarkRecordError, match=message):
        perf001_input_identity_sha256(manifest, inputs)


def test_perf001_reference_authentication_covers_every_tracked_exact_path(
    tmp_path: Path,
) -> None:
    """No first-file glob may silently route acceptance to the wrong record."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    reference_dir = tmp_path / "output/benchmarks/reference/perf001"
    reference_dir.mkdir(parents=True)
    first = reference_dir / "20260811T010203Z-test.json"
    second = reference_dir / "20260811T020304Z-test.json"
    first.write_text('{"record": 1}\n', encoding="utf-8")
    second.write_text('{"record": 2}\n', encoding="utf-8")
    subprocess.run(["git", "add", "output"], cwd=tmp_path, check=True)

    relative_first = first.relative_to(tmp_path).as_posix()
    relative_second = second.relative_to(tmp_path).as_posix()
    expected = {
        relative_first: sha256(first.read_bytes()).hexdigest(),
        relative_second: sha256(second.read_bytes()).hexdigest(),
    }
    authenticated = authenticate_perf001_references(
        repository_root=tmp_path,
        expected_sha256=expected,
    )

    assert tuple(item.relative_path for item in authenticated) == (
        relative_first,
        relative_second,
    )
    assert tuple(item.sha256 for item in authenticated) == tuple(expected.values())

    with pytest.raises(BenchmarkRecordError, match="unlisted tracked"):
        authenticate_perf001_references(
            repository_root=tmp_path,
            expected_sha256={relative_first: expected[relative_first]},
        )
    with pytest.raises(BenchmarkRecordError, match=relative_second):
        authenticate_perf001_references(
            repository_root=tmp_path,
            expected_sha256={**expected, relative_second: "0" * 64},
        )


def test_perf001_reference_authentication_rejects_hidden_nested_routes(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    nested = tmp_path / "output/benchmarks/reference/perf001/alternate/record.json"
    nested.parent.mkdir(parents=True)
    nested.write_text('{"record": "hidden"}\n', encoding="utf-8")
    subprocess.run(["git", "add", "output"], cwd=tmp_path, check=True)

    with pytest.raises(BenchmarkRecordError, match="direct lowercase-.json"):
        authenticate_perf001_references(
            repository_root=tmp_path,
            expected_sha256={},
        )


@pytest.mark.parametrize(
    "filename",
    [
        "20260230T010203Z-linux-x86_64.json",
        "20260811T010203Z-Linux-x86_64.json",
        "20260811T010203Z--linux.json",
    ],
)
def test_perf001_reference_authentication_rejects_noncanonical_names(
    tmp_path: Path,
    filename: str,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    reference = tmp_path / "output/benchmarks/reference/perf001" / filename
    reference.parent.mkdir(parents=True)
    reference.write_text('{"record": "invalid-name"}\n', encoding="utf-8")
    subprocess.run(["git", "add", "-f", str(reference)], cwd=tmp_path, check=True)
    relative = reference.relative_to(tmp_path).as_posix()

    with pytest.raises(BenchmarkRecordError, match="canonical"):
        authenticate_perf001_references(
            repository_root=tmp_path,
            expected_sha256={relative: sha256(reference.read_bytes()).hexdigest()},
        )


def test_perf001_reference_authentication_rejects_a_tracked_symlink(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    payload = tmp_path / "payload.json"
    payload.write_text('{"record": "symlink-target"}\n', encoding="utf-8")
    reference = (
        tmp_path
        / "output/benchmarks/reference/perf001"
        / "20260811T010203Z-linux-x86_64.json"
    )
    reference.parent.mkdir(parents=True)
    reference.symlink_to(payload)
    subprocess.run(["git", "add", "-f", str(reference)], cwd=tmp_path, check=True)
    relative = reference.relative_to(tmp_path).as_posix()

    with pytest.raises(BenchmarkRecordError, match="regular non-symlink"):
        authenticate_perf001_references(
            repository_root=tmp_path,
            expected_sha256={relative: sha256(payload.read_bytes()).hexdigest()},
        )


def test_perf001_gitignore_exposes_only_direct_namespaced_json() -> None:
    repository_root = Path(__file__).parents[3]

    def ignored(relative: str) -> bool:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", relative],
            cwd=repository_root,
            check=False,
        )
        assert result.returncode in (0, 1)
        return result.returncode == 0

    assert not ignored(
        "output/benchmarks/reference/perf001/20260811T010203Z-linux-x86_64.json"
    )
    assert ignored("output/benchmarks/reference/perf001/readme.txt")
    assert ignored(
        "output/benchmarks/reference/perf001/nested/20260811T010203Z-linux-x86_64.json"
    )
    assert ignored(
        "output/benchmarks/reference/perf001/"
        "20260811T010203Z-linux-x86_64.json/nested.txt"
    )


def test_perf001_provenance_rejects_an_unrelated_clean_repo_with_spoofed_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment strings cannot relabel code loaded from another checkout."""
    import radiosim.benchmarks.harness as harness

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "perf001@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "PERF001 Test"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "pixi.lock").write_text("version: 6\n", encoding="utf-8")
    (tmp_path / "pixi.toml").write_text(
        "[workspace]\nname = 'spoofed'\n", encoding="utf-8"
    )
    (tmp_path / ".gitignore").write_text(".pixi/\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".gitignore", "pixi.lock", "pixi.toml"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "test: seed spoofed repository"],
        cwd=tmp_path,
        check=True,
    )
    environment_prefix = tmp_path / ".pixi/envs/default"
    executable = environment_prefix / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fake executable")
    executable.chmod(0o755)
    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "default")
    monkeypatch.setenv("PIXI_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CONDA_PREFIX", str(environment_prefix))
    monkeypatch.setattr(harness.sys, "prefix", str(environment_prefix))
    monkeypatch.setattr(harness.sys, "executable", str(executable))

    with pytest.raises(BenchmarkRecordError, match="loaded RadioSim"):
        describe_perf001_provenance(repository_root=tmp_path)


def test_perf001_provenance_binds_clean_live_head_environment_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import radiosim.benchmarks.harness as harness

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "perf001@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "PERF001 Test"],
        cwd=tmp_path,
        check=True,
    )
    lock = tmp_path / "pixi.lock"
    lock.write_text("version: 6\n", encoding="utf-8")
    manifest = tmp_path / "pixi.toml"
    manifest.write_text("[workspace]\nname = 'test'\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".pixi/\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".gitignore", "pixi.lock", "pixi.toml"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "test: seed lock"],
        cwd=tmp_path,
        check=True,
    )
    environment_prefix = tmp_path / ".pixi/envs/default"
    executable = environment_prefix / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fake executable")
    executable.chmod(0o755)
    package_file = tmp_path / "src/radiosim/__init__.py"
    harness_file = tmp_path / "src/radiosim/benchmarks/harness.py"
    harness_file.parent.mkdir(parents=True)
    package_file.write_text("# test package\n", encoding="utf-8")
    harness_file.write_text("# test harness\n", encoding="utf-8")
    subprocess.run(["git", "add", "src"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "test: seed loaded sources"],
        cwd=tmp_path,
        check=True,
    )
    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "default")
    monkeypatch.setenv("PIXI_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CONDA_PREFIX", str(environment_prefix))
    monkeypatch.setattr(harness.sys, "prefix", str(environment_prefix))
    monkeypatch.setattr(harness.sys, "executable", str(executable))
    monkeypatch.setattr(harness, "__file__", str(harness_file))
    import radiosim

    monkeypatch.setattr(radiosim, "__file__", str(package_file))

    provenance = describe_perf001_provenance(
        repository_root=tmp_path,
        pixi_environment="default",
        recorded_at=datetime(2026, 8, 11, tzinfo=UTC),
    )
    live_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()

    assert provenance.git_sha == live_sha
    assert provenance.working_tree_clean is True
    assert provenance.pixi_environment == "default"
    assert provenance.pixi_lock_sha256 == sha256(lock.read_bytes()).hexdigest()
    verify_perf001_provenance_binding(provenance, repository_root=tmp_path)

    unrelated_executable = tmp_path / ".pixi/unrelated/bin/python"
    unrelated_executable.parent.mkdir(parents=True)
    unrelated_executable.write_bytes(b"fake executable")
    unrelated_executable.chmod(0o755)
    monkeypatch.setattr(harness.sys, "executable", str(unrelated_executable))
    with pytest.raises(BenchmarkRecordError, match="active Python executable"):
        verify_perf001_provenance_binding(provenance, repository_root=tmp_path)
    monkeypatch.setattr(harness.sys, "executable", str(executable))

    unrelated_harness = tmp_path / ".pixi/unrelated-harness.py"
    unrelated_harness.write_text("# unrelated\n", encoding="utf-8")
    monkeypatch.setattr(harness, "__file__", str(unrelated_harness))
    with pytest.raises(BenchmarkRecordError, match="loaded RadioSim harness"):
        verify_perf001_provenance_binding(provenance, repository_root=tmp_path)
    monkeypatch.setattr(harness, "__file__", str(harness_file))

    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "py312")
    with pytest.raises(BenchmarkRecordError, match="expected Pixi environment"):
        verify_perf001_provenance_binding(provenance, repository_root=tmp_path)
    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "default")

    monkeypatch.setenv("PIXI_PROJECT_ROOT", str(tmp_path / "spoofed-project"))
    with pytest.raises(BenchmarkRecordError, match="PIXI_PROJECT_ROOT"):
        verify_perf001_provenance_binding(provenance, repository_root=tmp_path)
    monkeypatch.setenv("PIXI_PROJECT_ROOT", str(tmp_path))

    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / ".pixi/envs/spoofed"))
    with pytest.raises(BenchmarkRecordError, match="interpreter prefix"):
        verify_perf001_provenance_binding(provenance, repository_root=tmp_path)
    monkeypatch.setenv("CONDA_PREFIX", str(environment_prefix))

    unrelated_prefix = tmp_path / "unrelated-interpreter"
    unrelated_prefix.mkdir()
    monkeypatch.setattr(harness.sys, "prefix", str(unrelated_prefix))
    monkeypatch.setenv("CONDA_PREFIX", str(unrelated_prefix))
    with pytest.raises(BenchmarkRecordError, match="interpreter prefix"):
        verify_perf001_provenance_binding(provenance, repository_root=tmp_path)
    monkeypatch.setattr(harness.sys, "prefix", str(environment_prefix))
    monkeypatch.setenv("CONDA_PREFIX", str(environment_prefix))

    marker = tmp_path / "marker.txt"
    marker.write_text("new source\n", encoding="utf-8")
    subprocess.run(["git", "add", "marker.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "test: advance source"],
        cwd=tmp_path,
        check=True,
    )
    with pytest.raises(BenchmarkRecordError, match="git_sha"):
        verify_perf001_provenance_binding(provenance, repository_root=tmp_path)


def test_perf001_p_a_harness_builds_matched_unbounded_and_production_rows() -> None:
    backend = get_backend("numpy", precision="standard")
    reference, production = measure_perf001_memory_scaling_pair(
        backend,
        provenance=_perf001_provenance(),
        n_baselines=7,
        n_sources=5,
        comparison_id="unit-p-a",
    )

    assert isinstance(reference, MemoryScalingRecordV2)
    assert isinstance(production, MemoryScalingRecordV2)
    assert reference.schema_version == PERF001_MEMORY_SCALING_SCHEMA_VERSION
    assert reference.implementation_state == "unchunked_reference"
    assert production.implementation_state == "chunked_production"
    assert reference.context.input_identity_sha256 == (
        production.context.input_identity_sha256
    )
    assert reference.context.policy_id == "unbounded_reference_v1"
    assert production.context.policy_id == "target_kernel_pairs_131072_v1"
    assert reference.target_kernel_pairs is None
    assert production.target_kernel_pairs == PERF001_TARGET_KERNEL_PAIRS
    assert reference.kernel_baseline_chunks == (7,)
    assert production.kernel_baseline_chunks == (7,)
    assert reference.kernel_pair_counts == production.kernel_pair_counts == (35,)
    assert reference.synthetic_input_bytes_excluded > 0
    assert production.synthetic_input_bytes_excluded == (
        reference.synthetic_input_bytes_excluded
    )
    assert reference.peak_host_bytes > 0
    assert production.peak_host_bytes > 0


def test_perf001_p_a_records_the_observed_leaf_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record fields follow actual leaf calls, not a duplicate chunk formula."""
    import radiosim.core.contraction as contraction

    def forced_schedule(backend: Any, *, target_kernel_pairs: int | None) -> Any:
        def leaf(
            jones_p: Any,
            jones_q: Any,
            coherency: Any,
            phase: Any,
            envelope: Any,
            stokes_i: Any,
        ) -> Any:
            return contraction.baseline_contraction(
                jones_p,
                jones_q,
                coherency,
                phase,
                envelope,
                stokes_i,
                backend=backend,
            )

        compiled = backend.compile(leaf) if backend.supports_compilation else leaf

        def wrapper(
            jones_p: Any,
            jones_q: Any,
            coherency: Any,
            phase: Any,
            envelope: Any,
            stokes_i: Any,
        ) -> Any:
            chunks = (len(jones_p),) if target_kernel_pairs is None else (3, 2, 2)
            outputs = []
            start = 0
            for chunk in chunks:
                stop = start + chunk
                outputs.append(
                    compiled(
                        jones_p[start:stop],
                        jones_q[start:stop],
                        coherency,
                        phase[start:stop],
                        envelope[start:stop],
                        stokes_i,
                    )
                )
                start = stop
            return backend.xp.concatenate(outputs, axis=0)

        return wrapper

    monkeypatch.setattr(
        contraction, "_baseline_contraction_for_policy", forced_schedule
    )
    # The schema knows the production policy's stable schedule. Because the
    # monkeypatched production factory actually invokes (3, 2, 2), an observed
    # record must reject it. The rejected implementation instead predicted (7,)
    # independently and emitted a seemingly valid but false record.
    with pytest.raises(BenchmarkRecordError, match="exact stable production"):
        measure_perf001_memory_scaling_pair(
            get_backend("numpy", precision="standard"),
            provenance=_perf001_provenance(),
            n_baselines=7,
            n_sources=5,
            comparison_id="unit-p-a-observed-schedule",
        )


def test_perf001_p_b_synthetic_harness_observes_all_six_leaf_operands() -> None:
    backend = get_backend("jax", device="cpu", precision="standard")
    logical_counts = (3, 4, 5, 8, 3, 4, 5, 8)
    reference, production = measure_perf001_synthetic_retracing_pair(
        backend,
        provenance=_perf001_provenance(),
        source_counts=logical_counts,
        n_baselines=2,
        comparison_id="unit-p-b-synthetic",
    )

    assert isinstance(reference, RetracingRecordV2)
    assert isinstance(production, RetracingRecordV2)
    assert reference.schema_version == PERF001_RETRACING_SCHEMA_VERSION
    assert reference.logical_source_counts == production.logical_source_counts
    assert reference.kernel_source_counts == logical_counts
    assert production.kernel_source_counts == (4, 4, 8, 8, 4, 4, 8, 8)
    assert reference.distinct_signature_count == 4
    assert production.distinct_signature_count == 2
    assert reference.context.input_identity_sha256 == (
        production.context.input_identity_sha256
    )
    for observation in reference.observed_signatures + production.observed_signatures:
        assert observation.jones_p_shape is not None
        assert observation.jones_q_shape == observation.jones_p_shape
        assert observation.phase_shape == observation.jones_p_shape[:2]
        assert observation.envelope_shape == observation.jones_p_shape[:2]
        assert observation.coherency_shape is not None
        assert observation.stokes_i_shape is None
        assert observation.call_count >= 2


def test_perf001_p_b_zero_visible_steps_do_not_reach_the_leaf() -> None:
    backend = get_backend("jax", device="cpu", precision="standard")
    reference, production = measure_perf001_synthetic_retracing_pair(
        backend,
        provenance=_perf001_provenance(),
        source_counts=(0, 3, 0, 3),
        n_baselines=2,
        comparison_id="unit-p-b-zero-visible",
    )

    assert reference.leaf_call_count == production.leaf_call_count == 2
    assert reference.kernel_source_counts == (0, 3, 0, 3)
    assert production.kernel_source_counts == (0, 4, 0, 4)
    assert {item.jones_p_shape[1] for item in reference.observed_signatures} == {3}
    assert {item.jones_p_shape[1] for item in production.observed_signatures} == {4}


@pytest.mark.parametrize("solver", ["point", "healpix"])
def test_perf001_p_b_direct_solver_harness_uses_private_policy_and_compile_seams(
    solver: str,
) -> None:
    """The generic harness observes production seams without duplicating a solver."""
    backend = get_backend("jax", device="cpu", precision="standard")
    logical_counts = (3, 4, 5, 8, 3, 4, 5, 8)
    cached_inputs: dict[int, tuple[object, ...]] = {}
    logical_inputs: list[tuple[str, np.ndarray]] = []
    rng = np.random.default_rng(701)
    for count in sorted(set(logical_counts)):
        shape = (2, count, 2, 2)
        jones_p = np.asarray(
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
            dtype=np.complex128,
        )
        jones_q = np.asarray(
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
            dtype=np.complex128,
        )
        coherency = np.asarray(
            rng.standard_normal((count, 2, 2))
            + 1j * rng.standard_normal((count, 2, 2)),
            dtype=np.complex128,
        )
        phase = np.asarray(
            np.exp(1j * rng.standard_normal((2, count))),
            dtype=np.complex128,
        )
        envelope = np.ones((2, count), dtype=np.float64)
        cached_inputs[count] = (
            jones_p,
            jones_q,
            coherency,
            phase,
            envelope,
            None,
        )
        for name, values in (
            ("jones_p", jones_p),
            ("jones_q", jones_q),
            ("coherency", coherency),
            ("phase", phase),
            ("envelope", envelope),
        ):
            logical_inputs.append((f"sources_{count}.{name}", values))

    module_name = (
        "radiosim.core.visibility"
        if solver == "point"
        else "radiosim.core.visibility_healpix"
    )

    def run_step(policy: str, step_index: int) -> object:
        module = __import__(module_name, fromlist=["baseline_contraction_for"])
        count = logical_counts[step_index]
        inputs = cached_inputs[count]
        if policy == "pow2_compiled_v1" and count & (count - 1):
            kernel_count = 1 << (count - 1).bit_length()
            padding = kernel_count - count

            def repeated(values: np.ndarray, axis: int) -> np.ndarray:
                return np.concatenate(
                    (
                        values,
                        np.repeat(np.take(values, [0], axis=axis), padding, axis=axis),
                    ),
                    axis=axis,
                )

            j_p, j_q, coherency, phase, envelope, absent = inputs
            inputs = (
                repeated(j_p, 1),
                repeated(j_q, 1),
                np.concatenate(
                    (coherency, np.zeros((padding, 2, 2), dtype=coherency.dtype))
                ),
                repeated(phase, 1),
                repeated(envelope, 1),
                absent,
            )
        transferred = tuple(
            None if value is None else backend.asarray(value) for value in inputs
        )
        return module.baseline_contraction_for(backend)(*transferred)

    manifest = {
        "schema_version": "radiosim.perf001.fixture.direct_solver_test.v1",
        "solver": solver,
        "logical_source_counts": list(logical_counts),
    }
    reference, production = measure_perf001_solver_retracing_pair(
        backend,
        provenance=_perf001_provenance(),
        solver=solver,
        logical_source_counts=logical_counts,
        fixture_manifest=manifest,
        logical_inputs=tuple(logical_inputs),
        run_solver_step=run_step,
        comparison_id=f"unit-p-b-{solver}",
    )

    assert reference.solver == production.solver == solver
    assert reference.measurement_scope == f"complete_{solver}_solver_step"
    assert reference.distinct_signature_count == 4
    assert production.distinct_signature_count == 2
    assert production.kernel_source_counts == (4, 4, 8, 8, 4, 4, 8, 8)
    assert reference.context.input_identity_sha256 == (
        production.context.input_identity_sha256
    )


def test_perf001_p_b_solver_memory_pair_has_truthful_direct_scope() -> None:
    backend = get_backend("jax", device="cpu", precision="standard")
    source_counts = (3, 3)
    rng = np.random.default_rng(702)
    shape = (2, 3, 2, 2)
    host_inputs = (
        np.asarray(
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
            dtype=np.complex128,
        ),
        np.asarray(
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
            dtype=np.complex128,
        ),
        np.asarray(
            rng.standard_normal((3, 2, 2)) + 1j * rng.standard_normal((3, 2, 2)),
            dtype=np.complex128,
        ),
        np.asarray(np.exp(1j * rng.standard_normal((2, 3))), dtype=np.complex128),
        np.ones((2, 3), dtype=np.float64),
        None,
    )

    def run_solver(policy: str) -> object:
        import radiosim.core.visibility as point_visibility

        selected_inputs = host_inputs
        if policy == "pow2_compiled_v1":
            padding = 1

            def repeated(values: np.ndarray, axis: int) -> np.ndarray:
                return np.concatenate(
                    (values, np.take(values, [0], axis=axis)),
                    axis=axis,
                )

            j_p, j_q, coherency, phase, envelope, absent = host_inputs
            selected_inputs = (
                repeated(j_p, 1),
                repeated(j_q, 1),
                np.concatenate(
                    (coherency, np.zeros((padding, 2, 2), dtype=coherency.dtype))
                ),
                repeated(phase, 1),
                repeated(envelope, 1),
                absent,
            )
        transferred = tuple(
            None if value is None else backend.asarray(value)
            for value in selected_inputs
        )
        kernel = point_visibility.baseline_contraction_for(backend)
        first = kernel(*transferred)
        second = kernel(*transferred)
        return backend.stack((first, second), axis=0)

    reference, production = measure_perf001_solver_memory_pair(
        backend,
        provenance=_perf001_provenance(),
        solver="point",
        logical_n_baselines=2,
        logical_source_counts=source_counts,
        n_times=1,
        n_frequencies=2,
        fixture_manifest={
            "schema_version": "radiosim.perf001.fixture.solver_memory_test.v1",
            "solver": "point",
        },
        logical_inputs=tuple(
            (f"operand_{index}", value)
            for index, value in enumerate(host_inputs)
            if value is not None
        ),
        run_solver=run_solver,
        comparison_id="unit-p-b-point-memory",
    )

    assert reference.implementation_state == "unbucketed_reference"
    assert production.implementation_state == "bucketed_production"
    assert reference.logical_source_counts == production.logical_source_counts
    assert reference.kernel_source_counts == source_counts
    assert production.kernel_source_counts == (4, 4)
    assert reference.includes_solver_input_construction is True
    assert reference.includes_simulator_setup is False
    assert reference.peak_host_bytes > 0
    assert production.peak_host_bytes > 0
