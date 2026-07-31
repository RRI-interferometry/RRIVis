"""Tier 6 acceptance evidence that can be checked without running a benchmark.

``Tier6HybridRuntimePlan.md`` Section 32.9 requires "one committed set of records
under ``output/benchmarks/`` reproduced by the reviewer on their own machine".
This module is the half of that requirement a reviewer can check in the fast
suite: that the committed set exists, that it validates against the Section 23
schema, that it claims no accelerator, that it covers the Section 13.4 matrix on
all three backends, and that the benchmarks are wired so they never gate.

Reproducing the numbers is the reviewer's own ``pixi run bench``; nothing here
runs a benchmark, and nothing here asserts a time.

Tier 6J extends this file. Tier 6I creates it with the evidence Tier 6I owns.
"""

from __future__ import annotations

import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

from radiosim.benchmarks import (
    BENCHMARK_SCHEMA_VERSION,
    MEMORY_SCALING_SCHEMA_VERSION,
    RETRACING_SCHEMA_VERSION,
    BenchmarkRecord,
    MemoryScalingRecord,
    RetracingRecord,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIRECTORY = REPOSITORY_ROOT / "output" / "benchmarks" / "reference"

#: Every Section 13.4 row, by the name the harness records.
SECTION_13_4_WORKLOADS = frozenset(
    {
        "point_unpolarized_1time_2freq",
        "point_polarized_2times",
        "point_gaussian_morphology",
        "healpix_scalar",
        "healpix_polarized",
        "hybrid_point_plus_healpix",
        "heterogeneous_receptor_bases",
    }
)


def _reference_documents() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(REFERENCE_DIRECTORY.glob("*.json"))
    ]


@pytest.fixture(scope="module")
def committed() -> dict:
    documents = _reference_documents()
    assert documents, (
        f"Section 32.9 requires a committed record set under {REFERENCE_DIRECTORY}; "
        "found none. Run 'pixi run bench' and copy one output file there."
    )
    return documents[0]


def test_the_committed_records_validate_against_the_section_23_schema(
    committed: dict,
) -> None:
    """A published record must reconstruct as a complete record, or it is prose."""
    assert committed["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert committed["records"]

    declared = {field.name for field in fields(BenchmarkRecord)}
    for raw in committed["records"]:
        assert set(raw) == declared
        # Round-trips through the same validation the harness used, so a record
        # cannot be hand-edited into the repository in a state the schema
        # would have rejected.
        record = BenchmarkRecord.create(**raw)
        assert record.schema_version == BENCHMARK_SCHEMA_VERSION


def test_every_committed_record_claims_no_accelerator(committed: dict) -> None:
    """Section 4: Tier 6 makes no accelerator claim, and the evidence says so."""
    for raw in committed["records"]:
        assert raw["accelerator"] == "none", raw["workload"]
        assert raw["accelerator_driver"] is None
        assert raw["device_kind"] == "cpu"
        assert "gpu" in raw["unmeasured"]
        assert "tpu" in raw["unmeasured"]
        assert "distributed" in raw["unmeasured"]


def test_the_committed_records_cover_every_section_13_4_row_on_every_backend(
    committed: dict,
) -> None:
    """Section 32.9: the record set is the workload set, not a sample of it."""
    measured: dict[str, set[str]] = {}
    for raw in committed["records"]:
        measured.setdefault(raw["workload"], set()).add(raw["backend_requested"])

    assert SECTION_13_4_WORKLOADS <= set(measured)
    for workload in SECTION_13_4_WORKLOADS:
        assert measured[workload] == {"numpy", "jax", "dask"}, workload


def test_the_committed_records_state_correctness_against_numpy(
    committed: dict,
) -> None:
    """Section 22.2: every record carries its delta against the reference."""
    for raw in committed["records"]:
        assert raw["reference_backend"] == "numpy"
        assert raw["within_tolerance"] is True, raw["workload"]
        if raw["backend_requested"] in {"numpy", "dask"}:
            # Dask delegates to the same NumPy operations, so anything but zero
            # would be a real defect rather than a floating-point artifact.
            assert raw["max_absolute_deviation"] == 0.0, raw["workload"]


def test_the_committed_records_carry_the_full_measurement_context(
    committed: dict,
) -> None:
    """Fix.md Section 15's mandatory list, checked on the published evidence."""
    for raw in committed["records"]:
        assert raw["cpu_model"]
        assert raw["platform"]
        assert raw["cpu_count_logical"] >= 1
        assert raw["backend_version"]
        assert raw["git_sha"] and raw["git_sha"] != "unknown"
        assert raw["steady_state_iterations"] >= 5
        assert raw["setup_seconds"] > 0.0
        assert raw["peak_host_bytes"] > 0
        assert raw["backend_memory_info"]


def test_the_committed_set_discharges_both_tier6h_acceptance_obligations(
    committed: dict,
) -> None:
    """The two measurements the Tier 6H acceptance routed to Tier 6I.

    One: retracing under a time-varying visible-source count, which the Section
    22.2 timing loop (repeated identical calls) cannot surface. Two: the
    compiled kernel's ``(B, S, 2, 2)`` working set, which no Section 13.4
    workload is large enough to expose.
    """
    retracing = committed["retracing"]
    memory_scaling = committed["memory_scaling"]

    assert retracing, "no retracing measurement in the committed record set"
    assert memory_scaling, "no kernel working-set measurement in the committed set"

    for raw in retracing:
        assert raw["schema_version"] == RETRACING_SCHEMA_VERSION
        assert set(raw) == {field.name for field in fields(RetracingRecord)}
        RetracingRecord.create(**raw)
        assert raw["distinct_source_counts"] > 1
        assert raw["steps"] > raw["distinct_source_counts"]

    compiling = [raw for raw in retracing if raw["compilation_used"]]
    uncompiled = [raw for raw in retracing if not raw["compilation_used"]]

    assert compiling and uncompiled, (
        "the retracing measurement is only readable as a comparison between a "
        "compiling and a non-compiling backend"
    )
    # The measured fact, stated as a comparison rather than a threshold: a
    # compiling backend pays a first-call penalty at a new source count that a
    # non-compiling one does not.
    assert (
        compiling[0]["max_first_to_repeat_ratio"]
        > uncompiled[0]["max_first_to_repeat_ratio"]
    )

    for raw in memory_scaling:
        assert raw["schema_version"] == MEMORY_SCALING_SCHEMA_VERSION
        assert set(raw) == {field.name for field in fields(MemoryScalingRecord)}
        MemoryScalingRecord.create(**raw)
        assert raw["pair_count"] == raw["n_baselines"] * raw["n_sources"]
        assert raw["bytes_per_pair"] > 0.0

    ordered = sorted(memory_scaling, key=lambda raw: raw["pair_count"])
    assert len(ordered) >= 3, "one point is not a scaling measurement"
    assert ordered[-1]["peak_host_bytes"] > ordered[0]["peak_host_bytes"]
    # Linear in the product, not in either factor alone.
    assert 0.5 < ordered[-1]["bytes_per_pair"] / ordered[0]["bytes_per_pair"] < 2.0


def test_the_bench_task_is_exactly_the_documented_command() -> None:
    """Section 22.1 names one task; the documentation names the same one."""
    pixi_toml = (REPOSITORY_ROOT / "pixi.toml").read_text(encoding="utf-8")

    assert 'bench = "python -m pytest tests/performance/ -m performance"' in pixi_toml

    for document in (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "CLAUDE.md",
        REPOSITORY_ROOT / "docs" / "user_guide" / "backends.rst",
    ):
        assert "pixi run bench" in document.read_text(encoding="utf-8"), document


def test_the_benchmarks_never_gate() -> None:
    """Section 22.3: CI runs ``-m "not slow"`` and the benchmarks are slow."""
    performance = (
        REPOSITORY_ROOT / "tests" / "performance" / "test_backend_benchmarks.py"
    ).read_text(encoding="utf-8")
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "pytestmark = [pytest.mark.performance, pytest.mark.slow]" in performance
    assert '-m "not slow"' in workflow
    assert "tests/performance" not in workflow
    assert "pixi run bench" not in workflow
    assert "bench" not in workflow


def test_no_benchmark_number_is_hard_coded_into_a_performance_assertion() -> None:
    """Section 22.1: the performance tests assert correctness, never a time.

    A threshold on shared hardware is a flake generator. The check is textual on
    purpose: it is the assertion *shape* that must never appear, and a reviewer
    can confirm it by reading the same file.

    Comparing a duration against zero is a *positivity* check, not a threshold --
    "this was measured at all" is exactly what the records must prove -- and
    comparing one measured duration against another measured duration is a
    comparison, not a hard-coded number. Both stay allowed; a duration compared
    against any other literal does not.
    """
    performance = (
        REPOSITORY_ROOT / "tests" / "performance" / "test_backend_benchmarks.py"
    ).read_text(encoding="utf-8")

    assert_lines = [
        line for line in performance.splitlines() if line.strip().startswith("assert")
    ]

    thresholds = [
        line
        for line in assert_lines
        for literal in re.findall(r"_seconds\s*[<>]=?\s*([0-9][0-9_.eE+-]*)", line)
        if float(literal) != 0.0
    ]
    assert not thresholds, thresholds

    for forbidden in ("faster", "speedup", "at least .* seconds"):
        offending = [line for line in assert_lines if re.search(forbidden, line)]
        assert not offending, offending


def test_no_active_document_states_a_speed_without_citing_a_record() -> None:
    """Section 26's closing rule, enforced on the surfaces Tier 6 owns."""
    citation = "output/benchmarks/reference/"
    for document in (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "CLAUDE.md",
        REPOSITORY_ROOT / "docs" / "user_guide" / "backends.rst",
    ):
        text = document.read_text(encoding="utf-8")
        makes_a_speed_claim = any(
            phrase in text
            for phrase in ("slower than", "faster than", "steady-state median")
        )
        if makes_a_speed_claim:
            assert citation in text, document
