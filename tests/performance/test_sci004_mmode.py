r"""SCI-004 phase-M2 red oracles for the non-gating m-mode benchmark record.

``docs/development/sci004_mmode_design.md`` Section 11 gives SCI-004 its own
performance record under ``output/benchmarks/reference/sci004/<UTC>-<host>.json``
with the exact top-level schema literal ``radiosim.benchmark.sci004.v1``.  It
"deliberately defines its own schema rather than extending the accepted
``radiosim.benchmark.perf001.v1`` inventory: every SCI-004 row must join a frame
certificate, scientific identity, deterministic block schedule, and
direct/backend comparison that the PERF-001 record has no analogue for, and each
schema remains governed by its own strict validator."

The official ``v1`` record is a fixed Cartesian product -- fixtures
``mmode_single_scalar_mode``, ``mmode_point_stokes_i``,
``mmode_point_full_stokes`` crossed with backends ``numpy``, ``jax``, ``dask``
-- so ``comparison_group_id == fixture_id``,
``workload_id == fixture_id + ":" + backend + ":standard"``, and the array has
exactly nine rows.

**This module measures nothing and gates nothing.**  Section 11 is explicit:
"A record is evidence only of these nine measured CPU rows.  Timing values never
gate CI and license neither a speedup nor a memory/accelerator advantage.
``PERF-001`` statements remain governed by separate accepted PERF-001 records."
The module is marked ``performance`` and ``slow`` for exactly that reason, so the
standard ``-m "not slow"`` gate never collects it.  What it binds is the record
*schema and inventory* -- the structure a later ``E3`` has to fill truthfully --
not any measured number, and the actual generation venue is ``S3``/``E3``, not
this red slice.

The Section 13.4 owner is ``radiosim.benchmarks``, whose SCI-004 record surface
does not exist at ``A1``; imports are function-local so each node yields its own
Section 14.1 outcome.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = [pytest.mark.performance, pytest.mark.slow]

#: Section 11's exact top-level and provenance schema literals.
SCI004_BENCHMARK_SCHEMA = "radiosim.benchmark.sci004.v1"
SCI004_PROVENANCE_SCHEMA = "radiosim.benchmark.sci004.provenance.v1"

#: Section 11's exact top-level key set.
TOP_LEVEL_KEYS: tuple[str, ...] = ("schema_version", "provenance", "workloads")

#: Section 11's fixed fixture and backend axes, in record order, as the accepted
#: 2026-08-24 accepted-capability-characterization correction amended them: "the
#: performance record's fixture product becomes the three point-family groups".
#: The two removed groups are not merely unmeasured -- measured through the
#: public solve path, a HEALPix-bearing payload published an identically zero
#: cube and the hybrid payload silently dropped its diffuse half -- and both are
#: now Section 8 rejections.
FIXTURE_IDS: tuple[str, ...] = (
    "mmode_single_scalar_mode",
    "mmode_point_stokes_i",
    "mmode_point_full_stokes",
)
BACKENDS: tuple[str, ...] = ("numpy", "jax", "dask")
WORKLOAD_COUNT = len(FIXTURE_IDS) * len(BACKENDS)

#: The superseded product this module pinned at ``R2``, retained *only* as the
#: preimage of the accepted phase-M2 red record's ``m2.performance.schema-
#: literals`` fixture bytes.  Section 13.7 keeps an accepted phase artifact
#: immutable, and that record's strict validator recomputes the fixture digest
#: from the bytes this module still declares, so the historical preimage is
#: spelled out here rather than interpolated from the live axis above.  It is a
#: record of what was true at ``R2``, never a live pin.
_R2_RETAINED_FIXTURE_IDS: tuple[str, ...] = (
    "mmode_point_full_stokes",
    "mmode_healpix_full_stokes",
    "mmode_hybrid_full_stokes",
)

#: Section 11's exact workload-row key set, in order.
WORKLOAD_KEYS: tuple[str, ...] = (
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

#: Section 11's exact lexicographically sorted claim array on every row.
CLAIMS_NOT_LICENSED: tuple[str, ...] = (
    "general_speedup",
    "gpu_or_accelerator_support",
    "perf001_evidence_or_closure",
    "performance_regression_gate",
    "unmeasured_workloads",
)

#: Section 11's two comparison predicate literals.
DIRECT_PREDICATE_ID = "sci004_two_tier_direct.v3"
BACKEND_PREDICATE_ID = "sci004_backend_complex128.v1"

#: Section 9's recorded transform-execution policy, which is not an accelerator
#: claim, and Section 11's fixed record directory.
TRANSFORM_EXECUTION_POLICY = "host_harmonics_backend_native_dense_v1"
RECORD_DIRECTORY = "output/benchmarks/reference/sci004"

_INVENTORY_FIXTURE = f"""\
schema_version: {SCI004_BENCHMARK_SCHEMA}
record_directory: {RECORD_DIRECTORY}
fixtures: ["{_R2_RETAINED_FIXTURE_IDS[0]}", "{_R2_RETAINED_FIXTURE_IDS[1]}", "{_R2_RETAINED_FIXTURE_IDS[2]}"]
backends: ["numpy", "jax", "dask"]
workload_count: {WORKLOAD_COUNT}
""".encode()

_PROVENANCE_FIXTURE = f"""\
schema_version: {SCI004_PROVENANCE_SCHEMA}
transform_execution_policy: {TRANSFORM_EXECUTION_POLICY}
pixi_environment: default
workload_count: {WORKLOAD_COUNT}
numeric_packages:
  - astropy
  - dask
  - erfa
  - healpy
  - iers_package
  - jax
  - jaxlib
  - numpy
  - scipy
""".encode()

_ROW_FIXTURE = f"""\
schema_version: {SCI004_BENCHMARK_SCHEMA}
device_kind: cpu
precision: standard
accumulation_dtype: complex128
result_dtype: complex128
claims_not_licensed:
  - general_speedup
  - gpu_or_accelerator_support
  - perf001_evidence_or_closure
  - performance_regression_gate
  - unmeasured_workloads
""".encode()

_COMPARISON_FIXTURE = f"""\
direct_predicate_id: {DIRECT_PREDICATE_ID}
backend_predicate_id: {BACKEND_PREDICATE_ID}
rtol: 1e-12
""".encode()

_PERF001_ORACLE = (
    "tests/performance/test_sci004_mmode.py::"
    "test_the_perf001_record_surface_and_non_gating_markers_hold_today"
)

_BENCHMARK_IMPORT_PATTERN = (
    r"ImportError: cannot import name '\w+' from 'radiosim\.benchmarks'"
)


def _local(function: str) -> str:
    return f"tests/performance/test_sci004_mmode.py::{function}"


def _case(
    case_id: str,
    requirement_id: str,
    function: str,
    fixture: bytes,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": _local(function),
        "expected_failure_kind": "missing-symbol",
        "expected_failure_pattern": _BENCHMARK_IMPORT_PATTERN,
        "fixture_defect_excluded_by": _PERF001_ORACLE,
        "fixture_bytes": fixture,
    }


SCI004_PHASE2_RED_CASES: tuple[dict[str, Any], ...] = (
    _case(
        "m2.performance.schema-literals",
        "sci004.section-11.sci004-record-declares-its-own-schema",
        "test_the_sci004_record_declares_its_own_schema_literals",
        _INVENTORY_FIXTURE,
    ),
    _case(
        "m2.performance.nine-row-inventory",
        "sci004.section-11.official-v1-inventory-is-nine-rows",
        "test_the_official_v1_inventory_is_the_exact_nine_row_product",
        _PROVENANCE_FIXTURE,
    ),
    _case(
        "m2.performance.workload-row-shape",
        "sci004.section-11.workload-row-and-claim-array",
        "test_every_workload_row_carries_its_exact_fields_and_claim_array",
        _ROW_FIXTURE,
    ),
    _case(
        "m2.performance.comparison-predicates",
        "sci004.section-11.direct-and-backend-comparison-predicates",
        "test_the_two_comparison_predicates_are_the_section_11_literals",
        _COMPARISON_FIXTURE,
    ),
)

SCI004_PHASE2_RED_GREEN_CONTROLS: tuple[str, ...] = (_PERF001_ORACLE,)


# --- green control ------------------------------------------------------------


def test_the_perf001_record_surface_and_non_gating_markers_hold_today() -> None:
    """The accepted PERF-001 surface is intact and this module never gates.

    Section 11 keeps the two schemas separate, so the PERF-001 inventory must
    still be exactly what it was; and the ``performance``/``slow`` markers this
    module carries are what keep the standard ``-m "not slow"`` gate from
    collecting an SCI-004 benchmark oracle.  Both hold at ``A1``, so a red
    failure below is the absence of the SCI-004 record surface.
    """
    from radiosim.benchmarks import (
        PERF001_SCHEMA_VERSION,
        BenchmarkRecord,
        benchmark_output_directory,
    )

    assert PERF001_SCHEMA_VERSION.startswith("radiosim.benchmark.perf001")
    assert PERF001_SCHEMA_VERSION != SCI004_BENCHMARK_SCHEMA
    assert isinstance(BenchmarkRecord, type)
    assert callable(benchmark_output_directory)

    marks = {mark.name for mark in pytestmark}
    assert marks == {"performance", "slow"}


# --- Section 11 red oracles ---------------------------------------------------


def test_the_sci004_record_declares_its_own_schema_literals() -> None:
    """Section 11: ``radiosim.benchmark.sci004.v1``, never a PERF-001 extension."""
    from radiosim.benchmarks import (
        SCI004_BENCHMARK_SCHEMA_VERSION,
        SCI004_PROVENANCE_SCHEMA_VERSION,
        sci004_reference_output_directory,
    )

    assert SCI004_BENCHMARK_SCHEMA_VERSION == SCI004_BENCHMARK_SCHEMA
    assert SCI004_PROVENANCE_SCHEMA_VERSION == SCI004_PROVENANCE_SCHEMA
    assert str(sci004_reference_output_directory()).endswith(RECORD_DIRECTORY)


def test_the_official_v1_inventory_is_the_exact_nine_row_product() -> None:
    """Section 11: the fixed fixture-by-backend Cartesian product, in order."""
    from radiosim.benchmarks import SCI004_WORKLOAD_INVENTORY

    inventory = tuple(SCI004_WORKLOAD_INVENTORY)
    assert len(inventory) == WORKLOAD_COUNT

    expected = tuple(
        (fixture, backend, f"{fixture}:{backend}:standard")
        for fixture in FIXTURE_IDS
        for backend in BACKENDS
    )
    observed = tuple(
        (row.fixture_id, row.backend, row.workload_id) for row in inventory
    )
    assert observed == expected
    for row in inventory:
        assert row.comparison_group_id == row.fixture_id
        assert row.device_kind == "cpu"
        assert row.precision == "standard"

    # Section 11, as amended: "The sky representation is ``point`` for all three
    # fixture groups, with positive point counts, zero for the absent HEALPix
    # representation, and distinct input identities across groups."
    for row in inventory:
        assert row.sky_representation == "point"


def test_every_workload_row_carries_its_exact_fields_and_claim_array() -> None:
    """Section 11: the exact ordered field set and the sorted claim array."""
    from radiosim.benchmarks import SCI004_WORKLOAD_KEYS, sci004_claims_not_licensed

    assert tuple(SCI004_WORKLOAD_KEYS) == WORKLOAD_KEYS
    claims = tuple(sci004_claims_not_licensed())
    assert claims == CLAIMS_NOT_LICENSED
    assert list(claims) == sorted(set(claims))
    # The claim array is what keeps a timing number from becoming a claim.
    assert "general_speedup" in claims
    assert "gpu_or_accelerator_support" in claims


def test_the_two_comparison_predicates_are_the_section_11_literals() -> None:
    """Section 11: ``sci004_two_tier_direct.v3`` and ``sci004_backend_complex128.v1``."""
    from radiosim.benchmarks import (
        SCI004_BACKEND_PREDICATE_ID,
        SCI004_DIRECT_PREDICATE_ID,
    )

    assert SCI004_DIRECT_PREDICATE_ID == DIRECT_PREDICATE_ID
    assert SCI004_BACKEND_PREDICATE_ID == BACKEND_PREDICATE_ID
    assert SCI004_DIRECT_PREDICATE_ID != SCI004_BACKEND_PREDICATE_ID
