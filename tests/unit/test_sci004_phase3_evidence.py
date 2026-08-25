"""Strict authentication of the SCI-004 phase-M3 evidence and Section 11 record.

``docs/development/sci004_mmode_design.md`` Sections 13.5, 14.2 and 14.4 freeze
this module's successor authority: it lands in ``S3`` with all four approved
constants as the literal ``None``, the official evidence path and the retained
performance record **absent**, and every synthetic strict schema and digest
fixture passing.  ``E3`` then changes *only* the four constants below and adds
the two artifacts plus the reproduction record.  No import, expression,
annotation, key, surrounding token, or other literal in any of the four
assignments may change, so this module's own token stream outside those four
spans is comparable to its direct-parent ``S3`` bytes.

``E3`` is the one phase whose write authority is four paths rather than three:
Section 13.5 grants the evidence JSON, its reproduction record, this
validator's constants, and exactly one
``output/benchmarks/reference/sci004/<UTC>-<host>.json`` record.  The fourth is
host- and timestamp-bound, so it cannot be a fixed literal here; it is named by
``APPROVED_PERFORMANCE_PATH`` at ``E3`` and the ``E3`` diff is checked against
that name rather than against a pattern that any file under the directory would
satisfy.

The superseded-versus-operative ``design_sha``.  Section 13.7's bounded
corrections move the operative ``D`` between ``R`` and ``S`` -- Section 14.4
stars the ``R3 ->* S3`` edge for exactly that reason -- so the evidence's
``design_sha`` and the retained ``R3`` record's own ``design_sha`` may differ.
Nothing here equates them, and a synthetic fixture below proves the difference
is accepted rather than merely unchecked.

Importing this module loads only the Python standard library plus ``pytest``,
following the phase-1 and phase-2 validators: an evidence-critical validator
must not depend on a package that is merely transitively present, because a lock
update could drop it and silently turn a hard authentication into a collection
error.  The generator at ``tools/sci004_mmode_phase3_evidence.py`` is the
normative producer; the checks below restate the same structure, key order and
encodings in their own code -- including an independent canonical-JSON and
domain-digest implementation -- rather than importing the producer's opinion of
them.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import struct
import subprocess
import sys
import time
import tokenize
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Section 14.2/13.5's four approved constants.  ``E3`` replaces exactly these
#: four ``None`` literals and nothing else in this module.
APPROVED_SOURCE_SHA: str | None = "b07925ab14b56b3ca0fa863f806290748a31df6b"
APPROVED_ARTIFACT_SHA256: str | None = (
    "600b51ac4d70778ee2d3bdf7b8842b83ba77dc34d541784ad1ad7d8e5be5f8ae"
)
APPROVED_PERFORMANCE_PATH: str | None = (
    "output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json"
)
APPROVED_PERFORMANCE_SHA256: str | None = (
    "07e59d3176866a78c17244849d6493365e9d410547e884cf56b254e60babe193"
)

TOOL = "tools/sci004_mmode_phase3_evidence.py"
ARTIFACT = "docs/development/sci004_mmode_phase3_evidence.json"
REPRODUCTION = "docs/development/sci004_mmode_phase3_evidence.md"
RED_RECORD = "docs/development/sci004_mmode_phase3_red_failures.json"
POST_SOURCE_RED_RECORD = (
    "docs/development/sci004_mmode_phase3_post_source_red_failures.json"
)
RED_RECORD_SCHEMA = "radiosim.sci004.mmode-phase3-red-failures.v1"
POST_SOURCE_RED_RECORD_SCHEMA = (
    "radiosim.sci004.mmode-phase3-post-source-red-failures.v1"
)
POST_SOURCE_PRE_FIX_SHA = "a61526d686ab768f05ecffa80cfd6223d4ee4c62"
HISTORICAL_RED_RECORD_SHA256 = (
    "486705a8d5e51c08f972c91aeae60f0a0bfeef5480b622515282295a6a3cde05"
)
DEPENDENCY_CERTIFICATE = "docs/development/sci004_mmode_phase3_sci005_dependency.json"
PERFORMANCE_DIRECTORY = "output/benchmarks/reference/sci004"

VALIDATOR = "tests/unit/test_sci004_phase3_evidence.py"

#: Section 14.2's exact MyST front matter for the reproduction record.
REPRODUCTION_FRONT_MATTER = "---\norphan: true\n---"

#: The four spans Section 13.5 lets ``E3`` rewrite inside this module.
APPROVED_CONSTANT_NAMES: tuple[str, ...] = (
    "APPROVED_SOURCE_SHA",
    "APPROVED_ARTIFACT_SHA256",
    "APPROVED_PERFORMANCE_PATH",
    "APPROVED_PERFORMANCE_SHA256",
)

GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
PERFORMANCE_PATH = re.compile(
    r"\Aoutput/benchmarks/reference/sci004/"
    r"\d{8}T\d{6}Z-[a-z0-9][a-z0-9-]{0,62}\.json\Z"
)

EVIDENCE_SCHEMA = "radiosim.sci004.mmode-phase3-evidence.v1"
BENCHMARK_SCHEMA = "radiosim.benchmark.sci004.v1"
BENCHMARK_PROVENANCE_SCHEMA = "radiosim.benchmark.sci004.provenance.v1"
SELF_REFERENCE_REASON = "self-reference: A binds the containing E commit"
TRANSFORM_EXECUTION_POLICY = "host_harmonics_backend_native_dense_v1"

#: Section 14.2's exact envelope key order.
ENVELOPE_KEYS: tuple[str, ...] = (
    "schema_version",
    "phase",
    "status",
    "generated_at_utc",
    "design_sha",
    "red_commit_sha",
    "source_sha",
    "evidence_commit_sha",
    "evidence_commit_sha_reason",
    "working_tree_clean",
    "environment",
    "source_identities",
    "red_failure_record",
    "results",
    "commands",
    "limitations",
    "claims_not_licensed",
)

#: Section 14.2's exact M3 ``results`` key order.
RESULT_KEYS: tuple[str, ...] = (
    "dependency_certificate",
    "output_cases",
    "fingerprint_rows",
    "ci_artifacts",
    "performance_record",
    "release_scan_cases",
    "rejection_cases",
)

DEPENDENCY_CERTIFICATE_KEYS: tuple[str, ...] = (
    "sci005_stage2_acceptance_commit_sha",
    "sci005_stage2_acceptance_artifact_sha256",
    "sci005_stage2_certificate_stdout_sha256",
)

OUTPUT_ROW_KEYS: tuple[str, ...] = (
    "format",
    "fixture_id",
    "written_solver_sha256",
    "read_solver_sha256",
    "time_sha256",
    "feed_sha256",
    "correlation_sha256",
    "file_sha256",
    "written_cube_sha256",
    "read_cube_sha256",
    "scientific_sha256",
    "pass",
)

FINGERPRINT_ROW_KEYS: tuple[str, ...] = (
    "family_id",
    "fixture_id",
    "input_identity_sha256",
    "canonical_era_grid_sha256",
    "solver_snapshot_sha256",
    "cube_sha256",
    "scientific_sha256",
    "expected_change_reason",
    "pass",
)

CI_ARTIFACT_ROW_KEYS: tuple[str, ...] = (
    "family_id",
    "fixture_id",
    "source_sha",
    "environment",
    "dispatch_identity",
    "cube_sha256",
    "scientific_sha256",
    "numeric_delta",
    "expected_change_reason",
    "ci001_verdict",
    "pass",
)

PERFORMANCE_RECORD_KEYS: tuple[str, ...] = (
    "path",
    "sha256",
    "schema_version",
    "source_sha",
    "workload_count",
    "workload_identities",
    "authenticated",
    "claims_not_licensed",
)

WORKLOAD_IDENTITY_KEYS: tuple[str, ...] = (
    "workload_id",
    "input_identity_sha256",
    "frame_certificate_sha256",
    "scientific_sha256",
    "result_cube_sha256",
)

RELEASE_SCAN_ROW_KEYS: tuple[str, ...] = (
    "scan_id",
    "command_index",
    "roadmap_occurrences",
    "done_occurrences",
    "unsupported_claim_occurrences",
    "expected_counts",
    "pass",
)

REJECTION_ROW_KEYS: tuple[str, ...] = (
    "fixture_id",
    "config_path",
    "exception_type",
    "issue_code",
    "exact_message",
    "test_nodeid",
    "allocation_started",
    "output_path_created",
    "pass",
)

#: Section 11's four characterized families, in the amended memo order, and its
#: three performance fixture groups in record order.
SECTION_11_FAMILIES: tuple[str, ...] = (
    "mmode_single_scalar_mode",
    "mmode_point_stokes_i",
    "mmode_point_full_stokes",
    "mmode_circular_receptor",
)
PERFORMANCE_FIXTURES: tuple[str, ...] = (
    "mmode_single_scalar_mode",
    "mmode_point_stokes_i",
    "mmode_point_full_stokes",
)
POLARIZED_FIXTURE = "mmode_point_full_stokes"
BACKENDS: tuple[str, ...] = ("numpy", "jax", "dask")

#: Section 10's three reader round trips, in ``ResultFormat`` vocabulary.
OUTPUT_FORMATS: tuple[str, ...] = ("hdf5", "uvfits", "ms")
LOSSLESS_CUBE_FORMATS: frozenset[str] = frozenset({"hdf5"})

#: Section 11's exact Section 11 record key sets.
BENCHMARK_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "schema_version",
    "provenance",
    "workloads",
    "dense_invariance",
)
PROVENANCE_KEYS: tuple[str, ...] = (
    "schema_version",
    "recorded_at_utc",
    "radiosim_version",
    "source_sha",
    "git_tree_sha256",
    "working_tree_clean",
    "host_tag",
    "platform",
    "machine",
    "cpu_model",
    "cpu_count_logical",
    "python_version",
    "pixi_environment",
    "pixi_manifest_sha256",
    "pixi_lock_sha256",
    "numeric_packages",
    "iers_table_sha256",
    "transform_execution_policy",
    "workload_count",
)
BENCHMARK_NUMERIC_PACKAGES: tuple[str, ...] = (
    "astropy",
    "dask",
    "erfa",
    "healpy",
    "iers_package",
    "jax",
    "jaxlib",
    "numpy",
    "scipy",
)
DENSE_INVARIANCE_KEYS: tuple[str, ...] = (
    "comparison_group_id",
    "numpy_cube_sha256",
    "jax_cube_sha256",
    "dask_cube_sha256",
    "identical",
)
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
    "dense_execution",
    "kernel_backend_block",
    "claims_not_licensed",
)
TIMING_KEYS: tuple[str, ...] = (
    "clock",
    "warmup_iterations",
    "synchronization_method",
    "frame",
    "sky_transform",
    "beam_transfer",
    "dense_contraction_and_synthesis",
    "host_transfer",
    "total",
    "direct_reference",
)
#: Section 11: the fused shared dense series.  ``per_m_contraction`` and
#: ``synthesis`` "denote exactly the kernel-block stages below, never shared row
#: series", so they must never appear as siblings of these keys.
MEASURED_SERIES: tuple[str, ...] = (
    "frame",
    "sky_transform",
    "beam_transfer",
    "dense_contraction_and_synthesis",
    "total",
)
KERNEL_STAGE_NAMES: tuple[str, ...] = ("per_m_contraction", "synthesis")
MEMORY_KEYS: tuple[str, ...] = (
    "measurement_scope",
    "estimated_host_peak_bytes",
    "measured_host_peak_bytes",
    "host_measurement_method",
    "host_measurement_limitations",
    "measured_native_peak_bytes",
    "measured_native_peak_bytes_reason",
    "native_measurement_method",
    "native_measurement_limitations",
    "estimate_covers_measured_host_peak",
)
MEASUREMENT_SCOPE = "single_mmode_solver_call_excluding_fixture_and_output_v1"
HOST_MEASUREMENT_METHOD = "process_rss_sampled_delta_v1"
SHARED_NATIVE_METHODS: frozenset[str] = frozenset({"unavailable"})
#: The backend-device methods Section 11 admits *only* inside a kernel block.
DEVICE_NATIVE_METHODS: tuple[str, ...] = (
    "jax_device_memory_stats_v1",
    "dask_worker_metrics_v1",
)
BLOCK_DIMENSION_KEYS: tuple[str, ...] = (
    "frequency_block_max",
    "signed_m_block_max",
    "baseline_block_max",
    "packed_value_block_max",
    "scheduled_block_count",
    "schedule_rows",
    "schedule_sha256",
)
SCHEDULE_ROW_KEYS: tuple[str, ...] = (
    "block_index",
    "frequency_start",
    "frequency_stop",
    "signed_m_start",
    "signed_m_stop",
    "baseline_start",
    "baseline_stop",
    "packed_value_count",
)
DIRECT_COMPARISON_KEYS: tuple[str, ...] = (
    "predicate_id",
    "reference_cube_sha256",
    "candidate_cube_sha256",
    "reference_error_cube_sha256",
    "horizon_free_cube_sha256",
    "horizon_free_qcheck_cube_sha256",
    "quadrature_shell_cube_sha256",
    "expected_cell_count",
    "compared_finite_cell_count",
    "evaluated_error_cell_count",
    "numerical_scale_jy",
    "horizon_free_shell_max_jy",
    "horizon_free_shell_l2",
    "horizon_free_shell_max_limit_jy",
    "horizon_free_shell_l2_limit",
    "quadrature_shell_max_jy",
    "quadrature_shell_l2",
    "reference_scale_jy",
    "deficit_max_jy",
    "deficit_l2",
    "deficit_max_quarter_jy",
    "deficit_max_half_jy",
    "convergence_factor",
    "pass",
)
BACKEND_COMPARISON_KEYS: tuple[str, ...] = (
    "predicate_id",
    "reference_workload_id",
    "reference_cube_sha256",
    "candidate_cube_sha256",
    "expected_cell_count",
    "compared_finite_cell_count",
    "reference_scale_jy",
    "maximum_absolute_deviation_jy",
    "maximum_relative_deviation",
    "rtol",
    "atol_jy",
    "pass",
)
#: Section 11's eleven-field kernel ``stage_comparison``.
STAGE_COMPARISON_KEYS: tuple[str, ...] = (
    "predicate_id",
    "reference_stage_sha256",
    "candidate_stage_sha256",
    "expected_cell_count",
    "compared_finite_cell_count",
    "reference_scale_jy",
    "maximum_absolute_deviation_jy",
    "maximum_relative_deviation",
    "rtol",
    "atol_jy",
    "pass",
)
KERNEL_STAGE_KEYS: tuple[str, ...] = (
    "sample_seconds",
    "synchronization_method",
    "native_measurement_method",
    "measured_native_peak_bytes",
    "measured_native_peak_bytes_reason",
    "stage_comparison",
)
KERNEL_SYNCHRONIZATION_METHODS = {
    "jax": "jax_block_until_ready_v1",
    "dask": "dask_compute_v1",
}
DIRECT_PREDICATE_ID = "sci004_two_tier_direct.v3"
BACKEND_PREDICATE_ID = "sci004_backend_complex128.v1"
BACKEND_RTOL = 1e-12
BACKEND_ATOL_FACTOR = 1e-12
DENSE_EXECUTION = "numpy_host_v1"
CLOCK = "time.perf_counter_ns"
MINIMUM_SAMPLES = 5
HORIZON_FREE_L2_LIMIT = 1e-8
CONVERGENCE_FACTOR_FLOOR = 2.0

#: Section 11's exact lexicographically sorted per-row claim array.
BENCHMARK_CLAIMS: tuple[str, ...] = (
    "general_speedup",
    "gpu_or_accelerator_support",
    "mmode_end_to_end_backend_execution",
    "perf001_evidence_or_closure",
    "performance_regression_gate",
    "unmeasured_workloads",
)

#: The three deferrals the accepted corrections require this phase to carry,
#: keyed by the topic prefix each ``claims_not_licensed`` literal opens with.
DEFERRAL_TOPICS: tuple[str, ...] = (
    "diffuse",
    "end-to-end-backend",
    "non-scalar-beam",
)

#: The venue requirement the reproduction record must state.  A second checkout
#: that shares the first tree's Pixi environment imports the *first* tree's
#: source through its editable ``.pth``, so a replayer who skips this observes a
#: tree other than the one the artifact describes.
REPRODUCTION_VENUE_TOKENS: tuple[str, ...] = ("pixi install", "editable")


def _tool() -> Any:
    """Import the tracked generator without adding an import-time dependency."""
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    try:
        import sci004_mmode_phase3_evidence as module
    finally:
        sys.path.pop(0)
    return module


# ---------------------------------------------------------------------------
# Independent canonical JSON and digest vocabulary
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> bytes:
    """Return Section 14's canonical JSON, implemented independently here.

    The synthetic fixtures hold only strings, booleans and exact integers, so
    the RFC 8785 number spelling the generator implements is not exercised by
    this encoder; the numeric spelling is checked directly against the
    generator's own renderer below.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _domain_digest(domain: str, payload: bytes) -> str:
    """Return Section 14.0's ``D(d,p) = SHA256(d || NUL || U64(len(p)) || p)``."""
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(len(payload).to_bytes(8, "big", signed=False))
    digest.update(payload)
    return digest.hexdigest()


def _object_digest(domain: str, value: Any) -> str:
    return _domain_digest(domain, _canonical(value))


def _f64be(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

SIXTY_FOUR = "0" * 63 + "1"
FORTY = "0" * 39 + "1"
OTHER_FORTY = "0" * 39 + "2"

SYNTHETIC_SAMPLES = 3
SYNTHETIC_BASELINES = 1
SYNTHETIC_FREQUENCIES = 1
SYNTHETIC_CELLS = 4 * SYNTHETIC_SAMPLES * SYNTHETIC_BASELINES * SYNTHETIC_FREQUENCIES
SYNTHETIC_PERFORMANCE_PATH = (
    f"{PERFORMANCE_DIRECTORY}/20260824T120000Z-synthetic-host.json"
)


def _digest(label: str) -> str:
    """Return a distinct, deterministic 64-hex stand-in for one named cube."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _schedule() -> dict[str, Any]:
    rows = [
        {
            "block_index": 0,
            "frequency_start": 0,
            "frequency_stop": SYNTHETIC_FREQUENCIES,
            "signed_m_start": 0,
            "signed_m_stop": 3,
            "baseline_start": 0,
            "baseline_stop": SYNTHETIC_BASELINES,
            "packed_value_count": 6,
        }
    ]
    return {
        "frequency_block_max": SYNTHETIC_FREQUENCIES,
        "signed_m_block_max": 3,
        "baseline_block_max": SYNTHETIC_BASELINES,
        "packed_value_block_max": 6,
        "scheduled_block_count": len(rows),
        "schedule_rows": rows,
        "schedule_sha256": _object_digest("radiosim.sci004.block-schedule.v1", rows),
    }


def _timings() -> dict[str, Any]:
    stage = {"status": "measured", "sample_seconds": [0.1] * MINIMUM_SAMPLES}
    return {
        "clock": CLOCK,
        "warmup_iterations": 1,
        "synchronization_method": "numpy_eager_v1",
        "frame": copy.deepcopy(stage),
        "sky_transform": copy.deepcopy(stage),
        "beam_transfer": copy.deepcopy(stage),
        "dense_contraction_and_synthesis": copy.deepcopy(stage),
        "host_transfer": {
            "status": "not_applicable",
            "reason": "the dense path is host NumPy, so no transfer occurs",
        },
        "total": {"status": "measured", "sample_seconds": [1.0] * MINIMUM_SAMPLES},
        "direct_reference": {
            "status": "not_measured",
            "reason": "the every-run gate is the mandatory correctness comparison",
        },
    }


NATIVE_UNAVAILABLE_REASON = (
    "the shared dense path is host NumPy, so there is no native device "
    "allocation to measure"
)
HOST_MEASUREMENT_LIMITATIONS: tuple[str, ...] = tuple(
    sorted(
        (
            "10 ms sampling may miss a shorter transient resident-set peak",
            "a baseline delta does not count solver allocations satisfied from "
            "pages already resident before the call",
            "current-process RSS excludes child-process and accelerator-device memory",
            "Section 9's dense estimate excludes the every-run Section 4.2 frame "
            "certificate and its retained ledgers",
        )
    )
)


def _memory() -> dict[str, Any]:
    """Return the covering case: the estimate happens to exceed the peak."""
    return {
        "measurement_scope": MEASUREMENT_SCOPE,
        "estimated_host_peak_bytes": 4096,
        "measured_host_peak_bytes": 2048,
        "host_measurement_method": HOST_MEASUREMENT_METHOD,
        "host_measurement_limitations": list(HOST_MEASUREMENT_LIMITATIONS),
        "measured_native_peak_bytes": None,
        "measured_native_peak_bytes_reason": NATIVE_UNAVAILABLE_REASON,
        "native_measurement_method": "unavailable",
        "native_measurement_limitations": [NATIVE_UNAVAILABLE_REASON],
        "estimate_covers_measured_host_peak": True,
    }


def _uncovered_memory() -> dict[str, Any]:
    """Return a valid sampled case the dense estimate does not cover.

    Correction #24 permits either observed relation because a finite-cadence RSS
    baseline delta may fall above or below the dense estimate.  This synthetic
    row deliberately exercises ``false`` while both budget inequalities hold.
    """
    memory = _memory()
    memory["estimated_host_peak_bytes"] = 14_471_104
    memory["measured_host_peak_bytes"] = 32_629_309
    memory["estimate_covers_measured_host_peak"] = False
    return memory


def _direct_comparison(fixture: str, cube: str) -> dict[str, Any]:
    scale = 1.0
    return {
        "predicate_id": DIRECT_PREDICATE_ID,
        "reference_cube_sha256": _digest(f"{fixture}:frozen128"),
        "candidate_cube_sha256": cube,
        "reference_error_cube_sha256": _digest(f"{fixture}:error"),
        "horizon_free_cube_sha256": _digest(f"{fixture}:w0"),
        "horizon_free_qcheck_cube_sha256": _digest(f"{fixture}:wq"),
        "quadrature_shell_cube_sha256": _digest(f"{fixture}:vq"),
        "expected_cell_count": SYNTHETIC_CELLS,
        "compared_finite_cell_count": SYNTHETIC_CELLS,
        "evaluated_error_cell_count": SYNTHETIC_CELLS,
        "numerical_scale_jy": scale,
        "horizon_free_shell_max_jy": 1e-12,
        "horizon_free_shell_l2": 1e-12,
        "horizon_free_shell_max_limit_jy": 1e-8 * scale + 1e-10,
        "horizon_free_shell_l2_limit": HORIZON_FREE_L2_LIMIT,
        "quadrature_shell_max_jy": 1e-6,
        "quadrature_shell_l2": 1e-7,
        "reference_scale_jy": scale,
        "deficit_max_jy": 1e-6,
        "deficit_l2": 1e-7,
        "deficit_max_quarter_jy": 4e-6,
        "deficit_max_half_jy": 2e-6,
        "convergence_factor": 4.0,
        "pass": True,
    }


def _backend_comparison(fixture: str, cube: str) -> dict[str, Any]:
    scale = 1.0
    return {
        "predicate_id": BACKEND_PREDICATE_ID,
        "reference_workload_id": f"{fixture}:numpy:standard",
        "reference_cube_sha256": cube,
        "candidate_cube_sha256": cube,
        "expected_cell_count": SYNTHETIC_CELLS,
        "compared_finite_cell_count": SYNTHETIC_CELLS,
        "reference_scale_jy": scale,
        "maximum_absolute_deviation_jy": 0.0,
        "maximum_relative_deviation": 0.0,
        "rtol": BACKEND_RTOL,
        "atol_jy": BACKEND_ATOL_FACTOR * scale,
        "pass": True,
    }


KERNEL_SCALAR_REASON = (
    "the resolved payload is scalar, so the production block table carries one "
    "field, while the routed contraction kernel's contract covers exactly "
    "Section 5.3's four science fields; a per-m kernel measurement for this "
    "group would describe nothing its own solve does"
)
KERNEL_NUMPY_REASON = (
    "the NumPy row is the shared dense reference, so it carries no separate "
    "backend kernel measurement"
)
KERNEL_NATIVE_REASON = "this CPU-only backend build exposes no device allocator counter"


def _kernel_stage(fixture: str, backend: str, stage: str) -> dict[str, Any]:
    scale = 1.0
    return {
        "sample_seconds": [0.01] * MINIMUM_SAMPLES,
        "synchronization_method": KERNEL_SYNCHRONIZATION_METHODS[backend],
        "native_measurement_method": "unavailable",
        "measured_native_peak_bytes": None,
        "measured_native_peak_bytes_reason": KERNEL_NATIVE_REASON,
        "stage_comparison": {
            "predicate_id": BACKEND_PREDICATE_ID,
            "reference_stage_sha256": _digest(f"{fixture}:{stage}:numpy"),
            "candidate_stage_sha256": _digest(f"{fixture}:{stage}:{backend}"),
            "expected_cell_count": 24,
            "compared_finite_cell_count": 24,
            "reference_scale_jy": scale,
            "maximum_absolute_deviation_jy": 0.0,
            "maximum_relative_deviation": 0.0,
            "rtol": BACKEND_RTOL,
            "atol_jy": BACKEND_ATOL_FACTOR * scale,
            "pass": True,
        },
    }


def _kernel_block(fixture: str, backend: str) -> dict[str, Any]:
    if backend == "numpy":
        return {"status": "not_applicable", "reason": KERNEL_NUMPY_REASON}
    if fixture != POLARIZED_FIXTURE:
        return {"status": "not_applicable_scalar_table", "reason": KERNEL_SCALAR_REASON}
    return {
        "status": "measured",
        "per_m_contraction": _kernel_stage(fixture, backend, "per_m_contraction"),
        "synthesis": _kernel_stage(fixture, backend, "synthesis"),
    }


BACKEND_RUNTIME_PAIRS = {
    "numpy": ("NumPy", "NumPy"),
    "jax": ("JAX", "JAXlib"),
    "dask": ("Dask", "NumPy"),
}


def _workload_row(fixture: str, backend: str, shared: dict[str, Any]) -> dict[str, Any]:
    implementation, kernel_runtime = BACKEND_RUNTIME_PAIRS[backend]
    cube = _digest(f"{fixture}:cube")
    return {
        "workload_id": f"{fixture}:{backend}:standard",
        "comparison_group_id": fixture,
        "fixture_id": fixture,
        "input_identity_sha256": _digest(f"{fixture}:input"),
        "frame_certificate_sha256": _digest(f"{fixture}:certificate"),
        "scientific_sha256": _digest(f"{fixture}:scientific"),
        "result_cube_sha256": cube,
        "source_sha": FORTY,
        "working_tree_clean": True,
        "backend": backend,
        "backend_runtime": {
            "implementation": implementation,
            "implementation_version": "1.0.0",
            "kernel_runtime": kernel_runtime,
            "kernel_runtime_version": "1.0.0",
        },
        "device_kind": "cpu",
        "precision": "standard",
        "accumulation_dtype": "complex128",
        "result_dtype": "complex128",
        "workers": 1,
        "n_antennas": 2,
        "n_baselines": SYNTHETIC_BASELINES,
        "n_frequencies": SYNTHETIC_FREQUENCIES,
        "sidereal_samples": SYNTHETIC_SAMPLES,
        "lmax": 4,
        "mmax": 4,
        "quadrature_nside": 4,
        "n_point_sources": 1,
        "n_healpix_pixels": 0,
        "sky_representation": "point",
        "working_memory_bytes": 1 << 30,
        "resolved_block_dimensions": shared["schedule"],
        "timings": shared["timings"],
        "memory": shared["memory"],
        "direct_comparison": shared["direct_comparison"],
        "backend_comparison": _backend_comparison(fixture, cube),
        "dense_execution": DENSE_EXECUTION,
        "kernel_backend_block": _kernel_block(fixture, backend),
        "claims_not_licensed": list(BENCHMARK_CLAIMS),
    }


def _synthetic_performance_document() -> dict[str, Any]:
    """Return one complete synthetic Section 11 record."""
    workloads: list[dict[str, Any]] = []
    invariance: list[dict[str, Any]] = []
    for fixture in PERFORMANCE_FIXTURES:
        cube = _digest(f"{fixture}:cube")
        shared = {
            "schedule": _schedule(),
            "timings": _timings(),
            "memory": _memory(),
            "direct_comparison": _direct_comparison(fixture, cube),
        }
        for backend in BACKENDS:
            workloads.append(_workload_row(fixture, backend, shared))
        invariance.append(
            {
                "comparison_group_id": fixture,
                "numpy_cube_sha256": cube,
                "jax_cube_sha256": cube,
                "dask_cube_sha256": cube,
                "identical": True,
            }
        )
    return {
        "schema_version": BENCHMARK_SCHEMA,
        "provenance": {
            "schema_version": BENCHMARK_PROVENANCE_SCHEMA,
            "recorded_at_utc": "2026-08-24T12:00:00Z",
            "radiosim_version": "0.4.0",
            "source_sha": FORTY,
            "git_tree_sha256": SIXTY_FOUR,
            "working_tree_clean": True,
            "host_tag": "synthetic-host",
            "platform": "darwin",
            "machine": "arm64",
            "cpu_model": "arm64",
            "cpu_count_logical": 10,
            "python_version": "3.11.13",
            "pixi_environment": "default",
            "pixi_manifest_sha256": SIXTY_FOUR,
            "pixi_lock_sha256": SIXTY_FOUR,
            "numeric_packages": dict.fromkeys(BENCHMARK_NUMERIC_PACKAGES, "1.0.0"),
            "iers_table_sha256": SIXTY_FOUR,
            "transform_execution_policy": TRANSFORM_EXECUTION_POLICY,
            "workload_count": 9,
        },
        "workloads": workloads,
        "dense_invariance": invariance,
    }


def _fixture_input_rows() -> list[dict[str, Any]]:
    rows = []
    for fixture in sorted(SECTION_11_FAMILIES):
        manifest = {
            "schema_version": "radiosim.mmode-input-identity.v1",
            "fixture_id": fixture,
        }
        rows.append(
            {
                "fixture_id": fixture,
                "input_identity_manifest": manifest,
                "input_identity_sha256": _object_digest(
                    "radiosim.mmode-input-identity.v1", manifest
                ),
            }
        )
    return rows


def _output_rows() -> list[dict[str, Any]]:
    written_cube = _digest("output:written")
    solver = _digest("output:solver")
    rows = []
    for fmt in OUTPUT_FORMATS:
        read_cube = (
            written_cube if fmt in LOSSLESS_CUBE_FORMATS else _digest(f"output:{fmt}")
        )
        rows.append(
            {
                "format": fmt,
                "fixture_id": POLARIZED_FIXTURE,
                "written_solver_sha256": solver,
                "read_solver_sha256": solver,
                "time_sha256": _digest("output:time"),
                "feed_sha256": _digest("output:feed"),
                "correlation_sha256": _digest("output:correlation"),
                "file_sha256": _digest(f"output:file:{fmt}"),
                "written_cube_sha256": written_cube,
                "read_cube_sha256": read_cube,
                "scientific_sha256": _digest("output:scientific"),
                "pass": True,
            }
        )
    return rows


def _synthetic_document(module: Any) -> dict[str, Any]:
    """Return one complete synthetic Section 14.2 M3 envelope."""
    performance = _synthetic_performance_document()
    performance_bytes = module.canonical_json(performance)
    rows = _fixture_input_rows()
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "phase": "M3",
        "status": "candidate",
        "generated_at_utc": "2026-08-24T12:00:00Z",
        # Section 13.7/14.4: the operative ``D`` frozen at ``R3`` and the ``R3``
        # record's own ``design_sha`` are expected to differ, so these are
        # deliberately distinct values here.
        "design_sha": FORTY,
        "red_commit_sha": OTHER_FORTY,
        "source_sha": FORTY,
        "evidence_commit_sha": None,
        "evidence_commit_sha_reason": SELF_REFERENCE_REASON,
        "working_tree_clean": True,
        "environment": {
            "python": "3.11.13",
            "platform": "darwin",
            "machine": "arm64",
            "pixi_environment": "default",
            "pixi_lock_sha256": SIXTY_FOUR,
            "astropy_version": "7.0.0",
            "erfa_version": "2.0.0",
            "iers_package_version": "0.2024.0",
            "iers_table_sha256": SIXTY_FOUR,
            "numeric_packages": dict.fromkeys(module.NUMERIC_PACKAGES, "1.0.0"),
        },
        "source_identities": {
            "git_tree_sha256": SIXTY_FOUR,
            "pixi_manifest_sha256": SIXTY_FOUR,
            "pixi_lock_sha256": SIXTY_FOUR,
            "convention_identity_sha256": SIXTY_FOUR,
            "fixture_input_rows": rows,
            "input_identity_set_sha256": _object_digest(
                "radiosim.sci004-phase-input-set.v1", rows
            ),
        },
        "red_failure_record": {
            "path": RED_RECORD,
            "sha256": SIXTY_FOUR,
            "schema_version": RED_RECORD_SCHEMA,
            "pre_fix_source_sha": OTHER_FORTY,
            "validated": True,
            "post_source_delta": {
                "path": POST_SOURCE_RED_RECORD,
                "sha256": SIXTY_FOUR,
                "schema_version": POST_SOURCE_RED_RECORD_SCHEMA,
                "pre_fix_source_sha": POST_SOURCE_PRE_FIX_SHA,
                "validated": True,
            },
        },
        "results": {
            "dependency_certificate": {
                "sci005_stage2_acceptance_commit_sha": FORTY,
                "sci005_stage2_acceptance_artifact_sha256": SIXTY_FOUR,
                "sci005_stage2_certificate_stdout_sha256": SIXTY_FOUR,
            },
            "output_cases": _output_rows(),
            "fingerprint_rows": [
                {
                    "family_id": family,
                    "fixture_id": family,
                    "input_identity_sha256": _digest(f"{family}:input"),
                    "canonical_era_grid_sha256": _digest(f"{family}:grid"),
                    "solver_snapshot_sha256": _digest(f"{family}:snapshot"),
                    "cube_sha256": _digest(f"{family}:cube"),
                    "scientific_sha256": _digest(f"{family}:scientific"),
                    "expected_change_reason": (
                        "a changed pin requires old and new cubes and an "
                        "equation-level explanation"
                    ),
                    "pass": True,
                }
                for family in SECTION_11_FAMILIES
            ],
            "ci_artifacts": [
                {
                    "family_id": family,
                    "fixture_id": family,
                    "source_sha": FORTY,
                    "environment": "osx-arm64-py311",
                    "dispatch_identity": "accepted-baseline-dispatch",
                    "cube_sha256": _digest(f"{family}:cube"),
                    "scientific_sha256": _digest(f"{family}:scientific"),
                    "numeric_delta": 0.0,
                    "expected_change_reason": (
                        "the retained observation set is the pin; a new cell is "
                        "admitted by adjudication"
                    ),
                    "ci001_verdict": "accepted-observation-set",
                    "pass": True,
                }
                for family in SECTION_11_FAMILIES
            ],
            "performance_record": {
                "path": SYNTHETIC_PERFORMANCE_PATH,
                "sha256": hashlib.sha256(performance_bytes).hexdigest(),
                "schema_version": BENCHMARK_SCHEMA,
                "source_sha": FORTY,
                "workload_count": 9,
                "workload_identities": [
                    {
                        "workload_id": row["workload_id"],
                        "input_identity_sha256": row["input_identity_sha256"],
                        "frame_certificate_sha256": row["frame_certificate_sha256"],
                        "scientific_sha256": row["scientific_sha256"],
                        "result_cube_sha256": row["result_cube_sha256"],
                    }
                    for row in performance["workloads"]
                ],
                "authenticated": True,
                "claims_not_licensed": list(BENCHMARK_CLAIMS),
            },
            "release_scan_cases": [
                {
                    "scan_id": "m3.release.register-still-roadmap",
                    "command_index": 1,
                    "roadmap_occurrences": 1,
                    "done_occurrences": 0,
                    "unsupported_claim_occurrences": 0,
                    "expected_counts": {
                        "roadmap_occurrences": 1,
                        "done_occurrences": 0,
                        "unsupported_claim_occurrences": 0,
                    },
                    "pass": True,
                }
            ],
            "rejection_cases": [
                {
                    "fixture_id": code,
                    "config_path": "execution.simulator",
                    "exception_type": (
                        "radiosim.io.config_resolution.UnsupportedConfigError"
                    ),
                    "issue_code": code,
                    "exact_message": "the public m-mode path refuses this payload",
                    "test_nodeid": f"tests/characterization/test_sci004_mmode.py::{code}",
                    "allocation_started": False,
                    "output_path_created": False,
                    "pass": True,
                }
                for code in ("mmode_public_components", "mmode_public_beam")
            ],
        },
        "commands": [
            {
                "argv": ["pixi", "run", "python", TOOL, "generate"],
                "cwd": ".",
                "pixi_environment": "default",
                "started_at_utc": "2026-08-24T12:00:00Z",
                "duration_seconds": 1.0,
                "exit_code": 0,
                "stdout_sha256": hashlib.sha256(b"").hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            }
        ],
        "limitations": sorted(module.LIMITATIONS),
        "claims_not_licensed": sorted(module.CLAIMS_NOT_LICENSED),
    }


def _rejects(module: Any, document: Any, *, performance: bool = False) -> str:
    """Require the named validator to refuse ``document`` and return the detail."""
    validator = (
        module.validate_performance_document
        if performance
        else module.validate_evidence_document
    )
    with pytest.raises(module.EvidenceError) as excinfo:
        validator(document)
    return str(excinfo.value.detail)


# ---------------------------------------------------------------------------
# S3-state authority
# ---------------------------------------------------------------------------


def test_the_approved_constants_are_null_sentinels_before_e3() -> None:
    """Section 14.2: at ``S3`` all four approved values are ``None``."""
    constants = (
        APPROVED_SOURCE_SHA,
        APPROVED_ARTIFACT_SHA256,
        APPROVED_PERFORMANCE_PATH,
        APPROVED_PERFORMANCE_SHA256,
    )
    if any(constant is None for constant in constants):
        assert all(constant is None for constant in constants), (
            "the four approved constants flip together at E3, never partially"
        )
        return
    assert GIT_SHA.fullmatch(str(APPROVED_SOURCE_SHA))
    assert SHA256.fullmatch(str(APPROVED_ARTIFACT_SHA256))
    assert SHA256.fullmatch(str(APPROVED_PERFORMANCE_SHA256))
    assert PERFORMANCE_PATH.fullmatch(str(APPROVED_PERFORMANCE_PATH))


def test_the_official_artifacts_are_absent_in_the_s3_state() -> None:
    """Section 14.2: null constants require both declared outputs to be absent.

    For M3 the declared set is two files, not one, so both are checked; a
    retained performance record without its evidence envelope is exactly the
    partial set Section 14.2 refuses to reuse.
    """
    if APPROVED_ARTIFACT_SHA256 is not None:
        return
    assert not (REPOSITORY_ROOT / ARTIFACT).exists()
    directory = REPOSITORY_ROOT / PERFORMANCE_DIRECTORY
    assert not directory.exists() or not sorted(directory.glob("*.json"))


def test_the_tracked_generator_and_its_inputs_exist_at_s3() -> None:
    """Section 14.2: the generator and its retained inputs are tracked at ``S3``."""
    assert (REPOSITORY_ROOT / TOOL).is_file()
    assert (REPOSITORY_ROOT / RED_RECORD).is_file()
    assert (REPOSITORY_ROOT / POST_SOURCE_RED_RECORD).is_file()
    assert (REPOSITORY_ROOT / DEPENDENCY_CERTIFICATE).is_file()


def test_the_generator_imports_only_the_standard_library() -> None:
    """An evidence-critical verifier carries no import-time package dependency."""
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    head = source[: source.index("class EvidenceError")]
    for forbidden in ("import numpy", "import astropy", "import pytest", "import yaml"):
        assert forbidden not in head, forbidden


def test_the_generator_refuses_before_producing_anything() -> None:
    """Section 14.2: the pre-output check runs before any output is opened.

    The probe is a real run of the tracked generator invoked with a
    ``--source-sha`` that is not ``HEAD``.  It must fail closed with the frozen
    prefix, print nothing on stdout, and leave both declared outputs
    byte-identical.
    """
    module = _tool()
    artifact = REPOSITORY_ROOT / ARTIFACT
    before = artifact.read_bytes() if artifact.exists() else None
    directory = REPOSITORY_ROOT / PERFORMANCE_DIRECTORY
    records_before = (
        sorted(path.name for path in directory.glob("*.json"))
        if directory.exists()
        else []
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / TOOL),
            "generate",
            "--source-sha",
            "0" * 40,
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr.startswith(module.PREFLIGHT + ": ")
    assert "Traceback" not in completed.stderr
    assert any(
        reason in completed.stderr
        for reason in (
            "is not the approved source",
            "not globally clean",
            "already exists",
        )
    )
    after = artifact.read_bytes() if artifact.exists() else None
    assert after == before
    records_after = (
        sorted(path.name for path in directory.glob("*.json"))
        if directory.exists()
        else []
    )
    assert records_after == records_before


def test_the_preflight_refuses_a_dirty_tree_before_any_output(monkeypatch) -> None:
    """Section 14.2's dirty-tree refusal, exercised without dirtying anything."""
    module = _tool()
    real = module._git

    def fake(*arguments: str) -> str:
        if arguments[:1] == ("status",):
            return " M src/radiosim/core/mmode/solver.py\n"
        return real(*arguments)

    monkeypatch.setattr(module, "_git", fake)
    with pytest.raises(module.EvidenceError) as excinfo:
        module.preflight()
    assert excinfo.value.prefix == module.PREFLIGHT
    assert "not globally clean" in excinfo.value.detail


def test_the_preflight_refuses_an_existing_declared_output(monkeypatch) -> None:
    """Section 14.2: an already-present declared output stops generation."""
    module = _tool()
    real = module._git

    def fake(*arguments: str) -> str:
        if arguments[:1] == ("status",):
            return ""
        return real(*arguments)

    monkeypatch.setattr(module, "_git", fake)
    with pytest.raises(module.EvidenceError) as excinfo:
        module.preflight(declared=(TOOL,))
    assert excinfo.value.prefix == module.PREFLIGHT
    assert "already exists" in excinfo.value.detail


def test_the_generator_produces_at_a_clean_source_rather_than_refusing() -> None:
    """Section 14.2/14.4: ``generate`` is bound to a venue, not prohibited."""
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    body = source[source.index('if arguments.command == "generate":') :]
    assert "return build_phase3_evidence(arguments.source_sha)" in body
    build = source[source.index("def build_phase3_evidence") :]
    build = build[: build.index("def main(")]
    assert "validate_performance_document(performance_document)" in build
    assert "validate_evidence_document(document)" in build
    assert "write_atomic_no_overwrite(" in build


def test_the_generator_publishes_the_performance_record_before_the_envelope() -> None:
    """Section 14.2: "performance first and evidence last"."""
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    build = source[source.index("def build_phase3_evidence") :]
    build = build[: build.index("def main(")]
    first = build.index("write_atomic_no_overwrite(REPOSITORY_ROOT / performance_path")
    second = build.index(
        "write_atomic_no_overwrite(REPOSITORY_ROOT / EVIDENCE_ARTIFACT"
    )
    assert first < second
    assert build.index("require_declared_outputs_only(declared)") > second


def test_the_generator_declares_exactly_the_two_m3_outputs() -> None:
    """Section 14.2: for M3 the declared set is the envelope and one record."""
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    assert "declared = (performance_path, EVIDENCE_ARTIFACT)" in source


def test_the_design_sha_is_the_frozen_binding_not_a_memo_tip_search() -> None:
    """Section 13.1: never "a phase-local memo tip or a search result".

    Section 14.4 stars the ``R3 ->* S3`` edge, so accepted corrections stand
    between the frozen ``D`` and this checkout by construction; a generator that
    derived ``design_sha`` as the newest memo-touching commit would both perform
    the forbidden search and refuse to run here.
    """
    module = _tool()
    frozen = module._frozen_binding("APPROVED_SCI004_D_SHA")
    assert GIT_SHA.fullmatch(frozen)
    assert module._design_sha() == frozen
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    assert '"--", "docs/development/sci004_mmode_design.md"' not in source, (
        "the operative D may not be derived by searching the memo's history"
    )


def test_the_frozen_design_binding_is_read_from_the_dependency_validator() -> None:
    """Section 14.0: the binding has exactly one site, read by AST."""
    module = _tool()
    frozen = module._frozen_binding("APPROVED_SCI004_D_SHA")
    text = (REPOSITORY_ROOT / "tests/unit/test_sci004_phase3_dependency.py").read_text(
        encoding="utf-8"
    )
    assert f'APPROVED_SCI004_D_SHA = "{frozen}"' in text
    tool_source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    assert frozen not in tool_source, (
        "the generator must read the frozen binding, never restate it"
    )


# ---------------------------------------------------------------------------
# Correction #24: external RSS sampler and dual red-record binding
# ---------------------------------------------------------------------------


def test_the_linux_rss_parser_uses_resident_pages_and_positive_page_size(
    monkeypatch,
) -> None:
    module = _tool()
    monkeypatch.setattr(module.Path, "read_text", lambda *_args, **_kwargs: "9 7 5\n")
    monkeypatch.setattr(module.os, "sysconf", lambda _name: 4096)
    assert module._linux_rss_bytes(123) == 7 * 4096


@pytest.mark.parametrize("statm", ["", "1", "1 nope", "1 -2"])
def test_the_linux_rss_parser_rejects_malformed_resident_pages(
    monkeypatch, statm: str
) -> None:
    module = _tool()
    monkeypatch.setattr(module.Path, "read_text", lambda *_args, **_kwargs: statm)
    monkeypatch.setattr(module.os, "sysconf", lambda _name: 4096)
    with pytest.raises(module.EvidenceError, match="malformed"):
        module._linux_rss_bytes(123)


def test_the_linux_rss_parser_rejects_nonpositive_page_size(monkeypatch) -> None:
    module = _tool()
    monkeypatch.setattr(module.Path, "read_text", lambda *_args, **_kwargs: "1 2")
    monkeypatch.setattr(module.os, "sysconf", lambda _name: 0)
    with pytest.raises(module.EvidenceError, match="not positive"):
        module._linux_rss_bytes(123)


def test_the_linux_rss_parser_rejects_uint64_overflow(monkeypatch) -> None:
    module = _tool()
    pages = 1 << 63
    monkeypatch.setattr(
        module.Path, "read_text", lambda *_args, **_kwargs: f"1 {pages}"
    )
    monkeypatch.setattr(module.os, "sysconf", lambda _name: 4096)
    with pytest.raises(module.EvidenceError, match="overflows"):
        module._linux_rss_bytes(123)


class _FakeProcPidInfo:
    def __init__(self, returned_size: int, resident_size: int = 8192) -> None:
        self.argtypes: Any = None
        self.restype: Any = None
        self.returned_size = returned_size
        self.resident_size = resident_size

    def __call__(
        self,
        _pid: int,
        flavor: int,
        _arg: int,
        buffer: Any,
        size: int,
    ) -> int:
        import ctypes

        assert flavor == 4
        assert size == 96
        words = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint64))
        words[1] = self.resident_size
        return self.returned_size


class _FakeLibproc:
    def __init__(self, returned_size: int, resident_size: int = 8192) -> None:
        self.proc_pidinfo = _FakeProcPidInfo(returned_size, resident_size)


def test_the_darwin_sampler_uses_the_complete_96_byte_proc_taskinfo() -> None:
    import ctypes

    module = _tool()
    assert ctypes.sizeof(module._darwin_proc_taskinfo_type()) == 96
    assert module._darwin_rss_bytes(123, _FakeLibproc(96, 12_345)) == 12_345


def test_a_short_darwin_proc_pidinfo_result_is_rejected() -> None:
    module = _tool()
    with pytest.raises(module.EvidenceError, match="returned 95 bytes, not 96"):
        module._darwin_rss_bytes(123, _FakeLibproc(95))


def test_an_unsupported_rss_sampling_platform_is_rejected(monkeypatch) -> None:
    module = _tool()
    monkeypatch.setattr(module.os, "getppid", lambda: 123)
    with pytest.raises(module.EvidenceError, match="unsupported"):
        module._instantaneous_rss_bytes(123, "freebsd")


def test_a_changed_sampler_parent_is_rejected_before_the_counter(monkeypatch) -> None:
    module = _tool()
    monkeypatch.setattr(module.os, "getppid", lambda: 456)
    with pytest.raises(module.EvidenceError, match="not the live sampler parent"):
        module._instantaneous_rss_bytes(123, "linux")


def _ready_record(module: Any, target_pid: int = 123) -> dict[str, Any]:
    return {
        "status": "READY",
        "target_pid": target_pid,
        "sampling_interval_ns": module.RSS_SAMPLING_INTERVAL_NS,
        "baseline_rss_bytes": 1000,
    }


def _result_record(module: Any, target_pid: int = 123) -> dict[str, Any]:
    return {
        "status": "RESULT",
        "target_pid": target_pid,
        "sampling_interval_ns": module.RSS_SAMPLING_INTERVAL_NS,
        "baseline_rss_bytes": 1000,
        "peak_rss_bytes": 1300,
        "final_rss_bytes": 1100,
        "sample_count": 2,
        "measured_host_peak_bytes": 300,
    }


def _install_scripted_sampler(
    monkeypatch,
    module: Any,
    body: str,
    *,
    before_ready: str = "",
) -> list[subprocess.Popen[bytes]]:
    """Replace only the sampler child with one bounded protocol script."""
    real_popen = subprocess.Popen
    script = f"""
import json
import os
import sys
import time

def emit(value):
    sys.stdout.buffer.write(
        json.dumps(value, sort_keys=True, separators=(\",\", \":\")).encode(\"utf-8\")
        + b\"\\n\"
    )
    sys.stdout.buffer.flush()

target = os.getppid()
interval = 10_000_000
baseline = 1000
{before_ready}
emit({{
    \"status\": \"READY\",
    \"target_pid\": target,
    \"sampling_interval_ns\": interval,
    \"baseline_rss_bytes\": baseline,
}})
result = {{
    \"status\": \"RESULT\",
    \"target_pid\": target,
    \"sampling_interval_ns\": interval,
    \"baseline_rss_bytes\": baseline,
    \"peak_rss_bytes\": 1300,
    \"final_rss_bytes\": 1100,
    \"sample_count\": 2,
    \"measured_host_peak_bytes\": 300,
}}
{body}
"""
    processes: list[subprocess.Popen[bytes]] = []

    def factory(_argv: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        process = real_popen([sys.executable, "-c", script], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(module.subprocess, "Popen", factory)
    return processes


def _assert_sampler_processes_reaped(processes: list[subprocess.Popen[bytes]]) -> None:
    assert processes
    assert all(process.poll() is not None for process in processes)


def test_sampler_protocol_records_require_exact_canonical_json() -> None:
    module = _tool()
    ready = _ready_record(module)
    payload = module.canonical_json(ready)
    assert module._protocol_record(payload, module.RSS_READY_KEYS, "READY") == ready
    with pytest.raises(module.EvidenceError, match="not canonical"):
        module._protocol_record(
            json.dumps(ready).encode("utf-8"), module.RSS_READY_KEYS, "READY"
        )
    ready["extra"] = 1
    with pytest.raises(module.EvidenceError, match="exactly"):
        module._protocol_record(
            module.canonical_json(ready), module.RSS_READY_KEYS, "READY"
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_pid", True, "exact JSON integer"),
        ("sampling_interval_ns", 1, "fixed 10 ms"),
        ("baseline_rss_bytes", 999, "does not match READY"),
        ("peak_rss_bytes", 999, "include the baseline and final"),
        ("final_rss_bytes", 1400, "include the baseline and final"),
        ("sample_count", 1, "at least 2"),
        ("measured_host_peak_bytes", 299, "peak minus baseline"),
    ],
)
def test_sampler_result_protocol_rejects_inconsistent_values(
    field: str, value: Any, message: str
) -> None:
    module = _tool()
    result = _result_record(module)
    result[field] = value
    with pytest.raises(module.EvidenceError, match=message):
        module._validate_result_record(result, 123, 1000)


def test_sampler_ready_protocol_rejects_pid_and_boolean_integers() -> None:
    module = _tool()
    ready = _ready_record(module)
    ready["target_pid"] = True
    with pytest.raises(module.EvidenceError, match="exact JSON integer"):
        module._validate_ready_record(ready, 123)
    ready = _ready_record(module, 124)
    with pytest.raises(module.EvidenceError, match="does not match parent"):
        module._validate_ready_record(ready, 123)


def test_sampler_line_timeout_fails_closed() -> None:
    module = _tool()
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            with pytest.raises(module.EvidenceError, match="READY timed out"):
                module._protocol_line(stream, 0.01, "READY")
    finally:
        os.close(write_fd)


def test_a_partial_protocol_line_cannot_defeat_the_single_deadline() -> None:
    module = _tool()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,time; os.write(1, b'{'); time.sleep(10)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    started = time.monotonic()
    try:
        assert process.stdout is not None
        with pytest.raises(module.EvidenceError, match="READY timed out"):
            module._protocol_line(process.stdout, 0.05, "READY")
        assert time.monotonic() - started < 0.5
    finally:
        module._cleanup_sampler(process)
    assert process.poll() is not None


class _CleanupStream(io.BytesIO):
    pass


class _StubbornSampler:
    def __init__(self) -> None:
        self.stdin = _CleanupStream()
        self.stdout = _CleanupStream()
        self.stderr = _CleanupStream()
        self.terminated = False
        self.killed = False
        self.waits = 0

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        self.waits += 1
        if timeout is not None:
            raise subprocess.TimeoutExpired("sampler", timeout)
        return -9


def test_sampler_cleanup_terminates_kills_reaps_and_closes_every_pipe() -> None:
    module = _tool()
    process = _StubbornSampler()
    module._cleanup_sampler(process)
    assert process.terminated and process.killed and process.waits == 2
    assert process.stdin.closed and process.stdout.closed and process.stderr.closed


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "sys.stdin.buffer.read(); "
            "sys.stdout.buffer.write(b'{\\n'); sys.stdout.buffer.flush()",
            "RESULT is not JSON",
        ),
        ("sys.stdin.buffer.read()", "RESULT line is malformed"),
    ],
)
def test_malformed_or_missing_result_fails_closed_and_reaps(
    monkeypatch, body: str, message: str
) -> None:
    module = _tool()
    processes = _install_scripted_sampler(monkeypatch, module, body)
    with pytest.raises(module.EvidenceError, match=message):
        module._measure_with_process_rss(lambda: "solved")
    _assert_sampler_processes_reaped(processes)


def test_extra_sampler_stdout_fails_closed_and_reaps(monkeypatch) -> None:
    module = _tool()
    processes = _install_scripted_sampler(
        monkeypatch,
        module,
        "sys.stdin.buffer.read(); emit(result); "
        "sys.stdout.buffer.write(b'extra\\n'); sys.stdout.buffer.flush()",
    )
    with pytest.raises(module.EvidenceError, match="extra stdout"):
        module._measure_with_process_rss(lambda: "solved")
    _assert_sampler_processes_reaped(processes)


def test_nonempty_sampler_stderr_fails_closed_and_reaps(monkeypatch) -> None:
    module = _tool()
    processes = _install_scripted_sampler(
        monkeypatch,
        module,
        "sys.stdin.buffer.read(); emit(result); "
        "sys.stderr.buffer.write(b'bad stderr'); sys.stderr.buffer.flush()",
    )
    with pytest.raises(module.EvidenceError, match="emitted stderr"):
        module._measure_with_process_rss(lambda: "solved")
    _assert_sampler_processes_reaped(processes)


def test_nonzero_sampler_exit_fails_closed_and_reaps(monkeypatch) -> None:
    module = _tool()
    processes = _install_scripted_sampler(
        monkeypatch,
        module,
        "sys.stdin.buffer.read(); emit(result); raise SystemExit(3)",
    )
    with pytest.raises(module.EvidenceError, match="did not exit zero"):
        module._measure_with_process_rss(lambda: "solved")
    _assert_sampler_processes_reaped(processes)


def test_result_timeout_terminates_and_reaps_the_sampler(monkeypatch) -> None:
    module = _tool()
    monkeypatch.setattr(module, "RSS_RESULT_TIMEOUT_SECONDS", 0.05)
    processes = _install_scripted_sampler(
        monkeypatch,
        module,
        "sys.stdin.buffer.read(); time.sleep(10)",
    )
    started = time.monotonic()
    with pytest.raises(module.EvidenceError, match="RESULT timed out"):
        module._measure_with_process_rss(lambda: "solved")
    assert time.monotonic() - started < 0.75
    _assert_sampler_processes_reaped(processes)


def test_clean_exit_timeout_terminates_and_reaps_the_sampler(monkeypatch) -> None:
    module = _tool()
    monkeypatch.setattr(module, "RSS_RESULT_TIMEOUT_SECONDS", 0.05)
    processes = _install_scripted_sampler(
        monkeypatch,
        module,
        "sys.stdin.buffer.read(); emit(result); time.sleep(10)",
    )
    started = time.monotonic()
    with pytest.raises(module.EvidenceError, match="clean exit timed out"):
        module._measure_with_process_rss(lambda: "solved")
    assert time.monotonic() - started < 0.75
    _assert_sampler_processes_reaped(processes)


def test_early_sampler_input_close_is_translated_and_reaped(monkeypatch) -> None:
    module = _tool()
    processes = _install_scripted_sampler(
        monkeypatch,
        module,
        "time.sleep(10)",
        before_ready="os.close(0)",
    )
    with pytest.raises(module.EvidenceError, match="STOP pipe failed") as excinfo:
        module._measure_with_process_rss(lambda: "solved")
    assert isinstance(excinfo.value.__cause__, BrokenPipeError)
    _assert_sampler_processes_reaped(processes)


def test_primary_solver_exception_survives_a_cleanup_exception(monkeypatch) -> None:
    module = _tool()
    processes = _install_scripted_sampler(
        monkeypatch,
        module,
        "time.sleep(10)",
    )
    real_cleanup = module._cleanup_sampler

    def cleanup_then_fail(process: Any) -> None:
        real_cleanup(process)
        raise RuntimeError("cleanup failed too")

    monkeypatch.setattr(module, "_cleanup_sampler", cleanup_then_fail)

    class SolverFailure(RuntimeError):
        pass

    with pytest.raises(SolverFailure, match="primary solver failure") as excinfo:
        module._measure_with_process_rss(
            lambda: (_ for _ in ()).throw(SolverFailure("primary solver failure"))
        )
    assert any("cleanup failed too" in note for note in excinfo.value.__notes__)
    _assert_sampler_processes_reaped(processes)


def test_cleanup_failure_on_success_still_fails_closed(monkeypatch) -> None:
    module = _tool()
    processes = _install_scripted_sampler(
        monkeypatch,
        module,
        "sys.stdin.buffer.read(); emit(result)",
    )
    real_cleanup = module._cleanup_sampler

    def cleanup_then_fail(process: Any) -> None:
        real_cleanup(process)
        raise RuntimeError("cleanup failed after success")

    monkeypatch.setattr(module, "_cleanup_sampler", cleanup_then_fail)
    with pytest.raises(RuntimeError, match="cleanup failed after success"):
        module._measure_with_process_rss(lambda: "solved")
    _assert_sampler_processes_reaped(processes)


def test_a_supported_host_sampler_child_smoke_test_has_no_extra_protocol_bytes() -> (
    None
):
    module = _tool()
    outcome, measured = module._measure_with_process_rss(lambda: sum(range(10_000)))
    assert outcome == sum(range(10_000))
    assert type(measured) is int and measured >= 0


def test_the_sampler_uses_no_forbidden_or_in_process_memory_instrument() -> None:
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    for forbidden in (
        "tracemalloc",
        "ru_maxrss",
        "resource.getrusage",
        "psutil",
        "threading",
    ):
        assert forbidden not in source, forbidden
    assert '"_sample-rss"' in source
    assert "subprocess.Popen(" in source
    assert "RSS_SAMPLING_INTERVAL_NS = 10_000_000" in source


def test_each_fixture_group_keeps_one_separate_untimed_whole_solver_measurement() -> (
    None
):
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    measure_group = source[source.index("def _measure_group(") :]
    measure_group = measure_group[: measure_group.index("def _dense_invariance_row")]
    assert measure_group.count("_measure_with_process_rss(") == 1
    workload_rows = source[source.index("def _workload_rows(") :]
    workload_rows = workload_rows[: workload_rows.index("def build_phase3_evidence")]
    assert "for fixture_id in PERFORMANCE_FIXTURES:" in workload_rows
    assert "for backend in BACKENDS:" in workload_rows


def test_a_solver_exception_is_propagated_after_sampler_cleanup() -> None:
    module = _tool()

    class SolverFailure(RuntimeError):
        pass

    def fail() -> None:
        raise SolverFailure("solver failed")

    with pytest.raises(SolverFailure, match="solver failed"):
        module._measure_with_process_rss(fail)


def test_the_fresh_red_commit_is_the_commit_containing_the_supplement() -> None:
    module = _tool()
    red_commit = module._red_commit_sha()
    assert GIT_SHA.fullmatch(red_commit)
    changed = _git("diff-tree", "--no-commit-id", "--name-only", "-r", red_commit)
    assert POST_SOURCE_RED_RECORD in changed.splitlines()


def test_the_generator_authenticates_and_joins_both_red_records() -> None:
    module = _tool()
    reference = module._red_failure_record_reference(module._red_commit_sha())
    historical_raw = (REPOSITORY_ROOT / RED_RECORD).read_bytes()
    post_source_raw = (REPOSITORY_ROOT / POST_SOURCE_RED_RECORD).read_bytes()
    assert hashlib.sha256(historical_raw).hexdigest() == (HISTORICAL_RED_RECORD_SHA256)
    assert reference == {
        "path": RED_RECORD,
        "sha256": HISTORICAL_RED_RECORD_SHA256,
        "schema_version": RED_RECORD_SCHEMA,
        "pre_fix_source_sha": json.loads(historical_raw)["pre_fix_source_sha"],
        "validated": True,
        "post_source_delta": {
            "path": POST_SOURCE_RED_RECORD,
            "sha256": hashlib.sha256(post_source_raw).hexdigest(),
            "schema_version": POST_SOURCE_RED_RECORD_SCHEMA,
            "pre_fix_source_sha": POST_SOURCE_PRE_FIX_SHA,
            "validated": True,
        },
    }


def test_the_artifact_validator_authenticates_the_fresh_r3_and_both_inputs() -> None:
    module = _tool()
    document = _synthetic_document(module)
    red_commit = module._red_commit_sha()
    document["red_commit_sha"] = red_commit
    document["red_failure_record"] = module._red_failure_record_reference(red_commit)
    module.validate_evidence_artifact(document)
    document["red_failure_record"]["post_source_delta"]["sha256"] = SIXTY_FOUR
    with pytest.raises(module.EvidenceError, match="authenticate and join"):
        module.validate_evidence_artifact(document)


def test_the_generator_rejects_a_supplement_not_contained_in_fresh_r3(
    monkeypatch,
) -> None:
    module = _tool()
    real = module._git_blob
    monkeypatch.setattr(
        module,
        "_git_blob",
        lambda commit, path: (
            b"forged" if path == POST_SOURCE_RED_RECORD else real(commit, path)
        ),
    )
    with pytest.raises(module.EvidenceError, match="does not contain"):
        module._red_failure_record_reference(module._red_commit_sha())


# ---------------------------------------------------------------------------
# Synthetic strict schema and digest fixtures -- the evidence envelope
# ---------------------------------------------------------------------------


def test_the_synthetic_envelope_satisfies_every_section_14_2_rule() -> None:
    """The fixture is a positive control for every rejection below."""
    module = _tool()
    envelope = module.validate_evidence_document(_synthetic_document(module))
    assert set(envelope) == set(ENVELOPE_KEYS)
    assert set(envelope["results"]) == set(RESULT_KEYS)


def test_the_m3_red_join_has_six_outer_and_five_nested_keys() -> None:
    module = _tool()
    document = _synthetic_document(module)
    red = document["red_failure_record"]
    assert set(red) == {
        "path",
        "sha256",
        "schema_version",
        "pre_fix_source_sha",
        "validated",
        "post_source_delta",
    }
    assert set(red["post_source_delta"]) == {
        "path",
        "sha256",
        "schema_version",
        "pre_fix_source_sha",
        "validated",
    }
    module.validate_evidence_document(document)


def test_a_missing_or_extra_outer_red_join_key_is_rejected() -> None:
    module = _tool()
    missing = _synthetic_document(module)
    missing["red_failure_record"].pop("post_source_delta")
    assert "post_source_delta" in _rejects(module, missing)
    extra = _synthetic_document(module)
    extra["red_failure_record"]["extra"] = True
    assert "unknown ['extra']" in _rejects(module, extra)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("path", RED_RECORD, "correction-24 record"),
        ("schema_version", RED_RECORD_SCHEMA, "correction-24 schema"),
        ("pre_fix_source_sha", FORTY, "superseded a61526d6"),
        ("validated", False, "must be true"),
    ],
)
def test_a_forged_post_source_red_reference_is_rejected(
    field: str, value: Any, message: str
) -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["red_failure_record"]["post_source_delta"][field] = value
    assert message in _rejects(module, document)


def test_an_extra_post_source_red_reference_field_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["red_failure_record"]["post_source_delta"]["extra"] = True
    assert "exactly" in _rejects(module, document)


@pytest.mark.parametrize("key", ENVELOPE_KEYS)
def test_a_missing_top_level_key_is_rejected(key: str) -> None:
    module = _tool()
    document = _synthetic_document(module)
    document.pop(key)
    assert "evidence envelope" in _rejects(module, document)


def test_an_unknown_top_level_key_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["extra"] = 1
    assert "evidence envelope" in _rejects(module, document)


@pytest.mark.parametrize("key", RESULT_KEYS)
def test_a_missing_results_key_is_rejected(key: str) -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["results"].pop(key)
    assert "results" in _rejects(module, document)


def test_an_m1_or_m2_results_shape_is_rejected_in_m3() -> None:
    """M3 has its own ``results`` key set; a sibling phase's shape is refused."""
    module = _tool()
    document = _synthetic_document(module)
    document["results"] = {
        "frame_certificate_cases": [],
        "polarization_cases": [],
        "sky_component_cases": [],
        "direct_convergence_cases": [],
        "truncation_cases": [],
        "backend_parity_cases": [],
        "memory_cases": [],
        "capability_cases": [],
        "rejection_cases": [],
    }
    assert "results" in _rejects(module, document)


def test_a_non_null_evidence_commit_sha_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["evidence_commit_sha"] = FORTY
    assert "evidence_commit_sha must be JSON null" in _rejects(module, document)


def test_a_reworded_self_reference_reason_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["evidence_commit_sha_reason"] = "self reference"
    assert "self-reference reason" in _rejects(module, document)


def test_a_superseded_red_design_sha_is_accepted_not_equated() -> None:
    """Section 13.7: ``design_sha`` and the ``R3`` record's may differ."""
    module = _tool()
    document = _synthetic_document(module)
    assert document["design_sha"] != document["red_commit_sha"]
    module.validate_evidence_document(document)
    document["design_sha"] = document["red_commit_sha"]
    module.validate_evidence_document(document)


def test_an_orphan_fixture_input_row_is_rejected() -> None:
    """Section 14.0: the row set equals the phase's non-rejection fixture IDs."""
    module = _tool()
    document = _synthetic_document(module)
    rows = document["source_identities"]["fixture_input_rows"]
    manifest = {"schema_version": "radiosim.mmode-input-identity.v1", "fixture_id": "x"}
    rows.append(
        {
            "fixture_id": "zz_orphan",
            "input_identity_manifest": manifest,
            "input_identity_sha256": _object_digest(
                "radiosim.mmode-input-identity.v1", manifest
            ),
        }
    )
    document["source_identities"]["input_identity_set_sha256"] = _object_digest(
        "radiosim.sci004-phase-input-set.v1", rows
    )
    assert "no orphan" in _rejects(module, document)


def test_a_missing_fixture_input_row_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    rows = document["source_identities"]["fixture_input_rows"][:-1]
    document["source_identities"]["fixture_input_rows"] = rows
    document["source_identities"]["input_identity_set_sha256"] = _object_digest(
        "radiosim.sci004-phase-input-set.v1", rows
    )
    assert "no orphan" in _rejects(module, document)


def test_a_tampered_input_identity_manifest_breaks_its_digest() -> None:
    module = _tool()
    document = _synthetic_document(module)
    rows = document["source_identities"]["fixture_input_rows"]
    rows[0]["input_identity_manifest"]["fixture_id"] = "tampered"
    assert "must rebuild from its manifest" in _rejects(module, document)


def test_a_tampered_input_identity_set_digest_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["source_identities"]["input_identity_set_sha256"] = SIXTY_FOUR
    assert "input_identity_set_sha256" in _rejects(module, document)


def test_an_output_row_whose_reader_lost_the_solver_snapshot_is_rejected() -> None:
    """Section 10: "reader round trips must ... authenticate the snapshot"."""
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["output_cases"][0]["read_solver_sha256"] = SIXTY_FOUR
    assert "reconstruct the written solver snapshot" in _rejects(module, document)


def test_a_lossless_output_row_that_lost_the_cube_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    rows = document["results"]["output_cases"]
    lossless = next(row for row in rows if row["format"] in LOSSLESS_CUBE_FORMATS)
    lossless["read_cube_sha256"] = SIXTY_FOUR
    assert "must preserve the cube" in _rejects(module, document)


def test_a_narrowing_output_row_restating_the_written_cube_is_rejected() -> None:
    """A read identity copied from the written one describes no round trip."""
    module = _tool()
    document = _synthetic_document(module)
    rows = document["results"]["output_cases"]
    narrowing = next(row for row in rows if row["format"] not in LOSSLESS_CUBE_FORMATS)
    narrowing["read_cube_sha256"] = narrowing["written_cube_sha256"]
    assert "did not happen" in _rejects(module, document)


def test_the_output_rows_must_cover_the_three_reader_formats_in_order() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["output_cases"].reverse()
    assert "in order" in _rejects(module, document)


def test_an_output_row_for_a_second_fixture_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["output_cases"][1]["fixture_id"] = "mmode_point_stokes_i"
    assert "one fixture" in _rejects(module, document)


def test_an_output_row_outside_the_phase_input_set_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    for row in document["results"]["output_cases"]:
        row["fixture_id"] = "not_a_family"
    assert "join the phase input set" in _rejects(module, document)


def test_the_fingerprint_rows_are_the_four_amended_families_in_order() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["fingerprint_rows"].reverse()
    assert "amended family order" in _rejects(module, document)


def test_a_removed_fingerprint_family_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["fingerprint_rows"] = document["results"]["fingerprint_rows"][
        :3
    ]
    assert "amended family order" in _rejects(module, document)


def test_a_duplicate_family_cell_dispatch_tuple_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    rows = document["results"]["ci_artifacts"]
    rows.insert(1, copy.deepcopy(rows[0]))
    assert "duplicate family/cell/dispatch tuple" in _rejects(module, document)


def test_a_ci_row_verdict_other_than_the_observation_set_is_rejected() -> None:
    """Section 11: a family pin is an observation set, never a bare digest."""
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["ci_artifacts"][0]["ci001_verdict"] = "green"
    assert "accepted observation set" in _rejects(module, document)


def test_a_performance_record_workload_count_other_than_nine_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["performance_record"]["workload_count"] = 6
    assert "must be nine" in _rejects(module, document)


def test_reordered_workload_identities_are_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["performance_record"]["workload_identities"].reverse()
    assert "in Section 11 order" in _rejects(module, document)


def test_a_performance_record_outside_the_retained_directory_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["performance_record"]["path"] = "output/benchmarks/x.json"
    assert "retained host-bound path" in _rejects(module, document)


def test_an_unauthenticated_performance_record_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["performance_record"]["authenticated"] = False
    assert "authenticated must be true" in _rejects(module, document)


def test_a_release_scan_that_reports_sci004_done_is_rejected() -> None:
    """Section 14.3: ``A3`` requires a scan that still reports ROADMAP."""
    module = _tool()
    document = _synthetic_document(module)
    scan = document["results"]["release_scan_cases"][0]
    scan["roadmap_occurrences"] = 0
    scan["done_occurrences"] = 1
    scan["expected_counts"]["roadmap_occurrences"] = 0
    scan["expected_counts"]["done_occurrences"] = 1
    assert "still report SCI-004 as ROADMAP" in _rejects(module, document)


def test_a_release_scan_finding_an_unsupported_claim_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    scan = document["results"]["release_scan_cases"][0]
    scan["unsupported_claim_occurrences"] = 1
    scan["expected_counts"]["unsupported_claim_occurrences"] = 1
    assert "no unsupported claim" in _rejects(module, document)


def test_a_release_scan_whose_observed_count_differs_from_expected_is_rejected() -> (
    None
):
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["release_scan_cases"][0]["roadmap_occurrences"] = 2
    assert "must equal its expected count" in _rejects(module, document)


def test_a_rejection_row_that_allocated_first_is_rejected() -> None:
    """Section 8: the two public-path refusals precede any solver work."""
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["rejection_cases"][0]["allocation_started"] = True
    assert "refusal precedes any work" in _rejects(module, document)


def test_a_rejection_row_that_created_an_output_path_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["rejection_cases"][0]["output_path_created"] = True
    assert "output_path_created" in _rejects(module, document)


def test_a_non_zero_command_exit_code_is_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["commands"][0]["exit_code"] = 1
    assert "commands[0]" in _rejects(module, document)


def test_unsorted_claim_arrays_are_rejected() -> None:
    module = _tool()
    document = _synthetic_document(module)
    document["claims_not_licensed"] = list(reversed(document["claims_not_licensed"]))
    assert "sorted and unique" in _rejects(module, document)


@pytest.mark.parametrize("topic", DEFERRAL_TOPICS)
def test_a_missing_deferral_claim_is_rejected(topic: str) -> None:
    """The three deferrals the accepted corrections require, one at a time."""
    module = _tool()
    document = _synthetic_document(module)
    document["claims_not_licensed"] = sorted(
        literal
        for literal in document["claims_not_licensed"]
        if not literal.startswith(topic + ":")
    )
    detail = _rejects(module, document)
    assert "claims_not_licensed" in detail


def test_the_declared_claims_carry_all_three_deferrals() -> None:
    module = _tool()
    for topic in DEFERRAL_TOPICS:
        assert any(
            literal.startswith(topic + ":") for literal in module.CLAIMS_NOT_LICENSED
        ), topic


# ---------------------------------------------------------------------------
# Synthetic strict schema fixtures -- the Section 11 performance record
# ---------------------------------------------------------------------------


def test_the_synthetic_performance_record_satisfies_every_section_11_rule() -> None:
    module = _tool()
    record = module.validate_performance_document(_synthetic_performance_document())
    assert set(record) == set(BENCHMARK_TOP_LEVEL_KEYS)
    assert len(record["workloads"]) == 9


@pytest.mark.parametrize("key", BENCHMARK_TOP_LEVEL_KEYS)
def test_a_missing_benchmark_top_level_key_is_rejected(key: str) -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record.pop(key)
    assert "benchmark record" in _rejects(module, record, performance=True)


@pytest.mark.parametrize("key", PROVENANCE_KEYS)
def test_a_missing_provenance_key_is_rejected(key: str) -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["provenance"].pop(key)
    assert "provenance" in _rejects(module, record, performance=True)


def test_a_perf001_schema_literal_is_rejected() -> None:
    """Section 11: the SCI-004 record deliberately defines its own schema."""
    module = _tool()
    record = _synthetic_performance_document()
    record["schema_version"] = "radiosim.benchmark.perf001.v1"
    assert "schema literal" in _rejects(module, record, performance=True)


def test_a_provenance_workload_count_other_than_nine_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["provenance"]["workload_count"] = 3
    assert "exactly nine" in _rejects(module, record, performance=True)


def test_a_dirty_provenance_tree_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["provenance"]["working_tree_clean"] = False
    assert "working_tree_clean" in _rejects(module, record, performance=True)


def test_a_non_default_pixi_environment_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["provenance"]["pixi_environment"] = "gpu"
    assert "pixi_environment" in _rejects(module, record, performance=True)


def test_a_wrong_transform_execution_policy_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["provenance"]["transform_execution_policy"] = "backend_native_v1"
    assert "Section 9 literal" in _rejects(module, record, performance=True)


def test_a_reordered_workload_product_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"].reverse()
    assert "Cartesian product" in _rejects(module, record, performance=True)


def test_a_removed_workload_row_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"] = record["workloads"][:-1]
    assert "Cartesian product" in _rejects(module, record, performance=True)


@pytest.mark.parametrize("key", WORKLOAD_KEYS)
def test_a_missing_workload_key_is_rejected(key: str) -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][0].pop(key)
    detail = _rejects(module, record, performance=True)
    # ``workload_id`` is read before the row schema, so its absence is caught by
    # the ordered-product check rather than the per-row key set; either refusal
    # is a refusal, and both are named here rather than weakened to a substring.
    assert "workloads[0]" in detail or "Cartesian product" in detail


def test_a_row_claiming_a_backend_dense_execution_is_rejected() -> None:
    """Section 11: ``dense_execution`` is ``numpy_host_v1`` on every row."""
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][4]["dense_execution"] = "jax_device_v1"
    assert "numpy_host_v1" in _rejects(module, record, performance=True)


def test_a_row_naming_a_non_cpu_device_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][4]["device_kind"] = "gpu"
    assert "device_kind" in _rejects(module, record, performance=True)


def test_a_healpix_or_hybrid_representation_row_is_rejected() -> None:
    """Section 11: the sky representation is ``point`` for all three groups."""
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][0]["sky_representation"] = "healpix"
    assert "must be point" in _rejects(module, record, performance=True)


def test_a_non_zero_healpix_pixel_count_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][0]["n_healpix_pixels"] = 12
    assert "absent representation" in _rejects(module, record, performance=True)


def test_a_mismatched_backend_runtime_pair_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][4]["backend_runtime"]["kernel_runtime"] = "NumPy"
    assert "must name the jax pair" in _rejects(module, record, performance=True)


def test_the_shared_series_are_carried_identically_by_every_row_in_a_group() -> None:
    """Section 11: the group measures once on the NumPy row."""
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][1]["timings"] = copy.deepcopy(record["workloads"][1]["timings"])
    record["workloads"][1]["timings"]["total"]["sample_seconds"] = [
        2.0
    ] * MINIMUM_SAMPLES
    detail = _rejects(module, record, performance=True)
    assert "shared timings" in detail


def test_a_group_row_with_its_own_memory_object_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][2]["memory"] = copy.deepcopy(record["workloads"][2]["memory"])
    record["workloads"][2]["memory"]["measured_host_peak_bytes"] = 1024
    assert "shared memory" in _rejects(module, record, performance=True)


def test_two_groups_sharing_one_input_identity_are_rejected() -> None:
    """Section 11: "distinct input identities across groups"."""
    module = _tool()
    record = _synthetic_performance_document()
    borrowed = record["workloads"][0]["input_identity_sha256"]
    for row in record["workloads"][3:6]:
        row["input_identity_sha256"] = borrowed
    assert "distinct across fixture groups" in _rejects(
        module, record, performance=True
    )


# --- the fused dense series -------------------------------------------------


def test_the_fused_dense_series_must_be_measured() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["timings"]["dense_contraction_and_synthesis"] = {
            "status": "not_measured",
            "reason": "no",
        }
    assert "must be measured" in _rejects(module, record, performance=True)


def test_a_row_series_split_into_the_kernel_stage_names_is_rejected() -> None:
    """Section 11: those two names "denote exactly the kernel-block stages"."""
    module = _tool()
    record = _synthetic_performance_document()
    timings = record["workloads"][0]["timings"]
    fused = timings.pop("dense_contraction_and_synthesis")
    for name in KERNEL_STAGE_NAMES:
        timings[name] = copy.deepcopy(fused)
    assert "timings" in _rejects(module, record, performance=True)


def test_a_total_smaller_than_the_sum_of_its_stages_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["timings"]["total"]["sample_seconds"] = [0.2] * MINIMUM_SAMPLES
    assert "not be smaller than the sum" in _rejects(module, record, performance=True)


def test_unequal_sample_cardinality_across_the_measured_series_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["timings"]["frame"]["sample_seconds"] = [0.1] * (MINIMUM_SAMPLES + 1)
    assert "one sample cardinality" in _rejects(module, record, performance=True)


def test_fewer_than_five_samples_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        for name in MEASURED_SERIES:
            row["timings"][name]["sample_seconds"] = [0.1] * (MINIMUM_SAMPLES - 1)
    assert "at least 5 samples" in _rejects(module, record, performance=True)


def test_a_not_measured_direct_reference_is_admitted() -> None:
    """Section 11 admits ``not_measured`` for the direct reference alone."""
    module = _tool()
    record = _synthetic_performance_document()
    assert (
        record["workloads"][0]["timings"]["direct_reference"]["status"]
        == "not_measured"
    )
    module.validate_performance_document(record)


def test_a_timing_series_with_a_null_field_is_rejected() -> None:
    """Section 11: "No timing-series field is nullable"."""
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["timings"]["host_transfer"] = {"status": "not_applicable", "reason": None}
    assert "reason must be non-empty" in _rejects(module, record, performance=True)


# --- the three kernel-block statuses ----------------------------------------


def test_a_numpy_row_carrying_a_measured_kernel_block_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][6]["kernel_backend_block"] = _kernel_block(
        POLARIZED_FIXTURE, "jax"
    )
    assert "NumPy row must be not_applicable" in _rejects(
        module, record, performance=True
    )


def test_a_scalar_group_claiming_measured_kernel_stages_is_rejected() -> None:
    """The scalar-table exception is a status, not an option."""
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][1]["kernel_backend_block"] = _kernel_block(
        POLARIZED_FIXTURE, "jax"
    )
    assert "not_applicable_scalar_table" in _rejects(module, record, performance=True)


def test_a_polarized_group_claiming_the_scalar_exception_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][7]["kernel_backend_block"] = {
        "status": "not_applicable_scalar_table",
        "reason": KERNEL_SCALAR_REASON,
    }
    assert "must be measured" in _rejects(module, record, performance=True)


def test_a_scalar_exception_whose_reason_omits_the_kernel_contract_is_rejected() -> (
    None
):
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][1]["kernel_backend_block"]["reason"] = "not measured"
    assert "four-field kernel contract" in _rejects(module, record, performance=True)


@pytest.mark.parametrize("stage", KERNEL_STAGE_NAMES)
def test_a_measured_kernel_block_missing_a_stage_is_rejected(stage: str) -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][7]["kernel_backend_block"].pop(stage)
    assert "kernel_backend_block" in _rejects(module, record, performance=True)


@pytest.mark.parametrize("key", KERNEL_STAGE_KEYS)
def test_a_kernel_stage_missing_a_field_is_rejected(key: str) -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][7]["kernel_backend_block"]["per_m_contraction"].pop(key)
    assert "per_m_contraction" in _rejects(module, record, performance=True)


def test_a_kernel_stage_with_the_wrong_synchronization_method_is_rejected() -> None:
    """Section 11: the row's own method, applied to exactly those kernel calls."""
    module = _tool()
    record = _synthetic_performance_document()
    stage = record["workloads"][7]["kernel_backend_block"]["per_m_contraction"]
    stage["synchronization_method"] = "numpy_eager_v1"
    assert "must be the jax method" in _rejects(module, record, performance=True)


def test_a_dask_kernel_stage_borrowing_the_jax_method_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    stage = record["workloads"][8]["kernel_backend_block"]["synthesis"]
    stage["synchronization_method"] = KERNEL_SYNCHRONIZATION_METHODS["jax"]
    assert "must be the dask method" in _rejects(module, record, performance=True)


@pytest.mark.parametrize("key", STAGE_COMPARISON_KEYS)
def test_a_stage_comparison_missing_a_field_is_rejected(key: str) -> None:
    """Section 11's eleven-field ``stage_comparison``."""
    module = _tool()
    record = _synthetic_performance_document()
    comparison = record["workloads"][7]["kernel_backend_block"]["per_m_contraction"][
        "stage_comparison"
    ]
    comparison.pop(key)
    assert "stage_comparison" in _rejects(module, record, performance=True)


def test_a_stage_comparison_under_a_foreign_predicate_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    comparison = record["workloads"][7]["kernel_backend_block"]["synthesis"][
        "stage_comparison"
    ]
    comparison["predicate_id"] = "sci004_backend_complex64.v1"
    assert "predicate_id must be" in _rejects(module, record, performance=True)


def test_the_kernel_reference_stage_is_computed_on_the_numpy_backend() -> None:
    """Section 11: the reference is "the NumPy kernel output on identical inputs".

    "Never a self-comparison" is a claim about the reference's *provenance*, and
    provenance is not decidable from the retained digests: two backends that
    agree exactly publish identical stage identities, which is the expected
    outcome here, so a rule requiring the two digests to differ would forbid
    exact agreement.  It is enforced where it is decidable -- in the generator's
    tracked bytes, which compute the reference stage on the NumPy backend from
    the same block inputs the candidate receives.
    """
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    body = source[source.index("def _kernel_stage_block") :]
    body = body[: body.index("\ndef ")]
    assert 'reference_backend = get_backend("numpy")' in body
    assert "candidate_backend = get_backend(backend_name)" in body
    assert "backend=reference_backend" in body
    assert "backend=candidate_backend" in body
    # The candidate is never substituted for the reference.
    assert "reference_contraction = contract_per_m_block(" in body
    assert "reference_synthesis = synthesize_time_series(" in body


def test_exactly_agreeing_kernel_stages_are_admitted() -> None:
    """The expected CPU outcome -- zero deviation -- must not be refused."""
    module = _tool()
    record = _synthetic_performance_document()
    for backend_index in (7, 8):
        for stage_name in KERNEL_STAGE_NAMES:
            comparison = record["workloads"][backend_index]["kernel_backend_block"][
                stage_name
            ]["stage_comparison"]
            comparison["reference_stage_sha256"] = comparison["candidate_stage_sha256"]
    module.validate_performance_document(record)


def test_a_stage_relative_deviation_that_is_not_the_scaled_maximum_is_rejected() -> (
    None
):
    module = _tool()
    record = _synthetic_performance_document()
    comparison = record["workloads"][7]["kernel_backend_block"]["synthesis"][
        "stage_comparison"
    ]
    comparison["maximum_absolute_deviation_jy"] = 1e-13
    assert "over the reference scale" in _rejects(module, record, performance=True)


def test_a_widened_stage_rtol_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    comparison = record["workloads"][7]["kernel_backend_block"]["synthesis"][
        "stage_comparison"
    ]
    comparison["rtol"] = 1e-6
    assert "rtol must be exactly 1e-12" in _rejects(module, record, performance=True)


def test_a_failing_stage_comparison_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    comparison = record["workloads"][7]["kernel_backend_block"]["synthesis"][
        "stage_comparison"
    ]
    comparison["pass"] = False
    assert "pass must be true" in _rejects(module, record, performance=True)


def test_a_null_kernel_native_peak_claiming_the_measured_reason_is_rejected() -> None:
    """Section 11: the reason is exactly ``measured`` only for an integer peak."""
    module = _tool()
    record = _synthetic_performance_document()
    stage = record["workloads"][7]["kernel_backend_block"]["per_m_contraction"]
    stage["measured_native_peak_bytes_reason"] = "measured"
    assert "never measured" in _rejects(module, record, performance=True)


def test_a_null_shared_native_peak_claiming_the_measured_reason_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["memory"]["measured_native_peak_bytes_reason"] = "measured"
    assert "never measured" in _rejects(module, record, performance=True)


def test_a_kernel_stage_with_an_integer_native_peak_needs_a_real_method() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    stage = record["workloads"][7]["kernel_backend_block"]["per_m_contraction"]
    stage["measured_native_peak_bytes"] = 4096
    stage["measured_native_peak_bytes_reason"] = "measured"
    assert "requires a real method" in _rejects(module, record, performance=True)


def test_a_kernel_stage_may_carry_a_device_native_method() -> None:
    """Section 11: the backend-device methods appear only inside kernel blocks."""
    module = _tool()
    record = _synthetic_performance_document()
    stage = record["workloads"][7]["kernel_backend_block"]["per_m_contraction"]
    stage["measured_native_peak_bytes"] = 4096
    stage["measured_native_peak_bytes_reason"] = "measured"
    stage["native_measurement_method"] = "jax_device_memory_stats_v1"
    module.validate_performance_document(record)


# --- shared memory ----------------------------------------------------------


@pytest.mark.parametrize("method", DEVICE_NATIVE_METHODS)
def test_a_shared_memory_object_naming_a_device_method_is_rejected(method: str) -> None:
    """Section 11: "never a backend-device method" in the shared object."""
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["memory"]["native_measurement_method"] = method
    assert "never a backend-device method" in _rejects(module, record, performance=True)


def test_the_process_rss_method_is_rejected_in_the_native_field() -> None:
    """Correction #24: sampled RSS is host-only; native stays unavailable."""
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["memory"]["native_measurement_method"] = "process_rss_sampled_delta_v1"
        row["memory"]["measured_native_peak_bytes"] = 8192
        row["memory"]["measured_native_peak_bytes_reason"] = "measured"
    assert "must be unavailable" in _rejects(module, record, performance=True)


def test_an_estimate_below_the_measured_host_peak_is_admitted_when_declared() -> None:
    """Section 11 admits an honestly observed ``false`` coverage relation.

    Correction #24 does not keep
    ``measured_host_peak_bytes <= estimated_host_peak_bytes`` as a hard
    predicate; only the two budget inequalities gate.  The sampled relation is
    recomputed and retained as observed, and this fixture exercises false.
    """
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["memory"].update(_uncovered_memory())
    memory = module.validate_performance_document(record)["workloads"][0]["memory"]
    assert memory["estimate_covers_measured_host_peak"] is False
    assert memory["measured_host_peak_bytes"] > memory["estimated_host_peak_bytes"]
    assert tuple(memory["host_measurement_limitations"]) == (
        HOST_MEASUREMENT_LIMITATIONS
    )


def test_a_memory_row_missing_one_sampled_rss_limitation_is_rejected() -> None:
    """Correction #24 requires all four ruled limitations on every row."""
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        memory = _uncovered_memory()
        memory["host_measurement_limitations"] = list(HOST_MEASUREMENT_LIMITATIONS[1:])
        row["memory"].update(memory)
    assert "exactly the four sampled-RSS limitations" in _rejects(
        module, record, performance=True
    )


def test_a_reworded_sampled_rss_limitation_is_rejected() -> None:
    """A paraphrase cannot replace one of correction #24's four literals."""
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        memory = _uncovered_memory()
        memory["host_measurement_limitations"] = sorted(
            [*HOST_MEASUREMENT_LIMITATIONS[1:], "sampling can miss a peak"]
        )
        row["memory"].update(memory)
    assert "exactly the four sampled-RSS limitations" in _rejects(
        module, record, performance=True
    )


def test_a_coverage_boolean_disagreeing_with_the_measured_relation_is_rejected() -> (
    None
):
    """Section 11: the boolean is "retained as observed, never chosen"."""
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        memory = _uncovered_memory()
        # The values say the estimate does not cover the peak; the boolean lies.
        memory["estimate_covers_measured_host_peak"] = True
        row["memory"].update(memory)
    assert "recomputed from this row's own values" in _rejects(
        module, record, performance=True
    )


def test_a_true_coverage_boolean_on_a_covering_row_is_required() -> None:
    """The converse: a covering row may not declare itself uncovered."""
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["memory"]["estimate_covers_measured_host_peak"] = False
    assert "recomputed from this row's own values" in _rejects(
        module, record, performance=True
    )


def test_a_non_boolean_coverage_flag_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["memory"]["estimate_covers_measured_host_peak"] = 1
    assert "must be a JSON boolean" in _rejects(module, record, performance=True)


def test_a_measured_host_peak_above_the_working_memory_budget_is_rejected() -> None:
    """Section 11's first hard predicate, added by the same correction."""
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        memory = _uncovered_memory()
        memory["measured_host_peak_bytes"] = (1 << 30) + 1
        row["memory"].update(memory)
    assert "measured host peak must not exceed the working-memory budget" in _rejects(
        module, record, performance=True
    )


def test_an_estimate_above_the_working_memory_budget_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["memory"]["estimated_host_peak_bytes"] = (1 << 30) + 1
    assert "working-memory budget" in _rejects(module, record, performance=True)


def test_a_null_native_peak_whose_reason_is_absent_from_the_limitations_is_rejected() -> (
    None
):
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["memory"]["native_measurement_limitations"] = ["a different sentence"]
    assert "must also occur in the limitations" in _rejects(
        module, record, performance=True
    )


def test_a_wrong_measurement_scope_literal_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["memory"]["measurement_scope"] = "one_dense_pass_v1"
    assert "measurement_scope" in _rejects(module, record, performance=True)


# --- schedule ---------------------------------------------------------------


def test_a_schedule_digest_that_does_not_rebuild_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["resolved_block_dimensions"]["schedule_sha256"] = SIXTY_FOUR
    assert "must rebuild from the retained rows" in _rejects(
        module, record, performance=True
    )


def test_a_non_contiguous_block_index_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        rows = row["resolved_block_dimensions"]["schedule_rows"]
        rows[0]["block_index"] = 1
        row["resolved_block_dimensions"]["schedule_sha256"] = _object_digest(
            "radiosim.sci004.block-schedule.v1", rows
        )
    assert "contiguous from zero" in _rejects(module, record, performance=True)


def test_a_scheduled_block_count_that_is_not_the_row_count_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["resolved_block_dimensions"]["scheduled_block_count"] = 2
    assert "equal the row count" in _rejects(module, record, performance=True)


# --- the two fixed numerical predicates -------------------------------------


def test_a_direct_cell_count_that_is_not_k_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["direct_comparison"]["expected_cell_count"] = SYNTHETIC_CELLS + 1
    assert "must equal K" in _rejects(module, record, performance=True)


def test_a_widened_tier_1a_maximum_limit_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["direct_comparison"]["horizon_free_shell_max_limit_jy"] = 1e-4
    assert "1e-8*S_num + 1e-10" in _rejects(module, record, performance=True)


def test_a_widened_tier_1a_l2_limit_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["direct_comparison"]["horizon_free_shell_l2_limit"] = 1e-6
    assert "exactly 1e-8" in _rejects(module, record, performance=True)


def test_a_tier_1a_shell_above_its_fixed_limit_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["direct_comparison"]["horizon_free_shell_max_jy"] = 1e-3
    assert "tier-1a maximum predicate" in _rejects(module, record, performance=True)


def test_a_non_monotone_deficit_sequence_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["direct_comparison"]["deficit_max_half_jy"] = 8e-6
    assert "convergence ordering" in _rejects(module, record, performance=True)


def test_a_quarter_to_full_factor_below_two_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        direct = row["direct_comparison"]
        direct["deficit_max_jy"] = 3e-6
        direct["deficit_max_half_jy"] = 3.5e-6
        direct["deficit_max_quarter_jy"] = 4e-6
        direct["convergence_factor"] = 4e-6 / 3e-6
    assert "at least two" in _rejects(module, record, performance=True)


def test_a_convergence_factor_that_is_not_the_ratio_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["direct_comparison"]["convergence_factor"] = 3.0
    assert "quarter-to-final ratio" in _rejects(module, record, performance=True)


def test_a_direct_candidate_that_is_not_the_rows_cube_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["direct_comparison"]["candidate_cube_sha256"] = SIXTY_FOUR
    assert "row's retained cube" in _rejects(module, record, performance=True)


def test_a_failing_direct_comparison_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    for row in record["workloads"][:3]:
        row["direct_comparison"]["pass"] = False
    assert "pass must be true" in _rejects(module, record, performance=True)


def test_a_backend_comparison_referencing_another_group_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][4]["backend_comparison"]["reference_workload_id"] = (
        "mmode_single_scalar_mode:numpy:standard"
    )
    assert "its group's NumPy row" in _rejects(module, record, performance=True)


def test_a_widened_backend_atol_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][4]["backend_comparison"]["atol_jy"] = 1e-6
    assert "1e-12 times the reference scale" in _rejects(
        module, record, performance=True
    )


def test_a_row_claim_array_missing_the_end_to_end_literal_is_rejected() -> None:
    """The honest-backend-axis correction's sixth claim literal."""
    module = _tool()
    record = _synthetic_performance_document()
    record["workloads"][0]["claims_not_licensed"] = sorted(
        set(BENCHMARK_CLAIMS) - {"mmode_end_to_end_backend_execution"}
    )
    assert "exact six literals" in _rejects(module, record, performance=True)


# --- dense invariance -------------------------------------------------------


def test_a_dense_invariance_entry_per_group_in_fixture_order_is_required() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["dense_invariance"].reverse()
    assert "in fixture order" in _rejects(module, record, performance=True)


def test_a_non_identical_backend_cube_is_rejected() -> None:
    """Section 11 retains the measured bit-identity as fact, not as an option."""
    module = _tool()
    record = _synthetic_performance_document()
    record["dense_invariance"][1]["jax_cube_sha256"] = SIXTY_FOUR
    assert "bit-identical" in _rejects(module, record, performance=True)


def test_a_false_identical_flag_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    record["dense_invariance"][0]["identical"] = False
    assert "identical must be true" in _rejects(module, record, performance=True)


def test_a_dense_invariance_row_that_does_not_join_its_group_cube_is_rejected() -> None:
    module = _tool()
    record = _synthetic_performance_document()
    entry = record["dense_invariance"][2]
    for name in ("numpy_cube_sha256", "jax_cube_sha256", "dask_cube_sha256"):
        entry[name] = SIXTY_FOUR
    assert "join its group's retained cube identity" in _rejects(
        module, record, performance=True
    )


# ---------------------------------------------------------------------------
# Canonical encoding
# ---------------------------------------------------------------------------


def test_canonical_json_sorts_keys_and_emits_no_whitespace() -> None:
    module = _tool()
    assert module.canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


@pytest.mark.parametrize(
    ("value", "text"),
    [(1.0, "1"), (0.5, "0.5"), (1e-10, "1e-10"), (-0.0, "0"), (1e21, "1e+21")],
)
def test_canonical_numbers_use_the_ecmascript_spelling(value: float, text: str) -> None:
    module = _tool()
    assert module.canonical_json(value) == text.encode("ascii")


def test_canonical_json_forbids_nan_and_infinity() -> None:
    module = _tool()
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(module.EvidenceError):
            module.canonical_json(value)


def test_the_domain_digest_matches_its_printed_definition() -> None:
    """Section 14.0: ``D(d,p) = SHA256(d || NUL || U64(len(p)) || p)``."""
    module = _tool()
    assert module.domain_digest("x.v1", b"payload") == _domain_digest(
        "x.v1", b"payload"
    )


def test_a_distinct_domain_gives_a_distinct_digest() -> None:
    module = _tool()
    assert module.domain_digest("a.v1", b"p") != module.domain_digest("b.v1", b"p")


def test_the_f64be_encoding_is_the_big_endian_double() -> None:
    module = _tool()
    assert module.f64be(1.0) == _f64be(1.0) == "3ff0000000000000"


# ---------------------------------------------------------------------------
# E3-state commit shape
# ---------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    """Return the stdout of one hermetic ``git`` invocation in this repository."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
    )
    return completed.stdout


def _locate_evidence_commit() -> str:
    """Return the unique commit that introduced the phase evidence artifact."""
    introductions = _git(
        "log", "--diff-filter=A", "--format=%H", "HEAD", "--", ARTIFACT
    ).split()
    assert len(introductions) == 1, (
        f"{ARTIFACT} must be introduced by exactly one commit on HEAD's "
        f"ancestry; observed {introductions}"
    )
    located = introductions[0]
    assert GIT_SHA.fullmatch(located)
    return located


def _constant_spans(source: str) -> tuple[list[tuple[int, int]], list[list[Any]]]:
    """Return the token ranges of the four approved-constant assignments."""
    tokens = [
        token
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in (tokenize.ENCODING, tokenize.ENDMARKER)
    ]
    spans: list[tuple[int, int]] = []
    bodies: list[list[Any]] = []
    for index, token in enumerate(tokens):
        if (
            token.type != tokenize.NAME
            or token.string not in APPROVED_CONSTANT_NAMES
            or token.start[1] != 0
        ):
            continue
        stop = index
        while tokens[stop].type != tokenize.NEWLINE:
            stop += 1
        spans.append((index, stop + 1))
        bodies.append(tokens[index : stop + 1])
    assert len(spans) == len(APPROVED_CONSTANT_NAMES), (
        f"expected one assignment per approved constant; found {len(spans)}"
    )
    return spans, bodies


def _outside_spans(source: str) -> list[tuple[int, str]]:
    """Return the ``(type, string)`` token stream outside the four spans."""
    spans, _bodies = _constant_spans(source)
    tokens = [
        token
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in (tokenize.ENCODING, tokenize.ENDMARKER)
    ]
    excised = {index for start, stop in spans for index in range(start, stop)}
    return [
        (token.type, token.string)
        for index, token in enumerate(tokens)
        if index not in excised
    ]


def _assigned_literal(body: list[Any]) -> str:
    """Return the single value token of one approved-constant assignment."""
    values = [
        token
        for token in body
        if token.type in (tokenize.STRING, tokenize.NAME)
        and token.string not in (*APPROVED_CONSTANT_NAMES, "str", "None")
    ]
    names = [token for token in body if token.string == "None"]
    if not values:
        assert names, "an approved-constant assignment carries no value token"
        return "None"
    assert len(values) == 1, (
        "an approved-constant assignment must carry exactly one value token"
    )
    return values[0].string


def _e3_authorized_paths() -> frozenset[str]:
    """Return Section 13.5's exact four-path ``E3`` write authority."""
    assert APPROVED_PERFORMANCE_PATH is not None
    return frozenset(
        {ARTIFACT, REPRODUCTION, VALIDATOR, str(APPROVED_PERFORMANCE_PATH)}
    )


def test_the_artifact_introducing_commit_directly_parents_the_approved_source() -> None:
    """Section 14.2's ``E3`` ancestry clause, skipped until the constants flip."""
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_SOURCE_SHA is None:
        pytest.skip("the M3 evidence artifact is authorized at E3")
    located = _locate_evidence_commit()
    lineage = _git("rev-list", "--parents", "-n", "1", located).split()
    assert lineage[0] == located
    assert len(lineage) == 2, (
        f"the artifact-introducing commit {located} must be a non-merge commit "
        f"with exactly one parent; observed {lineage[1:]}"
    )
    assert lineage[1] == APPROVED_SOURCE_SHA, (
        f"the direct parent of {located} is {lineage[1]}, not the approved "
        f"source {APPROVED_SOURCE_SHA}"
    )
    payload = _git("show", f"{located}:{ARTIFACT}")
    assert (
        hashlib.sha256(payload.encode("utf-8")).hexdigest() == APPROVED_ARTIFACT_SHA256
    )


def test_the_e3_diff_writes_only_the_section_13_5_authorized_paths() -> None:
    """Section 13.5: the envelope, its record, this module, and one benchmark."""
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_PERFORMANCE_PATH is None:
        pytest.skip("the M3 evidence artifact is authorized at E3")
    located = _locate_evidence_commit()
    changed = set(
        _git("diff-tree", "--no-commit-id", "--name-only", "-r", located).split()
    )
    assert ARTIFACT in changed
    assert APPROVED_PERFORMANCE_PATH in changed, (
        "the E3 commit must add the authenticated performance record beside the "
        "envelope; a partial set is invalid"
    )
    unauthorized = sorted(changed - _e3_authorized_paths())
    assert not unauthorized, (
        f"the E3 commit {located} writes {unauthorized}, which Section 13.5 "
        f"does not authorize"
    )
    retained = sorted(
        path for path in changed if path.startswith(PERFORMANCE_DIRECTORY + "/")
    )
    assert retained == [APPROVED_PERFORMANCE_PATH], (
        "Section 13.5 authorizes exactly one new performance record"
    )
    record = _git("show", f"{located}:{REPRODUCTION}")
    assert record.startswith(REPRODUCTION_FRONT_MATTER), (
        "the reproduction record must open with Section 14.2's exact MyST front matter"
    )


def test_the_e3_diff_changes_only_the_four_approved_constant_assignments() -> None:
    """Section 14.2: this module's own ``E3`` diff is the four constants alone."""
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_SOURCE_SHA is None:
        pytest.skip("the M3 evidence artifact is authorized at E3")
    located = _locate_evidence_commit()
    parent = _git("rev-list", "--parents", "-n", "1", located).split()[1]
    before = _git("show", f"{parent}:{VALIDATOR}")
    after = _git("show", f"{located}:{VALIDATOR}")

    assert _outside_spans(before) == _outside_spans(after), (
        f"the E3 commit {located} changed this module outside the four approved "
        "constant assignments"
    )

    _spans_before, bodies_before = _constant_spans(before)
    _spans_after, bodies_after = _constant_spans(after)
    approved = (
        APPROVED_SOURCE_SHA,
        APPROVED_ARTIFACT_SHA256,
        APPROVED_PERFORMANCE_PATH,
        APPROVED_PERFORMANCE_SHA256,
    )
    for name, body_before, body_after, value in zip(
        APPROVED_CONSTANT_NAMES, bodies_before, bodies_after, approved, strict=True
    ):
        assert _assigned_literal(body_before) == "None", (
            f"{name} must be the null sentinel at the direct parent {parent}"
        )
        assert _assigned_literal(body_after) == f'"{value}"', (
            f"{name} at {located} is not the approved literal"
        )


def test_the_retained_artifact_authenticates_against_the_approved_constants() -> None:
    """Section 14.2's ``E3`` state, skipped until the constants are flipped."""
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_SOURCE_SHA is None:
        pytest.skip("the M3 evidence artifact is authorized at E3")
    payload = (REPOSITORY_ROOT / ARTIFACT).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == APPROVED_ARTIFACT_SHA256
    document = json.loads(payload.decode("utf-8"))
    module = _tool()
    module.validate_evidence_document(document)
    assert document["source_sha"] == APPROVED_SOURCE_SHA


def test_the_retained_performance_record_authenticates_and_joins_the_envelope() -> None:
    """Section 14.2: the envelope binds the record's raw canonical bytes."""
    if APPROVED_PERFORMANCE_SHA256 is None or APPROVED_PERFORMANCE_PATH is None:
        pytest.skip("the M3 performance record is authorized at E3")
    path = REPOSITORY_ROOT / APPROVED_PERFORMANCE_PATH
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == APPROVED_PERFORMANCE_SHA256
    module = _tool()
    record = module.validate_performance_document(json.loads(payload.decode("utf-8")))
    assert module.canonical_json(record) == payload, (
        "the retained record must already be canonical bytes"
    )
    document = json.loads((REPOSITORY_ROOT / ARTIFACT).read_bytes().decode("utf-8"))
    bound = document["results"]["performance_record"]
    assert bound["path"] == APPROVED_PERFORMANCE_PATH
    assert bound["sha256"] == APPROVED_PERFORMANCE_SHA256
    assert bound["source_sha"] == record["provenance"]["source_sha"]
    assert bound["source_sha"] == document["source_sha"]
    for bound_row, benchmark_row in zip(
        bound["workload_identities"], record["workloads"], strict=True
    ):
        for name in WORKLOAD_IDENTITY_KEYS:
            assert bound_row[name] == benchmark_row[name], name


def test_the_retained_record_path_matches_its_own_provenance_binding() -> None:
    """Section 11: ``<UTC>-<host>.json`` is the provenance stamp and host tag."""
    if APPROVED_PERFORMANCE_PATH is None:
        pytest.skip("the M3 performance record is authorized at E3")
    payload = (REPOSITORY_ROOT / APPROVED_PERFORMANCE_PATH).read_bytes()
    provenance = json.loads(payload.decode("utf-8"))["provenance"]
    stamp = re.sub(r"[^0-9TZ]", "", provenance["recorded_at_utc"])
    expected = f"{PERFORMANCE_DIRECTORY}/{stamp}-{provenance['host_tag']}.json"
    assert APPROVED_PERFORMANCE_PATH == expected


def test_the_reproduction_record_states_the_own_environment_requirement() -> None:
    """The venue law: a second checkout must own its Pixi environment."""
    if APPROVED_ARTIFACT_SHA256 is None:
        pytest.skip("the reproduction record is authorized at E3")
    text = (REPOSITORY_ROOT / REPRODUCTION).read_text(encoding="utf-8")
    assert text.startswith(REPRODUCTION_FRONT_MATTER)
    for token in REPRODUCTION_VENUE_TOKENS:
        assert token in text, token
    assert TOOL in text
    assert str(APPROVED_PERFORMANCE_PATH) in text
