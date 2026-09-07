"""Strict authentication of the SCI-004 phase-M3 evidence and Section 11 record.

``docs/development/sci004_mmode_design.md`` Sections 13.5, 14.2 and 14.4 freeze
this module's successor authority: terminal ``S3`` has all four approved
constants as the literal ``None``, the official evidence path and the retained
performance record **absent**, and every synthetic strict schema and digest
fixture passing. During intermediate S, D31 permits only exact rejected historical
bytes pending individually authenticated disposal; these are never current approvals.
``E3`` then changes *only* the four constants below and adds
the two artifacts plus the reproduction record.  No import, expression,
annotation, key, surrounding token, or other literal in any of the four
assignments may change, so this module's own token stream outside those four
spans is comparable to its direct-parent ``S3`` bytes.

``E3`` has four core paths plus D30's optional factual completion-ledger companion:
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
import shlex
import struct
import subprocess
import sys
import time
import tokenize
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Section 14.2/13.5's four approved constants.  ``E3`` replaces exactly these
#: four ``None`` literals and nothing else in this module.
APPROVED_SOURCE_SHA: str | None = None  # fmt: skip
APPROVED_ARTIFACT_SHA256: str | None = None  # fmt: skip
APPROVED_PERFORMANCE_PATH: str | None = None  # fmt: skip
APPROVED_PERFORMANCE_SHA256: str | None = None  # fmt: skip

# Historical rejected bytes remain eligible only for intermediate S retention.
# The four narrowly skipped lines keep future E's literal-only edits format-stable.
HISTORICAL_REJECT_S = "b07925ab14b56b3ca0fa863f806290748a31df6b"
HISTORICAL_REJECT_E = "886e62fd9f8328826b388b8960ed7413da26b6d1"
HISTORICAL_REJECT_A = "8529da951e2378115ffde8d5da3e2af56f3323d0"
HISTORICAL_REJECT_A_PATH = "docs/development/sci004_mmode_phase3_acceptance.json"
HISTORICAL_REJECT_A_SHA256 = (
    "283fb5264f5ecd86aed1300ae504b85946cf1f4d36b1c4c09bc92bb4f269421d"
)
HISTORICAL_REJECT_PERFORMANCE_PATH = (
    "output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json"
)
HISTORICAL_REJECT_PINS = {
    "docs/development/sci004_mmode_phase3_evidence.json": (
        "600b51ac4d70778ee2d3bdf7b8842b83ba77dc34d541784ad1ad7d8e5be5f8ae"
    ),
    "docs/development/sci004_mmode_phase3_evidence.md": (
        "039539a865b5d92e86379f44a324271232e8a947301e380ec7b1b1848e907b4e"
    ),
    HISTORICAL_REJECT_PERFORMANCE_PATH: (
        "07e59d3176866a78c17244849d6493365e9d410547e884cf56b254e60babe193"
    ),
}

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
POST_SOURCE_RED_RECORD_SHA256 = (
    "724f75c246ebfcf5956fc40fb2f5e349d91ccca3e6a188b3785a65f4ae4c1e10"
)
POST_SOURCE_DESIGN_SHA = "4d507bf1333ccaa4c8beec3815370ba0f6043bb2"
FINGERPRINT_RED_RECORD = (
    "docs/development/sci004_mmode_phase3_fingerprint_post_source_red_failures.json"
)
FINGERPRINT_RED_RECORD_SCHEMA = (
    "radiosim.sci004.mmode-phase3-fingerprint-post-source-red-failures.v1"
)
FINGERPRINT_RED_RECORD_SHA256 = (
    "6bf1cf94b30961fd7a27519fad1252169155fdeee0e81618ea15115b50fbdb68"
)
FINGERPRINT_DESIGN_SHA = "ca3c37171aaaeec175b5ad72d324957762303853"
ORIGINAL_FINGERPRINT_R3_SHA = "a65c53a46e84f63c163c5ad15fba8645df33d1d2"
CURRENT_D31_SHA = "f2e5edbcc97450262482672bb322cf926622b208"
HISTORICAL_SOURCE_D32_SHA = "bcd79b1d6268859368d77c3f94cef334b001cb37"
CURRENT_SOURCE_D33_SHA = "343ea0467420d452e9d728f0475167e74721e22f"
RANGE_ORIGIN_D30_SHA = "d3ddb10ae01ab450f5337d06c9588ce8144cf1e5"
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


def _historical_evidence_bytes() -> dict[str, bytes]:
    """Authenticate original rejected S/E/A objects without using approval pins."""
    for commit, parent in (
        (HISTORICAL_REJECT_E, HISTORICAL_REJECT_S),
        (HISTORICAL_REJECT_A, HISTORICAL_REJECT_E),
    ):
        assert _git("rev-list", "--parents", "-n", "1", commit).split() == [
            commit,
            parent,
        ]
    rejected = _e3_regular_blob(HISTORICAL_REJECT_A, HISTORICAL_REJECT_A_PATH)
    assert hashlib.sha256(rejected).hexdigest() == HISTORICAL_REJECT_A_SHA256
    acceptance = json.loads(rejected)
    assert acceptance["verdict"] == "REJECT"
    assert acceptance["evidence_commit_sha"] == HISTORICAL_REJECT_E
    assert acceptance["source_sha"] == HISTORICAL_REJECT_S
    assert acceptance["evidence_artifact_sha256"] == HISTORICAL_REJECT_PINS[ARTIFACT]
    payloads: dict[str, bytes] = {}
    for path, digest in HISTORICAL_REJECT_PINS.items():
        payload = _e3_regular_blob(HISTORICAL_REJECT_E, path)
        assert hashlib.sha256(payload).hexdigest() == digest
        payloads[path] = payload
    envelope = json.loads(payloads[ARTIFACT])
    performance = json.loads(payloads[HISTORICAL_REJECT_PERFORMANCE_PATH])
    assert envelope["source_sha"] == HISTORICAL_REJECT_S
    assert performance["provenance"]["source_sha"] == HISTORICAL_REJECT_S
    bound = envelope["results"]["performance_record"]
    assert bound["path"] == HISTORICAL_REJECT_PERFORMANCE_PATH
    assert bound["sha256"] == HISTORICAL_REJECT_PINS[HISTORICAL_REJECT_PERFORMANCE_PATH]
    return payloads


def _evidence_lifecycle(
    root: Path,
    source_sha: str | None,
    artifact_sha256: str | None,
    performance_path: str | None,
    performance_sha256: str | None,
) -> str:
    """Permit exact historical retention without licensing terminal S or E."""
    constants = (source_sha, artifact_sha256, performance_path, performance_sha256)
    assert all(value is None for value in constants) or all(
        value is not None for value in constants
    ), "mixed approval pins"
    if source_sha is not None:
        for value, pattern in zip(
            constants, (GIT_SHA, SHA256, PERFORMANCE_PATH, SHA256), strict=True
        ):
            assert type(value) is str and pattern.fullmatch(value)
        assert source_sha != HISTORICAL_REJECT_S, "rejected S cannot be approved"
        assert artifact_sha256 != HISTORICAL_REJECT_PINS[ARTIFACT], (
            "rejected artifact cannot be approved"
        )
        assert performance_path != HISTORICAL_REJECT_PERFORMANCE_PATH, (
            "rejected performance path cannot be approved"
        )
        assert (
            performance_sha256
            != HISTORICAL_REJECT_PINS[HISTORICAL_REJECT_PERFORMANCE_PATH]
        ), "rejected performance bytes cannot be approved"
        return "current-approval"
    directory = root / PERFORMANCE_DIRECTORY
    assert not directory.is_symlink(), (
        "phase performance directory must not be a symlink"
    )
    if directory.exists():
        assert directory.is_dir(), "phase performance directory must be a directory"
        for path in directory.iterdir():
            assert path == root / HISTORICAL_REJECT_PERFORMANCE_PATH, (
                "unapproved entry in phase performance directory"
            )
            assert not path.is_symlink(), "historical benchmark must not be a symlink"
            assert path.is_file(), "historical benchmark must be a regular file"
    retained: dict[str, bytes] = {}
    for relative in HISTORICAL_REJECT_PINS:
        path = root / relative
        assert not path.is_symlink(), "historical artifact must not be a symlink"
        if path.exists():
            assert path.is_file(), "historical artifact must be a regular file"
            retained[relative] = path.read_bytes()
    if not retained:
        return "unapproved-absent"
    historical = _historical_evidence_bytes()
    for relative, raw in retained.items():
        assert raw == historical[relative], "unapproved bytes differ from historical E"
    return "historical-reject"


def test_the_approved_constants_are_null_sentinels_before_e3() -> None:
    """All four approvals transition together; rejected history is never current."""
    _ = _evidence_lifecycle(
        REPOSITORY_ROOT,
        APPROVED_SOURCE_SHA,
        APPROVED_ARTIFACT_SHA256,
        APPROVED_PERFORMANCE_PATH,
        APPROVED_PERFORMANCE_SHA256,
    )


def test_the_official_artifacts_are_absent_in_the_s3_state() -> None:
    """Intermediate S permits exact history; source readiness requires disposal."""
    if APPROVED_ARTIFACT_SHA256 is not None:
        return
    assert _evidence_lifecycle(REPOSITORY_ROOT, None, None, None, None) in {
        "unapproved-absent",
        "historical-reject",
    }


@pytest.fixture(scope="module")
def historical_evidence_payloads() -> dict[str, bytes]:
    return _historical_evidence_bytes()


@pytest.mark.parametrize("retained_mask", range(8))
def test_evidence_lifecycle_allows_individual_historical_disposal(
    tmp_path: Path, historical_evidence_payloads: dict[str, bytes], retained_mask: int
) -> None:
    for index, (relative, raw) in enumerate(historical_evidence_payloads.items()):
        if retained_mask & (1 << index):
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_bytes(raw)
    expected = "historical-reject" if retained_mask else "unapproved-absent"
    assert _evidence_lifecycle(tmp_path, None, None, None, None) == expected


@pytest.mark.parametrize("null_mask", range(1, 15))
def test_evidence_lifecycle_rejects_every_mixed_approval(
    tmp_path: Path, null_mask: int
) -> None:
    current = (FORTY, SIXTY_FOUR, SYNTHETIC_PERFORMANCE_PATH, SIXTY_FOUR)
    pins = [None if null_mask & (1 << i) else value for i, value in enumerate(current)]
    with pytest.raises(AssertionError, match="mixed approval pins"):
        _ = _evidence_lifecycle(tmp_path, *pins)


@pytest.mark.parametrize("historical_index", range(4))
def test_evidence_lifecycle_never_approves_rejected_identities(
    tmp_path: Path, historical_index: int
) -> None:
    current = [FORTY, SIXTY_FOUR, SYNTHETIC_PERFORMANCE_PATH, SIXTY_FOUR]
    assert _evidence_lifecycle(tmp_path, *current) == "current-approval"
    historical = (
        HISTORICAL_REJECT_S,
        HISTORICAL_REJECT_PINS[ARTIFACT],
        HISTORICAL_REJECT_PERFORMANCE_PATH,
        HISTORICAL_REJECT_PINS[HISTORICAL_REJECT_PERFORMANCE_PATH],
    )
    current[historical_index] = historical[historical_index]
    with pytest.raises(AssertionError, match="cannot be approved"):
        _ = _evidence_lifecycle(tmp_path, *current)


@pytest.mark.parametrize("relative", tuple(HISTORICAL_REJECT_PINS))
@pytest.mark.parametrize("mutation", ["bytes", "symlink", "directory"])
def test_evidence_lifecycle_rejects_changed_or_nonregular_history(
    tmp_path: Path,
    historical_evidence_payloads: dict[str, bytes],
    relative: str,
    mutation: str,
) -> None:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if mutation == "bytes":
        _ = path.write_bytes(historical_evidence_payloads[relative] + b"\n")
    elif mutation == "symlink":
        target = tmp_path / "target"
        _ = target.write_bytes(historical_evidence_payloads[relative])
        path.symlink_to(target)
    else:
        path.mkdir()
    with pytest.raises(AssertionError, match="historical E|symlink|regular file"):
        _ = _evidence_lifecycle(tmp_path, None, None, None, None)


@pytest.mark.parametrize("historical_present", [False, True])
@pytest.mark.parametrize("mutation", ["new-record", "symlink", "fifo", "nested"])
def test_evidence_lifecycle_rejects_unapproved_phase_performance_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    historical_evidence_payloads: dict[str, bytes],
    historical_present: bool,
    mutation: str,
) -> None:
    directory = tmp_path / PERFORMANCE_DIRECTORY
    directory.mkdir(parents=True)
    if historical_present:
        historical = tmp_path / HISTORICAL_REJECT_PERFORMANCE_PATH
        _ = historical.write_bytes(
            historical_evidence_payloads[HISTORICAL_REJECT_PERFORMANCE_PATH]
        )
    path = tmp_path / SYNTHETIC_PERFORMANCE_PATH
    if mutation == "new-record":
        _ = path.write_bytes(b'{"unapproved":true}\n')
    elif mutation == "symlink":
        path.symlink_to(tmp_path / "missing-benchmark")
    elif mutation == "fifo":
        os.mkfifo(path)
    else:
        path = directory / "nested" / "record.json"
        path.parent.mkdir()
        _ = path.write_bytes(b"{}")
    with pytest.raises(AssertionError, match="unapproved entry"):
        _ = _evidence_lifecycle(tmp_path, None, None, None, None)
    monkeypatch.setattr(sys.modules[__name__], "REPOSITORY_ROOT", tmp_path)
    for name in APPROVED_CONSTANT_NAMES:
        monkeypatch.setattr(sys.modules[__name__], name, None)
    with pytest.raises(AssertionError, match="unapproved entry"):
        test_the_official_artifacts_are_absent_in_the_s3_state()


@pytest.mark.parametrize("mutation", ["dangling-symlink", "directory-symlink", "file"])
def test_evidence_lifecycle_rejects_nonregular_phase_directory(
    tmp_path: Path, mutation: str
) -> None:
    directory = tmp_path / PERFORMANCE_DIRECTORY
    directory.parent.mkdir(parents=True)
    if mutation == "file":
        _ = directory.write_bytes(b"not a directory")
    else:
        target = tmp_path / "target"
        if mutation == "directory-symlink":
            target.mkdir()
        directory.symlink_to(target, target_is_directory=True)
    with pytest.raises(AssertionError, match="phase performance directory"):
        _ = _evidence_lifecycle(tmp_path, None, None, None, None)


def test_evidence_approval_literal_changes_are_formatter_stable() -> None:
    source = (REPOSITORY_ROOT / VALIDATOR).read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    _spans, bodies = _constant_spans(source)
    nulls = "".join(lines[int(body[0].start[0]) - 1] for body in bodies)
    approved = nulls
    for value in (FORTY, SIXTY_FOUR, SYNTHETIC_PERFORMANCE_PATH, SIXTY_FOUR):
        approved = approved.replace("= None", f'= "{value}"', 1)
    assert nulls.count("# fmt: skip") == len(APPROVED_CONSTANT_NAMES)
    for original in (nulls, approved):
        completed = subprocess.run(
            ["ruff", "format", "--stdin-filename", VALIDATOR, "-"],
            input=original.encode("utf-8"),
            capture_output=True,
            check=True,
            cwd=REPOSITORY_ROOT,
        )
        assert completed.stdout == original.encode("utf-8")


@pytest.mark.parametrize(
    "relative", (*HISTORICAL_REJECT_PINS, HISTORICAL_REJECT_A_PATH)
)
def test_evidence_readiness_still_requires_every_rejected_output_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    module = _tool()
    monkeypatch.setattr(module, "REPOSITORY_ROOT", tmp_path)

    def clean_preflight(
        _source_sha: str | None, _declared: tuple[str, ...]
    ) -> dict[str, str]:
        return {"source_sha": FORTY}

    monkeypatch.setattr(module, "preflight", clean_preflight)
    monkeypatch.setattr(module, "_red_commit_sha", lambda: FORTY)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(b"retained rejected output")
    with pytest.raises(module.EvidenceError, match="rejected output remains at source"):
        module.source_readiness(FORTY, ())


@pytest.mark.parametrize("overlay", ["replace", "replace-blob", "graft", "routing"])
def test_historical_evidence_reads_actual_git_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overlay: str
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "REPOSITORY_ROOT", tmp_path)
    _ = _git("init", "-q")
    for key, value in {
        "user.name": "Synthetic Fixture",
        "user.email": "fixture@example.invalid",
        "commit.gpgsign": "false",
        "core.autocrlf": "false",
        "core.hooksPath": os.devnull,
    }.items():
        _ = _git("config", key, value)

    def commit(files: dict[str, bytes]) -> str:
        for relative, raw in files.items():
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_bytes(raw)
        _ = _git("add", "--all")
        _ = _git("commit", "--allow-empty", "-qm", "synthetic historical E")
        return _git("rev-parse", "HEAD").strip()

    source = commit({"seed": b"synthetic"})
    performance = json.dumps({"provenance": {"source_sha": source}}).encode()
    raw = (
        json.dumps(
            {
                "source_sha": source,
                "results": {
                    "performance_record": {
                        "path": HISTORICAL_REJECT_PERFORMANCE_PATH,
                        "sha256": hashlib.sha256(performance).hexdigest(),
                    }
                },
            }
        ).encode()
        + b"\r\n"
    )
    payloads = {
        ARTIFACT: raw,
        REPRODUCTION: b"synthetic\r\n",
        HISTORICAL_REJECT_PERFORMANCE_PATH: performance,
    }
    evidence = commit(payloads)
    rejected_raw = json.dumps(
        {
            "verdict": "REJECT",
            "source_sha": source,
            "evidence_commit_sha": evidence,
            "evidence_artifact_sha256": hashlib.sha256(raw).hexdigest(),
        }
    ).encode()
    rejected = commit({HISTORICAL_REJECT_A_PATH: rejected_raw})
    for name, value in (
        ("HISTORICAL_REJECT_S", source),
        ("HISTORICAL_REJECT_E", evidence),
        ("HISTORICAL_REJECT_A", rejected),
        ("HISTORICAL_REJECT_A_SHA256", hashlib.sha256(rejected_raw).hexdigest()),
    ):
        monkeypatch.setattr(sys.modules[__name__], name, value)
    monkeypatch.setattr(
        sys.modules[__name__],
        "HISTORICAL_REJECT_PINS",
        {path: hashlib.sha256(value).hexdigest() for path, value in payloads.items()},
    )
    if overlay == "replace":
        alternate = _git(
            "commit-tree", _git("write-tree").strip(), "-p", source, "-m", "forged"
        ).strip()
        _ = _git("replace", rejected, alternate)
    elif overlay == "replace-blob":
        bad = tmp_path / "bad-blob"
        _ = bad.write_bytes(b"forged bytes")
        _ = _git(
            "replace",
            _git("rev-parse", f"{evidence}:{ARTIFACT}").strip(),
            _git("hash-object", "-w", str(bad)).strip(),
        )
        assert (
            subprocess.check_output(
                ["git", "show", f"{evidence}:{ARTIFACT}"], cwd=tmp_path
            )
            != raw
        )
    elif overlay == "graft":
        graft = tmp_path / ".git/info/grafts"
        graft.parent.mkdir(parents=True, exist_ok=True)
        _ = graft.write_text(f"{rejected} {source}\n")
    else:
        for name in ("GIT_DIR", "GIT_OBJECT_DIRECTORY", "GIT_CONFIG_GLOBAL"):
            monkeypatch.setenv(name, str(tmp_path / "missing"))
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "diff.external")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", "/usr/bin/false")
    if overlay in {"replace", "graft"}:
        assert subprocess.check_output(
            ["git", "rev-list", "--parents", "-n", "1", rejected], cwd=tmp_path
        ).decode().split() == [rejected, source]
    assert _historical_evidence_bytes() == payloads


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
    assert (
        "_publish_evidence_payload(payload, performance_path, performance_payload)"
        in build
    )


def test_the_generator_publishes_the_performance_record_before_the_envelope() -> None:
    """Section 14.2: "performance first and evidence last"."""
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    build = source[source.index("def build_phase3_evidence") :]
    build = build[: build.index("def main(")]
    publication = build.index("_publish_evidence_payload(payload,")
    assert build.index("validate_evidence_document(document)") < publication
    assert build.index("payload = canonical_json(document)") < publication
    assert (
        build.index('_require_raw_tracked_checkout(state["source_sha"])') < publication
    )
    assert (
        build.index('require_declared_outputs_only(declared, state["source_sha"])')
        > publication
    )
    helper = source[source.index("def _publish_evidence_payload(") :]
    helper = helper[: helper.index("def _read_evidence_payload(")]
    first = helper.index("write_atomic_no_overwrite(REPOSITORY_ROOT / performance_path")
    second = helper.index(
        "write_atomic_no_overwrite(REPOSITORY_ROOT / EVIDENCE_ARTIFACT"
    )
    assert helper.index("len(payload) < EVIDENCE_BYTE_LIMIT") < first < second


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
    frozen = module._frozen_binding("SOURCE_DESIGN_SHA")
    assert GIT_SHA.fullmatch(frozen)
    assert frozen == CURRENT_SOURCE_D33_SHA
    assert module._design_sha() == frozen
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    assert '"--", "docs/development/sci004_mmode_design.md"' not in source, (
        "the operative D may not be derived by searching the memo's history"
    )


def test_the_frozen_design_binding_is_read_from_the_dependency_validator() -> None:
    """Section 14.0: the binding has exactly one site, read by AST."""
    module = _tool()
    frozen = module._frozen_binding("SOURCE_DESIGN_SHA")
    text = (REPOSITORY_ROOT / "tests/unit/test_sci004_phase3_dependency.py").read_text(
        encoding="utf-8"
    )
    assert f'SOURCE_DESIGN_SHA = "{frozen}"' in text
    tool_source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    assert frozen not in tool_source, (
        "the generator must read the frozen binding, never restate it"
    )


def test_current_and_historical_design_roles_are_exact_and_distinct() -> None:
    """Corrections #31–33 separate the range origin, R, and source authorities."""
    module = _tool()
    post_source = json.loads((REPOSITORY_ROOT / POST_SOURCE_RED_RECORD).read_bytes())
    fingerprint = json.loads((REPOSITORY_ROOT / FINGERPRINT_RED_RECORD).read_bytes())
    assert module._design_sha() == CURRENT_SOURCE_D33_SHA
    assert post_source["design_sha"] == POST_SOURCE_DESIGN_SHA
    assert fingerprint["design_sha"] == FINGERPRINT_DESIGN_SHA
    assert (
        len(
            {
                POST_SOURCE_DESIGN_SHA,
                FINGERPRINT_DESIGN_SHA,
                CURRENT_D31_SHA,
                ORIGINAL_FINGERPRINT_R3_SHA,
            }
        )
        == 4
    )
    from tests.unit.test_sci004_phase3_dependency import resolve_r3_replay_anchor

    anchor = resolve_r3_replay_anchor()
    assert anchor.commit not in {
        POST_SOURCE_DESIGN_SHA,
        FINGERPRINT_DESIGN_SHA,
        RANGE_ORIGIN_D30_SHA,
        ORIGINAL_FINGERPRINT_R3_SHA,
    }
    if anchor.role == "pre-commit-authoring-tip":
        with pytest.raises(
            module.EvidenceError, match="authoring cannot generate evidence"
        ):
            module._red_commit_sha()
    else:
        assert anchor.commit != CURRENT_D31_SHA
        assert module._red_commit_sha() == anchor.commit


@pytest.mark.parametrize(
    "target,field",
    [
        ("dependency", "__file__"),
        ("dependency", "REPOSITORY_ROOT"),
        ("history", "__file__"),
        ("history", "REPOSITORY_ROOT"),
    ],
)
def test_terminal_r3_rejects_cached_validators_from_another_checkout(
    monkeypatch, tmp_path, target, field
):
    from tests.unit import test_sci004_phase3_dependency as dependency
    from tools import sci004_phase3_history as history

    module = _tool()
    peer = dependency if target == "dependency" else history
    wrong = tmp_path / "other.py"
    wrong.write_text("# unrelated cached module\n")
    monkeypatch.setattr(peer, field, str(wrong) if field == "__file__" else tmp_path)
    with pytest.raises(module.EvidenceError, match="another checkout"):
        module._red_commit_sha()


@pytest.mark.parametrize(
    "target,field,replacement",
    [
        ("dependency", "APPROVED_SCI004_D_SHA", POST_SOURCE_DESIGN_SHA),
        ("history", "OPERATIVE_DESIGN_SHA", POST_SOURCE_DESIGN_SHA),
        ("dependency", "APPROVED_SCI004_D_SHA", RANGE_ORIGIN_D30_SHA),
        ("history", "OPERATIVE_DESIGN_SHA", RANGE_ORIGIN_D30_SHA),
        ("dependency", "D30_SHA", CURRENT_D31_SHA),
        ("history", "DESIGN_SHA", CURRENT_D31_SHA),
    ],
)
def test_terminal_r3_rejects_stale_loaded_design_bindings(
    monkeypatch, target, field, replacement
):
    from tests.unit import test_sci004_phase3_dependency as dependency
    from tools import sci004_phase3_history as history

    module = _tool()
    peer = dependency if target == "dependency" else history
    monkeypatch.setattr(peer, field, replacement)
    with pytest.raises(module.EvidenceError, match="loaded design differs"):
        module._red_commit_sha()


def test_terminal_r3_preserves_distinct_range_origin_and_operative_design(
    monkeypatch,
):
    from types import SimpleNamespace

    from tests.unit import test_sci004_phase3_dependency as dependency
    from tools import sci004_phase3_history as history

    module = _tool()
    assert history.DESIGN_SHA == dependency.D30_SHA == RANGE_ORIGIN_D30_SHA
    assert (
        history.OPERATIVE_DESIGN_SHA
        == dependency.APPROVED_SCI004_D_SHA
        == CURRENT_D31_SHA
    )
    assert module._frozen_binding("D30_SHA") == RANGE_ORIGIN_D30_SHA
    assert module._design_sha() == CURRENT_SOURCE_D33_SHA
    terminal = "f" * 40
    monkeypatch.setattr(
        dependency,
        "resolve_r3_replay_anchor",
        lambda: SimpleNamespace(role="r3", commit=terminal),
    )
    assert module._red_commit_sha() == terminal


def test_terminal_r3_rejects_jointly_substituted_range_origin(monkeypatch):
    from tests.unit import test_sci004_phase3_dependency as dependency
    from tools import sci004_phase3_history as history

    module = _tool()
    monkeypatch.setattr(dependency, "D30_SHA", CURRENT_D31_SHA)
    monkeypatch.setattr(history, "DESIGN_SHA", CURRENT_D31_SHA)
    with pytest.raises(module.EvidenceError, match="loaded design differs"):
        module._red_commit_sha()


def _mock_red_binding_for_historical_checks(
    module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise historical joins during R3 without licensing evidence generation.

    The real dependency resolver authenticates the committed candidate range.
    Only these unit tests substitute that tip for a future frozen red boundary;
    separate unmocked tests require production refusal of the authoring role.
    """
    from tests.unit.test_sci004_phase3_dependency import resolve_r3_replay_anchor

    anchor = resolve_r3_replay_anchor()
    assert anchor.role in {"pre-commit-authoring-tip", "r3"}
    monkeypatch.setattr(module, "_red_commit_sha", lambda: anchor.commit)


def _mutate_retained_record(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    path: str,
    field: str,
    value: str,
) -> None:
    real = module._canonical_artifact

    def mutated(requested: str, *, label: str) -> tuple[bytes, dict[str, Any]]:
        raw, document = real(requested, label=label)
        if requested == path:
            document = copy.deepcopy(document)
            document[field] = value
        return raw, document

    monkeypatch.setattr(module, "_canonical_artifact", mutated)


@pytest.mark.parametrize(
    ("path", "replacement", "expected"),
    (
        (POST_SOURCE_RED_RECORD, SIXTY_FOUR, "correction #24's D"),
        (FINGERPRINT_RED_RECORD, SIXTY_FOUR, "fingerprint red record"),
        (POST_SOURCE_RED_RECORD, CURRENT_D31_SHA, "correction #24's D"),
        (FINGERPRINT_RED_RECORD, CURRENT_D31_SHA, "fingerprint red record"),
    ),
)
def test_historical_design_mutation_or_current_d31_substitution_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    replacement: str,
    expected: str,
) -> None:
    """D24 and D25 reject both independent mutation and D31 substitution."""
    module = _tool()
    _mock_red_binding_for_historical_checks(module, monkeypatch)
    _mutate_retained_record(
        module,
        monkeypatch,
        path=path,
        field="design_sha",
        value=replacement,
    )
    with pytest.raises(module.EvidenceError, match=expected):
        module._red_failure_record_reference(module._red_commit_sha())


@pytest.mark.parametrize(
    "historical_design",
    [
        POST_SOURCE_DESIGN_SHA,
        FINGERPRINT_DESIGN_SHA,
        RANGE_ORIGIN_D30_SHA,
        CURRENT_D31_SHA,
        HISTORICAL_SOURCE_D32_SHA,
    ],
)
def test_a_historical_design_cannot_replace_the_current_d33_envelope_binding(
    historical_design: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _tool()
    _mock_red_binding_for_historical_checks(module, monkeypatch)
    document = _synthetic_document(module)
    document["design_sha"] = historical_design
    document["red_commit_sha"] = module._red_commit_sha()
    document["red_failure_record"] = module._red_failure_record_reference(
        document["red_commit_sha"]
    )
    with pytest.raises(module.EvidenceError, match="current frozen operative D33"):
        module.validate_evidence_artifact(document)


def test_a_design_sha_cannot_stand_in_for_red_commit_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _tool()
    _mock_red_binding_for_historical_checks(module, monkeypatch)
    document = _synthetic_document(module)
    document["design_sha"] = CURRENT_SOURCE_D33_SHA
    document["red_commit_sha"] = CURRENT_D31_SHA
    document["red_failure_record"] = module._red_failure_record_reference(
        module._red_commit_sha()
    )
    hypothetical_future_r3 = "f" * 40
    monkeypatch.setattr(module, "_red_commit_sha", lambda: hypothetical_future_r3)
    with pytest.raises(module.EvidenceError, match="red_commit_sha must name"):
        module.validate_evidence_artifact(document)


def test_historical_record_bytes_and_cross_record_digests_are_authenticated(
    monkeypatch,
) -> None:
    module = _tool()
    _mock_red_binding_for_historical_checks(module, monkeypatch)
    historical_raw = (REPOSITORY_ROOT / RED_RECORD).read_bytes()
    post_raw = (REPOSITORY_ROOT / POST_SOURCE_RED_RECORD).read_bytes()
    fingerprint_raw = (REPOSITORY_ROOT / FINGERPRINT_RED_RECORD).read_bytes()
    fingerprint = json.loads(fingerprint_raw)
    assert hashlib.sha256(historical_raw).hexdigest() == HISTORICAL_RED_RECORD_SHA256
    assert hashlib.sha256(post_raw).hexdigest() == POST_SOURCE_RED_RECORD_SHA256
    assert hashlib.sha256(fingerprint_raw).hexdigest() == FINGERPRINT_RED_RECORD_SHA256
    assert fingerprint["schema_version"] == FINGERPRINT_RED_RECORD_SCHEMA
    assert fingerprint["historical_red_record_sha256"] == (HISTORICAL_RED_RECORD_SHA256)
    assert fingerprint["correction24_post_source_red_record_sha256"] == (
        POST_SOURCE_RED_RECORD_SHA256
    )
    module._red_failure_record_reference(module._red_commit_sha())


@pytest.mark.parametrize(
    "field",
    ["historical_red_record_sha256", "correction24_post_source_red_record_sha256"],
)
def test_a_fingerprint_cross_record_digest_mutation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    module = _tool()
    _mock_red_binding_for_historical_checks(module, monkeypatch)
    _mutate_retained_record(
        module,
        monkeypatch,
        path=FINGERPRINT_RED_RECORD,
        field=field,
        value=SIXTY_FOUR,
    )
    with pytest.raises(module.EvidenceError, match="fingerprint red record"):
        module._red_failure_record_reference(module._red_commit_sha())


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
    """Historical containment is distinct from a sealed generation boundary."""
    from tests.unit.test_sci004_phase3_dependency import resolve_r3_replay_anchor

    module = _tool()
    anchor = resolve_r3_replay_anchor()
    for retained in (RED_RECORD, POST_SOURCE_RED_RECORD, FINGERPRINT_RED_RECORD):
        assert (
            module._git_blob(anchor.commit, retained)
            == (REPOSITORY_ROOT / retained).read_bytes()
        )
    if anchor.role == "pre-commit-authoring-tip":
        with pytest.raises(
            module.EvidenceError, match="first S3 has not frozen terminal R3"
        ):
            module._red_commit_sha()
    else:
        assert module._red_commit_sha() == anchor.commit


def test_the_generator_authenticates_and_joins_both_red_records(monkeypatch) -> None:
    module = _tool()
    _mock_red_binding_for_historical_checks(module, monkeypatch)
    reference = module._red_failure_record_reference(module._red_commit_sha())
    historical_raw = (REPOSITORY_ROOT / RED_RECORD).read_bytes()
    post_source_raw = (REPOSITORY_ROOT / POST_SOURCE_RED_RECORD).read_bytes()
    fingerprint_raw = (REPOSITORY_ROOT / FINGERPRINT_RED_RECORD).read_bytes()
    assert hashlib.sha256(historical_raw).hexdigest() == (HISTORICAL_RED_RECORD_SHA256)
    assert hashlib.sha256(post_source_raw).hexdigest() == POST_SOURCE_RED_RECORD_SHA256
    assert hashlib.sha256(fingerprint_raw).hexdigest() == FINGERPRINT_RED_RECORD_SHA256
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


def test_the_artifact_validator_authenticates_the_fresh_r3_and_both_inputs(
    monkeypatch,
) -> None:
    module = _tool()
    _mock_red_binding_for_historical_checks(module, monkeypatch)
    document = _synthetic_document(module)
    red_commit = module._red_commit_sha()
    document["design_sha"] = CURRENT_SOURCE_D33_SHA
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
    _mock_red_binding_for_historical_checks(module, monkeypatch)
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


def _git_bytes(*arguments: str) -> bytes:
    # Read actual objects in this repository, not a caller's redirected store,
    # replacement refs, grafted ancestry or configured diff presentation.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_GRAFT_FILE=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_CONFIG_GLOBAL=os.devnull,
    )
    if arguments[0] == "diff":
        arguments = (
            "diff",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--no-relative",
            "--ignore-submodules=none",
            *arguments[1:],
        )
    elif arguments[0] == "show":
        arguments = ("show", "--no-ext-diff", "--no-textconv", *arguments[1:])
    completed = subprocess.run(
        [
            "git",
            "--no-pager",
            "--no-replace-objects",
            "--literal-pathspecs",
            "-c",
            "color.ui=false",
            "-c",
            "core.commitGraph=false",
            *arguments,
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"git {' '.join(arguments)} failed: {completed.stderr!r}"
    )
    return completed.stdout


def _git(*arguments: str) -> str:
    return _git_bytes(*arguments).decode("utf-8")


def _locate_evidence_commit() -> str:
    """Select current E on HEAD's actual first-parent chain after terminal S."""
    parent = APPROVED_SOURCE_SHA
    assert parent is not None and GIT_SHA.fullmatch(parent)
    assert parent in _git("rev-list", "--first-parent", "HEAD").split(), (
        "approved S is not on HEAD's first-parent chain"
    )
    assert len(_git("rev-list", "--parents", "-n", "1", parent).split()) == 2, (
        "approved S must have exactly one parent"
    )
    current = _git(
        "rev-list", "--first-parent", "--ancestry-path", "--reverse", f"{parent}..HEAD"
    ).split()
    assert current, "HEAD has no first-parent descendant after approved S"
    previous = parent
    for commit in current:
        assert GIT_SHA.fullmatch(commit)
        assert _git("rev-list", "--parents", "-n", "1", commit).split() == [
            commit,
            previous,
        ], "current evidence ancestry must be a sole-parent chain"
        previous = commit
    return current[0]


def _e3_regular_blob(commit: str, path: str) -> bytes:
    entry = _git_bytes("ls-tree", "-z", commit, "--", path)
    assert re.fullmatch(
        rb"100644 blob [0-9a-f]{40}\t" + re.escape(path.encode("utf-8")) + rb"\x00",
        entry,
    ), f"{path} must be an exact regular-file entry at {commit}"
    return _git_bytes("show", f"{commit}:{path}")


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


def _e3_factual_ledger_suffix() -> bytes:
    """Supported optional companion: append these pin-derived facts verbatim.

    D30 permits factual status prose; this exact supported form is the validator's
    implementation choice, not syntax prescribed by D30. Parent ledger bytes and
    regular-file mode are preserved. No acceptance or contract claim is permitted.
    """
    return (
        "\n\n## Current M3 evidence record\n\n"
        f"- Source commit: `{APPROVED_SOURCE_SHA}`.\n"
        f"- Evidence artifact: `{ARTIFACT}`; SHA-256 `{APPROVED_ARTIFACT_SHA256}`.\n"
        f"- Performance record: `{APPROVED_PERFORMANCE_PATH}`; "
        f"SHA-256 `{APPROVED_PERFORMANCE_SHA256}`.\n"
        "- Independent M3 acceptance remains pending.\n"
    ).encode()


def test_the_artifact_introducing_commit_directly_parents_the_approved_source() -> None:
    """Section 14.2's ``E3`` ancestry clause, skipped until the constants flip."""
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_SOURCE_SHA is None:
        pytest.skip("the M3 evidence artifact is authorized at E3")
    located = _locate_evidence_commit()
    _authenticate_e_artifacts(located)


def _authenticate_e_artifacts(located: str) -> None:
    assert APPROVED_SOURCE_SHA is not None
    for path, digest in (
        (ARTIFACT, APPROVED_ARTIFACT_SHA256),
        (APPROVED_PERFORMANCE_PATH, APPROVED_PERFORMANCE_SHA256),
        (REPRODUCTION, None),
    ):
        assert path is not None
        assert not _git_bytes("ls-tree", "-z", APPROVED_SOURCE_SHA, "--", path), (
            f"{path} must be absent at terminal S"
        )
        payload = _e3_regular_blob(located, path)
        if digest is not None:
            assert hashlib.sha256(payload).hexdigest() == digest
    artifact = json.loads(_e3_regular_blob(located, ARTIFACT))
    assert artifact["source_sha"] == APPROVED_SOURCE_SHA


def test_the_e3_diff_writes_only_the_section_13_5_authorized_paths() -> None:
    """Section 13.5: the envelope, its record, this module, and one benchmark."""
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_PERFORMANCE_PATH is None:
        pytest.skip("the M3 evidence artifact is authorized at E3")
    located = _locate_evidence_commit()
    changed = set(
        _git_bytes("diff", "--name-only", "-z", f"{located}^", located, "--")
        .removesuffix(b"\0")
        .split(b"\0")
    )
    core = {path.encode("utf-8") for path in _e3_authorized_paths()}
    ledger = "docs/development/completion_ledger.md"
    assert core <= changed <= core | {ledger.encode("utf-8")}, (
        f"E3 must write its four core paths and only its optional ledger; {changed!r}"
    )
    if ledger.encode("utf-8") in changed:
        _authenticate_e_artifacts(located)
        before = _e3_regular_blob(f"{located}^", ledger)
        after = _e3_regular_blob(located, ledger)
        assert after == before + _e3_factual_ledger_suffix(), (
            "E3 ledger companion must preserve parent bytes and append only its facts"
        )
    record = _e3_regular_blob(located, REPRODUCTION).decode("utf-8")
    assert record.startswith(REPRODUCTION_FRONT_MATTER), (
        "the reproduction record must open with Section 14.2's exact MyST front matter"
    )


def test_the_e3_diff_changes_only_the_four_approved_constant_assignments() -> None:
    """Section 14.2: this module's own ``E3`` diff is the four constants alone."""
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_SOURCE_SHA is None:
        pytest.skip("the M3 evidence artifact is authorized at E3")
    located = _locate_evidence_commit()
    parent = _git("rev-list", "--parents", "-n", "1", located).split()[1]
    before = _e3_regular_blob(parent, VALIDATOR).decode("utf-8")
    after = _e3_regular_blob(located, VALIDATOR).decode("utf-8")

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
    replacements: list[tuple[int, int, str]] = []
    lines = before.splitlines(keepends=True)
    for name, body_before, body_after, value in zip(
        APPROVED_CONSTANT_NAMES, bodies_before, bodies_after, approved, strict=True
    ):
        assert _assigned_literal(body_before) == "None", (
            f"{name} must be the null sentinel at the direct parent {parent}"
        )
        assert _assigned_literal(body_after) == f'"{value}"', (
            f"{name} at {located} is not the approved literal"
        )
        equal = next(i for i, token in enumerate(body_before) if token.string == "=")
        expected = [(token.type, token.string) for token in body_before]
        rhs_nulls = [
            i
            for i in range(equal + 1, len(expected))
            if expected[i] == (tokenize.NAME, "None")
        ]
        assert len(rhs_nulls) == 1, "approved source must have exactly one RHS None"
        index = rhs_nulls[0]
        expected[index] = (tokenize.STRING, f'"{value}"')
        token = body_before[index]
        start = sum(map(len, lines[: int(token.start[0]) - 1])) + int(token.start[1])
        stop = sum(map(len, lines[: int(token.end[0]) - 1])) + int(token.end[1])
        replacements.append((start, stop, f'"{value}"'))
        assert [(token.type, token.string) for token in body_after] == expected, (
            f"{name} changed tokens other than its approved value"
        )
    expected_source = before
    for start, stop, literal in reversed(replacements):
        expected_source = expected_source[:start] + literal + expected_source[stop:]
    assert after == expected_source, "E3 may replace only the four RHS None literals"


def _e_topology_commit(files: Mapping[str, bytes | None]) -> str:
    """Write synthetic bytes only inside the fixture's temporary repository."""
    for name, raw in files.items():
        path = REPOSITORY_ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if raw is None:
            path.unlink()
        else:
            _ = path.write_bytes(raw)
    _ = _git("add", "--all")
    _ = _git("commit", "--allow-empty", "-qm", "synthetic E topology")
    return _git("rev-parse", "HEAD").strip()


def _e_topology_merge(*parents: str) -> str:
    arguments = ["commit-tree", _git("write-tree").strip(), "-m", "synthetic merge"]
    for parent in parents:
        arguments.extend(("-p", parent))
    commit = _git(*arguments).strip()
    _ = _git("update-ref", "HEAD", commit)
    return commit


@pytest.fixture
def evidence_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[str, str]:
    monkeypatch.setattr(sys.modules[__name__], "REPOSITORY_ROOT", tmp_path)
    _ = _git("init", "-q")
    for key, value in {
        "user.name": "Synthetic Fixture",
        "user.email": "fixture@example.invalid",
        "commit.gpgsign": "false",
        "core.autocrlf": "false",
        "core.hooksPath": os.devnull,
    }.items():
        _ = _git("config", key, value)
    _ = _e_topology_commit({ARTIFACT: b'{"verdict":"REJECT"}\n'})
    nulls = "".join(f"{name}: str | None = None\n" for name in APPROVED_CONSTANT_NAMES)
    parenthesized = getattr(request, "param", "plain") == "parenthesized"
    if parenthesized:
        nulls = nulls.replace("= None\n", "= (\n    None  # retained comment\n)\n")
    source = _e_topology_commit(
        {
            ARTIFACT: None,
            VALIDATOR: nulls.encode(),
            "docs/development/completion_ledger.md": b"# Synthetic ledger\r\n",
        }
    )
    raw = json.dumps({"source_sha": source, "label": "é"}, ensure_ascii=False).encode()
    raw += b"\r\n"
    performance_path = PERFORMANCE_DIRECTORY + "/synthetic.json"
    performance = b'{"synthetic":true}\r\n'
    values = (
        source,
        hashlib.sha256(raw).hexdigest(),
        performance_path,
        hashlib.sha256(performance).hexdigest(),
    )
    approved = nulls
    for name, value in zip(APPROVED_CONSTANT_NAMES, values, strict=True):
        monkeypatch.setattr(sys.modules[__name__], name, value)
        if parenthesized:
            approved = approved.replace("    None", f'    "{value}"', 1)
        else:
            approved = approved.replace("= None", f'= "{value}"', 1)
    evidence = _e_topology_commit(
        {
            ARTIFACT: raw,
            REPRODUCTION: (REPRODUCTION_FRONT_MATTER + "\nsynthetic\n").encode(),
            VALIDATOR: approved.encode(),
            performance_path: performance,
        }
    )
    return source, evidence


def _check_current_e() -> None:
    test_the_artifact_introducing_commit_directly_parents_the_approved_source()
    test_the_e3_diff_writes_only_the_section_13_5_authorized_paths()
    test_the_e3_diff_changes_only_the_four_approved_constant_assignments()


def test_evidence_topology_selects_current_add_with_raw_utf8(
    evidence_git: tuple[str, str],
) -> None:
    _source, evidence = evidence_git
    _ = _e_topology_commit({"later.txt": b"ordinary descendant"})
    assert (
        len(_git("log", "--diff-filter=A", "--format=%H", "--", ARTIFACT).split()) == 2
    )
    assert _locate_evidence_commit() == evidence
    _check_current_e()


@pytest.mark.parametrize(
    "mutation", ["at-s", "side-s", "merge-s", "merge-e", "later-merge", "gap"]
)
def test_evidence_topology_rejects_ancestry(
    evidence_git: tuple[str, str], monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    source, evidence = evidence_git
    ancestor = _git("rev-parse", f"{source}^").strip()
    if mutation == "at-s":
        _ = _git("checkout", "--detach", "-q", source)
    elif mutation == "side-s":
        _ = _e_topology_merge(ancestor, source)
    elif mutation == "merge-s":
        _ = _git("checkout", "--detach", "-q", source)
        merged = _e_topology_merge(source, ancestor)
        monkeypatch.setattr(sys.modules[__name__], "APPROVED_SOURCE_SHA", merged)
        _ = _e_topology_commit({})
    elif mutation in {"merge-e", "later-merge"}:
        _ = _e_topology_merge(source if mutation == "merge-e" else evidence, ancestor)
    else:
        _ = _git("checkout", "--detach", "-q", source)
        _ = _e_topology_commit({"intermediate.txt": b"unruled gap"})
        _ = _e_topology_commit({ARTIFACT: _git_bytes("show", f"{evidence}:{ARTIFACT}")})
    with pytest.raises(AssertionError):
        _check_current_e()


@pytest.mark.parametrize(
    "mutation",
    [
        "artifact-bytes",
        "performance-bytes",
        "normalized-digest",
        "source-binding",
        "path-newline",
        "path-tab",
        "missing-record",
        "preexisting",
        "symlink",
        "gitlink",
        "validator-mode",
        "annotation",
        "expression",
        "comment",
        "logic",
        "wrong-pin",
        "spacing",
    ],
)
def test_evidence_topology_rejects_content(
    evidence_git: tuple[str, str], monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    source, evidence = evidence_git
    files = {
        path: _git_bytes("show", f"{evidence}:{path}")
        for path in _e3_authorized_paths()
    }
    _ = _git("checkout", "--detach", "-q", source)
    if mutation in {"artifact-bytes", "performance-bytes"}:
        path = (
            ARTIFACT if mutation == "artifact-bytes" else str(APPROVED_PERFORMANCE_PATH)
        )
        files[path] += b" "
    elif mutation in {"normalized-digest", "source-binding"}:
        raw = files[ARTIFACT].replace(b"\r\n", b"\n")
        if mutation == "source-binding":
            raw = files[ARTIFACT].replace(source.encode(), evidence.encode())
            files[ARTIFACT] = raw
        monkeypatch.setattr(
            sys.modules[__name__],
            "APPROVED_ARTIFACT_SHA256",
            hashlib.sha256(raw).hexdigest(),
        )
    elif mutation.startswith("path-"):
        files[
            "forbidden\npath" if mutation == "path-newline" else "forbidden\tpath"
        ] = b"extra"
    elif mutation == "missing-record":
        del files[REPRODUCTION]
    elif mutation == "preexisting":
        parent = _e_topology_commit({ARTIFACT: files[ARTIFACT]})
        monkeypatch.setattr(sys.modules[__name__], "APPROVED_SOURCE_SHA", parent)
    elif mutation in {"symlink", "gitlink"}:
        del files[ARTIFACT]
        if mutation == "symlink":
            path = REPOSITORY_ROOT / ARTIFACT
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to("missing-target")
    elif mutation == "annotation":
        files[VALIDATOR] = files[VALIDATOR].replace(b"str | None", b"str", 1)
    elif mutation == "expression":
        files[VALIDATOR] = (
            files[VALIDATOR].replace(b"= ", b"= str(", 1).replace(b"\n", b")\n", 1)
        )
    elif mutation == "comment":
        files[VALIDATOR] = files[VALIDATOR].replace(b"\n", b" # unauthorized\n", 1)
    elif mutation == "logic":
        files[VALIDATOR] += b"changed = True\n"
    elif mutation == "wrong-pin":
        files[VALIDATOR] = files[VALIDATOR].replace(source.encode(), evidence.encode())
    elif mutation == "spacing":
        files[VALIDATOR] = files[VALIDATOR].replace(b" = ", b"  = ", 1)
    _ = _e_topology_commit(files)
    if mutation in {"gitlink", "validator-mode"}:
        if mutation == "gitlink":
            _ = _git(
                "update-index", "--add", "--cacheinfo", f"160000,{evidence},{ARTIFACT}"
            )
        else:
            _ = _git("update-index", "--chmod=+x", VALIDATOR)
        _ = _git("commit", "--amend", "--no-edit", "-q")
    with pytest.raises(AssertionError):
        _check_current_e()


@pytest.mark.parametrize(
    "overlay", ["replace", "replace-blob", "graft-file", "graft-environment"]
)
def test_evidence_topology_rejects_history_overlays(
    evidence_git: tuple[str, str], monkeypatch: pytest.MonkeyPatch, overlay: str
) -> None:
    source, good = evidence_git
    files = {
        path: _git_bytes("show", f"{good}:{path}") for path in _e3_authorized_paths()
    }
    raw = files[ARTIFACT]
    _ = _git("checkout", "--detach", "-q", source)
    if overlay == "replace":
        files["forbidden.py"] = b"unauthorized"
    elif overlay == "replace-blob":
        files[ARTIFACT] += b" "
    else:
        _ = _e_topology_commit({})
    bad = _e_topology_commit(files)
    if overlay == "replace":
        _ = _git("replace", bad, good)
    elif overlay == "replace-blob":
        _ = _git(
            "replace",
            _git("rev-parse", f"{bad}:{ARTIFACT}").strip(),
            _git("rev-parse", f"{good}:{ARTIFACT}").strip(),
        )
    else:
        graft = REPOSITORY_ROOT / (
            ".git/info/grafts" if overlay == "graft-file" else "external-graft"
        )
        graft.parent.mkdir(parents=True, exist_ok=True)
        _ = graft.write_text(f"{bad} {source}\n")
        if overlay == "graft-environment":
            monkeypatch.setenv("GIT_GRAFT_FILE", str(graft))
    # Ordinary Git sees the forged parent/tree/blob, while authentication rejects it.
    assert subprocess.check_output(
        ["git", "rev-list", "--parents", "-n", "1", bad], cwd=REPOSITORY_ROOT
    ).decode().split() == [bad, source]
    assert (
        subprocess.check_output(
            ["git", "show", f"{bad}:{ARTIFACT}"], cwd=REPOSITORY_ROOT
        )
        == raw
    )
    with pytest.raises(AssertionError):
        _check_current_e()


def test_evidence_topology_ignores_environment_and_presentation_redirects(
    evidence_git: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = evidence_git
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        monkeypatch.setenv(key, str(REPOSITORY_ROOT / "missing"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "diff.external")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "/usr/bin/false")
    _ = _git("config", "diff.external", "/usr/bin/false")
    _ = _git("config", "diff.relative", "true")
    _check_current_e()
    assert os.environ["GIT_CONFIG_COUNT"] == "1"


@pytest.mark.parametrize("evidence_git", ["parenthesized"], indirect=True)
def test_evidence_topology_preserves_parenthesized_rhs_comments(
    evidence_git: tuple[str, str],
) -> None:
    _ = evidence_git
    _check_current_e()


@pytest.mark.parametrize(
    "mutation",
    ["facts", "prior-bytes", "forged-pin", "acceptance", "contract", "mode", "delete"],
)
def test_evidence_topology_restricts_optional_factual_ledger(
    evidence_git: tuple[str, str], mutation: str
) -> None:
    source, evidence = evidence_git
    ledger = "docs/development/completion_ledger.md"
    before = _git_bytes("show", f"{source}:{ledger}")
    files: dict[str, bytes | None] = {
        path: _git_bytes("show", f"{evidence}:{path}")
        for path in _e3_authorized_paths()
    }
    suffix = _e3_factual_ledger_suffix()
    additions = {
        "acceptance": b"M3 is accepted.\n",
        "contract": b"Future tolerances may be relaxed.\n",
    }
    if mutation == "forged-pin":
        suffix = suffix.replace(source.encode(), ("0" * 40).encode())
    files[ledger] = (
        None
        if mutation == "delete"
        else (b"rewritten prior ledger\n" if mutation == "prior-bytes" else before)
        + suffix
        + additions.get(mutation, b"")
    )
    _ = _git("checkout", "--detach", "-q", source)
    _ = _e_topology_commit(files)
    if mutation == "mode":
        _ = _git("update-index", "--chmod=+x", ledger)
        _ = _git("commit", "--amend", "--no-edit", "-q")
    if mutation == "facts":
        _check_current_e()
    else:
        with pytest.raises(AssertionError, match="ledger companion|regular-file"):
            _check_current_e()


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


# First-S readiness is structural and generation-only; these fixtures never solve.
def _ready_source_blobs(module):
    keys = module._SOURCE_CONTRACT_KEYS

    def literal_dict(kind, values=None):
        replacements = values or {}
        return (
            "{"
            + ", ".join(
                f"{key!r}: {replacements.get(key, 'None')}" for key in keys[kind]
            )
            + "}"
        )

    input_literal = literal_dict(
        "input",
        {
            "schema_version": "MMODE_CHARACTERIZATION_INPUT_DOMAIN",
            "phase_input_identity_manifest": "phase",
            "phase_input_identity_sha256": "phase_digest",
        },
    )
    record_literal = literal_dict(
        "record",
        {
            "characterization_input_manifest": "characterization_input",
            "input_identity_sha256": "object_digest(MMODE_CHARACTERIZATION_INPUT_"
            "DOMAIN, characterization_input)",
        },
    )
    result = (
        "MMODE_CHARACTERIZATION_INPUT_DOMAIN = "
        "'radiosim.sci004.characterization-input.v2'\n"
        "_MMODE_PHASE_INPUT_DOMAIN = 'radiosim.mmode-input-identity.v1'\n"
        + "".join(
            f"{name} = {keys[kind]!r}\n"
            for name, kind in (
                ("MMODE_CHARACTERIZATION_RECORD_KEYS", "record"),
                ("_CHARACTERIZATION_INPUT_KEYS", "input"),
                ("_MMODE_PHASE_INPUT_KEYS", "phase"),
            )
        )
        + "def _characterization_input_manifest(result, family_id, "
        "phase_input_identity_manifest):\n"
        "    phase = _characterization_mapping(phase_input_identity"
        "_manifest, _MMODE_PHASE_INPUT_KEYS, "
        "field_name='phase_input_identity_manifest')\n"
        "    if phase['schema_version'] != "
        "_MMODE_PHASE_INPUT_DOMAIN:\n        raise "
        "ValueError('schema')\n"
        "    phase_digest = object_digest(_MMODE_PHASE_INPUT_DOMAIN, phase)\n"
        f"    manifest = {input_literal}\n"
        "    if tuple(manifest) != _CHARACTERIZATION_INPUT_KEYS:\n  "
        "      raise InvalidResultError('shape')\n"
        "    return manifest\n"
        "def mmode_characterization_record(result, *, family_id, "
        "phase_input_identity_manifest):\n"
        "    characterization_input = "
        "_characterization_input_manifest(result, family_id, "
        "phase_input_identity_manifest)\n"
        f"    record = {record_literal}\n"
        "    if tuple(record) != "
        "MMODE_CHARACTERIZATION_RECORD_KEYS:\n        raise "
        "InvalidResultError('shape')\n"
        "    return record\n"
    )
    declarations = (
        "SECTION_11_FAMILIES = ('mmode_single_scalar_mode', "
        "'mmode_point_stokes_i', 'mmode_point_full_stokes', "
        "'mmode_circular_receptor')\n"
        + "".join(
            f"{name} = {keys[kind]!r}\n"
            for name, kind in (
                ("ENVELOPE_KEYS", "envelope"),
                ("FINGERPRINT_ROW_KEYS", "fingerprint"),
                ("RED_FAILURE_RECORD_KEYS", "red"),
            )
        )
    )
    results_literal = literal_dict("results", {"fingerprint_rows": "fingerprint_rows"})
    document_literal = literal_dict(
        "envelope",
        {"phase_ranges": "state['phase_ranges']", "results": results_literal},
    )
    tool = declarations + (
        "def build_phase3_evidence(source_sha):\n"
        "    with tempfile.TemporaryDirectory() as scratch:\n"
        "        fingerprint_rows = _fingerprint_rows(results, bundles)\n"
        "        groups = {}\n"
        f"    document = {document_literal}\n    return document\n"
        "def _red_failure_record_reference(red):\n"
        f"    return {literal_dict('red')}\n"
        "def _fingerprint_rows(results, bundles):\n    from "
        "radiosim.core.result import "
        "mmode_characterization_record\n    rows = []\n    for "
        "family_id in SECTION_11_FAMILIES:\n"
        "        result = results[family_id]\n"
        "        record = mmode_characterization_record(result, "
        "family_id=family_id, "
        "phase_input_identity_manifest=bundles[family_id]['input_id"
        "entity_manifest'])\n"
        f"        rows.append({literal_dict('fingerprint')})\n    return rows\n"
        "def validate_evidence_artifact(document):\n    envelope = "
        "validate_evidence_document(document)\n"
        "    history.validate_phase_ranges(envelope['phase_ranges']"
        ", design_sha=envelope['design_sha'], "
        "red_sha=envelope['red_commit_sha'], "
        "source_sha=envelope['source_sha'], root=REPOSITORY_ROOT)\n"
        "    return envelope\n"
    )
    blobs = {"src/radiosim/core/result.py": result, TOOL: tool}
    for path, names in module._SOURCE_SENTINELS.items():
        blobs[path] = "".join(f"{name}: str | None = None\n" for name in names)
    blobs[VALIDATOR] += declarations
    return blobs


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "v1",
        "old_return",
        "optional_input",
        "input_schema",
        "phase_schema",
        "old_fingerprint",
        "missing_third",
        "missing_ranges",
        "wrong_ranges",
        "unwired_ranges",
        "decoy_ranges",
        "wrong_family",
        "lambda_ranges",
        "short_circuit_ranges",
        "false_schema_literal",
    ],
)
def test_source_readiness_checks_actual_schema_and_wiring(monkeypatch, mutation):
    import ast

    module = _tool()
    blobs = _ready_source_blobs(module)
    result = "src/radiosim/core/result.py"
    if mutation == "v1":
        blobs[result] = blobs[result].replace(
            "characterization-input.v2", "characterization-input.v1"
        )
    elif mutation == "old_return":
        blobs[result] = blobs[result].replace(
            "'characterization_input_manifest': characterization_input, ", ""
        )
    elif mutation == "optional_input":
        blobs[result] = blobs[result].replace(
            "*, family_id, phase_input_identity_manifest)",
            "*, family_id, phase_input_identity_manifest=None)",
        )
    elif mutation in ("input_schema", "phase_schema"):
        blobs[result] = (
            blobs[result].replace("'polarization_basis', ", "")
            if mutation == "input_schema"
            else blobs[result].replace("'sky_component_rows', ", "")
        )
    elif mutation == "old_fingerprint":
        blobs[TOOL] = blobs[TOOL].replace(
            "'characterization_time_manifest': None, ", ""
        )
    elif mutation == "missing_third":
        blobs[TOOL] = blobs[TOOL].replace(", 'fingerprint_post_source_delta': None", "")
    elif mutation == "missing_ranges":
        blobs[TOOL] = blobs[TOOL].replace("'phase_ranges': state['phase_ranges'], ", "")
    elif mutation == "wrong_ranges":
        blobs[TOOL] = blobs[TOOL].replace("state['phase_ranges']", "{}")
    elif mutation in ("unwired_ranges", "decoy_ranges"):
        blobs[TOOL] = blobs[TOOL].replace(
            "    history.validate_phase_ranges",
            "    if False:\n        history.validate_phase_ranges"
            if mutation == "decoy_ranges"
            else "    other_validator",
        )
    elif mutation in ("lambda_ranges", "short_circuit_ranges"):
        blobs[TOOL] = blobs[TOOL].replace(
            "    history.validate_phase_ranges",
            "    unused = lambda: history.validate_phase_ranges"
            if mutation == "lambda_ranges"
            else "    False and history.validate_phase_ranges",
        )
    elif mutation == "false_schema_literal":
        blobs[result] = blobs[result].replace(
            "'schema_version': MMODE_CHARACTERIZATION_INPUT_DOMAIN",
            "'schema_version': 'radiosim.sci004.characterization-input.v1'",
        )
    elif mutation == "wrong_family":
        blobs[TOOL] = blobs[TOOL].replace("bundles[family_id]", "bundles['other']")
    monkeypatch.setattr(
        module, "_source_tree", lambda _head, path: ast.parse(blobs[path])
    )
    if mutation is None:
        module._require_source_schema_contract(FORTY)
    else:
        with pytest.raises(module.EvidenceError, match="source contract"):
            module._require_source_schema_contract(FORTY)


@pytest.mark.parametrize(
    "source",
    [
        "FLAG = 'None'",
        "FLAG = str(None)",
        "FLAG = None\nFLAG = None",
        "if False:\n    FLAG = None",
        "FLAG = None\ndel FLAG",
        "FLAG = None\nfrom other import FLAG",
        "FLAG = None\ndef f(FLAG): pass",
        "FLAG = None\nFLAG += 1",
    ],
)
def test_source_readiness_requires_unique_ast_null_sentinels(source):
    import ast

    module = _tool()
    with pytest.raises(module.EvidenceError, match="source contract"):
        module._source_literal(ast.parse(source), "FLAG", None)
    module._source_literal(ast.parse("FLAG: str | None = None"), "FLAG", None)


@pytest.fixture
def ready_source_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, ...]:
    from typing import cast

    from tests.unit import test_sci004_phase3_dependency as dependency
    from tests.unit.test_sci004_phase3_history import (
        PhaseObjects,
        phase_objects,
        source_design_object,
    )
    from tools import sci004_phase3_history as history

    objects = cast(PhaseObjects, phase_objects.__wrapped__(tmp_path, monkeypatch))
    root, commit, _, _, _, red, prior_source, _ = objects
    module = _tool()
    blobs = _ready_source_blobs(module)
    blobs[TOOL] = f"SCI004_R3_TERMINAL_SHA = {red!r}\n" + blobs[TOOL]
    edits = {
        path: history._git(root, "show", f"{prior_source}:{path}")
        for path in history.SOURCE_PATHS
    }
    edits.update({path: raw.encode() for path, raw in blobs.items()})
    # First S retains its exact R parent before the two reviewed design edges.
    first = commit(red, edits)
    source_designs: list[history.SourceDesignEdge] = []
    parent = first
    for edge in history.SOURCE_DESIGN_EDGES:
        created = source_design_object(root, commit, parent, edge)
        source_designs.append(created)
        parent = created.sha
    monkeypatch.setattr(history, "SOURCE_DESIGN_EDGES", tuple(source_designs))
    monkeypatch.setattr(history, "HISTORICAL_SOURCE_DESIGN_SHA", source_designs[0].sha)
    monkeypatch.setattr(history, "SOURCE_DESIGN_SHA", source_designs[1].sha)
    head = commit(parent, dict.fromkeys(history.DISPOSAL_PINS))
    for path, content in edits.items():
        raw = content.decode()
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = destination.write_text(raw)
    monkeypatch.setattr(module, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(dependency, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(
        dependency, "APPROVED_SCI004_D_SHA", history.OPERATIVE_DESIGN_SHA
    )
    original_peel = dependency._peel_to_commit

    def fixture_peel(ref: str) -> str:
        return head if ref == "HEAD" else original_peel(ref)

    def fixture_preflight(*_args: Any) -> dict[str, str]:
        return {"source_sha": head}

    monkeypatch.setattr(dependency, "_peel_to_commit", fixture_peel)
    monkeypatch.setattr(module, "preflight", fixture_preflight)
    monkeypatch.setattr(
        module, "_red_commit_sha", lambda: dependency.resolve_r3_replay_anchor().commit
    )
    monkeypatch.setattr(module, "_design_sha", lambda: history.SOURCE_DESIGN_SHA)
    return module, history, dependency, root, red, first, head


def test_source_readiness_accepts_complete_synthetic_source_history(
    ready_source_objects,
):
    module, history, dependency, _, red, first, head = ready_source_objects
    assert dependency._commit_parents(first) == (red,)
    assert dependency._terminal_r3_metadata(red) is None
    assert dependency._terminal_r3_metadata(first) == red
    state = module.source_readiness(head, ())
    assert state["red_commit_sha"] == red
    assert state["phase_ranges"]["source"]["terminal_sha"] == head
    assert set(history.SOURCE_PATHS) | set(history.DISPOSAL_PINS) == {
        path
        for entry in state["phase_ranges"]["source"]["commits"]
        if entry["role"] != "source-design-successor"
        for path in entry["paths"]
    }


@pytest.mark.parametrize(
    "mutation", ["output", "symlink", "partial", "sentinel", "working_source"]
)
def test_source_readiness_refuses_before_any_measurement(
    ready_source_objects, monkeypatch, mutation
):
    module, history, _, root, _, first, head = ready_source_objects
    if mutation in ("output", "symlink"):
        path = root / next(iter(history.DISPOSAL_PINS))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(root / "absent") if mutation == "symlink" else path.write_bytes(
            b"rejected"
        )
    elif mutation == "partial":
        monkeypatch.setattr(module, "preflight", lambda *_args: {"source_sha": first})
    elif mutation == "sentinel":
        original = module._source_tree
        import ast

        monkeypatch.setattr(
            module,
            "_source_tree",
            lambda sha, path: ast.parse("APPROVED_SOURCE_SHA = 'old'\n")
            if path == VALIDATOR
            else original(sha, path),
        )
    else:
        (root / "src/radiosim/core/result.py").write_text("# different source\n")
    monkeypatch.setattr(
        module,
        "_red_failure_record_reference",
        lambda *_: pytest.fail("measurement boundary reached"),
    )
    with pytest.raises(module.EvidenceError):
        module.build_phase3_evidence(head)


@pytest.mark.parametrize(
    "foreign",
    [
        None,
        "radiosim",
        "radiosim.core.result",
        "radiosim.core.mmode.solver",
        "radiosim.core.mmode.types",
        "missing_file",
        "stale_domain",
        "stale_keys",
        "optional_keyword",
        "missing_keyword",
        "missing_bridge",
        "optional_bridge",
        "mutable_bridge",
    ],
)
def test_generation_requires_own_scientific_imports(tmp_path, monkeypatch, foreign):
    import importlib
    from dataclasses import field, make_dataclass
    from types import SimpleNamespace

    module = _tool()
    names = (
        "radiosim",
        "radiosim.core.result",
        "radiosim.core.mmode.solver",
        "radiosim.core.mmode.types",
    )
    paths = (
        "src/radiosim/__init__.py",
        "src/radiosim/core/result.py",
        "src/radiosim/core/mmode/solver.py",
        "src/radiosim/core/mmode/types.py",
    )
    origins = {}
    for name, path in zip(names, paths, strict=True):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# fixture\n")
        origins[name] = str(target)
    if foreign == "missing_file":
        origins["radiosim"] = None
    elif foreign in names:
        other = tmp_path / "foreign.py"
        other.write_text("# another checkout\n")
        origins[foreign] = str(other)
    modules = {
        name: SimpleNamespace(__file__=origin) for name, origin in origins.items()
    }
    result = modules["radiosim.core.result"]
    result.MMODE_CHARACTERIZATION_INPUT_DOMAIN = (
        "radiosim.sci004.characterization-input.v2"
    )
    result.MMODE_CHARACTERIZATION_RECORD_KEYS = module._SOURCE_CONTRACT_KEYS["record"]

    def factory(result, *, family_id, phase_input_identity_manifest):
        pass

    result.mmode_characterization_record = factory
    bridge = [("input_identity_sha256", str)]
    if foreign == "stale_domain":
        result.MMODE_CHARACTERIZATION_INPUT_DOMAIN = (
            "radiosim.sci004.characterization-input.v1"
        )
    elif foreign == "stale_keys":
        result.MMODE_CHARACTERIZATION_RECORD_KEYS = (
            result.MMODE_CHARACTERIZATION_RECORD_KEYS[:7]
        )
    elif foreign == "optional_keyword":
        result.mmode_characterization_record = (
            lambda result, *, family_id, phase_input_identity_manifest=None: None
        )
    elif foreign == "missing_keyword":
        result.mmode_characterization_record = lambda result, *, family_id: None
    elif foreign == "missing_bridge":
        bridge = []
    elif foreign == "optional_bridge":
        bridge = [("input_identity_sha256", str, field(default=None))]
    modules["radiosim.core.mmode.solver"].MModeSolverSnapshot = make_dataclass(
        "SnapshotFixture", bridge, frozen=foreign != "mutable_bridge"
    )
    monkeypatch.setattr(module, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(importlib, "import_module", lambda name: modules[name])
    if foreign is None:
        module._require_generation_source_imports()
    else:
        with pytest.raises(module.EvidenceError, match="generation"):
            module._require_generation_source_imports()


def test_first_source_metadata_names_the_reviewed_terminal_red(monkeypatch):
    from tests.unit import test_sci004_phase3_dependency as dependency

    module = _tool()
    terminal = "567f9ac68730044fc8e887930d3531d794534412"
    assert module.SCI004_R3_TERMINAL_SHA == terminal
    assert dependency._terminal_r3_metadata(terminal) is None
    raw = (REPOSITORY_ROOT / TOOL).read_bytes()
    monkeypatch.setattr(dependency, "_tree_blob", lambda *_args: raw)
    assert dependency._terminal_r3_metadata("current source candidate") == terminal


@pytest.mark.parametrize(
    "mutation",
    [
        "unreachable_ranges",
        "yield_ranges",
        "empty_iteration",
        "wrong_result",
        "wrong_family_keyword",
        "break",
        "continue",
        "wrong_return",
        "wrong_target",
        "dirty_rows",
        "foreign_factory",
        "clear_record",
        "alias_record",
        "clear_manifest",
        "mutating_guard",
        "unused_phase_call",
        "unused_factory_call",
    ],
)
def test_source_readiness_rejects_control_flow_and_dictionary_escape(
    monkeypatch, mutation
):
    import ast

    module = _tool()
    blobs = _ready_source_blobs(module)
    result = "src/radiosim/core/result.py"
    if mutation in ("unreachable_ranges", "yield_ranges"):
        prefix = (
            "return envelope" if mutation == "unreachable_ranges" else "yield envelope"
        )
        blobs[TOOL] = blobs[TOOL].replace(
            "    history.validate_phase_ranges",
            f"    {prefix}\n    history.validate_phase_ranges",
        )
    elif mutation == "empty_iteration":
        blobs[TOOL] = blobs[TOOL].replace("in SECTION_11_FAMILIES:", "in ():")
    elif mutation == "wrong_result":
        blobs[TOOL] = blobs[TOOL].replace(
            "result = results[family_id]", "result = results['other']"
        )
    elif mutation == "wrong_family_keyword":
        blobs[TOOL] = blobs[TOOL].replace("family_id=family_id,", "family_id='other',")
    elif mutation in ("break", "continue"):
        blobs[TOOL] = blobs[TOOL].replace(
            "        result =", f"        {mutation}\n        result ="
        )
    elif mutation == "wrong_return":
        blobs[TOOL] = blobs[TOOL].replace("    return rows", "    return []")
    elif mutation == "wrong_target":
        blobs[TOOL] = blobs[TOOL].replace("for family_id in", "for other in")
    elif mutation == "dirty_rows":
        blobs[TOOL] = blobs[TOOL].replace("rows = []", "rows = [{}]")
    elif mutation == "foreign_factory":
        blobs[TOOL] = blobs[TOOL].replace(
            "from radiosim.core.result import", "from other import"
        )
    elif mutation in ("clear_record", "alias_record"):
        change = (
            "dict.clear(record)"
            if mutation == "clear_record"
            else "alias = record\n    alias.clear()"
        )
        blobs[result] = blobs[result].replace(
            "    if tuple(record)", f"    {change}\n    if tuple(record)"
        )
    elif mutation == "clear_manifest":
        blobs[result] = blobs[result].replace(
            "    if tuple(manifest)", "    dict.clear(manifest)\n    if tuple(manifest)"
        )
    elif mutation == "mutating_guard":
        blobs[result] = blobs[result].replace(
            "raise InvalidResultError('shape')",
            "raise InvalidResultError(manifest.clear())",
        )
    elif mutation == "unused_phase_call":
        blobs[result] = blobs[result].replace(
            "    phase = _characterization_mapping",
            "    phase = {}\n    unused = _characterization_mapping",
        )
    else:
        blobs[result] = blobs[result].replace(
            "    characterization_input = _characterization_input_manifest",
            "    characterization_input = {}\n    unused = "
            "_characterization_input_manifest",
        )
    monkeypatch.setattr(
        module, "_source_tree", lambda _head, path: ast.parse(blobs[path])
    )
    with pytest.raises(module.EvidenceError, match="source contract"):
        module._require_source_schema_contract(FORTY)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_families",
        "empty_families",
        "substituted_families",
        "rebound_families",
        "missing_call",
        "stale_call",
        "nested_decoy",
        "disconnected_rows",
        "late_rows",
        "other_workspace",
        "duplicate_rows",
    ],
)
def test_source_readiness_binds_generator_family_inventory_and_rows(
    monkeypatch, mutation
):
    import ast

    module = _tool()
    blobs = _ready_source_blobs(module)
    source = blobs[TOOL]
    family_line = source.splitlines()[0]
    if mutation == "missing_families":
        source = source.replace(family_line + "\n", "", 1)
    elif mutation == "empty_families":
        source = source.replace(family_line, "SECTION_11_FAMILIES = ()", 1)
    elif mutation == "substituted_families":
        source = source.replace("mmode_point_stokes_i", "other_family", 1)
    elif mutation == "rebound_families":
        source += "SECTION_11_FAMILIES = ()\n"
    elif mutation == "missing_call":
        source = source.replace(
            "        fingerprint_rows = _fingerprint_rows(results, bundles)\n", ""
        )
    elif mutation == "stale_call":
        source = source.replace(
            "fingerprint_rows = _fingerprint_rows(results, bundles)",
            "fingerprint_rows = _fingerprint_rows(results)",
        )
    elif mutation == "nested_decoy":
        source = source.replace(
            "        fingerprint_rows =",
            "        if False:\n            fingerprint_rows =",
        )
    elif mutation == "disconnected_rows":
        source = source.replace(
            "'fingerprint_rows': fingerprint_rows", "'fingerprint_rows': []"
        )
    elif mutation == "late_rows":
        source = source.replace(
            "        fingerprint_rows = _fingerprint_rows(results, "
            "bundles)\n        groups = {}",
            "        groups = {}\n        fingerprint_rows = "
            "_fingerprint_rows(results, bundles)",
        )
    elif mutation == "other_workspace":
        source = source.replace("tempfile.TemporaryDirectory()", "other_workspace()")
    else:
        source = source.replace(
            "    return document", "    fingerprint_rows = []\n    return document"
        )
    assert source != blobs[TOOL]
    blobs[TOOL] = source
    monkeypatch.setattr(
        module, "_source_tree", lambda _head, path: ast.parse(blobs[path])
    )
    with pytest.raises(module.EvidenceError, match="source contract"):
        module._require_source_schema_contract(FORTY)


@pytest.fixture
def synthetic_runtime_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any, dict[str, Any], list[object], Any]:
    """Isolate the public adapter copy seam; do not claim a numerical solve."""
    from types import SimpleNamespace
    from typing import cast

    import numpy as np

    from radiosim.backends.numpy_backend import NumPyBackend
    from radiosim.core.mmode import solver
    from radiosim.core.result import MModeSolverResultProvenance
    from radiosim.simulator.base import SkySolveRequest
    from tests.unit.test_io.test_standard_visibility import build_mmode_result

    result = build_mmode_result(tmp_path)
    assert isinstance(result.solver, MModeSolverResultProvenance)
    snapshot = result.solver.snapshot
    assert isinstance(snapshot, solver.MModeSolverSnapshot)
    request = cast(
        SkySolveRequest,
        SimpleNamespace(
            sky_representation="point_sources",
            sky_model=SimpleNamespace(healpix=None),
            beam_system=SimpleNamespace(
                state=SimpleNamespace(resolved=SimpleNamespace(assignments=()))
            ),
            backend=NumPyBackend(),
        ),
    )
    solved: dict[str, Any] = {
        "gate": snapshot.direct_gate,
        "grid": SimpleNamespace(sidereal_samples=snapshot.sidereal_samples),
        "dimensions": SimpleNamespace(
            lmax=snapshot.lmax,
            mmax=snapshot.mmax,
            quadrature_nside=snapshot.quadrature_nside,
        ),
        "certificate": SimpleNamespace(
            certificate_sha256=snapshot.frame_certificate_sha256,
            frozen_gauss128_cube_sha256=snapshot.frozen_gauss128_cube_sha256,
            frozen_enclosure_error_cube_sha256=(
                snapshot.frozen_enclosure_error_cube_sha256
            ),
        ),
        "cube": result.visibilities.copy(),
        "point_cirs": np.zeros((snapshot.component_element_counts[0], 3)),
        "execution_path": snapshot.execution_path,
        "tangent_polarization_frame": snapshot.tangent_polarization_frame,
        "frame": SimpleNamespace(iers_table_sha256=snapshot.iers_table_sha256),
        "input_identity_sha256": _digest("synthetic-pipeline-input"),
    }
    calls: list[object] = []

    def pipeline(actual: object) -> dict[str, Any]:
        assert actual is request
        calls.append(actual)
        return solved

    monkeypatch.setattr(solver, "_mmode_pipeline", pipeline)
    return solver, request, solved, calls, result


@pytest.mark.parametrize("identity", [_digest("bridge-one"), _digest("bridge-two")])
def test_runtime_bridge_copies_the_single_pipeline_identity(
    synthetic_runtime_bridge: tuple[Any, Any, dict[str, Any], list[object], Any],
    identity: str,
) -> None:
    """The adapter retains the pipeline identity without changing its cube."""
    import numpy as np

    solver, request, solved, calls, result = synthetic_runtime_bridge
    solved["input_identity_sha256"] = identity
    original_cube = solved["cube"].tobytes()
    outcome = solver.solve_mmode(request)
    snapshot = outcome.solver_record
    assert calls == [request]
    assert snapshot.input_identity_sha256 == identity
    assert identity != result.solver.snapshot.input_identity_sha256
    assert identity != result.scientific_sha256
    assert solved["cube"].tobytes() == original_cube
    assert np.asarray(outcome.receptor_visibilities).tobytes() == original_cube
    assert snapshot.as_mapping() == result.solver.snapshot.as_mapping()
    assert snapshot.solver_snapshot_sha256() == (
        result.solver.snapshot.solver_snapshot_sha256()
    )
    solved["input_identity_sha256"] = _digest("later-pipeline-mutation")
    assert snapshot.input_identity_sha256 == identity


def test_runtime_bridge_does_not_invent_a_missing_pipeline_identity(
    synthetic_runtime_bridge: tuple[Any, Any, dict[str, Any], list[object], Any],
) -> None:
    solver, request, solved, calls, _ = synthetic_runtime_bridge
    del solved["input_identity_sha256"]
    with pytest.raises(KeyError, match="input_identity_sha256"):
        solver.solve_mmode(request)
    assert calls == [request]


def test_runtime_bridge_is_required_immutable_and_absent_from_serialization(
    synthetic_runtime_bridge: tuple[Any, Any, dict[str, Any], list[object], Any],
) -> None:
    from dataclasses import MISSING, FrozenInstanceError, fields, replace

    from radiosim.core.result import (
        MMODE_SOLVER_SNAPSHOT_KEYS,
        LoadedMModeSolverSnapshot,
    )

    solver, request, _, _, _ = synthetic_runtime_bridge
    snapshot = solver.solve_mmode(request).solver_record
    identity_field = next(
        item for item in fields(snapshot) if item.name == "input_identity_sha256"
    )
    assert identity_field.type in (str, "str")
    assert identity_field.default is MISSING
    assert identity_field.default_factory is MISSING
    with pytest.raises(FrozenInstanceError):
        snapshot.input_identity_sha256 = _digest("reassignment")
    prior_fields = {
        item.name: getattr(snapshot, item.name)
        for item in fields(snapshot)
        if item.name != "input_identity_sha256"
    }
    with pytest.raises(TypeError, match="input_identity_sha256"):
        solver.MModeSolverSnapshot(**prior_fields)
    other = replace(snapshot, input_identity_sha256=_digest("foreign-runtime"))
    assert other.input_identity_sha256 != snapshot.input_identity_sha256
    assert len(snapshot.as_mapping()) == 20
    assert tuple(snapshot.as_mapping()) == MMODE_SOLVER_SNAPSHOT_KEYS
    assert "input_identity_sha256" not in snapshot.as_mapping()
    assert other.as_mapping() == snapshot.as_mapping()
    assert other.to_snapshot() == snapshot.to_snapshot()
    assert _canonical(other.to_snapshot()) == _canonical(snapshot.to_snapshot())
    assert other.solver_snapshot_sha256() == snapshot.solver_snapshot_sha256()
    loaded = LoadedMModeSolverSnapshot(stored=snapshot.to_snapshot())
    assert not hasattr(loaded, "input_identity_sha256")


def test_result_runtime_identity_requires_the_live_snapshot_owner(
    synthetic_runtime_bridge: tuple[Any, Any, dict[str, Any], list[object], Any],
) -> None:
    from types import SimpleNamespace

    from radiosim.core.result import (
        InvalidResultError,
        LoadedMModeSolverSnapshot,
        MModeSolverResultProvenance,
    )

    solver, request, solved, _, _ = synthetic_runtime_bridge
    snapshot = solver.solve_mmode(request).solver_record
    provenance = MModeSolverResultProvenance(snapshot=snapshot)
    assert provenance.input_identity_sha256 == solved["input_identity_sha256"]
    assert "input_identity_sha256" not in provenance.as_mapping()
    for foreign in (
        LoadedMModeSolverSnapshot(stored=snapshot.as_mapping()),
        SimpleNamespace(input_identity_sha256=solved["input_identity_sha256"]),
        None,
    ):
        with pytest.raises(InvalidResultError, match="requires a live"):
            _ = MModeSolverResultProvenance(snapshot=foreign).input_identity_sha256


@pytest.mark.parametrize(
    "identity", [None, True, 1, b"a" * 64, "", "a" * 63, "A" * 64, "g" * 64]
)
def test_result_runtime_identity_rejects_malformed_owned_values(
    synthetic_runtime_bridge: tuple[Any, Any, dict[str, Any], list[object], Any],
    identity: object,
) -> None:
    from dataclasses import replace

    from radiosim.core.result import InvalidResultError, MModeSolverResultProvenance

    solver, request, _, _, _ = synthetic_runtime_bridge
    snapshot = solver.solve_mmode(request).solver_record
    malformed = replace(snapshot, input_identity_sha256=identity)
    assert malformed.as_mapping() == snapshot.as_mapping()
    with pytest.raises(InvalidResultError, match="not a SHA-256"):
        _ = MModeSolverResultProvenance(snapshot=malformed).input_identity_sha256


@pytest.mark.parametrize("consumer", ["fingerprints", "ci"])
def test_characterization_consumers_use_each_familys_phase_preimage(
    monkeypatch: pytest.MonkeyPatch, consumer: str
) -> None:
    """Require same-family argument wiring without running a scientific solve."""
    from radiosim.core import result as production

    module = _tool()
    results = {family: object() for family in module.SECTION_11_FAMILIES}
    bundles = {
        family: {"input_identity_manifest": {"synthetic_family": family}}
        for family in results
    }
    calls: list[str] = []

    def record(
        actual: object, *, family_id: str, phase_input_identity_manifest: Any
    ) -> dict[str, Any]:
        assert actual is results[family_id]
        assert (
            phase_input_identity_manifest
            is bundles[family_id]["input_identity_manifest"]
        )
        calls.append(family_id)
        return {
            "input_identity_sha256": SIXTY_FOUR,
            "era_utc_grid_sha256": SIXTY_FOUR,
            "solver_snapshot": {},
            "raw_cube_sha256": SIXTY_FOUR,
            "scientific_sha256": SIXTY_FOUR,
        }

    monkeypatch.setattr(production, "mmode_characterization_record", record)

    def observation_set(family_id: str) -> dict[str, tuple[str, ...]]:
        assert family_id in results
        return {"synthetic-cell": (SIXTY_FOUR,)}

    monkeypatch.setattr(
        production, "mmode_characterization_observation_set", observation_set
    )
    rows = (
        module._fingerprint_rows(results, bundles)
        if consumer == "fingerprints"
        else module._ci_artifacts(results, FORTY, bundles)
    )
    assert calls == list(module.SECTION_11_FAMILIES)
    assert [row["family_id"] for row in rows] == calls


@pytest.fixture(scope="module")
def genuine_phase_synthetic_result(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Any, dict[str, Any]]:
    """Pair genuine input preparation with an explicitly synthetic output cube.

    This fixture stops before certificates or harmonic work. It tests schemas
    and visible-result joins; real public family runs remain separate gates.
    """
    from dataclasses import replace
    from importlib import import_module
    from unittest.mock import patch

    import numpy as np

    from radiosim.api.simulator import Simulator
    from radiosim.core.mmode.time import CanonicalEraGrid
    from radiosim.core.mmode.types import derive_mmode_dimensions
    from radiosim.core.result import (
        MModeSolverResultProvenance,
        build_simulation_result,
    )
    from tests.unit.test_io.test_standard_visibility import build_mmode_result

    # This synthetic fixture instruments private preparation seams explicitly;
    # production callers still reach them only through the public solve path.
    solver: Any = import_module("radiosim.core.mmode.solver")
    captured: list[Simulator] = []
    original = Simulator.from_mapping

    def resolve(*args: Any, **kwargs: Any) -> Simulator:
        simulator = original(*args, **kwargs)
        captured.append(simulator)
        return simulator

    with patch.object(Simulator, "from_mapping", side_effect=resolve):
        seed = build_mmode_result(tmp_path_factory.mktemp("phase-schema").resolve())
    (simulator,) = captured
    request = simulator.build_solve_request()
    grid, block = request.era_grid, request.mmode
    assert isinstance(grid, CanonicalEraGrid) and block is not None
    dimensions = derive_mmode_dimensions(
        lmax=int(block.lmax),
        mmax=int(block.mmax),
        quadrature_nside=int(block.quadrature_nside),
    )
    longitude, latitude, height = solver._site_geodetic(request.location)
    frame = solver.build_frozen_frame(
        start_time=grid.start_time_iso,
        longitude_deg=longitude,
        latitude_deg=latitude,
        height_m=height,
    )
    context = solver._kernel_context(request, frame, grid)
    point_cirs, point_stokes, point_icrs = solver._resolve_point_component(
        request, frame, context
    )
    ledger = solver.build_direction_ledger(
        frame=frame,
        dimensions=dimensions,
        point_cirs=point_cirs,
        point_stokes=point_stokes,
        point_icrs=point_icrs,
        native_cirs=np.zeros((0, 3), dtype=np.float64),
        native_stokes=np.zeros((0, context.n_frequencies, 4), dtype=np.float64),
        native_icrs=np.zeros((0, 2), dtype=np.float64),
        native_solid_angle=0.0,
    )
    tangent_frame = solver._resolved_tangent_frame(request, point_stokes)
    phase, digest = solver.build_input_identity(
        request=request,
        grid=grid,
        frame=frame,
        context=context,
        dimensions=dimensions,
        directions=ledger,
        tangent_frame=tangent_frame,
    )
    assert isinstance(seed.solver, MModeSolverResultProvenance)
    assert isinstance(seed.solver.snapshot, solver.MModeSolverSnapshot)
    snapshot = replace(
        seed.solver.snapshot,
        input_identity_sha256=digest,
        component_element_counts=(len(point_cirs),),
        execution_path="polarized"
        if solver._payload_is_polarized(point_stokes)
        else "scalar",
        iers_table_sha256=frame.iers_table_sha256,
        tangent_polarization_frame=tangent_frame,
    )
    result = build_simulation_result(
        receptor_visibilities=seed.visibilities.reshape(
            *seed.visibilities.shape[:3], 2, 2
        ),
        backend=request.backend,
        time_grid=seed.time_grid,
        frequencies_hz=seed.frequencies_hz.tolist(),
        channel_widths_hz=seed.channel_widths_hz.tolist(),
        instrument=seed.instrument,
        selection=seed.selection,
        beam_state=seed.beam_state,
        receptors=seed.receptors,
        jones_terms=request.jones,
        phase_center=seed.phase_center,
        backend_provenance=seed.backend,
        solver_provenance=MModeSolverResultProvenance(snapshot=snapshot),
        resolved_config=seed.resolved_config,
        configuration_provenance=None,
        performance=seed.performance,
        history=("synthetic schema fixture; no scientific solve or acceptance",),
    )
    return result, phase


def _rehash_phase_schema_fixture(phase: dict[str, Any]) -> str:
    """Rehash adversarial preimages so stale digests cannot explain rejection."""
    for field, digest in (
        ("site_manifest", "site_sha256"),
        ("canonical_era_turn_grid", "canonical_era_turn_grid_sha256"),
        ("utc_manifest", "utc_sha256"),
        ("ut1_manifest", "ut1_sha256"),
    ):
        phase[digest] = _object_digest(phase[field]["schema_version"], phase[field])
    era = phase["canonical_era_grid"]
    era["canonical_era_turn_grid_sha256"] = phase["canonical_era_turn_grid_sha256"]
    phase["canonical_era_grid_sha256"] = _object_digest(era["schema_version"], era)
    for row in phase["beam_rows"] + phase["jones_term_rows"]:
        manifest = row["parameter_identity_manifest"]
        row["parameter_identity_sha256"] = _object_digest(
            manifest["schema_version"], manifest
        )
    changed: dict[str, str] = {}
    for row in phase["direction_input_rows"]:
        manifest = row["direction_input_manifest"]
        digest = _object_digest(manifest["schema_version"], manifest)
        changed[row["direction_input_sha256"]] = digest
        row["direction_input_sha256"] = digest
    for row in phase["sky_component_rows"]:
        manifest = row["morphology_identity_manifest"]
        row["morphology_identity_sha256"] = _object_digest(
            manifest["schema_version"], manifest
        )
        row["polarization_frame_sha256"] = _object_digest(
            "radiosim.sky-tangent-polarization.v1", row["polarization_frame"]
        )
        row["direction_input_sha256s"] = [
            changed.get(digest, digest) for digest in row["direction_input_sha256s"]
        ]
    for group in phase["transfer_grid_catalog"]:
        ids = [
            row["direction_input_manifest"]["direction_id"]
            for row in phase["direction_input_rows"]
            if row["direction_input_manifest"].get("source_kind")
            == "transfer_quadrature"
            and row["direction_input_manifest"].get("transfer_role")
            == group["transfer_role"]
            and row["direction_input_manifest"].get("transfer_nside")
            == group["transfer_nside"]
        ]
        group["direction_id_ledger_sha256"] = _object_digest(
            "radiosim.mmode-transfer-grid-direction-ids.v1", ids
        )
    return _object_digest("radiosim.mmode-input-identity.v1", phase)


def test_production_v2_retains_the_complete_owned_input_preimage(
    genuine_phase_synthetic_result: tuple[Any, dict[str, Any]],
) -> None:
    from radiosim.core.result import (
        MMODE_CHARACTERIZATION_RECORD_KEYS,
        mmode_characterization_record,
    )

    result, phase = genuine_phase_synthetic_result
    before = _canonical(phase)
    record = mmode_characterization_record(
        result,
        family_id="mmode_single_scalar_mode",
        phase_input_identity_manifest=phase,
    )
    assert tuple(record) == MMODE_CHARACTERIZATION_RECORD_KEYS
    assert len(record) == 9
    manifest = record["characterization_input_manifest"]
    assert manifest["schema_version"] == "radiosim.sci004.characterization-input.v2"
    assert len(manifest) == 14
    assert _canonical(manifest["phase_input_identity_manifest"]) == before
    assert (
        manifest["phase_input_identity_sha256"] == result.solver.input_identity_sha256
    )
    assert record["input_identity_sha256"] == _object_digest(
        "radiosim.sci004.characterization-input.v2", manifest
    )
    assert record["era_utc_grid_sha256"] == _object_digest(
        "radiosim.sci004.characterization-time.v1",
        record["characterization_time_manifest"],
    )
    manifest["phase_input_identity_manifest"]["frequency_rows"][0][
        "frequency_index"
    ] = 9
    assert _canonical(phase) == before


@pytest.mark.parametrize("ownership", ["foreign", "missing", "malformed"])
def test_production_v2_requires_the_same_live_runtime_identity(
    genuine_phase_synthetic_result: tuple[Any, dict[str, Any]], ownership: str
) -> None:
    from dataclasses import replace

    from radiosim.core.result import (
        InvalidResultError,
        LoadedMModeSolverSnapshot,
        MModeSolverResultProvenance,
        mmode_characterization_record,
    )

    result, phase = genuine_phase_synthetic_result
    snapshot = (
        LoadedMModeSolverSnapshot(stored=result.solver.as_mapping())
        if ownership == "missing"
        else replace(
            result.solver.snapshot,
            input_identity_sha256=_digest("foreign-phase")
            if ownership == "foreign"
            else "invalid",
        )
    )
    # Deliberately forge only the owner on a separate result; the public result
    # constructor correctly forbids dataclasses.replace.
    changed = copy.copy(result)
    object.__setattr__(
        changed, "solver", MModeSolverResultProvenance(snapshot=snapshot)
    )
    with pytest.raises(InvalidResultError):
        _ = mmode_characterization_record(
            changed,
            family_id="mmode_single_scalar_mode",
            phase_input_identity_manifest=phase,
        )


def test_production_v2_rejects_a_rehashed_foreign_phase_against_unchanged_result(
    genuine_phase_synthetic_result: tuple[Any, dict[str, Any]],
) -> None:
    from radiosim.core.result import InvalidResultError, mmode_characterization_record

    result, original = genuine_phase_synthetic_result
    phase = copy.deepcopy(original)
    direction = phase["direction_input_rows"][0]["direction_input_manifest"]
    direction["resolved_stokes_iau_f64be"][0] = _f64be(123.0)
    digest = _rehash_phase_schema_fixture(phase)
    assert digest != result.solver.input_identity_sha256
    with pytest.raises(InvalidResultError, match="identity|same.run"):
        _ = mmode_characterization_record(
            result,
            family_id="mmode_single_scalar_mode",
            phase_input_identity_manifest=phase,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), "radiosim.mmode-input-identity.v99"),
        (("site_manifest", "schema_version"), "radiosim.mmode-site.v99"),
        (("site_manifest", "height_m_f64be"), "7ff8000000000000"),
        (("site_manifest", "itrs_xyz_m_f64be"), []),
        (("canonical_era_turn_grid", "sidereal_samples"), True),
        (("canonical_era_turn_grid", "integration_fraction_ratio"), "2/2"),
        (("canonical_era_turn_grid", "exposure_width_turn"), "1/24"),
        (("canonical_era_grid", "tau_f64be"), _f64be(6.0)),
        (("utc_manifest", "shape"), [True]),
        (("ut1_manifest", "axis_order"), ["foreign"]),
        (("mmode_dimensions", "lmax"), True),
        (("mmode_dimensions", "qcheck"), 16),
        (("antenna_rows", 0, "antenna_index"), False),
        (("baseline_rows", 0, "antenna1_index"), False),
        (("frequency_rows", 0, "frequency_index"), False),
        (("frequency_rows", 0, "center_hz_f64be"), "7ff0000000000000"),
        (("frequency_rows", 0, "width_hz_f64be"), _f64be(-1.0)),
        (("receptor_rows", 0, "antenna_index"), False),
        (("correlation_rows", 0, "correlation_index"), False),
        (("beam_rows", 0, "assigned_antenna_indices"), [False]),
        (("beam_rows", 0, "assigned_antenna_indices"), [999]),
        (("beam_rows", 0, "normalization"), "foreign"),
        (
            ("beam_rows", 0, "parameter_identity_manifest", "scalar_rows", 0, "name"),
            "layout_path",
        ),
        (("sky_component_rows", 0, "component_index"), False),
        (("sky_component_rows", 0, "representation"), "foreign"),
        (("sky_component_rows", 0, "direction_input_sha256s"), [SIXTY_FOUR]),
        (
            ("direction_input_rows", 0, "direction_input_manifest", "source_index"),
            False,
        ),
        (
            ("direction_input_rows", 0, "direction_input_manifest", "source_kind"),
            "foreign",
        ),
        (
            (
                "direction_input_rows",
                0,
                "direction_input_manifest",
                "active_frequency_mask",
            ),
            [False, False],
        ),
        (
            (
                "direction_input_rows",
                0,
                "direction_input_manifest",
                "run_frequency_hz_f64be",
            ),
            [],
        ),
        (
            (
                "direction_input_rows",
                0,
                "direction_input_manifest",
                "resolved_stokes_iau_f64be",
            ),
            [],
        ),
        (
            (
                "direction_input_rows",
                0,
                "direction_input_manifest",
                "integration_weight_f64be",
            ),
            _f64be(2.0),
        ),
        (("transfer_grid_catalog", 0, "expected_direction_count"), 47),
        (("transfer_grid_catalog", 0, "transfer_role"), "foreign"),
        (("precision",), "ultra"),
        (("layout_path",), "/foreign/input.csv"),
        (("direction_input_rows", 0, "direction_input_manifest", "unknown"), 0),
        (
            ("direction_input_rows", 0, "direction_input_manifest", "schema_version"),
            "radiosim.mmode-direction-input.v99",
        ),
        (
            (
                "direction_input_rows",
                0,
                "direction_input_manifest",
                "cirs_direction_f64be",
            ),
            [_f64be(2.0), _f64be(0.0), _f64be(0.0)],
        ),
        (
            ("beam_rows", 0, "parameter_identity_manifest", "array_rows"),
            [dict[str, Any]()],
        ),
        (
            (
                "sky_component_rows",
                0,
                "morphology_identity_manifest",
                "scalar_rows",
                0,
                "value",
            ),
            "01",
        ),
        (
            (
                "direction_input_rows",
                1,
                "direction_input_manifest",
                "resolved_stokes_iau_f64be",
            ),
            [_f64be(1.0)] * 8,
        ),
        (("ut1_manifest", "center_jd1_f64be", 0), _f64be(1e308)),
        (("ut1_manifest", "center_jd1_f64be", 0), _f64be(-1e308)),
    ],
)
def test_production_v2_rejects_rehashed_semantic_mutations(
    genuine_phase_synthetic_result: tuple[Any, dict[str, Any]],
    path: tuple[str | int, ...],
    value: Any,
) -> None:
    from dataclasses import replace

    from radiosim.core.result import (
        InvalidResultError,
        MModeSolverResultProvenance,
        mmode_characterization_record,
    )

    result, original = genuine_phase_synthetic_result
    phase = copy.deepcopy(original)
    target: Any = phase
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    digest = _rehash_phase_schema_fixture(phase)
    owner = replace(result.solver.snapshot, input_identity_sha256=digest)
    changed = copy.copy(result)
    object.__setattr__(changed, "solver", MModeSolverResultProvenance(snapshot=owner))
    with pytest.raises(InvalidResultError):
        _ = mmode_characterization_record(
            changed,
            family_id="mmode_single_scalar_mode",
            phase_input_identity_manifest=phase,
        )


@pytest.mark.parametrize(
    "invalid",
    [
        "phase_key",
        "owner",
        "lmax_text",
        "lmax_bool",
        "lmax_negative",
        "snapshot_none",
        "snapshot_foreign",
    ],
)
def test_production_v2_rejects_before_derived_record_work(
    genuine_phase_synthetic_result: tuple[Any, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    """Malformed input must fail before harmonic-table or time-record work."""
    from dataclasses import replace

    from radiosim.core import result as production
    from radiosim.core.mmode import harmonics

    result, original = genuine_phase_synthetic_result
    phase = copy.deepcopy(original)
    changed = copy.copy(result)
    if invalid == "phase_key":
        phase["unknown"] = 0
    elif invalid in {"snapshot_none", "snapshot_foreign"}:
        snapshot = None if invalid == "snapshot_none" else object()
        object.__setattr__(
            changed, "solver", production.MModeSolverResultProvenance(snapshot=snapshot)
        )
    else:
        values: dict[str, Any] = {
            "owner": {"input_identity_sha256": _digest("foreign-owner")},
            "lmax_text": {"lmax": "bad"},
            "lmax_bool": {"lmax": True},
            "lmax_negative": {"lmax": -1},
        }
        snapshot = replace(result.solver.snapshot, **values[invalid])
        object.__setattr__(
            changed, "solver", production.MModeSolverResultProvenance(snapshot=snapshot)
        )
    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append("derived-record work")
        raise AssertionError("invalid input reached derived-record construction")

    monkeypatch.setattr(harmonics, "scalar_packed_block_table", forbidden)
    monkeypatch.setattr(production, "_characterization_time_manifest", forbidden)
    with pytest.raises(production.InvalidResultError):
        _ = production.mmode_characterization_record(
            changed,
            family_id="mmode_single_scalar_mode",
            phase_input_identity_manifest=phase,
        )
    assert calls == []


# D33 transport tests use tiny objects; a transport roundtrip is no science verdict.
def _storage_envelope(
    payload: bytes, compressed: bytes | None = None
) -> dict[str, Any]:
    import base64
    import zlib

    return {
        "schema": "radiosim.sci004.frame-certificate-storage.v1",
        "codec": "zlib+base64",
        "uncompressed_byte_count": len(payload),
        "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "data_base64": base64.b64encode(
            zlib.compress(payload) if compressed is None else compressed
        ).decode("ascii"),
    }


def test_certificate_storage_roundtrip_uses_section14_j_and_accepts_other_levels() -> (
    None
):
    import base64
    import zlib

    module = _tool()
    certificate = {"nested": {"finite": 0.125}, "unicode": "é", "rows": [1.0, True]}
    raw = b'{"nested":{"finite":0.125},"rows":[1,true],"unicode":"\\u00e9"}'
    envelope = module.encode_frame_certificate_storage(certificate)
    assert envelope == _storage_envelope(raw, zlib.compress(raw, level=9))
    assert zlib.decompress(base64.b64decode(envelope["data_base64"])) == raw
    for level in (0, 1, 9):
        assert (
            module.decode_frame_certificate_storage(
                _storage_envelope(raw, zlib.compress(raw, level=level)),
                label="transport",
            )
            == certificate
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema", "other"),
        ("codec", "gzip"),
        ("uncompressed_byte_count", True),
        ("uncompressed_byte_count", 0),
        ("uncompressed_byte_count", -1),
        ("uncompressed_byte_count", 33_554_433),
        ("uncompressed_byte_count", 2.0),
        ("uncompressed_sha256", "A" * 64),
        ("uncompressed_sha256", "0" * 63),
        ("data_base64", None),
    ],
)
def test_certificate_storage_rejects_envelope_fields(key: str, value: Any) -> None:
    module = _tool()
    envelope = _storage_envelope(b"{}")
    envelope[key] = value
    with pytest.raises(module.EvidenceError):
        module.decode_frame_certificate_storage(envelope, label="transport")


@pytest.mark.parametrize("mutation", ["missing", "extra", "not-object"])
def test_certificate_storage_requires_closed_envelope(mutation: str) -> None:
    module = _tool()
    envelope: Any = _storage_envelope(b"{}")
    if mutation == "missing":
        del envelope["codec"]
    elif mutation == "extra":
        envelope["extra"] = True
    else:
        envelope = []
    with pytest.raises(module.EvidenceError):
        module.decode_frame_certificate_storage(envelope, label="transport")


@pytest.mark.parametrize(
    "encoded", ["AAAA\nAAA", "____", "e30", "e30=====", "Zh==", "éAAA"]
)
def test_certificate_storage_rejects_noncanonical_base64(encoded: str) -> None:
    module = _tool()
    envelope = _storage_envelope(b"{}")
    envelope["data_base64"] = encoded
    with pytest.raises(module.EvidenceError):
        module.decode_frame_certificate_storage(envelope, label="transport")


@pytest.mark.parametrize("encoded", ["A" * 16, "AAAA" * 3])
def test_certificate_storage_bounds_before_base64_allocation(
    monkeypatch: pytest.MonkeyPatch, encoded: str
) -> None:
    module = _tool()
    envelope = _storage_envelope(b"{}")
    envelope["data_base64"] = encoded
    monkeypatch.setattr(module, "FRAME_CERTIFICATE_STORAGE_LIMIT", 8)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("base64 buffer allocated before size refusal")

    monkeypatch.setattr(module.base64, "b64decode", forbidden)
    with pytest.raises(module.EvidenceError, match="length"):
        module.decode_frame_certificate_storage(envelope, label="transport")


@pytest.mark.parametrize(
    "mutation",
    [
        "raw",
        "gzip",
        "dictionary",
        "truncated",
        "garbage",
        "concatenated",
        "bomb",
        "short",
        "hash",
    ],
)
def test_certificate_storage_rejects_stream_and_authentication(mutation: str) -> None:
    import zlib

    module = _tool()
    raw = b"{}"
    compressed = zlib.compress(raw)
    if mutation in {"raw", "gzip", "dictionary"}:
        if mutation == "dictionary":
            encoder = zlib.compressobj(zdict=b"{}")
        else:
            encoder = zlib.compressobj(wbits=-15 if mutation == "raw" else 31)
        compressed = encoder.compress(raw) + encoder.flush()
    elif mutation == "truncated":
        compressed = compressed[:-1]
    elif mutation == "garbage":
        compressed += b"trailing"
    elif mutation == "concatenated":
        compressed += zlib.compress(raw)
    elif mutation == "bomb":
        compressed = zlib.compress(b"x" * 100_000)
    envelope = _storage_envelope(raw, compressed)
    if mutation == "short":
        envelope["uncompressed_byte_count"] = 3
    elif mutation == "hash":
        envelope["uncompressed_sha256"] = "0" * 64
    with pytest.raises(module.EvidenceError):
        module.decode_frame_certificate_storage(envelope, label="transport")


@pytest.mark.parametrize(
    "payload",
    [
        b'{"x":1,"x":2}',
        b'{"x":1,"\\u0078":2}',
        b"[]",
        b"\xef\xbb\xbf{}",
        b"{}{}",
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":-Infinity}',
        b'{"x":1e9999}',
        b'{"x": 1}',
        b'{"x":1.0}',
        b"{}\n",
        b'{"x":"\xff"}',
    ],
)
def test_certificate_storage_rejects_noncanonical_finite_json(payload: bytes) -> None:
    module = _tool()
    with pytest.raises(module.EvidenceError):
        module.decode_frame_certificate_storage(
            _storage_envelope(payload), label="transport"
        )


def test_certificate_storage_decompression_is_bounded_without_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _tool()
    envelope = _storage_envelope(b"{}")
    original = module.zlib.decompressobj
    calls: list[int] = []

    class Bounded:
        def __init__(self, **kwargs: Any) -> None:
            self.stream = original(**kwargs)

        def decompress(self, data: bytes, max_length: int) -> bytes:
            calls.append(max_length)
            assert max_length == 3
            return self.stream.decompress(data, max_length)

        def __getattr__(self, name: str) -> Any:
            assert name != "flush", "unbounded finalization attempted"
            return getattr(self.stream, name)

    monkeypatch.setattr(module.zlib, "decompressobj", Bounded)
    assert module.decode_frame_certificate_storage(envelope, label="transport") == {}
    assert calls == [3]


@pytest.mark.parametrize(
    "certificate", [[], {"value": float("nan")}, {"value": float("inf")}]
)
def test_certificate_storage_encoder_requires_finite_object(certificate: Any) -> None:
    module = _tool()
    with pytest.raises(module.EvidenceError):
        module.encode_frame_certificate_storage(certificate)


def test_certificate_storage_encoder_checks_both_size_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _tool()
    monkeypatch.setattr(module, "FRAME_CERTIFICATE_STORAGE_LIMIT", 2)
    with pytest.raises(module.EvidenceError, match="uncompressed size"):
        module.encode_frame_certificate_storage({"long": "value"})
    with pytest.raises(module.EvidenceError, match="compressed size"):
        module.encode_frame_certificate_storage({})


@pytest.mark.parametrize(
    "value,raw",
    [
        ({"é": {"value": "𝄞"}}, b'{"\\u00e9":{"value":"\\ud834\\udd1e"}}'),
        ({"n": 2**53 - 1}, b'{"n":9007199254740991}'),
        ({"n": -(2**53 - 1)}, b'{"n":-9007199254740991}'),
        ({"n": 2**53}, b'{"n":9007199254740992}'),
        ({"n": -(2**53)}, b'{"n":-9007199254740992}'),
        ({"n": 10**21}, b'{"n":1e+21}'),
        ({"n": 1e-7}, b'{"n":1e-7}'),
    ],
)
def test_certificate_storage_exact_section14_unicode_and_numbers(
    value: Any, raw: bytes
) -> None:
    import base64
    import zlib

    module = _tool()
    encoded = module.encode_frame_certificate_storage(value)
    assert zlib.decompress(base64.b64decode(encoded["data_base64"])) == raw
    assert (
        module.decode_frame_certificate_storage(
            _storage_envelope(raw), label="transport"
        )
        == value
    )


@pytest.mark.parametrize(
    "raw",
    [
        '{"unicode":"é"}'.encode(),
        '{"é":1}'.encode(),
        b'{"n":1000000000000000000000}',
        b'{"n":9007199254740993}',
        b'{"n":-9007199254740993}',
        b'{"n":-0}',
        b'{"x":"\\ud800"}',
    ],
)
def test_certificate_storage_rejects_alternate_or_invalid_j(raw: bytes) -> None:
    module = _tool()
    with pytest.raises(module.EvidenceError):
        module.decode_frame_certificate_storage(
            _storage_envelope(raw), label="transport"
        )


@pytest.mark.parametrize(
    "value",
    [{"n": 2**53 + 1}, {"n": -(2**53 + 1)}, {"x": "\ud800"}, {1: "non-string key"}],
)
def test_certificate_storage_encoder_refuses_lossy_or_invalid_json(value: Any) -> None:
    module = _tool()
    with pytest.raises(module.EvidenceError):
        module.encode_frame_certificate_storage(value)


@pytest.mark.parametrize("size", [7, 8, 9])
def test_evidence_size_bound_precedes_both_final_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    module = _tool()
    monkeypatch.setattr(module, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(module, "EVIDENCE_BYTE_LIMIT", 8)
    performance = "performance.json"
    payload = b"x" * size
    if size >= 8:
        with pytest.raises(module.EvidenceError, match="complete evidence payload"):
            module._publish_evidence_payload(payload, performance, b"benchmark")
        assert list(tmp_path.iterdir()) == []
    else:
        module._publish_evidence_payload(payload, performance, b"benchmark")
        assert (tmp_path / performance).read_bytes() == b"benchmark"
        assert (tmp_path / module.EVIDENCE_ARTIFACT).read_bytes() == payload


def test_evidence_publication_keeps_performance_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _tool()
    monkeypatch.setattr(module, "REPOSITORY_ROOT", tmp_path)
    events: list[Path] = []
    original = module.write_atomic_no_overwrite

    def record(path: Path, payload: bytes) -> None:
        events.append(path)
        original(path, payload)

    monkeypatch.setattr(module, "write_atomic_no_overwrite", record)
    module._publish_evidence_payload(b"{}", "performance.json", b"{}")
    assert events == [
        tmp_path / "performance.json",
        tmp_path / module.EVIDENCE_ARTIFACT,
    ]


@pytest.mark.parametrize("size", [7, 8, 9])
def test_evidence_reader_enforces_strict_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    module = _tool()
    monkeypatch.setattr(module, "EVIDENCE_BYTE_LIMIT", 8)
    path = tmp_path / "evidence.json"
    _ = path.write_bytes(b"x" * size)
    if size >= 8:
        with pytest.raises(module.EvidenceError, match="evidence artifact"):
            module._read_evidence_payload(path)
    else:
        assert module._read_evidence_payload(path) == b"x" * size


def test_evidence_reader_caps_io_without_trusting_a_stat_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _tool()
    monkeypatch.setattr(module, "EVIDENCE_BYTE_LIMIT", 8)

    class GrowingReader:
        def __enter__(self) -> GrowingReader:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def read(self, size: int) -> bytes:
            assert size == 8, "unbounded or stat-dependent read"
            return io.BytesIO(b"x" * 100).read(size)

    class GrowingPath:
        def open(self, mode: str) -> GrowingReader:
            assert mode == "rb"
            return GrowingReader()

        def stat(self) -> Any:
            pytest.fail("a stale stat size cannot authorize the read")

    with pytest.raises(module.EvidenceError, match="evidence artifact"):
        module._read_evidence_payload(GrowingPath())


def test_evidence_check_rejects_size_before_json_or_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _tool()
    monkeypatch.setattr(module, "EVIDENCE_BYTE_LIMIT", 8)
    path = tmp_path / "oversized.json"
    _ = path.write_bytes(b"not-json" * 2)

    def forbidden(value: Any) -> None:
        pytest.fail("oversized artifact reached scientific validator")

    monkeypatch.setattr(module, "validate_evidence_artifact", forbidden)
    assert module.main(["check", "--artifact", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "evidence artifact must be smaller" in captured.err


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"é": "😀"}, b'{"\\u00e9":"\\ud83d\\ude00"}'),
        ({"n": 10**21}, b'{"n":1e+21}'),
        ({"n": 9007199254740992}, b'{"n":9007199254740992}'),
    ],
)
def test_general_j_uses_section14_unicode_and_number_bytes(
    value: Any, expected: bytes
) -> None:
    module = _tool()
    assert module.canonical_json(value) == expected
    assert (
        module.canonical_json(module._canonical_json_object(expected, "test"))
        == expected
    )
    assert module.object_digest("test.v1", value) == _domain_digest("test.v1", expected)


@pytest.mark.parametrize("value", [9007199254740993, {1: "key"}, "\ud800"])
def test_general_j_rejects_values_without_faithful_json_identity(value: Any) -> None:
    module = _tool()
    with pytest.raises(module.EvidenceError):
        module.canonical_json(value)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1,"a":1}',
        b'{"a": {"b":1,"b":1}}',
        b'{"a":NaN}',
        b'{"a":1e999}',
        b'{"a":1.0}',
        b'{"a":1} ',
        b'{"a":"\xc3\xa9"}',
        b'{"a":1000000000000000000000}',
        b"[]",
    ],
)
@pytest.mark.parametrize("target", ["artifact", "performance"])
def test_check_rejects_noncanonical_input_before_its_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes, target: str
) -> None:
    module = _tool()
    artifact = tmp_path / "evidence.json"
    performance = tmp_path / "performance.json"
    _ = artifact.write_bytes(payload if target == "artifact" else b"{}")
    _ = performance.write_bytes(payload)

    def forbidden(_value: Any) -> None:
        pytest.fail("noncanonical raw bytes reached the semantic validator")

    def valid_evidence(_value: Any) -> None:
        return None

    if target == "artifact":
        monkeypatch.setattr(module, "validate_evidence_artifact", forbidden)
    else:
        monkeypatch.setattr(module, "validate_evidence_artifact", valid_evidence)
        monkeypatch.setattr(module, "validate_performance_document", forbidden)
    arguments = ["check", "--artifact", str(artifact)]
    if target == "performance":
        arguments.extend(["--performance", str(performance)])
    assert module.main(arguments) == 1


@pytest.mark.parametrize("payload", [b"", b"\x00\xff\x01"])
def test_scientific_segment_authenticates_independent_raw_bytes(payload: bytes) -> None:
    module = _tool()
    segment = {
        "tag": "flags.data",
        "payload_hex": payload.hex(),
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert module.decode_scientific_segment(segment, "flags.data", "test") == payload


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("extra", 1),
        ("tag", "weights.data"),
        ("tag", 1),
        ("byte_count", True),
        ("byte_count", -1),
        ("byte_count", 2),
        ("payload_hex", "Ff"),
        ("payload_hex", " f"),
        ("payload_hex", "f"),
        ("payload_hex", "gg"),
        ("payload_hex", 255),
        ("sha256", "0" * 64),
        ("sha256", "A" * 64),
    ],
)
def test_scientific_segment_rejects_untrusted_envelope(key: str, value: Any) -> None:
    module = _tool()
    segment = {
        "tag": "flags.data",
        "payload_hex": "ff",
        "byte_count": 1,
        "sha256": hashlib.sha256(b"\xff").hexdigest(),
    }
    segment[key] = value
    with pytest.raises(module.EvidenceError):
        module.decode_scientific_segment(segment, "flags.data", "test")
    if key != "extra":
        del segment[key]
        with pytest.raises(module.EvidenceError):
            module.decode_scientific_segment(segment, "flags.data", "test")


@pytest.mark.parametrize(
    "payload",
    [
        b'"radiosim.result.v1"',
        b'["XX","XY","YX","YY"]',
        b'{"n":1.0,"z":-0.0}',
        b'{"n":9007199254740993}',
        '{"é":"😀"}'.encode(),
    ],
)
def test_scientific_json_preserves_result_encoding_not_section14_j(
    payload: bytes,
) -> None:
    module = _tool()
    value = module._scientific_json(payload, "test")
    assert (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        == payload
    )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1,"a":1}',
        b'{"a":1,"\\u0061":1}',
        b'{"a":{"x":1,"x":2}}',
        b'{"n":NaN}',
        b'{"n":Infinity}',
        b'{"n":1e999}',
        b'{"n":1e-999}',
        b'{"n":1.00}',
        b'{"z":-0}',
        b'{"b":1,"a":2}',
        b"{} ",
        b"{}{}",
        b'"\\u00e9"',
        b'"\\ud800"',
        b"\xef\xbb\xbf{}",
        b'"\xff"',
    ],
)
def test_scientific_json_rejects_nonfinite_duplicate_or_noncanonical_bytes(
    payload: bytes,
) -> None:
    module = _tool()
    with pytest.raises(module.EvidenceError):
        module._scientific_json(payload, "test")


@pytest.mark.parametrize(
    ("role", "dtype", "shape", "unit"),
    [
        ("visibilities", "<c16", [49, 3, 1, 4], struct.pack("<dd", 1.5, -2.0)),
        ("flags", "|b1", [49, 3, 1, 4], b"\x01"),
        ("weights", "<f8", [49, 3, 1, 4], struct.pack("<d", 1.0)),
        *[
            (role, "<f8", [49], struct.pack("<d", 0.5))
            for role in (
                "time.utc_jd1",
                "time.utc_jd2",
                "time.integration_time_seconds",
            )
        ],
        *[
            (role, "<f8", [1], struct.pack("<d", 0.5))
            for role in ("frequency_hz", "channel_width_hz")
        ],
    ],
)
def test_scientific_array_requires_exact_independent_layout(
    role: str,
    dtype: str,
    shape: list[int],
    unit: bytes,
) -> None:
    module = _tool()
    count = 1
    for extent in shape:
        count *= extent
    payload = unit * count
    metadata = {"dtype": dtype, "shape": shape}
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    module.validate_scientific_array(role, raw, payload)
    with pytest.raises(module.EvidenceError):
        module.validate_scientific_array(role, raw, payload + b"\0")
    for changed in (
        {**metadata, "dtype": ">f8"},
        {**metadata, "shape": [True]},
        {**metadata, "shape": [0]},
        {**metadata, "extra": 1},
    ):
        with pytest.raises(module.EvidenceError):
            encoded = json.dumps(
                changed, sort_keys=True, separators=(",", ":")
            ).encode()
            module.validate_scientific_array(role, encoded, payload)
    bad = b"\x02" if role == "flags" else struct.pack("<d", float("nan"))
    with pytest.raises(module.EvidenceError):
        module.validate_scientific_array(role, raw, payload[: -len(bad)] + bad)
    with pytest.raises(module.EvidenceError):
        module.validate_scientific_array("unknown", raw, payload)


@pytest.mark.parametrize(
    "source",
    [
        "",
        'SOURCE_DESIGN_SHA = "x"\nSOURCE_DESIGN_SHA = "y"\n',
        'SOURCE_DESIGN_SHA = "x"\nSOURCE_DESIGN_SHA = str(1)\n',
        "SOURCE_DESIGN_SHA = str(1)\n",
        'SOURCE_DESIGN_SHA: str = "x"\n',
        'if True:\n    SOURCE_DESIGN_SHA = "x"\n',
        'SOURCE_DESIGN_SHA = other = "x"\n',
        'SOURCE_DESIGN_SHA = "x"\ndef f():\n    SOURCE_DESIGN_SHA = "y"\n',
        "SOURCE_DESIGN_SHA =\n",
    ],
)
def test_source_design_disk_binding_requires_one_plain_literal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    module = _tool()
    path = tmp_path / module.DEPENDENCY_VALIDATOR_PATH
    path.parent.mkdir(parents=True)
    _ = path.write_text(source)
    monkeypatch.setattr(module, "REPOSITORY_ROOT", tmp_path)
    with pytest.raises(module.EvidenceError):
        _ = module._frozen_binding("SOURCE_DESIGN_SHA")


@pytest.mark.parametrize("field", ["__file__", "REPOSITORY_ROOT", "missing_file"])
def test_source_design_rejects_foreign_loaded_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    from tests.unit import test_sci004_phase3_dependency as dependency

    module = _tool()
    if field == "missing_file":
        monkeypatch.delattr(dependency, "__file__")
    else:
        monkeypatch.setattr(
            dependency,
            field,
            str(tmp_path / "absent.py") if field == "__file__" else tmp_path,
        )
    with pytest.raises(module.EvidenceError, match="another checkout"):
        _ = module._design_sha()


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("SOURCE_DESIGN_SHA", CURRENT_D31_SHA),
        ("SOURCE_DESIGN_SHA", HISTORICAL_SOURCE_D32_SHA),
        ("HISTORICAL_SOURCE_DESIGN_SHA", CURRENT_SOURCE_D33_SHA),
        ("D30_SHA", CURRENT_D31_SHA),
        ("APPROVED_SCI004_D_SHA", CURRENT_SOURCE_D33_SHA),
        ("joint_source_roles", HISTORICAL_SOURCE_D32_SHA),
    ],
)
def test_source_design_rejects_loaded_roles_different_from_disk(
    monkeypatch: pytest.MonkeyPatch, field: str, replacement: str
) -> None:
    from tests.unit import test_sci004_phase3_dependency as dependency

    module = _tool()
    if field == "joint_source_roles":
        from tools import sci004_phase3_history as history

        for peer in (dependency, history):
            monkeypatch.setattr(peer, "SOURCE_DESIGN_SHA", HISTORICAL_SOURCE_D32_SHA)
            monkeypatch.setattr(
                peer, "HISTORICAL_SOURCE_DESIGN_SHA", CURRENT_SOURCE_D33_SHA
            )
    else:
        monkeypatch.setattr(dependency, field, replacement)
    with pytest.raises(module.EvidenceError, match="loaded design differs"):
        _ = module._design_sha()


@pytest.mark.parametrize("result", [CURRENT_D31_SHA, HISTORICAL_SOURCE_D32_SHA, None])
def test_source_design_requires_exact_authenticator_return(
    monkeypatch: pytest.MonkeyPatch, result: str | None
) -> None:
    from tests.unit import test_sci004_phase3_dependency as dependency

    module = _tool()

    def authenticate() -> str:
        if result is None:
            raise dependency.DependencyCertificateError("hostile edge")
        return result

    monkeypatch.setattr(dependency, "authenticate_source_design_bindings", authenticate)
    with pytest.raises(module.EvidenceError, match="source (authentication|design)"):
        _ = module._design_sha()


def test_source_design_authentication_uses_hardened_dependency_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ast
    import inspect

    from tests.unit import test_sci004_phase3_dependency as dependency
    from tools import sci004_phase3_history as history

    module = _tool()

    def refuse_ambient(*_arguments: object) -> str:
        raise AssertionError("current design must not use the evidence Git reader")

    monkeypatch.setattr(module, "_git", refuse_ambient)
    assert module._design_sha() == CURRENT_SOURCE_D33_SHA
    assert (
        history.DESIGN_SHA,
        history.RED_DESIGN_SHA,
        history.HISTORICAL_SOURCE_DESIGN_SHA,
        history.SOURCE_DESIGN_SHA,
    ) == (
        RANGE_ORIGIN_D30_SHA,
        CURRENT_D31_SHA,
        HISTORICAL_SOURCE_D32_SHA,
        CURRENT_SOURCE_D33_SHA,
    )
    assert (
        len(
            {
                dependency.D30_SHA,
                dependency.APPROVED_SCI004_D_SHA,
                dependency.HISTORICAL_SOURCE_DESIGN_SHA,
                dependency.SOURCE_DESIGN_SHA,
            }
        )
        == 4
    )
    tree = ast.parse(inspect.getsource(module._design_sha))
    assert not any(
        isinstance(node, ast.Name) and node.id in {"_git", "subprocess"}
        for node in ast.walk(tree)
    )


@pytest.fixture
def raw_evidence_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str, str]:
    module = _tool()
    monkeypatch.setattr(sys.modules[__name__], "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(module, "REPOSITORY_ROOT", tmp_path)
    _ = _git("init", "-q")
    for key, value in {
        "user.name": "Synthetic evidence",
        "user.email": "evidence@example.invalid",
        "commit.gpgsign": "false",
        "core.autocrlf": "false",
        "core.hooksPath": os.devnull,
    }.items():
        _ = _git("config", key, value)
    parent = _e_topology_commit(
        {"source.py": b"VALUE = 1\n", "pixi.toml": b"manifest", "pixi.lock": b"lock"}
    )
    child = _e_topology_commit({"source.py": b"VALUE = 2\n", "raw.bin": b"\xff\0\r\n"})
    return tmp_path, parent, child


def _native_evidence_git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        check=True,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        },
    ).stdout


@pytest.mark.parametrize("overlay", ["commit", "blob", "graft", "graft-env"])
def test_evidence_git_authenticates_original_objects(
    raw_evidence_repository: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
    overlay: str,
) -> None:
    root, parent, child = raw_evidence_repository
    module = _tool()
    if overlay in {"commit", "blob"}:
        left, right = child, parent
        if overlay == "blob":
            left = _git("rev-parse", f"{child}:source.py").strip()
            right = _git("rev-parse", f"{parent}:source.py").strip()
        _ = _native_evidence_git(root, "replace", left, right)
        assert (
            _native_evidence_git(root, "show", f"{child}:source.py") == b"VALUE = 1\n"
        )
        _ = (root / "source.py").write_bytes(b"VALUE = 1\n")
        with pytest.raises(module.EvidenceError, match="differs from committed"):
            module._source_tree(child, "source.py")
    else:
        path = root / ".git/info/grafts"
        if overlay == "graft-env":
            path = root / ".git/alternate-grafts"
            monkeypatch.setenv("GIT_GRAFT_FILE", str(path))
        _ = path.write_text(f"{child}\n")
    assert module._git_blob(child, "source.py") == b"VALUE = 2\n"
    assert module._git_blob(child, "raw.bin") == b"\xff\0\r\n"
    assert module._commit_parents(child) == (parent,)
    assert module._is_ancestor(parent, child)
    assert not module._is_ancestor(child, parent)
    with pytest.raises(module.EvidenceError, match="ancestry"):
        module._is_ancestor("0" * 40, child)


@pytest.mark.parametrize(
    "variable",
    [
        "GIT_DIR",
        "GIT_COMMON_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CONFIG_COUNT",
    ],
)
def test_evidence_git_ignores_caller_routing(
    raw_evidence_repository: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    root, parent, child = raw_evidence_repository
    monkeypatch.setenv(variable, str(root / "missing-context"))
    module = _tool()
    assert module._git_blob(child, "source.py") == b"VALUE = 2\n"
    assert module._is_ancestor(parent, child)
    assert module._git("status", "--porcelain") == ""
    assert os.environ[variable] == str(root / "missing-context")


@pytest.mark.parametrize("registered", [False, True])
def test_evidence_preflight_binds_the_actual_worktree(
    raw_evidence_repository: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
    registered: bool,
) -> None:
    root, _parent, child = raw_evidence_repository
    module = _tool()
    actual = root
    if registered:
        actual = root.parent / (root.name + "-registered")
        _ = _git("worktree", "add", "--detach", str(actual), child)
    copy = root.parent / (root.name + "-clean")
    copy.mkdir()
    for name in ("source.py", "raw.bin", "pixi.toml", "pixi.lock"):
        _ = (copy / name).write_bytes((actual / name).read_bytes())
    try:
        if registered:
            _ = _git("config", "extensions.worktreeConfig", "true")
            _ = _native_evidence_git(
                actual, "config", "--worktree", "core.worktree", str(copy)
            )
        else:
            _ = _git("config", "core.worktree", str(copy))
        _ = (actual / "source.py").write_bytes(b"VALUE = 999\n")
        assert _native_evidence_git(actual, "status", "--porcelain") == b""
        monkeypatch.setattr(module, "REPOSITORY_ROOT", actual)
        with pytest.raises(module.EvidenceError, match="not globally clean"):
            module.preflight()
    finally:
        if registered:
            _ = _native_evidence_git(
                actual, "config", "--worktree", "--unset", "core.worktree"
            )
            _ = _git("config", "--unset", "extensions.worktreeConfig")
            _ = _git("worktree", "remove", "--force", str(actual))
        else:
            _ = _git("config", "--unset", "core.worktree")


def test_evidence_git_preserves_tree_bytes_and_visible_gitlinks(
    raw_evidence_repository: tuple[Path, str, str],
) -> None:
    root, _parent, child = raw_evidence_repository
    module = _tool()
    tree = _native_evidence_git(root, "ls-tree", "-r", "-z", "--full-tree", child)
    assert module.preflight()["git_tree_sha256"] == _domain_digest(
        "radiosim.sci004.git-tree.v1", tree
    )
    _ = _git("update-index", "--add", "--cacheinfo", "160000", child, "reference")
    _ = _git("commit", "-qm", "synthetic gitlink")
    tip = _git("rev-parse", "HEAD").strip()
    _ = _git("config", "diff.ignoreSubmodules", "all")
    assert _native_evidence_git(root, "diff", "--name-only", child, tip) == b""
    assert module._changed_paths(tip) == frozenset({"reference"})
    with pytest.raises(module.EvidenceError):
        module._git_bytes("ls-tree", "missing-tree")


def test_evidence_git_query_failures_keep_typed_preflight_refusal(
    raw_evidence_repository: tuple[Path, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _parent, _child = raw_evidence_repository
    module = _tool()

    def missing_head(*arguments: str) -> str:
        return "missing-tree" if arguments[0] == "rev-parse" else ""

    monkeypatch.setattr(module, "_git", missing_head)
    with pytest.raises(module.EvidenceError, match="ls-tree"):
        module.preflight()
    monkeypatch.setattr(module, "REPOSITORY_ROOT", root / "absent-directory")
    with pytest.raises(module.EvidenceError, match="cannot start"):
        module._git_bytes("rev-parse", "HEAD")


@pytest.mark.parametrize("concealment", ["assume", "skip", "stat"])
@pytest.mark.parametrize("after_publication", [False, True])
def test_evidence_raw_checkout_refuses_hidden_scientific_changes(
    raw_evidence_repository: tuple[Path, str, str],
    concealment: str,
    after_publication: bool,
) -> None:
    root, _, _ = raw_evidence_repository
    module = _tool()
    relative = "src/radiosim/core/mmode/frame.py"
    head = _e_topology_commit({relative: b"VALUE = 1\n"})
    path = root / relative
    _ = _native_evidence_git(root, "status", "--porcelain")
    os.utime(path, (1_600_000_000, 1_600_000_000))
    _ = _native_evidence_git(root, "update-index", "--refresh")
    before_stat = path.stat()
    if concealment == "stat":
        _ = _native_evidence_git(root, "config", "core.trustctime", "false")
        _ = _native_evidence_git(root, "config", "core.checkStat", "minimal")
    else:
        flag = "--assume-unchanged" if concealment == "assume" else "--skip-worktree"
        _ = _native_evidence_git(root, "update-index", flag, relative)
    before_index = (root / ".git/index").read_bytes()
    _ = path.write_bytes(b"VALUE = 2\n")
    if concealment == "stat":
        os.utime(path, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
    assert _native_evidence_git(root, "status", "--porcelain") == b""
    with pytest.raises(module.EvidenceError, match="tracked raw bytes changed"):
        if after_publication:
            _ = (root / "evidence.json").write_bytes(b"{}")
            module.require_declared_outputs_only(("evidence.json",), head)
        else:
            module.preflight()
    assert (root / ".git/index").read_bytes() == before_index


@pytest.mark.parametrize(
    "mutation", ["missing", "symlink", "directory", "mode", "parent"]
)
def test_evidence_raw_checkout_authenticates_types_and_parent_directories(
    raw_evidence_repository: tuple[Path, str, str], mutation: str
) -> None:
    root, _, _ = raw_evidence_repository
    module = _tool()
    relative = "tracked/leaf.py"
    head = _e_topology_commit({relative: b"VALUE = 1\n"})
    path = root / relative
    if mutation == "parent":
        _ = (root / "tracked").rename(root / "other")
        (root / "tracked").symlink_to("other", target_is_directory=True)
    elif mutation == "mode":
        path.chmod(0o755)
    else:
        path.unlink()
        if mutation == "symlink":
            _ = (root / "other.py").write_bytes(b"VALUE = 1\n")
            path.symlink_to("../other.py")
        elif mutation == "directory":
            path.mkdir()
    before_index = (root / ".git/index").read_bytes()
    with pytest.raises(module.EvidenceError) as error:
        module._require_raw_tracked_checkout(head)
    assert error.value.prefix == module.PREFLIGHT
    assert (root / ".git/index").read_bytes() == before_index


@pytest.mark.parametrize(
    "representation", ["pointer", "content", "wrong-size", "wrong-hash"]
)
def test_evidence_raw_checkout_authenticates_lfs_without_filters(
    raw_evidence_repository: tuple[Path, str, str], representation: str
) -> None:
    root, _, _ = raw_evidence_repository
    module = _tool()
    content = b"retained scientific data\x00\xff"
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{hashlib.sha256(content).hexdigest()}\nsize {len(content)}\n"
    ).encode()
    head = _e_topology_commit({"data.bin": pointer})
    _ = _native_evidence_git(root, "config", "filter.test.clean", "false")
    _ = (root / ".git/info/attributes").write_text("data.bin filter=test\n")
    payload = {
        "pointer": pointer,
        "content": content,
        "wrong-size": content + b"x",
        "wrong-hash": b"x" * len(content),
    }[representation]
    _ = (root / "data.bin").write_bytes(payload)
    before_index = (root / ".git/index").read_bytes()
    if representation in {"pointer", "content"}:
        module._require_raw_tracked_checkout(head)
    else:
        with pytest.raises(module.EvidenceError, match="tracked raw bytes changed"):
            module._require_raw_tracked_checkout(head)
    assert (root / ".git/index").read_bytes() == before_index


def test_evidence_raw_checkout_preserves_symlinks_flags_and_uninitialized_gitlinks(
    raw_evidence_repository: tuple[Path, str, str],
) -> None:
    root, _, child = raw_evidence_repository
    module = _tool()
    (root / "reference").mkdir()
    (root / "link").symlink_to("missing-target")
    _ = _native_evidence_git(root, "add", "link")
    _ = _native_evidence_git(
        root, "update-index", "--add", "--cacheinfo", f"160000,{child},reference"
    )
    _ = _native_evidence_git(root, "commit", "-qm", "Synthetic tracked types")
    _ = _native_evidence_git(root, "update-index", "--skip-worktree", "source.py")
    before_index = (root / ".git/index").read_bytes()
    module.preflight()
    assert (root / ".git/index").read_bytes() == before_index


def _synthetic_scientific_stream(circular: bool = False) -> tuple[list[Any], str]:
    """Build all 25 segments and a stdlib oracle without production helpers."""
    payloads: list[tuple[str, bytes]] = []

    def add_json(tag: str, value: Any) -> None:
        payloads.append(
            (
                tag,
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8"),
            )
        )

    add_json("schema", "radiosim.result.v1")
    for role, dtype, shape, data in (
        ("visibilities", "<c16", [49, 3, 1, 4], struct.pack("<dd", 1.25, -0.5) * 588),
        ("flags", "|b1", [49, 3, 1, 4], b"\0\1" * 294),
        ("weights", "<f8", [49, 3, 1, 4], struct.pack("<d", 2.0) * 588),
        ("time.utc_jd1", "<f8", [49], struct.pack("<d", 2451545.0) * 49),
        ("time.utc_jd2", "<f8", [49], struct.pack("<d", -0.0) * 49),
        ("time.integration_time_seconds", "<f8", [49], struct.pack("<d", 1.0) * 49),
        ("frequency_hz", "<f8", [1], struct.pack("<d", 1e8)),
        ("channel_width_hz", "<f8", [1], struct.pack("<d", 1e6)),
    ):
        add_json(role + ".metadata", {"dtype": dtype, "shape": shape})
        payloads.append((role + ".data", data))
    add_json(
        "correlations",
        ["RR", "RL", "LR", "LL"] if circular else ["XX", "XY", "YX", "YY"],
    )
    add_json("polarization_basis", "circular_rl" if circular else "linear_xy")
    for tag in (
        "receptor",
        "instrument",
        "selection",
        "beam",
        "phase_center",
        "solver",
    ):
        add_json(tag, {"synthetic": "λ", "value": -0.0, "integer": 9007199254740993})
    framed = bytearray()
    segments: list[Any] = []
    for tag, payload in payloads:
        encoded_tag = tag.encode("utf-8")
        framed.extend(struct.pack("<Q", len(encoded_tag)))
        framed.extend(encoded_tag)
        framed.extend(struct.pack("<Q", len(payload)))
        framed.extend(payload)
        segments.append(
            {
                "tag": tag,
                "payload_hex": payload.hex(),
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return segments, hashlib.sha256(framed).hexdigest()


@pytest.mark.parametrize("circular", [False, True])
def test_scientific_stream_matches_independent_framing(circular: bool) -> None:
    module = _tool()
    segments, expected = _synthetic_scientific_stream(circular)
    decoded = module.decode_scientific_stream(segments[:24], segments[24], label="test")
    assert decoded["scientific_sha256"] == expected
    assert len(decoded["payloads"]) == 25
    assert len(decoded["json_segments"]) == 17
    for segment in segments:
        assert decoded["payloads"][segment["tag"]] == bytes.fromhex(
            segment["payload_hex"]
        )
    assert decoded["json_segments"]["solver"]["integer"] == 9007199254740993
    assert decoded["payloads"]["time.utc_jd2.data"] == struct.pack("<d", -0.0) * 49


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "reorder", "duplicate", "solver-tag", "tuple", "jones"],
)
def test_scientific_stream_requires_exact_order_and_inventory(mutation: str) -> None:
    module = _tool()
    segments, _ = _synthetic_scientific_stream()
    common: Any = segments[:24]
    solver = segments[24]
    if mutation == "missing":
        common.pop()
    elif mutation == "extra":
        common.append(solver)
    elif mutation == "reorder":
        common[1], common[2] = common[2], common[1]
    elif mutation == "duplicate":
        common[2] = common[1]
    elif mutation == "solver-tag":
        solver["tag"] = "phase_center"
    elif mutation == "tuple":
        common = tuple(common)
    else:
        common[-1]["tag"] = "jones"
    with pytest.raises(module.EvidenceError):
        module.decode_scientific_stream(common, solver, label="test")


@pytest.mark.parametrize(
    ("tag", "value"),
    [
        ("schema", "radiosim.result.v2"),
        ("polarization_basis", "unknown"),
        ("polarization_basis", {}),
        ("correlations", ["YY", "YX", "XY", "XX"]),
        ("correlations", ["RR", "RL", "LR", "LL"]),
        ("correlations", ["XX", "XY", "YX"]),
        ("visibilities.metadata", {"dtype": "<c16", "shape": [49, 3, 1, 3]}),
        ("flags.metadata", {"dtype": "|b1", "shape": [49, 1, 3, 4]}),
        ("receptor", []),
        ("instrument", None),
        ("selection", True),
        ("beam", "beam"),
        ("phase_center", 0),
        ("solver", []),
    ],
)
def test_scientific_stream_rejects_authenticated_wrong_roots_and_joins(
    tag: str, value: Any
) -> None:
    module = _tool()
    segments, _ = _synthetic_scientific_stream()
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    index = next(
        index for index, segment in enumerate(segments) if segment["tag"] == tag
    )
    segments[index] = {
        "tag": tag,
        "payload_hex": payload.hex(),
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    with pytest.raises(module.EvidenceError):
        module.decode_scientific_stream(segments[:24], segments[24], label="test")


def _scientific_solver_fixture(family: str) -> dict[str, Any]:
    polarized = family in {"mmode_point_full_stokes", "mmode_circular_receptor"}
    return {
        "solver": "mmode",
        "sky_representation": "point_sources",
        "convention": "radiosim.mmode-forward.v1",
        "execution_path": "polarized" if polarized else "scalar",
        "components": ["point"],
        "component_element_counts": [1 if family == "mmode_single_scalar_mode" else 3],
        "time_grid_convention": "radiosim.mmode-era-turn-grid.v1",
        "frame_model": "radiosim.frozen-cirs-rigid-era.v1",
        "harmonic_convention": "radiosim.shaw-polarized-harmonics.v1",
        "sidereal_samples": 49,
        "lmax": 16,
        "mmax": 16,
        "quadrature_nside": 8,
        "quadrature_policy": "iso-gauss-ring-production-plus-qcheck.v1",
        "truncation_policy": "complete-frozen-direct-plus-local-shells.v1",
        "tangent_polarization_frame": {
            "schema_version": "radiosim.sky-tangent-polarization.v1",
            "coordinate_frame": "icrs",
            "axes": "north_east",
            "position_angle": "north_through_east",
            "linear_complex": "q_plus_i_u",
            "stokes_v": "iau_incoming_r_minus_l",
        }
        if polarized
        else "not_applicable_scalar_m1",
        "stokes_v_basis_bridge": "radiosim.stokes-ne-theta-phi.v1",
        "iers_table_sha256": "a" * 64,
        "frame_certificate_sha256": "b" * 64,
        "transform_execution_policy": "host_harmonics_backend_native_dense_v1",
    }


@pytest.mark.parametrize(
    "family",
    [
        "mmode_single_scalar_mode",
        "mmode_point_stokes_i",
        "mmode_point_full_stokes",
        "mmode_circular_receptor",
    ],
)
def test_scientific_solver_accepts_exact_family_and_source_hash_endpoint(
    family: str,
) -> None:
    module = _tool()
    row = _scientific_solver_fixture(family)
    assert (
        module.validate_scientific_solver(row, family, "a" * 64, label="solver") == row
    )
    row["frame_certificate_sha256"] = "c" * 64
    assert (
        module.validate_scientific_solver(row, family, "a" * 64, label="solver") == row
    )


@pytest.mark.parametrize(
    "key", tuple(_scientific_solver_fixture("mmode_point_full_stokes"))
)
def test_scientific_solver_refuses_missing_fields_and_wrong_values(key: str) -> None:
    module = _tool()
    row = _scientific_solver_fixture("mmode_point_full_stokes")
    del row[key]
    with pytest.raises(module.EvidenceError):
        module.validate_scientific_solver(
            row, "mmode_point_full_stokes", "a" * 64, label="solver"
        )
    row[key] = None
    with pytest.raises(module.EvidenceError):
        module.validate_scientific_solver(
            row, "mmode_point_full_stokes", "a" * 64, label="solver"
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "unknown-family",
        "wrong-family",
        "iers",
        "bad-iers",
        "certificate",
        "bool-count",
        "float-count",
        "tuple-count",
        "tuple-components",
        "tangent-extra",
        "tangent-axis",
        "tangent-null",
        "float-dimension",
    ],
)
def test_scientific_solver_refuses_closed_schema_and_typed_mutations(
    mutation: str,
) -> None:
    module = _tool()
    family = (
        "mmode_single_scalar_mode"
        if mutation == "bool-count"
        else "mmode_point_full_stokes"
    )
    row = _scientific_solver_fixture(family)
    iers = "a" * 64
    if mutation == "extra":
        row["backend"] = "numpy"
    elif mutation == "unknown-family":
        family = "other"
    elif mutation == "wrong-family":
        family = "mmode_point_stokes_i"
    elif mutation == "iers":
        iers = "d" * 64
    elif mutation == "bad-iers":
        iers = "G" * 64
    elif mutation == "certificate":
        row["frame_certificate_sha256"] = "B" * 64
    elif mutation == "bool-count":
        row["component_element_counts"] = [True]
    elif mutation == "float-count":
        row["component_element_counts"] = [3.0]
    elif mutation == "tuple-count":
        row["component_element_counts"] = (3,)
    elif mutation == "tuple-components":
        row["components"] = ("point",)
    elif mutation == "tangent-extra":
        row["tangent_polarization_frame"]["extra"] = 0
    elif mutation == "tangent-axis":
        row["tangent_polarization_frame"]["axes"] = "east_north"
    elif mutation == "tangent-null":
        row["tangent_polarization_frame"] = None
    elif mutation == "float-dimension":
        row["lmax"] = 16.0
    with pytest.raises(module.EvidenceError):
        module.validate_scientific_solver(row, family, iers, label="solver")


def _frame_manifest_test_digest(name: str, manifest: dict[str, Any]) -> str:
    """Independent identity oracle for these string-only synthetic manifests."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    if name == "horizon_scan_manifest":
        return hashlib.sha256(payload).hexdigest()
    prefix = manifest["schema_version"].encode("ascii") + b"\0"
    return hashlib.sha256(
        prefix + struct.pack(">Q", len(payload)) + payload
    ).hexdigest()


def _synthetic_frame_manifests() -> dict[str, Any]:
    """Build four manifest specimens, without claiming a full frame certificate."""

    def f64(value: float) -> str:
        return struct.pack(">d", value).hex()

    row: dict[str, Any] = {
        "input_identity_sha256": "a" * 64,
        "iers_table_sha256": "b" * 64,
    }
    row["site_manifest"] = {
        "schema_version": "radiosim.mmode-site.v1",
        "longitude_deg_f64be": f64(0.0),
        "latitude_deg_f64be": f64(0.0),
        "height_m_f64be": f64(-0.0),
        "itrs_xyz_m_f64be": [f64(6378137.0), f64(0.0), f64(0.0)],
    }
    identity = [f64(float(i == j)) for i in range(3) for j in range(3)]
    row["frame_matrix_manifest"] = {
        "schema_version": "radiosim.mmode-frame-matrices.v1",
        "era0_rad_f64be": f64(0.0),
        "rpom0_f64be": identity,
        "cirs_to_itrs_anchor_f64be": list(identity),
        "local_east_itrs_f64be": [f64(0.0), f64(1.0), f64(0.0)],
        "local_north_itrs_f64be": [f64(0.0), f64(0.0), f64(1.0)],
        "local_up_itrs_f64be": [f64(1.0), f64(0.0), f64(0.0)],
    }
    tables = {
        "direct_integrand_enclosure_manifest": [
            ("coherency_half_factor", "binary64", f64(0.5)),
            ("enclosure_accumulation_rounding", "literal", "toward_positive_infinity"),
            ("fringe_operator_norm_ceiling", "binary64", f64(1.0)),
            ("gauss_order_high", "integer", "128"),
            ("gauss_order_low", "integer", "64"),
            ("hadamard_factor_norm_ceiling", "binary64", f64(1.0)),
            (
                "magnitude_ceiling_rounding",
                "literal",
                "nextafter_toward_positive_infinity",
            ),
            ("rectangle_form", "literal", "[-G_abs,G_abs,-G_abs,G_abs]"),
            ("root_cell_nominal_contribution", "literal", "exact_complex_zero"),
        ],
        "horizon_scan_manifest": [
            ("L_op", "binary64", f64(6.2895)),
            ("h_0", "rational", "1/4096"),
            ("probe_magnitude_floor", "binary64", f64(1e-10)),
            ("probe_offset_turn", "rational", "1/100000000"),
            ("root_enclosure_width_rad", "binary64", f64(1e-11)),
            ("root_residual_bound", "binary64", f64(5e-12)),
            ("scan_algorithm", "literal", "radiosim.mmode-operational-horizon-scan.v1"),
            ("unresolved_width_turn", "rational", "1/17592186044416"),
        ],
    }
    for name, table in tables.items():
        scan = name == "horizon_scan_manifest"
        schema = (
            "radiosim.mmode-operational-horizon-scan.v1"
            if scan
            else "radiosim.mmode-direct-integrand-enclosure.v1"
        )
        row[name] = {
            "schema_version": schema,
            "algorithm_id": schema,
            "implementation_files": [
                {"path": "src/radiosim/core/mmode/" + path, "sha256": "c" * 64}
                for path in (
                    ("frame.py", "time.py")
                    if scan
                    else ("frame.py", "solver.py", "transfer.py")
                )
            ],
            "constant_rows": [
                {"name": key, "type": kind, "value": value}
                for key, kind, value in table
            ],
        }
        if scan:
            row[name].update(
                iers_table_sha256=row["iers_table_sha256"],
                astropy_version="synthetic.dev1",
                erfa_version="synthetic+test",
            )
        else:
            row[name].update(
                input_identity_sha256=row["input_identity_sha256"],
                frame_matrix_sha256=_frame_manifest_test_digest(
                    "frame_matrix_manifest", row["frame_matrix_manifest"]
                ),
            )
    for name in ("site_manifest", "frame_matrix_manifest", *tables):
        row[name.removesuffix("_manifest") + "_sha256"] = _frame_manifest_test_digest(
            name, row[name]
        )
    return row


def test_frame_manifests_match_independent_identities_without_mutating_inputs() -> None:
    row = _synthetic_frame_manifests()
    before = copy.deepcopy(row)
    manifests = _tool().validate_frame_manifest_structure(row, label="test")
    assert row == before
    assert [len(manifest) for manifest in manifests.values()] == [5, 7, 6, 7]
    assert manifests == {name: row[name] for name in manifests}
    assert manifests["site_manifest"]["height_m_f64be"] == "8000000000000000"


@pytest.mark.parametrize(
    "name",
    [
        "site_manifest",
        "frame_matrix_manifest",
        "direct_integrand_enclosure_manifest",
        "horizon_scan_manifest",
    ],
)
def test_frame_manifests_require_closed_roots_and_independent_digests(
    name: str,
) -> None:
    module = _tool()
    row = _synthetic_frame_manifests()
    mutations: list[Any] = [[], {**row[name], "extra": True}]
    mutations.extend(
        {key: value for key, value in row[name].items() if key != omitted}
        for omitted in row[name]
    )
    mutations.append({**row[name], "schema_version": "wrong"})
    for manifest in mutations:
        changed = copy.deepcopy(row)
        changed[name] = manifest
        with pytest.raises(module.EvidenceError) as error:
            module.validate_frame_manifest_structure(changed, label="test")
        assert error.value.prefix == module.SCHEMA
    field = name.removesuffix("_manifest") + "_sha256"
    row[field] = "f" * 64
    if name == "frame_matrix_manifest":
        enclosure = row["direct_integrand_enclosure_manifest"]
        enclosure[field] = row[field]
        row["direct_integrand_enclosure_sha256"] = _frame_manifest_test_digest(
            "direct_integrand_enclosure_manifest", enclosure
        )
    with pytest.raises(module.EvidenceError, match="manifest digest mismatch"):
        module.validate_frame_manifest_structure(row, label="test")


@pytest.mark.parametrize(
    "value",
    [True, 0.0, "xyz", "ABCDEF0123456789", "7ff0000000000000", "7ff8000000000000"],
)
def test_frame_manifests_reject_resigned_invalid_f64(value: Any) -> None:
    module = _tool()
    for name, field in (
        ("site_manifest", "height_m_f64be"),
        ("frame_matrix_manifest", "rpom0_f64be"),
    ):
        row = _synthetic_frame_manifests()
        if name == "site_manifest":
            row[name][field] = value
        else:
            row[name][field][0] = value
        row[name.removesuffix("_manifest") + "_sha256"] = _frame_manifest_test_digest(
            name, row[name]
        )
        with pytest.raises(module.EvidenceError) as error:
            module.validate_frame_manifest_structure(row, label="test")
        assert error.value.prefix == module.SCHEMA


@pytest.mark.parametrize("mutation", ["short", "long", "nested", "tuple"])
def test_frame_manifests_require_exact_vector_and_matrix_layouts(mutation: str) -> None:
    module = _tool()
    for name, field in (
        ("site_manifest", "itrs_xyz_m_f64be"),
        ("frame_matrix_manifest", "rpom0_f64be"),
    ):
        row = _synthetic_frame_manifests()
        values = row[name][field]
        row[name][field] = {
            "short": values[:-1],
            "long": values + values[:1],
            "nested": [values],
            "tuple": tuple(values),
        }[mutation]
        with pytest.raises(module.EvidenceError):
            module.validate_frame_manifest_structure(row, label="test")


@pytest.mark.parametrize(
    "name", ["direct_integrand_enclosure_manifest", "horizon_scan_manifest"]
)
@pytest.mark.parametrize("field", ["implementation_files", "constant_rows"])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "reverse",
        "extra",
        "wrong-root",
        "row-extra",
        "row-missing",
        "wrong-value",
        "wrong-type",
    ],
)
def test_frame_manifest_inventories_reject_resigned_mutations(
    name: str, field: str, mutation: str
) -> None:
    module = _tool()
    row = _synthetic_frame_manifests()
    entries = row[name][field]
    if mutation == "missing":
        entries.pop()
    elif mutation == "duplicate":
        entries[1] = entries[0]
    elif mutation == "reverse":
        entries.reverse()
    elif mutation == "extra":
        entries.append(entries[0])
    elif mutation == "wrong-root":
        row[name][field] = {}
    elif mutation == "row-extra":
        entries[0]["extra"] = "x"
    elif mutation == "row-missing":
        entries[0].pop("path" if field == "implementation_files" else "type")
    else:
        key = "sha256" if field == "implementation_files" else "value"
        entries[0][key] = True if mutation == "wrong-type" else "f" * 16
    row[name.removesuffix("_manifest") + "_sha256"] = _frame_manifest_test_digest(
        name, row[name]
    )
    with pytest.raises(module.EvidenceError) as error:
        module.validate_frame_manifest_structure(row, label="test")
    assert error.value.prefix == module.SCHEMA


@pytest.mark.parametrize(
    ("name", "field", "value"),
    [
        ("horizon_scan_manifest", "astropy_version", ""),
        ("horizon_scan_manifest", "erfa_version", True),
        ("horizon_scan_manifest", "algorithm_id", "wrong"),
        ("direct_integrand_enclosure_manifest", "algorithm_id", "wrong"),
        ("direct_integrand_enclosure_manifest", "input_identity_sha256", "f" * 64),
        ("direct_integrand_enclosure_manifest", "frame_matrix_sha256", "f" * 64),
        ("horizon_scan_manifest", "iers_table_sha256", "f" * 64),
    ],
)
def test_frame_manifests_enforce_versions_algorithms_and_resigned_local_joins(
    name: str, field: str, value: Any
) -> None:
    module = _tool()
    row = _synthetic_frame_manifests()
    row[name][field] = value
    row[name.removesuffix("_manifest") + "_sha256"] = _frame_manifest_test_digest(
        name, row[name]
    )
    with pytest.raises(module.EvidenceError):
        module.validate_frame_manifest_structure(row, label="test")


def test_frame_scan_manifest_uses_raw_hash_not_domain_digest() -> None:
    module = _tool()
    row = _synthetic_frame_manifests()
    scan = row["horizon_scan_manifest"]
    payload = json.dumps(scan, sort_keys=True, separators=(",", ":")).encode()
    row["horizon_scan_sha256"] = hashlib.sha256(
        scan["schema_version"].encode()
        + b"\0"
        + struct.pack(">Q", len(payload))
        + payload
    ).hexdigest()
    with pytest.raises(module.EvidenceError, match="manifest digest mismatch"):
        module.validate_frame_manifest_structure(row, label="test")


def _synthetic_frame_structure() -> dict[str, Any]:
    """Only a structural specimen; nested empty values cannot establish a certificate."""
    row: dict[str, Any] = {}
    for key in """
        certificate_sha256 site_sha256 input_identity_sha256 iers_table_sha256
        frame_matrix_sha256 canonical_era_turn_grid_sha256 canonical_era_grid_sha256
        transfer_grid_catalog_sha256 direction_ledger_sha256 horizon_scan_sha256
        horizon_scan_ledger_sha256 horizon_root_pair_ledger_sha256
        horizon_slab_ledger_sha256 horizon_sign_interval_ledger_sha256
        horizon_membership_ledger_sha256 direct_split_ledger_sha256
        direct_integrand_enclosure_sha256 frozen_gauss64_cube_sha256
        frozen_gauss128_cube_sha256 operational_gauss64_cube_sha256
        operational_gauss128_cube_sha256 frozen_enclosure_error_cube_sha256
        operational_enclosure_error_cube_sha256
    """.split():
        row[key] = "0" * 64
    for key in """
        sidereal_samples quadrature_nside n_baselines n_frequencies n_correlations
        expected_point_direction_count evaluated_point_direction_count
        expected_native_healpix_direction_count
        evaluated_native_healpix_direction_count
        expected_production_transfer_direction_count
        evaluated_production_transfer_direction_count
        expected_diagnostic_transfer_direction_count
        evaluated_diagnostic_transfer_direction_count
        expected_transfer_quadrature_direction_count
        evaluated_transfer_quadrature_direction_count expected_direction_count
        evaluated_direction_count expected_phase_comparison_count
        evaluated_phase_comparison_count expected_horizon_trajectory_count
        evaluated_horizon_trajectory_count expected_horizon_root_pair_row_count
        evaluated_horizon_root_pair_row_count expected_horizon_membership_count
        evaluated_horizon_membership_count expected_direct_exposure_split_count
        evaluated_direct_exposure_split_count expected_direct_split_row_count
        evaluated_direct_split_row_count expected_frozen_gauss64_node_count
        evaluated_frozen_gauss64_node_count expected_frozen_gauss128_node_count
        evaluated_frozen_gauss128_node_count expected_operational_gauss64_node_count
        evaluated_operational_gauss64_node_count
        expected_operational_gauss128_node_count
        evaluated_operational_gauss128_node_count horizon_isolation_interval_count
        horizon_unresolved_interval_count expected_horizon_slab_row_count
        evaluated_horizon_slab_row_count expected_horizon_sign_interval_count
        evaluated_horizon_sign_interval_count horizon_root_count_mismatches
        horizon_root_orientation_mismatches horizon_membership_mismatches
        horizon_outside_slab_sign_mismatches horizon_paired_root_count
        horizon_mismatch_slab_count expected_cube_cell_count
        evaluated_frozen_gauss64_cube_cell_count
        evaluated_frozen_gauss128_cube_cell_count
        evaluated_operational_gauss64_cube_cell_count
        evaluated_operational_gauss128_cube_cell_count
        compared_frozen_gauss_change_cell_count
        compared_operational_gauss_change_cell_count
        evaluated_frozen_enclosure_error_cell_count
        evaluated_operational_enclosure_error_cell_count
    """.split():
        row[key] = 0
    for key in """
        xp0_arcsec yp0_arcsec das2r_rad_per_arcsec xp0_rad yp0_rad sp0_rad
    """.split():
        row[key] = "0000000000000000"
    for key in """
        horizon_mismatch_measure_rad horizon_mismatch_measure_limit_rad
        horizon_root_max_rad horizon_root_limit_rad phase_max_rad phase_limit_rad
        direct_gauss_scale_jy frozen_gauss_change_max_jy
        operational_gauss_change_max_jy direct_gauss_change_max_jy
        direct_gauss_change_limit_jy cube_scale_jy cube_max_jy cube_limit_jy cube_l2
        cube_l2_limit direction_diagnostic_max_rad basis_diagnostic_max_rad
    """.split():
        row[key] = 0
    for key in """
        horizon_mismatch_measure_turn direction_diagnostic_argmax_phase
        basis_diagnostic_argmax_phase
    """.split():
        row[key] = "0/1"
    for key in """
        pm_source_unit pom00_argument_unit
    """.split():
        row[key] = ""
    for key in """
        direction_diagnostic_argmax_id basis_diagnostic_argmax_id
    """.split():
        row[key] = ""
    for key in """
        transfer_grid_catalog direction_rows horizon_scan_crossing_rows
        horizon_scan_summary_rows horizon_root_pair_rows horizon_slab_rows
        horizon_sign_interval_rows horizon_membership_mask_rows direct_split_rows
    """.split():
        row[key] = []
    for key in """
        site_manifest frame_matrix_manifest horizon_scan_manifest
        direct_integrand_enclosure_manifest
    """.split():
        row[key] = {}
    for key in """
        diagnostic_qcheck_nsides
    """.split():
        row[key] = [16]
    row.update(pm_source_unit="arcsec", pom00_argument_unit="rad")
    _seal_synthetic_frame_structure(row)
    return row


def _seal_synthetic_frame_structure(row: dict[str, Any]) -> None:
    preimage = {key: value for key, value in row.items() if key != "certificate_sha256"}
    payload = json.dumps(preimage, sort_keys=True, separators=(",", ":")).encode()
    domain = b"radiosim.mmode-frame-certificate.v1\0"
    row["certificate_sha256"] = hashlib.sha256(
        domain + struct.pack(">Q", len(payload)) + payload
    ).hexdigest()


def test_frame_structure_authenticates_exact_125_field_preimage() -> None:
    module = _tool()
    row = _synthetic_frame_structure()
    assert len(row) == 126
    preimage = module.validate_frame_certificate_structure(row, label="test")
    assert len(preimage) == 125 and "certificate_sha256" not in preimage
    assert preimage["site_manifest"] == {}  # Explicitly pending nested validation.
    row["direction_diagnostic_argmax_phase"] = "-1/98"
    _seal_synthetic_frame_structure(row)
    module.validate_frame_certificate_structure(row, label="test")
    row["cube_max_jy"] = 0.0  # Section 14 J normalizes this to the already hashed zero.
    module.validate_frame_certificate_structure(row, label="test")
    row["site_manifest"] = {"tampered": True}
    with pytest.raises(module.EvidenceError, match="digest mismatch"):
        module.validate_frame_certificate_structure(row, label="test")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("fixture_id", "fake"),
        ("pass", True),
        ("n_baselines", True),
        ("n_baselines", 1.0),
        ("n_baselines", "1"),
        ("n_baselines", -1),
        ("cube_max_jy", True),
        ("cube_max_jy", "0"),
        ("cube_max_jy", -1),
        ("cube_max_jy", float("inf")),
        ("cube_max_jy", float("nan")),
        ("cube_max_jy", 10**400),
        ("xp0_rad", "7ff0000000000000"),
        ("xp0_rad", "7ff8000000000000"),
        ("xp0_rad", 0),
        ("site_sha256", "A" * 64),
        ("pm_source_unit", "rad"),
        ("direction_diagnostic_argmax_id", None),
        ("horizon_mismatch_measure_turn", "-1/2"),
        ("horizon_mismatch_measure_turn", "0/2"),
        ("direction_diagnostic_argmax_phase", "2/4"),
        ("direction_diagnostic_argmax_phase", "01/2"),
        ("direction_diagnostic_argmax_phase", "1/-2"),
        ("direction_diagnostic_argmax_phase", "-0/1"),
        ("direction_diagnostic_argmax_phase", 0),
        ("direction_rows", {}),
        ("site_manifest", []),
        ("diagnostic_qcheck_nsides", [0]),
        ("diagnostic_qcheck_nsides", [True]),
        ("diagnostic_qcheck_nsides", [16.0]),
        ("diagnostic_qcheck_nsides", [16, 16]),
        ("diagnostic_qcheck_nsides", [32, 16]),
        ("diagnostic_qcheck_nsides", (16,)),
    ],
)
def test_frame_structure_refuses_untyped_scalars_and_roots(
    key: str, value: Any
) -> None:
    module = _tool()
    row = _synthetic_frame_structure()
    row[key] = value
    with pytest.raises(module.EvidenceError) as error:
        module.validate_frame_certificate_structure(row, label="test")
    assert error.value.prefix == module.SCHEMA
    row = _synthetic_frame_structure()
    del row["site_manifest"]
    with pytest.raises(module.EvidenceError):
        module.validate_frame_certificate_structure(row, label="test")


@pytest.fixture(scope="module")
def prepared_characterization_time(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Prepare the genuine Section11 input and UTC grid without a scientific solve."""
    from importlib import import_module

    import numpy as np

    from radiosim.api.simulator import Simulator
    from radiosim.core.mmode.time import CanonicalEraGrid
    from radiosim.core.mmode.types import derive_mmode_dimensions

    module = _tool()
    root = tmp_path_factory.mktemp("characterization-time").resolve()
    simulator = Simulator.from_mapping(
        module._family_mapping(root, "mmode_point_stokes_i"), base_dir=root
    )
    request = simulator.build_solve_request()
    grid, block = request.era_grid, request.mmode
    assert isinstance(grid, CanonicalEraGrid) and block is not None
    dimensions = derive_mmode_dimensions(
        lmax=int(block.lmax),
        mmax=int(block.mmax),
        quadrature_nside=int(block.quadrature_nside),
    )
    solver: Any = import_module("radiosim.core.mmode.solver")
    longitude, latitude, height = solver._site_geodetic(request.location)
    frame = solver.build_frozen_frame(
        start_time=grid.start_time_iso,
        longitude_deg=longitude,
        latitude_deg=latitude,
        height_m=height,
    )
    context = solver._kernel_context(request, frame, grid)
    point_cirs, point_stokes, point_icrs = solver._resolve_point_component(
        request, frame, context
    )
    ledger = solver.build_direction_ledger(
        frame=frame,
        dimensions=dimensions,
        point_cirs=point_cirs,
        point_stokes=point_stokes,
        point_icrs=point_icrs,
        native_cirs=np.zeros((0, 3)),
        native_stokes=np.zeros((0, context.n_frequencies, 4)),
        native_icrs=np.zeros((0, 2)),
        native_solid_angle=0.0,
    )
    phase, _digest_value = solver.build_input_identity(
        request=request,
        grid=grid,
        frame=frame,
        context=context,
        dimensions=dimensions,
        directions=ledger,
        tangent_frame=solver._resolved_tangent_frame(request, point_stokes),
    )
    manifest: dict[str, Any] = {
        "schema_version": "radiosim.sci004.characterization-time.v1",
        "axis_order": ["sample"],
        "shape": [len(grid)],
        "interval_semantics": "half_open_sample_centers",
        "start_time_iso": grid.start_time_iso,
        "center_jd1_f64be": [module.f64be(float(x)) for x in grid.utc_two_part[0]],
        "center_jd2_f64be": [module.f64be(float(x)) for x in grid.utc_two_part[1]],
        "integration_time_seconds_f64be": [
            module.f64be(float(x)) for x in grid.integration_time_seconds
        ],
    }
    assert len(phase) == 26 and manifest["shape"] == [49]
    assert _object_digest(manifest["schema_version"], manifest) == (
        "558758efff6d46ea559705bf6b6ab2245bf948a6d6792ed722e048e1ef41d877"
    )
    assert phase["canonical_era_grid_sha256"] == (
        "f865447ee34816c865e42d9202f26d388a6072c3f6be068973d9b9510ae357aa"
    )
    return module, manifest, phase


def test_characterization_time_prepared_grid_and_context(
    prepared_characterization_time: tuple[Any, dict[str, Any], dict[str, Any]],
) -> None:
    from importlib import import_module

    time_type: Any = import_module("astropy.time").Time
    iers: Any = import_module("astropy.utils.iers")
    module, manifest, phase = prepared_characterization_time
    cached_table = iers.IERS_A.iers_table
    previous = (
        iers.conf.auto_download,
        iers.conf.iers_degraded_accuracy,
        iers.earth_orientation_table.get(),
    )
    digest = _object_digest(manifest["schema_version"], manifest)
    assert (
        module.validate_characterization_time_manifest(manifest, digest, phase)
        == manifest
    )
    # A normalized millisecond ISO cannot recover the exact retained UTC JD pair.
    assert (
        module.f64be(float(time_type(manifest["start_time_iso"], scale="utc").jd2))
        != manifest["center_jd2_f64be"][0]
    )
    assert (
        iers.conf.auto_download,
        iers.conf.iers_degraded_accuracy,
        iers.earth_orientation_table.get(),
    ) == previous
    assert iers.IERS_A.iers_table is cached_table
    assert iers.IERS_A.open() is cached_table


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("float_shape", "time shape"),
        ("phase_float_shape", "phase time shape"),
        ("float_turn_count", "exact turn grid"),
        ("sample_bool", "exact JSON integer"),
        ("unreduced_turn", "exact turn grid"),
        ("unknown", "must have exactly"),
        ("missing", "must have exactly"),
        ("bool_shape", "time shape"),
        ("uppercase_hex", "lower-case hex"),
        ("nonfinite", "nonfinite binary64"),
        ("negative_width", "must be positive"),
        ("centers", "utc center jd2"),
        ("width", "exposure widths"),
        ("iso", "normalized UTC anchor"),
        ("ut1_center", "ut1 center jd2"),
        ("utc_edge", "utc lower jd2"),
        ("turn", "exact turn grid"),
        ("radian", "radian grid"),
        ("iers", "phase IERS"),
        ("outside_coverage", "(mapping|coverage)"),
    ],
)
def test_characterization_time_rehashed_semantic_mutations(
    prepared_characterization_time: tuple[Any, dict[str, Any], dict[str, Any]],
    mutation: str,
    expected: str,
) -> None:
    from importlib import import_module

    iers: Any = import_module("astropy.utils.iers")
    module, original, input_phase = prepared_characterization_time
    manifest, phase = copy.deepcopy((original, input_phase))
    if mutation == "float_shape":
        manifest["shape"] = [49.0]
    elif mutation == "phase_float_shape":
        phase["utc_manifest"]["shape"] = [49.0]
    elif mutation == "float_turn_count":
        phase["canonical_era_turn_grid"]["sidereal_samples"] = 49.0
    elif mutation == "sample_bool":
        phase["mmode_dimensions"]["sidereal_samples"] = True
    elif mutation == "unreduced_turn":
        phase["canonical_era_turn_grid"]["center_turns"][0] = "0/2"
    elif mutation == "unknown":
        manifest["extra"] = 1
    elif mutation == "missing":
        del manifest["axis_order"]
    elif mutation == "bool_shape":
        manifest["shape"] = [True]
    elif mutation == "uppercase_hex":
        manifest["center_jd1_f64be"][0] = "4142C60280000000"
    elif mutation == "nonfinite":
        manifest["center_jd2_f64be"][0] = "7ff0000000000000"
    elif mutation == "negative_width":
        manifest["integration_time_seconds_f64be"][0] = module.f64be(-1)
    elif mutation == "centers":
        manifest["center_jd2_f64be"][0] = module.f64be(0.125)
        phase["utc_manifest"]["center_jd2_f64be"][0] = module.f64be(0.125)
    elif mutation == "width":
        manifest["integration_time_seconds_f64be"][0] = module.f64be(1)
    elif mutation == "iso":
        manifest["start_time_iso"] = "2025-01-01T00:00:00"
    elif mutation == "ut1_center":
        phase["ut1_manifest"]["center_jd2_f64be"][1] = module.f64be(0.125)
    elif mutation == "utc_edge":
        phase["utc_manifest"]["lower_jd2_f64be"][0] = module.f64be(0.125)
    elif mutation == "turn":
        phase["canonical_era_turn_grid"]["center_turns"][1] = "1/48"
    elif mutation == "radian":
        phase["canonical_era_grid"]["delta_alpha_rad_f64be"] = module.f64be(0.125)
    elif mutation == "iers":
        phase["iers_table_sha256"] = "0" * 64
    elif mutation == "outside_coverage":
        for scale in ("utc", "ut1"):
            for position in ("center", "lower", "upper"):
                phase[f"{scale}_manifest"][f"{position}_jd1_f64be"][:] = [
                    module.f64be(3000000)
                ] * 49
        manifest["center_jd1_f64be"][:] = [module.f64be(3000000)] * 49
    _ = _rehash_phase_schema_fixture(phase)
    digest = _object_digest(manifest["schema_version"], manifest)
    cached_table = iers.IERS_A.iers_table
    previous = (
        iers.conf.auto_download,
        iers.conf.iers_degraded_accuracy,
        iers.earth_orientation_table.get(),
    )
    with pytest.raises(module.EvidenceError, match=expected):
        module.validate_characterization_time_manifest(manifest, digest, phase)
    assert (
        iers.conf.auto_download,
        iers.conf.iers_degraded_accuracy,
        iers.earth_orientation_table.get(),
    ) == previous
    assert iers.IERS_A.iers_table is cached_table
    assert iers.IERS_A.open() is cached_table


def test_characterization_time_authenticates_installed_table_bytes(
    prepared_characterization_time: tuple[Any, dict[str, Any], dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib import resources

    module, manifest, phase = prepared_characterization_time
    destination = tmp_path / "data/finals2000A.all"
    destination.parent.mkdir()
    _ = destination.write_bytes(b"substituted installed resource")

    def substituted_resource(_package: object) -> Path:
        return tmp_path

    monkeypatch.setattr(resources, "files", substituted_resource)
    with pytest.raises(module.EvidenceError, match="locked IERS bytes"):
        module.validate_characterization_time_manifest(
            manifest, _object_digest(manifest["schema_version"], manifest), phase
        )


@pytest.mark.parametrize(
    "failure", ["missing", "unreadable", "package", "parser", "read"]
)
def test_characterization_time_resource_failures_preserve_context(
    prepared_characterization_time: tuple[Any, dict[str, Any], dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from importlib import import_module, resources

    iers: Any = import_module("astropy.utils.iers")
    module, manifest, phase = prepared_characterization_time
    cached_table = iers.IERS_A.iers_table
    previous = (
        iers.conf.auto_download,
        iers.conf.iers_degraded_accuracy,
        iers.earth_orientation_table.get(),
    )
    cause: Exception = {
        "missing": FileNotFoundError("missing resource"),
        "unreadable": PermissionError("unreadable resource"),
        "package": ModuleNotFoundError("missing IERS package"),
        "parser": ValueError("invalid IERS table"),
        "read": OSError("failed IERS parser read"),
    }[failure]
    if failure == "missing":

        def missing_resource(_package: object) -> Path:
            return tmp_path

        monkeypatch.setattr(resources, "files", missing_resource)
    elif failure == "unreadable":

        def unreadable_resource(_path: Path) -> bytes:
            raise cause

        monkeypatch.setattr(Path, "read_bytes", unreadable_resource)
    elif failure == "package":

        def missing_package(_package: object) -> Path:
            raise cause

        monkeypatch.setattr(resources, "files", missing_package)
    else:

        def failed_parser(*_args: object, **_kwargs: object) -> object:
            raise cause

        monkeypatch.setattr(iers.IERS_A, "read", failed_parser)
    with pytest.raises(
        module.EvidenceError, match="cannot (load|parse) locked IERS resource"
    ) as caught:
        module.validate_characterization_time_manifest(
            manifest, _object_digest(manifest["schema_version"], manifest), phase
        )
    if failure == "missing":
        assert isinstance(caught.value.__cause__, FileNotFoundError)
    else:
        assert caught.value.__cause__ is cause
    assert (
        iers.conf.auto_download,
        iers.conf.iers_degraded_accuracy,
        iers.earth_orientation_table.get(),
    ) == previous
    assert iers.IERS_A.iers_table is cached_table
    assert iers.IERS_A.open() is cached_table


@pytest.mark.parametrize("after_publication", [False, True])
@pytest.mark.parametrize("move_head", [False, True])
def test_evidence_source_head_stays_bound_across_native_status_filter(
    raw_evidence_repository: tuple[Path, str, str],
    after_publication: bool,
    move_head: bool,
) -> None:
    root, _, original = raw_evidence_repository
    module = _tool()
    _ = _native_evidence_git(root, "commit", "--allow-empty", "-qm", "same tree")
    replacement = _git("rev-parse", "HEAD").strip()
    _ = _native_evidence_git(root, "update-ref", "HEAD", original)
    assert module.preflight(original)["source_sha"] == original
    path = root / "source.py"
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns - 20_000_000_000))
    _ = _native_evidence_git(root, "update-index", "--refresh", "--", "source.py")
    _ = (root / ".git/info/attributes").write_text("source.py filter=movehead\n")
    helper = root / ".git/movehead.py"
    _ = helper.write_text(
        "import subprocess,sys\n"
        + (
            f"subprocess.run(['git','update-ref','HEAD',{replacement!r}],check=True)\n"
            if move_head
            else ""
        )
        + "sys.stdout.buffer.write(b'VALUE = 2\\n')\n"
    )
    _ = _native_evidence_git(
        root,
        "config",
        "filter.movehead.clean",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(helper))}",
    )
    os.utime(path, None)
    entries = _native_evidence_git(root, "ls-files", "--stage", "-v", "-z")
    if after_publication:
        _ = (root / "evidence.json").write_bytes(b"{}")

    def check() -> None:
        if after_publication:
            module.require_declared_outputs_only(("evidence.json",), original)
        else:
            _ = module.preflight(original)

    if move_head:
        with pytest.raises(module.EvidenceError, match="source HEAD changed"):
            check()
    else:
        check()
    assert _git("rev-parse", "HEAD").strip() == (replacement if move_head else original)
    assert path.read_bytes() == b"VALUE = 2\n"
    assert _native_evidence_git(root, "ls-files", "--stage", "-v", "-z") == entries


def test_evidence_raw_checkout_rejects_stale_source_head_with_identical_tree(
    raw_evidence_repository: tuple[Path, str, str],
) -> None:
    root, _, original = raw_evidence_repository
    module = _tool()
    module._require_raw_tracked_checkout(original)
    _ = _native_evidence_git(root, "commit", "--allow-empty", "-qm", "same tree")
    with pytest.raises(module.EvidenceError, match="source HEAD changed"):
        module._require_raw_tracked_checkout(original)


def test_evidence_raw_checkout_rechecks_source_head_after_reading_blobs(
    raw_evidence_repository: tuple[Path, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, original = raw_evidence_repository
    module = _tool()
    _ = _native_evidence_git(root, "commit", "--allow-empty", "-qm", "same tree")
    replacement = _git("rev-parse", "HEAD").strip()
    _ = _native_evidence_git(root, "update-ref", "HEAD", original)
    read = module._git_bytes

    def move_after_blob(*arguments: str) -> bytes:
        payload: bytes = read(*arguments)
        if arguments[0] == "cat-file":
            _ = _native_evidence_git(root, "update-ref", "HEAD", replacement)
        return payload

    monkeypatch.setattr(module, "_git_bytes", move_after_blob)
    with pytest.raises(module.EvidenceError, match="source HEAD changed"):
        module._require_raw_tracked_checkout(original)
    assert _git("rev-parse", "HEAD").strip() == replacement
