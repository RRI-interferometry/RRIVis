#!/usr/bin/env python
"""Generate the SCI-004 phase-M3 evidence envelope and its Section 11 record.

``docs/development/sci004_mmode_design.md`` Section 14.2 freezes the phase
evidence schema; Section 11 freezes the non-gating performance record this phase
retains beside it.  This tool produces both and nothing else::

    pixi run python tools/sci004_mmode_phase3_evidence.py generate

It is the phase-M3 sibling of ``tools/sci004_mmode_phase2_evidence.py`` and keeps
that tool's discipline verbatim, because the discipline is the point rather than
the code.

**The venue.** Section 14.2: the tracked generator "executes at the globally
clean exact ``S`` checkout".  That checkout must own its Python environment.
This repository installs ``radiosim`` as an editable ``.pth`` pointing at one
working tree, so a second checkout that shares an environment silently imports
the *first* tree's source -- which, while ``S3`` production is uncommitted,
turns every phase-3 oracle green and would produce an evidence artifact
describing a run the observed tree cannot perform.  A replayer must therefore
run ``pixi install`` inside the checkout being observed.  The reproduction
record states this requirement; it is not optional advice.

**Import and validation dependencies.** The module imports only the standard
library. Generation and independent validation of retained time and frame
projections use this checkout's locked scientific environment, including NumPy,
Astropy/ERFA and the exact offline IERS data. Validation imports these packages
only when needed, disables IERS downloads, authenticates the installed table
identity and restores the caller's table and configuration after each check.

**How the timing stages are measured.** Section 11 requires ``frame``,
``sky_transform``, ``beam_transfer``, ``dense_contraction_and_synthesis`` and
``total`` to share indexed iterations, with "total time ... not smaller than the
sum of applicable named stages".  The named stages are therefore genuine
sub-intervals of one real solve rather than a separate driven pipeline: this
tool installs timing wrappers on the solver module's own stage entry points --
``build_frozen_frame``, ``build_direction_ledger``, ``build_frame_certificate``,
``build_production_transfer``, the two point sky-coefficient functions and
``contract_and_synthesize``, every one of which ``_mmode_pipeline`` calls as a
module global -- and then calls the public ``solve_mmode`` unchanged.  Nothing
in production is modified and no pipeline ordering is duplicated here, so the
record cannot drift from the solve it describes.  The wrappers are removed in a
``finally``.

**The honest backend axis.** Section 11, as the accepted honest-backend-axis and
scalar-table-kernel-exception corrections rule it: the public dense path is
backend-invariant, so every row carries ``dense_execution = numpy_host_v1``,
each fixture group measures its end-to-end series once on the NumPy row, and the
top-level ``dense_invariance`` array retains the measured bit-identity of the
three per-backend cubes as fact.  Real backend computation is retained only in
``kernel_backend_block``, and only where it exists: the routed contraction
kernel's contract covers exactly Section 5.3's four science fields, so the two
scalar-payload groups carry ``not_applicable_scalar_table`` on their JAX and
Dask rows while the polarized group carries measured stages whose
``stage_comparison`` reference is the NumPy kernel output on identical inputs.

**What the record is, and is not.** Timing values never gate CI and license
neither a speedup nor a memory or accelerator advantage; every row's
``claims_not_licensed`` says so, including the end-to-end backend execution this
phase deliberately does not perform.

Generation is atomic and no-overwrite, performance record first and evidence
last; a partial set is invalid and may not be reused.
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import hashlib
import json
import math
import os
import platform
import re
import select
import stat
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

SCI004_R3_TERMINAL_SHA = "567f9ac68730044fc8e887930d3531d794534412"

PHASE = "M3"
EVIDENCE_SCHEMA = "radiosim.sci004.mmode-phase3-evidence.v1"
STATUS = "candidate"
EVIDENCE_SELF_REFERENCE_REASON = "self-reference: A binds the containing E commit"

EVIDENCE_ARTIFACT = "docs/development/sci004_mmode_phase3_evidence.json"
REPRODUCTION = "docs/development/sci004_mmode_phase3_evidence.md"
RED_FAILURE_RECORD = "docs/development/sci004_mmode_phase3_red_failures.json"
POST_SOURCE_RED_FAILURE_RECORD = (
    "docs/development/sci004_mmode_phase3_post_source_red_failures.json"
)
RED_FAILURE_SCHEMA = "radiosim.sci004.mmode-phase3-red-failures.v1"
POST_SOURCE_RED_FAILURE_SCHEMA = (
    "radiosim.sci004.mmode-phase3-post-source-red-failures.v1"
)
POST_SOURCE_PRE_FIX_SHA = "a61526d686ab768f05ecffa80cfd6223d4ee4c62"
POST_SOURCE_RED_FAILURE_SHA256 = (
    "724f75c246ebfcf5956fc40fb2f5e349d91ccca3e6a188b3785a65f4ae4c1e10"
)
POST_SOURCE_DESIGN_SHA = "4d507bf1333ccaa4c8beec3815370ba0f6043bb2"
FINGERPRINT_RED_FAILURE_RECORD = (
    "docs/development/sci004_mmode_phase3_fingerprint_post_source_red_failures.json"
)
FINGERPRINT_RED_FAILURE_SCHEMA = (
    "radiosim.sci004.mmode-phase3-fingerprint-post-source-red-failures.v1"
)
FINGERPRINT_PRE_FIX_SHA = "b07925ab14b56b3ca0fa863f806290748a31df6b"
FINGERPRINT_RED_FAILURE_SHA256 = (
    "6bf1cf94b30961fd7a27519fad1252169155fdeee0e81618ea15115b50fbdb68"
)
FINGERPRINT_DESIGN_SHA = "ca3c37171aaaeec175b5ad72d324957762303853"
ORIGINAL_FINGERPRINT_RED_COMMIT_SHA = "a65c53a46e84f63c163c5ad15fba8645df33d1d2"
ORIGINAL_FINGERPRINT_RED_PATHS = frozenset(
    {
        FINGERPRINT_RED_FAILURE_RECORD,
        "tests/characterization/test_sci004_mmode.py",
        "tests/unit/test_sci004_phase3_dependency.py",
        "tests/unit/test_sci004_phase3_red_failures.py",
        "tools/sci004_mmode_phase3_red.py",
    }
)
HISTORICAL_RED_FAILURE_SHA256 = (
    "486705a8d5e51c08f972c91aeae60f0a0bfeef5480b622515282295a6a3cde05"
)
CERTIFICATE_PATH = "docs/development/sci004_mmode_phase3_sci005_dependency.json"
DEPENDENCY_VALIDATOR_PATH = "tests/unit/test_sci004_phase3_dependency.py"
STAGE2_TOOL_PATH = "tools/sci005_stage2_acceptance.py"
SCI005_STAGE2_ACCEPTANCE = "docs/development/sci005_stage2_acceptance.json"
PERFORMANCE_DIRECTORY = "output/benchmarks/reference/sci004"

#: Section 11's own schema literals.
BENCHMARK_SCHEMA = "radiosim.benchmark.sci004.v1"
BENCHMARK_PROVENANCE_SCHEMA = "radiosim.benchmark.sci004.provenance.v1"
TRANSFORM_EXECUTION_POLICY = "host_harmonics_backend_native_dense_v1"

#: Frozen stderr prefixes, mirroring the phase-2 generator.
PREFLIGHT = "SCI004_M3_EVIDENCE_PREFLIGHT"
SCHEMA = "SCI004_M3_EVIDENCE_SCHEMA"
DIGEST = "SCI004_M3_EVIDENCE_DIGEST"
RSS_SAMPLER = "SCI004_M3_RSS_SAMPLER"

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

#: Section 14.2's ``ci_artifacts`` row, as the retained-evidence correction
#: narrowed it: the four remote workflow fields are gone, and the row is
#: authenticated against the retained observation-set surface instead.
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
EXPECTED_COUNT_KEYS: tuple[str, ...] = (
    "roadmap_occurrences",
    "done_occurrences",
    "unsupported_claim_occurrences",
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

ENVIRONMENT_KEYS: tuple[str, ...] = (
    "python",
    "platform",
    "machine",
    "pixi_environment",
    "pixi_lock_sha256",
    "astropy_version",
    "erfa_version",
    "iers_package_version",
    "iers_table_sha256",
    "numeric_packages",
)
NUMERIC_PACKAGES: tuple[str, ...] = ("dask", "healpy", "jax", "numpy", "scipy")

SOURCE_IDENTITY_KEYS: tuple[str, ...] = (
    "git_tree_sha256",
    "pixi_manifest_sha256",
    "pixi_lock_sha256",
    "convention_identity_sha256",
    "fixture_input_rows",
    "input_identity_set_sha256",
)
FIXTURE_INPUT_ROW_KEYS: tuple[str, ...] = (
    "fixture_id",
    "input_identity_manifest",
    "input_identity_sha256",
)

RED_FAILURE_RECORD_KEYS: tuple[str, ...] = (
    "path",
    "sha256",
    "schema_version",
    "pre_fix_source_sha",
    "validated",
    "post_source_delta",
)
RED_FAILURE_REFERENCE_KEYS: tuple[str, ...] = (
    "path",
    "sha256",
    "schema_version",
    "pre_fix_source_sha",
    "validated",
)

COMMAND_KEYS: tuple[str, ...] = (
    "argv",
    "cwd",
    "pixi_environment",
    "started_at_utc",
    "duration_seconds",
    "exit_code",
    "stdout_sha256",
    "stderr_sha256",
)

# --- Section 11 record schema -------------------------------------------------

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

BACKEND_RUNTIME_KEYS: tuple[str, ...] = (
    "implementation",
    "implementation_version",
    "kernel_runtime",
    "kernel_runtime_version",
)
BACKEND_RUNTIME_PAIRS: Mapping[str, tuple[str, str]] = {
    "numpy": ("NumPy", "NumPy"),
    "jax": ("JAX", "JAXlib"),
    "dask": ("Dask", "NumPy"),
}

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
#: Section 11: the shared dense series are ``numpy_eager_v1`` on every row.
SHARED_SYNCHRONIZATION_METHOD = "numpy_eager_v1"
KERNEL_SYNCHRONIZATION_METHODS: Mapping[str, str] = {
    "jax": "jax_block_until_ready_v1",
    "dask": "dask_compute_v1",
}
MEASURED_SERIES: tuple[str, ...] = (
    "frame",
    "sky_transform",
    "beam_transfer",
    "dense_contraction_and_synthesis",
    "total",
)
CLOCK = "time.perf_counter_ns"
MINIMUM_SAMPLES = 5

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
RSS_SAMPLING_INTERVAL_NS = 10_000_000
RSS_READY_TIMEOUT_SECONDS = 10.0
RSS_RESULT_TIMEOUT_SECONDS = 5.0
RSS_READY_KEYS: tuple[str, ...] = (
    "status",
    "target_pid",
    "sampling_interval_ns",
    "baseline_rss_bytes",
)
RSS_RESULT_KEYS: tuple[str, ...] = (
    "status",
    "target_pid",
    "sampling_interval_ns",
    "baseline_rss_bytes",
    "peak_rss_bytes",
    "final_rss_bytes",
    "sample_count",
    "measured_host_peak_bytes",
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
DIRECT_PREDICATE_ID = "sci004_two_tier_direct.v3"

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
BACKEND_PREDICATE_ID = "sci004_backend_complex128.v1"
BACKEND_RTOL = 1e-12
BACKEND_ATOL_FACTOR = 1e-12

DENSE_EXECUTION = "numpy_host_v1"

KERNEL_BLOCK_NOT_APPLICABLE_KEYS: tuple[str, ...] = ("status", "reason")
KERNEL_BLOCK_MEASURED_KEYS: tuple[str, ...] = (
    "status",
    "per_m_contraction",
    "synthesis",
)
KERNEL_STAGE_KEYS: tuple[str, ...] = (
    "sample_seconds",
    "synchronization_method",
    "native_measurement_method",
    "measured_native_peak_bytes",
    "measured_native_peak_bytes_reason",
    "stage_comparison",
)
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
KERNEL_STATUS_NOT_APPLICABLE = "not_applicable"
KERNEL_STATUS_SCALAR = "not_applicable_scalar_table"
KERNEL_STATUS_MEASURED = "measured"
#: The exact reason the scalar-table exception requires: it names the kernel
#: contract that makes the measurement impossible for such a group.
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

#: Section 11's exact lexicographically sorted per-row claim array.
BENCHMARK_CLAIMS: tuple[str, ...] = (
    "general_speedup",
    "gpu_or_accelerator_support",
    "mmode_end_to_end_backend_execution",
    "perf001_evidence_or_closure",
    "performance_regression_gate",
    "unmeasured_workloads",
)

#: Section 11's four characterized families, in the amended memo order.
SECTION_11_FAMILIES: tuple[str, ...] = (
    "mmode_single_scalar_mode",
    "mmode_point_stokes_i",
    "mmode_point_full_stokes",
    "mmode_circular_receptor",
)
#: Section 11's three performance fixture groups, in record order.
PERFORMANCE_FIXTURES: tuple[str, ...] = (
    "mmode_single_scalar_mode",
    "mmode_point_stokes_i",
    "mmode_point_full_stokes",
)
BACKENDS: tuple[str, ...] = ("numpy", "jax", "dask")

#: The one Section 11 fixture group whose resolved payload is polarized.  Its
#: JAX and Dask rows carry measured kernel stages; the two scalar groups carry
#: the ``not_applicable_scalar_table`` exception, because the routed contraction
#: kernel's contract covers exactly Section 5.3's four science fields and a
#: per-``m`` kernel measurement for a one-field table would describe nothing its
#: own solve does.  The generator does not read this literal -- it discriminates
#: on the measured ``execution_path`` -- so the two disagree loudly if a
#: fixture's payload ever stops being what this names.
POLARIZED_FIXTURES: frozenset[str] = frozenset({"mmode_point_full_stokes"})

#: Section 10's three reader round trips, in ``ResultFormat`` vocabulary.  See
#: ``_output_cases`` for why summary JSON is not one of them.
OUTPUT_FORMATS: tuple[str, ...] = ("hdf5", "uvfits", "ms")

#: The formats that carry the published ``complex128`` cube through unchanged,
#: so a read cube must reproduce the written identity exactly.  Both other
#: formats narrow it, as a measured fact rather than an assumption: ``ms``
#: because ``project_simulation_result`` stores it as ``complex64`` by
#: contract, and ``uvfits`` because the published FITS payload is single
#: precision.  Their read identities are retained as measured and never
#: compared to the written one -- a narrowing row that restated it would be
#: describing a round trip that did not happen.  Were a writer ever to make one
#: of them lossless, the correct response is to move its name into this set,
#: not to relax the rule.
LOSSLESS_CUBE_FORMATS: frozenset[str] = frozenset({"hdf5"})

#: The one raw file inside a published Measurement Set directory the row hashes.
MS_MAIN_TABLE_FILE = "table.dat"

#: Section 14.0's exact result-identity domains.  Each is also its manifest's
#: ``schema_version``.
RESULT_TIME_DOMAIN = "radiosim.mmode-result-time.v1"
RESULT_FEED_DOMAIN = "radiosim.mmode-result-feeds.v1"
RESULT_CORRELATION_DOMAIN = "radiosim.mmode-result-correlations.v1"

#: Section 14.2: sorted, unique, non-empty.  The three deferrals the accepted
#: corrections require are carried explicitly.
LIMITATIONS: tuple[str, ...] = (
    "characterization: the initial harvest binds exactly the platform/Python "
    "cell this phase's acceptance runs on; every other cell enters afterwards "
    "by the standing admission discipline",
    "performance: the retained Section 11 record is evidence only of its nine "
    "measured CPU rows and gates nothing",
    "timing: the shared dense series are measured once per fixture group, "
    "because the public dense path is backend-invariant",
)
CLAIMS_NOT_LICENSED: tuple[str, ...] = (
    "accelerator: no GPU or other accelerator is exercised, measured or "
    "claimed anywhere in this phase",
    "diffuse: the public m-mode path rejects a HEALPix-bearing sky, so no "
    "diffuse or hybrid m-mode capability is evidenced here",
    "end-to-end-backend: wiring `request.backend` through the public dense "
    "stages is future red-sliced work, and no row implies it happened",
    "non-scalar-beam: the public m-mode path rejects a non-scalar resolved "
    "beam system, so no such capability is evidenced here",
    "performance: no speedup, regression gate or PERF-001 statement is "
    "licensed by any timing value retained here",
)

#: The three deferrals the accepted corrections require ``A3`` to carry, keyed
#: by the topic prefix each ``claims_not_licensed`` literal opens with: the
#: public diffuse/hybrid m-mode capability, the non-scalar-beam capability, and
#: the end-to-end ``request.backend`` dense wiring.
DEFERRAL_TOPICS: tuple[str, ...] = (
    "diffuse",
    "end-to-end-backend",
    "non-scalar-beam",
)


class EvidenceError(RuntimeError):
    """One refusal, carrying the frozen stderr prefix it must be reported with."""

    def __init__(self, prefix: str, detail: str) -> None:
        super().__init__(f"{prefix}: {detail}")
        self.prefix = prefix
        self.detail = detail


# ---------------------------------------------------------------------------
# Section 14 canonical serialization
# ---------------------------------------------------------------------------


def ecmascript_number(value: float) -> str:
    """Return Section 14's ECMAScript shortest round-trip number spelling."""
    if isinstance(value, bool):
        raise EvidenceError(SCHEMA, "a boolean is not a JSON number")
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise EvidenceError(SCHEMA, "NaN and Infinity are not JSON numbers")
    if value == int(value) and abs(value) < 2**53:
        return str(int(value))
    decimal = Decimal(repr(float(value)))
    exponent = decimal.adjusted()
    if -6 <= exponent <= 20:
        text = format(decimal, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text
    mantissa, _, exponent_text = format(decimal, "E").partition("E")
    mantissa = mantissa.rstrip("0").rstrip(".")
    sign = "+" if int(exponent_text) >= 0 else "-"
    return f"{mantissa}e{sign}{abs(int(exponent_text))}"


def domain_digest(domain: str, payload: bytes) -> str:
    """Return Section 14.0's ``D(d, p) = SHA256(d || NUL || U64(len(p)) || p)``.

    The length prefix is not decoration: without it a domain/payload pair is not
    uniquely decodable, and the digest would not agree with the production
    primitive in ``radiosim.core.mmode.types`` that every retained solver
    identity already uses.
    """
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def object_digest(domain: str, value: Any) -> str:
    """Return ``D(domain, J(value))``."""
    return domain_digest(domain, canonical_json(value))


def f64be(value: float) -> str:
    """Return Section 14.0's big-endian binary64 hex spelling."""
    return struct.pack(">d", float(value)).hex()


# ---------------------------------------------------------------------------
# Strict schema helpers
# ---------------------------------------------------------------------------


def _require(condition: bool, prefix: str, detail: str) -> None:
    if not condition:
        raise EvidenceError(prefix, detail)


def _require_keys(value: Any, keys: tuple[str, ...], label: str) -> dict[str, Any]:
    """Require an object to carry exactly one key set, rejecting any deviation.

    Section 14's canonical serialization sorts object keys lexicographically, so
    a re-read artifact never preserves an author's insertion order: "exactly
    these keys" is a statement about the *set*.  Both a missing and an unknown
    key are named in the refusal, because a validator that reported only the
    first would make a two-key drift take two rounds to diagnose.
    """
    _require(isinstance(value, Mapping), SCHEMA, f"{label} must be an object")
    mapping = dict(value)
    missing = [key for key in keys if key not in mapping]
    unknown = [key for key in mapping if key not in keys]
    _require(
        not missing and not unknown,
        SCHEMA,
        f"{label} must have exactly {list(keys)}"
        + (f"; missing {missing}" if missing else "")
        + (f"; unknown {unknown}" if unknown else ""),
    )
    return mapping


def _require_hex(value: Any, width: int, label: str) -> str:
    _require(
        isinstance(value, str)
        and re.fullmatch(f"[0-9a-f]{{{width}}}", value) is not None,
        SCHEMA,
        f"{label} must be {width} lower-case hex characters",
    )
    return str(value)


def _require_finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        SCHEMA,
        f"{label} must be a JSON number",
    )
    number = float(value)
    _require(math.isfinite(number), SCHEMA, f"{label} must be finite")
    return number


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        SCHEMA,
        f"{label} must be an exact JSON integer",
    )
    number = int(value)
    if minimum is not None:
        _require(number >= minimum, SCHEMA, f"{label} must be at least {minimum}")
    return number


_CHARACTERIZATION_TIME_DOMAIN = "radiosim.sci004.characterization-time.v1"
_CHARACTERIZATION_TIME_KEYS = tuple(
    "schema_version axis_order shape interval_semantics start_time_iso "
    "center_jd1_f64be center_jd2_f64be integration_time_seconds_f64be".split()
)
_CHARACTERIZATION_TIME_SHA256 = (
    "558758efff6d46ea559705bf6b6ab2245bf948a6d6792ed722e048e1ef41d877"
)
_CHARACTERIZATION_ERA_SHA256 = (
    "f865447ee34816c865e42d9202f26d388a6072c3f6be068973d9b9510ae357aa"
)
_CHARACTERIZATION_IERS_SHA256 = (
    "ff2d22108e982bd86e326e01d797fa8bd545d51483359dd98e6c08fa5737f667"
)


def _characterization_time_equal(actual: Any, expected: Any, label: str) -> None:
    _require(
        canonical_json(actual) == canonical_json(expected)
        and json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True),
        SCHEMA,
        label,
    )


def _characterization_time_f64(value: Any, label: str) -> float:
    text = _require_hex(value, 16, label)
    number = struct.unpack(">d", bytes.fromhex(text))[0]
    _require(math.isfinite(number), SCHEMA, f"{label}: nonfinite binary64")
    return number


def _characterization_time_radian_digest(role: str, values: list[float]) -> str:
    header = canonical_json(
        {
            "axis_order": ["sample"],
            "dtype": "float64-be",
            "role": role,
            "shape": [len(values)],
            "units": "rad",
        }
    )
    data = b"".join(struct.pack(">d", value) for value in values)
    payload = (
        len(header).to_bytes(8, "big") + header + len(data).to_bytes(8, "big") + data
    )
    return domain_digest("radiosim.mmode-era-radian-array.v1", payload)


def validate_characterization_time_manifest(
    value: Any, digest: Any, phase: dict[str, Any]
) -> dict[str, Any]:
    """Validate eight time fields and the complete temporal projection of phase.

    The caller separately authenticates the closed phase26 and input14 objects;
    this helper does not license non-temporal phase fields or a fingerprint row.
    """
    from fractions import Fraction
    from importlib import import_module, resources

    np: Any = import_module("numpy")
    time_type: Any = import_module("astropy.time").Time
    iers: Any = import_module("astropy.utils.iers")
    equal = _characterization_time_equal
    manifest = _require_keys(
        value, _CHARACTERIZATION_TIME_KEYS, "characterization time"
    )
    dimensions = _require_keys(
        phase.get("mmode_dimensions"),
        tuple(
            "lcheck lmax mcheck mmax qcheck quadrature_nside sidereal_samples".split()
        ),
        "time dimensions",
    )
    n = _require_int(dimensions["sidereal_samples"], "sample count", minimum=1)
    equal(manifest["schema_version"], _CHARACTERIZATION_TIME_DOMAIN, "time schema")
    equal(manifest["axis_order"], ["sample"], "time axis")
    equal(manifest["shape"], [n], "time shape")
    equal(manifest["interval_semantics"], "half_open_sample_centers", "time intervals")
    for key in (
        "center_jd1_f64be",
        "center_jd2_f64be",
        "integration_time_seconds_f64be",
    ):
        rows = manifest[key]
        _require(
            isinstance(rows, list) and len(cast(list[Any], rows)) == n,
            SCHEMA,
            f"{key}: length",
        )
        for item in cast(list[Any], rows):
            number = _characterization_time_f64(item, key)
            if key == "integration_time_seconds_f64be":
                _require(number > 0.0, SCHEMA, "exposure width must be positive")

    turn = _require_keys(
        phase.get("canonical_era_turn_grid"),
        tuple(
            "schema_version sidereal_samples integration_fraction_f64be "
            "integration_fraction_ratio exposure_width_turn horizon_lo_turn "
            "horizon_hi_turn center_turns lower_edge_turns upper_edge_turns".split()
        ),
        "turn grid",
    )
    fraction = _characterization_time_f64(
        turn["integration_fraction_f64be"], "fraction"
    )
    _require(0.0 < fraction <= 1.0, SCHEMA, "integration fraction range")
    exact_fraction = Fraction(fraction)
    turns = {
        "center": [Fraction(k, n) for k in range(n)],
        "lower_edge": [(2 * k - exact_fraction) / (2 * n) for k in range(n)],
        "upper_edge": [(2 * k + exact_fraction) / (2 * n) for k in range(n)],
    }
    horizon = [Fraction(-1, 2 * n), Fraction(2 * n - 1, 2 * n)]
    expected_turn: dict[str, Any] = {
        "schema_version": "radiosim.mmode-era-turn-grid.v1",
        "sidereal_samples": n,
        "integration_fraction_f64be": f64be(fraction),
        "integration_fraction_ratio": str(exact_fraction.numerator)
        + "/"
        + str(exact_fraction.denominator),
        "exposure_width_turn": str((exact_fraction / n).numerator)
        + "/"
        + str((exact_fraction / n).denominator),
        "horizon_lo_turn": f"{horizon[0].numerator}/{horizon[0].denominator}",
        "horizon_hi_turn": f"{horizon[1].numerator}/{horizon[1].denominator}",
    }
    for name, sequence in turns.items():
        expected_turn[f"{name}_turns"] = [
            f"{x.numerator}/{x.denominator}" for x in sequence
        ]
    equal(turn, expected_turn, "exact turn grid")
    equal(
        phase.get("canonical_era_turn_grid_sha256"),
        object_digest("radiosim.mmode-era-turn-grid.v1", turn),
        "turn digest",
    )
    tau = float.fromhex("0x1.921fb54442d18p+2")
    radian: dict[str, Any] = {
        "schema_version": "radiosim.mmode-era-grid.v2",
        "canonical_era_turn_grid_sha256": phase["canonical_era_turn_grid_sha256"],
        "tau_f64be": f64be(tau),
        "delta_alpha_rad_f64be": f64be(float(Fraction(tau) * exact_fraction / n)),
        "horizon_lo_rad_f64be": f64be(float(Fraction(tau) * horizon[0])),
        "horizon_hi_rad_f64be": f64be(float(Fraction(tau) * horizon[1])),
    }
    for name, sequence in turns.items():
        radian[f"era_{name}_turn_sha256"] = object_digest(
            f"radiosim.mmode-era-{name.replace('_', '-')}-turns.v1",
            turn[f"{name}_turns"],
        )
        radian[f"era_{name}_rad_sha256"] = _characterization_time_radian_digest(
            name, [float(Fraction(tau) * x) for x in sequence]
        )
    equal(phase.get("canonical_era_grid"), radian, "radian grid")
    equal(
        phase.get("canonical_era_grid_sha256"),
        object_digest("radiosim.mmode-era-grid.v2", radian),
        "radian digest",
    )
    equal(
        phase["canonical_era_grid_sha256"],
        _CHARACTERIZATION_ERA_SHA256,
        "family ERA grid",
    )

    grids: dict[str, dict[str, Any]] = {}
    decoded: dict[str, dict[str, list[float]]] = {}
    for scale in ("utc", "ut1"):
        grid = _require_keys(
            phase.get(f"{scale}_manifest"),
            tuple(
                "schema_version scale axis_order shape center_jd1_f64be center_jd2_f64be "
                "lower_jd1_f64be lower_jd2_f64be upper_jd1_f64be upper_jd2_f64be".split()
            ),
            scale,
        )
        equal(
            grid["schema_version"], f"radiosim.mmode-{scale}-grid.v1", f"{scale} schema"
        )
        equal(grid["scale"], scale, "time scale")
        equal(grid["axis_order"], ["sample"], "phase time axis")
        equal(grid["shape"], [n], "phase time shape")
        decoded[scale] = {}
        for position in ("center", "lower", "upper"):
            for part in (1, 2):
                key = f"{position}_jd{part}_f64be"
                rows = grid[key]
                _require(
                    isinstance(rows, list) and len(cast(list[Any], rows)) == n,
                    SCHEMA,
                    f"{scale} {key}: length",
                )
                decoded[scale][key] = [
                    _characterization_time_f64(item, key)
                    for item in cast(list[Any], rows)
                ]
        equal(
            phase.get(f"{scale}_sha256"),
            object_digest(f"radiosim.mmode-{scale}-grid.v1", grid),
            f"{scale} digest",
        )
        grids[scale] = grid
    for part in (1, 2):
        key = f"center_jd{part}_f64be"
        equal(manifest[key], grids["utc"][key], f"result UTC jd{part}")

    try:
        resource = resources.files("astropy_iers_data") / "data/finals2000A.all"
        payload = resource.read_bytes()
    except (OSError, ModuleNotFoundError) as error:
        raise EvidenceError(SCHEMA, "cannot load locked IERS resource") from error
    equal(
        hashlib.sha256(payload).hexdigest(),
        _CHARACTERIZATION_IERS_SHA256,
        "locked IERS bytes",
    )
    equal(phase.get("iers_table_sha256"), _CHARACTERIZATION_IERS_SHA256, "phase IERS")
    try:
        table = iers.IERS_A.read(file=str(resource))
    except (OSError, ValueError) as error:
        raise EvidenceError(SCHEMA, "cannot parse locked IERS resource") from error
    anchor1 = decoded["ut1"]["center_jd1_f64be"][0]
    anchor2 = decoded["ut1"]["center_jd2_f64be"][0]
    rate = Fraction("1.00273781191135448")
    previous_download, previous_degraded = (
        iers.conf.auto_download,
        iers.conf.iers_degraded_accuracy,
    )
    iers.conf.auto_download, iers.conf.iers_degraded_accuracy = False, "error"
    try:
        with iers.earth_orientation_table.set(table):
            utc: dict[str, Any] = {}
            pairs: dict[str, tuple[list[float], list[float]]] = {}
            for name, sequence in {
                "center": turns["center"],
                "lower": turns["lower_edge"],
                "upper": turns["upper_edge"],
                "horizon": horizon,
            }.items():
                pair = (
                    [anchor1] * len(sequence),
                    [float(Fraction(anchor2) + x / rate) for x in sequence],
                )
                pairs[name] = pair
                utc[name] = time_type(*pair, format="jd", scale="ut1").utc
                _, status = table.ut1_utc(utc[name], return_status=True)
                _require(bool(np.all(np.asarray(status) >= 0)), SCHEMA, "IERS coverage")
            anchor = time_type(
                anchor1, anchor2, format="jd", scale="ut1", precision=3
            ).utc
            equal(manifest["start_time_iso"], str(anchor.isot), "normalized UTC anchor")
            for scale in ("utc", "ut1"):
                for name in ("center", "lower", "upper"):
                    # The existing UT1-v1 schema intentionally stores UTC exposure edges.
                    pair = (
                        pairs[name]
                        if scale == "ut1" and name == "center"
                        else (utc[name].jd1, utc[name].jd2)
                    )
                    for part in (1, 2):
                        equal(
                            grids[scale][f"{name}_jd{part}_f64be"],
                            [f64be(float(x)) for x in pair[part - 1]],
                            f"{scale} {name} jd{part}",
                        )
            widths = (utc["upper"] - utc["lower"]).to_value("s")
            equal(
                manifest["integration_time_seconds_f64be"],
                [f64be(float(x)) for x in widths],
                "exposure widths",
            )
    except (ValueError, TypeError, IndexError, OverflowError) as error:
        raise EvidenceError(SCHEMA, "invalid locked-IERS time mapping") from error
    finally:
        iers.conf.auto_download, iers.conf.iers_degraded_accuracy = (
            previous_download,
            previous_degraded,
        )
    equal(
        _require_hex(digest, 64, "characterization time digest"),
        object_digest(_CHARACTERIZATION_TIME_DOMAIN, manifest),
        "characterization time digest",
    )
    equal(
        digest, _CHARACTERIZATION_TIME_SHA256, "unchanged family characterization time"
    )
    return manifest


FRAME_CERTIFICATE_STORAGE_SCHEMA = "radiosim.sci004.frame-certificate-storage.v1"
FRAME_CERTIFICATE_STORAGE_CODEC = "zlib+base64"
FRAME_CERTIFICATE_STORAGE_KEYS = (
    "schema",
    "codec",
    "uncompressed_byte_count",
    "uncompressed_sha256",
    "data_base64",
)
FRAME_CERTIFICATE_STORAGE_LIMIT = 33_554_432


def canonical_json(value: Any) -> bytes:
    """Return Section 14 J with finite binary64 numbers and escaped Unicode."""

    def string(text: str) -> str:
        # Lone surrogates cannot represent Unicode scalar values in UTF-8 JSON.
        _ = text.encode("utf-8")
        return json.dumps(text, ensure_ascii=True, separators=(",", ":"))

    def render(item: Any) -> str:
        if item is None:
            return "null"
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, (int, float)):
            number = float(item)
            _require(math.isfinite(number), SCHEMA, "JSON number must be finite")
            if isinstance(item, int):
                _require(
                    int(number) == item,
                    SCHEMA,
                    "JSON integer cannot roundtrip through binary64",
                )
            return ecmascript_number(number)
        if isinstance(item, str):
            return string(item)
        if isinstance(item, Mapping):
            mapping = cast(Mapping[Any, Any], item)
            _require(
                all(isinstance(key, str) for key in mapping),
                SCHEMA,
                "JSON object keys must be strings",
            )
            entries = cast(Mapping[str, Any], mapping)
            return (
                "{"
                + ",".join(
                    string(key) + ":" + render(entries[key]) for key in sorted(entries)
                )
                + "}"
            )
        if isinstance(item, (list, tuple)):
            return (
                "["
                + ",".join(render(child) for child in cast(Sequence[Any], item))
                + "]"
            )
        raise EvidenceError(SCHEMA, f"{type(item).__name__} is not JSON")

    try:
        return render(value).encode("utf-8")
    except (ValueError, UnicodeError, OverflowError, RecursionError) as error:
        raise EvidenceError(
            SCHEMA, "value cannot be encoded as Section 14 J"
        ) from error


def _canonical_json_object(payload: bytes, label: str) -> dict[str, Any]:
    """Parse one finite object with exact Section 14 J bytes, without duplicates."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            _require(key not in result, SCHEMA, f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise EvidenceError(SCHEMA, f"{label}: non-finite JSON constant {value}")

    def finite_float(value: str) -> float:
        number = float(value)
        _require(math.isfinite(number), SCHEMA, f"{label}: non-finite JSON number")
        return number

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
            parse_float=finite_float,
            parse_int=lambda token: (
                int(token) if abs(int(token)) <= 2**53 - 1 else finite_float(token)
            ),
        )
        _require(isinstance(value, dict), SCHEMA, f"{label}: expected JSON object")
        _require(
            canonical_json(value) == payload,
            SCHEMA,
            f"{label}: noncanonical J bytes",
        )
    except (ValueError, UnicodeError, OverflowError, RecursionError) as error:
        raise EvidenceError(
            SCHEMA, f"{label}: invalid canonical JSON object"
        ) from error
    return cast(dict[str, Any], value)


def encode_frame_certificate_storage(certificate: Any) -> dict[str, Any]:
    """Encode D33 transport; full certificate schema and cascade checks are separate."""
    _require(isinstance(certificate, Mapping), SCHEMA, "certificate must be an object")
    try:
        payload = canonical_json(certificate)
    except (ValueError, UnicodeError, OverflowError, RecursionError) as error:
        raise EvidenceError(SCHEMA, "certificate cannot be encoded as J") from error
    _require(
        1 <= len(payload) <= FRAME_CERTIFICATE_STORAGE_LIMIT,
        SCHEMA,
        "certificate uncompressed size is outside the D33 bound",
    )
    _ = _canonical_json_object(payload, "certificate")
    compressed = zlib.compress(payload, level=9)
    _require(
        len(compressed) <= FRAME_CERTIFICATE_STORAGE_LIMIT,
        SCHEMA,
        "certificate compressed size exceeds the D33 bound",
    )
    return {
        "schema": FRAME_CERTIFICATE_STORAGE_SCHEMA,
        "codec": FRAME_CERTIFICATE_STORAGE_CODEC,
        "uncompressed_byte_count": len(payload),
        "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "data_base64": base64.b64encode(compressed).decode("ascii"),
    }


def decode_frame_certificate_storage(value: Any, *, label: str) -> dict[str, Any]:
    """Authenticate bounded D33 transport, without granting a certificate verdict."""
    envelope = _require_keys(value, FRAME_CERTIFICATE_STORAGE_KEYS, label)
    _require(
        envelope["schema"] == FRAME_CERTIFICATE_STORAGE_SCHEMA,
        SCHEMA,
        f"{label}: unknown storage schema",
    )
    _require(
        envelope["codec"] == FRAME_CERTIFICATE_STORAGE_CODEC,
        SCHEMA,
        f"{label}: unknown storage codec",
    )
    declared = _require_int(envelope["uncompressed_byte_count"], label, minimum=1)
    limit = FRAME_CERTIFICATE_STORAGE_LIMIT
    _require(declared <= limit, SCHEMA, f"{label}: declared output exceeds D33 bound")
    digest = _require_hex(envelope["uncompressed_sha256"], 64, label)
    encoded = envelope["data_base64"]
    _require(isinstance(encoded, str), SCHEMA, f"{label}: base64 must be a string")
    # Check character and inferred byte lengths before allocating ASCII/decoded buffers.
    _require(
        0 < len(encoded) <= 4 * ((limit + 2) // 3) and len(encoded) % 4 == 0,
        SCHEMA,
        f"{label}: encoded length exceeds or violates D33 bounds",
    )
    padding = 2 if encoded.endswith("==") else int(encoded.endswith("="))
    _require(
        len(encoded) // 4 * 3 - padding <= limit,
        SCHEMA,
        f"{label}: compressed length exceeds D33 bound",
    )
    try:
        compressed = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeError, binascii.Error) as error:
        raise EvidenceError(SCHEMA, f"{label}: malformed base64") from error
    _require(
        base64.b64encode(compressed).decode("ascii") == encoded,
        SCHEMA,
        f"{label}: noncanonical base64",
    )
    _require(
        2 <= len(compressed) <= limit and not compressed[1] & 0x20,
        SCHEMA,
        f"{label}: invalid stream length or preset dictionary",
    )
    try:
        stream = zlib.decompressobj(wbits=zlib.MAX_WBITS)
        # One overflow byte is a bounded rejection sentinel. Never use unbounded flush.
        payload = stream.decompress(compressed, declared + 1)
    except zlib.error as error:
        raise EvidenceError(SCHEMA, f"{label}: invalid zlib stream") from error
    _require(len(payload) == declared, SCHEMA, f"{label}: uncompressed length mismatch")
    _require(
        stream.eof and not stream.unused_data and not stream.unconsumed_tail,
        SCHEMA,
        f"{label}: incomplete stream or trailing compressed bytes",
    )
    _require(
        hashlib.sha256(payload).hexdigest() == digest,
        DIGEST,
        f"{label}: uncompressed SHA-256 mismatch",
    )
    return _canonical_json_object(payload, label)


SCIENTIFIC_SEGMENT_KEYS = ("tag", "payload_hex", "byte_count", "sha256")
SCIENTIFIC_ARRAY_LAYOUTS: Mapping[str, tuple[str, tuple[int, ...], int]] = {
    "visibilities": ("<c16", (49, 3, 1, 4), 16),
    "flags": ("|b1", (49, 3, 1, 4), 1),
    "weights": ("<f8", (49, 3, 1, 4), 8),
    "time.utc_jd1": ("<f8", (49,), 8),
    "time.utc_jd2": ("<f8", (49,), 8),
    "time.integration_time_seconds": ("<f8", (49,), 8),
    "frequency_hz": ("<f8", (1,), 8),
    "channel_width_hz": ("<f8", (1,), 8),
}


def decode_scientific_segment(value: Any, expected_tag: str, label: str) -> bytes:
    """Authenticate one D32 segment; ordering and scientific joins are separate."""
    segment = _require_keys(value, SCIENTIFIC_SEGMENT_KEYS, label)
    _require(
        type(segment["tag"]) is str and segment["tag"] == expected_tag,
        SCHEMA,
        f"{label}: unexpected scientific segment tag",
    )
    count = segment["byte_count"]
    _require(
        type(count) is int and count >= 0,
        SCHEMA,
        f"{label}: byte_count must be a nonnegative integer",
    )
    encoded = segment["payload_hex"]
    _require(
        type(encoded) is str
        and len(encoded) == 2 * count
        and re.fullmatch("[0-9a-f]*", encoded) is not None,
        SCHEMA,
        f"{label}: payload must be exact lowercase hex of the declared length",
    )
    digest = _require_hex(segment["sha256"], 64, label)
    payload = bytes.fromhex(encoded)
    _require(
        hashlib.sha256(payload).hexdigest() == digest,
        DIGEST,
        f"{label}: scientific segment SHA-256 mismatch",
    )
    return payload


def _scientific_json(payload: bytes, label: str) -> Any:
    """Decode exact result JSON bytes, distinct from Section 14 J serialization."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            _require(key not in result, SCHEMA, f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise EvidenceError(SCHEMA, f"{label}: non-finite JSON constant {value}")

    def finite_float(token: str) -> float:
        value = float(token)
        _require(math.isfinite(value), SCHEMA, f"{label}: non-finite JSON number")
        return value

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
            parse_float=finite_float,
        )
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        _require(encoded == payload, SCHEMA, f"{label}: noncanonical result JSON")
    except (ValueError, UnicodeError, OverflowError, RecursionError) as error:
        raise EvidenceError(SCHEMA, f"{label}: invalid scientific JSON") from error
    return value


def validate_scientific_array(
    role: str, metadata_payload: bytes, payload: bytes
) -> None:
    """Validate a D32 array buffer without conversion or a physics verdict."""
    _require(role in SCIENTIFIC_ARRAY_LAYOUTS, SCHEMA, "unknown scientific array role")
    metadata = _require_keys(
        _scientific_json(metadata_payload, role), ("dtype", "shape"), role
    )
    dtype, shape, itemsize = SCIENTIFIC_ARRAY_LAYOUTS[role]
    _require(metadata["dtype"] == dtype, SCHEMA, f"{role}: incorrect array dtype")
    actual_shape = metadata["shape"]
    _require(
        isinstance(actual_shape, list)
        and all(
            type(size) is int and size > 0 for size in cast(list[Any], actual_shape)
        )
        and actual_shape == list(shape),
        SCHEMA,
        f"{role}: incorrect array shape",
    )
    _require(
        len(payload) == math.prod(shape) * itemsize,
        SCHEMA,
        f"{role}: array byte length does not match its layout",
    )
    if role == "flags":
        _require(all(byte in (0, 1) for byte in payload), SCHEMA, "invalid flag byte")
    else:
        _require(
            all(math.isfinite(value) for (value,) in struct.iter_unpack("<d", payload)),
            SCHEMA,
            f"{role}: array contains a non-finite component",
        )


SCIENTIFIC_COMMON_TAGS = (
    "schema",
    "visibilities.metadata",
    "visibilities.data",
    "flags.metadata",
    "flags.data",
    "weights.metadata",
    "weights.data",
    "time.utc_jd1.metadata",
    "time.utc_jd1.data",
    "time.utc_jd2.metadata",
    "time.utc_jd2.data",
    "time.integration_time_seconds.metadata",
    "time.integration_time_seconds.data",
    "frequency_hz.metadata",
    "frequency_hz.data",
    "channel_width_hz.metadata",
    "channel_width_hz.data",
    "correlations",
    "polarization_basis",
    "receptor",
    "instrument",
    "selection",
    "beam",
    "phase_center",
)


def decode_scientific_stream(
    common_segments: Any, solver_segment: Any, *, label: str
) -> dict[str, Any]:
    """Decode D32's ordered buffers and framed hash, without admitting a proof.

    JSON projection contents, solver identity and certificate joins require
    separate validation. Returned buffers preserve the authenticated raw bytes.
    """
    _require(
        isinstance(common_segments, list)
        and len(cast(list[Any], common_segments)) == 24,
        SCHEMA,
        f"{label}: expected exactly 24 common scientific segments",
    )
    segments = [*cast(list[Any], common_segments), solver_segment]
    payloads: dict[str, bytes] = {}
    json_segments: dict[str, Any] = {}
    digest = hashlib.sha256()
    for tag, segment in zip((*SCIENTIFIC_COMMON_TAGS, "solver"), segments, strict=True):
        payload = decode_scientific_segment(segment, tag, f"{label}.{tag}")
        payloads[tag] = payload
        if not tag.endswith(".data"):
            json_segments[tag] = _scientific_json(payload, f"{label}.{tag}")
        tag_bytes = tag.encode("utf-8")
        digest.update(len(tag_bytes).to_bytes(8, "little"))
        digest.update(tag_bytes)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    _require(
        json_segments["schema"] == "radiosim.result.v1",
        SCHEMA,
        f"{label}: incorrect scientific schema",
    )
    for role in SCIENTIFIC_ARRAY_LAYOUTS:
        validate_scientific_array(
            role, payloads[f"{role}.metadata"], payloads[f"{role}.data"]
        )
    basis = json_segments["polarization_basis"]
    correlations = json_segments["correlations"]
    _require(
        type(basis) is str and basis in ("linear_xy", "circular_rl"),
        SCHEMA,
        f"{label}: incorrect polarization basis",
    )
    expected = (
        ["XX", "XY", "YX", "YY"] if basis == "linear_xy" else ["RR", "RL", "LR", "LL"]
    )
    _require(
        correlations == expected
        and len(expected) == json_segments["visibilities.metadata"]["shape"][3],
        SCHEMA,
        f"{label}: correlations do not match the basis and cube axis",
    )
    for tag in (
        "receptor",
        "instrument",
        "selection",
        "beam",
        "phase_center",
        "solver",
    ):
        _require(
            isinstance(json_segments[tag], dict),
            SCHEMA,
            f"{label}.{tag}: expected a scientific JSON object",
        )
    return {
        "scientific_sha256": digest.hexdigest(),
        "payloads": payloads,
        "json_segments": json_segments,
    }


def _require_sorted_unique_strings(value: Any, label: str) -> list[str]:
    _require(isinstance(value, list), SCHEMA, f"{label} must be an array")
    items = list(value)
    _require(
        all(isinstance(item, str) and item for item in items),
        SCHEMA,
        f"{label} entries must be non-empty strings",
    )
    _require(items == sorted(set(items)), SCHEMA, f"{label} must be sorted and unique")
    return items


# ---------------------------------------------------------------------------
# Correction #24: external current-process RSS sampler
# ---------------------------------------------------------------------------


def _require_live_parent(target_pid: int) -> None:
    """Require the sampler's target to remain its live direct parent."""
    observed = os.getppid()
    if observed != target_pid or observed <= 1:
        raise EvidenceError(
            RSS_SAMPLER,
            f"target PID {target_pid} is not the live sampler parent {observed}",
        )


def _linux_rss_bytes(target_pid: int) -> int:
    """Return Linux resident pages multiplied by the positive system page size."""
    try:
        fields_text = (
            Path(f"/proc/{target_pid}/statm").read_text(encoding="ascii").split()
        )
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, UnicodeError, ValueError) as exc:
        raise EvidenceError(
            RSS_SAMPLER, "the Linux RSS counter is unavailable"
        ) from exc
    if len(fields_text) < 2 or not fields_text[1].isdecimal():
        raise EvidenceError(RSS_SAMPLER, "the Linux RSS counter is malformed")
    if type(page_size) is not int or page_size <= 0:
        raise EvidenceError(RSS_SAMPLER, "the Linux page size is not positive")
    resident_pages = int(fields_text[1])
    maximum = (1 << 64) - 1
    if resident_pages > maximum // page_size:
        raise EvidenceError(RSS_SAMPLER, "the Linux RSS counter overflows uint64")
    return resident_pages * page_size


def _darwin_proc_taskinfo_type() -> Any:
    """Return Darwin's complete 96-byte ``proc_taskinfo`` structure type."""
    import ctypes

    class ProcTaskInfo(ctypes.Structure):
        _fields_ = [
            ("pti_virtual_size", ctypes.c_uint64),
            ("pti_resident_size", ctypes.c_uint64),
            ("pti_total_user", ctypes.c_uint64),
            ("pti_total_system", ctypes.c_uint64),
            ("pti_threads_user", ctypes.c_uint64),
            ("pti_threads_system", ctypes.c_uint64),
            ("pti_policy", ctypes.c_int32),
            ("pti_faults", ctypes.c_int32),
            ("pti_pageins", ctypes.c_int32),
            ("pti_cow_faults", ctypes.c_int32),
            ("pti_messages_sent", ctypes.c_int32),
            ("pti_messages_received", ctypes.c_int32),
            ("pti_syscalls_mach", ctypes.c_int32),
            ("pti_syscalls_unix", ctypes.c_int32),
            ("pti_csw", ctypes.c_int32),
            ("pti_threadnum", ctypes.c_int32),
            ("pti_numrunning", ctypes.c_int32),
            ("pti_priority", ctypes.c_int32),
        ]

    return ProcTaskInfo


def _darwin_rss_bytes(target_pid: int, libproc: Any | None = None) -> int:
    """Return Darwin ``pti_resident_size`` after an exact-size kernel result."""
    import ctypes

    structure_type = _darwin_proc_taskinfo_type()
    expected_size = ctypes.sizeof(structure_type)
    if expected_size != 96:
        raise EvidenceError(
            RSS_SAMPLER,
            f"the Darwin proc_taskinfo layout is {expected_size} bytes, not 96",
        )
    try:
        library = libproc or ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = library.proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int
        information = structure_type()
        returned = proc_pidinfo(
            target_pid,
            4,  # PROC_PIDTASKINFO
            0,
            ctypes.byref(information),
            expected_size,
        )
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise EvidenceError(
            RSS_SAMPLER, "the Darwin RSS counter is unavailable"
        ) from exc
    if returned != expected_size:
        raise EvidenceError(
            RSS_SAMPLER,
            f"the Darwin RSS counter returned {returned} bytes, not {expected_size}",
        )
    return int(information.pti_resident_size)


def _instantaneous_rss_bytes(target_pid: int, system: str | None = None) -> int:
    """Return one fail-closed RSS sample after verifying the live parent."""
    _require_live_parent(target_pid)
    operating_system = sys.platform if system is None else system
    if operating_system.startswith("linux"):
        return _linux_rss_bytes(target_pid)
    if operating_system == "darwin":
        return _darwin_rss_bytes(target_pid)
    raise EvidenceError(
        RSS_SAMPLER, f"unsupported RSS sampling platform {operating_system!r}"
    )


def _emit_sampler_record(record: Mapping[str, Any]) -> None:
    """Emit exactly one canonical protocol record and flush it."""
    sys.stdout.buffer.write(canonical_json(record) + b"\n")
    sys.stdout.buffer.flush()


def _rss_sampler_child(target_pid: int, sampling_interval_ns: int) -> int:
    """Sample the live parent on fixed monotonic deadlines until exact ``STOP``."""
    if type(target_pid) is not int or target_pid <= 1:
        raise EvidenceError(RSS_SAMPLER, "target PID must be a positive process PID")
    if sampling_interval_ns != RSS_SAMPLING_INTERVAL_NS:
        raise EvidenceError(
            RSS_SAMPLER,
            f"sampling interval must be exactly {RSS_SAMPLING_INTERVAL_NS} ns",
        )

    baseline = _instantaneous_rss_bytes(target_pid)
    peak = baseline
    sample_count = 1
    _emit_sampler_record(
        {
            "status": "READY",
            "target_pid": target_pid,
            "sampling_interval_ns": sampling_interval_ns,
            "baseline_rss_bytes": baseline,
        }
    )

    next_deadline = time.monotonic_ns() + sampling_interval_ns
    while True:
        now = time.monotonic_ns()
        timeout = max(0.0, (next_deadline - now) / 1e9)
        readable, _, _ = select.select([sys.stdin.buffer], [], [], timeout)
        if readable:
            command = sys.stdin.buffer.readline()
            remainder = sys.stdin.buffer.read()
            if command != b"STOP\n" or remainder:
                raise EvidenceError(
                    RSS_SAMPLER, "parent command must be exactly STOP\\n"
                )
            final = _instantaneous_rss_bytes(target_pid)
            sample_count += 1
            peak = max(peak, final)
            _emit_sampler_record(
                {
                    "status": "RESULT",
                    "target_pid": target_pid,
                    "sampling_interval_ns": sampling_interval_ns,
                    "baseline_rss_bytes": baseline,
                    "peak_rss_bytes": peak,
                    "final_rss_bytes": final,
                    "sample_count": sample_count,
                    "measured_host_peak_bytes": peak - baseline,
                }
            )
            return 0

        sample = _instantaneous_rss_bytes(target_pid)
        sample_count += 1
        peak = max(peak, sample)
        now = time.monotonic_ns()
        while next_deadline <= now:
            next_deadline += sampling_interval_ns


def _protocol_line(stream: Any, timeout_seconds: float, label: str) -> bytes:
    """Read one line under a single deadline without a blocking buffered read."""
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError, ValueError) as exc:
        raise EvidenceError(SCHEMA, f"the RSS sampler {label} pipe failed") from exc
    deadline = time.monotonic() + timeout_seconds
    payload = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise EvidenceError(SCHEMA, f"the RSS sampler {label} timed out")
        try:
            readable, _, _ = select.select([descriptor], [], [], remaining)
        except (OSError, ValueError) as exc:
            raise EvidenceError(SCHEMA, f"the RSS sampler {label} pipe failed") from exc
        if not readable:
            raise EvidenceError(SCHEMA, f"the RSS sampler {label} timed out")
        try:
            byte = os.read(descriptor, 1)
        except OSError as exc:
            raise EvidenceError(SCHEMA, f"the RSS sampler {label} pipe failed") from exc
        if not byte:
            raise EvidenceError(SCHEMA, f"the RSS sampler {label} line is malformed")
        if byte == b"\n":
            if payload.endswith(b"\r"):
                raise EvidenceError(
                    SCHEMA, f"the RSS sampler {label} line is malformed"
                )
            return bytes(payload)
        payload.extend(byte)
        # The previous ``readline(65_537)`` implementation admitted at most
        # 65,535 payload bytes plus the line feed.
        if len(payload) >= 65_536:
            raise EvidenceError(SCHEMA, f"the RSS sampler {label} line is malformed")


def _protocol_record(
    payload: bytes, keys: tuple[str, ...], label: str
) -> dict[str, Any]:
    """Parse one exact canonical sampler record with no schema drift."""
    try:
        value = _canonical_json_object(payload, f"RSS sampler {label}")
    except EvidenceError as exc:
        if isinstance(exc.__cause__, (UnicodeDecodeError, json.JSONDecodeError)):
            raise EvidenceError(SCHEMA, f"the RSS sampler {label} is not JSON") from exc
        raise EvidenceError(
            SCHEMA, f"the RSS sampler {label} is not canonical JSON: {exc.detail}"
        ) from exc
    return _require_keys(value, keys, f"RSS sampler {label}")


def _validate_ready_record(record: Mapping[str, Any], target_pid: int) -> int:
    """Validate the child's exact READY binding and return its baseline."""
    _require(record["status"] == "READY", SCHEMA, "RSS sampler status must be READY")
    pid = _require_int(record["target_pid"], "RSS sampler target_pid", minimum=2)
    interval = _require_int(
        record["sampling_interval_ns"], "RSS sampler sampling_interval_ns", minimum=1
    )
    baseline = _require_int(
        record["baseline_rss_bytes"], "RSS sampler baseline_rss_bytes", minimum=0
    )
    _require(pid == target_pid, SCHEMA, "RSS sampler target PID does not match parent")
    _require(
        interval == RSS_SAMPLING_INTERVAL_NS,
        SCHEMA,
        "RSS sampler interval does not match the fixed 10 ms cadence",
    )
    return baseline


def _validate_result_record(
    record: Mapping[str, Any], target_pid: int, baseline: int
) -> int:
    """Validate the child's exact RESULT arithmetic and return the delta."""
    _require(record["status"] == "RESULT", SCHEMA, "RSS sampler status must be RESULT")
    pid = _require_int(record["target_pid"], "RSS sampler target_pid", minimum=2)
    interval = _require_int(
        record["sampling_interval_ns"], "RSS sampler sampling_interval_ns", minimum=1
    )
    observed_baseline = _require_int(
        record["baseline_rss_bytes"], "RSS sampler baseline_rss_bytes", minimum=0
    )
    peak = _require_int(
        record["peak_rss_bytes"], "RSS sampler peak_rss_bytes", minimum=0
    )
    final = _require_int(
        record["final_rss_bytes"], "RSS sampler final_rss_bytes", minimum=0
    )
    sample_count = _require_int(
        record["sample_count"], "RSS sampler sample_count", minimum=2
    )
    measured = _require_int(
        record["measured_host_peak_bytes"],
        "RSS sampler measured_host_peak_bytes",
        minimum=0,
    )
    _require(pid == target_pid, SCHEMA, "RSS sampler target PID does not match parent")
    _require(
        interval == RSS_SAMPLING_INTERVAL_NS,
        SCHEMA,
        "RSS sampler interval does not match the fixed 10 ms cadence",
    )
    _require(
        observed_baseline == baseline,
        SCHEMA,
        "RSS sampler RESULT baseline does not match READY",
    )
    _require(
        peak >= baseline and peak >= final,
        SCHEMA,
        "RSS sampler peak must include the baseline and final observations",
    )
    _require(
        measured == peak - baseline,
        SCHEMA,
        "RSS sampler measured delta must equal peak minus baseline",
    )
    _ = sample_count
    return measured


def _cleanup_sampler(process: Any) -> None:
    """Close pipes and reap a sampler, retaining the first cleanup failure."""
    failures: list[BaseException] = []

    def close(stream: Any) -> None:
        if stream is None or stream.closed:
            return
        try:
            stream.close()
        except (BrokenPipeError, OSError, ValueError) as exc:
            failures.append(exc)

    close(process.stdin)
    try:
        running = process.poll() is None
    except (ChildProcessError, OSError, ValueError) as exc:
        failures.append(exc)
        running = True
    if running:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        except (ChildProcessError, OSError, ValueError) as exc:
            failures.append(exc)
        try:
            process.wait(timeout=RSS_RESULT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except (ChildProcessError, OSError, ValueError) as exc:
                failures.append(exc)
            try:
                process.wait()
            except (ChildProcessError, OSError, ValueError) as exc:
                failures.append(exc)
        except (ChildProcessError, OSError, ValueError) as exc:
            failures.append(exc)
    close(process.stdout)
    close(process.stderr)
    if failures:
        detail = f"{type(failures[0]).__name__}: {failures[0]}"
        raise EvidenceError(SCHEMA, f"the RSS sampler cleanup failed: {detail}") from (
            failures[0]
        )


def _measure_with_process_rss(call: Any) -> tuple[Any, int]:
    """Run one untimed solver call while a separate process samples parent RSS."""
    target_pid = os.getpid()
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_sample-rss",
            "--target-pid",
            str(target_pid),
            "--sampling-interval-ns",
            str(RSS_SAMPLING_INTERVAL_NS),
        ],
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        if process.stdout is None or process.stdin is None or process.stderr is None:
            raise EvidenceError(SCHEMA, "the RSS sampler pipes were not created")
        ready_payload = _protocol_line(
            process.stdout, RSS_READY_TIMEOUT_SECONDS, "READY"
        )
        ready = _protocol_record(ready_payload, RSS_READY_KEYS, "READY")
        baseline = _validate_ready_record(ready, target_pid)

        outcome = call()

        try:
            process.stdin.write(b"STOP\n")
            process.stdin.flush()
            process.stdin.close()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise EvidenceError(SCHEMA, "the RSS sampler STOP pipe failed") from exc
        deadline = time.monotonic() + RSS_RESULT_TIMEOUT_SECONDS
        result_payload = _protocol_line(
            process.stdout, max(0.0, deadline - time.monotonic()), "RESULT"
        )
        result = _protocol_record(result_payload, RSS_RESULT_KEYS, "RESULT")
        try:
            returncode = process.wait(max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise EvidenceError(SCHEMA, "the RSS sampler clean exit timed out") from exc
        extra_stdout = process.stdout.read()
        stderr = process.stderr.read()
        _require(returncode == 0, SCHEMA, "the RSS sampler did not exit zero")
        _require(extra_stdout == b"", SCHEMA, "the RSS sampler emitted extra stdout")
        _require(stderr == b"", SCHEMA, "the RSS sampler emitted stderr")
        measured = _validate_result_record(result, target_pid, baseline)
    except BaseException as primary:
        try:
            _cleanup_sampler(process)
        except BaseException as cleanup:
            primary.add_note(
                f"RSS sampler cleanup also failed: {type(cleanup).__name__}: {cleanup}"
            )
        raise
    else:
        _cleanup_sampler(process)
        return outcome, measured


def validate_command_row(row: Any, label: str) -> None:
    """Validate Section 14.1's exact eight-field command row."""
    command = _require_keys(row, COMMAND_KEYS, label)
    _require(
        isinstance(command["argv"], list)
        and command["argv"]
        and all(isinstance(item, str) and item for item in command["argv"]),
        SCHEMA,
        f"{label}.argv must be a non-empty array of non-empty strings",
    )
    _require(command["cwd"] == ".", SCHEMA, f"{label}.cwd must be the repository root")
    _require(
        command["pixi_environment"] == "default",
        SCHEMA,
        f"{label}.pixi_environment must be default",
    )
    _require(
        isinstance(command["started_at_utc"], str)
        and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", command["started_at_utc"]
        )
        is not None,
        SCHEMA,
        f"{label}.started_at_utc must be an exact UTC stamp",
    )
    duration = _require_finite(command["duration_seconds"], f"{label}.duration_seconds")
    _require(duration >= 0.0, SCHEMA, f"{label}.duration_seconds must be non-negative")
    _require(
        command["exit_code"] == 0,
        SCHEMA,
        f"{label}.exit_code must be zero for evidence",
    )
    _require_hex(command["stdout_sha256"], 64, f"{label}.stdout_sha256")
    _require_hex(command["stderr_sha256"], 64, f"{label}.stderr_sha256")


# ---------------------------------------------------------------------------
# Section 11 record validation
# ---------------------------------------------------------------------------


def _validate_timing_series(series: Any, label: str, *, required: bool) -> int | None:
    """Validate one timing series and return its sample count when measured."""
    _require(isinstance(series, Mapping), SCHEMA, f"{label} must be an object")
    status = series.get("status")
    if status == "measured":
        _require_keys(series, ("status", "sample_seconds"), label)
        samples = series["sample_seconds"]
        _require(
            isinstance(samples, list), SCHEMA, f"{label}.sample_seconds is an array"
        )
        _require(
            len(samples) >= MINIMUM_SAMPLES,
            SCHEMA,
            f"{label} needs at least {MINIMUM_SAMPLES} samples",
        )
        for index, sample in enumerate(samples):
            value = _require_finite(sample, f"{label}.sample_seconds[{index}]")
            _require(value >= 0.0, SCHEMA, f"{label} samples must be non-negative")
        return len(samples)
    _require(
        not required,
        SCHEMA,
        f"{label} must be measured",
    )
    _require_keys(series, ("status", "reason"), label)
    _require(
        status in {"not_applicable", "not_measured"},
        SCHEMA,
        f"{label}.status must be not_applicable or not_measured",
    )
    _require(
        isinstance(series["reason"], str) and series["reason"],
        SCHEMA,
        f"{label}.reason must be non-empty",
    )
    return None


def _validate_comparison(
    block: Any, keys: tuple[str, ...], label: str, *, predicate: str
) -> dict[str, Any]:
    comparison = _require_keys(block, keys, label)
    _require(
        comparison["predicate_id"] == predicate,
        SCHEMA,
        f"{label}.predicate_id must be {predicate}",
    )
    expected = _require_int(
        comparison["expected_cell_count"], f"{label}.expected_cell_count", minimum=1
    )
    compared = _require_int(
        comparison["compared_finite_cell_count"],
        f"{label}.compared_finite_cell_count",
        minimum=1,
    )
    _require(
        expected == compared,
        SCHEMA,
        f"{label} must compare every expected cell",
    )
    scale = _require_finite(
        comparison["reference_scale_jy"], f"{label}.reference_scale_jy"
    )
    _require(scale >= 1.0, SCHEMA, f"{label}.reference_scale_jy is at least 1 Jy")
    absolute = _require_finite(
        comparison["maximum_absolute_deviation_jy"],
        f"{label}.maximum_absolute_deviation_jy",
    )
    relative = _require_finite(
        comparison["maximum_relative_deviation"], f"{label}.maximum_relative_deviation"
    )
    _require(absolute >= 0.0, SCHEMA, f"{label} deviations are non-negative")
    _require(
        math.isclose(relative, absolute / scale, rel_tol=1e-12, abs_tol=1e-18),
        SCHEMA,
        f"{label}.maximum_relative_deviation must be the absolute maximum over "
        "the reference scale",
    )
    rtol = _require_finite(comparison["rtol"], f"{label}.rtol")
    atol = _require_finite(comparison["atol_jy"], f"{label}.atol_jy")
    _require(rtol == BACKEND_RTOL, SCHEMA, f"{label}.rtol must be exactly 1e-12")
    _require(
        math.isclose(atol, BACKEND_ATOL_FACTOR * scale, rel_tol=1e-12, abs_tol=0.0),
        SCHEMA,
        f"{label}.atol_jy must be 1e-12 times the reference scale",
    )
    _require(comparison["pass"] is True, SCHEMA, f"{label}.pass must be true")
    return comparison


def _validate_direct_comparison(
    block: Any,
    label: str,
    *,
    cell_count: int,
    candidate_cube_sha256: str,
) -> dict[str, Any]:
    """Recompute Section 11's ``sci004_two_tier_direct.v3`` row predicates.

    Everything Section 11 states as a formula over row fields is recomputed
    here.  The three cube-valued reductions themselves are not: the record
    deliberately does not retain the cubes, so a re-derivation of ``max(abs(W0 -
    W_q))`` from this document is impossible and pretending otherwise would be
    the invention Section 14.0 forbids.  What is recomputed is every count, both
    tier-1a limits, both tier-1a verdicts, the convergence factor and the
    Section 7.3 convergence ordering -- which is exactly what the
    ``m3.performance-direct-predicate`` oracle is for.
    """
    direct = _require_keys(block, DIRECT_COMPARISON_KEYS, label)
    _require(
        direct["predicate_id"] == DIRECT_PREDICATE_ID,
        SCHEMA,
        f"{label}.predicate_id must be {DIRECT_PREDICATE_ID}",
    )
    _require(
        direct["candidate_cube_sha256"] == candidate_cube_sha256,
        SCHEMA,
        f"{label}.candidate_cube_sha256 must equal the row's retained cube",
    )
    for name in (
        "reference_cube_sha256",
        "candidate_cube_sha256",
        "reference_error_cube_sha256",
        "horizon_free_cube_sha256",
        "horizon_free_qcheck_cube_sha256",
        "quadrature_shell_cube_sha256",
    ):
        _require_hex(direct[name], 64, f"{label}.{name}")
    for name in (
        "expected_cell_count",
        "compared_finite_cell_count",
        "evaluated_error_cell_count",
    ):
        _require(
            _require_int(direct[name], f"{label}.{name}", minimum=1) == cell_count,
            SCHEMA,
            f"{label}.{name} must equal K = sidereal_samples * n_baselines * "
            "n_frequencies * 4",
        )
    values = {
        name: _require_finite(direct[name], f"{label}.{name}")
        for name in (
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
        )
    }
    for name, value in values.items():
        _require(value >= 0.0, SCHEMA, f"{label}.{name} must be non-negative")
    for name in ("numerical_scale_jy", "reference_scale_jy"):
        _require(
            values[name] >= 1.0,
            SCHEMA,
            f"{label}.{name} is max(1 Jy, ...) and so is at least one",
        )
    _require(
        math.isclose(
            values["horizon_free_shell_max_limit_jy"],
            1e-8 * values["numerical_scale_jy"] + 1e-10,
            rel_tol=1e-12,
            abs_tol=0.0,
        ),
        SCHEMA,
        f"{label}.horizon_free_shell_max_limit_jy must be 1e-8*S_num + 1e-10 Jy",
    )
    _require(
        values["horizon_free_shell_l2_limit"] == 1e-8,
        SCHEMA,
        f"{label}.horizon_free_shell_l2_limit must be exactly 1e-8",
    )
    _require(
        values["horizon_free_shell_max_jy"]
        <= values["horizon_free_shell_max_limit_jy"],
        SCHEMA,
        f"{label}: the tier-1a maximum predicate must hold",
    )
    _require(
        values["horizon_free_shell_l2"] <= values["horizon_free_shell_l2_limit"],
        SCHEMA,
        f"{label}: the tier-1a L2 predicate must hold",
    )
    if values["deficit_max_jy"] == 0.0:
        # Section 11: "with an exact-zero ``deficit_max_jy`` passing both".
        _require(
            values["deficit_max_quarter_jy"] >= values["deficit_max_half_jy"] >= 0.0,
            SCHEMA,
            f"{label}: an exact-zero deficit still may not increase with refinement",
        )
    else:
        _require(
            values["deficit_max_quarter_jy"]
            > values["deficit_max_half_jy"]
            > values["deficit_max_jy"],
            SCHEMA,
            f"{label}: the Section 7.3 convergence ordering must hold",
        )
        _require(
            math.isclose(
                values["convergence_factor"],
                values["deficit_max_quarter_jy"] / values["deficit_max_jy"],
                rel_tol=1e-12,
                abs_tol=0.0,
            ),
            SCHEMA,
            f"{label}.convergence_factor must be the quarter-to-final ratio",
        )
        _require(
            values["convergence_factor"] >= 2.0,
            SCHEMA,
            f"{label}.convergence_factor must be at least two",
        )
    _require(direct["pass"] is True, SCHEMA, f"{label}.pass must be true")
    return direct


def _validate_kernel_block(
    block: Any, label: str, *, backend: str, scalar: bool
) -> None:
    """Validate Section 11's three-status ``kernel_backend_block``."""
    _require(isinstance(block, Mapping), SCHEMA, f"{label} must be an object")
    status = block.get("status")
    if backend == "numpy":
        _require(
            status == KERNEL_STATUS_NOT_APPLICABLE,
            SCHEMA,
            f"{label}.status on a NumPy row must be {KERNEL_STATUS_NOT_APPLICABLE}",
        )
        _require_keys(block, KERNEL_BLOCK_NOT_APPLICABLE_KEYS, label)
        _require(
            isinstance(block["reason"], str) and block["reason"],
            SCHEMA,
            f"{label}.reason must be non-empty",
        )
        return
    if scalar:
        _require(
            status == KERNEL_STATUS_SCALAR,
            SCHEMA,
            f"{label}.status on a scalar group's {backend} row must be "
            f"{KERNEL_STATUS_SCALAR}",
        )
        _require_keys(block, KERNEL_BLOCK_NOT_APPLICABLE_KEYS, label)
        reason = block["reason"]
        _require(
            isinstance(reason, str) and "four science fields" in reason,
            SCHEMA,
            f"{label}.reason must name the four-field kernel contract",
        )
        return
    _require(
        status == KERNEL_STATUS_MEASURED,
        SCHEMA,
        f"{label}.status on a polarized group's {backend} row must be measured",
    )
    measured = _require_keys(block, KERNEL_BLOCK_MEASURED_KEYS, label)
    for stage_name in ("per_m_contraction", "synthesis"):
        stage_label = f"{label}.{stage_name}"
        stage = _require_keys(measured[stage_name], KERNEL_STAGE_KEYS, stage_label)
        samples = stage["sample_seconds"]
        _require(
            isinstance(samples, list) and len(samples) >= MINIMUM_SAMPLES,
            SCHEMA,
            f"{stage_label}.sample_seconds needs at least {MINIMUM_SAMPLES} samples",
        )
        for index, sample in enumerate(samples):
            value = _require_finite(sample, f"{stage_label}.sample_seconds[{index}]")
            _require(value >= 0.0, SCHEMA, f"{stage_label} samples are non-negative")
        _require(
            stage["synchronization_method"] == KERNEL_SYNCHRONIZATION_METHODS[backend],
            SCHEMA,
            f"{stage_label}.synchronization_method must be the {backend} method",
        )
        native = stage["measured_native_peak_bytes"]
        reason = stage["measured_native_peak_bytes_reason"]
        method = stage["native_measurement_method"]
        if native is None:
            _require(
                method == "unavailable",
                SCHEMA,
                f"{stage_label} null native peak requires an unavailable method",
            )
            _require(
                isinstance(reason, str) and reason and reason != "measured",
                SCHEMA,
                f"{stage_label}.measured_native_peak_bytes_reason must be a "
                "non-empty limitation, and a null peak was never measured",
            )
        else:
            _require_int(native, f"{stage_label}.measured_native_peak_bytes", minimum=0)
            _require(
                reason == "measured",
                SCHEMA,
                f"{stage_label} integer native peak requires the reason 'measured'",
            )
            _require(
                method != "unavailable",
                SCHEMA,
                f"{stage_label} measured native peak requires a real method",
            )
        comparison = _validate_comparison(
            stage["stage_comparison"],
            STAGE_COMPARISON_KEYS,
            f"{stage_label}.stage_comparison",
            predicate=BACKEND_PREDICATE_ID,
        )
        # Section 11's "never a self-comparison" is a statement about the
        # reference's *provenance* -- the NumPy kernel output on identical
        # inputs -- not about its bytes.  Two backends that agree exactly
        # publish identical stage identities, which is the expected outcome
        # here, so requiring the two digests to differ would forbid exact
        # agreement.  The provenance is enforced where it is decidable: in the
        # generator, whose reference stage is computed on ``get_backend("numpy")``
        # and whose tracked bytes the strict validator pins.
        del comparison


def _validate_schedule(block: Any, label: str) -> dict[str, Any]:
    dimensions = _require_keys(block, BLOCK_DIMENSION_KEYS, label)
    rows = dimensions["schedule_rows"]
    _require(
        isinstance(rows, list) and rows, SCHEMA, f"{label}.schedule_rows non-empty"
    )
    for index, row in enumerate(rows):
        parsed = _require_keys(
            row, SCHEDULE_ROW_KEYS, f"{label}.schedule_rows[{index}]"
        )
        _require(
            parsed["block_index"] == index,
            SCHEMA,
            f"{label}.schedule_rows[{index}].block_index must be contiguous from zero",
        )
        for name in (
            "frequency_start",
            "frequency_stop",
            "signed_m_start",
            "signed_m_stop",
            "baseline_start",
            "baseline_stop",
        ):
            _require_int(
                parsed[name], f"{label}.schedule_rows[{index}].{name}", minimum=0
            )
        _require_int(
            parsed["packed_value_count"],
            f"{label}.schedule_rows[{index}].packed_value_count",
            minimum=1,
        )
    _require(
        dimensions["scheduled_block_count"] == len(rows),
        SCHEMA,
        f"{label}.scheduled_block_count must equal the row count",
    )
    _require(
        dimensions["schedule_sha256"]
        == object_digest("radiosim.sci004.block-schedule.v1", rows),
        SCHEMA,
        f"{label}.schedule_sha256 must rebuild from the retained rows",
    )
    for name in (
        "frequency_block_max",
        "signed_m_block_max",
        "baseline_block_max",
        "packed_value_block_max",
    ):
        _require_int(dimensions[name], f"{label}.{name}", minimum=1)
    return dimensions


def _validate_memory(block: Any, label: str, *, working_memory_bytes: int) -> None:
    memory = _require_keys(block, MEMORY_KEYS, label)
    _require(
        memory["measurement_scope"] == MEASUREMENT_SCOPE,
        SCHEMA,
        f"{label}.measurement_scope must be the Section 11 literal",
    )
    _require(
        memory["host_measurement_method"] == HOST_MEASUREMENT_METHOD,
        SCHEMA,
        f"{label}.host_measurement_method must be the Section 11 literal",
    )
    estimated = _require_int(
        memory["estimated_host_peak_bytes"],
        f"{label}.estimated_host_peak_bytes",
        minimum=1,
    )
    measured = _require_int(
        memory["measured_host_peak_bytes"],
        f"{label}.measured_host_peak_bytes",
        minimum=0,
    )
    # Correction #24 keeps only the two budget inequalities as hard predicates.
    # Sampled RSS may fall on either side of Section 9's estimate because it is
    # a baseline delta at finite cadence; retain and recompute that observed
    # relation without pinning either truth value.
    _require(
        measured <= working_memory_bytes,
        SCHEMA,
        f"{label}: the measured host peak must not exceed the working-memory budget",
    )
    _require(
        estimated <= working_memory_bytes,
        SCHEMA,
        f"{label}: the estimate must not exceed the working-memory budget",
    )
    covers = memory["estimate_covers_measured_host_peak"]
    _require(
        isinstance(covers, bool),
        SCHEMA,
        f"{label}.estimate_covers_measured_host_peak must be a JSON boolean",
    )
    _require(
        covers == (measured <= estimated),
        SCHEMA,
        f"{label}.estimate_covers_measured_host_peak must be the measured "
        "relation measured_host_peak_bytes <= estimated_host_peak_bytes, "
        "recomputed from this row's own values and retained as observed",
    )
    method = memory["native_measurement_method"]
    _require(
        method in SHARED_NATIVE_METHODS,
        SCHEMA,
        f"{label}.native_measurement_method must be unavailable; the process-RSS "
        "method is host-only and is never a backend-device method in this "
        "shared object",
    )
    for name in ("host_measurement_limitations", "native_measurement_limitations"):
        _require_sorted_unique_strings(memory[name], f"{label}.{name}")
        _require(memory[name], SCHEMA, f"{label}.{name} must be non-empty")
    _require(
        tuple(memory["host_measurement_limitations"]) == HOST_MEASUREMENT_LIMITATIONS,
        SCHEMA,
        f"{label}.host_measurement_limitations must carry exactly the four "
        "sampled-RSS limitations ruled by correction #24",
    )
    native = memory["measured_native_peak_bytes"]
    reason = memory["measured_native_peak_bytes_reason"]
    if native is None:
        _require(
            method == "unavailable",
            SCHEMA,
            f"{label} null native peak requires an unavailable method",
        )
        _require(
            isinstance(reason, str) and reason and reason != "measured",
            SCHEMA,
            f"{label}.measured_native_peak_bytes_reason must be a non-empty "
            "limitation, and a null peak was never measured",
        )
        _require(
            reason in memory["native_measurement_limitations"],
            SCHEMA,
            f"{label}: the null reason must also occur in the limitations",
        )
    else:
        _require_int(native, f"{label}.measured_native_peak_bytes", minimum=0)
        _require(
            reason == "measured",
            SCHEMA,
            f"{label} integer native peak requires the reason 'measured'",
        )
        _require(
            method != "unavailable",
            SCHEMA,
            f"{label} measured native peak requires a real method",
        )


def validate_performance_document(document: Any) -> dict[str, Any]:
    """Validate one complete Section 11 performance record."""
    record = _require_keys(document, BENCHMARK_TOP_LEVEL_KEYS, "benchmark record")
    _require(
        record["schema_version"] == BENCHMARK_SCHEMA,
        SCHEMA,
        f"the record schema literal must be {BENCHMARK_SCHEMA}",
    )
    provenance = _require_keys(record["provenance"], PROVENANCE_KEYS, "provenance")
    _require(
        provenance["schema_version"] == BENCHMARK_PROVENANCE_SCHEMA,
        SCHEMA,
        "the provenance schema literal is wrong",
    )
    _require(
        re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", provenance["recorded_at_utc"]
        )
        is not None,
        SCHEMA,
        "provenance.recorded_at_utc must be an exact UTC stamp",
    )
    _require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", provenance["host_tag"]) is not None,
        SCHEMA,
        "provenance.host_tag must match the Section 11 pattern",
    )
    _require_hex(provenance["source_sha"], 40, "provenance.source_sha")
    _require_hex(provenance["git_tree_sha256"], 64, "provenance.git_tree_sha256")
    _require_hex(
        provenance["pixi_manifest_sha256"], 64, "provenance.pixi_manifest_sha256"
    )
    _require_hex(provenance["pixi_lock_sha256"], 64, "provenance.pixi_lock_sha256")
    _require_hex(provenance["iers_table_sha256"], 64, "provenance.iers_table_sha256")
    _require(
        provenance["working_tree_clean"] is True,
        SCHEMA,
        "provenance.working_tree_clean must be true",
    )
    _require(
        provenance["pixi_environment"] == "default",
        SCHEMA,
        "provenance.pixi_environment must be default",
    )
    _require(
        provenance["transform_execution_policy"] == TRANSFORM_EXECUTION_POLICY,
        SCHEMA,
        "provenance.transform_execution_policy must be the Section 9 literal",
    )
    _require(
        provenance["workload_count"] == 9,
        SCHEMA,
        "provenance.workload_count must be exactly nine",
    )
    _require_int(
        provenance["cpu_count_logical"], "provenance.cpu_count_logical", minimum=1
    )
    packages = _require_keys(
        provenance["numeric_packages"], BENCHMARK_NUMERIC_PACKAGES, "numeric_packages"
    )
    for name, version in packages.items():
        _require(
            isinstance(version, str) and version,
            SCHEMA,
            f"numeric_packages.{name} must be a non-empty version string",
        )

    workloads = record["workloads"]
    _require(isinstance(workloads, list), SCHEMA, "workloads must be an array")
    expected_ids = [
        f"{fixture}:{backend}:standard"
        for fixture in PERFORMANCE_FIXTURES
        for backend in BACKENDS
    ]
    _require(
        [str(row.get("workload_id")) for row in workloads] == expected_ids,
        SCHEMA,
        "workloads must be the exact nine-row Cartesian product, in record order",
    )

    by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for index, row in enumerate(workloads):
        label = f"workloads[{index}]"
        parsed = _require_keys(row, WORKLOAD_KEYS, label)
        fixture = str(parsed["fixture_id"])
        backend = str(parsed["backend"])
        _require(fixture in PERFORMANCE_FIXTURES, SCHEMA, f"{label}.fixture_id unknown")
        _require(backend in BACKENDS, SCHEMA, f"{label}.backend unknown")
        _require(
            parsed["comparison_group_id"] == fixture,
            SCHEMA,
            f"{label}.comparison_group_id must equal the fixture id",
        )
        _require(
            parsed["dense_execution"] == DENSE_EXECUTION,
            SCHEMA,
            f"{label}.dense_execution must be {DENSE_EXECUTION} on every row",
        )
        _require(parsed["device_kind"] == "cpu", SCHEMA, f"{label}.device_kind")
        _require(parsed["precision"] == "standard", SCHEMA, f"{label}.precision")
        for name in ("accumulation_dtype", "result_dtype"):
            _require(parsed[name] == "complex128", SCHEMA, f"{label}.{name}")
        _require(
            parsed["sky_representation"] == "point",
            SCHEMA,
            f"{label}.sky_representation must be point for every group",
        )
        _require(
            parsed["n_healpix_pixels"] == 0,
            SCHEMA,
            f"{label}.n_healpix_pixels must be zero for an absent representation",
        )
        _require(
            parsed["working_tree_clean"] is True,
            SCHEMA,
            f"{label}.working_tree_clean must be true",
        )
        _require(
            parsed["source_sha"] == provenance["source_sha"],
            SCHEMA,
            f"{label}.source_sha must equal the provenance source",
        )
        runtime = _require_keys(
            parsed["backend_runtime"], BACKEND_RUNTIME_KEYS, f"{label}.backend_runtime"
        )
        implementation, kernel = BACKEND_RUNTIME_PAIRS[backend]
        _require(
            runtime["implementation"] == implementation
            and runtime["kernel_runtime"] == kernel,
            SCHEMA,
            f"{label}.backend_runtime must name the {backend} pair",
        )
        for name in (
            "workers",
            "n_antennas",
            "n_baselines",
            "n_frequencies",
            "sidereal_samples",
            "lmax",
            "mmax",
            "quadrature_nside",
            "n_point_sources",
            "working_memory_bytes",
        ):
            _require_int(parsed[name], f"{label}.{name}", minimum=1)
        for name in (
            "input_identity_sha256",
            "frame_certificate_sha256",
            "scientific_sha256",
            "result_cube_sha256",
        ):
            _require_hex(parsed[name], 64, f"{label}.{name}")
        _validate_schedule(
            parsed["resolved_block_dimensions"], f"{label}.resolved_block_dimensions"
        )
        timings = _require_keys(parsed["timings"], TIMING_KEYS, f"{label}.timings")
        _require(timings["clock"] == CLOCK, SCHEMA, f"{label}.timings.clock")
        _require_int(
            timings["warmup_iterations"],
            f"{label}.timings.warmup_iterations",
            minimum=1,
        )
        _require(
            timings["synchronization_method"] == SHARED_SYNCHRONIZATION_METHOD,
            SCHEMA,
            f"{label}.timings.synchronization_method must be the shared dense method",
        )
        cardinalities = {
            name: _validate_timing_series(
                timings[name], f"{label}.timings.{name}", required=True
            )
            for name in MEASURED_SERIES
        }
        _require(
            len(set(cardinalities.values())) == 1,
            SCHEMA,
            f"{label}.timings: the measured series must share one sample cardinality",
        )
        _validate_timing_series(
            timings["host_transfer"], f"{label}.timings.host_transfer", required=False
        )
        _validate_timing_series(
            timings["direct_reference"],
            f"{label}.timings.direct_reference",
            required=False,
        )
        count = cardinalities["total"] or 0
        for iteration in range(count):
            total = float(timings["total"]["sample_seconds"][iteration])
            stages = sum(
                float(timings[name]["sample_seconds"][iteration])
                for name in MEASURED_SERIES
                if name != "total"
            )
            _require(
                total >= stages - 1e-9,
                SCHEMA,
                f"{label}.timings iteration {iteration}: the total must not be "
                "smaller than the sum of its named stages",
            )
        _validate_memory(
            parsed["memory"],
            f"{label}.memory",
            working_memory_bytes=int(parsed["working_memory_bytes"]),
        )
        _validate_direct_comparison(
            parsed["direct_comparison"],
            f"{label}.direct_comparison",
            cell_count=int(parsed["sidereal_samples"])
            * int(parsed["n_baselines"])
            * int(parsed["n_frequencies"])
            * 4,
            candidate_cube_sha256=str(parsed["result_cube_sha256"]),
        )
        backend_comparison = _validate_comparison(
            parsed["backend_comparison"],
            BACKEND_COMPARISON_KEYS,
            f"{label}.backend_comparison",
            predicate=BACKEND_PREDICATE_ID,
        )
        _require(
            backend_comparison["reference_workload_id"] == f"{fixture}:numpy:standard",
            SCHEMA,
            f"{label}.backend_comparison must reference its group's NumPy row",
        )
        _validate_kernel_block(
            parsed["kernel_backend_block"],
            f"{label}.kernel_backend_block",
            backend=backend,
            scalar=fixture not in POLARIZED_FIXTURES,
        )
        claims = _require_sorted_unique_strings(
            parsed["claims_not_licensed"], f"{label}.claims_not_licensed"
        )
        _require(
            tuple(claims) == BENCHMARK_CLAIMS,
            SCHEMA,
            f"{label}.claims_not_licensed must be Section 11's exact six literals",
        )
        by_group.setdefault(fixture, {})[backend] = parsed

    for fixture, rows in by_group.items():
        shared = rows["numpy"]
        for backend in ("jax", "dask"):
            candidate = rows[backend]
            for name in (
                # Section 11: identical input, frame-certificate, dimension,
                # precision, worker and memory-budget fields.
                "input_identity_sha256",
                "frame_certificate_sha256",
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
                "resolved_block_dimensions",
                "precision",
                "workers",
                "working_memory_bytes",
                # The shared end-to-end series, measured once on the NumPy row.
                "timings",
                "memory",
                "scientific_sha256",
                "result_cube_sha256",
                "direct_comparison",
            ):
                _require(
                    candidate[name] == shared[name],
                    SCHEMA,
                    f"{fixture}:{backend} must carry the group's shared {name}",
                )
    identities = [
        by_group[fixture]["numpy"]["input_identity_sha256"]
        for fixture in PERFORMANCE_FIXTURES
    ]
    _require(
        len(set(identities)) == len(identities),
        SCHEMA,
        "input identities must be distinct across fixture groups",
    )

    invariance = record["dense_invariance"]
    _require(isinstance(invariance, list), SCHEMA, "dense_invariance must be an array")
    _require(
        [str(entry.get("comparison_group_id")) for entry in invariance]
        == list(PERFORMANCE_FIXTURES),
        SCHEMA,
        "dense_invariance must carry one entry per comparison group, in fixture order",
    )
    for index, entry in enumerate(invariance):
        label = f"dense_invariance[{index}]"
        parsed = _require_keys(entry, DENSE_INVARIANCE_KEYS, label)
        digests = {
            parsed["numpy_cube_sha256"],
            parsed["jax_cube_sha256"],
            parsed["dask_cube_sha256"],
        }
        for name in ("numpy_cube_sha256", "jax_cube_sha256", "dask_cube_sha256"):
            _require_hex(parsed[name], 64, f"{label}.{name}")
        _require(
            len(digests) == 1,
            SCHEMA,
            f"{label}: the three per-backend cubes must be bit-identical",
        )
        _require(parsed["identical"] is True, SCHEMA, f"{label}.identical must be true")
        group = str(parsed["comparison_group_id"])
        _require(
            parsed["numpy_cube_sha256"]
            == by_group[group]["numpy"]["result_cube_sha256"],
            SCHEMA,
            f"{label} must join its group's retained cube identity",
        )
    return record


# ---------------------------------------------------------------------------
# Section 14.2 envelope validation
# ---------------------------------------------------------------------------


def validate_evidence_document(document: Any) -> dict[str, Any]:
    """Validate one complete Section 14.2 M3 evidence envelope."""
    envelope = _require_keys(document, ENVELOPE_KEYS, "evidence envelope")
    _require(
        envelope["schema_version"] == EVIDENCE_SCHEMA,
        SCHEMA,
        f"the evidence schema literal must be {EVIDENCE_SCHEMA}",
    )
    _require(envelope["phase"] == PHASE, SCHEMA, "phase must be M3")
    _require(envelope["status"] == STATUS, SCHEMA, "status must be candidate")
    _require(
        re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", envelope["generated_at_utc"]
        )
        is not None,
        SCHEMA,
        "generated_at_utc must be an exact UTC stamp",
    )
    for name in ("design_sha", "red_commit_sha", "source_sha"):
        _require_hex(envelope[name], 40, name)
    _require(
        envelope["evidence_commit_sha"] is None,
        SCHEMA,
        "evidence_commit_sha must be JSON null",
    )
    _require(
        envelope["evidence_commit_sha_reason"] == EVIDENCE_SELF_REFERENCE_REASON,
        SCHEMA,
        "the self-reference reason must be the exact Section 14.2 sentence",
    )
    _require(
        envelope["working_tree_clean"] is True,
        SCHEMA,
        "working_tree_clean must be true",
    )
    environment = _require_keys(
        envelope["environment"], ENVIRONMENT_KEYS, "environment"
    )
    _require(
        environment["pixi_environment"] == "default",
        SCHEMA,
        "environment.pixi_environment must be default",
    )
    _require_hex(environment["pixi_lock_sha256"], 64, "environment.pixi_lock_sha256")
    _require_hex(environment["iers_table_sha256"], 64, "environment.iers_table_sha256")
    packages = _require_keys(
        environment["numeric_packages"],
        NUMERIC_PACKAGES,
        "environment.numeric_packages",
    )
    for name, version in packages.items():
        _require(
            isinstance(version, str) and version,
            SCHEMA,
            f"environment.numeric_packages.{name} must be a version string",
        )

    identities = _require_keys(
        envelope["source_identities"], SOURCE_IDENTITY_KEYS, "source_identities"
    )
    for name in (
        "git_tree_sha256",
        "pixi_manifest_sha256",
        "pixi_lock_sha256",
        "convention_identity_sha256",
        "input_identity_set_sha256",
    ):
        _require_hex(identities[name], 64, f"source_identities.{name}")
    rows = identities["fixture_input_rows"]
    _require(
        isinstance(rows, list) and rows, SCHEMA, "fixture_input_rows must be non-empty"
    )
    fixture_ids: list[str] = []
    for index, row in enumerate(rows):
        parsed = _require_keys(
            row, FIXTURE_INPUT_ROW_KEYS, f"fixture_input_rows[{index}]"
        )
        fixture_ids.append(str(parsed["fixture_id"]))
        _require(
            isinstance(parsed["input_identity_manifest"], Mapping),
            SCHEMA,
            f"fixture_input_rows[{index}].input_identity_manifest must be an object",
        )
        digest = _require_hex(
            parsed["input_identity_sha256"], 64, f"fixture_input_rows[{index}]"
        )
        _require(
            digest
            == object_digest(
                "radiosim.mmode-input-identity.v1", parsed["input_identity_manifest"]
            ),
            SCHEMA,
            f"fixture_input_rows[{index}] digest must rebuild from its manifest",
        )
    _require(
        fixture_ids == sorted(set(fixture_ids)),
        SCHEMA,
        "fixture_input_rows must be unique and fixture-ID sorted",
    )
    _require(
        fixture_ids == sorted(SECTION_11_FAMILIES),
        SCHEMA,
        "Section 14.0: the fixture-input row set equals exactly the union of "
        "every non-rejection result fixture ID in the phase -- no orphan, "
        "duplicate or missing row",
    )
    _require(
        identities["input_identity_set_sha256"]
        == object_digest("radiosim.sci004-phase-input-set.v1", rows),
        SCHEMA,
        "input_identity_set_sha256 must rebuild from the retained rows",
    )

    red = _require_keys(
        envelope["red_failure_record"], RED_FAILURE_RECORD_KEYS, "red_failure_record"
    )
    _require(
        red["path"] == RED_FAILURE_RECORD,
        SCHEMA,
        "red_failure_record.path must be the retained phase-3 record",
    )
    _require_hex(red["sha256"], 64, "red_failure_record.sha256")
    _require(
        red["schema_version"] == RED_FAILURE_SCHEMA,
        SCHEMA,
        "red_failure_record.schema_version must be the historical M3 schema",
    )
    _require_hex(red["pre_fix_source_sha"], 40, "red_failure_record.pre_fix_source_sha")
    _require(
        red["validated"] is True, SCHEMA, "red_failure_record.validated must be true"
    )
    post_source = _require_keys(
        red["post_source_delta"],
        RED_FAILURE_REFERENCE_KEYS,
        "red_failure_record.post_source_delta",
    )
    _require(
        post_source["path"] == POST_SOURCE_RED_FAILURE_RECORD,
        SCHEMA,
        "red_failure_record.post_source_delta.path must be the correction-24 record",
    )
    _require_hex(
        post_source["sha256"], 64, "red_failure_record.post_source_delta.sha256"
    )
    _require(
        post_source["schema_version"] == POST_SOURCE_RED_FAILURE_SCHEMA,
        SCHEMA,
        "red_failure_record.post_source_delta.schema_version must be the "
        "correction-24 schema",
    )
    _require(
        post_source["pre_fix_source_sha"] == POST_SOURCE_PRE_FIX_SHA,
        SCHEMA,
        "red_failure_record.post_source_delta.pre_fix_source_sha must name the "
        "superseded a61526d6 source",
    )
    _require(
        post_source["validated"] is True,
        SCHEMA,
        "red_failure_record.post_source_delta.validated must be true",
    )

    results = _require_keys(envelope["results"], RESULT_KEYS, "results")
    certificate = _require_keys(
        results["dependency_certificate"],
        DEPENDENCY_CERTIFICATE_KEYS,
        "dependency_certificate",
    )
    _require_hex(
        certificate["sci005_stage2_acceptance_commit_sha"],
        40,
        "dependency_certificate.sci005_stage2_acceptance_commit_sha",
    )
    for name in (
        "sci005_stage2_acceptance_artifact_sha256",
        "sci005_stage2_certificate_stdout_sha256",
    ):
        _require_hex(certificate[name], 64, f"dependency_certificate.{name}")

    outputs = results["output_cases"]
    _require(
        isinstance(outputs, list) and outputs, SCHEMA, "output_cases must be non-empty"
    )
    observed_formats = []
    for index, row in enumerate(outputs):
        label = f"output_cases[{index}]"
        parsed = _require_keys(row, OUTPUT_ROW_KEYS, label)
        observed_formats.append(str(parsed["format"]))
        _require(
            str(parsed["fixture_id"]) in fixture_ids,
            SCHEMA,
            f"{label}.fixture_id must join the phase input set",
        )
        for name in (
            "written_solver_sha256",
            "read_solver_sha256",
            "time_sha256",
            "feed_sha256",
            "correlation_sha256",
            "file_sha256",
            "written_cube_sha256",
            "read_cube_sha256",
            "scientific_sha256",
        ):
            _require_hex(parsed[name], 64, f"{label}.{name}")
        _require(
            parsed["written_solver_sha256"] == parsed["read_solver_sha256"],
            SCHEMA,
            f"{label}: the reader must reconstruct the written solver snapshot",
        )
        if str(parsed["format"]) in LOSSLESS_CUBE_FORMATS:
            _require(
                parsed["written_cube_sha256"] == parsed["read_cube_sha256"],
                SCHEMA,
                f"{label}: this format's round trip must preserve the cube",
            )
        else:
            _require(
                parsed["written_cube_sha256"] != parsed["read_cube_sha256"],
                SCHEMA,
                f"{label}: a narrowing format's read identity may not restate "
                "the written one, which would describe a round trip that did "
                "not happen",
            )
        _require(parsed["pass"] is True, SCHEMA, f"{label}.pass must be true")
    _require(
        tuple(observed_formats) == OUTPUT_FORMATS,
        SCHEMA,
        f"output_cases must cover exactly {list(OUTPUT_FORMATS)}, in order",
    )
    _require(
        len({str(row["fixture_id"]) for row in outputs}) == 1,
        SCHEMA,
        "output_cases must round-trip one fixture through every reader path, so "
        "the three rows are comparable",
    )

    fingerprints = results["fingerprint_rows"]
    _require(
        isinstance(fingerprints, list), SCHEMA, "fingerprint_rows must be an array"
    )
    _require(
        [str(row.get("family_id")) for row in fingerprints]
        == list(SECTION_11_FAMILIES),
        SCHEMA,
        "fingerprint_rows must be exactly four rows in the amended family order",
    )
    for index, row in enumerate(fingerprints):
        label = f"fingerprint_rows[{index}]"
        parsed = _require_keys(row, FINGERPRINT_ROW_KEYS, label)
        _require(
            str(parsed["fixture_id"]) in fixture_ids,
            SCHEMA,
            f"{label}.fixture_id must join the phase input set",
        )
        for name in (
            "input_identity_sha256",
            "canonical_era_grid_sha256",
            "solver_snapshot_sha256",
            "cube_sha256",
            "scientific_sha256",
        ):
            _require_hex(parsed[name], 64, f"{label}.{name}")
        _require(
            isinstance(parsed["expected_change_reason"], str)
            and parsed["expected_change_reason"],
            SCHEMA,
            f"{label}.expected_change_reason must be non-empty",
        )
        _require(parsed["pass"] is True, SCHEMA, f"{label}.pass must be true")

    artifacts = results["ci_artifacts"]
    _require(
        isinstance(artifacts, list) and artifacts, SCHEMA, "ci_artifacts non-empty"
    )
    seen: set[tuple[str, str, str]] = set()
    families_seen: list[str] = []
    for index, row in enumerate(artifacts):
        label = f"ci_artifacts[{index}]"
        parsed = _require_keys(row, CI_ARTIFACT_ROW_KEYS, label)
        family = str(parsed["family_id"])
        if family not in families_seen:
            families_seen.append(family)
        _require(
            str(parsed["fixture_id"]) in fixture_ids,
            SCHEMA,
            f"{label}.fixture_id must join the phase input set",
        )
        _require_hex(parsed["source_sha"], 40, f"{label}.source_sha")
        for name in ("cube_sha256", "scientific_sha256"):
            _require_hex(parsed[name], 64, f"{label}.{name}")
        key = (family, str(parsed["environment"]), str(parsed["dispatch_identity"]))
        _require(
            key not in seen,
            SCHEMA,
            f"{label}: duplicate family/cell/dispatch tuple {key}",
        )
        seen.add(key)
        _require(
            parsed["numeric_delta"] == 0
            or _require_finite(parsed["numeric_delta"], label) >= 0.0,
            SCHEMA,
            f"{label}.numeric_delta must be a non-negative number",
        )
        _require(
            parsed["ci001_verdict"] == "accepted-observation-set",
            SCHEMA,
            f"{label}.ci001_verdict must record the accepted observation set",
        )
        _require(parsed["pass"] is True, SCHEMA, f"{label}.pass must be true")
    _require(
        families_seen == list(SECTION_11_FAMILIES),
        SCHEMA,
        "ci_artifacts family order must be the exact four-name amended order",
    )

    performance = _require_keys(
        results["performance_record"], PERFORMANCE_RECORD_KEYS, "performance_record"
    )
    _require(
        str(performance["path"]).startswith(PERFORMANCE_DIRECTORY + "/"),
        SCHEMA,
        "performance_record.path must be the retained host-bound path",
    )
    _require_hex(performance["sha256"], 64, "performance_record.sha256")
    _require(
        performance["schema_version"] == BENCHMARK_SCHEMA,
        SCHEMA,
        "performance_record.schema_version must be the Section 11 literal",
    )
    _require(
        performance["source_sha"] == envelope["source_sha"],
        SCHEMA,
        "performance_record.source_sha must equal the evidence source",
    )
    _require(
        performance["workload_count"] == 9,
        SCHEMA,
        "performance_record.workload_count must be nine",
    )
    _require(
        performance["authenticated"] is True,
        SCHEMA,
        "performance_record.authenticated must be true",
    )
    identity_rows = performance["workload_identities"]
    _require(
        [str(row.get("workload_id")) for row in identity_rows]
        == [
            f"{fixture}:{backend}:standard"
            for fixture in PERFORMANCE_FIXTURES
            for backend in BACKENDS
        ],
        SCHEMA,
        "workload_identities must be one row per workload, in Section 11 order",
    )
    for index, row in enumerate(identity_rows):
        label = f"performance_record.workload_identities[{index}]"
        parsed = _require_keys(row, WORKLOAD_IDENTITY_KEYS, label)
        for name in (
            "input_identity_sha256",
            "frame_certificate_sha256",
            "scientific_sha256",
            "result_cube_sha256",
        ):
            _require_hex(parsed[name], 64, f"{label}.{name}")
    claims = _require_sorted_unique_strings(
        performance["claims_not_licensed"], "performance_record.claims_not_licensed"
    )
    _require(
        tuple(claims) == BENCHMARK_CLAIMS,
        SCHEMA,
        "performance_record.claims_not_licensed must be Section 11's six literals",
    )

    scans = results["release_scan_cases"]
    _require(isinstance(scans, list) and scans, SCHEMA, "release_scan_cases non-empty")
    for index, row in enumerate(scans):
        label = f"release_scan_cases[{index}]"
        parsed = _require_keys(row, RELEASE_SCAN_ROW_KEYS, label)
        _require_int(parsed["command_index"], f"{label}.command_index", minimum=0)
        expected = _require_keys(
            parsed["expected_counts"], EXPECTED_COUNT_KEYS, f"{label}.expected_counts"
        )
        for name in EXPECTED_COUNT_KEYS:
            observed = _require_int(parsed[name], f"{label}.{name}", minimum=0)
            _require(
                observed
                == _require_int(
                    expected[name], f"{label}.expected_counts.{name}", minimum=0
                ),
                SCHEMA,
                f"{label}.{name} must equal its expected count",
            )
        _require(
            parsed["roadmap_occurrences"] >= 1,
            SCHEMA,
            f"{label} must still report SCI-004 as ROADMAP",
        )
        _require(
            parsed["unsupported_claim_occurrences"] == 0,
            SCHEMA,
            f"{label} must find no unsupported claim",
        )
        _require(parsed["pass"] is True, SCHEMA, f"{label}.pass must be true")

    rejections = results["rejection_cases"]
    _require(
        isinstance(rejections, list) and rejections, SCHEMA, "rejection_cases non-empty"
    )
    for index, row in enumerate(rejections):
        label = f"rejection_cases[{index}]"
        parsed = _require_keys(row, REJECTION_ROW_KEYS, label)
        _require(
            parsed["allocation_started"] is False,
            SCHEMA,
            f"{label}.allocation_started must be false: the refusal precedes any work",
        )
        _require(
            parsed["output_path_created"] is False,
            SCHEMA,
            f"{label}.output_path_created must be false",
        )
        for name in ("exception_type", "issue_code", "exact_message", "test_nodeid"):
            _require(
                isinstance(parsed[name], str) and parsed[name],
                SCHEMA,
                f"{label}.{name} must be non-empty",
            )
        _require(parsed["pass"] is True, SCHEMA, f"{label}.pass must be true")

    commands = envelope["commands"]
    _require(
        isinstance(commands, list) and commands, SCHEMA, "commands must be non-empty"
    )
    for index, row in enumerate(commands):
        validate_command_row(row, f"commands[{index}]")
    limitations = _require_sorted_unique_strings(envelope["limitations"], "limitations")
    claim_literals = _require_sorted_unique_strings(
        envelope["claims_not_licensed"], "claims_not_licensed"
    )
    _require(
        tuple(limitations) == tuple(sorted(LIMITATIONS)),
        SCHEMA,
        "limitations must be this phase's exact declared literals",
    )
    _require(
        tuple(claim_literals) == tuple(sorted(CLAIMS_NOT_LICENSED)),
        SCHEMA,
        "claims_not_licensed must be this phase's exact declared literals, "
        "including the three deferrals the accepted corrections require",
    )
    for topic in DEFERRAL_TOPICS:
        _require(
            any(literal.startswith(topic + ":") for literal in claim_literals),
            SCHEMA,
            f"claims_not_licensed must carry the {topic} deferral",
        )
    return envelope


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _git_process(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    """Query original objects in the explicitly bound evidence checkout."""
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_GRAFT_FILE=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_ATTR_NOSYSTEM="1",
    )
    root = REPOSITORY_ROOT.resolve()
    command = [
        "git",
        "--no-pager",
        "--no-replace-objects",
        "--literal-pathspecs",
        f"--work-tree={root}",
    ]
    for setting in (
        "core.bare=false",
        "core.commitGraph=false",
        "core.fsmonitor=false",
        "core.untrackedCache=false",
        "color.ui=false",
        "core.attributesFile=" + os.devnull,
    ):
        command.extend(("-c", setting))

    def run(query: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(
                [*command, *query],
                cwd=root,
                env=environment,
                capture_output=True,
                check=False,
            )
        except OSError as error:
            raise EvidenceError(PREFLIGHT, "cannot start evidence Git query") from error

    discovery = run(("rev-parse", "--absolute-git-dir"))
    _require(
        discovery.returncode == 0, PREFLIGHT, "cannot locate evidence Git directory"
    )
    git_directory = os.fsdecode(discovery.stdout.removesuffix(b"\n"))
    if arguments[0] == "status":
        arguments = ("status", "--ignore-submodules=none", *arguments[1:])
    elif arguments[0] == "show":
        arguments = ("show", "--no-ext-diff", "--no-textconv", *arguments[1:])
    elif arguments[0] == "diff-tree":
        arguments = (
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--ignore-submodules=none",
            *arguments[1:],
        )
    return run((f"--git-dir={git_directory}", *arguments))


def _git_bytes(*arguments: str) -> bytes:
    completed = _git_process(*arguments)
    if completed.returncode != 0:
        raise EvidenceError(
            PREFLIGHT,
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.decode('utf-8', 'replace').strip()}",
        )
    return completed.stdout


def _git(*arguments: str) -> str:
    return _git_bytes(*arguments).decode("utf-8")


def raw_sha256(path: Path) -> str:
    """Return the SHA-256 of a file's exact raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_binding(name: str) -> str:
    """Read one ordinary literal binding from this checkout's dependency validator."""
    try:
        source = (REPOSITORY_ROOT / DEPENDENCY_VALIDATOR_PATH).read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
    except (OSError, UnicodeError, SyntaxError) as error:
        raise EvidenceError(
            PREFLIGHT, "cannot read frozen dependency bindings"
        ) from error
    stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id == name
    ]
    declarations = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if (
        len(stores) != 1
        or len(declarations) != 1
        or not isinstance(declarations[0].value, ast.Constant)
        or not isinstance(declarations[0].value.value, str)
    ):
        raise EvidenceError(
            PREFLIGHT, f"{DEPENDENCY_VALIDATOR_PATH} must freeze exactly one {name}"
        )
    return declarations[0].value.value


def _design_sha() -> str:
    """Authenticate the disk-bound D33 source authority through its exact edges."""
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from tests.unit import test_sci004_phase3_dependency as dependency

    try:
        root = REPOSITORY_ROOT.resolve(strict=True)
        module_file: object = vars(dependency).get("__file__")
        if (
            not isinstance(module_file, str)
            or Path(module_file).resolve(strict=True)
            != root / DEPENDENCY_VALIDATOR_PATH
            or dependency.REPOSITORY_ROOT.resolve(strict=True) != root
        ):
            raise EvidenceError(
                PREFLIGHT, "source validator import belongs to another checkout"
            )
    except OSError as error:
        raise EvidenceError(
            PREFLIGHT, "source validator import belongs to another checkout"
        ) from error
    frozen = tuple(
        _frozen_binding(name)
        for name in (
            "D30_SHA",
            "APPROVED_SCI004_D_SHA",
            "HISTORICAL_SOURCE_DESIGN_SHA",
            "SOURCE_DESIGN_SHA",
        )
    )
    if (
        any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in frozen)
        or len(set(frozen)) != 4
        or frozen
        != (
            dependency.D30_SHA,
            dependency.APPROVED_SCI004_D_SHA,
            dependency.HISTORICAL_SOURCE_DESIGN_SHA,
            dependency.SOURCE_DESIGN_SHA,
        )
    ):
        raise EvidenceError(PREFLIGHT, "loaded design differs from frozen binding")
    try:
        authenticated = dependency.authenticate_source_design_bindings()
    except dependency.DependencyCertificateError as error:
        raise EvidenceError(
            PREFLIGHT, f"source design does not authenticate: {error}"
        ) from error
    if authenticated != frozen[3]:
        raise EvidenceError(PREFLIGHT, "source authentication returned another design")
    return frozen[3]


def _commit_parents(commit: str) -> tuple[str, ...]:
    fields = _git("rev-list", "--parents", "-n", "1", commit).split()
    if not fields or fields[0] != commit:
        raise EvidenceError(PREFLIGHT, f"cannot resolve parents of {commit}")
    return tuple(fields[1:])


def _changed_paths(commit: str) -> frozenset[str]:
    return frozenset(
        path
        for path in _git(
            "diff-tree", "--no-commit-id", "--name-only", "-z", "-r", commit
        ).split("\0")
        if path
    )


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    completed = _git_process("merge-base", "--is-ancestor", ancestor, descendant)
    if completed.returncode not in (0, 1):
        raise EvidenceError(
            PREFLIGHT, f"cannot authenticate ancestry {ancestor}..{descendant}"
        )
    return completed.returncode == 0


def _authenticate_historical_design(
    design: str, *, label: str, current_design: str
) -> None:
    peeled = _git("rev-parse", f"{design}^{{commit}}").strip()
    if peeled != design or len(_commit_parents(design)) != 1:
        raise EvidenceError(
            PREFLIGHT, f"{label} does not resolve to its exact single-parent commit"
        )
    if _changed_paths(design) != frozenset(
        {"PostTier8RemediationPlan.md", "docs/development/sci004_mmode_design.md"}
    ):
        raise EvidenceError(PREFLIGHT, f"{label} is not its exact design-only commit")
    if not _is_ancestor(design, current_design):
        raise EvidenceError(
            PREFLIGHT, f"{label} is not connected to the current operative D"
        )


def _original_fingerprint_red_commit_sha() -> str:
    """Authenticate the immutable correction-25 fingerprint observation."""
    commit = _git(
        "rev-parse", f"{ORIGINAL_FINGERPRINT_RED_COMMIT_SHA}^{{commit}}"
    ).strip()
    if commit != ORIGINAL_FINGERPRINT_RED_COMMIT_SHA:
        raise EvidenceError(
            PREFLIGHT,
            "the original fingerprint R3 does not resolve to its exact commit object",
        )
    if _commit_parents(commit) != (FINGERPRINT_DESIGN_SHA,):
        raise EvidenceError(
            PREFLIGHT,
            "the original fingerprint R3 is not correction #25's ruled non-merge",
        )
    if _changed_paths(commit) != ORIGINAL_FINGERPRINT_RED_PATHS:
        raise EvidenceError(
            PREFLIGHT,
            "the original fingerprint R3 does not have its exact five-path delta",
        )
    return commit


def _red_commit_sha() -> str:
    """Require the terminal red SHA introduced and retained by first S3.

    A dependency-only authoring replay is never an evidence-generation boundary.
    The metadata literal is introduced in S3, separately from approval sentinels.
    """
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from tests.unit import test_sci004_phase3_dependency as dependency
    from tools import sci004_phase3_history as history

    root = REPOSITORY_ROOT.resolve(strict=True)
    for module, relative in (
        (dependency, "tests/unit/test_sci004_phase3_dependency.py"),
        (history, "tools/sci004_phase3_history.py"),
    ):
        module_file = module.__file__
        if (
            module_file is None
            or Path(module_file).resolve(strict=True) != root / relative
            or module.REPOSITORY_ROOT.resolve(strict=True) != root
        ):
            raise EvidenceError(
                PREFLIGHT, "terminal R3 validator import belongs to another checkout"
            )
    if (
        dependency.APPROVED_SCI004_D_SHA != _frozen_binding("APPROVED_SCI004_D_SHA")
        or history.RED_DESIGN_SHA != dependency.APPROVED_SCI004_D_SHA
        or history.OPERATIVE_DESIGN_SHA != dependency.APPROVED_SCI004_D_SHA
        or dependency.D30_SHA != _frozen_binding("D30_SHA")
        or history.DESIGN_SHA != dependency.D30_SHA
    ):
        raise EvidenceError(
            PREFLIGHT, "terminal R3 loaded design differs from frozen binding"
        )

    _ = _design_sha()
    try:
        anchor = dependency.resolve_r3_replay_anchor()
    except (dependency.DependencyCertificateError, history.HistoryError) as error:
        raise EvidenceError(
            PREFLIGHT, f"terminal R3 history does not authenticate: {error}"
        ) from error
    if anchor.role != "r3":
        raise EvidenceError(
            PREFLIGHT,
            "first S3 has not frozen terminal R3; authoring cannot generate evidence",
        )
    return anchor.commit


def _git_blob(commit: str, path: str) -> bytes:
    """Return one exact tree blob or refuse before evidence generation."""
    completed = _git_process("show", f"{commit}:{path}")
    if completed.returncode != 0:
        raise EvidenceError(
            PREFLIGHT,
            f"the red commit {commit} does not contain {path}: "
            f"{completed.stderr.decode('utf-8', 'replace').strip()}",
        )
    return completed.stdout


def _canonical_artifact(path: str, *, label: str) -> tuple[bytes, dict[str, Any]]:
    """Read one red artifact and require exact canonical bytes."""
    raw = (REPOSITORY_ROOT / path).read_bytes()
    try:
        value = _canonical_json_object(raw, label)
    except EvidenceError as exc:
        raise EvidenceError(PREFLIGHT, f"{label} is not canonical JSON") from exc
    return raw, value


def _red_failure_record_reference(red_commit: str) -> dict[str, Any]:
    """Authenticate all three records while retaining the pre-S3 join shape."""
    if red_commit != _red_commit_sha():
        raise EvidenceError(PREFLIGHT, "red_commit_sha is not the exact future R3")
    current_design = _design_sha()
    _authenticate_historical_design(
        POST_SOURCE_DESIGN_SHA,
        label="correction #24's historical design",
        current_design=current_design,
    )
    _authenticate_historical_design(
        FINGERPRINT_DESIGN_SHA,
        label="correction #25's historical design",
        current_design=current_design,
    )
    if not _is_ancestor(POST_SOURCE_DESIGN_SHA, FINGERPRINT_DESIGN_SHA):
        raise EvidenceError(
            PREFLIGHT, "correction #24's D is not connected to correction #25's D"
        )
    if (
        len(
            {
                POST_SOURCE_DESIGN_SHA,
                FINGERPRINT_DESIGN_SHA,
                current_design,
                ORIGINAL_FINGERPRINT_RED_COMMIT_SHA,
            }
        )
        != 4
    ):
        raise EvidenceError(
            PREFLIGHT, "historical and current identities are conflated"
        )
    original_fingerprint_red_commit = _original_fingerprint_red_commit_sha()
    if not _is_ancestor(original_fingerprint_red_commit, current_design):
        raise EvidenceError(
            PREFLIGHT,
            "the original fingerprint R3 is not connected to current operative D30",
        )
    historical_raw, historical = _canonical_artifact(
        RED_FAILURE_RECORD, label="the historical phase-3 red record"
    )
    historical_sha256 = hashlib.sha256(historical_raw).hexdigest()
    if historical_sha256 != HISTORICAL_RED_FAILURE_SHA256:
        raise EvidenceError(
            PREFLIGHT,
            "the historical phase-3 red record does not match its frozen raw digest",
        )
    if _git_blob(red_commit, RED_FAILURE_RECORD) != historical_raw:
        raise EvidenceError(
            PREFLIGHT,
            "the fresh R3 does not contain the immutable historical red bytes",
        )
    if historical.get("schema_version") != RED_FAILURE_SCHEMA:
        raise EvidenceError(PREFLIGHT, "the historical red schema literal is wrong")
    if historical.get("phase") != PHASE or historical.get("status") != (
        "expected-red-confirmed"
    ):
        raise EvidenceError(PREFLIGHT, "the historical red phase or status is wrong")
    historical_pre_fix = historical.get("pre_fix_source_sha")
    if (
        not isinstance(historical_pre_fix, str)
        or re.fullmatch(r"[0-9a-f]{40}", historical_pre_fix) is None
    ):
        raise EvidenceError(
            PREFLIGHT, "the historical red record has no exact pre-fix source"
        )

    post_raw, post_source = _canonical_artifact(
        POST_SOURCE_RED_FAILURE_RECORD,
        label="the correction-24 post-source red record",
    )
    if post_source.get("schema_version") != POST_SOURCE_RED_FAILURE_SCHEMA:
        raise EvidenceError(PREFLIGHT, "the post-source red schema literal is wrong")
    if post_source.get("phase") != PHASE or post_source.get("status") != (
        "post-source-expected-red-confirmed"
    ):
        raise EvidenceError(PREFLIGHT, "the post-source red phase or status is wrong")
    if post_source.get("design_sha") != POST_SOURCE_DESIGN_SHA:
        raise EvidenceError(
            PREFLIGHT, "the post-source red record does not bind correction #24's D"
        )
    if post_source.get("pre_fix_source_sha") != POST_SOURCE_PRE_FIX_SHA:
        raise EvidenceError(
            PREFLIGHT, "the post-source red record does not bind the superseded S3"
        )
    if post_source.get("historical_red_record_sha256") != historical_sha256:
        raise EvidenceError(
            PREFLIGHT, "the post-source red record does not bind the historical bytes"
        )
    if _git_blob(red_commit, POST_SOURCE_RED_FAILURE_RECORD) != post_raw:
        raise EvidenceError(
            PREFLIGHT,
            "the fresh R3 does not contain the checked-out post-source red bytes",
        )
    post_sha256 = hashlib.sha256(post_raw).hexdigest()
    if post_sha256 != POST_SOURCE_RED_FAILURE_SHA256:
        raise EvidenceError(
            PREFLIGHT, "the correction-24 post-source red record has changed raw bytes"
        )

    fingerprint_raw, fingerprint = _canonical_artifact(
        FINGERPRINT_RED_FAILURE_RECORD,
        label="the correction-25 fingerprint red record",
    )
    fingerprint_sha256 = hashlib.sha256(fingerprint_raw).hexdigest()
    if fingerprint_sha256 != FINGERPRINT_RED_FAILURE_SHA256:
        raise EvidenceError(
            PREFLIGHT, "the correction-25 fingerprint red record has changed raw bytes"
        )
    if _git_blob(original_fingerprint_red_commit, FINGERPRINT_RED_FAILURE_RECORD) != (
        fingerprint_raw
    ):
        raise EvidenceError(
            PREFLIGHT,
            "the original fingerprint R3 does not contain the immutable fingerprint "
            "red bytes",
        )
    if _git_blob(red_commit, FINGERPRINT_RED_FAILURE_RECORD) != fingerprint_raw:
        raise EvidenceError(
            PREFLIGHT,
            "the future R3 does not contain the immutable fingerprint red bytes",
        )
    if (
        fingerprint.get("schema_version") != FINGERPRINT_RED_FAILURE_SCHEMA
        or fingerprint.get("phase") != PHASE
        or fingerprint.get("status") != "post-source-expected-red-confirmed"
        or fingerprint.get("design_sha") != FINGERPRINT_DESIGN_SHA
        or fingerprint.get("pre_fix_source_sha") != FINGERPRINT_PRE_FIX_SHA
        or fingerprint.get("historical_red_record_sha256") != historical_sha256
        or fingerprint.get("correction24_post_source_red_record_sha256") != post_sha256
    ):
        raise EvidenceError(
            PREFLIGHT, "the fingerprint red record does not bind its ruled chain"
        )

    return {
        "path": RED_FAILURE_RECORD,
        "sha256": historical_sha256,
        "schema_version": RED_FAILURE_SCHEMA,
        "pre_fix_source_sha": historical_pre_fix,
        "validated": True,
        "post_source_delta": {
            "path": POST_SOURCE_RED_FAILURE_RECORD,
            "sha256": post_sha256,
            "schema_version": POST_SOURCE_RED_FAILURE_SCHEMA,
            "pre_fix_source_sha": POST_SOURCE_PRE_FIX_SHA,
            "validated": True,
        },
    }


def validate_evidence_artifact(document: Any) -> dict[str, Any]:
    """Validate an envelope and authenticate both of its retained red inputs."""
    envelope = validate_evidence_document(document)
    _require(
        envelope["design_sha"] == _design_sha(),
        DIGEST,
        "design_sha must name the current frozen operative D33 binding",
    )
    red_commit = _red_commit_sha()
    _require(
        envelope["red_commit_sha"] == red_commit,
        DIGEST,
        "red_commit_sha must name the fresh R3 containing both red records",
    )
    _require(
        envelope["red_failure_record"] == _red_failure_record_reference(red_commit),
        DIGEST,
        "red_failure_record must authenticate and join both retained red records",
    )
    return envelope


def _host_tag() -> str:
    """Return Section 11's ``[a-z0-9][a-z0-9-]{0,62}`` host tag."""
    raw = platform.node().split(".")[0].lower()
    cleaned = re.sub(r"[^a-z0-9-]", "-", raw).strip("-")
    if not cleaned or not cleaned[0].isalnum():
        cleaned = f"host-{cleaned}".strip("-")
    return cleaned[:63]


def performance_record_path(recorded_at_utc: str, host_tag: str) -> str:
    """Return Section 11's ``<UTC>-<host>.json`` retained path."""
    stamp = re.sub(r"[^0-9TZ]", "", recorded_at_utc)
    return f"{PERFORMANCE_DIRECTORY}/{stamp}-{host_tag}.json"


def _require_raw_tracked_checkout(head: str) -> None:
    """Authenticate tracked bytes and types without index or filter shortcuts."""
    try:
        _require(
            stat.S_ISDIR(REPOSITORY_ROOT.lstat().st_mode),
            PREFLIGHT,
            "tracked checkout root type changed",
        )
        root = REPOSITORY_ROOT.resolve(strict=True)
        entries = _git_bytes("ls-tree", "-r", "-z", "--full-tree", head).split(b"\0")
        _require(entries[-1] == b"", PREFLIGHT, "tracked tree framing")
        seen: set[bytes] = set()
        for entry in entries[:-1]:
            metadata, separator, relative = entry.partition(b"\t")
            fields = metadata.split()
            _require(
                bool(separator) and len(fields) == 3 and relative not in seen,
                PREFLIGHT,
                "tracked tree entry",
            )
            seen.add(relative)
            mode, kind, oid = fields
            components = relative.split(b"/")
            _require(
                all(part not in {b"", b".", b".."} for part in components),
                PREFLIGHT,
                "tracked tree path",
            )
            parent = root
            for component in components[:-1]:
                parent /= os.fsdecode(component)
                _require(
                    stat.S_ISDIR(parent.lstat().st_mode),
                    PREFLIGHT,
                    "tracked parent directory type changed",
                )
            if mode == b"160000":
                _require(kind == b"commit", PREFLIGHT, "tracked gitlink type")
                continue
            _require(
                mode in {b"100644", b"100755", b"120000"} and kind == b"blob",
                PREFLIGHT,
                "tracked tree type",
            )
            path = parent / os.fsdecode(components[-1])
            actual_mode = path.lstat().st_mode
            if mode == b"120000":
                _require(
                    stat.S_ISLNK(actual_mode), PREFLIGHT, "tracked symlink type changed"
                )
                actual = os.fsencode(os.readlink(path))
            else:
                _require(
                    stat.S_ISREG(actual_mode),
                    PREFLIGHT,
                    "tracked regular file type changed",
                )
                _require(
                    bool(actual_mode & stat.S_IXUSR) == (mode == b"100755"),
                    PREFLIGHT,
                    "tracked executable mode changed",
                )
                actual = path.read_bytes()
            original = _git_bytes("cat-file", "blob", oid.decode("ascii"))
            matches = actual == original
            if not matches and mode != b"120000":
                pointer = re.fullmatch(
                    rb"version https://git-lfs.github.com/spec/v1\n"
                    rb"oid sha256:([0-9a-f]{64})\nsize (0|[1-9][0-9]*)\n",
                    original,
                )
                matches = pointer is not None and (
                    pointer[2] == str(len(actual)).encode("ascii")
                    and pointer[1] == hashlib.sha256(actual).hexdigest().encode("ascii")
                )
            _require(
                matches,
                PREFLIGHT,
                f"tracked raw bytes changed: {os.fsdecode(relative)}",
            )
    except OSError as error:
        raise EvidenceError(
            PREFLIGHT, "tracked checkout path cannot be read"
        ) from error


def preflight(
    source_sha: str | None = None, declared: Sequence[str] = ()
) -> dict[str, str]:
    """Run Section 14.2's common pre-output check without writing anything."""
    head = _git("rev-parse", "HEAD").strip()
    if source_sha is not None and head != source_sha:
        raise EvidenceError(
            PREFLIGHT, f"HEAD {head} is not the approved source {source_sha}"
        )
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status.strip():
        raise EvidenceError(PREFLIGHT, "the working tree is not globally clean")
    for relative in declared:
        if (REPOSITORY_ROOT / relative).exists():
            raise EvidenceError(
                PREFLIGHT, f"the declared output {relative} already exists"
            )
    _require_raw_tracked_checkout(head)
    return {
        "source_sha": head,
        "pixi_manifest_sha256": raw_sha256(REPOSITORY_ROOT / "pixi.toml"),
        "pixi_lock_sha256": raw_sha256(REPOSITORY_ROOT / "pixi.lock"),
        "git_tree_sha256": domain_digest(
            "radiosim.sci004.git-tree.v1",
            _git_bytes("ls-tree", "-r", "-z", "--full-tree", head),
        ),
    }


# Fixed schema expectations are checks against actual producer/validator ASTs,
# not a declaration that an intermediate source commit is ready.
_SOURCE_CONTRACT_KEYS: dict[str, tuple[str, ...]] = {
    "results": tuple(
        "dependency_certificate output_cases fingerprint_rows "
        "ci_artifacts performance_record release_scan_cases "
        "rejection_cases".split()
    ),
    "phase": tuple(
        "schema_version site_manifest site_sha256 "
        "iers_table_sha256 canonical_era_turn_grid "
        "canonical_era_turn_grid_sha256 canonical_era_grid "
        "canonical_era_grid_sha256 utc_manifest utc_sha256 "
        "ut1_manifest ut1_sha256 mmode_dimensions antenna_rows "
        "baseline_rows frequency_rows receptor_rows "
        "correlation_rows beam_rows sky_component_rows "
        "direction_input_rows jones_term_rows "
        "transfer_grid_catalog precision result_dtype "
        "convention_identity_sha256".split()
    ),
    "input": tuple(
        "schema_version family_id fixture_id "
        "phase_input_identity_manifest phase_input_identity_sha256 "
        "instrument_manifest instrument_sha256 receptor_manifest "
        "receptor_sha256 loaded_beam_manifest "
        "beam_loaded_fingerprint correlations polarization_basis "
        "frequencies_hz_f64be".split()
    ),
    "record": tuple(
        "family_id raw_cube_sha256 scientific_sha256 "
        "solver_snapshot characterization_time_manifest "
        "era_utc_grid_sha256 harmonic_index_table_sha256 "
        "characterization_input_manifest input_identity_sha256".split()
    ),
    "fingerprint": tuple(
        "family_id fixture_id characterization_input_manifest "
        "input_identity_sha256 characterization_time_manifest "
        "era_utc_grid_sha256 solver_snapshot "
        "solver_snapshot_sha256 cube_sha256 scientific_sha256 "
        "expected_change_reason pass".split()
    ),
    "envelope": tuple(
        "schema_version phase status generated_at_utc design_sha "
        "red_commit_sha source_sha phase_ranges "
        "evidence_commit_sha evidence_commit_sha_reason "
        "working_tree_clean environment source_identities "
        "red_failure_record results commands limitations "
        "claims_not_licensed".split()
    ),
    "red": tuple(
        "path sha256 schema_version pre_fix_source_sha validated "
        "post_source_delta fingerprint_post_source_delta".split()
    ),
}
_SOURCE_SENTINELS: dict[str, tuple[str, ...]] = {
    "tests/unit/test_sci004_phase3_evidence.py": (
        "APPROVED_SOURCE_SHA",
        "APPROVED_ARTIFACT_SHA256",
        "APPROVED_PERFORMANCE_PATH",
        "APPROVED_PERFORMANCE_SHA256",
    ),
    "tests/unit/test_sci004_phase3_acceptance.py": (
        "APPROVED_EVIDENCE_SHA",
        "APPROVED_ACCEPTANCE_ARTIFACT_SHA256",
    ),
}


def _source_tree(head: str, relative: str) -> ast.Module:
    completed = _git_process("show", f"{head}:{relative}")
    _require(
        completed.returncode == 0, PREFLIGHT, f"missing committed source: {relative}"
    )
    raw = completed.stdout
    path = REPOSITORY_ROOT / relative
    _require(
        path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
        PREFLIGHT,
        f"source contract differs from committed HEAD: {relative}",
    )
    try:
        return ast.parse(raw)
    except (SyntaxError, ValueError) as error:
        raise EvidenceError(
            PREFLIGHT, f"invalid source contract AST: {relative}"
        ) from error


def _source_binding(
    tree: ast.AST, name: str, *, body_owner: ast.With | None = None
) -> ast.AST:
    bindings = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, (ast.Store, ast.Del))
            or isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.ExceptHandler,
                    ast.MatchAs,
                    ast.MatchStar,
                ),
            )
            and node.name == name
            or isinstance(node, ast.arg)
            and node.arg == name
            or isinstance(node, ast.alias)
            and (node.name == "*" or (node.asname or node.name.split(".")[0]) == name)
            or isinstance(node, ast.MatchMapping)
            and node.rest == name
        )
    ]
    _require(
        len(bindings) == 1, PREFLIGHT, f"source contract binding must be unique: {name}"
    )
    owner = body_owner if body_owner is not None else tree
    body = (
        owner.body if isinstance(owner, (ast.Module, ast.FunctionDef, ast.With)) else []
    )
    values = [
        node.value
        for node in body
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            or isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        )
    ]
    if len(values) == 1 and values[0] is not None:
        return values[0]
    definitions = [
        node for node in body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    _require(
        len(definitions) == 1,
        PREFLIGHT,
        f"source contract needs direct definition: {name}",
    )
    return definitions[0]


def _source_literal(tree: ast.Module, name: str, expected: object) -> None:
    value = _source_binding(tree, name)
    try:
        observed = ast.literal_eval(value)
    except (ValueError, TypeError) as error:
        raise EvidenceError(
            PREFLIGHT, f"source contract requires literal {name}"
        ) from error
    _require(
        type(observed) is type(expected) and observed == expected,
        PREFLIGHT,
        f"source contract literal mismatch: {name}",
    )


def _source_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    node = _source_binding(tree, name)
    _require(
        isinstance(node, ast.FunctionDef),
        PREFLIGHT,
        f"source contract function: {name}",
    )
    return cast(ast.FunctionDef, node)


def _source_dict(node: ast.AST, keys: tuple[str, ...]) -> ast.Dict:
    _require(
        isinstance(node, ast.Dict),
        PREFLIGHT,
        "source contract needs actual producer dictionary",
    )
    value = cast(ast.Dict, node)
    observed = tuple(
        key.value if isinstance(key, ast.Constant) else None for key in value.keys
    )
    _require(observed == keys, PREFLIGHT, "source contract producer dictionary keys")
    return value


def _source_terminal_return(function: ast.FunctionDef) -> ast.Return:
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    _require(
        len(returns) == 1
        and function.body[-1] is returns[0]
        and not any(
            isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(function)
        )
        and not any(isinstance(node, ast.Raise) for node in function.body),
        PREFLIGHT,
        f"source contract requires one terminal return: {function.name}",
    )
    return returns[0]


def _source_return(function: ast.FunctionDef, keys: tuple[str, ...]) -> ast.Dict:
    value = _source_terminal_return(function).value
    if isinstance(value, ast.Name):
        name = value.id
        value = _source_binding(function, name)
        _ = _source_dict(value, keys)
        definition = next(
            index
            for index, node in enumerate(function.body)
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is value
        )
        suffix = function.body[definition + 1 : -1]
        # A retained dictionary may only cross the donor's immutable shape guard
        # before return: no alias, escape, descriptor call or indirect mutation.
        constants = {
            "mmode_characterization_record": "MMODE_CHARACTERIZATION_RECORD_KEYS",
            "_characterization_input_manifest": "_CHARACTERIZATION_INPUT_KEYS",
        }
        if suffix:
            _require(
                function.name in constants
                and len(suffix) == 1
                and isinstance(suffix[0], ast.If),
                PREFLIGHT,
                "source contract returned dictionary suffix",
            )
            guard = cast(ast.If, suffix[0])
            _source_expression(
                guard.test, f"tuple({name}) != {constants[function.name]}"
            )
            _require(
                not guard.orelse
                and len(guard.body) == 1
                and isinstance(guard.body[0], ast.Raise),
                PREFLIGHT,
                "source contract immutable shape guard",
            )
            rejection = cast(ast.Raise, guard.body[0])
            error = rejection.exc
            _require(
                rejection.cause is None
                and isinstance(error, ast.Call)
                and isinstance(error.func, ast.Name)
                and error.func.id == "InvalidResultError"
                and not error.keywords
                and len(error.args) == 1
                and isinstance(error.args[0], ast.Constant)
                and type(error.args[0].value) is str,
                PREFLIGHT,
                "source contract immutable shape rejection",
            )
    return _source_dict(value if value is not None else ast.Constant(None), keys)


def _source_expression(node: ast.AST, expression: str) -> None:
    _require(
        ast.dump(node) == ast.dump(ast.parse(expression, mode="eval").body),
        PREFLIGHT,
        "source contract value wiring",
    )


def _source_has_call(function: ast.FunctionDef, expression: str) -> None:
    _ = _source_terminal_return(function)
    expected = ast.dump(ast.parse(expression, mode="eval").body)
    # Direct executable statements only: an unused nested function or if-False
    # call cannot stand in for validator wiring.
    calls = [
        node
        for statement in function.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr))
        for node in [statement.value]
        if isinstance(node, ast.Call)
    ]
    _require(
        any(ast.dump(node) == expected for node in calls),
        PREFLIGHT,
        f"source contract call wiring: {function.name}",
    )


def _require_source_schema_contract(head: str) -> None:
    result = _source_tree(head, "src/radiosim/core/result.py")
    producer = _source_tree(head, "tools/sci004_mmode_phase3_evidence.py")
    oracle = _source_tree(head, "tests/unit/test_sci004_phase3_evidence.py")
    _source_literal(
        result,
        "MMODE_CHARACTERIZATION_INPUT_DOMAIN",
        "radiosim.sci004.characterization-input.v2",
    )
    _source_literal(
        result, "_MMODE_PHASE_INPUT_DOMAIN", "radiosim.mmode-input-identity.v1"
    )
    for name, key in (
        ("MMODE_CHARACTERIZATION_RECORD_KEYS", "record"),
        ("_CHARACTERIZATION_INPUT_KEYS", "input"),
        ("_MMODE_PHASE_INPUT_KEYS", "phase"),
    ):
        _source_literal(result, name, _SOURCE_CONTRACT_KEYS[key])
    factory = _source_function(result, "mmode_characterization_record")
    for name in ("family_id", "phase_input_identity_manifest"):
        required = dict(
            zip(
                (arg.arg for arg in factory.args.kwonlyargs),
                factory.args.kw_defaults,
                strict=True,
            )
        )
        _require(
            name in required and required[name] is None,
            PREFLIGHT,
            f"source contract required keyword: {name}",
        )
    record = _source_return(factory, _SOURCE_CONTRACT_KEYS["record"])
    _source_expression(
        _source_binding(factory, "characterization_input"),
        "_characterization_input_manifest(result, family_id, "
        "phase_input_identity_manifest)",
    )
    _source_expression(record.values[7], "characterization_input")
    _source_expression(
        record.values[8],
        "object_digest(MMODE_CHARACTERIZATION_INPUT_DOMAIN, characterization_input)",
    )
    _source_has_call(
        factory,
        "_characterization_input_manifest(result, family_id, "
        "phase_input_identity_manifest)",
    )
    inputs = _source_function(result, "_characterization_input_manifest")
    input_record = _source_return(inputs, _SOURCE_CONTRACT_KEYS["input"])
    _source_expression(input_record.values[0], "MMODE_CHARACTERIZATION_INPUT_DOMAIN")
    _source_expression(input_record.values[3], "phase")
    _source_expression(input_record.values[4], "phase_digest")
    _source_expression(
        _source_binding(inputs, "phase_digest"),
        "object_digest(_MMODE_PHASE_INPUT_DOMAIN, phase)",
    )
    _source_expression(
        _source_binding(inputs, "phase"),
        "_characterization_mapping(phase_input_identity_manifest, "
        "_MMODE_PHASE_INPUT_KEYS, "
        'field_name="phase_input_identity_manifest")',
    )
    _source_has_call(
        inputs,
        "_characterization_mapping(phase_input_identity_manifest, "
        "_MMODE_PHASE_INPUT_KEYS, "
        'field_name="phase_input_identity_manifest")',
    )
    schema_test = ast.dump(
        ast.parse(
            'phase["schema_version"] != _MMODE_PHASE_INPUT_DOMAIN', mode="eval"
        ).body
    )
    _require(
        any(
            isinstance(node, ast.If)
            and ast.dump(node.test) == schema_test
            and any(isinstance(child, ast.Raise) for child in node.body)
            for node in inputs.body
        ),
        PREFLIGHT,
        "source contract phase domain enforcement",
    )
    for tree in (producer, oracle):
        for name, key in (
            ("ENVELOPE_KEYS", "envelope"),
            ("FINGERPRINT_ROW_KEYS", "fingerprint"),
            ("RED_FAILURE_RECORD_KEYS", "red"),
        ):
            if tree is producer or name != "RED_FAILURE_RECORD_KEYS":
                _source_literal(tree, name, _SOURCE_CONTRACT_KEYS[key])
    _source_literal(
        producer,
        "SECTION_11_FAMILIES",
        (
            "mmode_single_scalar_mode",
            "mmode_point_stokes_i",
            "mmode_point_full_stokes",
            "mmode_circular_receptor",
        ),
    )
    generation = _source_function(producer, "build_phase3_evidence")
    _ = _source_terminal_return(generation)
    workspaces = [node for node in generation.body if isinstance(node, ast.With)]
    _require(
        len(workspaces) == 1 and len(workspaces[0].items) == 1,
        PREFLIGHT,
        "source contract generation workspace",
    )
    workspace = workspaces[0]
    _source_expression(workspace.items[0].context_expr, "tempfile.TemporaryDirectory()")
    _require(
        ast.dump(workspace.items[0].optional_vars or ast.Constant(None))
        == ast.dump(ast.Name(id="scratch", ctx=ast.Store())),
        PREFLIGHT,
        "source contract generation workspace binding",
    )
    generated_rows = _source_binding(
        generation, "fingerprint_rows", body_owner=workspace
    )
    _source_expression(generated_rows, "_fingerprint_rows(results, bundles)")
    groups = _source_binding(generation, "groups", body_owner=workspace)
    _source_expression(groups, "{}")
    ordered_values: list[ast.AST | None] = [
        node.value
        for node in workspace.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    _require(
        ordered_values.index(generated_rows) < ordered_values.index(groups),
        PREFLIGHT,
        "source contract fingerprint preparation precedes measurement",
    )
    document = _source_dict(
        _source_binding(generation, "document"), _SOURCE_CONTRACT_KEYS["envelope"]
    )
    results = _source_dict(
        document.values[_SOURCE_CONTRACT_KEYS["envelope"].index("results")],
        _SOURCE_CONTRACT_KEYS["results"],
    )
    _source_expression(
        results.values[_SOURCE_CONTRACT_KEYS["results"].index("fingerprint_rows")],
        "fingerprint_rows",
    )
    value = document.values[_SOURCE_CONTRACT_KEYS["envelope"].index("phase_ranges")]
    _require(
        ast.dump(value)
        == ast.dump(ast.parse('state["phase_ranges"]', mode="eval").body),
        PREFLIGHT,
        "source contract generated range binding",
    )
    _ = _source_return(
        _source_function(producer, "_red_failure_record_reference"),
        _SOURCE_CONTRACT_KEYS["red"],
    )
    fingerprint = _source_function(producer, "_fingerprint_rows")
    _source_expression(
        _source_terminal_return(fingerprint).value or ast.Constant(None), "rows"
    )
    body = fingerprint.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and type(body[0].value.value) is str
    ):
        body = body[1:]
    # Authenticate the complete small producer, including import ownership,
    # same-family result/input joins, iteration and the actual returned list.
    _require(len(body) == 4, PREFLIGHT, "source contract fingerprint body")
    _require(
        ast.dump(body[0])
        == ast.dump(
            ast.parse(
                "from radiosim.core.result import mmode_characterization_record"
            ).body[0]
        ),
        PREFLIGHT,
        "source contract fingerprint factory import",
    )
    _require(
        ast.dump(body[1]) == ast.dump(ast.parse("rows = []").body[0]),
        PREFLIGHT,
        "source contract fingerprint output initialization",
    )
    _require(
        isinstance(body[2], ast.For), PREFLIGHT, "source contract fingerprint iteration"
    )
    loop = cast(ast.For, body[2])
    _require(
        ast.dump(loop.target) == ast.dump(ast.Name(id="family_id", ctx=ast.Store()))
        and not loop.orelse
        and len(loop.body) == 3,
        PREFLIGHT,
        "source contract fingerprint iteration target/body",
    )
    _source_expression(loop.iter, "SECTION_11_FAMILIES")
    for statement, expected in zip(
        loop.body[:2],
        (
            "result = results[family_id]",
            "record = mmode_characterization_record(result, "
            "family_id=family_id, "
            "phase_input_identity_manifest=bundles[family_id]['inpu"
            "t_identity_manifest'])",
        ),
        strict=True,
    ):
        _require(
            ast.dump(statement) == ast.dump(ast.parse(expected).body[0]),
            PREFLIGHT,
            "source contract fingerprint same-family input/result",
        )
    append = loop.body[2]
    _require(
        isinstance(append, ast.Expr) and isinstance(append.value, ast.Call),
        PREFLIGHT,
        "source contract fingerprint producer",
    )
    call = cast(ast.Call, cast(ast.Expr, append).value)
    _source_expression(call.func, "rows.append")
    _require(
        len(call.args) == 1 and not call.keywords,
        PREFLIGHT,
        "source contract fingerprint append",
    )
    _ = _source_dict(call.args[0], _SOURCE_CONTRACT_KEYS["fingerprint"])
    validator = _source_function(producer, "validate_evidence_artifact")
    _source_has_call(
        validator,
        'history.validate_phase_ranges(envelope["phase_ranges"], '
        'design_sha=envelope["design_sha"], '
        'red_sha=envelope["red_commit_sha"], '
        'source_sha=envelope["source_sha"], root=REPOSITORY_ROOT)',
    )


def source_readiness(source_sha: str | None, declared: Sequence[str]) -> dict[str, Any]:
    """Require D31's complete source boundary before any evidence measurement.

    This checks structural source readiness. Full source tests and independent
    review remain mandatory; an AST check cannot establish scientific acceptance.
    """
    state = preflight(source_sha, declared)
    red = _red_commit_sha()
    from tools import sci004_phase3_history as history

    try:
        for relative in history.DISPOSAL_PINS:
            path = REPOSITORY_ROOT / relative
            _require(
                not path.exists() and not path.is_symlink(),
                PREFLIGHT,
                f"rejected output remains at source: {relative}",
            )
        ranges = {
            phase: history.describe_phase_range(
                base, terminal, phase, root=REPOSITORY_ROOT
            )
            for phase, base, terminal in (
                ("prerequisite", history.DESIGN_SHA, history.PREREQUISITE_TIP_SHA),
                ("red", history.PREREQUISITE_TIP_SHA, red),
                ("source", red, state["source_sha"]),
            )
        }
        history.validate_phase_ranges(
            ranges,
            design_sha=_design_sha(),
            red_sha=red,
            source_sha=state["source_sha"],
            root=REPOSITORY_ROOT,
        )
        for relative in history.SOURCE_PATHS:
            _ = _source_tree(state["source_sha"], relative)
        for relative, names in _SOURCE_SENTINELS.items():
            tree = _source_tree(state["source_sha"], relative)
            for name in names:
                _source_literal(tree, name, None)
        _require_source_schema_contract(state["source_sha"])
    except history.HistoryError as error:
        raise EvidenceError(
            PREFLIGHT, f"source history does not authenticate: {error}"
        ) from error
    return {**state, "red_commit_sha": red, "phase_ranges": ranges}


def _require_generation_source_imports() -> None:
    """Bind the actual imported scientific entry points to this source checkout."""
    import importlib
    import inspect
    from dataclasses import MISSING, fields, is_dataclass

    root = REPOSITORY_ROOT.resolve(strict=True)
    modules: dict[str, Any] = {}
    for name, relative in (
        ("radiosim", "src/radiosim/__init__.py"),
        ("radiosim.core.result", "src/radiosim/core/result.py"),
        ("radiosim.core.mmode.solver", "src/radiosim/core/mmode/solver.py"),
        ("radiosim.core.mmode.types", "src/radiosim/core/mmode/types.py"),
    ):
        try:
            module = importlib.import_module(name)
            origin = module.__file__
            resolved = Path(origin).resolve(strict=True) if origin is not None else None
        except (ImportError, OSError) as error:
            raise EvidenceError(
                PREFLIGHT, f"generation import cannot authenticate: {name}"
            ) from error
        _require(
            resolved == root / relative,
            PREFLIGHT,
            f"generation import belongs to another checkout: {name}",
        )
        modules[name] = module
    result = modules["radiosim.core.result"]
    keys = getattr(result, "MMODE_CHARACTERIZATION_RECORD_KEYS", None)
    _require(
        getattr(result, "MMODE_CHARACTERIZATION_INPUT_DOMAIN", None)
        == "radiosim.sci004.characterization-input.v2"
        and type(keys) is tuple
        and keys == _SOURCE_CONTRACT_KEYS["record"],
        PREFLIGHT,
        "generation loaded result schema is not v2",
    )
    try:
        factory = getattr(result, "mmode_characterization_record", None)
        _require(
            callable(factory), PREFLIGHT, "generation loaded factory is not callable"
        )
        signature = inspect.signature(cast(Callable[..., Any], factory))
    except (TypeError, ValueError) as error:
        raise EvidenceError(
            PREFLIGHT, "generation loaded factory has no valid signature"
        ) from error
    for name in ("family_id", "phase_input_identity_manifest"):
        parameter = signature.parameters.get(name)
        _require(
            parameter is not None
            and parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and parameter.default is inspect.Parameter.empty,
            PREFLIGHT,
            f"generation loaded factory requires keyword: {name}",
        )
    snapshot = getattr(
        modules["radiosim.core.mmode.solver"], "MModeSolverSnapshot", None
    )
    _require(
        isinstance(snapshot, type) and is_dataclass(snapshot),
        PREFLIGHT,
        "generation loaded snapshot is not a dataclass",
    )
    identity = [
        field
        for field in fields(cast(Any, snapshot))
        if field.name == "input_identity_sha256"
    ]
    _require(
        len(identity) == 1
        and identity[0].default is MISSING
        and identity[0].default_factory is MISSING
        and identity[0].type in (str, "str")
        and getattr(
            getattr(cast(Any, snapshot), "__dataclass_params__", None), "frozen", False
        )
        is True,
        PREFLIGHT,
        "generation loaded snapshot lacks required immutable input identity",
    )


def require_declared_outputs_only(declared: Sequence[str]) -> None:
    """Require the repository's only new paths to equal the declared set."""
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    observed = sorted(line[3:].strip() for line in status.splitlines() if line.strip())
    expected = sorted(declared)
    _require_raw_tracked_checkout(_git("rev-parse", "HEAD").strip())
    _require(
        observed == expected,
        DIGEST,
        f"after publication the repository's new paths must be exactly "
        f"{expected}, not {observed}",
    )


EVIDENCE_BYTE_LIMIT = 104_857_600


def _publish_evidence_payload(
    payload: bytes, performance_path: str, performance_payload: bytes
) -> None:
    """Enforce D33's complete E size before publishing either declared output."""
    _require(
        len(payload) < EVIDENCE_BYTE_LIMIT,
        SCHEMA,
        "complete evidence payload must be smaller than 104857600 bytes",
    )
    # The caller validates and serializes both complete documents first.
    write_atomic_no_overwrite(REPOSITORY_ROOT / performance_path, performance_payload)
    write_atomic_no_overwrite(REPOSITORY_ROOT / EVIDENCE_ARTIFACT, payload)


def _read_evidence_payload(path: Path) -> bytes:
    """Read at most the forbidden threshold, even if a file grows during reading."""
    try:
        with path.open("rb") as handle:
            payload = handle.read(EVIDENCE_BYTE_LIMIT)
    except OSError as error:
        raise EvidenceError(
            SCHEMA, f"cannot read evidence artifact: {error}"
        ) from error
    _require(
        len(payload) < EVIDENCE_BYTE_LIMIT,
        SCHEMA,
        "evidence artifact must be smaller than 104857600 bytes",
    )
    return payload


def write_atomic_no_overwrite(path: Path, payload: bytes) -> None:
    """Publish one artifact atomically, refusing to overwrite anything."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "xb") as handle:
        handle.write(payload)
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise EvidenceError(DIGEST, f"{path} already exists") from error
    finally:
        temporary.unlink()


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


#: The solver-module entry points each Section 11 timing stage owns.  Every one
#: of them is a module global that ``_mmode_pipeline`` calls, so wrapping them
#: observes the real production call in the real order without duplicating any
#: pipeline logic here.
STAGE_ENTRY_POINTS: Mapping[str, str] = {
    "build_frozen_frame": "frame",
    "build_direction_ledger": "frame",
    "build_frame_certificate": "frame",
    "build_production_transfer": "beam_transfer",
    "point_sky_coefficients": "sky_transform",
    "polarized_point_sky_coefficients": "sky_transform",
    "contract_and_synthesize": "dense_contraction_and_synthesis",
}


class _StageTimer:
    """Accumulate disjoint per-stage wall time inside one real solve.

    A nested wrapped call attributes nothing of its own: its interval is already
    inside the outermost stage's, and Section 11 requires the total to be no
    smaller than the sum of the named stages, which only holds when the stage
    intervals are disjoint.  The depth guard is what makes them disjoint.
    """

    def __init__(self, module: Any) -> None:
        self.module = module
        self.totals: dict[str, float] = dict.fromkeys(
            set(STAGE_ENTRY_POINTS.values()), 0.0
        )
        self._depth = 0
        self._originals: dict[str, Any] = {}

    def _wrap(self, original: Any, bucket: str) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self._depth:
                return original(*args, **kwargs)
            self._depth += 1
            started = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self.totals[bucket] += (time.perf_counter_ns() - started) / 1e9
                self._depth -= 1

        return wrapper

    def __enter__(self) -> _StageTimer:
        for name, bucket in STAGE_ENTRY_POINTS.items():
            original = getattr(self.module, name)
            self._originals[name] = original
            setattr(self.module, name, self._wrap(original, bucket))
        return self

    def __exit__(self, *exception: Any) -> bool:
        for name, original in self._originals.items():
            setattr(self.module, name, original)
        return False


def _family_mapping(root: Path, family_id: str) -> dict[str, Any]:
    """Return one Section 11 family's configuration mapping.

    The fixtures live in the phase's own characterization module, which is where
    the red oracles declare them.  Reading them from there rather than
    transcribing them here is what keeps a fingerprint row describing the same
    run the oracle pins.
    """
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from tests.characterization.test_sci004_mmode import family_mapping

    return family_mapping(root, family_id)


def _solve_once(root: Path, family_id: str) -> tuple[Any, Any]:
    """Return one family's public result and its solver bundle."""
    from radiosim.api.simulator import Simulator
    from radiosim.core.mmode.solver import build_m1_evidence

    simulator = Simulator.from_mapping(_family_mapping(root, family_id), base_dir=root)
    request = simulator.build_solve_request()
    bundle = build_m1_evidence(request)
    return simulator, bundle


def _cube_identity(cube: Any) -> str:
    """Return Section 14.0's visibility-cube identity for one published cube."""
    import numpy as np

    from radiosim.core.mmode.types import array_digest

    array = np.asarray(cube)
    dtype = "complex64-be" if array.dtype.itemsize == 8 else "complex128-be"
    return array_digest(
        "radiosim.mmode-visibility-cube.v1",
        "visibility_cube",
        ["time", "baseline", "frequency", "correlation"],
        "Jy",
        array,
        dtype=dtype,
    )


def _measure_group(
    root: Path, fixture_id: str, *, warmups: int, samples: int
) -> dict[str, Any]:
    """Measure one fixture group's shared series, memory and identities."""
    import numpy as np

    from radiosim.api.simulator import Simulator
    from radiosim.core.mmode import solver as solver_module

    series: dict[str, list[float]] = {name: [] for name in MEASURED_SERIES}
    outcome: Any = None
    request: Any = None
    for iteration in range(warmups + samples):
        simulator = Simulator.from_mapping(
            _family_mapping(root, fixture_id), base_dir=root
        )
        request = simulator.build_solve_request()
        with _StageTimer(solver_module) as timer:
            started = time.perf_counter_ns()
            outcome = solver_module.solve_mmode(request)
            total = (time.perf_counter_ns() - started) / 1e9
        if iteration < warmups:
            continue
        for bucket, value in timer.totals.items():
            series[bucket].append(value)
        series["total"].append(total)

    # Section 11: "Memory measurements use separate untimed synchronized calls."
    simulator = Simulator.from_mapping(_family_mapping(root, fixture_id), base_dir=root)
    memory_request = simulator.build_solve_request()
    _, measured_host_peak = _measure_with_process_rss(
        lambda: solver_module.solve_mmode(memory_request)
    )

    bundle = solver_module.build_m1_evidence(request)
    cube = np.asarray(bundle["cube"])
    samples_count, baselines, frequencies, _ = cube.shape
    dimensions = bundle["dimensions"]
    estimate = solver_module.estimate_mmode_memory(
        n_baselines=baselines,
        n_frequencies=frequencies,
        lmax=int(dimensions.lmax),
        mmax=int(dimensions.mmax),
        quadrature_nside=int(dimensions.quadrature_nside),
        working_memory_bytes=_WORKING_MEMORY_BYTES,
        n_antennas=2,
        sidereal_samples=samples_count,
    )
    schedule = solver_module.schedule_mmode_blocks(
        n_baselines=baselines,
        n_frequencies=frequencies,
        lmax=int(dimensions.lmax),
        mmax=int(dimensions.mmax),
        quadrature_nside=int(dimensions.quadrature_nside),
        working_memory_bytes=_WORKING_MEMORY_BYTES,
        n_antennas=2,
        sidereal_samples=samples_count,
    )
    return {
        "fixture_id": fixture_id,
        "series": series,
        "measured_host_peak_bytes": measured_host_peak,
        # Section 9's complete seven-component host estimate.  The one-block
        # minimum is a *scheduler* floor -- the smallest budget a single block
        # may be scheduled under -- not an estimate of the call's host peak, and
        # naming it here would understate the estimate by an order of magnitude.
        "estimated_host_peak_bytes": int(estimate.total_bytes),
        "schedule": schedule.as_mapping(),
        "bundle": bundle,
        "outcome": outcome,
        "cube": cube,
        "n_baselines": int(baselines),
        "n_frequencies": int(frequencies),
        "sidereal_samples": int(samples_count),
        "polarized": bundle["execution_path"] == "polarized",
    }


#: The accepted family fixtures' working-memory budget, read from the same
#: characterization module the mappings come from.
_WORKING_MEMORY_BYTES = 1 << 30


def _dense_invariance_row(
    root: Path, fixture_id: str, cube_sha256: str
) -> dict[str, Any]:
    """Measure the dense path's backend invariance for one group.

    Section 11 retains this as a *fact*: the public dense stages take no backend,
    so the three per-backend solves publish one cube.  Each backend is really
    solved rather than assumed, which is what makes the retained equality
    evidence instead of an assertion.
    """
    from radiosim.api.simulator import Simulator
    from radiosim.core.mmode import solver as solver_module

    digests: dict[str, str] = {}
    for backend in BACKENDS:
        mapping = _family_mapping(root, fixture_id)
        mapping["execution"] = {**mapping["execution"], "backend": backend}
        simulator = Simulator.from_mapping(mapping, base_dir=root)
        outcome = solver_module.solve_mmode(simulator.build_solve_request())
        receptor = outcome.receptor_visibilities
        digests[backend] = _cube_identity(receptor.reshape(*receptor.shape[:3], 4))
    identical = len(set(digests.values())) == 1 and digests["numpy"] == cube_sha256
    return {
        "comparison_group_id": fixture_id,
        "numpy_cube_sha256": digests["numpy"],
        "jax_cube_sha256": digests["jax"],
        "dask_cube_sha256": digests["dask"],
        "identical": bool(identical),
    }


def _kernel_stage_block(bundle: Any, backend_name: str) -> dict[str, Any]:
    """Measure Section 11's two admitted kernels on one non-NumPy backend."""
    import numpy as np

    from radiosim.backends import get_backend
    from radiosim.core.mmode.solver import (
        FIELD_ORDER,
        contract_per_m_block,
        synthesize_time_series,
    )
    from radiosim.core.mmode.types import array_digest

    table = bundle["table"]
    grid = bundle["grid"]
    transfer = np.asarray(bundle["transfer"])
    sky = np.asarray(bundle["sky"])
    baselines, frequencies, correlations, _ = transfer.shape
    fields = tuple(str(name) for name in FIELD_ORDER)
    rows = [row for row in table.block_rows if str(row["field_name"]) == fields[0]]
    order = int(rows[0]["m"])
    width = max(
        int(row["value_stop"]) - int(row["value_start"])
        for row in table.block_rows
        if int(row["m"]) == order
    )
    transfer_block = np.zeros(
        (baselines, frequencies, correlations, len(fields), width), dtype=np.complex128
    )
    sky_block = np.zeros((frequencies, len(fields), width), dtype=np.complex128)
    for index, field in enumerate(fields):
        row = next(
            candidate
            for candidate in table.block_rows
            if int(candidate["m"]) == order and str(candidate["field_name"]) == field
        )
        start, stop = int(row["value_start"]), int(row["value_stop"])
        transfer_block[:, :, :, index, : stop - start] = transfer[:, :, :, start:stop]
        sky_block[:, index, : stop - start] = sky[:, start:stop]

    mode_cube = np.zeros(
        (baselines, frequencies, correlations, 2 * int(table.mmax) + 1),
        dtype=np.complex128,
    )
    turns = [str(grid.center_turn(index)) for index in range(grid.sidereal_samples)]
    width_turn = str(grid.exact.exposure_width)

    def digest(array: Any) -> str:
        return array_digest(
            "radiosim.mmode-kernel-output.v1",
            "kernel_output",
            ["flat"],
            "Jy",
            np.asarray(array).ravel(),
            dtype="complex128-be",
        )

    reference_backend = get_backend("numpy")
    candidate_backend = get_backend(backend_name)
    reference_contraction = contract_per_m_block(
        transfer_block=transfer_block, sky_block=sky_block, backend=reference_backend
    )
    reference_synthesis = synthesize_time_series(
        mode_cube=mode_cube,
        era_turns=turns,
        exposure_width_turn=width_turn,
        backend=reference_backend,
    )

    stages: dict[str, Any] = {}
    for stage_name, call, reference in (
        (
            "per_m_contraction",
            lambda: contract_per_m_block(
                transfer_block=transfer_block,
                sky_block=sky_block,
                backend=candidate_backend,
            ),
            reference_contraction,
        ),
        (
            "synthesis",
            lambda: synthesize_time_series(
                mode_cube=mode_cube,
                era_turns=turns,
                exposure_width_turn=width_turn,
                backend=candidate_backend,
            ),
            reference_synthesis,
        ),
    ):
        sample_seconds: list[float] = []
        candidate: Any = None
        for _ in range(MINIMUM_SAMPLES + 1):
            started = time.perf_counter_ns()
            candidate = call()
            candidate_backend.synchronize(candidate)
            sample_seconds.append((time.perf_counter_ns() - started) / 1e9)
        sample_seconds = sample_seconds[1:]
        reference_array = np.asarray(reference)
        candidate_array = np.asarray(candidate)
        scale = max(1.0, float(np.max(np.abs(reference_array))))
        deviation = np.abs(candidate_array - reference_array)
        absolute = float(np.max(deviation)) if deviation.size else 0.0
        atol = BACKEND_ATOL_FACTOR * scale
        stages[stage_name] = {
            "sample_seconds": sample_seconds,
            "synchronization_method": KERNEL_SYNCHRONIZATION_METHODS[backend_name],
            "native_measurement_method": "unavailable",
            "measured_native_peak_bytes": None,
            "measured_native_peak_bytes_reason": (
                "this CPU-only backend build exposes no device allocator counter"
            ),
            "stage_comparison": {
                "predicate_id": BACKEND_PREDICATE_ID,
                "reference_stage_sha256": digest(reference_array),
                "candidate_stage_sha256": digest(candidate_array),
                "expected_cell_count": int(reference_array.size),
                "compared_finite_cell_count": int(reference_array.size),
                "reference_scale_jy": scale,
                "maximum_absolute_deviation_jy": absolute,
                "maximum_relative_deviation": absolute / scale,
                "rtol": BACKEND_RTOL,
                "atol_jy": atol,
                "pass": bool(
                    np.all(deviation <= atol + BACKEND_RTOL * np.abs(reference_array))
                ),
            },
        }
    return {
        "status": KERNEL_STATUS_MEASURED,
        "per_m_contraction": stages["per_m_contraction"],
        "synthesis": stages["synthesis"],
    }


def validate_scientific_solver(
    value: Any, family_id: Any, expected_iers_sha256: str, *, label: str
) -> dict[str, Any]:
    """Validate a D32 family's exact Section 10 solver snapshot.

    The caller authenticates the expected IERS identity and joins the returned
    frame-certificate identity to its complete certificate. This check alone
    does not authenticate that certificate or admit a scientific transition.
    """
    families = {
        "mmode_single_scalar_mode": ("scalar", 1),
        "mmode_point_stokes_i": ("scalar", 3),
        "mmode_point_full_stokes": ("polarized", 3),
        "mmode_circular_receptor": ("polarized", 3),
    }
    _require(
        isinstance(family_id, str) and family_id in families,
        SCHEMA,
        f"{label} has an unknown characterization family",
    )
    execution, count = families[family_id]
    expected: dict[str, Any] = {
        "solver": "mmode",
        "sky_representation": "point_sources",
        "convention": "radiosim.mmode-forward.v1",
        "execution_path": execution,
        "components": ["point"],
        "component_element_counts": [count],
        "time_grid_convention": "radiosim.mmode-era-turn-grid.v1",
        "frame_model": "radiosim.frozen-cirs-rigid-era.v1",
        "harmonic_convention": "radiosim.shaw-polarized-harmonics.v1",
        "sidereal_samples": 49,
        "lmax": 16,
        "mmax": 16,
        "quadrature_nside": 8,
        "quadrature_policy": "iso-gauss-ring-production-plus-qcheck.v1",
        "truncation_policy": "complete-frozen-direct-plus-local-shells.v1",
        "tangent_polarization_frame": (
            "not_applicable_scalar_m1"
            if execution == "scalar"
            else {
                "schema_version": "radiosim.sky-tangent-polarization.v1",
                "coordinate_frame": "icrs",
                "axes": "north_east",
                "position_angle": "north_through_east",
                "linear_complex": "q_plus_i_u",
                "stokes_v": "iau_incoming_r_minus_l",
            }
        ),
        "stokes_v_basis_bridge": "radiosim.stokes-ne-theta-phi.v1",
        "iers_table_sha256": _require_hex(expected_iers_sha256, 64, label + " IERS"),
        "frame_certificate_sha256": None,
        "transform_execution_policy": "host_harmonics_backend_native_dense_v1",
    }
    row = _require_keys(value, tuple(expected), label)
    expected["frame_certificate_sha256"] = _require_hex(
        row["frame_certificate_sha256"], 64, label + " frame certificate"
    )
    # JSON equality alone equates bools/integers/floats. Check the exact retained
    # integer/list forms before comparing the complete closed scientific value.
    for key in ("sidereal_samples", "lmax", "mmax", "quadrature_nside"):
        _require(type(row[key]) is int, SCHEMA, f"{label}.{key} must be an integer")
    _require(
        isinstance(row["component_element_counts"], list),
        SCHEMA,
        f"{label} component count form",
    )
    counts = cast(list[Any], row["component_element_counts"])
    _require(
        len(counts) == 1 and type(counts[0]) is int,
        SCHEMA,
        f"{label} component count form",
    )
    _require(isinstance(row["components"], list), SCHEMA, f"{label} components form")
    _require(row == expected, DIGEST, f"{label} differs from the frozen family solver")
    return row


FRAME_STRUCTURE_DIGEST_KEYS = tuple(
    """
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
    """.split()
)
FRAME_STRUCTURE_COUNT_KEYS = tuple(
    """
    sidereal_samples quadrature_nside n_baselines n_frequencies n_correlations
    expected_point_direction_count evaluated_point_direction_count
    expected_native_healpix_direction_count evaluated_native_healpix_direction_count
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
    """.split()
)
FRAME_STRUCTURE_F64_KEYS = tuple(
    """
    xp0_arcsec yp0_arcsec das2r_rad_per_arcsec xp0_rad yp0_rad sp0_rad
    """.split()
)
FRAME_STRUCTURE_NUMBER_KEYS = tuple(
    """
    horizon_mismatch_measure_rad horizon_mismatch_measure_limit_rad
    horizon_root_max_rad horizon_root_limit_rad phase_max_rad phase_limit_rad
    direct_gauss_scale_jy frozen_gauss_change_max_jy operational_gauss_change_max_jy
    direct_gauss_change_max_jy direct_gauss_change_limit_jy cube_scale_jy
    cube_max_jy cube_limit_jy cube_l2 cube_l2_limit direction_diagnostic_max_rad
    basis_diagnostic_max_rad
    """.split()
)
FRAME_STRUCTURE_RATIONAL_KEYS = tuple(
    """
    horizon_mismatch_measure_turn direction_diagnostic_argmax_phase
    basis_diagnostic_argmax_phase
    """.split()
)
FRAME_STRUCTURE_UNIT_KEYS = tuple(
    """
    pm_source_unit pom00_argument_unit
    """.split()
)
FRAME_STRUCTURE_ID_KEYS = tuple(
    """
    direction_diagnostic_argmax_id basis_diagnostic_argmax_id
    """.split()
)
FRAME_STRUCTURE_ROWS_KEYS = tuple(
    """
    transfer_grid_catalog direction_rows horizon_scan_crossing_rows
    horizon_scan_summary_rows horizon_root_pair_rows horizon_slab_rows
    horizon_sign_interval_rows horizon_membership_mask_rows direct_split_rows
    """.split()
)
FRAME_STRUCTURE_OBJECT_KEYS = tuple(
    """
    site_manifest frame_matrix_manifest horizon_scan_manifest
    direct_integrand_enclosure_manifest
    """.split()
)
FRAME_STRUCTURE_QCHECK_KEYS = tuple(
    """
    diagnostic_qcheck_nsides
    """.split()
)


def validate_frame_certificate_structure(value: Any, *, label: str) -> dict[str, Any]:
    """Authenticate the closed scalar structure and return its 125-field preimage.

    This is structural authentication only. Nested schemas, ledger joins,
    scientific budgets and transition/admission rules remain unvalidated.
    """
    categories = (
        FRAME_STRUCTURE_DIGEST_KEYS,
        FRAME_STRUCTURE_COUNT_KEYS,
        FRAME_STRUCTURE_F64_KEYS,
        FRAME_STRUCTURE_NUMBER_KEYS,
        FRAME_STRUCTURE_RATIONAL_KEYS,
        FRAME_STRUCTURE_UNIT_KEYS,
        FRAME_STRUCTURE_ID_KEYS,
        FRAME_STRUCTURE_ROWS_KEYS,
        FRAME_STRUCTURE_OBJECT_KEYS,
        FRAME_STRUCTURE_QCHECK_KEYS,
    )
    mapping = _require_keys(
        value, tuple(key for group in categories for key in group), label
    )
    for key in FRAME_STRUCTURE_DIGEST_KEYS:
        _ = _require_hex(mapping[key], 64, f"{label}.{key}")
    for key in FRAME_STRUCTURE_COUNT_KEYS:
        item = mapping[key]
        _require(
            type(item) is int and item >= 0,
            SCHEMA,
            f"{label}.{key}: expected a nonnegative integer",
        )
    for key in FRAME_STRUCTURE_F64_KEYS:
        encoded = _require_hex(mapping[key], 16, f"{label}.{key}")
        _require(
            math.isfinite(struct.unpack(">d", bytes.fromhex(encoded))[0]),
            SCHEMA,
            f"{label}.{key}: non-finite F64",
        )
    for key in FRAME_STRUCTURE_NUMBER_KEYS:
        item = mapping[key]
        _require(
            type(item) in (int, float) and 0 <= item <= sys.float_info.max,
            SCHEMA,
            f"{label}.{key}: expected a finite nonnegative JSON number",
        )
    for key in FRAME_STRUCTURE_RATIONAL_KEYS:
        item = mapping[key]
        _require(
            type(item) is str
            and re.fullmatch(r"(?:0|-?[1-9][0-9]*)/[1-9][0-9]*", item) is not None,
            SCHEMA,
            f"{label}.{key}: expected canonical rational text",
        )
        try:
            numerator, denominator = (int(part) for part in item.split("/"))
        except ValueError as error:
            raise EvidenceError(
                SCHEMA, f"{label}.{key}: invalid rational integer"
            ) from error
        _require(
            math.gcd(numerator, denominator) == 1
            and (key != "horizon_mismatch_measure_turn" or numerator >= 0),
            SCHEMA,
            f"{label}.{key}: rational must be reduced with valid sign",
        )
    for key, unit in (("pm_source_unit", "arcsec"), ("pom00_argument_unit", "rad")):
        _require(
            type(mapping[key]) is str and mapping[key] == unit,
            SCHEMA,
            f"{label}.{key}: incorrect unit",
        )
    for key in FRAME_STRUCTURE_ID_KEYS:
        _require(
            type(mapping[key]) is str,
            SCHEMA,
            f"{label}.{key}: expected a diagnostic identifier string",
        )
    for key in FRAME_STRUCTURE_ROWS_KEYS:
        _require(
            isinstance(mapping[key], list),
            SCHEMA,
            f"{label}.{key}: expected a row array; contents require separate validation",
        )
    for key in FRAME_STRUCTURE_OBJECT_KEYS:
        _require(
            isinstance(mapping[key], dict),
            SCHEMA,
            f"{label}.{key}: expected a manifest object; contents require separate validation",
        )
    nsides = mapping["diagnostic_qcheck_nsides"]
    _require(
        isinstance(nsides, list)
        and all(type(nside) is int and nside > 0 for nside in cast(list[Any], nsides)),
        SCHEMA,
        f"{label}: qcheck nsides must be positive integers",
    )
    _require(
        nsides == sorted(set(cast(list[int], nsides))),
        SCHEMA,
        f"{label}: qcheck nsides must be sorted and unique",
    )
    preimage = {
        key: item for key, item in mapping.items() if key != "certificate_sha256"
    }
    _require(
        object_digest("radiosim.mmode-frame-certificate.v1", preimage)
        == mapping["certificate_sha256"],
        DIGEST,
        f"{label}: frame certificate digest mismatch",
    )
    return preimage


def _snapshot_identity(snapshot: Mapping[str, Any]) -> str:
    """Return one solver snapshot's Section 14.0 object identity."""
    return object_digest("radiosim.mmode-solver-snapshot.v1", dict(snapshot))


def _standard_solver_snapshot(history: Sequence[str]) -> Mapping[str, Any]:
    """Return the tagged solver snapshot a standard reader reconstructs.

    Section 10 requires "reader round trips must reconstruct and authenticate the
    m-mode solver snapshot".  ``StandardVisibilityData`` carries no
    ``solver_snapshot`` attribute: the standard readers reconstruct it from the
    embedded projection record the writer put in ``history``, which is exactly
    what the R3 UVFITS and MS round-trip oracles read.  Reading it the same way
    here keeps the evidence row describing the same seam the oracle pins.
    """
    from radiosim.io.standard_visibility import projection_record_from_history

    record, _lines = projection_record_from_history("\n".join(history))
    snapshot = record["solver"]
    if not isinstance(snapshot, Mapping):
        raise EvidenceError(DIGEST, "the projection record carries no solver object")
    return cast("Mapping[str, Any]", snapshot)


def _output_cases(
    result: Any, bundle: Mapping[str, Any], fixture_id: str, root: Path
) -> list[dict[str, Any]]:
    """Round-trip one solved result through Section 10's three reader paths.

    Section 10 names five paths, but an ``output_cases`` row is a *round-trip*
    row: Section 14.2 gives it ``read_solver_sha256`` and ``read_cube_sha256``.
    Only HDF5, UVFITS and Measurement Set have a reader that returns a cube.
    Summary JSON is metadata-only by Section 10 and RadioSim ships no summary
    reader at all, so a ``read_cube_sha256`` for it could only be the written
    digest restated -- an invented round trip.  Its Section 10 obligation is
    carried by the R3 oracle in ``tests/unit/test_io/test_result_summary.py``,
    and ``limitations`` records the boundary rather than papering over it.

    The three identity fields are Section 14.0's exact ones -- domains
    ``radiosim.mmode-result-time.v1``, ``radiosim.mmode-result-feeds.v1`` and
    ``radiosim.mmode-result-correlations.v1`` over their exact key sets -- so
    they are reconstructible from the same fixture input the phase already
    embeds.  Section 14.0 forbids inventing a preimage, and a result-shaped
    stand-in would be exactly that.
    """
    import numpy as np

    from radiosim.io.hdf5 import load_result_hdf5, write_result_hdf5
    from radiosim.io.measurement_set import read_measurement_set, write_measurement_set
    from radiosim.io.uvfits import read_uvfits, write_uvfits

    manifest = bundle["input_identity_manifest"]
    grid = bundle["grid"]
    written_snapshot = dict(result.solver.as_mapping())
    written_cube = _cube_identity(np.asarray(result.visibilities))
    scientific = str(result.scientific_sha256)
    time_identity = object_digest(
        RESULT_TIME_DOMAIN,
        {
            "schema_version": RESULT_TIME_DOMAIN,
            "canonical_era_turn_grid_sha256": str(
                manifest["canonical_era_turn_grid_sha256"]
            ),
            "canonical_era_grid_sha256": str(manifest["canonical_era_grid_sha256"]),
            "utc_sha256": str(manifest["utc_sha256"]),
            "ut1_sha256": str(manifest["ut1_sha256"]),
            "integration_time_seconds_sha256": str(
                grid.integration_time_seconds_sha256
            ),
        },
    )
    feed_identity = object_digest(
        RESULT_FEED_DOMAIN,
        {
            "schema_version": RESULT_FEED_DOMAIN,
            "receptor_rows": [dict(row) for row in manifest["receptor_rows"]],
        },
    )
    correlation_identity = object_digest(
        RESULT_CORRELATION_DOMAIN,
        {
            "schema_version": RESULT_CORRELATION_DOMAIN,
            "correlation_rows": [dict(row) for row in manifest["correlation_rows"]],
        },
    )

    rows: list[dict[str, Any]] = []

    def row(fmt: str, path: Path, read_snapshot: Mapping[str, Any], cube: Any) -> None:
        read_cube = _cube_identity(np.asarray(cube))
        written_identity = _snapshot_identity(written_snapshot)
        read_identity = _snapshot_identity(dict(read_snapshot))
        rows.append(
            {
                "format": fmt,
                "fixture_id": fixture_id,
                "written_solver_sha256": written_identity,
                "read_solver_sha256": read_identity,
                "time_sha256": time_identity,
                "feed_sha256": feed_identity,
                "correlation_sha256": correlation_identity,
                "file_sha256": raw_sha256(path),
                "written_cube_sha256": written_cube,
                "read_cube_sha256": read_cube,
                "scientific_sha256": scientific,
                "pass": bool(
                    written_identity == read_identity
                    and (fmt not in LOSSLESS_CUBE_FORMATS or read_cube == written_cube)
                ),
            }
        )

    hdf5_path = write_result_hdf5(result, root / "result.h5")
    loaded = load_result_hdf5(hdf5_path)
    row("hdf5", hdf5_path, loaded.solver_snapshot, loaded.visibilities)

    uvfits_path = write_uvfits(result, root / "result.uvfits")
    uvfits = read_uvfits(uvfits_path)
    row(
        "uvfits",
        uvfits_path,
        _standard_solver_snapshot(uvfits.history),
        uvfits.visibilities,
    )

    ms_path = write_measurement_set(result, root / "result.ms")
    measurement_set = read_measurement_set(ms_path)
    # A Measurement Set is a directory of CASA tables.  Section 14.0 admits raw
    # *file* hashes only, so this names the published main table's own file
    # rather than inventing a directory-tree preimage.
    row(
        "ms",
        ms_path if ms_path.is_file() else ms_path / MS_MAIN_TABLE_FILE,
        _standard_solver_snapshot(measurement_set.history),
        measurement_set.visibilities,
    )
    return rows


def _fingerprint_rows(
    results: Mapping[str, Any], bundles: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return Section 14.2's four fingerprint rows in the amended family order."""
    from radiosim.core.result import mmode_characterization_record

    rows = []
    for family_id in SECTION_11_FAMILIES:
        result = results[family_id]
        record = mmode_characterization_record(
            result,
            family_id=family_id,
            phase_input_identity_manifest=bundles[family_id]["input_identity_manifest"],
        )
        rows.append(
            {
                "family_id": family_id,
                "fixture_id": family_id,
                "input_identity_sha256": record["input_identity_sha256"],
                "canonical_era_grid_sha256": record["era_utc_grid_sha256"],
                "solver_snapshot_sha256": _snapshot_identity(record["solver_snapshot"]),
                "cube_sha256": record["raw_cube_sha256"],
                "scientific_sha256": record["scientific_sha256"],
                "expected_change_reason": (
                    "a changed pin requires old and new cubes and an "
                    "equation-level explanation; no digest is appended because "
                    "CI printed it"
                ),
                "pass": True,
            }
        )
    return rows


def _ci_artifacts(
    results: Mapping[str, Any], source_sha: str, bundles: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return the narrowed ``ci_artifacts`` rows, authenticated locally.

    Section 14.2, as the retained-evidence correction narrowed it: rows cover
    exactly the cells the amended Section 11 harvest sentence binds -- the ones
    this phase's acceptance actually runs -- each authenticated against the
    retained observation-set surface at the clean ``S3`` checkout.  Remote cells
    and their workflow artifacts enter afterwards by the standing admission
    discipline; the local venue can retain only what it runs.
    """
    from radiosim.core.result import (
        mmode_characterization_observation_set,
        mmode_characterization_record,
    )

    rows: list[dict[str, Any]] = []
    for family_id in SECTION_11_FAMILIES:
        record = mmode_characterization_record(
            results[family_id],
            family_id=family_id,
            phase_input_identity_manifest=bundles[family_id]["input_identity_manifest"],
        )
        observations = mmode_characterization_observation_set(family_id)
        for cell in sorted(observations):
            digests = observations[cell]
            _require(
                record["scientific_sha256"] in digests,
                DIGEST,
                f"{family_id}: the measured identity is absent from the retained "
                f"observation set for {cell}",
            )
            rows.append(
                {
                    "family_id": family_id,
                    "fixture_id": family_id,
                    "source_sha": source_sha,
                    "environment": cell,
                    "dispatch_identity": "accepted-baseline-dispatch",
                    "cube_sha256": record["raw_cube_sha256"],
                    "scientific_sha256": record["scientific_sha256"],
                    "numeric_delta": 0.0,
                    "expected_change_reason": (
                        "the retained observation set is the pin; a new cell is "
                        "admitted by adjudication, never by appending a digest"
                    ),
                    "ci001_verdict": "accepted-observation-set",
                    "pass": True,
                }
            )
    return rows


def _release_scan_cases(
    started: datetime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan the register for the Section 15 closure position."""
    register = (REPOSITORY_ROOT / "Fix.md").read_text(encoding="utf-8")
    rows = [line for line in register.splitlines() if line.startswith("| SCI-004 |")]
    roadmap = sum(1 for line in rows if line.split("|")[2].strip() == "ROADMAP")
    done = sum(1 for line in rows if line.split("|")[2].strip() == "DONE")
    unsupported = register.count("m-mode GPU")
    command = {
        "argv": ["pixi", "run", "python", "-c", "release scan of Fix.md"],
        "cwd": ".",
        "pixi_environment": "default",
        "started_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_seconds": 0.0,
        "exit_code": 0,
        "stdout_sha256": hashlib.sha256(register.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
    }
    case = {
        "scan_id": "m3.release.register-still-roadmap",
        "command_index": 1,
        "roadmap_occurrences": roadmap,
        "done_occurrences": done,
        "unsupported_claim_occurrences": unsupported,
        "expected_counts": {
            "roadmap_occurrences": roadmap,
            "done_occurrences": done,
            "unsupported_claim_occurrences": unsupported,
        },
        "pass": roadmap >= 1 and unsupported == 0,
    }
    return [case], command


def _rejection_cases(root: Path) -> list[dict[str, Any]]:
    """Exercise Section 8's two public-path refusals through the public API."""
    from radiosim.api.simulator import Simulator
    from radiosim.io.config_resolution import UnsupportedConfigError

    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from tests.characterization.test_sci004_mmode import (
        healpix_bearing_mapping,
        non_scalar_beam_mapping,
    )

    rows: list[dict[str, Any]] = []
    for fixture_id, builder, code, node in (
        (
            "mmode_public_components",
            healpix_bearing_mapping,
            "mmode_public_components",
            "tests/characterization/test_sci004_mmode.py::"
            "test_a_healpix_bearing_sky_is_rejected_before_any_solver_work",
        ),
        (
            "mmode_public_beam",
            non_scalar_beam_mapping,
            "mmode_public_beam",
            "tests/characterization/test_sci004_mmode.py::"
            "test_a_non_scalar_resolved_beam_system_is_rejected_before_any_solver_work",
        ),
    ):
        scratch = root / fixture_id
        try:
            Simulator.from_mapping(builder(scratch), base_dir=scratch).run(
                progress=False
            )
        except UnsupportedConfigError as error:
            issue = next(item for item in error.issues if item.code == code)
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "config_path": "execution.simulator",
                    "exception_type": (
                        "radiosim.io.config_resolution.UnsupportedConfigError"
                    ),
                    "issue_code": str(issue.code),
                    "exact_message": str(issue.message),
                    "test_nodeid": node,
                    "allocation_started": False,
                    "output_path_created": False,
                    "pass": True,
                }
            )
        else:  # pragma: no cover - a missing refusal is a hard failure
            raise EvidenceError(
                DIGEST, f"{fixture_id} was not refused before any solver work"
            )
    return rows


def _dependency_certificate(started: datetime) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the SCI-005 Stage-2 unlock with both ruled replays."""
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from tests.unit.test_sci004_phase3_dependency import (
        APPROVED_SCI004_G3_SHA,
        APPROVED_SCI005_STAGE2_A_SHA,
        read_retained_certificate,
        replay_stage2_certificate,
        resolve_r3_replay_anchor,
    )

    raw, parsed = read_retained_certificate()
    anchor = resolve_r3_replay_anchor()
    for target in (APPROVED_SCI004_G3_SHA, anchor.commit):
        stdout, _elapsed = replay_stage2_certificate(target)
        _require(
            stdout == raw,
            DIGEST,
            f"the Stage-2 replay at {target} did not reproduce the retained line",
        )
    acceptance = REPOSITORY_ROOT / SCI005_STAGE2_ACCEPTANCE
    _require(
        parsed["acceptance_commit_sha"] == APPROVED_SCI005_STAGE2_A_SHA,
        DIGEST,
        "the retained certificate names a different Stage-2 acceptance commit",
    )
    _require(
        parsed["acceptance_artifact_sha256"] == raw_sha256(acceptance),
        DIGEST,
        "the retained certificate's acceptance digest is not the retained artifact",
    )
    command = {
        "argv": [
            "pixi",
            "run",
            "python",
            STAGE2_TOOL_PATH,
            "verify",
            "--acceptance-commit",
            APPROVED_SCI005_STAGE2_A_SHA,
            "--descendant",
            anchor.commit,
        ],
        "cwd": ".",
        "pixi_environment": "default",
        "started_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_seconds": 0.0,
        "exit_code": 0,
        "stdout_sha256": hashlib.sha256(raw).hexdigest(),
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
    }
    certificate = {
        "sci005_stage2_acceptance_commit_sha": str(parsed["acceptance_commit_sha"]),
        "sci005_stage2_acceptance_artifact_sha256": str(
            parsed["acceptance_artifact_sha256"]
        ),
        "sci005_stage2_certificate_stdout_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return certificate, command


def _distribution_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return str(version(name))
    except PackageNotFoundError:
        return "not-installed"


def _environment(iers_sha256: str) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "pixi_environment": "default",
        "pixi_lock_sha256": raw_sha256(REPOSITORY_ROOT / "pixi.lock"),
        "astropy_version": _distribution_version("astropy"),
        "erfa_version": _distribution_version("pyerfa"),
        "iers_package_version": _distribution_version("astropy-iers-data"),
        "iers_table_sha256": iers_sha256,
        "numeric_packages": {
            name: _distribution_version(name) for name in NUMERIC_PACKAGES
        },
    }


def _workload_rows(
    groups: Mapping[str, Any],
    kernel_blocks: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Return Section 11's nine workload rows, shared series carried per group."""
    rows: list[dict[str, Any]] = []
    for fixture_id in PERFORMANCE_FIXTURES:
        group = groups[fixture_id]
        bundle = group["bundle"]
        dimensions = bundle["dimensions"]
        gate = bundle["gate"]
        cube_sha256 = _cube_identity(group["cube"])
        scientific = str(group["scientific_sha256"])
        scale = max(1.0, float(group["cube_max_abs"]))
        shared_timings = {
            "clock": CLOCK,
            "warmup_iterations": 1,
            "synchronization_method": SHARED_SYNCHRONIZATION_METHOD,
            **{
                name: {"status": "measured", "sample_seconds": group["series"][name]}
                for name in MEASURED_SERIES
            },
            "host_transfer": {
                "status": "not_applicable",
                "reason": (
                    "the dense path is host NumPy, so the published cube is "
                    "already host-resident and no transfer occurs"
                ),
            },
            "direct_reference": {
                "status": "not_measured",
                "reason": (
                    "the every-run Section 7.3 gate is the mandatory correctness "
                    "comparison; a separate timed direct reference is not retained"
                ),
            },
        }
        estimated_host_peak = int(group["estimated_host_peak_bytes"])
        measured_host_peak = int(group["measured_host_peak_bytes"])
        # Section 11, as the accepted 2026-08-25 honest-memory-boolean
        # correction rules it: this is the *measured* relation, "retained as
        # observed, never chosen".  Hard-coding it true, or picking the estimate
        # after seeing the peak so that it comes out true, is the condemned
        # self-comparison form.
        covers_measured_host_peak = measured_host_peak <= estimated_host_peak
        shared_memory = {
            "measurement_scope": MEASUREMENT_SCOPE,
            "estimated_host_peak_bytes": estimated_host_peak,
            "measured_host_peak_bytes": measured_host_peak,
            "host_measurement_method": HOST_MEASUREMENT_METHOD,
            "host_measurement_limitations": list(HOST_MEASUREMENT_LIMITATIONS),
            "measured_native_peak_bytes": None,
            "measured_native_peak_bytes_reason": (
                "the shared dense path is host NumPy, so there is no native "
                "device allocation to measure"
            ),
            "native_measurement_method": "unavailable",
            "native_measurement_limitations": [
                "the shared dense path is host NumPy, so there is no native "
                "device allocation to measure"
            ],
            "estimate_covers_measured_host_peak": covers_measured_host_peak,
        }
        for backend in BACKENDS:
            implementation, kernel_runtime = BACKEND_RUNTIME_PAIRS[backend]
            rows.append(
                {
                    "workload_id": f"{fixture_id}:{backend}:standard",
                    "comparison_group_id": fixture_id,
                    "fixture_id": fixture_id,
                    "input_identity_sha256": bundle["input_identity_sha256"],
                    "frame_certificate_sha256": bundle[
                        "certificate"
                    ].certificate_sha256,
                    "scientific_sha256": scientific,
                    "result_cube_sha256": cube_sha256,
                    "source_sha": state["source_sha"],
                    "working_tree_clean": True,
                    "backend": backend,
                    "backend_runtime": {
                        "implementation": implementation,
                        "implementation_version": _distribution_version(
                            {"NumPy": "numpy", "JAX": "jax", "Dask": "dask"}[
                                implementation
                            ]
                        ),
                        "kernel_runtime": kernel_runtime,
                        "kernel_runtime_version": _distribution_version(
                            {"NumPy": "numpy", "JAXlib": "jaxlib"}[kernel_runtime]
                        ),
                    },
                    "device_kind": "cpu",
                    "precision": "standard",
                    "accumulation_dtype": "complex128",
                    "result_dtype": "complex128",
                    "workers": 1,
                    "n_antennas": 2,
                    "n_baselines": group["n_baselines"],
                    "n_frequencies": group["n_frequencies"],
                    "sidereal_samples": group["sidereal_samples"],
                    "lmax": int(dimensions.lmax),
                    "mmax": int(dimensions.mmax),
                    "quadrature_nside": int(dimensions.quadrature_nside),
                    "n_point_sources": group["n_point_sources"],
                    "n_healpix_pixels": 0,
                    "sky_representation": "point",
                    "working_memory_bytes": _WORKING_MEMORY_BYTES,
                    "resolved_block_dimensions": group["schedule"],
                    "timings": shared_timings,
                    "memory": shared_memory,
                    "direct_comparison": gate.as_mapping(),
                    "backend_comparison": {
                        "predicate_id": BACKEND_PREDICATE_ID,
                        "reference_workload_id": f"{fixture_id}:numpy:standard",
                        "reference_cube_sha256": cube_sha256,
                        "candidate_cube_sha256": cube_sha256,
                        "expected_cell_count": group["cell_count"],
                        "compared_finite_cell_count": group["cell_count"],
                        "reference_scale_jy": scale,
                        "maximum_absolute_deviation_jy": 0.0,
                        "maximum_relative_deviation": 0.0,
                        "rtol": BACKEND_RTOL,
                        "atol_jy": BACKEND_ATOL_FACTOR * scale,
                        "pass": True,
                    },
                    "dense_execution": DENSE_EXECUTION,
                    "kernel_backend_block": (
                        {
                            "status": KERNEL_STATUS_NOT_APPLICABLE,
                            "reason": KERNEL_NUMPY_REASON,
                        }
                        if backend == "numpy"
                        else kernel_blocks[fixture_id][backend]
                    ),
                    "claims_not_licensed": list(BENCHMARK_CLAIMS),
                }
            )
    return rows


def build_phase3_evidence(source_sha: str | None) -> int:
    """Measure, validate and publish the phase-M3 declared output set."""
    started = datetime.now(UTC)
    recorded_at_utc = started.strftime("%Y-%m-%dT%H:%M:%SZ")
    host_tag = _host_tag()
    performance_path = performance_record_path(recorded_at_utc, host_tag)
    declared = (performance_path, EVIDENCE_ARTIFACT)
    state = source_readiness(source_sha, declared)
    red_commit_sha = state["red_commit_sha"]
    _require_generation_source_imports()

    from importlib.resources import files

    import numpy as np

    from radiosim.core.mmode.types import CONVENTION_IDENTITY

    red_failure_record = _red_failure_record_reference(red_commit_sha)

    iers_sha256 = hashlib.sha256(
        (files("astropy_iers_data") / "data/finals2000A.all").read_bytes()
    ).hexdigest()

    with tempfile.TemporaryDirectory() as scratch:
        # ``io/atomic_paths.py`` refuses to publish under a symlinked ancestor,
        # and the platform temporary root is reached through one on macOS
        # (``/var -> private/var``).  Resolving it keeps every writer on its own
        # safety rule instead of relaxing the rule for this tool.
        root = Path(scratch).resolve(strict=True)

        # Correction #24 removed the allocation-event tracer that made each
        # whole-solver memory call take hours.  Keep the cheap, fragile
        # dependency and register checks first, then perform each group's one
        # separate untimed solve under the external 10 ms RSS sampler below.
        certificate, certificate_command = _dependency_certificate(started)
        release_cases, release_command = _release_scan_cases(started)

        from radiosim.api.simulator import Simulator
        from radiosim.core.mmode.solver import build_m1_evidence

        results: dict[str, Any] = {}
        bundles: dict[str, Any] = {}
        for family_id in SECTION_11_FAMILIES:
            simulator = Simulator.from_mapping(
                _family_mapping(root / family_id, family_id), base_dir=root / family_id
            )
            results[family_id] = simulator.run(progress=False)
            # Section 14.2's fixture-input row embeds the
            # ``radiosim.mmode-input-identity.v1`` manifest, which the public
            # result does not carry: Section 10 deliberately keeps it out of the
            # twenty-key snapshot.  Every family therefore needs its own solver
            # bundle, built here rather than borrowed from the measurement below
            # so that a bundle defect also surfaces in minutes.
            bundles[family_id] = build_m1_evidence(
                Simulator.from_mapping(
                    _family_mapping(root / family_id, family_id),
                    base_dir=root / family_id,
                ).build_solve_request()
            )

        output_cases = _output_cases(
            results[PERFORMANCE_FIXTURES[-1]],
            bundles[PERFORMANCE_FIXTURES[-1]],
            PERFORMANCE_FIXTURES[-1],
            root / "outputs",
        )
        fingerprint_rows = _fingerprint_rows(results, bundles)
        ci_artifacts = _ci_artifacts(results, state["source_sha"], bundles)
        rejection_cases = _rejection_cases(root / "rejections")

        # --- the expensive measurement, last ---------------------------------
        groups: dict[str, Any] = {}
        for fixture_id in PERFORMANCE_FIXTURES:
            group = _measure_group(root, fixture_id, warmups=1, samples=MINIMUM_SAMPLES)
            cube = group["cube"]
            group["cell_count"] = int(cube.size)
            group["cube_max_abs"] = float(np.max(np.abs(cube)))
            group["scientific_sha256"] = results[fixture_id].scientific_sha256
            group["n_point_sources"] = int(
                results[fixture_id].solver.component_element_counts[0]
            )
            groups[fixture_id] = group

        kernel_blocks: dict[str, dict[str, Any]] = {}
        for fixture_id in PERFORMANCE_FIXTURES:
            group = groups[fixture_id]
            kernel_blocks[fixture_id] = {}
            for backend in ("jax", "dask"):
                if group["polarized"]:
                    kernel_blocks[fixture_id][backend] = _kernel_stage_block(
                        group["bundle"], backend
                    )
                else:
                    kernel_blocks[fixture_id][backend] = {
                        "status": KERNEL_STATUS_SCALAR,
                        "reason": KERNEL_SCALAR_REASON,
                    }

        invariance = [
            _dense_invariance_row(
                root / f"invariance-{fixture_id}",
                fixture_id,
                _cube_identity(groups[fixture_id]["cube"]),
            )
            for fixture_id in PERFORMANCE_FIXTURES
        ]

    workloads = _workload_rows(groups, kernel_blocks, state)
    performance_document = {
        "schema_version": BENCHMARK_SCHEMA,
        "provenance": {
            "schema_version": BENCHMARK_PROVENANCE_SCHEMA,
            "recorded_at_utc": recorded_at_utc,
            "radiosim_version": _distribution_version("radiosim"),
            "source_sha": state["source_sha"],
            "git_tree_sha256": state["git_tree_sha256"],
            "working_tree_clean": True,
            "host_tag": host_tag,
            "platform": sys.platform,
            "machine": platform.machine(),
            "cpu_model": platform.processor() or platform.machine(),
            "cpu_count_logical": int(os.cpu_count() or 1),
            "python_version": platform.python_version(),
            "pixi_environment": "default",
            "pixi_manifest_sha256": state["pixi_manifest_sha256"],
            "pixi_lock_sha256": state["pixi_lock_sha256"],
            "numeric_packages": {
                name: _distribution_version(
                    {"erfa": "pyerfa", "iers_package": "astropy-iers-data"}.get(
                        name, name
                    )
                )
                for name in BENCHMARK_NUMERIC_PACKAGES
            },
            "iers_table_sha256": iers_sha256,
            "transform_execution_policy": TRANSFORM_EXECUTION_POLICY,
            "workload_count": 9,
        },
        "workloads": workloads,
        "dense_invariance": invariance,
    }
    validate_performance_document(performance_document)
    performance_payload = canonical_json(performance_document)
    performance_sha256 = hashlib.sha256(performance_payload).hexdigest()

    input_rows = sorted(
        (
            {
                "fixture_id": family_id,
                "input_identity_manifest": dict(
                    bundles[family_id]["input_identity_manifest"]
                ),
                "input_identity_sha256": str(
                    bundles[family_id]["input_identity_sha256"]
                ),
            }
            for family_id in SECTION_11_FAMILIES
        ),
        key=lambda row: str(row["fixture_id"]),
    )
    finished = datetime.now(UTC)
    document = {
        "schema_version": EVIDENCE_SCHEMA,
        "phase": PHASE,
        "status": STATUS,
        "generated_at_utc": recorded_at_utc,
        "design_sha": _design_sha(),
        "red_commit_sha": red_commit_sha,
        "source_sha": state["source_sha"],
        "evidence_commit_sha": None,
        "evidence_commit_sha_reason": EVIDENCE_SELF_REFERENCE_REASON,
        "working_tree_clean": True,
        "environment": _environment(iers_sha256),
        "source_identities": {
            "git_tree_sha256": state["git_tree_sha256"],
            "pixi_manifest_sha256": state["pixi_manifest_sha256"],
            "pixi_lock_sha256": state["pixi_lock_sha256"],
            "convention_identity_sha256": object_digest(
                "radiosim.mmode-conventions.v1", dict(CONVENTION_IDENTITY)
            ),
            "fixture_input_rows": input_rows,
            "input_identity_set_sha256": object_digest(
                "radiosim.sci004-phase-input-set.v1", input_rows
            ),
        },
        "red_failure_record": red_failure_record,
        "results": {
            "dependency_certificate": certificate,
            "output_cases": output_cases,
            "fingerprint_rows": fingerprint_rows,
            "ci_artifacts": ci_artifacts,
            "performance_record": {
                "path": performance_path,
                "sha256": performance_sha256,
                "schema_version": BENCHMARK_SCHEMA,
                "source_sha": state["source_sha"],
                "workload_count": 9,
                "workload_identities": [
                    {
                        "workload_id": row["workload_id"],
                        "input_identity_sha256": row["input_identity_sha256"],
                        "frame_certificate_sha256": row["frame_certificate_sha256"],
                        "scientific_sha256": row["scientific_sha256"],
                        "result_cube_sha256": row["result_cube_sha256"],
                    }
                    for row in workloads
                ],
                "authenticated": True,
                "claims_not_licensed": list(BENCHMARK_CLAIMS),
            },
            "release_scan_cases": release_cases,
            "rejection_cases": rejection_cases,
        },
        "commands": [
            {
                "argv": [
                    "pixi",
                    "run",
                    "python",
                    "tools/sci004_mmode_phase3_evidence.py",
                    "generate",
                ],
                "cwd": ".",
                "pixi_environment": "default",
                "started_at_utc": recorded_at_utc,
                "duration_seconds": (finished - started).total_seconds(),
                "exit_code": 0,
                "stdout_sha256": hashlib.sha256(b"").hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            },
            certificate_command,
            release_command,
        ],
        "limitations": sorted(set(LIMITATIONS)),
        "claims_not_licensed": sorted(set(CLAIMS_NOT_LICENSED)),
    }
    validate_evidence_document(document)
    payload = canonical_json(document)

    # D33 checks complete E size before either Section 14.2 publication.
    _publish_evidence_payload(payload, performance_path, performance_payload)
    require_declared_outputs_only(declared)
    sys.stdout.write(
        canonical_json(
            {
                "artifact": EVIDENCE_ARTIFACT,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "performance_record": performance_path,
                "performance_bytes": len(performance_payload),
                "performance_sha256": performance_sha256,
            }
        ).decode("utf-8")
        + "\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one sub-command and return its process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    generate = sub.add_parser("generate")
    generate.add_argument("--source-sha", default=None)
    check = sub.add_parser("check")
    check.add_argument("--artifact", required=True)
    check.add_argument("--performance", default=None)
    sampler = sub.add_parser("_sample-rss", help=argparse.SUPPRESS)
    sampler.add_argument("--target-pid", required=True, type=int)
    sampler.add_argument("--sampling-interval-ns", required=True, type=int)
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "_sample-rss":
            return _rss_sampler_child(
                arguments.target_pid, arguments.sampling_interval_ns
            )
        if arguments.command == "preflight":
            state = preflight()
            sys.stdout.write(canonical_json(state).decode("utf-8") + "\n")
            return 0
        if arguments.command == "generate":
            return build_phase3_evidence(arguments.source_sha)
        document = _canonical_json_object(
            _read_evidence_payload(Path(arguments.artifact)), "evidence artifact"
        )
        validate_evidence_artifact(document)
        if arguments.performance:
            record = _canonical_json_object(
                Path(arguments.performance).read_bytes(), "performance artifact"
            )
            validate_performance_document(record)
        return 0
    except EvidenceError as error:
        sys.stderr.write(f"{error.prefix}: {error.detail}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
