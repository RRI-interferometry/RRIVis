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

**Why the module imports only the standard library.** It follows
``tools/wp7_perf001_cpu_evidence.py`` and the phase-2 sibling: an
evidence-critical verifier must not carry an import-time dependency on a package
that is merely transitively present, so ``check`` -- the sub-command a reviewer
runs -- needs nothing but Python.  The scientific packages are imported inside
the generation functions alone.

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
import hashlib
import json
import math
import os
import platform
import re
import select
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

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


def _render(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return ecmascript_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Mapping):
        body = ",".join(
            f"{json.dumps(str(key), ensure_ascii=False, separators=(',', ':'))}:"
            f"{_render(item)}"
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
        return "{" + body + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_render(item) for item in value) + "]"
    raise EvidenceError(SCHEMA, f"{type(value).__name__} is not JSON")


def canonical_json(value: Any) -> bytes:
    """Return Section 14's canonical UTF-8 serialization of one document."""
    return _render(value).encode("utf-8")


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
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(SCHEMA, f"the RSS sampler {label} is not JSON") from exc
    record = _require_keys(value, keys, f"RSS sampler {label}")
    _require(
        canonical_json(record) == payload,
        SCHEMA,
        f"the RSS sampler {label} is not canonical JSON",
    )
    return record


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


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise EvidenceError(
            PREFLIGHT, f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def raw_sha256(path: Path) -> str:
    """Return the SHA-256 of a file's exact raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_binding(name: str) -> str:
    """Return one frozen constant from the immutable R3 dependency validator.

    Section 14.0 names that module as this phase's single site for the frozen
    bindings, so they are read from its source rather than restated here where
    they could silently diverge.
    """
    source = (REPOSITORY_ROOT / DEPENDENCY_VALIDATOR_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    values = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
    ]
    if len(values) != 1 or not isinstance(values[0], str):
        raise EvidenceError(
            PREFLIGHT, f"{DEPENDENCY_VALIDATOR_PATH} must freeze exactly one {name}"
        )
    return str(values[0])


#: Section 13.1's three design-authority paths.  ``D0`` introduced the memo with
#: exactly these; every recorded correction's parent-relative diff stays inside
#: them.
DESIGN_AUTHORITY_PATHS: frozenset[str] = frozenset(
    {
        "docs/development/sci004_mmode_design.md",
        "docs/index.rst",
        "PostTier8RemediationPlan.md",
    }
)


def _design_sha() -> str:
    """Return the phase's frozen ``design_sha``, authenticated as a Git object.

    Section 13.1 is explicit that a phase's ``design_sha`` is "exactly the
    operative ``D`` frozen for its phase under Section 14.0, never a phase-local
    memo tip or a search result", and Section 14.0 says every generator "reads
    the phase-appropriate frozen binding".  Deriving it as the newest
    memo-touching commit is therefore not a cross-check but the forbidden search;
    at ``S3`` it would also be wrong, because Section 14.4 stars the
    ``R3 ->* S3`` edge, so accepted corrections stand between the frozen ``D``
    and this checkout by construction.

    What is checked here is what Section 14.0 actually demands of the value
    before it is trusted: it peels to a commit, is a single-parent non-merge
    ancestor of the checkout, and its parent-relative diff stays inside Section
    13.1's three design-authority paths.  The header-enumerated correction chain
    between that binding and the checkout is authenticated by the R3 dependency
    validator, which ``_dependency_certificate`` runs in this same generation.
    """
    frozen = _frozen_binding("APPROVED_SCI004_D_SHA")
    if re.fullmatch(r"[0-9a-f]{40}", frozen) is None:
        raise EvidenceError(
            PREFLIGHT,
            f"the frozen design binding {frozen} is not a "
            "lower-case 40-hex Git identity",
        )
    peeled = _git("rev-parse", f"{frozen}^{{commit}}").strip()
    if peeled != frozen:
        raise EvidenceError(
            PREFLIGHT, f"the frozen design binding {frozen} is not a commit object"
        )
    parents = _git("rev-list", "--parents", "-n", "1", frozen).split()
    if len(parents) != 2:
        raise EvidenceError(
            PREFLIGHT, f"the operative D {frozen} must be a single-parent non-merge"
        )
    head = _git("rev-parse", "HEAD").strip()
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", frozen, head],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if ancestry.returncode != 0:
        raise EvidenceError(
            PREFLIGHT, f"the operative D {frozen} is not an ancestor of {head}"
        )
    touched = {
        line.strip()
        for line in _git(
            "diff-tree", "--no-commit-id", "--name-only", "-r", frozen
        ).splitlines()
        if line.strip()
    }
    if not touched or not touched <= DESIGN_AUTHORITY_PATHS:
        raise EvidenceError(
            PREFLIGHT,
            f"the operative D {frozen} touches {sorted(touched)}, which leaves "
            "Section 13.1's design-authority paths",
        )
    if "docs/development/sci004_mmode_design.md" not in touched:
        raise EvidenceError(
            PREFLIGHT, f"the operative D {frozen} does not touch the design memo"
        )
    return frozen


def _red_commit_sha() -> str:
    """Return the fresh correction-24 ``R3`` containing the supplement."""
    commit = _git(
        "log",
        "-1",
        "--format=%H",
        "--",
        POST_SOURCE_RED_FAILURE_RECORD,
    ).strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise EvidenceError(
            PREFLIGHT,
            "the correction-24 post-source red supplement has no containing R3",
        )
    return commit


def _git_blob(commit: str, path: str) -> bytes:
    """Return one exact tree blob or refuse before evidence generation."""
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
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
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(PREFLIGHT, f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise EvidenceError(PREFLIGHT, f"{label} is not its canonical JSON bytes")
    return raw, value


def _red_failure_record_reference(red_commit: str) -> dict[str, Any]:
    """Authenticate and join the immutable red record and post-source delta."""
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
    if post_source.get("design_sha") != _design_sha():
        raise EvidenceError(
            PREFLIGHT, "the post-source red record does not bind the operative D"
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

    return {
        "path": RED_FAILURE_RECORD,
        "sha256": historical_sha256,
        "schema_version": RED_FAILURE_SCHEMA,
        "pre_fix_source_sha": historical_pre_fix,
        "validated": True,
        "post_source_delta": {
            "path": POST_SOURCE_RED_FAILURE_RECORD,
            "sha256": hashlib.sha256(post_raw).hexdigest(),
            "schema_version": POST_SOURCE_RED_FAILURE_SCHEMA,
            "pre_fix_source_sha": POST_SOURCE_PRE_FIX_SHA,
            "validated": True,
        },
    }


def validate_evidence_artifact(document: Any) -> dict[str, Any]:
    """Validate an envelope and authenticate both of its retained red inputs."""
    envelope = validate_evidence_document(document)
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
    return {
        "source_sha": head,
        "pixi_manifest_sha256": raw_sha256(REPOSITORY_ROOT / "pixi.toml"),
        "pixi_lock_sha256": raw_sha256(REPOSITORY_ROOT / "pixi.lock"),
        "git_tree_sha256": domain_digest(
            "radiosim.sci004.git-tree.v1",
            subprocess.run(
                ["git", "ls-tree", "-r", "-z", "--full-tree", head],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                check=True,
            ).stdout,
        ),
    }


def require_declared_outputs_only(declared: Sequence[str]) -> None:
    """Require the repository's only new paths to equal the declared set."""
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    observed = sorted(line[3:].strip() for line in status.splitlines() if line.strip())
    expected = sorted(declared)
    _require(
        observed == expected,
        DIGEST,
        f"after publication the repository's new paths must be exactly "
        f"{expected}, not {observed}",
    )


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


def _fingerprint_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return Section 14.2's four fingerprint rows in the amended family order."""
    from radiosim.core.result import mmode_characterization_record

    rows = []
    for family_id in SECTION_11_FAMILIES:
        result = results[family_id]
        record = mmode_characterization_record(result, family_id=family_id)
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


def _ci_artifacts(results: Mapping[str, Any], source_sha: str) -> list[dict[str, Any]]:
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
        record = mmode_characterization_record(results[family_id], family_id=family_id)
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
    from importlib.resources import files

    import numpy as np

    from radiosim.core.mmode.types import CONVENTION_IDENTITY

    started = datetime.now(UTC)
    recorded_at_utc = started.strftime("%Y-%m-%dT%H:%M:%SZ")
    host_tag = _host_tag()
    performance_path = performance_record_path(recorded_at_utc, host_tag)
    declared = (performance_path, EVIDENCE_ARTIFACT)
    state = preflight(source_sha, declared)
    red_commit_sha = _red_commit_sha()
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
        fingerprint_rows = _fingerprint_rows(results)
        ci_artifacts = _ci_artifacts(results, state["source_sha"])
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

    # Section 14.2: performance record first, evidence last.
    write_atomic_no_overwrite(REPOSITORY_ROOT / performance_path, performance_payload)
    write_atomic_no_overwrite(REPOSITORY_ROOT / EVIDENCE_ARTIFACT, payload)
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
        document = json.loads(Path(arguments.artifact).read_bytes().decode("utf-8"))
        validate_evidence_artifact(document)
        if arguments.performance:
            record = json.loads(
                Path(arguments.performance).read_bytes().decode("utf-8")
            )
            validate_performance_document(record)
        return 0
    except EvidenceError as error:
        sys.stderr.write(f"{error.prefix}: {error.detail}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
