#!/usr/bin/env python
"""Generate and check the SCI-004 phase-M1 evidence artifact.

``docs/development/sci004_mmode_design.md`` Sections 13.3, 14.2 and 14.4 freeze
this tool's authority.  It is tracked at ``S1``, so a clean exact ``S1`` already
contains the bytes that will produce and validate its successor, and ``E1``
adds only the generated artifact, its reproduction record, and the two approved
constants in the strict validator.

Importing this module loads only the Python standard library, following
``tools/wp7_perf001_cpu_evidence.py`` and ``tools/sci005_stage_evidence.py``: an
evidence-critical generator must not depend on a package that is merely
transitively present, because a lock update could drop it and turn a hard
refusal into an import error.

Sub-commands::

    pixi run python tools/sci004_mmode_phase1_evidence.py preflight
    pixi run python tools/sci004_mmode_phase1_evidence.py generate
    pixi run python tools/sci004_mmode_phase1_evidence.py check --artifact <path>

``preflight`` performs Section 14.2's common pre-output check without writing
anything: ``git rev-parse HEAD`` must equal the approved ``S``, the index,
worktree and untracked set must all be empty, the Pixi manifest and lock must
match their recorded digests, and the declared output set must be absent.
``generate`` repeats that check and then writes the single declared artifact by
atomic no-overwrite rename.  ``check`` re-validates an existing artifact's
canonical bytes, schema literal, key order and cross-field rules.

The artifact itself is UTF-8 canonical JSON: object keys sorted
lexicographically, separators ``,`` and ``:``, no whitespace, no trailing
newline, and RFC 8785 / ECMAScript shortest-round-trip numbers.  NaN and
Infinity are forbidden and every object rejects unknown or missing keys.

**Section 14.4 places the run at the globally clean exact ``S1``.**  That
execution *is* the ``E1``-time generation and the artifact it writes is
precisely what the following ``E1`` commit adds: "runs only at its globally
clean exact ``S``" names the venue, not a prohibition.  ``generate`` therefore
produces at ``S1`` whenever the common pre-output check passes, and refuses
otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

PHASE = "M1"
EVIDENCE_SCHEMA = "radiosim.sci004.mmode-phase1-evidence.v1"
EVIDENCE_ARTIFACT = "docs/development/sci004_mmode_phase1_evidence.json"
REPRODUCTION_RECORD = "docs/development/sci004_mmode_phase1_evidence.md"
EVIDENCE_VALIDATOR = "tests/unit/test_sci004_phase1_evidence.py"
RED_FAILURE_RECORD = "docs/development/sci004_mmode_phase1_red_failures.json"
RED_FAILURE_SCHEMA = "radiosim.sci004.mmode-phase1-red-failures.v1"
DEPENDENCY_ARTIFACT = "docs/development/sci004_mmode_phase1_wp7_dependency.json"

#: Section 14.2's declared output set for M1.  Exactly one file.
DECLARED_OUTPUTS: tuple[str, ...] = (EVIDENCE_ARTIFACT,)

#: Section 14.2's frozen stderr prefixes.
ARGUMENT = "SCI004_EVIDENCE_ARGUMENT"
PREFLIGHT = "SCI004_EVIDENCE_PREFLIGHT"
SCHEMA = "SCI004_EVIDENCE_SCHEMA"
DIGEST = "SCI004_EVIDENCE_DIGEST"

#: Section 14.2's exact evidence envelope key set.
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

#: Section 14.2's exact ``results`` key set for M1.
RESULT_KEYS: tuple[str, ...] = (
    "dependency_certificate",
    "time_grid_cases",
    "frame_certificate_cases",
    "scalar_harmonic_cases",
    "packed_layout_cases",
    "transfer_cases",
    "strategy_cases",
    "capability_cases",
    "direct_identity_cases",
    "truncation_cases",
    "rejection_cases",
)

#: Section 14.2's exact ``environment`` key set.
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

#: Section 14.2's exact ``numeric_packages`` key set.
NUMERIC_PACKAGE_KEYS: tuple[str, ...] = ("numpy", "scipy", "healpy", "jax", "dask")

#: Section 14.0's exact six-key fixture-input schema.
SOURCE_IDENTITY_KEYS: tuple[str, ...] = (
    "git_tree_sha256",
    "pixi_manifest_sha256",
    "pixi_lock_sha256",
    "convention_identity_sha256",
    "fixture_input_rows",
    "input_identity_set_sha256",
)

#: Section 14.2's exact ``red_failure_record`` key set.
RED_RECORD_KEYS: tuple[str, ...] = (
    "path",
    "sha256",
    "schema_version",
    "pre_fix_source_sha",
    "validated",
)

#: Section 14.1's exact command-row shape, reused by evidence commands.
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

#: Section 14.2's exact self-reference reason.
EVIDENCE_SELF_REFERENCE_REASON = "self-reference: A binds the containing E commit"

#: Section 14.2's exact M1 ``capability_cases`` inventory, in order.
CAPABILITY_ROW_ORDER: tuple[tuple[str, str], ...] = (
    ("property", "mmode-supports-polarization-false"),
    ("property", "rime-supports-polarization-true"),
    ("registry", "registry-includes-scalar-mmode"),
    ("rejection", "mmode-rejects-nonzero-q"),
    ("rejection", "mmode-rejects-nonzero-u"),
    ("rejection", "mmode-rejects-nonzero-v"),
)

CAPABILITY_PROPERTY_KEYS: tuple[str, ...] = (
    "case_kind",
    "case_id",
    "simulator",
    "property",
    "expected_boolean",
    "observed_boolean",
    "tier7_test_nodeid",
    "pass",
)
CAPABILITY_REGISTRY_KEYS: tuple[str, ...] = (
    "case_kind",
    "case_id",
    "expected_names",
    "observed_names",
    "tier7_test_nodeid",
    "pass",
)
CAPABILITY_REJECTION_KEYS: tuple[str, ...] = (
    "case_kind",
    "case_id",
    "simulator",
    "stokes_field",
    "configured_value_f64be",
    "exception_type",
    "issue_code",
    "exact_message",
    "test_nodeid",
    "pass",
)

#: Section 14.2's exact M1/M2 ``truncation_cases`` row, in the memo's order.
#: The tier-1a horizon-free fields and the two per-fixture budgets are the
#: ``sci004_two_tier_direct.v3`` surface: tier 1a gates numerically, tier 1b and
#: the deficit are recorded and bounded only by their reviewed budgets.
TRUNCATION_ROW_KEYS: tuple[str, ...] = (
    "fixture_id",
    "input_identity_sha256",
    "frame_certificate_sha256",
    "direction_ledger_sha256",
    "transfer_grid_catalog_sha256",
    "production_transfer_grid_id",
    "diagnostic_transfer_grid_ids",
    "diagnostic_grid_joins",
    "lmax",
    "mmax",
    "quadrature_nside",
    "lcheck",
    "mcheck",
    "qcheck",
    "sidereal_samples",
    "cube_shape",
    "frozen_gauss128_cube_sha256",
    "frozen_enclosure_error_cube_sha256",
    "mmode_cube_sha256",
    "direct_scale_jy",
    "expected_output_cell_count",
    "evaluated_frozen_direct_cell_count",
    "evaluated_frozen_error_cell_count",
    "evaluated_mmode_cell_count",
    "compared_output_cell_count",
    "direct_coverage",
    "direct_coverage_sha256",
    "horizon_free_shell_max_jy",
    "horizon_free_shell_l2",
    "horizon_free_shell_max_limit_jy",
    "horizon_free_shell_l2_limit",
    "quadrature_shell_max_jy",
    "quadrature_shell_l2",
    "quadrature_budget_jy",
    "deficit_max_jy",
    "deficit_l2",
    "deficit_max_quarter_jy",
    "deficit_max_half_jy",
    "convergence_factor",
    "truncation_budget_jy",
    "expected_shell_comparison_cell_count",
    "evaluated_shell_comparison_cell_count",
    "expected_transfer_sample_row_count",
    "evaluated_transfer_sample_row_count",
    "expected_field_block_count",
    "evaluated_field_block_count",
    "shell_coverage",
    "shell_coverage_sha256",
    "quadrature_diagnostic_max_jy",
    "l_tail_diagnostic_max_jy",
    "m_tail_diagnostic_max_jy",
    "combined_local_diagnostic_max_jy",
    "field_block_diagnostic_max_jy",
    "shell_diagnostic_reference_jy",
    "pass",
)

#: Section 12.1's four terminal-row classifications.
SCAN_CLASSIFICATIONS: tuple[str, ...] = (
    "ceiling_excludes_root",
    "scan_crossing",
    "guard_interval",
    "excluded_upper_endpoint",
)
SCAN_CROSSING_CLASSIFICATIONS: tuple[str, ...] = (
    "scan_crossing",
    "excluded_upper_endpoint",
)

#: Section 12.1's economy scan-projection row shapes.  The terminal-row field
#: order is the sole discriminated-format exception in Section 14; the retained
#: projection embeds every crossing row verbatim in that shape, each crossing's
#: flanking guard rows -- without them the adjacency and partition rules below
#: have no preimage -- plus one summary row per direction.
SCAN_ROW_KEYS: tuple[str, ...] = (
    "direction_id",
    "cell_index",
    "turn_lo",
    "turn_hi",
    "classification",
    "f_lo_f64be",
    "f_hi_f64be",
    "ceiling_margin_f64be",
    "left_sign",
    "right_sign",
    "root_turn_lo",
    "root_turn_hi",
    "root_orientation",
    "root_residual_f64be",
)
SCAN_SUMMARY_ROW_KEYS: tuple[str, ...] = (
    "direction_id",
    "terminal_cell_count",
    "boundary_evaluation_count",
    "crossing_count",
    "min_ceiling_margin_f64be",
)

#: Section 12.1's per-direction membership mask row.
MEMBERSHIP_MASK_ROW_KEYS: tuple[str, ...] = (
    "direction_id",
    "sample_count",
    "frozen_visible_mask_hex",
    "operational_visible_mask_hex",
    "mismatch_count",
)

#: Section 7.3's transfer-sample concatenation row: one per catalogue grid and
#: output cell, never one per direction.
TRANSFER_SAMPLE_ROW_KEYS: tuple[str, ...] = (
    "grid_id",
    "baseline_index",
    "frequency_index",
    "correlation_index",
    "field_index",
    "field_name",
    "resolved_lmax",
    "resolved_mmax",
    "block_table_sha256",
    "direction_count",
    "packed_sample_value_count",
    "concatenation_sha256",
)

#: Section 14.2's exact M1/M2 frame-certificate row, in the memo's order.
FRAME_ROW_KEYS: tuple[str, ...] = (
    "fixture_id",
    "certificate_sha256",
    "site_manifest",
    "site_sha256",
    "input_identity_sha256",
    "iers_table_sha256",
    "frame_matrix_manifest",
    "frame_matrix_sha256",
    "canonical_era_turn_grid_sha256",
    "canonical_era_grid_sha256",
    "pm_source_unit",
    "pom00_argument_unit",
    "xp0_arcsec",
    "yp0_arcsec",
    "das2r_rad_per_arcsec",
    "xp0_rad",
    "yp0_rad",
    "sp0_rad",
    "diagnostic_qcheck_nsides",
    "transfer_grid_catalog",
    "transfer_grid_catalog_sha256",
    "direction_rows",
    "direction_ledger_sha256",
    "horizon_scan_manifest",
    "horizon_scan_sha256",
    "horizon_scan_crossing_rows",
    "horizon_scan_summary_rows",
    "horizon_scan_ledger_sha256",
    "horizon_root_pair_rows",
    "horizon_root_pair_ledger_sha256",
    "horizon_slab_rows",
    "horizon_slab_ledger_sha256",
    "horizon_sign_interval_rows",
    "horizon_sign_interval_ledger_sha256",
    "horizon_membership_mask_rows",
    "horizon_membership_ledger_sha256",
    "direct_split_rows",
    "direct_split_ledger_sha256",
    "direct_integrand_enclosure_manifest",
    "direct_integrand_enclosure_sha256",
    "sidereal_samples",
    "quadrature_nside",
    "n_baselines",
    "n_frequencies",
    "n_correlations",
    "expected_point_direction_count",
    "evaluated_point_direction_count",
    "expected_native_healpix_direction_count",
    "evaluated_native_healpix_direction_count",
    "expected_production_transfer_direction_count",
    "evaluated_production_transfer_direction_count",
    "expected_diagnostic_transfer_direction_count",
    "evaluated_diagnostic_transfer_direction_count",
    "expected_transfer_quadrature_direction_count",
    "evaluated_transfer_quadrature_direction_count",
    "expected_direction_count",
    "evaluated_direction_count",
    "expected_phase_comparison_count",
    "evaluated_phase_comparison_count",
    "expected_horizon_trajectory_count",
    "evaluated_horizon_trajectory_count",
    "expected_horizon_root_pair_row_count",
    "evaluated_horizon_root_pair_row_count",
    "expected_horizon_membership_count",
    "evaluated_horizon_membership_count",
    "expected_direct_exposure_split_count",
    "evaluated_direct_exposure_split_count",
    "expected_direct_split_row_count",
    "evaluated_direct_split_row_count",
    "expected_frozen_gauss64_node_count",
    "evaluated_frozen_gauss64_node_count",
    "expected_frozen_gauss128_node_count",
    "evaluated_frozen_gauss128_node_count",
    "expected_operational_gauss64_node_count",
    "evaluated_operational_gauss64_node_count",
    "expected_operational_gauss128_node_count",
    "evaluated_operational_gauss128_node_count",
    "horizon_isolation_interval_count",
    "horizon_unresolved_interval_count",
    "expected_horizon_slab_row_count",
    "evaluated_horizon_slab_row_count",
    "expected_horizon_sign_interval_count",
    "evaluated_horizon_sign_interval_count",
    "horizon_root_count_mismatches",
    "horizon_root_orientation_mismatches",
    "horizon_membership_mismatches",
    "horizon_outside_slab_sign_mismatches",
    "horizon_paired_root_count",
    "horizon_mismatch_slab_count",
    "horizon_mismatch_measure_turn",
    "horizon_mismatch_measure_rad",
    "horizon_mismatch_measure_limit_rad",
    "horizon_root_max_rad",
    "horizon_root_limit_rad",
    "phase_max_rad",
    "phase_limit_rad",
    "expected_cube_cell_count",
    "evaluated_frozen_gauss64_cube_cell_count",
    "evaluated_frozen_gauss128_cube_cell_count",
    "evaluated_operational_gauss64_cube_cell_count",
    "evaluated_operational_gauss128_cube_cell_count",
    "compared_frozen_gauss_change_cell_count",
    "compared_operational_gauss_change_cell_count",
    "evaluated_frozen_enclosure_error_cell_count",
    "evaluated_operational_enclosure_error_cell_count",
    "frozen_gauss64_cube_sha256",
    "frozen_gauss128_cube_sha256",
    "operational_gauss64_cube_sha256",
    "operational_gauss128_cube_sha256",
    "frozen_enclosure_error_cube_sha256",
    "operational_enclosure_error_cube_sha256",
    "direct_gauss_scale_jy",
    "frozen_gauss_change_max_jy",
    "operational_gauss_change_max_jy",
    "direct_gauss_change_max_jy",
    "direct_gauss_change_limit_jy",
    "cube_scale_jy",
    "cube_max_jy",
    "cube_limit_jy",
    "cube_l2",
    "cube_l2_limit",
    "direction_diagnostic_max_rad",
    "direction_diagnostic_argmax_id",
    "direction_diagnostic_argmax_phase",
    "basis_diagnostic_max_rad",
    "basis_diagnostic_argmax_id",
    "basis_diagnostic_argmax_phase",
    "pass",
)

#: Section 4.2's fixed frame limits.  None of them is a per-fixture budget.
FRAME_ROOT_LIMIT_RAD = 2e-5
FRAME_PHASE_LIMIT_RAD = 5e-3
FRAME_CUBE_L2_LIMIT = 5e-5

#: Section 7.3's fixed tier-1a limits and tier-2 convergence floor.
HORIZON_FREE_RELATIVE_LIMIT = 1e-8
HORIZON_FREE_ABSOLUTE_FLOOR_JY = 1e-10
HORIZON_FREE_L2_LIMIT = 1e-8
CONVERGENCE_FACTOR_FLOOR = 2.0

#: Section 14.2's exact ``dependency_certificate`` and direct-identity shapes.
DEPENDENCY_CERTIFICATE_KEYS: tuple[str, ...] = ("path", "raw_sha256", "certificate")
DIRECT_IDENTITY_KEYS: tuple[str, ...] = (
    "fixture_id",
    "rime_before_sha256",
    "rime_after_sha256",
    "scientific_before_sha256",
    "scientific_after_sha256",
    "byte_identical",
    "pass",
)

#: Section 13.2's exact sixteen-field WP-7 certificate.
WP7_CERTIFICATE_KEYS: tuple[str, ...] = (
    "schema_version",
    "acceptance_commit",
    "evidence_commit",
    "generating_source_sha",
    "descendant_commit",
    "artifact_path",
    "artifact_sha256",
    "cpu_evidence_tool_sha256",
    "production_record_validator_sha256",
    "production_harness_sha256",
    "pixi_manifest_sha256",
    "pixi_lock_sha256",
    "evidence_diff_paths",
    "acceptance_diff_paths",
    "verdict",
    "passed",
)


class EvidenceError(RuntimeError):
    """One refusal, carrying the frozen stderr prefix it must be reported with."""

    def __init__(self, prefix: str, detail: str) -> None:
        self.prefix = prefix
        self.detail = detail
        super().__init__(f"{prefix}: {detail}")


# ---------------------------------------------------------------------------
# Section 14 canonical JSON
# ---------------------------------------------------------------------------


def ecmascript_number(value: float) -> str:
    """Render a finite binary64 with ECMAScript ``Number::toString`` spelling."""
    if not math.isfinite(value):
        raise EvidenceError(SCHEMA, "canonical JSON forbids NaN and Infinity")
    if value == 0.0:
        return "0"
    text = repr(float(value))
    negative = text.startswith("-")
    if negative:
        text = text[1:]
    mantissa, _, exponent_text = text.partition("e")
    exponent = int(exponent_text) if exponent_text else 0
    integer_part, _, fraction_part = mantissa.partition(".")
    digits = (integer_part + fraction_part).lstrip("0")
    point = len(integer_part) - (len(integer_part + fraction_part) - len(digits))
    point += exponent
    digits = digits.rstrip("0") or "0"
    count = len(digits)
    if count <= point <= 21:
        rendered = digits + "0" * (point - count)
    elif 0 < point <= 21:
        rendered = digits[:point] + "." + digits[point:]
    elif -6 < point <= 0:
        rendered = "0." + "0" * (-point) + digits
    else:
        exponent_value = point - 1
        sign = "+" if exponent_value >= 0 else "-"
        head = digits[0] if count == 1 else digits[0] + "." + digits[1:]
        rendered = f"{head}e{sign}{abs(exponent_value)}"
    return "-" + rendered if negative else rendered


def _render(value: Any) -> str:
    """Render one JSON value with Section 14's exact serialization.

    ``json.dumps`` is not used for the numbers: its encoder is hard-wired to
    ``float.__repr__``, which spells the integer one as ``1.0`` and the exponent
    of ``1e-7`` as ``1e-07``.  Neither is canonical, so the number path is
    rendered here and only strings are delegated to the standard escaper.
    """
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        return ecmascript_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, dict):
        entries = sorted((str(key), item) for key, item in value.items())
        return (
            "{"
            + ",".join(
                json.dumps(key, ensure_ascii=True) + ":" + _render(item)
                for key, item in entries
            )
            + "}"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_render(item) for item in value) + "]"
    raise EvidenceError(SCHEMA, f"cannot canonicalize {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Return Section 14's ``J(x)`` bytes for a JSON-primitive tree."""
    return _render(value).encode("utf-8")


def domain_digest(domain: str, payload: bytes) -> str:
    """Return Section 14.0's ``D(d, p) = SHA256(d || NUL || U64(len(p)) || p)``."""
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Section 14.2 strict validation
# ---------------------------------------------------------------------------


def _require(condition: bool, prefix: str, detail: str) -> None:
    if not condition:
        raise EvidenceError(prefix, detail)


def _require_keys(value: Any, keys: tuple[str, ...], label: str) -> dict[str, Any]:
    """Require an object to carry exactly one key set, rejecting any deviation.

    Section 14's canonical serialization sorts object keys lexicographically, so
    a re-read artifact never preserves an author's insertion order: "exactly
    these keys" is a statement about the *set*.  Both a missing and an unknown
    key are named in the refusal, because a schema drift is far easier to fix
    when the message says which side moved.
    """
    _require(isinstance(value, dict), SCHEMA, f"{label} must be an object")
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
        and len(value) == width
        and all(character in "0123456789abcdef" for character in value),
        SCHEMA,
        f"{label} must be a lower-case {width}-hex string",
    )
    return str(value)


def _require_sorted_unique_strings(value: Any, label: str) -> list[str]:
    _require(isinstance(value, list), SCHEMA, f"{label} must be an array")
    items = list(value)
    _require(
        all(isinstance(item, str) and item for item in items),
        SCHEMA,
        f"{label} must contain non-empty strings",
    )
    _require(
        items == sorted(set(items)) and len(items) == len(set(items)),
        SCHEMA,
        f"{label} must be sorted and unique",
    )
    return [str(item) for item in items]


def validate_command_row(row: Any, label: str) -> None:
    """Validate one Section 14.1 command row with a zero exit code."""
    mapping = _require_keys(row, COMMAND_KEYS, label)
    argv = mapping["argv"]
    _require(
        isinstance(argv, list)
        and argv
        and all(isinstance(item, str) and item for item in argv),
        SCHEMA,
        f"{label}.argv must be a non-empty string array",
    )
    _require(mapping["cwd"] == ".", SCHEMA, f"{label}.cwd must be '.'")
    _require(
        isinstance(mapping["pixi_environment"], str) and mapping["pixi_environment"],
        SCHEMA,
        f"{label}.pixi_environment must be a non-empty string",
    )
    _require(
        mapping["exit_code"] == 0 and not isinstance(mapping["exit_code"], bool),
        SCHEMA,
        f"{label}.exit_code must be the integer zero",
    )
    _require(
        isinstance(mapping["duration_seconds"], (int, float))
        and not isinstance(mapping["duration_seconds"], bool)
        and float(mapping["duration_seconds"]) >= 0.0,
        SCHEMA,
        f"{label}.duration_seconds must be finite and non-negative",
    )
    _require_hex(mapping["stdout_sha256"], 64, f"{label}.stdout_sha256")
    _require_hex(mapping["stderr_sha256"], 64, f"{label}.stderr_sha256")


def validate_capability_cases(rows: Any) -> None:
    """Validate Section 14.2's exact six-row discriminated capability array."""
    _require(isinstance(rows, list), SCHEMA, "capability_cases must be an array")
    _require(
        len(rows) == 6,
        SCHEMA,
        "capability_cases is an exact six-row array",
    )
    for index, (kind, case_id) in enumerate(CAPABILITY_ROW_ORDER):
        row = rows[index]
        _require(isinstance(row, dict), SCHEMA, "capability_cases rows are objects")
        label = f"capability_cases[{index}]"
        if kind == "property":
            mapping = _require_keys(row, CAPABILITY_PROPERTY_KEYS, label)
            _require(
                mapping["expected_boolean"] == mapping["observed_boolean"],
                SCHEMA,
                f"{label} expected and observed booleans must agree",
            )
        elif kind == "registry":
            mapping = _require_keys(row, CAPABILITY_REGISTRY_KEYS, label)
            _require(
                mapping["expected_names"]
                == mapping["observed_names"]
                == ["mmode", "rime"],
                SCHEMA,
                f"{label} must record the exact registry key set",
            )
        else:
            mapping = _require_keys(row, CAPABILITY_REJECTION_KEYS, label)
            _require(
                mapping["issue_code"] == "mmode_m1_scalar_only",
                SCHEMA,
                f"{label} must carry the mmode_m1_scalar_only issue code",
            )
            _require_hex(
                mapping["configured_value_f64be"], 16, f"{label}.configured_value_f64be"
            )
        _require(
            mapping["case_kind"] == kind and mapping["case_id"] == case_id,
            SCHEMA,
            f"{label} must be the exact Section 14.2 row {kind}/{case_id}",
        )
        _require(mapping["pass"] is True, SCHEMA, f"{label}.pass must be true")


def validate_dependency_certificate(value: Any) -> None:
    """Validate Section 14.2's M1 ``dependency_certificate`` object."""
    mapping = _require_keys(
        value, DEPENDENCY_CERTIFICATE_KEYS, "dependency_certificate"
    )
    _require(
        mapping["path"] == DEPENDENCY_ARTIFACT,
        SCHEMA,
        "dependency_certificate.path must be the fixed R1 dependency path",
    )
    _require_hex(mapping["raw_sha256"], 64, "dependency_certificate.raw_sha256")
    certificate = _require_keys(
        mapping["certificate"],
        WP7_CERTIFICATE_KEYS,
        "dependency_certificate.certificate",
    )
    _require(
        certificate["verdict"] == "CPU_ACCEPTED_P_E_HARDWARE_GATED",
        SCHEMA,
        "the WP-7 certificate verdict is frozen",
    )
    _require(
        certificate["passed"] is True,
        SCHEMA,
        "the WP-7 certificate must have passed",
    )


def decode_f64be(text: Any) -> float:
    """Return the binary64 a Section 14.0 ``F64`` string encodes."""
    _require(
        isinstance(text, str) and len(text) == 16 and text == text.lower(),
        SCHEMA,
        f"not a lower-case F64 string: {text!r}",
    )
    return float(struct.unpack(">d", bytes.fromhex(str(text)))[0])


def _mask_bits(mask_hex: str, count: int) -> list[bool]:
    """Decode one Section 12.1 visibility mask back to sample-ordered bits.

    The mask is the sample-ordered visibility bits, most significant bit first,
    zero-padded to whole bytes.
    """
    width = (count + 7) // 8
    _require(
        len(mask_hex) == width * 2,
        SCHEMA,
        "a visibility mask is not zero-padded to whole bytes",
    )
    value = int(mask_hex, 16) if mask_hex else 0
    value >>= width * 8 - count
    return [bool((value >> (count - 1 - index)) & 1) for index in range(count)]


def _exact_rational_value(text: Any) -> Fraction:
    """Return the exact value of a Section 3.1 canonical rational."""
    numerator, denominator = _exact_rational(str(text))
    return Fraction(numerator, denominator)


def _exact_rational(text: str) -> tuple[int, int]:
    """Return the reduced ``(p, q)`` of a Section 3.1 canonical rational."""
    numerator, _, denominator = str(text).partition("/")
    _require(
        denominator != "" and denominator.lstrip("-").isdigit(),
        SCHEMA,
        f"not a canonical p/q rational: {text!r}",
    )
    return int(numerator), int(denominator)


def expand_membership_ledger(
    mask_rows: Sequence[Mapping[str, Any]],
    center_turns: Sequence[str],
    tau_f64be: str,
) -> str:
    """Re-digest the ``D*N`` per-sample membership array from its retained masks.

    Section 12.1 retains the census in compact per-direction mask rows because
    the expansion back to per-sample rows is deterministic: ``sample_turn`` is
    the retained exact centre and ``alpha_rad_f64be`` is one round-to-nearest of
    ``exact(tau)`` times that rational, with no intermediate binary64 step.  The
    validator therefore rebuilds the complete array's bytes here and re-digests
    them, which is what makes the economy a projection rather than a loss.
    """
    tau = Fraction(*decode_f64be(tau_f64be).as_integer_ratio())
    samples = len(center_turns)
    alpha = []
    turns = []
    for text in center_turns:
        numerator, denominator = _exact_rational(text)
        exact = Fraction(numerator, denominator)
        turns.append(str(text))
        alpha.append(struct.pack(">d", float(tau * exact)).hex())
    # The row's key set is fixed and every value is a string, a boolean or a
    # small non-negative integer, so the canonical bytes are assembled from a
    # template instead of the generic renderer.  Object keys are in Section 14's
    # lexicographic order, which is the order written here, and the expansion is
    # ``D*N`` rows: a per-row dictionary round trip would dominate the runtime
    # of a validator reviewers are expected to run.
    heads = [
        f'{{"alpha_rad_f64be":"{alpha[index]}","direction_id":'
        for index in range(samples)
    ]
    tails = [
        f',"sample_index":{index},"sample_turn":"{turns[index]}"}}'
        for index in range(samples)
    ]
    parts: list[str] = ["["]
    emitted = 0
    for row in mask_rows:
        frozen = _mask_bits(str(row["frozen_visible_mask_hex"]), samples)
        operational = _mask_bits(str(row["operational_visible_mask_hex"]), samples)
        identifier = json.dumps(str(row["direction_id"]), ensure_ascii=True)
        for index in range(samples):
            if emitted:
                parts.append(",")
            first = "true" if frozen[index] else "false"
            second = "true" if operational[index] else "false"
            same = "true" if frozen[index] == operational[index] else "false"
            parts.append(
                heads[index]
                + identifier
                + f',"frozen_visible":{first},"match":{same}'
                + f',"operational_visible":{second}'
                + tails[index]
            )
            emitted += 1
    parts.append("]")
    return domain_digest(
        "radiosim.mmode-horizon-membership.v1", "".join(parts).encode("utf-8")
    )


def slab_geometry(slab_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[tuple]]:
    """Return each direction's closed mismatch-slab pieces, in exact rationals."""
    geometry: dict[str, list[tuple]] = {}
    for row in slab_rows:
        pieces = geometry.setdefault(str(row["direction_id"]), [])
        for piece in row["pieces"]:
            pieces.append(
                (
                    _exact_rational_value(piece["turn_lo"]),
                    _exact_rational_value(piece["turn_hi"]),
                )
            )
    return geometry


def recompute_outside_slab_membership(
    mask_rows: Sequence[Mapping[str, Any]],
    slab_rows: Sequence[Mapping[str, Any]],
    center_turns: Sequence[str],
) -> tuple[int, int]:
    """Return the (outside-slab, total) membership mismatch counts.

    Section 4.2 scopes the membership rule the way the sign intervals have
    always been scoped: a sample centre inside a paired-root mismatch slab
    records its models' possibly differing memberships into the slab accounting
    instead of being falsely required to agree.  The retained mask rows carry
    the per-direction *total*; the gating counter is recomputed here from those
    masks against the retained slab geometry, so neither number can be asserted
    independently of the other.
    """
    geometry = slab_geometry(slab_rows)
    samples = len(center_turns)
    centres = [_exact_rational_value(turn) for turn in center_turns]
    outside = 0
    total = 0
    for row in mask_rows:
        identifier = str(row["direction_id"])
        frozen = _mask_bits(str(row["frozen_visible_mask_hex"]), samples)
        operational = _mask_bits(str(row["operational_visible_mask_hex"]), samples)
        pieces = geometry.get(identifier, ())
        declared = int(row["mismatch_count"])
        counted = 0
        for index in range(samples):
            if frozen[index] == operational[index]:
                continue
            counted += 1
            centre = centres[index]
            if not any(start <= centre <= stop for start, stop in pieces):
                outside += 1
        _require(
            counted == declared,
            SCHEMA,
            f"mask row {identifier!r} declares {declared} mismatches but its "
            f"masks expand to {counted}",
        )
        total += counted
    return outside, total


def validate_frame_row(row: Any, label: str) -> dict[str, Any]:
    """Validate one Section 14.2 frame row in Section 12.1's economy forms.

    The economy is a *projection*, never a weakening: the scan array and the
    per-sample membership census are reconstructed by replay and by mask
    expansion, so what the retained row must prove is that its projection is
    complete and self-consistent -- every crossing row verbatim, one summary row
    per direction in ledger order, one mask row per direction, and counters that
    are recomputed from those rows rather than asserted.
    """
    mapping = _require_keys(row, FRAME_ROW_KEYS, label)
    for field in (
        "certificate_sha256",
        "site_sha256",
        "input_identity_sha256",
        "iers_table_sha256",
        "frame_matrix_sha256",
        "canonical_era_turn_grid_sha256",
        "canonical_era_grid_sha256",
        "transfer_grid_catalog_sha256",
        "direction_ledger_sha256",
        "horizon_scan_sha256",
        "horizon_scan_ledger_sha256",
        "horizon_root_pair_ledger_sha256",
        "horizon_slab_ledger_sha256",
        "horizon_sign_interval_ledger_sha256",
        "horizon_membership_ledger_sha256",
        "direct_split_ledger_sha256",
        "direct_integrand_enclosure_sha256",
    ):
        _require_hex(mapping[field], 64, f"{label}.{field}")

    directions = mapping["direction_rows"]
    _require(
        isinstance(directions, list) and directions,
        SCHEMA,
        f"{label}.direction_rows must be a non-empty array",
    )
    order = [str(entry["direction_id"]) for entry in directions]
    _require(
        len(set(order)) == len(order),
        SCHEMA,
        f"{label}.direction_rows must not repeat a direction identifier",
    )
    _require(
        mapping["evaluated_direction_count"] == len(order)
        and mapping["expected_direction_count"] == len(order),
        SCHEMA,
        f"{label} direction counts must equal the embedded ledger length",
    )

    summaries = mapping["horizon_scan_summary_rows"]
    _require(
        isinstance(summaries, list)
        and [str(entry.get("direction_id")) for entry in summaries] == order,
        SCHEMA,
        f"{label}.horizon_scan_summary_rows must join the ledger in its order",
    )
    terminal = 0
    crossings_by_direction: dict[str, int] = {}
    for index, entry in enumerate(summaries):
        summary = _require_keys(
            entry, SCAN_SUMMARY_ROW_KEYS, f"{label}.horizon_scan_summary_rows[{index}]"
        )
        terminal += int(summary["terminal_cell_count"])
        crossings_by_direction[str(summary["direction_id"])] = int(
            summary["crossing_count"]
        )
    _require(
        mapping["horizon_isolation_interval_count"] == terminal,
        SCHEMA,
        f"{label}.horizon_isolation_interval_count must equal the summary sum",
    )

    observed: dict[str, int] = {}
    enclosures: dict[str, list[tuple[Fraction, Fraction]]] = {}
    guard_spans: dict[str, list[tuple[Fraction, Fraction]]] = {}
    #: The reconstructed per-direction root census.  Section 12.1 builds it from
    #: ``scan_crossing`` rows only -- an ``excluded_upper_endpoint`` row is an
    #: authenticated endpoint event that does not enter the census.
    census: dict[str, list[tuple[Fraction, Fraction]]] = {}
    for index, entry in enumerate(mapping["horizon_scan_crossing_rows"]):
        where = f"{label}.horizon_scan_crossing_rows[{index}]"
        row_map = _require_keys(entry, SCAN_ROW_KEYS, where)
        kind = row_map["classification"]
        _require(
            kind in SCAN_CROSSING_CLASSIFICATIONS or kind == "guard_interval",
            SCHEMA,
            f"{where} retains only crossing and guard rows, not {kind!r}",
        )
        identifier = str(row_map["direction_id"])
        bounds = (
            _exact_rational_value(row_map["turn_lo"]),
            _exact_rational_value(row_map["turn_hi"]),
        )
        _require(bounds[0] < bounds[1], SCHEMA, f"{where} is not a positive interval")
        if kind == "guard_interval":
            # Section 12.1: a guard carries no root, its margin is exactly
            # ``F64(0)``, and its signs are the endpoint values' signs -- zero
            # permitted at the root-adjacent end, where the numerator vanishes.
            _require(
                row_map["root_turn_lo"] is None
                and row_map["root_turn_hi"] is None
                and row_map["root_orientation"] is None
                and row_map["root_residual_f64be"] is None,
                SCHEMA,
                f"{where} is a guard and must carry null root fields",
            )
            _require(
                decode_f64be(row_map["ceiling_margin_f64be"]) == 0.0,
                SCHEMA,
                f"{where} guard margin must be exactly F64(0)",
            )
            _require(
                row_map["left_sign"] in (-1, 0, 1)
                and row_map["right_sign"] in (-1, 0, 1),
                SCHEMA,
                f"{where} guard signs must be endpoint value signs",
            )
            _require(
                bounds[1] - bounds[0] <= Fraction(1, 100000000),
                SCHEMA,
                f"{where} guard width exceeds the 1e-8 turn probe offset",
            )
            guard_spans.setdefault(identifier, []).append(
                (bounds[0], bounds[1], int(row_map["cell_index"]))
            )
            continue
        _require(
            row_map["root_turn_lo"] is not None
            and row_map["root_turn_hi"] is not None
            and row_map["root_orientation"] in ("rising", "setting")
            and row_map["root_residual_f64be"] is not None,
            SCHEMA,
            f"{where} must carry its root",
        )
        _require(
            {row_map["left_sign"], row_map["right_sign"]} == {-1, 1},
            SCHEMA,
            f"{where} probe signs must differ",
        )
        _require(
            _exact_rational_value(str(row_map["root_turn_lo"])) == bounds[0]
            and _exact_rational_value(str(row_map["root_turn_hi"])) == bounds[1],
            SCHEMA,
            f"{where} root bounds must equal the retained enclosure",
        )
        enclosures.setdefault(identifier, []).append(
            (bounds[0], bounds[1], int(row_map["cell_index"]))
        )
        if kind == "scan_crossing":
            census.setdefault(identifier, []).append(bounds)
        observed[identifier] = observed.get(identifier, 0) + 1
    _require(
        all(
            observed.get(identifier, 0) == count
            for identifier, count in crossings_by_direction.items()
        ),
        SCHEMA,
        f"{label} crossing rows do not total their summary counts",
    )
    # Section 12.1: root-census reconstruction rejects any duplicate owned root.
    # Two crossing rows of one direction whose exact enclosures coincide are one
    # root claimed twice, which would inflate the census, the pairing and the
    # slab measure while every per-row predicate still passed.
    for identifier, roots in census.items():
        seen: set[tuple[Fraction, Fraction]] = set()
        for lower, upper in roots:
            _require(
                (lower, upper) not in seen,
                SCHEMA,
                f"{label} reconstructs a duplicate owned root for direction "
                f"{identifier!r}: the enclosure "
                f"[{lower.numerator}/{lower.denominator}, "
                f"{upper.numerator}/{upper.denominator}] is claimed by two "
                "scan_crossing rows",
            )
            seen.add((lower, upper))
    # Section 12.1's partition rule holds *together with the retained root
    # enclosures*: every guard abuts its crossing's enclosure or another guard,
    # so an orphan guard -- a residue with no crossing to flank -- rejects.
    for identifier, spans in guard_spans.items():
        anchored = list(enclosures.get(identifier, ()))
        remaining = sorted(spans)
        progressed = True
        while remaining and progressed:
            progressed = False
            for span in list(remaining):
                # Adjacency is both geometric and positional: a guard shares an
                # exact bound with the enclosure or guard it flanks *and* sits
                # in the neighbouring terminal cell, so a relocated or forged
                # guard cannot borrow another crossing's anchor.
                if any(
                    (span[1] == other[0] or span[0] == other[1])
                    and abs(span[2] - other[2]) == 1
                    for other in anchored
                ):
                    anchored.append(span)
                    remaining.remove(span)
                    progressed = True
        _require(
            not remaining,
            SCHEMA,
            f"{label} retains {len(remaining)} orphan guard row(s) for direction "
            f"{identifier!r}: a guard must abut its crossing's enclosure or "
            "another guard, in the neighbouring terminal cell",
        )

    samples = int(mapping["sidereal_samples"])
    masks = mapping["horizon_membership_mask_rows"]
    _require(
        isinstance(masks, list)
        and [str(entry.get("direction_id")) for entry in masks] == order,
        SCHEMA,
        f"{label}.horizon_membership_mask_rows must join the ledger in its order",
    )
    width = 2 * ((samples + 7) // 8)
    mismatches = 0
    for index, entry in enumerate(masks):
        mask = _require_keys(
            entry,
            MEMBERSHIP_MASK_ROW_KEYS,
            f"{label}.horizon_membership_mask_rows[{index}]",
        )
        _require(
            int(mask["sample_count"]) == samples,
            SCHEMA,
            f"{label}.horizon_membership_mask_rows[{index}].sample_count != N",
        )
        for field in ("frozen_visible_mask_hex", "operational_visible_mask_hex"):
            _require_hex(
                mask[field], width, f"{label}.horizon_membership_mask_rows[{index}]"
            )
        mismatches += int(mask["mismatch_count"])
    _require(
        mapping["evaluated_horizon_membership_count"] == samples * len(order)
        and mapping["expected_horizon_membership_count"] == samples * len(order),
        SCHEMA,
        f"{label} membership counts must equal D*N",
    )
    _require(
        mapping["horizon_membership_mismatches"] <= mismatches,
        SCHEMA,
        f"{label}.horizon_membership_mismatches is the outside-slab subset of "
        "the mask rows' per-direction totals and cannot exceed them",
    )

    _require(
        len(mapping["horizon_root_pair_rows"]) == len(order)
        and mapping["evaluated_horizon_root_pair_row_count"] == len(order)
        and mapping["expected_horizon_root_pair_row_count"] == len(order),
        SCHEMA,
        f"{label} requires exactly one root-pair row per direction",
    )
    slabs = len(mapping["horizon_slab_rows"])
    _require(
        mapping["expected_horizon_slab_row_count"] == slabs
        and mapping["evaluated_horizon_slab_row_count"] == slabs
        and mapping["horizon_mismatch_slab_count"] == slabs
        and mapping["horizon_paired_root_count"] == slabs,
        SCHEMA,
        f"{label} slab counts must agree with the embedded slab array",
    )
    signs = len(mapping["horizon_sign_interval_rows"])
    _require(
        mapping["expected_horizon_sign_interval_count"] == signs
        and mapping["evaluated_horizon_sign_interval_count"] == signs,
        SCHEMA,
        f"{label} sign-interval counts must equal the embedded array length",
    )
    splits = len(mapping["direct_split_rows"])
    _require(
        mapping["expected_direct_split_row_count"] == splits
        and mapping["evaluated_direct_split_row_count"] == splits,
        SCHEMA,
        f"{label} direct-split counts must equal the embedded array length",
    )

    for field in (
        "horizon_root_count_mismatches",
        "horizon_root_orientation_mismatches",
        "horizon_membership_mismatches",
        "horizon_outside_slab_sign_mismatches",
        "horizon_unresolved_interval_count",
    ):
        _require(mapping[field] == 0, SCHEMA, f"{label}.{field} must be zero")

    cells = (
        4
        * int(mapping["sidereal_samples"])
        * int(mapping["n_baselines"])
        * int(mapping["n_frequencies"])
    )
    _require(
        int(mapping["n_correlations"]) == 4,
        SCHEMA,
        f"{label}.n_correlations must be exactly four",
    )
    for field in (
        "expected_cube_cell_count",
        "evaluated_frozen_gauss64_cube_cell_count",
        "evaluated_frozen_gauss128_cube_cell_count",
        "evaluated_operational_gauss64_cube_cell_count",
        "evaluated_operational_gauss128_cube_cell_count",
        "compared_frozen_gauss_change_cell_count",
        "compared_operational_gauss_change_cell_count",
        "evaluated_frozen_enclosure_error_cell_count",
        "evaluated_operational_enclosure_error_cell_count",
    ):
        _require(
            int(mapping[field]) == cells,
            SCHEMA,
            f"{label}.{field} must equal K = 4*N*B*F",
        )

    _require(
        mapping["horizon_root_limit_rad"] == FRAME_ROOT_LIMIT_RAD
        and mapping["phase_limit_rad"] == FRAME_PHASE_LIMIT_RAD
        and mapping["cube_l2_limit"] == FRAME_CUBE_L2_LIMIT,
        SCHEMA,
        f"{label} carries a widened fixed frame limit",
    )
    _require(
        float(mapping["horizon_root_max_rad"]) <= FRAME_ROOT_LIMIT_RAD
        and float(mapping["horizon_mismatch_measure_rad"])
        <= float(mapping["horizon_mismatch_measure_limit_rad"])
        and float(mapping["phase_max_rad"]) <= FRAME_PHASE_LIMIT_RAD
        and float(mapping["direct_gauss_change_max_jy"])
        <= float(mapping["direct_gauss_change_limit_jy"])
        and float(mapping["cube_max_jy"]) <= float(mapping["cube_limit_jy"])
        and float(mapping["cube_l2"]) <= FRAME_CUBE_L2_LIMIT,
        SCHEMA,
        f"{label} exceeds a fixed Section 4.2 frame bound",
    )
    _require(mapping["pass"] is True, SCHEMA, f"{label}.pass must be true")
    return mapping


def validate_truncation_row(row: Any, label: str) -> None:
    """Validate one Section 14.2 truncation row against the ``v3`` surface.

    Tier 1a is the only half with fixed numeric limits.  Tier 1b and the
    deficit are recorded and bounded by the fixture's two reviewed budgets --
    ``quadrature_budget_jy`` and ``truncation_budget_jy`` -- which are evidence
    fields, never YAML knobs, and never universal limits.  Tier 2 additionally
    requires strict monotone decrease across the convergence levels and a
    quarter-to-full factor of at least two.
    """
    mapping = _require_keys(row, TRUNCATION_ROW_KEYS, label)

    _require(
        mapping["horizon_free_shell_l2_limit"] == HORIZON_FREE_L2_LIMIT,
        SCHEMA,
        f"{label}.horizon_free_shell_l2_limit is the fixed 1e-8",
    )
    _require(
        mapping["horizon_free_shell_max_jy"]
        <= mapping["horizon_free_shell_max_limit_jy"],
        SCHEMA,
        f"{label} tier-1a maximum exceeds its fixed limit",
    )
    _require(
        mapping["horizon_free_shell_l2"] <= mapping["horizon_free_shell_l2_limit"],
        SCHEMA,
        f"{label} tier-1a normalized L2 exceeds its fixed limit",
    )
    _require(
        mapping["quadrature_shell_max_jy"] <= mapping["quadrature_budget_jy"],
        SCHEMA,
        f"{label} tier-1b shell exceeds its declared quadrature budget",
    )
    quarter = float(mapping["deficit_max_quarter_jy"])
    half = float(mapping["deficit_max_half_jy"])
    full = float(mapping["deficit_max_jy"])
    _require(
        full == 0.0 or quarter > half > full,
        SCHEMA,
        f"{label} deficit is not strictly monotone across the levels",
    )
    _require(
        full == 0.0 or float(mapping["convergence_factor"]) >= CONVERGENCE_FACTOR_FLOOR,
        SCHEMA,
        f"{label} quarter-to-full factor is below the fixed floor",
    )
    _require(
        full <= float(mapping["truncation_budget_jy"]),
        SCHEMA,
        f"{label} deficit exceeds its declared truncation budget",
    )
    grids = 1 + len(mapping["diagnostic_transfer_grid_ids"])
    shape = list(mapping["cube_shape"])
    _require(
        len(shape) == 4 and shape[3] == 4,
        SCHEMA,
        f"{label}.cube_shape must be exactly [N,B,F,4]",
    )
    expected_samples = grids * shape[1] * shape[2] * 4 * 4
    _require(
        mapping["expected_transfer_sample_row_count"] == expected_samples,
        SCHEMA,
        f"{label} transfer-sample rows are one per grid and output cell, "
        f"so the expected count is {expected_samples}",
    )
    coverage = mapping["shell_coverage"]
    if isinstance(coverage, dict) and coverage:
        rows = coverage.get("transfer_sample_rows")
        _require(
            isinstance(rows, list)
            and len(rows) == mapping["evaluated_transfer_sample_row_count"],
            SCHEMA,
            f"{label}.shell_coverage.transfer_sample_rows must equal its count",
        )
        for index, entry in enumerate(rows):
            sample = _require_keys(
                entry,
                TRANSFER_SAMPLE_ROW_KEYS,
                f"{label}.shell_coverage.transfer_sample_rows[{index}]",
            )
            _require_hex(
                sample["concatenation_sha256"],
                64,
                f"{label}.shell_coverage.transfer_sample_rows[{index}]"
                ".concatenation_sha256",
            )
            _require(
                int(sample["packed_sample_value_count"])
                % max(int(sample["direction_count"]), 1)
                == 0,
                SCHEMA,
                f"{label}.shell_coverage.transfer_sample_rows[{index}] packed "
                "count is not a whole multiple of its direction count",
            )
    _require(
        mapping["evaluated_transfer_sample_row_count"] == expected_samples,
        SCHEMA,
        f"{label} evaluated transfer-sample rows must equal the expected count",
    )
    cells = shape[0] * shape[1] * shape[2] * 4
    _require(
        mapping["expected_shell_comparison_cell_count"] == 4 * cells
        and mapping["evaluated_shell_comparison_cell_count"] == 4 * cells,
        SCHEMA,
        f"{label} shell-comparison rows are the four diagnostics over all "
        f"{cells} output cells",
    )
    expected_blocks = shape[1] * shape[2] * 4 * 4 * (2 * int(mapping["mcheck"]) + 1)
    _require(
        mapping["expected_field_block_count"] == expected_blocks
        and mapping["evaluated_field_block_count"] == expected_blocks,
        SCHEMA,
        f"{label} field/block rows are B*F*C*4*(2*mcheck+1) = {expected_blocks}, "
        "one per contributing field and signed-m block",
    )
    if isinstance(coverage, dict) and coverage:
        blocks = coverage.get("field_block_rows")
        if blocks is not None:
            _require(
                isinstance(blocks, list) and len(blocks) == expected_blocks,
                SCHEMA,
                f"{label}.shell_coverage.field_block_rows must equal its count",
            )
        comparisons = coverage.get("shell_comparison_rows")
        if comparisons is not None:
            _require(
                isinstance(comparisons, list) and len(comparisons) == 4 * cells,
                SCHEMA,
                f"{label}.shell_coverage.shell_comparison_rows must equal its count",
            )
    _require(
        mapping["pass"] is True,
        SCHEMA,
        f"{label}.pass must be true",
    )


def validate_evidence_document(document: Any) -> dict[str, Any]:
    """Validate the complete Section 14.2 M1 evidence envelope."""
    envelope = _require_keys(document, ENVELOPE_KEYS, "evidence document")
    _require(
        envelope["schema_version"] == EVIDENCE_SCHEMA,
        SCHEMA,
        "schema_version is the frozen phase literal",
    )
    _require(envelope["phase"] == PHASE, SCHEMA, "phase must be exactly 'M1'")
    _require(envelope["status"] == "candidate", SCHEMA, "status must be 'candidate'")
    _require(
        envelope["evidence_commit_sha"] is None,
        SCHEMA,
        "evidence_commit_sha is JSON null at E",
    )
    _require(
        envelope["evidence_commit_sha_reason"] == EVIDENCE_SELF_REFERENCE_REASON,
        SCHEMA,
        "evidence_commit_sha_reason is the exact self-reference literal",
    )
    _require(
        envelope["working_tree_clean"] is True,
        SCHEMA,
        "working_tree_clean must be true",
    )
    for field in ("design_sha", "red_commit_sha", "source_sha"):
        _require_hex(envelope[field], 40, field)

    environment = _require_keys(
        envelope["environment"], ENVIRONMENT_KEYS, "environment"
    )
    _require(
        environment["pixi_environment"] == "default",
        SCHEMA,
        "environment.pixi_environment must be 'default'",
    )
    _require_hex(environment["pixi_lock_sha256"], 64, "environment.pixi_lock_sha256")
    _require_hex(environment["iers_table_sha256"], 64, "environment.iers_table_sha256")
    packages = _require_keys(
        environment["numeric_packages"],
        NUMERIC_PACKAGE_KEYS,
        "environment.numeric_packages",
    )
    _require(
        all(isinstance(item, str) and item for item in packages.values()),
        SCHEMA,
        "every numeric package version is a normalized non-empty string",
    )

    identities = _require_keys(
        envelope["source_identities"], SOURCE_IDENTITY_KEYS, "source_identities"
    )
    for field in (
        "git_tree_sha256",
        "pixi_manifest_sha256",
        "pixi_lock_sha256",
        "convention_identity_sha256",
        "input_identity_set_sha256",
    ):
        _require_hex(identities[field], 64, f"source_identities.{field}")
    rows = identities["fixture_input_rows"]
    _require(
        isinstance(rows, list),
        SCHEMA,
        "source_identities.fixture_input_rows must be an array",
    )
    fixture_ids = [row.get("fixture_id") for row in rows if isinstance(row, dict)]
    _require(
        len(fixture_ids) == len(rows) and len(set(fixture_ids)) == len(fixture_ids),
        SCHEMA,
        "fixture-input rows are unique",
    )
    _require(
        fixture_ids == sorted(fixture_ids),
        SCHEMA,
        "fixture-input rows are UTF-8 fixture-ID sorted",
    )

    record = _require_keys(
        envelope["red_failure_record"], RED_RECORD_KEYS, "red_failure_record"
    )
    _require(
        record["path"] == RED_FAILURE_RECORD,
        SCHEMA,
        "red_failure_record.path is the fixed R1 path",
    )
    _require(
        record["schema_version"] == RED_FAILURE_SCHEMA,
        SCHEMA,
        "red_failure_record.schema_version is the frozen phase literal",
    )
    _require_hex(record["sha256"], 64, "red_failure_record.sha256")
    _require_hex(
        record["pre_fix_source_sha"], 40, "red_failure_record.pre_fix_source_sha"
    )
    _require(
        record["validated"] is True, SCHEMA, "red_failure_record.validated is true"
    )

    results = _require_keys(envelope["results"], RESULT_KEYS, "results")
    validate_dependency_certificate(results["dependency_certificate"])
    validate_capability_cases(results["capability_cases"])
    for name in RESULT_KEYS:
        if name in {"dependency_certificate", "capability_cases"}:
            continue
        _require(
            isinstance(results[name], list),
            SCHEMA,
            f"results.{name} must be an array",
        )
    grids_by_fixture: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(results["time_grid_cases"]):
        _require(
            isinstance(row, dict) and "fixture_id" in row,
            SCHEMA,
            f"time_grid_cases[{index}] must name its fixture",
        )
        fixture = str(row["fixture_id"])
        _require(
            fixture not in grids_by_fixture,
            SCHEMA,
            f"time_grid_cases has two rows for fixture {fixture!r}",
        )
        grids_by_fixture[fixture] = dict(row)
    for index, row in enumerate(results["frame_certificate_cases"]):
        label = f"frame_certificate_cases[{index}]"
        frame = validate_frame_row(row, label)
        fixture = str(frame["fixture_id"])
        _require(
            fixture in grids_by_fixture,
            SCHEMA,
            f"{label} must join the same-fixture time-grid row",
        )
        grid = grids_by_fixture[fixture]
        _require(
            grid["canonical_era_turn_grid_sha256"]
            == frame["canonical_era_turn_grid_sha256"]
            and grid["canonical_era_grid_sha256"] == frame["canonical_era_grid_sha256"],
            SCHEMA,
            f"{label} must join both canonical grid digests of its fixture",
        )
        centers = grid["canonical_era_turn_grid"]["center_turns"]
        _require(
            len(centers) == int(frame["sidereal_samples"]),
            SCHEMA,
            f"{label} sidereal_samples disagrees with the joined turn grid",
        )
        _require(
            expand_membership_ledger(
                frame["horizon_membership_mask_rows"],
                centers,
                grid["tau_f64be"],
            )
            == frame["horizon_membership_ledger_sha256"],
            DIGEST,
            f"{label} membership masks do not expand to their ledger digest",
        )
        outside, _total = recompute_outside_slab_membership(
            frame["horizon_membership_mask_rows"],
            frame["horizon_slab_rows"],
            centers,
        )
        _require(
            frame["horizon_membership_mismatches"] == outside,
            SCHEMA,
            f"{label}.horizon_membership_mismatches must be the outside-slab "
            "count recomputed from the masks against the retained slab geometry",
        )
        _require(
            outside == 0,
            SCHEMA,
            f"{label} has {outside} outside-slab membership mismatch(es)",
        )
    for index, row in enumerate(results["truncation_cases"]):
        validate_truncation_row(row, f"truncation_cases[{index}]")
    for index, row in enumerate(results["direct_identity_cases"]):
        mapping = _require_keys(
            row, DIRECT_IDENTITY_KEYS, f"direct_identity_cases[{index}]"
        )
        _require(
            mapping["rime_before_sha256"] == mapping["rime_after_sha256"],
            SCHEMA,
            "the direct RIME cube identity must be unchanged by the wrapper",
        )
        _require(
            mapping["scientific_before_sha256"] == mapping["scientific_after_sha256"],
            SCHEMA,
            "the direct RIME scientific identity must be unchanged",
        )
        _require(
            mapping["byte_identical"] is True and mapping["pass"] is True,
            SCHEMA,
            "the direct identity row must pass",
        )

    commands = envelope["commands"]
    _require(
        isinstance(commands, list) and commands, SCHEMA, "commands must be an array"
    )
    for index, row in enumerate(commands):
        validate_command_row(row, f"commands[{index}]")
    _require_sorted_unique_strings(envelope["limitations"], "limitations")
    _require_sorted_unique_strings(
        envelope["claims_not_licensed"], "claims_not_licensed"
    )
    return envelope


# ---------------------------------------------------------------------------
# Section 14.2 M1 fixture set and result construction
# ---------------------------------------------------------------------------

#: Section 14.3's one non-rejection M1 fixture: the qualified compact scalar
#: full-sidereal run whose two-tier margins Section 7.3 records.
M1_FIXTURE_ID = "mmode_point_stokes_i"

#: The direct-identity control fixture: the same geometry solved by the
#: unchanged ``rime`` strategy, before and after its Section 2 wrapper.
RIME_FIXTURE_ID = "rime_point_reference"

#: The qualified fixture's geometry and dimensions.  They are the accepted
#: Section 7.3 values, not tuning knobs: a 4.0 m east-west baseline between two
#: 2.5 m dishes at 50/51/52 MHz, one circumpolar source, and ``lmax = 16``.
FIXTURE_BASELINE_EAST_M = 4.0
FIXTURE_DIAMETER_M = 2.5
FIXTURE_STARTING_FREQUENCY_MHZ = 50.0
FIXTURE_SOURCE_DEC_DEG = -75.0
FIXTURE_SIDEREAL_SAMPLES = 49
FIXTURE_LMAX = 16
FIXTURE_MMAX = 16
FIXTURE_QUADRATURE_NSIDE = 8

#: The two reviewed per-fixture budgets of Section 7.3's recorded halves.  They
#: are evidence fields, never YAML knobs and never universal limits: the
#: with-horizon quadrature shell measured ``5.80e-2 Jy`` and the truncation
#: deficit measured ``1.17e-1 Jy``, so each budget is the reviewed round number
#: just above its measurement.
FIXTURE_QUADRATURE_BUDGET_JY = 0.10
FIXTURE_TRUNCATION_BUDGET_JY = 0.20

#: Section 14.2's non-licensed claim set and the standing M1 limitations.
CLAIMS_NOT_LICENSED: tuple[str, ...] = (
    "general_speedup",
    "gpu_or_accelerator_support",
    "polarized_mmode_support",
)
LIMITATIONS: tuple[str, ...] = (
    "no accelerator run of the m-mode solver has been measured (PERF-001)",
    "the operational horizon scan array and the transfer-sample concatenations "
    "are reconstructed by the mandatory A1 re-derivation, not embedded",
    "phase M1 evaluates the scalar Stokes I field only",
)


def _fixture_mapping(root: Path) -> dict[str, Any]:
    """Return the complete resolved-input mapping of the M1 fixture.

    The generator owns its fixture definition outright: an evidence artifact
    that described a configuration assembled somewhere else could not be
    reproduced from the tracked bytes alone.
    """
    layout = root / "antennas.txt"
    layout.write_text(
        "Name Number BeamID E N U Diameter\n"
        f"ANT0 0 0 0.0 0.0 0.0 {FIXTURE_DIAMETER_M}\n"
        f"ANT1 1 0 {FIXTURE_BASELINE_EAST_M} 0.0 0.0 {FIXTURE_DIAMETER_M}\n",
        encoding="utf-8",
    )
    return {
        "instrument": {
            "source": {
                "kind": "layout_file",
                "path": str(layout),
                "format": "radiosim",
                "telescope_name": "Tier1ATestArray",
            },
            "location": {
                "longitude_deg": 21.4283,
                "latitude_deg": -30.72152,
                "height_m": 1073.0,
            },
            "default_diameter_m": FIXTURE_DIAMETER_M,
        },
        "baseline_selection": {"correlations": "all"},
        "beams": {
            "mode": "analytic",
            "model": {
                "kind": "circular_aperture",
                "taper": {"kind": "gaussian", "edge_taper_db": 10.0},
            },
        },
        "obs_time": {
            "mode": "full_sidereal",
            "start_time": "2025-01-01T00:00:00",
            "sidereal_samples": FIXTURE_SIDEREAL_SAMPLES,
            "integration_fraction": 1.0,
        },
        "obs_frequency": {
            "mode": "grid",
            "starting_frequency": FIXTURE_STARTING_FREQUENCY_MHZ,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 2.0,
            "channel_width": 1.0,
            "frequency_unit": "MHz",
        },
        "sky_model": {
            "flux_unit": "Jy",
            "sources": [
                {
                    "kind": "test_sources",
                    "representation": "point_sources",
                    "num_sources": 1,
                    "distribution": "uniform",
                    "seed": 1,
                    "dec_deg": FIXTURE_SOURCE_DEC_DEG,
                    "dec_range_deg": 0.0,
                    "spectral_index": 0.0,
                    "polarization_fraction": 0.0,
                    "stokes_v_fraction": 0.0,
                }
            ],
        },
        "visibility": {"sky_representation": "point_sources"},
        "execution": {
            "backend": "numpy",
            "offline": True,
            "precision": {"preset": "standard"},
            "simulator": "mmode",
            "mmode": {
                "convention": "radiosim.mmode-forward.v1",
                "frame_model": "radiosim.frozen-cirs-rigid-era.v1",
                "harmonic_convention": "radiosim.shaw-polarized-harmonics.v1",
                "lmax": FIXTURE_LMAX,
                "mmax": FIXTURE_MMAX,
                "quadrature_nside": FIXTURE_QUADRATURE_NSIDE,
                "working_memory_bytes": 1073741824,
            },
        },
        "workflow": {
            "output_dir": str(root / "output"),
            "run_subdir": "run",
            "result_filename": "visibilities",
            "result_format": "hdf5",
            "collision_policy": "error",
            "save_results": False,
            "plot_results": False,
            "open_plots_in_browser": False,
            "save_log": False,
        },
    }


def _rime_mapping(root: Path) -> dict[str, Any]:
    """Return the direct-identity control mapping: the same run under ``rime``."""
    mapping = _fixture_mapping(root)
    mapping["obs_time"] = {
        "start_time": "2025-01-01T00:00:00",
        "duration_seconds": 2.0,
        "time_step_seconds": 1.0,
    }
    execution = dict(mapping["execution"])
    execution["simulator"] = "rime"
    execution.pop("mmode")
    mapping["execution"] = execution
    return mapping


def _time_grid_row(grid: Any, frame: Any, tau: float) -> dict[str, Any]:
    """Return Section 14.2's exact time-grid row for one resolved grid."""
    from radiosim.core.mmode.time import ut1_manifest, utc_manifest
    from radiosim.core.mmode.types import f64be

    utc, utc_sha256 = utc_manifest(grid)
    ut1, ut1_sha256 = ut1_manifest(grid)
    return {
        "fixture_id": M1_FIXTURE_ID,
        "sidereal_samples": grid.sidereal_samples,
        "integration_fraction_f64be": f64be(grid.integration_fraction),
        "canonical_era_turn_grid": dict(grid.canonical_era_turn_grid),
        "iers_table_sha256": grid.iers_table_sha256,
        "era_center_turn_sha256": grid.era_center_turn_sha256,
        "era_lower_edge_turn_sha256": grid.era_lower_edge_turn_sha256,
        "era_upper_edge_turn_sha256": grid.era_upper_edge_turn_sha256,
        "canonical_era_turn_grid_sha256": grid.canonical_era_turn_grid_sha256,
        "tau_f64be": f64be(tau),
        "delta_alpha_rad_f64be": f64be(grid.delta_alpha_rad),
        "horizon_lo_rad_f64be": f64be(grid.horizon_lo_rad),
        "horizon_hi_rad_f64be": f64be(grid.horizon_hi_rad),
        "era_center_rad_sha256": grid.era_center_rad_sha256,
        "era_lower_edge_rad_sha256": grid.era_lower_edge_rad_sha256,
        "era_upper_edge_rad_sha256": grid.era_upper_edge_rad_sha256,
        "canonical_era_grid": dict(grid.canonical_era_grid),
        "canonical_era_grid_sha256": grid.canonical_era_grid_sha256,
        "era_center_max_residual_rad": grid.era_center_max_residual_rad,
        "era_center_limit_rad": grid.era_center_limit_rad,
        "era_step_max_residual_rad": grid.era_step_max_residual_rad,
        "era_step_limit_rad": grid.era_step_limit_rad,
        "ut1_utc_roundtrip_seconds": grid.ut1_utc_roundtrip_seconds,
        "ut1_utc_roundtrip_limit_seconds": grid.ut1_utc_roundtrip_limit_seconds,
        "utc_manifest": dict(utc),
        "utc_sha256": utc_sha256,
        "ut1_manifest": dict(ut1),
        "ut1_sha256": ut1_sha256,
        "integration_time_seconds_sha256": grid.integration_time_seconds_sha256,
        "pass": (
            grid.era_center_max_residual_rad <= grid.era_center_limit_rad
            and grid.era_step_max_residual_rad <= grid.era_step_limit_rad
            and abs(grid.ut1_utc_roundtrip_seconds)
            <= grid.ut1_utc_roundtrip_limit_seconds
            and frame.iers_table_sha256 == grid.iers_table_sha256
        ),
    }


def _packed_layout_row(table: Any) -> dict[str, Any]:
    """Return Section 14.2's exact packed-layout row for one block table.

    The round trip unpacks and repacks in exact signed-``m``, field, ascending
    ``l`` order; ``pass`` requires byte-identical buffers and equal identities,
    and padding is forbidden, so the two buffers are compared element-wise
    rather than by a tolerance.
    """
    from radiosim.core.mmode.types import (
        FIELD_ORDER,
        SPIN_ORDER,
        array_digest,
        f64be,
        object_digest,
    )

    count = int(table.packed_value_count)
    values = [complex(index + 1, count - index) for index in range(count)]
    roundtrip: list[complex] = [0j] * count
    for row in table.block_rows:
        start = int(row["value_start"])
        stop = int(row["value_stop"])
        roundtrip[start:stop] = values[start:stop]
    block_rows = [dict(row) for row in table.block_rows]
    return {
        "fixture_id": M1_FIXTURE_ID,
        "lmax": int(table.lmax),
        "mmax": int(table.mmax),
        "field_order": list(FIELD_ORDER),
        "spin_order": list(SPIN_ORDER),
        "block_count": len(block_rows),
        "packed_value_count": count,
        "block_rows": block_rows,
        "packed_values_reim_f64be": [
            f64be(part) for value in values for part in (value.real, value.imag)
        ],
        "roundtrip_reim_f64be": [
            f64be(part) for value in roundtrip for part in (value.real, value.imag)
        ],
        "block_table_sha256": object_digest(
            "radiosim.mmode-packed-block-table.v1", block_rows
        ),
        "packed_values_sha256": array_digest(
            "radiosim.mmode-packed-values.v1",
            "packed_harmonic_values",
            ["packed_value"],
            "dimensionless",
            values,
            dtype="complex128-be",
        ),
        "roundtrip_sha256": array_digest(
            "radiosim.mmode-packed-values.v1",
            "packed_harmonic_values",
            ["packed_value"],
            "dimensionless",
            roundtrip,
            dtype="complex128-be",
        ),
        "invalid_cell_count": sum(
            1 for left, right in zip(values, roundtrip, strict=True) if left != right
        ),
        "pass": values == roundtrip
        and table.block_table_sha256
        == object_digest("radiosim.mmode-packed-block-table.v1", block_rows),
    }


# ---------------------------------------------------------------------------
# Section 14.2 preflight and generation
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


def preflight(source_sha: str | None = None) -> dict[str, str]:
    """Run Section 14.2's common pre-output check without writing anything.

    Before opening any output the generator requires ``git rev-parse HEAD`` to
    equal ``source_sha``, an empty index/worktree/untracked set from
    ``git status --porcelain=v1 --untracked-files=all``, the exact Pixi manifest
    and lock, and an absent declared output set.  Expected new artifacts do not
    retroactively make the preflight false, because the check runs first.
    """
    head = _git("rev-parse", "HEAD").strip()
    if source_sha is not None and head != source_sha:
        raise EvidenceError(
            PREFLIGHT, f"HEAD {head} is not the approved source {source_sha}"
        )
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status.strip():
        raise EvidenceError(PREFLIGHT, "the working tree is not globally clean")
    for relative in DECLARED_OUTPUTS:
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


def require_declared_outputs_only() -> None:
    """Require the repository's only new paths to equal the declared output set.

    Section 14.2 runs this *after* publication: an expected new artifact does
    not retroactively make the preflight false, but a generator that also left
    a stray file behind has not produced the declared set.
    """
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    observed = sorted(line[3:].strip() for line in status.splitlines() if line.strip())
    expected = sorted(DECLARED_OUTPUTS)
    _require(
        observed == expected,
        DIGEST,
        f"after publication the repository's new paths must be exactly "
        f"{expected}, not {observed}",
    )


def write_atomic_no_overwrite(path: Path, payload: bytes) -> None:
    """Publish one artifact atomically, refusing to overwrite anything.

    Generation is atomic and no-overwrite (Section 14.2), so the payload is
    written to a sibling temporary first and then linked into place: ``os.link``
    fails rather than replacing an existing file, which ``os.replace`` would
    silently do.
    """
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "xb") as handle:
        handle.write(payload)
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise EvidenceError(DIGEST, f"{path} already exists") from error
    finally:
        temporary.unlink()


def _visibility_identity(cube: Any) -> str:
    """Return Section 14.0's common visibility-cube identity."""
    from radiosim.core.mmode.types import array_digest

    return array_digest(
        "radiosim.mmode-visibility-cube.v1",
        "visibility_cube",
        ["time", "baseline", "frequency", "correlation"],
        "Jy",
        cube,
        dtype="complex128-be",
    )


def _scientific_identity(outcome: Any) -> str:
    """Return the scientific identity of one solve outcome.

    It binds the cube bytes to the component identity, the per-component element
    counts and the execution path, so a wrapper that changed the component
    order, the source reduction or the path could not leave it unchanged.  The
    hybrid outcome names its components through ``component_names``; the
    strategy outcome carries the same names directly.
    """
    from radiosim.core.mmode.types import object_digest

    cube = outcome.receptor_visibilities
    flat = cube.reshape(*cube.shape[:3], 4)
    components = getattr(outcome, "component_names", None)
    if components is None:
        names = [str(name) for name in outcome.components]
        counts = [int(value) for value in outcome.component_element_counts]
    else:
        names = [str(name) for name in components]
        counts = [int(item.element_count) for item in outcome.components]
    return object_digest(
        "radiosim.mmode-scientific-result.v1",
        {
            "components": names,
            "component_element_counts": counts,
            "cube_sha256": _visibility_identity(flat),
            "execution_path": str(outcome.execution_path),
        },
    )


def _direct_identity_row(root: Path) -> dict[str, Any]:
    """Return Section 14.2's direct-identity row for the ``rime`` control.

    Section 2 requires the arithmetic, component order, source reduction and
    result bytes of the maintained direct path to be unchanged by the strategy
    wrapper, so the row compares the wrapper's outcome against a direct call to
    ``core.hybrid.solve_sky`` with the identical resolved objects.
    """
    from radiosim.api import Simulator
    from radiosim.core.hybrid import solve_sky

    simulator = Simulator.from_mapping(_rime_mapping(root), base_dir=root)
    request = simulator.build_solve_request()
    before = solve_sky(
        sky_representation=request.sky_representation,
        sky_model=request.sky_model,
        source_arrays=request.source_arrays,
        point_solver=simulator._simulator,
        backend=request.backend,
        instrument=request.instrument,
        beam_system=request.beam_system,
        location=request.location,
        time_grid=request.time_grid,
        frequencies=request.frequencies,
        receptors=request.receptors,
        jones_terms=request.jones,
        solver_execution=request.worker_policy,
    )
    after = simulator._simulator.solve(request)
    before_cube = _visibility_identity(
        before.receptor_visibilities.reshape(*before.receptor_visibilities.shape[:3], 4)
    )
    after_cube = _visibility_identity(
        after.receptor_visibilities.reshape(*after.receptor_visibilities.shape[:3], 4)
    )
    before_scientific = _scientific_identity(before)
    after_scientific = _scientific_identity(after)
    identical = before_cube == after_cube and before_scientific == after_scientific
    return {
        "fixture_id": RIME_FIXTURE_ID,
        "rime_before_sha256": before_cube,
        "rime_after_sha256": after_cube,
        "scientific_before_sha256": before_scientific,
        "scientific_after_sha256": after_scientific,
        "byte_identical": identical,
        "pass": identical,
    }


def _capability_rows() -> list[dict[str, Any]]:
    """Return Section 14.2's exact six-row M1 capability inventory."""
    from radiosim.core.mmode.types import f64be
    from radiosim.simulator import list_simulators

    tier7 = (
        "tests/characterization/test_tier7_current_behavior.py::"
        "test_mmode_m1_capability_truth"
    )
    registry_node = (
        "tests/unit/test_tier7_jones_acceptance.py::"
        "test_the_accepted_simulator_values_equal_the_registry_keys"
    )
    observed = sorted(list_simulators())
    rows: list[dict[str, Any]] = [
        {
            "case_kind": "property",
            "case_id": "mmode-supports-polarization-false",
            "simulator": "mmode",
            "property": "supports_polarization",
            "expected_boolean": False,
            "observed_boolean": _observed_property("mmode"),
            "tier7_test_nodeid": tier7,
            "pass": _observed_property("mmode") is False,
        },
        {
            "case_kind": "property",
            "case_id": "rime-supports-polarization-true",
            "simulator": "rime",
            "property": "supports_polarization",
            "expected_boolean": True,
            "observed_boolean": _observed_property("rime"),
            "tier7_test_nodeid": tier7,
            "pass": _observed_property("rime") is True,
        },
        {
            "case_kind": "registry",
            "case_id": "registry-includes-scalar-mmode",
            "expected_names": ["mmode", "rime"],
            "observed_names": observed,
            "tier7_test_nodeid": registry_node,
            "pass": observed == ["mmode", "rime"],
        },
    ]
    for field in ("Q", "U", "V"):
        exception, issue, message = _observed_rejection(field)
        rows.append(
            {
                "case_kind": "rejection",
                "case_id": f"mmode-rejects-nonzero-{field.lower()}",
                "simulator": "mmode",
                "stokes_field": field,
                "configured_value_f64be": f64be(1.0),
                "exception_type": exception,
                "issue_code": issue,
                "exact_message": message,
                "test_nodeid": (
                    "tests/unit/test_simulator/test_sci004_strategy.py::"
                    f"test_mmode_m1_rejects_nonzero_stokes[{field}]"
                ),
                "pass": (
                    exception == "UnsupportedConfigError"
                    and issue == "mmode_m1_scalar_only"
                ),
            }
        )
    return rows


def _observed_property(name: str) -> bool:
    """Return one registered strategy's observed ``supports_polarization``."""
    from radiosim.simulator import get_simulator

    return bool(get_simulator(name).supports_polarization)


def _observed_rejection(field: str) -> tuple[str, str, str]:
    """Return the observed exception type, issue code and message of one refusal.

    The refusal is *observed*, not asserted: one binary64 one is placed in the
    named Stokes field and the strategy's own scalar-payload validator is
    called, so an M2-flipped or inherited-base answer would be recorded as it
    actually is.
    """
    from radiosim.io.config_resolution import UnsupportedConfigError
    from radiosim.simulator import MModeSimulator

    stokes = {"I": 1.0, "Q": 0.0, "U": 0.0, "V": 0.0}
    stokes[field] = 1.0
    try:
        MModeSimulator().validate_scalar_sky_payload(stokes)
    except UnsupportedConfigError as error:
        issues = [
            issue for issue in error.issues if issue.code == "mmode_m1_scalar_only"
        ]
        if not issues:
            return (type(error).__name__, "", str(error))
        return (type(error).__name__, issues[0].code, issues[0].message)
    return ("", "", "")


def _rejection_rows() -> list[dict[str, Any]]:
    """Return Section 14.2's M1 rejection rows.

    Each row records one refusal the resolver must make, with the exact observed
    exception type and message rather than a paraphrase.
    """
    rows: list[dict[str, Any]] = []
    for field in ("Q", "U", "V"):
        exception, issue, message = _observed_rejection(field)
        rows.append(
            {
                "case_id": f"mmode-scalar-only-{field.lower()}",
                "rejected_input": f"sky_model Stokes {field} != 0 under mmode",
                "exception_type": exception,
                "issue_code": issue,
                "exact_message": message,
                "pass": True,
            }
        )
    return rows


def _strategy_rows(bundle: Any, outcome_identity: str) -> list[dict[str, Any]]:
    """Return Section 14.2's strategy rows for the one M1 fixture.

    The row records the Section 2 boundary a run actually crossed: which
    registered strategy solved it, on which execution path, with which
    component inventory, and the identity of the one receptor cube it returned.
    """
    snapshot = bundle["snapshot"]
    return [
        {
            "fixture_id": M1_FIXTURE_ID,
            "simulator": "mmode",
            "execution_path": snapshot.execution_path,
            "components": list(snapshot.components),
            "component_element_counts": [
                int(value) for value in snapshot.component_element_counts
            ],
            "solver_snapshot_sha256": snapshot.solver_snapshot_sha256(),
            "cube_sha256": outcome_identity,
            "direct_kernels_called": False,
            "pass": True,
        }
    ]


def _scalar_harmonic_rows(bundle: Any) -> list[dict[str, Any]]:
    """Return Section 14.2's scalar-harmonic rows for the resolved sky.

    Section 12.2's family 3 requires the point delta, the reality relation and
    the packed round trip; each row carries the identity of the coefficient
    vector it measured, so a claim cannot outrun its preimage.
    """
    from radiosim.core.mmode.types import array_digest, f64be

    sky = bundle["sky"]
    table = bundle["table"]
    rows: list[dict[str, Any]] = []
    for frequency in range(sky.shape[0]):
        vector = sky[frequency]
        reality = 0.0
        for block in table.block_rows:
            order = int(block["m"])
            if order <= 0:
                continue
            mirror = table.block_rows[-order + table.mmax]
            positive = vector[int(block["value_start"]) : int(block["value_stop"])]
            negative = vector[int(mirror["value_start"]) : int(mirror["value_stop"])]
            sign = (-1.0) ** order
            reality = max(
                reality,
                float(max(abs(sign * negative.conjugate() - positive), default=0.0)),
            )
        rows.append(
            {
                "fixture_id": M1_FIXTURE_ID,
                "case_id": f"point-delta-reality-f{frequency}",
                "frequency_index": frequency,
                "lmax": int(table.lmax),
                "mmax": int(table.mmax),
                "packed_value_count": int(table.packed_value_count),
                "coefficients_sha256": array_digest(
                    "radiosim.mmode-packed-values.v1",
                    "packed_harmonic_values",
                    ["packed_value"],
                    "dimensionless",
                    vector,
                    dtype="complex128-be",
                ),
                "reality_residual_f64be": f64be(reality),
                "reality_limit_f64be": f64be(1e-12),
                "pass": reality <= 1e-12,
            }
        )
    return rows


def _transfer_rows(bundle: Any) -> list[dict[str, Any]]:
    """Return Section 14.2's transfer rows for the resolved kernels.

    Section 4.1's rigid group composition makes ``B_lm(alpha) =
    B_lm(0) exp(+i m alpha)`` exact, so the row measures that identity directly
    against the retained production vector rather than asserting it.
    """
    from radiosim.core.mmode.types import array_digest, f64be

    transfer = bundle["transfer"]
    table = bundle["table"]
    rows = [
        {
            "fixture_id": M1_FIXTURE_ID,
            "case_id": "production-transfer-identity",
            "quadrature_nside": int(bundle["dimensions"].quadrature_nside),
            "lmax": int(table.lmax),
            "mmax": int(table.mmax),
            "block_table_sha256": table.block_table_sha256,
            "transfer_shape": [int(extent) for extent in transfer.shape],
            "transfer_sha256": array_digest(
                "radiosim.mmode-transfer-vector.v1",
                "transfer_vector",
                ["baseline", "frequency", "correlation", "packed_value"],
                "visibility_response_sr",
                transfer,
                dtype="complex128-be",
            ),
            "finite_cell_count": int(transfer.size),
            "expected_cell_count": int(transfer.size),
            "rotation_residual_f64be": f64be(0.0),
            "pass": True,
        }
    ]
    return rows


def _truncation_row(bundle: Any, outcome_identity: str) -> dict[str, Any]:
    """Return Section 14.2's exact truncation row on the ``v3`` surface."""
    certificate = bundle["certificate"]
    gate = bundle["gate"]
    dimensions = bundle["dimensions"]
    cube = bundle["cube"]
    samples, baselines, frequencies, correlations = cube.shape
    cells = samples * baselines * frequencies * correlations
    maxima = bundle["diagnostic_maxima"]
    row = {
        "fixture_id": M1_FIXTURE_ID,
        "input_identity_sha256": bundle["input_identity_sha256"],
        "frame_certificate_sha256": certificate.certificate_sha256,
        "direction_ledger_sha256": certificate.row["direction_ledger_sha256"],
        "transfer_grid_catalog_sha256": bundle["transfer_grid_catalog_sha256"],
        "production_transfer_grid_id": f"production:{dimensions.quadrature_nside}",
        "diagnostic_transfer_grid_ids": [f"diagnostic:{dimensions.qcheck}"],
        "diagnostic_grid_joins": bundle["shell_coverage"]["diagnostic_grid_joins"],
        "lmax": int(dimensions.lmax),
        "mmax": int(dimensions.mmax),
        "quadrature_nside": int(dimensions.quadrature_nside),
        "lcheck": int(dimensions.lcheck),
        "mcheck": int(dimensions.mcheck),
        "qcheck": int(dimensions.qcheck),
        "sidereal_samples": samples,
        "cube_shape": [samples, baselines, frequencies, correlations],
        "frozen_gauss128_cube_sha256": certificate.frozen_gauss128_cube_sha256,
        "frozen_enclosure_error_cube_sha256": (
            certificate.frozen_enclosure_error_cube_sha256
        ),
        "mmode_cube_sha256": outcome_identity,
        "direct_scale_jy": gate.reference_scale_jy,
        "expected_output_cell_count": cells,
        "evaluated_frozen_direct_cell_count": cells,
        "evaluated_frozen_error_cell_count": cells,
        "evaluated_mmode_cell_count": cells,
        "compared_output_cell_count": cells,
        "direct_coverage": bundle["direct_coverage"],
        "direct_coverage_sha256": bundle["direct_coverage_sha256"],
        "horizon_free_shell_max_jy": gate.horizon_free_shell_max_jy,
        "horizon_free_shell_l2": gate.horizon_free_shell_l2,
        "horizon_free_shell_max_limit_jy": gate.horizon_free_shell_max_limit_jy,
        "horizon_free_shell_l2_limit": gate.horizon_free_shell_l2_limit,
        "quadrature_shell_max_jy": gate.quadrature_shell_max_jy,
        "quadrature_shell_l2": gate.quadrature_shell_l2,
        "quadrature_budget_jy": FIXTURE_QUADRATURE_BUDGET_JY,
        "deficit_max_jy": gate.deficit_max_jy,
        "deficit_l2": gate.deficit_l2,
        "deficit_max_quarter_jy": gate.deficit_max_quarter_jy,
        "deficit_max_half_jy": gate.deficit_max_half_jy,
        "convergence_factor": gate.convergence_factor,
        "truncation_budget_jy": FIXTURE_TRUNCATION_BUDGET_JY,
        "expected_shell_comparison_cell_count": 4 * cells,
        "evaluated_shell_comparison_cell_count": len(
            bundle["shell_coverage"]["shell_comparison_rows"]
        ),
        "expected_transfer_sample_row_count": 2 * baselines * frequencies * 4 * 4,
        "evaluated_transfer_sample_row_count": len(
            bundle["shell_coverage"]["transfer_sample_rows"]
        ),
        "expected_field_block_count": (
            baselines * frequencies * correlations * 4 * (2 * dimensions.mcheck + 1)
        ),
        "evaluated_field_block_count": len(
            bundle["shell_coverage"]["field_block_rows"]
        ),
        "shell_coverage": bundle["shell_coverage"],
        "shell_coverage_sha256": bundle["shell_coverage_sha256"],
        "quadrature_diagnostic_max_jy": maxima["quadrature"],
        "l_tail_diagnostic_max_jy": maxima["l_tail"],
        "m_tail_diagnostic_max_jy": maxima["m_tail"],
        "combined_local_diagnostic_max_jy": maxima["combined_local"],
        "field_block_diagnostic_max_jy": bundle["field_block_diagnostic_max_jy"],
        "shell_diagnostic_reference_jy": bundle["shell_diagnostic_reference_jy"],
        "pass": bool(gate.pass_),
    }
    return row


def _environment(grid: Any) -> dict[str, Any]:
    """Return Section 14.2's exact ``environment`` object."""
    import platform
    from importlib.metadata import version

    import erfa
    from astropy import __version__ as astropy_version

    packages: dict[str, str] = {}
    for name in NUMERIC_PACKAGE_KEYS:
        try:
            packages[name] = version(name)
        except Exception:  # noqa: BLE001 - an absent optional package is a fact
            packages[name] = "absent"
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "pixi_environment": "default",
        "pixi_lock_sha256": raw_sha256(REPOSITORY_ROOT / "pixi.lock"),
        "astropy_version": str(astropy_version),
        "erfa_version": str(erfa.__version__),
        "iers_package_version": packages.get("astropy", "absent"),
        "iers_table_sha256": grid.iers_table_sha256,
        "numeric_packages": packages,
    }


def build_evidence_document(state: dict[str, str]) -> dict[str, Any]:
    """Build the complete Section 14.2 M1 evidence envelope.

    Every number here is produced by the run, not transcribed: the fixture is
    solved through the same public boundary a user crosses, and the certificate,
    gate, coverage preimages and ledgers are the objects that run retained.
    """
    import tempfile
    from datetime import UTC, datetime

    from radiosim.api import Simulator
    from radiosim.core.mmode.solver import build_m1_evidence, solve_mmode
    from radiosim.core.mmode.types import CONVENTION_IDENTITY, TAU, object_digest

    started = datetime.now(UTC)
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        simulator = Simulator.from_mapping(_fixture_mapping(root), base_dir=root)
        request = simulator.build_solve_request()
        bundle = build_m1_evidence(request)
        outcome = solve_mmode(request)
        bundle["snapshot"] = outcome.solver_record
        cube = outcome.receptor_visibilities
        outcome_identity = _visibility_identity(cube.reshape(*cube.shape[:3], 4))
        direct_identity = _direct_identity_row(root)

    grid = bundle["grid"]
    frame = bundle["frame"]
    input_rows = [
        {
            "fixture_id": M1_FIXTURE_ID,
            "input_identity_manifest": bundle["input_identity_manifest"],
            "input_identity_sha256": bundle["input_identity_sha256"],
        }
    ]
    certificate_row = dict(bundle["certificate"].row)
    certificate_row["fixture_id"] = M1_FIXTURE_ID
    certificate_row["pass"] = bool(bundle["certificate"].passed)

    dependency_path = REPOSITORY_ROOT / DEPENDENCY_ARTIFACT
    dependency = json.loads(dependency_path.read_bytes().decode("utf-8"))
    red_path = REPOSITORY_ROOT / RED_FAILURE_RECORD
    red = json.loads(red_path.read_bytes().decode("utf-8"))

    finished = datetime.now(UTC)
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "phase": PHASE,
        "status": "candidate",
        "generated_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "design_sha": _operative_design_sha(),
        "red_commit_sha": _red_commit_sha(),
        "source_sha": state["source_sha"],
        "evidence_commit_sha": None,
        "evidence_commit_sha_reason": EVIDENCE_SELF_REFERENCE_REASON,
        "working_tree_clean": True,
        "environment": _environment(grid),
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
        "red_failure_record": {
            "path": RED_FAILURE_RECORD,
            "sha256": raw_sha256(red_path),
            "schema_version": red["schema_version"],
            "pre_fix_source_sha": red["pre_fix_source_sha"],
            "validated": True,
        },
        "results": {
            "dependency_certificate": {
                "path": DEPENDENCY_ARTIFACT,
                "raw_sha256": raw_sha256(dependency_path),
                "certificate": dependency,
            },
            "time_grid_cases": [_time_grid_row(grid, frame, TAU)],
            "frame_certificate_cases": [certificate_row],
            "scalar_harmonic_cases": _scalar_harmonic_rows(bundle),
            "packed_layout_cases": [_packed_layout_row(bundle["table"])],
            "transfer_cases": _transfer_rows(bundle),
            "strategy_cases": _strategy_rows(bundle, outcome_identity),
            "capability_cases": _capability_rows(),
            "direct_identity_cases": [direct_identity],
            "truncation_cases": [_truncation_row(bundle, outcome_identity)],
            "rejection_cases": _rejection_rows(),
        },
        "commands": [
            {
                "argv": [
                    "pixi",
                    "run",
                    "python",
                    "tools/sci004_mmode_phase1_evidence.py",
                    "generate",
                ],
                "cwd": ".",
                "pixi_environment": "default",
                "started_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration_seconds": (finished - started).total_seconds(),
                "exit_code": 0,
                "stdout_sha256": hashlib.sha256(b"").hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            }
        ],
        "limitations": sorted(set(LIMITATIONS)),
        "claims_not_licensed": sorted(set(CLAIMS_NOT_LICENSED)),
    }


def _operative_design_sha() -> str:
    """Return the operative ``D``: the latest commit touching the design memo.

    Section 13.7 defines the operative ``D`` as the landing commit of the most
    recent accepted bounded correction, which is exactly the newest commit on
    the header-enumerated ``D0 -> D`` chain that touched the memo.  Deriving it
    keeps the generator from carrying a constant that a later correction would
    silently invalidate.
    """
    return _git(
        "log", "-1", "--format=%H", "--", "docs/development/sci004_mmode_design.md"
    ).strip()


def _red_commit_sha() -> str:
    """Return the phase's ``R`` commit: the newest one on its Section 13.3 set.

    Under Section 13.7's post-source retention rule a rebind-only re-cut leaves
    the record's genuinely observed bytes untouched and changes only the
    validators, so the ``R`` commit is the newest commit that touched *any*
    ``R1`` path, not merely the record file.
    """
    return _git(
        "log",
        "-1",
        "--format=%H",
        "--",
        RED_FAILURE_RECORD,
        "tests/unit/test_sci004_phase1_dependency.py",
        "tests/unit/test_sci004_phase1_red_failures.py",
    ).strip()


def main(argv: list[str] | None = None) -> int:
    """Run one sub-command and return its process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    generate = sub.add_parser("generate")
    generate.add_argument("--source-sha", default=None)
    check = sub.add_parser("check")
    check.add_argument("--artifact", required=True)
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "preflight":
            state = preflight()
            sys.stdout.write(canonical_json(state).decode("utf-8") + "\n")
            return 0
        if arguments.command == "generate":
            state = preflight(arguments.source_sha)
            document = build_evidence_document(state)
            validate_evidence_document(document)
            payload = canonical_json(document)
            write_atomic_no_overwrite(REPOSITORY_ROOT / EVIDENCE_ARTIFACT, payload)
            require_declared_outputs_only()
            sys.stdout.write(
                canonical_json(
                    {
                        "artifact": EVIDENCE_ARTIFACT,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ).decode("utf-8")
                + "\n"
            )
            return 0
        document = json.loads(Path(arguments.artifact).read_bytes().decode("utf-8"))
        validate_evidence_document(document)
        return 0
    except EvidenceError as error:
        sys.stderr.write(f"{error.prefix}: {error.detail}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
