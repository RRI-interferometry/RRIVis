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

**This generator produces nothing at ``S1``.**  Section 14.4 places the run at
``E1``: the phase generator "runs only at its globally clean exact ``S``", and
until that commit exists the preflight deliberately refuses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
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
    _require(isinstance(value, dict), SCHEMA, f"{label} must be an object")
    mapping = dict(value)
    _require(
        tuple(mapping) == keys,
        SCHEMA,
        f"{label} must have exactly {list(keys)} in that order",
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
            preflight(arguments.source_sha)
            raise EvidenceError(
                PREFLIGHT,
                "the M1 evidence artifact is generated at E1, from the globally "
                "clean exact S1; this tool is tracked at S1 and produces nothing "
                "there (design Sections 13.3 and 14.4)",
            )
        document = json.loads(Path(arguments.artifact).read_bytes().decode("utf-8"))
        validate_evidence_document(document)
        return 0
    except EvidenceError as error:
        sys.stderr.write(f"{error.prefix}: {error.detail}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
