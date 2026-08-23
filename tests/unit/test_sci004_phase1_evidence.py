"""Strict authentication of the SCI-004 phase-M1 evidence artifact.

``docs/development/sci004_mmode_design.md`` Sections 13.3, 14.2 and 14.4 freeze
this module's successor authority: it lands in ``S1`` with both approved
constants as the literal ``None``, the official evidence path **absent**, and
every synthetic strict schema and digest fixture passing.  ``E1`` then changes
*only* the two constants below, from ``None`` to the exact lower-case 40- and
64-hexadecimal literals, and adds the artifact and its reproduction record.  No
import, expression, annotation, key, surrounding token, or other literal in
either assignment may change, so the validator's own token stream outside those
two spans is comparable to its direct-parent ``S1`` bytes.

In the ``S1`` state the null constants require the official artifact to be
absent while the synthetic fixtures prove the schema rules.  In the ``E1`` state
the validator additionally requires the artifact's ``source_sha`` and the
constant to equal the approved ``S1``, authenticates the raw artifact bytes,
locates the unique artifact-introducing ``E1`` commit and requires its direct
parent to be ``S1``, and checks the ``S1..E1`` diff against Section 13.  It
deliberately does **not** require the current checkout or ``E1`` to equal
``source_sha``.

Importing this module loads only the Python standard library plus ``pytest``,
following ``tools/wp7_perf001_cpu_evidence.py``: an evidence-critical validator
must not depend on a package that is merely transitively present, because a lock
update could drop it and silently turn a hard authentication into a collection
error.  The generator at ``tools/sci004_mmode_phase1_evidence.py`` is the
normative producer; the checks below enforce the same structure, key order and
encodings in their own code rather than importing the producer's opinion of
them.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Section 14.2's two approved constants.  ``E1`` replaces exactly these two
#: ``None`` literals and nothing else in this module.
APPROVED_SOURCE_SHA: str | None = None
APPROVED_ARTIFACT_SHA256: str | None = None

TOOL = "tools/sci004_mmode_phase1_evidence.py"
ARTIFACT = "docs/development/sci004_mmode_phase1_evidence.json"
REPRODUCTION = "docs/development/sci004_mmode_phase1_evidence.md"
RED_RECORD = "docs/development/sci004_mmode_phase1_red_failures.json"
DEPENDENCY = "docs/development/sci004_mmode_phase1_wp7_dependency.json"

GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

EVIDENCE_SCHEMA = "radiosim.sci004.mmode-phase1-evidence.v1"
RED_FAILURE_SCHEMA = "radiosim.sci004.mmode-phase1-red-failures.v1"
SELF_REFERENCE_REASON = "self-reference: A binds the containing E commit"

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

#: Section 14.2's exact M1 ``results`` key order.
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

#: Section 14.2's exact six-row ``capability_cases`` inventory, in order.
CAPABILITY_ROWS: tuple[tuple[str, str], ...] = (
    ("property", "mmode-supports-polarization-false"),
    ("property", "rime-supports-polarization-true"),
    ("registry", "registry-includes-scalar-mmode"),
    ("rejection", "mmode-rejects-nonzero-q"),
    ("rejection", "mmode-rejects-nonzero-u"),
    ("rejection", "mmode-rejects-nonzero-v"),
)

TIER7_PROPERTY_NODEID = (
    "tests/characterization/test_tier7_current_behavior.py::"
    "test_mmode_m1_capability_truth"
)
TIER7_REGISTRY_NODEID = (
    "tests/unit/test_tier7_jones_acceptance.py::"
    "test_the_accepted_simulator_values_equal_the_registry_keys"
)
STOKES_NODEIDS: dict[str, str] = {
    "Q": (
        "tests/unit/test_simulator/test_sci004_strategy.py::"
        "test_mmode_m1_rejects_nonzero_stokes[Q]"
    ),
    "U": (
        "tests/unit/test_simulator/test_sci004_strategy.py::"
        "test_mmode_m1_rejects_nonzero_stokes[U]"
    ),
    "V": (
        "tests/unit/test_simulator/test_sci004_strategy.py::"
        "test_mmode_m1_rejects_nonzero_stokes[V]"
    ),
}

MMODE_M1_SCALAR_ONLY_MESSAGE = (
    "MModeSimulator phase M1 accepts Stokes I only; non-zero Q, U, or V "
    "requires accepted phase M2."
)


def _tool() -> Any:
    """Import the tracked generator without adding an import-time dependency."""
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    try:
        import sci004_mmode_phase1_evidence as module
    finally:
        sys.path.pop(0)
    return module


def _canonical(value: Any) -> bytes:
    return _tool().canonical_json(value)


def _synthetic_capability_rows() -> list[dict[str, Any]]:
    """Return the exact six-row capability array a passing artifact carries."""
    rows: list[dict[str, Any]] = [
        {
            "case_kind": "property",
            "case_id": "mmode-supports-polarization-false",
            "simulator": "mmode",
            "property": "supports_polarization",
            "expected_boolean": False,
            "observed_boolean": False,
            "tier7_test_nodeid": TIER7_PROPERTY_NODEID,
            "pass": True,
        },
        {
            "case_kind": "property",
            "case_id": "rime-supports-polarization-true",
            "simulator": "rime",
            "property": "supports_polarization",
            "expected_boolean": True,
            "observed_boolean": True,
            "tier7_test_nodeid": TIER7_PROPERTY_NODEID,
            "pass": True,
        },
        {
            "case_kind": "registry",
            "case_id": "registry-includes-scalar-mmode",
            "expected_names": ["mmode", "rime"],
            "observed_names": ["mmode", "rime"],
            "tier7_test_nodeid": TIER7_REGISTRY_NODEID,
            "pass": True,
        },
    ]
    for field in ("Q", "U", "V"):
        rows.append(
            {
                "case_kind": "rejection",
                "case_id": f"mmode-rejects-nonzero-{field.lower()}",
                "simulator": "mmode",
                "stokes_field": field,
                "configured_value_f64be": "3ff0000000000000",
                "exception_type": (
                    "radiosim.io.config_resolution.UnsupportedConfigError"
                ),
                "issue_code": "mmode_m1_scalar_only",
                "exact_message": MMODE_M1_SCALAR_ONLY_MESSAGE,
                "test_nodeid": STOKES_NODEIDS[field],
                "pass": True,
            }
        )
    return rows


def _synthetic_truncation_row() -> dict[str, Any]:
    """Return a truncation row on the ``sci004_two_tier_direct.v3`` surface.

    The numbers are the qualified M1 fixture's own measured values: tier 1a at
    the ``1e-13`` level against its ``1e-8`` limit, the recorded with-horizon
    shell inside its reviewed budget, and the three-level deficit sequence whose
    quarter-to-full factor is ``6.12`` against the fixed floor of two.
    """
    sixty_four = "0" * 63 + "1"
    row: dict[str, Any] = {
        "fixture_id": "mmode_point_stokes_i",
        "input_identity_sha256": sixty_four,
        "frame_certificate_sha256": sixty_four,
        "direction_ledger_sha256": sixty_four,
        "transfer_grid_catalog_sha256": sixty_four,
        "production_transfer_grid_id": "production:8",
        "diagnostic_transfer_grid_ids": ["diagnostic:16"],
        "diagnostic_grid_joins": [],
        "lmax": 16,
        "mmax": 16,
        "quadrature_nside": 8,
        "lcheck": 24,
        "mcheck": 24,
        "qcheck": 16,
        "sidereal_samples": 49,
        "cube_shape": [49, 3, 3, 4],
        "frozen_gauss128_cube_sha256": sixty_four,
        "frozen_enclosure_error_cube_sha256": sixty_four,
        "mmode_cube_sha256": sixty_four,
        "direct_scale_jy": 2.251504,
        "expected_output_cell_count": 1764,
        "evaluated_frozen_direct_cell_count": 1764,
        "evaluated_frozen_error_cell_count": 1764,
        "evaluated_mmode_cell_count": 1764,
        "compared_output_cell_count": 1764,
        "direct_coverage": {},
        "direct_coverage_sha256": sixty_four,
        "horizon_free_shell_max_jy": 1.285638e-13,
        "horizon_free_shell_l2": 5.856862e-14,
        "horizon_free_shell_max_limit_jy": 2.261504e-08,
        "horizon_free_shell_l2_limit": 1e-8,
        "quadrature_shell_max_jy": 5.80459e-02,
        "quadrature_shell_l2": 1.086599e-02,
        "quadrature_budget_jy": 0.1,
        "deficit_max_jy": 1.169865e-01,
        "deficit_l2": 3.07602e-02,
        "deficit_max_quarter_jy": 7.159405e-01,
        "deficit_max_half_jy": 1.846944e-01,
        "convergence_factor": 6.119855,
        "truncation_budget_jy": 0.2,
        "expected_shell_comparison_cell_count": 7056,
        "evaluated_shell_comparison_cell_count": 7056,
        "expected_transfer_sample_row_count": 552960,
        "evaluated_transfer_sample_row_count": 552960,
        "expected_field_block_count": 7056,
        "evaluated_field_block_count": 7056,
        "shell_coverage": {},
        "shell_coverage_sha256": sixty_four,
        "quadrature_diagnostic_max_jy": 5.80459e-02,
        "l_tail_diagnostic_max_jy": 1.0e-02,
        "m_tail_diagnostic_max_jy": 1.0e-03,
        "combined_local_diagnostic_max_jy": 6.0e-02,
        "field_block_diagnostic_max_jy": 2.0e-02,
        "shell_diagnostic_reference_jy": 2.3e-06,
        "pass": True,
    }
    return row


def _synthetic_envelope() -> dict[str, Any]:
    """Return a complete synthetic envelope that satisfies every Section 14.2 rule."""
    forty = "0" * 39 + "1"
    sixty_four = "0" * 63 + "1"
    command = {
        "argv": ["pixi", "run", "test", "--", "-m", "not slow"],
        "cwd": ".",
        "pixi_environment": "default",
        "started_at_utc": "2026-08-22T00:00:00Z",
        "duration_seconds": 1.5,
        "exit_code": 0,
        "stdout_sha256": sixty_four,
        "stderr_sha256": sixty_four,
    }
    certificate = {
        "schema_version": "radiosim.perf001.cpu_acceptance_certificate.v1",
        "acceptance_commit": forty,
        "evidence_commit": forty,
        "generating_source_sha": forty,
        "descendant_commit": forty,
        "artifact_path": "docs/development/wp7_perf001_cpu_evidence.json",
        "artifact_sha256": sixty_four,
        "cpu_evidence_tool_sha256": sixty_four,
        "production_record_validator_sha256": sixty_four,
        "production_harness_sha256": sixty_four,
        "pixi_manifest_sha256": sixty_four,
        "pixi_lock_sha256": sixty_four,
        "evidence_diff_paths": [],
        "acceptance_diff_paths": [],
        "verdict": "CPU_ACCEPTED_P_E_HARDWARE_GATED",
        "passed": True,
    }
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "phase": "M1",
        "status": "candidate",
        "generated_at_utc": "2026-08-22T00:00:00Z",
        "design_sha": forty,
        "red_commit_sha": forty,
        "source_sha": forty,
        "evidence_commit_sha": None,
        "evidence_commit_sha_reason": SELF_REFERENCE_REASON,
        "working_tree_clean": True,
        "environment": {
            "python": "3.11.13",
            "platform": "darwin",
            "machine": "arm64",
            "pixi_environment": "default",
            "pixi_lock_sha256": sixty_four,
            "astropy_version": "7.1.0",
            "erfa_version": "2.0.1.5",
            "iers_package_version": "0.2025.1.1.0.0.0",
            "iers_table_sha256": sixty_four,
            "numeric_packages": {
                "numpy": "2.3.2",
                "scipy": "1.17.0",
                "healpy": "1.19.0",
                "jax": "0.7.0",
                "dask": "2025.1.0",
            },
        },
        "source_identities": {
            "git_tree_sha256": sixty_four,
            "pixi_manifest_sha256": sixty_four,
            "pixi_lock_sha256": sixty_four,
            "convention_identity_sha256": sixty_four,
            "fixture_input_rows": [
                {
                    "fixture_id": "mmode_point_stokes_i",
                    "input_identity_manifest": {},
                    "input_identity_sha256": sixty_four,
                }
            ],
            "input_identity_set_sha256": sixty_four,
        },
        "red_failure_record": {
            "path": RED_RECORD,
            "sha256": sixty_four,
            "schema_version": RED_FAILURE_SCHEMA,
            "pre_fix_source_sha": forty,
            "validated": True,
        },
        "results": {
            "dependency_certificate": {
                "path": DEPENDENCY,
                "raw_sha256": sixty_four,
                "certificate": certificate,
            },
            "time_grid_cases": [],
            "frame_certificate_cases": [],
            "scalar_harmonic_cases": [],
            "packed_layout_cases": [],
            "transfer_cases": [],
            "strategy_cases": [],
            "capability_cases": _synthetic_capability_rows(),
            "direct_identity_cases": [
                {
                    "fixture_id": "rime_point_reference",
                    "rime_before_sha256": sixty_four,
                    "rime_after_sha256": sixty_four,
                    "scientific_before_sha256": sixty_four,
                    "scientific_after_sha256": sixty_four,
                    "byte_identical": True,
                    "pass": True,
                }
            ],
            "truncation_cases": [_synthetic_truncation_row()],
            "rejection_cases": [],
        },
        "commands": [command],
        "limitations": [
            "no accelerator run of the m-mode solver has been measured",
        ],
        "claims_not_licensed": [
            "general_speedup",
            "gpu_or_accelerator_support",
            "polarized_mmode_support",
        ],
    }


# ---------------------------------------------------------------------------
# S1 state: the official artifact must be absent
# ---------------------------------------------------------------------------


def test_the_approved_constants_are_null_sentinels_before_e1() -> None:
    """Section 13.5: at ``S1`` both approved digests are the literal ``None``.

    The pair moves together.  A half-flipped module would authenticate an
    artifact against a source it never approved, which is exactly the
    substitution the two-constant rule exists to prevent.
    """
    if APPROVED_SOURCE_SHA is None or APPROVED_ARTIFACT_SHA256 is None:
        assert APPROVED_SOURCE_SHA is None
        assert APPROVED_ARTIFACT_SHA256 is None
        return
    assert GIT_SHA.fullmatch(APPROVED_SOURCE_SHA)
    assert SHA256.fullmatch(APPROVED_ARTIFACT_SHA256)


def test_the_official_evidence_artifact_is_absent_in_the_s1_state() -> None:
    """Section 14.2: null constants require the not-yet-authorized paths absent."""
    if APPROVED_ARTIFACT_SHA256 is not None:
        return
    assert not (REPOSITORY_ROOT / ARTIFACT).exists()
    assert not (REPOSITORY_ROOT / REPRODUCTION).exists()


def test_the_tracked_generator_and_its_inputs_exist_at_s1() -> None:
    """Section 14.4: a clean exact ``S1`` already contains the producing bytes."""
    assert (REPOSITORY_ROOT / TOOL).is_file()
    assert (REPOSITORY_ROOT / RED_RECORD).is_file()
    assert (REPOSITORY_ROOT / DEPENDENCY).is_file()


def test_the_generator_imports_only_the_standard_library() -> None:
    """An evidence-critical generator carries no transitive package dependency."""
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    for forbidden in ("import numpy", "import astropy", "import pytest", "import yaml"):
        assert forbidden not in source, forbidden


def test_the_generator_refuses_to_produce_anything_from_a_dirty_tree() -> None:
    """Section 14.2: the common pre-output check fails closed."""
    module = _tool()
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / TOOL), "generate"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr.startswith(module.PREFLIGHT + ": ")


# ---------------------------------------------------------------------------
# Synthetic strict schema fixtures
# ---------------------------------------------------------------------------


def test_the_synthetic_envelope_satisfies_every_section_14_2_rule() -> None:
    """The fixture is a positive control: the rules below reject deviations."""
    module = _tool()
    envelope = module.validate_evidence_document(_synthetic_envelope())
    assert tuple(envelope) == ENVELOPE_KEYS
    assert tuple(envelope["results"]) == RESULT_KEYS


@pytest.mark.parametrize("key", ENVELOPE_KEYS)
def test_a_missing_top_level_key_is_rejected(key: str) -> None:
    """Section 14: every object rejects unknown or missing keys."""
    module = _tool()
    document = _synthetic_envelope()
    del document[key]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_an_unknown_top_level_key_is_rejected() -> None:
    """An extra key is a different document, not a superset of this one."""
    module = _tool()
    document = _synthetic_envelope()
    document["extra"] = 1
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_non_null_evidence_commit_sha_is_rejected() -> None:
    """Section 14.4: ``E`` artifacts use a null self SHA, bound later by ``A``."""
    module = _tool()
    document = _synthetic_envelope()
    document["evidence_commit_sha"] = "0" * 40
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_reworded_self_reference_reason_is_rejected() -> None:
    """The reason is an exact literal, not a paraphrase."""
    module = _tool()
    document = _synthetic_envelope()
    document["evidence_commit_sha_reason"] = "self reference"
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


@pytest.mark.parametrize("index", range(6))
def test_a_reordered_capability_row_fails_m1(index: int) -> None:
    """Section 14.2: missing, duplicate, reordered or inherited rows fail M1."""
    module = _tool()
    document = _synthetic_envelope()
    rows = document["results"]["capability_cases"]
    rows[index], rows[(index + 1) % 6] = rows[(index + 1) % 6], rows[index]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_an_m2_flipped_polarization_row_fails_m1() -> None:
    """Only accepted M2 may flip the m-mode polarization property to true."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["capability_cases"][0]["observed_boolean"] = True
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_registry_row_missing_the_second_key_fails() -> None:
    """Section 2's invariant is exact: the registry keys are ``mmode`` and ``rime``."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["capability_cases"][2]["observed_names"] = ["rime"]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_changed_direct_identity_digest_is_rejected() -> None:
    """Section 13.3: M1 must prove the wrapper leaves every direct path unchanged."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["direct_identity_cases"][0]["rime_after_sha256"] = "1" * 64
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_the_synthetic_truncation_row_carries_the_v3_surface() -> None:
    """Section 14.2: the corrected truncation row, in the memo's exact order."""
    module = _tool()
    document = _synthetic_envelope()
    row = document["results"]["truncation_cases"][0]

    assert tuple(row) == module.TRUNCATION_ROW_KEYS
    module.validate_truncation_row(row, "truncation_cases[0]")


def test_a_tier_1a_shell_above_its_fixed_limit_is_rejected() -> None:
    """Section 7.3: tier 1a is the half with a fixed numeric limit."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["truncation_cases"][0]["horizon_free_shell_max_jy"] = 1.0
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_non_monotone_deficit_sequence_is_rejected() -> None:
    """Section 7.3: a non-converging harmonic representation licenses nothing."""
    module = _tool()
    document = _synthetic_envelope()
    row = document["results"]["truncation_cases"][0]
    row["deficit_max_half_jy"] = row["deficit_max_quarter_jy"] * 2.0
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_quarter_to_full_factor_below_two_is_rejected() -> None:
    """Section 7.3: the fixed floor is two, and it is never widened."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["truncation_cases"][0]["convergence_factor"] = 1.9
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_deficit_above_its_declared_budget_is_rejected() -> None:
    """Section 7.3: the per-fixture budget is reviewed evidence, and it binds."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["truncation_cases"][0]["truncation_budget_jy"] = 1e-3
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_with_horizon_shell_above_its_declared_budget_is_rejected() -> None:
    """Section 7.3: tier 1b carries no universal limit, but its budget binds."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["truncation_cases"][0]["quadrature_budget_jy"] = 1e-3
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_failed_wp7_dependency_certificate_is_rejected() -> None:
    """Section 14.2: the replay sets no independent pass flag that could disagree."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["dependency_certificate"]["certificate"]["passed"] = False
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_non_zero_command_exit_code_is_rejected() -> None:
    """Section 14.2: evidence commands require exit code zero."""
    module = _tool()
    document = _synthetic_envelope()
    document["commands"][0]["exit_code"] = 1
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_unsorted_claim_arrays_are_rejected() -> None:
    """``limitations`` and ``claims_not_licensed`` are sorted and unique."""
    module = _tool()
    document = _synthetic_envelope()
    document["claims_not_licensed"] = ["gpu_or_accelerator_support", "general_speedup"]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


# ---------------------------------------------------------------------------
# Canonical JSON and digest rules
# ---------------------------------------------------------------------------


def test_canonical_json_sorts_keys_and_emits_no_whitespace() -> None:
    """Section 14: sorted keys, tight separators, ASCII, no trailing newline."""
    assert _canonical({"b": 1, "a": [1, 2]}) == b'{"a":[1,2],"b":1}'


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, b"1"),
        (0.0, b"0"),
        (-0.5, b"-0.5"),
        (1e21, b"1e+21"),
        (1e-7, b"1e-7"),
        (1.5e300, b"1.5e+300"),
    ],
)
def test_canonical_numbers_use_the_ecmascript_spelling(
    value: float, expected: bytes
) -> None:
    """``1.0`` and ``1e0`` are not canonical record bytes for the integer one."""
    assert _canonical(value) == expected


def test_canonical_json_forbids_nan_and_infinity() -> None:
    """Section 14: NaN and Infinity are forbidden."""
    module = _tool()
    with pytest.raises((module.EvidenceError, ValueError)):
        _canonical(float("nan"))


def test_the_domain_digest_matches_its_printed_definition() -> None:
    """Section 14.0: ``D(d, p) = SHA256(d || NUL || U64(len(p)) || p)``."""
    module = _tool()
    payload = b"payload"
    expected = hashlib.sha256(
        b"radiosim.example.v1" + b"\x00" + len(payload).to_bytes(8, "big") + payload
    ).hexdigest()
    assert module.domain_digest("radiosim.example.v1", payload) == expected


def test_a_distinct_domain_gives_a_distinct_digest() -> None:
    """The domain is part of the preimage, so two roles never collide."""
    module = _tool()
    first = module.domain_digest("radiosim.a.v1", b"x")
    second = module.domain_digest("radiosim.b.v1", b"x")
    assert first != second


# ---------------------------------------------------------------------------
# E1 state: authenticate the retained artifact
# ---------------------------------------------------------------------------


def test_the_retained_artifact_authenticates_against_the_approved_constants() -> None:
    """Section 14.2's ``E1`` state, skipped until the constants are flipped."""
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_SOURCE_SHA is None:
        pytest.skip("the M1 evidence artifact is authorized at E1")
    path = REPOSITORY_ROOT / ARTIFACT
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == APPROVED_ARTIFACT_SHA256
    document = json.loads(payload.decode("utf-8"))
    module = _tool()
    module.validate_evidence_document(document)
    assert document["source_sha"] == APPROVED_SOURCE_SHA
    assert module.canonical_json(document) == payload
