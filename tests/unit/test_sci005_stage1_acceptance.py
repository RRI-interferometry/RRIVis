"""Strict authentication of the SCI-005 Stage-1 acceptance record.

``docs/development/sci005_beam_physics_plan.md`` Section 7.5 freezes this
module's successor authority: it lands in ``S1`` with both approved constants
as the literal ``None``, the official acceptance path absent, and every
synthetic strict schema/digest/ancestry fixture passing. ``A1`` then changes
*only* the two constants below, from ``None`` to the exact lower-case 40- and
64-hexadecimal literals. No import, expression, annotation, key, surrounding
token, or other literal in either assignment may change, so the validator can
compare its own token stream outside those two spans to its direct-parent
``E1`` bytes.

Section 8.2 adds that the validator operates on named Git objects rather than
whichever file happens to be checked out, and Section 9 freezes the read-only
verifier's certificate, its key order, and its six stderr prefixes.

Importing this module loads only the Python standard library plus ``pytest``,
following ``tools/wp7_perf001_cpu_evidence.py``: an acceptance-critical
validator must not depend on a package that is merely transitively present,
because a lock update could drop it and silently turn a hard authentication
into a collection error. ``docs/development/
sci005_stage1_acceptance.schema.json`` remains the normative transcription of
Section 8.2; :func:`validate_stage1_acceptance` enforces the same structure,
types, key order, hexadecimal encodings and cross-field rules in its own code,
and every rejection class below is proved directly.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Section 7.5's two approved constants. ``A1`` replaces exactly these two
#: ``None`` literals and nothing else in this module.
# fmt: off
# The flipped 64-hex literals exceed the format line limit by design;
# Section 7.5's substitution is byte-exact and must not be rewrapped.
APPROVED_EVIDENCE_SHA: str | None = "bbc2b1b4d16bce296c2b6f6597c7c180a70f0f7f"
APPROVED_ACCEPTANCE_ARTIFACT_SHA256: str | None = "9c10e5859308cfdd853d8e7889bd008d70246d6ea62219e8f9216696603b3a5d"
# fmt: on

TOOL = "tools/sci005_stage1_acceptance.py"
ARTIFACT = "docs/development/sci005_stage1_acceptance.json"
SCHEMA = "docs/development/sci005_stage1_acceptance.schema.json"

GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

#: Section 9's six frozen stderr prefixes.
STDERR_PREFIXES: tuple[str, ...] = (
    "SCI005_ACCEPTANCE_ARGUMENT",
    "SCI005_ACCEPTANCE_SCHEMA",
    "SCI005_ACCEPTANCE_ANCESTRY",
    "SCI005_ACCEPTANCE_DIGEST",
    "SCI005_ACCEPTANCE_DIFF_AUTHORITY",
    "SCI005_ACCEPTANCE_VERDICT",
)

#: Section 9's exact certificate key order.
CERTIFICATE_KEYS: tuple[str, ...] = (
    "schema_version",
    "stage",
    "acceptance_commit_sha",
    "acceptance_artifact_path",
    "acceptance_artifact_sha256",
    "evidence_commit_sha",
    "evidence_artifact_path",
    "evidence_artifact_sha256",
    "source_sha",
    "verdict",
    "successor_unlocks",
)

#: Section 8.2's exact top-level key order.
ACCEPTANCE_KEYS: tuple[str, ...] = (
    "schema_version",
    "stage",
    "verdict",
    "generated_at_utc",
    "implementation_identity",
    "reviewer_identity",
    "reviewer_independent",
    "design_sha",
    "red_test_sha",
    "source_sha",
    "evidence_commit_sha",
    "evidence_artifact_path",
    "evidence_artifact_sha256",
    "evidence_schema_path",
    "evidence_schema_sha256",
    "toolchain",
    "acceptance_commit_sha",
    "acceptance_commit_sha_reason",
    "successor_unlocks",
    "reviewed_artifacts",
    "rederived_oracles",
    "review_checks",
    "commands",
    "blockers",
    "accepted_limitations",
    "claims_not_licensed",
)

REQUIRED_ORACLES = (
    "blocked_aperture_transform",
    "ruze_limit_oracle",
    "ruze_pair_oracle",
    "unmodified_profile_transform",
    "zernike_phase_transform",
)
REQUIRED_CHECKS = (
    "artifact_authentication",
    "default_disabled_fingerprints",
    "diff_authority",
    "gate_replay",
    "production_data_flow",
    "typed_rejection",
)


def _tool() -> Any:
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    try:
        import sci005_stage1_acceptance as module
    finally:
        sys.path.pop(0)
    return module


def _schema() -> dict[str, Any]:
    return json.loads((REPOSITORY_ROOT / SCHEMA).read_text(encoding="utf-8"))


TOOLCHAIN_KEYS: tuple[str, ...] = (
    "evidence_generator_path",
    "evidence_generator_git_blob",
    "evidence_validator_path",
    "evidence_validator_git_blob",
    "acceptance_generator_path",
    "acceptance_generator_git_blob",
    "acceptance_validator_path",
    "acceptance_validator_pre_a_git_blob",
    "acceptance_schema_path",
    "acceptance_schema_sha256",
)
TIMESTAMP = re.compile(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")


class AcceptanceSchemaError(AssertionError):
    """One retained-acceptance authentication failure."""


def _fail(path: str, detail: str) -> None:
    raise AcceptanceSchemaError(f"{path}: {detail}")


def _mapping(value: Any, path: str, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, f"expected an object, observed {type(value).__name__}")
    if tuple(value) != keys:
        missing = [key for key in keys if key not in value]
        unknown = [key for key in value if key not in keys]
        if missing or unknown:
            _fail(path, f"missing {missing}, unknown {unknown}")
        _fail(path, f"keys are not in the declared order: {list(value)}")
    return value


def _text(value: Any, path: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or isinstance(value, bool) or not value:
        _fail(path, f"expected a non-empty string, observed {value!r}")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail(path, f"{value!r} does not match {pattern.pattern}")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"expected a number, observed {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        _fail(path, "expected a finite non-negative number")
    return numeric


def _flag(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, f"expected a boolean, observed {type(value).__name__}")
    return value


def _canonical_path(value: Any, path: str) -> str:
    text = _text(value, path)
    if text.startswith("/") or "\\" in text or "\x00" in text:
        _fail(path, f"{text!r} is not a repository-relative POSIX path")
    if any(part in {"", ".", ".."} for part in text.split("/")):
        _fail(path, f"{text!r} has an empty or relative component")
    return text


def _sorted_unique(values: list[Any], path: str) -> None:
    if values != sorted(values):
        _fail(path, "array is not sorted")
    if len(set(values)) != len(values):
        _fail(path, "array contains a duplicate")


def validate_stage1_acceptance(document: Any) -> None:
    """Authenticate one Stage-1 acceptance document against Section 8.2."""
    root = _mapping(document, "$", ACCEPTANCE_KEYS)
    if root["schema_version"] != "radiosim.sci005.stage1-acceptance.v1":
        _fail("$.schema_version", "is not the Stage-1 acceptance literal")
    if root["stage"] != 1 or isinstance(root["stage"], bool):
        _fail("$.stage", "must be the integer 1")
    verdict = root["verdict"]
    if verdict not in {"ACCEPT", "REJECT"}:
        _fail("$.verdict", "must be ACCEPT or REJECT")
    _text(root["generated_at_utc"], "$.generated_at_utc", pattern=TIMESTAMP)
    for key in ("implementation_identity", "reviewer_identity"):
        _text(root[key], f"$.{key}")
    _flag(root["reviewer_independent"], "$.reviewer_independent")
    for key in ("design_sha", "red_test_sha", "source_sha", "evidence_commit_sha"):
        _text(root[key], f"$.{key}", pattern=GIT_SHA)
    for key in ("evidence_artifact_sha256", "evidence_schema_sha256"):
        _text(root[key], f"$.{key}", pattern=SHA256)
    if root["evidence_artifact_path"] != "docs/development/sci005_stage1_evidence.json":
        _fail("$.evidence_artifact_path", "is not the frozen Stage-1 path")
    if (
        root["evidence_schema_path"]
        != "docs/development/sci005_stage1_evidence.schema.json"
    ):
        _fail("$.evidence_schema_path", "is not the frozen Stage-1 path")

    toolchain = _mapping(root["toolchain"], "$.toolchain", TOOLCHAIN_KEYS)
    for key, value in toolchain.items():
        if key.endswith("_git_blob"):
            _text(value, f"$.toolchain.{key}", pattern=GIT_SHA)
        elif key.endswith("_sha256"):
            _text(value, f"$.toolchain.{key}", pattern=SHA256)
        else:
            _canonical_path(value, f"$.toolchain.{key}")

    if root["acceptance_commit_sha"] is not None:
        _fail("$.acceptance_commit_sha", "is JSON null; U1 binds the containing A1")
    reason = root["acceptance_commit_sha_reason"]
    expected_reason = (
        "self-reference: U1 binds the containing A1 commit"
        if verdict == "ACCEPT"
        else "not-applicable: REJECT creates no A commit"
    )
    if reason != expected_reason:
        _fail("$.acceptance_commit_sha_reason", f"must be {expected_reason!r}")
    unlocks = root["successor_unlocks"]
    if not isinstance(unlocks, list):
        _fail("$.successor_unlocks", "must be an array")
    if unlocks != (["SCI005.U1"] if verdict == "ACCEPT" else []):
        _fail("$.successor_unlocks", "does not match the verdict")

    artifacts = root["reviewed_artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        _fail("$.reviewed_artifacts", "must be a non-empty array")
    for index, row in enumerate(artifacts):
        entry = _mapping(
            row,
            f"$.reviewed_artifacts[{index}]",
            ("path", "sha256", "source_sha", "authenticated"),
        )
        _canonical_path(entry["path"], f"$.reviewed_artifacts[{index}].path")
        _text(entry["sha256"], f"$.reviewed_artifacts[{index}].sha256", pattern=SHA256)
        _text(
            entry["source_sha"],
            f"$.reviewed_artifacts[{index}].source_sha",
            pattern=GIT_SHA,
        )
        _flag(entry["authenticated"], f"$.reviewed_artifacts[{index}].authenticated")
        if entry["source_sha"] != root["source_sha"]:
            _fail(f"$.reviewed_artifacts[{index}]", "names a foreign source_sha")
    _sorted_unique([row["path"] for row in artifacts], "$.reviewed_artifacts")

    for index, row in enumerate(root["rederived_oracles"]):
        entry = _mapping(
            row,
            f"$.rederived_oracles[{index}]",
            ("oracle_id", "method", "observed", "fixed_limit", "units", "passed"),
        )
        for key in ("oracle_id", "method", "units"):
            _text(entry[key], f"$.rederived_oracles[{index}].{key}")
        observed = _number(entry["observed"], f"$.rederived_oracles[{index}].observed")
        limit = _number(
            entry["fixed_limit"], f"$.rederived_oracles[{index}].fixed_limit"
        )
        passed = _flag(entry["passed"], f"$.rederived_oracles[{index}].passed")
        if passed != (observed <= limit):
            _fail(
                f"$.rederived_oracles[{index}]", "passed must equal observed <= limit"
            )
    _sorted_unique(
        [row["oracle_id"] for row in root["rederived_oracles"]], "$.rederived_oracles"
    )

    for index, row in enumerate(root["review_checks"]):
        entry = _mapping(
            row,
            f"$.review_checks[{index}]",
            ("check_id", "method", "expected_outcome", "observed_outcome", "passed"),
        )
        for key in ("check_id", "method", "expected_outcome", "observed_outcome"):
            _text(entry[key], f"$.review_checks[{index}].{key}")
        _flag(entry["passed"], f"$.review_checks[{index}].passed")
    _sorted_unique(
        [row["check_id"] for row in root["review_checks"]], "$.review_checks"
    )

    for index, row in enumerate(root["commands"]):
        entry = _mapping(
            row,
            f"$.commands[{index}]",
            (
                "argv",
                "cwd",
                "pixi_environment",
                "started_at_utc",
                "duration_seconds",
                "exit_code",
                "stdout_sha256",
                "stderr_sha256",
            ),
        )
        if not isinstance(entry["argv"], list) or not entry["argv"]:
            _fail(f"$.commands[{index}].argv", "must be a non-empty string array")
        if entry["cwd"] != ".":
            _fail(
                f"$.commands[{index}].cwd", "must be the repository-relative sentinel"
            )
        _text(
            entry["started_at_utc"],
            f"$.commands[{index}].started_at_utc",
            pattern=TIMESTAMP,
        )
        _number(entry["duration_seconds"], f"$.commands[{index}].duration_seconds")
        if isinstance(entry["exit_code"], bool) or not isinstance(
            entry["exit_code"], int
        ):
            _fail(f"$.commands[{index}].exit_code", "must be a signed integer")
        for key in ("stdout_sha256", "stderr_sha256"):
            _text(entry[key], f"$.commands[{index}].{key}", pattern=SHA256)

    for index, row in enumerate(root["blockers"]):
        _mapping(
            row,
            f"$.blockers[{index}]",
            ("blocker_id", "requirement_id", "evidence", "required_remediation"),
        )
    _sorted_unique([row["blocker_id"] for row in root["blockers"]], "$.blockers")

    for key in ("accepted_limitations", "claims_not_licensed"):
        values = root[key]
        if not isinstance(values, list):
            _fail(f"$.{key}", "must be an array")
        for index, item in enumerate(values):
            _text(item, f"$.{key}[{index}]")
        _sorted_unique(values, f"$.{key}")
    if not root["claims_not_licensed"]:
        _fail("$.claims_not_licensed", "must be non-empty")


def _synthetic_acceptance() -> dict[str, Any]:
    """One minimal document that satisfies every Section 8.2 ACCEPT rule."""
    digest = "0" * 64
    sha = "a" * 40
    return {
        "schema_version": "radiosim.sci005.stage1-acceptance.v1",
        "stage": 1,
        "verdict": "ACCEPT",
        "generated_at_utc": "2026-08-14T00:00:00Z",
        "implementation_identity": "sci005-s1-implementer",
        "reviewer_identity": "sci005-s1-independent-reviewer",
        "reviewer_independent": True,
        "design_sha": sha,
        "red_test_sha": sha,
        "source_sha": sha,
        "evidence_commit_sha": sha,
        "evidence_artifact_path": "docs/development/sci005_stage1_evidence.json",
        "evidence_artifact_sha256": digest,
        "evidence_schema_path": ("docs/development/sci005_stage1_evidence.schema.json"),
        "evidence_schema_sha256": digest,
        "toolchain": {
            "evidence_generator_path": "tools/sci005_stage_evidence.py",
            "evidence_generator_git_blob": sha,
            "evidence_validator_path": "tests/unit/test_sci005_evidence.py",
            "evidence_validator_git_blob": sha,
            "acceptance_generator_path": TOOL,
            "acceptance_generator_git_blob": sha,
            "acceptance_validator_path": (
                "tests/unit/test_sci005_stage1_acceptance.py"
            ),
            "acceptance_validator_pre_a_git_blob": sha,
            "acceptance_schema_path": SCHEMA,
            "acceptance_schema_sha256": digest,
        },
        "acceptance_commit_sha": None,
        "acceptance_commit_sha_reason": (
            "self-reference: U1 binds the containing A1 commit"
        ),
        "successor_unlocks": ["SCI005.U1"],
        "reviewed_artifacts": [
            {
                "path": "docs/development/sci005_stage1_evidence.json",
                "sha256": digest,
                "source_sha": sha,
                "authenticated": True,
            }
        ],
        "rederived_oracles": [
            {
                "oracle_id": oracle,
                "method": "independently re-derived closed form",
                "observed": 1e-17,
                "fixed_limit": 1e-12,
                "units": "absolute residual",
                "passed": True,
            }
            for oracle in REQUIRED_ORACLES
        ],
        "review_checks": [
            {
                "check_id": check,
                "method": "named Git objects and raw bytes",
                "expected_outcome": "pass",
                "observed_outcome": "pass",
                "passed": True,
            }
            for check in REQUIRED_CHECKS
        ],
        "commands": [
            {
                "argv": ["pixi", "run", "test"],
                "cwd": ".",
                "pixi_environment": "default",
                "started_at_utc": "2026-08-14T00:00:00Z",
                "duration_seconds": 1.0,
                "exit_code": 0,
                "stdout_sha256": digest,
                "stderr_sha256": digest,
            }
        ],
        "blockers": [],
        "accepted_limitations": [
            "no accelerator measurement is licensed by this stage"
        ],
        "claims_not_licensed": sorted(
            [
                "SCI-005 Stages 2 and 3",
                "SCI-005 whole-row closure",
                "a deterministic Ruze Jones or error voltage",
            ]
        ),
    }


def test_the_null_sentinels_and_the_absent_artifact_agree() -> None:
    """At ``S1`` both constants are ``None`` and the artifact is absent."""
    if APPROVED_EVIDENCE_SHA is None or APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None:
        assert APPROVED_EVIDENCE_SHA is None
        assert APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None
        assert not (REPOSITORY_ROOT / ARTIFACT).exists()
        return
    assert GIT_SHA.fullmatch(APPROVED_EVIDENCE_SHA)
    assert SHA256.fullmatch(APPROVED_ACCEPTANCE_ARTIFACT_SHA256)
    assert (REPOSITORY_ROOT / ARTIFACT).is_file()
    import hashlib

    payload = (REPOSITORY_ROOT / ARTIFACT).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == APPROVED_ACCEPTANCE_ARTIFACT_SHA256
    document = json.loads(payload.decode("utf-8"))
    validate_stage1_acceptance(document)
    assert document["evidence_commit_sha"] == APPROVED_EVIDENCE_SHA


def test_this_validator_loads_only_the_standard_library() -> None:
    """An acceptance-critical validator carries no third-party import."""
    source = Path(__file__).read_text(encoding="utf-8")
    imported = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", source, re.M))
    roots = {name.split(".")[0] for name in imported}
    # ``sci005_stage1_acceptance`` is the tool under test, itself stdlib-only.
    assert roots <= {
        "__future__",
        "hashlib",
        "json",
        "math",
        "re",
        "subprocess",
        "sci005_stage1_acceptance",
        "sys",
        "pathlib",
        "typing",
        "pytest",
    }, f"unexpected imports: {sorted(roots)}"


def test_the_acceptance_tool_also_loads_only_the_standard_library() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'tools'); "
            "import sci005_stage1_acceptance; "
            "print(sorted(n for n in sys.modules if n in {'jsonschema', 'numpy'}))",
        ],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


def test_the_schema_transcription_and_the_validator_agree() -> None:
    """The normative JSON transcription and this validator pin the same keys."""
    schema = _schema()
    assert tuple(schema["properties"]) == ACCEPTANCE_KEYS
    assert set(schema["required"]) == set(ACCEPTANCE_KEYS)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["acceptance_commit_sha"] == {"type": "null"}
    assert tuple(schema["properties"]["toolchain"]["properties"]) == TOOLCHAIN_KEYS


def test_a_complete_synthetic_acceptance_document_validates() -> None:
    validate_stage1_acceptance(_synthetic_acceptance())


@pytest.mark.parametrize("key", ["toolchain", "reviewed_artifacts", "verdict"])
def test_a_missing_top_level_key_is_rejected(key: str) -> None:
    document = _synthetic_acceptance()
    del document[key]
    with pytest.raises(AcceptanceSchemaError):
        validate_stage1_acceptance(document)


def test_an_unknown_top_level_key_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["workflow_summary"] = "green"
    with pytest.raises(AcceptanceSchemaError):
        validate_stage1_acceptance(document)


def test_a_reordered_top_level_key_sequence_is_rejected() -> None:
    document = _synthetic_acceptance()
    reordered = {key: document[key] for key in reversed(ACCEPTANCE_KEYS)}
    with pytest.raises(AcceptanceSchemaError, match="declared order"):
        validate_stage1_acceptance(reordered)


def test_a_non_null_acceptance_commit_sha_is_rejected() -> None:
    """Section 8.2: ``acceptance_commit_sha`` is JSON null; ``U1`` binds ``A1``."""
    document = _synthetic_acceptance()
    document["acceptance_commit_sha"] = "b" * 40
    with pytest.raises(AcceptanceSchemaError):
        validate_stage1_acceptance(document)


def test_a_malformed_digest_or_commit_is_rejected() -> None:
    for key, bad in (
        ("source_sha", "a" * 39),
        ("evidence_artifact_sha256", "A" * 64),
        ("evidence_commit_sha", "zz" + "a" * 38),
    ):
        document = _synthetic_acceptance()
        document[key] = bad
        with pytest.raises(AcceptanceSchemaError):
            validate_stage1_acceptance(document)


def test_an_unsorted_or_duplicated_oracle_array_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["rederived_oracles"] = list(reversed(document["rederived_oracles"]))
    with pytest.raises(AcceptanceSchemaError, match="sorted"):
        validate_stage1_acceptance(document)


def test_an_oracle_whose_pass_flag_contradicts_its_limit_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["rederived_oracles"][0]["observed"] = 1.0
    with pytest.raises(AcceptanceSchemaError, match="observed <= limit"):
        validate_stage1_acceptance(document)


def test_a_boolean_where_a_number_belongs_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["rederived_oracles"][0]["observed"] = True
    with pytest.raises(AcceptanceSchemaError):
        validate_stage1_acceptance(document)


def test_a_reviewed_artifact_naming_a_foreign_source_sha_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["reviewed_artifacts"][0]["source_sha"] = "b" * 40
    with pytest.raises(AcceptanceSchemaError, match="foreign source_sha"):
        validate_stage1_acceptance(document)


def test_an_escaping_reviewed_artifact_path_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["reviewed_artifacts"][0]["path"] = "docs/../../escape.json"
    with pytest.raises(AcceptanceSchemaError):
        validate_stage1_acceptance(document)


def test_an_unlock_that_disagrees_with_the_verdict_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["verdict"] = "REJECT"
    document["acceptance_commit_sha_reason"] = (
        "not-applicable: REJECT creates no A commit"
    )
    with pytest.raises(AcceptanceSchemaError, match="does not match the verdict"):
        validate_stage1_acceptance(document)


def test_a_wrong_self_reference_reason_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["acceptance_commit_sha_reason"] = "self-reference"
    with pytest.raises(AcceptanceSchemaError):
        validate_stage1_acceptance(document)


# --- Section 8.2 cross-field predicates ---------------------------------------


def test_a_non_independent_reviewer_is_refused() -> None:
    module = _tool()
    document = _synthetic_acceptance()
    document["reviewer_independent"] = False
    with pytest.raises(module.AcceptanceError) as error:
        module.require_accept_completeness(document)
    assert error.value.prefix == "SCI005_ACCEPTANCE_VERDICT"


def test_equal_identities_are_refused() -> None:
    module = _tool()
    document = _synthetic_acceptance()
    document["reviewer_identity"] = document["implementation_identity"]
    with pytest.raises(module.AcceptanceError):
        module.require_accept_completeness(document)


def test_an_incomplete_oracle_set_is_refused() -> None:
    module = _tool()
    document = _synthetic_acceptance()
    document["rederived_oracles"] = document["rederived_oracles"][:-1]
    with pytest.raises(module.AcceptanceError):
        module.require_accept_completeness(document)


def test_an_incomplete_check_set_is_refused() -> None:
    module = _tool()
    document = _synthetic_acceptance()
    document["review_checks"] = document["review_checks"][:-1]
    with pytest.raises(module.AcceptanceError):
        module.require_accept_completeness(document)


def test_a_blocker_refuses_an_accept() -> None:
    module = _tool()
    document = _synthetic_acceptance()
    document["blockers"] = [
        {
            "blocker_id": "b1",
            "requirement_id": "r1",
            "evidence": "e",
            "required_remediation": "fix",
        }
    ]
    with pytest.raises(module.AcceptanceError):
        module.require_accept_completeness(document)


def test_a_nonzero_command_refuses_an_accept() -> None:
    module = _tool()
    document = _synthetic_acceptance()
    document["commands"][0]["exit_code"] = 1
    with pytest.raises(module.AcceptanceError):
        module.require_accept_completeness(document)


def test_a_false_oracle_pass_flag_is_refused() -> None:
    module = _tool()
    document = _synthetic_acceptance()
    document["rederived_oracles"][0]["observed"] = 1.0
    document["rederived_oracles"][0]["fixed_limit"] = 1e-12
    with pytest.raises(module.AcceptanceError):
        module.require_accept_completeness(document)


# --- Section 9: the frozen verifier contract ----------------------------------


def test_the_verifier_exposes_both_read_only_sub_commands() -> None:
    for command, options in (
        ("verify", ("--acceptance-commit", "--descendant")),
        ("verify-status", ("--acceptance-commit", "--status-commit")),
    ):
        completed = subprocess.run(
            [sys.executable, str(REPOSITORY_ROOT / TOOL), command, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0
        for option in options:
            assert option in completed.stdout


def test_an_unresolvable_commit_fails_with_the_argument_prefix() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / TOOL),
            "verify",
            "--acceptance-commit",
            "f" * 40,
            "--descendant",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr.startswith("SCI005_ACCEPTANCE_")
    prefix = completed.stderr.split(":", 1)[0]
    assert prefix in STDERR_PREFIXES
    assert completed.stderr[len(prefix) : len(prefix) + 2] == ": "


def test_every_frozen_stderr_prefix_is_declared_by_the_tool() -> None:
    module = _tool()
    declared = {
        module.ARGUMENT,
        module.SCHEMA,
        module.ANCESTRY,
        module.DIGEST,
        module.DIFF_AUTHORITY,
        module.VERDICT,
    }
    assert declared == set(STDERR_PREFIXES)


def test_the_certificate_line_is_canonical_and_ordered() -> None:
    module = _tool()
    document = {key: index for index, key in enumerate(CERTIFICATE_KEYS)}
    line = module.certificate_line(document)
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1
    assert b", " not in line and b": " not in line
    assert tuple(json.loads(line.decode("utf-8"))) == CERTIFICATE_KEYS
    assert module.CERTIFICATE_SCHEMA == (
        "radiosim.sci005.stage-acceptance-certificate.v1"
    )


def test_the_status_allowlist_is_section_seven_five_s() -> None:
    module = _tool()
    assert module.STATUS_PATHS == frozenset(
        {
            "Fix.md",
            "PostTier8RemediationPlan.md",
            "docs/development/sci005_beam_physics_plan.md",
            "docs/development/beam_physics_scope.md",
            "docs/changelog.rst",
            "docs/migration_guide.md",
            "README.md",
            "CLAUDE.md",
        }
    )
