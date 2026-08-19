"""Strict authentication of the SCI-005 Stage-3 acceptance record.

``docs/development/sci005_beam_physics_plan.md`` Section 7.5 freezes this
module's successor authority: it lands in ``S3`` with both approved constants
as the literal ``None``, the official acceptance path absent, and every
synthetic strict schema/digest/ancestry fixture passing. ``A3`` then changes
*only* the two constants below, from ``None`` to the exact lower-case 40- and
64-hexadecimal literals. No import, expression, annotation, key, surrounding
token, or other literal in either assignment may change, so the validator can
compare its own token stream outside those two spans to its direct-parent
``E3`` bytes.

Section 8.2 adds that the validator operates on named Git objects rather than
whichever file happens to be checked out, Section 8.3 adds the ``U2 ->* D3``
succession the Stage-3 tool authenticates from Git objects alone, and Section 9
freezes the read-only verifier's certificate, its key order, and its six stderr
prefixes. Sections 8.3 and 9 also make that same tool the closure-parent
certificate verifier, without adding a sub-command for the job.

Importing this module loads only the Python standard library plus ``pytest``,
following ``tools/wp7_perf001_cpu_evidence.py``: an acceptance-critical
validator must not depend on a package that is merely transitively present,
because a lock update could drop it and silently turn a hard authentication
into a collection error. ``docs/development/
sci005_stage3_acceptance.schema.json`` remains the normative transcription of
Section 8.2; :func:`validate_stage3_acceptance` enforces the same structure,
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

#: Section 7.5's two approved constants. ``A3`` replaces exactly these two
#: ``None`` literals and nothing else in this module.
# fmt: off
# The flipped 64-hex literals exceed the format line limit by design;
# Section 7.5's substitution is byte-exact and must not be rewrapped.
APPROVED_EVIDENCE_SHA: str | None = None
APPROVED_ACCEPTANCE_ARTIFACT_SHA256: str | None = None
# fmt: on

TOOL = "tools/sci005_stage3_acceptance.py"
ARTIFACT = "docs/development/sci005_stage3_acceptance.json"
SCHEMA = "docs/development/sci005_stage3_acceptance.schema.json"
EVIDENCE_ARTIFACT = "docs/development/sci005_stage3_evidence.json"
EVIDENCE_SCHEMA = "docs/development/sci005_stage3_evidence.schema.json"

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

#: Section 8.2's exact Stage-3 oracle identifiers.
REQUIRED_ORACLES = (
    "common_efield_normalization",
    "ludwig3_basis_conversion",
    "noncommuting_chain_order",
    "receptor_output_basis_factorization",
    "standard_output_roundtrip",
)
#: Section 8.2's six common review checks; the same set at every stage.
REQUIRED_CHECKS = (
    "artifact_authentication",
    "default_disabled_fingerprints",
    "diff_authority",
    "gate_replay",
    "production_data_flow",
    "typed_rejection",
)

SELF_REFERENCE_REASON = "self-reference: U3 binds the containing A3 commit"
REJECT_REASON = "not-applicable: REJECT creates no A commit"
UNLOCKS = ["SCI005.U3"]

#: Section 8.3's observed ``U2..D3`` interval, transcribed in ancestry order
#: from the memo header.
INTERVAL_KINDS: dict[str, str] = {
    "2adc2acca8606b3a9774e14f28725a5687c0ecc8": "superseded-design",
    "139a8e411da1f50be29cee94ee351009437e10bc": "superseded-red-slice",
    "9956e77477b0597129e71b38a183c8dcd3cb761e": "superseded-design",
    "ea06bc649ae9987253c8002150e21b03a842cb45": "superseded-red-slice",
}

#: The last commit of that interval; the operative ``D3`` is its single
#: first-parent child.
LAST_INTERVAL_COMMIT = "ea06bc649ae9987253c8002150e21b03a842cb45"

#: The six Section 7.4 test paths the first superseded red slice cut, which are
#: also the union bounding the kind.
FIRST_RED_SLICE_PATHS = frozenset(
    {
        "tests/fixtures/beamfits.py",
        "tests/integration/test_sci005_beam_physics.py",
        "tests/unit/test_core/test_beam_pyuvdata_contract.py",
        "tests/unit/test_core/test_sci005_full_efield.py",
        "tests/unit/test_io/test_sci005_beam_config.py",
        "tests/unit/test_jones/test_chain_order.py",
    }
)
#: The three the re-cut slice touched; unlike Stage 2, the two differ.
SECOND_RED_SLICE_PATHS = frozenset(
    {
        "tests/fixtures/beamfits.py",
        "tests/unit/test_core/test_beam_pyuvdata_contract.py",
        "tests/unit/test_core/test_sci005_full_efield.py",
    }
)

#: The retained Stage-1 and Stage-2 evidence and acceptance artifacts, their
#: schemas, their validators and their tools, which Section 8.3 requires to
#: remain byte-identical to ``U2`` across the whole starred interval.
RETAINED_EARLIER_STAGE_PATHS: tuple[str, ...] = (
    "docs/development/sci005_stage1_acceptance.json",
    "docs/development/sci005_stage1_acceptance.schema.json",
    "docs/development/sci005_stage1_evidence.json",
    "docs/development/sci005_stage1_evidence.schema.json",
    "docs/development/sci005_stage2_acceptance.json",
    "docs/development/sci005_stage2_acceptance.schema.json",
    "docs/development/sci005_stage2_evidence.json",
    "docs/development/sci005_stage2_evidence.schema.json",
    "tests/unit/test_sci005_evidence.py",
    "tests/unit/test_sci005_stage1_acceptance.py",
    "tests/unit/test_sci005_stage2_acceptance.py",
    "tools/sci005_stage1_acceptance.py",
    "tools/sci005_stage2_acceptance.py",
    "tools/sci005_stage_evidence.py",
)

#: ``U2`` itself, named once for the interval-boundary reads below.
STATUS_SUCCESSOR = "f275e7538a19f713b99e07563a1c5a2a45e83a3d"


def _tool() -> Any:
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    try:
        import sci005_stage3_acceptance as module
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


def validate_stage3_acceptance(document: Any) -> None:
    """Authenticate one Stage-3 acceptance document against Section 8.2."""
    root = _mapping(document, "$", ACCEPTANCE_KEYS)
    if root["schema_version"] != "radiosim.sci005.stage3-acceptance.v1":
        _fail("$.schema_version", "is not the Stage-3 acceptance literal")
    if root["stage"] != 3 or isinstance(root["stage"], bool):
        _fail("$.stage", "must be the integer 3")
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
    if root["evidence_artifact_path"] != EVIDENCE_ARTIFACT:
        _fail("$.evidence_artifact_path", "is not the frozen Stage-3 path")
    if root["evidence_schema_path"] != EVIDENCE_SCHEMA:
        _fail("$.evidence_schema_path", "is not the frozen Stage-3 path")

    toolchain = _mapping(root["toolchain"], "$.toolchain", TOOLCHAIN_KEYS)
    for key, value in toolchain.items():
        if key.endswith("_git_blob"):
            _text(value, f"$.toolchain.{key}", pattern=GIT_SHA)
        elif key.endswith("_sha256"):
            _text(value, f"$.toolchain.{key}", pattern=SHA256)
        else:
            _canonical_path(value, f"$.toolchain.{key}")
    if toolchain["acceptance_generator_path"] != TOOL:
        _fail("$.toolchain.acceptance_generator_path", "is not the Stage-3 tool")
    if toolchain["acceptance_validator_path"] != (
        "tests/unit/test_sci005_stage3_acceptance.py"
    ):
        _fail("$.toolchain.acceptance_validator_path", "is not this validator")
    if toolchain["acceptance_schema_path"] != SCHEMA:
        _fail("$.toolchain.acceptance_schema_path", "is not the Stage-3 schema")

    if root["acceptance_commit_sha"] is not None:
        _fail("$.acceptance_commit_sha", "is JSON null; U3 binds the containing A3")
    reason = root["acceptance_commit_sha_reason"]
    expected_reason = SELF_REFERENCE_REASON if verdict == "ACCEPT" else REJECT_REASON
    if reason != expected_reason:
        _fail("$.acceptance_commit_sha_reason", f"must be {expected_reason!r}")
    unlocks = root["successor_unlocks"]
    if not isinstance(unlocks, list):
        _fail("$.successor_unlocks", "must be an array")
    if unlocks != (list(UNLOCKS) if verdict == "ACCEPT" else []):
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
        "schema_version": "radiosim.sci005.stage3-acceptance.v1",
        "stage": 3,
        "verdict": "ACCEPT",
        "generated_at_utc": "2026-08-19T00:00:00Z",
        "implementation_identity": "sci005-s3-implementer",
        "reviewer_identity": "sci005-s3-independent-reviewer",
        "reviewer_independent": True,
        "design_sha": sha,
        "red_test_sha": "b" * 40,
        "source_sha": "c" * 40,
        "evidence_commit_sha": "d" * 40,
        "evidence_artifact_path": EVIDENCE_ARTIFACT,
        "evidence_artifact_sha256": digest,
        "evidence_schema_path": EVIDENCE_SCHEMA,
        "evidence_schema_sha256": digest,
        "toolchain": {
            "evidence_generator_path": "tools/sci005_stage_evidence.py",
            "evidence_generator_git_blob": sha,
            "evidence_validator_path": "tests/unit/test_sci005_evidence.py",
            "evidence_validator_git_blob": sha,
            "acceptance_generator_path": TOOL,
            "acceptance_generator_git_blob": sha,
            "acceptance_validator_path": (
                "tests/unit/test_sci005_stage3_acceptance.py"
            ),
            "acceptance_validator_pre_a_git_blob": sha,
            "acceptance_schema_path": SCHEMA,
            "acceptance_schema_sha256": digest,
        },
        "acceptance_commit_sha": None,
        "acceptance_commit_sha_reason": SELF_REFERENCE_REASON,
        "successor_unlocks": list(UNLOCKS),
        "reviewed_artifacts": [
            {
                "path": EVIDENCE_ARTIFACT,
                "sha256": digest,
                "source_sha": "c" * 40,
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
                "started_at_utc": "2026-08-19T00:00:00Z",
                "duration_seconds": 1.0,
                "exit_code": 0,
                "stdout_sha256": digest,
                "stderr_sha256": digest,
            }
        ],
        "blockers": [],
        "accepted_limitations": [
            "conversion, factorization and oracle comparisons are evaluated at "
            "stored-grid directions, where the accepted interpolation is exact"
        ],
        "claims_not_licensed": sorted(
            [
                "SCI-005 whole-row closure",
                "a measured-efield beam outside the accepted "
                "uvbeam_peak_common_v1 subset",
                "an accelerator performance or GPU claim",
            ]
        ),
    }


def test_the_null_sentinels_and_the_absent_artifact_agree() -> None:
    """At ``S3`` both constants are ``None`` and the artifact is absent."""
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
    validate_stage3_acceptance(document)
    assert document["evidence_commit_sha"] == APPROVED_EVIDENCE_SHA


def test_this_validator_loads_only_the_standard_library() -> None:
    """An acceptance-critical validator carries no third-party import."""
    source = Path(__file__).read_text(encoding="utf-8")
    imported = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", source, re.M))
    roots = {name.split(".")[0] for name in imported}
    # ``sci005_stage3_acceptance`` is the tool under test, itself stdlib-only.
    assert roots <= {
        "__future__",
        "hashlib",
        "json",
        "math",
        "re",
        "subprocess",
        "sci005_stage3_acceptance",
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
            "import sci005_stage3_acceptance; "
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
    assert schema["properties"]["stage"]["const"] == 3
    assert schema["properties"]["schema_version"]["const"] == (
        "radiosim.sci005.stage3-acceptance.v1"
    )
    assert schema["properties"]["acceptance_commit_sha_reason"]["enum"] == [
        SELF_REFERENCE_REASON,
        REJECT_REASON,
    ]
    assert schema["properties"]["successor_unlocks"]["items"]["enum"] == UNLOCKS
    assert schema["properties"]["evidence_artifact_path"]["const"] == EVIDENCE_ARTIFACT
    assert schema["properties"]["evidence_schema_path"]["const"] == EVIDENCE_SCHEMA


def test_a_complete_synthetic_acceptance_document_validates() -> None:
    validate_stage3_acceptance(_synthetic_acceptance())


@pytest.mark.parametrize("key", ["toolchain", "reviewed_artifacts", "verdict"])
def test_a_missing_top_level_key_is_rejected(key: str) -> None:
    document = _synthetic_acceptance()
    del document[key]
    with pytest.raises(AcceptanceSchemaError):
        validate_stage3_acceptance(document)


def test_an_unknown_top_level_key_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["workflow_summary"] = "green"
    with pytest.raises(AcceptanceSchemaError):
        validate_stage3_acceptance(document)


def test_a_reordered_top_level_key_sequence_is_rejected() -> None:
    document = _synthetic_acceptance()
    reordered = {key: document[key] for key in reversed(ACCEPTANCE_KEYS)}
    with pytest.raises(AcceptanceSchemaError, match="declared order"):
        validate_stage3_acceptance(reordered)


def test_the_stage2_schema_literal_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["schema_version"] = "radiosim.sci005.stage2-acceptance.v1"
    with pytest.raises(AcceptanceSchemaError, match="Stage-3 acceptance literal"):
        validate_stage3_acceptance(document)


def test_a_stage_two_declaration_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["stage"] = 2
    with pytest.raises(AcceptanceSchemaError, match="integer 3"):
        validate_stage3_acceptance(document)


def test_a_stage2_evidence_path_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["evidence_artifact_path"] = "docs/development/sci005_stage2_evidence.json"
    with pytest.raises(AcceptanceSchemaError, match="frozen Stage-3 path"):
        validate_stage3_acceptance(document)


def test_a_stage2_toolchain_path_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["toolchain"]["acceptance_generator_path"] = (
        "tools/sci005_stage2_acceptance.py"
    )
    with pytest.raises(AcceptanceSchemaError, match="Stage-3 tool"):
        validate_stage3_acceptance(document)


def test_a_non_null_acceptance_commit_sha_is_rejected() -> None:
    """Section 8.2: ``acceptance_commit_sha`` is JSON null; ``U3`` binds ``A3``."""
    document = _synthetic_acceptance()
    document["acceptance_commit_sha"] = "b" * 40
    with pytest.raises(AcceptanceSchemaError):
        validate_stage3_acceptance(document)


def test_the_stage2_self_reference_reason_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["acceptance_commit_sha_reason"] = (
        "self-reference: U2 and SCI004.M3 bind the containing A2 commit"
    )
    with pytest.raises(AcceptanceSchemaError, match="must be"):
        validate_stage3_acceptance(document)


def test_a_malformed_digest_or_commit_is_rejected() -> None:
    for key, bad in (
        ("source_sha", "a" * 39),
        ("evidence_artifact_sha256", "A" * 64),
        ("evidence_commit_sha", "zz" + "a" * 38),
    ):
        document = _synthetic_acceptance()
        document[key] = bad
        with pytest.raises(AcceptanceSchemaError):
            validate_stage3_acceptance(document)


def test_the_stage2_unlock_pair_is_rejected() -> None:
    """Section 8.2 grants Stage 3 exactly one unlock; the M3 pair is Stage 2's."""
    document = _synthetic_acceptance()
    document["successor_unlocks"] = ["SCI004.M3", "SCI005.U2"]
    with pytest.raises(AcceptanceSchemaError, match="does not match the verdict"):
        validate_stage3_acceptance(document)


def test_the_stage1_unlock_array_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["successor_unlocks"] = ["SCI005.U1"]
    with pytest.raises(AcceptanceSchemaError, match="does not match the verdict"):
        validate_stage3_acceptance(document)


def test_a_reject_that_still_grants_an_unlock_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["verdict"] = "REJECT"
    document["acceptance_commit_sha_reason"] = REJECT_REASON
    with pytest.raises(AcceptanceSchemaError, match="does not match the verdict"):
        validate_stage3_acceptance(document)


def test_an_unsorted_or_duplicated_oracle_array_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["rederived_oracles"] = list(reversed(document["rederived_oracles"]))
    with pytest.raises(AcceptanceSchemaError, match="sorted"):
        validate_stage3_acceptance(document)


def test_an_oracle_whose_pass_flag_contradicts_its_limit_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["rederived_oracles"][0]["observed"] = 1.0
    with pytest.raises(AcceptanceSchemaError, match="observed <= limit"):
        validate_stage3_acceptance(document)


def test_a_boolean_where_a_number_belongs_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["rederived_oracles"][0]["observed"] = True
    with pytest.raises(AcceptanceSchemaError):
        validate_stage3_acceptance(document)


def test_a_reviewed_artifact_naming_a_foreign_source_sha_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["reviewed_artifacts"][0]["source_sha"] = "b" * 40
    with pytest.raises(AcceptanceSchemaError, match="foreign source_sha"):
        validate_stage3_acceptance(document)


def test_an_escaping_reviewed_artifact_path_is_rejected() -> None:
    document = _synthetic_acceptance()
    document["reviewed_artifacts"][0]["path"] = "docs/../../escape.json"
    with pytest.raises(AcceptanceSchemaError):
        validate_stage3_acceptance(document)


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


def test_a_stage2_oracle_identifier_is_refused() -> None:
    module = _tool()
    document = _synthetic_acceptance()
    document["rederived_oracles"][0]["oracle_id"] = "squint_frequency_law"
    document["rederived_oracles"].sort(key=lambda row: row["oracle_id"])
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


def test_the_tool_declares_the_frozen_stage3_constants() -> None:
    module = _tool()
    assert module.STAGE == 3
    assert module.ACCEPTANCE_ARTIFACT == ARTIFACT
    assert module.ACCEPTANCE_SCHEMA == SCHEMA
    assert module.ACCEPTANCE_VALIDATOR == (
        "tests/unit/test_sci005_stage3_acceptance.py"
    )
    assert module.EVIDENCE_ARTIFACT == EVIDENCE_ARTIFACT
    assert module.EVIDENCE_SCHEMA == EVIDENCE_SCHEMA
    assert module.SELF_REFERENCE_REASON == SELF_REFERENCE_REASON
    assert module.REJECT_REASON == REJECT_REASON
    assert module.UNLOCKS == UNLOCKS
    assert module.UNLOCKS == sorted(module.UNLOCKS)
    assert module.REQUIRED_ORACLES == frozenset(REQUIRED_ORACLES)
    assert module.REQUIRED_CHECKS == frozenset(REQUIRED_CHECKS)
    assert module.EVIDENCE_APPROVED_SOURCE_CONSTANT == "APPROVED_STAGE3_SOURCE_SHA"
    assert module.EVIDENCE_APPROVED_DIGEST_CONSTANT == (
        "APPROVED_STAGE3_EVIDENCE_ARTIFACT_SHA256"
    )


def test_the_tool_anchors_the_edge_on_the_stage2_retained_surface() -> None:
    """Section 8.3 anchors ``U2 ->* D3`` on Stage 2's own approved constants."""
    module = _tool()
    assert module.STAGE2_ACCEPTANCE_ARTIFACT == (
        "docs/development/sci005_stage2_acceptance.json"
    )
    assert module.STAGE2_ACCEPTANCE_VALIDATOR == (
        "tests/unit/test_sci005_stage2_acceptance.py"
    )
    assert module.STAGE2_ACCEPTANCE_TOOL == "tools/sci005_stage2_acceptance.py"
    assert module.STAGE2_APPROVED_EVIDENCE_CONSTANT == "APPROVED_EVIDENCE_SHA"
    assert module.STAGE2_APPROVED_ARTIFACT_CONSTANT == (
        "APPROVED_ACCEPTANCE_ARTIFACT_SHA256"
    )


# --- Section 8.3: the starred ``U2 ->* D3`` succession -------------------------


def test_the_interval_table_is_the_header_enumerated_one() -> None:
    """Exactly four commits, their kinds, and their exact recorded path sets."""
    module = _tool()
    assert len(module.INTERVAL_COMMITS) == 4
    assert {sha: kind for sha, (kind, _paths) in module.INTERVAL_COMMITS.items()} == (
        INTERVAL_KINDS
    )
    memo = frozenset({"docs/development/sci005_beam_physics_plan.md"})
    assert module.INTERVAL_COMMITS["2adc2acca8606b3a9774e14f28725a5687c0ecc8"][1] == (
        memo
    )
    assert module.INTERVAL_COMMITS["139a8e411da1f50be29cee94ee351009437e10bc"][1] == (
        FIRST_RED_SLICE_PATHS
    )
    assert module.INTERVAL_COMMITS["9956e77477b0597129e71b38a183c8dcd3cb761e"][1] == (
        memo
    )
    assert module.INTERVAL_COMMITS["ea06bc649ae9987253c8002150e21b03a842cb45"][1] == (
        SECOND_RED_SLICE_PATHS
    )


def test_the_two_red_slices_differ_and_the_union_bounds_the_kind() -> None:
    """Unlike Stage 2's single slice, the recorded sets are not one frozenset."""
    module = _tool()
    assert module.FIRST_SUPERSEDED_RED_SLICE_PATHS == FIRST_RED_SLICE_PATHS
    assert module.SECOND_SUPERSEDED_RED_SLICE_PATHS == SECOND_RED_SLICE_PATHS
    assert module.SECOND_SUPERSEDED_RED_SLICE_PATHS < (
        module.FIRST_SUPERSEDED_RED_SLICE_PATHS
    )
    assert module.STAGE3_RED_SLICE_PATHS == (
        FIRST_RED_SLICE_PATHS | SECOND_RED_SLICE_PATHS
    )
    assert len(module.STAGE3_RED_SLICE_PATHS) == 6


def test_every_recorded_touch_set_matches_real_git_history() -> None:
    """The header table is checked against the objects, not against itself."""
    module = _tool()
    for commit, (_kind, recorded) in module.INTERVAL_COMMITS.items():
        assert module.touched_paths(commit) == recorded, commit


def test_the_retained_earlier_stage_surface_is_byte_identical_across_the_edge() -> None:
    """Section 8.3: no ``U2..D3`` commit may move a retained Stage-1/2 byte.

    The exact recorded-set comparison already makes this true by construction —
    no admissible kind names one of these paths — so this reads the objects and
    proves the construction rather than restating it.
    """
    module = _tool()
    design = _operative_design_commit(module)
    changed = module.run_git(
        "diff",
        "--name-only",
        f"{STATUS_SUCCESSOR}..{design}",
        "--",
        *RETAINED_EARLIER_STAGE_PATHS,
    ).split()
    assert changed == []


def test_the_operative_design_edge_authenticates_against_real_history() -> None:
    """The whole starred edge, read from Git objects at the operative ``D3``."""
    module = _tool()
    design = _operative_design_commit(module)
    module.authenticate_design_edge(design)


def test_the_located_stage2_acceptance_and_status_commits_are_the_recorded_ones() -> (
    None
):
    module = _tool()
    design = _operative_design_commit(module)
    accepted = module.locate_stage2_acceptance_commit(design)
    assert accepted == "7523706c8c8d480de079100bc21871eb5616536e"
    assert module.locate_status_successor(design, accepted) == STATUS_SUCCESSOR


def test_a_design_commit_equal_to_its_red_commit_is_refused() -> None:
    """The recorded evidence-generator defect fails succession authentication."""
    module = _tool()
    with pytest.raises(module.AcceptanceError) as error:
        module.authenticate_succession("a" * 40, "a" * 40)
    assert error.value.prefix == "SCI005_ACCEPTANCE_ANCESTRY"


def test_an_interval_commit_touching_a_foreign_path_is_refused() -> None:
    module = _tool()
    with pytest.raises(module.AcceptanceError) as error:
        module.require_interval_kind(
            "2adc2acca8606b3a9774e14f28725a5687c0ecc8",
            "superseded-design",
            frozenset(
                {
                    "docs/development/sci005_beam_physics_plan.md",
                    "src/radiosim/core/beam/runtime.py",
                }
            ),
        )
    assert error.value.prefix == "SCI005_ACCEPTANCE_DIFF_AUTHORITY"


def test_one_red_slice_recorded_set_does_not_authorize_the_other() -> None:
    """The per-commit recorded set is the authority, not the kind-wide union."""
    module = _tool()
    with pytest.raises(module.AcceptanceError) as error:
        module.require_interval_kind(
            "ea06bc649ae9987253c8002150e21b03a842cb45",
            "superseded-red-slice",
            module.FIRST_SUPERSEDED_RED_SLICE_PATHS,
        )
    assert error.value.prefix == "SCI005_ACCEPTANCE_DIFF_AUTHORITY"


def test_a_red_slice_reaching_outside_section_seven_four_is_refused() -> None:
    """``STAGE3_RED_SLICE_PATHS`` bounds the kind even against its own record.

    ``U2`` is a status successor: its touch set is admissible prose, never a
    Section 7.4 test path, so claiming it as a red slice fails on the union
    bound rather than on the recorded-set comparison.
    """
    module = _tool()
    with pytest.raises(module.AcceptanceError) as error:
        module.require_interval_kind(
            STATUS_SUCCESSOR,
            "superseded-red-slice",
            module.touched_paths(STATUS_SUCCESSOR),
        )
    assert error.value.prefix == "SCI005_ACCEPTANCE_DIFF_AUTHORITY"
    assert "Section 7.4 test path" in error.value.detail


def test_a_status_prose_commit_that_touched_tests_is_refused() -> None:
    """A red slice's touch set is not admissible for a status-prose kind."""
    module = _tool()
    with pytest.raises(module.AcceptanceError) as error:
        module.require_interval_kind(
            "ea06bc649ae9987253c8002150e21b03a842cb45",
            "status-prose",
            module.SECOND_SUPERSEDED_RED_SLICE_PATHS,
        )
    assert error.value.prefix == "SCI005_ACCEPTANCE_DIFF_AUTHORITY"


def test_a_superseded_design_commit_must_touch_the_memo() -> None:
    module = _tool()
    with pytest.raises(module.AcceptanceError) as error:
        module.require_interval_kind(
            "139a8e411da1f50be29cee94ee351009437e10bc",
            "superseded-design",
            module.FIRST_SUPERSEDED_RED_SLICE_PATHS,
        )
    assert error.value.prefix == "SCI005_ACCEPTANCE_DIFF_AUTHORITY"


def _operative_design_commit(module: Any) -> str:
    """Return the operative ``D3``: the newest header-recorded correction.

    The interval table names every commit between ``U2`` and ``D3``; ``D3`` is
    the single child of the last of them on the first-parent chain. Reading it
    that way keeps this test bound to the same Git objects the tool uses rather
    than to a second copy of the ``D3`` literal.
    """
    listing = module.run_git("rev-list", "--first-parent", "--parents", "HEAD")
    children = [
        line.split()[0]
        for line in listing.splitlines()
        if len(line.split()) == 2 and line.split()[1] == LAST_INTERVAL_COMMIT
    ]
    assert len(children) == 1, children
    return children[0]


# --- Section 9: the frozen verifier contract ----------------------------------


def test_the_verifier_exposes_both_read_only_sub_commands() -> None:
    """Both closure-parent forms; Section 8.3 adds no third sub-command."""
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


def test_the_tool_exposes_exactly_the_three_frozen_sub_commands() -> None:
    """The closure-parent obligation reuses ``verify``/``verify-status``."""
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / TOOL), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "{generate,verify,verify-status}" in completed.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["verify", "--acceptance-commit", "f" * 40, "--descendant", "HEAD"],
        ["verify-status", "--acceptance-commit", "f" * 40, "--status-commit", "INDEX"],
    ],
)
def test_an_unresolvable_commit_fails_with_a_frozen_prefix(argv: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / TOOL), *argv],
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


def test_the_tool_reads_parents_not_peels() -> None:
    """``<sha>^{commit}`` peels; only ``<sha>^`` is the direct parent.

    Section 8.1 records that confusion as the evidence generator's Stage-2
    defect. Inside this tool the same confusion would turn ``R3^ == D3`` into a
    tautology, so the distinction is pinned against real repository objects.
    """
    module = _tool()
    head = module.run_git("rev-parse", "HEAD").strip()
    assert module.run_git("rev-parse", f"{head}^{{commit}}").strip() == head
    parent = module.parent_of(head)
    assert GIT_SHA.fullmatch(parent)
    assert parent != head
    assert parent == module.run_git("rev-parse", "HEAD^").strip()


def test_the_red_commit_must_be_a_direct_child_of_the_design_commit() -> None:
    """``R3^ == D3`` against real history, and a foreign parent is refused."""
    module = _tool()
    design = _operative_design_commit(module)
    red_slice = _red_commit(module, design)
    module.authenticate_succession(design, red_slice)
    with pytest.raises(module.AcceptanceError) as error:
        module.authenticate_succession(module.parent_of(design), red_slice)
    assert error.value.prefix == "SCI005_ACCEPTANCE_ANCESTRY"


def _red_commit(module: Any, design: str) -> str:
    """Return ``R3``: the single first-parent child of the operative ``D3``."""
    listing = module.run_git("rev-list", "--first-parent", "--parents", "HEAD")
    children = [
        line.split()[0]
        for line in listing.splitlines()
        if len(line.split()) == 2 and line.split()[1] == design
    ]
    if not children:
        pytest.skip("R3 has not been cut yet; the D3 -> R3 edge cannot exist")
    assert len(children) == 1, children
    return children[0]
