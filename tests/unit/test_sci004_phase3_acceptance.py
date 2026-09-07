"""Strict authentication of the SCI-004 phase-M3 independent acceptance record.

``docs/development/sci004_mmode_design.md`` Sections 13.5, 14.3 and 14.4 freeze
this module's successor authority: terminal ``S3`` has both approved
constants as the literal ``None`` and the official acceptance path **absent**,
and every synthetic schema fixture passing. During the incomplete source range,
D31 permits only the authenticated historical REJECT artifact to remain; it
never activates current acceptance and must be disposed before terminal S3.
``A3`` then changes *only* the two
constants below and adds the acceptance JSON, plus the status prose Section 13.5
authorizes.  No import, expression, annotation, key, surrounding token, or other
literal in either assignment may change, so this module's own token stream
outside those two spans is comparable to its direct-parent ``E3`` bytes.

In the ``A3`` state the validator authenticates the approved ``E3``, the raw
acceptance bytes, the unique introducing ``A3`` commit and the exact ``E3..A3``
diff authority.  It never requires the evidence artifact's ``source_sha`` to
equal ``E3``: Section 14.2 binds that value to ``S3``.

Section 14.3's ``A3`` obligations are wider than ``A2``'s because ``E3`` retains
two artifacts.  The tests below therefore also require the record to have read
and hashed the retained Section 11 performance record at its host-bound path,
and -- once the record exists -- re-run the active performance validation, join
the ordered workload identities, and require a release scan that still
reports ``SCI-004`` as ROADMAP.  No elapsed-time threshold is asserted anywhere:
Section 11's timings gate nothing.

Importing this module loads only the Python standard library plus ``pytest``.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tokenize
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Section 14.3's two approved constants.  ``A3`` replaces exactly these two
#: ``None`` literals and nothing else in this module.
APPROVED_EVIDENCE_SHA: str | None = None
APPROVED_ACCEPTANCE_ARTIFACT_SHA256: str | None = None

# Historical rejected attempt, never eligible as the current approval.
HISTORICAL_REJECT_A = "8529da951e2378115ffde8d5da3e2af56f3323d0"
HISTORICAL_REJECT_E = "886e62fd9f8328826b388b8960ed7413da26b6d1"
HISTORICAL_REJECT_S = "b07925ab14b56b3ca0fa863f806290748a31df6b"
HISTORICAL_REJECT_SHA256 = (
    "283fb5264f5ecd86aed1300ae504b85946cf1f4d36b1c4c09bc92bb4f269421d"
)

TOOL = "tools/sci004_mmode_phase3_acceptance.py"
ARTIFACT = "docs/development/sci004_mmode_phase3_acceptance.json"
EVIDENCE_ARTIFACT = "docs/development/sci004_mmode_phase3_evidence.json"
EVIDENCE_GENERATOR = "tools/sci004_mmode_phase3_evidence.py"
EVIDENCE_VALIDATOR = "tests/unit/test_sci004_phase3_evidence.py"
VALIDATOR = "tests/unit/test_sci004_phase3_acceptance.py"
PERFORMANCE_DIRECTORY = "output/benchmarks/reference/sci004"

#: Section 13.5's complete ``A3`` write authority.
A3_AUTHORIZED_PATHS: frozenset[str] = frozenset(
    {
        ARTIFACT,
        VALIDATOR,
        "docs/development/sci004_mmode_design.md",
        "PostTier8RemediationPlan.md",
        "docs/changelog.rst",
        "docs/migration_guide.md",
        "docs/development/completion_ledger.md",
    }
)

#: Section 14.3: "no production-source path in the ``E..A`` diff".
PRODUCTION_PREFIXES: tuple[str, ...] = ("src/", "tools/")

#: The two spans Section 13.5 lets ``A3`` rewrite inside this module.
APPROVED_CONSTANT_NAMES: tuple[str, ...] = (
    "APPROVED_EVIDENCE_SHA",
    "APPROVED_ACCEPTANCE_ARTIFACT_SHA256",
)

GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
PERFORMANCE_PATH = re.compile(
    r"\Aoutput/benchmarks/reference/sci004/"
    r"\d{8}T\d{6}Z-[a-z0-9][a-z0-9-]{0,62}\.json\Z"
)

ACCEPTANCE_SCHEMA = "radiosim.sci004.mmode-phase3-acceptance.v1"
SELF_REFERENCE_REASON = "self-reference: the next R or C binds the containing A commit"

#: Section 14.3's exact top-level key order.
ACCEPTANCE_KEYS: tuple[str, ...] = (
    "schema_version",
    "phase",
    "verdict",
    "generated_at_utc",
    "reviewer_identity",
    "reviewer_independent",
    "design_sha",
    "red_commit_sha",
    "source_sha",
    "evidence_commit_sha",
    "evidence_artifact_path",
    "evidence_artifact_sha256",
    "acceptance_commit_sha",
    "acceptance_commit_sha_reason",
    "reviewed_artifacts",
    "rederived_oracles",
    "commands",
    "blockers",
    "accepted_limitations",
    "claims_not_licensed",
)

#: Section 14.3's ten required ``A3`` re-derivation identifiers, restated here
#: rather than imported so a silently shortened generator list fails.
REQUIRED_ORACLES: tuple[str, ...] = (
    "m3.sci005-dependency-gate",
    "m3.performance-schema",
    "m3.performance-provenance",
    "m3.performance-inventory",
    "m3.performance-schedule",
    "m3.performance-timing",
    "m3.performance-memory",
    "m3.performance-direct-predicate",
    "m3.performance-backend-predicate",
    "m3.fingerprint-authentication",
)

#: The three accepted-correction deferrals plus the two standing non-claims.
REQUIRED_CLAIM_TOPICS: tuple[str, ...] = (
    "accelerator",
    "diffuse",
    "end-to-end-backend",
    "non-scalar-beam",
    "performance",
)

#: The fixed artifacts every ``A3`` record must carry, restated locally.
REQUIRED_REVIEWED_PATHS: tuple[str, ...] = (
    EVIDENCE_ARTIFACT,
    EVIDENCE_VALIDATOR,
    EVIDENCE_GENERATOR,
    VALIDATOR,
    "docs/development/sci004_mmode_phase3_red_failures.json",
    "docs/development/sci004_mmode_phase3_sci005_dependency.json",
    "tests/unit/test_sci004_phase3_dependency.py",
    "tests/characterization/test_sci004_mmode.py",
)

FORTY = "0" * 39 + "1"
OTHER_FORTY = "0" * 39 + "2"
SIXTY_FOUR = "0" * 63 + "1"
SYNTHETIC_PERFORMANCE_PATH = (
    f"{PERFORMANCE_DIRECTORY}/20260824T120000Z-synthetic-host.json"
)


def _tool() -> Any:
    """Import the tracked generator without adding an import-time dependency."""
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    try:
        import sci004_mmode_phase3_acceptance as module
    finally:
        sys.path.pop(0)
    return module


def _synthetic_record(verdict: str = "ACCEPT") -> dict[str, Any]:
    """Return a complete synthetic record satisfying every Section 14.3 rule."""
    oracles = [
        {
            "oracle_id": name,
            "method": "independent re-derivation, dimensionless residual",
            "observed": 0.0,
            "fixed_limit": 5e-12,
            "pass": True,
        }
        for name in REQUIRED_ORACLES
    ]
    reviewed = sorted(
        (
            {
                "path": path,
                "sha256": SIXTY_FOUR,
                "source_sha": FORTY,
                "authenticated": True,
            }
            for path in (*REQUIRED_REVIEWED_PATHS, SYNTHETIC_PERFORMANCE_PATH)
        ),
        key=lambda row: str(row["path"]),
    )
    return {
        "schema_version": ACCEPTANCE_SCHEMA,
        "phase": "M3",
        "verdict": verdict,
        "generated_at_utc": "2026-08-24T00:00:00Z",
        "reviewer_identity": "sci004-a3-independent-reviewer",
        "reviewer_independent": True,
        # Section 13.7/14.4 supersede the design between ``R3`` and ``S3``, so
        # the operative ``design_sha`` differs from the red record's binding.
        "design_sha": FORTY,
        "red_commit_sha": OTHER_FORTY,
        "source_sha": FORTY,
        "evidence_commit_sha": FORTY,
        "evidence_artifact_path": EVIDENCE_ARTIFACT,
        "evidence_artifact_sha256": SIXTY_FOUR,
        "acceptance_commit_sha": None,
        "acceptance_commit_sha_reason": SELF_REFERENCE_REASON,
        "reviewed_artifacts": reviewed,
        "rederived_oracles": oracles,
        "commands": [
            {
                "argv": [
                    "pixi",
                    "run",
                    "python",
                    EVIDENCE_GENERATOR,
                    "check",
                    "--artifact",
                    EVIDENCE_ARTIFACT,
                    "--performance",
                    SYNTHETIC_PERFORMANCE_PATH,
                ],
                "cwd": ".",
                "pixi_environment": "default",
                "started_at_utc": "2026-08-24T00:00:00Z",
                "duration_seconds": 1.0,
                "exit_code": 0,
                "stdout_sha256": SIXTY_FOUR,
                "stderr_sha256": SIXTY_FOUR,
            }
        ],
        "blockers": []
        if verdict == "ACCEPT"
        else [
            {
                "blocker_id": "b1",
                "requirement_id": "sci004.section-11.performance-record",
                "evidence": "a retained workload row did not re-derive",
                "required_remediation": "regenerate and re-review",
            }
        ],
        "accepted_limitations": [
            "phase M3 makes no speed, accelerator or regression-gate claim",
        ],
        "claims_not_licensed": sorted(
            [
                "accelerator: no GPU or other accelerator is exercised here",
                "diffuse: the public m-mode path rejects a HEALPix-bearing sky",
                "end-to-end-backend: wiring request.backend through the public "
                "dense stages is future red-sliced work",
                "non-scalar-beam: the public m-mode path rejects a non-scalar "
                "resolved beam system",
                "performance: no speedup, regression gate or PERF-001 statement "
                "is licensed here",
            ]
        ),
    }


def _rejects(module: Any, record: Any) -> str:
    with pytest.raises(module.AcceptanceError) as excinfo:
        module.validate_acceptance_document(record)
    return str(excinfo.value.detail)


# ---------------------------------------------------------------------------
# Pre-A3 state
# ---------------------------------------------------------------------------


def _historical_reject_bytes() -> bytes:
    """Authenticate the immutable rejected attempt independently of live pins."""
    assert _git("rev-list", "--parents", "-n", "1", HISTORICAL_REJECT_A).split() == [
        HISTORICAL_REJECT_A,
        HISTORICAL_REJECT_E,
    ]
    raw = _git_bytes("show", f"{HISTORICAL_REJECT_A}:{ARTIFACT}")
    assert hashlib.sha256(raw).hexdigest() == HISTORICAL_REJECT_SHA256
    record = json.loads(raw)
    assert record["verdict"] == "REJECT"
    assert record["evidence_commit_sha"] == HISTORICAL_REJECT_E
    assert record["source_sha"] == HISTORICAL_REJECT_S
    return raw


def _acceptance_lifecycle(
    root: Path, evidence_sha: str | None, artifact_sha256: str | None
) -> str:
    """Classify pre-A state without licensing intermediate S as terminal S."""
    assert (evidence_sha is None) == (artifact_sha256 is None), "mixed approval pins"
    artifact = root / ARTIFACT
    assert not artifact.is_symlink(), "acceptance artifact must not be a symlink"
    if evidence_sha is not None:
        assert type(evidence_sha) is str and GIT_SHA.fullmatch(evidence_sha)
        assert type(artifact_sha256) is str and SHA256.fullmatch(artifact_sha256)
        assert evidence_sha != HISTORICAL_REJECT_E, "rejected E cannot be approved"
        assert artifact_sha256 != HISTORICAL_REJECT_SHA256, "REJECT cannot be approved"
        return "current-approval"
    if not artifact.exists():
        return "unapproved-absent"
    assert artifact.is_file(), "retained historical REJECT must be a regular file"
    raw = artifact.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == HISTORICAL_REJECT_SHA256, (
        "unapproved artifact is not the exact historical REJECT"
    )
    assert raw == _historical_reject_bytes(), "historical REJECT blob differs"
    return "historical-reject"


def test_the_approval_lifecycle_keeps_rejected_history_separate() -> None:
    """D31 permits frozen rejected bytes only while the source range is incomplete."""
    _ = _acceptance_lifecycle(
        REPOSITORY_ROOT, APPROVED_EVIDENCE_SHA, APPROVED_ACCEPTANCE_ARTIFACT_SHA256
    )


def test_null_approval_pins_allow_absent_or_exact_historical_reject(
    tmp_path: Path,
) -> None:
    assert _acceptance_lifecycle(tmp_path, None, None) == "unapproved-absent"
    artifact = tmp_path / ARTIFACT
    artifact.parent.mkdir(parents=True)
    _ = artifact.write_bytes(_historical_reject_bytes())
    assert _acceptance_lifecycle(tmp_path, None, None) == "historical-reject"


@pytest.mark.parametrize(
    ("evidence_sha", "digest"),
    [
        (FORTY, None),
        (None, SIXTY_FOUR),
        (HISTORICAL_REJECT_E, HISTORICAL_REJECT_SHA256),
        (HISTORICAL_REJECT_E, SIXTY_FOUR),
        (FORTY, HISTORICAL_REJECT_SHA256),
    ],
)
def test_mixed_or_historical_approval_pins_are_rejected(
    tmp_path: Path, evidence_sha: str | None, digest: str | None
) -> None:
    with pytest.raises(AssertionError, match="mixed approval|cannot be approved"):
        _ = _acceptance_lifecycle(tmp_path, evidence_sha, digest)


@pytest.mark.parametrize("mutation", ["bytes", "verdict", "evidence", "source"])
def test_changed_historical_reject_is_not_a_pre_a_allowance(
    tmp_path: Path, mutation: str
) -> None:
    raw = _historical_reject_bytes()
    if mutation == "bytes":
        raw += b"\n"
    else:
        record = json.loads(raw)
        field = {
            "verdict": "verdict",
            "evidence": "evidence_commit_sha",
            "source": "source_sha",
        }[mutation]
        record[field] = "ACCEPT" if mutation == "verdict" else FORTY
        raw = json.dumps(record).encode()
    artifact = tmp_path / ARTIFACT
    artifact.parent.mkdir(parents=True)
    _ = artifact.write_bytes(raw)
    with pytest.raises(AssertionError, match="exact historical REJECT"):
        _ = _acceptance_lifecycle(tmp_path, None, None)


def test_historical_reject_symlink_is_not_a_pre_a_allowance(tmp_path: Path) -> None:
    target = tmp_path / "rejected.json"
    _ = target.write_bytes(_historical_reject_bytes())
    artifact = tmp_path / ARTIFACT
    artifact.parent.mkdir(parents=True)
    artifact.symlink_to(target)
    with pytest.raises(AssertionError, match="symlink"):
        _ = _acceptance_lifecycle(tmp_path, None, None)


def test_retained_historical_reject_keeps_acceptance_generation_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _tool()
    artifact = tmp_path / ARTIFACT
    artifact.parent.mkdir(parents=True)
    _ = artifact.write_bytes(_historical_reject_bytes())
    evidence = tmp_path / EVIDENCE_ARTIFACT
    _ = evidence.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "REPOSITORY_ROOT", tmp_path)

    def clean_git(*args: str) -> str:
        return FORTY if args == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(module, "_git", clean_git)
    with pytest.raises(module.AcceptanceError, match="already exists"):
        module.preflight()


def test_the_acceptance_generator_is_already_tracked_at_s3() -> None:
    """Section 14.3: the generator and validator are already tracked at ``S``."""
    assert (REPOSITORY_ROOT / TOOL).is_file()
    assert (REPOSITORY_ROOT / VALIDATOR).is_file()


def test_the_generator_imports_only_the_standard_library() -> None:
    """An acceptance-critical verifier carries no transitive package dependency."""
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    for forbidden in ("import numpy", "import astropy", "import pytest", "import yaml"):
        assert forbidden not in source, forbidden


def test_the_generator_refuses_before_the_evidence_commit_exists() -> None:
    """Section 14.3: the generator runs only from a globally clean exact ``E3``.

    At ``S3`` the phase evidence artifact does not exist yet, so the refusal
    names that; at ``E3`` the preflight passes and the empty review record is
    refused as a malformed argument; at ``A3`` and beyond the declared
    acceptance output already exists, so the no-overwrite rule refuses first;
    and a dirty tree refuses before any of those.  In every state the assertion
    names a *reason* rather than accepting any non-zero exit, which a generator
    that refused unconditionally would also produce, and the process always
    fails closed with a frozen prefix rather than a traceback.
    """
    module = _tool()
    artifact = REPOSITORY_ROOT / ARTIFACT
    before = artifact.read_bytes() if artifact.exists() else None
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / TOOL),
            "generate",
            "--review-record",
            "/dev/null",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert completed.stdout == ""
    prefixes = (
        module.ANCESTRY + ": ",
        module.DIGEST + ": ",
        module.ARGUMENT + ": ",
    )
    assert completed.stderr.startswith(prefixes)
    assert "Traceback" not in completed.stderr
    after = artifact.read_bytes() if artifact.exists() else None
    assert after == before
    reasons = (
        "not globally clean",
        "commit that adds the phase evidence artifact",
        "is not UTF-8 JSON",
        "already exists",
    )
    assert any(reason in completed.stderr for reason in reasons)


def test_the_generator_produces_at_a_clean_evidence_commit() -> None:
    """Section 14.3/14.4: ``generate`` is bound to a venue, not prohibited."""
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    body = source[source.index('if arguments.command == "generate":') :]
    body = body[: body.index("document = json.loads(")]
    assert "load_review_record(" in body
    assert "build_acceptance_document(state, review)" in body
    assert "validate_acceptance_document(document)" in body
    assert "write_atomic_no_overwrite(" in body
    assert "raise AcceptanceError(" not in body


def test_the_generator_checks_both_retained_artifacts() -> None:
    """Section 14.2/14.3: ``E3``'s declared set is two files, so both are checked."""
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    body = source[source.index("def _run_evidence_validator") :]
    body = body[: body.index("\ndef ")]
    assert '"--artifact"' in body
    assert '"--performance"' in body


# ---------------------------------------------------------------------------
# Synthetic strict schema fixtures
# ---------------------------------------------------------------------------


def test_the_synthetic_accept_record_satisfies_every_rule() -> None:
    """The fixture is a positive control for the rejections below."""
    module = _tool()
    record = module.validate_acceptance_document(_synthetic_record())
    assert set(record) == set(ACCEPTANCE_KEYS)


def test_the_synthetic_reject_record_satisfies_every_rule() -> None:
    module = _tool()
    record = module.validate_acceptance_document(_synthetic_record("REJECT"))
    assert record["verdict"] == "REJECT"


@pytest.mark.parametrize("key", ACCEPTANCE_KEYS)
def test_a_missing_top_level_key_is_rejected(key: str) -> None:
    module = _tool()
    record = _synthetic_record()
    record.pop(key)
    assert "acceptance document" in _rejects(module, record)


def test_an_unknown_top_level_key_is_rejected() -> None:
    module = _tool()
    record = _synthetic_record()
    record["extra"] = 1
    assert "acceptance document" in _rejects(module, record)


def test_a_phase2_schema_literal_is_rejected() -> None:
    module = _tool()
    record = _synthetic_record()
    record["schema_version"] = "radiosim.sci004.mmode-phase2-acceptance.v1"
    assert "frozen phase literal" in _rejects(module, record)


def test_accept_with_a_blocker_is_rejected() -> None:
    module = _tool()
    record = _synthetic_record()
    record["blockers"] = _synthetic_record("REJECT")["blockers"]
    assert "empty blockers array" in _rejects(module, record)


def test_reject_without_a_blocker_is_rejected() -> None:
    module = _tool()
    record = _synthetic_record("REJECT")
    record["blockers"] = []
    assert "at least one concrete blocker" in _rejects(module, record)


def test_accept_with_a_false_oracle_is_rejected() -> None:
    module = _tool()
    record = _synthetic_record()
    record["rederived_oracles"][3]["pass"] = False
    assert "no false oracle" in _rejects(module, record)


@pytest.mark.parametrize("oracle_id", REQUIRED_ORACLES)
def test_accept_missing_a_required_oracle_is_rejected(oracle_id: str) -> None:
    """Section 14.3's ten ``A3`` identifiers are required, not optional prose."""
    module = _tool()
    record = _synthetic_record()
    record["rederived_oracles"] = [
        row for row in record["rederived_oracles"] if row["oracle_id"] != oracle_id
    ]
    assert oracle_id in _rejects(module, record)


def test_a_duplicate_oracle_identifier_is_rejected() -> None:
    module = _tool()
    record = _synthetic_record()
    record["rederived_oracles"].append(copy.deepcopy(record["rederived_oracles"][0]))
    assert "unique" in _rejects(module, record)


def test_the_generator_and_validator_declare_the_same_ten_oracles() -> None:
    module = _tool()
    assert tuple(module.REQUIRED_ORACLES) == REQUIRED_ORACLES


def test_the_generator_documents_the_verbatim_a3_clause_mapping() -> None:
    """Each required identifier is traceable to a Section 14.3 clause."""
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    docstring = source[: source.index('"""', source.index('"""') + 3)]
    for oracle_id in REQUIRED_ORACLES:
        assert f"``{oracle_id}``" in docstring, oracle_id


@pytest.mark.parametrize("topic", REQUIRED_CLAIM_TOPICS)
def test_accept_missing_a_required_claim_topic_is_rejected(topic: str) -> None:
    """The three deferrals and the two standing non-claims."""
    module = _tool()
    record = _synthetic_record()
    record["claims_not_licensed"] = sorted(
        claim
        for claim in record["claims_not_licensed"]
        if not claim.startswith(topic + ":")
    )
    assert topic in _rejects(module, record)


def test_the_generator_and_validator_declare_the_same_claim_topics() -> None:
    module = _tool()
    assert tuple(module.REQUIRED_CLAIM_TOPICS) == REQUIRED_CLAIM_TOPICS


@pytest.mark.parametrize("path", REQUIRED_REVIEWED_PATHS)
def test_a_record_that_did_not_review_a_required_artifact_is_rejected(
    path: str,
) -> None:
    module = _tool()
    record = _synthetic_record()
    record["reviewed_artifacts"] = [
        row for row in record["reviewed_artifacts"] if row["path"] != path
    ]
    assert path in _rejects(module, record)


def test_a_record_without_the_retained_performance_record_is_rejected() -> None:
    """Section 14.3: the envelope alone leaves the measurements unauthenticated."""
    module = _tool()
    record = _synthetic_record()
    record["reviewed_artifacts"] = [
        row
        for row in record["reviewed_artifacts"]
        if not str(row["path"]).startswith(PERFORMANCE_DIRECTORY + "/")
    ]
    assert "exactly one retained Section 11 record" in _rejects(module, record)


def test_a_record_reviewing_two_performance_records_is_rejected() -> None:
    module = _tool()
    record = _synthetic_record()
    record["reviewed_artifacts"].append(
        {
            "path": f"{PERFORMANCE_DIRECTORY}/20260824T130000Z-other-host.json",
            "sha256": SIXTY_FOUR,
            "source_sha": FORTY,
            "authenticated": True,
        }
    )
    record["reviewed_artifacts"].sort(key=lambda row: str(row["path"]))
    assert "exactly one retained Section 11 record" in _rejects(module, record)


def test_an_unauthenticated_reviewed_artifact_is_rejected() -> None:
    module = _tool()
    record = _synthetic_record()
    record["reviewed_artifacts"][0]["authenticated"] = False
    assert "must be authenticated" in _rejects(module, record)


def test_a_dependent_reviewer_is_rejected() -> None:
    module = _tool()
    record = _synthetic_record()
    record["reviewer_independent"] = False
    assert "reviewer_independent" in _rejects(module, record)


def test_a_non_null_acceptance_commit_sha_is_rejected() -> None:
    module = _tool()
    record = _synthetic_record()
    record["acceptance_commit_sha"] = FORTY
    assert "JSON null" in _rejects(module, record)


def test_a_reworded_self_reference_reason_is_rejected() -> None:
    module = _tool()
    record = _synthetic_record()
    record["acceptance_commit_sha_reason"] = "self reference"
    assert "self-reference literal" in _rejects(module, record)


def test_a_non_finite_oracle_measurement_is_rejected() -> None:
    module = _tool()
    record = _synthetic_record()
    record["rederived_oracles"][0]["observed"] = float("inf")
    assert "finite number" in _rejects(module, record)


def test_a_boolean_oracle_measurement_is_rejected() -> None:
    module = _tool()
    record = _synthetic_record()
    record["rederived_oracles"][0]["fixed_limit"] = True
    assert "finite number" in _rejects(module, record)


def test_unsorted_claim_arrays_are_rejected() -> None:
    module = _tool()
    record = _synthetic_record()
    record["claims_not_licensed"] = list(reversed(record["claims_not_licensed"]))
    assert "sorted and unique" in _rejects(module, record)


def test_a_non_zero_command_exit_code_is_rejected() -> None:
    module = _tool()
    record = _synthetic_record()
    record["commands"][0]["exit_code"] = 1
    assert "exit_code" in _rejects(module, record)


def test_the_generator_does_not_equate_the_red_and_operative_design_sha() -> None:
    """Section 13.7: the two values are expected to differ."""
    module = _tool()
    record = _synthetic_record()
    assert record["design_sha"] != record["red_commit_sha"]
    module.validate_acceptance_document(record)
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    assert 'record["design_sha"] == record["red_commit_sha"]' not in source


def test_a_review_record_supplying_a_derived_field_is_rejected(tmp_path) -> None:
    """Section 14.3: a reviewer may not overwrite a derived field."""
    module = _tool()
    for field in module.DERIVED_FIELDS:
        payload = {
            "reviewer_identity": "r",
            "reviewer_independent": True,
            "verdict": "ACCEPT",
            "rederived_oracles": [],
            "blockers": [],
            "accepted_limitations": [],
            "claims_not_licensed": [],
            field: "x",
        }
        path = tmp_path / f"review-{field}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(module.AcceptanceError) as excinfo:
            module.load_review_record(path)
        assert field in str(excinfo.value.detail)


def test_a_review_record_missing_a_reviewer_key_is_rejected(tmp_path) -> None:
    module = _tool()
    for field in module.REVIEW_RECORD_KEYS:
        payload = {
            "reviewer_identity": "r",
            "reviewer_independent": True,
            "verdict": "ACCEPT",
            "rederived_oracles": [],
            "blockers": [],
            "accepted_limitations": [],
            "claims_not_licensed": [],
        }
        payload.pop(field)
        path = tmp_path / f"missing-{field}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(module.AcceptanceError):
            module.load_review_record(path)


def test_the_review_record_key_set_is_the_seven_reviewer_fields() -> None:
    module = _tool()
    assert set(module.REVIEW_RECORD_KEYS) == {
        "reviewer_identity",
        "reviewer_independent",
        "verdict",
        "rederived_oracles",
        "blockers",
        "accepted_limitations",
        "claims_not_licensed",
    }


def test_a_dependent_review_record_is_rejected(tmp_path) -> None:
    module = _tool()
    payload = {
        "reviewer_identity": "r",
        "reviewer_independent": False,
        "verdict": "ACCEPT",
        "rederived_oracles": [],
        "blockers": [],
        "accepted_limitations": [],
        "claims_not_licensed": [],
    }
    path = tmp_path / "dependent.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(module.AcceptanceError) as excinfo:
        module.load_review_record(path)
    assert "independent reviewer" in str(excinfo.value.detail)


def test_canonical_json_matches_the_evidence_tool_spelling() -> None:
    """The two tools must agree on Section 14's serialization byte for byte."""
    module = _tool()
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    try:
        import sci004_mmode_phase3_evidence as evidence
    finally:
        sys.path.pop(0)
    for value in ({"b": 1, "a": [1.0, 0.5, 1e-10]}, [True, False, None], 1e21, -0.5):
        assert module.canonical_json(value) == evidence.canonical_json(value), value


# ---------------------------------------------------------------------------
# A3-state commit shape
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


def _locate_acceptance_commit() -> str:
    """Select current A on HEAD's first-parent chain after approved E."""
    parent = APPROVED_EVIDENCE_SHA
    assert parent is not None and GIT_SHA.fullmatch(parent)
    assert parent in _git("rev-list", "--first-parent", "HEAD").split(), (
        "approved E is not on HEAD's first-parent chain"
    )
    assert len(_git("rev-list", "--parents", "-n", "1", parent).split()) == 2, (
        "approved E must have exactly one parent"
    )
    current = _git(
        "rev-list", "--first-parent", "--ancestry-path", "--reverse", f"{parent}..HEAD"
    ).split()
    assert current, "HEAD has no first-parent descendant after approved E"
    previous = parent
    for commit in current:
        assert GIT_SHA.fullmatch(commit)
        assert _git("rev-list", "--parents", "-n", "1", commit).split() == [
            commit,
            previous,
        ], "current acceptance ancestry must be a sole-parent chain"
        previous = commit
    return current[0]


def _authenticate_a_artifact(located: str) -> None:
    assert APPROVED_EVIDENCE_SHA is not None
    assert not _git_bytes("ls-tree", "-z", APPROVED_EVIDENCE_SHA, "--", ARTIFACT), (
        "the acceptance path must be absent at E"
    )
    entry = _git_bytes("ls-tree", "-z", located, "--", ARTIFACT)
    assert entry.startswith(b"100644 blob "), "A must add a regular acceptance file"
    payload = _git_bytes("show", f"{located}:{ARTIFACT}")
    assert hashlib.sha256(payload).hexdigest() == APPROVED_ACCEPTANCE_ARTIFACT_SHA256
    record = json.loads(payload)
    assert record["evidence_commit_sha"] == APPROVED_EVIDENCE_SHA


def _constant_spans(source: str) -> tuple[list[tuple[int, int]], list[list[Any]]]:
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


def test_the_record_introducing_commit_directly_parents_the_approved_evidence() -> None:
    """Section 14.3's ``A3`` ancestry clause, skipped until the constants flip."""
    if APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None or APPROVED_EVIDENCE_SHA is None:
        pytest.skip("the M3 acceptance record is authorized at A3")
    located = _locate_acceptance_commit()
    _authenticate_a_artifact(located)


def test_the_a3_diff_writes_only_the_section_13_5_authorized_paths() -> None:
    """Section 13.5/14.3: no production-source path in the ``E..A`` diff."""
    if APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None:
        pytest.skip("the M3 acceptance record is authorized at A3")
    located = _locate_acceptance_commit()
    changed = set(
        _git(
            "diff",
            "--no-ext-diff",
            "--no-renames",
            "--name-only",
            "-z",
            f"{located}^",
            located,
            "--",
        )
        .rstrip("\0")
        .split("\0")
    )
    assert {ARTIFACT, VALIDATOR} <= changed
    unauthorized = sorted(changed - A3_AUTHORIZED_PATHS)
    assert not unauthorized, (
        f"the A3 commit {located} writes {unauthorized}, which Section 13.5 "
        f"does not authorize; it may write only {sorted(A3_AUTHORIZED_PATHS)}"
    )
    # Section 14.3 states the same bound a second way -- "no production-source
    # path in the ``E..A`` diff" -- and it is checked separately rather than
    # left implied by the path list, because a future widening of that list
    # must not silently admit a production edit.
    production = sorted(
        path for path in changed if path.startswith(PRODUCTION_PREFIXES)
    )
    assert not production, (
        f"the A3 commit {located} touches production source {production}"
    )


def test_the_a3_diff_changes_only_the_two_approved_constant_assignments() -> None:
    """Section 14.3: this module's own ``A3`` diff is the two constants alone."""
    if APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None or APPROVED_EVIDENCE_SHA is None:
        pytest.skip("the M3 acceptance record is authorized at A3")
    located = _locate_acceptance_commit()
    parent = _git("rev-list", "--parents", "-n", "1", located).split()[1]
    for commit in (parent, located):
        assert _git_bytes("ls-tree", commit, "--", VALIDATOR).startswith(
            b"100644 blob "
        ), "the acceptance validator must retain its regular-file mode"
    before = _git("show", f"{parent}:{VALIDATOR}")
    after = _git("show", f"{located}:{VALIDATOR}")
    assert _outside_spans(before) == _outside_spans(after), (
        f"the A3 commit {located} changed this module outside the two approved "
        "constant assignments"
    )
    _spans_before, bodies_before = _constant_spans(before)
    _spans_after, bodies_after = _constant_spans(after)
    approved = (APPROVED_EVIDENCE_SHA, APPROVED_ACCEPTANCE_ARTIFACT_SHA256)
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
        assert expected[equal + 1] == (tokenize.NAME, "None")
        expected[equal + 1] = (tokenize.STRING, f'"{value}"')
        assert [(token.type, token.string) for token in body_after] == expected, (
            f"{name} changed tokens other than its approved value"
        )


def _topology_commit(
    files: Mapping[str, bytes | None], *, gitlink: str | None = None
) -> str:
    """Commit synthetic fixture bytes only in the monkeypatched temporary repo."""
    for name, raw in files.items():
        path = REPOSITORY_ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if raw is None:
            path.unlink()
        else:
            _ = path.write_bytes(raw)
    _ = _git("add", "--all")
    if gitlink is not None:
        _ = _git("update-index", "--add", "--cacheinfo", f"160000,{gitlink},{ARTIFACT}")
    _ = _git("commit", "--allow-empty", "-qm", "synthetic topology fixture")
    return _git("rev-parse", "HEAD").strip()


def _topology_merge(*parents: str) -> str:
    arguments = ["commit-tree", _git("write-tree").strip(), "-m", "synthetic merge"]
    for parent in parents:
        arguments.extend(("-p", parent))
    commit = _git(*arguments).strip()
    _ = _git("update-ref", "HEAD", commit)
    return commit


@pytest.fixture
def acceptance_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, str, str, bytes]:
    """Real add/delete/add history; no scientific runs or ambient Git writes."""
    _ = _tool()  # Load the real generator before redirecting repository reads.
    monkeypatch.setattr(sys.modules[__name__], "REPOSITORY_ROOT", tmp_path)
    _ = _git("init", "-q")
    _ = _git("config", "user.name", "Synthetic Fixture")
    _ = _git("config", "user.email", "fixture@example.invalid")
    _ = _git("config", "commit.gpgsign", "false")
    _ = _git("config", "core.autocrlf", "false")
    _ = _git("config", "core.hooksPath", "/dev/null")
    _ = _topology_commit({ARTIFACT: b'{"verdict":"REJECT"}\n'})
    source = _topology_commit({ARTIFACT: None})
    nulls = "".join(f"{name}: str | None = None\n" for name in APPROVED_CONSTANT_NAMES)
    evidence = _topology_commit({VALIDATOR: nulls.encode()})
    raw = json.dumps({"evidence_commit_sha": evidence}).encode() + b"\r\n"
    digest = hashlib.sha256(raw).hexdigest()
    approved = nulls.replace("= None", f'= "{evidence}"', 1)
    approved = approved.replace("= None", f'= "{digest}"', 1)
    acceptance = _topology_commit({ARTIFACT: raw, VALIDATOR: approved.encode()})
    monkeypatch.setattr(sys.modules[__name__], "APPROVED_EVIDENCE_SHA", evidence)
    monkeypatch.setattr(
        sys.modules[__name__], "APPROVED_ACCEPTANCE_ARTIFACT_SHA256", digest
    )
    return source, evidence, acceptance, raw


def _check_current_a() -> None:
    test_the_record_introducing_commit_directly_parents_the_approved_evidence()
    test_the_a3_diff_writes_only_the_section_13_5_authorized_paths()
    test_the_a3_diff_changes_only_the_two_approved_constant_assignments()


@pytest.mark.parametrize("attack", ["replacement", "graft", "caller-routing"])
def test_acceptance_tool_reads_original_ancestry_and_checkout(
    acceptance_git: tuple[str, str, str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    """Generator preflight and parent checks share the actual Git context."""
    _source, evidence, acceptance, _raw = acceptance_git
    head = _topology_commit({"later.txt": b"actual checkout"})
    module = _tool()
    monkeypatch.setattr(module, "REPOSITORY_ROOT", REPOSITORY_ROOT)
    if attack == "replacement":
        _ = _git("replace", head, acceptance)
    elif attack == "graft":
        _ = (REPOSITORY_ROOT / ".git/info/grafts").write_text(f"{head} {evidence}\n")
    else:
        for name in (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        ):
            monkeypatch.setenv(name, str(REPOSITORY_ROOT / "missing"))
    assert module._git("rev-parse", "HEAD").strip() == head
    assert module._git("rev-list", "--parents", "-n", "1", head).split() == [
        head,
        acceptance,
    ]
    _ = (REPOSITORY_ROOT / "unapproved.txt").write_text("visible dirty state")
    with pytest.raises(module.AcceptanceError, match="not globally clean"):
        module.preflight()


@pytest.mark.parametrize("registered", [False, True])
def test_acceptance_tool_ignores_local_worktree_redirection(
    acceptance_git: tuple[str, str, str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    registered: bool,
) -> None:
    """A clean foreign directory cannot hide edits in the intended checkout."""
    _source, _evidence, head, _raw = acceptance_git
    checkout = REPOSITORY_ROOT
    if registered:
        checkout = REPOSITORY_ROOT.parent / (REPOSITORY_ROOT.name + "-linked")
        _ = _git("worktree", "add", "--detach", "-q", str(checkout), head)
    try:
        clean = REPOSITORY_ROOT.parent / (REPOSITORY_ROOT.name + "-clean")
        for relative in _git("ls-files", "-z").rstrip("\0").split("\0"):
            target = clean / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _ = target.write_bytes((checkout / relative).read_bytes())
        _ = _git("config", "core.worktree", str(clean))
        changed = checkout / VALIDATOR
        _ = changed.write_bytes(changed.read_bytes() + b"# actual dirty checkout\n")
        module = _tool()
        monkeypatch.setattr(module, "REPOSITORY_ROOT", checkout)
        assert module._git("rev-parse", "HEAD").strip() == head
        assert VALIDATOR in module._git("status", "--porcelain=v1")
        with pytest.raises(module.AcceptanceError, match="not globally clean"):
            module.preflight()
    finally:
        _ = _git("config", "--unset", "core.worktree")
        if registered:
            _ = _git("worktree", "remove", "--force", str(checkout))


@pytest.mark.parametrize("attack", ["blob-replacement", "caller-routing"])
def test_historical_reject_reads_actual_raw_blob(
    acceptance_git: tuple[str, str, str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    """Both the parent check and retained bytes ignore hostile Git overlays."""
    source, _evidence, parent, _raw = acceptance_git
    raw = (
        json.dumps(
            {"verdict": "REJECT", "evidence_commit_sha": parent, "source_sha": source}
        ).encode()
        + b"\r\n"
    )
    rejected = _topology_commit({ARTIFACT: raw})
    module = sys.modules[__name__]
    for name, value in (
        ("HISTORICAL_REJECT_S", source),
        ("HISTORICAL_REJECT_E", parent),
        ("HISTORICAL_REJECT_A", rejected),
        ("HISTORICAL_REJECT_SHA256", hashlib.sha256(raw).hexdigest()),
    ):
        monkeypatch.setattr(module, name, value)
    if attack == "blob-replacement":
        actual_blob = _git("rev-parse", f"{rejected}:{ARTIFACT}").strip()
        forged_path = REPOSITORY_ROOT / "forged-blob.json"
        _ = forged_path.write_bytes(b'{"verdict":"ACCEPT"}\n')
        forged_blob = _git("hash-object", "-w", str(forged_path)).strip()
        _ = _git("replace", actual_blob, forged_blob)
    else:
        for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_OBJECT_DIRECTORY"):
            monkeypatch.setenv(name, str(REPOSITORY_ROOT / "absent"))
    assert _historical_reject_bytes() == raw


def test_acceptance_topology_uses_current_add_and_raw_bytes(
    acceptance_git: tuple[str, str, str, bytes],
) -> None:
    _source, _evidence, acceptance, _raw = acceptance_git
    _ = _topology_commit({"later.txt": b"later ordinary descendant"})
    assert (
        len(_git("log", "--diff-filter=A", "--format=%H", "--", ARTIFACT).split()) == 2
    )
    assert _locate_acceptance_commit() == acceptance
    _check_current_a()


@pytest.mark.parametrize(
    "mutation",
    ["at-e", "unrelated-e", "side-e", "merge-e", "merge-a", "later-merge", "gap"],
)
def test_acceptance_topology_rejects_hostile_ancestry(
    acceptance_git: tuple[str, str, str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source, evidence, acceptance, _raw = acceptance_git
    if mutation == "at-e":
        _ = _git("checkout", "--detach", "-q", evidence)
    elif mutation == "unrelated-e":
        orphan = _git("commit-tree", _git("write-tree").strip(), "-m", "orphan").strip()
        monkeypatch.setattr(sys.modules[__name__], "APPROVED_EVIDENCE_SHA", orphan)
    elif mutation == "side-e":
        _ = _topology_merge(source, evidence)
    elif mutation == "merge-e":
        _ = _git("checkout", "--detach", "-q", evidence)
        merged = _topology_merge(source, _git("rev-parse", f"{source}^").strip())
        monkeypatch.setattr(sys.modules[__name__], "APPROVED_EVIDENCE_SHA", merged)
        _ = _topology_commit({})
    elif mutation == "merge-a":
        _ = _topology_merge(evidence, source)
    elif mutation == "later-merge":
        _ = _topology_merge(acceptance, source)
    else:
        _ = _git("checkout", "--detach", "-q", evidence)
        _ = _topology_commit({"docs/development/completion_ledger.md": b"intervening"})
        _ = _topology_commit({ARTIFACT: _git_bytes("show", f"{acceptance}:{ARTIFACT}")})
    with pytest.raises(AssertionError):
        _check_current_a()


@pytest.mark.parametrize(
    "mutation",
    [
        "preexisting",
        "symlink",
        "gitlink",
        "wrong-bytes",
        "normalized-digest",
        "wrong-e-binding",
        "production",
        "tool",
        "unknown-path",
        "logic",
        "annotation",
        "expression",
        "comment",
        "missing-validator",
        "validator-mode",
        "wrong-pin",
    ],
)
def test_acceptance_topology_rejects_hostile_a_content(
    acceptance_git: tuple[str, str, str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _source, evidence, acceptance, raw = acceptance_git
    validator = _git_bytes("show", f"{acceptance}:{VALIDATOR}")
    _ = _git("checkout", "--detach", "-q", evidence)
    files = {ARTIFACT: raw, VALIDATOR: validator}
    if mutation == "preexisting":
        evidence = _topology_commit({ARTIFACT: raw})
        monkeypatch.setattr(sys.modules[__name__], "APPROVED_EVIDENCE_SHA", evidence)
    elif mutation == "wrong-bytes":
        files[ARTIFACT] += b" "
    elif mutation == "normalized-digest":
        digest = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
        monkeypatch.setattr(
            sys.modules[__name__], "APPROVED_ACCEPTANCE_ARTIFACT_SHA256", digest
        )
    elif mutation == "wrong-e-binding":
        files[ARTIFACT] = json.dumps({"evidence_commit_sha": acceptance}).encode()
        monkeypatch.setattr(
            sys.modules[__name__],
            "APPROVED_ACCEPTANCE_ARTIFACT_SHA256",
            hashlib.sha256(files[ARTIFACT]).hexdigest(),
        )
    elif mutation in {"production", "tool", "unknown-path"}:
        name = {
            "production": "src/forbidden file.py",
            "tool": TOOL,
            "unknown-path": "unknown.txt",
        }[mutation]
        files[name] = b"unauthorized"
    elif mutation == "logic":
        files[VALIDATOR] += b"changed = True\n"
    elif mutation == "annotation":
        files[VALIDATOR] = validator.replace(b"str | None", b"str", 1)
    elif mutation == "expression":
        files[VALIDATOR] = validator.replace(b"= ", b"= str(", 1).replace(
            b"\n", b")\n", 1
        )
    elif mutation == "comment":
        files[VALIDATOR] = validator.replace(b"\n", b"  # changed\n", 1)
    elif mutation == "missing-validator":
        del files[VALIDATOR]
    elif mutation == "wrong-pin":
        files[VALIDATOR] = validator.replace(evidence.encode(), acceptance.encode())
    if mutation in {"symlink", "gitlink"}:
        del files[ARTIFACT]
        if mutation == "symlink":
            path = REPOSITORY_ROOT / ARTIFACT
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to("missing-target")
    _ = _topology_commit(files, gitlink=acceptance if mutation == "gitlink" else None)
    if mutation == "validator-mode":
        _ = _git("update-index", "--chmod=+x", VALIDATOR)
        _ = _git("commit", "--amend", "--no-edit", "-q")
    if mutation in {"symlink", "gitlink"}:
        mode = b"120000" if mutation == "symlink" else b"160000"
        assert _git_bytes("ls-tree", "HEAD", "--", ARTIFACT).startswith(mode)
    with pytest.raises(AssertionError):
        _check_current_a()


@pytest.mark.parametrize(
    "overlay", ["replace", "replace-blob", "graft-file", "graft-environment"]
)
def test_acceptance_topology_rejects_history_overlays(
    acceptance_git: tuple[str, str, str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    overlay: str,
) -> None:
    _source, evidence, good, raw = acceptance_git
    validator = _git_bytes("show", f"{good}:{VALIDATOR}")
    _ = _git("checkout", "--detach", "-q", evidence)
    files = {ARTIFACT: raw, VALIDATOR: validator}
    if overlay == "replace":
        files["src/forbidden.py"] = b"unauthorized\n"
    elif overlay == "replace-blob":
        files[ARTIFACT] += b" "
    else:
        _ = _topology_commit({})  # Unruled gap: actual A is not E's direct child.
    bad = _topology_commit(files)
    with pytest.raises(AssertionError):
        _check_current_a()
    if overlay == "replace":
        _ = _git("replace", bad, good)
    elif overlay == "replace-blob":
        _ = _git(
            "replace",
            _git("rev-parse", f"{bad}:{ARTIFACT}").strip(),
            _git("rev-parse", f"{good}:{ARTIFACT}").strip(),
        )
    else:
        graft = REPOSITORY_ROOT / ".git/info/grafts"
        if overlay == "graft-environment":
            graft = REPOSITORY_ROOT / "external-graft"
            monkeypatch.setenv("GIT_GRAFT_FILE", str(graft))
        graft.parent.mkdir(parents=True, exist_ok=True)
        _ = graft.write_text(f"{bad} {evidence}\n")
    # Prove native Git really interprets the forged edge/tree in this fixture.
    apparent = (
        subprocess.check_output(
            ["git", "rev-list", "--parents", "-n", "1", bad], cwd=REPOSITORY_ROOT
        )
        .decode()
        .split()
    )
    assert apparent == [bad, evidence]
    apparent_paths = (
        subprocess.check_output(
            ["git", "diff", "--name-only", evidence, bad], cwd=REPOSITORY_ROOT
        )
        .decode()
        .splitlines()
    )
    assert set(apparent_paths) == {ARTIFACT, VALIDATOR}
    apparent_raw = subprocess.check_output(
        ["git", "show", f"{bad}:{ARTIFACT}"], cwd=REPOSITORY_ROOT
    )
    assert apparent_raw == raw
    with pytest.raises(AssertionError):
        _check_current_a()


def test_acceptance_topology_ignores_git_environment_redirects(
    acceptance_git: tuple[str, str, str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = acceptance_git
    monkeypatch.setenv("GIT_DIR", str(REPOSITORY_ROOT / "missing-repository"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(REPOSITORY_ROOT / "missing-objects"))
    _check_current_a()
    assert os.environ["GIT_DIR"].endswith("missing-repository")


def test_acceptance_topology_rejects_config_hidden_gitlink_change(
    acceptance_git: tuple[str, str, str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, evidence, good, _raw = acceptance_git
    _ = _git("checkout", "--detach", "-q", evidence)
    path = "forbidden-module"
    _ = _git("update-index", "--add", "--cacheinfo", f"160000,{source},{path}")
    _ = _git("commit", "--amend", "--no-edit", "-q")
    evidence = _git("rev-parse", "HEAD").strip()
    raw = json.dumps({"evidence_commit_sha": evidence}).encode()
    digest = hashlib.sha256(raw).hexdigest()
    validator = "".join(
        f'{name}: str | None = "{value}"\n'
        for name, value in zip(APPROVED_CONSTANT_NAMES, (evidence, digest), strict=True)
    ).encode()
    _ = _topology_commit({ARTIFACT: raw, VALIDATOR: validator})
    _ = _git("update-index", "--add", "--cacheinfo", f"160000,{good},{path}")
    _ = _git("commit", "--amend", "--no-edit", "-q")
    monkeypatch.setattr(sys.modules[__name__], "APPROVED_EVIDENCE_SHA", evidence)
    monkeypatch.setattr(
        sys.modules[__name__], "APPROVED_ACCEPTANCE_ARTIFACT_SHA256", digest
    )
    _ = _git("config", "diff.ignoreSubmodules", "all")
    apparent = (
        subprocess.check_output(
            ["git", "diff", "--name-only", evidence, "HEAD"], cwd=REPOSITORY_ROOT
        )
        .decode()
        .splitlines()
    )
    assert set(apparent) == {ARTIFACT, VALIDATOR}
    assert path in _git("diff", "--name-only", evidence, "HEAD").splitlines()
    with pytest.raises(AssertionError, match="forbidden-module"):
        _check_current_a()


def test_acceptance_topology_allows_authorized_ledger_companion(
    acceptance_git: tuple[str, str, str, bytes],
) -> None:
    _source, evidence, acceptance, raw = acceptance_git
    validator = _git_bytes("show", f"{acceptance}:{VALIDATOR}")
    _ = _git("checkout", "--detach", "-q", evidence)
    _ = _topology_commit(
        {
            ARTIFACT: raw,
            VALIDATOR: validator,
            "docs/development/completion_ledger.md": b"Synthetic status companion.\n",
        }
    )
    _check_current_a()


def test_the_retained_record_authenticates_against_the_approved_constants() -> None:
    """Section 14.3's ``A3`` state, skipped until the constants are flipped."""
    if APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None or APPROVED_EVIDENCE_SHA is None:
        pytest.skip("the M3 acceptance record is authorized at A3")
    payload = (REPOSITORY_ROOT / ARTIFACT).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == APPROVED_ACCEPTANCE_ARTIFACT_SHA256
    module = _tool()
    record = module.validate_acceptance_document(json.loads(payload.decode("utf-8")))
    assert record["evidence_commit_sha"] == APPROVED_EVIDENCE_SHA
    evidence = json.loads(
        (REPOSITORY_ROOT / EVIDENCE_ARTIFACT).read_bytes().decode("utf-8")
    )
    # Section 14.3: the evidence artifact's ``source_sha`` is ``S3``, never
    # ``E3``, and this never requires them equal.
    assert record["source_sha"] == evidence["source_sha"]
    assert (
        record["evidence_artifact_sha256"]
        == hashlib.sha256(
            (REPOSITORY_ROOT / EVIDENCE_ARTIFACT).read_bytes()
        ).hexdigest()
    )


def test_the_a3_record_authenticates_the_raw_performance_path_and_digest() -> None:
    """Section 14.3: the ``A3`` validator authenticates the raw record."""
    if APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None:
        pytest.skip("the M3 acceptance record is authorized at A3")
    record = json.loads((REPOSITORY_ROOT / ARTIFACT).read_bytes().decode("utf-8"))
    retained = [
        row
        for row in record["reviewed_artifacts"]
        if str(row["path"]).startswith(PERFORMANCE_DIRECTORY + "/")
    ]
    assert len(retained) == 1
    path = str(retained[0]["path"])
    assert PERFORMANCE_PATH.fullmatch(path)
    payload = (REPOSITORY_ROOT / path).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == retained[0]["sha256"]
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    try:
        import sci004_mmode_phase3_evidence as evidence_tool
    finally:
        sys.path.pop(0)
    benchmark = evidence_tool.validate_performance_document(
        json.loads(payload.decode("utf-8"))
    )
    evidence = json.loads(
        (REPOSITORY_ROOT / EVIDENCE_ARTIFACT).read_bytes().decode("utf-8")
    )
    bound = evidence["results"]["performance_record"]
    assert bound["path"] == path
    assert bound["sha256"] == retained[0]["sha256"]
    assert benchmark["provenance"]["source_sha"] == record["source_sha"]
    assert (
        benchmark["provenance"]["pixi_lock_sha256"]
        == (evidence["source_identities"]["pixi_lock_sha256"])
    )
    for bound_row, benchmark_row in zip(
        bound["workload_identities"], benchmark["workloads"], strict=True
    ):
        assert bound_row["workload_id"] == benchmark_row["workload_id"]
        assert bound_row["result_cube_sha256"] == benchmark_row["result_cube_sha256"]


def test_the_a3_state_requires_a_release_scan_that_still_reports_roadmap() -> None:
    """Section 14.3: ``A3`` requires ``SCI-004`` to still be ROADMAP."""
    if APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None:
        pytest.skip("the M3 acceptance record is authorized at A3")
    evidence = json.loads(
        (REPOSITORY_ROOT / EVIDENCE_ARTIFACT).read_bytes().decode("utf-8")
    )
    scans = evidence["results"]["release_scan_cases"]
    assert scans
    for scan in scans:
        assert scan["roadmap_occurrences"] >= 1
        assert scan["done_occurrences"] == 0
        assert scan["unsupported_claim_occurrences"] == 0
    register = (REPOSITORY_ROOT / "Fix.md").read_text(encoding="utf-8")
    rows = [line for line in register.splitlines() if line.startswith("| SCI-004 |")]
    assert rows
    assert all(line.split("|")[2].strip() == "ROADMAP" for line in rows)


def test_the_a3_validator_asserts_no_elapsed_time_threshold() -> None:
    """Section 14.3: "It asserts no elapsed-time threshold".

    The needles are assembled at run time so that naming them here does not
    make this module contain them and fail itself; the check is over this
    module's compiled name and attribute stream rather than its raw text, so a
    threshold hidden behind a different spelling is still found.
    """
    import ast

    source = (REPOSITORY_ROOT / VALIDATOR).read_text(encoding="utf-8")
    tree = ast.parse(source)
    timing_names = {"".join(("sample", "_seconds")), "".join(("duration", "_seconds"))}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        rendered = ast.dump(node)
        for name in timing_names:
            assert f"'{name}'" not in rendered, name
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "time" not in imported
    assert "timeit" not in imported
