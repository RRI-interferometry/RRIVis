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
import re
import subprocess
import sys
import tokenize
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
    completed = subprocess.run(
        ["git", "show", f"{HISTORICAL_REJECT_A}:{ARTIFACT}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, "historical REJECT blob is unavailable"
    raw = completed.stdout
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


def _git(*arguments: str) -> str:
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


def _locate_acceptance_commit() -> str:
    """Return the unique commit that introduced the acceptance artifact."""
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
    lineage = _git("rev-list", "--parents", "-n", "1", located).split()
    assert lineage[0] == located
    assert len(lineage) == 2, (
        f"the acceptance-introducing commit {located} must be a non-merge commit"
    )
    assert lineage[1] == APPROVED_EVIDENCE_SHA
    payload = _git("show", f"{located}:{ARTIFACT}")
    assert (
        hashlib.sha256(payload.encode("utf-8")).hexdigest()
        == APPROVED_ACCEPTANCE_ARTIFACT_SHA256
    )


def test_the_a3_diff_writes_only_the_section_13_5_authorized_paths() -> None:
    """Section 13.5/14.3: no production-source path in the ``E..A`` diff."""
    if APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None:
        pytest.skip("the M3 acceptance record is authorized at A3")
    located = _locate_acceptance_commit()
    changed = set(
        _git("diff-tree", "--no-commit-id", "--name-only", "-r", located).split()
    )
    assert ARTIFACT in changed
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
