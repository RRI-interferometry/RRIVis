#!/usr/bin/env python
"""Generate and verify the SCI-005 Stage-1 independent acceptance record.

Importing this module loads only the Python standard library, following
``tools/wp7_perf001_cpu_evidence.py``: an acceptance-critical verifier must not
depend on a package that is merely transitively present, because a lock update
could drop it and turn a hard refusal into an import error.
``docs/development/sci005_stage1_acceptance.schema.json`` stays the normative
transcription of Section 8.2, and the checks below enforce the same structure,
key order and encodings in their own code.

``docs/development/sci005_beam_physics_plan.md`` Sections 8.2 and 9 freeze this
tool's three sub-commands.

Generation is all-or-rollback and owns the complete admissible pre-``A1`` diff::

    pixi run python tools/sci005_stage1_acceptance.py generate \\
      --review-record <absolute-temporary-review-record.json>

The generator derives every commit, path, digest, toolchain, reviewed-artifact,
self-reference and unlock field; the caller supplies a verdict and its
measurements only, and cannot override a derived field. It runs from a globally
clean exact ``E1``, first invokes the active evidence validator, and for an
``ACCEPT`` prepares both the previously absent canonical JSON and the phase
validator with exactly ``APPROVED_EVIDENCE_SHA: None -> E1`` and
``APPROVED_ACCEPTANCE_ARTIFACT_SHA256: None -> sha256(JSON)``.

The read-only verifier is the complete SCI-005 export for the WP-9 M3
dependency::

    pixi run python tools/sci005_stage1_acceptance.py verify \\
      --acceptance-commit <A1> --descendant <SHA-or-HEAD>

It emits exactly one canonical UTF-8 JSON line on success. Failure emits no
certificate on stdout, exits non-zero, and writes a stderr line beginning with
exactly one of the six frozen prefixes, one colon, one space, and the detail.

A status successor is checked before and after commit with::

    pixi run python tools/sci005_stage1_acceptance.py verify-status \\
      --acceptance-commit <A1> --status-commit <U1-or-INDEX>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

STAGE = 1
CERTIFICATE_SCHEMA = "radiosim.sci005.stage-acceptance-certificate.v1"
ACCEPTANCE_ARTIFACT = "docs/development/sci005_stage1_acceptance.json"
ACCEPTANCE_SCHEMA = "docs/development/sci005_stage1_acceptance.schema.json"
ACCEPTANCE_VALIDATOR = "tests/unit/test_sci005_stage1_acceptance.py"
EVIDENCE_ARTIFACT = "docs/development/sci005_stage1_evidence.json"
EVIDENCE_SCHEMA = "docs/development/sci005_stage1_evidence.schema.json"
EVIDENCE_GENERATOR = "tools/sci005_stage_evidence.py"
EVIDENCE_VALIDATOR = "tests/unit/test_sci005_evidence.py"
ACCEPTANCE_GENERATOR = "tools/sci005_stage1_acceptance.py"

SELF_REFERENCE_REASON = "self-reference: U1 binds the containing A1 commit"
REJECT_REASON = "not-applicable: REJECT creates no A commit"
UNLOCKS = ["SCI005.U1"]

#: Section 9's six frozen stderr prefixes.
ARGUMENT = "SCI005_ACCEPTANCE_ARGUMENT"
SCHEMA = "SCI005_ACCEPTANCE_SCHEMA"
ANCESTRY = "SCI005_ACCEPTANCE_ANCESTRY"
DIGEST = "SCI005_ACCEPTANCE_DIGEST"
DIFF_AUTHORITY = "SCI005_ACCEPTANCE_DIFF_AUTHORITY"
VERDICT = "SCI005_ACCEPTANCE_VERDICT"

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

#: Section 8.2's exact review-record key set.
REVIEW_KEYS: tuple[str, ...] = (
    "generated_at_utc",
    "implementation_identity",
    "reviewer_identity",
    "reviewer_independent",
    "verdict",
    "rederived_oracles",
    "review_checks",
    "commands",
    "blockers",
    "accepted_limitations",
    "claims_not_licensed",
)

REQUIRED_ORACLES = frozenset(
    {
        "blocked_aperture_transform",
        "ruze_limit_oracle",
        "ruze_pair_oracle",
        "unmodified_profile_transform",
        "zernike_phase_transform",
    }
)
REQUIRED_CHECKS = frozenset(
    {
        "artifact_authentication",
        "default_disabled_fingerprints",
        "diff_authority",
        "gate_replay",
        "production_data_flow",
        "typed_rejection",
    }
)


class AcceptanceError(RuntimeError):
    """One acceptance failure carrying its frozen stderr prefix."""

    def __init__(self, prefix: str, detail: str) -> None:
        self.prefix = prefix
        self.detail = detail
        super().__init__(f"{prefix}: {detail}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def run_git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AcceptanceError(
            ANCESTRY, f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def resolve_commit(reference: str, prefix: str = ARGUMENT) -> str:
    try:
        return run_git("rev-parse", f"{reference}^{{commit}}").strip()
    except AcceptanceError as error:
        raise AcceptanceError(
            prefix, f"{reference!r} does not resolve to a commit"
        ) from error


def git_show(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AcceptanceError(DIGEST, f"{path} is absent at {commit}")
    return completed.stdout


def git_blob(commit: str, path: str) -> str:
    output = run_git("rev-parse", f"{commit}:{path}").strip()
    if len(output) != 40:
        raise AcceptanceError(DIGEST, f"{path} has no blob object at {commit}")
    return output


def is_ancestor(ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def read_strict_json(path: Path, prefix: str = SCHEMA) -> Any:
    if path.is_symlink() or not path.is_file():
        raise AcceptanceError(prefix, f"{path} is not a regular file")
    return parse_strict_json(path.read_text(encoding="utf-8"), str(path), prefix)


def parse_strict_json(text: str, origin: str, prefix: str = SCHEMA) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                raise AcceptanceError(prefix, f"duplicate JSON key {key!r} in {origin}")
            seen[key] = value
        return seen

    def reject_non_finite(_value: str) -> float:
        raise AcceptanceError(prefix, f"non-finite JSON number in {origin}")

    return json.loads(
        text, object_pairs_hook=reject_duplicates, parse_constant=reject_non_finite
    )


def canonical_json_bytes(document: Any) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")


def certificate_line(document: dict[str, Any]) -> bytes:
    """Serialize Section 9's canonical one-line certificate."""
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
TIMESTAMP = re.compile(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")

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


def validate_acceptance_document(document: Any) -> None:
    """Enforce Section 8.2's structure, key order and encodings, stdlib only."""
    if not isinstance(document, dict):
        raise AcceptanceError(SCHEMA, "the acceptance record must be a JSON object")
    if tuple(document) != ACCEPTANCE_KEYS:
        raise AcceptanceError(
            SCHEMA, "top-level keys are not in Section 8.2's declared order"
        )
    schema = json.loads(
        (REPOSITORY_ROOT / ACCEPTANCE_SCHEMA).read_text(encoding="utf-8")
    )
    if tuple(schema["properties"]) != ACCEPTANCE_KEYS:
        raise AcceptanceError(
            SCHEMA, "the schema transcription and this validator disagree on key order"
        )
    if document["schema_version"] != "radiosim.sci005.stage1-acceptance.v1":
        raise AcceptanceError(SCHEMA, "schema_version is not the Stage-1 literal")
    if document["stage"] != 1 or isinstance(document["stage"], bool):
        raise AcceptanceError(SCHEMA, "stage must be the integer 1")
    if document["verdict"] not in {"ACCEPT", "REJECT"}:
        raise AcceptanceError(SCHEMA, "verdict must be ACCEPT or REJECT")
    if TIMESTAMP.fullmatch(str(document["generated_at_utc"])) is None:
        raise AcceptanceError(SCHEMA, "generated_at_utc is not a canonical timestamp")
    for key in ("implementation_identity", "reviewer_identity"):
        if not isinstance(document[key], str) or not document[key]:
            raise AcceptanceError(SCHEMA, f"{key} must be a non-empty string")
    if not isinstance(document["reviewer_independent"], bool):
        raise AcceptanceError(SCHEMA, "reviewer_independent must be a boolean")
    for key in ("design_sha", "red_test_sha", "source_sha", "evidence_commit_sha"):
        if GIT_SHA.fullmatch(str(document[key])) is None:
            raise AcceptanceError(SCHEMA, f"{key} is not a lower-case git_sha")
    for key in ("evidence_artifact_sha256", "evidence_schema_sha256"):
        if SHA256.fullmatch(str(document[key])) is None:
            raise AcceptanceError(SCHEMA, f"{key} is not a lower-case sha256")
    if document["evidence_artifact_path"] != EVIDENCE_ARTIFACT:
        raise AcceptanceError(SCHEMA, "evidence_artifact_path is not the frozen path")
    if document["evidence_schema_path"] != EVIDENCE_SCHEMA:
        raise AcceptanceError(SCHEMA, "evidence_schema_path is not the frozen path")
    toolchain = document["toolchain"]
    if not isinstance(toolchain, dict) or tuple(toolchain) != TOOLCHAIN_KEYS:
        raise AcceptanceError(SCHEMA, "toolchain keys are not the declared ones")
    for key, value in toolchain.items():
        if key.endswith("_git_blob") and GIT_SHA.fullmatch(str(value)) is None:
            raise AcceptanceError(SCHEMA, f"toolchain.{key} is not a git blob name")
        if key.endswith("_sha256") and SHA256.fullmatch(str(value)) is None:
            raise AcceptanceError(SCHEMA, f"toolchain.{key} is not a sha256")
    if document["acceptance_commit_sha"] is not None:
        raise AcceptanceError(
            SCHEMA, "acceptance_commit_sha is JSON null; U1 binds the containing A1"
        )
    if document["acceptance_commit_sha_reason"] not in {
        SELF_REFERENCE_REASON,
        REJECT_REASON,
    }:
        raise AcceptanceError(
            SCHEMA, "acceptance_commit_sha_reason is not a frozen one"
        )
    for key in (
        "successor_unlocks",
        "reviewed_artifacts",
        "rederived_oracles",
        "review_checks",
        "commands",
        "blockers",
        "accepted_limitations",
        "claims_not_licensed",
    ):
        if not isinstance(document[key], list):
            raise AcceptanceError(SCHEMA, f"{key} must be an array")
    if not document["claims_not_licensed"]:
        raise AcceptanceError(SCHEMA, "claims_not_licensed must be non-empty")


def require_accept_completeness(document: dict[str, Any]) -> None:
    """Apply Section 8.2's ``ACCEPT`` cross-field predicates."""
    if document["verdict"] != "ACCEPT":
        return
    if not document["reviewer_independent"]:
        raise AcceptanceError(
            VERDICT, "a retained ACCEPT requires an independent reviewer"
        )
    if document["implementation_identity"] == document["reviewer_identity"]:
        raise AcceptanceError(
            VERDICT, "implementation and reviewer identities are equal"
        )
    if document["blockers"]:
        raise AcceptanceError(VERDICT, "an ACCEPT requires no blockers")
    if document["successor_unlocks"] != UNLOCKS:
        raise AcceptanceError(VERDICT, f"successor_unlocks must be {UNLOCKS}")
    if document["acceptance_commit_sha_reason"] != SELF_REFERENCE_REASON:
        raise AcceptanceError(
            VERDICT, "acceptance_commit_sha_reason is not the frozen one"
        )
    oracles = {row["oracle_id"] for row in document["rederived_oracles"]}
    if oracles != REQUIRED_ORACLES:
        raise AcceptanceError(
            VERDICT, f"rederived_oracles must be exactly {sorted(REQUIRED_ORACLES)}"
        )
    checks = {row["check_id"] for row in document["review_checks"]}
    if checks != REQUIRED_CHECKS:
        raise AcceptanceError(
            VERDICT, f"review_checks must be exactly {sorted(REQUIRED_CHECKS)}"
        )
    for row in document["rederived_oracles"]:
        if row["passed"] != (row["observed"] <= row["fixed_limit"]):
            raise AcceptanceError(
                VERDICT, f"oracle {row['oracle_id']} has a false pass flag"
            )
        if not row["passed"]:
            raise AcceptanceError(VERDICT, f"oracle {row['oracle_id']} did not pass")
    for row in document["review_checks"]:
        if not row["passed"] or row["expected_outcome"] != row["observed_outcome"]:
            raise AcceptanceError(VERDICT, f"check {row['check_id']} did not pass")
    for row in document["commands"]:
        if row["exit_code"] != 0:
            raise AcceptanceError(
                VERDICT, "an ACCEPT requires every exit code to be zero"
            )
    for row in document["reviewed_artifacts"]:
        if not row["authenticated"]:
            raise AcceptanceError(
                VERDICT, f"artifact {row['path']} is not authenticated"
            )
        if row["source_sha"] != document["source_sha"]:
            raise AcceptanceError(
                VERDICT, f"artifact {row['path']} names a foreign source_sha"
            )
    for name, rows in (
        ("reviewed_artifacts", document["reviewed_artifacts"]),
        ("rederived_oracles", document["rederived_oracles"]),
        ("review_checks", document["review_checks"]),
        ("blockers", document["blockers"]),
    ):
        key = {
            "reviewed_artifacts": "path",
            "rederived_oracles": "oracle_id",
            "review_checks": "check_id",
            "blockers": "blocker_id",
        }[name]
        keys = [row[key] for row in rows]
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise AcceptanceError(SCHEMA, f"{name} is not sorted by unique {key}")


def verify(acceptance_commit: str, descendant: str) -> bytes:
    """Read-only verification returning Section 9's canonical certificate line."""
    accepted = resolve_commit(acceptance_commit)
    target = resolve_commit(descendant)
    if not is_ancestor(accepted, target):
        raise AcceptanceError(ANCESTRY, f"{accepted} is not an ancestor of {target}")
    payload = git_show(accepted, ACCEPTANCE_ARTIFACT)
    document = parse_strict_json(payload.decode("utf-8"), ACCEPTANCE_ARTIFACT)
    validate_acceptance_document(document)
    require_accept_completeness(document)
    if document["verdict"] != "ACCEPT":
        raise AcceptanceError(VERDICT, "the retained verdict is not ACCEPT")

    evidence_commit = document["evidence_commit_sha"]
    parent = run_git("rev-parse", f"{accepted}^").strip()
    if parent != evidence_commit:
        raise AcceptanceError(
            ANCESTRY,
            f"A1^ is {parent}, not the bound evidence commit {evidence_commit}",
        )
    evidence_bytes = git_show(evidence_commit, EVIDENCE_ARTIFACT)
    if sha256_bytes(evidence_bytes) != document["evidence_artifact_sha256"]:
        raise AcceptanceError(DIGEST, "the retained evidence artifact digest disagrees")
    schema_bytes = git_show(evidence_commit, EVIDENCE_SCHEMA)
    if sha256_bytes(schema_bytes) != document["evidence_schema_sha256"]:
        raise AcceptanceError(DIGEST, "the retained evidence schema digest disagrees")
    toolchain = document["toolchain"]
    for path_key, blob_key in (
        ("evidence_generator_path", "evidence_generator_git_blob"),
        ("evidence_validator_path", "evidence_validator_git_blob"),
        ("acceptance_generator_path", "acceptance_generator_git_blob"),
        ("acceptance_validator_path", "acceptance_validator_pre_a_git_blob"),
    ):
        if git_blob(evidence_commit, toolchain[path_key]) != toolchain[blob_key]:
            raise AcceptanceError(
                DIGEST, f"{toolchain[path_key]} blob disagrees with the retained one"
            )
    if (
        sha256_bytes(git_show(evidence_commit, ACCEPTANCE_SCHEMA))
        != toolchain["acceptance_schema_sha256"]
    ):
        raise AcceptanceError(DIGEST, "the retained acceptance schema digest disagrees")

    changed = sorted(
        run_git("diff", "--name-only", f"{evidence_commit}..{accepted}").split()
    )
    if changed != sorted([ACCEPTANCE_ARTIFACT, ACCEPTANCE_VALIDATOR]):
        raise AcceptanceError(
            DIFF_AUTHORITY,
            f"the E1..A1 diff must be exactly the two A1 paths; observed {changed}",
        )
    return certificate_line(
        {
            "schema_version": CERTIFICATE_SCHEMA,
            "stage": STAGE,
            "acceptance_commit_sha": accepted,
            "acceptance_artifact_path": ACCEPTANCE_ARTIFACT,
            "acceptance_artifact_sha256": sha256_bytes(payload),
            "evidence_commit_sha": evidence_commit,
            "evidence_artifact_path": EVIDENCE_ARTIFACT,
            "evidence_artifact_sha256": document["evidence_artifact_sha256"],
            "source_sha": document["source_sha"],
            "verdict": "ACCEPT",
            "successor_unlocks": list(UNLOCKS),
        }
    )


STATUS_PATHS = frozenset(
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


def verify_status(acceptance_commit: str, status_commit: str) -> None:
    """Check one status successor against Section 7.5's path allowlist."""
    accepted = resolve_commit(acceptance_commit)
    verify(accepted, accepted)
    if status_commit == "INDEX":
        head = run_git("rev-parse", "HEAD").strip()
        if head != accepted:
            raise AcceptanceError(
                ANCESTRY, "the INDEX sentinel requires HEAD == the acceptance commit"
            )
        changed = sorted(run_git("diff", "--name-only", "--cached").split())
    else:
        status = resolve_commit(status_commit)
        parent = run_git("rev-parse", f"{status}^").strip()
        if parent != accepted:
            raise AcceptanceError(ANCESTRY, "U1^ must be the acceptance commit")
        changed = sorted(
            run_git("diff", "--name-only", f"{accepted}..{status}").split()
        )
    if not changed:
        raise AcceptanceError(
            DIFF_AUTHORITY, "a status successor changes at least one path"
        )
    outside = [path for path in changed if path not in STATUS_PATHS]
    if outside:
        raise AcceptanceError(
            DIFF_AUTHORITY, f"paths outside the status allowlist: {outside}"
        )


def _atomic_write(target: Path, payload: bytes) -> None:
    handle, temporary = tempfile.mkstemp(dir=str(target.parent))
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _substitute(text: str, evidence_commit: str, artifact_digest: str) -> str:
    lines = text.splitlines(keepends=True)
    for name, value in (
        ("APPROVED_EVIDENCE_SHA", evidence_commit),
        ("APPROVED_ACCEPTANCE_ARTIFACT_SHA256", artifact_digest),
    ):
        marker = f"{name}: str | None = None"
        matches = [index for index, line in enumerate(lines) if line.strip() == marker]
        if len(matches) != 1:
            raise AcceptanceError(
                DIFF_AUTHORITY, f"expected exactly one `{marker}` assignment"
            )
        lines[matches[0]] = lines[matches[0]].replace(
            marker, f'{name}: str | None = "{value}"'
        )
    return "".join(lines)


def generate(review_record: Path, reject_output: Path | None) -> None:
    """Run Section 8.2's all-or-rollback acceptance generation."""
    if not review_record.is_absolute():
        raise AcceptanceError(ARGUMENT, "--review-record must be an absolute path")
    record = read_strict_json(review_record)
    if not isinstance(record, dict) or tuple(record) != REVIEW_KEYS:
        raise AcceptanceError(
            SCHEMA, f"review record keys must be exactly {list(REVIEW_KEYS)}"
        )
    status = run_git("status", "--porcelain")
    if status.strip():
        raise AcceptanceError(
            DIFF_AUTHORITY, "acceptance generation requires a globally clean tree"
        )
    evidence_commit = run_git("rev-parse", "HEAD").strip()
    evidence_file = REPOSITORY_ROOT / EVIDENCE_ARTIFACT
    if not evidence_file.is_file():
        raise AcceptanceError(DIGEST, f"{EVIDENCE_ARTIFACT} is absent at HEAD")
    evidence = read_strict_json(evidence_file)

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", EVIDENCE_VALIDATOR, "-q"],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AcceptanceError(VERDICT, "the active evidence validator did not pass")

    verdict = record["verdict"]
    document: dict[str, Any] = {
        "schema_version": "radiosim.sci005.stage1-acceptance.v1",
        "stage": STAGE,
        "verdict": verdict,
        "generated_at_utc": record["generated_at_utc"],
        "implementation_identity": record["implementation_identity"],
        "reviewer_identity": record["reviewer_identity"],
        "reviewer_independent": record["reviewer_independent"],
        "design_sha": evidence["design_sha"],
        "red_test_sha": evidence["red_test_sha"],
        "source_sha": evidence["source_sha"],
        "evidence_commit_sha": evidence_commit,
        "evidence_artifact_path": EVIDENCE_ARTIFACT,
        "evidence_artifact_sha256": sha256_bytes(evidence_file.read_bytes()),
        "evidence_schema_path": EVIDENCE_SCHEMA,
        "evidence_schema_sha256": sha256_bytes(
            (REPOSITORY_ROOT / EVIDENCE_SCHEMA).read_bytes()
        ),
        "toolchain": {
            "evidence_generator_path": EVIDENCE_GENERATOR,
            "evidence_generator_git_blob": git_blob(
                evidence_commit, EVIDENCE_GENERATOR
            ),
            "evidence_validator_path": EVIDENCE_VALIDATOR,
            "evidence_validator_git_blob": git_blob(
                evidence_commit, EVIDENCE_VALIDATOR
            ),
            "acceptance_generator_path": ACCEPTANCE_GENERATOR,
            "acceptance_generator_git_blob": git_blob(
                evidence_commit, ACCEPTANCE_GENERATOR
            ),
            "acceptance_validator_path": ACCEPTANCE_VALIDATOR,
            "acceptance_validator_pre_a_git_blob": git_blob(
                evidence_commit, ACCEPTANCE_VALIDATOR
            ),
            "acceptance_schema_path": ACCEPTANCE_SCHEMA,
            "acceptance_schema_sha256": sha256_bytes(
                (REPOSITORY_ROOT / ACCEPTANCE_SCHEMA).read_bytes()
            ),
        },
        "acceptance_commit_sha": None,
        "acceptance_commit_sha_reason": (
            SELF_REFERENCE_REASON if verdict == "ACCEPT" else REJECT_REASON
        ),
        "successor_unlocks": list(UNLOCKS) if verdict == "ACCEPT" else [],
        "reviewed_artifacts": _reviewed_artifacts(evidence),
        "rederived_oracles": record["rederived_oracles"],
        "review_checks": record["review_checks"],
        "commands": record["commands"],
        "blockers": record["blockers"],
        "accepted_limitations": record["accepted_limitations"],
        "claims_not_licensed": record["claims_not_licensed"],
    }
    validate_acceptance_document(document)
    if verdict == "REJECT":
        if reject_output is None or not reject_output.is_absolute():
            raise AcceptanceError(ARGUMENT, "--reject-output must be an absolute path")
        if reject_output.exists():
            raise AcceptanceError(ARGUMENT, "--reject-output must not already exist")
        if reject_output.resolve().is_relative_to(REPOSITORY_ROOT.resolve()):
            raise AcceptanceError(
                ARGUMENT, "--reject-output must be outside the repository"
            )
        if not record["blockers"]:
            raise AcceptanceError(VERDICT, "a REJECT requires at least one blocker")
        reject_output.write_bytes(canonical_json_bytes(document))
        return
    require_accept_completeness(document)

    target = REPOSITORY_ROOT / ACCEPTANCE_ARTIFACT
    if target.exists():
        raise AcceptanceError(DIFF_AUTHORITY, "the acceptance artifact already exists")
    payload = canonical_json_bytes(document)
    validator = REPOSITORY_ROOT / ACCEPTANCE_VALIDATOR
    original = validator.read_bytes()
    updated = _substitute(
        original.decode("utf-8"), evidence_commit, sha256_bytes(payload)
    ).encode("utf-8")
    try:
        _atomic_write(target, payload)
        _atomic_write(validator, updated)
        changed = sorted(
            line[3:] for line in run_git("status", "--porcelain").splitlines()
        )
        if changed != sorted([ACCEPTANCE_ARTIFACT, ACCEPTANCE_VALIDATOR]):
            raise AcceptanceError(
                DIFF_AUTHORITY,
                f"the working diff must be exactly the two A1 paths; observed {changed}",
            )
    except Exception:
        validator.write_bytes(original)
        target.unlink(missing_ok=True)
        raise


def _reviewed_artifacts(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive the union Section 8.2 requires, sorted by unique path."""
    paths = {EVIDENCE_ARTIFACT, EVIDENCE_SCHEMA, ACCEPTANCE_SCHEMA}
    paths.update(row["path"] for row in evidence["artifacts"])
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        candidate = REPOSITORY_ROOT / path
        if candidate.is_symlink() or not candidate.is_file():
            raise AcceptanceError(DIGEST, f"{path} is not a reachable regular file")
        rows.append(
            {
                "path": path,
                "sha256": sha256_bytes(candidate.read_bytes()),
                "source_sha": evidence["source_sha"],
                "authenticated": True,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generator = subparsers.add_parser("generate")
    generator.add_argument("--review-record", type=Path, required=True)
    generator.add_argument("--reject-output", type=Path, default=None)
    verifier = subparsers.add_parser("verify")
    verifier.add_argument("--acceptance-commit", required=True)
    verifier.add_argument("--descendant", required=True)
    status = subparsers.add_parser("verify-status")
    status.add_argument("--acceptance-commit", required=True)
    status.add_argument("--status-commit", required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "generate":
            generate(arguments.review_record, arguments.reject_output)
        elif arguments.command == "verify":
            sys.stdout.buffer.write(
                verify(arguments.acceptance_commit, arguments.descendant)
            )
            sys.stdout.buffer.flush()
        else:
            verify_status(arguments.acceptance_commit, arguments.status_commit)
    except AcceptanceError as error:
        print(f"{error.prefix}: {error.detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
