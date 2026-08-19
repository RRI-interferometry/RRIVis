#!/usr/bin/env python
"""Generate and verify the SCI-005 Stage-3 independent acceptance record.

Importing this module loads only the Python standard library, following
``tools/wp7_perf001_cpu_evidence.py``: an acceptance-critical verifier must not
depend on a package that is merely transitively present, because a lock update
could drop it and turn a hard refusal into an import error.
``docs/development/sci005_stage3_acceptance.schema.json`` stays the normative
transcription of Section 8.2, and the checks below enforce the same structure,
key order and encodings in their own code.

``docs/development/sci005_beam_physics_plan.md`` Sections 8.2, 8.3 and 9 freeze
this tool's three sub-commands.

Generation is all-or-rollback and owns the complete admissible pre-``A3`` diff::

    pixi run python tools/sci005_stage3_acceptance.py generate \\
      --review-record <absolute-temporary-review-record.json>

The generator derives every commit, path, digest, toolchain, reviewed-artifact,
self-reference and unlock field; the caller supplies a verdict and its
measurements only, and cannot override a derived field. It runs from a globally
clean exact ``E3``, first invokes the active evidence validator, and for an
``ACCEPT`` prepares both the previously absent canonical JSON and the phase
validator with exactly ``APPROVED_EVIDENCE_SHA: None -> E3`` and
``APPROVED_ACCEPTANCE_ARTIFACT_SHA256: None -> sha256(JSON)``.

The read-only verifier is the complete SCI-005 export for the WP-9 M3
dependency::

    pixi run python tools/sci005_stage3_acceptance.py verify \\
      --acceptance-commit <A3> --descendant <SHA-or-HEAD>

Beyond the Stage-1 checks it authenticates the Section 8.3 succession from Git
objects alone: ``R3^ == D3``; ``A2`` located as the unique commit introducing
the Stage-2 acceptance artifact and authenticated by the Stage-2 approved
constants; ``U2`` located as the unique commit on ``D3``'s first-parent ancestry
whose direct parent is ``A2``, satisfying the committed Stage-2
``verify-status`` form; ``A2`` an ancestor of ``D3`` through the committed
Stage-2 ``verify`` form; and every commit in ``U2..D3`` other than ``D3``
matched against the header-enumerated three-kind interval rule with its own
recorded touch set. Unlike the Stage-2 edge, the two header-recorded superseded
red slices have *different* touch sets, so each commit's recorded set is what
:func:`require_interval_kind` compares against and
:data:`STAGE3_RED_SLICE_PATHS` bounds the kind. The operative ``D3`` itself
touches only the design memo.

It emits exactly one canonical UTF-8 JSON line on success. Failure emits no
certificate on stdout, exits non-zero, and writes a stderr line beginning with
exactly one of the six frozen prefixes, one colon, one space, and the detail.

A status successor is checked before and after commit with::

    pixi run python tools/sci005_stage3_acceptance.py verify-status \\
      --acceptance-commit <A3> --status-commit <U3-or-INDEX>

Sections 8.3 and 9 additionally make this tool the *closure-parent certificate
verifier*, and add no sub-command for that job. Before the whole-row closure
successor ``C`` may be committed, both committed read-only forms above must be
run against the accepted ``A3``: ``verify`` with ``--acceptance-commit <A3>``
and ``--descendant`` naming the proposed ``C``, which must emit the canonical
certificate line with verdict ``ACCEPT`` and the unlock array
``["SCI005.U3"]`` and exit zero; and ``verify-status`` with
``--acceptance-commit <A3>`` and ``--status-commit`` naming ``U3``, which must
exit zero and silent. Those two runs reauthenticate ``U3^ == A3`` and the
``A3..U3`` diff allowlist from committed objects rather than from the working
tree. A non-zero exit or an absent certificate is a closure blocker, not a
condition to be reconciled in prose; a green workflow, an appended memo
sentence, or the ``SCI005.U3`` unlock literal alone is insufficient.
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

STAGE = 3
CERTIFICATE_SCHEMA = "radiosim.sci005.stage-acceptance-certificate.v1"
ACCEPTANCE_ARTIFACT = "docs/development/sci005_stage3_acceptance.json"
ACCEPTANCE_SCHEMA = "docs/development/sci005_stage3_acceptance.schema.json"
ACCEPTANCE_VALIDATOR = "tests/unit/test_sci005_stage3_acceptance.py"
EVIDENCE_ARTIFACT = "docs/development/sci005_stage3_evidence.json"
EVIDENCE_SCHEMA = "docs/development/sci005_stage3_evidence.schema.json"
EVIDENCE_GENERATOR = "tools/sci005_stage_evidence.py"
EVIDENCE_VALIDATOR = "tests/unit/test_sci005_evidence.py"
ACCEPTANCE_GENERATOR = "tools/sci005_stage3_acceptance.py"

#: Section 7.5's two Stage-3 approved evidence constants, named here because
#: ``verify`` authenticates the retained evidence artifact against the pins the
#: evidence validator itself carries at ``E3``.
EVIDENCE_APPROVED_SOURCE_CONSTANT = "APPROVED_STAGE3_SOURCE_SHA"
EVIDENCE_APPROVED_DIGEST_CONSTANT = "APPROVED_STAGE3_EVIDENCE_ARTIFACT_SHA256"

#: The Stage-2 retained surface the Section 8.3 starred edge is anchored on.
STAGE2_ACCEPTANCE_ARTIFACT = "docs/development/sci005_stage2_acceptance.json"
STAGE2_ACCEPTANCE_VALIDATOR = "tests/unit/test_sci005_stage2_acceptance.py"
STAGE2_ACCEPTANCE_TOOL = "tools/sci005_stage2_acceptance.py"
STAGE2_APPROVED_EVIDENCE_CONSTANT = "APPROVED_EVIDENCE_SHA"
STAGE2_APPROVED_ARTIFACT_CONSTANT = "APPROVED_ACCEPTANCE_ARTIFACT_SHA256"

DESIGN_MEMO = "docs/development/sci005_beam_physics_plan.md"

SELF_REFERENCE_REASON = "self-reference: U3 binds the containing A3 commit"
REJECT_REASON = "not-applicable: REJECT creates no A commit"
UNLOCKS = ["SCI005.U3"]

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
        "common_efield_normalization",
        "ludwig3_basis_conversion",
        "noncommuting_chain_order",
        "receptor_output_basis_factorization",
        "standard_output_roundtrip",
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

#: Section 8.3's three header-recorded interval-commit kinds.
STATUS_PROSE_KIND = "status-prose"
SUPERSEDED_DESIGN_KIND = "superseded-design"
SUPERSEDED_RED_SLICE_KIND = "superseded-red-slice"

#: The Section 7.4 test paths the header-recorded superseded *first* Stage-3 red
#: slice cut, reopened by the basis-vector and provenance correction.
FIRST_SUPERSEDED_RED_SLICE_PATHS = frozenset(
    {
        "tests/fixtures/beamfits.py",
        "tests/integration/test_sci005_beam_physics.py",
        "tests/unit/test_core/test_beam_pyuvdata_contract.py",
        "tests/unit/test_core/test_sci005_full_efield.py",
        "tests/unit/test_io/test_sci005_beam_config.py",
        "tests/unit/test_jones/test_chain_order.py",
    }
)

#: The Section 7.4 test paths the header-recorded superseded *re-cut* Stage-3
#: red slice touched, reopened by the response-identity and oracle-domain
#: correction for a second governed re-cut.
SECOND_SUPERSEDED_RED_SLICE_PATHS = frozenset(
    {
        "tests/fixtures/beamfits.py",
        "tests/unit/test_core/test_beam_pyuvdata_contract.py",
        "tests/unit/test_core/test_sci005_full_efield.py",
    }
)

#: Every Section 7.4 test path a superseded red-slice commit may touch, and the
#: only bound this kind carries. Unlike the Stage-2 edge, whose single slice
#: made the kind bound and the recorded set the same frozenset, the two Stage-3
#: slices have different touch sets: this union bounds the *kind*, while
#: :data:`INTERVAL_COMMITS` carries each commit's own exact recorded set and
#: :func:`require_interval_kind` compares against that.
STAGE3_RED_SLICE_PATHS = (
    FIRST_SUPERSEDED_RED_SLICE_PATHS | SECOND_SUPERSEDED_RED_SLICE_PATHS
)

#: Section 8.3's observed ``U2..D3`` interval, transcribed in ancestry order
#: from this memo's own header records: each commit's SHA, its kind, and the
#: exact paths its header record names. A commit in the interval that this table
#: does not name invalidates the starred edge.
INTERVAL_COMMITS: dict[str, tuple[str, frozenset[str]]] = {
    # The superseded original Stage-3 design gate: memo only.
    "2adc2acca8606b3a9774e14f28725a5687c0ecc8": (
        SUPERSEDED_DESIGN_KIND,
        frozenset({DESIGN_MEMO}),
    ),
    # The superseded first Stage-3 red slice, reopened by the basis-vector and
    # provenance correction for a governed re-cut.
    "139a8e411da1f50be29cee94ee351009437e10bc": (
        SUPERSEDED_RED_SLICE_KIND,
        FIRST_SUPERSEDED_RED_SLICE_PATHS,
    ),
    # The superseded basis-vector and provenance correction: memo only.
    "9956e77477b0597129e71b38a183c8dcd3cb761e": (
        SUPERSEDED_DESIGN_KIND,
        frozenset({DESIGN_MEMO}),
    ),
    # The superseded re-cut Stage-3 red slice, reopened by the
    # response-identity and oracle-domain correction for a second re-cut.
    "ea06bc649ae9987253c8002150e21b03a842cb45": (
        SUPERSEDED_RED_SLICE_KIND,
        SECOND_SUPERSEDED_RED_SLICE_PATHS,
    ),
}


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


def parent_of(commit: str) -> str:
    """Return one commit's *direct parent*, never the commit itself.

    ``<rev>^{commit}`` is git's **peel** form: on a commit object it resolves
    back to that same commit. Section 8.1 records exactly this confusion as the
    evidence generator's Stage-2 defect, so every direct-parent question in this
    tool goes through this one function rather than an inline expression.
    """
    parent = run_git("rev-parse", f"{commit}^").strip()
    if GIT_SHA.fullmatch(parent) is None or parent == commit:
        raise AcceptanceError(ANCESTRY, f"{commit} has no distinct direct parent")
    return parent


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
    if document["schema_version"] != "radiosim.sci005.stage3-acceptance.v1":
        raise AcceptanceError(SCHEMA, "schema_version is not the Stage-3 literal")
    if document["stage"] != 3 or isinstance(document["stage"], bool):
        raise AcceptanceError(SCHEMA, "stage must be the integer 3")
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
            SCHEMA, "acceptance_commit_sha is JSON null; U3 binds the containing A3"
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


# --- the Section 8.3 ``U2 ->* D3`` succession ---------------------------------


def read_approved_constant(source: str, name: str, origin: str) -> str:
    """Read one ``NAME: str | None = "<hex>"`` approved constant literal.

    This is the same line-oriented read the Section 7.5 substitution writes and
    the evidence generator already uses for the ``D1`` binding: exactly one
    assignment, a quoted lower-case hexadecimal literal, and nothing else.
    """
    matches = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(f"{name}:")
    ]
    if len(matches) != 1:
        raise AcceptanceError(
            DIGEST, f"expected exactly one {name} assignment in {origin}"
        )
    value = matches[0].split("=", 1)[1].strip().strip('"').strip("'")
    if GIT_SHA.fullmatch(value) is None and SHA256.fullmatch(value) is None:
        raise AcceptanceError(
            DIGEST, f"{name} in {origin} is not an approved hexadecimal literal"
        )
    return value


def locate_stage2_acceptance_commit(descendant: str) -> str:
    """Return ``A2``: the unique commit introducing the Stage-2 artifact.

    Section 8.3 requires ``A2`` to be *located*, not assumed, and then
    authenticated by the Stage-2 approved constants rather than by name.
    """
    introductions = run_git(
        "log",
        "--diff-filter=A",
        "--format=%H",
        descendant,
        "--",
        STAGE2_ACCEPTANCE_ARTIFACT,
    ).split()
    if len(introductions) != 1:
        raise AcceptanceError(
            ANCESTRY,
            f"{STAGE2_ACCEPTANCE_ARTIFACT} must be introduced exactly once on the "
            f"ancestry of {descendant}; observed {introductions}",
        )
    accepted = introductions[0]

    validator = git_show(accepted, STAGE2_ACCEPTANCE_VALIDATOR).decode("utf-8")
    approved_evidence = read_approved_constant(
        validator, STAGE2_APPROVED_EVIDENCE_CONSTANT, STAGE2_ACCEPTANCE_VALIDATOR
    )
    approved_artifact = read_approved_constant(
        validator, STAGE2_APPROVED_ARTIFACT_CONSTANT, STAGE2_ACCEPTANCE_VALIDATOR
    )
    payload = git_show(accepted, STAGE2_ACCEPTANCE_ARTIFACT)
    if sha256_bytes(payload) != approved_artifact:
        raise AcceptanceError(
            DIGEST,
            "the located A2 artifact does not match the Stage-2 approved digest",
        )
    parent = parent_of(accepted)
    if parent != approved_evidence:
        raise AcceptanceError(
            ANCESTRY,
            f"A2^ is {parent}, not the Stage-2 approved evidence commit "
            f"{approved_evidence}",
        )
    document = parse_strict_json(
        payload.decode("utf-8"), STAGE2_ACCEPTANCE_ARTIFACT, SCHEMA
    )
    if document.get("evidence_commit_sha") != approved_evidence:
        raise AcceptanceError(
            DIGEST, "the located A2 artifact names a foreign evidence commit"
        )
    if document.get("verdict") != "ACCEPT":
        raise AcceptanceError(VERDICT, "the located A2 artifact is not an ACCEPT")
    return accepted


def run_stage2_tool(*arguments: str) -> None:
    """Run the committed Stage-2 acceptance tool and require a zero exit."""
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / STAGE2_ACCEPTANCE_TOOL), *arguments],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AcceptanceError(
            ANCESTRY,
            f"the committed Stage-2 {arguments[0]} form failed: "
            f"{completed.stderr.strip()}",
        )


def locate_status_successor(design_commit: str, stage2_acceptance: str) -> str:
    """Return ``U2``: the unique first-parent child of ``A2`` under ``D3``."""
    listing = run_git("rev-list", "--first-parent", "--parents", design_commit)
    candidates: list[str] = []
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == stage2_acceptance:
            candidates.append(fields[0])
    if len(candidates) != 1:
        raise AcceptanceError(
            ANCESTRY,
            "exactly one commit on D3's first-parent ancestry must directly "
            f"parent A2; observed {candidates}",
        )
    return candidates[0]


def touched_paths(commit: str) -> frozenset[str]:
    return frozenset(
        run_git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).split()
    )


def require_single_parent(commit: str) -> str:
    parents = run_git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    if len(parents) != 1:
        raise AcceptanceError(
            ANCESTRY, f"{commit} is a merge or a root; the interval admits neither"
        )
    return parents[0]


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


def require_interval_kind(commit: str, kind: str, recorded: frozenset[str]) -> None:
    """Check one interval commit's touch set against its header-recorded kind.

    The recorded set is the per-commit authority: the two Stage-3 red slices cut
    different path sets, so a shared kind-wide frozenset could not express
    either one. :data:`STAGE3_RED_SLICE_PATHS` still bounds the kind, so a slice
    that reached outside Section 7.4's test paths is refused even if its own
    header record were to name that path.
    """
    require_single_parent(commit)
    touched = touched_paths(commit)
    if not touched:
        raise AcceptanceError(DIFF_AUTHORITY, f"{commit} changes no path")
    if touched != recorded:
        raise AcceptanceError(
            DIFF_AUTHORITY,
            f"{commit} touches {sorted(touched)}, not the header-recorded "
            f"{sorted(recorded)}",
        )
    if kind == STATUS_PROSE_KIND:
        outside = sorted(touched - STATUS_PATHS)
        if outside:
            raise AcceptanceError(
                DIFF_AUTHORITY,
                f"status-prose commit {commit} touches non-status paths {outside}",
            )
    elif kind == SUPERSEDED_DESIGN_KIND:
        if DESIGN_MEMO not in touched:
            raise AcceptanceError(
                DIFF_AUTHORITY,
                f"superseded-design commit {commit} does not touch the design memo",
            )
    elif kind == SUPERSEDED_RED_SLICE_KIND:
        outside = sorted(touched - STAGE3_RED_SLICE_PATHS)
        if outside:
            raise AcceptanceError(
                DIFF_AUTHORITY,
                f"superseded red-slice commit {commit} touches {outside}, which is "
                "not a Section 7.4 test path",
            )
    else:  # pragma: no cover - the table only carries the three frozen kinds
        raise AcceptanceError(ANCESTRY, f"{kind!r} is not a header-recorded kind")


def authenticate_design_edge(design_sha: str) -> None:
    """Authenticate the Section 8.3 starred edge ``U2 ->* D3``.

    Every fact here comes from Git objects. ``A2`` is located by artifact
    introduction and authenticated by the Stage-2 approved constants; ``U2`` is
    the unique first-parent child of ``A2`` under ``D3`` and must satisfy the
    committed Stage-2 ``verify-status`` form; ``A2`` must be an ancestor of
    ``D3`` through the committed Stage-2 ``verify`` form; and every commit in
    ``U2..D3`` other than ``D3`` must be named by the header-enumerated interval
    table and touch exactly the paths its own record names. Because no
    admissible kind may touch the retained Stage-1 or Stage-2 evidence and
    acceptance surface, that surface is byte-identical to ``U2`` by
    construction.
    """
    stage2_acceptance = locate_stage2_acceptance_commit(design_sha)
    if not is_ancestor(stage2_acceptance, design_sha):
        raise AcceptanceError(
            ANCESTRY, f"A2 {stage2_acceptance} is not an ancestor of D3 {design_sha}"
        )
    run_stage2_tool(
        "verify", "--acceptance-commit", stage2_acceptance, "--descendant", design_sha
    )
    status_successor = locate_status_successor(design_sha, stage2_acceptance)
    run_stage2_tool(
        "verify-status",
        "--acceptance-commit",
        stage2_acceptance,
        "--status-commit",
        status_successor,
    )

    interval = [
        commit
        for commit in run_git("rev-list", f"{status_successor}..{design_sha}").split()
        if commit != design_sha
    ]
    if sorted(interval) != sorted(INTERVAL_COMMITS):
        raise AcceptanceError(
            ANCESTRY,
            "the observed U2..D3 interval is not the header-enumerated one; "
            f"observed {sorted(interval)}",
        )
    for commit in interval:
        kind, recorded = INTERVAL_COMMITS[commit]
        require_interval_kind(commit, kind, recorded)

    require_single_parent(design_sha)
    design_touched = touched_paths(design_sha)
    if design_touched != frozenset({DESIGN_MEMO}):
        raise AcceptanceError(
            DIFF_AUTHORITY,
            f"the operative D3 must touch only {DESIGN_MEMO}; observed "
            f"{sorted(design_touched)}",
        )


def authenticate_succession(design_sha: str, red_test_sha: str) -> None:
    """Authenticate ``R3^ == D3`` and then the starred ``U2 ->* D3`` edge.

    Section 8.3: ``D3`` is the unambiguous direct parent of ``R3``. The
    evidence generator's recorded defect resolved ``Di`` as git's peel form of
    ``HEAD^`` and would have written ``design_sha == red_test_sha``; this is
    where that class of record fails acceptance.
    """
    if design_sha == red_test_sha:
        raise AcceptanceError(
            ANCESTRY, "D3 and R3 are the same commit; Section 8.3 requires R3^ == D3"
        )
    observed = parent_of(red_test_sha)
    if observed != design_sha:
        raise AcceptanceError(
            ANCESTRY, f"R3^ is {observed}, not the recorded design commit {design_sha}"
        )
    authenticate_design_edge(design_sha)


def verify(acceptance_commit: str, descendant: str) -> bytes:
    """Read-only verification returning Section 9's canonical certificate line.

    Sections 8.3 and 9 also make this the closure-parent form: run with the
    accepted ``A3`` and the proposed ``C`` as descendant, its zero exit and
    emitted certificate are what authorize ``C``'s direct parent.
    """
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
    parent = parent_of(accepted)
    if parent != evidence_commit:
        raise AcceptanceError(
            ANCESTRY,
            f"A3^ is {parent}, not the bound evidence commit {evidence_commit}",
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

    evidence_validator = git_show(evidence_commit, EVIDENCE_VALIDATOR).decode("utf-8")
    approved_source = read_approved_constant(
        evidence_validator, EVIDENCE_APPROVED_SOURCE_CONSTANT, EVIDENCE_VALIDATOR
    )
    approved_digest = read_approved_constant(
        evidence_validator, EVIDENCE_APPROVED_DIGEST_CONSTANT, EVIDENCE_VALIDATOR
    )
    if approved_source != document["source_sha"]:
        raise AcceptanceError(
            DIGEST, "the Stage-3 approved source constant disagrees with the record"
        )
    if approved_digest != document["evidence_artifact_sha256"]:
        raise AcceptanceError(
            DIGEST, "the Stage-3 approved evidence digest disagrees with the record"
        )
    evidence = parse_strict_json(
        evidence_bytes.decode("utf-8"), EVIDENCE_ARTIFACT, SCHEMA
    )
    for key in ("design_sha", "red_test_sha", "source_sha"):
        if evidence.get(key) != document[key]:
            raise AcceptanceError(
                DIGEST, f"the acceptance record's {key} disagrees with the evidence"
            )
    authenticate_succession(evidence["design_sha"], evidence["red_test_sha"])

    changed = sorted(
        run_git("diff", "--name-only", f"{evidence_commit}..{accepted}").split()
    )
    if changed != sorted([ACCEPTANCE_ARTIFACT, ACCEPTANCE_VALIDATOR]):
        raise AcceptanceError(
            DIFF_AUTHORITY,
            f"the E3..A3 diff must be exactly the two A3 paths; observed {changed}",
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


def verify_status(acceptance_commit: str, status_commit: str) -> None:
    """Check one status successor against Section 7.5's path allowlist.

    Sections 8.3 and 9 also make this the second closure-parent form: run with
    the accepted ``A3`` and ``U3``, its silent zero exit reauthenticates
    ``U3^ == A3`` and the ``A3..U3`` allowlist from committed objects.
    """
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
        parent = parent_of(status)
        if parent != accepted:
            raise AcceptanceError(ANCESTRY, "U3^ must be the acceptance commit")
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
        "schema_version": "radiosim.sci005.stage3-acceptance.v1",
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
    authenticate_succession(evidence["design_sha"], evidence["red_test_sha"])

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
                f"the working diff must be exactly the two A3 paths; observed {changed}",
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
