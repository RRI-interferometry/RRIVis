#!/usr/bin/env python
"""Generate and verify the SCI-004 phase-M1 independent acceptance record.

``docs/development/sci004_mmode_design.md`` Sections 13.3, 14.3 and 14.4 freeze
this tool's authority.  It is tracked at ``S1`` beside the evidence generator,
so a clean exact ``S1`` already contains the bytes that will produce and
validate its successors, and ``A1`` adds only the acceptance JSON and the two
approved constants in the strict validator.

Importing this module loads only the Python standard library, following
``tools/sci005_stage1_acceptance.py``: an acceptance-critical verifier must not
depend on a package that is merely transitively present, because a lock update
could drop it and turn a hard refusal into an import error.

Sub-commands::

    pixi run python tools/sci004_mmode_phase1_acceptance.py preflight
    pixi run python tools/sci004_mmode_phase1_acceptance.py generate \\
      --review-record <absolute-temporary-review-record.json>
    pixi run python tools/sci004_mmode_phase1_acceptance.py check \\
      --artifact <path>

The generator derives every commit, path, digest, reviewed-artifact and
self-reference field; the reviewer supplies a verdict, the re-derived oracles
and any blockers, and cannot override a derived field.  It runs from a globally
clean exact ``E1``, first invokes the active evidence validator, and writes the
previously absent canonical JSON.  Section 14.4 names that venue rather than
forbidding production there: a ``generate`` that refuses after a passing
preflight would make the record unproducible.

``ACCEPT`` requires an independent reviewer, no false oracle, an empty
``blockers`` array, exact ``S -> E`` ancestry, an authenticated phase evidence
artifact, and no production-source path in the ``E..A`` diff.  ``REJECT``
requires at least one concrete blocker and does not unlock the next phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

PHASE = "M1"
ACCEPTANCE_SCHEMA = "radiosim.sci004.mmode-phase1-acceptance.v1"
ACCEPTANCE_ARTIFACT = "docs/development/sci004_mmode_phase1_acceptance.json"
ACCEPTANCE_VALIDATOR = "tests/unit/test_sci004_phase1_acceptance.py"
EVIDENCE_ARTIFACT = "docs/development/sci004_mmode_phase1_evidence.json"
EVIDENCE_VALIDATOR = "tests/unit/test_sci004_phase1_evidence.py"
EVIDENCE_GENERATOR = "tools/sci004_mmode_phase1_evidence.py"

#: Section 14.3's declared output set for A1.  Exactly one file.
DECLARED_OUTPUTS: tuple[str, ...] = (ACCEPTANCE_ARTIFACT,)

#: Frozen stderr prefixes, mirroring the SCI-005 acceptance verifier.
ARGUMENT = "SCI004_ACCEPTANCE_ARGUMENT"
SCHEMA = "SCI004_ACCEPTANCE_SCHEMA"
ANCESTRY = "SCI004_ACCEPTANCE_ANCESTRY"
DIGEST = "SCI004_ACCEPTANCE_DIGEST"
DIFF_AUTHORITY = "SCI004_ACCEPTANCE_DIFF_AUTHORITY"
VERDICT = "SCI004_ACCEPTANCE_VERDICT"

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

REVIEWED_ARTIFACT_KEYS: tuple[str, ...] = (
    "path",
    "sha256",
    "source_sha",
    "authenticated",
)
REDERIVED_ORACLE_KEYS: tuple[str, ...] = (
    "oracle_id",
    "method",
    "observed",
    "fixed_limit",
    "pass",
)
BLOCKER_KEYS: tuple[str, ...] = (
    "blocker_id",
    "requirement_id",
    "evidence",
    "required_remediation",
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

#: Section 14.3's exact self-reference reason.
ACCEPTANCE_SELF_REFERENCE_REASON = (
    "self-reference: the next R or C binds the containing A commit"
)

#: Section 14.3's required ``A1`` re-derivation identifiers.  These are required
#: ``rederived_oracles`` identifiers, not optional reviewer prose.
REQUIRED_ORACLES: tuple[str, ...] = (
    "m1.canonical-era-grid",
    "m1.polar-motion-unit-path",
    "m1.rigid-era-attitude",
    "m1.public-astropy-tangent-jacobian",
    "m1.analytic-horizon-roots",
    "m1.dft-sign",
    "m1.bounded-driver-frame-certificate",
    "m1.direct-rime-byte-identity",
    "m1.complete-frozen-direct-gate",
    "m1.scalar-capability-rows",
    "m1.wp7-dependency-gate",
)


class AcceptanceError(RuntimeError):
    """One refusal, carrying the frozen stderr prefix it must be reported with."""

    def __init__(self, prefix: str, detail: str) -> None:
        self.prefix = prefix
        self.detail = detail
        super().__init__(f"{prefix}: {detail}")


def ecmascript_number(value: float) -> str:
    """Render a finite binary64 with ECMAScript ``Number::toString`` spelling."""
    if not math.isfinite(value):
        raise AcceptanceError(SCHEMA, "canonical JSON forbids NaN and Infinity")
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
    """Render one JSON value with Section 14's exact serialization."""
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
    raise AcceptanceError(SCHEMA, f"cannot canonicalize {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Return Section 14's ``J(x)`` bytes for a JSON-primitive tree."""
    return _render(value).encode("utf-8")


def _require(condition: bool, prefix: str, detail: str) -> None:
    if not condition:
        raise AcceptanceError(prefix, detail)


def _require_keys(value: Any, keys: tuple[str, ...], label: str) -> dict[str, Any]:
    """Require an object to carry exactly one key set, rejecting any deviation.

    Section 14's canonical serialization sorts object keys lexicographically, so
    a re-read record never preserves an author's insertion order: "exactly these
    keys" is a statement about the *set*.  Both a missing and an unknown key are
    named in the refusal.
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


def _require_sorted_unique_strings(value: Any, label: str) -> None:
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


def validate_acceptance_document(document: Any) -> dict[str, Any]:
    """Validate the complete Section 14.3 M1 acceptance record."""
    record = _require_keys(document, ACCEPTANCE_KEYS, "acceptance document")
    _require(
        record["schema_version"] == ACCEPTANCE_SCHEMA,
        SCHEMA,
        "schema_version is the frozen phase literal",
    )
    _require(record["phase"] == PHASE, SCHEMA, "phase must be exactly 'M1'")
    _require(
        record["verdict"] in {"ACCEPT", "REJECT"},
        VERDICT,
        "verdict must be ACCEPT or REJECT",
    )
    _require(
        isinstance(record["reviewer_identity"], str) and record["reviewer_identity"],
        SCHEMA,
        "reviewer_identity is a non-empty role/task identifier",
    )
    _require(
        record["reviewer_independent"] is True,
        VERDICT,
        "reviewer_independent must be true",
    )
    for field in ("design_sha", "red_commit_sha", "source_sha", "evidence_commit_sha"):
        _require_hex(record[field], 40, field)
    _require(
        record["evidence_artifact_path"] == EVIDENCE_ARTIFACT,
        SCHEMA,
        "evidence_artifact_path is the fixed E1 path",
    )
    _require_hex(record["evidence_artifact_sha256"], 64, "evidence_artifact_sha256")
    _require(
        record["acceptance_commit_sha"] is None,
        SCHEMA,
        "acceptance_commit_sha is JSON null at A",
    )
    _require(
        record["acceptance_commit_sha_reason"] == ACCEPTANCE_SELF_REFERENCE_REASON,
        SCHEMA,
        "acceptance_commit_sha_reason is the exact self-reference literal",
    )

    reviewed = record["reviewed_artifacts"]
    _require(
        isinstance(reviewed, list) and reviewed,
        SCHEMA,
        "reviewed_artifacts must be a non-empty array",
    )
    for index, row in enumerate(reviewed):
        mapping = _require_keys(
            row, REVIEWED_ARTIFACT_KEYS, f"reviewed_artifacts[{index}]"
        )
        _require_hex(mapping["sha256"], 64, f"reviewed_artifacts[{index}].sha256")
        _require_hex(
            mapping["source_sha"], 40, f"reviewed_artifacts[{index}].source_sha"
        )
        _require(
            mapping["authenticated"] is True,
            VERDICT,
            f"reviewed_artifacts[{index}] must be authenticated",
        )

    oracles = record["rederived_oracles"]
    _require(isinstance(oracles, list), SCHEMA, "rederived_oracles must be an array")
    observed_ids: list[str] = []
    for index, row in enumerate(oracles):
        mapping = _require_keys(
            row, REDERIVED_ORACLE_KEYS, f"rederived_oracles[{index}]"
        )
        _require(
            isinstance(mapping["method"], str) and mapping["method"],
            SCHEMA,
            f"rederived_oracles[{index}].method names the oracle's units",
        )
        for field in ("observed", "fixed_limit"):
            value = mapping[field]
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)),
                SCHEMA,
                f"rederived_oracles[{index}].{field} must be a finite number",
            )
        observed_ids.append(str(mapping["oracle_id"]))

    blockers = record["blockers"]
    _require(isinstance(blockers, list), SCHEMA, "blockers must be an array")
    for index, row in enumerate(blockers):
        _require_keys(row, BLOCKER_KEYS, f"blockers[{index}]")

    commands = record["commands"]
    _require(isinstance(commands, list), SCHEMA, "commands must be an array")
    for index, row in enumerate(commands):
        mapping = _require_keys(row, COMMAND_KEYS, f"commands[{index}]")
        _require(
            mapping["exit_code"] == 0 and not isinstance(mapping["exit_code"], bool),
            SCHEMA,
            f"commands[{index}].exit_code must be the integer zero",
        )
        _require(mapping["cwd"] == ".", SCHEMA, f"commands[{index}].cwd must be '.'")

    _require_sorted_unique_strings(
        record["accepted_limitations"], "accepted_limitations"
    )
    _require_sorted_unique_strings(record["claims_not_licensed"], "claims_not_licensed")

    if record["verdict"] == "ACCEPT":
        _require(
            not blockers,
            VERDICT,
            "ACCEPT requires an empty blockers array",
        )
        _require(
            all(row["pass"] is True for row in oracles),
            VERDICT,
            "ACCEPT requires no false oracle",
        )
        missing = [name for name in REQUIRED_ORACLES if name not in observed_ids]
        _require(
            not missing,
            VERDICT,
            f"ACCEPT requires the A1 re-derivations {missing}",
        )
    else:
        _require(
            len(blockers) >= 1,
            VERDICT,
            "REJECT requires at least one concrete blocker",
        )
    return record


#: Section 14.3's exact reviewer-supplied key set.  Everything else in the
#: record is derived, and a review record that carries a derived field is
#: rejected rather than silently overridden.
REVIEW_RECORD_KEYS: tuple[str, ...] = (
    "reviewer_identity",
    "reviewer_independent",
    "verdict",
    "rederived_oracles",
    "blockers",
    "accepted_limitations",
    "claims_not_licensed",
)

#: The derived fields a reviewer may never supply.
DERIVED_FIELDS: tuple[str, ...] = (
    "acceptance_commit_sha",
    "acceptance_commit_sha_reason",
    "design_sha",
    "evidence_artifact_path",
    "evidence_artifact_sha256",
    "evidence_commit_sha",
    "generated_at_utc",
    "phase",
    "red_commit_sha",
    "reviewed_artifacts",
    "schema_version",
    "source_sha",
)


def load_review_record(path: Path) -> dict[str, Any]:
    """Load and validate the reviewer's own contribution to the record.

    The reviewer supplies exactly a verdict, an identity, an independence
    declaration, the re-derived oracles, any blockers, and the two claim
    arrays.  Every other field is derived here from the repository, so a review
    record that names one is refused instead of being allowed to overwrite the
    derivation.
    """
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise AcceptanceError(
            ARGUMENT, f"the review record {path} could not be read: {error}"
        ) from error
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcceptanceError(
            ARGUMENT,
            f"the review record {path} is not UTF-8 JSON: {error}",
        ) from error
    record = _require_keys(document, REVIEW_RECORD_KEYS, "review record")
    for field in DERIVED_FIELDS:
        _require(
            field not in document,
            ARGUMENT,
            f"the review record may not supply the derived field {field!r}",
        )
    _require(
        isinstance(record["reviewer_identity"], str) and record["reviewer_identity"],
        ARGUMENT,
        "review record reviewer_identity is a non-empty role/task identifier",
    )
    _require(
        record["reviewer_independent"] is True,
        VERDICT,
        "an acceptance record requires an independent reviewer",
    )
    _require(
        record["verdict"] in {"ACCEPT", "REJECT"},
        VERDICT,
        "review record verdict must be ACCEPT or REJECT",
    )
    for field in ("rederived_oracles", "blockers"):
        _require(
            isinstance(record[field], list),
            ARGUMENT,
            f"review record {field} must be an array",
        )
    return record


def _evidence_source_sha(evidence: dict[str, Any]) -> str:
    """Return the ``S1`` the phase evidence artifact binds."""
    return _require_hex(evidence["source_sha"], 40, "evidence source_sha")


def _require_exact_ancestry(source_sha: str, evidence_commit_sha: str) -> None:
    """Require ``E1``'s direct parent to be exactly ``S1`` (Section 14.4)."""
    parents = _git("rev-list", "--parents", "-n", "1", evidence_commit_sha).split()
    _require(
        len(parents) == 2 and parents[1] == source_sha,
        ANCESTRY,
        f"the direct parent of {evidence_commit_sha} must be exactly {source_sha}",
    )


def _reviewed_artifacts(source_sha: str, evidence_sha256: str) -> list[dict[str, Any]]:
    """Return the authenticated artifact set the reviewer read.

    Every path here is read from the working tree and hashed now, so the record
    cannot claim to have reviewed bytes that are not the ones present at ``E1``.
    """
    rows = [
        {
            "path": EVIDENCE_ARTIFACT,
            "sha256": evidence_sha256,
            "source_sha": source_sha,
            "authenticated": True,
        }
    ]
    for relative in (
        EVIDENCE_VALIDATOR,
        EVIDENCE_GENERATOR,
        ACCEPTANCE_VALIDATOR,
        "docs/development/sci004_mmode_phase1_red_failures.json",
        "docs/development/sci004_mmode_phase1_wp7_dependency.json",
    ):
        path = REPOSITORY_ROOT / relative
        _require(
            path.is_file(),
            DIGEST,
            f"the reviewed artifact {relative} is absent at E1",
        )
        rows.append(
            {
                "path": relative,
                "sha256": raw_sha256(path),
                "source_sha": source_sha,
                "authenticated": True,
            }
        )
    return sorted(rows, key=lambda row: str(row["path"]))


def _run_evidence_validator() -> dict[str, Any]:
    """Run the active evidence validator and record its exact command row.

    Section 14.3 requires the acceptance generator to invoke the *active*
    validator rather than to restate its verdict, so the command row carries the
    real exit code and stream digests.
    """
    started = datetime.now(UTC)
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / EVIDENCE_GENERATOR),
            "check",
            "--artifact",
            str(REPOSITORY_ROOT / EVIDENCE_ARTIFACT),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    duration = (datetime.now(UTC) - started).total_seconds()
    _require(
        completed.returncode == 0,
        DIGEST,
        "the active evidence validator rejected the artifact: "
        + completed.stderr.decode("utf-8", "replace").strip(),
    )
    return {
        "argv": [
            "pixi",
            "run",
            "python",
            EVIDENCE_GENERATOR,
            "check",
            "--artifact",
            EVIDENCE_ARTIFACT,
        ],
        "cwd": ".",
        "pixi_environment": "default",
        "started_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_seconds": duration,
        "exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
    }


def build_acceptance_document(
    state: dict[str, str], review: dict[str, Any]
) -> dict[str, Any]:
    """Build the complete Section 14.3 M1 acceptance record.

    The reviewer's verdict, oracles, blockers and claim arrays are carried
    through unchanged; every commit, path and digest is derived here.
    """
    evidence_commit_sha = state["evidence_commit_sha"]
    evidence = json.loads(
        (REPOSITORY_ROOT / EVIDENCE_ARTIFACT).read_bytes().decode("utf-8")
    )
    source_sha = _evidence_source_sha(evidence)
    _require_exact_ancestry(source_sha, evidence_commit_sha)
    command = _run_evidence_validator()
    return {
        "schema_version": ACCEPTANCE_SCHEMA,
        "phase": PHASE,
        "verdict": review["verdict"],
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reviewer_identity": review["reviewer_identity"],
        "reviewer_independent": True,
        "design_sha": _require_hex(evidence["design_sha"], 40, "design_sha"),
        "red_commit_sha": _require_hex(
            evidence["red_commit_sha"], 40, "red_commit_sha"
        ),
        "source_sha": source_sha,
        "evidence_commit_sha": evidence_commit_sha,
        "evidence_artifact_path": EVIDENCE_ARTIFACT,
        "evidence_artifact_sha256": state["evidence_artifact_sha256"],
        "acceptance_commit_sha": None,
        "acceptance_commit_sha_reason": ACCEPTANCE_SELF_REFERENCE_REASON,
        "reviewed_artifacts": _reviewed_artifacts(
            source_sha, state["evidence_artifact_sha256"]
        ),
        "rederived_oracles": list(review["rederived_oracles"]),
        "commands": [command],
        "blockers": list(review["blockers"]),
        "accepted_limitations": sorted(set(review["accepted_limitations"])),
        "claims_not_licensed": sorted(set(review["claims_not_licensed"])),
    }


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AcceptanceError(
            ANCESTRY, f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def raw_sha256(path: Path) -> str:
    """Return the SHA-256 of a file's exact raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight() -> dict[str, str]:
    """Run Section 14.3's pre-output check without writing anything."""
    head = _git("rev-parse", "HEAD").strip()
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status.strip():
        raise AcceptanceError(ANCESTRY, "the working tree is not globally clean")
    evidence = REPOSITORY_ROOT / EVIDENCE_ARTIFACT
    if not evidence.is_file():
        raise AcceptanceError(
            DIGEST,
            "the acceptance generator runs only from a globally clean exact E1, "
            "which is the commit that adds the phase evidence artifact",
        )
    for relative in DECLARED_OUTPUTS:
        if (REPOSITORY_ROOT / relative).exists():
            raise AcceptanceError(
                DIGEST, f"the declared output {relative} already exists"
            )
    return {
        "evidence_commit_sha": head,
        "evidence_artifact_sha256": raw_sha256(evidence),
    }


def require_declared_outputs_only() -> None:
    """Require the repository's only new paths to equal the declared output set.

    Section 14.3 runs this *after* publication: the acceptance record is the one
    expected new path, and a generator that also left a stray file behind has
    not produced the declared set.
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
    """Publish one artifact atomically, refusing to overwrite anything."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "xb") as handle:
        handle.write(payload)
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise AcceptanceError(DIGEST, f"{path} already exists") from error
    finally:
        temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    """Run one sub-command and return its process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    generate = sub.add_parser("generate")
    generate.add_argument("--review-record", required=True)
    check = sub.add_parser("check")
    check.add_argument("--artifact", required=True)
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "preflight":
            state = preflight()
            sys.stdout.write(canonical_json(state).decode("utf-8") + "\n")
            return 0
        if arguments.command == "generate":
            state = preflight()
            review = load_review_record(Path(arguments.review_record))
            document = build_acceptance_document(state, review)
            validate_acceptance_document(document)
            payload = canonical_json(document)
            write_atomic_no_overwrite(REPOSITORY_ROOT / ACCEPTANCE_ARTIFACT, payload)
            require_declared_outputs_only()
            sys.stdout.write(
                canonical_json(
                    {
                        "artifact": ACCEPTANCE_ARTIFACT,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "verdict": document["verdict"],
                    }
                ).decode("utf-8")
                + "\n"
            )
            return 0
        document = json.loads(Path(arguments.artifact).read_bytes().decode("utf-8"))
        validate_acceptance_document(document)
        return 0
    except AcceptanceError as error:
        sys.stderr.write(f"{error.prefix}: {error.detail}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
