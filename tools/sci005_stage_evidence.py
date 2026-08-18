#!/usr/bin/env python
"""Generate one SCI-005 stage evidence artifact.

Importing this module loads only the Python standard library. That follows
``tools/wp7_perf001_cpu_evidence.py`` deliberately: an acceptance-critical
generator must not depend on a package that is merely transitively present,
because a lock update could drop it and turn a hard refusal into an import
error. ``docs/development/sci005_stage{i}_evidence.schema.json`` stays the
normative transcription of Section 8.1; the structural self-check below
enforces the same key sets, order and encodings in its own code before any
repository byte is written, and ``tests/unit/test_sci005_evidence.py`` is the
independent validator of record.

``docs/development/sci005_beam_physics_plan.md`` Section 8.1 freezes the
retained evidence contract and its generation transaction. The exact
invocation is::

    pixi run python tools/sci005_stage_evidence.py generate \\
      --stage <1|2|3> --measurement-record <absolute-temporary-record.json>

The generator *derives* every identity, provenance and digest field and refuses
a caller override of any of them: ``schema_version``, ``stage``, ``status``,
``design_sha``, ``red_test_sha``, ``source_sha``, ``evidence_sha``,
``working_tree_clean``, the runtime/platform/Pixi fields, ``pixi_lock_sha256``,
and ``artifacts``. The caller supplies measurements only.

The transaction is all-or-rollback. From globally clean ``HEAD == Si`` it
prepares, in memory, the previously absent evidence JSON and
``tests/unit/test_sci005_evidence.py`` with exactly the target stage's two
``None`` sentinels replaced, writes both through same-directory temporary
files, restores every original byte and removes every new target on any
failure, and then requires the working diff to be exactly the two Section 7.5
``Ei`` paths. Success is silent. Manual artifact copying or pinning is
forbidden: this transaction owns the complete admissible pre-``Ei`` diff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

_GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

EVIDENCE_VALIDATOR = "tests/unit/test_sci005_evidence.py"
DEPENDENCY_CERTIFICATE = "docs/development/sci005_stage1_wp7_dependency.json"

#: Section 8.1's common measurement-record keys, in order.
COMMON_MEASUREMENT_KEYS: tuple[str, ...] = (
    "generated_at_utc",
    "scientific_conventions",
    "config_cases",
    "analytic_invariants",
    "rejection_probes",
    "backend_parity",
    "solver_cases",
    "output_cases",
    "fingerprint_diff",
    "commands",
    "artifact_inputs",
    "limitations",
    "claims_not_licensed",
)

#: Stage 1 appends exactly these three arrays, in this order.
STAGE1_MEASUREMENT_KEYS: tuple[str, ...] = (
    "pupil_profiles",
    "support_masks",
    "ruze_power_diagnostics",
)

#: Section 8.1's common evidence field sequence, in order.
COMMON_EVIDENCE_KEYS: tuple[str, ...] = (
    "schema_version",
    "stage",
    "status",
    "generated_at_utc",
    "design_sha",
    "red_test_sha",
    "source_sha",
    "evidence_sha",
    "working_tree_clean",
    "radiosim_version",
    "python_version",
    "platform",
    "machine",
    "pixi_environment",
    "pixi_lock_sha256",
    "scientific_conventions",
    "config_cases",
    "analytic_invariants",
    "rejection_probes",
    "backend_parity",
    "solver_cases",
    "output_cases",
    "fingerprint_diff",
    "commands",
    "artifacts",
    "limitations",
    "claims_not_licensed",
)


class EvidenceError(RuntimeError):
    """One evidence-generation precondition or transaction failure."""


def evidence_artifact_path(stage: int) -> str:
    return f"docs/development/sci005_stage{stage}_evidence.json"


def evidence_schema_path(stage: int) -> str:
    return f"docs/development/sci005_stage{stage}_evidence.schema.json"


def approved_constant_names(stage: int) -> tuple[str, str]:
    return (
        f"APPROVED_STAGE{stage}_SOURCE_SHA",
        f"APPROVED_STAGE{stage}_EVIDENCE_ARTIFACT_SHA256",
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_path(value: str) -> str:
    """Validate one repository-relative POSIX ``canonical_path``.

    Section 8.1: non-empty, no leading slash, backslash, NUL, empty component,
    ``.`` component or ``..`` component, byte-for-byte equal to its normalized
    form, and resolving inside the repository without traversing a symlink.
    """
    if not value or value != value.strip():
        raise EvidenceError(f"path {value!r} is empty or not normalized")
    if value.startswith("/") or "\\" in value or "\x00" in value:
        raise EvidenceError(f"path {value!r} is not a repository-relative path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise EvidenceError(f"path {value!r} has an empty or relative component")
    if os.path.normpath(value) != value:
        raise EvidenceError(f"path {value!r} is not byte-equal to its normal form")
    resolved = (REPOSITORY_ROOT / value).resolve()
    if not resolved.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise EvidenceError(f"path {value!r} escapes the repository root")
    return value


def run_git(*arguments: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=str(cwd or REPOSITORY_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise EvidenceError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def require_globally_clean() -> str:
    """Return ``HEAD`` after requiring a globally clean working tree."""
    status = run_git("status", "--porcelain")
    if status.strip():
        raise EvidenceError(
            "evidence generation requires a globally clean working tree; "
            f"observed:\n{status}"
        )
    return run_git("rev-parse", "HEAD").strip()


def read_strict_json(path: Path) -> Any:
    """Read one strict UTF-8 JSON object, rejecting duplicates and non-finite."""
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"{path} is not a regular file")
    text = path.read_text(encoding="utf-8")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                raise EvidenceError(f"duplicate JSON key {key!r} in {path}")
            seen[key] = value
        return seen

    def reject_non_finite(_value: str) -> float:
        raise EvidenceError(f"non-finite JSON number in {path}")

    return json.loads(
        text,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_non_finite,
    )


def canonical_json_bytes(document: Any) -> bytes:
    """Serialize with Section 8.1's exact output conventions."""
    text = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )
    return (text + "\n").encode("utf-8")


def derive_runtime_fields() -> dict[str, Any]:
    from radiosim import __version__ as radiosim_version

    lock = REPOSITORY_ROOT / "pixi.lock"
    return {
        "radiosim_version": str(radiosim_version),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pixi_environment": os.environ.get("PIXI_ENVIRONMENT_NAME", "default"),
        "pixi_lock_sha256": sha256_bytes(lock.read_bytes()),
    }


def resolve_design_sha(stage: int) -> str:
    """Resolve ``Di`` for the stage.

    Section 8.1: Stage 1 resolves ``D1`` **only** from the immutable dependency
    validator's binding constant, never by choosing a matching commit from
    history; Stages 2 and 3 resolve ``Di`` as the direct parent of ``Ri``.
    """
    if stage != 1:
        return run_git("rev-parse", "HEAD^^{commit}").strip()
    binding = REPOSITORY_ROOT / "tests" / "unit" / "test_sci005_stage1_dependency.py"
    if not binding.is_file():
        raise EvidenceError(f"{binding} is absent; the R1 design binding is required")
    for line in binding.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("APPROVED_SCI005_D1_SHA"):
            value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
                raise EvidenceError(
                    f"APPROVED_SCI005_D1_SHA is not a git_sha: {value!r}"
                )
            return value
    raise EvidenceError("APPROVED_SCI005_D1_SHA is absent from the R1 binding module")


def resolve_red_test_sha() -> str:
    """Return ``Ri``: the direct parent of the clean checked-out ``Si``."""
    red_test_sha = run_git("rev-parse", "HEAD^^{commit}").strip()
    source_sha = run_git("rev-parse", "HEAD^{commit}").strip()
    if red_test_sha == source_sha:
        raise EvidenceError("Ri resolved equal to Si; Section 8.3 requires Si^ == Ri")
    return red_test_sha


def build_artifacts(rows: list[dict[str, Any]], stage: int) -> list[dict[str, Any]]:
    """Derive the sorted ``artifacts`` array and its raw digests.

    The caller supplies target paths and roles only; Section 8.1 forbids a
    caller-supplied digest, so every ``sha256`` here is read from the file.
    """
    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if set(row) != {"path", "input_kind", "input_path", "media_type", "role"}:
            raise EvidenceError(
                f"artifact_inputs row has unexpected keys: {sorted(row)}"
            )
        target = canonical_path(str(row["path"]))
        if target in seen:
            raise EvidenceError(f"duplicate artifact target path {target!r}")
        seen.add(target)
        if row["input_kind"] != "repository":
            raise EvidenceError(
                f"input_kind {row['input_kind']!r} is not legal at stage {stage}"
            )
        if row["input_path"] != target:
            raise EvidenceError(
                "a repository artifact input must name the same canonical path"
            )
        source = REPOSITORY_ROOT / target
        if source.is_symlink() or not source.is_file():
            raise EvidenceError(f"{target} is not a regular file at this commit")
        artifacts.append(
            {
                "path": target,
                "sha256": sha256_bytes(source.read_bytes()),
                "media_type": str(row["media_type"]),
                "role": str(row["role"]),
            }
        )
    artifacts.sort(key=lambda item: item["path"])
    return artifacts


def substitute_sentinels(
    text: str,
    stage: int,
    source_sha: str,
    artifact_sha256: str,
) -> str:
    """Replace exactly the target stage's two ``None`` sentinels."""
    source_name, digest_name = approved_constant_names(stage)
    replacements = ((source_name, source_sha), (digest_name, artifact_sha256))
    lines = text.splitlines(keepends=True)
    for name, value in replacements:
        prefix = f"{name}: str | None = None"
        matches = [index for index, line in enumerate(lines) if line.strip() == prefix]
        if len(matches) != 1:
            raise EvidenceError(
                f"expected exactly one `{prefix}` assignment; found {len(matches)}"
            )
        index = matches[0]
        lines[index] = lines[index].replace(
            f"{name}: str | None = None", f'{name}: str | None = "{value}"'
        )
    return "".join(lines)


def generate(stage: int, measurement_record: Path) -> None:
    """Run Section 8.1's complete all-or-rollback evidence transaction."""
    if stage not in {1, 2, 3}:
        raise EvidenceError("--stage must be 1, 2, or 3")
    if not measurement_record.is_absolute():
        raise EvidenceError("--measurement-record must be an absolute path")
    source_sha = require_globally_clean()
    record = read_strict_json(measurement_record)
    if not isinstance(record, dict):
        raise EvidenceError("the measurement record must be a JSON object")
    expected = COMMON_MEASUREMENT_KEYS + (STAGE1_MEASUREMENT_KEYS if stage == 1 else ())
    if tuple(record) != expected:
        raise EvidenceError(
            "measurement record keys must be exactly "
            f"{list(expected)}; observed {list(record)}"
        )

    artifact_path = canonical_path(evidence_artifact_path(stage))
    target = REPOSITORY_ROOT / artifact_path
    if target.exists():
        raise EvidenceError(f"{artifact_path} already exists; Ei adds it exactly once")
    schema_file = REPOSITORY_ROOT / evidence_schema_path(stage)
    if not schema_file.is_file():
        raise EvidenceError(f"{evidence_schema_path(stage)} is absent")

    document: dict[str, Any] = {
        "schema_version": f"radiosim.sci005.stage{stage}.v1",
        "stage": stage,
        "status": "candidate",
        "generated_at_utc": record["generated_at_utc"],
        "design_sha": resolve_design_sha(stage),
        "red_test_sha": resolve_red_test_sha(),
        "source_sha": source_sha,
        "evidence_sha": None,
        "working_tree_clean": True,
        **derive_runtime_fields(),
        "scientific_conventions": record["scientific_conventions"],
    }
    for key in COMMON_EVIDENCE_KEYS:
        if key in document:
            continue
        if key == "artifacts":
            document[key] = build_artifacts(list(record["artifact_inputs"]), stage)
        else:
            document[key] = record[key]
    if stage == 1:
        for key in STAGE1_MEASUREMENT_KEYS:
            document[key] = record[key]

    for row in document["commands"]:
        if row.get("exit_code") != 0:
            raise EvidenceError(
                "candidate evidence requires every exit code to be zero"
            )
    _require_no_false_rows(document, stage)

    payload = canonical_json_bytes(document)
    _self_check_document(document, schema_file, stage)

    validator = REPOSITORY_ROOT / EVIDENCE_VALIDATOR
    original = validator.read_bytes()
    updated = substitute_sentinels(
        original.decode("utf-8"), stage, source_sha, sha256_bytes(payload)
    ).encode("utf-8")
    try:
        _atomic_write(target, payload)
        _atomic_write(validator, updated)
        diff = sorted(
            line[3:] for line in run_git("status", "--porcelain").splitlines()
        )
        if diff != sorted([artifact_path, EVIDENCE_VALIDATOR]):
            raise EvidenceError(
                f"the working diff must be exactly the two Ei paths; observed {diff}"
            )
    except Exception:
        validator.write_bytes(original)
        target.unlink(missing_ok=True)
        raise


def _require_no_false_rows(document: dict[str, Any], stage: int) -> None:
    for key, rows in document.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("passed") is False:
                raise EvidenceError(
                    f"{key} contains a false row; candidate evidence "
                    "records only passing measurements"
                )
    if stage == 1:
        for row in document["ruze_power_diagnostics"]:
            kinds = [item["kind"] for item in row["limit_oracles"]]
            if len(set(kinds)) != 6:
                raise EvidenceError(
                    "every ruze_power_diagnostics row needs all six limit-oracle kinds"
                )


def _self_check_document(
    document: dict[str, Any], schema_file: Path, stage: int
) -> None:
    """Refuse to write a document that does not match its own transcription.

    The generator built this document, so its self-check is structural: exact
    top-level key set and order against the normative schema, the frozen
    literals, and the digest/commit encodings. The independent enforcement of
    every Section 8.1 cross-field rule belongs to
    ``tests/unit/test_sci005_evidence.py``, which is not this file.
    """
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    declared = tuple(schema["properties"])
    if tuple(document) != declared:
        raise EvidenceError(
            "the generated document's key order does not match its schema "
            f"transcription; expected {list(declared)}, observed {list(document)}"
        )
    if set(schema["required"]) != set(declared):
        raise EvidenceError("the schema transcription does not require every key")
    if document["schema_version"] != f"radiosim.sci005.stage{stage}.v1":
        raise EvidenceError("schema_version does not name this stage")
    if document["status"] != "candidate" or document["evidence_sha"] is not None:
        raise EvidenceError("candidate evidence carries a null evidence_sha")
    if document["working_tree_clean"] is not True:
        raise EvidenceError("candidate evidence requires a clean working tree")
    for key in ("design_sha", "red_test_sha", "source_sha"):
        value = document[key]
        if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
            raise EvidenceError(f"{key} is not a 40-character lower-case git_sha")
    if _SHA256.fullmatch(document["pixi_lock_sha256"]) is None:
        raise EvidenceError("pixi_lock_sha256 is not a 64-character lower-case digest")
    conventions = schema["properties"]["scientific_conventions"]["properties"]
    for key, constraint in conventions.items():
        if document["scientific_conventions"].get(key) != constraint["const"]:
            raise EvidenceError(
                f"scientific_conventions.{key} must be {constraint['const']!r}"
            )
    if set(document["scientific_conventions"]) != set(conventions):
        raise EvidenceError("scientific_conventions carries an unexpected key")


def _atomic_write(target: Path, payload: bytes) -> None:
    handle, temporary = tempfile.mkstemp(dir=str(target.parent))
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generator = subparsers.add_parser("generate")
    generator.add_argument("--stage", type=int, required=True)
    generator.add_argument("--measurement-record", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        generate(arguments.stage, arguments.measurement_record)
    except EvidenceError as error:
        print(f"SCI005_EVIDENCE: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
