"""Strict ``R1`` dependency validator and the in-tree ``D1`` acceptance record.

``docs/development/sci005_beam_physics_plan.md`` Section 7.1 makes the SCI-005
Stage-1 red slice conditional on two authenticated facts, and this module is the
validator that holds both.

**The ``D1`` acceptance record.** The operative ``D1`` is the *second* of two
independently accepted design commits, and the lineage matters because Section
7.1 binds exactly one of them.

First, the acceptance-succession governance amendment that opens the governing
memo landed at ``58e7fb3d09dbcaec6f8201a778a653b55996c1aa``. Its own header
required *fresh independent governance and computational review of its exact
bytes*; both reviews were performed on 2026-08-14 against that commit and both
returned ``ACCEPT``, binding its reviewed parent-relative diff
``sha256:f987ceb7061ea08f43be092d22342a8ef1c752cababd3ddbff500a21e21dcf40`` and
its memo blob
``sha256:1a7843b892c3b1f975aab663618d7e3afe2b6e06511587ce7e695568bb3ed387``.

Second, red-test authoring against that accepted design surfaced one genuine
Section 3.2 ambiguity -- the leg-wider-than-resolved-diameter rejection named no
owner reachable under Section 2's taxonomy, because per-antenna diameters
resolve only with the instrument. The bounded ownership correction that rules
that owner landed at ``1222f69a5ea0a6e008e38b7bfd4ba2d7f6c168d5``. Its exact
pre-landing file bytes
``sha256:2fb7c2f328f343fb6054a183e35aa58a7795f6a5ab6efe3c2ea29660a3aae001`` and
pre-landing parent-relative diff
``sha256:3d772d7be5d290bd68e64211c9cb5ffd065ec56e1ca7a430810de26172b3e347``
received separate fresh independent governance and computational ``ACCEPT``
verdicts on 2026-08-14, each reconfirmed after a single-sentence Section 3.5
precision fix; both digests are recorded inside the landed header itself. It
landed with only that record sentence added -- mirroring the
``8935052cc4e49e3ff7bb92f645d03cee6b9e8ad2`` precedent -- so the landed blob is
:data:`D1_MEMO_BLOB_SHA256` rather than the pre-landing value, and the landed
parent-relative diff over the memo path is :data:`D1_REVIEWED_DIFF_SHA256`.
Both are authenticated below, and the landed blob is still the byte-identical
content of the memo at ``HEAD``.

That correction commit supersedes the amendment commit as "the independently
accepted commit containing this amendment" in Section 7.1's sense, and is what
confers ``D1``. This docstring is the in-tree record of that acceptance;
:data:`APPROVED_SCI005_D1_SHA` below is the single design binding Section 7.1
authorises, and every Stage-1 ``design_sha`` derives from it rather than from a
search of history.

**The WP-7 dependency gate.** Section 7.1 also requires an intervening clean
tip ``G1`` with both ``D1`` and the accepted PERF-001 CPU acceptance commit as
inclusive ancestors, and requires ``R1`` to retain the exact upstream
certificate line at ``docs/development/sci005_stage1_wp7_dependency.json``.
Here ``G1 == D1``, with the accepted WP-7 ``A``
(``7e5f469c835c1137a3a3a870d27c5d9f5e8f3520``) as its ancestor. A live checkout
cannot rerun the upstream interface directly, because that verifier requires a
clean ``HEAD == --descendant``; the replay below therefore attaches a detached
worktree at exact ``G1``, executes ``tools/wp7_perf001_cpu_evidence.py`` *from
that tree*, and compares its stdout byte-for-byte with the retained line. That
byte comparison is what authenticates the retained certificate, so no digest of
it is pinned here: a certificate re-captured at a new ``G1`` is re-authenticated
without editing a dependency-validator byte. Inability to create, authenticate,
execute, or clean up that worktree is a hard failure and never mutates the
caller's checkout.

These tests pass at ``R1``: they authenticate an already accepted chain rather
than any Stage-1 physics.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

#: The independently accepted SCI-005 design gate. Section 7.1: "The same red
#: commit creates ``tests/unit/test_sci005_stage1_dependency.py`` with exactly
#: one design binding assignment."  No later stage may change this literal.
APPROVED_SCI005_D1_SHA = "1222f69a5ea0a6e008e38b7bfd4ba2d7f6c168d5"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DESIGN_MEMO_PATH = "docs/development/sci005_beam_physics_plan.md"
CERTIFICATE_PATH = "docs/development/sci005_stage1_wp7_dependency.json"
CPU_EVIDENCE_TOOL_PATH = "tools/wp7_perf001_cpu_evidence.py"

#: The landed ``D1`` memo blob: the accepted pre-landing bytes plus the single
#: record sentence the header itself documents.
D1_MEMO_BLOB_SHA256 = "20b4cda18f4cadc1f6ba3d9e096a8831e3351c4e78816be2af6297bff8901b5a"
#: The landed ``D1`` parent-relative diff over the memo path.
D1_REVIEWED_DIFF_SHA256 = (
    "32199b2e505eac0dba90c1ae412992914e70bfdab8d48de02838bc4b4e0d87ee"
)
#: The superseded acceptance-succession amendment, recorded for lineage only.
SUPERSEDED_AMENDMENT_SHA = "58e7fb3d09dbcaec6f8201a778a653b55996c1aa"
SUPERSEDED_AMENDMENT_BLOB_SHA256 = (
    "1a7843b892c3b1f975aab663618d7e3afe2b6e06511587ce7e695568bb3ed387"
)
SUPERSEDED_AMENDMENT_DIFF_SHA256 = (
    "f987ceb7061ea08f43be092d22342a8ef1c752cababd3ddbff500a21e21dcf40"
)

#: Section 7.1's exact certificate field list, in the order the memo prints it.
CERTIFICATE_FIELDS: tuple[str, ...] = (
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
CERTIFICATE_SCHEMA = "radiosim.perf001.cpu_acceptance_certificate.v1"
CERTIFICATE_VERDICT = "CPU_ACCEPTED_P_E_HARDWARE_GATED"

_COMMIT_FIELDS: tuple[str, ...] = (
    "acceptance_commit",
    "evidence_commit",
    "generating_source_sha",
    "descendant_commit",
)
_DIGEST_FIELDS: tuple[str, ...] = (
    "artifact_sha256",
    "cpu_evidence_tool_sha256",
    "production_record_validator_sha256",
    "production_harness_sha256",
    "pixi_manifest_sha256",
    "pixi_lock_sha256",
)
_DIFF_PATH_FIELDS: tuple[str, ...] = (
    "evidence_diff_paths",
    "acceptance_diff_paths",
)

#: Hermetic Git configuration: the pinned amendment digest must not depend on
#: the invoking user's pager, colour, prefix, or external-diff settings.
_HERMETIC_GIT: tuple[str, ...] = (
    "git",
    "-c",
    "core.pager=cat",
    "-c",
    "color.ui=false",
    "-c",
    "diff.noprefix=false",
    "-c",
    "diff.mnemonicPrefix=false",
    "-c",
    "diff.external=",
)


class DependencyCertificateError(AssertionError):
    """The retained WP-7 dependency certificate failed strict validation."""


def _git(*arguments: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT if cwd is None else cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise DependencyCertificateError(
            f"git {' '.join(arguments)} failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    return completed.stdout


def _git_blob(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise DependencyCertificateError(
            f"cannot read {relative!r} at {commit}: "
            f"{completed.stderr.decode('utf-8', 'replace').strip()}"
        )
    return completed.stdout


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    """Return the inclusive ancestry answer Section 7.1 requires."""
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise DependencyCertificateError(
            "git merge-base --is-ancestor failed for "
            f"{ancestor}..{descendant}: {completed.stderr.decode('utf-8', 'replace')}"
        )
    return completed.returncode == 0


def _is_lower_hex(value: object, *, width: int) -> bool:
    return (
        type(value) is str
        and len(value) == width
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise DependencyCertificateError(f"duplicate certificate key {key!r}")
        parsed[key] = value
    return parsed


def parse_dependency_certificate(raw: bytes) -> Mapping[str, Any]:
    """Parse the retained certificate line under Section 7.1's strict rules."""
    if type(raw) is not bytes:
        raise DependencyCertificateError("certificate bytes must be exact bytes")
    if not raw.endswith(b"\n") or raw[:-1].endswith(b"\n"):
        raise DependencyCertificateError(
            "certificate must end with exactly one final LF"
        )
    if b"\n" in raw[:-1] or b"\r" in raw:
        raise DependencyCertificateError("certificate must be exactly one line")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:  # pragma: no cover - defensive
        raise DependencyCertificateError("certificate must be UTF-8") from error
    try:
        parsed = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise DependencyCertificateError(f"certificate is not JSON: {error}") from error
    if type(parsed) is not dict:
        raise DependencyCertificateError("certificate must be a JSON object")
    if tuple(parsed) != tuple(sorted(CERTIFICATE_FIELDS)):
        raise DependencyCertificateError(
            "certificate must carry exactly the Section 7.1 fields in sorted "
            f"order; observed {tuple(parsed)!r}"
        )
    canonical = json.dumps(parsed, sort_keys=True, ensure_ascii=False) + "\n"
    if canonical.encode("utf-8") != raw:
        raise DependencyCertificateError("certificate is not canonical sorted JSON")
    if parsed["schema_version"] != CERTIFICATE_SCHEMA:
        raise DependencyCertificateError(
            f"certificate schema must be {CERTIFICATE_SCHEMA!r}"
        )
    if parsed["verdict"] != CERTIFICATE_VERDICT:
        raise DependencyCertificateError(
            f"certificate verdict must be {CERTIFICATE_VERDICT!r}"
        )
    if parsed["passed"] is not True:
        raise DependencyCertificateError("certificate must record passed: true")
    for field in _COMMIT_FIELDS:
        if not _is_lower_hex(parsed[field], width=40):
            raise DependencyCertificateError(f"{field} must be 40 lower-case hex")
    for field in _DIGEST_FIELDS:
        if not _is_lower_hex(parsed[field], width=64):
            raise DependencyCertificateError(f"{field} must be 64 lower-case hex")
    for field in _DIFF_PATH_FIELDS:
        if type(parsed[field]) is not list:
            raise DependencyCertificateError(f"{field} must be a JSON array")
    for field in ("artifact_path", *_DIFF_PATH_FIELDS):
        entry = parsed[field]
        values = [entry] if field == "artifact_path" else list(entry)
        for value in values:
            if type(value) is not str or not value:
                raise DependencyCertificateError(f"{field} must hold non-empty paths")
            if value != str(Path(value)) or value.startswith("/") or ".." in value:
                raise DependencyCertificateError(
                    f"{field} must hold normalized repository-relative paths"
                )
    for field in _DIFF_PATH_FIELDS:
        entries = list(parsed[field])
        if entries != sorted(set(entries)):
            raise DependencyCertificateError(f"{field} must be sorted and unique")
    return parsed


def read_retained_certificate() -> tuple[bytes, Mapping[str, Any]]:
    """Read and strictly parse the retained ``R1`` certificate bytes."""
    path = REPOSITORY_ROOT / CERTIFICATE_PATH
    if not path.is_file() or path.is_symlink():
        raise DependencyCertificateError(
            f"{CERTIFICATE_PATH} must be a retained regular file"
        )
    raw = path.read_bytes()
    return raw, parse_dependency_certificate(raw)


def replay_dependency_certificate(
    certificate: Mapping[str, Any],
) -> tuple[bytes, float]:
    """Replay the Section 7.1 upstream command from a detached ``G1`` worktree.

    Returns the replayed stdout bytes and the elapsed wall-clock seconds. The
    worktree and its temporary directory are removed on success and on failure,
    and the caller's checkout is never mutated.
    """
    descendant = str(certificate["descendant_commit"])
    acceptance = str(certificate["acceptance_commit"])
    temporary = Path(tempfile.mkdtemp(prefix="sci005-g1-"))
    worktree = temporary / "g1"
    started = time.monotonic()
    try:
        _git("worktree", "add", "--detach", str(worktree), descendant)
        resolved = _git("rev-parse", "HEAD", cwd=worktree).strip()
        if resolved != descendant:
            raise DependencyCertificateError(
                f"detached worktree resolved to {resolved!r}, not {descendant!r}"
            )
        status = _git("status", "--porcelain", cwd=worktree)
        if status.strip():
            raise DependencyCertificateError(
                f"detached G1 worktree is dirty:\n{status}"
            )
        tool = worktree / CPU_EVIDENCE_TOOL_PATH
        if not tool.is_file():
            raise DependencyCertificateError(
                f"{CPU_EVIDENCE_TOOL_PATH} is absent from the G1 tree"
            )
        tool_digest = hashlib.sha256(tool.read_bytes()).hexdigest()
        if tool_digest != certificate["cpu_evidence_tool_sha256"]:
            raise DependencyCertificateError(
                "executed tool digest "
                f"{tool_digest!r} does not equal the certificate's "
                f"cpu_evidence_tool_sha256 {certificate['cpu_evidence_tool_sha256']!r}"
            )
        completed = subprocess.run(
            [
                sys.executable,
                str(tool),
                "verify-accepted",
                "--acceptance-commit",
                acceptance,
                "--descendant",
                descendant,
            ],
            cwd=worktree,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise DependencyCertificateError(
                "upstream verify-accepted exited "
                f"{completed.returncode}: "
                f"{completed.stderr.decode('utf-8', 'replace').strip()}"
            )
        if completed.stderr != b"":
            raise DependencyCertificateError(
                "upstream verify-accepted wrote to stderr: "
                f"{completed.stderr.decode('utf-8', 'replace').strip()}"
            )
        elapsed = time.monotonic() - started
        return completed.stdout, elapsed
    finally:
        removal_errors: list[str] = []
        if worktree.exists():
            try:
                _git("worktree", "remove", "--force", str(worktree))
            except DependencyCertificateError as error:
                removal_errors.append(str(error))
                shutil.rmtree(worktree, ignore_errors=True)
        try:
            _git("worktree", "prune")
        except DependencyCertificateError as error:  # pragma: no cover - defensive
            removal_errors.append(str(error))
        shutil.rmtree(temporary, ignore_errors=True)
        if temporary.exists() or removal_errors:
            raise DependencyCertificateError(
                "replay cleanup failed: "
                + "; ".join(removal_errors or [f"{temporary} still exists"])
            )


@pytest.fixture(scope="module")
def certificate() -> Mapping[str, Any]:
    _raw, parsed = read_retained_certificate()
    return parsed


def test_module_carries_exactly_one_design_binding_assignment() -> None:
    """Section 7.1 authorises exactly one ``D1`` binding in this file."""
    source = Path(__file__).read_bytes().decode("utf-8")
    tree = ast.parse(source)
    bindings = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "APPROVED_SCI005_D1_SHA"
            for target in node.targets
        )
    ]

    assert len(bindings) == 1
    value = bindings[0].value
    assert isinstance(value, ast.Constant)
    assert value.value == APPROVED_SCI005_D1_SHA
    assert _is_lower_hex(APPROVED_SCI005_D1_SHA, width=40)


def _memo_diff_digest(commit: str) -> str:
    """Return the hermetic parent-relative memo diff digest for ``commit``."""
    completed = subprocess.run(
        [
            *_HERMETIC_GIT,
            "diff",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--find-renames",
            "-U3",
            f"{commit}^",
            commit,
            "--",
            DESIGN_MEMO_PATH,
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    return hashlib.sha256(completed.stdout).hexdigest()


def test_approved_d1_memo_blob_authenticates_against_the_pinned_digest() -> None:
    """The landed correction bytes are exactly what the reviews authenticated."""
    blob = _git_blob(APPROVED_SCI005_D1_SHA, DESIGN_MEMO_PATH)

    assert hashlib.sha256(blob).hexdigest() == D1_MEMO_BLOB_SHA256
    # Section 8.3: "Across ``D1..G1``, the exact ``D1`` memo blob ... remain
    # unchanged", so the checked-out memo is still the accepted design.
    checked_out = (REPOSITORY_ROOT / DESIGN_MEMO_PATH).read_bytes()
    assert hashlib.sha256(checked_out).hexdigest() == D1_MEMO_BLOB_SHA256


def test_approved_d1_amendment_diff_authenticates_against_the_pinned_digest() -> None:
    """The reviewed parent-relative correction diff is reproducible byte-for-byte."""
    assert _memo_diff_digest(APPROVED_SCI005_D1_SHA) == D1_REVIEWED_DIFF_SHA256


def test_the_superseded_amendment_remains_an_authenticated_ancestor() -> None:
    """The ownership correction supersedes the amendment; it does not erase it.

    The amendment's own two ``ACCEPT`` verdicts are what let the correction be a
    *bounded* correction rather than a fresh design, so its exact bytes stay
    reachable and authenticated even though Section 7.1 now binds the successor.
    """
    assert SUPERSEDED_AMENDMENT_SHA != APPROVED_SCI005_D1_SHA
    assert _is_ancestor(SUPERSEDED_AMENDMENT_SHA, APPROVED_SCI005_D1_SHA)
    blob = _git_blob(SUPERSEDED_AMENDMENT_SHA, DESIGN_MEMO_PATH)
    assert hashlib.sha256(blob).hexdigest() == SUPERSEDED_AMENDMENT_BLOB_SHA256
    assert (
        _memo_diff_digest(SUPERSEDED_AMENDMENT_SHA) == SUPERSEDED_AMENDMENT_DIFF_SHA256
    )
    # The correction is a design-only successor: it touches the memo and nothing
    # else, which is the whole of Section 7.1's design-only authority.
    changed = _git(
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        APPROVED_SCI005_D1_SHA,
    ).split()
    assert changed == [DESIGN_MEMO_PATH]


def test_approved_d1_is_a_single_parent_non_merge_commit() -> None:
    """Section 8.3: no named commit in the succession is a merge."""
    parents = _git("rev-list", "--parents", "-n", "1", APPROVED_SCI005_D1_SHA).split()

    assert parents[0] == APPROVED_SCI005_D1_SHA
    assert len(parents) == 2


def test_retained_certificate_parses_strictly_with_exactly_sixteen_fields(
    certificate: Mapping[str, Any],
) -> None:
    """Section 7.1 freezes the certificate's schema, field set, verdict, and flag."""
    assert set(certificate) == set(CERTIFICATE_FIELDS)
    assert len(CERTIFICATE_FIELDS) == 16
    assert certificate["schema_version"] == CERTIFICATE_SCHEMA
    assert certificate["verdict"] == CERTIFICATE_VERDICT
    assert certificate["passed"] is True
    assert certificate["artifact_path"] == (
        "output/benchmarks/reference/perf001/20260813T185841Z-darwin-arm64.json"
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("extra_field", "sorted order"),
        ("missing_field", "sorted order"),
        ("reordered_keys", "sorted order"),
        ("compact_separators", "canonical"),
        ("duplicate_key", "duplicate"),
        ("wrong_schema", "schema"),
        ("wrong_verdict", "verdict"),
        ("passed_false", "passed"),
        ("no_final_newline", "final LF"),
        ("two_lines", "exactly one line"),
        ("trailing_text", "not JSON"),
        ("short_digest", "64 lower-case hex"),
        ("upper_case_commit", "40 lower-case hex"),
        ("absolute_diff_path", "repository-relative"),
        ("unsorted_diff_paths", "sorted and unique"),
    ],
)
def test_strict_parser_rejects_every_certificate_mutation(
    mutation: str,
    expected: str,
) -> None:
    """Section 7.1: an upstream interface mismatch is a hard failure."""
    raw, parsed = read_retained_certificate()
    document = dict(parsed)
    if mutation == "extra_field":
        document["extra"] = "value"
    elif mutation == "missing_field":
        del document["evidence_commit"]
    elif mutation == "wrong_schema":
        document["schema_version"] = "radiosim.perf001.other.v1"
    elif mutation == "wrong_verdict":
        document["verdict"] = "CPU_ACCEPTED"
    elif mutation == "passed_false":
        document["passed"] = False
    elif mutation == "short_digest":
        document["artifact_sha256"] = "abc"
    elif mutation == "upper_case_commit":
        document["descendant_commit"] = str(document["descendant_commit"]).upper()
    elif mutation == "absolute_diff_path":
        document["evidence_diff_paths"] = ["/etc/passwd"]
    elif mutation == "unsorted_diff_paths":
        document["evidence_diff_paths"] = list(
            reversed(list(document["evidence_diff_paths"]))
        )

    if mutation == "reordered_keys":
        mutated = (
            json.dumps(
                dict(reversed(list(document.items()))),
                sort_keys=False,
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
    elif mutation == "compact_separators":
        mutated = (
            json.dumps(
                document,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    elif mutation == "duplicate_key":
        mutated = raw[:-2] + b', "verdict": "OTHER"}\n'
    elif mutation == "no_final_newline":
        mutated = raw[:-1]
    elif mutation == "two_lines":
        mutated = raw + raw
    elif mutation == "trailing_text":
        mutated = raw[:-1] + b" trailing\n"
    else:
        mutated = (
            json.dumps(document, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")

    with pytest.raises(DependencyCertificateError, match=expected):
        parse_dependency_certificate(mutated)


def test_design_gate_and_wp7_acceptance_are_ancestors_of_the_gate_tip(
    certificate: Mapping[str, Any],
) -> None:
    """Section 7.1: both ancestry tests are inclusive, so ``G1`` may equal ``A``."""
    descendant = str(certificate["descendant_commit"])
    acceptance = str(certificate["acceptance_commit"])

    assert _is_ancestor(APPROVED_SCI005_D1_SHA, descendant)
    assert _is_ancestor(acceptance, descendant)
    assert _is_ancestor(str(certificate["evidence_commit"]), acceptance)
    assert _is_ancestor(str(certificate["generating_source_sha"]), acceptance)


def test_gate_tip_remains_an_ancestor_of_the_current_head(
    certificate: Mapping[str, Any],
) -> None:
    """Phase-aware: ``G1`` is ``HEAD`` at ``R1`` authoring and an ancestor after."""
    head = _git("rev-parse", "HEAD").strip()

    assert _is_ancestor(str(certificate["descendant_commit"]), head)


def test_detached_worktree_replay_reproduces_the_retained_certificate(
    certificate: Mapping[str, Any],
) -> None:
    """Section 7.1's replay, including its mandatory cleanup discipline.

    The upstream verifier requires a clean ``HEAD == --descendant``, so the
    validator attaches a detached worktree at exact ``G1``, executes the tool
    *from that tree* after authenticating its raw digest, and compares stdout
    byte-for-byte with the retained line.
    """
    before = _git("status", "--porcelain")

    stdout, elapsed = replay_dependency_certificate(certificate)

    raw, _parsed = read_retained_certificate()
    assert stdout == raw
    assert parse_dependency_certificate(stdout) == certificate
    assert elapsed >= 0.0
    # The replay never mutates the caller's checkout.
    assert _git("status", "--porcelain") == before
    assert not any(
        line.split()[0] == "worktree" and "sci005-g1-" in line
        for line in _git("worktree", "list", "--porcelain").splitlines()
        if line.strip()
    )
