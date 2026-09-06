"""Authenticate SCI-004 D30 history and its exact D31 design succession.

Historical review recovery is portable: D30 pins the complete retained record,
whose original archive contributions were independently authenticated before
landing. CI verifies those retained bytes and their Git joins without requiring
the author's private session directory.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DESIGN_SHA = "d3ddb10ae01ab450f5337d06c9588ce8144cf1e5"
OPERATIVE_DESIGN_SHA = "f2e5edbcc97450262482672bb322cf926622b208"
DESIGN_SUCCESSOR_PARENT = "87b16ba16c8a4ab4ff8b9e6bf213c5ce45a41bfe"
DESIGN_SUCCESSOR_BLOBS = (
    "4f5e61dfc03a983d0806c656e8785c656f84bc17f2c52e1fa1151639dcd16f33",
    "0a92f7c96298a1b76aa11c1f77c63d62d204024a7b50781d6dbf1d69a16d566c",
)
DESIGN_SUCCESSOR_DIFF = (
    "555d08016aab5cc29106e0a1b9bf1389a580f1300d1d9efb4e4a83e1714182e7"
)
DESIGN_SUCCESSOR_REVIEW_PINS = (
    "7fc98597d564c1e2201b365392691468270e9bbcfecfb1c083d43ce4d006dc92",
    "68a6f2bc58d954423b9294ee3c7fb0bb5a6c4e8f6302043d90e97c43b33482ae",
    "f9d2257ef156cec2b7872eec45dd55c0dd3b69555bf4cdec3e8c00b00e818de9",
)
SOLVER_PATH = "src/radiosim/core/mmode/solver.py"
SNAPSHOT_FIXTURE_PATH = "tests/unit/test_io/test_standard_visibility.py"
PREREQUISITE_TIP_SHA = "cfad247831629241842ffecd5f7aaa5b2084493c"
REVIEW_RECOVERY_PATH = "docs/development/sci004_review_recovery.json"
REVIEW_RECOVERY_SHA256 = (
    "eb9b00fcdb7703cb40982bc7e445ba6e042fb45ca26bd0515387dfb644975d54"
)
RECOVERED_DESIGNS = frozenset(
    {
        "67da2b818b89511df8476b7010230c65d6cb6a75",
        "cfc9b10d655a4d9bedbd7d7750c4743f504bbaf9",
    }
)
DESIGN_PATHS = (
    "PostTier8RemediationPlan.md",
    "docs/development/sci004_mmode_design.md",
)


class HistoryError(ValueError):
    """A retained record does not authenticate its declared history."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HistoryError(message)


def _git(root: Path, *arguments: str) -> bytes:
    # Raw historical diff hashes must not depend on local presentation choices.
    if arguments[0] == "diff":
        arguments = (
            "diff",
            "--no-color",
            "--no-textconv",
            "--no-ext-diff",
            "--no-renames",
            "--indent-heuristic",
            "--diff-algorithm=myers",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            "--unified=3",
            "--inter-hunk-context=0",
            "--no-relative",
            "--output-indicator-new=+",
            "--output-indicator-old=-",
            "--output-indicator-context= ",
            "-O/dev/null",
            *arguments[1:],
        )
    environment = os.environ.copy()
    _ = environment.pop("GIT_DIFF_OPTS", None)
    completed = subprocess.run(
        [
            "git",
            "--no-pager",
            "-c",
            "color.ui=false",
            "-c",
            "diff.suppressBlankEmpty=false",
            *arguments,
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        check=False,
    )
    _require(completed.returncode == 0, f"Git history query failed: {arguments!r}")
    return completed.stdout


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _raw_record(record: dict[str, Any], session_path: str) -> dict[str, Any]:
    raw = record["raw_json_line_utf8"].encode("utf-8")
    _require(record["source_path"] == session_path, "archive session substitution")
    _require(
        type(record["line_1based"]) is int and record["line_1based"] > 0,
        "invalid archive line locator",
    )
    _require(len(raw) == record["raw_json_line_byte_count"], "archive byte count")
    _require(_sha256(raw) == record["raw_json_line_sha256"], "archive line digest")
    _require(raw.endswith(b"\n") and raw.count(b"\n") == 1, "archive line framing")
    return json.loads(raw)


def _authenticate_archive_records(value: Any, session_path: str) -> None:
    if isinstance(value, dict):
        record = cast(dict[str, Any], value)
        if "raw_json_line_utf8" in record:
            _ = _raw_record(record, session_path)
        else:
            for child in record.values():
                _authenticate_archive_records(child, session_path)
    elif isinstance(value, list):
        for child in cast(list[Any], value):
            _authenticate_archive_records(child, session_path)


def _authenticate_recovery(document: dict[str, Any], root: Path) -> None:
    corrections = document["corrections"]
    _require(len(corrections) == 2, "exactly two recovered designs are required")
    _require(
        frozenset(row["commit_sha"] for row in corrections) == RECOVERED_DESIGNS,
        "recovered design identity set",
    )
    for correction in corrections:
        commit = correction["commit_sha"]
        parent = correction["parent_sha"]
        parents = (
            _git(root, "rev-list", "--parents", "-n", "1", commit).decode().split()
        )
        _require(parents == [commit, parent], "recovered design sole parent")
        _require(
            _git(root, "rev-parse", f"{commit}^{{tree}}").decode().strip()
            == correction["tree_sha"],
            "recovered design tree",
        )
        paths = (
            _git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
            .decode()
            .splitlines()
        )
        _require(
            paths == list(DESIGN_PATHS) == correction["paths"],
            "recovered design path inventory",
        )
        patch = _git(
            root, "diff", "--no-ext-diff", "--binary", "--full-index", parent, commit
        )
        _require(
            _sha256(patch)
            == correction["complete_parent_relative_binary_full_index_diff_sha256"],
            "recovered design complete diff",
        )
        _require(
            [row["path"] for row in correction["committed_blobs"]] == paths,
            "recovered blob inventory",
        )
        candidate_pins = [
            correction["complete_parent_relative_binary_full_index_diff_sha256"]
        ]
        for blob in correction["committed_blobs"]:
            raw = _git(root, "show", f"{commit}:{blob['path']}")
            _require(
                _sha256(raw) == blob["sha256"] and len(raw) == blob["byte_count"],
                "recovered candidate blob",
            )
            _require(
                _git(root, "rev-parse", f"{commit}:{blob['path']}").decode().strip()
                == blob["git_blob_oid"],
                "recovered blob object",
            )
            blob_diff = _git(
                root,
                "diff",
                "--no-ext-diff",
                "--binary",
                "--full-index",
                parent,
                commit,
                "--",
                blob["path"],
            )
            _require(
                _sha256(blob_diff)
                == blob["parent_relative_binary_full_index_diff_sha256"],
                "recovered candidate individual diff",
            )
            candidate_pins.append(blob["sha256"])
        reviews = correction["reviews"]
        _require(
            len(reviews) == 2
            and len({r["session_id"] for r in reviews}) == 2
            and len({r["reviewer_identity"] for r in reviews}) == 2,
            "two distinct original reviewers are required",
        )
        for review in reviews:
            _require(review["verdict"] == "ACCEPT", "original review verdict")
            session = review["session_path"]
            _authenticate_archive_records(review, session)
            _require(
                review["session_id"] in session, "original review session identity"
            )
            final = review["final_response"]
            payload = _raw_record(final["source"], session)["payload"]
            _require(
                payload["type"] == "message"
                and payload["role"] == "assistant"
                and payload["phase"] == "final_answer"
                and payload["id"] == final["message_id"],
                "original final message",
            )
            text = "".join(part["text"] for part in payload["content"])
            _require(
                text == final["text_utf8"]
                and len(text.encode()) == final["text_byte_count"]
                and _sha256(text.encode()) == final["text_sha256"],
                "original final contribution digest",
            )
            pin_outputs = [
                _raw_record(record, session)["payload"]["output"]
                for record in review["pin_producing_tool_output_records"]
            ]
            rendered = "\n".join(
                part["text"] for output in pin_outputs for part in output
            )
            _require(
                all(pin in rendered for pin in candidate_pins),
                "reviewer's own candidate pin join",
            )


def authenticate_review_recovery(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Read exact D30-pinned recovery bytes and authenticate their Git joins."""
    raw = (root / REVIEW_RECOVERY_PATH).read_bytes()
    _require(_sha256(raw) == REVIEW_RECOVERY_SHA256, "frozen review recovery digest")
    design = _git(root, "show", f"{DESIGN_SHA}:{DESIGN_PATHS[1]}")
    _require(REVIEW_RECOVERY_SHA256.encode() in design, "D30 recovery binding")
    try:
        document = json.loads(raw)
        _authenticate_recovery(document, root)
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise HistoryError("malformed review recovery record") from error
    return document


STATUS_PATH = "docs/development/completion_ledger.md"
RED_PATHS = frozenset(
    {
        "tests/characterization/test_sci004_mmode.py",
        "tests/unit/test_sci004_phase3_dependency.py",
        "tests/unit/test_sci004_phase3_evidence.py",
        "tests/unit/test_sci004_phase3_red_failures.py",
        "tools/sci004_mmode_phase3_evidence.py",
        "tools/sci004_phase3_history.py",
        "tests/unit/test_sci004_phase3_history.py",
    }
)
SOURCE_PATHS = frozenset(
    {
        SOLVER_PATH,
        SNAPSHOT_FIXTURE_PATH,
        "src/radiosim/core/result.py",
        "tools/sci004_mmode_phase3_evidence.py",
        "tests/unit/test_sci004_phase3_evidence.py",
        "tools/sci004_mmode_phase3_acceptance.py",
        "tests/unit/test_sci004_phase3_acceptance.py",
    }
)
DISPOSAL_PINS = {
    "docs/development/sci004_mmode_phase3_acceptance.json": "283fb5264f5ecd86aed1300ae504b85946cf1f4d36b1c4c09bc92bb4f269421d",
    "docs/development/sci004_mmode_phase3_evidence.json": "600b51ac4d70778ee2d3bdf7b8842b83ba77dc34d541784ad1ad7d8e5be5f8ae",
    "docs/development/sci004_mmode_phase3_evidence.md": "039539a865b5d92e86379f44a324271232e8a947301e380ec7b1b1848e907b4e",
    "output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json": "07e59d3176866a78c17244849d6493365e9d410547e884cf56b254e60babe193",
}
# These already reviewed maintenance commits are exact immutable prerequisites,
# not a generic permission for source edits in a red phase.
PREREQUISITE_ROLES = {
    "860222ac90eaa7b9a2a1c3b282e3ec0f51b7834b": "review-recovery",
    "c245593df808e0a757925d5a02416b4608cd8661": "status",
    "1909829d828078fd36a905aa68cde50fcb4bfa16": "frame-regression",
    "cfad247831629241842ffecd5f7aaa5b2084493c": "frame-context-repair",
}
PREREQUISITE_PATHS = {
    "review-recovery": {REVIEW_RECOVERY_PATH},
    "status": {STATUS_PATH},
    "frame-regression": {"tests/unit/test_core/test_sci004_frame.py"},
    "frame-context-repair": {"src/radiosim/core/mmode/solver.py"},
}


def _exact_commit(root: Path, value: str) -> str:
    _require(
        type(value) is str
        and len(value) == 40
        and all(c in "0123456789abcdef" for c in value),
        "exact commit SHA required",
    )
    _require(
        _git(root, "rev-parse", "--verify", f"{value}^{{commit}}").decode().strip()
        == value,
        "commit SHA did not peel exactly",
    )
    return value


def _commit_delta(root: Path, parent: str, commit: str) -> dict[str, str]:
    fields = _git(root, "diff", "--name-status", "-z", parent, commit, "--").split(
        b"\0"
    )
    _require(fields[-1] == b"", "Git path framing")
    _ = fields.pop()
    _require(len(fields) % 2 == 0, "Git delta framing")
    delta = {
        path.decode(): status.decode()
        for status, path in zip(fields[::2], fields[1::2], strict=True)
    }
    _require(
        len(delta) * 2 == len(fields) and bool(delta), "empty/duplicate commit delta"
    )
    return delta


def authenticate_design_successor(root: Path = REPOSITORY_ROOT) -> None:
    """Authenticate the one ordinary, finalized D31 design edge and own header."""
    commit, parent = OPERATIVE_DESIGN_SHA, DESIGN_SUCCESSOR_PARENT
    _require(
        _git(root, "rev-list", "--parents", "-n", "1", commit).decode().split()
        == [commit, parent],
        "design successor sole parent",
    )
    _require(
        _commit_delta(root, parent, commit) == dict.fromkeys(DESIGN_PATHS, "M"),
        "design successor exact paths/change kinds",
    )
    for path, pin in zip(DESIGN_PATHS, DESIGN_SUCCESSOR_BLOBS, strict=True):
        _require(
            _sha256(_git(root, "show", f"{commit}:{path}")) == pin,
            "design successor landed blob",
        )
    _require(
        _sha256(_git(root, "diff", "--binary", "--full-index", parent, commit, "--"))
        == DESIGN_SUCCESSOR_DIFF,
        "design successor complete diff",
    )
    memo = _git(root, "show", f"{commit}:{DESIGN_PATHS[1]}").decode()
    header = memo.split("**Bounded correction #31 candidate", 1)[-1].split(
        "**Bounded correction #30 candidate", 1
    )[0]
    added = "\n".join(
        line[1:]
        for line in _git(root, "diff", parent, commit, "--", DESIGN_PATHS[1])
        .decode()
        .splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    _require(
        "**Review verification" in header
        and "each returned exact\n`ACCEPT`" in header
        and all(pin in header and pin in added for pin in DESIGN_SUCCESSOR_REVIEW_PINS)
        and not set(DESIGN_SUCCESSOR_REVIEW_PINS) & set(DESIGN_SUCCESSOR_BLOBS)
        and len({DESIGN_SUCCESSOR_REVIEW_PINS[2], DESIGN_SUCCESSOR_DIFF}) == 2,
        "design successor ordinary own-header review pins",
    )


def require_design_successor(records: list[dict[str, Any]]) -> None:
    """Require D31 once in an already Git-authenticated current red inventory."""
    rows = [row for row in records if row["role"] == "design-successor"]
    _require(
        len(rows) == 1 and rows[0]["sha"] == OPERATIVE_DESIGN_SHA,
        "current red range requires exact design successor once",
    )


def _bridge_ast(raw: bytes, path: str, required: bool | None) -> str:
    """Remove only D31's exact required runtime field and same-run keyword."""
    try:
        tree = ast.parse(raw)
    except (SyntaxError, ValueError) as exc:
        raise HistoryError("source bridge requires valid Python AST") from exc
    solver = path == SOLVER_PATH
    fields: list[ast.AnnAssign] = []
    scope: ast.AST = tree
    if solver:
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MModeSolverSnapshot"
        ]
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "solve_mmode"
        ]
        _require(len(classes) == len(functions) == 1, "source bridge owner identity")
        owner = classes[0]
        fields = [
            node
            for node in owner.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "input_identity_sha256"
        ]
        _require(len(fields) <= 1, "source bridge field cardinality")
        if fields:
            field = fields[0]
            _require(
                isinstance(field.annotation, ast.Name)
                and field.annotation.id == "str"
                and field.value is None
                and field.simple == 1,
                "source bridge required string field",
            )
            owner.body.remove(field)
        scope = functions[0]
    calls = [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MModeSolverSnapshot"
    ]
    _require(len(calls) == 1, "source bridge constructor cardinality")
    keywords = [kw for kw in calls[0].keywords if kw.arg == "input_identity_sha256"]
    _require(len(keywords) <= 1, "source bridge keyword cardinality")
    if keywords:
        expression = (
            'solved["input_identity_sha256"]'
            if solver
            else '_mmode_fixture_digest("input_identity")'
        )
        _require(
            ast.dump(keywords[0].value)
            == ast.dump(ast.parse(expression, mode="eval").body),
            "source bridge same-run keyword value",
        )
        calls[0].keywords.remove(keywords[0])
    present = bool(keywords)
    _require(not solver or bool(fields) == present, "source bridge field/keyword pair")
    _require(required is None or present == required, "source bridge required presence")
    return ast.dump(tree)


def _validate_bridge_delta(
    root: Path, parent: str, commit: str, path: str, *, cumulative: bool = False
) -> None:
    before = _bridge_ast(
        _git(root, "show", f"{parent}:{path}"), path, False if cumulative else None
    )
    after = _bridge_ast(_git(root, "show", f"{commit}:{path}"), path, True)
    _require(before == after, "source bridge changes the preceding AST")


def _phase_role(
    root: Path, phase: str, commit: str, parent: str, delta: dict[str, str]
) -> str:
    paths = set(delta)
    if phase == "prerequisite":
        _require(commit in PREREQUISITE_ROLES, "unknown prerequisite commit")
        role = PREREQUISITE_ROLES[commit]
        _require(paths == PREREQUISITE_PATHS[role], "prerequisite role paths")
        return role
    if paths == {STATUS_PATH}:
        _require(set(delta.values()) == {"M"}, "status commits only modify the ledger")
        return "status"
    if phase == "red":
        if commit == OPERATIVE_DESIGN_SHA:
            authenticate_design_successor(root)
            return "design-successor"
        _require(
            paths <= RED_PATHS and set(delta.values()) <= {"A", "M"},
            "red role path/change-kind boundary",
        )
        return "red"
    _require(phase == "source", "unknown phase role")
    if paths <= DISPOSAL_PINS.keys():
        _require(set(delta.values()) == {"D"}, "disposal must only delete artifacts")
        for path in paths:
            _require(
                _sha256(_git(root, "show", f"{parent}:{path}")) == DISPOSAL_PINS[path],
                "disposal predecessor bytes are not the rejected artifact",
            )
        return "disposal"
    _require(
        paths <= SOURCE_PATHS and set(delta.values()) == {"M"},
        "source role path/change-kind boundary",
    )
    for path in paths & {SOLVER_PATH, SNAPSHOT_FIXTURE_PATH}:
        _validate_bridge_delta(root, parent, commit, path)
    return "source"


def describe_phase_range(
    base_sha: str,
    terminal_sha: str,
    phase: str,
    *,
    root: Path = REPOSITORY_ROOT,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Recompute an exclusive-base/inclusive-tip first-parent phase range.

    Partial authoring ranges can be inspected with ``require_complete=False``;
    evidence/acceptance always call the complete three-range validator below.
    This authenticates topology, paths, change kinds, exact bytes and D31's
    restricted runtime-bridge AST delta. Other scientific semantics, factual
    status prose, sentinels and phase acceptance remain checks
    of the composing validators and independent reviews.
    """
    _require(phase in {"prerequisite", "red", "source"}, "unknown phase")
    base = _exact_commit(root, base_sha)
    terminal = _exact_commit(root, terminal_sha)
    shas = (
        _git(
            root,
            "rev-list",
            "--first-parent",
            "--ancestry-path",
            "--reverse",
            f"{base}..{terminal}",
        )
        .decode()
        .split()
    )
    _require(
        (not shas and base == terminal) or (bool(shas) and shas[-1] == terminal),
        "phase terminal is not on the declared ancestry path",
    )
    previous = base
    records: list[dict[str, Any]] = []
    aggregate: set[str] = set()
    for commit in shas:
        parents = (
            _git(root, "rev-list", "--parents", "-n", "1", commit).decode().split()
        )
        _require(
            parents == [commit, previous], "phase requires contiguous sole parents"
        )
        delta = _commit_delta(root, previous, commit)
        role = _phase_role(root, phase, commit, previous, delta)
        for path, status in delta.items():
            if status != "D":
                metadata = _git(root, "ls-tree", commit, "--", path).decode().split()
                _require(
                    len(metadata) >= 3
                    and metadata[0] in {"100644", "100755"}
                    and metadata[1] == "blob",
                    "phase files must be regular blobs",
                )
        records.append(
            {
                "sha": commit,
                "parent_sha": previous,
                "role": role,
                "paths": sorted(delta),
                "parent_diff_sha256": _sha256(
                    _git(
                        root, "diff", "--binary", "--full-index", previous, commit, "--"
                    )
                ),
            }
        )
        if role not in {"status", "design-successor"}:
            aggregate.update(delta)
        previous = commit
    if require_complete:
        if phase == "prerequisite":
            _require(
                base == DESIGN_SHA
                and terminal == PREREQUISITE_TIP_SHA
                and shas == list(PREREQUISITE_ROLES),
                "frozen prerequisite range",
            )
        else:
            if phase == "source":
                for path in (SOLVER_PATH, SNAPSHOT_FIXTURE_PATH):
                    _validate_bridge_delta(root, base, terminal, path, cumulative=True)
            expected = (
                RED_PATHS if phase == "red" else SOURCE_PATHS | DISPOSAL_PINS.keys()
            )
            _require(
                bool(shas) and frozenset(aggregate) == frozenset(expected),
                "complete phase path inventory",
            )
    return {"base_sha": base, "terminal_sha": terminal, "commits": records}


def validate_phase_ranges(
    document: Any,
    *,
    design_sha: str,
    red_sha: str,
    source_sha: str,
    root: Path = REPOSITORY_ROOT,
) -> None:
    """Authenticate the exact closed range schema and all three terminal joins."""
    _require(
        type(document) is dict and set(document) == {"prerequisite", "red", "source"},
        "phase_ranges keys",
    )
    _require(design_sha == OPERATIVE_DESIGN_SHA, "phase_ranges operative design")
    endpoints = (
        (DESIGN_SHA, PREREQUISITE_TIP_SHA),
        (PREREQUISITE_TIP_SHA, red_sha),
        (red_sha, source_sha),
    )
    for phase, (base, terminal) in zip(
        ("prerequisite", "red", "source"), endpoints, strict=True
    ):
        actual = document[phase]
        _require(
            type(actual) is dict
            and set(actual) == {"base_sha", "terminal_sha", "commits"}
            and type(actual["base_sha"]) is str
            and type(actual["terminal_sha"]) is str
            and type(actual["commits"]) is list,
            f"{phase} range schema",
        )
        entries = cast(dict[str, Any], actual)["commits"]
        for entry in cast(list[Any], entries):
            _require(
                type(entry) is dict
                and set(entry)
                == {"sha", "parent_sha", "role", "paths", "parent_diff_sha256"}
                and all(
                    type(entry[key]) is str
                    for key in ("sha", "parent_sha", "role", "parent_diff_sha256")
                )
                and type(entry["paths"]) is list
                and all(type(path) is str for path in entry["paths"]),
                f"{phase} commit schema",
            )
        expected = describe_phase_range(base, terminal, phase, root=root)
        if phase == "red":
            require_design_successor(expected["commits"])
        _require(actual == expected, f"{phase} range differs from exact Git history")
