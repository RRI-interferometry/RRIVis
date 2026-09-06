"""Authenticate SCI-004 correction #30's immutable review and phase history.

Historical review recovery is portable: D30 pins the complete retained record,
whose original archive contributions were independently authenticated before
landing. CI verifies those retained bytes and their Git joins without requiring
the author's private session directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DESIGN_SHA = "d3ddb10ae01ab450f5337d06c9588ce8144cf1e5"
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
    environment.pop("GIT_DIFF_OPTS", None)
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
        if "raw_json_line_utf8" in value:
            _raw_record(value, session_path)
        else:
            for child in value.values():
                _authenticate_archive_records(child, session_path)
    elif isinstance(value, list):
        for child in value:
            _authenticate_archive_records(child, session_path)


def _authenticate_recovery(document: dict[str, Any], root: Path) -> None:
    corrections = document["corrections"]
    _require(len(corrections) == 2, "exactly two recovered designs are required")
    _require(
        {row["commit_sha"] for row in corrections} == RECOVERED_DESIGNS,
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
