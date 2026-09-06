"""Hostile joins for D30's finite historical-review recovery exceptions."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from tools import sci004_phase3_history as history


@pytest.fixture
def recovered():
    return history.authenticate_review_recovery()


def test_original_review_recovery_authenticates_both_designs(recovered) -> None:
    assert {row["commit_sha"] for row in recovered["corrections"]} == {
        "67da2b818b89511df8476b7010230c65d6cb6a75",
        "cfc9b10d655a4d9bedbd7d7750c4743f504bbaf9",
    }
    assert [len(row["reviews"]) for row in recovered["corrections"]] == [2, 2]


def test_recovery_rejects_byte_changes_before_any_git_query(tmp_path, monkeypatch):
    path = tmp_path / history.REVIEW_RECOVERY_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(
        (history.REPOSITORY_ROOT / history.REVIEW_RECOVERY_PATH).read_bytes() + b"\n"
    )

    def unexpected_git(*_args):
        pytest.fail("untrusted recovery bytes reached Git authentication")

    monkeypatch.setattr(history, "_git", unexpected_git)
    with pytest.raises(history.HistoryError, match="frozen review recovery digest"):
        history.authenticate_review_recovery(tmp_path)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("commit_sha", history.DESIGN_SHA, "design identity set"),
        ("parent_sha", history.DESIGN_SHA, "sole parent"),
        ("tree_sha", "0" * 40, "design tree"),
        ("paths", ["Fix.md"], "path inventory"),
        (
            "complete_parent_relative_binary_full_index_diff_sha256",
            "0" * 64,
            "complete diff",
        ),
    ],
)
def test_recovery_rejects_false_git_joins(recovered, field, value, error):
    candidate = copy.deepcopy(recovered)
    candidate["corrections"][0][field] = value
    with pytest.raises(history.HistoryError, match=error):
        history._authenticate_recovery(candidate, history.REPOSITORY_ROOT)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("sha256", "0" * 64, "candidate blob"),
        ("byte_count", 1, "candidate blob"),
        ("git_blob_oid", "0" * 40, "blob object"),
        ("parent_relative_binary_full_index_diff_sha256", "0" * 64, "individual diff"),
    ],
)
def test_recovery_rejects_false_candidate_blob_joins(recovered, field, value, error):
    candidate = copy.deepcopy(recovered)
    candidate["corrections"][0]["committed_blobs"][0][field] = value
    with pytest.raises(history.HistoryError, match=error):
        history._authenticate_recovery(candidate, history.REPOSITORY_ROOT)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "same_reviewer",
        "wrong_session",
        "wrong_final",
        "no_pins",
        "changed_archive",
        "wrong_verdict",
    ],
)
def test_recovery_rejects_missing_or_misattributed_reviews(recovered, mutation):
    candidate = copy.deepcopy(recovered)
    correction = candidate["corrections"][1]
    reviews = correction["reviews"]
    if mutation == "missing":
        reviews.pop()
    elif mutation == "duplicate":
        candidate["corrections"][0] = copy.deepcopy(correction)
    elif mutation == "same_reviewer":
        reviews[1]["reviewer_identity"] = reviews[0]["reviewer_identity"]
    elif mutation == "wrong_session":
        reviews[0]["session_path"] = reviews[1]["session_path"]
    elif mutation == "wrong_final":
        reviews[0]["final_response"]["text_utf8"] = "VERDICT: ACCEPT"
    elif mutation == "no_pins":
        # The terse D29 verdict cannot authenticate its candidate alone.
        reviews[0]["pin_producing_tool_output_records"] = []
    elif mutation == "changed_archive":
        reviews[0]["original_user_continuation_prompt"]["source"][
            "raw_json_line_sha256"
        ] = "0" * 64
    else:
        reviews[0]["verdict"] = "REJECT"
    with pytest.raises(history.HistoryError):
        history._authenticate_recovery(candidate, history.REPOSITORY_ROOT)


def test_recovery_is_portable_without_the_original_session_directory(monkeypatch):
    read_bytes = Path.read_bytes

    def repository_bytes_only(path):
        assert path.is_relative_to(history.REPOSITORY_ROOT)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", repository_bytes_only)
    assert len(history.authenticate_review_recovery()["corrections"]) == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("diff.context", "0"),
        ("diff.noprefix", "true"),
        ("color.ui", "always"),
        ("diff.mnemonicPrefix", "true"),
        ("diff.algorithm", "histogram"),
        ("diff.interHunkContext", "100"),
        ("diff.outputIndicatorNew", ">"),
        ("diff.outputIndicatorOld", "<"),
        ("diff.outputIndicatorContext", "."),
        ("diff.suppressBlankEmpty", "true"),
        ("diff.renames", "true"),
        ("diff.relative", "true"),
    ],
)
def test_recovery_diff_hashes_ignore_git_presentation_configuration(
    monkeypatch, key, value
):
    # Per-process Git configuration leaves the user's repository/config untouched.
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", key)
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", value)
    assert len(history.authenticate_review_recovery()["corrections"]) == 2


def test_recovery_neutralizes_git_diff_environment_without_mutating_caller(monkeypatch):
    monkeypatch.setenv("GIT_DIFF_OPTS", "--unified=0")
    assert len(history.authenticate_review_recovery()["corrections"]) == 2
    assert os.environ["GIT_DIFF_OPTS"] == "--unified=0"


def test_live_prerequisite_range_retains_every_commit_and_role():
    result = history.describe_phase_range(
        history.DESIGN_SHA, history.PREREQUISITE_TIP_SHA, "prerequisite"
    )
    assert [row["sha"] for row in result["commits"]] == list(history.PREREQUISITE_ROLES)
    assert [row["role"] for row in result["commits"]] == [
        "review-recovery",
        "status",
        "frame-regression",
        "frame-context-repair",
    ]
    assert all(len(row["parent_diff_sha256"]) == 64 for row in result["commits"])


@pytest.fixture
def phase_objects(tmp_path, monkeypatch):
    """Synthetic Git objects only: no checkout, refs or implementation worktree."""
    import subprocess

    root = tmp_path / "objects.git"
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "History fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "History fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
        "GIT_INDEX_FILE": str(tmp_path / "fixture.index"),
    }
    subprocess.run(
        ["git", "init", "--bare", str(root)], check=True, capture_output=True
    )

    def git(*args, data=None):
        return (
            subprocess.run(
                ["git", *args],
                cwd=root,
                env=environment,
                input=data,
                check=True,
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )

    def commit(parent, edits, *, extra_parent=None, mode="100644"):
        git("read-tree", f"{parent}^{{tree}}" if parent else "--empty")
        for path, content in edits.items():
            if content is None:
                git(
                    "update-index",
                    "--index-info",
                    data=f"0 {'0' * 40}\t{path}\n".encode(),
                )
            else:
                blob = git("hash-object", "-w", "--stdin", data=content)
                git("update-index", "--add", "--cacheinfo", mode, blob, path)
        parents = ["-p", parent] if parent else []
        if extra_parent:
            parents.extend(["-p", extra_parent])
        return git("commit-tree", git("write-tree"), *parents, data=b"fixture\n")

    # Preserve real path grants, substituting only synthetic prerequisite identities
    # and rejected-artifact byte pins; production historical tests above use originals.
    initial = dict.fromkeys(history.SOURCE_PATHS, b"before\n")
    initial.update(dict.fromkeys(history.DISPOSAL_PINS, b"rejected\n"))
    initial[history.STATUS_PATH] = b"initial status\n"
    base = commit(None, initial)
    prerequisite = commit(base, {history.REVIEW_RECOVERY_PATH: b"fixture recovery\n"})
    monkeypatch.setattr(history, "DESIGN_SHA", base)
    monkeypatch.setattr(history, "PREREQUISITE_TIP_SHA", prerequisite)
    monkeypatch.setattr(
        history, "PREREQUISITE_ROLES", {prerequisite: "review-recovery"}
    )
    monkeypatch.setattr(
        history,
        "DISPOSAL_PINS",
        {path: history._sha256(b"rejected\n") for path in history.DISPOSAL_PINS},
    )
    red = commit(prerequisite, dict.fromkeys(history.RED_PATHS, b"red fixture\n"))
    status = commit(red, {history.STATUS_PATH: b"red status\n"})
    source = commit(status, dict.fromkeys(history.SOURCE_PATHS, b"source fixture\n"))
    terminal = commit(source, dict.fromkeys(history.DISPOSAL_PINS))
    return root, commit, base, prerequisite, red, status, source, terminal


def _phase_document(objects):
    root, _, base, prerequisite, _, red, _, source = objects
    return {
        phase: history.describe_phase_range(start, end, phase, root=root)
        for phase, start, end in (
            ("prerequisite", base, prerequisite),
            ("red", prerequisite, red),
            ("source", red, source),
        )
    }


def _validate_document(document, objects):
    root, _, base, _, _, red, _, source = objects
    history.validate_phase_ranges(
        document, design_sha=base, red_sha=red, source_sha=source, root=root
    )


def test_complete_ranges_join_git_objects_and_keep_status_and_disposal(phase_objects):
    document = _phase_document(phase_objects)
    _validate_document(document, phase_objects)
    assert [row["role"] for row in document["red"]["commits"]] == ["red", "status"]
    assert [row["role"] for row in document["source"]["commits"]] == [
        "source",
        "disposal",
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_phase",
        "extra_phase",
        "extra_range_key",
        "extra_commit_key",
        "tuple_commits",
        "tuple_paths",
        "missing_commit",
        "duplicate_commit",
        "reordered",
        "wrong_parent",
        "wrong_digest",
        "wrong_role",
        "unsorted_paths",
        "duplicate_path",
        "wrong_base",
        "wrong_terminal",
        "wrong_design",
        "red_source_swap",
    ],
)
def test_range_document_rejects_forged_or_incomplete_history(phase_objects, mutation):
    document = _phase_document(phase_objects)
    red = document["red"]
    row = red["commits"][0]
    if mutation == "missing_phase":
        document.pop("source")
    elif mutation == "extra_phase":
        document["evidence"] = {}
    elif mutation == "extra_range_key":
        red["accepted"] = True
    elif mutation == "extra_commit_key":
        row["accepted"] = True
    elif mutation == "tuple_commits":
        red["commits"] = tuple(red["commits"])
    elif mutation == "tuple_paths":
        row["paths"] = tuple(row["paths"])
    elif mutation == "missing_commit":
        red["commits"].pop()
    elif mutation == "duplicate_commit":
        red["commits"].append(copy.deepcopy(row))
    elif mutation == "reordered":
        red["commits"].reverse()
    elif mutation == "wrong_parent":
        row["parent_sha"] = row["sha"]
    elif mutation == "wrong_digest":
        row["parent_diff_sha256"] = "0" * 64
    elif mutation == "wrong_role":
        row["role"] = "status"
    elif mutation == "unsorted_paths":
        row["paths"].reverse()
    elif mutation == "duplicate_path":
        row["paths"].append(row["paths"][0])
    elif mutation == "wrong_base":
        red["base_sha"] = red["terminal_sha"]
    elif mutation == "wrong_terminal":
        red["terminal_sha"] = red["base_sha"]
    elif mutation == "wrong_design":
        with pytest.raises(history.HistoryError, match="operative design"):
            history.validate_phase_ranges(
                document,
                design_sha="0" * 40,
                red_sha=phase_objects[5],
                source_sha=phase_objects[7],
            )
        return
    else:
        document["red"], document["source"] = document["source"], document["red"]
    with pytest.raises(history.HistoryError):
        _validate_document(document, phase_objects)


@pytest.mark.parametrize(
    "violation",
    [
        "merge",
        "nonancestor",
        "ungranted_path",
        "symlink",
        "incomplete_paths",
        "mixed_status",
        "empty_red",
        "abbreviated_sha",
        "modified_disposal",
        "forged_disposal",
    ],
)
def test_actual_git_history_rejects_role_and_topology_violations(
    phase_objects, violation
):
    root, commit, base, prerequisite, red, status, source, terminal = phase_objects
    phase = "red"
    start = prerequisite
    end = red
    if violation == "merge":
        side = commit(prerequisite, {history.STATUS_PATH: b"side\n"})
        end = commit(red, {history.STATUS_PATH: b"merge\n"}, extra_parent=side)
    elif violation == "nonancestor":
        end = commit(base, {history.STATUS_PATH: b"other descendant\n"})
    elif violation == "ungranted_path":
        end = commit(red, {"Fix.md": b"ungranted\n"})
    elif violation == "symlink":
        end = commit(red, {sorted(history.RED_PATHS)[0]: b"somewhere"}, mode="120000")
    elif violation == "incomplete_paths":
        end = commit(prerequisite, {sorted(history.RED_PATHS)[0]: b"one only\n"})
    elif violation == "mixed_status":
        end = commit(
            red,
            {history.STATUS_PATH: b"mixed\n", sorted(history.RED_PATHS)[0]: b"mixed\n"},
        )
    elif violation == "empty_red":
        end = start
    elif violation == "abbreviated_sha":
        end = red[:12]
    else:
        phase, start = "source", status
        path = next(iter(history.DISPOSAL_PINS))
        forged = commit(source, {path: b"replacement artifact\n"})
        end = (
            forged if violation == "modified_disposal" else commit(forged, {path: None})
        )
        if violation == "forged_disposal":
            start = forged
    with pytest.raises(history.HistoryError):
        history.describe_phase_range(start, end, phase, root=root)
