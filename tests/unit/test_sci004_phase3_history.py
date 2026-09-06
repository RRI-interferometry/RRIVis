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
