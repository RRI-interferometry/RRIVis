"""Hostile joins for D30's finite historical-review recovery exceptions."""

from __future__ import annotations

import copy
import os
import subprocess
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from tools import sci004_phase3_history as history

# One reference to the internal reader shared by raw-object boundary tests.
history_git = history._git


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


SOLVER_BEFORE = b"""from dataclasses import dataclass
@dataclass(frozen=True)
class MModeSolverSnapshot:
    value: str

def solve_mmode():
    solved = {'input_identity_sha256': 'identity'}
    return MModeSolverSnapshot(value='unchanged')
"""
SOLVER_AFTER = SOLVER_BEFORE.replace(
    b"    value: str", b"    input_identity_sha256: str\n    value: str"
).replace(
    b"value='unchanged')",
    b"value='unchanged', input_identity_sha256=solved['input_identity_sha256'])",
)
FIXTURE_BEFORE = b"snapshot = MModeSolverSnapshot(value='unchanged')\n"
FIXTURE_AFTER = FIXTURE_BEFORE.replace(
    b"value='unchanged')",
    b"value='unchanged', input_identity_sha256=_mmode_fixture_digest('input_identity'))",
)


def source_design_object(
    root: Path,
    commit: Callable[..., str],
    parent: str,
    edge: history.SourceDesignEdge,
) -> history.SourceDesignEdge:
    """Create a minimal ordinary source-design edge using actual Git objects."""
    memo = (
        f"**Bounded correction #{edge.correction} candidate\n"
        "**Review verification\n"
        f"`{edge.reviewers[0]}` and `{edge.reviewers[1]}` "
        "each returned exact\n`ACCEPT`\n"
        f"complete round-{edge.round_number} candidate bytes\n{parent}\n"
        + "\n".join(edge.review_pins)
        + "\n"
    ).encode()
    companion = (
        f"**Current continuation #{edge.correction} candidate\n"
        "physics/governance and computational/provenance each returned ACCEPT\n"
        f"complete round-{edge.round_number} final D{edge.correction} header\n"
    ).encode()
    retained: list[bytes] = []
    for path, label in zip(
        history.DESIGN_PATHS,
        ("Current continuation", "Bounded correction"),
        strict=True,
    ):
        tail = history_git(root, "show", f"{parent}:{path}")
        # Preserve the old review lines so Git identifies the new header as added.
        # Reversed-order hostile fixtures may already contain this heading.
        tail = tail.replace(
            f"**{label} #{edge.correction} candidate".encode(), b"**Retained candidate"
        )
        end = f"**{label} #{edge.correction - 1} candidate".encode()
        if f"**{label} #{edge.correction + 1} candidate".encode() in tail:
            # A deliberately reversed edge must still have one bounded own review.
            tail = end + b"\n" + tail.replace(end, b"**Retained prior boundary")
        if end not in tail:
            tail = end + b"\n" + tail
        retained.append(tail)
    companion += retained[0]
    memo += retained[1]
    sha = commit(
        parent, dict(zip(history.DESIGN_PATHS, (companion, memo), strict=True))
    )
    return replace(
        edge,
        sha=sha,
        parent=parent,
        blobs=(sha256(companion).hexdigest(), sha256(memo).hexdigest()),
        patch=sha256(
            history_git(root, "diff", "--binary", "--full-index", parent, sha, "--")
        ).hexdigest(),
    )


@pytest.fixture
def phase_objects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PhaseObjects:
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
    initial = dict.fromkeys(history.SOURCE_PATHS, b"# before\n")
    initial[history.SOLVER_PATH] = SOLVER_BEFORE
    initial[history.SNAPSHOT_FIXTURE_PATH] = FIXTURE_BEFORE
    frame, workflow = _d34_original_blobs()
    initial[history.FRAME_PATH] = frame
    initial[history.WORKFLOW_PATH] = workflow
    initial.update(dict.fromkeys(history.DESIGN_PATHS, b"old design\n"))
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
    red = commit(prerequisite, dict.fromkeys(history.RED_PATHS, b"# red fixture\n"))
    memo = (
        "**Bounded correction #31 candidate\n**Review verification\n"
        "each returned exact\n`ACCEPT`\n"
        + "\n".join(history.DESIGN_SUCCESSOR_REVIEW_PINS)
        + "\n**Bounded correction #30 candidate\n"
    ).encode()
    design = commit(
        red, dict(zip(history.DESIGN_PATHS, (b"new plan\n", memo), strict=True))
    )
    monkeypatch.setattr(history, "OPERATIVE_DESIGN_SHA", design)
    monkeypatch.setattr(history, "RED_DESIGN_SHA", design)
    monkeypatch.setattr(history, "DESIGN_SUCCESSOR_PARENT", red)
    monkeypatch.setattr(
        history,
        "DESIGN_SUCCESSOR_BLOBS",
        tuple(history._sha256(raw) for raw in (b"new plan\n", memo)),
    )
    monkeypatch.setattr(
        history,
        "DESIGN_SUCCESSOR_DIFF",
        history._sha256(
            history_git(root, "diff", "--binary", "--full-index", red, design, "--")
        ),
    )
    status = commit(design, {history.STATUS_PATH: b"red status\n"})
    edits = dict.fromkeys(history.SOURCE_PATHS, b"# source fixture\n")
    edits.update(
        {
            history.SOLVER_PATH: SOLVER_AFTER,
            history.SNAPSHOT_FIXTURE_PATH: FIXTURE_AFTER,
            history.FRAME_PATH: _d34_repaired_frame(frame),
        }
    )
    source_designs: list[history.SourceDesignEdge] = []
    parent = status
    for edge in history.SOURCE_DESIGN_EDGES:
        created = source_design_object(
            root, cast(Callable[..., str], commit), parent, edge
        )
        source_designs.append(created)
        parent = created.sha
    monkeypatch.setattr(history, "SOURCE_DESIGN_EDGES", tuple(source_designs))
    monkeypatch.setattr(history, "HISTORICAL_SOURCE_DESIGN_SHA", source_designs[0].sha)
    monkeypatch.setattr(
        history, "HISTORICAL_D33_SOURCE_DESIGN_SHA", source_designs[1].sha
    )
    monkeypatch.setattr(history, "SOURCE_DESIGN_SHA", source_designs[2].sha)
    monkeypatch.setattr(
        history, "HISTORICAL_D34_SOURCE_DESIGN_SHA", source_designs[2].sha
    )
    regression = commit(
        parent, {history.FRAME_TEST_PATH: edits.pop(history.FRAME_TEST_PATH)}
    )
    source = commit(regression, edits)
    terminal = commit(source, dict.fromkeys(history.DISPOSAL_PINS))
    return (
        root,
        cast(Callable[..., str], commit),
        base,
        prerequisite,
        red,
        status,
        source,
        terminal,
    )


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
    root, _, _, _, _, red, _, source = objects
    history.validate_phase_ranges(
        document,
        design_sha=history.SOURCE_DESIGN_SHA,
        red_sha=red,
        source_sha=source,
        root=root,
    )


def test_complete_ranges_join_git_objects_and_keep_status_and_disposal(phase_objects):
    document = _phase_document(phase_objects)
    _validate_document(document, phase_objects)
    assert [row["role"] for row in document["red"]["commits"]] == [
        "red",
        "design-successor",
        "status",
    ]
    assert [row["role"] for row in document["source"]["commits"]] == [
        "source-design-successor",
        "source-design-successor",
        "source-design-successor",
        "source",
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


def test_live_d31_authentication_and_historical_range_origin():
    history.authenticate_design_successor()
    old = history.describe_phase_range(
        history.PREREQUISITE_TIP_SHA, history.DESIGN_SUCCESSOR_PARENT, "red"
    )
    with pytest.raises(history.HistoryError, match="successor once"):
        history.require_design_successor(old["commits"])
    current = history.describe_phase_range(
        history.PREREQUISITE_TIP_SHA, history.OPERATIVE_DESIGN_SHA, "red"
    )
    history.require_design_successor(current["commits"])
    assert current["commits"][:-1] == old["commits"]


@pytest.mark.parametrize(
    "pin",
    [
        "DESIGN_SUCCESSOR_PARENT",
        "DESIGN_SUCCESSOR_BLOBS",
        "DESIGN_SUCCESSOR_DIFF",
        "DESIGN_SUCCESSOR_REVIEW_PINS",
    ],
)
def test_design_successor_rejects_forged_pins(phase_objects, monkeypatch, pin):
    value = getattr(history, pin)
    monkeypatch.setattr(
        history,
        pin,
        ("0" * 64,) * len(value) if isinstance(value, tuple) else "0" * len(value),
    )
    with pytest.raises(history.HistoryError):
        history.authenticate_design_successor(phase_objects[0])


def test_no_second_design_edge_or_omitted_successor(phase_objects):
    root, commit, _, prerequisite, red, status, _, _ = phase_objects
    historical = history.describe_phase_range(prerequisite, red, "red", root=root)
    with pytest.raises(history.HistoryError, match="successor once"):
        history.require_design_successor(historical["commits"])
    later = commit(status, {history.DESIGN_PATHS[0]: b"unauthorized design\n"})
    with pytest.raises(history.HistoryError, match="red role"):
        history.describe_phase_range(prerequisite, later, "red", root=root)


@pytest.mark.parametrize(
    "path,before,after",
    [
        (history.SOLVER_PATH, SOLVER_BEFORE, SOLVER_AFTER),
        (history.SNAPSHOT_FIXTURE_PATH, FIXTURE_BEFORE, FIXTURE_AFTER),
    ],
)
def test_bridge_allows_only_exact_ast_additions(phase_objects, path, before, after):
    root, commit, _, _, _, status, _, _ = phase_objects
    tip = commit(status, {path: after})
    history.describe_phase_range(
        status, tip, "source", root=root, require_complete=False
    )
    assert history._bridge_ast(before, path, False) == history._bridge_ast(
        after, path, True
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "optional_field",
        "wrong_type",
        "wrong_key",
        "fallback",
        "extra_computation",
        "serialization",
        "field_only",
        "keyword_only",
        "duplicate_keyword",
        "duplicate_field",
        "wrong_fixture",
        "fixture_extra",
        "syntax",
        "transient_change",
        "bridge_removed",
        "bridge_predates_source",
    ],
)
def test_source_bridge_rejects_semantic_or_scope_changes(phase_objects, mutation):
    root, commit, _, _, _, status, source, _ = phase_objects
    path, raw = history.SOLVER_PATH, SOLVER_AFTER
    if mutation == "optional_field":
        raw = raw.replace(b"sha256: str", b"sha256: str = None")
    elif mutation == "wrong_type":
        raw = raw.replace(b"sha256: str", b"sha256: object")
    elif mutation == "wrong_key":
        raw = raw.replace(
            b"solved['input_identity_sha256']", b"solved['output_sha256']"
        )
    elif mutation == "fallback":
        raw = raw.replace(
            b"solved['input_identity_sha256']", b"solved.get('input_identity_sha256')"
        )
    elif mutation in {"extra_computation", "transient_change"}:
        raw += b"numerical_change = 42\n"
    elif mutation == "serialization":
        raw = raw.replace(
            b"    value: str",
            b"    def as_mapping(self): return {'extra': self.input_identity_sha256}\n    value: str",
        )
    elif mutation == "field_only":
        raw = raw.replace(
            b", input_identity_sha256=solved['input_identity_sha256']", b""
        )
    elif mutation == "keyword_only":
        raw = raw.replace(b"    input_identity_sha256: str\n", b"")
    elif mutation == "duplicate_keyword":
        raw = raw.replace(
            b"value='unchanged',",
            b"value='unchanged', input_identity_sha256=solved['input_identity_sha256'],",
        )
    elif mutation == "duplicate_field":
        raw = raw.replace(
            b"    value: str", b"    input_identity_sha256: str\n    value: str"
        )
    elif mutation in {"wrong_fixture", "fixture_extra"}:
        path = history.SNAPSHOT_FIXTURE_PATH
        raw = (
            FIXTURE_AFTER.replace(b"'input_identity'", b"'output_identity'")
            if mutation == "wrong_fixture"
            else FIXTURE_AFTER + b"unrelated = 1\n"
        )
    elif mutation == "syntax":
        raw = b"not Python (\n"
    elif mutation == "bridge_removed":
        status, raw = source, SOLVER_BEFORE
    elif mutation == "bridge_predates_source":
        with pytest.raises(history.HistoryError, match="required presence"):
            history._validate_bridge_delta(root, source, source, path, cumulative=True)
        return
    tip = commit(status, {path: raw})
    if mutation == "transient_change":
        tip = commit(tip, {path: SOLVER_AFTER})
    with pytest.raises(history.HistoryError):
        history.describe_phase_range(
            status, tip, "source", root=root, require_complete=False
        )


@pytest.mark.parametrize(
    "violation", ["pins_in_old_header", "missing_verdict", "extra_path"]
)
def test_rebound_design_objects_still_need_own_review_header(
    phase_objects, monkeypatch, violation
):
    root, commit, _, _, red, _, _, _ = phase_objects
    original = history_git(
        root, "show", f"{history.OPERATIVE_DESIGN_SHA}:{history.DESIGN_PATHS[1]}"
    )
    if violation == "pins_in_old_header":
        memo = original.replace(
            b"**Review verification",
            b"**Bounded correction #30 candidate\n**Review verification",
        )
    else:
        memo = (
            original.replace(b"`ACCEPT`", b"pending")
            if violation == "missing_verdict"
            else original
        )
    edits = {history.DESIGN_PATHS[0]: b"new plan\n", history.DESIGN_PATHS[1]: memo}
    if violation == "extra_path":
        edits[history.STATUS_PATH] = b"mixed status\n"
    forged = commit(red, edits)
    monkeypatch.setattr(history, "OPERATIVE_DESIGN_SHA", forged)
    monkeypatch.setattr(
        history,
        "DESIGN_SUCCESSOR_BLOBS",
        tuple(history._sha256(edits[path]) for path in history.DESIGN_PATHS),
    )
    monkeypatch.setattr(
        history,
        "DESIGN_SUCCESSOR_DIFF",
        history._sha256(
            history_git(root, "diff", "--binary", "--full-index", red, forged, "--")
        ),
    )
    with pytest.raises(history.HistoryError):
        history.authenticate_design_successor(root)


PhaseObjects = tuple[Path, Callable[..., str], str, str, str, str, str, str]


def _hostile_git(root: Path, *arguments: str, data: bytes | None = None) -> bytes:
    """Create hostile overlays only inside the synthetic bare fixture."""
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        },
        input=data,
        capture_output=True,
        check=True,
    ).stdout


def test_complete_history_rejects_replaced_unauthorized_terminal(
    phase_objects: PhaseObjects,
) -> None:
    root, commit, _, _, _, _, source, good = phase_objects
    edits: dict[str, bytes | None] = dict.fromkeys(history.DISPOSAL_PINS)
    edits["Fix.md"] = b"unauthorized source change\n"
    bad = commit(source, edits)
    bad_objects = (*phase_objects[:-1], bad)
    with pytest.raises(history.HistoryError, match="source role"):
        _ = _phase_document(bad_objects)
    _ = _hostile_git(root, "replace", bad, good)
    # The ambient Git view reproduces the reported complete-validator bypass.
    assert b"Fix.md" not in _hostile_git(root, "diff", "--name-only", source, bad)
    assert b"Fix.md" in history_git(root, "diff", "--name-only", source, bad)
    with pytest.raises(history.HistoryError, match="source role"):
        _ = _phase_document(bad_objects)
    _validate_document(_phase_document(phase_objects), phase_objects)


def test_history_reads_original_blob_despite_replacement(
    phase_objects: PhaseObjects,
) -> None:
    root, _, base, _, _, _, _, _ = phase_objects
    path = next(iter(history.DISPOSAL_PINS))
    original = _hostile_git(root, "rev-parse", f"{base}:{path}").decode().strip()
    replacement = (
        _hostile_git(
            root, "hash-object", "-w", "--stdin", data=b"counterfeit artifact\n"
        )
        .decode()
        .strip()
    )
    _ = _hostile_git(root, "replace", original, replacement)
    assert _hostile_git(root, "show", f"{base}:{path}") == b"counterfeit artifact\n"
    assert history_git(root, "show", f"{base}:{path}") == b"rejected\n"
    _validate_document(_phase_document(phase_objects), phase_objects)


@pytest.mark.parametrize("external", [False, True])
def test_history_ignores_grafted_parents(
    phase_objects: PhaseObjects, monkeypatch: pytest.MonkeyPatch, external: bool
) -> None:
    root, _, base, _, _, _, source, terminal = phase_objects
    graft = root / ("external-grafts" if external else "info/grafts")
    _ = graft.write_text(f"{terminal} {base}\n")
    if external:
        monkeypatch.setenv("GIT_GRAFT_FILE", str(graft))
    else:
        assert _hostile_git(
            root, "rev-list", "--parents", "-n", "1", terminal
        ).split() == [terminal.encode(), base.encode()]
    assert history_git(root, "rev-list", "--parents", "-n", "1", terminal).split() == [
        terminal.encode(),
        source.encode(),
    ]
    _validate_document(_phase_document(phase_objects), phase_objects)


@pytest.mark.parametrize(
    "variable",
    [
        "GIT_DIR",
        "GIT_COMMON_DIR",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_INDEX_FILE",
        "GIT_SHALLOW_FILE",
        "GIT_CONFIG",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_NAMESPACE",
        "GIT_REPLACE_REF_BASE",
        "GIT_ATTR_SOURCE",
        "GIT_CEILING_DIRECTORIES",
    ],
)
def test_complete_history_ignores_caller_git_routing(
    phase_objects: PhaseObjects, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    expected = _phase_document(phase_objects)
    value = str(phase_objects[0] / "absent-redirect")
    monkeypatch.setenv(variable, value)
    observed = _phase_document(phase_objects)
    assert observed == expected
    _validate_document(observed, phase_objects)
    assert os.environ[variable] == value


def test_history_rejects_config_hidden_gitlink(
    phase_objects: PhaseObjects,
) -> None:
    root, _, _, _, _, _, source, terminal = phase_objects
    tree = _hostile_git(root, "ls-tree", f"{terminal}^{{tree}}")
    tree += f"160000 commit {source}\tforbidden-module\n".encode()
    new_tree = _hostile_git(root, "mktree", data=tree).decode().strip()
    raw = _hostile_git(root, "cat-file", "commit", terminal)
    _, remainder = raw.split(b"\n", 1)
    bad = (
        _hostile_git(
            root,
            "hash-object",
            "-w",
            "-t",
            "commit",
            "--stdin",
            data=f"tree {new_tree}\n".encode() + remainder,
        )
        .decode()
        .strip()
    )
    _ = _hostile_git(root, "config", "diff.ignoreSubmodules", "all")
    assert b"forbidden-module" not in _hostile_git(
        root, "diff", "--name-only", source, bad
    )
    assert b"forbidden-module" in history_git(root, "diff", "--name-only", source, bad)
    assert b"forbidden-module" in history_git(
        root, "diff-tree", "--no-commit-id", "--name-only", "-r", bad
    )
    with pytest.raises(history.HistoryError, match="source role"):
        _ = _phase_document((*phase_objects[:-1], bad))


def test_history_disables_local_external_diff_and_textconv(
    phase_objects: PhaseObjects,
) -> None:
    root, _, _, _, _, _, source, _ = phase_objects
    expected = _phase_document(phase_objects)
    _ = _hostile_git(root, "config", "diff.external", "false")
    assert _phase_document(phase_objects) == expected
    _ = (root / "info/attributes").write_text("*.py diff=hostile\n")
    _ = _hostile_git(root, "config", "diff.hostile.textconv", "false")
    assert history_git(root, "show", f"{source}:src/radiosim/core/result.py") == (
        b"# source fixture\n"
    )
    with pytest.raises(history.HistoryError, match="effective diff attribute"):
        _ = _phase_document(phase_objects)


@pytest.mark.parametrize(
    "attribute", ["-diff", "diff", "diff=hostile", "diff=unspecified"]
)
def test_complete_history_refuses_local_patch_attribute_transforms(
    phase_objects: PhaseObjects,
    attribute: str,
) -> None:
    root, _, _, _, _, status, source, _ = phase_objects
    original = _phase_document(phase_objects)
    _validate_document(original, phase_objects)
    _ = (root / "info/attributes").write_text(
        f"src/radiosim/core/result.py {attribute}\n"
    )
    for driver in ("hostile", "unspecified"):
        _ = _hostile_git(root, "config", f"diff.{driver}.binary", "true")
    if attribute != "diff":
        assert b"GIT binary patch" in _hostile_git(
            root, "diff", "--binary", "--full-index", status, source, "--"
        )
    with pytest.raises(history.HistoryError, match="effective diff attribute"):
        _ = _phase_document(phase_objects)
    with pytest.raises(history.HistoryError, match="effective diff attribute"):
        _validate_document(original, phase_objects)


def test_history_preserves_unrelated_lfs_and_disabled_diff_attributes(
    phase_objects: PhaseObjects,
) -> None:
    root, _, _, _, _, _, _, _ = phase_objects
    expected = _phase_document(phase_objects)
    _ = (root / "info/attributes").write_text(
        "*.fits filter=lfs diff=lfs merge=lfs -text\n"
        "src/radiosim/core/result.py !diff\n"
    )
    _ = _hostile_git(root, "config", "diff.unspecified.binary", "true")
    assert _phase_document(phase_objects) == expected
    _validate_document(expected, phase_objects)


def test_history_ignores_local_binary_threshold(
    phase_objects: PhaseObjects,
) -> None:
    root, _, _, _, _, status, source, _ = phase_objects
    expected = _phase_document(phase_objects)
    args = ("diff", "--full-index", status, source, "--")
    original_patch = history_git(root, *args)
    _ = _hostile_git(root, "config", "core.bigFileThreshold", "0")
    # --binary preloads buffers before classification in Git 2.55; exercise
    # the ordinary patch form also used to authenticate review-header additions.
    altered_patch = _hostile_git(root, *args)
    assert b"Binary files " in altered_patch
    assert altered_patch != original_patch
    assert history_git(root, *args) == original_patch
    assert _phase_document(phase_objects) == expected
    _validate_document(expected, phase_objects)


def test_complete_history_refuses_configured_default_diff_driver(
    phase_objects: PhaseObjects,
) -> None:
    root, _, _, _, _, status, source, _ = phase_objects
    expected = _phase_document(phase_objects)
    _ = _hostile_git(root, "config", "diff.default.binary", "true")
    assert (
        _hostile_git(
            root, "check-attr", "--all", "-z", "--", "src/radiosim/core/result.py"
        )
        == b""
    )
    assert b"GIT binary patch" in _hostile_git(
        root, "diff", "--binary", "--full-index", status, source, "--"
    )
    with pytest.raises(history.HistoryError, match="configured default diff driver"):
        _ = _phase_document(phase_objects)
    with pytest.raises(history.HistoryError, match="configured default diff driver"):
        _validate_document(expected, phase_objects)


@pytest.mark.parametrize("edge", history.SOURCE_DESIGN_EDGES)
def test_live_source_design_edges_authenticate(edge: history.SourceDesignEdge) -> None:
    history.authenticate_source_design_successor(edge.sha)
    assert history.RED_DESIGN_SHA == history.OPERATIVE_DESIGN_SHA
    assert tuple(item.sha for item in history.SOURCE_DESIGN_EDGES) == (
        history.HISTORICAL_SOURCE_DESIGN_SHA,
        history.HISTORICAL_D33_SOURCE_DESIGN_SHA,
        history.SOURCE_DESIGN_SHA,
    )
    assert history.SOURCE_DESIGN_SHA == history.D34_DESIGN_EDGE.sha
    assert edge.reviewers == (
        "/root/e_lifecycle_physics"
        if edge.correction == 34
        else "/root/d30_physics_review",
        "/root/d30_provenance_review",
    )


@pytest.mark.parametrize("sha", [history.RED_DESIGN_SHA, history.DESIGN_SHA, "HEAD"])
def test_source_design_authentication_rejects_other_identities(sha: str) -> None:
    with pytest.raises(history.HistoryError, match="unknown source design"):
        history.authenticate_source_design_successor(sha)


@pytest.mark.parametrize(
    "edge", (*history.SOURCE_DESIGN_EDGES, history.D35_DESIGN_EDGE)
)
@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "wrong_parent",
        "merge",
        "extra_path",
        "missing_path",
        "symlink",
        "blob_pin",
        "patch_pin",
        "missing_verification",
        "inherited_verification",
        "missing_reviewer",
        "substituted_physics_reviewer",
        "substituted_provenance_reviewer",
        "swapped_reviewers",
        "missing_verdict",
        "crossed_review_pins",
        "missing_companion_verdict",
        "inherited_companion_verdict",
        "wrong_round",
        "duplicate_header",
        "preexisting_verification",
    ],
)
def test_source_design_real_objects_require_own_review(
    phase_objects: PhaseObjects,
    monkeypatch: pytest.MonkeyPatch,
    edge: history.SourceDesignEdge,
    mutation: str | None,
) -> None:
    root, commit, base, _, red, _, _, _ = phase_objects
    edits = {
        path: history_git(
            history.REPOSITORY_ROOT, "show", f"{edge.sha}:{path}"
        ).replace(edge.parent.encode(), red.encode())
        for path in history.DESIGN_PATHS
    }
    companion_path, memo_path = history.DESIGN_PATHS
    memo = edits[memo_path]
    if mutation == "missing_verification":
        memo = memo.replace(b"**Review verification", b"**Pending review", 1)
    elif mutation == "inherited_verification":
        marker = f"**Bounded correction #{edge.correction - 1} candidate".encode()
        memo = memo.replace(marker, b"Historical candidate", 1).replace(
            b"**Review verification", marker + b"\n**Review verification", 1
        )
    elif mutation == "missing_reviewer":
        memo = memo.replace(f"`{edge.reviewers[0]}`".encode(), b"`another reviewer`", 1)
    elif mutation == "substituted_physics_reviewer":
        other_reviewer = (
            "/root/d30_physics_review"
            if edge.correction == 34
            else "/root/e_lifecycle_physics"
        )
        memo = memo.replace(
            f"`{edge.reviewers[0]}`".encode(), f"`{other_reviewer}`".encode(), 1
        )
    elif mutation == "substituted_provenance_reviewer":
        memo = memo.replace(
            f"`{edge.reviewers[1]}`".encode(), b"`/root/raw_git_governance`", 1
        )
    elif mutation == "swapped_reviewers":
        memo = memo.replace(
            f"`{edge.reviewers[0]}` and `{edge.reviewers[1]}`".encode(),
            f"`{edge.reviewers[1]}` and `{edge.reviewers[0]}`".encode(),
            1,
        )
    elif mutation == "missing_verdict":
        memo = memo.replace(b"each returned exact\n`ACCEPT`", b"review pending", 1)
    elif mutation == "crossed_review_pins":
        other = next(
            item
            for item in history.SOURCE_DESIGN_EDGES
            if item.correction != edge.correction
        )
        for pin, other_pin in zip(edge.review_pins, other.review_pins, strict=True):
            memo = memo.replace(pin.encode(), other_pin.encode(), 1)
    elif mutation == "missing_companion_verdict":
        edits[companion_path] = edits[companion_path].replace(
            b"returned ACCEPT", b"remain pending", 1
        )
    elif mutation == "inherited_companion_verdict":
        marker = f"**Current continuation #{edge.correction - 1} candidate".encode()
        edits[companion_path] = (
            edits[companion_path]
            .replace(marker, b"Historical continuation", 1)
            .replace(b"physics/governance", marker + b"\nphysics/governance", 1)
        )
    elif mutation == "wrong_round":
        memo = memo.replace(
            f"complete round-{edge.round_number} candidate bytes".encode(),
            b"complete round-99 candidate bytes",
            1,
        )
    elif mutation == "duplicate_header":
        memo += f"\n**Bounded correction #{edge.correction} candidate\n".encode()
    edits[memo_path] = memo
    parent = red
    if mutation == "preexisting_verification":
        parent = commit(red, edits)
        # Correct the new parent join while keeping review/pin lines inherited.
        # Only added-text membership should reject this otherwise valid header.
        edits[memo_path] = edits[memo_path].replace(red.encode(), parent.encode())
        edits = {path: raw + b"\nnew unrelated text\n" for path, raw in edits.items()}
    if mutation == "extra_path":
        edits[history.STATUS_PATH] = b"ungranted ledger companion\n"
    elif mutation == "missing_path":
        del edits[companion_path]
    forged = commit(
        parent,
        edits,
        extra_parent=base if mutation == "merge" else None,
        mode="120000" if mutation == "symlink" else "100644",
    )
    spec = replace(
        edge,
        sha=forged,
        parent=base if mutation == "wrong_parent" else parent,
        blobs=tuple(  # Both landed hashes are repinned to the actual forged tree.
            sha256(history_git(root, "show", f"{forged}:{path}")).hexdigest()
            for path in history.DESIGN_PATHS
        ),
        patch=sha256(
            history_git(root, "diff", "--binary", "--full-index", parent, forged, "--")
        ).hexdigest(),
    )
    if mutation == "blob_pin":
        spec = replace(spec, blobs=("0" * 64, spec.blobs[1]))
    elif mutation == "patch_pin":
        spec = replace(spec, patch="0" * 64)
    if edge.correction == 35:
        monkeypatch.setattr(history, "D35_DESIGN_EDGE", spec)
    else:
        monkeypatch.setattr(history, "SOURCE_DESIGN_EDGES", (spec,))

    def authenticate() -> None:
        if edge.correction == 35:
            history.authenticate_pending_d35(root)
        else:
            history.authenticate_source_design_successor(forged, root)

    if mutation is None:
        authenticate()
    else:
        with pytest.raises(history.HistoryError):
            authenticate()


@pytest.mark.parametrize(
    "order",
    [
        (),
        (32,),
        (33,),
        (34,),
        (33, 32),
        (32, 33),
        (32, 34, 33),
        (34, 33, 32),
        (32, 33, 34),
    ],
)
def test_complete_source_requires_three_designs_in_order(
    phase_objects: PhaseObjects,
    monkeypatch: pytest.MonkeyPatch,
    order: tuple[int, ...],
) -> None:
    root, commit, _, _, _, red, source, _ = phase_objects
    prototypes = {edge.correction: edge for edge in history.SOURCE_DESIGN_EDGES}
    edges: dict[int, history.SourceDesignEdge] = {}
    parent = red
    for correction in order:
        edge = source_design_object(root, commit, parent, prototypes[correction])
        edges[correction] = edge
        parent = edge.sha
    # Keep absent identities pinned too; their objects cannot be invented in S.
    expected = tuple(edges.get(number, prototypes[number]) for number in (32, 33, 34))
    monkeypatch.setattr(history, "SOURCE_DESIGN_EDGES", expected)
    monkeypatch.setattr(history, "HISTORICAL_SOURCE_DESIGN_SHA", expected[0].sha)
    monkeypatch.setattr(history, "HISTORICAL_D33_SOURCE_DESIGN_SHA", expected[1].sha)
    monkeypatch.setattr(history, "SOURCE_DESIGN_SHA", expected[2].sha)
    edits = {
        path: history_git(root, "show", f"{source}:{path}")
        for path in history.SOURCE_PATHS
    }
    if 34 not in order:
        _ = edits.pop(history.FRAME_PATH)
        _ = edits.pop(history.FRAME_TEST_PATH)
    if 34 in order:
        parent = commit(
            parent, {history.FRAME_TEST_PATH: edits.pop(history.FRAME_TEST_PATH)}
        )
    tip = commit(commit(parent, edits), dict.fromkeys(history.DISPOSAL_PINS))
    partial = history.describe_phase_range(
        red, tip, "source", root=root, require_complete=False
    )
    assert [
        row["sha"]
        for row in partial["commits"]
        if row["role"] == "source-design-successor"
    ] == [edges[number].sha for number in order]
    if order == (32, 33, 34):
        assert history.describe_phase_range(red, tip, "source", root=root) == partial
    else:
        with pytest.raises(
            history.HistoryError, match="exact D32 then D33 then D34 once"
        ):
            _ = history.describe_phase_range(red, tip, "source", root=root)


def test_complete_source_inventory_is_exactly_thirteen_modifications_and_four_disposals(
    phase_objects: PhaseObjects,
) -> None:
    expected = {
        history.SOLVER_PATH,
        history.SNAPSHOT_FIXTURE_PATH,
        history.FRAME_PATH,
        history.FRAME_TEST_PATH,
        "src/radiosim/core/result.py",
        "tools/sci004_mmode_phase3_evidence.py",
        "tests/unit/test_sci004_phase3_evidence.py",
        "tools/sci004_mmode_phase3_acceptance.py",
        "tests/unit/test_sci004_phase3_acceptance.py",
        "tools/sci004_phase3_history.py",
        "tests/unit/test_sci004_phase3_history.py",
        "tests/unit/test_sci004_phase3_dependency.py",
        "tests/unit/test_sci004_phase3_red_failures.py",
    }
    assert history.SOURCE_PATHS == expected
    assert len(expected) == 13 and len(history.DISPOSAL_PINS) == 4
    document = _phase_document(phase_objects)
    _validate_document(document, phase_objects)
    aggregate = {
        path
        for row in document["source"]["commits"]
        if row["role"] in {"source", "disposal"}
        for path in row["paths"]
    }
    assert aggregate == expected | history.DISPOSAL_PINS.keys()
    assert not aggregate.intersection(history.DESIGN_PATHS)


@pytest.mark.parametrize("missing", sorted(history.SOURCE_PATHS))
def test_complete_source_refuses_each_missing_modified_path(
    phase_objects: PhaseObjects,
    missing: str,
) -> None:
    root, commit, _, _, _, red, source, _ = phase_objects
    last_design = history.SOURCE_DESIGN_SHA
    edits = {
        path: history_git(root, "show", f"{source}:{path}")
        for path in history.SOURCE_PATHS
        if path != missing
    }
    if history.FRAME_TEST_PATH in edits:
        last_design = commit(
            last_design, {history.FRAME_TEST_PATH: edits.pop(history.FRAME_TEST_PATH)}
        )
    tip = commit(commit(last_design, edits), dict.fromkeys(history.DISPOSAL_PINS))
    with pytest.raises(history.HistoryError):
        _ = history.describe_phase_range(red, tip, "source", root=root)


@pytest.mark.parametrize("mutation", ["unknown_edge", "bad_pin", "wrong_phase"])
def test_source_design_role_authenticates_before_excluding_paths(
    phase_objects: PhaseObjects,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root, commit, _, prerequisite, _, red, _, terminal = phase_objects
    if mutation == "unknown_edge":
        tip = commit(
            terminal, dict.fromkeys(history.DESIGN_PATHS, b"unreviewed successor\n")
        )
        with pytest.raises(history.HistoryError, match="source role"):
            _ = history.describe_phase_range(red, tip, "source", root=root)
    elif mutation == "bad_pin":
        first, second, third = history.SOURCE_DESIGN_EDGES
        monkeypatch.setattr(
            history,
            "SOURCE_DESIGN_EDGES",
            (replace(first, patch="0" * 64), second, third),
        )
        with pytest.raises(history.HistoryError, match="source design complete diff"):
            _ = history.describe_phase_range(red, terminal, "source", root=root)
    else:
        with pytest.raises(history.HistoryError, match="red role"):
            _ = history.describe_phase_range(
                prerequisite, history.HISTORICAL_SOURCE_DESIGN_SHA, "red", root=root
            )


@pytest.mark.parametrize("mutation", ["duplicate", "reverse", "disguise", "historical"])
def test_source_document_cannot_rebind_or_disguise_design_entries(
    phase_objects: PhaseObjects,
    mutation: str,
) -> None:
    document = _phase_document(phase_objects)
    entries = document["source"]["commits"]
    if mutation == "duplicate":
        entries.insert(1, copy.deepcopy(entries[0]))
    elif mutation == "reverse":
        entries[0], entries[1] = entries[1], entries[0]
    elif mutation == "disguise":
        entries[0]["role"] = "status"
    else:
        for design in (
            history.OPERATIVE_DESIGN_SHA,
            history.HISTORICAL_SOURCE_DESIGN_SHA,
        ):
            with pytest.raises(history.HistoryError, match="current D34"):
                history.validate_phase_ranges(
                    document,
                    design_sha=design,
                    red_sha=phase_objects[5],
                    source_sha=phase_objects[7],
                    root=phase_objects[0],
                )
        return
    with pytest.raises(history.HistoryError):
        _validate_document(document, phase_objects)


def test_live_partial_source_preserves_early_s_and_authenticates_current_designs() -> (
    None
):
    red = "567f9ac68730044fc8e887930d3531d794534412"
    first = "72a0a5c5ebd203b63e091342f3655ebf808bac4b"
    early = history.describe_phase_range(red, first, "source", require_complete=False)
    assert [row["role"] for row in early["commits"]] == ["source"]
    head = history_git(history.REPOSITORY_ROOT, "rev-parse", "HEAD").decode().strip()
    current = history.describe_phase_range(red, head, "source", require_complete=False)
    history.require_source_design_successors(current["commits"])
    assert history.OPERATIVE_DESIGN_SHA == history.RED_DESIGN_SHA


def _d34_original_blobs() -> tuple[bytes, bytes]:
    """Read original Git objects; never import or execute the frame module."""
    frame = history_git(
        history.REPOSITORY_ROOT,
        "show",
        "a8a9f53943d7d964f475c376b6ce0dbb9b0157fc:src/radiosim/core/mmode/frame.py",
    )
    workflow = history_git(
        history.REPOSITORY_ROOT,
        "show",
        "90ef12e10c869b0928ad0afd51b9f7069729aa26:.github/workflows/ci.yml",
    )
    return frame, workflow


@pytest.fixture(scope="module")
def d34_guard_blobs() -> tuple[bytes, bytes]:
    return _d34_original_blobs()


def _d34_repaired_frame(original: bytes) -> bytes:
    """Independent literal fixture construction, not the production normalizer."""
    prefix, scanner = original.split(b"def scan_operational_horizon(", 1)
    scanner = scanner.replace(
        b"    horizon_lo, horizon_hi = grid.horizon_domain\n",
        b"""    if not isinstance(frozen_root_bounds, (list, tuple)):
        raise ValueError("frozen root bounds must be a list or tuple")
    for bounds in frozen_root_bounds:
        if not isinstance(bounds, (list, tuple)):
            raise ValueError("each frozen root-bound entry must be a list or tuple")
        for bound in bounds:
            if type(bound) is not Fraction:
                raise ValueError("frozen root-bound endpoints must be exact Fractions")
    horizon_lo, horizon_hi = grid.horizon_domain
""",
        1,
    )
    scanner = scanner.replace(
        b"    shared = sorted(bound for bound in base",
        b"    for bounds in frozen_root_bounds:\n        base.update(bounds)\n"
        b"    shared = sorted(bound for bound in base",
        1,
    )
    return prefix + b"def scan_operational_horizon(" + scanner


@pytest.mark.parametrize("repaired", [False, True])
@pytest.mark.parametrize("required", [None, False, True])
def test_d34_frame_partition_exact_two_states(
    d34_guard_blobs: tuple[bytes, bytes], repaired: bool, required: bool | None
) -> None:
    original, _ = d34_guard_blobs
    candidate = _d34_repaired_frame(original) if repaired else original
    if required is not None and repaired != required:
        with pytest.raises(history.HistoryError, match="required repair presence"):
            _ = history.frame_partition_ast(candidate, original, required)
    else:
        result = history.frame_partition_ast(candidate, original, required)
        assert result == history.frame_partition_ast(original, original, False)
        assert result == history.frame_partition_ast(
            b"# formatting only\n" + candidate + b"\n", original, required
        )


@pytest.mark.parametrize(
    "before,after",
    [
        (
            b"    horizon_lo, horizon_hi =",
            b"    arbitrary_call()\n    horizon_lo, horizon_hi =",
        ),
        (
            b"    horizon_lo, horizon_hi =",
            b"    return None\n    horizon_lo, horizon_hi =",
        ),
        (
            b"    horizon_lo, horizon_hi =",
            b"    import arbitrary_module\n    horizon_lo, horizon_hi =",
        ),
        (
            b"    horizon_lo, horizon_hi =",
            b"    def helper():\n        pass\n    horizon_lo, horizon_hi =",
        ),
        (b"with installed_iers_context():", b"with arbitrary_context():"),
        (
            b"matrix[position] = trajectory.at_common_turn(bound)",
            b"matrix[position] = 0.0",
        ),
        (b"base.update(bounds)", b"base.update(map(float, bounds))"),
        (b"base.update(bounds)", b"bounds.clear()"),
        (b"base.update(bounds)", b"base.update(bounds)\n        arbitrary_call()"),
        (b"horizon_lo <= bound <= horizon_hi", b"horizon_lo < bound < horizon_hi"),
        (b"type(bound) is not Fraction", b"not isinstance(bound, Fraction)"),
        (
            b"isinstance(bounds, (list, tuple))",
            b"isinstance(bounds, (list, tuple, str))",
        ),
        (
            b'raise ValueError("frozen root bounds',
            b'raise TypeError("frozen root bounds',
        ),
        (
            b"one frozen root-bound list is required per direction",
            b"changed cardinality",
        ),
        (b"direction_ids: Sequence[str]", b"direction_ids: Sequence[int]"),
        (b"SCAN_DERIVATIVE_CEILING_PER_TURN * widths", b"0.0 * widths"),
        (b"base.add(horizon_hi)", b"base.add(float(horizon_hi))"),
        (b"    for bounds in frozen_root_bounds:\n        base.update(bounds)\n", b""),
    ],
)
def test_d34_frame_partition_rejects_scanner_escape(
    d34_guard_blobs: tuple[bytes, bytes], before: bytes, after: bytes
) -> None:
    original, _ = d34_guard_blobs
    prefix, scanner = _d34_repaired_frame(original).split(
        b"def scan_operational_horizon(", 1
    )
    assert before in scanner
    candidate = (
        prefix + b"def scan_operational_horizon(" + scanner.replace(before, after, 1)
    )
    with pytest.raises(history.HistoryError, match="unapproved AST"):
        _ = history.frame_partition_ast(candidate, original, None)


@pytest.mark.parametrize(
    "suffix", [b"\ndef helper():\n    pass\n", b"\nNEW_CONSTANT = 1\n"]
)
def test_d34_frame_partition_preserves_whole_module(
    d34_guard_blobs: tuple[bytes, bytes], suffix: bytes
) -> None:
    original, _ = d34_guard_blobs
    with pytest.raises(history.HistoryError, match="unapproved AST"):
        _ = history.frame_partition_ast(
            _d34_repaired_frame(original) + suffix, original, True
        )


def test_d34_frame_partition_requires_pinned_baseline_and_valid_python(
    d34_guard_blobs: tuple[bytes, bytes],
) -> None:
    original, _ = d34_guard_blobs
    with pytest.raises(history.HistoryError, match="original blob"):
        _ = history.frame_partition_ast(original, original + b"\n", None)
    with pytest.raises(history.HistoryError, match="valid Python AST"):
        _ = history.frame_partition_ast(b"def broken(", original, None)


def test_d34_workflow_exact_raw_replacement(
    d34_guard_blobs: tuple[bytes, bytes],
) -> None:
    _, before = d34_guard_blobs
    after = before.replace(
        b"    timeout-minutes: 45\n", b"    timeout-minutes: 120\n", 1
    )
    history.validate_verification_workflow_bytes(before, after)
    assert (
        after.replace(b"    timeout-minutes: 120\n", b"    timeout-minutes: 45\n", 1)
        == before
    )


@pytest.mark.parametrize(
    "old,new",
    [
        (b"timeout-minutes: 120", b"timeout-minutes: 45"),
        (b"timeout-minutes: 120", b"timeout-minutes: 119"),
        (b"timeout-minutes: 120", b'timeout-minutes: "120"'),
        (b"timeout-minutes: 120", b"timeout-minutes: 120\n    timeout-minutes: 120"),
        (b"timeout-minutes: 30", b"timeout-minutes: 120"),
        (b"runs-on: ubuntu-24.04", b"runs-on: ubuntu-latest"),
        (b"contents: read", b"contents: write"),
        (b"# CI-001", b"# changed"),
        (b"\n", b"\r\n"),
    ],
)
def test_d34_workflow_rejects_any_extra_bytes(
    d34_guard_blobs: tuple[bytes, bytes], old: bytes, new: bytes
) -> None:
    _, before = d34_guard_blobs
    allowed = before.replace(
        b"    timeout-minutes: 45\n", b"    timeout-minutes: 120\n", 1
    )
    assert old in allowed
    with pytest.raises(history.HistoryError, match="exact scalar replacement"):
        history.validate_verification_workflow_bytes(
            before, allowed.replace(old, new, 1)
        )


@pytest.mark.parametrize("mode", ["alternate", "reverse", "repeat"])
def test_d34_workflow_rejects_unpinned_or_repeated_baseline(
    d34_guard_blobs: tuple[bytes, bytes], mode: str
) -> None:
    _, original = d34_guard_blobs
    allowed = original.replace(
        b"    timeout-minutes: 45\n", b"    timeout-minutes: 120\n", 1
    )
    before = original + b"\n" if mode == "alternate" else allowed
    after = before.replace(
        b"    timeout-minutes: 45\n", b"    timeout-minutes: 120\n", 1
    )
    if mode == "reverse":
        after = original
    with pytest.raises(history.HistoryError, match="original blob"):
        history.validate_verification_workflow_bytes(before, after)


@pytest.mark.parametrize(
    "old,new",
    [
        (
            b"SCAN_DERIVATIVE_CEILING_PER_TURN: Final[float] = 6.2895",
            b"SCAN_DERIVATIVE_CEILING_PER_TURN: Final[float] = 6.3",
        ),
        (b"cosine = math.cos(theta)", b"cosine = math.sin(theta)"),
    ],
)
def test_d34_frame_partition_preserves_existing_constant_and_other_function(
    d34_guard_blobs: tuple[bytes, bytes], old: bytes, new: bytes
) -> None:
    original, _ = d34_guard_blobs
    repaired = _d34_repaired_frame(original)
    assert old in repaired
    with pytest.raises(history.HistoryError, match="unapproved AST"):
        _ = history.frame_partition_ast(repaired.replace(old, new, 1), original, None)


@pytest.mark.parametrize("present", [False, True])
def test_d34_complete_source_records_optional_workflow(
    phase_objects: PhaseObjects, present: bool
) -> None:
    root, commit, _, _, _, red, source, terminal = phase_objects
    if present:
        before = history_git(root, "show", f"{source}:{history.WORKFLOW_PATH}")
        workflow = commit(
            source,
            {
                history.WORKFLOW_PATH: before.replace(
                    b"timeout-minutes: 45", b"timeout-minutes: 120", 1
                )
            },
        )
        partial = history.describe_phase_range(
            red, workflow, "source", root=root, require_complete=False
        )
        assert partial["commits"][-1]["role"] == "verification-workflow"
        terminal = commit(workflow, dict.fromkeys(history.DISPOSAL_PINS))
    described = history.describe_phase_range(red, terminal, "source", root=root)
    workflows = [
        row for row in described["commits"] if row["role"] == "verification-workflow"
    ]
    assert len(workflows) == int(present)
    aggregate = {
        path
        for row in described["commits"]
        if row["role"] not in {"source-design-successor", "status"}
        for path in row["paths"]
    }
    assert len(aggregate) == 17 + int(present)
    assert history.WORKFLOW_PATH not in history.SOURCE_PATHS


@pytest.mark.parametrize(
    "path", [history.FRAME_PATH, history.FRAME_TEST_PATH, history.WORKFLOW_PATH]
)
def test_d34_new_path_grants_require_prior_design(
    phase_objects: PhaseObjects, path: str
) -> None:
    root, commit, _, _, _, red, source, _ = phase_objects
    raw = history_git(root, "show", f"{source}:{path}")
    if path == history.WORKFLOW_PATH:
        raw = raw.replace(b"timeout-minutes: 45", b"timeout-minutes: 120", 1)
    before_design = history.HISTORICAL_D33_SOURCE_DESIGN_SHA
    tip = commit(before_design, {path: raw})
    with pytest.raises(history.HistoryError, match="D34 must precede"):
        _ = history.describe_phase_range(
            red, tip, "source", root=root, require_complete=False
        )


@pytest.mark.parametrize("revert", [False, True])
def test_d34_range_rejects_intermediate_forbidden_frame_ast(
    phase_objects: PhaseObjects, revert: bool
) -> None:
    root, commit, _, _, _, red, source, _ = phase_objects
    original = history_git(root, "show", f"{source}:{history.FRAME_PATH}")
    forbidden = original.replace(
        b"SCAN_ROOT_RESIDUAL: Final[float] = 5e-12",
        b"SCAN_ROOT_RESIDUAL: Final[float] = 6e-12",
    )
    assert forbidden != original
    tip = commit(source, {history.FRAME_PATH: forbidden})
    if revert:
        tip = commit(tip, {history.FRAME_PATH: original})
    with pytest.raises(history.HistoryError, match="unapproved AST"):
        _ = history.describe_phase_range(
            red, tip, "source", root=root, require_complete=False
        )


@pytest.mark.parametrize(
    "attack",
    [
        "wrong_value",
        "extra_path",
        "symlink",
        "executable",
        "merge",
        "revert",
        "terminal",
        "disguised",
    ],
)
def test_d34_workflow_range_rejects_invalid_entries(
    phase_objects: PhaseObjects, attack: str
) -> None:
    root, commit, base, _, _, red, source, terminal = phase_objects
    parent = terminal if attack == "terminal" else source
    before = history_git(root, "show", f"{parent}:{history.WORKFLOW_PATH}")
    after = before.replace(b"timeout-minutes: 45", b"timeout-minutes: 120", 1)
    if attack == "wrong_value":
        after = after.replace(b"timeout-minutes: 120", b"timeout-minutes: 119", 1)
    edits = {history.WORKFLOW_PATH: after}
    if attack == "extra_path":
        edits[history.STATUS_PATH] = b"extra status\n"
    tip = commit(
        parent,
        edits,
        mode={"symlink": "120000", "executable": "100755"}.get(attack, "100644"),
        extra_parent=base if attack == "merge" else None,
    )
    if attack == "revert":
        tip = commit(tip, {history.WORKFLOW_PATH: before})
    if attack == "disguised":
        end = commit(tip, dict.fromkeys(history.DISPOSAL_PINS))
        objects = (*phase_objects[:-1], end)
        document = _phase_document(objects)
        entry = next(
            row
            for row in document["source"]["commits"]
            if row["role"] == "verification-workflow"
        )
        entry["role"] = "status"
        with pytest.raises(history.HistoryError):
            _validate_document(document, objects)
    else:
        with pytest.raises(history.HistoryError):
            _ = history.describe_phase_range(
                red, tip, "source", root=root, require_complete=attack == "terminal"
            )


@pytest.mark.parametrize(
    "placement", ["absent", "same_commit", "separate", "before_design"]
)
def test_d34_frame_repair_requires_separate_regression(
    phase_objects: PhaseObjects, placement: str
) -> None:
    root, commit, _, _, _, red, source, _ = phase_objects
    parent = history.SOURCE_DESIGN_SHA
    frame = history_git(root, "show", f"{source}:{history.FRAME_PATH}")
    edits = {history.FRAME_PATH: frame}
    if placement == "same_commit":
        edits[history.FRAME_TEST_PATH] = b"# regression\n"
    elif placement == "separate":
        parent = commit(parent, {history.FRAME_TEST_PATH: b"# regression\n"})
    elif placement == "before_design":
        before = commit(
            history.HISTORICAL_D33_SOURCE_DESIGN_SHA,
            {history.FRAME_TEST_PATH: b"# regression\n"},
        )
        with pytest.raises(history.HistoryError, match="D34 must precede"):
            _ = history.describe_phase_range(
                red, before, "source", root=root, require_complete=False
            )
        return
    tip = commit(parent, edits)
    if placement == "separate":
        # The full range and a supported suffix authenticate the predecessor.
        for base in (red, parent):
            result = history.describe_phase_range(
                base, tip, "source", root=root, require_complete=False
            )
            assert result["commits"][-1]["sha"] == tip
    else:
        with pytest.raises(history.HistoryError, match="separate prior post-D34"):
            _ = history.describe_phase_range(
                red, tip, "source", root=root, require_complete=False
            )


@pytest.mark.parametrize("attack", ["symlink", "merge", "non_source"])
def test_d34_seeded_regression_authenticates_its_source_prefix(
    phase_objects: PhaseObjects, attack: str
) -> None:
    root, commit, base, _, _, _, source, _ = phase_objects
    edits = {history.FRAME_TEST_PATH: b"# regression\n"}
    if attack == "non_source":
        edits["ungranted.txt"] = b"ungranted\n"
    predecessor = commit(
        history.SOURCE_DESIGN_SHA,
        edits,
        mode="120000" if attack == "symlink" else "100644",
        extra_parent=base if attack == "merge" else None,
    )
    tip = commit(
        predecessor,
        {
            history.FRAME_PATH: history_git(
                root, "show", f"{source}:{history.FRAME_PATH}"
            )
        },
    )
    with pytest.raises(history.HistoryError):
        _ = history.describe_phase_range(
            predecessor, tip, "source", root=root, require_complete=False
        )


def test_pending_d35_authentication_does_not_activate_source_authority() -> None:
    """The real reviewed landing authenticates only through its pending entry."""
    history.authenticate_pending_d35()
    assert history.D35_DESIGN_EDGE.sha == "dd26e045897274f4b09de38ad4552904cc3c33ab"
    assert history.HISTORICAL_D34_SOURCE_DESIGN_SHA == history.D34_DESIGN_EDGE.sha
    assert history.SOURCE_DESIGN_SHA == history.HISTORICAL_D34_SOURCE_DESIGN_SHA
    assert [edge.correction for edge in history.SOURCE_DESIGN_EDGES] == [32, 33, 34]
    with pytest.raises(history.HistoryError, match="unknown source design successor"):
        history.authenticate_source_design_successor(history.D35_DESIGN_EDGE.sha)
    records = [
        {"sha": edge.sha, "role": "source-design-successor"}
        for edge in history.SOURCE_DESIGN_EDGES
    ]
    history.require_source_design_successors(records)
    with pytest.raises(history.HistoryError, match="exact D32 then D33 then D34 once"):
        history.require_source_design_successors(
            [
                *records,
                {"sha": history.D35_DESIGN_EDGE.sha, "role": "source-design-successor"},
            ]
        )


@pytest.mark.parametrize(
    "path", [history.FRAME_PATH, history.FRAME_TEST_PATH, history.WORKFLOW_PATH]
)
def test_historical_d34_roles_ignore_current_authority_change(
    phase_objects: PhaseObjects, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """Changing current authority cannot move D34's baseline or prefix boundary."""
    root, commit, _, _, _, _, source, _ = phase_objects
    parent = history.HISTORICAL_D34_SOURCE_DESIGN_SHA
    if path == history.FRAME_PATH:
        parent = commit(parent, {history.FRAME_TEST_PATH: b"# owned D34 regression\n"})
        raw = history_git(root, "show", f"{source}:{path}")
    elif path == history.FRAME_TEST_PATH:
        raw = b"# owned D34 regression\n"
    else:
        raw = history_git(root, "show", f"{parent}:{path}").replace(
            b"timeout-minutes: 45", b"timeout-minutes: 120", 1
        )
    tip = commit(parent, {path: raw})
    monkeypatch.setattr(history, "SOURCE_DESIGN_SHA", "f" * 40)
    result = history.describe_phase_range(
        parent, tip, "source", root=root, require_complete=False
    )
    assert result["commits"][-1]["sha"] == tip
    assert result["commits"][-1]["role"] == (
        "verification-workflow" if path == history.WORKFLOW_PATH else "source"
    )


@pytest.fixture
def geometry_profiles(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Independent tiny full modules exercise the guard, not solver physics."""
    import ast
    import sys

    states = {
        "O0": b"bridge = False\ngeometry = 'old'\n",
        "O1": b"bridge = True\ngeometry = 'old'\n",
        "G1": b"bridge = True\ngeometry = 'repaired'\n",
    }
    profile = {
        name: sha256(
            ast.dump(
                ast.parse(raw), annotate_fields=True, include_attributes=False
            ).encode()
        ).hexdigest()
        for name, raw in states.items()
    }
    monkeypatch.setattr(
        history,
        "D35_SOLVER_AST_PROFILES",
        {(sys.implementation.name, *sys.version_info[:2]): profile},
    )
    return states


@pytest.mark.parametrize("before", ["O0", "O1", "G1"])
@pytest.mark.parametrize("after", ["O0", "O1", "G1"])
@pytest.mark.parametrize("cumulative", [False, True])
def test_geometry_guard_directed_states(
    geometry_profiles: dict[str, bytes], before: str, after: str, cumulative: bool
) -> None:
    if cumulative:
        permitted = before == "O0" and after == "G1"
    else:
        permitted = (
            (before == "O0" and after == "O1")
            or (before in {"O1", "G1"} and after == "G1")
            or before == after == "O1"
        )
    arguments = (geometry_profiles[before], geometry_profiles[after])
    if permitted:
        assert history.validate_solver_geometry_transition(
            *arguments, cumulative=cumulative
        ) == (before, after)
    else:
        with pytest.raises(history.HistoryError, match="forbidden directed states"):
            _ = history.validate_solver_geometry_transition(
                *arguments, cumulative=cumulative
            )


@pytest.mark.parametrize(
    "raw",
    [
        b"bridge = False\ngeometry = 'repaired'\n",  # G0 is never authorized.
        b"bridge = True\ngeometry = 'foreign'\n",
        b"bridge = True\ngeometry = 'old'\nextra = 1\n",
        b"bridge = True\ngeometry = 'old'\ndef helper(): return 1\n",
    ],
)
def test_geometry_guard_rejects_complete_unknown_modules(
    geometry_profiles: dict[str, bytes], raw: bytes
) -> None:
    with pytest.raises(history.HistoryError, match="unapproved whole-module state"):
        _ = history.solver_geometry_state(raw)
    # An approved eventual endpoint cannot conceal the unapproved middle state.
    with pytest.raises(history.HistoryError, match="unapproved whole-module state"):
        _ = history.validate_solver_geometry_transition(raw, geometry_profiles["G1"])


@pytest.mark.parametrize("raw", [b"def broken(", b"\x00", b"\xff"])
def test_geometry_guard_rejects_invalid_python(raw: bytes) -> None:
    with pytest.raises(history.HistoryError, match="requires valid Python"):
        _ = history.solver_geometry_state(raw)


def test_geometry_guard_layout_runtime_and_mode_boundaries(
    geometry_profiles: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        history.solver_geometry_state(
            b"# comments are not AST statements\n\n" + geometry_profiles["G1"]
        )
        == "G1"
    )
    with pytest.raises(history.HistoryError, match="requires boolean mode"):
        _ = history.validate_solver_geometry_transition(
            geometry_profiles["O0"], geometry_profiles["G1"], cumulative=cast(bool, 1)
        )
    monkeypatch.setattr(history.sys, "version_info", (3, 14, 0, "final", 0))
    with pytest.raises(history.HistoryError, match="unsupported runtime"):
        _ = history.solver_geometry_state(geometry_profiles["O1"])


def test_geometry_guard_actual_prefix_and_observed_source() -> None:
    """Committed old states stay valid prefixes; only repaired source can finish."""
    old = history_git(
        history.REPOSITORY_ROOT,
        "show",
        "567f9ac68730044fc8e887930d3531d794534412:" + history.SOLVER_PATH,
    )
    preceding = history_git(
        history.REPOSITORY_ROOT,
        "show",
        history.D35_DESIGN_EDGE.sha + ":" + history.SOLVER_PATH,
    )
    assert history.solver_geometry_state(old) == "O0"
    assert history.solver_geometry_state(preceding) == "O1"
    assert history.validate_solver_geometry_transition(old, preceding) == ("O0", "O1")
    with pytest.raises(history.HistoryError, match="forbidden directed states"):
        _ = history.validate_solver_geometry_transition(old, preceding, cumulative=True)
    observed = (history.REPOSITORY_ROOT / history.SOLVER_PATH).read_bytes()
    state = history.solver_geometry_state(observed)
    assert state in {"O1", "G1"}
    if state == "G1":
        assert history.validate_solver_geometry_transition(
            old, observed, cumulative=True
        ) == ("O0", "G1")
    else:
        with pytest.raises(history.HistoryError, match="forbidden directed states"):
            _ = history.validate_solver_geometry_transition(
                old, observed, cumulative=True
            )
