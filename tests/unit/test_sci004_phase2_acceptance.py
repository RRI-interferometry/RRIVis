"""Strict authentication of the SCI-004 phase-M2 acceptance record.

``docs/development/sci004_mmode_design.md`` Sections 13.4, 14.3 and 14.4 freeze
this module's successor authority: it lands in ``S2`` with both approved
constants as the literal ``None``, the official acceptance path **absent**, and
every synthetic strict schema fixture passing.  ``A2`` then changes *only* the
two constants below, from ``None`` to the exact lower-case 40- and
64-hexadecimal literals, and adds the acceptance JSON plus the authorized status
prose.  No import, expression, annotation, key, surrounding token, or other
literal in either assignment may change, so this module's own token stream
outside those two spans is comparable to its direct-parent ``E2`` bytes.

In the pre-``A2`` state the null constants require that JSON to be absent while
the synthetic schema tests pass.  In the ``A2`` state the active validator
authenticates the approved ``E2``, the raw acceptance bytes, the unique
introducing ``A2`` commit and the exact ``E2..A2`` authority; it never requires
the evidence artifact's ``source_sha`` to equal ``E2``.

Importing this module loads only the Python standard library plus ``pytest``,
following ``tools/sci004_mmode_phase1_acceptance.py``: an acceptance-critical
validator must not depend on a package that is merely transitively present.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import sys
import tokenize
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Section 14.3's two approved constants.  ``A2`` replaces exactly these two
#: ``None`` literals and nothing else in this module.
APPROVED_EVIDENCE_SHA: str | None = None
APPROVED_ACCEPTANCE_ARTIFACT_SHA256: str | None = None

TOOL = "tools/sci004_mmode_phase2_acceptance.py"
ARTIFACT = "docs/development/sci004_mmode_phase2_acceptance.json"
EVIDENCE_ARTIFACT = "docs/development/sci004_mmode_phase2_evidence.json"
VALIDATOR = "tests/unit/test_sci004_phase2_acceptance.py"

#: Section 13.4's complete ``A2`` write authority.  The commit that introduces
#: the acceptance record may touch these paths and nothing else.
A2_AUTHORIZED_PATHS: frozenset[str] = frozenset(
    {
        ARTIFACT,
        VALIDATOR,
        "docs/development/sci004_mmode_design.md",
        "PostTier8RemediationPlan.md",
    }
)

#: The two spans Section 13.4 lets ``A2`` rewrite inside this module.
APPROVED_CONSTANT_NAMES: tuple[str, ...] = (
    "APPROVED_EVIDENCE_SHA",
    "APPROVED_ACCEPTANCE_ARTIFACT_SHA256",
)

GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

ACCEPTANCE_SCHEMA = "radiosim.sci004.mmode-phase2-acceptance.v1"
SELF_REFERENCE_REASON = "self-reference: the next R or C binds the containing A commit"

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

#: Section 14.3's required ``A2`` re-derivation identifiers, in the order its
#: sentence names them.  The generator's module docstring carries the verbatim
#: clause-to-identifier mapping; this array is the independent restatement of
#: it, written here rather than imported so the validator enforces the design's
#: inventory instead of the producer's opinion of it.
REQUIRED_ORACLES: tuple[str, ...] = (
    "m2.v-bridge-north-east-to-theta-phi",
    "m2.polarized-blm-equation",
    "m2.horizon-split-exposure",
    "m2.phase-local-frame-certificate-and-direct-error-cubes",
    "m2.two-tier-gate-predicates",
    "m2.transfer-grid-catalogue-and-joins",
    "m2.direct-transfer-sample-shell-block-coverage",
    "m2.local-shell-diagnostics",
    "m2.backend-and-memory-predicates",
    "m2.tier7-capability-flip",
)


def _tool() -> Any:
    """Import the tracked generator without adding an import-time dependency."""
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    try:
        import sci004_mmode_phase2_acceptance as module
    finally:
        sys.path.pop(0)
    return module


def _synthetic_record(verdict: str = "ACCEPT") -> dict[str, Any]:
    """Return a complete synthetic record satisfying every Section 14.3 rule."""
    forty = "0" * 39 + "1"
    other_forty = "0" * 39 + "2"
    sixty_four = "0" * 63 + "1"
    oracles = [
        {
            "oracle_id": name,
            "method": "independent re-derivation, dimensionless residual",
            "observed": 0.0,
            "fixed_limit": 5e-12,
            "pass": True,
        }
        for name in REQUIRED_ORACLES
    ]
    return {
        "schema_version": ACCEPTANCE_SCHEMA,
        "phase": "M2",
        "verdict": verdict,
        "generated_at_utc": "2026-08-24T00:00:00Z",
        "reviewer_identity": "sci004-a2-independent-reviewer",
        "reviewer_independent": True,
        # Section 13.7 supersedes the design between ``R`` and ``S``, so the
        # operative ``design_sha`` differs from the red record's frozen binding.
        # The synthetic fixture uses two distinct values deliberately.
        "design_sha": forty,
        "red_commit_sha": other_forty,
        "source_sha": forty,
        "evidence_commit_sha": forty,
        "evidence_artifact_path": EVIDENCE_ARTIFACT,
        "evidence_artifact_sha256": sixty_four,
        "acceptance_commit_sha": None,
        "acceptance_commit_sha_reason": SELF_REFERENCE_REASON,
        "reviewed_artifacts": [
            {
                "path": EVIDENCE_ARTIFACT,
                "sha256": sixty_four,
                "source_sha": forty,
                "authenticated": True,
            }
        ],
        "rederived_oracles": oracles,
        "commands": [
            {
                "argv": ["pixi", "run", "test", "--", "-m", "not slow"],
                "cwd": ".",
                "pixi_environment": "default",
                "started_at_utc": "2026-08-24T00:00:00Z",
                "duration_seconds": 1.0,
                "exit_code": 0,
                "stdout_sha256": sixty_four,
                "stderr_sha256": sixty_four,
            }
        ],
        "blockers": []
        if verdict == "ACCEPT"
        else [
            {
                "blocker_id": "b1",
                "requirement_id": "sci004.section-6.polarized-kernel",
                "evidence": "the polarized transfer did not reproduce",
                "required_remediation": "recompute and re-review",
            }
        ],
        "accepted_limitations": [
            "phase M2 makes no fingerprint, speed or accelerator claim",
        ],
        "claims_not_licensed": [
            "general_speedup",
            "gpu_or_accelerator_support",
            "retained_fingerprint_pins",
        ],
    }


# ---------------------------------------------------------------------------
# Pre-A2 state
# ---------------------------------------------------------------------------


def test_the_approved_constants_are_null_sentinels_before_a2() -> None:
    """Section 14.3: at ``S2``/``E2`` both approved digests are ``None``."""
    if APPROVED_EVIDENCE_SHA is None or APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None:
        assert APPROVED_EVIDENCE_SHA is None
        assert APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None
        return
    assert GIT_SHA.fullmatch(APPROVED_EVIDENCE_SHA)
    assert SHA256.fullmatch(APPROVED_ACCEPTANCE_ARTIFACT_SHA256)


def test_the_official_acceptance_artifact_is_absent_before_a2() -> None:
    """Section 14.3: null constants require the acceptance JSON to be absent."""
    if APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is not None:
        return
    assert not (REPOSITORY_ROOT / ARTIFACT).exists()


def test_the_acceptance_generator_is_already_tracked_at_s2() -> None:
    """Section 14.3: the generator and validator are already tracked at ``S``."""
    assert (REPOSITORY_ROOT / TOOL).is_file()


def test_the_generator_imports_only_the_standard_library() -> None:
    """An acceptance-critical verifier carries no transitive package dependency."""
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    for forbidden in ("import numpy", "import astropy", "import pytest", "import yaml"):
        assert forbidden not in source, forbidden


def test_the_generator_refuses_before_the_evidence_commit_exists() -> None:
    """Section 14.3: the generator runs only from a globally clean exact ``E2``.

    At ``S2`` the phase evidence artifact does not exist yet, so the refusal
    names that; at ``E2`` the preflight passes and the empty review record is
    refused as a malformed argument; at ``A2`` and beyond the declared
    acceptance output already exists, so the no-overwrite rule refuses first;
    and a dirty tree refuses before any of those.  In every state the assertion
    names a *reason* rather than accepting any non-zero exit, which a generator
    that refused unconditionally would also produce, and the process always
    fails closed with a frozen prefix rather than a traceback.  The refusing run
    must open no output: absent before ``A2``, the acceptance artifact stays
    absent, and at ``A2`` where it legitimately exists it stays byte-identical.
    """
    module = _tool()
    artifact = REPOSITORY_ROOT / ARTIFACT
    before = artifact.read_bytes() if artifact.exists() else None
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / TOOL),
            "generate",
            "--review-record",
            "/dev/null",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert completed.stdout == ""
    prefixes = (
        module.ANCESTRY + ": ",
        module.DIGEST + ": ",
        module.ARGUMENT + ": ",
    )
    assert completed.stderr.startswith(prefixes)
    assert "Traceback" not in completed.stderr
    after = artifact.read_bytes() if artifact.exists() else None
    assert after == before
    reasons = (
        "not globally clean",
        "commit that adds the phase evidence artifact",
        "is not UTF-8 JSON",
        "already exists",
    )
    assert any(reason in completed.stderr for reason in reasons)


def test_the_generator_produces_at_a_clean_evidence_commit() -> None:
    """Section 14.3/14.4: ``generate`` is bound to a venue, not prohibited.

    The complement of the refusal above is pinned in the tracked bytes: after a
    passing preflight the sub-command loads the review record, derives the
    record, validates it, and publishes it by atomic no-overwrite rename, with
    no unconditional post-preflight refusal.
    """
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    body = source[source.index('if arguments.command == "generate":') :]
    body = body[: body.index("document = json.loads(")]
    assert "load_review_record(" in body
    assert "build_acceptance_document(state, review)" in body
    assert "validate_acceptance_document(document)" in body
    assert "write_atomic_no_overwrite(" in body
    assert "raise AcceptanceError(" not in body


# ---------------------------------------------------------------------------
# Synthetic strict schema fixtures
# ---------------------------------------------------------------------------


def test_the_synthetic_accept_record_satisfies_every_rule() -> None:
    """The fixture is a positive control for the rejections below."""
    module = _tool()
    record = module.validate_acceptance_document(_synthetic_record())
    assert set(record) == set(ACCEPTANCE_KEYS)


def test_the_synthetic_reject_record_satisfies_every_rule() -> None:
    """``REJECT`` is a first-class verdict with at least one concrete blocker."""
    module = _tool()
    record = module.validate_acceptance_document(_synthetic_record("REJECT"))
    assert record["verdict"] == "REJECT"
    assert len(record["blockers"]) == 1


@pytest.mark.parametrize("key", ACCEPTANCE_KEYS)
def test_a_missing_top_level_key_is_rejected(key: str) -> None:
    """Section 14: every object rejects unknown or missing keys."""
    module = _tool()
    document = _synthetic_record()
    del document[key]
    with pytest.raises(module.AcceptanceError):
        module.validate_acceptance_document(document)


def test_an_unknown_top_level_key_is_rejected() -> None:
    """An extra key is a different record, not a superset of this one."""
    module = _tool()
    document = _synthetic_record()
    document["extra"] = 1
    with pytest.raises(module.AcceptanceError):
        module.validate_acceptance_document(document)


def test_accept_with_a_blocker_is_rejected() -> None:
    """Section 14.3: ``ACCEPT`` requires an empty ``blockers`` array."""
    module = _tool()
    document = _synthetic_record()
    document["blockers"] = _synthetic_record("REJECT")["blockers"]
    with pytest.raises(module.AcceptanceError):
        module.validate_acceptance_document(document)


def test_reject_without_a_blocker_is_rejected() -> None:
    """Section 14.3: ``REJECT`` requires at least one concrete blocker."""
    module = _tool()
    document = _synthetic_record("REJECT")
    document["blockers"] = []
    with pytest.raises(module.AcceptanceError):
        module.validate_acceptance_document(document)


def test_accept_with_a_false_oracle_is_rejected() -> None:
    """Section 14.3: ``ACCEPT`` requires no false oracle."""
    module = _tool()
    document = _synthetic_record()
    document["rederived_oracles"][0]["pass"] = False
    with pytest.raises(module.AcceptanceError):
        module.validate_acceptance_document(document)


@pytest.mark.parametrize("oracle_id", REQUIRED_ORACLES)
def test_accept_missing_a_required_oracle_is_rejected(oracle_id: str) -> None:
    """The ``A2`` identifiers are required, not optional reviewer prose."""
    module = _tool()
    document = _synthetic_record()
    document["rederived_oracles"] = [
        row for row in document["rederived_oracles"] if row["oracle_id"] != oracle_id
    ]
    with pytest.raises(module.AcceptanceError):
        module.validate_acceptance_document(document)


def test_the_generator_and_validator_declare_the_same_ten_oracles() -> None:
    """The design's inventory is one list, written twice and compared.

    The validator restates Section 14.3's identifiers rather than importing
    them, so this is the join that makes the restatement load-bearing: a
    generator that quietly dropped, renamed or added an identifier would pass
    its own checks and fail here.
    """
    module = _tool()
    assert tuple(module.REQUIRED_ORACLES) == REQUIRED_ORACLES
    assert len(REQUIRED_ORACLES) == 10
    assert len(set(REQUIRED_ORACLES)) == 10
    assert all(name.startswith("m2.") for name in REQUIRED_ORACLES)


def test_the_generator_documents_the_verbatim_a2_clause_mapping() -> None:
    """Section 14.3's ``A2`` sentence maps 1:1 onto the ten identifiers.

    The mapping is prose in the generator's docstring, so it is pinned here
    rather than merely written: every identifier appears in the table, the two
    rescoped clauses are quoted, and the transport-sign clause is present as an
    explicit *exclusion* rather than as an eleventh oracle.
    """
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    docstring = source[: source.index('"""', source.index('"""') + 3)]
    for name in REQUIRED_ORACLES:
        assert f"``{name}``" in docstring, name
    assert '"the North/East-to-theta/phi V bridge"' in docstring
    assert '"one polarized ``B_lm`` equation"' in docstring
    assert "no mount tangent rotation" in docstring
    assert "same celestial tangent basis as the sky expansion" in docstring
    assert "transport-sign obligation binds only" in docstring


def test_the_generator_does_not_equate_the_red_and_operative_design_sha() -> None:
    """Section 13.7: ``R2``'s frozen ``design_sha`` is superseded at ``S2``.

    A bounded correction between ``R`` and ``S`` moves the operative ``D``, so
    the red record's binding and the evidence's binding legitimately differ. A
    generator that compared them would refuse exactly the phases Section 13.7
    exists to permit, so the absence of that comparison is pinned here and the
    reason is required to be written down.
    """
    module = _tool()
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    assert "superseded" in source
    assert "expected to differ" in source
    document = _synthetic_record()
    assert document["design_sha"] != document["red_commit_sha"]
    module.validate_acceptance_document(document)


def test_a_dependent_reviewer_is_rejected() -> None:
    """``reviewer_independent`` must be true."""
    module = _tool()
    document = _synthetic_record()
    document["reviewer_independent"] = False
    with pytest.raises(module.AcceptanceError):
        module.validate_acceptance_document(document)


def test_a_non_null_acceptance_commit_sha_is_rejected() -> None:
    """Section 14.4: ``A`` artifacts use a null self SHA."""
    module = _tool()
    document = _synthetic_record()
    document["acceptance_commit_sha"] = "0" * 40
    with pytest.raises(module.AcceptanceError):
        module.validate_acceptance_document(document)


def test_a_non_finite_oracle_measurement_is_rejected() -> None:
    """``observed`` and ``fixed_limit`` are finite numbers in the oracle's units."""
    module = _tool()
    document = _synthetic_record()
    document["rederived_oracles"][0]["observed"] = float("inf")
    with pytest.raises(module.AcceptanceError):
        module.validate_acceptance_document(document)


def test_unsorted_claim_arrays_are_rejected() -> None:
    """``accepted_limitations`` and ``claims_not_licensed`` are sorted and unique."""
    module = _tool()
    document = _synthetic_record()
    document["claims_not_licensed"] = ["gpu_or_accelerator_support", "general_speedup"]
    with pytest.raises(module.AcceptanceError):
        module.validate_acceptance_document(document)


def test_a_review_record_supplying_a_derived_field_is_rejected(tmp_path) -> None:
    """Section 14.3: the reviewer supplies a verdict, not a derivation."""
    module = _tool()
    record = {
        "reviewer_identity": "sci004-a2-independent-reviewer",
        "reviewer_independent": True,
        "verdict": "ACCEPT",
        "rederived_oracles": [],
        "blockers": [],
        "accepted_limitations": [],
        "claims_not_licensed": [],
        "source_sha": "0" * 40,
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(module.AcceptanceError):
        module.load_review_record(path)


def test_canonical_json_matches_the_evidence_tool_spelling() -> None:
    """Both SCI-004 phase-2 tools serialize identically; a divergence is a fork."""
    module = _tool()
    assert module.canonical_json({"b": 1.0, "a": [1e-7]}) == b'{"a":[1e-7],"b":1}'


# ---------------------------------------------------------------------------
# A2 state: authenticate the retained record and its introducing commit
# ---------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    """Return the stdout of one hermetic ``git`` invocation in this repository.

    The validator carries no package dependency, so ancestry facts are read from
    ``git`` itself rather than from a library, exactly as the refusal probe above
    runs the generator itself rather than trusting a description of it.
    """
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
    )
    return completed.stdout


def _locate_acceptance_commit() -> str:
    """Return the unique commit that introduced the acceptance record.

    Section 14.3 requires ``A2`` to be *located*, not assumed: the artifact is
    an added path, so the introducing commits on the current history are read
    with ``--diff-filter=A`` and there must be exactly one.  Two introductions
    would mean the record had been deleted and re-added, which is precisely the
    substitution the uniqueness clause exists to refuse.
    """
    introductions = _git(
        "log", "--diff-filter=A", "--format=%H", "HEAD", "--", ARTIFACT
    ).split()
    assert len(introductions) == 1, (
        f"{ARTIFACT} must be introduced by exactly one commit on HEAD's "
        f"ancestry; observed {introductions}"
    )
    located = introductions[0]
    assert GIT_SHA.fullmatch(located)
    return located


def _constant_spans(source: str) -> tuple[list[tuple[int, int]], list[list[Any]]]:
    """Return the token ranges of the two approved-constant assignments.

    A span runs from the constant's own ``NAME`` token to the ``NEWLINE`` that
    ends its logical line, so a value that the formatter wrapped in parentheses
    -- which it does for the 64-hex digest, whose inline form exceeds the line
    length -- is still exactly one span.
    """
    tokens = [
        token
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in (tokenize.ENCODING, tokenize.ENDMARKER)
    ]
    spans: list[tuple[int, int]] = []
    bodies: list[list[Any]] = []
    for index, token in enumerate(tokens):
        if (
            token.type != tokenize.NAME
            or token.string not in APPROVED_CONSTANT_NAMES
            or token.start[1] != 0
        ):
            continue
        stop = index
        while tokens[stop].type != tokenize.NEWLINE:
            stop += 1
        spans.append((index, stop + 1))
        bodies.append(tokens[index : stop + 1])
    assert len(spans) == len(APPROVED_CONSTANT_NAMES), (
        f"expected one assignment per approved constant; found {len(spans)}"
    )
    return spans, bodies


def _outside_spans(source: str) -> list[tuple[int, str]]:
    """Return the ``(type, string)`` token stream outside the two spans."""
    spans, _bodies = _constant_spans(source)
    tokens = [
        token
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in (tokenize.ENCODING, tokenize.ENDMARKER)
    ]
    excised = {index for start, stop in spans for index in range(start, stop)}
    return [
        (token.type, token.string)
        for index, token in enumerate(tokens)
        if index not in excised
    ]


def _assigned_literal(body: list[Any]) -> str:
    """Return the single value token of one approved-constant assignment."""
    values = [
        token
        for token in body
        if token.type in (tokenize.STRING, tokenize.NAME)
        and token.string not in (*APPROVED_CONSTANT_NAMES, "str", "None")
    ]
    names = [token for token in body if token.string == "None"]
    if not values:
        assert names, "an approved-constant assignment carries no value token"
        return "None"
    assert len(values) == 1, (
        "an approved-constant assignment must carry exactly one value token"
    )
    return values[0].string


def test_the_record_introducing_commit_directly_parents_the_approved_evidence() -> None:
    """Section 14.3/14.4's ``A2`` ancestry clause, skipped until the flip.

    ``A2`` is located from history rather than named, and its **direct** parent
    must be the approved ``E2``.  A merge commit is refused outright: a record
    introduced on a merge has no single evidence tree it was reviewed against.
    """
    if APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None or APPROVED_EVIDENCE_SHA is None:
        pytest.skip("the M2 acceptance record is authorized at A2")
    located = _locate_acceptance_commit()
    lineage = _git("rev-list", "--parents", "-n", "1", located).split()
    assert lineage[0] == located
    assert len(lineage) == 2, (
        f"the record-introducing commit {located} must be a non-merge commit "
        f"with exactly one parent; observed {lineage[1:]}"
    )
    assert lineage[1] == APPROVED_EVIDENCE_SHA, (
        f"the direct parent of {located} is {lineage[1]}, not the approved "
        f"evidence commit {APPROVED_EVIDENCE_SHA}"
    )
    payload = _git("show", f"{located}:{ARTIFACT}")
    assert (
        hashlib.sha256(payload.encode("utf-8")).hexdigest()
        == APPROVED_ACCEPTANCE_ARTIFACT_SHA256
    )


def test_the_a2_diff_writes_only_the_section_13_4_authorized_paths() -> None:
    """Section 13.4/14.3: ``A2`` adds the record and the status prose, nothing else.

    In particular Section 14.3 requires no production-source path in the
    ``E..A`` diff, which this enforces by inventory rather than by inspection:
    the four authorized paths are the record, this validator, the append-only
    design note and the WP-9 ledger.
    """
    if APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None or APPROVED_EVIDENCE_SHA is None:
        pytest.skip("the M2 acceptance record is authorized at A2")
    located = _locate_acceptance_commit()
    changed = set(
        _git("diff-tree", "--no-commit-id", "--name-only", "-r", located).split()
    )
    assert ARTIFACT in changed
    unauthorized = sorted(changed - A2_AUTHORIZED_PATHS)
    assert not unauthorized, (
        f"the A2 commit {located} writes {unauthorized}, which Section 13.4 "
        f"does not authorize; it may write only {sorted(A2_AUTHORIZED_PATHS)}"
    )


def test_the_a2_diff_changes_only_the_two_approved_constant_assignments() -> None:
    """Section 14.3: this module's own ``A2`` diff is the two constants alone.

    The comparison is a token stream taken **outside** the two assignment spans,
    which is what makes it survive the formatter wrapping the 64-hex digest in
    parentheses while still refusing any other edit -- an added import, a
    reworded docstring, a relaxed assertion, a deleted test.  Inside the spans
    only the value may move, from the ``None`` sentinel to the approved literal.
    """
    if APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None or APPROVED_EVIDENCE_SHA is None:
        pytest.skip("the M2 acceptance record is authorized at A2")
    located = _locate_acceptance_commit()
    parent = _git("rev-list", "--parents", "-n", "1", located).split()[1]
    before = _git("show", f"{parent}:{VALIDATOR}")
    after = _git("show", f"{located}:{VALIDATOR}")

    assert _outside_spans(before) == _outside_spans(after), (
        f"the A2 commit {located} changed this module outside the two approved "
        "constant assignments"
    )

    _spans_before, bodies_before = _constant_spans(before)
    _spans_after, bodies_after = _constant_spans(after)
    approved = (APPROVED_EVIDENCE_SHA, APPROVED_ACCEPTANCE_ARTIFACT_SHA256)
    for name, body_before, body_after, value in zip(
        APPROVED_CONSTANT_NAMES, bodies_before, bodies_after, approved, strict=True
    ):
        assert _assigned_literal(body_before) == "None", (
            f"{name} must be the null sentinel at the direct parent {parent}"
        )
        assert _assigned_literal(body_after) == f'"{value}"', (
            f"{name} at {located} is not the approved literal"
        )


def test_the_retained_record_authenticates_against_the_approved_constants() -> None:
    """Section 14.3's ``A2`` state, skipped until the constants are flipped."""
    if APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None or APPROVED_EVIDENCE_SHA is None:
        pytest.skip("the M2 acceptance record is authorized at A2")
    path = REPOSITORY_ROOT / ARTIFACT
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == APPROVED_ACCEPTANCE_ARTIFACT_SHA256
    document = json.loads(payload.decode("utf-8"))
    module = _tool()
    module.validate_acceptance_document(document)
    assert document["evidence_commit_sha"] == APPROVED_EVIDENCE_SHA
    assert module.canonical_json(document) == payload
