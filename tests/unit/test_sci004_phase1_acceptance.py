"""Strict authentication of the SCI-004 phase-M1 acceptance record.

``docs/development/sci004_mmode_design.md`` Sections 13.3, 14.3 and 14.4 freeze
this module's successor authority: it lands in ``S1`` with both approved
constants as the literal ``None``, the official acceptance path **absent**, and
every synthetic strict schema fixture passing.  ``A1`` then changes *only* the
two constants below, from ``None`` to the exact lower-case 40- and
64-hexadecimal literals, and adds the acceptance JSON plus the authorized status
prose.  No import, expression, annotation, key, surrounding token, or other
literal in either assignment may change.

In the pre-``A1`` state the null constants require that JSON to be absent while
the synthetic schema tests pass.  In the ``A1`` state the active validator
authenticates the approved ``E1``, the raw acceptance bytes, the unique
introducing ``A1`` commit and the exact ``E1..A1`` authority; it never requires
the evidence artifact's ``source_sha`` to equal ``E1``.

Importing this module loads only the Python standard library plus ``pytest``,
following ``tools/sci005_stage1_acceptance.py``: an acceptance-critical
validator must not depend on a package that is merely transitively present.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Section 14.3's two approved constants.  ``A1`` replaces exactly these two
#: ``None`` literals and nothing else in this module.
APPROVED_EVIDENCE_SHA: str | None = None
APPROVED_ACCEPTANCE_ARTIFACT_SHA256: str | None = None

TOOL = "tools/sci004_mmode_phase1_acceptance.py"
ARTIFACT = "docs/development/sci004_mmode_phase1_acceptance.json"
EVIDENCE_ARTIFACT = "docs/development/sci004_mmode_phase1_evidence.json"

GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

ACCEPTANCE_SCHEMA = "radiosim.sci004.mmode-phase1-acceptance.v1"
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

#: Section 14.3's required ``A1`` re-derivation identifiers.
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


def _tool() -> Any:
    """Import the tracked generator without adding an import-time dependency."""
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    try:
        import sci004_mmode_phase1_acceptance as module
    finally:
        sys.path.pop(0)
    return module


def _synthetic_record(verdict: str = "ACCEPT") -> dict[str, Any]:
    """Return a complete synthetic record satisfying every Section 14.3 rule."""
    forty = "0" * 39 + "1"
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
        "phase": "M1",
        "verdict": verdict,
        "generated_at_utc": "2026-08-22T00:00:00Z",
        "reviewer_identity": "sci004-a1-independent-reviewer",
        "reviewer_independent": True,
        "design_sha": forty,
        "red_commit_sha": forty,
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
                "started_at_utc": "2026-08-22T00:00:00Z",
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
                "requirement_id": "sci004.section-4.2.frame-certificate",
                "evidence": "the certificate did not reproduce",
                "required_remediation": "recompute and re-review",
            }
        ],
        "accepted_limitations": [
            "phase M1 is scalar only",
        ],
        "claims_not_licensed": [
            "general_speedup",
            "gpu_or_accelerator_support",
            "polarized_mmode_support",
        ],
    }


# ---------------------------------------------------------------------------
# Pre-A1 state
# ---------------------------------------------------------------------------


def test_the_approved_constants_are_null_sentinels_before_a1() -> None:
    """Section 14.3: at ``S1``/``E1`` both approved digests are ``None``."""
    if APPROVED_EVIDENCE_SHA is None or APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None:
        assert APPROVED_EVIDENCE_SHA is None
        assert APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None
        return
    assert GIT_SHA.fullmatch(APPROVED_EVIDENCE_SHA)
    assert SHA256.fullmatch(APPROVED_ACCEPTANCE_ARTIFACT_SHA256)


def test_the_official_acceptance_artifact_is_absent_before_a1() -> None:
    """Section 14.3: null constants require the acceptance JSON to be absent."""
    if APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is not None:
        return
    assert not (REPOSITORY_ROOT / ARTIFACT).exists()


def test_the_acceptance_generator_is_already_tracked_at_s1() -> None:
    """Section 14.3: the generator and validator are already tracked at ``S``."""
    assert (REPOSITORY_ROOT / TOOL).is_file()


def test_the_generator_imports_only_the_standard_library() -> None:
    """An acceptance-critical verifier carries no transitive package dependency."""
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    for forbidden in ("import numpy", "import astropy", "import pytest", "import yaml"):
        assert forbidden not in source, forbidden


def test_the_generator_refuses_before_the_evidence_commit_exists() -> None:
    """Section 14.3: the generator runs only from a globally clean exact ``E1``.

    At ``S1`` the phase evidence artifact does not exist yet, so the refusal
    names that; at ``E1`` the preflight passes and the empty review record is
    refused as a malformed argument.  Either way the assertion names a *reason*
    rather than accepting any non-zero exit, which a generator that refused
    unconditionally would also produce, and the process always fails closed
    with a frozen prefix rather than a traceback.
    """
    module = _tool()
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
    assert not (REPOSITORY_ROOT / ARTIFACT).exists()
    reasons = (
        "not globally clean",
        "commit that adds the phase evidence artifact",
        "is not UTF-8 JSON",
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
    """The ``A1`` identifiers are required, not optional reviewer prose."""
    module = _tool()
    document = _synthetic_record()
    document["rederived_oracles"] = [
        row for row in document["rederived_oracles"] if row["oracle_id"] != oracle_id
    ]
    with pytest.raises(module.AcceptanceError):
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


def test_canonical_json_matches_the_evidence_tool_spelling() -> None:
    """Both SCI-004 tools serialize identically; a divergence would be a fork."""
    module = _tool()
    assert module.canonical_json({"b": 1.0, "a": [1e-7]}) == b'{"a":[1e-7],"b":1}'


# ---------------------------------------------------------------------------
# A1 state
# ---------------------------------------------------------------------------


def test_the_retained_record_authenticates_against_the_approved_constants() -> None:
    """Section 14.3's ``A1`` state, skipped until the constants are flipped."""
    if APPROVED_ACCEPTANCE_ARTIFACT_SHA256 is None or APPROVED_EVIDENCE_SHA is None:
        pytest.skip("the M1 acceptance record is authorized at A1")
    path = REPOSITORY_ROOT / ARTIFACT
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == APPROVED_ACCEPTANCE_ARTIFACT_SHA256
    document = json.loads(payload.decode("utf-8"))
    module = _tool()
    module.validate_acceptance_document(document)
    assert document["evidence_commit_sha"] == APPROVED_EVIDENCE_SHA
    assert module.canonical_json(document) == payload
