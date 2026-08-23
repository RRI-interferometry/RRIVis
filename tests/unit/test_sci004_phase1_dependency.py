"""Strict ``R1`` dependency and design-binding validator for SCI-004 phase M1.

``docs/development/sci004_mmode_design.md`` Sections 13.2, 13.7 and 14.0 make the
Phase-M1 red slice conditional on facts that no red oracle can establish for
itself, and this module is the validator that holds all of them. It passes at
``R1``: it authenticates an already accepted chain, never any m-mode physics.

**The operative ``D``.** Section 13.7 defines ``D0`` as the commit that
introduced this memo and the *operative* ``D`` as the latest independently
accepted, header-recorded design-gate commit. ``D0`` landed the WP-9 candidate;
its two required fresh independent Phase-0 reviews on 2026-08-21 -- physics /
governance and computational -- both returned ``REJECT`` against pinned
candidate bytes ``sha256:01f8c56a...``, which are exactly the ``D0`` memo blob
authenticated below. One combined bounded correction resolved every recorded
blocker; its exact pre-landing file bytes and parent-relative diff were pinned
and dual-``ACCEPT``ed, and it landed as
``71d3deb05b0d981653472dff9b17330b3dc9f9cf``. Authoring this very red slice
then surfaced the Tier-6 directory-listing writable-list gap, and the dated
2026-08-22 R1-authoring-reconciliation correction -- dual-``ACCEPT``ed on its
own pinned bytes and diff, and carrying the required supersession citation of
``71d3deb`` -- landed as
``a3afec87f201d0691430070023ac980c863cb224``. Implementing S1 against that
design then proved three requirements defective (the operational interval
enclosure, the equal-area transfer quadrature, and the continuous-field
constant-map interpretation), and the dated 2026-08-22 S1-feasibility
reconciliation landed as ``ef3aa7aac270068ac8ca3d275886ceb25e732d80``,
reopening the first red slice ``724ef94`` for a governed re-cut.
Completing S1 against *that* design then measured the Section 7.3
every-run ``1e-8`` direct-equality gate to be mathematically unattainable
-- a band-limited projection of the strict-horizon kernel converges only
algebraically -- and the dated 2026-08-23 two-tier-acceptance-gate
correction landed as ``10ae8628556d7ea95c0b70af086a82cf8bb569ec``,
reopening the re-cut red slice ``fe3f786`` in turn. Qualifying the
retuned acceptance fixture against that gate then proved its tier 1
unattainable by the same quadrature mechanism, and the dated 2026-08-23
tier-1-horizon-free-shell correction -- dual-``ACCEPT``ed on its own
pinned bytes and diff, and carrying the required supersession citation of
``10ae862`` -- landed as ``a67f3c8401e6d6ca4e6f531757df8cdf1598e941``
with the pending ``fe3f786`` reopening standing, and that re-cut landed
as ``b5af353`` encoding the final two-tier form. Completing S1 to a fully
green suite then surfaced the beam evaluator's own below-horizon gate,
and the dated 2026-08-23 ablation-clarification correction --
dual-``ACCEPT``ed on its own pinned bytes and diff, carrying the required
supersession citation of ``a67f3c8``, and reopening ``b5af353`` solely
for rebinding and record regeneration while closing every deferred
advisory -- landed as ``b8333c52688e9358e4d1747173e70196a60209ab``; the
rebind re-cut landed as ``35db7fb`` and the green source slice as
``46b7703``. Attempting E1 there then found both tracked phase generators
to be stubs and the evidence-embedding letter five orders past house
scale, and the dated 2026-08-23 evidence-generation-reconciliation
correction -- dual-``ACCEPT``ed on its own pinned bytes and diff,
carrying the required supersession citation of ``b8333c5``, reopening
``46b7703`` as a superseded implementation and ``35db7fb`` for this very
rebind -- landed as ``1ae7d5a94434cea35534647d4dbcef692b9e245c``.
Executing that rebind then proved the record-regeneration obligation
self-contradictory -- the operative tree contains ``S1``, so nothing is
red -- and the dated 2026-08-23 post-source-record-retention correction,
carrying the required supersession citation of ``1ae7d5a``, landed as
``112570ff2bba42e6ab57be133318e3c0bfe32f7c``, the operative ``D``; per
its rule this re-cut retains the record's last genuinely observed bytes
(``design_sha`` = ``b8333c5``, a header-enumerated chain commit) instead
of regenerating them.

``a3afec8`` remains the frozen **gate anchor** (the operative ``D`` when
the ``G1`` gate ran, equal to ``G1`` itself), and per the corrected
Section 13.2 the anchor precedes ``G1`` while the operative ``D`` follows
it through the header-enumerated chain; a correction accepted after a gate
has run does not re-run that gate. The chain is exactly nine links -- the
memo-introducing ``D0``, seven superseded design commits, and the operative
``D`` -- and the tests below prove from Git objects that no other commit
between ``D0`` and the operative ``D`` touched the memo.

**The M1 gate.** Section 13.2 (as corrected) records why the WP-7 verifier
cannot run against ``G1`` itself: it requires clean ``HEAD == --descendant`` and
re-diffs the WP-7-frozen ``pixi.toml``/``pixi.lock`` bytes, and the accepted
``v0.4.0`` release commit changed ``pixi.toml`` after that freeze, so the
protected-source rule rejects every descendant of ``ae2650f`` -- which is every
legally constructible ``G1``. The M1 gate therefore replays the certificate at
the *frozen historical replay descendant*
``c6a5ce90ae3160150b1699f97b45bb693d4ed886``, the ``descendant_commit`` recorded
inside the already-authenticated SCI-005 Stage-1 dependency artifact, and proves
the ``G1`` ancestry facts from Git objects directly. ``R1`` retains those exact
stdout bytes at ``docs/development/sci004_mmode_phase1_wp7_dependency.json``.

The byte comparison against the replayed stdout is what authenticates the
retained certificate, so no digest of it is pinned here. Inability to create,
authenticate, execute, or clean up the temporary worktree is a hard failure and
never mutates the caller's checkout.
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
from typing import Any, NamedTuple

import pytest

#: The operative SCI-004 design commit ``D`` (Section 13.7). Section 14.0
#: authorises exactly one such assignment in this file, and no later phase may
#: change it.
APPROVED_SCI004_D_SHA = "112570ff2bba42e6ab57be133318e3c0bfe32f7c"

#: The globally clean programme tip ``G1`` (Section 13.2). It equals the
#: frozen gate anchor -- the operative ``D`` at gate time -- and per the
#: corrected Section 13.2 it is an ancestor of the operative ``D``, not the
#: other way around, because six accepted corrections landed after the gate.
APPROVED_SCI004_G1_SHA = "a3afec87f201d0691430070023ac980c863cb224"

#: The independently accepted WP-7 CPU acceptance commit ``A`` (Section 13.2).
APPROVED_WP7_CPU_A_SHA = "7e5f469c835c1137a3a3a870d27c5d9f5e8f3520"

#: The frozen historical replay descendant the M1 gate runs the WP-7 verifier
#: at, because no descendant of the accepted ``v0.4.0`` release commit can
#: satisfy the upstream protected-source rule (Section 13.2).
APPROVED_WP7_REPLAY_DESCENDANT_SHA = "c6a5ce90ae3160150b1699f97b45bb693d4ed886"

#: Section 13.7's memo-introducing commit. It is not a binding constant: the
#: memo names it in its own Section 13.7 text, which the chain test checks.
DESIGN_D0_SHA = "978fef6ddd885355dd06f1deeb04aa2927626d71"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DESIGN_MEMO_PATH = "docs/development/sci004_mmode_design.md"
DESIGN_INDEX_PATH = "docs/index.rst"
DESIGN_LEDGER_PATH = "PostTier8RemediationPlan.md"
REGISTER_PATH = "Fix.md"

#: Section 13.1's complete design authority, sorted.
DESIGN_AUTHORITY_PATHS: tuple[str, ...] = (
    DESIGN_LEDGER_PATH,
    DESIGN_MEMO_PATH,
    DESIGN_INDEX_PATH,
)

CERTIFICATE_PATH = "docs/development/sci004_mmode_phase1_wp7_dependency.json"
CPU_EVIDENCE_TOOL_PATH = "tools/wp7_perf001_cpu_evidence.py"

#: The landed operative-``D`` memo blob.
D_MEMO_BLOB_SHA256 = "637bc67826af6269dee02a29054d0f2a43aaadd4619159daf25c848aeb1595ba"
#: The landed operative-``D`` parent-relative memo diff, reproduced hermetically.
D_LANDED_MEMO_DIFF_SHA256 = (
    "8b2d62c160a72b7e6af3b43f9ce54658390cfa4d7ecd957765915927980a0435"
)
#: The superseded Phase-0 correction landing (Section 13.7's ``superseded
#: design`` interval commit) and its landed memo blob.
D1_SHA = "71d3deb05b0d981653472dff9b17330b3dc9f9cf"
D1_MEMO_BLOB_SHA256 = "7287d2e371ced9373933cc6ce8d9d99c67e0c85bd69d9ed1aad71312f09e0d78"
#: The Phase-0 correction's pre-landing review pins, recorded in the memo
#: header by its own landed paragraph.
D1_PRE_LANDING_FILE_SHA256 = (
    "7beb148eda543f21e56c8720f11b51e6e7cfd3f593c013609f0136200437a6aa"
)
D1_PRE_LANDING_DIFF_SHA256 = (
    "7d4dc4af03f384ac04deb29411a1c32a1b1916251bd9d903b6bb5e528b3f86c7"
)
#: The superseded R1-authoring-reconciliation landing -- the frozen gate
#: anchor -- its landed memo blob, and its pre-landing review pins.
D2_SHA = "a3afec87f201d0691430070023ac980c863cb224"
D2_MEMO_BLOB_SHA256 = "5ae8fb80ef7668e575ef784fc79dfa1ca15d56cef09f2f390913803eb6513dca"
D2_PRE_LANDING_FILE_SHA256 = (
    "fe2f8334dda068451c08d86ec43a83bf8654d47d00c4f91f27d377e735d8288a"
)
D2_PRE_LANDING_DIFF_SHA256 = (
    "c7e3eda08556a55b22dc33a867116803f24725884f0fb56ab650c4f7201698e9"
)
#: The ``D0`` memo blob. The memo header records this exact digest as the pinned
#: candidate bytes both Phase-0 reviews were run against, so it is authenticated
#: twice: against the Git object and against the accepted header text.
D0_MEMO_BLOB_SHA256 = "01f8c56a32e3f649c576393d53b3ad29967b9c4b69bc2ba82c33ff17312a5591"
#: The superseded S1-feasibility-reconciliation landing, its landed memo blob
#: and parent-relative memo diff, and its pre-landing review pins.
D3_SHA = "ef3aa7aac270068ac8ca3d275886ceb25e732d80"
D3_MEMO_BLOB_SHA256 = "219b445234b9ea631304c8cd0cab38001d89bd7a9b41678b2b09f6b3f4124548"
D3_LANDED_MEMO_DIFF_SHA256 = (
    "1967562c6a5937c9a942d95db3c052591438a00da21f2003c5d10a5c9152b3a6"
)
D3_PRE_LANDING_FILE_SHA256 = (
    "09ea150e8285ea8067b4bc57056390cda2805dcde056c1dbce02364eb1f442bf"
)
D3_PRE_LANDING_DIFF_SHA256 = (
    "478dc114c8c1be3a2635389f54d216e1570c7b272c20bab5228e9c03f0e8a82f"
)
#: The superseded two-tier-acceptance-gate landing, likewise.
D4_SHA = "10ae8628556d7ea95c0b70af086a82cf8bb569ec"
D4_MEMO_BLOB_SHA256 = "72f83b7b003938a6ef300da6319a5330f6544a8ed7b3a564c42d5edf0d40f7b9"
D4_LANDED_MEMO_DIFF_SHA256 = (
    "37f0688b1b5f171c0192f123c7ce8186df0ce86b1d1b1250dd366462a1bae3c7"
)
D4_PRE_LANDING_FILE_SHA256 = (
    "437ae3a2e8cd0b733eb21a97435714dd70535038820c136fe39d4ead1dacd069"
)
D4_PRE_LANDING_DIFF_SHA256 = (
    "f8ada4348e3ba3598f01510cd5ed10946721ad404ac0a847df37c61e2cc4a760"
)
#: The superseded tier-1-horizon-free-shell landing, its landed memo blob and
#: hunk, and its pre-landing review pins.
D5_SHA = "a67f3c8401e6d6ca4e6f531757df8cdf1598e941"
D5_MEMO_BLOB_SHA256 = "5cb27420d1cbb79ef9c1cededee26c4560ee7dd532e0556368b9c2d91a9d4bc5"
D5_LANDED_MEMO_DIFF_SHA256 = (
    "bd15e8c2b459732ef67febc654f2c6fca88df2b710be7ab61f96cdce633ea4bc"
)
D5_PRE_LANDING_FILE_SHA256 = (
    "05e535126a3ae345f50b1772e9e7af7f35ae4e5612e105ba9cd28373b03f12e9"
)
D5_PRE_LANDING_DIFF_SHA256 = (
    "df522a6d95d6ef946ec6116f6594d17aa6a401d9c09759bc3ae9485f380dc7c3"
)
#: The superseded ablation-clarification landing, its landed memo blob and
#: hunk, and its pre-landing review pins.
D6_SHA = "b8333c52688e9358e4d1747173e70196a60209ab"
D6_MEMO_BLOB_SHA256 = "51a3f5f730860734d3ff62469f09173af52a72f847c79eeb60acc76d8f9b0830"
D6_LANDED_MEMO_DIFF_SHA256 = (
    "6c5a48561abf61cdc7f684b211b9074a2c86470785384304526197e2e94276de"
)
D6_PRE_LANDING_FILE_SHA256 = (
    "757224b46ae36240020444863043c07ea82b04e6da7e3e5fa32138c1e01f6258"
)
D6_PRE_LANDING_DIFF_SHA256 = (
    "f24b8c183709ff3b734a82e5ee558e48720a85f2662ed0f3136b347f95fed9a0"
)
#: The superseded evidence-generation-reconciliation landing, its landed memo
#: blob and hunk, and its pre-landing review pins.
D7_SHA = "1ae7d5a94434cea35534647d4dbcef692b9e245c"
D7_MEMO_BLOB_SHA256 = "5d73b47495469d83dfcccbba589d7ae701c9e2d2b60c99ed6c880e0d309df17c"
D7_LANDED_MEMO_DIFF_SHA256 = (
    "eea177125e6e511da84144e426611c3854ab857fd4c3c192dc0ecbc05bf87868"
)
D7_PRE_LANDING_FILE_SHA256 = (
    "f052a9654d2770edf2492a5f860617dc76b1326150e7b53354ffd436de3d807b"
)
D7_PRE_LANDING_DIFF_SHA256 = (
    "22bde951688b9704ba7feeb61b4994e6b5fab0b91f1e77e8dc4d50cd7dd9d5c6"
)
#: The operative correction's exact *pre-landing* file bytes and
#: parent-relative diff. These were never committed -- the correction landed
#: with its own header record appended -- so the accepted header text is their
#: only authority.
D_PRE_LANDING_FILE_SHA256 = (
    "d63cf1419678a60bacc7d5cd286a536e61c6fbcfbf5ae3098c7cb280bea9d8ea"
)
D_PRE_LANDING_DIFF_SHA256 = (
    "65daba30fab2db1f064c3bd860ad361d9e01866b5d7d7e74975f3b0226bfb44d"
)
#: The four reopened red-slice commits Section 13.7 records as ``superseded
#: red slice`` interval commits, each reopened by a correction; and the one
#: ``superseded implementation`` commit -- the M1 source slice whose stub
#: generators the evidence-generation reconciliation reopened.
REOPENED_RED_SLICE_SHAS: tuple[str, ...] = (
    "724ef948bb7a251d3269247341e109f8bd2c3893",
    "fe3f7865ad4684de8bfa7a305661e4e4bf2fd233",
    "b5af3539324bfc0784dd544d935cb479289692c4",
    "35db7fb16665e191feb5c6c4ced9aa3e52e5acaa",
)
SUPERSEDED_IMPLEMENTATION_SHAS: tuple[str, ...] = (
    "46b7703a727fdf3afd258034d274933e81ded289",
)

#: Section 13.2's exact certificate field list, in the order the memo prints it.
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

#: Hermetic Git configuration: a pinned diff digest must not depend on the
#: invoking user's pager, colour, prefix, or external-diff settings.
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
    """The SCI-004 M1 dependency or design binding failed strict validation."""


class _DesignCommit(NamedTuple):
    """One header-recorded commit in the ``D0 -> operative D`` chain."""

    sha: str
    kind: str
    #: Exactly the paths Section 13.7 allows this kind to touch, sorted.
    allowed_paths: tuple[str, ...]
    memo_blob_sha256: str
    label: str
    #: The hermetically reproducible parent-relative memo diff, for the links
    #: whose landed hunk the accepted record pins.  ``None`` for ``D0``, whose
    #: memo hunk is the whole introduced file rather than a correction hunk.
    landed_memo_diff_sha256: str | None = None


#: Section 13.7's complete header-enumerated chain, oldest first. The last entry
#: is the operative ``D`` and must equal :data:`APPROVED_SCI004_D_SHA`.
SCI004_DESIGN_CHAIN: tuple[_DesignCommit, ...] = (
    _DesignCommit(
        sha=DESIGN_D0_SHA,
        kind="memo-introducing",
        allowed_paths=tuple(sorted(DESIGN_AUTHORITY_PATHS)),
        memo_blob_sha256=D0_MEMO_BLOB_SHA256,
        label="WP-9 design-gate candidate",
    ),
    _DesignCommit(
        sha=D1_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D1_MEMO_BLOB_SHA256,
        label="Phase-0 bounded correction",
    ),
    _DesignCommit(
        sha=D2_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D2_MEMO_BLOB_SHA256,
        label="R1-authoring reconciliation (gate anchor)",
    ),
    _DesignCommit(
        sha=D3_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D3_MEMO_BLOB_SHA256,
        label="S1 feasibility reconciliation",
        landed_memo_diff_sha256=D3_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D4_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D4_MEMO_BLOB_SHA256,
        label="two-tier acceptance gate",
        landed_memo_diff_sha256=D4_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D5_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D5_MEMO_BLOB_SHA256,
        label="tier-1 horizon-free shell",
        landed_memo_diff_sha256=D5_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D6_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D6_MEMO_BLOB_SHA256,
        label="ablation clarification and deferred advisories",
        landed_memo_diff_sha256=D6_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D7_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D7_MEMO_BLOB_SHA256,
        label="evidence generation reconciliation",
        landed_memo_diff_sha256=D7_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=APPROVED_SCI004_D_SHA,
        kind="operative design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D_MEMO_BLOB_SHA256,
        label="post-source record retention",
        landed_memo_diff_sha256=D_LANDED_MEMO_DIFF_SHA256,
    ),
)


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


def _peel_to_commit(revision: str) -> str:
    """Resolve ``revision`` with ``^{commit}`` peeling (Section 14.0).

    ``<sha>^{commit}`` *peels* an object to the commit it names; it is not the
    parent operator. Parent resolution is ``<sha>^``, which the chain and
    ancestry tests below use explicitly.
    """
    return _git("rev-parse", "--verify", f"{revision}^{{commit}}").strip()


def _commit_parents(commit: str) -> tuple[str, ...]:
    fields = _git("rev-list", "--parents", "-n", "1", commit).split()
    if not fields or fields[0] != _peel_to_commit(commit):
        raise DependencyCertificateError(f"cannot resolve parents of {commit!r}")
    return tuple(fields[1:])


def _tree_blob(commit: str, relative: str) -> bytes:
    """Read ``relative`` at ``commit`` from Git objects, never the checkout."""
    listing = _git("ls-tree", "-z", commit, "--", relative)
    entries = [entry for entry in listing.split("\0") if entry]
    if len(entries) != 1:
        raise DependencyCertificateError(
            f"{relative!r} does not resolve to exactly one tree entry at {commit}"
        )
    metadata, _tab, name = entries[0].partition("\t")
    mode, object_type, object_id = metadata.split()
    # ``docs/index.rst`` carries a historical executable bit, so both regular
    # file modes are accepted; a symlink or gitlink is not.
    if name != relative or object_type != "blob" or mode not in ("100644", "100755"):
        raise DependencyCertificateError(
            f"{relative!r} at {commit} is not a regular blob: {entries[0]!r}"
        )
    completed = subprocess.run(
        ["git", "cat-file", "blob", object_id],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise DependencyCertificateError(
            f"cannot cat-file {object_id} for {relative!r} at {commit}"
        )
    return completed.stdout


def _changed_paths(commit: str) -> tuple[str, ...]:
    """Return the sorted parent-relative changed paths of a single-parent commit."""
    if len(_commit_parents(commit)) != 1:
        raise DependencyCertificateError(f"{commit!r} is not a single-parent commit")
    listing = _git("diff-tree", "--no-commit-id", "--name-only", "-r", "-z", commit)
    return tuple(sorted(entry for entry in listing.split("\0") if entry))


def _hermetic_diff_digest(commit: str, *paths: str) -> str:
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
            *paths,
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise DependencyCertificateError(
            f"hermetic diff of {commit!r} failed: "
            f"{completed.stderr.decode('utf-8', 'replace').strip()}"
        )
    return hashlib.sha256(completed.stdout).hexdigest()


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    """Return the inclusive ancestry answer Section 13.2 requires."""
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
    """Parse the retained certificate line under Section 13.2's strict rules."""
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
            "certificate must carry exactly the Section 13.2 fields in sorted "
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
    """Replay Section 13.2's upstream command from the frozen replay descendant.

    Returns the replayed stdout bytes and the elapsed wall-clock seconds. The
    worktree and its temporary directory are removed on success and on failure,
    and the caller's checkout is never mutated.
    """
    descendant = str(certificate["descendant_commit"])
    acceptance = str(certificate["acceptance_commit"])
    if descendant != APPROVED_WP7_REPLAY_DESCENDANT_SHA:
        raise DependencyCertificateError(
            f"the replay anchor must be {APPROVED_WP7_REPLAY_DESCENDANT_SHA!r}"
        )
    if acceptance != APPROVED_WP7_CPU_A_SHA:
        raise DependencyCertificateError(
            f"the acceptance commit must be {APPROVED_WP7_CPU_A_SHA!r}"
        )
    temporary = Path(tempfile.mkdtemp(prefix="sci004-m1-replay-"))
    worktree = temporary / "replay"
    started = time.monotonic()
    try:
        _git("worktree", "add", "--detach", str(worktree), descendant)
        resolved = _git("rev-parse", "HEAD", cwd=worktree).strip()
        if resolved != descendant:
            raise DependencyCertificateError(
                f"detached worktree resolved to {resolved!r}, not {descendant!r}"
            )
        status = _git("status", "--porcelain=v1", "--untracked-files=all", cwd=worktree)
        if status.strip():
            raise DependencyCertificateError(
                f"the detached replay worktree is dirty:\n{status}"
            )
        tool = worktree / CPU_EVIDENCE_TOOL_PATH
        if not tool.is_file():
            raise DependencyCertificateError(
                f"{CPU_EVIDENCE_TOOL_PATH} is absent from the replay tree"
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


# ---------------------------------------------------------------------------
# Section 14.0 -- the frozen bindings and the authenticated operative ``D``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("APPROVED_SCI004_D_SHA", APPROVED_SCI004_D_SHA),
        ("APPROVED_SCI004_G1_SHA", APPROVED_SCI004_G1_SHA),
        ("APPROVED_WP7_CPU_A_SHA", APPROVED_WP7_CPU_A_SHA),
        (
            "APPROVED_WP7_REPLAY_DESCENDANT_SHA",
            APPROVED_WP7_REPLAY_DESCENDANT_SHA,
        ),
    ],
)
def test_this_module_freezes_exactly_one_assignment_per_binding(
    name: str,
    expected: str,
) -> None:
    """Section 13.2 freezes exactly these four constants, here and nowhere else."""
    tree = ast.parse(Path(__file__).read_bytes().decode("utf-8"))
    bindings = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    ]

    assert len(bindings) == 1
    value = bindings[0].value
    assert isinstance(value, ast.Constant)
    assert value.value == expected
    assert _is_lower_hex(expected, width=40)


def test_the_operative_design_commit_peels_and_is_a_single_parent_non_merge() -> None:
    """Section 14.0: resolve with ``^{commit}`` peeling, then require a non-merge.

    ``^{commit}`` peels an object; it is emphatically not the parent operator,
    so the parent count is read separately from ``rev-list --parents``.
    """
    assert _peel_to_commit(APPROVED_SCI004_D_SHA) == APPROVED_SCI004_D_SHA

    parents = _commit_parents(APPROVED_SCI004_D_SHA)
    assert len(parents) == 1
    assert _peel_to_commit(f"{APPROVED_SCI004_D_SHA}^") == parents[0]


def test_the_operative_design_commit_is_an_ancestor_of_the_current_head() -> None:
    """Section 14.0: ``D`` is an ancestor of this phase's ``R`` commit."""
    head = _git("rev-parse", "HEAD").strip()

    assert _is_ancestor(APPROVED_SCI004_D_SHA, head)


def test_the_memo_was_introduced_at_d0_with_its_exact_three_path_diff() -> None:
    """Section 13.7: ``D0``'s parent-relative diff touches exactly Section 13.1."""
    assert _changed_paths(DESIGN_D0_SHA) == tuple(sorted(DESIGN_AUTHORITY_PATHS))
    assert len(DESIGN_AUTHORITY_PATHS) == 3

    # The memo is *introduced* there: absent from the parent, present at D0.
    with pytest.raises(DependencyCertificateError):
        _tree_blob(f"{DESIGN_D0_SHA}^", DESIGN_MEMO_PATH)
    assert _tree_blob(DESIGN_D0_SHA, DESIGN_MEMO_PATH)


def test_the_header_enumerated_chain_from_d0_to_the_operative_d_is_exact() -> None:
    """Section 13.7/14.0: every chain link matches its kind and its allowed paths.

    The chain is exactly nine links here -- the memo-introducing ``D0``,
    the superseded Phase-0, R1-authoring-reconciliation, S1-feasibility,
    two-tier-acceptance-gate, tier-1-horizon-free-shell,
    ablation-clarification and evidence-generation-reconciliation
    corrections, and the accepted post-source-record-retention correction
    that is the operative
    ``D`` -- and
    the test proves from Git objects that no *other* commit between ``D0``
    and the operative ``D`` touched the memo, so the enumeration is complete
    rather than merely consistent.
    """
    assert len(SCI004_DESIGN_CHAIN) == 9
    assert SCI004_DESIGN_CHAIN[0].sha == DESIGN_D0_SHA
    assert SCI004_DESIGN_CHAIN[-1].sha == APPROVED_SCI004_D_SHA
    assert [entry.kind for entry in SCI004_DESIGN_CHAIN] == [
        "memo-introducing",
        "superseded design",
        "superseded design",
        "superseded design",
        "superseded design",
        "superseded design",
        "superseded design",
        "superseded design",
        "operative design",
    ]
    for earlier, later in zip(
        SCI004_DESIGN_CHAIN, SCI004_DESIGN_CHAIN[1:], strict=False
    ):
        assert _is_ancestor(earlier.sha, later.sha), later.label

    for entry in SCI004_DESIGN_CHAIN:
        assert len(_commit_parents(entry.sha)) == 1, entry.label
        assert _changed_paths(entry.sha) == tuple(sorted(entry.allowed_paths)), (
            entry.label
        )
        blob = _tree_blob(entry.sha, DESIGN_MEMO_PATH)
        assert hashlib.sha256(blob).hexdigest() == entry.memo_blob_sha256, entry.label

    touching = _git(
        "rev-list",
        f"{DESIGN_D0_SHA}..{APPROVED_SCI004_D_SHA}",
        "--",
        DESIGN_MEMO_PATH,
    ).split()
    assert touching == [entry.sha for entry in reversed(SCI004_DESIGN_CHAIN[1:])]


def test_every_pinned_correction_diff_reproduces_hermetically() -> None:
    """Each pinned correction's memo hunk is byte-reproducible from Git.

    The three corrections that landed after the gate carry a pinned landed
    hunk as well as their pre-landing review pins, so the record is checkable
    from Git objects and not only against the accepted header text.
    """
    pinned = [entry for entry in SCI004_DESIGN_CHAIN if entry.landed_memo_diff_sha256]

    assert len(pinned) == 6
    for entry in pinned:
        assert (
            _hermetic_diff_digest(entry.sha, DESIGN_MEMO_PATH)
            == entry.landed_memo_diff_sha256
        ), entry.label


def test_every_recorded_review_digest_appears_in_the_accepted_memo_header() -> None:
    """The memo header is the primary record of what the two reviews bound.

    A correction that lands with its own record sentence cannot have its
    reviewed *pre-landing* bytes recovered from Git -- those bytes were never
    committed -- so authenticating those pins means checking them against the
    accepted memo text. The ``D0`` candidate pin is stronger: it is recorded in
    the header *and* reproducible as the ``D0`` blob.
    """
    memo = _tree_blob(APPROVED_SCI004_D_SHA, DESIGN_MEMO_PATH).decode("utf-8")

    for digest in (
        D0_MEMO_BLOB_SHA256,
        D1_PRE_LANDING_FILE_SHA256,
        D1_PRE_LANDING_DIFF_SHA256,
        D2_PRE_LANDING_FILE_SHA256,
        D2_PRE_LANDING_DIFF_SHA256,
        D3_PRE_LANDING_FILE_SHA256,
        D3_PRE_LANDING_DIFF_SHA256,
        D4_PRE_LANDING_FILE_SHA256,
        D4_PRE_LANDING_DIFF_SHA256,
        D5_PRE_LANDING_FILE_SHA256,
        D5_PRE_LANDING_DIFF_SHA256,
        D6_PRE_LANDING_FILE_SHA256,
        D6_PRE_LANDING_DIFF_SHA256,
        D7_PRE_LANDING_FILE_SHA256,
        D7_PRE_LANDING_DIFF_SHA256,
        D_PRE_LANDING_FILE_SHA256,
        D_PRE_LANDING_DIFF_SHA256,
    ):
        assert _is_lower_hex(digest, width=64)
        assert f"sha256:{digest}" in memo, digest
    assert DESIGN_D0_SHA in memo
    # Section 13.7 requires every correction's header record to cite the commit
    # it superseded by SHA, and a correction that reopens a committed red or
    # source slice to cite that slice too.
    for superseded in (D1_SHA, D2_SHA, D3_SHA, D4_SHA, D5_SHA, D6_SHA, D7_SHA):
        assert superseded in memo, superseded
    for reopened in REOPENED_RED_SLICE_SHAS + SUPERSEDED_IMPLEMENTATION_SHAS:
        assert reopened in memo, reopened
    # A pre-landing pin is never equal to the bytes that actually landed.
    assert D1_PRE_LANDING_FILE_SHA256 != D1_MEMO_BLOB_SHA256
    assert D2_PRE_LANDING_FILE_SHA256 != D2_MEMO_BLOB_SHA256
    assert D3_PRE_LANDING_FILE_SHA256 != D3_MEMO_BLOB_SHA256
    assert D4_PRE_LANDING_FILE_SHA256 != D4_MEMO_BLOB_SHA256
    assert D5_PRE_LANDING_FILE_SHA256 != D5_MEMO_BLOB_SHA256
    assert D6_PRE_LANDING_FILE_SHA256 != D6_MEMO_BLOB_SHA256
    assert D7_PRE_LANDING_FILE_SHA256 != D7_MEMO_BLOB_SHA256
    assert D_PRE_LANDING_FILE_SHA256 != D_MEMO_BLOB_SHA256
    for entry in SCI004_DESIGN_CHAIN:
        if entry.landed_memo_diff_sha256:
            assert entry.landed_memo_diff_sha256 != entry.memo_blob_sha256
    assert D_PRE_LANDING_DIFF_SHA256 != D_LANDED_MEMO_DIFF_SHA256


def test_the_operative_design_tree_records_sci004_as_roadmap() -> None:
    """Section 14.0: authenticate the ``D``-tree register row without claiming
    ``Fix.md`` was writable at ``D``."""
    register = _tree_blob(APPROVED_SCI004_D_SHA, REGISTER_PATH).decode("utf-8")
    rows = [line for line in register.splitlines() if line.startswith("| SCI-004 |")]

    assert len(rows) == 1
    assert rows[0].split("|")[2].strip() == "ROADMAP"
    assert REGISTER_PATH not in _changed_paths(APPROVED_SCI004_D_SHA)


def test_the_design_index_entry_names_the_memo_at_the_operative_commit() -> None:
    """Section 13.1's index entry is part of the authenticated design authority."""
    index = _tree_blob(APPROVED_SCI004_D_SHA, DESIGN_INDEX_PATH).decode("utf-8")
    stem = DESIGN_MEMO_PATH.removeprefix("docs/").removesuffix(".md")
    entries = [line for line in index.splitlines() if line.strip() == stem]

    assert entries == [f"   {stem}"]


# ---------------------------------------------------------------------------
# Section 13.2 -- the ``G1`` ancestry facts and the immutable-byte rule
# ---------------------------------------------------------------------------


def test_the_gate_anchor_and_wp7_acceptance_are_ancestors_of_the_gate_tip() -> None:
    """Section 13.2 (as corrected): the gate anchor -- the operative ``D``
    when the ``G1`` gate ran -- and the WP-7 acceptance precede ``G1``, and
    the operative ``D`` follows the anchor through the header-enumerated
    chain rather than preceding ``G1``."""
    assert _is_ancestor(D2_SHA, APPROVED_SCI004_G1_SHA)
    assert _is_ancestor(APPROVED_WP7_CPU_A_SHA, APPROVED_SCI004_G1_SHA)
    assert _is_ancestor(APPROVED_SCI004_G1_SHA, APPROVED_SCI004_D_SHA)


def test_the_first_parent_range_from_the_gate_anchor_to_the_design_has_no_merge() -> (
    None
):
    """Section 13.2/13.7: the anchor-to-operative-``D`` interval is merge-free.

    The gate ran at ``G1 == gate anchor``; the operative ``D`` now follows it
    through the superseded red slice and the accepted correction, so the
    merge-free first-parent rule applies to ``G1..D`` and every interval
    commit must be single-parent.
    """
    merges = _git(
        "rev-list",
        "--first-parent",
        "--merges",
        f"{APPROVED_SCI004_G1_SHA}..{APPROVED_SCI004_D_SHA}",
    ).split()

    assert merges == []
    for commit in _git(
        "rev-list",
        "--first-parent",
        f"{APPROVED_SCI004_G1_SHA}..{APPROVED_SCI004_D_SHA}",
    ).split():
        assert len(_commit_parents(commit)) == 1, commit


def test_the_gate_tip_remains_an_ancestor_of_the_current_head() -> None:
    """Phase-aware: ``G1`` is ``HEAD`` while ``R1`` is authored, an ancestor after."""
    head = _git("rev-parse", "HEAD").strip()

    assert _is_ancestor(APPROVED_SCI004_G1_SHA, head)


def test_the_gate_tip_carries_no_new_sci004_byte() -> None:
    """Section 13.2: the gate tip contains no new SCI-004 red, source,
    evidence, or acceptance byte.

    Ancestry is inclusive, so ``G1`` may *be* a header-enumerated design
    commit -- whose bytes are design-authority bytes, which Section 13.2
    does not ban from the gate tip. Only when ``G1`` is some other commit
    must it carry no SCI-004 byte at all.
    """
    changed = _changed_paths(APPROVED_SCI004_G1_SHA)
    chain_entry = next(
        (entry for entry in SCI004_DESIGN_CHAIN if entry.sha == APPROVED_SCI004_G1_SHA),
        None,
    )

    if chain_entry is not None:
        assert changed == tuple(sorted(chain_entry.allowed_paths))
    else:
        assert not any("sci004" in path for path in changed)
        assert DESIGN_MEMO_PATH not in changed


@pytest.mark.parametrize(
    "relative",
    [DESIGN_MEMO_PATH, DESIGN_INDEX_PATH, DESIGN_LEDGER_PATH, REGISTER_PATH],
)
def test_the_protected_design_bytes_are_identical_across_the_gate_range(
    relative: str,
) -> None:
    """Section 13.2's immutable-byte rule, restated for the gate anchor.

    The gate ran with ``G1`` equal to the anchor, so the original ``D..G1``
    identity is the anchor's own trivial identity. What survives the two
    later corrections is: ``Fix.md`` and the index entry are byte-identical
    from the anchor through the operative ``D`` (no correction may touch
    them), while the memo and the PostTier ledger may differ from the
    anchor *only* through header-enumerated chain commits, which the chain
    test proves exhaustively.
    """
    at_anchor = _tree_blob(D2_SHA, relative)
    at_gate = _tree_blob(APPROVED_SCI004_G1_SHA, relative)
    assert hashlib.sha256(at_anchor).hexdigest() == hashlib.sha256(at_gate).hexdigest()

    if relative in (REGISTER_PATH, DESIGN_INDEX_PATH):
        at_design = _tree_blob(APPROVED_SCI004_D_SHA, relative)
        assert (
            hashlib.sha256(at_design).hexdigest() == hashlib.sha256(at_gate).hexdigest()
        )
    else:
        touching = _git(
            "rev-list",
            f"{APPROVED_SCI004_G1_SHA}..{APPROVED_SCI004_D_SHA}",
            "--",
            relative,
        ).split()
        chain_shas = {entry.sha for entry in SCI004_DESIGN_CHAIN}
        assert all(commit in chain_shas for commit in touching), touching


def test_the_wp9_ledger_cells_still_state_the_gated_roadmap_position() -> None:
    """Section 13.2's PostTier WP-9 ledger cells, read from Git objects."""
    ledger = _tree_blob(APPROVED_SCI004_G1_SHA, DESIGN_LEDGER_PATH).decode("utf-8")
    rows = [line for line in ledger.splitlines() if line.startswith("| WP-9 |")]

    assert rows
    assert any("SCI-004` remains ROADMAP" in row or "ROADMAP" in row for row in rows)


# ---------------------------------------------------------------------------
# Section 13.2 -- the retained certificate and its detached-worktree replay
# ---------------------------------------------------------------------------


def test_retained_certificate_parses_strictly_with_exactly_sixteen_fields(
    certificate: Mapping[str, Any],
) -> None:
    """Section 13.2 freezes the certificate's schema, field set, verdict, and flag."""
    assert set(certificate) == set(CERTIFICATE_FIELDS)
    assert len(CERTIFICATE_FIELDS) == 16
    assert certificate["schema_version"] == CERTIFICATE_SCHEMA
    assert certificate["verdict"] == CERTIFICATE_VERDICT
    assert certificate["passed"] is True
    assert certificate["acceptance_commit"] == APPROVED_WP7_CPU_A_SHA
    assert certificate["descendant_commit"] == APPROVED_WP7_REPLAY_DESCENDANT_SHA


def test_the_retained_certificate_is_the_authenticated_sci005_stage1_line() -> None:
    """Section 13.2: the M1 line is the already-authenticated Stage-1 artifact.

    Retaining a re-derived look-alike would defeat the point: the frozen replay
    descendant is defined as the ``descendant_commit`` *recorded inside* that
    artifact, so the two files must be the same bytes.
    """
    retained, _parsed = read_retained_certificate()
    upstream = (
        REPOSITORY_ROOT / "docs" / "development" / "sci005_stage1_wp7_dependency.json"
    ).read_bytes()

    assert retained == upstream


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
    """Section 13.2: an upstream interface mismatch is a hard failure."""
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


def test_the_replay_anchor_is_not_a_descendant_of_the_v040_release_commit() -> None:
    """The corrected Section 13.2 reason the live tip cannot be the anchor.

    ``ae2650f`` changed ``pixi.toml`` after the WP-7 freeze, so the upstream
    protected-source rule rejects every descendant of it -- including every
    legally constructible ``G1``. The frozen historical descendant predates it,
    which is exactly why the replay is possible at all.
    """
    release = "ae2650fb9b8a380eea5ba0e3052f93626416335c"

    assert _is_ancestor(release, APPROVED_SCI004_G1_SHA)
    assert not _is_ancestor(release, APPROVED_WP7_REPLAY_DESCENDANT_SHA)
    assert _is_ancestor(APPROVED_WP7_CPU_A_SHA, APPROVED_WP7_REPLAY_DESCENDANT_SHA)


def test_detached_worktree_replay_reproduces_the_retained_certificate(
    certificate: Mapping[str, Any],
) -> None:
    """Section 13.2's replay, including its mandatory cleanup discipline.

    The upstream verifier requires a clean ``HEAD == --descendant``, so the
    validator attaches a detached worktree at the frozen replay descendant,
    authenticates the tool blob it is about to run, executes it *from that tree*,
    and compares stdout byte-for-byte with the retained line.
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
        line.split()[0] == "worktree" and "sci004-m1-replay-" in line
        for line in _git("worktree", "list", "--porcelain").splitlines()
        if line.strip()
    )
