"""Strict ``R3`` dependency and design-binding validator for SCI-004 phase M3.

``docs/development/sci004_mmode_design.md`` Sections 13.2, 13.7 and 14.0 make the
phase-M3 red slice conditional on facts no output oracle can establish for
itself, and this module is the validator that holds all of them. It passes at
``R3``: it authenticates an already accepted chain, never any m-mode physics.

**The phase unlock.** The retained phase-M2 acceptance record carries
``acceptance_commit_sha = null`` with the reason "self-reference: the next R or C
binds the containing A commit". This slice is that next ``R``, so it binds
``A2`` explicitly: the commit is peeled, required to be a single-parent non-merge
whose parent is exactly ``E2``, required to touch exactly the four paths
Section 13.4 authorises an ``A2`` to touch, and required to carry the accepted
``ACCEPT`` artifact at its retained digest, with the null self-SHA and its exact
reason. That is the same shape ``tests/unit/test_sci004_phase1_dependency.py``
used to bind the upstream WP-7 acceptance for M1 and
``tests/unit/test_sci004_phase2_red_failures.py`` used to bind ``A1`` for M2.

**The ``G3`` gate.** Section 13.2: "For M3, accepted SCI-004 ``A2`` and the
independently accepted SCI-005 Stage-2 ``A2`` must both be ancestors of globally
clean ``G3``; ancestry is inclusive and the first-parent range from SCI-004
``A2`` to ``G3`` contains no merge."  Both are proved here from Git objects.
Ancestry is inclusive, and the SCI-004 ``A2`` unlock is the later of the two
dependency commits, so ``G3`` *is* that commit -- exactly the shape the M1 gate
took, where ``G1`` equalled the operative ``D`` of its own time. The
consequences of that inclusive equality are asserted, not assumed:
:func:`test_the_gate_tip_carries_no_new_sci004_byte` requires the tip's
parent-relative diff to be precisely the four paths Section 13.4 grants ``A2``
when the tip *is* the named unlock, and to carry no SCI-004 byte at all
otherwise. See the module note below on the one design-text tension this
reading resolves.

**The two replays.** Section 13.2 rules that at ``HEAD==G3``, before any M3 red
byte exists, the exact Stage-2 verifier command runs and emits one canonical
UTF-8 JSON line with a final LF, which ``R3`` retains at
``docs/development/sci004_mmode_phase3_sci005_dependency.json``; and that "the
M3 validator additionally creates a clean detached worktree at exact ``R3``,
runs the Stage-2 verifier with ``--descendant <R3>``, and requires the stdout
bytes to be identical to the retained ``G3`` line; the verifier output is
descendant-independent while both ancestry checks must pass".  Both replays are
performed here in fresh detached worktrees whose tool blob is authenticated
before it is executed, and both compare stdout byte-for-byte with the retained
file.

``R3`` is the commit this file is committed in, so its SHA cannot be a frozen
constant -- the same self-reference Section 14.1 handles with a null
``red_commit_sha``. The ``R3`` replay anchor is therefore *derived* from Git
rather than declared, and the derivation is itself ruled: it is the first
first-parent commit *after the operative* ``D``, whose parent must be exactly
that commit. Section 13.2 now reads "``R3^==G3`` unless a Section 13.7 accepted
correction stars the ``G3 -> R3`` edge, in which case ``R3`` directly parents
the operative correction commit", and this file is the re-cut that edge
produced. The superseded derivation -- the first commit after ``G3`` that
*added* this validator -- resolves the reopened red slice ``62a7d3d9…`` as an
immutable Git fact forever and would silently authenticate it; the accepted
correction's mandate says so in those words, so the live derivation searches
strictly after the operative ``D``, where the superseded slice cannot appear.
Before the re-cut commit exists the anchor is ``HEAD``, which the same rule
forces to be the operative ``D`` itself, and the module says so in
:func:`test_the_r3_replay_anchor_is_the_live_child_of_the_operative_correction`
rather than pretending a not-yet-existing commit was replayed.

**A design-text tension, resolved the way the M1 precedent resolved its own.**
Section 13.2 closes with "Neither gate tip contains a new SCI-004 red, source,
evidence, or acceptance byte", while inclusive ancestry makes the accepted
SCI-004 ``A2`` -- an acceptance commit -- a legal ``G3``. The M1 gate met the
same shape: ``G1`` equalled the operative ``D``, whose bytes are
design-authority bytes, and its validator recorded that Section 13.2 "does not
ban" those from the gate tip. The M3 reading is the exact analogue: when the
gate tip *is* one of the two named dependency commits the gate authenticates,
its own bytes are that dependency's accepted bytes rather than a *new* SCI-004
byte smuggled into an otherwise unrelated tip, and the sentence binds a tip
that is some other commit. The bounded alternative -- an interposed
SCI-004-free commit -- is not constructible inside Section 13.5's ``R3``
writable list, which is why the analogue is taken rather than invented.

The byte comparison against the replayed stdout is what authenticates the
retained certificate; its raw digest is pinned as well, because Section 14.3
makes "the raw stdout digest" part of what the ``A3``
``m3.sci005-dependency-gate`` oracle authenticates. Inability to create,
authenticate, execute, or clean up a temporary worktree is a hard failure and
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

from tests.unit.test_sci004_phase1_dependency import (
    SCI004_DESIGN_CHAIN as M1_DESIGN_CHAIN,
)
from tests.unit.test_sci004_phase1_dependency import (
    _DesignCommit,
)

#: The operative SCI-004 design commit ``D`` (Section 13.7), frozen for phase M3
#: exactly as Section 14.0 requires: "R1's dependency validator,
#: ``tests/unit/test_sci004_phase2_red_failures.py`` at R2, and R3's dependency
#: validator each freeze the exact assignment ``APPROVED_SCI004_D_SHA=...``".
#: It is accepted correction #25, which reopens the rejected fingerprint
#: attempt after the fourth phase-3 red slice and committed source slice.
#: Section 14.0 binds
#: "the operative ``D`` current at that phase's ``R``", and the retained-evidence
#: record states the rule for a re-cut: "a fresh ``R`` takes the ``D`` current at
#: its own cut".  The binding therefore advances again, past the
#: retained-evidence landing the previous cut froze.
APPROVED_SCI004_D_SHA = "ca3c37171aaaeec175b5ad72d324957762303853"

#: The globally clean programme tip ``G3`` (Section 13.2). Ancestry is
#: inclusive, and this tip is the later of the two named dependency commits --
#: the accepted SCI-004 ``A2`` itself.
APPROVED_SCI004_G3_SHA = "a28d16fc30926b53b50dad7165b15056ce252bb0"

#: The independently accepted SCI-005 Stage-2 acceptance commit (Section 13.2).
APPROVED_SCI005_STAGE2_A_SHA = "7523706c8c8d480de079100bc21871eb5616536e"

#: The accepted SCI-004 phase-M2 acceptance commit ``A2`` -- this phase's
#: unlock, whose retained artifact carries the null self SHA that "the next R or
#: C" binds. It is the same commit as ``G3`` under Section 13.2's inclusive
#: ancestry; the two names are kept apart because the roles are.
APPROVED_SCI004_A2_SHA = "a28d16fc30926b53b50dad7165b15056ce252bb0"

#: The phase-M2 evidence commit ``E2``. Section 14.4: each ``A`` directly
#: parents and names its ``E``, so this is ``A2^``.
APPROVED_SCI004_E2_SHA = "50772ec1462c3561e350b46be404c5de9e74b8f7"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DESIGN_MEMO_PATH = "docs/development/sci004_mmode_design.md"
DESIGN_INDEX_PATH = "docs/index.rst"
DESIGN_LEDGER_PATH = "PostTier8RemediationPlan.md"
REGISTER_PATH = "Fix.md"

CERTIFICATE_PATH = "docs/development/sci004_mmode_phase3_sci005_dependency.json"
STAGE2_TOOL_PATH = "tools/sci005_stage2_acceptance.py"
M2_ACCEPTANCE_PATH = "docs/development/sci004_mmode_phase2_acceptance.json"
M2_ACCEPTANCE_VALIDATOR_PATH = "tests/unit/test_sci004_phase2_acceptance.py"
DEPENDENCY_VALIDATOR_PATH = "tests/unit/test_sci004_phase3_dependency.py"

#: Section 13.4's exact ``A2`` grant, sorted, which the ``A2`` commit must touch
#: exactly.
A2_AUTHORIZED_PATHS: tuple[str, ...] = (
    DESIGN_LEDGER_PATH,
    DESIGN_MEMO_PATH,
    M2_ACCEPTANCE_PATH,
    M2_ACCEPTANCE_VALIDATOR_PATH,
)

#: Correction #25's exact fresh fingerprint-delta ``R3`` writable list.
#: own diff
#: is checked against it, so a first child that is not this phase's red slice is
#: refused rather than replayed.
R3_AUTHORIZED_PATHS: frozenset[str] = frozenset(
    {
        "docs/development/sci004_mmode_phase3_fingerprint_post_source_red_failures.json",
        "tests/characterization/test_sci004_mmode.py",
        "tests/unit/test_sci004_phase3_dependency.py",
        "tests/unit/test_sci004_phase3_red_failures.py",
        "tools/sci004_mmode_phase3_red.py",
    }
)

#: The superseded phase-3 slices before correction #25, in cut order. None may
#: ever be resolved as the live anchor for the fresh fingerprint retry.
SUPERSEDED_RED_SLICE_SHA = "62a7d3d90dcbf0488e8b7c875ae5f95acba007b6"
SUPERSEDED_RECUT_RED_SLICE_SHA = "a07279f4e1220f4e064d747406350df6fd1190fb"
SUPERSEDED_SECOND_RECUT_RED_SLICE_SHA = "c6cc74bb88bb123b20b4c549bc92da73cc057c1e"
SUPERSEDED_THIRD_RECUT_RED_SLICE_SHA = "7070cc3ddb1c2557d02e4a3f2a89b907575bed0b"
SUPERSEDED_IMPLEMENTATION_SHA = "a61526d686ab768f05ecffa80cfd6223d4ee4c62"

D24_SHA = "4d507bf1333ccaa4c8beec3815370ba0f6043bb2"
SUPERSEDED_FINGERPRINT_R3_SHA = "944e0ee66ebdaffafab86f4f8f4253a404aa902c"
SUPERSEDED_FINGERPRINT_S3_SHA = "b07925ab14b56b3ca0fa863f806290748a31df6b"
REJECTED_E3_SHA = "886e62fd9f8328826b388b8960ed7413da26b6d1"
REJECTED_A3_SHA = "8529da951e2378115ffde8d5da3e2af56f3323d0"

SUPERSEDED_FINGERPRINT_R3_PATHS: tuple[str, ...] = (
    "docs/development/sci004_mmode_phase3_post_source_red_failures.json",
    "tests/unit/test_io/test_hdf5_result.py",
    "tests/unit/test_sci004_phase3_dependency.py",
    "tests/unit/test_sci004_phase3_red_failures.py",
    "tools/sci004_mmode_phase3_red.py",
)
SUPERSEDED_FINGERPRINT_S3_PATHS: tuple[str, ...] = (
    "src/radiosim/io/hdf5.py",
    "tests/unit/test_sci004_phase3_evidence.py",
    "tools/sci004_mmode_phase3_evidence.py",
)
REJECTED_E3_PATHS: tuple[str, ...] = (
    "docs/development/sci004_mmode_phase3_evidence.json",
    "docs/development/sci004_mmode_phase3_evidence.md",
    "output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json",
    "tests/unit/test_sci004_phase3_evidence.py",
)
REJECTED_A3_PATHS: tuple[str, ...] = (
    "docs/development/sci004_mmode_phase3_acceptance.json",
    "tests/unit/test_sci004_phase3_acceptance.py",
)

REJECTED_A3_ARTIFACT_SHA256 = (
    "283fb5264f5ecd86aed1300ae504b85946cf1f4d36b1c4c09bc92bb4f269421d"
)
REJECTED_E3_ARTIFACT_SHA256 = (
    "600b51ac4d70778ee2d3bdf7b8842b83ba77dc34d541784ad1ad7d8e5be5f8ae"
)
REJECTED_E3_REPRODUCTION_SHA256 = (
    "039539a865b5d92e86379f44a324271232e8a947301e380ec7b1b1848e907b4e"
)
REJECTED_E3_PERFORMANCE_SHA256 = (
    "07e59d3176866a78c17244849d6493365e9d410547e884cf56b254e60babe193"
)
EXTERNAL_REVIEW_PATH = Path(
    "/Users/kartikmandar/MacProjects/sci004-a3-independent-review-reject-20260826.json"
)
EXTERNAL_REVIEW_SHA256 = (
    "43c12807aa9f316af53e6058ebec7f18dd0b6ea66d308cb1c488d77185907d82"
)
REJECTED_A3_ARTIFACT_PATH = "docs/development/sci004_mmode_phase3_acceptance.json"
REJECTED_E3_ARTIFACT_PATH = "docs/development/sci004_mmode_phase3_evidence.json"
REJECTED_E3_REPRODUCTION_PATH = "docs/development/sci004_mmode_phase3_evidence.md"
REJECTED_E3_PERFORMANCE_PATH = (
    "output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json"
)

SUPERSEDED_THIRD_RECUT_RED_SLICE_PATHS: tuple[str, ...] = (
    "docs/development/sci004_mmode_phase3_red_failures.json",
    "tests/characterization/test_sci004_mmode.py",
    "tests/performance/test_sci004_mmode.py",
    "tests/unit/test_sci004_phase3_dependency.py",
    "tests/unit/test_sci004_phase3_red_failures.py",
    "tools/sci004_mmode_phase3_red.py",
)

SUPERSEDED_IMPLEMENTATION_PATHS: tuple[str, ...] = (
    ".gitignore",
    "docs/api/io.rst",
    "docs/user_guide/backends.rst",
    "docs/user_guide/configuration_support.rst",
    "output/benchmarks/reference/README.md",
    "src/radiosim/benchmarks/__init__.py",
    "src/radiosim/core/mmode/solver.py",
    "src/radiosim/core/result.py",
    "src/radiosim/io/hdf5.py",
    "src/radiosim/io/standard_visibility.py",
    "src/radiosim/io/summary_json.py",
    "tests/unit/test_sci004_phase3_acceptance.py",
    "tests/unit/test_sci004_phase3_evidence.py",
    "tools/sci004_mmode_phase3_acceptance.py",
    "tools/sci004_mmode_phase3_evidence.py",
)

#: The retained phase-M2 acceptance artifact, by raw digest.
M2_ACCEPTANCE_SHA256 = (
    "2de45da5ad35caa2105340ce0870f5370e0f840958b51ac424c938a8fbe4b0dd"
)

#: The retained upstream certificate line, by raw digest (Section 14.3's "raw
#: stdout digest"). The byte comparison against the replay is the primary
#: authentication; this pin makes a silent substitution detectable without one.
RETAINED_CERTIFICATE_SHA256 = (
    "00f0150b8654023e574d70f2ae7efa775f46c708d88a8cd1c26b9d0138a5b499"
)

#: The Stage-2 verifier blob at ``G3``, authenticated before it is executed in
#: either replay worktree. Section 13.2: "authenticates the invoked tool blob
#: from that tip".
STAGE2_TOOL_SHA256 = "92f0d94c3fcc963d93d2ec49247f5dc24e436ce350eab8d40defb82dc5af5b70"

#: Section 13.2's exact Stage-2 certificate field list, in the order the memo
#: prints it -- which is also the order the upstream tool emits.
CERTIFICATE_FIELDS: tuple[str, ...] = (
    "schema_version",
    "stage",
    "acceptance_commit_sha",
    "acceptance_artifact_path",
    "acceptance_artifact_sha256",
    "evidence_commit_sha",
    "evidence_artifact_path",
    "evidence_artifact_sha256",
    "source_sha",
    "verdict",
    "successor_unlocks",
)
CERTIFICATE_SCHEMA = "radiosim.sci005.stage-acceptance-certificate.v1"
CERTIFICATE_VERDICT = "ACCEPT"
CERTIFICATE_STAGE = 2
#: Section 13.2: "``successor_unlocks==["SCI004.M3","SCI005.U2"]``".
CERTIFICATE_UNLOCKS: tuple[str, ...] = ("SCI004.M3", "SCI005.U2")

_COMMIT_FIELDS: tuple[str, ...] = (
    "acceptance_commit_sha",
    "evidence_commit_sha",
    "source_sha",
)
_DIGEST_FIELDS: tuple[str, ...] = (
    "acceptance_artifact_sha256",
    "evidence_artifact_sha256",
)
_PATH_FIELDS: tuple[str, ...] = (
    "acceptance_artifact_path",
    "evidence_artifact_path",
)

#: The operative-``D`` memo blob.
D_MEMO_BLOB_SHA256 = "4f8020b5ef432393de2fad48f385a8bf03573e61760d7834f9b86bbface750dd"

#: Section 13.7's ``D0 -> D`` chain continues past the eleven links the M1
#: dependency validator froze. These are the fourteen that landed after ``A1``:
#: thirteen ``superseded design`` corrections and the operative one. Each entry
#: carries its landed memo blob, its hermetically reproducible landed memo hunk,
#: and the pre-landing review pins its own header paragraph records.
D10_SHA = "d8adeaaee1045b930fb7ca7e4bd0905655cd4725"
D10_MEMO_BLOB_SHA256 = (
    "15eb0b7dcf3800562443b38ba3276718f416f7e98956549e59bed0939a097efd"
)
D10_LANDED_MEMO_DIFF_SHA256 = (
    "30b511dd886d911f879b28b653c396076c931a797dbd15faa3d7d69f41945be1"
)
D10_PRE_LANDING_FILE_SHA256 = (
    "45b3a939ab588ce636bf29613cae3e582f73e5d5cb40935ac8cf40ee5f395646"
)
D10_PRE_LANDING_DIFF_SHA256 = (
    "239b1a0fa127be99d41c407d936ded9686baa98e868ead4001e59b0e5ab125a1"
)

D11_SHA = "e02f3975607b821b31c083a197cf7ea23865c062"
D11_MEMO_BLOB_SHA256 = (
    "714b1a3f2009ec457b3ab4c14395d93b6228692bf5f513e42d482f96b0e13ef7"
)
D11_LANDED_MEMO_DIFF_SHA256 = (
    "848492d3e3f77ac86d020e20be964a2c17443d93b5e81687517b92c4cab7a14b"
)
D11_PRE_LANDING_FILE_SHA256 = (
    "cb97177cec7d112fba492952ab6a4857995ed0e283797c9c4468925014e49d7a"
)
D11_PRE_LANDING_DIFF_SHA256 = (
    "2e3e65737124d283a45c70093097511464cc48dcf998ef75195e163c851dee1e"
)

D12_SHA = "3b28e615ba6e752ce040f0464e3e55c36604b4a3"
D12_MEMO_BLOB_SHA256 = (
    "e6d12176eb9e4c2ddb9caead4f7536e227f5e92113d6180b38d2e0340996a651"
)
D12_LANDED_MEMO_DIFF_SHA256 = (
    "f3a9ecbbf3865caacf64b6babb49f727769db86cf126e5fb1514dcdeb99c3c24"
)
D12_PRE_LANDING_FILE_SHA256 = (
    "585f0d87e030cb726731ff849754add5b406b69d2df13a6803375c210619e8dd"
)
D12_PRE_LANDING_DIFF_SHA256 = (
    "f06ea8c8684738086aa080f6cafbe7e5ef21cbbeba88f1d4730c6d34b94cc298"
)

D13_SHA = "d0ccab7718959dc06a5fb66bc16af9b0524c4546"
D13_MEMO_BLOB_SHA256 = (
    "50783e8e264a015b5e08385ab396ebb1720b785d27e911b4065138ebadabd424"
)
D13_LANDED_MEMO_DIFF_SHA256 = (
    "69d367caf2184f7636db393943c1c74d058e91efcf8fed247a8c5e665bba9780"
)
D13_PRE_LANDING_FILE_SHA256 = (
    "03e2d4a20e1d362651043b978086582d8cb1ab7e962dfd529e078120df05d112"
)
D13_PRE_LANDING_DIFF_SHA256 = (
    "32843f801027139b0f34c98f526e528ceb6089396152565f63ed1e4c360e3265"
)

D14_SHA = "d806854997cbaf9469c4cf33e36c277e287c37c3"
D14_MEMO_BLOB_SHA256 = (
    "b56508665c4c34cecbd3eeafb8c68533b541eeb58da9c195630c9b162090439b"
)
D14_LANDED_MEMO_DIFF_SHA256 = (
    "a85382f6bb28002a598f33af03327481351014df66a3374a5b2ea62f584c31d8"
)
D14_PRE_LANDING_FILE_SHA256 = (
    "bd23d410b2a69e7376ca715e303a591b90d043427f68b40c23ff9efc000be44c"
)
D14_PRE_LANDING_DIFF_SHA256 = (
    "b24dd5616e669f74c6e1be93738f1aa91d402b7780b3a3051ffbab7b927dfb6a"
)

D15_SHA = "b9a9d7a8a49974bae4634f24fbc805077cdc4ef8"
D15_MEMO_BLOB_SHA256 = (
    "a4df523709a88cb985dfed052915c43ee464feda02b6a5fd0a68c16c6345497c"
)
D15_LANDED_MEMO_DIFF_SHA256 = (
    "e0e6cea99069ff22fecc4280c0c87008c0d134d44b3af75c9a32064d5f577113"
)
D15_PRE_LANDING_FILE_SHA256 = (
    "be58258c75cdf88e4d838e4fe7753a415d642295e328ba86a3508f540bfc297e"
)
D15_PRE_LANDING_DIFF_SHA256 = (
    "d64a4de836331186c3eaa69b45a3d9bdb4b9879ff353e9705b1c47dfb9724d85"
)

D16_SHA = "e7902d04ce042bd3a16ab9ae3a336695e971db81"
D16_MEMO_BLOB_SHA256 = (
    "4b82759ff2bb6c1337829ed6fd901394453c87beb13c4b7f00592409f38c98af"
)
D16_LANDED_MEMO_DIFF_SHA256 = (
    "0d7dc33248dcb91029c57a56a79944de5d7e05fa0923e1ac40a2ca03862cccb3"
)
D16_PRE_LANDING_FILE_SHA256 = (
    "20e104fe73130431ca1122905d3e99a9236981fe5cba067f6601008a15c121ea"
)
D16_PRE_LANDING_DIFF_SHA256 = (
    "ce79e0ec83552f8453968ef08ce70e6a28d439ca7dc202df02beda0d67270752"
)

#: The accepted-capability-characterization-envelope landing.  It was the
#: operative ``D`` the *superseded* re-cut froze, and both later corrections
#: record it becoming "a ``superseded design`` interval commit on the
#: header-enumerated ``D0 -> D`` chain".
D17_SHA = "53ee53c3b829512ef02f81215238090be63937d9"
D17_MEMO_BLOB_SHA256 = (
    "450179d46552934dc064b71bf463adbafcbdef3a9c3f11854f9b3b7a87438183"
)
D17_LANDED_MEMO_DIFF_SHA256 = (
    "119109f12fb436b540efe06d843060ac44166c776d4effb3d53868782d431519"
)
D17_PRE_LANDING_FILE_SHA256 = (
    "6ea19f19ee1d368687043477140b7d938d4668ec2aca0c7123a824484f3a0d4d"
)
D17_PRE_LANDING_DIFF_SHA256 = (
    "dec9df0fb5b37edcb87067092288f74bc92861c6163a08a0344cc0246819739a"
)

#: The performance-product landing, which granted ``S3`` the benchmark fixture
#: constant and which the retained-evidence correction superseded in turn.
D18_SHA = "29c702cfc824ad73b2e0aeacd5b4b23bcc6c18cf"
D18_MEMO_BLOB_SHA256 = (
    "648314e084a468a9feeaeec0c56d48d3e60f5a5f17005f525b91d2e0b5f352db"
)
D18_LANDED_MEMO_DIFF_SHA256 = (
    "740fc8360f203ac499b693b6a1fe7e30a486375759206361be277f280d2f2d99"
)
D18_PRE_LANDING_FILE_SHA256 = (
    "f32a4f6793abb983c42f6605b444f9eecaf706be09843dd914b952f4cae43e14"
)
D18_PRE_LANDING_DIFF_SHA256 = (
    "f2b5f1abfe7891dc40788713db106c5ef6bbcedb65d6f8fa1b36b92ab142b975"
)

#: The retained-evidence-surfaces landing, which conformed the five evidence
#: surfaces and took the fresh-``R`` route for the falsified performance oracle.
#: The honest-backend-axis correction superseded it in turn.
D19_SHA = "83d98f70fef0bf35977a3b6d4a7101ff67a7a953"
D19_MEMO_BLOB_SHA256 = (
    "f1d22b2bd832d0c7aa34a62c2b7cb5408b9092b8a358cf151a97f0066d4fd6c3"
)
D19_LANDED_MEMO_DIFF_SHA256 = (
    "9b10aad942732c83adad97b31c9654fa42b303ec731f1162500d5872739d4742"
)
D19_PRE_LANDING_FILE_SHA256 = (
    "d53ed7f0de129ec2dd6f7ca760b110d83b2fbcbc2527aac744a084ba38a40b5a"
)
D19_PRE_LANDING_DIFF_SHA256 = (
    "80fe0e816c7b3fba169736f92aa23f0983de36224420913eb3e106b1a5a067d2"
)

D20_SHA = "923ae332c02d9b2d4edfddf09d1d61241e9d5a63"
D20_MEMO_BLOB_SHA256 = (
    "a208d502554901dd5be15be52df04f8a4bed568b1a01ebcbc7c2fbb9d53d0e05"
)
D20_LANDED_MEMO_DIFF_SHA256 = (
    "e280e6873840975bacf1cac79a93c2dd7d7330b77f3b05f2ca1a1f780668cf84"
)
D20_PRE_LANDING_FILE_SHA256 = (
    "4f60cc3b464658a5b8adfcd9dea8417a9651f439c686b34715f662d40652dc5e"
)
D20_PRE_LANDING_DIFF_SHA256 = (
    "53ecb8da1ebf1d6e36599863c11cbf70ef5a61fb0322a1f44f5e908581484aad"
)

D21_SHA = "2422c5765a82e55328c25bb3b8fc08e8377c176f"
D21_MEMO_BLOB_SHA256 = (
    "e49e6624f0c56b5ca28ee47fe5e0c819d81f6e47c829f4562160640c7575a4ce"
)
D21_LANDED_MEMO_DIFF_SHA256 = (
    "35fdf496cd806553ec9e5f43c3c1e1d66b748a25f16c9da858d03b9b99d594f3"
)
D21_PRE_LANDING_FILE_SHA256 = (
    "b7f69e7aff9945ea9c35a22062ccfad7f9c63beabf1a53fcabfc6c7997a0b33e"
)
D21_PRE_LANDING_DIFF_SHA256 = (
    "49100c3684d088273382316ef2e0dba6079c411065cca6140b28c50588cbba9b"
)

D22_SHA = "6fb8b0a8d54bcf946b32a777f69359c8b83bd527"
D22_MEMO_BLOB_SHA256 = (
    "3bd5f5c166ffccd0716b0f327c94d6896f03b799ba3ac33e0abafcd6ab81d2b1"
)
D22_LANDED_MEMO_DIFF_SHA256 = (
    "fe99bc84fee310e816defeaf4d34b2022aecc86262bf28ff60c072ce6dc56dc6"
)
D22_PRE_LANDING_FILE_SHA256 = (
    "1f51e88500392f8c33ff92646ac52e44d755ef1d554a7256bc99a4a43d75ad29"
)
D22_PRE_LANDING_DIFF_SHA256 = (
    "b349f03fb48739550b6f8b50881d8bcff9cbf85aaa04a1fa82f3ec615086c40d"
)

D_LANDED_MEMO_DIFF_SHA256 = (
    "70ce2abcf200b359ef012cf45cbe93637341622f244f820c1b551ebc656d89ba"
)
D_PRE_LANDING_FILE_SHA256 = (
    "4b595da0c6946ce333c795343ef1a7db7e8c16a7ff1dc05c54af550d8f15b107"
)
D_PRE_LANDING_DIFF_SHA256 = (
    "6ae375bd2a9d1e880dfa6f9af051f700a4ae40ea433104eaabd3d53506133bc6"
)
D_PRE_LANDING_LEDGER_DIFF_SHA256 = (
    "79a9b91ee2e1156ce5c5866db5d1d28c6fd60d8cfae0ac90b5211a521555aca8"
)

D25_MEMO_BLOB_SHA256 = (
    "eb45da5adfe412cc3447303f8ac77448988317f286b27df95d783f047481791f"
)
D25_LANDED_MEMO_DIFF_SHA256 = (
    "6ac059de3a4e867560a0ecc83615e9938b45777e9e77714dcc96a54f06897ffa"
)
D25_FINAL_DESIGN_DIFF_SHA256 = (
    "5ab8d5cf856f78640585be8f9257a50e72ebf9a86cac7766887967722f70d7a8"
)
D25_FINAL_LEDGER_DIFF_SHA256 = (
    "84a215a0bb556432f5db5b09385d553f7c8b09116f13104d6e54ed7b94d47a09"
)
D25_FINAL_COMPLETE_DIFF_SHA256 = (
    "8c21e2f0193475422925ecaa4d0e6fab296d46517b115118846bb600a90911f0"
)
D25_PRE_LANDING_FILE_SHA256 = (
    "5d54c4b8c5c0312b29d2391c0de76b51a004b6c0605d2543a51ae2a46bbff2a6"
)
D25_PRE_LANDING_DIFF_SHA256 = (
    "1052133587a3af0489cf079c69e2f7a5b8869f20959bdc3d1cda8ed09d7c1acb"
)
D25_PRE_LANDING_LEDGER_SHA256 = (
    "f79329d0e0438ce5ff5c2c65d0b443fa04dae0bffeded5f20b750b64415105af"
)
D25_PRE_LANDING_LEDGER_DIFF_SHA256 = (
    "d9d30802b46941cba5d2c52ce7cd1ef405d0bc966a4a5fcf1dc4d4d12dabff44"
)
D25_PRE_LANDING_COMPLETE_DIFF_SHA256 = (
    "ccea0e4e0477ea43174f64aece99369c31b4ce221cdd0b16d38788e1bdd4dc76"
)

#: The ``D0 -> operative D`` chain past ``A1``, oldest first. Section 13.7's
#: interval kinds are the authority for the allowed-path tuples.
SCI004_DESIGN_CHAIN_CONTINUATION: tuple[_DesignCommit, ...] = (
    _DesignCommit(
        sha=D10_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D10_MEMO_BLOB_SHA256,
        label="post-acceptance repairs and the starred A1 -> R2 edge",
        landed_memo_diff_sha256=D10_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D11_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D11_MEMO_BLOB_SHA256,
        label="celestial tangent transport in the transfer kernel",
        landed_memo_diff_sha256=D11_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D12_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D12_MEMO_BLOB_SHA256,
        label="resolved-input route for the tangent frame",
        landed_memo_diff_sha256=D12_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D13_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D13_MEMO_BLOB_SHA256,
        label="the direct-RIME basis for constant receptor cells",
        landed_memo_diff_sha256=D13_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D14_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D14_MEMO_BLOB_SHA256,
        label="the singular capability pin",
        landed_memo_diff_sha256=D14_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D15_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D15_MEMO_BLOB_SHA256,
        label="the description follows the accepted capability",
        landed_memo_diff_sha256=D15_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D16_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D16_MEMO_BLOB_SHA256,
        label="un-ignoring the granted reference records",
        landed_memo_diff_sha256=D16_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D17_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D17_MEMO_BLOB_SHA256,
        label="the accepted-capability characterization envelope",
        landed_memo_diff_sha256=D17_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D18_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D18_MEMO_BLOB_SHA256,
        label="the performance product follows the envelope",
        landed_memo_diff_sha256=D18_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D19_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D19_MEMO_BLOB_SHA256,
        label="the retained-evidence surfaces follow the envelope",
        landed_memo_diff_sha256=D19_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D20_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D20_MEMO_BLOB_SHA256,
        label="the honest backend axis",
        landed_memo_diff_sha256=D20_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D21_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D21_MEMO_BLOB_SHA256,
        label="the scalar-table kernel exception",
        landed_memo_diff_sha256=D21_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D22_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D22_MEMO_BLOB_SHA256,
        label="the honest memory boolean",
        landed_memo_diff_sha256=D22_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=D24_SHA,
        kind="superseded design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D_MEMO_BLOB_SHA256,
        label="sampled RSS and the polarized-HDF5 post-source oracle",
        landed_memo_diff_sha256=D_LANDED_MEMO_DIFF_SHA256,
    ),
    _DesignCommit(
        sha=APPROVED_SCI004_D_SHA,
        kind="operative design",
        allowed_paths=(DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH),
        memo_blob_sha256=D25_MEMO_BLOB_SHA256,
        label="reconstructible path-independent M3 fingerprints",
        landed_memo_diff_sha256=D25_LANDED_MEMO_DIFF_SHA256,
    ),
)

#: The complete header-enumerated chain for this phase: the eleven links R1
#: froze -- which "no later phase may change" -- followed by the fifteen that landed
#: after ``A1``. The M1 tuple is imported rather than restated for exactly that
#: reason. Its final entry was the operative ``D`` when R1 froze it and is now a
#: ``superseded design`` chain commit: every correction's header record says so
#: in those words ("that commit becomes a ``superseded design`` interval commit
#: on the header-enumerated ``D0 -> D`` chain"), so the *kind* advances with the
#: chain while the SHA, the paths and the pinned memo blob stay exactly what R1
#: froze.
SCI004_DESIGN_CHAIN: tuple[_DesignCommit, ...] = (
    *M1_DESIGN_CHAIN[:-1],
    M1_DESIGN_CHAIN[-1]._replace(kind="superseded design"),
    *SCI004_DESIGN_CHAIN_CONTINUATION,
)

#: Section 13.7: "An accepted phase acceptance commit inside the ``D0 -> D``
#: range is not a chain commit and needs no interval kind: its memo diff is
#: exactly its own phase's Section 13 append-only acceptance note."  ``A1`` is
#: the only such commit inside this range.
NON_CHAIN_ACCEPTANCE_SHAS: tuple[str, ...] = (
    "445bc83edcf7073511c41b3485ad5d326d4e1552",
    APPROVED_SCI004_A2_SHA,
)

#: The pre-landing review pins the six continuation records carry. They were
#: never committed, so the accepted header text is their only authority.
CONTINUATION_REVIEW_PINS: tuple[tuple[str, str], ...] = (
    (D10_PRE_LANDING_FILE_SHA256, D10_PRE_LANDING_DIFF_SHA256),
    (D11_PRE_LANDING_FILE_SHA256, D11_PRE_LANDING_DIFF_SHA256),
    (D12_PRE_LANDING_FILE_SHA256, D12_PRE_LANDING_DIFF_SHA256),
    (D13_PRE_LANDING_FILE_SHA256, D13_PRE_LANDING_DIFF_SHA256),
    (D14_PRE_LANDING_FILE_SHA256, D14_PRE_LANDING_DIFF_SHA256),
    (D15_PRE_LANDING_FILE_SHA256, D15_PRE_LANDING_DIFF_SHA256),
    (D16_PRE_LANDING_FILE_SHA256, D16_PRE_LANDING_DIFF_SHA256),
    (D17_PRE_LANDING_FILE_SHA256, D17_PRE_LANDING_DIFF_SHA256),
    (D18_PRE_LANDING_FILE_SHA256, D18_PRE_LANDING_DIFF_SHA256),
    (D19_PRE_LANDING_FILE_SHA256, D19_PRE_LANDING_DIFF_SHA256),
    (D20_PRE_LANDING_FILE_SHA256, D20_PRE_LANDING_DIFF_SHA256),
    (D21_PRE_LANDING_FILE_SHA256, D21_PRE_LANDING_DIFF_SHA256),
    (D22_PRE_LANDING_FILE_SHA256, D22_PRE_LANDING_DIFF_SHA256),
    (D_PRE_LANDING_FILE_SHA256, D_PRE_LANDING_DIFF_SHA256),
    (D25_PRE_LANDING_FILE_SHA256, D25_PRE_LANDING_DIFF_SHA256),
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
    """The SCI-004 M3 dependency or design binding failed strict validation."""


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
    """Resolve ``revision`` with ``^{commit}`` peeling (Section 14.0)."""
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


def _tracked_paths(commit: str) -> tuple[str, ...]:
    listing = _git("ls-tree", "-r", "-z", "--name-only", "--full-tree", commit)
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


def _binary_full_index_diff(commit: str, *paths: str) -> bytes:
    """Return the correction-governed raw binary/full-index diff bytes."""
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--no-ext-diff",
            "--binary",
            "--full-index",
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
            f"binary/full-index diff of {commit!r} failed: "
            f"{completed.stderr.decode('utf-8', 'replace').strip()}"
        )
    return completed.stdout


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
    """Parse the retained Stage-2 certificate line under Section 13.2's rules."""
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
    if tuple(parsed) != CERTIFICATE_FIELDS:
        raise DependencyCertificateError(
            "certificate must carry exactly the Section 13.2 fields in the memo's "
            f"printed order; observed {tuple(parsed)!r}"
        )
    canonical = (
        json.dumps(parsed, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if canonical != raw:
        raise DependencyCertificateError("certificate is not the canonical one line")
    if parsed["schema_version"] != CERTIFICATE_SCHEMA:
        raise DependencyCertificateError(
            f"certificate schema must be {CERTIFICATE_SCHEMA!r}"
        )
    if parsed["stage"] != CERTIFICATE_STAGE or type(parsed["stage"]) is not int:
        raise DependencyCertificateError("certificate stage must be the integer 2")
    if parsed["verdict"] != CERTIFICATE_VERDICT:
        raise DependencyCertificateError(
            f"certificate verdict must be {CERTIFICATE_VERDICT!r}"
        )
    if list(parsed["successor_unlocks"]) != list(CERTIFICATE_UNLOCKS):
        raise DependencyCertificateError(
            f"successor_unlocks must be exactly {list(CERTIFICATE_UNLOCKS)!r}"
        )
    for field in _COMMIT_FIELDS:
        if not _is_lower_hex(parsed[field], width=40):
            raise DependencyCertificateError(f"{field} must be 40 lower-case hex")
    for field in _DIGEST_FIELDS:
        if not _is_lower_hex(parsed[field], width=64):
            raise DependencyCertificateError(f"{field} must be 64 lower-case hex")
    for field in _PATH_FIELDS:
        value = parsed[field]
        if type(value) is not str or not value:
            raise DependencyCertificateError(f"{field} must be a non-empty path")
        if value != str(Path(value)) or value.startswith("/") or ".." in value:
            raise DependencyCertificateError(
                f"{field} must be a normalized repository-relative path"
            )
    return parsed


def read_retained_certificate() -> tuple[bytes, Mapping[str, Any]]:
    """Read and strictly parse the retained ``R3`` certificate bytes."""
    path = REPOSITORY_ROOT / CERTIFICATE_PATH
    if not path.is_file() or path.is_symlink():
        raise DependencyCertificateError(
            f"{CERTIFICATE_PATH} must be a retained regular file"
        )
    raw = path.read_bytes()
    return raw, parse_dependency_certificate(raw)


class _ReplayAnchor(NamedTuple):
    """One replay anchor: the commit a worktree is attached at, and its role."""

    commit: str
    role: str


def resolve_r3_replay_anchor() -> _ReplayAnchor:
    """Return the live ``R3`` replay anchor, derived from Git rather than declared.

    ``R3`` is the commit that contains this file, so it cannot be a frozen
    constant here -- the same self-reference Section 14.1 answers with a null
    ``red_commit_sha``.  It is derived instead, and *how* it is derived is
    itself a ruled fact.

    The superseded derivation asked which first-parent commit after ``G3``
    *added* this validator.  That question has one answer forever: the
    superseded red slice ``62a7d3d9…``, which added the file and whose parent
    genuinely is ``G3``.  The accepted 2026-08-24
    accepted-capability-characterization-envelope correction reopened that slice
    and recorded the consequence in its own mandate -- the ``--diff-filter=A``
    derivation "resolves the superseded add-commit as an immutable git fact
    forever and would silently authenticate it".  A re-cut validator that kept
    it would replay a commit the memo has superseded while reporting success.

    The live derivation follows the starred edge instead.  Section 13.7's
    reopened-phase rule and Section 14.4 make the re-cut ``R3`` the direct child
    of the operative correction commit -- Section 13.2's ``R3^==G3`` now reads
    "unless a Section 13.7 accepted correction stars the ``G3 -> R3`` edge, in
    which case ``R3`` directly parents the operative correction commit" -- so
    the anchor is the first commit on the first-parent chain *after* the
    operative ``D``, and its parent is required to be exactly that commit.  The
    superseded slice is an ancestor of the operative ``D`` and therefore outside
    the search range entirely, which is what makes the substitution impossible
    rather than merely unlikely.

    Until the re-cut commit exists the anchor is ``HEAD``, which the same rule
    forces to be the operative ``D`` itself.
    """
    head = _peel_to_commit("HEAD")
    if not _is_ancestor(APPROVED_SCI004_D_SHA, head):
        raise DependencyCertificateError(
            f"the operative D {APPROVED_SCI004_D_SHA} is not an ancestor of HEAD "
            f"{head}; the live R3 replay anchor cannot be derived"
        )
    successors = _git(
        "rev-list",
        "--first-parent",
        "--reverse",
        f"{APPROVED_SCI004_D_SHA}..{head}",
    ).split()
    if not successors:
        if head != APPROVED_SCI004_D_SHA:
            raise DependencyCertificateError(
                f"HEAD {head} is past the operative D with no first-parent "
                "successor; the live R3 replay anchor cannot be derived"
            )
        return _ReplayAnchor(commit=head, role="pre-commit-authoring-tip")
    anchor = successors[0]
    parents = _commit_parents(anchor)
    if parents != (APPROVED_SCI004_D_SHA,):
        raise DependencyCertificateError(
            "Section 13.2's starred G3 -> R3 edge requires the re-cut R3 to "
            f"directly parent the operative correction commit "
            f"{APPROVED_SCI004_D_SHA}; {anchor} parents {parents}"
        )
    touched = _changed_paths(anchor)
    if touched != tuple(sorted(R3_AUTHORIZED_PATHS)):
        raise DependencyCertificateError(
            f"the derived R3 anchor {anchor} touches {touched}, which is not a "
            "correction-25 R3 slice with the exact five authorized paths"
        )
    return _ReplayAnchor(commit=anchor, role="r3")


def replay_stage2_certificate(anchor: str) -> tuple[bytes, float]:
    """Replay Section 13.2's exact Stage-2 command in a clean detached worktree.

    Returns the replayed stdout bytes and the elapsed wall-clock seconds. The
    worktree and its temporary directory are removed on success and on failure,
    and the caller's checkout is never mutated. Cleanup deliberately avoids a
    repository-wide ``git worktree prune`` unless a removal actually failed: a
    global prune is shared mutable state that races a concurrently running
    sibling replay.
    """
    temporary = Path(tempfile.mkdtemp(prefix="sci004-m3-replay-"))
    worktree = temporary / "replay"
    started = time.monotonic()
    try:
        _git("worktree", "add", "--detach", str(worktree), anchor)
        resolved = _git("rev-parse", "HEAD", cwd=worktree).strip()
        if resolved != anchor:
            raise DependencyCertificateError(
                f"detached worktree resolved to {resolved!r}, not {anchor!r}"
            )
        status = _git("status", "--porcelain=v1", "--untracked-files=all", cwd=worktree)
        if status.strip():
            raise DependencyCertificateError(
                f"the detached replay worktree is dirty:\n{status}"
            )
        tool = worktree / STAGE2_TOOL_PATH
        if not tool.is_file():
            raise DependencyCertificateError(
                f"{STAGE2_TOOL_PATH} is absent from the replay tree"
            )
        tool_digest = hashlib.sha256(tool.read_bytes()).hexdigest()
        if tool_digest != STAGE2_TOOL_SHA256:
            raise DependencyCertificateError(
                f"executed tool digest {tool_digest!r} is not the authenticated "
                f"{STAGE2_TOOL_SHA256!r}"
            )
        completed = subprocess.run(
            [
                sys.executable,
                str(tool),
                "verify",
                "--acceptance-commit",
                APPROVED_SCI005_STAGE2_A_SHA,
                "--descendant",
                anchor,
            ],
            cwd=worktree,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise DependencyCertificateError(
                "upstream Stage-2 verify exited "
                f"{completed.returncode}: "
                f"{completed.stderr.decode('utf-8', 'replace').strip()}"
            )
        if completed.stderr != b"":
            raise DependencyCertificateError(
                "upstream Stage-2 verify wrote to stderr: "
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
                except DependencyCertificateError as prune_error:  # pragma: no cover
                    removal_errors.append(str(prune_error))
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
        ("APPROVED_SCI004_G3_SHA", APPROVED_SCI004_G3_SHA),
        ("APPROVED_SCI005_STAGE2_A_SHA", APPROVED_SCI005_STAGE2_A_SHA),
        ("APPROVED_SCI004_A2_SHA", APPROVED_SCI004_A2_SHA),
        ("APPROVED_SCI004_E2_SHA", APPROVED_SCI004_E2_SHA),
    ],
)
def test_this_module_freezes_exactly_one_assignment_per_binding(
    name: str,
    expected: str,
) -> None:
    """Section 13.2 freezes exactly these constants, here and nowhere else.

    "``R3`` retains the exact stdout bytes at
    ``docs/development/sci004_mmode_phase3_sci005_dependency.json`` and freezes
    in ``tests/unit/test_sci004_phase3_dependency.py`` exactly
    ``APPROVED_SCI004_D_SHA``, ``APPROVED_SCI004_G3_SHA``, and
    ``APPROVED_SCI005_STAGE2_A_SHA``."  The unlock and its evidence parent are
    frozen alongside them for the same reason the M2 red validator froze ``A1``
    and ``E1``: the retained ``A2`` artifact's null self-SHA is bound by this
    slice, and a binding that were not a single exact assignment could drift.
    """
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


def test_the_phase_three_binding_advances_the_r1_binding_through_the_chain() -> None:
    """Section 14.0's chain-advance branch, because corrections intervened.

    "The later bindings byte-match the R1 binding unless a Section 13.7 accepted
    correction intervened, in which case the later binding names the newer
    operative ``D`` and its validator authenticates the header-enumerated
    correction chain between the two bindings."  Every entry of the
    continuation tuple below did, so the byte-match is
    not asserted; R1's own frozen constant is instead required to be unchanged
    ("no later phase may change those constants") and to be the eleventh link of
    the chain this module extends.

    The binding also advances past what the *superseded* re-cut froze.  Section
    14.0 binds the ``D`` "current at that phase's ``R``", and the accepted
    retained-evidence correction spells out the consequence for a re-cut: "a
    fresh ``R`` takes the ``D`` current at its own cut".  Both superseded
    landings are therefore chain links here rather than bindings.
    """
    from tests.unit.test_sci004_phase1_dependency import (
        APPROVED_SCI004_D_SHA as R1_APPROVED_SCI004_D_SHA,
    )

    assert len(M1_DESIGN_CHAIN) == 11
    assert M1_DESIGN_CHAIN[-1].sha == R1_APPROVED_SCI004_D_SHA
    assert APPROVED_SCI004_D_SHA != R1_APPROVED_SCI004_D_SHA
    assert _is_ancestor(R1_APPROVED_SCI004_D_SHA, APPROVED_SCI004_D_SHA)
    assert len(SCI004_DESIGN_CHAIN_CONTINUATION) == 15
    assert SCI004_DESIGN_CHAIN[-1].sha == APPROVED_SCI004_D_SHA
    for superseded in (D17_SHA, D18_SHA, D19_SHA, D20_SHA, D21_SHA, D22_SHA):
        assert APPROVED_SCI004_D_SHA != superseded
        assert _is_ancestor(superseded, APPROVED_SCI004_D_SHA)


def test_the_operative_design_commit_peels_and_is_a_single_parent_non_merge() -> None:
    """Section 14.0: peel with ``^{commit}``, then require a non-merge."""
    assert _peel_to_commit(APPROVED_SCI004_D_SHA) == APPROVED_SCI004_D_SHA

    parents = _commit_parents(APPROVED_SCI004_D_SHA)
    assert len(parents) == 1
    assert _peel_to_commit(f"{APPROVED_SCI004_D_SHA}^") == parents[0]


def test_the_operative_design_commit_follows_the_gate_and_precedes_this_r() -> None:
    """Section 13.2/14.4: the operative ``D`` *follows* the gate tip.

    The accepted-capability-characterization-envelope correction landed after
    ``G3`` had already run and after ``A2`` was accepted, which is the shape
    Section 13.2's gate-anchor rule and Section 14.4's starred ``G3 ->* R3``
    edge exist for.  The ancestry direction is therefore the reverse of the
    unstarred one and is asserted in that direction rather than assumed, exactly
    as the phase-M2 red validator asserts it for its own starred edge.
    """
    head = _peel_to_commit("HEAD")

    assert _is_ancestor(APPROVED_SCI004_D_SHA, head)
    assert _is_ancestor(APPROVED_SCI004_G3_SHA, APPROVED_SCI004_D_SHA)
    assert _is_ancestor(APPROVED_SCI004_A2_SHA, APPROVED_SCI004_D_SHA)
    assert not _is_ancestor(APPROVED_SCI004_D_SHA, APPROVED_SCI004_G3_SHA)


def test_the_header_enumerated_chain_from_d0_to_the_operative_d_is_exact() -> None:
    """Section 13.7/14.0: every chain link matches its kind and allowed paths.

    The chain is twenty-six links here -- the memo-introducing ``D0``,
    twenty-four ``superseded design`` corrections, and the operative one --
    and the test
    proves from Git objects that the only *other* commits between ``D0`` and the
    operative ``D`` that touched the memo are ``A1`` and ``A2``, which
    Section 13.7 explicitly rules "is not a chain commit and needs no interval
    kind" for an accepted phase acceptance commit inside the range.
    """
    assert len(SCI004_DESIGN_CHAIN) == 26
    assert SCI004_DESIGN_CHAIN[0].kind == "memo-introducing"
    assert SCI004_DESIGN_CHAIN[-1].kind == "operative design"
    assert [entry.kind for entry in SCI004_DESIGN_CHAIN[1:-1]] == [
        "superseded design"
    ] * 24
    assert len({entry.sha for entry in SCI004_DESIGN_CHAIN}) == 26

    for earlier, later in zip(
        SCI004_DESIGN_CHAIN, SCI004_DESIGN_CHAIN[1:], strict=False
    ):
        assert _is_ancestor(earlier.sha, later.sha), later.label

    for entry in SCI004_DESIGN_CHAIN_CONTINUATION:
        assert len(_commit_parents(entry.sha)) == 1, entry.label
        assert _changed_paths(entry.sha) == tuple(sorted(entry.allowed_paths)), (
            entry.label
        )
        blob = _tree_blob(entry.sha, DESIGN_MEMO_PATH)
        assert hashlib.sha256(blob).hexdigest() == entry.memo_blob_sha256, entry.label

    touching = _git(
        "rev-list",
        f"{SCI004_DESIGN_CHAIN[0].sha}..{APPROVED_SCI004_D_SHA}",
        "--",
        DESIGN_MEMO_PATH,
    ).split()
    expected = [entry.sha for entry in reversed(SCI004_DESIGN_CHAIN[1:])]
    assert sorted(touching) == sorted(expected + list(NON_CHAIN_ACCEPTANCE_SHAS))
    for sha in NON_CHAIN_ACCEPTANCE_SHAS:
        assert sha not in {entry.sha for entry in SCI004_DESIGN_CHAIN}


def test_every_continuation_correction_diff_reproduces_hermetically() -> None:
    """Each pinned correction's memo hunk is byte-reproducible from Git."""
    for entry in SCI004_DESIGN_CHAIN_CONTINUATION:
        assert entry.landed_memo_diff_sha256, entry.label
        assert (
            _hermetic_diff_digest(entry.sha, DESIGN_MEMO_PATH)
            == entry.landed_memo_diff_sha256
        ), entry.label


def test_correction_25_final_binary_full_index_diff_identities_are_exact() -> None:
    """Authenticate D25's two files and fixed-order complete raw patch."""
    design = _binary_full_index_diff(APPROVED_SCI004_D_SHA, DESIGN_MEMO_PATH)
    ledger = _binary_full_index_diff(APPROVED_SCI004_D_SHA, DESIGN_LEDGER_PATH)
    complete = _binary_full_index_diff(
        APPROVED_SCI004_D_SHA,
        DESIGN_MEMO_PATH,
        DESIGN_LEDGER_PATH,
    )

    assert hashlib.sha256(design).hexdigest() == D25_FINAL_DESIGN_DIFF_SHA256
    assert hashlib.sha256(ledger).hexdigest() == D25_FINAL_LEDGER_DIFF_SHA256
    assert hashlib.sha256(complete).hexdigest() == D25_FINAL_COMPLETE_DIFF_SHA256


def test_rejected_fingerprint_attempt_is_authenticated_from_git_objects() -> None:
    """Bind the immutable E3/A3 rejection that correction #25 retries."""
    assert _commit_parents(REJECTED_E3_SHA) == (SUPERSEDED_FINGERPRINT_S3_SHA,)
    assert _commit_parents(REJECTED_A3_SHA) == (REJECTED_E3_SHA,)
    assert _commit_parents(APPROVED_SCI004_D_SHA) == (REJECTED_A3_SHA,)
    assert _changed_paths(REJECTED_E3_SHA) == REJECTED_E3_PATHS
    assert _changed_paths(REJECTED_A3_SHA) == REJECTED_A3_PATHS

    evidence_raw = _tree_blob(REJECTED_E3_SHA, REJECTED_E3_ARTIFACT_PATH)
    reproduction_raw = _tree_blob(REJECTED_E3_SHA, REJECTED_E3_REPRODUCTION_PATH)
    performance_raw = _tree_blob(REJECTED_E3_SHA, REJECTED_E3_PERFORMANCE_PATH)
    acceptance_raw = _tree_blob(REJECTED_A3_SHA, REJECTED_A3_ARTIFACT_PATH)
    assert hashlib.sha256(evidence_raw).hexdigest() == REJECTED_E3_ARTIFACT_SHA256
    assert (
        hashlib.sha256(reproduction_raw).hexdigest() == REJECTED_E3_REPRODUCTION_SHA256
    )
    assert hashlib.sha256(performance_raw).hexdigest() == REJECTED_E3_PERFORMANCE_SHA256
    assert hashlib.sha256(acceptance_raw).hexdigest() == REJECTED_A3_ARTIFACT_SHA256

    evidence = json.loads(evidence_raw)
    assert evidence["status"] == "candidate"
    assert evidence["design_sha"] == D24_SHA
    assert evidence["source_sha"] == SUPERSEDED_FINGERPRINT_S3_SHA
    assert evidence["red_commit_sha"] == SUPERSEDED_FINGERPRINT_R3_SHA

    acceptance = json.loads(acceptance_raw)
    assert acceptance["verdict"] == "REJECT"
    assert acceptance["reviewer_identity"] == (
        "sci004-m3-independent-acceptance-reviewer"
    )
    assert acceptance["reviewer_independent"] is True
    assert acceptance["evidence_commit_sha"] == REJECTED_E3_SHA
    assert acceptance["evidence_artifact_sha256"] == REJECTED_E3_ARTIFACT_SHA256
    assert [blocker["blocker_id"] for blocker in acceptance["blockers"]] == [
        "m3.fingerprint-input-preimage-not-retained"
    ]

    assert EXTERNAL_REVIEW_PATH.is_file()
    assert not EXTERNAL_REVIEW_PATH.is_symlink()
    external_raw = EXTERNAL_REVIEW_PATH.read_bytes()
    assert hashlib.sha256(external_raw).hexdigest() == EXTERNAL_REVIEW_SHA256
    external = json.loads(external_raw)
    assert external["verdict"] == "REJECT"
    assert external["reviewer_identity"] == (
        "sci004-m3-independent-acceptance-reviewer"
    )
    assert external["reviewer_independent"] is True


def test_every_continuation_review_digest_appears_in_the_accepted_header() -> None:
    """The memo header is the only authority for the pre-landing review pins.

    A correction that lands with its own record sentence cannot have its
    reviewed *pre-landing* bytes recovered from Git -- those bytes were never
    committed -- so authenticating those pins means checking them against the
    accepted memo text, and requiring each to differ from what actually landed.
    """
    memo = _tree_blob(APPROVED_SCI004_D_SHA, DESIGN_MEMO_PATH).decode("utf-8")

    for file_pin, diff_pin in CONTINUATION_REVIEW_PINS:
        assert _is_lower_hex(file_pin, width=64)
        assert _is_lower_hex(diff_pin, width=64)
        assert f"sha256:{file_pin}" in memo or f"`{file_pin}`" in memo, file_pin
        assert f"sha256:{diff_pin}" in memo or f"`{diff_pin}`" in memo, diff_pin
    assert _is_lower_hex(D_PRE_LANDING_LEDGER_DIFF_SHA256, width=64)
    assert f"sha256:{D_PRE_LANDING_LEDGER_DIFF_SHA256}" in memo
    for pin in (
        D25_PRE_LANDING_LEDGER_SHA256,
        D25_PRE_LANDING_LEDGER_DIFF_SHA256,
        D25_PRE_LANDING_COMPLETE_DIFF_SHA256,
    ):
        assert _is_lower_hex(pin, width=64)
        assert f"`{pin}`" in memo
    assert "`/root/physics_governance_review`" in memo
    assert "`/root/computational_provenance_review`" in memo
    assert "each returned exact `ACCEPT`" in memo
    for entry in SCI004_DESIGN_CHAIN_CONTINUATION:
        assert entry.sha in memo or entry.sha == APPROVED_SCI004_D_SHA, entry.label
    for (file_pin, _diff_pin), entry in zip(
        CONTINUATION_REVIEW_PINS, SCI004_DESIGN_CHAIN_CONTINUATION, strict=True
    ):
        assert file_pin != entry.memo_blob_sha256, entry.label
        assert file_pin != entry.landed_memo_diff_sha256, entry.label


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
# Section 14.4 -- the ``A2`` unlock this slice binds
# ---------------------------------------------------------------------------


def test_the_m2_acceptance_commit_is_a_single_parent_child_of_e2() -> None:
    """Section 14.4: "Each ``A`` directly parents and names ``E``"."""
    assert _peel_to_commit(APPROVED_SCI004_A2_SHA) == APPROVED_SCI004_A2_SHA
    parents = _commit_parents(APPROVED_SCI004_A2_SHA)

    assert parents == (APPROVED_SCI004_E2_SHA,)
    assert len(_commit_parents(APPROVED_SCI004_E2_SHA)) == 1


def test_the_m2_acceptance_commit_touches_exactly_its_authorized_paths() -> None:
    """Section 13.4/14.4: an ``A`` changes only its artifact, constants, prose."""
    assert _changed_paths(APPROVED_SCI004_A2_SHA) == tuple(sorted(A2_AUTHORIZED_PATHS))


def test_the_retained_m2_acceptance_artifact_is_an_accept_at_its_exact_digest() -> None:
    """Section 14.3: the accepted artifact, read from the ``A2`` tree object.

    Its ``acceptance_commit_sha`` is null with the self-reference reason, which
    is precisely why this module exists: the phase-M3 red slice is "the next R"
    the reason names, and binding ``A2`` here is what closes that reference.
    """
    blob = _tree_blob(APPROVED_SCI004_A2_SHA, M2_ACCEPTANCE_PATH)
    assert hashlib.sha256(blob).hexdigest() == M2_ACCEPTANCE_SHA256
    assert not blob.endswith(b"\n")

    artifact = json.loads(
        blob.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    assert artifact["schema_version"] == "radiosim.sci004.mmode-phase2-acceptance.v1"
    assert artifact["phase"] == "M2"
    assert artifact["verdict"] == "ACCEPT"
    # Section 13.7: "An accepted artifact is immutable and no commit may touch
    # one."  It therefore still names the ``D`` operative at its own phase --
    # the commit two later corrections have since superseded -- and requiring it
    # to name the new operative ``D`` would demand editing a retained accepted
    # artifact.
    assert artifact["design_sha"] == D15_SHA
    assert artifact["design_sha"] != APPROVED_SCI004_D_SHA
    assert D15_SHA in {entry.sha for entry in SCI004_DESIGN_CHAIN}
    assert artifact["evidence_commit_sha"] == APPROVED_SCI004_E2_SHA
    assert artifact["acceptance_commit_sha"] is None
    assert artifact["acceptance_commit_sha_reason"] == (
        "self-reference: the next R or C binds the containing A commit"
    )
    assert artifact["reviewer_independent"] is True
    assert artifact["blockers"] == []

    # Section 13.7: "An accepted artifact is immutable and no commit may touch
    # one."  The same bytes are still in the working tree.
    assert (REPOSITORY_ROOT / M2_ACCEPTANCE_PATH).read_bytes() == blob


def test_the_m2_acceptance_reserves_the_fingerprint_pins_for_this_phase() -> None:
    """Section 11/13.5: a retained m-mode family pin is phase-M3 scope.

    The phase-M3 characterization oracles are red *because* that is true, so the
    accepted ``A2`` claim array is read here rather than paraphrased.
    """
    artifact = json.loads(
        _tree_blob(APPROVED_SCI004_A2_SHA, M2_ACCEPTANCE_PATH).decode("utf-8")
    )
    claims = artifact["claims_not_licensed"]

    assert isinstance(claims, list) and claims
    assert claims == sorted(set(claims))
    joined = " ".join(claims).lower()
    assert "fingerprint pin" in joined
    assert "m3" in joined
    for forbidden in ("speed", "gpu"):
        assert forbidden in joined, forbidden


# ---------------------------------------------------------------------------
# Section 13.2 -- the ``G3`` ancestry facts and the immutable-byte rule
# ---------------------------------------------------------------------------


def test_both_named_dependencies_are_ancestors_of_the_gate_tip() -> None:
    """Section 13.2: "accepted SCI-004 ``A2`` and the independently accepted
    SCI-005 Stage-2 ``A2`` must both be ancestors of globally clean ``G3``;
    ancestry is inclusive"."""
    assert _is_ancestor(APPROVED_SCI004_A2_SHA, APPROVED_SCI004_G3_SHA)
    assert _is_ancestor(APPROVED_SCI005_STAGE2_A_SHA, APPROVED_SCI004_G3_SHA)
    # The SCI-005 dependency is the *earlier* of the two, so the inclusive
    # endpoint is the SCI-004 unlock rather than the upstream one.
    assert _is_ancestor(APPROVED_SCI005_STAGE2_A_SHA, APPROVED_SCI004_A2_SHA)
    assert not _is_ancestor(APPROVED_SCI004_A2_SHA, APPROVED_SCI005_STAGE2_A_SHA)


def test_the_first_parent_range_from_the_m2_acceptance_to_g3_has_no_merge() -> None:
    """Section 13.2: "the first-parent range from SCI-004 ``A2`` to ``G3``
    contains no merge"."""
    merges = _git(
        "rev-list",
        "--first-parent",
        "--merges",
        f"{APPROVED_SCI004_A2_SHA}..{APPROVED_SCI004_G3_SHA}",
    ).split()

    assert merges == []
    for commit in _git(
        "rev-list",
        "--first-parent",
        f"{APPROVED_SCI004_A2_SHA}..{APPROVED_SCI004_G3_SHA}",
    ).split():
        assert len(_commit_parents(commit)) == 1, commit


def test_the_gate_tip_remains_an_ancestor_of_the_current_head() -> None:
    """Phase-aware: ``G3`` is ``HEAD`` while ``R3`` is authored, an ancestor after."""
    head = _peel_to_commit("HEAD")

    assert _is_ancestor(APPROVED_SCI004_G3_SHA, head)


def test_the_gate_tip_carries_no_new_sci004_byte() -> None:
    """Section 13.2: "Neither gate tip contains a new SCI-004 red, source,
    evidence, or acceptance byte."

    Ancestry is inclusive, so ``G3`` may *be* one of the two named dependency
    commits the gate authenticates. When it is, its bytes are that dependency's
    own accepted bytes -- here exactly the four paths Section 13.4 grants
    ``A2`` -- rather than a new SCI-004 byte introduced at an unrelated tip, and
    the sentence binds the other case: a gate tip that is some other commit must
    carry no SCI-004 byte at all. This is the M1 precedent applied unchanged;
    its validator carved the same branch when ``G1`` equalled the operative
    ``D``.
    """
    changed = _changed_paths(APPROVED_SCI004_G3_SHA)

    if APPROVED_SCI004_G3_SHA == APPROVED_SCI004_A2_SHA:
        assert changed == tuple(sorted(A2_AUTHORIZED_PATHS))
    elif APPROVED_SCI004_G3_SHA == APPROVED_SCI005_STAGE2_A_SHA:
        assert not any("sci004" in path for path in changed)
    else:
        assert not any("sci004" in path for path in changed)
        assert DESIGN_MEMO_PATH not in changed
    # No phase-M3 byte exists at the gate tip in any branch of the reading.
    assert not any("phase3" in path for path in changed)


@pytest.mark.parametrize(
    "relative",
    [DESIGN_MEMO_PATH, DESIGN_INDEX_PATH, DESIGN_LEDGER_PATH, REGISTER_PATH],
)
def test_the_protected_design_bytes_are_identical_across_the_gate_range(
    relative: str,
) -> None:
    """Section 13.2's immutable-byte rule over ``SCI-004 A2..G3``."""
    at_acceptance = _tree_blob(APPROVED_SCI004_A2_SHA, relative)
    at_gate = _tree_blob(APPROVED_SCI004_G3_SHA, relative)

    assert (
        hashlib.sha256(at_acceptance).hexdigest() == hashlib.sha256(at_gate).hexdigest()
    )


def test_every_sci004_owned_byte_at_the_m2_acceptance_survives_to_the_gate_tip() -> (
    None
):
    """Section 13.2: "Across SCI-004 ``A2..G3``, every SCI-004-owned byte at
    ``A2``, including prior artifacts and validators, remains byte-identical"."""
    owned = [
        path
        for path in _tracked_paths(APPROVED_SCI004_A2_SHA)
        if "sci004" in path.lower()
    ]

    assert owned, "the A2 tree must carry the retained SCI-004 artifacts"
    assert any(path.endswith("_red_failures.json") for path in owned)
    assert any(path.endswith("_evidence.json") for path in owned)
    assert any(path.endswith("_acceptance.json") for path in owned)
    for path in owned:
        before = _tree_blob(APPROVED_SCI004_A2_SHA, path)
        after = _tree_blob(APPROVED_SCI004_G3_SHA, path)
        assert (
            hashlib.sha256(before).hexdigest() == hashlib.sha256(after).hexdigest()
        ), path


def test_the_wp9_ledger_cells_still_state_the_gated_roadmap_position() -> None:
    """Section 13.2's PostTier WP-9 ledger cells, read from Git objects."""
    ledger = _tree_blob(APPROVED_SCI004_G3_SHA, DESIGN_LEDGER_PATH).decode("utf-8")
    rows = [line for line in ledger.splitlines() if line.startswith("| WP-9 |")]

    assert rows
    assert any("ROADMAP" in row for row in rows)


# ---------------------------------------------------------------------------
# Section 13.2 -- the retained certificate and its two worktree replays
# ---------------------------------------------------------------------------


def test_retained_certificate_parses_strictly_with_exactly_eleven_fields(
    certificate: Mapping[str, Any],
) -> None:
    """Section 13.2 freezes the certificate's schema, fields, verdict, unlocks."""
    assert tuple(certificate) == CERTIFICATE_FIELDS
    assert len(CERTIFICATE_FIELDS) == 11
    assert certificate["schema_version"] == CERTIFICATE_SCHEMA
    assert certificate["stage"] == CERTIFICATE_STAGE
    assert certificate["verdict"] == CERTIFICATE_VERDICT
    assert certificate["acceptance_commit_sha"] == APPROVED_SCI005_STAGE2_A_SHA
    assert list(certificate["successor_unlocks"]) == list(CERTIFICATE_UNLOCKS)
    assert "SCI004.M3" in certificate["successor_unlocks"]


def test_the_retained_certificate_bytes_carry_their_pinned_raw_digest() -> None:
    """Section 14.3's ``A3`` oracle authenticates "the raw stdout digest"."""
    raw, _parsed = read_retained_certificate()

    assert hashlib.sha256(raw).hexdigest() == RETAINED_CERTIFICATE_SHA256
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1


def test_the_retained_certificate_names_the_accepted_stage_two_artifacts() -> None:
    """The certificate's artifact joins resolve in Git at the commits it names."""
    _raw, certificate = read_retained_certificate()
    acceptance = _tree_blob(
        str(certificate["acceptance_commit_sha"]),
        str(certificate["acceptance_artifact_path"]),
    )
    evidence = _tree_blob(
        str(certificate["evidence_commit_sha"]),
        str(certificate["evidence_artifact_path"]),
    )

    assert (
        hashlib.sha256(acceptance).hexdigest()
        == certificate["acceptance_artifact_sha256"]
    )
    assert (
        hashlib.sha256(evidence).hexdigest() == certificate["evidence_artifact_sha256"]
    )
    assert _commit_parents(str(certificate["acceptance_commit_sha"])) == (
        str(certificate["evidence_commit_sha"]),
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("extra_field", "printed order"),
        ("missing_field", "printed order"),
        ("reordered_keys", "printed order"),
        ("spaced_separators", "canonical"),
        ("duplicate_key", "duplicate"),
        ("wrong_schema", "schema"),
        ("wrong_verdict", "verdict"),
        ("wrong_stage", "stage"),
        ("dropped_unlock", "successor_unlocks"),
        ("no_final_newline", "final LF"),
        ("two_lines", "exactly one line"),
        ("trailing_text", "not JSON"),
        ("short_digest", "64 lower-case hex"),
        ("upper_case_commit", "40 lower-case hex"),
        ("absolute_artifact_path", "repository-relative"),
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
        del document["evidence_commit_sha"]
    elif mutation == "wrong_schema":
        document["schema_version"] = "radiosim.sci005.other.v1"
    elif mutation == "wrong_verdict":
        document["verdict"] = "REJECT"
    elif mutation == "wrong_stage":
        document["stage"] = 3
    elif mutation == "dropped_unlock":
        document["successor_unlocks"] = ["SCI005.U2"]
    elif mutation == "short_digest":
        document["acceptance_artifact_sha256"] = "abc"
    elif mutation == "upper_case_commit":
        document["source_sha"] = str(document["source_sha"]).upper()
    elif mutation == "absolute_artifact_path":
        document["evidence_artifact_path"] = "/etc/passwd"

    if mutation == "reordered_keys":
        mutated = (
            json.dumps(
                dict(reversed(list(document.items()))),
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
    elif mutation == "spaced_separators":
        mutated = (json.dumps(document, ensure_ascii=False) + "\n").encode("utf-8")
    elif mutation == "duplicate_key":
        mutated = raw[:-2] + b',"verdict":"REJECT"}\n'
    elif mutation == "no_final_newline":
        mutated = raw[:-1]
    elif mutation == "two_lines":
        mutated = raw + raw
    elif mutation == "trailing_text":
        mutated = raw[:-1] + b" trailing\n"
    else:
        mutated = (
            json.dumps(document, separators=(",", ":"), ensure_ascii=False) + "\n"
        ).encode("utf-8")

    with pytest.raises(DependencyCertificateError, match=expected):
        parse_dependency_certificate(mutated)


def test_the_r3_replay_anchor_is_the_live_child_of_the_operative_correction() -> None:
    """Section 13.2's starred ``G3 -> R3`` edge, derived rather than declared.

    Before the re-cut ``R3`` exists the anchor is ``HEAD``, which the same rule
    forces to equal the operative ``D``; the two roles are named explicitly so
    that a reader of a passing run can tell which state produced it.  Either
    way the anchor is never the superseded red slice, which the correction
    reopened and which the ``--diff-filter=A`` derivation would have resolved
    forever.
    """
    anchor = resolve_r3_replay_anchor()

    assert anchor.role in ("pre-commit-authoring-tip", "r3")
    assert _peel_to_commit(anchor.commit) == anchor.commit
    assert anchor.commit not in (
        SUPERSEDED_RED_SLICE_SHA,
        SUPERSEDED_RECUT_RED_SLICE_SHA,
        SUPERSEDED_SECOND_RECUT_RED_SLICE_SHA,
        SUPERSEDED_THIRD_RECUT_RED_SLICE_SHA,
    )
    if anchor.role == "pre-commit-authoring-tip":
        assert anchor.commit == APPROVED_SCI004_D_SHA
    else:
        assert _commit_parents(anchor.commit) == (APPROVED_SCI004_D_SHA,)
        assert DEPENDENCY_VALIDATOR_PATH in _changed_paths(anchor.commit)
        assert _changed_paths(anchor.commit) == tuple(sorted(R3_AUTHORIZED_PATHS))


def test_the_starred_g3_to_r3_interval_is_exactly_the_enumerated_commits() -> None:
    """Section 13.7: "A commit the header does not name invalidates the edge".

    The exhaustive form of that sentence is an equality, not a membership test:
    the first-parent range ``G3..D`` must be exactly correction #25's eighteen
    commits, oldest-first, with nothing else in it. Five are superseded red
    slices, eight are superseded designs, two are superseded implementations,
    one is rejected evidence, one is rejected acceptance, and the final commit
    is the operative design. Every commit in the range is
    required to be a single-parent non-merge, which is Section 14.4's separate
    "No commit in either starred first-parent range is a merge".
    """
    expected = (
        SUPERSEDED_RED_SLICE_SHA,
        D16_SHA,
        D17_SHA,
        SUPERSEDED_RECUT_RED_SLICE_SHA,
        D18_SHA,
        D19_SHA,
        SUPERSEDED_SECOND_RECUT_RED_SLICE_SHA,
        D20_SHA,
        SUPERSEDED_THIRD_RECUT_RED_SLICE_SHA,
        D21_SHA,
        D22_SHA,
        SUPERSEDED_IMPLEMENTATION_SHA,
        D24_SHA,
        SUPERSEDED_FINGERPRINT_R3_SHA,
        SUPERSEDED_FINGERPRINT_S3_SHA,
        REJECTED_E3_SHA,
        REJECTED_A3_SHA,
        APPROVED_SCI004_D_SHA,
    )
    observed = tuple(
        _git(
            "rev-list",
            "--first-parent",
            "--reverse",
            f"{APPROVED_SCI004_G3_SHA}..{APPROVED_SCI004_D_SHA}",
        ).split()
    )

    assert observed == expected
    every_parent = tuple(
        _git(
            "rev-list",
            "--reverse",
            f"{APPROVED_SCI004_G3_SHA}..{APPROVED_SCI004_D_SHA}",
        ).split()
    )
    assert every_parent == expected

    previous = APPROVED_SCI004_G3_SHA
    for sha in expected:
        assert _peel_to_commit(sha) == sha, sha
        assert _commit_parents(sha) == (previous,), sha
        previous = sha

    assert _changed_paths(SUPERSEDED_THIRD_RECUT_RED_SLICE_SHA) == (
        SUPERSEDED_THIRD_RECUT_RED_SLICE_PATHS
    )
    assert _changed_paths(SUPERSEDED_IMPLEMENTATION_SHA) == (
        SUPERSEDED_IMPLEMENTATION_PATHS
    )
    assert _changed_paths(SUPERSEDED_FINGERPRINT_R3_SHA) == (
        SUPERSEDED_FINGERPRINT_R3_PATHS
    )
    assert _changed_paths(SUPERSEDED_FINGERPRINT_S3_SHA) == (
        SUPERSEDED_FINGERPRINT_S3_PATHS
    )
    assert _changed_paths(REJECTED_E3_SHA) == REJECTED_E3_PATHS
    assert _changed_paths(REJECTED_A3_SHA) == REJECTED_A3_PATHS
    for sha in (
        D16_SHA,
        D17_SHA,
        D18_SHA,
        D19_SHA,
        D20_SHA,
        D21_SHA,
        D22_SHA,
        D24_SHA,
        APPROVED_SCI004_D_SHA,
    ):
        assert _changed_paths(sha) == (DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH), sha


def test_the_memo_header_records_the_reopening_and_rebind_mandates() -> None:
    """Section 13.7: the header is the authority for what was reopened.

    A validator that trusted only its own constants would authenticate nothing:
    the operative ``D`` blob is read from Git and required to name *all four*
    superseded red slices it reopened, the four-family set that replaced the
    unconstructible fixtures, and both Section 8 rejection codes the first
    re-cut pinned.

    The two later landings add their own mandates, and they are checked from the
    same blob: the granted performance oracle by path, the ``R3``-list route
    the governance review substituted for the rejected ``S3`` grant, the
    conformed evidence surfaces -- four fingerprint rows and a ``ci_artifacts``
    object with no remote workflow field left in it -- and the honest backend
    axis, whose four ruled literals are what keep a NumPy run from being
    recorded under another runtime's label.
    """
    memo = _tree_blob(APPROVED_SCI004_D_SHA, DESIGN_MEMO_PATH).decode("utf-8")

    assert SUPERSEDED_RED_SLICE_SHA in memo
    assert SUPERSEDED_RECUT_RED_SLICE_SHA in memo
    assert SUPERSEDED_SECOND_RECUT_RED_SLICE_SHA in memo
    assert SUPERSEDED_THIRD_RECUT_RED_SLICE_SHA in memo
    assert SUPERSEDED_IMPLEMENTATION_SHA in memo
    assert "superseded red slice" in memo
    assert "mmode_circular_receptor" in memo
    assert "mmode_public_components" in memo
    assert "mmode_public_beam" in memo
    assert "--diff-filter=A" in memo
    assert "G3 ->* R3" in memo
    assert "process_rss_sampled_delta_v1" in memo
    assert "SCI004_PHASE3_POST_SOURCE_RED_CASES" in memo
    assert "post-source-expected-red-confirmed" in memo
    assert SUPERSEDED_FINGERPRINT_R3_SHA in memo
    assert SUPERSEDED_FINGERPRINT_S3_SHA in memo
    assert REJECTED_E3_SHA in memo
    assert REJECTED_A3_SHA in memo
    assert REJECTED_A3_ARTIFACT_SHA256 in memo
    assert REJECTED_E3_ARTIFACT_SHA256 in memo
    assert REJECTED_E3_REPRODUCTION_SHA256 in memo
    assert REJECTED_E3_PERFORMANCE_SHA256 in memo
    assert EXTERNAL_REVIEW_SHA256 in memo
    assert "canonical independent\n`A3` record" in memo
    assert "The rejected `E3`" in memo
    assert "reconstructible path-independent M3\nfingerprints" in memo
    assert "m3.fingerprint-input-preimage-not-retained" in memo
    assert "sci004-m3-independent-acceptance-reviewer" in memo
    assert "reviewer_independent=true" in memo

    # The retained-evidence correction's own rulings, in its own words.
    assert "tests/performance/test_sci004_mmode.py" in memo
    assert "a fresh `R`" in memo
    assert "R3_AUTHORIZED_PATHS" in memo

    # The conformed Section 14.2 body, sliced from its own headings so that the
    # header record's narrative quotation of the superseded literals cannot
    # satisfy an assertion about the ruling text.
    body = memo.split("### 14.2 ")[1].split("### 14.3 ")[0]
    assert "There are exactly four rows in the" in body
    assert "exactly seven rows" not in body
    assert "six CI-001 platform/Python cells" not in body
    for removed in ("`run_id`", "`job_id`", "`artifact_id`"):
        assert (
            f"{removed}, "
            not in body.split("Each `ci_artifacts` entry")[1].split("and `pass`.")[0]
        ), removed

    # The honest-backend-axis correction's ruled literals, sliced from Section
    # 11's own body for the same reason.  ``dense_execution`` and
    # ``kernel_backend_block`` are the row fields that separate the invariant
    # dense path from the two stages a backend genuinely computes;
    # ``dense_invariance`` is the top-level object that retains the measured
    # bit-identity as fact rather than hiding it; and the sixth claim literal is
    # the end-to-end execution this record must never assert.
    section_11 = memo.split("## 11. ")[1].split("## 12. ")[0]
    for literal in (
        "dense_execution",
        "kernel_backend_block",
        "dense_invariance",
        "mmode_end_to_end_backend_execution",
        "numpy_host_v1",
        "stage_comparison",
    ):
        assert literal in section_11, literal
    assert "`request.backend` reaches no dense array work" in section_11
    # The shared memory object may never borrow a backend-device method.
    assert "never the host RSS method or a\nbackend-device method" in section_11


def test_detached_worktree_replay_at_g3_reproduces_the_retained_certificate(
    certificate: Mapping[str, Any],
) -> None:
    """Section 13.2's ``G3`` replay, including its mandatory cleanup discipline."""
    before = _git("status", "--porcelain")

    stdout, elapsed = replay_stage2_certificate(APPROVED_SCI004_G3_SHA)

    raw, _parsed = read_retained_certificate()
    assert stdout == raw
    assert parse_dependency_certificate(stdout) == certificate
    assert elapsed >= 0.0
    assert _git("status", "--porcelain") == before
    assert not any(
        line.split()[0] == "worktree" and "sci004-m3-replay-" in line
        for line in _git("worktree", "list", "--porcelain").splitlines()
        if line.strip()
    )


def test_detached_worktree_replay_at_r3_reproduces_the_same_bytes(
    certificate: Mapping[str, Any],
) -> None:
    """Section 13.2: "The M3 validator additionally creates a clean detached
    worktree at exact ``R3``, runs the Stage-2 verifier with ``--descendant
    <R3>``, and requires the stdout bytes to be identical to the retained ``G3``
    line; the verifier output is descendant-independent while both ancestry
    checks must pass."

    Before ``R3`` is committed the anchor is the authoring tip ``G3`` and this
    replay repeats the first one exactly; from ``R3`` onwards it is the ruled
    descendant-independence check. Either way the bytes must be identical.
    """
    anchor = resolve_r3_replay_anchor()
    before = _git("status", "--porcelain")

    stdout, elapsed = replay_stage2_certificate(anchor.commit)

    raw, _parsed = read_retained_certificate()
    assert stdout == raw
    assert parse_dependency_certificate(stdout) == certificate
    assert elapsed >= 0.0
    assert _git("status", "--porcelain") == before
