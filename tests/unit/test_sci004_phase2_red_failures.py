"""Strict validator for the retained SCI-004 phase-M2 red-failure record.

``docs/development/sci004_mmode_design.md`` Section 14.1 fixes the schema of
``docs/development/sci004_mmode_phase2_red_failures.json`` and says the phase red
validator "authenticates the file bytes, schema literal, node set, command
hashes, pre-fix SHA, protected hashes, and expected non-zero outcomes **before**
``S`` is allowed to start". Section 14.0 additionally names *this file* as the
phase-M2 site of the frozen design binding: "R1's dependency validator,
``tests/unit/test_sci004_phase2_red_failures.py`` at R2, and R3's dependency
validator each freeze the exact assignment ``APPROVED_SCI004_D_SHA=...``, naming
the operative ``D`` current at that phase's ``R``. The later bindings byte-match
the R1 binding unless a Section 13.7 accepted correction intervened". No such
correction intervened between ``A1`` and this slice, so the binding below
byte-matches R1's and that equality is asserted rather than assumed.

Like its M1 counterpart this module deliberately **does not re-run the red
nodes**. They are red by construction at ``R2``; executing them here would fail
the suite a second time and would authenticate nothing the record does not
already carry. The record's own bytes are what get checked -- re-serialized under
Section 14's canonical rules and compared to the raw file -- with three
independent cross-checks that make the comparison meaningful:

* every case row's ``fixture_identity_sha256`` is recomputed from the Section
  14.0 preimage, so a row cannot claim a fixture it does not name;
* the node set is compared against the ``SCI004_PHASE2_RED_CASES`` tables the red
  modules themselves declare, so a node cannot be quietly dropped from the record
  while remaining red in the tree, and cannot appear twice; and
* the observation tree named by ``pre_fix_source_sha`` is read **from Git
  objects** and proved to lack every phase-M2 production capability the record
  claims is absent. Section 13.7 is explicit that an ``expected-red-confirmed``
  status must never be fabricated "against a tree where nothing is red", so the
  absence is authenticated rather than asserted.

**The phase unlock, and the starred ``A1 -> R2`` edge.** The retained M1
acceptance record carries ``acceptance_commit_sha = null`` with the reason
"self-reference: the next R or C binds the containing A commit". This slice is
that next ``R``, so it binds ``A1`` explicitly here: the commit is peeled,
required to be a single-parent non-merge whose parent is exactly ``E1``,
required to touch exactly the four paths Section 13.3 authorizes an ``A`` to
touch, and required to carry the accepted ``ACCEPT`` artifact at its retained
digest. That is the same shape ``tests/unit/test_sci004_phase1_dependency.py``
used to bind the upstream WP-7 acceptance for M1.

Authoring this slice proved Section 14.4's original ``R2^ == A1`` sole
direct-parent edge unsatisfiable, and the accepted bounded correction
``d8adeaaee1045b930fb7ca7e4bd0905655cd4725`` ruled it: Section 13.7 gains a
sixth interval-commit kind, ``post-acceptance repair``, the memo header
enumerates ``fea87708dd8bb4557a11970d4e350e66c58ca4d6`` and
``1d31baac111ec62ec45f73e355d8ad7b83b5fda8`` under it, and Section 14.4's
equation now reads ``A1 ->* R2`` with the rule that "``R2`` directly parents the
operative commit of the header's ``A1 -> R2`` post-acceptance-repairs
correction". That landing commit is the operative ``D`` bound below, so
``design_sha`` and ``pre_fix_source_sha`` are the same commit again and this
module asserts the *exact* edge rather than the interim ancestry weakening it
carried before the ruling.

Section 13.7 requires the next phase's red validator to authenticate each
enumerated repair "by full SHA and exact touched paths", and states that "a
commit the header does not name invalidates the edge". Both duties are
discharged here: the first-parent range ``A1..D`` is required to be *exactly*
the three enumerated commits, oldest-first, each a single-parent non-merge, with
the two repairs touching only ``tests/unit/test_sci004_phase1_acceptance.py``
and the landing touching only the two Section 13.1 design-authority paths a
correction may touch.

**Why this binding no longer byte-matches R1's.** Section 14.0 says the later
bindings byte-match the R1 binding "unless a Section 13.7 accepted correction
intervened, in which case the later binding names the newer operative ``D`` and
its validator authenticates the header-enumerated correction chain between the
two bindings". One did, so the equality is replaced by the chain-advance
authentication that clause prescribes: R1's frozen ``1712575e...`` is required
to still be the ``superseded design`` chain commit the M1 dependency validator
enumerates -- byte-identical memo blob, and R1's own constant unchanged, because
"no later phase may change those constants" -- and to reach the new operative
``D`` through ``A1`` plus the enumerated starred interval. Section 13.7 also
rules explicitly that ``A1`` itself "is not a chain commit and needs no interval
kind: its memo diff is exactly its own phase's Section 13 append-only acceptance
note", which is checked byte-wise here by prefix-composition against the
superseded ``D`` memo blob.

These tests pass at ``R2``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

#: The operative SCI-004 design commit ``D`` (Section 13.7), frozen for phase M2
#: exactly as Section 14.0 requires. It is the post-acceptance-repairs
#: correction landing, which supersedes R1's binding; Section 14.4 makes it the
#: commit ``R2`` directly parents.
APPROVED_SCI004_D_SHA = "d8adeaaee1045b930fb7ca7e4bd0905655cd4725"

#: The ``superseded design`` chain commit the operative ``D`` displaced -- and
#: R1's own frozen binding, which no later phase may change (Section 13.2).
SUPERSEDED_SCI004_D_SHA = "1712575e6c634457d9da737e9c144147e3b9bbc4"

#: The independently accepted phase-M1 acceptance commit ``A1`` (Section 14.4).
#: The retained M1 acceptance artifact's null self-SHA is bound by this
#: assignment: "the next R or C binds the containing A commit".
APPROVED_SCI004_A1_SHA = "445bc83edcf7073511c41b3485ad5d326d4e1552"

#: The phase-M1 evidence commit ``E1``. Section 14.4: each ``A`` directly parents
#: and names its ``E``, so this is ``A1^`` and the record's own
#: ``evidence_commit_sha``.
APPROVED_SCI004_E1_SHA = "dc736c692e4037e15b7e51253067fa262204bde2"

#: Section 13.7's ``post-acceptance repair`` interval commits, oldest-first,
#: exactly as the operative ``D`` memo header enumerates them on the starred
#: ``A1 -> R2`` edge. A commit in that range the header does not name
#: invalidates the edge, so this tuple is exhaustive by construction.
POST_ACCEPTANCE_REPAIR_SHAS: tuple[str, ...] = (
    "fea87708dd8bb4557a11970d4e350e66c58ca4d6",
    "1d31baac111ec62ec45f73e355d8ad7b83b5fda8",
)

#: The operative ``D`` memo blob, and the superseded one the M1 dependency
#: validator already pins under the same digest.
D_MEMO_BLOB_SHA256 = "15eb0b7dcf3800562443b38ba3276718f416f7e98956549e59bed0939a097efd"
SUPERSEDED_D_MEMO_BLOB_SHA256 = (
    "8bd62f986d8e152296ecf1a0370e487855e6fff067c9b5e6d85789e80d954d90"
)

#: The operative correction's exact *pre-landing* file bytes and parent-relative
#: diff. They were never committed -- the correction landed with its own header
#: record appended -- so the accepted header text is their only authority, which
#: is why they are read back out of the landed memo below rather than trusted.
D_PRE_LANDING_FILE_SHA256 = (
    "45b3a939ab588ce636bf29613cae3e582f73e5d5cb40935ac8cf40ee5f395646"
)
D_PRE_LANDING_DIFF_SHA256 = (
    "239b1a0fa127be99d41c407d936ded9686baa98e868ead4001e59b0e5ab125a1"
)

#: Section 13.7's kind literal for the two enumerated repairs.
POST_ACCEPTANCE_REPAIR_KIND = "post-acceptance repair"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

RECORD_PATH = "docs/development/sci004_mmode_phase2_red_failures.json"
M1_ACCEPTANCE_PATH = "docs/development/sci004_mmode_phase1_acceptance.json"
DESIGN_MEMO_PATH = "docs/development/sci004_mmode_design.md"
DESIGN_LEDGER_PATH = "PostTier8RemediationPlan.md"
M1_ACCEPTANCE_VALIDATOR_PATH = "tests/unit/test_sci004_phase1_acceptance.py"

#: The retained M1 acceptance artifact, by raw digest. The memo's own append-only
#: acceptance note records this exact value.
M1_ACCEPTANCE_SHA256 = (
    "19a8ca668e5cc0e29c54206f14c2cafc123b72e468effede1962db563d012002"
)

SCHEMA_VERSION = "radiosim.sci004.mmode-phase2-red-failures.v1"
PHASE = "M2"
STATUS = "expected-red-confirmed"
RED_COMMIT_SHA_REASON = "self-reference: E binds the containing R commit"

#: Section 14.1's exact top-level key set.
TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "phase",
        "status",
        "generated_at_utc",
        "design_sha",
        "pre_fix_source_sha",
        "red_commit_sha",
        "red_commit_sha_reason",
        "protected_source_clean",
        "authorized_red_paths",
        "environment",
        "cases",
        "commands",
        "claims_not_licensed",
    }
)

#: Section 14.1's exact sixteen-field case row.
CASE_KEYS: frozenset[str] = frozenset(
    {
        "case_id",
        "requirement_id",
        "test_nodeid",
        "invalid_config_raw_sha256",
        "fixture_identity_sha256",
        "expected_failure_kind",
        "expected_failure_pattern",
        "command_index",
        "exit_code",
        "observed_outcome",
        "observed_exception_type",
        "observed_message",
        "stdout_sha256",
        "stderr_sha256",
        "fixture_defect_excluded_by",
        "red_failure_confirmed",
    }
)

#: Section 14.1's exact eight-field command row.
COMMAND_KEYS: frozenset[str] = frozenset(
    {
        "argv",
        "cwd",
        "pixi_environment",
        "started_at_utc",
        "duration_seconds",
        "exit_code",
        "stdout_sha256",
        "stderr_sha256",
    }
)

#: Section 14.2's exact environment object.
ENVIRONMENT_KEYS: frozenset[str] = frozenset(
    {
        "python",
        "platform",
        "machine",
        "pixi_environment",
        "pixi_lock_sha256",
        "astropy_version",
        "erfa_version",
        "iers_package_version",
        "iers_table_sha256",
        "numeric_packages",
    }
)

NUMERIC_PACKAGES: frozenset[str] = frozenset(
    {"numpy", "scipy", "healpy", "jax", "dask"}
)

#: Section 14.1's five legal kinds.
FAILURE_KINDS: frozenset[str] = frozenset(
    {"assertion", "exception", "import", "missing-symbol", "schema"}
)

#: Section 13.4's complete ``R2`` writable list.
R2_AUTHORIZED_PATHS: frozenset[str] = frozenset(
    {
        "docs/development/sci004_mmode_phase2_red_failures.json",
        "tests/characterization/test_tier6_current_behavior.py",
        "tests/characterization/test_tier7_current_behavior.py",
        "tests/integration/test_sci004_mmode.py",
        "tests/performance/test_sci004_mmode.py",
        "tests/unit/test_backends/test_sci004_backend_parity.py",
        "tests/unit/test_core/test_sci004_direct_convergence.py",
        "tests/unit/test_core/test_sci004_polarization.py",
        "tests/unit/test_core/test_sci004_sky_harmonics.py",
        "tests/unit/test_core/test_sci004_transfer.py",
        "tests/unit/test_sci004_phase2_red_failures.py",
        "tests/unit/test_simulator/test_sci004_memory.py",
        "tools/sci004_mmode_phase2_red.py",
    }
)

#: Section 13.3's ``A`` grant, which the ``A1`` commit must touch exactly.
A1_AUTHORIZED_PATHS: tuple[str, ...] = (
    DESIGN_LEDGER_PATH,
    DESIGN_MEMO_PATH,
    M1_ACCEPTANCE_PATH,
    M1_ACCEPTANCE_VALIDATOR_PATH,
)

#: The red modules whose declared phase-M2 tables the node set is compared
#: against. The two characterization files carry no red-record machinery, so
#: their nodes are declared by the module that owns the requirement.
RED_MODULES: tuple[str, ...] = (
    "tests.unit.test_core.test_sci004_polarization",
    "tests.unit.test_core.test_sci004_sky_harmonics",
    "tests.unit.test_core.test_sci004_transfer",
    "tests.unit.test_core.test_sci004_direct_convergence",
    "tests.unit.test_backends.test_sci004_backend_parity",
    "tests.unit.test_simulator.test_sci004_memory",
    "tests.integration.test_sci004_mmode",
    "tests.performance.test_sci004_mmode",
)

#: Every red oracle file Section 13.4 authorizes must contribute at least one
#: node, and no other file may.
COVERED_FILES: frozenset[str] = frozenset(
    {
        "tests/characterization/test_tier7_current_behavior.py",
        "tests/integration/test_sci004_mmode.py",
        "tests/performance/test_sci004_mmode.py",
        "tests/unit/test_backends/test_sci004_backend_parity.py",
        "tests/unit/test_core/test_sci004_direct_convergence.py",
        "tests/unit/test_core/test_sci004_polarization.py",
        "tests/unit/test_core/test_sci004_sky_harmonics.py",
        "tests/unit/test_core/test_sci004_transfer.py",
        "tests/unit/test_simulator/test_sci004_memory.py",
    }
)

#: The four claim categories Section 14.1 requires.
REQUIRED_CLAIM_CATEGORIES: tuple[str, ...] = (
    "acceptance",
    "fingerprint",
    "performance",
    "production",
)

#: The phase-M2 capability-absence proof, read from Git objects at the exact
#: observation tree. Each entry is ``(path, sentinel)``: the sentinel is the
#: phase-M2 production text whose presence would mean the observation was not
#: genuinely red.
ABSENT_PHASE2_CAPABILITIES: tuple[tuple[str, str], ...] = (
    ("src/radiosim/core/mmode/harmonics.py", "def spin_ylm"),
    ("src/radiosim/core/mmode/harmonics.py", "def polarized_packed_block_table"),
    ("src/radiosim/core/mmode/harmonics.py", "def spin_transform_reference"),
    ("src/radiosim/core/mmode/sky.py", "def point_polarized_coefficients"),
    ("src/radiosim/core/mmode/sky.py", "def healpix_polarized_coefficients"),
    ("src/radiosim/core/mmode/sky.py", "def hybrid_polarized_coefficients"),
    ("src/radiosim/core/mmode/transfer.py", "def build_polarized_baseline_transfer"),
    ("src/radiosim/core/mmode/solver.py", "def contract_per_m_block"),
    ("src/radiosim/core/mmode/solver.py", "def synthesize_time_series"),
    ("src/radiosim/core/mmode/solver.py", "def estimate_mmode_memory"),
    ("src/radiosim/core/mmode/solver.py", "def schedule_mmode_blocks"),
    ("src/radiosim/core/mmode/solver.py", "def solve_polarized_fixture"),
    ("src/radiosim/core/polarization.py", "def shaw_basis_bridge"),
    ("src/radiosim/core/sky/containers/__init__.py", "TangentPolarizationFrame"),
    ("src/radiosim/benchmarks/__init__.py", "SCI004_BENCHMARK_SCHEMA_VERSION"),
)

#: The M1 capability literal that must still be in the observation tree: the
#: whole phase-M2 slice exists to flip it.
M1_SCALAR_CAPABILITY = "supports_polarization = False"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_UTC_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class RedRecordSchemaError(AssertionError):
    """The retained phase-M2 red-failure record failed strict validation."""


# --- Section 14 canonical serialization, re-derived independently -------------


def _es_number(value: float | int) -> str:
    if isinstance(value, bool):
        raise RedRecordSchemaError("a boolean is not a JSON number")
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise RedRecordSchemaError("NaN and Infinity are forbidden")
    if value == int(value) and abs(value) < 2**53:
        return str(int(value))
    decimal = Decimal(repr(float(value)))
    exponent = decimal.adjusted()
    if -6 <= exponent <= 20:
        text = format(decimal, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text
    digits = format(decimal.scaleb(-exponent), "f").rstrip("0").rstrip(".")
    sign = "+" if exponent >= 0 else "-"
    return f"{digits}e{sign}{abs(exponent)}"


def canonical_json_bytes(value: Any) -> bytes:
    """Section 14's ``J``: sorted keys, ``,``/``:``, ASCII, no whitespace or LF."""
    return _canonical(value).encode("utf-8")


def _canonical(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _es_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, Mapping):
        return (
            "{"
            + ",".join(
                f"{json.dumps(key, ensure_ascii=True)}:{_canonical(item)}"
                for key, item in sorted(value.items(), key=lambda pair: pair[0])
            )
            + "}"
        )
    if isinstance(value, Sequence):
        return "[" + ",".join(_canonical(item) for item in value) + "]"
    raise RedRecordSchemaError(f"cannot canonicalize {type(value).__name__}")


def domain_digest(domain: str, payload: bytes) -> str:
    """Section 14.0's ``D(d, p) = SHA256(d || NUL || U64(len(p)) || p)``."""
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + struct.pack(">Q", len(payload)) + payload
    ).hexdigest()


# --- Git object access --------------------------------------------------------


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RedRecordSchemaError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def _peel_to_commit(revision: str) -> str:
    return _git("rev-parse", "--verify", f"{revision}^{{commit}}").strip()


def _commit_parents(revision: str) -> tuple[str, ...]:
    listing = _git("rev-list", "--parents", "-n", "1", revision).split()
    return tuple(listing[1:])


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def _tree_blob(commit: str, relative: str) -> bytes:
    listing = _git("ls-tree", commit, "--", relative).split()
    if not listing:
        raise RedRecordSchemaError(f"{relative} is absent from {commit}")
    return subprocess.run(
        ["git", "cat-file", "blob", listing[2]],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
    ).stdout


def _changed_paths(commit: str) -> tuple[str, ...]:
    listing = _git(
        "diff-tree", "--no-commit-id", "--name-only", "-r", "-z", commit
    ).split("\0")
    return tuple(sorted(entry for entry in listing if entry))


# --- loading ------------------------------------------------------------------


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise RedRecordSchemaError(f"duplicate JSON key {key!r}")
        parsed[key] = value
    return parsed


def _reject_non_finite(_value: str) -> float:
    raise RedRecordSchemaError("NaN and Infinity are forbidden in a canonical record")


def read_record() -> tuple[bytes, dict[str, Any]]:
    path = REPOSITORY_ROOT / RECORD_PATH
    if not path.is_file() or path.is_symlink():
        raise RedRecordSchemaError(f"{RECORD_PATH} must be a retained regular file")
    raw = path.read_bytes()
    if raw.endswith(b"\n"):
        raise RedRecordSchemaError("canonical record bytes carry no trailing newline")
    document = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite,
    )
    if not isinstance(document, dict):
        raise RedRecordSchemaError("the record must be a JSON object")
    return raw, document


@pytest.fixture(scope="module")
def record() -> dict[str, Any]:
    _raw, document = read_record()
    return document


@pytest.fixture(scope="module")
def declared_nodes() -> dict[str, dict[str, Any]]:
    """The phase-M2 case tables the red test modules themselves declare."""
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    declared: dict[str, dict[str, Any]] = {}
    for module_name in RED_MODULES:
        module = importlib.import_module(module_name)
        for case in module.SCI004_PHASE2_RED_CASES:
            nodeid = str(case["test_nodeid"])
            if nodeid in declared:
                raise RedRecordSchemaError(f"{nodeid} is declared twice")
            declared[nodeid] = dict(case)
    return declared


# --- Section 14.0: the frozen design binding ----------------------------------


def test_this_module_freezes_exactly_one_assignment_per_binding() -> None:
    """Section 13.2/14.0: one assignment each, in the one authorised file."""
    source = (
        REPOSITORY_ROOT / "tests/unit/test_sci004_phase2_red_failures.py"
    ).read_text(encoding="utf-8")
    for name in (
        "APPROVED_SCI004_D_SHA",
        "SUPERSEDED_SCI004_D_SHA",
        "APPROVED_SCI004_A1_SHA",
        "APPROVED_SCI004_E1_SHA",
    ):
        assignments = re.findall(
            rf"^{name} = \"[0-9a-f]{{40}}\"$", source, re.MULTILINE
        )
        assert len(assignments) == 1, name
    for value in (
        APPROVED_SCI004_D_SHA,
        SUPERSEDED_SCI004_D_SHA,
        APPROVED_SCI004_A1_SHA,
        APPROVED_SCI004_E1_SHA,
        *POST_ACCEPTANCE_REPAIR_SHAS,
    ):
        assert _SHA1.match(value) is not None
    assert len(set(POST_ACCEPTANCE_REPAIR_SHAS)) == len(POST_ACCEPTANCE_REPAIR_SHAS)


def test_the_phase_two_design_binding_advances_the_r1_binding_by_a_correction() -> None:
    """Section 14.0: the chain-advance branch, because a correction intervened.

    "The later bindings byte-match the R1 binding unless a Section 13.7 accepted
    correction intervened, in which case the later binding names the newer
    operative ``D`` and its validator authenticates the header-enumerated
    correction chain between the two bindings."  One did -- the
    post-acceptance-repairs correction -- so the byte-match is *not* asserted.
    What is asserted instead is everything that clause substitutes for it: R1's
    frozen constant is unchanged ("no later phase may change those constants"),
    it is still the last entry of the chain R1 enumerates, that entry's recorded
    kind is still what R1 recorded, and the two bindings differ.
    """
    from tests.unit.test_sci004_phase1_dependency import (
        APPROVED_SCI004_D_SHA as R1_APPROVED_SCI004_D_SHA,
    )
    from tests.unit.test_sci004_phase1_dependency import (
        SCI004_DESIGN_CHAIN,
    )

    assert R1_APPROVED_SCI004_D_SHA == SUPERSEDED_SCI004_D_SHA
    assert APPROVED_SCI004_D_SHA != R1_APPROVED_SCI004_D_SHA
    assert SCI004_DESIGN_CHAIN[-1].sha == SUPERSEDED_SCI004_D_SHA
    assert SCI004_DESIGN_CHAIN[-1].memo_blob_sha256 == SUPERSEDED_D_MEMO_BLOB_SHA256
    # R1's chain range stops at its own binding, so the later correction cannot
    # retroactively appear inside a range that phase already froze.
    interval = _git(
        "rev-list",
        "--first-parent",
        f"{SUPERSEDED_SCI004_D_SHA}..{APPROVED_SCI004_D_SHA}",
    ).split()
    assert APPROVED_SCI004_D_SHA in interval
    assert SUPERSEDED_SCI004_D_SHA not in interval


def test_the_operative_design_commit_peels_and_descends_from_the_m1_acceptance() -> (
    None
):
    """Section 13.7/14.4: the operative ``D`` now *follows* ``A1``.

    The correction landed after the phase it unblocks was accepted, which is the
    whole shape Section 13.7's gate-anchor and starred-edge rules exist for, so
    the ancestry direction is the reverse of the pre-correction one and is
    asserted in that direction rather than assumed.
    """
    assert _peel_to_commit(APPROVED_SCI004_D_SHA) == APPROVED_SCI004_D_SHA
    parents = _commit_parents(APPROVED_SCI004_D_SHA)

    assert len(parents) == 1, "the operative D is a single-parent non-merge"
    assert _is_ancestor(APPROVED_SCI004_A1_SHA, APPROVED_SCI004_D_SHA)
    assert _is_ancestor(SUPERSEDED_SCI004_D_SHA, APPROVED_SCI004_D_SHA)
    assert not _is_ancestor(APPROVED_SCI004_D_SHA, APPROVED_SCI004_A1_SHA)


def test_the_starred_a1_to_d_edge_interval_is_exactly_the_enumerated_commits() -> None:
    """Section 13.7: "A commit the header does not name invalidates the edge".

    The exhaustive form of that sentence is an equality, not a membership test:
    the first-parent range ``A1..D`` must be *exactly* the enumerated repairs
    followed by the correction landing, oldest-first, with nothing else in it.
    Every commit in the range is also required to be a single-parent non-merge,
    which is Section 14.4's separate "No commit in either starred first-parent
    range is a merge".
    """
    expected = (*POST_ACCEPTANCE_REPAIR_SHAS, APPROVED_SCI004_D_SHA)
    observed = tuple(
        _git(
            "rev-list",
            "--first-parent",
            "--reverse",
            f"{APPROVED_SCI004_A1_SHA}..{APPROVED_SCI004_D_SHA}",
        ).split()
    )

    assert observed == expected
    # The full range, following every parent, must equal the first-parent range:
    # a side branch merged in would satisfy the latter while smuggling in commits
    # the header never enumerated.
    every_parent = tuple(
        _git(
            "rev-list",
            "--reverse",
            f"{APPROVED_SCI004_A1_SHA}..{APPROVED_SCI004_D_SHA}",
        ).split()
    )
    assert every_parent == expected

    previous = APPROVED_SCI004_A1_SHA
    for sha in expected:
        assert _peel_to_commit(sha) == sha, sha
        assert _commit_parents(sha) == (previous,), sha
        previous = sha


def test_every_enumerated_post_acceptance_repair_touches_only_its_phase_validator() -> (
    None
):
    """Section 13.7: the sixth kind's exact path grant, by full SHA.

    A ``post-acceptance repair`` "may touch only that phase's Section 13 tool and
    validator test paths -- never production source, never a retained artifact,
    never this memo -- and the next phase's red validator authenticates each
    enumerated repair by full SHA and exact touched paths".  Both repairs touch
    exactly the M1 acceptance validator, so the negative half of that sentence is
    a consequence of the positive half rather than a second, looser check.
    """
    for sha in POST_ACCEPTANCE_REPAIR_SHAS:
        assert _changed_paths(sha) == (M1_ACCEPTANCE_VALIDATOR_PATH,), sha
        # The two things a repair may never touch, stated explicitly because the
        # design states them explicitly.
        assert DESIGN_MEMO_PATH not in _changed_paths(sha), sha
        assert M1_ACCEPTANCE_PATH not in _changed_paths(sha), sha


def test_the_operative_correction_landing_touches_only_the_design_authority() -> None:
    """Section 13.7: a bounded correction is design-only, and lands unchanged.

    "A bounded design correction is drafted as edits to
    ``docs/development/sci004_mmode_design.md`` alone -- plus
    ``PostTier8RemediationPlan.md`` WP-9/Q5/dependency/ledger wording when the
    correction changes a fact that ledger states."  The landing's parent-relative
    diff is required to be exactly those two paths.
    """
    assert _changed_paths(APPROVED_SCI004_D_SHA) == (
        DESIGN_LEDGER_PATH,
        DESIGN_MEMO_PATH,
    )
    blob = _tree_blob(APPROVED_SCI004_D_SHA, DESIGN_MEMO_PATH)
    assert hashlib.sha256(blob).hexdigest() == D_MEMO_BLOB_SHA256


def test_the_memo_header_enumerates_the_starred_edge_interval_exhaustively() -> None:
    """Section 13.7: the header must enumerate every interval commit by SHA.

    A validator that trusted its own constants would authenticate nothing: the
    authority is the accepted memo text, so the operative ``D`` blob is read from
    Git and required to name both repairs in full 40-hex form, the sixth kind's
    literal, the commit it supersedes, and its own pinned pre-landing review
    digests -- which were never committed, making the accepted header their only
    authority.
    """
    header = _tree_blob(APPROVED_SCI004_D_SHA, DESIGN_MEMO_PATH).decode("utf-8")

    for sha in POST_ACCEPTANCE_REPAIR_SHAS:
        assert sha in header, sha
    assert POST_ACCEPTANCE_REPAIR_KIND in header
    assert SUPERSEDED_SCI004_D_SHA in header
    assert APPROVED_SCI004_A1_SHA in header
    assert D_PRE_LANDING_FILE_SHA256 in header
    assert D_PRE_LANDING_DIFF_SHA256 in header
    # Section 14.4's amended equation and its ``R^`` restatement.
    assert "A1 ->* R2" in header
    assert "R2^==A1" not in header.replace("`R2^==A1`", "")


def test_the_superseded_design_memo_and_the_a1_note_prefix_compose() -> None:
    """Section 13.7/14.0: ``A1`` is not a chain commit; its memo diff is the note.

    "An accepted phase acceptance commit inside the ``D0 -> D`` range is not a
    chain commit and needs no interval kind: its memo diff is exactly its own
    phase's Section 13 append-only acceptance note, authenticated by its own
    phase machinery under the Section 14.0 rule that tools authenticate the
    operative ``D`` blob plus the separately authorized ``A`` diffs."  The
    mechanical statement of that sentence is prefix-composition: the superseded
    ``D`` memo blob -- R1's frozen binding, which this phase does not change --
    is a byte prefix of ``A1``'s, and the appended tail is the acceptance note
    and nothing else.
    """
    superseded = _tree_blob(SUPERSEDED_SCI004_D_SHA, DESIGN_MEMO_PATH)
    at_acceptance = _tree_blob(APPROVED_SCI004_A1_SHA, DESIGN_MEMO_PATH)

    assert hashlib.sha256(superseded).hexdigest() == SUPERSEDED_D_MEMO_BLOB_SHA256
    assert at_acceptance.startswith(superseded)
    assert len(at_acceptance) > len(superseded)

    appended = at_acceptance[len(superseded) :].decode("utf-8")
    assert appended.lstrip().startswith("## Acceptance notes (append-only)")
    assert "Phase M1 accepted" in appended
    assert "SCI-004" in appended and "ROADMAP" in appended
    # An append-only note adds; it never rewrites what it follows.
    assert "\n## " not in appended[appended.index("(append-only)") :]

    # The operative ``D`` is *not* a further append: it amended Sections 13.7 and
    # 14.4 in place, which is exactly why it needed its own dual review rather
    # than an acceptance note.
    operative = _tree_blob(APPROVED_SCI004_D_SHA, DESIGN_MEMO_PATH)
    assert not operative.startswith(at_acceptance)
    assert len(operative) > len(at_acceptance)


# --- Section 14.4: the A1 unlock ----------------------------------------------


def test_the_m1_acceptance_commit_is_a_single_parent_child_of_e1() -> None:
    """Section 14.4: "Each ``A`` directly parents and names ``E``"."""
    assert _peel_to_commit(APPROVED_SCI004_A1_SHA) == APPROVED_SCI004_A1_SHA
    parents = _commit_parents(APPROVED_SCI004_A1_SHA)

    assert parents == (APPROVED_SCI004_E1_SHA,)
    assert len(_commit_parents(APPROVED_SCI004_E1_SHA)) == 1


def test_the_m1_acceptance_commit_touches_exactly_its_authorized_paths() -> None:
    """Section 13.3/14.4: an ``A`` changes only its artifact, constants and prose."""
    assert _changed_paths(APPROVED_SCI004_A1_SHA) == A1_AUTHORIZED_PATHS


def test_the_retained_m1_acceptance_artifact_is_an_accept_at_its_exact_digest() -> None:
    """Section 14.3: the accepted artifact, read from the ``A1`` tree object.

    Its ``acceptance_commit_sha`` is null with the self-reference reason, which
    is precisely why this module exists: the phase-M2 red slice is "the next R"
    the reason names, and binding ``A1`` here is what closes that reference.
    """
    blob = _tree_blob(APPROVED_SCI004_A1_SHA, M1_ACCEPTANCE_PATH)
    assert hashlib.sha256(blob).hexdigest() == M1_ACCEPTANCE_SHA256
    assert not blob.endswith(b"\n")

    artifact = json.loads(
        blob.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    assert artifact["schema_version"] == "radiosim.sci004.mmode-phase1-acceptance.v1"
    assert artifact["phase"] == "M1"
    assert artifact["verdict"] == "ACCEPT"
    # Section 13.7: "An accepted artifact is immutable and no commit may touch
    # one."  It therefore still names the ``D`` operative at its own phase's
    # ``R`` -- the commit this phase's correction superseded -- and requiring it
    # to name the new operative ``D`` would demand editing it.
    assert artifact["design_sha"] == SUPERSEDED_SCI004_D_SHA
    assert artifact["design_sha"] != APPROVED_SCI004_D_SHA
    assert artifact["evidence_commit_sha"] == APPROVED_SCI004_E1_SHA
    assert artifact["acceptance_commit_sha"] is None
    assert artifact["acceptance_commit_sha_reason"] == (
        "self-reference: the next R or C binds the containing A commit"
    )
    assert artifact["reviewer_independent"] is True

    # The same bytes are still in the working tree; an accepted artifact is
    # immutable and no later commit may touch one.
    assert (REPOSITORY_ROOT / M1_ACCEPTANCE_PATH).read_bytes() == blob


def test_the_m1_acceptance_licenses_no_polarized_capability() -> None:
    """Section 9: M1 licenses the scalar registry entry only.

    The phase-M2 red slice is red *because* that is true, so the claim array of
    the accepted M1 artifact is read here rather than paraphrased.
    """
    artifact = json.loads(
        _tree_blob(APPROVED_SCI004_A1_SHA, M1_ACCEPTANCE_PATH).decode("utf-8")
    )
    claims = artifact["claims_not_licensed"]

    assert isinstance(claims, list) and claims
    assert claims == sorted(set(claims))
    joined = " ".join(claims).lower()
    assert "polariz" in joined
    for forbidden in ("speed", "accelerator", "gpu"):
        assert forbidden in joined, forbidden


# --- Section 14.1: the record's own bytes -------------------------------------


def test_the_record_is_exactly_its_canonical_serialization() -> None:
    """Section 14: sorted keys, ``,``/``:``, ASCII, no whitespace or trailing LF."""
    raw, document = read_record()

    assert canonical_json_bytes(document) == raw


def test_the_record_carries_the_exact_top_level_key_set(
    record: dict[str, Any],
) -> None:
    """Section 14.1's frozen envelope, at the phase-M2 schema literal."""
    assert set(record) == TOP_LEVEL_KEYS
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["phase"] == PHASE
    assert record["status"] == STATUS
    assert _UTC_STAMP.match(str(record["generated_at_utc"])) is not None


def test_the_record_binds_the_frozen_design_and_its_exact_observation_tree(
    record: dict[str, Any],
) -> None:
    """Section 14.0/14.4: ``design_sha`` is the binding, and ``R2`` parents it.

    Section 14.4 as corrected rules that "``R2`` directly parents the operative
    commit of the header's ``A1 -> R2`` post-acceptance-repairs correction", so
    the tree the observations were made from *is* the operative ``D`` and the two
    fields are the same commit -- the same shape the M1 record has, restored by
    the ruling. The equality is asserted exactly rather than through the interim
    ancestry weakening this module carried before the correction landed.
    """
    assert record["design_sha"] == APPROVED_SCI004_D_SHA

    observed = str(record["pre_fix_source_sha"])
    assert _SHA1.match(observed) is not None
    assert observed == APPROVED_SCI004_D_SHA
    assert _peel_to_commit(observed) == observed
    # The starred edge stands between the observation tree and ``A1``; its
    # exhaustive authentication lives in its own node above.
    assert _is_ancestor(APPROVED_SCI004_A1_SHA, observed)


def test_the_m1_acceptance_commit_is_not_a_chain_commit() -> None:
    """Section 13.7: an accepted phase acceptance commit needs no interval kind.

    "An accepted phase acceptance commit inside the ``D0 -> D`` range is not a
    chain commit and needs no interval kind."  ``A1`` therefore must not appear
    in the ``D0 -> D`` chain the M1 dependency validator enumerates, nor among
    the two ``post-acceptance repair`` commits the starred edge names -- it is
    the edge's *origin*, not a member of its interval.
    """
    from tests.unit.test_sci004_phase1_dependency import SCI004_DESIGN_CHAIN

    chain = {entry.sha for entry in SCI004_DESIGN_CHAIN}
    assert APPROVED_SCI004_A1_SHA not in chain
    assert APPROVED_SCI004_A1_SHA not in POST_ACCEPTANCE_REPAIR_SHAS
    assert APPROVED_SCI004_D_SHA not in chain
    # It is an ``A``: it carries its phase's acceptance artifact and its memo
    # diff is the append-only note, both proved from Git objects elsewhere here.
    assert M1_ACCEPTANCE_PATH in _changed_paths(APPROVED_SCI004_A1_SHA)


def test_the_observation_tree_genuinely_lacks_every_phase_two_capability(
    record: dict[str, Any],
) -> None:
    """Section 13.7: an ``expected-red-confirmed`` status is never fabricated.

    "A record regenerated against such a tree would fabricate
    ``expected-red-confirmed`` observations."  The converse obligation is
    discharged here: every phase-M2 production capability the record's cases
    claim is absent is proved absent from the observation tree's own Git blobs,
    and the M1 scalar-only capability literal the slice exists to flip is proved
    present.
    """
    observed = str(record["pre_fix_source_sha"])
    blobs: dict[str, str] = {}
    for relative, sentinel in ABSENT_PHASE2_CAPABILITIES:
        if relative not in blobs:
            blobs[relative] = _tree_blob(observed, relative).decode("utf-8")
        assert sentinel not in blobs[relative], (relative, sentinel)

    simulator = _tree_blob(observed, "src/radiosim/simulator/mmode.py").decode("utf-8")
    assert M1_SCALAR_CAPABILITY in simulator
    assert "supports_polarization = True" not in simulator


def test_the_self_reference_is_a_null_sha_with_its_exact_reason(
    record: dict[str, Any],
) -> None:
    """Section 14.1/14.4: an ``R`` artifact uses a null self SHA, bound by ``E``."""
    assert record["red_commit_sha"] is None
    assert record["red_commit_sha_reason"] == RED_COMMIT_SHA_REASON


def test_the_protected_source_is_declared_clean_and_the_diff_is_authorized(
    record: dict[str, Any],
) -> None:
    """Section 14.1: an uncommitted red artifact never claims a clean whole tree.

    The generator hashes every protected path outside Section 13.4's ``R2`` list
    before and after execution and records only the authorized diff, so
    ``protected_source_clean`` is a statement about the *protected* set.
    """
    assert record["protected_source_clean"] is True

    paths = record["authorized_red_paths"]
    assert isinstance(paths, list)
    assert all(isinstance(path, str) and path for path in paths)
    assert paths == sorted(set(paths))
    assert set(paths) <= R2_AUTHORIZED_PATHS
    assert RECORD_PATH in paths


def test_the_environment_object_is_exactly_section_14_2s(
    record: dict[str, Any],
) -> None:
    """Section 14.2's environment, including its five numeric packages."""
    environment = record["environment"]

    assert set(environment) == ENVIRONMENT_KEYS
    assert _SHA256.match(str(environment["pixi_lock_sha256"])) is not None
    assert _SHA256.match(str(environment["iers_table_sha256"])) is not None
    for field in (
        "python",
        "platform",
        "machine",
        "pixi_environment",
        "astropy_version",
        "erfa_version",
        "iers_package_version",
    ):
        assert isinstance(environment[field], str) and environment[field], field

    packages = environment["numeric_packages"]
    assert set(packages) == NUMERIC_PACKAGES
    assert list(packages) == sorted(packages)
    for name, version in packages.items():
        assert isinstance(version, str) and version, name


def test_the_claims_not_licensed_array_is_sorted_unique_and_complete(
    record: dict[str, Any],
) -> None:
    """Section 14.1: production, acceptance, fingerprint and performance claims."""
    claims = record["claims_not_licensed"]

    assert isinstance(claims, list) and claims
    assert all(isinstance(claim, str) and claim for claim in claims)
    assert claims == sorted(set(claims))
    for category in REQUIRED_CLAIM_CATEGORIES:
        assert any(claim.startswith(f"{category}:") for claim in claims), category


# --- Section 14.1: the command rows -------------------------------------------


def test_every_command_row_has_its_exact_shape_and_a_non_zero_exit(
    record: dict[str, Any],
) -> None:
    """Section 14.1: ``argv`` runs without a shell from the repository root."""
    commands = record["commands"]

    assert isinstance(commands, list) and commands
    for index, command in enumerate(commands):
        assert set(command) == COMMAND_KEYS, index
        argv = command["argv"]
        assert isinstance(argv, list) and argv
        assert all(isinstance(entry, str) and entry for entry in argv)
        assert command["cwd"] == "."
        assert isinstance(command["pixi_environment"], str)
        assert command["pixi_environment"]
        assert _UTC_STAMP.match(str(command["started_at_utc"])) is not None
        duration = command["duration_seconds"]
        assert isinstance(duration, (int, float)) and not isinstance(duration, bool)
        assert math.isfinite(float(duration)) and float(duration) >= 0.0
        assert isinstance(command["exit_code"], int)
        assert not isinstance(command["exit_code"], bool)
        assert command["exit_code"] != 0, index
        assert _SHA256.match(str(command["stdout_sha256"])) is not None
        assert _SHA256.match(str(command["stderr_sha256"])) is not None


# --- Section 14.1: the case rows ----------------------------------------------


def test_every_case_row_has_its_exact_sixteen_field_shape(
    record: dict[str, Any],
) -> None:
    """Section 14.1's frozen case row, field for field."""
    cases = record["cases"]

    assert isinstance(cases, list) and cases
    assert len(CASE_KEYS) == 16
    for case in cases:
        assert set(case) == CASE_KEYS, case.get("case_id")


def test_every_case_records_a_confirmed_non_zero_red_outcome(
    record: dict[str, Any],
) -> None:
    """Section 14.1: a skipped, xfailed, passed or unrelated outcome is invalid."""
    commands = record["commands"]

    for case in record["cases"]:
        identifier = case["case_id"]
        assert case["expected_failure_kind"] in FAILURE_KINDS, identifier
        assert case["observed_outcome"] in FAILURE_KINDS, identifier
        assert case["observed_outcome"] == case["expected_failure_kind"], identifier
        assert isinstance(case["observed_exception_type"], str)
        assert case["observed_exception_type"], identifier
        assert "." in case["observed_exception_type"], identifier
        assert isinstance(case["observed_message"], str)
        assert case["observed_message"], identifier
        assert (
            re.search(str(case["expected_failure_pattern"]), case["observed_message"])
            is not None
        ), identifier
        assert isinstance(case["fixture_defect_excluded_by"], str)
        assert case["fixture_defect_excluded_by"], identifier
        assert case["red_failure_confirmed"] is True, identifier

        index = case["command_index"]
        assert isinstance(index, int) and not isinstance(index, bool)
        assert 0 <= index < len(commands), identifier
        assert case["exit_code"] == commands[index]["exit_code"], identifier
        assert case["exit_code"] != 0, identifier
        assert case["stdout_sha256"] == commands[index]["stdout_sha256"], identifier
        assert case["stderr_sha256"] == commands[index]["stderr_sha256"], identifier


def test_every_case_id_is_a_phase_two_identifier(record: dict[str, Any]) -> None:
    """The M1 record's identifiers are retained and untouched; these are new.

    Every phase-M2 case identifier is namespaced ``m2.``, so the two retained
    records can never be confused for one another and a copied M1 row cannot
    silently satisfy a phase-M2 requirement.
    """
    for case in record["cases"]:
        assert str(case["case_id"]).startswith("m2."), case["case_id"]
        assert str(case["requirement_id"]).startswith("sci004.section-"), case[
            "case_id"
        ]


def test_every_case_fixture_identity_recomputes_from_its_section_14_0_preimage(
    record: dict[str, Any],
) -> None:
    """Section 14.0: the identity covers exactly six fields, including raw bytes."""
    for case in record["cases"]:
        assert _SHA256.match(str(case["invalid_config_raw_sha256"])) is not None
        expected = domain_digest(
            "radiosim.sci004-red-fixture.v1",
            canonical_json_bytes(
                {
                    "phase": record["phase"],
                    "fixture_id": case["case_id"],
                    "requirement_id": case["requirement_id"],
                    "test_nodeid": case["test_nodeid"],
                    "pre_fix_source_sha": record["pre_fix_source_sha"],
                    "invalid_config_raw_sha256": case["invalid_config_raw_sha256"],
                }
            ),
        )
        assert case["fixture_identity_sha256"] == expected, case["case_id"]


def test_every_phase_red_node_appears_exactly_once_and_lives_in_the_r2_list(
    record: dict[str, Any],
    declared_nodes: dict[str, dict[str, Any]],
) -> None:
    """Section 14.1: the record's node set equals the declared red inventory."""
    nodes = [str(case["test_nodeid"]) for case in record["cases"]]
    identifiers = [str(case["case_id"]) for case in record["cases"]]

    assert len(set(nodes)) == len(nodes)
    assert len(set(identifiers)) == len(identifiers)
    assert set(nodes) == set(declared_nodes)
    for nodeid in nodes:
        relative, separator, name = nodeid.partition("::")
        assert separator == "::" and name, nodeid
        assert relative in R2_AUTHORIZED_PATHS, nodeid
        assert (REPOSITORY_ROOT / relative).is_file(), nodeid


def test_every_case_agrees_with_the_declaration_its_red_module_carries(
    record: dict[str, Any],
    declared_nodes: dict[str, dict[str, Any]],
) -> None:
    """The record cannot restate a requirement or fixture the node does not own."""
    for case in record["cases"]:
        declared = declared_nodes[str(case["test_nodeid"])]
        assert case["case_id"] == declared["case_id"]
        assert case["requirement_id"] == declared["requirement_id"]
        assert case["expected_failure_kind"] == declared["expected_failure_kind"]
        assert case["expected_failure_pattern"] == declared["expected_failure_pattern"]
        assert (
            case["fixture_defect_excluded_by"] == declared["fixture_defect_excluded_by"]
        )
        assert (
            case["invalid_config_raw_sha256"]
            == hashlib.sha256(declared["fixture_bytes"]).hexdigest()
        )


def test_the_record_covers_every_authorized_red_test_file(
    record: dict[str, Any],
) -> None:
    """Every red oracle file Section 13.4 authorizes contributes at least one node."""
    covered = {str(case["test_nodeid"]).split("::", 1)[0] for case in record["cases"]}

    assert covered == COVERED_FILES


def test_the_retained_m1_red_record_is_untouched_by_this_slice() -> None:
    """Section 13.7: a phase may not edit a previous phase's retained artifact.

    The M1 red record was retained at its last genuinely observed bytes under the
    post-source retention rule; this slice adds a second record beside it and
    changes nothing about the first.
    """
    m1_path = REPOSITORY_ROOT / "docs/development/sci004_mmode_phase1_red_failures.json"
    assert m1_path.is_file()

    working = m1_path.read_bytes()
    committed = _tree_blob(
        "HEAD", "docs/development/sci004_mmode_phase1_red_failures.json"
    )
    assert working == committed

    m1_document = json.loads(working.decode("utf-8"))
    _raw, m2_document = read_record()
    assert m1_document["phase"] == "M1"
    assert m2_document["phase"] == "M2"
    assert m1_document["schema_version"] != m2_document["schema_version"]


# --- the strict serializer refuses the mutations that matter ------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, "1"),
        (0, "0"),
        (1.0, "1"),
        (-2.0, "-2"),
        (0.5, "0.5"),
        (1.25e-3, "0.00125"),
        (1e-7, "1e-7"),
        (1e21, "1e+21"),
    ],
)
def test_the_canonical_number_rule_is_rfc_8785_shortest_round_trip(
    value: float | int, expected: str
) -> None:
    """Section 14: ``1.0`` and ``1e0`` are not canonical record bytes."""
    assert _es_number(value) == expected


@pytest.mark.parametrize(
    "payload",
    [float("nan"), float("inf"), float("-inf")],
)
def test_the_canonical_serializer_refuses_non_finite_numbers(payload: float) -> None:
    """Section 14: NaN and Infinity are forbidden."""
    with pytest.raises(RedRecordSchemaError):
        canonical_json_bytes({"value": payload})


def test_a_reordered_or_prettified_document_is_not_the_canonical_bytes() -> None:
    """Section 14: the record's identity is its exact serialization, not its data."""
    raw, document = read_record()

    assert json.dumps(document, indent=2).encode("utf-8") != raw
    assert (
        json.dumps(document, sort_keys=True, ensure_ascii=True).encode("utf-8") != raw
    )
