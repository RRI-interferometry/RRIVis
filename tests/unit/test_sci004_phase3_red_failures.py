"""Strict validator for the retained SCI-004 phase-M3 red-failure record.

``docs/development/sci004_mmode_design.md`` Section 14.1 fixes the schema of
``docs/development/sci004_mmode_phase3_red_failures.json`` and says the phase red
validator "authenticates the file bytes, schema literal, node set, command
hashes, pre-fix SHA, protected hashes, and expected non-zero outcomes **before**
``S`` is allowed to start".

The three immutable records are authenticated from retained bytes and Git
objects. The fingerprint partition is additionally replayed in an exact-SHA
detached verification fixture: two governed assertion failures followed by
three passing controls. The general characterization oracle runs separately
and must pass. Record authentication uses three independent cross-checks:

* every case row's ``fixture_identity_sha256`` is recomputed from the Section
  14.0 preimage, so a row cannot claim a fixture it does not name;
* the node set is compared against the ``SCI004_PHASE3_RED_CASES`` tables the
  red modules themselves declare, so a node cannot be quietly dropped from the
  record while remaining red in the tree, and cannot appear twice; and
* the observation tree named by ``pre_fix_source_sha`` is read **from Git
  objects** and proved to lack every phase-M3 output capability the record
  claims is absent. Section 13.7 is explicit that an ``expected-red-confirmed``
  status must never be fabricated "against a tree where nothing is red", so the
  absence is authenticated rather than asserted.

**The design binding, and where it lives.** Section 14.0 names R3's *dependency*
validator as this phase's single site for ``APPROVED_SCI004_D_SHA``, so this
module imports the binding from ``tests/unit/test_sci004_phase3_dependency.py``
rather than freezing a second copy of it. The same file holds the ``A2``, ``E2``,
``G3`` and SCI-005 Stage-2 bindings, and authenticates the header-enumerated
correction chain and both worktree replays; what is left to this module is the
record.

**The phase unlock.** The retained M2 acceptance record carries
``acceptance_commit_sha = null`` with the reason "self-reference: the next R or C
binds the containing A commit". This slice is that next ``R``; the dependency
validator binds ``A2`` in full, and the two facts this module needs from that
binding -- that the observation tree is the gate tip and that the gate tip is the
accepted unlock -- are asserted here against the record's own fields.

These tests pass at ``R3``.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import site
import stat
import struct
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest

from tests.unit.test_sci004_phase3_dependency import (
    APPROVED_SCI004_A2_SHA,
    APPROVED_SCI004_D_SHA,
    APPROVED_SCI004_G3_SHA,
    D24_SHA,
    D25_SHA,
    D26_SHA,
    D27_SHA,
    D28_SHA,
    D29_SHA,
    D30_R1_TERMINAL_SHA,
    D30_SHA,
    D30_STATUS_BRIDGE_SHA,
    DESIGN_LEDGER_PATH,
    DESIGN_MEMO_PATH,
    OLD_FRESH_VALIDATOR_R3_PATHS,
    OLD_FRESH_VALIDATOR_R3_SHA,
    ORIGINAL_FINGERPRINT_R3_SHA,
    REJECTED_A3_SHA,
    REJECTED_E3_SHA,
    SUPERSEDED_FINGERPRINT_R3_SHA,
    SUPERSEDED_FINGERPRINT_S3_SHA,
    resolve_r3_replay_anchor,
)
from tools import sci004_phase3_history as phase_history

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

RECORD_PATH = "docs/development/sci004_mmode_phase3_red_failures.json"
POST_SOURCE_RECORD_PATH = (
    "docs/development/sci004_mmode_phase3_post_source_red_failures.json"
)
FINGERPRINT_POST_SOURCE_RECORD_PATH = (
    "docs/development/sci004_mmode_phase3_fingerprint_post_source_red_failures.json"
)
M2_RED_RECORD_PATH = "docs/development/sci004_mmode_phase2_red_failures.json"
CERTIFICATE_PATH = "docs/development/sci004_mmode_phase3_sci005_dependency.json"
HISTORICAL_RED_SLICE_SHA = "7070cc3ddb1c2557d02e4a3f2a89b907575bed0b"
HISTORICAL_DESIGN_SHA = "923ae332c02d9b2d4edfddf09d1d61241e9d5a63"
POST_SOURCE_PRE_FIX_SHA = "a61526d686ab768f05ecffa80cfd6223d4ee4c62"
HISTORICAL_RED_RECORD_SHA256 = (
    "486705a8d5e51c08f972c91aeae60f0a0bfeef5480b622515282295a6a3cde05"
)
POST_SOURCE_ORACLE_PATH = "tests/unit/test_io/test_hdf5_result.py"
FINGERPRINT_POST_SOURCE_ORACLE_PATH = "tests/characterization/test_sci004_mmode.py"
CORRECTION24_POST_SOURCE_RED_RECORD_SHA256 = (
    "724f75c246ebfcf5956fc40fb2f5e349d91ccca3e6a188b3785a65f4ae4c1e10"
)

SCHEMA_VERSION = "radiosim.sci004.mmode-phase3-red-failures.v1"
PHASE = "M3"
STATUS = "expected-red-confirmed"
RED_COMMIT_SHA_REASON = "self-reference: E binds the containing R commit"
POST_SOURCE_RED_COMMIT_SHA_REASON = (
    "self-reference: E binds the containing post-source R commit"
)
POST_SOURCE_SCHEMA_VERSION = "radiosim.sci004.mmode-phase3-post-source-red-failures.v1"
POST_SOURCE_STATUS = "post-source-expected-red-confirmed"
FINGERPRINT_POST_SOURCE_RED_COMMIT_SHA_REASON = (
    "self-reference: E binds the containing fingerprint-retry R3 commit"
)
FINGERPRINT_POST_SOURCE_SCHEMA_VERSION = (
    "radiosim.sci004.mmode-phase3-fingerprint-post-source-red-failures.v1"
)

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

POST_SOURCE_TOP_LEVEL_KEYS: frozenset[str] = TOP_LEVEL_KEYS | {
    "historical_red_record_sha256",
    "oracle_patch_paths",
    "oracle_patch_sha256",
}
FINGERPRINT_POST_SOURCE_TOP_LEVEL_KEYS: frozenset[str] = POST_SOURCE_TOP_LEVEL_KEYS | {
    "passing_controls",
    "correction24_post_source_red_record_sha256",
}

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

PASSING_CONTROL_KEYS: frozenset[str] = frozenset(
    {
        "control_id",
        "requirement_id",
        "purpose",
        "test_nodeid",
        "command_index",
        "observed_outcome",
        "exit_code",
        "pass",
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

#: Section 13.5's complete ``R3`` writable list.
HISTORICAL_R3_AUTHORIZED_PATHS: frozenset[str] = frozenset(
    {
        "docs/development/sci004_mmode_phase3_red_failures.json",
        "docs/development/sci004_mmode_phase3_sci005_dependency.json",
        "tests/characterization/test_sci004_mmode.py",
        "tests/unit/test_io/test_hdf5_result.py",
        "tests/unit/test_io/test_measurement_set.py",
        "tests/unit/test_io/test_result_summary.py",
        "tests/unit/test_io/test_standard_visibility.py",
        "tests/unit/test_io/test_uvfits.py",
        "tests/unit/test_sci004_phase3_dependency.py",
        "tests/unit/test_sci004_phase3_red_failures.py",
        "tests/unit/test_tier8_release_acceptance.py",
        "tools/sci004_mmode_phase3_red.py",
        # Granted by the accepted retained-evidence correction, "scoped to the
        # pinned fixture-product literals only ... a red-oracle edit, so it
        # belongs to an ``R`` commit".
        "tests/performance/test_sci004_mmode.py",
    }
)

R3_AUTHORIZED_PATHS: frozenset[str] = frozenset(
    {
        POST_SOURCE_RECORD_PATH,
        POST_SOURCE_ORACLE_PATH,
        "tests/unit/test_sci004_phase3_dependency.py",
        "tests/unit/test_sci004_phase3_red_failures.py",
        "tools/sci004_mmode_phase3_red.py",
    }
)

FINGERPRINT_R3_AUTHORIZED_PATHS: frozenset[str] = frozenset(
    {
        FINGERPRINT_POST_SOURCE_RECORD_PATH,
        FINGERPRINT_POST_SOURCE_ORACLE_PATH,
        "tests/unit/test_sci004_phase3_dependency.py",
        "tests/unit/test_sci004_phase3_red_failures.py",
        "tools/sci004_mmode_phase3_red.py",
    }
)

FINGERPRINT_SUPPLEMENT_SHA256 = (
    "6bf1cf94b30961fd7a27519fad1252169155fdeee0e81618ea15115b50fbdb68"
)

#: The red modules whose declared phase-M3 tables the node set is compared
#: against.
RED_MODULES: tuple[str, ...] = (
    "tests.unit.test_io.test_standard_visibility",
    "tests.unit.test_io.test_hdf5_result",
    "tests.unit.test_io.test_result_summary",
    "tests.unit.test_io.test_uvfits",
    "tests.unit.test_io.test_measurement_set",
    "tests.characterization.test_sci004_mmode",
    "tests.unit.test_tier8_release_acceptance",
)

#: Every red oracle file Section 13.5 authorizes must contribute at least one
#: node, and no other file may.
COVERED_FILES: frozenset[str] = frozenset(
    {
        "tests/characterization/test_sci004_mmode.py",
        "tests/performance/test_sci004_mmode.py",
        "tests/unit/test_io/test_hdf5_result.py",
        "tests/unit/test_io/test_measurement_set.py",
        "tests/unit/test_io/test_result_summary.py",
        "tests/unit/test_io/test_standard_visibility.py",
        "tests/unit/test_io/test_uvfits.py",
        "tests/unit/test_tier8_release_acceptance.py",
    }
)

#: The four claim categories Section 14.1 requires.
REQUIRED_CLAIM_CATEGORIES: tuple[str, ...] = (
    "acceptance",
    "fingerprint",
    "performance",
    "production",
)

POST_SOURCE_CLAIMS_NOT_LICENSED: tuple[str, ...] = (
    "acceptance: this supplemental record expresses no phase-M3 acceptance "
    "verdict and unlocks no successor commit",
    "fingerprint: this supplement pins, harvests, and adjudicates no Section 11 "
    "m-mode family or dispatch-class observation",
    "performance: this supplement measures no timing, speedup, memory, or "
    "accelerator behavior and licenses no performance claim",
    "production: this supplement records only the six hostile HDF5-reader "
    "rejection defects and certifies neither broader output behavior nor "
    "production readiness",
)

FINGERPRINT_POST_SOURCE_CLAIMS_NOT_LICENSED: tuple[str, ...] = (
    "acceptance: this fingerprint supplement expresses no phase-M3 acceptance "
    "verdict and unlocks no successor commit",
    "fingerprint: this supplement retains two expected-red fingerprint defects "
    "but pins, harvests, and adjudicates no Section 11 m-mode family or "
    "dispatch-class observation",
    "performance: this supplement measures no timing, speedup, memory, or "
    "accelerator behavior and licenses no performance claim",
    "production: this supplement records only the absent reconstructible "
    "path-independent characterization-input surface and certifies neither "
    "broader output behavior nor production readiness",
)

FINGERPRINT_NODEIDS: tuple[str, ...] = (
    "tests/characterization/test_sci004_mmode.py::"
    "test_characterization_input_preimage_is_retained_and_reconstructible",
    "tests/characterization/test_sci004_mmode.py::"
    "test_characterization_input_identity_is_equal_under_distinct_layout_roots",
    "tests/characterization/test_sci004_mmode.py::"
    "test_every_new_family_records_its_six_section_11_parts"
    "[mmode_single_scalar_mode]",
    "tests/characterization/test_sci004_mmode.py::"
    "test_distinct_layout_roots_preserve_scientific_and_cube_identities",
    "tests/characterization/test_sci004_mmode.py::"
    "test_characterization_input_identity_changes_for_semantic_instrument_content",
)

GENERAL_TRANSITION_NODEID = (
    "tests/characterization/test_sci004_mmode.py::"
    "test_a_family_pin_is_a_ci001_observation_set_not_a_bare_digest"
)

FINGERPRINT_CASE_EXPECTATIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "m3.fingerprint.preimage-retained",
        "SCI-004-14.2-M3-FINGERPRINT-PREIMAGE",
        FINGERPRINT_NODEIDS[0],
        "characterization input manifest is absent from the family record",
    ),
    (
        "m3.fingerprint.path-independent",
        "SCI-004-11-PATH-INDEPENDENT-CHARACTERIZATION",
        FINGERPRINT_NODEIDS[1],
        "characterization input identity changed under filesystem relocation",
    ),
)

FINGERPRINT_CONTROL_EXPECTATIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "m3.fingerprint.family-record-schema",
        "SCI-004-11-FAMILY-RECORD-SCHEMA",
        FINGERPRINT_NODEIDS[2],
        "exact domain-discriminated family-record schema and all pre-existing "
        "family joins remain valid",
    ),
    (
        "m3.fingerprint.relocation-science-control",
        "SCI-004-11-PATH-INDEPENDENT-CHARACTERIZATION",
        FINGERPRINT_NODEIDS[3],
        "relocation fixture preserves independently derived scientific and "
        "raw-cube identities",
    ),
    (
        "m3.fingerprint.semantic-separation-control",
        "SCI-004-11-SEMANTIC-INPUT-SEPARATION",
        FINGERPRINT_NODEIDS[4],
        "semantic antenna-layout mutation changes characterization input identity",
    ),
)

#: The phase-M3 capability-absence proof, read from Git objects at the exact
#: observation tree. Each entry is ``(path, sentinel)``: the sentinel is the
#: phase-M3 production or documentation text whose presence would mean the
#: observation was not genuinely red.
ABSENT_PHASE3_CAPABILITIES: tuple[tuple[str, str], ...] = (
    ("src/radiosim/core/result.py", "def mmode_characterization_record"),
    ("src/radiosim/core/result.py", "MMODE_CHARACTERIZATION_FAMILIES"),
    ("src/radiosim/core/result.py", "def mmode_characterization_observation_set"),
    ("src/radiosim/io/standard_visibility.py", "mmode"),
    ("src/radiosim/io/uvfits.py", "mmode"),
    ("src/radiosim/io/measurement_set.py", "mmode"),
    ("docs/api/io.rst", "mmode"),
    ("docs/user_guide/configuration_support.rst", "mmode"),
    # The two Section 8 public-path guards the accepted correction ruled. Their
    # absence is what the two rejection oracles record.
    ("src/radiosim/core/mmode/solver.py", "mmode_public_components"),
    ("src/radiosim/core/mmode/solver.py", "mmode_public_beam"),
    # The amended Section 11 performance product. The benchmark surface at this
    # observation tree still enumerates the superseded trio, which is precisely
    # why the granted performance oracle is red here.
    ("src/radiosim/benchmarks/__init__.py", "mmode_single_scalar_mode"),
    # The honest-backend-axis row schema. The same surface still carries the
    # thirty-three-key row and the five-literal claim array, so neither ruled
    # literal exists there yet -- the second granted oracle's red reason.
    ("src/radiosim/benchmarks/__init__.py", "dense_execution"),
    ("src/radiosim/benchmarks/__init__.py", "kernel_backend_block"),
    ("src/radiosim/benchmarks/__init__.py", "mmode_end_to_end_backend_execution"),
)

#: The two defects the output oracles measure, present at the observation tree.
#: A red slice that could not point at them would be describing a capability
#: nobody has demanded rather than one the writers get wrong.
PRESENT_PHASE3_DEFECTS: tuple[tuple[str, str], ...] = (
    (
        "src/radiosim/io/standard_visibility.py",
        'components = ",".join(solver.components)',
    ),
    (
        "src/radiosim/io/summary_json.py",
        "minimum of cadence_seconds and remaining observation duration",
    ),
    # The measured root cause of the accepted correction's narrowing: the public
    # solve path builds a point component and nothing else, which is why a
    # HEALPix-bearing sky publishes an identically zero cube instead of being
    # refused.
    ("src/radiosim/core/mmode/solver.py", 'components=("point",)'),
    # The superseded product the granted oracle still pins at this tree.
    ("src/radiosim/benchmarks/__init__.py", "mmode_healpix_full_stokes"),
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_UTC_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class RedRecordSchemaError(AssertionError):
    """The retained phase-M3 red-failure record failed strict validation."""


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


def _red_git_environment() -> dict[str, str]:
    """Keep caller Git routing and object overlays outside raw authentication."""
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_GRAFT_FILE=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_ATTR_NOSYSTEM="1",
    )
    return environment


def _red_git_process(
    arguments: Sequence[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Read original objects and path inventories from the explicit repository."""
    if arguments[0] in {"diff", "diff-tree"}:
        command, *rest = arguments
        arguments = (
            command,
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=none",
            "--find-renames" if command == "diff" else "--no-renames",
            *rest,
        )
    elif arguments[0] == "show":
        arguments = ("show", "--no-ext-diff", "--no-textconv", *arguments[1:])
    return subprocess.run(
        [
            "git",
            "--no-pager",
            "--no-replace-objects",
            "--literal-pathspecs",
            "-c",
            "core.commitGraph=false",
            "-c",
            "core.attributesFile=" + os.devnull,
            "-c",
            "color.ui=false",
            *arguments,
        ],
        cwd=(REPOSITORY_ROOT if cwd is None else cwd).resolve(),
        env=_red_git_environment(),
        capture_output=True,
        check=False,
    )


def _git(*arguments: str) -> str:
    completed = _red_git_process(arguments)
    if completed.returncode != 0:
        raise RedRecordSchemaError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.decode('utf-8', 'replace').strip()}"
        )
    return completed.stdout.decode("utf-8")


def _peel_to_commit(revision: str) -> str:
    return _git("rev-parse", "--verify", f"{revision}^{{commit}}").strip()


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    completed = _red_git_process(("merge-base", "--is-ancestor", ancestor, descendant))
    if completed.returncode not in (0, 1):
        raise RedRecordSchemaError("raw Git ancestry query failed")
    return completed.returncode == 0


def _tree_blob(commit: str, relative: str) -> bytes:
    listing = _git("ls-tree", "-z", commit, "--", relative).split("\0")
    if len(listing) != 2 or listing[-1] != "" or "\t" not in listing[0]:
        raise RedRecordSchemaError(f"{relative} is absent or ambiguous at {commit}")
    metadata, path = listing[0].split("\t", 1)
    fields = metadata.split()
    if (
        len(fields) != 3
        or fields[0] not in {"100644", "100755"}
        or fields[1] != "blob"
        or path != relative
    ):
        raise RedRecordSchemaError(f"{relative} is not a regular blob at {commit}")
    completed = _red_git_process(("cat-file", "blob", fields[2]))
    if completed.returncode != 0:
        raise RedRecordSchemaError(f"cannot read raw blob {relative} at {commit}")
    return completed.stdout


def _native_red_git(root: Path, *arguments: str, data: bytes | None = None) -> bytes:
    """Create/control hostile Git state only in the synthetic test repository."""
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="Synthetic raw Git test",
        GIT_AUTHOR_EMAIL="raw@example.invalid",
        GIT_COMMITTER_NAME="Synthetic raw Git test",
        GIT_COMMITTER_EMAIL="raw@example.invalid",
    )
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        env=environment,
        input=data,
        capture_output=True,
        check=True,
    ).stdout


def _raw_red_commit(root: Path, parent: str | None, payload: bytes) -> str:
    blob = (
        _native_red_git(root, "hash-object", "-w", "--stdin", data=payload)
        .decode()
        .strip()
    )
    tree = (
        _native_red_git(
            root, "mktree", data=f"100644 blob {blob}\toracle.py\n".encode()
        )
        .decode()
        .strip()
    )
    arguments = ["commit-tree", tree]
    if parent is not None:
        arguments.extend(("-p", parent))
    return (
        _native_red_git(root, *arguments, data=b"synthetic commit\n").decode().strip()
    )


@pytest.fixture
def raw_red_git_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str, str]:
    root = tmp_path / "objects.git"
    root.mkdir()
    _ = _native_red_git(root, "init", "--bare", ".")
    parent = _raw_red_commit(root, None, b"before\n")
    child = _raw_red_commit(root, parent, b"after\n")
    monkeypatch.setattr(sys.modules[__name__], "REPOSITORY_ROOT", root)
    return root, parent, child


@pytest.mark.parametrize("overlay", ["commit", "blob", "graft", "graft-env"])
def test_raw_red_git_reads_original_objects_and_parents(
    raw_red_git_repository: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
    overlay: str,
) -> None:
    root, parent, child = raw_red_git_repository
    expected = _git("diff", "--binary", "--full-index", parent, child)
    if overlay in {"commit", "blob"}:
        left, right = child, parent
        if overlay == "blob":
            left = (
                _native_red_git(root, "rev-parse", f"{child}:oracle.py")
                .decode()
                .strip()
            )
            right = (
                _native_red_git(root, "rev-parse", f"{parent}:oracle.py")
                .decode()
                .strip()
            )
        _ = _native_red_git(root, "replace", left, right)
        assert _native_red_git(root, "show", f"{child}:oracle.py") == b"before\n"
    else:
        graft = root / "info/grafts"
        if overlay == "graft-env":
            graft = root / "outside-graft"
            monkeypatch.setenv("GIT_GRAFT_FILE", str(graft))
        _ = graft.write_text(f"{child}\n")
        native = subprocess.run(
            ["git", "rev-list", "--parents", "-n", "1", child],
            cwd=root,
            capture_output=True,
            check=True,
        )
        assert native.stdout.split() == [child.encode()]
    assert _tree_blob(child, "oracle.py") == b"after\n"
    assert _git("rev-list", "--parents", "-n", "1", child).split() == [child, parent]
    assert _git("diff", "--binary", "--full-index", parent, child) == expected
    assert _is_ancestor(parent, child) and not _is_ancestor(child, parent)
    with pytest.raises(RedRecordSchemaError, match="ancestry"):
        _ = _is_ancestor("0" * 40, child)


@pytest.mark.parametrize(
    "variable",
    [
        "GIT_DIR",
        "GIT_COMMON_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ],
)
def test_raw_red_git_ignores_caller_repository_routing(
    raw_red_git_repository: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    root, parent, child = raw_red_git_repository
    monkeypatch.setenv(variable, str(root / "missing-routing-target"))
    assert _tree_blob(child, "oracle.py") == b"after\n"
    assert _is_ancestor(parent, child)
    assert _git("diff", "--name-only", parent, child) == "oracle.py\n"
    assert os.environ[variable] == str(root / "missing-routing-target")


def test_raw_red_git_path_inventory_cannot_hide_gitlinks(
    raw_red_git_repository: tuple[Path, str, str],
) -> None:
    root, _, child = raw_red_git_repository
    entries = _native_red_git(root, "ls-tree", child)
    entries += f"160000 commit {child}\tforbidden-module\n".encode()
    tree = _native_red_git(root, "mktree", data=entries).decode().strip()
    bad = (
        _native_red_git(root, "commit-tree", tree, "-p", child, data=b"gitlink\n")
        .decode()
        .strip()
    )
    _ = _native_red_git(root, "config", "diff.ignoreSubmodules", "all")
    assert _native_red_git(root, "diff", "--name-only", child, bad) == b""
    assert _git("diff", "--name-only", child, bad) == "forbidden-module\n"
    assert (
        _git("diff-tree", "--no-commit-id", "--name-only", "-r", bad)
        == "forbidden-module\n"
    )


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


def _read_record_path(relative: str) -> tuple[bytes, dict[str, Any]]:
    path = REPOSITORY_ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise RedRecordSchemaError(f"{relative} must be a retained regular file")
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


def read_record() -> tuple[bytes, dict[str, Any]]:
    return _read_record_path(RECORD_PATH)


def read_post_source_record() -> tuple[bytes, dict[str, Any]]:
    return _read_record_path(POST_SOURCE_RECORD_PATH)


def read_fingerprint_post_source_record() -> tuple[bytes, dict[str, Any]]:
    return _read_record_path(FINGERPRINT_POST_SOURCE_RECORD_PATH)


@pytest.fixture(scope="module")
def record() -> dict[str, Any]:
    _raw, document = read_record()
    return document


@pytest.fixture(scope="module")
def post_source_record() -> dict[str, Any]:
    _raw, document = read_post_source_record()
    return document


@pytest.fixture(scope="module")
def fingerprint_post_source_record() -> dict[str, Any]:
    _raw, document = read_fingerprint_post_source_record()
    return document


@pytest.fixture(scope="module")
def declared_nodes() -> dict[str, dict[str, Any]]:
    """The phase-M3 case tables the red test modules themselves declare."""
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    declared: dict[str, dict[str, Any]] = {}
    for module_name in RED_MODULES:
        module = importlib.import_module(module_name)
        for case in module.SCI004_PHASE3_RED_CASES:
            nodeid = str(case["test_nodeid"])
            if nodeid in declared:
                raise RedRecordSchemaError(f"{nodeid} is declared twice")
            declared[nodeid] = dict(case)
    return declared


@pytest.fixture(scope="module")
def post_source_declared_nodes() -> dict[str, dict[str, Any]]:
    """Correction #24's six separately declared hostile HDF5 nodes."""
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    module = importlib.import_module("tests.unit.test_io.test_hdf5_result")
    declared: dict[str, dict[str, Any]] = {}
    for case in module.SCI004_PHASE3_POST_SOURCE_RED_CASES:
        nodeid = str(case["test_nodeid"])
        if nodeid in declared:
            raise RedRecordSchemaError(f"{nodeid} is declared twice")
        declared[nodeid] = dict(case)
    if len(declared) != 6:
        raise RedRecordSchemaError("the post-source table must contain six nodes")
    return declared


# --- Section 14.0: the frozen design binding ----------------------------------


def test_this_module_freezes_no_binding_of_its_own() -> None:
    """Section 14.0 names exactly one site for this phase's bindings.

    "R1's dependency validator, ``tests/unit/test_sci004_phase2_red_failures.py``
    at R2, and R3's dependency validator each freeze the exact assignment
    ``APPROVED_SCI004_D_SHA=...``."  For M3 that site is the dependency
    validator, so this module imports the value; a second assignment here would
    be a silently divergent binding.
    """
    source = Path(__file__).read_text(encoding="utf-8")

    assert re.search(r"^APPROVED_[A-Z0-9_]+ = ", source, re.MULTILINE) is None
    assert _SHA1.match(APPROVED_SCI004_D_SHA) is not None
    assert _SHA1.match(APPROVED_SCI004_G3_SHA) is not None
    assert _SHA1.match(APPROVED_SCI004_A2_SHA) is not None


# --- Section 14.1: the record's own bytes -------------------------------------


def test_the_record_is_exactly_its_canonical_serialization() -> None:
    """Section 14: sorted keys, ``,``/``:``, ASCII, no whitespace or trailing LF."""
    raw, document = read_record()

    assert canonical_json_bytes(document) == raw


def test_the_record_carries_the_exact_top_level_key_set(
    record: dict[str, Any],
) -> None:
    """Section 14.1's frozen envelope, at the phase-M3 schema literal."""
    assert set(record) == TOP_LEVEL_KEYS
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["phase"] == PHASE
    assert record["status"] == STATUS
    assert _UTC_STAMP.match(str(record["generated_at_utc"])) is not None


def test_the_record_binds_the_frozen_design_and_its_exact_observation_tree(
    record: dict[str, Any],
) -> None:
    """Section 14.0/14.4: ``design_sha`` is the binding, and the edge is starred.

    Section 14.4 now reads ``G3 ->* R3``: the accepted correction that reopened
    the first phase-3 red slice makes the re-cut ``R3`` directly parent that
    correction's landing, so the tree the observations were made from is the
    operative ``D`` and the two fields are the same commit.  ``G3`` stands
    behind it through the enumerated interval the dependency validator
    authenticates exhaustively.
    """
    assert record["design_sha"] == HISTORICAL_DESIGN_SHA
    assert HISTORICAL_DESIGN_SHA != APPROVED_SCI004_D_SHA

    observed = str(record["pre_fix_source_sha"])
    assert _SHA1.match(observed) is not None
    assert observed == HISTORICAL_DESIGN_SHA
    assert _peel_to_commit(observed) == observed
    assert _is_ancestor(APPROVED_SCI004_G3_SHA, observed)
    assert _is_ancestor(APPROVED_SCI004_A2_SHA, observed)
    assert _is_ancestor(observed, APPROVED_SCI004_D_SHA)


def test_the_historical_record_is_retained_byte_for_byte_from_its_last_red_slice() -> (
    None
):
    """Section 13.7: S already exists, so the genuine historical bytes remain."""
    raw, _document = read_record()

    assert hashlib.sha256(raw).hexdigest() == HISTORICAL_RED_RECORD_SHA256
    assert raw == _tree_blob(HISTORICAL_RED_SLICE_SHA, RECORD_PATH)


def test_the_observation_tree_genuinely_lacks_every_phase_three_capability(
    record: dict[str, Any],
) -> None:
    """Section 13.7: an ``expected-red-confirmed`` status is never fabricated.

    Every phase-M3 output capability the record's cases claim is absent is
    proved absent from the observation tree's own Git blobs, and the two
    concrete defects the output oracles measure are proved present -- a red
    slice that could point at neither would be describing a capability nobody
    has demanded rather than one the writers get wrong.
    """
    observed = str(record["pre_fix_source_sha"])
    blobs: dict[str, str] = {}
    for relative, sentinel in ABSENT_PHASE3_CAPABILITIES:
        if relative not in blobs:
            blobs[relative] = _tree_blob(observed, relative).decode("utf-8")
        assert sentinel not in blobs[relative], (relative, sentinel)
    for relative, sentinel in PRESENT_PHASE3_DEFECTS:
        if relative not in blobs:
            blobs[relative] = _tree_blob(observed, relative).decode("utf-8")
        assert sentinel in blobs[relative], (relative, sentinel)


def test_the_self_reference_is_a_null_sha_with_its_exact_reason(
    record: dict[str, Any],
) -> None:
    """Section 14.1/14.4: an ``R`` artifact uses a null self SHA, bound by ``E``."""
    assert record["red_commit_sha"] is None
    assert record["red_commit_sha_reason"] == RED_COMMIT_SHA_REASON


def test_the_protected_source_is_declared_clean_and_the_diff_is_authorized(
    record: dict[str, Any],
) -> None:
    """Section 14.1: an uncommitted red artifact never claims a clean whole tree."""
    assert record["protected_source_clean"] is True

    paths = record["authorized_red_paths"]
    assert isinstance(paths, list)
    assert all(isinstance(path, str) and path for path in paths)
    assert paths == sorted(set(paths))
    assert set(paths) <= HISTORICAL_R3_AUTHORIZED_PATHS
    assert RECORD_PATH in paths
    # The retained SCI-005 certificate is *not* in this list, and must not be:
    # the superseded red slice already committed those exact bytes and the
    # governed re-cut does not touch them, so they are unchanged rather than
    # newly authorized.  Section 13.2's retention requirement is a statement
    # about the bytes, which is what is checked here.
    assert CERTIFICATE_PATH not in paths
    retained = REPOSITORY_ROOT / CERTIFICATE_PATH
    assert retained.is_file() and not retained.is_symlink()
    assert retained.read_bytes() == _tree_blob("HEAD", CERTIFICATE_PATH)


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


def test_every_case_id_is_a_phase_three_identifier(record: dict[str, Any]) -> None:
    """The M1 and M2 identifiers are retained and untouched; these are new."""
    for case in record["cases"]:
        assert str(case["case_id"]).startswith("m3."), case["case_id"]
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


def test_every_phase_red_node_appears_exactly_once_and_lives_in_the_r3_list(
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
        assert relative in HISTORICAL_R3_AUTHORIZED_PATHS, nodeid
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
    """Every red oracle file Section 13.5 authorizes contributes at least one node."""
    covered = {str(case["test_nodeid"]).split("::", 1)[0] for case in record["cases"]}

    assert covered == COVERED_FILES


def test_the_five_output_writers_each_carry_at_least_one_case(
    record: dict[str, Any],
) -> None:
    """Section 12.2's ninth family names five output paths, and this covers them.

    "Results: in-memory, summary, HDF5, UVFITS, and MS round trips with phase,
    feed, correlation, time, solver, and fingerprint metadata."  The shared
    standard-visibility projection is the in-memory seam UVFITS and MS pass
    through, so the five files below are that sentence's five paths.
    """
    counts: dict[str, int] = {}
    for case in record["cases"]:
        relative = str(case["test_nodeid"]).split("::", 1)[0]
        counts[relative] = counts.get(relative, 0) + 1

    for relative in (
        "tests/unit/test_io/test_standard_visibility.py",
        "tests/unit/test_io/test_result_summary.py",
        "tests/unit/test_io/test_hdf5_result.py",
        "tests/unit/test_io/test_uvfits.py",
        "tests/unit/test_io/test_measurement_set.py",
    ):
        assert counts.get(relative, 0) >= 1, relative


def test_every_section_11_family_has_its_own_case(record: dict[str, Any]) -> None:
    """Section 12.2's tenth family: "every new family" is a per-family obligation.

    The set is the accepted four-family envelope. The three families the
    correction removed must not reappear: two of them published identically zero
    cubes through the public path and one silently dropped its diffuse half, so
    a case naming them would pin a run that characterizes nothing.
    """
    from tests.characterization.test_sci004_mmode import SECTION_11_FAMILIES

    identifiers = {str(case["case_id"]) for case in record["cases"]}

    assert len(SECTION_11_FAMILIES) == 4
    assert SECTION_11_FAMILIES == (
        "mmode_single_scalar_mode",
        "mmode_point_stokes_i",
        "mmode_point_full_stokes",
        "mmode_circular_receptor",
    )
    for family_id in SECTION_11_FAMILIES:
        assert f"m3.characterization.family-record.{family_id}" in identifiers
    for removed in (
        "mmode_healpix_stokes_i",
        "mmode_healpix_full_stokes",
        "mmode_hybrid_full_stokes",
        "mmode_nonscalar_east_x",
    ):
        assert f"m3.characterization.family-record.{removed}" not in identifiers


def test_both_section_8_public_path_rejections_have_their_own_case(
    record: dict[str, Any],
) -> None:
    """Section 11's deferral paragraph: the public path *rejects* rather than
    silently producing a vacuous or half-dropped result.

    "The public path rejects a HEALPix-bearing payload and a non-scalar resolved
    beam system with the Section 8 typed issues before any work."  Both
    rejections are red here, and their observed failures are deliberately
    different: one records that nothing is raised at all, the other that an
    untyped beam error is raised instead of the typed one.
    """
    by_id = {str(case["case_id"]): case for case in record["cases"]}

    components = by_id["m3.rejection.public-components"]
    beam = by_id["m3.rejection.public-beam"]
    assert components["expected_failure_kind"] == "assertion"
    assert "DID NOT RAISE" in str(components["observed_message"])
    assert beam["expected_failure_kind"] == "exception"
    assert "BeamEvaluationError" in str(beam["observed_exception_type"])


# --- Correction #24: the separate post-source six-case red delta ------------


def _oracle_worktree_snapshot() -> tuple[bytes, bytes]:
    """Inspect the fixed path before Git can hide it through conversion/cache."""
    path = POST_SOURCE_ORACLE_PATH
    attrs = _red_git_process(("check-attr", "--all", "-z", "--", path))
    fields = attrs.stdout.split(b"\0")
    transforms = {
        b"diff",
        b"filter",
        b"text",
        b"eol",
        b"crlf",
        b"ident",
        b"working-tree-encoding",
    }
    config = _git("config", "--null", "--name-only", "--list").split("\0")
    if (
        attrs.returncode != 0
        or fields[-1] != b""
        or (len(fields) - 1) % 3
        or transforms.intersection(fields[1:-1:3])
        or any(key.startswith("diff.default.") for key in config)
    ):
        raise RedRecordSchemaError("worktree oracle conversion/driver is not raw")
    expected = b"H " + path.encode("utf-8", "surrogateescape") + b"\0"
    tracked = _red_git_process(("ls-files", "-v", "-z", "--", path))
    if tracked.returncode != 0 or tracked.stdout != expected:
        raise RedRecordSchemaError("worktree oracle index flags/path are not ordinary")
    source = REPOSITORY_ROOT / path
    try:
        mode = source.lstat().st_mode
        if not stat.S_ISREG(mode) or not source.parent.resolve(
            strict=True
        ).is_relative_to(REPOSITORY_ROOT.resolve()):
            raise RedRecordSchemaError("worktree oracle is not an owned regular file")
        return (b"100755" if mode & stat.S_IXUSR else b"100644", source.read_bytes())
    except OSError as exc:
        raise RedRecordSchemaError("worktree oracle cannot be read") from exc


def _bind_oracle_patch_to_raw_worktree(
    snapshot: tuple[bytes, bytes], patch: bytes
) -> None:
    """Bind untouched full-index output to actual bytes, even with stale stat data."""

    def oid(raw: bytes) -> bytes:
        # The governed repository uses Git's 40-hex SHA-1 blob object format.
        return (
            hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw)
            .hexdigest()
            .encode("ascii")
        )

    original = _tree_blob(POST_SOURCE_PRE_FIX_SHA, POST_SOURCE_ORACLE_PATH)
    old_mode = (
        _git("ls-tree", "-z", POST_SOURCE_PRE_FIX_SHA, "--", POST_SOURCE_ORACLE_PATH)
        .split(" ", 1)[0]
        .encode("ascii")
    )
    mode, raw = snapshot
    lines = patch.splitlines()
    indexes = [line for line in lines if line.startswith(b"index ")]
    expected = (
        []
        if raw == original
        else [
            b"index "
            + oid(original)
            + b".."
            + oid(raw)
            + (b" " + mode if mode == old_mode else b"")
        ]
    )
    modes = [line for line in lines if line.startswith((b"old mode ", b"new mode "))]
    expected_modes = (
        [] if mode == old_mode else [b"old mode " + old_mode, b"new mode " + mode]
    )
    if (
        indexes != expected
        or modes != expected_modes
        or (raw == original and mode == old_mode and patch)
    ):
        raise RedRecordSchemaError("oracle patch does not bind raw worktree bytes/mode")


def _post_source_oracle_diff() -> bytes:
    anchor = resolve_r3_replay_anchor()
    argv = [
        "git",
        "diff",
        "--no-ext-diff",
        "--binary",
        "--full-index",
        POST_SOURCE_PRE_FIX_SHA,
    ]
    if anchor.role == "r3":
        argv.append(anchor.commit)
    argv.extend(("--", POST_SOURCE_ORACLE_PATH))
    snapshot = _oracle_worktree_snapshot() if anchor.role != "r3" else None
    completed = _red_git_process(argv[1:])
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    if snapshot is not None:
        if _oracle_worktree_snapshot() != snapshot:
            raise RedRecordSchemaError("worktree oracle changed during patch read")
        _bind_oracle_patch_to_raw_worktree(snapshot, completed.stdout)
    return completed.stdout


@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "committed",
        "filter",
        "text",
        "eol",
        "ident",
        "encoding",
        "assume-unchanged",
        "skip-worktree",
        "stat-cache",
        "autocrlf",
        "mode-cache",
    ],
)
def test_raw_oracle_worktree_native_conversion_and_cache_controls(
    raw_red_git_repository: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    from types import SimpleNamespace

    root, parent, child = raw_red_git_repository
    original = b"$Id$\n" if case == "ident" else b"after\n"
    if case == "ident":
        child = _raw_red_commit(root, parent, original)
    config = (root / "config").read_bytes()
    with _detached_fingerprint_replay_worktree(child) as (worktree, _):
        with monkeypatch.context() as scoped:
            module = sys.modules[__name__]
            scoped.setattr(module, "REPOSITORY_ROOT", worktree)
            scoped.setattr(module, "POST_SOURCE_PRE_FIX_SHA", child)
            scoped.setattr(module, "POST_SOURCE_ORACLE_PATH", "oracle.py")
            scoped.setattr(
                module,
                "resolve_r3_replay_anchor",
                lambda: SimpleNamespace(role="pre-commit-authoring-tip", commit=child),
            )
            assert _post_source_oracle_diff() == b""
            path = worktree / "oracle.py"
            before = path.stat()
            # Cached mtime must predate the index write to avoid Git's racy-index
            # safeguard in the native stale-stat positive control.
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns - 10_000_000_000))
            _ = _native_red_git(
                worktree, "update-index", "--refresh", "--", "oracle.py"
            )
            before = path.stat()
            try:
                attribute = {
                    "filter": "filter=cloak",
                    "text": "text",
                    "eol": "eol=lf",
                    "ident": "ident",
                    "encoding": "working-tree-encoding=UTF-16LE",
                }.get(case)
                altered = b"evil! changed\n" if case == "ordinary" else b"evil!\n"
                if attribute:
                    _ = (root / "info/attributes").write_text(
                        f"oracle.py {attribute}\n"
                    )
                if case == "filter":
                    _ = _native_red_git(
                        root, "config", "filter.cloak.clean", "printf 'after\\n'"
                    )
                elif case in {"text", "eol", "autocrlf"}:
                    altered = b"after\r\n"
                    if case == "autocrlf":
                        _ = _native_red_git(root, "config", "core.autocrlf", "true")
                elif case == "ident":
                    altered = b"$Id: forged $\n"
                elif case == "encoding":
                    altered = "after\n".encode("utf-16-le")
                elif case in {"assume-unchanged", "skip-worktree"}:
                    _ = _native_red_git(
                        worktree, "update-index", f"--{case}", "oracle.py"
                    )
                elif case == "stat-cache":
                    _ = _native_red_git(root, "config", "core.trustctime", "false")
                    _ = _native_red_git(root, "config", "core.checkStat", "minimal")
                elif case == "mode-cache":
                    _ = _native_red_git(root, "config", "core.fileMode", "false")
                    altered = original
                    path.chmod(before.st_mode | stat.S_IXUSR)
                _ = path.write_bytes(altered)
                if case == "stat-cache":
                    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                native = _native_red_git(
                    worktree,
                    "diff",
                    "--binary",
                    "--full-index",
                    child,
                    "--",
                    "oracle.py",
                )
                if case == "ordinary":
                    assert native and b"+evil!" in native
                    assert _post_source_oracle_diff() == native
                elif case == "committed":
                    scoped.setattr(
                        module,
                        "resolve_r3_replay_anchor",
                        lambda: SimpleNamespace(role="r3", commit=child),
                    )
                    assert _post_source_oracle_diff() == b""
                else:
                    assert native == b"", (
                        "native conversion/cache must conceal the changed bytes or mode"
                    )
                    with pytest.raises(RedRecordSchemaError):
                        _ = _post_source_oracle_diff()
                if case in {
                    "filter",
                    "stat-cache",
                    "assume-unchanged",
                    "skip-worktree",
                    "mode-cache",
                }:
                    assert (
                        _native_red_git(
                            worktree, "status", "--porcelain", "--untracked-files=no"
                        )
                        == b""
                    )
                    with pytest.raises(AssertionError, match="historical tracked"):
                        _assert_immutable_replay_checkout(worktree, child)
                with pytest.raises(AssertionError, match="historical tracked"):
                    _assert_raw_tracked_checkout(worktree, child)
            finally:
                _ = (root / "config").write_bytes(config)
                (root / "info/attributes").unlink(missing_ok=True)
                _ = _native_red_git(
                    worktree,
                    "update-index",
                    "--no-assume-unchanged",
                    "--no-skip-worktree",
                    "oracle.py",
                )
                path.chmod(before.st_mode)
                _ = path.write_bytes(original)


def test_the_post_source_record_is_exactly_its_canonical_serialization() -> None:
    raw, document = read_post_source_record()

    assert raw and not raw.endswith(b"\n")
    assert canonical_json_bytes(document) == raw


def test_the_post_source_record_has_the_exact_supplement_schema(
    post_source_record: dict[str, Any],
) -> None:
    assert set(post_source_record) == POST_SOURCE_TOP_LEVEL_KEYS
    assert post_source_record["schema_version"] == POST_SOURCE_SCHEMA_VERSION
    assert post_source_record["phase"] == PHASE
    assert post_source_record["status"] == POST_SOURCE_STATUS
    assert _UTC_STAMP.match(str(post_source_record["generated_at_utc"])) is not None
    assert post_source_record["red_commit_sha"] is None
    assert post_source_record["red_commit_sha_reason"] == (
        POST_SOURCE_RED_COMMIT_SHA_REASON
    )


def test_the_post_source_record_binds_d24_and_the_exact_superseded_source(
    post_source_record: dict[str, Any],
) -> None:
    assert post_source_record["design_sha"] == D24_SHA
    assert post_source_record["pre_fix_source_sha"] == POST_SOURCE_PRE_FIX_SHA
    assert _peel_to_commit(POST_SOURCE_PRE_FIX_SHA) == POST_SOURCE_PRE_FIX_SHA
    assert _peel_to_commit(D24_SHA) == D24_SHA
    assert _git("rev-parse", f"{D24_SHA}^").strip() == POST_SOURCE_PRE_FIX_SHA
    assert (
        _git(
            "diff",
            "--name-only",
            POST_SOURCE_PRE_FIX_SHA,
            D24_SHA,
            "--",
            "src/radiosim",
        ).split()
        == []
    )


def test_the_post_source_record_binds_the_immutable_historical_record(
    post_source_record: dict[str, Any],
) -> None:
    historical = (REPOSITORY_ROOT / RECORD_PATH).read_bytes()

    assert post_source_record["historical_red_record_sha256"] == (
        HISTORICAL_RED_RECORD_SHA256
    )
    assert hashlib.sha256(historical).hexdigest() == HISTORICAL_RED_RECORD_SHA256
    assert historical == _tree_blob(HISTORICAL_RED_SLICE_SHA, RECORD_PATH)


def test_the_post_source_record_binds_the_exact_hdf5_oracle_patch(
    post_source_record: dict[str, Any],
) -> None:
    oracle = _post_source_oracle_diff()

    assert post_source_record["oracle_patch_paths"] == [POST_SOURCE_ORACLE_PATH]
    assert oracle
    assert (
        hashlib.sha256(oracle).hexdigest() == post_source_record["oracle_patch_sha256"]
    )


def test_the_post_source_authorized_paths_are_exactly_the_five_path_grant(
    post_source_record: dict[str, Any],
) -> None:
    expected = sorted(R3_AUTHORIZED_PATHS)

    assert post_source_record["protected_source_clean"] is True
    assert post_source_record["authorized_red_paths"] == expected


def test_the_post_source_environment_and_claims_keep_the_base_red_shapes(
    post_source_record: dict[str, Any],
) -> None:
    environment = post_source_record["environment"]
    assert set(environment) == ENVIRONMENT_KEYS
    assert set(environment["numeric_packages"]) == NUMERIC_PACKAGES
    assert list(environment["numeric_packages"]) == sorted(NUMERIC_PACKAGES)
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
    for name, version in environment["numeric_packages"].items():
        assert isinstance(version, str) and version, name

    claims = post_source_record["claims_not_licensed"]
    assert claims == list(POST_SOURCE_CLAIMS_NOT_LICENSED)
    assert isinstance(claims, list) and claims
    assert all(isinstance(claim, str) and claim for claim in claims)
    assert claims == sorted(set(claims))
    for category in REQUIRED_CLAIM_CATEGORIES:
        assert any(claim.startswith(f"{category}:") for claim in claims)


def test_the_post_source_replay_is_one_serial_nonzero_command(
    post_source_record: dict[str, Any],
) -> None:
    commands = post_source_record["commands"]
    assert len(commands) == 1
    command = commands[0]
    assert set(command) == COMMAND_KEYS
    assert isinstance(command["exit_code"], int)
    assert not isinstance(command["exit_code"], bool)
    assert command["exit_code"] != 0
    assert command["cwd"] == "."
    argv = command["argv"]
    assert isinstance(argv, list) and argv
    assert all(isinstance(entry, str) and entry for entry in argv)
    assert isinstance(command["pixi_environment"], str)
    assert command["pixi_environment"]
    assert _UTC_STAMP.match(str(command["started_at_utc"])) is not None
    duration = command["duration_seconds"]
    assert isinstance(duration, (int, float)) and not isinstance(duration, bool)
    assert math.isfinite(float(duration)) and float(duration) >= 0.0
    assert argv[1:7] == [
        "-m",
        "pytest",
        "-p",
        "no:randomly",
        "-p",
        "no:xdist",
    ]
    assert argv[7] == "--junit-xml"
    assert isinstance(argv[8], str) and argv[8]
    assert "-n" not in argv
    assert len(argv[9:]) == 11
    assert _SHA256.match(str(command["stdout_sha256"])) is not None
    assert _SHA256.match(str(command["stderr_sha256"])) is not None


def test_the_historical_and_post_source_generator_semantics_are_separate() -> None:
    """Correction #24 extends generation without changing the historical form."""
    tool_path = REPOSITORY_ROOT / "tools/sci004_mmode_phase3_red.py"
    source = tool_path.read_text(encoding="utf-8")
    historical = source.split("def generate(output: Path) -> None:", 1)[1].split(
        "def generate_post_source() -> None:", 1
    )[0]
    supplemental = source.split("def generate_post_source() -> None:", 1)[1].split(
        "def _fingerprint_fixture_bytes", 1
    )[0]
    fingerprint = source.split("def generate_fingerprint_post_source() -> None:", 1)[
        1
    ].split("def _atomic_no_overwrite", 1)[0]

    assert '"red_commit_sha_reason": RED_COMMIT_SHA_REASON' in historical
    assert '"claims_not_licensed": list(CLAIMS_NOT_LICENSED)' in historical
    assert "POST_SOURCE_RED_COMMIT_SHA_REASON" not in historical
    assert "POST_SOURCE_CLAIMS_NOT_LICENSED" not in historical
    assert '"red_commit_sha_reason": POST_SOURCE_RED_COMMIT_SHA_REASON' in supplemental
    assert (
        '"claims_not_licensed": list(POST_SOURCE_CLAIMS_NOT_LICENSED)' in supplemental
    )
    assert "FINGERPRINT_POST_SOURCE_RED_COMMIT_SHA_REASON" not in supplemental
    assert (
        '"red_commit_sha_reason": FINGERPRINT_POST_SOURCE_RED_COMMIT_SHA_REASON'
        in fingerprint
    )
    assert (
        '"claims_not_licensed": list(FINGERPRINT_POST_SOURCE_CLAIMS_NOT_LICENSED)'
        in fingerprint
    )

    completed = subprocess.run(
        [sys.executable, str(tool_path), "generate"],
        cwd=REPOSITORY_ROOT,
        env=_closed_fingerprint_replay_environment(REPOSITORY_ROOT),
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert completed.stdout == b""
    assert completed.stderr.decode("utf-8").endswith(
        f"{RECORD_PATH} already exists; generation never overwrites\n"
    )


def test_the_historical_and_post_source_node_sets_form_one_disjoint_inventory(
    record: dict[str, Any],
    declared_nodes: dict[str, dict[str, Any]],
    post_source_record: dict[str, Any],
    post_source_declared_nodes: dict[str, dict[str, Any]],
) -> None:
    historical = [str(case["test_nodeid"]) for case in record["cases"]]
    supplemental = [str(case["test_nodeid"]) for case in post_source_record["cases"]]

    assert set(historical) == set(declared_nodes)
    assert set(supplemental) == set(post_source_declared_nodes)
    assert set(historical).isdisjoint(supplemental)
    assert len(set(historical) | set(supplemental)) == len(historical) + 6


def test_every_post_source_case_is_the_exact_confirmed_regex_mismatch(
    post_source_record: dict[str, Any],
    post_source_declared_nodes: dict[str, dict[str, Any]],
) -> None:
    cases = post_source_record["cases"]
    assert len(cases) == 6
    assert len({case["case_id"] for case in cases}) == 6
    assert len({case["test_nodeid"] for case in cases}) == 6
    for case in cases:
        identifier = str(case["case_id"])
        nodeid = str(case["test_nodeid"])
        declared = post_source_declared_nodes[nodeid]
        assert set(case) == CASE_KEYS, identifier
        assert case["requirement_id"] == declared["requirement_id"]
        assert case["expected_failure_kind"] == "assertion"
        assert case["observed_outcome"] == "assertion"
        assert case["observed_exception_type"] in (
            "builtins.AssertionError",
            "_pytest.outcomes.Failed",
        )
        assert case["red_failure_confirmed"] is True
        assert case["command_index"] == 0
        assert case["exit_code"] == post_source_record["commands"][0]["exit_code"]
        assert (
            case["stdout_sha256"] == post_source_record["commands"][0]["stdout_sha256"]
        )
        assert (
            case["stderr_sha256"] == post_source_record["commands"][0]["stderr_sha256"]
        )
        assert isinstance(case["fixture_defect_excluded_by"], str)
        assert case["fixture_defect_excluded_by"]
        assert identifier.startswith("m3.")
        assert str(case["requirement_id"]).startswith("sci004.section-")
        assert "HDF5 result failed canonical model or fingerprint validation" in str(
            case["observed_message"]
        )
        assert re.search(
            str(case["expected_failure_pattern"]),
            str(case["observed_message"]),
        )
        fixture = declared["fixture_bytes"]
        assert canonical_json_bytes(json.loads(fixture.decode("utf-8"))) == fixture
        invalid_digest = hashlib.sha256(fixture).hexdigest()
        assert case["invalid_config_raw_sha256"] == invalid_digest
        expected_identity = domain_digest(
            "radiosim.sci004-red-fixture.v1",
            canonical_json_bytes(
                {
                    "phase": PHASE,
                    "fixture_id": case["case_id"],
                    "requirement_id": case["requirement_id"],
                    "test_nodeid": case["test_nodeid"],
                    "pre_fix_source_sha": POST_SOURCE_PRE_FIX_SHA,
                    "invalid_config_raw_sha256": invalid_digest,
                }
            ),
        )
        assert case["fixture_identity_sha256"] == expected_identity

    hdf5_module = importlib.import_module("tests.unit.test_io.test_hdf5_result")
    assert hdf5_module._POST_SOURCE_HDF5_EARLY_PATTERN == (
        r"^HDF5 solver_json is invalid$"
    )


# --- Correction #25: fresh fingerprint post-source red delta -----------------


def _fingerprint_oracle_diff() -> bytes:
    argv = [
        "git",
        "-c",
        "color.ui=false",
        "--no-pager",
        "diff",
        "--no-ext-diff",
        "--binary",
        "--full-index",
        D25_SHA,
        ORIGINAL_FINGERPRINT_R3_SHA,
    ]
    argv.extend(("--", FINGERPRINT_POST_SOURCE_ORACLE_PATH))
    completed = _red_git_process(argv[argv.index("diff") :])
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    return completed.stdout


def _fingerprint_fixture(case_id: str) -> bytes:
    common: dict[str, Any] = {
        "schema_version": "radiosim.sci004.fingerprint-red-fixture.v1",
        "family_id": "mmode_single_scalar_mode",
        "layout_document_raw_sha256": (
            "a2ce7bace30e2fe962eb6454db1f6c7e2d63a9a28ad559323e824a36fcd2a4e0"
        ),
    }
    if case_id == "m3.fingerprint.preimage-retained":
        common.update(
            {
                "root_labels": ["ROOT-A"],
                "required_record_keys": [
                    "family_id",
                    "raw_cube_sha256",
                    "scientific_sha256",
                    "solver_snapshot",
                    "characterization_time_manifest",
                    "era_utc_grid_sha256",
                    "harmonic_index_table_sha256",
                    "characterization_input_manifest",
                    "input_identity_sha256",
                ],
            }
        )
    else:
        assert case_id == "m3.fingerprint.path-independent"
        common.update(
            {
                "root_labels": ["ROOT-A", "ROOT-B"],
                "required_equal_identities": [
                    "scientific_sha256",
                    "raw_cube_sha256",
                    "era_utc_grid_sha256",
                    "input_identity_sha256",
                ],
            }
        )
    return canonical_json_bytes(common)


def test_the_fingerprint_supplement_is_canonical_and_exactly_bound(
    fingerprint_post_source_record: dict[str, Any],
) -> None:
    raw, document = read_fingerprint_post_source_record()
    assert raw and not raw.endswith(b"\n")
    assert canonical_json_bytes(document) == raw
    assert set(document) == FINGERPRINT_POST_SOURCE_TOP_LEVEL_KEYS
    assert document["schema_version"] == FINGERPRINT_POST_SOURCE_SCHEMA_VERSION
    assert document["phase"] == PHASE
    assert document["status"] == POST_SOURCE_STATUS
    assert _UTC_STAMP.fullmatch(str(document["generated_at_utc"])) is not None
    assert hashlib.sha256(raw).hexdigest() == FINGERPRINT_SUPPLEMENT_SHA256
    assert document["design_sha"] == D25_SHA
    assert document["pre_fix_source_sha"] == SUPERSEDED_FINGERPRINT_S3_SHA
    assert document["red_commit_sha"] is None
    assert (
        document["red_commit_sha_reason"]
        == FINGERPRINT_POST_SOURCE_RED_COMMIT_SHA_REASON
    )
    assert document["protected_source_clean"] is True
    assert document["authorized_red_paths"] == sorted(FINGERPRINT_R3_AUTHORIZED_PATHS)
    assert document["claims_not_licensed"] == list(
        FINGERPRINT_POST_SOURCE_CLAIMS_NOT_LICENSED
    )
    assert document["claims_not_licensed"] == sorted(
        set(document["claims_not_licensed"])
    )
    for category in REQUIRED_CLAIM_CATEGORIES:
        assert any(
            claim.startswith(f"{category}:")
            for claim in document["claims_not_licensed"]
        )
    environment = document["environment"]
    assert set(environment) == ENVIRONMENT_KEYS
    assert set(environment["numeric_packages"]) == NUMERIC_PACKAGES
    assert list(environment["numeric_packages"]) == sorted(NUMERIC_PACKAGES)
    assert fingerprint_post_source_record is document or (
        fingerprint_post_source_record == document
    )

    assert phase_history.DESIGN_SHA == D30_SHA
    assert phase_history.OPERATIVE_DESIGN_SHA == APPROVED_SCI004_D_SHA
    prerequisite = phase_history.describe_phase_range(
        D30_SHA,
        phase_history.PREREQUISITE_TIP_SHA,
        "prerequisite",
        root=REPOSITORY_ROOT,
    )
    preceding_red = phase_history.describe_phase_range(
        phase_history.PREREQUISITE_TIP_SHA,
        D30_R1_TERMINAL_SHA,
        "red",
        root=REPOSITORY_ROOT,
    )
    chain = (
        D24_SHA,
        SUPERSEDED_FINGERPRINT_R3_SHA,
        SUPERSEDED_FINGERPRINT_S3_SHA,
        REJECTED_E3_SHA,
        REJECTED_A3_SHA,
        D25_SHA,
        ORIGINAL_FINGERPRINT_R3_SHA,
        D26_SHA,
        OLD_FRESH_VALIDATOR_R3_SHA,
        D27_SHA,
        D28_SHA,
        D29_SHA,
        D30_STATUS_BRIDGE_SHA,
        D30_SHA,
        *(row["sha"] for row in prerequisite["commits"]),
        *(row["sha"] for row in preceding_red["commits"]),
        APPROVED_SCI004_D_SHA,
    )
    previous = _git("rev-parse", f"{D24_SHA}^").strip()
    for commit in chain:
        assert _peel_to_commit(commit) == commit
        assert _git("rev-list", "--parents", "-n", "1", commit).split() == [
            commit,
            previous,
        ]
        previous = commit
    anchor = resolve_r3_replay_anchor()
    assert _git("rev-parse", f"{ORIGINAL_FINGERPRINT_R3_SHA}^").strip() == D25_SHA
    original_changed = tuple(
        sorted(
            _git(
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                ORIGINAL_FINGERPRINT_R3_SHA,
            ).split()
        )
    )
    assert original_changed == tuple(sorted(FINGERPRINT_R3_AUTHORIZED_PATHS))
    assert tuple(
        sorted(
            _git(
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                D26_SHA,
            ).split()
        )
    ) == (DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH)
    assert (
        tuple(
            sorted(
                _git(
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    OLD_FRESH_VALIDATOR_R3_SHA,
                ).split()
            )
        )
        == OLD_FRESH_VALIDATOR_R3_PATHS
    )
    for design in (D27_SHA, D28_SHA, D29_SHA, D30_SHA, APPROVED_SCI004_D_SHA):
        assert tuple(
            sorted(
                _git(
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    design,
                ).split()
            )
        ) == (DESIGN_LEDGER_PATH, DESIGN_MEMO_PATH)
    assert _git(
        "diff-tree", "--no-commit-id", "--name-only", "-r", D30_STATUS_BRIDGE_SHA
    ).splitlines() == [phase_history.STATUS_PATH]
    assert anchor.role in {"r3", "pre-commit-authoring-tip"}
    current_red = phase_history.describe_phase_range(
        phase_history.PREREQUISITE_TIP_SHA,
        anchor.commit,
        "red",
        root=REPOSITORY_ROOT,
        require_complete=anchor.role == "r3",
    )
    phase_history.require_design_successor(current_red["commits"])
    if anchor.role == "pre-commit-authoring-tip":
        assert anchor.commit == _peel_to_commit("HEAD")


def test_the_fingerprint_supplement_binds_both_immutable_prior_records(
    fingerprint_post_source_record: dict[str, Any],
) -> None:
    historical = (REPOSITORY_ROOT / RECORD_PATH).read_bytes()
    correction24 = (REPOSITORY_ROOT / POST_SOURCE_RECORD_PATH).read_bytes()
    assert hashlib.sha256(historical).hexdigest() == HISTORICAL_RED_RECORD_SHA256
    assert historical == _tree_blob(HISTORICAL_RED_SLICE_SHA, RECORD_PATH)
    assert (
        hashlib.sha256(correction24).hexdigest()
        == CORRECTION24_POST_SOURCE_RED_RECORD_SHA256
    )
    assert correction24 == _tree_blob(
        SUPERSEDED_FINGERPRINT_R3_SHA,
        POST_SOURCE_RECORD_PATH,
    )
    assert fingerprint_post_source_record["historical_red_record_sha256"] == (
        HISTORICAL_RED_RECORD_SHA256
    )
    assert (
        fingerprint_post_source_record["correction24_post_source_red_record_sha256"]
        == CORRECTION24_POST_SOURCE_RED_RECORD_SHA256
    )


def test_all_three_red_records_are_inherited_byte_for_byte_at_the_live_r3() -> None:
    """Every D31 red-range tip inherits the exact three historical records."""
    anchor = resolve_r3_replay_anchor()
    for relative, expected_sha256, original in (
        (RECORD_PATH, HISTORICAL_RED_RECORD_SHA256, HISTORICAL_RED_SLICE_SHA),
        (
            POST_SOURCE_RECORD_PATH,
            CORRECTION24_POST_SOURCE_RED_RECORD_SHA256,
            SUPERSEDED_FINGERPRINT_R3_SHA,
        ),
        (
            FINGERPRINT_POST_SOURCE_RECORD_PATH,
            FINGERPRINT_SUPPLEMENT_SHA256,
            ORIGINAL_FINGERPRINT_R3_SHA,
        ),
    ):
        raw = (REPOSITORY_ROOT / relative).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_sha256
        assert raw == _tree_blob(original, relative)
        assert raw == _tree_blob(anchor.commit, relative)


def test_the_fingerprint_oracle_patch_digest_uses_the_exact_governed_framing(
    fingerprint_post_source_record: dict[str, Any],
) -> None:
    oracle = _fingerprint_oracle_diff()
    assert oracle
    assert fingerprint_post_source_record["oracle_patch_paths"] == [
        FINGERPRINT_POST_SOURCE_ORACLE_PATH
    ]
    assert (
        hashlib.sha256(oracle).hexdigest()
        == fingerprint_post_source_record["oracle_patch_sha256"]
    )


def test_the_fingerprint_cases_and_controls_are_the_exact_five_node_partition(
    fingerprint_post_source_record: dict[str, Any],
) -> None:
    cases = fingerprint_post_source_record["cases"]
    controls = fingerprint_post_source_record["passing_controls"]
    command = fingerprint_post_source_record["commands"]
    assert len(command) == 1
    assert set(command[0]) == COMMAND_KEYS
    assert command[0]["exit_code"] == 1
    argv = command[0]["argv"]
    assert argv[1:7] == [
        "-m",
        "pytest",
        "-p",
        "no:randomly",
        "-p",
        "no:xdist",
    ]
    assert argv[7] == "--junit-xml"
    assert argv[9:] == list(FINGERPRINT_NODEIDS)
    assert "-n" not in argv

    assert len(cases) == 2
    for row, expected in zip(cases, FINGERPRINT_CASE_EXPECTATIONS, strict=True):
        case_id, requirement_id, nodeid, pattern = expected
        assert set(row) == CASE_KEYS
        assert (
            row["case_id"],
            row["requirement_id"],
            row["test_nodeid"],
            row["expected_failure_pattern"],
        ) == expected
        assert row["expected_failure_kind"] == "assertion"
        assert row["observed_outcome"] == "assertion"
        assert row["observed_exception_type"] in (
            "builtins.AssertionError",
            "_pytest.outcomes.Failed",
        )
        assert pattern in row["observed_message"]
        assert row["command_index"] == 0
        assert row["exit_code"] == 1
        assert row["red_failure_confirmed"] is True
        fixture = _fingerprint_fixture(case_id)
        invalid_digest = hashlib.sha256(fixture).hexdigest()
        assert row["invalid_config_raw_sha256"] == invalid_digest
        expected_identity = domain_digest(
            "radiosim.sci004-red-fixture.v1",
            canonical_json_bytes(
                {
                    "phase": PHASE,
                    "fixture_id": case_id,
                    "requirement_id": requirement_id,
                    "test_nodeid": nodeid,
                    "pre_fix_source_sha": SUPERSEDED_FINGERPRINT_S3_SHA,
                    "invalid_config_raw_sha256": invalid_digest,
                }
            ),
        )
        assert row["fixture_identity_sha256"] == expected_identity
        assert row["stdout_sha256"] == command[0]["stdout_sha256"]
        assert row["stderr_sha256"] == command[0]["stderr_sha256"]

    assert len(controls) == 3
    for row, expected in zip(
        controls,
        FINGERPRINT_CONTROL_EXPECTATIONS,
        strict=True,
    ):
        control_id, requirement_id, nodeid, purpose = expected
        assert set(row) == PASSING_CONTROL_KEYS
        assert (
            row["control_id"],
            row["requirement_id"],
            row["test_nodeid"],
            row["purpose"],
        ) == expected
        assert row["command_index"] == 0
        assert row["observed_outcome"] == "pass"
        assert row["exit_code"] == 0
        assert row["pass"] is True

    inventory = [str(row["test_nodeid"]) for row in (*cases, *controls)]
    assert inventory == list(FINGERPRINT_NODEIDS)
    assert len(inventory) == len(set(inventory)) == 5


def test_all_three_red_records_form_the_exact_disjoint_expected_red_union(
    record: dict[str, Any],
    post_source_record: dict[str, Any],
    fingerprint_post_source_record: dict[str, Any],
) -> None:
    inventories = [
        [str(case["test_nodeid"]) for case in document["cases"]]
        for document in (record, post_source_record, fingerprint_post_source_record)
    ]
    assert [len(rows) for rows in inventories] == [29, 6, 2]
    assert set(inventories[0]).isdisjoint(inventories[1])
    assert set(inventories[0]).isdisjoint(inventories[2])
    assert set(inventories[1]).isdisjoint(inventories[2])
    assert len(set().union(*(set(rows) for rows in inventories))) == sum(
        map(len, inventories)
    )


def test_the_characterization_delta_has_only_the_ruled_surface() -> None:
    parent = ast.parse(_tree_blob(D25_SHA, FINGERPRINT_POST_SOURCE_ORACLE_PATH))
    child_raw = _tree_blob(
        ORIGINAL_FINGERPRINT_R3_SHA,
        FINGERPRINT_POST_SOURCE_ORACLE_PATH,
    )
    child = ast.parse(child_raw)
    parent_functions = {
        node.name: node
        for node in parent.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    child_functions = {
        node.name: node
        for node in child.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    allowed_new = {
        "_family_result_and_phase_input_manifest",
        "_characterization_record_for_active_domain",
        "_relocated_family_records",
        "_semantic_layout_mutation",
        "test_characterization_input_preimage_is_retained_and_reconstructible",
        "test_characterization_input_identity_is_equal_under_distinct_layout_roots",
        "test_distinct_layout_roots_preserve_scientific_and_cube_identities",
        "test_characterization_input_identity_changes_for_semantic_instrument_content",
    }
    assert set(child_functions) - set(parent_functions) == allowed_new
    assert set(parent_functions) <= set(child_functions)
    modified = "test_every_new_family_records_its_six_section_11_parts"
    for name, node in parent_functions.items():
        if name != modified:
            assert ast.dump(node, include_attributes=False) == ast.dump(
                child_functions[name], include_attributes=False
            ), name
    parent_asserts = {
        ast.dump(node, include_attributes=False)
        for node in ast.walk(parent_functions[modified])
        if isinstance(node, ast.Assert)
        and "FAMILY_RECORD_KEYS" not in ast.dump(node, include_attributes=False)
        and "mmode_characterization_record"
        not in ast.dump(node, include_attributes=False)
    }
    child_asserts = {
        ast.dump(node, include_attributes=False)
        for node in ast.walk(child_functions[modified])
        if isinstance(node, ast.Assert)
    }
    assert parent_asserts <= child_asserts
    assert len(child_asserts) >= len(parent_asserts) + 3
    modified_source = ast.unparse(child_functions[modified])
    assert modified_source.count("_characterization_record_for_active_domain") == 2

    def assigned_names(module: ast.Module) -> set[str]:
        return {
            target.id
            for node in module.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }

    parent_names = assigned_names(parent)
    child_names = assigned_names(child)
    assert parent_names - child_names == {"FAMILY_RECORD_KEYS"}
    assert child_names - parent_names == {
        "FAMILY_RECORD_V1_KEYS",
        "FAMILY_RECORD_V2_KEYS",
        "FINGERPRINT_RED_LAYOUT_BYTES",
    }


def test_the_fingerprint_generator_refuses_to_overwrite_its_retained_record() -> None:
    tool_path = REPOSITORY_ROOT / "tools/sci004_mmode_phase3_red.py"
    completed = subprocess.run(
        [sys.executable, str(tool_path), "generate-fingerprint-post-source"],
        cwd=REPOSITORY_ROOT,
        env=_closed_fingerprint_replay_environment(REPOSITORY_ROOT),
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert completed.stdout == b""
    assert completed.stderr.decode("utf-8").endswith(
        f"{FINGERPRINT_POST_SOURCE_RECORD_PATH} already exists; "
        "generation never overwrites\n"
    )


def _closed_fingerprint_replay_environment(worktree: Path) -> dict[str, str]:
    replay_env = {
        key: value
        for key, value in _red_git_environment().items()
        if not key.startswith(("PYTHON", "_PYTHON", "PYTEST_"))
        and key != "__PYVENV_LAUNCHER__"
    }
    replay_env["PYTHONPATH"] = str(worktree / "src")
    replay_env["PYTHONNOUSERSITE"] = "1"
    replay_env["PYTHONSAFEPATH"] = "1"
    return replay_env


def _validate_fingerprint_replay_import(stdout: bytes, worktree: Path) -> Path:
    try:
        rendered = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssertionError("detached radiosim import path is not UTF-8") from exc
    lines = rendered.splitlines()
    assert rendered.endswith("\n")
    assert len(lines) == 1 and lines[0]
    reported = Path(lines[0])
    assert reported.is_absolute()
    try:
        imported = reported.resolve(strict=True)
        detached_package = (worktree / "src" / "radiosim").resolve(strict=True)
        invoking_src = (REPOSITORY_ROOT / "src").resolve(strict=True)
    except OSError as exc:
        raise AssertionError(
            "detached radiosim import path cannot be resolved"
        ) from exc
    assert imported.is_file()
    assert detached_package.is_relative_to(worktree.resolve(strict=True) / "src")
    assert imported.is_relative_to(detached_package)
    assert not imported.is_relative_to(invoking_src)
    registered_worktrees = {
        Path(line.removeprefix("worktree ")).resolve()
        for line in _git("worktree", "list", "--porcelain").splitlines()
        if line.startswith("worktree ")
    }
    for registered in registered_worktrees - {worktree.resolve()}:
        assert not imported.is_relative_to(registered)
    site_paths = {
        Path(path).resolve()
        for path in (
            site.getusersitepackages(),
            sysconfig.get_path("purelib"),
            sysconfig.get_path("platlib"),
        )
        if path
    }
    for site_path in site_paths:
        assert not imported.is_relative_to(site_path)
    return imported


def _preflight_fingerprint_replay_import(
    worktree: Path,
    replay_env: Mapping[str, str],
) -> Path:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path\n"
                "import radiosim\n"
                "print(Path(radiosim.__file__).resolve(strict=True))\n"
            ),
        ],
        cwd=worktree,
        env=dict(replay_env),
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stderr == b""
    return _validate_fingerprint_replay_import(completed.stdout, worktree)


def _assert_raw_tracked_checkout(worktree: Path, anchor: str) -> None:
    """Compare actual tracked bytes/types/modes with the original anchor tree.

    Git status applies clean filters and stat-cache shortcuts. Preserve its
    index/gitlink checks separately, but never use it as proof of file bytes.
    Symlinks authenticate the stored link target, without following the link.
    Uninitialized gitlinks retain the existing status-based checkout semantics.
    """
    assert stat.S_ISDIR(worktree.lstat().st_mode), (
        "historical checkout root type changed"
    )
    tree = _red_git_process(("ls-tree", "-r", "-z", anchor), cwd=worktree)
    assert tree.returncode == 0, tree.stderr.decode("utf-8", "replace")
    entries = tree.stdout.split(b"\0")
    assert entries[-1] == b"", "historical tree framing"
    root = worktree.resolve(strict=True)
    seen: set[bytes] = set()
    for entry in entries[:-1]:
        metadata, separator, relative = entry.partition(b"\t")
        fields = metadata.split()
        assert separator and len(fields) == 3 and relative not in seen, (
            "historical tree entry"
        )
        seen.add(relative)
        mode, kind, oid = fields
        path = root / relative.decode("utf-8", "surrogateescape")
        try:
            parent = root
            for component in relative.split(b"/")[:-1]:
                assert component not in {b"", b".", b".."}, "historical tracked path"
                parent /= os.fsdecode(component)
                assert stat.S_ISDIR(parent.lstat().st_mode), (
                    "historical tracked parent directory type changed"
                )
            if mode == b"160000":
                assert kind == b"commit", "historical gitlink type"
                continue
            assert mode in {b"100644", b"100755", b"120000"} and kind == b"blob", (
                "historical tracked type"
            )
            actual_mode = path.lstat().st_mode
            if mode == b"120000":
                assert stat.S_ISLNK(actual_mode), (
                    "historical tracked symlink type changed"
                )
                actual = os.fsencode(os.readlink(path))
            else:
                assert stat.S_ISREG(actual_mode), (
                    "historical tracked regular type changed"
                )
                assert bool(actual_mode & stat.S_IXUSR) == (mode == b"100755"), (
                    "historical tracked executable mode changed"
                )
                actual = path.read_bytes()
        except OSError as exc:
            raise AssertionError("historical tracked path cannot be read") from exc
        original = _red_git_process(
            ("cat-file", "blob", oid.decode("ascii")), cwd=worktree
        )
        assert original.returncode == 0, original.stderr.decode("utf-8", "replace")
        matches = actual == original.stdout
        if not matches and mode != b"120000":
            # Canonical LFS pointers authenticate either the pointer itself or
            # its materialized content, without invoking a clean/smudge filter.
            pointer = re.fullmatch(
                rb"version https://git-lfs.github.com/spec/v1\n"
                rb"oid sha256:([0-9a-f]{64})\nsize (0|[1-9][0-9]*)\n",
                original.stdout,
            )
            matches = pointer is not None and (
                pointer[2] == str(len(actual)).encode("ascii")
                and pointer[1] == hashlib.sha256(actual).hexdigest().encode("ascii")
            )
        assert matches, "historical tracked raw bytes changed"


def _assert_immutable_replay_checkout(worktree: Path, anchor: str) -> None:
    for arguments, expected in (
        (["rev-parse", "HEAD"], anchor.encode() + b"\n"),
        (["status", "--porcelain", "--untracked-files=no"], b""),
    ):
        checked = _red_git_process(
            (
                "--git-dir=" + str(worktree / ".git"),
                "--work-tree=" + str(worktree),
                "-c",
                "core.bare=false",
                *arguments,
            ),
            cwd=worktree,
        )
        assert checked.returncode == 0, checked.stderr.decode("utf-8", "replace")
        assert checked.stdout == expected, "historical replay checkout changed"
    _assert_raw_tracked_checkout(worktree, anchor)


@contextmanager
def _detached_fingerprint_replay_worktree(
    anchor: str,
) -> Iterator[tuple[Path, Path]]:
    assert re.fullmatch(r"[0-9a-f]{40}", anchor), "replay requires an exact SHA"
    temporary = Path(tempfile.mkdtemp(prefix="sci004-m3-fingerprint-r3-replay-"))
    worktree = temporary / "replay"
    registered = False
    try:
        added = _red_git_process(("worktree", "add", "--detach", str(worktree), anchor))
        assert added.returncode == 0, added.stderr.decode("utf-8", "replace")
        registered = True
        _assert_immutable_replay_checkout(worktree, anchor)
        yield worktree, temporary
    finally:
        cleanup_errors: list[str] = []
        if registered:
            try:
                _assert_immutable_replay_checkout(worktree, anchor)
            except AssertionError as exc:
                cleanup_errors.append(str(exc))
            removed = _red_git_process(("worktree", "remove", "--force", str(worktree)))
            if removed.returncode != 0:
                cleanup_errors.append(
                    "git worktree remove failed: "
                    + removed.stderr.decode("utf-8", "replace").strip()
                )
        if worktree.exists():
            cleanup_errors.append(f"owned replay worktree still exists: {worktree}")
        try:
            if temporary.exists():
                shutil.rmtree(temporary)
        except OSError as exc:
            cleanup_errors.append(f"owned replay directory removal failed: {exc}")
        if temporary.exists():
            cleanup_errors.append(f"owned replay directory still exists: {temporary}")
        if cleanup_errors:
            raise AssertionError("; ".join(cleanup_errors))


@pytest.mark.parametrize("owner_kind", ["bare", "linked"])
def test_raw_red_git_detached_lifecycle_preserves_owner_configuration(
    raw_red_git_repository: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    owner_kind: str,
) -> None:
    root, _, child = raw_red_git_repository
    owner = root
    _ = _native_red_git(root, "config", "extensions.worktreeConfig", "true")
    if owner_kind == "linked":
        owner = tmp_path / "owner"
        _ = _native_red_git(root, "worktree", "add", "--detach", str(owner), child)
        _ = _native_red_git(owner, "config", "--worktree", "core.bare", "false")
        _ = _native_red_git(owner, "config", "--worktree", "core.worktree", str(owner))
    monkeypatch.setattr(sys.modules[__name__], "REPOSITORY_ROOT", owner)
    before = _native_red_git(root, "worktree", "list", "--porcelain")
    protected = [root / "config"]
    if owner_kind == "linked":
        gitdir = Path(
            _native_red_git(owner, "rev-parse", "--absolute-git-dir").decode().strip()
        )
        protected.extend(
            (gitdir / "config.worktree", gitdir / "index", gitdir / "HEAD")
        )
    saved = {path: path.read_bytes() for path in protected}
    monkeypatch.setenv("GIT_DIR", str(root / "missing-routing-target"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "wrong-worktree"))
    with _detached_fingerprint_replay_worktree(child) as (worktree, temporary):
        assert (worktree / "oracle.py").read_bytes() == b"after\n"
        native = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=worktree, capture_output=True, check=False
        )
        assert native.returncode != 0
        _assert_immutable_replay_checkout(worktree, child)
    assert not worktree.exists() and not temporary.exists()
    with pytest.raises(RuntimeError, match="synthetic body failure"):
        with _detached_fingerprint_replay_worktree(child) as (worktree, temporary):
            raise RuntimeError("synthetic body failure")
    assert not worktree.exists() and not temporary.exists()
    with pytest.raises(AssertionError, match="historical replay checkout changed"):
        with _detached_fingerprint_replay_worktree(child) as (worktree, temporary):
            _ = (worktree / "oracle.py").write_bytes(b"changed\n")
    assert not worktree.exists() and not temporary.exists()
    assert _native_red_git(root, "worktree", "list", "--porcelain") == before
    assert {path: path.read_bytes() for path in protected} == saved


@pytest.mark.parametrize("kind", ["inside", "outside", "file", "dangling", "fifo"])
def test_raw_red_git_rejects_redirected_tracked_parent_directories(
    raw_red_git_repository: tuple[Path, str, str], tmp_path: Path, kind: str
) -> None:
    root, _, child = raw_red_git_repository
    subtree = _native_red_git(root, "rev-parse", f"{child}^{{tree}}").decode().strip()
    tree = (
        _native_red_git(
            root, "mktree", data=f"040000 tree {subtree}\tdirectory\n".encode()
        )
        .decode()
        .strip()
    )
    anchor = (
        _native_red_git(
            root, "commit-tree", tree, "-p", child, data=b"nested tracked parent\n"
        )
        .decode()
        .strip()
    )
    with _detached_fingerprint_replay_worktree(anchor) as (worktree, _):
        directory = worktree / "directory"
        alias = tmp_path / "outside-alias" if kind == "outside" else worktree / "alias"
        _ = directory.rename(alias)
        try:
            if kind in {"inside", "outside"}:
                directory.symlink_to(alias, target_is_directory=True)
                assert (directory / "oracle.py").read_bytes() == b"after\n"
            elif kind == "file":
                _ = directory.write_bytes(b"not a directory\n")
            elif kind == "dangling":
                directory.symlink_to("missing-alias", target_is_directory=True)
            else:
                os.mkfifo(directory)
            _ = _native_red_git(
                worktree, "update-index", "--skip-worktree", "--", "directory/oracle.py"
            )
            assert (
                _native_red_git(
                    worktree, "status", "--porcelain", "--untracked-files=no"
                )
                == b""
            )
            for check in (
                _assert_raw_tracked_checkout,
                _assert_immutable_replay_checkout,
            ):
                with pytest.raises(
                    AssertionError, match="parent directory type changed"
                ):
                    check(worktree, anchor)
        finally:
            directory.unlink(missing_ok=True)
            _ = alias.rename(directory)
            _ = _native_red_git(
                worktree,
                "update-index",
                "--no-skip-worktree",
                "--",
                "directory/oracle.py",
            )


def test_raw_red_git_authenticates_executable_symlink_and_gitlink_types(
    raw_red_git_repository: tuple[Path, str, str],
) -> None:
    root, _, child = raw_red_git_repository
    blob = _native_red_git(root, "rev-parse", f"{child}:oracle.py").decode().strip()
    link = (
        _native_red_git(root, "hash-object", "-w", "--stdin", data=b"oracle.py")
        .decode()
        .strip()
    )
    tree = (
        _native_red_git(
            root,
            "mktree",
            data=(
                f"100644 blob {blob}\toracle.py\n100755 blob {blob}\texecutable\n120000 blob {link}\tlink\n160000 commit {child}\tmodule\n"
            ).encode(),
        )
        .decode()
        .strip()
    )
    anchor = (
        _native_red_git(root, "commit-tree", tree, "-p", child, data=b"tracked types\n")
        .decode()
        .strip()
    )
    with _detached_fingerprint_replay_worktree(anchor) as (worktree, _):
        _assert_raw_tracked_checkout(worktree, anchor)
        executable = worktree / "executable"
        original_mode = executable.stat().st_mode
        try:
            _ = _native_red_git(root, "config", "core.fileMode", "false")
            executable.chmod(original_mode & ~stat.S_IXUSR)
            assert (
                _native_red_git(
                    worktree, "status", "--porcelain", "--untracked-files=no"
                )
                == b""
            )
            with pytest.raises(AssertionError, match="executable mode changed"):
                _assert_raw_tracked_checkout(worktree, anchor)
        finally:
            executable.chmod(original_mode)
            _ = _native_red_git(root, "config", "--unset", "core.fileMode")
        symlink = worktree / "link"
        symlink.unlink()
        symlink.symlink_to("executable")
        try:
            with pytest.raises(AssertionError, match="raw bytes changed"):
                _assert_raw_tracked_checkout(worktree, anchor)
        finally:
            symlink.unlink()
            symlink.symlink_to("oracle.py")
        # LFS/other conversion directives elsewhere must not invalidate raw,
        # unchanged files or require materializing the reference gitlink.
        _ = (root / "info/attributes").write_text(
            "oracle.py filter=lfs diff=lfs -text\n"
        )
        _assert_immutable_replay_checkout(worktree, anchor)


@pytest.mark.parametrize(
    "case",
    [
        "pointer",
        "materialized",
        "wrong-size",
        "wrong-digest",
        "noncanonical",
        "symlink",
        "executable",
    ],
)
def test_raw_red_git_lfs_content_is_bound_to_original_pointer(
    raw_red_git_repository: tuple[Path, str, str], case: str
) -> None:
    root, parent, _ = raw_red_git_repository
    payload = b"materialized sample\n"
    digest = hashlib.sha256(payload).hexdigest()
    size = str(len(payload))
    if case == "noncanonical":
        size = "0" + size
    pointer = (
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {size}\n"
    ).encode("ascii")
    anchor = _raw_red_commit(root, parent, pointer)
    saved_config = (root / "config").read_bytes()
    with _detached_fingerprint_replay_worktree(anchor) as (worktree, _):
        source = worktree / "oracle.py"
        original_mode = source.stat().st_mode
        try:
            _ = (root / "info/attributes").write_text("oracle.py filter=lfs -text\n")
            # Native Git's clean filter deliberately masks every replacement.
            # The independent proof must use the original pointer's identity.
            command = "printf '" + pointer.decode("ascii").replace("\n", "\\n") + "'"
            _ = _native_red_git(root, "config", "filter.lfs.clean", command)
            actual = pointer if case == "pointer" else payload
            if case == "wrong-size":
                actual += b"extra"
            elif case == "wrong-digest":
                actual = b"X" + payload[1:]
            _ = source.write_bytes(actual)
            if case == "symlink":
                source.unlink()
                source.symlink_to("missing-target")
            elif case == "executable":
                source.chmod(original_mode | stat.S_IXUSR)
            else:
                # Materialized LFS checkouts cache the content's actual stat
                # size while retaining the original pointer as the index blob.
                _ = _native_red_git(worktree, "add", "--", "oracle.py")
                assert _native_red_git(worktree, "rev-parse", ":oracle.py") == (
                    _native_red_git(root, "rev-parse", f"{anchor}:oracle.py")
                )
                assert (
                    _native_red_git(
                        worktree, "status", "--porcelain", "--untracked-files=no"
                    )
                    == b""
                )
            if case in {"pointer", "materialized"}:
                _assert_immutable_replay_checkout(worktree, anchor)
            else:
                with pytest.raises(AssertionError, match="historical tracked"):
                    _assert_raw_tracked_checkout(worktree, anchor)
        finally:
            if source.is_symlink():
                source.unlink()
            _ = source.write_bytes(pointer)
            source.chmod(original_mode)
            _ = (root / "config").write_bytes(saved_config)
            (root / "info/attributes").unlink(missing_ok=True)
            _ = _native_red_git(worktree, "add", "--", "oracle.py")


def test_the_fingerprint_replay_environment_replaces_inherited_import_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/untrusted/editable/src")
    blocked = (
        "PYTHONHOME",
        "PYTHONOPTIMIZE",
        "PYTHONIOENCODING",
        "PYTHONUSERBASE",
        "_PYTHON_SYSCONFIGDATA_NAME",
        "__PYVENV_LAUNCHER__",
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "PYTEST_CURRENT_TEST",
        "GIT_DIR",
    )
    for key in blocked:
        monkeypatch.setenv(key, "hostile inherited value")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("SCI004_ENV_CONTROL", "preserved")
    parent = dict(os.environ)
    worktree = tmp_path / "detached"

    replay_env = _closed_fingerprint_replay_environment(worktree)

    assert replay_env["PYTHONPATH"] == str(worktree / "src")
    assert replay_env["PYTHONNOUSERSITE"] == "1"
    assert "/untrusted/editable/src" not in replay_env["PYTHONPATH"]
    assert replay_env["PYTHONSAFEPATH"] == "1"
    assert not set(blocked).intersection(replay_env)
    assert replay_env["OMP_NUM_THREADS"] == "2"
    assert replay_env["SCI004_ENV_CONTROL"] == "preserved"
    assert replay_env["PATH"] == parent["PATH"]
    assert dict(os.environ) == parent


@pytest.mark.parametrize(
    "route", ["PYTHONPATH", "PYTHONUSERBASE", "PYTHONHOME", "PYTHONOPTIMIZE"]
)
def test_python_child_environment_blocks_native_startup_routes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, route: str
) -> None:
    for key in tuple(os.environ):
        if (
            key.startswith(("PYTHON", "_PYTHON", "PYTEST_"))
            or key == "__PYVENV_LAUNCHER__"
        ):
            monkeypatch.delenv(key)
    package = tmp_path / "src/radiosim"
    package.mkdir(parents=True)
    imported = package / "__init__.py"
    _ = imported.write_text("# owned detached import fixture\n")
    marker = tmp_path / "external-startup"
    external = tmp_path / "external"
    external.mkdir()
    if route == "PYTHONUSERBASE":
        monkeypatch.setenv(route, str(external))
        located = subprocess.run(
            [sys.executable, "-c", "import site; print(site.getusersitepackages())"],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )
        external = Path(located.stdout.decode().strip())
        assert external.is_relative_to(tmp_path)
        external.mkdir(parents=True)
    elif route == "PYTHONOPTIMIZE":
        monkeypatch.setenv(route, "1")
    else:
        monkeypatch.setenv(route, str(external))
    if route in {"PYTHONPATH", "PYTHONUSERBASE"}:
        startup = "sitecustomize.py" if route == "PYTHONPATH" else "usercustomize.py"
        _ = (external / startup).write_text(
            f"from pathlib import Path; Path({str(marker)!r}).write_text('external')\n"
        )
    argv = [sys.executable, "-c", "import sys; print(sys.flags.optimize)"]
    native = subprocess.run(argv, cwd=tmp_path, capture_output=True, check=False)
    if route == "PYTHONHOME":
        assert native.returncode != 0
    elif route == "PYTHONOPTIMIZE":
        assert native.returncode == 0 and native.stdout == b"1\n"
    else:
        assert native.returncode == 0 and marker.read_text() == "external"
        marker.unlink()
    safe = _closed_fingerprint_replay_environment(tmp_path)
    checked = subprocess.run(
        argv, cwd=tmp_path, env=safe, capture_output=True, check=False
    )
    assert checked.returncode == 0 and checked.stdout == b"0\n"
    assert checked.stderr == b"" and not marker.exists()
    assert _preflight_fingerprint_replay_import(tmp_path, safe) == imported
    assert not marker.exists()


@pytest.mark.parametrize("route", ["PYTEST_ADDOPTS", "PYTEST_PLUGINS"])
def test_python_child_environment_preserves_pytest_defaults_without_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, route: str
) -> None:
    for key in tuple(os.environ):
        if (
            key.startswith(("PYTHON", "_PYTHON", "PYTEST_"))
            or key == "__PYVENV_LAUNCHER__"
        ):
            monkeypatch.delenv(key)
    source = tmp_path / "src"
    source.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(source))
    marker = tmp_path / "plugin-marker"
    _ = (source / "hostile_plugin.py").write_text(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('loaded')\n"
    )
    report = tmp_path / "plugins.json"
    test = tmp_path / "test_child.py"
    _ = test.write_text(
        "import json\nfrom pathlib import Path\n"
        "def test_child(request):\n"
        "    names = sorted(d.project_name for _, d in request.config.pluginmanager.list_plugin_distinfo())\n"
        f"    Path({str(report)!r}).write_text(json.dumps(names))\n"
    )
    junit = tmp_path / "junit.xml"
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:randomly",
        "-p",
        "no:xdist",
        "--junit-xml",
        str(junit),
        str(test),
    ]
    reference = subprocess.run(argv, cwd=tmp_path, capture_output=True, check=False)
    assert reference.returncode == 0
    expected_plugins = report.read_bytes()
    assert json.loads(expected_plugins), "trusted environment autoload positive control"
    monkeypatch.setenv(
        route, "-k absent_test" if route == "PYTEST_ADDOPTS" else "hostile_plugin"
    )
    native = subprocess.run(argv, cwd=tmp_path, capture_output=True, check=False)
    if route == "PYTEST_ADDOPTS":
        assert native.returncode == 5
    else:
        assert native.returncode == 0 and marker.read_text() == "loaded"
        marker.unlink()
    # Also refuse inherited autoload disablement; preserve the clean reference.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    safe = _closed_fingerprint_replay_environment(tmp_path)
    checked = subprocess.run(
        argv, cwd=tmp_path, env=safe, capture_output=True, check=False
    )
    assert checked.returncode == 0, checked.stderr.decode("utf-8", "replace")
    assert not marker.exists() and report.read_bytes() == expected_plugins
    cases = list(ElementTree.parse(junit).iter("testcase"))
    assert len(cases) == 1 and cases[0].get("name") == "test_child"
    assert not list(cases[0])


def test_the_fingerprint_import_preflight_selects_only_detached_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "radiosim"
    package.mkdir(parents=True)
    imported = package / "__init__.py"
    imported.write_text('"""Detached replay probe fixture."""\n')
    monkeypatch.setenv("PYTHONPATH", str(REPOSITORY_ROOT / "src"))
    replay_env = _closed_fingerprint_replay_environment(tmp_path)

    assert _preflight_fingerprint_replay_import(tmp_path, replay_env) == imported

    outside = tmp_path / "outside.py"
    outside.write_text("# outside the detached package\n")
    with pytest.raises(AssertionError):
        _validate_fingerprint_replay_import(
            f"{outside.resolve()}\n".encode(),
            tmp_path,
        )
    with pytest.raises(AssertionError):
        _validate_fingerprint_replay_import(
            f"{(REPOSITORY_ROOT / 'src' / 'radiosim' / '__init__.py').resolve()}\n".encode(),
            tmp_path,
        )


@pytest.mark.parametrize("link_component", ["src", "radiosim"])
def test_the_fingerprint_import_rejects_source_symlinks_outside_the_tree(
    tmp_path: Path, link_component: str
) -> None:
    worktree = tmp_path / "replay"
    worktree.mkdir()
    external = tmp_path / "unregistered-copy"
    package = external / "radiosim"
    package.mkdir(parents=True)
    imported = package / "__init__.py"
    imported.write_text("# external source must not satisfy historical isolation\n")
    if link_component == "src":
        (worktree / "src").symlink_to(external, target_is_directory=True)
    else:
        (worktree / "src").mkdir()
        (worktree / "src" / "radiosim").symlink_to(package, target_is_directory=True)
    with pytest.raises(AssertionError):
        _validate_fingerprint_replay_import(f"{imported}\n".encode(), worktree)


def test_the_owned_fingerprint_replay_worktree_cleans_success_and_failure() -> None:
    before = _git("worktree", "list", "--porcelain")

    with _detached_fingerprint_replay_worktree(APPROVED_SCI004_D_SHA) as (
        worktree,
        temporary,
    ):
        assert worktree.is_dir()
        assert temporary.is_dir()
    assert _git("worktree", "list", "--porcelain") == before

    with pytest.raises(RuntimeError, match="synthetic child failure"):
        with _detached_fingerprint_replay_worktree(APPROVED_SCI004_D_SHA):
            raise RuntimeError("synthetic child failure")
    assert _git("worktree", "list", "--porcelain") == before


def _assert_fingerprint_replay_result(
    completed: subprocess.CompletedProcess[bytes],
    junit: Path,
    nodeids: tuple[str, ...],
    failure_patterns: tuple[str, ...],
) -> None:
    expected_failures = len(failure_patterns)
    expected_passes = len(nodeids) - expected_failures
    assert completed.returncode == (1 if expected_failures else 0), (
        completed.stdout.decode("utf-8", "replace")
        + completed.stderr.decode("utf-8", "replace")
    )
    rendered = completed.stdout.decode("utf-8", "replace").lower()
    expected_summary = (
        f"{expected_failures} failed, {expected_passes} passed"
        if expected_failures
        else f"{expected_passes} passed"
    )
    assert rendered.strip(), "pytest returned no terminal summary"
    terminal_summary = rendered.rstrip().splitlines()[-1].strip("= ")
    assert re.fullmatch(
        re.escape(expected_summary)
        + r"(?:, \d+ warnings?)? in \d+(?:\.\d+)?s(?: \([0-9:]+\))?",
        terminal_summary,
    )
    assert not re.search(r"\b\d+ (?:skipped|xfailed|xpassed|errors?)\b", rendered)
    root = ElementTree.parse(junit).getroot()
    cases = list(root.iter("testcase"))
    assert [(case.get("classname"), case.get("name")) for case in cases] == [
        (
            node.split("::", 1)[0].removesuffix(".py").replace("/", "."),
            node.split("::", 1)[1],
        )
        for node in nodeids
    ]
    assert not list(root.iter("error"))
    assert not list(root.iter("skipped"))
    assert len(list(root.iter("failure"))) == expected_failures
    for index, case in enumerate(cases):
        failures = case.findall("failure")
        assert len(failures) == (1 if index < expected_failures else 0)
        if failures:
            message = failures[0].get("message") or ""
            assert message.startswith("AssertionError: ")
            assert failure_patterns[index] in message


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "exit",
        "missing",
        "duplicate",
        "reordered",
        "class",
        "frame",
        "error",
        "skip",
        "xfail",
        "xpass",
        "wrong-control",
        "duplicate-failure",
        "inflated-count",
    ],
)
def test_the_replay_result_rejects_any_partition_substitution(
    tmp_path: Path, mutation: str | None
) -> None:
    root = ElementTree.Element("testsuites")
    suite = ElementTree.SubElement(root, "testsuite")
    for index, node in enumerate(FINGERPRINT_NODEIDS):
        case = ElementTree.SubElement(
            suite,
            "testcase",
            classname=node.split("::")[0].removesuffix(".py").replace("/", "."),
            name=node.split("::", 1)[1],
        )
        if index < 2:
            ElementTree.SubElement(
                case,
                "failure",
                message="AssertionError: " + FINGERPRINT_CASE_EXPECTATIONS[index][3],
            )
    stdout = b"2 failed, 3 passed in 1.00s"
    returncode = 1
    if mutation == "exit":
        returncode = 0
    elif mutation == "inflated-count":
        stdout = b"12 failed, 3 passed in 1.00s"
    elif mutation == "missing":
        suite.remove(suite[-1])
    elif mutation == "duplicate":
        suite.append(ElementTree.fromstring(ElementTree.tostring(suite[-1])))
    elif mutation == "reordered":
        first = suite[0]
        suite.remove(first)
        suite.append(first)
    elif mutation == "class":
        suite[0].set("classname", "wrong.module")
    elif mutation == "frame":
        suite[0][0].set("message", "RuntimeError: frame certificate rejected")
    elif mutation in {"error", "skip"}:
        ElementTree.SubElement(suite[-1], "error" if mutation == "error" else "skipped")
    elif mutation in {"xfail", "xpass"}:
        stdout += b", 1 " + (b"xfailed" if mutation == "xfail" else b"xpassed")
    elif mutation == "wrong-control":
        failure = suite[0][0]
        suite[0].remove(failure)
        suite[-1].append(failure)
    elif mutation == "duplicate-failure":
        ElementTree.SubElement(suite[0], "failure", message="AssertionError: extra")
    junit = tmp_path / "partition.xml"
    ElementTree.ElementTree(root).write(junit)
    completed = subprocess.CompletedProcess(
        ["synthetic partition"], returncode, stdout, b""
    )
    arguments = (
        completed,
        junit,
        FINGERPRINT_NODEIDS,
        tuple(row[3] for row in FINGERPRINT_CASE_EXPECTATIONS),
    )
    if mutation is None:
        _assert_fingerprint_replay_result(*arguments)
    else:
        with pytest.raises(AssertionError):
            _assert_fingerprint_replay_result(*arguments)


@pytest.mark.parametrize("invalid", ["skip", "inflated-count"])
def test_the_separate_general_replay_requires_one_real_pass(
    tmp_path: Path, invalid: str
) -> None:
    junit = tmp_path / "general.xml"
    root = ElementTree.Element("testsuite")
    case = ElementTree.SubElement(
        root,
        "testcase",
        classname="tests.characterization.test_sci004_mmode",
        name=GENERAL_TRANSITION_NODEID.split("::", 1)[1],
    )
    completed = subprocess.CompletedProcess(
        ["synthetic general"], 0, b"1 passed in 0.10s", b""
    )
    ElementTree.ElementTree(root).write(junit)
    _assert_fingerprint_replay_result(
        completed, junit, (GENERAL_TRANSITION_NODEID,), ()
    )
    if invalid == "skip":
        ElementTree.SubElement(case, "skipped")
    else:
        completed.stdout = b"11 passed in 0.10s"
    ElementTree.ElementTree(root).write(junit)
    with pytest.raises(AssertionError):
        _assert_fingerprint_replay_result(
            completed, junit, (GENERAL_TRANSITION_NODEID,), ()
        )


def test_the_fresh_r3_detached_replay_reproduces_the_five_node_partition() -> None:
    anchor = resolve_r3_replay_anchor()
    # Before S freezes its parent, the authenticated committed R tip is still
    # a valid replay target. It must already have the complete R path inventory.
    assert anchor.role in {"r3", "pre-commit-authoring-tip"}
    phase_history.describe_phase_range(
        phase_history.PREREQUISITE_TIP_SHA,
        anchor.commit,
        "red",
        root=REPOSITORY_ROOT,
    )
    with _detached_fingerprint_replay_worktree(anchor.commit) as (
        worktree,
        temporary,
    ):
        replay_env = _closed_fingerprint_replay_environment(worktree)
        imported = _preflight_fingerprint_replay_import(worktree, replay_env)
        assert imported.is_relative_to((worktree / "src" / "radiosim").resolve())
        junit = temporary / "junit.xml"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:randomly",
                "-p",
                "no:xdist",
                "--junit-xml",
                str(junit),
                *FINGERPRINT_NODEIDS,
            ],
            cwd=worktree,
            env=replay_env,
            capture_output=True,
            check=False,
        )
        _assert_fingerprint_replay_result(
            completed,
            junit,
            FINGERPRINT_NODEIDS,
            tuple(row[3] for row in FINGERPRINT_CASE_EXPECTATIONS),
        )
        general_junit = temporary / "general-junit.xml"
        general = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:randomly",
                "-p",
                "no:xdist",
                "--junit-xml",
                str(general_junit),
                GENERAL_TRANSITION_NODEID,
            ],
            cwd=worktree,
            env=replay_env,
            capture_output=True,
            check=False,
        )
        _assert_fingerprint_replay_result(
            general,
            general_junit,
            (GENERAL_TRANSITION_NODEID,),
            (),
        )


def test_the_retained_earlier_red_records_are_untouched_by_this_slice() -> None:
    """Section 13.7: a phase may not edit a previous phase's retained artifact."""
    for relative in (
        "docs/development/sci004_mmode_phase1_red_failures.json",
        M2_RED_RECORD_PATH,
    ):
        path = REPOSITORY_ROOT / relative
        assert path.is_file(), relative
        assert path.read_bytes() == _tree_blob("HEAD", relative), relative

    m2_document = json.loads(
        (REPOSITORY_ROOT / M2_RED_RECORD_PATH).read_bytes().decode("utf-8")
    )
    _raw, m3_document = read_record()
    assert m2_document["phase"] == "M2"
    assert m3_document["phase"] == "M3"
    assert m2_document["schema_version"] != m3_document["schema_version"]


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
