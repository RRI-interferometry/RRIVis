"""Strict validator for the retained SCI-004 phase-M3 red-failure record.

``docs/development/sci004_mmode_design.md`` Section 14.1 fixes the schema of
``docs/development/sci004_mmode_phase3_red_failures.json`` and says the phase red
validator "authenticates the file bytes, schema literal, node set, command
hashes, pre-fix SHA, protected hashes, and expected non-zero outcomes **before**
``S`` is allowed to start".

Like its M1 and M2 counterparts this module deliberately **does not re-run the
red nodes**. They are red by construction at ``R3``; executing them here would
fail the suite a second time and would authenticate nothing the record does not
already carry. The record's own bytes are what get checked -- re-serialized under
Section 14's canonical rules and compared to the raw file -- with three
independent cross-checks that make the comparison meaningful:

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

from tests.unit.test_sci004_phase3_dependency import (
    APPROVED_SCI004_A2_SHA,
    APPROVED_SCI004_D_SHA,
    APPROVED_SCI004_G3_SHA,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

RECORD_PATH = "docs/development/sci004_mmode_phase3_red_failures.json"
M2_RED_RECORD_PATH = "docs/development/sci004_mmode_phase2_red_failures.json"
CERTIFICATE_PATH = "docs/development/sci004_mmode_phase3_sci005_dependency.json"

SCHEMA_VERSION = "radiosim.sci004.mmode-phase3-red-failures.v1"
PHASE = "M3"
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

#: Section 13.5's complete ``R3`` writable list.
R3_AUTHORIZED_PATHS: frozenset[str] = frozenset(
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
    }
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
    """Section 14.0/14.4: ``design_sha`` is the binding, and ``R3^ == G3``.

    Section 14.4's ``R3^==G3`` is an unstarred sole direct-parent edge, so the
    tree the observations were made from is the gate tip exactly. The operative
    ``D`` precedes it through the header-enumerated chain the dependency
    validator authenticates.
    """
    assert record["design_sha"] == APPROVED_SCI004_D_SHA

    observed = str(record["pre_fix_source_sha"])
    assert _SHA1.match(observed) is not None
    assert observed == APPROVED_SCI004_G3_SHA
    assert observed == APPROVED_SCI004_A2_SHA
    assert _peel_to_commit(observed) == observed
    assert _is_ancestor(APPROVED_SCI004_D_SHA, observed)


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
    assert set(paths) <= R3_AUTHORIZED_PATHS
    assert RECORD_PATH in paths
    assert CERTIFICATE_PATH in paths


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
        assert relative in R3_AUTHORIZED_PATHS, nodeid
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
    """Section 12.2's tenth family: "every new family" is a per-family obligation."""
    from tests.characterization.test_sci004_mmode import SECTION_11_FAMILIES

    identifiers = {str(case["case_id"]) for case in record["cases"]}

    assert len(SECTION_11_FAMILIES) == 7
    for family_id in SECTION_11_FAMILIES:
        assert f"m3.characterization.family-record.{family_id}" in identifiers


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
