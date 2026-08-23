"""Strict validator for the retained SCI-004 phase-M1 red-failure record.

``docs/development/sci004_mmode_design.md`` Section 14.1 fixes the schema of
``docs/development/sci004_mmode_phase1_red_failures.json`` and says the phase red
validator "authenticates the file bytes, schema literal, node set, command
hashes, pre-fix SHA, protected hashes, and expected non-zero outcomes **before**
``S`` is allowed to start". That is what this module does, and it is the whole of
what it does.

It deliberately **does not re-run the red nodes**. They are red by construction
at ``R1``; executing them here would fail the suite a second time and would
authenticate nothing the record does not already carry. The record is the
evidence, and the record's own bytes are what get checked -- re-serialized under
Section 14's canonical rules and compared to the raw file, so a hand-edited or
re-ordered document cannot pass.

The two independent cross-checks that make that comparison meaningful are:

* every case row's ``fixture_identity_sha256`` is recomputed here from the
  Section 14.0 preimage, so a row cannot claim a fixture it does not name; and
* the node set is compared against the ``SCI004_RED_CASES`` tables the red test
  modules themselves declare, so a node cannot be quietly dropped from the record
  while remaining red in the tree, and cannot appear twice.

Section 13.2's retained WP-7 certificate is checked by raw digest for the same
reason: the M1 gate's authority is the exact upstream bytes, and the red slice
retains them.

These tests pass at ``R1``.
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

from tests.unit.test_sci004_phase1_dependency import (
    APPROVED_SCI004_D_SHA,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

RECORD_PATH = "docs/development/sci004_mmode_phase1_red_failures.json"
CERTIFICATE_PATH = "docs/development/sci004_mmode_phase1_wp7_dependency.json"

#: Section 13.2: the retained M1 certificate is the already-authenticated
#: SCI-005 Stage-1 line, byte for byte.
RETAINED_CERTIFICATE_SHA256 = (
    "1bc43ce8b08192753ca2f2ace23effb2b33bfdc6d896037bf6c70e2e5eba734b"
)

SCHEMA_VERSION = "radiosim.sci004.mmode-phase1-red-failures.v1"
PHASE = "M1"
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

#: Section 13.3's ``R1`` writable list.
R1_AUTHORIZED_PATHS: frozenset[str] = frozenset(
    {
        "docs/development/sci004_mmode_phase1_red_failures.json",
        "docs/development/sci004_mmode_phase1_wp7_dependency.json",
        "tests/characterization/test_tier6_current_behavior.py",
        "tests/characterization/test_tier7_current_behavior.py",
        "tests/integration/test_sci004_mmode.py",
        "tests/unit/test_core/test_sci004_era_grid.py",
        "tests/unit/test_core/test_sci004_frame.py",
        "tests/unit/test_core/test_sci004_scalar_harmonics.py",
        "tests/unit/test_core/test_sci004_transfer.py",
        "tests/unit/test_io/test_sci004_config.py",
        "tests/unit/test_sci004_phase1_dependency.py",
        "tests/unit/test_sci004_phase1_red_failures.py",
        "tests/unit/test_simulator/test_sci004_strategy.py",
        "tests/unit/test_tier7_jones_acceptance.py",
        "tools/sci004_mmode_phase1_red.py",
    }
)

#: The red modules whose declared tables the node set is compared against.
RED_MODULES: tuple[str, ...] = (
    "tests.unit.test_io.test_sci004_config",
    "tests.unit.test_core.test_sci004_era_grid",
    "tests.unit.test_core.test_sci004_frame",
    "tests.unit.test_core.test_sci004_scalar_harmonics",
    "tests.unit.test_core.test_sci004_transfer",
    "tests.unit.test_simulator.test_sci004_strategy",
    "tests.integration.test_sci004_mmode",
)

#: The four claim categories Section 14.1 requires.
REQUIRED_CLAIM_CATEGORIES: tuple[str, ...] = (
    "acceptance",
    "fingerprint",
    "performance",
    "production",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class RedRecordSchemaError(AssertionError):
    """The retained red-failure record failed strict validation."""


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
    """The case tables the red test modules themselves declare."""
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    declared: dict[str, dict[str, Any]] = {}
    for module_name in RED_MODULES:
        module = importlib.import_module(module_name)
        for case in module.SCI004_RED_CASES:
            nodeid = str(case["test_nodeid"])
            if nodeid in declared:
                raise RedRecordSchemaError(f"{nodeid} is declared twice")
            declared[nodeid] = dict(case)
    return declared


# --- Section 13.2: the retained upstream certificate --------------------------


def test_the_retained_wp7_dependency_certificate_has_its_exact_raw_digest() -> None:
    """Section 13.2: ``R1`` retains the exact upstream certificate bytes."""
    path = REPOSITORY_ROOT / CERTIFICATE_PATH
    assert path.is_file() and not path.is_symlink()
    raw = path.read_bytes()

    assert hashlib.sha256(raw).hexdigest() == RETAINED_CERTIFICATE_SHA256
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1


# --- Section 14.1: the record's own bytes -------------------------------------


def test_the_record_is_exactly_its_canonical_serialization() -> None:
    """Section 14: sorted keys, ``,``/``:``, ASCII, no whitespace or trailing LF."""
    raw, document = read_record()

    assert canonical_json_bytes(document) == raw


def test_the_record_carries_the_exact_top_level_key_set(
    record: dict[str, Any],
) -> None:
    """Section 14.1's frozen envelope."""
    assert set(record) == TOP_LEVEL_KEYS
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["phase"] == PHASE
    assert record["status"] == STATUS
    assert _UTC_STAMP.match(str(record["generated_at_utc"])) is not None


def test_the_record_binds_the_frozen_design_and_pre_fix_source(
    record: dict[str, Any],
) -> None:
    """Section 13.7 (post-source retention) with Section 14.0/14.1.

    Once the phase ``S`` exists, redness can no longer be observed in the
    operative tree, so the retained record keeps its last genuinely
    observed bytes: its ``design_sha`` and ``pre_fix_source_sha`` name the
    header-enumerated chain commit from whose tree the observations were
    genuinely made -- a tree that must predate the committed production --
    connected to the operative ``D`` through the chain. Fabricating an
    ``expected-red-confirmed`` observation against a tree where nothing is
    red is exactly what this rule forbids.
    """
    from tests.unit.test_sci004_phase1_dependency import SCI004_DESIGN_CHAIN

    chain_shas = [entry.sha for entry in SCI004_DESIGN_CHAIN]

    assert record["design_sha"] in chain_shas
    assert record["pre_fix_source_sha"] == record["design_sha"]
    # The observation tree must genuinely predate the committed production.
    listing = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            record["design_sha"],
            "src/radiosim/core/",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "src/radiosim/core/mmode/" not in listing
    # And it must connect to the operative ``D`` through the chain.
    assert chain_shas.index(record["design_sha"]) < len(chain_shas)
    assert APPROVED_SCI004_D_SHA == chain_shas[-1]


def test_the_self_reference_is_a_null_sha_with_its_exact_reason(
    record: dict[str, Any],
) -> None:
    """Section 14.1/14.4: an ``R`` artifact uses a null self SHA, bound by ``E``."""
    assert record["red_commit_sha"] is None
    assert record["red_commit_sha_reason"] == RED_COMMIT_SHA_REASON


def test_the_protected_source_is_declared_clean_and_the_diff_is_authorized(
    record: dict[str, Any],
) -> None:
    """Section 14.1: an uncommitted red artifact never claims a globally clean tree.

    The generator hashes every protected path outside Section 13.3's ``R1`` list
    before and after execution and records only the authorized diff, so
    ``protected_source_clean`` is a statement about the *protected* set, not
    about the working tree as a whole.
    """
    assert record["protected_source_clean"] is True

    paths = record["authorized_red_paths"]
    assert isinstance(paths, list)
    assert all(isinstance(path, str) and path for path in paths)
    assert paths == sorted(set(paths))
    assert set(paths) <= R1_AUTHORIZED_PATHS
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


def test_every_phase_red_node_appears_exactly_once_and_lives_in_the_r1_list(
    record: dict[str, Any],
    declared_nodes: dict[str, dict[str, Any]],
) -> None:
    """Section 14.1: the record's node set equals the declared red inventory.

    Comparing against the tables the red modules declare -- rather than against a
    list restated here -- is what stops a node from being quietly dropped from
    the record while it is still red in the tree.
    """
    nodes = [str(case["test_nodeid"]) for case in record["cases"]]
    identifiers = [str(case["case_id"]) for case in record["cases"]]

    assert len(set(nodes)) == len(nodes)
    assert len(set(identifiers)) == len(identifiers)
    assert set(nodes) == set(declared_nodes)
    for nodeid in nodes:
        relative, separator, name = nodeid.partition("::")
        assert separator == "::" and name, nodeid
        assert relative in R1_AUTHORIZED_PATHS, nodeid
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
    """Every red oracle file Section 13.3 authorizes contributes at least one node."""
    covered = {str(case["test_nodeid"]).split("::", 1)[0] for case in record["cases"]}

    assert covered == {
        "tests/characterization/test_tier7_current_behavior.py",
        "tests/integration/test_sci004_mmode.py",
        "tests/unit/test_core/test_sci004_era_grid.py",
        "tests/unit/test_core/test_sci004_frame.py",
        "tests/unit/test_core/test_sci004_scalar_harmonics.py",
        "tests/unit/test_core/test_sci004_transfer.py",
        "tests/unit/test_io/test_sci004_config.py",
        "tests/unit/test_simulator/test_sci004_strategy.py",
        "tests/unit/test_tier7_jones_acceptance.py",
    }


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
