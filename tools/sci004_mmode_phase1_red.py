#!/usr/bin/env python
"""Generate the SCI-004 phase-M1 retained red-failure record.

``docs/development/sci004_mmode_design.md`` Section 12 opens every
implementation phase with "a red-test commit and a retained record naming the
node ID, expected equation/behavior, observed pre-fix failure, and why the
fixture is not defective". Section 14.1 freezes that record's schema; this tool
produces it and nothing else::

    pixi run python tools/sci004_mmode_phase1_red.py generate

**Why the tool's own imports are standard library only.** It follows
``tools/wp7_perf001_cpu_evidence.py`` and ``tools/sci005_stage1_acceptance.py``:
a record-critical generator must not depend on a package that is merely
transitively present, because a lock update could drop it and turn a hard
refusal into an import error. The one deliberate exception is that the tool
*loads the red test modules as data* -- they are the authoritative location of
the Section 14.1 case table and of the exact fixture bytes the record hashes --
and those modules import ``pytest``. Loading them is intrinsic to the job: a
case table transcribed into this file could drift from the node it describes.
The offline IERS resource is read through ``importlib.resources`` exactly as
Section 3.1 prescribes, so ``environment.iers_table_sha256`` names the same
bytes the frame would install.

**What the record is, and is not.** It records that a named set of nodes failed
in a named way at a named clean source SHA, that every protected path outside
Section 13.3's ``R1`` list was byte-identical before and after the run, and that
each fixture is excluded from defect by a control that passed in the same
invocation. It licenses no production, acceptance, fingerprint, or performance
claim; ``claims_not_licensed`` says so in the record itself.

Generation is atomic and refuses to overwrite. Any unconfirmed case -- a
skipped, xfailed, unexpectedly passed, collection-only, or unrelated failure,
or a green control that did not pass -- aborts before a byte is written.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

PHASE = "M1"
SCHEMA_VERSION = "radiosim.sci004.mmode-phase1-red-failures.v1"
STATUS = "expected-red-confirmed"
RED_COMMIT_SHA_REASON = "self-reference: E binds the containing R commit"

OUTPUT_PATH = "docs/development/sci004_mmode_phase1_red_failures.json"
DEPENDENCY_VALIDATOR_PATH = "tests/unit/test_sci004_phase1_dependency.py"
RETAINED_CERTIFICATE_PATH = "docs/development/sci004_mmode_phase1_wp7_dependency.json"

#: Section 13.3's complete ``R1`` writable list, sorted. Every path outside it
#: is protected and must be byte-identical before and after this run.
R1_AUTHORIZED_PATHS: tuple[str, ...] = tuple(
    sorted(
        (
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
        )
    )
)

#: The red modules that declare ``SCI004_RED_CASES``, in record order.
RED_MODULES: tuple[str, ...] = (
    "tests.unit.test_io.test_sci004_config",
    "tests.unit.test_core.test_sci004_era_grid",
    "tests.unit.test_core.test_sci004_frame",
    "tests.unit.test_core.test_sci004_scalar_harmonics",
    "tests.unit.test_core.test_sci004_transfer",
    "tests.unit.test_simulator.test_sci004_strategy",
    "tests.integration.test_sci004_mmode",
)

#: Section 14.1's five legal outcome kinds.
FAILURE_KINDS: frozenset[str] = frozenset(
    {"assertion", "exception", "import", "missing-symbol", "schema"}
)

#: The deterministic classification from a fully qualified exception class to a
#: Section 14.1 kind. Anything unlisted is a plain ``exception``.
KIND_BY_EXCEPTION: Mapping[str, str] = {
    "builtins.ModuleNotFoundError": "import",
    "builtins.ImportError": "missing-symbol",
    "builtins.AttributeError": "missing-symbol",
    "builtins.NameError": "missing-symbol",
    "builtins.AssertionError": "assertion",
    "_pytest.outcomes.Failed": "assertion",
    "radiosim.io.config_resolution.ConfigSchemaError": "schema",
    "radiosim.io.config_resolution.ConfigSourceError": "schema",
    "radiosim.io.config_resolution.ConfigParseError": "schema",
}

#: Section 14.1: sorted, unique, non-empty, and covering production, acceptance,
#: fingerprint, and performance.
CLAIMS_NOT_LICENSED: tuple[str, ...] = (
    "acceptance: this record expresses no phase-M1 acceptance verdict and "
    "unlocks no successor commit",
    "fingerprint: no characterization fingerprint family is added, pinned, or "
    "changed by this record",
    "performance: no timing, speedup, memory, or accelerator advantage is "
    "measured or claimed here",
    "production: no m-mode production code exists at this source SHA, so every "
    "recorded failure is the absence of it",
)

#: Section 14.2's ``numeric_packages`` set, exactly.
NUMERIC_PACKAGES: tuple[str, ...] = ("dask", "healpy", "jax", "numpy", "scipy")

IERS_PACKAGE = "astropy_iers_data"
IERS_RESOURCE = "data/finals2000A.all"


class RedRecordError(RuntimeError):
    """The phase-M1 red record could not be generated as specified."""


# --- Section 14 canonical JSON ------------------------------------------------


def _es_number(value: float | int) -> str:
    """Serialize one finite number with RFC 8785 / ECMAScript shortest round trip.

    Python's ``repr`` is already shortest-round-trip, but it spells an integral
    float ``1.0`` and switches to an exponent below ``1e-4``; ECMAScript spells
    those ``1`` and ``0.0001``. Section 14 rejects the alternate spellings, so
    the two differences are normalized here rather than hoped away.
    """
    if isinstance(value, bool):
        raise RedRecordError("a boolean is not a JSON number")
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise RedRecordError("NaN and Infinity are forbidden in a canonical record")
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
        items = sorted(value.items(), key=lambda item: item[0])
        for key, _ in items:
            if not isinstance(key, str):
                raise RedRecordError("canonical JSON object keys must be strings")
        return (
            "{"
            + ",".join(
                f"{json.dumps(key, ensure_ascii=True)}:{_canonical(item)}"
                for key, item in items
            )
            + "}"
        )
    if isinstance(value, Sequence):
        return "[" + ",".join(_canonical(item) for item in value) + "]"
    raise RedRecordError(f"cannot canonicalize {type(value).__name__}")


def domain_digest(domain: str, payload: bytes) -> str:
    """Section 14.0's ``D(d, p) = SHA256(d || NUL || U64(len(p)) || p)``."""
    if not domain or not domain.isascii() or "\x00" in domain:
        raise RedRecordError(f"invalid digest domain {domain!r}")
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + struct.pack(">Q", len(payload)) + payload
    ).hexdigest()


def fixture_identity_sha256(
    *,
    phase: str,
    fixture_id: str,
    requirement_id: str,
    test_nodeid: str,
    pre_fix_source_sha: str,
    invalid_config_raw_sha256: str,
) -> str:
    """Section 14.0's red fixture identity, over exactly its six named fields.

    The record row calls the first field ``case_id``; it is the same value, and
    is what makes the identity unique per case rather than per node.
    """
    return domain_digest(
        "radiosim.sci004-red-fixture.v1",
        canonical_json_bytes(
            {
                "phase": phase,
                "fixture_id": fixture_id,
                "requirement_id": requirement_id,
                "test_nodeid": test_nodeid,
                "pre_fix_source_sha": pre_fix_source_sha,
                "invalid_config_raw_sha256": invalid_config_raw_sha256,
            }
        ),
    )


# --- Git and filesystem -------------------------------------------------------


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RedRecordError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def _frozen_binding(name: str) -> str:
    """Read one frozen constant from the single file Section 14.0 authorises.

    Reading it rather than restating it is the point: Section 13.2 permits
    exactly one assignment of each binding, and a generator carrying its own
    copy would be a second, silently divergent one.
    """
    source = (REPOSITORY_ROOT / DEPENDENCY_VALIDATOR_PATH).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(
            node.value.value, str
        ):
            raise RedRecordError(f"{name} is not a string constant")
        value = node.value.value
        if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            raise RedRecordError(f"{name} is not a 40-character lower-case git sha")
        return value
    raise RedRecordError(f"{name} is not bound in {DEPENDENCY_VALIDATOR_PATH}")


def _tracked_entries() -> list[tuple[str, str, str]]:
    """Return ``(mode, object_id, path)`` for every tracked index entry."""
    listing = _git("ls-files", "-s", "-z")
    entries: list[tuple[str, str, str]] = []
    for record in listing.split("\0"):
        if not record:
            continue
        metadata, _tab, path = record.partition("\t")
        mode, object_id, _stage = metadata.split()
        entries.append((mode, object_id, path))
    return sorted(entries, key=lambda entry: entry[2])


def _protected_digest() -> str:
    """One digest over every tracked path outside the ``R1`` authorized list.

    A regular file is hashed from the working tree, because that is what detects
    an edit. A gitlink -- the 41 third-party checkouts under ``simulators/`` are
    submodules, not files -- is represented by its recorded commit id, which is
    the only thing the superproject owns.
    """
    authorized = set(R1_AUTHORIZED_PATHS)
    rows: list[dict[str, str]] = []
    for mode, object_id, relative in _tracked_entries():
        if relative in authorized:
            continue
        if mode == "160000":
            rows.append({"path": relative, "sha256": f"gitlink:{object_id}"})
            continue
        target = REPOSITORY_ROOT / relative
        if mode == "120000":
            rows.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(
                        os.readlink(target).encode("utf-8")
                    ).hexdigest(),
                }
            )
            continue
        if target.is_symlink() or not target.is_file():
            raise RedRecordError(f"protected path {relative} is not a regular file")
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    return domain_digest(
        "radiosim.sci004.protected-source.v1", canonical_json_bytes(rows)
    )


def _changed_paths() -> tuple[str, ...]:
    listing = _git("status", "--porcelain=v1", "--untracked-files=all", "-z")
    changed: set[str] = set()
    entries = [entry for entry in listing.split("\0") if entry]
    for entry in entries:
        if len(entry) < 4:
            continue
        changed.add(entry[3:])
    return tuple(sorted(changed))


# --- the case inventory -------------------------------------------------------


def _load_cases() -> tuple[list[dict[str, Any]], dict[str, tuple[str, ...]]]:
    """Load the declared case table and green controls from the red modules."""
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    cases: list[dict[str, Any]] = []
    controls: dict[str, tuple[str, ...]] = {}
    for module_name in RED_MODULES:
        module = importlib.import_module(module_name)
        declared = getattr(module, "SCI004_RED_CASES", None)
        if not declared:
            raise RedRecordError(f"{module_name} declares no SCI004_RED_CASES")
        for case in declared:
            missing = {
                "case_id",
                "requirement_id",
                "test_nodeid",
                "expected_failure_kind",
                "expected_failure_pattern",
                "fixture_defect_excluded_by",
                "fixture_bytes",
            } - set(case)
            if missing:
                raise RedRecordError(
                    f"{module_name}: case is missing {sorted(missing)}"
                )
            if case["expected_failure_kind"] not in FAILURE_KINDS:
                raise RedRecordError(
                    f"{case['case_id']}: illegal kind {case['expected_failure_kind']!r}"
                )
            cases.append(dict(case))
        controls[module_name] = tuple(getattr(module, "SCI004_RED_GREEN_CONTROLS", ()))
    identifiers = [case["case_id"] for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise RedRecordError("case_id values must be unique")
    nodes = [case["test_nodeid"] for case in cases]
    if len(set(nodes)) != len(nodes):
        raise RedRecordError("every phase red node must appear exactly once")
    return cases, controls


def _group_by_file(
    cases: Sequence[Mapping[str, Any]],
    controls: Mapping[str, tuple[str, ...]],
) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """Group node IDs by test file, in declaration order, with their controls."""
    order: list[str] = []
    red_by_file: dict[str, list[str]] = {}
    for case in cases:
        relative = str(case["test_nodeid"]).split("::", 1)[0]
        if relative not in red_by_file:
            red_by_file[relative] = []
            order.append(relative)
        red_by_file[relative].append(str(case["test_nodeid"]))
    controls_by_file: dict[str, list[str]] = {}
    for nodeids in controls.values():
        for nodeid in nodeids:
            relative = nodeid.split("::", 1)[0]
            controls_by_file.setdefault(relative, []).append(nodeid)
    return [
        (
            relative,
            tuple(red_by_file[relative]),
            tuple(controls_by_file.get(relative, ())),
        )
        for relative in order
    ]


# --- pytest execution and junit parsing ---------------------------------------


def _run_pytest(
    nodeids: Sequence[str], junit_path: Path
) -> tuple[dict[str, Any], bytes, bytes]:
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        "0",
        "-p",
        "no:cacheprovider",
        "--junit-xml",
        str(junit_path),
        *nodeids,
    ]
    started = datetime.now(UTC)
    clock = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    duration = time.perf_counter() - clock
    row = {
        "argv": list(argv),
        "cwd": ".",
        "pixi_environment": os.environ.get("PIXI_ENVIRONMENT_NAME", "default"),
        "started_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_seconds": round(duration, 6),
        "exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
    }
    return row, completed.stdout, completed.stderr


def _parse_junit(junit_path: Path) -> dict[str, dict[str, str]]:
    """Return ``{test name: {outcome, exception_type, message}}`` from junit XML."""
    tree = ElementTree.parse(junit_path)
    observed: dict[str, dict[str, str]] = {}
    for testcase in tree.iter("testcase"):
        name = testcase.get("name") or ""
        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped = testcase.find("skipped")
        if skipped is not None:
            observed[name] = {"outcome": "skipped", "type": "", "message": ""}
            continue
        node = failure if failure is not None else error
        if node is None:
            observed[name] = {"outcome": "passed", "type": "", "message": ""}
            continue
        raw = (node.get("message") or "").strip()
        first_line = raw.splitlines()[0] if raw else ""
        observed[name] = {
            "outcome": "collected" if error is not None else "failed",
            "type": _exception_type(node.get("type"), first_line),
            "message": first_line,
        }
    return observed


_QUALIFIED_NAME = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*): "
)


def _exception_type(declared: str | None, first_line: str) -> str:
    """Derive the fully qualified exception class Section 14.1 requires.

    pytest's junit report carries no ``type`` attribute for an assertion, and a
    rewritten bare assertion's crash line can begin with ``assert`` rather than
    with the class name. Both shapes are handled explicitly; anything else fails
    loudly rather than being guessed into the record.
    """
    if declared:
        return declared if "." in declared else f"builtins.{declared}"
    match = _QUALIFIED_NAME.match(first_line)
    if match is not None:
        name = match.group("name")
        return name if "." in name else f"builtins.{name}"
    if first_line.startswith("assert"):
        return "builtins.AssertionError"
    raise RedRecordError(
        f"cannot derive a fully qualified exception class from {first_line!r}"
    )


def _classify(exception_type: str) -> str:
    return KIND_BY_EXCEPTION.get(exception_type, "exception")


# --- the environment object ---------------------------------------------------


def _distribution_version(name: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _environment() -> dict[str, Any]:
    """Section 14.2's exact environment object."""
    import importlib.resources as resources

    resource = resources.files(IERS_PACKAGE) / IERS_RESOURCE
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pixi_environment": os.environ.get("PIXI_ENVIRONMENT_NAME", "default"),
        "pixi_lock_sha256": hashlib.sha256(
            (REPOSITORY_ROOT / "pixi.lock").read_bytes()
        ).hexdigest(),
        "astropy_version": _distribution_version("astropy"),
        # ``erfa`` is the module; ``pyerfa`` is the distribution that ships it.
        "erfa_version": _distribution_version("pyerfa"),
        "iers_package_version": _distribution_version("astropy-iers-data"),
        "iers_table_sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
        "numeric_packages": {
            name: _distribution_version(name) for name in NUMERIC_PACKAGES
        },
    }


# --- generation ---------------------------------------------------------------


def generate(output: Path) -> None:
    if output.exists():
        raise RedRecordError(f"{output} already exists; generation never overwrites")

    design_sha = _frozen_binding("APPROVED_SCI004_D_SHA")
    # Section 13.2 (as corrected): the red slice directly parents the
    # operative correction commit, so the pre-fix source is the operative
    # ``D`` rather than the gate anchor ``G1``.
    pre_fix_source_sha = _frozen_binding("APPROVED_SCI004_D_SHA")
    head = _git("rev-parse", "HEAD").strip()
    if head != pre_fix_source_sha:
        raise RedRecordError(
            f"HEAD is {head}; the red record is generated at exact G1 "
            f"{pre_fix_source_sha}"
        )
    certificate = REPOSITORY_ROOT / RETAINED_CERTIFICATE_PATH
    if not certificate.is_file() or certificate.is_symlink():
        raise RedRecordError(f"{RETAINED_CERTIFICATE_PATH} must be retained first")

    cases, controls = _load_cases()
    groups = _group_by_file(cases, controls)
    protected_before = _protected_digest()

    commands: list[dict[str, Any]] = []
    observed: dict[str, dict[str, str]] = {}
    command_index_by_file: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="sci004-m1-red-") as scratch:
        for index, (relative, red_nodes, green_nodes) in enumerate(groups):
            junit_path = Path(scratch) / f"junit-{index}.xml"
            row, stdout, _stderr = _run_pytest((*red_nodes, *green_nodes), junit_path)
            if row["exit_code"] == 0:
                raise RedRecordError(
                    f"{relative}: pytest exited zero, so nothing was red"
                )
            if not junit_path.is_file():
                raise RedRecordError(
                    f"{relative}: pytest produced no junit report\n"
                    f"{stdout.decode('utf-8', 'replace')[-4000:]}"
                )
            results = _parse_junit(junit_path)
            for nodeid in green_nodes:
                name = nodeid.split("::", 1)[1]
                entry = results.get(name)
                if entry is None or entry["outcome"] != "passed":
                    raise RedRecordError(
                        f"green control {nodeid} did not pass: {entry}"
                    )
            for nodeid in red_nodes:
                observed[nodeid] = results.get(
                    nodeid.split("::", 1)[1],
                    {"outcome": "absent", "type": "", "message": ""},
                )
            command_index_by_file[relative] = index
            commands.append(row)

    protected_after = _protected_digest()
    if protected_before != protected_after:
        raise RedRecordError(
            "a protected path outside the R1 list changed during generation"
        )

    rows: list[dict[str, Any]] = []
    for case in cases:
        nodeid = str(case["test_nodeid"])
        relative = nodeid.split("::", 1)[0]
        entry = observed[nodeid]
        if entry["outcome"] != "failed":
            raise RedRecordError(
                f"{nodeid}: observed {entry['outcome']!r}; a skipped, xfailed, "
                "unexpectedly passed, collection-only, or absent outcome is invalid"
            )
        kind = _classify(entry["type"])
        if kind != case["expected_failure_kind"]:
            raise RedRecordError(
                f"{nodeid}: observed kind {kind!r} is not the expected "
                f"{case['expected_failure_kind']!r} ({entry['message']})"
            )
        if re.search(str(case["expected_failure_pattern"]), entry["message"]) is None:
            raise RedRecordError(
                f"{nodeid}: {entry['message']!r} does not match "
                f"{case['expected_failure_pattern']!r}"
            )
        fixture_bytes = case["fixture_bytes"]
        if not isinstance(fixture_bytes, bytes) or not fixture_bytes:
            raise RedRecordError(f"{nodeid}: fixture bytes must be non-empty bytes")
        invalid_config_raw_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
        command_index = command_index_by_file[relative]
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "requirement_id": str(case["requirement_id"]),
                "test_nodeid": nodeid,
                "invalid_config_raw_sha256": invalid_config_raw_sha256,
                "fixture_identity_sha256": fixture_identity_sha256(
                    phase=PHASE,
                    fixture_id=str(case["case_id"]),
                    requirement_id=str(case["requirement_id"]),
                    test_nodeid=nodeid,
                    pre_fix_source_sha=pre_fix_source_sha,
                    invalid_config_raw_sha256=invalid_config_raw_sha256,
                ),
                "expected_failure_kind": str(case["expected_failure_kind"]),
                "expected_failure_pattern": str(case["expected_failure_pattern"]),
                "command_index": command_index,
                "exit_code": int(commands[command_index]["exit_code"]),
                "observed_outcome": kind,
                "observed_exception_type": entry["type"],
                "observed_message": entry["message"],
                "stdout_sha256": str(commands[command_index]["stdout_sha256"]),
                "stderr_sha256": str(commands[command_index]["stderr_sha256"]),
                "fixture_defect_excluded_by": str(case["fixture_defect_excluded_by"]),
                "red_failure_confirmed": True,
            }
        )

    authorized = sorted(set(_changed_paths()) | {OUTPUT_PATH})
    outside = [path for path in authorized if path not in R1_AUTHORIZED_PATHS]
    if outside:
        raise RedRecordError(f"paths outside the R1 authority changed: {outside}")

    document = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": STATUS,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "design_sha": design_sha,
        "pre_fix_source_sha": pre_fix_source_sha,
        "red_commit_sha": None,
        "red_commit_sha_reason": RED_COMMIT_SHA_REASON,
        "protected_source_clean": True,
        "authorized_red_paths": authorized,
        "environment": _environment(),
        "cases": rows,
        "commands": commands,
        "claims_not_licensed": list(CLAIMS_NOT_LICENSED),
    }
    payload = canonical_json_bytes(document)
    _atomic_no_overwrite(output, payload)
    print(
        f"{output.relative_to(REPOSITORY_ROOT)} "
        f"sha256={hashlib.sha256(payload).hexdigest()} "
        f"cases={len(rows)} commands={len(commands)}"
    )


def _atomic_no_overwrite(target: Path, payload: bytes) -> None:
    handle, temporary = tempfile.mkstemp(dir=str(target.parent))
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        os.close(descriptor)
        os.replace(temporary, target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generator = subparsers.add_parser("generate")
    generator.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / OUTPUT_PATH,
        help="the retained record path; it must not already exist",
    )
    arguments = parser.parse_args(argv)
    try:
        generate(arguments.output)
    except RedRecordError as error:
        print(f"SCI004_M1_RED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
