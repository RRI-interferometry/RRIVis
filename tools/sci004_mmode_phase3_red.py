#!/usr/bin/env python
"""Generate the SCI-004 phase-M3 retained red-failure record.

``docs/development/sci004_mmode_design.md`` Section 12 opens every
implementation phase with "a red-test commit and a retained record naming the
node ID, expected equation/behavior, observed pre-fix failure, and why the
fixture is not defective". Section 14.1 freezes that record's schema; this tool
produces it and nothing else::

    pixi run python tools/sci004_mmode_phase3_red.py generate

It is the phase-M3 sibling of ``tools/sci004_mmode_phase2_red.py`` and keeps
that tool's discipline verbatim, because the discipline is the point rather
than the code.

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

**The tool refuses to fabricate.** It observes redness by running pytest and
reading the JUnit report, never by asserting it. If a group's pytest exits zero
the run aborts with "pytest exited zero, so nothing was red"; a skipped,
xfailed, unexpectedly passed, collection-only, or unrelated outcome aborts; a
green control that did not pass aborts; and a declared expectation that the
observed failure does not match aborts before a byte is written. Section 13.7's
rule that a record must never claim ``expected-red-confirmed`` "against a tree
where nothing is red" is enforced here mechanically, not by convention.

**The phase unlock and the starred ``G3 -> R3`` edge.** The retained M2
acceptance record carries a null self SHA with the reason "self-reference: the
next R or C binds the containing A commit". This slice is that next ``R``, so
the generator authenticates ``A2`` from Git objects -- single-parent non-merge,
parent exactly ``E2``, carrying the accepted ``ACCEPT`` artifact -- before
writing anything.

Section 14.4's edge is now ``G3 ->* R3``. Three accepted corrections have
reopened a phase-3 red slice -- the accepted-capability-characterization-envelope
one, the retained-evidence-surfaces one, and the honest-backend-axis one -- and,
per Section 13.7's reopened-phase rule, each re-cut ``R3`` directly parents its
own correction's landing. The
observation tree is therefore the operative ``D`` and any other ``HEAD`` is
refused, and the starred interval is authenticated exhaustively rather than by a
membership test: ``G3..D`` must be exactly the eight commits the operative record
enumerates -- three reopened red slices and five design landings, oldest-first --
each a single-parent non-merge. Section
13.7's "A commit the header does not name invalidates the edge" is enforced as
an equality on that range. The frozen bindings are read from
``tests/unit/test_sci004_phase3_dependency.py``, which Section 14.0 names as
this phase's single site for them, rather than copied here where they could
silently diverge.

**What the record is, and is not.** It records that a named set of nodes failed
in a named way at a named clean source SHA, that every protected path outside
Section 13.5's ``R3`` list was byte-identical before and after the run, and that
each fixture is excluded from defect by a control that passed in the same
invocation. It licenses no production, acceptance, fingerprint, or performance
claim; ``claims_not_licensed`` says so in the record itself.

Generation is atomic and refuses to overwrite.
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

PHASE = "M3"
SCHEMA_VERSION = "radiosim.sci004.mmode-phase3-red-failures.v1"
STATUS = "expected-red-confirmed"
RED_COMMIT_SHA_REASON = "self-reference: E binds the containing R commit"
POST_SOURCE_RED_COMMIT_SHA_REASON = (
    "self-reference: E binds the containing post-source R commit"
)
FINGERPRINT_POST_SOURCE_RED_COMMIT_SHA_REASON = (
    "self-reference: E binds the containing fingerprint-retry R3 commit"
)

OUTPUT_PATH = "docs/development/sci004_mmode_phase3_red_failures.json"
POST_SOURCE_OUTPUT_PATH = (
    "docs/development/sci004_mmode_phase3_post_source_red_failures.json"
)
POST_SOURCE_SCHEMA_VERSION = "radiosim.sci004.mmode-phase3-post-source-red-failures.v1"
POST_SOURCE_STATUS = "post-source-expected-red-confirmed"
POST_SOURCE_PRE_FIX_SHA = "a61526d686ab768f05ecffa80cfd6223d4ee4c62"
FINGERPRINT_POST_SOURCE_OUTPUT_PATH = (
    "docs/development/sci004_mmode_phase3_fingerprint_post_source_red_failures.json"
)
FINGERPRINT_POST_SOURCE_SCHEMA_VERSION = (
    "radiosim.sci004.mmode-phase3-fingerprint-post-source-red-failures.v1"
)
FINGERPRINT_POST_SOURCE_STATUS = "post-source-expected-red-confirmed"
FINGERPRINT_POST_SOURCE_PRE_FIX_SHA = "b07925ab14b56b3ca0fa863f806290748a31df6b"
CORRECTION24_POST_SOURCE_RED_RECORD_SHA256 = (
    "724f75c246ebfcf5956fc40fb2f5e349d91ccca3e6a188b3785a65f4ae4c1e10"
)
FINGERPRINT_POST_SOURCE_ORACLE_PATH = "tests/characterization/test_sci004_mmode.py"
HISTORICAL_RED_SLICE_SHA = "7070cc3ddb1c2557d02e4a3f2a89b907575bed0b"
HISTORICAL_RED_RECORD_SHA256 = (
    "486705a8d5e51c08f972c91aeae60f0a0bfeef5480b622515282295a6a3cde05"
)
POST_SOURCE_ORACLE_PATH = "tests/unit/test_io/test_hdf5_result.py"
DEPENDENCY_VALIDATOR_PATH = "tests/unit/test_sci004_phase3_dependency.py"
M2_ACCEPTANCE_PATH = "docs/development/sci004_mmode_phase2_acceptance.json"

#: Section 13.5's complete ``R3`` writable list, sorted. Every path outside it
#: is protected and must be byte-identical before and after this run.
R3_AUTHORIZED_PATHS: tuple[str, ...] = tuple(
    sorted(
        (
            "docs/development/sci004_mmode_phase3_red_failures.json",
            "docs/development/sci004_mmode_phase3_sci005_dependency.json",
            "tests/characterization/test_sci004_mmode.py",
            "tests/unit/test_io/test_hdf5_result.py",
            "tests/unit/test_io/test_measurement_set.py",
            "tests/unit/test_io/test_result_summary.py",
            "tests/unit/test_io/test_standard_visibility.py",
            "tests/unit/test_io/test_uvfits.py",
            "tests/performance/test_sci004_mmode.py",
            "tests/unit/test_sci004_phase3_dependency.py",
            "tests/unit/test_sci004_phase3_red_failures.py",
            "tests/unit/test_tier8_release_acceptance.py",
            "tools/sci004_mmode_phase3_red.py",
        )
    )
)

POST_SOURCE_R3_AUTHORIZED_PATHS: tuple[str, ...] = tuple(
    sorted(
        (
            POST_SOURCE_OUTPUT_PATH,
            POST_SOURCE_ORACLE_PATH,
            "tests/unit/test_sci004_phase3_dependency.py",
            "tests/unit/test_sci004_phase3_red_failures.py",
            "tools/sci004_mmode_phase3_red.py",
        )
    )
)
POST_SOURCE_NON_ARTIFACT_PATHS: tuple[str, ...] = tuple(
    path for path in POST_SOURCE_R3_AUTHORIZED_PATHS if path != POST_SOURCE_OUTPUT_PATH
)

FINGERPRINT_R3_AUTHORIZED_PATHS: tuple[str, ...] = tuple(
    sorted(
        (
            FINGERPRINT_POST_SOURCE_OUTPUT_PATH,
            FINGERPRINT_POST_SOURCE_ORACLE_PATH,
            "tests/unit/test_sci004_phase3_dependency.py",
            "tests/unit/test_sci004_phase3_red_failures.py",
            "tools/sci004_mmode_phase3_red.py",
        )
    )
)
FINGERPRINT_NON_ARTIFACT_PATHS: tuple[str, ...] = tuple(
    path
    for path in FINGERPRINT_R3_AUTHORIZED_PATHS
    if path != FINGERPRINT_POST_SOURCE_OUTPUT_PATH
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

FINGERPRINT_FAMILY_RECORD_V2_KEYS: tuple[str, ...] = (
    "family_id",
    "raw_cube_sha256",
    "scientific_sha256",
    "solver_snapshot",
    "characterization_time_manifest",
    "era_utc_grid_sha256",
    "harmonic_index_table_sha256",
    "characterization_input_manifest",
    "input_identity_sha256",
)

FINGERPRINT_LAYOUT_RAW_SHA256 = (
    "a2ce7bace30e2fe962eb6454db1f6c7e2d63a9a28ad559323e824a36fcd2a4e0"
)

#: The red modules that declare ``SCI004_PHASE3_RED_CASES``, in record order.
RED_MODULES: tuple[str, ...] = (
    "tests.unit.test_io.test_standard_visibility",
    "tests.unit.test_io.test_hdf5_result",
    "tests.unit.test_io.test_result_summary",
    "tests.unit.test_io.test_uvfits",
    "tests.unit.test_io.test_measurement_set",
    "tests.characterization.test_sci004_mmode",
    "tests.unit.test_tier8_release_acceptance",
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
    "radiosim.io.config_resolution.ConfigSemanticError": "schema",
    "radiosim.io.config_resolution.UnsupportedConfigError": "schema",
}

#: Section 14.1: sorted, unique, non-empty, and covering production, acceptance,
#: fingerprint, and performance.
CLAIMS_NOT_LICENSED: tuple[str, ...] = (
    "acceptance: this record expresses no phase-M3 acceptance verdict and "
    "unlocks no successor commit",
    "fingerprint: no Section 11 m-mode family is pinned, harvested, or "
    "adjudicated by this record, and no dispatch-class observation set exists "
    "at this source SHA",
    "performance: the non-gating Section 11 benchmark record measures nothing "
    "here, and no timing, speedup, memory, or accelerator advantage is claimed",
    "production: no m-mode output capability exists at this source SHA beyond "
    "the accepted M1 summary and HDF5 snapshot surface, so every recorded "
    "failure is the absence of one",
)

#: Correction #24 observes the completed superseded S3, so the historical
#: phase-red production disclaimer above is false for this supplement. These
#: four claims retain the required category shape while describing only what
#: the post-source hostile-reader replay actually establishes.
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

#: Correction #25's fresh delta licenses only the observed fingerprint defects.
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

FINGERPRINT_CASE_SPECS: tuple[dict[str, str], ...] = (
    {
        "case_id": "m3.fingerprint.preimage-retained",
        "requirement_id": "SCI-004-14.2-M3-FINGERPRINT-PREIMAGE",
        "test_nodeid": FINGERPRINT_NODEIDS[0],
        "expected_failure_kind": "assertion",
        "expected_failure_pattern": (
            "characterization input manifest is absent from the family record"
        ),
        "fixture_defect_excluded_by": (
            "the same-run phase input manifest and exact canonical v2 preimage "
            "are independently reconstructed before the production assertion"
        ),
    },
    {
        "case_id": "m3.fingerprint.path-independent",
        "requirement_id": "SCI-004-11-PATH-INDEPENDENT-CHARACTERIZATION",
        "test_nodeid": FINGERPRINT_NODEIDS[1],
        "expected_failure_kind": "assertion",
        "expected_failure_pattern": (
            "characterization input identity changed under filesystem relocation"
        ),
        "fixture_defect_excluded_by": (
            "independent roots preserve locally derived science and raw-cube "
            "identities while differing only in retained filesystem location"
        ),
    },
)

FINGERPRINT_CONTROL_SPECS: tuple[dict[str, str], ...] = (
    {
        "control_id": "m3.fingerprint.family-record-schema",
        "requirement_id": "SCI-004-11-FAMILY-RECORD-SCHEMA",
        "test_nodeid": FINGERPRINT_NODEIDS[2],
        "purpose": (
            "exact domain-discriminated family-record schema and all pre-existing "
            "family joins remain valid"
        ),
    },
    {
        "control_id": "m3.fingerprint.relocation-science-control",
        "requirement_id": "SCI-004-11-PATH-INDEPENDENT-CHARACTERIZATION",
        "test_nodeid": FINGERPRINT_NODEIDS[3],
        "purpose": (
            "relocation fixture preserves independently derived scientific and "
            "raw-cube identities"
        ),
    },
    {
        "control_id": "m3.fingerprint.semantic-separation-control",
        "requirement_id": "SCI-004-11-SEMANTIC-INPUT-SEPARATION",
        "test_nodeid": FINGERPRINT_NODEIDS[4],
        "purpose": (
            "semantic antenna-layout mutation changes characterization input identity"
        ),
    },
)

#: Section 14.2's ``numeric_packages`` set, exactly.
NUMERIC_PACKAGES: tuple[str, ...] = ("dask", "healpy", "jax", "numpy", "scipy")

IERS_PACKAGE = "astropy_iers_data"
IERS_RESOURCE = "data/finals2000A.all"


class RedRecordError(RuntimeError):
    """The phase-M3 red record could not be generated as specified."""


# --- Section 14 canonical JSON ------------------------------------------------


def _es_number(value: float | int) -> str:
    """Serialize one finite number with RFC 8785 / ECMAScript shortest round trip."""
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
    """Section 14.0's red fixture identity, over exactly its six named fields."""
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


def _diff_tree_paths(commit: str) -> tuple[str, ...]:
    """Return one commit's sorted parent-relative diff paths, from Git objects."""
    listing = _git(
        "diff-tree", "--no-commit-id", "--name-only", "-r", "-z", commit
    ).split("\0")
    return tuple(sorted(entry for entry in listing if entry))


def _frozen_binding(name: str) -> str:
    """Read one frozen constant from the single file Section 14.0 authorises.

    Section 14.0 permits exactly one assignment of the phase-M3 bindings, in
    ``tests/unit/test_sci004_phase3_dependency.py``, and a generator carrying
    its own copy would be a second, silently divergent one.
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


def _protected_digest(
    authorized_paths: Sequence[str] = R3_AUTHORIZED_PATHS,
) -> str:
    """One digest over every tracked path outside the ``R3`` authorized list."""
    authorized = set(authorized_paths)
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


def _tree_blob(commit: str, relative: str) -> bytes:
    listing = _git("ls-tree", "-z", commit, "--", relative)
    entries = [entry for entry in listing.split("\0") if entry]
    if len(entries) != 1:
        raise RedRecordError(f"{relative} is not one blob at {commit}")
    metadata, _tab, name = entries[0].partition("\t")
    _mode, object_type, object_id = metadata.split()
    if name != relative or object_type != "blob":
        raise RedRecordError(f"{relative} is not a regular blob at {commit}")
    completed = subprocess.run(
        ["git", "cat-file", "blob", object_id],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RedRecordError(f"cannot read {relative} at {commit}")
    return completed.stdout


def _post_source_oracle_diff(commit: str | None = None) -> bytes:
    argv = [
        "git",
        "diff",
        "--no-ext-diff",
        "--binary",
        "--full-index",
        POST_SOURCE_PRE_FIX_SHA,
    ]
    if commit is not None:
        argv.append(commit)
    argv.extend(("--", POST_SOURCE_ORACLE_PATH))
    completed = subprocess.run(
        argv,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RedRecordError(
            f"the post-source oracle diff failed: "
            f"{completed.stderr.decode('utf-8', 'replace').strip()}"
        )
    return completed.stdout


def _fingerprint_oracle_diff(commit: str | None = None) -> bytes:
    argv = [
        "git",
        "-c",
        "color.ui=false",
        "--no-pager",
        "diff",
        "--no-ext-diff",
        "--binary",
        "--full-index",
        _frozen_binding("APPROVED_SCI004_D_SHA"),
    ]
    if commit is not None:
        argv.append(commit)
    argv.extend(("--", FINGERPRINT_POST_SOURCE_ORACLE_PATH))
    completed = subprocess.run(
        argv,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RedRecordError(
            "the fingerprint oracle diff failed: "
            + completed.stderr.decode("utf-8", "replace").strip()
        )
    return completed.stdout


def _regular_retained_blob(path: str, commit: str, expected_sha256: str) -> bytes:
    target = REPOSITORY_ROOT / path
    if target.is_symlink() or not target.is_file():
        raise RedRecordError(f"retained artifact {path} is not a regular file")
    raw = target.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RedRecordError(f"retained artifact {path} has the wrong raw digest")
    if raw != _tree_blob(commit, path):
        raise RedRecordError(f"retained artifact {path} differs from {commit}'s blob")
    return raw


def _authenticate_fingerprint_retry_chain(head: str) -> None:
    """Authenticate D25 and the rejected E3/A3 attempt from Git objects."""
    design = _frozen_binding("APPROVED_SCI004_D_SHA")
    d24 = _frozen_binding("D24_SHA")
    old_r3 = _frozen_binding("SUPERSEDED_FINGERPRINT_R3_SHA")
    old_s3 = _frozen_binding("SUPERSEDED_FINGERPRINT_S3_SHA")
    rejected_e3 = _frozen_binding("REJECTED_E3_SHA")
    rejected_a3 = _frozen_binding("REJECTED_A3_SHA")
    if head != design:
        raise RedRecordError(
            f"generate-fingerprint-post-source requires exact D25 {design}, not {head}"
        )
    chain = (d24, old_r3, old_s3, rejected_e3, rejected_a3, design)
    previous = _git("rev-parse", f"{d24}^").strip()
    for commit in chain:
        if _git("rev-parse", "--verify", f"{commit}^{{commit}}").strip() != commit:
            raise RedRecordError(f"{commit} does not peel to itself")
        parents = tuple(_git("rev-list", "--parents", "-n", "1", commit).split()[1:])
        if parents != (previous,):
            raise RedRecordError(
                f"fingerprint retry chain commit {commit} parents {parents}, "
                f"not {(previous,)}"
            )
        previous = commit

    exact_paths = {
        d24: ("PostTier8RemediationPlan.md", "docs/development/sci004_mmode_design.md"),
        old_r3: tuple(
            sorted(
                (
                    POST_SOURCE_OUTPUT_PATH,
                    POST_SOURCE_ORACLE_PATH,
                    DEPENDENCY_VALIDATOR_PATH,
                    "tests/unit/test_sci004_phase3_red_failures.py",
                    "tools/sci004_mmode_phase3_red.py",
                )
            )
        ),
        old_s3: tuple(
            sorted(
                (
                    "src/radiosim/io/hdf5.py",
                    "tests/unit/test_sci004_phase3_evidence.py",
                    "tools/sci004_mmode_phase3_evidence.py",
                )
            )
        ),
        rejected_e3: tuple(
            sorted(
                (
                    "docs/development/sci004_mmode_phase3_evidence.json",
                    "docs/development/sci004_mmode_phase3_evidence.md",
                    "output/benchmarks/reference/sci004/"
                    "20260825T122048Z-macbook-pro-2.json",
                    "tests/unit/test_sci004_phase3_evidence.py",
                )
            )
        ),
        rejected_a3: tuple(
            sorted(
                (
                    "docs/development/sci004_mmode_phase3_acceptance.json",
                    "tests/unit/test_sci004_phase3_acceptance.py",
                )
            )
        ),
        design: (
            "PostTier8RemediationPlan.md",
            "docs/development/sci004_mmode_design.md",
        ),
    }
    for commit, paths in exact_paths.items():
        if _diff_tree_paths(commit) != paths:
            raise RedRecordError(
                f"{commit} touches {_diff_tree_paths(commit)}, not exact {paths}"
            )

    design_path = "docs/development/sci004_mmode_design.md"
    ledger_path = "PostTier8RemediationPlan.md"
    final_blobs = (
        (
            design_path,
            371403,
            "eb45da5adfe412cc3447303f8ac77448988317f286b27df95d783f047481791f",
        ),
        (
            ledger_path,
            52079,
            "33d610102339aecf6046ebec243a958e66f7be10148db3abf9fe96fe53f91a0f",
        ),
    )
    for path, size, digest in final_blobs:
        raw = _tree_blob(design, path)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise RedRecordError(f"D25 final blob identity changed for {path}")
    diff_expectations = (
        (
            (design_path,),
            "5ab8d5cf856f78640585be8f9257a50e72ebf9a86cac7766887967722f70d7a8",
        ),
        (
            (ledger_path,),
            "84a215a0bb556432f5db5b09385d553f7c8b09116f13104d6e54ed7b94d47a09",
        ),
        (
            (design_path, ledger_path),
            "8c21e2f0193475422925ecaa4d0e6fab296d46517b115118846bb600a90911f0",
        ),
    )
    for paths, expected_digest in diff_expectations:
        completed = subprocess.run(
            [
                "git",
                "diff",
                "--no-ext-diff",
                "--binary",
                "--full-index",
                rejected_a3,
                design,
                "--",
                *paths,
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0 or hashlib.sha256(
            completed.stdout
        ).hexdigest() != (expected_digest):
            raise RedRecordError(f"D25 final diff identity changed for {paths}")

    acceptance_path = "docs/development/sci004_mmode_phase3_acceptance.json"
    evidence_path = "docs/development/sci004_mmode_phase3_evidence.json"
    reproduction_path = "docs/development/sci004_mmode_phase3_evidence.md"
    performance_path = (
        "output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json"
    )
    acceptance_raw = _regular_retained_blob(
        acceptance_path,
        rejected_a3,
        "283fb5264f5ecd86aed1300ae504b85946cf1f4d36b1c4c09bc92bb4f269421d",
    )
    evidence_raw = _regular_retained_blob(
        evidence_path,
        rejected_e3,
        "600b51ac4d70778ee2d3bdf7b8842b83ba77dc34d541784ad1ad7d8e5be5f8ae",
    )
    _regular_retained_blob(
        reproduction_path,
        rejected_e3,
        "039539a865b5d92e86379f44a324271232e8a947301e380ec7b1b1848e907b4e",
    )
    _regular_retained_blob(
        performance_path,
        rejected_e3,
        "07e59d3176866a78c17244849d6493365e9d410547e884cf56b254e60babe193",
    )
    evidence = json.loads(evidence_raw)
    if (
        evidence.get("status") != "candidate"
        or evidence.get("source_sha") != old_s3
        or evidence.get("red_commit_sha") != old_r3
    ):
        raise RedRecordError("the rejected E3 candidate bindings changed")
    acceptance = json.loads(acceptance_raw)
    blockers = acceptance.get("blockers")
    if (
        acceptance.get("verdict") != "REJECT"
        or acceptance.get("reviewer_identity")
        != "sci004-m3-independent-acceptance-reviewer"
        or acceptance.get("reviewer_independent") is not True
        or acceptance.get("evidence_commit_sha") != rejected_e3
        or acceptance.get("evidence_artifact_sha256")
        != "600b51ac4d70778ee2d3bdf7b8842b83ba77dc34d541784ad1ad7d8e5be5f8ae"
        or not isinstance(blockers, list)
        or [row.get("blocker_id") for row in blockers]
        != ["m3.fingerprint-input-preimage-not-retained"]
    ):
        raise RedRecordError("the canonical rejected A3 bindings changed")
    external_review_path = Path(
        "/Users/kartikmandar/MacProjects/"
        "sci004-a3-independent-review-reject-20260826.json"
    )
    if external_review_path.is_symlink() or not external_review_path.is_file():
        raise RedRecordError("the external independent review contribution is absent")
    external_review_raw = external_review_path.read_bytes()
    if hashlib.sha256(external_review_raw).hexdigest() != (
        "43c12807aa9f316af53e6058ebec7f18dd0b6ea66d308cb1c488d77185907d82"
    ):
        raise RedRecordError("the external independent review digest changed")
    external_review = json.loads(external_review_raw)
    if (
        external_review.get("verdict") != "REJECT"
        or external_review.get("reviewer_identity")
        != "sci004-m3-independent-acceptance-reviewer"
        or external_review.get("reviewer_independent") is not True
    ):
        raise RedRecordError("the external independent review identity changed")


def _authenticate_phase_unlock(observation_sha: str) -> None:
    """Prove the Section 14.4 unlock and the ``G3`` edge before writing anything.

    The retained M2 acceptance artifact carries a null self SHA whose reason is
    "self-reference: the next R or C binds the containing A commit"; this slice
    is that next ``R``. Nothing here trusts a 40-hex string: the commit is
    peeled, its parent is required to be exactly ``E2``, the accepted artifact
    is read from the ``A2`` tree rather than from the checkout, and its verdict
    is required to be ``ACCEPT``.

    Section 14.4's edge is ``G3 ->* R3``: the accepted correction that reopened
    the first phase-3 red slice makes the re-cut ``R3`` directly parent that
    correction's landing, so the observation tree is the operative ``D``.  The
    starred interval is authenticated exhaustively, and ``G3`` itself is still
    required to carry both named dependencies as inclusive ancestors.
    """
    acceptance = _frozen_binding("APPROVED_SCI004_A2_SHA")
    evidence = _frozen_binding("APPROVED_SCI004_E2_SHA")
    design = _frozen_binding("APPROVED_SCI004_D_SHA")
    gate = _frozen_binding("APPROVED_SCI004_G3_SHA")
    upstream = _frozen_binding("APPROVED_SCI005_STAGE2_A_SHA")
    superseded_red = _frozen_binding("SUPERSEDED_RED_SLICE_SHA")
    superseded_recut = _frozen_binding("SUPERSEDED_RECUT_RED_SLICE_SHA")
    superseded_second_recut = _frozen_binding("SUPERSEDED_SECOND_RECUT_RED_SLICE_SHA")
    un_ignoring = _frozen_binding("D16_SHA")
    envelope = _frozen_binding("D17_SHA")
    performance_product = _frozen_binding("D18_SHA")
    retained_evidence = _frozen_binding("D19_SHA")

    peeled = _git("rev-parse", "--verify", f"{acceptance}^{{commit}}").strip()
    if peeled != acceptance:
        raise RedRecordError(f"{acceptance} does not peel to itself")
    parents = _git("rev-list", "--parents", "-n", "1", acceptance).split()[1:]
    if tuple(parents) != (evidence,):
        raise RedRecordError(
            f"A2 must directly parent E2 {evidence}; its parents are {parents}"
        )
    if observation_sha != design:
        raise RedRecordError(
            f"HEAD is {observation_sha}; Section 14.4's starred G3 -> R3 edge makes "
            f"the re-cut R3 directly parent the operative D {design}, so the record "
            "is generated at exactly that tree"
        )
    for ancestor in (acceptance, upstream):
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, gate],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RedRecordError(f"{ancestor} is not an ancestor of G3 {gate}")
    if _git("rev-list", "--first-parent", f"{acceptance}..{gate}").split():
        raise RedRecordError("the A2..G3 interval must be empty; G3 is the unlock")
    expected_interval = (
        superseded_red,
        un_ignoring,
        envelope,
        superseded_recut,
        performance_product,
        retained_evidence,
        superseded_second_recut,
        design,
    )
    for flags in (("--first-parent", "--reverse"), ("--reverse",)):
        observed = tuple(_git("rev-list", *flags, f"{gate}..{design}").split())
        if observed != expected_interval:
            raise RedRecordError(
                f"the starred G3..D interval is {observed}, not the enumerated "
                f"{expected_interval}; an unenumerated commit invalidates the edge"
            )
    previous = gate
    for sha in expected_interval:
        if _git("rev-parse", "--verify", f"{sha}^{{commit}}").strip() != sha:
            raise RedRecordError(f"{sha} does not peel to itself")
        if tuple(_git("rev-list", "--parents", "-n", "1", sha).split()[1:]) != (
            previous,
        ):
            raise RedRecordError(f"{sha} is not a single-parent child of {previous}")
        previous = sha
    # Containment, not equality: this re-cut's own grant added a path to the
    # Section 13.5 R3 list that neither superseded slice could have touched, so
    # an equality would fail for that reason alone rather than for a defect.
    for sha in (superseded_red, superseded_recut, superseded_second_recut):
        touched = _diff_tree_paths(sha)
        if not set(touched) <= set(R3_AUTHORIZED_PATHS):
            raise RedRecordError(
                f"the superseded red slice {sha} touches {touched}, which is not "
                "within the Section 13.5 R3 list"
            )
        if DEPENDENCY_VALIDATOR_PATH not in touched:
            raise RedRecordError(f"{sha} is not a phase-3 red slice")
    for sha in (
        un_ignoring,
        envelope,
        performance_product,
        retained_evidence,
        design,
    ):
        if _diff_tree_paths(sha) != (
            "PostTier8RemediationPlan.md",
            "docs/development/sci004_mmode_design.md",
        ):
            raise RedRecordError(f"{sha} is not a design-only correction landing")

    listing = _git("ls-tree", acceptance, "--", M2_ACCEPTANCE_PATH).split()
    if not listing:
        raise RedRecordError(f"{M2_ACCEPTANCE_PATH} is absent from A2")
    blob = subprocess.run(
        ["git", "cat-file", "blob", listing[2]],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    artifact = json.loads(blob.decode("utf-8"))
    if artifact.get("phase") != "M2" or artifact.get("verdict") != "ACCEPT":
        raise RedRecordError("the retained M2 acceptance artifact is not an ACCEPT")
    # Section 13.7: an accepted artifact is immutable, so it still names the
    # ``D`` operative at its own phase -- two corrections back.
    superseded_design = _frozen_binding("D15_SHA")
    if artifact.get("design_sha") != superseded_design:
        raise RedRecordError(
            f"the M2 acceptance artifact names {artifact.get('design_sha')!r}, not the "
            f"superseded operative D {superseded_design}"
        )
    if superseded_design == design:
        raise RedRecordError("the superseded and operative D bindings must differ")
    if artifact.get("acceptance_commit_sha") is not None:
        raise RedRecordError("the M2 acceptance artifact must carry a null self SHA")


# --- the case inventory -------------------------------------------------------


def _load_cases() -> tuple[list[dict[str, Any]], dict[str, tuple[str, ...]]]:
    """Load the declared phase-M3 case table and green controls."""
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    cases: list[dict[str, Any]] = []
    controls: dict[str, tuple[str, ...]] = {}
    for module_name in RED_MODULES:
        module = importlib.import_module(module_name)
        declared = getattr(module, "SCI004_PHASE3_RED_CASES", None)
        if not declared:
            raise RedRecordError(f"{module_name} declares no SCI004_PHASE3_RED_CASES")
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
            if not str(case["case_id"]).startswith("m3."):
                raise RedRecordError(
                    f"{case['case_id']}: a phase-M3 identifier is namespaced 'm3.'"
                )
            cases.append(dict(case))
        controls[module_name] = tuple(
            getattr(module, "SCI004_PHASE3_RED_GREEN_CONTROLS", ())
        )
    identifiers = [case["case_id"] for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise RedRecordError("case_id values must be unique")
    nodes = [case["test_nodeid"] for case in cases]
    if len(set(nodes)) != len(nodes):
        raise RedRecordError("every phase red node must appear exactly once")
    return cases, controls


def _load_post_source_cases() -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Load correction #24's separate six-case HDF5 red-delta table."""
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    module = importlib.import_module("tests.unit.test_io.test_hdf5_result")
    declared = getattr(module, "SCI004_PHASE3_POST_SOURCE_RED_CASES", None)
    controls = getattr(
        module,
        "SCI004_PHASE3_POST_SOURCE_RED_GREEN_CONTROLS",
        None,
    )
    if not declared or len(declared) != 6:
        raise RedRecordError("the post-source table must declare exactly six cases")
    if not controls or len(controls) != 5:
        raise RedRecordError(
            "the post-source replay must declare exactly five controls"
        )
    cases = [dict(case) for case in declared]
    required = {
        "case_id",
        "requirement_id",
        "test_nodeid",
        "expected_failure_kind",
        "expected_failure_pattern",
        "fixture_defect_excluded_by",
        "fixture_bytes",
    }
    for case in cases:
        missing = required - set(case)
        if missing:
            raise RedRecordError(
                f"{case.get('case_id')}: post-source case lacks {sorted(missing)}"
            )
        if case["expected_failure_kind"] != "assertion":
            raise RedRecordError(
                f"{case['case_id']}: the superseded-source outcome must be assertion"
            )
        if "HDF5 result failed canonical model or fingerprint validation" not in str(
            case["expected_failure_pattern"]
        ):
            raise RedRecordError(
                f"{case['case_id']}: the red pattern does not name the late failure"
            )
    nodes = [str(case["test_nodeid"]) for case in cases]
    identifiers = [str(case["case_id"]) for case in cases]
    if len(set(nodes)) != 6 or len(set(identifiers)) != 6:
        raise RedRecordError("post-source case IDs and node IDs must be unique")
    return cases, tuple(str(nodeid) for nodeid in controls)


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
        "-p",
        "no:randomly",
        "-p",
        "no:xdist",
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


def _junit_entry(testcase: ElementTree.Element) -> dict[str, str]:
    """Classify one JUnit testcase without losing its full failure message."""
    failure = testcase.find("failure")
    error = testcase.find("error")
    skipped = testcase.find("skipped")
    if skipped is not None:
        return {"outcome": "skipped", "type": "", "message": ""}
    node = failure if failure is not None else error
    if node is None:
        return {"outcome": "passed", "type": "", "message": ""}
    raw = (node.get("message") or "").strip()
    first_line = raw.splitlines()[0] if raw else ""
    return {
        "outcome": "collected" if error is not None else "failed",
        "type": _exception_type(node.get("type"), first_line),
        "message": raw,
    }


def _parse_junit_inventory(
    junit_path: Path,
) -> list[tuple[str, dict[str, str]]]:
    """Return the ordered JUnit inventory, retaining duplicates for rejection."""
    tree = ElementTree.parse(junit_path)
    return [
        (testcase.get("name") or "", _junit_entry(testcase))
        for testcase in tree.iter("testcase")
    ]


def _parse_junit(junit_path: Path) -> dict[str, dict[str, str]]:
    """Return ``{test name: {outcome, exception_type, message}}`` from junit XML."""
    observed: dict[str, dict[str, str]] = {}
    for name, entry in _parse_junit_inventory(junit_path):
        observed[name] = entry
    return observed


_QUALIFIED_NAME = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*): "
)


#: Bare crash-line names whose defining module is not ``builtins``.
#: ``pytest.raises`` reports a missing exception with its own outcome class,
#: whose module is ``_pytest.outcomes``; recording it as ``builtins.Failed``
#: would name a class that does not exist.
_BARE_NAME_MODULES: Mapping[str, str] = {"Failed": "_pytest.outcomes"}


def _qualify(name: str) -> str:
    if "." in name:
        return name
    return f"{_BARE_NAME_MODULES.get(name, 'builtins')}.{name}"


def _exception_type(declared: str | None, first_line: str) -> str:
    """Derive the fully qualified exception class Section 14.1 requires."""
    if declared:
        return _qualify(declared)
    match = _QUALIFIED_NAME.match(first_line)
    if match is not None:
        return _qualify(match.group("name"))
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
    if (REPOSITORY_ROOT / OUTPUT_PATH).exists():
        raise RedRecordError(
            "the historical phase-M3 record already exists and is retained; "
            "historical generation cannot regenerate it at another path"
        )

    design_sha = _frozen_binding("APPROVED_SCI004_D_SHA")
    # Section 14.1: ``pre_fix_source_sha`` names the tree the observations were
    # genuinely made from. Section 14.4's ``R3^ == G3`` makes that tree the gate
    # tip, and the unlock check below refuses any other ``HEAD``.
    pre_fix_source_sha = _git("rev-parse", "HEAD").strip()
    _authenticate_phase_unlock(pre_fix_source_sha)

    cases, controls = _load_cases()
    groups = _group_by_file(cases, controls)
    protected_before = _protected_digest()

    commands: list[dict[str, Any]] = []
    observed: dict[str, dict[str, str]] = {}
    command_index_by_file: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="sci004-m3-red-") as scratch:
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
            "a protected path outside the R3 list changed during generation"
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
    outside = [path for path in authorized if path not in R3_AUTHORIZED_PATHS]
    if outside:
        raise RedRecordError(f"paths outside the R3 authority changed: {outside}")

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


def generate_post_source() -> None:
    """Generate correction #24's separate six-case post-source red delta."""
    output = REPOSITORY_ROOT / POST_SOURCE_OUTPUT_PATH
    if output.exists():
        raise RedRecordError(f"{output} already exists; generation never overwrites")

    design_sha = _frozen_binding("APPROVED_SCI004_D_SHA")
    head = _git("rev-parse", "HEAD").strip()
    if head != design_sha:
        raise RedRecordError(
            "generate-post-source requires HEAD at operative D "
            f"{design_sha}, not {head}"
        )
    parents = tuple(_git("rev-list", "--parents", "-n", "1", head).split()[1:])
    if parents != (POST_SOURCE_PRE_FIX_SHA,):
        raise RedRecordError(
            f"correction #24 must directly parent {POST_SOURCE_PRE_FIX_SHA}; "
            f"observed {parents}"
        )
    changed = _changed_paths()
    if changed != POST_SOURCE_NON_ARTIFACT_PATHS:
        raise RedRecordError(
            "generate-post-source requires exactly the four non-artifact R3 paths "
            f"dirty; observed {changed}"
        )
    production_changes = _git(
        "diff",
        "--name-only",
        POST_SOURCE_PRE_FIX_SHA,
        design_sha,
        "--",
        "src/radiosim",
    ).split()
    if production_changes:
        raise RedRecordError(
            "the superseded source and correction D differ in production: "
            f"{production_changes}"
        )

    historical_path = REPOSITORY_ROOT / OUTPUT_PATH
    historical = historical_path.read_bytes()
    if hashlib.sha256(historical).hexdigest() != HISTORICAL_RED_RECORD_SHA256:
        raise RedRecordError("the historical M3 red record digest changed")
    if historical != _tree_blob(HISTORICAL_RED_SLICE_SHA, OUTPUT_PATH):
        raise RedRecordError(
            "the historical M3 red record is not byte-identical to its 7070cc3 blob"
        )

    cases, controls = _load_post_source_cases()
    protected_before = _protected_digest(POST_SOURCE_R3_AUTHORIZED_PATHS)
    with tempfile.TemporaryDirectory(prefix="sci004-m3-post-source-red-") as scratch:
        junit_path = Path(scratch) / "junit.xml"
        red_nodes = tuple(str(case["test_nodeid"]) for case in cases)
        command, stdout, _stderr = _run_pytest((*red_nodes, *controls), junit_path)
        if command["exit_code"] == 0:
            raise RedRecordError(
                "pytest exited zero, so the post-source delta was not red"
            )
        if not junit_path.is_file():
            raise RedRecordError(
                "post-source pytest produced no junit report\n"
                + stdout.decode("utf-8", "replace")[-4000:]
            )
        observed = _parse_junit(junit_path)

    for nodeid in controls:
        entry = observed.get(nodeid.split("::", 1)[1])
        if entry is None or entry["outcome"] != "passed":
            raise RedRecordError(f"green control {nodeid} did not pass: {entry}")

    rows: list[dict[str, Any]] = []
    for case in cases:
        nodeid = str(case["test_nodeid"])
        entry = observed.get(
            nodeid.split("::", 1)[1],
            {"outcome": "absent", "type": "", "message": ""},
        )
        if entry["outcome"] != "failed":
            raise RedRecordError(
                f"{nodeid}: observed {entry['outcome']!r}; expected one red failure"
            )
        kind = _classify(entry["type"])
        if kind != "assertion":
            raise RedRecordError(f"{nodeid}: observed {kind!r}, not assertion")
        pattern = str(case["expected_failure_pattern"])
        if re.search(pattern, entry["message"]) is None:
            raise RedRecordError(
                f"{nodeid}: {entry['message']!r} does not match {pattern!r}"
            )
        fixture_bytes = case["fixture_bytes"]
        if not isinstance(fixture_bytes, bytes) or not fixture_bytes:
            raise RedRecordError(f"{nodeid}: fixture bytes must be non-empty bytes")
        invalid_digest = hashlib.sha256(fixture_bytes).hexdigest()
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "requirement_id": str(case["requirement_id"]),
                "test_nodeid": nodeid,
                "invalid_config_raw_sha256": invalid_digest,
                "fixture_identity_sha256": fixture_identity_sha256(
                    phase=PHASE,
                    fixture_id=str(case["case_id"]),
                    requirement_id=str(case["requirement_id"]),
                    test_nodeid=nodeid,
                    pre_fix_source_sha=POST_SOURCE_PRE_FIX_SHA,
                    invalid_config_raw_sha256=invalid_digest,
                ),
                "expected_failure_kind": "assertion",
                "expected_failure_pattern": pattern,
                "command_index": 0,
                "exit_code": int(command["exit_code"]),
                "observed_outcome": kind,
                "observed_exception_type": entry["type"],
                "observed_message": entry["message"],
                "stdout_sha256": str(command["stdout_sha256"]),
                "stderr_sha256": str(command["stderr_sha256"]),
                "fixture_defect_excluded_by": str(case["fixture_defect_excluded_by"]),
                "red_failure_confirmed": True,
            }
        )

    protected_after = _protected_digest(POST_SOURCE_R3_AUTHORIZED_PATHS)
    if protected_before != protected_after:
        raise RedRecordError("a protected path changed during post-source generation")

    oracle_diff = _post_source_oracle_diff()
    if not oracle_diff:
        raise RedRecordError("the post-source HDF5 oracle diff is empty")
    document = {
        "schema_version": POST_SOURCE_SCHEMA_VERSION,
        "phase": PHASE,
        "status": POST_SOURCE_STATUS,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "design_sha": design_sha,
        "pre_fix_source_sha": POST_SOURCE_PRE_FIX_SHA,
        "red_commit_sha": None,
        "red_commit_sha_reason": POST_SOURCE_RED_COMMIT_SHA_REASON,
        "historical_red_record_sha256": HISTORICAL_RED_RECORD_SHA256,
        "oracle_patch_paths": [POST_SOURCE_ORACLE_PATH],
        "oracle_patch_sha256": hashlib.sha256(oracle_diff).hexdigest(),
        "protected_source_clean": True,
        "authorized_red_paths": list(POST_SOURCE_R3_AUTHORIZED_PATHS),
        "environment": _environment(),
        "cases": rows,
        "commands": [command],
        "claims_not_licensed": list(POST_SOURCE_CLAIMS_NOT_LICENSED),
    }
    payload = canonical_json_bytes(document)
    _atomic_no_overwrite(output, payload)
    print(
        f"{POST_SOURCE_OUTPUT_PATH} sha256={hashlib.sha256(payload).hexdigest()} "
        f"cases={len(rows)} commands=1"
    )


def _fingerprint_fixture_bytes(case_id: str) -> bytes:
    common: dict[str, Any] = {
        "schema_version": "radiosim.sci004.fingerprint-red-fixture.v1",
        "family_id": "mmode_single_scalar_mode",
        "layout_document_raw_sha256": FINGERPRINT_LAYOUT_RAW_SHA256,
    }
    if case_id == "m3.fingerprint.preimage-retained":
        common.update(
            {
                "root_labels": ["ROOT-A"],
                "required_record_keys": list(FINGERPRINT_FAMILY_RECORD_V2_KEYS),
            }
        )
    elif case_id == "m3.fingerprint.path-independent":
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
    else:  # pragma: no cover - closed governed table
        raise RedRecordError(f"unknown fingerprint case {case_id!r}")
    return canonical_json_bytes(common)


def generate_fingerprint_post_source() -> None:
    """Generate correction #25's exact two-red/three-green fingerprint delta."""
    output = REPOSITORY_ROOT / FINGERPRINT_POST_SOURCE_OUTPUT_PATH
    if output.exists():
        raise RedRecordError(f"{output} already exists; generation never overwrites")

    head = _git("rev-parse", "HEAD").strip()
    _authenticate_fingerprint_retry_chain(head)
    changed = _changed_paths()
    if changed != FINGERPRINT_NON_ARTIFACT_PATHS:
        raise RedRecordError(
            "generate-fingerprint-post-source requires exactly the four "
            f"non-artifact R3 paths dirty; observed {changed}"
        )

    _regular_retained_blob(
        OUTPUT_PATH,
        HISTORICAL_RED_SLICE_SHA,
        HISTORICAL_RED_RECORD_SHA256,
    )
    old_r3 = _frozen_binding("SUPERSEDED_FINGERPRINT_R3_SHA")
    _regular_retained_blob(
        POST_SOURCE_OUTPUT_PATH,
        old_r3,
        CORRECTION24_POST_SOURCE_RED_RECORD_SHA256,
    )

    protected_before = _protected_digest(FINGERPRINT_R3_AUTHORIZED_PATHS)
    with tempfile.TemporaryDirectory(
        prefix="sci004-m3-fingerprint-post-source-red-"
    ) as scratch:
        junit_path = Path(scratch) / "junit.xml"
        command, stdout, _stderr = _run_pytest(FINGERPRINT_NODEIDS, junit_path)
        if command["exit_code"] != 1:
            raise RedRecordError(
                "fingerprint pytest must exit exactly 1; observed "
                f"{command['exit_code']}\n{stdout.decode('utf-8', 'replace')[-4000:]}"
            )
        if not junit_path.is_file():
            raise RedRecordError("fingerprint pytest produced no JUnit report")
        inventory = _parse_junit_inventory(junit_path)

    expected_names = tuple(nodeid.split("::", 1)[1] for nodeid in FINGERPRINT_NODEIDS)
    observed_names = tuple(name for name, _entry in inventory)
    if observed_names != expected_names:
        raise RedRecordError(
            "fingerprint JUnit inventory was missing, duplicated, or reordered: "
            f"{observed_names}"
        )
    observed_by_nodeid = {
        nodeid: inventory[index][1] for index, nodeid in enumerate(FINGERPRINT_NODEIDS)
    }
    for spec in FINGERPRINT_CASE_SPECS:
        entry = observed_by_nodeid[spec["test_nodeid"]]
        if entry["outcome"] != "failed" or _classify(entry["type"]) != "assertion":
            raise RedRecordError(
                f"{spec['test_nodeid']} was not one ordinary assertion: {entry}"
            )
        if spec["expected_failure_pattern"] not in entry["message"]:
            raise RedRecordError(
                f"{spec['test_nodeid']} did not contain its governed message"
            )
    for spec in FINGERPRINT_CONTROL_SPECS:
        entry = observed_by_nodeid[spec["test_nodeid"]]
        if entry != {"outcome": "passed", "type": "", "message": ""}:
            raise RedRecordError(
                f"passing control {spec['test_nodeid']} did not pass exactly: {entry}"
            )
    if sum(entry["outcome"] == "failed" for _name, entry in inventory) != 2:
        raise RedRecordError("the fingerprint partition did not contain two failures")
    if sum(entry["outcome"] == "passed" for _name, entry in inventory) != 3:
        raise RedRecordError("the fingerprint partition did not contain three passes")

    protected_after = _protected_digest(FINGERPRINT_R3_AUTHORIZED_PATHS)
    if protected_before != protected_after:
        raise RedRecordError("a protected path changed during fingerprint generation")

    case_rows: list[dict[str, Any]] = []
    expected_fixture_digests = {
        "m3.fingerprint.preimage-retained": (
            "4c11755ecae7597f8ffb30f7aa5653eda41a58994fae19086bb15109c60558b6",
            "b5c765aaae957ea3d686e3693b9a2469f7e491b0c247ac695e4ad3e0178b8a0b",
        ),
        "m3.fingerprint.path-independent": (
            "a24c64f4d981fce69c9f6cebaadd1bca0ae52fed0783e94b67ac3d8245df4a4f",
            "98cb2605eacaa5e473fbc573fa135ec91e0b2320e9759bbfa646c706628ef6ac",
        ),
    }
    for spec in FINGERPRINT_CASE_SPECS:
        entry = observed_by_nodeid[spec["test_nodeid"]]
        fixture_bytes = _fingerprint_fixture_bytes(spec["case_id"])
        invalid_digest = hashlib.sha256(fixture_bytes).hexdigest()
        fixture_digest = fixture_identity_sha256(
            phase=PHASE,
            fixture_id=spec["case_id"],
            requirement_id=spec["requirement_id"],
            test_nodeid=spec["test_nodeid"],
            pre_fix_source_sha=FINGERPRINT_POST_SOURCE_PRE_FIX_SHA,
            invalid_config_raw_sha256=invalid_digest,
        )
        if (invalid_digest, fixture_digest) != expected_fixture_digests[
            spec["case_id"]
        ]:
            raise RedRecordError(
                f"{spec['case_id']} fixture identities do not match correction #25"
            )
        case_rows.append(
            {
                **spec,
                "invalid_config_raw_sha256": invalid_digest,
                "fixture_identity_sha256": fixture_digest,
                "command_index": 0,
                "exit_code": 1,
                "observed_outcome": "assertion",
                "observed_exception_type": entry["type"],
                "observed_message": entry["message"],
                "stdout_sha256": command["stdout_sha256"],
                "stderr_sha256": command["stderr_sha256"],
                "red_failure_confirmed": True,
            }
        )
    control_rows = [
        {
            **spec,
            "command_index": 0,
            "observed_outcome": "pass",
            "exit_code": 0,
            "pass": True,
        }
        for spec in FINGERPRINT_CONTROL_SPECS
    ]

    oracle_diff = _fingerprint_oracle_diff()
    if not oracle_diff:
        raise RedRecordError("the fingerprint characterization oracle diff is empty")
    document = {
        "schema_version": FINGERPRINT_POST_SOURCE_SCHEMA_VERSION,
        "phase": PHASE,
        "status": FINGERPRINT_POST_SOURCE_STATUS,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "design_sha": head,
        "pre_fix_source_sha": FINGERPRINT_POST_SOURCE_PRE_FIX_SHA,
        "red_commit_sha": None,
        "red_commit_sha_reason": FINGERPRINT_POST_SOURCE_RED_COMMIT_SHA_REASON,
        "protected_source_clean": True,
        "authorized_red_paths": list(FINGERPRINT_R3_AUTHORIZED_PATHS),
        "environment": _environment(),
        "cases": case_rows,
        "passing_controls": control_rows,
        "commands": [command],
        "claims_not_licensed": list(FINGERPRINT_POST_SOURCE_CLAIMS_NOT_LICENSED),
        "historical_red_record_sha256": HISTORICAL_RED_RECORD_SHA256,
        "correction24_post_source_red_record_sha256": (
            CORRECTION24_POST_SOURCE_RED_RECORD_SHA256
        ),
        "oracle_patch_paths": [FINGERPRINT_POST_SOURCE_ORACLE_PATH],
        "oracle_patch_sha256": hashlib.sha256(oracle_diff).hexdigest(),
    }
    payload = canonical_json_bytes(document)
    _atomic_no_overwrite(output, payload)
    after_publication = _changed_paths()
    if after_publication != FINGERPRINT_R3_AUTHORIZED_PATHS:
        output.unlink(missing_ok=True)
        raise RedRecordError(
            "fingerprint publication did not leave the exact five-path R3 diff: "
            f"{after_publication}"
        )
    print(
        json.dumps(
            {
                "path": FINGERPRINT_POST_SOURCE_OUTPUT_PATH,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "red_case_count": len(case_rows),
                "passing_control_count": len(control_rows),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
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
    subparsers.add_parser("generate-post-source")
    subparsers.add_parser("generate-fingerprint-post-source")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "generate":
            generate(arguments.output)
        elif arguments.command == "generate-post-source":
            generate_post_source()
        else:
            generate_fingerprint_post_source()
    except RedRecordError as error:
        print(f"SCI004_M3_RED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
