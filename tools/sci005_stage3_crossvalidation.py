"""Generate and validate the SCI-005 Stage-3 cross-validation artifact.

``docs/development/sci005_beam_physics_plan.md`` Section 7.4 grants this module
for exactly one purpose: it is the non-gating cross-validation artifact
generator and standalone validator that Section 8.1's evidence-generation
transaction freezes, and nothing else.  It is never imported by production code
and never by a gating test.

Two forms, two environments
===========================

``generate`` runs only from the optional ``crossval`` Pixi environment, from a
globally clean tree whose ``HEAD`` equals the approved Stage-3 source commit::

    pixi run --environment crossval -- python \\
      tools/sci005_stage3_crossvalidation.py generate \\
      --approved-source-sha <40hex-S3> \\
      --output <absolute-temporary-artifact.json>

``--output`` must be an absolute regular-file path that does not exist and that
resolves **outside** repository root, because the only admissible way for those
bytes to enter the repository is Section 8.1's evidence transaction.

``validate`` is read-only, uses the Python standard library alone, and runs in
the **standard** environment where ``pyuvsim``, ``pyradiosky`` and ``mpi4py``
are absent::

    pixi run python tools/sci005_stage3_crossvalidation.py validate \\
      --approved-source-sha <40hex-S3> \\
      --artifact-sha256 <64hex> \\
      --input output/crossvalidation/<date>-sci005-efield-pyuvsim-1.4.0.json

Every scientific dependency -- ``numpy``, ``astropy``, ``pyuvdata``,
``pyradiosky``, ``pyuvsim`` and ``radiosim`` itself -- is therefore imported
**lazily, inside the generate path only**.  Module import and the whole
validate path must keep working with the standard library alone; this mirrors
the accepted ``tools/wp6_sci007_evidence.py`` precedent.  The validator
authenticates recorded strings and numbers, not arrays.

What the comparison compares
============================

Section 5.5: the same full-efield UVBeam file, full-Stokes point sky, times,
frequencies and antennas drive both simulators, under the accepted east-X and
fringe mappings, and the run records per-correlation absolute and relative
residuals, the exact source commit, input content hashes and every convention
mapping.

The RadioSim side runs the Stage-3 full-efield path end to end through the
public :class:`radiosim.Simulator`: a ``beams.shared_fits`` source whose
``normalization`` is the ``uvbeam_peak_common_v1`` literal, so
``core/beam/fits.py`` takes the accepted full-efield subset and
``core/beam/runtime.py`` publishes ``E = C^dagger J_native``.  The reference
side drives ``pyuvsim.UVEngine`` over ``pyuvsim.uvsim.UVTask`` directly,
because ``pyuvsim``'s MPI driver needs ``mpi4py``, which the ``crossval``
feature does not carry; that is the accepted approach of
``tests/crossvalidation/test_pyuvsim_comparison.py`` and reimplements no
``pyuvsim`` code.

Language discipline
===================

Neither form gates, and the comparison licenses only the sentence "compared
against pyuvsim for the named fixture, with the recorded agreements and open
disagreements".  It never licenses an unqualified "validated against pyuvsim".

Fixtures are built by this module into a temporary directory outside the
repository and the exact bytes fed to both simulators are hashed into
``input_hashes``; no test package is imported, and nothing but the frozen
absolute out-of-repository artifact is ever written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "radiosim.sci005.stage3-crossvalidation.v1"
REFERENCE_PACKAGE = "pyuvsim"
REFERENCE_VERSION = "1.4.0"
PYUVDATA_VERSION = "3.2.1"
PIXI_GENERATION_ENVIRONMENT = "crossval"
TARGET_DIRECTORY = "output/crossvalidation"
TARGET_SUFFIX = "sci005-efield-pyuvsim-1.4.0.json"

#: The artifact's exact key set, in the exact order Section 8.1 freezes.
ARTIFACT_KEY_ORDER: tuple[str, ...] = (
    "schema_version",
    "generated_at_utc",
    "source_sha",
    "target_path",
    "gating",
    "reference_package",
    "reference_version",
    "pyuvdata_version",
    "pyradiosky_version",
    "astropy_version",
    "radiosim_version",
    "output_basis",
    "input_hashes",
    "convention_mappings",
    "correlation_residuals",
    "open_disagreements",
    "commands",
)

INPUT_HASH_KEYS: tuple[str, ...] = ("name", "sha256")
CONVENTION_KEYS: tuple[str, ...] = (
    "radiosim_convention",
    "reference_convention",
    "equivalent",
)
RESIDUAL_KEYS: tuple[str, ...] = (
    "correlation",
    "max_abs_residual",
    "max_rel_residual",
    "reference_max_abs",
)
COMMAND_KEYS: tuple[str, ...] = (
    "argv",
    "cwd",
    "pixi_environment",
    "started_at_utc",
    "duration_seconds",
    "exit_code",
    "stdout_sha256",
    "stderr_sha256",
)

#: Section 8.1 requires ``input_hashes`` to cover *at minimum* the UVBeam file,
#: the sky model, the antenna layout, and the observation specification.  These
#: are the exact names this generator writes for those four roles; the tuple is
#: already in lexical order, which is also the required sort order.
REQUIRED_INPUT_NAMES: tuple[str, ...] = (
    "antenna_layout",
    "observation_specification",
    "sky_model",
    "uvbeam_file",
)

#: Section 8.1 requires ``convention_mappings`` to cover *at minimum* the
#: east-X orientation, the fringe sign, the Stokes-to-coherency factor, and the
#: beam normalization.  Lexically ordered, which is the required sort order.
REQUIRED_CONVENTION_NAMES: tuple[str, ...] = (
    "beam_normalization",
    "east_x_orientation",
    "fringe_sign",
    "stokes_to_coherency_factor",
)

#: A literal transcription of ``CORRELATION_LABELS`` in
#: ``src/radiosim/core/polarization_basis.py``.  The validate path may not
#: import ``radiosim`` (that would pull in NumPy), so the table is mirrored
#: here; the generate path *does* import the production mapping and asserts
#: this transcription against it, so the two cannot drift silently.
CORRELATION_LABELS_BY_BASIS: dict[str, tuple[str, ...]] = {
    "linear_xy": ("XX", "XY", "YX", "YY"),
    "circular_rl": ("RR", "RL", "LR", "LL"),
}

# Resolvable ambiguity, recorded here as the memo's Section 8.1 review
# requires.  The global rule for a declared-sorted array says the sort is
# strictly lexical by the named key; the `crossvalidation_comparisons` row
# prose instead says the four values are "the complete *ordered* label set"
# that `core/polarization_basis.py` gives the row's `output_basis`.  This
# module implements **lexical sort by `correlation`** plus **set equality**
# against `CORRELATION_LABELS[output_basis]`, and generates with
# `output_basis == "linear_xy"`, whose canonical order `XX, XY, YX, YY` is
# already lexical -- so under this artifact both readings coincide exactly and
# nothing is decided that the memo left open.  (For `circular_rl` the two
# readings would differ: canonical `RR, RL, LR, LL` versus lexical
# `LL, LR, RL, RR`.)
OUTPUT_BASIS = "linear_xy"

# --------------------------------------------------------------------------
# Declared comparison bounds.
#
# These are *classification* bounds, not gates.  The generator never fails a
# measurement: it records every measured residual and enumerates each declared
# bound the measurement exceeds in `open_disagreements`, so the artifact tells
# the truth whatever the truth is.  The granted pytest module asserts the same
# bounds hard, so a disagreement is loud where a human will see it, while still
# never gating (that module is marked `crossval` and `slow`).
# --------------------------------------------------------------------------

#: Total intensity -- the `XX + YY` trace -- is invariant under a unitary
#: rotation of the tangent basis, because `J -> J R` and `B -> R^H B R` leave
#: `Tr(J_1 B J_2^H)` unchanged.  The residual frame difference between
#: RadioSim's parallactic angle and pyradiosky's exact tangent-basis rotation
#: (WP-6 SCI-007) therefore cannot enter the trace, and what remains is
#: interpolation and double-precision round-off.
TRACE_RELATIVE_BOUND = 1e-6

#: An individual correlation is *not* frame invariant: the SCI-007 record
#: measures the RadioSim-minus-pyradiosky frame angle at about 0.05 degrees for
#: this class of fixture, i.e. a few parts in a thousand of the linearly
#: polarized amplitude.  One part in a hundred sits comfortably above that
#: known milli-radian effect and far below the order-unity disagreement an
#: actually wrong convention mapping produces.
PER_CORRELATION_RELATIVE_BOUND = 1e-2

#: The comparison must not be vacuous: undoing the fringe Hermitian mapping has
#: to move the cubes by order unity, which proves the assertions above test the
#: fringe rather than an accidental agreement of two near-zero arrays.
CONTROL_RELATIVE_FLOOR = 0.1

# --------------------------------------------------------------------------
# The authored fixture.  Every number here is an input to both simulators.
# --------------------------------------------------------------------------

TELESCOPE_NAME = "SCI005CV"
LATITUDE_DEG = -30.72152
LONGITUDE_DEG = 21.42830
HEIGHT_M = 1073.0

#: Coplanar in Up, so every baseline's `w` is exactly zero and the fringe
#: convention mapping reduces to a per-baseline Hermitian conjugate with no
#: residual `w` phase.
ANTENNA_ENU_M: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (50.0, 0.0, 0.0),
    (0.0, 70.0, 0.0),
)
ANTENNA_DIAMETER_M = 10.0

#: Strictly inside the beam file's intrinsic grid, so the default cubic
#: frequency interpolation has four intrinsic samples to work from.
FREQUENCIES_HZ: tuple[float, ...] = (105_000_000.0, 115_000_000.0, 125_000_000.0)
CHANNEL_WIDTH_HZ = 1_000_000.0
START_TIME_ISO = "2025-01-01T00:00:00"
CADENCE_SECONDS = 120.0
TIME_SAMPLES = 3

#: A genuinely full-Stokes sky: every source carries at least one nonzero
#: `Q`, `U` or `V`.
SOURCE_IQUV: tuple[tuple[float, float, float, float], ...] = (
    (3.0, 0.6, -0.4, 0.2),
    (1.5, -0.3, 0.5, -0.1),
    (2.25, 0.0, 0.0, 0.9),
)
SOURCE_RA_DEG: tuple[float, ...] = (20.0, 25.0, 15.0)
SOURCE_DEC_DEG: tuple[float, ...] = (-30.72, -26.0, -35.0)

#: The full-efield BeamFITS grid.  Eight endpoint-excluded azimuth samples and
#: five zenith-angle samples from the zenith through the horizon give complete
#: visible coverage and exact wrap continuity; four intrinsic frequencies are
#: the minimum cubic frequency interpolation accepts.
BEAM_AZIMUTH_SAMPLES = 8
BEAM_ZENITH_ANGLE_SAMPLES = 5
BEAM_FREQUENCIES_HZ: tuple[float, ...] = (
    100_000_000.0,
    110_000_000.0,
    120_000_000.0,
    130_000_000.0,
)
BEAM_NORMALIZATION_LITERAL = "uvbeam_peak_common_v1"

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_RFC3339_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_TARGET_PATH = re.compile(
    rf"{re.escape(TARGET_DIRECTORY)}/(\d{{4}}-\d{{2}}-\d{{2}})-{re.escape(TARGET_SUFFIX)}\Z"
)


class CrossValidationError(ValueError):
    """One refusal from either form of this tool."""


# --------------------------------------------------------------------------
# Standard-library helpers shared by both forms.
# --------------------------------------------------------------------------


def _validate_sha(value: object, *, length: int, label: str) -> str:
    pattern = _HEX40 if length == 40 else _HEX64
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise CrossValidationError(
            f"{label} must be exactly {length} lowercase hexadecimal characters"
        )
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _serialize(record: dict[str, Any]) -> bytes:
    """Return the artifact's exact bytes.

    UTF-8 without a byte-order mark, LF endings, ``ensure_ascii=false``,
    ``allow_nan=false``, two-space indentation, one final newline, and the
    caller's key order preserved.
    """
    text = json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2)
    return (text + "\n").encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise CrossValidationError(f"duplicate JSON key: {key!r}")
        seen[key] = value
    return seen


def _reject_nonfinite_constant(value: str) -> None:
    raise CrossValidationError(f"non-finite JSON constant is forbidden: {value}")


def _decode_json(raw: bytes, source: Path) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise CrossValidationError(f"{source}: artifact must carry no byte-order mark")
    if b"\r" in raw:
        raise CrossValidationError(f"{source}: artifact must use LF line endings")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise CrossValidationError(f"{source}: artifact must end with one newline")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CrossValidationError(f"{source}: artifact is not valid UTF-8") from exc
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as exc:
        raise CrossValidationError(
            f"{source}: artifact is not valid JSON: {exc}"
        ) from exc
    if type(decoded) is not dict:
        raise CrossValidationError(f"{source}: artifact must be a JSON object")
    return decoded


def _require_key_order(
    value: object, expected: tuple[str, ...], path: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise CrossValidationError(f"{path} must be a JSON object")
    actual = tuple(value)
    if actual != expected:
        raise CrossValidationError(
            f"{path} must carry exactly {list(expected)} in that order, got "
            f"{list(actual)}"
        )
    return value


def _require_string(value: object, path: str) -> str:
    if type(value) is not str or not value:
        raise CrossValidationError(f"{path} must be a non-empty string")
    return value


def _require_literal(value: object, expected: str, path: str) -> str:
    if value != expected:
        raise CrossValidationError(
            f"{path} must be exactly {expected!r}, got {value!r}"
        )
    return expected


def _require_number(value: object, path: str) -> float:
    if type(value) is not int and type(value) is not float:
        raise CrossValidationError(f"{path} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise CrossValidationError(f"{path} must be finite")
    return number


def _require_non_negative(value: object, path: str) -> float:
    number = _require_number(value, path)
    if number < 0.0:
        raise CrossValidationError(f"{path} must be non-negative, got {number!r}")
    return number


def _require_boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise CrossValidationError(f"{path} must be a boolean")
    return value


def _require_list(value: object, path: str, *, minimum: int) -> list[Any]:
    if type(value) is not list:
        raise CrossValidationError(f"{path} must be a JSON array")
    if len(value) < minimum:
        raise CrossValidationError(f"{path} must carry at least {minimum} entries")
    return value


def _require_timestamp(value: object, path: str) -> str:
    text = _require_string(value, path)
    if _RFC3339_UTC.fullmatch(text) is None:
        raise CrossValidationError(f"{path} must be an RFC 3339 UTC timestamp")
    try:
        datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise CrossValidationError(f"{path} is not a real UTC instant: {text}") from exc
    return text


def _require_sorted_unique_keys(names: list[str], path: str, *, key_name: str) -> None:
    if len(set(names)) != len(names):
        raise CrossValidationError(f"{path} must carry a unique {key_name} per row")
    if names != sorted(names):
        raise CrossValidationError(f"{path} must be sorted by {key_name}")


# --------------------------------------------------------------------------
# The standalone read-only validator.
# --------------------------------------------------------------------------


def _validate_input_hashes(value: object) -> None:
    rows = _require_list(value, "input_hashes", minimum=1)
    names: list[str] = []
    for index, row in enumerate(rows):
        path = f"input_hashes[{index}]"
        mapping = _require_key_order(row, INPUT_HASH_KEYS, path)
        names.append(_require_string(mapping["name"], f"{path}.name"))
        _validate_sha(mapping["sha256"], length=64, label=f"{path}.sha256")
    _require_sorted_unique_keys(names, "input_hashes", key_name="name")
    missing = [name for name in REQUIRED_INPUT_NAMES if name not in set(names)]
    if missing:
        raise CrossValidationError(
            f"input_hashes must cover at minimum {list(REQUIRED_INPUT_NAMES)}; "
            f"missing {missing}"
        )


def _validate_convention_mappings(value: object) -> None:
    rows = _require_list(value, "convention_mappings", minimum=1)
    names: list[str] = []
    for index, row in enumerate(rows):
        path = f"convention_mappings[{index}]"
        mapping = _require_key_order(row, CONVENTION_KEYS, path)
        names.append(
            _require_string(
                mapping["radiosim_convention"], f"{path}.radiosim_convention"
            )
        )
        _require_string(mapping["reference_convention"], f"{path}.reference_convention")
        _require_boolean(mapping["equivalent"], f"{path}.equivalent")
    _require_sorted_unique_keys(
        names, "convention_mappings", key_name="radiosim_convention"
    )
    missing = [name for name in REQUIRED_CONVENTION_NAMES if name not in set(names)]
    if missing:
        raise CrossValidationError(
            "convention_mappings must cover at minimum "
            f"{list(REQUIRED_CONVENTION_NAMES)}; missing {missing}"
        )


def _validate_correlation_residuals(value: object, output_basis: str) -> None:
    rows = _require_list(value, "correlation_residuals", minimum=4)
    if len(rows) != 4:
        raise CrossValidationError("correlation_residuals must carry exactly four rows")
    names: list[str] = []
    for index, row in enumerate(rows):
        path = f"correlation_residuals[{index}]"
        mapping = _require_key_order(row, RESIDUAL_KEYS, path)
        names.append(_require_string(mapping["correlation"], f"{path}.correlation"))
        _require_non_negative(mapping["max_abs_residual"], f"{path}.max_abs_residual")
        _require_non_negative(mapping["max_rel_residual"], f"{path}.max_rel_residual")
        _require_non_negative(mapping["reference_max_abs"], f"{path}.reference_max_abs")
    # Lexical sort by `correlation`, plus set equality against the production
    # label set for the artifact's `output_basis`; see the module comment on
    # `OUTPUT_BASIS` for why the two readings coincide here.
    _require_sorted_unique_keys(names, "correlation_residuals", key_name="correlation")
    expected = set(CORRELATION_LABELS_BY_BASIS[output_basis])
    if set(names) != expected:
        raise CrossValidationError(
            f"correlation_residuals must name exactly {sorted(expected)} for "
            f"output_basis={output_basis!r}, got {sorted(names)}"
        )


def _validate_open_disagreements(value: object) -> None:
    rows = _require_list(value, "open_disagreements", minimum=0)
    entries = [
        _require_string(row, f"open_disagreements[{index}]")
        for index, row in enumerate(rows)
    ]
    _require_sorted_unique_keys(entries, "open_disagreements", key_name="value")


def _validate_commands(value: object) -> None:
    rows = _require_list(value, "commands", minimum=1)
    for index, row in enumerate(rows):
        path = f"commands[{index}]"
        mapping = _require_key_order(row, COMMAND_KEYS, path)
        argv = _require_list(mapping["argv"], f"{path}.argv", minimum=1)
        for position, token in enumerate(argv):
            _require_string(token, f"{path}.argv[{position}]")
        _require_literal(mapping["cwd"], ".", f"{path}.cwd")
        _require_string(mapping["pixi_environment"], f"{path}.pixi_environment")
        _require_timestamp(mapping["started_at_utc"], f"{path}.started_at_utc")
        _require_non_negative(mapping["duration_seconds"], f"{path}.duration_seconds")
        exit_code = mapping["exit_code"]
        if type(exit_code) is not int:
            raise CrossValidationError(f"{path}.exit_code must be a signed integer")
        if exit_code != 0:
            raise CrossValidationError(
                f"{path}.exit_code must be zero for a candidate artifact"
            )
        _validate_sha(
            mapping["stdout_sha256"], length=64, label=f"{path}.stdout_sha256"
        )
        _validate_sha(
            mapping["stderr_sha256"], length=64, label=f"{path}.stderr_sha256"
        )


def validate_record(record: dict[str, Any], *, approved_source_sha: str) -> None:
    """Validate one decoded artifact against the frozen Section 8.1 contract."""
    _validate_sha(approved_source_sha, length=40, label="approved source SHA")
    _require_key_order(record, ARTIFACT_KEY_ORDER, "artifact")

    _require_literal(record["schema_version"], SCHEMA_VERSION, "schema_version")
    generated_at = _require_timestamp(record["generated_at_utc"], "generated_at_utc")
    source_sha = _validate_sha(record["source_sha"], length=40, label="source_sha")
    if source_sha != approved_source_sha:
        raise CrossValidationError(
            f"source_sha {source_sha} does not equal the approved Stage-3 source "
            f"commit {approved_source_sha}"
        )

    target_path = _require_string(record["target_path"], "target_path")
    match = _TARGET_PATH.fullmatch(target_path)
    if match is None:
        raise CrossValidationError(
            "target_path must be exactly "
            f"{TARGET_DIRECTORY}/<YYYY-MM-DD>-{TARGET_SUFFIX}, got {target_path!r}"
        )
    if match.group(1) != generated_at[:10]:
        raise CrossValidationError(
            f"target_path date {match.group(1)} does not equal the UTC date of "
            f"generated_at_utc {generated_at}"
        )

    if record["gating"] is not False:
        raise CrossValidationError("gating must be exactly the boolean false")

    _require_literal(
        record["reference_package"], REFERENCE_PACKAGE, "reference_package"
    )
    _require_literal(
        record["reference_version"], REFERENCE_VERSION, "reference_version"
    )
    _require_literal(record["pyuvdata_version"], PYUVDATA_VERSION, "pyuvdata_version")
    _require_string(record["pyradiosky_version"], "pyradiosky_version")
    _require_string(record["astropy_version"], "astropy_version")
    _require_string(record["radiosim_version"], "radiosim_version")

    output_basis = _require_string(record["output_basis"], "output_basis")
    if output_basis not in CORRELATION_LABELS_BY_BASIS:
        raise CrossValidationError(
            f"output_basis must be 'linear_xy' or 'circular_rl', got {output_basis!r}"
        )

    _validate_input_hashes(record["input_hashes"])
    _validate_convention_mappings(record["convention_mappings"])
    _validate_correlation_residuals(record["correlation_residuals"], output_basis)
    _validate_open_disagreements(record["open_disagreements"])
    _validate_commands(record["commands"])


def validate_artifact(
    input_path: Path,
    *,
    approved_source_sha: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    """Authenticate artifact bytes and semantics, returning the parsed record."""
    _validate_sha(artifact_sha256, length=64, label="artifact SHA-256")
    try:
        raw = input_path.read_bytes()
    except FileNotFoundError as exc:
        raise CrossValidationError(f"artifact is absent: {input_path}") from exc
    except OSError as exc:
        raise CrossValidationError(
            f"could not read artifact {input_path}: {exc}"
        ) from exc
    measured = _sha256_bytes(raw)
    if measured != artifact_sha256:
        raise CrossValidationError(
            f"artifact SHA-256 mismatch: expected {artifact_sha256}, measured "
            f"{measured}"
        )
    record = _decode_json(raw, input_path)
    validate_record(record, approved_source_sha=approved_source_sha)
    if _serialize(record) != raw:
        raise CrossValidationError(
            f"{input_path}: artifact bytes are not the canonical serialization "
            "(UTF-8 without BOM, LF endings, ensure_ascii=false, two-space "
            "indentation, one final newline, frozen key order)"
        )
    return record


# --------------------------------------------------------------------------
# Generation: environment, source, and output preflight.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CommandOutcome:
    """One recorded subprocess invocation and its captured stdout."""

    stdout: str
    row: dict[str, Any]


def _run_recorded(argv: list[str]) -> _CommandOutcome:
    """Run one command without a shell and build its Section 8.1 row."""
    started_at = _utc_now()
    clock = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, never a shell
            argv,
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise CrossValidationError(f"{' '.join(argv)} failed: {exc}") from exc
    duration = time.monotonic() - clock
    row = {
        "argv": list(argv),
        "cwd": ".",
        "pixi_environment": os.environ.get("PIXI_ENVIRONMENT_NAME", ""),
        "started_at_utc": started_at,
        "duration_seconds": round(duration, 6) + 0.0,
        "exit_code": int(completed.returncode),
        "stdout_sha256": _sha256_bytes(completed.stdout),
        "stderr_sha256": _sha256_bytes(completed.stderr),
    }
    if completed.returncode != 0:
        raise CrossValidationError(
            f"{' '.join(argv)} exited with {completed.returncode}: "
            f"{completed.stderr.decode('utf-8', 'replace').strip()}"
        )
    return _CommandOutcome(stdout=completed.stdout.decode("utf-8").strip(), row=row)


def _assert_generation_environment() -> None:
    """Require the repository's locked optional ``crossval`` environment."""
    if os.environ.get("PIXI_ENVIRONMENT_NAME") != PIXI_GENERATION_ENVIRONMENT:
        raise CrossValidationError(
            "generation requires PIXI_ENVIRONMENT_NAME="
            f"{PIXI_GENERATION_ENVIRONMENT}; use the frozen command "
            "`pixi run --environment crossval -- python "
            "tools/sci005_stage3_crossvalidation.py generate ...`"
        )
    expected_prefix = (
        REPO_ROOT / ".pixi" / "envs" / PIXI_GENERATION_ENVIRONMENT
    ).resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise CrossValidationError(
            "generation is not running from the repository's locked crossval "
            f"environment: expected {expected_prefix}, got {Path(sys.prefix)}"
        )
    project_root = os.environ.get("PIXI_PROJECT_ROOT")
    if not project_root or Path(project_root).resolve() != REPO_ROOT:
        raise CrossValidationError(
            "PIXI_PROJECT_ROOT must resolve to the generating repository root"
        )
    manifest = os.environ.get("PIXI_PROJECT_MANIFEST")
    if not manifest or Path(manifest).resolve() != (REPO_ROOT / "pixi.toml").resolve():
        raise CrossValidationError(
            "PIXI_PROJECT_MANIFEST must resolve to the generating pixi.toml"
        )
    if Path.cwd().resolve() != REPO_ROOT:
        raise CrossValidationError(
            f"generation must run from repository root {REPO_ROOT}, got {Path.cwd()}"
        )


def _resolve_output(output: Path) -> Path:
    """Require an absolute, absent, out-of-repository regular-file target."""
    if not output.is_absolute():
        raise CrossValidationError(f"--output must be an absolute path, got {output}")
    if output.exists() or output.is_symlink():
        raise CrossValidationError(f"refusing to overwrite existing --output: {output}")
    parent = output.parent
    if not parent.is_dir():
        raise CrossValidationError(f"--output parent directory is absent: {parent}")
    resolved = parent.resolve(strict=True) / output.name
    if resolved == REPO_ROOT or resolved.is_relative_to(REPO_ROOT):
        raise CrossValidationError(
            f"--output must resolve outside repository root {REPO_ROOT}; the only "
            "admissible way for these bytes to enter the repository is the "
            "Section 8.1 evidence transaction"
        )
    if resolved.exists() or resolved.is_symlink():
        raise CrossValidationError(
            f"refusing to overwrite existing --output: {resolved}"
        )
    return resolved


def _assert_clean_source(approved_source_sha: str) -> list[dict[str, Any]]:
    """Require ``HEAD == S3`` and a globally clean tree; record both commands."""
    head = _run_recorded(["git", "rev-parse", "HEAD"])
    if head.stdout != approved_source_sha:
        raise CrossValidationError(
            f"HEAD {head.stdout} does not equal the approved source SHA "
            f"{approved_source_sha}"
        )
    status = _run_recorded(["git", "status", "--porcelain", "--untracked-files=all"])
    if status.stdout:
        raise CrossValidationError(
            f"generation requires a globally clean tree; git reported:\n{status.stdout}"
        )
    return [head.row, status.row]


def _atomic_write_new(path: Path, data: bytes) -> None:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CrossValidationError(
                f"refusing to overwrite existing artifact: {path}"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# The comparison itself.  Every scientific import below is lazy.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComparisonInputs:
    """The exact bytes fed to both simulators, and their digests."""

    beamfits: Path
    skyh5: Path
    array: Path
    specification: Path
    hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """One completed comparison, richer than the artifact it serializes to.

    The artifact carries the frozen Section 8.1 subset; the extra fields here
    exist so the granted pytest module can assert the physical invariants and
    the non-vacuity control on the same measurement.
    """

    output_basis: str
    correlations: tuple[str, ...]
    residuals: dict[str, tuple[float, float, float]]
    trace_relative: float
    control_relative_without_fringe_mapping: float
    reference_scale: float
    open_disagreements: tuple[str, ...]
    input_hashes: tuple[tuple[str, str], ...]
    shape: tuple[int, ...]


def _earth_location() -> Any:
    from astropy import units
    from astropy.coordinates import EarthLocation

    return EarthLocation.from_geodetic(
        lon=LONGITUDE_DEG * units.deg,
        lat=LATITUDE_DEG * units.deg,
        height=HEIGHT_M * units.m,
    )


def _write_specification(path: Path) -> dict[str, Any]:
    """Write and return the observation specification both codes are driven by.

    This is a real input rather than a description of one: the RadioSim mapping
    and the reference driver are both built from the parsed file, so the digest
    recorded under ``observation_specification`` covers the bytes that actually
    determined the run.
    """
    specification = {
        "antennas_enu_m": [list(position) for position in ANTENNA_ENU_M],
        "antenna_diameter_m": ANTENNA_DIAMETER_M,
        "cadence_seconds": CADENCE_SECONDS,
        "channel_width_hz": CHANNEL_WIDTH_HZ,
        "correlations": "cross",
        "frequencies_hz": list(FREQUENCIES_HZ),
        "location": {
            "height_m": HEIGHT_M,
            "latitude_deg": LATITUDE_DEG,
            "longitude_deg": LONGITUDE_DEG,
        },
        "mount_type": "alt-az",
        "output_basis_request": "linear",
        "source_dec_deg": list(SOURCE_DEC_DEG),
        "source_iquv_jy": [list(values) for values in SOURCE_IQUV],
        "source_ra_deg": list(SOURCE_RA_DEG),
        "start_time_iso": START_TIME_ISO,
        "telescope_name": TELESCOPE_NAME,
        "time_samples": TIME_SAMPLES,
    }
    path.write_bytes(_serialize(specification))
    return specification


def _write_efield_beamfits(path: Path) -> None:
    """Write the full-efield BeamFITS both simulators read.

    Two crossed ideal dipoles, stored in pyuvdata 3.2.1's own ``az_za`` efield
    ``data_array`` convention -- first axis ``[azimuth, zenith angle]``, second
    axis ``[east-aligned feed, north-aligned feed]``, azimuth zero at East and
    increasing through North.  The stored grid maximum is exactly ``1`` at every
    intrinsic frequency (``|cos(0)| == 1`` lies on the azimuth grid), which is
    what the accepted ``uvbeam_peak_common_v1`` subset requires; the stored
    ``basis_vector_array`` is the exact native identity, the mount is ``fixed``,
    the bandpass is unit, and the feed angles are the ``(pi/2, 0)`` pair a
    linear ``('x', 'y')`` receptor with zero static feed rotation requires.
    """
    import numpy as np
    from pyuvdata import UVBeam

    azimuth = np.linspace(
        0.0, 2.0 * np.pi, BEAM_AZIMUTH_SAMPLES, endpoint=False, dtype=np.float64
    )
    zenith_angle = np.linspace(
        0.0, np.pi / 2.0, BEAM_ZENITH_ANGLE_SAMPLES, dtype=np.float64
    )
    frequencies = np.array(BEAM_FREQUENCIES_HZ, dtype=np.float64)

    azimuth_grid, zenith_grid = np.broadcast_arrays(
        azimuth[np.newaxis, :], zenith_angle[:, np.newaxis]
    )
    plane = np.zeros((2, 2) + azimuth_grid.shape, dtype=np.complex128)
    plane[0, 0] = -np.sin(azimuth_grid)
    plane[0, 1] = np.cos(azimuth_grid)
    plane[1, 0] = np.cos(zenith_grid) * np.cos(azimuth_grid)
    plane[1, 1] = np.cos(zenith_grid) * np.sin(azimuth_grid)

    data = np.zeros(
        (2, 2, frequencies.size, zenith_angle.size, azimuth.size),
        dtype=np.complex128,
    )
    for index in range(frequencies.size):
        data[:, :, index] = plane

    basis = np.zeros((2, 2, zenith_angle.size, azimuth.size), dtype=np.float64)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0

    beam = UVBeam.new(
        telescope_name="RadioSim SCI-005 Stage-3 cross-validation efield beam",
        data_normalization="peak",
        freq_array=frequencies,
        feed_name="RadioSim full efield",
        feed_version="sci005-stage3-full-efield-v1",
        model_name="RadioSim crossed_ideal_dipole efield beam",
        model_version="sci005-stage3-full-efield-v1",
        feed_array=np.array(["x", "y"]),
        feed_angle=np.array([np.pi / 2.0, 0.0], dtype=np.float64),
        mount_type="fixed",
        axis1_array=azimuth,
        axis2_array=zenith_angle,
        bandpass_array=np.ones(frequencies.size, dtype=np.float64),
        basis_vector_array=basis,
        data_array=data,
        history="RadioSim SCI-005 Stage-3 cross-validation full-efield beam. ",
    )
    if beam.beam_type != "efield" or beam.pixel_coordinate_system != "az_za":
        raise CrossValidationError("the fixture beam is not a full-efield az_za UVBeam")
    beam.write_beamfits(str(path), clobber=False)


def _write_sky(path: Path, specification: dict[str, Any]) -> None:
    """Write the single full-Stokes ``skyh5`` both simulators read."""
    import numpy as np
    from astropy import units
    from astropy.coordinates import SkyCoord
    from pyradiosky import SkyModel

    frequencies = np.array(specification["frequencies_hz"], dtype=np.float64)
    iquv = specification["source_iquv_jy"]
    stokes = np.zeros((4, frequencies.size, len(iquv))) * units.Jy
    for index, values in enumerate(iquv):
        for component, value in enumerate(values):
            stokes[component, :, index] = value * units.Jy
    sky = SkyModel(
        name=np.array([f"S{index}" for index in range(len(iquv))]),
        skycoord=SkyCoord(
            ra=np.asarray(specification["source_ra_deg"]) * units.deg,
            dec=np.asarray(specification["source_dec_deg"]) * units.deg,
            frame="icrs",
        ),
        stokes=stokes,
        spectral_type="full",
        freq_array=frequencies * units.Hz,
    )
    sky.write_skyh5(str(path), clobber=False)


def _write_array(path: Path, specification: dict[str, Any]) -> None:
    """Write the metadata-only UVFITS carrying the array and its mount type.

    ``mount_type`` reaches a RadioSim ``instrument:`` section only through a
    pyuvdata dataset source -- an ENU layout file has no column for it -- and
    the ``alt-az`` mount is what makes ``jones.P`` applicable, so that both
    codes rotate the receptor frame with the sky rather than only one of them.
    """
    import warnings

    import numpy as np
    from pyuvdata import UVData
    from pyuvdata.telescopes import Telescope
    from pyuvdata.utils import ECEF_from_ENU

    location = _earth_location()
    geocentric = np.array([value.to_value("m") for value in location.geocentric])
    enu = np.array(specification["antennas_enu_m"], dtype=np.float64)
    positions = ECEF_from_ENU(enu, center_loc=location) - geocentric
    count = enu.shape[0]
    telescope = Telescope.new(
        name=specification["telescope_name"],
        location=location,
        antenna_names=[f"A{index:03d}" for index in range(count)],
        antenna_numbers=np.arange(count),
        antenna_positions=positions,
        instrument=specification["telescope_name"],
        mount_type=specification["mount_type"],
        feed_array=["x", "y"],
        feed_angle=[np.pi / 2.0, 0.0],
        antenna_diameters=np.full(count, specification["antenna_diameter_m"]),
    )
    frequencies = np.array(specification["frequencies_hz"], dtype=np.float64)
    uvdata = UVData.new(
        freq_array=frequencies,
        # UVFITS rejects channels spaced more widely than their declared width;
        # only the telescope table of this file is ever read back.
        channel_width=np.full(frequencies.size, 1.0e7),
        polarization_array=np.array([-5, -6, -7, -8]),
        telescope=telescope,
        times=np.array([2460676.5, 2460676.50138889]),
        antpairs=[(0, 1), (0, 2), (1, 2)],
        do_blt_outer=True,
        integration_time=specification["cadence_seconds"],
        empty=True,
        phase_center_catalog={
            0: {"cat_name": "unprojected", "cat_type": "unprojected"}
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uvdata.write_uvfits(str(path), force_phase=True)


def build_inputs(directory: Path) -> ComparisonInputs:
    """Build the complete fixture below ``directory`` and hash every input."""
    specification_path = directory / "observation_specification.json"
    specification = _write_specification(specification_path)
    beamfits = directory / "sci005-stage3-efield.beamfits"
    _write_efield_beamfits(beamfits)
    skyh5 = directory / "sci005-stage3-full-stokes.skyh5"
    _write_sky(skyh5, specification)
    array = directory / "sci005-stage3-altaz.uvfits"
    _write_array(array, specification)
    hashes = tuple(
        sorted(
            (
                ("antenna_layout", _sha256_file(array)),
                ("observation_specification", _sha256_file(specification_path)),
                ("sky_model", _sha256_file(skyh5)),
                ("uvbeam_file", _sha256_file(beamfits)),
            )
        )
    )
    return ComparisonInputs(
        beamfits=beamfits,
        skyh5=skyh5,
        array=array,
        specification=specification_path,
        hashes=hashes,
    )


def _radiosim_mapping(
    inputs: ComparisonInputs, output_directory: Path
) -> dict[str, Any]:
    """Return the exact RadioSim document the comparison runs."""
    return {
        "instrument": {
            "source": {
                "kind": "layout_file",
                "path": str(inputs.array),
                "format": "uvfits",
            },
            "default_diameter_m": ANTENNA_DIAMETER_M,
        },
        "beams": {
            "mode": "shared_fits",
            "beam": {
                "kind": "fits",
                "path": str(inputs.beamfits),
                # The Stage-3 accepted full-efield subset.
                "normalization": BEAM_NORMALIZATION_LITERAL,
            },
        },
        "receptors": {"output_basis": "linear"},
        "baseline_selection": {"correlations": "cross"},
        "sky_model": {
            "flux_unit": "Jy",
            "sources": [{"kind": "pyradiosky_file", "filename": str(inputs.skyh5)}],
        },
        "obs_time": {
            "start_time": START_TIME_ISO,
            "duration_seconds": TIME_SAMPLES * CADENCE_SECONDS,
            "time_step_seconds": CADENCE_SECONDS,
        },
        "obs_frequency": {
            "mode": "explicit",
            "channel_frequencies_hz": list(FREQUENCIES_HZ),
            "channel_widths_hz": [CHANNEL_WIDTH_HZ] * len(FREQUENCIES_HZ),
        },
        "visibility": {"sky_representation": "point_sources"},
        "jones": {"P": {"enabled": True}},
        "execution": {"backend": "numpy", "offline": True},
        "workflow": {"save_results": False, "output_dir": str(output_directory)},
    }


def _run_radiosim(inputs: ComparisonInputs, output_directory: Path) -> Any:
    from radiosim import Simulator

    simulator = Simulator.from_mapping(_radiosim_mapping(inputs, output_directory))
    simulator.setup()
    return simulator.run(progress=False)


def _reference_sky(inputs: ComparisonInputs) -> Any:
    """Read the one committed sky file and apply the Stokes-V sign mapping.

    RadioSim builds ``B = (1/2) [[I+Q, U+iV], [U-iV, I-Q]]``
    (``core/polarization.py``); ``pyradiosky`` uses the mirror image,
    ``0.5 * [[I+Q, U-iV], [U+iV, I-Q]]`` (``pyradiosky/utils.py``,
    ``stokes_to_coherency``).  The forward model is linear in the Stokes
    parameters, so the reference run of the *same* sky bytes under the
    *opposite* ``V`` sign is exactly the reference RadioSim's convention
    predicts.  The conversion is derived from the two published definitions and
    is recorded as a non-equivalent convention mapping, never fitted.
    """
    from pyradiosky import SkyModel

    sky = SkyModel()
    sky.read_skyh5(str(inputs.skyh5))
    sky.stokes[3] = -sky.stokes[3]
    return sky


def _reference_cube(result: Any, inputs: ComparisonInputs, sky: Any) -> Any:
    """Evaluate the same cube with ``pyuvsim``'s own engine.

    Antenna positions, two-part UTC instants, frequencies and baseline ordering
    are taken from RadioSim's *resolved* result rather than from the inputs, so
    neither code is compared against a differently rounded version of the
    other's geometry or time.
    """
    import warnings

    import numpy as np
    from astropy.time import Time
    from pyuvdata import UVBeam
    from pyuvsim import Antenna, Baseline, BeamList, SkyModelData, Telescope, UVEngine
    from pyuvsim.uvsim import UVTask

    beam_list = BeamList([UVBeam.from_file(str(inputs.beamfits))])
    telescope = Telescope(TELESCOPE_NAME, _earth_location(), beam_list)
    antennas = {
        antenna.id.number: Antenna(
            f"A{antenna.id.number:03d}",
            int(antenna.id.number),
            np.asarray(antenna.position_enu_m, dtype=np.float64),
            0,
        )
        for antenna in result.instrument.antennas
    }
    grid = result.time_grid
    jd1 = np.asarray(grid.utc_jd1)
    jd2 = np.asarray(grid.utc_jd2)
    frequencies = np.asarray(result.frequencies_hz, dtype=np.float64)
    pairs = [
        (baseline.ant1.number, baseline.ant2.number)
        for baseline in result.selection.baselines
    ]
    cube = np.zeros((jd1.size, len(pairs), frequencies.size, 4), dtype=np.complex128)
    indices = np.arange(sky.Ncomponents)
    sky_data = SkyModelData(sky)
    for time_index in range(jd1.size):
        instant = Time(jd1[time_index], jd2[time_index], format="jd", scale="utc")
        for baseline_index, (first, second) in enumerate(pairs):
            baseline = Baseline(antennas[first], antennas[second])
            for freq_index, frequency in enumerate(frequencies):
                sources = sky_data.get_skymodel(indices)
                task = UVTask(
                    sources,
                    instant,
                    frequency,
                    baseline,
                    telescope,
                    freq_i=freq_index,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    visibility = UVEngine(task).make_visibility()
                # pyuvsim returns [xx, yy, xy, yx]; RadioSim reports
                # [XX, XY, YX, YY].
                cube[time_index, baseline_index, freq_index] = np.asarray(
                    [visibility[0], visibility[2], visibility[3], visibility[1]]
                )
    return cube


def _apply_fringe_mapping(cube: Any) -> Any:
    """Apply the fringe convention mapping: a per-baseline Hermitian conjugate.

    RadioSim evaluates ``exp(-2j*pi*(u*l + v*m + w*(n-1)))``
    (``core/jones/geometric.py``); ``pyuvsim`` evaluates
    ``exp(+2j*pi*(u*l + v*m + w*n))`` (``pyuvsim/uvsim.py``,
    ``UVEngine.make_visibility``).  Both build the baseline vector as
    ``antenna2 - antenna1`` in ENU, and the fixture array is coplanar in Up, so
    ``w`` is exactly zero and the two differ by conjugation of the exponent
    alone; in correlation order that is
    ``[XX, XY, YX, YY] -> conj([XX, YX, XY, YY])``.
    """
    import numpy as np

    return np.conj(cube[..., [0, 2, 1, 3]])


def _convention_mappings() -> list[dict[str, Any]]:
    """Return the frozen, sorted convention mapping rows."""
    rows = [
        {
            "radiosim_convention": "beam_normalization",
            "reference_convention": (
                "both codes read the same peak-normalized full-efield BeamFITS "
                "and neither renormalizes it: RadioSim's "
                "'uvbeam_peak_common_v1' accepted subset requires a stored "
                "full-grid unit peak at every intrinsic frequency and never "
                "calls UVBeam.peak_normalize, and pyuvsim reads the same bytes "
                "through UVBeam.from_file"
            ),
            "equivalent": True,
        },
        {
            "radiosim_convention": "east_x_orientation",
            "reference_convention": (
                "both codes report linear receptors as (X=east, Y=north) after "
                "SCI-006, so raw Q and U are compared directly with no frame "
                "compensation"
            ),
            "equivalent": True,
        },
        {
            "radiosim_convention": "fringe_sign",
            "reference_convention": (
                "RadioSim evaluates exp(-2j*pi*(u*l + v*m + w*(n-1))) and "
                "pyuvsim exp(+2j*pi*(u*l + v*m + w*n)); on the coplanar fixture "
                "array w is exactly zero, so the comparison applies the derived "
                "per-baseline Hermitian conjugate "
                "[XX, XY, YX, YY] -> conj([XX, YX, XY, YY])"
            ),
            "equivalent": False,
        },
        {
            "radiosim_convention": "stokes_to_coherency_factor",
            "reference_convention": (
                "RadioSim uses B = (1/2)[[I+Q, U+iV], [U-iV, I-Q]] and "
                "pyradiosky the mirror image 0.5*[[I+Q, U-iV], [U+iV, I-Q]]; "
                "both carry the same 1/2 factor, and the comparison drives the "
                "reference with the sign-mapped Stokes V read from the same sky "
                "bytes"
            ),
            "equivalent": False,
        },
    ]
    return sorted(rows, key=lambda row: row["radiosim_convention"])


def run_comparison(directory: Path) -> ComparisonResult:
    """Run the full Stage-3 comparison once below ``directory``.

    ``directory`` must be a writable location outside repository root; the
    fixture, the RadioSim working directory and nothing else are written there.
    """
    import numpy as np
    from astropy.utils import iers

    from radiosim.core.polarization_basis import (
        CORRELATION_LABELS,
        basis_for_correlations,
    )

    # The mirrored transcription this module's validate path relies on must
    # equal the production table exactly; the generate path is where that can
    # be checked, because it may import radiosim.
    if {
        basis: tuple(labels) for basis, labels in CORRELATION_LABELS.items()
    } != CORRELATION_LABELS_BY_BASIS:
        raise CrossValidationError(
            "the mirrored CORRELATION_LABELS_BY_BASIS transcription has drifted "
            "from src/radiosim/core/polarization_basis.py"
        )

    inputs = build_inputs(directory)
    iers_table = iers.IERS_A.open(iers.IERS_A_FILE)
    with (
        iers.conf.set_temp("auto_download", False),
        iers.earth_orientation_table.set(iers_table),
    ):
        result = _run_radiosim(inputs, directory / "radiosim-run")
        reference = _apply_fringe_mapping(
            _reference_cube(result, inputs, _reference_sky(inputs))
        )

    ours = np.asarray(result.visibilities)
    if ours.shape != reference.shape:
        raise CrossValidationError(
            f"cube shape mismatch: RadioSim {ours.shape}, reference {reference.shape}"
        )
    correlations = tuple(result.correlations)
    output_basis = basis_for_correlations(correlations)

    residuals: dict[str, tuple[float, float, float]] = {}
    disagreements: list[str] = []
    for index, label in enumerate(correlations):
        reference_max = float(np.max(np.abs(reference[..., index])))
        absolute = float(np.max(np.abs(ours[..., index] - reference[..., index])))
        if reference_max <= 0.0:
            raise CrossValidationError(
                f"correlation {label} has a zero reference scale; the comparison "
                "would be vacuous"
            )
        relative = absolute / reference_max
        residuals[label] = (absolute + 0.0, relative + 0.0, reference_max + 0.0)
        if relative > PER_CORRELATION_RELATIVE_BOUND:
            disagreements.append(
                f"{label}: max_rel_residual exceeds the declared per-correlation "
                f"bound {PER_CORRELATION_RELATIVE_BOUND!r}"
            )

    ours_trace = ours[..., 0] + ours[..., 3]
    reference_trace = reference[..., 0] + reference[..., 3]
    trace_scale = float(np.max(np.abs(reference_trace)))
    if trace_scale <= 0.0:
        raise CrossValidationError("the reference total-intensity scale is zero")
    trace_relative = float(np.max(np.abs(ours_trace - reference_trace))) / trace_scale
    if trace_relative > TRACE_RELATIVE_BOUND:
        disagreements.append(
            "total_intensity: the frame-invariant XX + YY residual exceeds the "
            f"declared bound {TRACE_RELATIVE_BOUND!r}"
        )

    scale = float(np.max(np.abs(reference)))
    control = float(np.max(np.abs(ours - _apply_fringe_mapping(reference)))) / scale

    return ComparisonResult(
        output_basis=output_basis,
        correlations=correlations,
        residuals=residuals,
        trace_relative=trace_relative,
        control_relative_without_fringe_mapping=control,
        reference_scale=scale,
        open_disagreements=tuple(sorted(set(disagreements))),
        input_hashes=inputs.hashes,
        shape=tuple(int(value) for value in ours.shape),
    )


# --------------------------------------------------------------------------
# Record construction and the generate form.
# --------------------------------------------------------------------------


def _package_version(name: str) -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise CrossValidationError(f"{name} is not installed") from exc


def _assert_reference_versions() -> None:
    for name, expected in (
        (REFERENCE_PACKAGE, REFERENCE_VERSION),
        ("pyuvdata", PYUVDATA_VERSION),
    ):
        measured = _package_version(name)
        if measured != expected:
            raise CrossValidationError(
                f"{name} {measured} is not the pinned reference {expected}"
            )


def build_record(
    *,
    approved_source_sha: str,
    comparison: ComparisonResult,
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the artifact in the frozen Section 8.1 key order."""
    generated_at = _utc_now()
    residual_rows = [
        {
            "correlation": label,
            "max_abs_residual": comparison.residuals[label][0],
            "max_rel_residual": comparison.residuals[label][1],
            "reference_max_abs": comparison.residuals[label][2],
        }
        # Lexical sort by `correlation`; see the `OUTPUT_BASIS` comment.
        for label in sorted(comparison.residuals)
    ]
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "source_sha": approved_source_sha,
        "target_path": f"{TARGET_DIRECTORY}/{generated_at[:10]}-{TARGET_SUFFIX}",
        "gating": False,
        "reference_package": REFERENCE_PACKAGE,
        "reference_version": REFERENCE_VERSION,
        "pyuvdata_version": PYUVDATA_VERSION,
        "pyradiosky_version": _package_version("pyradiosky"),
        "astropy_version": _package_version("astropy"),
        "radiosim_version": _package_version("radiosim"),
        "output_basis": comparison.output_basis,
        "input_hashes": [
            {"name": name, "sha256": digest} for name, digest in comparison.input_hashes
        ],
        "convention_mappings": _convention_mappings(),
        "correlation_residuals": residual_rows,
        "open_disagreements": list(comparison.open_disagreements),
        "commands": list(commands),
    }
    return record


def generate_artifact(
    *, approved_source_sha: str, output: Path
) -> tuple[dict[str, Any], str]:
    """Measure from an approved clean source and write the artifact atomically."""
    _validate_sha(approved_source_sha, length=40, label="approved source SHA")
    _assert_generation_environment()
    resolved_output = _resolve_output(output)
    _assert_reference_versions()

    commands = _assert_clean_source(approved_source_sha)
    with tempfile.TemporaryDirectory(prefix="sci005-stage3-crossval-") as raw:
        comparison = run_comparison(Path(raw))
    commands.extend(_assert_clean_source(approved_source_sha))

    if resolved_output.exists() or resolved_output.is_symlink():
        raise CrossValidationError(
            f"--output appeared during measurement: {resolved_output}"
        )

    record = build_record(
        approved_source_sha=approved_source_sha,
        comparison=comparison,
        commands=commands,
    )
    serialized = _serialize(record)
    # Refuse to publish anything the standalone validator would reject.
    validate_record(record, approved_source_sha=approved_source_sha)
    artifact_sha256 = _sha256_bytes(serialized)
    _atomic_write_new(resolved_output, serialized)
    return record, artifact_sha256


# --------------------------------------------------------------------------
# Command line.
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or validate the SCI-005 Stage-3 cross-validation artifact. "
            "Neither form gates. The comparison licenses only the sentence "
            "'compared against pyuvsim for the named fixture, with the recorded "
            "agreements and open disagreements'."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help=(
            "measure the comparison in the optional crossval environment and "
            "write the artifact to an absolute out-of-repository path"
        ),
    )
    generate.add_argument("--approved-source-sha", required=True)
    generate.add_argument("--output", required=True, type=Path)

    validate = subparsers.add_parser(
        "validate",
        help=(
            "authenticate pinned artifact bytes and the complete frozen schema; "
            "standard library only, so it runs in the standard environment"
        ),
    )
    validate.add_argument("--approved-source-sha", required=True)
    validate.add_argument("--artifact-sha256", required=True)
    validate.add_argument("--input", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "generate":
            record, artifact_sha256 = generate_artifact(
                approved_source_sha=args.approved_source_sha,
                output=args.output,
            )
        else:
            record = validate_artifact(
                args.input,
                approved_source_sha=args.approved_source_sha,
                artifact_sha256=args.artifact_sha256,
            )
            artifact_sha256 = args.artifact_sha256
    except (CrossValidationError, AssertionError, OSError) as exc:
        print(f"SCI-005 Stage-3 cross-validation error: {exc}", file=sys.stderr)
        return 1
    summary = {
        "artifact_sha256": artifact_sha256,
        "gating": record["gating"],
        "open_disagreements": record["open_disagreements"],
        "passed": True,
        "source_sha": record["source_sha"],
        "target_path": record["target_path"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
