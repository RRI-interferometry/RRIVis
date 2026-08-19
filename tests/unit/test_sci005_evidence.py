"""Strict authentication of the SCI-005 retained stage-evidence artifacts.

``docs/development/sci005_beam_physics_plan.md`` Section 8.1 freezes the
evidence contract and Section 7.5 freezes its successor authority: the
generator, schema and this validator land in the stage's ``S`` commit, and at
that point the official evidence JSON is **absent** and the target stage's two
approved constants are the literal ``None``. The ``Ei`` successor adds only the
artifact and flips exactly those two constants to the exact lower-case 40- and
64-hexadecimal literals. It may not change validator logic, any other test byte
or path, the schemas, production, documentation, or an earlier artifact.

Importing this module loads only the Python standard library plus ``pytest``.
That is deliberate, and it follows ``tools/wp7_perf001_cpu_evidence.py``: an
acceptance-critical validator must not depend on a package that is merely
transitively present, because a lock update could drop it and silently turn a
hard authentication into an import error. ``docs/development/
sci005_stage1_evidence.schema.json`` remains the normative transcription of
Section 8.1; :func:`validate_stage1_evidence` below enforces the same
structure, types, key order, hexadecimal encodings and cross-field rules in its
own code, and :func:`test_the_schema_transcription_and_the_validator_agree`
holds the two to the same key sets.

Every rejection class has its own test, so a weakened check is visible as a
failing test rather than as an artifact that quietly passes.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Section 7.5's approved-digest constants. ``Ei`` replaces exactly these two
#: ``None`` literals and nothing else in this module.
# fmt: off
# The flipped 64-hex literals exceed the format line limit by design;
# Section 7.5's substitution is byte-exact and must not be rewrapped.
APPROVED_STAGE1_SOURCE_SHA: str | None = "881b1a963b4f3b250b38989335c2ee0ea2a491bd"
APPROVED_STAGE1_EVIDENCE_ARTIFACT_SHA256: str | None = "4a0c8e96c275bad2bfd84535940a075b4c219c39b705ddada23de16ded85a2c4"
APPROVED_STAGE2_SOURCE_SHA: str | None = None
APPROVED_STAGE2_EVIDENCE_ARTIFACT_SHA256: str | None = None
APPROVED_STAGE3_SOURCE_SHA: str | None = None
APPROVED_STAGE3_EVIDENCE_ARTIFACT_SHA256: str | None = None
# fmt: on

GENERATOR = "tools/sci005_stage_evidence.py"
GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
TIMESTAMP = re.compile(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")

STAGE_CONSTANTS: dict[int, tuple[str | None, str | None]] = {
    1: (APPROVED_STAGE1_SOURCE_SHA, APPROVED_STAGE1_EVIDENCE_ARTIFACT_SHA256),
    2: (APPROVED_STAGE2_SOURCE_SHA, APPROVED_STAGE2_EVIDENCE_ARTIFACT_SHA256),
    3: (APPROVED_STAGE3_SOURCE_SHA, APPROVED_STAGE3_EVIDENCE_ARTIFACT_SHA256),
}

#: Section 8.1's exact Stage-1 top-level key sequence.
STAGE1_KEYS: tuple[str, ...] = (
    "schema_version",
    "stage",
    "status",
    "generated_at_utc",
    "design_sha",
    "red_test_sha",
    "source_sha",
    "evidence_sha",
    "working_tree_clean",
    "radiosim_version",
    "python_version",
    "platform",
    "machine",
    "pixi_environment",
    "pixi_lock_sha256",
    "scientific_conventions",
    "config_cases",
    "analytic_invariants",
    "rejection_probes",
    "backend_parity",
    "solver_cases",
    "output_cases",
    "fingerprint_diff",
    "commands",
    "artifacts",
    "limitations",
    "claims_not_licensed",
    "pupil_profiles",
    "support_masks",
    "ruze_power_diagnostics",
)

SCIENTIFIC_CONVENTIONS: dict[str, str] = {
    "pupil_profile_set": "radiosim.circular_stage1_pupil_profiles.v1",
    "aperture_normalization": "unmodified_ideal_aperture_v1",
    "aperture_axes": "north_east_azimuth_north_through_east_v1",
    "support_mask": "radiosim.central_disk_outward_half_strip_ne.v1",
    "zernike_surface": "radiosim.real_unit_rms_disk_surface_height.v1",
    "aperture_method": "boundary_fitted_polar_gauss_legendre_v1",
    "ruze_covariance": "gaussian_one_over_e_surface_covariance_v1",
    "ruze_method": "poisson_paired_pupil_separation_v1",
}

#: Section 3.4.2's declared convergence field order.
CONVERGENCE_KEYS: tuple[str, ...] = (
    "real_dtype",
    "complex_dtype",
    "poisson_mu",
    "poisson_first_order",
    "poisson_last_order",
    "poisson_term_count",
    "poisson_lower_omitted_mass",
    "poisson_upper_omitted_mass",
    "poisson_total_omitted_mass",
    "poisson_retained_weight_sum",
    "separation_cut_m",
    "separation_omitted_bound",
    "separation_radial_order",
    "separation_angular_order_max",
    "separation_node_count",
    "separation_evaluation_count",
    "separation_penultimate_max_abs_delta",
    "separation_final_max_abs_delta",
    "separation_imaginary_max_abs_residual",
    "separation_topology_sha256",
    "aperture_method",
    "aperture_partition_count",
    "aperture_topology_breakpoint_count",
    "aperture_topology_sha256",
    "aperture_refinement_count",
    "aperture_max_node_count",
    "aperture_penultimate_max_abs_delta",
    "aperture_final_max_abs_delta",
    "aperture_q_max",
    "surface_phase_kappa",
    "surface_radial_derivative_bound",
    "surface_angular_derivative_bound",
    "fhat_evaluation_count",
    "phase_product_count",
    "batch_size",
    "atol",
    "rtol",
    "estimated_peak_bytes",
    "maximum_abs_e_deterministic",
    "minimum_scattered_power",
    "maximum_total_power",
    "returned_balance_max_abs_residual",
)

#: Section 8.1 classifies exactly these convergence keys as exact integers.
CONVERGENCE_INTEGERS: frozenset[str] = frozenset(
    {
        "poisson_first_order",
        "poisson_last_order",
        "poisson_term_count",
        "separation_radial_order",
        "separation_angular_order_max",
        "separation_node_count",
        "separation_evaluation_count",
        "aperture_partition_count",
        "aperture_topology_breakpoint_count",
        "aperture_refinement_count",
        "aperture_max_node_count",
        "fhat_evaluation_count",
        "phase_product_count",
        "batch_size",
        "estimated_peak_bytes",
    }
)
CONVERGENCE_STRINGS: frozenset[str] = frozenset(
    {
        "real_dtype",
        "complex_dtype",
        "aperture_method",
        "separation_topology_sha256",
        "aperture_topology_sha256",
    }
)

LIMIT_ORACLE_KINDS: frozenset[str] = frozenset(
    {
        "mu_first_order",
        "infinite_correlation_length",
        "asymmetric_phase",
        "entire_plane_shift",
        "gaussian_characteristic_function",
        "covariance_only_counterexample",
    }
)
NUMERIC_DTYPES: frozenset[str] = frozenset(
    {"float32", "float64", "float128", "complex64", "complex128", "complex256"}
)
EXTENDED_INVARIANTS: frozenset[str] = frozenset(
    {"extended_precision_unmodified_profile", "extended_precision_mask_plus_zernike"}
)
REQUIRED_CLAIMS: tuple[str, ...] = (
    "SCI-005 Stage-1 acceptance",
    "SCI-005 Stages 2 and 3",
    "SCI-005 whole-row closure",
    "a deterministic Ruze Jones or error voltage",
)


class EvidenceSchemaError(AssertionError):
    """One retained-evidence authentication failure."""


# --- stdlib primitives --------------------------------------------------------


def _fail(path: str, detail: str) -> None:
    raise EvidenceSchemaError(f"{path}: {detail}")


def _mapping(value: Any, path: str, keys: tuple[str, ...]) -> dict[str, Any]:
    """Require an object whose keys are exactly ``keys``, in that order."""
    if not isinstance(value, dict):
        _fail(path, f"expected an object, observed {type(value).__name__}")
    observed = tuple(value)
    if observed != keys:
        missing = [key for key in keys if key not in observed]
        unknown = [key for key in observed if key not in keys]
        if missing or unknown:
            _fail(path, f"missing {missing}, unknown {unknown}")
        _fail(path, f"keys are not in the declared order: {list(observed)}")
    return value


def _string(
    value: Any,
    path: str,
    *,
    pattern: re.Pattern[str] | None = None,
    allowed: frozenset[str] | None = None,
    const: str | None = None,
) -> str:
    if not isinstance(value, str) or isinstance(value, bool):
        _fail(path, f"expected a string, observed {type(value).__name__}")
    if const is not None and value != const:
        _fail(path, f"expected the literal {const!r}, observed {value!r}")
    if const is None and not value:
        _fail(path, "expected a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail(path, f"{value!r} does not match {pattern.pattern}")
    if allowed is not None and value not in allowed:
        _fail(path, f"{value!r} is not one of {sorted(allowed)}")
    return value


def _boolean(value: Any, path: str, *, const: bool | None = None) -> bool:
    if not isinstance(value, bool):
        _fail(path, f"expected a boolean, observed {type(value).__name__}")
    if const is not None and value is not const:
        _fail(path, f"expected {const!r}")
    return value


def _number(value: Any, path: str, *, minimum: float | None = 0.0) -> float:
    """Require a finite, non-boolean JSON number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"expected a number, observed {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        _fail(path, "expected a finite number")
    if minimum is not None and numeric < minimum:
        _fail(path, f"expected >= {minimum}, observed {numeric!r}")
    return numeric


def _integer(value: Any, path: str, *, minimum: int | None = 0) -> int:
    """Require a non-boolean JSON integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, f"expected an integer, observed {type(value).__name__}")
    if minimum is not None and value < minimum:
        _fail(path, f"expected >= {minimum}, observed {value!r}")
    return value


def _array(value: Any, path: str, *, minimum_length: int = 0) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, f"expected an array, observed {type(value).__name__}")
    if len(value) < minimum_length:
        _fail(path, f"expected at least {minimum_length} items")
    return value


def _canonical_path(value: Any, path: str) -> str:
    """Require Section 8.1's ``canonical_path`` encoding."""
    text = _string(value, path)
    if text.startswith("/") or "\\" in text or "\x00" in text:
        _fail(path, f"{text!r} is not a repository-relative POSIX path")
    if any(part in {"", ".", ".."} for part in text.split("/")):
        _fail(path, f"{text!r} has an empty or relative component")
    return text


def _sorted_unique(values: list[Any], path: str) -> None:
    if values != sorted(values):
        _fail(path, "array is not sorted")
    if len(set(values)) != len(values):
        _fail(path, "array contains a duplicate")


def _rows_sorted_by(rows: list[dict[str, Any]], key: str, path: str) -> None:
    _sorted_unique([row[key] for row in rows], f"{path}[*].{key}")


def _nullable(value: Any, path: str, checker: Any) -> Any:
    return None if value is None else checker(value, path)


# --- Section 8.1 shared projections -------------------------------------------


def _numeric_projection(value: Any, path: str) -> dict[str, Any]:
    row = _mapping(
        value,
        path,
        ("dtype", "shape", "c_order_sha256", "minimum_abs", "maximum_abs"),
    )
    _string(row["dtype"], f"{path}.dtype", allowed=NUMERIC_DTYPES)
    shape = _array(row["shape"], f"{path}.shape", minimum_length=1)
    product = 1
    for index, extent in enumerate(shape):
        product *= _integer(extent, f"{path}.shape[{index}]", minimum=0)
    if product <= 0:
        _fail(f"{path}.shape", "the shape product must be positive")
    _string(row["c_order_sha256"], f"{path}.c_order_sha256", pattern=SHA256)
    _number(row["minimum_abs"], f"{path}.minimum_abs")
    _number(row["maximum_abs"], f"{path}.maximum_abs")
    return row


def _array_projection(value: Any, path: str) -> dict[str, Any]:
    row = _mapping(
        value, path, ("dtype", "shape", "c_order_sha256", "minimum", "maximum")
    )
    _string(row["dtype"], f"{path}.dtype", allowed=frozenset({"float32", "float64"}))
    shape = _array(row["shape"], f"{path}.shape", minimum_length=1)
    if len(shape) != 1:
        _fail(f"{path}.shape", "an array_projection has exactly one dimension")
    _integer(shape[0], f"{path}.shape[0]", minimum=1)
    _string(row["c_order_sha256"], f"{path}.c_order_sha256", pattern=SHA256)
    _number(row["minimum"], f"{path}.minimum", minimum=None)
    _number(row["maximum"], f"{path}.maximum", minimum=None)
    return row


def _antenna_projection(value: Any, path: str) -> dict[str, Any]:
    row = _mapping(value, path, ("number", "name"))
    _integer(row["number"], f"{path}.number", minimum=None)
    _string(row["name"], f"{path}.name")
    return row


def _command_row(value: Any, path: str) -> dict[str, Any]:
    row = _mapping(
        value,
        path,
        (
            "argv",
            "cwd",
            "pixi_environment",
            "started_at_utc",
            "duration_seconds",
            "exit_code",
            "stdout_sha256",
            "stderr_sha256",
        ),
    )
    argv = _array(row["argv"], f"{path}.argv", minimum_length=1)
    for index, item in enumerate(argv):
        _string(item, f"{path}.argv[{index}]")
    _string(row["cwd"], f"{path}.cwd", const=".")
    _string(row["pixi_environment"], f"{path}.pixi_environment")
    _string(row["started_at_utc"], f"{path}.started_at_utc", pattern=TIMESTAMP)
    _number(row["duration_seconds"], f"{path}.duration_seconds")
    _integer(row["exit_code"], f"{path}.exit_code", minimum=None)
    if row["exit_code"] != 0:
        _fail(f"{path}.exit_code", "candidate evidence requires a zero exit code")
    _string(row["stdout_sha256"], f"{path}.stdout_sha256", pattern=SHA256)
    _string(row["stderr_sha256"], f"{path}.stderr_sha256", pattern=SHA256)
    return row


# --- Section 8.1 evidence rows ------------------------------------------------


def _config_case(value: Any, path: str) -> None:
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "test_node_id",
            "input_sha256",
            "expected_outcome",
            "observed_outcome",
            "resolved_scientific_sha256",
            "exception_type",
            "issue_code",
            "exact_message",
            "passed",
        ),
    )
    outcomes = frozenset({"accepted", "rejected"})
    _string(row["case_id"], f"{path}.case_id")
    _string(row["test_node_id"], f"{path}.test_node_id")
    _string(row["input_sha256"], f"{path}.input_sha256", pattern=SHA256)
    expected = _string(
        row["expected_outcome"], f"{path}.expected_outcome", allowed=outcomes
    )
    observed = _string(
        row["observed_outcome"], f"{path}.observed_outcome", allowed=outcomes
    )
    if expected != observed:
        _fail(path, "expected and observed outcomes must agree")
    _boolean(row["passed"], f"{path}.passed", const=True)
    errors = ("exception_type", "issue_code", "exact_message")
    if observed == "accepted":
        _string(
            row["resolved_scientific_sha256"],
            f"{path}.resolved_scientific_sha256",
            pattern=SHA256,
        )
        for key in errors:
            if row[key] is not None:
                _fail(f"{path}.{key}", "an accepted observation has null error fields")
    else:
        if row["resolved_scientific_sha256"] is not None:
            _fail(
                f"{path}.resolved_scientific_sha256",
                "a rejected observation has a null resolution",
            )
        for key in errors:
            _string(row[key], f"{path}.{key}")


def _analytic_invariant(value: Any, path: str) -> dict[str, Any]:
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "invariant_id",
            "backend",
            "test_node_id",
            "input_manifest_sha256",
            "expected",
            "observed",
            "max_abs_residual",
            "max_rel_residual",
            "atol",
            "rtol",
            "passed",
        ),
    )
    _string(row["case_id"], f"{path}.case_id")
    _string(row["invariant_id"], f"{path}.invariant_id")
    _string(
        row["backend"],
        f"{path}.backend",
        allowed=frozenset({"numpy", "jax", "dask", "independent_oracle"}),
    )
    _string(row["test_node_id"], f"{path}.test_node_id")
    _string(
        row["input_manifest_sha256"], f"{path}.input_manifest_sha256", pattern=SHA256
    )
    expected = _numeric_projection(row["expected"], f"{path}.expected")
    observed = _numeric_projection(row["observed"], f"{path}.observed")
    if expected["dtype"] != observed["dtype"] or expected["shape"] != observed["shape"]:
        _fail(path, "expected and observed projections need identical dtype and shape")
    for key in ("max_abs_residual", "max_rel_residual", "atol", "rtol"):
        _number(row[key], f"{path}.{key}")
    _boolean(row["passed"], f"{path}.passed", const=True)
    return row


def _rejection_probe(value: Any, path: str) -> None:
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "config_path",
            "exception_type",
            "issue_code",
            "exact_message",
            "test_node_id",
            "input_sha256",
            "passed",
        ),
    )
    for key in (
        "case_id",
        "config_path",
        "exception_type",
        "issue_code",
        "exact_message",
        "test_node_id",
    ):
        _string(row[key], f"{path}.{key}")
    _string(row["input_sha256"], f"{path}.input_sha256", pattern=SHA256)
    _boolean(row["passed"], f"{path}.passed", const=True)


def _backend_parity(value: Any, path: str) -> dict[str, Any]:
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "backend",
            "actual_device",
            "real_dtype",
            "complex_dtype",
            "input_sha256",
            "reference_result_sha256",
            "observed_result_sha256",
            "max_abs_difference",
            "max_rel_difference",
            "atol",
            "rtol",
            "passed",
        ),
    )
    _string(row["case_id"], f"{path}.case_id")
    _string(
        row["backend"], f"{path}.backend", allowed=frozenset({"numpy", "jax", "dask"})
    )
    _string(row["actual_device"], f"{path}.actual_device")
    pair = (row["real_dtype"], row["complex_dtype"])
    if pair not in {("float32", "complex64"), ("float64", "complex128")}:
        _fail(path, f"{pair} is not one of the two Section 3.4.2 dtype pairs")
    for key in ("input_sha256", "reference_result_sha256", "observed_result_sha256"):
        _string(row[key], f"{path}.{key}", pattern=SHA256)
    for key in ("max_abs_difference", "max_rel_difference", "atol", "rtol"):
        _number(row[key], f"{path}.{key}")
    _boolean(row["passed"], f"{path}.passed", const=True)
    return row


def _solver_case(value: Any, path: str) -> None:
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "effect",
            "test_node_id",
            "input_sha256",
            "jones_sha256",
            "visibility_sha256",
            "diagnostic_sha256",
            "jones_call_count",
            "visibility_changed_element_count",
            "visibility_change_expected",
            "passed",
        ),
    )
    effects = frozenset(
        {
            "blockage",
            "zernike",
            "ruze_coherent_voltage",
            "ruze_power_diagnostic_non_visibility",
        }
    )
    _string(row["case_id"], f"{path}.case_id")
    effect = _string(row["effect"], f"{path}.effect", allowed=effects)
    _string(row["test_node_id"], f"{path}.test_node_id")
    for key in ("input_sha256", "jones_sha256", "visibility_sha256"):
        _string(row[key], f"{path}.{key}", pattern=SHA256)
    calls = _integer(row["jones_call_count"], f"{path}.jones_call_count")
    changed = _integer(
        row["visibility_changed_element_count"],
        f"{path}.visibility_changed_element_count",
    )
    expected = _boolean(
        row["visibility_change_expected"], f"{path}.visibility_change_expected"
    )
    _boolean(row["passed"], f"{path}.passed", const=True)
    if effect == "ruze_power_diagnostic_non_visibility":
        _string(row["diagnostic_sha256"], f"{path}.diagnostic_sha256", pattern=SHA256)
        if calls or changed or expected:
            _fail(
                path,
                "a diagnostic-only row needs zero Jones calls, zero changed "
                "visibility elements, and a false change expectation",
            )
    elif row["diagnostic_sha256"] is not None:
        _fail(f"{path}.diagnostic_sha256", f"must be null for effect {effect!r}")


def _output_case(value: Any, path: str) -> None:
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "format",
            "writer_test_node_id",
            "reader_test_node_id",
            "artifact_sha256",
            "in_memory_sha256",
            "observed_projection_sha256",
            "roundtrip_max_abs_difference",
            "tolerance",
            "passed",
        ),
    )
    formats = frozenset(
        {
            "in_memory",
            "summary_json",
            "hdf5",
            "uvfits",
            "measurement_set",
            "reader_projection",
        }
    )
    _string(row["case_id"], f"{path}.case_id")
    kind = _string(row["format"], f"{path}.format", allowed=formats)
    _string(row["writer_test_node_id"], f"{path}.writer_test_node_id")
    _string(row["in_memory_sha256"], f"{path}.in_memory_sha256", pattern=SHA256)
    _boolean(row["passed"], f"{path}.passed", const=True)
    nullable_keys = (
        "reader_test_node_id",
        "artifact_sha256",
        "roundtrip_max_abs_difference",
        "tolerance",
    )
    if kind == "in_memory":
        return
    for key in nullable_keys:
        if row[key] is None:
            _fail(f"{path}.{key}", "only an in_memory row may leave this null")


def _fingerprint_row(value: Any, path: str) -> dict[str, Any]:
    row = _mapping(
        value,
        path,
        (
            "environment",
            "workload",
            "old_scientific_sha256",
            "new_scientific_sha256",
            "old_raw_cube_sha256",
            "new_raw_cube_sha256",
            "changed_element_count",
            "maximum_delta",
            "change_expected",
            "test_node_id",
            "passed",
        ),
    )
    for key in ("environment", "workload", "test_node_id"):
        _string(row[key], f"{path}.{key}")
    for key in (
        "old_scientific_sha256",
        "new_scientific_sha256",
        "old_raw_cube_sha256",
        "new_raw_cube_sha256",
    ):
        _string(row[key], f"{path}.{key}", pattern=SHA256)
    changed = _integer(row["changed_element_count"], f"{path}.changed_element_count")
    _number(row["maximum_delta"], f"{path}.maximum_delta")
    expected = _boolean(row["change_expected"], f"{path}.change_expected")
    _boolean(row["passed"], f"{path}.passed", const=True)
    identical = row["old_scientific_sha256"] == row["new_scientific_sha256"]
    if expected and identical:
        _fail(path, "a changed workload cannot keep its scientific digest")
    if not expected and (not identical or changed):
        _fail(path, "an unchanged workload must be byte-identical")
    return row


def _artifact_row(value: Any, path: str) -> dict[str, Any]:
    row = _mapping(value, path, ("path", "sha256", "media_type", "role"))
    _canonical_path(row["path"], f"{path}.path")
    _string(row["sha256"], f"{path}.sha256", pattern=SHA256)
    _string(row["media_type"], f"{path}.media_type")
    _string(
        row["role"],
        f"{path}.role",
        allowed=frozenset(
            {"schema", "command_log", "output", "fingerprint", "auxiliary"}
        ),
    )
    return row


def _pupil_profile(value: Any, path: str) -> None:
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "model_kind",
            "taper_kind",
            "edge_taper_db",
            "mixture_weight",
            "profile_convention",
            "hankel_convention",
            "outcome",
            "exception_type",
            "issue_code",
            "test_node_id",
            "max_abs_residual",
            "tolerance",
        ),
    )
    _string(row["case_id"], f"{path}.case_id")
    _string(
        row["model_kind"],
        f"{path}.model_kind",
        allowed=frozenset(
            {
                "circular_aperture",
                "analytical_illumination",
                "numerical_illumination",
                "rectangular_aperture",
                "elliptical_aperture",
                "fits",
            }
        ),
    )
    _nullable(
        row["taper_kind"],
        f"{path}.taper_kind",
        lambda item, where: _string(
            item,
            where,
            allowed=frozenset(
                {"uniform", "parabolic", "parabolic_squared", "gaussian", "cosine"}
            ),
        ),
    )
    _string(row["test_node_id"], f"{path}.test_node_id")
    outcome = _string(
        row["outcome"], f"{path}.outcome", allowed=frozenset({"accepted", "rejected"})
    )
    accepted_keys = (
        "profile_convention",
        "hankel_convention",
        "max_abs_residual",
        "tolerance",
    )
    rejected_keys = ("exception_type", "issue_code")
    if outcome == "accepted":
        for key in accepted_keys:
            if row[key] is None:
                _fail(f"{path}.{key}", "an accepted profile row requires this value")
        for key in rejected_keys:
            if row[key] is not None:
                _fail(f"{path}.{key}", "an accepted profile row has null error fields")
    else:
        for key in accepted_keys:
            if row[key] is not None:
                _fail(f"{path}.{key}", "a rejected profile row leaves this null")
        for key in rejected_keys:
            _string(row[key], f"{path}.{key}")


def _support_mask(value: Any, path: str) -> None:
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "diameter_m",
            "central_diameter_ratio",
            "legs",
            "probes",
            "union_control_id",
            "antipodal_control_id",
            "topology_sha256",
            "test_node_id",
            "passed",
        ),
    )
    for key in ("case_id", "union_control_id", "antipodal_control_id", "test_node_id"):
        _string(row[key], f"{path}.{key}")
    _number(row["diameter_m"], f"{path}.diameter_m")
    _number(row["central_diameter_ratio"], f"{path}.central_diameter_ratio")
    _string(row["topology_sha256"], f"{path}.topology_sha256", pattern=SHA256)
    for index, leg in enumerate(_array(row["legs"], f"{path}.legs")):
        entry = _mapping(
            leg, f"{path}.legs[{index}]", ("position_angle_deg", "width_m")
        )
        _number(
            entry["position_angle_deg"],
            f"{path}.legs[{index}].position_angle_deg",
            minimum=None,
        )
        _number(entry["width_m"], f"{path}.legs[{index}].width_m")
    probes = _array(row["probes"], f"{path}.probes", minimum_length=1)
    for index, probe in enumerate(probes):
        entry = _mapping(
            probe,
            f"{path}.probes[{index}]",
            ("north_m", "east_m", "expected_transmitting", "observed_transmitting"),
        )
        _number(entry["north_m"], f"{path}.probes[{index}].north_m", minimum=None)
        _number(entry["east_m"], f"{path}.probes[{index}].east_m", minimum=None)
        expected = _boolean(
            entry["expected_transmitting"],
            f"{path}.probes[{index}].expected_transmitting",
        )
        observed = _boolean(
            entry["observed_transmitting"],
            f"{path}.probes[{index}].observed_transmitting",
        )
        if expected is not observed:
            _fail(f"{path}.probes[{index}]", "expected and observed must agree")
    _boolean(row["passed"], f"{path}.passed", const=True)


def _ruze_convergence(value: Any, path: str) -> dict[str, Any]:
    row = _mapping(value, path, CONVERGENCE_KEYS)
    pair = (row["real_dtype"], row["complex_dtype"])
    if pair not in {("float32", "complex64"), ("float64", "complex128")}:
        _fail(path, f"{pair} is not one of the two Section 3.4.2 dtype pairs")
    _string(
        row["aperture_method"],
        f"{path}.aperture_method",
        const="boundary_fitted_polar_gauss_legendre_v1",
    )
    for key in ("separation_topology_sha256", "aperture_topology_sha256"):
        _string(row[key], f"{path}.{key}", pattern=SHA256)
    for key in CONVERGENCE_KEYS:
        if key in CONVERGENCE_STRINGS:
            continue
        if key in CONVERGENCE_INTEGERS:
            _integer(row[key], f"{path}.{key}")
        else:
            _number(row[key], f"{path}.{key}")

    first = row["poisson_first_order"]
    last = row["poisson_last_order"]
    count = row["poisson_term_count"]
    zero_term = count == 0
    if zero_term:
        if (first, last) != (0, 0):
            _fail(path, "a zero-term Poisson case has the exact interval [0, 0]")
        zeros = (
            "poisson_retained_weight_sum",
            "separation_radial_order",
            "separation_angular_order_max",
            "separation_node_count",
            "separation_evaluation_count",
            "separation_penultimate_max_abs_delta",
            "separation_final_max_abs_delta",
            "separation_imaginary_max_abs_residual",
            "separation_cut_m",
            "separation_omitted_bound",
        )
        for key in zeros:
            if row[key] != 0:
                _fail(f"{path}.{key}", "must be exactly zero in the zero-term case")
        if row["poisson_lower_omitted_mass"] != 0.0:
            _fail(f"{path}.poisson_lower_omitted_mass", "must be zero")
        expected_mass = -math.expm1(-row["poisson_mu"])
        for key in ("poisson_upper_omitted_mass", "poisson_total_omitted_mass"):
            if row[key] != expected_mass:
                _fail(f"{path}.{key}", "must equal -expm1(-poisson_mu)")
    else:
        if last < first or count != last - first + 1:
            _fail(path, "term count must equal last - first + 1")
        if count > 256:
            _fail(path, "more than 256 retained Poisson terms is refused")
        if row["separation_radial_order"] <= 0:
            _fail(f"{path}.separation_radial_order", "must be positive")
        order = row["separation_angular_order_max"]
        if order < 16 or order & (order - 1):
            _fail(
                f"{path}.separation_angular_order_max",
                "must be a power of two not below sixteen",
            )
        if row["separation_omitted_bound"] > row["atol"] / 8.0:
            _fail(
                f"{path}.separation_omitted_bound", "exceeds the frozen atol/8 budget"
            )

    total = row["poisson_lower_omitted_mass"] + row["poisson_upper_omitted_mass"]
    if total != row["poisson_total_omitted_mass"]:
        _fail(path, "lower plus upper omitted mass must equal the total")
    if row["poisson_total_omitted_mass"] > row["atol"] / 8.0:
        _fail(f"{path}.poisson_total_omitted_mass", "exceeds the frozen atol/8 budget")
    caps = (
        ("aperture_max_node_count", 2**18),
        ("separation_evaluation_count", 2**20),
        ("phase_product_count", 2**34),
        ("estimated_peak_bytes", 8 * 2**30),
    )
    for key, cap in caps:
        if row[key] > cap:
            _fail(f"{path}.{key}", f"exceeds the frozen cap {cap}")
    if row["batch_size"] > 256:
        _fail(f"{path}.batch_size", "exceeds the frozen 256 batch cap")
    tolerance = row["atol"] + row["rtol"]
    if row["maximum_abs_e_deterministic"] > 1.0 + tolerance:
        _fail(f"{path}.maximum_abs_e_deterministic", "exceeds one beyond tolerance")
    if row["maximum_total_power"] > 1.0 + tolerance:
        _fail(f"{path}.maximum_total_power", "exceeds one beyond tolerance")
    if row["returned_balance_max_abs_residual"] != 0.0:
        _fail(
            f"{path}.returned_balance_max_abs_residual",
            "the returned balance is exact in the result dtype",
        )
    return row


def _ruze_row(value: Any, path: str) -> dict[str, Any]:
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "resolved_aperture_scientific_sha256",
            "diagnostic",
            "direct_pair_oracle",
            "limit_oracles",
            "test_node_ids",
        ),
    )
    _string(row["case_id"], f"{path}.case_id")
    _string(
        row["resolved_aperture_scientific_sha256"],
        f"{path}.resolved_aperture_scientific_sha256",
        pattern=SHA256,
    )
    diagnostic = _mapping(
        row["diagnostic"],
        f"{path}.diagnostic",
        (
            "schema_version",
            "method",
            "covariance_convention",
            "normalization_convention",
            "antenna_id",
            "frequency_hz",
            "time_mjd",
            "rms_surface_error_m",
            "correlation_length_m",
            "altitude_rad",
            "azimuth_rad",
            "coherent_main_power",
            "total_ensemble_power",
            "scattered_power",
            "convergence",
        ),
    )
    literals = {
        "schema_version": "radiosim.ruze_power_diagnostic.v1",
        "method": "poisson_paired_pupil_separation_v1",
        "covariance_convention": "gaussian_one_over_e_surface_covariance_v1",
        "normalization_convention": "unmodified_ideal_aperture_v1",
    }
    for key, literal in literals.items():
        _string(diagnostic[key], f"{path}.diagnostic.{key}", const=literal)
    _antenna_projection(diagnostic["antenna_id"], f"{path}.diagnostic.antenna_id")
    for key in ("frequency_hz", "rms_surface_error_m", "correlation_length_m"):
        if _number(diagnostic[key], f"{path}.diagnostic.{key}") <= 0.0:
            _fail(f"{path}.diagnostic.{key}", "must be positive")
    _number(diagnostic["time_mjd"], f"{path}.diagnostic.time_mjd", minimum=None)
    shapes: set[int] = set()
    for key in (
        "altitude_rad",
        "azimuth_rad",
        "coherent_main_power",
        "total_ensemble_power",
        "scattered_power",
    ):
        projection = _array_projection(diagnostic[key], f"{path}.diagnostic.{key}")
        shapes.add(projection["shape"][0])
    if len(shapes) != 1:
        _fail(
            f"{path}.diagnostic", "every direction-sized projection shares shape (S,)"
        )
    _ruze_convergence(diagnostic["convergence"], f"{path}.diagnostic.convergence")

    oracle = _mapping(
        row["direct_pair_oracle"],
        f"{path}.direct_pair_oracle",
        (
            "test_node_id",
            "aperture_node_count",
            "direction_count",
            "max_abs_residual",
            "tolerance",
        ),
    )
    _string(oracle["test_node_id"], f"{path}.direct_pair_oracle.test_node_id")
    for key in ("aperture_node_count", "direction_count"):
        _integer(oracle[key], f"{path}.direct_pair_oracle.{key}", minimum=1)
    for key in ("max_abs_residual", "tolerance"):
        _number(oracle[key], f"{path}.direct_pair_oracle.{key}")

    oracles = _array(row["limit_oracles"], f"{path}.limit_oracles", minimum_length=6)
    kinds: list[str] = []
    for index, item in enumerate(oracles):
        entry = _mapping(
            item,
            f"{path}.limit_oracles[{index}]",
            ("kind", "test_node_id", "input_sha256", "max_abs_residual", "tolerance"),
        )
        kinds.append(
            _string(
                entry["kind"],
                f"{path}.limit_oracles[{index}].kind",
                allowed=LIMIT_ORACLE_KINDS,
            )
        )
        _string(entry["test_node_id"], f"{path}.limit_oracles[{index}].test_node_id")
        _string(
            entry["input_sha256"],
            f"{path}.limit_oracles[{index}].input_sha256",
            pattern=SHA256,
        )
        for key in ("max_abs_residual", "tolerance"):
            _number(entry[key], f"{path}.limit_oracles[{index}].{key}")
    if set(kinds) != LIMIT_ORACLE_KINDS:
        _fail(f"{path}.limit_oracles", "every row contains each of the six kinds once")
    _sorted_unique(kinds, f"{path}.limit_oracles[*].kind")

    identifiers = _array(
        row["test_node_ids"], f"{path}.test_node_ids", minimum_length=1
    )
    for index, item in enumerate(identifiers):
        _string(item, f"{path}.test_node_ids[{index}]")
    _sorted_unique(identifiers, f"{path}.test_node_ids")
    return row


# --- the complete Stage-1 validator -------------------------------------------


def validate_stage1_evidence(document: Any) -> None:
    """Authenticate one Stage-1 evidence document against Section 8.1.

    Enforced entirely with the standard library: exact key sets and order, the
    frozen literals, the ``git_sha``/``sha256``/timestamp encodings, JSON number
    and integer distinctions that reject booleans, sorted-unique arrays, and
    every cross-field predicate Section 8.1 names.
    """
    root = _mapping(document, "$", STAGE1_KEYS)
    _string(
        root["schema_version"], "$.schema_version", const="radiosim.sci005.stage1.v1"
    )
    if root["stage"] != 1 or isinstance(root["stage"], bool):
        _fail("$.stage", "must be the integer 1")
    _string(root["status"], "$.status", const="candidate")
    _string(root["generated_at_utc"], "$.generated_at_utc", pattern=TIMESTAMP)
    for key in ("design_sha", "red_test_sha", "source_sha"):
        _string(root[key], f"$.{key}", pattern=GIT_SHA)
    if root["evidence_sha"] is not None:
        _fail(
            "$.evidence_sha", "must be JSON null; the file cannot contain its own SHA"
        )
    _boolean(root["working_tree_clean"], "$.working_tree_clean", const=True)
    for key in (
        "radiosim_version",
        "python_version",
        "platform",
        "machine",
        "pixi_environment",
    ):
        _string(root[key], f"$.{key}")
    _string(root["pixi_lock_sha256"], "$.pixi_lock_sha256", pattern=SHA256)

    conventions = _mapping(
        root["scientific_conventions"],
        "$.scientific_conventions",
        tuple(SCIENTIFIC_CONVENTIONS),
    )
    for key, literal in SCIENTIFIC_CONVENTIONS.items():
        _string(conventions[key], f"$.scientific_conventions.{key}", const=literal)

    simple = (
        ("config_cases", _config_case),
        ("rejection_probes", _rejection_probe),
        ("solver_cases", _solver_case),
        ("output_cases", _output_case),
        ("pupil_profiles", _pupil_profile),
        ("support_masks", _support_mask),
    )
    for key, checker in simple:
        rows = _array(root[key], f"$.{key}", minimum_length=1)
        for index, row in enumerate(rows):
            checker(row, f"$.{key}[{index}]")
        _rows_sorted_by(rows, "case_id", f"$.{key}")

    invariants = _array(
        root["analytic_invariants"], "$.analytic_invariants", minimum_length=1
    )
    parsed = [
        _analytic_invariant(row, f"$.analytic_invariants[{index}]")
        for index, row in enumerate(invariants)
    ]
    _rows_sorted_by(parsed, "case_id", "$.analytic_invariants")
    for invariant in EXTENDED_INVARIANTS:
        extended = [
            row
            for row in parsed
            if row["invariant_id"] == invariant and row["backend"] == "numpy"
        ]
        if len(extended) != 1:
            _fail(
                "$.analytic_invariants",
                f"exactly one numpy row is required for {invariant!r}",
            )
        if extended[0]["observed"]["dtype"] != "complex256":
            _fail(
                "$.analytic_invariants",
                f"{invariant!r} must retain a complex256 projection",
            )

    parity = _array(root["backend_parity"], "$.backend_parity", minimum_length=1)
    parsed_parity = [
        _backend_parity(row, f"$.backend_parity[{index}]")
        for index, row in enumerate(parity)
    ]
    keys = [(row["case_id"], row["backend"]) for row in parsed_parity]
    _sorted_unique(keys, "$.backend_parity")
    by_case: dict[str, set[str]] = {}
    for row in parsed_parity:
        by_case.setdefault(row["case_id"], set()).add(row["backend"])
    for case_id, backends in by_case.items():
        if backends != {"numpy", "jax", "dask"}:
            _fail("$.backend_parity", f"case {case_id!r} is missing a backend")

    fingerprints = _array(
        root["fingerprint_diff"], "$.fingerprint_diff", minimum_length=1
    )
    parsed_fingerprints = [
        _fingerprint_row(row, f"$.fingerprint_diff[{index}]")
        for index, row in enumerate(fingerprints)
    ]
    _sorted_unique(
        [(row["environment"], row["workload"]) for row in parsed_fingerprints],
        "$.fingerprint_diff",
    )
    expectations = {row["change_expected"] for row in parsed_fingerprints}
    if expectations != {True, False}:
        _fail(
            "$.fingerprint_diff",
            "both an enabled and a disabled/default control are required",
        )

    commands = _array(root["commands"], "$.commands", minimum_length=1)
    for index, row in enumerate(commands):
        _command_row(row, f"$.commands[{index}]")

    artifacts = _array(root["artifacts"], "$.artifacts", minimum_length=1)
    parsed_artifacts = [
        _artifact_row(row, f"$.artifacts[{index}]")
        for index, row in enumerate(artifacts)
    ]
    _rows_sorted_by(parsed_artifacts, "path", "$.artifacts")

    limitations = _array(root["limitations"], "$.limitations")
    for index, item in enumerate(limitations):
        _string(item, f"$.limitations[{index}]")
    _sorted_unique(limitations, "$.limitations")
    claims = _array(
        root["claims_not_licensed"], "$.claims_not_licensed", minimum_length=1
    )
    for index, item in enumerate(claims):
        _string(item, f"$.claims_not_licensed[{index}]")
    _sorted_unique(claims, "$.claims_not_licensed")
    for required in REQUIRED_CLAIMS:
        if required not in claims:
            _fail("$.claims_not_licensed", f"must name {required!r}")

    diagnostics = _array(
        root["ruze_power_diagnostics"], "$.ruze_power_diagnostics", minimum_length=1
    )
    parsed_diagnostics = [
        _ruze_row(row, f"$.ruze_power_diagnostics[{index}]")
        for index, row in enumerate(diagnostics)
    ]
    _rows_sorted_by(parsed_diagnostics, "case_id", "$.ruze_power_diagnostics")


# --- fixtures -----------------------------------------------------------------


def artifact_path(stage: int) -> Path:
    return REPOSITORY_ROOT / f"docs/development/sci005_stage{stage}_evidence.json"


def schema_path(stage: int) -> Path:
    return (
        REPOSITORY_ROOT / f"docs/development/sci005_stage{stage}_evidence.schema.json"
    )


def _projection(dtype: str = "complex128") -> dict[str, Any]:
    return {
        "dtype": dtype,
        "shape": [3],
        "c_order_sha256": "0" * 64,
        "minimum_abs": 0.0,
        "maximum_abs": 1.0,
    }


def _direction_projection() -> dict[str, Any]:
    return {
        "dtype": "float64",
        "shape": [3],
        "c_order_sha256": "0" * 64,
        "minimum": 0.0,
        "maximum": 1.0,
    }


def _convergence() -> dict[str, Any]:
    values: dict[str, Any] = {
        "real_dtype": "float64",
        "complex_dtype": "complex128",
        "poisson_mu": 0.007028106169663435,
        "poisson_first_order": 1,
        "poisson_last_order": 5,
        "poisson_term_count": 5,
        "poisson_lower_omitted_mass": 0.0,
        "poisson_upper_omitted_mass": 1.674e-16,
        "poisson_total_omitted_mass": 1.674e-16,
        "poisson_retained_weight_sum": 0.007003,
        "separation_cut_m": 10.9014609,
        "separation_omitted_bound": 1.24e-13,
        "separation_radial_order": 548,
        "separation_angular_order_max": 512,
        "separation_node_count": 169408,
        "separation_evaluation_count": 392256,
        "separation_penultimate_max_abs_delta": 1.0e-15,
        "separation_final_max_abs_delta": 1.0e-16,
        "separation_imaginary_max_abs_residual": 2.116e-18,
        "separation_topology_sha256": "0" * 64,
        "aperture_method": "boundary_fitted_polar_gauss_legendre_v1",
        "aperture_partition_count": 1,
        "aperture_topology_breakpoint_count": 2,
        "aperture_topology_sha256": "0" * 64,
        "aperture_refinement_count": 8,
        "aperture_max_node_count": 158400,
        "aperture_penultimate_max_abs_delta": 1.0e-14,
        "aperture_final_max_abs_delta": 1.0e-15,
        "aperture_q_max": 14.66,
        "surface_phase_kappa": 4.191690043903364,
        "surface_radial_derivative_bound": 0.1386,
        "surface_angular_derivative_bound": 0.0,
        "fhat_evaluation_count": 27,
        "phase_product_count": 3620112580,
        "batch_size": 256,
        "atol": 1e-12,
        "rtol": 1e-10,
        "estimated_peak_bytes": 2634508000,
        "maximum_abs_e_deterministic": 0.131565,
        "minimum_scattered_power": 0.0,
        "maximum_total_power": 0.017473,
        "returned_balance_max_abs_residual": 0.0,
    }
    return {key: values[key] for key in CONVERGENCE_KEYS}


def synthetic_stage1_document() -> dict[str, Any]:
    """One minimal document that satisfies every Section 8.1 Stage-1 rule."""
    digest = "0" * 64
    sha = "a" * 40
    node = "tests/unit/test_core/test_sci005_aperture_physics.py::case"
    command = {
        "argv": ["pixi", "run", "test"],
        "cwd": ".",
        "pixi_environment": "default",
        "started_at_utc": "2026-08-14T00:00:00Z",
        "duration_seconds": 1.0,
        "exit_code": 0,
        "stdout_sha256": digest,
        "stderr_sha256": digest,
    }
    document = {
        "schema_version": "radiosim.sci005.stage1.v1",
        "stage": 1,
        "status": "candidate",
        "generated_at_utc": "2026-08-14T00:00:00Z",
        "design_sha": sha,
        "red_test_sha": sha,
        "source_sha": sha,
        "evidence_sha": None,
        "working_tree_clean": True,
        "radiosim_version": "0.3.0",
        "python_version": "3.11.13",
        "platform": "macOS-15",
        "machine": "arm64",
        "pixi_environment": "default",
        "pixi_lock_sha256": digest,
        "scientific_conventions": dict(SCIENTIFIC_CONVENTIONS),
        "config_cases": [
            {
                "case_id": "accepted_uniform",
                "test_node_id": node,
                "input_sha256": digest,
                "expected_outcome": "accepted",
                "observed_outcome": "accepted",
                "resolved_scientific_sha256": digest,
                "exception_type": None,
                "issue_code": None,
                "exact_message": None,
                "passed": True,
            },
            {
                "case_id": "rejected_gaussian",
                "test_node_id": node,
                "input_sha256": digest,
                "expected_outcome": "rejected",
                "observed_outcome": "rejected",
                "resolved_scientific_sha256": None,
                "exception_type": "UnsupportedConfigError",
                "issue_code": "beam.aperture_physics.unsupported_pupil_profile",
                "exact_message": "Stage-1 aperture physics requires a canonical circular pupil; ...",
                "passed": True,
            },
        ],
        "analytic_invariants": [
            {
                "case_id": "blocked_uniform",
                "invariant_id": "blocked_aperture_transform",
                "backend": "numpy",
                "test_node_id": node,
                "input_manifest_sha256": digest,
                "expected": _projection(),
                "observed": _projection(),
                "max_abs_residual": 1e-15,
                "max_rel_residual": 1e-15,
                "atol": 1e-12,
                "rtol": 1e-10,
                "passed": True,
            },
            {
                "case_id": "extended_composed",
                "invariant_id": "extended_precision_mask_plus_zernike",
                "backend": "numpy",
                "test_node_id": node,
                "input_manifest_sha256": digest,
                "expected": _projection("complex256"),
                "observed": _projection("complex256"),
                "max_abs_residual": 1e-18,
                "max_rel_residual": 1e-18,
                "atol": 1e-12,
                "rtol": 1e-10,
                "passed": True,
            },
            {
                "case_id": "extended_unmodified",
                "invariant_id": "extended_precision_unmodified_profile",
                "backend": "numpy",
                "test_node_id": node,
                "input_manifest_sha256": digest,
                "expected": _projection("complex256"),
                "observed": _projection("complex256"),
                "max_abs_residual": 1e-18,
                "max_rel_residual": 1e-18,
                "atol": 1e-12,
                "rtol": 1e-10,
                "passed": True,
            },
        ],
        "rejection_probes": [
            {
                "case_id": "gaussian_pupil",
                "config_path": "beams.aperture_physics",
                "exception_type": "UnsupportedConfigError",
                "issue_code": "beam.aperture_physics.unsupported_pupil_profile",
                "exact_message": "Stage-1 aperture physics requires ...",
                "test_node_id": node,
                "input_sha256": digest,
                "passed": True,
            }
        ],
        "backend_parity": [
            {
                "case_id": "point_blockage",
                "backend": backend,
                "actual_device": "cpu",
                "real_dtype": "float64",
                "complex_dtype": "complex128",
                "input_sha256": digest,
                "reference_result_sha256": digest,
                "observed_result_sha256": digest,
                "max_abs_difference": 0.0,
                "max_rel_difference": 0.0,
                "atol": 1e-12,
                "rtol": 1e-10,
                "passed": True,
            }
            for backend in ("dask", "jax", "numpy")
        ],
        "solver_cases": [
            {
                "case_id": "blockage_point",
                "effect": "blockage",
                "test_node_id": node,
                "input_sha256": digest,
                "jones_sha256": digest,
                "visibility_sha256": digest,
                "diagnostic_sha256": None,
                "jones_call_count": 4,
                "visibility_changed_element_count": 8,
                "visibility_change_expected": True,
                "passed": True,
            },
            {
                "case_id": "diagnostic_only",
                "effect": "ruze_power_diagnostic_non_visibility",
                "test_node_id": node,
                "input_sha256": digest,
                "jones_sha256": digest,
                "visibility_sha256": digest,
                "diagnostic_sha256": digest,
                "jones_call_count": 0,
                "visibility_changed_element_count": 0,
                "visibility_change_expected": False,
                "passed": True,
            },
        ],
        "output_cases": [
            {
                "case_id": "hdf5_blockage",
                "format": "hdf5",
                "writer_test_node_id": node,
                "reader_test_node_id": node,
                "artifact_sha256": digest,
                "in_memory_sha256": digest,
                "observed_projection_sha256": digest,
                "roundtrip_max_abs_difference": 0.0,
                "tolerance": 1e-12,
                "passed": True,
            },
            {
                "case_id": "in_memory_blockage",
                "format": "in_memory",
                "writer_test_node_id": node,
                "reader_test_node_id": None,
                "artifact_sha256": None,
                "in_memory_sha256": digest,
                "observed_projection_sha256": digest,
                "roundtrip_max_abs_difference": None,
                "tolerance": None,
                "passed": True,
            },
        ],
        "fingerprint_diff": [
            {
                "environment": "default",
                "workload": "point_blockage",
                "old_scientific_sha256": digest,
                "new_scientific_sha256": "b" * 64,
                "old_raw_cube_sha256": digest,
                "new_raw_cube_sha256": "b" * 64,
                "changed_element_count": 8,
                "maximum_delta": 0.5,
                "change_expected": True,
                "test_node_id": node,
                "passed": True,
            },
            {
                "environment": "default",
                "workload": "point_default",
                "old_scientific_sha256": digest,
                "new_scientific_sha256": digest,
                "old_raw_cube_sha256": digest,
                "new_raw_cube_sha256": digest,
                "changed_element_count": 0,
                "maximum_delta": 0.0,
                "change_expected": False,
                "test_node_id": node,
                "passed": True,
            },
        ],
        "commands": [command],
        "artifacts": [
            {
                "path": "docs/development/sci005_stage1_evidence.schema.json",
                "sha256": digest,
                "media_type": "application/schema+json",
                "role": "schema",
            }
        ],
        "limitations": [],
        "claims_not_licensed": sorted(REQUIRED_CLAIMS),
        "pupil_profiles": [
            {
                "case_id": "uniform",
                "model_kind": "circular_aperture",
                "taper_kind": "uniform",
                "edge_taper_db": None,
                "mixture_weight": None,
                "profile_convention": "U",
                "hankel_convention": "two_J1_over_x",
                "outcome": "accepted",
                "exception_type": None,
                "issue_code": None,
                "test_node_id": node,
                "max_abs_residual": 1e-15,
                "tolerance": 1e-12,
            }
        ],
        "support_masks": [
            {
                "case_id": "north_leg",
                "diameter_m": 14.0,
                "central_diameter_ratio": 0.2,
                "legs": [{"position_angle_deg": 0.0, "width_m": 0.6}],
                "probes": [
                    {
                        "north_m": 4.0,
                        "east_m": 0.0,
                        "expected_transmitting": False,
                        "observed_transmitting": False,
                    }
                ],
                "union_control_id": "union",
                "antipodal_control_id": "antipodal",
                "topology_sha256": digest,
                "test_node_id": node,
                "passed": True,
            }
        ],
        "ruze_power_diagnostics": [
            {
                "case_id": "oracle_fixture",
                "resolved_aperture_scientific_sha256": digest,
                "diagnostic": {
                    "schema_version": "radiosim.ruze_power_diagnostic.v1",
                    "method": "poisson_paired_pupil_separation_v1",
                    "covariance_convention": "gaussian_one_over_e_surface_covariance_v1",
                    "normalization_convention": "unmodified_ideal_aperture_v1",
                    "antenna_id": {"number": 0, "name": "ANT0"},
                    "frequency_hz": 1.0e8,
                    "time_mjd": 60000.0,
                    "rms_surface_error_m": 0.02,
                    "correlation_length_m": 2.0,
                    "altitude_rad": _direction_projection(),
                    "azimuth_rad": _direction_projection(),
                    "coherent_main_power": _direction_projection(),
                    "total_ensemble_power": _direction_projection(),
                    "scattered_power": _direction_projection(),
                    "convergence": _convergence(),
                },
                "direct_pair_oracle": {
                    "test_node_id": node,
                    "aperture_node_count": 2048,
                    "direction_count": 3,
                    "max_abs_residual": 9.588e-19,
                    "tolerance": 1e-12,
                },
                "limit_oracles": [
                    {
                        "kind": kind,
                        "test_node_id": f"{node}::{kind}",
                        "input_sha256": digest,
                        "max_abs_residual": 1e-15,
                        "tolerance": 1e-12,
                    }
                    for kind in sorted(LIMIT_ORACLE_KINDS)
                ],
                "test_node_ids": [node],
            }
        ],
    }
    return {key: document[key] for key in STAGE1_KEYS}


# --- Section 7.5: the S/E state -----------------------------------------------


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_absent_artifact_and_null_sentinels_agree(stage: int) -> None:
    """Section 7.5: at ``S`` the artifact is absent and both constants are None."""
    source, digest = STAGE_CONSTANTS[stage]
    if source is None or digest is None:
        assert source is None and digest is None, (
            "the two approved constants for one stage move together"
        )
        assert not artifact_path(stage).exists()
        return
    assert GIT_SHA.fullmatch(source)
    assert SHA256.fullmatch(digest)
    assert artifact_path(stage).is_file()
    payload = artifact_path(stage).read_bytes()
    import hashlib

    assert hashlib.sha256(payload).hexdigest() == digest
    document = json.loads(payload.decode("utf-8"))
    assert document["source_sha"] == source
    if stage == 1:
        validate_stage1_evidence(document)


def test_this_validator_loads_only_the_standard_library() -> None:
    """An acceptance-critical validator carries no third-party import.

    ``tools/wp7_perf001_cpu_evidence.py`` set this precedent deliberately: a
    package that is only transitively present can vanish on a lock update and
    turn a hard authentication into a collection error.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    imported = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", source, re.M))
    roots = {name.split(".")[0] for name in imported}
    assert roots <= {
        "__future__",
        "hashlib",
        "json",
        "math",
        "re",
        "subprocess",
        "sys",
        "pathlib",
        "typing",
        "pytest",
    }, f"unexpected imports: {sorted(roots)}"


def test_the_schema_transcription_and_the_validator_agree() -> None:
    """The normative JSON transcription and this validator pin the same keys."""
    schema = json.loads(schema_path(1).read_text(encoding="utf-8"))
    assert tuple(schema["properties"]) == STAGE1_KEYS
    assert set(schema["required"]) == set(STAGE1_KEYS)
    assert schema["additionalProperties"] is False
    convergence = schema["$defs"]["ruze_convergence"]
    assert tuple(convergence["required"]) == CONVERGENCE_KEYS
    integers = {
        key
        for key, value in convergence["properties"].items()
        if value.get("type") == "integer"
    }
    assert integers == CONVERGENCE_INTEGERS
    conventions = schema["properties"]["scientific_conventions"]["properties"]
    assert {key: value["const"] for key, value in conventions.items()} == (
        SCIENTIFIC_CONVENTIONS
    )


def test_generator_is_present_and_exposes_the_frozen_invocation() -> None:
    """Section 8.1 freezes the generate sub-command and its two options."""
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / GENERATOR), "generate", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--stage" in completed.stdout
    assert "--measurement-record" in completed.stdout


def test_the_generator_also_loads_only_the_standard_library() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'tools'); "
            "import sci005_stage_evidence; "
            "print(sorted(n for n in sys.modules if n in {'jsonschema', 'numpy'}))",
        ],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


# --- rejection classes --------------------------------------------------------


def test_a_complete_synthetic_stage1_document_validates() -> None:
    validate_stage1_evidence(synthetic_stage1_document())


@pytest.mark.parametrize(
    "key",
    ["schema_version", "scientific_conventions", "ruze_power_diagnostics", "artifacts"],
)
def test_a_missing_top_level_key_is_rejected(key: str) -> None:
    document = synthetic_stage1_document()
    del document[key]
    with pytest.raises(EvidenceSchemaError):
        validate_stage1_evidence(document)


def test_an_unknown_top_level_key_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["converged"] = True
    with pytest.raises(EvidenceSchemaError):
        validate_stage1_evidence(document)


def test_a_reordered_top_level_key_sequence_is_rejected() -> None:
    document = synthetic_stage1_document()
    reordered = {key: document[key] for key in reversed(STAGE1_KEYS)}
    with pytest.raises(EvidenceSchemaError, match="declared order"):
        validate_stage1_evidence(reordered)


def test_a_non_null_evidence_sha_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["evidence_sha"] = "c" * 40
    with pytest.raises(EvidenceSchemaError):
        validate_stage1_evidence(document)


def test_a_short_or_upper_case_digest_is_rejected() -> None:
    for bad in ("0" * 63, "A" * 64):
        document = synthetic_stage1_document()
        document["pixi_lock_sha256"] = bad
        with pytest.raises(EvidenceSchemaError):
            validate_stage1_evidence(document)


def test_the_superseded_ruze_method_literal_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["scientific_conventions"]["ruze_method"] = (
        "poisson_gauss_hermite_aperture_v1"
    )
    with pytest.raises(EvidenceSchemaError):
        validate_stage1_evidence(document)


def test_a_boolean_where_a_number_belongs_is_rejected() -> None:
    """JSON booleans are never numbers, however Python spells ``bool``."""
    document = synthetic_stage1_document()
    document["ruze_power_diagnostics"][0]["diagnostic"]["convergence"]["poisson_mu"] = (
        True
    )
    with pytest.raises(EvidenceSchemaError):
        validate_stage1_evidence(document)


def test_a_boolean_where_an_integer_belongs_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["ruze_power_diagnostics"][0]["diagnostic"]["convergence"]["batch_size"] = (
        True
    )
    with pytest.raises(EvidenceSchemaError):
        validate_stage1_evidence(document)


def test_a_non_finite_number_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["ruze_power_diagnostics"][0]["diagnostic"]["convergence"]["atol"] = float(
        "inf"
    )
    with pytest.raises(EvidenceSchemaError):
        validate_stage1_evidence(document)


def test_a_converged_boolean_in_the_convergence_projection_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["ruze_power_diagnostics"][0]["diagnostic"]["convergence"]["converged"] = (
        True
    )
    with pytest.raises(EvidenceSchemaError):
        validate_stage1_evidence(document)


def test_a_backend_field_on_the_diagnostic_projection_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["ruze_power_diagnostics"][0]["diagnostic"]["backend"] = "numpy"
    with pytest.raises(EvidenceSchemaError):
        validate_stage1_evidence(document)


def test_a_missing_limit_oracle_kind_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["ruze_power_diagnostics"][0]["limit_oracles"].pop()
    with pytest.raises(EvidenceSchemaError):
        validate_stage1_evidence(document)


def test_a_cap_breach_in_the_convergence_projection_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["ruze_power_diagnostics"][0]["diagnostic"]["convergence"][
        "phase_product_count"
    ] = 2**34 + 1
    with pytest.raises(EvidenceSchemaError, match="cap"):
        validate_stage1_evidence(document)


def test_a_non_power_of_two_separation_angular_order_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["ruze_power_diagnostics"][0]["diagnostic"]["convergence"][
        "separation_angular_order_max"
    ] = 500
    with pytest.raises(EvidenceSchemaError, match="power of two"):
        validate_stage1_evidence(document)


def test_an_inconsistent_poisson_interval_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["ruze_power_diagnostics"][0]["diagnostic"]["convergence"][
        "poisson_term_count"
    ] = 4
    with pytest.raises(EvidenceSchemaError):
        validate_stage1_evidence(document)


def test_a_separation_truncation_bound_above_the_budget_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["ruze_power_diagnostics"][0]["diagnostic"]["convergence"][
        "separation_omitted_bound"
    ] = 1.0e-10
    with pytest.raises(EvidenceSchemaError, match="atol/8"):
        validate_stage1_evidence(document)


def test_an_inexact_returned_balance_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["ruze_power_diagnostics"][0]["diagnostic"]["convergence"][
        "returned_balance_max_abs_residual"
    ] = 1e-30
    with pytest.raises(EvidenceSchemaError, match="exact"):
        validate_stage1_evidence(document)


def test_a_diagnostic_row_that_changed_a_visibility_is_rejected() -> None:
    """Section 3.4.1: diagnostic power moving a visibility is a design violation."""
    document = synthetic_stage1_document()
    row = document["solver_cases"][1]
    row["visibility_changed_element_count"] = 1
    with pytest.raises(EvidenceSchemaError, match="diagnostic-only"):
        validate_stage1_evidence(document)


def test_a_diagnostic_row_that_called_evaluate_jones_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["solver_cases"][1]["jones_call_count"] = 1
    with pytest.raises(EvidenceSchemaError, match="diagnostic-only"):
        validate_stage1_evidence(document)


def test_a_disabled_workload_whose_fingerprint_moved_is_rejected() -> None:
    """Section 2: an absent block changes no fingerprint and no result byte."""
    document = synthetic_stage1_document()
    document["fingerprint_diff"][1]["new_scientific_sha256"] = "c" * 64
    with pytest.raises(EvidenceSchemaError, match="byte-identical"):
        validate_stage1_evidence(document)


def test_a_missing_disabled_control_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["fingerprint_diff"] = document["fingerprint_diff"][:1]
    with pytest.raises(EvidenceSchemaError, match="control"):
        validate_stage1_evidence(document)


def test_a_backend_parity_case_missing_a_backend_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["backend_parity"] = document["backend_parity"][:2]
    with pytest.raises(EvidenceSchemaError, match="missing a backend"):
        validate_stage1_evidence(document)


def test_an_unsorted_case_id_sequence_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["config_cases"] = list(reversed(document["config_cases"]))
    with pytest.raises(EvidenceSchemaError, match="sorted"):
        validate_stage1_evidence(document)


def test_a_duplicate_case_id_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["config_cases"][1]["case_id"] = document["config_cases"][0]["case_id"]
    with pytest.raises(EvidenceSchemaError):
        validate_stage1_evidence(document)


def test_an_accepted_config_case_carrying_an_error_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["config_cases"][0]["issue_code"] = "beam.aperture_physics.identity_block"
    with pytest.raises(EvidenceSchemaError, match="null error fields"):
        validate_stage1_evidence(document)


def test_a_rejected_config_case_carrying_a_resolution_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["config_cases"][1]["resolved_scientific_sha256"] = "0" * 64
    with pytest.raises(EvidenceSchemaError, match="null resolution"):
        validate_stage1_evidence(document)


def test_a_missing_extended_precision_invariant_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["analytic_invariants"] = [
        row
        for row in document["analytic_invariants"]
        if row["invariant_id"] != "extended_precision_mask_plus_zernike"
    ]
    with pytest.raises(EvidenceSchemaError, match="exactly one numpy row"):
        validate_stage1_evidence(document)


def test_an_extended_invariant_narrowed_to_complex128_is_rejected() -> None:
    document = synthetic_stage1_document()
    for row in document["analytic_invariants"]:
        if row["invariant_id"] == "extended_precision_unmodified_profile":
            row["expected"]["dtype"] = "complex128"
            row["observed"]["dtype"] = "complex128"
    with pytest.raises(EvidenceSchemaError, match="complex256"):
        validate_stage1_evidence(document)


def test_a_nonzero_command_exit_code_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["commands"][0]["exit_code"] = 1
    with pytest.raises(EvidenceSchemaError, match="zero exit code"):
        validate_stage1_evidence(document)


def test_an_absolute_or_escaping_artifact_path_is_rejected() -> None:
    for bad in ("/etc/passwd", "docs/../../escape.json", "docs\\development\\x.json"):
        document = synthetic_stage1_document()
        document["artifacts"][0]["path"] = bad
        with pytest.raises(EvidenceSchemaError):
            validate_stage1_evidence(document)


def test_an_unsorted_artifacts_array_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["artifacts"].insert(
        0,
        {
            "path": "zz/last.json",
            "sha256": "0" * 64,
            "media_type": "application/json",
            "role": "output",
        },
    )
    with pytest.raises(EvidenceSchemaError, match="sorted"):
        validate_stage1_evidence(document)


def test_a_missing_required_claim_is_rejected() -> None:
    document = synthetic_stage1_document()
    document["claims_not_licensed"] = [
        claim for claim in document["claims_not_licensed"] if "closure" not in claim
    ]
    with pytest.raises(EvidenceSchemaError, match="must name"):
        validate_stage1_evidence(document)


# === Section 8.1's Stage-2 evidence envelope ==================================
#
# Everything below is appended by ``S2``. It adds the Stage-2 validation and its
# synthetic-document tests and changes no Stage-1 validation logic, constant,
# synthetic fixture or test, exactly as the Stage-2 envelope requires.


#: Section 8.1's exact Stage-2 top-level key sequence: the common field
#: sequence followed by the five stage arrays, in that order.
STAGE2_KEYS: tuple[str, ...] = (
    "schema_version",
    "stage",
    "status",
    "generated_at_utc",
    "design_sha",
    "red_test_sha",
    "source_sha",
    "evidence_sha",
    "working_tree_clean",
    "radiosim_version",
    "python_version",
    "platform",
    "machine",
    "pixi_environment",
    "pixi_lock_sha256",
    "scientific_conventions",
    "config_cases",
    "analytic_invariants",
    "rejection_probes",
    "backend_parity",
    "solver_cases",
    "output_cases",
    "fingerprint_diff",
    "commands",
    "artifacts",
    "limitations",
    "claims_not_licensed",
    "squint_frequency_laws",
    "squint_geometries",
    "native_feed_factorizations",
    "stokes_v_leakages",
    "squint_setup_rejections",
)

#: The four frozen Stage-2 convention literals. They are facets of the one
#: ``cotton_uson_exact_v1`` version literal (Section 4.2.1), not independent
#: version axes, and the envelope retains all four verbatim.
STAGE2_SCIENTIFIC_CONVENTIONS: dict[str, str] = {
    "squint_frequency_law": "cotton_uson_exact_v1",
    "squint_direction": "feed_ray_plus_half_pi_north_through_east_v1",
    "squint_beam_frame": "pointing_then_squint_great_circle_v1",
    "squint_factorization": "receptor_conjugated_native_diagonal_v1",
}

#: The Stage-2 envelope supersedes the Stage-1-scoped member rule with exactly
#: these four members.
STAGE2_REQUIRED_CLAIMS: tuple[str, ...] = (
    "SCI-005 Stage-2 acceptance",
    "SCI-005 Stage 3",
    "SCI-005 whole-row closure",
    "a full cross-polar or measured-efield beam response",
)

#: Section 4.1's five mount literals, already owned by ``jones.P``.
STAGE2_MOUNT_TYPES: frozenset[str] = frozenset(
    {"alt-az", "equatorial", "fixed", "alt-az+nasmyth-l", "alt-az+nasmyth-r"}
)

#: The native feed labels each resolved receptor basis owns. Section 4.1.1:
#: ``x``/``y`` require ``linear`` and ``r``/``l`` require ``circular``.
STAGE2_FEEDS_BY_BASIS: dict[str, frozenset[str]] = {
    "linear": frozenset({"x", "y"}),
    "circular": frozenset({"r", "l"}),
}
STAGE2_RECEPTOR_BASES: frozenset[str] = frozenset(STAGE2_FEEDS_BY_BASIS)
STAGE2_NATIVE_FEEDS: frozenset[str] = frozenset({"x", "y", "r", "l"})

#: Section 8.1's frozen geometry-probe ``kind -> relation`` map. Only the
#: opposite-mount-sign probe is a lower bound; every other kind is an upper one.
STAGE2_PROBE_RELATIONS: dict[str, str] = {
    "orthogonality_dot_abs": "le",
    "handedness_plus_half_pi_residual_rad": "le",
    "midpoint_center_residual_rad": "le",
    "total_separation_residual_rad": "le",
    "mount_rotation_residual_rad": "le",
    "opposite_mount_sign_min_abs_delta_rad": "ge",
    "mechanical_rotation_residual_rad": "le",
    "feed_sign_reversal_center_residual_rad": "le",
}
STAGE2_PROBE_KINDS: frozenset[str] = frozenset(STAGE2_PROBE_RELATIONS)

#: Section 8.1's frozen setup-rejection ``case_kind -> exception_type`` map.
STAGE2_SETUP_EXCEPTIONS: dict[str, str] = {
    "unknown_antenna": "UnknownBeamAntennaError",
    "duplicate_antenna": "DuplicateBeamAssignmentError",
    "frequency_domain": "SquintFrequencyDomainError",
    "receptor_basis": "SquintReceptorBasisError",
    "boresight_degenerate": "BeamAngularDomainError",
}
STAGE2_SETUP_KINDS: frozenset[str] = frozenset(STAGE2_SETUP_EXCEPTIONS)

#: Section 4.2.1's exact frozen adapter message for a rotating mount whose
#: resolved boresight altitude is exactly ``pi/2`` in binary64.
BORESIGHT_DEGENERATE_MESSAGE = (
    "Beam squint on a rotating mount is undefined at an exactly zenith boresight."
)

#: The frozen ``beams.squint.default.*`` / ``beams.squint.per_antenna[i].*``
#: alternation Section 4.1.1 fixes for every value-domain rejection path.
_SQUINT_RECORD_PATH = r"beams\.squint\.(?:default|per_antenna\[[0-9]+\])"

#: Section 4.1.1's five frozen document rejections, as
#: ``issue_code -> (exception_type, config-path pattern, exact-message pattern)``.
#: The two rendered placeholders — ``{value!r}`` and ``{mode!r}`` — are the only
#: part of a frozen message a document may vary.
STAGE2_DOCUMENT_REJECTIONS: dict[str, tuple[str, re.Pattern[str], re.Pattern[str]]] = {
    "beam.squint.identity_block": (
        "ConfigSemanticError",
        re.compile(r"\Abeams\.squint\Z"),
        re.compile(
            r"\AA beams\.squint block must carry a default record or at least "
            r"one per-antenna record\.\Z"
        ),
    ),
    "beam.squint.reference_frequency_domain": (
        "ConfigSemanticError",
        re.compile(rf"\A{_SQUINT_RECORD_PATH}\.reference_frequency_hz\Z"),
        re.compile(
            r"\Asquint reference_frequency_hz must be a positive finite "
            r"frequency in Hz; resolved .+\.\Z"
        ),
    ),
    "beam.squint.offset_domain": (
        "ConfigSemanticError",
        re.compile(rf"\A{_SQUINT_RECORD_PATH}\.per_feed_offset_deg_at_reference\Z"),
        re.compile(
            r"\Asquint per_feed_offset_deg_at_reference must lie in the open "
            r"interval \(0, 90\); resolved .+\.\Z"
        ),
    ),
    "beam.squint.mechanical_angle_domain": (
        "ConfigSemanticError",
        re.compile(rf"\A{_SQUINT_RECORD_PATH}\.mechanical_feed_position_angle_deg\Z"),
        re.compile(
            r"\Asquint mechanical_feed_position_angle_deg must lie in "
            r"\(-180, 180\]; resolved .+\.\Z"
        ),
    ),
    "beam.squint.unsupported_beam_family": (
        "UnsupportedConfigError",
        re.compile(r"\Abeams\.squint\Z"),
        re.compile(
            r"\AStage-2 beam squint supports only the analytic beams mode; "
            r"resolved beams mode is .+\.\Z"
        ),
    ),
}

#: Section 8.1's Stage-2 solver effects, both of which must appear.
STAGE2_SOLVER_EFFECTS: frozenset[str] = frozenset({"squint_point", "squint_healpix"})

#: The one required extended-width factorization row.
EXTENDED_FACTORIZATION_CASE_ID = "extended_precision_native_feed_factorization"

#: The frequency-law recomputation budget: the recorded binary64 values must
#: agree with an independent binary64 recomputation to this absolute difference.
FREQUENCY_LAW_ABS_AGREEMENT = 1e-15

#: Half pi in binary64: the open upper end of every retained offset interval.
_HALF_PI = math.pi / 2.0


# --- Stage-2 shared helpers ---------------------------------------------------


def _open_interval(
    value: Any, path: str, *, lower: float, upper: float, closed_upper: bool = False
) -> float:
    """Require a finite number strictly inside ``(lower, upper)``."""
    numeric = _number(value, path, minimum=None)
    upper_ok = numeric <= upper if closed_upper else numeric < upper
    if not (numeric > lower and upper_ok):
        bracket = "]" if closed_upper else ")"
        _fail(path, f"{numeric!r} is outside the interval ({lower}, {upper}{bracket}")
    return numeric


def _positive(value: Any, path: str) -> float:
    numeric = _number(value, path, minimum=None)
    if not numeric > 0.0:
        _fail(path, f"expected a positive number, observed {numeric!r}")
    return numeric


def _signed(value: Any, path: str) -> float:
    return _number(value, path, minimum=None)


def _basis_and_feed(row: dict[str, Any], path: str) -> tuple[str, str]:
    """Require a resolved basis and a native feed label belonging to it."""
    basis = _string(
        row["receptor_basis"], f"{path}.receptor_basis", allowed=STAGE2_RECEPTOR_BASES
    )
    feed = _string(
        row["positive_native_feed"],
        f"{path}.positive_native_feed",
        allowed=STAGE2_NATIVE_FEEDS,
    )
    if feed not in STAGE2_FEEDS_BY_BASIS[basis]:
        _fail(
            f"{path}.positive_native_feed",
            f"{feed!r} does not belong to the {basis!r} receptor basis",
        )
    return basis, feed


def _factorization_projection(value: Any, path: str) -> dict[str, Any]:
    """A ``numeric_projection`` restricted to the Stage-2 factorization shapes."""
    row = _numeric_projection(value, path)
    _string(
        row["dtype"],
        f"{path}.dtype",
        allowed=frozenset({"complex128", "complex256"}),
    )
    shape = list(row["shape"])
    if shape[-2:] != [2, 2] or len(shape) not in (2, 3):
        _fail(f"{path}.shape", "must be [2, 2] or [S, 2, 2] with S >= 1")
    if len(shape) == 3 and shape[0] < 1:
        _fail(f"{path}.shape", "the leading batch extent must be at least one")
    return row


def _complex_pair(value: Any, path: str) -> complex:
    row = _mapping(value, path, ("real", "imag"))
    return complex(
        _signed(row["real"], f"{path}.real"), _signed(row["imag"], f"{path}.imag")
    )


# --- Section 8.1's Stage-2 rows -----------------------------------------------


def _squint_frequency_law(value: Any, path: str) -> dict[str, Any]:
    """One ``squint_frequency_laws`` row, with its binary64 recomputations."""
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "reference_frequency_hz",
            "per_feed_offset_deg_at_reference",
            "samples",
            "small_angle_control_frequency_hz",
            "small_angle_abs_separation",
            "max_abs_residual",
            "tolerance",
            "test_node_id",
            "passed",
        ),
    )
    _string(row["case_id"], f"{path}.case_id")
    _string(row["test_node_id"], f"{path}.test_node_id")
    reference = _positive(
        row["reference_frequency_hz"], f"{path}.reference_frequency_hz"
    )
    control = _positive(
        row["small_angle_control_frequency_hz"],
        f"{path}.small_angle_control_frequency_hz",
    )
    offset_deg = _open_interval(
        row["per_feed_offset_deg_at_reference"],
        f"{path}.per_feed_offset_deg_at_reference",
        lower=0.0,
        upper=90.0,
    )
    separation = _number(
        row["small_angle_abs_separation"], f"{path}.small_angle_abs_separation"
    )
    residual = _number(row["max_abs_residual"], f"{path}.max_abs_residual")
    tolerance = _positive(row["tolerance"], f"{path}.tolerance")
    _boolean(row["passed"], f"{path}.passed", const=True)

    samples = _array(row["samples"], f"{path}.samples", minimum_length=3)
    frequencies: list[float] = []
    residuals: list[float] = []
    control_separations: list[float] = []
    for index, item in enumerate(samples):
        where = f"{path}.samples[{index}]"
        sample = _mapping(
            item,
            where,
            (
                "frequency_hz",
                "expected_offset_rad",
                "observed_offset_rad",
                "small_angle_offset_rad",
            ),
        )
        frequency = _positive(sample["frequency_hz"], f"{where}.frequency_hz")
        offsets = {
            key: _open_interval(
                sample[key], f"{where}.{key}", lower=0.0, upper=_HALF_PI
            )
            for key in (
                "expected_offset_rad",
                "observed_offset_rad",
                "small_angle_offset_rad",
            )
        }
        if frequencies and frequency <= frequencies[-1]:
            _fail(f"{where}.frequency_hz", "samples are strictly increasing")
        frequencies.append(frequency)

        argument = (reference / frequency) * math.sin(math.radians(offset_deg))
        if not -1.0 <= argument <= 1.0:
            _fail(where, "the recomputed arcsine argument leaves [-1, 1]")
        if abs(math.asin(argument) - offsets["expected_offset_rad"]) > (
            FREQUENCY_LAW_ABS_AGREEMENT
        ):
            _fail(
                f"{where}.expected_offset_rad",
                "disagrees with the binary64 Cotton/Uson arcsine recomputation",
            )
        small = math.radians(offset_deg) * reference / frequency
        if abs(small - offsets["small_angle_offset_rad"]) > FREQUENCY_LAW_ABS_AGREEMENT:
            _fail(
                f"{where}.small_angle_offset_rad",
                "disagrees with the binary64 small-angle recomputation",
            )
        residuals.append(
            abs(offsets["observed_offset_rad"] - offsets["expected_offset_rad"])
        )
        if frequency == control:
            control_separations.append(
                abs(offsets["small_angle_offset_rad"] - offsets["expected_offset_rad"])
            )

    if max(residuals) != residual:
        _fail(
            f"{path}.max_abs_residual",
            "must equal the largest binary64 observed-minus-expected difference",
        )
    if residual > tolerance:
        _fail(f"{path}.max_abs_residual", "exceeds the retained tolerance")
    if control == reference:
        _fail(
            f"{path}.small_angle_control_frequency_hz",
            "must differ from the reference frequency",
        )
    if len(control_separations) != 1:
        _fail(
            f"{path}.small_angle_control_frequency_hz",
            "must name exactly one retained sample frequency",
        )
    if control_separations[0] != separation:
        _fail(
            f"{path}.small_angle_abs_separation",
            "must equal the control sample's binary64 small-angle separation",
        )
    if separation < 8.0 * tolerance:
        _fail(
            f"{path}.small_angle_abs_separation",
            "the small-angle control must clear eight times the tolerance",
        )
    return row


def _squint_geometry(value: Any, path: str) -> dict[str, Any]:
    """One ``squint_geometries`` row and its probe relations."""
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "mount_type",
            "mechanical_feed_position_angle_deg",
            "positive_native_feed",
            "receptor_basis",
            "parallactic_angle_rad",
            "boresight_altitude_rad",
            "frequency_hz",
            "resolved_offset_rad",
            "probes",
            "test_node_id",
            "passed",
        ),
    )
    _string(row["case_id"], f"{path}.case_id")
    _string(row["test_node_id"], f"{path}.test_node_id")
    _string(row["mount_type"], f"{path}.mount_type", allowed=STAGE2_MOUNT_TYPES)
    _open_interval(
        row["mechanical_feed_position_angle_deg"],
        f"{path}.mechanical_feed_position_angle_deg",
        lower=-180.0,
        upper=180.0,
        closed_upper=True,
    )
    _basis_and_feed(row, path)
    _signed(row["parallactic_angle_rad"], f"{path}.parallactic_angle_rad")
    _signed(row["boresight_altitude_rad"], f"{path}.boresight_altitude_rad")
    _positive(row["frequency_hz"], f"{path}.frequency_hz")
    _open_interval(
        row["resolved_offset_rad"],
        f"{path}.resolved_offset_rad",
        lower=0.0,
        upper=_HALF_PI,
    )
    _boolean(row["passed"], f"{path}.passed", const=True)

    probes = _array(row["probes"], f"{path}.probes", minimum_length=1)
    kinds: list[str] = []
    for index, item in enumerate(probes):
        where = f"{path}.probes[{index}]"
        probe = _mapping(
            item, where, ("kind", "observed", "bound", "relation", "passed")
        )
        kind = _string(probe["kind"], f"{where}.kind", allowed=STAGE2_PROBE_KINDS)
        kinds.append(kind)
        observed = _number(probe["observed"], f"{where}.observed")
        bound = _number(probe["bound"], f"{where}.bound")
        relation = _string(
            probe["relation"], f"{where}.relation", allowed=frozenset({"le", "ge"})
        )
        if relation != STAGE2_PROBE_RELATIONS[kind]:
            _fail(
                f"{where}.relation",
                f"kind {kind!r} is frozen to {STAGE2_PROBE_RELATIONS[kind]!r}",
            )
        satisfied = observed <= bound if relation == "le" else observed >= bound
        if _boolean(probe["passed"], f"{where}.passed") is not satisfied:
            _fail(f"{where}.passed", "must equal the probe's own relation outcome")
        if not satisfied:
            _fail(where, "candidate evidence records only satisfied probes")
    if len(set(kinds)) != len(kinds):
        _fail(f"{path}.probes", "a probe kind is repeated within the row")
    return row


def _native_feed_factorization(value: Any, path: str) -> dict[str, Any]:
    """One ``native_feed_factorizations`` row and its by-basis order control."""
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "receptor_basis",
            "feed_rotation_deg",
            "parallactic_angle_rad",
            "positive_native_feed",
            "b_plus",
            "b_minus",
            "expected",
            "observed",
            "factorization_max_abs_residual",
            "chain_order_max_abs_residual",
            "order_control_max_abs_difference",
            "atol",
            "test_node_id",
            "passed",
        ),
    )
    _string(row["case_id"], f"{path}.case_id")
    _string(row["test_node_id"], f"{path}.test_node_id")
    basis, _feed = _basis_and_feed(row, path)
    _signed(row["feed_rotation_deg"], f"{path}.feed_rotation_deg")
    _signed(row["parallactic_angle_rad"], f"{path}.parallactic_angle_rad")
    b_plus = _complex_pair(row["b_plus"], f"{path}.b_plus")
    b_minus = _complex_pair(row["b_minus"], f"{path}.b_minus")
    if b_plus == b_minus:
        _fail(path, "b_plus and b_minus must differ as complex pairs")
    expected = _factorization_projection(row["expected"], f"{path}.expected")
    observed = _factorization_projection(row["observed"], f"{path}.observed")
    if expected["dtype"] != observed["dtype"] or expected["shape"] != observed["shape"]:
        _fail(path, "expected and observed projections need identical dtype and shape")
    factorization = _number(
        row["factorization_max_abs_residual"], f"{path}.factorization_max_abs_residual"
    )
    chain_order = _number(
        row["chain_order_max_abs_residual"], f"{path}.chain_order_max_abs_residual"
    )
    order_control = _number(
        row["order_control_max_abs_difference"],
        f"{path}.order_control_max_abs_difference",
    )
    atol = _positive(row["atol"], f"{path}.atol")
    _boolean(row["passed"], f"{path}.passed", const=True)
    if factorization > atol:
        _fail(f"{path}.factorization_max_abs_residual", "exceeds the retained atol")
    if chain_order > atol:
        _fail(f"{path}.chain_order_max_abs_residual", "exceeds the retained atol")
    if basis == "linear":
        floor = max(1e-3, 1024.0 * atol)
        if order_control < floor:
            _fail(
                f"{path}.order_control_max_abs_difference",
                f"a rotated linear receptor must not commute; expected >= {floor}",
            )
    elif order_control > atol:
        _fail(
            f"{path}.order_control_max_abs_difference",
            "Section 4.2's circular commutation identity requires exact vanishing",
        )
    return row


def _stokes_v_leakage(value: Any, path: str) -> dict[str, Any]:
    """One ``stokes_v_leakages`` row; reciprocity is checked across the array."""
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "positive_native_feed",
            "reversed_case_id",
            "frequency_hz",
            "probe_altitude_rad",
            "probe_azimuth_rad",
            "observed_v_over_i",
            "expected_sign",
            "observed_sign",
            "min_abs_v_over_i",
            "test_node_id",
            "passed",
        ),
    )
    _string(row["case_id"], f"{path}.case_id")
    _string(row["test_node_id"], f"{path}.test_node_id")
    _string(row["reversed_case_id"], f"{path}.reversed_case_id")
    feed = _string(
        row["positive_native_feed"],
        f"{path}.positive_native_feed",
        allowed=frozenset({"r", "l"}),
    )
    _positive(row["frequency_hz"], f"{path}.frequency_hz")
    _signed(row["probe_altitude_rad"], f"{path}.probe_altitude_rad")
    _signed(row["probe_azimuth_rad"], f"{path}.probe_azimuth_rad")
    leakage = _signed(row["observed_v_over_i"], f"{path}.observed_v_over_i")
    signs: dict[str, int] = {}
    for key in ("expected_sign", "observed_sign"):
        signs[key] = _integer(row[key], f"{path}.{key}", minimum=None)
        if signs[key] not in (-1, 1):
            _fail(f"{path}.{key}", "must be exactly the signed integer -1 or 1")
    floor = _positive(row["min_abs_v_over_i"], f"{path}.min_abs_v_over_i")
    _boolean(row["passed"], f"{path}.passed", const=True)
    if signs["expected_sign"] != (1 if feed == "r" else -1):
        _fail(
            f"{path}.expected_sign",
            "is +1 exactly when the positive native feed is 'r'",
        )
    if signs["observed_sign"] != signs["expected_sign"]:
        _fail(f"{path}.observed_sign", "must equal the expected leakage sign")
    if leakage == 0.0 or (leakage > 0.0) != (signs["observed_sign"] == 1):
        _fail(f"{path}.observed_v_over_i", "does not carry the observed sign")
    if abs(leakage) < floor:
        _fail(f"{path}.observed_v_over_i", "is below the retained magnitude floor")
    if row["reversed_case_id"] == row["case_id"]:
        _fail(f"{path}.reversed_case_id", "must name the other row, not itself")
    return row


def _squint_setup_rejection(value: Any, path: str) -> dict[str, Any]:
    """One ``squint_setup_rejections`` row and its frozen exception type."""
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "case_kind",
            "exception_type",
            "exact_message",
            "test_node_id",
            "passed",
        ),
    )
    for key in ("case_id", "exception_type", "exact_message", "test_node_id"):
        _string(row[key], f"{path}.{key}")
    kind = _string(row["case_kind"], f"{path}.case_kind", allowed=STAGE2_SETUP_KINDS)
    _boolean(row["passed"], f"{path}.passed", const=True)
    if row["exception_type"] != STAGE2_SETUP_EXCEPTIONS[kind]:
        _fail(
            f"{path}.exception_type",
            f"kind {kind!r} is frozen to {STAGE2_SETUP_EXCEPTIONS[kind]!r}",
        )
    if kind == "boresight_degenerate" and (
        row["exact_message"] != BORESIGHT_DEGENERATE_MESSAGE
    ):
        _fail(
            f"{path}.exact_message",
            "must be Section 4.2.1's exact frozen zenith-boresight literal",
        )
    return row


def _stage2_solver_case(value: Any, path: str) -> dict[str, Any]:
    """One Stage-2 ``solver_cases`` row.

    The Stage-2 envelope replaces the Stage-1 effect enum outright: every row
    is a squint row, so no row carries a diagnostic digest, every row expects a
    visibility change and every row must actually have moved one.
    """
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "effect",
            "test_node_id",
            "input_sha256",
            "jones_sha256",
            "visibility_sha256",
            "diagnostic_sha256",
            "jones_call_count",
            "visibility_changed_element_count",
            "visibility_change_expected",
            "passed",
        ),
    )
    _string(row["case_id"], f"{path}.case_id")
    _string(row["effect"], f"{path}.effect", allowed=STAGE2_SOLVER_EFFECTS)
    _string(row["test_node_id"], f"{path}.test_node_id")
    for key in ("input_sha256", "jones_sha256", "visibility_sha256"):
        _string(row[key], f"{path}.{key}", pattern=SHA256)
    if row["diagnostic_sha256"] is not None:
        _fail(f"{path}.diagnostic_sha256", "is null on every Stage-2 solver row")
    _integer(row["jones_call_count"], f"{path}.jones_call_count")
    _integer(
        row["visibility_changed_element_count"],
        f"{path}.visibility_changed_element_count",
        minimum=1,
    )
    _boolean(
        row["visibility_change_expected"],
        f"{path}.visibility_change_expected",
        const=True,
    )
    _boolean(row["passed"], f"{path}.passed", const=True)
    return row


# --- the complete Stage-2 validator -------------------------------------------


def validate_stage2_evidence(document: Any) -> None:
    """Authenticate one Stage-2 evidence document against Section 8.1.

    Pure document validation, standard library only: exact key sets and order,
    the frozen literals, the ``git_sha``/``sha256``/timestamp encodings, JSON
    number and integer distinctions that reject booleans, sorted-unique arrays,
    and every Stage-2 cross-field predicate the envelope names.
    :func:`authenticate_stage2_succession` holds the three Git-object ancestry
    facts separately, exactly as Stage 1 keeps repository authentication out of
    :func:`validate_stage1_evidence`.
    """
    root = _mapping(document, "$", STAGE2_KEYS)
    _string(
        root["schema_version"], "$.schema_version", const="radiosim.sci005.stage2.v1"
    )
    if root["stage"] != 2 or isinstance(root["stage"], bool):
        _fail("$.stage", "must be the integer 2")
    _string(root["status"], "$.status", const="candidate")
    _string(root["generated_at_utc"], "$.generated_at_utc", pattern=TIMESTAMP)
    for key in ("design_sha", "red_test_sha", "source_sha"):
        _string(root[key], f"$.{key}", pattern=GIT_SHA)
    if root["evidence_sha"] is not None:
        _fail(
            "$.evidence_sha", "must be JSON null; the file cannot contain its own SHA"
        )
    _boolean(root["working_tree_clean"], "$.working_tree_clean", const=True)
    for key in (
        "radiosim_version",
        "python_version",
        "platform",
        "machine",
        "pixi_environment",
    ):
        _string(root[key], f"$.{key}")
    _string(root["pixi_lock_sha256"], "$.pixi_lock_sha256", pattern=SHA256)

    conventions = _mapping(
        root["scientific_conventions"],
        "$.scientific_conventions",
        tuple(STAGE2_SCIENTIFIC_CONVENTIONS),
    )
    for key, literal in STAGE2_SCIENTIFIC_CONVENTIONS.items():
        _string(conventions[key], f"$.scientific_conventions.{key}", const=literal)

    for key, checker in (
        ("config_cases", _config_case),
        ("rejection_probes", _rejection_probe),
    ):
        rows = _array(root[key], f"$.{key}", minimum_length=1)
        for index, row in enumerate(rows):
            checker(row, f"$.{key}[{index}]")
        _rows_sorted_by(rows, "case_id", f"$.{key}")

    probes = root["rejection_probes"]
    observed_codes = {row["issue_code"] for row in probes}
    for code, (
        exception,
        path_pattern,
        message_pattern,
    ) in STAGE2_DOCUMENT_REJECTIONS.items():
        if code not in observed_codes:
            _fail("$.rejection_probes", f"must carry the frozen code {code!r}")
        for index, row in enumerate(probes):
            if row["issue_code"] != code:
                continue
            where = f"$.rejection_probes[{index}]"
            if row["exception_type"] != exception:
                _fail(f"{where}.exception_type", f"{code!r} is frozen to {exception!r}")
            if path_pattern.fullmatch(row["config_path"]) is None:
                _fail(f"{where}.config_path", f"is not {code!r}'s frozen path")
            if message_pattern.fullmatch(row["exact_message"]) is None:
                _fail(f"{where}.exact_message", f"is not {code!r}'s frozen message")

    invariants = _array(
        root["analytic_invariants"], "$.analytic_invariants", minimum_length=1
    )
    parsed_invariants = [
        _analytic_invariant(row, f"$.analytic_invariants[{index}]")
        for index, row in enumerate(invariants)
    ]
    _rows_sorted_by(parsed_invariants, "case_id", "$.analytic_invariants")

    parity = _array(root["backend_parity"], "$.backend_parity", minimum_length=1)
    parsed_parity = [
        _backend_parity(row, f"$.backend_parity[{index}]")
        for index, row in enumerate(parity)
    ]
    _sorted_unique(
        [(row["case_id"], row["backend"]) for row in parsed_parity], "$.backend_parity"
    )
    by_case: dict[str, set[str]] = {}
    standard_width: set[str] = set()
    for row in parsed_parity:
        by_case.setdefault(row["case_id"], set()).add(row["backend"])
        if (row["real_dtype"], row["complex_dtype"]) == ("float64", "complex128"):
            standard_width.add(row["case_id"])
    for case_id, backends in by_case.items():
        if backends != {"numpy", "jax", "dask"}:
            _fail("$.backend_parity", f"case {case_id!r} is missing a backend")
    if not standard_width:
        _fail(
            "$.backend_parity",
            "at least one squint-enabled case needs the float64/complex128 pair",
        )

    solver_rows = _array(root["solver_cases"], "$.solver_cases", minimum_length=1)
    parsed_solver = [
        _stage2_solver_case(row, f"$.solver_cases[{index}]")
        for index, row in enumerate(solver_rows)
    ]
    _rows_sorted_by(parsed_solver, "case_id", "$.solver_cases")
    if {row["effect"] for row in parsed_solver} != STAGE2_SOLVER_EFFECTS:
        _fail(
            "$.solver_cases",
            "both squint_point and squint_healpix must appear at least once",
        )

    outputs = _array(root["output_cases"], "$.output_cases", minimum_length=1)
    for index, row in enumerate(outputs):
        _output_case(row, f"$.output_cases[{index}]")
    _rows_sorted_by(outputs, "case_id", "$.output_cases")
    formats = {row["format"] for row in outputs}
    for required_format in ("in_memory", "hdf5"):
        if required_format not in formats:
            _fail(
                "$.output_cases",
                f"a squint-enabled {required_format} row is required",
            )

    fingerprints = _array(
        root["fingerprint_diff"], "$.fingerprint_diff", minimum_length=1
    )
    parsed_fingerprints = [
        _fingerprint_row(row, f"$.fingerprint_diff[{index}]")
        for index, row in enumerate(fingerprints)
    ]
    _sorted_unique(
        [(row["environment"], row["workload"]) for row in parsed_fingerprints],
        "$.fingerprint_diff",
    )
    if {row["change_expected"] for row in parsed_fingerprints} != {True, False}:
        _fail(
            "$.fingerprint_diff",
            "both an enabled and a disabled/default control are required",
        )

    commands = _array(root["commands"], "$.commands", minimum_length=1)
    for index, row in enumerate(commands):
        _command_row(row, f"$.commands[{index}]")

    artifacts = _array(root["artifacts"], "$.artifacts", minimum_length=1)
    parsed_artifacts = [
        _artifact_row(row, f"$.artifacts[{index}]")
        for index, row in enumerate(artifacts)
    ]
    _rows_sorted_by(parsed_artifacts, "path", "$.artifacts")

    limitations = _array(root["limitations"], "$.limitations")
    for index, item in enumerate(limitations):
        _string(item, f"$.limitations[{index}]")
    _sorted_unique(limitations, "$.limitations")
    claims = _array(
        root["claims_not_licensed"], "$.claims_not_licensed", minimum_length=1
    )
    for index, item in enumerate(claims):
        _string(item, f"$.claims_not_licensed[{index}]")
    _sorted_unique(claims, "$.claims_not_licensed")
    for required in STAGE2_REQUIRED_CLAIMS:
        if required not in claims:
            _fail("$.claims_not_licensed", f"must name {required!r}")

    laws = _array(
        root["squint_frequency_laws"], "$.squint_frequency_laws", minimum_length=1
    )
    parsed_laws = [
        _squint_frequency_law(row, f"$.squint_frequency_laws[{index}]")
        for index, row in enumerate(laws)
    ]
    _rows_sorted_by(parsed_laws, "case_id", "$.squint_frequency_laws")

    geometries = _array(
        root["squint_geometries"], "$.squint_geometries", minimum_length=1
    )
    parsed_geometries = [
        _squint_geometry(row, f"$.squint_geometries[{index}]")
        for index, row in enumerate(geometries)
    ]
    _rows_sorted_by(parsed_geometries, "case_id", "$.squint_geometries")
    covered = {probe["kind"] for row in parsed_geometries for probe in row["probes"]}
    if covered != STAGE2_PROBE_KINDS:
        _fail(
            "$.squint_geometries",
            f"missing probe kinds {sorted(STAGE2_PROBE_KINDS - covered)}",
        )
    mounts = {row["mount_type"] for row in parsed_geometries}
    for required_mount in ("alt-az", "fixed"):
        if required_mount not in mounts:
            _fail(
                "$.squint_geometries",
                f"at least one row must carry mount_type {required_mount!r}",
            )

    factorizations = _array(
        root["native_feed_factorizations"],
        "$.native_feed_factorizations",
        minimum_length=1,
    )
    parsed_factorizations = [
        _native_feed_factorization(row, f"$.native_feed_factorizations[{index}]")
        for index, row in enumerate(factorizations)
    ]
    _rows_sorted_by(parsed_factorizations, "case_id", "$.native_feed_factorizations")
    bases = {row["receptor_basis"] for row in parsed_factorizations}
    if "circular" not in bases:
        _fail(
            "$.native_feed_factorizations",
            "the circular commutation witness row is required",
        )
    rotated_linear = [
        row
        for row in parsed_factorizations
        if row["receptor_basis"] == "linear" and row["feed_rotation_deg"] != 0.0
    ]
    if not rotated_linear:
        _fail(
            "$.native_feed_factorizations",
            "a linear row with a non-zero feed_rotation_deg is required",
        )
    extended = [
        row
        for row in parsed_factorizations
        if row["case_id"] == EXTENDED_FACTORIZATION_CASE_ID
    ]
    if len(extended) != 1:
        _fail(
            "$.native_feed_factorizations",
            f"exactly one {EXTENDED_FACTORIZATION_CASE_ID!r} row is required",
        )
    if extended[0]["observed"]["dtype"] != "complex256":
        _fail(
            "$.native_feed_factorizations",
            "the extended-width row must retain complex256 projections",
        )

    leakages = _array(
        root["stokes_v_leakages"], "$.stokes_v_leakages", minimum_length=1
    )
    parsed_leakages = [
        _stokes_v_leakage(row, f"$.stokes_v_leakages[{index}]")
        for index, row in enumerate(leakages)
    ]
    _rows_sorted_by(parsed_leakages, "case_id", "$.stokes_v_leakages")
    by_id = {row["case_id"]: row for row in parsed_leakages}
    for index, row in enumerate(parsed_leakages):
        where = f"$.stokes_v_leakages[{index}]"
        partner = by_id.get(row["reversed_case_id"])
        if partner is None:
            _fail(f"{where}.reversed_case_id", "does not name a retained row")
        if partner["reversed_case_id"] != row["case_id"]:
            _fail(f"{where}.reversed_case_id", "the reciprocity is not mutual")
        if partner["positive_native_feed"] == row["positive_native_feed"]:
            _fail(where, "the reversed row must carry the opposite native feed")
        if partner["observed_sign"] != -row["observed_sign"]:
            _fail(where, "the reversed row must carry the opposite leakage sign")

    rejections = _array(
        root["squint_setup_rejections"], "$.squint_setup_rejections", minimum_length=1
    )
    parsed_rejections = [
        _squint_setup_rejection(row, f"$.squint_setup_rejections[{index}]")
        for index, row in enumerate(rejections)
    ]
    _rows_sorted_by(parsed_rejections, "case_id", "$.squint_setup_rejections")
    observed_kinds = {row["case_kind"] for row in parsed_rejections}
    if observed_kinds != STAGE2_SETUP_KINDS:
        _fail(
            "$.squint_setup_rejections",
            f"missing case kinds {sorted(STAGE2_SETUP_KINDS - observed_kinds)}",
        )


# --- Section 8.1's three Stage-2 Git-object ancestry facts ---------------------


def _stage2_git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise EvidenceSchemaError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _stage2_parent_of(commit: str) -> str:
    """Return one commit's *direct parent*, never the commit itself.

    ``<rev>^{commit}`` is git's **peel** form: on a commit object it resolves
    back to that same commit. Section 8.1 records exactly that confusion as the
    evidence generator's Stage-2 defect, so every direct-parent question here
    goes through this one function rather than an inline expression.
    """
    parent = _stage2_git("rev-parse", f"{commit}^")
    if GIT_SHA.fullmatch(parent) is None or parent == commit:
        raise EvidenceSchemaError(f"{commit} has no distinct direct parent")
    return parent


def authenticate_stage2_succession(document: dict[str, Any]) -> None:
    """Authenticate ``R2^ == D2``, ``S2^ == R2`` and ``D2 != R2`` from Git.

    The Stage-2 envelope requires these three facts to be read from Git objects
    rather than trusted from the document, so the generator defect it records —
    a ``design_sha`` resolved as git's peel form of ``HEAD^`` and therefore
    equal to the ``red_test_sha`` — cannot pass validation again. This runs only
    against the real retained artifact; a synthetic document names no commit.
    """
    design = document["design_sha"]
    red_test = document["red_test_sha"]
    source = document["source_sha"]
    if design == red_test:
        raise EvidenceSchemaError(
            "$.design_sha: D2 and R2 are the same commit; Section 8.3 requires "
            "R2 to be a distinct child of D2"
        )
    observed_design = _stage2_parent_of(red_test)
    if observed_design != design:
        raise EvidenceSchemaError(
            f"$.design_sha: R2^ is {observed_design}, not the recorded {design}"
        )
    observed_red = _stage2_parent_of(source)
    if observed_red != red_test:
        raise EvidenceSchemaError(
            f"$.red_test_sha: S2^ is {observed_red}, not the recorded {red_test}"
        )


# --- the Stage-2 synthetic fixture --------------------------------------------


def _factorization_projection_value(dtype: str) -> dict[str, Any]:
    return {
        "dtype": dtype,
        "shape": [2, 2],
        "c_order_sha256": "0" * 64,
        "minimum_abs": 0.0,
        "maximum_abs": 1.0,
    }


def _frequency_law_row(
    case_id: str,
    node: str,
    reference_hz: float,
    offset_deg: float,
    frequencies: tuple[float, ...],
    control_hz: float,
    tolerance: float,
) -> dict[str, Any]:
    """Build one exactly self-consistent ``squint_frequency_laws`` row."""
    samples: list[dict[str, Any]] = []
    separation = 0.0
    for frequency in frequencies:
        expected = math.asin(
            (reference_hz / frequency) * math.sin(math.radians(offset_deg))
        )
        small = math.radians(offset_deg) * reference_hz / frequency
        samples.append(
            {
                "frequency_hz": frequency,
                "expected_offset_rad": expected,
                "observed_offset_rad": expected,
                "small_angle_offset_rad": small,
            }
        )
        if frequency == control_hz:
            separation = abs(small - expected)
    return {
        "case_id": case_id,
        "reference_frequency_hz": reference_hz,
        "per_feed_offset_deg_at_reference": offset_deg,
        "samples": samples,
        "small_angle_control_frequency_hz": control_hz,
        "small_angle_abs_separation": separation,
        "max_abs_residual": 0.0,
        "tolerance": tolerance,
        "test_node_id": node,
        "passed": True,
    }


def _geometry_probes(kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for kind in kinds:
        relation = STAGE2_PROBE_RELATIONS[kind]
        probes.append(
            {
                "kind": kind,
                "observed": 1.0 if relation == "ge" else 0.0,
                "bound": 1e-3 if relation == "ge" else 1e-12,
                "relation": relation,
                "passed": True,
            }
        )
    return probes


def _document_rejection_probe(
    case_id: str, code: str, config_path: str, message: str
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "config_path": config_path,
        "exception_type": STAGE2_DOCUMENT_REJECTIONS[code][0],
        "issue_code": code,
        "exact_message": message,
        "test_node_id": "tests/unit/test_io/test_sci005_beam_config.py::case",
        "input_sha256": "0" * 64,
        "passed": True,
    }


def synthetic_stage2_document() -> dict[str, Any]:
    """One minimal document that satisfies every Section 8.1 Stage-2 rule."""
    digest = "0" * 64
    sha = "a" * 40
    node = "tests/unit/test_core/test_sci005_beam_squint.py::case"
    command = {
        "argv": ["pixi", "run", "test"],
        "cwd": ".",
        "pixi_environment": "default",
        "started_at_utc": "2026-08-19T00:00:00Z",
        "duration_seconds": 1.0,
        "exit_code": 0,
        "stdout_sha256": digest,
        "stderr_sha256": digest,
    }
    document = {
        "schema_version": "radiosim.sci005.stage2.v1",
        "stage": 2,
        "status": "candidate",
        "generated_at_utc": "2026-08-19T00:00:00Z",
        "design_sha": sha,
        "red_test_sha": "b" * 40,
        "source_sha": "c" * 40,
        "evidence_sha": None,
        "working_tree_clean": True,
        "radiosim_version": "0.3.0",
        "python_version": "3.11.13",
        "platform": "macOS-15",
        "machine": "arm64",
        "pixi_environment": "default",
        "pixi_lock_sha256": digest,
        "scientific_conventions": dict(STAGE2_SCIENTIFIC_CONVENTIONS),
        "config_cases": [
            {
                "case_id": "accepted_default_squint",
                "test_node_id": node,
                "input_sha256": digest,
                "expected_outcome": "accepted",
                "observed_outcome": "accepted",
                "resolved_scientific_sha256": digest,
                "exception_type": None,
                "issue_code": None,
                "exact_message": None,
                "passed": True,
            },
            {
                "case_id": "rejected_identity_block",
                "test_node_id": node,
                "input_sha256": digest,
                "expected_outcome": "rejected",
                "observed_outcome": "rejected",
                "resolved_scientific_sha256": None,
                "exception_type": "ConfigSemanticError",
                "issue_code": "beam.squint.identity_block",
                "exact_message": (
                    "A beams.squint block must carry a default record or at "
                    "least one per-antenna record."
                ),
                "passed": True,
            },
        ],
        "analytic_invariants": [
            {
                "case_id": "cotton_uson_law",
                "invariant_id": "squint_frequency_law",
                "backend": "numpy",
                "test_node_id": node,
                "input_manifest_sha256": digest,
                "expected": _projection(),
                "observed": _projection(),
                "max_abs_residual": 1e-16,
                "max_rel_residual": 1e-16,
                "atol": 1e-12,
                "rtol": 1e-10,
                "passed": True,
            }
        ],
        "rejection_probes": [
            _document_rejection_probe(
                "identity_block",
                "beam.squint.identity_block",
                "beams.squint",
                "A beams.squint block must carry a default record or at least "
                "one per-antenna record.",
            ),
            _document_rejection_probe(
                "mechanical_angle_domain",
                "beam.squint.mechanical_angle_domain",
                "beams.squint.default.mechanical_feed_position_angle_deg",
                "squint mechanical_feed_position_angle_deg must lie in "
                "(-180, 180]; resolved 180.5.",
            ),
            _document_rejection_probe(
                "offset_domain",
                "beam.squint.offset_domain",
                "beams.squint.per_antenna[0].per_feed_offset_deg_at_reference",
                "squint per_feed_offset_deg_at_reference must lie in the open "
                "interval (0, 90); resolved 0.0.",
            ),
            _document_rejection_probe(
                "reference_frequency_domain",
                "beam.squint.reference_frequency_domain",
                "beams.squint.default.reference_frequency_hz",
                "squint reference_frequency_hz must be a positive finite "
                "frequency in Hz; resolved -1.0.",
            ),
            _document_rejection_probe(
                "unsupported_beam_family",
                "beam.squint.unsupported_beam_family",
                "beams.squint",
                "Stage-2 beam squint supports only the analytic beams mode; "
                "resolved beams mode is 'shared_fits'.",
            ),
        ],
        "backend_parity": [
            {
                "case_id": "squint_point",
                "backend": backend,
                "actual_device": "cpu",
                "real_dtype": "float64",
                "complex_dtype": "complex128",
                "input_sha256": digest,
                "reference_result_sha256": digest,
                "observed_result_sha256": digest,
                "max_abs_difference": 0.0,
                "max_rel_difference": 0.0,
                "atol": 1e-12,
                "rtol": 1e-10,
                "passed": True,
            }
            for backend in ("dask", "jax", "numpy")
        ],
        "solver_cases": [
            {
                "case_id": "squint_healpix_case",
                "effect": "squint_healpix",
                "test_node_id": node,
                "input_sha256": digest,
                "jones_sha256": digest,
                "visibility_sha256": digest,
                "diagnostic_sha256": None,
                "jones_call_count": 4,
                "visibility_changed_element_count": 8,
                "visibility_change_expected": True,
                "passed": True,
            },
            {
                "case_id": "squint_point_case",
                "effect": "squint_point",
                "test_node_id": node,
                "input_sha256": digest,
                "jones_sha256": digest,
                "visibility_sha256": digest,
                "diagnostic_sha256": None,
                "jones_call_count": 4,
                "visibility_changed_element_count": 8,
                "visibility_change_expected": True,
                "passed": True,
            },
        ],
        "output_cases": [
            {
                "case_id": "hdf5_squint",
                "format": "hdf5",
                "writer_test_node_id": node,
                "reader_test_node_id": node,
                "artifact_sha256": digest,
                "in_memory_sha256": digest,
                "observed_projection_sha256": digest,
                "roundtrip_max_abs_difference": 0.0,
                "tolerance": 1e-12,
                "passed": True,
            },
            {
                "case_id": "in_memory_squint",
                "format": "in_memory",
                "writer_test_node_id": node,
                "reader_test_node_id": None,
                "artifact_sha256": None,
                "in_memory_sha256": digest,
                "observed_projection_sha256": digest,
                "roundtrip_max_abs_difference": None,
                "tolerance": None,
                "passed": True,
            },
        ],
        "fingerprint_diff": [
            {
                "environment": "default",
                "workload": "point_default",
                "old_scientific_sha256": digest,
                "new_scientific_sha256": digest,
                "old_raw_cube_sha256": digest,
                "new_raw_cube_sha256": digest,
                "changed_element_count": 0,
                "maximum_delta": 0.0,
                "change_expected": False,
                "test_node_id": node,
                "passed": True,
            },
            {
                "environment": "default",
                "workload": "point_squint",
                "old_scientific_sha256": digest,
                "new_scientific_sha256": "b" * 64,
                "old_raw_cube_sha256": digest,
                "new_raw_cube_sha256": "b" * 64,
                "changed_element_count": 8,
                "maximum_delta": 0.5,
                "change_expected": True,
                "test_node_id": node,
                "passed": True,
            },
        ],
        "commands": [command],
        "artifacts": [
            {
                "path": "docs/development/sci005_stage2_evidence.schema.json",
                "sha256": digest,
                "media_type": "application/schema+json",
                "role": "schema",
            }
        ],
        "limitations": [
            "an alt-az antenna with no pointing offset has an exactly zenith "
            "boresight and is refused rather than approximated"
        ],
        "claims_not_licensed": sorted(STAGE2_REQUIRED_CLAIMS),
        "squint_frequency_laws": [
            _frequency_law_row(
                "cotton_uson_three_frequencies",
                node,
                1.0e8,
                1.0,
                (1.0e8, 1.5e8, 2.0e8),
                1.5e8,
                1e-12,
            )
        ],
        "squint_geometries": [
            {
                "case_id": "altaz_linear",
                "mount_type": "alt-az",
                "mechanical_feed_position_angle_deg": 45.0,
                "positive_native_feed": "x",
                "receptor_basis": "linear",
                "parallactic_angle_rad": 0.3,
                "boresight_altitude_rad": 1.2,
                "frequency_hz": 1.5e8,
                "resolved_offset_rad": 0.0116352,
                "probes": _geometry_probes(
                    (
                        "orthogonality_dot_abs",
                        "handedness_plus_half_pi_residual_rad",
                        "midpoint_center_residual_rad",
                        "total_separation_residual_rad",
                    )
                ),
                "test_node_id": node,
                "passed": True,
            },
            {
                "case_id": "fixed_circular",
                "mount_type": "fixed",
                "mechanical_feed_position_angle_deg": -30.0,
                "positive_native_feed": "r",
                "receptor_basis": "circular",
                "parallactic_angle_rad": 0.0,
                "boresight_altitude_rad": 1.4,
                "frequency_hz": 1.0e8,
                "resolved_offset_rad": 0.0174533,
                "probes": _geometry_probes(
                    (
                        "mount_rotation_residual_rad",
                        "opposite_mount_sign_min_abs_delta_rad",
                        "mechanical_rotation_residual_rad",
                        "feed_sign_reversal_center_residual_rad",
                    )
                ),
                "test_node_id": node,
                "passed": True,
            },
        ],
        "native_feed_factorizations": [
            {
                "case_id": "circular_commutation_witness",
                "receptor_basis": "circular",
                "feed_rotation_deg": 17.0,
                "parallactic_angle_rad": 0.4,
                "positive_native_feed": "r",
                "b_plus": {"real": 1.0, "imag": 0.0},
                "b_minus": {"real": 0.8, "imag": 0.0},
                "expected": _factorization_projection_value("complex128"),
                "observed": _factorization_projection_value("complex128"),
                "factorization_max_abs_residual": 0.0,
                "chain_order_max_abs_residual": 0.0,
                "order_control_max_abs_difference": 0.0,
                "atol": 1e-12,
                "test_node_id": node,
                "passed": True,
            },
            {
                "case_id": EXTENDED_FACTORIZATION_CASE_ID,
                "receptor_basis": "linear",
                "feed_rotation_deg": 30.0,
                "parallactic_angle_rad": 0.4,
                "positive_native_feed": "x",
                "b_plus": {"real": 1.0, "imag": 0.0},
                "b_minus": {"real": 0.8, "imag": 0.0},
                "expected": _factorization_projection_value("complex256"),
                "observed": _factorization_projection_value("complex256"),
                "factorization_max_abs_residual": 0.0,
                "chain_order_max_abs_residual": 0.0,
                "order_control_max_abs_difference": 0.1,
                "atol": 1e-15,
                "test_node_id": node,
                "passed": True,
            },
            {
                "case_id": "linear_rotated_order_control",
                "receptor_basis": "linear",
                "feed_rotation_deg": 30.0,
                "parallactic_angle_rad": 0.4,
                "positive_native_feed": "x",
                "b_plus": {"real": 1.0, "imag": 0.0},
                "b_minus": {"real": 0.8, "imag": 0.0},
                "expected": _factorization_projection_value("complex128"),
                "observed": _factorization_projection_value("complex128"),
                "factorization_max_abs_residual": 0.0,
                "chain_order_max_abs_residual": 0.0,
                "order_control_max_abs_difference": 0.1,
                "atol": 1e-12,
                "test_node_id": node,
                "passed": True,
            },
        ],
        "stokes_v_leakages": [
            {
                "case_id": "leakage_left_positive",
                "positive_native_feed": "l",
                "reversed_case_id": "leakage_right_positive",
                "frequency_hz": 1.5e8,
                "probe_altitude_rad": 1.3,
                "probe_azimuth_rad": 0.7,
                "observed_v_over_i": -0.05,
                "expected_sign": -1,
                "observed_sign": -1,
                "min_abs_v_over_i": 0.01,
                "test_node_id": node,
                "passed": True,
            },
            {
                "case_id": "leakage_right_positive",
                "positive_native_feed": "r",
                "reversed_case_id": "leakage_left_positive",
                "frequency_hz": 1.5e8,
                "probe_altitude_rad": 1.3,
                "probe_azimuth_rad": 0.7,
                "observed_v_over_i": 0.05,
                "expected_sign": 1,
                "observed_sign": 1,
                "min_abs_v_over_i": 0.01,
                "test_node_id": node,
                "passed": True,
            },
        ],
        "squint_setup_rejections": [
            {
                "case_id": "boresight_zenith",
                "case_kind": "boresight_degenerate",
                "exception_type": "BeamAngularDomainError",
                "exact_message": BORESIGHT_DEGENERATE_MESSAGE,
                "test_node_id": node,
                "passed": True,
            },
            {
                "case_id": "duplicate_reference",
                "case_kind": "duplicate_antenna",
                "exception_type": "DuplicateBeamAssignmentError",
                "exact_message": "duplicate beam assignment for antenna 0",
                "test_node_id": node,
                "passed": True,
            },
            {
                "case_id": "frequency_out_of_domain",
                "case_kind": "frequency_domain",
                "exception_type": "SquintFrequencyDomainError",
                "exact_message": "squint arcsine argument leaves [-1, 1]",
                "test_node_id": node,
                "passed": True,
            },
            {
                "case_id": "receptor_basis_mismatch",
                "case_kind": "receptor_basis",
                "exception_type": "SquintReceptorBasisError",
                "exact_message": "positive_native_feed 'r' is not a linear feed",
                "test_node_id": node,
                "passed": True,
            },
            {
                "case_id": "unknown_reference",
                "case_kind": "unknown_antenna",
                "exception_type": "UnknownBeamAntennaError",
                "exact_message": "unknown beam antenna reference 'ANT99'",
                "test_node_id": node,
                "passed": True,
            },
        ],
    }
    return {key: document[key] for key in STAGE2_KEYS}


# --- Section 7.5: the Stage-2 S/E state ---------------------------------------


def test_the_stage2_artifact_and_its_null_sentinels_agree() -> None:
    """At ``S2`` the artifact is absent; at ``E2`` it validates completely.

    This is the Stage-2 half of Section 7.5's ``S``/``E`` rule. It is a separate
    test rather than a change to the Stage-1 parametrized one, because ``S2``
    changes no Stage-1 validator byte.
    """
    source, digest = STAGE_CONSTANTS[2]
    if source is None or digest is None:
        assert source is None and digest is None, (
            "the two approved constants for one stage move together"
        )
        assert not artifact_path(2).exists()
        return
    assert GIT_SHA.fullmatch(source)
    assert SHA256.fullmatch(digest)
    assert artifact_path(2).is_file()
    payload = artifact_path(2).read_bytes()
    import hashlib

    assert hashlib.sha256(payload).hexdigest() == digest
    document = json.loads(payload.decode("utf-8"))
    validate_stage2_evidence(document)
    assert document["source_sha"] == source
    authenticate_stage2_succession(document)


def test_the_stage2_schema_transcription_and_the_validator_agree() -> None:
    """The normative Stage-2 transcription and this validator pin the same keys."""
    schema = json.loads(schema_path(2).read_text(encoding="utf-8"))
    assert tuple(schema["properties"]) == STAGE2_KEYS
    assert set(schema["required"]) == set(STAGE2_KEYS)
    assert schema["additionalProperties"] is False
    conventions = schema["properties"]["scientific_conventions"]["properties"]
    assert {key: value["const"] for key, value in conventions.items()} == (
        STAGE2_SCIENTIFIC_CONVENTIONS
    )
    assert schema["properties"]["stage"]["const"] == 2
    assert schema["properties"]["schema_version"]["const"] == (
        "radiosim.sci005.stage2.v1"
    )
    assert schema["properties"]["evidence_sha"] == {"type": "null"}
    assert set(schema["$defs"]["solver_case"]["properties"]["effect"]["enum"]) == (
        STAGE2_SOLVER_EFFECTS
    )
    assert (
        set(schema["$defs"]["squint_geometry_probe"]["properties"]["kind"]["enum"])
        == STAGE2_PROBE_KINDS
    )
    assert (
        set(
            schema["$defs"]["squint_setup_rejection"]["properties"]["case_kind"]["enum"]
        )
        == STAGE2_SETUP_KINDS
    )
    assert (
        set(schema["$defs"]["squint_geometry"]["properties"]["mount_type"]["enum"])
        == STAGE2_MOUNT_TYPES
    )


def test_the_generator_declares_the_five_stage2_measurement_keys() -> None:
    """Section 8.1: Stage 2 appends exactly those five arrays, in that order.

    The common field sequence is shared with Stage 1, so the Stage-2 key
    sequence must be the Stage-1 one with the three Stage-1 arrays replaced by
    these five, in the envelope's declared order.
    """
    stage_specific = (
        "squint_frequency_laws",
        "squint_geometries",
        "native_feed_factorizations",
        "stokes_v_leakages",
        "squint_setup_rejections",
    )
    assert STAGE2_KEYS[-5:] == stage_specific
    assert STAGE2_KEYS[:-5] == STAGE1_KEYS[:-3]

    source = (REPOSITORY_ROOT / GENERATOR).read_text(encoding="utf-8")
    assert "\nSTAGE2_MEASUREMENT_KEYS" in source, (
        f"{GENERATOR} must declare STAGE2_MEASUREMENT_KEYS"
    )
    body = source.split("\nSTAGE2_MEASUREMENT_KEYS", 1)[1].split(")", 1)[0]
    declared = tuple(re.findall(r'"([a-z0-9_]+)"', body))
    assert declared == stage_specific


def test_the_generator_resolves_the_stage2_design_sha_as_the_grandparent() -> None:
    """The Stage-2 envelope's recorded generator defect must stay repaired.

    ``HEAD^^{commit}`` is git's peel form of ``HEAD^``; the grandparent needs
    ``HEAD~2``. A regression here would silently record ``design_sha ==
    red_test_sha`` for every Stage-2 and Stage-3 evidence artifact.
    """
    source = (REPOSITORY_ROOT / GENERATOR).read_text(encoding="utf-8")
    body = source.split("def resolve_design_sha", 1)[1].split("\ndef ", 1)[0]
    assert 'run_git("rev-parse", "HEAD~2^{commit}")' in body
    assert 'run_git("rev-parse", "HEAD^^{commit}")' not in body


# --- Stage-2 rejection classes -------------------------------------------------


def test_a_complete_synthetic_stage2_document_validates() -> None:
    validate_stage2_evidence(synthetic_stage2_document())


@pytest.mark.parametrize(
    "key",
    [
        "schema_version",
        "scientific_conventions",
        "squint_frequency_laws",
        "squint_geometries",
        "native_feed_factorizations",
        "stokes_v_leakages",
        "squint_setup_rejections",
    ],
)
def test_a_missing_stage2_top_level_key_is_rejected(key: str) -> None:
    document = synthetic_stage2_document()
    del document[key]
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_an_unknown_stage2_top_level_key_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["ruze_power_diagnostics"] = []
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_a_reordered_stage2_top_level_key_sequence_is_rejected() -> None:
    document = synthetic_stage2_document()
    reordered = {key: document[key] for key in reversed(STAGE2_KEYS)}
    with pytest.raises(EvidenceSchemaError, match="declared order"):
        validate_stage2_evidence(reordered)


def test_a_stage1_schema_version_on_a_stage2_document_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["schema_version"] = "radiosim.sci005.stage1.v1"
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_a_stage2_document_declaring_stage_one_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["stage"] = 1
    with pytest.raises(EvidenceSchemaError, match="integer 2"):
        validate_stage2_evidence(document)


def test_a_boolean_stage_number_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["stage"] = True
    with pytest.raises(EvidenceSchemaError, match="integer 2"):
        validate_stage2_evidence(document)


def test_a_non_null_stage2_evidence_sha_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["evidence_sha"] = "d" * 40
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_a_stage1_convention_literal_on_a_stage2_document_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["scientific_conventions"]["squint_frequency_law"] = (
        "cotton_uson_small_angle_v1"
    )
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_an_extra_scientific_convention_key_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["scientific_conventions"]["aperture_method"] = (
        "boundary_fitted_polar_gauss_legendre_v1"
    )
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_the_stage1_claim_member_rule_does_not_satisfy_stage2() -> None:
    document = synthetic_stage2_document()
    document["claims_not_licensed"] = sorted(REQUIRED_CLAIMS)
    with pytest.raises(EvidenceSchemaError, match="must name"):
        validate_stage2_evidence(document)


def test_a_missing_stage2_claim_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["claims_not_licensed"] = [
        claim
        for claim in document["claims_not_licensed"]
        if claim != "a full cross-polar or measured-efield beam response"
    ]
    with pytest.raises(EvidenceSchemaError, match="must name"):
        validate_stage2_evidence(document)


def test_a_stage1_solver_effect_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["solver_cases"][0]["effect"] = "blockage"
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_a_missing_stage2_solver_effect_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["solver_cases"] = document["solver_cases"][:1]
    with pytest.raises(EvidenceSchemaError, match="squint_point"):
        validate_stage2_evidence(document)


def test_a_solver_row_carrying_a_diagnostic_digest_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["solver_cases"][0]["diagnostic_sha256"] = "0" * 64
    with pytest.raises(EvidenceSchemaError, match="null on every Stage-2"):
        validate_stage2_evidence(document)


def test_a_solver_row_that_moved_no_visibility_element_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["solver_cases"][0]["visibility_changed_element_count"] = 0
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_a_solver_row_expecting_no_visibility_change_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["solver_cases"][0]["visibility_change_expected"] = False
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_a_backend_parity_case_missing_a_backend_is_rejected_at_stage2() -> None:
    document = synthetic_stage2_document()
    document["backend_parity"] = document["backend_parity"][:2]
    with pytest.raises(EvidenceSchemaError, match="missing a backend"):
        validate_stage2_evidence(document)


def test_a_stage2_parity_set_without_the_standard_width_pair_is_rejected() -> None:
    document = synthetic_stage2_document()
    for row in document["backend_parity"]:
        row["real_dtype"] = "float32"
        row["complex_dtype"] = "complex64"
    with pytest.raises(EvidenceSchemaError, match="float64/complex128"):
        validate_stage2_evidence(document)


def test_a_stage2_output_set_without_an_hdf5_row_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["output_cases"] = [
        row for row in document["output_cases"] if row["format"] != "hdf5"
    ]
    with pytest.raises(EvidenceSchemaError, match="hdf5"):
        validate_stage2_evidence(document)


def test_a_stage2_output_set_without_an_in_memory_row_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["output_cases"] = [
        row for row in document["output_cases"] if row["format"] != "in_memory"
    ]
    with pytest.raises(EvidenceSchemaError, match="in_memory"):
        validate_stage2_evidence(document)


def test_a_missing_stage2_rejection_probe_code_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["rejection_probes"] = [
        row
        for row in document["rejection_probes"]
        if row["issue_code"] != "beam.squint.offset_domain"
    ]
    with pytest.raises(EvidenceSchemaError, match="frozen code"):
        validate_stage2_evidence(document)


def test_a_rejection_probe_with_a_foreign_config_path_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["rejection_probes"][0]["config_path"] = "beams.pointing"
    with pytest.raises(EvidenceSchemaError, match="frozen path"):
        validate_stage2_evidence(document)


def test_a_rejection_probe_with_a_paraphrased_message_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["rejection_probes"][0]["exact_message"] = "beams.squint must not be empty."
    with pytest.raises(EvidenceSchemaError, match="frozen message"):
        validate_stage2_evidence(document)


def test_a_rejection_probe_with_the_wrong_exception_type_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["rejection_probes"][0]["exception_type"] = "UnsupportedConfigError"
    with pytest.raises(EvidenceSchemaError, match="frozen to"):
        validate_stage2_evidence(document)


def test_a_frequency_law_row_with_fewer_than_three_samples_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_frequency_laws"][0]["samples"] = document["squint_frequency_laws"][
        0
    ]["samples"][:2]
    with pytest.raises(EvidenceSchemaError, match="at least 3"):
        validate_stage2_evidence(document)


def test_unsorted_frequency_law_samples_are_rejected() -> None:
    document = synthetic_stage2_document()
    samples = document["squint_frequency_laws"][0]["samples"]
    document["squint_frequency_laws"][0]["samples"] = list(reversed(samples))
    with pytest.raises(EvidenceSchemaError, match="strictly increasing"):
        validate_stage2_evidence(document)


def test_a_small_angle_offset_recorded_as_the_exact_law_is_rejected() -> None:
    """The small-angle column is the *control*, not a second copy of the law."""
    document = synthetic_stage2_document()
    for sample in document["squint_frequency_laws"][0]["samples"]:
        sample["small_angle_offset_rad"] = sample["expected_offset_rad"]
    with pytest.raises(EvidenceSchemaError, match="small-angle recomputation"):
        validate_stage2_evidence(document)


def test_an_expected_offset_that_fails_the_arcsine_recomputation_is_rejected() -> None:
    document = synthetic_stage2_document()
    sample = document["squint_frequency_laws"][0]["samples"][0]
    sample["expected_offset_rad"] += 1e-12
    with pytest.raises(EvidenceSchemaError, match="arcsine recomputation"):
        validate_stage2_evidence(document)


def test_a_max_abs_residual_that_is_not_the_recomputed_maximum_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_frequency_laws"][0]["max_abs_residual"] = 1e-30
    with pytest.raises(EvidenceSchemaError, match="largest binary64"):
        validate_stage2_evidence(document)


def test_a_max_abs_residual_above_its_tolerance_is_rejected() -> None:
    document = synthetic_stage2_document()
    row = document["squint_frequency_laws"][0]
    row["samples"][0]["observed_offset_rad"] = (
        row["samples"][0]["expected_offset_rad"] + 1e-9
    )
    row["max_abs_residual"] = abs(
        row["samples"][0]["observed_offset_rad"]
        - row["samples"][0]["expected_offset_rad"]
    )
    with pytest.raises(EvidenceSchemaError, match="exceeds the retained tolerance"):
        validate_stage2_evidence(document)


def test_a_small_angle_control_at_the_reference_frequency_is_rejected() -> None:
    document = synthetic_stage2_document()
    row = document["squint_frequency_laws"][0]
    row["small_angle_control_frequency_hz"] = row["reference_frequency_hz"]
    with pytest.raises(EvidenceSchemaError, match="differ from the reference"):
        validate_stage2_evidence(document)


def test_a_small_angle_control_naming_no_sample_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_frequency_laws"][0]["small_angle_control_frequency_hz"] = 3.0e8
    with pytest.raises(EvidenceSchemaError, match="exactly one retained sample"):
        validate_stage2_evidence(document)


def test_a_small_angle_separation_below_eight_tolerances_is_rejected() -> None:
    document = synthetic_stage2_document()
    row = document["squint_frequency_laws"][0]
    row["tolerance"] = row["small_angle_abs_separation"]
    with pytest.raises(EvidenceSchemaError, match="eight times the tolerance"):
        validate_stage2_evidence(document)


def test_a_geometry_probe_with_the_wrong_relation_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_geometries"][0]["probes"][0]["relation"] = "ge"
    with pytest.raises(EvidenceSchemaError, match="frozen to"):
        validate_stage2_evidence(document)


def test_the_opposite_mount_sign_probe_is_frozen_to_a_lower_bound() -> None:
    document = synthetic_stage2_document()
    for probe in document["squint_geometries"][1]["probes"]:
        if probe["kind"] == "opposite_mount_sign_min_abs_delta_rad":
            probe["relation"] = "le"
    with pytest.raises(EvidenceSchemaError, match="frozen to"):
        validate_stage2_evidence(document)


def test_a_geometry_probe_whose_pass_flag_contradicts_its_bound_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_geometries"][0]["probes"][0]["observed"] = 1.0
    with pytest.raises(EvidenceSchemaError, match="relation outcome"):
        validate_stage2_evidence(document)


def test_a_repeated_probe_kind_inside_one_geometry_row_is_rejected() -> None:
    document = synthetic_stage2_document()
    probes = document["squint_geometries"][0]["probes"]
    probes.append(dict(probes[0]))
    with pytest.raises(EvidenceSchemaError, match="repeated"):
        validate_stage2_evidence(document)


def test_a_missing_geometry_probe_kind_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_geometries"][0]["probes"] = document["squint_geometries"][0][
        "probes"
    ][:3]
    with pytest.raises(EvidenceSchemaError, match="missing probe kinds"):
        validate_stage2_evidence(document)


def test_a_geometry_array_without_an_alt_az_row_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_geometries"][0]["mount_type"] = "equatorial"
    with pytest.raises(EvidenceSchemaError, match="alt-az"):
        validate_stage2_evidence(document)


def test_a_geometry_array_without_a_fixed_row_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_geometries"][1]["mount_type"] = "alt-az+nasmyth-r"
    with pytest.raises(EvidenceSchemaError, match="fixed"):
        validate_stage2_evidence(document)


def test_an_unknown_mount_literal_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_geometries"][0]["mount_type"] = "alt-az+nasmyth"
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_a_mechanical_angle_outside_its_canonical_interval_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_geometries"][0]["mechanical_feed_position_angle_deg"] = -180.0
    with pytest.raises(EvidenceSchemaError, match="outside the interval"):
        validate_stage2_evidence(document)


def test_a_feed_label_from_the_wrong_basis_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_geometries"][0]["positive_native_feed"] = "r"
    with pytest.raises(EvidenceSchemaError, match="does not belong"):
        validate_stage2_evidence(document)


def test_a_resolved_offset_at_or_beyond_half_pi_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_geometries"][0]["resolved_offset_rad"] = math.pi / 2.0
    with pytest.raises(EvidenceSchemaError, match="outside the interval"):
        validate_stage2_evidence(document)


def test_a_linear_factorization_row_that_commutes_is_rejected() -> None:
    """Section 4.2: a rotated linear receptor's order control cannot vanish."""
    document = synthetic_stage2_document()
    document["native_feed_factorizations"][2]["order_control_max_abs_difference"] = 0.0
    with pytest.raises(EvidenceSchemaError, match="must not commute"):
        validate_stage2_evidence(document)


def test_a_circular_factorization_row_that_fails_to_commute_is_rejected() -> None:
    """Section 4.2: the circular order control is identically zero."""
    document = synthetic_stage2_document()
    document["native_feed_factorizations"][0]["order_control_max_abs_difference"] = 0.1
    with pytest.raises(EvidenceSchemaError, match="commutation identity"):
        validate_stage2_evidence(document)


def test_a_linear_order_control_below_the_thousand_atol_floor_is_rejected() -> None:
    document = synthetic_stage2_document()
    row = document["native_feed_factorizations"][2]
    row["atol"] = 1e-3
    row["order_control_max_abs_difference"] = 1.0
    with pytest.raises(EvidenceSchemaError, match="must not commute"):
        validate_stage2_evidence(document)


def test_a_factorization_residual_above_its_atol_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["native_feed_factorizations"][0]["factorization_max_abs_residual"] = 1e-6
    with pytest.raises(EvidenceSchemaError, match="exceeds the retained atol"):
        validate_stage2_evidence(document)


def test_a_chain_order_residual_above_its_atol_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["native_feed_factorizations"][0]["chain_order_max_abs_residual"] = 1e-6
    with pytest.raises(EvidenceSchemaError, match="exceeds the retained atol"):
        validate_stage2_evidence(document)


def test_equal_native_feed_samples_are_rejected() -> None:
    document = synthetic_stage2_document()
    row = document["native_feed_factorizations"][0]
    row["b_minus"] = dict(row["b_plus"])
    with pytest.raises(EvidenceSchemaError, match="must differ"):
        validate_stage2_evidence(document)


def test_a_missing_extended_precision_factorization_row_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["native_feed_factorizations"] = [
        row
        for row in document["native_feed_factorizations"]
        if row["case_id"] != EXTENDED_FACTORIZATION_CASE_ID
    ]
    with pytest.raises(EvidenceSchemaError, match="exactly one"):
        validate_stage2_evidence(document)


def test_an_extended_factorization_row_narrowed_to_complex128_is_rejected() -> None:
    document = synthetic_stage2_document()
    for row in document["native_feed_factorizations"]:
        if row["case_id"] == EXTENDED_FACTORIZATION_CASE_ID:
            row["expected"]["dtype"] = "complex128"
            row["observed"]["dtype"] = "complex128"
    with pytest.raises(EvidenceSchemaError, match="complex256"):
        validate_stage2_evidence(document)


def test_a_factorization_projection_of_the_wrong_shape_is_rejected() -> None:
    document = synthetic_stage2_document()
    row = document["native_feed_factorizations"][0]
    row["expected"]["shape"] = [3]
    row["observed"]["shape"] = [3]
    with pytest.raises(EvidenceSchemaError, match=r"\[2, 2\]"):
        validate_stage2_evidence(document)


def test_a_float_factorization_projection_dtype_is_rejected() -> None:
    document = synthetic_stage2_document()
    row = document["native_feed_factorizations"][0]
    row["expected"]["dtype"] = "float64"
    row["observed"]["dtype"] = "float64"
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_a_factorization_array_without_a_circular_row_is_rejected() -> None:
    document = synthetic_stage2_document()
    row = document["native_feed_factorizations"][0]
    row["receptor_basis"] = "linear"
    row["positive_native_feed"] = "x"
    row["order_control_max_abs_difference"] = 0.1
    with pytest.raises(EvidenceSchemaError, match="circular commutation witness"):
        validate_stage2_evidence(document)


def test_a_factorization_array_without_a_rotated_linear_row_is_rejected() -> None:
    document = synthetic_stage2_document()
    for row in document["native_feed_factorizations"]:
        if row["receptor_basis"] == "linear":
            row["feed_rotation_deg"] = 0.0
    with pytest.raises(EvidenceSchemaError, match="non-zero feed_rotation_deg"):
        validate_stage2_evidence(document)


def test_a_stokes_row_whose_expected_sign_contradicts_its_feed_is_rejected() -> None:
    document = synthetic_stage2_document()
    row = document["stokes_v_leakages"][1]
    row["expected_sign"] = -1
    with pytest.raises(EvidenceSchemaError, match=r"\+1 exactly when"):
        validate_stage2_evidence(document)


def test_a_stokes_row_whose_observed_sign_disagrees_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["stokes_v_leakages"][1]["observed_sign"] = -1
    with pytest.raises(EvidenceSchemaError, match="expected leakage sign"):
        validate_stage2_evidence(document)


def test_a_stokes_row_whose_ratio_contradicts_its_sign_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["stokes_v_leakages"][1]["observed_v_over_i"] = -0.05
    with pytest.raises(EvidenceSchemaError, match="observed sign"):
        validate_stage2_evidence(document)


def test_a_stokes_row_below_its_magnitude_floor_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["stokes_v_leakages"][1]["min_abs_v_over_i"] = 1.0
    with pytest.raises(EvidenceSchemaError, match="magnitude floor"):
        validate_stage2_evidence(document)


def test_a_stokes_row_naming_itself_as_its_reversal_is_rejected() -> None:
    document = synthetic_stage2_document()
    row = document["stokes_v_leakages"][0]
    row["reversed_case_id"] = row["case_id"]
    with pytest.raises(EvidenceSchemaError, match="not itself"):
        validate_stage2_evidence(document)


def test_a_stokes_row_naming_an_absent_reversal_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["stokes_v_leakages"][0]["reversed_case_id"] = "absent_case"
    with pytest.raises(EvidenceSchemaError, match="does not name a retained row"):
        validate_stage2_evidence(document)


def test_a_broken_stokes_reciprocity_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["stokes_v_leakages"].append(
        {
            "case_id": "leakage_third_wheel",
            "positive_native_feed": "r",
            "reversed_case_id": "leakage_left_positive",
            "frequency_hz": 1.5e8,
            "probe_altitude_rad": 1.3,
            "probe_azimuth_rad": 0.7,
            "observed_v_over_i": 0.05,
            "expected_sign": 1,
            "observed_sign": 1,
            "min_abs_v_over_i": 0.01,
            "test_node_id": "tests/unit/test_core/test_sci005_beam_squint.py::case",
            "passed": True,
        }
    )
    with pytest.raises(EvidenceSchemaError, match="not mutual"):
        validate_stage2_evidence(document)


def test_a_setup_rejection_with_a_foreign_exception_type_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_setup_rejections"][0]["exception_type"] = "BeamLoadError"
    with pytest.raises(EvidenceSchemaError, match="frozen to"):
        validate_stage2_evidence(document)


def test_a_paraphrased_zenith_boresight_message_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_setup_rejections"][0]["exact_message"] = (
        "Beam squint on a rotating mount is undefined at zenith."
    )
    with pytest.raises(EvidenceSchemaError, match="frozen zenith-boresight"):
        validate_stage2_evidence(document)


def test_a_missing_setup_rejection_kind_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_setup_rejections"] = document["squint_setup_rejections"][:-1]
    with pytest.raises(EvidenceSchemaError, match="missing case kinds"):
        validate_stage2_evidence(document)


def test_an_unsorted_stage2_case_id_sequence_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_setup_rejections"] = list(
        reversed(document["squint_setup_rejections"])
    )
    with pytest.raises(EvidenceSchemaError, match="sorted"):
        validate_stage2_evidence(document)


def test_a_duplicate_stage2_case_id_is_rejected() -> None:
    document = synthetic_stage2_document()
    rows = document["native_feed_factorizations"]
    rows[2]["case_id"] = rows[1]["case_id"]
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_a_boolean_where_a_stage2_number_belongs_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_frequency_laws"][0]["tolerance"] = True
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_a_boolean_where_a_stage2_sign_belongs_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["stokes_v_leakages"][1]["observed_sign"] = True
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_a_non_finite_stage2_number_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_geometries"][0]["parallactic_angle_rad"] = float("nan")
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_a_false_stage2_row_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_setup_rejections"][0]["passed"] = False
    with pytest.raises(EvidenceSchemaError):
        validate_stage2_evidence(document)


def test_a_zero_tolerance_frequency_law_row_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["squint_frequency_laws"][0]["tolerance"] = 0.0
    with pytest.raises(EvidenceSchemaError, match="positive"):
        validate_stage2_evidence(document)


def test_a_nonzero_command_exit_code_is_rejected_at_stage2() -> None:
    document = synthetic_stage2_document()
    document["commands"][0]["exit_code"] = 1
    with pytest.raises(EvidenceSchemaError, match="zero exit code"):
        validate_stage2_evidence(document)


def test_a_stage2_disabled_workload_whose_fingerprint_moved_is_rejected() -> None:
    document = synthetic_stage2_document()
    document["fingerprint_diff"][0]["new_scientific_sha256"] = "c" * 64
    with pytest.raises(EvidenceSchemaError, match="byte-identical"):
        validate_stage2_evidence(document)


def test_a_stage2_document_whose_design_equals_its_red_test_is_refused() -> None:
    """The recorded generator defect must fail authentication, not pass it."""
    document = synthetic_stage2_document()
    document["design_sha"] = document["red_test_sha"]
    with pytest.raises(EvidenceSchemaError, match="same commit"):
        authenticate_stage2_succession(document)


def test_the_stage2_succession_reads_parents_not_peels() -> None:
    """``<sha>^{commit}`` peels; only ``<sha>^`` is the direct parent.

    Section 8.1 records that confusion as the evidence generator's Stage-2
    defect. The same confusion inside this validator would make the three
    ancestry facts tautologies, so the distinction is pinned against real
    repository objects rather than assumed.
    """
    head = _stage2_git("rev-parse", "HEAD")
    assert _stage2_git("rev-parse", f"{head}^{{commit}}") == head
    parent = _stage2_parent_of(head)
    assert GIT_SHA.fullmatch(parent)
    assert parent != head
    assert parent == _stage2_git("rev-parse", "HEAD^")
