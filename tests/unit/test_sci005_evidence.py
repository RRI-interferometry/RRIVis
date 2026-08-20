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
APPROVED_STAGE2_SOURCE_SHA: str | None = "5c94d925352b389768e0476079e04e811db996e1"
APPROVED_STAGE2_EVIDENCE_ARTIFACT_SHA256: str | None = "5e3c57eb93c634649e535b02386e1d4211345f23fd851a8a3529480b7c7f1171"
APPROVED_STAGE3_SOURCE_SHA: str | None = "2fa9ce4e76f78fff74bf2e46d67601294bf7c173"
APPROVED_STAGE3_EVIDENCE_ARTIFACT_SHA256: str | None = "a59537cb390b85ce656e0be9fbe13859d65d6395e5db710366d44e5fd388ae04"
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


# === Section 8.1's Stage-3 evidence envelope ==================================
#
# Everything below is appended by ``S3``. It adds the Stage-3 validation and its
# synthetic-document tests and changes no Stage-1 or Stage-2 validation logic,
# constant, synthetic fixture or test, exactly as the Stage-3 envelope requires.


#: Section 8.1's exact Stage-3 top-level key sequence: the common field
#: sequence followed by the five stage arrays, in that order.
STAGE3_KEYS: tuple[str, ...] = (
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
    "efield_file_contracts",
    "basis_conversions",
    "receptor_factorizations",
    "ixr_diagnostics",
    "crossvalidation_comparisons",
)

#: The four frozen Stage-3 convention literals. The first is Section 5.1.1's
#: authored configuration literal; the other three are the derived facets
#: Section 5.2.1 fixes, versioned by that one literal exactly as Stage 2's three
#: squint facets are versioned by ``cotton_uson_exact_v1``.
STAGE3_SCIENTIFIC_CONVENTIONS: dict[str, str] = {
    "efield_normalization": "uvbeam_peak_common_v1",
    "efield_basis_conversion": "uvbeam_theta_phi_chain_tangent_v1",
    "efield_zenith_limit": "north_east_tangent_limit_v1",
    "efield_factorization": "receptor_conjugated_native_efield_v1",
}

#: The Stage-3 envelope supersedes the Stage-1-scoped member rule with exactly
#: these five members: "``claims_not_licensed`` must contain exactly the
#: members", so the retained array is this set and nothing else.
STAGE3_REQUIRED_CLAIMS: tuple[str, ...] = (
    "SCI-005 Stage-3 acceptance",
    "SCI-005 whole-row closure",
    "a station, array-factor, or mutual-coupling response",
    "near-field or Fresnel-regime behavior",
    "an unqualified validation against pyuvsim",
)

#: Section 8.1's Stage-3 solver effects, both of which must appear.
STAGE3_SOLVER_EFFECTS: frozenset[str] = frozenset({"efield_point", "efield_healpix"})

STAGE3_RECEPTOR_BASES: frozenset[str] = frozenset({"linear", "circular"})
STAGE3_OUTPUT_BASES: frozenset[str] = frozenset({"linear_xy", "circular_rl"})

#: The two accepted Stage-3 file-contract probe kinds; every other kind is a
#: rejection.
STAGE3_ACCEPTED_PROBE_KINDS: frozenset[str] = frozenset(
    {"accepted_linear_pair", "accepted_circular_pair"}
)

#: Section 8.1's frozen ``probe_kind -> exception_type`` table for every
#: rejected Stage-3 file-contract kind. ``basis_vector_not_identity`` replaces
#: the retired ``basis_vector_dtype`` and ``basis_vector_degenerate`` kinds.
STAGE3_PROBE_EXCEPTIONS: dict[str, str] = {
    "power_beam": "UnsupportedBeamTypeError",
    "phased_array_antenna": "UnsupportedBeamTypeError",
    "healpix_pixels": "UnsupportedBeamCoordinateError",
    "zenith_single_valued": "UnsupportedBeamCoordinateError",
    "wrap_continuity": "UnsupportedBeamCoordinateError",
    "grid_coverage": "BeamAngularDomainError",
    "vector_dimension": "UnsupportedBeamBasisError",
    "basis_vector_not_identity": "UnsupportedBeamBasisError",
    "basis_vector_complex": "UnsupportedBeamBasisError",
    "feed_pair": "UnsupportedBeamFeedError",
    "feed_pair_receptor_mismatch": "UnsupportedBeamFeedError",
    "feed_angle": "UnsupportedBeamFeedError",
    "derived_orientation": "UnsupportedBeamFeedError",
    "mount": "UnsupportedBeamFeedError",
    "data_dtype": "UnsupportedBeamPrecisionError",
    "extended_precision": "UnsupportedBeamPrecisionError",
    "basis_vector_non_finite": "NonFiniteBeamResponseError",
    "data_non_finite": "NonFiniteBeamResponseError",
    "data_normalization": "BeamNormalizationError",
    "bandpass": "BeamNormalizationError",
    "visible_only_peak": "BeamNormalizationError",
}

#: The complete twenty-three-kind inventory: every one appears at least once.
STAGE3_PROBE_KINDS: frozenset[str] = STAGE3_ACCEPTED_PROBE_KINDS | frozenset(
    STAGE3_PROBE_EXCEPTIONS
)

#: The one byte-frozen load-stage message Stage 3 retains. Its only variable
#: part is the rendered file path, exactly as Stage 2's frozen messages vary
#: only in their rendered placeholders.
STAGE3_FLOAT128_MESSAGE = re.compile(
    r"\ABeamFITS .+: beam precision 'float128' would require complex256, but "
    r"accepted files and pyuvdata interpolation provide at most complex128; "
    r"select beam float32 or float64\.\Z"
)

#: The amended Section 8.1 contract for the two blocks the chain-basis and
#: comparison correction added to every ``crossvalidation_comparisons`` row.
#: ``bounded_quantities`` names the only quantities this comparison bounds,
#: because they are the only ones the adjudicated ``pyradiosky`` local-frame
#: linear mirror leaves untouched; every bound must sit at or below the
#: accepted SCI-007 frame residual.
STAGE3_BOUNDED_QUANTITIES: frozenset[str] = frozenset(
    {"total_intensity", "stokes_v_class"}
)
STAGE3_BOUNDED_QUANTITY_CEILING = 1.9e-3

#: ``reference_frame_mirror`` is the structured record of the open
#: disagreement; its transfer-solve residual is the quantitative claim that the
#: mechanism is complete, checked directly so an unexplained residual cannot
#: hide behind prose.
STAGE3_MIRROR_MECHANISM = "pyradiosky_local_linear_mirror_v1"
STAGE3_MIRROR_CITATIONS: tuple[str, ...] = (
    "pyradiosky/skymodel.py:2667-2676",
    "pyradiosky/utils.py:105-120",
)
STAGE3_MIRROR_TRANSFER_SOLVE_CEILING = 1e-3

#: The amended Section 8.1 freezes the **construction** that produces that
#: residual as well as its ceiling, "because a ceiling without a construction
#: makes two regenerations incomparable" -- and they were not: one adjudication
#: measured ``6.8e-5`` with one construction while the campaign measured
#: ``7.36e-4`` with a different, parameter-free one. The frozen procedure
#: applies ``C[0,1] -> -C[1,0]`` and ``C[1,0] -> -C[0,1]`` -- the local-frame
#: reading of the adjudicated ``chi -> 2 psi - chi`` mirror -- to pyuvsim's own
#: ``local_coherency`` and propagates it through pyuvsim's own beam Jones and
#: fringe. It fits nothing.
STAGE3_MIRROR_CONSTRUCTION = "local_u_negation_in_reference_coherency_v1"

#: ``reassembly_gap`` is the residual between that reassembly with the
#: substitution **disabled** and the comparison tool's own reference path: it
#: is what proves the substitution is the only difference between the two.
STAGE3_MIRROR_REASSEMBLY_GAP_CEILING = 1e-12

#: The one convention mapping the amended Section 5.5 requires to be
#: equivalent, and the two it forbids from being equivalent.
STAGE3_EQUALIZED_CONVENTION = "interpolation_order"
STAGE3_NON_EQUIVALENT_CONVENTIONS: frozenset[str] = frozenset(
    {"east_x_orientation", "stokes_to_coherency_factor"}
)

#: Section 8.1's four Stage-3 conversion oracle kinds; each appears at least
#: once, and ``scalar_subset_control`` is the retained divergence witness rather
#: than an agreement row.
STAGE3_ORACLE_KINDS: frozenset[str] = frozenset(
    {
        "crossed_ideal_dipole",
        "quadrupolar",
        "chain_tangent_mapping",
        "scalar_subset_control",
    }
)
STAGE3_SCALAR_SUBSET_CONTROL = "scalar_subset_control"

#: Section 5.3's three frozen IXR states; each appears at least once.
STAGE3_IXR_STATES: frozenset[str] = frozenset(
    {"nonsingular", "unitary_scaled", "singular"}
)

#: The complete ordered correlation label set ``core/polarization_basis.py``
#: gives each reporting basis, transcribed here because this validator loads
#: only the standard library and may not import the package under review.
STAGE3_CORRELATION_LABELS: dict[str, frozenset[str]] = {
    "linear_xy": frozenset({"XX", "XY", "YX", "YY"}),
    "circular_rl": frozenset({"RR", "RL", "LR", "LL"}),
}

#: Section 5.4's ten required ``output_cases`` rows, as ``case_id -> format``.
#: The basis is carried in the identifier because the common row schema has no
#: basis field and is not changed.
STAGE3_REQUIRED_OUTPUT_CASES: dict[str, str] = {
    "efield_hdf5_circular_rl": "hdf5",
    "efield_hdf5_linear_xy": "hdf5",
    "efield_in_memory_circular_rl": "in_memory",
    "efield_in_memory_linear_xy": "in_memory",
    "efield_measurement_set_circular_rl": "measurement_set",
    "efield_measurement_set_linear_xy": "measurement_set",
    "efield_summary_json_circular_rl": "summary_json",
    "efield_summary_json_linear_xy": "summary_json",
    "efield_uvfits_circular_rl": "uvfits",
    "efield_uvfits_linear_xy": "uvfits",
}

#: The authorable positions of the widened ``normalization`` literal, and the
#: blocks that carry no such field at all.
_FITS_SOURCE_PATH = r"beams\.(?:beam|assignments\[[0-9]+\]\.beam)"
_ANY_BEAMS_PATH = r"beams(?:\.[a-z_]+|\[[0-9]+\])*"

#: Section 8.1's four Stage-3 document-stage rejections, as
#: ``issue_code -> (exception_type, config-path pattern, exact-message pattern)``.
#: Section 5.1.1 pins the two Pydantic codes "by code and path rather than by
#: rendered bytes", because widening the accepted literal necessarily rewrites
#: Pydantic's own rendered message; their message pattern is therefore ``None``
#: and the row records the full rendered message as observed data. The two
#: reused family codes keep their frozen Stage-1 and Stage-2 messages.
STAGE3_DOCUMENT_REJECTIONS: dict[
    str, tuple[str, re.Pattern[str], re.Pattern[str] | None]
] = {
    "literal_error": (
        "ConfigSchemaError",
        re.compile(rf"\A{_FITS_SOURCE_PATH}\.normalization\Z"),
        None,
    ),
    "extra_forbidden": (
        "ConfigSchemaError",
        re.compile(rf"\A{_ANY_BEAMS_PATH}\.normalization\Z"),
        None,
    ),
    "beam.squint.unsupported_beam_family": (
        "UnsupportedConfigError",
        re.compile(r"\Abeams\.squint\Z"),
        re.compile(
            r"\AStage-2 beam squint supports only the analytic beams mode; "
            r"resolved beams mode is .+\.\Z"
        ),
    ),
    "beam.aperture_physics.unsupported_beam_family": (
        "UnsupportedConfigError",
        re.compile(r"\Abeams\.aperture_physics\Z"),
        re.compile(
            r"\AStage-1 aperture physics does not support resolved beam family "
            r".+\.\Z"
        ),
    ),
}

#: Section 7.4's frozen dated cross-validation basename and its directory.
STAGE3_CROSSVALIDATION_DIRECTORY = "output/crossvalidation"
STAGE3_CROSSVALIDATION_BASENAME = re.compile(
    r"\A(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})-sci005-efield-pyuvsim-1\.4\.0\.json\Z"
)

#: The three pinned reference versions every comparison row retains.
STAGE3_CROSSVALIDATION_VERSIONS: dict[str, str] = {
    "reference_package": "pyuvsim",
    "reference_version": "1.4.0",
    "pyuvdata_version": "3.2.1",
}

#: The binary64 agreement budget for the non-commutation recomputation.
NONCOMMUTING_ABS_AGREEMENT = 1e-15

#: The relative agreement budget for the four recomputed IXR quantities.
IXR_RELATIVE_AGREEMENT = 1e-9

#: Section 5.3's frozen fixed relative classification tolerance, and the upper
#: bound a realized ``unitary_scaled`` condition number may reach.
IXR_CLASSIFICATION_RELATIVE = 1e-12
UNITARY_CONDITION_UPPER_BOUND = 1.0 + 2e-12

#: Section 8.1: the two Ludwig-3 basis predicates are judged at this fixed
#: tolerance, because they are predicates on the real ``float64`` conversion
#: matrix rather than on the converted-matrix pair.
BASIS_FIXED_TOLERANCE = 1e-12

#: The extended-width dtype no Stage-3 projection may carry.
FORBIDDEN_STAGE3_DTYPE = "complex256"


# --- Stage-3 shared helpers ---------------------------------------------------


def _conversion_projection(value: Any, path: str) -> dict[str, Any]:
    """A ``numeric_projection`` restricted to the Stage-3 conversion shape."""
    row = _numeric_projection(value, path)
    _string(row["dtype"], f"{path}.dtype", const="complex128")
    shape = list(row["shape"])
    if len(shape) != 3 or shape[1:] != [2, 2]:
        _fail(f"{path}.shape", "must be [S, 2, 2] with S >= 1")
    return row


def _jones_projection(value: Any, path: str) -> dict[str, Any]:
    """A ``numeric_projection`` restricted to the Stage-3 factorization shapes."""
    row = _numeric_projection(value, path)
    _string(row["dtype"], f"{path}.dtype", const="complex128")
    shape = list(row["shape"])
    if shape[-2:] != [2, 2] or len(shape) not in (2, 3):
        _fail(f"{path}.shape", "must be [2, 2] or [S, 2, 2] with S >= 1")
    return row


def _two_by_two_matrix(value: Any, path: str) -> list[list[complex]]:
    """One row-major two-by-two array of ``{real, imag}`` objects."""
    rows = _array(value, path, minimum_length=2)
    if len(rows) != 2:
        _fail(path, "must be exactly two rows")
    matrix: list[list[complex]] = []
    for index, item in enumerate(rows):
        columns = _array(item, f"{path}[{index}]", minimum_length=2)
        if len(columns) != 2:
            _fail(f"{path}[{index}]", "must be exactly two columns")
        matrix.append(
            [
                _complex_pair(entry, f"{path}[{index}][{column}]")
                for column, entry in enumerate(columns)
            ]
        )
    return matrix


def _relative_agreement(
    observed: float, expected: float, path: str, detail: str
) -> None:
    """Require a recorded value to agree with its binary64 recomputation."""
    scale = max(abs(expected), 1.0)
    if abs(observed - expected) > IXR_RELATIVE_AGREEMENT * scale:
        _fail(path, detail)


def _forbid_extended_projections(value: Any, path: str = "$") -> None:
    """Refuse a ``complex256`` projection anywhere in the Stage-3 envelope.

    Section 8.1: the full-efield path accepts only ``complex64`` and
    ``complex128`` native data, pyuvdata interpolation returns ``complex128``,
    and a resolved ``float128`` beam precision is rejected outright, so there
    exists no Stage-3 computation an extended-width projection could
    authenticate.
    """
    if isinstance(value, dict):
        if "c_order_sha256" in value and value.get("dtype") == FORBIDDEN_STAGE3_DTYPE:
            _fail(path, "no Stage-3 projection may carry complex256")
        for key, item in value.items():
            _forbid_extended_projections(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _forbid_extended_projections(item, f"{path}[{index}]")


# --- Section 8.1's Stage-3 rows -----------------------------------------------


def _efield_file_contract(value: Any, path: str) -> dict[str, Any]:
    """One ``efield_file_contracts`` row and its frozen per-kind exception."""
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "probe_kind",
            "outcome",
            "beam_type",
            "antenna_type",
            "pixel_coordinate_system",
            "feed_array",
            "feed_angle_rad",
            "derived_x_orientation",
            "mount_type",
            "data_normalization",
            "basis_vector_dtype",
            "stored_basis_is_identity",
            "native_data_dtype",
            "receptor_basis",
            "receptor_feed_rotation_rad",
            "common_peak_by_frequency",
            "normalization_tolerance",
            "exception_type",
            "exact_message",
            "test_node_id",
            "input_sha256",
            "passed",
        ),
    )
    _string(row["case_id"], f"{path}.case_id")
    _string(row["test_node_id"], f"{path}.test_node_id")
    kind = _string(row["probe_kind"], f"{path}.probe_kind", allowed=STAGE3_PROBE_KINDS)
    outcome = _string(
        row["outcome"], f"{path}.outcome", allowed=frozenset({"accepted", "rejected"})
    )
    for key in (
        "beam_type",
        "antenna_type",
        "pixel_coordinate_system",
        "mount_type",
        "data_normalization",
        "basis_vector_dtype",
        "native_data_dtype",
    ):
        _string(row[key], f"{path}.{key}")
    feeds = _array(row["feed_array"], f"{path}.feed_array", minimum_length=2)
    if len(feeds) != 2:
        _fail(f"{path}.feed_array", "must be exactly two non-empty strings")
    for index, feed in enumerate(feeds):
        _string(feed, f"{path}.feed_array[{index}]")
    angles = _array(row["feed_angle_rad"], f"{path}.feed_angle_rad", minimum_length=2)
    if len(angles) != 2:
        _fail(f"{path}.feed_angle_rad", "must be exactly two numbers")
    for index, angle in enumerate(angles):
        _signed(angle, f"{path}.feed_angle_rad[{index}]")
    orientation = row["derived_x_orientation"]
    if orientation is not None:
        _string(
            orientation,
            f"{path}.derived_x_orientation",
            allowed=frozenset({"east", "north"}),
        )
    _boolean(row["stored_basis_is_identity"], f"{path}.stored_basis_is_identity")
    _string(
        row["receptor_basis"], f"{path}.receptor_basis", allowed=STAGE3_RECEPTOR_BASES
    )
    _open_interval(
        row["receptor_feed_rotation_rad"],
        f"{path}.receptor_feed_rotation_rad",
        lower=-math.pi,
        upper=math.pi,
        closed_upper=True,
    )
    tolerance = _positive(
        row["normalization_tolerance"], f"{path}.normalization_tolerance"
    )
    _string(row["input_sha256"], f"{path}.input_sha256", pattern=SHA256)
    _boolean(row["passed"], f"{path}.passed", const=True)

    peaks = _array(
        row["common_peak_by_frequency"],
        f"{path}.common_peak_by_frequency",
        minimum_length=1,
    )
    frequencies: list[float] = []
    observed_peaks: list[float] = []
    for index, item in enumerate(peaks):
        where = f"{path}.common_peak_by_frequency[{index}]"
        sample = _mapping(item, where, ("frequency_hz", "observed_peak"))
        frequency = _positive(sample["frequency_hz"], f"{where}.frequency_hz")
        if frequencies and frequency <= frequencies[-1]:
            _fail(f"{where}.frequency_hz", "samples are strictly increasing")
        frequencies.append(frequency)
        observed_peaks.append(
            _number(sample["observed_peak"], f"{where}.observed_peak")
        )

    accepted = kind in STAGE3_ACCEPTED_PROBE_KINDS
    if accepted != (outcome == "accepted"):
        _fail(
            f"{path}.outcome",
            "the two accepted_* kinds are accepted and every other kind is rejected",
        )
    if accepted:
        for key in ("exception_type", "exact_message"):
            if row[key] is not None:
                _fail(f"{path}.{key}", "an accepted row carries no error field")
        _string(
            row["basis_vector_dtype"],
            f"{path}.basis_vector_dtype",
            allowed=frozenset({"float32", "float64"}),
        )
        _string(
            row["native_data_dtype"],
            f"{path}.native_data_dtype",
            allowed=frozenset({"complex64", "complex128"}),
        )
        if row["stored_basis_is_identity"] is not True:
            _fail(
                f"{path}.stored_basis_is_identity",
                "an accepted file stores exactly the native identity basis",
            )
        if orientation is None:
            _fail(
                f"{path}.derived_x_orientation",
                "an accepted row records the orientation the file and the resolved "
                "receptor agree on",
            )
        for index, peak in enumerate(observed_peaks):
            if abs(peak - 1.0) > tolerance:
                _fail(
                    f"{path}.common_peak_by_frequency[{index}].observed_peak",
                    "an accepted file is unit-peak over the complete stored grid",
                )
    else:
        for key in ("exception_type", "exact_message"):
            _string(row[key], f"{path}.{key}")
        if row["exception_type"] != STAGE3_PROBE_EXCEPTIONS[kind]:
            _fail(
                f"{path}.exception_type",
                f"kind {kind!r} is frozen to {STAGE3_PROBE_EXCEPTIONS[kind]!r}",
            )
        if kind == "extended_precision" and (
            STAGE3_FLOAT128_MESSAGE.fullmatch(row["exact_message"]) is None
        ):
            _fail(
                f"{path}.exact_message",
                "must be the existing frozen float128 rejection literal",
            )
    if kind == "basis_vector_not_identity" and row["stored_basis_is_identity"]:
        _fail(
            f"{path}.stored_basis_is_identity",
            "a basis_vector_not_identity row records a stored basis that is not "
            "the native identity",
        )
    return row


def _basis_conversion(value: Any, path: str) -> dict[str, Any]:
    """One ``basis_conversions`` row and its residual and control predicates."""
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "oracle_kind",
            "receptor_basis",
            "frequency_hz",
            "probe_azimuth_rad",
            "probe_zenith_angle_rad",
            "expected",
            "observed",
            "max_abs_residual",
            "power_preservation_max_abs_residual",
            "orthogonality_max_abs_residual",
            "wrap_continuity_max_abs_delta",
            "wrap_continuity_bound",
            "zenith_limit_max_abs_delta",
            "atol",
            "rtol",
            "test_node_id",
            "passed",
        ),
    )
    _string(row["case_id"], f"{path}.case_id")
    _string(row["test_node_id"], f"{path}.test_node_id")
    oracle = _string(
        row["oracle_kind"], f"{path}.oracle_kind", allowed=STAGE3_ORACLE_KINDS
    )
    _string(
        row["receptor_basis"], f"{path}.receptor_basis", allowed=STAGE3_RECEPTOR_BASES
    )
    _positive(row["frequency_hz"], f"{path}.frequency_hz")
    azimuth = _array_projection(row["probe_azimuth_rad"], f"{path}.probe_azimuth_rad")
    zenith = _array_projection(
        row["probe_zenith_angle_rad"], f"{path}.probe_zenith_angle_rad"
    )
    for name, projection in (
        ("probe_azimuth_rad", azimuth),
        ("probe_zenith_angle_rad", zenith),
    ):
        _string(projection["dtype"], f"{path}.{name}.dtype", const="float64")
    if azimuth["shape"] != zenith["shape"]:
        _fail(path, "the two probe projections need identical shape")
    expected = _conversion_projection(row["expected"], f"{path}.expected")
    observed = _conversion_projection(row["observed"], f"{path}.observed")
    if expected["dtype"] != observed["dtype"] or expected["shape"] != observed["shape"]:
        _fail(path, "expected and observed projections need identical dtype and shape")
    if list(observed["shape"])[0] != list(azimuth["shape"])[0]:
        _fail(
            f"{path}.observed.shape",
            "the leading extent must be the retained probe count S",
        )
    residual = _number(row["max_abs_residual"], f"{path}.max_abs_residual")
    power = _number(
        row["power_preservation_max_abs_residual"],
        f"{path}.power_preservation_max_abs_residual",
    )
    orthogonality = _number(
        row["orthogonality_max_abs_residual"],
        f"{path}.orthogonality_max_abs_residual",
    )
    wrap = _number(
        row["wrap_continuity_max_abs_delta"], f"{path}.wrap_continuity_max_abs_delta"
    )
    wrap_bound = _number(row["wrap_continuity_bound"], f"{path}.wrap_continuity_bound")
    zenith_limit = _number(
        row["zenith_limit_max_abs_delta"], f"{path}.zenith_limit_max_abs_delta"
    )
    atol = _positive(row["atol"], f"{path}.atol")
    rtol = _number(row["rtol"], f"{path}.rtol")
    _boolean(row["passed"], f"{path}.passed", const=True)

    if power > BASIS_FIXED_TOLERANCE:
        _fail(
            f"{path}.power_preservation_max_abs_residual",
            f"the real conversion matrix preserves power to {BASIS_FIXED_TOLERANCE}",
        )
    if orthogonality > BASIS_FIXED_TOLERANCE:
        _fail(
            f"{path}.orthogonality_max_abs_residual",
            f"the real conversion matrix is orthogonal to {BASIS_FIXED_TOLERANCE}",
        )
    if oracle == STAGE3_SCALAR_SUBSET_CONTROL:
        floor = max(1e-3, 1024.0 * atol)
        for key, measured in (
            ("max_abs_residual", residual),
            ("zenith_limit_max_abs_delta", zenith_limit),
        ):
            if measured < floor:
                _fail(
                    f"{path}.{key}",
                    "the scalar-subset control retains a measured divergence; "
                    f"expected >= {floor}",
                )
        return row
    bound = atol + rtol * float(observed["maximum_abs"])
    if residual > bound:
        _fail(f"{path}.max_abs_residual", "exceeds the retained atol/rtol bound")
    if zenith_limit > bound:
        _fail(
            f"{path}.zenith_limit_max_abs_delta",
            "exceeds the retained atol/rtol bound",
        )
    if wrap > wrap_bound:
        _fail(
            f"{path}.wrap_continuity_max_abs_delta",
            "exceeds its own retained continuity bound",
        )
    return row


def _receptor_factorization(value: Any, path: str) -> dict[str, Any]:
    """One ``receptor_factorizations`` row and its non-commutation recomputation."""
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "receptor_basis",
            "feed_rotation_deg",
            "output_basis",
            "parallactic_angle_rad",
            "frequency_hz",
            "e_matrix",
            "noncommuting_component",
            "j_native",
            "composed_e",
            "factorization_max_abs_residual",
            "chain_order_max_abs_residual",
            "order_control_max_abs_difference",
            "output_basis_max_abs_residual",
            "atol",
            "test_node_id",
            "passed",
        ),
    )
    _string(row["case_id"], f"{path}.case_id")
    _string(row["test_node_id"], f"{path}.test_node_id")
    _string(
        row["receptor_basis"], f"{path}.receptor_basis", allowed=STAGE3_RECEPTOR_BASES
    )
    _string(row["output_basis"], f"{path}.output_basis", allowed=STAGE3_OUTPUT_BASES)
    _signed(row["feed_rotation_deg"], f"{path}.feed_rotation_deg")
    _signed(row["parallactic_angle_rad"], f"{path}.parallactic_angle_rad")
    _positive(row["frequency_hz"], f"{path}.frequency_hz")
    matrix = _two_by_two_matrix(row["e_matrix"], f"{path}.e_matrix")
    component = _number(row["noncommuting_component"], f"{path}.noncommuting_component")
    native = _jones_projection(row["j_native"], f"{path}.j_native")
    composed = _jones_projection(row["composed_e"], f"{path}.composed_e")
    if native["dtype"] != composed["dtype"] or native["shape"] != composed["shape"]:
        _fail(path, "j_native and composed_e need identical dtype and shape")
    # Section 8.1's witness-adequacy guard. This is an obligation on the
    # evidence author's choice of fixture, **not** a theorem: since
    # ``E = C^dagger J_native``, coincidence occurs exactly when J_native's
    # columns lie in the ``+1`` eigenspace of ``C^dagger``, and that eigenspace
    # exists in both bases -- the linear ``C(chi) = P_swap R(chi)`` is a
    # reflection with eigenvalues exactly ``{+1, -1}`` at every chi, and the
    # circular ``C^dagger`` has an isolated unit eigenvalue at ``chi = pi/4``
    # modulo ``2*pi``, an unremarkable ``feed_rotation_deg`` of 45.0. Such a row
    # is legitimate physics; what it cannot be is the retained *witness*, since
    # the mis-projection this rule exists to catch -- retaining ``C @ E``, which
    # *is* J_native, in the field reserved for E -- would be undetectable there.
    if composed["c_order_sha256"] == native["c_order_sha256"]:
        _fail(
            f"{path}.composed_e",
            "repeats j_native byte for byte, so the row is an inadequate "
            "witness rather than an unphysical one: a scenario in which E "
            "coincides with J_native demonstrates nothing about C^dagger "
            "conjugation, and a projection that retained C @ E in the field "
            "reserved for E would be undetectable in it",
        )
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
    output_basis_residual = _number(
        row["output_basis_max_abs_residual"], f"{path}.output_basis_max_abs_residual"
    )
    atol = _positive(row["atol"], f"{path}.atol")
    _boolean(row["passed"], f"{path}.passed", const=True)

    for key, measured in (
        ("factorization_max_abs_residual", factorization),
        ("chain_order_max_abs_residual", chain_order),
        ("output_basis_max_abs_residual", output_basis_residual),
    ):
        if measured > atol:
            _fail(f"{path}.{key}", "exceeds the retained atol")
    recomputed = abs(matrix[0][0] - matrix[1][1]) + abs(matrix[0][1] + matrix[1][0])
    if abs(recomputed - component) > NONCOMMUTING_ABS_AGREEMENT:
        _fail(
            f"{path}.noncommuting_component",
            "disagrees with the binary64 |E00 - E11| + |E01 + E10| recomputation",
        )
    floor = max(1e-3, 1024.0 * atol)
    for key, measured in (
        ("noncommuting_component", component),
        ("order_control_max_abs_difference", order_control),
    ):
        if measured < floor:
            _fail(
                f"{path}.{key}",
                f"a general J_native does not commute with a real rotation; "
                f"expected >= {floor}",
            )
    return row


def _ixr_diagnostic(value: Any, path: str) -> dict[str, Any]:
    """One ``ixr_diagnostics`` row, with Section 5.3's exact state rule."""
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "state",
            "receptor_basis",
            "frequency_hz",
            "probe_altitude_rad",
            "probe_azimuth_rad",
            "j_matrix",
            "sigma_max",
            "sigma_min",
            "condition_number",
            "ixr_linear",
            "ixr_db",
            "leakage_magnitude",
            "test_node_id",
            "passed",
        ),
    )
    _string(row["case_id"], f"{path}.case_id")
    _string(row["test_node_id"], f"{path}.test_node_id")
    state = _string(row["state"], f"{path}.state", allowed=STAGE3_IXR_STATES)
    _string(
        row["receptor_basis"], f"{path}.receptor_basis", allowed=STAGE3_RECEPTOR_BASES
    )
    _positive(row["frequency_hz"], f"{path}.frequency_hz")
    _signed(row["probe_altitude_rad"], f"{path}.probe_altitude_rad")
    _signed(row["probe_azimuth_rad"], f"{path}.probe_azimuth_rad")
    _two_by_two_matrix(row["j_matrix"], f"{path}.j_matrix")
    sigma_max = _number(row["sigma_max"], f"{path}.sigma_max")
    sigma_min = _number(row["sigma_min"], f"{path}.sigma_min")
    _boolean(row["passed"], f"{path}.passed", const=True)
    if sigma_max < sigma_min:
        _fail(f"{path}.sigma_min", "must not exceed sigma_max")

    if sigma_max == 0.0 or sigma_min <= IXR_CLASSIFICATION_RELATIVE * sigma_max:
        recomputed = "singular"
    elif sigma_max - sigma_min <= IXR_CLASSIFICATION_RELATIVE * sigma_max:
        recomputed = "unitary_scaled"
    else:
        recomputed = "nonsingular"
    if state != recomputed:
        _fail(
            f"{path}.state",
            f"Section 5.3's deterministic rule classifies this row {recomputed!r}",
        )

    derived = ("condition_number", "ixr_linear", "ixr_db", "leakage_magnitude")
    if state == "singular":
        for key in derived:
            if row[key] is not None:
                _fail(f"{path}.{key}", "a singular row records null derived fields")
        return row
    if state == "unitary_scaled":
        condition = _number(
            row["condition_number"], f"{path}.condition_number", minimum=None
        )
        if not 1.0 <= condition <= UNITARY_CONDITION_UPPER_BOUND:
            _fail(
                f"{path}.condition_number",
                "a unitary_scaled row retains the realized ratio in "
                f"[1.0, {UNITARY_CONDITION_UPPER_BOUND}]",
            )
        for key in ("ixr_linear", "ixr_db", "leakage_magnitude"):
            if row[key] is not None:
                _fail(
                    f"{path}.{key}",
                    "an infinite IXR is recorded by the literal and a null field, "
                    "never by a non-finite number",
                )
        return row

    values: dict[str, float] = {}
    for key in derived:
        if row[key] is None:
            _fail(f"{path}.{key}", "a nonsingular row records all four quantities")
        values[key] = _number(row[key], f"{path}.{key}", minimum=None)
    condition = sigma_max / sigma_min
    ixr_linear = ((condition + 1.0) / (condition - 1.0)) ** 2
    _relative_agreement(
        values["condition_number"],
        condition,
        f"{path}.condition_number",
        "disagrees with the binary64 sigma_max / sigma_min recomputation",
    )
    _relative_agreement(
        values["ixr_linear"],
        ixr_linear,
        f"{path}.ixr_linear",
        "disagrees with the binary64 ((k + 1) / (k - 1))**2 recomputation",
    )
    _relative_agreement(
        values["ixr_db"],
        10.0 * math.log10(ixr_linear),
        f"{path}.ixr_db",
        "disagrees with the binary64 10 log10 recomputation",
    )
    _relative_agreement(
        values["leakage_magnitude"],
        1.0 / math.sqrt(ixr_linear),
        f"{path}.leakage_magnitude",
        "disagrees with the binary64 1 / sqrt recomputation",
    )
    return row


def _crossvalidation_comparison(value: Any, path: str) -> dict[str, Any]:
    """One ``crossvalidation_comparisons`` row and its artifact bindings."""
    row = _mapping(
        value,
        path,
        (
            "case_id",
            "artifact_path",
            "artifact_sha256",
            "artifact_generated_at_utc",
            "reference_package",
            "reference_version",
            "pyuvdata_version",
            "pyradiosky_version",
            "astropy_version",
            "radiosim_source_sha",
            "input_hashes",
            "convention_mappings",
            "correlation_residuals",
            "bounded_quantities",
            "reference_frame_mirror",
            "output_basis",
            "gating",
            "open_disagreements",
            "test_node_id",
            "passed",
        ),
    )
    _string(row["case_id"], f"{path}.case_id")
    _string(row["test_node_id"], f"{path}.test_node_id")
    artifact_path = _canonical_path(row["artifact_path"], f"{path}.artifact_path")
    _string(row["artifact_sha256"], f"{path}.artifact_sha256", pattern=SHA256)
    stamp = _string(
        row["artifact_generated_at_utc"],
        f"{path}.artifact_generated_at_utc",
        pattern=TIMESTAMP,
    )
    for key, literal in STAGE3_CROSSVALIDATION_VERSIONS.items():
        _string(row[key], f"{path}.{key}", const=literal)
    for key in ("pyradiosky_version", "astropy_version"):
        _string(row[key], f"{path}.{key}")
    _string(row["radiosim_source_sha"], f"{path}.radiosim_source_sha", pattern=GIT_SHA)
    basis = _string(
        row["output_basis"], f"{path}.output_basis", allowed=STAGE3_OUTPUT_BASES
    )
    if row["gating"] is not False:
        _fail(f"{path}.gating", "must be exactly the boolean false")
    _boolean(row["passed"], f"{path}.passed", const=True)

    directory, _separator, basename = artifact_path.rpartition("/")
    if directory != STAGE3_CROSSVALIDATION_DIRECTORY:
        _fail(
            f"{path}.artifact_path",
            f"must live in {STAGE3_CROSSVALIDATION_DIRECTORY}/",
        )
    matched = STAGE3_CROSSVALIDATION_BASENAME.fullmatch(basename)
    if matched is None:
        _fail(
            f"{path}.artifact_path",
            "is not the frozen Section 7.4 dated cross-validation basename",
        )
    if matched.group("date") != stamp[:10]:
        _fail(
            f"{path}.artifact_path",
            "the dated basename must carry the UTC date of artifact_generated_at_utc",
        )

    hashes = _array(row["input_hashes"], f"{path}.input_hashes", minimum_length=4)
    for index, item in enumerate(hashes):
        where = f"{path}.input_hashes[{index}]"
        entry = _mapping(item, where, ("name", "sha256"))
        _string(entry["name"], f"{where}.name")
        _string(entry["sha256"], f"{where}.sha256", pattern=SHA256)
    _sorted_unique([item["name"] for item in hashes], f"{path}.input_hashes")

    mappings = _array(
        row["convention_mappings"], f"{path}.convention_mappings", minimum_length=6
    )
    equivalence: dict[str, bool] = {}
    for index, item in enumerate(mappings):
        where = f"{path}.convention_mappings[{index}]"
        entry = _mapping(
            item, where, ("radiosim_convention", "reference_convention", "equivalent")
        )
        name = _string(entry["radiosim_convention"], f"{where}.radiosim_convention")
        _string(entry["reference_convention"], f"{where}.reference_convention")
        _boolean(entry["equivalent"], f"{where}.equivalent")
        equivalence[name] = entry["equivalent"]
    _sorted_unique(
        [item["radiosim_convention"] for item in mappings],
        f"{path}.convention_mappings",
    )
    # Amended Section 5.5: an unequalized comparison is inadmissible as
    # evidence, so the interpolation mapping must be present and equivalent.
    if equivalence.get(STAGE3_EQUALIZED_CONVENTION) is not True:
        _fail(
            f"{path}.convention_mappings",
            f"must carry {STAGE3_EQUALIZED_CONVENTION!r} with equivalent true",
        )
    # And the adjudicated mechanism makes these two false as written.
    for name in sorted(STAGE3_NON_EQUIVALENT_CONVENTIONS):
        if equivalence.get(name) is True:
            _fail(
                f"{path}.convention_mappings",
                f"must not record {name!r} as equivalent",
            )

    residuals = _array(
        row["correlation_residuals"], f"{path}.correlation_residuals", minimum_length=4
    )
    if len(residuals) != 4:
        _fail(f"{path}.correlation_residuals", "must be exactly four rows")
    labels: list[str] = []
    for index, item in enumerate(residuals):
        where = f"{path}.correlation_residuals[{index}]"
        entry = _mapping(
            item,
            where,
            (
                "correlation",
                "max_abs_residual",
                "max_rel_residual",
                "reference_max_abs",
            ),
        )
        labels.append(_string(entry["correlation"], f"{where}.correlation"))
        for key in ("max_abs_residual", "max_rel_residual", "reference_max_abs"):
            _number(entry[key], f"{where}.{key}")
    _sorted_unique(labels, f"{path}.correlation_residuals")
    if frozenset(labels) != STAGE3_CORRELATION_LABELS[basis]:
        _fail(
            f"{path}.correlation_residuals",
            f"must name the complete {basis!r} label set "
            f"{sorted(STAGE3_CORRELATION_LABELS[basis])}",
        )

    disagreements = _array(row["open_disagreements"], f"{path}.open_disagreements")
    for index, item in enumerate(disagreements):
        _string(item, f"{path}.open_disagreements[{index}]")
    _sorted_unique(disagreements, f"{path}.open_disagreements")

    bounded = _array(
        row["bounded_quantities"], f"{path}.bounded_quantities", minimum_length=2
    )
    quantities: list[str] = []
    for index, item in enumerate(bounded):
        where = f"{path}.bounded_quantities[{index}]"
        entry = _mapping(
            item, where, ("quantity", "max_rel_residual", "bound", "passed")
        )
        quantities.append(
            _string(
                entry["quantity"],
                f"{where}.quantity",
                allowed=STAGE3_BOUNDED_QUANTITIES,
            )
        )
        residual = _number(entry["max_rel_residual"], f"{where}.max_rel_residual")
        bound = _number(entry["bound"], f"{where}.bound")
        if bound > STAGE3_BOUNDED_QUANTITY_CEILING:
            _fail(
                f"{where}.bound",
                "must sit at or below the accepted SCI-007 frame residual "
                f"{STAGE3_BOUNDED_QUANTITY_CEILING!r}",
            )
        _boolean(entry["passed"], f"{where}.passed")
        if entry["passed"] is not (residual <= bound):
            _fail(f"{where}.passed", "must equal max_rel_residual <= bound")
    _sorted_unique(quantities, f"{path}.bounded_quantities")
    if frozenset(quantities) != STAGE3_BOUNDED_QUANTITIES:
        _fail(
            f"{path}.bounded_quantities",
            "must carry exactly "
            f"{sorted(STAGE3_BOUNDED_QUANTITIES)}: these are the only quantities "
            "the Section-5.5 mechanism leaves untouched",
        )

    mirror = _mapping(
        row["reference_frame_mirror"],
        f"{path}.reference_frame_mirror",
        (
            "mechanism",
            "construction",
            "citations",
            "transfer_solve_max_abs_residual",
            "transfer_solve_max_rel_residual",
            "reassembly_gap",
            "affected_correlations",
            "observed_rel_residual_min",
            "observed_rel_residual_max",
        ),
    )
    where = f"{path}.reference_frame_mirror"
    _string(mirror["mechanism"], f"{where}.mechanism", const=STAGE3_MIRROR_MECHANISM)
    _string(
        mirror["construction"],
        f"{where}.construction",
        const=STAGE3_MIRROR_CONSTRUCTION,
    )
    citations = _array(mirror["citations"], f"{where}.citations", minimum_length=2)
    for index, item in enumerate(citations):
        _string(item, f"{where}.citations[{index}]")
    _sorted_unique(citations, f"{where}.citations")
    if not frozenset(STAGE3_MIRROR_CITATIONS) <= frozenset(citations):
        _fail(
            f"{where}.citations",
            f"must name at minimum {list(STAGE3_MIRROR_CITATIONS)}",
        )
    transfer = _number(
        mirror["transfer_solve_max_abs_residual"],
        f"{where}.transfer_solve_max_abs_residual",
    )
    if transfer > STAGE3_MIRROR_TRANSFER_SOLVE_CEILING:
        _fail(
            f"{where}.transfer_solve_max_abs_residual",
            "must not exceed "
            f"{STAGE3_MIRROR_TRANSFER_SOLVE_CEILING!r}: it is the quantitative "
            "claim that the mechanism is complete and sufficient",
        )
    # Deliberately **informational-only**: recorded and type-checked, but with
    # no ceiling, because the denominator is a per-correlation scale that moves
    # with the fixture's Stokes content. The absolute residual is anchored to
    # the comparison's own cube scale and is therefore the meaningful gate.
    _number(
        mirror["transfer_solve_max_rel_residual"],
        f"{where}.transfer_solve_max_rel_residual",
    )
    gap = _number(mirror["reassembly_gap"], f"{where}.reassembly_gap")
    if gap > STAGE3_MIRROR_REASSEMBLY_GAP_CEILING:
        _fail(
            f"{where}.reassembly_gap",
            "must not exceed "
            f"{STAGE3_MIRROR_REASSEMBLY_GAP_CEILING!r}: it is what proves the "
            "frozen substitution is the only difference between the "
            "construction's reassembly and the comparison tool's own reference "
            "path",
        )
    affected = _array(
        mirror["affected_correlations"],
        f"{where}.affected_correlations",
        minimum_length=1,
    )
    for index, item in enumerate(affected):
        _string(item, f"{where}.affected_correlations[{index}]")
    _sorted_unique(affected, f"{where}.affected_correlations")
    if not frozenset(affected) <= frozenset(labels):
        _fail(
            f"{where}.affected_correlations",
            "must be a subset of the row's own correlation labels",
        )
    minimum = _number(
        mirror["observed_rel_residual_min"], f"{where}.observed_rel_residual_min"
    )
    maximum = _number(
        mirror["observed_rel_residual_max"], f"{where}.observed_rel_residual_max"
    )
    if minimum > maximum:
        _fail(f"{where}.observed_rel_residual_min", "must not exceed the maximum")
    # A mirrored correlation can never be silently counted as an agreement.
    for label in affected:
        if not any(entry.startswith(f"{label}:") for entry in disagreements):
            _fail(
                f"{where}.affected_correlations",
                f"{label!r} must appear in open_disagreements",
            )
        if label in frozenset(quantities):
            _fail(
                f"{where}.affected_correlations",
                f"{label!r} must not appear in bounded_quantities",
            )
    return row


def _stage3_solver_case(value: Any, path: str) -> dict[str, Any]:
    """One Stage-3 ``solver_cases`` row.

    The Stage-3 envelope replaces the Stage-1 effect enum outright: every row is
    a full-efield row, so no row carries a diagnostic digest, every row expects
    a visibility change and every row must actually have moved one.
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
    _string(row["effect"], f"{path}.effect", allowed=STAGE3_SOLVER_EFFECTS)
    _string(row["test_node_id"], f"{path}.test_node_id")
    for key in ("input_sha256", "jones_sha256", "visibility_sha256"):
        _string(row[key], f"{path}.{key}", pattern=SHA256)
    if row["diagnostic_sha256"] is not None:
        _fail(f"{path}.diagnostic_sha256", "is null on every Stage-3 solver row")
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


# --- the complete Stage-3 validator -------------------------------------------


def validate_stage3_evidence(document: Any) -> None:
    """Authenticate one Stage-3 evidence document against Section 8.1.

    Pure document validation, standard library only: exact key sets and order,
    the frozen literals, the ``git_sha``/``sha256``/timestamp encodings, JSON
    number and integer distinctions that reject booleans, sorted-unique arrays,
    and every Stage-3 cross-field predicate the envelope names.
    :func:`authenticate_stage3_succession` holds the Git-object ancestry facts
    separately, exactly as Stages 1 and 2 keep repository authentication out of
    their document validators.
    """
    root = _mapping(document, "$", STAGE3_KEYS)
    _string(
        root["schema_version"], "$.schema_version", const="radiosim.sci005.stage3.v1"
    )
    if root["stage"] != 3 or isinstance(root["stage"], bool):
        _fail("$.stage", "must be the integer 3")
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
        tuple(STAGE3_SCIENTIFIC_CONVENTIONS),
    )
    for key, literal in STAGE3_SCIENTIFIC_CONVENTIONS.items():
        _string(conventions[key], f"$.scientific_conventions.{key}", const=literal)

    _forbid_extended_projections(root)

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
    ) in STAGE3_DOCUMENT_REJECTIONS.items():
        if code not in observed_codes:
            _fail("$.rejection_probes", f"must carry the Stage-3 code {code!r}")
        for index, row in enumerate(probes):
            if row["issue_code"] != code:
                continue
            where = f"$.rejection_probes[{index}]"
            if row["exception_type"] != exception:
                _fail(f"{where}.exception_type", f"{code!r} is frozen to {exception!r}")
            if path_pattern.fullmatch(row["config_path"]) is None:
                _fail(f"{where}.config_path", f"is not {code!r}'s frozen path")
            if (
                message_pattern is not None
                and message_pattern.fullmatch(row["exact_message"]) is None
            ):
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
    for index, row in enumerate(parsed_parity):
        by_case.setdefault(row["case_id"], set()).add(row["backend"])
        if (row["real_dtype"], row["complex_dtype"]) == ("float64", "complex128"):
            standard_width.add(row["case_id"])
        where = f"$.backend_parity[{index}]"
        if row["backend"] in {"numpy", "dask"}:
            if row["observed_result_sha256"] != row["reference_result_sha256"]:
                _fail(
                    f"{where}.observed_result_sha256",
                    "NumPy and Dask must be byte-identical at Stage 3",
                )
            if row["max_abs_difference"] != 0.0:
                _fail(
                    f"{where}.max_abs_difference",
                    "a byte-identical backend records an exactly zero difference",
                )
        elif row["max_abs_difference"] > row["atol"] or (
            row["max_rel_difference"] > row["rtol"]
        ):
            _fail(where, "JAX must agree at the retained float64 tolerance")
    for case_id, backends in by_case.items():
        if backends != {"numpy", "jax", "dask"}:
            _fail("$.backend_parity", f"case {case_id!r} is missing a backend")
    if not standard_width:
        _fail(
            "$.backend_parity",
            "at least one full-efield case needs the float64/complex128 pair",
        )

    solver_rows = _array(root["solver_cases"], "$.solver_cases", minimum_length=1)
    parsed_solver = [
        _stage3_solver_case(row, f"$.solver_cases[{index}]")
        for index, row in enumerate(solver_rows)
    ]
    _rows_sorted_by(parsed_solver, "case_id", "$.solver_cases")
    if {row["effect"] for row in parsed_solver} != STAGE3_SOLVER_EFFECTS:
        _fail(
            "$.solver_cases",
            "both efield_point and efield_healpix must appear at least once",
        )

    outputs = _array(root["output_cases"], "$.output_cases", minimum_length=10)
    for index, row in enumerate(outputs):
        _output_case(row, f"$.output_cases[{index}]")
    _rows_sorted_by(outputs, "case_id", "$.output_cases")
    observed_outputs = {row["case_id"]: row["format"] for row in outputs}
    for case_id, required_format in STAGE3_REQUIRED_OUTPUT_CASES.items():
        if case_id not in observed_outputs:
            _fail("$.output_cases", f"the Section 5.4 row {case_id!r} is required")
        if observed_outputs[case_id] != required_format:
            _fail(
                "$.output_cases",
                f"{case_id!r} must carry format {required_format!r}",
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
    scalar_controls: list[dict[str, Any]] = []
    for index, row in enumerate(parsed_fingerprints):
        if row["change_expected"]:
            continue
        where = f"$.fingerprint_diff[{index}]"
        if row["old_raw_cube_sha256"] != row["new_raw_cube_sha256"]:
            _fail(where, "an unchanged workload keeps byte-identical cube bytes")
        workload = row["workload"].lower()
        if "peak" in workload and "fits" in workload:
            scalar_controls.append(row)
    if not scalar_controls:
        _fail(
            "$.fingerprint_diff",
            "the disabled control must include a scalar peak FITS workload whose "
            "old and new digests and cube bytes are equal",
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
    if set(claims) != set(STAGE3_REQUIRED_CLAIMS):
        _fail(
            "$.claims_not_licensed",
            f"must contain exactly {sorted(STAGE3_REQUIRED_CLAIMS)}",
        )

    contracts = _array(
        root["efield_file_contracts"], "$.efield_file_contracts", minimum_length=1
    )
    parsed_contracts = [
        _efield_file_contract(row, f"$.efield_file_contracts[{index}]")
        for index, row in enumerate(contracts)
    ]
    _rows_sorted_by(parsed_contracts, "case_id", "$.efield_file_contracts")
    covered = {row["probe_kind"] for row in parsed_contracts}
    if covered != STAGE3_PROBE_KINDS:
        _fail(
            "$.efield_file_contracts",
            f"missing probe kinds {sorted(STAGE3_PROBE_KINDS - covered)}",
        )

    conversions = _array(
        root["basis_conversions"], "$.basis_conversions", minimum_length=1
    )
    parsed_conversions = [
        _basis_conversion(row, f"$.basis_conversions[{index}]")
        for index, row in enumerate(conversions)
    ]
    _rows_sorted_by(parsed_conversions, "case_id", "$.basis_conversions")
    oracles = {row["oracle_kind"] for row in parsed_conversions}
    if oracles != STAGE3_ORACLE_KINDS:
        _fail(
            "$.basis_conversions",
            f"missing oracle kinds {sorted(STAGE3_ORACLE_KINDS - oracles)}",
        )

    factorizations = _array(
        root["receptor_factorizations"], "$.receptor_factorizations", minimum_length=1
    )
    parsed_factorizations = [
        _receptor_factorization(row, f"$.receptor_factorizations[{index}]")
        for index, row in enumerate(factorizations)
    ]
    _rows_sorted_by(parsed_factorizations, "case_id", "$.receptor_factorizations")
    combinations = {
        (row["receptor_basis"], row["output_basis"]) for row in parsed_factorizations
    }
    required_combinations = {
        (receptor, output)
        for receptor in sorted(STAGE3_RECEPTOR_BASES)
        for output in sorted(STAGE3_OUTPUT_BASES)
    }
    if combinations != required_combinations:
        _fail(
            "$.receptor_factorizations",
            "missing receptor/output basis combinations "
            f"{sorted(required_combinations - combinations)}",
        )
    if not any(
        row["receptor_basis"] == "linear" and row["feed_rotation_deg"] != 0.0
        for row in parsed_factorizations
    ):
        _fail(
            "$.receptor_factorizations",
            "a linear row with a non-zero feed_rotation_deg is required",
        )

    diagnostics = _array(root["ixr_diagnostics"], "$.ixr_diagnostics", minimum_length=1)
    parsed_diagnostics = [
        _ixr_diagnostic(row, f"$.ixr_diagnostics[{index}]")
        for index, row in enumerate(diagnostics)
    ]
    _rows_sorted_by(parsed_diagnostics, "case_id", "$.ixr_diagnostics")
    states = {row["state"] for row in parsed_diagnostics}
    if states != STAGE3_IXR_STATES:
        _fail(
            "$.ixr_diagnostics",
            f"missing IXR states {sorted(STAGE3_IXR_STATES - states)}",
        )

    comparisons = _array(
        root["crossvalidation_comparisons"],
        "$.crossvalidation_comparisons",
        minimum_length=1,
    )
    parsed_comparisons = [
        _crossvalidation_comparison(row, f"$.crossvalidation_comparisons[{index}]")
        for index, row in enumerate(comparisons)
    ]
    _rows_sorted_by(parsed_comparisons, "case_id", "$.crossvalidation_comparisons")
    for index, row in enumerate(parsed_comparisons):
        if row["radiosim_source_sha"] != root["source_sha"]:
            _fail(
                f"$.crossvalidation_comparisons[{index}].radiosim_source_sha",
                "must equal the record's own source_sha",
            )
    named = {
        (row["artifact_path"], row["artifact_sha256"]) for row in parsed_comparisons
    }
    if len(named) != 1:
        _fail(
            "$.crossvalidation_comparisons",
            "every row names the same artifact path and digest; exactly one dated "
            "artifact exists",
        )
    artifact_path, artifact_digest = next(iter(named))
    digests = {row["path"]: row["sha256"] for row in parsed_artifacts}
    if artifact_path not in digests:
        _fail(
            "$.crossvalidation_comparisons",
            f"{artifact_path!r} is not retained in the artifacts array",
        )
    if digests[artifact_path] != artifact_digest:
        _fail(
            "$.crossvalidation_comparisons",
            "the retained cross-validation digest disagrees with its artifacts row",
        )


# --- Section 8.3's three Stage-3 Git-object ancestry facts ---------------------


def _stage3_parent_of(commit: str) -> str:
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


def authenticate_stage3_succession(document: dict[str, Any]) -> None:
    """Authenticate ``R3^ == D3``, ``S3^ == R3`` and ``D3 != R3`` from Git.

    Section 8.3 keeps ``D3`` the unambiguous direct parent of ``R3`` even though
    ``U2 ->* D3`` is starred: the starred edge is the reachability *above*
    ``D3``, which ``tools/sci005_stage3_acceptance.py`` authenticates, while the
    three facts below are the unstarred edges this record asserts. This runs
    only against the real retained artifact; a synthetic document names no
    commit.
    """
    design = document["design_sha"]
    red_test = document["red_test_sha"]
    source = document["source_sha"]
    if design == red_test:
        raise EvidenceSchemaError(
            "$.design_sha: D3 and R3 are the same commit; Section 8.3 requires "
            "R3 to be a distinct child of D3"
        )
    observed_design = _stage3_parent_of(red_test)
    if observed_design != design:
        raise EvidenceSchemaError(
            f"$.design_sha: R3^ is {observed_design}, not the recorded {design}"
        )
    observed_red = _stage3_parent_of(source)
    if observed_red != red_test:
        raise EvidenceSchemaError(
            f"$.red_test_sha: S3^ is {observed_red}, not the recorded {red_test}"
        )


# --- the Stage-3 synthetic fixture --------------------------------------------


STAGE3_CROSSVALIDATION_ARTIFACT = (
    "output/crossvalidation/2026-08-19-sci005-efield-pyuvsim-1.4.0.json"
)


def _stage3_projection(shape: list[int], digest: str = "0" * 64) -> dict[str, Any]:
    return {
        "dtype": "complex128",
        "shape": shape,
        "c_order_sha256": digest,
        "minimum_abs": 0.0,
        "maximum_abs": 1.0,
    }


def _stage3_probe_projection(extent: int) -> dict[str, Any]:
    return {
        "dtype": "float64",
        "shape": [extent],
        "c_order_sha256": "0" * 64,
        "minimum": 0.0,
        "maximum": 1.0,
    }


def _stage3_matrix() -> list[list[dict[str, float]]]:
    """One deliberately non-commuting ``E``: ``|E00-E11| + |E01+E10| == 1.8``."""
    return [
        [{"real": 1.0, "imag": 0.0}, {"real": 0.5, "imag": 0.0}],
        [{"real": 0.5, "imag": 0.0}, {"real": 0.2, "imag": 0.0}],
    ]


def _stage3_noncommuting_component() -> float:
    matrix = [
        [complex(cell["real"], cell["imag"]) for cell in row]
        for row in _stage3_matrix()
    ]
    return abs(matrix[0][0] - matrix[1][1]) + abs(matrix[0][1] + matrix[1][0])


def _efield_contract_row(kind: str, node: str) -> dict[str, Any]:
    """One self-consistent ``efield_file_contracts`` row for one probe kind."""
    accepted = kind in STAGE3_ACCEPTED_PROBE_KINDS
    circular = kind == "accepted_circular_pair"
    message = "an offending metadata value was observed"
    if kind == "extended_precision":
        message = (
            "BeamFITS /tmp/beam.beamfits: beam precision 'float128' would require "
            "complex256, but accepted files and pyuvdata interpolation provide at "
            "most complex128; select beam float32 or float64."
        )
    return {
        "case_id": kind,
        "probe_kind": kind,
        "outcome": "accepted" if accepted else "rejected",
        "beam_type": "efield",
        "antenna_type": "simple",
        "pixel_coordinate_system": "az_za",
        "feed_array": ["r", "l"] if circular else ["x", "y"],
        "feed_angle_rad": [0.0, 0.0] if circular else [math.pi / 2.0, 0.0],
        "derived_x_orientation": "east" if accepted else None,
        "mount_type": "fixed",
        "data_normalization": "peak",
        "basis_vector_dtype": "float64",
        "stored_basis_is_identity": kind != "basis_vector_not_identity",
        "native_data_dtype": "complex128",
        "receptor_basis": "circular" if circular else "linear",
        "receptor_feed_rotation_rad": 0.0,
        "common_peak_by_frequency": [
            {"frequency_hz": 1.0e8, "observed_peak": 1.0},
            {"frequency_hz": 1.5e8, "observed_peak": 1.0},
        ],
        "normalization_tolerance": 1e-6,
        "exception_type": None if accepted else STAGE3_PROBE_EXCEPTIONS[kind],
        "exact_message": None if accepted else message,
        "test_node_id": node,
        "input_sha256": "0" * 64,
        "passed": True,
    }


def _basis_conversion_row(oracle: str, node: str) -> dict[str, Any]:
    control = oracle == STAGE3_SCALAR_SUBSET_CONTROL
    return {
        "case_id": oracle,
        "oracle_kind": oracle,
        "receptor_basis": "circular" if oracle == "quadrupolar" else "linear",
        "frequency_hz": 1.5e8,
        "probe_azimuth_rad": _stage3_probe_projection(3),
        "probe_zenith_angle_rad": _stage3_probe_projection(3),
        "expected": _stage3_projection([3, 2, 2]),
        "observed": _stage3_projection([3, 2, 2]),
        "max_abs_residual": 0.5 if control else 0.0,
        "power_preservation_max_abs_residual": 0.0,
        "orthogonality_max_abs_residual": 0.0,
        "wrap_continuity_max_abs_delta": 0.0,
        "wrap_continuity_bound": 1e-12,
        "zenith_limit_max_abs_delta": 0.5 if control else 0.0,
        "atol": 1e-12,
        "rtol": 1e-10,
        "test_node_id": node,
        "passed": True,
    }


def _receptor_factorization_row(
    case_id: str, receptor: str, output: str, rotation: float, node: str
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "receptor_basis": receptor,
        "feed_rotation_deg": rotation,
        "output_basis": output,
        "parallactic_angle_rad": 0.4,
        "frequency_hz": 1.5e8,
        "e_matrix": _stage3_matrix(),
        "noncommuting_component": _stage3_noncommuting_component(),
        # Distinct placeholder digests: the two fields hold *different*
        # quantities, and Section 8.1's witness-adequacy guard now requires a
        # retained row to say so. The shared ``"0" * 64`` placeholder this
        # builder used would make every synthetic document an inadequate
        # witness, which is the consequence the fifth re-cut recorded.
        "j_native": _stage3_projection([2, 2], digest="1" * 64),
        "composed_e": _stage3_projection([2, 2], digest="2" * 64),
        "factorization_max_abs_residual": 0.0,
        "chain_order_max_abs_residual": 0.0,
        "order_control_max_abs_difference": 0.5,
        "output_basis_max_abs_residual": 0.0,
        "atol": 1e-12,
        "test_node_id": node,
        "passed": True,
    }


def _ixr_row(case_id: str, state: str, node: str) -> dict[str, Any]:
    sigma_max, sigma_min = {
        "nonsingular": (2.0, 1.0),
        "unitary_scaled": (1.0, 1.0),
        "singular": (1.0, 0.0),
    }[state]
    condition: float | None = None
    ixr_linear: float | None = None
    ixr_db: float | None = None
    leakage: float | None = None
    if state == "unitary_scaled":
        condition = sigma_max / sigma_min
    elif state == "nonsingular":
        condition = sigma_max / sigma_min
        ixr_linear = ((condition + 1.0) / (condition - 1.0)) ** 2
        ixr_db = 10.0 * math.log10(ixr_linear)
        leakage = 1.0 / math.sqrt(ixr_linear)
    return {
        "case_id": case_id,
        "state": state,
        "receptor_basis": "linear",
        "frequency_hz": 1.5e8,
        "probe_altitude_rad": 1.2,
        "probe_azimuth_rad": 0.7,
        "j_matrix": _stage3_matrix(),
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "condition_number": condition,
        "ixr_linear": ixr_linear,
        "ixr_db": ixr_db,
        "leakage_magnitude": leakage,
        "test_node_id": node,
        "passed": True,
    }


def _stage3_rejection_probe(
    case_id: str, code: str, config_path: str, message: str, node: str
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "config_path": config_path,
        "exception_type": STAGE3_DOCUMENT_REJECTIONS[code][0],
        "issue_code": code,
        "exact_message": message,
        "test_node_id": node,
        "input_sha256": "0" * 64,
        "passed": True,
    }


def synthetic_stage3_document() -> dict[str, Any]:
    """One minimal document that satisfies every Section 8.1 Stage-3 rule."""
    digest = "0" * 64
    node = "tests/unit/test_core/test_sci005_full_efield.py::case"
    io_node = "tests/unit/test_io/test_sci005_beam_config.py::case"
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
    crossvalidation_digest = "e" * 64
    document = {
        "schema_version": "radiosim.sci005.stage3.v1",
        "stage": 3,
        "status": "candidate",
        "generated_at_utc": "2026-08-19T00:00:00Z",
        "design_sha": "a" * 40,
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
        "scientific_conventions": dict(STAGE3_SCIENTIFIC_CONVENTIONS),
        "config_cases": [
            {
                "case_id": "accepted_full_efield_literal",
                "test_node_id": io_node,
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
                "case_id": "rejected_unknown_literal",
                "test_node_id": io_node,
                "input_sha256": digest,
                "expected_outcome": "rejected",
                "observed_outcome": "rejected",
                "resolved_scientific_sha256": None,
                "exception_type": "ConfigSchemaError",
                "issue_code": "literal_error",
                "exact_message": "Input should be 'peak' or 'uvbeam_peak_common_v1'",
                "passed": True,
            },
        ],
        "analytic_invariants": [
            {
                "case_id": "chain_tangent_mapping",
                "invariant_id": "chain_basis_conversion",
                "backend": "numpy",
                "test_node_id": node,
                "input_manifest_sha256": digest,
                "expected": _stage3_projection([3, 2, 2]),
                "observed": _stage3_projection([3, 2, 2]),
                "max_abs_residual": 1e-16,
                "max_rel_residual": 1e-16,
                "atol": 1e-12,
                "rtol": 1e-10,
                "passed": True,
            }
        ],
        "rejection_probes": [
            _stage3_rejection_probe(
                "aperture_physics_family",
                "beam.aperture_physics.unsupported_beam_family",
                "beams.aperture_physics",
                "Stage-1 aperture physics does not support resolved beam family "
                "'fits'.",
                io_node,
            ),
            _stage3_rejection_probe(
                "normalization_on_analytic_model",
                "extra_forbidden",
                "beams.model.normalization",
                "Extra inputs are not permitted",
                io_node,
            ),
            _stage3_rejection_probe(
                "squint_against_fits",
                "beam.squint.unsupported_beam_family",
                "beams.squint",
                "Stage-2 beam squint supports only the analytic beams mode; "
                "resolved beams mode is 'shared_fits'.",
                io_node,
            ),
            _stage3_rejection_probe(
                "unknown_normalization_literal",
                "literal_error",
                "beams.beam.normalization",
                "Input should be 'peak' or 'uvbeam_peak_common_v1'",
                io_node,
            ),
        ],
        "backend_parity": [
            {
                "case_id": "efield_point",
                "backend": backend,
                "actual_device": "cpu",
                "real_dtype": "float64",
                "complex_dtype": "complex128",
                "input_sha256": digest,
                "reference_result_sha256": digest,
                "observed_result_sha256": ("d" * 64 if backend == "jax" else digest),
                "max_abs_difference": 1e-15 if backend == "jax" else 0.0,
                "max_rel_difference": 1e-15 if backend == "jax" else 0.0,
                "atol": 1e-12,
                "rtol": 1e-10,
                "passed": True,
            }
            for backend in ("dask", "jax", "numpy")
        ],
        "solver_cases": [
            {
                "case_id": "efield_healpix_case",
                "effect": "efield_healpix",
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
                "case_id": "efield_point_case",
                "effect": "efield_point",
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
                "case_id": case_id,
                "format": row_format,
                "writer_test_node_id": node,
                "reader_test_node_id": None if row_format == "in_memory" else node,
                "artifact_sha256": None if row_format == "in_memory" else digest,
                "in_memory_sha256": digest,
                "observed_projection_sha256": digest,
                "roundtrip_max_abs_difference": (
                    None if row_format == "in_memory" else 0.0
                ),
                "tolerance": None if row_format == "in_memory" else 1e-12,
                "passed": True,
            }
            for case_id, row_format in sorted(STAGE3_REQUIRED_OUTPUT_CASES.items())
        ],
        "fingerprint_diff": [
            {
                "environment": "default",
                "workload": "point_efield",
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
                "workload": "point_scalar_peak_fits",
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
                "path": "docs/development/sci005_stage3_evidence.schema.json",
                "sha256": digest,
                "media_type": "application/schema+json",
                "role": "schema",
            },
            {
                "path": STAGE3_CROSSVALIDATION_ARTIFACT,
                "sha256": crossvalidation_digest,
                "media_type": "application/json",
                "role": "output",
            },
        ],
        "limitations": [
            "Backend parity covers the CPU-only JAX build and Dask over NumPy; "
            "no accelerator device was exercised, consistent with PERF-001."
        ],
        "claims_not_licensed": sorted(STAGE3_REQUIRED_CLAIMS),
        "efield_file_contracts": [
            _efield_contract_row(kind, node) for kind in sorted(STAGE3_PROBE_KINDS)
        ],
        "basis_conversions": [
            _basis_conversion_row(oracle, node)
            for oracle in sorted(STAGE3_ORACLE_KINDS)
        ],
        "receptor_factorizations": [
            _receptor_factorization_row(
                "circular_receptor_circular_output",
                "circular",
                "circular_rl",
                0.0,
                node,
            ),
            _receptor_factorization_row(
                "circular_receptor_linear_output", "circular", "linear_xy", 0.0, node
            ),
            _receptor_factorization_row(
                "linear_receptor_circular_output", "linear", "circular_rl", 30.0, node
            ),
            _receptor_factorization_row(
                "linear_receptor_linear_output", "linear", "linear_xy", 0.0, node
            ),
        ],
        "ixr_diagnostics": [
            _ixr_row("ixr_nonsingular", "nonsingular", node),
            _ixr_row("ixr_singular", "singular", node),
            _ixr_row("ixr_unitary_scaled", "unitary_scaled", node),
        ],
        "crossvalidation_comparisons": [
            {
                "case_id": "efield_pyuvsim_linear_xy",
                "artifact_path": STAGE3_CROSSVALIDATION_ARTIFACT,
                "artifact_sha256": crossvalidation_digest,
                "artifact_generated_at_utc": "2026-08-19T00:00:00Z",
                "reference_package": "pyuvsim",
                "reference_version": "1.4.0",
                "pyuvdata_version": "3.2.1",
                "pyradiosky_version": "1.1.0",
                "astropy_version": "7.0.0",
                "radiosim_source_sha": "c" * 40,
                "input_hashes": [
                    {"name": "antenna_layout", "sha256": digest},
                    {"name": "observation_specification", "sha256": digest},
                    {"name": "sky_model", "sha256": digest},
                    {"name": "uvbeam_file", "sha256": digest},
                ],
                "convention_mappings": [
                    {
                        "radiosim_convention": "beam_normalization",
                        "reference_convention": "uvbeam peak normalization",
                        "equivalent": True,
                    },
                    {
                        "radiosim_convention": "chain_sky_tangent_basis",
                        "reference_convention": "correctly paired (theta, phi_uv)",
                        "equivalent": True,
                    },
                    {
                        "radiosim_convention": "east_x_orientation",
                        "reference_convention": "pyradiosky local linear mirror",
                        "equivalent": False,
                    },
                    {
                        "radiosim_convention": "fringe_sign",
                        "reference_convention": "conjugated fringe exponent",
                        "equivalent": False,
                    },
                    {
                        "radiosim_convention": "interpolation_order",
                        "reference_convention": (
                            "BeamList spline_interp_opts {'kx': 1, 'ky': 1, 's': 0}"
                        ),
                        "equivalent": True,
                    },
                    {
                        "radiosim_convention": "stokes_to_coherency_factor",
                        "reference_convention": "pyradiosky (South, East) frame pair",
                        "equivalent": False,
                    },
                ],
                "correlation_residuals": [
                    {
                        "correlation": label,
                        "max_abs_residual": 1e-12,
                        "max_rel_residual": 1e-12,
                        "reference_max_abs": 1.0,
                    }
                    for label in sorted(STAGE3_CORRELATION_LABELS["linear_xy"])
                ],
                "bounded_quantities": [
                    {
                        "quantity": "stokes_v_class",
                        "max_rel_residual": 4e-4,
                        "bound": 1.9e-3,
                        "passed": True,
                    },
                    {
                        "quantity": "total_intensity",
                        "max_rel_residual": 4e-4,
                        "bound": 1.9e-3,
                        "passed": True,
                    },
                ],
                "reference_frame_mirror": {
                    "mechanism": STAGE3_MIRROR_MECHANISM,
                    "construction": STAGE3_MIRROR_CONSTRUCTION,
                    "citations": list(STAGE3_MIRROR_CITATIONS),
                    "transfer_solve_max_abs_residual": 7.36e-4,
                    "transfer_solve_max_rel_residual": 3.52e-4,
                    "reassembly_gap": 5.03e-14,
                    "affected_correlations": ["XY", "YX"],
                    "observed_rel_residual_min": 4.16e-2,
                    "observed_rel_residual_max": 1.024e-1,
                },
                "output_basis": "linear_xy",
                "gating": False,
                "open_disagreements": [
                    "XY: mirrored by the recorded pyradiosky local linear mirror",
                    "YX: mirrored by the recorded pyradiosky local linear mirror",
                ],
                "test_node_id": (
                    "tests/crossvalidation/test_sci005_efield_pyuvsim.py::case"
                ),
                "passed": True,
            }
        ],
    }
    return {key: document[key] for key in STAGE3_KEYS}


# --- Section 7.5: the Stage-3 S/E state ---------------------------------------


def test_the_stage3_artifact_and_its_null_sentinels_agree() -> None:
    """At ``S3`` the artifact is absent; at ``E3`` it validates completely.

    This is the Stage-3 half of Section 7.5's ``S``/``E`` rule. It is a separate
    test rather than a change to the Stage-1 parametrized one, because ``S3``
    changes no Stage-1 or Stage-2 validator byte.
    """
    source, digest = STAGE_CONSTANTS[3]
    if source is None or digest is None:
        assert source is None and digest is None, (
            "the two approved constants for one stage move together"
        )
        assert not artifact_path(3).exists()
        return
    assert GIT_SHA.fullmatch(source)
    assert SHA256.fullmatch(digest)
    assert artifact_path(3).is_file()
    payload = artifact_path(3).read_bytes()
    import hashlib

    assert hashlib.sha256(payload).hexdigest() == digest
    document = json.loads(payload.decode("utf-8"))
    validate_stage3_evidence(document)
    assert document["source_sha"] == source
    authenticate_stage3_succession(document)


def test_the_stage3_schema_transcription_and_the_validator_agree() -> None:
    """The normative Stage-3 transcription and this validator pin the same keys."""
    schema = json.loads(schema_path(3).read_text(encoding="utf-8"))
    assert tuple(schema["properties"]) == STAGE3_KEYS
    assert set(schema["required"]) == set(STAGE3_KEYS)
    assert schema["additionalProperties"] is False
    conventions = schema["properties"]["scientific_conventions"]["properties"]
    assert {key: value["const"] for key, value in conventions.items()} == (
        STAGE3_SCIENTIFIC_CONVENTIONS
    )
    assert schema["properties"]["stage"]["const"] == 3
    assert schema["properties"]["schema_version"]["const"] == (
        "radiosim.sci005.stage3.v1"
    )
    assert schema["properties"]["evidence_sha"] == {"type": "null"}
    assert set(schema["$defs"]["solver_case"]["properties"]["effect"]["enum"]) == (
        STAGE3_SOLVER_EFFECTS
    )
    assert (
        set(schema["$defs"]["efield_file_contract"]["properties"]["probe_kind"]["enum"])
        == STAGE3_PROBE_KINDS
    )
    assert (
        set(schema["$defs"]["basis_conversion"]["properties"]["oracle_kind"]["enum"])
        == STAGE3_ORACLE_KINDS
    )
    assert (
        set(schema["$defs"]["ixr_diagnostic"]["properties"]["state"]["enum"])
        == STAGE3_IXR_STATES
    )
    assert schema["$defs"]["crossvalidation_comparison"]["properties"]["gating"] == (
        {"const": False}
    )
    # Section 8.1: no ``complex256`` projection appears anywhere in this envelope.
    assert (
        FORBIDDEN_STAGE3_DTYPE
        not in (schema["$defs"]["numeric_projection"]["properties"]["dtype"]["enum"])
    )


def test_the_generator_declares_the_five_stage3_measurement_keys() -> None:
    """Section 8.1: Stage 3 appends exactly those five arrays, in that order."""
    stage_specific = (
        "efield_file_contracts",
        "basis_conversions",
        "receptor_factorizations",
        "ixr_diagnostics",
        "crossvalidation_comparisons",
    )
    assert STAGE3_KEYS[-5:] == stage_specific
    assert STAGE3_KEYS[:-5] == STAGE1_KEYS[:-3]

    source = (REPOSITORY_ROOT / GENERATOR).read_text(encoding="utf-8")
    assert "\nSTAGE3_MEASUREMENT_KEYS" in source, (
        f"{GENERATOR} must declare STAGE3_MEASUREMENT_KEYS"
    )
    body = source.split("\nSTAGE3_MEASUREMENT_KEYS", 1)[1].split(")", 1)[0]
    declared = tuple(re.findall(r'"([a-z0-9_]+)"', body))
    assert declared == stage_specific


def test_the_generator_registers_the_stage3_measurement_table_entry() -> None:
    """Section 8.1: ``S3`` adds the Stage-3 entry to the stage-keyed table."""
    source = (REPOSITORY_ROOT / GENERATOR).read_text(encoding="utf-8")
    body = source.split("\nSTAGE_MEASUREMENT_KEYS", 1)[1].split("}", 1)[0]
    assert "3: STAGE3_MEASUREMENT_KEYS" in body
    assert "1: STAGE1_MEASUREMENT_KEYS" in body
    assert "2: STAGE2_MEASUREMENT_KEYS" in body


def test_the_generator_declares_the_stage3_crossvalidation_input_kind() -> None:
    """Section 8.1's one conditional artifact-input kind and its frozen schema."""
    source = (REPOSITORY_ROOT / GENERATOR).read_text(encoding="utf-8")
    assert 'STAGE3_CROSSVALIDATION_INPUT_KIND = "stage3_crossvalidation_temp"' in source
    assert (
        'STAGE3_CROSSVALIDATION_SCHEMA = "radiosim.sci005.stage3-crossvalidation.v1"'
        in source
    )
    assert (
        f'STAGE3_CROSSVALIDATION_DIRECTORY = "{STAGE3_CROSSVALIDATION_DIRECTORY}"'
        in (source)
    )
    body = source.split("\nSTAGE3_CROSSVALIDATION_KEYS", 1)[1].split(")", 1)[0]
    assert tuple(re.findall(r'"([a-z0-9_]+)"', body)) == (
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
    versions = source.split("\nSTAGE3_CROSSVALIDATION_VERSIONS", 1)[1].split("}", 1)[0]
    for key, literal in STAGE3_CROSSVALIDATION_VERSIONS.items():
        assert f'"{key}": "{literal}"' in versions


def _generator_probe(body: str) -> subprocess.CompletedProcess[str]:
    """Exercise the stdlib-only generator out of process.

    This validator may not import the tool directly:
    :func:`test_this_validator_loads_only_the_standard_library` scans every
    ``import`` statement in this file, and the Stage-2 tests already set the
    precedent of reading the generator's source or driving it in a subprocess.
    """
    return subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, 'tools')\n" + body],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


_TEMP_KIND_ROWS = (
    "rows = [{'path': %r, 'input_kind': 'stage3_crossvalidation_temp',\n"
    "         'input_path': %r, 'media_type': 'application/json',\n"
    "         'role': 'output'}]\n"
)


def test_the_temporary_crossvalidation_kind_is_refused_before_stage_three() -> None:
    """Section 8.1: the kind is legal exactly once, only at Stage 3."""
    completed = _generator_probe(
        "import sci005_stage_evidence as m\n"
        + _TEMP_KIND_ROWS % (STAGE3_CROSSVALIDATION_ARTIFACT, "/tmp/absent.json")
        + "try:\n"
        "    m.build_artifacts(rows, 2, {}, 'c' * 40)\n"
        "except m.EvidenceError as error:\n"
        "    print(error)\n"
        "else:\n"
        "    raise SystemExit('the temporary kind was accepted at stage 2')\n"
    )
    assert completed.returncode == 0, completed.stderr
    assert "stage 3" in completed.stdout


def test_a_crossvalidation_input_inside_the_repository_is_refused() -> None:
    """Section 8.1: the temporary input resolves outside repository root."""
    completed = _generator_probe(
        "import sci005_stage_evidence as m\n"
        + _TEMP_KIND_ROWS
        % (STAGE3_CROSSVALIDATION_ARTIFACT, str(REPOSITORY_ROOT / "pyproject.toml"))
        + "try:\n"
        "    m.build_artifacts(rows, 3, {}, 'c' * 40)\n"
        "except m.EvidenceError as error:\n"
        "    print(error)\n"
        "else:\n"
        "    raise SystemExit('an in-repository temporary input was accepted')\n"
    )
    assert completed.returncode == 0, completed.stderr
    assert "outside the repository" in completed.stdout


# --- Stage-3 rejection classes -------------------------------------------------


def test_a_complete_synthetic_stage3_document_validates() -> None:
    validate_stage3_evidence(synthetic_stage3_document())


@pytest.mark.parametrize(
    "key",
    [
        "schema_version",
        "scientific_conventions",
        "efield_file_contracts",
        "basis_conversions",
        "receptor_factorizations",
        "ixr_diagnostics",
        "crossvalidation_comparisons",
    ],
)
def test_a_missing_stage3_top_level_key_is_rejected(key: str) -> None:
    document = synthetic_stage3_document()
    del document[key]
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_an_unknown_stage3_top_level_key_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["squint_geometries"] = []
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_a_reordered_stage3_top_level_key_sequence_is_rejected() -> None:
    document = synthetic_stage3_document()
    reordered = {key: document[key] for key in reversed(STAGE3_KEYS)}
    with pytest.raises(EvidenceSchemaError, match="declared order"):
        validate_stage3_evidence(reordered)


def test_a_stage2_schema_version_on_a_stage3_document_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["schema_version"] = "radiosim.sci005.stage2.v1"
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_a_stage3_document_declaring_stage_two_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["stage"] = 2
    with pytest.raises(EvidenceSchemaError, match="integer 3"):
        validate_stage3_evidence(document)


def test_a_boolean_stage3_stage_number_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["stage"] = True
    with pytest.raises(EvidenceSchemaError, match="integer 3"):
        validate_stage3_evidence(document)


def test_a_non_null_stage3_evidence_sha_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["evidence_sha"] = "c" * 40
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_a_stage2_convention_literal_on_a_stage3_document_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["scientific_conventions"]["efield_factorization"] = (
        "receptor_conjugated_native_diagonal_v1"
    )
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_an_extra_stage3_scientific_convention_key_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["scientific_conventions"]["squint_frequency_law"] = "cotton_uson_exact_v1"
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_a_complex256_projection_anywhere_is_rejected() -> None:
    """Section 8.1: no Stage-3 computation can be performed at extended width."""
    document = synthetic_stage3_document()
    document["analytic_invariants"][0]["observed"]["dtype"] = "complex256"
    with pytest.raises(EvidenceSchemaError, match="complex256"):
        validate_stage3_evidence(document)


def test_the_stage2_claim_member_set_does_not_satisfy_stage3() -> None:
    document = synthetic_stage3_document()
    document["claims_not_licensed"] = sorted(
        [
            "SCI-005 Stage-2 acceptance",
            "SCI-005 Stage 3",
            "SCI-005 whole-row closure",
            "a full cross-polar or measured-efield beam response",
        ]
    )
    with pytest.raises(EvidenceSchemaError, match="exactly"):
        validate_stage3_evidence(document)


def test_a_missing_stage3_claim_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["claims_not_licensed"] = [
        claim
        for claim in document["claims_not_licensed"]
        if claim != "an unqualified validation against pyuvsim"
    ]
    with pytest.raises(EvidenceSchemaError, match="exactly"):
        validate_stage3_evidence(document)


def test_an_extra_stage3_claim_is_rejected() -> None:
    """The Stage-3 rule is "exactly the members", not "at least"."""
    document = synthetic_stage3_document()
    document["claims_not_licensed"] = sorted(
        [*document["claims_not_licensed"], "a whole other physics"]
    )
    with pytest.raises(EvidenceSchemaError, match="exactly"):
        validate_stage3_evidence(document)


def test_a_stage2_solver_effect_is_rejected_at_stage3() -> None:
    document = synthetic_stage3_document()
    document["solver_cases"][0]["effect"] = "squint_healpix"
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_a_missing_stage3_solver_effect_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["solver_cases"] = document["solver_cases"][:1]
    with pytest.raises(EvidenceSchemaError, match="efield_point"):
        validate_stage3_evidence(document)


def test_a_stage3_solver_row_carrying_a_diagnostic_digest_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["solver_cases"][0]["diagnostic_sha256"] = "0" * 64
    with pytest.raises(EvidenceSchemaError, match="null on every"):
        validate_stage3_evidence(document)


def test_a_stage3_solver_row_that_moved_no_visibility_element_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["solver_cases"][0]["visibility_changed_element_count"] = 0
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_a_dask_parity_row_that_is_not_byte_identical_is_rejected() -> None:
    """Section 8.1: NumPy and Dask must be byte-identical at Stage 3."""
    document = synthetic_stage3_document()
    document["backend_parity"][0]["observed_result_sha256"] = "f" * 64
    with pytest.raises(EvidenceSchemaError, match="byte-identical"):
        validate_stage3_evidence(document)


def test_a_dask_parity_row_with_a_nonzero_difference_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["backend_parity"][0]["max_abs_difference"] = 1e-18
    with pytest.raises(EvidenceSchemaError, match="exactly zero"):
        validate_stage3_evidence(document)


def test_a_jax_parity_row_beyond_its_tolerance_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["backend_parity"]:
        if row["backend"] == "jax":
            row["max_abs_difference"] = 1.0
    with pytest.raises(EvidenceSchemaError, match="float64 tolerance"):
        validate_stage3_evidence(document)


def test_a_stage3_parity_case_missing_a_backend_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["backend_parity"] = document["backend_parity"][:2]
    with pytest.raises(EvidenceSchemaError, match="missing a backend"):
        validate_stage3_evidence(document)


def test_a_missing_required_stage3_output_row_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["output_cases"] = [
        row
        for row in document["output_cases"]
        if row["case_id"] != "efield_uvfits_circular_rl"
    ]
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_a_required_stage3_output_row_with_the_wrong_format_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["output_cases"]:
        if row["case_id"] == "efield_hdf5_linear_xy":
            row["format"] = "summary_json"
    with pytest.raises(EvidenceSchemaError, match="must carry format"):
        validate_stage3_evidence(document)


def test_a_missing_stage3_rejection_probe_code_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["rejection_probes"] = [
        row
        for row in document["rejection_probes"]
        if row["issue_code"] != "literal_error"
    ]
    with pytest.raises(EvidenceSchemaError, match="literal_error"):
        validate_stage3_evidence(document)


def test_a_stage3_rejection_probe_with_a_foreign_config_path_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["rejection_probes"]:
        if row["issue_code"] == "literal_error":
            row["config_path"] = "beams.squint.default.convention"
    with pytest.raises(EvidenceSchemaError, match="frozen path"):
        validate_stage3_evidence(document)


def test_a_stage3_rejection_probe_with_the_wrong_exception_type_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["rejection_probes"]:
        if row["issue_code"] == "extra_forbidden":
            row["exception_type"] = "ConfigSemanticError"
    with pytest.raises(EvidenceSchemaError, match="frozen to"):
        validate_stage3_evidence(document)


def test_a_paraphrased_reused_family_message_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["rejection_probes"]:
        if row["issue_code"] == "beam.squint.unsupported_beam_family":
            row["exact_message"] = "squint is not supported for FITS beams"
    with pytest.raises(EvidenceSchemaError, match="frozen message"):
        validate_stage3_evidence(document)


def test_a_stage3_disabled_workload_whose_fingerprint_moved_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["fingerprint_diff"][1]["new_scientific_sha256"] = "c" * 64
    with pytest.raises(EvidenceSchemaError, match="byte-identical"):
        validate_stage3_evidence(document)


def test_a_disabled_control_whose_cube_bytes_moved_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["fingerprint_diff"][1]["new_raw_cube_sha256"] = "c" * 64
    with pytest.raises(EvidenceSchemaError, match="cube bytes"):
        validate_stage3_evidence(document)


def test_a_fingerprint_set_without_a_scalar_peak_fits_control_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["fingerprint_diff"][1]["workload"] = "point_scalar_default"
    with pytest.raises(EvidenceSchemaError, match="scalar peak FITS workload"):
        validate_stage3_evidence(document)


def test_a_missing_efield_probe_kind_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["efield_file_contracts"] = document["efield_file_contracts"][:-1]
    with pytest.raises(EvidenceSchemaError, match="missing probe kinds"):
        validate_stage3_evidence(document)


def test_an_accepted_file_row_carrying_an_exception_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["efield_file_contracts"]:
        if row["probe_kind"] == "accepted_linear_pair":
            row["exception_type"] = "UnsupportedBeamFeedError"
    with pytest.raises(EvidenceSchemaError, match="no error field"):
        validate_stage3_evidence(document)


def test_a_rejected_file_row_with_a_null_message_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["efield_file_contracts"]:
        if row["probe_kind"] == "power_beam":
            row["exact_message"] = None
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_a_file_row_with_the_wrong_frozen_exception_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["efield_file_contracts"]:
        if row["probe_kind"] == "grid_coverage":
            row["exception_type"] = "UnsupportedBeamCoordinateError"
    with pytest.raises(EvidenceSchemaError, match="frozen to"):
        validate_stage3_evidence(document)


def test_an_accepted_kind_recorded_as_rejected_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["efield_file_contracts"]:
        if row["probe_kind"] == "accepted_circular_pair":
            row["outcome"] = "rejected"
    with pytest.raises(EvidenceSchemaError, match="accepted_\\* kinds"):
        validate_stage3_evidence(document)


def test_a_basis_vector_not_identity_row_claiming_the_identity_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["efield_file_contracts"]:
        if row["probe_kind"] == "basis_vector_not_identity":
            row["stored_basis_is_identity"] = True
    with pytest.raises(EvidenceSchemaError, match="native identity"):
        validate_stage3_evidence(document)


def test_an_accepted_row_without_the_stored_identity_basis_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["efield_file_contracts"]:
        if row["probe_kind"] == "accepted_linear_pair":
            row["stored_basis_is_identity"] = False
    with pytest.raises(EvidenceSchemaError, match="native identity basis"):
        validate_stage3_evidence(document)


def test_an_accepted_row_whose_peak_leaves_its_tolerance_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["efield_file_contracts"]:
        if row["probe_kind"] == "accepted_linear_pair":
            row["common_peak_by_frequency"][1]["observed_peak"] = 1.5
    with pytest.raises(EvidenceSchemaError, match="unit-peak"):
        validate_stage3_evidence(document)


def test_an_accepted_row_narrowed_to_a_non_native_dtype_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["efield_file_contracts"]:
        if row["probe_kind"] == "accepted_linear_pair":
            row["native_data_dtype"] = "float64"
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_a_non_increasing_common_peak_frequency_grid_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["efield_file_contracts"]:
        if row["probe_kind"] == "accepted_linear_pair":
            row["common_peak_by_frequency"][1]["frequency_hz"] = 1.0e8
    with pytest.raises(EvidenceSchemaError, match="strictly increasing"):
        validate_stage3_evidence(document)


def test_a_paraphrased_float128_message_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["efield_file_contracts"]:
        if row["probe_kind"] == "extended_precision":
            row["exact_message"] = "float128 beam precision is not supported"
    with pytest.raises(EvidenceSchemaError, match="float128 rejection literal"):
        validate_stage3_evidence(document)


def test_a_feed_array_of_the_wrong_length_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["efield_file_contracts"][0]["feed_array"] = ["x", "y", "z"]
    with pytest.raises(EvidenceSchemaError, match="exactly two"):
        validate_stage3_evidence(document)


def test_a_receptor_feed_rotation_outside_its_interval_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["efield_file_contracts"][0]["receptor_feed_rotation_rad"] = -math.pi
    with pytest.raises(EvidenceSchemaError, match="outside the interval"):
        validate_stage3_evidence(document)


def test_a_missing_basis_conversion_oracle_kind_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["basis_conversions"] = document["basis_conversions"][:-1]
    with pytest.raises(EvidenceSchemaError, match="missing oracle kinds"):
        validate_stage3_evidence(document)


def test_a_conversion_residual_above_its_bound_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["basis_conversions"]:
        if row["oracle_kind"] == "crossed_ideal_dipole":
            row["max_abs_residual"] = 1.0
    with pytest.raises(EvidenceSchemaError, match="atol/rtol bound"):
        validate_stage3_evidence(document)


def test_a_zenith_limit_delta_above_its_bound_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["basis_conversions"]:
        if row["oracle_kind"] == "chain_tangent_mapping":
            row["zenith_limit_max_abs_delta"] = 1.0
    with pytest.raises(EvidenceSchemaError, match="atol/rtol bound"):
        validate_stage3_evidence(document)


def test_a_wrap_continuity_delta_above_its_own_bound_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["basis_conversions"]:
        if row["oracle_kind"] == "quadrupolar":
            row["wrap_continuity_max_abs_delta"] = 1.0
    with pytest.raises(EvidenceSchemaError, match="continuity bound"):
        validate_stage3_evidence(document)


def test_a_power_preservation_residual_above_the_fixed_tolerance_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["basis_conversions"][0]["power_preservation_max_abs_residual"] = 1e-6
    with pytest.raises(EvidenceSchemaError, match="preserves power"):
        validate_stage3_evidence(document)


def test_an_orthogonality_residual_above_the_fixed_tolerance_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["basis_conversions"][0]["orthogonality_max_abs_residual"] = 1e-6
    with pytest.raises(EvidenceSchemaError, match="orthogonal"):
        validate_stage3_evidence(document)


def test_a_scalar_subset_control_that_agrees_is_rejected() -> None:
    """The control is the retained divergence witness, not an agreement row."""
    document = synthetic_stage3_document()
    for row in document["basis_conversions"]:
        if row["oracle_kind"] == STAGE3_SCALAR_SUBSET_CONTROL:
            row["max_abs_residual"] = 0.0
    with pytest.raises(EvidenceSchemaError, match="measured divergence"):
        validate_stage3_evidence(document)


def test_a_conversion_projection_narrowed_to_complex64_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["basis_conversions"][0]["observed"]["dtype"] = "complex64"
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_a_conversion_projection_of_the_wrong_shape_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["basis_conversions"][0]["expected"]["shape"] = [2, 2]
    document["basis_conversions"][0]["observed"]["shape"] = [2, 2]
    with pytest.raises(EvidenceSchemaError, match="S, 2, 2"):
        validate_stage3_evidence(document)


def test_a_probe_projection_that_disagrees_with_the_conversion_extent_is_rejected() -> (
    None
):
    document = synthetic_stage3_document()
    document["basis_conversions"][0]["probe_azimuth_rad"]["shape"] = [4]
    document["basis_conversions"][0]["probe_zenith_angle_rad"]["shape"] = [4]
    with pytest.raises(EvidenceSchemaError, match="retained probe count"):
        validate_stage3_evidence(document)


def test_a_missing_receptor_output_basis_combination_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["receptor_factorizations"] = document["receptor_factorizations"][:-1]
    with pytest.raises(EvidenceSchemaError, match="missing receptor/output"):
        validate_stage3_evidence(document)


def test_a_stage3_factorization_array_without_a_rotated_linear_row_is_rejected() -> (
    None
):
    document = synthetic_stage3_document()
    for row in document["receptor_factorizations"]:
        row["feed_rotation_deg"] = 0.0
    with pytest.raises(EvidenceSchemaError, match="non-zero feed_rotation_deg"):
        validate_stage3_evidence(document)


def test_a_stage3_factorization_residual_above_its_atol_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["receptor_factorizations"][0]["factorization_max_abs_residual"] = 1.0
    with pytest.raises(EvidenceSchemaError, match="retained atol"):
        validate_stage3_evidence(document)


def test_an_output_basis_residual_above_its_atol_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["receptor_factorizations"][0]["output_basis_max_abs_residual"] = 1.0
    with pytest.raises(EvidenceSchemaError, match="retained atol"):
        validate_stage3_evidence(document)


def test_a_noncommuting_component_that_fails_its_recomputation_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["receptor_factorizations"][0]["noncommuting_component"] = 1.0
    with pytest.raises(EvidenceSchemaError, match="recomputation"):
        validate_stage3_evidence(document)


def test_a_commuting_factorization_row_is_rejected() -> None:
    """A diagonal, symmetric ``E`` commutes with every real rotation."""
    document = synthetic_stage3_document()
    row = document["receptor_factorizations"][0]
    row["e_matrix"] = [
        [{"real": 1.0, "imag": 0.0}, {"real": 0.0, "imag": 0.0}],
        [{"real": 0.0, "imag": 0.0}, {"real": 1.0, "imag": 0.0}],
    ]
    row["noncommuting_component"] = 0.0
    with pytest.raises(EvidenceSchemaError, match="does not commute"):
        validate_stage3_evidence(document)


def test_an_order_control_below_the_thousand_atol_floor_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["receptor_factorizations"][0]["order_control_max_abs_difference"] = 1e-9
    with pytest.raises(EvidenceSchemaError, match="does not commute"):
        validate_stage3_evidence(document)


def test_a_factorization_row_whose_composed_e_repeats_j_native_is_rejected() -> None:
    """Section 8.1's witness-adequacy guard,
    ``composed_e.c_order_sha256 != j_native.c_order_sha256`` on every row.

    This is "a **witness-adequacy** rule, not a theorem, and the distinction is
    load-bearing". The document mutated below is *legitimately computable*
    rather than unphysical: since ``E = C^dagger J_native``, coincidence occurs
    exactly when ``J_native``'s columns lie in the ``+1`` eigenspace of
    ``C^dagger``, and that eigenspace exists in **both** bases -- the linear
    ``C(chi) = P_swap R(chi)`` is a reflection with eigenvalues exactly
    ``{+1, -1}`` at every ``chi``, and the circular ``C^dagger`` has an
    isolated unit eigenvalue at ``chi = pi/4`` modulo ``2*pi``, an entirely
    unremarkable ``feed_rotation_deg`` of ``45.0``.

    It is rejected because such a scenario "**cannot serve as the retained
    witness**": where ``E`` coincides with ``J_native`` the row "demonstrates
    nothing about ``C^dagger`` conjugation, and the mis-projection this rule
    exists to catch -- retaining ``C @ E``, which *is* ``J_native``, in the
    field reserved for ``E`` -- would be undetectable in it". That is exactly
    the defect the ``A3`` acceptance review found in the retained bytes, whose
    four rows carried the two fields byte-identical.

    The coincidence is written in explicitly rather than inherited from the
    synthetic builder, so the mutation states the rejected condition whatever
    digests that builder emits, and the unmutated document is validated first
    so the rejection is attributable to this mutation and to nothing else.
    """
    document = synthetic_stage3_document()
    validate_stage3_evidence(document)

    for row in document["receptor_factorizations"]:
        row["composed_e"]["c_order_sha256"] = row["j_native"]["c_order_sha256"]

    with pytest.raises(EvidenceSchemaError, match="receptor_factorizations"):
        validate_stage3_evidence(document)


def test_a_missing_ixr_state_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["ixr_diagnostics"] = document["ixr_diagnostics"][:-1]
    with pytest.raises(EvidenceSchemaError, match="missing IXR states"):
        validate_stage3_evidence(document)


def test_an_ixr_state_that_contradicts_its_singular_values_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["ixr_diagnostics"]:
        if row["case_id"] == "ixr_nonsingular":
            row["state"] = "unitary_scaled"
    with pytest.raises(EvidenceSchemaError, match="classifies this row"):
        validate_stage3_evidence(document)


def test_a_singular_row_carrying_a_derived_quantity_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["ixr_diagnostics"]:
        if row["state"] == "singular":
            row["condition_number"] = 1.0
    with pytest.raises(EvidenceSchemaError, match="null derived fields"):
        validate_stage3_evidence(document)


def test_a_unitary_scaled_row_carrying_an_ixr_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["ixr_diagnostics"]:
        if row["state"] == "unitary_scaled":
            row["ixr_linear"] = 1e30
    with pytest.raises(EvidenceSchemaError, match="non-finite number"):
        validate_stage3_evidence(document)


def test_a_unitary_scaled_row_with_a_forced_condition_number_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["ixr_diagnostics"]:
        if row["state"] == "unitary_scaled":
            row["condition_number"] = 1.5
    with pytest.raises(EvidenceSchemaError, match="realized ratio"):
        validate_stage3_evidence(document)


def test_a_nonsingular_row_that_fails_its_recomputation_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["ixr_diagnostics"]:
        if row["state"] == "nonsingular":
            row["ixr_db"] = 3.0
    with pytest.raises(EvidenceSchemaError, match="log10 recomputation"):
        validate_stage3_evidence(document)


def test_a_nonsingular_row_missing_a_derived_quantity_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["ixr_diagnostics"]:
        if row["state"] == "nonsingular":
            row["leakage_magnitude"] = None
    with pytest.raises(EvidenceSchemaError, match="all four quantities"):
        validate_stage3_evidence(document)


def test_an_ixr_row_whose_minimum_exceeds_its_maximum_is_rejected() -> None:
    document = synthetic_stage3_document()
    for row in document["ixr_diagnostics"]:
        if row["state"] == "nonsingular":
            row["sigma_min"] = 3.0
    with pytest.raises(EvidenceSchemaError, match="must not exceed sigma_max"):
        validate_stage3_evidence(document)


def test_a_comparison_naming_a_foreign_source_sha_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["crossvalidation_comparisons"][0]["radiosim_source_sha"] = "f" * 40
    with pytest.raises(EvidenceSchemaError, match="own source_sha"):
        validate_stage3_evidence(document)


def test_a_comparison_whose_artifact_is_not_the_dated_one_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["crossvalidation_comparisons"][0]["artifact_path"] = (
        "output/crossvalidation/2026-08-19-pyuvsim-1.4.0.json"
    )
    with pytest.raises(EvidenceSchemaError, match="dated cross-validation basename"):
        validate_stage3_evidence(document)


def test_a_comparison_whose_basename_date_disagrees_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["crossvalidation_comparisons"][0]["artifact_generated_at_utc"] = (
        "2026-08-20T00:00:00Z"
    )
    with pytest.raises(EvidenceSchemaError, match="UTC date"):
        validate_stage3_evidence(document)


def test_a_comparison_digest_disagreeing_with_its_artifacts_row_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["crossvalidation_comparisons"][0]["artifact_sha256"] = "1" * 64
    with pytest.raises(EvidenceSchemaError, match="disagrees with its artifacts row"):
        validate_stage3_evidence(document)


def test_a_comparison_absent_from_the_artifacts_array_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["artifacts"] = document["artifacts"][:1]
    with pytest.raises(EvidenceSchemaError, match="not retained in the artifacts"):
        validate_stage3_evidence(document)


def test_a_gating_comparison_row_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["crossvalidation_comparisons"][0]["gating"] = True
    with pytest.raises(EvidenceSchemaError, match="boolean false"):
        validate_stage3_evidence(document)


def test_a_floated_reference_version_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["crossvalidation_comparisons"][0]["reference_version"] = "1.4.2"
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_correlation_residuals_from_the_wrong_basis_are_rejected() -> None:
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    row["correlation_residuals"] = [
        {
            "correlation": label,
            "max_abs_residual": 1e-12,
            "max_rel_residual": 1e-12,
            "reference_max_abs": 1.0,
        }
        for label in sorted(STAGE3_CORRELATION_LABELS["circular_rl"])
    ]
    with pytest.raises(EvidenceSchemaError, match="complete 'linear_xy' label set"):
        validate_stage3_evidence(document)


def test_unsorted_correlation_residuals_are_rejected() -> None:
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    row["correlation_residuals"] = list(reversed(row["correlation_residuals"]))
    with pytest.raises(EvidenceSchemaError, match="sorted"):
        validate_stage3_evidence(document)


def test_an_incomplete_convention_mapping_set_is_rejected() -> None:
    """The amended contract's minimum coverage is six rows, not four.

    The chain-basis and comparison correction added the chain sky tangent basis
    and the interpolation order to the mapping set Section 8.1 requires at
    minimum.
    """
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    row["convention_mappings"] = row["convention_mappings"][:3]
    with pytest.raises(EvidenceSchemaError, match="at least 6"):
        validate_stage3_evidence(document)


def test_an_unequalized_interpolation_order_is_rejected() -> None:
    """Amended Section 5.5: a run without equalized interpolation order "is not
    evidence and may not be retained"."""
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    for mapping in row["convention_mappings"]:
        if mapping["radiosim_convention"] == STAGE3_EQUALIZED_CONVENTION:
            mapping["equivalent"] = False
    with pytest.raises(EvidenceSchemaError, match="equivalent true"):
        validate_stage3_evidence(document)


@pytest.mark.parametrize("name", sorted(STAGE3_NON_EQUIVALENT_CONVENTIONS))
def test_a_falsely_equivalent_convention_mapping_is_rejected(name: str) -> None:
    """Amended Section 8.1: "The east-X and Stokes-to-coherency rows may **not**
    be recorded ``equivalent: true``"."""
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    for mapping in row["convention_mappings"]:
        if mapping["radiosim_convention"] == name:
            mapping["equivalent"] = True
    with pytest.raises(EvidenceSchemaError, match="must not record"):
        validate_stage3_evidence(document)


def test_a_bounded_quantity_above_the_frame_residual_ceiling_is_rejected() -> None:
    """Amended Section 8.1: "every ``bound`` must be at or below the accepted
    ``1.9e-3`` SCI-007 frame residual"."""
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    row["bounded_quantities"][0]["bound"] = 1e-2
    with pytest.raises(EvidenceSchemaError, match="SCI-007 frame residual"):
        validate_stage3_evidence(document)


def test_a_bounded_quantity_whose_verdict_disagrees_is_rejected() -> None:
    """A row's ``passed`` equals ``max_rel_residual <= bound``."""
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    row["bounded_quantities"][0]["max_rel_residual"] = 1.5e-3
    row["bounded_quantities"][0]["bound"] = 1e-3
    with pytest.raises(EvidenceSchemaError, match="max_rel_residual <= bound"):
        validate_stage3_evidence(document)


def test_an_incomplete_bounded_quantity_set_is_rejected() -> None:
    """Both ``total_intensity`` and ``stokes_v_class`` must appear."""
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    row["bounded_quantities"] = row["bounded_quantities"][:1]
    with pytest.raises(EvidenceSchemaError, match="at least 2"):
        validate_stage3_evidence(document)


def test_a_mirror_transfer_solve_residual_above_its_bound_is_rejected() -> None:
    """Amended Section 8.1: the transfer-solve residual "must not exceed
    ``1e-3``, and it is the quantitative claim that the mechanism is complete --
    a validator checks it directly, so an unexplained residual cannot hide
    behind prose"."""
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    row["reference_frame_mirror"]["transfer_solve_max_abs_residual"] = 1e-2
    with pytest.raises(EvidenceSchemaError, match="mechanism is complete"):
        validate_stage3_evidence(document)


def test_a_mirror_construction_literal_that_is_not_the_frozen_one_is_rejected() -> None:
    """Amended Section 8.1 freezes the construction, not only the ceiling.

    "A ceiling without a construction makes two regenerations incomparable" --
    and they were not: the adjudication measured ``6.8e-5`` with one
    construction while the campaign measured ``7.36e-4`` with the frozen
    parameter-free one, and only the latter governs a retained row.
    """
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    row["reference_frame_mirror"]["construction"] = "fitted_frame_rotation_v1"
    with pytest.raises(EvidenceSchemaError, match="construction"):
        validate_stage3_evidence(document)


def test_a_reassembly_gap_above_its_ceiling_is_rejected() -> None:
    """The gap "is what proves the substitution is the only difference"."""
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    row["reference_frame_mirror"]["reassembly_gap"] = 1e-9
    with pytest.raises(EvidenceSchemaError, match="only difference"):
        validate_stage3_evidence(document)


def test_the_relative_transfer_solve_residual_carries_no_ceiling() -> None:
    """Amended Section 8.1 rules it **informational-only**.

    "Freezing a relative ceiling would freeze a fixture-dependent bound,
    because the denominator is a per-correlation scale that moves with the
    fixture's Stokes content"; the absolute ``1e-3`` governs. It is still
    type-checked, so a non-number is refused.
    """
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    row["reference_frame_mirror"]["transfer_solve_max_rel_residual"] = 0.5
    validate_stage3_evidence(document)

    row["reference_frame_mirror"]["transfer_solve_max_rel_residual"] = "large"
    with pytest.raises(EvidenceSchemaError, match="transfer_solve_max_rel_residual"):
        validate_stage3_evidence(document)


def test_a_mirror_without_its_two_citations_is_rejected() -> None:
    """The mechanism is retained with its line citations or not at all."""
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    row["reference_frame_mirror"]["citations"] = ["pyradiosky/utils.py:105-120"]
    with pytest.raises(EvidenceSchemaError, match="at least 2"):
        validate_stage3_evidence(document)


def test_a_mirrored_correlation_missing_from_open_disagreements_is_rejected() -> None:
    """Amended Section 8.1's cross-field rule: "a mirrored correlation can never
    be silently counted as an agreement"."""
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    row["open_disagreements"] = [
        entry for entry in row["open_disagreements"] if not entry.startswith("XY:")
    ]
    with pytest.raises(EvidenceSchemaError, match="must appear in open_disagreements"):
        validate_stage3_evidence(document)


def test_an_incomplete_input_hash_set_is_rejected() -> None:
    document = synthetic_stage3_document()
    row = document["crossvalidation_comparisons"][0]
    row["input_hashes"] = row["input_hashes"][:3]
    with pytest.raises(EvidenceSchemaError, match="at least 4"):
        validate_stage3_evidence(document)


def test_an_unsorted_stage3_case_id_sequence_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["efield_file_contracts"] = list(
        reversed(document["efield_file_contracts"])
    )
    with pytest.raises(EvidenceSchemaError, match="sorted"):
        validate_stage3_evidence(document)


def test_a_duplicate_stage3_case_id_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["ixr_diagnostics"][1]["case_id"] = document["ixr_diagnostics"][0][
        "case_id"
    ]
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_a_boolean_where_a_stage3_number_belongs_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["basis_conversions"][0]["atol"] = True
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_a_non_finite_stage3_number_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["receptor_factorizations"][0]["order_control_max_abs_difference"] = float(
        "inf"
    )
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_a_false_stage3_row_is_rejected() -> None:
    document = synthetic_stage3_document()
    document["ixr_diagnostics"][0]["passed"] = False
    with pytest.raises(EvidenceSchemaError):
        validate_stage3_evidence(document)


def test_a_nonzero_command_exit_code_is_rejected_at_stage3() -> None:
    document = synthetic_stage3_document()
    document["commands"][0]["exit_code"] = 1
    with pytest.raises(EvidenceSchemaError, match="zero exit code"):
        validate_stage3_evidence(document)


def test_a_stage3_document_whose_design_equals_its_red_test_is_refused() -> None:
    """The recorded generator defect must fail authentication, not pass it."""
    document = synthetic_stage3_document()
    document["design_sha"] = document["red_test_sha"]
    with pytest.raises(EvidenceSchemaError, match="same commit"):
        authenticate_stage3_succession(document)


def test_the_stage3_succession_reads_parents_not_peels() -> None:
    """``<sha>^{commit}`` peels; only ``<sha>^`` is the direct parent.

    Section 8.1 records that confusion as the evidence generator's Stage-2
    defect. The same confusion inside this validator would make the three
    Stage-3 ancestry facts tautologies, so the distinction is pinned against
    real repository objects rather than assumed.
    """
    head = _stage2_git("rev-parse", "HEAD")
    assert _stage2_git("rev-parse", f"{head}^{{commit}}") == head
    parent = _stage3_parent_of(head)
    assert GIT_SHA.fullmatch(parent)
    assert parent != head
    assert parent == _stage2_git("rev-parse", "HEAD^")
