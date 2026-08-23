"""Strict authentication of the SCI-004 phase-M2 evidence artifact.

``docs/development/sci004_mmode_design.md`` Sections 13.4, 14.2 and 14.4 freeze
this module's successor authority: it lands in ``S2`` with both approved
constants as the literal ``None``, the official evidence path **absent**, and
every synthetic strict schema and digest fixture passing.  ``E2`` then changes
*only* the two constants below, from ``None`` to the exact lower-case 40- and
64-hexadecimal literals, and adds the artifact and its reproduction record.  No
import, expression, annotation, key, surrounding token, or other literal in
either assignment may change, so the validator's own token stream outside those
two spans is comparable to its direct-parent ``S2`` bytes.

In the ``S2`` state the null constants require the official artifact to be
absent while the synthetic fixtures prove the schema rules.  In the ``E2`` state
the validator additionally requires the artifact's ``source_sha`` and the
constant to equal the approved ``S2``, authenticates the raw artifact bytes,
locates the unique artifact-introducing ``E2`` commit and requires its direct
parent to be ``S2``, and checks the ``S2..E2`` diff against Section 13.4's
``E2`` list.  It deliberately does **not** require the current checkout or
``E2`` to equal ``source_sha``.

The superseded-versus-operative ``design_sha``.  Section 13.7's bounded
corrections move the operative ``D`` between ``R`` and ``S``, so the evidence's
``design_sha`` and the retained ``R2`` record's own ``design_sha`` are
*expected to differ*.  Nothing here equates them, and a synthetic fixture below
proves the difference is accepted rather than merely unchecked.

Importing this module loads only the Python standard library plus ``pytest``,
following ``tests/unit/test_sci004_phase1_evidence.py``: an evidence-critical
validator must not depend on a package that is merely transitively present,
because a lock update could drop it and silently turn a hard authentication
into a collection error.  The generator at
``tools/sci004_mmode_phase2_evidence.py`` is the normative producer; the checks
below enforce the same structure, key order and encodings in their own code
rather than importing the producer's opinion of them.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import struct
import subprocess
import sys
import tokenize
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Section 14.2's two approved constants.  ``E2`` replaces exactly these two
#: ``None`` literals and nothing else in this module.
APPROVED_SOURCE_SHA: str | None = None
APPROVED_ARTIFACT_SHA256: str | None = None

TOOL = "tools/sci004_mmode_phase2_evidence.py"
ARTIFACT = "docs/development/sci004_mmode_phase2_evidence.json"
REPRODUCTION = "docs/development/sci004_mmode_phase2_evidence.md"
RED_RECORD = "docs/development/sci004_mmode_phase2_red_failures.json"

VALIDATOR = "tests/unit/test_sci004_phase2_evidence.py"

#: Section 13.4's complete ``E2`` write authority.  The commit that introduces
#: the artifact may touch these paths and nothing else.
E2_AUTHORIZED_PATHS: frozenset[str] = frozenset({ARTIFACT, REPRODUCTION, VALIDATOR})

#: Section 14.2's exact MyST front matter for the reproduction record.
REPRODUCTION_FRONT_MATTER = "---\norphan: true\n---"

#: The two spans Section 13.4 lets ``E2`` rewrite inside this module.
APPROVED_CONSTANT_NAMES: tuple[str, ...] = (
    "APPROVED_SOURCE_SHA",
    "APPROVED_ARTIFACT_SHA256",
)

GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

EVIDENCE_SCHEMA = "radiosim.sci004.mmode-phase2-evidence.v1"
RED_FAILURE_SCHEMA = "radiosim.sci004.mmode-phase2-red-failures.v1"
SELF_REFERENCE_REASON = "self-reference: A binds the containing E commit"

#: Section 14.2's exact envelope key order.
ENVELOPE_KEYS: tuple[str, ...] = (
    "schema_version",
    "phase",
    "status",
    "generated_at_utc",
    "design_sha",
    "red_commit_sha",
    "source_sha",
    "evidence_commit_sha",
    "evidence_commit_sha_reason",
    "working_tree_clean",
    "environment",
    "source_identities",
    "red_failure_record",
    "results",
    "commands",
    "limitations",
    "claims_not_licensed",
)

#: Section 14.2's exact M2 ``results`` key order.  There is no
#: ``dependency_certificate``: that object belongs to ``M1`` and ``M3``.
RESULT_KEYS: tuple[str, ...] = (
    "frame_certificate_cases",
    "polarization_cases",
    "sky_component_cases",
    "direct_convergence_cases",
    "truncation_cases",
    "backend_parity_cases",
    "memory_cases",
    "capability_cases",
    "rejection_cases",
)

#: Section 14.2's exact M2 row key sets.
POLARIZATION_ROW_KEYS: tuple[str, ...] = (
    "fixture_id",
    "input_frame_sha256",
    "transported_frame_sha256",
    "stokes_case",
    "expected_cube_sha256",
    "observed_cube_sha256",
    "absolute_residual",
    "fixed_tolerance",
    "pass",
)
SKY_COMPONENT_ROW_KEYS: tuple[str, ...] = (
    "fixture_id",
    "representation",
    "point_coefficients_sha256",
    "healpix_coefficients_sha256",
    "hybrid_coefficients_sha256",
    "expected_sum_sha256",
    "ring_nest_equal",
    "pass",
)
WRONG_SIGN_KEYS: tuple[str, ...] = (
    "fourier_sign_jy",
    "v_bridge_jy",
    "tangent_transport_jy",
    "east_x_permutation_jy",
)
CAPABILITY_ROW_KEYS: tuple[str, ...] = (
    "simulator",
    "property",
    "expected",
    "observed",
    "tier7_test_nodeid",
    "pass",
)
REJECTION_ROW_KEYS: tuple[str, ...] = (
    "fixture_id",
    "config_path",
    "exception_type",
    "issue_code",
    "exact_message",
    "test_nodeid",
    "allocation_started",
    "output_path_created",
    "pass",
)

#: Section 7.3's fixed tier-1a limits and tier-2 convergence floor.
HORIZON_FREE_L2_LIMIT = 1e-8
CONVERGENCE_FACTOR_FLOOR = 2.0

#: Section 12.2's analytic complex128 residual limit and non-vacuity margin.
ANALYTIC_RESIDUAL_LIMIT = 5e-12
NON_VACUITY_FACTOR = 10.0

#: Section 3.1's ``tau``, as the exact binary64 the canonical grid retains.
TAU_F64BE = "401921fb54442d18"

#: Section 9's seven separately reported memory components, sorted.
MEMORY_COMPONENT_NAMES: tuple[str, ...] = (
    "backend_native_allocations",
    "canonical_sky_coefficients",
    "largest_baseline_transfer_block",
    "per_antenna_harmonic_cache",
    "quadrature_directions_weights_and_jones",
    "retained_mmode_visibilities",
    "time_domain_output_and_synthesis",
)

TIER7_CAPABILITY_NODE = (
    "tests/characterization/test_tier7_current_behavior.py::"
    "test_mmode_m1_capability_truth"
)


def _tool() -> Any:
    """Import the tracked generator without adding an import-time dependency."""
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    try:
        import sci004_mmode_phase2_evidence as module
    finally:
        sys.path.pop(0)
    return module


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

SIXTY_FOUR = "0" * 63 + "1"
FORTY = "0" * 39 + "1"
OTHER_FORTY = "0" * 39 + "2"

#: A minimal but complete frame geometry: one direction, three samples, one
#: baseline, one frequency.  ``K = 4*N*B*F = 12``.
SYNTHETIC_SAMPLES = 3
SYNTHETIC_BASELINES = 1
SYNTHETIC_FREQUENCIES = 1
SYNTHETIC_CELLS = 4 * SYNTHETIC_SAMPLES * SYNTHETIC_BASELINES * SYNTHETIC_FREQUENCIES
SYNTHETIC_MCHECK = 2
SYNTHETIC_DIRECTION = "point:0"


def _f64be(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def _center_turns(samples: int) -> list[str]:
    turns = []
    for index in range(samples):
        exact = Fraction(2 * index, 2 * samples)
        turns.append(f"{exact.numerator}/{exact.denominator}")
    return turns


def _mask_hex(bits: list[bool]) -> str:
    width = (len(bits) + 7) // 8
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    value <<= width * 8 - len(bits)
    return f"{value:0{width * 2}x}"


def _membership_digest(module: Any, mask_rows: list[dict[str, Any]]) -> str:
    return module.expand_membership_ledger(
        mask_rows, _center_turns(SYNTHETIC_SAMPLES), TAU_F64BE
    )


def _synthetic_frame_row(module: Any) -> dict[str, Any]:
    """Return a frame row satisfying every Section 12.1/14.2 projection rule."""
    visible = [True] * SYNTHETIC_SAMPLES
    mask_rows = [
        {
            "direction_id": SYNTHETIC_DIRECTION,
            "sample_count": SYNTHETIC_SAMPLES,
            "frozen_visible_mask_hex": _mask_hex(visible),
            "operational_visible_mask_hex": _mask_hex(visible),
            "mismatch_count": 0,
        }
    ]
    row: dict[str, Any] = {
        "fixture_id": "mmode_point_full_stokes",
        "certificate_sha256": SIXTY_FOUR,
        "site_manifest": {"schema_version": "radiosim.mmode-site.v1"},
        "site_sha256": SIXTY_FOUR,
        "input_identity_sha256": SIXTY_FOUR,
        "iers_table_sha256": SIXTY_FOUR,
        "frame_matrix_manifest": {"schema_version": "radiosim.mmode-frame.v1"},
        "frame_matrix_sha256": SIXTY_FOUR,
        "canonical_era_turn_grid_sha256": SIXTY_FOUR,
        "canonical_era_grid_sha256": SIXTY_FOUR,
        "pm_source_unit": "arcsec",
        "pom00_argument_unit": "rad",
        "xp0_arcsec": 0.1,
        "yp0_arcsec": 0.2,
        "das2r_rad_per_arcsec": 4.84813681109536e-06,
        "xp0_rad": 4.84813681109536e-07,
        "yp0_rad": 9.69627362219072e-07,
        "sp0_rad": 0.0,
        "diagnostic_qcheck_nsides": [16],
        "transfer_grid_catalog": [],
        "transfer_grid_catalog_sha256": SIXTY_FOUR,
        "direction_rows": [{"direction_id": SYNTHETIC_DIRECTION}],
        "direction_ledger_sha256": SIXTY_FOUR,
        "horizon_scan_manifest": {"schema_version": "radiosim.mmode-horizon-scan.v1"},
        "horizon_scan_sha256": SIXTY_FOUR,
        "horizon_scan_crossing_rows": [],
        "horizon_scan_summary_rows": [
            {
                "direction_id": SYNTHETIC_DIRECTION,
                "terminal_cell_count": 4,
                "boundary_evaluation_count": 5,
                "crossing_count": 0,
                "min_ceiling_margin_f64be": _f64be(1.0),
            }
        ],
        "horizon_scan_ledger_sha256": SIXTY_FOUR,
        "horizon_root_pair_rows": [{"direction_id": SYNTHETIC_DIRECTION}],
        "horizon_root_pair_ledger_sha256": SIXTY_FOUR,
        "horizon_slab_rows": [],
        "horizon_slab_ledger_sha256": SIXTY_FOUR,
        "horizon_sign_interval_rows": [],
        "horizon_sign_interval_ledger_sha256": SIXTY_FOUR,
        "horizon_membership_mask_rows": mask_rows,
        "horizon_membership_ledger_sha256": _membership_digest(module, mask_rows),
        "direct_split_rows": [],
        "direct_split_ledger_sha256": SIXTY_FOUR,
        "direct_integrand_enclosure_manifest": {"schema_version": "x"},
        "direct_integrand_enclosure_sha256": SIXTY_FOUR,
        "sidereal_samples": SYNTHETIC_SAMPLES,
        "quadrature_nside": 8,
        "n_baselines": SYNTHETIC_BASELINES,
        "n_frequencies": SYNTHETIC_FREQUENCIES,
        "n_correlations": 4,
        "horizon_isolation_interval_count": 4,
        "horizon_unresolved_interval_count": 0,
        "expected_horizon_slab_row_count": 0,
        "evaluated_horizon_slab_row_count": 0,
        "expected_horizon_sign_interval_count": 0,
        "evaluated_horizon_sign_interval_count": 0,
        "horizon_root_count_mismatches": 0,
        "horizon_root_orientation_mismatches": 0,
        "horizon_membership_mismatches": 0,
        "horizon_outside_slab_sign_mismatches": 0,
        "horizon_paired_root_count": 0,
        "horizon_mismatch_slab_count": 0,
        "horizon_mismatch_measure_turn": "0/1",
        "horizon_mismatch_measure_rad": 0.0,
        "horizon_mismatch_measure_limit_rad": 0.0,
        "horizon_root_max_rad": 0.0,
        "horizon_root_limit_rad": 2e-5,
        "phase_max_rad": 1e-6,
        "phase_limit_rad": 5e-3,
        "direct_gauss_scale_jy": 1.0,
        "frozen_gauss_change_max_jy": 0.0,
        "operational_gauss_change_max_jy": 0.0,
        "direct_gauss_change_max_jy": 0.0,
        "direct_gauss_change_limit_jy": 1e-11,
        "cube_scale_jy": 1.0,
        "cube_max_jy": 0.0,
        "cube_limit_jy": 1e-10,
        "cube_l2": 0.0,
        "cube_l2_limit": 5e-5,
        "direction_diagnostic_max_rad": 0.0,
        "direction_diagnostic_argmax_id": SYNTHETIC_DIRECTION,
        "direction_diagnostic_argmax_phase": "0/1",
        "basis_diagnostic_max_rad": 0.0,
        "basis_diagnostic_argmax_id": SYNTHETIC_DIRECTION,
        "basis_diagnostic_argmax_phase": "0/1",
        "frozen_gauss64_cube_sha256": SIXTY_FOUR,
        "frozen_gauss128_cube_sha256": SIXTY_FOUR,
        "operational_gauss64_cube_sha256": SIXTY_FOUR,
        "operational_gauss128_cube_sha256": SIXTY_FOUR,
        "frozen_enclosure_error_cube_sha256": SIXTY_FOUR,
        "operational_enclosure_error_cube_sha256": SIXTY_FOUR,
        "pass": True,
    }
    counts = {
        "expected_point_direction_count": 1,
        "evaluated_point_direction_count": 1,
        "expected_native_healpix_direction_count": 0,
        "evaluated_native_healpix_direction_count": 0,
        "expected_production_transfer_direction_count": 768,
        "evaluated_production_transfer_direction_count": 768,
        "expected_diagnostic_transfer_direction_count": 3072,
        "evaluated_diagnostic_transfer_direction_count": 3072,
        "expected_transfer_quadrature_direction_count": 3840,
        "evaluated_transfer_quadrature_direction_count": 3840,
        "expected_direction_count": 1,
        "evaluated_direction_count": 1,
        "expected_phase_comparison_count": 9,
        "evaluated_phase_comparison_count": 9,
        "expected_horizon_trajectory_count": 1,
        "evaluated_horizon_trajectory_count": 1,
        "expected_horizon_root_pair_row_count": 1,
        "evaluated_horizon_root_pair_row_count": 1,
        "expected_horizon_membership_count": SYNTHETIC_SAMPLES,
        "evaluated_horizon_membership_count": SYNTHETIC_SAMPLES,
        "expected_direct_exposure_split_count": SYNTHETIC_SAMPLES,
        "evaluated_direct_exposure_split_count": SYNTHETIC_SAMPLES,
        "expected_direct_split_row_count": 0,
        "evaluated_direct_split_row_count": 0,
        "expected_frozen_gauss64_node_count": 0,
        "evaluated_frozen_gauss64_node_count": 0,
        "expected_frozen_gauss128_node_count": 0,
        "evaluated_frozen_gauss128_node_count": 0,
        "expected_operational_gauss64_node_count": 0,
        "evaluated_operational_gauss64_node_count": 0,
        "expected_operational_gauss128_node_count": 0,
        "evaluated_operational_gauss128_node_count": 0,
        "expected_cube_cell_count": SYNTHETIC_CELLS,
        "evaluated_frozen_gauss64_cube_cell_count": SYNTHETIC_CELLS,
        "evaluated_frozen_gauss128_cube_cell_count": SYNTHETIC_CELLS,
        "evaluated_operational_gauss64_cube_cell_count": SYNTHETIC_CELLS,
        "evaluated_operational_gauss128_cube_cell_count": SYNTHETIC_CELLS,
        "compared_frozen_gauss_change_cell_count": SYNTHETIC_CELLS,
        "compared_operational_gauss_change_cell_count": SYNTHETIC_CELLS,
        "evaluated_frozen_enclosure_error_cell_count": SYNTHETIC_CELLS,
        "evaluated_operational_enclosure_error_cell_count": SYNTHETIC_CELLS,
    }
    row.update(counts)
    return row


def _synthetic_truncation_row() -> dict[str, Any]:
    grids = 2
    samples = grids * SYNTHETIC_BASELINES * SYNTHETIC_FREQUENCIES * 4 * 4
    blocks = (
        SYNTHETIC_BASELINES * SYNTHETIC_FREQUENCIES * 4 * 4 * (2 * SYNTHETIC_MCHECK + 1)
    )
    return {
        "fixture_id": "mmode_point_full_stokes",
        "input_identity_sha256": SIXTY_FOUR,
        "frame_certificate_sha256": SIXTY_FOUR,
        "direction_ledger_sha256": SIXTY_FOUR,
        "transfer_grid_catalog_sha256": SIXTY_FOUR,
        "production_transfer_grid_id": "production:8",
        "diagnostic_transfer_grid_ids": ["diagnostic:16"],
        "diagnostic_grid_joins": [],
        "lmax": 16,
        "mmax": 16,
        "quadrature_nside": 8,
        "lcheck": 24,
        "mcheck": SYNTHETIC_MCHECK,
        "qcheck": 16,
        "sidereal_samples": SYNTHETIC_SAMPLES,
        "cube_shape": [
            SYNTHETIC_SAMPLES,
            SYNTHETIC_BASELINES,
            SYNTHETIC_FREQUENCIES,
            4,
        ],
        "frozen_gauss128_cube_sha256": SIXTY_FOUR,
        "frozen_enclosure_error_cube_sha256": SIXTY_FOUR,
        "mmode_cube_sha256": SIXTY_FOUR,
        "direct_scale_jy": 5.5,
        "expected_output_cell_count": SYNTHETIC_CELLS,
        "evaluated_frozen_direct_cell_count": SYNTHETIC_CELLS,
        "evaluated_frozen_error_cell_count": SYNTHETIC_CELLS,
        "evaluated_mmode_cell_count": SYNTHETIC_CELLS,
        "compared_output_cell_count": SYNTHETIC_CELLS,
        "direct_coverage": {},
        "direct_coverage_sha256": SIXTY_FOUR,
        "horizon_free_shell_max_jy": 1e-11,
        "horizon_free_shell_l2": 1e-12,
        "horizon_free_shell_max_limit_jy": 1e-8,
        "horizon_free_shell_l2_limit": HORIZON_FREE_L2_LIMIT,
        "quadrature_shell_max_jy": 0.05,
        "quadrature_shell_l2": 0.01,
        "quadrature_budget_jy": 0.2,
        "deficit_max_jy": 0.117,
        "deficit_l2": 0.02,
        "deficit_max_quarter_jy": 0.716,
        "deficit_max_half_jy": 0.185,
        "convergence_factor": 6.12,
        "truncation_budget_jy": 0.4,
        "expected_shell_comparison_cell_count": 4 * SYNTHETIC_CELLS,
        "evaluated_shell_comparison_cell_count": 4 * SYNTHETIC_CELLS,
        "expected_transfer_sample_row_count": samples,
        "evaluated_transfer_sample_row_count": samples,
        "expected_field_block_count": blocks,
        "evaluated_field_block_count": blocks,
        "shell_coverage": {},
        "shell_coverage_sha256": SIXTY_FOUR,
        "quadrature_diagnostic_max_jy": 0.01,
        "l_tail_diagnostic_max_jy": 0.02,
        "m_tail_diagnostic_max_jy": 0.03,
        "combined_local_diagnostic_max_jy": 0.04,
        "field_block_diagnostic_max_jy": 0.05,
        "shell_diagnostic_reference_jy": 1e-06,
        "pass": True,
    }


def _synthetic_direct_convergence_row() -> dict[str, Any]:
    passing = 0.117
    return {
        "fixture_id": "mmode_point_full_stokes",
        "input_identity_sha256": SIXTY_FOUR,
        "frame_certificate_sha256": SIXTY_FOUR,
        "cube_shape": [
            SYNTHETIC_SAMPLES,
            SYNTHETIC_BASELINES,
            SYNTHETIC_FREQUENCIES,
            4,
        ],
        "expected_cell_count": SYNTHETIC_CELLS,
        "compared_finite_cell_count": SYNTHETIC_CELLS,
        "frozen_gauss64_cube_sha256": SIXTY_FOUR,
        "frozen_gauss128_cube_sha256": SIXTY_FOUR,
        "frozen_enclosure_error_cube_sha256": SIXTY_FOUR,
        "mmode_cube_sha256": SIXTY_FOUR,
        "gauss_change_max_jy": 0.0,
        "gauss_change_limit_jy": 1e-11,
        "analytic_piecewise_residual": 1e-15,
        "analytic_piecewise_limit": ANALYTIC_RESIDUAL_LIMIT,
        "direct_scale_jy": 5.5,
        "deficit_max_jy": passing,
        "deficit_l2": 0.02,
        "deficit_max_quarter_jy": 0.716,
        "deficit_max_half_jy": 0.185,
        "convergence_factor": 6.12,
        "truncation_budget_jy": 0.4,
        "wrong_sign_residuals": {
            "fourier_sign_jy": 20.0 * passing,
            "v_bridge_jy": 15.0 * passing,
            "tangent_transport_jy": 30.0 * passing,
            "east_x_permutation_jy": 40.0 * passing,
        },
        # Each control's reference is its own Section 12.2 family's residual --
        # the analytic ``5e-12`` for three of them and the certified frame
        # reduction for the transport -- so the values above clear every
        # corresponding limit by many orders of magnitude.
        "pass": True,
    }


def _synthetic_memory_row(module: Any) -> dict[str, Any]:
    schedule_rows = [
        {
            "block_index": 0,
            "frequency_start": 0,
            "frequency_stop": 1,
            "signed_m_start": 0,
            "signed_m_stop": 5,
            "baseline_start": 0,
            "baseline_stop": 1,
            "packed_value_count": 7,
        }
    ]
    return {
        "fixture_id": "mmode_point_full_stokes",
        "logical_dimensions": {
            "n_times": SYNTHETIC_SAMPLES,
            "n_baselines": SYNTHETIC_BASELINES,
            "n_frequencies": SYNTHETIC_FREQUENCIES,
            "n_correlations": 4,
            "n_packed_values": 2244,
            "n_quadrature_directions": 3840,
        },
        "block_dimensions": {
            "frequency_block_max": 1,
            "signed_m_block_max": 5,
            "baseline_block_max": 1,
            "packed_value_block_max": 7,
            "scheduled_block_count": 1,
        },
        "included_allocations": [
            {
                "name": "canonical_sky_coefficients",
                "bytes": 128,
                "measurement_domain": "host",
            }
        ],
        "excluded_allocations": [
            {
                "name": "backend_native_allocations",
                "bytes": 0,
                "measurement_domain": "backend_native",
            }
        ],
        "estimated_components": [
            {"name": name, "bytes": 1024} for name in MEMORY_COMPONENT_NAMES
        ],
        "estimated_peak_bytes": 7168,
        "measured_host_peak_bytes": 4096,
        "host_measurement_method": "tracemalloc peak over the scoped dense block",
        "measured_native_peak_bytes": None,
        "measured_native_peak_bytes_reason": (
            "the NumPy reference backend exposes no native allocator counter"
        ),
        "native_measurement_method": "none_available",
        "working_memory_bytes": 1073741824,
        "schedule_rows": schedule_rows,
        "schedule_sha256": module.domain_digest(
            "radiosim.sci004.block-schedule.v1", module.canonical_json(schedule_rows)
        ),
        "pass": True,
    }


def _synthetic_document(module: Any) -> dict[str, Any]:
    """Return a complete synthetic envelope satisfying every Section 14.2 rule."""
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "phase": "M2",
        "status": "candidate",
        "generated_at_utc": "2026-08-24T00:00:00Z",
        # Section 13.7: the operative ``D`` at ``S2`` is *not* the ``R2``
        # record's frozen binding, so the fixture uses two distinct values.
        "design_sha": FORTY,
        "red_commit_sha": OTHER_FORTY,
        "source_sha": FORTY,
        "evidence_commit_sha": None,
        "evidence_commit_sha_reason": SELF_REFERENCE_REASON,
        "working_tree_clean": True,
        "environment": {
            "python": "3.11.13",
            "platform": "darwin",
            "machine": "arm64",
            "pixi_environment": "default",
            "pixi_lock_sha256": SIXTY_FOUR,
            "astropy_version": "7.1.0",
            "erfa_version": "2.0.1.5",
            "iers_package_version": "7.1.0",
            "iers_table_sha256": SIXTY_FOUR,
            "numeric_packages": {
                "numpy": "2.3.2",
                "scipy": "1.16.0",
                "healpy": "1.18.1",
                "jax": "0.10.2",
                "dask": "2025.7.0",
            },
        },
        "source_identities": {
            "git_tree_sha256": SIXTY_FOUR,
            "pixi_manifest_sha256": SIXTY_FOUR,
            "pixi_lock_sha256": SIXTY_FOUR,
            "convention_identity_sha256": SIXTY_FOUR,
            "fixture_input_rows": [
                {
                    "fixture_id": "mmode_point_full_stokes",
                    "input_identity_manifest": {},
                    "input_identity_sha256": SIXTY_FOUR,
                }
            ],
            "input_identity_set_sha256": SIXTY_FOUR,
        },
        "red_failure_record": {
            "path": RED_RECORD,
            "sha256": SIXTY_FOUR,
            "schema_version": RED_FAILURE_SCHEMA,
            "pre_fix_source_sha": OTHER_FORTY,
            "validated": True,
        },
        "results": {
            "frame_certificate_cases": [_synthetic_frame_row(module)],
            "polarization_cases": [
                {
                    "fixture_id": "mmode_point_full_stokes",
                    "input_frame_sha256": SIXTY_FOUR,
                    "transported_frame_sha256": SIXTY_FOUR,
                    "stokes_case": "stokes_superposition",
                    "expected_cube_sha256": SIXTY_FOUR,
                    "observed_cube_sha256": SIXTY_FOUR,
                    "absolute_residual": 1e-15,
                    "fixed_tolerance": ANALYTIC_RESIDUAL_LIMIT,
                    "pass": True,
                }
            ],
            "sky_component_cases": [
                {
                    "fixture_id": "mmode_point_full_stokes",
                    "representation": "hybrid",
                    "point_coefficients_sha256": SIXTY_FOUR,
                    "healpix_coefficients_sha256": SIXTY_FOUR,
                    "hybrid_coefficients_sha256": SIXTY_FOUR,
                    "expected_sum_sha256": SIXTY_FOUR,
                    "ring_nest_equal": True,
                    "pass": True,
                }
            ],
            "direct_convergence_cases": [_synthetic_direct_convergence_row()],
            "truncation_cases": [_synthetic_truncation_row()],
            "backend_parity_cases": [
                {
                    "fixture_id": "mmode_point_full_stokes:per-m-contraction",
                    "requested_backend": "numpy",
                    "actual_backend": "numpy",
                    "actual_device": "cpu",
                    "dtype": "complex128",
                    "workers": 1,
                    "working_memory_bytes": 1073741824,
                    "numpy_sha256": SIXTY_FOUR,
                    "candidate_sha256": SIXTY_FOUR,
                    "absolute_max": 0.0,
                    "relative_max": 0.0,
                    "rtol": 1e-12,
                    "atol": 1e-12,
                    "pass": True,
                }
            ],
            "memory_cases": [_synthetic_memory_row(module)],
            "capability_cases": [
                {
                    "simulator": "mmode",
                    "property": "supports_polarization",
                    "expected": True,
                    "observed": True,
                    "tier7_test_nodeid": TIER7_CAPABILITY_NODE,
                    "pass": True,
                },
                {
                    "simulator": "rime",
                    "property": "supports_polarization",
                    "expected": True,
                    "observed": True,
                    "tier7_test_nodeid": TIER7_CAPABILITY_NODE,
                    "pass": True,
                },
                {
                    "simulator": "mmode",
                    "property": "supports_gpu",
                    "expected": False,
                    "observed": False,
                    "tier7_test_nodeid": TIER7_CAPABILITY_NODE,
                    "pass": True,
                },
            ],
            "rejection_cases": [
                {
                    "fixture_id": "mmode_point_full_stokes",
                    "config_path": "sky_model.sources[0].tangent_polarization_frame",
                    "exception_type": "ConfigSemanticError",
                    "issue_code": "mmode_polarization_frame",
                    "exact_message": (
                        "polarized m-mode input requires an explicit canonical "
                        "tangent-polarization frame."
                    ),
                    "test_nodeid": (
                        "tests/integration/test_sci004_mmode.py::"
                        "test_a_polarized_mmode_input_without_a_tangent_frame_is_rejected"
                    ),
                    "allocation_started": False,
                    "output_path_created": False,
                    "pass": True,
                }
            ],
        },
        "commands": [
            {
                "argv": ["pixi", "run", "test", "--", "-m", "not slow"],
                "cwd": ".",
                "pixi_environment": "default",
                "started_at_utc": "2026-08-24T00:00:00Z",
                "duration_seconds": 1.0,
                "exit_code": 0,
                "stdout_sha256": SIXTY_FOUR,
                "stderr_sha256": SIXTY_FOUR,
            }
        ],
        "limitations": ["phase M2 carries one production fixture"],
        "claims_not_licensed": [
            "general_speedup",
            "gpu_or_accelerator_support",
            "retained_fingerprint_pins",
        ],
    }


# ---------------------------------------------------------------------------
# Pre-E2 state
# ---------------------------------------------------------------------------


def test_the_approved_constants_are_null_sentinels_before_e2() -> None:
    """Section 14.2: at ``S2`` both approved digests are ``None``."""
    if APPROVED_SOURCE_SHA is None or APPROVED_ARTIFACT_SHA256 is None:
        assert APPROVED_SOURCE_SHA is None
        assert APPROVED_ARTIFACT_SHA256 is None
        return
    assert GIT_SHA.fullmatch(APPROVED_SOURCE_SHA)
    assert SHA256.fullmatch(APPROVED_ARTIFACT_SHA256)


def test_the_official_evidence_artifact_is_absent_in_the_s2_state() -> None:
    """Section 14.2: null constants require the evidence JSON to be absent."""
    if APPROVED_ARTIFACT_SHA256 is not None:
        return
    assert not (REPOSITORY_ROOT / ARTIFACT).exists()


def test_the_tracked_generator_and_its_inputs_exist_at_s2() -> None:
    """Section 14.2: the generator and the retained red record are tracked."""
    assert (REPOSITORY_ROOT / TOOL).is_file()
    assert (REPOSITORY_ROOT / RED_RECORD).is_file()


def test_the_generator_imports_only_the_standard_library() -> None:
    """An evidence-critical verifier carries no import-time package dependency.

    The scientific packages are imported inside the generation functions alone,
    so ``check`` -- which a reviewer runs -- never needs them.
    """
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    head = source[: source.index("class EvidenceError")]
    for forbidden in ("import numpy", "import astropy", "import pytest", "import yaml"):
        assert forbidden not in head, forbidden


def test_the_generator_refuses_before_producing_anything(monkeypatch) -> None:
    """Section 14.2: the pre-output check runs before any output is opened.

    The probe is a real run of the tracked generator, not a description of one:
    it is invoked with a ``--source-sha`` that is not ``HEAD``, and the refusal
    must name that, print nothing on stdout, fail closed with the frozen prefix
    rather than a traceback, and leave the declared output byte-identical --
    absent before ``E2``, and unchanged at ``E2`` where it legitimately exists,
    in which case the no-overwrite rule refuses first.

    Unlike the phase-1 probe this run **does not dirty the working tree**.  A
    globally visible untracked file races every other test that snapshots
    ``git status``, and the refusal it was there to exercise is reached by the
    hermetic in-process check below instead.
    """
    module = _tool()
    artifact = REPOSITORY_ROOT / ARTIFACT
    before = artifact.read_bytes() if artifact.exists() else None
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / TOOL),
            "generate",
            "--source-sha",
            "0" * 40,
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr.startswith(module.PREFLIGHT + ": ")
    assert "Traceback" not in completed.stderr
    reasons = (
        "is not the approved source",
        "not globally clean",
        "already exists",
    )
    assert any(reason in completed.stderr for reason in reasons)
    after = artifact.read_bytes() if artifact.exists() else None
    assert after == before
    del monkeypatch


def test_the_preflight_refuses_a_dirty_tree_before_any_output(monkeypatch) -> None:
    """Section 14.2's dirty-tree refusal, exercised without dirtying anything.

    ``git status --porcelain=v1 --untracked-files=all`` is the authority, so the
    probe replaces exactly that one command's output and requires the refusal to
    carry the frozen preflight prefix and to name the dirty tree.  Nothing in
    the repository is touched, so the check is hermetic under parallel
    execution -- which a globally visible probe file is not.
    """
    module = _tool()
    real = module._git

    def fake(*arguments: str) -> str:
        if arguments[:1] == ("status",):
            return " M src/radiosim/core/mmode/solver.py\n"
        return real(*arguments)

    monkeypatch.setattr(module, "_git", fake)
    with pytest.raises(module.EvidenceError) as excinfo:
        module.preflight()
    assert excinfo.value.prefix == module.PREFLIGHT
    assert "not globally clean" in excinfo.value.detail


def test_the_generator_produces_at_a_clean_source_rather_than_refusing() -> None:
    """Section 14.2/14.4: ``generate`` is bound to a venue, not prohibited.

    The complement of the refusal above is pinned in the tracked bytes: after a
    passing preflight the sub-command builds the document, validates it, and
    publishes it by atomic no-overwrite rename, with no unconditional
    post-preflight refusal.
    """
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    body = source[source.index('if arguments.command == "generate":') :]
    body = body[: body.index("document = json.loads(")]
    assert "build_evidence_document(state)" in body
    assert "validate_evidence_document(document)" in body
    assert "write_atomic_no_overwrite(" in body
    assert "raise EvidenceError(" not in body


# ---------------------------------------------------------------------------
# Synthetic strict schema and digest fixtures
# ---------------------------------------------------------------------------


def test_the_synthetic_envelope_satisfies_every_section_14_2_rule() -> None:
    """The fixture is a positive control for every rejection below."""
    module = _tool()
    envelope = module.validate_evidence_document(_synthetic_document(module))
    assert set(envelope) == set(ENVELOPE_KEYS)
    assert set(envelope["results"]) == set(RESULT_KEYS)


@pytest.mark.parametrize("key", ENVELOPE_KEYS)
def test_a_missing_top_level_key_is_rejected(key: str) -> None:
    """Section 14: every object rejects unknown or missing keys."""
    module = _tool()
    document = _synthetic_document(module)
    del document[key]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_an_unknown_top_level_key_is_rejected() -> None:
    """An extra key is a different envelope, not a superset of this one."""
    module = _tool()
    document = _synthetic_document(module)
    document["extra"] = 1
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


@pytest.mark.parametrize("key", RESULT_KEYS)
def test_a_missing_results_key_is_rejected(key: str) -> None:
    """Section 14.2's M2 ``results`` key set is exact and closed."""
    module = _tool()
    document = _synthetic_document(module)
    del document["results"][key]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_an_m1_dependency_certificate_is_rejected_in_m2() -> None:
    """Section 14.2: M2 ``results`` has no ``dependency_certificate``."""
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["dependency_certificate"] = {}
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_non_null_evidence_commit_sha_is_rejected() -> None:
    """Section 14.4: ``E`` artifacts use a null self SHA."""
    module = _tool()
    document = _synthetic_document(module)
    document["evidence_commit_sha"] = FORTY
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_reworded_self_reference_reason_is_rejected() -> None:
    """The self-reference reason is an exact literal, not a paraphrase."""
    module = _tool()
    document = _synthetic_document(module)
    document["evidence_commit_sha_reason"] = "self reference"
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_superseded_red_design_sha_is_accepted_not_equated() -> None:
    """Section 13.7: the operative ``D`` at ``S2`` supersedes ``R2``'s binding.

    A bounded correction between ``R`` and ``S`` moves the operative ``D``, so
    the two values legitimately differ.  The synthetic envelope carries two
    distinct values *and passes*; a validator that equated them would refuse
    exactly the phases Section 13.7 exists to permit.
    """
    module = _tool()
    document = _synthetic_document(module)
    assert document["design_sha"] != document["red_commit_sha"]
    assert (
        document["design_sha"] != document["red_failure_record"]["pre_fix_source_sha"]
    )
    module.validate_evidence_document(document)


def test_the_m2_capability_rows_must_state_the_flip() -> None:
    """Section 9: accepted M2 flips the m-mode property, stated with ``rime``."""
    module = _tool()
    document = _synthetic_document(module)
    rows = document["results"]["capability_cases"]
    rows[0]["expected"] = False
    rows[0]["observed"] = False
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_capability_row_missing_the_direct_solver_is_rejected() -> None:
    """Section 9 requires the two capability facts to be stated *together*."""
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["capability_cases"] = [
        row
        for row in document["results"]["capability_cases"]
        if row["simulator"] != "rime"
    ]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_capability_row_bound_to_a_non_tier7_node_is_rejected() -> None:
    """The authoritative pin is the Tier 7 characterization file, by node ID."""
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["capability_cases"][0]["tier7_test_nodeid"] = (
        "tests/unit/test_simulator/test_sci004_strategy.py::test_anything"
    )
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_frame_row_embedding_the_per_sample_membership_array_is_rejected() -> None:
    """Section 12.1's economy is a projection: the mask rows are the retained form."""
    module = _tool()
    document = _synthetic_document(module)
    frame = document["results"]["frame_certificate_cases"][0]
    frame["horizon_membership_mask_rows"][0]["per_sample_rows"] = []
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_flipped_mask_bit_breaks_the_expanded_ledger_digest() -> None:
    """The membership economy is authenticated by expansion, not by assertion.

    M2 retains no time-grid row, so the centres are *derived* from
    ``sidereal_samples`` under Section 3.1's exact rule -- which makes this a
    stronger check than a join to a retained copy of them.
    """
    module = _tool()
    document = _synthetic_document(module)
    frame = document["results"]["frame_certificate_cases"][0]
    frame["horizon_membership_mask_rows"][0]["frozen_visible_mask_hex"] = _mask_hex(
        [False] + [True] * (SYNTHETIC_SAMPLES - 1)
    )
    frame["horizon_membership_mask_rows"][0]["mismatch_count"] = 1
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_the_derived_centre_turns_are_the_section_3_1_rule() -> None:
    """Section 3.1: the centres are ``Fraction(2k, 2N)`` in reduced spelling."""
    module = _tool()
    assert module.canonical_center_turns(4) == ["0/1", "1/4", "1/2", "3/4"]
    assert module.canonical_center_turns(3) == ["0/1", "1/3", "2/3"]


def test_a_mask_row_whose_sample_count_is_not_n_is_rejected() -> None:
    """Section 12.1: each mask row's ``sample_count`` is exactly ``N``."""
    module = _tool()
    document = _synthetic_document(module)
    frame = document["results"]["frame_certificate_cases"][0]
    frame["horizon_membership_mask_rows"][0]["sample_count"] = SYNTHETIC_SAMPLES + 1
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_the_synthetic_projection_retains_its_guard_rows() -> None:
    """Section 12.1: a crossing's flanking guards are part of the projection."""
    module = _tool()
    document = _synthetic_document(module)
    frame = document["results"]["frame_certificate_cases"][0]
    guard = {
        "direction_id": SYNTHETIC_DIRECTION,
        "cell_index": 1,
        "turn_lo": "1/8",
        "turn_hi": "1/4",
        "classification": "guard_interval",
        "f_lo_f64be": _f64be(1.0),
        "f_hi_f64be": _f64be(1.0),
        "ceiling_margin_f64be": _f64be(0.0),
        "left_sign": 1,
        "right_sign": 1,
        "root_turn_lo": None,
        "root_turn_hi": None,
        "root_orientation": None,
        "root_residual_f64be": None,
    }
    frame["horizon_scan_crossing_rows"] = [guard]
    frame["horizon_scan_summary_rows"][0]["crossing_count"] = 1
    # An orphan guard -- one with no crossing enclosure to flank -- is rejected.
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_duplicate_owned_root_is_rejected() -> None:
    """Section 12.1: two crossing rows claiming one enclosure inflate the census."""
    module = _tool()
    document = _synthetic_document(module)
    frame = document["results"]["frame_certificate_cases"][0]
    crossing = {
        "direction_id": SYNTHETIC_DIRECTION,
        "cell_index": 0,
        "turn_lo": "1/8",
        "turn_hi": "1/4",
        "classification": "scan_crossing",
        "f_lo_f64be": _f64be(-1.0),
        "f_hi_f64be": _f64be(1.0),
        "ceiling_margin_f64be": _f64be(1.0),
        "left_sign": -1,
        "right_sign": 1,
        "root_turn_lo": "1/8",
        "root_turn_hi": "1/4",
        "root_orientation": "rising",
        "root_residual_f64be": _f64be(1e-12),
    }
    frame["horizon_scan_crossing_rows"] = [crossing, dict(crossing)]
    frame["horizon_scan_summary_rows"][0]["crossing_count"] = 2
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_cube_count_that_is_not_k_is_rejected() -> None:
    """Section 14.2: every frame cube count equals ``K = 4*N*B*F``."""
    module = _tool()
    document = _synthetic_document(module)
    frame = document["results"]["frame_certificate_cases"][0]
    frame["evaluated_frozen_gauss128_cube_cell_count"] = SYNTHETIC_CELLS - 1
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_widened_fixed_frame_limit_is_rejected() -> None:
    """Section 4.2's frame limits are fixed and are never widened."""
    module = _tool()
    document = _synthetic_document(module)
    frame = document["results"]["frame_certificate_cases"][0]
    frame["horizon_root_limit_rad"] = 2e-4
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_tier_1a_shell_above_its_fixed_limit_is_rejected() -> None:
    """Section 7.3 tier 1a is the sharp half and gates at a fixed ``1e-8``."""
    module = _tool()
    document = _synthetic_document(module)
    row = document["results"]["truncation_cases"][0]
    row["horizon_free_shell_l2"] = 1e-6
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_non_monotone_deficit_sequence_is_rejected() -> None:
    """Section 7.3 tier 2 requires strict monotone decrease across the levels."""
    module = _tool()
    document = _synthetic_document(module)
    row = document["results"]["truncation_cases"][0]
    row["deficit_max_half_jy"] = row["deficit_max_quarter_jy"]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_quarter_to_full_factor_below_two_is_rejected() -> None:
    """The quarter-to-full factor floor is fixed at two and never widened."""
    module = _tool()
    document = _synthetic_document(module)
    row = document["results"]["truncation_cases"][0]
    row["convergence_factor"] = 1.9
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_deficit_above_its_declared_budget_is_rejected() -> None:
    """The truncation budget is a reviewed evidence field the deficit obeys."""
    module = _tool()
    document = _synthetic_document(module)
    row = document["results"]["truncation_cases"][0]
    row["truncation_budget_jy"] = 0.05
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_with_horizon_shell_above_its_declared_budget_is_rejected() -> None:
    """Tier 1b is recorded and bounded by its reviewed per-fixture budget."""
    module = _tool()
    document = _synthetic_document(module)
    row = document["results"]["truncation_cases"][0]
    row["quadrature_budget_jy"] = 0.01
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_truncation_row_without_its_frame_row_is_rejected() -> None:
    """Section 14.2: a truncation row joins the unique same-fixture frame row."""
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["truncation_cases"][0]["fixture_id"] = "other"
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_truncation_row_that_does_not_join_its_frame_digests_is_rejected() -> None:
    """The two rows must name the same certificate, not merely the same fixture."""
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["truncation_cases"][0]["frame_certificate_sha256"] = "0" * 64
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


@pytest.mark.parametrize("control", WRONG_SIGN_KEYS)
def test_a_wrong_sign_control_that_stops_missing_is_rejected(control: str) -> None:
    """Section 12.2: every control misses **its corresponding** passing residual.

    "Corresponding" is enforced per control: each defect breaks a different
    Section 12.2 oracle family, so the reference is that family's own passing
    residual rather than one convenient number.  Setting a control to exactly
    its own reference must be refused for every one of the four.
    """
    module = _tool()
    document = _synthetic_document(module)
    row = document["results"]["direct_convergence_cases"][0]
    _family, reference_field = module.CONTROL_PASSING_RESIDUAL[control]
    row["wrong_sign_residuals"][control] = float(row[reference_field])
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_the_control_reference_families_are_the_section_12_2_assignment() -> None:
    """Each control names the Section 12.2 family whose residual it is judged by.

    The ``D`` bridge and the SCI-006 east-X permutation are the same kind of
    object -- exact matrix conventions Section 12.2 lists together in family 4
    -- so both are referred to the analytic residual; the Fourier sign belongs
    to family 1's analytic exposure-sinc identity; and the omitted tangent
    transport belongs to family 2's certified frame reduction.
    """
    module = _tool()
    assert set(module.CONTROL_PASSING_RESIDUAL) == set(WRONG_SIGN_KEYS)
    assert module.CONTROL_PASSING_RESIDUAL["v_bridge_jy"] == (
        "analytic polarization",
        "analytic_piecewise_limit",
    )
    assert module.CONTROL_PASSING_RESIDUAL["east_x_permutation_jy"] == (
        "analytic polarization",
        "analytic_piecewise_limit",
    )
    assert module.CONTROL_PASSING_RESIDUAL["fourier_sign_jy"] == (
        "analytic ERA/DFT",
        "analytic_piecewise_limit",
    )
    assert module.CONTROL_PASSING_RESIDUAL["tangent_transport_jy"] == (
        "frame",
        "gauss_change_limit_jy",
    )


@pytest.mark.parametrize("control", WRONG_SIGN_KEYS)
def test_a_missing_wrong_sign_control_is_rejected(control: str) -> None:
    """The four Section 12.2 controls are an exact, closed key set."""
    module = _tool()
    document = _synthetic_document(module)
    del document["results"]["direct_convergence_cases"][0]["wrong_sign_residuals"][
        control
    ]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_direct_convergence_row_that_does_not_join_its_frame_cubes_is_rejected() -> (
    None
):
    """Section 14.2: its three certificate digests equal the same-fixture frame row."""
    module = _tool()
    document = _synthetic_document(module)
    row = document["results"]["direct_convergence_cases"][0]
    row["frozen_gauss64_cube_sha256"] = "0" * 64
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_an_analytic_piecewise_residual_above_its_limit_is_rejected() -> None:
    """Section 12.2's exposure-sinc oracle carries a fixed ``5e-12`` limit."""
    module = _tool()
    document = _synthetic_document(module)
    row = document["results"]["direct_convergence_cases"][0]
    row["analytic_piecewise_residual"] = 1e-9
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_hybrid_sum_that_is_not_the_component_sum_is_rejected() -> None:
    """Section 7.1 adds components in the fixed order before any product."""
    module = _tool()
    document = _synthetic_document(module)
    row = document["results"]["sky_component_cases"][0]
    row["expected_sum_sha256"] = "0" * 64
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_ring_nest_inequality_is_rejected() -> None:
    """Section 7.1: RING and NEST inputs yield identical coefficients."""
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["sky_component_cases"][0]["ring_nest_equal"] = False
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_complex64_row_borrowing_the_complex128_predicate_is_rejected() -> None:
    """Section 9: the complex64 contract never replaces the acceptance row."""
    module = _tool()
    document = _synthetic_document(module)
    row = document["results"]["backend_parity_cases"][0]
    row["dtype"] = "complex64"
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_an_estimate_below_the_measured_peak_is_rejected() -> None:
    """Section 9: the estimate is proved not smaller than the measured peak."""
    module = _tool()
    document = _synthetic_document(module)
    row = document["results"]["memory_cases"][0]
    row["measured_host_peak_bytes"] = row["estimated_peak_bytes"] + 1
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_memory_row_missing_a_section_9_component_is_rejected() -> None:
    """Section 9's seven components are reported separately and exhaustively."""
    module = _tool()
    document = _synthetic_document(module)
    row = document["results"]["memory_cases"][0]
    row["estimated_components"] = row["estimated_components"][:-1]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_schedule_digest_that_does_not_rebuild_is_rejected() -> None:
    """Section 11's schedule digest is rebuilt from the retained rows."""
    module = _tool()
    document = _synthetic_document(module)
    row = document["results"]["memory_cases"][0]
    row["schedule_rows"][0]["block_index"] = 7
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_native_peak_claiming_measured_without_a_value_is_rejected() -> None:
    """A null native peak names a measurement limitation, never ``measured``."""
    module = _tool()
    document = _synthetic_document(module)
    row = document["results"]["memory_cases"][0]
    row["measured_native_peak_bytes_reason"] = "measured"
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_rejection_row_that_allocated_first_is_rejected() -> None:
    """Section 8: failure precedes allocation and output-path creation."""
    module = _tool()
    document = _synthetic_document(module)
    document["results"]["rejection_cases"][0]["allocation_started"] = True
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_non_zero_command_exit_code_is_rejected() -> None:
    """Section 14.2: evidence commands require exit code zero."""
    module = _tool()
    document = _synthetic_document(module)
    document["commands"][0]["exit_code"] = 1
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_unsorted_claim_arrays_are_rejected() -> None:
    """``limitations`` and ``claims_not_licensed`` are sorted and unique."""
    module = _tool()
    document = _synthetic_document(module)
    document["claims_not_licensed"] = ["gpu_or_accelerator_support", "general_speedup"]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_canonical_json_sorts_keys_and_emits_no_whitespace() -> None:
    """Section 14's ``J(x)`` is key-sorted, compact and ASCII-escaped."""
    module = _tool()
    assert module.canonical_json({"b": 1.0, "a": [1e-7]}) == b'{"a":[1e-7],"b":1}'


@pytest.mark.parametrize(
    ("value", "text"),
    [(0.0, "0"), (1.0, "1"), (1e-7, "1e-7"), (1e21, "1e+21"), (-0.5, "-0.5")],
)
def test_canonical_numbers_use_the_ecmascript_spelling(value: float, text: str) -> None:
    """Section 14.0 fixes the number spelling as ``Number::toString``."""
    module = _tool()
    assert module.ecmascript_number(value) == text


def test_canonical_json_forbids_nan_and_infinity() -> None:
    """A non-finite number has no canonical spelling and is refused."""
    module = _tool()
    with pytest.raises(module.EvidenceError):
        module.canonical_json({"x": float("nan")})


def test_the_domain_digest_matches_its_printed_definition() -> None:
    """Section 14.0: ``D(d, p) = SHA256(d || NUL || U64(len(p)) || p)``."""
    module = _tool()
    payload = b"payload"
    expected = hashlib.sha256(
        b"radiosim.a.v1" + b"\x00" + len(payload).to_bytes(8, "big") + payload
    ).hexdigest()
    assert module.domain_digest("radiosim.a.v1", payload) == expected


def test_a_distinct_domain_gives_a_distinct_digest() -> None:
    """The domain is part of the preimage, not a label beside it."""
    module = _tool()
    assert module.domain_digest("radiosim.a.v1", b"x") != module.domain_digest(
        "radiosim.b.v1", b"x"
    )


# ---------------------------------------------------------------------------
# E2 state: authenticate the retained artifact and its introducing commit
# ---------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    """Return the stdout of one hermetic ``git`` invocation in this repository."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
    )
    return completed.stdout


def _locate_evidence_commit() -> str:
    """Return the unique commit that introduced the phase evidence artifact.

    Section 14.2 requires ``E2`` to be *located*, not assumed: the artifact is
    an added path, so the introducing commits on the current history are read
    with ``--diff-filter=A`` and there must be exactly one.  Two introductions
    would mean the artifact had been deleted and re-added, which is precisely
    the substitution the uniqueness clause exists to refuse.
    """
    introductions = _git(
        "log", "--diff-filter=A", "--format=%H", "HEAD", "--", ARTIFACT
    ).split()
    assert len(introductions) == 1, (
        f"{ARTIFACT} must be introduced by exactly one commit on HEAD's "
        f"ancestry; observed {introductions}"
    )
    located = introductions[0]
    assert GIT_SHA.fullmatch(located)
    return located


def _constant_spans(source: str) -> tuple[list[tuple[int, int]], list[list[Any]]]:
    """Return the token ranges of the two approved-constant assignments.

    A span runs from the constant's own ``NAME`` token to the ``NEWLINE`` that
    ends its logical line, so a value that the formatter wrapped in parentheses
    -- which it does for the 64-hex digest, whose inline form exceeds the line
    length -- is still exactly one span.
    """
    tokens = [
        token
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in (tokenize.ENCODING, tokenize.ENDMARKER)
    ]
    spans: list[tuple[int, int]] = []
    bodies: list[list[Any]] = []
    for index, token in enumerate(tokens):
        if (
            token.type != tokenize.NAME
            or token.string not in APPROVED_CONSTANT_NAMES
            or token.start[1] != 0
        ):
            continue
        stop = index
        while tokens[stop].type != tokenize.NEWLINE:
            stop += 1
        spans.append((index, stop + 1))
        bodies.append(tokens[index : stop + 1])
    assert len(spans) == len(APPROVED_CONSTANT_NAMES), (
        f"expected one assignment per approved constant; found {len(spans)}"
    )
    return spans, bodies


def _outside_spans(source: str) -> list[tuple[int, str]]:
    """Return the ``(type, string)`` token stream outside the two spans."""
    spans, _bodies = _constant_spans(source)
    tokens = [
        token
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in (tokenize.ENCODING, tokenize.ENDMARKER)
    ]
    excised = {index for start, stop in spans for index in range(start, stop)}
    return [
        (token.type, token.string)
        for index, token in enumerate(tokens)
        if index not in excised
    ]


def _assigned_literal(body: list[Any]) -> str:
    """Return the single value token of one approved-constant assignment."""
    values = [
        token
        for token in body
        if token.type in (tokenize.STRING, tokenize.NAME)
        and token.string not in (*APPROVED_CONSTANT_NAMES, "str", "None")
    ]
    names = [token for token in body if token.string == "None"]
    if not values:
        assert names, "an approved-constant assignment carries no value token"
        return "None"
    assert len(values) == 1, (
        "an approved-constant assignment must carry exactly one value token"
    )
    return values[0].string


def test_the_artifact_introducing_commit_directly_parents_the_approved_source() -> None:
    """Section 14.2's ``E2`` ancestry clause, skipped until the constants flip.

    ``E2`` is located from history rather than named, and its **direct** parent
    must be the approved ``S2``.  A merge commit is refused outright: an
    artifact introduced on a merge has no single source tree it was generated
    from, which is the whole point of binding the two.
    """
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_SOURCE_SHA is None:
        pytest.skip("the M2 evidence artifact is authorized at E2")
    located = _locate_evidence_commit()
    lineage = _git("rev-list", "--parents", "-n", "1", located).split()
    assert lineage[0] == located
    assert len(lineage) == 2, (
        f"the artifact-introducing commit {located} must be a non-merge commit "
        f"with exactly one parent; observed {lineage[1:]}"
    )
    assert lineage[1] == APPROVED_SOURCE_SHA, (
        f"the direct parent of {located} is {lineage[1]}, not the approved "
        f"source {APPROVED_SOURCE_SHA}"
    )
    payload = _git("show", f"{located}:{ARTIFACT}")
    assert (
        hashlib.sha256(payload.encode("utf-8")).hexdigest() == APPROVED_ARTIFACT_SHA256
    )


def test_the_e2_diff_writes_only_the_section_13_4_authorized_paths() -> None:
    """Section 13.4/14.2: ``E2`` adds the artifact and its record, nothing else."""
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_SOURCE_SHA is None:
        pytest.skip("the M2 evidence artifact is authorized at E2")
    located = _locate_evidence_commit()
    changed = set(
        _git("diff-tree", "--no-commit-id", "--name-only", "-r", located).split()
    )
    assert ARTIFACT in changed
    unauthorized = sorted(changed - E2_AUTHORIZED_PATHS)
    assert not unauthorized, (
        f"the E2 commit {located} writes {unauthorized}, which Section 13.4 "
        f"does not authorize; it may write only {sorted(E2_AUTHORIZED_PATHS)}"
    )
    if REPRODUCTION in changed:
        record = _git("show", f"{located}:{REPRODUCTION}")
        assert record.startswith(REPRODUCTION_FRONT_MATTER), (
            "the reproduction record must open with Section 14.2's exact MyST "
            "front matter"
        )


def test_the_e2_diff_changes_only_the_two_approved_constant_assignments() -> None:
    """Section 14.2: this module's own ``E2`` diff is the two constants alone.

    The comparison is a token stream taken **outside** the two assignment spans,
    which is what makes it survive the formatter wrapping the 64-hex digest in
    parentheses while still refusing any other edit -- an added import, a
    reworded docstring, a relaxed assertion, a deleted test.  Inside the spans
    only the value may move, from the ``None`` sentinel to the approved literal.
    """
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_SOURCE_SHA is None:
        pytest.skip("the M2 evidence artifact is authorized at E2")
    located = _locate_evidence_commit()
    parent = _git("rev-list", "--parents", "-n", "1", located).split()[1]
    before = _git("show", f"{parent}:{VALIDATOR}")
    after = _git("show", f"{located}:{VALIDATOR}")

    assert _outside_spans(before) == _outside_spans(after), (
        f"the E2 commit {located} changed this module outside the two approved "
        "constant assignments"
    )

    _spans_before, bodies_before = _constant_spans(before)
    _spans_after, bodies_after = _constant_spans(after)
    approved = (APPROVED_SOURCE_SHA, APPROVED_ARTIFACT_SHA256)
    for name, body_before, body_after, value in zip(
        APPROVED_CONSTANT_NAMES, bodies_before, bodies_after, approved, strict=True
    ):
        assert _assigned_literal(body_before) == "None", (
            f"{name} must be the null sentinel at the direct parent {parent}"
        )
        assert _assigned_literal(body_after) == f'"{value}"', (
            f"{name} at {located} is not the approved literal"
        )


def test_the_retained_artifact_authenticates_against_the_approved_constants() -> None:
    """Section 14.2's ``E2`` state, skipped until the constants are flipped."""
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_SOURCE_SHA is None:
        pytest.skip("the M2 evidence artifact is authorized at E2")
    path = REPOSITORY_ROOT / ARTIFACT
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == APPROVED_ARTIFACT_SHA256
    document = json.loads(payload.decode("utf-8"))
    module = _tool()
    module.validate_evidence_document(document)
    assert document["source_sha"] == APPROVED_SOURCE_SHA
    assert module.canonical_json(document) == payload
