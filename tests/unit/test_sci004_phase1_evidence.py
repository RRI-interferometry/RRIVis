"""Strict authentication of the SCI-004 phase-M1 evidence artifact.

``docs/development/sci004_mmode_design.md`` Sections 13.3, 14.2 and 14.4 freeze
this module's successor authority: it lands in ``S1`` with both approved
constants as the literal ``None``, the official evidence path **absent**, and
every synthetic strict schema and digest fixture passing.  ``E1`` then changes
*only* the two constants below, from ``None`` to the exact lower-case 40- and
64-hexadecimal literals, and adds the artifact and its reproduction record.  No
import, expression, annotation, key, surrounding token, or other literal in
either assignment may change, so the validator's own token stream outside those
two spans is comparable to its direct-parent ``S1`` bytes.

In the ``S1`` state the null constants require the official artifact to be
absent while the synthetic fixtures prove the schema rules.  In the ``E1`` state
the validator additionally requires the artifact's ``source_sha`` and the
constant to equal the approved ``S1``, authenticates the raw artifact bytes,
locates the unique artifact-introducing ``E1`` commit and requires its direct
parent to be ``S1``, and checks the ``S1..E1`` diff against Section 13.  It
deliberately does **not** require the current checkout or ``E1`` to equal
``source_sha``.

Importing this module loads only the Python standard library plus ``pytest``,
following ``tools/wp7_perf001_cpu_evidence.py``: an evidence-critical validator
must not depend on a package that is merely transitively present, because a lock
update could drop it and silently turn a hard authentication into a collection
error.  The generator at ``tools/sci004_mmode_phase1_evidence.py`` is the
normative producer; the checks below enforce the same structure, key order and
encodings in their own code rather than importing the producer's opinion of
them.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import sys
import tokenize
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Section 14.2's two approved constants.  ``E1`` replaces exactly these two
#: ``None`` literals and nothing else in this module.
APPROVED_SOURCE_SHA: str | None = "8dfc9af889c5d89f1783ac852f7d0cf6d4589740"
APPROVED_ARTIFACT_SHA256: str | None = (
    "c3a0ee6b72fb6e7c6013d40a30ed1d90ec0771cb0f91de6eb1862bc6ae60b86a"
)

TOOL = "tools/sci004_mmode_phase1_evidence.py"
ARTIFACT = "docs/development/sci004_mmode_phase1_evidence.json"
REPRODUCTION = "docs/development/sci004_mmode_phase1_evidence.md"
RED_RECORD = "docs/development/sci004_mmode_phase1_red_failures.json"
DEPENDENCY = "docs/development/sci004_mmode_phase1_wp7_dependency.json"

VALIDATOR = "tests/unit/test_sci004_phase1_evidence.py"

#: Section 13.3's complete ``E1`` write authority.  The commit that introduces
#: the artifact may touch these paths and nothing else.
E1_AUTHORIZED_PATHS: frozenset[str] = frozenset({ARTIFACT, REPRODUCTION, VALIDATOR})

#: Section 14.2's exact MyST front matter for the reproduction record.
REPRODUCTION_FRONT_MATTER = "---\norphan: true\n---"

#: The two spans Section 13.3 lets ``E1`` rewrite inside this module.
APPROVED_CONSTANT_NAMES: tuple[str, ...] = (
    "APPROVED_SOURCE_SHA",
    "APPROVED_ARTIFACT_SHA256",
)

GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

EVIDENCE_SCHEMA = "radiosim.sci004.mmode-phase1-evidence.v1"
RED_FAILURE_SCHEMA = "radiosim.sci004.mmode-phase1-red-failures.v1"
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

#: Section 14.2's exact M1 ``results`` key order.
RESULT_KEYS: tuple[str, ...] = (
    "dependency_certificate",
    "time_grid_cases",
    "frame_certificate_cases",
    "scalar_harmonic_cases",
    "packed_layout_cases",
    "transfer_cases",
    "strategy_cases",
    "capability_cases",
    "direct_identity_cases",
    "truncation_cases",
    "rejection_cases",
)

#: Section 14.2's exact six-row ``capability_cases`` inventory, in order.
CAPABILITY_ROWS: tuple[tuple[str, str], ...] = (
    ("property", "mmode-supports-polarization-false"),
    ("property", "rime-supports-polarization-true"),
    ("registry", "registry-includes-scalar-mmode"),
    ("rejection", "mmode-rejects-nonzero-q"),
    ("rejection", "mmode-rejects-nonzero-u"),
    ("rejection", "mmode-rejects-nonzero-v"),
)

TIER7_PROPERTY_NODEID = (
    "tests/characterization/test_tier7_current_behavior.py::"
    "test_mmode_m1_capability_truth"
)
TIER7_REGISTRY_NODEID = (
    "tests/unit/test_tier7_jones_acceptance.py::"
    "test_the_accepted_simulator_values_equal_the_registry_keys"
)
STOKES_NODEIDS: dict[str, str] = {
    "Q": (
        "tests/unit/test_simulator/test_sci004_strategy.py::"
        "test_mmode_m1_rejects_nonzero_stokes[Q]"
    ),
    "U": (
        "tests/unit/test_simulator/test_sci004_strategy.py::"
        "test_mmode_m1_rejects_nonzero_stokes[U]"
    ),
    "V": (
        "tests/unit/test_simulator/test_sci004_strategy.py::"
        "test_mmode_m1_rejects_nonzero_stokes[V]"
    ),
}

MMODE_M1_SCALAR_ONLY_MESSAGE = (
    "MModeSimulator phase M1 accepts Stokes I only; non-zero Q, U, or V "
    "requires accepted phase M2."
)


def _tool() -> Any:
    """Import the tracked generator without adding an import-time dependency."""
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    try:
        import sci004_mmode_phase1_evidence as module
    finally:
        sys.path.pop(0)
    return module


def _canonical(value: Any) -> bytes:
    return _tool().canonical_json(value)


def _synthetic_capability_rows() -> list[dict[str, Any]]:
    """Return the exact six-row capability array a passing artifact carries."""
    rows: list[dict[str, Any]] = [
        {
            "case_kind": "property",
            "case_id": "mmode-supports-polarization-false",
            "simulator": "mmode",
            "property": "supports_polarization",
            "expected_boolean": False,
            "observed_boolean": False,
            "tier7_test_nodeid": TIER7_PROPERTY_NODEID,
            "pass": True,
        },
        {
            "case_kind": "property",
            "case_id": "rime-supports-polarization-true",
            "simulator": "rime",
            "property": "supports_polarization",
            "expected_boolean": True,
            "observed_boolean": True,
            "tier7_test_nodeid": TIER7_PROPERTY_NODEID,
            "pass": True,
        },
        {
            "case_kind": "registry",
            "case_id": "registry-includes-scalar-mmode",
            "expected_names": ["mmode", "rime"],
            "observed_names": ["mmode", "rime"],
            "tier7_test_nodeid": TIER7_REGISTRY_NODEID,
            "pass": True,
        },
    ]
    for field in ("Q", "U", "V"):
        rows.append(
            {
                "case_kind": "rejection",
                "case_id": f"mmode-rejects-nonzero-{field.lower()}",
                "simulator": "mmode",
                "stokes_field": field,
                "configured_value_f64be": "3ff0000000000000",
                "exception_type": (
                    "radiosim.io.config_resolution.UnsupportedConfigError"
                ),
                "issue_code": "mmode_m1_scalar_only",
                "exact_message": MMODE_M1_SCALAR_ONLY_MESSAGE,
                "test_nodeid": STOKES_NODEIDS[field],
                "pass": True,
            }
        )
    return rows


#: The synthetic frame fixture's dimensions.  They are deliberately tiny --
#: two directions, three samples, one baseline, one frequency -- because the
#: rules under test are structural, and a fixture at the production scale would
#: make a schema failure unreadable.
SYNTHETIC_DIRECTIONS: tuple[str, ...] = (
    "point:0:0",
    "transfer_quadrature:production:1:0",
)
SYNTHETIC_SAMPLES = 3
SYNTHETIC_BASELINES = 1
SYNTHETIC_FREQUENCIES = 1


#: The synthetic fixture's exact retained centre turns and ``tau`` bits.  The
#: mask expansion derives every ``alpha_rad`` from these, so they are the whole
#: preimage the economy form relies on.
SYNTHETIC_CENTER_TURNS: tuple[str, ...] = ("0/1", "1/3", "2/3")
SYNTHETIC_TAU_F64BE = "401921fb54442d18"


def _synthetic_time_grid_row() -> dict[str, Any]:
    """Return a time-grid row carrying the two canonical grid objects.

    Only the fields the frame join and the membership expansion consume are
    load-bearing here; the rest are structurally valid placeholders, because the
    rules under test are the join and the expansion rather than Section 3.1's
    own residual predicates.
    """
    sixty_four = "0" * 63 + "1"
    return {
        "fixture_id": "mmode_point_stokes_i",
        "sidereal_samples": SYNTHETIC_SAMPLES,
        "integration_fraction_f64be": "3ff0000000000000",
        "canonical_era_turn_grid": {
            "schema_version": "radiosim.mmode-era-turn-grid.v1",
            "sidereal_samples": SYNTHETIC_SAMPLES,
            "center_turns": list(SYNTHETIC_CENTER_TURNS),
            "lower_edge_turns": ["-1/6", "1/6", "1/2"],
            "upper_edge_turns": ["1/6", "1/2", "5/6"],
            "exposure_width_turn": "1/3",
            "horizon_lo_turn": "-1/6",
            "horizon_hi_turn": "5/6",
            "integration_fraction_f64be": "3ff0000000000000",
            "integration_fraction_ratio": "1/1",
        },
        "iers_table_sha256": sixty_four,
        "era_center_turn_sha256": sixty_four,
        "era_lower_edge_turn_sha256": sixty_four,
        "era_upper_edge_turn_sha256": sixty_four,
        "canonical_era_turn_grid_sha256": sixty_four,
        "tau_f64be": SYNTHETIC_TAU_F64BE,
        "delta_alpha_rad_f64be": "4010c15238000000",
        "horizon_lo_rad_f64be": "bff0c15238000000",
        "horizon_hi_rad_f64be": "4014f1a970000000",
        "era_center_rad_sha256": sixty_four,
        "era_lower_edge_rad_sha256": sixty_four,
        "era_upper_edge_rad_sha256": sixty_four,
        "canonical_era_grid": {
            "schema_version": "radiosim.mmode-era-grid.v1",
            "canonical_era_turn_grid_sha256": sixty_four,
            "era_center_turn_sha256": sixty_four,
            "era_lower_edge_turn_sha256": sixty_four,
            "era_upper_edge_turn_sha256": sixty_four,
            "era_center_rad_sha256": sixty_four,
            "era_lower_edge_rad_sha256": sixty_four,
            "era_upper_edge_rad_sha256": sixty_four,
            "tau_f64be": SYNTHETIC_TAU_F64BE,
            "delta_alpha_rad_f64be": "4010c15238000000",
            "horizon_lo_rad_f64be": "bff0c15238000000",
            "horizon_hi_rad_f64be": "4014f1a970000000",
        },
        "canonical_era_grid_sha256": sixty_four,
        "era_center_max_residual_rad": 0.0,
        "era_center_limit_rad": 1e-9,
        "era_step_max_residual_rad": 0.0,
        "era_step_limit_rad": 1e-9,
        "ut1_utc_roundtrip_seconds": 0.0,
        "ut1_utc_roundtrip_limit_seconds": 1e-6,
        "utc_manifest": {"schema_version": "radiosim.mmode-utc.v1"},
        "utc_sha256": sixty_four,
        "ut1_manifest": {"schema_version": "radiosim.mmode-ut1.v1"},
        "ut1_sha256": sixty_four,
        "integration_time_seconds_sha256": sixty_four,
        "pass": True,
    }


def _synthetic_scan_crossing_row(direction_id: str) -> dict[str, Any]:
    """Return one verbatim Section 12.1 crossing row in its exact field order."""
    return {
        "direction_id": direction_id,
        "cell_index": 1,
        "turn_lo": "1/4",
        "turn_hi": "1/2",
        "classification": "scan_crossing",
        "f_lo_f64be": "bfd0000000000000",
        "f_hi_f64be": "3fd0000000000000",
        "ceiling_margin_f64be": "0000000000000000",
        "left_sign": -1,
        "right_sign": 1,
        "root_turn_lo": "1/4",
        "root_turn_hi": "1/2",
        "root_orientation": "rising",
        "root_residual_f64be": "0000000000000000",
    }


def _synthetic_guard_row(direction_id: str) -> dict[str, Any]:
    """Return the guard row flanking the synthetic crossing's upper end.

    It abuts the enclosure exactly, carries null root fields and an exact zero
    margin, and its root-adjacent endpoint sign is zero -- the one place
    Section 12.1 permits a zero sign, because the numerator vanishes there.
    """
    return {
        "direction_id": direction_id,
        "cell_index": 2,
        "turn_lo": "1/2",
        "turn_hi": "500000000001/1000000000000",
        "classification": "guard_interval",
        "f_lo_f64be": "0000000000000000",
        "f_hi_f64be": "3e45798ee2308c3a",
        "ceiling_margin_f64be": "0000000000000000",
        "left_sign": 0,
        "right_sign": 1,
        "root_turn_lo": None,
        "root_turn_hi": None,
        "root_orientation": None,
        "root_residual_f64be": None,
    }


def _synthetic_frame_row() -> dict[str, Any]:
    """Return a frame row in Section 12.1's retained economy forms.

    Every ledger here is the *projection* the corrected Section 12.1 retains:
    the scan as crossing rows plus one summary row per direction, the membership
    census as one visibility-mask row per direction, and the direct partition as
    its own split rows -- not the sixteen-million-row terminal array or the
    ``D*N`` per-sample rows the pre-correction letter demanded.
    """
    sixty_four = "0" * 63 + "1"
    cells = 4 * SYNTHETIC_SAMPLES * SYNTHETIC_BASELINES * SYNTHETIC_FREQUENCIES
    directions = list(SYNTHETIC_DIRECTIONS)
    mask = "00"  # three samples, zero-padded to one whole byte
    row: dict[str, Any] = {
        "fixture_id": "mmode_point_stokes_i",
        "certificate_sha256": sixty_four,
        "site_manifest": {"schema_version": "radiosim.mmode-site.v1"},
        "site_sha256": sixty_four,
        "input_identity_sha256": sixty_four,
        "iers_table_sha256": sixty_four,
        "frame_matrix_manifest": {"schema_version": "radiosim.mmode-frame-matrices.v1"},
        "frame_matrix_sha256": sixty_four,
        "canonical_era_turn_grid_sha256": sixty_four,
        "canonical_era_grid_sha256": sixty_four,
        "pm_source_unit": "arcsec",
        "pom00_argument_unit": "rad",
        "xp0_arcsec": "0000000000000000",
        "yp0_arcsec": "0000000000000000",
        "das2r_rad_per_arcsec": "3ea0f2b17c8d0e21",
        "xp0_rad": "0000000000000000",
        "yp0_rad": "0000000000000000",
        "sp0_rad": "0000000000000000",
        "diagnostic_qcheck_nsides": [2],
        "transfer_grid_catalog": [],
        "transfer_grid_catalog_sha256": sixty_four,
        "direction_rows": [{"direction_id": identifier} for identifier in directions],
        "direction_ledger_sha256": sixty_four,
        "horizon_scan_manifest": {
            "schema_version": "radiosim.mmode-operational-horizon-scan.v1"
        },
        "horizon_scan_sha256": sixty_four,
        "horizon_scan_crossing_rows": [
            row
            for identifier in directions
            for row in (
                _synthetic_scan_crossing_row(identifier),
                _synthetic_guard_row(identifier),
            )
        ],
        "horizon_scan_summary_rows": [
            {
                "direction_id": identifier,
                "terminal_cell_count": 6,
                "boundary_evaluation_count": 7,
                "crossing_count": 1,
                "min_ceiling_margin_f64be": "3f50624dd2f1a9fc",
            }
            for identifier in directions
        ],
        "horizon_scan_ledger_sha256": sixty_four,
        "horizon_root_pair_rows": [
            {
                "direction_id": identifier,
                "frozen_root_count": 1,
                "operational_root_count": 1,
                "orientation_mismatch_count": 0,
                "pairs": [],
            }
            for identifier in directions
        ],
        "horizon_root_pair_ledger_sha256": sixty_four,
        "horizon_slab_rows": [],
        "horizon_slab_ledger_sha256": sixty_four,
        "horizon_sign_interval_rows": [],
        "horizon_sign_interval_ledger_sha256": sixty_four,
        "horizon_membership_mask_rows": [
            {
                "direction_id": identifier,
                "sample_count": SYNTHETIC_SAMPLES,
                "frozen_visible_mask_hex": mask,
                "operational_visible_mask_hex": mask,
                "mismatch_count": 0,
            }
            for identifier in directions
        ],
        "horizon_membership_ledger_sha256": "",
        "direct_split_rows": [],
        "direct_split_ledger_sha256": sixty_four,
        "direct_integrand_enclosure_manifest": {
            "schema_version": "radiosim.mmode-direct-integrand-enclosure.v1"
        },
        "direct_integrand_enclosure_sha256": sixty_four,
        "sidereal_samples": SYNTHETIC_SAMPLES,
        "quadrature_nside": 1,
        "n_baselines": SYNTHETIC_BASELINES,
        "n_frequencies": SYNTHETIC_FREQUENCIES,
        "n_correlations": 4,
        "expected_point_direction_count": 1,
        "evaluated_point_direction_count": 1,
        "expected_native_healpix_direction_count": 0,
        "evaluated_native_healpix_direction_count": 0,
        "expected_production_transfer_direction_count": 1,
        "evaluated_production_transfer_direction_count": 1,
        "expected_diagnostic_transfer_direction_count": 0,
        "evaluated_diagnostic_transfer_direction_count": 0,
        "expected_transfer_quadrature_direction_count": 1,
        "evaluated_transfer_quadrature_direction_count": 1,
        "expected_direction_count": len(directions),
        "evaluated_direction_count": len(directions),
        "expected_phase_comparison_count": 12,
        "evaluated_phase_comparison_count": 12,
        "expected_horizon_trajectory_count": len(directions),
        "evaluated_horizon_trajectory_count": len(directions),
        "expected_horizon_root_pair_row_count": len(directions),
        "evaluated_horizon_root_pair_row_count": len(directions),
        "expected_horizon_membership_count": len(directions) * SYNTHETIC_SAMPLES,
        "evaluated_horizon_membership_count": len(directions) * SYNTHETIC_SAMPLES,
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
        "horizon_isolation_interval_count": 12,
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
        "phase_max_rad": 1.0e-9,
        "phase_limit_rad": 5e-3,
        "expected_cube_cell_count": cells,
        "evaluated_frozen_gauss64_cube_cell_count": cells,
        "evaluated_frozen_gauss128_cube_cell_count": cells,
        "evaluated_operational_gauss64_cube_cell_count": cells,
        "evaluated_operational_gauss128_cube_cell_count": cells,
        "compared_frozen_gauss_change_cell_count": cells,
        "compared_operational_gauss_change_cell_count": cells,
        "evaluated_frozen_enclosure_error_cell_count": cells,
        "evaluated_operational_enclosure_error_cell_count": cells,
        "frozen_gauss64_cube_sha256": sixty_four,
        "frozen_gauss128_cube_sha256": sixty_four,
        "operational_gauss64_cube_sha256": sixty_four,
        "operational_gauss128_cube_sha256": sixty_four,
        "frozen_enclosure_error_cube_sha256": sixty_four,
        "operational_enclosure_error_cube_sha256": sixty_four,
        "direct_gauss_scale_jy": 1.0,
        "frozen_gauss_change_max_jy": 0.0,
        "operational_gauss_change_max_jy": 0.0,
        "direct_gauss_change_max_jy": 0.0,
        "direct_gauss_change_limit_jy": 1e-11,
        "cube_scale_jy": 1.0,
        "cube_max_jy": 0.0,
        "cube_limit_jy": 5e-5 + 1e-10,
        "cube_l2": 0.0,
        "cube_l2_limit": 5e-5,
        "direction_diagnostic_max_rad": 1.0e-9,
        "direction_diagnostic_argmax_id": directions[0],
        "direction_diagnostic_argmax_phase": "0/1",
        "basis_diagnostic_max_rad": 1.0e-9,
        "basis_diagnostic_argmax_id": directions[0],
        "basis_diagnostic_argmax_phase": "0/1",
        "pass": True,
    }
    # The retained digest is the *expansion* of the retained masks, computed by
    # the same rule the strict validator applies, so the positive control cannot
    # pass with a placeholder and the negative controls below genuinely fail.
    row["horizon_membership_ledger_sha256"] = _tool().expand_membership_ledger(
        row["horizon_membership_mask_rows"],
        SYNTHETIC_CENTER_TURNS,
        SYNTHETIC_TAU_F64BE,
    )
    return row


def _synthetic_transfer_sample_rows(
    grids: int, baselines: int, frequencies: int
) -> list[dict[str, Any]]:
    """Return Section 7.3's concatenation rows: one per grid and output cell."""
    sixty_four = "0" * 63 + "1"
    rows: list[dict[str, Any]] = []
    for grid in range(grids):
        for baseline in range(baselines):
            for frequency in range(frequencies):
                for correlation in range(4):
                    for field_index, field_name in enumerate(("I", "+2", "-2", "V")):
                        rows.append(
                            {
                                "grid_id": (
                                    "production:8" if grid == 0 else "diagnostic:16"
                                ),
                                "baseline_index": baseline,
                                "frequency_index": frequency,
                                "correlation_index": correlation,
                                "field_index": field_index,
                                "field_name": field_name,
                                "resolved_lmax": 16 if grid == 0 else 24,
                                "resolved_mmax": 16 if grid == 0 else 24,
                                "block_table_sha256": sixty_four,
                                "direction_count": 768 if grid == 0 else 3072,
                                "packed_sample_value_count": (
                                    768 * 289 if grid == 0 else 3072 * 625
                                ),
                                "concatenation_sha256": sixty_four,
                            }
                        )
    return rows


def _synthetic_truncation_row() -> dict[str, Any]:
    """Return a truncation row on the ``sci004_two_tier_direct.v3`` surface.

    The numbers are the qualified M1 fixture's own measured values: tier 1a at
    the ``1e-13`` level against its ``1e-8`` limit, the recorded with-horizon
    shell inside its reviewed budget, and the three-level deficit sequence whose
    quarter-to-full factor is ``6.12`` against the fixed floor of two.  The
    transfer-sample ledger is the corrected Section 7.3 concatenation form --
    ``(1+len(Q_diag))*B*F*C*4 = 288`` rows for this fixture, one per catalogue
    grid and output cell, not one per direction.
    """
    sixty_four = "0" * 63 + "1"
    transfer_rows = _synthetic_transfer_sample_rows(2, 3, 3)
    row: dict[str, Any] = {
        "fixture_id": "mmode_point_stokes_i",
        "input_identity_sha256": sixty_four,
        "frame_certificate_sha256": sixty_four,
        "direction_ledger_sha256": sixty_four,
        "transfer_grid_catalog_sha256": sixty_four,
        "production_transfer_grid_id": "production:8",
        "diagnostic_transfer_grid_ids": ["diagnostic:16"],
        "diagnostic_grid_joins": [],
        "lmax": 16,
        "mmax": 16,
        "quadrature_nside": 8,
        "lcheck": 24,
        "mcheck": 24,
        "qcheck": 16,
        "sidereal_samples": 49,
        "cube_shape": [49, 3, 3, 4],
        "frozen_gauss128_cube_sha256": sixty_four,
        "frozen_enclosure_error_cube_sha256": sixty_four,
        "mmode_cube_sha256": sixty_four,
        "direct_scale_jy": 2.251504,
        "expected_output_cell_count": 1764,
        "evaluated_frozen_direct_cell_count": 1764,
        "evaluated_frozen_error_cell_count": 1764,
        "evaluated_mmode_cell_count": 1764,
        "compared_output_cell_count": 1764,
        "direct_coverage": {},
        "direct_coverage_sha256": sixty_four,
        "horizon_free_shell_max_jy": 1.285638e-13,
        "horizon_free_shell_l2": 5.856862e-14,
        "horizon_free_shell_max_limit_jy": 2.261504e-08,
        "horizon_free_shell_l2_limit": 1e-8,
        "quadrature_shell_max_jy": 5.80459e-02,
        "quadrature_shell_l2": 1.086599e-02,
        "quadrature_budget_jy": 0.1,
        "deficit_max_jy": 1.169865e-01,
        "deficit_l2": 3.07602e-02,
        "deficit_max_quarter_jy": 7.159405e-01,
        "deficit_max_half_jy": 1.846944e-01,
        "convergence_factor": 6.119855,
        "truncation_budget_jy": 0.2,
        "expected_shell_comparison_cell_count": 7056,
        "evaluated_shell_comparison_cell_count": 7056,
        "expected_transfer_sample_row_count": 288,
        "evaluated_transfer_sample_row_count": 288,
        "expected_field_block_count": 7056,
        "evaluated_field_block_count": 7056,
        "shell_coverage": {
            "schema_version": "radiosim.mmode-shell-coverage.v1",
            "transfer_sample_rows": transfer_rows,
        },
        "shell_coverage_sha256": sixty_four,
        "quadrature_diagnostic_max_jy": 5.80459e-02,
        "l_tail_diagnostic_max_jy": 1.0e-02,
        "m_tail_diagnostic_max_jy": 1.0e-03,
        "combined_local_diagnostic_max_jy": 6.0e-02,
        "field_block_diagnostic_max_jy": 2.0e-02,
        "shell_diagnostic_reference_jy": 2.3e-06,
        "pass": True,
    }
    return row


def _synthetic_envelope() -> dict[str, Any]:
    """Return a complete synthetic envelope that satisfies every Section 14.2 rule."""
    forty = "0" * 39 + "1"
    sixty_four = "0" * 63 + "1"
    command = {
        "argv": ["pixi", "run", "test", "--", "-m", "not slow"],
        "cwd": ".",
        "pixi_environment": "default",
        "started_at_utc": "2026-08-22T00:00:00Z",
        "duration_seconds": 1.5,
        "exit_code": 0,
        "stdout_sha256": sixty_four,
        "stderr_sha256": sixty_four,
    }
    certificate = {
        "schema_version": "radiosim.perf001.cpu_acceptance_certificate.v1",
        "acceptance_commit": forty,
        "evidence_commit": forty,
        "generating_source_sha": forty,
        "descendant_commit": forty,
        "artifact_path": "docs/development/wp7_perf001_cpu_evidence.json",
        "artifact_sha256": sixty_four,
        "cpu_evidence_tool_sha256": sixty_four,
        "production_record_validator_sha256": sixty_four,
        "production_harness_sha256": sixty_four,
        "pixi_manifest_sha256": sixty_four,
        "pixi_lock_sha256": sixty_four,
        "evidence_diff_paths": [],
        "acceptance_diff_paths": [],
        "verdict": "CPU_ACCEPTED_P_E_HARDWARE_GATED",
        "passed": True,
    }
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "phase": "M1",
        "status": "candidate",
        "generated_at_utc": "2026-08-22T00:00:00Z",
        "design_sha": forty,
        "red_commit_sha": forty,
        "source_sha": forty,
        "evidence_commit_sha": None,
        "evidence_commit_sha_reason": SELF_REFERENCE_REASON,
        "working_tree_clean": True,
        "environment": {
            "python": "3.11.13",
            "platform": "darwin",
            "machine": "arm64",
            "pixi_environment": "default",
            "pixi_lock_sha256": sixty_four,
            "astropy_version": "7.1.0",
            "erfa_version": "2.0.1.5",
            "iers_package_version": "0.2025.1.1.0.0.0",
            "iers_table_sha256": sixty_four,
            "numeric_packages": {
                "numpy": "2.3.2",
                "scipy": "1.17.0",
                "healpy": "1.19.0",
                "jax": "0.7.0",
                "dask": "2025.1.0",
            },
        },
        "source_identities": {
            "git_tree_sha256": sixty_four,
            "pixi_manifest_sha256": sixty_four,
            "pixi_lock_sha256": sixty_four,
            "convention_identity_sha256": sixty_four,
            "fixture_input_rows": [
                {
                    "fixture_id": "mmode_point_stokes_i",
                    "input_identity_manifest": {},
                    "input_identity_sha256": sixty_four,
                }
            ],
            "input_identity_set_sha256": sixty_four,
        },
        "red_failure_record": {
            "path": RED_RECORD,
            "sha256": sixty_four,
            "schema_version": RED_FAILURE_SCHEMA,
            "pre_fix_source_sha": forty,
            "validated": True,
        },
        "results": {
            "dependency_certificate": {
                "path": DEPENDENCY,
                "raw_sha256": sixty_four,
                "certificate": certificate,
            },
            "time_grid_cases": [_synthetic_time_grid_row()],
            "frame_certificate_cases": [_synthetic_frame_row()],
            "scalar_harmonic_cases": [],
            "packed_layout_cases": [],
            "transfer_cases": [],
            "strategy_cases": [],
            "capability_cases": _synthetic_capability_rows(),
            "direct_identity_cases": [
                {
                    "fixture_id": "rime_point_reference",
                    "rime_before_sha256": sixty_four,
                    "rime_after_sha256": sixty_four,
                    "scientific_before_sha256": sixty_four,
                    "scientific_after_sha256": sixty_four,
                    "byte_identical": True,
                    "pass": True,
                }
            ],
            "truncation_cases": [_synthetic_truncation_row()],
            "rejection_cases": [],
        },
        "commands": [command],
        "limitations": [
            "no accelerator run of the m-mode solver has been measured",
        ],
        "claims_not_licensed": [
            "general_speedup",
            "gpu_or_accelerator_support",
            "polarized_mmode_support",
        ],
    }


# ---------------------------------------------------------------------------
# S1 state: the official artifact must be absent
# ---------------------------------------------------------------------------


def test_the_approved_constants_are_null_sentinels_before_e1() -> None:
    """Section 13.5: at ``S1`` both approved digests are the literal ``None``.

    The pair moves together.  A half-flipped module would authenticate an
    artifact against a source it never approved, which is exactly the
    substitution the two-constant rule exists to prevent.
    """
    if APPROVED_SOURCE_SHA is None or APPROVED_ARTIFACT_SHA256 is None:
        assert APPROVED_SOURCE_SHA is None
        assert APPROVED_ARTIFACT_SHA256 is None
        return
    assert GIT_SHA.fullmatch(APPROVED_SOURCE_SHA)
    assert SHA256.fullmatch(APPROVED_ARTIFACT_SHA256)


def test_the_official_evidence_artifact_is_absent_in_the_s1_state() -> None:
    """Section 14.2: null constants require the not-yet-authorized paths absent."""
    if APPROVED_ARTIFACT_SHA256 is not None:
        return
    assert not (REPOSITORY_ROOT / ARTIFACT).exists()
    assert not (REPOSITORY_ROOT / REPRODUCTION).exists()


def test_the_tracked_generator_and_its_inputs_exist_at_s1() -> None:
    """Section 14.4: a clean exact ``S1`` already contains the producing bytes."""
    assert (REPOSITORY_ROOT / TOOL).is_file()
    assert (REPOSITORY_ROOT / RED_RECORD).is_file()
    assert (REPOSITORY_ROOT / DEPENDENCY).is_file()


def test_the_generator_imports_only_the_standard_library() -> None:
    """An evidence-critical generator carries no transitive package dependency."""
    source = (REPOSITORY_ROOT / TOOL).read_text(encoding="utf-8")
    for forbidden in ("import numpy", "import astropy", "import pytest", "import yaml"):
        assert forbidden not in source, forbidden


def test_the_generator_refuses_to_produce_anything_from_a_dirty_tree(
    tmp_path: Path,
) -> None:
    """Section 14.2: the common pre-output check fails closed on a dirty tree.

    The refusal must be caused by the dirt, so the test *creates* it: an
    untracked marker file inside the repository makes
    ``git status --porcelain=v1 --untracked-files=all`` non-empty for the
    duration of the run and is removed afterwards.  Asserting the refusal
    without dirtying the tree would pass for whatever reason the generator
    happened to refuse for -- including a generator that refuses
    unconditionally -- which is exactly the blind assertion this replaces.
    """
    del tmp_path
    module = _tool()
    artifact = REPOSITORY_ROOT / ARTIFACT
    before = artifact.read_bytes() if artifact.exists() else None
    marker = REPOSITORY_ROOT / ".sci004-dirty-tree-probe"
    marker.write_text("probe\n", encoding="utf-8")
    try:
        completed = subprocess.run(
            [sys.executable, str(REPOSITORY_ROOT / TOOL), "generate"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        marker.unlink()
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr.startswith(module.PREFLIGHT + ": ")
    assert "not globally clean" in completed.stderr
    # The refusal happened before any output could be opened, so the declared
    # output set is exactly as it was: absent at ``S1``, and byte-identical at
    # ``E1`` where the artifact legitimately exists.
    after = artifact.read_bytes() if artifact.exists() else None
    assert after == before


def test_the_generator_produces_at_a_clean_source_rather_than_refusing() -> None:
    """Section 14.2/14.4: ``generate`` is bound to a venue, not prohibited.

    A generator whose ``generate`` refuses unconditionally would satisfy the
    dirty-tree test above for the wrong reason, so this pins the complementary
    fact directly in the tracked bytes: the sub-command runs the preflight and
    then builds, validates and publishes, and it carries no unconditional
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
# Synthetic strict schema fixtures
# ---------------------------------------------------------------------------


def test_the_synthetic_envelope_satisfies_every_section_14_2_rule() -> None:
    """The fixture is a positive control: the rules below reject deviations."""
    module = _tool()
    envelope = module.validate_evidence_document(_synthetic_envelope())
    assert tuple(envelope) == ENVELOPE_KEYS
    assert tuple(envelope["results"]) == RESULT_KEYS


@pytest.mark.parametrize("key", ENVELOPE_KEYS)
def test_a_missing_top_level_key_is_rejected(key: str) -> None:
    """Section 14: every object rejects unknown or missing keys."""
    module = _tool()
    document = _synthetic_envelope()
    del document[key]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_an_unknown_top_level_key_is_rejected() -> None:
    """An extra key is a different document, not a superset of this one."""
    module = _tool()
    document = _synthetic_envelope()
    document["extra"] = 1
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_non_null_evidence_commit_sha_is_rejected() -> None:
    """Section 14.4: ``E`` artifacts use a null self SHA, bound later by ``A``."""
    module = _tool()
    document = _synthetic_envelope()
    document["evidence_commit_sha"] = "0" * 40
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_reworded_self_reference_reason_is_rejected() -> None:
    """The reason is an exact literal, not a paraphrase."""
    module = _tool()
    document = _synthetic_envelope()
    document["evidence_commit_sha_reason"] = "self reference"
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


@pytest.mark.parametrize("index", range(6))
def test_a_reordered_capability_row_fails_m1(index: int) -> None:
    """Section 14.2: missing, duplicate, reordered or inherited rows fail M1."""
    module = _tool()
    document = _synthetic_envelope()
    rows = document["results"]["capability_cases"]
    rows[index], rows[(index + 1) % 6] = rows[(index + 1) % 6], rows[index]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_an_m2_flipped_polarization_row_fails_m1() -> None:
    """Only accepted M2 may flip the m-mode polarization property to true."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["capability_cases"][0]["observed_boolean"] = True
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_registry_row_missing_the_second_key_fails() -> None:
    """Section 2's invariant is exact: the registry keys are ``mmode`` and ``rime``."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["capability_cases"][2]["observed_names"] = ["rime"]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_changed_direct_identity_digest_is_rejected() -> None:
    """Section 13.3: M1 must prove the wrapper leaves every direct path unchanged."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["direct_identity_cases"][0]["rime_after_sha256"] = "1" * 64
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_the_synthetic_frame_row_carries_the_economy_projection() -> None:
    """The fixture is a positive control for Section 12.1's corrected letter."""
    module = _tool()
    row = module.validate_frame_row(_synthetic_frame_row(), "frame")
    assert set(row) == set(module.FRAME_ROW_KEYS)
    assert len(row["horizon_membership_mask_rows"]) == len(SYNTHETIC_DIRECTIONS)
    assert len(row["horizon_scan_summary_rows"]) == len(SYNTHETIC_DIRECTIONS)


def test_a_frame_row_embedding_the_per_sample_membership_array_is_rejected() -> None:
    """Section 12.1: the retained form is the mask row, not ``D*N`` rows.

    The pre-correction letter embedded one row per direction and sample centre.
    Re-embedding that array is now a schema failure, because the mask row's key
    set is exact and a per-sample row does not have it.
    """
    module = _tool()
    row = _synthetic_frame_row()
    row["horizon_membership_mask_rows"] = [
        {
            "direction_id": identifier,
            "sample_index": index,
            "sample_turn": "0/1",
            "alpha_rad_f64be": "0000000000000000",
            "frozen_visible": True,
            "operational_visible": True,
            "match": True,
        }
        for identifier in SYNTHETIC_DIRECTIONS
        for index in range(SYNTHETIC_SAMPLES)
    ]
    with pytest.raises(module.EvidenceError):
        module.validate_frame_row(row, "frame")


def test_a_flipped_mask_bit_breaks_the_expanded_ledger_digest() -> None:
    """Section 12.1: the expansion is what makes the mask form lossless.

    The retained digest covers the complete ``D*N`` per-sample array, so a mask
    the reviewer cannot expand back to it is a substituted census, not a
    compression.
    """
    module = _tool()
    envelope = _synthetic_envelope()
    frame = envelope["results"]["frame_certificate_cases"][0]
    frame["horizon_membership_mask_rows"][0]["frozen_visible_mask_hex"] = "80"
    frame["horizon_membership_mask_rows"][0]["mismatch_count"] = 1
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(envelope)


def test_a_frame_row_without_its_time_grid_row_is_rejected() -> None:
    """Section 14.3: a frame row joins the same-fixture canonical grid objects."""
    module = _tool()
    envelope = _synthetic_envelope()
    envelope["results"]["time_grid_cases"] = []
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(envelope)


def test_a_mask_row_whose_sample_count_is_not_n_is_rejected() -> None:
    """Section 12.1: every mask row covers exactly ``N`` sample centres."""
    module = _tool()
    row = _synthetic_frame_row()
    row["horizon_membership_mask_rows"][0]["sample_count"] = SYNTHETIC_SAMPLES + 1
    with pytest.raises(module.EvidenceError):
        module.validate_frame_row(row, "frame")


def test_a_mask_row_mismatch_that_the_masks_do_not_show_is_rejected() -> None:
    """Section 4.2/14.2: both counters are recomputed from the masks.

    The per-direction total and the gating outside-slab counter are recomputed
    from the same mask expansion against the retained slab geometry, so a row
    that declares a mismatch its masks do not contain cannot pass by asserting
    it.
    """
    module = _tool()
    envelope = _synthetic_envelope()
    frame = envelope["results"]["frame_certificate_cases"][0]
    frame["horizon_membership_mask_rows"][0]["mismatch_count"] = 1
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(envelope)


def test_an_outside_slab_membership_mismatch_is_rejected() -> None:
    """Section 4.2: outside-slab disagreement is fatal, not attributable.

    A centre outside every mismatch slab where the two independent models
    disagree is exactly the failure the certificate exists to catch, so the
    recomputed counter must be zero and the declared counter must equal it.
    """
    module = _tool()
    envelope = _synthetic_envelope()
    frame = envelope["results"]["frame_certificate_cases"][0]
    mask = frame["horizon_membership_mask_rows"][0]
    mask["operational_visible_mask_hex"] = "80"
    mask["mismatch_count"] = 1
    frame["horizon_membership_mismatches"] = 1
    frame["horizon_membership_ledger_sha256"] = module.expand_membership_ledger(
        frame["horizon_membership_mask_rows"],
        SYNTHETIC_CENTER_TURNS,
        SYNTHETIC_TAU_F64BE,
    )
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(envelope)


def test_a_scan_summary_sum_that_contradicts_the_counter_is_rejected() -> None:
    """Section 12.1: ``horizon_isolation_interval_count`` is the summary sum."""
    module = _tool()
    row = _synthetic_frame_row()
    row["horizon_scan_summary_rows"][0]["terminal_cell_count"] = 4
    with pytest.raises(module.EvidenceError):
        module.validate_frame_row(row, "frame")


def test_the_synthetic_projection_retains_its_guard_rows() -> None:
    """Section 12.1: the retained projection carries each crossing's flanks."""
    module = _tool()
    row = module.validate_frame_row(_synthetic_frame_row(), "frame")
    kinds = [entry["classification"] for entry in row["horizon_scan_crossing_rows"]]
    assert kinds.count("guard_interval") == len(SYNTHETIC_DIRECTIONS)
    assert kinds.count("scan_crossing") == len(SYNTHETIC_DIRECTIONS)


def test_an_orphan_guard_row_is_rejected() -> None:
    """Section 12.1: a guard must abut its crossing's enclosure or another guard.

    A residue with no crossing to flank is a deep tangency, not a guard, and the
    partition holds only *together with* the retained root enclosures.
    """
    module = _tool()
    row = _synthetic_frame_row()
    for entry in row["horizon_scan_crossing_rows"]:
        if entry["classification"] == "guard_interval":
            # Still a legal guard width -- only its anchor is gone.
            entry["turn_lo"] = "3/4"
            entry["turn_hi"] = "7500000000000001/10000000000000000"
            break
    with pytest.raises(module.EvidenceError, match="orphan guard"):
        module.validate_frame_row(row, "frame")


def test_a_duplicate_owned_root_is_rejected() -> None:
    """Section 12.1: root-census reconstruction rejects a duplicate owned root.

    Two ``scan_crossing`` rows of one direction whose exact enclosures coincide
    are one root claimed twice.  Every per-row predicate still passes -- the
    probe signs differ, the root bounds equal the enclosure, the summary count
    is consistent -- so only the reconstruction itself catches it, which is why
    the rule lives there rather than in the row schema.
    """
    module = _tool()
    row = _synthetic_frame_row()
    rows = row["horizon_scan_crossing_rows"]
    original = next(
        entry for entry in rows if entry["classification"] == "scan_crossing"
    )
    duplicate = dict(original)
    duplicate["cell_index"] = int(original["cell_index"]) + 2
    rows.insert(rows.index(original) + 1, duplicate)
    # Keep every count the duplicate would otherwise contradict consistent, so
    # the refusal can only come from the census reconstruction.
    for entry in row["horizon_scan_summary_rows"]:
        if entry["direction_id"] == original["direction_id"]:
            entry["crossing_count"] = 2
            entry["terminal_cell_count"] = 7
    row["horizon_isolation_interval_count"] = 13
    with pytest.raises(module.EvidenceError, match="duplicate owned root"):
        module.validate_frame_row(row, "frame")


def test_a_guard_relocated_to_a_distant_terminal_cell_is_rejected() -> None:
    """Section 12.1: adjacency is positional as well as geometric.

    A guard that shares a bound with an enclosure but claims a terminal cell
    index far from it is not the flank of that crossing.
    """
    module = _tool()
    row = _synthetic_frame_row()
    for entry in row["horizon_scan_crossing_rows"]:
        if entry["classification"] == "guard_interval":
            entry["cell_index"] = 97
            break
    with pytest.raises(module.EvidenceError, match="orphan guard"):
        module.validate_frame_row(row, "frame")


def test_a_guard_row_carrying_a_root_is_rejected() -> None:
    """Section 12.1: the three root fields are null for a guard row."""
    module = _tool()
    row = _synthetic_frame_row()
    for entry in row["horizon_scan_crossing_rows"]:
        if entry["classification"] == "guard_interval":
            entry["root_turn_lo"] = "1/2"
            entry["root_turn_hi"] = "500000000001/1000000000000"
            entry["root_orientation"] = "rising"
            entry["root_residual_f64be"] = "0000000000000000"
            break
    with pytest.raises(module.EvidenceError):
        module.validate_frame_row(row, "frame")


def test_a_guard_wider_than_the_probe_offset_is_rejected() -> None:
    """Section 12.1: each guard's width is at most the ``1e-8`` turn probe offset."""
    module = _tool()
    row = _synthetic_frame_row()
    for entry in row["horizon_scan_crossing_rows"]:
        if entry["classification"] == "guard_interval":
            entry["turn_hi"] = "3/4"
            break
    with pytest.raises(module.EvidenceError):
        module.validate_frame_row(row, "frame")


def test_a_guard_counted_as_a_crossing_is_rejected() -> None:
    """Section 12.1: guards never enter the root census or its summary count."""
    module = _tool()
    row = _synthetic_frame_row()
    for entry in row["horizon_scan_summary_rows"]:
        entry["crossing_count"] = 2
    with pytest.raises(module.EvidenceError):
        module.validate_frame_row(row, "frame")


def test_an_unknown_scan_classification_is_rejected() -> None:
    """Section 12.1 freezes exactly four terminal-row classifications."""
    module = _tool()
    assert module.SCAN_CLASSIFICATIONS == (
        "ceiling_excludes_root",
        "scan_crossing",
        "guard_interval",
        "excluded_upper_endpoint",
    )
    row = _synthetic_frame_row()
    row["horizon_scan_crossing_rows"][0]["classification"] = "ceiling_excludes_root"
    with pytest.raises(module.EvidenceError):
        module.validate_frame_row(row, "frame")


def test_a_crossing_row_without_its_root_bounds_is_rejected() -> None:
    """Section 12.1: the three root fields are null only for a root-free row."""
    module = _tool()
    row = _synthetic_frame_row()
    row["horizon_scan_crossing_rows"][0]["root_turn_lo"] = None
    with pytest.raises(module.EvidenceError):
        module.validate_frame_row(row, "frame")


def test_a_root_free_row_smuggled_into_the_crossing_projection_is_rejected() -> None:
    """Section 12.1 retains *crossing* rows verbatim; a ceiling row is not one."""
    module = _tool()
    row = _synthetic_frame_row()
    row["horizon_scan_crossing_rows"][0]["classification"] = "ceiling_excludes_root"
    with pytest.raises(module.EvidenceError):
        module.validate_frame_row(row, "frame")


def test_a_summary_row_set_that_does_not_join_the_ledger_is_rejected() -> None:
    """Section 12.1: summary rows are in direction-ledger order, one per row."""
    module = _tool()
    row = _synthetic_frame_row()
    row["horizon_scan_summary_rows"] = list(reversed(row["horizon_scan_summary_rows"]))
    with pytest.raises(module.EvidenceError):
        module.validate_frame_row(row, "frame")


def test_a_nonzero_frame_mismatch_counter_is_rejected() -> None:
    """Section 4.2: all four mismatch counters and the unresolved count are zero."""
    module = _tool()
    row = _synthetic_frame_row()
    row["horizon_root_count_mismatches"] = 1
    with pytest.raises(module.EvidenceError):
        module.validate_frame_row(row, "frame")


def test_a_widened_fixed_frame_limit_is_rejected() -> None:
    """Section 4.2's frame limits are fixed; a per-fixture budget is not one."""
    module = _tool()
    row = _synthetic_frame_row()
    row["phase_limit_rad"] = 5e-2
    with pytest.raises(module.EvidenceError):
        module.validate_frame_row(row, "frame")


def test_a_cube_count_that_is_not_k_is_rejected() -> None:
    """Section 12: every evaluated and compared cube count equals ``K=4*N*B*F``."""
    module = _tool()
    row = _synthetic_frame_row()
    row["evaluated_frozen_gauss128_cube_cell_count"] = 1
    with pytest.raises(module.EvidenceError):
        module.validate_frame_row(row, "frame")


def test_the_per_direction_transfer_sample_ledger_is_rejected() -> None:
    """Section 7.3: the ledger is one row per grid and output cell.

    The pre-correction form was one row per catalogued *direction* and cell --
    ``552,960`` rows for this fixture. Declaring that count now fails, because
    the expected count is ``(1+len(Q_diag))*B*F*C*4``.
    """
    module = _tool()
    row = _synthetic_truncation_row()
    row["expected_transfer_sample_row_count"] = 552960
    row["evaluated_transfer_sample_row_count"] = 552960
    with pytest.raises(module.EvidenceError):
        module.validate_truncation_row(row, "truncation")


def test_a_transfer_sample_row_missing_its_concatenation_digest_is_rejected() -> None:
    """Section 7.3: the concatenation digest is the omission-detection guarantee."""
    module = _tool()
    row = _synthetic_truncation_row()
    del row["shell_coverage"]["transfer_sample_rows"][0]["concatenation_sha256"]
    with pytest.raises(module.EvidenceError):
        module.validate_truncation_row(row, "truncation")


def test_a_transfer_sample_row_count_below_its_declaration_is_rejected() -> None:
    """Section 7.3: the embedded array must equal the evaluated count."""
    module = _tool()
    row = _synthetic_truncation_row()
    row["shell_coverage"]["transfer_sample_rows"].pop()
    with pytest.raises(module.EvidenceError):
        module.validate_truncation_row(row, "truncation")


def test_the_synthetic_truncation_row_carries_the_v3_surface() -> None:
    """Section 14.2: the corrected truncation row, in the memo's exact order."""
    module = _tool()
    document = _synthetic_envelope()
    row = document["results"]["truncation_cases"][0]

    assert tuple(row) == module.TRUNCATION_ROW_KEYS
    module.validate_truncation_row(row, "truncation_cases[0]")


def test_a_tier_1a_shell_above_its_fixed_limit_is_rejected() -> None:
    """Section 7.3: tier 1a is the half with a fixed numeric limit."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["truncation_cases"][0]["horizon_free_shell_max_jy"] = 1.0
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_non_monotone_deficit_sequence_is_rejected() -> None:
    """Section 7.3: a non-converging harmonic representation licenses nothing."""
    module = _tool()
    document = _synthetic_envelope()
    row = document["results"]["truncation_cases"][0]
    row["deficit_max_half_jy"] = row["deficit_max_quarter_jy"] * 2.0
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_quarter_to_full_factor_below_two_is_rejected() -> None:
    """Section 7.3: the fixed floor is two, and it is never widened."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["truncation_cases"][0]["convergence_factor"] = 1.9
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_deficit_above_its_declared_budget_is_rejected() -> None:
    """Section 7.3: the per-fixture budget is reviewed evidence, and it binds."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["truncation_cases"][0]["truncation_budget_jy"] = 1e-3
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_with_horizon_shell_above_its_declared_budget_is_rejected() -> None:
    """Section 7.3: tier 1b carries no universal limit, but its budget binds."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["truncation_cases"][0]["quadrature_budget_jy"] = 1e-3
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_failed_wp7_dependency_certificate_is_rejected() -> None:
    """Section 14.2: the replay sets no independent pass flag that could disagree."""
    module = _tool()
    document = _synthetic_envelope()
    document["results"]["dependency_certificate"]["certificate"]["passed"] = False
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_a_non_zero_command_exit_code_is_rejected() -> None:
    """Section 14.2: evidence commands require exit code zero."""
    module = _tool()
    document = _synthetic_envelope()
    document["commands"][0]["exit_code"] = 1
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


def test_unsorted_claim_arrays_are_rejected() -> None:
    """``limitations`` and ``claims_not_licensed`` are sorted and unique."""
    module = _tool()
    document = _synthetic_envelope()
    document["claims_not_licensed"] = ["gpu_or_accelerator_support", "general_speedup"]
    with pytest.raises(module.EvidenceError):
        module.validate_evidence_document(document)


# ---------------------------------------------------------------------------
# Canonical JSON and digest rules
# ---------------------------------------------------------------------------


def test_canonical_json_sorts_keys_and_emits_no_whitespace() -> None:
    """Section 14: sorted keys, tight separators, ASCII, no trailing newline."""
    assert _canonical({"b": 1, "a": [1, 2]}) == b'{"a":[1,2],"b":1}'


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, b"1"),
        (0.0, b"0"),
        (-0.5, b"-0.5"),
        (1e21, b"1e+21"),
        (1e-7, b"1e-7"),
        (1.5e300, b"1.5e+300"),
    ],
)
def test_canonical_numbers_use_the_ecmascript_spelling(
    value: float, expected: bytes
) -> None:
    """``1.0`` and ``1e0`` are not canonical record bytes for the integer one."""
    assert _canonical(value) == expected


def test_canonical_json_forbids_nan_and_infinity() -> None:
    """Section 14: NaN and Infinity are forbidden."""
    module = _tool()
    with pytest.raises((module.EvidenceError, ValueError)):
        _canonical(float("nan"))


def test_the_domain_digest_matches_its_printed_definition() -> None:
    """Section 14.0: ``D(d, p) = SHA256(d || NUL || U64(len(p)) || p)``."""
    module = _tool()
    payload = b"payload"
    expected = hashlib.sha256(
        b"radiosim.example.v1" + b"\x00" + len(payload).to_bytes(8, "big") + payload
    ).hexdigest()
    assert module.domain_digest("radiosim.example.v1", payload) == expected


def test_a_distinct_domain_gives_a_distinct_digest() -> None:
    """The domain is part of the preimage, so two roles never collide."""
    module = _tool()
    first = module.domain_digest("radiosim.a.v1", b"x")
    second = module.domain_digest("radiosim.b.v1", b"x")
    assert first != second


# ---------------------------------------------------------------------------
# E1 state: authenticate the retained artifact
# ---------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    """Return the stdout of one hermetic ``git`` invocation in this repository.

    The validator carries no package dependency, so ancestry facts are read from
    ``git`` itself rather than from a library, exactly as the dirty-tree probe
    above runs the generator itself rather than trusting a description of it.
    """
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

    Section 14.2 requires ``E1`` to be *located*, not assumed: the artifact is
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
    """Section 14.2's ``E1`` ancestry clause, skipped until the constants flip.

    ``E1`` is located from history rather than named, and its **direct** parent
    must be the approved ``S1``.  A merge commit is refused outright: an
    artifact introduced on a merge has no single source tree it was generated
    from, which is the whole point of binding the two.
    """
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_SOURCE_SHA is None:
        pytest.skip("the M1 evidence artifact is authorized at E1")
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
    # The located commit is the one the approved digest authenticates.
    payload = _git("show", f"{located}:{ARTIFACT}")
    assert (
        hashlib.sha256(payload.encode("utf-8")).hexdigest() == APPROVED_ARTIFACT_SHA256
    )


def test_the_e1_diff_writes_only_the_section_13_3_authorized_paths() -> None:
    """Section 13.3/14.2: ``E1`` adds the artifact and its record, nothing else."""
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_SOURCE_SHA is None:
        pytest.skip("the M1 evidence artifact is authorized at E1")
    located = _locate_evidence_commit()
    changed = set(
        _git("diff-tree", "--no-commit-id", "--name-only", "-r", located).split()
    )
    assert ARTIFACT in changed
    unauthorized = sorted(changed - E1_AUTHORIZED_PATHS)
    assert not unauthorized, (
        f"the E1 commit {located} writes {unauthorized}, which Section 13.3 "
        f"does not authorize; it may write only {sorted(E1_AUTHORIZED_PATHS)}"
    )
    if REPRODUCTION in changed:
        record = _git("show", f"{located}:{REPRODUCTION}")
        assert record.startswith(REPRODUCTION_FRONT_MATTER), (
            "the reproduction record must open with Section 14.2's exact MyST "
            "front matter"
        )


def test_the_e1_diff_changes_only_the_two_approved_constant_assignments() -> None:
    """Section 14.2: this module's own ``E1`` diff is the two constants alone.

    The comparison is a token stream taken **outside** the two assignment spans,
    which is what makes it survive the formatter wrapping the 64-hex digest in
    parentheses while still refusing any other edit -- an added import, a
    reworded docstring, a relaxed assertion, a deleted test.  Inside the spans
    only the value may move, from the ``None`` sentinel to the approved literal.
    """
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_SOURCE_SHA is None:
        pytest.skip("the M1 evidence artifact is authorized at E1")
    located = _locate_evidence_commit()
    parent = _git("rev-list", "--parents", "-n", "1", located).split()[1]
    before = _git("show", f"{parent}:{VALIDATOR}")
    after = _git("show", f"{located}:{VALIDATOR}")

    assert _outside_spans(before) == _outside_spans(after), (
        f"the E1 commit {located} changed this module outside the two approved "
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
    """Section 14.2's ``E1`` state, skipped until the constants are flipped."""
    if APPROVED_ARTIFACT_SHA256 is None or APPROVED_SOURCE_SHA is None:
        pytest.skip("the M1 evidence artifact is authorized at E1")
    path = REPOSITORY_ROOT / ARTIFACT
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == APPROVED_ARTIFACT_SHA256
    document = json.loads(payload.decode("utf-8"))
    module = _tool()
    module.validate_evidence_document(document)
    assert document["source_sha"] == APPROVED_SOURCE_SHA
    assert module.canonical_json(document) == payload
