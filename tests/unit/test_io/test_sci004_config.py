"""Strict parse and typed rejection for the SCI-004 phase-M1 configuration surface.

``docs/development/sci004_mmode_design.md`` Section 8 defines one new
``execution.mmode`` block, one new strict ``obs_time`` variant
(``mode: full_sidereal``, Section 3.2), and a frozen table of semantic issue
codes with exact messages. This module is the red slice for that surface: none
of it exists at ``G1``, so every rejection node below observes the *absence* of
the surface rather than the rejection the design specifies.

**Error taxonomy (Section 8, last paragraph).** Whether a value can be *read* as
the declared kind of thing is ``ConfigSchemaError``; the cross-field and domain
failures in the Section 8 table are ``ConfigSemanticError``; an accepted schema
combined with an unsupported physical payload is ``UnsupportedConfigError``. All
three live at ``radiosim.io.config_resolution``. Failure occurs before backend
allocation, output-path creation, or harmonic work.

**Fixture bytes.** Every case in :data:`SCI004_RED_CASES` carries the exact
UTF-8 YAML bytes of its own fixture *override document*, and
``tools/sci004_mmode_phase1_red.py`` hashes those bytes raw into the Section
14.0 ``invalid_config_raw_sha256`` field. The override is deep-merged into the
shared valid mapping, whose only non-deterministic content is the ``tmp_path``
the harness supplies; keeping the override separate is what makes the retained
digest reproducible. A key whose name begins with ``_`` is a fixture-local
materialization directive consumed by the test, never configuration.

Every rejection asserts the concrete exception type together with the issue path
and code and the exact Section 8 message, never a message substring alone.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

# --- Section 8's frozen issue codes and exact messages ------------------------

#: Section 8's complete table. Only the M1-reachable subset is exercised here;
#: ``mmode_polarization_frame`` is an M2 payload rule and
#: ``mmode_horizon_unresolved`` is raised by the Section 12.1 certifier, not by
#: configuration resolution.
MMODE_MESSAGES: dict[str, str] = {
    "mmode_block_required": (
        "execution.simulator='mmode' requires an explicit execution.mmode block."
    ),
    "mmode_block_forbidden": (
        "execution.mmode is only valid when execution.simulator='mmode'."
    ),
    "mmode_time_grid_required": (
        "execution.simulator='mmode' requires obs_time.mode='full_sidereal'; a "
        "UTC-uniform interval is not an m-mode grid."
    ),
    "rime_time_grid_required": (
        "obs_time.mode='full_sidereal' is only valid when execution.simulator='mmode'."
    ),
    "mmode_exposure_resolution": (
        "obs_time.integration_fraction is too small for distinct canonical "
        "binary64 exposure edges at this sidereal_samples."
    ),
    "mmode_nyquist": (
        "obs_time.sidereal_samples must be at least 2 * execution.mmode.mmax + 1."
    ),
    "mmode_tail_nyquist": (
        "obs_time.sidereal_samples must be at least 2 * resolved mcheck + 1 for "
        "the mandatory m-tail diagnostic."
    ),
    "mmode_quadrature": (
        "execution.mmode.lmax must be at most 2 * execution.mmode.quadrature_nside."
    ),
    "mmode_time_smearing": (
        "execution.simulator='mmode' owns ERA top-hat integration; "
        "jones.Q.time_smearing must be false."
    ),
    "mmode_static_gain": (
        "execution.simulator='mmode' requires jones.G.time_model.kind='constant'."
    ),
    "mmode_phase_center": (
        "execution.simulator='mmode' requires the canonical fixed zenith-drift "
        "phase centre."
    ),
    "mmode_point_morphology": (
        "execution.simulator='mmode' does not yet support Gaussian point-source "
        "morphology; use rime or remove the morphology."
    ),
    "mmode_polarization_frame": (
        "polarized m-mode input requires an explicit canonical "
        "tangent-polarization frame."
    ),
    "mmode_iers_range": (
        "the full-sidereal UTC mapping is outside the available offline IERS table."
    ),
    "mmode_truncation_check": (
        "execution.mmode.lmax leaves no room for the required harmonic tail check."
    ),
    "mmode_horizon_unresolved": (
        "execution.simulator='mmode' could not certify complete horizon-root "
        "isolation; tangent, identically-zero, and unresolved intervals are "
        "rejected."
    ),
    "mmode_m1_scalar_only": (
        "MModeSimulator phase M1 accepts Stokes I only; non-zero Q, U, or V "
        "requires accepted phase M2."
    ),
}

#: Section 8's three required exact convention literals.
MMODE_CONVENTION = "radiosim.mmode-forward.v1"
MMODE_FRAME_MODEL = "radiosim.frozen-cirs-rigid-era.v1"
MMODE_HARMONIC_CONVENTION = "radiosim.shaw-polarized-harmonics.v1"


# --- fixture override documents ----------------------------------------------

_ACCEPTED_MMODE_EXECUTION = f"""\
execution:
  simulator: mmode
  mmode:
    convention: {MMODE_CONVENTION}
    frame_model: {MMODE_FRAME_MODEL}
    harmonic_convention: {MMODE_HARMONIC_CONVENTION}
    lmax: 64
    mmax: 64
    quadrature_nside: 64
    working_memory_bytes: 1073741824
  solver:
    workers: 1
    executor: thread
"""

_FULL_SIDEREAL_TIME = """\
obs_time:
  mode: full_sidereal
  start_time: "2025-01-01T00:00:00"
  sidereal_samples: 257
  integration_fraction: 1.0
"""


def _mmode_document(
    *,
    lmax: int = 64,
    mmax: int = 64,
    quadrature_nside: int = 64,
    sidereal_samples: int = 257,
    integration_fraction: str = "1.0",
    start_time: str = "2025-01-01T00:00:00",
    extra: str = "",
) -> bytes:
    """Return the exact bytes of one complete accepted-shape override document."""
    document = (
        _ACCEPTED_MMODE_EXECUTION.replace("lmax: 64", f"lmax: {lmax}")
        .replace("mmax: 64", f"mmax: {mmax}")
        .replace("quadrature_nside: 64", f"quadrature_nside: {quadrature_nside}")
        + _FULL_SIDEREAL_TIME.replace(
            "sidereal_samples: 257", f"sidereal_samples: {sidereal_samples}"
        )
        .replace(
            "integration_fraction: 1.0", f"integration_fraction: {integration_fraction}"
        )
        .replace('"2025-01-01T00:00:00"', f'"{start_time}"')
        + extra
    )
    return document.encode("utf-8")


FIXTURES: dict[str, bytes] = {
    "m1.config.accepted-shape": _mmode_document(),
    "m1.config.full-sidereal-time-variant": _mmode_document(integration_fraction="0.5"),
    "m1.config.mmode_block_required": (
        "execution:\n  simulator: mmode\n" + _FULL_SIDEREAL_TIME
    ).encode("utf-8"),
    "m1.config.mmode_block_forbidden": (
        _ACCEPTED_MMODE_EXECUTION.replace("simulator: mmode", "simulator: rime")
    ).encode("utf-8"),
    "m1.config.mmode_time_grid_required": _ACCEPTED_MMODE_EXECUTION.encode("utf-8"),
    "m1.config.rime_time_grid_required": _FULL_SIDEREAL_TIME.encode("utf-8"),
    "m1.config.mmode_exposure_resolution": _mmode_document(
        integration_fraction="1.0e-308"
    ),
    "m1.config.mmode_nyquist": _mmode_document(sidereal_samples=17),
    "m1.config.mmode_tail_nyquist": _mmode_document(
        lmax=8, mmax=8, quadrature_nside=8, sidereal_samples=17
    ),
    "m1.config.mmode_quadrature": _mmode_document(quadrature_nside=16),
    "m1.config.mmode_truncation_check": _mmode_document(
        lmax=4096, mmax=4096, quadrature_nside=2048, sidereal_samples=8193
    ),
    "m1.config.mmode_iers_range": _mmode_document(start_time="2200-01-01T00:00:00"),
    "m1.config.mmode_time_smearing": _mmode_document(
        extra="jones:\n  Q:\n    bandwidth_smearing: true\n    time_smearing: true\n"
    ),
    "m1.config.mmode_static_gain": _mmode_document(
        extra=(
            "jones:\n"
            "  G:\n"
            "    amplitude_error: 0.05\n"
            "    time_model:\n"
            "      kind: linear_drift\n"
            "      rate_per_hour: 0.01\n"
        )
    ),
    "m1.config.mmode_phase_center": _mmode_document(
        extra=(
            "beams:\n"
            "  mode: analytic\n"
            "  model:\n"
            "    kind: circular_aperture\n"
            "    taper:\n"
            "      kind: gaussian\n"
            "      edge_taper_db: 10.0\n"
            "  pointing:\n"
            "    default:\n"
            "      azimuth_offset_deg: 0.0\n"
            "      elevation_offset_deg: 1.5\n"
        )
    ),
    "m1.config.pre-allocation": _mmode_document(quadrature_nside=16),
}


def _fixture_pattern(case_id: str) -> str:
    """The absent-surface failure every mmode document produces at ``G1``."""
    del case_id
    return r"ConfigSchemaError: execution\.mmode: unknown or removed field"


#: Section 14.1's per-case declaration, consumed by the phase red generator and
#: independently re-checked by ``tests/unit/test_sci004_phase1_red_failures.py``.
SCI004_RED_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "m1.config.accepted-shape",
        "requirement_id": "sci004.section-8.accepted-execution-mmode-block",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_accepted_mmode_execution_block_resolves"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": _fixture_pattern("accepted"),
        "fixture_defect_excluded_by": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_unmodified_base_fixture_still_resolves_under_rime"
        ),
        "fixture_bytes": FIXTURES["m1.config.accepted-shape"],
    },
    {
        "case_id": "m1.config.full-sidereal-time-variant",
        "requirement_id": "sci004.section-3.2.typed-full-sidereal-time-input",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_full_sidereal_time_variant_is_strict_and_complete"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": _fixture_pattern("time"),
        "fixture_defect_excluded_by": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_unmodified_base_fixture_still_resolves_under_rime"
        ),
        "fixture_bytes": FIXTURES["m1.config.full-sidereal-time-variant"],
    },
    {
        "case_id": "m1.config.mmode_block_required",
        "requirement_id": "sci004.section-8.mmode_block_required",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_mmode_requires_an_explicit_execution_mmode_block"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": (
            r"ConfigSchemaError: execution\.simulator: Input should be 'rime'"
        ),
        "fixture_defect_excluded_by": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_unmodified_base_fixture_still_resolves_under_rime"
        ),
        "fixture_bytes": FIXTURES["m1.config.mmode_block_required"],
    },
    {
        "case_id": "m1.config.mmode_block_forbidden",
        "requirement_id": "sci004.section-8.mmode_block_forbidden",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_an_mmode_block_is_forbidden_under_the_direct_simulator"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": _fixture_pattern("forbidden"),
        "fixture_defect_excluded_by": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_unmodified_base_fixture_still_resolves_under_rime"
        ),
        "fixture_bytes": FIXTURES["m1.config.mmode_block_forbidden"],
    },
    {
        "case_id": "m1.config.mmode_time_grid_required",
        "requirement_id": "sci004.section-8.mmode_time_grid_required",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_mmode_rejects_the_utc_uniform_interval"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": _fixture_pattern("time-grid"),
        "fixture_defect_excluded_by": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_unmodified_base_fixture_still_resolves_under_rime"
        ),
        "fixture_bytes": FIXTURES["m1.config.mmode_time_grid_required"],
    },
    {
        "case_id": "m1.config.rime_time_grid_required",
        "requirement_id": "sci004.section-8.rime_time_grid_required",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_full_sidereal_grid_is_rejected_under_the_direct_simulator"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": (
            r"ConfigSchemaError: obs_time\.duration_seconds: Field required"
        ),
        "fixture_defect_excluded_by": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_unmodified_base_fixture_still_resolves_under_rime"
        ),
        "fixture_bytes": FIXTURES["m1.config.rime_time_grid_required"],
    },
    {
        "case_id": "m1.config.mmode_exposure_resolution",
        "requirement_id": "sci004.section-8.mmode_exposure_resolution",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_a_collapsed_binary64_exposure_edge_is_rejected"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": _fixture_pattern("exposure"),
        "fixture_defect_excluded_by": (
            "tests/unit/test_core/test_sci004_era_grid.py::"
            "test_the_fixture_fraction_reconstructs_its_exact_ieee_ratio"
        ),
        "fixture_bytes": FIXTURES["m1.config.mmode_exposure_resolution"],
    },
    {
        "case_id": "m1.config.mmode_nyquist",
        "requirement_id": "sci004.section-8.mmode_nyquist",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_sidereal_samples_below_the_retained_mode_nyquist_bound"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": _fixture_pattern("nyquist"),
        "fixture_defect_excluded_by": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_unmodified_base_fixture_still_resolves_under_rime"
        ),
        "fixture_bytes": FIXTURES["m1.config.mmode_nyquist"],
    },
    {
        "case_id": "m1.config.mmode_tail_nyquist",
        "requirement_id": "sci004.section-8.mmode_tail_nyquist",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_sidereal_samples_below_the_m_tail_nyquist_bound"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": _fixture_pattern("tail"),
        "fixture_defect_excluded_by": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_derived_tail_dimensions_follow_the_frozen_formulas"
        ),
        "fixture_bytes": FIXTURES["m1.config.mmode_tail_nyquist"],
    },
    {
        "case_id": "m1.config.mmode_quadrature",
        "requirement_id": "sci004.section-8.mmode_quadrature",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_lmax_above_twice_the_quadrature_nside_is_rejected"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": _fixture_pattern("quadrature"),
        "fixture_defect_excluded_by": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_derived_tail_dimensions_follow_the_frozen_formulas"
        ),
        "fixture_bytes": FIXTURES["m1.config.mmode_quadrature"],
    },
    {
        "case_id": "m1.config.mmode_truncation_check",
        "requirement_id": "sci004.section-8.mmode_truncation_check",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_an_lmax_leaving_no_harmonic_tail_room_is_rejected"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": _fixture_pattern("truncation"),
        "fixture_defect_excluded_by": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_derived_tail_dimensions_follow_the_frozen_formulas"
        ),
        "fixture_bytes": FIXTURES["m1.config.mmode_truncation_check"],
    },
    {
        "case_id": "m1.config.mmode_iers_range",
        "requirement_id": "sci004.section-8.mmode_iers_range",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_a_full_sidereal_grid_outside_the_offline_iers_table_is_rejected"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": _fixture_pattern("iers"),
        "fixture_defect_excluded_by": (
            "tests/unit/test_core/test_sci004_frame.py::"
            "test_the_bundled_iers_resource_resolves_and_hashes_today"
        ),
        "fixture_bytes": FIXTURES["m1.config.mmode_iers_range"],
    },
    {
        "case_id": "m1.config.mmode_time_smearing",
        "requirement_id": "sci004.section-8.mmode_time_smearing",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_q_time_smearing_is_rejected_because_the_top_hat_owns_it"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": _fixture_pattern("smearing"),
        "fixture_defect_excluded_by": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_same_jones_blocks_resolve_today_under_the_direct_simulator"
        ),
        "fixture_bytes": FIXTURES["m1.config.mmode_time_smearing"],
    },
    {
        "case_id": "m1.config.mmode_static_gain",
        "requirement_id": "sci004.section-8.mmode_static_gain",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_a_nonconstant_gain_time_model_is_rejected"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": _fixture_pattern("gain"),
        "fixture_defect_excluded_by": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_same_jones_blocks_resolve_today_under_the_direct_simulator"
        ),
        "fixture_bytes": FIXTURES["m1.config.mmode_static_gain"],
    },
    {
        "case_id": "m1.config.mmode_phase_center",
        "requirement_id": "sci004.section-8.mmode_phase_center",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_a_boresight_displaced_from_the_zenith_is_rejected"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": _fixture_pattern("phase-centre"),
        "fixture_defect_excluded_by": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_same_pointing_block_resolves_today_under_the_direct_simulator"
        ),
        "fixture_bytes": FIXTURES["m1.config.mmode_phase_center"],
    },
    {
        "case_id": "m1.config.pre-allocation",
        "requirement_id": "sci004.section-8.rejection-precedes-allocation",
        "test_nodeid": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_a_typed_rejection_creates_no_backend_and_no_output_path"
        ),
        "expected_failure_kind": "schema",
        "expected_failure_pattern": _fixture_pattern("pre-allocation"),
        "fixture_defect_excluded_by": (
            "tests/unit/test_io/test_sci004_config.py::"
            "test_the_unmodified_base_fixture_still_resolves_under_rime"
        ),
        "fixture_bytes": FIXTURES["m1.config.pre-allocation"],
    },
)

#: Nodes that must be executed alongside the red set and must pass. They are the
#: fixture-defect exclusions named above, so a green control failing invalidates
#: the record rather than being recorded as a red case.
SCI004_RED_GREEN_CONTROLS: tuple[str, ...] = (
    "tests/unit/test_io/test_sci004_config.py::"
    "test_the_unmodified_base_fixture_still_resolves_under_rime",
    "tests/unit/test_io/test_sci004_config.py::"
    "test_the_same_jones_blocks_resolve_today_under_the_direct_simulator",
    "tests/unit/test_io/test_sci004_config.py::"
    "test_the_same_pointing_block_resolves_today_under_the_direct_simulator",
    "tests/unit/test_io/test_sci004_config.py::"
    "test_the_derived_tail_dimensions_follow_the_frozen_formulas",
)


# --- harness ------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
            and key != "beams"
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _document(tmp_path: Path, case_id: str) -> dict[str, Any]:
    """Deep-merge one exact fixture override into the shared valid mapping."""
    import yaml

    from tests.fixtures.configs import valid_config_mapping

    override = yaml.safe_load(FIXTURES[case_id].decode("utf-8"))
    assert isinstance(override, dict)
    override = {key: value for key, value in override.items() if key[0] != "_"}
    if "obs_time" in override:
        # The full-sidereal variant is a complete alternative, not a fragment.
        base = valid_config_mapping(tmp_path)
        base["obs_time"] = {}
        return _deep_merge(base, override)
    return _deep_merge(valid_config_mapping(tmp_path), override)


def _resolve(tmp_path: Path, case_id: str) -> Any:
    from radiosim.io.config_resolution import ConfigurationSource, resolve_config

    return resolve_config(
        _document(tmp_path, case_id),
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )


def _assert_issue(error: Any, code: str) -> None:
    """Assert the concrete code *and* the exact Section 8 message."""
    codes = [issue.code for issue in error.issues]
    assert code in codes, codes
    issue = next(issue for issue in error.issues if issue.code == code)
    assert issue.message == MMODE_MESSAGES[code]


def _derived_dimensions(lmax: int, mmax: int, quadrature_nside: int) -> dict[str, int]:
    """Section 7.3's frozen derivations, evaluated independently in the test body."""
    lcheck = min(lmax + max(8, lmax // 8), 4096)
    mcheck = min(lcheck, mmax + max(8, max(1, mmax // 8)))
    target = max(2 * quadrature_nside, -(-lcheck // 2))
    qcheck = 1
    while qcheck < target:
        qcheck *= 2
    return {"lcheck": lcheck, "mcheck": mcheck, "qcheck": qcheck}


# --- green controls -----------------------------------------------------------


def test_the_unmodified_base_fixture_still_resolves_under_rime(tmp_path) -> None:
    """The harness itself is sound: the base mapping resolves at ``G1``."""
    from radiosim.io.config_resolution import ConfigurationSource, resolve_config
    from tests.fixtures.configs import valid_config_mapping

    bundle = resolve_config(
        valid_config_mapping(tmp_path),
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )

    assert bundle.runtime.execution.simulator == "rime"


def test_the_same_jones_blocks_resolve_today_under_the_direct_simulator(
    tmp_path,
) -> None:
    """The ``Q`` and ``G`` blocks the mmode rules reject are legal ``rime`` input.

    That is the whole point of ``mmode_time_smearing`` and ``mmode_static_gain``:
    they are not schema errors, they are m-mode stationarity rules over
    configuration the direct solver accepts.
    """
    from radiosim.io.config_resolution import ConfigurationSource, resolve_config
    from tests.fixtures.configs import valid_config_mapping

    document = valid_config_mapping(tmp_path)
    document["jones"] = {
        "Q": {"bandwidth_smearing": True, "time_smearing": True},
        "G": {
            "amplitude_error": 0.05,
            "time_model": {"kind": "linear_drift", "rate_per_hour": 0.01},
        },
    }
    bundle = resolve_config(
        document,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )

    assert set(bundle.runtime.jones.configured_terms) >= {"G", "Q"}


def test_the_same_pointing_block_resolves_today_under_the_direct_simulator(
    tmp_path,
) -> None:
    """A displaced boresight is legal ``rime`` input, so the fixture is sound."""
    from radiosim.io.config_resolution import ConfigurationSource, resolve_config
    from tests.fixtures.configs import valid_config_mapping

    document = valid_config_mapping(tmp_path)
    document["beams"] = {
        "mode": "analytic",
        "model": {
            "kind": "circular_aperture",
            "taper": {"kind": "gaussian", "edge_taper_db": 10.0},
        },
        "pointing": {
            "default": {"azimuth_offset_deg": 0.0, "elevation_offset_deg": 1.5}
        },
    }
    bundle = resolve_config(
        document,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )

    assert bundle.runtime is not None


def test_the_derived_tail_dimensions_follow_the_frozen_formulas() -> None:
    """Section 7.3's derivations, checked against the fixture dimensions.

    This is the analytic oracle the Nyquist, quadrature and truncation cases
    cite: it proves each fixture really does violate the bound it names, so the
    red failure cannot be an arithmetic slip in the fixture.
    """
    tail = _derived_dimensions(8, 8, 8)
    assert tail == {"lcheck": 16, "mcheck": 16, "qcheck": 16}
    # The ``mmode_tail_nyquist`` fixture: 17 clears 2*mmax+1 but not 2*mcheck+1.
    assert 17 >= 2 * 8 + 1
    assert 17 < 2 * tail["mcheck"] + 1

    # The ``mmode_nyquist`` fixture: 17 samples cannot carry mmax = 64.
    assert 17 < 2 * 64 + 1
    # The ``mmode_quadrature`` fixture: lmax 64 exceeds 2 * nside 16.
    assert 64 > 2 * 16
    # The ``mmode_truncation_check`` fixture: 4096 is above the 4088 ceiling.
    assert 4096 > 4088


# --- Section 8 rejection surface (red at G1) ----------------------------------


def test_the_accepted_mmode_execution_block_resolves(tmp_path) -> None:
    """Section 8's accepted shape, with its three required exact literals."""
    bundle = _resolve(tmp_path, "m1.config.accepted-shape")
    mmode = bundle.runtime.execution.mmode

    assert bundle.runtime.execution.simulator == "mmode"
    assert mmode.convention == MMODE_CONVENTION
    assert mmode.frame_model == MMODE_FRAME_MODEL
    assert mmode.harmonic_convention == MMODE_HARMONIC_CONVENTION
    assert (mmode.lmax, mmode.mmax, mmode.quadrature_nside) == (64, 64, 64)
    assert mmode.working_memory_bytes == 1073741824


def test_the_full_sidereal_time_variant_is_strict_and_complete(tmp_path) -> None:
    """Section 3.2: strict positive ``sidereal_samples`` and finite fraction."""
    bundle = _resolve(tmp_path, "m1.config.full-sidereal-time-variant")
    obs_time = bundle.runtime.obs_time

    assert obs_time.mode == "full_sidereal"
    assert obs_time.sidereal_samples == 257
    assert obs_time.integration_fraction == 0.5
    for removed in ("duration_seconds", "time_step_seconds"):
        assert not hasattr(obs_time, removed)


def test_mmode_requires_an_explicit_execution_mmode_block(tmp_path) -> None:
    """Section 8: ``mmode_block_required``."""
    from radiosim.io.config_resolution import ConfigSemanticError

    with pytest.raises(ConfigSemanticError) as excinfo:
        _resolve(tmp_path, "m1.config.mmode_block_required")

    _assert_issue(excinfo.value, "mmode_block_required")


def test_an_mmode_block_is_forbidden_under_the_direct_simulator(tmp_path) -> None:
    """Section 8: ``mmode_block_forbidden``; an absent block never changes rime."""
    from radiosim.io.config_resolution import ConfigSemanticError

    with pytest.raises(ConfigSemanticError) as excinfo:
        _resolve(tmp_path, "m1.config.mmode_block_forbidden")

    _assert_issue(excinfo.value, "mmode_block_forbidden")


def test_mmode_rejects_the_utc_uniform_interval(tmp_path) -> None:
    """Section 8: ``mmode_time_grid_required``."""
    from radiosim.io.config_resolution import ConfigSemanticError

    with pytest.raises(ConfigSemanticError) as excinfo:
        _resolve(tmp_path, "m1.config.mmode_time_grid_required")

    _assert_issue(excinfo.value, "mmode_time_grid_required")


def test_the_full_sidereal_grid_is_rejected_under_the_direct_simulator(
    tmp_path,
) -> None:
    """Section 8: ``rime_time_grid_required``."""
    from radiosim.io.config_resolution import ConfigSemanticError

    with pytest.raises(ConfigSemanticError) as excinfo:
        _resolve(tmp_path, "m1.config.rime_time_grid_required")

    _assert_issue(excinfo.value, "rime_time_grid_required")


def test_a_collapsed_binary64_exposure_edge_is_rejected(tmp_path) -> None:
    """Section 3.1/8: ``mmode_exposure_resolution``.

    The fraction is finite and inside ``(0, 1]``, so this is a domain failure of
    the *derived* binary64 edges, not a schema failure of the authored value.
    """
    from radiosim.io.config_resolution import ConfigSemanticError

    with pytest.raises(ConfigSemanticError) as excinfo:
        _resolve(tmp_path, "m1.config.mmode_exposure_resolution")

    _assert_issue(excinfo.value, "mmode_exposure_resolution")


def test_sidereal_samples_below_the_retained_mode_nyquist_bound(tmp_path) -> None:
    """Section 6/8: ``mmode_nyquist``; ``N >= 2*mmax+1`` is mandatory."""
    from radiosim.io.config_resolution import ConfigSemanticError

    with pytest.raises(ConfigSemanticError) as excinfo:
        _resolve(tmp_path, "m1.config.mmode_nyquist")

    _assert_issue(excinfo.value, "mmode_nyquist")


def test_sidereal_samples_below_the_m_tail_nyquist_bound(tmp_path) -> None:
    """Section 7.3/8: ``mmode_tail_nyquist``; the tail diagnostic needs mcheck."""
    from radiosim.io.config_resolution import ConfigSemanticError

    with pytest.raises(ConfigSemanticError) as excinfo:
        _resolve(tmp_path, "m1.config.mmode_tail_nyquist")

    _assert_issue(excinfo.value, "mmode_tail_nyquist")


def test_lmax_above_twice_the_quadrature_nside_is_rejected(tmp_path) -> None:
    """Section 7.3/8: ``mmode_quadrature``."""
    from radiosim.io.config_resolution import ConfigSemanticError

    with pytest.raises(ConfigSemanticError) as excinfo:
        _resolve(tmp_path, "m1.config.mmode_quadrature")

    _assert_issue(excinfo.value, "mmode_quadrature")


def test_an_lmax_leaving_no_harmonic_tail_room_is_rejected(tmp_path) -> None:
    """Section 7.3/8: ``mmode_truncation_check``; 4096 is a rejection, not a claim."""
    from radiosim.io.config_resolution import ConfigSemanticError

    with pytest.raises(ConfigSemanticError) as excinfo:
        _resolve(tmp_path, "m1.config.mmode_truncation_check")

    _assert_issue(excinfo.value, "mmode_truncation_check")


def test_a_full_sidereal_grid_outside_the_offline_iers_table_is_rejected(
    tmp_path,
) -> None:
    """Section 3.1/8: ``mmode_iers_range``; no network lookup is ever permitted."""
    from radiosim.io.config_resolution import ConfigSemanticError

    with pytest.raises(ConfigSemanticError) as excinfo:
        _resolve(tmp_path, "m1.config.mmode_iers_range")

    _assert_issue(excinfo.value, "mmode_iers_range")


def test_q_time_smearing_is_rejected_because_the_top_hat_owns_it(tmp_path) -> None:
    """Section 6/8: ``mmode_time_smearing``; exposure averaging is not applied twice."""
    from radiosim.io.config_resolution import ConfigSemanticError

    with pytest.raises(ConfigSemanticError) as excinfo:
        _resolve(tmp_path, "m1.config.mmode_time_smearing")

    _assert_issue(excinfo.value, "mmode_time_smearing")


def test_a_nonconstant_gain_time_model_is_rejected(tmp_path) -> None:
    """Section 8: ``mmode_static_gain``; every accepted term is ground-stationary."""
    from radiosim.io.config_resolution import ConfigSemanticError

    with pytest.raises(ConfigSemanticError) as excinfo:
        _resolve(tmp_path, "m1.config.mmode_static_gain")

    _assert_issue(excinfo.value, "mmode_static_gain")


def test_a_boresight_displaced_from_the_zenith_is_rejected(tmp_path) -> None:
    """Section 1/8: ``mmode_phase_center``; the driver fixes the zenith boresight."""
    from radiosim.io.config_resolution import ConfigSemanticError

    with pytest.raises(ConfigSemanticError) as excinfo:
        _resolve(tmp_path, "m1.config.mmode_phase_center")

    _assert_issue(excinfo.value, "mmode_phase_center")


def test_a_typed_rejection_creates_no_backend_and_no_output_path(tmp_path) -> None:
    """Section 8: failure precedes backend allocation and output-path creation."""
    from radiosim.io.config_resolution import ConfigSemanticError

    output_dir = tmp_path / "output"
    assert not output_dir.exists()

    with pytest.raises(ConfigSemanticError) as excinfo:
        _resolve(tmp_path, "m1.config.pre-allocation")

    _assert_issue(excinfo.value, "mmode_quadrature")
    assert not output_dir.exists()
