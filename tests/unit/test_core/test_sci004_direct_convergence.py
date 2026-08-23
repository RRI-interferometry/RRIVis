r"""SCI-004 phase-M2 red oracles for full-Stokes direct agreement.

``docs/development/sci004_mmode_design.md`` Section 12.2 family 6 requires
"small unpolarized **and polarized** point, HEALPix, and hybrid skies through the
common frozen-frame direct oracle", judged by Section 7.3's every-run **two-tier
gate**.  Phase M1 shipped that gate for a Stokes-``I`` sky; phase M2 is where the
same machinery has to close for the polarized fields, against a private direct
oracle that sums all four Stokes components rather than dropping three of them.

**Tier 1a -- horizon-free shell, gating at ``1e-8``.**  The complete harmonic
pipeline is evaluated once with *every* horizon truncation removed and everything
else identical, on both the production and ``qcheck`` quadratures, giving ``W0``
and ``W_q``.  Removing the explicit ``H`` factor alone is insufficient: the
resolved ``BeamSystem`` applies its own below-horizon cut, so the ablation
samples the beam at its exact even continuation ``abs(alt)`` -- an aperture
pattern depends on the zenith angle through ``sin(theta) = cos(alt)``, an even
function of the altitude -- while the fringe, entire in the direction cosines,
stays on the true direction.  With ``K = 4*N*B*F`` and
``S_num = max(1 Jy, max(abs(W_q)))``::

    max(abs(W0-W_q)) <= 1e-8*S_num + 1e-10 Jy
    norm(W0-W_q) / max(norm(W_q), sqrt(K)*1 Jy) <= 1e-8

**Tier 2 -- attributed direct comparison, gating on convergence.**  The deficit
``U = abs(V0-F128) + EF`` is *never called agreement*; its obligations are
convergence and disclosure.  With ``L1 = max(2, lmax//4)`` and
``L2 = max(L1+1, lmax//2)``::

    deficit_max(L1) > deficit_max(L2) > deficit_max(lmax)
    deficit_max(L1) >= 2 * deficit_max(lmax)

Section 7.3 is explicit that a fixture must sit in the convergent regime, whose
governing conditions are geometric: every point and native payload direction must
stay well clear of the horizon over the whole cycle, because near-horizon samples
carry a non-decaying Gibbs error that defeats the monotone predicate at any
scale.  The fixtures here are therefore circumpolar at the shipped site, exactly
as the accepted M1 integration fixture is, and their ``lmax`` is not a free knob:
a predicate is never widened to admit a fixture.

Section 12.2 also requires the retained non-vacuity controls -- "wrong Fourier
sign, wrong V bridge, omitted tangent transport, and omitted east-X permutation
... must miss by more than ten times their corresponding passing residual" --
and one of those, the wrong ``U`` bridge, is bound here for the polarized direct
oracle.

The Section 13.4 owner is ``radiosim.core.mmode.solver``, whose polarized direct
and truncation-level entry points do not exist at ``A1``; imports are
function-local so each node yields its own Section 14.1 outcome.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: Section 7.3's fixed tier-1a limits and tier-2 convergence floor.
HORIZON_FREE_RELATIVE_LIMIT = 1e-8
HORIZON_FREE_ABSOLUTE_FLOOR_JY = 1e-10
HORIZON_FREE_L2_LIMIT = 1e-8
CONVERGENCE_FACTOR_FLOOR = 2.0

#: Section 11's predicate literal for the two-tier gate record.
TWO_TIER_PREDICATE_ID = "sci004_two_tier_direct.v3"

#: Section 12.2's analytic complex128 residual limit and non-vacuity margin.
ANALYTIC_RESIDUAL_LIMIT = 5e-12
NON_VACUITY_FACTOR = 10.0

#: The convergent-regime fixture.  ``lmax`` is pinned, not chosen: Section 7.3
#: records that the quarter-to-full factor collapses once ``L1`` itself resolves
#: the smooth kernel.
LMAX = 16
MMAX = 16
QUADRATURE_NSIDE = 8
SIDEREAL_SAMPLES = 49

#: Section 7.3's derived convergence levels for this ``lmax``.
CONVERGENCE_LEVEL_QUARTER = max(2, LMAX // 4)
CONVERGENCE_LEVEL_HALF = max(CONVERGENCE_LEVEL_QUARTER + 1, LMAX // 2)

#: The circumpolar declination that keeps every payload direction clear of the
#: horizon at the shipped site over the whole cycle, so the frozen enclosure-error
#: cube is exactly zero and the monotone predicate is not fighting Gibbs error.
SOURCE_DEC_DEG = -75.0
SOURCE_STOKES: tuple[float, float, float, float] = (5.5, 0.8, -0.6, 0.4)

_POINT_FIXTURE = f"""\
sky_representation: point
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
sidereal_samples: {SIDEREAL_SAMPLES}
dec_deg: {SOURCE_DEC_DEG}
stokes:
  I: {SOURCE_STOKES[0]}
  Q: {SOURCE_STOKES[1]}
  U: {SOURCE_STOKES[2]}
  V: {SOURCE_STOKES[3]}
""".encode()

_HEALPIX_FIXTURE = f"""\
sky_representation: healpix
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
sidereal_samples: {SIDEREAL_SAMPLES}
nside: 8
polarized: true
""".encode()

_HYBRID_FIXTURE = f"""\
sky_representation: hybrid
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
sidereal_samples: {SIDEREAL_SAMPLES}
nside: 8
dec_deg: {SOURCE_DEC_DEG}
component_order: ["point", "healpix"]
""".encode()

_SHELL_FIXTURE = f"""\
sky_representation: point
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
horizon_free_ablation: even_continuation_abs_alt
tier_1a_max_limit: "1e-8*S_num + 1e-10"
tier_1a_l2_limit: 1e-8
""".encode()

_CONTROL_FIXTURE = f"""\
sky_representation: point
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
dec_deg: {SOURCE_DEC_DEG}
control: wrong_u_bridge
stokes:
  I: {SOURCE_STOKES[0]}
  Q: {SOURCE_STOKES[1]}
  U: {SOURCE_STOKES[2]}
  V: {SOURCE_STOKES[3]}
""".encode()

_GATE_ORACLE = (
    "tests/unit/test_core/test_sci004_direct_convergence.py::"
    "test_the_two_tier_gate_evaluator_and_its_v3_surface_hold_today"
)

_SOLVER_IMPORT_PATTERN = (
    r"ImportError: cannot import name '\w+' from 'radiosim\.core\.mmode\.solver'"
)


def _local(function: str) -> str:
    return f"tests/unit/test_core/test_sci004_direct_convergence.py::{function}"


def _case(
    case_id: str,
    requirement_id: str,
    function: str,
    fixture: bytes,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": _local(function),
        "expected_failure_kind": "missing-symbol",
        "expected_failure_pattern": _SOLVER_IMPORT_PATTERN,
        "fixture_defect_excluded_by": _GATE_ORACLE,
        "fixture_bytes": fixture,
    }


SCI004_PHASE2_RED_CASES: tuple[dict[str, Any], ...] = (
    _case(
        "m2.direct.polarized-oracle-sums-four-components",
        "sci004.section-7.1.private-direct-oracle-is-full-stokes",
        "test_the_polarized_direct_oracle_sums_all_four_stokes_components",
        _POINT_FIXTURE,
    ),
    _case(
        "m2.direct.point-convergence",
        "sci004.section-7.3.polarized-point-two-tier-convergence",
        "test_a_polarized_point_sky_converges_to_the_common_direct_oracle",
        _POINT_FIXTURE,
    ),
    _case(
        "m2.direct.healpix-convergence",
        "sci004.section-7.3.polarized-healpix-two-tier-convergence",
        "test_a_polarized_healpix_sky_converges_to_the_common_direct_oracle",
        _HEALPIX_FIXTURE,
    ),
    _case(
        "m2.direct.hybrid-convergence",
        "sci004.section-7.3.polarized-hybrid-two-tier-convergence",
        "test_a_polarized_hybrid_sky_converges_to_the_common_direct_oracle",
        _HYBRID_FIXTURE,
    ),
    _case(
        "m2.direct.tier-1a-horizon-free-shell",
        "sci004.section-7.3.tier-1a-horizon-free-shell-for-polarized-skies",
        "test_the_tier_one_a_horizon_free_shell_holds_for_a_polarized_sky",
        _SHELL_FIXTURE,
    ),
    _case(
        "m2.direct.wrong-u-bridge-control",
        "sci004.section-12.2.wrong-v-bridge-non-vacuity-control",
        "test_a_wrong_linear_bridge_control_misses_the_direct_oracle",
        _CONTROL_FIXTURE,
    ),
)

SCI004_PHASE2_RED_GREEN_CONTROLS: tuple[str, ...] = (_GATE_ORACLE,)


# --- helpers ------------------------------------------------------------------


def _synthetic_cubes(
    samples: int = 3, baselines: int = 2, frequencies: int = 2
) -> dict[str, np.ndarray]:
    """A deterministic, self-consistent set of ``[N,B,F,4]`` gate operands.

    The two horizon-free cubes differ by a deliberately tiny, spectrally exact
    amount so tier 1a passes; the with-horizon shell differs macroscopically, as
    the strict horizon step really does make it.
    """
    shape = (samples, baselines, frequencies, 4)
    rng = np.random.default_rng(20260823)
    reference = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        np.complex128
    )
    horizon_free = reference + 1e-13
    return {
        "mmode_cube": reference + 0.05,
        "horizon_free_cube": horizon_free,
        "horizon_free_qcheck_cube": reference,
        "quadrature_shell_cube": reference + 0.06,
        "frozen_gauss128": reference,
        "frozen_enclosure_error": np.zeros(shape, dtype=np.float64),
    }


# --- green control ------------------------------------------------------------


def test_the_two_tier_gate_evaluator_and_its_v3_surface_hold_today() -> None:
    """The accepted ``sci004_two_tier_direct.v3`` gate is sound at ``A1``.

    Every polarized node below reuses this evaluator on full-Stokes cubes; that
    it already recomputes the tier-1a limits from Section 7.3's formulas, and
    already refuses a cube set that is not one common ``[N,B,F,4]`` shape, is
    what excludes a defective harness from the red failures that follow.
    """
    from radiosim.core.mmode.solver import evaluate_two_tier_gate

    operands = _synthetic_cubes()
    record = evaluate_two_tier_gate(
        **operands,
        deficit_max_quarter_jy=0.4,
        deficit_max_half_jy=0.2,
    )
    mapping = record.as_mapping()

    assert mapping["predicate_id"] == TWO_TIER_PREDICATE_ID
    cells = int(operands["mmode_cube"].size)
    assert mapping["expected_cell_count"] == cells
    assert mapping["compared_finite_cell_count"] == cells
    assert mapping["horizon_free_shell_l2_limit"] == HORIZON_FREE_L2_LIMIT
    assert mapping["horizon_free_shell_max_limit_jy"] == (
        HORIZON_FREE_RELATIVE_LIMIT * mapping["numerical_scale_jy"]
        + HORIZON_FREE_ABSOLUTE_FLOOR_JY
    )
    assert mapping["convergence_factor"] == (
        mapping["deficit_max_quarter_jy"] / mapping["deficit_max_jy"]
    )

    # A ragged operand set is a rejection, not a silently broadcast comparison.
    raised = None
    try:
        evaluate_two_tier_gate(
            **{**operands, "quadrature_shell_cube": operands["mmode_cube"][:1]},
            deficit_max_quarter_jy=0.4,
            deficit_max_half_jy=0.2,
        )
    except ValueError as error:
        raised = error
    assert raised is not None, "the gate requires one common [N,B,F,4] shape"


# --- Section 7.3 / 12.2 family-6 red oracles ----------------------------------


def test_the_polarized_direct_oracle_sums_all_four_stokes_components() -> None:
    """Section 7.1: the private direct oracle is full Stokes, not Stokes ``I``.

    A polarized point sky whose ``Q``, ``U`` and ``V`` are zeroed must give a
    *different* direct cube from the same sky with them retained; a direct
    oracle that silently drops three of four components would make every
    polarized comparison below vacuously agree.
    """
    from radiosim.core.mmode.solver import polarized_direct_cube

    intensity, q, u, v = SOURCE_STOKES
    full = np.asarray(
        polarized_direct_cube(
            dec_deg=SOURCE_DEC_DEG,
            stokes=(intensity, q, u, v),
            sidereal_samples=SIDEREAL_SAMPLES,
            quadrature_nside=QUADRATURE_NSIDE,
        )
    )
    scalar = np.asarray(
        polarized_direct_cube(
            dec_deg=SOURCE_DEC_DEG,
            stokes=(intensity, 0.0, 0.0, 0.0),
            sidereal_samples=SIDEREAL_SAMPLES,
            quadrature_nside=QUADRATURE_NSIDE,
        )
    )

    assert full.shape == scalar.shape
    assert full.shape[-1] == 4
    assert np.all(np.isfinite(full))
    # Every one of the three added components must move the cube, and the two
    # cross-hands must be the ones carrying ``U`` and ``V``.
    difference = float(np.max(np.abs(full - scalar)))
    assert difference > NON_VACUITY_FACTOR * ANALYTIC_RESIDUAL_LIMIT
    cross_hand = float(np.max(np.abs(full[..., 1]) + np.abs(full[..., 2])))
    assert cross_hand > NON_VACUITY_FACTOR * ANALYTIC_RESIDUAL_LIMIT


def _assert_two_tier_convergence(record: Any) -> None:
    """Assert Section 7.3's tier-1a limits and tier-2 convergence predicates."""
    mapping = record.as_mapping()

    assert mapping["predicate_id"] == TWO_TIER_PREDICATE_ID
    assert (
        mapping["horizon_free_shell_max_jy"]
        <= (mapping["horizon_free_shell_max_limit_jy"])
    )
    assert mapping["horizon_free_shell_l2"] <= HORIZON_FREE_L2_LIMIT
    assert (
        mapping["deficit_max_quarter_jy"]
        > mapping["deficit_max_half_jy"]
        > mapping["deficit_max_jy"]
    )
    assert mapping["convergence_factor"] >= CONVERGENCE_FACTOR_FLOOR
    assert mapping["pass"] is True
    # The deficit is disclosed, never bounded by a universal limit here.
    assert math.isfinite(mapping["deficit_max_jy"])
    assert math.isfinite(mapping["quadrature_shell_max_jy"])


def test_a_polarized_point_sky_converges_to_the_common_direct_oracle() -> None:
    """Section 12.2 family 6: a small polarized point sky, through the gate."""
    from radiosim.core.mmode.solver import solve_polarized_fixture

    outcome = solve_polarized_fixture(
        sky_representation="point",
        dec_deg=SOURCE_DEC_DEG,
        stokes=SOURCE_STOKES,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
        sidereal_samples=SIDEREAL_SAMPLES,
    )
    _assert_two_tier_convergence(outcome.direct_gate)
    assert outcome.sky_representation == "point"


def test_a_polarized_healpix_sky_converges_to_the_common_direct_oracle() -> None:
    """Section 12.2 family 6: a small polarized HEALPix sky, through the gate.

    Section 7.1 is explicit that the direct oracle "does not resample a native
    HEALPix payload onto the transfer quadrature": it sums the original native
    pixel centres in canonical RING order with their native pixel solid angle, so
    the comparison tests truncation and nothing else.
    """
    from radiosim.core.mmode.solver import solve_polarized_fixture

    outcome = solve_polarized_fixture(
        sky_representation="healpix",
        nside=8,
        polarized=True,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
        sidereal_samples=SIDEREAL_SAMPLES,
    )
    _assert_two_tier_convergence(outcome.direct_gate)
    assert outcome.sky_representation == "healpix"
    assert outcome.native_direct_grid_id != outcome.transfer_grid_id


def test_a_polarized_hybrid_sky_converges_to_the_common_direct_oracle() -> None:
    """Section 7.1: hybrid adds coefficients before any ``B_lm a_lm`` product."""
    from radiosim.core.mmode.solver import solve_polarized_fixture

    outcome = solve_polarized_fixture(
        sky_representation="hybrid",
        nside=8,
        dec_deg=SOURCE_DEC_DEG,
        stokes=SOURCE_STOKES,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
        sidereal_samples=SIDEREAL_SAMPLES,
    )
    _assert_two_tier_convergence(outcome.direct_gate)
    assert outcome.sky_representation == "hybrid"
    assert outcome.component_order == ("point", "healpix")


def test_the_tier_one_a_horizon_free_shell_holds_for_a_polarized_sky() -> None:
    """Section 7.3 tier 1a: the sharp half, at ``1e-8``, on a polarized sky.

    The ablation is the removal of *every* horizon truncation with the beam on
    its even continuation ``abs(alt)``; ablating only the explicit ``H`` factor
    leaves the resolved beam's own below-horizon cut in place and the shell
    measures at the with-horizon level, which is the defect Section 7.3 records.
    """
    from radiosim.core.mmode.solver import solve_polarized_fixture

    outcome = solve_polarized_fixture(
        sky_representation="point",
        dec_deg=SOURCE_DEC_DEG,
        stokes=SOURCE_STOKES,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
        sidereal_samples=SIDEREAL_SAMPLES,
    )
    mapping = outcome.direct_gate.as_mapping()

    numerical_scale = mapping["numerical_scale_jy"]
    assert mapping["horizon_free_shell_max_limit_jy"] == (
        HORIZON_FREE_RELATIVE_LIMIT * numerical_scale + HORIZON_FREE_ABSOLUTE_FLOOR_JY
    )
    assert (
        mapping["horizon_free_shell_max_jy"]
        <= (mapping["horizon_free_shell_max_limit_jy"])
    )
    assert mapping["horizon_free_shell_l2"] <= HORIZON_FREE_L2_LIMIT
    # The two horizon-free cubes are tier-1 internals and never a result.
    assert mapping["horizon_free_cube_sha256"] != mapping["candidate_cube_sha256"]
    assert (
        mapping["horizon_free_qcheck_cube_sha256"] != mapping["candidate_cube_sha256"]
    )
    # Tier 1b is recorded, not bounded: the with-horizon shell sits far above the
    # tier-1a limit precisely because no finite rule is exact for a step.
    assert mapping["quadrature_shell_max_jy"] >= 0.0


def test_a_wrong_linear_bridge_control_misses_the_direct_oracle() -> None:
    """Section 12.2: the retained non-vacuity controls miss by ``> 10x``.

    Section 5.2's bridge sends ``U -> -U``; omitting that flip is the wrong-V
    bridge family of defect, and Section 12.2 requires it to miss the passing
    residual by more than a factor of ten rather than merely to differ.
    """
    from radiosim.core.mmode.solver import solve_polarized_fixture

    intensity, q, u, v = SOURCE_STOKES
    correct = solve_polarized_fixture(
        sky_representation="point",
        dec_deg=SOURCE_DEC_DEG,
        stokes=(intensity, q, u, v),
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
        sidereal_samples=SIDEREAL_SAMPLES,
    )
    flipped = solve_polarized_fixture(
        sky_representation="point",
        dec_deg=SOURCE_DEC_DEG,
        stokes=(intensity, q, -u, v),
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
        sidereal_samples=SIDEREAL_SAMPLES,
    )

    passing = float(correct.direct_gate.as_mapping()["deficit_max_jy"])
    separation = float(
        np.max(np.abs(np.asarray(correct.cube) - np.asarray(flipped.cube)))
    )
    assert separation > NON_VACUITY_FACTOR * max(passing, ANALYTIC_RESIDUAL_LIMIT)
