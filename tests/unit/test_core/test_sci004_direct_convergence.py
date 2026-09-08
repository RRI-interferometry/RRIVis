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


# D36 causal controls: bounded dependency injection into the real direct owner.
# No scan, harmonic solve, or production expected-value helper participates.
D36_CAUSAL_NODES = (
    "test_d36_operational_smooth_cube_owns_its_fringe",
    "test_d36_shifted_roots_keep_opposite_piece_signs",
    "test_d36_frozen_ambiguity_does_not_classify_operational_piece",
    "test_d36_piece_radius_is_the_least_binary64_upper_bound",
    "test_d36_error_cells_add_stored_radii_toward_positive_infinity",
    "test_d36_guarded_partition_keeps_inner_root_bounds",
)


def _d36_direct_case(
    monkeypatch: Any,
    *,
    frozen_roots: tuple[tuple[Any, ...], ...] = ((),),
    operational_roots: tuple[tuple[Any, ...], ...] = ((),),
    frozen_sign_root: Any = None,
    operational_sign_root: Any = None,
    frozen_below: bool = False,
    operational_east: float = 0.0,
    intensity: float = 2.0,
    public_operational: bool = False,
) -> tuple[dict[str, Any], list[Any]]:
    """Inject owned geometry/payload only; execute real kernels and reductions.

    The 49 exact exposure intervals match N=49, integration_fraction=1. The
    scanner is intentionally absent: controlled trajectories isolate direct
    ownership, not scan correctness or successful-family admission.
    """
    from fractions import Fraction
    from importlib import import_module

    solver: Any = import_module("radiosim.core.mmode.solver")

    class Grid:
        horizon_domain = (Fraction(-1, 98), Fraction(97, 98))
        sidereal_samples = 49
        canonical_era_grid_sha256 = "0" * 64

        def exposure_turns(self, index: int) -> tuple[Fraction, Fraction]:
            return Fraction(2 * index - 1, 98), Fraction(2 * index + 1, 98)

    class Beam:
        def evaluate_jones(self, _antenna: Any, **kwargs: Any) -> np.ndarray:
            size = len(kwargs["altitude_rad"])
            return np.broadcast_to(np.eye(2, dtype=np.complex128), (size, 2, 2))

    class Frozen:
        def __init__(self, roots: tuple[Any, ...]):
            self.roots = roots

        def value_interval(
            self, lower: Fraction, upper: Fraction
        ) -> tuple[float, float]:
            if frozen_sign_root is not None:
                sign = 1.0 if (lower + upper) / 2 > frozen_sign_root else -1.0
            else:
                sign = -1.0 if frozen_below else 1.0
            return sign, sign

    calls: list[Any] = []

    class Operational(solver._operational_directions):
        pass

        def at_pairs(self, indices: Any, turns: Any) -> np.ndarray:
            calls.append((tuple(indices), tuple(turns)))
            result = np.tile(
                [operational_east, 0.0, math.sqrt(1.0 - operational_east**2)],
                (len(turns), 1),
            )
            if operational_sign_root is not None:
                result[:, 2] = [
                    1.0 if Fraction(turn) > operational_sign_root else -1.0
                    for turn in turns
                ]
            elif any(operational_roots):
                for position, (index, turn) in enumerate(
                    zip(indices, turns, strict=True)
                ):
                    roots = operational_roots[index]
                    if roots:
                        previous = [root for root in roots if root.turn_hi <= turn]
                        following = [root for root in roots if turn <= root.turn_lo]
                        if previous:
                            result[position, 2] = (
                                1.0 if previous[-1].orientation == "rising" else -1.0
                            )
                        elif following:
                            result[position, 2] = (
                                -1.0 if following[0].orientation == "rising" else 1.0
                            )
            return result

    def frozen_enu(_frame: Any, cirs: np.ndarray, _phase: float) -> np.ndarray:
        return np.tile([0.0, 0.0, 1.0], (len(cirs), 1))

    monkeypatch.setattr(solver, "frozen_enu_at_phase", frozen_enu)
    if not public_operational:
        monkeypatch.setattr(solver, "_operational_directions", Operational)
    from radiosim.core.mmode.frame import (
        OperationalHorizonScan,
        build_frozen_frame,
    )

    frame: Any = build_frozen_frame(
        start_time="2020-01-01T00:00:00",
        longitude_deg=0.0,
        latitude_deg=-30.0,
        height_m=0.0,
    )
    grid = Grid()
    context = solver.KernelContext(
        frame=frame,
        beam_system=Beam(),
        antenna_ids=(0, 1),
        selected_pairs=((0, 1),),
        baseline_vectors_enu_m=np.array([[20.0, 0.0, 0.0]]),
        frequencies_hz=np.array([299792458.0]),
        time_mjd=0.0,
    )
    directions = tuple(
        solver.LedgerDirection(
            direction_id=f"point:0:{index}",
            source_kind="point",
            component_index=0,
            source_index=index,
            transfer_role="",
            transfer_nside=0,
            cirs_direction=np.array([0.0, 0.0, 1.0]),
            icrs_ra_rad=0.0,
            icrs_dec_rad=-math.pi / 2 if public_operational else 0.0,
            active_frequency_mask=(intensity != 0.0,),
            resolved_stokes_iau=np.array([[intensity, 0.0, 0.0, 0.0]]),
            integration_weight=1.0,
        )
        for index in range(len(frozen_roots))
    )
    # Construct a controlled successful-scan-shaped owner, not a real census.
    # Actual scan correctness is tested separately; no missing argument is red.
    frame_module: Any = import_module("radiosim.core.mmode.frame")
    live: Any = frame_module._OperationalTrajectory(
        frame,
        grid,
        [row.icrs_ra_rad for row in directions],
        [row.icrs_dec_rad for row in directions],
    )
    live.evaluations = 7 * len(directions)
    live.per_direction = [7] * len(directions)

    def forbidden_live(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("direct evaluation called the live scanner")

    live.at_pairs = forbidden_live
    live.at_common_turn = forbidden_live
    import struct

    def hex64(value: float) -> str:
        return struct.pack(">d", value).hex()

    def rational(value: Fraction) -> str:
        return f"{value.numerator}/{value.denominator}"

    crossings: list[dict[str, Any]] = []
    for direction, roots in zip(directions, operational_roots, strict=True):
        projected: list[dict[str, Any]] = []
        for root in roots:
            signs = (-1, 1) if root.orientation == "rising" else (1, -1)
            projected.append(
                {
                    "direction_id": direction.direction_id,
                    "classification": "scan_crossing",
                    "turn_lo": rational(root.turn_lo),
                    "turn_hi": rational(root.turn_hi),
                    "root_turn_lo": rational(root.turn_lo),
                    "root_turn_hi": rational(root.turn_hi),
                    "root_orientation": root.orientation,
                    "root_residual_f64be": hex64(root.residual),
                    "f_lo_f64be": hex64(0.0),
                    "f_hi_f64be": hex64(0.0),
                    "ceiling_margin_f64be": hex64(0.0),
                    "left_sign": signs[0],
                    "right_sign": signs[1],
                }
            )
            for lower, upper, before, after in (
                (root.ambiguous_span[0], root.turn_lo, float(signs[0]), 0.0),
                (root.turn_hi, root.ambiguous_span[1], 0.0, float(signs[1])),
            ):
                if lower == upper:
                    continue
                projected.append(
                    {
                        "direction_id": direction.direction_id,
                        "classification": "guard_interval",
                        "turn_lo": rational(lower),
                        "turn_hi": rational(upper),
                        "root_turn_lo": None,
                        "root_turn_hi": None,
                        "root_orientation": None,
                        "root_residual_f64be": None,
                        "f_lo_f64be": hex64(before),
                        "f_hi_f64be": hex64(after),
                        "ceiling_margin_f64be": hex64(0.0),
                        "left_sign": int(before > 0) - int(before < 0),
                        "right_sign": int(after > 0) - int(after < 0),
                    }
                )
        projected.sort(key=lambda row: Fraction(row["turn_lo"]))
        for index, row in enumerate(projected):
            crossings.append({**row, "cell_index": index})
    scan = OperationalHorizonScan(
        crossing_rows=tuple(crossings),
        summary_rows=tuple(
            {
                "direction_id": row.direction_id,
                "crossing_count": len(roots),
                "terminal_cell_count": 49,
                "boundary_evaluation_count": 7,
                "min_ceiling_margin_f64be": hex64(0.0),
            }
            for row, roots in zip(directions, operational_roots, strict=True)
        ),
        ledger_sha256="3" * 64,
        roots=operational_roots,
        centre_values=np.ones((49, len(directions))),
        evaluator=live,
        guard_count=sum(row["classification"] == "guard_interval" for row in crossings),
        evaluation_count=live.evaluations,
        isolation_interval_count=49 * len(directions),
        astropy_version="synthetic-control",
        erfa_version="synthetic-control",
        iers_table_sha256=frame.iers_table_sha256,
    )
    state = (live.evaluations, tuple(live.per_direction), scan.manifest())
    preimages: list[Any] = []
    original_digest = solver.object_digest

    def capture_preimage(domain: str, value: Any) -> str:
        if domain in {
            "radiosim.mmode-direct-piece-cell.v1",
            "radiosim.mmode-direct-piece-error.v1",
        }:
            preimages.append((domain, value))
        return original_digest(domain, value)

    monkeypatch.setattr(solver, "object_digest", capture_preimage)
    arguments: dict[str, Any] = {
        "grid": grid,
        "frame": frame,
        "context": context,
        "directions": directions,
        "frozen": tuple(Frozen(roots) for roots in frozen_roots),
        "operational_roots": operational_roots,
        "operational_scan": scan,
        "beam_peak_ceiling": 1.0,
        "input_identity_sha256": "1" * 64,
        "enclosure_manifest_sha256": "2" * 64,
    }
    result = solver._direct_cubes(**arguments)
    assert (live.evaluations, tuple(live.per_direction), scan.manifest()) == state
    result["_d36_preimages"] = preimages
    result["_d36_arguments"] = arguments
    result["_d36_scan"] = scan
    result["_d36_witness"] = solver._operational_directions(frame, directions)
    return result, calls


def _d36_root(lower: Any, upper: Any, **guards: Any) -> Any:
    from radiosim.core.mmode.frame import HorizonRootEnclosure

    return HorizonRootEnclosure(lower, upper, "rising", 0.0, **guards)


def _d36_rows(result: dict[str, Any], lower: Any, upper: Any) -> list[Any]:
    from fractions import Fraction

    return [
        row
        for row in result["split_rows"]
        if Fraction(row["turn_lo"]) == lower
        and Fraction(row["turn_hi"]) == upper
        and row["correlation_index"] == 0
    ]


def test_d36_operational_smooth_cube_owns_its_fringe(monkeypatch: Any) -> None:
    # At lambda=1 m, the real identity-Jones Stokes-I kernel has unit XX/YY.
    # Its analytic fringe for an east displacement epsilon is exp(-2pi*i*20*eps).
    epsilon = 1e-6
    result, calls = _d36_direct_case(monkeypatch, operational_east=epsilon)
    expected = complex(
        math.cos(-math.tau * 20 * epsilon), math.sin(-math.tau * 20 * epsilon)
    )
    np.testing.assert_allclose(result["F128"][..., 0], 1.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["O128"][..., 0], expected, rtol=0, atol=1e-14)
    assert calls, "operational direction owner must actually be evaluated"


def test_d36_shifted_roots_keep_opposite_piece_signs(monkeypatch: Any) -> None:
    from fractions import Fraction

    frozen_root = Fraction(1, 4)
    operational_root = frozen_root + Fraction(1, 2**42)
    result, _ = _d36_direct_case(
        monkeypatch,
        frozen_roots=((_d36_root(frozen_root, frozen_root),),),
        operational_roots=((_d36_root(operational_root, operational_root),),),
        frozen_sign_root=frozen_root,
        operational_sign_root=operational_root,
    )
    rows = _d36_rows(result, frozen_root, operational_root)
    assert len(rows) == 1
    assert rows[0]["frozen_piece_class"] == "smooth_above"
    assert rows[0]["operational_piece_class"] == "smooth_below"
    assert rows[0]["operational_gauss128_node_count"] == 0


def test_d36_frozen_ambiguity_does_not_classify_operational_piece(
    monkeypatch: Any,
) -> None:
    from fractions import Fraction

    lower, upper = Fraction(1, 4), Fraction(1, 4) + Fraction(1, 2**48)
    result, _ = _d36_direct_case(
        monkeypatch, frozen_roots=((_d36_root(lower, upper),),)
    )
    rows = _d36_rows(result, lower, upper)
    assert len(rows) == 1
    assert rows[0]["frozen_piece_class"] == "root_enclosure"
    assert rows[0]["operational_piece_class"] == "smooth_above"
    assert rows[0]["operational_gauss128_node_count"] == 128


def test_d36_piece_radius_is_the_least_binary64_upper_bound(monkeypatch: Any) -> None:
    import struct
    from fractions import Fraction

    lower = Fraction(1, 4)
    upper = Fraction(40407052320769, 161628209283072)
    exact = Fraction(4503599627370497, 14855280471424563298789490688)
    expected = struct.unpack(">d", bytes.fromhex("3d55555555555557"))[0]
    assert Fraction(math.nextafter(expected, -math.inf)) < exact <= Fraction(expected)
    result, _ = _d36_direct_case(
        monkeypatch, frozen_roots=((_d36_root(lower, upper),),), frozen_below=True
    )
    rows = _d36_rows(result, lower, upper)
    assert len(rows) == 1
    assert rows[0]["frozen_enclosure_error_f64be"] == "3d55555555555557"
    assert result["EF"][12, 0, 0, 0] == expected


def test_d36_error_cells_add_stored_radii_toward_positive_infinity(
    monkeypatch: Any,
) -> None:
    import struct
    from fractions import Fraction

    lower = Fraction(1, 4)
    ends = (lower + Fraction(1, 49 * 2**41), lower + Fraction(1, 49 * 2**43))
    exact = Fraction(22517998136852485, 39614081257132168796771975168)
    expected = struct.unpack(">d", bytes.fromhex("3d64000000000002"))[0]
    assert Fraction(math.nextafter(expected, -math.inf)) < exact <= Fraction(expected)
    result, _ = _d36_direct_case(
        monkeypatch,
        frozen_roots=tuple((_d36_root(lower, end),) for end in ends),
        operational_roots=((), ()),
        frozen_below=True,
    )
    for end, bits in zip(ends, ("3d60000000000001", "3d40000000000001"), strict=True):
        rows = _d36_rows(result, lower, end)
        assert len(rows) == 1
        assert rows[0]["frozen_enclosure_error_f64be"] == bits
    assert result["EF"][12, 0, 0, 0] == expected


def test_d36_guarded_partition_keeps_inner_root_bounds() -> None:
    from fractions import Fraction
    from importlib import import_module

    solver: Any = import_module("radiosim.core.mmode.solver")

    start = Fraction(1, 4)
    step = Fraction(1, 2**49)
    bounds = tuple(start + index * step for index in range(4))
    exposure = (Fraction(23, 98), Fraction(25, 98))
    root = _d36_root(
        bounds[1], bounds[2], guard_turn_lo=bounds[0], guard_turn_hi=bounds[3]
    )
    expected = (exposure[0], *bounds, exposure[1])
    # Duplicate owners and touching exposure endpoints cannot duplicate cuts.
    endpoint = _d36_root(exposure[0], exposure[0])
    assert solver._piece_cuts(*exposure, (endpoint,), (root, root)) == expected


def test_d36_exact_radius_and_zero_payload_controls(monkeypatch: Any) -> None:
    from fractions import Fraction

    lower, upper = Fraction(1, 4), Fraction(1, 4) + Fraction(1, 49 * 2**41)
    result, _ = _d36_direct_case(
        monkeypatch, frozen_roots=((_d36_root(lower, upper),),), frozen_below=True
    )
    expected = float.fromhex("0x1.0000000000001p-41")
    assert result["EF"][12, 0, 0, 0] == expected
    assert (
        _d36_rows(result, lower, upper)[0]["frozen_enclosure_error_f64be"]
        == "3d60000000000001"
    )
    empty, _ = _d36_direct_case(
        monkeypatch,
        frozen_roots=((_d36_root(lower, upper),),),
        frozen_below=True,
        intensity=0.0,
    )
    for key in ("F64", "F128", "O64", "O128", "EF", "EO"):
        assert not np.any(empty[key])


def test_d36_identity_fringe_and_zero_error_control(monkeypatch: Any) -> None:
    result, _ = _d36_direct_case(monkeypatch)
    for key in ("F64", "F128", "O64", "O128"):
        np.testing.assert_allclose(result[key][..., (0, 3)], 1.0, rtol=0, atol=1e-14)
        assert not np.any(result[key][..., (1, 2)])
    assert not np.any(result["EF"])
    assert not np.any(result["EO"])


def test_d36_adapter_owned_open_pieces_and_rejections(monkeypatch: Any) -> None:
    """A singleton endpoint does not extend ambiguity into either open piece."""
    from dataclasses import replace
    from fractions import Fraction
    from importlib import import_module

    import pytest

    solver: Any = import_module("radiosim.core.mmode.solver")
    root = Fraction(1, 4)
    result, _ = _d36_direct_case(
        monkeypatch,
        operational_roots=((_d36_root(root, root),),),
        operational_sign_root=root,
    )
    scan, witness = result["_d36_scan"], result["_d36_witness"]
    classify = solver._classify_operational_direct_piece
    arguments = {"scan": scan, "witness_owner": witness, "direction_index": 0}
    before = (scan.evaluator.evaluations, tuple(scan.evaluator.per_direction))
    assert (
        classify(**arguments, lower=root - Fraction(1, 1000), upper=root)
        == "smooth_below"
    )
    assert (
        classify(**arguments, lower=root, upper=root + Fraction(1, 1000))
        == "smooth_above"
    )
    with pytest.raises(ValueError, match="omits a root bound"):
        classify(
            **arguments, lower=root - Fraction(1, 1000), upper=root + Fraction(1, 1000)
        )
    for invalid in (
        replace(scan, crossing_rows=()),
        replace(scan, iers_table_sha256="f" * 64),
        replace(scan, summary_rows=()),
    ):
        with pytest.raises(ValueError, match="owner mismatch|projection is incomplete"):
            classify(
                **{**arguments, "scan": invalid},
                lower=root,
                upper=root + Fraction(1, 1000),
            )
    with pytest.raises(ValueError, match="owner mismatch"):
        classify(
            **{**arguments, "witness_owner": scan.evaluator},
            lower=root,
            upper=root + Fraction(1, 1000),
        )
    with pytest.raises(ValueError, match="successful scan"):
        classify(
            **{**arguments, "scan": object()},
            lower=root,
            upper=root + Fraction(1, 1000),
        )
    assert (scan.evaluator.evaluations, tuple(scan.evaluator.per_direction)) == before


def test_d36_all_model_arrays_join_independent_serialized_preimages(
    monkeypatch: Any,
) -> None:
    """Rebuild the cell digests and all four reductions from serialized operands."""
    import hashlib
    import json
    import struct

    epsilon = 1e-6
    result, _ = _d36_direct_case(monkeypatch, operational_east=epsilon)
    cubes = {key: np.zeros_like(result[key]) for key in ("F64", "F128", "O64", "O128")}
    node_buffers: dict[tuple[str, int, int], np.ndarray] = {}
    rows = result["split_rows"]
    lookup = {
        (r["sample_index"], r["correlation_index"], r["piece_index"]): r for r in rows
    }
    for domain, value in result["_d36_preimages"]:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        digest = hashlib.sha256(
            domain.encode() + b"\0" + struct.pack(">Q", len(raw)) + raw
        ).hexdigest()
        row = lookup[
            value["sample_index"], value["correlation_index"], value["piece_index"]
        ]
        model = value["model"]
        if domain.endswith("error.v1"):
            assert row[f"{model}_enclosure_error_preimage_sha256"] == digest
            assert value["enclosure_error_f64be"] == "0000000000000000"
            continue
        order = value["gauss_order"]
        assert row[f"{model}_gauss{order}_contribution_sha256"] == digest

        def decode(h: str) -> float:
            return struct.unpack(">d", bytes.fromhex(h))[0]

        parts = [decode(h) for h in value["integrand_reim_f64be"]]
        samples = np.array(parts[::2]) + 1j * np.array(parts[1::2])
        expected = (
            0j
            if value["correlation_index"] in (1, 2)
            else (
                complex(
                    math.cos(-math.tau * 20 * epsilon),
                    math.sin(-math.tau * 20 * epsilon),
                )
                if model == "operational"
                else 1 + 0j
            )
        )
        np.testing.assert_allclose(samples, expected, rtol=0, atol=1e-15)
        weights = np.array([decode(h) for h in value["weights_f64be"]])
        contribution = complex(np.sum(samples * weights))
        assert (
            struct.pack(">d", contribution.real).hex()
            == value["contribution_real_f64be"]
        )
        assert (
            struct.pack(">d", contribution.imag).hex()
            == value["contribution_imag_f64be"]
        )
        key = ("F" if model == "frozen" else "O") + str(order)
        group = (key, value["sample_index"], value["piece_index"])
        buffer = node_buffers.setdefault(
            group, np.zeros((order, 1, 1, 4), dtype=np.complex128)
        )
        buffer[:, 0, 0, value["correlation_index"]] = samples * weights
    # Preserve the existing vector-axis reduction separately from the recorded
    # scalar per-cell sum: NumPy may choose different addition groupings.
    for (key, sample, _piece), buffer in node_buffers.items():
        cubes[key][sample] += np.sum(buffer, axis=0)
    for key, cube in cubes.items():
        np.testing.assert_array_equal(cube, result[key])
    assert not np.any(result["EF"]) and not np.any(result["EO"])


def test_d36_public_astropy_operational_nodes_own_direct_preimages(
    monkeypatch: Any,
) -> None:
    """Public AltAz at the retained UT1 nodes supplies O's real fringe."""
    import struct
    from fractions import Fraction
    from importlib import import_module

    u: Any = import_module("astropy.units")
    coordinates: Any = import_module("astropy.coordinates")
    time: Any = import_module("astropy.time")

    from radiosim.core.mmode.time import (
        ERA_RATE_TURNS_PER_UT1_DAY,
        installed_iers_context,
    )

    result, _ = _d36_direct_case(monkeypatch, public_operational=True)
    frame = result["_d36_scan"].evaluator._frame
    rows = [
        v
        for d, v in result["_d36_preimages"]
        if d.endswith("cell.v1")
        and v["model"] == "operational"
        and v["sample_index"] == 0
        and v["correlation_index"] == 0
    ]
    assert {v["gauss_order"] for v in rows} == {64, 128}
    for row in rows:
        turns = [Fraction(v) for v in row["node_turns"]]
        jd1, jd2 = frame.ut1_two_part
        times = time.Time(
            np.full(len(turns), float(jd1)),
            [float(Fraction(jd2) + v / ERA_RATE_TURNS_PER_UT1_DAY) for v in turns],
            format="jd",
            scale="ut1",
        )
        site = coordinates.EarthLocation.from_geodetic(0 * u.deg, -30 * u.deg, 0 * u.m)
        with installed_iers_context():
            horizontal = coordinates.SkyCoord(
                ra=np.zeros(len(turns)) * u.rad,
                dec=np.full(len(turns), -math.pi / 2) * u.rad,
                frame="icrs",
            ).transform_to(coordinates.AltAz(obstime=times, location=site, pressure=0))
        east = np.cos(horizontal.alt.rad) * np.sin(horizontal.az.rad)
        expected = np.exp(-1j * math.tau * 20 * east)
        parts = [
            struct.unpack(">d", bytes.fromhex(v))[0]
            for v in row["integrand_reim_f64be"]
        ]
        actual = np.array(parts[::2]) + 1j * np.array(parts[1::2])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        assert np.max(np.abs(actual - 1)) > 1e-4
    assert np.max(np.abs(result["O128"] - result["F128"])) > 1e-4
    assert result["_d36_scan"].evaluator.evaluations == 7


def test_d36_adapter_guard_interiors_and_excluded_upper_endpoint(
    monkeypatch: Any,
) -> None:
    from dataclasses import replace
    from fractions import Fraction
    from importlib import import_module

    import pytest

    solver: Any = import_module("radiosim.core.mmode.solver")
    root = Fraction(1, 4)
    step = Fraction(1, 2**48)
    enclosure = _d36_root(
        root, root + step, guard_turn_lo=root - step, guard_turn_hi=root + 2 * step
    )
    result, _ = _d36_direct_case(monkeypatch, operational_roots=((enclosure,),))
    scan, witness = result["_d36_scan"], result["_d36_witness"]
    classify = solver._classify_operational_direct_piece
    args = {"scan": scan, "witness_owner": witness, "direction_index": 0}
    for lo, hi in (
        (root - step, root),
        (root, root + step),
        (root + step, root + 2 * step),
    ):
        assert classify(**args, lower=lo, upper=hi) == "root_enclosure"
    assert (
        classify(**args, lower=root + 2 * step, upper=root + 3 * step) == "smooth_above"
    )
    with pytest.raises(ValueError, match="omits a guard bound"):
        classify(**args, lower=root - 2 * step, upper=root)
    endpoint = scan.evaluator._grid.horizon_domain[1]
    endpoint_text = f"{endpoint.numerator}/{endpoint.denominator}"
    excluded = {
        **next(
            row
            for row in scan.crossing_rows
            if row["classification"] == "scan_crossing"
        ),
        "classification": "excluded_upper_endpoint",
        "cell_index": 48,
        "turn_lo": endpoint_text,
        "turn_hi": endpoint_text,
        "root_turn_lo": endpoint_text,
        "root_turn_hi": endpoint_text,
        "root_orientation": "setting",
        "left_sign": 1,
        "right_sign": -1,
    }
    amended = replace(
        scan,
        crossing_rows=(*scan.crossing_rows, excluded),
        summary_rows=({**scan.summary_rows[0], "crossing_count": 2},),
    )
    assert (
        classify(**{**args, "scan": amended}, lower=endpoint - step, upper=endpoint)
        == "smooth_above"
    )
    wrong = replace(
        amended,
        crossing_rows=(
            *scan.crossing_rows,
            {**excluded, "root_turn_hi": str(endpoint - step)},
        ),
    )
    with pytest.raises(ValueError, match="invalid crossing/excluded projection"):
        classify(**{**args, "scan": wrong}, lower=endpoint - step, upper=endpoint)
    original = witness.at_pairs

    def zero_witness(indices: Any, turns: Any) -> np.ndarray:
        return np.zeros((len(turns), 3))

    monkeypatch.setattr(witness, "at_pairs", zero_witness)
    with pytest.raises(ValueError, match="zero witness"):
        classify(**args, lower=root + 2 * step, upper=root + 3 * step)
    monkeypatch.setattr(witness, "at_pairs", original)


def test_d36_direct_rejects_foreign_owners_before_nodes(monkeypatch: Any) -> None:
    from dataclasses import replace
    from importlib import import_module

    import pytest

    solver: Any = import_module("radiosim.core.mmode.solver")
    result, calls = _d36_direct_case(monkeypatch)
    arguments = result["_d36_arguments"]
    scan = result["_d36_scan"]
    before = (
        len(calls),
        scan.evaluator.evaluations,
        tuple(scan.evaluator.per_direction),
    )
    changes = (
        {"frame": replace(arguments["frame"])},
        {"grid": type(arguments["grid"])()},
        {"operational_roots": tuple(root for root in scan.roots)},
        {"operational_scan": replace(scan, iers_table_sha256="f" * 64)},
        {"directions": (replace(arguments["directions"][0], direction_id="other"),)},
        {"directions": (replace(arguments["directions"][0], icrs_ra_rad=0.01),)},
    )
    for change in changes:
        with pytest.raises(ValueError, match="ownership mismatch"):
            solver._direct_cubes(**{**arguments, **change})
        assert (
            len(calls),
            scan.evaluator.evaluations,
            tuple(scan.evaluator.per_direction),
        ) == before


def test_d36_complete_projection_refuses_rehashed_local_contradictions(
    monkeypatch: Any,
) -> None:
    import struct
    from dataclasses import replace
    from fractions import Fraction
    from importlib import import_module

    import pytest

    solver: Any = import_module("radiosim.core.mmode.solver")
    lower, step = Fraction(1, 4), Fraction(1, 2**48)
    root = _d36_root(
        lower, lower + step, guard_turn_lo=lower - step, guard_turn_hi=lower + 2 * step
    )
    result, _ = _d36_direct_case(monkeypatch, operational_roots=((root,),))
    scan, witness = result["_d36_scan"], result["_d36_witness"]
    classify = solver._classify_operational_direct_piece
    args = {
        "witness_owner": witness,
        "direction_index": 0,
        "lower": lower,
        "upper": lower + step,
    }
    assert classify(scan=scan, **args) == "root_enclosure"
    rows = list(scan.crossing_rows)
    assert [row["classification"] for row in rows] == [
        "guard_interval",
        "scan_crossing",
        "guard_interval",
    ]

    def text(v: Fraction) -> str:
        return f"{v.numerator}/{v.denominator}"

    shifted = ({**rows[0], "turn_lo": text(lower - 2 * step)}, *rows[1:])
    endpoint = (
        {**rows[0], "f_hi_f64be": struct.pack(">d", 2**-50).hex(), "right_sign": 1},
        *rows[1:],
    )
    residual = (
        rows[0],
        {**rows[1], "root_residual_f64be": struct.pack(">d", 2**-50).hex()},
        rows[2],
    )
    orphan = {
        **rows[2],
        "cell_index": 3,
        "turn_lo": "3/4",
        "turn_hi": text(Fraction(3, 4) + step),
    }
    invalid_excluded = {**rows[1], "classification": "excluded_upper_endpoint"}
    variants = (
        (
            replace(scan, crossing_rows=(rows[1],), guard_count=0),
            "root/guard/residual mismatch",
        ),
        (replace(scan, crossing_rows=shifted), "root/guard/residual mismatch"),
        (replace(scan, crossing_rows=endpoint), "guard endpoint mismatch"),
        (replace(scan, crossing_rows=residual), "root/guard/residual mismatch"),
        (
            replace(scan, crossing_rows=(rows[0], invalid_excluded, rows[2])),
            "invalid crossing/excluded projection",
        ),
        (replace(scan, guard_count=0), "inconsistent scan counts"),
        (
            replace(scan, crossing_rows=(*rows, orphan), guard_count=3),
            "orphan guard interval",
        ),
        (
            replace(
                scan,
                crossing_rows=(rows[0], {**rows[0], "cell_index": 1}, *rows[1:]),
                guard_count=3,
            ),
            "projection domain/order",
        ),
        (
            replace(
                scan, summary_rows=({**scan.summary_rows[0], "crossing_count": 2},)
            ),
            "projection is incomplete",
        ),
        (
            replace(scan, roots=((replace(root, orientation="setting"),),)),
            "root/guard/residual mismatch",
        ),
    )
    variants += (
        (
            replace(scan, roots=((replace(root, turn_lo=float(root.turn_lo)),),)),
            "root/guard/residual mismatch",
        ),
        (
            replace(
                scan, crossing_rows=({**rows[0], "direction_id": "foreign"}, *rows[1:])
            ),
            "inconsistent scan counts/identifiers",
        ),
    )
    for invalid, message in variants:
        with pytest.raises(ValueError, match=message):
            classify(scan=invalid, **args)
    assert scan.evaluator.evaluations == 7


def test_d36_orientation_constrains_separate_witness_for_both_crossings(
    monkeypatch: Any,
) -> None:
    from dataclasses import replace
    from fractions import Fraction
    from importlib import import_module

    import pytest

    solver: Any = import_module("radiosim.core.mmode.solver")
    turn = Fraction(1, 4)
    for orientation, before_sign in (("rising", -1), ("setting", 1)):
        root = replace(_d36_root(turn, turn), orientation=orientation)
        result, _ = _d36_direct_case(monkeypatch, operational_roots=((root,),))
        scan, witness = result["_d36_scan"], result["_d36_witness"]
        classify = solver._classify_operational_direct_piece
        for lo, hi, expected in (
            (turn - Fraction(1, 1000), turn, before_sign),
            (turn, turn + Fraction(1, 1000), -before_sign),
        ):
            args = {
                "scan": scan,
                "witness_owner": witness,
                "direction_index": 0,
                "lower": lo,
                "upper": hi,
            }
            assert classify(**args) == (
                "smooth_above" if expected > 0 else "smooth_below"
            )
            original = witness.at_pairs

            def contrary(indices: Any, turns: Any, sign: int = -expected) -> np.ndarray:
                return np.tile([0.0, 0.0, float(sign)], (len(turns), 1))

            monkeypatch.setattr(witness, "at_pairs", contrary)
            with pytest.raises(
                ValueError, match="witness contradicts crossing orientation"
            ):
                classify(**args)
            monkeypatch.setattr(witness, "at_pairs", original)
        assert scan.evaluator.evaluations == 7


def test_d36_finite_excluded_ambiguity_refuses_without_ungranted_cuts(
    monkeypatch: Any,
) -> None:
    import struct
    from dataclasses import replace
    from fractions import Fraction
    from importlib import import_module

    import pytest

    solver: Any = import_module("radiosim.core.mmode.solver")
    result, _ = _d36_direct_case(monkeypatch)
    scan, witness = result["_d36_scan"], result["_d36_witness"]
    endpoint = scan.evaluator._grid.horizon_domain[1]
    step = Fraction(1, 2**48)

    def text(v: Fraction) -> str:
        return f"{v.numerator}/{v.denominator}"

    zero = struct.pack(">d", 0.0).hex()
    event = {
        "direction_id": "point:0:0",
        "cell_index": 48,
        "turn_lo": text(endpoint),
        "turn_hi": text(endpoint),
        "classification": "excluded_upper_endpoint",
        "f_lo_f64be": zero,
        "f_hi_f64be": zero,
        "ceiling_margin_f64be": zero,
        "left_sign": 1,
        "right_sign": -1,
        "root_turn_lo": text(endpoint),
        "root_turn_hi": text(endpoint),
        "root_orientation": "setting",
        "root_residual_f64be": zero,
    }
    amended = replace(
        scan,
        crossing_rows=(event,),
        summary_rows=({**scan.summary_rows[0], "crossing_count": 1},),
    )
    args = {
        "witness_owner": witness,
        "direction_index": 0,
        "lower": endpoint - 2 * step,
        "upper": endpoint,
    }
    assert (
        solver._classify_operational_direct_piece(scan=amended, **args)
        == "smooth_above"
    )
    finite_event = {
        **event,
        "turn_lo": text(endpoint - step),
        "root_turn_lo": text(endpoint - step),
    }
    finite = replace(amended, crossing_rows=(finite_event,))
    with pytest.raises(
        ValueError, match="positive excluded ambiguity is unpartitioned"
    ):
        solver._classify_operational_direct_piece(scan=finite, **args)
    guard = {
        **finite_event,
        "cell_index": 47,
        "turn_lo": text(endpoint - 2 * step),
        "turn_hi": text(endpoint - step),
        "classification": "guard_interval",
        "left_sign": 1,
        "right_sign": 0,
        "f_lo_f64be": struct.pack(">d", 1.0).hex(),
        "root_turn_lo": None,
        "root_turn_hi": None,
        "root_orientation": None,
        "root_residual_f64be": None,
    }
    guarded = replace(finite, crossing_rows=(guard, finite_event), guard_count=1)
    with pytest.raises(
        ValueError, match="positive excluded ambiguity is unpartitioned"
    ):
        solver._classify_operational_direct_piece(scan=guarded, **args)
    assert (
        solver._classify_operational_direct_piece(
            scan=guarded,
            **{**args, "lower": endpoint - 3 * step, "upper": endpoint - 2 * step},
        )
        == "smooth_above"
    )


def test_d36_middle_enclosure_in_alternating_three_event_census(
    monkeypatch: Any,
) -> None:
    """Gap signs may differ across an enclosure that contains a setting event."""
    from dataclasses import replace
    from fractions import Fraction
    from importlib import import_module

    solver: Any = import_module("radiosim.core.mmode.solver")
    width = Fraction(1, 2**49)
    roots = tuple(
        replace(_d36_root(turn, turn + width), orientation=orientation)
        for turn, orientation in (
            (Fraction(1, 4), "rising"),
            (Fraction(1, 2), "setting"),
            (Fraction(3, 4), "rising"),
        )
    )
    result, _ = _d36_direct_case(monkeypatch, operational_roots=(roots,))
    scan, witness = result["_d36_scan"], result["_d36_witness"]
    middle = roots[1]
    projected = _d36_rows(result, middle.turn_lo, middle.turn_hi)
    assert len(projected) == 1
    assert projected[0]["operational_piece_class"] == "root_enclosure"
    assert projected[0]["operational_gauss128_node_count"] == 0
    state = (
        scan.evaluator.evaluations,
        tuple(scan.evaluator.per_direction),
        scan.manifest(),
    )

    def forbidden_witness(indices: Any, turns: Any) -> np.ndarray:
        raise AssertionError(
            "an authenticated ambiguous piece must not use a gap witness"
        )

    monkeypatch.setattr(witness, "at_pairs", forbidden_witness)
    assert (
        solver._classify_operational_direct_piece(
            scan=scan,
            witness_owner=witness,
            direction_index=0,
            lower=middle.turn_lo,
            upper=middle.turn_hi,
        )
        == "root_enclosure"
    )
    assert (
        scan.evaluator.evaluations,
        tuple(scan.evaluator.per_direction),
        scan.manifest(),
    ) == state
