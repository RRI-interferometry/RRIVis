"""SCI-004 phase-M1 end-to-end red oracles for the m-mode forward simulator.

``docs/development/sci004_mmode_design.md`` Section 10 keeps the public result
exactly one ``(T, B, F, 2, 2)`` receptor cube with the existing four correlation
labels; point, HEALPix and hybrid remain *solver provenance*, not separate output
products. M1 widens the solver record to a strict tagged union in which the
current ``rime`` snapshot stays byte-identical and an m-mode snapshot carries the
exact common fields ``solver``, ``sky_representation``, ``convention``,
``execution_path``, ``components`` and ``component_element_counts``, followed by
the exact m-mode block this module pins. In M1 ``tangent_polarization_frame`` is
the exact literal ``not_applicable_scalar_m1`` and ``stokes_v_basis_bridge`` is
always ``radiosim.stokes-ne-theta-phi.v1``; neither field is nullable. That
twenty-key set is **unchanged** by the two-tier gate: the measured deficit enters
the result's *provenance record*, never the Section 10 snapshot.

**The Section 7.3 two-tier gate.** It executes on every production run before any
result or output path is created, and it replaces the withdrawn single ``1e-8``
direct-equality predicate, which is mathematically unattainable for this forward
model: the transfer kernel carries the strict horizon step, a band-limited
projection of a discontinuous kernel converges only algebraically, and for a
delta sky the forward product is exactly ``S*K_L(n_s)`` -- the band-limited kernel
sample -- so the deficit against the exact direct sum is a property of the
method, not an implementation defect.

*Tier 1a* is the sharp half. The complete harmonic pipeline is evaluated once
with the horizon factor replaced by ``H === 1`` and everything else identical --
same grids, beam, fringe, packing, contraction and synthesis -- on both the
production and ``qcheck`` quadratures, giving ``W0`` and ``W_q``. That integrand
is smooth, Gauss-Legendre is spectrally exact through the band, and both

``max(abs(W0-W_q)) <= 1e-8*S_num + 1e-10 Jy`` and
``norm(W0-W_q)/max(norm(W_q), sqrt(K)*1 Jy) <= 1e-8``

must hold, so any sign, normalization, weight, packing or dropped-mode defect in
the shared pipeline fails sharply. *Tier 1b* still computes the with-horizon
shell ``V_q`` and records its maximum and normalized L2 in provenance, bounded
per acceptance fixture by a reviewed ``quadrature_budget_jy`` that lives in the
phase evidence rather than here. *Tier 2* computes the truncation deficit
``U = abs(V0-F128) + EF`` at the convergence levels ``L1 = max(2, lmax//4)`` and
``L2 = max(L1+1, lmax//2)`` and at ``lmax``, and requires strict monotone
decrease with ``deficit_max(L1) >= 2 * deficit_max(lmax)``. The deficit is never
called agreement; its obligations are convergence and disclosure.

**The qualified fixture.** Section 7.3 requires an acceptance fixture to sit in
the convergent regime, whose governing conditions are geometric: every payload
direction must stay well clear of the horizon over the whole cycle, and ``lmax``
is pinned by the accepted evidence because the quarter-to-full factor is not
monotone in ``lmax``. This fixture was qualified by measurement, not by
argument. At the shipped site (latitude ``-30.72152``) a source pinned at
declination ``-75`` is circumpolar between ``+15.7`` and ``+45.7`` degrees
altitude, has **zero** frozen horizon roots, and therefore an exactly zero
enclosure-error cube. Its measured three-level deficit sequence is
``7.159e-1 -> 1.847e-1 -> 1.170e-1 Jy``: strict monotone with a quarter-to-full
factor of ``6.12`` against the required ``2``. That margin is stable -- ``5.0``
to ``6.5`` across declinations ``-74`` to ``-86``, ``6.06`` to ``6.12`` across
``N`` of 49 to 97, ``6.12`` to ``7.23`` across the production grid, and ``5.26``
to ``7.19`` across 45 to 55 MHz -- while the previously shipped transiting
geometry measured ``1.53`` and a larger ``lmax`` measured ``1.65``. The
horizon-free shell of the same pipeline measures at the ``1e-10`` level.

This module is the M1 end-to-end red slice for a scalar (Stokes-I point source)
``full_sidereal`` run through the public :class:`radiosim.api.Simulator` API. It
is red at ``G1`` because ``execution.simulator: mmode`` and
``obs_time.mode: full_sidereal`` are not accepted configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

MMODE_CONVENTION = "radiosim.mmode-forward.v1"
MMODE_FRAME_MODEL = "radiosim.frozen-cirs-rigid-era.v1"
MMODE_HARMONIC_CONVENTION = "radiosim.shaw-polarized-harmonics.v1"
MMODE_TIME_GRID_CONVENTION = "radiosim.mmode-era-turn-grid.v1"
MMODE_TANGENT_FRAME_M1 = "not_applicable_scalar_m1"
MMODE_STOKES_BRIDGE = "radiosim.stokes-ne-theta-phi.v1"
MMODE_QUADRATURE_POLICY = "iso-gauss-ring-production-plus-qcheck.v1"
MMODE_TRUNCATION_POLICY = "complete-frozen-direct-plus-local-shells.v1"
MMODE_EXECUTION_POLICY = "host_harmonics_backend_native_dense_v1"

#: Section 10's exact m-mode solver snapshot key set. The two-tier gate does not
#: widen it: Section 7.3 puts the deficit in the provenance record instead.
MMODE_SNAPSHOT_KEYS: tuple[str, ...] = (
    "solver",
    "sky_representation",
    "convention",
    "execution_path",
    "components",
    "component_element_counts",
    "time_grid_convention",
    "frame_model",
    "harmonic_convention",
    "sidereal_samples",
    "lmax",
    "mmax",
    "quadrature_nside",
    "quadrature_policy",
    "truncation_policy",
    "tangent_polarization_frame",
    "stokes_v_basis_bridge",
    "iers_table_sha256",
    "frame_certificate_sha256",
    "transform_execution_policy",
)

#: Section 11's exact ``sci004_two_tier_direct.v3`` field surface, in order.
TWO_TIER_GATE_KEYS: tuple[str, ...] = (
    "predicate_id",
    "reference_cube_sha256",
    "candidate_cube_sha256",
    "reference_error_cube_sha256",
    "horizon_free_cube_sha256",
    "horizon_free_qcheck_cube_sha256",
    "quadrature_shell_cube_sha256",
    "expected_cell_count",
    "compared_finite_cell_count",
    "evaluated_error_cell_count",
    "numerical_scale_jy",
    "horizon_free_shell_max_jy",
    "horizon_free_shell_l2",
    "horizon_free_shell_max_limit_jy",
    "horizon_free_shell_l2_limit",
    "quadrature_shell_max_jy",
    "quadrature_shell_l2",
    "reference_scale_jy",
    "deficit_max_jy",
    "deficit_l2",
    "deficit_max_quarter_jy",
    "deficit_max_half_jy",
    "convergence_factor",
    "pass",
)

TWO_TIER_PREDICATE_ID = "sci004_two_tier_direct.v3"

#: Section 7.3's fixed tier-1a limits and tier-2 convergence floor.
HORIZON_FREE_RELATIVE_LIMIT = 1e-8
HORIZON_FREE_ABSOLUTE_FLOOR_JY = 1e-10
HORIZON_FREE_L2_LIMIT = 1e-8
CONVERGENCE_FACTOR_FLOOR = 2.0

#: The qualified fixture's dimensions. ``lmax`` is pinned by the accepted
#: evidence, not chosen for convenience: Section 7.3 records that the
#: quarter-to-full factor collapses once ``L1`` itself resolves the smooth
#: kernel, which the measured ``lmax = 20`` and ``24`` variants confirm.
SIDEREAL_SAMPLES = 49
LMAX = 16
MMAX = 16
QUADRATURE_NSIDE = 8

#: Section 7.3's derived convergence levels for this ``lmax``.
CONVERGENCE_LEVEL_QUARTER = max(2, LMAX // 4)
CONVERGENCE_LEVEL_HALF = max(CONVERGENCE_LEVEL_QUARTER + 1, LMAX // 2)

#: The qualified compact geometry: a 4.0 m east-west baseline between two 2.5 m
#: dishes at 50/51/52 MHz. ``b > D`` so the two antennas do not overlap.
BASELINE_EAST_M = 4.0
DISH_DIAMETER_M = 2.5
STARTING_FREQUENCY_MHZ = 50.0

#: The circumpolar declination that keeps every payload direction clear of the
#: horizon over the whole cycle, and the flux the retained loader produces.
SOURCE_DEC_DEG = -75.0
MEASURED_SOURCE_FLUX_JY = 5.5

_SCALAR_RUN_FIXTURE = f"""\
_antenna_layout:
  baseline_east_m: {BASELINE_EAST_M}
  diameter_m: {DISH_DIAMETER_M}
instrument:
  default_diameter_m: {DISH_DIAMETER_M}
execution:
  simulator: mmode
  mmode:
    convention: {MMODE_CONVENTION}
    frame_model: {MMODE_FRAME_MODEL}
    harmonic_convention: {MMODE_HARMONIC_CONVENTION}
    lmax: {LMAX}
    mmax: {MMAX}
    quadrature_nside: {QUADRATURE_NSIDE}
    working_memory_bytes: 1073741824
obs_frequency:
  mode: grid
  starting_frequency: {STARTING_FREQUENCY_MHZ}
  frequency_interval: 1.0
  frequency_bandwidth: 2.0
  channel_width: 1.0
  frequency_unit: MHz
obs_time:
  mode: full_sidereal
  start_time: "2025-01-01T00:00:00"
  sidereal_samples: {SIDEREAL_SAMPLES}
  integration_fraction: 1.0
sky_model:
  flux_unit: Jy
  sources:
    - kind: test_sources
      representation: point_sources
      num_sources: 1
      distribution: uniform
      seed: 1
      dec_deg: {SOURCE_DEC_DEG}
      dec_range_deg: 0.0
      spectral_index: 0.0
      polarization_fraction: 0.0
      stokes_v_fraction: 0.0
""".encode()

_GREEN_CONTROL = (
    "tests/integration/test_sci004_mmode.py::"
    "test_the_public_api_runs_the_direct_strategy_end_to_end_today"
)


def _case(
    case_id: str,
    requirement_id: str,
    function: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": f"tests/integration/test_sci004_mmode.py::{function}",
        "expected_failure_kind": "schema",
        "expected_failure_pattern": (
            r"ConfigSchemaError: execution\.mmode: unknown or removed field"
        ),
        "fixture_defect_excluded_by": _GREEN_CONTROL,
        "fixture_bytes": _SCALAR_RUN_FIXTURE,
    }


SCI004_RED_CASES: tuple[dict[str, Any], ...] = (
    _case(
        "m1.integration.scalar-receptor-cube",
        "sci004.section-10.public-result-stays-one-receptor-cube",
        "test_a_scalar_full_sidereal_mmode_run_produces_the_receptor_cube",
    ),
    _case(
        "m1.integration.every-run-two-tier-gate",
        "sci004.section-7.3.all-run-two-tier-gate",
        "test_every_production_run_executes_the_two_tier_gate",
    ),
    _case(
        "m1.integration.solver-snapshot",
        "sci004.section-10.tagged-mmode-solver-snapshot",
        "test_the_mmode_solver_snapshot_carries_its_exact_tagged_fields",
    ),
)

SCI004_RED_GREEN_CONTROLS: tuple[str, ...] = (_GREEN_CONTROL,)


# --- harness ------------------------------------------------------------------


def _mmode_mapping(tmp_path: Path) -> dict[str, Any]:
    """Deep-merge the exact fixture override into the shared valid mapping.

    A key whose name begins with ``_`` is a fixture-local materialization
    directive consumed here, never configuration: ``_antenna_layout`` rewrites
    the shared two-antenna file to the qualified compact geometry, in place, so
    the resolved ``instrument.source.path`` stays the one the base mapping
    already points at and the retained fixture bytes stay reproducible.
    """
    import yaml

    from tests.fixtures.configs import valid_config_mapping

    override = yaml.safe_load(_SCALAR_RUN_FIXTURE.decode("utf-8"))
    mapping = valid_config_mapping(tmp_path)

    layout = override.pop("_antenna_layout")
    Path(mapping["instrument"]["source"]["path"]).write_text(
        "Name Number BeamID E N U Diameter\n"
        f"ANT0 0 0 0.0 0.0 0.0 {layout['diameter_m']}\n"
        f"ANT1 1 0 {layout['baseline_east_m']} 0.0 0.0 {layout['diameter_m']}\n"
    )

    mapping["instrument"] = {**mapping["instrument"], **override["instrument"]}
    mapping["obs_frequency"] = override["obs_frequency"]
    mapping["obs_time"] = override["obs_time"]
    mapping["sky_model"] = override["sky_model"]
    mapping["execution"] = {**mapping["execution"], **override["execution"]}
    return mapping


def _run(tmp_path: Path) -> Any:
    from radiosim.api import Simulator

    return Simulator.from_mapping(_mmode_mapping(tmp_path), base_dir=tmp_path).run(
        progress=False
    )


# --- green control ------------------------------------------------------------


def test_the_public_api_runs_the_direct_strategy_end_to_end_today(tmp_path) -> None:
    """The public API harness the red nodes use is sound at ``G1``.

    Section 10 keeps the result unchanged, so the same assertions the m-mode red
    node makes already hold for ``rime``. The published cube is the four
    row-major correlation labels, which *are* the flattened receptor cube of
    Section 6's ``(time, baseline, frequency, receptor-row, receptor-column)``
    axes; reading it back in that view is what Section 10's ``(T, B, F, 2, 2)``
    sentence names. That is what excludes a defective fixture: the red failures
    below are the absence of the m-mode configuration surface, not a broken run
    harness.
    """
    import numpy as np

    from radiosim.api import Simulator
    from tests.fixtures.configs import valid_config_mapping

    result = Simulator.from_mapping(
        valid_config_mapping(tmp_path), base_dir=tmp_path
    ).run(progress=False)
    cube = np.asarray(result.visibilities)

    assert cube.ndim == 4
    assert cube.shape[-1] == 4
    assert cube.reshape(*cube.shape[:3], 2, 2).shape[-2:] == (2, 2)
    assert str(cube.dtype) == "complex128"
    assert result.solver.solver == "rime"
    assert len(result.correlations) == 4


# --- Section 7.3 / 10 red oracles ---------------------------------------------


def test_a_scalar_full_sidereal_mmode_run_produces_the_receptor_cube(
    tmp_path,
) -> None:
    """Section 10: one ``(T, B, F, 2, 2)`` receptor cube, four correlation labels."""
    import numpy as np

    result = _run(tmp_path)
    cube = np.asarray(result.visibilities)
    receptor_cube = cube.reshape(*cube.shape[:3], 2, 2)

    assert cube.shape == (
        SIDEREAL_SAMPLES,
        len(result.selection.baselines),
        len(result.frequencies_hz),
        4,
    )
    assert receptor_cube.shape[-2:] == (2, 2)
    assert str(cube.dtype) == "complex128"
    assert len(result.correlations) == 4
    assert result.solver.solver == "mmode"
    assert float(np.max(np.abs(cube))) > 0.0


def test_every_production_run_executes_the_two_tier_gate(tmp_path) -> None:
    """Section 7.3/11: the every-run two-tier gate on its ``v3`` field surface.

    Tier 1a is the only half with a fixed numeric limit. Tier 1b is recorded and
    carries no universal limit -- its budget is a reviewed phase-evidence field,
    not a predicate here -- and tier 2 gates on convergence, never on equality.
    """
    import math

    result = _run(tmp_path)
    gate = result.solver.direct_gate
    record = gate.as_mapping()

    assert tuple(record) == TWO_TIER_GATE_KEYS
    assert record["predicate_id"] == TWO_TIER_PREDICATE_ID

    expected_cells = (
        SIDEREAL_SAMPLES
        * len(result.selection.baselines)
        * len(result.frequencies_hz)
        * 4
    )
    assert record["expected_cell_count"] == expected_cells
    assert record["compared_finite_cell_count"] == expected_cells
    assert record["evaluated_error_cell_count"] == expected_cells

    # Tier 1a -- the horizon-free shell, gating at 1e-8 against S_num.
    assert record["horizon_free_shell_max_limit_jy"] == (
        HORIZON_FREE_RELATIVE_LIMIT * record["numerical_scale_jy"]
        + HORIZON_FREE_ABSOLUTE_FLOOR_JY
    )
    assert record["horizon_free_shell_l2_limit"] == HORIZON_FREE_L2_LIMIT
    assert (
        record["horizon_free_shell_max_jy"] <= record["horizon_free_shell_max_limit_jy"]
    )
    assert record["horizon_free_shell_l2"] <= record["horizon_free_shell_l2_limit"]
    assert (
        record["horizon_free_cube_sha256"] != record["horizon_free_qcheck_cube_sha256"]
    )

    # Tier 1b -- the with-horizon shell is recorded, and deliberately unbounded
    # here: the strict horizon step makes no finite quadrature exact, so this
    # value carries a reviewed per-fixture budget in the phase evidence instead.
    assert math.isfinite(record["quadrature_shell_max_jy"])
    assert math.isfinite(record["quadrature_shell_l2"])
    assert record["quadrature_shell_max_jy"] >= 0.0
    assert record["quadrature_shell_cube_sha256"] != record["candidate_cube_sha256"]

    # Tier 2 -- strict monotone decrease and the quarter-to-full factor.
    assert (
        record["deficit_max_quarter_jy"]
        > record["deficit_max_half_jy"]
        > record["deficit_max_jy"]
    )
    assert record["convergence_factor"] == (
        record["deficit_max_quarter_jy"] / record["deficit_max_jy"]
    )
    assert record["convergence_factor"] >= CONVERGENCE_FACTOR_FLOOR
    assert record["deficit_l2"] >= 0.0

    assert record["pass"] is True
    # The tier-2 reference is the certificate's own retained cube, not a
    # recompute, and the tier-1a cubes are internals that never become a result.
    assert record["reference_cube_sha256"] == (
        result.solver.frozen_gauss128_cube_sha256
    )
    assert record["reference_error_cube_sha256"] == (
        result.solver.frozen_enclosure_error_cube_sha256
    )


def test_the_mmode_solver_snapshot_carries_its_exact_tagged_fields(tmp_path) -> None:
    """Section 10: the exact tagged snapshot key set, with no nullable field.

    Section 7.3 is explicit that the measured deficit enters the result's
    *provenance record* and not this key set, so the twenty keys below are
    unchanged by the two-tier gate and the deficit is asserted absent from them.
    """
    result = _run(tmp_path)
    snapshot = result.solver.as_mapping()

    assert tuple(snapshot) == MMODE_SNAPSHOT_KEYS
    assert snapshot["solver"] == "mmode"
    assert snapshot["convention"] == MMODE_CONVENTION
    assert snapshot["frame_model"] == MMODE_FRAME_MODEL
    assert snapshot["harmonic_convention"] == MMODE_HARMONIC_CONVENTION
    assert snapshot["time_grid_convention"] == MMODE_TIME_GRID_CONVENTION
    assert snapshot["quadrature_policy"] == MMODE_QUADRATURE_POLICY
    assert snapshot["truncation_policy"] == MMODE_TRUNCATION_POLICY
    assert snapshot["tangent_polarization_frame"] == MMODE_TANGENT_FRAME_M1
    assert snapshot["stokes_v_basis_bridge"] == MMODE_STOKES_BRIDGE
    assert snapshot["transform_execution_policy"] == MMODE_EXECUTION_POLICY
    assert snapshot["sidereal_samples"] == SIDEREAL_SAMPLES
    assert (snapshot["lmax"], snapshot["mmax"]) == (LMAX, MMAX)
    assert snapshot["quadrature_nside"] == QUADRATURE_NSIDE
    assert all(value is not None for value in snapshot.values())
    for absent in ("deficit_max_jy", "deficit_l2", "direct_gate"):
        assert absent not in snapshot
