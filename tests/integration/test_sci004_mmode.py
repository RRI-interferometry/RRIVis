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
always ``radiosim.stokes-ne-theta-phi.v1``; neither field is nullable.

Section 7.3's authoritative truncation gate is not a fixture-only diagnostic: it
runs on **every production run**, comparing ``V0`` with the complete final
128-node horizon-split frozen-frame direct cube ``F128`` and its root-enclosure
error cube ``EF`` already retained for that run's Section 4.2 certificate, under

``max(U_direct) <= 1e-8*S_direct + 1e-10 Jy`` and
``norm(U_direct)/max(norm(F128), sqrt(K)*1 Jy) <= 1e-8``

*before any result or output path is created*. The local shells are attribution
diagnostics and are explicitly not a correctness bound.

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

#: Section 10's exact m-mode solver snapshot key set.
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

#: Section 7.3's fixed all-run direct-gate limits.
DIRECT_MAX_RELATIVE_LIMIT = 1e-8
DIRECT_ABSOLUTE_FLOOR_JY = 1e-10
DIRECT_L2_LIMIT = 1e-8

SIDEREAL_SAMPLES = 33
LMAX = 8
MMAX = 8
QUADRATURE_NSIDE = 8

_SCALAR_RUN_FIXTURE = f"""\
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
        "m1.integration.every-run-direct-gate",
        "sci004.section-7.3.all-run-complete-frozen-direct-gate",
        "test_every_production_run_executes_the_complete_frozen_direct_gate",
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
    """Deep-merge the exact fixture override into the shared valid mapping."""
    import yaml

    from tests.fixtures.configs import valid_config_mapping

    override = yaml.safe_load(_SCALAR_RUN_FIXTURE.decode("utf-8"))
    mapping = valid_config_mapping(tmp_path)
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


def test_every_production_run_executes_the_complete_frozen_direct_gate(
    tmp_path,
) -> None:
    """Section 7.3: the all-run gate, with its fixed limits and full coverage."""
    result = _run(tmp_path)
    gate = result.solver.direct_gate

    expected_cells = (
        SIDEREAL_SAMPLES
        * len(result.selection.baselines)
        * len(result.frequencies_hz)
        * 4
    )
    assert gate.expected_cell_count == expected_cells
    assert gate.compared_finite_cell_count == expected_cells
    assert gate.evaluated_error_cell_count == expected_cells
    assert gate.maximum_absolute_limit_jy == (
        DIRECT_MAX_RELATIVE_LIMIT * gate.reference_scale_jy + DIRECT_ABSOLUTE_FLOOR_JY
    )
    assert gate.normalized_l2_limit == DIRECT_L2_LIMIT
    assert gate.maximum_absolute_deviation_jy <= gate.maximum_absolute_limit_jy
    assert gate.normalized_l2 <= gate.normalized_l2_limit
    assert gate.pass_ is True
    # The gate reference is the certificate's own retained cube, not a recompute.
    assert gate.reference_cube_sha256 == result.solver.frozen_gauss128_cube_sha256
    assert gate.reference_error_cube_sha256 == (
        result.solver.frozen_enclosure_error_cube_sha256
    )


def test_the_mmode_solver_snapshot_carries_its_exact_tagged_fields(tmp_path) -> None:
    """Section 10: the exact tagged snapshot key set, with no nullable field."""
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
