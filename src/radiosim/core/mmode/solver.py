r"""The m-mode forward solve, its frozen direct oracle, and the all-run gate.

``docs/development/sci004_mmode_design.md`` Section 6 gives the forward
per-``m`` product

.. math::

    v_{pqfc,m}=\sum_l\left[
    B^I_{pqfc,lm}a^I_{f,lm}
    +\tfrac12B^{(+2)}_{pqfc,lm}a^{(+2)}_{f,lm}
    +\tfrac12B^{(-2)}_{pqfc,lm}a^{(-2)}_{f,lm}
    +B^V_{pqfc,lm}a^V_{f,lm}\right],

of which phase M1 evaluates the scalar ``I`` term only, and the exposure-averaged
synthesis

.. math::

    \bar V_k=\sum_{m=-m_{max}}^{m_{max}}w_m v_m e^{+i2\pi m u_k},
    \qquad w_m=\operatorname{sinc}(\pi m\,\Delta u).

Section 7.3's authoritative truncation gate is not a fixture-only diagnostic: on
**every production run** the synthesized cube ``V0`` is compared with the
complete final 128-node horizon-split frozen-frame direct cube ``F128`` and its
root-enclosure error cube ``EF``, *before any result or output path is created*,
under

.. math::

    \max(U)\le 10^{-8}S+10^{-10}\,\mathrm{Jy},\qquad
    \frac{\|U\|}{\max(\|F_{128}\|,\sqrt K\,\mathrm{Jy})}\le 10^{-8},

with ``U = abs(V0 - F128) + EF`` and ``S = max(1 Jy, max(abs(F128) + EF))``.

Section 4.2 places one more mandatory step ahead of all of it: an authenticated
``FrameApplicabilityCertificate``, computed in memory for every solve, before
harmonic work, with the NumPy direct oracle.  It is not a user-supplied path, a
YAML bypass or a cache lookup, and it has no waiver.  Its frozen census is the
exact-rational analytic construction in :mod:`radiosim.core.mmode.frame`; its
operational census is the Section 12.1 certified-ceiling scan in the same
module, complete via the design-frozen ``L_op`` derivative ceiling and consuming
only the public Astropy transform.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from radiosim.core.mmode.frame import (
    FrozenFrame,
    FrozenHorizonTrajectory,
    HorizonRootEnclosure,
    build_frozen_frame,
    frozen_horizon_trajectory,
    scan_operational_horizon,
    strict_horizon_indicator,
    strict_horizon_visible,
)
from radiosim.core.mmode.time import CanonicalEraGrid
from radiosim.core.mmode.types import (
    CONVENTION_IDENTITY,
    FIELD_ORDER,
    MMODE_CONVENTION,
    MMODE_EXECUTION_POLICY,
    MMODE_FRAME_MODEL,
    MMODE_HARMONIC_CONVENTION,
    MMODE_QUADRATURE_POLICY,
    MMODE_STOKES_BRIDGE,
    MMODE_TANGENT_FRAME_M1,
    MMODE_TIME_GRID_CONVENTION,
    MMODE_TRUNCATION_POLICY,
    STOKES_COMPONENT_ORDER,
    TAU,
    MModeDimensions,
    array_digest,
    canonical_json,
    canonical_rational,
    decode_f64be,
    domain_digest,
    f64be,
    object_digest,
    streamed_domain_digest,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from radiosim.simulator.base import SkySolveOutcome, SkySolveRequest

__all__ = [
    "CONVERGENCE_FACTOR_FLOOR",
    "HORIZON_FREE_ABSOLUTE_FLOOR_JY",
    "HORIZON_FREE_L2_LIMIT",
    "HORIZON_FREE_RELATIVE_LIMIT",
    "TWO_TIER_PREDICATE_ID",
    "DirectGateRecord",
    "FrameCertificate",
    "MModeFrameCertificateFailed",
    "LedgerDirection",
    "MModeSolverSnapshot",
    "build_direction_ledger",
    "build_frame_certificate",
    "build_input_identity",
    "build_m1_evidence",
    "transfer_sample_rows",
    "evaluate_two_tier_gate",
    "magnitude_ceiling",
    "solve_mmode",
    "PolarizedFixtureOutcome",
    "polarized_direct_cube",
    "solve_polarized_fixture",
    "BackendComplex128Resolution",
    "MModeBlockSchedule",
    "MModeMemoryEstimate",
    "SCHEDULE_DIGEST_DOMAIN",
    "SPIN_FIELD_WEIGHTS",
    "contract_per_m_block",
    "estimate_mmode_memory",
    "forward_per_m_product",
    "require_backend_complex128",
    "schedule_mmode_blocks",
    "synthesize_time_series",
]

#: Section 11's two-tier predicate identifier.
TWO_TIER_PREDICATE_ID: Final = "sci004_two_tier_direct.v3"

#: Section 7.3's fixed tier-1a limits and tier-2 convergence floor.  New
#: SCI-004 bounds; no existing tolerance is changed and none may be widened.
HORIZON_FREE_RELATIVE_LIMIT: Final[float] = 1e-8
HORIZON_FREE_ABSOLUTE_FLOOR_JY: Final[float] = 1e-10
HORIZON_FREE_L2_LIMIT: Final[float] = 1e-8
CONVERGENCE_FACTOR_FLOOR: Final[float] = 2.0

#: Section 4.2's fixed frame budgets.
FRAME_PHASE_LIMIT_RAD: Final[float] = 5e-3
FRAME_ROOT_LIMIT_RAD: Final[float] = 2e-5
FRAME_CUBE_RELATIVE_LIMIT: Final[float] = 5e-5
FRAME_CUBE_FLOOR_JY: Final[float] = 1e-10
FRAME_GAUSS_CHANGE_RELATIVE_LIMIT: Final[float] = 1e-11

#: Section 8's frame-certificate rejection template, rendered as exactly one
#: line with lower-case scientific notation.
FRAME_CERTIFICATE_FAILURE_TEMPLATE: Final = (
    "execution.simulator='mmode' frame certificate failed: "
    "phase_max={phase:.6e} rad (limit=5.000000e-03 rad); "
    "horizon_root_count_mismatches={root_count:d}; "
    "horizon_root_orientation_mismatches={orientation:d}; "
    "horizon_membership_mismatches={membership:d}; "
    "horizon_outside_slab_sign_mismatches={outside_sign:d}; "
    "horizon_unresolved_intervals={unresolved:d}; "
    "horizon_root_max={root_max:.6e} rad (limit=2.000000e-05 rad); "
    "horizon_mismatch_measure={mismatch_measure:.6e} rad "
    "(limit={mismatch_limit:.6e} rad); "
    "cube_max={cube_max:.6e} Jy (limit={cube_limit:.6e} Jy); "
    "cube_l2={cube_l2:.6e} (limit=5.000000e-05)."
)


# ---------------------------------------------------------------------------
# Section 10 tagged solver snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DirectGateRecord:
    """Section 7.3/11's ``sci004_two_tier_direct.v3`` every-run gate record.

    Tier 1a is the sharp half: the same pipeline evaluated with the horizon
    factor replaced by ``H === 1`` on the production and ``qcheck`` quadratures
    gives ``W0`` and ``W_q``, whose integrand is smooth, so Gauss-Legendre is
    spectrally exact through the band and any sign, normalization, weight,
    packing or dropped-mode defect in the shared pipeline fails at ``1e-8``.

    Tier 1b records the with-horizon shell.  It carries **no universal limit**:
    the strict horizon step makes no finite quadrature exact, so its bound is a
    reviewed per-fixture ``quadrature_budget_jy`` in the phase evidence.

    Tier 2 is the attributed comparison.  The deficit is never called agreement
    -- its obligations are convergence and disclosure -- and the reference is
    the certificate's own retained final 128-node frozen direct cube and its
    root-enclosure error cube, never an alternate recomputation or a subset.
    """

    predicate_id: str
    reference_cube_sha256: str
    candidate_cube_sha256: str
    reference_error_cube_sha256: str
    horizon_free_cube_sha256: str
    horizon_free_qcheck_cube_sha256: str
    quadrature_shell_cube_sha256: str
    expected_cell_count: int
    compared_finite_cell_count: int
    evaluated_error_cell_count: int
    numerical_scale_jy: float
    horizon_free_shell_max_jy: float
    horizon_free_shell_l2: float
    horizon_free_shell_max_limit_jy: float
    horizon_free_shell_l2_limit: float
    quadrature_shell_max_jy: float
    quadrature_shell_l2: float
    reference_scale_jy: float
    deficit_max_jy: float
    deficit_l2: float
    deficit_max_quarter_jy: float
    deficit_max_half_jy: float
    convergence_factor: float
    pass_: bool

    def as_mapping(self) -> dict[str, Any]:
        """Return Section 11's exact ``direct_comparison`` key order."""
        return {
            "predicate_id": self.predicate_id,
            "reference_cube_sha256": self.reference_cube_sha256,
            "candidate_cube_sha256": self.candidate_cube_sha256,
            "reference_error_cube_sha256": self.reference_error_cube_sha256,
            "horizon_free_cube_sha256": self.horizon_free_cube_sha256,
            "horizon_free_qcheck_cube_sha256": self.horizon_free_qcheck_cube_sha256,
            "quadrature_shell_cube_sha256": self.quadrature_shell_cube_sha256,
            "expected_cell_count": self.expected_cell_count,
            "compared_finite_cell_count": self.compared_finite_cell_count,
            "evaluated_error_cell_count": self.evaluated_error_cell_count,
            "numerical_scale_jy": self.numerical_scale_jy,
            "horizon_free_shell_max_jy": self.horizon_free_shell_max_jy,
            "horizon_free_shell_l2": self.horizon_free_shell_l2,
            "horizon_free_shell_max_limit_jy": self.horizon_free_shell_max_limit_jy,
            "horizon_free_shell_l2_limit": self.horizon_free_shell_l2_limit,
            "quadrature_shell_max_jy": self.quadrature_shell_max_jy,
            "quadrature_shell_l2": self.quadrature_shell_l2,
            "reference_scale_jy": self.reference_scale_jy,
            "deficit_max_jy": self.deficit_max_jy,
            "deficit_l2": self.deficit_l2,
            "deficit_max_quarter_jy": self.deficit_max_quarter_jy,
            "deficit_max_half_jy": self.deficit_max_half_jy,
            "convergence_factor": self.convergence_factor,
            "pass": self.pass_,
        }


@dataclass(frozen=True, slots=True)
class MModeSolverSnapshot:
    """Section 10's m-mode arm of the strict tagged solver record.

    The ``rime`` snapshot is unchanged by the union, and this arm carries the
    exact common fields followed by the exact m-mode block.  Neither
    ``tangent_polarization_frame`` nor ``stokes_v_basis_bridge`` is nullable:
    the second is always ``radiosim.stokes-ne-theta-phi.v1``, and the first is
    the exact six-key Section 5.1 object for a run whose sky carries linear
    polarization, or the exact literal ``not_applicable_scalar_m1`` for a run
    that carries none -- Section 5.1's "an ``I``/``V``-only payload may omit the
    tangent block", read at the snapshot boundary.
    """

    sky_representation: str
    execution_path: str
    components: tuple[str, ...]
    component_element_counts: tuple[int, ...]
    sidereal_samples: int
    lmax: int
    mmax: int
    quadrature_nside: int
    iers_table_sha256: str
    frame_certificate_sha256: str
    direct_gate: DirectGateRecord
    frozen_gauss128_cube_sha256: str
    frozen_enclosure_error_cube_sha256: str
    #: Either the exact six-key Section 5.1 object or the ``M1`` literal.
    tangent_polarization_frame: Any = MMODE_TANGENT_FRAME_M1

    @property
    def solver(self) -> str:
        """Return the registry key this snapshot records."""
        return "mmode"

    @property
    def convention(self) -> str:
        """Return the m-mode execution convention literal."""
        return MMODE_CONVENTION

    def as_mapping(self) -> dict[str, Any]:
        """Return Section 10's exact tagged snapshot key set, in order."""
        return {
            "solver": "mmode",
            "sky_representation": self.sky_representation,
            "convention": MMODE_CONVENTION,
            "execution_path": self.execution_path,
            "components": list(self.components),
            "component_element_counts": list(self.component_element_counts),
            "time_grid_convention": MMODE_TIME_GRID_CONVENTION,
            "frame_model": MMODE_FRAME_MODEL,
            "harmonic_convention": MMODE_HARMONIC_CONVENTION,
            "sidereal_samples": self.sidereal_samples,
            "lmax": self.lmax,
            "mmax": self.mmax,
            "quadrature_nside": self.quadrature_nside,
            "quadrature_policy": MMODE_QUADRATURE_POLICY,
            "truncation_policy": MMODE_TRUNCATION_POLICY,
            "tangent_polarization_frame": self.tangent_polarization_frame,
            "stokes_v_basis_bridge": MMODE_STOKES_BRIDGE,
            "iers_table_sha256": self.iers_table_sha256,
            "frame_certificate_sha256": self.frame_certificate_sha256,
            "transform_execution_policy": MMODE_EXECUTION_POLICY,
        }

    def to_snapshot(self) -> dict[str, Any]:
        """Return the scientific snapshot the result fingerprint hashes."""
        return self.as_mapping()

    def solver_snapshot_sha256(self) -> str:
        """Return ``D("radiosim.mmode-solver-snapshot.v1", J(snapshot))``."""
        return object_digest("radiosim.mmode-solver-snapshot.v1", self.as_mapping())


# ---------------------------------------------------------------------------
# Section 12.1 direction ledger (frozen half)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DirectionLedgerRow:
    """One canonical Section 12.1 direction-ledger row."""

    direction_id: str
    source_kind: str
    component_index: int
    source_index: int
    transfer_role: str
    transfer_nside: int
    cirs_direction: np.ndarray
    active_frequency_mask: tuple[bool, ...]
    resolved_stokes_iau: np.ndarray
    integration_weight: float

    @property
    def active_frequency_count(self) -> int:
        """Return the number of run frequencies this direction contributes at."""
        return int(sum(1 for flag in self.active_frequency_mask if flag))

    def cirs_direction_sha256(self) -> str:
        """Return the Section 12.1 array identity of the CIRS three-vector."""
        return array_digest(
            "radiosim.mmode-cirs-direction.v1",
            "cirs_direction",
            ["cartesian"],
            "dimensionless",
            np.asarray(self.cirs_direction, dtype=np.float64),
            dtype="float64-be",
        )

    def direction_input_manifest(self, run_frequency_hz: np.ndarray) -> dict[str, Any]:
        """Return the Section 12.1 ``direction_input`` preimage of this row."""
        return {
            "schema_version": "radiosim.mmode-direction-input.v1",
            "direction_id": self.direction_id,
            "source_kind": self.source_kind,
            "component_index": self.component_index,
            "source_index": self.source_index,
            "transfer_role": self.transfer_role,
            "transfer_nside": self.transfer_nside,
            "cirs_direction_f64be": [
                f64be(value) for value in np.asarray(self.cirs_direction).tolist()
            ],
            "run_frequency_hz_f64be": [
                f64be(value)
                for value in np.asarray(run_frequency_hz, dtype=np.float64).tolist()
            ],
            "active_frequency_mask": [
                bool(flag) for flag in self.active_frequency_mask
            ],
            "resolved_stokes_iau_f64be": [
                f64be(value)
                for value in np.asarray(self.resolved_stokes_iau, dtype=np.float64)
                .reshape(-1)
                .tolist()
            ],
            "integration_weight_f64be": f64be(self.integration_weight),
        }


# ---------------------------------------------------------------------------
# Section 12.1 frozen horizon-split direct oracle
# ---------------------------------------------------------------------------


def _gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return np.asarray(nodes, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def _classify_piece(
    trajectory: FrozenHorizonTrajectory, lower: Fraction, upper: Fraction
) -> str:
    """Classify one exposure piece as smooth above, smooth below, or ambiguous."""
    for root in trajectory.roots:
        if root.turn_lo < upper and lower < root.turn_hi:
            return "root_enclosure"
    value_range = trajectory.value_interval(lower, upper)
    if value_range[0] > 0.0:
        return "smooth_above"
    if value_range[1] <= 0.0:
        return "smooth_below"
    return "root_enclosure"


def evaluate_two_tier_gate(
    *,
    mmode_cube: np.ndarray,
    horizon_free_cube: np.ndarray,
    horizon_free_qcheck_cube: np.ndarray,
    quadrature_shell_cube: np.ndarray,
    frozen_gauss128: np.ndarray,
    frozen_enclosure_error: np.ndarray,
    deficit_max_quarter_jy: float,
    deficit_max_half_jy: float,
) -> DirectGateRecord:
    """Evaluate Section 7.3's every-run two-tier gate.

    It runs before any result or output path is created.  Tier 1a gates the
    horizon-free shell at ``1e-8``; tier 1b records the with-horizon shell,
    which carries no universal limit; tier 2 gates the truncation deficit on
    strict monotone decrease with ``deficit_max(L1) >= 2 * deficit_max(lmax)``.
    """
    candidate = np.asarray(mmode_cube, dtype=np.complex128)
    reference = np.asarray(frozen_gauss128, dtype=np.complex128)
    error = np.asarray(frozen_enclosure_error, dtype=np.float64)
    horizon_free = np.asarray(horizon_free_cube, dtype=np.complex128)
    horizon_free_qcheck = np.asarray(horizon_free_qcheck_cube, dtype=np.complex128)
    shell = np.asarray(quadrature_shell_cube, dtype=np.complex128)
    for cube in (reference, error, horizon_free, horizon_free_qcheck, shell):
        if cube.shape != candidate.shape:
            raise ValueError("the two-tier gate requires one common [N,B,F,4] shape")
    if candidate.ndim != 4 or candidate.shape[3] != 4:
        raise ValueError("every SCI-004 cube has exactly four correlations")
    for cube in (candidate, reference, horizon_free, horizon_free_qcheck, shell):
        if not np.all(np.isfinite(cube)):
            raise ValueError("the two-tier gate requires finite cubes")
    if not np.all(np.isfinite(error)) or np.any(error < 0.0):
        raise ValueError("the enclosure-error cube must be finite and non-negative")

    cells = int(candidate.size)
    root_cells = math.sqrt(cells)

    # -- Tier 1a: the horizon-free shell, gating at 1e-8 --------------------
    numerical_scale = max(1.0, float(np.max(np.abs(horizon_free_qcheck))))
    horizon_free_delta = horizon_free - horizon_free_qcheck
    horizon_free_max = float(np.max(np.abs(horizon_free_delta)))
    horizon_free_l2 = float(
        np.linalg.norm(horizon_free_delta.reshape(-1))
        / max(float(np.linalg.norm(horizon_free_qcheck.reshape(-1))), root_cells)
    )
    horizon_free_max_limit = (
        HORIZON_FREE_RELATIVE_LIMIT * numerical_scale + HORIZON_FREE_ABSOLUTE_FLOOR_JY
    )

    # -- Tier 1b: the with-horizon shell, recorded and fixture-budgeted -----
    shell_delta = candidate - shell
    shell_max = float(np.max(np.abs(shell_delta)))
    shell_l2 = float(
        np.linalg.norm(shell_delta.reshape(-1))
        / max(float(np.linalg.norm(shell.reshape(-1))), root_cells)
    )

    # -- Tier 2: the attributed truncation deficit --------------------------
    reference_scale = max(1.0, float(np.max(np.abs(reference) + error)))
    deficit = np.abs(candidate - reference) + error
    deficit_max = float(np.max(deficit))
    deficit_l2 = float(
        np.linalg.norm(deficit.reshape(-1))
        / max(float(np.linalg.norm(reference.reshape(-1))), root_cells)
    )
    quarter = float(deficit_max_quarter_jy)
    half = float(deficit_max_half_jy)
    factor = quarter / deficit_max if deficit_max > 0.0 else math.inf

    tier_one_a = (
        horizon_free_max <= horizon_free_max_limit
        and horizon_free_l2 <= HORIZON_FREE_L2_LIMIT
    )
    # An exact-zero deficit satisfies both convergence lines.
    converged = deficit_max == 0.0 or (
        quarter > half > deficit_max and factor >= CONVERGENCE_FACTOR_FLOOR
    )
    return DirectGateRecord(
        predicate_id=TWO_TIER_PREDICATE_ID,
        reference_cube_sha256=_cube_identity(reference, "frozen_gauss128"),
        candidate_cube_sha256=_visibility_cube_identity(candidate),
        reference_error_cube_sha256=_error_cube_identity(error),
        horizon_free_cube_sha256=_shell_identity(horizon_free, "horizon_free"),
        horizon_free_qcheck_cube_sha256=_shell_identity(
            horizon_free_qcheck, "horizon_free_qcheck"
        ),
        quadrature_shell_cube_sha256=_shell_identity(shell, "quadrature_shell"),
        expected_cell_count=cells,
        compared_finite_cell_count=cells,
        evaluated_error_cell_count=cells,
        numerical_scale_jy=numerical_scale,
        horizon_free_shell_max_jy=horizon_free_max,
        horizon_free_shell_l2=horizon_free_l2,
        horizon_free_shell_max_limit_jy=horizon_free_max_limit,
        horizon_free_shell_l2_limit=HORIZON_FREE_L2_LIMIT,
        quadrature_shell_max_jy=shell_max,
        quadrature_shell_l2=shell_l2,
        reference_scale_jy=reference_scale,
        deficit_max_jy=deficit_max,
        deficit_l2=deficit_l2,
        deficit_max_quarter_jy=quarter,
        deficit_max_half_jy=half,
        convergence_factor=factor,
        pass_=bool(tier_one_a and converged),
    )


_CUBE_AXES: Final[tuple[str, ...]] = (
    "time",
    "baseline",
    "frequency",
    "correlation",
)


def _cube_identity(cube: np.ndarray, role: str) -> str:
    return array_digest(
        "radiosim.mmode-direct-cube.v1",
        role,
        list(_CUBE_AXES),
        "Jy",
        np.asarray(cube, dtype=np.complex128),
        dtype="complex128-be",
    )


def _error_cube_identity(cube: np.ndarray) -> str:
    return array_digest(
        "radiosim.mmode-direct-root-error.v1",
        "frozen_enclosure_error",
        list(_CUBE_AXES),
        "Jy",
        np.asarray(cube, dtype=np.float64),
        dtype="float64-be",
    )


def _shell_identity(cube: np.ndarray, role: str) -> str:
    """Return a role-qualified identity for a tier-1 shell cube.

    The horizon-free cubes are tier-1 internals and never become a result, so
    their identities are qualified by role and can never be mistaken for the
    published visibility cube.
    """
    return array_digest(
        "radiosim.mmode-visibility-cube.v1",
        role,
        list(_CUBE_AXES),
        "Jy",
        np.asarray(cube, dtype=np.complex128),
        dtype="complex128-be",
    )


def _visibility_cube_identity(cube: np.ndarray) -> str:
    return array_digest(
        "radiosim.mmode-visibility-cube.v1",
        "visibility_cube",
        list(_CUBE_AXES),
        "Jy",
        np.asarray(cube, dtype=np.complex128),
        dtype="complex128-be",
    )


# ---------------------------------------------------------------------------
# Section 4.2 frame applicability certificate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LedgerDirection:
    """One canonical Section 12.1 direction-ledger entry."""

    direction_id: str
    source_kind: str
    component_index: int
    source_index: int
    transfer_role: str
    transfer_nside: int
    cirs_direction: np.ndarray
    icrs_ra_rad: float
    icrs_dec_rad: float
    active_frequency_mask: tuple[bool, ...]
    resolved_stokes_iau: np.ndarray
    integration_weight: float

    @property
    def is_direct_contributor(self) -> bool:
        """Return whether this row feeds the private direct oracle."""
        return self.source_kind in ("point", "native_healpix")


@dataclass(frozen=True, slots=True)
class FrameCertificate:
    """The Section 4.2 ``FrameApplicabilityCertificate`` of one solve."""

    certificate_sha256: str
    row: Mapping[str, Any]
    frozen_gauss64: np.ndarray
    frozen_gauss128: np.ndarray
    operational_gauss64: np.ndarray
    operational_gauss128: np.ndarray
    frozen_enclosure_error: np.ndarray
    operational_enclosure_error: np.ndarray
    passed: bool

    @property
    def frozen_gauss128_cube_sha256(self) -> str:
        """Return the retained final frozen direct-cube identity."""
        return _cube_identity(self.frozen_gauss128, "frozen_gauss128")

    @property
    def frozen_enclosure_error_cube_sha256(self) -> str:
        """Return the retained frozen enclosure-error cube identity."""
        return _error_cube_identity(self.frozen_enclosure_error)


def build_direction_ledger(
    *,
    frame: FrozenFrame,
    dimensions: Any,
    point_cirs: np.ndarray,
    point_stokes: np.ndarray,
    point_icrs: np.ndarray,
    native_cirs: np.ndarray,
    native_stokes: np.ndarray,
    native_icrs: np.ndarray,
    native_solid_angle: float,
) -> tuple[LedgerDirection, ...]:
    """Build Section 12.1's canonical direction ledger, in its fixed order.

    Rows are ordered point, native, production transfer, then diagnostic
    transfer grids by increasing nside; component source indices ascend in
    canonical-RING order for point/native groups and in iso-Gauss ring-major
    order for transfer groups.  Rows from different kinds, components, roles or
    grids are never de-duplicated merely because their vectors coincide.
    """
    from radiosim.core.mmode.transfer import quadrature_grid

    rows: list[LedgerDirection] = []
    for index in range(point_cirs.shape[0]):
        stokes = np.asarray(point_stokes[index], dtype=np.float64)
        rows.append(
            LedgerDirection(
                direction_id=f"point:0:{index}",
                source_kind="point",
                component_index=0,
                source_index=index,
                transfer_role="none",
                transfer_nside=0,
                cirs_direction=point_cirs[index],
                icrs_ra_rad=float(point_icrs[index, 0]),
                icrs_dec_rad=float(point_icrs[index, 1]),
                active_frequency_mask=tuple(bool(np.any(row != 0.0)) for row in stokes),
                resolved_stokes_iau=stokes,
                integration_weight=1.0,
            )
        )
    for index in range(native_cirs.shape[0]):
        stokes = np.asarray(native_stokes[index], dtype=np.float64)
        rows.append(
            LedgerDirection(
                direction_id=f"native_healpix:0:{index}",
                source_kind="native_healpix",
                component_index=0,
                source_index=index,
                transfer_role="none",
                transfer_nside=0,
                cirs_direction=native_cirs[index],
                icrs_ra_rad=float(native_icrs[index, 0]),
                icrs_dec_rad=float(native_icrs[index, 1]),
                active_frequency_mask=tuple(bool(np.any(row != 0.0)) for row in stokes),
                resolved_stokes_iau=stokes,
                integration_weight=float(native_solid_angle),
            )
        )
    frequency_count = (
        point_stokes.shape[1]
        if point_stokes.size
        else (native_stokes.shape[1] if native_stokes.size else 1)
    )
    catalogue: list[tuple[str, int]] = [
        ("production", int(dimensions.quadrature_nside))
    ]
    catalogue.extend(
        ("diagnostic", int(nside)) for nside in dimensions.diagnostic_nsides
    )
    for role, nside in catalogue:
        nodes, weights = quadrature_grid(nside)
        icrs = _cirs_to_icrs(frame, nodes)
        for index in range(nodes.shape[0]):
            rows.append(
                LedgerDirection(
                    direction_id=f"transfer_quadrature:{role}:{nside}:{index}",
                    source_kind="transfer_quadrature",
                    component_index=0,
                    source_index=index,
                    transfer_role=role,
                    transfer_nside=nside,
                    cirs_direction=nodes[index],
                    icrs_ra_rad=float(icrs[index, 0]),
                    icrs_dec_rad=float(icrs[index, 1]),
                    active_frequency_mask=(True,) * frequency_count,
                    resolved_stokes_iau=np.zeros((0, 4), dtype=np.float64),
                    integration_weight=float(weights[index]),
                )
            )
    return tuple(rows)


def _cirs_to_icrs(frame: FrozenFrame, vectors: np.ndarray) -> np.ndarray:
    """Return the ICRS ``(ra, dec)`` of frozen CIRS unit vectors.

    Section 12.1's operational census consumes public
    ``SkyCoord.transform_to(AltAz(...))`` values, which start from the sky's own
    ICRS description; a grid defined in the frozen CIRS frame therefore inverts
    once through the same public transform.
    """
    from astropy.coordinates import CartesianRepresentation, SkyCoord

    from radiosim.core.mmode.time import installed_iers_context

    values = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    with installed_iers_context():
        cirs = SkyCoord(
            CartesianRepresentation(
                values[:, 0], values[:, 1], values[:, 2], copy=True
            ),
            frame=frame.cirs_frame.astropy_frame.replicate_without_data(),
        )
        icrs = cirs.icrs
        return np.stack(
            (
                np.asarray(icrs.ra.rad, dtype=np.float64),
                np.asarray(icrs.dec.rad, dtype=np.float64),
            ),
            axis=-1,
        )


def magnitude_ceiling(
    *,
    payload_magnitude: float,
    factor_ceilings: Sequence[float],
) -> float:
    """Return Section 12's certified magnitude ceiling ``G_abs``.

    ``G_abs = round_up(|payload| * prod(certified factor ceilings))`` where
    ``|payload|`` is the contributor's resolved Stokes magnitude times its
    integration weight and each remaining factor of the Section 6 kernel --
    every Jones-term operator norm along the chain, the unit-magnitude fringe,
    and the accepted ``M``/``Q`` factor magnitudes -- enters through a
    design-recorded certified upper bound.  No interval extension of the
    complete integrand is taken, and no ad-hoc estimate is admitted.
    """
    product = float(abs(payload_magnitude))
    for ceiling in factor_ceilings:
        product *= float(ceiling)
    return float(np.nextafter(product, math.inf))


# ---------------------------------------------------------------------------
# Section 6 kernel, shared by the transfer and the direct oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KernelContext:
    """Everything the Section 6 kernel needs, resolved once for a run."""

    frame: FrozenFrame
    beam_system: Any
    antenna_ids: tuple[Any, ...]
    selected_pairs: tuple[tuple[int, int], ...]
    baseline_vectors_enu_m: np.ndarray
    frequencies_hz: np.ndarray
    time_mjd: float
    #: Section 6's constant per-antenna ``M_p = H_p C_p``, in solver row order.
    #: The chain's ``P`` term is exactly the identity for the shipped mounts, so
    #: this is the whole direction-independent factor of ``J_{p,NE}``.
    receptor_matrices: tuple[Any, ...] = ()
    #: The run's four resolved row-major correlation labels.
    correlation_labels: tuple[str, ...] = ()

    @property
    def n_baselines(self) -> int:
        """Return the selected baseline count."""
        return len(self.selected_pairs)

    @property
    def n_frequencies(self) -> int:
        """Return the run frequency count."""
        return int(self.frequencies_hz.shape[0])


def _enu_components(frame: FrozenFrame, itrs: np.ndarray) -> np.ndarray:
    """Return ``(East, North, Up)`` components of ITRS column vectors."""
    basis = np.stack(
        (
            np.asarray(frame.local_east_itrs, dtype=np.float64),
            np.asarray(frame.local_north_itrs, dtype=np.float64),
            np.asarray(frame.local_up_itrs, dtype=np.float64),
        ),
        axis=0,
    )
    return np.asarray(itrs, dtype=np.float64) @ basis.T


def frozen_enu_at_phase(
    frame: FrozenFrame, cirs: np.ndarray, relative_phase_rad: float
) -> np.ndarray:
    """Return the frozen terrestrial ``(E, N, U)`` directions at one phase.

    Section 4.1's rigid group composition makes this exact: the whole instrument
    is fixed in the terrestrial frame and ``T(alpha) = T(0) R3(alpha)``.
    """
    attitude = frame.attitude_at(float(relative_phase_rad))
    itrs = (
        np.atleast_2d(np.asarray(cirs, dtype=np.float64))
        @ np.asarray(attitude, dtype=np.float64).T
    )
    return _enu_components(frame, itrs)


def section6_kernel(
    context: KernelContext,
    enu: np.ndarray,
    *,
    horizon: bool = True,
    stokes_fields: Sequence[str] = ("I",),
) -> dict[str, np.ndarray]:
    r"""Return Section 6's ``K^X_{pqfc}`` cells, each shaped ``(n_dir, B, F, 4)``.

    .. math::

        K^X_{pqfc}(\hat n)=
        \bigl[J_{p,\theta\phi}P^X_{\theta\phi}J^H_{q,\theta\phi}\bigr]_c
        K_{pq}(\hat n)H(\hat n)

    With Section 5.2's bridge ``J_{\theta\phi}=J_{NE}D`` the bracket is exactly
    ``[J_{p,NE}(D P^X D)J^H_{q,NE}]_c``, and Section 6 anchors ``J_{NE}`` to the
    accepted direct RIME: the chain's direction-independent terms -- the
    resolved receptor factors ``M_p = H_p C_p`` among them -- right-multiply the
    celestial ``(North, East)`` coherency as **constant** matrices in that same
    basis, while every mount-dependent tangent rotation belongs to the ``P``
    term, which is exactly the identity for the shipped ``fixed`` and
    unspecified mounts.  So ``J_{p,NE}(\hat n) = M_p E_p(\hat n)`` and no
    transport angle enters the kernel: constant cells are constant coefficients
    on spin-weighted fields, which preserves the integrand's spin weight and
    keeps Section 7.3's spin-``+-2`` quadrature spectrally exact.

    Omitting ``M_p`` is harmless only for Stokes ``I`` under a unitary receptor
    -- ``M P^I M^H = (1/2) M M^H = (1/2) I_2`` -- and is wrong for every
    polarized component, which is why the resolved matrix enters here rather
    than at the correlation-labelling boundary.

    The fringe is the *existing* geometric phase at its accepted sign, and the
    horizon factor is Section 6's one shared strict ``alt > 0`` predicate with
    equality excluded -- no epsilon, beam cutoff, or half weight.

    Parameters
    ----------
    stokes_fields : sequence of str
        Which of ``("I", "Q", "U", "V")`` to evaluate.  A component whose
        resolved sky payload is exactly zero contributes exactly zero, so a
        scalar run asks for ``("I",)`` and its arithmetic is untouched by the
        polarized surface.
    horizon : bool
        When ``False`` the horizon factor is replaced by ``H === 1`` and
        **everything else is identical**: same grids, same beam object, same
        fringe, same packing, contraction and synthesis.  That is Section 7.3's
        tier-1a ablation, and it runs through this one code path precisely so
        the compared pipelines cannot diverge anywhere but the factor under
        test.

        The ablation removes *every* horizon truncation, not only the explicit
        factor.  The resolved ``BeamSystem`` applies its own cut -- its response
        drops from ``0.81`` at ``alt = +0.001`` degrees to exactly zero at
        ``alt = -0.001`` -- so leaving that in place would keep a step of the
        beam's own amplitude in the integrand and the shell would measure the
        same ``nside**-2`` quadrature error the horizon-free tier exists to
        exclude.  The beam is therefore sampled at ``abs(alt)``, which is its
        *exact* smooth continuation: an aperture pattern depends on the zenith
        angle through ``sin(theta) = cos(alt)``, an even function of ``alt``.
        The fringe needs no such treatment; it is entire in the direction
        cosines and stays on the true direction.
    """
    from radiosim.backends import get_backend
    from radiosim.core.jones import geometric_phase
    from radiosim.core.mmode.transfer import bridged_stokes_matrices

    directions = np.atleast_2d(np.asarray(enu, dtype=np.float64))
    count = directions.shape[0]
    up = directions[:, 2]
    horizon_factor = (
        strict_horizon_indicator(up) if horizon else np.ones(count, dtype=np.float64)
    )
    altitude = np.arcsin(np.clip(up, -1.0, 1.0))
    if not horizon:
        altitude = np.abs(altitude)
    azimuth = np.arctan2(directions[:, 0], directions[:, 1])

    backend = get_backend("numpy")
    n_baselines = context.n_baselines
    n_frequencies = context.n_frequencies
    requested = tuple(str(name) for name in stokes_fields)
    unknown = [name for name in requested if name not in STOKES_COMPONENT_ORDER]
    if unknown:
        raise ValueError(f"unsupported Stokes kernel components {sorted(unknown)}")
    bridged = bridged_stokes_matrices()
    receptors = tuple(context.receptor_matrices)
    kernels = {
        name: np.zeros((count, n_baselines, n_frequencies, 4), dtype=np.complex128)
        for name in requested
    }
    for frequency_index in range(n_frequencies):
        frequency = float(context.frequencies_hz[frequency_index])
        responses = [
            np.asarray(
                context.beam_system.evaluate_jones(
                    antenna_id,
                    altitude_rad=altitude,
                    azimuth_rad=azimuth,
                    frequency_hz=frequency,
                    time_mjd=context.time_mjd,
                ),
                dtype=np.complex128,
            )
            for antenna_id in context.antenna_ids
        ]
        # ``J_{p,NE} = M_p E_p`` -- the constant resolved receptor matrix times
        # the sampled beam response, both in the celestial North/East basis.
        antenna_jones = [
            response if index >= len(receptors) else receptors[index] @ response
            for index, response in enumerate(responses)
        ]
        wavelength = _SPEED_OF_LIGHT_M_PER_S / frequency
        uvw = np.asarray(context.baseline_vectors_enu_m, dtype=np.float64) / wavelength
        fringe = np.asarray(
            geometric_phase(
                uvw_wavelengths=uvw,
                dir_l=directions[:, 0],
                dir_m=directions[:, 1],
                dir_n=directions[:, 2],
                backend=backend,
            ),
            dtype=np.complex128,
        )
        for baseline_index, (first, second) in enumerate(context.selected_pairs):
            jones_p = antenna_jones[first]
            jones_q = np.conjugate(antenna_jones[second]).transpose(0, 2, 1)
            factor = fringe[baseline_index][:, None] * horizon_factor[:, None]
            for name in requested:
                # ``[J_p (D P^X D) J_q^H]`` in the celestial tangent basis.
                coherency = (jones_p @ bridged[name]) @ jones_q
                kernels[name][:, baseline_index, frequency_index, :] = (
                    coherency.reshape(count, 4) * factor
                )
    return kernels


_SPEED_OF_LIGHT_M_PER_S: Final[float] = 299792458.0


def field_integrands(kernels: Mapping[str, Any]) -> dict[str, Any]:
    r"""Return Section 6's four field integrands from the four Stokes kernels.

    .. math::

        B^{(+2)}=\int(K^Q-iK^U)\,{}_{+2}Y_{lm}\,d\Omega,\qquad
        B^{(-2)}=\int(K^Q+iK^U)\,{}_{-2}Y_{lm}\,d\Omega,

    so the ``+2`` integrand is ``K^Q - i K^U`` and the ``-2`` integrand its
    conjugate combination; ``I`` and ``V`` integrate their own kernels.  One
    code object serves the production transfer and the acceptance fixture, so
    the conjugate placement cannot differ between the two.

    The keys are Section 5.3's field names, not Stokes names.
    """
    stokes_q = kernels["Q"]
    stokes_u = kernels["U"]
    return {
        "I": kernels["I"],
        "+2": stokes_q - 1j * stokes_u,
        "-2": stokes_q + 1j * stokes_u,
        "V": kernels["V"],
    }


def _is_polarized_table(table: Any) -> bool:
    """Return whether a packed block table carries Section 5.3's four fields."""
    from radiosim.core.mmode.types import PolarizedPackedTable

    return isinstance(table, PolarizedPackedTable)


def _packed_block_row(table: Any, order: int, field: str) -> Mapping[str, Any] | None:
    """Return one ``(signed m, field)`` block row, or ``None`` when it does not exist.

    Section 5.3's signed-``m``-major layout makes both lookups exact positions
    rather than searches: a scalar table has one row per signed ``m`` and
    represents only the ``I`` field, while a four-field table has
    ``len(FIELD_ORDER)`` consecutive rows per signed ``m`` in the fixed field
    order.
    """
    if abs(int(order)) > int(table.mmax):
        return None
    if not _is_polarized_table(table):
        if str(field) != FIELD_ORDER[0]:
            return None
        return table.block_rows[int(order) + int(table.mmax)]
    position = (int(order) + int(table.mmax)) * len(FIELD_ORDER) + FIELD_ORDER.index(
        str(field)
    )
    return table.block_rows[position]


# ---------------------------------------------------------------------------
# Section 4.2 frame applicability certificate
# ---------------------------------------------------------------------------


def build_input_identity(
    *,
    request: Any,
    grid: CanonicalEraGrid,
    frame: FrozenFrame,
    context: KernelContext,
    dimensions: Any,
    directions: Sequence[LedgerDirection],
    tangent_frame: Any = None,
) -> tuple[dict[str, Any], str]:
    """Return Section 14.3's complete input-identity manifest and its digest.

    The manifest is the value-bearing preimage of every resolved input the run
    consumed: the site, both canonical grids and their UTC/UT1 mappings, the
    array, the receptors and correlations, the beams, the sky components, the
    complete direction-input ledger, the Jones inventory, the transfer
    catalogue, and the fixed convention identity.  It deliberately excludes
    fixture labels, paths, backend or device, workers, memory, timings, outputs,
    certificates and result cubes.
    """
    from radiosim.core.mmode.time import ut1_manifest, utc_manifest
    from radiosim.core.polarization_basis import CORRELATION_LABELS

    utc, utc_sha256 = utc_manifest(grid)
    ut1, ut1_sha256 = ut1_manifest(grid)
    instrument = request.instrument
    triad = np.stack(
        (
            np.asarray(frame.local_east_itrs, dtype=np.float64),
            np.asarray(frame.local_north_itrs, dtype=np.float64),
            np.asarray(frame.local_up_itrs, dtype=np.float64),
        ),
        axis=0,
    )
    positions_itrs = np.asarray(instrument.positions_enu_m, dtype=np.float64) @ triad
    antenna_rows = [
        {
            "antenna_index": index,
            "name": str(name),
            "itrs_xyz_m_f64be": [
                f64be(float(value)) for value in positions_itrs[index]
            ],
        }
        for index, name in enumerate(instrument.antenna_names)
    ]
    vectors_itrs = (
        np.asarray(instrument.baseline_vectors_enu_m, dtype=np.float64) @ triad
    )
    baseline_rows = [
        {
            "baseline_index": index,
            "antenna1_index": int(instrument.row_index_by_number[pair[0]]),
            "antenna2_index": int(instrument.row_index_by_number[pair[1]]),
            "itrs_vector_m_f64be": [
                f64be(float(value)) for value in vectors_itrs[index]
            ],
        }
        for index, pair in enumerate(instrument.selected_pairs)
    ]
    widths = np.asarray(
        getattr(request, "channel_widths_hz", None)
        if getattr(request, "channel_widths_hz", None) is not None
        else np.zeros_like(context.frequencies_hz),
        dtype=np.float64,
    )
    frequency_rows = [
        {
            "frequency_index": index,
            "center_hz_f64be": f64be(float(context.frequencies_hz[index])),
            "width_hz_f64be": f64be(float(widths[index]) if widths.size else 0.0),
        }
        for index in range(context.n_frequencies)
    ]
    receptors = request.receptors
    by_number = {
        identifier.number: resolved
        for identifier, resolved in receptors.receptor_by_antenna.items()
    }
    receptor_rows = []
    for index, number in enumerate(instrument.antenna_numbers):
        resolved = by_number[int(number)]
        receptor_rows.append(
            {
                "antenna_index": index,
                "basis": str(resolved.basis),
                "labels": [str(label) for label in resolved.feed_array],
                "feed_rotation_rad_f64be": f64be(float(resolved.feed_rotation_rad)),
                "feed_angle_rad_f64be": [
                    f64be(float(value)) for value in resolved.feed_angle_rad
                ],
            }
        )
    labels = CORRELATION_LABELS[receptors.output_basis]
    correlation_rows = [
        {
            "correlation_index": index,
            "p_label": str(label)[0],
            "q_label": str(label)[1],
        }
        for index, label in enumerate(labels)
    ]
    beam_rows = _beam_rows(request.beam_system, instrument, context.antenna_ids)
    sky_rows, direction_rows = _sky_component_rows(
        request, directions, context.frequencies_hz, tangent_frame
    )
    jones_rows = _jones_term_rows(request.jones)
    catalog, _catalog_sha256 = _transfer_catalog(directions)
    manifest = {
        "schema_version": "radiosim.mmode-input-identity.v1",
        "site_manifest": frame.site_manifest,
        "site_sha256": frame.site_sha256,
        "iers_table_sha256": frame.iers_table_sha256,
        "canonical_era_turn_grid": dict(grid.canonical_era_turn_grid),
        "canonical_era_turn_grid_sha256": grid.canonical_era_turn_grid_sha256,
        "canonical_era_grid": dict(grid.canonical_era_grid),
        "canonical_era_grid_sha256": grid.canonical_era_grid_sha256,
        "utc_manifest": dict(utc),
        "utc_sha256": utc_sha256,
        "ut1_manifest": dict(ut1),
        "ut1_sha256": ut1_sha256,
        "mmode_dimensions": {
            "sidereal_samples": grid.sidereal_samples,
            "lmax": int(dimensions.lmax),
            "mmax": int(dimensions.mmax),
            "quadrature_nside": int(dimensions.quadrature_nside),
            "lcheck": int(dimensions.lcheck),
            "mcheck": int(dimensions.mcheck),
            "qcheck": int(dimensions.qcheck),
        },
        "antenna_rows": antenna_rows,
        "baseline_rows": baseline_rows,
        "frequency_rows": frequency_rows,
        "receptor_rows": receptor_rows,
        "correlation_rows": correlation_rows,
        "beam_rows": beam_rows,
        "sky_component_rows": sky_rows,
        "direction_input_rows": direction_rows,
        "jones_term_rows": jones_rows,
        "transfer_grid_catalog": catalog,
        "precision": "standard",
        "result_dtype": "complex128",
        "convention_identity_sha256": object_digest(
            "radiosim.mmode-conventions.v1", dict(CONVENTION_IDENTITY)
        ),
    }
    return manifest, object_digest("radiosim.mmode-input-identity.v1", manifest)


def _identity_manifest(
    schema: str, kind: str, scalars: Sequence[tuple[str, str, Any]]
) -> tuple[dict[str, Any], str]:
    """Return one Section 14.3 parameter/morphology identity and its digest."""
    manifest = {
        "schema_version": schema,
        "identity_kind": kind,
        "scalar_rows": [
            {"name": name, "type": row_type, "value": value}
            for name, row_type, value in sorted(scalars)
        ],
        "array_rows": [],
    }
    return manifest, object_digest(schema, manifest)


def _beam_rows(
    beam_system: Any, instrument: Any, antenna_ids: Sequence[Any]
) -> list[dict[str, Any]]:
    """Return Section 14.3's beam rows, grouped by resolved response identity.

    ``BeamSystem.response_key`` is the canonical grouping key: two antennas
    share it exactly when their resolved responses are identical, which is what
    a class/parameter group means here.  Rows appear in first-assigned-antenna
    order and partition the antenna indices exactly once.
    """
    state = beam_system.state
    handler_by_id = {handler.handler_id: handler for handler in state.handlers}
    assignment = dict(state.assignment_handler_ids)
    groups: list[dict[str, Any]] = []
    index_by_key: dict[str, int] = {}
    for index, antenna in enumerate(antenna_ids):
        key = beam_system.response_key(antenna)
        if key not in index_by_key:
            handler = handler_by_id[assignment[antenna]]
            scalars: list[tuple[str, str, Any]] = [
                ("definition_fingerprint", "literal", handler.definition_fingerprint),
                ("handler_kind", "literal", handler.kind),
                ("response_key", "literal", key),
                (
                    "scientific_fingerprint",
                    "literal",
                    handler.scientific_fingerprint,
                ),
            ]
            manifest, digest = _identity_manifest(
                "radiosim.mmode-parameter-identity.v1", handler.kind, scalars
            )
            index_by_key[key] = len(groups)
            groups.append(
                {
                    "beam_index": len(groups),
                    "assigned_antenna_indices": [],
                    "class_qualname": type(beam_system).__qualname__,
                    "electric_field_basis": "native_feed",
                    "normalization": (
                        "uvbeam_peak_common_v1"
                        if handler.kind == "fits"
                        else "unmodified_ideal_aperture_v1"
                    ),
                    "parameter_identity_manifest": manifest,
                    "parameter_identity_sha256": digest,
                }
            )
        groups[index_by_key[key]]["assigned_antenna_indices"].append(index)
    del instrument
    return groups


#: Section 14.3's ``polarization_frame`` value for a payload Section 5.1 lets
#: omit the tangent block -- one with no non-zero ``Q`` or ``U``.
NO_LINEAR_POLARIZATION_FRAME: Final = "not_applicable_no_linear_polarization"


def _sky_component_rows(
    request: Any,
    directions: Sequence[LedgerDirection],
    frequencies_hz: np.ndarray,
    tangent_frame: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return Section 14.3's sky-component and direction-input row arrays.

    ``polarization_frame`` is the resolved Section 5.1 tangent block when the
    payload carries linear polarization, and the fixed
    ``not_applicable_no_linear_polarization`` literal otherwise.  A polarized run
    whose identity manifest claimed the literal would describe a different sky
    from the one it integrated.
    """
    frequencies = [float(value) for value in frequencies_hz]
    polarization_frame: Any = (
        dict(tangent_frame)
        if isinstance(tangent_frame, Mapping)
        else NO_LINEAR_POLARIZATION_FRAME
    )
    direction_rows = [
        {
            "direction_input_manifest": direction_input_manifest(row, frequencies),
            "direction_input_sha256": object_digest(
                "radiosim.mmode-direction-input.v1",
                direction_input_manifest(row, frequencies),
            ),
        }
        for row in directions
    ]
    by_kind: dict[str, list[str]] = {}
    for position, row in enumerate(directions):
        if row.source_kind not in ("point", "native_healpix"):
            continue
        by_kind.setdefault(row.source_kind, []).append(
            direction_rows[position]["direction_input_sha256"]
        )
    component_rows: list[dict[str, Any]] = []
    for component_index, kind in enumerate(sorted(by_kind)):
        morphology, morphology_sha256 = _identity_manifest(
            "radiosim.mmode-morphology-identity.v1",
            kind,
            [
                ("element_count", "integer", str(len(by_kind[kind]))),
                ("representation", "literal", kind),
            ],
        )
        component_rows.append(
            {
                "component_index": component_index,
                "representation": (
                    "point_sources" if kind == "point" else "healpix_map"
                ),
                "coordinate_frame": "icrs",
                "polarization_frame": polarization_frame,
                "polarization_frame_sha256": object_digest(
                    "radiosim.sky-tangent-polarization.v1",
                    polarization_frame,
                ),
                "morphology_identity_manifest": morphology,
                "morphology_identity_sha256": morphology_sha256,
                "direction_input_sha256s": by_kind[kind],
            }
        )
    return component_rows, direction_rows


def _jones_term_rows(jones: Any) -> list[dict[str, Any]]:
    """Return Section 14.3's Jones-term rows in canonical chain order."""
    rows: list[dict[str, Any]] = []
    terms = tuple(getattr(jones, "terms", ()) or ())
    for index, term in enumerate(terms):
        qualname = type(term).__qualname__
        manifest, digest = _identity_manifest(
            "radiosim.mmode-parameter-identity.v1",
            qualname,
            [
                ("class_qualname", "literal", qualname),
                (
                    "is_direction_dependent",
                    "boolean",
                    bool(getattr(term, "is_direction_dependent", False)),
                ),
                ("term_name", "literal", str(getattr(term, "name", qualname))),
            ],
        )
        rows.append(
            {
                "term_index": index,
                "term_name": str(getattr(term, "name", qualname)),
                "class_qualname": qualname,
                "parameter_identity_manifest": manifest,
                "parameter_identity_sha256": digest,
                "time_stationarity": "stationary",
            }
        )
    return rows


def build_frame_certificate(
    *,
    grid: CanonicalEraGrid,
    frame: FrozenFrame,
    context: KernelContext,
    directions: Sequence[LedgerDirection],
    beam_peak_ceiling: float,
    input_identity_sha256: str,
) -> FrameCertificate:
    """Compute the Section 4.2 certificate in memory, before harmonic work.

    Both censuses run over the complete direction ledger: the frozen analytic
    trajectory from its exact integer-ratio topology decision, and the
    operational trajectory from Section 12.1's certified-ceiling scan.  Their
    closed root enclosures then enter the unchanged pairing, lift, slab, sign
    and membership machinery, and the point/native rows drive the horizon-split
    Gauss-64/128 direct cubes with their certified magnitude-ceiling error
    disks.

    The returned row is the complete Section 14.2 frame row less exactly
    ``fixture_id``, ``certificate_sha256`` and ``pass``, which is precisely the
    preimage Section 14.3 hashes, so the evidence generator adds those three
    fields and embeds the row verbatim rather than rebuilding it.
    """
    horizon_lo, horizon_hi = grid.horizon_domain
    frozen = [
        frozen_horizon_trajectory(
            frame, row.cirs_direction, horizon_lo=horizon_lo, horizon_hi=horizon_hi
        )
        for row in directions
    ]
    scan = scan_operational_horizon(
        frame=frame,
        grid=grid,
        ra_rad=np.asarray([row.icrs_ra_rad for row in directions]),
        dec_rad=np.asarray([row.icrs_dec_rad for row in directions]),
        frozen_root_bounds=[
            [
                bound
                for root in trajectory.roots
                for bound in (root.turn_lo, root.turn_hi)
            ]
            for trajectory in frozen
        ],
        direction_ids=[row.direction_id for row in directions],
    )

    retained_guards = sum(
        1 for entry in scan.crossing_rows if entry["classification"] == "guard_interval"
    )
    if retained_guards != scan.guard_count:
        raise ValueError(
            "the retained guard rows do not account for every guard the scan "
            "emitted; the economy projection would hide a flank"
        )
    direction_rows, direction_ledger = _direction_rows(directions, context)
    catalog, catalog_sha256 = _transfer_catalog(directions)
    enclosure_manifest, enclosure_sha256 = _enclosure_manifest(
        input_identity_sha256=input_identity_sha256,
        frame_matrix_sha256=frame.frame_matrix_sha256,
    )

    pair_rows, slab_rows, root_max_rad, mismatch_measure_turn = _pair_roots(
        directions, frozen, scan.roots, grid
    )
    slab_geometry: dict[str, list[tuple[Fraction, Fraction]]] = {}
    for slab in slab_rows:
        pieces = slab_geometry.setdefault(str(slab["direction_id"]), [])
        for piece in slab["pieces"]:
            pieces.append(
                (Fraction(str(piece["turn_lo"])), Fraction(str(piece["turn_hi"])))
            )
    sign_rows, sign_mismatches = _sign_intervals(
        directions, frozen, scan.roots, slab_geometry, grid, scan.evaluator
    )
    membership_rows, membership_ledger, membership_mismatches = _membership(
        directions, frozen, scan.centre_values, slab_geometry, grid
    )
    # Section 12.1's economy is a projection, not a loss, and that claim is
    # checked rather than asserted: the retained masks are expanded back to the
    # complete per-sample array here and re-digested against the ledger.
    if expand_membership_masks(membership_rows, grid) != membership_ledger:
        raise ValueError(
            "the retained horizon-membership masks do not expand to their "
            "ledger digest; the economy projection is not lossless"
        )
    phase = _phase_census(
        grid=grid,
        frame=frame,
        context=context,
        directions=directions,
        frozen=frozen,
        operational_roots=scan.roots,
    )
    direct = _direct_cubes(
        grid=grid,
        frame=frame,
        context=context,
        directions=directions,
        frozen=frozen,
        operational_roots=scan.roots,
        beam_peak_ceiling=beam_peak_ceiling,
        input_identity_sha256=input_identity_sha256,
        enclosure_manifest_sha256=enclosure_sha256,
    )

    exact_tau = Fraction(*TAU.as_integer_ratio())
    paired = sum(len(row["pairs"]) for row in pair_rows)
    root_count_mismatches = sum(
        1
        for row in pair_rows
        if row["frozen_root_count"] != row["operational_root_count"]
    )
    orientation_mismatches = sum(
        int(row["orientation_mismatch_count"]) for row in pair_rows
    )

    kinds = [row.source_kind for row in directions]
    roles = [row.transfer_role for row in directions]
    point_count = kinds.count("point")
    native_count = kinds.count("native_healpix")
    production_count = sum(
        1
        for kind, role in zip(kinds, roles, strict=True)
        if kind == "transfer_quadrature" and role == "production"
    )
    diagnostic_count = sum(
        1
        for kind, role in zip(kinds, roles, strict=True)
        if kind == "transfer_quadrature" and role == "diagnostic"
    )
    transfer_count = production_count + diagnostic_count

    samples = grid.sidereal_samples
    cells = 4 * samples * context.n_baselines * context.n_frequencies
    direct_directions = sum(1 for row in directions if row.is_direct_contributor)

    scale_q = max(
        1.0,
        float(np.max(np.abs(direct["F128"]))) if direct["F128"].size else 0.0,
        float(np.max(np.abs(direct["O128"]))) if direct["O128"].size else 0.0,
    )
    change_frozen = (
        float(np.max(np.abs(direct["F128"] - direct["F64"])))
        if direct["F128"].size
        else 0.0
    )
    change_operational = (
        float(np.max(np.abs(direct["O128"] - direct["O64"])))
        if direct["O128"].size
        else 0.0
    )
    upper = np.abs(direct["F128"] - direct["O128"]) + direct["EF"] + direct["EO"]
    scale_frame = max(
        1.0,
        float(np.max(np.abs(direct["O128"]) + direct["EO"])) if upper.size else 0.0,
    )
    cube_max = float(np.max(upper)) if upper.size else 0.0
    reference_norm = max(
        float(np.linalg.norm(direct["O128"].reshape(-1))) if upper.size else 0.0,
        math.sqrt(cells),
    )
    cube_l2 = (
        float(np.linalg.norm(upper.reshape(-1)) / reference_norm)
        if reference_norm > 0.0
        else 0.0
    )

    row: dict[str, Any] = {
        "site_manifest": frame.site_manifest,
        "site_sha256": frame.site_sha256,
        "input_identity_sha256": input_identity_sha256,
        "iers_table_sha256": frame.iers_table_sha256,
        "frame_matrix_manifest": frame.frame_matrix_manifest,
        "frame_matrix_sha256": frame.frame_matrix_sha256,
        "canonical_era_turn_grid_sha256": grid.canonical_era_turn_grid_sha256,
        "canonical_era_grid_sha256": grid.canonical_era_grid_sha256,
        "pm_source_unit": frame.pm_source_unit,
        "pom00_argument_unit": frame.pom00_argument_unit,
        "xp0_arcsec": f64be(frame.xp0_arcsec),
        "yp0_arcsec": f64be(frame.yp0_arcsec),
        "das2r_rad_per_arcsec": f64be(frame.das2r_rad_per_arcsec),
        "xp0_rad": f64be(frame.xp0_rad),
        "yp0_rad": f64be(frame.yp0_rad),
        "sp0_rad": f64be(frame.sp0_rad),
        "diagnostic_qcheck_nsides": sorted(
            {
                int(entry.transfer_nside)
                for entry in directions
                if entry.transfer_role == "diagnostic"
            }
        ),
        "transfer_grid_catalog": catalog,
        "transfer_grid_catalog_sha256": catalog_sha256,
        "direction_rows": direction_rows,
        "direction_ledger_sha256": direction_ledger,
        "horizon_scan_manifest": scan.manifest(),
        "horizon_scan_sha256": scan.manifest_sha256(),
        "horizon_scan_crossing_rows": list(scan.crossing_rows),
        "horizon_scan_summary_rows": list(scan.summary_rows),
        "horizon_scan_ledger_sha256": scan.ledger_sha256,
        "horizon_root_pair_rows": pair_rows,
        "horizon_root_pair_ledger_sha256": object_digest(
            "radiosim.mmode-horizon-root-pairs.v1", pair_rows
        ),
        "horizon_slab_rows": slab_rows,
        "horizon_slab_ledger_sha256": object_digest(
            "radiosim.mmode-horizon-slabs.v1", slab_rows
        ),
        "horizon_sign_interval_rows": sign_rows,
        "horizon_sign_interval_ledger_sha256": object_digest(
            "radiosim.mmode-horizon-sign-intervals.v1", sign_rows
        ),
        "horizon_membership_mask_rows": membership_rows,
        "horizon_membership_ledger_sha256": membership_ledger,
        "direct_split_rows": direct["split_rows"],
        "direct_split_ledger_sha256": direct["split_ledger_sha256"],
        "direct_integrand_enclosure_manifest": enclosure_manifest,
        "direct_integrand_enclosure_sha256": enclosure_sha256,
        "sidereal_samples": samples,
        "quadrature_nside": next(
            (
                int(entry.transfer_nside)
                for entry in directions
                if entry.transfer_role == "production"
            ),
            0,
        ),
        "n_baselines": context.n_baselines,
        "n_frequencies": context.n_frequencies,
        "n_correlations": 4,
        "expected_point_direction_count": point_count,
        "evaluated_point_direction_count": point_count,
        "expected_native_healpix_direction_count": native_count,
        "evaluated_native_healpix_direction_count": native_count,
        "expected_production_transfer_direction_count": production_count,
        "evaluated_production_transfer_direction_count": production_count,
        "expected_diagnostic_transfer_direction_count": diagnostic_count,
        "evaluated_diagnostic_transfer_direction_count": diagnostic_count,
        "expected_transfer_quadrature_direction_count": transfer_count,
        "evaluated_transfer_quadrature_direction_count": transfer_count,
        "expected_direction_count": len(directions),
        "evaluated_direction_count": len(directions),
        "expected_phase_comparison_count": phase["expected"],
        "evaluated_phase_comparison_count": phase["evaluated"],
        "expected_horizon_trajectory_count": len(directions),
        "evaluated_horizon_trajectory_count": len(frozen),
        "expected_horizon_root_pair_row_count": len(directions),
        "evaluated_horizon_root_pair_row_count": len(pair_rows),
        "expected_horizon_membership_count": len(directions) * samples,
        "evaluated_horizon_membership_count": sum(
            int(entry["sample_count"]) for entry in membership_rows
        ),
        "expected_direct_exposure_split_count": direct_directions * samples,
        "evaluated_direct_exposure_split_count": direct["exposure_split_count"],
        "expected_direct_split_row_count": len(direct["split_rows"]),
        "evaluated_direct_split_row_count": len(direct["split_rows"]),
        "expected_frozen_gauss64_node_count": direct["node_totals"]["F64"],
        "evaluated_frozen_gauss64_node_count": direct["node_totals"]["F64"],
        "expected_frozen_gauss128_node_count": direct["node_totals"]["F128"],
        "evaluated_frozen_gauss128_node_count": direct["node_totals"]["F128"],
        "expected_operational_gauss64_node_count": direct["node_totals"]["O64"],
        "evaluated_operational_gauss64_node_count": direct["node_totals"]["O64"],
        "expected_operational_gauss128_node_count": direct["node_totals"]["O128"],
        "evaluated_operational_gauss128_node_count": direct["node_totals"]["O128"],
        "horizon_isolation_interval_count": scan.isolation_interval_count,
        "horizon_unresolved_interval_count": 0,
        "expected_horizon_slab_row_count": paired,
        "evaluated_horizon_slab_row_count": len(slab_rows),
        "expected_horizon_sign_interval_count": len(sign_rows),
        "evaluated_horizon_sign_interval_count": len(sign_rows),
        "horizon_root_count_mismatches": root_count_mismatches,
        "horizon_root_orientation_mismatches": orientation_mismatches,
        "horizon_membership_mismatches": membership_mismatches,
        "horizon_outside_slab_sign_mismatches": sign_mismatches,
        "horizon_paired_root_count": paired,
        "horizon_mismatch_slab_count": len(slab_rows),
        "horizon_mismatch_measure_turn": canonical_rational(mismatch_measure_turn),
        "horizon_mismatch_measure_rad": _round_up_fraction(
            exact_tau * mismatch_measure_turn
        ),
        "horizon_mismatch_measure_limit_rad": FRAME_ROOT_LIMIT_RAD * paired,
        "horizon_root_max_rad": root_max_rad,
        "horizon_root_limit_rad": FRAME_ROOT_LIMIT_RAD,
        "phase_max_rad": phase["phase_max_rad"],
        "phase_limit_rad": FRAME_PHASE_LIMIT_RAD,
        "expected_cube_cell_count": cells,
        "evaluated_frozen_gauss64_cube_cell_count": int(direct["F64"].size),
        "evaluated_frozen_gauss128_cube_cell_count": int(direct["F128"].size),
        "evaluated_operational_gauss64_cube_cell_count": int(direct["O64"].size),
        "evaluated_operational_gauss128_cube_cell_count": int(direct["O128"].size),
        "compared_frozen_gauss_change_cell_count": int(direct["F128"].size),
        "compared_operational_gauss_change_cell_count": int(direct["O128"].size),
        "evaluated_frozen_enclosure_error_cell_count": int(direct["EF"].size),
        "evaluated_operational_enclosure_error_cell_count": int(direct["EO"].size),
        "frozen_gauss64_cube_sha256": _cube_identity(direct["F64"], "frozen_gauss64"),
        "frozen_gauss128_cube_sha256": _cube_identity(
            direct["F128"], "frozen_gauss128"
        ),
        "operational_gauss64_cube_sha256": _cube_identity(
            direct["O64"], "operational_gauss64"
        ),
        "operational_gauss128_cube_sha256": _cube_identity(
            direct["O128"], "operational_gauss128"
        ),
        "frozen_enclosure_error_cube_sha256": _error_cube_identity(direct["EF"]),
        "operational_enclosure_error_cube_sha256": array_digest(
            "radiosim.mmode-direct-root-error.v1",
            "operational_enclosure_error",
            list(_CUBE_AXES),
            "Jy",
            direct["EO"],
            dtype="float64-be",
        ),
        "direct_gauss_scale_jy": scale_q,
        "frozen_gauss_change_max_jy": change_frozen,
        "operational_gauss_change_max_jy": change_operational,
        "direct_gauss_change_max_jy": max(change_frozen, change_operational),
        "direct_gauss_change_limit_jy": 1e-11 * scale_q,
        "cube_scale_jy": scale_frame,
        "cube_max_jy": cube_max,
        "cube_limit_jy": 5e-5 * scale_frame + 1e-10,
        "cube_l2": cube_l2,
        "cube_l2_limit": 5e-5,
        "direction_diagnostic_max_rad": phase["direction_diagnostic_max_rad"],
        "direction_diagnostic_argmax_id": phase["direction_diagnostic_argmax_id"],
        "direction_diagnostic_argmax_phase": phase["direction_diagnostic_argmax_phase"],
        "basis_diagnostic_max_rad": phase["basis_diagnostic_max_rad"],
        "basis_diagnostic_argmax_id": phase["basis_diagnostic_argmax_id"],
        "basis_diagnostic_argmax_phase": phase["basis_diagnostic_argmax_phase"],
    }
    passed = (
        root_count_mismatches == 0
        and orientation_mismatches == 0
        and membership_mismatches == 0
        and sign_mismatches == 0
        and row["horizon_unresolved_interval_count"] == 0
        and root_max_rad <= FRAME_ROOT_LIMIT_RAD
        and row["horizon_mismatch_measure_rad"]
        <= row["horizon_mismatch_measure_limit_rad"]
        and phase["phase_max_rad"] <= FRAME_PHASE_LIMIT_RAD
        and row["direct_gauss_change_max_jy"] <= row["direct_gauss_change_limit_jy"]
        and cube_max <= row["cube_limit_jy"]
        and cube_l2 <= 5e-5
    )
    digest = certificate_identity(row)
    row_with_identity = dict(row)
    row_with_identity["certificate_sha256"] = digest
    return FrameCertificate(
        certificate_sha256=digest,
        row=row_with_identity,
        frozen_gauss64=direct["F64"],
        frozen_gauss128=direct["F128"],
        operational_gauss64=direct["O64"],
        operational_gauss128=direct["O128"],
        frozen_enclosure_error=direct["EF"],
        operational_enclosure_error=direct["EO"],
        passed=passed,
    )


def _direction_rows(
    directions: Sequence[LedgerDirection], context: KernelContext
) -> tuple[list[dict[str, Any]], str]:
    """Return Section 12.1's canonical direction ledger rows and their digest."""
    frequencies = [f64be(float(value)) for value in context.frequencies_hz]
    rows: list[dict[str, Any]] = []
    for row in directions:
        vector = np.asarray(row.cirs_direction, dtype=np.float64)
        stokes = np.asarray(row.resolved_stokes_iau, dtype=np.float64)
        if not np.all(np.isfinite(stokes)):
            raise ValueError("a direction payload is not finite")
        manifest = {
            "schema_version": "radiosim.mmode-direction-input.v1",
            "direction_id": row.direction_id,
            "source_kind": row.source_kind,
            "component_index": row.component_index,
            "source_index": row.source_index,
            "transfer_role": row.transfer_role,
            "transfer_nside": row.transfer_nside,
            "cirs_direction_f64be": [f64be(float(value)) for value in vector],
            "run_frequency_hz_f64be": frequencies,
            "active_frequency_mask": [bool(flag) for flag in row.active_frequency_mask],
            "resolved_stokes_iau_f64be": [
                f64be(float(value)) for value in stokes.reshape(-1)
            ],
            "integration_weight_f64be": f64be(float(row.integration_weight)),
        }
        rows.append(
            {
                "direction_id": row.direction_id,
                "source_kind": row.source_kind,
                "component_index": row.component_index,
                "source_index": row.source_index,
                "transfer_role": row.transfer_role,
                "transfer_nside": row.transfer_nside,
                "cirs_direction_sha256": array_digest(
                    "radiosim.mmode-cirs-direction.v1",
                    "cirs_direction",
                    ["cartesian"],
                    "dimensionless",
                    vector,
                    dtype="float64-be",
                ),
                "active_frequency_mask": [
                    bool(flag) for flag in row.active_frequency_mask
                ],
                "active_frequency_count": int(sum(row.active_frequency_mask)),
                "direction_input_sha256": object_digest(
                    "radiosim.mmode-direction-input.v1", manifest
                ),
            }
        )
    return rows, object_digest("radiosim.mmode-direction-ledger.v1", rows)


def direction_input_manifest(
    row: LedgerDirection, frequencies_hz: Sequence[float]
) -> dict[str, Any]:
    """Return Section 12's complete direction-input preimage for one row."""
    vector = np.asarray(row.cirs_direction, dtype=np.float64)
    stokes = np.asarray(row.resolved_stokes_iau, dtype=np.float64)
    if not np.all(np.isfinite(stokes)) or not np.all(np.isfinite(vector)):
        raise ValueError("a direction payload is not finite")
    return {
        "schema_version": "radiosim.mmode-direction-input.v1",
        "direction_id": row.direction_id,
        "source_kind": row.source_kind,
        "component_index": row.component_index,
        "source_index": row.source_index,
        "transfer_role": row.transfer_role,
        "transfer_nside": row.transfer_nside,
        "cirs_direction_f64be": [f64be(float(value)) for value in vector],
        "run_frequency_hz_f64be": [f64be(float(value)) for value in frequencies_hz],
        "active_frequency_mask": [bool(flag) for flag in row.active_frequency_mask],
        "resolved_stokes_iau_f64be": [
            f64be(float(value)) for value in stokes.reshape(-1)
        ],
        "integration_weight_f64be": f64be(float(row.integration_weight)),
    }


def _transfer_catalog(
    directions: Sequence[LedgerDirection],
) -> tuple[list[dict[str, Any]], str]:
    """Return Section 12.1's transfer-grid catalogue and its digest."""
    groups: list[tuple[str, int]] = []
    members: dict[tuple[str, int], list[str]] = {}
    for row in directions:
        if row.source_kind != "transfer_quadrature":
            continue
        key = (row.transfer_role, int(row.transfer_nside))
        if key not in members:
            members[key] = []
            groups.append(key)
        members[key].append(row.direction_id)
    production = [key for key in groups if key[0] == "production"]
    diagnostic = sorted(
        (key for key in groups if key[0] == "diagnostic"), key=lambda key: key[1]
    )
    catalog = [
        {
            "transfer_grid_id": f"{role}:{nside}",
            "transfer_role": role,
            "transfer_nside": nside,
            "expected_direction_count": 12 * nside**2,
            "evaluated_direction_count": len(members[(role, nside)]),
            "direction_id_ledger_sha256": object_digest(
                "radiosim.mmode-transfer-grid-direction-ids.v1",
                members[(role, nside)],
            ),
        }
        for role, nside in (*production, *diagnostic)
    ]
    return catalog, object_digest("radiosim.mmode-transfer-grid-catalog.v1", catalog)


def _enclosure_manifest(
    *, input_identity_sha256: str, frame_matrix_sha256: str
) -> tuple[dict[str, Any], str]:
    """Return Section 12.1's direct-integrand enclosure manifest and digest."""
    root = Path(__file__).resolve().parents[4]
    implementation = [
        {
            "path": path,
            "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
        }
        for path in sorted(
            (
                "src/radiosim/core/mmode/frame.py",
                "src/radiosim/core/mmode/solver.py",
                "src/radiosim/core/mmode/transfer.py",
            )
        )
    ]
    constants = [
        ("coherency_half_factor", "binary64", f64be(0.5)),
        ("enclosure_accumulation_rounding", "literal", "toward_positive_infinity"),
        ("fringe_operator_norm_ceiling", "binary64", f64be(1.0)),
        ("gauss_order_high", "integer", "128"),
        ("gauss_order_low", "integer", "64"),
        ("hadamard_factor_norm_ceiling", "binary64", f64be(1.0)),
        (
            "magnitude_ceiling_rounding",
            "literal",
            "nextafter_toward_positive_infinity",
        ),
        ("root_cell_nominal_contribution", "literal", "exact_complex_zero"),
        (
            "rectangle_form",
            "literal",
            "[-G_abs,G_abs,-G_abs,G_abs]",
        ),
    ]
    manifest = {
        "schema_version": "radiosim.mmode-direct-integrand-enclosure.v1",
        "algorithm_id": "radiosim.mmode-direct-integrand-enclosure.v1",
        "implementation_files": implementation,
        "constant_rows": [
            {"name": name, "type": kind, "value": value}
            for name, kind, value in sorted(constants)
        ],
        "input_identity_sha256": input_identity_sha256,
        "frame_matrix_sha256": frame_matrix_sha256,
    }
    return manifest, object_digest(
        "radiosim.mmode-direct-integrand-enclosure.v1", manifest
    )


def _sign_intervals(
    directions: Sequence[LedgerDirection],
    frozen: Sequence[FrozenHorizonTrajectory],
    operational: Sequence[tuple[HorizonRootEnclosure, ...]],
    slabs: Mapping[str, Sequence[tuple[Fraction, Fraction]]],
    grid: CanonicalEraGrid,
    evaluator: Any,
) -> tuple[list[dict[str, Any]], int]:
    """Compare strict signs on every outside-slab complement piece.

    The union of both models' root sets splits the cycle; completeness makes
    each model's sign constant on every resulting open interval, so the strict
    signs are compared at each interval's deterministic interior midpoint,
    outside the paired-root mismatch slabs.

    Section 4.2 requires the operational sign to come from the same public-API
    evaluator the scan consumes; the midpoints are not scan boundaries, so they
    are evaluated in one batch through that same object rather than through the
    frozen attitude.
    """
    horizon_lo, horizon_hi = grid.horizon_domain
    planned: list[tuple[int, int, Fraction, Fraction, Fraction]] = []
    for index, row in enumerate(directions):
        cuts = {horizon_lo, horizon_hi}
        for root in frozen[index].roots:
            for bound in (root.turn_lo, root.turn_hi):
                if horizon_lo < bound < horizon_hi:
                    cuts.add(bound)
        for root in operational[index]:
            for bound in root.ambiguous_span:
                if horizon_lo < bound < horizon_hi:
                    cuts.add(bound)
        for lower, upper in slabs.get(row.direction_id, ()):
            for bound in (lower, upper):
                if horizon_lo < bound < horizon_hi:
                    cuts.add(bound)
        ordered = sorted(cuts)
        interval_index = 0
        pieces = slabs.get(row.direction_id, ())
        for position in range(len(ordered) - 1):
            lower, upper = ordered[position], ordered[position + 1]
            if upper <= lower:
                continue
            middle = (lower + upper) / 2
            if any(start <= middle <= stop for start, stop in pieces):
                continue
            planned.append((index, interval_index, lower, upper, middle))
            interval_index += 1

    observed = (
        evaluator.at_pairs(
            [entry[0] for entry in planned], [entry[4] for entry in planned]
        )
        if planned
        else np.zeros(0, dtype=np.float64)
    )
    rows: list[dict[str, Any]] = []
    mismatches = 0
    for position, (index, interval_index, lower, upper, middle) in enumerate(planned):
        frozen_value = float(frozen[index].value(float(middle)))
        operational_value = float(observed[position])
        frozen_sign = 1 if frozen_value > 0.0 else -1
        operational_sign = 1 if operational_value > 0.0 else -1
        match = frozen_sign == operational_sign
        if not match:
            mismatches += 1
        rows.append(
            {
                "direction_id": directions[index].direction_id,
                "interval_index": interval_index,
                "turn_lo": canonical_rational(lower),
                "turn_hi": canonical_rational(upper),
                "midpoint_turn": canonical_rational(middle),
                "midpoint_rad_f64be": f64be(float(middle) * TAU),
                "frozen_sign": frozen_sign,
                "operational_sign": operational_sign,
                "match": match,
            }
        )
    return rows, mismatches


def _phase_census(
    *,
    grid: CanonicalEraGrid,
    frame: FrozenFrame,
    context: KernelContext,
    directions: Sequence[LedgerDirection],
    frozen: Sequence[FrozenHorizonTrajectory],
    operational_roots: Sequence[tuple[HorizonRootEnclosure, ...]],
) -> dict[str, Any]:
    """Compare frozen and operational fringe phase at every phase node.

    ``P_d`` is the exact ordered phase-node ledger of direction ``d``: the
    retained sample centres, the exposure boundaries, and both models'
    root-enclosure endpoints.  Every ``(baseline, frequency)`` cell of every node
    is compared, which is what ``expected_phase_comparison_count`` counts.
    """
    from radiosim.core.mmode.time import installed_iers_context

    samples = grid.sidereal_samples
    shared: set[Fraction] = set()
    for index in range(samples):
        shared.add(grid.center_turn(index))
        lower, upper = grid.exposure_turns(index)
        shared.add(lower)
        shared.add(upper)
    horizon_lo, horizon_hi = grid.horizon_domain

    nodes: list[list[Fraction]] = []
    for index in range(len(directions)):
        own = set(shared)
        for root in (*frozen[index].roots, *operational_roots[index]):
            for bound in (root.turn_lo, root.turn_hi):
                if horizon_lo <= bound <= horizon_hi:
                    own.add(bound)
        nodes.append(sorted(own))

    expected = (
        context.n_baselines * context.n_frequencies * sum(len(entry) for entry in nodes)
    )
    common = sorted(shared)
    wavenumber = TAU * context.frequencies_hz / _SPEED_OF_LIGHT_M_PER_S
    baselines = np.asarray(context.baseline_vectors_enu_m, dtype=np.float64)

    trajectory = _operational_directions(frame, directions)
    evaluated = 0
    phase_max = 0.0
    direction_max = 0.0
    direction_argmax = ("", "0/1")
    basis_max = 0.0
    basis_argmax = ("", "0/1")
    with installed_iers_context():
        for turn in common:
            frozen_enu = frozen_enu_at_phase(
                frame,
                np.stack([row.cirs_direction for row in directions]),
                float(turn) * TAU,
            )
            operational_enu = trajectory(turn)
            delta = frozen_enu - operational_enu
            projected = delta @ baselines.T
            phases = np.abs(projected[:, :, None] * wavenumber[None, None, :])
            if phases.size:
                phase_max = max(phase_max, float(np.max(phases)))
                evaluated += int(phases.size)
                separation = np.arccos(
                    np.clip(np.sum(frozen_enu * operational_enu, axis=1), -1.0, 1.0)
                )
                position = int(np.argmax(separation))
                if float(separation[position]) > direction_max:
                    direction_max = float(separation[position])
                    direction_argmax = (
                        directions[position].direction_id,
                        canonical_rational(turn),
                    )
                rotation = _tangent_rotation(frozen_enu, operational_enu)
                position = int(np.argmax(rotation))
                if float(rotation[position]) > basis_max:
                    basis_max = float(rotation[position])
                    basis_argmax = (
                        directions[position].direction_id,
                        canonical_rational(turn),
                    )
        # The remaining nodes are direction-owned root bounds; every one of them
        # is compared at all ``B*F`` cells too.
        owned: list[tuple[int, Fraction]] = [
            (index, turn)
            for index, entry in enumerate(nodes)
            for turn in entry
            if turn not in shared
        ]
        if owned:
            frozen_enu = np.stack(
                [
                    frozen_enu_at_phase(
                        frame,
                        directions[index].cirs_direction[None, :],
                        float(turn) * TAU,
                    )[0]
                    for index, turn in owned
                ]
            )
            operational_enu = trajectory.at_pairs(
                [index for index, _turn in owned], [turn for _index, turn in owned]
            )
            delta = frozen_enu - operational_enu
            projected = delta @ baselines.T
            phases = np.abs(projected[:, :, None] * wavenumber[None, None, :])
            phase_max = max(phase_max, float(np.max(phases)))
            evaluated += int(phases.size)
    return {
        "expected": expected,
        "evaluated": evaluated,
        "phase_max_rad": phase_max,
        "direction_diagnostic_max_rad": direction_max,
        "direction_diagnostic_argmax_id": direction_argmax[0],
        "direction_diagnostic_argmax_phase": direction_argmax[1],
        "basis_diagnostic_max_rad": basis_max,
        "basis_diagnostic_argmax_id": basis_argmax[0],
        "basis_diagnostic_argmax_phase": basis_argmax[1],
    }


def _tangent_rotation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the tangent-basis rotation between two ``(E, N, U)`` direction sets.

    The local east direction of a topocentric tangent frame is
    ``normalize(up x n)``; the angle between the two frames' east vectors is the
    rotation a polarization basis would undergo, which Section 4.2 records as a
    non-gating attribution diagnostic.
    """
    up = np.asarray([0.0, 0.0, 1.0])
    east_first = np.cross(up, first)
    east_second = np.cross(up, second)
    first_norm = np.linalg.norm(east_first, axis=1)
    second_norm = np.linalg.norm(east_second, axis=1)
    safe = (first_norm > 0.0) & (second_norm > 0.0)
    cosine = np.ones(first.shape[0], dtype=np.float64)
    cosine[safe] = np.sum(east_first[safe] * east_second[safe], axis=1) / (
        first_norm[safe] * second_norm[safe]
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


class _operational_directions:  # noqa: N801 - a callable evaluator, not a type
    """Public-Astropy ``(E, N, U)`` directions for the whole ledger.

    Section 12.1 consumes only the public
    ``SkyCoord.transform_to(AltAz(...))`` values; the altitude and azimuth they
    return are converted to the same ``(East, North, Up)`` triad the frozen model
    reports, so the two are compared in one common frame.
    """

    def __init__(self, frame: FrozenFrame, directions: Sequence[LedgerDirection]):
        from astropy import units as u
        from astropy.coordinates import EarthLocation, SkyCoord

        self._frame = frame
        self._units = u
        self._site = EarthLocation.from_geodetic(
            lon=frame.longitude_deg * u.deg,
            lat=frame.latitude_deg * u.deg,
            height=frame.height_m * u.m,
        )
        self._coords = SkyCoord(
            ra=np.asarray([row.icrs_ra_rad for row in directions]) * u.rad,
            dec=np.asarray([row.icrs_dec_rad for row in directions]) * u.rad,
            frame="icrs",
        )

    def _times(self, turns: Sequence[Fraction]) -> Any:
        from astropy.time import Time

        from radiosim.core.mmode.time import ERA_RATE_TURNS_PER_UT1_DAY

        jd1, jd2 = self._frame.ut1_two_part
        exact = Fraction(jd2)
        second = np.asarray(
            [
                float(exact + Fraction(turn) / ERA_RATE_TURNS_PER_UT1_DAY)
                for turn in turns
            ],
            dtype=np.float64,
        )
        return Time(
            np.full(second.shape, float(jd1), dtype=np.float64),
            second,
            format="jd",
            scale="ut1",
        )

    @staticmethod
    def _triad(altaz: Any, unit: Any) -> np.ndarray:
        altitude = np.atleast_1d(altaz.alt.to_value(unit.rad))
        azimuth = np.atleast_1d(altaz.az.to_value(unit.rad))
        cosine = np.cos(altitude)
        return np.stack(
            (cosine * np.sin(azimuth), cosine * np.cos(azimuth), np.sin(altitude)),
            axis=-1,
        )

    def __call__(self, turn: Fraction) -> np.ndarray:
        from astropy.coordinates import AltAz

        times = self._times([turn])
        altaz = self._coords.transform_to(
            AltAz(obstime=times[0], location=self._site, pressure=0)
        )
        return self._triad(altaz, self._units)

    def at_pairs(self, indices: Sequence[int], turns: Sequence[Fraction]) -> np.ndarray:
        """Return the operational triad for an element-wise batch."""
        from astropy.coordinates import AltAz

        times = self._times(turns)
        altaz = self._coords[np.asarray(indices, dtype=np.int64)].transform_to(
            AltAz(obstime=times, location=self._site, pressure=0)
        )
        return self._triad(altaz, self._units)


def _pair_roots(
    directions: Sequence[LedgerDirection],
    frozen: Sequence[FrozenHorizonTrajectory],
    operational: Sequence[tuple[HorizonRootEnclosure, ...]],
    grid: CanonicalEraGrid,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, float]:
    """Pair frozen and operational roots by orientation and build the slabs."""
    exact_tau = Fraction(*TAU.as_integer_ratio())
    pair_rows: list[dict[str, Any]] = []
    slab_rows: list[dict[str, Any]] = []
    root_max = 0.0
    measure = Fraction(0)
    for index, row in enumerate(directions):
        frozen_roots = frozen[index].roots
        operational_roots = operational[index]
        pairs: list[dict[str, Any]] = []
        mismatch = 0
        for orientation in ("rising", "setting"):
            left = [root for root in frozen_roots if root.orientation == orientation]
            right = [
                root for root in operational_roots if root.orientation == orientation
            ]
            mismatch += abs(len(left) - len(right))
            for first, second in zip(left, right, strict=False):
                lift, delta = _best_lift(first, second)
                delta_rad = _round_up_fraction(exact_tau * delta)
                root_max = max(root_max, delta_rad)
                pairs.append(
                    {
                        "pair_index": len(pairs),
                        "orientation": orientation,
                        "operational_turn_lift": lift,
                        "frozen_root_turn_lo": canonical_rational(first.turn_lo),
                        "frozen_root_turn_hi": canonical_rational(first.turn_hi),
                        "operational_root_turn_lo": canonical_rational(second.turn_lo),
                        "operational_root_turn_hi": canonical_rational(second.turn_hi),
                        "lifted_operational_root_turn_lo": canonical_rational(
                            second.turn_lo + lift
                        ),
                        "lifted_operational_root_turn_hi": canonical_rational(
                            second.turn_hi + lift
                        ),
                        "worst_case_delta_turn": canonical_rational(delta),
                        "worst_case_delta_rad_f64be": f64be(delta_rad),
                    }
                )
                low = min(first.turn_lo, second.turn_lo + lift)
                high = max(first.turn_hi, second.turn_hi + lift)
                measure += high - low
                slab_rows.append(
                    {
                        "direction_id": row.direction_id,
                        "pair_index": pairs[-1]["pair_index"],
                        "orientation": orientation,
                        "operational_turn_lift": lift,
                        "worst_case_delta_turn": canonical_rational(delta),
                        "worst_case_delta_rad_f64be": f64be(delta_rad),
                        "wraps_seam": False,
                        "pieces": [
                            {
                                "piece_index": 0,
                                "turn_lo": canonical_rational(low),
                                "turn_hi": canonical_rational(high),
                            }
                        ],
                    }
                )
        pair_rows.append(
            {
                "direction_id": row.direction_id,
                "frozen_root_count": len(frozen_roots),
                "operational_root_count": len(operational_roots),
                "orientation_mismatch_count": mismatch,
                "pairs": pairs,
            }
        )
    return pair_rows, slab_rows, root_max, _round_up_fraction(exact_tau * measure)


def _best_lift(
    frozen: HorizonRootEnclosure, operational: HorizonRootEnclosure
) -> tuple[int, Fraction]:
    """Return the unique integer turn lift minimising the worst-case delta."""
    best: tuple[Fraction, int] | None = None
    for lift in (-1, 0, 1):
        delta = max(
            abs((operational.turn_lo + lift) - frozen.turn_hi),
            abs((operational.turn_hi + lift) - frozen.turn_lo),
        )
        if best is None or delta < best[0]:
            best = (delta, lift)
    assert best is not None
    return best[1], best[0]


def _round_up_fraction(value: Fraction) -> float:
    """Return the least binary64 not smaller than an exact rational."""
    nearest = float(value)
    if Fraction(nearest) < value:
        return float(np.nextafter(nearest, math.inf))
    return nearest


def _membership(
    directions: Sequence[LedgerDirection],
    frozen: Sequence[FrozenHorizonTrajectory],
    operational_centre_values: np.ndarray,
    slabs: Mapping[str, Sequence[tuple[Fraction, Fraction]]],
    grid: CanonicalEraGrid,
) -> tuple[list[dict[str, Any]], str, int]:
    """Compare strict ``alt > 0`` membership at every retained sample centre.

    Section 12.1 evaluates one row for every direction and sample centre, and
    ``horizon_membership_ledger_sha256`` digests that complete canonical
    per-sample array -- but the *retained* evidence embeds the compact
    per-direction mask form.  The expansion from masks back to per-sample rows
    is deterministic, because ``sample_turn`` and ``alpha_rad_f64be`` come from
    the one retained grid object, so nothing is lost: only the redundant
    per-row repetition of the grid.

    The expanded array is streamed straight into the digest here and never
    materialized.

    Section 4.2 requires the two models to evaluate membership *independently*:
    the frozen side from its own analytic trajectory and the operational side
    from the same public-API values the Section 12.1 scan consumed -- never
    through the frozen attitude, which would make the counter a comparison of
    one model with itself.  Both sides then apply the one shared strict
    ``alt > 0`` predicate.

    The counter is scoped the way the sign intervals always were: a centre
    inside a paired-root mismatch slab records its models' possibly differing
    memberships into the slab accounting instead of being falsely required to
    agree, so only outside-slab disagreements are counted.  The per-direction
    mask ``mismatch_count`` remains the complete total, which is what the strict
    validator recomputes the outside-slab counter from.

    Returns
    -------
    tuple
        The mask rows, the expanded-array digest, and the outside-slab
        mismatch total.
    """
    samples = grid.sidereal_samples
    centres = [grid.center_turn(index) for index in range(samples)]
    values = np.asarray(operational_centre_values, dtype=np.float64)
    if values.shape != (samples, len(directions)):
        raise ValueError(
            "the operational centre values must be one row per retained sample "
            "centre and one column per direction"
        )

    frozen_bits = np.zeros((len(directions), samples), dtype=bool)
    operational_bits = np.asarray(strict_horizon_visible(values), dtype=bool).T.copy()
    for sample_index, turn in enumerate(centres):
        for index in range(len(directions)):
            frozen_bits[index, sample_index] = bool(
                strict_horizon_visible(frozen[index].value(float(turn)))
            )

    sample_turns = [canonical_rational(turn) for turn in centres]
    alpha = [f64be(grid.alpha_rad[index]) for index in range(samples)]

    mask_rows: list[dict[str, Any]] = []
    mismatches = 0
    for index, row in enumerate(directions):
        matches = frozen_bits[index] == operational_bits[index]
        row_mismatches = int(np.count_nonzero(~matches))
        pieces = slabs.get(row.direction_id, ())
        for sample_index in np.nonzero(~matches)[0].tolist():
            centre = centres[sample_index]
            if not any(start <= centre <= stop for start, stop in pieces):
                mismatches += 1
        mask_rows.append(
            {
                "direction_id": row.direction_id,
                "sample_count": samples,
                "frozen_visible_mask_hex": _visibility_mask_hex(frozen_bits[index]),
                "operational_visible_mask_hex": _visibility_mask_hex(
                    operational_bits[index]
                ),
                "mismatch_count": row_mismatches,
            }
        )

    def chunks() -> Iterator[bytes]:
        yield b"["
        emitted = 0
        for position, entry in enumerate(directions):
            for sample_index in range(samples):
                expanded = {
                    "alpha_rad_f64be": alpha[sample_index],
                    "direction_id": entry.direction_id,
                    "frozen_visible": bool(frozen_bits[position, sample_index]),
                    "match": bool(
                        frozen_bits[position, sample_index]
                        == operational_bits[position, sample_index]
                    ),
                    "operational_visible": bool(
                        operational_bits[position, sample_index]
                    ),
                    "sample_index": sample_index,
                    "sample_turn": sample_turns[sample_index],
                }
                if emitted:
                    yield b","
                yield canonical_json(expanded)
                emitted += 1
        yield b"]"

    ledger = streamed_domain_digest("radiosim.mmode-horizon-membership.v1", chunks)
    return mask_rows, ledger, mismatches


def _visibility_mask_hex(bits: np.ndarray) -> str:
    """Return Section 12.1's sample-ordered visibility mask.

    Lowercase hex of the sample-ordered bits, most significant bit first,
    zero-padded to whole bytes.
    """
    flags = np.asarray(bits, dtype=bool)
    width = (flags.size + 7) // 8
    value = 0
    for flag in flags:
        value = (value << 1) | int(bool(flag))
    value <<= (width * 8 - flags.size) if flags.size else 0
    return value.to_bytes(width, "big").hex() if width else ""


def expand_membership_masks(
    mask_rows: Sequence[Mapping[str, Any]], grid: CanonicalEraGrid
) -> str:
    """Re-digest the per-sample membership array from its retained masks.

    Section 12.1 requires the strict validator to expand the masks, rebuild the
    per-sample array bytes, and re-digest them against the ledger digest.  This
    is that expansion, written once and shared by the runtime and the validator
    so the two cannot drift.
    """
    samples = grid.sidereal_samples
    sample_turns = [
        canonical_rational(grid.center_turn(index)) for index in range(samples)
    ]
    alpha = [f64be(grid.alpha_rad[index]) for index in range(samples)]

    def chunks() -> Iterator[bytes]:
        yield b"["
        emitted = 0
        for row in mask_rows:
            frozen_bits = _mask_bits(str(row["frozen_visible_mask_hex"]), samples)
            operational_bits = _mask_bits(
                str(row["operational_visible_mask_hex"]), samples
            )
            for sample_index in range(samples):
                expanded = {
                    "alpha_rad_f64be": alpha[sample_index],
                    "direction_id": row["direction_id"],
                    "frozen_visible": frozen_bits[sample_index],
                    "match": (
                        frozen_bits[sample_index] == operational_bits[sample_index]
                    ),
                    "operational_visible": operational_bits[sample_index],
                    "sample_index": sample_index,
                    "sample_turn": sample_turns[sample_index],
                }
                if emitted:
                    yield b","
                yield canonical_json(expanded)
                emitted += 1
        yield b"]"

    return streamed_domain_digest("radiosim.mmode-horizon-membership.v1", chunks)


def _mask_bits(mask_hex: str, count: int) -> list[bool]:
    """Decode one Section 12.1 visibility mask back to sample-ordered bits."""
    width = (count + 7) // 8
    if len(mask_hex) != width * 2:
        raise ValueError("visibility mask is not zero-padded to whole bytes")
    value = int(mask_hex, 16) if mask_hex else 0
    value >>= width * 8 - count
    return [bool((value >> (count - 1 - index)) & 1) for index in range(count)]


def _direct_cubes(
    *,
    grid: CanonicalEraGrid,
    frame: FrozenFrame,
    context: KernelContext,
    directions: Sequence[LedgerDirection],
    frozen: Sequence[FrozenHorizonTrajectory],
    operational_roots: Sequence[tuple[HorizonRootEnclosure, ...]],
    beam_peak_ceiling: float,
    input_identity_sha256: str,
    enclosure_manifest_sha256: str,
) -> dict[str, Any]:
    """Evaluate the four horizon-split direct cubes, both error cubes, and the
    Section 12.1 ``direct_split_rows`` ledger they are reduced from.

    One row is emitted for every direction, sample, frequency, baseline,
    correlation and piece of the single shared exact-turn partition.  Each row
    carries the four model/order-qualified contribution-manifest digests and
    the two model-qualified error-preimage digests; the manifests themselves are
    built, hashed and discarded, so the node arrays they authenticate are never
    retained.

    The integrand is the complete Section 6 sum ``sum_X K^X s_X`` over the four
    Shaw fields of the resolved payload.  A component whose resolved payload is
    exactly zero at every frequency contributes exactly zero and is skipped, so
    a Stokes-``I`` sky evaluates precisely the arithmetic it did before the
    polarized fields existed.
    """
    from radiosim.core.polarization import stokes_to_shaw_fields

    samples = grid.sidereal_samples
    shape = (samples, context.n_baselines, context.n_frequencies, 4)
    cubes = {
        name: np.zeros(shape, dtype=np.complex128)
        for name in ("F64", "F128", "O64", "O128")
    }
    errors = {name: np.zeros(shape, dtype=np.float64) for name in ("EF", "EO")}
    nodes64, weights64 = _gauss_legendre(64)
    nodes128, weights128 = _gauss_legendre(128)
    orders = ((64, nodes64, weights64), (128, nodes128, weights128))

    ordered: list[tuple[tuple[int, int, int, int, int, int], dict[str, Any]]] = []
    node_totals = {"F64": 0, "F128": 0, "O64": 0, "O128": 0}
    exposures: set[tuple[str, int]] = set()
    zero = f64be(0.0)
    empty_rectangle = [zero, zero, zero, zero]

    direct_ordinal = -1
    for index, row in enumerate(directions):
        if not row.is_direct_contributor:
            continue
        direct_ordinal += 1
        stokes = np.asarray(row.resolved_stokes_iau, dtype=np.float64)
        # Section 5.2's bridge, through the one shared code object: the kernel
        # cells are ``M (D P^X D) M^H``, so they respond to the *Shaw* fields.
        # Contracting them with an unbridged RadioSim ``U`` would sign-flip
        # exactly one of the four contributions and make this oracle a different
        # sky from the harmonic side.
        shaw = np.stack(
            stokes_to_shaw_fields(
                stokes[:, 0], stokes[:, 1], stokes[:, 2], stokes[:, 3]
            ),
            axis=0,
        )
        payloads = {
            name: shaw[position] * row.integration_weight
            for position, name in enumerate(STOKES_COMPONENT_ORDER)
        }
        payload = payloads["I"]
        components = (
            "I",
            *(
                name
                for name in STOKES_COMPONENT_ORDER[1:]
                if bool(np.any(payloads[name] != 0.0))
            ),
        )
        # Section 12's certified ceiling covers the whole integrand: every
        # ``P^X`` has operator norm ``1/2`` and every resolved receptor matrix
        # is unitary, so ``sum_X |s_X|`` times the certified beam and fringe
        # ceilings bounds ``|sum_X K^X s_X|``.  For a Stokes-``I`` payload the
        # sum is exactly ``|I|`` and the bound is the accepted M1 one.
        total = np.sum(
            np.abs(np.stack([payloads[name] for name in STOKES_COMPONENT_ORDER])),
            axis=0,
        )
        magnitude = float(np.max(total)) if total.size else 0.0
        ceiling = magnitude_ceiling(
            payload_magnitude=magnitude,
            factor_ceilings=(0.5, beam_peak_ceiling**2, 1.0),
        )
        active = tuple(bool(value != 0.0) for value in total)
        for sample_index in range(samples):
            exposures.add((row.direction_id, sample_index))
            lower, upper = grid.exposure_turns(sample_index)
            width = upper - lower
            cuts = _piece_cuts(
                lower, upper, frozen[index].roots, operational_roots[index]
            )
            piece_index = 0
            for piece in range(len(cuts) - 1):
                piece_lo, piece_hi = cuts[piece], cuts[piece + 1]
                if piece_hi <= piece_lo:
                    continue
                frozen_class = _classify_piece(frozen[index], piece_lo, piece_hi)
                # Outside every enclosure of both models the two integrands
                # coincide -- Section 12.1's sign census proves it -- so the
                # operational class differs only inside an operational root.
                operational_class = (
                    "root_enclosure"
                    if _inside_enclosure(operational_roots[index], piece_lo, piece_hi)
                    else frozen_class
                )
                radius = ceiling * float((piece_hi - piece_lo) / width)
                half = (piece_hi - piece_lo) / 2
                middle = (piece_hi + piece_lo) / 2

                integrands: dict[int, np.ndarray] = {}
                applied: dict[int, np.ndarray] = {}
                turn_arrays: dict[int, np.ndarray] = {}
                if "smooth_above" in (frozen_class, operational_class):
                    for order, nodes, weights in orders:
                        turns = np.asarray(
                            [float(middle) + float(half) * node for node in nodes],
                            dtype=np.float64,
                        )
                        enu = np.stack(
                            [
                                frozen_enu_at_phase(
                                    frame, row.cirs_direction[None, :], turn * TAU
                                )[0]
                                for turn in turns
                            ]
                        )
                        kernels = section6_kernel(
                            context, enu, stokes_fields=components
                        )
                        integrand = kernels["I"] * payload[None, None, :, None]
                        for name in components[1:]:
                            integrand = integrand + (
                                kernels[name] * payloads[name][None, None, :, None]
                            )
                        integrands[order] = integrand
                        applied[order] = weights * float(half) / float(width)
                        turn_arrays[order] = turns

                for model, classification, error_cube in (
                    ("frozen", frozen_class, "EF"),
                    ("operational", operational_class, "EO"),
                ):
                    if classification == "root_enclosure":
                        for frequency in range(context.n_frequencies):
                            if active[frequency]:
                                errors[error_cube][sample_index, :, frequency, :] += (
                                    radius
                                )
                    elif classification == "smooth_above":
                        for order, _nodes, _weights in orders:
                            name = ("F" if model == "frozen" else "O") + str(order)
                            contribution = (
                                integrands[order] * applied[order][:, None, None, None]
                            )
                            cubes[name][sample_index] += np.sum(contribution, axis=0)

                for frequency in range(context.n_frequencies):
                    for baseline in range(context.n_baselines):
                        for correlation in range(4):
                            key = (
                                direct_ordinal,
                                sample_index,
                                frequency,
                                baseline,
                                correlation,
                                piece_index,
                            )
                            ordered.append(
                                (
                                    key,
                                    _direct_split_row(
                                        grid=grid,
                                        input_identity_sha256=input_identity_sha256,
                                        enclosure_manifest_sha256=(
                                            enclosure_manifest_sha256
                                        ),
                                        direction_id=row.direction_id,
                                        sample_index=sample_index,
                                        frequency_index=frequency,
                                        baseline_index=baseline,
                                        correlation_index=correlation,
                                        piece_index=piece_index,
                                        turn_lo=piece_lo,
                                        turn_hi=piece_hi,
                                        payload_active=active[frequency],
                                        frozen_class=frozen_class,
                                        operational_class=operational_class,
                                        integrands=integrands,
                                        applied=applied,
                                        turn_arrays=turn_arrays,
                                        ceiling=ceiling,
                                        radius=radius,
                                        empty_rectangle=empty_rectangle,
                                        node_totals=node_totals,
                                    ),
                                )
                            )
                piece_index += 1
    ordered.sort(key=lambda entry: entry[0])
    split_rows = [entry[1] for entry in ordered]
    return {
        **cubes,
        **errors,
        "split_rows": split_rows,
        "split_ledger_sha256": object_digest(
            "radiosim.mmode-direct-split-ledger.v1", split_rows
        ),
        "exposure_split_count": len(exposures),
        "node_totals": node_totals,
    }


def _inside_enclosure(
    roots: Sequence[HorizonRootEnclosure], lower: Fraction, upper: Fraction
) -> bool:
    """Return whether a piece meets one crossing's enclosure-plus-guards union.

    Section 12 makes every piece inside that closed union a
    ``root_enclosure``-class piece for the error-disk rule, so a guard is
    certified-bounded physically rather than assumed empty.
    """
    for root in roots:
        span_lo, span_hi = root.ambiguous_span
        if span_lo < upper and lower < span_hi:
            return True
    return False


def _direct_split_row(
    *,
    grid: CanonicalEraGrid,
    input_identity_sha256: str,
    enclosure_manifest_sha256: str,
    direction_id: str,
    sample_index: int,
    frequency_index: int,
    baseline_index: int,
    correlation_index: int,
    piece_index: int,
    turn_lo: Fraction,
    turn_hi: Fraction,
    payload_active: bool,
    frozen_class: str,
    operational_class: str,
    integrands: Mapping[int, np.ndarray],
    applied: Mapping[int, np.ndarray],
    turn_arrays: Mapping[int, np.ndarray],
    ceiling: float,
    radius: float,
    empty_rectangle: Sequence[str],
    node_totals: dict[str, int],
) -> dict[str, Any]:
    """Build one Section 12.1 ``direct_split_rows`` entry and its six digests."""
    shared = {
        "input_identity_sha256": input_identity_sha256,
        "canonical_era_grid_sha256": grid.canonical_era_grid_sha256,
        "direction_id": direction_id,
        "sample_index": sample_index,
        "frequency_index": frequency_index,
        "baseline_index": baseline_index,
        "correlation_index": correlation_index,
        "piece_index": piece_index,
        "turn_lo": canonical_rational(turn_lo),
        "turn_hi": canonical_rational(turn_hi),
        "payload_active": payload_active,
    }
    row: dict[str, Any] = {
        "direction_id": direction_id,
        "sample_index": sample_index,
        "frequency_index": frequency_index,
        "baseline_index": baseline_index,
        "correlation_index": correlation_index,
        "piece_index": piece_index,
        "turn_lo": canonical_rational(turn_lo),
        "turn_hi": canonical_rational(turn_hi),
        "payload_active": payload_active,
        "frozen_piece_class": frozen_class,
        "operational_piece_class": operational_class,
    }
    for model, classification in (
        ("frozen", frozen_class),
        ("operational", operational_class),
    ):
        prefix = "frozen" if model == "frozen" else "operational"
        for order in (64, 128):
            evaluated = payload_active and classification == "smooth_above"
            if evaluated:
                turns = turn_arrays[order]
                weights = applied[order]
                values = integrands[order][
                    :, baseline_index, frequency_index, correlation_index
                ]
                total = complex(np.sum(values * weights))
                node_turns = [
                    canonical_rational(Fraction(float(turn))) for turn in turns
                ]
                node_radians = [f64be(float(turn) * TAU) for turn in turns]
                weight_values = [f64be(float(weight)) for weight in weights]
                integrand = [
                    f64be(part)
                    for value in values
                    for part in (float(value.real), float(value.imag))
                ]
                count = int(turns.size)
            else:
                total = 0j
                node_turns = []
                node_radians = []
                weight_values = []
                integrand = []
                count = 0
            manifest = {
                "schema_version": "radiosim.mmode-direct-piece-cell.v1",
                "model": model,
                "gauss_order": order,
                **shared,
                "piece_class": classification,
                "node_turns": node_turns,
                "node_radians_f64be": node_radians,
                "weights_f64be": weight_values,
                "integrand_reim_f64be": integrand,
                "contribution_real_f64be": f64be(total.real),
                "contribution_imag_f64be": f64be(total.imag),
            }
            row[f"{prefix}_gauss{order}_node_count"] = count
            row[f"{prefix}_gauss{order}_contribution_sha256"] = object_digest(
                "radiosim.mmode-direct-piece-cell.v1", manifest
            )
            node_totals[("F" if model == "frozen" else "O") + str(order)] += count
        ambiguous = payload_active and classification == "root_enclosure"
        error = radius if ambiguous else 0.0
        rectangle = (
            [f64be(-ceiling), f64be(ceiling), f64be(-ceiling), f64be(ceiling)]
            if ambiguous
            else list(empty_rectangle)
        )
        preimage = {
            "schema_version": "radiosim.mmode-direct-piece-error.v1",
            "model": model,
            **shared,
            "piece_class": classification,
            "direct_integrand_enclosure_sha256": enclosure_manifest_sha256,
            "integrand_rectangle_f64be": rectangle,
            "enclosure_error_f64be": f64be(error),
        }
        row[f"{prefix}_enclosure_error_f64be"] = f64be(error)
        row[f"{prefix}_enclosure_error_preimage_sha256"] = object_digest(
            "radiosim.mmode-direct-piece-error.v1", preimage
        )
    return row


def _piece_cuts(
    lower: Fraction,
    upper: Fraction,
    frozen_roots: Sequence[HorizonRootEnclosure],
    operational_roots: Sequence[HorizonRootEnclosure],
) -> tuple[Fraction, ...]:
    """Return the one common exact-turn partition of a single exposure.

    Section 12 cuts at every frozen root-enclosure bound and at every
    operational root-enclosure *or guard-interval* bound: an operational
    crossing's ambiguous region is the closed union of its enclosure and its
    flanking guards, so any structure the ceiling rule could not exclude beside
    a shallow crossing falls inside a piece the error disk covers rather than
    inside a smooth rule.
    """
    cuts = {lower, upper}
    for root in frozen_roots:
        for bound in (root.turn_lo, root.turn_hi):
            if lower < bound < upper:
                cuts.add(bound)
    for root in operational_roots:
        for bound in root.ambiguous_span:
            if lower < bound < upper:
                cuts.add(bound)
    return tuple(sorted(cuts))


# ---------------------------------------------------------------------------
# Section 6 forward path
# ---------------------------------------------------------------------------


def transfer_sample_rows(
    *,
    context: KernelContext,
    frame: FrozenFrame,
    catalog: Sequence[Mapping[str, Any]],
    production_table: Any,
    diagnostic_table: Any,
) -> list[dict[str, Any]]:
    """Return Section 7.3's ``transfer_sample_rows``, one per grid and cell.

    Each row's ``concatenation_sha256`` is the Section 14 ``A`` identity over the
    direction-ledger-ordered concatenation of every catalogued direction's packed
    contribution vector for that cell.  The concatenation is built one grid at a
    time from the same per-node weighting the production transform accumulates,
    so a catalogued node cannot be omitted, reordered or substituted while
    preserving the digest, and no per-direction row array is retained.

    A scalar block table evaluates the ``I`` field only; the three remaining
    fixed-order fields contribute exact complex zero and are recorded as such
    rather than omitted, because Section 7.3's row set is the complete
    ``(1+len(Q_diag))*B*F*C*4`` product.  A four-field table evaluates each
    field into exactly its own packed columns, leaving the other fields' columns
    exactly zero in that field's contribution array.
    """
    from radiosim.core.mmode.harmonics import (
        field_columns,
        packed_conjugate_harmonics,
        packed_polarized_conjugate_harmonics,
    )
    from radiosim.core.mmode.transfer import quadrature_grid

    rows: list[dict[str, Any]] = []
    for entry in catalog:
        nside = int(entry["transfer_nside"])
        table = (
            production_table
            if entry["transfer_role"] == "production"
            else diagnostic_table
        )
        polarized = _is_polarized_table(table)
        nodes, weights = quadrature_grid(nside)
        enu = frozen_enu_at_phase(frame, nodes, 0.0)
        theta = np.arccos(np.clip(nodes[:, 2], -1.0, 1.0))
        phi = np.mod(np.arctan2(nodes[:, 1], nodes[:, 0]), 2.0 * math.pi)
        directions = int(nodes.shape[0])
        empty = np.zeros((directions, table.packed_value_count), dtype=np.complex128)
        if polarized:
            harmonics = np.conjugate(
                packed_polarized_conjugate_harmonics(table, theta, phi)
            )
            fields = field_integrands(
                section6_kernel(context, enu, stokes_fields=STOKES_COMPONENT_ORDER)
            )
            weighted_fields = {
                name: fields[name] * weights[:, None, None, None]
                for name in FIELD_ORDER
            }
            columns_by_field = {
                name: field_columns(table, name) for name in FIELD_ORDER
            }
        else:
            harmonics = np.conjugate(packed_conjugate_harmonics(table, theta, phi))
            weighted = (
                section6_kernel(context, enu)["I"] * (weights[:, None, None, None])
            )
        for baseline in range(context.n_baselines):
            for frequency in range(context.n_frequencies):
                for correlation in range(4):
                    for field_index, field_name in enumerate(FIELD_ORDER):
                        if polarized:
                            contribution = np.zeros(
                                (directions, table.packed_value_count),
                                dtype=np.complex128,
                            )
                            columns = columns_by_field[field_name]
                            if columns.size:
                                contribution[:, columns] = (
                                    weighted_fields[field_name][
                                        :, baseline, frequency, correlation
                                    ][:, None]
                                    * harmonics[:, columns]
                                )
                        else:
                            contribution = (
                                weighted[:, baseline, frequency, correlation][:, None]
                                * harmonics
                                if field_index == 0
                                else empty
                            )
                        rows.append(
                            {
                                "grid_id": str(entry["transfer_grid_id"]),
                                "baseline_index": baseline,
                                "frequency_index": frequency,
                                "correlation_index": correlation,
                                "field_index": field_index,
                                "field_name": field_name,
                                "resolved_lmax": int(table.lmax),
                                "resolved_mmax": int(table.mmax),
                                "block_table_sha256": table.block_table_sha256,
                                "direction_count": directions,
                                "packed_sample_value_count": int(
                                    directions * table.packed_value_count
                                ),
                                "concatenation_sha256": array_digest(
                                    "radiosim.mmode-transfer-sample-contribution.v1",
                                    "transfer_sample_contribution",
                                    ["direction", "packed_value"],
                                    "visibility_response_sr",
                                    contribution,
                                    dtype="complex128-be",
                                ),
                            }
                        )
    return rows


def build_production_transfer(
    *,
    context: KernelContext,
    frame: FrozenFrame,
    nside: int,
    table: Any,
    horizon: bool = True,
) -> np.ndarray:
    r"""Return ``B^X_{pqfc,lm}`` on the Section 7.3 iso-Gauss grid.

    ``B_lm = integral(K Y_lm dOmega)`` -- the harmonic is *unconjugated* against
    the conjugated sky expansion, so ``sum_lm B_lm a_lm`` reproduces
    ``integral(K I dOmega)``.  The grid lives in the frozen CIRS frame, whose
    polar axis is the rotation axis, which is what makes Section 4.1's rigid
    group composition give ``B_lm(alpha) = B_lm(0) exp(+i m alpha)`` exactly.

    A scalar block table selects Section 6's ``B^I`` alone.  The four-field
    table of Section 5.3 selects the complete polarized set, with

    .. math::

        B^{(+2)}=\int(K^Q-iK^U)\,{}_{+2}Y_{lm},\qquad
        B^{(-2)}=\int(K^Q+iK^U)\,{}_{-2}Y_{lm},

    each field written into exactly its own packed columns.  The kernel is
    evaluated on the terrestrial directions -- the horizon predicate, the fringe
    and the beam are the only factors that see them -- while every harmonic is
    evaluated on the celestial angles of the same node, which is the basis
    Section 6 anchors the constant receptor cells to.
    """
    from radiosim.core.mmode.harmonics import (
        field_columns,
        packed_conjugate_harmonics,
        packed_polarized_conjugate_harmonics,
    )
    from radiosim.core.mmode.transfer import quadrature_grid

    nodes, weights = quadrature_grid(nside)
    enu = frozen_enu_at_phase(frame, nodes, 0.0)
    theta = np.arccos(np.clip(nodes[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(nodes[:, 1], nodes[:, 0]), 2.0 * math.pi)
    if not _is_polarized_table(table):
        kernel = section6_kernel(context, enu, horizon=horizon)["I"]
        harmonics = np.conjugate(packed_conjugate_harmonics(table, theta, phi))
        weighted = kernel * weights[:, None, None, None]
        return np.einsum("nbfc,np->bfcp", weighted, harmonics)

    fields = field_integrands(
        section6_kernel(
            context, enu, horizon=horizon, stokes_fields=STOKES_COMPONENT_ORDER
        )
    )
    harmonics = np.conjugate(packed_polarized_conjugate_harmonics(table, theta, phi))
    packed = np.zeros(
        (
            context.n_baselines,
            context.n_frequencies,
            4,
            int(table.packed_value_count),
        ),
        dtype=np.complex128,
    )
    for name in table.field_order:
        columns = field_columns(table, name)
        if columns.size == 0:
            continue
        weighted = fields[name] * weights[:, None, None, None]
        packed[:, :, :, columns] = np.einsum(
            "nbfc,np->bfcp", weighted, harmonics[:, columns]
        )
    return packed


def project_packed(values: np.ndarray, *, source: Any, target: Any) -> np.ndarray:
    r"""Project a packed buffer onto a lower ``(lmax, mmax)`` block table.

    Section 7.3's convergence levels are *projections of the retained vectors*,
    never re-transforms.  The projection is exact and is a pure slice: the
    signed-``m``-major table gives block ``m`` the degrees ``l = |m| .. lmax``
    in ascending order, so the same block truncated to ``l = |m| .. L`` is a
    contiguous **prefix** of the retained row.  Nothing is recomputed, and no
    padded cell can enter the result because none exists.

    Parameters
    ----------
    values : ndarray
        A packed buffer whose last axis is ``source.packed_value_count``.
    source, target : ScalarPackedTable
        The retained table and the level table, with
        ``target.lmax <= source.lmax`` and ``target.mmax <= source.mmax``.
    """
    if target.lmax > source.lmax or target.mmax > source.mmax:
        raise ValueError("a packed projection may only narrow its block table")
    buffer = np.asarray(values)
    if buffer.shape[-1] != source.packed_value_count:
        raise ValueError("packed buffer does not match its declared source table")
    # A four-field table carries one row per signed ``m`` *and* field, and a
    # field's ``l_start = max(abs(m), abs(spin))`` does not move with ``lmax``,
    # so the same prefix rule holds per ``(m, field)`` block.
    origins = {
        (int(row["m"]), str(row.get("field_name", ""))): row
        for row in source.block_rows
    }
    projected = np.empty(
        (*buffer.shape[:-1], target.packed_value_count), dtype=buffer.dtype
    )
    for row in target.block_rows:
        origin = origins[(int(row["m"]), str(row.get("field_name", "")))]
        count = int(row["value_stop"]) - int(row["value_start"])
        start = int(origin["value_start"])
        projected[..., int(row["value_start"]) : int(row["value_stop"])] = buffer[
            ..., start : start + count
        ]
    return projected


def contract_and_synthesize(
    *,
    grid: CanonicalEraGrid,
    table: Any,
    transfer: np.ndarray,
    sky: np.ndarray,
    mmax: int,
) -> np.ndarray:
    r"""Return Section 6's exposure-averaged time-domain cube ``V0``.

    The per-``m`` forward product is

    .. math::

        v_{pqfc,m}=\sum_l\Bigl[B^Ia^I+\tfrac12B^{(+2)}a^{(+2)}
        +\tfrac12B^{(-2)}a^{(-2)}+B^Va^V\Bigr];

    the exposure top hat is the diagonal ``w_m = sinc(pi m Delta_u)`` factor, and
    the synthesis is ``bar V_k = sum_m w_m v_m exp(+i 2 pi m u_k)`` over the
    retained exact turns.  Neither step regenerates topology from ``k``, ``N``,
    radians or ``tau``.

    The two one-half factors on the spin pair are :data:`SPIN_FIELD_WEIGHTS` --
    a theorem, not a normalization choice -- applied per block row, which is
    exactly one factor per field.  A scalar block table has one row per signed
    ``m`` whose weight is ``1``, so its arithmetic is unchanged.
    """
    from radiosim.core.mmode.time import exposure_sinc_weights, unit_circle_turn

    n_baselines, n_frequencies, n_correlations, _ = transfer.shape
    samples = grid.sidereal_samples
    weights = exposure_sinc_weights(grid, mmax=mmax)
    polarized = _is_polarized_table(table)
    output = np.zeros(
        (samples, n_baselines, n_frequencies, n_correlations), dtype=np.complex128
    )
    for row in table.block_rows:
        order = int(row["m"])
        if abs(order) > mmax:
            continue
        start, stop = int(row["value_start"]), int(row["value_stop"])
        if stop <= start:
            continue
        # ``v_m`` for every baseline, frequency and correlation at once.
        per_mode = np.einsum(
            "bfcp,fp->bfc", transfer[:, :, :, start:stop], sky[:, start:stop]
        )
        mode_weight = weights[order]
        if polarized:
            mode_weight = mode_weight * SPIN_FIELD_WEIGHTS[str(row["field_name"])]
        weighted = per_mode * mode_weight
        for sample_index in range(samples):
            phase = unit_circle_turn(order * grid.center_turn(sample_index))
            output[sample_index] += weighted * phase
    return output


def point_sky_coefficients(
    *,
    table: Any,
    cirs: np.ndarray,
    flux_per_frequency: np.ndarray,
) -> np.ndarray:
    """Return the analytic point-delta sky coefficients, ``(frequency, packed)``.

    Section 7.1: point components are not silently rasterized.  The harmonics
    are evaluated at the exact transported source direction.
    """
    from radiosim.core.mmode.harmonics import packed_conjugate_harmonics

    directions = np.atleast_2d(np.asarray(cirs, dtype=np.float64))
    theta, phi = _celestial_angles(directions)
    harmonics = packed_conjugate_harmonics(table, theta, phi)
    return np.asarray(flux_per_frequency, dtype=np.complex128) @ harmonics


def _celestial_angles(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen-frame ``(colatitude, longitude)`` of unit vectors.

    Section 6's landed direct-RIME basis puts every harmonic evaluation on the
    celestial angles of the direction, so the transfer nodes, the analytic point
    coefficients and the pixel-measure projection all reach this one conversion.
    """
    return (
        np.arccos(np.clip(directions[:, 2], -1.0, 1.0)),
        np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * math.pi),
    )


def polarized_point_sky_coefficients(
    *,
    table: Any,
    cirs: np.ndarray,
    stokes_per_frequency: np.ndarray,
    tangent_frame: Any = None,
) -> np.ndarray:
    """Return the analytic full-Stokes point coefficients, ``(frequency, packed)``.

    Section 7.1's analytic delta-function rule, with all four Section 5.3 fields:
    the scalar ``I``/``V`` expansions and the spin ``+-2`` pair are evaluated at
    the exact transported source direction, never rasterized.  Section 5.2's
    bridge is applied once, by the shared
    :func:`~radiosim.core.mmode.sky.resolve_stokes_fields`.
    """
    from radiosim.core.mmode.sky import point_polarized_coefficients_per_frequency

    directions = np.atleast_2d(np.asarray(cirs, dtype=np.float64))
    theta, phi = _celestial_angles(directions)
    return point_polarized_coefficients_per_frequency(
        table=table,
        colatitude=theta,
        longitude=phi,
        stokes=np.asarray(stokes_per_frequency, dtype=np.float64),
        tangent_frame=tangent_frame,
    )


def certificate_identity(row: Mapping[str, Any]) -> str:
    """Return Section 14.0's ``certificate_sha256`` of a frame-certificate row.

    The digest is taken over the row excluding exactly ``fixture_id``,
    ``certificate_sha256`` and ``pass``, so no nested manifest can contain the
    digest that contains it.
    """
    payload = {
        key: value
        for key, value in row.items()
        if key not in {"fixture_id", "certificate_sha256", "pass"}
    }
    return object_digest("radiosim.mmode-frame-certificate.v1", payload)


def render_frame_certificate_failure(**values: Any) -> str:
    """Render Section 8's dynamic frame-certificate rejection line.

    If no paired horizon root exists, ``horizon_root_max`` is rendered as the
    fixed numeric ``0.000000e+00``, and the mismatch measure and limit are also
    rendered as ``0.000000e+00``.
    """
    resolved = {
        "phase": float(values.get("phase", 0.0)),
        "root_count": int(values.get("root_count", 0)),
        "orientation": int(values.get("orientation", 0)),
        "membership": int(values.get("membership", 0)),
        "outside_sign": int(values.get("outside_sign", 0)),
        "unresolved": int(values.get("unresolved", 0)),
        "root_max": float(values.get("root_max", 0.0)),
        "mismatch_measure": float(values.get("mismatch_measure", 0.0)),
        "mismatch_limit": float(values.get("mismatch_limit", 0.0)),
        "cube_max": float(values.get("cube_max", 0.0)),
        "cube_limit": float(values.get("cube_limit", 0.0)),
        "cube_l2": float(values.get("cube_l2", 0.0)),
    }
    return FRAME_CERTIFICATE_FAILURE_TEMPLATE.format(**resolved)


def _turn_row(value: Fraction) -> str:
    """Return the canonical ``p/q`` spelling of an exact retained turn."""
    return canonical_rational(value)


class MModeFrameCertificateFailed(RuntimeError):
    """Section 4.2's frame-applicability certificate rejected the run.

    The certificate is computed in memory for every solve, before harmonic
    work, and it has no waiver: a run whose frozen and operational censuses
    disagree, whose roots or mismatch measure exceed their fixed limits, or
    whose two models' direct cubes differ by more than the certified frame
    bound has no result.
    """

    def __init__(self, certificate: FrameCertificate) -> None:
        self.certificate = certificate
        row = certificate.row
        super().__init__(
            render_frame_certificate_failure(
                phase=row["phase_max_rad"],
                root_count=row["horizon_root_count_mismatches"],
                orientation=row["horizon_root_orientation_mismatches"],
                membership=row["horizon_membership_mismatches"],
                outside_sign=row["horizon_outside_slab_sign_mismatches"],
                unresolved=row["horizon_unresolved_interval_count"],
                root_max=row["horizon_root_max_rad"],
                mismatch_measure=row["horizon_mismatch_measure_rad"],
                mismatch_limit=row["horizon_mismatch_measure_limit_rad"],
                cube_max=row["cube_max_jy"],
                cube_limit=row["cube_limit_jy"],
                cube_l2=row["cube_l2"],
            )
        )


class MModeTruncationGateFailed(RuntimeError):
    """Section 7.3's every-run two-tier gate rejected the run.

    The gate is authoritative and runs before any result or output path is
    created.  Tier 1a's ``1e-8`` limits and tier 2's monotone-decrease and
    quarter-to-full predicates are fixed SCI-004 bounds and may not be widened,
    so a run whose pipeline or truncation cannot meet them has no result.
    """

    def __init__(self, gate: DirectGateRecord) -> None:
        self.gate = gate
        super().__init__(
            "execution.simulator='mmode' two-tier gate failed: "
            f"horizon_free_max={gate.horizon_free_shell_max_jy:.6e} Jy "
            f"(limit={gate.horizon_free_shell_max_limit_jy:.6e} Jy); "
            f"horizon_free_l2={gate.horizon_free_shell_l2:.6e} "
            f"(limit={gate.horizon_free_shell_l2_limit:.6e}); "
            f"deficit_max={gate.deficit_max_jy:.6e} Jy "
            f"(quarter={gate.deficit_max_quarter_jy:.6e}, "
            f"half={gate.deficit_max_half_jy:.6e}, "
            f"factor={gate.convergence_factor:.6f}, "
            f"floor={CONVERGENCE_FACTOR_FLOOR:.6f})."
        )


def _site_geodetic(location: Any) -> tuple[float, float, float]:
    """Return the resolved geodetic site as ``(longitude, latitude, height)``.

    Section 4.1 builds the fixed local ITRS basis from the resolved geodetic
    site, so the frame reads the run's one observing ``EarthLocation`` rather
    than re-deriving a position from the antenna layout.
    """
    from astropy import units as u

    return (
        float(location.lon.to_value(u.deg)),
        float(location.lat.to_value(u.deg)),
        float(location.height.to_value(u.m)),
    )


def _kernel_context(
    request: SkySolveRequest, frame: FrozenFrame, grid: CanonicalEraGrid
) -> KernelContext:
    """Resolve the Section 6 kernel inputs once for the run."""
    from astropy.time import Time

    from radiosim.core.beam.runtime import AntennaId
    from radiosim.core.mmode.transfer import resolved_receptor_matrices
    from radiosim.core.polarization_basis import CORRELATION_LABELS

    instrument = request.instrument
    antenna_ids = tuple(
        AntennaId(number, name)
        for number, name in zip(
            instrument.antenna_numbers, instrument.antenna_names, strict=True
        )
    )
    reference = Time(
        float(grid.utc_two_part[0][0]),
        float(grid.utc_two_part[1][0]),
        format="jd",
        scale="utc",
    )
    return KernelContext(
        frame=frame,
        beam_system=request.beam_system,
        antenna_ids=antenna_ids,
        selected_pairs=tuple(
            (
                instrument.row_index_by_number[first],
                instrument.row_index_by_number[second],
            )
            for first, second in instrument.selected_pairs
        ),
        baseline_vectors_enu_m=np.asarray(
            instrument.baseline_vectors_enu_m, dtype=np.float64
        ),
        frequencies_hz=np.asarray(request.frequencies, dtype=np.float64),
        time_mjd=float(np.asarray(reference.mjd)),
        # Section 6's constant ``M_p = H_p C_p``, from the maintained direct-RIME
        # receptor code objects rather than a second derivation.
        receptor_matrices=resolved_receptor_matrices(
            receptors=request.receptors, instrument=instrument
        ),
        correlation_labels=tuple(
            str(label) for label in CORRELATION_LABELS[request.receptors.output_basis]
        ),
    )


def _resolve_point_component(
    request: SkySolveRequest, frame: FrozenFrame, context: KernelContext
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve the point payload into CIRS directions and per-frequency Stokes."""
    arrays = request.source_arrays
    if arrays is None:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, context.n_frequencies, 4), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
        )
    ra = np.asarray(arrays["ra_rad"], dtype=np.float64)
    dec = np.asarray(arrays["dec_rad"], dtype=np.float64)
    flux = np.asarray(arrays["flux"], dtype=np.float64)
    spectral_index = np.asarray(
        arrays.get("spectral_index", np.zeros_like(flux)), dtype=np.float64
    )
    reference = np.asarray(
        arrays.get("ref_freq", np.full_like(flux, context.frequencies_hz[0])),
        dtype=np.float64,
    )
    # A synthetic source may carry no reference frequency; the run's first
    # channel is then the reference, which makes a flat spectrum exact.
    reference = np.where(
        np.isfinite(reference) & (reference > 0.0),
        reference,
        float(context.frequencies_hz[0]),
    )
    ratio = context.frequencies_hz[None, :] / reference[:, None]
    spectrum = ratio ** spectral_index[:, None]
    stokes = np.zeros((ra.shape[0], context.n_frequencies, 4), dtype=np.float64)
    stokes[:, :, 0] = flux[:, None] * spectrum
    # Section 12.1's ledger row carries ``resolved_stokes_iau`` and derives its
    # ``active_frequency_mask`` from it, so all four components are resolved
    # here.  Zeroing ``Q``/``U``/``V`` would make a polarized payload describe
    # itself to the ledger, the certificate and the direct machinery as an
    # unpolarized one -- the ledger would then be a record of a run that never
    # happened.  A missing column is genuinely zero, not merely unread.
    for index, name in enumerate(("stokes_q", "stokes_u", "stokes_v"), start=1):
        column = arrays.get(name)
        if column is None:
            continue
        values = np.asarray(column, dtype=np.float64)
        if values.size == 0:
            continue
        stokes[:, :, index] = (
            np.broadcast_to(values.reshape(-1, 1), (ra.shape[0], context.n_frequencies))
            * spectrum
        )
    return (
        frame.cirs_directions(ra, dec),
        stokes,
        np.stack((ra, dec), axis=-1),
    )


#: Section 7.3's four local attribution diagnostics, in their fixed order.
DIAGNOSTIC_IDS: Final[tuple[str, ...]] = (
    "quadrature",
    "l_tail",
    "m_tail",
    "combined_local",
)


def build_m1_evidence(request: SkySolveRequest) -> dict[str, Any]:
    """Return every Section 14.2 M1 preimage one fixture's run produces.

    The forward solve is the *same* pipeline the strategy runs -- the shared
    ``_mmode_pipeline`` body -- so the evidence cannot describe a run the solver
    never performed.  On top of it this builds Section 7.3's two complete
    coverage preimages: ``direct_coverage`` over every output cell of the
    authoritative frozen-direct comparison, and ``shell_coverage`` with the
    transfer-sample concatenation rows, the four diagnostic cubes' per-cell
    comparisons, and the per-field/per-signed-``m`` block diagnostics.
    """
    from radiosim.core.mmode.harmonics import (
        polarized_packed_block_table,
        scalar_packed_block_table,
    )

    solved = _mmode_pipeline(request)
    grid = solved["grid"]
    frame = solved["frame"]
    context = solved["context"]
    dimensions = solved["dimensions"]
    directions = solved["directions"]
    certificate = solved["certificate"]
    cube = solved["cube"]
    sky = solved["sky"]
    table = solved["table"]
    polarized = solved["execution_path"] == "polarized"

    def block_table(level_lmax: int, level_mmax: int) -> Any:
        if polarized:
            return polarized_packed_block_table(lmax=level_lmax, mmax=level_mmax)
        return scalar_packed_block_table(lmax=level_lmax, mmax=level_mmax)

    catalog, catalog_sha256 = _transfer_catalog(directions)
    check_table = block_table(dimensions.lcheck, dimensions.mcheck)
    check_transfer = build_production_transfer(
        context=context, frame=frame, nside=dimensions.qcheck, table=check_table
    )
    if polarized:
        check_sky = polarized_point_sky_coefficients(
            table=check_table,
            cirs=solved["point_cirs"],
            stokes_per_frequency=solved["point_stokes"],
            tangent_frame=(
                solved["tangent_polarization_frame"]
                if isinstance(solved["tangent_polarization_frame"], Mapping)
                else None
            ),
        )
    else:
        check_sky = point_sky_coefficients(
            table=check_table,
            cirs=solved["point_cirs"],
            flux_per_frequency=solved["point_stokes"][:, :, 0].T,
        )

    def synthesize(lmax: int, mmax: int) -> np.ndarray:
        level = block_table(lmax, mmax)
        return contract_and_synthesize(
            grid=grid,
            table=level,
            transfer=project_packed(check_transfer, source=check_table, target=level),
            sky=project_packed(check_sky, source=check_table, target=level),
            mmax=level.mmax,
        )

    operands = {
        ("diagnostic", dimensions.lmax, dimensions.mmax): synthesize(
            dimensions.lmax, dimensions.mmax
        ),
        ("diagnostic", dimensions.lcheck, dimensions.mmax): synthesize(
            dimensions.lcheck, dimensions.mmax
        ),
        ("diagnostic", dimensions.lcheck, dimensions.mcheck): synthesize(
            dimensions.lcheck, dimensions.mcheck
        ),
        ("production", dimensions.lmax, dimensions.mmax): cube,
    }
    production_id = f"production:{dimensions.quadrature_nside}"
    diagnostic_id = f"diagnostic:{dimensions.qcheck}"
    pairs = (
        (
            "quadrature",
            (diagnostic_id, dimensions.lmax, dimensions.mmax),
            (production_id, dimensions.lmax, dimensions.mmax),
        ),
        (
            "l_tail",
            (diagnostic_id, dimensions.lcheck, dimensions.mmax),
            (diagnostic_id, dimensions.lmax, dimensions.mmax),
        ),
        (
            "m_tail",
            (diagnostic_id, dimensions.lcheck, dimensions.mcheck),
            (diagnostic_id, dimensions.lcheck, dimensions.mmax),
        ),
        (
            "combined_local",
            (diagnostic_id, dimensions.lcheck, dimensions.mcheck),
            (production_id, dimensions.lmax, dimensions.mmax),
        ),
    )

    joins: list[dict[str, Any]] = []
    deltas: dict[str, np.ndarray] = {}
    for name, left, right in pairs:
        left_cube = operands[
            (
                "production" if left[0] == production_id else "diagnostic",
                left[1],
                left[2],
            )
        ]
        right_cube = operands[
            (
                "production" if right[0] == production_id else "diagnostic",
                right[1],
                right[2],
            )
        ]
        delta = left_cube - right_cube
        deltas[name] = delta
        joins.append(
            {
                "diagnostic_id": name,
                "lhs_grid_id": left[0],
                "lhs_lmax": int(left[1]),
                "lhs_mmax": int(left[2]),
                "rhs_grid_id": right[0],
                "rhs_lmax": int(right[1]),
                "rhs_mmax": int(right[2]),
                "lhs_cube_sha256": _visibility_cube_identity(left_cube),
                "rhs_cube_sha256": _visibility_cube_identity(right_cube),
                "delta_cube_sha256": _visibility_cube_identity(delta),
            }
        )

    samples, baselines, frequencies, correlations = cube.shape
    reference = certificate.frozen_gauss128
    error = certificate.frozen_enclosure_error
    upper = np.abs(cube - reference) + error
    direct_rows = [
        {
            "sample_index": sample,
            "baseline_index": baseline,
            "frequency_index": frequency,
            "correlation_index": correlation,
            "frozen_real_f64be": f64be(
                float(reference[sample, baseline, frequency, correlation].real)
            ),
            "frozen_imag_f64be": f64be(
                float(reference[sample, baseline, frequency, correlation].imag)
            ),
            "frozen_error_f64be": f64be(
                float(error[sample, baseline, frequency, correlation])
            ),
            "mmode_real_f64be": f64be(
                float(cube[sample, baseline, frequency, correlation].real)
            ),
            "mmode_imag_f64be": f64be(
                float(cube[sample, baseline, frequency, correlation].imag)
            ),
            "upper_delta_f64be": f64be(
                float(upper[sample, baseline, frequency, correlation])
            ),
        }
        for sample in range(samples)
        for baseline in range(baselines)
        for frequency in range(frequencies)
        for correlation in range(correlations)
    ]
    direct_coverage = {
        "schema_version": "radiosim.mmode-direct-output-coverage.v1",
        "input_identity_sha256": solved["input_identity_sha256"],
        "cube_shape": [samples, baselines, frequencies, correlations],
        "frozen_gauss128_cube_sha256": certificate.frozen_gauss128_cube_sha256,
        "frozen_enclosure_error_cube_sha256": (
            certificate.frozen_enclosure_error_cube_sha256
        ),
        "mmode_cube_sha256": _visibility_cube_identity(cube),
        "rows": direct_rows,
    }

    shell_rows = [
        {
            "diagnostic_id": name,
            "time_index": sample,
            "baseline_index": baseline,
            "frequency_index": frequency,
            "correlation_index": correlation,
            "absolute_delta_jy_f64be": f64be(
                float(abs(deltas[name][sample, baseline, frequency, correlation]))
            ),
        }
        for name in DIAGNOSTIC_IDS
        for sample in range(samples)
        for baseline in range(baselines)
        for frequency in range(frequencies)
        for correlation in range(correlations)
    ]
    field_rows = _field_block_rows(
        grid=grid,
        check_table=check_table,
        check_transfer=check_transfer,
        check_sky=check_sky,
        production_table=table,
        production_transfer=solved["transfer"],
        production_sky=sky,
        dimensions=dimensions,
    )
    shell_coverage = {
        "schema_version": "radiosim.mmode-shell-coverage.v1",
        "input_identity_sha256": solved["input_identity_sha256"],
        "frame_certificate_sha256": certificate.certificate_sha256,
        "direction_ledger_sha256": certificate.row["direction_ledger_sha256"],
        "transfer_grid_catalog_sha256": catalog_sha256,
        "diagnostic_grid_joins": joins,
        "transfer_sample_rows": transfer_sample_rows(
            context=context,
            frame=frame,
            catalog=catalog,
            production_table=table,
            diagnostic_table=check_table,
        ),
        "shell_comparison_rows": shell_rows,
        "field_block_rows": field_rows,
    }
    reference_jy = 1e-6 * max(
        1.0,
        float(
            np.max(
                np.abs(operands[("diagnostic", dimensions.lcheck, dimensions.mcheck)])
            )
        ),
    )
    return {
        **solved,
        "transfer_grid_catalog": catalog,
        "transfer_grid_catalog_sha256": catalog_sha256,
        "direct_coverage": direct_coverage,
        "direct_coverage_sha256": object_digest(
            "radiosim.mmode-direct-output-coverage.v1", direct_coverage
        ),
        "shell_coverage": shell_coverage,
        "shell_coverage_sha256": object_digest(
            "radiosim.mmode-shell-coverage.v1", shell_coverage
        ),
        "diagnostic_maxima": {
            name: float(np.max(np.abs(deltas[name]))) for name in DIAGNOSTIC_IDS
        },
        "field_block_diagnostic_max_jy": max(
            (
                max(
                    decode_f64be(row[f"{name}_max_abs_jy_f64be"])
                    for name in DIAGNOSTIC_IDS
                )
                for row in field_rows
            ),
            default=0.0,
        ),
        "shell_diagnostic_reference_jy": reference_jy,
    }


def _field_block_rows(
    *,
    grid: CanonicalEraGrid,
    check_table: Any,
    check_transfer: np.ndarray,
    check_sky: np.ndarray,
    production_table: Any,
    production_transfer: np.ndarray,
    production_sky: np.ndarray,
    dimensions: Any,
) -> list[dict[str, Any]]:
    """Return Section 7.3's per-field, per-signed-``m`` block diagnostics.

    Every one of the four zero-extended diagnostic operands is measured *before*
    field summation, one signed-``m`` block at a time, so a cancellation between
    blocks or fields cannot hide inside a summed cube.  A block whose signed
    ``m`` lies outside an operand's retained range contributes the exact zero
    vector, which is what "zero-extended through signed ``m`` in
    ``[-mcheck, +mcheck]``" means; a null column is forbidden.

    A scalar run evaluates the ``I`` field only, so it is the one field with a
    non-zero vector, in the fixed field order's first position; a full-Stokes run
    measures all four, each from its own packed columns and carrying its own
    Section 6 forward weight.
    """
    from radiosim.core.mmode.time import exposure_sinc_weights, unit_circle_turn

    samples = grid.sidereal_samples
    baselines, frequencies, correlations, _ = check_transfer.shape
    weights = exposure_sinc_weights(grid, mmax=dimensions.mcheck)
    phase_by_order = {
        order: np.asarray(
            [
                unit_circle_turn(order * grid.center_turn(index))
                for index in range(samples)
            ],
            dtype=np.complex128,
        )
        for order in range(-dimensions.mcheck, dimensions.mcheck + 1)
    }
    zero = np.zeros(samples, dtype=np.complex128)

    #: The four Section 7.3 joins as ``(left, right)`` operand descriptors,
    #: where an operand is ``(grid, lmax, mmax)`` and ``grid`` selects the
    #: retained transfer vector the block is projected from.
    joins = {
        "quadrature": (
            ("check", dimensions.lmax, dimensions.mmax),
            ("production", dimensions.lmax, dimensions.mmax),
        ),
        "l_tail": (
            ("check", dimensions.lcheck, dimensions.mmax),
            ("check", dimensions.lmax, dimensions.mmax),
        ),
        "m_tail": (
            ("check", dimensions.lcheck, dimensions.mcheck),
            ("check", dimensions.lcheck, dimensions.mmax),
        ),
        "combined_local": (
            ("check", dimensions.lcheck, dimensions.mcheck),
            ("production", dimensions.lmax, dimensions.mmax),
        ),
    }
    sources = {
        "check": (check_table, check_transfer, check_sky),
        "production": (production_table, production_transfer, production_sky),
    }

    def block_value(
        source: str,
        lmax: int,
        mmax: int,
        order: int,
        field: str,
        baseline: int,
        frequency: int,
        correlation: int,
    ) -> complex:
        """Return one operand's ``sum_l B_lm a_lm`` for one field and block."""
        table, transfer, sky = sources[source]
        if abs(order) > mmax or abs(order) > table.mmax:
            return 0j
        block = _packed_block_row(table, order, field)
        if block is None:
            return 0j
        start = int(block["value_start"])
        stop = int(block["value_stop"])
        degrees = np.arange(int(block["l_start"]), int(block["l_stop"]))
        keep = degrees <= lmax
        return complex(
            np.sum(
                transfer[baseline, frequency, correlation, start:stop][keep]
                * sky[frequency, start:stop][keep]
            )
        )

    rows: list[dict[str, Any]] = []
    for baseline in range(baselines):
        for frequency in range(frequencies):
            for correlation in range(correlations):
                for order in range(-dimensions.mcheck, dimensions.mcheck + 1):
                    weight = complex(weights[order])
                    phase = phase_by_order[order]
                    for field in FIELD_ORDER:
                        rows.append(
                            _field_block_row(
                                baseline=baseline,
                                frequency=frequency,
                                correlation=correlation,
                                field=field,
                                order=order,
                                weight=weight * SPIN_FIELD_WEIGHTS[field],
                                phase=phase,
                                zero=zero,
                                samples=samples,
                                joins=joins,
                                block_value=block_value,
                            )
                        )
    return rows


def _field_block_row(
    *,
    baseline: int,
    frequency: int,
    correlation: int,
    field: str,
    order: int,
    weight: complex,
    phase: np.ndarray,
    zero: np.ndarray,
    samples: int,
    joins: Mapping[str, Any],
    block_value: Any,
) -> dict[str, Any]:
    """Return one Section 7.3 field/block diagnostic row and its four vectors.

    A scalar run evaluates the ``I`` field, so the three remaining fixed-order
    fields have no packed columns and contribute the exact zero time vector.
    They are recorded rather than omitted, because the ledger's row count is the
    complete ``B*F*C*4*(2*mcheck+1)`` product and a missing field would read as
    coverage that was never attempted.
    """
    row: dict[str, Any] = {
        "baseline_index": baseline,
        "frequency_index": frequency,
        "correlation_index": correlation,
        "field": field,
        "signed_m": order,
        "diagnostic_ids": list(DIAGNOSTIC_IDS),
    }
    for name in DIAGNOSTIC_IDS:
        left, right = joins[name]
        delta = block_value(
            left[0],
            left[1],
            left[2],
            order,
            field,
            baseline,
            frequency,
            correlation,
        ) - block_value(
            right[0],
            right[1],
            right[2],
            order,
            field,
            baseline,
            frequency,
            correlation,
        )
        vector = zero if delta == 0j else delta * weight * phase
        if not np.all(np.isfinite(vector)):
            raise ValueError("a field/block diagnostic vector is not finite")
        row[f"{name}_time_value_count"] = samples
        row[f"{name}_time_values_sha256"] = array_digest(
            "radiosim.mmode-field-block-diagnostic.v1",
            f"{name}_field_block_delta",
            ["time"],
            "Jy",
            vector,
            dtype="complex128-be",
        )
        row[f"{name}_max_abs_jy_f64be"] = f64be(
            float(np.max(np.abs(vector))) if vector.size else 0.0
        )
    return row


def _mmode_pipeline(request: SkySolveRequest) -> dict[str, Any]:
    """Run one full-sidereal m-mode forward solve up to its two-tier gate.

    The order is the design's: the resolved payload is validated, the exact-turn
    grid and frozen frame are resolved, the Section 4.2 certificate is computed
    **before** any harmonic work, and Section 7.3's authoritative complete
    frozen-direct gate is evaluated before any result exists.  The direct point
    and HEALPix production kernels are never called from this path.

    Both public entry points share this one body: ``solve_mmode`` turns the
    result into the strategy outcome, and ``build_m1_evidence`` retains the same
    objects as the Section 14.2 preimages.  A second, divergent pipeline would
    let the evidence describe a run the solver never performed.
    """
    from radiosim.core.mmode.harmonics import (
        polarized_packed_block_table,
        scalar_packed_block_table,
    )
    from radiosim.core.mmode.types import derive_mmode_dimensions

    grid = request.era_grid
    if not isinstance(grid, CanonicalEraGrid):
        raise ValueError(
            "an m-mode solve requires the retained CanonicalEraGrid of its "
            "obs_time.mode='full_sidereal' variant"
        )
    block = request.mmode
    if block is None:
        raise ValueError("an m-mode solve requires its resolved execution.mmode block")
    dimensions = derive_mmode_dimensions(
        lmax=int(block.lmax),
        mmax=int(block.mmax),
        quadrature_nside=int(block.quadrature_nside),
    )
    longitude, latitude, height = _site_geodetic(request.location)
    frame = build_frozen_frame(
        start_time=grid.start_time_iso,
        longitude_deg=longitude,
        latitude_deg=latitude,
        height_m=height,
    )
    context = _kernel_context(request, frame, grid)
    point_cirs, point_stokes, point_icrs = _resolve_point_component(
        request, frame, context
    )
    ledger = build_direction_ledger(
        frame=frame,
        dimensions=dimensions,
        point_cirs=point_cirs,
        point_stokes=point_stokes,
        point_icrs=point_icrs,
        native_cirs=np.zeros((0, 3), dtype=np.float64),
        native_stokes=np.zeros((0, context.n_frequencies, 4), dtype=np.float64),
        native_icrs=np.zeros((0, 2), dtype=np.float64),
        native_solid_angle=0.0,
    )
    tangent_frame = _resolved_tangent_frame(request, point_stokes)
    polarized = _payload_is_polarized(point_stokes)
    input_manifest, input_identity_sha256 = build_input_identity(
        request=request,
        grid=grid,
        frame=frame,
        context=context,
        dimensions=dimensions,
        directions=ledger,
        tangent_frame=tangent_frame,
    )
    certificate = build_frame_certificate(
        grid=grid,
        frame=frame,
        context=context,
        directions=ledger,
        beam_peak_ceiling=1.0,
        input_identity_sha256=input_identity_sha256,
    )
    if not certificate.passed:
        raise MModeFrameCertificateFailed(certificate)

    def block_table(level_lmax: int, level_mmax: int) -> Any:
        if polarized:
            return polarized_packed_block_table(lmax=level_lmax, mmax=level_mmax)
        return scalar_packed_block_table(lmax=level_lmax, mmax=level_mmax)

    def sky_coefficients(level_table: Any) -> np.ndarray:
        if polarized:
            return polarized_point_sky_coefficients(
                table=level_table,
                cirs=point_cirs,
                stokes_per_frequency=point_stokes,
                tangent_frame=(
                    tangent_frame if isinstance(tangent_frame, Mapping) else None
                ),
            )
        return point_sky_coefficients(
            table=level_table,
            cirs=point_cirs,
            flux_per_frequency=point_stokes[:, :, 0].T,
        )

    table = block_table(dimensions.lmax, dimensions.mmax)
    transfer = build_production_transfer(
        context=context,
        frame=frame,
        nside=dimensions.quadrature_nside,
        table=table,
    )
    sky = sky_coefficients(table)
    cube = contract_and_synthesize(
        grid=grid, table=table, transfer=transfer, sky=sky, mmax=dimensions.mmax
    )

    # Tier 1b: the with-horizon quadrature shell, recorded and fixture-budgeted.
    shell_transfer = build_production_transfer(
        context=context, frame=frame, nside=dimensions.qcheck, table=table
    )
    shell_cube = contract_and_synthesize(
        grid=grid, table=table, transfer=shell_transfer, sky=sky, mmax=dimensions.mmax
    )

    # Tier 1a: the same pipeline with ``H === 1`` and everything else identical.
    horizon_free = contract_and_synthesize(
        grid=grid,
        table=table,
        transfer=build_production_transfer(
            context=context,
            frame=frame,
            nside=dimensions.quadrature_nside,
            table=table,
            horizon=False,
        ),
        sky=sky,
        mmax=dimensions.mmax,
    )
    horizon_free_qcheck = contract_and_synthesize(
        grid=grid,
        table=table,
        transfer=build_production_transfer(
            context=context,
            frame=frame,
            nside=dimensions.qcheck,
            table=table,
            horizon=False,
        ),
        sky=sky,
        mmax=dimensions.mmax,
    )

    # Tier 2: the convergence levels are exact block-table projections of the
    # retained vectors -- Section 7.3 pairs each level with ``min(mmax, level)``
    # and the production quadrature -- never re-transforms.
    quarter_level = max(2, dimensions.lmax // 4)
    half_level = max(quarter_level + 1, dimensions.lmax // 2)
    level_deficits: list[float] = []
    for level in (quarter_level, half_level):
        level_table = block_table(level, min(dimensions.mmax, level))
        level_cube = contract_and_synthesize(
            grid=grid,
            table=level_table,
            transfer=project_packed(transfer, source=table, target=level_table),
            sky=project_packed(sky, source=table, target=level_table),
            mmax=level_table.mmax,
        )
        level_deficits.append(
            float(
                np.max(
                    np.abs(level_cube - certificate.frozen_gauss128)
                    + certificate.frozen_enclosure_error
                )
            )
        )

    gate = evaluate_two_tier_gate(
        mmode_cube=cube,
        horizon_free_cube=horizon_free,
        horizon_free_qcheck_cube=horizon_free_qcheck,
        quadrature_shell_cube=shell_cube,
        frozen_gauss128=certificate.frozen_gauss128,
        frozen_enclosure_error=certificate.frozen_enclosure_error,
        deficit_max_quarter_jy=level_deficits[0],
        deficit_max_half_jy=level_deficits[1],
    )
    return {
        "grid": grid,
        "frame": frame,
        "context": context,
        "dimensions": dimensions,
        "directions": ledger,
        "table": table,
        "transfer": transfer,
        "sky": sky,
        "cube": cube,
        "shell_cube": shell_cube,
        "horizon_free": horizon_free,
        "horizon_free_qcheck": horizon_free_qcheck,
        "certificate": certificate,
        "gate": gate,
        "input_identity_manifest": input_manifest,
        "input_identity_sha256": input_identity_sha256,
        "point_cirs": point_cirs,
        # Section 4.1's *untransported* catalogue coordinates, retained beside
        # the transported directions so Section 12.2's omitted-tangent-transport
        # non-vacuity control can be measured against the same run rather than
        # against a second, differently configured one.
        "point_icrs": point_icrs,
        "point_stokes": point_stokes,
        "tangent_polarization_frame": tangent_frame,
        "execution_path": "polarized" if polarized else "scalar",
        "deficit_max_quarter_jy": level_deficits[0],
        "deficit_max_half_jy": level_deficits[1],
        "quarter_level": quarter_level,
        "half_level": half_level,
    }


def _payload_is_polarized(point_stokes: np.ndarray) -> bool:
    """Return whether a resolved payload has any non-zero ``Q``, ``U`` or ``V``.

    This selects Section 10's ``execution_path``.  ``V`` counts here although it
    does not trigger Section 5.1's tangent-frame requirement -- ``V`` is a
    spin-0 field with no tangent-basis dependence, but it is still a polarized
    contribution that the ``I``-only forward product would drop.
    """
    stokes = np.asarray(point_stokes, dtype=np.float64)
    if stokes.size == 0:
        return False
    polarized = stokes[:, :, 1:]
    finite = polarized[np.isfinite(polarized)]
    return bool(finite.size) and bool(np.any(finite != 0.0))


def _resolved_tangent_frame(request: SkySolveRequest, point_stokes: np.ndarray) -> Any:
    """Return Section 10's ``tangent_polarization_frame`` value for one run.

    Section 5.1 requires "every point or HEALPix payload with non-zero ``Q`` or
    ``U``" to carry the six-key block and lets "an ``I``/``V``-only payload omit
    the tangent block", so the value is read from the *resolved* payload rather
    than from the document: a source may declare a frame it does not need, and a
    run's snapshot should describe the sky it actually integrated.

    ``coordinate_frame`` is ``icrs`` because the harmonic expansion itself is
    performed in the frozen celestial frame of Section 4.1 -- a Galactic-declared
    payload is transported into it, with its tangent basis, before any spin
    expansion, which is exactly what Section 5.1 forbids skipping.  The remaining
    five keys are Section 5.1's fixed canonical literals; a document may declare
    the HEALPix/CMB ``north_through_west`` *source* convention, but what is
    stored, integrated and recorded here is always the IAU one.
    """
    del request  # the resolved payload, not the declaration, is authoritative
    from radiosim.core.sky.containers import TangentPolarizationFrame

    stokes = np.asarray(point_stokes, dtype=np.float64)
    linear = stokes[:, :, 1:3] if stokes.size else stokes
    finite = linear[np.isfinite(linear)] if linear.size else linear
    if finite.size == 0 or not bool(np.any(finite != 0.0)):
        return MMODE_TANGENT_FRAME_M1
    return TangentPolarizationFrame.canonical("icrs").as_mapping()


#: Section 8's exact ``mmode_public_components`` code and message.
MMODE_PUBLIC_COMPONENTS_CODE: Final = "mmode_public_components"
MMODE_PUBLIC_COMPONENTS_MESSAGE: Final = (
    "execution.simulator='mmode' supports point-source components only in this "
    "phase; a HEALPix-bearing sky requires a future accepted phase."
)

#: Section 8's exact ``mmode_public_beam`` code and message.
MMODE_PUBLIC_BEAM_CODE: Final = "mmode_public_beam"
MMODE_PUBLIC_BEAM_MESSAGE: Final = (
    "execution.simulator='mmode' supports the scalar beam response only in this "
    "phase; a non-scalar resolved beam system requires a future accepted phase."
)

#: The Stage-3 accepted-subset literal a full-efield FITS definition carries.
_FULL_EFIELD_SUBSET: Final = "sci005-stage3-full-efield-v1"


def _rejects_public_capability(request: SkySolveRequest) -> list[Any]:
    """Return the Section 8 issues this request must be refused with, if any.

    ``docs/development/sci004_mmode_design.md`` Section 11, as narrowed by the
    accepted accepted-capability-characterization-envelope correction: "the
    public path rejects a HEALPix-bearing payload and a non-scalar resolved beam
    system with the Section 8 typed issues before any work".

    Both refusals close a measured silent defect rather than a hypothetical one.
    The public solve path builds a point component and nothing else, so a
    HEALPix-bearing sky previously ran to completion and published an
    identically zero cube whose ``component_element_counts`` was ``[0]`` -- a
    result the Section 7.3 gate passes vacuously through its exact-zero corner --
    and a hybrid payload silently lost its diffuse half while its gate passed on
    the point half alone.  A non-scalar resolved beam previously failed after
    the whole frame and transfer stage with an untyped ``BeamEvaluationError``
    naming a missing ``boresight_parallactic_rad``.  Rejecting is not a
    narrowing of the accepted M2 capability: accepted M2 never licensed either
    run through the public path, and Section 11 records both as deferred,
    future red-sliced work.
    """
    from radiosim.io.config import ConfigIssue

    issues: list[Any] = []
    sky_model = request.sky_model
    healpix_bearing = getattr(sky_model, "healpix", None) is not None or str(
        request.sky_representation
    ) in ("healpix_map", "hybrid")
    if healpix_bearing:
        issues.append(
            ConfigIssue(
                "sky_model",
                MMODE_PUBLIC_COMPONENTS_CODE,
                MMODE_PUBLIC_COMPONENTS_MESSAGE,
                stage="unsupported",
            )
        )
    assignments = request.beam_system.state.resolved.assignments
    non_scalar = any(
        assignment.squint is not None
        or getattr(assignment.definition, "accepted_subset_version", None)
        == _FULL_EFIELD_SUBSET
        for assignment in assignments
    )
    if non_scalar:
        issues.append(
            ConfigIssue(
                "beams",
                MMODE_PUBLIC_BEAM_CODE,
                MMODE_PUBLIC_BEAM_MESSAGE,
                stage="unsupported",
            )
        )
    return issues


def solve_mmode(request: SkySolveRequest) -> SkySolveOutcome:
    """Solve one full-sidereal m-mode run and return the strategy outcome.

    Section 8's two public-capability rejections run first, before any solver
    work: a HEALPix-bearing sky and a non-scalar resolved beam system are
    refused with their exact typed issues rather than silently producing a
    vacuous or half-dropped result.  Section 7.3's two-tier gate is then
    authoritative and runs before any result or output path is created, so a
    failing gate raises instead of returning a cube.
    """
    from radiosim.io.config_resolution import UnsupportedConfigError
    from radiosim.simulator.base import SkySolveOutcome as Outcome

    unsupported = _rejects_public_capability(request)
    if unsupported:
        raise UnsupportedConfigError(unsupported)

    solved = _mmode_pipeline(request)
    gate = solved["gate"]
    if not gate.pass_:
        raise MModeTruncationGateFailed(gate)

    grid = solved["grid"]
    dimensions = solved["dimensions"]
    certificate = solved["certificate"]
    cube = solved["cube"]
    point_cirs = solved["point_cirs"]
    execution_path = str(solved["execution_path"])
    snapshot = MModeSolverSnapshot(
        tangent_polarization_frame=solved["tangent_polarization_frame"],
        sky_representation=str(request.sky_representation),
        execution_path=execution_path,
        components=("point",),
        component_element_counts=(int(point_cirs.shape[0]),),
        sidereal_samples=grid.sidereal_samples,
        lmax=dimensions.lmax,
        mmax=dimensions.mmax,
        quadrature_nside=dimensions.quadrature_nside,
        iers_table_sha256=solved["frame"].iers_table_sha256,
        frame_certificate_sha256=certificate.certificate_sha256,
        direct_gate=gate,
        frozen_gauss128_cube_sha256=certificate.frozen_gauss128_cube_sha256,
        frozen_enclosure_error_cube_sha256=(
            certificate.frozen_enclosure_error_cube_sha256
        ),
    )
    receptor = cube.reshape(*cube.shape[:3], 2, 2)
    return Outcome(
        receptor_visibilities=request.backend.asarray(receptor),
        components=("point",),
        component_element_counts=(int(point_cirs.shape[0]),),
        execution_path="scalar",
        component_seconds=(0.0,),
        solver_record=snapshot,
    )


# ---------------------------------------------------------------------------
# Section 6 forward per-``m`` product
# ---------------------------------------------------------------------------


#: Section 6's forward per-``m`` product weights.  The two one-half factors on
#: the spin terms are a theorem, not a normalization choice: substituting a
#: delta sky's coefficients and using
#: ``sum_lm _{s}Y_lm(n) conj(_{s}Y_lm(n_s)) -> delta(n - n_s)`` collapses the
#: pair to exactly ``K^Q Q_H + K^U U_H``.  Dropping either factor doubles that
#: contribution.
SPIN_FIELD_WEIGHTS: Final[Mapping[str, float]] = MappingProxyType(
    {"I": 1.0, "+2": 0.5, "-2": 0.5, "V": 1.0}
)


def forward_per_m_product(
    *,
    transfer_block: np.ndarray,
    sky_block: np.ndarray,
    field_order: Sequence[str] = FIELD_ORDER,
    backend: Any = None,
) -> Any:
    r"""Evaluate Section 6's forward per-``m`` contraction.

    .. math::

        v_{pqfc,m}=\sum_l\Bigl[B^I a^I+\tfrac12B^{(+2)}a^{(+2)}
        +\tfrac12B^{(-2)}a^{(-2)}+B^V a^V\Bigr].

    "Per-``m`` solve" means exactly this forward matrix-vector contraction;
    ``SCI-004`` does not expose Shaw's map-making pseudo-inverse or solve for a
    sky.  The field axis is the leading axis of both operands and the packed
    axis is the last; any axes between them broadcast.

    Examples
    --------
    >>> import numpy as np
    >>> transfer = np.ones((4, 2), dtype=np.complex128)
    >>> sky = np.ones((4, 2), dtype=np.complex128)
    >>> complex(forward_per_m_product(transfer_block=transfer, sky_block=sky))
    (6+0j)
    """
    fields = tuple(str(name) for name in field_order)
    if set(fields) != set(SPIN_FIELD_WEIGHTS):
        raise ValueError(
            "the forward product covers exactly Section 5.3's four science fields"
        )
    resolved = backend if backend is not None else _numpy_backend()
    transfer = resolved.asarray(transfer_block)
    sky = resolved.asarray(sky_block)
    total: Any = None
    for index, name in enumerate(fields):
        contribution = resolved.xp.sum(transfer[index] * sky[index], axis=-1)
        scaled = contribution * SPIN_FIELD_WEIGHTS[name]
        total = scaled if total is None else resolved.add(total, scaled)
    return total


def _numpy_backend() -> Any:
    """Return the NumPy reference backend Section 9 makes normative."""
    from radiosim.backends import get_backend

    return get_backend("numpy")


# ---------------------------------------------------------------------------
# Section 9 backend routing for the dense stages
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BackendComplex128Resolution:
    """The resolved dense-work precision of one backend."""

    backend: str
    dtype_name: str
    x64_enabled: bool


def require_backend_complex128(
    backend: str, *, x64_enabled: bool | None = None
) -> BackendComplex128Resolution:
    """Resolve complex128 dense work on one backend, or fail explicitly.

    Section 9: "Complex128 JAX requires x64 and fails explicitly if
    unavailable."  The failure has to be explicit because the silent
    alternative -- demoting to complex64 -- would substitute the separately
    named low-precision contract for the complex128 acceptance row without any
    record that it happened.
    """
    name = str(backend)
    if name != "jax":
        return BackendComplex128Resolution(
            backend=name, dtype_name="complex128", x64_enabled=True
        )
    if x64_enabled is None:
        import jax

        x64_enabled = bool(jax.config.jax_enable_x64)
    if not x64_enabled:
        raise RuntimeError(
            "complex128 m-mode dense work on JAX requires x64; enable "
            "jax_enable_x64 rather than demoting to the separately named "
            "complex64 row (SCI-004 Section 9)"
        )
    return BackendComplex128Resolution(
        backend=name, dtype_name="complex128", x64_enabled=True
    )


def contract_per_m_block(
    *,
    transfer_block: np.ndarray,
    sky_block: np.ndarray,
    field_order: Sequence[str] = FIELD_ORDER,
    backend: Any = None,
    workers: int = 1,
    accumulation_dtype: str = "complex128",
) -> np.ndarray:
    """Contract one dense per-``m`` block, optionally on a non-NumPy backend.

    Section 9 admits JAX and Dask for exactly this stage and the time synthesis;
    every other stage is host-side NumPy work.  ``workers`` owns independent
    frequency-block construction and is clamped to the frequency count, and the
    blocks are assembled in canonical frequency order, so one worker and many
    workers return **bit-identical** results rather than merely tolerant ones.

    The transfer block is ``(baseline, frequency, correlation, field, packed)``
    and the sky block ``(frequency, field, packed)``.
    """
    resolved = backend if backend is not None else _numpy_backend()
    require_backend_complex128(getattr(resolved, "name", "numpy").split("-")[0])
    transfer = np.asarray(transfer_block)
    sky = np.asarray(sky_block)
    if transfer.ndim != 5 or sky.ndim != 3:
        raise ValueError(
            "contract_per_m_block takes a (B,F,C,X,P) transfer and an (F,X,P) sky"
        )
    if transfer.shape[1] != sky.shape[0] or transfer.shape[3:] != sky.shape[1:]:
        raise ValueError("the transfer and sky blocks disagree on F, X or P")

    dtype = np.complex64 if accumulation_dtype == "complex64" else np.complex128
    frequencies = int(transfer.shape[1])
    clamped = max(1, min(int(workers), frequencies))
    del clamped  # the schedule is canonical-order, so the count cannot change it

    fields = tuple(str(name) for name in field_order)
    blocks = []
    for frequency in range(frequencies):
        transfer_slice = resolved.asarray(
            np.moveaxis(transfer[:, frequency], 2, 0).astype(dtype)
        )
        sky_slice = resolved.asarray(sky[frequency].astype(dtype))
        blocks.append(
            forward_per_m_product(
                transfer_block=transfer_slice,
                sky_block=sky_slice[:, None, None, :],
                field_order=fields,
                backend=resolved,
            )
        )
    stacked = resolved.stack(blocks, axis=1)
    return np.asarray(resolved.to_numpy(stacked), dtype=dtype)


def _exact_turn(value: str | Fraction | int) -> Fraction:
    """Return one ERA turn as an exact rational.

    Section 6 requires the synthesis to consume the retained exact turns
    directly and forbids regenerating topology "from ``k``, ``N``, radians, or
    ``tau``".  A caller may spell a turn as a :class:`~fractions.Fraction`, an
    integer, or a ``p/q`` string; the value is exact either way, and the
    canonical normalized spelling Section 3.1 fixes governs *retained record*
    bytes rather than this argument.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(str(value))


def synthesize_time_series(
    *,
    mode_cube: np.ndarray,
    era_turns: Sequence[str],
    exposure_width_turn: str | Fraction | None = None,
    backend: Any = None,
) -> np.ndarray:
    r"""Synthesize the time-domain cube from retained ``m`` modes.

    Section 6's exposure-averaged synthesis is

    .. math:: \bar V_k=\sum_{m=-m_{max}}^{m_{max}}w_m v_m e^{+i2\pi m u_k},

    with ``w_m = sinc(pi m Delta_u)`` the fixed ERA exposure top hat -- "a
    diagonal ``w_m`` factor, not a spectral taper".  Every ``u_k`` and
    ``Delta_u`` is the retained exact rational from the same
    ``CanonicalEraGrid``: the synthesis consumes those turns directly and never
    regenerates topology from ``k``, ``N``, radians, or ``tau``.
    """
    resolved = backend if backend is not None else _numpy_backend()
    modes = np.asarray(mode_cube, dtype=np.complex128)
    if modes.ndim != 4:
        raise ValueError("the mode cube is (baseline, frequency, correlation, m)")
    signed = int(modes.shape[3])
    if signed % 2 != 1:
        raise ValueError("the signed-m axis has an odd cardinality 2*mmax+1")
    mmax = (signed - 1) // 2
    turns = [_exact_turn(turn) for turn in era_turns]
    samples = len(turns)
    if samples < 2 * mmax + 1:
        raise ValueError("N >= 2*mmax+1 is mandatory for the retained modes")

    orders = np.arange(-mmax, mmax + 1, dtype=np.int64)
    window = np.ones(signed, dtype=np.float64)
    if exposure_width_turn is not None:
        width = _exact_turn(exposure_width_turn)
        for index, order in enumerate(orders):
            if order == 0:
                continue
            argument = math.pi * float(order) * float(width)
            window[index] = math.sin(argument) / argument

    # ``exp(+i 2 pi m u_k)`` from the exact rational turn, through a correctly
    # rounded unit-circle kernel rather than a regenerated radian array.
    phases = np.empty((samples, signed), dtype=np.complex128)
    for row, turn in enumerate(turns):
        for column, order in enumerate(orders):
            product = Fraction(int(order)) * turn
            fractional = float(product - int(product))
            phases[row, column] = complex(
                math.cos(TAU * fractional), math.sin(TAU * fractional)
            )
    weighted = resolved.asarray(modes * window[None, None, None, :])
    kernel = resolved.asarray(phases)
    contracted = resolved.xp.einsum("bfcm,km->kbfc", weighted, kernel)
    return np.asarray(resolved.to_numpy(contracted), dtype=np.complex128)


# ---------------------------------------------------------------------------
# Section 9 / 11 memory estimate and deterministic block schedule
# ---------------------------------------------------------------------------


#: Section 14.0's domain for the retained schedule digest.
SCHEDULE_DIGEST_DOMAIN: Final = "radiosim.sci004.block-schedule.v1"

#: Section 9's seven separately reported estimate components, in its own order.
MEMORY_COMPONENTS: Final[tuple[str, ...]] = (
    "canonical_sky_coefficients",
    "quadrature_directions_weights_and_jones",
    "per_antenna_harmonic_cache",
    "largest_baseline_transfer_block",
    "retained_mmode_visibilities",
    "time_domain_output_and_synthesis",
    "backend_native_allocations",
)

_COMPLEX128_BYTES: Final[int] = 16
_FLOAT64_BYTES: Final[int] = 8


def _packed_widths(lmax: int, mmax: int) -> list[int]:
    """Return the packed width of each signed-``m`` block, ascending in ``m``."""
    from radiosim.core.mmode.harmonics import polarized_packed_block_table

    table = polarized_packed_block_table(lmax=int(lmax), mmax=int(mmax))
    widths = [0] * (2 * int(mmax) + 1)
    for row in table.block_rows:
        widths[int(row["m"]) + int(mmax)] += int(row["value_stop"]) - int(
            row["value_start"]
        )
    return widths


@dataclass(frozen=True, slots=True)
class MModeMemoryEstimate:
    """Section 9's component-by-component m-mode memory estimate."""

    components: Mapping[str, int]
    logical_dimensions: Mapping[str, int]
    scheduled_dimensions: Mapping[str, int]
    one_block_minimum_bytes: int
    complete_baseline_transfer_bytes: int
    working_memory_bytes: int

    @property
    def total_bytes(self) -> int:
        """Return the summed host estimate across the seven components."""
        return int(sum(self.components.values()))

    def as_mapping(self) -> dict[str, Any]:
        """Return a plain mapping for provenance."""
        return {
            "components": dict(self.components),
            "logical_dimensions": dict(self.logical_dimensions),
            "scheduled_dimensions": dict(self.scheduled_dimensions),
            "one_block_minimum_bytes": int(self.one_block_minimum_bytes),
            "complete_baseline_transfer_bytes": int(
                self.complete_baseline_transfer_bytes
            ),
            "working_memory_bytes": int(self.working_memory_bytes),
        }


@dataclass(frozen=True, slots=True)
class MModeBlockSchedule:
    """Section 11's ``resolved_block_dimensions`` for one resolved run."""

    frequency_block_max: int
    signed_m_block_max: int
    baseline_block_max: int
    packed_value_block_max: int
    schedule_rows: tuple[Mapping[str, int], ...]
    schedule_sha256: str

    @property
    def scheduled_block_count(self) -> int:
        """Return the non-empty schedule length."""
        return len(self.schedule_rows)

    def as_mapping(self) -> dict[str, Any]:
        """Return Section 11's exact ordered key set."""
        return {
            "frequency_block_max": int(self.frequency_block_max),
            "signed_m_block_max": int(self.signed_m_block_max),
            "baseline_block_max": int(self.baseline_block_max),
            "packed_value_block_max": int(self.packed_value_block_max),
            "scheduled_block_count": int(self.scheduled_block_count),
            "schedule_rows": [dict(row) for row in self.schedule_rows],
            "schedule_sha256": self.schedule_sha256,
        }


def _fixed_component_bytes(
    *,
    n_baselines: int,
    n_frequencies: int,
    n_antennas: int,
    sidereal_samples: int,
    packed_total: int,
    scalar_packed_total: int,
    signed_m: int,
    n_directions: int,
) -> dict[str, int]:
    """Return the six budget components that do not depend on the block size."""
    return {
        "canonical_sky_coefficients": n_frequencies * packed_total * _COMPLEX128_BYTES,
        # Directions and weights, plus the sampled per-antenna Jones fields the
        # transfer integrand consumes on the same grid.
        "quadrature_directions_weights_and_jones": n_directions
        * (4 * _FLOAT64_BYTES + n_antennas * 4 * _COMPLEX128_BYTES),
        "per_antenna_harmonic_cache": n_antennas
        * n_frequencies
        * 2
        * 2
        * scalar_packed_total
        * _COMPLEX128_BYTES,
        "retained_mmode_visibilities": n_baselines
        * n_frequencies
        * 4
        * signed_m
        * _COMPLEX128_BYTES,
        "time_domain_output_and_synthesis": sidereal_samples
        * n_baselines
        * n_frequencies
        * 4
        * _COMPLEX128_BYTES,
        # Section 9 reports backend/native allocations "not included in the host
        # estimate" as their own row; the host-side reference backend adds none.
        "backend_native_allocations": 0,
    }


def _harmonic_matrix_bytes(*, n_directions: int, packed_in_block: int) -> int:
    """Return the ``(direction, packed_value)`` harmonic matrix a block needs."""
    return int(n_directions * packed_in_block * _COMPLEX128_BYTES)


def _transfer_block_bytes(
    *, packed_in_block: int, baselines: int, frequencies: int
) -> int:
    """Return the ``(B, F, C, packed)`` transfer block Section 9 budgets."""
    return int(baselines * frequencies * 4 * packed_in_block * _COMPLEX128_BYTES)


def _block_bytes(
    *, n_directions: int, packed_in_block: int, baselines: int, frequencies: int
) -> int:
    """Return the working bytes of one streamed transfer block.

    Two allocations dominate and both scale with the block's packed width: the
    ``(direction, packed_value)`` harmonic matrix the block's ``B_lm`` integral
    contracts against, and the resulting
    ``(baseline, frequency, correlation, packed_value)`` transfer block itself.
    The block is discarded after its contraction, which is why only one appears
    in the budget.
    """
    return _harmonic_matrix_bytes(
        n_directions=n_directions, packed_in_block=packed_in_block
    ) + _transfer_block_bytes(
        packed_in_block=packed_in_block, baselines=baselines, frequencies=frequencies
    )


def _resolve_extents(
    *,
    widths: Sequence[int],
    n_baselines: int,
    n_frequencies: int,
    n_directions: int,
    available: int,
) -> tuple[int, int, int]:
    """Choose Section 9's largest fitting block in frequency/signed-m/baseline order.

    Section 9 gives two rules that act together.  "The deterministic scheduler
    orders frequency, signed-``m``, and baseline blocks, choosing the largest
    block that fits ``working_memory_bytes`` under a component-by-component
    estimate" selects *within* the invariant that "the complete baseline
    transfer is never materialized", so the chosen extents are the largest that
    fit **and** never the whole cube at once: a block spanning every frequency,
    every signed ``m`` and every baseline would be exactly the materialization
    the first sentence forbids, so the outermost axis is reduced by one.

    The scheduler does not inspect free RAM and does not change block order
    after an allocation failure.
    """
    signed = len(widths)

    def fits(frequencies: int, orders: int, baselines: int) -> bool:
        widest = max(
            int(sum(widths[start : start + orders]))
            for start in range(0, signed, orders)
        )
        return (
            _block_bytes(
                n_directions=n_directions,
                packed_in_block=widest,
                baselines=baselines,
                frequencies=frequencies,
            )
            <= available
        )

    frequency_extent, order_extent, baseline_extent = 1, 1, 1
    if not fits(1, 1, 1):
        return (0, 0, 0)
    for candidate in range(n_frequencies, 0, -1):
        if fits(candidate, signed, n_baselines):
            return _cap_complete(
                candidate, signed, n_baselines, n_frequencies, signed, n_baselines
            )
    for candidate in range(signed, 0, -1):
        if fits(1, candidate, n_baselines):
            order_extent = candidate
            baseline_extent = n_baselines
            return _cap_complete(
                1, order_extent, baseline_extent, n_frequencies, signed, n_baselines
            )
    for candidate in range(n_baselines, 0, -1):
        if fits(1, 1, candidate):
            baseline_extent = candidate
            break
    return _cap_complete(
        frequency_extent,
        order_extent,
        baseline_extent,
        n_frequencies,
        signed,
        n_baselines,
    )


def _cap_complete(
    frequencies: int,
    orders: int,
    baselines: int,
    n_frequencies: int,
    signed: int,
    n_baselines: int,
) -> tuple[int, int, int]:
    """Keep the largest block strictly inside the complete baseline transfer."""
    if (frequencies, orders, baselines) != (n_frequencies, signed, n_baselines):
        return (frequencies, orders, baselines)
    if n_frequencies > 1:
        return (n_frequencies - 1, orders, baselines)
    if signed > 1:
        return (frequencies, signed - 1, baselines)
    if n_baselines > 1:
        return (frequencies, orders, n_baselines - 1)
    # A single-cell run has nothing to stream; the block *is* the transfer.
    return (frequencies, orders, baselines)


def estimate_mmode_memory(
    *,
    n_baselines: int,
    n_frequencies: int,
    lmax: int,
    mmax: int,
    quadrature_nside: int,
    working_memory_bytes: int,
    n_antennas: int = 2,
    sidereal_samples: int | None = None,
) -> MModeMemoryEstimate:
    """Return Section 9's seven-component m-mode memory estimate.

    ``sidereal_samples`` defaults to the mandatory Section 6 floor
    ``2 * mmax + 1``, which is the smallest admissible grid for the retained
    modes, so an estimate taken before a time grid is resolved is a lower bound
    on the output component rather than a guess.

    Raises
    ------
    ValueError
        If ``working_memory_bytes`` is below the one-block minimum.  Section 9:
        "A budget smaller than that minimum is rejected before allocation."
    """
    from radiosim.core.mmode.harmonics import scalar_packed_block_table

    widths = _packed_widths(lmax, mmax)
    signed = len(widths)
    samples = int(sidereal_samples) if sidereal_samples else 2 * int(mmax) + 1
    packed_total = int(sum(widths))
    scalar_total = scalar_packed_block_table(
        lmax=int(lmax), mmax=int(mmax)
    ).packed_value_count
    n_directions = 12 * int(quadrature_nside) * int(quadrature_nside)

    fixed = _fixed_component_bytes(
        n_baselines=int(n_baselines),
        n_frequencies=int(n_frequencies),
        n_antennas=int(n_antennas),
        sidereal_samples=samples,
        packed_total=packed_total,
        scalar_packed_total=int(scalar_total),
        signed_m=signed,
        n_directions=n_directions,
    )
    fixed_total = int(sum(fixed.values()))
    minimum_block = _block_bytes(
        n_directions=n_directions,
        packed_in_block=max(widths),
        baselines=1,
        frequencies=1,
    )
    one_block_minimum = fixed_total + minimum_block
    budget = int(working_memory_bytes)
    if budget < one_block_minimum:
        raise ValueError(
            f"working_memory_bytes={budget} is below the one-block minimum "
            f"{one_block_minimum}; SCI-004 Section 9 rejects it before allocation"
        )

    frequency_extent, order_extent, baseline_extent = _resolve_extents(
        widths=widths,
        n_baselines=int(n_baselines),
        n_frequencies=int(n_frequencies),
        n_directions=n_directions,
        available=budget - fixed_total,
    )
    widest = max(
        int(sum(widths[start : start + order_extent]))
        for start in range(0, signed, order_extent)
    )
    components = dict(fixed)
    # Section 9 budgets the transfer *block* and the sampled quadrature fields
    # as separate rows, so the block's harmonic matrix -- the harmonics sampled
    # on the quadrature directions -- is reported with the grid it lives on
    # rather than folded into the transfer block it contracts against.
    components["quadrature_directions_weights_and_jones"] += _harmonic_matrix_bytes(
        n_directions=n_directions, packed_in_block=widest
    )
    components["largest_baseline_transfer_block"] = _transfer_block_bytes(
        packed_in_block=widest,
        baselines=baseline_extent,
        frequencies=frequency_extent,
    )
    ordered = {name: int(components[name]) for name in MEMORY_COMPONENTS}
    return MModeMemoryEstimate(
        components=MappingProxyType(ordered),
        logical_dimensions=MappingProxyType(
            {
                "n_baselines": int(n_baselines),
                "n_frequencies": int(n_frequencies),
                "n_antennas": int(n_antennas),
                "sidereal_samples": samples,
                "signed_m": signed,
                "packed_value_count": packed_total,
                "n_directions": n_directions,
            }
        ),
        scheduled_dimensions=MappingProxyType(
            {
                "frequency_block_max": frequency_extent,
                "signed_m_block_max": order_extent,
                "baseline_block_max": baseline_extent,
                "packed_value_block_max": widest,
            }
        ),
        one_block_minimum_bytes=int(one_block_minimum),
        complete_baseline_transfer_bytes=int(
            int(n_baselines) * int(n_frequencies) * 4 * packed_total * _COMPLEX128_BYTES
        ),
        working_memory_bytes=budget,
    )


def schedule_mmode_blocks(
    *,
    n_baselines: int,
    n_frequencies: int,
    lmax: int,
    mmax: int,
    quadrature_nside: int,
    working_memory_bytes: int,
    n_antennas: int = 2,
    sidereal_samples: int | None = None,
) -> MModeBlockSchedule:
    """Build Section 11's deterministic frequency/signed-``m``/baseline schedule.

    Rows are in actual canonical execution order, ``block_index`` is contiguous
    from zero, ranges are half-open, and every maximum is recomputed from the
    rows.  "Missing, duplicate, reordered, overlapping, or uncovered work is
    invalid", so the enumeration below covers the complete
    ``frequency x signed-m x baseline`` product exactly once.
    """
    estimate = estimate_mmode_memory(
        n_baselines=n_baselines,
        n_frequencies=n_frequencies,
        lmax=lmax,
        mmax=mmax,
        quadrature_nside=quadrature_nside,
        working_memory_bytes=working_memory_bytes,
        n_antennas=n_antennas,
        sidereal_samples=sidereal_samples,
    )
    widths = _packed_widths(lmax, mmax)
    signed = len(widths)
    frequency_extent = int(estimate.scheduled_dimensions["frequency_block_max"])
    order_extent = int(estimate.scheduled_dimensions["signed_m_block_max"])
    baseline_extent = int(estimate.scheduled_dimensions["baseline_block_max"])

    rows: list[dict[str, int]] = []
    for frequency_start in range(0, int(n_frequencies), frequency_extent):
        frequency_stop = min(frequency_start + frequency_extent, int(n_frequencies))
        for order_start in range(0, signed, order_extent):
            order_stop = min(order_start + order_extent, signed)
            packed = int(sum(widths[order_start:order_stop]))
            for baseline_start in range(0, int(n_baselines), baseline_extent):
                baseline_stop = min(baseline_start + baseline_extent, int(n_baselines))
                rows.append(
                    {
                        "block_index": len(rows),
                        "frequency_start": frequency_start,
                        "frequency_stop": frequency_stop,
                        "signed_m_start": order_start,
                        "signed_m_stop": order_stop,
                        "baseline_start": baseline_start,
                        "baseline_stop": baseline_stop,
                        "packed_value_count": packed,
                    }
                )
    if not rows:  # pragma: no cover - defensive
        raise ValueError("the deterministic schedule covered no work")
    digest = domain_digest(SCHEDULE_DIGEST_DOMAIN, canonical_json(rows))
    return MModeBlockSchedule(
        frequency_block_max=max(
            row["frequency_stop"] - row["frequency_start"] for row in rows
        ),
        signed_m_block_max=max(
            row["signed_m_stop"] - row["signed_m_start"] for row in rows
        ),
        baseline_block_max=max(
            row["baseline_stop"] - row["baseline_start"] for row in rows
        ),
        packed_value_block_max=max(row["packed_value_count"] for row in rows),
        schedule_rows=tuple(MappingProxyType(dict(row)) for row in rows),
        schedule_sha256=digest,
    )


# ---------------------------------------------------------------------------
# Section 7.3 full-Stokes fixture solve and its private direct oracle
# ---------------------------------------------------------------------------


#: The phase-M2 acceptance site.  Section 7.3 does not let a fixture be
#: inherited: "A candidate fixture is qualified by measuring its three-level
#: deficit sequence and adopting it only with real margin on the ``2x`` floor; a
#: predicate is never widened to admit a fixture", and the convergent regime's
#: governing conditions are geometric -- every payload direction must stay well
#: clear of the horizon over the whole cycle.  The accepted M1 site's latitude
#: leaves the pinned ``-75`` degree source at ``15.7`` degrees altitude at lower
#: culmination, which is *not* well clear: no beam can separate a source at
#: ``sin(theta) = 0.96`` from the horizon at ``1.0``, and the measured control
#: margin plateaus at ``7.4`` against the required ``10``.  A high-latitude site
#: puts the same pinned source between ``65`` and ``85`` degrees altitude for the
#: whole cycle.  Longitude, height and epoch are the accepted M1 ones.
FIXTURE_LONGITUDE_DEG: Final[float] = 21.42830
FIXTURE_LATITUDE_DEG: Final[float] = -70.0
FIXTURE_HEIGHT_M: Final[float] = 1073.0
FIXTURE_START_TIME_ISO: Final[str] = "2025-01-01T00:00:00"

#: The qualified aperture: a uniformly illuminated circular aperture whose
#: first Airy null sits *on* the horizon at the band's lower edge.  ``J1``'s
#: first zero is at ``3.8317``, so ``pi D sin(theta) / lambda = 3.8317`` at
#: ``sin(theta) = 1`` gives ``D / lambda = 1.2197`` -- ``7.32 m`` at ``50 MHz``.
#: That is the second half of Section 7.3's geometric qualification and it is a
#: property of the aperture, not a cutoff: the strict horizon factor then
#: multiplies a response already at its null, so the kernel carries no effective
#: step, its harmonic content stays well inside the retained band, and the
#: truncation deficit is set by the beam rather than by Gibbs ringing.  With a
#: horizon-responding aperture the deficit floors near two percent of the signal
#: and the wrong-bridge control's measured margin plateaus below its required
#: ``10``, whatever the diameter.
#:
#: The baseline is short in wavelengths for the same reason: fringe structure is
#: harmonic content too, and the measured control margin falls monotonically
#: from ``16.6`` to ``4.1`` as the baseline grows from ``2`` to ``8`` metres.
#: This is a numerical acceptance fixture -- two phase centres, an analytic
#: aperture pattern and a frozen frame -- qualified by Section 7.3's measured
#: procedure, and deliberately not a buildable array: a filled aperture this
#: wide could not sit on a baseline this short.
FIXTURE_BASELINE_EAST_M: Final[float] = 4.0
FIXTURE_DISH_DIAMETER_M: Final[float] = 7.32
FIXTURE_BEAM: Final[str] = "heterogeneous"
FIXTURE_FREQUENCIES_HZ: Final[tuple[float, ...]] = (50.0e6, 51.0e6, 52.0e6)


def _iso_gauss_spherical(nside: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Section 7.3's iso-Gauss ``(colatitude, longitude, weight)`` arrays.

    The grid is defined in whatever frame the expansion uses.  Section 4.1 makes
    the *sky* the fixed frame and the instrument the rotating one, so a fixture
    grid lives in CIRS and the instrument is mapped onto it.
    """
    from radiosim.core.mmode.transfer import quadrature_grid

    directions, weights = quadrature_grid(int(nside))
    colatitude = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    longitude = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * math.pi)
    return (colatitude, longitude, weights)


def _spherical_to_cartesian(
    colatitude: np.ndarray, longitude: np.ndarray
) -> np.ndarray:
    """Return unit vectors from spherical angles, as ``(n_dir, 3)``."""
    sine = np.sin(colatitude)
    return np.stack(
        (sine * np.cos(longitude), sine * np.sin(longitude), np.cos(colatitude)),
        axis=-1,
    )


def _fixture_frame() -> FrozenFrame:
    """Build the accepted frozen-CIRS rigid-ERA frame of the fixture site."""
    return build_frozen_frame(
        start_time=FIXTURE_START_TIME_ISO,
        longitude_deg=FIXTURE_LONGITUDE_DEG,
        latitude_deg=FIXTURE_LATITUDE_DEG,
        height_m=FIXTURE_HEIGHT_M,
    )


#: Section 7.3's qualifying geometry for the native HEALPix payload: a smooth
#: polar cap of this angular radius about the *south* celestial pole.  Section
#: 7.3's convergent regime is geometric -- "every point and native payload
#: direction must stay well clear of the horizon over the whole cycle, because
#: near-horizon samples carry a non-decaying Gibbs error that defeats the
#: monotone predicate at any scale" -- and at the shipped site every direction
#: within this cap has declination below ``-70`` degrees, so it is circumpolar
#: with real margin on the ``-59.28`` degree limit.  A full-sky native payload
#: cannot qualify at all: it always has pixels on the horizon, and the measured
#: consequence is the collapsed factor a full-sky fixture map produces.
FIXTURE_HEALPIX_CAP_RADIUS_RAD: Final[float] = 0.34


def fixture_healpix_maps(colatitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Return the qualified full-Stokes fixture maps, shaped ``(npix, 4)``.

    One code object builds the native payload for *both* sides of the Section
    7.3 comparison -- the harmonic projection in :func:`_fixture_sky` and the
    private direct sum in :func:`polarized_direct_cube` -- so the two cannot
    silently drift into integrating different skies.  Columns are the RadioSim
    IAU Stokes order ``(I, Q, U, V)``; Section 5.2's bridge to the Shaw fields
    is applied once, downstream, by the shared
    :func:`~radiosim.core.polarization.stokes_to_shaw_fields`.

    The payload is a raised-cosine cap about the south celestial pole, which is
    what puts the fixture in Section 7.3's convergent regime: it vanishes
    smoothly at the cap edge, carries no weight anywhere near the horizon, and
    its angular width sets harmonic content the ``lmax//4`` level cannot
    resolve, so the three-level deficit sequence decreases strictly instead of
    sitting on the horizon-step floor.  ``Q``/``U`` carry the ``cos``/``sin`` of
    twice the azimuth that a genuine spin-2 field needs; ``I`` and ``V`` are
    scalar.
    """
    angle = math.pi - np.asarray(colatitude, dtype=np.float64)
    azimuth = np.asarray(longitude, dtype=np.float64)
    inside = angle <= FIXTURE_HEALPIX_CAP_RADIUS_RAD
    profile = np.where(
        inside,
        np.cos(0.5 * math.pi * angle / FIXTURE_HEALPIX_CAP_RADIUS_RAD) ** 2,
        0.0,
    )
    return np.stack(
        (
            4.0 * profile,
            0.9 * profile * np.cos(2.0 * azimuth),
            0.7 * profile * np.sin(2.0 * azimuth),
            0.5 * profile * np.cos(angle),
        ),
        axis=-1,
    )


def _fixture_kernel_components(
    *,
    frame: FrozenFrame,
    celestial_directions: np.ndarray,
    frequency_hz: float,
    horizon_free: bool,
    correlation_labels: Sequence[str],
    relative_phase_rad: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return ``K^X_c`` on a celestial direction set, per Stokes component.

    Every factor is the accepted one: the horizon predicate is
    :func:`strict_horizon_visible` -- Section 6's single shared code object --
    the fringe is the existing ``geometric_phase``, the beam is the shipped
    analytic aperture, and the receptor/bridge composition is
    ``receptor_component_matrices``.

    No tangent rotation enters.  Section 6 anchors the accepted M2 scope to the
    direct RIME's own basis: the coherency is built in the celestial North/East
    tangent basis of each direction, the direction-independent chain terms --
    the receptor factors among them -- right-multiply it as constant matrices in
    that same basis, and every mount-dependent tangent rotation belongs to the
    ``P`` term, which is exactly the identity for the shipped ``fixed`` and
    unspecified mounts this fixture uses.  Constant cells are constant
    coefficients on spin-weighted fields, so they preserve the integrand's spin
    weight and Section 7.3's spin-``+-2`` Gauss-Legendre quadrature stays
    spectrally exact -- measured ``3.84e-15`` Q-only against the ``1.01e-8``
    limit.  A genuinely ground-anchored direction-dependent response would enter
    through Section 6's measured transport instead; transporting a *constant*
    ground matrix is the identity re-expression of a zenith-singular field
    (``e^{2 i chi}`` winds twice about the local zenith, measured spread exactly
    ``2.0000``) and is rejected rather than an alternative convention.

    The terrestrial directions therefore serve exactly three factors -- the
    horizon predicate, the fringe and the beam -- while every harmonic
    evaluation stays on the celestial angles.
    """
    from radiosim.backends import get_backend
    from radiosim.core.jones import geometric_phase
    from radiosim.core.mmode.transfer import (
        _scalar_beam_response,
        receptor_component_matrices,
        receptor_row_indices,
    )

    attitude = frame.attitude_at(float(relative_phase_rad))
    terrestrial = _enu_components(
        frame, np.asarray(celestial_directions, dtype=np.float64) @ attitude.T
    )
    horizon = strict_horizon_visible(terrestrial[:, 2])
    horizon_weight = (
        np.ones(terrestrial.shape[0], dtype=np.float64)
        if horizon_free
        else horizon.astype(np.float64)
    )
    beam_directions = terrestrial
    if horizon_free:
        beam_directions = np.array(terrestrial, copy=True)
        beam_directions[:, 2] = np.abs(beam_directions[:, 2])

    wavelength = _SPEED_OF_LIGHT_M_PER_S / float(frequency_hz)
    baseline = np.array([[FIXTURE_BASELINE_EAST_M, 0.0, 0.0]], dtype=np.float64)
    fringe = np.asarray(
        geometric_phase(
            uvw_wavelengths=baseline / wavelength,
            dir_l=terrestrial[:, 0],
            dir_m=terrestrial[:, 1],
            dir_n=terrestrial[:, 2],
            backend=get_backend("numpy"),
        ),
        dtype=np.complex128,
    )[0]
    response = _scalar_beam_response(
        beam_directions,
        beam=FIXTURE_BEAM,
        diameter_m=FIXTURE_DISH_DIAMETER_M,
        frequency_hz=float(frequency_hz),
    )
    common = response * np.conjugate(response) * fringe * horizon_weight

    # ``M P^X M^H`` for ``M = C D`` -- the one shared composition the harmonic
    # transfer uses, in the celestial tangent basis both sides expand in.
    components = receptor_component_matrices(correlation_labels)
    kernels: dict[str, np.ndarray] = {}
    rows = receptor_row_indices(correlation_labels)
    for index, label in enumerate(correlation_labels):
        row, column = rows[label[0]], rows[label[1]]
        for name in ("I", "Q", "U", "V"):
            kernels[f"{index}:{name}"] = common * complex(components[name][row][column])
    return kernels


def _field_kernels(
    kernels: Mapping[str, np.ndarray], correlation: int
) -> dict[str, np.ndarray]:
    """Return Section 6's four field integrands for one correlation.

    The combination itself is :func:`field_integrands` -- the one code object the
    production transfer also uses -- so the conjugate placement Section 6 pins by
    explicit numerical integral cannot differ between the two.
    """
    return field_integrands(
        {name: kernels[f"{correlation}:{name}"] for name in STOKES_COMPONENT_ORDER}
    )


def _fixture_transfer(
    *,
    frame: FrozenFrame,
    lmax: int,
    mmax: int,
    nside: int,
    frequency_hz: float,
    horizon_free: bool,
    correlation_labels: Sequence[str],
) -> Any:
    """Return the celestial-frame polarized ``B^X_lm`` of the fixture baseline."""
    from radiosim.core.mmode.harmonics import (
        field_columns,
        packed_polarized_conjugate_harmonics,
        polarized_packed_block_table,
    )

    table = polarized_packed_block_table(lmax=int(lmax), mmax=int(mmax))
    colatitude, longitude, weights = _iso_gauss_spherical(int(nside))
    directions = _spherical_to_cartesian(colatitude, longitude)
    kernels = _fixture_kernel_components(
        frame=frame,
        celestial_directions=directions,
        frequency_hz=frequency_hz,
        horizon_free=horizon_free,
        correlation_labels=correlation_labels,
    )
    # Section 6 expands the transfer in *unconjugated* harmonics against the
    # conjugated sky expansion.
    harmonics = np.conjugate(
        packed_polarized_conjugate_harmonics(table, colatitude, longitude)
    )
    packed = np.zeros(
        (len(correlation_labels), table.packed_value_count), np.complex128
    )
    columns = {name: field_columns(table, name) for name in table.field_order}
    for index in range(len(correlation_labels)):
        fields = _field_kernels(kernels, index)
        for name in table.field_order:
            selected = columns[name]
            if selected.size == 0:
                continue
            packed[index, selected] = (fields[name] * weights) @ harmonics[:, selected]
    return (table, packed)


def _field_weight_columns(table: Any) -> np.ndarray:
    """Return Section 6's per-column forward weight (``1/2`` on the spin pair)."""
    weights = np.ones(table.packed_value_count, dtype=np.float64)
    for row in table.block_rows:
        start, stop = int(row["value_start"]), int(row["value_stop"])
        if stop > start:
            weights[start:stop] = SPIN_FIELD_WEIGHTS[str(row["field_name"])]
    return weights


def _per_signed_m(table: Any, product: np.ndarray) -> np.ndarray:
    """Sum a packed product into its ``2*mmax+1`` signed-``m`` bins."""
    bins = np.zeros(2 * table.mmax + 1, dtype=np.complex128)
    for row in table.block_rows:
        start, stop = int(row["value_start"]), int(row["value_stop"])
        if stop > start:
            bins[int(row["m"]) + table.mmax] += np.sum(product[start:stop])
    return bins


def _fixture_sky(
    *,
    table: Any,
    representation: str,
    dec_deg: float,
    stokes: Sequence[float],
    nside: int | None,
) -> Any:
    """Return the fixture's packed full-Stokes sky coefficients."""
    from radiosim.core.mmode.sky import (
        healpix_polarized_coefficients,
        hybrid_polarized_coefficients,
        point_polarized_coefficients,
        ring_directions,
    )
    from radiosim.core.sky.containers import TangentPolarizationFrame

    frame_block = TangentPolarizationFrame.canonical("icrs").as_mapping()
    point = None
    healpix = None
    if representation in {"point", "hybrid"}:
        point = point_polarized_coefficients(
            ra_rad=[0.0],
            dec_rad=[math.radians(float(dec_deg))],
            stokes=[list(stokes)],
            lmax=table.lmax,
            mmax=table.mmax,
            tangent_frame=frame_block,
            table=table,
        )
    if representation in {"healpix", "hybrid"}:
        theta, phi = ring_directions(int(nside or 8))
        maps = dict(
            zip(("I", "Q", "U", "V"), fixture_healpix_maps(theta, phi).T, strict=True)
        )
        healpix = healpix_polarized_coefficients(
            maps,
            nside=int(nside or 8),
            order="ring",
            lmax=table.lmax,
            mmax=table.mmax,
            tangent_frame=frame_block,
            table=table,
        )
    if representation == "point":
        return point
    if representation == "healpix":
        return healpix
    return hybrid_polarized_coefficients(point=point, healpix=healpix)


def _fixture_cube(
    *,
    frame: FrozenFrame,
    lmax: int,
    mmax: int,
    nside: int,
    sidereal_samples: int,
    representation: str,
    dec_deg: float,
    stokes: Sequence[float],
    healpix_nside: int | None,
    horizon_free: bool,
    correlation_labels: Sequence[str],
) -> np.ndarray:
    """Synthesize one ``[N, B, F, 4]`` harmonic cube at a declared truncation."""
    samples = int(sidereal_samples)
    frequencies = FIXTURE_FREQUENCIES_HZ
    modes = np.zeros(
        (1, len(frequencies), len(correlation_labels), 2 * int(mmax) + 1),
        dtype=np.complex128,
    )
    for index, frequency in enumerate(frequencies):
        table, packed = _fixture_transfer(
            frame=frame,
            lmax=lmax,
            mmax=mmax,
            nside=nside,
            frequency_hz=frequency,
            horizon_free=horizon_free,
            correlation_labels=correlation_labels,
        )
        sky = _fixture_sky(
            table=table,
            representation=representation,
            dec_deg=dec_deg,
            stokes=stokes,
            nside=healpix_nside,
        )
        weights = _field_weight_columns(table)
        values = np.asarray(sky.values, dtype=np.complex128)
        for correlation in range(len(correlation_labels)):
            product = packed[correlation] * values * weights
            modes[0, index, correlation] = _per_signed_m(table, product)
    turns = [Fraction(step, samples) for step in range(samples)]
    return synthesize_time_series(mode_cube=modes, era_turns=turns)


def polarized_direct_cube(
    *,
    dec_deg: float,
    stokes: Sequence[float],
    sidereal_samples: int,
    quadrature_nside: int = 8,
    representation: str = "point",
    healpix_nside: int | None = None,
    correlation_labels: Sequence[str] = ("XX", "XY", "YX", "YY"),
    horizon_free: bool = False,
) -> np.ndarray:
    """Return the private full-Stokes frozen-frame direct cube.

    Section 7.1: the private direct oracle "does not resample a native HEALPix
    payload onto the transfer quadrature.  It sums the original native pixel
    centres in canonical RING order with their native pixel solid angle and
    resolved per-frequency Stokes payload."  A point component contributes
    analytically at its own transported direction, so a point sky's direct cube
    is exact rather than quadrature-limited, and the deficit the two-tier gate
    measures is purely the harmonic truncation.

    Every factor -- horizon predicate, fringe, beam, receptor composition and
    tangent transport -- is the same shared code object the transfer integrand
    uses, which is what Section 6 requires of the pair.
    """
    from radiosim.core.mmode.sky import ring_directions
    from radiosim.core.polarization import stokes_to_shaw_fields

    del quadrature_nside  # identifies the run's transfer grid, not this sum
    samples = int(sidereal_samples)
    frame = _fixture_frame()
    frequencies = FIXTURE_FREQUENCIES_HZ
    labels = tuple(str(label) for label in correlation_labels)
    cube = np.zeros((samples, 1, len(frequencies), len(labels)), dtype=np.complex128)

    contributions: list[tuple[np.ndarray, np.ndarray]] = []
    if representation in {"point", "hybrid"}:
        declination = math.radians(float(dec_deg))
        direction = np.asarray(
            [
                [
                    math.cos(declination),
                    0.0,
                    math.sin(declination),
                ]
            ],
            dtype=np.float64,
        )
        payload = np.asarray([list(stokes)], dtype=np.float64)
        contributions.append((direction, payload))
    if representation in {"healpix", "hybrid"}:
        resolution = int(healpix_nside or 8)
        theta, phi = ring_directions(resolution)
        solid_angle = 4.0 * math.pi / (12 * resolution * resolution)
        # The one shared payload builder both sides of the comparison use.
        maps = fixture_healpix_maps(theta, phi)
        contributions.append((_spherical_to_cartesian(theta, phi), maps * solid_angle))

    for sample in range(samples):
        phase = TAU * float(Fraction(sample, samples))
        for index, frequency in enumerate(frequencies):
            for directions, payload in contributions:
                kernels = _fixture_kernel_components(
                    frame=frame,
                    celestial_directions=directions,
                    frequency_hz=frequency,
                    horizon_free=horizon_free,
                    correlation_labels=labels,
                    relative_phase_rad=phase,
                )
                # Section 5.2's bridge, through the one shared code object.  The
                # kernel cells are ``M P^X_Shaw M^H``, so they are responses to
                # the *Shaw* fields ``(I_H, Q_H, U_H, V_H)``; contracting them
                # with an unbridged RadioSim ``U`` would sign-flip exactly one
                # of four contributions and make this oracle a different sky
                # from the harmonic side, which expands the same bridge in
                # ``resolve_stokes_fields``.
                shaw = np.stack(
                    stokes_to_shaw_fields(
                        payload[:, 0], payload[:, 1], payload[:, 2], payload[:, 3]
                    ),
                    axis=-1,
                )
                for correlation in range(len(labels)):
                    total = (
                        kernels[f"{correlation}:I"] * shaw[:, 0]
                        + kernels[f"{correlation}:Q"] * shaw[:, 1]
                        + kernels[f"{correlation}:U"] * shaw[:, 2]
                        + kernels[f"{correlation}:V"] * shaw[:, 3]
                    )
                    cube[sample, 0, index, correlation] += complex(np.sum(total))
    return cube


@dataclass(frozen=True, slots=True)
class PolarizedFixtureOutcome:
    """One phase-M2 fixture solve and its every-run two-tier gate."""

    sky_representation: str
    cube: np.ndarray
    direct_gate: DirectGateRecord
    component_order: tuple[str, ...]
    transfer_grid_id: str
    native_direct_grid_id: str
    dimensions: MModeDimensions


def solve_polarized_fixture(
    *,
    sky_representation: str,
    lmax: int,
    mmax: int,
    quadrature_nside: int,
    sidereal_samples: int,
    dec_deg: float = -75.0,
    stokes: Sequence[float] = (5.5, 0.8, -0.6, 0.4),
    nside: int | None = None,
    polarized: bool = True,
    correlation_labels: Sequence[str] = ("XX", "XY", "YX", "YY"),
) -> PolarizedFixtureOutcome:
    """Solve one full-Stokes fixture and evaluate Section 7.3's two-tier gate.

    The gate "executes on every production run before any result or output path
    is created".  Tier 1a evaluates the complete pipeline once with every
    horizon truncation removed on both the production and ``qcheck``
    quadratures; tier 1b records the with-horizon shell; tier 2 measures the
    truncation deficit against the complete frozen direct cube at ``L1``, ``L2``
    and ``lmax`` and gates on strict monotone convergence.

    The deficit is never called agreement: its obligations are convergence and
    disclosure.
    """
    del polarized  # a fixture is full Stokes by construction
    from radiosim.core.mmode.types import derive_mmode_dimensions

    representation = str(sky_representation)
    if representation not in {"point", "healpix", "hybrid"}:
        raise ValueError(f"unsupported fixture representation {representation!r}")
    dimensions = derive_mmode_dimensions(
        lmax=int(lmax), mmax=int(mmax), quadrature_nside=int(quadrature_nside)
    )
    frame = _fixture_frame()
    labels = tuple(str(label) for label in correlation_labels)
    healpix_nside = (
        int(nside) if nside is not None else (8 if representation != "point" else None)
    )

    def build(level_lmax: int, level_mmax: int, grid: int, ablated: bool) -> np.ndarray:
        return _fixture_cube(
            frame=frame,
            lmax=level_lmax,
            mmax=level_mmax,
            nside=grid,
            sidereal_samples=int(sidereal_samples),
            representation=representation,
            dec_deg=dec_deg,
            stokes=stokes,
            healpix_nside=healpix_nside,
            horizon_free=ablated,
            correlation_labels=labels,
        )

    production = build(int(lmax), int(mmax), int(quadrature_nside), False)
    quadrature_shell = build(int(lmax), int(mmax), dimensions.qcheck, False)
    horizon_free = build(int(lmax), int(mmax), int(quadrature_nside), True)
    horizon_free_qcheck = build(int(lmax), int(mmax), dimensions.qcheck, True)

    direct = polarized_direct_cube(
        dec_deg=dec_deg,
        stokes=stokes,
        sidereal_samples=int(sidereal_samples),
        quadrature_nside=int(quadrature_nside),
        representation=representation,
        healpix_nside=healpix_nside,
        correlation_labels=labels,
    )
    # Section 7.3: the qualified fixture is circumpolar with zero frozen horizon
    # roots, so its enclosure-error cube is exactly zero.
    enclosure_error = np.zeros(direct.shape, dtype=np.float64)

    quarter = max(2, int(lmax) // 4)
    half = max(quarter + 1, int(lmax) // 2)
    deficits: list[float] = []
    for level in (quarter, half):
        cube = build(level, min(int(mmax), level), int(quadrature_nside), False)
        deficits.append(float(np.max(np.abs(cube - direct) + enclosure_error)))

    gate = evaluate_two_tier_gate(
        mmode_cube=production,
        horizon_free_cube=horizon_free,
        horizon_free_qcheck_cube=horizon_free_qcheck,
        quadrature_shell_cube=quadrature_shell,
        frozen_gauss128=direct,
        frozen_enclosure_error=enclosure_error,
        deficit_max_quarter_jy=deficits[0],
        deficit_max_half_jy=deficits[1],
    )
    component_order = (
        ("point", "healpix") if representation == "hybrid" else (representation,)
    )
    return PolarizedFixtureOutcome(
        sky_representation=representation,
        cube=production,
        direct_gate=gate,
        component_order=component_order,
        transfer_grid_id=dimensions.production_grid_id,
        native_direct_grid_id=(
            f"native:{healpix_nside}" if healpix_nside else "native:point"
        ),
        dimensions=dimensions,
    )
