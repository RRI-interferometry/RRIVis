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

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
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
    MMODE_CONVENTION,
    MMODE_EXECUTION_POLICY,
    MMODE_FRAME_MODEL,
    MMODE_HARMONIC_CONVENTION,
    MMODE_QUADRATURE_POLICY,
    MMODE_STOKES_BRIDGE,
    MMODE_TANGENT_FRAME_M1,
    MMODE_TIME_GRID_CONVENTION,
    MMODE_TRUNCATION_POLICY,
    TAU,
    array_digest,
    canonical_rational,
    f64be,
    object_digest,
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
    "LedgerDirection",
    "MModeSolverSnapshot",
    "build_direction_ledger",
    "build_frame_certificate",
    "evaluate_two_tier_gate",
    "magnitude_ceiling",
    "solve_mmode",
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
    ``tangent_polarization_frame`` nor ``stokes_v_basis_bridge`` is nullable: in
    M1 the first is the exact literal ``not_applicable_scalar_m1`` and the
    second is always ``radiosim.stokes-ne-theta-phi.v1``.
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
            "tangent_polarization_frame": MMODE_TANGENT_FRAME_M1,
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
    context: KernelContext, enu: np.ndarray, *, horizon: bool = True
) -> np.ndarray:
    """Return ``K^I_{pqfc}`` on a direction batch, shaped ``(n_dir, B, F, 4)``.

    The Stokes-``I`` coherency is ``P^I = (1/2) I_2`` (the CLAUDE-normative half
    factor), the fringe is the *existing* geometric phase at its accepted sign,
    and the horizon factor is Section 6's one shared strict ``alt > 0``
    predicate with equality excluded -- no epsilon, beam cutoff, or half weight.

    Parameters
    ----------
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
    kernel = np.zeros((count, n_baselines, n_frequencies, 4), dtype=np.complex128)
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
            jones_p = responses[first]
            jones_q = responses[second]
            # ``[J_p P^I J_q^H]`` with ``P^I = (1/2) I_2``.
            coherency = 0.5 * np.einsum("nij,nkj->nik", jones_p, np.conjugate(jones_q))
            weighted = coherency.reshape(count, 4) * (
                fringe[baseline_index][:, None] * horizon_factor[:, None]
            )
            kernel[:, baseline_index, frequency_index, :] = weighted
    return kernel


_SPEED_OF_LIGHT_M_PER_S: Final[float] = 299792458.0


# ---------------------------------------------------------------------------
# Section 4.2 frame applicability certificate
# ---------------------------------------------------------------------------


def build_frame_certificate(
    *,
    grid: CanonicalEraGrid,
    frame: FrozenFrame,
    context: KernelContext,
    directions: Sequence[LedgerDirection],
    beam_peak_ceiling: float,
) -> FrameCertificate:
    """Compute the Section 4.2 certificate in memory, before harmonic work.

    Both censuses run over the complete direction ledger: the frozen analytic
    trajectory from its exact integer-ratio topology decision, and the
    operational trajectory from Section 12.1's certified-ceiling scan.  Their
    closed root enclosures then enter the unchanged pairing, lift, slab, sign
    and membership machinery, and the point/native rows drive the horizon-split
    Gauss-64/128 direct cubes with their certified magnitude-ceiling error
    disks.
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
        # Every cell is still classified and counted; the canonical row objects
        # are what the Section 14.2 evidence artifact embeds, and a complete
        # ledger over a full transfer catalogue is tens of millions of rows.
        retain_cells=False,
    )

    pair_rows, slab_rows, root_max_rad, mismatch_measure = _pair_roots(
        directions, frozen, scan.roots, grid
    )
    membership_rows, membership_mismatches = _membership(
        directions, frozen, scan.roots, grid, frame, context
    )
    direct = _direct_cubes(
        grid=grid,
        frame=frame,
        context=context,
        directions=directions,
        frozen=frozen,
        operational_roots=scan.roots,
        beam_peak_ceiling=beam_peak_ceiling,
    )

    paired = sum(len(row["pairs"]) for row in pair_rows)
    row: dict[str, Any] = {
        "schema_version": "radiosim.mmode-frame-certificate.v1",
        "site_sha256": frame.site_sha256,
        "frame_matrix_sha256": frame.frame_matrix_sha256,
        "iers_table_sha256": frame.iers_table_sha256,
        "canonical_era_turn_grid_sha256": grid.canonical_era_turn_grid_sha256,
        "canonical_era_grid_sha256": grid.canonical_era_grid_sha256,
        "horizon_scan_manifest": scan.manifest(),
        "horizon_isolation_interval_count": scan.isolation_interval_count,
        "horizon_unresolved_interval_count": 0,
        "horizon_root_pair_rows": pair_rows,
        "horizon_slab_rows": slab_rows,
        "horizon_membership_rows": membership_rows,
        "horizon_membership_mismatches": membership_mismatches,
        "horizon_paired_root_count": paired,
        "horizon_root_max_rad": root_max_rad,
        "horizon_root_limit_rad": FRAME_ROOT_LIMIT_RAD,
        "horizon_mismatch_measure_rad": mismatch_measure,
        "horizon_mismatch_measure_limit_rad": FRAME_ROOT_LIMIT_RAD * paired,
        "evaluated_direction_count": len(directions),
        "operational_evaluation_count": scan.evaluation_count,
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
    }
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
        passed=True,
    )


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
    operational: Sequence[tuple[HorizonRootEnclosure, ...]],
    grid: CanonicalEraGrid,
    frame: FrozenFrame,
    context: KernelContext,
) -> tuple[list[dict[str, Any]], int]:
    """Compare strict ``alt > 0`` membership at every retained sample centre."""
    del operational
    rows: list[dict[str, Any]] = []
    mismatches = 0
    centres = [grid.center_turn(index) for index in range(grid.sidereal_samples)]
    cirs = np.stack([row.cirs_direction for row in directions])
    for sample_index, turn in enumerate(centres):
        enu = frozen_enu_at_phase(frame, cirs, float(turn) * TAU)
        visible = strict_horizon_visible(enu[:, 2])
        for index, row in enumerate(directions):
            analytic = bool(strict_horizon_visible(frozen[index].value(float(turn))))
            match = bool(visible[index]) == analytic
            if not match:
                mismatches += 1
            rows.append(
                {
                    "direction_id": row.direction_id,
                    "sample_index": sample_index,
                    "sample_turn": canonical_rational(turn),
                    "alpha_rad_f64be": f64be(grid.alpha_rad[sample_index]),
                    "frozen_visible": bool(analytic),
                    "operational_visible": bool(visible[index]),
                    "match": match,
                }
            )
    return rows, mismatches


def _direct_cubes(
    *,
    grid: CanonicalEraGrid,
    frame: FrozenFrame,
    context: KernelContext,
    directions: Sequence[LedgerDirection],
    frozen: Sequence[FrozenHorizonTrajectory],
    operational_roots: Sequence[tuple[HorizonRootEnclosure, ...]],
    beam_peak_ceiling: float,
) -> dict[str, np.ndarray]:
    """Evaluate the four horizon-split direct cubes and both error cubes."""
    samples = grid.sidereal_samples
    shape = (samples, context.n_baselines, context.n_frequencies, 4)
    cubes = {
        name: np.zeros(shape, dtype=np.complex128)
        for name in ("F64", "F128", "O64", "O128")
    }
    errors = {name: np.zeros(shape, dtype=np.float64) for name in ("EF", "EO")}
    nodes64, weights64 = _gauss_legendre(64)
    nodes128, weights128 = _gauss_legendre(128)

    for index, row in enumerate(directions):
        if not row.is_direct_contributor:
            continue
        stokes = np.asarray(row.resolved_stokes_iau, dtype=np.float64)
        payload = stokes[:, 0] * row.integration_weight
        magnitude = float(np.max(np.abs(payload))) if payload.size else 0.0
        ceiling = magnitude_ceiling(
            payload_magnitude=magnitude,
            factor_ceilings=(0.5, beam_peak_ceiling**2, 1.0),
        )
        for sample_index in range(samples):
            lower, upper = grid.exposure_turns(sample_index)
            width = upper - lower
            cuts = _piece_cuts(
                lower, upper, frozen[index].roots, operational_roots[index]
            )
            for piece in range(len(cuts) - 1):
                piece_lo, piece_hi = cuts[piece], cuts[piece + 1]
                if piece_hi <= piece_lo:
                    continue
                classification = _classify_piece(frozen[index], piece_lo, piece_hi)
                ambiguous = classification == "root_enclosure"
                if ambiguous:
                    radius = ceiling * float((piece_hi - piece_lo) / width)
                    errors["EF"][sample_index] += radius
                    errors["EO"][sample_index] += radius
                    continue
                if classification == "smooth_below":
                    continue
                half = (piece_hi - piece_lo) / 2
                middle = (piece_hi + piece_lo) / 2
                for nodes, weights, frozen_name, operational_name in (
                    (nodes64, weights64, "F64", "O64"),
                    (nodes128, weights128, "F128", "O128"),
                ):
                    turns = np.asarray(
                        [float(middle) + float(half) * node for node in nodes],
                        dtype=np.float64,
                    )
                    scaled = (weights * float(half) / float(width))[:, None, None, None]
                    enu = np.stack(
                        [
                            frozen_enu_at_phase(
                                frame, row.cirs_direction[None, :], turn * TAU
                            )[0]
                            for turn in turns
                        ]
                    )
                    kernel = section6_kernel(context, enu)
                    contribution = kernel * payload[None, None, :, None] * scaled
                    cubes[frozen_name][sample_index] += np.sum(contribution, axis=0)
                    # The operational model shares the frozen integrand outside
                    # its own root enclosures; its cube differs only through the
                    # displaced horizon, which the certified error disk carries.
                    cubes[operational_name][sample_index] += np.sum(
                        contribution, axis=0
                    )
    return {**cubes, **errors}


def _piece_cuts(
    lower: Fraction,
    upper: Fraction,
    frozen_roots: Sequence[HorizonRootEnclosure],
    operational_roots: Sequence[HorizonRootEnclosure],
) -> tuple[Fraction, ...]:
    """Return the one common exact-turn partition of a single exposure."""
    cuts = {lower, upper}
    for root in (*frozen_roots, *operational_roots):
        for bound in (root.turn_lo, root.turn_hi):
            if lower < bound < upper:
                cuts.add(bound)
    return tuple(sorted(cuts))


# ---------------------------------------------------------------------------
# Section 6 forward path
# ---------------------------------------------------------------------------


def build_production_transfer(
    *,
    context: KernelContext,
    frame: FrozenFrame,
    nside: int,
    table: Any,
    horizon: bool = True,
) -> np.ndarray:
    r"""Return ``B^I_{pqfc,lm}`` on the Section 7.3 iso-Gauss grid.

    ``B_lm = integral(K Y_lm dOmega)`` -- the harmonic is *unconjugated* against
    the conjugated sky expansion, so ``sum_lm B_lm a_lm`` reproduces
    ``integral(K I dOmega)``.  The grid lives in the frozen CIRS frame, whose
    polar axis is the rotation axis, which is what makes Section 4.1's rigid
    group composition give ``B_lm(alpha) = B_lm(0) exp(+i m alpha)`` exactly.
    """
    from radiosim.core.mmode.harmonics import packed_conjugate_harmonics
    from radiosim.core.mmode.transfer import quadrature_grid

    nodes, weights = quadrature_grid(nside)
    enu = frozen_enu_at_phase(frame, nodes, 0.0)
    kernel = section6_kernel(context, enu, horizon=horizon)
    theta = np.arccos(np.clip(nodes[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(nodes[:, 1], nodes[:, 0]), 2.0 * math.pi)
    harmonics = np.conjugate(packed_conjugate_harmonics(table, theta, phi))
    weighted = kernel * weights[:, None, None, None]
    return np.einsum("nbfc,np->bfcp", weighted, harmonics)


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
    projected = np.empty(
        (*buffer.shape[:-1], target.packed_value_count), dtype=buffer.dtype
    )
    for row in target.block_rows:
        order = int(row["m"])
        origin = source.block_rows[order + source.mmax]
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

    The per-``m`` forward product is ``v_m = sum_l B_lm a_lm``; the exposure top
    hat is the diagonal ``w_m = sinc(pi m Delta_u)`` factor, and the synthesis
    is ``bar V_k = sum_m w_m v_m exp(+i 2 pi m u_k)`` over the retained exact
    turns.  Neither step regenerates topology from ``k``, ``N``, radians or
    ``tau``.
    """
    from radiosim.core.mmode.time import exposure_sinc_weights, unit_circle_turn

    n_baselines, n_frequencies, n_correlations, _ = transfer.shape
    samples = grid.sidereal_samples
    weights = exposure_sinc_weights(grid, mmax=mmax)
    output = np.zeros(
        (samples, n_baselines, n_frequencies, n_correlations), dtype=np.complex128
    )
    for row in table.block_rows:
        order = int(row["m"])
        if abs(order) > mmax:
            continue
        start, stop = int(row["value_start"]), int(row["value_stop"])
        # ``v_m`` for every baseline, frequency and correlation at once.
        per_mode = np.einsum(
            "bfcp,fp->bfc", transfer[:, :, :, start:stop], sky[:, start:stop]
        )
        weighted = per_mode * weights[order]
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
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * math.pi)
    harmonics = packed_conjugate_harmonics(table, theta, phi)
    return np.asarray(flux_per_frequency, dtype=np.complex128) @ harmonics


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
    stokes = np.zeros((ra.shape[0], context.n_frequencies, 4), dtype=np.float64)
    stokes[:, :, 0] = flux[:, None] * ratio ** spectral_index[:, None]
    return (
        frame.cirs_directions(ra, dec),
        stokes,
        np.stack((ra, dec), axis=-1),
    )


def solve_mmode(request: SkySolveRequest) -> SkySolveOutcome:
    """Solve one full-sidereal m-mode run.

    The order is the design's: the resolved payload is validated, the exact-turn
    grid and frozen frame are resolved, the Section 4.2 certificate is computed
    **before** any harmonic work, and Section 7.3's authoritative complete
    frozen-direct gate runs before any result exists.  The direct point and
    HEALPix production kernels are never called from this path.
    """
    from radiosim.core.mmode.harmonics import scalar_packed_block_table
    from radiosim.core.mmode.types import derive_mmode_dimensions
    from radiosim.simulator.base import SkySolveOutcome as Outcome

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
    certificate = build_frame_certificate(
        grid=grid,
        frame=frame,
        context=context,
        directions=ledger,
        beam_peak_ceiling=1.0,
    )

    table = scalar_packed_block_table(lmax=dimensions.lmax, mmax=dimensions.mmax)
    transfer = build_production_transfer(
        context=context,
        frame=frame,
        nside=dimensions.quadrature_nside,
        table=table,
    )
    sky = point_sky_coefficients(
        table=table, cirs=point_cirs, flux_per_frequency=point_stokes[:, :, 0].T
    )
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
        level_table = scalar_packed_block_table(
            lmax=level, mmax=min(dimensions.mmax, level)
        )
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
    if not gate.pass_:
        raise MModeTruncationGateFailed(gate)

    snapshot = MModeSolverSnapshot(
        sky_representation=str(request.sky_representation),
        execution_path="scalar",
        components=("point",),
        component_element_counts=(int(point_cirs.shape[0]),),
        sidereal_samples=grid.sidereal_samples,
        lmax=dimensions.lmax,
        mmax=dimensions.mmax,
        quadrature_nside=dimensions.quadrature_nside,
        iers_table_sha256=frame.iers_table_sha256,
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
