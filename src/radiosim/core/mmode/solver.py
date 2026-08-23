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
    TAU,
    array_digest,
    canonical_json,
    canonical_rational,
    decode_f64be,
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


def build_input_identity(
    *,
    request: Any,
    grid: CanonicalEraGrid,
    frame: FrozenFrame,
    context: KernelContext,
    dimensions: Any,
    directions: Sequence[LedgerDirection],
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
        request, directions, context.frequencies_hz
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


def _sky_component_rows(
    request: Any,
    directions: Sequence[LedgerDirection],
    frequencies_hz: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return Section 14.3's sky-component and direction-input row arrays."""
    frequencies = [float(value) for value in frequencies_hz]
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
                "polarization_frame": ("not_applicable_no_linear_polarization"),
                "polarization_frame_sha256": object_digest(
                    "radiosim.sky-tangent-polarization.v1",
                    "not_applicable_no_linear_polarization",
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
    """
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
        payload = stokes[:, 0] * row.integration_weight
        magnitude = float(np.max(np.abs(payload))) if payload.size else 0.0
        ceiling = magnitude_ceiling(
            payload_magnitude=magnitude,
            factor_ceilings=(0.5, beam_peak_ceiling**2, 1.0),
        )
        active = tuple(bool(value != 0.0) for value in payload)
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
                        kernel = section6_kernel(context, enu)
                        integrands[order] = kernel * payload[None, None, :, None]
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

    Phase M1 evaluates the scalar ``I`` field only; the three remaining
    fixed-order fields contribute exact complex zero and are recorded as such
    rather than omitted, because Section 7.3's row set is the complete
    ``(1+len(Q_diag))*B*F*C*4`` product.
    """
    from radiosim.core.mmode.harmonics import packed_conjugate_harmonics
    from radiosim.core.mmode.transfer import quadrature_grid

    rows: list[dict[str, Any]] = []
    for entry in catalog:
        nside = int(entry["transfer_nside"])
        table = (
            production_table
            if entry["transfer_role"] == "production"
            else diagnostic_table
        )
        nodes, weights = quadrature_grid(nside)
        enu = frozen_enu_at_phase(frame, nodes, 0.0)
        kernel = section6_kernel(context, enu)
        theta = np.arccos(np.clip(nodes[:, 2], -1.0, 1.0))
        phi = np.mod(np.arctan2(nodes[:, 1], nodes[:, 0]), 2.0 * math.pi)
        harmonics = np.conjugate(packed_conjugate_harmonics(table, theta, phi))
        weighted = kernel * weights[:, None, None, None]
        directions = int(nodes.shape[0])
        empty = np.zeros((directions, table.packed_value_count), dtype=np.complex128)
        for baseline in range(context.n_baselines):
            for frequency in range(context.n_frequencies):
                for correlation in range(4):
                    for field_index, field_name in enumerate(FIELD_ORDER):
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
    from radiosim.core.mmode.harmonics import scalar_packed_block_table

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

    catalog, catalog_sha256 = _transfer_catalog(directions)
    check_table = scalar_packed_block_table(
        lmax=dimensions.lcheck, mmax=dimensions.mcheck
    )
    check_transfer = build_production_transfer(
        context=context, frame=frame, nside=dimensions.qcheck, table=check_table
    )
    check_sky = point_sky_coefficients(
        table=check_table,
        cirs=solved["point_cirs"],
        flux_per_frequency=solved["point_stokes"][:, :, 0].T,
    )

    def synthesize(lmax: int, mmax: int) -> np.ndarray:
        level = scalar_packed_block_table(lmax=lmax, mmax=mmax)
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

    Phase M1 evaluates the scalar ``I`` field only, so it is the one field
    recorded, in the fixed field order's first position.
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
        baseline: int,
        frequency: int,
        correlation: int,
    ) -> complex:
        """Return one operand's ``sum_l B_lm a_lm`` for a signed-``m`` block."""
        table, transfer, sky = sources[source]
        if abs(order) > mmax or abs(order) > table.mmax:
            return 0j
        block = table.block_rows[order + table.mmax]
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
                                weight=weight,
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

    Phase M1 evaluates the scalar ``I`` field, so the three remaining
    fixed-order fields contribute the exact zero time vector.  They are recorded
    rather than omitted, because the ledger's row count is the complete
    ``B*F*C*4*(2*mcheck+1)`` product and a missing field would read as coverage
    that was never attempted.
    """
    row: dict[str, Any] = {
        "baseline_index": baseline,
        "frequency_index": frequency,
        "correlation_index": correlation,
        "field": field,
        "signed_m": order,
        "diagnostic_ids": list(DIAGNOSTIC_IDS),
    }
    scalar = field == FIELD_ORDER[0]
    for name in DIAGNOSTIC_IDS:
        left, right = joins[name]
        delta = (
            block_value(
                left[0],
                left[1],
                left[2],
                order,
                baseline,
                frequency,
                correlation,
            )
            - block_value(
                right[0],
                right[1],
                right[2],
                order,
                baseline,
                frequency,
                correlation,
            )
            if scalar
            else 0j
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
    from radiosim.core.mmode.harmonics import scalar_packed_block_table
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
    input_manifest, input_identity_sha256 = build_input_identity(
        request=request,
        grid=grid,
        frame=frame,
        context=context,
        dimensions=dimensions,
        directions=ledger,
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
        "point_stokes": point_stokes,
        "deficit_max_quarter_jy": level_deficits[0],
        "deficit_max_half_jy": level_deficits[1],
        "quarter_level": quarter_level,
        "half_level": half_level,
    }


def solve_mmode(request: SkySolveRequest) -> SkySolveOutcome:
    """Solve one full-sidereal m-mode run and return the strategy outcome.

    Section 7.3's two-tier gate is authoritative and runs before any result or
    output path is created, so a failing gate raises instead of returning a
    cube.
    """
    from radiosim.simulator.base import SkySolveOutcome as Outcome

    solved = _mmode_pipeline(request)
    gate = solved["gate"]
    if not gate.pass_:
        raise MModeTruncationGateFailed(gate)

    grid = solved["grid"]
    dimensions = solved["dimensions"]
    certificate = solved["certificate"]
    cube = solved["cube"]
    point_cirs = solved["point_cirs"]
    snapshot = MModeSolverSnapshot(
        sky_representation=str(request.sky_representation),
        execution_path="scalar",
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
