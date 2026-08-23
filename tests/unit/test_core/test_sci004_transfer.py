r"""SCI-004 phase-M1 red oracles for the baseline transfer function ``B_lm``.

``docs/development/sci004_mmode_design.md`` Section 6 defines the reference-phase
response from *the same* Jones and fringe factors as the direct RIME,

.. math::

    K^X_{pqfc}(\hat n)=
    [J_{p,\theta\phi}P^X_{\theta\phi}J^H_{q,\theta\phi}]_c
    K_{pq}(\hat n)M_{pq}(\hat n)Q_{pq}(\hat n)H(\hat n),

with the existing geometric phase ``K`` at its accepted sign, the accepted
baseline closure factor ``M``, ``Q``'s accepted bandwidth smearing, and the
strict horizon factor ``H = 1[alt > 0]``. Equality is excluded, matching both
maintained direct solvers: no epsilon, beam cutoff, or half-weight at the
horizon is allowed, and ``H`` is part of the transfer function rather than an
after-the-fact time mask.

Every SCI-004 cube therefore has ``C = 4`` in the exact resolved matrix order;
``n_correlations != 4``, an omitted cross-hand, or an evidence formula treating
``C`` as a free cardinality rejects before work.

The scalar transfer this phase binds is
``B^I_{pqfc,lm} = integral K^I_{pqfc} Y_lm dOmega`` -- note the *conjugate*
placement, which matches Shaw's definition that the transfer function itself is
expanded in conjugate harmonics -- and the rigid-rotation law that Section 4.1's
group composition delivers,

.. math:: B^X_{pqfc,lm}(\alpha)=B^X_{pqfc,lm}(0)e^{im\alpha}.

Section 12.2 family 5 is the required oracle set: analytic unit beam with a zero
baseline, one non-zero baseline fringe, heterogeneous beams, every correlation,
frequency scaling, and that rotation law. The analytic complex128 residual limit
is ``5e-12``.

The Section 13.3 owner is ``radiosim.core.mmode.transfer``, absent at ``G1``;
imports are function-local so each node yields its own Section 14.1 outcome.

**Phase M2 extension.** The second half of this module binds the *polarized*
transfer Section 6 defines beside the scalar one, on the same grid and from the
same kernel::

    B^I_{pqfc,lm}      = integral K^I Y_lm dOmega
    B^V_{pqfc,lm}      = integral K^V Y_lm dOmega
    B^{(+2)}_{pqfc,lm} = integral (K^Q - i K^U) _{+2}Y_lm dOmega
    B^{(-2)}_{pqfc,lm} = integral (K^Q + i K^U) _{-2}Y_lm dOmega

with the forward per-``m`` product carrying the two explicit one-half factors

    v = sum_l [ B^I a^I + (1/2) B^(+2) a^(+2) + (1/2) B^(-2) a^(-2) + B^V a^V ].

Those halves are not a normalization choice: substituting the delta-sky
coefficients and using ``_{-s}Y_lm = (-1)**(s+m) conj(_{s}Y_{l,-m})`` collapses
the two spin terms to exactly ``K^Q Q_H + K^U U_H``, which is what
:func:`test_the_forward_per_m_product_carries_the_one_half_spin_factors`
re-derives.  The Section 13.4 owners are ``radiosim.core.mmode.transfer`` and
``radiosim.core.mmode.solver``; neither polarized entry point exists at ``A1``.
The phase-M2 cases are declared separately in ``SCI004_PHASE2_RED_CASES`` so the
retained M1 record's node set stays exactly what it was.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

#: Section 12.2's analytic complex128 residual limit.
ANALYTIC_RESIDUAL_LIMIT = 5e-12

#: Section 6: every SCI-004 cube has exactly four correlations.
N_CORRELATIONS = 4

LMAX = 4
MMAX = 3
QUADRATURE_NSIDE = 16

SPEED_OF_LIGHT_M_PER_S = 299792458.0

_UNIT_BEAM_FIXTURE = f"""\
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
beam: unit
baseline_enu_m: [0.0, 0.0, 0.0]
frequency_hz: 150000000.0
""".encode()

_FRINGE_FIXTURE = f"""\
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
beam: unit
baseline_enu_m: [14.0, 0.0, 0.0]
frequency_hz: 150000000.0
""".encode()

_HETEROGENEOUS_FIXTURE = f"""\
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
beam: heterogeneous
antenna_diameters_m: [14.0, 6.0]
baseline_enu_m: [14.0, 0.0, 0.0]
frequency_hz: 150000000.0
""".encode()

_FREQUENCY_FIXTURE = f"""\
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
beam: unit
baseline_enu_m: [14.0, 0.0, 0.0]
frequencies_hz: [100000000.0, 200000000.0]
""".encode()

_ROTATION_FIXTURE = f"""\
lmax: {LMAX}
mmax: {MMAX}
quadrature_nside: {QUADRATURE_NSIDE}
beam: unit
baseline_enu_m: [14.0, 0.0, 0.0]
frequency_hz: 150000000.0
relative_phases_rad: [0.0, 0.9, -1.7]
""".encode()

_FRINGE_ORACLE = (
    "tests/unit/test_core/test_sci004_transfer.py::"
    "test_the_existing_geometric_phase_is_the_fringe_authority_today"
)


def _case(
    case_id: str,
    requirement_id: str,
    function: str,
    fixture: bytes,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": f"tests/unit/test_core/test_sci004_transfer.py::{function}",
        "expected_failure_kind": "import",
        "expected_failure_pattern": (
            r"ModuleNotFoundError: No module named 'radiosim\.core\.mmode'"
        ),
        "fixture_defect_excluded_by": _FRINGE_ORACLE,
        "fixture_bytes": fixture,
    }


SCI004_RED_CASES: tuple[dict[str, Any], ...] = (
    _case(
        "m1.transfer.unit-beam-zero-baseline",
        "sci004.section-6.analytic-unit-beam-zero-baseline",
        "test_unit_beam_and_zero_baseline_reproduce_the_analytic_blm",
        _UNIT_BEAM_FIXTURE,
    ),
    _case(
        "m1.transfer.single-baseline-fringe",
        "sci004.section-6.non-zero-baseline-fringe",
        "test_one_non_zero_baseline_reproduces_the_analytic_fringe",
        _FRINGE_FIXTURE,
    ),
    _case(
        "m1.transfer.heterogeneous-beams",
        "sci004.section-7.2.heterogeneous-per-antenna-response",
        "test_heterogeneous_beams_give_distinct_baseline_transfers",
        _HETEROGENEOUS_FIXTURE,
    ),
    _case(
        "m1.transfer.four-correlations",
        "sci004.section-6.exactly-four-correlations",
        "test_every_transfer_cube_carries_exactly_four_correlations",
        _UNIT_BEAM_FIXTURE,
    ),
    _case(
        "m1.transfer.frequency-scaling",
        "sci004.section-6.fringe-frequency-scaling",
        "test_the_transfer_scales_with_frequency_through_the_fringe",
        _FREQUENCY_FIXTURE,
    ),
    _case(
        "m1.transfer.rigid-rotation-law",
        "sci004.section-6.blm-alpha-equals-blm-zero-times-exp-i-m-alpha",
        "test_rigid_rotation_multiplies_the_transfer_by_exp_i_m_alpha",
        _ROTATION_FIXTURE,
    ),
)

SCI004_RED_GREEN_CONTROLS: tuple[str, ...] = (_FRINGE_ORACLE,)


# --- green control ------------------------------------------------------------


def test_the_existing_geometric_phase_is_the_fringe_authority_today() -> None:
    """Section 6: ``K`` is the *existing* geometric phase at its accepted sign.

    The m-mode transfer does not introduce a second fringe convention, so the
    function every red node below expects to be reused already exists and is
    exercised here. The four-correlation cardinality is equally pre-existing.
    """
    import numpy as np

    from radiosim.backends import get_backend
    from radiosim.core.jones import geometric_phase
    from radiosim.core.polarization_basis import (
        CORRELATION_LABELS,
        basis_for_correlations,
    )

    backend = get_backend("numpy")
    wavelength = SPEED_OF_LIGHT_M_PER_S / 150e6
    uvw = np.array([[14.0, 0.0, 0.0]], dtype=np.float64) / wavelength
    zenith = np.array([0.0], dtype=np.float64)
    phase = np.asarray(
        geometric_phase(
            uvw_wavelengths=uvw,
            dir_l=zenith,
            dir_m=zenith,
            dir_n=np.array([1.0], dtype=np.float64),
            backend=backend,
        )
    )

    assert np.all(np.isfinite(phase))
    assert np.iscomplexobj(phase)
    # At the phase centre the geometric factor is exactly unity.
    assert phase.shape == (1, 1)
    assert abs(complex(phase[0, 0]) - 1.0) <= ANALYTIC_RESIDUAL_LIMIT

    basis = basis_for_correlations(("XX", "XY", "YX", "YY"))
    assert basis == "linear_xy"
    assert CORRELATION_LABELS[basis] == ("XX", "XY", "YX", "YY")
    assert len(CORRELATION_LABELS[basis]) == N_CORRELATIONS

    # The strict horizon predicate is ``alt > 0``: equality is excluded, and no
    # epsilon or half-weight is permitted at the boundary.
    horizon = np.array([1.0, 0.0, -1e-18, 0.0], dtype=np.float64)
    assert list((horizon > 0.0).astype(np.int64)) == [1, 0, 0, 0]


# --- Section 6 / 12.2 family 5 red oracles ------------------------------------


def _transfer(**kwargs: Any) -> Any:
    from radiosim.core.mmode.transfer import build_scalar_baseline_transfer

    return build_scalar_baseline_transfer(**kwargs)


def test_unit_beam_and_zero_baseline_reproduce_the_analytic_blm() -> None:
    """Section 6: with a unit beam and no fringe, ``B^I`` is the hemisphere integral.

    ``H`` restricts the integral to the strictly visible hemisphere, so only the
    even-parity modes survive on a zenith-symmetric horizon; the ``m != 0``
    coefficients vanish identically. The parallel-hand response carries the
    Section 5.2/CLAUDE-normative ``1/2`` factor.
    """
    from radiosim.core.mmode.harmonics import scalar_coefficient

    transfer = _transfer(
        beam="unit",
        baseline_enu_m=(0.0, 0.0, 0.0),
        frequency_hz=150e6,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
    )
    parallel = transfer.correlation_index("XX")

    monopole = scalar_coefficient(transfer.blm[0, 0, parallel], 0, 0)
    assert abs(monopole - 0.5 * math.sqrt(math.pi)) <= ANALYTIC_RESIDUAL_LIMIT
    for order in range(1, MMAX + 1):
        for degree in range(order, LMAX + 1):
            assert (
                abs(scalar_coefficient(transfer.blm[0, 0, parallel], degree, order))
                <= ANALYTIC_RESIDUAL_LIMIT
            )


def test_one_non_zero_baseline_reproduces_the_analytic_fringe() -> None:
    """Section 6: the fringe is the existing ``K``, not a re-derived convention."""
    import numpy as np

    from radiosim.backends import get_backend
    from radiosim.core.jones import geometric_phase

    transfer = _transfer(
        beam="unit",
        baseline_enu_m=(14.0, 0.0, 0.0),
        frequency_hz=150e6,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
    )

    directions = np.asarray(transfer.quadrature_directions_enu, dtype=np.float64)
    wavelength = SPEED_OF_LIGHT_M_PER_S / 150e6
    expected = np.asarray(
        geometric_phase(
            uvw_wavelengths=(
                np.array([[14.0, 0.0, 0.0]], dtype=np.float64) / wavelength
            ),
            dir_l=directions[:, 0],
            dir_m=directions[:, 1],
            dir_n=directions[:, 2],
            backend=get_backend("numpy"),
        )
    )

    assert transfer.fringe_convention == "existing_geometric_phase_v1"
    residual = float(np.max(np.abs(np.asarray(transfer.fringe) - expected)))
    assert residual <= ANALYTIC_RESIDUAL_LIMIT
    # A non-zero baseline must actually change the transfer.
    zero = _transfer(
        beam="unit",
        baseline_enu_m=(0.0, 0.0, 0.0),
        frequency_hz=150e6,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
    )
    assert float(np.max(np.abs(np.asarray(transfer.blm) - np.asarray(zero.blm)))) > 0.0


def test_heterogeneous_beams_give_distinct_baseline_transfers() -> None:
    """Section 7.2: the normative ``B_lm`` comes from the full RIME kernel."""
    import numpy as np

    heterogeneous = _transfer(
        beam="heterogeneous",
        antenna_diameters_m=(14.0, 6.0),
        baseline_enu_m=(14.0, 0.0, 0.0),
        frequency_hz=150e6,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
    )
    homogeneous = _transfer(
        beam="heterogeneous",
        antenna_diameters_m=(14.0, 14.0),
        baseline_enu_m=(14.0, 0.0, 0.0),
        frequency_hz=150e6,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
    )

    difference = float(
        np.max(np.abs(np.asarray(heterogeneous.blm) - np.asarray(homogeneous.blm)))
    )
    assert difference > 0.0
    assert (
        heterogeneous.per_antenna_beam_identities[0]
        != (heterogeneous.per_antenna_beam_identities[1])
    )


def test_every_transfer_cube_carries_exactly_four_correlations() -> None:
    """Section 6: ``C = 4`` exactly, in the resolved row-major matrix order."""
    transfer = _transfer(
        beam="unit",
        baseline_enu_m=(0.0, 0.0, 0.0),
        frequency_hz=150e6,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
    )

    assert transfer.blm.shape[2] == N_CORRELATIONS
    assert transfer.correlation_labels == ("XX", "XY", "YX", "YY")
    assert len(transfer.correlation_labels) == N_CORRELATIONS
    # An omitted cross-hand is a rejection, not a supported subset.
    assert transfer.correlation_index("XY") != transfer.correlation_index("YX")


def test_the_transfer_scales_with_frequency_through_the_fringe() -> None:
    """Section 6: doubling the frequency doubles the fringe rate, exactly."""
    import numpy as np

    low = _transfer(
        beam="unit",
        baseline_enu_m=(14.0, 0.0, 0.0),
        frequency_hz=100e6,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
    )
    high = _transfer(
        beam="unit",
        baseline_enu_m=(14.0, 0.0, 0.0),
        frequency_hz=200e6,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
    )

    doubled = np.asarray(low.fringe) ** 2
    residual = float(np.max(np.abs(np.asarray(high.fringe) - doubled)))
    assert residual <= ANALYTIC_RESIDUAL_LIMIT
    assert float(np.max(np.abs(np.asarray(high.blm) - np.asarray(low.blm)))) > 0.0


def test_rigid_rotation_multiplies_the_transfer_by_exp_i_m_alpha() -> None:
    """Section 6: ``B_lm(alpha) = B_lm(0) exp(+i m alpha)`` from the rigid frame."""
    import numpy as np

    from radiosim.core.mmode.harmonics import scalar_coefficient

    transfer = _transfer(
        beam="unit",
        baseline_enu_m=(14.0, 0.0, 0.0),
        frequency_hz=150e6,
        lmax=LMAX,
        mmax=MMAX,
        quadrature_nside=QUADRATURE_NSIDE,
    )
    reference = transfer.blm[0, 0, 0]

    for alpha in (0.0, 0.9, -1.7):
        rotated = transfer.at_relative_phase(alpha).blm[0, 0, 0]
        for degree in range(LMAX + 1):
            for order in range(-min(degree, MMAX), min(degree, MMAX) + 1):
                expected = scalar_coefficient(reference, degree, order) * np.exp(
                    1j * order * alpha
                )
                observed = scalar_coefficient(rotated, degree, order)
                assert abs(observed - expected) <= ANALYTIC_RESIDUAL_LIMIT, (
                    alpha,
                    degree,
                    order,
                )


# =============================================================================
# Phase M2 -- the polarized transfer (Section 6, Section 12.2 family 5)
# =============================================================================

#: Phase M2 dimensions.  Section 7.3 requires ``lmax <= 2 * quadrature_nside``.
PHASE2_LMAX = 4
PHASE2_MMAX = 3
PHASE2_QUADRATURE_NSIDE = 8

_PHASE2_UNIT_FIXTURE = f"""\
lmax: {PHASE2_LMAX}
mmax: {PHASE2_MMAX}
quadrature_nside: {PHASE2_QUADRATURE_NSIDE}
beam: unit
baseline_enu_m: [0.0, 0.0, 0.0]
frequency_hz: 150000000.0
fields: ["I", "+2", "-2", "V"]
""".encode()

_PHASE2_FRINGE_FIXTURE = f"""\
lmax: {PHASE2_LMAX}
mmax: {PHASE2_MMAX}
quadrature_nside: {PHASE2_QUADRATURE_NSIDE}
beam: unit
baseline_enu_m: [14.0, 0.0, 0.0]
frequency_hz: 150000000.0
fields: ["I", "+2", "-2", "V"]
""".encode()

_PHASE2_ROTATION_FIXTURE = f"""\
lmax: {PHASE2_LMAX}
mmax: {PHASE2_MMAX}
quadrature_nside: {PHASE2_QUADRATURE_NSIDE}
beam: unit
baseline_enu_m: [14.0, 0.0, 0.0]
frequency_hz: 150000000.0
fields: ["I", "+2", "-2", "V"]
relative_phases_rad: [0.0, 0.9, -1.7]
""".encode()

_PHASE2_FORWARD_FIXTURE = f"""\
lmax: {PHASE2_LMAX}
mmax: {PHASE2_MMAX}
quadrature_nside: {PHASE2_QUADRATURE_NSIDE}
beam: unit
baseline_enu_m: [14.0, 0.0, 0.0]
frequency_hz: 150000000.0
forward_product: "sum_l(B_I a_I + 0.5 B_p2 a_p2 + 0.5 B_m2 a_m2 + B_V a_V)"
""".encode()

#: Section 5.3's science field order and spin labels.
FIELD_ORDER: tuple[str, ...] = ("I", "+2", "-2", "V")
SPIN_ORDER: tuple[int, ...] = (0, 2, -2, 0)

_PHASE2_ORACLE = (
    "tests/unit/test_core/test_sci004_transfer.py::"
    "test_the_scalar_transfer_and_iso_gauss_grid_hold_today"
)

_TRANSFER_IMPORT_PATTERN = (
    r"ImportError: cannot import name '\w+' from 'radiosim\.core\.mmode\.transfer'"
)
_SOLVER_IMPORT_PATTERN = (
    r"ImportError: cannot import name '\w+' from 'radiosim\.core\.mmode\.solver'"
)


def _phase2_case(
    case_id: str,
    requirement_id: str,
    function: str,
    pattern: str,
    fixture: bytes,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": f"tests/unit/test_core/test_sci004_transfer.py::{function}",
        "expected_failure_kind": "missing-symbol",
        "expected_failure_pattern": pattern,
        "fixture_defect_excluded_by": _PHASE2_ORACLE,
        "fixture_bytes": fixture,
    }


SCI004_PHASE2_RED_CASES: tuple[dict[str, Any], ...] = (
    _phase2_case(
        "m2.transfer.four-science-fields",
        "sci004.section-5.3.transfer-carries-all-four-fields",
        "test_the_polarized_transfer_carries_the_four_science_fields",
        _TRANSFER_IMPORT_PATTERN,
        _PHASE2_UNIT_FIXTURE,
    ),
    _phase2_case(
        "m2.transfer.conjugate-placement",
        "sci004.section-6.spin-transfer-conjugate-placement",
        "test_the_spin_transfer_matches_the_section_6_conjugate_placement",
        _TRANSFER_IMPORT_PATTERN,
        _PHASE2_FRINGE_FIXTURE,
    ),
    _phase2_case(
        "m2.transfer.scalar-field-agrees-with-m1",
        "sci004.section-6.polarized-i-field-equals-the-scalar-cube",
        "test_the_intensity_field_of_the_polarized_transfer_reproduces_the_m1_cube",
        _TRANSFER_IMPORT_PATTERN,
        _PHASE2_UNIT_FIXTURE,
    ),
    _phase2_case(
        "m2.transfer.rotation-law-every-field",
        "sci004.section-6.blm-alpha-law-holds-field-by-field",
        "test_the_polarized_transfer_obeys_the_rotation_law_for_every_field",
        _TRANSFER_IMPORT_PATTERN,
        _PHASE2_ROTATION_FIXTURE,
    ),
    _phase2_case(
        "m2.transfer.forward-one-half-factors",
        "sci004.section-6.forward-per-m-product-one-half-spin-factors",
        "test_the_forward_per_m_product_carries_the_one_half_spin_factors",
        _SOLVER_IMPORT_PATTERN,
        _PHASE2_FORWARD_FIXTURE,
    ),
)

SCI004_PHASE2_RED_GREEN_CONTROLS: tuple[str, ...] = (_PHASE2_ORACLE,)


def _phase2_spin_two(
    spin: int, degree: int, order: int, colatitude: float, longitude: float
) -> complex:
    """Return ``_{s}Y_{2m}`` for ``s = +-2`` from the published closed forms.

    Goldberg et al. (1967) for the ``+2`` table; the ``-2`` values follow from
    ``_{-s}Y_lm = (-1)**(s+m) conj(_{s}Y_{l,-m})``.  Writing them out is what
    makes this an independent oracle rather than a second call into the code
    under test.
    """
    if degree != 2:
        raise ValueError("only the degree-two closed forms are written out here")
    if spin == -2:
        return ((-1.0) ** (2 + order)) * complex(
            _phase2_spin_two(2, degree, -order, colatitude, longitude)
        ).conjugate()
    cosine = math.cos(colatitude)
    sine = math.sin(colatitude)
    phase = cmath.exp(1j * order * longitude)
    if order == 2:
        return math.sqrt(5.0 / (64.0 * math.pi)) * (1.0 + cosine) ** 2 * phase
    if order == 1:
        return math.sqrt(5.0 / (16.0 * math.pi)) * sine * (1.0 + cosine) * phase
    if order == 0:
        return complex(math.sqrt(15.0 / (32.0 * math.pi)) * sine * sine, 0.0)
    if order == -1:
        return math.sqrt(5.0 / (16.0 * math.pi)) * sine * (1.0 - cosine) * phase
    if order == -2:
        return math.sqrt(5.0 / (64.0 * math.pi)) * (1.0 - cosine) ** 2 * phase
    raise ValueError(f"m={order} is not a degree-two order")


def _phase2_unit_beam_responses(
    labels: tuple[str, ...],
) -> dict[str, dict[str, complex]]:
    """Return ``K^X_c / (fringe * H)`` for an east-X receptor and a unit beam.

    With ``J_NE = C`` (SCI-006's ``(North, East) -> (X=east, Y=north)``
    permutation) and Section 5.2's ``J_thetaphi = J_NE D``, the four Shaw-basis
    component matrices ``P^X`` transport to

    ``M P^I M^H = (1/2) I_2``
    ``M P^Q M^H = (1/2) diag(-1, 1)``
    ``M P^U M^H = (1/2) [[0, -1], [-1, 0]]``
    ``M P^V M^H = (1/2) [[0, -i], [i, 0]]``

    with ``M = C D = [[0, 1], [-1, 0]]``.  Every entry is written out here, so a
    sign defect in the production chain cannot hide behind a shared helper.
    """
    matrices = {
        "I": ((0.5, 0.0), (0.0, 0.5)),
        "Q": ((-0.5, 0.0), (0.0, 0.5)),
        "U": ((0.0, -0.5), (-0.5, 0.0)),
        "V": ((0.0, -0.5j), (0.5j, 0.0)),
    }
    rows = {"X": 0, "Y": 1}
    return {
        field: {
            label: complex(matrix[rows[label[0]]][rows[label[1]]]) for label in labels
        }
        for field, matrix in matrices.items()
    }


# --- phase-M2 green control ---------------------------------------------------


def test_the_scalar_transfer_and_iso_gauss_grid_hold_today() -> None:
    """The M1 transfer and its iso-Gauss grid are sound at ``A1``.

    Section 7.3's grid has exactly ``12 * nside**2`` nodes, its weights sum to
    the whole sphere, and because ``3 * nside`` is even the strictly visible
    hemisphere carries exactly half the total weight.  The scalar monopole of a
    unit beam over that hemisphere is ``(1/2) * sqrt(pi)``.  All of that holds
    now, so a phase-M2 red failure below is the absence of the polarized entry
    points rather than a defective grid, horizon, or fringe.
    """
    import numpy as np

    from radiosim.core.mmode.harmonics import scalar_coefficient
    from radiosim.core.mmode.transfer import (
        quadrature_grid,
    )

    directions, weights = quadrature_grid(PHASE2_QUADRATURE_NSIDE)
    assert directions.shape == (12 * PHASE2_QUADRATURE_NSIDE**2, 3)
    assert abs(float(np.sum(weights)) - 4.0 * math.pi) <= ANALYTIC_RESIDUAL_LIMIT
    visible = directions[:, 2] > 0.0
    assert abs(float(np.sum(weights[visible])) - 2.0 * math.pi) <= (
        ANALYTIC_RESIDUAL_LIMIT
    )
    # Equality at the horizon is excluded, and no node sits on the equator.
    assert not bool(np.any(directions[:, 2] == 0.0))

    transfer = _transfer(
        beam="unit",
        baseline_enu_m=(0.0, 0.0, 0.0),
        frequency_hz=150e6,
        lmax=PHASE2_LMAX,
        mmax=PHASE2_MMAX,
        quadrature_nside=PHASE2_QUADRATURE_NSIDE,
    )
    monopole = scalar_coefficient(
        transfer.blm[0, 0, transfer.correlation_index("XX")], 0, 0
    )
    assert abs(monopole - 0.5 * math.sqrt(math.pi)) <= ANALYTIC_RESIDUAL_LIMIT
    assert transfer.correlation_labels == ("XX", "XY", "YX", "YY")


# --- Section 6 phase-M2 red oracles -------------------------------------------


def _polarized(**kwargs: Any) -> Any:
    from radiosim.core.mmode.transfer import build_polarized_baseline_transfer

    return build_polarized_baseline_transfer(**kwargs)


def test_the_polarized_transfer_carries_the_four_science_fields() -> None:
    """Section 5.3/6: one packed table with the exact field and spin order."""
    transfer = _polarized(
        beam="unit",
        baseline_enu_m=(0.0, 0.0, 0.0),
        frequency_hz=150e6,
        lmax=PHASE2_LMAX,
        mmax=PHASE2_MMAX,
        quadrature_nside=PHASE2_QUADRATURE_NSIDE,
    )

    assert transfer.table.field_order == FIELD_ORDER
    assert transfer.table.spin_order == SPIN_ORDER
    assert transfer.blm.values.shape == (
        1,
        1,
        N_CORRELATIONS,
        transfer.table.packed_value_count,
    )
    assert transfer.correlation_labels == ("XX", "XY", "YX", "YY")
    assert transfer.fringe_convention == "existing_geometric_phase_v1"


def test_the_spin_transfer_matches_the_section_6_conjugate_placement() -> None:
    """Section 6: ``B^(+2) = int (K^Q - i K^U) _{+2}Y``, ``B^(-2)`` its partner.

    The reference integral is summed here on the production grid, from the
    written-out unit-beam response matrices and the published spin closed forms.
    Section 6 says the placement "is pinned by an explicit numerical integral;
    changing a library API call until one test passes is not an alternative
    convention", which is exactly what this node does.
    """
    import numpy as np

    transfer = _polarized(
        beam="unit",
        baseline_enu_m=(14.0, 0.0, 0.0),
        frequency_hz=150e6,
        lmax=PHASE2_LMAX,
        mmax=PHASE2_MMAX,
        quadrature_nside=PHASE2_QUADRATURE_NSIDE,
    )

    directions = np.asarray(transfer.quadrature_directions_enu, dtype=np.float64)
    weights = np.asarray(transfer.quadrature_weights, dtype=np.float64)
    horizon = np.asarray(transfer.horizon_mask).astype(np.float64)
    fringe = np.asarray(transfer.fringe, dtype=np.complex128).reshape(-1)
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * math.pi)

    responses = _phase2_unit_beam_responses(transfer.correlation_labels)
    degree = 2
    for label in ("XX", "XY"):
        index = transfer.correlation_index(label)
        linear_plus = responses["Q"][label] - 1j * responses["U"][label]
        linear_minus = responses["Q"][label] + 1j * responses["U"][label]
        kernel = weights * horizon * fringe
        for order in range(-min(degree, PHASE2_MMAX), min(degree, PHASE2_MMAX) + 1):
            plus_basis = np.asarray(
                [
                    _phase2_spin_two(2, degree, order, float(t), float(p))
                    for t, p in zip(theta, phi, strict=True)
                ],
                dtype=np.complex128,
            )
            minus_basis = np.asarray(
                [
                    _phase2_spin_two(-2, degree, order, float(t), float(p))
                    for t, p in zip(theta, phi, strict=True)
                ],
                dtype=np.complex128,
            )
            expected_plus = complex(np.sum(kernel * linear_plus * plus_basis))
            expected_minus = complex(np.sum(kernel * linear_minus * minus_basis))

            observed_plus = transfer.coefficient(0, 0, index, "+2", degree, order)
            observed_minus = transfer.coefficient(0, 0, index, "-2", degree, order)
            assert abs(observed_plus - expected_plus) <= ANALYTIC_RESIDUAL_LIMIT, (
                label,
                order,
            )
            assert abs(observed_minus - expected_minus) <= ANALYTIC_RESIDUAL_LIMIT, (
                label,
                order,
            )


def test_the_intensity_field_of_the_polarized_transfer_reproduces_the_m1_cube() -> None:
    """Section 6: the polarized build changes no accepted scalar number.

    The ``I`` field of the polarized transfer is the same integral the accepted
    M1 scalar cube already carries, on the same grid, from the same kernel; a
    polarized implementation that perturbs it has changed the accepted M1
    physics rather than extended it.
    """
    import numpy as np

    from radiosim.core.mmode.harmonics import scalar_coefficient

    arguments = {
        "beam": "unit",
        "baseline_enu_m": (0.0, 0.0, 0.0),
        "frequency_hz": 150e6,
        "lmax": PHASE2_LMAX,
        "mmax": PHASE2_MMAX,
        "quadrature_nside": PHASE2_QUADRATURE_NSIDE,
    }
    polarized = _polarized(**arguments)
    scalar = _transfer(**arguments)

    for label in ("XX", "XY", "YX", "YY"):
        index = polarized.correlation_index(label)
        scalar_index = scalar.correlation_index(label)
        for degree in range(PHASE2_LMAX + 1):
            for order in range(-min(degree, PHASE2_MMAX), min(degree, PHASE2_MMAX) + 1):
                expected = scalar_coefficient(
                    scalar.blm[0, 0, scalar_index], degree, order
                )
                observed = polarized.coefficient(0, 0, index, "I", degree, order)
                assert abs(observed - expected) <= ANALYTIC_RESIDUAL_LIMIT, (
                    label,
                    degree,
                    order,
                )
    assert np.isfinite(np.asarray(polarized.blm.values)).all()


def test_the_polarized_transfer_obeys_the_rotation_law_for_every_field() -> None:
    """Section 6: ``B^X_lm(alpha) = B^X_lm(0) exp(+i m alpha)`` for all four ``X``."""
    import numpy as np

    transfer = _polarized(
        beam="unit",
        baseline_enu_m=(14.0, 0.0, 0.0),
        frequency_hz=150e6,
        lmax=PHASE2_LMAX,
        mmax=PHASE2_MMAX,
        quadrature_nside=PHASE2_QUADRATURE_NSIDE,
    )

    for alpha in (0.0, 0.9, -1.7):
        rotated = transfer.at_relative_phase(alpha)
        for field, spin in zip(FIELD_ORDER, SPIN_ORDER, strict=True):
            for degree in range(abs(spin), PHASE2_LMAX + 1):
                for order in range(
                    -min(degree, PHASE2_MMAX), min(degree, PHASE2_MMAX) + 1
                ):
                    reference = transfer.coefficient(0, 0, 0, field, degree, order)
                    expected = reference * np.exp(1j * order * alpha)
                    observed = rotated.coefficient(0, 0, 0, field, degree, order)
                    assert abs(observed - expected) <= ANALYTIC_RESIDUAL_LIMIT, (
                        alpha,
                        field,
                        degree,
                        order,
                    )


def test_the_forward_per_m_product_carries_the_one_half_spin_factors() -> None:
    """Section 6: the two explicit one-half factors on the spin terms.

    They are a theorem, not a knob.  Substituting a delta sky's coefficients
    ``a^(+2) = (Q_H + i U_H) conj(_{+2}Y(n_s))`` and its partner into the
    displayed product, and using
    ``sum_lm _{s}Y_lm(n) conj(_{s}Y_lm(n_s)) -> delta(n - n_s)``, collapses the
    two halves to ``K^Q Q_H + K^U U_H``.  Dropping either factor doubles that
    contribution, so the halves are asserted here against a product assembled
    term by term in the test body.
    """
    import numpy as np

    from radiosim.core.mmode.solver import forward_per_m_product

    rng = np.random.default_rng(20260823)
    transfer_values = rng.normal(size=(4, 7)) + 1j * rng.normal(size=(4, 7))
    sky_values = rng.normal(size=(4, 7)) + 1j * rng.normal(size=(4, 7))
    weights = {"I": 1.0, "+2": 0.5, "-2": 0.5, "V": 1.0}

    expected = complex(
        sum(
            weights[field] * complex(np.sum(transfer_values[index] * sky_values[index]))
            for index, field in enumerate(FIELD_ORDER)
        )
    )
    observed = forward_per_m_product(
        transfer_block=transfer_values,
        sky_block=sky_values,
        field_order=FIELD_ORDER,
    )
    assert abs(complex(observed) - expected) <= ANALYTIC_RESIDUAL_LIMIT

    # Non-vacuity: unit weights on the spin terms are a different equation.
    unweighted = complex(
        sum(
            complex(np.sum(transfer_values[index] * sky_values[index]))
            for index in range(len(FIELD_ORDER))
        )
    )
    assert abs(unweighted - expected) > 10.0 * ANALYTIC_RESIDUAL_LIMIT
