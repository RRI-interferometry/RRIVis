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
"""

from __future__ import annotations

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
