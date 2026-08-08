"""Tier 7E: the ``D`` term's physics, flags, and effect on the visibilities.

``Tier7JonesSciencePlan.md`` Section 20.3.  As for ``G`` and ``B``, every
reference value is the published closed form written out in the test body, never
a value read back from the production function (Section 29.1).

Invariants asserted here: **I2** (declared flags are true, and each declared
``False`` has a witness), **I3** (a DIE term returns ``(1, 2, 2)``), **I7** (a
configured term changes the visibilities), and Section 20.3's own three
statements -- ``D(0) = I2``, ``det D = 1 + d_p0 d_p1^*``, and the corrected
cross-hand prediction ``V_01 = (I/2)(d_p0 - d_q1)`` for an unpolarized source.

The cross-hand prediction is written in Section 20.3 as a first-order
expansion.  For the convention adopted there it is in fact **exact**: the
``[0, 1]`` element of ``D_p D_q^H`` is exactly ``d_p0 - d_q1``, with no
second-order remainder, because the two contributing products are ``1 * d_p0``
and ``d_p0's`` counterpart ``-d_q1 * 1``.  The test therefore asserts it at
machine precision rather than at a first-order tolerance -- a weaker assertion
would pass for a term that had the quadratic terms wrong.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.core.jones.polarization_leakage import (
    LeakageCoefficient,
    PolarizationLeakageJones,
)
from radiosim.core.visibility import calculate_visibility
from tests.characterization.test_tier6_current_behavior import (
    WORKLOAD_LOCATION,
    WORKLOAD_TIME_GRID,
    _workload_point_sources,
)
from tests.unit.test_core.test_jones_resolution import (
    resolve_for,
    solver_components_with_jones,
)

_BACKEND = get_backend("numpy")


def _empty_directions():
    from radiosim.core.jones.directions import DirectionBatch

    empty = np.zeros(3, dtype=np.float64)
    return DirectionBatch(
        alt_rad=empty,
        az_rad=empty,
        dir_l=empty,
        dir_m=empty,
        dir_n=empty,
        ra_rad=empty,
        dec_rad=empty,
        hour_angle_rad=empty,
        n_dir=3,
    )


def _constant(value: complex) -> LeakageCoefficient:
    """One feed's frequency-flat leakage ``d``."""
    return LeakageCoefficient(coefficients=(complex(value),))


def _term(
    d0: complex = 0.0,
    d1: complex = 0.0,
    *,
    rows: int = 2,
) -> PolarizationLeakageJones:
    """Build one ``D`` with the same two leakages on every antenna row."""
    return PolarizationLeakageJones(
        d_terms=tuple((_constant(d0), _constant(d1)) for _ in range(rows))
    )


def _evaluate(
    term: PolarizationLeakageJones,
    *,
    antenna_idx: int = 0,
    frequency_hz: float = 1.0e8,
):
    return np.asarray(
        term.compute_jones_batch(
            antenna_idx=antenna_idx,
            directions=_empty_directions(),
            frequency_hz=frequency_hz,
            freq_idx=0,
            time_mjd=60000.0,
            time_idx=0,
            backend=_BACKEND,
            dtype=np.complex128,
        )
    )


# ---------------------------------------------------------------------------
# The closed form (Section 20.3), evaluated independently
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("d0", "d1"),
    [
        (0.02 + 0.0j, 0.0 + 0.02j),
        (0.05 - 0.01j, -0.03 + 0.04j),
        (0.1 + 0.1j, 0.1 + 0.1j),
        (0.0 + 0.0j, 0.07 + 0.0j),
    ],
)
def test_the_matrix_is_the_published_first_order_form(
    d0: complex,
    d1: complex,
) -> None:
    """``D = [[1, d_0], [-conj(d_1), 1]]`` at machine precision.

    The conjugate-and-negate on the lower left is the Hamaker, Bregman & Sault
    convention Section 20.3 adopts, and it is the whole reason ``D`` reduces to a
    rotation for real, equal leakages.  Written out here from the published form;
    nothing in this assertion calls the term's own arithmetic.
    """
    block = _evaluate(_term(d0, d1))

    assert block.shape == (1, 2, 2)
    assert block[0, 0, 0] == 1.0 + 0.0j
    assert block[0, 1, 1] == 1.0 + 0.0j
    assert block[0, 0, 1] == pytest.approx(d0, rel=0.0, abs=1e-16)
    assert block[0, 1, 0] == pytest.approx(-complex(d1).conjugate(), rel=0.0, abs=1e-16)


def test_real_equal_leakages_reduce_to_a_scaled_rotation() -> None:
    """Section 20.3's stated reason for the conjugate-and-negate convention.

    For real ``d_0 = d_1 = d`` the matrix is ``[[1, d], [-d, 1]]``, which is
    ``sqrt(1 + d^2)`` times the rotation by ``arctan(d)``.  If the lower-left
    sign or conjugate were wrong this identity would fail, so it is the sharpest
    available check on the convention itself.
    """
    d = 0.08
    matrix = _evaluate(_term(d, d))[0]

    scale = math.hypot(1.0, d)
    angle = math.atan2(d, 1.0)
    rotation = np.array(
        [
            [math.cos(angle), math.sin(angle)],
            [-math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(matrix, scale * rotation, rtol=0.0, atol=1e-15)


def test_zero_leakage_is_exactly_the_identity() -> None:
    """``D(0) = I2``, exactly -- the statement R7 turns into a rejection."""
    block = _evaluate(_term(0.0, 0.0))

    np.testing.assert_array_equal(block[0], np.eye(2, dtype=np.complex128))


@pytest.mark.parametrize(
    ("d0", "d1"),
    [(0.02 + 0.0j, 0.0 + 0.02j), (0.3 - 0.2j, 0.1 + 0.4j), (0.05, 0.05)],
)
def test_the_determinant_is_one_plus_d0_times_conj_d1(
    d0: complex,
    d1: complex,
) -> None:
    """``det D = 1 + d_p0 d_p1^*``, so ``D`` is invertible for physical leakages.

    Section 20.3 states the determinant explicitly; it is what makes the term
    invertible (and therefore a calibratable corruption rather than a
    destruction of information) for every ``|d| < 1``.
    """
    matrix = _evaluate(_term(d0, d1))[0]

    expected = 1.0 + complex(d0) * complex(d1).conjugate()
    # Written out as ``ad - bc`` rather than handed to ``np.linalg.det``: the
    # 2x2 determinant is one line, and routing it through LAPACK would make the
    # assertion depend on a factorization rather than on the four numbers.
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

    assert determinant == pytest.approx(expected, rel=0.0, abs=1e-14)
    assert abs(expected) > 0.0


def test_a_die_term_returns_one_broadcast_matrix() -> None:
    """I3: ``(1, 2, 2)``, never ``n_dir`` copies of one constant."""
    term = _term(0.02, 0.03)

    assert term.is_direction_dependent is False
    assert _evaluate(term).shape == (1, 2, 2)


# ---------------------------------------------------------------------------
# The three parameterizations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ixr_db", [10.0, 20.0, 30.0, 40.0])
def test_the_ixr_parameterization_matches_carozzi_and_woan(
    tmp_path,
    ixr_db: float,
) -> None:
    """``|d| = 1 / sqrt(IXR_lin)``, with ``IXR_lin = 10^(IXR_dB/10)``.

    Carozzi & Woan (2011) define ``IXR_J = ((kappa + 1)/(kappa - 1))^2`` for the
    condition number ``kappa`` of the Jones matrix.  For
    ``D = [[1, d], [-d^*, 1]]`` the singular values are ``1 +- |d|``, so
    ``kappa = (1 + |d|)/(1 - |d|)`` and the two relations compose to
    ``|d| = 1/sqrt(IXR_lin)`` exactly -- the familiar
    ``IXR_dB = -20 log10 |d|``.

    The number is checked in both directions, because the whole risk in a dB
    conversion is getting it upside down: a *larger* IXR must mean a *smaller*
    leakage, and 30 dB must be about 3%, not about 94%.
    """
    resolved = resolve_for(
        tmp_path, {"D": {"d_terms": {"kind": "ixr", "ixr_db": ixr_db}}}
    )
    term = resolved.term("D")
    assert term is not None

    expected = 10.0 ** (-ixr_db / 20.0)
    values = term.d_terms_at_frequency(1.0e8)

    assert abs(values[0, 0]) == pytest.approx(expected, rel=1e-14)
    assert abs(values[0, 1]) == pytest.approx(expected, rel=1e-14)
    assert 10.0 * math.log10(1.0 / abs(values[0, 0]) ** 2) == pytest.approx(
        ixr_db, rel=1e-12
    )


def test_a_larger_ixr_is_a_smaller_leakage(tmp_path) -> None:
    """The monotonicity that a swapped numerator and denominator would break."""
    weak = resolve_for(tmp_path, {"D": {"d_terms": {"kind": "ixr", "ixr_db": 10.0}}})
    strong = resolve_for(tmp_path, {"D": {"d_terms": {"kind": "ixr", "ixr_db": 40.0}}})

    weak_term = weak.term("D")
    strong_term = strong.term("D")
    assert weak_term is not None and strong_term is not None

    assert abs(strong_term.d_terms_at_frequency(1.0e8)[0, 0]) < abs(
        weak_term.d_terms_at_frequency(1.0e8)[0, 0]
    )
    assert abs(weak_term.d_terms_at_frequency(1.0e8)[0, 0]) < 1.0


def test_the_ixr_phase_is_carried_onto_both_feeds(tmp_path) -> None:
    """``d = |d| exp(i phi)``: the modulus comes from IXR, the phase is given."""
    phase = 0.6
    resolved = resolve_for(
        tmp_path,
        {"D": {"d_terms": {"kind": "ixr", "ixr_db": 26.0, "phase_rad": phase}}},
    )
    term = resolved.term("D")
    assert term is not None

    expected = 10.0 ** (-26.0 / 20.0) * cmath.exp(1j * phase)

    assert term.d_terms_at_frequency(1.0e8)[0, 0] == pytest.approx(expected, rel=1e-14)


def test_the_frequency_polynomial_matches_its_closed_form() -> None:
    """``d(nu) = sum_k c_k x^k`` with ``x = (nu - nu_ref)/nu_scale``."""
    coefficients = (0.02 + 0.0j, 0.01 - 0.005j, -0.004 + 0.0j)
    reference = 1.0e8
    scale = 5.0e6
    coefficient = LeakageCoefficient(
        coefficients=coefficients,
        reference_frequency_hz=reference,
        scale_frequency_hz=scale,
    )
    term = PolarizationLeakageJones(d_terms=((coefficient, _constant(0.0)),))

    for frequency in (0.95e8, 1.0e8, 1.06e8):
        x = (frequency - reference) / scale
        expected = coefficients[0] + coefficients[1] * x + coefficients[2] * x * x
        block = _evaluate(term, frequency_hz=frequency)
        assert block[0, 0, 1] == pytest.approx(expected, rel=1e-14)


def test_a_constant_leakage_is_frequency_flat() -> None:
    """The explicit and IXR kinds carry no frequency structure at all."""
    term = _term(0.03 + 0.01j, 0.02)

    np.testing.assert_array_equal(
        _evaluate(term, frequency_hz=1.0e8), _evaluate(term, frequency_hz=3.0e8)
    )
    assert term.is_frequency_dependent is False


# ---------------------------------------------------------------------------
# I2 -- declared flags are true, and each declared False has a witness
# ---------------------------------------------------------------------------


def test_a_leaking_term_declares_neither_diagonality_nor_scalarity() -> None:
    """I2's converse direction: each declared ``False`` has a numeric witness."""
    term = _term(0.04 + 0.01j, 0.02 - 0.03j)

    assert term.is_diagonal() is False
    assert term.is_scalar() is False
    block = _evaluate(term)
    assert block[0, 0, 1] != 0.0
    assert block[0, 1, 0] != 0.0


def test_leakage_is_never_unitary_for_a_non_zero_d() -> None:
    """Section 20.3: ``D`` is non-unitary for any non-zero leakage.

    A leaking receptor moves power between the two feed chains; a matrix that
    did that while preserving ``J J^H = I`` would be a rotation, not a leakage.
    """
    for d0, d1 in ((0.02, 0.0), (0.0, 0.02), (0.05 + 0.05j, -0.01j)):
        term = _term(d0, d1)
        assert term.is_unitary() is False
        matrix = _evaluate(term)[0]
        assert not np.allclose(
            matrix @ matrix.conj().T, np.eye(2), rtol=0.0, atol=1e-12
        )


def test_a_zero_leakage_term_declares_the_identity_it_is() -> None:
    """The other half of I2: the flags follow the numbers, not the class.

    A zero-leakage ``D`` is exactly ``I2``, so it *is* diagonal, scalar and
    unitary, and saying otherwise would be a false ``False``.  Such a term is
    unreachable from configuration -- R7 rejects it -- which is precisely why the
    flags are computed from the resolved numbers rather than hard-coded.
    """
    term = _term(0.0, 0.0)

    assert term.is_diagonal() is True
    assert term.is_scalar() is True
    assert term.is_unitary() is True
    assert term.is_identity() is True


def test_d_declares_the_dependencies_it_actually_has() -> None:
    """Direction- and time-independent; frequency-dependent only when it varies."""
    flat = _term(0.03, 0.02)
    assert flat.is_direction_dependent is False
    assert flat.is_time_dependent is False
    assert flat.is_frequency_dependent is False

    varying = PolarizationLeakageJones(
        d_terms=(
            (
                LeakageCoefficient(
                    coefficients=(0.02 + 0j, 0.01 + 0j),
                    reference_frequency_hz=1.0e8,
                    scale_frequency_hz=1.0e7,
                ),
                _constant(0.0),
            ),
        )
    )
    assert varying.is_frequency_dependent is True
    assert not np.array_equal(
        _evaluate(varying, frequency_hz=1.0e8),
        _evaluate(varying, frequency_hz=1.1e8),
    )


def test_the_term_status_is_implemented() -> None:
    """Section 31 step 5, and the ``"implemented"`` half of invariant I20."""
    assert _term(0.02).term_status == "implemented"
    assert _term(0.02).name == "D"


# ---------------------------------------------------------------------------
# Construction and evaluation guards
# ---------------------------------------------------------------------------


def test_a_non_finite_leakage_cannot_be_constructed() -> None:
    """Caught at construction, so no ``nan`` ever reaches a chain."""
    with pytest.raises(ValueError):
        PolarizationLeakageJones(
            d_terms=((_constant(complex(np.nan, 0.0)), _constant(0.0)),)
        )


def test_an_empty_leakage_inventory_is_rejected() -> None:
    """A term with no antenna rows could only ever fail at evaluation."""
    with pytest.raises(ValueError):
        PolarizationLeakageJones(d_terms=())


def test_an_antenna_row_outside_the_array_is_rejected() -> None:
    """A row/number mix-up fails loudly rather than reading a neighbour's d."""
    with pytest.raises(IndexError):
        _evaluate(_term(0.02), antenna_idx=99)


# ---------------------------------------------------------------------------
# I7 and the Section 20.3 RIME invariants, end to end through the solver
# ---------------------------------------------------------------------------


def _cube(
    tmp_path,
    jones: dict[str, Any] | None,
    *,
    polarized: bool = True,
    **section_overrides: Any,
) -> np.ndarray:
    instrument, beam_system, receptors, jones_terms, frequencies = (
        solver_components_with_jones(tmp_path, jones, **section_overrides)
    )
    return np.asarray(
        calculate_visibility(
            instrument=instrument,
            beam_system=beam_system,
            source_arrays=_workload_point_sources(polarized=polarized, gaussian=False),
            location=WORKLOAD_LOCATION,
            time_grid=WORKLOAD_TIME_GRID,
            frequencies=frequencies,
            backend=_BACKEND,
            receptors=receptors,
            jones_terms=jones_terms,
        )
    )


def test_a_configured_leakage_changes_the_visibilities(tmp_path) -> None:
    """I7, made mechanical: ``Fix.md`` Section 16 rule 5."""
    baseline = _cube(tmp_path, None)
    perturbed = _cube(
        tmp_path,
        {"D": {"d_terms": {"kind": "explicit", "d0": [0.02, 0.0], "d1": [0.0, 0.02]}}},
    )

    scale = float(np.max(np.abs(baseline)))
    assert scale > 0.0
    assert float(np.max(np.abs(perturbed - baseline))) / scale > 1e-10


def _selected_pairs(tmp_path, **section_overrides: Any) -> list[tuple[int, int]]:
    """Return the solver's ``(row_p, row_q)`` pairs, in cube-baseline order.

    The shipped fixture selects **auto**correlations as well as the cross
    baseline, so a per-baseline prediction must know which two antennas each
    cube slot belongs to.  Reading the pairs from the solver view rather than
    assuming them is what keeps these tests honest if the fixture's baseline
    selection ever changes.
    """
    from radiosim.core.instrument_adapters import SolverInstrumentView

    instrument = solver_components_with_jones(tmp_path, None, **section_overrides)[0]
    assert isinstance(instrument, SolverInstrumentView)
    return list(instrument.selected_pairs)


def test_an_unpolarized_source_acquires_exactly_the_predicted_cross_hands(
    tmp_path,
) -> None:
    """Section 20.3's corrected prediction ``V_01 = (I/2)(d_p0 - d_q1)``.

    With an unpolarized sky, a scalar beam and identity receptor terms, the
    leakage-free visibility is ``c I2`` on every ``(time, baseline, channel)``
    cell, so the corrupted cell is exactly ``c D_p D_q^H`` and its ``[0, 1]``
    element is exactly ``c (d_p0 - d_q1)``.  ``c`` is read from the *unleaked*
    run's ``[0, 0]``, which is the point: the prediction is checked against a
    number this term had no part in producing.

    The sign is the one the design review corrected.  ``D_q^H`` contributes
    ``-d_q1`` at ``[0, 1]``, not ``+d_q1^*``; the difference is observable
    whenever the two antennas leak differently, which is why the two antennas
    here are given deliberately different leakages.
    """
    leakages = {
        (0, 0): 0.031 + 0.017j,
        (0, 1): 0.012 - 0.009j,
        (1, 0): -0.005 + 0.021j,
        (1, 1): -0.024 + 0.008j,
    }
    jones = {
        "D": {
            "d_terms": {"kind": "explicit", "d0": [0.0, 0.0], "d1": [0.0, 0.0]},
            "per_antenna": [
                {
                    "antenna": antenna,
                    "feed": feed,
                    "d_term": {"kind": "explicit", "d": [value.real, value.imag]},
                }
                for (antenna, feed), value in leakages.items()
            ],
        }
    }

    clean = _cube(tmp_path, None, polarized=False)
    leaked = _cube(tmp_path, jones, polarized=False)

    # The clean cube really is diagonal, which is what makes the prediction a
    # statement about the leakage rather than about the sky.
    np.testing.assert_allclose(clean[..., 0, 1], 0.0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(clean[..., 0, 0], clean[..., 1, 1], rtol=1e-13)

    for baseline, (row_p, row_q) in enumerate(_selected_pairs(tmp_path)):
        predicted = leakages[(row_p, 0)] - leakages[(row_q, 1)]
        np.testing.assert_allclose(
            leaked[:, baseline, :, 0, 1],
            clean[:, baseline, :, 0, 0] * predicted,
            rtol=1e-12,
            atol=0.0,
        )


def test_the_leakage_multiplies_the_visibility_on_both_sides(tmp_path) -> None:
    """``V_pq -> D_p V_pq D_q^H``, exactly, for an arbitrary polarized sky.

    ``D`` is the only configured term and ``H`` is the identity for the fixture's
    linear receptors, so the corrupted cube must be the clean cube conjugated by
    the two antennas' D-matrices, cell by cell.  This is the strongest available
    statement that the term entered the RIME on *both* sides of the coherency and
    was conjugate-transposed on the second antenna rather than merely
    transposed.
    """
    d_p = (0.04 + 0.02j, -0.01 + 0.03j)
    d_q = (-0.02 - 0.05j, 0.06 + 0.0j)
    jones = {
        "D": {
            "d_terms": {"kind": "explicit", "d0": [0.0, 0.0], "d1": [0.0, 0.0]},
            "per_antenna": [
                {
                    "antenna": antenna,
                    "feed": feed,
                    "d_term": {"kind": "explicit", "d": [value.real, value.imag]},
                }
                for antenna, values in ((0, d_p), (1, d_q))
                for feed, value in enumerate(values)
            ],
        }
    }

    clean = _cube(tmp_path, None)
    leaked = _cube(tmp_path, jones)

    def matrix(pair: tuple[complex, complex]) -> np.ndarray:
        return np.array(
            [[1.0, pair[0]], [-pair[1].conjugate(), 1.0]], dtype=np.complex128
        )

    matrices = {0: matrix(d_p), 1: matrix(d_q)}
    for baseline, (row_p, row_q) in enumerate(_selected_pairs(tmp_path)):
        expected = np.einsum(
            "ij,tfjk,lk->tfil",
            matrices[row_p],
            clean[:, baseline],
            matrices[row_q].conjugate(),
        )
        np.testing.assert_allclose(leaked[:, baseline], expected, rtol=1e-12, atol=0.0)


def test_leakage_does_not_commute_with_a_feed_asymmetric_gain(tmp_path) -> None:
    """``D`` and ``G`` commute only when the two feed gains agree.

    Both are correlator-side of ``C``; the canonical order puts ``G`` nearer the
    correlator, so the composite is ``G D`` and not ``D G``.  With a
    feed-asymmetric gain the two differ, and the difference is what makes the
    order in Section 12.2 a claim with observable content rather than a
    convention.  A feed-symmetric gain is scalar and commutes with everything,
    which is asserted too so that the non-commutation is attributed to the
    asymmetry and not to the mere presence of ``G``.
    """
    d0, d1 = 0.05 + 0.02j, -0.03 - 0.01j
    leakage = np.array([[1.0, d0], [-d1.conjugate(), 1.0]], dtype=np.complex128)
    asymmetric = np.diag(np.array([1.4 + 0.0j, 0.6 + 0.0j]))
    symmetric = np.diag(np.array([1.4 + 0.0j, 1.4 + 0.0j]))

    assert not np.allclose(asymmetric @ leakage, leakage @ asymmetric)
    np.testing.assert_allclose(symmetric @ leakage, leakage @ symmetric)

    jones = {
        "D": {
            "d_terms": {
                "kind": "explicit",
                "d0": [d0.real, d0.imag],
                "d1": [d1.real, d1.imag],
            }
        },
        "G": {
            "amplitude_error": 0.4,
            "per_antenna": [
                {"antenna": antenna, "feed": 1, "amplitude_error": -0.4}
                for antenna in (0, 1)
            ],
        },
    }

    clean = _cube(tmp_path, None)
    corrupted = _cube(tmp_path, jones)

    # The composite really is G D, not D G: reconstructing the cube with the
    # matrices in the wrong order does not reproduce it.
    correct = np.einsum(
        "ij,tbfjk,lk->tbfil",
        asymmetric @ leakage,
        clean,
        (asymmetric @ leakage).conjugate(),
    )
    reversed_order = np.einsum(
        "ij,tbfjk,lk->tbfil",
        leakage @ asymmetric,
        clean,
        (leakage @ asymmetric).conjugate(),
    )

    np.testing.assert_allclose(corrupted, correct, rtol=1e-12, atol=0.0)
    scale = float(np.max(np.abs(correct)))
    assert float(np.max(np.abs(correct - reversed_order))) / scale > 1e-6


def test_leakage_reaches_a_circular_receptor_in_its_own_basis(tmp_path) -> None:
    """Section 20.0: ``D`` is defined per feed index, in the receptor's basis.

    With circular receptors reported in a linear output basis, ``C`` and ``H``
    are both non-trivial, and the composite is ``H_p D_p C_p E_p``.  The
    leakage must still multiply on the receptor side of ``C`` -- that is the
    physical claim in Section 12.3 -- so reconstructing the cube by conjugating
    the clean one with ``H D H^H`` reproduces it exactly, while conjugating with
    the raw ``D`` does not.
    """
    receptors = {
        "receptors": {
            "default": {"basis": "circular", "feed_rotation_deg": 0.0},
            "output_basis": "linear",
        }
    }
    d0, d1 = 0.06 + 0.03j, -0.02 + 0.05j
    jones = {
        "D": {
            "d_terms": {
                "kind": "explicit",
                "d0": [d0.real, d0.imag],
                "d1": [d1.real, d1.imag],
            }
        }
    }

    clean = _cube(tmp_path, None, **receptors)
    leaked = _cube(tmp_path, jones, **receptors)

    leakage = np.array([[1.0, d0], [-d1.conjugate(), 1.0]], dtype=np.complex128)
    # SCI-006: H for circular native reported in east-X linear is P S^H.
    s_matrix = np.array([[1.0, 1.0j], [1.0, -1.0j]], dtype=np.complex128) / math.sqrt(
        2.0
    )
    permutation = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    transform = permutation @ s_matrix.conj().T
    in_output_basis = transform @ leakage @ transform.conj().T

    expected = np.einsum(
        "ij,tbfjk,lk->tbfil", in_output_basis, clean, in_output_basis.conjugate()
    )
    np.testing.assert_allclose(leaked, expected, rtol=1e-11, atol=0.0)

    wrong = np.einsum("ij,tbfjk,lk->tbfil", leakage, clean, leakage.conjugate())
    scale = float(np.max(np.abs(expected)))
    assert float(np.max(np.abs(expected - wrong))) / scale > 1e-6
