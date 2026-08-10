"""Tier 7E: the ``X`` term's physics, flags, and effect on the visibilities.

``Tier7JonesSciencePlan.md`` Section 20.4.  Cross-hand phase and cross-hand
delay are the same diagonal matrix -- one frequency-constant phase and one
frequency-linear one -- so they are one term with two parameters::

    X_p(nu) = diag( 1, exp( i (phi_x + 2 pi nu tau_x) ) )

Only the *relative* phase between the two feeds is physical, which is why the
first entry is exactly ``1`` rather than a second free parameter: a second
parameter would be degenerate with ``G``.

Invariants asserted here: **I2**, **I3**, **I7**, and Section 20.4's own sharpest
statement -- for a linear receptor a cross-hand phase ``phi_x`` rotates Stokes
``U`` into Stokes ``V`` by exactly ``phi_x``.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.core.jones.crosshand import CrosshandJones
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


def _term(
    phase_rad: float = 0.0,
    delay_s: float = 0.0,
    *,
    rows: int = 2,
) -> CrosshandJones:
    """Build one ``X`` with the same phase and delay on every antenna row."""
    return CrosshandJones(
        phases_rad=np.full(rows, phase_rad, dtype=np.float64),
        delays_s=np.full(rows, delay_s, dtype=np.float64),
    )


def _evaluate(
    term: CrosshandJones,
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
# The closed form (Section 20.4), evaluated independently
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phase_rad", "delay_s", "frequency_hz"),
    [
        (0.4, 0.0, 1.0e8),
        (0.0, 1.0e-9, 1.0e8),
        (-1.2, 2.5e-9, 1.5e8),
        (math.pi, -1.0e-9, 2.0e8),
    ],
)
def test_the_matrix_is_the_published_diagonal_form(
    phase_rad: float,
    delay_s: float,
    frequency_hz: float,
) -> None:
    """``X = diag(1, exp(i (phi + 2 pi nu tau)))`` at machine precision.

    The first entry is asserted to be *exactly* ``1``: it is not a rounded
    result of some computation but the statement that ``X`` carries one degree
    of freedom, the relative phase, and not two.
    """
    total = phase_rad + 2.0 * math.pi * frequency_hz * delay_s
    expected = complex(math.cos(total), math.sin(total))

    block = _evaluate(_term(phase_rad, delay_s), frequency_hz=frequency_hz)

    assert block.shape == (1, 2, 2)
    assert block[0, 0, 0] == 1.0 + 0.0j
    assert block[0, 1, 1] == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert block[0, 0, 1] == 0.0
    assert block[0, 1, 0] == 0.0


def test_the_cross_hand_delay_phase_is_exactly_linear_in_frequency() -> None:
    """The delay half of the term, checked by a fit rather than by a value.

    A cross-hand *delay* is the statement that the relative phase advances
    linearly with frequency; fitting the unwrapped phase and recovering
    ``2 pi tau`` as the slope is the direct test of that, and it would fail for
    any implementation that made the phase merely monotonic.
    """
    delay = 3.0e-9
    phase = 0.25
    term = _term(phase, delay)

    frequencies = np.linspace(1.0e8, 1.05e8, 32)
    measured = np.unwrap(
        np.angle([_evaluate(term, frequency_hz=nu)[0, 1, 1] for nu in frequencies])
    )
    slope, intercept = np.polyfit(frequencies, measured, 1)

    assert slope == pytest.approx(2.0 * math.pi * delay, rel=1e-10)
    assert math.remainder(intercept - phase, 2.0 * math.pi) == pytest.approx(
        0.0, abs=1e-6
    )


def test_a_die_term_returns_one_broadcast_matrix() -> None:
    """I3: ``(1, 2, 2)``, never ``n_dir`` copies of one constant."""
    term = _term(0.3, 1.0e-9)

    assert term.is_direction_dependent is False
    assert _evaluate(term).shape == (1, 2, 2)


# ---------------------------------------------------------------------------
# I2 -- declared flags are true, and each declared False has a witness
# ---------------------------------------------------------------------------


def test_x_is_always_diagonal_and_always_unitary() -> None:
    """Section 20.4: ``X`` is a pure relative phase, so it preserves power."""
    for phase, delay in ((0.4, 0.0), (-2.0, 1.0e-9), (1.1, -3.0e-9)):
        term = _term(phase, delay)
        assert term.is_diagonal() is True
        assert term.is_unitary() is True
        matrix = _evaluate(term)[0]
        assert matrix[0, 1] == 0.0
        assert matrix[1, 0] == 0.0
        np.testing.assert_allclose(
            matrix @ matrix.conj().T, np.eye(2), rtol=0.0, atol=1e-15
        )


def test_scalarity_is_declared_only_for_the_identity() -> None:
    """I2's converse: a declared ``False`` has a witness where it fails.

    ``X`` is scalar only when the relative phase is exactly zero, which is
    exactly when it is the identity -- and R7 rejects that from configuration.
    The flag is still computed from the resolved numbers rather than hard-coded
    ``False``, because a hard-coded flag is the vacuous claim invariant I2
    exists to prevent.
    """
    phased = _term(0.4)
    assert phased.is_scalar() is False
    witness = _evaluate(phased)[0]
    assert witness[0, 0] != witness[1, 1]

    trivial = _term(0.0, 0.0)
    assert trivial.is_scalar() is True
    assert trivial.is_identity() is True
    np.testing.assert_array_equal(_evaluate(trivial)[0], np.eye(2, dtype=np.complex128))


def test_x_declares_frequency_dependence_only_when_a_delay_is_configured() -> None:
    """A pure cross-hand *phase* is frequency-flat; a *delay* is not."""
    flat = _term(0.7, 0.0)
    assert flat.is_frequency_dependent is False
    np.testing.assert_array_equal(
        _evaluate(flat, frequency_hz=1.0e8), _evaluate(flat, frequency_hz=2.0e8)
    )

    dispersive = _term(0.0, 2.0e-9)
    assert dispersive.is_frequency_dependent is True
    assert not np.array_equal(
        _evaluate(dispersive, frequency_hz=1.0e8),
        _evaluate(dispersive, frequency_hz=2.0e8),
    )

    assert flat.is_direction_dependent is False
    assert flat.is_time_dependent is False


def test_the_term_status_is_implemented() -> None:
    """Section 31 step 5, and the ``"implemented"`` half of invariant I20."""
    assert _term(0.4).term_status == "implemented"
    assert _term(0.4).name == "X"


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_a_non_finite_parameter_cannot_be_constructed() -> None:
    """Caught at construction, so no ``nan`` ever reaches a chain."""
    with pytest.raises(ValueError):
        CrosshandJones(phases_rad=np.array([np.nan]), delays_s=np.array([0.0]))
    with pytest.raises(ValueError):
        CrosshandJones(phases_rad=np.array([0.0]), delays_s=np.array([np.inf]))


def test_mismatched_parameter_lengths_are_rejected() -> None:
    """One phase and one delay per antenna row, or nothing."""
    with pytest.raises(ValueError):
        CrosshandJones(phases_rad=np.array([0.1, 0.2]), delays_s=np.array([0.0]))


def test_an_antenna_row_outside_the_array_is_rejected() -> None:
    """A row/number mix-up fails loudly rather than reading a neighbour's phase."""
    with pytest.raises(IndexError):
        _evaluate(_term(0.4), antenna_idx=99)


# ---------------------------------------------------------------------------
# I7 and the Section 20.4 RIME invariants, end to end through the solver
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


def test_a_configured_cross_hand_phase_changes_the_visibilities(tmp_path) -> None:
    """I7, made mechanical: ``Fix.md`` Section 16 rule 5."""
    baseline = _cube(tmp_path, None)
    perturbed = _cube(tmp_path, {"X": {"phase_rad": 0.35}})

    scale = float(np.max(np.abs(baseline)))
    assert scale > 0.0
    assert float(np.max(np.abs(perturbed - baseline))) / scale > 1e-10


def test_a_cross_hand_phase_rotates_u_into_v_and_leaves_the_parallel_hands(
    tmp_path,
) -> None:
    """Section 20.4's classic X-Y phase signature, at machine precision.

    With linear receptors and the same ``X`` on both antennas,
    ``V -> X V X^H``, so the parallel hands are untouched and the cross hands
    pick up ``exp(-i phi)`` and ``exp(+i phi)``.  Writing the cross hand as
    ``(U - iV)/2`` in the clean east-X run, the corrupted cross hand is
    ``(U - iV) exp(-i phi)/2 = (U' - iV')/2`` -- which is exactly the statement
    that ``U`` rotates into ``V`` by ``phi``.  Both the algebraic form and the
    Stokes reading are asserted, because the second is the one a user cares
    about and the first is the one that localizes a bug.
    """
    phase = 0.63
    clean = _cube(tmp_path, None)
    rotated = _cube(tmp_path, {"X": {"phase_rad": phase}})

    np.testing.assert_allclose(rotated[..., 0, 0], clean[..., 0, 0], rtol=1e-13)
    np.testing.assert_allclose(rotated[..., 1, 1], clean[..., 1, 1], rtol=1e-13)
    np.testing.assert_allclose(
        rotated[..., 0, 1], clean[..., 0, 1] * cmath.exp(-1j * phase), rtol=1e-12
    )
    np.testing.assert_allclose(
        rotated[..., 1, 0], clean[..., 1, 0] * cmath.exp(1j * phase), rtol=1e-12
    )

    # The Stokes reading of the same numbers.  The reported cross hand is
    # ``(U - iV)/2`` times the common beam and fringe factor, so twice its real
    # part and minus twice its imaginary part are the ``(U, V)`` pair, and the
    # statement "``U`` rotates into ``V`` by ``phi``" is that this pair rotates
    # rigidly -- same modulus, angle advanced by exactly ``phi``.
    clean_u = 2.0 * np.real(clean[..., 0, 1])
    clean_v = -2.0 * np.imag(clean[..., 0, 1])
    rotated_u = 2.0 * np.real(rotated[..., 0, 1])
    rotated_v = -2.0 * np.imag(rotated[..., 0, 1])

    np.testing.assert_allclose(
        rotated_u,
        clean_u * math.cos(phase) - clean_v * math.sin(phase),
        rtol=1e-11,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        rotated_v,
        clean_v * math.cos(phase) + clean_u * math.sin(phase),
        rtol=1e-11,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.hypot(rotated_u, rotated_v), np.hypot(clean_u, clean_v), rtol=1e-11
    )


def test_a_cross_hand_phase_common_to_the_array_does_not_cancel(tmp_path) -> None:
    """The contrast with ``G``, and the reason ``X`` is a term of its own.

    A gain phase common to the whole array cancels between ``J_p`` and
    ``J_q^H``.  A cross-hand phase does not: it is a phase on one feed only, so
    it survives on the cross hands even when every antenna carries the same
    value.  This is the property that makes ``X`` non-degenerate with ``G``.
    """
    clean = _cube(tmp_path, None)
    rotated = _cube(tmp_path, {"X": {"phase_rad": 0.5}})

    scale = float(np.max(np.abs(clean)))
    assert float(np.max(np.abs(rotated[..., 0, 1] - clean[..., 0, 1]))) / scale > 1e-6


def test_a_cross_hand_delay_produces_a_frequency_dependent_rotation(
    tmp_path,
) -> None:
    """The delay half, end to end: the cross-hand phase differs per channel."""
    delay = 4.0e-8
    frequencies = np.asarray(
        solver_components_with_jones(tmp_path, None)[4], dtype=np.float64
    )

    clean = _cube(tmp_path, None)
    delayed = _cube(tmp_path, {"X": {"phase_rad": 0.0, "delay_s": delay}})

    for index, frequency in enumerate(frequencies):
        expected = cmath.exp(-2j * math.pi * float(frequency) * delay)
        np.testing.assert_allclose(
            delayed[:, :, index, 0, 1],
            clean[:, :, index, 0, 1] * expected,
            rtol=1e-11,
        )

    # And the channels really did receive different rotations.
    assert frequencies.size > 1
    first = delayed[:, :, 0, 0, 1] / clean[:, :, 0, 0, 1]
    last = delayed[:, :, -1, 0, 1] / clean[:, :, -1, 0, 1]
    assert float(np.max(np.abs(first - last))) > 1e-6


def test_x_commutes_with_gain_bandpass_delay_and_cable_reflection(tmp_path) -> None:
    """Section 20.4: all five correlator-side factors are diagonal.

    Their relative order in Section 12.2 is a *convention*, not a physical claim,
    and this is the assertion that makes the convention safe: two chains whose
    diagonal factors are composed in opposite orders give the same product.
    """
    phasor = np.diag(np.array([1.0 + 0.0j, cmath.exp(1j * 0.4)]))
    gain = np.diag(np.array([1.2 + 0.1j, 0.7 - 0.3j]))
    bandpass = np.diag(np.array([0.9 + 0.0j, 1.1 + 0.2j]))
    delay = np.diag(np.array([cmath.exp(-0.3j), cmath.exp(0.8j)], dtype=np.complex128))
    reflection = np.diag(np.array([1.0 + 0.05j, 1.0 - 0.02j]))

    for other in (gain, bandpass, delay, reflection):
        np.testing.assert_allclose(phasor @ other, other @ phasor, rtol=0.0, atol=1e-16)

    # And the same statement through the solver: G before X and X alone compose
    # to the same cube whichever way the two are written in the document.
    written_one_way = _cube(
        tmp_path,
        {"X": {"phase_rad": 0.4}, "G": {"amplitude_error": 0.2}},
    )
    written_the_other_way = _cube(
        tmp_path,
        {"G": {"amplitude_error": 0.2}, "X": {"phase_rad": 0.4}},
    )
    np.testing.assert_array_equal(written_one_way, written_the_other_way)


def test_a_cross_hand_phase_on_a_circular_receptor_moves_q_into_u(
    tmp_path,
) -> None:
    """The same term in the other receptor basis, where it means something else.

    ``X`` is defined per feed *index* in the antenna's own basis (Section 20.0),
    so on a circular receptor the phased feed is ``L`` rather than ``y``, and the
    corrupted correlations are the ``(RL, LR)`` pair.  Reported in a linear
    output basis the composite is ``H X H^H``, and asserting the cube against
    that -- rather than against the raw ``X`` -- is what pins the chain position
    of ``X`` correlator-side of ``C``.
    """
    phase = 0.55
    receptors = {
        "receptors": {
            "default": {"basis": "circular", "feed_rotation_deg": 0.0},
            "output_basis": "linear",
        }
    }

    clean = _cube(tmp_path, None, **receptors)
    phased = _cube(tmp_path, {"X": {"phase_rad": phase}}, **receptors)

    crosshand = np.diag(np.array([1.0 + 0.0j, cmath.exp(1j * phase)]))
    s_matrix = np.array([[1.0, 1.0j], [1.0, -1.0j]], dtype=np.complex128) / math.sqrt(
        2.0
    )
    permutation = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    transform = permutation @ s_matrix.conj().T
    in_output_basis = transform @ crosshand @ transform.conj().T

    expected = np.einsum(
        "ij,tbfjk,lk->tbfil", in_output_basis, clean, in_output_basis.conjugate()
    )
    np.testing.assert_allclose(phased, expected, rtol=1e-11, atol=1e-18)

    # In this basis the term is *not* diagonal in the reported correlations,
    # which is the whole reason the chain position matters.
    assert abs(in_output_basis[0, 1]) > 1e-3


# ---------------------------------------------------------------------------
# Resolution precedence
# ---------------------------------------------------------------------------


def test_a_per_antenna_entry_beats_the_array_wide_default(tmp_path) -> None:
    """Section 22 rule 5, on the resolved numbers.

    ``X``'s ``per_antenna`` entries carry no ``feed`` key, because the relative
    phase between the two feeds is one number per antenna: a feed index would
    have to name the feed the phase is *not* on.
    """
    resolved = resolve_for(
        tmp_path,
        {
            "X": {
                "phase_rad": 0.2,
                "delay_s": 0.0,
                "per_antenna": [{"antenna": 1, "phase_rad": 1.3, "delay_s": 5.0e-9}],
            }
        },
    )
    term = resolved.term("X")
    assert term is not None

    phasors = term.phasors_at_frequency(1.0e8)
    assert phasors[0] == pytest.approx(cmath.exp(1j * 0.2), rel=1e-14)
    assert phasors[1] == pytest.approx(
        cmath.exp(1j * (1.3 + 2.0 * math.pi * 1.0e8 * 5.0e-9)), rel=1e-14
    )
