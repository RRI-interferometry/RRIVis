"""Tier 7E: the ``Kd`` term's physics, flags, and effect on the visibilities.

``Tier7JonesSciencePlan.md`` Section 20.5::

    Kd_p(nu) = diag( exp(-2 pi i nu tau_p0), exp(-2 pi i nu tau_p1) )

The negative exponent is not a free choice: Section 20.0 fixes one sign
convention for the whole tier, matching the geometric phase's own
``exp(-2 pi i b.s)``, so that a positive delay produces ``exp(-i * positive)``
everywhere.  That is invariant **I4**, and it is asserted here directly rather
than inferred from a difference of two cubes.

Invariants asserted here: **I2**, **I3**, **I4**, **I7**, and Section 20.5's own
statement -- a delay common to both feeds of every antenna cancels exactly on
every cross-correlation, which is the cleanest possible "the effect really is
applied per antenna and conjugated on the second one" test.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.core.jones.delay import DelayJones
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


def _term(delay_0: float = 0.0, delay_1: float | None = None) -> DelayJones:
    """Build one two-row ``Kd`` with the same delays on both antenna rows."""
    second = delay_0 if delay_1 is None else delay_1
    return DelayJones(
        delays_s=np.array([[delay_0, second], [delay_0, second]], dtype=np.float64)
    )


def _evaluate(
    term: DelayJones,
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
# The closed form (Section 20.5), evaluated independently
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delay_0", "delay_1", "frequency_hz"),
    [
        (1.0e-9, 2.0e-9, 1.0e8),
        (-5.0e-10, 5.0e-10, 1.5e8),
        (0.0, 3.0e-9, 2.0e8),
        (1.0e-8, 1.0e-8, 1.2e8),
    ],
)
def test_the_matrix_is_the_published_delay_phasor(
    delay_0: float,
    delay_1: float,
    frequency_hz: float,
) -> None:
    """``Kd = diag(exp(-2 pi i nu tau_0), exp(-2 pi i nu tau_1))``.

    Written out here from Section 20.5; nothing in this assertion calls the
    term's own arithmetic.
    """
    block = _evaluate(_term(delay_0, delay_1), frequency_hz=frequency_hz)

    assert block.shape == (1, 2, 2)
    for feed, delay in enumerate((delay_0, delay_1)):
        angle = -2.0 * math.pi * frequency_hz * delay
        expected = complex(math.cos(angle), math.sin(angle))
        assert block[0, feed, feed] == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert block[0, 0, 1] == 0.0
    assert block[0, 1, 0] == 0.0


def test_a_positive_delay_gives_a_negative_phase(tmp_path) -> None:
    """I4: the sign convention, asserted directly rather than inferred.

    Section 20.0 fixes ``exp(-i phi)`` for a positive excess path length, so that
    ``Kd`` composes with the solver's own ``exp(-2 pi i b.s)`` geometric phase
    rather than fighting it.  A term written with the opposite sign would still
    pass every "the delay changes the visibilities" test and every parity test,
    and would be wrong.
    """
    frequency = 1.0e8
    delay = 2.5e-9

    phase = cmath.phase(_evaluate(_term(delay, delay), frequency_hz=frequency)[0, 0, 0])

    assert phase < 0.0
    assert phase == pytest.approx(
        math.remainder(-2.0 * math.pi * frequency * delay, 2.0 * math.pi), abs=1e-13
    )

    # And the same sign through the resolved configuration path, so that the
    # convention cannot be inverted by the resolution step.
    resolved = resolve_for(tmp_path, {"Kd": {"delay_s": delay}})
    term = resolved.term("Kd")
    assert term is not None
    assert cmath.phase(term.phasors_at_frequency(frequency)[0, 0]) < 0.0


def test_the_delay_phase_is_exactly_linear_in_frequency() -> None:
    """Section 20.5: the phase is linear in frequency, checkable by fitting."""
    delay = 7.0e-9
    term = _term(delay, delay)

    frequencies = np.linspace(1.0e8, 1.02e8, 48)
    measured = np.unwrap(
        np.angle([_evaluate(term, frequency_hz=nu)[0, 0, 0] for nu in frequencies])
    )
    slope = float(np.polyfit(frequencies, measured, 1)[0])

    assert slope == pytest.approx(-2.0 * math.pi * delay, rel=1e-10)


def test_a_die_term_returns_one_broadcast_matrix() -> None:
    """I3: ``(1, 2, 2)``, never ``n_dir`` copies of one constant."""
    term = _term(1.0e-9, 2.0e-9)

    assert term.is_direction_dependent is False
    assert _evaluate(term).shape == (1, 2, 2)


# ---------------------------------------------------------------------------
# I2 -- declared flags are true, and each declared False has a witness
# ---------------------------------------------------------------------------


def test_kd_is_always_diagonal_and_always_unitary() -> None:
    """Section 20.5: a pure delay is a pure phase, so it preserves power."""
    for delay_0, delay_1 in ((1.0e-9, 2.0e-9), (-3.0e-9, 0.0), (5.0e-9, 5.0e-9)):
        term = _term(delay_0, delay_1)
        assert term.is_diagonal() is True
        assert term.is_unitary() is True
        matrix = _evaluate(term)[0]
        assert matrix[0, 1] == 0.0
        assert matrix[1, 0] == 0.0
        np.testing.assert_allclose(
            matrix @ matrix.conj().T, np.eye(2), rtol=0.0, atol=1e-15
        )


def test_scalarity_is_declared_exactly_when_the_two_feeds_share_a_delay() -> None:
    """I2 in both directions, and ``Kd`` is the term where ``True`` is reachable.

    A delay common to both feeds of an antenna is a scalar phase on that
    antenna -- a physically meaningful, non-identity configuration -- so unlike
    ``X`` the scalarity flag here has a witness on both sides.
    """
    shared = _term(4.0e-9, 4.0e-9)
    assert shared.is_scalar() is True
    matrix = _evaluate(shared)[0]
    np.testing.assert_array_equal(matrix, matrix[0, 0] * np.eye(2, dtype=np.complex128))
    assert matrix[0, 0] != 1.0 + 0.0j

    split = _term(4.0e-9, 1.0e-9)
    assert split.is_scalar() is False
    witness = _evaluate(split)[0]
    assert witness[0, 0] != witness[1, 1]


def test_kd_declares_the_dependencies_it_actually_has() -> None:
    """Direction- and time-independent; frequency-dependent for any real delay."""
    term = _term(1.0e-9, 3.0e-9)

    assert term.is_direction_dependent is False
    assert term.is_time_dependent is False
    assert term.is_frequency_dependent is True
    assert not np.array_equal(
        _evaluate(term, frequency_hz=1.0e8), _evaluate(term, frequency_hz=1.1e8)
    )

    trivial = _term(0.0, 0.0)
    assert trivial.is_frequency_dependent is False
    assert trivial.is_identity() is True
    np.testing.assert_array_equal(_evaluate(trivial)[0], np.eye(2, dtype=np.complex128))


def test_the_term_status_is_implemented() -> None:
    """Section 31 step 5, and the ``"implemented"`` half of invariant I20."""
    assert _term(1.0e-9).term_status == "implemented"
    assert _term(1.0e-9).name == "Kd"


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_a_non_finite_delay_cannot_be_constructed() -> None:
    """Caught at construction, so no ``nan`` ever reaches a chain."""
    with pytest.raises(ValueError):
        DelayJones(delays_s=np.array([[np.nan, 0.0]], dtype=np.float64))


def test_a_wrongly_shaped_delay_table_is_rejected() -> None:
    """One delay per antenna row per feed, or nothing."""
    with pytest.raises(ValueError):
        DelayJones(delays_s=np.array([1.0e-9, 2.0e-9], dtype=np.float64))
    with pytest.raises(ValueError):
        DelayJones(delays_s=np.zeros((0, 2), dtype=np.float64))


def test_an_antenna_row_outside_the_array_is_rejected() -> None:
    """A row/number mix-up fails loudly rather than reading a neighbour's delay."""
    with pytest.raises(IndexError):
        _evaluate(_term(1.0e-9), antenna_idx=99)


# ---------------------------------------------------------------------------
# I7 and the Section 20.5 RIME invariants, end to end through the solver
# ---------------------------------------------------------------------------


def _cube(
    tmp_path,
    jones: dict[str, Any] | None,
    **section_overrides: Any,
) -> np.ndarray:
    instrument, beam_system, receptors, jones_terms, frequencies = (
        solver_components_with_jones(tmp_path, jones, **section_overrides)
    )
    return np.asarray(
        calculate_visibility(
            instrument=instrument,
            beam_system=beam_system,
            source_arrays=_workload_point_sources(polarized=True, gaussian=False),
            location=WORKLOAD_LOCATION,
            time_grid=WORKLOAD_TIME_GRID,
            frequencies=frequencies,
            backend=_BACKEND,
            receptors=receptors,
            jones_terms=jones_terms,
        )
    )


def test_a_configured_delay_changes_the_visibilities(tmp_path) -> None:
    """I7, made mechanical: ``Fix.md`` Section 16 rule 5.

    The delay is per feed here, because a delay common to both feeds of every
    antenna is exactly the configuration the next test proves has *no* effect --
    so using one would make this test pass for the wrong reason.
    """
    baseline = _cube(tmp_path, None)
    perturbed = _cube(
        tmp_path,
        {
            "Kd": {
                "delay_s": 0.0,
                "per_antenna": [{"antenna": 0, "feed": 1, "delay_s": 2.0e-8}],
            }
        },
    )

    scale = float(np.max(np.abs(baseline)))
    assert scale > 0.0
    assert float(np.max(np.abs(perturbed - baseline))) / scale > 1e-10


def test_a_delay_common_to_every_antenna_cancels_on_every_baseline(
    tmp_path,
) -> None:
    """Section 20.5's own invariant: the zero-differential baseline.

    ``Kd`` enters as ``exp(-2 pi i nu tau_p)`` on antenna ``p`` and as its
    conjugate on antenna ``q``, so a baseline whose two antennas share a delay
    sees ``exp(-2 pi i nu (tau_p - tau_q)) = 1``.  A term applied on one side
    only, or conjugated on the wrong side, would leave a residual phase here --
    which makes this the cleanest available proof that the term is applied per
    antenna and Hermitian-conjugated on the second.

    The delay chosen is large enough that a one-sided application would rotate
    the visibility by many radians, so the cancellation is not the cancellation
    of a small number.
    """
    delay = 3.0e-8
    baseline = _cube(tmp_path, None)
    delayed = _cube(tmp_path, {"Kd": {"delay_s": delay}})

    # A one-sided application would look like this, and it does not.
    rotated = baseline * cmath.exp(-2j * math.pi * 1.0e8 * delay)
    scale = float(np.max(np.abs(baseline)))
    assert float(np.max(np.abs(rotated - baseline))) / scale > 1.0

    np.testing.assert_allclose(delayed, baseline, rtol=1e-12, atol=0.0)


def test_a_differential_delay_is_a_pure_baseline_phase_slope(tmp_path) -> None:
    """The complement: the residual is exactly the differential-delay fringe.

    With ``tau_p`` on antenna 0's two feeds and nothing on antenna 1, every
    correlation of the baseline is multiplied by ``exp(-2 pi i nu tau_p)`` --
    amplitudes untouched, phase linear in frequency.  Both halves are asserted,
    because a term that got the amplitude wrong and the phase right would still
    pass a phase-only check.
    """
    delay = 5.0e-9
    frequencies = np.asarray(
        solver_components_with_jones(tmp_path, None)[4], dtype=np.float64
    )

    baseline = _cube(tmp_path, None)
    delayed = _cube(
        tmp_path,
        {
            "Kd": {
                "delay_s": 0.0,
                "per_antenna": [
                    {"antenna": 0, "feed": 0, "delay_s": delay},
                    {"antenna": 0, "feed": 1, "delay_s": delay},
                ],
            }
        },
    )

    np.testing.assert_allclose(
        np.abs(delayed), np.abs(baseline), rtol=1e-12, atol=1e-18
    )
    for index, frequency in enumerate(frequencies):
        expected = cmath.exp(-2j * math.pi * float(frequency) * delay)
        np.testing.assert_allclose(
            delayed[:, :, index, :, :],
            baseline[:, :, index, :, :] * expected,
            rtol=1e-11,
            atol=1e-18,
        )


def test_a_delay_leaves_the_closure_phase_invariant(tmp_path) -> None:
    """``Kd`` is antenna-based, so it cannot break closure.

    The same statement ``G`` carries, asserted again for ``Kd`` because the
    delay is the term most often *mistaken* for a baseline effect: a fringe
    slope looks like a baseline property, and it is not one.
    """
    from radiosim.core.instrument_adapters import SolverInstrumentView
    from tests.unit.test_core.test_jones_resolution import three_antenna_layout

    jones = {
        "Kd": {
            "delay_s": 0.0,
            "per_antenna": [
                {"antenna": antenna, "feed": feed, "delay_s": delay}
                for antenna, delay in ((0, 4.0e-9), (1, -7.0e-9), (2, 1.1e-8))
                for feed in (0, 1)
            ],
        }
    }
    triangle = three_antenna_layout(tmp_path)
    instrument = solver_components_with_jones(tmp_path, jones, **triangle)[0]
    assert isinstance(instrument, SolverInstrumentView)
    pairs = list(instrument.selected_pairs)

    def closure(cube: np.ndarray) -> np.ndarray:
        index = {pair: position for position, pair in enumerate(pairs)}
        first = cube[:, index[(0, 1)], :, 0, 0]
        second = cube[:, index[(1, 2)], :, 0, 0]
        third = cube[:, index[(0, 2)], :, 0, 0]
        return np.angle(first * second * np.conj(third))

    baseline = _cube(tmp_path, None, **triangle)
    perturbed = _cube(tmp_path, jones, **triangle)

    np.testing.assert_allclose(
        closure(perturbed), closure(baseline), rtol=0.0, atol=1e-12
    )
    scale = float(np.max(np.abs(baseline)))
    assert float(np.max(np.abs(perturbed - baseline))) / scale > 1e-6


def test_a_per_antenna_entry_beats_the_array_wide_default(tmp_path) -> None:
    """Section 22 rule 5, on the resolved numbers."""
    resolved = resolve_for(
        tmp_path,
        {
            "Kd": {
                "delay_s": 1.0e-9,
                "per_antenna": [{"antenna": 1, "feed": 0, "delay_s": 6.0e-9}],
            }
        },
    )
    term = resolved.term("Kd")
    assert term is not None

    phasors = term.phasors_at_frequency(1.0e8)
    assert phasors[0, 0] == pytest.approx(
        cmath.exp(-2j * math.pi * 1.0e8 * 1.0e-9), rel=1e-14
    )
    assert phasors[0, 1] == pytest.approx(
        cmath.exp(-2j * math.pi * 1.0e8 * 1.0e-9), rel=1e-14
    )
    assert phasors[1, 0] == pytest.approx(
        cmath.exp(-2j * math.pi * 1.0e8 * 6.0e-9), rel=1e-14
    )
    assert phasors[1, 1] == pytest.approx(
        cmath.exp(-2j * math.pi * 1.0e8 * 1.0e-9), rel=1e-14
    )
