"""Tier 7E: the ``Rc`` term's physics, flags, and effect on the visibilities.

``Tier7JonesSciencePlan.md`` Section 20.6::

    Rc_p(nu) = diag( r_p0(nu), r_p1(nu) )
    r_pf(nu) = 1 + A_pf exp( -2 pi i nu tau_cable,pf + i phi_pf )

This is the first-order, single-bounce reflection.  ``A`` is dimensionless with
``0 < |A| < 1`` enforced (R8), ``tau_cable`` is the round-trip cable delay in
seconds, and ``phi`` is a phase offset in radians.  The exponent's sign is the
tier-wide convention of Section 20.0, the same one ``Kd`` and the geometric
phase use.

Invariants asserted here: **I2**, **I3**, **I4**, **I7**, and Section 20.6's own
two statements -- ``|r|`` oscillates between ``1 - A`` and ``1 + A`` with
frequency period ``1/tau_cable``, and the delay-domain transform of a corrupted
spectrum carries a secondary peak at exactly ``tau_cable`` with relative
amplitude ``A``.  The second is the reason ``Rc`` is a term of its own rather
than a bandpass shape, so it is asserted numerically and not merely described.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.core.jones.delay import CableReflectionJones
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
    amplitude: float = 0.01,
    cable_delay_s: float = 1.5e-7,
    phase_rad: float = 0.0,
    *,
    second: tuple[float, float, float] | None = None,
    rows: int = 2,
) -> CableReflectionJones:
    """Build one ``Rc`` with the same parameters on every antenna row."""
    feed_1 = (amplitude, cable_delay_s, phase_rad) if second is None else second
    return CableReflectionJones(
        amplitudes=np.array([[amplitude, feed_1[0]]] * rows, dtype=np.float64),
        cable_delays_s=np.array([[cable_delay_s, feed_1[1]]] * rows, dtype=np.float64),
        phases_rad=np.array([[phase_rad, feed_1[2]]] * rows, dtype=np.float64),
    )


def _evaluate(
    term: CableReflectionJones,
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
# The closed form (Section 20.6), evaluated independently
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("amplitude", "cable_delay_s", "phase_rad", "frequency_hz"),
    [
        (0.01, 1.5e-7, 0.0, 1.0e8),
        (0.2, 3.0e-8, 1.1, 1.5e8),
        (-0.05, 1.0e-7, -0.4, 2.0e8),
        (0.5, 2.5e-7, math.pi, 1.2e8),
    ],
)
def test_the_matrix_is_the_published_reflection_form(
    amplitude: float,
    cable_delay_s: float,
    phase_rad: float,
    frequency_hz: float,
) -> None:
    """``r = 1 + A exp(-2 pi i nu tau_c + i phi)`` at machine precision."""
    angle = -2.0 * math.pi * frequency_hz * cable_delay_s + phase_rad
    expected = 1.0 + amplitude * complex(math.cos(angle), math.sin(angle))

    block = _evaluate(
        _term(amplitude, cable_delay_s, phase_rad), frequency_hz=frequency_hz
    )

    assert block.shape == (1, 2, 2)
    assert block[0, 0, 0] == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert block[0, 1, 1] == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert block[0, 0, 1] == 0.0
    assert block[0, 1, 0] == 0.0


def test_a_positive_cable_delay_gives_a_negative_phase() -> None:
    """I4, for the term whose delay hides inside a sum rather than an exponent.

    ``Rc``'s delay is not the phase of the matrix element -- the leading ``1``
    sees to that -- so the sign convention has to be read off the reflected
    component.  Subtracting the ``1`` recovers it exactly.
    """
    amplitude, delay, frequency = 0.3, 4.0e-9, 1.0e8

    reflected = (
        _evaluate(_term(amplitude, delay, 0.0), frequency_hz=frequency)[0, 0, 0] - 1.0
    )

    assert cmath.phase(reflected) < 0.0
    assert cmath.phase(reflected) == pytest.approx(
        math.remainder(-2.0 * math.pi * frequency * delay, 2.0 * math.pi), abs=1e-13
    )
    assert abs(reflected) == pytest.approx(amplitude, rel=1e-14)


def test_the_modulus_oscillates_between_one_minus_a_and_one_plus_a() -> None:
    """Section 20.6's first invariant, with the period checked too.

    ``|r|`` reaches ``1 + A`` where the reflected component is in phase with the
    direct one and ``1 - A`` where it is opposed, and the two extrema repeat with
    frequency period ``1/tau_cable``.
    """
    amplitude, delay = 0.25, 2.0e-7
    period = 1.0 / delay
    term = _term(amplitude, delay, 0.0)

    frequencies = np.linspace(1.0e8, 1.0e8 + 4.0 * period, 4001)
    moduli = np.array(
        [abs(_evaluate(term, frequency_hz=float(nu))[0, 0, 0]) for nu in frequencies]
    )

    assert moduli.max() == pytest.approx(1.0 + amplitude, rel=1e-5)
    assert moduli.min() == pytest.approx(1.0 - amplitude, rel=1e-5)
    assert bool(np.all(moduli <= 1.0 + amplitude + 1e-12))
    assert bool(np.all(moduli >= 1.0 - amplitude - 1e-12))

    # Periodicity: the response repeats exactly one cable period later.
    for base in (1.0e8, 1.03e8, 1.07e8):
        np.testing.assert_allclose(
            _evaluate(term, frequency_hz=base),
            _evaluate(term, frequency_hz=base + period),
            rtol=0.0,
            atol=1e-14,
        )


def test_the_delay_domain_transform_peaks_at_the_cable_delay() -> None:
    """Section 20.6's second invariant, and the reason ``Rc`` is its own term.

    A spectrum corrupted by a single-bounce reflection is ``1`` plus one complex
    sinusoid in frequency, so its discrete Fourier transform is a peak at zero
    delay plus one secondary peak at exactly ``tau_cable``.  The band here is
    chosen so that ``tau_cable`` falls on a DFT bin centre, which makes the
    expected relative amplitude exactly ``A`` rather than ``A`` reduced by
    spectral leakage -- an approximate assertion would pass for a term whose
    ripple was at the wrong delay by less than one bin.
    """
    channels = 64
    spacing_hz = 1.0e5
    resolution_s = 1.0 / (channels * spacing_hz)
    bin_index = 8
    delay = bin_index * resolution_s
    amplitude = 0.15

    term = _term(amplitude, delay, 0.35)
    frequencies = 1.0e8 + spacing_hz * np.arange(channels)
    spectrum = np.array(
        [_evaluate(term, frequency_hz=float(nu))[0, 0, 0] for nu in frequencies]
    )

    transform = np.abs(np.fft.fft(spectrum))
    delays = np.fft.fftfreq(channels, d=spacing_hz)

    assert delays[bin_index] == pytest.approx(delay, rel=1e-12)
    assert int(np.argmax(transform[1:])) + 1 == bin_index
    assert transform[bin_index] / transform[0] == pytest.approx(amplitude, rel=1e-10)

    # Nothing anywhere else: a single bounce is a single peak.
    others = np.delete(transform, [0, bin_index])
    assert float(others.max()) < 1e-9 * float(transform[0])


def test_a_die_term_returns_one_broadcast_matrix() -> None:
    """I3: ``(1, 2, 2)``, never ``n_dir`` copies of one constant."""
    term = _term(0.05, 1.0e-7)

    assert term.is_direction_dependent is False
    assert _evaluate(term).shape == (1, 2, 2)


# ---------------------------------------------------------------------------
# I2 -- declared flags are true, and each declared False has a witness
# ---------------------------------------------------------------------------


def test_rc_is_diagonal_and_never_unitary() -> None:
    """Section 20.6: a reflection changes the modulus, so it is not unitary."""
    term = _term(0.2, 1.0e-7, 0.3)

    assert term.is_diagonal() is True
    assert term.is_unitary() is False
    matrix = _evaluate(term)[0]
    assert matrix[0, 1] == 0.0
    assert matrix[1, 0] == 0.0
    assert not np.allclose(matrix @ matrix.conj().T, np.eye(2), rtol=0.0, atol=1e-12)


def test_scalarity_is_declared_exactly_when_both_feeds_share_a_cable() -> None:
    """I2 in both directions."""
    shared = _term(0.2, 1.0e-7, 0.3)
    assert shared.is_scalar() is True
    matrix = _evaluate(shared)[0]
    np.testing.assert_array_equal(matrix, matrix[0, 0] * np.eye(2, dtype=np.complex128))

    split = _term(0.2, 1.0e-7, 0.3, second=(0.05, 4.0e-8, -0.7))
    assert split.is_scalar() is False
    witness = _evaluate(split)[0]
    assert witness[0, 0] != witness[1, 1]


def test_rc_declares_frequency_dependence_only_when_a_cable_delay_is_present() -> None:
    """A reflection with zero cable delay is a constant offset, not a ripple.

    Such a configuration is legal -- it is not the identity, so R7 does not
    reject it -- and it really is frequency-flat, so claiming chromaticity for it
    would be a false ``True``.
    """
    ripple = _term(0.1, 1.0e-7, 0.0)
    assert ripple.is_frequency_dependent is True
    assert not np.array_equal(
        _evaluate(ripple, frequency_hz=1.0e8),
        _evaluate(ripple, frequency_hz=1.0e8 + 0.25e7),
    )

    flat = _term(0.1, 0.0, 0.4)
    assert flat.is_frequency_dependent is False
    np.testing.assert_array_equal(
        _evaluate(flat, frequency_hz=1.0e8), _evaluate(flat, frequency_hz=2.0e8)
    )
    assert flat.is_identity() is False

    assert ripple.is_direction_dependent is False
    assert ripple.is_time_dependent is False


def test_a_zero_amplitude_reflection_is_the_identity_the_schema_forbids() -> None:
    """The flags follow the numbers even where configuration cannot reach.

    R8 forbids ``A = 0`` outright, so this term is unconstructible from a
    document.  The flag is still computed from the resolved numbers, because a
    hard-coded ``False`` would be the vacuous claim invariant I2 exists to
    prevent.
    """
    term = _term(0.0, 1.0e-7, 0.0)

    assert term.is_identity() is True
    assert term.is_unitary() is True
    np.testing.assert_array_equal(_evaluate(term)[0], np.eye(2, dtype=np.complex128))


def test_the_term_status_is_implemented() -> None:
    """Section 31 step 5, and the ``"implemented"`` half of invariant I20."""
    assert _term().term_status == "implemented"
    assert _term().name == "Rc"


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_a_non_finite_parameter_cannot_be_constructed() -> None:
    """Caught at construction, so no ``nan`` ever reaches a chain."""
    with pytest.raises(ValueError):
        CableReflectionJones(
            amplitudes=np.array([[np.nan, 0.1]]),
            cable_delays_s=np.array([[1.0e-7, 1.0e-7]]),
            phases_rad=np.zeros((1, 2)),
        )


def test_a_reflection_amplitude_of_one_or_more_cannot_be_constructed() -> None:
    """``|A| < 1``: a reflection cannot return more power than it receives.

    Enforced at the constructor as well as at resolution (R8), because the
    constructor is reachable from library code that never sees a document.
    """
    for amplitude in (1.0, -1.0, 1.5):
        with pytest.raises(ValueError):
            CableReflectionJones(
                amplitudes=np.array([[amplitude, 0.1]]),
                cable_delays_s=np.array([[1.0e-7, 1.0e-7]]),
                phases_rad=np.zeros((1, 2)),
            )


def test_mismatched_parameter_tables_are_rejected() -> None:
    """One amplitude, delay and phase per antenna row per feed, or nothing."""
    with pytest.raises(ValueError):
        CableReflectionJones(
            amplitudes=np.array([[0.1, 0.1], [0.1, 0.1]]),
            cable_delays_s=np.array([[1.0e-7, 1.0e-7]]),
            phases_rad=np.zeros((2, 2)),
        )


def test_an_antenna_row_outside_the_array_is_rejected() -> None:
    """A row/number mix-up fails loudly rather than reading a neighbour's cable."""
    with pytest.raises(IndexError):
        _evaluate(_term(), antenna_idx=99)


# ---------------------------------------------------------------------------
# I7 and the Section 20.6 RIME invariants, end to end through the solver
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


def test_a_configured_reflection_changes_the_visibilities(tmp_path) -> None:
    """I7, made mechanical: ``Fix.md`` Section 16 rule 5."""
    baseline = _cube(tmp_path, None)
    perturbed = _cube(tmp_path, {"Rc": {"amplitude": 0.05, "cable_delay_s": 1.5e-7}})

    scale = float(np.max(np.abs(baseline)))
    assert scale > 0.0
    assert float(np.max(np.abs(perturbed - baseline))) / scale > 1e-10


def test_a_common_reflection_scales_every_correlation_by_the_squared_modulus(
    tmp_path,
) -> None:
    """The end-to-end form of ``V -> r(nu) r(nu)^* V``, channel by channel.

    With the same cable on both feeds of both antennas the term is a scalar
    ``r(nu)``, so ``V_pq(nu)`` is multiplied by ``|r(nu)|^2`` -- a real,
    frequency-dependent scaling that leaves every phase untouched.  Asserting
    the phase invariance as well as the amplitude is what distinguishes this
    from ``Kd``, whose common-mode effect is the exact opposite.
    """
    amplitude, delay, phase = 0.3, 1.5e-7, 0.4
    frequencies = np.asarray(
        solver_components_with_jones(tmp_path, None)[4], dtype=np.float64
    )

    baseline = _cube(tmp_path, None)
    rippled = _cube(
        tmp_path,
        {
            "Rc": {
                "amplitude": amplitude,
                "cable_delay_s": delay,
                "phase_rad": phase,
            }
        },
    )

    for index, frequency in enumerate(frequencies):
        angle = -2.0 * math.pi * float(frequency) * delay + phase
        response = 1.0 + amplitude * cmath.exp(1j * angle)
        np.testing.assert_allclose(
            rippled[:, :, index, :, :],
            baseline[:, :, index, :, :] * abs(response) ** 2,
            rtol=1e-12,
            atol=1e-18,
        )

    # Amplitude only: the ratio of the two cubes is real and positive
    # everywhere, so no phase moved.  Read from the ratio rather than from
    # ``np.angle`` differences, which would wrap for a visibility near the
    # branch cut and report a false failure.
    ratio = rippled / baseline
    np.testing.assert_allclose(np.imag(ratio), 0.0, rtol=0.0, atol=1e-12)
    assert bool(np.all(np.real(ratio) > 0.0))


def test_a_per_antenna_entry_beats_the_array_wide_default(tmp_path) -> None:
    """Section 22 rule 5, on the resolved numbers."""
    resolved = resolve_for(
        tmp_path,
        {
            "Rc": {
                "amplitude": 0.02,
                "cable_delay_s": 1.0e-7,
                "per_antenna": [
                    {"antenna": 1, "feed": 1, "amplitude": 0.4, "phase_rad": 0.9}
                ],
            }
        },
    )
    term = resolved.term("Rc")
    assert term is not None

    responses = term.responses_at_frequency(1.0e8)
    default = 1.0 + 0.02 * cmath.exp(-2j * math.pi * 1.0e8 * 1.0e-7)
    overridden = 1.0 + 0.4 * cmath.exp(-2j * math.pi * 1.0e8 * 1.0e-7 + 1j * 0.9)

    assert responses[0, 0] == pytest.approx(default, rel=1e-14)
    assert responses[0, 1] == pytest.approx(default, rel=1e-14)
    assert responses[1, 0] == pytest.approx(default, rel=1e-14)
    assert responses[1, 1] == pytest.approx(overridden, rel=1e-14)
