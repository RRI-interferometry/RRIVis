"""Tier 7D: the ``G`` term's physics, flags, and effect on the visibilities.

``Tier7JonesSciencePlan.md`` Section 20.1, with the reference values written out
in the test bodies from the published closed form rather than read back from the
production function (Section 29.1).  A cross-check that calls the code under
test is a tautology, and Section 29.1 says so because that is the single most
common way a validation suite becomes worthless.

Invariants asserted here: **I2** (declared flags are true, and each declared
``False`` has a witness), **I3** (a DIE term returns ``(1, 2, 2)``), and **I7**
(a configured term changes the visibilities).
"""

from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.core.jones.gain import GainJones, ResolvedGainTimeModel
from radiosim.core.jones_errors import JonesEvaluationError
from radiosim.core.visibility import calculate_visibility
from tests.characterization.test_tier6_current_behavior import (
    WORKLOAD_LOCATION,
    WORKLOAD_TIME_GRID,
    _workload_point_sources,
)
from tests.unit.test_core.test_jones_resolution import (
    resolve_for,
    solver_components_with_jones,
    three_antenna_layout,
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


def _evaluate(term: GainJones, *, antenna_idx: int = 0, time_mjd: float = 60000.0):
    return np.asarray(
        term.compute_jones_batch(
            antenna_idx=antenna_idx,
            directions=_empty_directions(),
            frequency_hz=1.0e8,
            freq_idx=0,
            time_mjd=time_mjd,
            time_idx=0,
            backend=_BACKEND,
            dtype=np.complex128,
        )
    )


def _term(
    *,
    amplitude: float = 0.0,
    phase: float = 0.0,
    second_amplitude: float | None = None,
    second_phase: float | None = None,
    time_model: ResolvedGainTimeModel | None = None,
    elevation_gain: float = 1.0,
    reference_time_mjd: float = 60000.0,
) -> GainJones:
    """Build one two-row ``G`` from explicit values, bypassing configuration."""
    feed_0 = (1.0 + amplitude) * cmath.exp(1j * phase)
    feed_1 = (1.0 + (amplitude if second_amplitude is None else second_amplitude)) * (
        cmath.exp(1j * (phase if second_phase is None else second_phase))
    )
    return GainJones(
        base_gains=np.array([[feed_0, feed_1], [feed_0, feed_1]], dtype=np.complex128),
        time_model=time_model or ResolvedGainTimeModel(kind="constant"),
        reference_time_mjd=reference_time_mjd,
        elevation_gain=elevation_gain,
    )


# ---------------------------------------------------------------------------
# The closed form (Section 20.1), evaluated independently
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("amplitude", "phase"),
    [(0.0, 0.3), (0.02, 0.0), (-0.15, -1.2), (0.5, math.pi), (0.05, 2.0 * math.pi)],
)
def test_the_matrix_is_the_published_diagonal_form(
    amplitude: float,
    phase: float,
) -> None:
    """``G = diag((1 + a) exp(i phi), ...)`` at machine precision.

    ``(1 + a) exp(i phi)`` is written out here from Hamaker, Bregman & Sault
    (1996) via Section 20.1; nothing in this assertion calls the term's own
    arithmetic.
    """
    expected = (1.0 + amplitude) * complex(math.cos(phase), math.sin(phase))

    block = _evaluate(_term(amplitude=amplitude, phase=phase))

    assert block.shape == (1, 2, 2)
    assert block[0, 0, 0] == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert block[0, 1, 1] == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert block[0, 0, 1] == 0.0
    assert block[0, 1, 0] == 0.0


def test_a_die_term_returns_one_broadcast_matrix(tmp_path) -> None:
    """I3: ``(1, 2, 2)``, never ``n_dir`` copies of one constant.

    Materialising a copy per direction would multiply the chain's memory by the
    direction count for no reason -- and on a HEALPix sky the direction count is
    the pixel count.
    """
    term = _term(amplitude=0.1)

    block = _evaluate(term)

    assert term.is_direction_dependent is False
    assert block.shape == (1, 2, 2)


# ---------------------------------------------------------------------------
# The three time models, evaluated independently
# ---------------------------------------------------------------------------


def test_the_constant_time_model_does_not_move() -> None:
    """``s(t) = 1``, at ``t0`` and eight hours later."""
    term = _term(amplitude=0.2, reference_time_mjd=60000.0)

    at_start = _evaluate(term, time_mjd=60000.0)
    at_eight_hours = _evaluate(term, time_mjd=60000.0 + 8.0 / 24.0)

    np.testing.assert_array_equal(at_start, at_eight_hours)


@pytest.mark.parametrize("hours", [0.0, 0.5, 3.0, -2.0])
def test_the_linear_drift_matches_its_closed_form(hours: float) -> None:
    """``s(t) = 1 + rate * dt``, with ``dt`` in hours from the first sample."""
    rate = 0.03
    amplitude = 0.1
    reference_mjd = 60000.0
    term = _term(
        amplitude=amplitude,
        time_model=ResolvedGainTimeModel(kind="linear_drift", rate_per_hour=rate),
        reference_time_mjd=reference_mjd,
    )

    # ``dt`` is recomputed from the two MJD values rather than reused from
    # ``hours``: a double holds MJD 60000 to about 1.3 microseconds, so the
    # round trip through an absolute date costs roughly 1e-11 hours.  That is
    # utterly irrelevant to a gain drift and would be a false failure at the
    # 1e-14 tolerance the closed form itself deserves.
    time_mjd = reference_mjd + hours / 24.0
    elapsed_hours = (time_mjd - reference_mjd) * 24.0
    expected = (1.0 + amplitude) * (1.0 + rate * elapsed_hours)

    block = _evaluate(term, time_mjd=time_mjd)

    assert block[0, 0, 0].real == pytest.approx(expected, rel=1e-14)
    assert block[0, 0, 0].imag == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("hours", [0.0, 0.25, 1.0, 2.5])
def test_the_sinusoid_matches_its_closed_form(hours: float) -> None:
    """``s(t) = 1 + depth sin(2 pi dt / period + phase)``."""
    depth, period, phase = 0.2, 3.0, 0.7
    reference_mjd = 60000.0
    term = _term(
        amplitude=0.0,
        time_model=ResolvedGainTimeModel(
            kind="sinusoidal", depth=depth, period_hours=period, phase_rad=phase
        ),
        reference_time_mjd=reference_mjd,
    )

    time_mjd = reference_mjd + hours / 24.0
    elapsed_hours = (time_mjd - reference_mjd) * 24.0
    expected = 1.0 + depth * math.sin(2.0 * math.pi * elapsed_hours / period + phase)

    block = _evaluate(term, time_mjd=time_mjd)

    assert block[0, 0, 0].real == pytest.approx(expected, rel=1e-14)


def test_every_time_model_is_reproducible_from_configuration() -> None:
    """Section 20.1: none of the three draws a random number.

    Two independently constructed terms with the same parameters agree bit for
    bit at the same time, which is what "exactly reproducible from
    configuration" has to mean for a forward model whose output is fingerprinted.
    """
    for model in (
        ResolvedGainTimeModel(kind="constant"),
        ResolvedGainTimeModel(kind="linear_drift", rate_per_hour=0.02),
        ResolvedGainTimeModel(
            kind="sinusoidal", depth=0.1, period_hours=1.5, phase_rad=0.4
        ),
    ):
        first = _term(amplitude=0.05, time_model=model)
        second = _term(amplitude=0.05, time_model=model)
        np.testing.assert_array_equal(
            _evaluate(first, time_mjd=60000.4),
            _evaluate(second, time_mjd=60000.4),
        )


# ---------------------------------------------------------------------------
# The elevation gain curve
# ---------------------------------------------------------------------------


def test_the_elevation_curve_is_evaluated_at_the_pointing_elevation(
    tmp_path,
) -> None:
    """``g_el(el) = sum_k c_k el^k`` at the zenith-drift pointing elevation.

    RadioSim's one phase convention is zenith drift, so ``el_ref`` is exactly
    90 degrees and the curve resolves to a single constant.  It is a real,
    non-identity gain and it does not vary -- and it will not until RadioSim
    gains a steerable phase centre.  The test pins the number so that the day
    the pointing becomes steerable, this fails rather than quietly changing.
    """
    coefficients = [1.0, -1.0e-4]
    expected = coefficients[0] + coefficients[1] * 90.0

    resolved = resolve_for(
        tmp_path, {"G": {"amplitude_error": 0.0, "elevation_curve": coefficients}}
    )
    term = resolved.term("G")
    assert term is not None

    assert term.gains_at_time(60000.0)[0, 0] == pytest.approx(expected, rel=1e-15)
    assert term.is_time_dependent is False


# ---------------------------------------------------------------------------
# I2 -- declared flags are true, and each declared False has a witness
# ---------------------------------------------------------------------------


def test_g_is_always_diagonal() -> None:
    """``is_diagonal`` is ``True``, and the off-diagonals are *exactly* zero."""
    for amplitude, phase in ((0.0, 0.0), (0.4, 1.1), (-0.3, -2.0)):
        term = _term(amplitude=amplitude, phase=phase, second_amplitude=0.9)
        assert term.is_diagonal() is True
        block = _evaluate(term)
        assert block[0, 0, 1] == 0.0
        assert block[0, 1, 0] == 0.0


def test_scalarity_is_declared_exactly_when_the_two_feeds_agree() -> None:
    """Both directions of I2's requirement, so a vacuous ``True`` is impossible."""
    equal_feeds = _term(amplitude=0.2, phase=0.5)
    assert equal_feeds.is_scalar() is True
    block = _evaluate(equal_feeds)
    np.testing.assert_array_equal(
        block[0], block[0, 0, 0] * np.eye(2, dtype=np.complex128)
    )

    unequal_feeds = _term(amplitude=0.2, phase=0.5, second_amplitude=0.35)
    assert unequal_feeds.is_scalar() is False
    witness = _evaluate(unequal_feeds)
    assert witness[0, 0, 0] != witness[0, 1, 1]


def test_unitarity_is_declared_exactly_when_every_gain_has_unit_modulus() -> None:
    """Section 20.1: ``G`` is **not** unitary unless every ``a`` is zero.

    Three ways of breaking it are checked -- an amplitude error, a drifting time
    model, and an elevation gain -- because a term that attenuates cannot
    preserve power however the attenuation was spelled.
    """
    pure_phase = _term(amplitude=0.0, phase=0.9, second_phase=-0.4)
    assert pure_phase.is_unitary() is True
    matrix = _evaluate(pure_phase)[0]
    np.testing.assert_allclose(
        matrix @ matrix.conj().T, np.eye(2), rtol=0.0, atol=1e-15
    )

    for broken in (
        _term(amplitude=0.02, phase=0.9),
        _term(
            amplitude=0.0,
            phase=0.9,
            time_model=ResolvedGainTimeModel(kind="linear_drift", rate_per_hour=0.1),
        ),
        _term(amplitude=0.0, phase=0.9, elevation_gain=0.8),
    ):
        assert broken.is_unitary() is False
        witness = _evaluate(broken, time_mjd=60000.5)[0]
        product = witness @ witness.conj().T
        assert not np.allclose(product, np.eye(2), rtol=0.0, atol=1e-12)


def test_g_declares_the_dependencies_it_actually_has() -> None:
    """Direction- and frequency-independent; time-dependent only when it moves."""
    static = _term(amplitude=0.1)
    assert static.is_direction_dependent is False
    assert static.is_frequency_dependent is False
    assert static.is_time_dependent is False
    np.testing.assert_array_equal(
        _evaluate(static, time_mjd=60000.0), _evaluate(static, time_mjd=60001.0)
    )

    drifting = _term(
        amplitude=0.1,
        time_model=ResolvedGainTimeModel(kind="linear_drift", rate_per_hour=0.1),
    )
    assert drifting.is_time_dependent is True
    assert not np.array_equal(
        _evaluate(drifting, time_mjd=60000.0), _evaluate(drifting, time_mjd=60001.0)
    )


def test_the_term_status_is_implemented() -> None:
    """Section 31 step 5, and the ``"implemented"`` half of invariant I20."""
    assert _term(amplitude=0.1).term_status == "implemented"
    assert _term(amplitude=0.1).name == "G"


# ---------------------------------------------------------------------------
# Construction and evaluation guards
# ---------------------------------------------------------------------------


def test_a_non_finite_gain_cannot_be_constructed() -> None:
    """Caught at construction, so no ``nan`` ever reaches a chain."""
    with pytest.raises(ValueError):
        GainJones(
            base_gains=np.array([[np.nan + 0j, 1 + 0j]], dtype=np.complex128),
            time_model=ResolvedGainTimeModel(kind="constant"),
            reference_time_mjd=60000.0,
        )


def test_a_time_model_that_produces_a_non_finite_gain_is_attributed_to_g() -> None:
    """Section 26: a ``nan`` surfaces at the term, not as a ``nan`` in the cube.

    A drift rate large enough to overflow at a distant time is not reachable
    from any sane configuration, which is exactly why the check exists: if it
    ever does happen, the failure must name the term.
    """
    term = GainJones(
        base_gains=np.array([[1e300 + 0j, 1e300 + 0j]], dtype=np.complex128),
        time_model=ResolvedGainTimeModel(kind="linear_drift", rate_per_hour=1e300),
        reference_time_mjd=0.0,
    )

    with pytest.raises(JonesEvaluationError) as caught:
        _evaluate(term, time_mjd=1.0e6)

    assert "'G'" in str(caught.value)


def test_an_antenna_row_outside_the_array_is_rejected() -> None:
    """A row/number mix-up fails loudly rather than reading a neighbour's gain."""
    with pytest.raises(IndexError):
        _evaluate(_term(amplitude=0.1), antenna_idx=99)


# ---------------------------------------------------------------------------
# I7 and the Section 20.1 RIME invariants, end to end through the solver
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


def test_a_configured_gain_changes_the_visibilities(tmp_path) -> None:
    """I7, made mechanical: ``Fix.md`` Section 16 rule 5.

    A term that is configured and does not change the output is the defect this
    tier exists to remove, so the threshold is asserted rather than eyeballed.
    """
    baseline = _cube(tmp_path, None)
    perturbed = _cube(tmp_path, {"G": {"amplitude_error": 0.02}})

    scale = float(np.max(np.abs(baseline)))
    assert scale > 0.0
    relative = float(np.max(np.abs(perturbed - baseline))) / scale
    assert relative > 1e-10


def test_a_common_amplitude_error_scales_every_correlation_by_one_plus_a_squared(
    tmp_path,
) -> None:
    """Section 20.1's own invariant, at machine precision.

    ``V_pq = J_p C J_q^H``, so a real scalar gain ``(1 + a)`` on both feeds of
    both antennas takes ``V`` to ``(1 + a)^2 V`` exactly.  This is the sharpest
    available statement that ``G`` entered the RIME on both sides of the
    coherency and not on one.
    """
    amplitude = 0.25
    baseline = _cube(tmp_path, None)
    scaled = _cube(tmp_path, {"G": {"amplitude_error": amplitude}})

    np.testing.assert_allclose(
        scaled, baseline * (1.0 + amplitude) ** 2, rtol=1e-13, atol=0.0
    )


def test_configured_linear_feed_values_select_east_x_and_north_y(tmp_path) -> None:
    """SCI-006: native feed 0 is east-X and feed 1 is north-Y.

    The asymmetric gains are resolved through the public ``jones.G``
    configuration path, then observed through the full ``H ... G ... C``
    chain.  Applying the same pair to every antenna gives an analytic factor
    for each reported product and removes baseline-dependent bookkeeping from
    the oracle.
    """
    east_x_gain = 1.25
    north_y_gain = 0.8
    per_antenna = [
        {
            "antenna": antenna,
            "feed": feed,
            "amplitude_error": gain - 1.0,
        }
        for antenna in (0, 1)
        for feed, gain in ((0, east_x_gain), (1, north_y_gain))
    ]
    baseline = _cube(tmp_path, None)
    asymmetric = _cube(
        tmp_path,
        {"G": {"amplitude_error": 0.0, "per_antenna": per_antenna}},
    )

    np.testing.assert_allclose(
        asymmetric[..., 0, 0],
        baseline[..., 0, 0] * east_x_gain**2,
        rtol=1e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        asymmetric[..., 0, 1],
        baseline[..., 0, 1] * east_x_gain * north_y_gain,
        rtol=1e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        asymmetric[..., 1, 0],
        baseline[..., 1, 0] * east_x_gain * north_y_gain,
        rtol=1e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        asymmetric[..., 1, 1],
        baseline[..., 1, 1] * north_y_gain**2,
        rtol=1e-13,
        atol=0.0,
    )


def test_a_common_phase_error_leaves_every_correlation_amplitude_unchanged(
    tmp_path,
) -> None:
    """Section 20.1: a pure phase error changes only phases.

    A *common* phase cancels between ``J_p`` and ``J_q^H`` on every baseline, so
    the visibilities are unchanged outright -- which is also the statement that
    ``G`` is applied as a voltage gain and conjugated on the second antenna.
    """
    baseline = _cube(tmp_path, None)
    rotated = _cube(tmp_path, {"G": {"phase_error_rad": 0.7}})

    np.testing.assert_allclose(rotated, baseline, rtol=1e-13, atol=0.0)


def test_a_per_antenna_phase_error_changes_phase_and_not_amplitude(
    tmp_path,
) -> None:
    """The same invariant where the phase does *not* cancel.

    With a phase on one antenna only, every correlation amplitude is preserved
    exactly and the phases move -- the two halves of "a pure phase error leaves
    all four correlation amplitudes unchanged and changes only phases".
    """
    baseline = _cube(tmp_path, None)
    rotated = _cube(
        tmp_path,
        {
            "G": {
                "phase_error_rad": 0.0,
                "per_antenna": [
                    {"antenna": 0, "feed": 0, "phase_error_rad": 0.9},
                    {"antenna": 0, "feed": 1, "phase_error_rad": 0.9},
                ],
            }
        },
    )

    np.testing.assert_allclose(np.abs(rotated), np.abs(baseline), rtol=1e-13, atol=0.0)
    scale = float(np.max(np.abs(baseline)))
    assert float(np.max(np.abs(rotated - baseline))) / scale > 1e-6


def test_a_gain_leaves_the_closure_phase_invariant(tmp_path) -> None:
    """The ``G`` half of invariant I11, which Tier 7H completes with ``M``.

    Antenna-based gains cancel around a closed triangle by construction.  This
    is asserted now rather than deferred because it is the property that
    distinguishes a *Jones* term from a baseline-dependent one, and getting it
    wrong here would be invisible until ``M`` arrived to be compared against.
    """
    from radiosim.core.instrument_adapters import SolverInstrumentView

    jones = {
        "G": {
            "phase_error_rad": 0.0,
            "per_antenna": [
                {"antenna": 0, "feed": 0, "phase_error_rad": 0.31},
                {"antenna": 0, "feed": 1, "phase_error_rad": 0.31},
                {"antenna": 1, "feed": 0, "phase_error_rad": -0.77},
                {"antenna": 1, "feed": 1, "phase_error_rad": -0.77},
                {"antenna": 2, "feed": 0, "phase_error_rad": 1.42},
                {"antenna": 2, "feed": 1, "phase_error_rad": 1.42},
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
    # And the visibilities themselves really did move, so the invariance above
    # is not the invariance of nothing having happened.
    scale = float(np.max(np.abs(baseline)))
    assert float(np.max(np.abs(perturbed - baseline))) / scale > 1e-6
