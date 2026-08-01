"""Tier 7D: the ``B`` term's physics, flags, and effect on the visibilities.

``Tier7JonesSciencePlan.md`` Section 20.2.  As for ``G``, every reference value
is the published closed form written out in the test body, never a value read
back from the production function (Section 29.1).

Invariants asserted here: **I2**, **I3**, **I7**, and Section 20.2's own
cross-term consistency check -- a real, frequency-flat bandpass is exactly a
``G`` amplitude error, and the two must produce the same cube.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.core.jones.bandpass import (
    BandpassJones,
    PolynomialBandpassResponse,
    TabulatedBandpassResponse,
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
_GRID = np.array([1.0e8, 1.01e8], dtype=np.float64)


def _empty_directions():
    from radiosim.core.jones.directions import DirectionBatch

    empty = np.zeros(4, dtype=np.float64)
    return DirectionBatch(
        alt_rad=empty,
        az_rad=empty,
        dir_l=empty,
        dir_m=empty,
        dir_n=empty,
        ra_rad=empty,
        dec_rad=empty,
        hour_angle_rad=empty,
        n_dir=4,
    )


def _evaluate(term: BandpassJones, *, frequency_hz: float, antenna_idx: int = 0):
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


def _polynomial_term(
    coefficients: tuple[complex, ...],
    *,
    reference_hz: float = 1.005e8,
    scale_hz: float = 5.0e5,
    second_feed: tuple[complex, ...] | None = None,
) -> BandpassJones:
    first = PolynomialBandpassResponse(
        coefficients=coefficients,
        reference_frequency_hz=reference_hz,
        scale_frequency_hz=scale_hz,
    )
    second = (
        first
        if second_feed is None
        else PolynomialBandpassResponse(
            coefficients=second_feed,
            reference_frequency_hz=reference_hz,
            scale_frequency_hz=scale_hz,
        )
    )
    return BandpassJones(responses=((first, second),), frequencies_hz=_GRID)


# ---------------------------------------------------------------------------
# The polynomial model, evaluated independently
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frequency", [1.0e8, 1.005e8, 1.01e8, 9.9e7])
def test_the_polynomial_matches_its_closed_form(frequency: float) -> None:
    """``b(nu) = sum_k c_k x^k``, ``x = (nu - nu_ref) / nu_scale``.

    The reference sum is written out here with an explicit power series, not
    with the production Horner evaluation, so the two agreeing is evidence
    rather than a restatement.  Includes a frequency *outside* the precomputed
    observation grid, which exercises the on-demand path.
    """
    coefficients = (1.0 + 0.0j, -0.02 + 0.01j, 0.005 + 0.0j, 0.001 - 0.002j)
    reference_hz, scale_hz = 1.005e8, 5.0e5

    normalized = (frequency - reference_hz) / scale_hz
    expected = sum(
        coefficient * normalized**order
        for order, coefficient in enumerate(coefficients)
    )

    block = _evaluate(
        _polynomial_term(coefficients, reference_hz=reference_hz, scale_hz=scale_hz),
        frequency_hz=frequency,
    )

    assert block.shape == (1, 2, 2)
    assert block[0, 0, 0] == pytest.approx(expected, rel=1e-14)
    assert block[0, 1, 1] == pytest.approx(expected, rel=1e-14)
    assert block[0, 0, 1] == 0.0
    assert block[0, 1, 0] == 0.0


def test_the_derived_reference_and_scale_span_the_band(tmp_path) -> None:
    """``nu_ref`` defaults to the band centre and ``nu_scale`` to the half-width.

    So ``x`` runs from exactly ``-1`` to exactly ``+1`` across the observed
    band, which is the whole point of normalizing: a low-order polynomial stays
    well conditioned regardless of where in the spectrum the band sits.
    """
    resolved = resolve_for(
        tmp_path,
        {"B": {"model": {"kind": "polynomial", "coefficients": [1.0, 0.5]}}},
    )
    term = resolved.term("B")
    assert term is not None

    # The fixture band is 100, 101 and 102 MHz.
    low = term.responses_at_frequency(1.0e8)[0, 0]
    middle = term.responses_at_frequency(1.01e8)[0, 0]
    high = term.responses_at_frequency(1.02e8)[0, 0]

    # x = -1 at the bottom channel, 0 at the centre and +1 at the top, so
    # b = 1 - 0.5, 1, and 1 + 0.5.
    assert low == pytest.approx(0.5, rel=1e-14)
    assert middle == pytest.approx(1.0, rel=1e-14)
    assert high == pytest.approx(1.5, rel=1e-14)


def test_a_single_channel_observation_needs_an_explicit_scale(tmp_path) -> None:
    """The derived half-bandwidth does not exist, so it is not invented.

    Silently substituting ``1`` -- or the centre frequency, or anything else --
    would make the same configuration mean different things at different band
    widths.  Exercised by resolving against a one-element band directly: the
    shipped grid schema cannot express a single channel, and the guard must
    still hold for the programmatic path that can.
    """
    from radiosim.core.jones_errors import InvalidJonesConfigError
    from radiosim.core.jones_terms import resolve_jones_terms
    from tests.unit.test_core.test_jones_resolution import simulator_for

    simulator = simulator_for(
        tmp_path,
        {"B": {"model": {"kind": "polynomial", "coefficients": [1.0, 0.5]}}},
    )

    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_jones_terms(
            simulator._resolved.jones,
            simulator._instrument_state.instrument,
            frequencies_hz=np.array([1.0e8], dtype=np.float64),
            channel_widths_hz=np.array([1.0e6], dtype=np.float64),
            time_grid=simulator._resolved.observation.time_grid,
            baseline_selection=simulator._instrument_state.selection,
            precision=simulator._precision,
        )

    assert "scale_frequency_hz explicitly" in str(caught.value)


def test_an_explicit_scale_makes_a_single_channel_band_resolvable(tmp_path) -> None:
    """The complement: naming the scale is all a one-channel band needs."""
    from radiosim.core.jones_terms import resolve_jones_terms
    from tests.unit.test_core.test_jones_resolution import simulator_for

    simulator = simulator_for(
        tmp_path,
        {
            "B": {
                "model": {
                    "kind": "polynomial",
                    "coefficients": [1.0, 0.5],
                    "reference_frequency_hz": 1.0e8,
                    "scale_frequency_hz": 1.0e6,
                }
            }
        },
    )

    resolved = resolve_jones_terms(
        simulator._resolved.jones,
        simulator._instrument_state.instrument,
        frequencies_hz=np.array([1.01e8], dtype=np.float64),
        channel_widths_hz=np.array([1.0e6], dtype=np.float64),
        time_grid=simulator._resolved.observation.time_grid,
        baseline_selection=simulator._instrument_state.selection,
        precision=simulator._precision,
    )

    term = resolved.term("B")
    assert term is not None
    # x = (101 - 100) MHz / 1 MHz = 1, so b = 1 + 0.5.
    assert term.responses_at_frequency(1.01e8)[0, 0] == pytest.approx(1.5, rel=1e-14)


# ---------------------------------------------------------------------------
# The tabulated model
# ---------------------------------------------------------------------------


def test_the_spline_passes_through_every_node() -> None:
    """Interpolation, not approximation: the nodes are reproduced exactly.

    Real and imaginary parts are splined separately (Section 20.2), so a complex
    node must come back complex and unrotated.
    """
    nodes = (9.0e7, 1.0e8, 1.1e8, 1.2e8, 1.3e8)
    gains = (0.90 + 0.00j, 1.00 + 0.05j, 0.98 - 0.02j, 0.93 + 0.01j, 0.88 + 0.0j)
    response = TabulatedBandpassResponse(node_frequencies_hz=nodes, gains=gains)

    values = response.evaluate(np.asarray(nodes))

    np.testing.assert_allclose(values, np.asarray(gains), rtol=1e-13, atol=1e-15)


def test_the_spline_reproduces_a_cubic_exactly() -> None:
    """A cubic sampled at five nodes must be recovered between them.

    This is the sharpest available statement that the interpolation is a
    genuine natural cubic and not, say, a linear fallback that happens to hit
    the nodes: a linear interpolant of a cubic is wrong everywhere between them.
    """

    def cubic(frequency: np.ndarray) -> np.ndarray:
        scaled = (frequency - 1.1e8) / 1.0e7
        return 1.0 + 0.1 * scaled - 0.05 * scaled**2 + 0.02 * scaled**3

    nodes = np.array([9.0e7, 1.0e8, 1.1e8, 1.2e8, 1.3e8])
    response = TabulatedBandpassResponse(
        node_frequencies_hz=tuple(float(node) for node in nodes),
        gains=tuple(complex(value) for value in cubic(nodes)),
    )

    probes = np.array([9.5e7, 1.05e8, 1.15e8, 1.25e8])

    np.testing.assert_allclose(
        response.evaluate(probes).real, cubic(probes), rtol=1e-10, atol=0.0
    )


def test_a_tabulated_response_reports_the_span_it_is_defined_over() -> None:
    """The span R11 checks the observation against."""
    response = TabulatedBandpassResponse(
        node_frequencies_hz=(9.0e7, 1.0e8, 1.1e8, 1.2e8),
        gains=(1.0 + 0j,) * 4,
    )

    assert response.frequency_span_hz == (9.0e7, 1.2e8)


# ---------------------------------------------------------------------------
# I2 and I3
# ---------------------------------------------------------------------------


def test_b_is_always_diagonal() -> None:
    """``is_diagonal`` is ``True``, off-diagonals exactly zero at every channel."""
    term = _polynomial_term((1.0 + 0j, -0.3 + 0.2j), second_feed=(0.8 + 0j, 0.1 + 0j))

    assert term.is_diagonal() is True
    for frequency in (1.0e8, 1.005e8, 1.01e8):
        block = _evaluate(term, frequency_hz=frequency)
        assert block[0, 0, 1] == 0.0
        assert block[0, 1, 0] == 0.0


def test_scalarity_is_declared_exactly_when_every_feed_shares_a_response() -> None:
    """I2 both ways: a declared ``True`` is checked, a declared ``False`` witnessed."""
    shared = _polynomial_term((1.0 + 0j, -0.1 + 0j))
    assert shared.is_scalar() is True
    block = _evaluate(shared, frequency_hz=1.0e8)
    np.testing.assert_array_equal(
        block[0], block[0, 0, 0] * np.eye(2, dtype=np.complex128)
    )

    split = _polynomial_term((1.0 + 0j, -0.1 + 0j), second_feed=(0.7 + 0j,))
    assert split.is_scalar() is False
    witness = _evaluate(split, frequency_hz=1.0e8)
    assert witness[0, 0, 0] != witness[0, 1, 1]


def test_unitarity_is_declared_only_for_a_pure_phase_bandpass() -> None:
    """A bandpass is an attenuation profile, so this is ``False`` in practice.

    The check is on the resolved numbers and not on the model kind, which is why
    a deliberately constructed unit-modulus bandpass can be ``True``.
    """
    attenuating = _polynomial_term((0.9 + 0j,))
    assert attenuating.is_unitary() is False
    witness = _evaluate(attenuating, frequency_hz=1.0e8)[0]
    assert not np.allclose(witness @ witness.conj().T, np.eye(2), atol=1e-12)

    phase_only = _polynomial_term((complex(np.cos(0.4), np.sin(0.4)),))
    assert phase_only.is_unitary() is True
    matrix = _evaluate(phase_only, frequency_hz=1.0e8)[0]
    np.testing.assert_allclose(
        matrix @ matrix.conj().T, np.eye(2), rtol=0.0, atol=1e-15
    )


def test_b_declares_the_dependencies_it_actually_has() -> None:
    """Frequency-dependent; direction- and time-independent."""
    term = _polynomial_term((1.0 + 0j, -0.4 + 0j))

    assert term.is_direction_dependent is False
    assert term.is_time_dependent is False
    assert term.is_frequency_dependent is True
    assert not np.array_equal(
        _evaluate(term, frequency_hz=1.0e8), _evaluate(term, frequency_hz=1.01e8)
    )


def test_a_die_term_returns_one_broadcast_matrix() -> None:
    """I3."""
    block = _evaluate(_polynomial_term((0.9 + 0j,)), frequency_hz=1.0e8)

    assert block.shape == (1, 2, 2)


def test_the_term_status_is_implemented() -> None:
    """Section 31 step 5."""
    term = _polynomial_term((0.9 + 0j,))

    assert term.term_status == "implemented"
    assert term.name == "B"


def test_the_lookup_is_keyed_on_the_physical_frequency() -> None:
    """Not on ``freq_idx``: an index is only meaningful against its own grid.

    A term that trusted the index would silently return the wrong channel the
    first time it was evaluated against a different frequency array.
    """
    term = _polynomial_term((1.0 + 0j, 0.5 + 0j))

    on_grid = _evaluate(term, frequency_hz=1.01e8)
    off_grid = _evaluate(term, frequency_hz=1.02e8)

    assert on_grid[0, 0, 0] != off_grid[0, 0, 0]


def test_an_antenna_row_outside_the_array_is_rejected() -> None:
    """A row/number mix-up fails loudly."""
    with pytest.raises(IndexError):
        _evaluate(_polynomial_term((0.9 + 0j,)), frequency_hz=1.0e8, antenna_idx=5)


# ---------------------------------------------------------------------------
# I7 and the Section 20.2 cross-term consistency check, through the solver
# ---------------------------------------------------------------------------


def _cube(tmp_path, jones: dict[str, Any] | None) -> np.ndarray:
    instrument, beam_system, receptors, jones_terms, frequencies = (
        solver_components_with_jones(tmp_path, jones)
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


def test_a_configured_bandpass_changes_the_visibilities(tmp_path) -> None:
    """I7."""
    baseline = _cube(tmp_path, None)
    perturbed = _cube(
        tmp_path,
        {"B": {"model": {"kind": "polynomial", "coefficients": [1.0, 0.0, -0.05]}}},
    )

    scale = float(np.max(np.abs(baseline)))
    assert scale > 0.0
    assert float(np.max(np.abs(perturbed - baseline))) / scale > 1e-10


def test_a_bandpass_shape_multiplies_channel_by_channel(tmp_path) -> None:
    """The defining property: ``B`` is frequency structure and nothing else.

    A real bandpass ``b(nu)`` on both antennas takes ``V(nu)`` to
    ``b(nu)^2 V(nu)``, independently at each channel.  The two channels are
    scaled by *different* factors here, which is what distinguishes ``B`` from
    the frequency-flat ``G``.
    """
    coefficients = [1.0, 0.2]
    baseline = _cube(tmp_path, None)
    shaped = _cube(
        tmp_path,
        {"B": {"model": {"kind": "polynomial", "coefficients": coefficients}}},
    )

    # x = -1, 0, +1 across the fixture's three channels (band-centre reference,
    # half-bandwidth scale), so b = 0.8, 1.0 and 1.2.
    expected = np.array([0.8, 1.0, 1.2]) ** 2
    np.testing.assert_allclose(
        shaped, baseline * expected[None, None, :, None, None], rtol=1e-13, atol=0.0
    )
    assert len({float(value) for value in expected}) == 3


def test_a_flat_real_bandpass_equals_a_gain_amplitude_error(tmp_path) -> None:
    """Section 20.2's cross-term consistency check.

    A real, frequency-flat bandpass *is* a ``G`` amplitude error.  The two stay
    separate terms because one is defined to carry frequency structure and the
    other is not -- but where they overlap they must agree exactly, and this is
    the only assertion in the suite that ties the two terms' arithmetic
    together.
    """
    flat = _cube(
        tmp_path, {"B": {"model": {"kind": "polynomial", "coefficients": [1.25]}}}
    )
    equivalent_gain = _cube(tmp_path, {"G": {"amplitude_error": 0.25}})

    np.testing.assert_allclose(flat, equivalent_gain, rtol=1e-13, atol=0.0)


def test_g_and_b_commute(tmp_path) -> None:
    """Section 20.1: ``G`` commutes with ``B``.

    Both are diagonal in the same basis, so the composed chain is the same
    whichever canonical slot they occupy -- which is why Section 20.12 is free
    to fix their mutual order by convention.  Checked as a product identity:
    enabling both must equal the product of enabling each alone, relative to the
    unperturbed cube.
    """
    baseline = _cube(tmp_path, None)
    gain_only = _cube(tmp_path, {"G": {"amplitude_error": 0.3}})
    bandpass_only = _cube(
        tmp_path, {"B": {"model": {"kind": "polynomial", "coefficients": [1.0, 0.2]}}}
    )
    both = _cube(
        tmp_path,
        {
            "G": {"amplitude_error": 0.3},
            "B": {"model": {"kind": "polynomial", "coefficients": [1.0, 0.2]}},
        },
    )

    nonzero = np.abs(baseline) > 0.0
    predicted = np.zeros_like(baseline)
    predicted[nonzero] = gain_only[nonzero] * bandpass_only[nonzero] / baseline[nonzero]

    np.testing.assert_allclose(both[nonzero], predicted[nonzero], rtol=1e-12, atol=0.0)
