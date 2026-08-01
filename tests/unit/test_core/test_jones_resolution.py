"""Tier 7D: ``resolve_jones_terms`` precedence, failure ordering, and rejections.

``Tier7JonesSciencePlan.md`` Sections 22, 24 and 26.1.  Every rejection this
slice owns is asserted by its **exact** message string, following the Tier 5
precedent that made every receptor rejection reproducible by hand: a test that
matches a substring cannot tell a helpful message from a degraded one.

The rejections that belong to a later term (R8-R10, R12-R16) are not here.  They
are not skipped or marked expected-failure either -- they simply do not exist
yet, and a placeholder asserting that a message is absent would be a test of
nothing.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np
import pytest

from radiosim.api.simulator import Simulator
from radiosim.core.jones_errors import (
    IdentityJonesTermError,
    InvalidJonesConfigError,
    JonesAssignmentError,
)
from radiosim.core.jones_terms import (
    CANONICAL_CHAIN_ORDER,
    EMPTY_JONES_TERMS,
    ResolvedJonesTerms,
    resolve_jones_terms,
)
from tests.fixtures.configs import valid_config_mapping

#: A ``G`` that is not the identity: a 2% amplitude error on every feed.
NONTRIVIAL_GAIN: dict[str, Any] = {"amplitude_error": 0.02}

#: A ``B`` that is not the identity: a mild quadratic roll-off across the band.
NONTRIVIAL_BANDPASS: dict[str, Any] = {
    "model": {"kind": "polynomial", "coefficients": [1.0, 0.0, -0.05]}
}

#: Tier 7E's four terms, each at a value that is not the identity.  Kept
#: together so that a test needing "some legal block for term X" never invents
#: its own and drifts from the schema.
NONTRIVIAL_LEAKAGE: dict[str, Any] = {
    "d_terms": {"kind": "explicit", "d0": [0.02, 0.0], "d1": [0.0, 0.02]}
}
NONTRIVIAL_CROSSHAND: dict[str, Any] = {"phase_rad": 0.35}
NONTRIVIAL_DELAY: dict[str, Any] = {"delay_s": 2.0e-9}
NONTRIVIAL_REFLECTION: dict[str, Any] = {"amplitude": 0.02, "cable_delay_s": 1.5e-7}

#: Every Tier 7E term whose ``per_antenna`` entries are keyed by
#: ``(antenna, feed)``.  ``X`` is deliberately absent: its parameter is the
#: *relative* phase between the two feeds, which is one number per antenna, so
#: its overrides carry no feed index at all (Section 20.4).
FEED_KEYED_7E_TERMS: dict[str, dict[str, Any]] = {
    "D": NONTRIVIAL_LEAKAGE,
    "Kd": NONTRIVIAL_DELAY,
    "Rc": NONTRIVIAL_REFLECTION,
}


def simulator_for(
    tmp_path,
    jones: dict[str, Any] | None = None,
    **section_overrides: Any,
) -> Simulator:
    """Return a Simulator resolved as far as its instrument and receptors.

    Stops short of the beam load deliberately: Section 26.1 requires every
    ``jones:`` rejection to be raised before the first side effect, and a helper
    that loaded a beam first would make that property untestable.
    """
    data = valid_config_mapping(tmp_path, **section_overrides)
    if jones is not None:
        data["jones"] = jones
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    return simulator


def resolve_for(tmp_path, jones: dict[str, Any] | None) -> ResolvedJonesTerms:
    """Resolve one ``jones:`` block against the standard fixture instrument."""
    simulator = simulator_for(tmp_path, jones)
    return resolve_jones_terms(
        simulator._resolved.jones,
        simulator._instrument_state.instrument,
        frequencies_hz=simulator._resolved.frequency.channel_frequencies_hz,
        time_grid=simulator._resolved.observation.time_grid,
        precision=simulator._precision,
    )


def three_antenna_layout(tmp_path) -> dict[str, Any]:
    """Return an ``instrument`` override giving a closed antenna triangle.

    The shipped fixture has two antennas and therefore one baseline, which
    cannot express a closure phase.  Written here rather than added to the
    shared fixture so that no other test's instrument changes underneath it.
    """
    layout = tmp_path / "triangle.txt"
    layout.write_text(
        "Name Number BeamID E N U Diameter\n"
        "ANT0 0 0 0.0 0.0 0.0 14.0\n"
        "ANT1 1 0 14.0 0.0 0.0 14.0\n"
        "ANT2 2 0 7.0 12.0 0.0 14.0\n"
    )
    return {"instrument": {"source": {"path": str(layout)}}}


def solver_components_with_jones(
    tmp_path,
    jones: dict[str, Any] | None,
    **section_overrides: Any,
):
    """Return the pieces of one solver call: view, beams, receptors, terms, band.

    The one place the low-level Jones tests build a solver workload, so a term
    is always exercised against the same instrument, beams and receptors that
    the shipped fixture configuration produces.

    The channel frequencies are returned too, and this is load-bearing rather
    than convenient: ``resolve_jones_terms`` derives the bandpass reference and
    scale frequencies -- and checks R11 -- against the *configured* band, so a
    solver call made with some other frequency array would silently normalize
    the polynomial against a band the run does not have.  Production cannot do
    that (``Simulator`` passes its own resolved grid to both), and a test helper
    must not either.
    """
    from radiosim.core.instrument_adapters import SolverInstrumentView

    simulator = simulator_for(tmp_path, jones, **section_overrides)
    simulator._ensure_jones_terms()
    simulator._ensure_beam_system()
    return (
        SolverInstrumentView.from_state(simulator._instrument_state),
        simulator.beam_system,
        simulator.receptors,
        simulator.jones_terms,
        np.asarray(
            simulator._resolved.frequency.channel_frequencies_hz, dtype=np.float64
        ),
    )


# ---------------------------------------------------------------------------
# Absence, emptiness, and the shape of a resolved inventory
# ---------------------------------------------------------------------------


def test_an_absent_section_resolves_to_the_empty_inventory(tmp_path) -> None:
    """The property invariant I1 rests on: absence changes nothing at all."""
    resolved = resolve_for(tmp_path, None)

    assert resolved is EMPTY_JONES_TERMS
    assert resolved.is_empty
    assert resolved.chain_terms == ()
    assert resolved.baseline_terms == ()
    assert resolved.to_snapshot() == {}
    assert resolved.provenance.enabled_terms == ()


def test_a_present_but_empty_section_is_rejected_with_the_r2_message(
    tmp_path,
) -> None:
    """R2, verbatim.

    An empty section is a statement of intent the document does not carry out.
    Treating it as absence would silently hide a deleted term or a mis-indented
    key, which is the quiet-nothing failure mode this whole tier removes.
    """
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(tmp_path, {})

    assert str(caught.value) == (
        "jones: is present but configures no term; remove the section or "
        "configure at least one term."
    )


def test_a_resolved_inventory_is_ordered_canonically_not_as_written(
    tmp_path,
) -> None:
    """Section 22 rule 4: chain shape depends on which terms, never on file order.

    Both key orders are written out explicitly, because "the dict happened to
    iterate the same way" is exactly what this rule exists to stop relying on.
    """
    written_g_first = resolve_for(
        tmp_path, {"G": NONTRIVIAL_GAIN, "B": NONTRIVIAL_BANDPASS}
    )
    written_b_first = resolve_for(
        tmp_path, {"B": NONTRIVIAL_BANDPASS, "G": NONTRIVIAL_GAIN}
    )

    assert written_g_first.configured_letters == ("G", "B")
    assert written_b_first.configured_letters == ("G", "B")
    assert (
        written_g_first.provenance.jones_sha256
        == written_b_first.provenance.jones_sha256
    )
    # And the order really is the canonical one, not alphabetical.
    positions = [CANONICAL_CHAIN_ORDER.index(letter) for letter in ("G", "B")]
    assert positions == sorted(positions)


def test_the_solver_owned_terms_are_recorded_but_never_constructed_here(
    tmp_path,
) -> None:
    """``H``, ``C`` and ``E`` are in the record and not in ``chain_terms``.

    ``E`` cannot be built at setup at all -- it closes over the directions,
    frequency and time of the step it is evaluated at -- so the record is where
    "what was actually applied" lives, and ``chain_terms`` is only what the
    configuration added.
    """
    resolved = resolve_for(tmp_path, {"G": NONTRIVIAL_GAIN})

    assert resolved.configured_letters == ("G",)
    assert resolved.provenance.chain_order == ("H", "G", "C", "E")
    assert resolved.provenance.enabled_terms == resolved.provenance.chain_order


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------


def test_an_explicit_per_antenna_entry_beats_the_array_wide_default(
    tmp_path,
) -> None:
    """Section 22 rule 5, checked on the resolved numbers.

    The reference values are the Section 20.1 closed form written out here, not
    read back from the term.
    """
    resolved = resolve_for(
        tmp_path,
        {
            "G": {
                "amplitude_error": 0.02,
                "phase_error_rad": 0.0,
                "per_antenna": [
                    {"antenna": 1, "feed": 0, "amplitude_error": 0.5},
                    {"antenna": 1, "feed": 1, "phase_error_rad": 0.25},
                ],
            }
        },
    )
    gain = resolved.term("G")
    assert gain is not None
    gains = gain.gains_at_time(0.0)

    default = (1.0 + 0.02) * np.exp(1j * 0.0)
    # Antenna number 1 is row 1 of the fixture instrument, whose antenna numbers
    # are 0..N-1 in canonical order.
    assert gains[1, 0] == pytest.approx((1.0 + 0.5) * np.exp(1j * 0.0))
    # The override that set only a phase keeps the array-wide amplitude.
    assert gains[1, 1] == pytest.approx((1.0 + 0.02) * np.exp(1j * 0.25))
    assert gains[0, 0] == pytest.approx(default)
    assert gains[0, 1] == pytest.approx(default)


def test_a_bandpass_override_replaces_only_that_feeds_model(tmp_path) -> None:
    """The same precedence for ``B``, whose overridable quantity is the model."""
    resolved = resolve_for(
        tmp_path,
        {
            "B": {
                "model": {"kind": "polynomial", "coefficients": [1.0]},
                "per_antenna": [
                    {
                        "antenna": 0,
                        "feed": 1,
                        "model": {"kind": "polynomial", "coefficients": [0.5]},
                    }
                ],
            }
        },
    )
    bandpass = resolved.term("B")
    assert bandpass is not None
    values = bandpass.responses_at_frequency(100e6)

    assert values[0, 0] == pytest.approx(1.0)
    assert values[0, 1] == pytest.approx(0.5)
    assert values[1, 0] == pytest.approx(1.0)
    assert values[1, 1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Rejections, by exact message
# ---------------------------------------------------------------------------


def test_an_unknown_antenna_number_is_rejected_with_the_r4_message(
    tmp_path,
) -> None:
    """R4, verbatim, including the list of numbers the instrument does have."""
    simulator = simulator_for(tmp_path)
    known = ", ".join(
        str(antenna.id.number)
        for antenna in simulator._instrument_state.instrument.antennas
    )

    with pytest.raises(JonesAssignmentError) as caught:
        resolve_for(
            tmp_path,
            {
                "G": {
                    "amplitude_error": 0.02,
                    "per_antenna": [
                        {"antenna": 9999, "feed": 0, "amplitude_error": 0.1}
                    ],
                }
            },
        )

    assert str(caught.value) == (
        "jones.G.per_antenna references antenna number 9999, which is not in "
        f"the resolved instrument; known numbers are {known}."
    )


def test_a_duplicate_antenna_feed_pair_is_rejected_with_the_r5_message(
    tmp_path,
) -> None:
    """R5, verbatim."""
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path,
            {
                "G": {
                    "amplitude_error": 0.02,
                    "per_antenna": [
                        {"antenna": 0, "feed": 1, "amplitude_error": 0.1},
                        {"antenna": 0, "feed": 1, "phase_error_rad": 0.2},
                    ],
                }
            },
        )

    assert str(caught.value) == (
        "jones.G.per_antenna contains a duplicate entry for antenna 0 feed 1; "
        "each (antenna, feed) may appear once."
    )


@pytest.mark.parametrize("feed", [-1, 2, 7])
def test_a_feed_index_outside_zero_and_one_gets_the_r6_message(
    tmp_path,
    feed: int,
) -> None:
    """R6, verbatim, for every way of being outside ``{0, 1}``."""
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path,
            {
                "G": {
                    "amplitude_error": 0.02,
                    "per_antenna": [
                        {"antenna": 0, "feed": feed, "amplitude_error": 0.1}
                    ],
                }
            },
        )

    assert str(caught.value) == (
        f"jones.G.per_antenna feed={feed} is invalid; feeds are indexed 0 and 1 "
        "in the antenna's own receptor basis."
    )


@pytest.mark.parametrize(
    ("letter", "block"),
    [
        ("G", {"amplitude_error": 0.0, "phase_error_rad": 0.0}),
        ("G", {"elevation_curve": [1.0]}),
        ("B", {"model": {"kind": "polynomial", "coefficients": [1.0]}}),
        ("B", {"model": {"kind": "polynomial", "coefficients": [1.0, 0.0, 0.0]}}),
    ],
)
def test_a_term_that_resolves_to_the_identity_gets_the_r7_message(
    tmp_path,
    letter: str,
    block: dict[str, Any],
) -> None:
    """R7, verbatim, asked of the resolved numbers and not of the text.

    Four spellings of "no-op" -- an all-default ``G``, a unit elevation curve, a
    constant unit polynomial, and the same polynomial padded with zero
    higher-order coefficients -- and every one of them must be caught.  A term
    that cannot change the visibilities is indistinguishable from no term, which
    is the ``SCI-001`` defect this tier exists to remove.
    """
    with pytest.raises(IdentityJonesTermError) as caught:
        resolve_for(tmp_path, {letter: block})

    assert str(caught.value) == (
        f"jones.{letter} is configured with parameters that make it exactly the "
        "identity; a term that cannot change the visibilities must be removed "
        "rather than configured."
    )


def test_a_gain_that_is_unity_only_at_t0_is_not_the_identity(tmp_path) -> None:
    """The other side of R7: a time model saves an otherwise trivial ``G``.

    Unity base gains with a non-zero drift rate are unity at ``t0`` and nowhere
    else, so rejecting this would reject a physically meaningful configuration.
    """
    resolved = resolve_for(
        tmp_path,
        {
            "G": {
                "amplitude_error": 0.0,
                "time_model": {"kind": "linear_drift", "rate_per_hour": 0.01},
            }
        },
    )

    gain = resolved.term("G")
    assert gain is not None
    assert not gain.is_identity()
    assert gain.is_time_dependent


def test_a_tabulated_bandpass_that_misses_a_channel_gets_the_r11_message(
    tmp_path,
) -> None:
    """R11, verbatim.

    RadioSim does not extrapolate a bandpass: a value continued past its own
    measurement is a fabricated number, and fabricating one silently is worse
    than refusing.
    """
    simulator = simulator_for(tmp_path)
    channels = np.asarray(simulator._resolved.frequency.channel_frequencies_hz)
    low = float(channels[0])
    high = float(channels[-1])
    # Nodes that stop short of the top of the band by 1 MHz.
    nodes = list(np.linspace(low - 1e6, high - 1e6, 5))

    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path,
            {
                "B": {
                    "model": {
                        "kind": "tabulated",
                        "node_frequencies_hz": nodes,
                        "gains": [[0.9, 0.0]] * 5,
                    }
                }
            },
        )

    assert str(caught.value) == (
        f"jones.B tabulated nodes span {nodes[0]}-{nodes[-1]} Hz but the "
        f"observation covers {low}-{high} Hz; RadioSim does not extrapolate a "
        "bandpass."
    )


def test_a_tabulated_bandpass_that_covers_the_band_is_accepted(tmp_path) -> None:
    """The complement of R11, so the rejection is not vacuously always-on."""
    simulator = simulator_for(tmp_path)
    channels = np.asarray(simulator._resolved.frequency.channel_frequencies_hz)
    nodes = list(np.linspace(float(channels[0]) - 1e6, float(channels[-1]) + 1e6, 5))

    resolved = resolve_for(
        tmp_path,
        {
            "B": {
                "model": {
                    "kind": "tabulated",
                    "node_frequencies_hz": nodes,
                    "gains": [[0.9, 0.0], [0.95, 0.0], [1.0, 0.0], [0.95, 0.0]]
                    + [[0.9, 0.0]],
                }
            }
        },
    )

    assert resolved.configured_letters == ("B",)


# ---------------------------------------------------------------------------
# Failure ordering (Section 26.1)
# ---------------------------------------------------------------------------


def test_structural_failures_are_raised_before_cross_object_ones(
    tmp_path,
) -> None:
    """Stage 3 before stage 5, for a document that is wrong in both ways.

    The order is part of the contract: a user fixing a configuration must not be
    sent around a loop, told about the bandpass coverage, fix it, and only then
    learn that an antenna number was wrong all along.
    """
    simulator = simulator_for(tmp_path)
    channels = np.asarray(simulator._resolved.frequency.channel_frequencies_hz)
    short_nodes = list(
        np.linspace(float(channels[0]) - 1e6, float(channels[-1]) - 1e6, 5)
    )

    with pytest.raises(JonesAssignmentError):
        resolve_for(
            tmp_path,
            {
                "G": {
                    "amplitude_error": 0.02,
                    "per_antenna": [
                        {"antenna": 4242, "feed": 0, "amplitude_error": 0.1}
                    ],
                },
                "B": {
                    "model": {
                        "kind": "tabulated",
                        "node_frequencies_hz": short_nodes,
                        "gains": [[0.9, 0.0]] * 5,
                    }
                },
            },
        )


def test_the_identity_check_runs_last(tmp_path) -> None:
    """Stage 6 after every other stage, for a document wrong in two ways.

    The identity check needs fully resolved values, so a document whose ``G`` is
    a no-op *and* whose ``B`` cannot cover the band must report the coverage
    problem: the resolution never got far enough to know the ``G`` was trivial.
    """
    simulator = simulator_for(tmp_path)
    channels = np.asarray(simulator._resolved.frequency.channel_frequencies_hz)
    short_nodes = list(
        np.linspace(float(channels[0]) - 1e6, float(channels[-1]) - 1e6, 5)
    )

    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path,
            {
                "G": {"amplitude_error": 0.0},
                "B": {
                    "model": {
                        "kind": "tabulated",
                        "node_frequencies_hz": short_nodes,
                        "gains": [[0.9, 0.0]] * 5,
                    }
                },
            },
        )

    assert not isinstance(caught.value, IdentityJonesTermError)
    assert "does not extrapolate a bandpass" in str(caught.value)


def test_every_jones_rejection_precedes_the_first_side_effect(tmp_path) -> None:
    """Section 26.1: stages 3-6 run before any beam load, sky load, or network.

    Asserted by observing that a failing ``setup()`` leaves the beam system
    unbuilt.  A beam load is the first thing that touches the filesystem, so it
    is the right witness for "reject before side effects".
    """
    data = valid_config_mapping(tmp_path)
    data["jones"] = {"G": {"amplitude_error": 0.0}}
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)

    with pytest.raises(IdentityJonesTermError):
        simulator.setup()

    assert simulator._beam_system is None
    assert simulator._sky_model is None


# ---------------------------------------------------------------------------
# Tier 7E: D, X, Kd and Rc through the same resolution contract
# ---------------------------------------------------------------------------


def _override_payload(letter: str) -> dict[str, Any]:
    """Return the value fields one ``per_antenna`` entry of ``letter`` needs."""
    if letter == "D":
        return {"d_term": {"kind": "explicit", "d": [0.05, 0.0]}}
    if letter == "Kd":
        return {"delay_s": 4.0e-9}
    return {"amplitude": 0.3}


def _override(term: str, **fields: Any) -> dict[str, Any]:
    """Return ``term``'s block carrying exactly one ``per_antenna`` entry."""
    block = dict(FEED_KEYED_7E_TERMS[term])
    block["per_antenna"] = [fields]
    return block


def test_all_six_configurable_terms_resolve_in_canonical_order(tmp_path) -> None:
    """Section 22 rule 4, now with six letters rather than two.

    The document is written in a deliberately scrambled order so that the
    canonical result cannot be the order the keys happened to arrive in.  The
    canonical order is ``H G B Rc Kd X D ...``, which is neither alphabetical nor
    the order the terms were implemented in.
    """
    resolved = resolve_for(
        tmp_path,
        {
            "D": NONTRIVIAL_LEAKAGE,
            "G": NONTRIVIAL_GAIN,
            "Rc": NONTRIVIAL_REFLECTION,
            "X": NONTRIVIAL_CROSSHAND,
            "B": NONTRIVIAL_BANDPASS,
            "Kd": NONTRIVIAL_DELAY,
        },
    )

    assert resolved.configured_letters == ("G", "B", "Rc", "Kd", "X", "D")
    assert resolved.provenance.chain_order == (
        "H",
        "G",
        "B",
        "Rc",
        "Kd",
        "X",
        "D",
        "C",
        "E",
    )
    positions = [
        CANONICAL_CHAIN_ORDER.index(letter) for letter in resolved.configured_letters
    ]
    assert positions == sorted(positions)
    assert set(resolved.provenance.term_snapshots) == {"G", "B", "Rc", "Kd", "X", "D"}


@pytest.mark.parametrize("letter", sorted(FEED_KEYED_7E_TERMS))
def test_an_unknown_antenna_number_is_rejected_for_every_7e_term(
    tmp_path,
    letter: str,
) -> None:
    """R4, verbatim, for each term that keys its overrides by antenna and feed."""
    simulator = simulator_for(tmp_path)
    known = ", ".join(
        str(antenna.id.number)
        for antenna in simulator._instrument_state.instrument.antennas
    )

    with pytest.raises(JonesAssignmentError) as caught:
        resolve_for(tmp_path, {letter: _override(letter, antenna=9999, feed=0)})

    assert str(caught.value) == (
        f"jones.{letter}.per_antenna references antenna number 9999, which is not "
        f"in the resolved instrument; known numbers are {known}."
    )


def test_an_unknown_antenna_number_is_rejected_for_the_crosshand_term(
    tmp_path,
) -> None:
    """R4, verbatim, for the one term whose overrides carry no feed index."""
    simulator = simulator_for(tmp_path)
    known = ", ".join(
        str(antenna.id.number)
        for antenna in simulator._instrument_state.instrument.antennas
    )

    with pytest.raises(JonesAssignmentError) as caught:
        resolve_for(
            tmp_path,
            {
                "X": {
                    "phase_rad": 0.2,
                    "per_antenna": [{"antenna": 4242, "phase_rad": 1.0}],
                }
            },
        )

    assert str(caught.value) == (
        "jones.X.per_antenna references antenna number 4242, which is not in "
        f"the resolved instrument; known numbers are {known}."
    )


@pytest.mark.parametrize("letter", sorted(FEED_KEYED_7E_TERMS))
def test_a_duplicate_antenna_feed_pair_is_rejected_for_every_7e_term(
    tmp_path,
    letter: str,
) -> None:
    """R5, verbatim, for each feed-keyed Tier 7E term."""
    entry = {"antenna": 0, "feed": 1} | _override_payload(letter)
    block = dict(FEED_KEYED_7E_TERMS[letter]) | {"per_antenna": [entry, dict(entry)]}

    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(tmp_path, {letter: block})

    assert str(caught.value) == (
        f"jones.{letter}.per_antenna contains a duplicate entry for antenna 0 "
        "feed 1; each (antenna, feed) may appear once."
    )


def test_a_duplicate_antenna_is_rejected_for_the_crosshand_term(tmp_path) -> None:
    """The ``X`` analogue of R5, for a term whose key is the antenna alone.

    R5's verbatim message names an ``(antenna, feed)`` pair, and ``X`` has no
    feed index to name -- writing one would mean naming the feed the relative
    phase is *not* on.  The message is therefore the same sentence with the pair
    reduced to the key that exists, rather than a fabricated feed number.
    """
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path,
            {
                "X": {
                    "phase_rad": 0.2,
                    "per_antenna": [
                        {"antenna": 0, "phase_rad": 1.0},
                        {"antenna": 0, "delay_s": 1.0e-9},
                    ],
                }
            },
        )

    assert str(caught.value) == (
        "jones.X.per_antenna contains a duplicate entry for antenna 0; each "
        "antenna may appear once."
    )


@pytest.mark.parametrize("letter", sorted(FEED_KEYED_7E_TERMS))
@pytest.mark.parametrize("feed", [-1, 2])
def test_a_feed_index_outside_zero_and_one_is_rejected_for_every_7e_term(
    tmp_path,
    letter: str,
    feed: int,
) -> None:
    """R6, verbatim, for each feed-keyed Tier 7E term."""
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path,
            {
                letter: _override(
                    letter, antenna=0, feed=feed, **_override_payload(letter)
                )
            },
        )

    assert str(caught.value) == (
        f"jones.{letter}.per_antenna feed={feed} is invalid; feeds are indexed 0 "
        "and 1 in the antenna's own receptor basis."
    )


@pytest.mark.parametrize("amplitude", [0.0, 1.0, -1.0, 1.5, -2.25])
def test_a_reflection_amplitude_outside_the_unit_interval_gets_the_r8_message(
    tmp_path,
    amplitude: float,
) -> None:
    """R8, verbatim, including the zero that R7 would otherwise have to catch.

    ``0`` is rejected here rather than as an identity, because Section 24 states
    the physical range as ``0 < |A| < 1``: a reflection of zero amplitude is not
    a reflection, and the message that says so is more useful than the generic
    identity one.
    """
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(tmp_path, {"Rc": {"amplitude": amplitude, "cable_delay_s": 1.5e-7}})

    assert str(caught.value) == (
        f"jones.Rc.amplitude={amplitude} must satisfy 0 < |A| < 1; a reflection "
        "cannot return more power than it receives."
    )


def test_a_per_antenna_reflection_amplitude_is_range_checked_too(tmp_path) -> None:
    """R8 applies to an override as well as to the array-wide default.

    A range rule that only guarded the default would be trivially escapable by
    writing the unphysical value one level down.
    """
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path,
            {
                "Rc": {
                    "amplitude": 0.02,
                    "cable_delay_s": 1.5e-7,
                    "per_antenna": [{"antenna": 1, "feed": 0, "amplitude": 1.2}],
                }
            },
        )

    assert str(caught.value) == (
        "jones.Rc.amplitude=1.2 must satisfy 0 < |A| < 1; a reflection cannot "
        "return more power than it receives."
    )


@pytest.mark.parametrize(
    ("letter", "block"),
    [
        ("D", {"d_terms": {"kind": "explicit", "d0": [0.0, 0.0], "d1": [0.0, 0.0]}}),
        ("D", {"d_terms": {"kind": "explicit"}}),
        (
            "D",
            {
                "d_terms": {
                    "kind": "frequency_polynomial",
                    "coefficients0": [0.0, 0.0],
                    "coefficients1": [[0.0, 0.0]],
                }
            },
        ),
        ("X", {"phase_rad": 0.0, "delay_s": 0.0}),
        ("X", {}),
        ("Kd", {"delay_s": 0.0}),
        ("Kd", {}),
    ],
)
def test_a_7e_term_that_resolves_to_the_identity_gets_the_r7_message(
    tmp_path,
    letter: str,
    block: dict[str, Any],
) -> None:
    """R7, verbatim, asked of the resolved numbers and not of the text.

    Every spelling of "no leakage", "no relative phase" and "no delay" must be
    caught, including the block written with no fields at all, whose defaults are
    the identity.  ``Rc`` is absent from this table on purpose: R8 forbids a zero
    reflection amplitude outright, so ``Rc`` cannot reach the identity check.
    """
    with pytest.raises(IdentityJonesTermError) as caught:
        resolve_for(tmp_path, {letter: block})

    assert str(caught.value) == (
        f"jones.{letter} is configured with parameters that make it exactly the "
        "identity; a term that cannot change the visibilities must be removed "
        "rather than configured."
    )


def test_a_term_saved_from_the_identity_by_one_override_is_accepted(
    tmp_path,
) -> None:
    """The other side of R7: an override is enough to make a term real."""
    resolved = resolve_for(
        tmp_path,
        {
            "Kd": {
                "delay_s": 0.0,
                "per_antenna": [{"antenna": 1, "feed": 0, "delay_s": 3.0e-9}],
            }
        },
    )

    term = resolved.term("Kd")
    assert term is not None
    assert not term.is_identity()


def test_the_reflection_range_check_precedes_the_identity_check(tmp_path) -> None:
    """Stage 4 before stage 6, for a document that is wrong in both ways.

    A zero-delay ``Kd`` is an identity (stage 6) and a unit-amplitude ``Rc`` is
    out of range (stage 4).  The user must be told about the range first, because
    that is the order Section 26.1 fixes and because the identity check has not
    yet been reached.
    """
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path,
            {
                "Kd": {"delay_s": 0.0},
                "Rc": {"amplitude": 1.0, "cable_delay_s": 1.5e-7},
            },
        )

    assert not isinstance(caught.value, IdentityJonesTermError)
    assert "must satisfy 0 < |A| < 1" in str(caught.value)


def test_the_structural_check_precedes_the_reflection_range_check(tmp_path) -> None:
    """Stage 3 before stage 4, for a document that is wrong in both ways."""
    with pytest.raises(JonesAssignmentError):
        resolve_for(
            tmp_path,
            {
                "D": _override(
                    "D",
                    antenna=31337,
                    feed=0,
                    d_term={"kind": "explicit", "d": [0.1, 0.0]},
                ),
                "Rc": {"amplitude": 3.0, "cable_delay_s": 1.5e-7},
            },
        )


def test_a_leakage_override_replaces_only_that_feeds_coefficient(tmp_path) -> None:
    """``D``'s precedence, on the resolved numbers.

    An override carries a **single-feed** ``d_term`` block, while the array-wide
    default carries a **two-feed** ``d_terms`` block.  The two field names differ
    by one letter because the two shapes differ: a per-``(antenna, feed)``
    override that had to restate both feeds would make the feed index it is keyed
    by meaningless.
    """
    resolved = resolve_for(
        tmp_path,
        {
            "D": {
                "d_terms": {"kind": "explicit", "d0": [0.02, 0.0], "d1": [0.0, 0.03]},
                "per_antenna": [
                    {
                        "antenna": 1,
                        "feed": 1,
                        "d_term": {"kind": "ixr", "ixr_db": 20.0},
                    }
                ],
            }
        },
    )
    term = resolved.term("D")
    assert term is not None
    values = term.d_terms_at_frequency(1.0e8)

    assert values[0, 0] == pytest.approx(0.02 + 0.0j)
    assert values[0, 1] == pytest.approx(0.03j)
    assert values[1, 0] == pytest.approx(0.02 + 0.0j)
    assert values[1, 1] == pytest.approx(0.1 + 0.0j)


def test_a_crosshand_override_may_change_only_one_of_its_two_values(
    tmp_path,
) -> None:
    """``X``'s precedence: an entry keeps whichever value it does not restate."""
    resolved = resolve_for(
        tmp_path,
        {
            "X": {
                "phase_rad": 0.25,
                "delay_s": 1.0e-9,
                "per_antenna": [{"antenna": 1, "delay_s": 5.0e-9}],
            }
        },
    )
    term = resolved.term("X")
    assert term is not None

    assert term.phasors_at_frequency(1.0e8)[0] == pytest.approx(
        cmath.exp(1j * (0.25 + 2.0 * math.pi * 1.0e8 * 1.0e-9)), rel=1e-14
    )
    assert term.phasors_at_frequency(1.0e8)[1] == pytest.approx(
        cmath.exp(1j * (0.25 + 2.0 * math.pi * 1.0e8 * 5.0e-9)), rel=1e-14
    )


def test_every_7e_rejection_precedes_the_first_side_effect(tmp_path) -> None:
    """Section 26.1 again, for the terms this slice added.

    Asserted by observing that a failing ``setup()`` leaves the beam system and
    the sky unbuilt: a beam load is the first thing that touches the filesystem.
    """
    data = valid_config_mapping(tmp_path)
    data["jones"] = {"Rc": {"amplitude": 2.0, "cable_delay_s": 1.0e-7}}
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)

    with pytest.raises(InvalidJonesConfigError):
        simulator.setup()

    assert simulator._beam_system is None
    assert simulator._sky_model is None


# ---------------------------------------------------------------------------
# The resolved dtypes (defect D15)
# ---------------------------------------------------------------------------


def test_every_term_letter_resolves_a_precision(tmp_path) -> None:
    """D15: no term is left inheriting someone else's dtype.

    Includes ``C`` and ``H``, which are in every chain and had no precision
    field of their own before Tier 7D, and both baseline terms.
    """
    resolved = resolve_for(tmp_path, {"G": NONTRIVIAL_GAIN})

    for letter in (*CANONICAL_CHAIN_ORDER, "K", "M", "Q"):
        complex_dtype, real_dtype = resolved.dtypes.by_term[letter]
        assert complex_dtype is np.complex128
        assert real_dtype is np.float64
    assert resolved.dtypes.accumulation_complex is np.complex128
