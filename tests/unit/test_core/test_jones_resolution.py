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
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from radiosim.api.simulator import Simulator
from radiosim.core.jones_errors import (
    IdentityJonesTermError,
    InvalidJonesConfigError,
    JonesAssignmentError,
    UnsupportedMountTypeError,
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


def restamp_mount_types(
    simulator: Simulator,
    mount_types: str | None | Sequence[str | None],
) -> None:
    """Give the simulator's already-resolved instrument these mount types.

    No instrument *source* RadioSim reads carries a mount type except a
    pyuvdata dataset: a layout file has no column for one and the known-telescope
    registry does not supply one, so every shipped fixture resolves to
    ``mount_type: None``.  Tier 7F's whole subject -- a field rotation, and the
    R12/R15 rejections that guard it -- is invisible on such an array, so the
    tests restamp the resolved instrument in place rather than inventing a
    dataset fixture and a fake pyuvdata loader for every case.

    The restamp goes through the production baseline functions and recomputes
    ``instrument_sha256`` from the canonical content, so what the rest of the
    run sees is a genuine ``ResolvedInstrumentState`` and not a mock: a mount
    type that failed any canonical invariant would fail here.
    """
    from dataclasses import replace

    from radiosim.core.baseline_resolution import (
        generate_resolved_baselines,
        select_resolved_baselines,
    )
    from radiosim.core.instrument import _compute_instrument_sha256
    from radiosim.core.instrument_adapters import ResolvedInstrumentState

    state = simulator._instrument_state
    assert state is not None, "restamp_mount_types runs after instrument resolution"
    instrument = state.instrument
    if isinstance(mount_types, str) or mount_types is None:
        wanted: tuple[str | None, ...] = (mount_types,) * len(instrument.antennas)
    else:
        wanted = tuple(mount_types)

    antennas = tuple(
        replace(antenna, mount_type=mount)
        for antenna, mount in zip(instrument.antennas, wanted, strict=True)
    )
    restamped = replace(
        instrument,
        antennas=antennas,
        provenance=replace(
            instrument.provenance,
            instrument_sha256=_compute_instrument_sha256(
                instrument.name,
                instrument.location,
                antennas,
                telescope_name_source=instrument.provenance.telescope_name_source,
                location_source=instrument.provenance.location_source,
            ),
        ),
    )
    all_baselines = generate_resolved_baselines(restamped)
    simulator._instrument_state = ResolvedInstrumentState(
        instrument=restamped,
        all_baselines=all_baselines,
        selection=select_resolved_baselines(
            all_baselines,
            instrument=restamped,
            config=simulator._resolved.baseline_selection,
        ),
    )


def simulator_for(
    tmp_path,
    jones: dict[str, Any] | None = None,
    *,
    mount_types: str | None | Sequence[str | None] = None,
    **section_overrides: Any,
) -> Simulator:
    """Return a Simulator resolved as far as its instrument and receptors.

    Stops short of the beam load deliberately: Section 26.1 requires every
    ``jones:`` rejection to be raised before the first side effect, and a helper
    that loaded a beam first would make that property untestable.

    ``mount_types`` restamps the resolved instrument between instrument and
    receptor resolution -- the only point at which a mount type can enter a run
    built from a layout file (see :func:`restamp_mount_types`).
    """
    data = valid_config_mapping(tmp_path, **section_overrides)
    if jones is not None:
        data["jones"] = jones
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    if mount_types is not None:
        restamp_mount_types(simulator, mount_types)
    simulator._ensure_receptor_set()
    return simulator


def resolve_for(
    tmp_path,
    jones: dict[str, Any] | None,
    *,
    mount_types: str | None | Sequence[str | None] = None,
    **section_overrides: Any,
) -> ResolvedJonesTerms:
    """Resolve one ``jones:`` block against the standard fixture instrument.

    Every argument the resolver needs comes from the *same* simulator: the
    channel centres and their declared widths, the time grid, and the resolved
    baseline selection.  Assembling them from anywhere else would let a test
    resolve ``M`` against baselines the run does not have or ``Q`` against a
    bandwidth the run does not declare, which is exactly what the resolver's
    required parameters exist to make impossible.
    """
    simulator = simulator_for(
        tmp_path, jones, mount_types=mount_types, **section_overrides
    )
    return resolve_jones_terms(
        simulator._resolved.jones,
        simulator._instrument_state.instrument,
        frequencies_hz=simulator._resolved.frequency.channel_frequencies_hz,
        channel_widths_hz=simulator._resolved.frequency.channel_widths_hz,
        time_grid=simulator._resolved.observation.time_grid,
        baseline_selection=simulator._instrument_state.selection,
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
    *,
    mount_types: str | None | Sequence[str | None] = None,
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

    simulator = simulator_for(
        tmp_path, jones, mount_types=mount_types, **section_overrides
    )
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
        resolve_for(
            tmp_path,
            {
                letter: _override(
                    letter, antenna=9999, feed=0, **_override_payload(letter)
                )
            },
        )

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


# ---------------------------------------------------------------------------
# Tier 7F: P, and the mount-type rejections R12 and R15
# ---------------------------------------------------------------------------
#
# ``P``'s whole configuration is ``enabled``, so its resolution contract is not
# about parameter precedence -- there is no parameter -- but about the pairing
# between the term and the resolved instrument's mount types.  Section 24's
# 7F correction states that pairing as a partition: a rotating mount requires
# ``P``, a non-rotating array must not configure it, and an unmodelled mount is
# rejected either way.

NONTRIVIAL_PARALLACTIC: dict[str, Any] = {"enabled": True}


def test_p_resolves_into_the_chain_sky_side_of_c(tmp_path) -> None:
    """The corrected Section 12.2 order, read off a resolved inventory.

    ``P`` is the one letter this tier *moved*: Tier 5 put it correlator-side of
    ``C`` and Section 12.1 shows that is wrong for a circular receptor.  The
    recorded ``chain_order`` is where a reader of a result finds out which
    order was applied, so it is asserted here rather than only in the chain.
    """
    resolved = resolve_for(
        tmp_path, {"P": NONTRIVIAL_PARALLACTIC}, mount_types="alt-az"
    )

    assert resolved.configured_letters == ("P",)
    assert resolved.provenance.chain_order == ("H", "C", "E", "P")
    assert CANONICAL_CHAIN_ORDER.index("P") > CANONICAL_CHAIN_ORDER.index("C")
    assert CANONICAL_CHAIN_ORDER.index("P") > CANONICAL_CHAIN_ORDER.index("E")
    assert CANONICAL_CHAIN_ORDER.index("P") < CANONICAL_CHAIN_ORDER.index("T")


def test_the_resolved_mount_types_reach_the_provenance(tmp_path) -> None:
    """Section 22: a record that explains a field rotation says what rotated."""
    resolved = resolve_for(
        tmp_path,
        {"P": NONTRIVIAL_PARALLACTIC},
        mount_types=("alt-az", "alt-az+nasmyth-l"),
    )

    assert dict(resolved.provenance.mount_types) == {0: "alt-az", 1: "alt-az+nasmyth-l"}
    assert resolved.provenance.term_snapshots["P"] == {"enabled": True}


@pytest.mark.parametrize(
    "mount_types",
    ["alt-az", "alt-az+nasmyth-l", "alt-az+nasmyth-r", ("alt-az", "fixed")],
)
def test_a_rotating_mount_without_p_is_rejected_with_the_r15_message(
    tmp_path,
    mount_types,
) -> None:
    """R15, verbatim, and it fires with no ``jones:`` section at all.

    That is the load-bearing half: Tier 5's blanket rejection lived in
    ``resolve_receptors`` and therefore ran on every document.  Moving the rule
    to ``resolve_jones_terms`` must not turn it into a rule that only applies to
    documents which happen to configure something.
    """
    with pytest.raises(UnsupportedMountTypeError) as caught:
        resolve_for(tmp_path, None, mount_types=mount_types)

    mount = mount_types if isinstance(mount_types, str) else mount_types[0]
    assert str(caught.value) == (
        f"antenna 0 has mount_type={mount}, whose feeds rotate with the sky; "
        "enable 'jones.P' or the simulation would silently treat it as a fixed "
        "mount."
    )


def test_r15_also_fires_for_a_document_that_configures_other_terms(
    tmp_path,
) -> None:
    """The same rejection, reached through a populated ``jones:`` section."""
    with pytest.raises(UnsupportedMountTypeError):
        resolve_for(tmp_path, {"G": NONTRIVIAL_GAIN}, mount_types="alt-az")


def test_p_disabled_is_not_p_enabled_for_the_mount_rule(tmp_path) -> None:
    """``enabled: false`` does not satisfy R15; it is still an absent rotation."""
    with pytest.raises(UnsupportedMountTypeError) as caught:
        resolve_for(tmp_path, {"P": {"enabled": False}}, mount_types="alt-az")

    assert "enable 'jones.P'" in str(caught.value)


@pytest.mark.parametrize("mount_type", ["phased", "space", "bizarre"])
def test_an_unmodelled_mount_is_rejected_with_the_r12_message(
    tmp_path,
    mount_type: str,
) -> None:
    """R12, verbatim.

    Section 24's 7F correction detaches this trigger from ``jones.P``: gating it
    on the term would mean a ``phased`` mount -- rejected outright by Tier 5 --
    became a silent ``fixed`` in every run that did not configure ``P``.
    """
    for jones in (None, {"P": NONTRIVIAL_PARALLACTIC}):
        with pytest.raises(UnsupportedMountTypeError) as caught:
            resolve_for(tmp_path, jones, mount_types=mount_type)

        assert str(caught.value) == (
            f"antenna 0 has mount_type={mount_type}, which the parallactic-angle "
            "term does not model; supported mounts are alt-az, equatorial, "
            "fixed, alt-az+nasmyth-l, alt-az+nasmyth-r."
        )


@pytest.mark.parametrize("mount_types", [None, "fixed", "equatorial"])
def test_a_non_rotating_array_needs_no_p_and_may_not_configure_one(
    tmp_path,
    mount_types,
) -> None:
    """The other half of the partition: R15 silent, R7 loud.

    ``equatorial`` is the case that makes the correction necessary.  Under the
    literal R15 an equatorial array would be told to enable ``jones.P``, and
    Section 20.7 gives ``equatorial`` the mount factor ``eta = 0``, so the term
    it was told to enable is exactly ``I2`` and R7 would reject it -- leaving
    the array with no accepted configuration at all.
    """
    assert resolve_for(tmp_path, None, mount_types=mount_types).is_empty

    with pytest.raises(IdentityJonesTermError) as caught:
        resolve_for(tmp_path, {"P": NONTRIVIAL_PARALLACTIC}, mount_types=mount_types)

    assert str(caught.value) == (
        "jones.P is configured with parameters that make it exactly the "
        "identity; a term that cannot change the visibilities must be removed "
        "rather than configured."
    )


def test_p_enabled_false_is_rejected_as_an_identity(tmp_path) -> None:
    """Section 21's "there is no ``enabled: false``", for the one term with a flag.

    ``P`` is the only block that carries an ``enabled`` key, because the
    parallactic angle has no other parameter.  Writing ``false`` there is a
    disabled term, and a disabled term must be removed rather than configured --
    which is exactly what R7 says, so R7 is what says it.
    """
    with pytest.raises(IdentityJonesTermError) as caught:
        resolve_for(tmp_path, {"P": {"enabled": False}}, mount_types="fixed")

    assert "jones.P is configured with parameters" in str(caught.value)


def test_a_mixed_array_needs_p_when_any_antenna_rotates(tmp_path) -> None:
    """One rotating antenna is enough to make the term non-identity."""
    resolved = resolve_for(
        tmp_path,
        {"P": NONTRIVIAL_PARALLACTIC},
        mount_types=("alt-az", "fixed"),
    )

    term = resolved.term("P")
    assert term is not None
    assert term.is_identity() is False
    assert term.mount_types == ("alt-az", "fixed")


def test_the_mount_rejections_precede_every_other_jones_failure(tmp_path) -> None:
    """Section 26.1: stage 5 runs before stage 6, and before any side effect.

    The document below is wrong twice -- an unmodelled mount and a ``G`` naming
    an antenna that does not exist -- and R4 is a *stage 3* rejection, so the
    ordering rule says the structural failure is reported first.
    """
    with pytest.raises(JonesAssignmentError):
        resolve_for(
            tmp_path,
            {
                "G": {
                    "amplitude_error": 0.02,
                    "per_antenna": [
                        {"antenna": 4242, "feed": 0, "amplitude_error": 0.1}
                    ],
                }
            },
            mount_types="phased",
        )

    # ... and with the structural mistake fixed, the mount rejection is what
    # remains, still ahead of the identity check at stage 6.
    with pytest.raises(UnsupportedMountTypeError):
        resolve_for(
            tmp_path,
            {"G": NONTRIVIAL_GAIN, "P": {"enabled": False}},
            mount_types="phased",
        )


def test_a_mount_rejection_stops_setup_before_the_first_side_effect(
    tmp_path,
) -> None:
    """R15 through the public entry point, with nothing loaded behind it."""
    data = valid_config_mapping(tmp_path)
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    restamp_mount_types(simulator, "alt-az")

    with pytest.raises(UnsupportedMountTypeError):
        simulator.setup()

    assert simulator._beam_system is None
    assert simulator._sky_model is None


def test_p_is_the_only_term_whose_snapshot_is_a_bare_flag(tmp_path) -> None:
    """Section 21.3, on the resolved snapshot rather than on the schema."""
    resolved = resolve_for(
        tmp_path,
        {"P": NONTRIVIAL_PARALLACTIC, "G": NONTRIVIAL_GAIN},
        mount_types="alt-az",
    )

    assert set(resolved.provenance.term_snapshots["P"]) == {"enabled"}
    assert resolved.configured_letters == ("G", "P")


# ---------------------------------------------------------------------------
# Tier 7G: T and Z, the range rejections R9 and R10, and the R13 elevation guard
# ---------------------------------------------------------------------------
#
# The two propagation terms whose parameters come from *both* the document and
# the instrument: ``T`` reads each antenna's height for the Saastamoinen delay
# and the Niell height correction, and ``Z`` reads each antenna's ENU position
# for the pierce point of its gradient screen.  Neither has a ``per_antenna``
# block for those quantities, deliberately -- they belong to the instrument.

NONTRIVIAL_TROPOSPHERE: dict[str, Any] = {
    "zenith_delay": {
        "kind": "saastamoinen",
        "surface_pressure_hpa": 1013.25,
        "zenith_wet_delay_m": 0.1,
    },
    "mapping_function": "niell",
    "minimum_elevation_deg": 1.0,
}

NONTRIVIAL_IONOSPHERE: dict[str, Any] = {
    "tec": {"kind": "constant", "vertical_tec_tecu": 12.0},
    "minimum_elevation_deg": 1.0,
}


def test_the_two_propagation_terms_resolve_sky_side_of_everything(tmp_path) -> None:
    """Section 12.2: ``Z`` is the outermost medium, then ``T``, then the array.

    The signal crosses the ionosphere first and the troposphere second, so ``Z``
    is the rightmost factor and ``T`` the next one in -- which is exactly the
    order the canonical constant already carried, and this reads it back off a
    resolved inventory rather than off the constant.
    """
    resolved = resolve_for(
        tmp_path, {"T": NONTRIVIAL_TROPOSPHERE, "Z": NONTRIVIAL_IONOSPHERE}
    )

    assert resolved.configured_letters == ("T", "Z")
    assert resolved.provenance.chain_order == ("H", "C", "E", "T", "Z")
    assert CANONICAL_CHAIN_ORDER.index("Z") == len(CANONICAL_CHAIN_ORDER) - 1
    assert CANONICAL_CHAIN_ORDER.index("T") == CANONICAL_CHAIN_ORDER.index("Z") - 1


def test_all_nine_configurable_terms_resolve_in_canonical_order(tmp_path) -> None:
    """Every per-antenna term in the chain, together, in one document.

    Tier 7G is the slice at which "every configurable term" stops being a
    growing subset: after it, only the two baseline-dependent terms are left,
    and they are not chain terms at all.
    """
    resolved = resolve_for(
        tmp_path,
        {
            "Z": NONTRIVIAL_IONOSPHERE,
            "D": NONTRIVIAL_LEAKAGE,
            "G": NONTRIVIAL_GAIN,
            "T": NONTRIVIAL_TROPOSPHERE,
            "Rc": NONTRIVIAL_REFLECTION,
            "P": NONTRIVIAL_PARALLACTIC,
            "B": NONTRIVIAL_BANDPASS,
            "X": NONTRIVIAL_CROSSHAND,
            "Kd": NONTRIVIAL_DELAY,
        },
        mount_types="alt-az",
    )

    assert resolved.configured_letters == (
        "G",
        "B",
        "Rc",
        "Kd",
        "X",
        "D",
        "P",
        "T",
        "Z",
    )
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
        "P",
        "T",
        "Z",
    )
    assert resolved.provenance.chain_order == CANONICAL_CHAIN_ORDER


def test_the_saastamoinen_delay_is_resolved_per_antenna_from_the_instrument(
    tmp_path,
) -> None:
    """Section 20.9: the pressure is configured; the latitude and heights are not.

    The resolved zenith delay is checked against the closed form evaluated at the
    fixture site, written out here, so a term that quietly used sea level or the
    wrong latitude would be caught by a number rather than by a shape.
    """
    simulator = simulator_for(tmp_path, {"T": NONTRIVIAL_TROPOSPHERE})
    instrument = simulator._instrument_state.instrument
    resolved = resolve_jones_terms(
        simulator._resolved.jones,
        instrument,
        frequencies_hz=simulator._resolved.frequency.channel_frequencies_hz,
        channel_widths_hz=simulator._resolved.frequency.channel_widths_hz,
        time_grid=simulator._resolved.observation.time_grid,
        baseline_selection=simulator._instrument_state.selection,
        precision=simulator._precision,
    )

    term = resolved.term("T")
    assert term is not None
    latitude_deg = instrument.location.latitude_deg
    for row, antenna in enumerate(instrument.antennas):
        height_m = instrument.location.height_m + antenna.position_enu_m[2]
        expected = (
            0.0022768
            * 1013.25
            / (
                1.0
                - 0.00266 * math.cos(2.0 * math.radians(latitude_deg))
                - 0.00028 * (height_m / 1000.0)
            )
        )
        assert float(term.zenith_hydrostatic_delay_m[row]) == pytest.approx(
            expected, rel=1e-12
        )
        assert float(term.zenith_wet_delay_m[row]) == pytest.approx(0.1, rel=0.0)


def test_an_explicit_zenith_delay_is_used_as_written(tmp_path) -> None:
    """The other variant: no formula, the two numbers the document gave."""
    resolved = resolve_for(
        tmp_path,
        {
            "T": {
                "zenith_delay": {
                    "kind": "explicit",
                    "zenith_hydrostatic_delay_m": 2.15,
                    "zenith_wet_delay_m": 0.35,
                },
                "minimum_elevation_deg": 2.0,
            }
        },
    )

    term = resolved.term("T")
    assert term is not None
    assert list(term.zenith_hydrostatic_delay_m) == [2.15, 2.15]
    assert list(term.zenith_wet_delay_m) == [0.35, 0.35]
    assert term.mapping_function == "niell"
    assert term.zenith_opacity is None


def test_a_per_antenna_rotation_measure_beats_the_array_wide_default(
    tmp_path,
) -> None:
    """Section 22 rule 5, for the one per-antenna value ``Z`` accepts."""
    resolved = resolve_for(
        tmp_path,
        {
            "Z": {
                "tec": {"kind": "constant", "vertical_tec_tecu": 8.0},
                "minimum_elevation_deg": 1.0,
                "faraday": {
                    "rotation_measure_rad_m2": 0.4,
                    "per_antenna": [{"antenna": 1, "rotation_measure_rad_m2": -1.1}],
                },
            }
        },
    )

    term = resolved.term("Z")
    assert term is not None
    assert list(term.rotation_measures_rad_m2) == [0.4, -1.1]
    assert term.shell_height_m == pytest.approx(350_000.0, rel=0.0)


def test_the_gradient_screen_reads_the_instruments_own_positions(tmp_path) -> None:
    """A gradient is only a differential if it is evaluated at each antenna."""
    resolved = resolve_for(
        tmp_path,
        {
            "Z": {
                "tec": {
                    "kind": "gradient",
                    "vertical_tec_tecu": 10.0,
                    "gradient_east_tecu_per_km": 0.2,
                },
                "minimum_elevation_deg": 1.0,
            }
        },
    )

    term = resolved.term("Z")
    assert term is not None
    assert term.tec_model.is_uniform is False
    assert term.tec_model.gradient_east_tecu_per_km == 0.2
    assert term.tec_model.gradient_north_tecu_per_km == 0.0


def test_an_unknown_antenna_number_is_rejected_for_the_ionosphere(tmp_path) -> None:
    """R4, verbatim, through ``Z``'s antenna-keyed Faraday overrides."""
    with pytest.raises(JonesAssignmentError) as caught:
        resolve_for(
            tmp_path,
            {
                "Z": {
                    "tec": {"kind": "constant", "vertical_tec_tecu": 8.0},
                    "minimum_elevation_deg": 1.0,
                    "faraday": {
                        "per_antenna": [{"antenna": 91, "rotation_measure_rad_m2": 0.5}]
                    },
                }
            },
        )

    assert str(caught.value) == (
        "jones.Z.per_antenna references antenna number 91, which is not in the "
        "resolved instrument; known numbers are 0, 1."
    )


def test_a_duplicate_antenna_is_rejected_for_the_ionosphere(tmp_path) -> None:
    """R5 in its antenna-only form: a rotation measure carries no feed index."""
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path,
            {
                "Z": {
                    "tec": {"kind": "constant", "vertical_tec_tecu": 8.0},
                    "minimum_elevation_deg": 1.0,
                    "faraday": {
                        "per_antenna": [
                            {"antenna": 0, "rotation_measure_rad_m2": 0.5},
                            {"antenna": 0, "rotation_measure_rad_m2": 0.9},
                        ]
                    },
                }
            },
        )

    assert str(caught.value) == (
        "jones.Z.per_antenna contains a duplicate entry for antenna 0; each "
        "antenna may appear once."
    )


def test_a_negative_vertical_tec_gets_the_r9_message(tmp_path) -> None:
    """R9, verbatim: there is no such thing as a negative electron column."""
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path,
            {
                "Z": {
                    "tec": {"kind": "constant", "vertical_tec_tecu": -3.0},
                    "minimum_elevation_deg": 1.0,
                }
            },
        )

    assert str(caught.value) == (
        "jones.Z.tec.vertical_tec_tecu=-3.0 must be non-negative."
    )


def test_a_negative_zenith_opacity_gets_the_r10_message(tmp_path) -> None:
    """R10, verbatim: a negative opacity would amplify."""
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path,
            {
                "T": {
                    **NONTRIVIAL_TROPOSPHERE,
                    "opacity": {"zenith_opacity": -0.02},
                }
            },
        )

    assert str(caught.value) == (
        "jones.T.opacity.zenith_opacity=-0.02 must be non-negative; a negative "
        "opacity would amplify."
    )


def test_a_zero_opacity_is_accepted_and_a_zero_screen_is_not(tmp_path) -> None:
    """The boundary between R10 and R7: zero is legal, but it is not a term.

    A zero opacity with a real delay is a transparent atmosphere -- accepted, and
    the case in which ``T`` is unitary.  A zero opacity with *no* delay cannot
    change a visibility, so it is R7 rather than R10.
    """
    resolved = resolve_for(
        tmp_path,
        {"T": {**NONTRIVIAL_TROPOSPHERE, "opacity": {"zenith_opacity": 0.0}}},
    )
    term = resolved.term("T")
    assert term is not None
    assert term.is_unitary() is True

    with pytest.raises(IdentityJonesTermError) as caught:
        resolve_for(
            tmp_path,
            {
                "T": {
                    "zenith_delay": {"kind": "explicit"},
                    "minimum_elevation_deg": 1.0,
                    "opacity": {"zenith_opacity": 0.0},
                }
            },
        )
    assert str(caught.value) == (
        "jones.T is configured with parameters that make it exactly the "
        "identity; a term that cannot change the visibilities must be removed "
        "rather than configured."
    )


def test_an_empty_ionosphere_gets_the_r7_message(tmp_path) -> None:
    """R7 for ``Z``: no electrons and no rotation is no term."""
    with pytest.raises(IdentityJonesTermError) as caught:
        resolve_for(
            tmp_path,
            {
                "Z": {
                    "tec": {"kind": "constant", "vertical_tec_tecu": 0.0},
                    "minimum_elevation_deg": 1.0,
                }
            },
        )

    assert str(caught.value) == (
        "jones.Z is configured with parameters that make it exactly the "
        "identity; a term that cannot change the visibilities must be removed "
        "rather than configured."
    )


def test_a_faraday_only_ionosphere_is_not_the_identity(tmp_path) -> None:
    """The two halves are independent: either one alone is a real ``Z``."""
    resolved = resolve_for(
        tmp_path,
        {
            "Z": {
                "tec": {"kind": "constant", "vertical_tec_tecu": 0.0},
                "minimum_elevation_deg": 1.0,
                "faraday": {"rotation_measure_rad_m2": 0.75},
            }
        },
    )

    term = resolved.term("Z")
    assert term is not None
    assert term.is_identity() is False
    assert term.is_scalar() is False


def test_the_range_rejections_precede_the_identity_check(tmp_path) -> None:
    """Section 26.1: stage 4 before stage 6, for both new terms at once.

    Each document below is wrong in two ways -- an out-of-range value *and* an
    otherwise-identity term -- and the stage-4 message is the one raised.
    """
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path,
            {
                "Z": {
                    "tec": {"kind": "constant", "vertical_tec_tecu": -1.0},
                    "minimum_elevation_deg": 1.0,
                }
            },
        )
    assert "must be non-negative." in str(caught.value)
    assert not isinstance(caught.value, IdentityJonesTermError)

    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path,
            {
                "T": {
                    "zenith_delay": {"kind": "explicit"},
                    "minimum_elevation_deg": 1.0,
                    "opacity": {"zenith_opacity": -1.0},
                }
            },
        )
    assert "would amplify." in str(caught.value)
    assert not isinstance(caught.value, IdentityJonesTermError)


def test_the_structural_check_precedes_the_ionospheric_range_check(tmp_path) -> None:
    """Stage 3 before stage 4, on one document that is wrong in both ways."""
    with pytest.raises(JonesAssignmentError):
        resolve_for(
            tmp_path,
            {
                "Z": {
                    "tec": {"kind": "constant", "vertical_tec_tecu": -5.0},
                    "minimum_elevation_deg": 1.0,
                    "faraday": {
                        "per_antenna": [
                            {"antenna": 404, "rotation_measure_rad_m2": 0.5}
                        ]
                    },
                }
            },
        )


def test_every_7g_rejection_precedes_the_first_side_effect(tmp_path) -> None:
    """Section 26.1's closing rule, for the two terms this slice adds."""
    for jones in (
        {
            "Z": {
                "tec": {"kind": "constant", "vertical_tec_tecu": -1.0},
                "minimum_elevation_deg": 1.0,
            }
        },
        {
            "T": {
                "zenith_delay": {"kind": "explicit", "zenith_hydrostatic_delay_m": 2.3},
                "minimum_elevation_deg": 1.0,
                "opacity": {"zenith_opacity": -0.5},
            }
        },
    ):
        data = valid_config_mapping(tmp_path)
        data["jones"] = jones
        simulator = Simulator.from_mapping(data, base_dir=tmp_path)

        with pytest.raises(InvalidJonesConfigError):
            simulator.setup()

        assert simulator._beam_system is None
        assert simulator._sky_model is None


def test_r13_is_raised_at_evaluation_because_it_is_about_directions(
    tmp_path,
) -> None:
    """Section 26.1's one stage that cannot run at resolution, and why.

    R13's condition is "a direction survives the horizon mask below
    ``minimum_elevation_deg``", and no direction exists until a solver resolves
    one for a ``(time, frequency)`` step.  So resolution *accepts* a high
    minimum elevation, and the term itself refuses the first batch that violates
    it -- with R13's own message, naming the term.
    """
    from radiosim.backends import get_backend
    from radiosim.core.jones.directions import DirectionBatch

    resolved = resolve_for(
        tmp_path,
        {"T": {**NONTRIVIAL_TROPOSPHERE, "minimum_elevation_deg": 30.0}},
    )
    term = resolved.term("T")
    assert term is not None
    assert term.minimum_elevation_deg == 30.0

    alt = np.radians(np.array([80.0, 10.0]))
    az = np.array([0.0, 1.0])
    directions = DirectionBatch.from_horizontal(
        alt_rad=alt,
        az_rad=az,
        dir_l=np.cos(alt) * np.sin(az),
        dir_m=np.cos(alt) * np.cos(az),
        dir_n=np.sin(alt),
        latitude_rad=-0.5362,
        local_sidereal_time_rad=0.0,
    )

    with pytest.raises(InvalidJonesConfigError) as caught:
        term.compute_jones_batch(
            antenna_idx=0,
            directions=directions,
            frequency_hz=1.5e8,
            freq_idx=0,
            time_mjd=60_676.0,
            time_idx=0,
            backend=get_backend("numpy"),
            dtype=np.complex128,
        )

    assert str(caught.value) == (
        "jones.T.minimum_elevation_deg=30.0 excludes no direction, but the "
        "mapping function diverges below 30.0 deg; raise the minimum elevation "
        "or the horizon mask."
    )


def test_the_two_new_terms_are_snapshotted_into_the_fingerprint(tmp_path) -> None:
    """Section 25.1: a Jones parameter that changes must change the digest."""
    base = resolve_for(
        tmp_path, {"T": NONTRIVIAL_TROPOSPHERE, "Z": NONTRIVIAL_IONOSPHERE}
    )
    changed_tec = resolve_for(
        tmp_path,
        {
            "T": NONTRIVIAL_TROPOSPHERE,
            "Z": {**NONTRIVIAL_IONOSPHERE, "shell_height_km": 300.0},
        },
    )
    changed_mapping = resolve_for(
        tmp_path,
        {
            "T": {**NONTRIVIAL_TROPOSPHERE, "mapping_function": "simple"},
            "Z": NONTRIVIAL_IONOSPHERE,
        },
    )

    assert set(base.provenance.term_snapshots) == {"T", "Z"}
    assert base.provenance.term_snapshots["Z"]["shell_height_km"] == 350.0
    assert base.provenance.term_snapshots["T"]["mapping_function"] == "niell"
    assert (
        len(
            {
                base.provenance.jones_sha256,
                changed_tec.provenance.jones_sha256,
                changed_mapping.provenance.jones_sha256,
            }
        )
        == 3
    )


# ---------------------------------------------------------------------------
# Tier 7H: M and Q, the two baseline-dependent terms
# ---------------------------------------------------------------------------
#
# Neither is a chain term, so what resolution owes them is different from what
# it owes the nine per-antenna letters: ``M`` is validated against the resolved
# *baseline selection* rather than against the antenna list (R14), and ``Q``
# takes no physical parameter at all -- its two grids come from the resolved
# observation configuration, because a smearing integration time that disagreed
# with the time grid the solver iterates would be a fabrication (Section 20.11).


#: One non-identity ``M``, and one ``Q`` with both envelopes active.
#: The parallel-hand entries are real because the shipped selection carries
#: autocorrelations and R17 refuses a complex factor on one; the cross-hand
#: entries are complex, where the physics allows it.
NONTRIVIAL_CLOSURE: dict[str, Any] = {
    "matrix": [[[1.04, 0.0], [0.98, 0.02]], [[1.01, -0.03], [0.96, 0.0]]]
}
NONTRIVIAL_SMEARING: dict[str, Any] = {
    "bandwidth_smearing": True,
    "time_smearing": True,
}


def test_a_baseline_term_is_resolved_outside_the_chain(tmp_path) -> None:
    """The inventory keeps the two paths apart, by construction."""
    from radiosim.core.jones.baseline_errors import (
        BaselineMultiplicativeJones,
        SmearingFactorJones,
    )

    resolved = resolve_for(
        tmp_path, {"M": NONTRIVIAL_CLOSURE, "Q": NONTRIVIAL_SMEARING}
    )

    assert resolved.chain_terms == ()
    assert resolved.configured_letters == ()
    assert resolved.baseline_letters == ("M", "Q")
    assert [type(term) for term in resolved.baseline_terms] == [
        BaselineMultiplicativeJones,
        SmearingFactorJones,
    ]
    assert resolved.provenance.chain_order == ("H", "C", "E")
    assert resolved.provenance.enabled_terms == ("H", "C", "E", "M", "Q")
    assert set(resolved.provenance.term_snapshots) == {"M", "Q"}


def test_baseline_terms_are_ordered_canonically_not_as_written(tmp_path) -> None:
    """``M`` before ``Q``, whichever order the document used."""
    resolved = resolve_for(
        tmp_path, {"Q": NONTRIVIAL_SMEARING, "M": NONTRIVIAL_CLOSURE}
    )

    assert [term.name for term in resolved.baseline_terms] == ["M", "Q"]


def test_a_baseline_term_composes_with_the_chain_terms(tmp_path) -> None:
    """A run may configure both kinds, and neither displaces the other."""
    resolved = resolve_for(
        tmp_path,
        {
            "G": {"amplitude_error": 0.03},
            "M": NONTRIVIAL_CLOSURE,
            "Q": NONTRIVIAL_SMEARING,
        },
    )

    assert resolved.configured_letters == ("G",)
    assert resolved.baseline_letters == ("M", "Q")
    assert resolved.provenance.enabled_terms == ("H", "G", "C", "E", "M", "Q")


def test_an_unknown_baseline_pair_is_rejected_with_the_r14_message(tmp_path) -> None:
    """R14, verbatim, from the file that owns Section 24's message table."""
    with pytest.raises(JonesAssignmentError) as caught:
        resolve_for(
            tmp_path,
            {
                "M": {
                    "per_baseline": [
                        {
                            "antennas": [0, 4],
                            "matrix": [
                                [[1.5, 0.0], [0.0, 0.0]],
                                [[0.0, 0.0], [1.0, 0.0]],
                            ],
                        }
                    ]
                }
            },
        )

    assert str(caught.value) == (
        "jones.M.per_baseline references baseline (0, 4), which is not in the "
        "resolved baseline selection."
    )


def test_a_duplicate_baseline_is_rejected_with_the_adapted_r5_message(
    tmp_path,
) -> None:
    """R5's bounded form for the one term keyed by a baseline (Section 20.10)."""
    entry = {
        "antennas": [0, 1],
        "matrix": [[[1.5, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
    }
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(tmp_path, {"M": {"per_baseline": [entry, dict(entry)]}})

    assert str(caught.value) == (
        "jones.M.per_baseline contains a duplicate entry for baseline (0, 1); "
        "each baseline may appear once."
    )


def test_a_smearing_block_with_nothing_enabled_is_rejected_with_r16(tmp_path) -> None:
    """R16, verbatim."""
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(
            tmp_path, {"Q": {"bandwidth_smearing": False, "time_smearing": False}}
        )

    assert str(caught.value) == (
        "jones.Q is enabled with both smearing kinds disabled; remove the "
        "section instead."
    )


def test_an_all_ones_closure_error_is_rejected_with_r7(tmp_path) -> None:
    """R7 reaches the baseline path too, at the value that really is neutral.

    The neutral element of a Hadamard product is the all-**ones** matrix.  An
    ``M`` of identity matrices is a different configuration entirely -- it nulls
    both cross-hands -- and is accepted, because it changes the visibilities.
    """
    with pytest.raises(IdentityJonesTermError) as caught:
        resolve_for(
            tmp_path,
            {"M": {"matrix": [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]}},
        )

    assert str(caught.value) == (
        "jones.M is configured with parameters that make it exactly the "
        "identity; a term that cannot change the visibilities must be removed "
        "rather than configured."
    )


def test_the_baseline_rejections_keep_the_mandatory_failure_order(tmp_path) -> None:
    """Section 26.1: structural (R14) before physical (R16) before identity (R7).

    A document with all three mistakes is told about the baseline it named that
    does not exist, because that is the one the reader can act on without first
    understanding the other two.
    """
    with pytest.raises(JonesAssignmentError):
        resolve_for(
            tmp_path,
            {
                "M": {
                    "matrix": [
                        [[1.0, 0.0], [1.0, 0.0]],
                        [[1.0, 0.0], [1.0, 0.0]],
                    ],
                    "per_baseline": [
                        {
                            "antennas": [0, 9],
                            "matrix": [
                                [[1.0, 0.0], [0.0, 0.0]],
                                [[0.0, 0.0], [1.0, 0.0]],
                            ],
                        }
                    ],
                },
                "Q": {"bandwidth_smearing": False, "time_smearing": False},
            },
        )


def test_the_baseline_terms_enter_the_fingerprint(tmp_path) -> None:
    """Section 25.1, for the two letters that are not in the chain."""
    base = resolve_for(tmp_path, {"M": NONTRIVIAL_CLOSURE, "Q": NONTRIVIAL_SMEARING})
    changed_matrix = resolve_for(
        tmp_path,
        {
            "M": {
                "matrix": [[[1.05, 0.0], [0.98, 0.02]], [[1.01, -0.03], [0.96, 0.0]]]
            },
            "Q": NONTRIVIAL_SMEARING,
        },
    )
    changed_smearing = resolve_for(
        tmp_path,
        {
            "M": NONTRIVIAL_CLOSURE,
            "Q": {"bandwidth_smearing": True, "time_smearing": False},
        },
    )

    assert base.provenance.term_snapshots["Q"] == {
        "bandwidth_smearing": True,
        "time_smearing": True,
    }
    assert (
        len(
            {
                base.provenance.jones_sha256,
                changed_matrix.provenance.jones_sha256,
                changed_smearing.provenance.jones_sha256,
            }
        )
        == 3
    )


def test_the_resolved_smearing_term_reads_the_runs_grids(tmp_path) -> None:
    """``dnu`` and ``dt`` are the run's own, and the term says which run."""
    resolved = resolve_for(tmp_path, {"Q": NONTRIVIAL_SMEARING})

    (term,) = resolved.baseline_terms
    np.testing.assert_allclose(term.channel_frequencies_hz, [1.0e8, 1.01e8, 1.02e8])
    np.testing.assert_allclose(term.channel_widths_hz, [1.0e6, 1.0e6, 1.0e6])
    np.testing.assert_allclose(term.integration_time_s, [1.0, 1.0])
