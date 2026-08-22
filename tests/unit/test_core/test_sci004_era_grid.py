r"""SCI-004 phase-M1 red oracles for the canonical exact-turn ERA grid.

``docs/development/sci004_mmode_design.md`` Section 3.1 makes the m-mode time
coordinate the continuously unwrapped number of Earth-rotation turns,

.. math::

    u(t)=\frac{\operatorname{ERA}(t)-\operatorname{ERA}(t_0)}{2\pi},
    \qquad u(t_0)=0,

and builds the whole grid in **exact rational arithmetic**: centres
``u_k = reduce(2k/2N)``, edges ``reduce((2k -/+ p_f)/(2N q_f))``, exposure width
``reduce(p_f/(N q_f))``, and the cell-centred cycle
``H_N = [reduce(-1/(2N)), reduce((2N-1)/(2N)))``. Radians are a *derived view*:
each value is one final round-to-nearest-ties-to-even of
``exact(tau) * <exact rational>`` with ``tau =
float.fromhex("0x1.921fb54442d18p+2")`` and no intermediate binary64 arithmetic.

Two consequences are load-bearing and are pinned below. First, exact-turn
equality, not rounded-radian subtraction, is the closure authority: Section 3.1
says explicitly that there is *deliberately no* assertion
``horizon_hi_rad - horizon_lo_rad == tau``, and Section 14.2 says a predicate
requiring it "is itself a validation failure". Second, there is no sample at
``u = 1``; that value exists only as the virtual closure point.

Section 12.2 family 1 adds the transform oracles: the normalized forward
``bar_v_m = (1/N) sum_k bar_V_k exp(-i 2 pi m u_k)`` and its inverse for
positive, negative and zero ``m``, the exposure ``sinc(pi m Delta_u)`` factor,
and the UTC/UT1 round trip. The analytic complex128 residual limit is
``5e-12``; the construction tolerances ``2e-11 rad``, ``2e-11 rad`` and
``1e-6 s`` are frozen constants, never YAML fields.

The Section 13.3 owner of all of this is ``radiosim.core.mmode.time``, which
does not exist at ``G1``. Imports are therefore function-local so that each node
fails individually with its own recordable Section 14.1 outcome rather than
collapsing the module into one collection error, which Section 14.1 rejects as
"collection-only".
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction
from typing import Any

import pytest

# --- Section 3.1 frozen literals ---------------------------------------------

#: The exact binary64 turn-to-radian constant. Section 3.1 spells it as a hex
#: literal precisely so that no decimal transcription can perturb it.
TAU = float.fromhex("0x1.921fb54442d18p+2")

TURN_GRID_SCHEMA = "radiosim.mmode-era-turn-grid.v1"
RADIAN_GRID_SCHEMA = "radiosim.mmode-era-grid.v2"

#: Section 3.1's exact ``canonical_era_turn_grid`` key order.
TURN_GRID_KEYS: tuple[str, ...] = (
    "schema_version",
    "sidereal_samples",
    "integration_fraction_f64be",
    "integration_fraction_ratio",
    "exposure_width_turn",
    "horizon_lo_turn",
    "horizon_hi_turn",
    "center_turns",
    "lower_edge_turns",
    "upper_edge_turns",
)

#: Section 3.1's exact ``canonical_era_grid`` key order.
RADIAN_GRID_KEYS: tuple[str, ...] = (
    "schema_version",
    "canonical_era_turn_grid_sha256",
    "era_center_turn_sha256",
    "era_lower_edge_turn_sha256",
    "era_upper_edge_turn_sha256",
    "tau_f64be",
    "delta_alpha_rad_f64be",
    "horizon_lo_rad_f64be",
    "horizon_hi_rad_f64be",
    "era_center_rad_sha256",
    "era_lower_edge_rad_sha256",
    "era_upper_edge_rad_sha256",
)

#: Section 3.1's eight component digest fields.
GRID_DIGEST_FIELDS: tuple[str, ...] = (
    "era_center_turn_sha256",
    "era_lower_edge_turn_sha256",
    "era_upper_edge_turn_sha256",
    "canonical_era_turn_grid_sha256",
    "era_center_rad_sha256",
    "era_lower_edge_rad_sha256",
    "era_upper_edge_rad_sha256",
    "canonical_era_grid_sha256",
)

#: Section 3.1's fixed construction tolerances. Constants, not YAML fields.
ERA_CENTER_LIMIT_RAD = 2e-11
ERA_STEP_LIMIT_RAD = 2e-11
UT1_UTC_ROUNDTRIP_LIMIT_SECONDS = 1e-6

#: Section 12.2's analytic complex128 DFT/single-mode residual limit.
ANALYTIC_RESIDUAL_LIMIT = 5e-12

#: Section 12.2 family 1's named grid: ``N = 17``, ``f = 1``.
EXACT_TURN_SAMPLES = 17
EXACT_TURN_FRACTION = 1.0

#: The nontrivial binary64 fraction whose exact IEEE ratio family 1 requires be
#: reconstructed rather than re-parsed from its decimal spelling.
NONTRIVIAL_FRACTION = 0.3

START_TIME = "2025-01-01T00:00:00"


# --- fixture documents --------------------------------------------------------

_FULL_WIDTH_FIXTURE = f"""\
sidereal_samples: {EXACT_TURN_SAMPLES}
integration_fraction: {EXACT_TURN_FRACTION}
start_time: "{START_TIME}"
""".encode()

_NONTRIVIAL_FIXTURE = f"""\
sidereal_samples: {EXACT_TURN_SAMPLES}
integration_fraction: {NONTRIVIAL_FRACTION}
start_time: "{START_TIME}"
""".encode()

_TRANSFORM_FIXTURE = f"""\
sidereal_samples: {EXACT_TURN_SAMPLES}
integration_fraction: 0.5
start_time: "{START_TIME}"
signed_m: [-3, 0, 5]
""".encode()

_COLLAPSED_FIXTURE = f"""\
sidereal_samples: 257
integration_fraction: 1.0e-308
start_time: "{START_TIME}"
""".encode()


def _case(
    case_id: str,
    requirement_id: str,
    function: str,
    fixture: bytes,
    *,
    excluded_by: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": f"tests/unit/test_core/test_sci004_era_grid.py::{function}",
        "expected_failure_kind": "import",
        "expected_failure_pattern": (
            r"ModuleNotFoundError: No module named 'radiosim\.core\.mmode'"
        ),
        "fixture_defect_excluded_by": excluded_by,
        "fixture_bytes": fixture,
    }


_RATIO_ORACLE = (
    "tests/unit/test_core/test_sci004_era_grid.py::"
    "test_the_fixture_fraction_reconstructs_its_exact_ieee_ratio"
)
_TURN_ORACLE = (
    "tests/unit/test_core/test_sci004_era_grid.py::"
    "test_the_exact_turn_construction_is_reproduced_in_the_test_body"
)
_DFT_ORACLE = (
    "tests/unit/test_core/test_sci004_era_grid.py::"
    "test_the_analytic_single_mode_dft_oracle_closes_in_the_test_body"
)

SCI004_RED_CASES: tuple[dict[str, Any], ...] = (
    _case(
        "m1.era.exact-turn-invariants",
        "sci004.section-3.1.exact-turn-construction",
        "test_the_exact_turn_grid_has_unit_width_and_full_containment",
        _FULL_WIDTH_FIXTURE,
        excluded_by=_TURN_ORACLE,
    ),
    _case(
        "m1.era.full-width-adjacency",
        "sci004.section-3.1.full-width-shared-edges",
        "test_full_width_adjacent_edges_are_the_same_exact_rational",
        _FULL_WIDTH_FIXTURE,
        excluded_by=_TURN_ORACLE,
    ),
    _case(
        "m1.era.no-sample-at-closure",
        "sci004.section-3.1.virtual-closure-point",
        "test_there_is_no_sample_at_the_virtual_closure_point",
        _FULL_WIDTH_FIXTURE,
        excluded_by=_TURN_ORACLE,
    ),
    _case(
        "m1.era.nontrivial-ieee-ratio",
        "sci004.section-3.1.exact-ieee-integration-fraction",
        "test_a_nontrivial_binary64_fraction_enters_as_its_exact_ieee_ratio",
        _NONTRIVIAL_FIXTURE,
        excluded_by=_RATIO_ORACLE,
    ),
    _case(
        "m1.era.single-final-rn",
        "sci004.section-3.1.one-final-round-to-nearest",
        "test_every_derived_radian_is_one_final_round_to_nearest",
        _NONTRIVIAL_FIXTURE,
        excluded_by=_TURN_ORACLE,
    ),
    _case(
        "m1.era.rounded-closure-is-not-authority",
        "sci004.section-3.1.exact-turn-closure-authority",
        "test_rounded_horizon_endpoints_are_not_the_closure_authority",
        _FULL_WIDTH_FIXTURE,
        excluded_by=_TURN_ORACLE,
    ),
    _case(
        "m1.era.turn-grid-schema",
        "sci004.section-3.1.canonical-era-turn-grid-schema",
        "test_the_canonical_turn_grid_object_carries_its_exact_key_order",
        _FULL_WIDTH_FIXTURE,
        excluded_by=_TURN_ORACLE,
    ),
    _case(
        "m1.era.radian-grid-schema",
        "sci004.section-3.1.canonical-era-grid-schema",
        "test_the_canonical_radian_grid_object_carries_its_exact_key_order",
        _FULL_WIDTH_FIXTURE,
        excluded_by=_TURN_ORACLE,
    ),
    _case(
        "m1.era.component-digests",
        "sci004.section-3.1.eight-component-digests",
        "test_the_grid_publishes_all_eight_component_digests",
        _FULL_WIDTH_FIXTURE,
        excluded_by=_TURN_ORACLE,
    ),
    _case(
        "m1.era.utc-ut1-round-trip",
        "sci004.section-3.1.ut1-utc-round-trip-tolerance",
        "test_the_utc_ut1_round_trip_stays_inside_the_frozen_tolerance",
        _FULL_WIDTH_FIXTURE,
        excluded_by=(
            "tests/unit/test_core/test_sci004_frame.py::"
            "test_the_bundled_iers_resource_resolves_and_hashes_today"
        ),
    ),
    _case(
        "m1.era.analytic-dft",
        "sci004.section-6.normalized-dft-and-synthesis",
        "test_the_normalized_transform_reproduces_analytic_single_modes",
        _TRANSFORM_FIXTURE,
        excluded_by=_DFT_ORACLE,
    ),
    _case(
        "m1.era.exposure-sinc",
        "sci004.section-6.exposure-top-hat-sinc",
        "test_the_exposure_sinc_factor_follows_the_exact_turn_width",
        _TRANSFORM_FIXTURE,
        excluded_by=_DFT_ORACLE,
    ),
    _case(
        "m1.era.collapsed-exposure-edges",
        "sci004.section-3.1.mmode_exposure_resolution",
        "test_collapsed_binary64_exposure_edges_raise_the_typed_rejection",
        _COLLAPSED_FIXTURE,
        excluded_by=_RATIO_ORACLE,
    ),
)

SCI004_RED_GREEN_CONTROLS: tuple[str, ...] = (
    _RATIO_ORACLE,
    _TURN_ORACLE,
    _DFT_ORACLE,
)


# --- independent oracles, evaluated in the test body --------------------------


def _exact_turns(samples: int, fraction: float) -> dict[str, Any]:
    """Section 3.1's construction, re-derived here in exact rational arithmetic."""
    numerator, denominator = fraction.as_integer_ratio()
    centers = [Fraction(2 * k, 2 * samples) for k in range(samples)]
    lower = [
        Fraction(2 * k * denominator - numerator, 2 * samples * denominator)
        for k in range(samples)
    ]
    upper = [
        Fraction(2 * k * denominator + numerator, 2 * samples * denominator)
        for k in range(samples)
    ]
    return {
        "centers": centers,
        "lower": lower,
        "upper": upper,
        "width": Fraction(numerator, samples * denominator),
        "horizon_lo": Fraction(-1, 2 * samples),
        "horizon_hi": Fraction(2 * samples - 1, 2 * samples),
    }


def _canonical_ratio(value: Fraction) -> str:
    """Section 3.1's serialized ``p/q`` form: shortest ASCII, positive ``q``."""
    return f"{value.numerator}/{value.denominator}"


def _round_to_nearest(exact: Fraction) -> float:
    """One correctly rounded binary64 view of ``exact(tau) * u``, no intermediates."""
    return float(Fraction(*TAU.as_integer_ratio()) * exact)


# --- green controls -----------------------------------------------------------


def test_the_fixture_fraction_reconstructs_its_exact_ieee_ratio() -> None:
    """Family 1's nontrivial fraction: the decimal spelling is not an input.

    ``0.3`` is not three tenths. Its exact IEEE-754 value is
    ``5404319552844595 / 2**54``, and Section 3.1 requires the grid to decode
    *that* ratio. This control proves the oracle arithmetic below is sound
    before any production module exists to compare against.
    """
    numerator, denominator = NONTRIVIAL_FRACTION.as_integer_ratio()

    assert (numerator, denominator) == (5404319552844595, 18014398509481984)
    assert denominator == 2**54
    assert Fraction(numerator, denominator) != Fraction(3, 10)
    assert float(Fraction(numerator, denominator)) == NONTRIVIAL_FRACTION
    assert math.gcd(abs(numerator), denominator) == 1
    assert 0 < numerator <= denominator

    # ``tau`` is the exact binary64 nearest two pi, spelled without decimals.
    assert TAU == float.fromhex("0x1.921fb54442d18p+2")
    assert TAU.hex() == "0x1.921fb54442d18p+2"


def test_the_exact_turn_construction_is_reproduced_in_the_test_body() -> None:
    """Section 3.1's invariants hold for the oracle at ``N = 17``, ``f = 1``.

    This is the fixture-defect exclusion for every exact-turn red node: the
    construction the production grid must reproduce is proved sound here in
    ``fractions.Fraction`` alone.
    """
    grid = _exact_turns(EXACT_TURN_SAMPLES, EXACT_TURN_FRACTION)

    assert grid["horizon_hi"] - grid["horizon_lo"] == Fraction(1, 1)
    for index in range(EXACT_TURN_SAMPLES):
        assert grid["horizon_lo"] <= grid["lower"][index]
        assert grid["lower"][index] < grid["centers"][index]
        assert grid["centers"][index] < grid["upper"][index]
        assert grid["upper"][index] <= grid["horizon_hi"]
        assert grid["upper"][index] - grid["lower"][index] == grid["width"]
    for index in range(EXACT_TURN_SAMPLES - 1):
        assert grid["upper"][index] == grid["lower"][index + 1]
    assert grid["lower"][0] == grid["horizon_lo"]
    assert grid["upper"][EXACT_TURN_SAMPLES - 1] == grid["horizon_hi"]
    assert Fraction(1, 1) not in grid["centers"]
    assert _canonical_ratio(grid["horizon_lo"]) == "-1/34"
    assert _canonical_ratio(grid["centers"][0]) == "0/1"

    # The closure authority is exact-turn equality. Whether the rounded endpoint
    # difference happens to equal ``tau`` is a property of this ``N``, not a
    # predicate -- Section 14.2 makes asserting it a validation failure. What is
    # always true is that radians cannot *own* the topology: dividing a derived
    # radian by binary64 ``tau`` does not return its exact turn, and
    # accumulating a binary64 step does not reproduce the one-final-RN centres.
    recovered = [
        Fraction(*(_round_to_nearest(turn) / TAU).as_integer_ratio())
        for turn in grid["centers"]
    ]
    assert recovered != grid["centers"]

    step = _round_to_nearest(Fraction(1, EXACT_TURN_SAMPLES))
    accumulated = 0.0
    lifted: list[float] = []
    for _ in range(EXACT_TURN_SAMPLES):
        lifted.append(accumulated)
        accumulated += step
    assert lifted != [_round_to_nearest(turn) for turn in grid["centers"]]


def test_the_analytic_single_mode_dft_oracle_closes_in_the_test_body() -> None:
    """Section 6's normalized pair and exposure ``sinc``, proved on the oracle.

    ``bar_v_m = (1/N) sum_k bar_V_k exp(-i 2 pi m u_k)`` and
    ``bar_V_k = sum_m bar_v_m exp(+i 2 pi m u_k)`` are an exact analytic pair for
    a single retained mode, and ``w_0 = 1`` while ``w_m`` is the top-hat sinc.
    """
    samples = EXACT_TURN_SAMPLES
    turns = [Fraction(2 * k, 2 * samples) for k in range(samples)]
    for mode in (-3, 0, 5):
        series = [
            complex(math.cos(TAU * float(u) * mode), math.sin(TAU * float(u) * mode))
            for u in turns
        ]
        coefficient = (
            sum(
                value
                * complex(
                    math.cos(-TAU * float(u) * mode), math.sin(-TAU * float(u) * mode)
                )
                for value, u in zip(series, turns, strict=True)
            )
            / samples
        )
        assert abs(coefficient - 1.0) <= ANALYTIC_RESIDUAL_LIMIT, mode

    width = Fraction(1, 2 * samples)
    for mode in (0, 1, -4):
        argument = math.pi * mode * float(width)
        expected = 1.0 if mode == 0 else math.sin(argument) / argument
        assert math.isfinite(expected)
        assert abs(expected) <= 1.0 + ANALYTIC_RESIDUAL_LIMIT, mode


# --- Section 3.1 / 12.2 family 1 red oracles ----------------------------------


def _build(samples: int, fraction: float) -> Any:
    from radiosim.core.mmode.time import build_canonical_era_grid

    return build_canonical_era_grid(
        sidereal_samples=samples,
        integration_fraction=fraction,
        start_time=START_TIME,
    )


def test_the_exact_turn_grid_has_unit_width_and_full_containment() -> None:
    """Section 3.1: exact rational comparison proves containment and width."""
    grid = _build(EXACT_TURN_SAMPLES, EXACT_TURN_FRACTION)
    oracle = _exact_turns(EXACT_TURN_SAMPLES, EXACT_TURN_FRACTION)

    assert grid.horizon_lo_turn == _canonical_ratio(oracle["horizon_lo"])
    assert grid.horizon_hi_turn == _canonical_ratio(oracle["horizon_hi"])
    assert Fraction(grid.horizon_hi_turn) - Fraction(grid.horizon_lo_turn) == Fraction(
        1, 1
    )
    assert grid.center_turns == tuple(
        _canonical_ratio(value) for value in oracle["centers"]
    )
    assert grid.lower_edge_turns == tuple(
        _canonical_ratio(value) for value in oracle["lower"]
    )
    assert grid.upper_edge_turns == tuple(
        _canonical_ratio(value) for value in oracle["upper"]
    )
    assert grid.exposure_width_turn == _canonical_ratio(oracle["width"])


def test_full_width_adjacent_edges_are_the_same_exact_rational() -> None:
    """Section 3.1: at ``f == 1`` the shared edge is one normalized rational."""
    grid = _build(EXACT_TURN_SAMPLES, EXACT_TURN_FRACTION)

    for index in range(EXACT_TURN_SAMPLES - 1):
        assert grid.upper_edge_turns[index] == grid.lower_edge_turns[index + 1]
        # Same rational in, same rounded radian out: bit-identical, not close.
        assert grid.upper_rad[index] == grid.lower_rad[index + 1]
    assert grid.lower_edge_turns[0] == grid.horizon_lo_turn
    assert grid.upper_edge_turns[-1] == grid.horizon_hi_turn


def test_there_is_no_sample_at_the_virtual_closure_point() -> None:
    """Section 3.1: ``u = 1`` exists only as the virtual closure point."""
    grid = _build(EXACT_TURN_SAMPLES, EXACT_TURN_FRACTION)

    assert "1/1" not in grid.center_turns
    assert all(Fraction(turn) < Fraction(1, 1) for turn in grid.center_turns)
    assert len(grid.center_turns) == EXACT_TURN_SAMPLES


def test_a_nontrivial_binary64_fraction_enters_as_its_exact_ieee_ratio() -> None:
    """Section 3.1: the decimal spelling is not an arithmetic input."""
    grid = _build(EXACT_TURN_SAMPLES, NONTRIVIAL_FRACTION)
    numerator, denominator = NONTRIVIAL_FRACTION.as_integer_ratio()

    assert grid.canonical_era_turn_grid["integration_fraction_ratio"] == (
        f"{numerator}/{denominator}"
    )
    assert grid.exposure_width_turn == _canonical_ratio(
        Fraction(numerator, EXACT_TURN_SAMPLES * denominator)
    )
    assert grid.canonical_era_turn_grid["integration_fraction_f64be"] == (
        struct.pack(">d", NONTRIVIAL_FRACTION).hex()
    )


def test_every_derived_radian_is_one_final_round_to_nearest() -> None:
    """Section 3.1: one final ``RN``, and no intermediate binary64 arithmetic."""
    grid = _build(EXACT_TURN_SAMPLES, NONTRIVIAL_FRACTION)
    oracle = _exact_turns(EXACT_TURN_SAMPLES, NONTRIVIAL_FRACTION)

    for index, center in enumerate(oracle["centers"]):
        assert grid.alpha_rad[index] == _round_to_nearest(center)
    for index, lower in enumerate(oracle["lower"]):
        assert grid.lower_rad[index] == _round_to_nearest(lower)
    for index, upper in enumerate(oracle["upper"]):
        assert grid.upper_rad[index] == _round_to_nearest(upper)
    assert grid.delta_alpha_rad == _round_to_nearest(oracle["width"])
    assert grid.horizon_lo_rad == _round_to_nearest(oracle["horizon_lo"])
    assert grid.horizon_hi_rad == _round_to_nearest(oracle["horizon_hi"])
    for index in range(EXACT_TURN_SAMPLES):
        assert grid.lower_rad[index] < grid.alpha_rad[index] < grid.upper_rad[index]


def test_rounded_horizon_endpoints_are_not_the_closure_authority() -> None:
    """Section 3.1/14.2: rounded-radian subtraction never owns the topology.

    A consumer may not lift a turn by adding binary64 ``tau`` or recover the
    topology by dividing radians by ``tau``. The grid must therefore *not*
    publish an exact rounded-endpoint identity, and must still close exactly.
    """
    grid = _build(EXACT_TURN_SAMPLES, EXACT_TURN_FRACTION)

    assert Fraction(grid.horizon_hi_turn) - Fraction(grid.horizon_lo_turn) == Fraction(
        1, 1
    )
    assert grid.canonical_era_grid["tau_f64be"] == struct.pack(">d", TAU).hex()

    # Radians are a derived view, so neither of the two shortcuts a consumer
    # might reach for recovers the exact grid.
    recovered = [
        Fraction(*(value / TAU).as_integer_ratio()) for value in grid.alpha_rad
    ]
    assert recovered != [Fraction(turn) for turn in grid.center_turns]
    step = grid.delta_alpha_rad
    accumulated = 0.0
    lifted: list[float] = []
    for _ in range(EXACT_TURN_SAMPLES):
        lifted.append(accumulated)
        accumulated += step
    assert lifted != list(grid.alpha_rad)


def test_the_canonical_turn_grid_object_carries_its_exact_key_order() -> None:
    """Section 3.1: the embedded turn grid has exactly these ten keys."""
    grid = _build(EXACT_TURN_SAMPLES, EXACT_TURN_FRACTION)
    embedded = grid.canonical_era_turn_grid

    assert tuple(embedded) == TURN_GRID_KEYS
    assert embedded["schema_version"] == TURN_GRID_SCHEMA
    assert embedded["sidereal_samples"] == EXACT_TURN_SAMPLES
    assert len(embedded["center_turns"]) == EXACT_TURN_SAMPLES
    assert len(embedded["lower_edge_turns"]) == EXACT_TURN_SAMPLES
    assert len(embedded["upper_edge_turns"]) == EXACT_TURN_SAMPLES


def test_the_canonical_radian_grid_object_carries_its_exact_key_order() -> None:
    """Section 3.1: the embedded radian grid has exactly these twelve keys."""
    grid = _build(EXACT_TURN_SAMPLES, EXACT_TURN_FRACTION)
    embedded = grid.canonical_era_grid

    assert tuple(embedded) == RADIAN_GRID_KEYS
    assert embedded["schema_version"] == RADIAN_GRID_SCHEMA


def test_the_grid_publishes_all_eight_component_digests() -> None:
    """Section 3.1: three turn arrays, three radian arrays and both grid objects."""
    grid = _build(EXACT_TURN_SAMPLES, EXACT_TURN_FRACTION)

    for field in GRID_DIGEST_FIELDS:
        digest = getattr(grid, field)
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)
    assert len({getattr(grid, field) for field in GRID_DIGEST_FIELDS}) == 8


def test_the_utc_ut1_round_trip_stays_inside_the_frozen_tolerance() -> None:
    """Section 3.1: the round trip is a fixed ``1e-6 s`` constant, not a knob."""
    grid = _build(EXACT_TURN_SAMPLES, EXACT_TURN_FRACTION)

    assert grid.ut1_utc_roundtrip_seconds <= UT1_UTC_ROUNDTRIP_LIMIT_SECONDS
    assert grid.era_center_max_residual_rad <= ERA_CENTER_LIMIT_RAD
    assert grid.era_step_max_residual_rad <= ERA_STEP_LIMIT_RAD
    assert len(grid.utc_two_part) == 2
    assert len(grid.ut1_two_part) == 2


def test_the_normalized_transform_reproduces_analytic_single_modes() -> None:
    """Section 6/12.2 family 1: exact sign and normalization for +m, -m and 0."""
    from radiosim.core.mmode.time import forward_m_transform, synthesize_time_series

    grid = _build(EXACT_TURN_SAMPLES, 0.5)
    for mode in (-3, 0, 5):
        series = [
            complex(
                math.cos(TAU * float(Fraction(u)) * mode),
                math.sin(TAU * float(Fraction(u)) * mode),
            )
            for u in grid.center_turns
        ]
        coefficients = forward_m_transform(grid, series, mmax=5)
        assert abs(coefficients[mode] - 1.0) <= ANALYTIC_RESIDUAL_LIMIT
        rebuilt = synthesize_time_series(grid, coefficients)
        residual = max(abs(a - b) for a, b in zip(rebuilt, series, strict=True))
        assert residual <= ANALYTIC_RESIDUAL_LIMIT


def test_the_exposure_sinc_factor_follows_the_exact_turn_width() -> None:
    """Section 6: ``w_m = sinc(pi m Delta_u)`` with ``w_0`` exactly one."""
    from radiosim.core.mmode.time import exposure_sinc_weights

    grid = _build(EXACT_TURN_SAMPLES, 0.5)
    weights = exposure_sinc_weights(grid, mmax=5)
    width = Fraction(grid.exposure_width_turn)

    assert weights[0] == 1.0
    for mode in (1, -4, 5):
        argument = math.pi * mode * float(width)
        assert abs(weights[mode] - math.sin(argument) / argument) <= (
            ANALYTIC_RESIDUAL_LIMIT
        )


def test_collapsed_binary64_exposure_edges_raise_the_typed_rejection() -> None:
    """Section 3.1/8: the constructor raises ``mmode_exposure_resolution``."""
    from radiosim.io.config_resolution import ConfigSemanticError

    with pytest.raises(ConfigSemanticError) as excinfo:
        _build(257, 1.0e-308)

    codes = [issue.code for issue in excinfo.value.issues]
    assert "mmode_exposure_resolution" in codes
