"""Tier 5C receptor-configuration Jones (term ``C``) mathematics.

Oracles are transcribed from ``Tier5ReceptorFeedPlan.md`` Sections 18.1, 18.2,
18.4, 18.5, and 18.6.  Invariants covered: the identity linear-to-linear case,
the analytic circular transform, S4, S5, S6, S7, S8.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.core.instrument import AntennaFieldSource, AntennaId
from radiosim.core.instrument_adapters import (
    InstrumentAdapterInvariantError,
    SolverInstrumentView,
)
from radiosim.core.jones.base import JonesTerm
from radiosim.core.jones.directions import DirectionBatch
from radiosim.core.jones.receptor import (
    LINEAR_TO_CIRCULAR,
    ReceptorConfigJones,
    basis_rotation_matrix,
    receptor_matrix,
)
from radiosim.core.polarization import stokes_to_coherency
from radiosim.core.receptor import (
    ReceptorAssignmentError,
    ReceptorProvenance,
    ResolvedReceptor,
    ResolvedReceptorSet,
    UnsupportedReceptorBasisError,
    _compute_receptor_sha256,
)

IDENTITY = np.eye(2, dtype=np.complex128)

PLAN_S_MATRIX = (1.0 / np.sqrt(2.0)) * np.array(
    [[1.0, 1.0j], [1.0, -1.0j]],
    dtype=np.complex128,
)

ROTATIONS_DEG = (0.0, 30.0, 45.0, 90.0, -15.0)


def plan_rotation(chi_rad: float) -> np.ndarray:
    """Return the Section 18.1 rotation ``R(chi)``, written out from the plan."""
    return np.array(
        [
            [math.cos(chi_rad), math.sin(chi_rad)],
            [-math.sin(chi_rad), math.cos(chi_rad)],
        ],
        dtype=np.complex128,
    )


def plan_receptor_matrix(basis: str, chi_rad: float) -> np.ndarray:
    """Return the Section 18.2 matrix ``C_p = M(basis) @ R(chi)``."""
    leading = IDENTITY if basis == "linear" else PLAN_S_MATRIX
    return leading @ plan_rotation(chi_rad)


# ---------------------------------------------------------------------------
# Fixtures built directly from the Tier 5B resolved models
# ---------------------------------------------------------------------------


def _feed_angles(basis: str, chi_rad: float) -> tuple[float, float]:
    if basis == "linear":
        return (math.pi / 2.0 + chi_rad, chi_rad)
    return (chi_rad, chi_rad)


def make_receptor_set(
    specification: tuple[tuple[str, float], ...],
    output_basis: str,
) -> ResolvedReceptorSet:
    """Build a resolved receptor set from ``(basis, rotation_deg)`` per antenna."""
    receptors: dict[AntennaId, ResolvedReceptor] = {}
    for index, (basis, rotation_deg) in enumerate(specification):
        chi_rad = math.radians(rotation_deg)
        antenna_id = AntennaId(index, f"ANT{index}")
        receptors[antenna_id] = ResolvedReceptor(
            basis=basis,  # type: ignore[arg-type]
            feed_rotation_rad=chi_rad,
            feed_array=("x", "y") if basis == "linear" else ("r", "l"),
            feed_angle_rad=_feed_angles(basis, chi_rad),
            source=AntennaFieldSource.CONFIG_DEFAULT,
        )
    provenance = ReceptorProvenance(
        schema_version="1.0.0",
        requested_output_basis="linear" if output_basis == "linear_xy" else "circular",
        output_basis_rule=(
            "explicit_linear" if output_basis == "linear_xy" else "explicit_circular"
        ),
        override_applications=(),
        receptor_sha256=_compute_receptor_sha256(output_basis, receptors),  # type: ignore[arg-type]
    )
    return ResolvedReceptorSet(
        output_basis=output_basis,  # type: ignore[arg-type]
        receptor_by_antenna=receptors,
        provenance=provenance,
    )


def make_instrument_view(count: int) -> SolverInstrumentView:
    """Build the minimal solver view the receptor terms resolve antennas against."""
    positions = np.array(
        [[14.0 * index, 0.0, 0.0] for index in range(count)],
        dtype=np.float64,
    )
    pairs = tuple(
        (first, second) for first in range(count) for second in range(first + 1, count)
    ) or ((0, 0),)  # a single-antenna view still needs one selected pair
    vectors = np.array(
        [positions[second] - positions[first] for first, second in pairs],
        dtype=np.float64,
    )
    return SolverInstrumentView(
        antenna_numbers=tuple(range(count)),
        antenna_names=tuple(f"ANT{index}" for index in range(count)),
        positions_enu_m=positions,
        diameters_m=np.full(count, 14.0, dtype=np.float64),
        row_index_by_number={number: number for number in range(count)},
        selected_pairs=pairs,
        baseline_vectors_enu_m=vectors,
    )


def direction_batch(n_dir: int = 3) -> DirectionBatch:
    """A direction batch that ``C`` and ``H`` must ignore entirely."""
    angles = np.linspace(0.1, 1.0, n_dir)
    return DirectionBatch(
        alt_rad=angles,
        az_rad=angles,
        dir_l=angles,
        dir_m=angles,
        dir_n=angles,
        ra_rad=angles,
        dec_rad=angles,
        hour_angle_rad=angles,
        n_dir=n_dir,
    )


def compute(term: JonesTerm, antenna_idx: int) -> np.ndarray:
    """Evaluate one direction-independent term on the numpy backend.

    Returns the ``(2, 2)`` matrix.  The direction-batched contract returns the
    mandated ``(1, 2, 2)`` broadcast form for a direction-independent term
    (invariant I3), which is asserted here once so that every caller can stay
    written in terms of the matrix itself.
    """
    batch = term.compute_jones_batch(
        antenna_idx=antenna_idx,
        directions=direction_batch(),
        frequency_hz=1.0e8,
        freq_idx=0,
        time_mjd=60_000.0,
        time_idx=0,
        backend=get_backend("numpy"),
        dtype=np.complex128,
    )
    matrix = np.asarray(batch)
    assert matrix.shape == (1, 2, 2)
    return matrix[0]


# ---------------------------------------------------------------------------
# Identity for the default linear array
# ---------------------------------------------------------------------------


def test_default_linear_receptors_yield_the_exact_identity() -> None:
    """Required-test matrix: C is exactly I2 for linear / chi = 0 / linear_xy."""
    receptors = make_receptor_set((("linear", 0.0),) * 3, "linear_xy")
    term = ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(3))

    for antenna_idx in range(3):
        np.testing.assert_array_equal(compute(term, antenna_idx), IDENTITY)


def test_default_linear_receptors_report_diagonal_and_scalar_hints() -> None:
    receptors = make_receptor_set((("linear", 0.0),) * 3, "linear_xy")
    term = ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(3))

    assert term.is_unitary() is True
    assert term.is_diagonal() is True
    assert term.is_scalar() is True


def test_term_metadata_matches_section_18_2() -> None:
    receptors = make_receptor_set((("circular", 0.0),), "circular_rl")
    term = ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(1))

    assert term.name == "C"
    assert term.is_direction_dependent is False
    assert term.is_time_dependent is False
    assert term.is_frequency_dependent is False
    assert term.is_baseline_dependent is False


@pytest.mark.parametrize(
    ("specification", "expected_diagonal"),
    (
        ((("linear", 0.0),), True),
        ((("linear", 30.0),), False),
        ((("circular", 0.0),), False),
        ((("circular", 45.0),), False),
        ((("linear", 0.0), ("linear", 15.0)), False),
    ),
)
def test_diagonal_hint_is_true_only_for_unrotated_linear_receptors(
    specification,
    expected_diagonal: bool,
) -> None:
    """Section 18.2: is_diagonal() is True only for basis == linear and chi == 0."""
    output_basis = "linear_xy" if specification[0][0] == "linear" else "circular_rl"
    receptors = make_receptor_set(specification, output_basis)
    term = ReceptorConfigJones(
        receptors=receptors,
        instrument=make_instrument_view(len(specification)),
    )

    assert term.is_diagonal() is expected_diagonal
    assert term.is_scalar() is expected_diagonal
    # Unitarity is unconditional, and now truthfully so.
    assert term.is_unitary() is True


# ---------------------------------------------------------------------------
# Analytic transforms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rotation_deg", ROTATIONS_DEG)
def test_the_rotation_building_block_matches_section_18_1(rotation_deg: float) -> None:
    """R(chi) == [[cos, sin], [-sin, cos]], elementwise."""
    np.testing.assert_allclose(
        basis_rotation_matrix(math.radians(rotation_deg)),
        plan_rotation(math.radians(rotation_deg)),
        rtol=0.0,
        atol=1e-15,
    )


def test_the_exported_basis_matrix_constant_matches_section_18_1() -> None:
    """The module constant S is the plan's matrix, and it is unitary."""
    np.testing.assert_allclose(LINEAR_TO_CIRCULAR, PLAN_S_MATRIX, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        LINEAR_TO_CIRCULAR @ LINEAR_TO_CIRCULAR.conj().T,
        IDENTITY,
        rtol=0.0,
        atol=1e-15,
    )


def test_the_basis_matrix_matches_section_18_1_elementwise() -> None:
    """S is (1/sqrt 2) [[1, i], [1, -i]] with rows (R, L) and columns (x, y)."""
    receptors = make_receptor_set((("circular", 0.0),), "circular_rl")
    term = ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(1))

    np.testing.assert_allclose(compute(term, 0), PLAN_S_MATRIX, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("basis", ("linear", "circular"))
@pytest.mark.parametrize("rotation_deg", ROTATIONS_DEG)
def test_receptor_matrix_equals_m_times_r(basis: str, rotation_deg: float) -> None:
    """Section 18.2: C_p == M(basis) @ R(chi_p), elementwise."""
    output_basis = "linear_xy" if basis == "linear" else "circular_rl"
    receptors = make_receptor_set(((basis, rotation_deg),), output_basis)
    term = ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(1))

    np.testing.assert_allclose(
        compute(term, 0),
        plan_receptor_matrix(basis, math.radians(rotation_deg)),
        rtol=0.0,
        atol=1e-15,
    )


def test_per_antenna_matrices_track_a_heterogeneous_array() -> None:
    """Each antenna row resolves its own basis and rotation, not the array's first."""
    specification = (
        ("linear", 0.0),
        ("circular", 0.0),
        ("linear", 45.0),
        ("circular", -15.0),
    )
    receptors = make_receptor_set(specification, "circular_rl")
    term = ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(4))

    for antenna_idx, (basis, rotation_deg) in enumerate(specification):
        np.testing.assert_allclose(
            compute(term, antenna_idx),
            plan_receptor_matrix(basis, math.radians(rotation_deg)),
            rtol=0.0,
            atol=1e-15,
        )


def test_matrices_are_unitary_for_every_accepted_combination() -> None:
    """S6 for the C factor alone: C^H C == C C^H == I2."""
    for basis in ("linear", "circular"):
        for rotation_deg in ROTATIONS_DEG:
            output_basis = "linear_xy" if basis == "linear" else "circular_rl"
            receptors = make_receptor_set(((basis, rotation_deg),), output_basis)
            term = ReceptorConfigJones(
                receptors=receptors,
                instrument=make_instrument_view(1),
            )
            matrix = compute(term, 0)
            np.testing.assert_allclose(
                matrix.conj().T @ matrix, IDENTITY, rtol=0.0, atol=1e-15
            )
            np.testing.assert_allclose(
                matrix @ matrix.conj().T, IDENTITY, rtol=0.0, atol=1e-15
            )


# ---------------------------------------------------------------------------
# The Section 18.4 correlation oracle, driven through the real term
# ---------------------------------------------------------------------------


REFERENCE_STOKES = (
    (10.0, 0.0, 0.0, 0.0),
    (10.0, 10.0, 0.0, 0.0),
    (10.0, 0.0, 10.0, 0.0),
    (10.0, 0.0, 0.0, 10.0),
    (7.5, -1.25, 3.0, -2.0),
)


@pytest.mark.parametrize("stokes", REFERENCE_STOKES)
def test_linear_output_reproduces_the_section_18_4_linear_table(stokes) -> None:
    stokes_i, stokes_q, stokes_u, stokes_v = stokes
    receptors = make_receptor_set((("linear", 0.0), ("linear", 0.0)), "linear_xy")
    term = ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(2))
    coherency = np.asarray(stokes_to_coherency(*stokes))

    jones_p = compute(term, 0)
    jones_q = compute(term, 1)
    visibility = jones_p @ coherency @ jones_q.conj().T

    assert visibility[0, 0] == pytest.approx((stokes_i + stokes_q) / 2.0)
    assert visibility[0, 1] == pytest.approx((stokes_u + 1j * stokes_v) / 2.0)
    assert visibility[1, 0] == pytest.approx((stokes_u - 1j * stokes_v) / 2.0)
    assert visibility[1, 1] == pytest.approx((stokes_i - stokes_q) / 2.0)


@pytest.mark.parametrize("stokes", REFERENCE_STOKES)
def test_circular_output_reproduces_the_section_18_4_circular_table(stokes) -> None:
    """S4 through the production term: RR=(I+V)/2, RL=(Q+iU)/2, LR, LL."""
    stokes_i, stokes_q, stokes_u, stokes_v = stokes
    receptors = make_receptor_set((("circular", 0.0), ("circular", 0.0)), "circular_rl")
    term = ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(2))
    coherency = np.asarray(stokes_to_coherency(*stokes))

    jones_p = compute(term, 0)
    jones_q = compute(term, 1)
    visibility = jones_p @ coherency @ jones_q.conj().T

    assert visibility[0, 0] == pytest.approx((stokes_i + stokes_v) / 2.0, abs=1e-14)
    assert visibility[0, 1] == pytest.approx(
        (stokes_q + 1j * stokes_u) / 2.0, abs=1e-14
    )
    assert visibility[1, 0] == pytest.approx(
        (stokes_q - 1j * stokes_u) / 2.0, abs=1e-14
    )
    assert visibility[1, 1] == pytest.approx((stokes_i - stokes_v) / 2.0, abs=1e-14)


def test_a_positive_v_source_emerges_as_pure_rr_through_the_term() -> None:
    """The corrected V sign, observed through the real receptor matrix."""
    total_flux = 4.0
    receptors = make_receptor_set((("circular", 0.0), ("circular", 0.0)), "circular_rl")
    term = ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(2))
    coherency = np.asarray(stokes_to_coherency(total_flux, 0.0, 0.0, total_flux))

    visibility = compute(term, 0) @ coherency @ compute(term, 1).conj().T

    assert visibility[0, 0].real == pytest.approx(total_flux, abs=1e-14)
    assert visibility[1, 1].real == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("basis", ("linear", "circular"))
@pytest.mark.parametrize("rotation_deg", ROTATIONS_DEG)
def test_unpolarized_energy_conservation_through_the_term(
    basis: str,
    rotation_deg: float,
) -> None:
    """S5 through the production term, in both bases and every rotation."""
    stokes_i = 9.0
    output_basis = "linear_xy" if basis == "linear" else "circular_rl"
    receptors = make_receptor_set(
        ((basis, rotation_deg), (basis, rotation_deg)),
        output_basis,
    )
    term = ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(2))
    coherency = np.asarray(stokes_to_coherency(stokes_i, 0.0, 0.0, 0.0))

    visibility = compute(term, 0) @ coherency @ compute(term, 1).conj().T

    assert visibility[0, 0].real + visibility[1, 1].real == pytest.approx(
        stokes_i, abs=1e-13
    )
    assert abs(visibility[0, 1]) == pytest.approx(0.0, abs=1e-14)
    assert abs(visibility[1, 0]) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("rotation_deg", ROTATIONS_DEG)
def test_linear_rotation_rotates_q_and_u_by_twice_chi_through_the_term(
    rotation_deg: float,
) -> None:
    """S7 through the production term."""
    from radiosim.core.polarization import coherency_to_stokes

    stokes_i, stokes_q, stokes_u, stokes_v = 7.5, -1.25, 3.0, -2.0
    receptors = make_receptor_set(
        (("linear", rotation_deg), ("linear", rotation_deg)),
        "linear_xy",
    )
    term = ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(2))
    coherency = np.asarray(stokes_to_coherency(stokes_i, stokes_q, stokes_u, stokes_v))

    visibility = compute(term, 0) @ coherency @ compute(term, 1).conj().T
    recovered_i, recovered_q, recovered_u, recovered_v = coherency_to_stokes(visibility)

    chi_rad = math.radians(rotation_deg)
    cos_2chi = math.cos(2.0 * chi_rad)
    sin_2chi = math.sin(2.0 * chi_rad)
    assert recovered_i == pytest.approx(stokes_i, abs=1e-13)
    assert recovered_v == pytest.approx(stokes_v, abs=1e-13)
    assert recovered_q == pytest.approx(
        stokes_q * cos_2chi + stokes_u * sin_2chi, abs=1e-13
    )
    assert recovered_u == pytest.approx(
        -stokes_q * sin_2chi + stokes_u * cos_2chi, abs=1e-13
    )


@pytest.mark.parametrize("rotation_deg", ROTATIONS_DEG)
def test_circular_rotation_phases_only_the_cross_hands_through_the_term(
    rotation_deg: float,
) -> None:
    """S8 through the production term."""
    stokes = (7.5, -1.25, 3.0, -2.0)
    coherency = np.asarray(stokes_to_coherency(*stokes))
    view = make_instrument_view(2)

    unrotated_term = ReceptorConfigJones(
        receptors=make_receptor_set(
            (("circular", 0.0), ("circular", 0.0)),
            "circular_rl",
        ),
        instrument=view,
    )
    rotated_term = ReceptorConfigJones(
        receptors=make_receptor_set(
            (("circular", rotation_deg), ("circular", rotation_deg)),
            "circular_rl",
        ),
        instrument=view,
    )

    unrotated = (
        compute(unrotated_term, 0) @ coherency @ compute(unrotated_term, 1).conj().T
    )
    rotated = compute(rotated_term, 0) @ coherency @ compute(rotated_term, 1).conj().T

    chi_rad = math.radians(rotation_deg)
    assert rotated[0, 0] == pytest.approx(unrotated[0, 0], abs=1e-13)
    assert rotated[1, 1] == pytest.approx(unrotated[1, 1], abs=1e-13)
    assert rotated[0, 1] == pytest.approx(
        np.exp(-2j * chi_rad) * unrotated[0, 1], abs=1e-13
    )
    assert rotated[1, 0] == pytest.approx(
        np.exp(+2j * chi_rad) * unrotated[1, 0], abs=1e-13
    )


# ---------------------------------------------------------------------------
# Constructor and error contract (Sections 24 and 25.1)
# ---------------------------------------------------------------------------


def test_the_permissive_stub_constructor_is_removed() -> None:
    """Section 24: feed_type= must raise a TypeError naming the replacement."""
    receptors = make_receptor_set((("linear", 0.0),), "linear_xy")
    view = make_instrument_view(1)

    with pytest.raises(TypeError) as excinfo:
        ReceptorConfigJones(feed_type="circular")  # type: ignore[call-arg]
    assert "receptors" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        ReceptorConfigJones(  # type: ignore[call-arg]
            receptors=receptors,
            instrument=view,
            feed_type="circular",
        )
    assert "feed_type" in str(excinfo.value)
    assert "receptors" in str(excinfo.value)

    with pytest.raises(TypeError):
        ReceptorConfigJones(receptors, view)  # type: ignore[misc]


def test_construction_rejects_wrong_argument_types() -> None:
    receptors = make_receptor_set((("linear", 0.0),), "linear_xy")
    view = make_instrument_view(1)

    with pytest.raises(TypeError):
        ReceptorConfigJones(receptors=object(), instrument=view)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ReceptorConfigJones(receptors=receptors, instrument=object())  # type: ignore[arg-type]


def test_an_antenna_absent_from_the_receptor_set_is_a_typed_failure() -> None:
    """Section 25.1: a missing assignment raises ReceptorAssignmentError."""
    receptors = make_receptor_set((("linear", 0.0),), "linear_xy")

    with pytest.raises(ReceptorAssignmentError) as excinfo:
        ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(2))
    assert "ANT1" in str(excinfo.value)


def test_an_out_of_range_antenna_row_is_rejected() -> None:
    receptors = make_receptor_set((("linear", 0.0),), "linear_xy")
    term = ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(1))

    with pytest.raises(InstrumentAdapterInvariantError):
        compute(term, 5)
    with pytest.raises(InstrumentAdapterInvariantError):
        compute(term, True)  # type: ignore[arg-type]


def test_an_unsupported_basis_reaching_the_matrix_builder_is_a_typed_failure() -> None:
    """Section 25.1: an unimplemented basis raises UnsupportedReceptorBasisError."""
    for basis in ("elliptical", "stokes", "LINEAR", ""):
        with pytest.raises(UnsupportedReceptorBasisError) as excinfo:
            receptor_matrix(basis, 0.0)
        assert basis in str(excinfo.value) or "supported" in str(excinfo.value)


@pytest.mark.parametrize("basis", ("linear", "circular"))
@pytest.mark.parametrize("rotation_deg", ROTATIONS_DEG)
def test_the_public_matrix_builder_matches_the_plan(
    basis: str, rotation_deg: float
) -> None:
    """The shared builder is the one place ``M(basis) @ R(chi)`` is written."""
    np.testing.assert_allclose(
        receptor_matrix(basis, math.radians(rotation_deg)),
        plan_receptor_matrix(basis, math.radians(rotation_deg)),
        rtol=0.0,
        atol=1e-15,
    )


def test_returned_matrices_do_not_alias_internal_state() -> None:
    receptors = make_receptor_set((("circular", 30.0),), "circular_rl")
    term = ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(1))

    first = compute(term, 0)
    first[0, 0] = 12345.0
    second = compute(term, 0)

    assert second[0, 0] != 12345.0


def test_get_config_reports_the_corrected_hints() -> None:
    receptors = make_receptor_set((("circular", 0.0),), "circular_rl")
    term = ReceptorConfigJones(receptors=receptors, instrument=make_instrument_view(1))

    config = term.get_config()

    assert config["name"] == "C"
    assert config["is_unitary"] is True
    assert config["is_diagonal"] is False
    assert config["is_direction_dependent"] is False
    assert config["is_frequency_dependent"] is False
