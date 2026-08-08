"""Tier 5C basis-transform Jones (term ``H``) mathematics.

Oracles transcribed from ``Tier5ReceptorFeedPlan.md`` Sections 18.1, 18.3, 18.6,
and 18.7.  Invariants covered: S6, S9, and the analytic exactness statement of
Section 11.3 that underpins S10.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from radiosim.core.instrument import AntennaId
from radiosim.core.instrument_adapters import InstrumentAdapterInvariantError
from radiosim.core.jones.receptor import (
    LINEAR_TO_CIRCULAR,
    BasisTransformJones,
    ReceptorConfigJones,
    basis_transform_matrix,
)
from radiosim.core.polarization import stokes_to_coherency
from radiosim.core.receptor import (
    ReceptorAssignmentError,
    UnsupportedBasisTransformError,
)
from tests.unit.test_jones.test_receptor import (
    IDENTITY,
    PLAN_P_MATRIX,
    PLAN_S_MATRIX,
    ROTATIONS_DEG,
    compute,
    make_instrument_view,
    make_receptor_set,
)

# Section 18.3, transcribed.
PLAN_TRANSFORMS = {
    ("linear", "linear_xy"): IDENTITY,
    ("circular", "circular_rl"): IDENTITY,
    ("linear", "circular_rl"): PLAN_S_MATRIX @ PLAN_P_MATRIX,
    ("circular", "linear_xy"): PLAN_P_MATRIX @ PLAN_S_MATRIX.conj().T,
}


# ---------------------------------------------------------------------------
# The Section 18.3 table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("native", "output"), tuple(PLAN_TRANSFORMS))
def test_transform_table_matches_section_18_3(native: str, output: str) -> None:
    np.testing.assert_allclose(
        basis_transform_matrix(native, output),
        PLAN_TRANSFORMS[(native, output)],
        rtol=0.0,
        atol=1e-15,
    )


@pytest.mark.parametrize(("native", "output"), tuple(PLAN_TRANSFORMS))
def test_the_term_returns_the_table_entry_for_each_antenna(
    native: str, output: str
) -> None:
    receptors = make_receptor_set(((native, 0.0), (native, 30.0)), output)
    term = BasisTransformJones(receptors=receptors, instrument=make_instrument_view(2))

    for antenna_idx in range(2):
        np.testing.assert_allclose(
            compute(term, antenna_idx),
            PLAN_TRANSFORMS[(native, output)],
            rtol=0.0,
            atol=1e-15,
        )


def test_h_is_independent_of_the_feed_rotation() -> None:
    """Section 18.3: H depends only on the two bases, never on chi."""
    view = make_instrument_view(1)
    reference = compute(
        BasisTransformJones(
            receptors=make_receptor_set((("circular", 0.0),), "linear_xy"),
            instrument=view,
        ),
        0,
    )
    for rotation_deg in ROTATIONS_DEG:
        term = BasisTransformJones(
            receptors=make_receptor_set((("circular", rotation_deg),), "linear_xy"),
            instrument=view,
        )
        np.testing.assert_array_equal(compute(term, 0), reference)


def test_term_metadata_matches_section_18_3() -> None:
    receptors = make_receptor_set((("linear", 0.0),), "circular_rl")
    term = BasisTransformJones(receptors=receptors, instrument=make_instrument_view(1))

    assert term.name == "H"
    assert term.is_direction_dependent is False
    assert term.is_time_dependent is False
    assert term.is_frequency_dependent is False
    assert term.is_unitary() is True


@pytest.mark.parametrize(
    ("native", "output", "expected_identity"),
    (
        ("linear", "linear_xy", True),
        ("circular", "circular_rl", True),
        ("linear", "circular_rl", False),
        ("circular", "linear_xy", False),
    ),
)
def test_diagonal_and_scalar_hints_track_the_identity_cases(
    native: str,
    output: str,
    expected_identity: bool,
) -> None:
    receptors = make_receptor_set(((native, 45.0),), output)
    term = BasisTransformJones(receptors=receptors, instrument=make_instrument_view(1))

    assert term.is_diagonal() is expected_identity
    assert term.is_scalar() is expected_identity
    assert term.is_unitary() is True


def test_a_heterogeneous_array_gets_per_antenna_transforms() -> None:
    """Section 13: mixed natives are brought into one explicit output basis."""
    specification = (("linear", 0.0), ("circular", 20.0), ("linear", -15.0))
    receptors = make_receptor_set(specification, "circular_rl")
    term = BasisTransformJones(receptors=receptors, instrument=make_instrument_view(3))

    np.testing.assert_allclose(
        compute(term, 0), PLAN_S_MATRIX @ PLAN_P_MATRIX, atol=1e-15
    )
    np.testing.assert_allclose(compute(term, 1), IDENTITY, atol=1e-15)
    np.testing.assert_allclose(
        compute(term, 2), PLAN_S_MATRIX @ PLAN_P_MATRIX, atol=1e-15
    )

    assert term.is_diagonal() is False
    assert term.is_scalar() is False


# ---------------------------------------------------------------------------
# S6 -- unitarity of every accepted (H, C) pair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("native", ("linear", "circular"))
@pytest.mark.parametrize("output", ("linear_xy", "circular_rl"))
@pytest.mark.parametrize("rotation_deg", ROTATIONS_DEG)
def test_h_times_c_is_unitary_for_every_accepted_combination(
    native: str,
    output: str,
    rotation_deg: float,
) -> None:
    """S6: (H C)^H (H C) == I2 for every (basis, chi, output_basis)."""
    receptors = make_receptor_set(((native, rotation_deg),), output)
    view = make_instrument_view(1)
    receptor_term = ReceptorConfigJones(receptors=receptors, instrument=view)
    transform_term = BasisTransformJones(receptors=receptors, instrument=view)

    combined = compute(transform_term, 0) @ compute(receptor_term, 0)

    np.testing.assert_allclose(
        combined.conj().T @ combined, IDENTITY, rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        combined @ combined.conj().T, IDENTITY, rtol=0.0, atol=1e-15
    )


def test_linear_to_circular_is_unitary_both_ways() -> None:
    """SCI-006: the exported linear-to-circular transform is ``S P``."""
    np.testing.assert_allclose(
        LINEAR_TO_CIRCULAR,
        PLAN_S_MATRIX @ PLAN_P_MATRIX,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        LINEAR_TO_CIRCULAR @ LINEAR_TO_CIRCULAR.conj().T,
        IDENTITY,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        LINEAR_TO_CIRCULAR.conj().T @ LINEAR_TO_CIRCULAR,
        IDENTITY,
        rtol=0.0,
        atol=1e-15,
    )


# ---------------------------------------------------------------------------
# S9 -- the round trip
# ---------------------------------------------------------------------------


def test_transform_round_trip_is_the_identity() -> None:
    """S9: T(linear->circular) @ T(circular->linear) == I2, and the reverse."""
    to_circular = basis_transform_matrix("linear", "circular_rl")
    to_linear = basis_transform_matrix("circular", "linear_xy")

    np.testing.assert_allclose(to_circular @ to_linear, IDENTITY, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(to_linear @ to_circular, IDENTITY, rtol=0.0, atol=1e-15)


def test_the_identity_transforms_are_exactly_the_identity() -> None:
    for native, output in (("linear", "linear_xy"), ("circular", "circular_rl")):
        np.testing.assert_array_equal(basis_transform_matrix(native, output), IDENTITY)


# ---------------------------------------------------------------------------
# Section 11.3 exactness -- the analytic core of S10
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stokes",
    (
        (10.0, 0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0, 10.0),
        (7.5, -1.25, 3.0, -2.0),
    ),
)
@pytest.mark.parametrize("rotation_deg", ROTATIONS_DEG)
def test_circular_native_into_linear_output_equals_linear_native(
    stokes,
    rotation_deg: float,
) -> None:
    """A change of representation, not of physics: H C is the same product.

    This is the per-antenna analytic core of S10; the full solver-level S10
    lands with Tier 5D's chain wiring.
    """
    view = make_instrument_view(2)
    coherency = np.asarray(stokes_to_coherency(*stokes))

    def visibility(native: str, output: str) -> np.ndarray:
        receptors = make_receptor_set(
            ((native, rotation_deg), (native, rotation_deg)),
            output,
        )
        receptor_term = ReceptorConfigJones(receptors=receptors, instrument=view)
        transform_term = BasisTransformJones(receptors=receptors, instrument=view)
        jones_p = compute(transform_term, 0) @ compute(receptor_term, 0)
        jones_q = compute(transform_term, 1) @ compute(receptor_term, 1)
        return jones_p @ coherency @ jones_q.conj().T

    np.testing.assert_allclose(
        visibility("circular", "linear_xy"),
        visibility("linear", "linear_xy"),
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        visibility("linear", "circular_rl"),
        visibility("circular", "circular_rl"),
        rtol=0.0,
        atol=1e-13,
    )


def test_the_collapse_identity_of_section_18_3_holds() -> None:
    """SCI-006: for circular output, H C == S R(chi) from either native."""
    view = make_instrument_view(1)
    for native in ("linear", "circular"):
        for rotation_deg in ROTATIONS_DEG:
            receptors = make_receptor_set(((native, rotation_deg),), "circular_rl")
            combined = compute(
                BasisTransformJones(receptors=receptors, instrument=view), 0
            ) @ compute(ReceptorConfigJones(receptors=receptors, instrument=view), 0)
            chi_rad = math.radians(rotation_deg)
            expected = PLAN_S_MATRIX @ np.array(
                [
                    [math.cos(chi_rad), math.sin(chi_rad)],
                    [-math.sin(chi_rad), math.cos(chi_rad)],
                ],
                dtype=np.complex128,
            )
            np.testing.assert_allclose(combined, expected, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("native", ("linear", "circular"))
@pytest.mark.parametrize("output", ("linear_xy", "circular_rl"))
@pytest.mark.parametrize("rotation_deg", ROTATIONS_DEG)
def test_unpolarized_energy_conservation_survives_every_transform(
    native: str,
    output: str,
    rotation_deg: float,
) -> None:
    """Section 18.6 through both production terms."""
    stokes_i = 9.0
    view = make_instrument_view(2)
    receptors = make_receptor_set(
        ((native, rotation_deg), (native, rotation_deg)),
        output,
    )
    receptor_term = ReceptorConfigJones(receptors=receptors, instrument=view)
    transform_term = BasisTransformJones(receptors=receptors, instrument=view)
    coherency = np.asarray(stokes_to_coherency(stokes_i, 0.0, 0.0, 0.0))

    jones_p = compute(transform_term, 0) @ compute(receptor_term, 0)
    jones_q = compute(transform_term, 1) @ compute(receptor_term, 1)
    visibility = jones_p @ coherency @ jones_q.conj().T

    assert visibility[0, 0].real + visibility[1, 1].real == pytest.approx(
        stokes_i, abs=1e-13
    )
    assert abs(visibility[0, 1]) == pytest.approx(0.0, abs=1e-14)
    assert abs(visibility[1, 0]) == pytest.approx(0.0, abs=1e-14)


# ---------------------------------------------------------------------------
# Constructor and error contract
# ---------------------------------------------------------------------------


def test_the_permissive_stub_constructor_is_removed() -> None:
    """Section 24: from_basis=/to_basis= must raise a TypeError naming receptors."""
    receptors = make_receptor_set((("linear", 0.0),), "linear_xy")
    view = make_instrument_view(1)

    with pytest.raises(TypeError) as excinfo:
        BasisTransformJones(from_basis="linear", to_basis="circular")  # type: ignore[call-arg]
    assert "receptors" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        BasisTransformJones(  # type: ignore[call-arg]
            receptors=receptors,
            instrument=view,
            to_basis="circular",
        )
    assert "to_basis" in str(excinfo.value)
    assert "receptors" in str(excinfo.value)

    with pytest.raises(TypeError):
        BasisTransformJones(receptors, view)  # type: ignore[misc]


def test_an_unsupported_transform_is_a_typed_failure() -> None:
    """Section 25.1: UnsupportedBasisTransformError names the requested pair."""
    for native, output in (
        ("linear", "stokes"),
        ("elliptical", "linear_xy"),
        ("circular", "circular"),
    ):
        with pytest.raises(UnsupportedBasisTransformError) as excinfo:
            basis_transform_matrix(native, output)
        assert "linear_xy" in str(excinfo.value)
        assert "circular_rl" in str(excinfo.value)


def test_an_antenna_absent_from_the_receptor_set_is_a_typed_failure() -> None:
    receptors = make_receptor_set((("linear", 0.0),), "linear_xy")

    with pytest.raises(ReceptorAssignmentError):
        BasisTransformJones(receptors=receptors, instrument=make_instrument_view(2))


def test_an_out_of_range_antenna_row_is_rejected() -> None:
    receptors = make_receptor_set((("linear", 0.0),), "linear_xy")
    term = BasisTransformJones(receptors=receptors, instrument=make_instrument_view(1))

    with pytest.raises(InstrumentAdapterInvariantError):
        compute(term, 7)


def test_returned_matrices_do_not_alias_internal_state() -> None:
    receptors = make_receptor_set((("linear", 0.0),), "circular_rl")
    term = BasisTransformJones(receptors=receptors, instrument=make_instrument_view(1))

    first = compute(term, 0)
    first[0, 0] = 999.0

    assert compute(term, 0)[0, 0] != 999.0


def test_both_terms_are_reachable_from_the_lazy_jones_namespace() -> None:
    import radiosim.core.jones as jones

    assert jones.ReceptorConfigJones is ReceptorConfigJones
    assert jones.BasisTransformJones is BasisTransformJones
    assert AntennaId(0, "ANT0") == AntennaId(0, "ANT0")


def test_neither_term_can_be_constructed_without_arguments() -> None:
    with pytest.raises(TypeError):
        ReceptorConfigJones()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        BasisTransformJones()  # type: ignore[call-arg]
