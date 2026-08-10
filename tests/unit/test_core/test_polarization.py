"""Tier 5C scientific invariants for the sky-linear brightness matrix.

Every oracle in this module is written out from ``Tier5ReceptorFeedPlan.md``
Sections 10.2, 18.1, 18.4, 18.5, and 18.6, as amended by SCI-006's east-X
linear binding, rather than derived from RadioSim source.  The assertions are
therefore checkable without reading the implementation.

Invariants covered here: S2, S3, S4, S5, S7, S8.
"""

from __future__ import annotations

import numpy as np
import pytest

import radiosim.core.polarization as polarization_module
from radiosim.core.polarization import (
    coherency_to_stokes,
    stokes_to_coherency,
)

# ---------------------------------------------------------------------------
# Plan oracles, transcribed rather than imported: S from Tier 5 Section 18.1
# and the east-X permutation P from the SCI-006 correction.
# ---------------------------------------------------------------------------

PLAN_S_MATRIX = (1.0 / np.sqrt(2.0)) * np.array(
    [[1.0, 1.0j], [1.0, -1.0j]],
    dtype=np.complex128,
)
PLAN_P_MATRIX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

ROTATIONS_DEG = (0.0, 30.0, 45.0, 90.0, -15.0)


def plan_rotation(chi_rad: float) -> np.ndarray:
    """Return the Section 18.1 receptor rotation ``R(chi)``."""
    cos_chi = np.cos(chi_rad)
    sin_chi = np.sin(chi_rad)
    return np.array(
        [[cos_chi, sin_chi], [-sin_chi, cos_chi]],
        dtype=np.complex128,
    )


def plan_brightness(
    stokes_i: float,
    stokes_q: float,
    stokes_u: float,
    stokes_v: float,
) -> np.ndarray:
    """Return the Section 10.1 brightness matrix, written out from the plan."""
    return 0.5 * np.array(
        [
            [stokes_i + stokes_q, stokes_u + 1j * stokes_v],
            [stokes_u - 1j * stokes_v, stokes_i - stokes_q],
        ],
        dtype=np.complex128,
    )


# The Section 18.4 reference cases: unpolarized, pure Q, pure U, pure V, mixed.
REFERENCE_STOKES = (
    (10.0, 0.0, 0.0, 0.0),
    (10.0, 10.0, 0.0, 0.0),
    (10.0, 0.0, 10.0, 0.0),
    (10.0, 0.0, 0.0, 10.0),
    (7.5, -1.25, 3.0, -2.0),
)


# ---------------------------------------------------------------------------
# S2 -- the corrected brightness matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stokes", REFERENCE_STOKES)
def test_stokes_to_coherency_reproduces_the_iau_hbs_brightness_matrix(stokes) -> None:
    """S2: C == 1/2 [[I+Q, U+iV], [U-iV, I-Q]] exactly."""
    coherency = np.asarray(stokes_to_coherency(*stokes))

    np.testing.assert_allclose(
        coherency,
        plan_brightness(*stokes),
        rtol=0.0,
        atol=0.0,
    )


def test_upper_right_element_carries_plus_i_v() -> None:
    """The V sign lives in C[0,1] as +iV/2, and in C[1,0] as -iV/2."""
    stokes_i, stokes_q, stokes_u, stokes_v = 10.0, 2.0, -1.0, 0.5
    coherency = np.asarray(stokes_to_coherency(stokes_i, stokes_q, stokes_u, stokes_v))

    assert coherency[0, 1].imag == pytest.approx(+stokes_v / 2.0)
    assert coherency[1, 0].imag == pytest.approx(-stokes_v / 2.0)
    assert coherency[0, 1].real == pytest.approx(stokes_u / 2.0)
    assert coherency[1, 0].real == pytest.approx(stokes_u / 2.0)


@pytest.mark.parametrize("stokes", REFERENCE_STOKES)
def test_coherency_matrix_is_hermitian(stokes) -> None:
    """A physical brightness matrix is Hermitian in either V convention."""
    coherency = np.asarray(stokes_to_coherency(*stokes))

    np.testing.assert_allclose(
        coherency,
        coherency.conj().T,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("stokes", REFERENCE_STOKES)
def test_half_power_normalization_is_untouched_by_the_v_correction(stokes) -> None:
    """The sky coherency has ``Tr(C)=I``; no post-Jones claim is made here."""
    stokes_i = stokes[0]
    coherency = np.asarray(stokes_to_coherency(*stokes))

    assert coherency[0, 0].real + coherency[1, 1].real == pytest.approx(stokes_i)


def test_v_enters_only_the_imaginary_part_of_the_cross_hands() -> None:
    """The correction's blast radius: parallel hands and Re(cross) are unmoved.

    This is the elementwise statement of the Section 10.2 claim that every
    ``V = 0`` result is bit-identical before and after the correction.
    """
    base = (7.5, -1.25, 3.0)
    without_v = np.asarray(stokes_to_coherency(*base, 0.0))

    for stokes_v in (-4.0, -0.25, 0.25, 4.0):
        with_v = np.asarray(stokes_to_coherency(*base, stokes_v))

        # Parallel hands identical, bit for bit.
        assert with_v[0, 0] == without_v[0, 0]
        assert with_v[1, 1] == without_v[1, 1]
        # Real part of both cross hands identical, bit for bit.
        assert with_v[0, 1].real == without_v[0, 1].real
        assert with_v[1, 0].real == without_v[1, 0].real
        # Only the cross-hand imaginary parts move.
        assert with_v[0, 1].imag == pytest.approx(+stokes_v / 2.0)
        assert with_v[1, 0].imag == pytest.approx(-stokes_v / 2.0)


def test_zero_v_construction_is_bit_identical_to_the_mirrored_convention() -> None:
    """For V == 0 the corrected and baseline constructions agree exactly."""
    for stokes_i, stokes_q, stokes_u in ((10.0, 2.0, -1.0), (1.0, 0.0, 0.0)):
        corrected = np.asarray(stokes_to_coherency(stokes_i, stokes_q, stokes_u, 0.0))
        mirrored = 0.5 * np.array(
            [
                [stokes_i + stokes_q, stokes_u - 0.0j],
                [stokes_u + 0.0j, stokes_i - stokes_q],
            ],
            dtype=np.complex128,
        )
        np.testing.assert_array_equal(corrected, mirrored)


def test_stokes_to_coherency_broadcasts_over_source_axes() -> None:
    """Array inputs keep the documented (..., 2, 2) output shape."""
    count = 6
    rng = np.random.default_rng(20260729)
    stokes_i = rng.uniform(1.0, 5.0, size=count)
    stokes_q = rng.uniform(-1.0, 1.0, size=count)
    stokes_u = rng.uniform(-1.0, 1.0, size=count)
    stokes_v = rng.uniform(-1.0, 1.0, size=count)

    coherency = np.asarray(
        stokes_to_coherency(stokes_i, stokes_q, stokes_u, stokes_v, xp=np)
    )

    assert coherency.shape == (count, 2, 2)
    np.testing.assert_allclose(
        coherency[..., 0, 1].imag,
        stokes_v / 2.0,
        rtol=0.0,
        atol=1e-15,
    )


# ---------------------------------------------------------------------------
# S3 -- the round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stokes", REFERENCE_STOKES)
def test_coherency_to_stokes_inverts_stokes_to_coherency(stokes) -> None:
    """S3: the inverse recovers arbitrary I, Q, U, V to machine precision."""
    recovered = coherency_to_stokes(np.asarray(stokes_to_coherency(*stokes)))

    np.testing.assert_allclose(
        np.asarray(recovered, dtype=np.float64),
        np.asarray(stokes, dtype=np.float64),
        rtol=0.0,
        atol=1e-14,
    )


def test_round_trip_holds_for_randomized_stokes_vectors() -> None:
    """S3 over a randomized batch, including strongly circular sources."""
    rng = np.random.default_rng(5031)
    count = 512
    stokes_i = rng.uniform(0.5, 20.0, size=count)
    stokes_q = rng.uniform(-5.0, 5.0, size=count)
    stokes_u = rng.uniform(-5.0, 5.0, size=count)
    stokes_v = rng.uniform(-5.0, 5.0, size=count)

    recovered = coherency_to_stokes(
        np.asarray(stokes_to_coherency(stokes_i, stokes_q, stokes_u, stokes_v))
    )

    for expected, actual in zip(
        (stokes_i, stokes_q, stokes_u, stokes_v),
        recovered,
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13)


def test_coherency_to_stokes_derives_v_from_the_upper_right_element() -> None:
    """The inverse reads V off C[0,1], matching the corrected construction."""
    coherency = np.asarray(stokes_to_coherency(7.5, -1.25, 3.0, -2.0))

    assert coherency_to_stokes(coherency)[3] == pytest.approx(
        2.0 * coherency[0, 1].imag
    )


# ---------------------------------------------------------------------------
# S4 -- circular correlations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stokes", REFERENCE_STOKES)
def test_circular_output_reproduces_the_section_18_4_table(stokes) -> None:
    """S4: S B S^H == [[(I+V)/2, (Q+iU)/2], [(Q-iU)/2, (I-V)/2]]."""
    stokes_i, stokes_q, stokes_u, stokes_v = stokes
    coherency = np.asarray(stokes_to_coherency(*stokes))

    circular = PLAN_S_MATRIX @ coherency @ PLAN_S_MATRIX.conj().T

    expected = np.array(
        [
            [(stokes_i + stokes_v) / 2.0, (stokes_q + 1j * stokes_u) / 2.0],
            [(stokes_q - 1j * stokes_u) / 2.0, (stokes_i - stokes_v) / 2.0],
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(circular, expected, rtol=0.0, atol=1e-14)


def test_a_positive_v_source_is_pure_rr_not_pure_ll() -> None:
    """S4's headline consequence and the observable defect Section 10.2 names."""
    total_flux = 4.0
    coherency = np.asarray(stokes_to_coherency(total_flux, 0.0, 0.0, total_flux))

    circular = PLAN_S_MATRIX @ coherency @ PLAN_S_MATRIX.conj().T

    # RR is index [0, 0] and LL is index [1, 1] in the (R, L) row ordering.
    assert circular[0, 0].real == pytest.approx(total_flux, abs=1e-14)
    assert circular[1, 1].real == pytest.approx(0.0, abs=1e-14)


def test_a_negative_v_source_is_pure_ll() -> None:
    """The mirror statement, so the test cannot pass under a symmetric bug."""
    total_flux = 4.0
    coherency = np.asarray(stokes_to_coherency(total_flux, 0.0, 0.0, -total_flux))

    circular = PLAN_S_MATRIX @ coherency @ PLAN_S_MATRIX.conj().T

    assert circular[0, 0].real == pytest.approx(0.0, abs=1e-14)
    assert circular[1, 1].real == pytest.approx(total_flux, abs=1e-14)


# ---------------------------------------------------------------------------
# S5 -- unpolarized energy conservation, every basis, every rotation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ("linear", "circular"))
@pytest.mark.parametrize("rotation_deg", ROTATIONS_DEG)
def test_unpolarized_energy_is_conserved_in_both_bases(
    basis: str,
    rotation_deg: float,
) -> None:
    """S5: V[0,0] + V[1,1] == I and V[0,1] == V[1,0] == 0 for any unitary J."""
    stokes_i = 9.0
    coherency = np.asarray(stokes_to_coherency(stokes_i, 0.0, 0.0, 0.0))
    chi_rad = np.radians(rotation_deg)
    leading = PLAN_P_MATRIX if basis == "linear" else PLAN_S_MATRIX
    jones = leading @ plan_rotation(chi_rad)

    visibility = jones @ coherency @ jones.conj().T

    assert visibility[0, 0].real + visibility[1, 1].real == pytest.approx(
        stokes_i, abs=1e-13
    )
    assert abs(visibility[0, 1]) == pytest.approx(0.0, abs=1e-14)
    assert abs(visibility[1, 0]) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("stokes", REFERENCE_STOKES)
def test_total_intensity_is_basis_independent(stokes) -> None:
    """XX + YY == RR + LL == I for every reference source."""
    stokes_i = stokes[0]
    coherency = np.asarray(stokes_to_coherency(*stokes))
    circular = PLAN_S_MATRIX @ coherency @ PLAN_S_MATRIX.conj().T

    assert coherency[0, 0].real + coherency[1, 1].real == pytest.approx(stokes_i)
    assert circular[0, 0].real + circular[1, 1].real == pytest.approx(
        stokes_i, abs=1e-14
    )


# ---------------------------------------------------------------------------
# S7 / S8 -- the rotation oracles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rotation_deg", ROTATIONS_DEG)
def test_linear_rotation_rotates_q_and_u_by_twice_chi(rotation_deg: float) -> None:
    """S7: R(chi) B R(chi)^T rotates (Q, U) by 2 chi and preserves I and V."""
    stokes_i, stokes_q, stokes_u, stokes_v = 7.5, -1.25, 3.0, -2.0
    chi_rad = np.radians(rotation_deg)
    rotation = plan_rotation(chi_rad)
    coherency = np.asarray(stokes_to_coherency(stokes_i, stokes_q, stokes_u, stokes_v))

    rotated = rotation @ coherency @ rotation.T
    recovered_i, recovered_q, recovered_u, recovered_v = coherency_to_stokes(rotated)

    cos_2chi = np.cos(2.0 * chi_rad)
    sin_2chi = np.sin(2.0 * chi_rad)
    assert recovered_i == pytest.approx(stokes_i, abs=1e-13)
    assert recovered_v == pytest.approx(stokes_v, abs=1e-13)
    assert recovered_q == pytest.approx(
        stokes_q * cos_2chi + stokes_u * sin_2chi, abs=1e-13
    )
    assert recovered_u == pytest.approx(
        -stokes_q * sin_2chi + stokes_u * cos_2chi, abs=1e-13
    )


@pytest.mark.parametrize("rotation_deg", ROTATIONS_DEG)
def test_circular_rotation_phases_only_the_cross_hands(rotation_deg: float) -> None:
    """S8: RR and LL are invariant; RL gains e^{-2i chi} and LR gains e^{+2i chi}."""
    stokes = (7.5, -1.25, 3.0, -2.0)
    chi_rad = np.radians(rotation_deg)
    coherency = np.asarray(stokes_to_coherency(*stokes))

    unrotated = PLAN_S_MATRIX @ coherency @ PLAN_S_MATRIX.conj().T
    jones = PLAN_S_MATRIX @ plan_rotation(chi_rad)
    rotated = jones @ coherency @ jones.conj().T

    assert rotated[0, 0] == pytest.approx(unrotated[0, 0], abs=1e-13)
    assert rotated[1, 1] == pytest.approx(unrotated[1, 1], abs=1e-13)
    assert rotated[0, 1] == pytest.approx(
        np.exp(-2j * chi_rad) * unrotated[0, 1], abs=1e-13
    )
    assert rotated[1, 0] == pytest.approx(
        np.exp(+2j * chi_rad) * unrotated[1, 0], abs=1e-13
    )


def test_section_18_5_rotation_identity_holds_elementwise() -> None:
    """S R(chi) == diag(e^{-i chi}, e^{+i chi}) S, the analytic Section 18.5 step."""
    for rotation_deg in ROTATIONS_DEG:
        chi_rad = np.radians(rotation_deg)
        left = PLAN_S_MATRIX @ plan_rotation(chi_rad)
        right = (
            np.diag(
                np.array(
                    [np.exp(-1j * chi_rad), np.exp(+1j * chi_rad)],
                    dtype=np.complex128,
                )
            )
            @ PLAN_S_MATRIX
        )
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Documented attribution
# ---------------------------------------------------------------------------


def test_module_docstring_records_the_iau_hbs_attribution() -> None:
    """The docstring must name the convention RadioSim actually implements."""
    docstring = polarization_module.__doc__ or ""

    assert "C[0,1] = (U + iV) / 2" in docstring
    assert "Hamaker" in docstring
    assert "Smirnov" in docstring
    # The refuted Tier 5A claim must be gone, not merely amended.
    assert "Africanus/Pauli" not in docstring
    assert "Matches: Codex-Africanus" not in docstring
    assert "NOT: (U + iV) / 2" not in docstring


def test_module_docstring_records_the_pyradiosky_divergence() -> None:
    """Risk-register requirement: the divergence must be stated in the source."""
    docstring = polarization_module.__doc__ or ""

    assert "pyradiosky" in docstring
    assert "(U - iV)" in docstring


# ---------------------------------------------------------------------------
# Tier 5H removals (Section 34.8, resolving Section 43 Q4 and Q5)
# ---------------------------------------------------------------------------

TIER5H_REMOVED_HELPERS = ("visibility_to_correlations", "mueller_from_jones")


@pytest.mark.parametrize("name", TIER5H_REMOVED_HELPERS)
def test_superseded_helper_is_gone_from_the_module(name: str) -> None:
    """Section 34.8 removes both on the Tier 5A no-production-caller evidence."""
    assert not hasattr(polarization_module, name)
    with pytest.raises(AttributeError):
        getattr(polarization_module, name)


@pytest.mark.parametrize("name", TIER5H_REMOVED_HELPERS)
def test_superseded_helper_cannot_be_imported(name: str) -> None:
    with pytest.raises(ImportError):
        exec(
            compile(
                f"from radiosim.core.polarization import {name}\n",
                "<tier5h>",
                "exec",
            ),
            {},
        )


def test_the_module_no_longer_hard_keys_the_linear_correlation_labels() -> None:
    """``visibility_to_correlations`` was the last linear-only label table.

    ``radiosim.core.polarization_basis.CORRELATION_LABELS`` is the sole
    authority, so no executable dictionary in this module may key correlations
    by literal linear labels.  Docstring prose -- which still cites the
    codex-africanus ``"XY": u + v*1j`` mapping as a convention reference -- is
    not code and is deliberately not scanned.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(polarization_module))
    labels = {"XX", "XY", "YX", "YY", "RR", "RL", "LR", "LL"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                assert key.value not in labels


def test_the_surviving_helpers_are_untouched() -> None:
    """Section 34.8's ledger names two helpers; their neighbours stay."""
    for name in (
        "stokes_to_coherency",
        "coherency_to_stokes",
        "apply_jones_matrices",
        "stokes_I_only_visibility",
        "jones_matrix_power",
    ):
        assert callable(getattr(polarization_module, name))


def test_the_core_package_no_longer_re_exports_the_removed_helper() -> None:
    import radiosim.core as core_package

    assert "visibility_to_correlations" not in core_package.__all__
    assert not hasattr(core_package, "visibility_to_correlations")
