"""Tier 7H: the ``M`` term's physics, its Hadamard path, and closure.

``Tier7JonesSciencePlan.md`` Section 20.10, with the reference values written
out in the test bodies rather than read back from the production code
(Section 29.1).

``M`` is the one term in the package that is **not** expressible as a product of
per-antenna Jones matrices, and that is its whole point.  The discriminating
assertion is invariant **I11**: on a closed three-antenna triangle an enabled
``G`` with arbitrary per-antenna phases leaves the closure phase exactly
invariant, while an enabled ``M`` changes it by a predicted amount.  That single
comparison proves three things at once -- that ``M`` is applied, that it is
baseline-dependent, and that the Hadamard path is distinct from the matrix chain
(``Fix.md`` Section 16, Workstream D).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.core.jones.baseline_errors import BaselineMultiplicativeJones
from radiosim.core.jones_errors import (
    IdentityJonesTermError,
    InvalidJonesConfigError,
    JonesAssignmentError,
    JonesEvaluationError,
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
    three_antenna_layout,
)

_BACKEND = get_backend("numpy")

#: Two matrices that are not the Hadamard neutral element.  Written out rather
#: than built from ``np.eye`` on purpose: under ``(*)`` the identity is not the
#: neutral element at all, and a test that used it as "some error" would be
#: quietly asserting something about nulled cross-hands.
_NON_NEUTRAL = np.array(
    [
        [[1.5 + 0.0j, 0.9 - 0.1j], [1.1 + 0.2j, 0.8 + 0.0j]],
        [[0.5 + 0.5j, 1.0 + 0.0j], [1.0 + 0.0j, 1.2 + 0.0j]],
    ]
)

#: The triangle's three cross baselines, and the three autocorrelations the
#: shipped ``correlations: all`` selection also carries.
_TRIANGLE_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 2),
    (2, 2),
)


def _complex_matrix(values: np.ndarray) -> list[list[list[float]]]:
    """Return one ``2x2`` complex matrix in the schema's ``[re, im]`` form."""
    return [
        [[float(values[row, col].real), float(values[row, col].imag)] for col in (0, 1)]
        for row in (0, 1)
    ]


def _directions(n_dir: int = 4):
    from radiosim.core.jones.directions import DirectionBatch

    values = np.linspace(0.3, 1.2, n_dir)
    return DirectionBatch(
        alt_rad=values,
        az_rad=values / 2.0,
        dir_l=np.cos(values) * np.sin(values / 2.0),
        dir_m=np.cos(values) * np.cos(values / 2.0),
        dir_n=np.sin(values),
        ra_rad=values,
        dec_rad=-values,
        hour_angle_rad=values / 3.0,
        n_dir=n_dir,
    )


def _term(matrices: np.ndarray, pairs=((0, 1), (0, 2))) -> BaselineMultiplicativeJones:
    return BaselineMultiplicativeJones(
        baseline_pairs=tuple(pairs),
        matrices=np.asarray(matrices, dtype=np.complex128),
    )


def _factor(
    term: BaselineMultiplicativeJones,
    *,
    n_dir: int = 4,
    dtype: Any = np.complex128,
) -> np.ndarray:
    pairs = term.baseline_pairs
    return np.asarray(
        term.compute_baseline_factor(
            baseline_pairs=pairs,
            baseline_uvw_wavelengths=np.zeros((len(pairs), 3), dtype=np.float64),
            directions=_directions(n_dir),
            frequency_hz=1.0e8,
            freq_idx=0,
            time_mjd=60_676.0,
            time_idx=0,
            backend=_BACKEND,
            dtype=dtype,
        )
    )


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


# ---------------------------------------------------------------------------
# The term itself
# ---------------------------------------------------------------------------


def test_the_factor_is_the_configured_matrix_per_baseline() -> None:
    """``M`` returns exactly what it was resolved with, one matrix per baseline.

    Direction-independent, so the shape is ``(B, 2, 2)`` and not ``(B, n_dir,
    2, 2)``: a closure error is a property of the correlator pair, not of where
    the array is looking.
    """
    matrices = np.array(
        [
            [[1.02 + 0.0j, 0.01 - 0.02j], [0.0 + 0.0j, 0.98 + 0.05j]],
            [[0.5 + 0.5j, 0.0 + 0.0j], [-0.25 + 0.0j, 1.0 + 0.0j]],
        ]
    )
    term = _term(matrices)

    for n_dir in (1, 4, 17):
        factor = _factor(term, n_dir=n_dir)
        assert factor.shape == (2, 2, 2)
        np.testing.assert_array_equal(factor, matrices)


def test_the_term_declares_what_it_is() -> None:
    """Its status, its attachment point, and its direction independence."""
    term = _term(_NON_NEUTRAL)

    assert term.name == "M"
    assert term.term_status == "implemented"
    assert term.is_direction_dependent is False
    assert term.hadamard_target == "correlation"


def test_the_factor_is_returned_in_the_dtype_it_was_given() -> None:
    """The solver resolves the precision; the term never chooses one (I17)."""
    term = _term(_NON_NEUTRAL)

    assert _factor(term, dtype=np.complex128).dtype == np.complex128
    assert _factor(term, dtype=np.complex64).dtype == np.complex64


def test_a_factor_for_baselines_the_term_was_not_resolved_against_is_refused() -> None:
    """A silent mis-indexing would apply antenna 3's error to antenna 5's pair."""
    term = _term(_NON_NEUTRAL)

    with pytest.raises(JonesEvaluationError) as caught:
        term.compute_baseline_factor(
            baseline_pairs=((0, 1), (1, 2)),
            baseline_uvw_wavelengths=np.zeros((2, 3), dtype=np.float64),
            directions=_directions(),
            frequency_hz=1.0e8,
            freq_idx=0,
            time_mjd=60_676.0,
            time_idx=0,
            backend=_BACKEND,
            dtype=np.complex128,
        )

    assert "M" in str(caught.value)


def test_the_constructor_refuses_a_shape_it_cannot_index() -> None:
    with pytest.raises(ValueError):
        BaselineMultiplicativeJones(
            baseline_pairs=((0, 1), (0, 2)),
            matrices=np.zeros((3, 2, 2), dtype=np.complex128),
        )
    with pytest.raises(ValueError):
        BaselineMultiplicativeJones(
            baseline_pairs=((0, 1),),
            matrices=np.zeros((1, 3, 3), dtype=np.complex128),
        )
    with pytest.raises(ValueError):
        BaselineMultiplicativeJones(
            baseline_pairs=((0, 1),),
            matrices=np.array([[[np.nan, 0.0], [0.0, 1.0]]], dtype=np.complex128),
        )


def test_the_constructor_refuses_the_physics_keyword_a_stub_would_swallow() -> None:
    """The 7A pin's ``matrices=...`` probe, from the other side."""
    with pytest.raises(TypeError):
        BaselineMultiplicativeJones()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Resolution: the two configuration sources and the rejections
# ---------------------------------------------------------------------------


def test_an_array_wide_matrix_applies_to_every_selected_baseline(tmp_path) -> None:
    """Section 20.10's "or as one array-wide value", with no override at all."""
    matrix = np.array([[1.05 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.95 + 0.0j]])
    resolved = resolve_for(tmp_path, {"M": {"matrix": _complex_matrix(matrix)}})

    (term,) = resolved.baseline_terms
    assert isinstance(term, BaselineMultiplicativeJones)
    assert term.baseline_pairs == ((0, 0), (0, 1), (1, 1))
    np.testing.assert_allclose(
        term.matrices, np.stack([matrix] * 3), rtol=0.0, atol=0.0
    )


def test_a_per_baseline_entry_overrides_the_array_wide_default(tmp_path) -> None:
    """The same precedence every other term uses (Section 22 rule 5)."""
    default = np.array([[1.05 + 0.0j, 0.0j], [0.0j, 0.95 + 0.0j]])
    override = np.array([[0.5 + 0.25j, 0.0j], [0.0j, 2.0 + 0.0j]])
    resolved = resolve_for(
        tmp_path,
        {
            "M": {
                "matrix": _complex_matrix(default),
                "per_baseline": [
                    {"antennas": [0, 1], "matrix": _complex_matrix(override)}
                ],
            }
        },
    )

    (term,) = resolved.baseline_terms
    expected = np.stack([default, override, default])
    np.testing.assert_allclose(term.matrices, expected, rtol=0.0, atol=0.0)


def test_a_baseline_named_by_nothing_is_left_exactly_alone(tmp_path) -> None:
    """With no array-wide value, an unnamed baseline carries **ones**.

    Ones and not ``I2``: the neutral element of a Hadamard product is the
    all-ones matrix, and defaulting to the identity would null the cross-hand
    correlations of every baseline the block did not mention.
    """
    override = np.array([[1.5 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.8 + 0.0j]])
    resolved = resolve_for(
        tmp_path,
        {
            "M": {
                "per_baseline": [
                    {"antennas": [0, 1], "matrix": _complex_matrix(override)}
                ]
            }
        },
    )

    (term,) = resolved.baseline_terms
    ones = np.ones((2, 2))
    expected = np.stack([ones, override, ones]).astype(np.complex128)
    np.testing.assert_allclose(term.matrices, expected, rtol=0.0, atol=0.0)


def test_a_pair_outside_the_selection_is_rejected_with_the_r14_message(
    tmp_path,
) -> None:
    """R14, verbatim.  The shipped fixture has antennas 0 and 1 only."""
    with pytest.raises(JonesAssignmentError) as caught:
        resolve_for(
            tmp_path,
            {
                "M": {
                    "per_baseline": [
                        {
                            "antennas": [1, 7],
                            "matrix": _complex_matrix(_NON_NEUTRAL[0]),
                        }
                    ]
                }
            },
        )

    assert str(caught.value) == (
        "jones.M.per_baseline references baseline (1, 7), which is not in the "
        "resolved baseline selection."
    )


def test_the_reversed_pair_of_a_selected_baseline_is_still_not_in_it(
    tmp_path,
) -> None:
    """The key is the *ordered* pair, and the selection is canonically ordered.

    Accepting ``[1, 0]`` as a synonym for ``[0, 1]`` would mean silently
    deciding whether the user meant the conjugate baseline, which is a physical
    question the configuration did not answer.
    """
    with pytest.raises(JonesAssignmentError) as caught:
        resolve_for(
            tmp_path,
            {
                "M": {
                    "per_baseline": [
                        {
                            "antennas": [1, 0],
                            "matrix": _complex_matrix(_NON_NEUTRAL[0]),
                        }
                    ]
                }
            },
        )

    assert "(1, 0)" in str(caught.value)


def test_a_duplicate_baseline_entry_is_rejected_with_the_r5_message(tmp_path) -> None:
    """R5, in the bounded form Section 20.10's correction gives for ``M``."""
    entry = {"antennas": [0, 1], "matrix": _complex_matrix(_NON_NEUTRAL[0])}
    with pytest.raises(InvalidJonesConfigError) as caught:
        resolve_for(tmp_path, {"M": {"per_baseline": [entry, dict(entry)]}})

    assert str(caught.value) == (
        "jones.M.per_baseline contains a duplicate entry for baseline (0, 1); "
        "each baseline may appear once."
    )


def test_an_all_ones_configuration_is_rejected_with_the_r7_message(
    tmp_path,
) -> None:
    """R7, verbatim: an ``M`` that cannot break closure is an ``M`` that is not there.

    The rejected matrix is all **ones**, the Hadamard neutral element.  A block
    of identity matrices is *not* rejected, and must not be: it nulls both
    cross-hands, which is a real -- if drastic -- configured effect.
    """
    with pytest.raises(IdentityJonesTermError) as caught:
        resolve_for(
            tmp_path,
            {
                "M": {
                    "per_baseline": [
                        {
                            "antennas": [0, 1],
                            "matrix": _complex_matrix(np.ones((2, 2))),
                        }
                    ]
                }
            },
        )

    assert str(caught.value) == (
        "jones.M is configured with parameters that make it exactly the "
        "identity; a term that cannot change the visibilities must be removed "
        "rather than configured."
    )


def test_a_block_that_configures_no_matrix_at_all_is_the_same_rejection(
    tmp_path,
) -> None:
    with pytest.raises(IdentityJonesTermError):
        resolve_for(tmp_path, {"M": {"per_baseline": []}})


def test_m_is_a_baseline_term_and_never_enters_the_chain(tmp_path) -> None:
    """The structural half of Workstream D's "enforce the distinction"."""
    resolved = resolve_for(
        tmp_path, {"M": {"matrix": _complex_matrix(_NON_NEUTRAL[0])}}
    )

    assert resolved.chain_terms == ()
    assert resolved.configured_letters == ()
    assert resolved.baseline_letters == ("M",)
    assert len(resolved.baseline_terms) == 1
    assert "M" in resolved.provenance.enabled_terms
    assert "M" not in resolved.provenance.chain_order
    assert "M" in resolved.provenance.term_snapshots


# ---------------------------------------------------------------------------
# I11 -- ``M`` breaks closure, and the Hadamard path is elementwise
# ---------------------------------------------------------------------------


def test_a_closure_error_multiplies_the_finished_visibility_elementwise(
    tmp_path,
) -> None:
    """The Hadamard path, asserted as an exact identity on the whole cube.

    ``M`` is applied to the kernel's ``(B, 2, 2)`` output, so the corrupted cube
    is exactly the clean cube times the resolved matrix, element by element, on
    every time and every channel.  Anything else -- a matrix product, a
    per-antenna application, a transposition -- fails here.
    """
    left = np.array([[1.3 + 0.2j, 0.05 - 0.01j], [-0.02 + 0.03j, 0.7 - 0.4j]])
    right = np.array([[0.9 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.1 + 0.0j]])
    jones = {
        "M": {
            "matrix": _complex_matrix(right),
            "per_baseline": [{"antennas": [0, 1], "matrix": _complex_matrix(left)}],
        }
    }

    clean = _cube(tmp_path, None)
    corrupted = _cube(tmp_path, jones)

    expected_factor = np.stack([right, left, right])  # (0,0), (0,1), (1,1)
    expected = clean * expected_factor[None, :, None, :, :]

    np.testing.assert_array_equal(corrupted, expected)


def test_a_closure_error_changes_the_closure_phase_by_the_predicted_amount(
    tmp_path,
) -> None:
    """Invariant **I11**, the half that no per-antenna term can reproduce.

    The closure phase of a triangle is ``arg(V_01 V_12 V_02*)``.  Every
    antenna-based gain cancels in that product identically -- which
    ``test_gain.py``'s companion asserts -- so the residual is exactly the
    closure phase of the ``M`` matrices themselves:

        dphi = arg(M_01) + arg(M_12) - arg(M_02)

    computed here from the configured numbers alone.
    """
    phases = {(0, 1): 0.37, (1, 2): -0.81, (0, 2): 0.19}
    jones = {
        "M": {
            "per_baseline": [
                {
                    "antennas": list(pair),
                    "matrix": _complex_matrix(np.full((2, 2), np.exp(1j * phase))),
                }
                for pair, phase in phases.items()
            ]
        }
    }
    triangle = three_antenna_layout(tmp_path)

    clean = _cube(tmp_path, None, **triangle)
    corrupted = _cube(tmp_path, jones, **triangle)

    index = {pair: position for position, pair in enumerate(_TRIANGLE_PAIRS)}

    def closure(cube: np.ndarray) -> np.ndarray:
        first = cube[:, index[(0, 1)], :, 0, 0]
        second = cube[:, index[(1, 2)], :, 0, 0]
        third = cube[:, index[(0, 2)], :, 0, 0]
        return np.angle(first * second * np.conj(third))

    predicted = phases[(0, 1)] + phases[(1, 2)] - phases[(0, 2)]
    assert abs(predicted) > 0.1

    delta = np.angle(np.exp(1j * (closure(corrupted) - closure(clean))))
    np.testing.assert_allclose(delta, predicted, rtol=0.0, atol=1e-12)


def test_a_closure_error_is_not_expressible_as_any_pair_of_antenna_gains(
    tmp_path,
) -> None:
    """The defining property of ``M``, asserted constructively.

    If ``M_pq`` were ``g_p g_q^*`` for some per-antenna complex gains, the
    closure phase would be invariant, because those gains cancel around the
    triangle.  It is not invariant, so no such gains exist -- and the same
    configuration that breaks closure leaves each individual baseline's
    amplitude exactly where a per-antenna model would allow it, which is why
    closure and not amplitude is the discriminator.
    """
    triangle = three_antenna_layout(tmp_path)
    jones = {
        "M": {
            "per_baseline": [
                {
                    "antennas": [0, 1],
                    "matrix": _complex_matrix(np.full((2, 2), np.exp(1j * 0.6))),
                }
            ]
        }
    }

    clean = _cube(tmp_path, None, **triangle)
    corrupted = _cube(tmp_path, jones, **triangle)
    index = {pair: position for position, pair in enumerate(_TRIANGLE_PAIRS)}

    def closure(cube: np.ndarray) -> np.ndarray:
        return np.angle(
            cube[:, index[(0, 1)], :, 0, 0]
            * cube[:, index[(1, 2)], :, 0, 0]
            * np.conj(cube[:, index[(0, 2)], :, 0, 0])
        )

    assert float(np.max(np.abs(closure(corrupted) - closure(clean)))) > 0.5

    # Only the named baseline moved; the other two are untouched, which is what
    # "baseline-dependent" means and what an antenna-based term cannot do.
    for pair in ((1, 2), (0, 2)):
        np.testing.assert_array_equal(corrupted[:, index[pair]], clean[:, index[pair]])
