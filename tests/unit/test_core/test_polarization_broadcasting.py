"""API-001 regression tests: ``stokes_to_coherency`` broadcasts its inputs.

The register row's words: ``stokes_to_coherency(np.ones(5))`` — "the single
most basic array-input call" — raised, because the scalar Q/U/V defaults
could not join the ``stack`` against an array ``I``. These tests pin the
adopted fix (PostTier8RemediationPlan.md §7): scalar and array Stokes inputs
broadcast against each other, genuinely incompatible shapes still raise
``ValueError``, and the matched-shape path is arithmetically untouched.

Expected values are built in the test body from the module's documented
closed form ``C = (1/2) [[I+Q, U+iV], [U-iV, I-Q]]``, not from RadioSim
source (Tier-1 style).
"""

from __future__ import annotations

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.core.polarization import stokes_to_coherency

_BACKEND_NAMES = ("numpy", "jax", "dask")


def _backend(name: str):
    if name == "jax":
        return get_backend("jax", device="cpu")
    if name == "dask":
        return get_backend("dask", mode="cpu")
    return get_backend("numpy")


def _to_numpy(backend, value) -> np.ndarray:
    return np.asarray(backend.to_numpy(value))


def _analytic_coherency(stokes_i, stokes_q=0, stokes_u=0, stokes_v=0):
    """Build the documented matrix independently of RadioSim's stack path."""
    stokes_i, stokes_q, stokes_u, stokes_v = np.broadcast_arrays(
        np.asarray(stokes_i, dtype=float),
        np.asarray(stokes_q, dtype=float),
        np.asarray(stokes_u, dtype=float),
        np.asarray(stokes_v, dtype=float),
    )
    expected = np.empty(stokes_i.shape + (2, 2), dtype=np.complex128)
    expected[..., 0, 0] = (stokes_i + stokes_q) / 2.0
    expected[..., 0, 1] = (stokes_u + 1j * stokes_v) / 2.0
    expected[..., 1, 0] = (stokes_u - 1j * stokes_v) / 2.0
    expected[..., 1, 1] = (stokes_i - stokes_q) / 2.0
    return expected


def _pre_api001_equal_shape(stokes_i, stokes_q, stokes_u, stokes_v, *, xp):
    """Reconstruct the exact pre-API-001 arithmetic, without broadcasting."""
    stokes_i = xp.asarray(stokes_i, dtype=float)
    stokes_q = xp.asarray(stokes_q, dtype=float)
    stokes_u = xp.asarray(stokes_u, dtype=float)
    stokes_v = xp.asarray(stokes_v, dtype=float)
    row_x = xp.stack([stokes_i + stokes_q, stokes_u + 1j * stokes_v], axis=-1)
    row_y = xp.stack([stokes_u - 1j * stokes_v, stokes_i - stokes_q], axis=-1)
    return xp.stack([row_x, row_y], axis=-2) / 2.0


@pytest.mark.parametrize("backend_name", _BACKEND_NAMES)
class TestBackendBroadcastingContract:
    """The adopted API works analytically in every selectable array domain."""

    def test_scalar_inputs_match_the_closed_form(self, backend_name):
        backend = _backend(backend_name)
        actual = _to_numpy(
            backend,
            stokes_to_coherency(4.0, 0.8, -0.4, 0.2, xp=backend.xp),
        )
        expected = _analytic_coherency(4.0, 0.8, -0.4, 0.2)

        assert actual.shape == (2, 2)
        np.testing.assert_array_equal(actual, expected)

    def test_array_i_broadcasts_untouched_scalar_defaults(self, backend_name):
        backend = _backend(backend_name)
        stokes_i = np.array([1.0, 2.5, 4.0, 7.5], dtype=np.float64)
        actual = _to_numpy(
            backend,
            stokes_to_coherency(stokes_i, xp=backend.xp),
        )
        expected = _analytic_coherency(stokes_i)

        assert actual.shape == (4, 2, 2)
        np.testing.assert_array_equal(actual, expected)

    def test_mixed_ranks_broadcast_to_one_analytic_cube(self, backend_name):
        backend = _backend(backend_name)
        stokes_i = np.array([[2.0], [3.0]], dtype=np.float64)
        stokes_q = np.array([0.4, -0.2, 0.1], dtype=np.float64)
        stokes_u = -0.125
        stokes_v = np.array([[0.2], [-0.3]], dtype=np.float64)
        actual = _to_numpy(
            backend,
            stokes_to_coherency(
                stokes_i,
                stokes_q,
                stokes_u,
                stokes_v,
                xp=backend.xp,
            ),
        )
        expected = _analytic_coherency(stokes_i, stokes_q, stokes_u, stokes_v)

        assert actual.shape == (2, 3, 2, 2)
        np.testing.assert_array_equal(actual, expected)

    def test_incompatible_shapes_still_raise_value_error(self, backend_name):
        backend = _backend(backend_name)
        with pytest.raises(ValueError):
            stokes_to_coherency(
                np.ones((2, 3)),
                np.zeros((4,)),
                xp=backend.xp,
            )


@pytest.mark.parametrize("backend_name", _BACKEND_NAMES)
@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_equal_shaped_inputs_are_byte_identical_to_the_pre_api001_path(
    backend_name, float_dtype
):
    """Broadcasting adds no arithmetic to the historically valid shape."""
    backend = _backend(backend_name)
    stokes_i = np.array([1.25, 2.5, 3.75, 5.0], dtype=float_dtype)
    stokes_q = np.array([0.5, -0.25, 0.125, -0.75], dtype=float_dtype)
    stokes_u = np.array([-0.125, 0.75, -0.5, 0.25], dtype=float_dtype)
    stokes_v = np.array([0.0625, -0.5, 0.375, -0.125], dtype=float_dtype)

    current = _to_numpy(
        backend,
        stokes_to_coherency(
            stokes_i,
            stokes_q,
            stokes_u,
            stokes_v,
            xp=backend.xp,
        ),
    )
    legacy = _to_numpy(
        backend,
        _pre_api001_equal_shape(
            stokes_i,
            stokes_q,
            stokes_u,
            stokes_v,
            xp=backend.xp,
        ),
    )

    assert current.shape == legacy.shape == (4, 2, 2)
    assert current.dtype == legacy.dtype == np.dtype(np.complex128)
    assert current.tobytes(order="C") == legacy.tobytes(order="C")


class TestScalarDefaultsBroadcastAgainstArrayI:
    """The signature's own defaults must accept an array ``I``."""

    def test_array_i_with_untouched_defaults_matches_explicit_zero_arrays(self):
        stokes_i = np.ones(5)
        zeros = np.zeros(5)

        from_defaults = np.asarray(stokes_to_coherency(stokes_i))
        explicit = np.asarray(stokes_to_coherency(stokes_i, zeros, zeros, zeros))

        assert from_defaults.shape == (5, 2, 2)
        np.testing.assert_array_equal(from_defaults, explicit)

    def test_scalar_q_against_array_i_equals_per_element_loop(self):
        rng = np.random.default_rng(20260805)
        stokes_i = rng.uniform(1.0, 5.0, size=7)
        stokes_q, stokes_u, stokes_v = 0.25, -0.5, 0.125

        batched = np.asarray(
            stokes_to_coherency(stokes_i, stokes_q, stokes_u, stokes_v)
        )

        # Per-element expectation from the documented closed form, built here.
        expected = np.empty((stokes_i.size, 2, 2), dtype=np.complex128)
        for k, i_val in enumerate(stokes_i):
            expected[k] = 0.5 * np.array(
                [
                    [i_val + stokes_q, stokes_u + 1j * stokes_v],
                    [stokes_u - 1j * stokes_v, i_val - stokes_q],
                ],
                dtype=np.complex128,
            )

        assert batched.shape == (stokes_i.size, 2, 2)
        np.testing.assert_allclose(batched, expected, rtol=0.0, atol=0.0)


class TestGenuineShapeErrorsSurvive:
    """Broadcasting must not swallow real shape incompatibilities."""

    def test_non_broadcastable_pair_still_raises_value_error(self):
        with pytest.raises(ValueError):
            stokes_to_coherency(np.ones(5), np.zeros(3))


class TestDtypeIsPreserved:
    """The broadcast path yields the same dtype as the matched-shape path."""

    @pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
    def test_broadcast_path_dtype_matches_matched_shape_path(self, float_dtype):
        stokes_i = np.linspace(1.0, 2.0, 4, dtype=float_dtype)
        zeros = np.zeros(4, dtype=float_dtype)

        broadcasted = np.asarray(stokes_to_coherency(stokes_i))
        matched = np.asarray(stokes_to_coherency(stokes_i, zeros, zeros, zeros))

        assert broadcasted.dtype == matched.dtype
        np.testing.assert_array_equal(broadcasted, matched)

    def test_float64_input_yields_complex128(self):
        out = np.asarray(stokes_to_coherency(np.ones(4), 0.5))
        assert out.dtype == np.complex128
