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

from radiosim.core.polarization import stokes_to_coherency


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
