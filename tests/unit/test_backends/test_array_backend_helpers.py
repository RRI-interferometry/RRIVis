"""Concrete :class:`ArrayBackend` helpers that the solvers assemble blocks with.

``Tier6HybridRuntimePlan.md`` Section 13.3 gives ``ArrayBackend`` exactly one new
concrete helper for the Tier 6D accumulation restructure, ``stack(arrays,
axis=0)``.  It is defined once on the base class in terms of ``self.xp``, so
every backend inherits it and no backend may silently diverge; these tests hold
that contract, including the dtype and shape properties the solvers rely on when
they assemble one ``(B, 2, 2)`` block per ``(time, frequency)``, one
``(B, F, 2, 2)`` block per time, and one ``(T, B, F, 2, 2)`` cube per call.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.backends.base import ArrayBackend
from radiosim.backends.numpy_backend import NumPyBackend
from radiosim.core.precision import PrecisionConfig


def test_stack_is_a_concrete_base_class_helper_not_an_abstract_method() -> None:
    """Every backend inherits one implementation written in terms of ``self.xp``."""
    assert hasattr(ArrayBackend, "stack")
    assert "stack" not in getattr(ArrayBackend, "__abstractmethods__", frozenset())
    assert ArrayBackend.stack is NumPyBackend.stack
    source = inspect.getsource(ArrayBackend.stack)
    assert "self.xp.stack(" in source


def test_stack_signature_defaults_to_axis_zero() -> None:
    signature = inspect.signature(ArrayBackend.stack)
    assert list(signature.parameters) == ["self", "arrays", "axis"]
    assert signature.parameters["axis"].default == 0


def test_stack_adds_a_new_leading_axis_by_default() -> None:
    backend = get_backend("numpy")
    blocks = [np.full((3, 2, 2), value, dtype=np.complex128) for value in (1.0, 2.0)]

    stacked = backend.stack(blocks)

    assert np.asarray(stacked).shape == (2, 3, 2, 2)
    np.testing.assert_array_equal(np.asarray(stacked)[0], blocks[0])
    np.testing.assert_array_equal(np.asarray(stacked)[1], blocks[1])


def test_stack_honours_an_interior_axis() -> None:
    """The per-time block is ``F`` baseline blocks stacked on ``axis=1``."""
    backend = get_backend("numpy")
    n_baselines, n_freqs = 3, 4
    blocks = [
        np.full((n_baselines, 2, 2), float(freq_idx), dtype=np.complex128)
        for freq_idx in range(n_freqs)
    ]

    block = np.asarray(backend.stack(blocks, axis=1))

    assert block.shape == (n_baselines, n_freqs, 2, 2)
    for freq_idx in range(n_freqs):
        np.testing.assert_array_equal(block[:, freq_idx], blocks[freq_idx])


def test_stack_preserves_the_element_dtype_exactly() -> None:
    """The solvers cast every cell to the output dtype *before* assembly."""
    backend = get_backend("numpy", precision=PrecisionConfig.fast())
    blocks = [np.zeros((2, 2, 2), dtype=np.complex64) for _ in range(3)]

    assert np.asarray(backend.stack(blocks)).dtype == np.dtype(np.complex64)


def test_stack_result_is_contiguous_so_the_cube_needs_no_transpose() -> None:
    backend = get_backend("numpy")
    blocks = [np.zeros((5, 3, 2, 2), dtype=np.complex128) for _ in range(4)]

    cube = np.asarray(backend.stack(blocks, axis=0))

    assert cube.shape == (4, 5, 3, 2, 2)
    assert cube.flags["C_CONTIGUOUS"]


def test_stack_is_functional_and_does_not_mutate_its_inputs() -> None:
    """JAX-safety: assembly must never write into a caller's array."""
    backend = get_backend("numpy")
    first = np.zeros((2, 2, 2), dtype=np.complex128)
    second = np.ones((2, 2, 2), dtype=np.complex128)

    stacked = np.asarray(backend.stack([first, second]))
    stacked[0] = 99.0

    np.testing.assert_array_equal(first, np.zeros((2, 2, 2), dtype=np.complex128))
    np.testing.assert_array_equal(second, np.ones((2, 2, 2), dtype=np.complex128))


def test_stack_rejects_an_empty_sequence() -> None:
    """The solvers guard degenerate axes rather than assembling nothing."""
    backend = get_backend("numpy")
    with pytest.raises(ValueError):
        backend.stack([])


def test_set_at_remains_on_the_surface_after_the_restructure() -> None:
    """Section 13.3: ``set_at`` stays; it is only unused in the solver hot path."""
    backend = get_backend("numpy")
    arr = np.zeros((2, 2), dtype=np.complex128)

    updated = backend.set_at(arr, (0, 1), 3.0 + 1j)

    assert np.asarray(updated)[0, 1] == 3.0 + 1j
