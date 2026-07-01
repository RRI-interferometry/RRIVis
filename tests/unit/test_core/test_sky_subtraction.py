"""Tests for sky subtraction internals and ownership contracts."""

from __future__ import annotations

import numpy as np

from radiosim.core.sky.operations.subtraction import _inpaint_by_alm


def test_inpaint_empty_mask_returns_fresh_copy():
    maps = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = _inpaint_by_alm(
        maps,
        nside=1,
        mask_pixels=np.array([], dtype=np.int64),
        max_iterations=1,
        rtol=1e-3,
    )
    assert out is not maps
    np.testing.assert_array_equal(out, maps)


def test_inpaint_empty_mask_copy_is_mutable_without_affecting_input():
    maps = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    out = _inpaint_by_alm(
        maps,
        nside=1,
        mask_pixels=np.array([], dtype=np.int64),
        max_iterations=1,
        rtol=1e-3,
    )
    out[0, 0] = 99.0
    assert maps[0, 0] == 1.0
