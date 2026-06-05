"""JAX backend precision behavior."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.backends.base import BackendNotAvailableError


def _get_jax_backend():
    pytest.importorskip("jax")
    try:
        return get_backend("jax", device="cpu")
    except BackendNotAvailableError as exc:
        pytest.skip(str(exc))


def test_jax_backend_enables_x64_for_standard_precision():
    backend = _get_jax_backend()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        real = backend.asarray([1.0], dtype=np.float64)
        complex_arr = backend.eye_complex(2, dtype=np.complex128)

    truncation_warnings = [
        warning for warning in caught if "will be truncated" in str(warning.message)
    ]
    assert truncation_warnings == []
    assert backend.to_numpy(real).dtype == np.float64
    assert backend.to_numpy(complex_arr).dtype == np.complex128
