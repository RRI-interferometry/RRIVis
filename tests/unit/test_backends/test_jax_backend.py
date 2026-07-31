"""JAX backend precision behavior."""

from __future__ import annotations

import warnings

import numpy as np

from radiosim.backends import get_backend


def _get_jax_backend():
    """Return the CPU JAX backend.

    No ``importorskip`` guard: Tier 6H made a CPU-only ``jax``/``jaxlib`` a
    declared dependency of every pixi environment precisely so the mandated
    NumPy/JAX parity evidence is measured rather than skipped
    (``Tier6HybridRuntimePlan.md`` Sections 28, 31, 32.8). A missing JAX is now
    a broken environment, and this must fail loudly rather than skip quietly.
    """
    return get_backend("jax", device="cpu")


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
