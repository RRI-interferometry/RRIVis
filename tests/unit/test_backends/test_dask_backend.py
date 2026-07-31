"""The renamed Dask backend, and the claims Tier 6H removed with the name.

``Tier6HybridRuntimePlan.md`` Section 14.1 renames ``NumbaBackend`` to
``DaskBackend`` and deletes three things that named capabilities the class never
had: the ``mode="gpu"`` CUDA-validation path, the ``jit_compile()`` helper, and
the ``numba`` selector. Section 39's risk row requires the rename to be
unmistakably *not* a capability gain, so the tests below assert the removals as
well as the new name.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from radiosim.backends import DaskBackend, get_backend, list_backends
from radiosim.backends.dask_backend import DASK_MODES


def test_the_backend_reports_what_it_is() -> None:
    backend = DaskBackend(mode="cpu")

    assert backend.name == "dask-cpu"
    assert backend.backend_type == "dask"
    assert backend.xp is np
    assert backend.device_kind == "cpu"
    assert backend.supports_compilation is False
    assert backend.is_available() is True


def test_distributed_mode_keeps_its_own_name() -> None:
    assert DASK_MODES == ("cpu", "distributed")
    # Constructing the distributed mode would start a cluster; the naming rule
    # is a property of the class, so assert it without paying for one.
    assert DaskBackend.name.fget(_FakeMode("distributed")) == "dask-distributed"
    assert DaskBackend.name.fget(_FakeMode("cpu")) == "dask-cpu"


class _FakeMode:
    def __init__(self, mode: str) -> None:
        self.mode = mode


def test_gpu_mode_is_removed_and_names_its_replacement() -> None:
    """It validated a CUDA device and then executed NumPy (defect D8)."""
    with pytest.raises(ValueError) as exc_info:
        DaskBackend(mode="gpu")

    message = str(exc_info.value)
    assert "removed before v1.0" in message
    assert "executed NumPy" in message
    assert "mode='cpu'" in message


def test_an_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        DaskBackend(mode="cuda")


def test_jit_compile_is_removed_and_names_its_replacement() -> None:
    """The helper had no caller and compiled nothing for the solver."""
    backend = DaskBackend(mode="cpu")

    for removed in ("jit_compile", "jit", "prange"):
        with pytest.raises(AttributeError) as exc_info:
            getattr(backend, removed)
        assert "removed before v1.0" in str(exc_info.value)
        assert "supports_compilation" in str(exc_info.value)


def test_the_numba_module_and_name_are_gone() -> None:
    assert "radiosim.backends.numba_backend" not in sys.modules
    with pytest.raises(ImportError):
        __import__("radiosim.backends.numba_backend")

    import radiosim.backends as backends_package

    assert not hasattr(backends_package, "NumbaBackend")
    assert not hasattr(backends_package, "is_numba_available")
    assert not hasattr(backends_package, "is_cuda_available")


def test_the_registry_advertises_dask_not_numba() -> None:
    available = list_backends()

    assert "numba" not in available
    assert "cuda" not in available
    assert available["dask"] is True
    assert available["numpy"] is True


def test_numba_is_not_constructible_and_the_error_names_dask() -> None:
    with pytest.raises(ValueError) as exc_info:
        get_backend("numba")

    message = str(exc_info.value)
    assert "removed before v1.0" in message
    assert "get_backend('dask')" in message


def test_synchronize_computes_a_lazy_array() -> None:
    """A NumPy array is already materialized; the argument form still returns it."""
    backend = DaskBackend(mode="cpu")
    array = backend.asarray([1.0, 2.0, 3.0])

    assert backend.synchronize() is None
    assert np.array_equal(backend.synchronize(array), array)
