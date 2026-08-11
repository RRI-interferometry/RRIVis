"""PERF-001 P-a acceptance tests for baseline-axis contraction chunking."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.backends.dask_backend import DaskBackend
from radiosim.backends.numpy_backend import NumPyBackend
from radiosim.core.contraction import (
    _TARGET_KERNEL_PAIRS,
    _baseline_contraction_for_policy,
    baseline_contraction_for,
)
from radiosim.core.precision import PrecisionConfig


class _RecordingCompilingBackend(NumPyBackend):
    """NumPy execution with an observable compile boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.compile_calls = 0
        self.leaf_calls: list[tuple[tuple[int, ...], ...]] = []
        self.leaf_arguments: list[tuple[Any, ...]] = []
        self.leaf_signature: inspect.Signature | None = None

    @property
    def supports_compilation(self) -> bool:
        return True

    def compile(self, func: Callable[..., Any]) -> Callable[..., Any]:
        self.compile_calls += 1
        self.leaf_signature = inspect.signature(func)

        def recorded(*args: Any) -> Any:
            self.leaf_arguments.append(args)
            self.leaf_calls.append(
                tuple(
                    tuple(value.shape) if hasattr(value, "shape") else ()
                    for value in args
                )
            )
            return func(*args)

        return recorded


def _inputs(
    n_baselines: int,
    n_sources: int,
    *,
    dtype: type[np.complexfloating[Any, Any]] = np.complex128,
    polarized: bool = True,
    scalar_envelope: bool = False,
) -> tuple[Any, ...]:
    rng = np.random.default_rng(71_001 + n_baselines * 100 + n_sources)
    real_dtype = np.float32 if np.dtype(dtype) == np.dtype(np.complex64) else np.float64

    def complex_array(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
            dtype
        )

    jones_p = complex_array(n_baselines, n_sources, 2, 2)
    jones_q = complex_array(n_baselines, n_sources, 2, 2)
    coherency = complex_array(n_sources, 2, 2) if polarized else None
    phase = complex_array(n_baselines, n_sources)
    envelope: float | np.ndarray
    if scalar_envelope:
        envelope = 1.0
    else:
        envelope = rng.uniform(0.1, 1.0, (n_baselines, n_sources)).astype(real_dtype)
    stokes_i = None
    if not polarized:
        stokes_i = rng.uniform(0.1, 5.0, n_sources).astype(real_dtype)
    return jones_p, jones_q, coherency, phase, envelope, stokes_i


@pytest.mark.parametrize("target", [False, True, 0, -1, 1.5, "4"])
def test_policy_rejects_nonpositive_or_noninteger_targets(target: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _baseline_contraction_for_policy(
            NumPyBackend(),
            target_kernel_pairs=target,  # type: ignore[arg-type]
        )


def test_none_is_the_only_unbounded_control() -> None:
    backend = _RecordingCompilingBackend()
    kernel = _baseline_contraction_for_policy(backend, target_kernel_pairs=None)
    values = _inputs(7, 5)

    result = kernel(*values)

    assert result.shape == (7, 2, 2)
    assert backend.compile_calls == 1
    assert [shapes[0] for shapes in backend.leaf_calls] == [(7, 5, 2, 2)]


@pytest.mark.parametrize(
    ("n_baselines", "n_sources", "target", "expected_chunks"),
    [
        (0, 5, 12, [0]),
        (5, 0, 12, [5]),
        (4, 3, 12, [4]),
        (5, 3, 12, [4, 1]),
        (10, 3, 12, [4, 4, 2]),
        (3, 7, 4, [1, 1, 1]),
    ],
)
def test_policy_chunks_only_the_baseline_axis(
    n_baselines: int,
    n_sources: int,
    target: int,
    expected_chunks: list[int],
) -> None:
    backend = _RecordingCompilingBackend()
    kernel = _baseline_contraction_for_policy(
        backend,
        target_kernel_pairs=target,
    )

    result = kernel(*_inputs(n_baselines, n_sources))

    assert result.shape == (n_baselines, 2, 2)
    assert backend.compile_calls == 1
    assert [shapes[0][0] for shapes in backend.leaf_calls] == expected_chunks
    assert all(shapes[0][1] == n_sources for shapes in backend.leaf_calls)
    assert all(shapes[2] == (n_sources, 2, 2) for shapes in backend.leaf_calls)
    if backend.leaf_arguments:
        assert all(
            arguments[2] is backend.leaf_arguments[0][2]
            for arguments in backend.leaf_arguments
        )


@pytest.mark.parametrize("backend_name", ["numpy", "dask"])
@pytest.mark.parametrize("polarized", [False, True])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("scalar_envelope", [False, True])
def test_chunked_numpy_and_dask_are_byte_identical_to_unbounded_control(
    backend_name: str,
    polarized: bool,
    dtype: type[np.complexfloating[Any, Any]],
    scalar_envelope: bool,
) -> None:
    backend = (
        NumPyBackend()
        if backend_name == "numpy"
        else DaskBackend(mode="cpu", use_dask_arrays=True)
    )
    values = _inputs(
        11,
        7,
        dtype=dtype,
        polarized=polarized,
        scalar_envelope=scalar_envelope,
    )
    backend_values = tuple(
        backend.asarray(value) if hasattr(value, "shape") else value for value in values
    )
    reference = _baseline_contraction_for_policy(
        backend,
        target_kernel_pairs=None,
    )(*backend_values)
    chunked = _baseline_contraction_for_policy(
        backend,
        target_kernel_pairs=20,
    )(*backend_values)

    reference_array = np.ascontiguousarray(backend.to_numpy(reference))
    chunked_array = np.ascontiguousarray(backend.to_numpy(chunked))
    assert chunked_array.shape == reference_array.shape == (11, 2, 2)
    assert chunked_array.dtype == reference_array.dtype == np.dtype(dtype)
    assert chunked_array.tobytes(order="C") == reference_array.tobytes(order="C")


def test_unpolarized_chunks_reuse_the_source_only_stokes_operand() -> None:
    backend = _RecordingCompilingBackend()
    values = list(_inputs(8, 5, scalar_envelope=True))
    stokes_i = np.linspace(0.5, 2.0, 5)
    values[2] = None
    values[5] = stokes_i

    result = _baseline_contraction_for_policy(
        backend,
        target_kernel_pairs=12,
    )(*values)

    assert result.shape == (8, 2, 2)
    assert len(backend.leaf_arguments) == 4
    assert all(arguments[2] is None for arguments in backend.leaf_arguments)
    assert all(arguments[4] is values[4] for arguments in backend.leaf_arguments)
    assert all(arguments[5] is stokes_i for arguments in backend.leaf_arguments)


@pytest.mark.parametrize("polarized", [False, True])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("scalar_envelope", [False, True])
def test_jax_cpu_chunking_matches_unbounded_control_within_tolerance(
    polarized: bool,
    dtype: type[np.complexfloating[Any, Any]],
    scalar_envelope: bool,
) -> None:
    precision = (
        PrecisionConfig.fast()
        if np.dtype(dtype) == np.dtype(np.complex64)
        else PrecisionConfig.standard()
    )
    backend = get_backend("jax", device="cpu", precision=precision)
    values = _inputs(
        11,
        7,
        dtype=dtype,
        polarized=polarized,
        scalar_envelope=scalar_envelope,
    )
    backend_values = tuple(
        backend.asarray(value) if hasattr(value, "shape") else value for value in values
    )
    reference = _baseline_contraction_for_policy(
        backend,
        target_kernel_pairs=None,
    )(*backend_values)
    chunked = _baseline_contraction_for_policy(
        backend,
        target_kernel_pairs=20,
    )(*backend_values)

    reference_array = np.asarray(backend.to_numpy(reference))
    chunked_array = np.asarray(backend.to_numpy(chunked))
    tolerance = 1e-5 if np.dtype(dtype) == np.dtype(np.complex64) else 1e-12
    assert chunked_array.shape == reference_array.shape == (11, 2, 2)
    assert chunked_array.dtype == reference_array.dtype == np.dtype(dtype)
    np.testing.assert_allclose(
        chunked_array,
        reference_array,
        rtol=tolerance,
        atol=tolerance,
    )


def test_public_factory_owns_the_production_target_and_six_argument_leaf() -> None:
    assert _TARGET_KERNEL_PAIRS == 131_072
    backend = _RecordingCompilingBackend()

    kernel = baseline_contraction_for(backend)
    result = kernel(*_inputs(3, 65_537, scalar_envelope=True))

    assert result.shape == (3, 2, 2)
    assert backend.compile_calls == 1
    assert backend.leaf_signature is not None
    assert tuple(backend.leaf_signature.parameters) == (
        "jones_p",
        "jones_q",
        "coherency",
        "phase",
        "envelope",
        "stokes_i",
    )
    assert [shapes[0][0] for shapes in backend.leaf_calls] == [1, 1, 1]
