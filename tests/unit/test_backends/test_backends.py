"""Tests for the backend abstraction layer.

Tests cover:
- NumPyBackend functionality
- Backend factory function
- Backend availability detection
- Mathematical operations correctness
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from rrivis.backends import (
    ArrayBackend,
    BackendNotAvailableError,
    NumPyBackend,
    get_backend,
    get_backend_info,
    list_backends,
)


class TestListBackends:
    """Tests for list_backends() function."""

    def test_returns_dict(self):
        """list_backends should return a dictionary."""
        backends = list_backends()
        assert isinstance(backends, dict)

    def test_numpy_always_available(self):
        """NumPy backend should always be available."""
        backends = list_backends()
        assert backends["numpy"] is True

    def test_contains_expected_keys(self):
        """Should contain all expected backend keys."""
        backends = list_backends()
        expected_keys = {"numpy", "numba", "jax", "cuda", "dask", "jax_gpu", "jax_tpu"}
        assert expected_keys.issubset(set(backends.keys()))


class TestGetBackend:
    """Tests for get_backend() factory function."""

    def test_auto_returns_backend(self):
        """Auto mode should return a valid backend."""
        backend = get_backend("auto")
        assert isinstance(backend, ArrayBackend)
        assert backend.is_available()

    def test_numpy_backend(self):
        """Should return NumPyBackend for 'numpy'."""
        backend = get_backend("numpy")
        assert isinstance(backend, NumPyBackend)
        assert backend.name == "numpy-cpu"

    def test_cpu_alias(self):
        """'cpu' should be an alias for 'numpy'."""
        backend = get_backend("cpu")
        assert isinstance(backend, NumPyBackend)

    def test_invalid_backend_raises(self):
        """Invalid backend name should raise ValueError."""
        with pytest.raises(ValueError):
            get_backend("invalid_backend_name")

    def test_case_insensitive(self):
        """Backend name should be case-insensitive."""
        backend1 = get_backend("NumPy")
        backend2 = get_backend("NUMPY")
        backend3 = get_backend("numpy")
        assert all(isinstance(b, NumPyBackend) for b in [backend1, backend2, backend3])


class TestNumPyBackend:
    """Tests for NumPyBackend implementation."""

    @pytest.fixture
    def backend(self):
        """Create NumPy backend for tests."""
        return NumPyBackend()

    def test_name(self, backend):
        """Backend name should be 'numpy-cpu'."""
        assert backend.name == "numpy-cpu"

    def test_is_available(self, backend):
        """NumPy backend should always be available."""
        assert backend.is_available() is True

    def test_xp_is_numpy(self, backend):
        """xp property should return numpy module."""
        assert backend.xp is np

    # === Array Creation Tests ===

    def test_asarray_from_list(self, backend):
        """Should convert list to array."""
        arr = backend.asarray([1, 2, 3])
        assert isinstance(arr, np.ndarray)
        assert_array_almost_equal(arr, [1, 2, 3])

    def test_asarray_with_dtype(self, backend):
        """Should respect dtype parameter."""
        arr = backend.asarray([1, 2, 3], dtype=np.complex128)
        assert arr.dtype == np.complex128

    def test_to_numpy(self, backend):
        """to_numpy should return numpy array."""
        arr = backend.asarray([1, 2, 3])
        result = backend.to_numpy(arr)
        assert isinstance(result, np.ndarray)
        assert_array_almost_equal(result, [1, 2, 3])

    def test_zeros(self, backend):
        """Should create zero-filled array."""
        arr = backend.zeros((3, 3), dtype=np.float64)
        assert arr.shape == (3, 3)
        assert np.all(arr == 0)

    def test_ones(self, backend):
        """Should create one-filled array."""
        arr = backend.ones((2, 2), dtype=np.float64)
        assert arr.shape == (2, 2)
        assert np.all(arr == 1)

    def test_eye(self, backend):
        """Should create identity matrix."""
        arr = backend.eye(3, dtype=np.float64)
        expected = np.eye(3)
        assert_array_almost_equal(arr, expected)

    # === Mathematical Operations Tests ===

    def test_exp(self, backend):
        """Exponential function should be correct."""
        arr = backend.asarray([0, 1, 2])
        result = backend.exp(arr)
        expected = np.exp([0, 1, 2])
        assert_array_almost_equal(result, expected)

    def test_sin(self, backend):
        """Sine function should be correct."""
        arr = backend.asarray([0, np.pi / 2, np.pi])
        result = backend.sin(arr)
        expected = np.sin([0, np.pi / 2, np.pi])
        assert_array_almost_equal(result, expected)

    def test_cos(self, backend):
        """Cosine function should be correct."""
        arr = backend.asarray([0, np.pi / 2, np.pi])
        result = backend.cos(arr)
        expected = np.cos([0, np.pi / 2, np.pi])
        assert_array_almost_equal(result, expected)

    def test_sqrt(self, backend):
        """Square root should be correct."""
        arr = backend.asarray([1, 4, 9, 16])
        result = backend.sqrt(arr)
        expected = np.array([1, 2, 3, 4])
        assert_array_almost_equal(result, expected)

    def test_abs(self, backend):
        """Absolute value should be correct."""
        arr = backend.asarray([-1, 2, -3, 4])
        result = backend.abs(arr)
        expected = np.array([1, 2, 3, 4])
        assert_array_almost_equal(result, expected)

    def test_sum(self, backend):
        """Sum should be correct."""
        arr = backend.asarray([[1, 2], [3, 4]])
        assert backend.sum(arr) == 10
        assert_array_almost_equal(backend.sum(arr, axis=0), [4, 6])
        assert_array_almost_equal(backend.sum(arr, axis=1), [3, 7])

    # === Matrix Operations Tests ===

    def test_matmul_2d(self, backend):
        """2D matrix multiplication should be correct."""
        a = backend.asarray([[1, 2], [3, 4]])
        b = backend.asarray([[5, 6], [7, 8]])
        result = backend.matmul(a, b)
        expected = np.array([[19, 22], [43, 50]])
        assert_array_almost_equal(result, expected)

    def test_matmul_batched(self, backend):
        """Batched matrix multiplication should be correct."""
        a = backend.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
        b = backend.asarray([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])  # (2, 2, 2)
        result = backend.matmul(a, b)
        assert result.shape == (2, 2, 2)
        # First batch: identity multiplication
        assert_array_almost_equal(result[0], [[1, 2], [3, 4]])
        # Second batch: scaling by 2
        assert_array_almost_equal(result[1], [[10, 12], [14, 16]])

    def test_conjugate_transpose_2d(self, backend):
        """2D conjugate transpose should be correct."""
        a = backend.asarray([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        result = backend.conjugate_transpose(a)
        expected = np.array([[1 - 1j, 3 - 3j], [2 - 2j, 4 - 4j]])
        assert_array_almost_equal(result, expected)

    def test_conjugate_transpose_batched(self, backend):
        """Batched conjugate transpose should be correct."""
        a = backend.asarray([[[1 + 1j, 2], [3, 4 - 1j]]])  # (1, 2, 2)
        result = backend.conjugate_transpose(a)
        assert result.shape == (1, 2, 2)
        expected = np.array([[[1 - 1j, 3], [2, 4 + 1j]]])
        assert_array_almost_equal(result, expected)

    def test_conj(self, backend):
        """Complex conjugate should be correct."""
        arr = backend.asarray([1 + 2j, 3 - 4j, 5 + 0j])
        result = backend.conj(arr)
        expected = np.array([1 - 2j, 3 + 4j, 5 + 0j])
        assert_array_almost_equal(result, expected)

    # === Complex Number Tests ===

    def test_complex_multiply(self, backend):
        """Complex multiplication should be correct."""
        a = backend.asarray([1 + 2j, 3 + 4j])
        b = backend.asarray([5 + 6j, 7 + 8j])
        result = backend.complex_multiply(a, b)
        expected = np.array([(1 + 2j) * (5 + 6j), (3 + 4j) * (7 + 8j)])
        assert_array_almost_equal(result, expected)

    def test_real(self, backend):
        """Real part extraction should be correct."""
        arr = backend.asarray([1 + 2j, 3 - 4j])
        result = backend.real(arr)
        expected = np.array([1, 3])
        assert_array_almost_equal(result, expected)

    def test_imag(self, backend):
        """Imaginary part extraction should be correct."""
        arr = backend.asarray([1 + 2j, 3 - 4j])
        result = backend.imag(arr)
        expected = np.array([2, -4])
        assert_array_almost_equal(result, expected)

    # === Device Info Tests ===

    def test_get_device_info(self, backend):
        """get_device_info should return valid info."""
        info = backend.get_device_info()
        assert isinstance(info, dict)
        assert info["backend"] == "numpy"
        assert info["device"] == "CPU"
        assert "architecture" in info

    def test_memory_info(self, backend):
        """memory_info should return valid info."""
        info = backend.memory_info()
        assert isinstance(info, dict)
        # Should have either memory info or a note about psutil
        assert "total_bytes" in info or "note" in info

    # === RIME-specific Tests ===

    def test_rime_phase_calculation(self, backend):
        """Test RIME-like phase calculation."""
        xp = backend.xp

        # Baseline coordinates
        u, v, w = 100.0, 50.0, 10.0

        # Source direction cosines
        dir_l = xp.array([0.0, 0.1, -0.1])
        dir_m = xp.array([0.0, 0.0, 0.1])
        dir_n = xp.sqrt(1 - dir_l**2 - dir_m**2)

        # Phase calculation
        phase = -2 * xp.pi * (u * dir_l + v * dir_m + w * (dir_n - 1))
        fringe = backend.exp(1j * phase)

        assert fringe.shape == (3,)
        assert xp.abs(fringe[0]) == pytest.approx(1.0)  # Unit magnitude

    def test_jones_matrix_multiplication(self, backend):
        """Test Jones matrix chain multiplication."""
        # Create Jones matrices
        J1 = backend.asarray([[1.0, 0.1], [0.05, 0.95]], dtype=np.complex128)
        J2 = backend.asarray([[0.98, 0.02], [0.01, 0.99]], dtype=np.complex128)

        # Coherency matrix (unpolarized source)
        C = backend.asarray([[0.5, 0], [0, 0.5]], dtype=np.complex128)

        # RIME: V = J1 @ C @ J2^H
        J2_H = backend.conjugate_transpose(J2)
        temp = backend.matmul(J1, C)
        V = backend.matmul(temp, J2_H)

        assert V.shape == (2, 2)
        # For unpolarized source with near-identity Jones, V should be close to C
        assert_allclose(V, C, atol=0.1)


class TestGetBackendInfo:
    """Tests for get_backend_info() function."""

    def test_returns_dict(self):
        """Should return dictionary."""
        info = get_backend_info()
        assert isinstance(info, dict)

    def test_contains_numpy(self):
        """Should always contain numpy info."""
        info = get_backend_info()
        assert "numpy" in info
        assert info["numpy"]["backend"] == "numpy"


class TestBackendNotAvailableError:
    """Tests for BackendNotAvailableError."""

    def test_is_exception(self):
        """Should be an Exception."""
        assert issubclass(BackendNotAvailableError, Exception)

    def test_can_be_raised(self):
        """Should be raisable with message."""
        with pytest.raises(BackendNotAvailableError) as exc_info:
            raise BackendNotAvailableError("Test error message")
        assert "Test error message" in str(exc_info.value)
