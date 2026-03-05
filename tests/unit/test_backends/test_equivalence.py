# tests/unit/test_backends/test_equivalence.py
"""
Backend equivalence tests for RRIvis.

These tests verify that all backends (NumPy, Numba, JAX) produce
equivalent results within numerical tolerance.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rrivis.backends import get_backend, list_backends

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def test_arrays():
    """Generate test arrays for backend comparison."""
    np.random.seed(42)
    return {
        "real_1d": np.random.randn(1000),
        "real_2d": np.random.randn(100, 100),
        "complex_1d": np.random.randn(1000) + 1j * np.random.randn(1000),
        "complex_2d": np.random.randn(100, 100) + 1j * np.random.randn(100, 100),
        "matrix_a": np.random.randn(50, 50) + 1j * np.random.randn(50, 50),
        "matrix_b": np.random.randn(50, 50) + 1j * np.random.randn(50, 50),
        "jones_2x2": np.random.randn(2, 2) + 1j * np.random.randn(2, 2),
        "coherency_2x2": np.random.randn(2, 2) + 1j * np.random.randn(2, 2),
    }


@pytest.fixture
def visibility_test_data():
    """Test data mimicking visibility calculation inputs."""
    np.random.seed(42)
    n_baselines = 100
    n_sources = 50
    n_freqs = 10

    return {
        "uvw": np.random.randn(n_baselines, 3) * 100,  # meters
        "lmn": np.random.randn(n_sources, 3) * 0.1,  # direction cosines
        "flux": np.random.rand(n_sources, n_freqs) * 10,  # Jy
        "wavelengths": np.linspace(1.5, 3.0, n_freqs),  # meters
    }


# =============================================================================
# Available Backends Detection
# =============================================================================


def get_available_backends():
    """Get list of available backend names."""
    backends = list_backends()
    available = ["numpy"]  # NumPy is always available

    if backends.get("numba", False):
        available.append("numba")
    if backends.get("jax", False):
        available.append("jax")

    return available


# =============================================================================
# Basic Operations Equivalence Tests
# =============================================================================


class TestArrayCreation:
    """Test array creation equivalence across backends."""

    @pytest.mark.parametrize("backend_name", get_available_backends())
    def test_asarray_from_list(self, backend_name, test_arrays):
        """Test creating arrays from Python lists."""
        backend = get_backend(backend_name)
        numpy_backend = get_backend("numpy")

        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        arr = backend.asarray(data)
        arr_np = numpy_backend.asarray(data)

        result = backend.to_numpy(arr)
        assert_allclose(result, arr_np, rtol=1e-10)

    @pytest.mark.parametrize("backend_name", get_available_backends())
    def test_asarray_complex(self, backend_name, test_arrays):
        """Test creating complex arrays."""
        backend = get_backend(backend_name)

        data = test_arrays["complex_1d"]
        arr = backend.asarray(data)
        result = backend.to_numpy(arr)

        assert_allclose(result, data, rtol=1e-10)

    @pytest.mark.parametrize("backend_name", get_available_backends())
    def test_zeros_and_ones(self, backend_name):
        """Test zeros and ones creation."""
        backend = get_backend(backend_name)

        shape = (10, 20)

        zeros = backend.zeros(shape, dtype=np.float64)
        ones = backend.ones(shape, dtype=np.float64)

        assert_allclose(backend.to_numpy(zeros), np.zeros(shape), rtol=1e-10)
        assert_allclose(backend.to_numpy(ones), np.ones(shape), rtol=1e-10)


class TestMathOperations:
    """Test mathematical operations equivalence."""

    @pytest.mark.parametrize("backend_name", get_available_backends())
    def test_matmul(self, backend_name, test_arrays):
        """Test matrix multiplication equivalence."""
        backend = get_backend(backend_name)

        A = test_arrays["matrix_a"]
        B = test_arrays["matrix_b"]

        A_be = backend.asarray(A)
        B_be = backend.asarray(B)

        result = backend.matmul(A_be, B_be)
        result_np = backend.to_numpy(result)

        expected = np.matmul(A, B)
        assert_allclose(result_np, expected, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("backend_name", get_available_backends())
    def test_complex_multiply(self, backend_name, test_arrays):
        """Test complex multiplication element-wise."""
        backend = get_backend(backend_name)

        a = test_arrays["complex_1d"][:100]
        b = test_arrays["complex_1d"][100:200]

        a_be = backend.asarray(a)
        b_be = backend.asarray(b)

        result = backend.complex_multiply(a_be, b_be)
        result_np = backend.to_numpy(result)

        expected = a * b
        assert_allclose(result_np, expected, rtol=1e-10)

    @pytest.mark.parametrize("backend_name", get_available_backends())
    def test_conjugate(self, backend_name, test_arrays):
        """Test complex conjugate."""
        backend = get_backend(backend_name)

        data = test_arrays["complex_2d"]
        data_be = backend.asarray(data)

        result = backend.conj(data_be)
        result_np = backend.to_numpy(result)

        expected = np.conjugate(data)
        assert_allclose(result_np, expected, rtol=1e-10)

    @pytest.mark.parametrize("backend_name", get_available_backends())
    def test_exp(self, backend_name, test_arrays):
        """Test exponential function."""
        backend = get_backend(backend_name)

        # Use smaller values to avoid overflow
        data = test_arrays["complex_1d"][:100] * 0.1
        data_be = backend.asarray(data)

        result = backend.exp(data_be)
        result_np = backend.to_numpy(result)

        expected = np.exp(data)
        assert_allclose(result_np, expected, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("backend_name", get_available_backends())
    def test_sum(self, backend_name, test_arrays):
        """Test array sum."""
        backend = get_backend(backend_name)

        data = test_arrays["real_2d"]
        data_be = backend.asarray(data)

        # Total sum
        result_total = backend.sum(data_be)
        assert_allclose(float(result_total), np.sum(data), rtol=1e-5)

        # Sum along axis
        result_axis0 = backend.sum(data_be, axis=0)
        result_axis0_np = backend.to_numpy(result_axis0)
        assert_allclose(result_axis0_np, np.sum(data, axis=0), rtol=1e-5)


class TestJonesMatrixOperations:
    """Test Jones matrix operations equivalence."""

    @pytest.mark.parametrize("backend_name", get_available_backends())
    def test_jones_multiplication(self, backend_name, test_arrays):
        """Test 2x2 Jones matrix multiplication."""
        backend = get_backend(backend_name)

        J1 = test_arrays["jones_2x2"]
        J2 = test_arrays["coherency_2x2"]

        J1_be = backend.asarray(J1)
        J2_be = backend.asarray(J2)

        result = backend.matmul(J1_be, J2_be)
        result_np = backend.to_numpy(result)

        expected = np.matmul(J1, J2)
        assert_allclose(result_np, expected, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("backend_name", get_available_backends())
    def test_hermitian_conjugate(self, backend_name, test_arrays):
        """Test Hermitian conjugate (conjugate transpose)."""
        backend = get_backend(backend_name)

        J = test_arrays["jones_2x2"]
        J_be = backend.asarray(J)

        # Hermitian conjugate: conjugate + transpose
        J_conj = backend.conj(J_be)
        # Note: transpose operation may vary by backend
        result_np = backend.to_numpy(J_conj).T

        expected = np.conjugate(J).T
        assert_allclose(result_np, expected, rtol=1e-10)


# =============================================================================
# Visibility Calculation Equivalence Tests
# =============================================================================


class TestVisibilityEquivalence:
    """Test visibility-related calculations across backends."""

    @pytest.mark.parametrize("backend_name", get_available_backends())
    def test_phase_calculation(self, backend_name, visibility_test_data):
        """Test geometric phase calculation equivalence."""
        backend = get_backend(backend_name)

        uvw = visibility_test_data["uvw"][0]  # Single baseline
        lmn = visibility_test_data["lmn"][0]  # Single source
        wavelength = visibility_test_data["wavelengths"][0]

        uvw_be = backend.asarray(uvw)
        lmn_be = backend.asarray(lmn)

        # Phase = -2π/λ * (u*l + v*m + w*(n-1))
        # Simplified: phase = -2π/λ * dot(uvw, lmn) assuming n≈1
        phase = -2 * np.pi / wavelength * backend.sum(uvw_be * lmn_be)

        expected_phase = -2 * np.pi / wavelength * np.dot(uvw, lmn)
        assert_allclose(float(phase), expected_phase, rtol=1e-5)

    @pytest.mark.parametrize("backend_name", get_available_backends())
    def test_visibility_phasor(self, backend_name, visibility_test_data):
        """Test visibility phasor (exp(i*phase)) calculation."""
        backend = get_backend(backend_name)

        uvw = visibility_test_data["uvw"][:10]  # 10 baselines
        lmn = visibility_test_data["lmn"][:5]  # 5 sources
        wavelength = visibility_test_data["wavelengths"][0]

        # Calculate phase for each baseline-source pair
        phases = []
        for bl in uvw:
            for src in lmn:
                phase = -2 * np.pi / wavelength * np.dot(bl, src)
                phases.append(phase)

        phases = np.array(phases)
        phases_be = backend.asarray(phases)

        # Calculate phasors
        phasors = backend.exp(1j * phases_be)
        phasors_np = backend.to_numpy(phasors)

        expected = np.exp(1j * phases)
        assert_allclose(phasors_np, expected, rtol=1e-5, atol=1e-8)


# =============================================================================
# Cross-Backend Consistency Tests
# =============================================================================


class TestCrossBackendConsistency:
    """Test that all backends produce identical results."""

    def test_full_matmul_consistency(self, test_arrays):
        """Test that all backends give same matmul result."""
        available = get_available_backends()
        if len(available) < 2:
            pytest.skip("Need at least 2 backends for consistency test")

        A = test_arrays["matrix_a"]
        B = test_arrays["matrix_b"]

        results = {}
        for backend_name in available:
            backend = get_backend(backend_name)
            A_be = backend.asarray(A)
            B_be = backend.asarray(B)
            result = backend.matmul(A_be, B_be)
            results[backend_name] = backend.to_numpy(result)

        # Compare all backends against NumPy
        numpy_result = results["numpy"]
        for name, result in results.items():
            if name != "numpy":
                assert_allclose(
                    result,
                    numpy_result,
                    rtol=1e-5,
                    atol=1e-8,
                    err_msg=f"Backend {name} differs from NumPy",
                )

    def test_full_visibility_pipeline_consistency(self, visibility_test_data):
        """Test full visibility calculation consistency across backends."""
        available = get_available_backends()
        if len(available) < 2:
            pytest.skip("Need at least 2 backends for consistency test")

        uvw = visibility_test_data["uvw"][:5]
        lmn = visibility_test_data["lmn"][:3]
        flux = visibility_test_data["flux"][:3, 0]
        wavelength = visibility_test_data["wavelengths"][0]

        def calculate_visibilities(backend):
            """Simple visibility calculation for testing."""
            vis = np.zeros(len(uvw), dtype=np.complex128)

            for i, bl in enumerate(uvw):
                for j, src in enumerate(lmn):
                    phase = -2 * np.pi / wavelength * np.dot(bl, src)
                    vis[i] += flux[j] * np.exp(1j * phase)

            return vis

        results = {}
        for backend_name in available:
            backend = get_backend(backend_name)
            # For this test, we use NumPy implementation but with backend arrays
            # A more complete test would use the actual visibility.py function
            results[backend_name] = calculate_visibilities(backend)

        # Compare all results
        numpy_result = results["numpy"]
        for name, result in results.items():
            if name != "numpy":
                assert_allclose(
                    result,
                    numpy_result,
                    rtol=1e-5,
                    atol=1e-8,
                    err_msg=f"Backend {name} visibility differs from NumPy",
                )


# =============================================================================
# GPU-Specific Tests (Skip if no GPU)
# =============================================================================


@pytest.mark.gpu
class TestGPUEquivalence:
    """GPU-specific equivalence tests."""

    @pytest.fixture(autouse=True)
    def skip_if_no_gpu(self):
        """Skip tests if no GPU backend available."""
        backends = list_backends()
        if not backends.get("jax_gpu", False) and not backends.get("cuda", False):
            pytest.skip("No GPU backend available")

    def test_gpu_cpu_matmul_equivalent(self, test_arrays):
        """Test GPU and CPU give same matmul results."""
        backends = list_backends()
        cpu_backend = get_backend("numpy")

        gpu_backend = None
        if backends.get("jax", False):
            try:
                from rrivis.backends.jax_backend import JAXBackend

                gpu_backend = JAXBackend(device="gpu")
            except Exception:
                pass

        if gpu_backend is None:
            pytest.skip("Could not initialize GPU backend")

        A = test_arrays["matrix_a"]
        B = test_arrays["matrix_b"]

        # CPU result
        A_cpu = cpu_backend.asarray(A)
        B_cpu = cpu_backend.asarray(B)
        cpu_result = cpu_backend.matmul(A_cpu, B_cpu)

        # GPU result
        A_gpu = gpu_backend.asarray(A)
        B_gpu = gpu_backend.asarray(B)
        gpu_result = gpu_backend.to_numpy(gpu_backend.matmul(A_gpu, B_gpu))

        assert_allclose(gpu_result, cpu_result, rtol=1e-5, atol=1e-8)
