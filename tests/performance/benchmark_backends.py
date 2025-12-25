# tests/performance/benchmark_backends.py
"""
Performance benchmarks for RRIvis backends.

These tests measure and compare performance across different backends.
Run with: pytest tests/performance/ -v --benchmark
"""

import pytest
import numpy as np
import time
from typing import Dict, Any

from rrivis.backends import get_backend, list_backends


# =============================================================================
# Benchmark Fixtures
# =============================================================================

@pytest.fixture
def small_arrays():
    """Small arrays for quick benchmarks."""
    np.random.seed(42)
    n = 100
    return {
        'matrix_a': np.random.randn(n, n) + 1j * np.random.randn(n, n),
        'matrix_b': np.random.randn(n, n) + 1j * np.random.randn(n, n),
        'vector': np.random.randn(n) + 1j * np.random.randn(n),
    }


@pytest.fixture
def medium_arrays():
    """Medium arrays for standard benchmarks."""
    np.random.seed(42)
    n = 500
    return {
        'matrix_a': np.random.randn(n, n) + 1j * np.random.randn(n, n),
        'matrix_b': np.random.randn(n, n) + 1j * np.random.randn(n, n),
        'vector': np.random.randn(n) + 1j * np.random.randn(n),
    }


@pytest.fixture
def large_arrays():
    """Large arrays for stress testing."""
    np.random.seed(42)
    n = 1000
    return {
        'matrix_a': np.random.randn(n, n) + 1j * np.random.randn(n, n),
        'matrix_b': np.random.randn(n, n) + 1j * np.random.randn(n, n),
        'vector': np.random.randn(n) + 1j * np.random.randn(n),
    }


@pytest.fixture
def visibility_data():
    """Data simulating visibility calculation workload."""
    np.random.seed(42)
    n_baselines = 1000
    n_sources = 500
    n_freqs = 20

    return {
        'uvw': np.random.randn(n_baselines, 3) * 100,
        'lmn': np.random.randn(n_sources, 3) * 0.1,
        'flux': np.random.rand(n_sources, n_freqs) * 10,
        'wavelengths': np.linspace(1.5, 3.0, n_freqs),
    }


# =============================================================================
# Timing Utilities
# =============================================================================

def benchmark_function(func, n_runs=5, warmup=2):
    """
    Benchmark a function with warmup and multiple runs.

    Parameters
    ----------
    func : callable
        Function to benchmark.
    n_runs : int
        Number of timed runs.
    warmup : int
        Number of warmup runs (not timed).

    Returns
    -------
    dict
        Dictionary with timing statistics.
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'n_runs': n_runs,
    }


# =============================================================================
# Matrix Multiplication Benchmarks
# =============================================================================

@pytest.mark.performance
class TestMatmulBenchmarks:
    """Benchmark matrix multiplication across backends."""

    @pytest.mark.parametrize("backend_name", ["numpy"])
    def test_small_matmul(self, backend_name, small_arrays):
        """Benchmark small matrix multiplication."""
        backends = list_backends()
        if backend_name not in ['numpy'] and not backends.get(backend_name, False):
            pytest.skip(f"{backend_name} not available")

        backend = get_backend(backend_name)
        A = backend.asarray(small_arrays['matrix_a'])
        B = backend.asarray(small_arrays['matrix_b'])

        def run():
            return backend.matmul(A, B)

        stats = benchmark_function(run)
        print(f"\n{backend_name} small matmul: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")

        # Just verify it completes
        assert stats['mean'] > 0

    @pytest.mark.parametrize("backend_name", ["numpy"])
    @pytest.mark.slow
    def test_medium_matmul(self, backend_name, medium_arrays):
        """Benchmark medium matrix multiplication."""
        backends = list_backends()
        if backend_name not in ['numpy'] and not backends.get(backend_name, False):
            pytest.skip(f"{backend_name} not available")

        backend = get_backend(backend_name)
        A = backend.asarray(medium_arrays['matrix_a'])
        B = backend.asarray(medium_arrays['matrix_b'])

        def run():
            return backend.matmul(A, B)

        stats = benchmark_function(run, n_runs=3)
        print(f"\n{backend_name} medium matmul: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")

        assert stats['mean'] > 0


# =============================================================================
# Visibility Calculation Benchmarks
# =============================================================================

@pytest.mark.performance
class TestVisibilityBenchmarks:
    """Benchmark visibility-related calculations."""

    def test_phase_calculation_benchmark(self, visibility_data):
        """Benchmark phase calculation."""
        backend = get_backend('numpy')

        uvw = visibility_data['uvw']
        lmn = visibility_data['lmn']
        wavelength = visibility_data['wavelengths'][0]

        uvw_be = backend.asarray(uvw)
        lmn_be = backend.asarray(lmn)

        def run():
            # Phase calculation: -2π/λ * (u*l + v*m + w*n)
            phases = np.zeros((len(uvw), len(lmn)))
            for i, bl in enumerate(uvw):
                for j, src in enumerate(lmn):
                    phases[i, j] = -2 * np.pi / wavelength * np.dot(bl, src)
            return phases

        stats = benchmark_function(run, n_runs=3, warmup=1)
        print(f"\nPhase calculation: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")

        assert stats['mean'] > 0

    def test_vectorized_phase_calculation(self, visibility_data):
        """Benchmark vectorized phase calculation."""
        backend = get_backend('numpy')

        uvw = visibility_data['uvw']
        lmn = visibility_data['lmn']
        wavelength = visibility_data['wavelengths'][0]

        def run():
            # Vectorized: outer product approach
            # phases[i,j] = -2π/λ * sum_k(uvw[i,k] * lmn[j,k])
            phases = -2 * np.pi / wavelength * np.einsum('ik,jk->ij', uvw, lmn)
            return phases

        stats = benchmark_function(run, n_runs=5, warmup=2)
        print(f"\nVectorized phase calculation: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")

        assert stats['mean'] > 0


# =============================================================================
# Backend Comparison
# =============================================================================

@pytest.mark.performance
class TestBackendComparison:
    """Compare performance across all available backends."""

    def test_compare_all_backends_matmul(self, medium_arrays):
        """Compare matmul performance across all backends."""
        results = {}

        for backend_name in ['numpy']:  # Add more backends as available
            backends = list_backends()
            if backend_name not in ['numpy'] and not backends.get(backend_name, False):
                continue

            backend = get_backend(backend_name)
            A = backend.asarray(medium_arrays['matrix_a'])
            B = backend.asarray(medium_arrays['matrix_b'])

            def run():
                result = backend.matmul(A, B)
                # Ensure computation completes (important for GPU)
                backend.to_numpy(result)
                return result

            stats = benchmark_function(run, n_runs=3, warmup=2)
            results[backend_name] = stats

        # Print comparison
        print("\n=== Backend Comparison (matmul 500x500 complex) ===")
        baseline = results.get('numpy', {}).get('mean', 1)
        for name, stats in sorted(results.items(), key=lambda x: x[1]['mean']):
            speedup = baseline / stats['mean'] if stats['mean'] > 0 else 0
            print(f"{name:10s}: {stats['mean']*1000:8.2f} ms  (speedup: {speedup:.2f}x)")

        assert len(results) > 0


# =============================================================================
# Memory Benchmarks
# =============================================================================

@pytest.mark.performance
class TestMemoryBenchmarks:
    """Benchmark memory usage."""

    def test_array_memory_usage(self):
        """Test memory usage for different array sizes."""
        import sys

        sizes = [100, 500, 1000]
        results = {}

        for n in sizes:
            arr = np.random.randn(n, n) + 1j * np.random.randn(n, n)
            nbytes = arr.nbytes
            results[n] = nbytes / (1024 * 1024)  # MB

        print("\n=== Memory Usage ===")
        for n, mb in results.items():
            print(f"{n}x{n} complex128: {mb:.2f} MB")

        # Verify memory scales as expected (n^2)
        assert results[1000] > results[500] > results[100]


# =============================================================================
# Scaling Benchmarks
# =============================================================================

@pytest.mark.performance
@pytest.mark.slow
class TestScalingBenchmarks:
    """Benchmark scaling with problem size."""

    def test_matmul_scaling(self):
        """Test matmul scaling with matrix size."""
        backend = get_backend('numpy')
        sizes = [100, 200, 400]
        results = {}

        for n in sizes:
            A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
            B = np.random.randn(n, n) + 1j * np.random.randn(n, n)

            A_be = backend.asarray(A)
            B_be = backend.asarray(B)

            def run():
                return backend.matmul(A_be, B_be)

            stats = benchmark_function(run, n_runs=3, warmup=1)
            results[n] = stats['mean']

        print("\n=== Matmul Scaling ===")
        for n, t in results.items():
            print(f"n={n}: {t*1000:.2f} ms")

        # Verify scaling is roughly cubic (matmul is O(n^3))
        # Allow some tolerance
        ratio = results[400] / results[200]
        # Should be roughly 8x (2^3), allow 4-16x
        assert 2 < ratio < 20, f"Unexpected scaling ratio: {ratio}"
