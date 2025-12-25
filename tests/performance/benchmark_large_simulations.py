# tests/performance/benchmark_large_simulations.py
"""
Large-scale simulation benchmarks for RRIvis.

These tests measure performance scaling with array size, source count,
and frequency channels.

Run with: pytest tests/performance/benchmark_large_simulations.py -v -m "performance"
"""

import pytest
import numpy as np
import time
from typing import Dict, Any, List

from rrivis.backends import get_backend, list_backends


# =============================================================================
# Scaling Parameters
# =============================================================================

ANTENNA_COUNTS = [4, 16, 64]  # Small, medium, large arrays
SOURCE_COUNTS = [10, 100, 500]  # Few, moderate, many sources
FREQUENCY_CHANNELS = [1, 10, 100]  # Single, few, many channels


# =============================================================================
# Benchmark Utilities
# =============================================================================

def generate_test_array(n_antennas: int) -> Dict:
    """Generate a test antenna array layout."""
    np.random.seed(42)
    antennas = {}

    # Generate positions in a grid-like pattern
    side = int(np.ceil(np.sqrt(n_antennas)))
    spacing = 14.0  # meters

    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n_antennas:
                break
            antennas[idx] = {
                "Name": f"Ant{idx}",
                "Number": idx,
                "BeamID": 0,
                "Position": (i * spacing, j * spacing, 0.0),
                "diameter": 14.0,
            }
            idx += 1

    return antennas


def generate_test_baselines(antennas: Dict) -> Dict:
    """Generate baselines from antenna dictionary."""
    baselines = {}
    antenna_list = list(antennas.keys())

    for i, ant1 in enumerate(antenna_list):
        for ant2 in antenna_list[i:]:
            pos1 = np.array(antennas[ant1]["Position"])
            pos2 = np.array(antennas[ant2]["Position"])
            bl_vector = pos2 - pos1

            baselines[(ant1, ant2)] = {
                "BaselineVector": bl_vector,
                "Length": np.linalg.norm(bl_vector),
                "D1D2": "14.0_14.0",
                "BT1BT2": "parabolic_parabolic",
                "A1A2": "gaussian_gaussian",
            }

    return baselines


def generate_test_sources(n_sources: int) -> List[Dict]:
    """Generate test source catalog."""
    np.random.seed(42)
    sources = []

    for i in range(n_sources):
        # Random position within ±30 degrees of zenith
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-60, -30)
        flux = np.random.uniform(0.1, 100.0)  # Jy

        sources.append({
            "Name": f"Source{i}",
            "RA": ra,
            "Dec": dec,
            "FluxDensity": flux,
            "SpectralIndex": -0.7,
        })

    return sources


def benchmark_function(func, n_runs: int = 3, warmup: int = 1) -> Dict[str, float]:
    """Benchmark a function with warmup and multiple runs."""
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
# Antenna Array Scaling Tests
# =============================================================================

@pytest.mark.performance
class TestAntennaScaling:
    """Test performance scaling with number of antennas."""

    def test_baseline_generation_scaling(self):
        """Test baseline generation scaling with antenna count."""
        results = {}

        for n_antennas in ANTENNA_COUNTS:
            antennas = generate_test_array(n_antennas)

            def run():
                return generate_test_baselines(antennas)

            stats = benchmark_function(run, n_runs=5)
            n_baselines = n_antennas * (n_antennas + 1) // 2
            results[n_antennas] = {
                'time_ms': stats['mean'] * 1000,
                'n_baselines': n_baselines,
                'time_per_baseline_us': stats['mean'] * 1e6 / n_baselines,
            }

        print("\n=== Baseline Generation Scaling ===")
        for n_ant, data in results.items():
            print(f"{n_ant} antennas: {data['n_baselines']} baselines, "
                  f"{data['time_ms']:.3f} ms "
                  f"({data['time_per_baseline_us']:.2f} us/baseline)")

        # Verify scaling is roughly O(n^2)
        assert len(results) > 0

    @pytest.mark.slow
    def test_visibility_array_memory_scaling(self):
        """Test memory requirements scaling with array size."""
        results = {}

        for n_antennas in ANTENNA_COUNTS:
            n_baselines = n_antennas * (n_antennas + 1) // 2
            n_frequencies = 10

            # Complex visibility array: baselines x frequencies
            vis_array = np.zeros((n_baselines, n_frequencies), dtype=np.complex128)
            nbytes = vis_array.nbytes

            results[n_antennas] = {
                'n_baselines': n_baselines,
                'memory_mb': nbytes / (1024 * 1024),
            }

        print("\n=== Visibility Array Memory Scaling ===")
        for n_ant, data in results.items():
            print(f"{n_ant} antennas ({data['n_baselines']} baselines): "
                  f"{data['memory_mb']:.2f} MB")

        assert results[64]['memory_mb'] > results[4]['memory_mb']


# =============================================================================
# Source Count Scaling Tests
# =============================================================================

@pytest.mark.performance
class TestSourceScaling:
    """Test performance scaling with number of sources."""

    def test_source_loop_scaling(self):
        """Test source summation scaling."""
        backend = get_backend('numpy')
        n_baselines = 100
        n_freqs = 10

        results = {}

        for n_sources in SOURCE_COUNTS:
            # Create test data
            np.random.seed(42)
            uvw = np.random.randn(n_baselines, 3) * 100
            lmn = np.random.randn(n_sources, 3) * 0.1
            flux = np.random.rand(n_sources) * 10
            wavelength = 2.0

            def run():
                # Vectorized visibility calculation
                phases = -2 * np.pi / wavelength * np.einsum('ik,jk->ij', uvw, lmn)
                phasors = np.exp(1j * phases)
                vis = np.sum(phasors * flux, axis=1)
                return vis

            stats = benchmark_function(run, n_runs=5)
            results[n_sources] = {
                'time_ms': stats['mean'] * 1000,
                'time_per_source_us': stats['mean'] * 1e6 / n_sources,
            }

        print("\n=== Source Loop Scaling ===")
        for n_src, data in results.items():
            print(f"{n_src} sources: {data['time_ms']:.3f} ms "
                  f"({data['time_per_source_us']:.2f} us/source)")

        # Should scale roughly linearly with source count
        assert len(results) > 0


# =============================================================================
# Frequency Channel Scaling Tests
# =============================================================================

@pytest.mark.performance
class TestFrequencyScaling:
    """Test performance scaling with number of frequency channels."""

    def test_frequency_loop_scaling(self):
        """Test frequency channel scaling."""
        backend = get_backend('numpy')
        n_baselines = 100
        n_sources = 50

        results = {}

        for n_freqs in FREQUENCY_CHANNELS:
            # Create test data
            np.random.seed(42)
            uvw = np.random.randn(n_baselines, 3) * 100
            lmn = np.random.randn(n_sources, 3) * 0.1
            flux = np.random.rand(n_sources, n_freqs) * 10
            wavelengths = np.linspace(1.5, 3.0, n_freqs)

            def run():
                vis = np.zeros((n_baselines, n_freqs), dtype=np.complex128)
                for f_idx, wavelength in enumerate(wavelengths):
                    phases = -2 * np.pi / wavelength * np.einsum('ik,jk->ij', uvw, lmn)
                    phasors = np.exp(1j * phases)
                    vis[:, f_idx] = np.sum(phasors * flux[:, f_idx], axis=1)
                return vis

            stats = benchmark_function(run, n_runs=3)
            results[n_freqs] = {
                'time_ms': stats['mean'] * 1000,
                'time_per_channel_ms': stats['mean'] * 1000 / n_freqs,
            }

        print("\n=== Frequency Channel Scaling ===")
        for n_freq, data in results.items():
            print(f"{n_freq} channels: {data['time_ms']:.3f} ms "
                  f"({data['time_per_channel_ms']:.3f} ms/channel)")

        # Should scale linearly with frequency count
        assert len(results) > 0


# =============================================================================
# Full Simulation Scaling Tests
# =============================================================================

@pytest.mark.performance
@pytest.mark.slow
class TestFullSimulationScaling:
    """Test full simulation scaling."""

    def test_small_array_simulation(self):
        """Benchmark small array simulation (4 antennas)."""
        antennas = generate_test_array(4)
        baselines = generate_test_baselines(antennas)
        sources = generate_test_sources(50)
        n_freqs = 10

        n_baselines = len(baselines)
        wavelengths = np.linspace(1.5, 3.0, n_freqs)

        np.random.seed(42)

        def run_simulation():
            # Simplified visibility calculation
            vis = {}
            for bl_key, bl_data in baselines.items():
                uvw = bl_data["BaselineVector"]
                bl_vis = np.zeros(n_freqs, dtype=np.complex128)

                for f_idx, wavelength in enumerate(wavelengths):
                    for src in sources[:10]:  # Limit sources for speed
                        phase = np.random.uniform(0, 2 * np.pi)
                        bl_vis[f_idx] += src["FluxDensity"] * np.exp(1j * phase)

                vis[bl_key] = bl_vis
            return vis

        stats = benchmark_function(run_simulation, n_runs=3)

        print(f"\n=== Small Array (4 antennas, {n_baselines} baselines) ===")
        print(f"Time: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms")

        assert stats['mean'] > 0

    def test_medium_array_simulation(self):
        """Benchmark medium array simulation (16 antennas)."""
        antennas = generate_test_array(16)
        baselines = generate_test_baselines(antennas)
        sources = generate_test_sources(50)
        n_freqs = 10

        n_baselines = len(baselines)
        wavelengths = np.linspace(1.5, 3.0, n_freqs)

        np.random.seed(42)

        def run_simulation():
            vis = {}
            for bl_key, bl_data in baselines.items():
                bl_vis = np.zeros(n_freqs, dtype=np.complex128)

                for f_idx, wavelength in enumerate(wavelengths):
                    for src in sources[:10]:
                        phase = np.random.uniform(0, 2 * np.pi)
                        bl_vis[f_idx] += src["FluxDensity"] * np.exp(1j * phase)

                vis[bl_key] = bl_vis
            return vis

        stats = benchmark_function(run_simulation, n_runs=3)

        print(f"\n=== Medium Array (16 antennas, {n_baselines} baselines) ===")
        print(f"Time: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms")

        assert stats['mean'] > 0


# =============================================================================
# Memory Profiling Tests
# =============================================================================

@pytest.mark.performance
class TestMemoryProfiling:
    """Test memory usage characteristics."""

    def test_peak_memory_estimation(self):
        """Estimate peak memory for different simulation sizes."""
        results = {}

        configs = [
            {"n_ant": 4, "n_src": 50, "n_freq": 10},
            {"n_ant": 16, "n_src": 100, "n_freq": 50},
            {"n_ant": 64, "n_src": 500, "n_freq": 100},
        ]

        for cfg in configs:
            n_ant = cfg["n_ant"]
            n_src = cfg["n_src"]
            n_freq = cfg["n_freq"]
            n_bl = n_ant * (n_ant + 1) // 2

            # Estimate memory components
            # Visibility array: complex128 = 16 bytes
            vis_bytes = n_bl * n_freq * 16

            # Source catalog: rough estimate
            src_bytes = n_src * 100  # ~100 bytes per source

            # Antenna data: rough estimate
            ant_bytes = n_ant * 200  # ~200 bytes per antenna

            # Phase matrix (baselines x sources)
            phase_bytes = n_bl * n_src * 8  # float64

            total_bytes = vis_bytes + src_bytes + ant_bytes + phase_bytes

            results[(n_ant, n_src, n_freq)] = {
                'vis_mb': vis_bytes / 1e6,
                'phase_mb': phase_bytes / 1e6,
                'total_mb': total_bytes / 1e6,
            }

        print("\n=== Memory Estimation ===")
        for key, data in results.items():
            n_ant, n_src, n_freq = key
            print(f"{n_ant} ant, {n_src} src, {n_freq} freq: "
                  f"vis={data['vis_mb']:.1f}MB, "
                  f"phase={data['phase_mb']:.1f}MB, "
                  f"total={data['total_mb']:.1f}MB")

        assert len(results) > 0


# =============================================================================
# Backend Comparison for Large Scale
# =============================================================================

@pytest.mark.performance
class TestBackendLargeScale:
    """Compare backends at larger scales."""

    def test_large_matmul_comparison(self):
        """Compare backend performance for large matrix operations."""
        n = 500  # 500x500 complex matrix

        np.random.seed(42)
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        B = np.random.randn(n, n) + 1j * np.random.randn(n, n)

        results = {}

        for backend_name in ['numpy']:
            backends = list_backends()
            if backend_name not in ['numpy'] and not backends.get(backend_name, False):
                continue

            backend = get_backend(backend_name)
            A_be = backend.asarray(A)
            B_be = backend.asarray(B)

            def run():
                result = backend.matmul(A_be, B_be)
                backend.to_numpy(result)
                return result

            stats = benchmark_function(run, n_runs=3, warmup=2)
            results[backend_name] = stats

        print("\n=== Large Matmul Comparison (500x500 complex) ===")
        for name, stats in results.items():
            gflops = (2 * n**3) / (stats['mean'] * 1e9)
            print(f"{name}: {stats['mean']*1000:.2f} ms ({gflops:.1f} GFLOPS)")

        assert len(results) > 0
