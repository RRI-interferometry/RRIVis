# tests/test_visibility.py

import unittest
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u
from rrivis.core.visibility import calculate_visibility
import sys


class TestVisibilityOptimized(unittest.TestCase):
    def setUp(self):
        # Sample valid antenna data - keyed by actual antenna numbers
        self.antennas = {
            136: {
                "Name": "HH136",
                "Number": 136,
                "BeamID": 0,
                "Position": (-156.5976, 2.9439, -0.1819),
            },
            140: {
                "Name": "HH140",
                "Number": 140,
                "BeamID": 0,
                "Position": (-98.1662, 3.1671, -0.3008),
            },
            121: {
                "Name": "HH121",
                "Number": 121,
                "BeamID": 0,
                "Position": (-90.8139, -9.4618, -0.1707),
            },
        }

        # Precomputed baselines - now using dict format with BaselineVector key
        self.baselines = {
            (136, 136): {"BaselineVector": np.array([0.0, 0.0, 0.0])},
            (136, 140): {"BaselineVector": np.array([58.4314, 0.2232, -0.1189])},
            (136, 121): {"BaselineVector": np.array([65.7837, 12.4057, -0.0112])},
        }

        # Mock sources with valid coordinates
        self.sources = [
            {
                "coords": SkyCoord(ra=45.0 * u.deg, dec=45.0 * u.deg, frame="icrs"),
                "flux": 1.0,
                "spectral_index": -0.7,
            },
            {
                "coords": SkyCoord(ra=60.0 * u.deg, dec=30.0 * u.deg, frame="icrs"),
                "flux": 2.0,
                "spectral_index": -0.5,
            },
        ]

        # Observation location and time
        self.location = EarthLocation(
            lat=45.0 * u.deg, lon=-93.0 * u.deg, height=300 * u.m
        )
        self.obstime = Time("2023-01-01T00:00:00", scale="utc")

        # Frequencies and wavelengths
        self.freqs = np.array([100e6, 110e6, 120e6])  # 100, 110, 120 MHz
        self.wavelengths = (3e8 / self.freqs) * u.m  # Wavelengths in meters

        # Half Power Beam Width (HPBW) per antenna per frequency
        theta_HPBW = np.radians(5.0)
        self.hpbw_per_antenna = {
            136: np.full(len(self.freqs), theta_HPBW),
            140: np.full(len(self.freqs), theta_HPBW),
            121: np.full(len(self.freqs), theta_HPBW),
        }

    def test_valid_visibility(self):
        """Test visibility calculation with valid inputs."""
        visibilities = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
        )

        self.assertEqual(
            len(visibilities),
            len(self.baselines),
            "Mismatch in the number of baselines.",
        )
        for key, vis in visibilities.items():
            # vis is now a dict with correlation products
            self.assertIsInstance(vis, dict, f"Visibility for {key} should be a dict of correlations")
            self.assertIn("I", vis, f"Visibility for {key} should have Stokes I")
            self.assertEqual(
                vis["I"].shape[-1],
                len(self.freqs),
                f"Mismatch in the number of frequencies for baseline {key}.",
            )
            self.assertTrue(
                np.iscomplexobj(vis["I"]), f"Visibility Stokes I for {key} should be complex."
            )

    def test_empty_sources(self):
        """Test visibility calculation with no sources."""
        visibilities = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=[],  # No sources
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
        )

        for vis in visibilities.values():
            # vis is now a dict with correlation products
            self.assertIsInstance(vis, dict)
            self.assertTrue(
                np.all(vis["I"] == 0), "Expected all zero visibilities for no sources."
            )

    def test_invalid_baselines(self):
        """Test visibility calculation with malformed baselines."""
        malformed_baselines = {
            (136, 140): {"BaselineVector": np.array([58.4314, 0.2232])},  # Missing one component
        }

        with self.assertRaises(
            (ValueError, IndexError), msg="Expected ValueError or IndexError for malformed baselines."
        ):
            calculate_visibility(
                antennas=self.antennas,
                baselines=malformed_baselines,
                sources=self.sources,
                location=self.location,
                obstime=self.obstime,
                wavelengths=self.wavelengths,
                freqs=self.freqs,
                hpbw_per_antenna=self.hpbw_per_antenna,
                duration_seconds=60.0,
                time_step_seconds=60.0,
            )

    def test_memory_efficiency(self):
        """Test memory efficiency of visibilities."""
        visibilities = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
        )

        # Calculate memory usage
        total_memory_bytes = sys.getsizeof(visibilities)
        for key, value in visibilities.items():
            total_memory_bytes += sys.getsizeof(key) + sys.getsizeof(value)
            # value is now a dict of correlation products
            if isinstance(value, dict):
                for corr_key, corr_val in value.items():
                    total_memory_bytes += sys.getsizeof(corr_key) + sys.getsizeof(corr_val)
                    if hasattr(corr_val, 'nbytes'):
                        total_memory_bytes += corr_val.nbytes

        total_memory_mb = total_memory_bytes / (1024 * 1024)
        print(f"Total memory used by visibilities: {total_memory_mb:.4f} MB")
        self.assertLess(
            total_memory_mb, 1.0, "Memory usage is unexpectedly high for small data."
        )


class TestBackendIntegration(unittest.TestCase):
    """Test visibility calculation with different backends."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample antenna data
        self.antennas = {
            0: {"Name": "Ant0", "Number": 0, "BeamID": 0, "Position": (0.0, 0.0, 0.0)},
            1: {"Name": "Ant1", "Number": 1, "BeamID": 0, "Position": (100.0, 0.0, 0.0)},
        }

        # Baselines
        self.baselines = {
            (0, 1): {"BaselineVector": np.array([100.0, 0.0, 0.0])},
        }

        # Sources
        self.sources = [
            {
                "coords": SkyCoord(ra=0.0 * u.deg, dec=45.0 * u.deg, frame="icrs"),
                "flux": 1.0,
                "spectral_index": -0.7,
            },
        ]

        # Location and time
        self.location = EarthLocation(lat=45.0 * u.deg, lon=0.0 * u.deg, height=0 * u.m)
        self.obstime = Time("2023-06-21T12:00:00", scale="utc")

        # Frequencies
        self.freqs = np.array([150e6])
        self.wavelengths = (3e8 / self.freqs) * u.m

        # HPBW
        theta_HPBW = np.radians(10.0)
        self.hpbw_per_antenna = {
            0: np.array([theta_HPBW]),
            1: np.array([theta_HPBW]),
        }

    def test_default_backend(self):
        """Test that default backend (None) works."""
        vis = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            backend=None,  # Default NumPy backend
        )
        self.assertIn((0, 1), vis)
        self.assertIn("I", vis[(0, 1)])

    def test_explicit_numpy_backend(self):
        """Test with explicit NumPy backend."""
        from rrivis.backends import get_backend

        backend = get_backend("numpy")
        vis = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            backend=backend,
        )
        self.assertIn((0, 1), vis)
        self.assertIn("I", vis[(0, 1)])

    def test_backend_consistency(self):
        """Test that default and explicit numpy backend give same results."""
        from rrivis.backends import get_backend

        vis_default = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            backend=None,
        )

        backend = get_backend("numpy")
        vis_explicit = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            backend=backend,
        )

        # Results should be identical
        for key in vis_default:
            np.testing.assert_array_almost_equal(
                vis_default[key]["I"],
                vis_explicit[key]["I"],
                decimal=10,
                err_msg=f"Mismatch for baseline {key}",
            )


class TestJonesChainIntegration(unittest.TestCase):
    """Test visibility calculation with JonesChain."""

    def setUp(self):
        """Set up test fixtures."""
        self.antennas = {
            0: {"Name": "Ant0", "Number": 0, "BeamID": 0, "Position": (0.0, 0.0, 0.0)},
            1: {"Name": "Ant1", "Number": 1, "BeamID": 0, "Position": (100.0, 0.0, 0.0)},
        }

        self.baselines = {
            (0, 1): {"BaselineVector": np.array([100.0, 0.0, 0.0])},
        }

        self.sources = [
            {
                "coords": SkyCoord(ra=0.0 * u.deg, dec=45.0 * u.deg, frame="icrs"),
                "flux": 1.0,
                "spectral_index": -0.7,
            },
        ]

        self.location = EarthLocation(lat=45.0 * u.deg, lon=0.0 * u.deg, height=0 * u.m)
        self.obstime = Time("2023-06-21T12:00:00", scale="utc")

        self.freqs = np.array([150e6])
        self.wavelengths = (3e8 / self.freqs) * u.m

        theta_HPBW = np.radians(10.0)
        self.hpbw_per_antenna = {
            0: np.array([theta_HPBW]),
            1: np.array([theta_HPBW]),
        }

    def test_jones_chain_mode(self):
        """Test visibility calculation with JonesChain mode enabled."""
        vis = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            use_jones_chain=True,
        )
        self.assertIn((0, 1), vis)
        self.assertIn("I", vis[(0, 1)])
        # Should have non-zero visibility for source above horizon
        # (Note: may be zero if source is below horizon at this time/location)

    def test_jones_chain_with_gains(self):
        """Test JonesChain with gains enabled."""
        jones_config = {
            "G": {"enabled": True, "sigma": 0.0},  # No perturbation for deterministic test
        }

        vis = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            use_jones_chain=True,
            jones_config=jones_config,
        )
        self.assertIn((0, 1), vis)

    def test_jones_chain_with_bandpass(self):
        """Test JonesChain with bandpass enabled."""
        jones_config = {
            "B": {"enabled": True},
        }

        vis = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            use_jones_chain=True,
            jones_config=jones_config,
        )
        self.assertIn((0, 1), vis)

    def test_jones_chain_with_leakage(self):
        """Test JonesChain with polarization leakage enabled."""
        jones_config = {
            "D": {"enabled": True},
        }

        vis = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            use_jones_chain=True,
            jones_config=jones_config,
        )
        self.assertIn((0, 1), vis)

    def test_legacy_vs_jones_chain_similar(self):
        """Test that legacy and JonesChain modes give similar results for basic case."""
        # Legacy mode (beam only)
        vis_legacy = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            use_jones_chain=False,
        )

        # JonesChain mode (K + E only, no extra effects)
        vis_chain = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            use_jones_chain=True,
            jones_config={},  # No extra effects
        )

        # Both should have the same baselines
        self.assertEqual(set(vis_legacy.keys()), set(vis_chain.keys()))

        # Check that visibilities are returned (may differ slightly due to implementation)
        for key in vis_legacy:
            self.assertIn("I", vis_legacy[key])
            self.assertIn("I", vis_chain[key])


class TestVectorizedCoherency(unittest.TestCase):
    """Test that vectorized stokes_to_coherency matches per-source loop."""

    def test_vectorized_coherency_matches_loop(self):
        """Vectorized coherency should match per-source list comprehension."""
        from rrivis.core.polarization import stokes_to_coherency

        I = np.array([1.0, 5.0, 10.0])
        Q = np.array([0.1, 0.5, 1.0])
        U = np.array([0.0, 0.2, -0.3])
        V = np.array([0.0, 0.0, 0.5])

        # Vectorized
        C_vec = stokes_to_coherency(I, Q, U, V)

        # Per-source loop
        C_loop = np.array([
            stokes_to_coherency(Ii, Qi, Ui, Vi)
            for Ii, Qi, Ui, Vi in zip(I, Q, U, V)
        ])

        np.testing.assert_array_almost_equal(C_vec, C_loop, decimal=14)


class TestStokesIFastPath(unittest.TestCase):
    """Test Stokes-I-only fast path matches full polarization path."""

    def setUp(self):
        self.antennas = {
            0: {"Name": "Ant0", "Number": 0, "BeamID": 0, "Position": (0.0, 0.0, 0.0)},
            1: {"Name": "Ant1", "Number": 1, "BeamID": 0, "Position": (100.0, 0.0, 0.0)},
        }
        self.baselines = {
            (0, 1): {"BaselineVector": np.array([100.0, 0.0, 0.0])},
        }
        # Unpolarized sources (Q=U=V=0)
        self.sources = [
            {
                "coords": SkyCoord(ra=0.0 * u.deg, dec=45.0 * u.deg, frame="icrs"),
                "flux": 1.0,
                "spectral_index": -0.7,
                "stokes_q": 0.0,
                "stokes_u": 0.0,
                "stokes_v": 0.0,
            },
            {
                "coords": SkyCoord(ra=30.0 * u.deg, dec=60.0 * u.deg, frame="icrs"),
                "flux": 2.0,
                "spectral_index": -0.5,
                "stokes_q": 0.0,
                "stokes_u": 0.0,
                "stokes_v": 0.0,
            },
        ]
        self.location = EarthLocation(lat=45.0 * u.deg, lon=0.0 * u.deg, height=0 * u.m)
        self.obstime = Time("2023-06-21T12:00:00", scale="utc")
        self.freqs = np.array([150e6])
        self.wavelengths = (3e8 / self.freqs) * u.m
        theta_HPBW = np.radians(10.0)
        self.hpbw_per_antenna = {
            0: np.array([theta_HPBW]),
            1: np.array([theta_HPBW]),
        }

    def test_unpolarized_matches_polarized(self):
        """Stokes-I fast path should match full path when Q=U=V=0."""
        # With Q=U=V=0, the fast path triggers internally
        vis_unpol = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
        )

        # Now add tiny polarization to force full path
        sources_pol = []
        for s in self.sources:
            sp = dict(s)
            sp["stokes_q"] = 1e-30  # Tiny but nonzero
            sources_pol.append(sp)

        vis_pol = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=sources_pol,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
        )

        # Should be nearly identical
        for key in vis_unpol:
            np.testing.assert_array_almost_equal(
                vis_unpol[key]["I"], vis_pol[key]["I"], decimal=8,
                err_msg=f"Stokes I mismatch for baseline {key}"
            )


class TestUniformBeam(unittest.TestCase):
    """Test uniform beam type."""

    def setUp(self):
        self.antennas = {
            0: {"Name": "Ant0", "Number": 0, "BeamID": 0, "Position": (0.0, 0.0, 0.0)},
            1: {"Name": "Ant1", "Number": 1, "BeamID": 0, "Position": (100.0, 0.0, 0.0)},
        }
        self.baselines = {
            (0, 1): {"BaselineVector": np.array([100.0, 0.0, 0.0])},
        }
        self.sources = [
            {
                "coords": SkyCoord(ra=0.0 * u.deg, dec=45.0 * u.deg, frame="icrs"),
                "flux": 10.0,
                "spectral_index": 0.0,  # No spectral scaling
            },
        ]
        self.location = EarthLocation(lat=45.0 * u.deg, lon=0.0 * u.deg, height=0 * u.m)
        self.obstime = Time("2023-06-21T12:00:00", scale="utc")
        self.freqs = np.array([76e6])  # At reference freq
        self.wavelengths = (3e8 / self.freqs) * u.m
        theta_HPBW = np.radians(10.0)
        self.hpbw_per_antenna = {
            0: np.array([theta_HPBW]),
            1: np.array([theta_HPBW]),
        }

    def test_uniform_beam_gives_expected_result(self):
        """Uniform beam should not attenuate sources."""
        vis_uniform = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            beam_pattern_per_antenna={0: 'uniform', 1: 'uniform'},
            duration_seconds=60.0,
            time_step_seconds=60.0,
        )
        vis_gauss = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            beam_pattern_per_antenna={0: 'gaussian', 1: 'gaussian'},
            duration_seconds=60.0,
            time_step_seconds=60.0,
        )

        # Uniform beam: |V_I| >= |V_gauss_I| (no attenuation)
        for key in vis_uniform:
            amp_uniform = np.abs(vis_uniform[key]["I"])
            amp_gauss = np.abs(vis_gauss[key]["I"])
            # Uniform should give at least as much signal as Gaussian
            self.assertTrue(
                np.all(amp_uniform >= amp_gauss * 0.99),
                f"Uniform beam should not attenuate below Gaussian for baseline {key}"
            )


class TestBeamCaching(unittest.TestCase):
    """Test that per-antenna beam caching gives correct results."""

    def setUp(self):
        # 3 antennas, but baselines share antennas -> caching should kick in
        self.antennas = {
            0: {"Name": "Ant0", "Number": 0, "BeamID": 0, "Position": (0.0, 0.0, 0.0)},
            1: {"Name": "Ant1", "Number": 1, "BeamID": 0, "Position": (100.0, 0.0, 0.0)},
            2: {"Name": "Ant2", "Number": 2, "BeamID": 0, "Position": (0.0, 100.0, 0.0)},
        }
        self.baselines = {
            (0, 1): {"BaselineVector": np.array([100.0, 0.0, 0.0])},
            (0, 2): {"BaselineVector": np.array([0.0, 100.0, 0.0])},
            (1, 2): {"BaselineVector": np.array([-100.0, 100.0, 0.0])},
        }
        self.sources = [
            {
                "coords": SkyCoord(ra=0.0 * u.deg, dec=45.0 * u.deg, frame="icrs"),
                "flux": 1.0,
                "spectral_index": -0.7,
            },
        ]
        self.location = EarthLocation(lat=45.0 * u.deg, lon=0.0 * u.deg, height=0 * u.m)
        self.obstime = Time("2023-06-21T12:00:00", scale="utc")
        self.freqs = np.array([150e6])
        self.wavelengths = (3e8 / self.freqs) * u.m
        theta_HPBW = np.radians(10.0)
        self.hpbw_per_antenna = {
            0: np.array([theta_HPBW]),
            1: np.array([theta_HPBW]),
            2: np.array([theta_HPBW]),
        }

    def test_cached_results_valid(self):
        """With beam caching, all baselines should produce valid results."""
        vis = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
        )

        for key in self.baselines:
            self.assertIn(key, vis)
            self.assertIn("I", vis[key])
            # All values should be finite
            self.assertTrue(np.all(np.isfinite(vis[key]["I"])),
                          f"Non-finite visibility for baseline {key}")


class TestJonesChainEquivalence(unittest.TestCase):
    """Test legacy vs JonesChain equivalence for E-term-only case."""

    def setUp(self):
        self.antennas = {
            0: {"Name": "Ant0", "Number": 0, "BeamID": 0, "Position": (0.0, 0.0, 0.0)},
            1: {"Name": "Ant1", "Number": 1, "BeamID": 0, "Position": (100.0, 0.0, 0.0)},
        }
        self.baselines = {
            (0, 1): {"BaselineVector": np.array([100.0, 0.0, 0.0])},
        }
        self.sources = [
            {
                "coords": SkyCoord(ra=0.0 * u.deg, dec=45.0 * u.deg, frame="icrs"),
                "flux": 5.0,
                "spectral_index": -0.7,
            },
            {
                "coords": SkyCoord(ra=30.0 * u.deg, dec=60.0 * u.deg, frame="icrs"),
                "flux": 3.0,
                "spectral_index": -0.5,
            },
        ]
        self.location = EarthLocation(lat=45.0 * u.deg, lon=0.0 * u.deg, height=0 * u.m)
        self.obstime = Time("2023-06-21T12:00:00", scale="utc")
        self.freqs = np.array([150e6])
        self.wavelengths = (3e8 / self.freqs) * u.m
        theta_HPBW = np.radians(10.0)
        self.hpbw_per_antenna = {
            0: np.array([theta_HPBW]),
            1: np.array([theta_HPBW]),
        }

    def test_legacy_vs_chain_stokes_I_close(self):
        """Legacy and JonesChain paths should give very similar Stokes I."""
        vis_legacy = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            use_jones_chain=False,
        )

        vis_chain = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            use_jones_chain=True,
            jones_config={},
        )

        for key in vis_legacy:
            np.testing.assert_array_almost_equal(
                vis_legacy[key]["I"],
                vis_chain[key]["I"],
                decimal=10,
                err_msg=f"Legacy vs JonesChain Stokes I mismatch for {key}",
            )

    def test_chain_with_all_stubs_matches_legacy(self):
        """JonesChain with all stub terms enabled should match legacy."""
        jones_config = {
            "G": {"enabled": True, "sigma": 0.0},
            "B": {"enabled": True},
            "D": {"enabled": True},
        }

        vis_chain = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            use_jones_chain=True,
            jones_config=jones_config,
        )

        vis_legacy = calculate_visibility(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            hpbw_per_antenna=self.hpbw_per_antenna,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            use_jones_chain=False,
        )

        # Stubs all return identity, so results should match
        for key in vis_legacy:
            np.testing.assert_array_almost_equal(
                vis_legacy[key]["I"],
                vis_chain[key]["I"],
                decimal=10,
                err_msg=f"Stubs should not alter visibility for {key}",
            )


if __name__ == "__main__":
    unittest.main()
