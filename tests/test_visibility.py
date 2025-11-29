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
                len(vis["I"]),
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


if __name__ == "__main__":
    unittest.main()
