# tests/test_visibility.py

import unittest
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u
from src.visibility import calculate_visibility_optimized
import sys


class TestVisibilityOptimized(unittest.TestCase):
    def setUp(self):
        # Sample valid antenna data
        self.antennas = {
            0: {
                "Name": "HH136",
                "Number": 136,
                "BeamID": 0,
                "Position": (-156.5976, 2.9439, -0.1819),
            },
            1: {
                "Name": "HH140",
                "Number": 140,
                "BeamID": 0,
                "Position": (-98.1662, 3.1671, -0.3008),
            },
            2: {
                "Name": "HH121",
                "Number": 121,
                "BeamID": 0,
                "Position": (-90.8139, -9.4618, -0.1707),
            },
        }

        # Precomputed baselines
        self.baselines = {
            (136, 136): np.array([0.0, 0.0, 0.0]),
            (136, 140): np.array([58.4314, 0.2232, -0.1189]),
            (136, 121): np.array([65.7837, 12.4057, -0.0112]),
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

        # Half Power Beam Width (HPBW) in radians
        self.theta_HPBW = np.radians(5.0)

    def test_valid_visibility(self):
        """Test visibility calculation with valid inputs."""
        visibilities = calculate_visibility_optimized(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            theta_HPBW=self.theta_HPBW,
        )

        self.assertEqual(
            len(visibilities),
            len(self.baselines),
            "Mismatch in the number of baselines.",
        )
        for key, vis in visibilities.items():
            self.assertEqual(
                len(vis),
                len(self.freqs),
                f"Mismatch in the number of frequencies for baseline {key}.",
            )
            self.assertTrue(
                np.iscomplexobj(vis), f"Visibility values for {key} are not complex."
            )

    def test_empty_sources(self):
        """Test visibility calculation with no sources."""
        visibilities = calculate_visibility_optimized(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=[],  # No sources
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            theta_HPBW=self.theta_HPBW,
        )

        for vis in visibilities.values():
            self.assertTrue(
                np.all(vis == 0), "Expected all zero visibilities for no sources."
            )

    def test_invalid_baselines(self):
        """Test visibility calculation with malformed baselines."""
        malformed_baselines = {
            (136, 140): np.array([58.4314, 0.2232]),  # Missing one component
        }

        with self.assertRaises(
            ValueError, msg="Expected ValueError for malformed baselines."
        ):
            calculate_visibility_optimized(
                antennas=self.antennas,
                baselines=malformed_baselines,
                sources=self.sources,
                location=self.location,
                obstime=self.obstime,
                wavelengths=self.wavelengths,
                freqs=self.freqs,
                theta_HPBW=self.theta_HPBW,
            )

    def test_memory_efficiency(self):
        """Test memory efficiency of visibilities."""
        visibilities = calculate_visibility_optimized(
            antennas=self.antennas,
            baselines=self.baselines,
            sources=self.sources,
            location=self.location,
            obstime=self.obstime,
            wavelengths=self.wavelengths,
            freqs=self.freqs,
            theta_HPBW=self.theta_HPBW,
        )

        # Calculate memory usage
        total_memory_bytes = sys.getsizeof(visibilities) + sum(
            sys.getsizeof(key) + sys.getsizeof(value) + value.nbytes
            for key, value in visibilities.items()
        )
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        print(f"Total memory used by visibilities: {total_memory_mb:.4f} MB")
        self.assertLess(
            total_memory_mb, 1.0, "Memory usage is unexpectedly high for small data."
        )


if __name__ == "__main__":
    unittest.main()
