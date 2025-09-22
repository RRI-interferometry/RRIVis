# tests/test_observation.py

import unittest
from src.observation import get_location_and_time
from astropy.time import Time
import astropy.units as u


class TestObservation(unittest.TestCase):

    def test_default_location(self):
        """Test default HERA location and current UTC time."""
        location, obstime = get_location_and_time()
        self.assertAlmostEqual(
            location.lat.deg,
            -30.72152777777791,
            places=5,
            msg="Default latitude mismatch.",
        )
        self.assertAlmostEqual(
            location.lon.deg,
            21.428305555555557,
            places=5,
            msg="Default longitude mismatch.",
        )
        self.assertAlmostEqual(
            location.height.to(u.m).value,
            1073.0,
            places=2,
            msg="Default height mismatch.",
        )
        self.assertIsInstance(
            obstime, Time, msg="Observation time is not a Time object."
        )

    def test_custom_location(self):
        """Test custom location values."""
        lat = -30.0
        lon = 21.0
        height = 1000.0
        location, obstime = get_location_and_time(lat, lon, height)
        self.assertAlmostEqual(
            location.lat.deg, lat, places=5, msg="Custom latitude mismatch."
        )
        self.assertAlmostEqual(
            location.lon.deg, lon, places=5, msg="Custom longitude mismatch."
        )
        self.assertAlmostEqual(
            location.height.to(u.m).value,
            height,
            places=2,
            msg="Custom height mismatch.",
        )

    def test_invalid_latitude(self):
        """Test invalid latitude values."""
        with self.assertRaises(
            ValueError, msg="Expected ValueError for invalid latitude."
        ):
            get_location_and_time(lat=100)  # Invalid latitude

    def test_invalid_longitude(self):
        """Test invalid longitude values."""
        with self.assertRaises(
            ValueError, msg="Expected ValueError for invalid longitude."
        ):
            get_location_and_time(lon=200)  # Invalid longitude

    def test_invalid_height(self):
        """Test invalid height values."""
        with self.assertRaises(
            ValueError, msg="Expected ValueError for invalid height."
        ):
            get_location_and_time(height=-10)  # Invalid height

    def test_valid_start_time(self):
        """Test valid start time in ISO format."""
        start_time = "2025-01-01T12:00:00"
        _, obstime = get_location_and_time(starttime=start_time)
        self.assertEqual(
            obstime.isot.split(".")[0], start_time, msg="Start time mismatch."
        )

    def test_invalid_start_time(self):
        """Test invalid start time format."""
        with self.assertRaises(
            ValueError, msg="Expected ValueError for invalid start time format."
        ):
            get_location_and_time(starttime="invalid_time")  # Invalid start time

    def test_default_start_time(self):
        """Test default start time (current UTC time)."""
        _, obstime = get_location_and_time()
        self.assertAlmostEqual(
            obstime.to_value("unix"),
            Time.now().to_value("unix"),
            places=1,
            msg="Default start time mismatch.",
        )

    def test_partial_inputs(self):
        """Test partial input (only latitude provided)."""
        location, _ = get_location_and_time(lat=-25.0)
        self.assertAlmostEqual(
            location.lat.deg,
            -25.0,
            places=5,
            msg="Latitude mismatch for partial input.",
        )
        self.assertAlmostEqual(
            location.lon.deg,
            21.428305555555557,
            places=5,
            msg="Default longitude mismatch for partial input.",
        )
        self.assertAlmostEqual(
            location.height.to(u.m).value,
            1073.0,
            places=2,
            msg="Default height mismatch for partial input.",
        )


if __name__ == "__main__":
    unittest.main()
