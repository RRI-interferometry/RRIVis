# test/test_baseline.py
import unittest
import numpy as np
from src.baseline import generate_baselines


class TestBaseline(unittest.TestCase):

    def setUp(self):
        self.valid_antennas = {
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
        self.malformed_antennas = {
            0: {"Name": "HH136", "BeamID": 0},  # Missing Number and Position
            1: {"Name": "HH140", "Number": 140, "BeamID": 0},  # Missing Position
        }
        self.empty_antennas = {}

    def test_generate_baselines_valid(self):
        baselines = generate_baselines(self.valid_antennas)
        expected = {
            (121, 121): np.array([0.0, 0.0, 0.0]),
            (121, 136): np.array([-65.7837, 12.4057, -0.0112]),
            (121, 140): np.array([-7.3523, 12.6289, -0.1301]),
            (136, 136): np.array([0.0, 0.0, 0.0]),
            (136, 140): np.array([58.4314, 0.2232, -0.1189]),
            (140, 140): np.array([0.0, 0.0, 0.0]),
        }

        for key, value in expected.items():
            with self.subTest(baseline=key):
                np.testing.assert_array_almost_equal(
                    baselines[key],
                    value,
                    decimal=4,
                    err_msg=f"Baseline {key} mismatch. Expected {value}, got {baselines[key]}.",
                )

    def test_generate_baselines_empty(self):
        with self.assertRaises(
            ValueError, msg="Expected ValueError for empty antennas dictionary."
        ):
            generate_baselines(self.empty_antennas)

    def test_generate_baselines_malformed(self):
        with self.assertRaises(
            KeyError,
            msg="Expected KeyError for missing 'Number' or 'Position' fields in antenna data.",
        ):
            generate_baselines(self.malformed_antennas)

    def test_generate_baselines_invalid_positions(self):
        invalid_positions_antennas = {
            0: {"Name": "HH136", "Number": 136, "BeamID": 0, "Position": "invalid"},
            1: {
                "Name": "HH140",
                "Number": 140,
                "BeamID": 0,
                "Position": (-98.1662, 3.1671, -0.3008),
            },
        }
        with self.assertRaises(
            TypeError,
            msg="Expected TypeError for invalid position data in antenna metadata.",
        ):
            generate_baselines(invalid_positions_antennas)


if __name__ == "__main__":
    unittest.main()
