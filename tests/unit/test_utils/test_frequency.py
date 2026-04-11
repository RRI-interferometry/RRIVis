"""Tests for rrivis.utils.frequency — frequency config parsing."""

import numpy as np
import pytest

from rrivis.utils.frequency import parse_frequency_config

# ---------------------------------------------------------------------------
# Unit scaling
# ---------------------------------------------------------------------------


class TestFrequencyUnits:
    def test_hz_unit(self):
        """frequency_unit='Hz' -> no scaling applied."""
        config = {
            "starting_frequency": 100e6,
            "frequency_interval": 1e6,
            "frequency_bandwidth": 5e6,
            "frequency_unit": "Hz",
        }
        freqs = parse_frequency_config(config)
        assert len(freqs) == 6
        np.testing.assert_allclose(freqs[0], 100e6)
        np.testing.assert_allclose(freqs[-1], 105e6)

    def test_khz_unit(self):
        """frequency_unit='kHz' -> 1e3 scaling."""
        config = {
            "starting_frequency": 100000.0,
            "frequency_interval": 1000.0,
            "frequency_bandwidth": 5000.0,
            "frequency_unit": "kHz",
        }
        freqs = parse_frequency_config(config)
        assert len(freqs) == 6
        np.testing.assert_allclose(freqs[0], 100e6)

    def test_mhz_unit(self):
        """100 MHz start -> 100e6 Hz in result."""
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 5.0,
            "frequency_unit": "MHz",
        }
        freqs = parse_frequency_config(config)
        np.testing.assert_allclose(freqs[0], 100e6)

    def test_ghz_unit(self):
        """1 GHz start -> 1e9 Hz in result."""
        config = {
            "starting_frequency": 1.0,
            "frequency_interval": 0.1,
            "frequency_bandwidth": 0.5,
            "frequency_unit": "GHz",
        }
        freqs = parse_frequency_config(config)
        np.testing.assert_allclose(freqs[0], 1e9)
        assert len(freqs) == 6

    def test_unknown_unit_raises(self):
        """Unknown frequency unit 'THz' raises ValueError."""
        config = {
            "starting_frequency": 1.0,
            "frequency_interval": 0.1,
            "frequency_bandwidth": 0.5,
            "frequency_unit": "THz",
        }
        with pytest.raises(ValueError, match="Unknown frequency unit"):
            parse_frequency_config(config)


# ---------------------------------------------------------------------------
# Validation and edge cases
# ---------------------------------------------------------------------------


class TestFrequencyValidation:
    def test_bandwidth_less_than_interval_raises(self):
        """bandwidth < interval -> ValueError (zero channels)."""
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 5.0,
            "frequency_bandwidth": 1.0,
            "frequency_unit": "MHz",
        }
        with pytest.raises(ValueError, match="bandwidth"):
            parse_frequency_config(config)

    def test_single_channel(self):
        """bandwidth == interval -> 2 channels (start and end inclusive)."""
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 5.0,
            "frequency_bandwidth": 5.0,
            "frequency_unit": "MHz",
        }
        freqs = parse_frequency_config(config)
        assert len(freqs) == 2
        np.testing.assert_allclose(freqs[0], 100e6)
        np.testing.assert_allclose(freqs[1], 105e6)

    def test_default_unit_is_mhz(self):
        """When frequency_unit key is absent, MHz is assumed."""
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 3.0,
        }
        freqs = parse_frequency_config(config)
        assert len(freqs) == 4
        np.testing.assert_allclose(freqs[0], 100e6)

    def test_channel_centers_correct(self):
        """100 MHz, 1 MHz interval, 5 MHz BW -> [100,101,102,103,104,105] MHz."""
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 5.0,
            "frequency_unit": "MHz",
        }
        freqs = parse_frequency_config(config)
        expected = np.array([100, 101, 102, 103, 104, 105]) * 1e6
        np.testing.assert_allclose(freqs, expected)


# ---------------------------------------------------------------------------
# Raw frequency array passthrough (frequencies_hz key)
# ---------------------------------------------------------------------------


class TestFrequenciesHzPassthrough:
    def test_frequencies_hz_returned_directly(self):
        """When frequencies_hz is provided, it is returned as-is."""
        raw = [100e6, 105e6, 115e6, 200e6]
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 33.3,
            "frequency_bandwidth": 100.0,
            "frequency_unit": "MHz",
            "frequencies_hz": raw,
        }
        freqs = parse_frequency_config(config)
        np.testing.assert_array_equal(freqs, np.array(raw, dtype=np.float64))

    def test_non_uniform_channels_preserved(self):
        """Non-uniform channel spacing is preserved via frequencies_hz."""
        raw = [100e6, 105e6, 115e6, 200e6]
        config = {"frequencies_hz": raw}
        freqs = parse_frequency_config(config)
        diffs = np.diff(freqs)
        # Verify the diffs are non-uniform (not all equal)
        assert not np.allclose(diffs, diffs[0])
        assert len(freqs) == 4
        np.testing.assert_allclose(freqs, raw)

    def test_frequencies_hz_as_numpy_array(self):
        """frequencies_hz can be a numpy array."""
        raw = np.array([100e6, 110e6, 120e6])
        config = {"frequencies_hz": raw}
        freqs = parse_frequency_config(config)
        np.testing.assert_array_equal(freqs, raw)
        assert freqs.dtype == np.float64
