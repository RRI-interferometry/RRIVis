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
        assert len(freqs) == 5
        np.testing.assert_allclose(freqs[0], 100e6)

    def test_khz_unit(self):
        """frequency_unit='kHz' -> 1e3 scaling."""
        config = {
            "starting_frequency": 100000.0,
            "frequency_interval": 1000.0,
            "frequency_bandwidth": 5000.0,
            "frequency_unit": "kHz",
        }
        freqs = parse_frequency_config(config)
        assert len(freqs) == 5
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
        assert len(freqs) == 5

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
        """bandwidth == interval -> exactly 1 channel."""
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 5.0,
            "frequency_bandwidth": 5.0,
            "frequency_unit": "MHz",
        }
        freqs = parse_frequency_config(config)
        assert len(freqs) == 1
        np.testing.assert_allclose(freqs[0], 100e6)

    def test_default_unit_is_mhz(self):
        """When frequency_unit key is absent, MHz is assumed."""
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 3.0,
        }
        freqs = parse_frequency_config(config)
        assert len(freqs) == 3
        np.testing.assert_allclose(freqs[0], 100e6)

    def test_channel_centers_correct(self):
        """100 MHz, 1 MHz interval, 5 MHz BW -> [100,101,102,103,104] MHz."""
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 5.0,
            "frequency_unit": "MHz",
        }
        freqs = parse_frequency_config(config)
        expected = np.array([100, 101, 102, 103, 104]) * 1e6
        np.testing.assert_allclose(freqs, expected)
