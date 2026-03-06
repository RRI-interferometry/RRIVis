"""
Tests for visibility_healpix.py — HEALPix beam weighting (Item 8).
"""

import numpy as np

from rrivis.core.visibility_healpix import (
    _compute_beam_power_pattern,
    compute_beam_squared_integral,
)


class TestBeamPowerPattern:
    """Tests for _compute_beam_power_pattern()."""

    def test_unity_at_zenith_gaussian(self):
        """Gaussian beam power should be 1.0 at zenith."""
        za = np.array([0.0])
        power = _compute_beam_power_pattern(
            za, hpbw_rad=np.deg2rad(10.0), beam_type="gaussian"
        )
        np.testing.assert_almost_equal(power[0], 1.0)

    def test_unity_at_zenith_cosine(self):
        """Cosine beam power should be 1.0 at zenith."""
        za = np.array([0.0])
        power = _compute_beam_power_pattern(
            za, hpbw_rad=np.deg2rad(30.0), beam_type="cosine"
        )
        np.testing.assert_almost_equal(power[0], 1.0)

    def test_unity_at_zenith_exponential(self):
        """Exponential beam power should be 1.0 at zenith."""
        za = np.array([0.0])
        power = _compute_beam_power_pattern(
            za, hpbw_rad=np.deg2rad(10.0), beam_type="exponential"
        )
        np.testing.assert_almost_equal(power[0], 1.0)

    def test_decreases_off_axis(self):
        """Beam power should decrease away from zenith."""
        za = np.array([0.0, np.deg2rad(5.0), np.deg2rad(15.0), np.deg2rad(30.0)])
        power = _compute_beam_power_pattern(za, hpbw_rad=np.deg2rad(10.0))
        # Should be monotonically decreasing for Gaussian
        assert np.all(np.diff(power) < 0)

    def test_different_hpbw_different_power(self):
        """Different HPBW should produce different power patterns."""
        za = np.array([np.deg2rad(10.0)])
        power_narrow = _compute_beam_power_pattern(za, hpbw_rad=np.deg2rad(5.0))
        power_wide = _compute_beam_power_pattern(za, hpbw_rad=np.deg2rad(30.0))
        # Wider beam should have more power at 10 degrees off-axis
        assert power_wide[0] > power_narrow[0]

    def test_power_is_voltage_squared(self):
        """Power pattern should be voltage pattern squared."""
        from rrivis.core.jones.beam.analytic import gaussian_A_theta_EBeam

        za = np.linspace(0, np.deg2rad(30), 10)
        hpbw = np.deg2rad(10.0)
        voltage = gaussian_A_theta_EBeam(za, hpbw)
        expected_power = np.asarray(voltage) ** 2
        actual_power = _compute_beam_power_pattern(
            za, hpbw_rad=hpbw, beam_type="gaussian"
        )
        np.testing.assert_allclose(actual_power, expected_power)

    def test_beam_manager_fallback(self):
        """When beam_manager returns None, falls back to analytic."""
        from unittest.mock import Mock

        mock_manager = Mock()
        mock_manager.get_jones_matrix.return_value = None

        za = np.array([0.0, np.deg2rad(5.0)])
        power = _compute_beam_power_pattern(
            za,
            hpbw_rad=np.deg2rad(10.0),
            beam_manager=mock_manager,
            antenna_number=0,
            azimuth=np.zeros(2),
            frequency=150e6,
        )
        # Should still return valid power (fallen back to analytic)
        np.testing.assert_almost_equal(power[0], 1.0)
        assert power[1] < 1.0


class TestBeamSquaredIntegral:
    """Tests for compute_beam_squared_integral()."""

    def test_uniform_beam(self):
        """Uniform beam (B^2=1 everywhere) gives total solid angle."""
        n_pixels = 100
        omega_pix = 4 * np.pi / n_pixels
        beam_power = np.ones(n_pixels)
        result = compute_beam_squared_integral(beam_power, omega_pix)
        np.testing.assert_almost_equal(result, 4 * np.pi, decimal=5)

    def test_zero_beam(self):
        """Zero beam gives zero integral."""
        beam_power = np.zeros(100)
        result = compute_beam_squared_integral(beam_power, 0.01)
        assert result == 0.0

    def test_partial_beam(self):
        """Partial beam gives less than uniform."""
        n_pixels = 100
        omega_pix = 4 * np.pi / n_pixels
        beam_power = np.ones(n_pixels) * 0.5  # Half power everywhere
        result = compute_beam_squared_integral(beam_power, omega_pix)
        assert result < 4 * np.pi
        np.testing.assert_almost_equal(result, 4 * np.pi * 0.5, decimal=5)
