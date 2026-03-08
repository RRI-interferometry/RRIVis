"""Tests for the numerical HPBW finder.

Validates ``compute_hpbw_numerical`` against known analytic HPBW
approximations for standard aperture taper patterns.
"""

import numpy as np

from rrivis.core.jones.beam.analytic.numerical_hpbw import compute_hpbw_numerical
from rrivis.core.jones.beam.analytic.taper import (
    gaussian_taper_pattern,
    parabolic_taper,
    uniform_taper,
)

_C: float = 299_792_458.0
"""Speed of light in m/s."""


class TestUniformTaper:
    """Numerical HPBW for uniform illumination (Airy pattern)."""

    def test_hpbw_matches_analytic(self) -> None:
        """Uniform taper HPBW should be approximately 1.02 * lambda / D."""
        freq_hz = 150e6
        diameter = 14.0
        wavelength = _C / freq_hz
        expected_hpbw = 1.02 * wavelength / diameter

        result = compute_hpbw_numerical(uniform_taper, freq_hz, diameter)

        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], expected_hpbw, rtol=0.10)


class TestGaussianTaper:
    """Numerical HPBW for Gaussian taper with 10 dB edge illumination."""

    def test_hpbw_matches_analytic(self) -> None:
        """Gaussian taper (10 dB) HPBW should be approximately 1.15 * lambda / D."""
        freq_hz = 150e6
        diameter = 14.0
        wavelength = _C / freq_hz
        expected_hpbw = 1.15 * wavelength / diameter

        result = compute_hpbw_numerical(
            gaussian_taper_pattern, freq_hz, diameter, edge_taper_dB=10.0
        )

        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], expected_hpbw, rtol=0.10)


class TestParabolicTaper:
    """Numerical HPBW for parabolic taper with 10 dB edge illumination."""

    def test_hpbw_matches_analytic(self) -> None:
        """Parabolic taper (10 dB) HPBW should be approximately 1.27 * lambda / D."""
        freq_hz = 150e6
        diameter = 14.0
        wavelength = _C / freq_hz
        expected_hpbw = 1.27 * wavelength / diameter

        result = compute_hpbw_numerical(
            parabolic_taper, freq_hz, diameter, edge_taper_dB=10.0
        )

        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], expected_hpbw, rtol=0.10)


class TestMultipleFrequencies:
    """Verify HPBW scales inversely with frequency."""

    def test_hpbw_inversely_proportional_to_frequency(self) -> None:
        """HPBW ratio should match the inverse frequency ratio (within 5%)."""
        freq_low = 100e6
        freq_high = 200e6
        diameter = 14.0

        result = compute_hpbw_numerical(
            uniform_taper, np.array([freq_low, freq_high]), diameter
        )

        assert result.shape == (2,)
        # HPBW ~ 1/freq, so ratio of HPBWs should be freq_high / freq_low
        hpbw_ratio = result[0] / result[1]
        freq_ratio = freq_high / freq_low
        np.testing.assert_allclose(hpbw_ratio, freq_ratio, rtol=0.05)


class TestScalarFrequency:
    """Ensure a single scalar frequency returns an array of length 1."""

    def test_single_frequency_returns_length_one_array(self) -> None:
        """Scalar freq_hz input should produce an ndarray with shape (1,)."""
        result = compute_hpbw_numerical(uniform_taper, 150e6, 14.0)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
