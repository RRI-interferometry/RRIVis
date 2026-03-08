"""Tests for aperture shape far-field pattern functions."""

import numpy as np
import pytest

from rrivis.core.jones.beam.aperture import (
    APERTURE_SHAPES,
    airy_voltage_pattern,
    compute_u_beam,
    elliptical_airy_voltage_pattern,
    sinc_voltage_pattern,
)


class TestAiryVoltagePattern:
    """Tests for the circular aperture (Airy) voltage pattern."""

    def test_on_axis_is_unity(self) -> None:
        """Airy pattern at u_beam=0 should be exactly 1.0."""
        result = airy_voltage_pattern(np.array([0.0]))
        np.testing.assert_allclose(result, [1.0], atol=1e-12)

    def test_first_null(self) -> None:
        """First null of Airy pattern at u_beam = 3.832 / pi ~ 1.22."""
        u_null = 3.8317059702075125 / np.pi  # first zero of J1(pi*u)/(pi*u)
        result = airy_voltage_pattern(np.array([u_null]))
        np.testing.assert_allclose(result, [0.0], atol=0.01)

    def test_half_power_point(self) -> None:
        """At u_beam ~ 0.514 the voltage pattern should be ~ 1/sqrt(2)."""
        u_hpbw = 0.5145
        result = airy_voltage_pattern(np.array([u_hpbw]))
        expected = 1.0 / np.sqrt(2.0)
        np.testing.assert_allclose(result, [expected], atol=0.01)

    def test_first_sidelobe_level(self) -> None:
        """First sidelobe at u_beam ~ 1.64 should be about -17.6 dB."""
        u_sidelobe = 1.6347
        result = airy_voltage_pattern(np.array([u_sidelobe]))
        amplitude = np.abs(result[0])
        db_level = 20.0 * np.log10(amplitude)
        np.testing.assert_allclose(db_level, -17.6, atol=1.0)

    def test_scalar_zero(self) -> None:
        """Scalar u_beam=0 should also return 1.0."""
        result = airy_voltage_pattern(np.float64(0.0))
        assert np.isclose(result, 1.0, atol=1e-12)

    def test_symmetry(self) -> None:
        """Pattern should be symmetric: E(u) == E(-u)."""
        u_vals = np.array([0.3, 0.7, 1.0, 1.5])
        pos = airy_voltage_pattern(u_vals)
        neg = airy_voltage_pattern(-u_vals)
        np.testing.assert_allclose(pos, neg, atol=1e-12)


class TestSincVoltagePattern:
    """Tests for the rectangular aperture (sinc) voltage pattern."""

    def test_on_axis_is_unity(self) -> None:
        """Sinc pattern at (0, 0) should be 1.0."""
        result = sinc_voltage_pattern(np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(result, [1.0], atol=1e-12)

    def test_first_null_x(self) -> None:
        """First null along x at ux=1, uy=0."""
        result = sinc_voltage_pattern(np.array([1.0]), np.array([0.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-12)

    def test_first_null_y(self) -> None:
        """First null along y at ux=0, uy=1."""
        result = sinc_voltage_pattern(np.array([0.0]), np.array([1.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-12)

    def test_sidelobe_level(self) -> None:
        """Sinc sidelobe at u=1.5: 20*log10(|sinc(1.5)|) ~ -13.26 dB."""
        result = sinc_voltage_pattern(np.array([1.5]), np.array([0.0]))
        amplitude = np.abs(result[0])
        db_level = 20.0 * np.log10(amplitude)
        np.testing.assert_allclose(db_level, -13.26, atol=1.0)

    def test_separability(self) -> None:
        """sinc(ux) * sinc(uy) should be separable in x and y."""
        ux = np.array([0.3, 0.7])
        uy = np.array([0.4, 0.9])
        result = sinc_voltage_pattern(ux, uy)
        expected = np.sinc(ux) * np.sinc(uy)
        np.testing.assert_allclose(result, expected, atol=1e-12)


class TestEllipticalAiryVoltagePattern:
    """Tests for the elliptical aperture voltage pattern."""

    def test_reduces_to_circular(self) -> None:
        """When Dx == Dy, elliptical pattern equals circular Airy."""
        diameter = 10.0
        wavelength = 1.0
        theta = np.linspace(0.01, 0.3, 50)
        phi = np.linspace(0.0, 2 * np.pi, 50)

        elliptical = elliptical_airy_voltage_pattern(
            theta, phi, diameter, diameter, wavelength
        )
        u_beam = diameter * np.sin(theta) / wavelength
        circular = airy_voltage_pattern(u_beam)

        np.testing.assert_allclose(elliptical, circular, atol=1e-12)

    def test_on_axis_is_unity(self) -> None:
        """At theta=0 the elliptical pattern should be 1.0 regardless of phi."""
        theta = np.array([0.0, 0.0, 0.0])
        phi = np.array([0.0, np.pi / 4, np.pi / 2])
        result = elliptical_airy_voltage_pattern(theta, phi, 10.0, 5.0, 1.0)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0], atol=1e-12)

    def test_asymmetry(self) -> None:
        """With Dx != Dy, pattern should differ along phi=0 vs phi=pi/2."""
        theta = np.array([0.05])
        phi_x = np.array([0.0])
        phi_y = np.array([np.pi / 2])
        result_x = elliptical_airy_voltage_pattern(theta, phi_x, 10.0, 5.0, 1.0)
        result_y = elliptical_airy_voltage_pattern(theta, phi_y, 10.0, 5.0, 1.0)
        # Wider aperture (Dx=10) should give narrower beam along phi=0
        assert result_x[0] != pytest.approx(result_y[0], abs=1e-6)


class TestComputeUBeam:
    """Tests for the u_beam computation helper."""

    def test_known_values(self) -> None:
        """Verify u_beam = D * sin(theta) * freq / c for known inputs."""
        theta = np.array([0.1])
        diameter = 10.0
        freq_hz = 1e9
        result = compute_u_beam(theta, diameter, freq_hz)
        expected = diameter * np.sin(0.1) * freq_hz / 299_792_458.0
        np.testing.assert_allclose(result, [expected], rtol=1e-12)

    def test_zero_angle(self) -> None:
        """u_beam should be 0 when theta is 0."""
        result = compute_u_beam(np.array([0.0]), 10.0, 1e9)
        np.testing.assert_allclose(result, [0.0], atol=1e-15)

    def test_array_input(self) -> None:
        """Should handle array inputs correctly."""
        theta = np.array([0.0, 0.05, 0.1, 0.2])
        result = compute_u_beam(theta, 14.0, 150e6)
        expected = 14.0 * np.sin(theta) * 150e6 / 299_792_458.0
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestApertureShapesRegistry:
    """Tests for the APERTURE_SHAPES dictionary."""

    def test_contains_expected_keys(self) -> None:
        """Registry should have circular, rectangular, and elliptical."""
        assert "circular" in APERTURE_SHAPES
        assert "rectangular" in APERTURE_SHAPES
        assert "elliptical" in APERTURE_SHAPES

    def test_circular_maps_to_airy(self) -> None:
        assert APERTURE_SHAPES["circular"] is airy_voltage_pattern

    def test_rectangular_maps_to_sinc(self) -> None:
        assert APERTURE_SHAPES["rectangular"] is sinc_voltage_pattern

    def test_elliptical_maps_to_elliptical_airy(self) -> None:
        assert APERTURE_SHAPES["elliptical"] is elliptical_airy_voltage_pattern
