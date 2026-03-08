"""Integration tests for the composed aperture beam model.

Tests that aperture shapes, tapers, feed models, and cross-polarization
models work together correctly via compute_aperture_beam.
"""

import numpy as np


class TestComposedBeamBasic:
    """Tests for compute_aperture_beam composed function."""

    def test_circular_uniform_is_airy(self):
        """Circular aperture + uniform taper = Airy pattern."""
        from rrivis.core.jones.beam.analytic import compute_aperture_beam
        from rrivis.core.jones.beam.aperture import airy_voltage_pattern, compute_u_beam

        theta = np.linspace(0.001, 0.1, 50)
        freq = 150e6
        diameter = 14.0

        jones = compute_aperture_beam(
            theta=theta,
            phi=None,
            frequency=freq,
            diameter=diameter,
            aperture_shape="circular",
            taper="uniform",
        )

        # Co-pol amplitude (diagonal elements)
        co_pol = np.abs(jones[:, 0, 0])

        # Compare with direct Airy
        u_beam = compute_u_beam(theta, diameter, freq)
        expected = np.abs(airy_voltage_pattern(u_beam))

        np.testing.assert_allclose(co_pol, expected, atol=1e-10)

    def test_circular_gaussian_taper_center(self):
        """Circular + gaussian taper: on-axis response = 1.0."""
        from rrivis.core.jones.beam.analytic import compute_aperture_beam

        jones = compute_aperture_beam(
            theta=np.array([0.0]),
            phi=None,
            frequency=150e6,
            diameter=14.0,
            aperture_shape="circular",
            taper="gaussian",
            edge_taper_dB=10.0,
        )

        # On-axis should be 1.0
        np.testing.assert_allclose(np.abs(jones[0, 0, 0]), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.abs(jones[0, 1, 1]), 1.0, atol=1e-10)

    def test_beam_narrows_with_frequency(self):
        """Beam should be narrower at higher frequency (HPBW ~ 1/freq)."""
        from rrivis.core.jones.beam.analytic import compute_aperture_beam

        theta = np.array([0.05])
        diameter = 14.0

        jones_low = compute_aperture_beam(
            theta=theta,
            phi=None,
            frequency=100e6,
            diameter=diameter,
            taper="uniform",
        )
        jones_high = compute_aperture_beam(
            theta=theta,
            phi=None,
            frequency=200e6,
            diameter=diameter,
            taper="uniform",
        )

        # At higher frequency, beam is narrower so response at fixed angle is lower
        copol_low = np.abs(jones_low[0, 0, 0])
        copol_high = np.abs(jones_high[0, 0, 0])
        assert copol_high < copol_low

    def test_scalar_input(self):
        """Scalar theta input returns (2, 2) Jones matrix."""
        from rrivis.core.jones.beam.analytic import compute_aperture_beam

        jones = compute_aperture_beam(
            theta=0.0,
            phi=None,
            frequency=150e6,
            diameter=14.0,
        )

        assert jones.shape == (2, 2)
        np.testing.assert_allclose(np.abs(jones[0, 0]), 1.0, atol=1e-10)

    def test_array_input(self):
        """Array theta input returns (N, 2, 2) Jones matrices."""
        from rrivis.core.jones.beam.analytic import compute_aperture_beam

        theta = np.array([0.0, 0.05, 0.1])
        jones = compute_aperture_beam(
            theta=theta,
            phi=None,
            frequency=150e6,
            diameter=14.0,
        )

        assert jones.shape == (3, 2, 2)


class TestComposedBeamFeed:
    """Tests for feed model integration."""

    def test_feed_derives_edge_taper(self):
        """Feed model with analytical computation derives edge taper."""
        from rrivis.core.jones.beam.analytic import compute_aperture_beam

        theta = np.array([0.0, 0.05])

        # With feed model
        jones = compute_aperture_beam(
            theta=theta,
            phi=None,
            frequency=150e6,
            diameter=14.0,
            feed_model="corrugated_horn",
            feed_params={"q": 1.15, "focal_ratio": 0.4},
            feed_computation="analytical",
        )

        assert jones.shape == (2, 2, 2)
        np.testing.assert_allclose(np.abs(jones[0, 0, 0]), 1.0, atol=1e-10)

    def test_feed_numerical_computation(self):
        """Feed model with numerical Hankel transform."""
        from rrivis.core.jones.beam.analytic import compute_aperture_beam

        theta = np.array([0.0, 0.03])

        jones = compute_aperture_beam(
            theta=theta,
            phi=None,
            frequency=150e6,
            diameter=14.0,
            feed_model="corrugated_horn",
            feed_params={"q": 1.15, "focal_ratio": 0.4},
            feed_computation="numerical",
        )

        assert jones.shape == (2, 2, 2)
        # On-axis should be 1.0 (normalized)
        np.testing.assert_allclose(np.abs(jones[0, 0, 0]), 1.0, atol=0.01)


class TestComposedBeamApertureShapes:
    """Tests for different aperture shapes."""

    def test_rectangular_aperture(self):
        """Rectangular aperture with sinc pattern."""
        from rrivis.core.jones.beam.analytic import compute_aperture_beam

        theta = np.array([0.0, 0.05])
        phi = np.array([0.0, 0.0])

        jones = compute_aperture_beam(
            theta=theta,
            phi=phi,
            frequency=150e6,
            diameter=14.0,
            aperture_shape="rectangular",
            aperture_params={"length_x": 14.0, "length_y": 10.0},
        )

        assert jones.shape == (2, 2, 2)
        np.testing.assert_allclose(np.abs(jones[0, 0, 0]), 1.0, atol=1e-10)

    def test_elliptical_aperture(self):
        """Elliptical aperture reduces to circular when Dx=Dy."""
        from rrivis.core.jones.beam.analytic import compute_aperture_beam

        theta = np.linspace(0.001, 0.05, 10)
        phi = np.zeros_like(theta)
        freq = 150e6
        diameter = 14.0

        # Circular
        jones_circ = compute_aperture_beam(
            theta=theta,
            phi=phi,
            frequency=freq,
            diameter=diameter,
            aperture_shape="circular",
            taper="uniform",
        )

        # Elliptical with equal diameters
        jones_ell = compute_aperture_beam(
            theta=theta,
            phi=phi,
            frequency=freq,
            diameter=diameter,
            aperture_shape="elliptical",
            aperture_params={"diameter_x": diameter, "diameter_y": diameter},
        )

        np.testing.assert_allclose(
            np.abs(jones_circ[:, 0, 0]),
            np.abs(jones_ell[:, 0, 0]),
            atol=0.05,  # Small tolerance for different computation paths
        )


class TestAnalyticBeamJonesNew:
    """Tests for the updated AnalyticBeamJones class."""

    def test_basic_construction(self):
        """AnalyticBeamJones can be constructed with new params."""
        from rrivis.core.jones.beam import AnalyticBeamJones

        source_altaz = np.array([[np.pi / 2, 0.0]])  # zenith
        frequencies = np.array([150e6])

        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
        )

        assert beam.name == "E"
        assert beam.is_direction_dependent is True
        assert beam.is_diagonal() is True

    def test_diameter_per_antenna(self):
        """Per-antenna diameter map is used correctly."""
        from rrivis.core.jones.beam import AnalyticBeamJones

        source_altaz = np.array([[np.pi / 4, 0.0]])  # 45 deg elevation
        frequencies = np.array([150e6])

        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
            diameter_per_antenna={0: 14.0, 1: 7.0},
        )

        assert beam._get_diameter_for_antenna(0) == 14.0
        assert beam._get_diameter_for_antenna(1) == 7.0
        assert beam._get_diameter_for_antenna(99) == 14.0  # fallback

    def test_get_config(self):
        """get_config returns expected keys."""
        from rrivis.core.jones.beam import AnalyticBeamJones

        beam = AnalyticBeamJones(
            source_altaz=np.array([[np.pi / 2, 0.0]]),
            frequencies=np.array([150e6]),
            diameter=14.0,
            taper="parabolic",
        )

        config = beam.get_config()
        assert config["diameter"] == 14.0
        assert config["taper"] == "parabolic"
        assert config["aperture_shape"] == "circular"
