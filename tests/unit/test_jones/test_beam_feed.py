"""Tests for feed pattern models and reflector geometry functions."""

import numpy as np
import pytest

from rrivis.core.jones.beam.feed import (
    FEED_MODELS,
    REFLECTOR_TYPES,
    cassegrain_angle,
    compute_edge_angle,
    compute_edge_taper_from_feed,
    corrugated_horn_pattern,
    dipole_ground_plane_pattern,
    feed_to_farfield_numerical,
    feed_to_taper,
    open_waveguide_pattern,
    prime_focus_angle,
)

# ---------------------------------------------------------------------------
# Corrugated horn pattern
# ---------------------------------------------------------------------------


class TestCorrugatedHornPattern:
    """Tests for corrugated_horn_pattern."""

    def test_on_axis_is_unity(self) -> None:
        """cos^q(0) = 1 for any q."""
        result = corrugated_horn_pattern(np.array([0.0]), q=1.15)
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], 1.0)

    def test_known_value_q1(self) -> None:
        """cos^1(pi/4) = cos(pi/4) ~ 0.7071."""
        result = corrugated_horn_pattern(np.array([np.pi / 4]), q=1.0)
        np.testing.assert_allclose(result[0], np.cos(np.pi / 4), rtol=1e-12)

    def test_monotonically_decreasing(self) -> None:
        """Pattern should decrease with angle in [0, pi/2)."""
        theta = np.linspace(0.0, np.pi / 2 - 0.01, 50)
        result = corrugated_horn_pattern(theta, q=1.15)
        assert np.all(np.diff(result) < 0)

    def test_array_input(self) -> None:
        """Should handle multi-element arrays."""
        theta = np.array([0.0, 0.1, 0.5, 1.0])
        result = corrugated_horn_pattern(theta)
        assert result.shape == (4,)
        assert result[0] == 1.0


# ---------------------------------------------------------------------------
# Open waveguide pattern
# ---------------------------------------------------------------------------


class TestOpenWaveguidePattern:
    """Tests for open_waveguide_pattern."""

    def test_on_axis_is_unity(self) -> None:
        """Both planes should be 1.0 at theta=0."""
        e_plane, h_plane = open_waveguide_pattern(np.array([0.0]))
        np.testing.assert_allclose(e_plane[0], 1.0)
        np.testing.assert_allclose(h_plane[0], 1.0)

    def test_e_plane_is_cosine(self) -> None:
        """E-plane is simply cos(theta)."""
        theta = np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3])
        e_plane, _ = open_waveguide_pattern(theta)
        np.testing.assert_allclose(e_plane, np.cos(theta), rtol=1e-12)

    def test_singularity_handling(self) -> None:
        """H-plane should return pi/4 at the singularity point."""
        # Singularity when 2*b*sin(theta)/lambda = 1, i.e. sin(theta) = 1/(2*b)
        b_over_lambda = 0.7
        sin_theta = 1.0 / (2.0 * b_over_lambda)
        theta = np.array([np.arcsin(sin_theta)])
        _, h_plane = open_waveguide_pattern(theta, b_over_lambda=b_over_lambda)
        np.testing.assert_allclose(h_plane[0], np.pi / 4, rtol=1e-10)

    def test_returns_tuple(self) -> None:
        """Should return a tuple of two arrays."""
        result = open_waveguide_pattern(np.array([0.0, 0.1]))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (2,)
        assert result[1].shape == (2,)


# ---------------------------------------------------------------------------
# Dipole ground-plane pattern
# ---------------------------------------------------------------------------


class TestDipoleGroundPlanePattern:
    """Tests for dipole_ground_plane_pattern."""

    def test_null_at_zenith_half_wave(self) -> None:
        """sin(2*pi*0.5*cos(0)) = sin(pi) = 0 when h=0.5 wavelengths."""
        result = dipole_ground_plane_pattern(np.array([0.0]), height_wavelengths=0.5)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-15)

    def test_quarter_wave_on_axis(self) -> None:
        """With h=0.25: cos(0)*sin(2*pi*0.25*cos(0)) = sin(pi/2) = 1."""
        result = dipole_ground_plane_pattern(np.array([0.0]), height_wavelengths=0.25)
        np.testing.assert_allclose(result[0], 1.0, rtol=1e-12)

    def test_array_input(self) -> None:
        """Should handle array inputs."""
        theta = np.linspace(0.0, np.pi / 3, 20)
        result = dipole_ground_plane_pattern(theta)
        assert result.shape == (20,)


# ---------------------------------------------------------------------------
# Reflector geometry
# ---------------------------------------------------------------------------


class TestReflectorGeometry:
    """Tests for prime_focus_angle and cassegrain_angle."""

    def test_prime_focus_on_axis(self) -> None:
        """rho=0 should give theta_feed=0."""
        result = prime_focus_angle(np.array([0.0]), focal_length=5.0)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-15)

    def test_cassegrain_m1_matches_prime_focus(self) -> None:
        """Cassegrain with M=1 should give the same angles as prime-focus."""
        rho = np.linspace(0.0, 7.0, 50)
        pf = prime_focus_angle(rho, focal_length=5.6)
        cg = cassegrain_angle(rho, focal_length=5.6, magnification=1.0)
        np.testing.assert_allclose(cg, pf, rtol=1e-14)

    def test_cassegrain_larger_m_gives_smaller_angle(self) -> None:
        """Larger magnification should produce smaller feed angles."""
        rho = np.array([3.5])
        angle_m1 = cassegrain_angle(rho, focal_length=5.6, magnification=1.0)
        angle_m5 = cassegrain_angle(rho, focal_length=5.6, magnification=5.0)
        assert angle_m5[0] < angle_m1[0]


# ---------------------------------------------------------------------------
# Edge angle
# ---------------------------------------------------------------------------


class TestComputeEdgeAngle:
    """Tests for compute_edge_angle."""

    def test_cassegrain_smaller_than_prime_focus(self) -> None:
        """Cassegrain should give a smaller edge angle for the same D, F."""
        D, F, M = 14.0, 5.6, 3.0
        pf_edge = compute_edge_angle(D, F, reflector_type="prime_focus")
        cg_edge = compute_edge_angle(D, F, reflector_type="cassegrain", magnification=M)
        assert cg_edge < pf_edge

    def test_prime_focus_formula(self) -> None:
        """Check against direct formula 2*arctan(D/(4*F))."""
        D, F = 25.0, 10.0
        expected = 2.0 * np.arctan(D / (4.0 * F))
        result = compute_edge_angle(D, F, reflector_type="prime_focus")
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_unknown_type_raises(self) -> None:
        """Should raise ValueError for unrecognised reflector type."""
        with pytest.raises(ValueError, match="Unknown reflector_type"):
            compute_edge_angle(14.0, 5.6, reflector_type="gregorian")


# ---------------------------------------------------------------------------
# Edge taper from feed
# ---------------------------------------------------------------------------


class TestComputeEdgeTaperFromFeed:
    """Tests for compute_edge_taper_from_feed."""

    def test_positive_taper(self) -> None:
        """Taper should be a positive dB value for a tapered feed."""
        taper_db = compute_edge_taper_from_feed(
            "corrugated_horn",
            dish_diameter=14.0,
            focal_length=5.6,
            feed_params={"q": 1.15},
        )
        assert taper_db > 0.0

    def test_unknown_model_raises(self) -> None:
        """Should raise KeyError for unknown feed model."""
        with pytest.raises(KeyError, match="Unknown feed model"):
            compute_edge_taper_from_feed(
                "nonexistent",
                dish_diameter=14.0,
                focal_length=5.6,
            )

    def test_higher_q_gives_more_taper(self) -> None:
        """Larger q exponent should give more edge taper."""
        taper_low = compute_edge_taper_from_feed(
            "corrugated_horn",
            dish_diameter=14.0,
            focal_length=5.6,
            feed_params={"q": 0.5},
        )
        taper_high = compute_edge_taper_from_feed(
            "corrugated_horn",
            dish_diameter=14.0,
            focal_length=5.6,
            feed_params={"q": 2.0},
        )
        assert taper_high > taper_low


# ---------------------------------------------------------------------------
# Feed to taper
# ---------------------------------------------------------------------------


class TestFeedToTaper:
    """Tests for feed_to_taper."""

    def test_centre_is_unity(self) -> None:
        """At aperture centre (rho/a=0), feed angle is 0 so pattern is 1."""
        result = feed_to_taper(
            np.array([0.0]),
            focal_length=5.6,
            aperture_radius=7.0,
            feed_model="corrugated_horn",
        )
        np.testing.assert_allclose(result[0], 1.0)

    def test_edge_less_than_centre(self) -> None:
        """Illumination at edge should be less than at centre."""
        rho_over_a = np.array([0.0, 1.0])
        result = feed_to_taper(
            rho_over_a,
            focal_length=5.6,
            aperture_radius=7.0,
            feed_model="corrugated_horn",
        )
        assert result[1] < result[0]

    def test_unknown_model_raises(self) -> None:
        """Should raise KeyError for unknown feed model."""
        with pytest.raises(KeyError, match="Unknown feed model"):
            feed_to_taper(
                np.array([0.0]),
                focal_length=5.6,
                aperture_radius=7.0,
                feed_model="nonexistent",
            )


# ---------------------------------------------------------------------------
# Far-field numerical
# ---------------------------------------------------------------------------


class TestFeedToFarfieldNumerical:
    """Tests for feed_to_farfield_numerical."""

    def test_normalized_at_zero(self) -> None:
        """E(0) should be 1.0 after normalisation."""
        u = np.array([0.0, 0.5, 1.0, 2.0])
        result = feed_to_farfield_numerical(
            "corrugated_horn",
            feed_params={"q": 1.15},
            focal_length=5.6,
            aperture_radius=7.0,
            u_beam=u,
        )
        np.testing.assert_allclose(result[0], 1.0, rtol=1e-12)

    def test_decreases_from_centre(self) -> None:
        """Far-field pattern should generally decrease away from centre."""
        u = np.linspace(0.0, 2.0, 100)
        result = feed_to_farfield_numerical(
            "corrugated_horn",
            feed_params={"q": 1.15},
            focal_length=5.6,
            aperture_radius=7.0,
            u_beam=u,
        )
        # First few values should be monotonically decreasing
        assert result[1] < result[0]
        assert result[5] < result[0]

    def test_default_u_beam(self) -> None:
        """Should work with default u_beam (None)."""
        result = feed_to_farfield_numerical(
            "corrugated_horn",
            focal_length=5.6,
            aperture_radius=7.0,
        )
        assert result.shape == (512,)
        np.testing.assert_allclose(result[0], 1.0, rtol=1e-12)

    def test_cassegrain_geometry(self) -> None:
        """Should work with Cassegrain reflector type."""
        u = np.array([0.0, 1.0])
        result = feed_to_farfield_numerical(
            "corrugated_horn",
            focal_length=5.6,
            aperture_radius=7.0,
            u_beam=u,
            reflector_type="cassegrain",
            magnification=3.0,
        )
        np.testing.assert_allclose(result[0], 1.0, rtol=1e-12)

    def test_open_waveguide_model(self) -> None:
        """Should work with open_waveguide feed model (tuple return)."""
        u = np.array([0.0, 0.5, 1.0])
        result = feed_to_farfield_numerical(
            "open_waveguide",
            focal_length=5.6,
            aperture_radius=7.0,
            u_beam=u,
        )
        np.testing.assert_allclose(result[0], 1.0, rtol=1e-12)

    def test_unknown_model_raises(self) -> None:
        """Should raise KeyError for unknown feed model."""
        with pytest.raises(KeyError, match="Unknown feed model"):
            feed_to_farfield_numerical("nonexistent")


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


class TestRegistries:
    """Tests for FEED_MODELS and REFLECTOR_TYPES registries."""

    def test_feed_models_keys(self) -> None:
        """FEED_MODELS should contain the three expected keys."""
        expected = {"corrugated_horn", "open_waveguide", "dipole_ground_plane"}
        assert set(FEED_MODELS.keys()) == expected

    def test_feed_models_are_callable(self) -> None:
        """All feed model entries should be callable."""
        for name, func in FEED_MODELS.items():
            assert callable(func), f"{name} is not callable"

    def test_reflector_types_keys(self) -> None:
        """REFLECTOR_TYPES should contain prime_focus and cassegrain."""
        expected = {"prime_focus", "cassegrain"}
        assert set(REFLECTOR_TYPES.keys()) == expected

    def test_reflector_types_are_callable(self) -> None:
        """All reflector type entries should be callable."""
        for name, func in REFLECTOR_TYPES.items():
            assert callable(func), f"{name} is not callable"
