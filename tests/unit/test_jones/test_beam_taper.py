"""Tests for illumination taper functions in rrivis.core.jones.beam.taper."""

import numpy as np
import pytest
from scipy.optimize import brentq

from rrivis.core.jones.beam.taper import (
    TAPER_FUNCTIONS,
    cosine_taper,
    gaussian_taper_pattern,
    parabolic_squared_taper,
    parabolic_taper,
    uniform_taper,
)


class TestAllTapersAtOrigin:
    """All taper functions must return 1.0 at u_beam=0."""

    @pytest.mark.parametrize(
        "taper_func",
        [
            uniform_taper,
            gaussian_taper_pattern,
            parabolic_taper,
            parabolic_squared_taper,
            cosine_taper,
        ],
        ids=[
            "uniform",
            "gaussian",
            "parabolic",
            "parabolic_squared",
            "cosine",
        ],
    )
    def test_value_at_origin_scalar(self, taper_func):
        result = taper_func(0.0)
        assert np.isclose(result, 1.0, atol=1e-14)

    @pytest.mark.parametrize(
        "taper_func",
        [
            uniform_taper,
            gaussian_taper_pattern,
            parabolic_taper,
            parabolic_squared_taper,
            cosine_taper,
        ],
        ids=[
            "uniform",
            "gaussian",
            "parabolic",
            "parabolic_squared",
            "cosine",
        ],
    )
    def test_value_at_origin_array(self, taper_func):
        u = np.array([0.0, 0.1, 0.5])
        result = taper_func(u)
        assert np.isclose(result[0], 1.0, atol=1e-14)


class TestUniformTaper:
    """Tests for uniform_taper (Airy pattern)."""

    def test_matches_airy_formula(self):
        """Verify uniform_taper matches 2*J1(pi*u)/(pi*u) directly."""
        from scipy.special import j1

        u = np.linspace(0.01, 3.0, 200)
        x = np.pi * u
        expected = 2.0 * j1(x) / x
        result = uniform_taper(u)
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_first_zero(self):
        """First zero of Airy pattern is at u = 1.22 (approx)."""
        u = np.linspace(1.20, 1.24, 1000)
        result = uniform_taper(u)
        # Find sign change
        sign_changes = np.where(np.diff(np.sign(result)))[0]
        assert len(sign_changes) > 0
        u_zero = u[sign_changes[0]]
        assert abs(u_zero - 1.22) < 0.02

    def test_real_valued(self):
        u = np.linspace(0, 5, 100)
        result = uniform_taper(u)
        assert np.all(np.isreal(result))

    def test_decreases_initially(self):
        """Pattern should decrease from the peak at u=0."""
        u = np.linspace(0, 0.5, 50)
        result = uniform_taper(u)
        assert result[0] == 1.0
        assert result[1] < result[0]
        assert result[5] < result[1]


class TestGaussianTaperPattern:
    """Tests for gaussian_taper_pattern."""

    def test_real_valued(self):
        u = np.linspace(0, 5, 100)
        result = gaussian_taper_pattern(u, edge_taper_dB=10.0)
        assert np.all(np.isreal(result))

    def test_decreases_initially(self):
        u = np.linspace(0, 1.0, 50)
        result = gaussian_taper_pattern(u, edge_taper_dB=10.0)
        assert result[0] >= result[1]
        assert result[1] >= result[5]

    def test_wider_than_uniform(self):
        """Gaussian taper should produce a wider main lobe than uniform."""
        u = np.linspace(0, 2.0, 1000)
        uniform_resp = uniform_taper(u)
        gaussian_resp = gaussian_taper_pattern(u, edge_taper_dB=10.0)

        # Find HPBW for each
        def find_hpbw(response):
            """Find u where response drops to 0.5."""
            for i in range(1, len(response)):
                if response[i] < 0.5:
                    # Linear interpolation
                    frac = (0.5 - response[i - 1]) / (response[i] - response[i - 1])
                    return u[i - 1] + frac * (u[i] - u[i - 1])
            return None

        hpbw_uniform = find_hpbw(uniform_resp)
        hpbw_gaussian = find_hpbw(gaussian_resp)
        assert hpbw_gaussian > hpbw_uniform

    def test_different_taper_levels(self):
        """Different taper levels produce different patterns."""
        u = np.linspace(0, 2.0, 1000)
        resp_5dB = gaussian_taper_pattern(u, edge_taper_dB=5.0)
        resp_15dB = gaussian_taper_pattern(u, edge_taper_dB=15.0)

        # Both start at 1.0
        np.testing.assert_allclose(resp_5dB[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(resp_15dB[0], 1.0, atol=1e-10)

        # They should differ at moderate u
        assert not np.allclose(resp_5dB, resp_15dB, atol=0.01)


class TestParabolicTaper:
    """Tests for parabolic_taper (n=1 pedestal-on-taper)."""

    def test_real_valued(self):
        u = np.linspace(0, 5, 100)
        result = parabolic_taper(u, edge_taper_dB=10.0)
        assert np.all(np.isreal(result))

    def test_decreases_initially(self):
        u = np.linspace(0, 1.0, 50)
        result = parabolic_taper(u, edge_taper_dB=10.0)
        assert result[0] == 1.0
        assert result[1] < result[0]
        assert result[5] < result[1]

    def test_hpbw_approx(self):
        """Parabolic taper half-voltage point is at u_beam ~ 0.81.

        The n=1 pedestal-on-taper with 10dB edge taper has its
        voltage half-power at u_beam ~ 0.81.
        """

        def pattern_minus_half(u_val):
            return (
                float(parabolic_taper(np.array([u_val]), edge_taper_dB=10.0)[0]) - 0.5
            )

        u_half = brentq(pattern_minus_half, 0.01, 2.0)
        expected_u_half = 0.81
        assert abs(u_half - expected_u_half) / expected_u_half < 0.10

    def test_pedestal_behavior(self):
        """At large u_beam, the pattern should be dominated by the pedestal.

        For 10dB taper, C ~ 0.316. The Airy envelope decays as ~1/u^1.5,
        so at large u the pattern amplitude should be roughly C * airy(u).
        We check that the pattern's oscillation envelope is consistent with C.
        """
        C = 10.0 ** (-10.0 / 20.0)  # ~ 0.316
        u_large = np.linspace(5.0, 10.0, 1000)
        result = parabolic_taper(u_large, edge_taper_dB=10.0)
        uniform_result = uniform_taper(u_large)

        # At large u, the parabolic component decays faster than Airy,
        # so the ratio result/uniform_result should approach C
        # Check that the ratio is in a reasonable range near C
        # Use peak-to-peak comparison of envelopes
        abs_result = np.abs(result)
        abs_uniform = np.abs(uniform_result)

        # Where uniform is significant, check the ratio
        significant = abs_uniform > 0.001
        if np.any(significant):
            ratios = abs_result[significant] / abs_uniform[significant]
            # The ratio should be near C for far sidelobes
            median_ratio = np.median(ratios)
            assert abs(median_ratio - C) / C < 0.5  # Within 50% of C


class TestParabolicSquaredTaper:
    """Tests for parabolic_squared_taper (n=2 pedestal-on-taper)."""

    def test_real_valued(self):
        u = np.linspace(0, 5, 100)
        result = parabolic_squared_taper(u, edge_taper_dB=10.0)
        assert np.all(np.isreal(result))

    def test_decreases_initially(self):
        u = np.linspace(0, 1.0, 50)
        result = parabolic_squared_taper(u, edge_taper_dB=10.0)
        assert result[0] == 1.0
        assert result[1] < result[0]

    def test_wider_than_parabolic(self):
        """Parabolic-squared should produce a wider beam than parabolic."""
        u = np.linspace(0, 2.0, 1000)
        para_resp = parabolic_taper(u, edge_taper_dB=10.0)
        para2_resp = parabolic_squared_taper(u, edge_taper_dB=10.0)

        def find_hpbw(response):
            for i in range(1, len(response)):
                if response[i] < 0.5:
                    frac = (0.5 - response[i - 1]) / (response[i] - response[i - 1])
                    return u[i - 1] + frac * (u[i] - u[i - 1])
            return None

        hpbw_para = find_hpbw(para_resp)
        hpbw_para2 = find_hpbw(para2_resp)
        assert hpbw_para2 > hpbw_para


class TestCosineTaper:
    """Tests for cosine_taper."""

    def test_value_at_origin(self):
        assert np.isclose(cosine_taper(0.0), 1.0, atol=1e-14)

    def test_singularity_at_one(self):
        """At u=+/-1, result should be pi/4 (L'Hopital limit)."""
        result_pos = cosine_taper(1.0)
        result_neg = cosine_taper(-1.0)
        assert np.isclose(result_pos, np.pi / 4.0, atol=1e-10)
        assert np.isclose(result_neg, np.pi / 4.0, atol=1e-10)

    def test_singularity_in_array(self):
        """Singularity handling works inside arrays."""
        u = np.array([-1.0, 0.0, 1.0])
        result = cosine_taper(u)
        np.testing.assert_allclose(
            result,
            [np.pi / 4.0, 1.0, np.pi / 4.0],
            atol=1e-10,
        )

    def test_real_valued(self):
        u = np.linspace(0, 3, 100)
        result = cosine_taper(u)
        assert np.all(np.isreal(result))

    def test_decreases_initially(self):
        u = np.linspace(0, 0.5, 50)
        result = cosine_taper(u)
        assert result[0] == 1.0
        assert result[1] < result[0]

    def test_hpbw_approx(self):
        """Cosine taper half-voltage point is at u_beam ~ 1.64.

        The cosine illumination pattern cos(pi*u/2)/(1-u^2) has its
        voltage half-point at u_beam ~ 1.64.
        """

        def pattern_minus_half(u_val):
            return float(cosine_taper(np.array([u_val]))[0]) - 0.5

        u_half = brentq(pattern_minus_half, 0.01, 3.0)
        expected_u_half = 1.64
        assert abs(u_half - expected_u_half) / expected_u_half < 0.05


class TestTaperFunctionsRegistry:
    """Tests for TAPER_FUNCTIONS dictionary."""

    def test_has_all_keys(self):
        expected_keys = {
            "uniform",
            "gaussian",
            "parabolic",
            "parabolic_squared",
            "cosine",
        }
        assert set(TAPER_FUNCTIONS.keys()) == expected_keys

    def test_has_exactly_five_entries(self):
        assert len(TAPER_FUNCTIONS) == 5

    def test_all_values_callable(self):
        for name, func in TAPER_FUNCTIONS.items():
            assert callable(func), f"TAPER_FUNCTIONS['{name}'] is not callable"

    def test_registry_functions_match_direct_imports(self):
        assert TAPER_FUNCTIONS["uniform"] is uniform_taper
        assert TAPER_FUNCTIONS["gaussian"] is gaussian_taper_pattern
        assert TAPER_FUNCTIONS["parabolic"] is parabolic_taper
        assert TAPER_FUNCTIONS["parabolic_squared"] is parabolic_squared_taper
        assert TAPER_FUNCTIONS["cosine"] is cosine_taper

    def test_all_registry_functions_return_one_at_origin(self):
        for name, func in TAPER_FUNCTIONS.items():
            result = func(0.0)
            assert np.isclose(result, 1.0, atol=1e-14), (
                f"TAPER_FUNCTIONS['{name}'](0.0) = {result}, expected 1.0"
            )
