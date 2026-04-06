"""Tests for rrivis.core.sky.spectral — spectral scaling and Faraday rotation."""

import numpy as np
import pytest

from rrivis.core.sky.constants import C_LIGHT
from rrivis.core.sky.spectral import apply_faraday_rotation, compute_spectral_scale

# ---------------------------------------------------------------------------
# TestComputeSpectralScale
# ---------------------------------------------------------------------------


class TestComputeSpectralScale:
    """Tests for compute_spectral_scale()."""

    def test_power_law_identity(self):
        """freq == ref_freq => scale = 1.0 exactly, regardless of alpha."""
        alphas = np.array([-2.0, -1.0, -0.7, 0.0, 0.5, 1.0, 3.0])
        freq = 150e6
        ref_freq = 150e6

        scale = compute_spectral_scale(alphas, None, freq, ref_freq)

        np.testing.assert_array_equal(scale, np.ones(len(alphas)))

    @pytest.mark.parametrize(
        "alpha, freq, ref_freq, expected",
        [
            # alpha=-0.7, freq/ref=2 => 2^(-0.7)
            (-0.7, 200e6, 100e6, 2.0 ** (-0.7)),
            # alpha=0.0, freq/ref=2 => 1.0 (flat spectrum)
            (0.0, 200e6, 100e6, 1.0),
            # alpha=1.0, freq/ref=3 => 3.0
            (1.0, 300e6, 100e6, 3.0),
            # alpha=-2.5, freq/ref=0.5 => 0.5^(-2.5) = 2^2.5 = 4*sqrt(2)
            (-2.5, 100e6, 200e6, 4.0 * np.sqrt(2)),
        ],
        ids=["steep", "flat", "inverted", "low-to-high"],
    )
    def test_power_law_known_values(self, alpha, freq, ref_freq, expected):
        """Parametrized power-law check against hand-computed values."""
        alpha_arr = np.array([alpha])
        scale = compute_spectral_scale(alpha_arr, None, freq, ref_freq)

        np.testing.assert_allclose(scale[0], expected, rtol=1e-12)

    def test_power_law_vectorized(self):
        """Array of 100 alphas: vectorized result matches element-wise scalar."""
        rng = np.random.default_rng(42)
        alphas = rng.uniform(-3.0, 2.0, size=100)
        freq = 180e6
        ref_freq = 150e6
        ratio = freq / ref_freq

        scale = compute_spectral_scale(alphas, None, freq, ref_freq)

        expected = np.array([ratio**a for a in alphas])
        np.testing.assert_allclose(scale, expected, rtol=1e-12)

    def test_log_polynomial_reduces_to_power_law(self):
        """spectral_coeffs with 1 column [[-0.7]] gives same as alpha=-0.7."""
        alpha = np.array([-0.7])
        freq = 200e6
        ref_freq = 100e6

        scale_power = compute_spectral_scale(alpha, None, freq, ref_freq)
        coeffs_1col = np.array([[-0.7]])
        scale_logpoly = compute_spectral_scale(alpha, coeffs_1col, freq, ref_freq)

        # With 1 column, shape[1]==1 so code takes the simple power-law branch
        np.testing.assert_allclose(scale_logpoly, scale_power, rtol=1e-14)

    def test_log_polynomial_curvature(self):
        """Two-term log-polynomial differs from simple power law.

        spectral_coeffs=[[-0.7, -0.1]], freq/ref=2:
        exponent = -0.7*log10(2)^0 + -0.1*log10(2)^1
                 = -0.7 + -0.1*log10(2)
        scale = 2^exponent
        """
        alpha = np.array([-0.7])
        coeffs = np.array([[-0.7, -0.1]])
        freq = 200e6
        ref_freq = 100e6
        ratio = freq / ref_freq  # 2.0

        scale = compute_spectral_scale(alpha, coeffs, freq, ref_freq)

        log_ratio = np.log10(ratio)
        exponent = -0.7 + -0.1 * log_ratio
        expected = ratio**exponent
        np.testing.assert_allclose(scale[0], expected, rtol=1e-12)

        # Verify it differs from simple power law
        scale_simple = compute_spectral_scale(alpha, None, freq, ref_freq)
        assert not np.isclose(scale[0], scale_simple[0], atol=0, rtol=1e-6), (
            "Curvature term should produce a different result than simple power law"
        )

    def test_log_polynomial_three_terms(self):
        """Three-column spectral_coeffs with hand-computed expected value.

        coeffs = [[-0.8, 0.2, -0.05]], freq/ref = 3:
        log_ratio = log10(3)
        exponent = -0.8*(log10(3))^0 + 0.2*(log10(3))^1 + -0.05*(log10(3))^2
                 = -0.8 + 0.2*log10(3) - 0.05*(log10(3))^2
        scale = 3^exponent
        """
        alpha = np.array([-0.8])
        coeffs = np.array([[-0.8, 0.2, -0.05]])
        freq = 300e6
        ref_freq = 100e6
        ratio = freq / ref_freq  # 3.0

        scale = compute_spectral_scale(alpha, coeffs, freq, ref_freq)

        log_ratio = np.log10(ratio)
        exponent = -0.8 + 0.2 * log_ratio + -0.05 * log_ratio**2
        expected = ratio**exponent
        np.testing.assert_allclose(scale[0], expected, rtol=1e-12)

    def test_multiple_sources_different_alpha(self):
        """N=5 sources with distinct alphas, each verified independently."""
        alphas = np.array([-0.5, -0.7, -0.8, -1.0, 0.0])
        freq = 300e6
        ref_freq = 150e6
        ratio = freq / ref_freq  # 2.0

        scale = compute_spectral_scale(alphas, None, freq, ref_freq)

        assert scale.shape == (5,)
        for i, a in enumerate(alphas):
            expected = ratio**a
            np.testing.assert_allclose(
                scale[i],
                expected,
                rtol=1e-12,
                err_msg=f"Mismatch for source {i} with alpha={a}",
            )


# ---------------------------------------------------------------------------
# TestApplyFaradayRotation
# ---------------------------------------------------------------------------


class TestApplyFaradayRotation:
    """Tests for apply_faraday_rotation()."""

    def test_no_rm_is_simple_scaling(self):
        """rm=None => Q_out = Q*scale, U_out = U*scale exactly."""
        q_ref = np.array([0.5, 1.0, -0.3])
        u_ref = np.array([0.3, -0.2, 0.8])
        scale = np.array([1.5, 0.8, 2.0])
        freq = 200e6
        ref_freq = 100e6

        q_out, u_out = apply_faraday_rotation(q_ref, u_ref, None, freq, ref_freq, scale)

        np.testing.assert_array_equal(q_out, q_ref * scale)
        np.testing.assert_array_equal(u_out, u_ref * scale)

    def test_zero_rm_array_is_simple_scaling(self):
        """rm=np.zeros(N) => same result as rm=None."""
        n = 5
        rng = np.random.default_rng(123)
        q_ref = rng.standard_normal(n)
        u_ref = rng.standard_normal(n)
        scale = rng.uniform(0.5, 2.0, size=n)
        freq = 180e6
        ref_freq = 150e6

        q_none, u_none = apply_faraday_rotation(
            q_ref, u_ref, None, freq, ref_freq, scale
        )
        q_zero, u_zero = apply_faraday_rotation(
            q_ref, u_ref, np.zeros(n), freq, ref_freq, scale
        )

        np.testing.assert_array_equal(q_zero, q_none)
        np.testing.assert_array_equal(u_zero, u_none)

    def test_known_rotation_angle(self):
        """RM=1 rad/m^2, freq=150MHz, ref=200MHz: hand-computed rotation.

        delta_chi = RM * ((c/freq)^2 - (c/ref_freq)^2)
                  = 1 * ((c/150e6)^2 - (c/200e6)^2)
        Q_out = Q*scale*cos(2*delta_chi) - U*scale*sin(2*delta_chi)
        U_out = Q*scale*sin(2*delta_chi) + U*scale*cos(2*delta_chi)
        """
        freq = 150e6
        ref_freq = 200e6
        rm_val = 1.0
        q_val = 0.5
        u_val = 0.3
        # Use alpha=-0.7 power-law scale at freq/ref = 150/200 = 0.75
        scale_val = (freq / ref_freq) ** (-0.7)

        q_ref = np.array([q_val])
        u_ref = np.array([u_val])
        rm = np.array([rm_val])
        scale = np.array([scale_val])

        q_out, u_out = apply_faraday_rotation(q_ref, u_ref, rm, freq, ref_freq, scale)

        # Compute expected values manually
        lambda_obs_sq = (C_LIGHT / freq) ** 2
        lambda_ref_sq = (C_LIGHT / ref_freq) ** 2
        delta_chi = rm_val * (lambda_obs_sq - lambda_ref_sq)
        cos2 = np.cos(2.0 * delta_chi)
        sin2 = np.sin(2.0 * delta_chi)
        q_scaled = q_val * scale_val
        u_scaled = u_val * scale_val
        q_expected = q_scaled * cos2 - u_scaled * sin2
        u_expected = q_scaled * sin2 + u_scaled * cos2

        np.testing.assert_allclose(q_out[0], q_expected, rtol=1e-12)
        np.testing.assert_allclose(u_out[0], u_expected, rtol=1e-12)

        # Verify the rotation angle is non-trivial (not near 0 or pi)
        assert abs(delta_chi) > 0.1, "delta_chi should be significant for this test"

    def test_faraday_rotation_preserves_polarized_intensity(self):
        """P^2 = Q^2 + U^2 is preserved under Faraday rotation.

        |Q_out^2 + U_out^2| = |Q_in^2 + U_in^2| * scale^2
        """
        q_ref = np.array([0.5])
        u_ref = np.array([0.3])
        scale = np.array([1.0])  # scale=1 to isolate rotation effect
        freq = 150e6
        ref_freq = 200e6

        p_in_sq = q_ref**2 + u_ref**2

        for rm_val in [0.0, 1.0, 10.0, 100.0, 1000.0]:
            rm = np.array([rm_val])
            q_out, u_out = apply_faraday_rotation(
                q_ref, u_ref, rm, freq, ref_freq, scale
            )
            p_out_sq = q_out**2 + u_out**2

            np.testing.assert_allclose(
                p_out_sq,
                p_in_sq * scale**2,
                rtol=1e-12,
                err_msg=f"Polarized intensity not conserved for RM={rm_val}",
            )

    def test_faraday_at_reference_frequency(self):
        """freq == ref_freq => delta_chi = 0 => no rotation, just scaling."""
        q_ref = np.array([0.5, -0.2, 1.0, 0.0])
        u_ref = np.array([0.3, 0.8, -0.5, 0.1])
        rm = np.array([1.0, 10.0, 100.0, 1000.0])
        scale = np.array([1.5, 0.8, 2.0, 1.2])
        freq = 150e6
        ref_freq = 150e6  # same as freq

        q_out, u_out = apply_faraday_rotation(q_ref, u_ref, rm, freq, ref_freq, scale)

        # delta_chi = RM * ((c/f)^2 - (c/f)^2) = 0, so rotation is identity
        np.testing.assert_allclose(q_out, q_ref * scale, rtol=1e-14)
        np.testing.assert_allclose(u_out, u_ref * scale, rtol=1e-14)

    def test_high_rm_rapid_rotation(self):
        """RM=1000 rad/m^2 at closely spaced frequencies: P^2 still conserved."""
        q_ref = np.array([0.5, -0.3, 1.2])
        u_ref = np.array([0.3, 0.7, -0.4])
        rm = np.array([1000.0, 1000.0, 1000.0])
        scale = np.array([1.0, 1.0, 1.0])  # unit scale to isolate rotation
        freq = 150.0e6
        ref_freq = 150.1e6  # closely spaced: 100 kHz apart

        q_out, u_out = apply_faraday_rotation(q_ref, u_ref, rm, freq, ref_freq, scale)

        p_in_sq = q_ref**2 + u_ref**2
        p_out_sq = q_out**2 + u_out**2

        np.testing.assert_allclose(
            p_out_sq,
            p_in_sq,
            rtol=1e-10,
            err_msg="Polarized intensity not conserved for high RM",
        )

        # The rotation angle should be non-trivial even for small frequency gap
        lambda_obs_sq = (C_LIGHT / freq) ** 2
        lambda_ref_sq = (C_LIGHT / ref_freq) ** 2
        delta_chi = 1000.0 * (lambda_obs_sq - lambda_ref_sq)
        assert abs(delta_chi) > 0.01, (
            "High RM should produce measurable rotation even at close frequencies"
        )

    def test_vectorized_multiple_sources(self):
        """N=10 sources with different RM, Q, U: verify element-wise."""
        rng = np.random.default_rng(99)
        n = 10
        q_ref = rng.uniform(-1.0, 1.0, size=n)
        u_ref = rng.uniform(-1.0, 1.0, size=n)
        rm = rng.uniform(-50.0, 50.0, size=n)
        scale = rng.uniform(0.5, 3.0, size=n)
        freq = 200e6
        ref_freq = 150e6

        q_out, u_out = apply_faraday_rotation(q_ref, u_ref, rm, freq, ref_freq, scale)

        # Verify element-by-element against scalar formula
        lambda_obs_sq = (C_LIGHT / freq) ** 2
        lambda_ref_sq = (C_LIGHT / ref_freq) ** 2

        for i in range(n):
            delta_chi = rm[i] * (lambda_obs_sq - lambda_ref_sq)
            cos2 = np.cos(2.0 * delta_chi)
            sin2 = np.sin(2.0 * delta_chi)
            q_scaled = q_ref[i] * scale[i]
            u_scaled = u_ref[i] * scale[i]
            q_expected = q_scaled * cos2 - u_scaled * sin2
            u_expected = q_scaled * sin2 + u_scaled * cos2

            np.testing.assert_allclose(
                q_out[i],
                q_expected,
                rtol=1e-12,
                err_msg=f"Q mismatch for source {i}",
            )
            np.testing.assert_allclose(
                u_out[i],
                u_expected,
                rtol=1e-12,
                err_msg=f"U mismatch for source {i}",
            )

        # Also check polarized intensity conservation
        p_in_sq = q_ref**2 + u_ref**2
        p_out_sq = q_out**2 + u_out**2
        np.testing.assert_allclose(
            p_out_sq,
            p_in_sq * scale**2,
            rtol=1e-10,
            err_msg="Polarized intensity not conserved in vectorized test",
        )
