"""Tests for rrivis.core.sky._analysis — pure compute functions."""

import healpy as hp
import numpy as np
import pytest

from rrivis.core.sky._analysis import (
    F_21_HZ,
    compute_angular_power_spectrum,
    compute_cross_cell,
    compute_delay_spectrum,
    compute_frequency_correlation,
    compute_kparallel,
    filter_ell_band,
    gaussianity_stats,
)

# ---------------------------------------------------------------------------
# Angular power spectrum
# ---------------------------------------------------------------------------


class TestAngularPowerSpectrum:
    def test_shape_default_lmax(self):
        nside = 8
        m = np.random.default_rng(0).normal(0, 1, hp.nside2npix(nside))
        cl = compute_angular_power_spectrum(m)
        assert cl.shape == (3 * nside,)

    def test_shape_custom_lmax(self):
        nside = 16
        m = np.random.default_rng(0).normal(0, 1, hp.nside2npix(nside))
        cl = compute_angular_power_spectrum(m, lmax=20)
        assert cl.shape == (21,)

    def test_pure_monopole_vanishes_after_removal(self):
        nside = 8
        m = np.full(hp.nside2npix(nside), 5.0)
        cl = compute_angular_power_spectrum(m, remove_monopole=True)
        # A uniform map with monopole removed is identically zero
        assert np.all(cl < 1e-20)

    def test_pure_monopole_without_removal(self):
        """With monopole kept, a uniform map has C_0 > 0 dominating higher ell."""
        nside = 8
        m = np.full(hp.nside2npix(nside), 5.0)
        cl = compute_angular_power_spectrum(m, remove_monopole=False)
        assert cl[0] > 0.0
        # ell > 0 is dominated by anafast numerical noise but still << C_0
        assert cl[1:].max() < 1e-6 * cl[0]


class TestCrossCell:
    def test_self_cross_equals_auto(self):
        nside = 8
        rng = np.random.default_rng(1)
        m = rng.normal(0, 1, hp.nside2npix(nside))
        auto = compute_angular_power_spectrum(m, remove_monopole=False)
        cross = compute_cross_cell(m, m, remove_monopole=False)
        np.testing.assert_allclose(auto, cross, rtol=1e-10)

    def test_different_nside_raises(self):
        m1 = np.zeros(hp.nside2npix(8))
        m2 = np.zeros(hp.nside2npix(16))
        with pytest.raises(ValueError, match="same size"):
            compute_cross_cell(m1, m2)


class TestFilterEllBand:
    def test_filtered_map_has_isolated_power(self):
        """After filtering to [lo, hi], power outside the band is negligible."""
        nside = 16
        m = np.random.default_rng(2).normal(0, 1, hp.nside2npix(nside))
        lmax = 3 * nside - 1
        filt = filter_ell_band(m, 5, 15, lmax=lmax)
        cl_filt = hp.anafast(filt, lmax=lmax)
        peak_in = cl_filt[5:16].max()
        peak_out = max(cl_filt[:5].max(), cl_filt[16:].max())
        # Out-of-band power should be ~machine-precision small
        assert peak_out < 1e-6 * peak_in

    def test_invalid_band_raises(self):
        m = np.zeros(hp.nside2npix(8))
        with pytest.raises(ValueError, match="ell band"):
            filter_ell_band(m, 10, 5)


# ---------------------------------------------------------------------------
# Frequency correlation
# ---------------------------------------------------------------------------


class TestFrequencyCorrelation:
    def test_identity_on_independent_channels(self):
        """Independent random channels have near-zero off-diagonal correlation."""
        rng = np.random.default_rng(3)
        cube = rng.normal(0, 1, (5, 10000))
        corr = compute_frequency_correlation(cube)
        np.testing.assert_allclose(np.diag(corr), 1.0, rtol=1e-10)
        off_diag = corr[~np.eye(5, dtype=bool)]
        assert np.abs(off_diag).max() < 0.1

    def test_identical_channels_give_unit_correlation(self):
        rng = np.random.default_rng(4)
        row = rng.normal(0, 1, 1000)
        cube = np.tile(row, (4, 1))
        corr = compute_frequency_correlation(cube, center=True)
        # All off-diagonals should also be ~1 (same centred signal)
        assert np.allclose(corr, 1.0, rtol=1e-10)

    def test_shape_error(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_frequency_correlation(np.zeros(10))


# ---------------------------------------------------------------------------
# Delay spectrum
# ---------------------------------------------------------------------------


class TestDelaySpectrum:
    def test_shape(self):
        n_freq = 32
        cube = np.random.default_rng(5).normal(0, 1, (n_freq, 100))
        freqs = np.linspace(80e6, 90e6, n_freq)
        tau, ps = compute_delay_spectrum(cube, freqs)
        assert tau.shape == (n_freq,)
        assert ps.shape == (n_freq,)

    def test_cosine_produces_peak(self):
        """A pure sinusoid in frequency should produce a delta at delta = 1/freq."""
        n_freq = 128
        dfreq = 0.1e6
        freqs = np.linspace(80e6, 80e6 + (n_freq - 1) * dfreq, n_freq)
        # Sinusoid at modulation period = 2 MHz (20 channels)
        period_hz = 2e6
        tau_expected = 1.0 / period_hz  # seconds
        signal = np.cos(2 * np.pi * (freqs - freqs[0]) / period_hz)
        cube = np.tile(signal[:, None], (1, 10))
        tau, ps = compute_delay_spectrum(cube, freqs, window="none", remove_mean=True)
        peak_idx = int(np.argmax(ps))
        peak_tau = abs(tau[peak_idx])
        # Allow some tolerance — FFT bin width
        assert abs(peak_tau - tau_expected) < 1.0 / (n_freq * dfreq)

    def test_invalid_window_raises(self):
        cube = np.zeros((32, 10))
        freqs = np.linspace(80e6, 90e6, 32)
        with pytest.raises(ValueError, match="window"):
            compute_delay_spectrum(cube, freqs, window="bogus")

    def test_shape_error(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_delay_spectrum(np.zeros(10), np.linspace(80e6, 90e6, 10))


# ---------------------------------------------------------------------------
# k_parallel
# ---------------------------------------------------------------------------


class TestKParallel:
    def test_zero_tau_is_zero_kpar(self):
        k = compute_kparallel(np.array([0.0, 1e-8]), z=15.0)
        assert k[0] == 0.0
        assert k[1] > 0.0

    def test_monotonic_with_tau(self):
        tau = np.linspace(0, 1e-6, 20)
        k = compute_kparallel(tau, z=15.0)
        assert np.all(np.diff(k) >= 0)

    def test_changes_with_redshift(self):
        """k_par at fixed tau scales with H(z)/(1+z)^2 (Parsons 2012 formula)."""
        from astropy.cosmology import Planck15

        tau = np.array([1e-7])
        for z in (8.0, 10.0, 15.0):
            k = compute_kparallel(tau, z=z)
            # Manually apply the Parsons formula
            h_kms_mpc = Planck15.H(z).to("km/s/Mpc").value
            c_kms = 299792.458
            expected = (
                2.0 * np.pi * tau[0] * h_kms_mpc * F_21_HZ / (c_kms * (1 + z) ** 2)
            )
            np.testing.assert_allclose(k[0], expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Gaussianity stats
# ---------------------------------------------------------------------------


class TestGaussianityStats:
    def test_normal_sample(self):
        x = np.random.default_rng(6).normal(0, 1, 5000)
        stats = gaussianity_stats(x)
        assert abs(stats["skewness"]) < 0.15
        assert abs(stats["excess_kurtosis"]) < 0.2
        # D'Agostino p-value should be > 0.01 for a genuine Gaussian
        assert stats["normaltest_p"] > 0.001

    def test_non_gaussian_detected(self):
        """Exponentially-distributed data should have high skewness."""
        x = np.random.default_rng(7).exponential(scale=1.0, size=5000)
        stats = gaussianity_stats(x)
        assert stats["skewness"] > 1.0
        # D'Agostino should reject null easily
        assert stats["normaltest_p"] < 1e-6

    def test_shapiro_included_when_requested(self):
        x = np.random.default_rng(8).normal(0, 1, 1000)
        stats = gaussianity_stats(x, run_shapiro=True)
        assert "shapiro_p" in stats
        assert "shapiro_stat" in stats

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 20"):
            gaussianity_stats(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------


def test_f21_constant():
    # 21cm rest frequency (Hz)
    assert abs(F_21_HZ - 1420.4057e6) < 1e3
