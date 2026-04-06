"""Tests for rrivis.core.sky.constants — brightness temperature conversion functions."""

import numpy as np
import pytest

from rrivis.core.sky.constants import (
    C_LIGHT,
    H_PLANCK,
    K_BOLTZMANN,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
)

# =============================================================================
# TestBrightnessToFlux — brightness_temp_to_flux_density
# =============================================================================


class TestBrightnessToFlux:
    """Tests for brightness_temp_to_flux_density."""

    def test_rayleigh_jeans_known_value(self):
        """Analytically compute S = 2kTv^2/c^2 * Omega / 1e-26 and verify."""
        T = 100.0  # K
        nu = 408e6  # 408 MHz
        omega = 1.0  # sr

        # Hand calculation:
        # S = 2 * 1.380649e-23 * 100 * (408e6)^2 / (299792458)^2 * 1.0 / 1e-26
        numerator = 2 * K_BOLTZMANN * T * nu**2
        denominator = C_LIGHT**2
        expected_jy = numerator / denominator * omega / 1e-26

        result = brightness_temp_to_flux_density(
            np.array([T]), nu, omega, method="rayleigh-jeans"
        )

        # Verify exact numerical value (pure arithmetic, no approximation)
        assert np.isclose(result[0], expected_jy, rtol=1e-14)
        # Sanity: the value should be positive and finite
        assert result[0] > 0
        assert np.isfinite(result[0])
        # Cross-check approximate magnitude: ~5.11e5 Jy
        assert 5e5 < expected_jy < 6e5

    def test_planck_known_value(self):
        """At low freq where hv << kT, Planck approx equals RJ within 0.001%."""
        T = 5000.0  # K — high temperature
        nu = 50e6  # 50 MHz — low frequency
        omega = 1.0  # sr

        # Check hv/kT << 1 condition
        x = H_PLANCK * nu / (K_BOLTZMANN * T)
        assert x < 0.001, f"hv/kT = {x} is not << 1"

        result_planck = brightness_temp_to_flux_density(
            np.array([T]), nu, omega, method="planck"
        )
        result_rj = brightness_temp_to_flux_density(
            np.array([T]), nu, omega, method="rayleigh-jeans"
        )

        # In the RJ limit, Planck and RJ should agree within 0.001%
        assert np.isclose(result_planck[0], result_rj[0], rtol=1e-5)

    def test_planck_vs_rj_divergence_at_high_freq(self):
        """At 100 GHz, T=3K (CMB-like), Planck < RJ. RJ overestimates by >10%."""
        T = 3.0  # K — CMB temperature
        nu = 100e9  # 100 GHz
        omega = 1.0  # sr

        # hv/kT should NOT be << 1 here
        x = H_PLANCK * nu / (K_BOLTZMANN * T)
        assert x > 1.0, f"hv/kT = {x} should be > 1 for this test"

        result_planck = brightness_temp_to_flux_density(
            np.array([T]), nu, omega, method="planck"
        )
        result_rj = brightness_temp_to_flux_density(
            np.array([T]), nu, omega, method="rayleigh-jeans"
        )

        # Planck must be strictly less than RJ at high x
        assert result_planck[0] < result_rj[0]

        # RJ overestimates by more than 10%
        overestimate_fraction = (result_rj[0] - result_planck[0]) / result_planck[0]
        assert overestimate_fraction > 0.10, (
            f"RJ overestimates Planck by only {overestimate_fraction * 100:.2f}%, "
            f"expected > 10%"
        )

    def test_zero_temperature_raises(self):
        """T=0 with method='planck' raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be strictly positive"):
            brightness_temp_to_flux_density(
                np.array([0.0]), 100e6, 1.0, method="planck"
            )

    def test_negative_temperature_raises(self):
        """T<0 raises ValueError for Planck method."""
        with pytest.raises(ValueError, match="temperature must be strictly positive"):
            brightness_temp_to_flux_density(
                np.array([-10.0]), 100e6, 1.0, method="planck"
            )

    def test_array_input(self):
        """Array of temperatures matches element-wise individual calls."""
        temps = np.array([10.0, 100.0, 1000.0, 10000.0])
        nu = 150e6  # 150 MHz
        omega = 1e-4  # sr

        for method in ("planck", "rayleigh-jeans"):
            result_array = brightness_temp_to_flux_density(
                temps, nu, omega, method=method
            )
            assert result_array.shape == (4,)

            for i, T in enumerate(temps):
                result_single = brightness_temp_to_flux_density(
                    np.array([T]), nu, omega, method=method
                )
                assert np.isclose(result_array[i], result_single[0], rtol=1e-14), (
                    f"method={method}, T={T}: array result {result_array[i]} "
                    f"!= single result {result_single[0]}"
                )

    def test_solid_angle_scaling(self):
        """Doubling solid angle doubles flux density (linear proportionality)."""
        T = np.array([500.0])
        nu = 200e6
        omega1 = 1e-3
        omega2 = 2e-3

        s1 = brightness_temp_to_flux_density(T, nu, omega1, method="rayleigh-jeans")
        s2 = brightness_temp_to_flux_density(T, nu, omega2, method="rayleigh-jeans")

        assert np.isclose(s2[0], 2 * s1[0], rtol=1e-14)

    def test_frequency_scaling_rj(self):
        """RJ: S proportional to v^2. Doubling frequency yields 4x flux."""
        T = np.array([300.0])
        omega = 1.0
        nu1 = 100e6
        nu2 = 200e6

        s1 = brightness_temp_to_flux_density(T, nu1, omega, method="rayleigh-jeans")
        s2 = brightness_temp_to_flux_density(T, nu2, omega, method="rayleigh-jeans")

        assert np.isclose(s2[0], 4 * s1[0], rtol=1e-14)


# =============================================================================
# TestFluxToTemp — flux_density_to_brightness_temp
# =============================================================================


class TestFluxToTemp:
    """Tests for flux_density_to_brightness_temp."""

    def test_round_trip_planck(self):
        """T -> flux -> T round-trip preserves values (Planck)."""
        temps_orig = np.array([10.0, 100.0, 1000.0, 10000.0])
        nu = 150e6  # 150 MHz
        omega = 1e-4  # sr

        flux = brightness_temp_to_flux_density(temps_orig, nu, omega, method="planck")
        temps_recovered = flux_density_to_brightness_temp(
            flux, nu, omega, method="planck"
        )

        assert np.allclose(temps_recovered, temps_orig, rtol=1e-10), (
            f"Round-trip failed: original={temps_orig}, "
            f"recovered={temps_recovered}, "
            f"relative error={np.abs(temps_recovered - temps_orig) / temps_orig}"
        )

    def test_round_trip_rayleigh_jeans(self):
        """T -> flux -> T round-trip preserves values (Rayleigh-Jeans)."""
        temps_orig = np.array([10.0, 100.0, 1000.0, 10000.0])
        nu = 150e6
        omega = 1e-4

        flux = brightness_temp_to_flux_density(
            temps_orig, nu, omega, method="rayleigh-jeans"
        )
        temps_recovered = flux_density_to_brightness_temp(
            flux, nu, omega, method="rayleigh-jeans"
        )

        assert np.allclose(temps_recovered, temps_orig, rtol=1e-14), (
            f"Round-trip failed: original={temps_orig}, recovered={temps_recovered}"
        )

    def test_planck_rj_consistency_low_freq(self):
        """At 50 MHz, 1000 K: both methods give nearly identical T (within 0.1%)."""
        nu = 50e6
        omega = 1e-3
        T_orig = 1000.0

        # Generate flux with Planck, recover T with both methods
        flux = brightness_temp_to_flux_density(
            np.array([T_orig]), nu, omega, method="planck"
        )

        T_planck = flux_density_to_brightness_temp(flux, nu, omega, method="planck")
        T_rj = flux_density_to_brightness_temp(flux, nu, omega, method="rayleigh-jeans")

        # Both should be close to original
        assert np.isclose(T_planck[0], T_orig, rtol=1e-10)
        # RJ should agree within 0.1% at this low frequency / high temperature
        assert np.isclose(T_rj[0], T_orig, rtol=1e-3), (
            f"RJ temperature {T_rj[0]:.6f} differs from original {T_orig} "
            f"by {abs(T_rj[0] - T_orig) / T_orig * 100:.4f}%"
        )

    def test_zero_flux_raises(self):
        """S=0 raises ValueError for Planck method."""
        with pytest.raises(ValueError, match="flux density must be strictly positive"):
            flux_density_to_brightness_temp(
                np.array([0.0]), 100e6, 1.0, method="planck"
            )

    def test_negative_flux_raises(self):
        """S<0 raises ValueError for Planck method."""
        with pytest.raises(ValueError, match="flux density must be strictly positive"):
            flux_density_to_brightness_temp(
                np.array([-5.0]), 100e6, 1.0, method="planck"
            )


# =============================================================================
# TestPhysicalConsistency — cross-function consistency checks
# =============================================================================


class TestPhysicalConsistency:
    """Cross-function tests verifying physical consistency."""

    @pytest.mark.parametrize("method", ["planck", "rayleigh-jeans"])
    def test_inverse_relationship(self, method):
        """brightness_temp_to_flux_density and flux_density_to_brightness_temp
        are exact inverses for both Planck and RJ methods."""
        temps = np.array([50.0, 200.0, 800.0, 5000.0])
        nu = 300e6  # 300 MHz
        omega = 5e-5  # sr

        # Forward: T -> S
        flux = brightness_temp_to_flux_density(temps, nu, omega, method=method)
        # Backward: S -> T
        temps_recovered = flux_density_to_brightness_temp(
            flux, nu, omega, method=method
        )

        assert np.allclose(temps_recovered, temps, rtol=1e-10), (
            f"method={method}: round-trip failed. "
            f"Max relative error: "
            f"{np.max(np.abs(temps_recovered - temps) / temps):.2e}"
        )

        # Also verify the reverse direction: S -> T -> S
        flux_recovered = brightness_temp_to_flux_density(
            temps_recovered, nu, omega, method=method
        )
        assert np.allclose(flux_recovered, flux, rtol=1e-10), (
            f"method={method}: reverse round-trip (S->T->S) failed. "
            f"Max relative error: "
            f"{np.max(np.abs(flux_recovered - flux) / flux):.2e}"
        )

    def test_planck_reduces_to_rj_at_low_freq(self):
        """At sufficiently low frequency and high temperature,
        Planck and RJ methods converge."""
        # Use very low frequency and high temperature to ensure hv << kT
        T = np.array([1e4, 5e4, 1e5])  # 10,000 to 100,000 K
        nu = 10e6  # 10 MHz
        omega = 1e-2  # sr

        # Verify hv/kT << 1 for all temperatures
        x_values = H_PLANCK * nu / (K_BOLTZMANN * T)
        assert np.all(x_values < 1e-4), f"hv/kT values {x_values} are not all << 1"

        # Forward comparison: T -> S
        flux_planck = brightness_temp_to_flux_density(T, nu, omega, method="planck")
        flux_rj = brightness_temp_to_flux_density(T, nu, omega, method="rayleigh-jeans")

        assert np.allclose(flux_planck, flux_rj, rtol=1e-4), (
            f"Planck and RJ flux disagree: "
            f"Planck={flux_planck}, RJ={flux_rj}, "
            f"relative diff={np.abs(flux_planck - flux_rj) / flux_rj}"
        )

        # Reverse comparison: S -> T
        T_planck = flux_density_to_brightness_temp(
            flux_planck, nu, omega, method="planck"
        )
        T_rj = flux_density_to_brightness_temp(
            flux_planck, nu, omega, method="rayleigh-jeans"
        )

        assert np.allclose(T_planck, T_rj, rtol=1e-4), (
            f"Planck and RJ temperatures disagree: Planck={T_planck}, RJ={T_rj}"
        )
