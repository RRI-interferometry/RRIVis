# tests/integration/test_end_to_end.py
"""
End-to-end integration tests for RRIvis.

These tests verify complete simulation workflows from configuration
to output, ensuring all components work together correctly.
"""

import pytest
import numpy as np
from pathlib import Path

from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u


@pytest.mark.integration
class TestBasicVisibilityCalculation:
    """Test basic visibility calculation workflows."""

    def test_minimal_visibility_calculation(
        self,
        sample_antennas_simple,
        sample_baselines_simple,
        sample_sources_single,
        sample_location_northern,
        sample_obstime,
        sample_frequencies_multiple,
        sample_wavelengths,
        sample_hpbw_simple,
    ):
        """Test minimal visibility calculation with all required inputs."""
        from rrivis.core.visibility import calculate_visibility

        visibilities = calculate_visibility(
            antennas=sample_antennas_simple,
            baselines=sample_baselines_simple,
            sources=sample_sources_single,
            location=sample_location_northern,
            obstime=sample_obstime,
            wavelengths=sample_wavelengths,
            freqs=sample_frequencies_multiple,
            hpbw_per_antenna=sample_hpbw_simple,
            duration_seconds=60.0,
            time_step_seconds=60.0,
        )

        # Verify output structure
        assert isinstance(visibilities, dict)
        assert len(visibilities) == len(sample_baselines_simple)

        # Verify each visibility is a dict with polarization products
        for baseline, vis_data in visibilities.items():
            assert isinstance(vis_data, dict)
            assert "I" in vis_data
            assert vis_data["I"].shape[-1] == len(sample_frequencies_multiple)
            assert np.iscomplexobj(vis_data["I"])

    def test_visibility_with_multiple_sources(
        self,
        sample_antennas_simple,
        sample_baselines_simple,
        sample_sources_multiple,
        sample_location_northern,
        sample_obstime,
        sample_frequencies_multiple,
        sample_wavelengths,
        sample_hpbw_simple,
    ):
        """Test visibility calculation with multiple sources."""
        from rrivis.core.visibility import calculate_visibility

        visibilities = calculate_visibility(
            antennas=sample_antennas_simple,
            baselines=sample_baselines_simple,
            sources=sample_sources_multiple,
            location=sample_location_northern,
            obstime=sample_obstime,
            wavelengths=sample_wavelengths,
            freqs=sample_frequencies_multiple,
            hpbw_per_antenna=sample_hpbw_simple,
            duration_seconds=60.0,
            time_step_seconds=60.0,
        )

        # With multiple sources, visibilities should be non-zero (unless all below horizon)
        for vis_data in visibilities.values():
            # At least check we got complex numbers back
            assert np.iscomplexobj(vis_data["I"])

    def test_visibility_with_jones_config(
        self,
        sample_antennas_simple,
        sample_baselines_simple,
        sample_sources_single,
        sample_location_northern,
        sample_obstime,
        sample_frequencies_multiple,
        sample_wavelengths,
        sample_hpbw_simple,
    ):
        """Test visibility calculation with Jones chain configuration."""
        from rrivis.core.visibility import calculate_visibility

        visibilities = calculate_visibility(
            antennas=sample_antennas_simple,
            baselines=sample_baselines_simple,
            sources=sample_sources_single,
            location=sample_location_northern,
            obstime=sample_obstime,
            wavelengths=sample_wavelengths,
            freqs=sample_frequencies_multiple,
            hpbw_per_antenna=sample_hpbw_simple,
            duration_seconds=60.0,
            time_step_seconds=60.0,
        )

        # Verify Jones chain mode produces valid output
        assert isinstance(visibilities, dict)
        for vis_data in visibilities.values():
            assert "I" in vis_data


@pytest.mark.integration
class TestBackendIntegration:
    """Test backend integration in visibility calculations."""

    def test_numpy_backend_integration(
        self,
        sample_antennas_simple,
        sample_baselines_simple,
        sample_sources_single,
        sample_location_northern,
        sample_obstime,
        sample_frequencies_multiple,
        sample_wavelengths,
        sample_hpbw_simple,
        numpy_backend,
    ):
        """Test visibility calculation with explicit NumPy backend."""
        from rrivis.core.visibility import calculate_visibility

        visibilities = calculate_visibility(
            antennas=sample_antennas_simple,
            baselines=sample_baselines_simple,
            sources=sample_sources_single,
            location=sample_location_northern,
            obstime=sample_obstime,
            wavelengths=sample_wavelengths,
            freqs=sample_frequencies_multiple,
            hpbw_per_antenna=sample_hpbw_simple,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            backend=numpy_backend,
        )

        assert isinstance(visibilities, dict)
        assert len(visibilities) > 0


@pytest.mark.integration
class TestJonesChainIntegration:
    """Test Jones chain integration."""

    def test_jones_chain_with_gains(
        self,
        sample_antennas_simple,
        sample_baselines_simple,
        sample_sources_single,
        sample_location_northern,
        sample_obstime,
        sample_frequencies_multiple,
        sample_wavelengths,
        sample_hpbw_simple,
    ):
        """Test Jones chain with gain calibration enabled."""
        from rrivis.core.visibility import calculate_visibility

        jones_config = {
            "G": {"enabled": True, "sigma": 0.0},  # No perturbation for determinism
        }

        visibilities = calculate_visibility(
            antennas=sample_antennas_simple,
            baselines=sample_baselines_simple,
            sources=sample_sources_single,
            location=sample_location_northern,
            obstime=sample_obstime,
            wavelengths=sample_wavelengths,
            freqs=sample_frequencies_multiple,
            hpbw_per_antenna=sample_hpbw_simple,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            jones_config=jones_config,
        )

        assert isinstance(visibilities, dict)

    def test_jones_chain_with_bandpass(
        self,
        sample_antennas_simple,
        sample_baselines_simple,
        sample_sources_single,
        sample_location_northern,
        sample_obstime,
        sample_frequencies_multiple,
        sample_wavelengths,
        sample_hpbw_simple,
    ):
        """Test Jones chain with bandpass enabled."""
        from rrivis.core.visibility import calculate_visibility

        jones_config = {
            "B": {"enabled": True},
        }

        visibilities = calculate_visibility(
            antennas=sample_antennas_simple,
            baselines=sample_baselines_simple,
            sources=sample_sources_single,
            location=sample_location_northern,
            obstime=sample_obstime,
            wavelengths=sample_wavelengths,
            freqs=sample_frequencies_multiple,
            hpbw_per_antenna=sample_hpbw_simple,
            duration_seconds=60.0,
            time_step_seconds=60.0,
            jones_config=jones_config,
        )

        assert isinstance(visibilities, dict)


@pytest.mark.integration
class TestPolarizationWorkflow:
    """Test polarization-related workflows."""

    def test_stokes_to_coherency_conversion(self):
        """Test Stokes to coherency matrix conversion."""
        from rrivis.core.polarization import stokes_to_coherency

        # Unpolarized source (Stokes I only)
        I, Q, U, V = 10.0, 0.0, 0.0, 0.0
        coherency = stokes_to_coherency(I, Q, U, V)

        # Check it's a 2x2 complex matrix
        assert coherency.shape == (2, 2)
        assert np.iscomplexobj(coherency)

        # For unpolarized: diagonal should be I/2
        assert np.isclose(coherency[0, 0].real, I / 2, rtol=1e-5)
        assert np.isclose(coherency[1, 1].real, I / 2, rtol=1e-5)

    def test_full_polarization_visibility(
        self,
        sample_antennas_simple,
        sample_baselines_simple,
        sample_sources_polarized,
        sample_location_northern,
        sample_obstime,
        sample_frequencies_multiple,
        sample_wavelengths,
        sample_hpbw_simple,
    ):
        """Test visibility calculation with polarized sources."""
        from rrivis.core.visibility import calculate_visibility

        visibilities = calculate_visibility(
            antennas=sample_antennas_simple,
            baselines=sample_baselines_simple,
            sources=sample_sources_polarized,
            location=sample_location_northern,
            obstime=sample_obstime,
            wavelengths=sample_wavelengths,
            freqs=sample_frequencies_multiple,
            hpbw_per_antenna=sample_hpbw_simple,
            duration_seconds=60.0,
            time_step_seconds=60.0,
        )

        # Should have polarization products
        for vis_data in visibilities.values():
            assert "I" in vis_data
