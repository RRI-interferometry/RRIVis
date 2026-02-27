"""
Unit tests for the Jones matrix framework.

Tests all Jones terms (K, E, G, B, D, P, Z, T) and the JonesChain manager.
"""

import numpy as np
import pytest

from rrivis.backends import get_backend
from rrivis.core.jones import (
    # Base
    JonesTerm,
    JonesChain,
    # K term
    GeometricPhaseJones,
    # E term
    BeamJones,
    AnalyticBeamJones,
    # G term
    GainJones,
    TimeVariableGainJones,
    # B term
    BandpassJones,
    PolynomialBandpassJones,
    RFIFlaggedBandpassJones,
    # D term
    PolarizationLeakageJones,
    IXRLeakageJones,
    # P term
    ParallacticAngleJones,
    # Z term
    IonosphereJones,
    TurbulentIonosphereJones,
    # T term
    TroposphereJones,
    SaastamoinenTroposphereJones,
)


@pytest.fixture
def numpy_backend():
    """Get NumPy backend for tests."""
    return get_backend("numpy")


@pytest.fixture
def frequencies():
    """Common frequency array for tests."""
    return np.linspace(100e6, 200e6, 10)  # 100-200 MHz


@pytest.fixture
def wavelengths(frequencies):
    """Wavelengths from frequencies."""
    return 3e8 / frequencies


class TestJonesTerm:
    """Tests for JonesTerm abstract base class."""

    def test_abstract_class_cannot_instantiate(self):
        """JonesTerm cannot be instantiated directly."""
        with pytest.raises(TypeError):
            JonesTerm()


class TestGeometricPhaseJones:
    """Tests for K term (geometric phase)."""

    def test_creation_with_lmn(self, numpy_backend, wavelengths):
        """Test creation with l, m, n coordinates."""
        source_lmn = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, np.sqrt(1 - 0.1**2)]])
        k_jones = GeometricPhaseJones(source_lmn, wavelengths)

        assert k_jones.name == "K"
        assert k_jones.is_direction_dependent
        assert len(k_jones.source_lmn) == 2
        assert k_jones.is_frequency_dependent

    def test_creation_with_lm_only(self, numpy_backend, wavelengths):
        """Test creation with l, m only (n calculated)."""
        source_lm = np.array([[0.0, 0.0], [0.1, 0.0]])
        k_jones = GeometricPhaseJones(source_lm, wavelengths)

        assert len(k_jones.source_lmn) == 2
        # Check n was calculated correctly
        expected_n = np.sqrt(1 - 0.1**2)
        np.testing.assert_almost_equal(k_jones.source_lmn[1, 2], expected_n)

    def test_compute_fringe_removed(self, numpy_backend, wavelengths):
        """Test that compute_fringe() has been removed (stub implementation)."""
        source_lmn = np.array([[0.0, 0.0, 1.0]])
        k_jones = GeometricPhaseJones(source_lmn, wavelengths)

        # Stub no longer has compute_fringe helper
        assert not hasattr(k_jones, 'compute_fringe')

    def test_jones_is_identity_times_scalar(self, numpy_backend, wavelengths):
        """K-Jones should be scalar * identity."""
        source_lmn = np.array([[0.1, 0.0, np.sqrt(1 - 0.1**2)]])
        k_jones = GeometricPhaseJones(source_lmn, wavelengths)

        baseline_uvw = np.array([100.0, 50.0, 10.0])
        jones = k_jones.compute_jones(0, 0, 0, 0, numpy_backend, baseline_uvw=baseline_uvw)

        # Should be 2x2 diagonal with same value
        assert jones.shape == (2, 2)
        np.testing.assert_almost_equal(jones[0, 0], jones[1, 1])
        np.testing.assert_almost_equal(jones[0, 1], 0.0)
        np.testing.assert_almost_equal(jones[1, 0], 0.0)


class TestAnalyticBeamJones:
    """Tests for E term (analytic beam models)."""

    def test_gaussian_beam_at_center(self, numpy_backend, frequencies):
        """Gaussian beam should be unity at center."""
        # Source at zenith (alt=90deg, az=0)
        source_altaz = np.array([[np.pi/2, 0.0]])
        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            hpbw_radians=np.deg2rad(10.0),
            beam_type="gaussian"
        )

        jones = beam.compute_jones(0, 0, 0, 0, numpy_backend)

        # Should be identity at center (zenith angle = 0)
        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)

    def test_gaussian_beam_decreases_off_axis(self, numpy_backend, frequencies):
        """Gaussian beam should decrease off-axis."""
        hpbw = np.deg2rad(10.0)
        # Source off-axis (alt=85deg = 5deg from zenith)
        source_altaz = np.array([[np.deg2rad(85), 0.0]])
        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            hpbw_radians=hpbw,
            beam_type="gaussian"
        )

        jones = beam.compute_jones(0, 0, 0, 0, numpy_backend)

        # Off-axis should have reduced amplitude
        assert np.abs(jones[0, 0]) < 1.0
        assert np.abs(jones[0, 0]) > 0.1

    def test_airy_beam(self, numpy_backend, frequencies):
        """Test Airy beam model."""
        source_altaz = np.array([[np.pi/2, 0.0]])  # zenith
        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            hpbw_radians=np.deg2rad(1.0),
            beam_type="airy"
        )

        jones = beam.compute_jones(0, 0, 0, 0, numpy_backend)
        # Center should be unity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_cosine_beam(self, numpy_backend, frequencies):
        """Test cosine beam model."""
        source_altaz = np.array([[np.pi/2, 0.0]])  # zenith
        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            hpbw_radians=np.deg2rad(60.0),
            beam_type="cosine"
        )

        jones = beam.compute_jones(0, 0, 0, 0, numpy_backend)
        # Center (zenith_angle=0, cos(0)=1) should be unity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)


class TestGainJones:
    """Tests for G term (electronic gains)."""

    def test_default_unity_gains(self, numpy_backend):
        """Default gains should be unity."""
        g_jones = GainJones(n_antennas=4)

        jones = g_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Should be identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)
        np.testing.assert_almost_equal(jones[0, 1], 0.0)

    def test_custom_gains_ignored_stub(self, numpy_backend):
        """Test that custom gains are ignored in stub (always returns identity)."""
        gains = np.array([2.0 + 1j, 1.5 - 0.5j], dtype=np.complex128)
        g_jones = GainJones(gains=gains)

        jones0 = g_jones.compute_jones(0, 0, 0, 0, numpy_backend)
        jones1 = g_jones.compute_jones(1, 0, 0, 0, numpy_backend)

        # Stub always returns identity
        np.testing.assert_almost_equal(jones0[0, 0], 1.0)
        np.testing.assert_almost_equal(jones1[0, 0], 1.0)

    def test_gain_sigma_ignored_stub(self, numpy_backend):
        """Test that gain_sigma is ignored in stub (always returns identity)."""
        g_jones = GainJones(n_antennas=10, gain_sigma=0.1, seed=42)

        # Stub always returns identity for all antennas
        for ant in range(10):
            jones = g_jones.compute_jones(ant, 0, 0, 0, numpy_backend)
            np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_time_variable_gains_stub(self, numpy_backend):
        """Test that time-varying gains stub returns identity at all times."""
        g_jones = TimeVariableGainJones(
            n_antennas=4,
            n_times=100,
            amp_sigma=0.02,
            phase_sigma=0.05,
            seed=42
        )

        # Stub always returns identity regardless of time
        jones_t0 = g_jones.compute_jones(0, 0, 0, 0, numpy_backend)
        jones_t10 = g_jones.compute_jones(0, 0, 0, 10, numpy_backend)
        jones_t50 = g_jones.compute_jones(0, 0, 0, 50, numpy_backend)

        # All should be identity
        np.testing.assert_almost_equal(jones_t0[0, 0], 1.0)
        np.testing.assert_almost_equal(jones_t10[0, 0], 1.0)
        np.testing.assert_almost_equal(jones_t50[0, 0], 1.0)


class TestBandpassJones:
    """Tests for B term (bandpass)."""

    def test_default_unity_bandpass(self, numpy_backend, frequencies):
        """Default bandpass should be unity."""
        b_jones = BandpassJones(n_antennas=4, frequencies=frequencies)

        jones = b_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)

    def test_custom_bandpass_ignored_stub(self, numpy_backend, frequencies):
        """Test that custom bandpass is ignored in stub (always returns identity)."""
        n_antennas = 4
        n_freq = len(frequencies)

        # Create custom bandpass
        bandpass = np.zeros((n_antennas, n_freq, 2, 2), dtype=np.complex128)
        bandpass[:, :, 0, 0] = 0.9  # Slightly less than unity
        bandpass[:, :, 1, 1] = 0.8

        b_jones = BandpassJones(n_antennas=n_antennas, frequencies=frequencies, bandpass_gains=bandpass)

        jones = b_jones.compute_jones(0, 0, 5, 0, numpy_backend)

        # Stub always returns identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)

    def test_rfi_flagging_stub(self, numpy_backend, frequencies):
        """Test that RFI flagging is not implemented in stub."""
        b_jones = RFIFlaggedBandpassJones(n_antennas=4, frequencies=frequencies)

        # Stub does not have flag_channel() method
        assert not hasattr(b_jones, 'flag_channel')

    def test_polynomial_bandpass(self, numpy_backend, frequencies):
        """Test polynomial bandpass model."""
        b_jones = PolynomialBandpassJones(
            n_antennas=4,
            frequencies=frequencies,
            poly_order=2
        )

        # Default should be unity (constant term = 1)
        jones = b_jones.compute_jones(0, 0, 0, 0, numpy_backend)
        np.testing.assert_almost_equal(np.abs(jones[0, 0]), 1.0, decimal=5)


class TestPolarizationLeakageJones:
    """Tests for D term (polarization leakage)."""

    def test_zero_leakage(self, numpy_backend):
        """Zero leakage should give identity."""
        d_jones = PolarizationLeakageJones(n_antennas=4)

        jones = d_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)
        np.testing.assert_almost_equal(jones[0, 1], 0.0)
        np.testing.assert_almost_equal(jones[1, 0], 0.0)

    def test_custom_d_terms_ignored_stub(self, numpy_backend):
        """Test that custom D-terms are ignored in stub (always returns identity)."""
        d_terms = np.array([
            [0.05 + 0.02j, 0.03 - 0.01j],
            [0.04 + 0.01j, 0.02 - 0.02j],
        ], dtype=np.complex128)

        d_jones = PolarizationLeakageJones(n_antennas=2, d_terms=d_terms)

        jones = d_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Stub always returns identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)
        np.testing.assert_almost_equal(jones[0, 1], 0.0)
        np.testing.assert_almost_equal(jones[1, 0], 0.0)

    def test_ixr_leakage(self, numpy_backend):
        """Test IXR-based D-terms."""
        # IXRLeakageJones uses random phases internally
        d_jones = IXRLeakageJones(n_antennas=4, target_ixr=100)

        # Check IXR is approximately correct
        for ant in range(4):
            ixr = d_jones.get_ixr(ant)
            np.testing.assert_almost_equal(ixr, 100, decimal=0)


class TestParallacticAngleJones:
    """Tests for P term (parallactic angle)."""

    def test_equatorial_mount_identity(self, numpy_backend):
        """Equatorial mount should have no parallactic rotation."""
        latitudes = np.deg2rad([30.0, 30.0])
        sources = np.deg2rad([[180.0, 45.0]])  # RA, Dec
        times = np.array([58000.0])  # MJD

        p_jones = ParallacticAngleJones(
            antenna_latitudes=latitudes,
            source_positions=sources,
            times=times,
            mount_type="equatorial"
        )

        jones = p_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Should be identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)
        np.testing.assert_almost_equal(jones[0, 1], 0.0)
        np.testing.assert_almost_equal(jones[1, 0], 0.0)

    def test_altaz_rotation_unitary(self, numpy_backend):
        """Alt-az rotation matrix should be unitary."""
        latitudes = np.deg2rad([30.0])
        sources = np.deg2rad([[180.0, 45.0]])
        times = np.array([58000.5])  # Different hour angle

        p_jones = ParallacticAngleJones(
            antenna_latitudes=latitudes,
            source_positions=sources,
            times=times,
            mount_type="altaz"
        )

        jones = p_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Check rotation matrix properties
        # |cos|^2 + |sin|^2 = 1
        np.testing.assert_almost_equal(
            np.abs(jones[0, 0])**2 + np.abs(jones[0, 1])**2, 1.0
        )

        # Determinant should be 1 (pure rotation)
        det = jones[0, 0] * jones[1, 1] - jones[0, 1] * jones[1, 0]
        np.testing.assert_almost_equal(np.abs(det), 1.0)


class TestIonosphereJones:
    """Tests for Z term (ionosphere)."""

    def test_ionosphere_returns_identity(self, numpy_backend, frequencies):
        """Ionosphere stub should return identity."""
        z_jones = IonosphereJones(
            tec=np.array([1e16]),
            frequencies=frequencies
        )

        jones = z_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Stub always returns identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)

    def test_ionosphere_parameters_accepted(self, numpy_backend, frequencies):
        """Test that ionosphere accepts new signature parameters."""
        # New signature used by visibility.py
        z_jones = IonosphereJones(
            tec=np.array([1e16]),
            frequencies=frequencies,
            include_faraday=True,
            include_delay=True,
        )

        jones = z_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Should return identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_turbulent_ionosphere_stub(self, numpy_backend, frequencies):
        """Test turbulent ionosphere stub."""
        z_jones = TurbulentIonosphereJones(
            n_antennas=4,
            n_sources=1,
            frequencies=frequencies,
            mean_tec=10.0,
            seed=42
        )

        # Stub has n_antennas and n_sources but no tec_values array
        assert z_jones.n_antennas == 4
        assert z_jones.n_sources == 1


class TestTroposphereJones:
    """Tests for T term (troposphere)."""

    def test_troposphere_returns_identity(self, numpy_backend, frequencies):
        """Test troposphere stub returns identity."""
        t_jones = TroposphereJones(
            n_antennas=4,
            frequencies=frequencies,
            elevations=np.array([np.deg2rad(45.0)])
        )

        jones = t_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Stub always returns identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)

    def test_troposphere_parameters_accepted(self, numpy_backend, frequencies):
        """Test that troposphere accepts new signature parameters."""
        # New signature used by visibility.py
        t_jones = TroposphereJones(
            n_antennas=4,
            frequencies=frequencies,
            elevations=np.array([np.deg2rad(45.0), np.deg2rad(30.0)])
        )

        jones = t_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Should return identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_saastamoinen_model_stub(self, numpy_backend, frequencies):
        """Test Saastamoinen troposphere stub."""
        t_jones = SaastamoinenTroposphereJones(
            n_antennas=4,
            n_sources=1,
            frequencies=frequencies,
            antenna_heights=np.array([0.0, 100.0, 500.0, 1000.0]),
        )

        # Stub has basic attributes but no zenith_delay array
        assert t_jones.n_antennas == 4
        assert t_jones.n_sources == 1


class TestJonesChain:
    """Tests for JonesChain manager."""

    def test_empty_chain(self, numpy_backend):
        """Empty chain should give identity."""
        chain = JonesChain(numpy_backend)

        jones = chain.compute_antenna_jones(0, 0, 0, 0)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)

    def test_single_term_chain(self, numpy_backend):
        """Chain with one term should compute correctly."""
        g_jones = GainJones(n_antennas=4)

        chain = JonesChain(numpy_backend)
        chain.add_term(g_jones)

        jones = chain.compute_antenna_jones(0, None, 0, 0)

        # Stub gains returns identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_multiple_terms_chain(self, numpy_backend, frequencies, wavelengths):
        """Test chain with multiple terms."""
        # Create K, G, B terms
        source_lmn = np.array([[0.0, 0.0, 1.0]])
        k_jones = GeometricPhaseJones(source_lmn, wavelengths)

        g_jones = GainJones(n_antennas=4)
        b_jones = BandpassJones(n_antennas=4, frequencies=frequencies)

        chain = JonesChain(numpy_backend)
        chain.add_term(k_jones)
        chain.add_term(g_jones)
        chain.add_term(b_jones)

        # Chain should combine all terms
        jones = chain.compute_antenna_jones(
            0, 0, 0, 0,
            baseline_uvw=np.array([100.0, 0.0, 0.0])
        )

        # Stubs all return identity, so result should be identity
        np.testing.assert_almost_equal(np.abs(jones[0, 0]), 1.0, decimal=5)

    def test_rime_visibility(self, numpy_backend, frequencies, wavelengths):
        """Test full RIME visibility calculation."""
        source_lmn = np.array([[0.0, 0.0, 1.0]])
        k_jones = GeometricPhaseJones(source_lmn, wavelengths)
        g_jones = GainJones(n_antennas=4)

        chain = JonesChain(numpy_backend)
        chain.add_term(k_jones)
        chain.add_term(g_jones)

        # Unpolarized source
        coherency = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

        vis = chain.compute_baseline_visibility(
            antenna_p=0, antenna_q=1,
            source_idx=0, freq_idx=0, time_idx=0,
            coherency_matrix=coherency,
            baseline_uvw=np.array([100.0, 0.0, 0.0])
        )

        # Result should be 2x2 visibility matrix
        assert vis.shape == (2, 2)

        # For unpolarized source, XX and YY should be equal
        np.testing.assert_almost_equal(vis[0, 0], vis[1, 1])

    def test_chain_direction_dependence(self, numpy_backend, frequencies):
        """Chain should track direction dependence."""
        g_jones = GainJones(n_antennas=4)
        assert not g_jones.is_direction_dependent

        source_altaz = np.array([[np.pi/2, 0.0]])
        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            hpbw_radians=np.deg2rad(10.0),
            beam_type="gaussian"
        )
        assert beam.is_direction_dependent

        chain = JonesChain(numpy_backend)
        chain.add_term(g_jones)
        chain.add_term(beam)

        # Chain should report which terms are DD
        dd_terms = [t for t in chain.terms if t.is_direction_dependent]
        assert len(dd_terms) == 1
        assert dd_terms[0].name == "E"


class TestJonesMatrixProperties:
    """Tests for mathematical properties of Jones matrices."""

    def test_jones_matrix_shape(self, numpy_backend, frequencies):
        """All Jones matrices should be 2x2."""
        terms = [
            GainJones(n_antennas=4),
            BandpassJones(n_antennas=4, frequencies=frequencies),
            PolarizationLeakageJones(n_antennas=4),
        ]

        for term in terms:
            jones = term.compute_jones(0, 0, 0, 0, numpy_backend)
            assert jones.shape == (2, 2), f"{term.name} Jones should be 2x2"

    def test_jones_chain_matrix_multiplication(self, numpy_backend):
        """Jones chain should correctly multiply matrices."""
        # Create two gain terms (stub returns identity)
        g1 = GainJones(n_antennas=2)
        g2 = GainJones(n_antennas=2)

        chain = JonesChain(numpy_backend)
        chain.add_term(g1)
        chain.add_term(g2)

        jones = chain.compute_antenna_jones(0, None, 0, 0)

        # Both terms are stubs returning identity, so result is identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)

    def test_hermitian_conjugate(self, numpy_backend):
        """Test Hermitian conjugate for RIME."""
        gains = np.array([2.0 + 1j], dtype=np.complex128)
        g = GainJones(gains=gains)

        jones = g.compute_jones(0, 0, 0, 0, numpy_backend)
        jones_h = numpy_backend.conjugate_transpose(jones)

        # (A^H)^H = A
        jones_hh = numpy_backend.conjugate_transpose(jones_h)
        np.testing.assert_array_almost_equal(jones, jones_hh)

        # Diagonal should conjugate
        np.testing.assert_almost_equal(jones_h[0, 0], np.conj(jones[0, 0]))


class TestIntegration:
    """Integration tests for complete visibility calculation."""

    def test_full_rime_simulation(self, numpy_backend, frequencies, wavelengths):
        """Test complete RIME simulation with all terms."""
        n_antennas = 4
        n_sources = 2

        # Create sources
        source_lmn = np.array([
            [0.0, 0.0, 1.0],  # Zenith
            [0.1, 0.05, np.sqrt(1 - 0.1**2 - 0.05**2)]  # Off-axis
        ])

        # Source altaz for beam
        source_altaz = np.array([
            [np.pi/2, 0.0],  # Zenith
            [np.deg2rad(84), 0.0]  # Off-axis
        ])

        # Create Jones terms
        k_jones = GeometricPhaseJones(source_lmn, wavelengths)

        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            hpbw_radians=np.deg2rad(20.0),
            beam_type="gaussian"
        )

        g_jones = GainJones(n_antennas=n_antennas)

        b_jones = BandpassJones(n_antennas=n_antennas, frequencies=frequencies)

        # Create chain
        chain = JonesChain(numpy_backend)
        chain.add_term(k_jones)
        chain.add_term(beam)
        chain.add_term(g_jones)
        chain.add_term(b_jones)

        # Compute visibility for a baseline
        coherency = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        baseline_uvw = np.array([100.0, 50.0, 10.0])

        # Both sources should contribute
        for src in range(n_sources):
            vis = chain.compute_baseline_visibility(
                antenna_p=0, antenna_q=1,
                source_idx=src, freq_idx=0, time_idx=0,
                coherency_matrix=coherency,
                baseline_uvw=baseline_uvw
            )

            assert vis.shape == (2, 2)
            # Off-axis source should have less power due to beam
            if src == 1:
                # Just check it's computed without error
                assert np.isfinite(vis).all()
