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

    def test_compute_fringe_1d_baseline(self, numpy_backend, wavelengths):
        """Test fringe calculation with 1D baseline."""
        source_lmn = np.array([[0.0, 0.0, 1.0]])
        k_jones = GeometricPhaseJones(source_lmn, wavelengths)

        baseline_uvw = np.array([100.0, 0.0, 0.0])  # 100 wavelengths in u
        fringe = k_jones.compute_fringe(0, 0, baseline_uvw, numpy_backend)

        # For source at zenith, fringe should be 1.0
        np.testing.assert_almost_equal(fringe, 1.0 + 0j, decimal=10)

    def test_compute_fringe_2d_baseline(self, numpy_backend, wavelengths):
        """Test fringe calculation with 2D baseline array."""
        source_lmn = np.array([[0.0, 0.0, 1.0]])
        k_jones = GeometricPhaseJones(source_lmn, wavelengths)

        baseline_uvw = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
        fringe = k_jones.compute_fringe(0, 0, baseline_uvw, numpy_backend)

        # Should return array for multiple baselines
        assert fringe.shape == (2,)

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

    def test_set_gains_via_array(self, numpy_backend):
        """Test setting custom gains via gains array."""
        gains = np.array([2.0 + 1j, 1.5 - 0.5j], dtype=np.complex128)
        g_jones = GainJones(gains=gains)

        jones0 = g_jones.compute_jones(0, 0, 0, 0, numpy_backend)
        jones1 = g_jones.compute_jones(1, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones0[0, 0], 2.0 + 1j)
        np.testing.assert_almost_equal(jones1[0, 0], 1.5 - 0.5j)

    def test_gain_with_perturbation(self, numpy_backend):
        """Test gains with random perturbations."""
        g_jones = GainJones(n_antennas=10, gain_sigma=0.1, seed=42)

        # Gains should vary from unity
        gains = []
        for ant in range(10):
            jones = g_jones.compute_jones(ant, 0, 0, 0, numpy_backend)
            gains.append(jones[0, 0])

        gains = np.array(gains)
        # Should have some variance
        assert np.std(np.abs(gains)) > 0.01

    def test_time_variable_gains(self, numpy_backend):
        """Test time-varying gains."""
        g_jones = TimeVariableGainJones(
            n_antennas=4,
            n_times=100,
            amp_sigma=0.02,
            phase_sigma=0.05,
            seed=42
        )

        # Get gains at different times
        jones_t0 = g_jones.compute_jones(0, 0, 0, 0, numpy_backend)
        jones_t10 = g_jones.compute_jones(0, 0, 0, 10, numpy_backend)
        jones_t50 = g_jones.compute_jones(0, 0, 0, 50, numpy_backend)

        # Gains should be different at different times
        assert not np.allclose(jones_t0[0, 0], jones_t10[0, 0])
        assert not np.allclose(jones_t10[0, 0], jones_t50[0, 0])


class TestBandpassJones:
    """Tests for B term (bandpass)."""

    def test_default_unity_bandpass(self, numpy_backend, frequencies):
        """Default bandpass should be unity."""
        b_jones = BandpassJones(n_antennas=4, frequencies=frequencies)

        jones = b_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)

    def test_custom_bandpass(self, numpy_backend, frequencies):
        """Test setting custom bandpass."""
        n_antennas = 4
        n_freq = len(frequencies)

        # Create custom bandpass
        bandpass = np.zeros((n_antennas, n_freq, 2, 2), dtype=np.complex128)
        bandpass[:, :, 0, 0] = 0.9  # Slightly less than unity
        bandpass[:, :, 1, 1] = 0.8

        b_jones = BandpassJones(n_antennas=n_antennas, frequencies=frequencies, bandpass_gains=bandpass)

        jones = b_jones.compute_jones(0, 0, 5, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 0.9)
        np.testing.assert_almost_equal(jones[1, 1], 0.8)

    def test_rfi_flagging(self, numpy_backend, frequencies):
        """Test RFI flagging functionality."""
        b_jones = RFIFlaggedBandpassJones(n_antennas=4, frequencies=frequencies)

        # Flag a channel
        b_jones.flag_channel(5)

        jones = b_jones.compute_jones(0, 0, 5, 0, numpy_backend)

        # Flagged channel should be zero
        np.testing.assert_almost_equal(jones[0, 0], 0.0)
        np.testing.assert_almost_equal(jones[1, 1], 0.0)

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

    def test_custom_d_terms(self, numpy_backend):
        """Test custom D-terms."""
        d_terms = np.array([
            [0.05 + 0.02j, 0.03 - 0.01j],
            [0.04 + 0.01j, 0.02 - 0.02j],
        ], dtype=np.complex128)

        d_jones = PolarizationLeakageJones(n_antennas=2, d_terms=d_terms)

        jones = d_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Check structure [[1, d_p], [d_q, 1]]
        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)
        np.testing.assert_almost_equal(jones[0, 1], 0.05 + 0.02j)
        np.testing.assert_almost_equal(jones[1, 0], 0.03 - 0.01j)

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

    def test_zero_tec_identity(self, numpy_backend, frequencies):
        """Zero TEC should give identity."""
        z_jones = IonosphereJones(
            n_antennas=4,
            n_sources=1,
            frequencies=frequencies
        )

        jones = z_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)

    def test_faraday_rotation(self, numpy_backend, frequencies):
        """Test Faraday rotation from RM."""
        rm_values = np.array([[[1.0]]])  # 1 rad/m²

        z_jones = IonosphereJones(
            n_antennas=1,
            n_sources=1,
            frequencies=frequencies,
            rotation_measure=rm_values
        )

        jones = z_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Should be a rotation matrix
        det = jones[0, 0] * jones[1, 1] - jones[0, 1] * jones[1, 0]
        np.testing.assert_almost_equal(np.abs(det), 1.0, decimal=5)

    def test_dispersive_delay(self, numpy_backend, frequencies):
        """Test dispersive delay calculation."""
        tec_values = np.array([[[10.0]]])  # 10 TECU

        z_jones = IonosphereJones(
            n_antennas=1,
            n_sources=1,
            frequencies=frequencies,
            tec_values=tec_values
        )

        delay = z_jones.get_dispersive_delay(0, 0, 0, 0)

        # Delay should be positive and reasonable
        assert delay > 0
        assert delay < 1e-6  # Less than 1 microsecond for typical TEC

    def test_turbulent_ionosphere(self, numpy_backend, frequencies):
        """Test turbulent ionosphere model."""
        z_jones = TurbulentIonosphereJones(
            n_antennas=4,
            n_sources=1,
            frequencies=frequencies,
            mean_tec=10.0,
            seed=42
        )

        # TEC should have variability
        tec_0 = z_jones.tec_values[0, 0, 0]
        tec_1 = z_jones.tec_values[1, 0, 0]

        # Different antennas should have different TEC
        assert not np.isclose(tec_0, tec_1)


class TestTroposphereJones:
    """Tests for T term (troposphere)."""

    def test_default_troposphere(self, numpy_backend, frequencies):
        """Test default troposphere model."""
        source_elevations = np.array([np.deg2rad(45.0)])

        t_jones = TroposphereJones(
            n_antennas=4,
            n_sources=1,
            frequencies=frequencies,
            source_elevations=source_elevations
        )

        jones = t_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Should be scalar * identity
        np.testing.assert_almost_equal(jones[0, 0], jones[1, 1])
        np.testing.assert_almost_equal(jones[0, 1], 0.0)

    def test_elevation_dependence(self, numpy_backend, frequencies):
        """Path delay should increase at lower elevation."""
        t_jones = TroposphereJones(
            n_antennas=1,
            n_sources=2,
            frequencies=frequencies,
            source_elevations=np.array([np.deg2rad(90.0), np.deg2rad(30.0)])
        )

        delay_zenith = t_jones.get_path_delay(0, 0, 0)
        delay_low = t_jones.get_path_delay(0, 1, 0)

        # Lower elevation should have more delay
        assert delay_low > delay_zenith

    def test_saastamoinen_model(self, numpy_backend, frequencies):
        """Test Saastamoinen troposphere model."""
        times = np.array([0.0])  # Single time
        t_jones = SaastamoinenTroposphereJones(
            n_antennas=4,
            n_sources=1,
            frequencies=frequencies,
            antenna_heights=np.array([0.0, 100.0, 500.0, 1000.0]),
            source_elevations=np.array([np.deg2rad(45.0)]),
            times=times
        )

        # Higher altitude should have less zenith delay
        delay_0 = t_jones.zenith_delay[0, 0]
        delay_1000 = t_jones.zenith_delay[3, 0]

        assert delay_1000 < delay_0


class TestJonesChain:
    """Tests for JonesChain manager."""

    def test_empty_chain(self, numpy_backend):
        """Empty chain should give identity."""
        chain = JonesChain(numpy_backend)

        jones = chain.compute_antenna_jones(0, 0, 0, 0)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)

    def test_single_term_chain(self, numpy_backend):
        """Chain with one term should equal that term."""
        gains = np.array([2.0, 1.0], dtype=np.complex128)
        g_jones = GainJones(gains=gains)

        chain = JonesChain(numpy_backend)
        chain.add_term(g_jones)

        jones = chain.compute_antenna_jones(0, None, 0, 0)

        np.testing.assert_almost_equal(jones[0, 0], 2.0)

    def test_multiple_terms_chain(self, numpy_backend, frequencies, wavelengths):
        """Test chain with multiple terms."""
        # Create K, G, B terms
        source_lmn = np.array([[0.0, 0.0, 1.0]])
        k_jones = GeometricPhaseJones(source_lmn, wavelengths)

        gains = np.array([1.1, 1.1, 1.1, 1.1], dtype=np.complex128)
        g_jones = GainJones(gains=gains)

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

        # For source at zenith with unity K, gain 1.1, unity B
        # Result should be 1.1 * identity
        np.testing.assert_almost_equal(np.abs(jones[0, 0]), 1.1, decimal=5)

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
        # Create two gain terms with known values
        gains1 = np.array([2.0, 2.0], dtype=np.complex128)
        g1 = GainJones(gains=gains1)

        gains2 = np.array([1.5, 1.5], dtype=np.complex128)
        g2 = GainJones(gains=gains2)

        chain = JonesChain(numpy_backend)
        chain.add_term(g1)
        chain.add_term(g2)

        jones = chain.compute_antenna_jones(0, None, 0, 0)

        # Result should be g2 @ g1 (terms are applied right to left)
        # g1 = diag(2, 2), g2 = diag(1.5, 1.5)
        # g2 @ g1 = diag(3, 3)
        np.testing.assert_almost_equal(jones[0, 0], 3.0)
        np.testing.assert_almost_equal(jones[1, 1], 3.0)

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
