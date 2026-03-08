"""
Unit tests for the Jones matrix framework.

Tests all Jones terms (K, E, G, B, D, P, Z, T) and the JonesChain manager.
"""

import numpy as np
import pytest

from rrivis.backends import get_backend
from rrivis.core.jones import (
    AnalyticBeamJones,
    ArrayFactorJones,
    # B term
    BandpassJones,
    # M / Q terms (baseline-dependent)
    BaselineMultiplicativeJones,
    BasisTransformJones,
    # E term
    CableReflectionJones,
    CrosshandDelayJones,
    # X / Kx / DF terms
    CrosshandPhaseJones,
    # Kd / Rc / ff terms
    DelayJones,
    DifferentialBeamJones,
    DifferentialFaradayJones,
    # Ee / a / dE terms
    ElementBeamJones,
    ElevationGainJones,
    # F term
    FaradayRotationJones,
    FrequencyDependentLeakageJones,
    FringeFitJones,
    # G term
    GainJones,
    # K term
    GeometricPhaseJones,
    # Z term
    IonosphereJones,
    IXRLeakageJones,
    JonesBaselineTerm,
    JonesChain,
    # Base
    JonesTerm,
    # P term
    ParallacticAngleJones,
    # D term
    PolarizationLeakageJones,
    PolynomialBandpassJones,
    # C + H terms
    ReceptorConfigJones,
    RFIFlaggedBandpassJones,
    SaastamoinenTroposphereJones,
    SmearingFactorJones,
    TimeVariableGainJones,
    # T term
    TroposphereJones,
    TroposphericOpacityJones,
    TurbulentIonosphereJones,
    WidefieldPolarimetricJones,
    # W term
    WPhaseJones,
    WProjectionJones,
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
        assert not hasattr(k_jones, "compute_fringe")

    def test_jones_is_identity_times_scalar(self, numpy_backend, wavelengths):
        """K-Jones should be scalar * identity."""
        source_lmn = np.array([[0.1, 0.0, np.sqrt(1 - 0.1**2)]])
        k_jones = GeometricPhaseJones(source_lmn, wavelengths)

        baseline_uvw = np.array([100.0, 50.0, 10.0])
        jones = k_jones.compute_jones(
            0, 0, 0, 0, numpy_backend, baseline_uvw=baseline_uvw
        )

        # Should be 2x2 diagonal with same value
        assert jones.shape == (2, 2)
        np.testing.assert_almost_equal(jones[0, 0], jones[1, 1])
        np.testing.assert_almost_equal(jones[0, 1], 0.0)
        np.testing.assert_almost_equal(jones[1, 0], 0.0)


class TestAnalyticBeamJones:
    """Tests for E term (Gaussian beam)."""

    def test_gaussian_beam_at_center(self, numpy_backend, frequencies):
        """Gaussian beam should be unity at center."""
        # Source at zenith (alt=90deg, az=0)
        source_altaz = np.array([[np.pi / 2, 0.0]])
        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
        )

        jones = beam.compute_jones(0, 0, 0, 0, numpy_backend)

        # Should be identity at center (zenith angle = 0)
        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)

    def test_gaussian_beam_decreases_off_axis(self, numpy_backend, frequencies):
        """Gaussian beam should decrease off-axis."""
        # Source off-axis (alt=85deg = 5deg from zenith)
        source_altaz = np.array([[np.deg2rad(85), 0.0]])
        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
        )

        jones = beam.compute_jones(0, 0, 0, 0, numpy_backend)

        # Off-axis should have reduced amplitude
        assert np.abs(jones[0, 0]) < 1.0
        assert np.abs(jones[0, 0]) > 0.1


class TestPerAntennaBeam:
    """Tests for per-antenna diameter support."""

    def test_different_diameter_per_antenna(self, numpy_backend, frequencies):
        """Different diameter per antenna produces different Jones matrices."""
        # Source off-axis at 5 degrees from zenith
        source_altaz = np.array([[np.deg2rad(85), 0.0]])

        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
            diameter_per_antenna={0: 14.0, 1: 7.0},
        )

        jones_ant0 = beam.compute_jones_all_sources(
            0, 1, 0, 0, numpy_backend, antenna_number=0
        )
        jones_ant1 = beam.compute_jones_all_sources(
            1, 1, 0, 0, numpy_backend, antenna_number=1
        )

        # Larger dish (ant0, 14m) has narrower beam, should attenuate more at 5deg off-axis
        # than smaller dish (ant1, 7m) which has a wider beam
        amp0 = np.abs(jones_ant0[0, 0, 0])
        amp1 = np.abs(jones_ant1[0, 0, 0])
        assert amp0 < amp1, (
            f"Narrow beam ({amp0}) should be weaker than wide beam ({amp1})"
        )

    def test_single_diameter_without_per_antenna(self, numpy_backend, frequencies):
        """Single diameter without per-antenna map still works."""
        source_altaz = np.array([[np.pi / 2, 0.0]])
        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
        )
        jones = beam.compute_jones(0, 0, 0, 0, numpy_backend)
        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_missing_antenna_falls_back_to_default(self, numpy_backend, frequencies):
        """Antenna not in per-antenna map falls back to default diameter."""
        source_altaz = np.array([[np.deg2rad(85), 0.0]])

        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
            diameter_per_antenna={0: 7.0},
        )

        # Antenna 99 not in map, should use default 14m diameter
        jones_default = beam.compute_jones_all_sources(
            99, 1, 0, 0, numpy_backend, antenna_number=99
        )
        # Antenna 0 has 7m diameter
        jones_ant0 = beam.compute_jones_all_sources(
            0, 1, 0, 0, numpy_backend, antenna_number=0
        )

        assert not np.allclose(jones_default, jones_ant0)


class TestIsDiagonalBeam:
    """Tests for is_diagonal() on analytic beams."""

    def test_is_diagonal_true_for_gaussian(self, frequencies):
        """is_diagonal() returns True for Gaussian beam (no cross-pol)."""
        source_altaz = np.array([[np.pi / 2, 0.0]])
        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
        )
        assert beam.is_diagonal()


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
            n_antennas=4, n_times=100, amp_sigma=0.02, phase_sigma=0.05, seed=42
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

        b_jones = BandpassJones(
            n_antennas=n_antennas, frequencies=frequencies, bandpass_gains=bandpass
        )

        jones = b_jones.compute_jones(0, 0, 5, 0, numpy_backend)

        # Stub always returns identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)

    def test_rfi_flagging_stub(self, numpy_backend, frequencies):
        """Test that RFI flagging is not implemented in stub."""
        b_jones = RFIFlaggedBandpassJones(n_antennas=4, frequencies=frequencies)

        # Stub does not have flag_channel() method
        assert not hasattr(b_jones, "flag_channel")

    def test_polynomial_bandpass(self, numpy_backend, frequencies):
        """Test polynomial bandpass model."""
        b_jones = PolynomialBandpassJones(
            n_antennas=4, frequencies=frequencies, poly_order=2
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
        d_terms = np.array(
            [
                [0.05 + 0.02j, 0.03 - 0.01j],
                [0.04 + 0.01j, 0.02 - 0.02j],
            ],
            dtype=np.complex128,
        )

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
            mount_type="equatorial",
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
            mount_type="altaz",
        )

        jones = p_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Check rotation matrix properties
        # |cos|^2 + |sin|^2 = 1
        np.testing.assert_almost_equal(
            np.abs(jones[0, 0]) ** 2 + np.abs(jones[0, 1]) ** 2, 1.0
        )

        # Determinant should be 1 (pure rotation)
        det = jones[0, 0] * jones[1, 1] - jones[0, 1] * jones[1, 0]
        np.testing.assert_almost_equal(np.abs(det), 1.0)


class TestIonosphereJones:
    """Tests for Z term (ionosphere)."""

    def test_ionosphere_returns_identity(self, numpy_backend, frequencies):
        """Ionosphere stub should return identity."""
        z_jones = IonosphereJones(tec=np.array([1e16]), frequencies=frequencies)

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
            n_antennas=4, n_sources=1, frequencies=frequencies, mean_tec=10.0, seed=42
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
            elevations=np.array([np.deg2rad(45.0)]),
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
            elevations=np.array([np.deg2rad(45.0), np.deg2rad(30.0)]),
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


class TestTroposphericOpacityJones:
    """Tests for tropospheric opacity term."""

    def test_opacity_returns_identity(self, numpy_backend, frequencies):
        """Tropospheric opacity stub should return identity."""
        t_jones = TroposphericOpacityJones(
            n_antennas=4,
            frequencies=frequencies,
            zenith_opacity=np.array([0.05, 0.06, 0.04, 0.07]),
        )

        jones = t_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Stub always returns identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)

    def test_opacity_parameters_accepted(self, numpy_backend, frequencies):
        """Test that opacity accepts all parameters without error."""
        t_jones = TroposphericOpacityJones(
            n_antennas=8,
            frequencies=frequencies,
            zenith_opacity=np.linspace(0.05, 0.1, 8),
        )

        jones = t_jones.compute_jones(0, 5, 0, 0, numpy_backend)

        # Should return identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)


class TestFaradayRotationJones:
    """Tests for F term (Faraday rotation)."""

    def test_faraday_returns_identity(self, numpy_backend, frequencies):
        """Faraday rotation stub should return identity."""
        f_jones = FaradayRotationJones(rotation_measure=100.0, frequencies=frequencies)

        jones = f_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Stub always returns identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)
        np.testing.assert_almost_equal(jones[1, 1], 1.0)

    def test_faraday_parameters_accepted(self, numpy_backend, frequencies):
        """Test that Faraday accepts all parameters without error."""
        f_jones = FaradayRotationJones(rotation_measure=50.0, frequencies=frequencies)

        jones = f_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        assert jones.shape == (2, 2)

    def test_differential_faraday_stub(self, numpy_backend, frequencies):
        """Test differential Faraday stub."""
        f_jones = DifferentialFaradayJones(
            n_antennas=4, n_sources=2, frequencies=frequencies
        )

        assert f_jones.n_antennas == 4
        assert f_jones.n_sources == 2


class TestWTermJones:
    """Tests for W term (non-coplanar baseline corrections)."""

    def test_wphase_returns_identity(self, numpy_backend):
        """W-phase stub should return identity."""
        w_jones = WPhaseJones(
            source_lmn=np.array([[0.1, 0.0, np.sqrt(1 - 0.1**2)]]),
            wavelengths=np.array([1.5]),
        )

        jones = w_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        # Stub always returns identity
        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_wprojection_parameters_accepted(self, numpy_backend):
        """Test that W-projection accepts all parameters without error."""
        w_jones = WProjectionJones(
            n_antennas=4,
            source_lmn=np.array([[0.0, 0.0, 1.0]]),
            wavelengths=np.array([3.0]),
        )

        w_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        assert w_jones.n_antennas == 4

    def test_widefield_polarimetric_stub(self, numpy_backend):
        """Test wide-field polarimetric term."""
        w_jones = WidefieldPolarimetricJones(
            source_lmn=np.array([[0.05, 0.03, np.sqrt(1 - 0.05**2 - 0.03**2)]])
        )

        jones = w_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)


class TestReceptorConfigJones:
    """Tests for C term (receptor configuration)."""

    def test_linear_receptor_identity(self, numpy_backend):
        """Linear receptor should return identity."""
        c_jones = ReceptorConfigJones(feed_type="linear")

        jones = c_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_circular_receptor(self, numpy_backend):
        """Circular receptor stub should return identity."""
        c_jones = ReceptorConfigJones(feed_type="circular")

        jones = c_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_basis_transform_stub(self, numpy_backend):
        """Test basis transformation term."""
        h_jones = BasisTransformJones(from_basis="linear", to_basis="circular")

        jones = h_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)


class TestElementBeamJones:
    """Tests for element beam and array factor terms."""

    def test_element_beam_returns_identity(self, numpy_backend, frequencies):
        """Element beam stub should return identity."""
        ee_jones = ElementBeamJones(n_antennas=4, frequencies=frequencies)

        jones = ee_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_array_factor_scalar(self, numpy_backend, frequencies):
        """Array factor stub should return identity."""
        a_jones = ArrayFactorJones(n_antennas=4, n_elements=16, frequencies=frequencies)

        jones = a_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_differential_beam_stub(self, numpy_backend, frequencies):
        """Test differential beam stub."""
        de_jones = DifferentialBeamJones(
            n_antennas=4, n_sources=2, frequencies=frequencies
        )

        jones = de_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)


class TestDelayJones:
    """Tests for delay-related terms."""

    def test_electronic_delay_returns_identity(self, numpy_backend, frequencies):
        """Electronic delay stub should return identity."""
        kd_jones = DelayJones(
            n_antennas=4,
            delays=np.array([0.0, 1e-9, 2e-9, 3e-9]),
            frequencies=frequencies,
        )

        jones = kd_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_cable_reflection_stub(self, numpy_backend, frequencies):
        """Test cable reflection term."""
        rc_jones = CableReflectionJones(
            n_antennas=4,
            reflection_coeff=0.05,
            cable_delay=1e-8,
            frequencies=frequencies,
        )

        jones = rc_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_fringe_fitting_stub(self, numpy_backend, frequencies):
        """Test VLBI fringe-fitting term."""
        ff_jones = FringeFitJones(
            n_antennas=4,
            delays=np.array([0.0, 1e-9, 2e-9, 3e-9]),
            rates=np.array([0.0, 1e-6, 2e-6, 3e-6]),
            phases=np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]),
            frequencies=frequencies,
            times=np.array([58000.0]),
        )

        jones = ff_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)


class TestCrosshandJones:
    """Tests for cross-hand effects."""

    def test_crosshand_phase_returns_identity(self, numpy_backend):
        """Cross-hand phase stub should return identity."""
        x_jones = CrosshandPhaseJones(phase_offset=np.pi / 4)

        jones = x_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_crosshand_delay_stub(self, numpy_backend, frequencies):
        """Test cross-hand delay term."""
        kx_jones = CrosshandDelayJones(delay=1e-9, frequencies=frequencies)

        jones = kx_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_frequency_dependent_leakage_stub(self, numpy_backend, frequencies):
        """Test frequency-dependent leakage term."""
        df_jones = FrequencyDependentLeakageJones(
            n_antennas=4,
            frequencies=frequencies,
            d_terms=np.array([0.05, 0.06, 0.04, 0.07]),
        )

        jones = df_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)


class TestBaselineTerms:
    """Tests for baseline-dependent Jones terms."""

    def test_jones_baseline_abstract_class(self):
        """JonesBaselineTerm cannot be instantiated directly."""
        with pytest.raises(TypeError):
            JonesBaselineTerm()

    def test_baseline_multiplicative_returns_identity(self, numpy_backend):
        """Baseline multiplicative error stub should return identity."""
        m_jones = BaselineMultiplicativeJones()

        result = m_jones.compute_baseline_term(
            antenna_p=0,
            antenna_q=1,
            source_idx=0,
            freq_idx=0,
            time_idx=0,
            backend=numpy_backend,
        )

        # Stub always returns identity
        np.testing.assert_almost_equal(result[0, 0], 1.0)
        np.testing.assert_almost_equal(result[1, 1], 1.0)

    def test_smearing_factor_returns_identity(self, numpy_backend):
        """Smearing factor stub should return identity."""
        q_jones = SmearingFactorJones(time_smearing=True, bandwidth_smearing=True)

        result = q_jones.compute_baseline_term(
            antenna_p=0,
            antenna_q=1,
            source_idx=0,
            freq_idx=0,
            time_idx=0,
            backend=numpy_backend,
        )

        np.testing.assert_almost_equal(result[0, 0], 1.0)

    def test_baseline_terms_not_jones_terms(self):
        """Baseline terms should NOT be subclasses of JonesTerm."""
        assert not issubclass(BaselineMultiplicativeJones, JonesTerm)
        assert not issubclass(SmearingFactorJones, JonesTerm)

    def test_baseline_terms_are_baseline_terms(self):
        """Baseline terms should be subclasses of JonesBaselineTerm."""
        assert issubclass(BaselineMultiplicativeJones, JonesBaselineTerm)
        assert issubclass(SmearingFactorJones, JonesBaselineTerm)


class TestElevationGainJones:
    """Tests for elevation-dependent gain term."""

    def test_elevation_gain_returns_identity(self, numpy_backend):
        """Elevation gain stub should return identity."""
        eg_jones = ElevationGainJones(
            n_antennas=4, gain_curve_coeffs=np.array([1.0, 0.5, 0.1])
        )

        jones = eg_jones.compute_jones(0, 0, 0, 0, numpy_backend)

        np.testing.assert_almost_equal(jones[0, 0], 1.0)

    def test_elevation_gain_is_gain_subclass(self):
        """Elevation gain should be a GainJones subclass."""
        assert issubclass(ElevationGainJones, GainJones)

    def test_elevation_gain_inherits_properties(self, numpy_backend):
        """Elevation gain should inherit G term properties."""
        eg_jones = ElevationGainJones(n_antennas=4)

        assert eg_jones.name == "G"
        assert not eg_jones.is_direction_dependent
        assert eg_jones.is_diagonal()


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
            0, 0, 0, 0, baseline_uvw=np.array([100.0, 0.0, 0.0])
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
            antenna_p=0,
            antenna_q=1,
            source_idx=0,
            freq_idx=0,
            time_idx=0,
            coherency_matrix=coherency,
            baseline_uvw=np.array([100.0, 0.0, 0.0]),
        )

        # Result should be 2x2 visibility matrix
        assert vis.shape == (2, 2)

        # For unpolarized source, XX and YY should be equal
        np.testing.assert_almost_equal(vis[0, 0], vis[1, 1])

    def test_chain_direction_dependence(self, numpy_backend, frequencies):
        """Chain should track direction dependence."""
        g_jones = GainJones(n_antennas=4)
        assert not g_jones.is_direction_dependent

        source_altaz = np.array([[np.pi / 2, 0.0]])
        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
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


class TestComputeJonesAllSources:
    """Tests for vectorized compute_jones_all_sources."""

    def test_gain_all_sources_matches_loop(self, numpy_backend):
        """Default loop fallback should produce correct (n_sources, 2, 2) output."""
        g = GainJones(n_antennas=4)
        n_sources = 5

        result = g.compute_jones_all_sources(0, n_sources, 0, 0, numpy_backend)

        assert result.shape == (n_sources, 2, 2)
        # Stubs return identity for each source
        for s in range(n_sources):
            np.testing.assert_almost_equal(result[s, 0, 0], 1.0)
            np.testing.assert_almost_equal(result[s, 1, 1], 1.0)
            np.testing.assert_almost_equal(result[s, 0, 1], 0.0)

    def test_geometric_all_sources_vectorized(self, numpy_backend, wavelengths):
        """GeometricPhaseJones vectorized override should match per-source loop."""
        source_lmn = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, np.sqrt(1 - 0.1**2)],
                [0.05, 0.05, np.sqrt(1 - 0.05**2 - 0.05**2)],
            ]
        )
        k = GeometricPhaseJones(source_lmn, wavelengths)
        baseline_uvw = np.array([100.0, 50.0, 10.0])

        # Vectorized
        result_vec = k.compute_jones_all_sources(
            0, 3, 0, 0, numpy_backend, baseline_uvw=baseline_uvw
        )

        # Per-source loop
        result_loop = np.zeros((3, 2, 2), dtype=np.complex128)
        for s in range(3):
            result_loop[s] = k.compute_jones(
                0, s, 0, 0, numpy_backend, baseline_uvw=baseline_uvw
            )

        np.testing.assert_array_almost_equal(result_vec, result_loop, decimal=12)

    def test_geometric_all_sources_no_baseline(self, numpy_backend, wavelengths):
        """K-term without baseline_uvw returns batch identity."""
        source_lmn = np.array([[0.1, 0.0, np.sqrt(1 - 0.1**2)]])
        k = GeometricPhaseJones(source_lmn, wavelengths)

        result = k.compute_jones_all_sources(0, 1, 0, 0, numpy_backend)

        assert result.shape == (1, 2, 2)
        np.testing.assert_almost_equal(result[0, 0, 0], 1.0)
        np.testing.assert_almost_equal(result[0, 1, 1], 1.0)

    def test_analytic_beam_all_sources_vectorized(self, numpy_backend, frequencies):
        """AnalyticBeamJones vectorized override should match per-source."""
        source_altaz = np.array(
            [
                [np.pi / 2, 0.0],  # Zenith
                [np.deg2rad(85), 0.0],  # 5 deg off
                [np.deg2rad(70), 0.0],  # 20 deg off
            ]
        )
        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
        )

        # Vectorized
        result_vec = beam.compute_jones_all_sources(0, 3, 0, 0, numpy_backend)

        # Per-source loop
        result_loop = np.zeros((3, 2, 2), dtype=np.complex128)
        for s in range(3):
            result_loop[s] = beam.compute_jones(0, s, 0, 0, numpy_backend)

        assert result_vec.shape == (3, 2, 2)
        np.testing.assert_array_almost_equal(result_vec, result_loop, decimal=10)


class TestComputeAntennaJonesAllSources:
    """Tests for JonesChain.compute_antenna_jones_all_sources."""

    def test_empty_chain_returns_identity(self, numpy_backend):
        """Empty chain should return batch identity."""
        chain = JonesChain(numpy_backend)
        result = chain.compute_antenna_jones_all_sources(0, 5, 0, 0)

        assert result.shape == (5, 2, 2)
        for s in range(5):
            np.testing.assert_almost_equal(result[s, 0, 0], 1.0)
            np.testing.assert_almost_equal(result[s, 1, 1], 1.0)
            np.testing.assert_almost_equal(result[s, 0, 1], 0.0)

    def test_single_di_term(self, numpy_backend):
        """Chain with single DI term should broadcast correctly."""
        g = GainJones(n_antennas=4)
        chain = JonesChain(numpy_backend)
        chain.add_term(g)

        result = chain.compute_antenna_jones_all_sources(0, 3, 0, 0)

        assert result.shape == (3, 2, 2)
        # Stub returns identity
        for s in range(3):
            np.testing.assert_almost_equal(result[s, 0, 0], 1.0)

    def test_dd_and_di_terms(self, numpy_backend, frequencies):
        """Chain with DDE and DIE terms should combine correctly."""
        source_altaz = np.array(
            [
                [np.pi / 2, 0.0],
                [np.deg2rad(80), 0.0],
            ]
        )
        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
        )
        g = GainJones(n_antennas=4)

        chain = JonesChain(numpy_backend)
        chain.add_term(beam)
        chain.add_term(g)

        result = chain.compute_antenna_jones_all_sources(0, 2, 0, 0)

        assert result.shape == (2, 2, 2)
        # Zenith source should have ~unity beam
        np.testing.assert_almost_equal(np.abs(result[0, 0, 0]), 1.0, decimal=5)
        # Off-axis source should have reduced beam
        assert np.abs(result[1, 0, 0]) < 1.0

    def test_matches_per_source_loop(self, numpy_backend, frequencies, wavelengths):
        """Vectorized chain should match per-source compute_antenna_jones."""
        source_lmn = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, np.sqrt(1 - 0.1**2)],
            ]
        )
        source_altaz = np.array(
            [
                [np.pi / 2, 0.0],
                [np.deg2rad(84), 0.0],
            ]
        )

        k = GeometricPhaseJones(source_lmn, wavelengths)
        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
        )
        g = GainJones(n_antennas=4)

        chain = JonesChain(numpy_backend)
        chain.add_term(k)
        chain.add_term(beam)
        chain.add_term(g)

        baseline_uvw = np.array([100.0, 50.0, 10.0])

        # Vectorized
        result_vec = chain.compute_antenna_jones_all_sources(
            0, 2, 0, 0, baseline_uvw=baseline_uvw
        )

        # Per-source loop
        result_loop = np.zeros((2, 2, 2), dtype=np.complex128)
        for s in range(2):
            result_loop[s] = chain.compute_antenna_jones(
                0, s, 0, 0, baseline_uvw=baseline_uvw
            )

        np.testing.assert_array_almost_equal(result_vec, result_loop, decimal=10)


class TestIntegration:
    """Integration tests for complete visibility calculation."""

    def test_full_rime_simulation(self, numpy_backend, frequencies, wavelengths):
        """Test complete RIME simulation with all terms."""
        n_antennas = 4
        n_sources = 2

        # Create sources
        source_lmn = np.array(
            [
                [0.0, 0.0, 1.0],  # Zenith
                [0.1, 0.05, np.sqrt(1 - 0.1**2 - 0.05**2)],  # Off-axis
            ]
        )

        # Source altaz for beam
        source_altaz = np.array(
            [
                [np.pi / 2, 0.0],  # Zenith
                [np.deg2rad(84), 0.0],  # Off-axis
            ]
        )

        # Create Jones terms
        k_jones = GeometricPhaseJones(source_lmn, wavelengths)

        beam = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
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
                antenna_p=0,
                antenna_q=1,
                source_idx=src,
                freq_idx=0,
                time_idx=0,
                coherency_matrix=coherency,
                baseline_uvw=baseline_uvw,
            )

            assert vis.shape == (2, 2)
            # Off-axis source should have less power due to beam
            if src == 1:
                # Just check it's computed without error
                assert np.isfinite(vis).all()
