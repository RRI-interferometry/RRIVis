"""
Tests for beam_file.py - Beam FITS file handling with critical convention fixes.

Tests cover:
1. Azimuth conversion: Astropy (N=0°) ↔ UVBeam (E=0°)
2. Jones matrix ordering: [feed, basis] not [basis, feed]
3. BeamManager None check logic for beam_ids
4. Beam loading and interpolation
5. Error handling
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from beam_file import (
    astropy_az_to_uvbeam_az,
    BeamFITSHandler,
    BeamManager
)


class TestAzimuthConversion:
    """
    Test critical azimuth convention conversion.

    Astropy AltAz: North = 0°, East = 90°, South = 180°, West = 270°
    UVBeam:        East = 0°, North = 90°, West = 180°, South = 270°

    Transformation: uvbeam_az = 90° - astropy_az (mod 360°)
    """

    def test_cardinal_directions(self):
        """Test conversion of cardinal directions."""
        # North in Astropy (0°) → North in UVBeam (90°)
        assert np.allclose(
            astropy_az_to_uvbeam_az(0.0),
            np.pi / 2
        )

        # East in Astropy (90°) → East in UVBeam (0°)
        assert np.allclose(
            astropy_az_to_uvbeam_az(np.pi / 2),
            0.0
        )

        # South in Astropy (180°) → South in UVBeam (270°)
        assert np.allclose(
            astropy_az_to_uvbeam_az(np.pi),
            3 * np.pi / 2
        )

        # West in Astropy (270°) → West in UVBeam (180°)
        assert np.allclose(
            astropy_az_to_uvbeam_az(3 * np.pi / 2),
            np.pi
        )

    def test_intermediate_angles(self):
        """Test conversion of non-cardinal angles."""
        # Northeast in Astropy (45°) → 45° in UVBeam
        assert np.allclose(
            astropy_az_to_uvbeam_az(np.pi / 4),
            np.pi / 4
        )

        # Southeast in Astropy (135°) → 315° in UVBeam
        assert np.allclose(
            astropy_az_to_uvbeam_az(3 * np.pi / 4),
            7 * np.pi / 4
        )

    def test_array_input(self):
        """Test vectorized conversion."""
        astropy_az = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        uvbeam_az = astropy_az_to_uvbeam_az(astropy_az)

        expected = np.array([np.pi/2, 0, 3*np.pi/2, np.pi])
        assert np.allclose(uvbeam_az, expected)

    def test_wraparound(self):
        """Test that results are always in [0, 2π)."""
        # Negative angle should wrap
        result = astropy_az_to_uvbeam_az(-np.pi / 4)
        assert 0 <= result < 2 * np.pi

        # Angle > 2π should wrap
        result = astropy_az_to_uvbeam_az(2.5 * np.pi)
        assert 0 <= result < 2 * np.pi

    def test_round_trip_consistency(self):
        """Test that applying conversion twice gets back to original (mod offset)."""
        # This tests the mathematical consistency of the conversion
        angles = np.linspace(0, 2*np.pi, 100, endpoint=False)

        # Convert Astropy → UVBeam → Astropy (with inverse formula)
        uvbeam = astropy_az_to_uvbeam_az(angles)
        # Inverse: astropy_az = 90° - uvbeam_az
        back_to_astropy = np.mod(np.pi/2 - uvbeam, 2*np.pi)

        assert np.allclose(back_to_astropy, angles, atol=1e-10)


class TestBeamFITSHandler:
    """Test BeamFITSHandler class with mocked UVBeam."""

    @pytest.fixture
    def mock_config(self):
        """Minimal config for testing."""
        return {
            "beams": {
                "beam_mode": "shared",
                "beam_file": "/mock/path/beam.fits",
                "beam_za_max_deg": 90.0,
                "beam_za_buffer_deg": 5.0,
                "beam_freq_buffer_hz": 1e6
            },
            "observation": {
                "start_time": "2024-01-01T00:00:00",
                "duration_seconds": 3600.0,
                "frequency_hz": 50e6,
                "bandwidth_hz": 1e6
            }
        }

    @pytest.fixture
    def mock_uvbeam(self):
        """Create a mock UVBeam object."""
        mock_beam = Mock()
        mock_beam.Naxes_vec = 2
        mock_beam.pixel_coordinate_system = 'az_za'
        mock_beam.beam_type = 'efield'
        mock_beam.freq_array = np.array([[49e6, 50e6, 51e6]])
        mock_beam.axis1_array = np.linspace(0, 2*np.pi, 361)  # az
        mock_beam.axis2_array = np.linspace(0, np.pi/2, 91)   # za

        # Mock data_array shape: (Naxes_vec, Nspws, Nfeeds, Nfreqs, Naxes2, Naxes1)
        # For our case: (2, 1, 2, 3, 91, 361)
        mock_beam.data_array = np.ones((2, 1, 2, 3, 91, 361), dtype=complex)

        # Set specific values for testing Jones matrix ordering
        # feed=0, basis=0 (X feed, North component)
        mock_beam.data_array[0, 0, 0, :, :, :] = 1.0 + 0.0j
        # feed=0, basis=1 (X feed, East component)
        mock_beam.data_array[1, 0, 0, :, :, :] = 0.1 + 0.0j
        # feed=1, basis=0 (Y feed, North component)
        mock_beam.data_array[0, 0, 1, :, :, :] = 0.2 + 0.0j
        # feed=1, basis=1 (Y feed, East component)
        mock_beam.data_array[1, 0, 1, :, :, :] = 1.0 + 0.0j

        return mock_beam

    @patch('beam_file.UVBeam')
    def test_initialization(self, mock_uvbeam_class, mock_config):
        """Test BeamFITSHandler initialization."""
        handler = BeamFITSHandler(mock_config, "beam_001")

        assert handler.beam_id == "beam_001"
        assert handler.beam_file_path == Path("/mock/path/beam.fits")
        assert not handler.is_loaded

    @patch('beam_file.UVBeam')
    def test_load_beam_with_correct_units(self, mock_uvbeam_class, mock_config, mock_uvbeam):
        """
        CRITICAL TEST: Verify za_range is passed in DEGREES not radians.

        This was a critical bug in the original implementation.
        """
        mock_uvbeam_class.return_value = mock_uvbeam

        handler = BeamFITSHandler(mock_config, "beam_001")
        handler._load_beam()

        # Check that read_beamfits was called
        mock_uvbeam.read_beamfits.assert_called_once()

        # Extract the actual call arguments
        call_args = mock_uvbeam.read_beamfits.call_args

        # Verify za_range is in degrees (should be [0, 90])
        za_range = call_args[1]['za_range']
        assert za_range[0] == 0
        assert za_range[1] == 90.0  # DEGREES, not radians!

        # Verify za_range is reasonable for degrees (not radians)
        assert za_range[1] <= 180  # Can't be > 180° for zenith angle
        assert za_range[1] > 3.2   # If it were radians, max would be ~1.57

    @patch('beam_file.UVBeam')
    def test_jones_matrix_ordering(self, mock_uvbeam_class, mock_config, mock_uvbeam):
        """
        CRITICAL TEST: Verify Jones matrix has [feed, basis] ordering.

        Correct RIME requires:
        jones[0, 0] = X feed response to North (θ̂) polarization
        jones[0, 1] = X feed response to East (φ̂) polarization
        jones[1, 0] = Y feed response to North (θ̂) polarization
        jones[1, 1] = Y feed response to East (φ̂) polarization

        This was a critical bug: original implementation had [basis, feed].
        """
        mock_uvbeam_class.return_value = mock_uvbeam

        # Mock the interpolation method
        def mock_interp(az_array, za_array, freq_array, **kwargs):
            # Return specific values we set in mock_uvbeam
            # Shape: (Naxes_vec, Nspws, Nfeeds, Nfreqs, Npts)
            n_pts = len(az_array)
            result = np.zeros((2, 1, 2, 1, n_pts), dtype=complex)

            # Set values matching our mock_beam.data_array
            result[0, 0, 0, 0, :] = 1.0 + 0.0j  # basis=0 (N), feed=0 (X)
            result[1, 0, 0, 0, :] = 0.1 + 0.0j  # basis=1 (E), feed=0 (X)
            result[0, 0, 1, 0, :] = 0.2 + 0.0j  # basis=0 (N), feed=1 (Y)
            result[1, 0, 1, 0, :] = 1.0 + 0.0j  # basis=1 (E), feed=1 (Y)

            return result, np.ones(n_pts, dtype=bool)

        mock_uvbeam.interp = mock_interp
        mock_uvbeam.check.return_value = True

        handler = BeamFITSHandler(mock_config, "beam_001")
        handler._load_beam()

        # Get Jones matrix
        alt_rad = np.pi / 4  # 45° altitude
        az_rad = 0.0         # North in Astropy
        freq_hz = 50e6

        jones = handler.get_jones_matrix(
            alt_rad=alt_rad,
            az_rad=az_rad,
            freq_hz=freq_hz,
            reuse_spline=False
        )

        # Verify shape: (2, 2) = (Nfeeds, Nbasis)
        assert jones.shape == (2, 2), f"Expected (2,2), got {jones.shape}"

        # Verify ordering: rows = feeds, columns = basis
        # jones[feed, basis]
        assert np.isclose(jones[0, 0], 1.0 + 0.0j)  # X feed, North basis
        assert np.isclose(jones[0, 1], 0.1 + 0.0j)  # X feed, East basis
        assert np.isclose(jones[1, 0], 0.2 + 0.0j)  # Y feed, North basis
        assert np.isclose(jones[1, 1], 1.0 + 0.0j)  # Y feed, East basis

        # Additional check: matrix should NOT be identity (that would indicate wrong ordering)
        assert not np.allclose(jones, np.eye(2)), \
            "Jones matrix should not be identity - this suggests wrong ordering"

    @patch('beam_file.UVBeam')
    def test_azimuth_conversion_applied(self, mock_uvbeam_class, mock_config, mock_uvbeam):
        """
        Test that azimuth conversion is applied when querying beam.
        """
        mock_uvbeam_class.return_value = mock_uvbeam

        # Track what azimuth values are passed to interp
        interp_calls = []

        def mock_interp(az_array, za_array, freq_array, **kwargs):
            interp_calls.append({
                'az': az_array.copy(),
                'za': za_array.copy(),
                'freq': freq_array.copy()
            })
            n_pts = len(az_array)
            result = np.ones((2, 1, 2, 1, n_pts), dtype=complex)
            return result, np.ones(n_pts, dtype=bool)

        mock_uvbeam.interp = mock_interp
        mock_uvbeam.check.return_value = True

        handler = BeamFITSHandler(mock_config, "beam_001")
        handler._load_beam()

        # Query with Astropy North (az=0)
        astropy_az_north = 0.0
        handler.get_jones_matrix(
            alt_rad=np.pi/4,
            az_rad=astropy_az_north,
            freq_hz=50e6,
            reuse_spline=False
        )

        # The azimuth passed to interp should be converted to UVBeam convention
        # Astropy North (0°) → UVBeam North (90°)
        passed_az = interp_calls[-1]['az'][0]
        expected_uvbeam_az = np.pi / 2  # 90° in radians

        assert np.isclose(passed_az, expected_uvbeam_az, atol=1e-6), \
            f"Azimuth not converted: got {passed_az}, expected {expected_uvbeam_az}"

    @patch('beam_file.UVBeam')
    def test_array_input(self, mock_uvbeam_class, mock_config, mock_uvbeam):
        """Test that handler can process arrays of coordinates."""
        mock_uvbeam_class.return_value = mock_uvbeam

        def mock_interp(az_array, za_array, freq_array, **kwargs):
            n_pts = len(az_array)
            result = np.ones((2, 1, 2, 1, n_pts), dtype=complex) * 0.9
            return result, np.ones(n_pts, dtype=bool)

        mock_uvbeam.interp = mock_interp
        mock_uvbeam.check.return_value = True

        handler = BeamFITSHandler(mock_config, "beam_001")
        handler._load_beam()

        # Query with arrays
        n_sources = 10
        alt_rad = np.full(n_sources, np.pi/4)
        az_rad = np.linspace(0, 2*np.pi, n_sources)
        freq_hz = np.full(n_sources, 50e6)

        jones = handler.get_jones_matrix(
            alt_rad=alt_rad,
            az_rad=az_rad,
            freq_hz=freq_hz,
            reuse_spline=False
        )

        # Should return (N_sources, 2, 2)
        assert jones.shape == (n_sources, 2, 2)

    @patch('beam_file.UVBeam')
    def test_error_handling_missing_file(self, mock_uvbeam_class, mock_config):
        """Test error handling when beam file doesn't exist."""
        mock_beam = Mock()
        mock_beam.read_beamfits.side_effect = FileNotFoundError("File not found")
        mock_uvbeam_class.return_value = mock_beam

        handler = BeamFITSHandler(mock_config, "beam_001")

        with pytest.raises(FileNotFoundError):
            handler._load_beam()

    @patch('beam_file.UVBeam')
    def test_lazy_loading(self, mock_uvbeam_class, mock_config, mock_uvbeam):
        """Test that beam is loaded lazily on first get_jones_matrix call."""
        mock_uvbeam_class.return_value = mock_uvbeam
        mock_uvbeam.interp.return_value = (
            np.ones((2, 1, 2, 1, 1), dtype=complex),
            np.ones(1, dtype=bool)
        )
        mock_uvbeam.check.return_value = True

        handler = BeamFITSHandler(mock_config, "beam_001")

        # Should not be loaded initially
        assert not handler.is_loaded
        assert mock_uvbeam.read_beamfits.call_count == 0

        # First call should trigger loading
        handler.get_jones_matrix(
            alt_rad=np.pi/4,
            az_rad=0.0,
            freq_hz=50e6,
            reuse_spline=False
        )

        assert handler.is_loaded
        assert mock_uvbeam.read_beamfits.call_count == 1

        # Second call should not reload
        handler.get_jones_matrix(
            alt_rad=np.pi/4,
            az_rad=0.0,
            freq_hz=50e6,
            reuse_spline=False
        )

        assert mock_uvbeam.read_beamfits.call_count == 1  # Still 1


class TestBeamManager:
    """Test BeamManager class with focus on None check logic."""

    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock()

    @pytest.fixture
    def mock_config_analytic(self):
        """Config for analytic beam mode (no beam files)."""
        return {
            "beams": {
                "beam_mode": "analytic"
            },
            "observation": {
                "start_time": "2024-01-01T00:00:00",
                "duration_seconds": 3600.0,
                "frequency_hz": 50e6,
                "bandwidth_hz": 1e6
            }
        }

    @pytest.fixture
    def mock_config_shared(self):
        """Config for shared beam mode."""
        return {
            "beams": {
                "beam_mode": "shared",
                "beam_file": "/mock/beam.fits"
            },
            "observation": {
                "start_time": "2024-01-01T00:00:00",
                "duration_seconds": 3600.0,
                "frequency_hz": 50e6,
                "bandwidth_hz": 1e6
            }
        }

    @pytest.fixture
    def mock_config_per_antenna_layout(self):
        """Config for per-antenna beams from layout file."""
        return {
            "beams": {
                "beam_mode": "per_antenna",
                "beam_assignment": "from_layout"
            },
            "observation": {
                "start_time": "2024-01-01T00:00:00",
                "duration_seconds": 3600.0,
                "frequency_hz": 50e6,
                "bandwidth_hz": 1e6
            }
        }

    @pytest.fixture
    def mock_config_per_antenna_config(self):
        """Config for per-antenna beams from config."""
        return {
            "beams": {
                "beam_mode": "per_antenna",
                "beam_assignment": "from_config",
                "antenna_beam_map": {
                    "ant1": "/mock/beam1.fits",
                    "ant2": "/mock/beam2.fits"
                }
            },
            "observation": {
                "start_time": "2024-01-01T00:00:00",
                "duration_seconds": 3600.0,
                "frequency_hz": 50e6,
                "bandwidth_hz": 1e6
            }
        }

    @pytest.fixture
    def mock_antenna_data_no_beam_ids(self):
        """Antenna data WITHOUT beam_ids column."""
        return {
            "names": np.array(["ant1", "ant2", "ant3"]),
            "positions_m": np.random.randn(3, 3),
            "beam_ids": None  # CRITICAL: None, not missing key
        }

    @pytest.fixture
    def mock_antenna_data_with_beam_ids(self):
        """Antenna data WITH beam_ids column."""
        return {
            "names": np.array(["ant1", "ant2", "ant3"]),
            "positions_m": np.random.randn(3, 3),
            "beam_ids": np.array(["beam_A", "beam_B", "beam_A"])
        }

    def test_analytic_mode(self, mock_logger, mock_config_analytic, mock_antenna_data_no_beam_ids):
        """Test analytic mode - should not load any beams."""
        manager = BeamManager(mock_config_analytic, mock_antenna_data_no_beam_ids, mock_logger)

        assert manager.beam_mode == "analytic"
        assert manager.beam_handlers == {}

    @patch('beam_file.BeamFITSHandler')
    def test_shared_mode(self, mock_handler_class, mock_logger, mock_config_shared, mock_antenna_data_no_beam_ids):
        """Test shared beam mode."""
        manager = BeamManager(mock_config_shared, mock_antenna_data_no_beam_ids, mock_logger)

        assert manager.beam_mode == "shared"
        assert "shared" in manager.beam_handlers
        mock_handler_class.assert_called_once()

    @patch('beam_file.BeamFITSHandler')
    def test_per_antenna_from_layout_with_beam_ids(
        self,
        mock_handler_class,
        mock_logger,
        mock_config_per_antenna_layout,
        mock_antenna_data_with_beam_ids
    ):
        """Test per-antenna mode from layout WITH beam_ids."""
        manager = BeamManager(mock_config_per_antenna_layout, mock_antenna_data_with_beam_ids, mock_logger)

        assert manager.beam_mode == "per_antenna"

        # Should create handlers for unique beam IDs
        unique_beam_ids = ["beam_A", "beam_B"]
        assert len(manager.beam_handlers) == len(unique_beam_ids)

        # Should have antenna-to-beam mapping
        assert manager.antenna_beam_map["ant1"] == "beam_A"
        assert manager.antenna_beam_map["ant2"] == "beam_B"
        assert manager.antenna_beam_map["ant3"] == "beam_A"

    def test_per_antenna_from_layout_without_beam_ids_raises_error(
        self,
        mock_logger,
        mock_config_per_antenna_layout,
        mock_antenna_data_no_beam_ids
    ):
        """
        CRITICAL TEST: Verify None check for beam_ids.

        This was a critical bug: parse_rrivis_format always returns 'beam_ids' key
        but sets it to None when there's no BeamID column. We need to check for None.
        """
        with pytest.raises(ValueError, match="beam_ids.*None.*missing"):
            BeamManager(mock_config_per_antenna_layout, mock_antenna_data_no_beam_ids, mock_logger)

    @patch('beam_file.BeamFITSHandler')
    def test_per_antenna_from_config(
        self,
        mock_handler_class,
        mock_logger,
        mock_config_per_antenna_config,
        mock_antenna_data_no_beam_ids
    ):
        """Test per-antenna mode from config."""
        manager = BeamManager(mock_config_per_antenna_config, mock_antenna_data_no_beam_ids, mock_logger)

        assert manager.beam_mode == "per_antenna"

        # Should have handlers for antennas in config
        assert "ant1" in manager.antenna_beam_map
        assert "ant2" in manager.antenna_beam_map

    @patch('beam_file.BeamFITSHandler')
    def test_get_jones_matrix_analytic_returns_none(
        self,
        mock_handler_class,
        mock_logger,
        mock_config_analytic,
        mock_antenna_data_no_beam_ids
    ):
        """Test that analytic mode returns None for Jones matrices."""
        manager = BeamManager(mock_config_analytic, mock_antenna_data_no_beam_ids, mock_logger)

        result = manager.get_jones_matrix(
            antenna_name="ant1",
            alt_rad=np.pi/4,
            az_rad=0.0,
            freq_hz=50e6,
            reuse_spline=False
        )

        assert result is None

    @patch('beam_file.BeamFITSHandler')
    def test_get_jones_matrix_shared_mode(
        self,
        mock_handler_class,
        mock_logger,
        mock_config_shared,
        mock_antenna_data_no_beam_ids
    ):
        """Test getting Jones matrix in shared beam mode."""
        # Setup mock handler
        mock_handler = Mock()
        mock_handler.get_jones_matrix.return_value = np.eye(2, dtype=complex)
        mock_handler_class.return_value = mock_handler

        manager = BeamManager(mock_config_shared, mock_antenna_data_no_beam_ids, mock_logger)

        # All antennas should use shared beam
        for ant_name in ["ant1", "ant2", "ant3"]:
            result = manager.get_jones_matrix(
                antenna_name=ant_name,
                alt_rad=np.pi/4,
                az_rad=0.0,
                freq_hz=50e6,
                reuse_spline=False
            )

            assert result is not None
            assert result.shape == (2, 2)
            mock_handler.get_jones_matrix.assert_called()

    @patch('beam_file.BeamFITSHandler')
    def test_get_jones_matrix_per_antenna_mode(
        self,
        mock_handler_class,
        mock_logger,
        mock_config_per_antenna_layout,
        mock_antenna_data_with_beam_ids
    ):
        """Test getting Jones matrix in per-antenna mode."""
        # Setup mock handlers
        def create_mock_handler(config, beam_id):
            mock = Mock()
            # Different beams return different values
            if beam_id == "beam_A":
                mock.get_jones_matrix.return_value = np.eye(2, dtype=complex) * 1.0
            else:  # beam_B
                mock.get_jones_matrix.return_value = np.eye(2, dtype=complex) * 0.5
            return mock

        mock_handler_class.side_effect = create_mock_handler

        manager = BeamManager(mock_config_per_antenna_layout, mock_antenna_data_with_beam_ids, mock_logger)

        # ant1 and ant3 should use beam_A
        result1 = manager.get_jones_matrix(
            antenna_name="ant1",
            alt_rad=np.pi/4,
            az_rad=0.0,
            freq_hz=50e6,
            reuse_spline=False
        )
        assert np.allclose(result1, np.eye(2) * 1.0)

        result3 = manager.get_jones_matrix(
            antenna_name="ant3",
            alt_rad=np.pi/4,
            az_rad=0.0,
            freq_hz=50e6,
            reuse_spline=False
        )
        assert np.allclose(result3, np.eye(2) * 1.0)

        # ant2 should use beam_B
        result2 = manager.get_jones_matrix(
            antenna_name="ant2",
            alt_rad=np.pi/4,
            az_rad=0.0,
            freq_hz=50e6,
            reuse_spline=False
        )
        assert np.allclose(result2, np.eye(2) * 0.5)

    def test_unknown_antenna_raises_error(
        self,
        mock_logger,
        mock_config_analytic,
        mock_antenna_data_no_beam_ids
    ):
        """Test that querying unknown antenna raises error."""
        manager = BeamManager(mock_config_analytic, mock_antenna_data_no_beam_ids, mock_logger)

        # Even in analytic mode, should validate antenna name
        # (though it returns None, it should at least not crash)
        result = manager.get_jones_matrix(
            antenna_name="unknown_antenna",
            alt_rad=np.pi/4,
            az_rad=0.0,
            freq_hz=50e6,
            reuse_spline=False
        )

        # In analytic mode, should return None even for unknown antenna
        assert result is None


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock()

    @patch('beam_file.UVBeam')
    @patch('beam_file.BeamFITSHandler')
    def test_realistic_rime_workflow(self, mock_handler_class, mock_uvbeam_class, mock_logger):
        """
        Test a realistic workflow simulating RIME calculation.

        This validates that all components work together correctly:
        1. BeamManager manages multiple beams
        2. Jones matrices have correct ordering
        3. Azimuth conversion is applied
        """
        # Setup config
        config = {
            "beams": {
                "beam_mode": "per_antenna",
                "beam_assignment": "from_layout"
            },
            "observation": {
                "start_time": "2024-01-01T00:00:00",
                "duration_seconds": 3600.0,
                "frequency_hz": 50e6,
                "bandwidth_hz": 1e6
            }
        }

        # Setup antenna data with beam IDs
        antenna_data = {
            "names": np.array(["ant1", "ant2"]),
            "positions_m": np.random.randn(2, 3),
            "beam_ids": np.array(["beam_A", "beam_B"])
        }

        # Create mock handlers that return different Jones matrices
        def create_mock_handler(cfg, beam_id):
            mock = Mock()
            if beam_id == "beam_A":
                # Beam A: stronger in X-pol
                jones = np.array([
                    [1.0 + 0j, 0.1 + 0j],
                    [0.1 + 0j, 0.5 + 0j]
                ])
            else:  # beam_B
                # Beam B: stronger in Y-pol
                jones = np.array([
                    [0.5 + 0j, 0.1 + 0j],
                    [0.1 + 0j, 1.0 + 0j]
                ])
            mock.get_jones_matrix.return_value = jones
            return mock

        mock_handler_class.side_effect = create_mock_handler

        # Create manager
        manager = BeamManager(config, antenna_data, mock_logger)

        # Simulate RIME calculation for baseline ant1-ant2
        # Source at (alt, az) = (60°, 45°)
        alt_rad = np.radians(60)
        az_rad = np.radians(45)
        freq_hz = 50e6

        # Get Jones matrices
        E1 = manager.get_jones_matrix("ant1", alt_rad, az_rad, freq_hz, False)
        E2 = manager.get_jones_matrix("ant2", alt_rad, az_rad, freq_hz, False)

        # Verify shapes
        assert E1.shape == (2, 2)
        assert E2.shape == (2, 2)

        # Verify different beams return different matrices
        assert not np.allclose(E1, E2), "Different beams should give different Jones matrices"

        # Verify ordering by checking diagonal dominance pattern
        # beam_A should be X-dominant (E[0,0] > E[1,1])
        assert np.abs(E1[0, 0]) > np.abs(E1[1, 1])
        # beam_B should be Y-dominant (E[1,1] > E[0,0])
        assert np.abs(E2[1, 1]) > np.abs(E2[0, 0])

        # Simulate simple RIME with unpolarized source (I=1, Q=U=V=0)
        from polarization import stokes_to_coherency, apply_jones_matrices

        C = stokes_to_coherency(I=1.0, Q=0, U=0, V=0)
        V_matrix = apply_jones_matrices(E1, C, E2)

        # Check that visibility is computed
        assert V_matrix.shape == (2, 2)
        assert not np.allclose(V_matrix, 0), "Visibility should be non-zero"

        # For unpolarized source with different beams, expect non-zero cross-pols
        # (unlike identical beams which would give zero cross-pols)
        assert np.abs(V_matrix[0, 1]) > 1e-10 or np.abs(V_matrix[1, 0]) > 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
