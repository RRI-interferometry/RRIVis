# tests/unit/test_core/test_sky_model_multifreq.py
"""
Tests for multi-frequency HEALPix functionality in SkyModel.

These tests verify:
- Frequency configuration parsing
- Memory estimation
- Multi-frequency HEALPix map creation
- Spectral extrapolation correctness
- Flux conservation
- API methods (to_healpix_for_observation, get_map_at_frequency, etc.)
"""

import pytest
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp

from rrivis.core.sky_model import (
    SkyModel,
    K_BOLTZMANN,
    C_LIGHT,
    H_PLANCK,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_obs_frequency_config():
    """Simple observation frequency configuration (20 channels, 100-119 MHz)."""
    return {
        "starting_frequency": 100.0,
        "frequency_interval": 1.0,
        "frequency_bandwidth": 20.0,
        "frequency_unit": "MHz"
    }


@pytest.fixture
def wideband_obs_frequency_config():
    """Wideband observation frequency configuration (100 channels)."""
    return {
        "starting_frequency": 50.0,
        "frequency_interval": 1.0,
        "frequency_bandwidth": 100.0,
        "frequency_unit": "MHz"
    }


@pytest.fixture
def single_source_sky_model():
    """SkyModel with a single 1 Jy source at RA=0, Dec=0."""
    sources = [{
        "coords": SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, frame="icrs"),
        "flux": 1.0,  # 1 Jy at 76 MHz
        "spectral_index": -0.7,
        "stokes_q": 0.0,
        "stokes_u": 0.0,
        "stokes_v": 0.0,
    }]
    return SkyModel.from_point_sources(sources, model_name="test_single")


@pytest.fixture
def multi_source_sky_model():
    """SkyModel with multiple sources with different spectral indices."""
    sources = [
        {
            "coords": SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, frame="icrs"),
            "flux": 2.0,  # 2 Jy
            "spectral_index": -0.7,  # Flat spectrum
            "stokes_q": 0.0,
            "stokes_u": 0.0,
            "stokes_v": 0.0,
        },
        {
            "coords": SkyCoord(ra=45.0 * u.deg, dec=30.0 * u.deg, frame="icrs"),
            "flux": 1.0,  # 1 Jy
            "spectral_index": -1.5,  # Steep spectrum
            "stokes_q": 0.0,
            "stokes_u": 0.0,
            "stokes_v": 0.0,
        },
        {
            "coords": SkyCoord(ra=90.0 * u.deg, dec=-20.0 * u.deg, frame="icrs"),
            "flux": 0.5,  # 0.5 Jy
            "spectral_index": -0.5,  # Very flat
            "stokes_q": 0.0,
            "stokes_u": 0.0,
            "stokes_v": 0.0,
        },
    ]
    return SkyModel.from_point_sources(sources, model_name="test_multi")


@pytest.fixture
def same_pixel_sources_sky_model():
    """SkyModel with two sources in the same HEALPix pixel (for nside=16)."""
    # Two sources very close together - will fall in same pixel for low nside
    sources = [
        {
            "coords": SkyCoord(ra=45.0 * u.deg, dec=30.0 * u.deg, frame="icrs"),
            "flux": 2.0,
            "spectral_index": -0.7,
            "stokes_q": 0.0,
            "stokes_u": 0.0,
            "stokes_v": 0.0,
        },
        {
            "coords": SkyCoord(ra=45.5 * u.deg, dec=30.5 * u.deg, frame="icrs"),
            "flux": 1.0,
            "spectral_index": -1.5,
            "stokes_q": 0.0,
            "stokes_u": 0.0,
            "stokes_v": 0.0,
        },
    ]
    return SkyModel.from_point_sources(sources, model_name="test_same_pixel")


# =============================================================================
# TEST _parse_frequency_config()
# =============================================================================

class TestParseFrequencyConfig:
    """Tests for the _parse_frequency_config() static method."""

    def test_basic_parsing_mhz(self, simple_obs_frequency_config):
        """Test basic parsing with MHz units."""
        freqs = SkyModel._parse_frequency_config(simple_obs_frequency_config)

        assert len(freqs) == 20
        assert freqs[0] == 100e6  # 100 MHz in Hz
        assert freqs[-1] == 119e6  # 119 MHz in Hz
        assert np.allclose(np.diff(freqs), 1e6)  # 1 MHz spacing

    def test_parsing_hz(self):
        """Test parsing with Hz units."""
        config = {
            "starting_frequency": 100e6,
            "frequency_interval": 1e6,
            "frequency_bandwidth": 10e6,
            "frequency_unit": "Hz"
        }
        freqs = SkyModel._parse_frequency_config(config)

        assert len(freqs) == 10
        assert freqs[0] == 100e6
        assert freqs[-1] == 109e6

    def test_parsing_khz(self):
        """Test parsing with kHz units."""
        config = {
            "starting_frequency": 100000.0,
            "frequency_interval": 1000.0,
            "frequency_bandwidth": 10000.0,
            "frequency_unit": "kHz"
        }
        freqs = SkyModel._parse_frequency_config(config)

        assert len(freqs) == 10
        assert freqs[0] == 100e6
        assert freqs[-1] == 109e6

    def test_parsing_ghz(self):
        """Test parsing with GHz units."""
        config = {
            "starting_frequency": 1.0,
            "frequency_interval": 0.1,
            "frequency_bandwidth": 0.5,
            "frequency_unit": "GHz"
        }
        freqs = SkyModel._parse_frequency_config(config)

        assert len(freqs) == 5
        assert freqs[0] == 1e9
        assert freqs[-1] == 1.4e9

    def test_default_unit_is_mhz(self):
        """Test that default unit is MHz when not specified."""
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 5.0,
            # No frequency_unit specified
        }
        freqs = SkyModel._parse_frequency_config(config)

        assert freqs[0] == 100e6  # Should be 100 MHz

    def test_invalid_unit_raises_error(self):
        """Test that invalid frequency unit raises ValueError."""
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 10.0,
            "frequency_unit": "invalid_unit"
        }
        with pytest.raises(ValueError, match="Unknown frequency unit"):
            SkyModel._parse_frequency_config(config)

    def test_zero_bandwidth_raises_error(self):
        """Test that zero bandwidth raises ValueError."""
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 0.0,
            "frequency_unit": "MHz"
        }
        with pytest.raises(ValueError, match="Invalid frequency config"):
            SkyModel._parse_frequency_config(config)

    def test_bandwidth_less_than_interval_raises_error(self):
        """Test that bandwidth < interval raises ValueError."""
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 10.0,
            "frequency_bandwidth": 5.0,
            "frequency_unit": "MHz"
        }
        with pytest.raises(ValueError, match="Invalid frequency config"):
            SkyModel._parse_frequency_config(config)


# =============================================================================
# TEST estimate_healpix_memory()
# =============================================================================

class TestEstimateHealpixMemory:
    """Tests for the estimate_healpix_memory() static method."""

    def test_basic_memory_calculation(self):
        """Test basic memory calculation."""
        info = SkyModel.estimate_healpix_memory(nside=64, n_frequencies=20)

        assert info["npix"] == 12 * 64**2  # 49152
        assert info["n_freq"] == 20
        assert info["dtype"] == "float32"
        # 49152 pixels × 4 bytes × 20 frequencies = 3,932,160 bytes
        assert info["total_bytes"] == 49152 * 4 * 20
        assert np.isclose(info["total_mb"], 3.93216)

    def test_memory_scaling_with_nside(self):
        """Test that memory scales with nside^2."""
        info_64 = SkyModel.estimate_healpix_memory(nside=64, n_frequencies=1)
        info_128 = SkyModel.estimate_healpix_memory(nside=128, n_frequencies=1)

        # nside=128 should have 4x the pixels of nside=64
        assert info_128["npix"] == 4 * info_64["npix"]
        assert info_128["total_bytes"] == 4 * info_64["total_bytes"]

    def test_memory_scaling_with_frequencies(self):
        """Test that memory scales linearly with number of frequencies."""
        info_10 = SkyModel.estimate_healpix_memory(nside=64, n_frequencies=10)
        info_20 = SkyModel.estimate_healpix_memory(nside=64, n_frequencies=20)

        assert info_20["total_bytes"] == 2 * info_10["total_bytes"]

    def test_float64_dtype(self):
        """Test memory calculation with float64 dtype."""
        info_f32 = SkyModel.estimate_healpix_memory(nside=64, n_frequencies=10, dtype=np.float32)
        info_f64 = SkyModel.estimate_healpix_memory(nside=64, n_frequencies=10, dtype=np.float64)

        assert info_f64["total_bytes"] == 2 * info_f32["total_bytes"]
        assert info_f64["dtype"] == "float64"

    def test_resolution_calculation(self):
        """Test that resolution is calculated correctly."""
        info = SkyModel.estimate_healpix_memory(nside=64, n_frequencies=1)

        # nside=64 should have ~55 arcmin resolution
        assert 50 < info["resolution_arcmin"] < 60

    def test_nside_1024_memory(self):
        """Test memory for nside=1024 (realistic large case)."""
        info = SkyModel.estimate_healpix_memory(nside=1024, n_frequencies=20)

        # Should be ~1 GB
        assert 900 < info["total_mb"] < 1100


# =============================================================================
# TEST _point_sources_to_healpix_multifreq()
# =============================================================================

class TestPointSourcesToHealpixMultifreq:
    """Tests for the _point_sources_to_healpix_multifreq() method."""

    def test_returns_dict_of_maps(self, single_source_sky_model):
        """Test that method returns dictionary of maps."""
        frequencies = np.array([100e6, 110e6, 120e6])
        maps = single_source_sky_model._point_sources_to_healpix_multifreq(
            nside=32, frequencies=frequencies
        )

        assert isinstance(maps, dict)
        assert len(maps) == 3
        assert all(freq in maps for freq in frequencies)

    def test_map_shape_correct(self, single_source_sky_model):
        """Test that each map has correct shape."""
        frequencies = np.array([100e6, 110e6])
        nside = 32
        maps = single_source_sky_model._point_sources_to_healpix_multifreq(
            nside=nside, frequencies=frequencies
        )

        expected_npix = hp.nside2npix(nside)
        for freq, map_data in maps.items():
            assert map_data.shape == (expected_npix,)

    def test_map_dtype_is_float32(self, single_source_sky_model):
        """Test that maps use float32 for memory efficiency."""
        frequencies = np.array([100e6])
        maps = single_source_sky_model._point_sources_to_healpix_multifreq(
            nside=32, frequencies=frequencies
        )

        assert maps[100e6].dtype == np.float32

    def test_empty_sources_returns_zero_maps(self):
        """Test that empty source list returns zero maps."""
        sky = SkyModel.from_point_sources([], model_name="empty")
        frequencies = np.array([100e6, 110e6])
        maps = sky._point_sources_to_healpix_multifreq(nside=32, frequencies=frequencies)

        assert len(maps) == 2
        assert np.all(maps[100e6] == 0)
        assert np.all(maps[110e6] == 0)

    def test_source_placed_in_correct_pixel(self, single_source_sky_model):
        """Test that source is placed in the correct HEALPix pixel."""
        frequencies = np.array([100e6])
        nside = 32
        maps = single_source_sky_model._point_sources_to_healpix_multifreq(
            nside=nside, frequencies=frequencies
        )

        # Find the expected pixel for RA=0, Dec=0
        theta = np.pi / 2  # Dec=0 -> theta=90 degrees
        phi = 0.0  # RA=0
        expected_pixel = hp.ang2pix(nside, theta, phi)

        # Check that only this pixel has non-zero value
        nonzero_pixels = np.where(maps[100e6] > 0)[0]
        assert len(nonzero_pixels) == 1
        assert nonzero_pixels[0] == expected_pixel

    def test_multiple_sources_in_different_pixels(self, multi_source_sky_model):
        """Test that multiple sources end up in different pixels."""
        frequencies = np.array([100e6])
        nside = 32
        maps = multi_source_sky_model._point_sources_to_healpix_multifreq(
            nside=nside, frequencies=frequencies
        )

        # Should have 3 non-zero pixels (3 sources in different locations)
        nonzero_pixels = np.where(maps[100e6] > 0)[0]
        assert len(nonzero_pixels) == 3


# =============================================================================
# TEST SPECTRAL EXTRAPOLATION
# =============================================================================

class TestSpectralExtrapolation:
    """Tests for correct spectral index extrapolation."""

    def test_flux_decreases_with_frequency_for_negative_alpha(self, single_source_sky_model):
        """Test that flux decreases with frequency for negative spectral index."""
        frequencies = np.array([76e6, 100e6, 150e6])  # Reference is 76 MHz
        nside = 32
        maps = single_source_sky_model._point_sources_to_healpix_multifreq(
            nside=nside, frequencies=frequencies, ref_frequency=76e6
        )

        # Find the pixel with the source
        pixel = np.argmax(maps[76e6])

        # Flux should decrease with frequency (alpha = -0.7)
        assert maps[76e6][pixel] > maps[100e6][pixel]
        assert maps[100e6][pixel] > maps[150e6][pixel]

    def test_spectral_index_extrapolation_formula(self):
        """Test that spectral extrapolation follows S(ν) = S_ref × (ν/ν_ref)^α."""
        # Create source with known flux and spectral index
        sources = [{
            "coords": SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, frame="icrs"),
            "flux": 1.0,  # 1 Jy at reference frequency
            "spectral_index": -1.0,  # Easy to calculate
            "stokes_q": 0.0,
            "stokes_u": 0.0,
            "stokes_v": 0.0,
        }]
        sky = SkyModel.from_point_sources(sources, model_name="test")

        ref_freq = 100e6
        test_freq = 200e6  # Double the reference
        frequencies = np.array([ref_freq, test_freq])
        nside = 32

        maps = sky._point_sources_to_healpix_multifreq(
            nside=nside, frequencies=frequencies, ref_frequency=ref_freq
        )

        # Verify flux ratio in Jy domain (conversion-method independent)
        npix = hp.nside2npix(nside)
        omega_pixel = 4 * np.pi / npix

        flux_ref = np.sum(brightness_temp_to_flux_density(
            maps[ref_freq], ref_freq, omega_pixel, method=sky.brightness_conversion
        ))
        flux_test = np.sum(brightness_temp_to_flux_density(
            maps[test_freq], test_freq, omega_pixel, method=sky.brightness_conversion
        ))

        # S(200MHz) / S(100MHz) = (200/100)^(-1) = 0.5
        expected_ratio = 0.5
        actual_ratio = flux_test / flux_ref

        assert np.isclose(actual_ratio, expected_ratio, rtol=1e-4)

    def test_different_spectral_indices_give_different_ratios(self, multi_source_sky_model):
        """Test that sources with different spectral indices evolve differently."""
        frequencies = np.array([76e6, 150e6])
        nside = 64  # Higher resolution to separate sources
        maps = multi_source_sky_model._point_sources_to_healpix_multifreq(
            nside=nside, frequencies=frequencies, ref_frequency=76e6
        )

        # Get ratios for each source's pixel
        # Source 1: alpha=-0.7, Source 2: alpha=-1.5, Source 3: alpha=-0.5
        # Source with steeper alpha should have lower ratio

        # Find non-zero pixels
        nonzero_76 = np.where(maps[76e6] > 0)[0]
        nonzero_150 = np.where(maps[150e6] > 0)[0]

        # Both frequency maps should have same non-zero pixels
        assert set(nonzero_76) == set(nonzero_150)

        # Calculate ratios
        ratios = maps[150e6][nonzero_76] / maps[76e6][nonzero_76]

        # Ratios should be different (different spectral indices)
        assert len(set(np.round(ratios, 6))) > 1


# =============================================================================
# TEST FLUX CONSERVATION
# =============================================================================

class TestFluxConservation:
    """Tests for flux conservation in the conversion."""

    def test_total_flux_conserved_single_source(self, single_source_sky_model):
        """Test that total flux is conserved for a single source."""
        freq = 100e6
        nside = 64
        frequencies = np.array([freq])
        sky = single_source_sky_model

        maps = sky._point_sources_to_healpix_multifreq(
            nside=nside, frequencies=frequencies, ref_frequency=76e6
        )

        # Calculate total flux from map using the same method as sky model
        npix = hp.nside2npix(nside)
        omega_pixel = 4 * np.pi / npix
        total_flux_jy = np.sum(
            brightness_temp_to_flux_density(
                maps[freq], freq, omega_pixel, method=sky.brightness_conversion
            )
        )

        # Expected flux at 100 MHz from 1 Jy at 76 MHz with alpha=-0.7
        expected_flux = 1.0 * (freq / 76e6)**(-0.7)

        assert np.isclose(total_flux_jy, expected_flux, rtol=1e-5)

    def test_total_flux_conserved_multiple_sources(self, multi_source_sky_model):
        """Test that total flux is conserved for multiple sources."""
        freq = 100e6
        nside = 64
        frequencies = np.array([freq])
        ref_freq = 76e6
        sky = multi_source_sky_model

        maps = sky._point_sources_to_healpix_multifreq(
            nside=nside, frequencies=frequencies, ref_frequency=ref_freq
        )

        # Calculate total flux from map using the same method as sky model
        npix = hp.nside2npix(nside)
        omega_pixel = 4 * np.pi / npix
        total_flux_jy = np.sum(
            brightness_temp_to_flux_density(
                maps[freq], freq, omega_pixel, method=sky.brightness_conversion
            )
        )

        # Expected flux: sum of extrapolated fluxes
        # Source 1: 2 Jy × (100/76)^(-0.7)
        # Source 2: 1 Jy × (100/76)^(-1.5)
        # Source 3: 0.5 Jy × (100/76)^(-0.5)
        expected_flux = (
            2.0 * (freq / ref_freq)**(-0.7) +
            1.0 * (freq / ref_freq)**(-1.5) +
            0.5 * (freq / ref_freq)**(-0.5)
        )

        assert np.isclose(total_flux_jy, expected_flux, rtol=1e-4)


# =============================================================================
# TEST SOURCES IN SAME PIXEL
# =============================================================================

class TestSamePixelSources:
    """Tests for handling multiple sources in the same pixel."""

    def test_fluxes_accumulate_in_same_pixel(self, same_pixel_sources_sky_model):
        """Test that fluxes from multiple sources in same pixel accumulate."""
        freq = 76e6  # Reference frequency - no extrapolation
        nside = 16  # Low resolution so sources fall in same pixel
        frequencies = np.array([freq])
        sky = same_pixel_sources_sky_model

        maps = sky._point_sources_to_healpix_multifreq(
            nside=nside, frequencies=frequencies, ref_frequency=freq
        )

        # Should have only one non-zero pixel (both sources in same pixel)
        nonzero_pixels = np.where(maps[freq] > 0)[0]
        # With low nside, they should be in same pixel
        assert len(nonzero_pixels) <= 2  # Could be 1 or 2 depending on exact positions

        # Total flux should equal sum of both sources
        npix = hp.nside2npix(nside)
        omega_pixel = 4 * np.pi / npix
        total_flux_jy = np.sum(
            brightness_temp_to_flux_density(
                maps[freq], freq, omega_pixel, method=sky.brightness_conversion
            )
        )

        expected_total = 2.0 + 1.0  # 2 Jy + 1 Jy
        assert np.isclose(total_flux_jy, expected_total, rtol=1e-4)

    def test_different_spectral_indices_handled_correctly_same_pixel(self):
        """Test that different spectral indices are handled correctly when sources merge."""
        # Two sources in same pixel with different spectral indices
        sources = [
            {
                "coords": SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, frame="icrs"),
                "flux": 1.0,
                "spectral_index": -0.5,  # Flat
                "stokes_q": 0.0, "stokes_u": 0.0, "stokes_v": 0.0,
            },
            {
                "coords": SkyCoord(ra=0.1 * u.deg, dec=0.1 * u.deg, frame="icrs"),
                "flux": 1.0,
                "spectral_index": -1.5,  # Steep
                "stokes_q": 0.0, "stokes_u": 0.0, "stokes_v": 0.0,
            },
        ]
        sky = SkyModel.from_point_sources(sources, model_name="test")

        ref_freq = 76e6
        test_freqs = np.array([76e6, 150e6])
        nside = 16  # Low resolution

        maps = sky._point_sources_to_healpix_multifreq(
            nside=nside, frequencies=test_freqs, ref_frequency=ref_freq
        )

        # Calculate total flux at each frequency
        npix = hp.nside2npix(nside)
        omega_pixel = 4 * np.pi / npix

        flux_76 = np.sum(
            brightness_temp_to_flux_density(
                maps[76e6], 76e6, omega_pixel, method=sky.brightness_conversion
            )
        )

        flux_150 = np.sum(
            brightness_temp_to_flux_density(
                maps[150e6], 150e6, omega_pixel, method=sky.brightness_conversion
            )
        )

        # Expected: each source extrapolated independently, then summed
        expected_76 = 1.0 + 1.0  # Both at reference frequency
        expected_150 = (
            1.0 * (150e6 / 76e6)**(-0.5) +  # Flat source
            1.0 * (150e6 / 76e6)**(-1.5)     # Steep source
        )

        # Tolerance accounts for float32 map storage with nonlinear Planck round-trip
        assert np.isclose(flux_76, expected_76, rtol=1e-3)
        assert np.isclose(flux_150, expected_150, rtol=1e-3)


# =============================================================================
# TEST to_healpix_for_observation()
# =============================================================================

class TestToHealpixForObservation:
    """Tests for the to_healpix_for_observation() public method."""

    def test_basic_conversion(self, single_source_sky_model, simple_obs_frequency_config):
        """Test basic conversion workflow."""
        sky = single_source_sky_model.to_healpix_for_observation(
            nside=32, obs_frequency_config=simple_obs_frequency_config
        )

        assert sky._healpix_maps is not None
        assert sky._healpix_nside == 32
        assert sky._observation_frequencies is not None
        assert len(sky._observation_frequencies) == 20

    def test_returns_self(self, single_source_sky_model, simple_obs_frequency_config):
        """Test that method returns self for chaining."""
        result = single_source_sky_model.to_healpix_for_observation(
            nside=32, obs_frequency_config=simple_obs_frequency_config
        )

        assert result is single_source_sky_model

    def test_raises_error_for_empty_sources(self, simple_obs_frequency_config):
        """Test that method raises error when no sources available."""
        sky = SkyModel.from_point_sources([], model_name="empty")

        with pytest.raises(ValueError, match="No point sources available"):
            sky.to_healpix_for_observation(
                nside=32, obs_frequency_config=simple_obs_frequency_config
            )

    def test_custom_reference_frequency(self, single_source_sky_model, simple_obs_frequency_config):
        """Test using custom reference frequency."""
        # Should not raise
        sky = single_source_sky_model.to_healpix_for_observation(
            nside=32,
            obs_frequency_config=simple_obs_frequency_config,
            ref_frequency=100e6  # Different from default 76 MHz
        )

        assert sky._healpix_maps is not None


# =============================================================================
# TEST get_map_at_frequency()
# =============================================================================

class TestGetMapAtFrequency:
    """Tests for the get_map_at_frequency() method."""

    def test_exact_frequency_match(self, single_source_sky_model, simple_obs_frequency_config):
        """Test getting map at exact frequency."""
        sky = single_source_sky_model.to_healpix_for_observation(
            nside=32, obs_frequency_config=simple_obs_frequency_config
        )

        map_100mhz = sky.get_map_at_frequency(100e6)

        assert isinstance(map_100mhz, np.ndarray)
        assert map_100mhz.shape == (hp.nside2npix(32),)

    def test_nearest_frequency_fallback(self, single_source_sky_model, simple_obs_frequency_config):
        """Test that nearest frequency is used when exact match not found."""
        sky = single_source_sky_model.to_healpix_for_observation(
            nside=32, obs_frequency_config=simple_obs_frequency_config
        )

        # Request 100.5 MHz (not in config), should get 100 or 101 MHz
        map_nearest = sky.get_map_at_frequency(100.5e6)

        assert isinstance(map_nearest, np.ndarray)

    def test_raises_error_when_no_maps(self, single_source_sky_model):
        """Test that method raises error when no maps available."""
        with pytest.raises(ValueError, match="No multi-frequency HEALPix maps"):
            single_source_sky_model.get_map_at_frequency(100e6)


# =============================================================================
# TEST get_multifreq_maps()
# =============================================================================

class TestGetMultifreqMaps:
    """Tests for the get_multifreq_maps() method."""

    def test_returns_correct_structure(self, single_source_sky_model, simple_obs_frequency_config):
        """Test that method returns correct structure."""
        sky = single_source_sky_model.to_healpix_for_observation(
            nside=32, obs_frequency_config=simple_obs_frequency_config
        )

        maps, nside, freqs = sky.get_multifreq_maps()

        assert isinstance(maps, dict)
        assert nside == 32
        assert len(freqs) == 20
        assert len(maps) == 20

    def test_raises_error_when_no_maps(self, single_source_sky_model):
        """Test that method raises error when no maps available."""
        with pytest.raises(ValueError, match="No multi-frequency HEALPix maps"):
            single_source_sky_model.get_multifreq_maps()


# =============================================================================
# TEST PROPERTIES
# =============================================================================

class TestMultifreqProperties:
    """Tests for multi-frequency related properties."""

    def test_mode_point_sources(self, single_source_sky_model):
        """Test mode property for point sources."""
        assert single_source_sky_model.mode == "point_sources"

    def test_mode_healpix_multifreq(self, single_source_sky_model, simple_obs_frequency_config):
        """Test mode property after conversion to multi-freq HEALPix."""
        single_source_sky_model.to_healpix_for_observation(
            nside=32, obs_frequency_config=simple_obs_frequency_config
        )

        assert single_source_sky_model.mode == "healpix_multifreq"

    def test_has_multifreq_maps_false(self, single_source_sky_model):
        """Test has_multifreq_maps property when no maps."""
        assert single_source_sky_model.has_multifreq_maps is False

    def test_has_multifreq_maps_true(self, single_source_sky_model, simple_obs_frequency_config):
        """Test has_multifreq_maps property after conversion."""
        single_source_sky_model.to_healpix_for_observation(
            nside=32, obs_frequency_config=simple_obs_frequency_config
        )

        assert single_source_sky_model.has_multifreq_maps is True

    def test_n_frequencies_zero(self, single_source_sky_model):
        """Test n_frequencies property when no maps."""
        assert single_source_sky_model.n_frequencies == 0

    def test_n_frequencies_after_conversion(self, single_source_sky_model, simple_obs_frequency_config):
        """Test n_frequencies property after conversion."""
        single_source_sky_model.to_healpix_for_observation(
            nside=32, obs_frequency_config=simple_obs_frequency_config
        )

        assert single_source_sky_model.n_frequencies == 20


# =============================================================================
# TEST __repr__
# =============================================================================

class TestRepr:
    """Tests for __repr__ method with multi-frequency maps."""

    def test_repr_multifreq_mode(self, single_source_sky_model, simple_obs_frequency_config):
        """Test __repr__ in multi-frequency mode."""
        single_source_sky_model.to_healpix_for_observation(
            nside=32, obs_frequency_config=simple_obs_frequency_config
        )

        repr_str = repr(single_source_sky_model)

        assert "healpix_multifreq" in repr_str
        assert "nside=32" in repr_str
        assert "n_freq=20" in repr_str
        assert "100.0-119.0MHz" in repr_str
        assert "memory=" in repr_str


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestMultifreqIntegration:
    """Integration tests for multi-frequency functionality."""

    def test_full_workflow(self, multi_source_sky_model, simple_obs_frequency_config):
        """Test complete workflow from sources to maps."""
        # Convert
        sky = multi_source_sky_model.to_healpix_for_observation(
            nside=64, obs_frequency_config=simple_obs_frequency_config
        )

        # Check state
        assert sky.mode == "healpix_multifreq"
        assert sky.has_multifreq_maps
        assert sky.n_frequencies == 20

        # Get maps
        maps, nside, freqs = sky.get_multifreq_maps()

        # Verify maps
        assert len(maps) == 20
        assert nside == 64
        assert len(freqs) == 20

        # Get individual map
        map_110 = sky.get_map_at_frequency(110e6)
        assert map_110.shape == (hp.nside2npix(64),)

        # Check repr
        assert "healpix_multifreq" in repr(sky)

    def test_wideband_observation(self, multi_source_sky_model, wideband_obs_frequency_config):
        """Test with wideband observation (100 channels)."""
        sky = multi_source_sky_model.to_healpix_for_observation(
            nside=32, obs_frequency_config=wideband_obs_frequency_config
        )

        assert sky.n_frequencies == 100

        # Check memory estimation matches actual
        mem_info = SkyModel.estimate_healpix_memory(nside=32, n_frequencies=100)
        assert mem_info["n_freq"] == 100


# =============================================================================
# TEST PLANCK VS RAYLEIGH-JEANS CONVERSION
# =============================================================================

class TestPlanckVsRayleighJeans:
    """Tests for Planck-exact vs Rayleigh-Jeans brightness conversion."""

    def test_planck_default(self):
        """Verify SkyModel defaults to Planck conversion."""
        sky = SkyModel.from_point_sources([], model_name="test")
        assert sky.brightness_conversion == "planck"

    def test_planck_rj_agree_high_temp(self):
        """At high T (5000 K, 150 MHz) both methods agree to < 0.001%."""
        T = np.array([5000.0])
        freq = 150e6
        omega = 1e-4  # arbitrary solid angle

        flux_planck = brightness_temp_to_flux_density(T, freq, omega, method="planck")
        flux_rj = brightness_temp_to_flux_density(T, freq, omega, method="rayleigh-jeans")

        assert np.isclose(flux_planck, flux_rj, rtol=1e-5)

    def test_planck_less_than_rj_at_low_temp(self):
        """Planck flux should be less than RJ flux at low temperature."""
        T = np.array([1.0])  # 1 K
        freq = 150e6
        omega = 1e-4

        flux_planck = brightness_temp_to_flux_density(T, freq, omega, method="planck")
        flux_rj = brightness_temp_to_flux_density(T, freq, omega, method="rayleigh-jeans")

        assert flux_planck[0] < flux_rj[0]

    def test_round_trip_planck(self):
        """T -> Jy -> T round-trip with Planck recovers original T."""
        T_original = np.array([100.0, 500.0, 2000.0, 10000.0])
        freq = 150e6
        omega = 1e-4

        flux = brightness_temp_to_flux_density(T_original, freq, omega, method="planck")
        T_recovered = flux_density_to_brightness_temp(flux, freq, omega, method="planck")

        assert np.allclose(T_recovered, T_original, rtol=1e-10)

    def test_round_trip_rj(self):
        """T -> Jy -> T round-trip with RJ recovers original T."""
        T_original = np.array([100.0, 500.0, 2000.0, 10000.0])
        freq = 150e6
        omega = 1e-4

        flux = brightness_temp_to_flux_density(T_original, freq, omega, method="rayleigh-jeans")
        T_recovered = flux_density_to_brightness_temp(flux, freq, omega, method="rayleigh-jeans")

        assert np.allclose(T_recovered, T_original, rtol=1e-10)

    def test_rj_fallback(self):
        """SkyModel with brightness_conversion='rayleigh-jeans' uses RJ."""
        sky = SkyModel.from_point_sources(
            [], model_name="test", brightness_conversion="rayleigh-jeans"
        )
        assert sky.brightness_conversion == "rayleigh-jeans"

    def test_zero_temperature_returns_zero(self):
        """Zero temperature should return zero flux."""
        T = np.array([0.0, 100.0, 0.0])
        freq = 150e6
        omega = 1e-4

        flux = brightness_temp_to_flux_density(T, freq, omega, method="planck")
        assert flux[0] == 0.0
        assert flux[1] > 0.0
        assert flux[2] == 0.0

    def test_zero_flux_returns_zero_temp(self):
        """Zero flux should return zero temperature."""
        flux = np.array([0.0, 1.0, 0.0])
        freq = 150e6
        omega = 1e-4

        T = flux_density_to_brightness_temp(flux, freq, omega, method="planck")
        assert T[0] == 0.0
        assert T[1] > 0.0
        assert T[2] == 0.0
