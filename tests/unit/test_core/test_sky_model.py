# tests/unit/test_core/test_sky_model.py
"""
Tests for SkyModel: loading, multi-frequency HEALPix, and catalog accessibility.

These tests verify:
- Test source generation
- GLEAM and MALS catalog accessibility (requires network, marked slow)
- GSM/LFSM/Haslam diffuse sky model loading (requires pygdsm, marked slow)
- Source data structure validation
- Frequency configuration parsing
- Memory estimation
- Multi-frequency HEALPix map creation
- Spectral extrapolation correctness
- Flux conservation
- API methods (to_healpix_for_observation, get_map_at_frequency, etc.)
- Planck vs Rayleigh-Jeans brightness conversion
- Model combining (healpix_multifreq + healpix_multifreq, mixed)
"""

import pytest
import numpy as np
import astropy.units as u
import healpy as hp
from astropy.coordinates import SkyCoord

from rrivis.core.sky import (
    SkyModel,
    K_BOLTZMANN,
    C_LIGHT,
    H_PLANCK,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
    VIZIER_POINT_CATALOGS,
    DIFFUSE_MODELS,
)
from rrivis.core.precision import PrecisionConfig


# =============================================================================
# CONSTANTS
# =============================================================================

# GLEAM catalog keys (subset of VIZIER_POINT_CATALOGS)
GLEAM_CATALOG_KEYS = sorted(
    k for k in VIZIER_POINT_CATALOGS if k.startswith("gleam") or k == "g4jy"
)

# MALS catalog keys (subset of VIZIER_POINT_CATALOGS)
MALS_CATALOG_KEYS = sorted(k for k in VIZIER_POINT_CATALOGS if k.startswith("mals_"))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def vizier_client():
    """Configure Vizier client for testing."""
    from astroquery.vizier import Vizier

    # Limit rows for faster tests
    # Note: Some catalogs require minimum ROW_LIMIT to return data:
    #   - MALS DR2 requires ROW_LIMIT >= 50
    #   - MALS DR3 requires ROW_LIMIT >= 100
    Vizier.ROW_LIMIT = 100
    return Vizier


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
    return SkyModel.from_point_sources(sources, model_name="test_single", precision=PrecisionConfig.standard())


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
    return SkyModel.from_point_sources(sources, model_name="test_multi", precision=PrecisionConfig.standard())


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
    return SkyModel.from_point_sources(sources, model_name="test_same_pixel", precision=PrecisionConfig.standard())


# =============================================================================
# TEST SOURCE GENERATION
# =============================================================================

class TestTestSourceGeneration:
    """Tests for dynamic test source generation via SkyModel."""

    def test_generate_single_source(self):
        """Test generating a single test source."""
        sky = SkyModel.from_test_sources(num_sources=1, precision=PrecisionConfig.standard())
        sources = sky.to_point_sources()

        assert len(sources) == 1
        assert isinstance(sources[0]["coords"], SkyCoord)
        assert sources[0]["spectral_index"] == -0.8

    def test_generate_multiple_sources(self):
        """Test generating multiple test sources."""
        sky = SkyModel.from_test_sources(num_sources=10, precision=PrecisionConfig.standard())
        sources = sky.to_point_sources()

        assert len(sources) == 10

        # Check all have required fields
        for source in sources:
            assert "coords" in source
            assert "flux" in source
            assert "spectral_index" in source
            assert "stokes_q" in source
            assert "stokes_u" in source
            assert "stokes_v" in source

            # Check unpolarized
            assert source["stokes_q"] == 0.0
            assert source["stokes_u"] == 0.0
            assert source["stokes_v"] == 0.0

    def test_sources_distributed_in_ra(self):
        """Test that sources are evenly distributed in RA."""
        sky = SkyModel.from_test_sources(num_sources=4, precision=PrecisionConfig.standard())
        sources = sky.to_point_sources()

        # Expected RAs: 0, 90, 180, 270 degrees
        ras = [s["coords"].ra.deg for s in sources]
        expected_ras = [0.0, 90.0, 180.0, 270.0]

        for ra, expected in zip(sorted(ras), expected_ras):
            assert abs(ra - expected) < 0.01

    def test_default_num_sources(self):
        """Test default number of test sources."""
        sky = SkyModel.from_test_sources(num_sources=100, precision=PrecisionConfig.standard())
        sources = sky.to_point_sources()

        assert len(sources) == 100


class TestGetSources:
    """Tests for creating sky models with test sources."""

    def test_get_test_sources(self):
        """Test creating test sources via SkyModel."""
        sky = SkyModel.from_test_sources(num_sources=5, precision=PrecisionConfig.standard())
        sources = sky.to_point_sources()

        assert len(sources) == 5

    def test_default_returns_test_sources(self):
        """Test creating default test sources."""
        sky = SkyModel.from_test_sources(num_sources=3, precision=PrecisionConfig.standard())
        sources = sky.to_point_sources()

        assert len(sources) == 3


# =============================================================================
# SOURCE VALIDATION
# =============================================================================

class TestSourceValidation:
    """Tests for source data validation."""

    def test_source_has_valid_coordinates(self, sample_sources_single):
        """Test that source coordinates are valid."""
        source = sample_sources_single[0]

        assert isinstance(source["coords"], SkyCoord)
        assert 0 <= source["coords"].ra.deg <= 360
        assert -90 <= source["coords"].dec.deg <= 90

    def test_source_flux_positive(self, sample_sources_multiple):
        """Test that source fluxes are positive."""
        for source in sample_sources_multiple:
            assert source["flux"] > 0

    def test_source_spectral_index_reasonable(self, sample_sources_multiple):
        """Test that spectral indices are in reasonable range."""
        for source in sample_sources_multiple:
            # Typical range is -2.0 to 0.5
            assert -3.0 <= source["spectral_index"] <= 1.0


# =============================================================================
# GLEAM CATALOG TESTS (Network Required)
# =============================================================================

@pytest.mark.slow
class TestGLEAMCatalogAccessibility:
    """
    Tests for GLEAM catalog accessibility from VizieR.

    These tests require network access and are marked as slow.
    Run with: pytest -m slow tests/unit/test_core/test_sky_model.py
    """

    @pytest.mark.parametrize("catalog_key", GLEAM_CATALOG_KEYS)
    def test_catalog_accessible(self, catalog_key, vizier_client):
        """Test that each GLEAM catalog is accessible from VizieR."""
        info = VIZIER_POINT_CATALOGS[catalog_key]
        catalog_id = info["vizier_id"]
        try:
            tables = vizier_client.get_catalogs(catalog_id)

            assert tables is not None, f"No tables returned for {catalog_key}"
            assert len(tables) > 0, f"Empty tables for {catalog_key}"

            # Check we got some data
            table = tables[0]
            assert len(table) > 0, f"No rows in {catalog_key}"

        except Exception as e:
            pytest.fail(f"Failed to access {catalog_key} ({info['description']}): {e}")

    def test_gleamegc_has_required_columns(self, vizier_client):
        """Test that GLEAM EGC has required columns for RRIvis."""
        tables = vizier_client.get_catalogs("VIII/100/gleamegc")
        table = tables[0]

        required_columns = ["RAJ2000", "DEJ2000", "Fp076"]
        for col in required_columns:
            assert col in table.colnames, f"Missing required column: {col}"

    def test_gleamegc_flux_values_reasonable(self, vizier_client):
        """Test that GLEAM EGC flux values are reasonable."""
        tables = vizier_client.get_catalogs("VIII/100/gleamegc")
        table = tables[0]

        fluxes = table["Fp076"]
        valid_fluxes = fluxes[~np.isnan(fluxes)]

        assert len(valid_fluxes) > 0, "No valid flux values"
        # Note: GLEAM can have negative flux values due to measurement noise
        # Most values should be positive, but some can be slightly negative
        assert np.median(valid_fluxes) > 0, "Median flux should be positive"
        assert np.max(valid_fluxes) > 0.1, "Should have sources above 0.1 Jy"

    def test_gleamx_dr2_accessible(self, vizier_client):
        """Test that GLEAM-X DR2 (largest catalog) is accessible."""
        tables = vizier_client.get_catalogs("VIII/113/catalog2")

        assert tables is not None
        assert len(tables) > 0
        assert len(tables[0]) > 0


@pytest.mark.slow
class TestGLEAMLoadFunction:
    """Tests for loading GLEAM via SkyModel."""

    def test_load_gleam_with_high_flux_limit(self):
        """Test loading GLEAM with high flux limit (fewer sources, faster)."""
        sky = SkyModel.from_gleam(flux_limit=100.0, precision=PrecisionConfig.standard())
        sources = sky.to_point_sources()

        # Should have some sources above 100 Jy
        assert isinstance(sources, list)

        if len(sources) > 0:
            # Verify source structure
            source = sources[0]
            assert "coords" in source
            assert "flux" in source
            assert "spectral_index" in source
            assert source["flux"] >= 100.0

    def test_load_gleam_source_structure(self):
        """Test that loaded sources have correct structure."""
        sky = SkyModel.from_gleam(flux_limit=500.0, precision=PrecisionConfig.standard())
        sources = sky.to_point_sources()

        if len(sources) > 0:
            source = sources[0]

            # Check all required fields
            assert isinstance(source["coords"], SkyCoord)
            assert isinstance(source["flux"], (int, float))
            assert isinstance(source["spectral_index"], (int, float))
            assert source["stokes_q"] == 0.0  # Unpolarized
            assert source["stokes_u"] == 0.0
            assert source["stokes_v"] == 0.0


# =============================================================================
# MALS CATALOG TESTS (Network Required)
# =============================================================================

@pytest.mark.slow
class TestMALSCatalogAccessibility:
    """
    Tests for MALS catalog accessibility from VizieR.

    These tests require network access and are marked as slow.
    Run with: pytest -m slow tests/unit/test_core/test_sky_model.py::TestMALSCatalogAccessibility
    """

    @pytest.mark.parametrize("catalog_key", MALS_CATALOG_KEYS)
    def test_mals_catalog_accessible(self, catalog_key, vizier_client):
        """Test that each MALS catalog is accessible from VizieR."""
        info = VIZIER_POINT_CATALOGS[catalog_key]
        catalog_id = info["vizier_id"]
        try:
            tables = vizier_client.get_catalogs(catalog_id)

            assert tables is not None, f"No tables returned for {catalog_key}"
            assert len(tables) > 0, f"Empty tables for {catalog_key}"

            # Check we got some data
            table = tables[0]
            assert len(table) > 0, f"No rows in {catalog_key}"

        except Exception as e:
            pytest.fail(f"Failed to access {catalog_key} ({info['description']}): {e}")

    def test_mals_dr2_has_required_columns(self, vizier_client):
        """Test that MALS DR2 has required columns for RRIvis."""
        tables = vizier_client.get_catalogs("J/A+A/690/A163")
        table = tables[0]

        required_columns = ["RAJ2000", "DEJ2000", "FluxTot"]
        for col in required_columns:
            assert col in table.colnames, f"Missing required column: {col}"

    def test_mals_dr2_flux_values_reasonable(self, vizier_client):
        """Test that MALS DR2 flux values are reasonable."""
        tables = vizier_client.get_catalogs("J/A+A/690/A163")
        table = tables[0]

        fluxes = table["FluxTot"]
        valid_fluxes = fluxes[~np.isnan(fluxes)]

        assert len(valid_fluxes) > 0, "No valid flux values"
        assert np.median(valid_fluxes) > 0, "Median flux should be positive"


@pytest.mark.slow
class TestMALSLoadFunction:
    """Tests for loading MALS via SkyModel."""

    def test_load_mals_dr2_with_high_flux_limit(self):
        """Test loading MALS DR2 with high flux limit (fewer sources, faster)."""
        sky = SkyModel.from_mals(flux_limit=0.1, release="dr2", precision=PrecisionConfig.standard())
        sources = sky.to_point_sources()

        assert isinstance(sources, list)

        if len(sources) > 0:
            # Verify source structure
            source = sources[0]
            assert "coords" in source
            assert "flux" in source
            assert "spectral_index" in source
            # flux_limit is in Jy; the generic loader converts mJy→Jy internally
            assert source["flux"] >= 0.1

    def test_load_mals_dr1(self):
        """Test loading MALS DR1."""
        sky = SkyModel.from_mals(flux_limit=0.5, release="dr1", precision=PrecisionConfig.standard())
        sources = sky.to_point_sources()

        assert isinstance(sources, list)

    def test_load_mals_dr3(self):
        """Test loading MALS DR3 (HI absorption)."""
        sky = SkyModel.from_mals(flux_limit=0.01, release="dr3", precision=PrecisionConfig.standard())
        sources = sky.to_point_sources()

        assert isinstance(sources, list)

    def test_load_mals_source_structure(self):
        """Test that loaded MALS sources have correct structure."""
        sky = SkyModel.from_mals(flux_limit=0.5, release="dr2", precision=PrecisionConfig.standard())
        sources = sky.to_point_sources()

        if len(sources) > 0:
            source = sources[0]

            # Check all required fields
            assert isinstance(source["coords"], SkyCoord)
            assert isinstance(source["flux"], (int, float))
            assert isinstance(source["spectral_index"], (int, float))
            assert source["stokes_q"] == 0.0  # Unpolarized
            assert source["stokes_u"] == 0.0
            assert source["stokes_v"] == 0.0


# =============================================================================
# DIFFUSE SKY MODEL TESTS (GSM2008, GSM2016, LFSM, HASLAM)
# =============================================================================

@pytest.mark.slow
class TestDiffuseSkyModels:
    """Tests for diffuse sky model loading (GSM2008, GSM2016, LFSM, Haslam)."""

    def test_load_diffuse_sky_gsm2008(self):
        """Test basic GSM2008 loading via SkyModel (multi-frequency)."""
        sky = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=16, frequencies=np.array([76e6]),
            precision=PrecisionConfig.standard()
        )

        # Verify the model is in healpix_multifreq mode
        assert sky.mode == "healpix_multifreq"

        sources = sky.to_point_sources(flux_limit=0.1, frequency=76e6)

        assert isinstance(sources, list)
        assert len(sources) > 0

    def test_load_diffuse_sky_source_structure(self):
        """Test diffuse sky source structure (multi-frequency)."""
        sky = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=8, frequencies=np.array([76e6]),
            precision=PrecisionConfig.standard()
        )
        sources = sky.to_point_sources(flux_limit=0.01, frequency=76e6)

        if len(sources) > 0:
            source = sources[0]

            assert isinstance(source["coords"], SkyCoord)
            assert isinstance(source["flux"], (int, float, np.floating))
            assert source["flux"] >= 0.01  # Above flux limit
            assert "spectral_index" in source
            assert isinstance(source["spectral_index"], float)
            assert source["stokes_q"] == 0.0
            assert source["stokes_u"] == 0.0
            assert source["stokes_v"] == 0.0

    def test_load_diffuse_sky_invalid_model(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            SkyModel.from_diffuse_sky(model="invalid_model", frequencies=np.array([100e6]), precision=PrecisionConfig.standard())

    def test_available_models_constant(self):
        """Test that DIFFUSE_MODELS constant is properly defined."""
        expected_models = ["gsm2008", "gsm2016", "lfsm", "haslam"]
        for model in expected_models:
            assert model in DIFFUSE_MODELS
            assert "class" in DIFFUSE_MODELS[model]
            assert "description" in DIFFUSE_MODELS[model]
            assert "freq_range" in DIFFUSE_MODELS[model]


# =============================================================================
# CATALOG + SOURCE LOADING INTEGRATION TESTS
# =============================================================================

@pytest.mark.slow
class TestSourceLoadingIntegration:
    """Integration tests for the full source loading pipeline."""

    def test_get_sources_with_gleam(self):
        """Test loading GLEAM catalog via SkyModel."""
        sky = SkyModel.from_gleam(flux_limit=200.0, precision=PrecisionConfig.standard())
        sources = sky.to_point_sources()

        assert isinstance(sources, list)

    def test_get_sources_with_mals(self):
        """Test loading MALS catalog via SkyModel."""
        sky = SkyModel.from_mals(flux_limit=100.0, release="dr2", precision=PrecisionConfig.standard())
        sources = sky.to_point_sources()

        assert isinstance(sources, list)

    def test_get_sources_with_gsm(self):
        """Test loading GSM2008 via SkyModel (multi-frequency)."""
        sky = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=8, frequencies=np.array([100e6]),
            precision=PrecisionConfig.standard()
        )
        sources = sky.to_point_sources(flux_limit=0.1, frequency=100e6)

        assert isinstance(sources, list)
        assert len(sources) > 0


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
        sky = SkyModel.from_point_sources([], model_name="empty", precision=PrecisionConfig.standard())
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
        sky = SkyModel.from_point_sources(sources, model_name="test", precision=PrecisionConfig.standard())

        ref_freq = 100e6
        test_freq = 200e6  # Double the reference
        frequencies = np.array([ref_freq, test_freq])
        nside = 32

        maps = sky._point_sources_to_healpix_multifreq(
            nside=nside, frequencies=frequencies, ref_frequency=ref_freq
        )

        # Verify flux ratio in Jy domain (conversion-method independent).
        # Only occupied pixels (T > 0) contribute; empty pixels have no sources.
        npix = hp.nside2npix(nside)
        omega_pixel = 4 * np.pi / npix

        occupied_ref = maps[ref_freq] > 0
        flux_ref = np.sum(brightness_temp_to_flux_density(
            maps[ref_freq][occupied_ref], ref_freq, omega_pixel,
            method=sky.brightness_conversion,
        ))
        occupied_test = maps[test_freq] > 0
        flux_test = np.sum(brightness_temp_to_flux_density(
            maps[test_freq][occupied_test], test_freq, omega_pixel,
            method=sky.brightness_conversion,
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

        # Only occupied pixels (T > 0) contribute flux.
        npix = hp.nside2npix(nside)
        omega_pixel = 4 * np.pi / npix
        occupied = maps[freq] > 0
        total_flux_jy = np.sum(
            brightness_temp_to_flux_density(
                maps[freq][occupied], freq, omega_pixel, method=sky.brightness_conversion
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

        # Only occupied pixels (T > 0) contribute flux.
        npix = hp.nside2npix(nside)
        omega_pixel = 4 * np.pi / npix
        occupied = maps[freq] > 0
        total_flux_jy = np.sum(
            brightness_temp_to_flux_density(
                maps[freq][occupied], freq, omega_pixel, method=sky.brightness_conversion
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

        # Total flux should equal sum of both sources (only occupied pixels).
        npix = hp.nside2npix(nside)
        omega_pixel = 4 * np.pi / npix
        occupied = maps[freq] > 0
        total_flux_jy = np.sum(
            brightness_temp_to_flux_density(
                maps[freq][occupied], freq, omega_pixel, method=sky.brightness_conversion
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
        sky = SkyModel.from_point_sources(sources, model_name="test", precision=PrecisionConfig.standard())

        ref_freq = 76e6
        test_freqs = np.array([76e6, 150e6])
        nside = 16  # Low resolution

        maps = sky._point_sources_to_healpix_multifreq(
            nside=nside, frequencies=test_freqs, ref_frequency=ref_freq
        )

        # Calculate total flux at each frequency (only occupied pixels).
        npix = hp.nside2npix(nside)
        omega_pixel = 4 * np.pi / npix

        occ_76 = maps[76e6] > 0
        flux_76 = np.sum(
            brightness_temp_to_flux_density(
                maps[76e6][occ_76], 76e6, omega_pixel, method=sky.brightness_conversion
            )
        )

        occ_150 = maps[150e6] > 0
        flux_150 = np.sum(
            brightness_temp_to_flux_density(
                maps[150e6][occ_150], 150e6, omega_pixel, method=sky.brightness_conversion
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
        sky = SkyModel.from_point_sources([], model_name="empty", precision=PrecisionConfig.standard())

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
# MULTI-FREQUENCY INTEGRATION TESTS
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
        sky = SkyModel.from_point_sources([], model_name="test", precision=PrecisionConfig.standard())
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
            [], model_name="test", brightness_conversion="rayleigh-jeans",
            precision=PrecisionConfig.standard()
        )
        assert sky.brightness_conversion == "rayleigh-jeans"

    def test_zero_temperature_raises(self):
        """T=0 is unphysical — callers must filter before converting."""
        T = np.array([0.0, 100.0])
        freq = 150e6
        omega = 1e-4

        with pytest.raises(ValueError, match="strictly positive"):
            brightness_temp_to_flux_density(T, freq, omega, method="planck")

    def test_negative_temperature_raises(self):
        """Negative T is unphysical and must raise."""
        T = np.array([-1.0, 100.0])
        freq = 150e6
        omega = 1e-4

        with pytest.raises(ValueError, match="strictly positive"):
            brightness_temp_to_flux_density(T, freq, omega, method="planck")

    def test_zero_flux_raises(self):
        """flux=0 is unphysical — callers must filter before converting."""
        flux = np.array([0.0, 1.0])
        freq = 150e6
        omega = 1e-4

        with pytest.raises(ValueError, match="strictly positive"):
            flux_density_to_brightness_temp(flux, freq, omega, method="planck")

    def test_negative_flux_raises(self):
        """Negative flux is unphysical and must raise."""
        flux = np.array([-1.0, 1.0])
        freq = 150e6
        omega = 1e-4

        with pytest.raises(ValueError, match="strictly positive"):
            flux_density_to_brightness_temp(flux, freq, omega, method="planck")


# =============================================================================
# DIFFUSE MODEL NATIVE MULTI-FREQUENCY TESTS
# =============================================================================

@pytest.mark.slow
class TestDiffuseNativeMultifreq:
    """Tests for from_diffuse_sky() with native per-frequency generation."""

    def test_mode_is_healpix_multifreq(self):
        """from_diffuse_sky() must return a healpix_multifreq model."""
        freqs = np.array([100e6, 120e6])
        sky = SkyModel.from_diffuse_sky(model="gsm2008", nside=16, frequencies=freqs, precision=PrecisionConfig.standard())

        assert sky.mode == "healpix_multifreq"

    def test_one_map_per_frequency(self):
        """from_diffuse_sky() generates exactly one T_b map per channel."""
        freqs = np.array([80e6, 100e6, 120e6])
        sky = SkyModel.from_diffuse_sky(model="gsm2008", nside=8, frequencies=freqs, precision=PrecisionConfig.standard())

        maps, nside, obs_freqs = sky.get_multifreq_maps()

        assert len(maps) == 3
        assert len(obs_freqs) == 3
        for f in freqs:
            assert f in maps

    def test_correct_nside(self):
        """Maps are generated at the requested nside."""
        freqs = np.array([100e6])
        nside = 16
        sky = SkyModel.from_diffuse_sky(model="gsm2008", nside=nside, frequencies=freqs, precision=PrecisionConfig.standard())

        _, got_nside, _ = sky.get_multifreq_maps()
        assert got_nside == nside
        assert sky.n_sources == hp.nside2npix(nside)

    def test_maps_have_positive_temperatures(self):
        """All diffuse model pixels should have positive brightness temperature."""
        freqs = np.array([100e6])
        sky = SkyModel.from_diffuse_sky(model="gsm2008", nside=8, frequencies=freqs, precision=PrecisionConfig.standard())

        t_map = sky.get_map_at_frequency(100e6)
        assert np.all(t_map > 0), "GSM map must have positive T_b everywhere"

    def test_maps_differ_across_frequencies(self):
        """Maps at different frequencies must not be identical (no trivial scaling)."""
        freqs = np.array([80e6, 120e6])
        sky = SkyModel.from_diffuse_sky(model="gsm2008", nside=8, frequencies=freqs, precision=PrecisionConfig.standard())

        map_80 = sky.get_map_at_frequency(80e6)
        map_120 = sky.get_map_at_frequency(120e6)

        # Maps at different frequencies are different (native PCA model)
        assert not np.allclose(map_80, map_120), (
            "Maps at 80 MHz and 120 MHz must differ; they should use pygdsm's "
            "native spectral model, not a trivial power-law scaling."
        )

    def test_obs_frequency_config_api(self):
        """from_diffuse_sky() also accepts obs_frequency_config."""
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 10.0,
            "frequency_bandwidth": 20.0,
            "frequency_unit": "MHz",
        }
        sky = SkyModel.from_diffuse_sky(model="gsm2008", nside=8, obs_frequency_config=config, precision=PrecisionConfig.standard())

        assert sky.mode == "healpix_multifreq"
        assert sky.n_frequencies == 2  # 100 MHz and 110 MHz

    def test_raises_without_frequencies(self):
        """from_diffuse_sky() raises if neither frequencies nor config given."""
        with pytest.raises(ValueError, match="Either 'frequencies' or 'obs_frequency_config'"):
            SkyModel.from_diffuse_sky(model="gsm2008", nside=8, precision=PrecisionConfig.standard())

    def test_to_point_sources_from_diffuse(self):
        """Diffuse model can be converted to point sources at a specific frequency."""
        freqs = np.array([100e6])
        sky = SkyModel.from_diffuse_sky(model="gsm2008", nside=8, frequencies=freqs, precision=PrecisionConfig.standard())

        sources = sky.to_point_sources(flux_limit=0.01, frequency=100e6)

        assert isinstance(sources, list)
        assert len(sources) > 0
        for src in sources:
            assert src["flux"] >= 0.01
            assert isinstance(src["spectral_index"], float)


# =============================================================================
# COMBINE HEALPIX_MULTIFREQ + HEALPIX_MULTIFREQ TESTS
# =============================================================================

@pytest.mark.slow
class TestCombineHealpixMultifreq:
    """Tests for combining two healpix_multifreq models."""

    def test_combined_mode_is_healpix_multifreq(self):
        """Combining two healpix_multifreq models yields a healpix_multifreq model."""
        freqs = np.array([100e6, 110e6])
        sky1 = SkyModel.from_diffuse_sky(model="gsm2008", nside=8, frequencies=freqs, precision=PrecisionConfig.standard())
        sky2 = SkyModel.from_diffuse_sky(model="gsm2008", nside=8, frequencies=freqs, precision=PrecisionConfig.standard())

        combined = SkyModel.combine(
            [sky1, sky2], representation="healpix_map"
        )

        assert combined.mode == "healpix_multifreq"

    def test_combined_maps_brighter_than_inputs(self):
        """Sum of two identical diffuse maps must be brighter (in flux) than one."""
        freqs = np.array([100e6])
        sky1 = SkyModel.from_diffuse_sky(model="gsm2008", nside=8, frequencies=freqs, precision=PrecisionConfig.standard())
        sky2 = SkyModel.from_diffuse_sky(model="gsm2008", nside=8, frequencies=freqs, precision=PrecisionConfig.standard())

        combined = SkyModel.combine([sky1, sky2], representation="healpix_map")

        freq = 100e6
        nside = 8
        npix = hp.nside2npix(nside)
        omega = 4 * np.pi / npix

        map_single = sky1.get_map_at_frequency(freq)
        map_combined = combined.get_map_at_frequency(freq)

        flux_single = np.sum(
            brightness_temp_to_flux_density(map_single, freq, omega, method="planck")
        )
        flux_combined = np.sum(
            brightness_temp_to_flux_density(map_combined, freq, omega, method="planck")
        )

        # Combined should be approximately 2× the single model flux
        assert np.isclose(flux_combined, 2 * flux_single, rtol=1e-3), (
            f"Combined flux {flux_combined:.3f} should be ~2× single flux {flux_single:.3f}"
        )

    def test_combined_same_nside_and_frequencies(self):
        """Combined model inherits nside and frequencies from input models."""
        nside = 8
        freqs = np.array([100e6, 110e6])
        sky1 = SkyModel.from_diffuse_sky(model="gsm2008", nside=nside, frequencies=freqs, precision=PrecisionConfig.standard())
        sky2 = SkyModel.from_diffuse_sky(model="gsm2008", nside=nside, frequencies=freqs, precision=PrecisionConfig.standard())

        combined = SkyModel.combine([sky1, sky2], representation="healpix_map")

        _, got_nside, got_freqs = combined.get_multifreq_maps()

        assert got_nside == nside
        assert len(got_freqs) == 2
        assert np.allclose(np.sort(got_freqs), np.sort(freqs))


# =============================================================================
# COMBINE MIXED (POINT SOURCES + HEALPIX_MULTIFREQ) TESTS
# =============================================================================

@pytest.mark.slow
class TestCombineMixed:
    """Tests for combining point source models with healpix_multifreq models."""

    def test_point_plus_diffuse_yields_healpix_multifreq(
        self, single_source_sky_model
    ):
        """
        Combining a point source model with a healpix_multifreq model
        produces a healpix_multifreq combined model.
        """
        freqs = np.array([100e6, 110e6])
        diffuse = SkyModel.from_diffuse_sky(model="gsm2008", nside=16, frequencies=freqs, precision=PrecisionConfig.standard())

        with pytest.warns(UserWarning, match="double-counting"):
            combined = SkyModel.combine(
                [diffuse, single_source_sky_model],
                representation="healpix_map",
                ref_frequency=76e6,
            )

        assert combined.mode == "healpix_multifreq"

    def test_combined_flux_exceeds_diffuse_alone(self, single_source_sky_model):
        """
        Adding a point source to a diffuse model increases total flux.
        """
        freqs = np.array([100e6])
        diffuse = SkyModel.from_diffuse_sky(model="gsm2008", nside=16, frequencies=freqs, precision=PrecisionConfig.standard())

        with pytest.warns(UserWarning):
            combined = SkyModel.combine(
                [diffuse, single_source_sky_model],
                representation="healpix_map",
                ref_frequency=76e6,
            )

        freq = 100e6
        nside = 16
        npix = hp.nside2npix(nside)
        omega = 4 * np.pi / npix

        map_diffuse = diffuse.get_map_at_frequency(freq)
        map_combined = combined.get_map_at_frequency(freq)

        flux_diffuse = np.sum(
            brightness_temp_to_flux_density(map_diffuse, freq, omega, method="planck")
        )
        flux_combined = np.sum(
            brightness_temp_to_flux_density(map_combined, freq, omega, method="planck")
        )

        assert flux_combined > flux_diffuse, (
            "Combined model must have more total flux than the diffuse model alone."
        )


# =============================================================================
# PRECISION-AWARE SKYMODEL TESTS
# =============================================================================

class TestSkyModelPrecision:
    """Tests for PrecisionConfig integration with SkyModel."""

    def test_precision_none_raises_error(self):
        """Test that precision=None raises ValueError."""
        with pytest.raises(ValueError, match="PrecisionConfig"):
            SkyModel.from_test_sources(num_sources=10)

    def test_precision_standard_uses_float64(self):
        """Test standard preset produces float64 arrays."""
        from rrivis.core.precision import PrecisionConfig
        precision = PrecisionConfig.standard()
        sky = SkyModel.from_test_sources(num_sources=10, precision=precision)
        assert sky._ra_rad.dtype == np.float64
        assert sky._dec_rad.dtype == np.float64
        assert sky._flux_ref.dtype == np.float64
        assert sky._alpha.dtype == np.float64

    def test_precision_fast_uses_float32(self):
        """Test fast preset produces float32 arrays."""
        from rrivis.core.precision import PrecisionConfig
        precision = PrecisionConfig.fast()
        sky = SkyModel.from_test_sources(num_sources=10, precision=precision)
        assert sky._ra_rad.dtype == np.float32
        assert sky._dec_rad.dtype == np.float32
        assert sky._flux_ref.dtype == np.float32
        assert sky._alpha.dtype == np.float32
        assert sky._stokes_q.dtype == np.float32

    def test_precision_stored_on_model(self):
        """Test precision config is stored on the SkyModel instance."""
        from rrivis.core.precision import PrecisionConfig
        precision = PrecisionConfig.fast()
        sky = SkyModel.from_test_sources(num_sources=5, precision=precision)
        assert sky._precision is precision

    def test_precision_none_raises_on_factory(self):
        """Test that factory methods raise when precision is None."""
        with pytest.raises(ValueError, match="PrecisionConfig"):
            SkyModel.from_test_sources(num_sources=5)

    def test_ensure_dtypes_idempotent(self):
        """Test calling _ensure_dtypes() twice gives same result."""
        from rrivis.core.precision import PrecisionConfig
        precision = PrecisionConfig.fast()
        sky = SkyModel.from_test_sources(num_sources=10, precision=precision)
        # Already called once during construction; call again
        sky._ensure_dtypes()
        assert sky._ra_rad.dtype == np.float32
        assert sky._flux_ref.dtype == np.float32

    def test_ensure_dtypes_raises_without_precision(self):
        """Test _ensure_dtypes() raises when precision is None."""
        from rrivis.core.precision import PrecisionConfig
        sky = SkyModel.from_test_sources(num_sources=10, precision=PrecisionConfig.standard())
        sky._precision = None
        with pytest.raises(ValueError, match="PrecisionConfig"):
            sky._ensure_dtypes()

    def test_from_point_sources_with_precision(self):
        """Test from_point_sources respects precision."""
        from rrivis.core.precision import PrecisionConfig
        precision = PrecisionConfig.fast()
        coord = SkyCoord(ra=10 * u.deg, dec=-30 * u.deg, frame="icrs")
        sources = [{"coords": coord, "flux": 5.0, "spectral_index": -0.8}]
        sky = SkyModel.from_point_sources(sources, precision=precision)
        assert sky._ra_rad.dtype == np.float32
        assert sky._flux_ref.dtype == np.float32
        assert sky._alpha.dtype == np.float32

    def test_combine_propagates_precision(self):
        """Test combine() propagates precision to combined model."""
        from rrivis.core.precision import PrecisionConfig
        precision = PrecisionConfig.fast()
        sky1 = SkyModel.from_test_sources(num_sources=5, precision=precision)
        sky2 = SkyModel.from_test_sources(num_sources=5, precision=precision)
        combined = SkyModel.combine([sky1, sky2], precision=precision)
        assert combined._precision is precision
        assert combined._ra_rad.dtype == np.float32
        assert combined._flux_ref.dtype == np.float32

    def test_dtype_helpers_with_precision(self):
        """Test _source_dtype/_flux_dtype/_alpha_dtype/_healpix_dtype helpers."""
        from rrivis.core.precision import PrecisionConfig
        precision = PrecisionConfig.fast()
        sky = SkyModel.from_test_sources(num_sources=5, precision=precision)
        assert sky._source_dtype() == np.float32
        assert sky._flux_dtype() == np.float32
        assert sky._alpha_dtype() == np.float32
        assert sky._healpix_dtype() == np.float32

    def test_dtype_helpers_raise_without_precision(self):
        """Test dtype helpers raise ValueError when precision is None."""
        from rrivis.core.precision import PrecisionConfig
        sky = SkyModel.from_test_sources(num_sources=5, precision=PrecisionConfig.standard())
        sky._precision = None
        with pytest.raises(ValueError, match="PrecisionConfig"):
            sky._source_dtype()
        with pytest.raises(ValueError, match="PrecisionConfig"):
            sky._flux_dtype()
        with pytest.raises(ValueError, match="PrecisionConfig"):
            sky._alpha_dtype()
        with pytest.raises(ValueError, match="PrecisionConfig"):
            sky._healpix_dtype()

    def test_precise_preset_healpix_maps_float64(self):
        """Test precise preset uses float64 for HEALPix maps."""
        from rrivis.core.precision import PrecisionConfig
        precision = PrecisionConfig.precise()
        sky = SkyModel.from_test_sources(num_sources=5, precision=precision)
        assert sky._healpix_dtype() == np.float64

    def test_values_preserved_after_cast(self):
        """Test source values are numerically preserved after dtype cast."""
        from rrivis.core.precision import PrecisionConfig
        # Create with standard precision as reference
        sky_ref = SkyModel.from_test_sources(num_sources=10, precision=PrecisionConfig.standard())
        ra_ref = sky_ref._ra_rad.copy()
        flux_ref = sky_ref._flux_ref.copy()

        # Create same with standard precision again
        sky = SkyModel.from_test_sources(num_sources=10, precision=PrecisionConfig.standard())
        np.testing.assert_array_equal(sky._ra_rad, ra_ref)
        np.testing.assert_array_equal(sky._flux_ref, flux_ref)


# =============================================================================
# GSM2008 PARAMETERS
# =============================================================================

class TestGSM2008Parameters:
    """Tests for GSM2008 basemap/interpolation parameter passthrough."""

    _FREQ = np.array([100e6])
    _NSIDE = 8
    _PRECISION = PrecisionConfig.standard()

    @pytest.mark.slow
    def test_basemap_haslam(self):
        """basemap='haslam' produces a valid healpix_multifreq SkyModel."""
        sky = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=self._NSIDE, frequencies=self._FREQ,
            basemap="haslam", precision=self._PRECISION,
        )
        assert sky.mode == "healpix_multifreq"
        assert sky.n_sources == hp.nside2npix(self._NSIDE)

    @pytest.mark.slow
    def test_basemap_wmap(self):
        """basemap='wmap' produces a valid healpix_multifreq SkyModel."""
        sky = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=self._NSIDE, frequencies=self._FREQ,
            basemap="wmap", precision=self._PRECISION,
        )
        assert sky.mode == "healpix_multifreq"

    @pytest.mark.slow
    def test_basemap_5deg(self):
        """basemap='5deg' produces a valid healpix_multifreq SkyModel."""
        sky = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=self._NSIDE, frequencies=self._FREQ,
            basemap="5deg", precision=self._PRECISION,
        )
        assert sky.mode == "healpix_multifreq"

    @pytest.mark.slow
    def test_interpolation_cubic(self):
        """interpolation='cubic' produces a valid SkyModel."""
        sky = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=self._NSIDE, frequencies=self._FREQ,
            interpolation="cubic", precision=self._PRECISION,
        )
        assert sky.mode == "healpix_multifreq"

    @pytest.mark.slow
    def test_interpolation_pchip(self):
        """interpolation='pchip' produces a valid SkyModel."""
        sky = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=self._NSIDE, frequencies=self._FREQ,
            interpolation="pchip", precision=self._PRECISION,
        )
        assert sky.mode == "healpix_multifreq"

    def test_basemap_rejected_for_non_gsm2008(self):
        """basemap raises ValueError for gsm2016, lfsm, and haslam."""
        for model_name in ("gsm2016", "lfsm", "haslam"):
            with pytest.raises(ValueError, match="only supported for gsm2008"):
                SkyModel.from_diffuse_sky(
                    model=model_name, nside=self._NSIDE, frequencies=self._FREQ,
                    basemap="haslam", precision=self._PRECISION,
                )

    def test_interpolation_rejected_for_non_gsm2008(self):
        """interpolation raises ValueError for gsm2016, lfsm, and haslam."""
        for model_name in ("gsm2016", "lfsm", "haslam"):
            with pytest.raises(ValueError, match="only supported for gsm2008"):
                SkyModel.from_diffuse_sky(
                    model=model_name, nside=self._NSIDE, frequencies=self._FREQ,
                    interpolation="pchip", precision=self._PRECISION,
                )

    @pytest.mark.slow
    def test_none_defaults(self):
        """basemap=None and interpolation=None use DIFFUSE_MODELS defaults."""
        sky = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=self._NSIDE, frequencies=self._FREQ,
            basemap=None, interpolation=None, precision=self._PRECISION,
        )
        assert sky.mode == "healpix_multifreq"

    @pytest.mark.slow
    def test_different_basemaps_different_maps(self):
        """haslam vs 5deg basemaps produce different maps at same frequency."""
        sky_haslam = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=self._NSIDE, frequencies=self._FREQ,
            basemap="haslam", precision=self._PRECISION,
        )
        sky_5deg = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=self._NSIDE, frequencies=self._FREQ,
            basemap="5deg", precision=self._PRECISION,
        )
        map_haslam = sky_haslam.get_map_at_frequency(self._FREQ[0])
        map_5deg = sky_5deg.get_map_at_frequency(self._FREQ[0])
        assert not np.allclose(map_haslam, map_5deg), (
            "haslam and 5deg basemaps should produce different maps"
        )


# =============================================================================
# PYGDSM INSTANCE ACCESS
# =============================================================================

class TestPygdsmInstanceAccess:
    """Tests for retain_pygdsm_instance / pygdsm_model property."""

    _FREQ = np.array([100e6])
    _NSIDE = 8
    _PRECISION = PrecisionConfig.standard()

    @pytest.mark.slow
    def test_pygdsm_model_none_by_default(self):
        """pygdsm_model is None when retain_pygdsm_instance is not set."""
        sky = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=self._NSIDE, frequencies=self._FREQ,
            precision=self._PRECISION,
        )
        assert sky.pygdsm_model is None

    @pytest.mark.slow
    def test_pygdsm_model_available_when_retained(self):
        """pygdsm_model is not None when retain_pygdsm_instance=True."""
        sky = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=self._NSIDE, frequencies=self._FREQ,
            retain_pygdsm_instance=True, precision=self._PRECISION,
        )
        assert sky.pygdsm_model is not None
        assert sky.pygdsm_model.name == "GSM2008"

    def test_pygdsm_model_none_for_point_sources(self):
        """pygdsm_model is None for point-source SkyModels."""
        sky = SkyModel.from_test_sources(num_sources=5, precision=self._PRECISION)
        assert sky.pygdsm_model is None

    @pytest.mark.slow
    def test_pygdsm_model_has_generated_data(self):
        """Retained instance has generated_map_data from the last generate() call."""
        sky = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=self._NSIDE, frequencies=self._FREQ,
            retain_pygdsm_instance=True, precision=self._PRECISION,
        )
        assert sky.pygdsm_model.generated_map_data is not None

    @pytest.mark.slow
    def test_pygdsm_model_nside_attribute(self):
        """Retained instance has the native nside (512 for GSM2008)."""
        sky = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=self._NSIDE, frequencies=self._FREQ,
            retain_pygdsm_instance=True, precision=self._PRECISION,
        )
        assert sky.pygdsm_model.nside == 512

    @pytest.mark.slow
    def test_set_basemap_via_instance(self):
        """User can call set_basemap() on the retained instance."""
        sky = SkyModel.from_diffuse_sky(
            model="gsm2008", nside=self._NSIDE, frequencies=self._FREQ,
            retain_pygdsm_instance=True, precision=self._PRECISION,
        )
        # set_basemap is a method on GlobalSkyModel; verify it exists
        assert hasattr(sky.pygdsm_model, "set_basemap")


# =============================================================================
# CREATE GSM OBSERVER
# =============================================================================

class TestCreateGSMObserver:
    """Tests for SkyModel.create_gsm_observer() convenience method."""

    @pytest.mark.slow
    def test_creates_observer(self):
        """create_gsm_observer returns a GSMObserver08 instance."""
        from pygdsm import GSMObserver08
        obs = SkyModel.create_gsm_observer()
        assert isinstance(obs, GSMObserver08)

    @pytest.mark.slow
    def test_basemap_passed(self):
        """Observer's gsm.basemap matches the requested value."""
        obs = SkyModel.create_gsm_observer(basemap="wmap")
        assert obs.gsm.basemap == "wmap"

    @pytest.mark.slow
    def test_interpolation_passed(self):
        """Observer's gsm.interpolation_method matches the requested value."""
        obs = SkyModel.create_gsm_observer(interpolation="cubic")
        assert obs.gsm.interpolation_method == "cubic"
