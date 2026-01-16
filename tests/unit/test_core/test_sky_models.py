# tests/unit/test_core/test_sky_models.py
"""
Tests for sky model loading functionality.

These tests verify:
- Test source generation
- GLEAM catalog accessibility (requires network)
- GSM2008 sky model loading (requires pygdsm)
- Source data structure validation
"""

import pytest
import numpy as np
from astropy.coordinates import SkyCoord


# =============================================================================
# TEST SOURCE GENERATION
# =============================================================================

class TestTestSourceGeneration:
    """Tests for dynamic test source generation."""

    def test_generate_single_source(self):
        """Test generating a single test source."""
        from rrivis.core.source import generate_test_sources

        sources = generate_test_sources(num_sources=1)

        assert len(sources) == 1
        assert isinstance(sources[0]["coords"], SkyCoord)
        assert sources[0]["flux"] == 4
        assert sources[0]["spectral_index"] == -0.8

    def test_generate_multiple_sources(self):
        """Test generating multiple test sources."""
        from rrivis.core.source import generate_test_sources

        sources = generate_test_sources(num_sources=10)

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
        from rrivis.core.source import generate_test_sources

        sources = generate_test_sources(num_sources=4)

        # Expected RAs: 0, 90, 180, 270 degrees
        ras = [s["coords"].ra.deg for s in sources]
        expected_ras = [0.0, 90.0, 180.0, 270.0]

        for ra, expected in zip(sorted(ras), expected_ras):
            assert abs(ra - expected) < 0.01

    def test_default_num_sources(self):
        """Test default number of sources when None passed."""
        from rrivis.core.source import generate_test_sources

        sources = generate_test_sources(num_sources=None)

        assert len(sources) == 3  # Default fallback


class TestGetSources:
    """Tests for the main get_sources() dispatcher."""

    def test_get_test_sources(self):
        """Test getting test sources via get_sources()."""
        from rrivis.core.source import get_sources

        sources, _ = get_sources(use_test_sources=True, num_sources=5)

        assert len(sources) == 5

    def test_default_returns_test_sources(self):
        """Test that default (all flags False) returns test sources."""
        from rrivis.core.source import get_sources

        sources, _ = get_sources(num_sources=3)

        assert len(sources) == 3


# =============================================================================
# GLEAM CATALOG TESTS (Network Required)
# =============================================================================

# All available GLEAM catalogs
GLEAM_CATALOGS = {
    "VIII/100/table1": "GLEAM first year observing parameters (28 rows)",
    "VIII/100/gleamegc": "GLEAM EGC catalog, version 2 (307455 rows)",
    "VIII/102/gleamgal": "GLEAM Galactic plane catalog (22037 rows)",
    "VIII/105/catalog": "G4Jy catalogue (18/01/2020) (1960 rows)",
    "VIII/109/gleamsgp": "GLEAM SGP catalogue (108851 rows)",
    "VIII/110/catalog": "First data release of GLEAM-X (78967 rows)",
    "VIII/113/catalog2": "Second data release of GLEAM-X (624866 rows)",
}


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


@pytest.mark.slow
class TestGLEAMCatalogAccessibility:
    """
    Tests for GLEAM catalog accessibility from VizieR.

    These tests require network access and are marked as slow.
    Run with: pytest -m slow tests/unit/test_core/test_sky_models.py
    """

    @pytest.mark.parametrize("catalog_id,description", list(GLEAM_CATALOGS.items()))
    def test_catalog_accessible(self, catalog_id, description, vizier_client):
        """Test that each GLEAM catalog is accessible from VizieR."""
        try:
            tables = vizier_client.get_catalogs(catalog_id)

            assert tables is not None, f"No tables returned for {catalog_id}"
            assert len(tables) > 0, f"Empty tables for {catalog_id}"

            # Check we got some data
            table = tables[0]
            assert len(table) > 0, f"No rows in {catalog_id}"

        except Exception as e:
            pytest.fail(f"Failed to access {catalog_id} ({description}): {e}")

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
    """Tests for the load_gleam() function."""

    def test_load_gleam_with_high_flux_limit(self):
        """Test loading GLEAM with high flux limit (fewer sources, faster)."""
        from rrivis.core.source import load_gleam

        # High flux limit = fewer sources = faster test
        sources, _ = load_gleam(flux_limit=100.0, gleam_catalogue="VIII/100/gleamegc")

        # Should have some sources above 100 Jy
        assert isinstance(sources, list)

        if len(sources) > 0:
            # Verify source structure
            source = sources[0]
            assert "coords" in source
            assert "flux" in source
            assert "spectral_index" in source
            assert source["flux"] >= 100.0

    def test_load_gleam_invalid_catalog(self):
        """Test that invalid catalog returns empty list."""
        from rrivis.core.source import load_gleam

        sources = load_gleam(flux_limit=1.0, gleam_catalogue="INVALID/CATALOG")

        assert sources == []

    def test_load_gleam_source_structure(self):
        """Test that loaded sources have correct structure."""
        from rrivis.core.source import load_gleam

        sources, _ = load_gleam(flux_limit=500.0, gleam_catalogue="VIII/100/gleamegc")

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
        """Test basic GSM2008 loading via load_diffuse_sky."""
        from rrivis.core.source import load_diffuse_sky

        sources, _ = load_diffuse_sky(
            frequency=76e6,
            nside=16,  # Low resolution for speed
            flux_limit=0.1,
            model="gsm2008",
        )

        assert isinstance(sources, list)
        assert len(sources) > 0

    def test_load_diffuse_sky_source_structure(self):
        """Test diffuse sky source structure."""
        from rrivis.core.source import load_diffuse_sky

        sources, _ = load_diffuse_sky(
            frequency=76e6, nside=8, flux_limit=0.01, model="gsm2008"
        )

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

    def test_load_diffuse_sky_spectral_index_computed(self):
        """Test that spectral indices are computed per-pixel."""
        from rrivis.core.source import load_diffuse_sky

        sources, _ = load_diffuse_sky(
            frequency=100e6,
            nside=8,
            flux_limit=0.0001,
            model="gsm2008",
            compute_spectral_index=True,
        )

        if len(sources) > 0:
            spec_indices = [s["spectral_index"] for s in sources]
            # Should have variation (not all the same)
            assert len(set(spec_indices)) > 1
            # Should be in typical synchrotron range
            assert all(-1.0 < si < 0.0 for si in spec_indices)

    def test_load_diffuse_sky_high_flux_limit_warning(self, caplog):
        """Test that high flux limit produces warning when no sources match."""
        from rrivis.core.source import load_diffuse_sky
        import logging

        with caplog.at_level(logging.WARNING):
            sources, _ = load_diffuse_sky(
                frequency=76e6,
                nside=8,
                flux_limit=1e10,  # Impossibly high
                model="gsm2008",
            )

        # Should have warning about no sources
        assert len(sources) == 0 or "No sources meet the flux limit" in caplog.text

    def test_load_diffuse_sky_invalid_model(self, caplog):
        """Test that invalid model name is handled gracefully."""
        from rrivis.core.source import load_diffuse_sky
        import logging

        with caplog.at_level(logging.ERROR):
            sources, _ = load_diffuse_sky(
                frequency=100e6,
                model="invalid_model",
            )

        assert len(sources) == 0
        assert "Invalid diffuse sky model" in caplog.text

    def test_available_models_constant(self):
        """Test that DIFFUSE_SKY_MODELS constant is properly defined."""
        from rrivis.core.source import DIFFUSE_SKY_MODELS

        expected_models = ["gsm2008", "gsm2016", "lfsm", "haslam"]
        for model in expected_models:
            assert model in DIFFUSE_SKY_MODELS
            assert "class" in DIFFUSE_SKY_MODELS[model]
            assert "description" in DIFFUSE_SKY_MODELS[model]
            assert "freq_range" in DIFFUSE_SKY_MODELS[model]


# =============================================================================
# SOURCE VALIDATION UTILITIES
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
# MALS CATALOG TESTS (Network Required)
# =============================================================================

# MALS catalog identifiers
MALS_RELEASES = {
    "dr1": ("J/ApJS/270/33", "MALS DR1: Stokes I at 1-1.4 GHz"),
    "dr2": ("J/A+A/690/A163", "MALS DR2: Wideband continuum"),
    "dr3": ("J/A+A/698/A120", "MALS DR3: HI 21-cm absorption"),
}


@pytest.mark.slow
class TestMALSCatalogAccessibility:
    """
    Tests for MALS catalog accessibility from VizieR.

    These tests require network access and are marked as slow.
    Run with: pytest -m slow tests/unit/test_core/test_sky_models.py::TestMALSCatalogAccessibility
    """

    @pytest.mark.parametrize("release,info", list(MALS_RELEASES.items()))
    def test_mals_catalog_accessible(self, release, info, vizier_client):
        """Test that each MALS catalog is accessible from VizieR."""
        catalog_id, description = info
        try:
            tables = vizier_client.get_catalogs(catalog_id)

            assert tables is not None, f"No tables returned for MALS {release}"
            assert len(tables) > 0, f"Empty tables for MALS {release}"

            # Check we got some data
            table = tables[0]
            assert len(table) > 0, f"No rows in MALS {release}"

        except Exception as e:
            pytest.fail(f"Failed to access MALS {release} ({description}): {e}")

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
    """Tests for the load_mals() function."""

    def test_load_mals_dr2_with_high_flux_limit(self):
        """Test loading MALS DR2 with high flux limit (fewer sources, faster)."""
        from rrivis.core.source import load_mals

        # High flux limit = fewer sources = faster test (100 mJy = 0.1 Jy)
        sources, _ = load_mals(flux_limit=100.0, release="dr2")

        assert isinstance(sources, list)

        if len(sources) > 0:
            # Verify source structure
            source = sources[0]
            assert "coords" in source
            assert "flux" in source
            assert "spectral_index" in source
            # Flux should be in Jy (converted from mJy)
            assert source["flux"] >= 0.1  # 100 mJy = 0.1 Jy

    def test_load_mals_dr1(self):
        """Test loading MALS DR1."""
        from rrivis.core.source import load_mals_dr1

        sources, _ = load_mals_dr1(flux_limit=500.0)  # 500 mJy limit

        assert isinstance(sources, list)

    def test_load_mals_dr3(self):
        """Test loading MALS DR3 (HI absorption)."""
        from rrivis.core.source import load_mals_dr3

        # DR3 has fewer sources, use lower limit
        sources, _ = load_mals_dr3(flux_limit=10.0)

        assert isinstance(sources, list)

    def test_load_mals_invalid_release(self):
        """Test that invalid release returns empty list."""
        from rrivis.core.source import load_mals

        sources, _ = load_mals(flux_limit=1.0, release="invalid")

        assert sources == []

    def test_load_mals_source_structure(self):
        """Test that loaded MALS sources have correct structure."""
        from rrivis.core.source import load_mals

        sources, _ = load_mals(flux_limit=500.0, release="dr2")

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
# INTEGRATION TEST: Full Source Loading Pipeline
# =============================================================================

@pytest.mark.slow
class TestSourceLoadingIntegration:
    """Integration tests for the full source loading pipeline."""

    def test_get_sources_with_gleam(self):
        """Test get_sources() with GLEAM catalog."""
        from rrivis.core.source import get_sources

        sources, _ = get_sources(
            use_gleam=True,
            flux_limit=200.0,  # High limit for speed
            gleam_catalogue="VIII/100/gleamegc",
        )

        assert isinstance(sources, list)

    def test_get_sources_with_mals(self):
        """Test get_sources() with MALS catalog."""
        from rrivis.core.source import get_sources

        sources, _ = get_sources(
            use_mals=True,
            mals_release="dr2",
            flux_limit=100.0,  # 100 mJy limit
        )

        assert isinstance(sources, list)

    def test_get_sources_with_gsm(self):
        """Test get_sources() with GSM2008."""
        from rrivis.core.source import get_sources

        sources, _ = get_sources(
            use_gsm=True,
            frequency=100e6,
            nside=8,
            flux_limit=0.1,
        )

        assert isinstance(sources, list)
        assert len(sources) > 0
