"""Tests for rrivis.core.sky.model — SkyModel class and module-level helpers."""

import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import SkyModel, SkyRegion
from rrivis.core.sky.model import _apply_faraday_rotation, _compute_spectral_scale

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def precision():
    return PrecisionConfig.standard()


@pytest.fixture
def test_sky(precision):
    return SkyModel.from_test_sources(
        num_sources=50,
        flux_range=(1.0, 10.0),
        dec_deg=-30.0,
        spectral_index=-0.7,
        precision=precision,
    )


@pytest.fixture
def obs_freq_config():
    return {
        "starting_frequency": 100.0,
        "frequency_interval": 1.0,
        "frequency_bandwidth": 5.0,
        "frequency_unit": "MHz",
    }


# ---------------------------------------------------------------------------
# from_test_sources
# ---------------------------------------------------------------------------


class TestFromTestSources:
    def test_from_test_sources_basic(self, test_sky):
        """Uniform distribution: correct count, mode, coordinate ranges, flux range."""
        assert test_sky.n_sources == 50
        assert test_sky.mode == "point_sources"
        assert test_sky.native_format == "point_sources"

        # RA in [0, 2pi]
        assert np.all(test_sky._ra_rad >= 0)
        assert np.all(test_sky._ra_rad <= 2 * np.pi)

        # Dec all approx -30 degrees in radians
        expected_dec = np.deg2rad(-30.0)
        assert np.allclose(test_sky._dec_rad, expected_dec, atol=1e-6)

        # Flux in [1.0, 10.0]
        assert np.all(test_sky._flux_ref >= 1.0)
        assert np.all(test_sky._flux_ref <= 10.0)

        # Spectral index all -0.7
        assert np.allclose(test_sky._alpha, -0.7)

    def test_from_test_sources_random(self, precision):
        """Random distribution: correct count, dec values vary."""
        sky = SkyModel.from_test_sources(
            num_sources=100,
            flux_range=(0.5, 5.0),
            dec_deg=0.0,
            spectral_index=-0.8,
            distribution="random",
            seed=42,
            precision=precision,
        )
        assert sky.n_sources == 100

        # Dec values should vary (not all identical)
        dec_unique = np.unique(sky._dec_rad)
        assert len(dec_unique) > 1

    def test_from_test_sources_missing_params(self, precision):
        """ValueError when required params are None."""
        with pytest.raises(ValueError, match="dec_deg"):
            SkyModel.from_test_sources(
                flux_range=(1.0, 10.0),
                spectral_index=-0.7,
                dec_deg=None,
                precision=precision,
            )
        with pytest.raises(ValueError, match="flux_range"):
            SkyModel.from_test_sources(
                dec_deg=-30.0,
                spectral_index=-0.7,
                flux_range=None,
                precision=precision,
            )
        with pytest.raises(ValueError, match="spectral_index"):
            SkyModel.from_test_sources(
                dec_deg=-30.0,
                flux_range=(1.0, 10.0),
                spectral_index=None,
                precision=precision,
            )


# ---------------------------------------------------------------------------
# from_point_sources / to_point_sources
# ---------------------------------------------------------------------------


class TestPointSourcesRoundtrip:
    def test_from_point_sources_roundtrip(self, test_sky, precision):
        """Round-trip: test sources -> to_point_sources -> from_point_sources."""
        sources = test_sky.to_point_sources()
        rebuilt = SkyModel.from_point_sources(sources, precision=precision)

        assert rebuilt.n_sources == test_sky.n_sources
        assert np.allclose(rebuilt._ra_rad, test_sky._ra_rad, atol=1e-10)
        assert np.allclose(rebuilt._dec_rad, test_sky._dec_rad, atol=1e-10)
        assert np.allclose(rebuilt._flux_ref, test_sky._flux_ref, atol=1e-10)

    def test_to_point_sources_dict_keys(self, test_sky):
        """Each dict in to_point_sources() has the expected keys."""
        sources = test_sky.to_point_sources()
        assert len(sources) == 50

        for s in sources:
            assert "coords" in s
            assert "flux" in s
            assert "spectral_index" in s
            assert "stokes_q" in s
            assert "stokes_u" in s
            assert "stokes_v" in s
            assert isinstance(s["coords"], SkyCoord)

    def test_to_point_sources_flux_limit(self, precision):
        """flux_limit filters out low-flux sources."""
        sky = SkyModel.from_test_sources(
            num_sources=50,
            flux_range=(0.5, 5.0),
            dec_deg=-30.0,
            spectral_index=-0.7,
            precision=precision,
        )
        sources = sky.to_point_sources(flux_limit=2.0)
        for s in sources:
            assert s["flux"] >= 2.0


# ---------------------------------------------------------------------------
# Mode & Conversion
# ---------------------------------------------------------------------------


class TestModeAndConversion:
    def test_mode_property(self, test_sky, obs_freq_config):
        """Mode switches from point_sources to healpix_multifreq after conversion."""
        assert test_sky.mode == "point_sources"
        test_sky.frequency = 100e6
        test_sky.to_healpix_for_observation(
            nside=32, obs_frequency_config=obs_freq_config
        )
        assert test_sky.mode == "healpix_multifreq"

    def test_to_healpix_for_observation(self, test_sky, obs_freq_config):
        """to_healpix_for_observation populates expected HEALPix attributes."""
        test_sky.frequency = 100e6
        test_sky.to_healpix_for_observation(
            nside=32, obs_frequency_config=obs_freq_config
        )
        assert test_sky.has_multifreq_maps is True
        assert test_sky.n_frequencies == 5
        assert test_sky._healpix_nside == 32

    def test_to_healpix_no_ref_frequency_raises(self, test_sky, obs_freq_config):
        """to_healpix_for_observation raises when no ref_frequency is available."""
        assert test_sky.frequency is None
        with pytest.raises(ValueError, match="ref_frequency must be provided"):
            test_sky.to_healpix_for_observation(
                nside=32, obs_frequency_config=obs_freq_config
            )

    def test_get_map_at_frequency(self, test_sky, obs_freq_config):
        """get_map_at_frequency returns a valid HEALPix map."""
        test_sky.frequency = 100e6
        test_sky.to_healpix_for_observation(
            nside=32, obs_frequency_config=obs_freq_config
        )
        m = test_sky.get_map_at_frequency(100e6)
        assert isinstance(m, np.ndarray)
        assert len(m) == hp.nside2npix(32)
        assert not np.any(np.isnan(m))

    def test_get_map_at_frequency_nearest(self, test_sky, obs_freq_config):
        """Requesting a non-exact frequency returns the nearest available map."""
        test_sky.frequency = 100e6
        test_sky.to_healpix_for_observation(
            nside=32, obs_frequency_config=obs_freq_config
        )
        # Request a frequency slightly off from any channel
        m = test_sky.get_map_at_frequency(100.5e6)
        assert isinstance(m, np.ndarray)
        assert len(m) == hp.nside2npix(32)


# ---------------------------------------------------------------------------
# get_for_visibility
# ---------------------------------------------------------------------------


class TestGetForVisibility:
    def test_get_for_visibility_point_sources(self, test_sky):
        """get_for_visibility('point_sources') returns self with arrays populated."""
        result = test_sky.get_for_visibility("point_sources")
        assert result is test_sky
        assert result._ra_rad is not None

    def test_get_for_visibility_healpix_requires_maps(self, test_sky):
        """Requesting healpix_multifreq without maps raises ValueError."""
        with pytest.raises(ValueError, match="Cannot convert"):
            test_sky.get_for_visibility("healpix_multifreq")


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------


class TestCombine:
    def test_combine_point_sources(self, precision):
        """Combining two point-source models concatenates sources."""
        sky1 = SkyModel.from_test_sources(
            num_sources=20,
            flux_range=(1.0, 5.0),
            dec_deg=-30.0,
            spectral_index=-0.7,
            precision=precision,
        )
        sky2 = SkyModel.from_test_sources(
            num_sources=30,
            flux_range=(2.0, 8.0),
            dec_deg=-30.0,
            spectral_index=-0.8,
            precision=precision,
        )
        combined = SkyModel.combine([sky1, sky2], precision=precision)
        assert combined.n_sources == 50
        assert combined.mode == "point_sources"


# ---------------------------------------------------------------------------
# filter_region
# ---------------------------------------------------------------------------


class TestFilterRegion:
    def test_filter_region_cone(self, precision):
        """Cone filter keeps only sources within the cone radius."""
        sky = SkyModel.from_test_sources(
            num_sources=100,
            flux_range=(1.0, 10.0),
            dec_deg=0.0,
            spectral_index=-0.7,
            distribution="random",
            seed=42,
            precision=precision,
        )
        region = SkyRegion.cone(ra_deg=180.0, dec_deg=0.0, radius_deg=30.0)
        filtered = sky.filter_region(region)

        assert filtered.n_sources < 100

        # All filtered sources should be within 30 degrees of the center
        center = SkyCoord(ra=180.0, dec=0.0, unit="deg", frame="icrs")
        coords = SkyCoord(
            ra=filtered._ra_rad, dec=filtered._dec_rad, unit="rad", frame="icrs"
        )
        separations = coords.separation(center)
        assert np.all(separations.deg <= 30.0 + 1e-6)

    def test_filter_region_box(self, precision):
        """Box filter keeps only sources within the box."""
        sky = SkyModel.from_test_sources(
            num_sources=100,
            flux_range=(1.0, 10.0),
            dec_deg=0.0,
            spectral_index=-0.7,
            distribution="random",
            seed=42,
            precision=precision,
        )
        region = SkyRegion.box(
            ra_deg=180.0, dec_deg=0.0, width_deg=40.0, height_deg=20.0
        )
        filtered = sky.filter_region(region)

        assert filtered.n_sources < 100
        assert filtered.n_sources > 0


# ---------------------------------------------------------------------------
# pixel_solid_angle / pixel_coords
# ---------------------------------------------------------------------------


class TestPixelProperties:
    def test_pixel_solid_angle(self, test_sky, obs_freq_config):
        """pixel_solid_angle is 4pi/npix for the given nside."""
        test_sky.frequency = 100e6
        test_sky.to_healpix_for_observation(
            nside=32, obs_frequency_config=obs_freq_config
        )
        expected = 4.0 * np.pi / hp.nside2npix(32)
        assert np.isclose(test_sky.pixel_solid_angle, expected)

    def test_pixel_solid_angle_no_healpix(self, test_sky):
        """pixel_solid_angle raises ValueError on a point-source-only model."""
        with pytest.raises(ValueError, match="No HEALPix maps"):
            _ = test_sky.pixel_solid_angle

    def test_pixel_coords(self, test_sky, obs_freq_config):
        """pixel_coords returns SkyCoord of correct length."""
        nside = 16
        test_sky.frequency = 100e6
        test_sky.to_healpix_for_observation(
            nside=nside, obs_frequency_config=obs_freq_config
        )
        coords = test_sky.pixel_coords
        assert isinstance(coords, SkyCoord)
        assert len(coords) == hp.nside2npix(nside)

    def test_pixel_coords_no_healpix(self, test_sky):
        """pixel_coords raises ValueError on a point-source-only model."""
        with pytest.raises(ValueError, match="No HEALPix maps"):
            _ = test_sky.pixel_coords


# ---------------------------------------------------------------------------
# Spectral helpers (module-level)
# ---------------------------------------------------------------------------


class TestComputeSpectralScale:
    def test_compute_spectral_scale_power_law(self):
        """Simple power law: (freq/ref_freq)^alpha."""
        alpha = np.array([-0.7])
        result = _compute_spectral_scale(alpha, None, 200e6, 100e6)
        expected = (200.0 / 100.0) ** (-0.7)
        assert np.allclose(result, expected)

    def test_compute_spectral_scale_log_polynomial(self):
        """Log-polynomial with 2 terms differs from simple power law."""
        alpha = np.array([-0.7])
        spectral_coeffs = np.array([[-0.7, -0.1]])
        result_lp = _compute_spectral_scale(alpha, spectral_coeffs, 200e6, 100e6)
        result_pl = _compute_spectral_scale(alpha, None, 200e6, 100e6)
        # They should differ because of the curvature term
        assert not np.allclose(result_lp, result_pl)


class TestApplyFaradayRotation:
    def test_apply_faraday_rotation_zero_rm(self):
        """With rm=None, Q/U are just scaled by spectral_scale."""
        q = np.array([1.0])
        u = np.array([0.5])
        scale = np.array([2.0])
        q_out, u_out = _apply_faraday_rotation(q, u, None, 200e6, 100e6, scale)
        assert np.allclose(q_out, q * scale)
        assert np.allclose(u_out, u * scale)

    def test_apply_faraday_rotation_nonzero(self):
        """With non-zero RM, Q/U are rotated — different from simple scaling."""
        q = np.array([1.0])
        u = np.array([0.5])
        rm = np.array([10.0])
        scale = np.array([2.0])
        q_out, u_out = _apply_faraday_rotation(q, u, rm, 200e6, 100e6, scale)
        # Should differ from simple scaling
        assert not np.allclose(q_out, q * scale)
        assert not np.allclose(u_out, u * scale)


# ---------------------------------------------------------------------------
# estimate_healpix_memory
# ---------------------------------------------------------------------------


class TestEstimateHealpixMemory:
    def test_estimate_healpix_memory(self):
        """estimate_healpix_memory returns dict with expected keys and positive values."""
        result = SkyModel.estimate_healpix_memory(nside=64, n_frequencies=10)
        assert "npix" in result
        assert "total_bytes" in result
        assert "total_mb" in result
        assert "total_gb" in result
        assert "bytes_per_map" in result
        assert result["total_bytes"] > 0
        assert result["npix"] == hp.nside2npix(64)


# ---------------------------------------------------------------------------
# _empty_sky
# ---------------------------------------------------------------------------


class TestEmptySky:
    def test_empty_sky(self):
        """_empty_sky creates a model with zero sources in point_sources mode."""
        sky = SkyModel._empty_sky("test")
        assert sky.n_sources == 0
        assert sky.mode == "point_sources"
        assert sky.model_name == "test"


# ---------------------------------------------------------------------------
# _parse_frequency_config
# ---------------------------------------------------------------------------


class TestParseFrequencyConfig:
    def test_parse_frequency_config_mhz(self):
        """Parse MHz config: correct Hz values and channel count."""
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 5.0,
            "frequency_unit": "MHz",
        }
        freqs = SkyModel._parse_frequency_config(config)
        assert len(freqs) == 5
        assert np.isclose(freqs[0], 100e6)
        assert np.isclose(freqs[1], 101e6)
        assert np.isclose(freqs[-1], 104e6)

    def test_parse_frequency_config_ghz(self):
        """Parse GHz config: correct Hz values."""
        config = {
            "starting_frequency": 1.0,
            "frequency_interval": 0.1,
            "frequency_bandwidth": 0.5,
            "frequency_unit": "GHz",
        }
        freqs = SkyModel._parse_frequency_config(config)
        assert len(freqs) == 5
        assert np.isclose(freqs[0], 1.0e9)
        assert np.isclose(freqs[-1], 1.4e9)
