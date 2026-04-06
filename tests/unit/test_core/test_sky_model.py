"""Tests for rrivis.core.sky.model — SkyModel class and module-level helpers."""

import dataclasses

import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import SkyModel, SkyRegion
from rrivis.core.sky.spectral import apply_faraday_rotation, compute_spectral_scale
from rrivis.utils.frequency import parse_frequency_config

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
# Frozen dataclass (immutability)
# ---------------------------------------------------------------------------


class TestFrozenDataclass:
    def test_frozen_instance_error(self, test_sky):
        """Assigning to a field on a frozen SkyModel raises FrozenInstanceError."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            test_sky.frequency = 100e6

    def test_with_frequency_returns_new(self, test_sky):
        """with_frequency returns a new instance with updated frequency."""
        sky2 = test_sky.with_frequency(100e6)
        assert sky2 is not test_sky
        assert sky2.frequency == 100e6
        assert test_sky.frequency is None


# ---------------------------------------------------------------------------
# __post_init__ validation
# ---------------------------------------------------------------------------


class TestPostInitValidation:
    def test_mismatched_array_lengths(self, precision):
        """ValueError when point-source arrays have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            SkyModel(
                _ra_rad=np.zeros(10),
                _dec_rad=np.zeros(5),
                _flux_ref=np.zeros(10),
                _alpha=np.zeros(10),
                _stokes_q=np.zeros(10),
                _stokes_u=np.zeros(10),
                _stokes_v=np.zeros(10),
                _precision=precision,
            )

    def test_partial_stokes_raises(self, precision):
        """ValueError when only some Stokes Q/U/V are provided."""
        with pytest.raises(ValueError, match="Stokes Q/U/V must be all set"):
            SkyModel(
                _ra_rad=np.zeros(5),
                _dec_rad=np.zeros(5),
                _flux_ref=np.zeros(5),
                _alpha=np.zeros(5),
                _stokes_q=np.zeros(5),
                _stokes_u=None,
                _stokes_v=np.zeros(5),
                _precision=precision,
            )

    def test_invalid_native_format(self, precision):
        """ValueError for invalid _native_format."""
        with pytest.raises(ValueError, match="_native_format must be"):
            SkyModel(
                _ra_rad=np.zeros(5),
                _dec_rad=np.zeros(5),
                _flux_ref=np.zeros(5),
                _alpha=np.zeros(5),
                _stokes_q=np.zeros(5),
                _stokes_u=np.zeros(5),
                _stokes_v=np.zeros(5),
                _native_format="invalid",
                _precision=precision,
            )


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

    def test_from_test_sources_polarized(self, precision):
        """Polarized test sources have non-zero Stokes Q/U/V."""
        sky = SkyModel.from_test_sources(
            num_sources=20,
            flux_range=(1.0, 5.0),
            dec_deg=-30.0,
            spectral_index=-0.7,
            polarization_fraction=0.1,
            polarization_angle_deg=45.0,
            stokes_v_fraction=0.02,
            precision=precision,
        )
        assert np.any(sky._stokes_q != 0)
        assert np.any(sky._stokes_u != 0)
        assert np.any(sky._stokes_v != 0)


# ---------------------------------------------------------------------------
# from_arrays / from_point_sources / as_point_source_dicts
# ---------------------------------------------------------------------------


class TestPointSourcesRoundtrip:
    def test_from_arrays(self, precision):
        """from_arrays creates a valid SkyModel."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0, 1.0, 2.0]),
            dec_rad=np.array([0.0, 0.1, -0.1]),
            flux_ref=np.array([1.0, 2.0, 3.0]),
            frequency=100e6,
            model_name="test_arrays",
            precision=precision,
        )
        assert sky.n_sources == 3
        assert sky.frequency == 100e6
        assert sky.model_name == "test_arrays"
        assert np.allclose(sky._alpha, -0.7)  # default

    def test_from_point_sources_roundtrip(self, test_sky, precision):
        """Round-trip: test sources -> as_point_source_dicts -> from_point_sources."""
        sources = test_sky.as_point_source_dicts()
        rebuilt = SkyModel.from_point_sources(sources, precision=precision)

        assert rebuilt.n_sources == test_sky.n_sources
        assert np.allclose(rebuilt._ra_rad, test_sky._ra_rad, atol=1e-10)
        assert np.allclose(rebuilt._dec_rad, test_sky._dec_rad, atol=1e-10)
        assert np.allclose(rebuilt._flux_ref, test_sky._flux_ref, atol=1e-10)

    def test_as_point_source_dicts_keys(self, test_sky):
        """Each dict in as_point_source_dicts() has the expected keys."""
        sources = test_sky.as_point_source_dicts()
        assert len(sources) == 50

        for s in sources:
            assert "coords" in s
            assert "flux" in s
            assert "spectral_index" in s
            assert "stokes_q" in s
            assert "stokes_u" in s
            assert "stokes_v" in s
            assert isinstance(s["coords"], SkyCoord)

    def test_as_point_source_dicts_flux_limit(self, precision):
        """flux_limit filters out low-flux sources."""
        sky = SkyModel.from_test_sources(
            num_sources=50,
            flux_range=(0.5, 5.0),
            dec_deg=-30.0,
            spectral_index=-0.7,
            precision=precision,
        )
        sources = sky.as_point_source_dicts(flux_limit=2.0)
        for s in sources:
            assert s["flux"] >= 2.0

    def test_as_point_source_dicts_no_mutation(self, test_sky):
        """as_point_source_dicts does not mutate the original SkyModel."""
        original_ra = test_sky._ra_rad.copy()
        _ = test_sky.as_point_source_dicts()
        assert np.array_equal(test_sky._ra_rad, original_ra)


# ---------------------------------------------------------------------------
# Mode & Conversion (immutable)
# ---------------------------------------------------------------------------


class TestModeAndConversion:
    def test_mode_property(self, test_sky, obs_freq_config):
        """Mode switches from point_sources to healpix_multifreq after conversion."""
        assert test_sky.mode == "point_sources"
        sky = test_sky.with_frequency(100e6)
        sky = sky.with_healpix_maps(nside=32, obs_frequency_config=obs_freq_config)
        assert sky.mode == "healpix_multifreq"

    def test_with_healpix_maps(self, test_sky, obs_freq_config):
        """with_healpix_maps populates expected HEALPix attributes."""
        sky = test_sky.with_frequency(100e6)
        sky = sky.with_healpix_maps(nside=32, obs_frequency_config=obs_freq_config)
        assert sky.has_multifreq_maps is True
        assert sky.n_frequencies == 5
        assert sky._healpix_nside == 32

    def test_with_healpix_maps_returns_new(self, test_sky, obs_freq_config):
        """with_healpix_maps returns a different object, original unchanged."""
        sky = test_sky.with_frequency(100e6)
        sky2 = sky.with_healpix_maps(nside=32, obs_frequency_config=obs_freq_config)
        assert sky2 is not sky
        assert sky.has_multifreq_maps is False
        assert sky2.has_multifreq_maps is True

    def test_with_healpix_maps_no_ref_frequency_raises(self, test_sky, obs_freq_config):
        """with_healpix_maps raises when no ref_frequency is available."""
        assert test_sky.frequency is None
        with pytest.raises(ValueError, match="ref_frequency must be provided"):
            test_sky.with_healpix_maps(nside=32, obs_frequency_config=obs_freq_config)

    def test_get_map_at_frequency(self, test_sky, obs_freq_config):
        """get_map_at_frequency returns a valid HEALPix map."""
        sky = test_sky.with_frequency(100e6).with_healpix_maps(
            nside=32, obs_frequency_config=obs_freq_config
        )
        m = sky.get_map_at_frequency(100e6)
        assert isinstance(m, np.ndarray)
        assert len(m) == hp.nside2npix(32)
        assert not np.any(np.isnan(m))

    def test_get_map_at_frequency_nearest(self, test_sky, obs_freq_config):
        """Requesting a non-exact frequency returns the nearest available map."""
        sky = test_sky.with_frequency(100e6).with_healpix_maps(
            nside=32, obs_frequency_config=obs_freq_config
        )
        # Request a frequency slightly off from any channel
        m = sky.get_map_at_frequency(100.5e6)
        assert isinstance(m, np.ndarray)
        assert len(m) == hp.nside2npix(32)


# ---------------------------------------------------------------------------
# with_representation
# ---------------------------------------------------------------------------


class TestWithRepresentation:
    def test_with_representation_point_sources(self, test_sky):
        """with_representation('point_sources') returns model with arrays populated."""
        result = test_sky.with_representation("point_sources")
        assert result._ra_rad is not None

    def test_with_representation_healpix_requires_maps(self, test_sky):
        """Requesting healpix_multifreq without maps raises ValueError."""
        with pytest.raises(ValueError, match="Cannot convert"):
            test_sky.with_representation("healpix_multifreq")


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
        sky = test_sky.with_frequency(100e6).with_healpix_maps(
            nside=32, obs_frequency_config=obs_freq_config
        )
        expected = 4.0 * np.pi / hp.nside2npix(32)
        assert np.isclose(sky.pixel_solid_angle, expected)

    def test_pixel_solid_angle_no_healpix(self, test_sky):
        """pixel_solid_angle raises ValueError on a point-source-only model."""
        with pytest.raises(ValueError, match="No HEALPix maps"):
            _ = test_sky.pixel_solid_angle

    def test_pixel_coords(self, test_sky, obs_freq_config):
        """pixel_coords returns SkyCoord of correct length."""
        nside = 16
        sky = test_sky.with_frequency(100e6).with_healpix_maps(
            nside=nside, obs_frequency_config=obs_freq_config
        )
        coords = sky.pixel_coords
        assert isinstance(coords, SkyCoord)
        assert len(coords) == hp.nside2npix(nside)

    def test_pixel_coords_no_healpix(self, test_sky):
        """pixel_coords raises ValueError on a point-source-only model."""
        with pytest.raises(ValueError, match="No HEALPix maps"):
            _ = test_sky.pixel_coords


# ---------------------------------------------------------------------------
# Spectral helpers (now in spectral.py)
# ---------------------------------------------------------------------------


class TestComputeSpectralScale:
    def test_compute_spectral_scale_power_law(self):
        """Simple power law: (freq/ref_freq)^alpha."""
        alpha = np.array([-0.7])
        result = compute_spectral_scale(alpha, None, 200e6, 100e6)
        expected = (200.0 / 100.0) ** (-0.7)
        assert np.allclose(result, expected)

    def test_compute_spectral_scale_log_polynomial(self):
        """Log-polynomial with 2 terms differs from simple power law."""
        alpha = np.array([-0.7])
        spectral_coeffs = np.array([[-0.7, -0.1]])
        result_lp = compute_spectral_scale(alpha, spectral_coeffs, 200e6, 100e6)
        result_pl = compute_spectral_scale(alpha, None, 200e6, 100e6)
        # They should differ because of the curvature term
        assert not np.allclose(result_lp, result_pl)


class TestApplyFaradayRotation:
    def test_apply_faraday_rotation_zero_rm(self):
        """With rm=None, Q/U are just scaled by spectral_scale."""
        q = np.array([1.0])
        u = np.array([0.5])
        scale = np.array([2.0])
        q_out, u_out = apply_faraday_rotation(q, u, None, 200e6, 100e6, scale)
        assert np.allclose(q_out, q * scale)
        assert np.allclose(u_out, u * scale)

    def test_apply_faraday_rotation_nonzero(self):
        """With non-zero RM, Q/U are rotated — different from simple scaling."""
        q = np.array([1.0])
        u = np.array([0.5])
        rm = np.array([10.0])
        scale = np.array([2.0])
        q_out, u_out = apply_faraday_rotation(q, u, rm, 200e6, 100e6, scale)
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

    def test_empty_sky_with_frequency(self):
        """_empty_sky accepts a frequency parameter."""
        sky = SkyModel._empty_sky("test", frequency=100e6)
        assert sky.frequency == 100e6


# ---------------------------------------------------------------------------
# parse_frequency_config (now in utils/frequency.py)
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
        freqs = parse_frequency_config(config)
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
        freqs = parse_frequency_config(config)
        assert len(freqs) == 5
        assert np.isclose(freqs[0], 1.0e9)
        assert np.isclose(freqs[-1], 1.4e9)


# ---------------------------------------------------------------------------
# Plot accessor
# ---------------------------------------------------------------------------


class TestPlotAccessor:
    def test_plot_returns_sky_plotter(self, test_sky):
        """sky.plot returns a SkyPlotter instance."""
        from rrivis.core.sky.plotter import SkyPlotter

        plotter = test_sky.plot
        assert isinstance(plotter, SkyPlotter)


# ---------------------------------------------------------------------------
# Additional __post_init__ validation
# ---------------------------------------------------------------------------


class TestPostInitValidationExpanded:
    def test_healpix_maps_without_nside_raises(self):
        """ValueError when healpix maps set but nside missing."""
        maps = np.zeros((3, hp.nside2npix(8)), dtype=np.float32)
        with pytest.raises(ValueError, match="_healpix_nside"):
            SkyModel(_healpix_maps=maps, _healpix_nside=None)

    def test_healpix_maps_wrong_ndim_raises(self):
        """ValueError when healpix maps are 1-D instead of 2-D."""
        maps = np.zeros(hp.nside2npix(8), dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            SkyModel(_healpix_maps=maps, _healpix_nside=8)

    def test_healpix_maps_wrong_npix_raises(self):
        """ValueError when npix doesn't match nside."""
        maps = np.zeros((3, 100), dtype=np.float32)  # wrong npix for any nside
        with pytest.raises(ValueError):
            SkyModel(_healpix_maps=maps, _healpix_nside=8)

    def test_healpix_pol_map_shape_mismatch_raises(self):
        """ValueError when Q map shape differs from I map."""
        nside = 8
        npix = hp.nside2npix(nside)
        i_maps = np.zeros((3, npix), dtype=np.float32)
        q_maps = np.zeros((2, npix), dtype=np.float32)  # wrong n_freq
        with pytest.raises(ValueError, match="does not match"):
            SkyModel(
                _healpix_maps=i_maps,
                _healpix_q_maps=q_maps,
                _healpix_nside=nside,
            )

    def test_invalid_brightness_conversion_raises(self, precision):
        """ValueError for invalid brightness_conversion string."""
        with pytest.raises(ValueError, match="brightness_conversion"):
            SkyModel(
                _ra_rad=np.zeros(5),
                _dec_rad=np.zeros(5),
                _flux_ref=np.zeros(5),
                _alpha=np.zeros(5),
                _stokes_q=np.zeros(5),
                _stokes_u=np.zeros(5),
                _stokes_v=np.zeros(5),
                brightness_conversion="invalid",
                _precision=precision,
            )

    def test_valid_empty_arrays_ok(self, precision):
        """Zero-length arrays are valid (empty model)."""
        sky = SkyModel(
            _ra_rad=np.zeros(0),
            _dec_rad=np.zeros(0),
            _flux_ref=np.zeros(0),
            _alpha=np.zeros(0),
            _stokes_q=np.zeros(0),
            _stokes_u=np.zeros(0),
            _stokes_v=np.zeros(0),
            _precision=precision,
        )
        assert sky.n_sources == 0


# ---------------------------------------------------------------------------
# with_healpix_maps expanded
# ---------------------------------------------------------------------------


class TestWithHealpixMapsExpanded:
    def test_with_healpix_maps_frequencies_array(self, test_sky, precision):
        """Pass frequencies array directly (not obs_frequency_config)."""
        sky = test_sky.with_frequency(100e6)
        freqs = np.array([100e6, 101e6, 102e6])
        result = sky.with_healpix_maps(nside=16, frequencies=freqs)
        assert result.n_frequencies == 3
        assert result._healpix_nside == 16

    def test_original_unchanged_after_with_healpix(self, test_sky, obs_freq_config):
        """Verify the original model has no healpix maps after calling with_healpix_maps."""
        sky = test_sky.with_frequency(100e6)
        sky2 = sky.with_healpix_maps(nside=16, obs_frequency_config=obs_freq_config)
        assert sky._healpix_maps is None
        assert sky2._healpix_maps is not None

    def test_multiple_nside_values(self, test_sky, precision):
        """Different nside values produce correct pixel counts."""
        sky = test_sky.with_frequency(100e6)
        freqs = np.array([100e6, 101e6])
        for nside in [8, 16, 32]:
            result = sky.with_healpix_maps(nside=nside, frequencies=freqs)
            m = result.get_map_at_frequency(100e6)
            assert len(m) == hp.nside2npix(nside)


# ---------------------------------------------------------------------------
# with_representation expanded
# ---------------------------------------------------------------------------


class TestWithRepresentationExpanded:
    def test_point_sources_already_populated_returns_self(self, test_sky):
        """Point-source model with arrays → returns self (no conversion needed)."""
        result = test_sky.with_representation("point_sources")
        assert result is test_sky

    def test_healpix_to_point_sources_conversion(self, test_sky, obs_freq_config):
        """Create healpix model, convert to point sources."""
        sky = test_sky.with_frequency(100e6).with_healpix_maps(
            nside=16, obs_frequency_config=obs_freq_config
        )
        # This model has healpix maps AND point-source arrays (from original).
        # Create a healpix-only model to test conversion.
        healpix_only = SkyModel(
            _healpix_maps=sky._healpix_maps,
            _healpix_nside=sky._healpix_nside,
            _observation_frequencies=sky._observation_frequencies,
            _native_format="healpix",
            frequency=100e6,
        )
        result = healpix_only.with_representation("point_sources", frequency=100e6)
        assert result._ra_rad is not None
        assert len(result._ra_rad) > 0


# ---------------------------------------------------------------------------
# as_point_source_dicts expanded
# ---------------------------------------------------------------------------


class TestAsPointSourceDictsExpanded:
    def test_includes_rotation_measure(self, precision):
        """Model with RM → dicts have 'rotation_measure' key."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0, 1.0]),
            dec_rad=np.array([0.0, 0.1]),
            flux_ref=np.array([1.0, 2.0]),
            rotation_measure=np.array([5.0, 10.0]),
            precision=precision,
        )
        dicts = sky.as_point_source_dicts()
        assert all("rotation_measure" in d for d in dicts)
        assert dicts[0]["rotation_measure"] == 5.0

    def test_includes_gaussian_morphology(self, precision):
        """Model with morphology → dicts have morphology keys."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux_ref=np.array([1.0]),
            major_arcsec=np.array([10.0]),
            minor_arcsec=np.array([5.0]),
            pa_deg=np.array([45.0]),
            precision=precision,
        )
        dicts = sky.as_point_source_dicts()
        assert dicts[0]["major_arcsec"] == 10.0
        assert dicts[0]["minor_arcsec"] == 5.0
        assert dicts[0]["pa_deg"] == 45.0

    def test_includes_spectral_coeffs(self, precision):
        """Model with spectral coefficients → dicts have 'spectral_coeffs' key."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux_ref=np.array([1.0]),
            spectral_coeffs=np.array([[-0.7, -0.1]]),
            precision=precision,
        )
        dicts = sky.as_point_source_dicts()
        assert "spectral_coeffs" in dicts[0]
        assert np.isclose(dicts[0]["spectral_coeffs"][0], -0.7)


# ---------------------------------------------------------------------------
# from_arrays expanded
# ---------------------------------------------------------------------------


class TestFromArraysExpanded:
    def test_from_arrays_with_all_optional_fields(self, precision):
        """Pass all optional fields → all stored correctly."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0, 1.0]),
            dec_rad=np.array([0.0, 0.1]),
            flux_ref=np.array([1.0, 2.0]),
            alpha=np.array([-0.5, -0.9]),
            stokes_q=np.array([0.1, 0.2]),
            stokes_u=np.array([0.05, 0.1]),
            stokes_v=np.array([0.01, 0.02]),
            rotation_measure=np.array([5.0, 10.0]),
            major_arcsec=np.array([10.0, 0.0]),
            minor_arcsec=np.array([5.0, 0.0]),
            pa_deg=np.array([45.0, 0.0]),
            spectral_coeffs=np.array([[-0.5, -0.1], [-0.9, 0.0]]),
            precision=precision,
        )
        assert sky.n_sources == 2
        assert np.isclose(sky._rotation_measure[0], 5.0)
        assert np.isclose(sky._major_arcsec[0], 10.0)
        assert sky._spectral_coeffs.shape == (2, 2)

    def test_from_arrays_single_source(self, precision):
        """Single source (N=1) → valid model."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([np.pi]),
            dec_rad=np.array([0.0]),
            flux_ref=np.array([5.0]),
            precision=precision,
        )
        assert sky.n_sources == 1

    def test_from_arrays_empty(self, precision):
        """Empty arrays (N=0) → valid empty model."""
        sky = SkyModel.from_arrays(
            ra_rad=np.zeros(0),
            dec_rad=np.zeros(0),
            flux_ref=np.zeros(0),
            precision=precision,
        )
        assert sky.n_sources == 0

    def test_from_arrays_default_alpha(self, precision):
        """No alpha passed → defaults to -0.7."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0, 1.0, 2.0]),
            dec_rad=np.array([0.0, 0.0, 0.0]),
            flux_ref=np.array([1.0, 2.0, 3.0]),
            precision=precision,
        )
        assert np.allclose(sky._alpha, -0.7)


# ---------------------------------------------------------------------------
# from_point_sources expanded
# ---------------------------------------------------------------------------


class TestFromPointSourcesExpanded:
    def test_empty_list(self, precision):
        """Empty list → empty model."""
        sky = SkyModel.from_point_sources([], precision=precision)
        assert sky.n_sources == 0

    def test_default_spectral_index(self, precision):
        """No 'spectral_index' in dict → defaults to -0.7."""
        src = {
            "coords": SkyCoord(ra=180.0, dec=-30.0, unit="deg", frame="icrs"),
            "flux": 5.0,
        }
        sky = SkyModel.from_point_sources([src], precision=precision)
        assert np.isclose(sky._alpha[0], -0.7)

    def test_optional_fields_preserved(self, precision):
        """Dicts with RM and morphology → preserved in model."""
        src = {
            "coords": SkyCoord(ra=0.0, dec=0.0, unit="deg", frame="icrs"),
            "flux": 1.0,
            "spectral_index": -0.8,
            "rotation_measure": 15.0,
            "major_arcsec": 20.0,
            "minor_arcsec": 10.0,
            "pa_deg": 30.0,
            "spectral_coeffs": [-0.8, -0.05],
        }
        sky = SkyModel.from_point_sources([src], precision=precision)
        assert np.isclose(sky._rotation_measure[0], 15.0)
        assert np.isclose(sky._major_arcsec[0], 20.0)
        assert sky._spectral_coeffs is not None


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_pyradiosky_point_source(self, test_sky):
        """Point-source model → pyradiosky SkyModel with correct structure."""
        sky = test_sky.with_frequency(100e6)
        psky = sky._to_pyradiosky()
        assert psky.component_type == "point"
        assert psky.Ncomponents == 50
        assert psky.spectral_type == "spectral_index"

    def test_to_pyradiosky_healpix(self, test_sky, obs_freq_config):
        """HEALPix-only model → pyradiosky SkyModel with healpix component."""
        sky = test_sky.with_frequency(100e6).with_healpix_maps(
            nside=8, obs_frequency_config=obs_freq_config
        )
        # Create a healpix-only model (no point-source arrays)
        healpix_only = SkyModel(
            _healpix_maps=sky._healpix_maps,
            _healpix_nside=sky._healpix_nside,
            _observation_frequencies=sky._observation_frequencies,
            _native_format="healpix",
            frequency=100e6,
            model_name="test_healpix",
        )
        psky = healpix_only._to_pyradiosky()
        assert psky.component_type == "healpix"
        assert psky.nside == 8

    def test_to_pyradiosky_empty_raises(self):
        """Empty model → ValueError."""
        sky = SkyModel._empty_sky("test")
        with pytest.raises(ValueError, match="empty"):
            sky._to_pyradiosky()

    def test_save_load_roundtrip(self, test_sky, tmp_path):
        """save() → load() preserves source count, coords, flux."""
        sky = test_sky.with_frequency(100e6)
        path = str(tmp_path / "test.skyh5")
        sky.save(path)

        loaded = SkyModel.load(path, precision=PrecisionConfig.standard())
        assert loaded.n_sources == sky.n_sources
        assert np.allclose(loaded._flux_ref, sky._flux_ref, rtol=1e-5)

    def test_save_warns_lost_data(self, precision, tmp_path):
        """Model with RM → save warns about lost data."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux_ref=np.array([1.0]),
            rotation_measure=np.array([10.0]),
            frequency=100e6,
            precision=precision,
        )
        path = str(tmp_path / "rm_test.skyh5")
        with pytest.warns(UserWarning, match="rotation measure"):
            sky.save(path)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_point_sources(self, test_sky):
        """Point-source model repr contains source count."""
        r = repr(test_sky)
        assert "n_sources=50" in r
        assert "point_sources" in r

    def test_repr_healpix(self, test_sky, obs_freq_config):
        """HEALPix model repr contains nside and frequency info."""
        sky = test_sky.with_frequency(100e6).with_healpix_maps(
            nside=16, obs_frequency_config=obs_freq_config
        )
        r = repr(sky)
        assert "healpix_multifreq" in r
        assert "nside=16" in r

    def test_repr_with_extensions(self, precision):
        """Model with RM and Gaussian → repr mentions them."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux_ref=np.array([1.0]),
            rotation_measure=np.array([10.0]),
            major_arcsec=np.array([5.0]),
            minor_arcsec=np.array([3.0]),
            pa_deg=np.array([0.0]),
            precision=precision,
        )
        r = repr(sky)
        assert "RM" in r
        assert "gaussian" in r


# ---------------------------------------------------------------------------
# Frequency resolution
# ---------------------------------------------------------------------------


class TestFrequencyResolution:
    def test_exact_frequency_found(self, test_sky, obs_freq_config):
        """Requesting exact channel frequency returns correct map."""
        sky = test_sky.with_frequency(100e6).with_healpix_maps(
            nside=8, obs_frequency_config=obs_freq_config
        )
        m = sky.get_map_at_frequency(100e6)
        assert isinstance(m, np.ndarray)

    def test_nearest_frequency_selected(self, test_sky, obs_freq_config):
        """Between two channels → nearest returned."""
        sky = test_sky.with_frequency(100e6).with_healpix_maps(
            nside=8, obs_frequency_config=obs_freq_config
        )
        # Channels at 100, 101, 102, 103, 104 MHz
        # 100.3 MHz → nearest is 100 MHz (index 0)
        m1 = sky.get_map_at_frequency(100.3e6)
        m0 = sky.get_map_at_frequency(100e6)
        assert np.array_equal(m1, m0)


# ---------------------------------------------------------------------------
# Multi-frequency accessors
# ---------------------------------------------------------------------------


class TestMultifreqAccessors:
    def test_get_multifreq_maps(self, test_sky, obs_freq_config):
        """Returns (maps, nside, frequencies) with correct shapes."""
        sky = test_sky.with_frequency(100e6).with_healpix_maps(
            nside=8, obs_frequency_config=obs_freq_config
        )
        maps, nside, freqs = sky.get_multifreq_maps()
        assert isinstance(maps, np.ndarray)
        assert maps.shape == (5, hp.nside2npix(8))
        assert nside == 8
        assert len(freqs) == 5

    def test_get_stokes_maps_unpolarized(self, test_sky, obs_freq_config):
        """Unpolarized model → Q, U, V are None."""
        sky = test_sky.with_frequency(100e6).with_healpix_maps(
            nside=8, obs_frequency_config=obs_freq_config
        )
        si, sq, su, sv = sky.get_stokes_maps_at_frequency(100e6)
        assert isinstance(si, np.ndarray)
        assert sq is None
        assert su is None
        assert sv is None

    def test_no_maps_raises(self, test_sky):
        """Point-source model → ValueError for healpix accessors."""
        with pytest.raises(ValueError):
            test_sky.get_multifreq_maps()
        with pytest.raises(ValueError):
            test_sky.get_map_at_frequency(100e6)
        with pytest.raises(ValueError):
            test_sky.get_stokes_maps_at_frequency(100e6)


# ---------------------------------------------------------------------------
# filter_region expanded
# ---------------------------------------------------------------------------


class TestFilterRegionExpanded:
    def test_filter_healpix_model(self, test_sky, obs_freq_config):
        """Cone filter on healpix → out-of-region pixels zeroed."""
        sky = test_sky.with_frequency(100e6).with_healpix_maps(
            nside=8, obs_frequency_config=obs_freq_config
        )
        # Create a healpix-only model to test the healpix filtering path
        healpix_only = SkyModel(
            _healpix_maps=sky._healpix_maps,
            _healpix_nside=sky._healpix_nside,
            _observation_frequencies=sky._observation_frequencies,
            _native_format="healpix",
            frequency=100e6,
            model_name="test",
        )
        region = SkyRegion.cone(ra_deg=0.0, dec_deg=-30.0, radius_deg=10.0)
        filtered = healpix_only.filter_region(region)
        # Some pixels should be zero (outside cone)
        m = filtered.get_map_at_frequency(100e6)
        assert np.any(m == 0)

    def test_filter_preserves_metadata(self, test_sky, precision):
        """Filtered model retains model_name, frequency, precision."""
        sky = test_sky.with_frequency(100e6)
        region = SkyRegion.cone(ra_deg=180.0, dec_deg=-30.0, radius_deg=30.0)
        filtered = sky.filter_region(region)
        assert filtered.model_name == sky.model_name
        assert filtered.frequency == sky.frequency

    def test_filter_union_region(self, precision):
        """Union of two cones → sources in either cone kept."""
        sky = SkyModel.from_test_sources(
            num_sources=200,
            flux_range=(1.0, 5.0),
            dec_deg=0.0,
            spectral_index=-0.7,
            distribution="random",
            seed=42,
            precision=precision,
        )
        r1 = SkyRegion.cone(ra_deg=90.0, dec_deg=0.0, radius_deg=20.0)
        r2 = SkyRegion.cone(ra_deg=270.0, dec_deg=0.0, radius_deg=20.0)
        union = SkyRegion.union([r1, r2])
        filtered = sky.filter_region(union)
        # Should have sources from both cones
        assert filtered.n_sources > 0
        assert filtered.n_sources < sky.n_sources


# ---------------------------------------------------------------------------
# Memory estimation expanded
# ---------------------------------------------------------------------------


class TestEstimateMemoryExpanded:
    def test_memory_scales_with_nside(self):
        """nside=64 vs nside=128 → 4x memory (npix ∝ nside²)."""
        m64 = SkyModel.estimate_healpix_memory(nside=64, n_frequencies=10)
        m128 = SkyModel.estimate_healpix_memory(nside=128, n_frequencies=10)
        assert np.isclose(m128["total_bytes"] / m64["total_bytes"], 4.0)

    def test_memory_scales_with_frequencies(self):
        """10 vs 20 frequencies → 2x memory."""
        m10 = SkyModel.estimate_healpix_memory(nside=64, n_frequencies=10)
        m20 = SkyModel.estimate_healpix_memory(nside=64, n_frequencies=20)
        assert np.isclose(m20["total_bytes"] / m10["total_bytes"], 2.0)

    def test_memory_scales_with_stokes(self):
        """n_stokes=4 → 4x memory vs n_stokes=1."""
        m1 = SkyModel.estimate_healpix_memory(nside=64, n_frequencies=10, n_stokes=1)
        m4 = SkyModel.estimate_healpix_memory(nside=64, n_frequencies=10, n_stokes=4)
        assert np.isclose(m4["total_bytes"] / m1["total_bytes"], 4.0)


# ---------------------------------------------------------------------------
# Degree/Radian conversion at precision
# ---------------------------------------------------------------------------


class TestDegRadConversion:
    def test_deg_to_rad_known_values(self):
        """180° → π, 90° → π/2."""
        arr = np.array([180.0, 90.0, 0.0])
        result = SkyModel._deg_to_rad_at_precision(arr, None)
        assert np.allclose(result, [np.pi, np.pi / 2, 0.0])

    def test_rad_to_deg_known_values(self):
        """π → 180°, π/2 → 90°."""
        arr = np.array([np.pi, np.pi / 2, 0.0])
        result = SkyModel._rad_to_deg_at_precision(arr, None)
        assert np.allclose(result, [180.0, 90.0, 0.0])

    def test_round_trip(self):
        """deg → rad → deg preserves values."""
        arr = np.array([0.0, 45.0, 90.0, 180.0, 270.0, 360.0])
        rad = SkyModel._deg_to_rad_at_precision(arr, None)
        deg = SkyModel._rad_to_deg_at_precision(rad, None)
        assert np.allclose(deg, arr)

    def test_precision_dtype(self):
        """With float32 precision → result is float32."""
        precision = PrecisionConfig.fast()  # uses float32
        arr = np.array([180.0, 90.0])
        result = SkyModel._deg_to_rad_at_precision(arr, precision)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Helper: build a healpix-only SkyModel for reuse in multiple test classes
# ---------------------------------------------------------------------------


def _make_healpix_only_model(
    nside=8,
    n_freq=3,
    with_q=False,
    with_u=False,
    with_v=False,
    precision=None,
    fill_value=1.0,
):
    """Build a healpix-only SkyModel (no point-source arrays)."""
    npix = hp.nside2npix(nside)
    freqs = np.array([100e6 + i * 1e6 for i in range(n_freq)])
    i_maps = np.full((n_freq, npix), fill_value, dtype=np.float64)
    q_maps = np.full((n_freq, npix), 0.01, dtype=np.float64) if with_q else None
    u_maps = np.full((n_freq, npix), 0.02, dtype=np.float64) if with_u else None
    v_maps = np.full((n_freq, npix), 0.005, dtype=np.float64) if with_v else None
    return SkyModel(
        _healpix_maps=i_maps,
        _healpix_q_maps=q_maps,
        _healpix_u_maps=u_maps,
        _healpix_v_maps=v_maps,
        _healpix_nside=nside,
        _observation_frequencies=freqs,
        _native_format="healpix",
        frequency=freqs[0],
        model_name="test_healpix",
        _precision=precision,
    )


# ---------------------------------------------------------------------------
# TestValidationPrecision
# ---------------------------------------------------------------------------


class TestValidationPrecision:
    def test_validate_precision_raises(self):
        """SkyModel with _precision=None → _source_dtype() raises ValueError."""
        sky = SkyModel(
            _ra_rad=np.zeros(3),
            _dec_rad=np.zeros(3),
            _flux_ref=np.zeros(3),
            _alpha=np.zeros(3),
            _stokes_q=np.zeros(3),
            _stokes_u=np.zeros(3),
            _stokes_v=np.zeros(3),
            _precision=None,
        )
        with pytest.raises(ValueError, match="PrecisionConfig"):
            sky._source_dtype()

    def test_ensure_dtypes_healpix_polarization(self):
        """HEALPix Q/U/V maps in float64 with fast() precision are cast to float32."""
        precision = PrecisionConfig.fast()
        sky = _make_healpix_only_model(
            nside=8,
            n_freq=2,
            with_q=True,
            with_u=True,
            with_v=True,
            precision=precision,
        )
        hp_dt = precision.sky_model.get_dtype("healpix_maps")
        assert sky._healpix_maps.dtype == hp_dt
        assert sky._healpix_q_maps.dtype == hp_dt
        assert sky._healpix_u_maps.dtype == hp_dt
        assert sky._healpix_v_maps.dtype == hp_dt


# ---------------------------------------------------------------------------
# TestPropertiesExpanded
# ---------------------------------------------------------------------------


class TestPropertiesExpanded:
    def test_n_frequencies_with_obs(self):
        """Healpix model → n_frequencies returns correct count."""
        sky = _make_healpix_only_model(n_freq=5)
        assert sky.n_frequencies == 5

    def test_pygdsm_model_default_none(self, precision):
        """Default model → pygdsm_model property is None."""
        sky = SkyModel.from_test_sources(
            num_sources=5,
            flux_range=(1, 5),
            dec_deg=-30,
            spectral_index=-0.7,
            precision=precision,
        )
        assert sky.pygdsm_model is None

    def test_n_sources_healpix_only(self):
        """Healpix-only model → n_sources returns npix."""
        nside = 8
        sky = _make_healpix_only_model(nside=nside)
        assert sky.n_sources == hp.nside2npix(nside)

    def test_has_polarized_healpix_true(self):
        """Model with Q maps → has_polarized_healpix_maps is True."""
        sky = _make_healpix_only_model(with_q=True)
        assert sky.has_polarized_healpix_maps is True


# ---------------------------------------------------------------------------
# TestFilterRegionHealpixPolarization
# ---------------------------------------------------------------------------


class TestFilterRegionHealpixPolarization:
    def test_filter_healpix_with_qmaps(self):
        """Cone filter on healpix model with Q/U maps → out-of-region pixels zeroed in Q/U."""
        sky = _make_healpix_only_model(
            nside=8,
            n_freq=2,
            with_q=True,
            with_u=True,
            fill_value=5.0,
        )
        # Before filtering, all Q pixels are 0.01
        assert np.all(sky._healpix_q_maps == 0.01)

        # Apply a small cone so most pixels are outside
        region = SkyRegion.cone(ra_deg=0.0, dec_deg=0.0, radius_deg=10.0)
        filtered = sky.filter_region(region)

        # Some Q pixels should now be 0 (outside the cone)
        assert np.any(filtered._healpix_q_maps == 0.0)
        # Some Q pixels should still be 0.01 (inside the cone)
        assert np.any(filtered._healpix_q_maps == 0.01)
        # Same for U
        assert np.any(filtered._healpix_u_maps == 0.0)
        assert np.any(filtered._healpix_u_maps == 0.02)


# ---------------------------------------------------------------------------
# TestWithHealpixMapsValidation
# ---------------------------------------------------------------------------


class TestWithHealpixMapsValidation:
    def test_no_point_sources_raises(self):
        """Healpix-only model → with_healpix_maps() raises ValueError (no point sources)."""
        sky = _make_healpix_only_model()
        freqs = np.array([100e6, 101e6])
        with pytest.raises(ValueError, match="No point sources"):
            sky.with_healpix_maps(nside=16, frequencies=freqs)

    def test_both_freq_args_raises(self, precision):
        """Pass both frequencies and obs_frequency_config → ValueError."""
        sky = SkyModel.from_test_sources(
            num_sources=5,
            flux_range=(1, 5),
            dec_deg=-30,
            spectral_index=-0.7,
            precision=precision,
        ).with_frequency(100e6)
        freqs = np.array([100e6, 101e6])
        obs_config = {
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 2.0,
            "frequency_unit": "MHz",
        }
        with pytest.raises(ValueError, match="not both"):
            sky.with_healpix_maps(
                nside=8, frequencies=freqs, obs_frequency_config=obs_config
            )

    def test_neither_freq_raises(self, precision):
        """Pass neither frequencies nor obs_frequency_config → ValueError."""
        sky = SkyModel.from_test_sources(
            num_sources=5,
            flux_range=(1, 5),
            dec_deg=-30,
            spectral_index=-0.7,
            precision=precision,
        ).with_frequency(100e6)
        with pytest.raises(ValueError, match="required"):
            sky.with_healpix_maps(nside=8)


# ---------------------------------------------------------------------------
# TestWithRepresentationExpanded2
# ---------------------------------------------------------------------------


class TestWithRepresentationExpanded2:
    def test_healpix_map_no_maps_raises(self, precision):
        """Point-source model → with_representation('healpix_map') raises ValueError."""
        sky = SkyModel.from_test_sources(
            num_sources=5,
            flux_range=(1, 5),
            dec_deg=-30,
            spectral_index=-0.7,
            precision=precision,
        )
        with pytest.raises(ValueError, match="Cannot convert"):
            sky.with_representation("healpix_map")

    def test_healpix_multifreq_no_maps_raises(self, precision):
        """Point-source model → with_representation('healpix_multifreq') raises ValueError."""
        sky = SkyModel.from_test_sources(
            num_sources=5,
            flux_range=(1, 5),
            dec_deg=-30,
            spectral_index=-0.7,
            precision=precision,
        )
        with pytest.raises(ValueError, match="Cannot convert"):
            sky.with_representation("healpix_multifreq")


# ---------------------------------------------------------------------------
# TestAsPointSourceDictsExpanded2
# ---------------------------------------------------------------------------


class TestAsPointSourceDictsExpanded2:
    def test_healpix_only_model(self):
        """Healpix-only model → as_point_source_dicts(frequency=...) returns dicts from temporary conversion."""
        sky = _make_healpix_only_model(nside=8, n_freq=3, fill_value=100.0)
        dicts = sky.as_point_source_dicts(frequency=100e6)
        assert isinstance(dicts, list)
        assert len(dicts) > 0
        assert all("coords" in d and "flux" in d for d in dicts)

    def test_empty_model_returns_empty(self):
        """_empty_sky() → as_point_source_dicts() returns []."""
        sky = SkyModel._empty_sky("test")
        result = sky.as_point_source_dicts()
        assert result == []

    def test_all_filtered_by_flux_limit(self, precision):
        """Model with max flux 10 Jy, flux_limit=100 → returns []."""
        sky = SkyModel.from_test_sources(
            num_sources=10,
            flux_range=(1.0, 10.0),
            dec_deg=-30,
            spectral_index=-0.7,
            precision=precision,
        )
        result = sky.as_point_source_dicts(flux_limit=100.0)
        assert result == []


# ---------------------------------------------------------------------------
# TestBackwardCompatStubs
# ---------------------------------------------------------------------------


class TestFrequencyResolutionExpanded:
    def test_resolve_frequency_index_no_freqs_raises(self, precision):
        """Model without observation_frequencies → _resolve_frequency_index() raises ValueError."""
        sky = SkyModel.from_test_sources(
            num_sources=5,
            flux_range=(1, 5),
            dec_deg=-30,
            spectral_index=-0.7,
            precision=precision,
        )
        with pytest.raises(ValueError, match="No observation frequencies"):
            sky._resolve_frequency_index(100e6)


# ---------------------------------------------------------------------------
# TestMultifreqStokesAccessors
# ---------------------------------------------------------------------------


class TestMultifreqStokesAccessors:
    def test_get_multifreq_stokes_maps_returns_tuple(self):
        """Healpix model → get_multifreq_stokes_maps() returns 6-element tuple."""
        sky = _make_healpix_only_model(
            nside=8,
            n_freq=3,
            with_q=True,
            with_u=True,
            with_v=True,
        )
        result = sky.get_multifreq_stokes_maps()
        assert isinstance(result, tuple)
        assert len(result) == 6
        i_maps, q_maps, u_maps, v_maps, nside, freqs = result
        assert i_maps.shape == (3, hp.nside2npix(8))
        assert q_maps is not None
        assert u_maps is not None
        assert v_maps is not None
        assert nside == 8
        assert len(freqs) == 3

    def test_get_multifreq_stokes_no_maps_raises(self, precision):
        """Point-source model → get_multifreq_stokes_maps() raises ValueError."""
        sky = SkyModel.from_test_sources(
            num_sources=5,
            flux_range=(1, 5),
            dec_deg=-30,
            spectral_index=-0.7,
            precision=precision,
        )
        with pytest.raises(ValueError, match="No multi-frequency"):
            sky.get_multifreq_stokes_maps()


# ---------------------------------------------------------------------------
# TestSerializationExpanded
# ---------------------------------------------------------------------------


class TestSerializationExpanded:
    def test_to_pyradiosky_point_with_polarization(self, precision):
        """Model with non-zero Q/U/V → pyradiosky stokes array rows 1-3 non-zero."""
        sky = SkyModel.from_test_sources(
            num_sources=10,
            flux_range=(1, 5),
            dec_deg=-30,
            spectral_index=-0.7,
            polarization_fraction=0.1,
            polarization_angle_deg=45.0,
            stokes_v_fraction=0.02,
            precision=precision,
        ).with_frequency(100e6)
        psky = sky._to_pyradiosky()
        stokes = psky.stokes.value  # shape (4, 1, N)
        assert np.any(stokes[1] != 0), "Q should be non-zero"
        assert np.any(stokes[2] != 0), "U should be non-zero"
        assert np.any(stokes[3] != 0), "V should be non-zero"

    def test_save_empty_model_raises(self):
        """Empty model → save() raises ValueError (via _to_pyradiosky)."""
        sky = SkyModel._empty_sky("test")
        with pytest.raises(ValueError, match="empty"):
            sky._to_pyradiosky()

    def test_save_warns_spectral_coeffs(self, precision, tmp_path):
        """Model with 2-term spectral_coeffs → save warns 'multi-term spectral coefficients'."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux_ref=np.array([1.0]),
            spectral_coeffs=np.array([[-0.7, -0.1]]),
            frequency=100e6,
            precision=precision,
        )
        path = str(tmp_path / "spec_test.skyh5")
        with pytest.warns(UserWarning, match="multi-term spectral coefficients"):
            sky.save(path)


# ---------------------------------------------------------------------------
# TestParallelLoading
# ---------------------------------------------------------------------------


class TestParallelLoading:
    def test_load_parallel_basic(self, precision):
        """load_parallel with a mocked loader returns a list of SkyModel."""
        from unittest.mock import patch

        def mock_loader(**kwargs):
            return SkyModel.from_test_sources(
                num_sources=5,
                flux_range=(1, 5),
                dec_deg=-30,
                spectral_index=-0.7,
                precision=kwargs.get("precision"),
            )

        with patch("rrivis.core.sky._registry.get_loader", return_value=mock_loader):
            results = SkyModel.load_parallel(
                loaders=[
                    ("mock_catalog", {}),
                    ("mock_catalog2", {}),
                ],
                precision=precision,
            )
        assert isinstance(results, list)
        assert len(results) == 2
        for sky in results:
            assert isinstance(sky, SkyModel)
            assert sky.n_sources == 5


# ---------------------------------------------------------------------------
# TestFromFreqDictMaps
# ---------------------------------------------------------------------------


class TestFromFreqDictMaps:
    def test_from_freq_dict_maps_polarization(self):
        """Pass i/q/u/v dicts → all 4 stacked as 2D arrays."""
        nside = 4
        npix = hp.nside2npix(nside)
        freqs = [100e6, 200e6]
        i_maps = {f: np.ones(npix) * (f / 1e6) for f in freqs}
        q_maps = {f: np.ones(npix) * 0.01 for f in freqs}
        u_maps = {f: np.ones(npix) * 0.02 for f in freqs}
        v_maps = {f: np.ones(npix) * 0.005 for f in freqs}

        sky = SkyModel._from_freq_dict_maps(
            i_maps=i_maps,
            q_maps=q_maps,
            u_maps=u_maps,
            v_maps=v_maps,
            nside=nside,
            _native_format="healpix",
            frequency=100e6,
            model_name="test_dict_maps",
        )
        assert sky._healpix_maps.shape == (2, npix)
        assert sky._healpix_q_maps is not None
        assert sky._healpix_q_maps.shape == (2, npix)
        assert sky._healpix_u_maps is not None
        assert sky._healpix_u_maps.shape == (2, npix)
        assert sky._healpix_v_maps is not None
        assert sky._healpix_v_maps.shape == (2, npix)
        # Frequencies should be sorted
        assert np.isclose(sky._observation_frequencies[0], 100e6)
        assert np.isclose(sky._observation_frequencies[1], 200e6)
        # I map at 100 MHz should have values of 100.0
        assert np.allclose(sky._healpix_maps[0], 100.0)


# ---------------------------------------------------------------------------
# TestFromTestSourcesEdgeCases
# ---------------------------------------------------------------------------


class TestFromTestSourcesEdgeCases:
    def test_invalid_distribution_raises(self, precision):
        """distribution='invalid' → ValueError."""
        with pytest.raises(ValueError, match="distribution"):
            SkyModel.from_test_sources(
                num_sources=5,
                flux_range=(1, 5),
                dec_deg=-30,
                spectral_index=-0.7,
                distribution="invalid",
                precision=precision,
            )

    def test_single_source(self, precision):
        """num_sources=1 uniform → RA=0, flux=(min+max)/2."""
        sky = SkyModel.from_test_sources(
            num_sources=1,
            flux_range=(2.0, 8.0),
            dec_deg=-30,
            spectral_index=-0.7,
            precision=precision,
        )
        assert sky.n_sources == 1
        assert np.isclose(sky._ra_rad[0], 0.0)
        assert np.isclose(sky._flux_ref[0], 5.0)


# ---------------------------------------------------------------------------
# TestFromPointSourcesSpectralCoeffs
# ---------------------------------------------------------------------------


class TestFromPointSourcesSpectralCoeffs:
    def test_mixed_spectral_coeffs(self, precision):
        """Sources where some have spectral_coeffs, others don't → padded array."""
        src1 = {
            "coords": SkyCoord(ra=0.0, dec=0.0, unit="deg", frame="icrs"),
            "flux": 1.0,
            "spectral_index": -0.7,
            "spectral_coeffs": [-0.7, -0.1],
        }
        src2 = {
            "coords": SkyCoord(ra=10.0, dec=0.0, unit="deg", frame="icrs"),
            "flux": 2.0,
            "spectral_index": -0.8,
            # No spectral_coeffs key
        }
        sky = SkyModel.from_point_sources([src1, src2], precision=precision)
        assert sky._spectral_coeffs is not None
        assert sky._spectral_coeffs.shape == (2, 2)
        # First source: coeffs as given
        assert np.isclose(sky._spectral_coeffs[0, 0], -0.7)
        assert np.isclose(sky._spectral_coeffs[0, 1], -0.1)
        # Second source: first coeff is spectral_index default, second is 0
        assert np.isclose(sky._spectral_coeffs[1, 0], -0.8)
        assert np.isclose(sky._spectral_coeffs[1, 1], 0.0)


# ---------------------------------------------------------------------------
# TestCatalogAccessors
# ---------------------------------------------------------------------------


class TestCatalogAccessors:
    def test_list_all_models_keys(self):
        """list_all_models() has keys 'diffuse', 'point_catalogs', 'racs'."""
        models = SkyModel.list_all_models()
        assert "diffuse" in models
        assert "point_catalogs" in models
        assert "racs" in models
        # Each should be a dict
        assert isinstance(models["diffuse"], dict)
        assert isinstance(models["point_catalogs"], dict)
        assert isinstance(models["racs"], dict)

    def test_get_catalog_info_vizier(self):
        """get_catalog_info('nvss') → dict with 'freq_mhz' key."""
        info = SkyModel.get_catalog_info("nvss")
        assert isinstance(info, dict)
        assert "freq_mhz" in info
        assert info["freq_mhz"] == 1400.0

    def test_get_catalog_info_diffuse(self):
        """get_catalog_info('gsm2008') → dict with 'parameters' key."""
        info = SkyModel.get_catalog_info("gsm2008")
        assert isinstance(info, dict)
        assert "parameters" in info
        assert isinstance(info["parameters"], dict)

    def test_get_catalog_info_unknown_raises(self):
        """get_catalog_info('nonexistent') → ValueError."""
        with pytest.raises(ValueError, match="Unknown catalog key"):
            SkyModel.get_catalog_info("nonexistent")


# ---------------------------------------------------------------------------
# TestReprExpanded
# ---------------------------------------------------------------------------


class TestReprExpanded:
    def test_repr_healpix_with_polarization(self):
        """Healpix model with Q maps → repr contains 'IQ' in stokes components."""
        sky = _make_healpix_only_model(nside=8, n_freq=2, with_q=True)
        r = repr(sky)
        assert "IQ" in r
        assert "healpix_multifreq" in r
        assert "nside=8" in r
