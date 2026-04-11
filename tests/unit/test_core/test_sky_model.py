"""Tests for rrivis.core.sky.model — SkyModel class and module-level helpers."""

import dataclasses

import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import SkyModel, SkyRegion
from rrivis.core.sky.discovery import (
    estimate_healpix_memory,
    get_catalog_info,
    list_all_models,
)
from rrivis.core.sky.model import SkyFormat
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
            test_sky.reference_frequency = 100e6

    def test_with_reference_frequency_returns_new(self, test_sky):
        """with_reference_frequency returns a new instance with updated reference_frequency."""
        sky2 = test_sky.with_reference_frequency(100e6)
        assert sky2 is not test_sky
        assert sky2.reference_frequency == 100e6
        assert test_sky.reference_frequency is None


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
                _flux=np.zeros(10),
                _spectral_index=np.zeros(10),
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
                _flux=np.zeros(5),
                _spectral_index=np.zeros(5),
                _stokes_q=np.zeros(5),
                _stokes_u=None,
                _stokes_v=np.zeros(5),
                _precision=precision,
            )

    def test_invalid_native_format(self, precision):
        """TypeError for non-enum _native_format."""
        with pytest.raises(TypeError, match="_native_format must be a SkyFormat enum"):
            SkyModel(
                _ra_rad=np.zeros(5),
                _dec_rad=np.zeros(5),
                _flux=np.zeros(5),
                _spectral_index=np.zeros(5),
                _stokes_q=np.zeros(5),
                _stokes_u=np.zeros(5),
                _stokes_v=np.zeros(5),
                _native_format="invalid",
                _precision=precision,
            )

    def test_mismatched_stokes_length_raises(self, precision):
        """ValueError when Stokes Q/U/V length doesn't match core arrays."""
        with pytest.raises(ValueError, match="_stokes_q has length 3"):
            SkyModel(
                _ra_rad=np.zeros(5),
                _dec_rad=np.zeros(5),
                _flux=np.zeros(5),
                _spectral_index=np.zeros(5),
                _stokes_q=np.zeros(3),
                _stokes_u=np.zeros(3),
                _stokes_v=np.zeros(3),
                _precision=precision,
            )

    def test_mismatched_rotation_measure_length_raises(self, precision):
        """ValueError when rotation measure length doesn't match core arrays."""
        with pytest.raises(ValueError, match="_rotation_measure has length 2"):
            SkyModel(
                _ra_rad=np.zeros(5),
                _dec_rad=np.zeros(5),
                _flux=np.zeros(5),
                _spectral_index=np.zeros(5),
                _stokes_q=np.zeros(5),
                _stokes_u=np.zeros(5),
                _stokes_v=np.zeros(5),
                _rotation_measure=np.zeros(2),
                _precision=precision,
            )

    def test_partial_morphology_raises(self, precision):
        """ValueError when only some morphology fields are set."""
        with pytest.raises(ValueError, match="Morphology fields"):
            SkyModel(
                _ra_rad=np.zeros(5),
                _dec_rad=np.zeros(5),
                _flux=np.zeros(5),
                _spectral_index=np.zeros(5),
                _stokes_q=np.zeros(5),
                _stokes_u=np.zeros(5),
                _stokes_v=np.zeros(5),
                _major_arcsec=np.zeros(5),
                _precision=precision,
            )

    def test_mismatched_spectral_coeffs_raises(self, precision):
        """ValueError when spectral_coeffs first dim doesn't match sources."""
        with pytest.raises(ValueError, match="_spectral_coeffs first dimension"):
            SkyModel(
                _ra_rad=np.zeros(5),
                _dec_rad=np.zeros(5),
                _flux=np.zeros(5),
                _spectral_index=np.zeros(5),
                _stokes_q=np.zeros(5),
                _stokes_u=np.zeros(5),
                _stokes_v=np.zeros(5),
                _spectral_coeffs=np.zeros((3, 2)),
                _precision=precision,
            )

    def test_valid_optional_arrays_accepted(self, precision):
        """All optional arrays with correct lengths are accepted."""
        sky = SkyModel(
            _ra_rad=np.zeros(5),
            _dec_rad=np.zeros(5),
            _flux=np.ones(5),
            _spectral_index=np.full(5, -0.7),
            _stokes_q=np.zeros(5),
            _stokes_u=np.zeros(5),
            _stokes_v=np.zeros(5),
            _rotation_measure=np.zeros(5),
            _major_arcsec=np.zeros(5),
            _minor_arcsec=np.zeros(5),
            _pa_deg=np.zeros(5),
            _spectral_coeffs=np.zeros((5, 2)),
            _precision=precision,
        )
        assert sky.n_point_sources == 5


# ---------------------------------------------------------------------------
# from_test_sources
# ---------------------------------------------------------------------------


class TestFromTestSources:
    def test_from_test_sources_basic(self, test_sky):
        """Uniform distribution: correct count, mode, coordinate ranges, flux range."""
        assert test_sky.n_point_sources == 50
        assert test_sky.mode == "point_sources"
        assert test_sky.native_format == "point_sources"

        # RA in [0, 2pi]
        assert np.all(test_sky.ra_rad >= 0)
        assert np.all(test_sky.ra_rad <= 2 * np.pi)

        # Dec all approx -30 degrees in radians
        expected_dec = np.deg2rad(-30.0)
        assert np.allclose(test_sky.dec_rad, expected_dec, atol=1e-6)

        # Flux in [1.0, 10.0]
        assert np.all(test_sky.flux >= 1.0)
        assert np.all(test_sky.flux <= 10.0)

        # Spectral index all -0.7
        assert np.allclose(test_sky.spectral_index, -0.7)

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
        assert sky.n_point_sources == 100

        # Dec values should vary (not all identical)
        dec_unique = np.unique(sky.dec_rad)
        assert len(dec_unique) > 1

    def test_from_test_sources_defaults(self, precision):
        """from_test_sources works with all-default parameters."""
        sky = SkyModel.from_test_sources(precision=precision)
        assert sky.n_point_sources == 100
        assert sky.spectral_index is not None
        np.testing.assert_allclose(sky.spectral_index, -0.7)

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
        assert np.any(sky.stokes_q != 0)
        assert np.any(sky.stokes_u != 0)
        assert np.any(sky.stokes_v != 0)


# ---------------------------------------------------------------------------
# from_arrays / from_point_sources / as_point_source_arrays
# ---------------------------------------------------------------------------


class TestPointSourcesRoundtrip:
    def test_from_arrays(self, precision):
        """from_arrays creates a valid SkyModel."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0, 1.0, 2.0]),
            dec_rad=np.array([0.0, 0.1, -0.1]),
            flux=np.array([1.0, 2.0, 3.0]),
            reference_frequency=100e6,
            model_name="test_arrays",
            precision=precision,
        )
        assert sky.n_point_sources == 3
        assert sky.reference_frequency == 100e6
        assert sky.model_name == "test_arrays"
        assert np.allclose(sky.spectral_index, -0.7)  # default

    def test_from_point_sources_roundtrip(self, test_sky, precision):
        """Round-trip: test sources -> from_point_sources via arrays."""
        arrays = test_sky.as_point_source_arrays()
        # from_point_sources expects list[dict] — test array keys instead
        assert arrays["ra_rad"] is not None
        assert len(arrays["ra_rad"]) == test_sky.n_point_sources
        np.testing.assert_allclose(arrays["ra_rad"], test_sky.ra_rad, atol=1e-10)
        np.testing.assert_allclose(arrays["dec_rad"], test_sky.dec_rad, atol=1e-10)
        np.testing.assert_allclose(arrays["flux"], test_sky.flux, atol=1e-10)

    def test_as_point_source_arrays_keys(self, test_sky):
        """as_point_source_arrays() has the expected keys."""
        arrays = test_sky.as_point_source_arrays()
        assert len(arrays["ra_rad"]) == 50
        for key in (
            "ra_rad",
            "dec_rad",
            "flux",
            "spectral_index",
            "stokes_q",
            "stokes_u",
            "stokes_v",
        ):
            assert key in arrays
            assert isinstance(arrays[key], np.ndarray)
        assert "ref_freq" in arrays
        assert isinstance(arrays["ref_freq"], np.ndarray)

    def test_as_point_source_arrays_flux_limit(self, precision):
        """flux_limit filters out low-flux sources."""
        sky = SkyModel.from_test_sources(
            num_sources=50,
            flux_range=(0.5, 5.0),
            dec_deg=-30.0,
            spectral_index=-0.7,
            precision=precision,
        )
        arrays = sky.as_point_source_arrays(flux_limit=2.0)
        assert np.all(arrays["flux"] >= 2.0)

    def test_as_point_source_arrays_no_mutation(self, test_sky):
        """as_point_source_arrays does not mutate the original SkyModel."""
        original_ra = test_sky.ra_rad.copy()
        _ = test_sky.as_point_source_arrays()
        assert np.array_equal(test_sky.ra_rad, original_ra)


# ---------------------------------------------------------------------------
# Mode & Conversion (immutable)
# ---------------------------------------------------------------------------


class TestModeAndConversion:
    def test_mode_property(self, test_sky, obs_freq_config):
        """Mode switches from point_sources to healpix_map after conversion."""
        assert test_sky.mode == "point_sources"
        sky = test_sky.with_reference_frequency(100e6)
        sky = sky.with_healpix_maps(nside=32, obs_frequency_config=obs_freq_config)
        assert sky.mode == "healpix_map"

    def test_with_healpix_maps(self, test_sky, obs_freq_config):
        """with_healpix_maps populates HEALPix and preserves point-source data."""
        sky = test_sky.with_reference_frequency(100e6)
        sky = sky.with_healpix_maps(nside=32, obs_frequency_config=obs_freq_config)
        assert sky.has_multifreq_maps is True
        assert sky.n_frequencies == 6
        assert sky.healpix_nside == 32
        assert sky.mode == "healpix_map"
        # PS arrays are preserved (dual representation)
        assert sky.ra_rad is not None
        assert sky.has_point_sources is True

    def test_with_healpix_maps_returns_new(self, test_sky, obs_freq_config):
        """with_healpix_maps returns a different object, original unchanged."""
        sky = test_sky.with_reference_frequency(100e6)
        sky2 = sky.with_healpix_maps(nside=32, obs_frequency_config=obs_freq_config)
        assert sky2 is not sky
        assert sky.has_multifreq_maps is False
        assert sky2.has_multifreq_maps is True
        # Original has PS only, new model has both representations
        assert sky.ra_rad is not None
        assert sky2.ra_rad is not None
        assert sky2.mode == "healpix_map"

    def test_with_healpix_maps_no_ref_frequency_raises(self, test_sky, obs_freq_config):
        """with_healpix_maps raises when no ref_frequency is available."""
        assert test_sky.reference_frequency is None
        with pytest.raises(ValueError, match="ref_frequency must be provided"):
            test_sky.with_healpix_maps(nside=32, obs_frequency_config=obs_freq_config)

    def test_with_healpix_maps_explicit_ref_frequency(self, test_sky, obs_freq_config):
        """with_healpix_maps succeeds when ref_frequency is passed explicitly,
        even if the model has reference_frequency=None (test_sources_healpix path)."""
        assert test_sky.reference_frequency is None
        sky = test_sky.with_healpix_maps(
            nside=8,
            obs_frequency_config=obs_freq_config,
            ref_frequency=100e6,
        )
        assert sky.has_multifreq_maps is True
        assert sky.healpix_nside == 8

    def test_get_map_at_frequency(self, test_sky, obs_freq_config):
        """get_map_at_frequency returns a valid HEALPix map."""
        sky = test_sky.with_reference_frequency(100e6).with_healpix_maps(
            nside=32, obs_frequency_config=obs_freq_config
        )
        m = sky.get_map_at_frequency(100e6)
        assert isinstance(m, np.ndarray)
        assert len(m) == hp.nside2npix(32)
        assert not np.any(np.isnan(m))

    def test_get_map_at_frequency_nearest(self, test_sky, obs_freq_config):
        """Requesting a non-exact frequency returns the nearest available map."""
        sky = test_sky.with_reference_frequency(100e6).with_healpix_maps(
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
        assert result.ra_rad is not None

    def test_with_representation_healpix_requires_maps(self, test_sky):
        """Requesting healpix_map without maps raises ValueError."""
        with pytest.raises(ValueError, match="Cannot convert"):
            test_sky.with_representation("healpix_map")


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
        assert combined.n_point_sources == 50
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

        assert filtered.n_point_sources < 100

        # All filtered sources should be within 30 degrees of the center
        center = SkyCoord(ra=180.0, dec=0.0, unit="deg", frame="icrs")
        coords = SkyCoord(
            ra=filtered.ra_rad, dec=filtered.dec_rad, unit="rad", frame="icrs"
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

        assert filtered.n_point_sources < 100
        assert filtered.n_point_sources > 0


# ---------------------------------------------------------------------------
# pixel_solid_angle / pixel_coords
# ---------------------------------------------------------------------------


class TestPixelProperties:
    def test_pixel_solid_angle(self, test_sky, obs_freq_config):
        """pixel_solid_angle is 4pi/npix for the given nside."""
        sky = test_sky.with_reference_frequency(100e6).with_healpix_maps(
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
        sky = test_sky.with_reference_frequency(100e6).with_healpix_maps(
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
        result = estimate_healpix_memory(nside=64, n_frequencies=10)
        assert "npix" in result
        assert "total_bytes" in result
        assert "total_mb" in result
        assert "total_gb" in result
        assert "bytes_per_map" in result
        assert result["total_bytes"] > 0
        assert result["npix"] == hp.nside2npix(64)


# ---------------------------------------------------------------------------
# empty_sky
# ---------------------------------------------------------------------------


class TestEmptySky:
    def test_empty_sky(self):
        """empty_sky creates a model with zero sources in point_sources mode."""
        sky = SkyModel.empty_sky("test")
        assert sky.n_point_sources == 0
        assert sky.mode == "point_sources"
        assert sky.model_name == "test"

    def test_empty_sky_with_reference_frequency(self):
        """empty_sky accepts a reference_frequency parameter."""
        sky = SkyModel.empty_sky("test", reference_frequency=100e6)
        assert sky.reference_frequency == 100e6


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
        assert len(freqs) == 6
        assert np.isclose(freqs[0], 100e6)
        assert np.isclose(freqs[1], 101e6)
        assert np.isclose(freqs[-1], 105e6)

    def test_parse_frequency_config_ghz(self):
        """Parse GHz config: correct Hz values."""
        config = {
            "starting_frequency": 1.0,
            "frequency_interval": 0.1,
            "frequency_bandwidth": 0.5,
            "frequency_unit": "GHz",
        }
        freqs = parse_frequency_config(config)
        assert len(freqs) == 6
        assert np.isclose(freqs[0], 1.0e9)
        assert np.isclose(freqs[-1], 1.5e9)


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
        freqs = np.array([100e6, 150e6, 200e6])
        with pytest.raises(ValueError, match="does not match"):
            SkyModel(
                _healpix_maps=i_maps,
                _healpix_q_maps=q_maps,
                _healpix_nside=nside,
                _observation_frequencies=freqs,
            )

    def test_healpix_maps_without_observation_frequencies_raises(self):
        """ValueError when healpix maps set but observation_frequencies missing."""
        nside = 8
        npix = hp.nside2npix(nside)
        maps = np.zeros((3, npix), dtype=np.float32)
        with pytest.raises(ValueError, match="_observation_frequencies"):
            SkyModel(_healpix_maps=maps, _healpix_nside=nside)

    def test_healpix_maps_frequency_count_mismatch_raises(self):
        """ValueError when observation_frequencies length != n_freq."""
        nside = 8
        npix = hp.nside2npix(nside)
        maps = np.zeros((3, npix), dtype=np.float32)
        freqs = np.array([100e6, 150e6])  # 2 != 3
        with pytest.raises(ValueError, match="must match"):
            SkyModel(
                _healpix_maps=maps,
                _healpix_nside=nside,
                _observation_frequencies=freqs,
            )

    def test_invalid_brightness_conversion_raises(self, precision):
        """ValueError for invalid brightness_conversion string."""
        with pytest.raises(ValueError, match="brightness_conversion"):
            SkyModel(
                _ra_rad=np.zeros(5),
                _dec_rad=np.zeros(5),
                _flux=np.zeros(5),
                _spectral_index=np.zeros(5),
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
            _flux=np.zeros(0),
            _spectral_index=np.zeros(0),
            _stokes_q=np.zeros(0),
            _stokes_u=np.zeros(0),
            _stokes_v=np.zeros(0),
            _precision=precision,
        )
        assert sky.n_point_sources == 0


# ---------------------------------------------------------------------------
# with_healpix_maps expanded
# ---------------------------------------------------------------------------


class TestWithHealpixMapsExpanded:
    def test_with_healpix_maps_frequencies_array(self, test_sky, precision):
        """Pass frequencies array directly (not obs_frequency_config)."""
        sky = test_sky.with_reference_frequency(100e6)
        freqs = np.array([100e6, 101e6, 102e6])
        result = sky.with_healpix_maps(nside=16, frequencies=freqs)
        assert result.n_frequencies == 3
        assert result.healpix_nside == 16

    def test_original_unchanged_after_with_healpix(self, test_sky, obs_freq_config):
        """Verify the original model has no healpix maps after calling with_healpix_maps."""
        sky = test_sky.with_reference_frequency(100e6)
        sky2 = sky.with_healpix_maps(nside=16, obs_frequency_config=obs_freq_config)
        assert sky.healpix_maps is None
        assert sky2.healpix_maps is not None

    def test_multiple_nside_values(self, test_sky, precision):
        """Different nside values produce correct pixel counts."""
        sky = test_sky.with_reference_frequency(100e6)
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
        sky = test_sky.with_reference_frequency(100e6).with_healpix_maps(
            nside=16, obs_frequency_config=obs_freq_config
        )
        # This model has healpix maps AND point-source arrays (from original).
        # Create a healpix-only model to test conversion.
        healpix_only = SkyModel(
            _healpix_maps=sky.healpix_maps,
            _healpix_nside=sky.healpix_nside,
            _observation_frequencies=sky.observation_frequencies,
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
        )
        result = healpix_only.with_representation("point_sources", frequency=100e6)
        assert result.ra_rad is not None
        assert len(result.ra_rad) > 0


# ---------------------------------------------------------------------------
# as_point_source_arrays expanded
# ---------------------------------------------------------------------------


class TestAsPointSourceArraysExpanded:
    def test_includes_rotation_measure(self, precision):
        """Model with RM → arrays have 'rotation_measure' key."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0, 1.0]),
            dec_rad=np.array([0.0, 0.1]),
            flux=np.array([1.0, 2.0]),
            rotation_measure=np.array([5.0, 10.0]),
            precision=precision,
        )
        arrays = sky.as_point_source_arrays()
        assert arrays["rotation_measure"] is not None
        assert arrays["rotation_measure"][0] == 5.0

    def test_includes_gaussian_morphology(self, precision):
        """Model with morphology → arrays have morphology keys."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux=np.array([1.0]),
            major_arcsec=np.array([10.0]),
            minor_arcsec=np.array([5.0]),
            pa_deg=np.array([45.0]),
            precision=precision,
        )
        arrays = sky.as_point_source_arrays()
        assert arrays["major_arcsec"][0] == 10.0
        assert arrays["minor_arcsec"][0] == 5.0
        assert arrays["pa_deg"][0] == 45.0

    def test_includes_spectral_coeffs(self, precision):
        """Model with spectral coefficients → arrays have 'spectral_coeffs' key."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux=np.array([1.0]),
            spectral_coeffs=np.array([[-0.7, -0.1]]),
            precision=precision,
        )
        arrays = sky.as_point_source_arrays()
        assert arrays["spectral_coeffs"] is not None
        assert np.isclose(arrays["spectral_coeffs"][0, 0], -0.7)


# ---------------------------------------------------------------------------
# from_arrays expanded
# ---------------------------------------------------------------------------


class TestFromArraysExpanded:
    def test_from_arrays_with_all_optional_fields(self, precision):
        """Pass all optional fields → all stored correctly."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0, 1.0]),
            dec_rad=np.array([0.0, 0.1]),
            flux=np.array([1.0, 2.0]),
            spectral_index=np.array([-0.5, -0.9]),
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
        assert sky.n_point_sources == 2
        assert np.isclose(sky.rotation_measure[0], 5.0)
        assert np.isclose(sky.major_arcsec[0], 10.0)
        assert sky.spectral_coeffs.shape == (2, 2)

    def test_from_arrays_single_source(self, precision):
        """Single source (N=1) → valid model."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([np.pi]),
            dec_rad=np.array([0.0]),
            flux=np.array([5.0]),
            precision=precision,
        )
        assert sky.n_point_sources == 1

    def test_from_arrays_empty(self, precision):
        """Empty arrays (N=0) → valid empty model."""
        sky = SkyModel.from_arrays(
            ra_rad=np.zeros(0),
            dec_rad=np.zeros(0),
            flux=np.zeros(0),
            precision=precision,
        )
        assert sky.n_point_sources == 0

    def test_from_arrays_default_spectral_index(self, precision):
        """No spectral_index passed → defaults to -0.7."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0, 1.0, 2.0]),
            dec_rad=np.array([0.0, 0.0, 0.0]),
            flux=np.array([1.0, 2.0, 3.0]),
            precision=precision,
        )
        assert np.allclose(sky.spectral_index, -0.7)


# ---------------------------------------------------------------------------
# from_point_sources expanded
# ---------------------------------------------------------------------------


class TestFromPointSourcesExpanded:
    def test_empty_list(self, precision):
        """Empty list → empty model."""
        with pytest.warns(DeprecationWarning, match="from_point_sources"):
            sky = SkyModel.from_point_sources([], precision=precision)
        assert sky.n_point_sources == 0

    def test_default_spectral_index(self, precision):
        """No 'spectral_index' in dict → defaults to -0.7."""
        src = {
            "coords": SkyCoord(ra=180.0, dec=-30.0, unit="deg", frame="icrs"),
            "flux": 5.0,
        }
        with pytest.warns(DeprecationWarning, match="from_point_sources"):
            sky = SkyModel.from_point_sources([src], precision=precision)
        assert np.isclose(sky.spectral_index[0], -0.7)

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
        with pytest.warns(DeprecationWarning, match="from_point_sources"):
            sky = SkyModel.from_point_sources([src], precision=precision)
        assert np.isclose(sky.rotation_measure[0], 15.0)
        assert np.isclose(sky.major_arcsec[0], 20.0)
        assert sky.spectral_coeffs is not None


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_pyradiosky_point_source(self, test_sky):
        """Point-source model → pyradiosky SkyModel with correct structure."""
        sky = test_sky.with_reference_frequency(100e6)
        psky = sky._to_pyradiosky()
        assert psky.component_type == "point"
        assert psky.Ncomponents == 50
        assert psky.spectral_type == "spectral_index"

    def test_to_pyradiosky_healpix(self, test_sky, obs_freq_config):
        """HEALPix-only model → pyradiosky SkyModel with healpix component."""
        sky = test_sky.with_reference_frequency(100e6).with_healpix_maps(
            nside=8, obs_frequency_config=obs_freq_config
        )
        # Create a healpix-only model (no point-source arrays)
        healpix_only = SkyModel(
            _healpix_maps=sky.healpix_maps,
            _healpix_nside=sky.healpix_nside,
            _observation_frequencies=sky.observation_frequencies,
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
            model_name="test_healpix",
        )
        psky = healpix_only._to_pyradiosky()
        assert psky.component_type == "healpix"
        assert psky.nside == 8

    def test_to_pyradiosky_empty_raises(self):
        """Empty model → ValueError."""
        sky = SkyModel.empty_sky("test")
        with pytest.raises(ValueError, match="empty"):
            sky._to_pyradiosky()

    def test_save_load_roundtrip(self, test_sky, tmp_path):
        """save() → load() preserves source count, coords, flux."""
        sky = test_sky.with_reference_frequency(100e6)
        path = str(tmp_path / "test.skyh5")
        sky.save(path)

        loaded = SkyModel.load(path, precision=PrecisionConfig.standard())
        assert loaded.n_point_sources == sky.n_point_sources
        assert np.allclose(loaded.flux, sky.flux, rtol=1e-5)

    def test_save_warns_lost_data(self, precision, tmp_path):
        """Model with RM → save warns about lost data."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux=np.array([1.0]),
            rotation_measure=np.array([10.0]),
            reference_frequency=100e6,
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
        sky = test_sky.with_reference_frequency(100e6).with_healpix_maps(
            nside=16, obs_frequency_config=obs_freq_config
        )
        r = repr(sky)
        assert "healpix_map" in r
        assert "nside=16" in r

    def test_repr_with_extensions(self, precision):
        """Model with RM and Gaussian → repr mentions them."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux=np.array([1.0]),
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
        sky = test_sky.with_reference_frequency(100e6).with_healpix_maps(
            nside=8, obs_frequency_config=obs_freq_config
        )
        m = sky.get_map_at_frequency(100e6)
        assert isinstance(m, np.ndarray)

    def test_nearest_frequency_selected(self, test_sky, obs_freq_config):
        """Between two channels → nearest returned."""
        sky = test_sky.with_reference_frequency(100e6).with_healpix_maps(
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
        sky = test_sky.with_reference_frequency(100e6).with_healpix_maps(
            nside=8, obs_frequency_config=obs_freq_config
        )
        maps, nside, freqs = sky.get_multifreq_maps()
        assert isinstance(maps, np.ndarray)
        assert maps.shape == (6, hp.nside2npix(8))
        assert nside == 8
        assert len(freqs) == 6

    def test_get_stokes_maps_unpolarized(self, test_sky, obs_freq_config):
        """Unpolarized model → Q, U, V are None."""
        sky = test_sky.with_reference_frequency(100e6).with_healpix_maps(
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
        sky = test_sky.with_reference_frequency(100e6).with_healpix_maps(
            nside=8, obs_frequency_config=obs_freq_config
        )
        # Create a healpix-only model to test the healpix filtering path
        healpix_only = SkyModel(
            _healpix_maps=sky.healpix_maps,
            _healpix_nside=sky.healpix_nside,
            _observation_frequencies=sky.observation_frequencies,
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
            model_name="test",
        )
        region = SkyRegion.cone(ra_deg=0.0, dec_deg=-30.0, radius_deg=10.0)
        filtered = healpix_only.filter_region(region)
        # Some pixels should be zero (outside cone)
        m = filtered.get_map_at_frequency(100e6)
        assert np.any(m == 0)

    def test_filter_preserves_metadata(self, test_sky, precision):
        """Filtered model retains model_name, reference_frequency, precision."""
        sky = test_sky.with_reference_frequency(100e6)
        region = SkyRegion.cone(ra_deg=180.0, dec_deg=-30.0, radius_deg=30.0)
        filtered = sky.filter_region(region)
        assert filtered.model_name == sky.model_name
        assert filtered.reference_frequency == sky.reference_frequency

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
        assert filtered.n_point_sources > 0
        assert filtered.n_point_sources < sky.n_point_sources


# ---------------------------------------------------------------------------
# Memory estimation expanded
# ---------------------------------------------------------------------------


class TestEstimateMemoryExpanded:
    def test_memory_scales_with_nside(self):
        """nside=64 vs nside=128 → 4x memory (npix ∝ nside²)."""
        m64 = estimate_healpix_memory(nside=64, n_frequencies=10)
        m128 = estimate_healpix_memory(nside=128, n_frequencies=10)
        assert np.isclose(m128["total_bytes"] / m64["total_bytes"], 4.0)

    def test_memory_scales_with_frequencies(self):
        """10 vs 20 frequencies → 2x memory."""
        m10 = estimate_healpix_memory(nside=64, n_frequencies=10)
        m20 = estimate_healpix_memory(nside=64, n_frequencies=20)
        assert np.isclose(m20["total_bytes"] / m10["total_bytes"], 2.0)

    def test_memory_scales_with_stokes(self):
        """n_stokes=4 → 4x memory vs n_stokes=1."""
        m1 = estimate_healpix_memory(nside=64, n_frequencies=10, n_stokes=1)
        m4 = estimate_healpix_memory(nside=64, n_frequencies=10, n_stokes=4)
        assert np.isclose(m4["total_bytes"] / m1["total_bytes"], 4.0)


# ---------------------------------------------------------------------------
# Degree/Radian conversion at precision
# ---------------------------------------------------------------------------


class TestDegRadConversion:
    def test_deg_to_rad_known_values(self):
        """180° → π, 90° → π/2."""
        arr = np.array([180.0, 90.0, 0.0])
        result = SkyModel.deg_to_rad_at_precision(arr, None)
        assert np.allclose(result, [np.pi, np.pi / 2, 0.0])

    def test_rad_to_deg_known_values(self):
        """π → 180°, π/2 → 90°."""
        arr = np.array([np.pi, np.pi / 2, 0.0])
        result = SkyModel.rad_to_deg_at_precision(arr, None)
        assert np.allclose(result, [180.0, 90.0, 0.0])

    def test_round_trip(self):
        """deg → rad → deg preserves values."""
        arr = np.array([0.0, 45.0, 90.0, 180.0, 270.0, 360.0])
        rad = SkyModel.deg_to_rad_at_precision(arr, None)
        deg = SkyModel.rad_to_deg_at_precision(rad, None)
        assert np.allclose(deg, arr)

    def test_precision_dtype(self):
        """With float32 precision → result is float32."""
        precision = PrecisionConfig.fast()  # uses float32
        arr = np.array([180.0, 90.0])
        result = SkyModel.deg_to_rad_at_precision(arr, precision)
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
    if precision is None:
        precision = PrecisionConfig.standard()
    hp_dt = precision.sky_model.get_dtype("healpix_maps")
    npix = hp.nside2npix(nside)
    freqs = np.array([100e6 + i * 1e6 for i in range(n_freq)])
    i_maps = np.full((n_freq, npix), fill_value, dtype=hp_dt)
    q_maps = np.full((n_freq, npix), 0.01, dtype=hp_dt) if with_q else None
    u_maps = np.full((n_freq, npix), 0.02, dtype=hp_dt) if with_u else None
    v_maps = np.full((n_freq, npix), 0.005, dtype=hp_dt) if with_v else None
    return SkyModel(
        _healpix_maps=i_maps,
        _healpix_q_maps=q_maps,
        _healpix_u_maps=u_maps,
        _healpix_v_maps=v_maps,
        _healpix_nside=nside,
        _observation_frequencies=freqs,
        _native_format=SkyFormat.HEALPIX,
        reference_frequency=freqs[0],
        model_name="test_healpix",
        _precision=precision,
    )


# ---------------------------------------------------------------------------
# TestValidationPrecision
# ---------------------------------------------------------------------------


class TestValidationPrecision:
    def test_precision_defaults_to_standard(self):
        """from_arrays with precision=None → defaults to PrecisionConfig.standard()."""
        sky = SkyModel.from_arrays(
            ra_rad=np.zeros(3),
            dec_rad=np.zeros(3),
            flux=np.zeros(3),
            precision=None,
        )
        assert sky.precision is not None
        assert sky._source_dtype() == np.float64

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
        assert sky.healpix_maps.dtype == hp_dt
        assert sky.healpix_q_maps.dtype == hp_dt
        assert sky.healpix_u_maps.dtype == hp_dt
        assert sky.healpix_v_maps.dtype == hp_dt


# ---------------------------------------------------------------------------
# TestPropertiesExpanded
# ---------------------------------------------------------------------------


class TestPropertiesExpanded:
    def test_n_frequencies_with_obs(self):
        """Healpix model → n_frequencies returns correct count."""
        sky = _make_healpix_only_model(n_freq=5)
        assert sky.n_frequencies == 5

    def test_n_sky_elements_healpix_only(self):
        """Healpix-only model → n_sky_elements returns npix, n_pixels agrees."""
        nside = 8
        sky = _make_healpix_only_model(nside=nside)
        assert sky.n_sky_elements == hp.nside2npix(nside)
        assert sky.n_pixels == hp.nside2npix(nside)
        assert sky.n_point_sources == 0

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
        assert np.all(sky.healpix_q_maps == 0.01)

        # Apply a small cone so most pixels are outside
        region = SkyRegion.cone(ra_deg=0.0, dec_deg=0.0, radius_deg=10.0)
        filtered = sky.filter_region(region)

        # Some Q pixels should now be 0 (outside the cone)
        assert np.any(filtered.healpix_q_maps == 0.0)
        # Some Q pixels should still be 0.01 (inside the cone)
        assert np.any(filtered.healpix_q_maps == 0.01)
        # Same for U
        assert np.any(filtered.healpix_u_maps == 0.0)
        assert np.any(filtered.healpix_u_maps == 0.02)


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
        ).with_reference_frequency(100e6)
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
        ).with_reference_frequency(100e6)
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

    def test_unknown_representation_raises(self, precision):
        """Unknown representation string raises ValueError."""
        sky = SkyModel.from_test_sources(
            num_sources=5,
            flux_range=(1, 5),
            dec_deg=-30,
            spectral_index=-0.7,
            precision=precision,
        )
        with pytest.raises(ValueError, match="Unknown representation"):
            sky.with_representation("invalid_mode")


# ---------------------------------------------------------------------------
# TestAsPointSourceDictsExpanded2
# ---------------------------------------------------------------------------


class TestAsPointSourceArraysExpanded2:
    def test_healpix_only_model(self):
        """Healpix-only model → as_point_source_arrays(frequency=...) converts on the fly."""
        sky = _make_healpix_only_model(nside=8, n_freq=3, fill_value=100.0)
        arrays = sky.as_point_source_arrays(frequency=100e6)
        assert isinstance(arrays, dict)
        assert len(arrays["ra_rad"]) > 0
        assert "flux" in arrays

    def test_empty_model_returns_empty_arrays(self):
        """empty_sky() → as_point_source_arrays() returns empty arrays."""
        sky = SkyModel.empty_sky("test")
        result = sky.as_point_source_arrays()
        assert len(result["ra_rad"]) == 0

    def test_all_filtered_by_flux_limit(self, precision):
        """Model with max flux 10 Jy, flux_limit=100 → returns empty arrays."""
        sky = SkyModel.from_test_sources(
            num_sources=10,
            flux_range=(1.0, 10.0),
            dec_deg=-30,
            spectral_index=-0.7,
            precision=precision,
        )
        result = sky.as_point_source_arrays(flux_limit=100.0)
        assert len(result["ra_rad"]) == 0


# ---------------------------------------------------------------------------
# TestBackwardCompatStubs
# ---------------------------------------------------------------------------


class TestFrequencyResolutionExpanded:
    def test_resolve_frequency_index_no_freqs_raises(self, precision):
        """Model without observation_frequencies → resolve_frequency_index() raises ValueError."""
        sky = SkyModel.from_test_sources(
            num_sources=5,
            flux_range=(1, 5),
            dec_deg=-30,
            spectral_index=-0.7,
            precision=precision,
        )
        with pytest.raises(ValueError, match="No observation frequencies"):
            sky.resolve_frequency_index(100e6)


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
        ).with_reference_frequency(100e6)
        psky = sky._to_pyradiosky()
        stokes = psky.stokes.value  # shape (4, 1, N)
        assert np.any(stokes[1] != 0), "Q should be non-zero"
        assert np.any(stokes[2] != 0), "U should be non-zero"
        assert np.any(stokes[3] != 0), "V should be non-zero"

    def test_save_empty_model_raises(self):
        """Empty model → save() raises ValueError (via _to_pyradiosky)."""
        sky = SkyModel.empty_sky("test")
        with pytest.raises(ValueError, match="empty"):
            sky._to_pyradiosky()

    def test_save_warns_spectral_coeffs(self, precision, tmp_path):
        """Model with 2-term spectral_coeffs → save warns 'multi-term spectral coefficients'."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux=np.array([1.0]),
            spectral_coeffs=np.array([[-0.7, -0.1]]),
            reference_frequency=100e6,
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
            assert sky.n_point_sources == 5


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

        sky = SkyModel.from_freq_dict_maps(
            i_maps=i_maps,
            q_maps=q_maps,
            u_maps=u_maps,
            v_maps=v_maps,
            nside=nside,
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
            model_name="test_dict_maps",
        )
        assert sky.healpix_maps.shape == (2, npix)
        assert sky.healpix_q_maps is not None
        assert sky.healpix_q_maps.shape == (2, npix)
        assert sky.healpix_u_maps is not None
        assert sky.healpix_u_maps.shape == (2, npix)
        assert sky.healpix_v_maps is not None
        assert sky.healpix_v_maps.shape == (2, npix)
        # Frequencies should be sorted
        assert np.isclose(sky.observation_frequencies[0], 100e6)
        assert np.isclose(sky.observation_frequencies[1], 200e6)
        # I map at 100 MHz should have values of 100.0
        assert np.allclose(sky.healpix_maps[0], 100.0)


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
        assert sky.n_point_sources == 1
        assert np.isclose(sky.ra_rad[0], 0.0)
        assert np.isclose(sky.flux[0], 5.0)


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
        with pytest.warns(DeprecationWarning, match="from_point_sources"):
            sky = SkyModel.from_point_sources([src1, src2], precision=precision)
        assert sky.spectral_coeffs is not None
        assert sky.spectral_coeffs.shape == (2, 2)
        # First source: coeffs as given
        assert np.isclose(sky.spectral_coeffs[0, 0], -0.7)
        assert np.isclose(sky.spectral_coeffs[0, 1], -0.1)
        # Second source: first coeff is spectral_index default, second is 0
        assert np.isclose(sky.spectral_coeffs[1, 0], -0.8)
        assert np.isclose(sky.spectral_coeffs[1, 1], 0.0)


# ---------------------------------------------------------------------------
# TestCatalogAccessors
# ---------------------------------------------------------------------------


class TestCatalogAccessors:
    def test_list_all_models_keys(self):
        """list_all_models() has keys 'diffuse', 'point_catalogs', 'racs'."""
        models = list_all_models()
        assert "diffuse" in models
        assert "point_catalogs" in models
        assert "racs" in models
        # Each should be a dict
        assert isinstance(models["diffuse"], dict)
        assert isinstance(models["point_catalogs"], dict)
        assert isinstance(models["racs"], dict)

    def test_get_catalog_info_vizier(self):
        """get_catalog_info('nvss') → dict with 'freq_mhz' key."""
        info = get_catalog_info("nvss")
        assert isinstance(info, dict)
        assert "freq_mhz" in info
        assert info["freq_mhz"] == 1400.0

    def test_get_catalog_info_diffuse(self):
        """get_catalog_info('gsm2008') → dict with 'parameters' key."""
        info = get_catalog_info("gsm2008")
        assert isinstance(info, dict)
        assert "parameters" in info
        assert isinstance(info["parameters"], dict)

    def test_get_catalog_info_unknown_raises(self):
        """get_catalog_info('nonexistent') → ValueError."""
        with pytest.raises(ValueError, match="Unknown catalog key"):
            get_catalog_info("nonexistent")


# ---------------------------------------------------------------------------
# TestReprExpanded
# ---------------------------------------------------------------------------


class TestRefFreqArray:
    """Tests for the per-source _ref_freq array."""

    def test_ref_freq_populated_by_from_arrays(self, precision):
        """from_arrays() with reference_frequency= populates _ref_freq."""
        n = 5
        sky = SkyModel.from_arrays(
            ra_rad=np.zeros(n),
            dec_rad=np.zeros(n),
            flux=np.ones(n),
            reference_frequency=150e6,
            precision=precision,
        )
        assert sky.ref_freq is not None
        assert len(sky.ref_freq) == n
        assert np.all(sky.ref_freq == 150e6)

    def test_ref_freq_explicit_array(self, precision):
        """from_arrays() with explicit ref_freq= uses it directly."""
        n = 3
        ref = np.array([100e6, 200e6, 1400e6])
        sky = SkyModel.from_arrays(
            ra_rad=np.zeros(n),
            dec_rad=np.zeros(n),
            flux=np.ones(n),
            ref_freq=ref,
            precision=precision,
        )
        np.testing.assert_array_equal(sky.ref_freq, ref)

    def test_ref_freq_in_as_point_source_arrays(self, precision):
        """as_point_source_arrays() returns ref_freq as array, not scalar."""
        ref = np.array([100e6, 1400e6])
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.0, 1.0]),
            dec_rad=np.array([0.0, 0.0]),
            flux=np.array([1.0, 2.0]),
            ref_freq=ref,
            precision=precision,
        )
        arrays = sky.as_point_source_arrays()
        assert isinstance(arrays["ref_freq"], np.ndarray)
        assert len(arrays["ref_freq"]) == 2
        np.testing.assert_array_equal(arrays["ref_freq"], ref)

    def test_ref_freq_preserved_through_filter(self, precision):
        """filter_region() correctly slices _ref_freq."""
        n = 10
        ref = np.arange(1, n + 1) * 100e6
        sky = SkyModel.from_arrays(
            ra_rad=np.linspace(0, 2 * np.pi, n),
            dec_rad=np.zeros(n),
            flux=np.ones(n),
            ref_freq=ref,
            precision=precision,
        )
        region = SkyRegion.cone(ra_deg=0.0, dec_deg=0.0, radius_deg=30.0)
        filtered = sky.filter_region(region)
        assert filtered.ref_freq is not None
        assert len(filtered.ref_freq) == filtered.n_point_sources
        # All retained ref_freq values should be from the original array
        for val in filtered.ref_freq:
            assert float(val) in ref

    def test_ref_freq_test_sources_none(self, precision):
        """Test sources get ref_freq=None (no catalog frequency)."""
        sky = SkyModel.from_test_sources(
            num_sources=5,
            flux_range=(1.0, 5.0),
            dec_deg=-30.0,
            spectral_index=-0.7,
            precision=precision,
        )
        assert sky.ref_freq is None

    def test_ref_freq_length_validation(self):
        """Mismatched _ref_freq length raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            SkyModel(
                _ra_rad=np.zeros(5),
                _dec_rad=np.zeros(5),
                _flux=np.ones(5),
                _spectral_index=np.zeros(5),
                _stokes_q=np.zeros(5),
                _stokes_u=np.zeros(5),
                _stokes_v=np.zeros(5),
                _ref_freq=np.zeros(3),  # wrong length
            )


class TestReprExpanded:
    def test_repr_healpix_with_polarization(self):
        """Healpix model with Q maps → repr contains 'IQ' in stokes components."""
        sky = _make_healpix_only_model(nside=8, n_freq=2, with_q=True)
        r = repr(sky)
        assert "IQ" in r
        assert "healpix_map" in r
        assert "nside=8" in r


# ---------------------------------------------------------------------------
# Bug 5: HEALPix→point-source conversion sets _ref_freq
# ---------------------------------------------------------------------------


class TestHealpixToPointSourceRefFreq:
    def test_healpix_to_point_sources_sets_ref_freq(self):
        """Converting healpix to point sources must populate _ref_freq."""
        sky = _make_healpix_only_model(nside=4, n_freq=2, fill_value=500.0)
        ps = sky.with_representation("point_sources", frequency=100e6)
        assert ps.ref_freq is not None
        assert len(ps.ref_freq) == len(ps.ra_rad)
        np.testing.assert_allclose(ps.ref_freq, 100e6)

    def test_healpix_to_point_sources_ref_freq_updates_with_conversion_frequency(self):
        """Different conversion frequency → different _ref_freq values."""
        sky = _make_healpix_only_model(nside=4, n_freq=2, fill_value=500.0)
        ps1 = sky.with_representation("point_sources", frequency=100e6)
        ps2 = sky.with_representation("point_sources", frequency=101e6)
        np.testing.assert_allclose(ps1.ref_freq, 100e6)
        np.testing.assert_allclose(ps2.ref_freq, 101e6)


# ---------------------------------------------------------------------------
# Bug 1: Single point-source model → healpix via with_representation
# ---------------------------------------------------------------------------


class TestSinglePointSourceToHealpix:
    def test_single_point_source_to_healpix(self):
        """with_representation('healpix_map') works when frequencies is passed."""
        precision = PrecisionConfig.standard()
        sky = SkyModel.from_test_sources(
            num_sources=10,
            flux_range=(1.0, 5.0),
            dec_deg=-30.0,
            spectral_index=-0.7,
            precision=precision,
        ).with_reference_frequency(100e6)
        freqs = np.array([100e6, 101e6, 102e6])
        result = sky.with_representation("healpix_map", nside=8, frequencies=freqs)
        assert result.mode == "healpix_map"
        assert result.healpix_maps is not None
        assert result.healpix_maps.shape[0] == 3

    def test_point_source_to_healpix_fails_without_frequencies(self):
        """with_representation('healpix_map') raises without frequency info."""
        precision = PrecisionConfig.standard()
        sky = SkyModel.from_test_sources(
            num_sources=10,
            flux_range=(1.0, 5.0),
            dec_deg=-30.0,
            spectral_index=-0.7,
            precision=precision,
        ).with_reference_frequency(100e6)
        with pytest.raises(ValueError, match="Cannot convert point sources"):
            sky.with_representation("healpix_map", nside=8)


# ---------------------------------------------------------------------------
# Bug 3: pyradiosky loader per-source reference frequencies
# ---------------------------------------------------------------------------

try:
    from pyradiosky import SkyModel as PyRadioSkyModel

    HAS_PYRADIOSKY = True
except ImportError:
    HAS_PYRADIOSKY = False


@pytest.mark.skipif(not HAS_PYRADIOSKY, reason="pyradiosky not installed")
class TestPyradioskyPerSourceRefFreq:
    def test_preserves_per_source_ref_freq(self, tmp_path, precision):
        """Loader must preserve per-source reference_frequency, not broadcast a scalar."""
        import astropy.units as u

        ref_freqs = np.array([100e6, 200e6, 1400e6])
        psky = PyRadioSkyModel(
            name=np.array(["src_0", "src_1", "src_2"]),
            skycoord=SkyCoord(
                ra=[10, 20, 30], dec=[-30, -30, -30], unit="deg", frame="icrs"
            ),
            stokes=np.array([[1.0, 2.0, 3.0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])[
                :, np.newaxis, :
            ]
            * u.Jy,
            spectral_type="spectral_index",
            spectral_index=np.array([-0.7, -0.8, -0.9]),
            reference_frequency=ref_freqs * u.Hz,
            component_type="point",
        )
        fpath = str(tmp_path / "test_perfreq.skyh5")
        psky.write_skyh5(fpath, clobber=True)

        from rrivis.core.sky._loaders_pyradiosky import load_pyradiosky_file

        loaded = load_pyradiosky_file(fpath, filetype="skyh5", precision=precision)
        assert loaded.ref_freq is not None
        np.testing.assert_allclose(
            np.sort(loaded.ref_freq), np.sort(ref_freqs), rtol=1e-6
        )


# ---------------------------------------------------------------------------
# Single-Representation Invariant
# ---------------------------------------------------------------------------


class TestDualRepresentation:
    """SkyModel supports dual representation (both PS and HEALPix)."""

    def test_constructor_accepts_dual_representation(self, precision):
        """Constructing with both PS arrays and healpix maps works."""
        nside = 8
        npix = hp.nside2npix(nside)
        n = 5
        sky = SkyModel(
            _ra_rad=np.zeros(n),
            _dec_rad=np.zeros(n),
            _flux=np.ones(n),
            _spectral_index=np.full(n, -0.7),
            _stokes_q=np.zeros(n),
            _stokes_u=np.zeros(n),
            _stokes_v=np.zeros(n),
            _healpix_maps=np.ones((1, npix), dtype=np.float32),
            _healpix_nside=nside,
            _observation_frequencies=np.array([100e6]),
            _precision=precision,
            _active_mode=SkyFormat.HEALPIX,
        )
        assert sky.has_point_sources
        assert sky.has_multifreq_maps
        assert SkyFormat.POINT_SOURCES in sky.representations
        assert SkyFormat.HEALPIX in sky.representations

    def test_with_healpix_maps_preserves_ps_arrays(self, test_sky, obs_freq_config):
        """with_healpix_maps preserves point-source data, switches mode."""
        sky = test_sky.with_reference_frequency(100e6)
        hp_sky = sky.with_healpix_maps(nside=8, obs_frequency_config=obs_freq_config)
        assert hp_sky.mode == "healpix_map"
        assert hp_sky.has_multifreq_maps is True
        # PS arrays preserved
        assert hp_sky.ra_rad is not None
        assert hp_sky.dec_rad is not None
        assert hp_sky.flux is not None
        assert hp_sky.spectral_index is not None
        assert hp_sky.has_point_sources is True
        # Original is unchanged
        assert sky.ra_rad is not None
        assert sky.has_multifreq_maps is False

    def test_with_representation_healpix_preserves_ps(self, test_sky, obs_freq_config):
        """with_representation('healpix_map') preserves PS arrays, switches mode."""
        sky = test_sky.with_reference_frequency(100e6)
        hp_sky = sky.with_representation(
            "healpix_map", nside=8, obs_frequency_config=obs_freq_config
        )
        assert hp_sky.mode == "healpix_map"
        assert hp_sky.ra_rad is not None
        assert hp_sky.has_multifreq_maps is True

    def test_with_representation_ps_preserves_healpix(self, test_sky, obs_freq_config):
        """with_representation('point_sources') on dual model preserves healpix."""
        sky = test_sky.with_reference_frequency(100e6)
        hp_sky = sky.with_healpix_maps(nside=8, obs_frequency_config=obs_freq_config)
        ps_sky = hp_sky.with_representation("point_sources", frequency=100e6)
        assert ps_sky.mode == "point_sources"
        assert ps_sky.ra_rad is not None
        # HEALPix data preserved
        assert ps_sky.has_multifreq_maps is True
        assert ps_sky.healpix_nside is not None

    def test_round_trip_preserves_data(self, test_sky, obs_freq_config):
        """Round-trip PS -> healpix -> PS preserves both representations."""
        sky = test_sky.with_reference_frequency(100e6)
        original_n = sky.n_point_sources

        # Step 1: PS -> healpix (PS preserved)
        hp_sky = sky.with_representation(
            "healpix_map", nside=8, obs_frequency_config=obs_freq_config
        )
        assert hp_sky.mode == "healpix_map"
        assert hp_sky.ra_rad is not None
        assert hp_sky.has_multifreq_maps is True
        assert hp_sky.n_point_sources == original_n

        # Step 2: healpix -> PS (uses preserved PS, no lossy conversion)
        ps_sky = hp_sky.with_representation("point_sources", frequency=100e6)
        assert ps_sky.mode == "point_sources"
        assert ps_sky.ra_rad is not None
        assert ps_sky.has_multifreq_maps is True
        # Original source count preserved (no quantization loss)
        assert ps_sky.n_point_sources == original_n

    def test_as_point_source_arrays_uses_preserved_ps(self, test_sky, obs_freq_config):
        """as_point_source_arrays on dual model returns original PS data."""
        sky = test_sky.with_reference_frequency(100e6)
        hp_sky = sky.with_healpix_maps(nside=8, obs_frequency_config=obs_freq_config)
        # PS data preserved — should return it directly
        assert hp_sky.ra_rad is not None

        arrays = hp_sky.as_point_source_arrays(frequency=100e6)
        assert len(arrays["ra_rad"]) > 0
        assert len(arrays["flux"]) > 0
        # Count matches original
        assert len(arrays["ra_rad"]) == test_sky.n_point_sources

    def test_as_point_source_arrays_healpix_only(self, obs_freq_config):
        """as_point_source_arrays on healpix-only model converts on the fly."""
        hp_sky = _make_healpix_only_model(nside=8, n_freq=3)
        assert hp_sky.ra_rad is None

        arrays = hp_sky.as_point_source_arrays(frequency=100e6)
        assert len(arrays["ra_rad"]) > 0
        assert len(arrays["flux"]) > 0
        # Model is not mutated (frozen)
        assert hp_sky.ra_rad is None

    def test_filter_region_dual_filters_both(self, test_sky, obs_freq_config):
        """filter_region on dual model filters both representations."""
        sky = test_sky.with_reference_frequency(100e6)
        hp_sky = sky.with_healpix_maps(nside=8, obs_frequency_config=obs_freq_config)
        region = SkyRegion.cone(ra_deg=180.0, dec_deg=-30.0, radius_deg=30.0)
        filtered = hp_sky.filter_region(region)
        assert filtered.mode == "healpix_map"
        assert filtered.has_multifreq_maps is True
        # PS data also filtered (subset of original)
        assert filtered.has_point_sources
        assert filtered.n_point_sources <= hp_sky.n_point_sources

    def test_mode_switch_is_cheap(self, test_sky, obs_freq_config):
        """Switching mode on a dual model returns quickly without conversion."""
        sky = test_sky.with_reference_frequency(100e6)
        hp_sky = sky.with_healpix_maps(nside=8, obs_frequency_config=obs_freq_config)
        assert hp_sky.mode == "healpix_map"

        # Switch to point_sources — no conversion needed
        ps_sky = hp_sky.with_representation("point_sources")
        assert ps_sky.mode == "point_sources"
        assert ps_sky.has_multifreq_maps is True  # healpix still there

        # Switch back — no conversion needed
        hp2 = ps_sky.with_representation("healpix_map")
        assert hp2.mode == "healpix_map"
        assert hp2.has_point_sources is True  # PS still there

    def test_with_representation_noop_ps(self, test_sky):
        """with_representation('point_sources') on PS model returns self."""
        result = test_sky.with_representation("point_sources")
        assert result is test_sky

    def test_with_representation_noop_healpix(self, test_sky, obs_freq_config):
        """with_representation('healpix_map') on healpix model returns self."""
        sky = test_sky.with_reference_frequency(100e6)
        hp_sky = sky.with_healpix_maps(nside=8, obs_frequency_config=obs_freq_config)
        result = hp_sky.with_representation("healpix_map")
        assert result is hp_sky


# ---------------------------------------------------------------------------
# _empty_source_arrays helper
# ---------------------------------------------------------------------------


class TestEmptySourceArrays:
    """Tests for the centralized empty SourceArrays factory."""

    def test_returns_correct_keys(self):
        from rrivis.core.sky.model import _empty_source_arrays

        result = _empty_source_arrays()
        assert set(result.keys()) == {
            "ra_rad",
            "dec_rad",
            "flux",
            "spectral_index",
            "stokes_q",
            "stokes_u",
            "stokes_v",
            "ref_freq",
            "rotation_measure",
            "major_arcsec",
            "minor_arcsec",
            "pa_deg",
            "spectral_coeffs",
        }

    def test_required_arrays_are_zero_length_float64(self):
        from rrivis.core.sky.model import _empty_source_arrays

        result = _empty_source_arrays()
        for key in (
            "ra_rad",
            "dec_rad",
            "flux",
            "spectral_index",
            "stokes_q",
            "stokes_u",
            "stokes_v",
            "ref_freq",
        ):
            assert len(result[key]) == 0
            assert result[key].dtype == np.float64

    def test_optional_keys_are_none(self):
        from rrivis.core.sky.model import _empty_source_arrays

        result = _empty_source_arrays()
        for key in (
            "rotation_measure",
            "major_arcsec",
            "minor_arcsec",
            "pa_deg",
            "spectral_coeffs",
        ):
            assert result[key] is None

    def test_arrays_are_independent(self):
        from rrivis.core.sky.model import _empty_source_arrays

        result = _empty_source_arrays()
        assert result["ra_rad"] is not result["dec_rad"]


# ---------------------------------------------------------------------------
# Equality and Hashing
# ---------------------------------------------------------------------------


class TestEquality:
    """SkyModel __eq__, __hash__, and is_close."""

    def test_eq_identical_models(self, precision):
        """Two models built from the same data are equal."""
        sky1 = SkyModel.from_arrays(
            ra_rad=np.array([0.1, 0.2]),
            dec_rad=np.array([-0.3, 0.4]),
            flux=np.array([1.0, 2.0]),
            model_name="test",
            precision=precision,
        )
        sky2 = SkyModel.from_arrays(
            ra_rad=np.array([0.1, 0.2]),
            dec_rad=np.array([-0.3, 0.4]),
            flux=np.array([1.0, 2.0]),
            model_name="test",
            precision=precision,
        )
        assert sky1 == sky2

    def test_eq_different_flux(self, precision):
        """Models with different flux values are not equal."""
        sky1 = SkyModel.from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([-0.3]),
            flux=np.array([1.0]),
            model_name="test",
            precision=precision,
        )
        sky2 = SkyModel.from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([-0.3]),
            flux=np.array([2.0]),
            model_name="test",
            precision=precision,
        )
        assert sky1 != sky2

    def test_eq_different_model_name(self, precision):
        """Models with different names are not equal."""
        sky1 = SkyModel.from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([-0.3]),
            flux=np.array([1.0]),
            model_name="a",
            precision=precision,
        )
        sky2 = SkyModel.from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([-0.3]),
            flux=np.array([1.0]),
            model_name="b",
            precision=precision,
        )
        assert sky1 != sky2

    def test_eq_non_skymodel_returns_not_implemented(self, precision):
        """Comparing SkyModel to a non-SkyModel returns NotImplemented."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([-0.3]),
            flux=np.array([1.0]),
            precision=precision,
        )
        assert sky.__eq__("not a model") is NotImplemented

    def test_unhashable(self, precision):
        """SkyModel is not hashable (since __eq__ is custom)."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([-0.3]),
            flux=np.array([1.0]),
            precision=precision,
        )
        with pytest.raises(TypeError, match="unhashable"):
            hash(sky)

    def test_is_close_within_tolerance(self, precision):
        """is_close returns True for nearly-equal models."""
        sky1 = SkyModel.from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([-0.3]),
            flux=np.array([1.0]),
            model_name="test",
            precision=precision,
        )
        sky2 = SkyModel.from_arrays(
            ra_rad=np.array([0.1 + 1e-10]),
            dec_rad=np.array([-0.3]),
            flux=np.array([1.0]),
            model_name="test",
            precision=precision,
        )
        assert sky1.is_close(sky2, rtol=1e-7)

    def test_is_close_outside_tolerance(self, precision):
        """is_close returns False for models that differ significantly."""
        sky1 = SkyModel.from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([-0.3]),
            flux=np.array([1.0]),
            model_name="test",
            precision=precision,
        )
        sky2 = SkyModel.from_arrays(
            ra_rad=np.array([0.5]),
            dec_rad=np.array([-0.3]),
            flux=np.array([1.0]),
            model_name="test",
            precision=precision,
        )
        assert not sky1.is_close(sky2)

    def test_is_close_non_skymodel(self, precision):
        """is_close returns False for non-SkyModel."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([-0.3]),
            flux=np.array([1.0]),
            precision=precision,
        )
        assert not sky.is_close("not a model")


# ---------------------------------------------------------------------------
# iter_frequency_maps and with_memmap_backing
# ---------------------------------------------------------------------------


class TestIterFrequencyMaps:
    """Tests for the per-frequency iterator."""

    def test_iter_yields_correct_count(self):
        """Iterator yields one tuple per frequency channel."""
        sky = _make_healpix_only_model(nside=8, n_freq=5)
        items = list(sky.iter_frequency_maps())
        assert len(items) == 5

    def test_iter_yields_correct_shape(self):
        """Each yielded map has shape (npix,)."""
        nside = 8
        sky = _make_healpix_only_model(nside=nside, n_freq=3)
        for _freq, s_i, s_q, s_u, s_v in sky.iter_frequency_maps():
            assert s_i.shape == (hp.nside2npix(nside),)
            assert s_q is None
            assert s_u is None
            assert s_v is None

    def test_iter_with_polarization(self):
        """Iterator includes Q/U/V when present."""
        sky = _make_healpix_only_model(
            nside=8, n_freq=2, with_q=True, with_u=True, with_v=True
        )
        for _freq, s_i, s_q, s_u, s_v in sky.iter_frequency_maps():
            assert s_i is not None
            assert s_q is not None
            assert s_u is not None
            assert s_v is not None

    def test_iter_no_healpix_raises(self, precision):
        """Raises ValueError when no HEALPix maps."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([-0.3]),
            flux=np.array([1.0]),
            precision=precision,
        )
        with pytest.raises(ValueError, match="No multi-frequency"):
            list(sky.iter_frequency_maps())


class TestMemmapBacking:
    """Tests for with_memmap_backing."""

    def test_memmap_preserves_data(self, tmp_path):
        """Memmap-backed model has the same data."""
        sky = _make_healpix_only_model(nside=8, n_freq=3, fill_value=42.0)
        mm_sky = sky.with_memmap_backing(path=str(tmp_path))
        np.testing.assert_array_equal(mm_sky.healpix_maps, sky.healpix_maps)

    def test_memmap_is_ndarray_subclass(self, tmp_path):
        """Memmap arrays are np.ndarray subclass (isinstance check passes)."""
        sky = _make_healpix_only_model(nside=8, n_freq=2)
        mm_sky = sky.with_memmap_backing(path=str(tmp_path))
        assert isinstance(mm_sky.healpix_maps, np.ndarray)
        assert isinstance(mm_sky.healpix_maps, np.memmap)

    def test_memmap_no_healpix_raises(self, precision):
        """Raises ValueError when no HEALPix maps."""
        sky = SkyModel.from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([-0.3]),
            flux=np.array([1.0]),
            precision=precision,
        )
        with pytest.raises(ValueError, match="No HEALPix maps"):
            sky.with_memmap_backing()

    def test_memmap_with_polarization(self, tmp_path):
        """Memmap backs Q/U/V maps when present."""
        sky = _make_healpix_only_model(
            nside=8, n_freq=2, with_q=True, with_u=True, with_v=True
        )
        mm_sky = sky.with_memmap_backing(path=str(tmp_path))
        assert isinstance(mm_sky.healpix_q_maps, np.memmap)
        assert isinstance(mm_sky.healpix_u_maps, np.memmap)
        assert isinstance(mm_sky.healpix_v_maps, np.memmap)
        np.testing.assert_array_equal(mm_sky.healpix_q_maps, sky.healpix_q_maps)
