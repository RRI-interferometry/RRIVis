"""Tests for rrivis.core.sky.combine — concat, combine, and healpix merge."""

import healpy as hp
import numpy as np
import pytest

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import SkyModel
from rrivis.core.sky.combine import combine_models, concat_point_sources

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def precision():
    return PrecisionConfig.standard()


def _make_point_sky(n: int, precision: PrecisionConfig, seed: int = 0) -> SkyModel:
    """Helper: deterministic point-source SkyModel with *n* sources."""
    rng = np.random.default_rng(seed)
    return SkyModel.from_arrays(
        ra_rad=rng.uniform(0, 2 * np.pi, n),
        dec_rad=rng.uniform(-np.pi / 2, np.pi / 2, n),
        flux_ref=rng.uniform(0.1, 10.0, n),
        alpha=np.full(n, -0.7),
        model_name=f"test_{n}",
        frequency=150e6,
        precision=precision,
    )


def _make_point_sky_with_rm(
    n: int, precision: PrecisionConfig, seed: int = 0
) -> SkyModel:
    """Point-source model with rotation measure populated."""
    rng = np.random.default_rng(seed)
    return SkyModel.from_arrays(
        ra_rad=rng.uniform(0, 2 * np.pi, n),
        dec_rad=rng.uniform(-np.pi / 2, np.pi / 2, n),
        flux_ref=rng.uniform(0.1, 10.0, n),
        alpha=np.full(n, -0.7),
        rotation_measure=rng.uniform(-10, 10, n),
        model_name=f"test_rm_{n}",
        frequency=150e6,
        precision=precision,
    )


# ---------------------------------------------------------------------------
# concat_point_sources
# ---------------------------------------------------------------------------


class TestConcatPointSources:
    def test_concat_two_models(self, precision):
        """Concatenating 20 + 30 sources gives 50 total."""
        sky_a = _make_point_sky(20, precision, seed=1)
        sky_b = _make_point_sky(30, precision, seed=2)

        data = concat_point_sources([sky_a, sky_b])
        assert len(data["_ra_rad"]) == 50
        assert len(data["_dec_rad"]) == 50
        assert len(data["_flux_ref"]) == 50
        assert len(data["_alpha"]) == 50

    def test_concat_preserves_optional_fields(self, precision):
        """Both models have RM -> RM preserved in result."""
        sky_a = _make_point_sky_with_rm(10, precision, seed=1)
        sky_b = _make_point_sky_with_rm(15, precision, seed=2)

        data = concat_point_sources([sky_a, sky_b])
        assert data["_rotation_measure"] is not None
        assert len(data["_rotation_measure"]) == 25

    def test_concat_mixed_optional_fields(self, precision):
        """One model has RM, other doesn't -> zeros fill in for missing."""
        sky_with_rm = _make_point_sky_with_rm(10, precision, seed=1)
        sky_without_rm = _make_point_sky(5, precision, seed=2)

        data = concat_point_sources([sky_with_rm, sky_without_rm])
        assert data["_rotation_measure"] is not None
        assert len(data["_rotation_measure"]) == 15
        # The last 5 entries (from sky_without_rm) should be zeros
        np.testing.assert_array_equal(data["_rotation_measure"][10:], 0.0)

    def test_concat_empty_model_skipped(self, precision):
        """An empty model is silently skipped."""
        empty = SkyModel(
            _ra_rad=np.zeros(0, dtype=np.float64),
            _dec_rad=np.zeros(0, dtype=np.float64),
            _flux_ref=np.zeros(0, dtype=np.float64),
            _alpha=np.zeros(0, dtype=np.float64),
            _stokes_q=np.zeros(0, dtype=np.float64),
            _stokes_u=np.zeros(0, dtype=np.float64),
            _stokes_v=np.zeros(0, dtype=np.float64),
            model_name="empty",
            _precision=precision,
        )
        nonempty = _make_point_sky(10, precision, seed=3)

        data = concat_point_sources([empty, nonempty])
        assert len(data["_ra_rad"]) == 10

    def test_concat_all_empty(self, precision):
        """All empty models -> empty zero-length arrays."""
        empty = SkyModel(
            _ra_rad=np.zeros(0, dtype=np.float64),
            _dec_rad=np.zeros(0, dtype=np.float64),
            _flux_ref=np.zeros(0, dtype=np.float64),
            _alpha=np.zeros(0, dtype=np.float64),
            _stokes_q=np.zeros(0, dtype=np.float64),
            _stokes_u=np.zeros(0, dtype=np.float64),
            _stokes_v=np.zeros(0, dtype=np.float64),
            model_name="empty",
            _precision=precision,
        )
        data = concat_point_sources([empty, empty])
        assert len(data["_ra_rad"]) == 0
        assert data["_rotation_measure"] is None


# ---------------------------------------------------------------------------
# combine_models
# ---------------------------------------------------------------------------


class TestCombineModels:
    def test_combine_models_empty_list(self, precision):
        """Empty list -> empty SkyModel."""
        result = combine_models([], precision=precision)
        assert result._ra_rad is not None
        assert len(result._ra_rad) == 0

    def test_combine_models_single_model(self, precision):
        """Single model -> same data as input."""
        sky = _make_point_sky(25, precision, seed=42)
        result = combine_models([sky], precision=precision)
        assert len(result._ra_rad) == 25
        np.testing.assert_allclose(result._ra_rad, sky._ra_rad)
        np.testing.assert_allclose(result._flux_ref, sky._flux_ref)

    def test_combine_models_point_sources(self, precision):
        """Two point-source models combined as point_sources representation."""
        sky_a = _make_point_sky(12, precision, seed=10)
        sky_b = _make_point_sky(18, precision, seed=20)

        result = combine_models([sky_a, sky_b], precision=precision)
        assert result.mode == "point_sources"
        assert len(result._ra_rad) == 30

    def test_combine_models_double_counting_warning(self, precision):
        """Combining catalog + diffuse model emits a UserWarning."""
        point_sky = _make_point_sky(10, precision, seed=1)

        # Build a minimal healpix model
        nside = 8
        npix = hp.nside2npix(nside)
        n_freq = 2
        freqs = np.array([100e6, 101e6])
        healpix_sky = SkyModel(
            _healpix_maps=np.ones((n_freq, npix), dtype=np.float32) * 100.0,
            _healpix_nside=nside,
            _observation_frequencies=freqs,
            _native_format="healpix",
            frequency=100e6,
            model_name="diffuse_test",
            _precision=precision,
        )

        with pytest.warns(UserWarning, match="double-counting"):
            combine_models(
                [point_sky, healpix_sky],
                representation="healpix_map",
                precision=precision,
            )

    def test_concat_with_morphology(self, precision):
        """Model A has morphology, model B does not -> B's entries get zeros."""
        rng = np.random.default_rng(42)
        n_a, n_b = 5, 3
        sky_a = SkyModel.from_arrays(
            ra_rad=rng.uniform(0, 2 * np.pi, n_a),
            dec_rad=rng.uniform(-np.pi / 2, np.pi / 2, n_a),
            flux_ref=rng.uniform(0.1, 10.0, n_a),
            alpha=np.full(n_a, -0.7),
            major_arcsec=np.full(n_a, 10.0),
            minor_arcsec=np.full(n_a, 5.0),
            pa_deg=np.full(n_a, 45.0),
            model_name="with_morph",
            frequency=150e6,
            precision=precision,
        )
        sky_b = _make_point_sky(n_b, precision, seed=99)

        data = concat_point_sources([sky_a, sky_b])
        assert data["_major_arcsec"] is not None
        assert len(data["_major_arcsec"]) == n_a + n_b
        # B's entries should be zero-filled
        np.testing.assert_array_equal(data["_major_arcsec"][n_a:], 0.0)
        np.testing.assert_array_equal(data["_minor_arcsec"][n_a:], 0.0)
        np.testing.assert_array_equal(data["_pa_deg"][n_a:], 0.0)
        # A's entries should be preserved
        np.testing.assert_allclose(data["_major_arcsec"][:n_a], 10.0)

    def test_concat_with_spectral_coeffs_padding(self, precision):
        """Model A has 2-term spectral_coeffs, model B has single alpha -> padded."""
        sky_a = SkyModel.from_arrays(
            ra_rad=np.array([1.0]),
            dec_rad=np.array([0.5]),
            flux_ref=np.array([5.0]),
            alpha=np.array([-0.7]),
            spectral_coeffs=np.array([[-0.7, -0.1]]),
            model_name="with_coeffs",
            frequency=150e6,
            precision=precision,
        )
        sky_b = SkyModel.from_arrays(
            ra_rad=np.array([2.0]),
            dec_rad=np.array([-0.3]),
            flux_ref=np.array([3.0]),
            alpha=np.array([-0.8]),
            model_name="no_coeffs",
            frequency=150e6,
            precision=precision,
        )

        data = concat_point_sources([sky_a, sky_b])
        sp = data["_spectral_coeffs"]
        assert sp is not None
        assert sp.shape == (2, 2)
        # A's row preserved
        np.testing.assert_allclose(sp[0], [-0.7, -0.1])
        # B's row: column 0 = alpha, column 1 = 0.0
        np.testing.assert_allclose(sp[1, 0], -0.8)
        np.testing.assert_allclose(sp[1, 1], 0.0)

    def test_combine_healpix_two_models(self, precision):
        """Two healpix models at same nside/frequencies -> combined T_b."""
        nside = 8
        npix = hp.nside2npix(nside)
        freqs = np.array([100e6, 101e6])

        sky1 = SkyModel.from_test_sources(
            num_sources=20,
            flux_range=(1, 5),
            dec_deg=-30,
            spectral_index=-0.7,
            precision=precision,
        )
        sky1 = sky1.with_frequency(100e6).with_healpix_maps(
            nside=nside,
            frequencies=freqs,
        )
        sky2 = SkyModel.from_test_sources(
            num_sources=20,
            flux_range=(1, 5),
            dec_deg=30,
            spectral_index=-0.7,
            precision=precision,
        )
        sky2 = sky2.with_frequency(100e6).with_healpix_maps(
            nside=nside,
            frequencies=freqs,
        )

        # Build healpix-only models
        sky1_hp = SkyModel(
            _healpix_maps=sky1._healpix_maps,
            _healpix_q_maps=sky1._healpix_q_maps,
            _healpix_u_maps=sky1._healpix_u_maps,
            _healpix_v_maps=sky1._healpix_v_maps,
            _healpix_nside=nside,
            _observation_frequencies=freqs,
            _native_format="healpix",
            frequency=100e6,
            model_name="hp1",
            _precision=precision,
        )
        sky2_hp = SkyModel(
            _healpix_maps=sky2._healpix_maps,
            _healpix_q_maps=sky2._healpix_q_maps,
            _healpix_u_maps=sky2._healpix_u_maps,
            _healpix_v_maps=sky2._healpix_v_maps,
            _healpix_nside=nside,
            _observation_frequencies=freqs,
            _native_format="healpix",
            frequency=100e6,
            model_name="hp2",
            _precision=precision,
        )

        result = SkyModel.combine(
            [sky1_hp, sky2_hp],
            representation="healpix_map",
        )
        assert result.mode == "healpix_multifreq"
        assert result._healpix_maps is not None
        assert result._healpix_maps.shape == (2, npix)

        # Where both models contribute, combined T_b should be >= each
        for freq_idx in range(2):
            m1 = sky1_hp._healpix_maps[freq_idx]
            m2 = sky2_hp._healpix_maps[freq_idx]
            combined = result._healpix_maps[freq_idx]
            # Any pixel nonzero in either should be nonzero in combined
            either_pos = (m1 > 0) | (m2 > 0)
            assert np.all(combined[either_pos] > 0)

    def test_combine_healpix_nside_mismatch_raises(self, precision):
        """Two healpix models with different nside -> ValueError."""
        freqs = np.array([100e6, 101e6])
        sky_a = SkyModel(
            _healpix_maps=np.ones((2, hp.nside2npix(8)), dtype=np.float32) * 100,
            _healpix_nside=8,
            _observation_frequencies=freqs,
            _native_format="healpix",
            frequency=100e6,
            model_name="nside8",
            _precision=precision,
        )
        sky_b = SkyModel(
            _healpix_maps=np.ones((2, hp.nside2npix(16)), dtype=np.float32) * 50,
            _healpix_nside=16,
            _observation_frequencies=freqs,
            _native_format="healpix",
            frequency=100e6,
            model_name="nside16",
            _precision=precision,
        )

        with pytest.raises(ValueError, match="nside"):
            SkyModel.combine(
                [sky_a, sky_b],
                representation="healpix_map",
            )

    def test_combine_healpix_freq_mismatch_raises(self, precision):
        """Two healpix models with different frequencies -> ValueError."""
        nside = 8
        npix = hp.nside2npix(nside)
        sky_a = SkyModel(
            _healpix_maps=np.ones((2, npix), dtype=np.float32) * 100,
            _healpix_nside=nside,
            _observation_frequencies=np.array([100e6, 101e6]),
            _native_format="healpix",
            frequency=100e6,
            model_name="freqs_a",
            _precision=precision,
        )
        sky_b = SkyModel(
            _healpix_maps=np.ones((2, npix), dtype=np.float32) * 50,
            _healpix_nside=nside,
            _observation_frequencies=np.array([200e6, 201e6]),
            _native_format="healpix",
            frequency=200e6,
            model_name="freqs_b",
            _precision=precision,
        )

        with pytest.raises(ValueError, match="frequency"):
            SkyModel.combine(
                [sky_a, sky_b],
                representation="healpix_map",
            )

    def test_combine_models_healpix_without_config_raises(self, precision):
        """Point-source model combined as healpix without obs_frequency_config -> ValueError."""
        sky = _make_point_sky(10, precision, seed=1)

        with pytest.raises(ValueError, match="obs_frequency_config"):
            combine_models(
                [sky],
                representation="healpix_map",
                obs_frequency_config=None,
                precision=precision,
            )

    def test_combine_models_with_obs_config(self, precision):
        """Point-source models combined with healpix_map and obs_config -> healpix model."""
        sky_a = _make_point_sky(15, precision, seed=10)
        sky_b = _make_point_sky(10, precision, seed=20)

        obs_config = {
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 2.0,
            "frequency_unit": "MHz",
        }
        result = combine_models(
            [sky_a, sky_b],
            representation="healpix_map",
            nside=8,
            obs_frequency_config=obs_config,
            precision=precision,
        )
        assert result.mode == "healpix_multifreq"
        assert result._healpix_maps is not None
        assert result._healpix_nside == 8
