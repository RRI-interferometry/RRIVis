"""Tests for rrivis.core.sky.combine — concat, combine, and healpix merge."""

import healpy as hp
import numpy as np
import pytest

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import SkyModel
from rrivis.core.sky.combine import combine_models, concat_point_sources
from rrivis.core.sky.model import SkyFormat

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
        flux=rng.uniform(0.1, 10.0, n),
        spectral_index=np.full(n, -0.7),
        model_name=f"test_{n}",
        reference_frequency=150e6,
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
        flux=rng.uniform(0.1, 10.0, n),
        spectral_index=np.full(n, -0.7),
        rotation_measure=rng.uniform(-10, 10, n),
        model_name=f"test_rm_{n}",
        reference_frequency=150e6,
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
        assert len(data["ra_rad"]) == 50
        assert len(data["dec_rad"]) == 50
        assert len(data["flux"]) == 50
        assert len(data["spectral_index"]) == 50

    def test_concat_preserves_optional_fields(self, precision):
        """Both models have RM -> RM preserved in result."""
        sky_a = _make_point_sky_with_rm(10, precision, seed=1)
        sky_b = _make_point_sky_with_rm(15, precision, seed=2)

        data = concat_point_sources([sky_a, sky_b])
        assert data["rotation_measure"] is not None
        assert len(data["rotation_measure"]) == 25

    def test_concat_mixed_optional_fields(self, precision):
        """One model has RM, other doesn't -> zeros fill in for missing."""
        sky_with_rm = _make_point_sky_with_rm(10, precision, seed=1)
        sky_without_rm = _make_point_sky(5, precision, seed=2)

        data = concat_point_sources([sky_with_rm, sky_without_rm])
        assert data["rotation_measure"] is not None
        assert len(data["rotation_measure"]) == 15
        # The last 5 entries (from sky_without_rm) should be zeros
        np.testing.assert_array_equal(data["rotation_measure"][10:], 0.0)

    def test_concat_empty_model_skipped(self, precision):
        """An empty model is silently skipped."""
        empty = SkyModel(
            _ra_rad=np.zeros(0, dtype=np.float64),
            _dec_rad=np.zeros(0, dtype=np.float64),
            _flux=np.zeros(0, dtype=np.float64),
            _spectral_index=np.zeros(0, dtype=np.float64),
            _stokes_q=np.zeros(0, dtype=np.float64),
            _stokes_u=np.zeros(0, dtype=np.float64),
            _stokes_v=np.zeros(0, dtype=np.float64),
            model_name="empty",
            _precision=precision,
        )
        nonempty = _make_point_sky(10, precision, seed=3)

        data = concat_point_sources([empty, nonempty])
        assert len(data["ra_rad"]) == 10

    def test_concat_all_empty(self, precision):
        """All empty models -> empty zero-length arrays."""
        empty = SkyModel(
            _ra_rad=np.zeros(0, dtype=np.float64),
            _dec_rad=np.zeros(0, dtype=np.float64),
            _flux=np.zeros(0, dtype=np.float64),
            _spectral_index=np.zeros(0, dtype=np.float64),
            _stokes_q=np.zeros(0, dtype=np.float64),
            _stokes_u=np.zeros(0, dtype=np.float64),
            _stokes_v=np.zeros(0, dtype=np.float64),
            model_name="empty",
            _precision=precision,
        )
        data = concat_point_sources([empty, empty])
        assert len(data["ra_rad"]) == 0
        assert data["rotation_measure"] is None


# ---------------------------------------------------------------------------
# combine_models
# ---------------------------------------------------------------------------


class TestCombineModels:
    def test_combine_models_empty_list(self, precision):
        """Empty list -> empty SkyModel."""
        result = combine_models([], precision=precision)
        assert result.ra_rad is not None
        assert len(result.ra_rad) == 0

    def test_combine_models_single_model(self, precision):
        """Single model -> same data as input."""
        sky = _make_point_sky(25, precision, seed=42)
        result = combine_models([sky], precision=precision)
        assert len(result.ra_rad) == 25
        np.testing.assert_allclose(result.ra_rad, sky.ra_rad)
        np.testing.assert_allclose(result.flux, sky.flux)

    def test_combine_models_point_sources(self, precision):
        """Two point-source models combined as point_sources representation."""
        sky_a = _make_point_sky(12, precision, seed=10)
        sky_b = _make_point_sky(18, precision, seed=20)

        result = combine_models([sky_a, sky_b], precision=precision)
        assert result.mode == SkyFormat.POINT_SOURCES
        assert len(result.ra_rad) == 30

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
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
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
            flux=rng.uniform(0.1, 10.0, n_a),
            spectral_index=np.full(n_a, -0.7),
            major_arcsec=np.full(n_a, 10.0),
            minor_arcsec=np.full(n_a, 5.0),
            pa_deg=np.full(n_a, 45.0),
            model_name="with_morph",
            reference_frequency=150e6,
            precision=precision,
        )
        sky_b = _make_point_sky(n_b, precision, seed=99)

        data = concat_point_sources([sky_a, sky_b])
        assert data["major_arcsec"] is not None
        assert len(data["major_arcsec"]) == n_a + n_b
        # B's entries should be zero-filled
        np.testing.assert_array_equal(data["major_arcsec"][n_a:], 0.0)
        np.testing.assert_array_equal(data["minor_arcsec"][n_a:], 0.0)
        np.testing.assert_array_equal(data["pa_deg"][n_a:], 0.0)
        # A's entries should be preserved
        np.testing.assert_allclose(data["major_arcsec"][:n_a], 10.0)

    def test_concat_with_spectral_coeffs_padding(self, precision):
        """Model A has 2-term spectral_coeffs, model B has single alpha -> padded."""
        sky_a = SkyModel.from_arrays(
            ra_rad=np.array([1.0]),
            dec_rad=np.array([0.5]),
            flux=np.array([5.0]),
            spectral_index=np.array([-0.7]),
            spectral_coeffs=np.array([[-0.7, -0.1]]),
            model_name="with_coeffs",
            reference_frequency=150e6,
            precision=precision,
        )
        sky_b = SkyModel.from_arrays(
            ra_rad=np.array([2.0]),
            dec_rad=np.array([-0.3]),
            flux=np.array([3.0]),
            spectral_index=np.array([-0.8]),
            model_name="no_coeffs",
            reference_frequency=150e6,
            precision=precision,
        )

        data = concat_point_sources([sky_a, sky_b])
        sp = data["spectral_coeffs"]
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
        sky1 = sky1.with_reference_frequency(100e6).with_healpix_maps(
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
        sky2 = sky2.with_reference_frequency(100e6).with_healpix_maps(
            nside=nside,
            frequencies=freqs,
        )

        # Build healpix-only models
        sky1_hp = SkyModel(
            _healpix_maps=sky1.healpix_maps,
            _healpix_q_maps=sky1.healpix_q_maps,
            _healpix_u_maps=sky1.healpix_u_maps,
            _healpix_v_maps=sky1.healpix_v_maps,
            _healpix_nside=nside,
            _observation_frequencies=freqs,
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
            model_name="hp1",
            _precision=precision,
        )
        sky2_hp = SkyModel(
            _healpix_maps=sky2.healpix_maps,
            _healpix_q_maps=sky2.healpix_q_maps,
            _healpix_u_maps=sky2.healpix_u_maps,
            _healpix_v_maps=sky2.healpix_v_maps,
            _healpix_nside=nside,
            _observation_frequencies=freqs,
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
            model_name="hp2",
            _precision=precision,
        )

        result = SkyModel.combine(
            [sky1_hp, sky2_hp],
            representation="healpix_map",
        )
        assert result.mode == SkyFormat.HEALPIX
        assert result.healpix_maps is not None
        assert result.healpix_maps.shape == (2, npix)

        # Where both models contribute, combined T_b should be >= each
        for freq_idx in range(2):
            m1 = sky1_hp.healpix_maps[freq_idx]
            m2 = sky2_hp.healpix_maps[freq_idx]
            combined = result.healpix_maps[freq_idx]
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
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
            model_name="nside8",
            _precision=precision,
        )
        sky_b = SkyModel(
            _healpix_maps=np.ones((2, hp.nside2npix(16)), dtype=np.float32) * 50,
            _healpix_nside=16,
            _observation_frequencies=freqs,
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
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
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
            model_name="freqs_a",
            _precision=precision,
        )
        sky_b = SkyModel(
            _healpix_maps=np.ones((2, npix), dtype=np.float32) * 50,
            _healpix_nside=nside,
            _observation_frequencies=np.array([200e6, 201e6]),
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=200e6,
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
        assert result.mode == SkyFormat.HEALPIX
        assert result.healpix_maps is not None
        assert result.healpix_nside == 8

    def test_combine_healpix_respects_precision_dtype(self):
        """combine_healpix output dtype matches precision.sky_model.healpix_maps."""
        nside = 8
        npix = hp.nside2npix(nside)
        freqs = np.array([100e6, 101e6])

        sky = SkyModel(
            _healpix_maps=np.ones((2, npix), dtype=np.float32) * 100.0,
            _healpix_nside=nside,
            _observation_frequencies=freqs,
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
            model_name="hp_prec",
            _precision=PrecisionConfig.precise(),
        )

        # precise() sets healpix_maps="float64"
        result = SkyModel.combine(
            [sky],
            representation="healpix_map",
            precision=PrecisionConfig.precise(),
        )
        assert result.healpix_maps.dtype == np.float64

        # fast() sets healpix_maps="float32"
        result_fast = SkyModel.combine(
            [sky],
            representation="healpix_map",
            precision=PrecisionConfig.fast(),
        )
        assert result_fast.healpix_maps.dtype == np.float32


# ---------------------------------------------------------------------------
# Per-source ref_freq in combination
# ---------------------------------------------------------------------------


class TestConcatRefFreq:
    def test_concat_preserves_per_source_ref_freq(self, precision):
        """Combining models with different ref_freq preserves per-source values."""
        sky_200 = SkyModel.from_arrays(
            ra_rad=np.array([0.0, 0.5]),
            dec_rad=np.array([0.0, 0.0]),
            flux=np.array([1.0, 2.0]),
            ref_freq=np.array([200e6, 200e6]),
            reference_frequency=200e6,
            precision=precision,
        )
        sky_1400 = SkyModel.from_arrays(
            ra_rad=np.array([1.0]),
            dec_rad=np.array([0.0]),
            flux=np.array([3.0]),
            ref_freq=np.array([1400e6]),
            reference_frequency=1400e6,
            precision=precision,
        )
        data = concat_point_sources([sky_200, sky_1400], precision=precision)
        assert "ref_freq" in data
        ref = data["ref_freq"]
        assert len(ref) == 3
        assert ref[0] == pytest.approx(200e6)
        assert ref[1] == pytest.approx(200e6)
        assert ref[2] == pytest.approx(1400e6)

    def test_combine_models_preserves_ref_freq(self, precision):
        """combine_models() with point_sources preserves per-source ref_freq."""
        sky_a = SkyModel.from_arrays(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux=np.array([1.0]),
            ref_freq=np.array([150e6]),
            reference_frequency=150e6,
            precision=precision,
        )
        sky_b = SkyModel.from_arrays(
            ra_rad=np.array([1.0]),
            dec_rad=np.array([0.0]),
            flux=np.array([2.0]),
            ref_freq=np.array([843e6]),
            reference_frequency=843e6,
            precision=precision,
        )
        combined = SkyModel.combine(
            [sky_a, sky_b],
            representation="point_sources",
            precision=precision,
        )
        assert combined.ref_freq is not None
        assert len(combined.ref_freq) == 2
        assert combined.ref_freq[0] == pytest.approx(150e6)
        assert combined.ref_freq[1] == pytest.approx(843e6)


# ---------------------------------------------------------------------------
# Auto-detect combine representation
# ---------------------------------------------------------------------------


class TestAutoRepresentation:
    def test_auto_picks_healpix_when_native(self, precision):
        """Auto-detection picks healpix_map when a model is HEALPix-native."""
        nside = 8
        npix = hp.nside2npix(nside)
        freqs = np.array([100e6, 110e6])
        sky_hp = SkyModel(
            _healpix_maps=np.ones((2, npix), dtype=np.float32),
            _healpix_nside=nside,
            _observation_frequencies=freqs,
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
            model_name="test_hp",
            _precision=precision,
        )
        combined = SkyModel.combine([sky_hp], precision=precision)
        assert combined.mode == SkyFormat.HEALPIX

    def test_auto_picks_point_sources_when_no_healpix(self, precision):
        """Auto-detection picks point_sources when no model is HEALPix-native."""
        sky = SkyModel.from_test_sources(
            num_sources=5,
            flux_range=(1.0, 5.0),
            dec_deg=-30.0,
            spectral_index=-0.7,
            precision=precision,
        )
        combined = SkyModel.combine([sky], precision=precision)
        assert combined.mode == SkyFormat.POINT_SOURCES


# ---------------------------------------------------------------------------
# Bug 2: Stale cache detection in concat_point_sources
# ---------------------------------------------------------------------------


class TestStaleCacheDetection:
    def test_concat_recomputes_for_different_frequency(self):
        """concat_point_sources must reconvert healpix models at the new frequency."""
        import healpy as hp

        nside = 4
        npix = hp.nside2npix(nside)
        freqs = np.array([100e6, 200e6])
        # Use a temperature that varies with frequency index to make the
        # T_b→Jy conversion yield different fluxes at different frequencies.
        i_maps = np.ones((2, npix), dtype=np.float64) * 500.0
        sky = SkyModel(
            _healpix_maps=i_maps,
            _healpix_nside=nside,
            _observation_frequencies=freqs,
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
            model_name="test_stale",
        )

        # First concat at 100 MHz — populates point-source cache
        data_100 = concat_point_sources([sky], reference_frequency=100e6)
        flux_100 = data_100["flux"]

        # Second concat at 200 MHz — must NOT reuse 100 MHz cache
        data_200 = concat_point_sources([sky], reference_frequency=200e6)
        flux_200 = data_200["flux"]

        # T_b→Jy conversion is frequency-dependent, so fluxes must differ
        assert len(flux_100) > 0
        assert len(flux_200) > 0
        assert not np.allclose(flux_100.sum(), flux_200.sum()), (
            "Flux sums should differ at different frequencies but were identical — "
            "stale cache was reused"
        )


# ---------------------------------------------------------------------------

# _empty_concat_data tests removed — functionality consolidated
# into _empty_source_arrays() in model.py (see test_sky_model.py).
