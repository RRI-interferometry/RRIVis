"""Tests for the validated Poisson confusion workflow."""

from __future__ import annotations

import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import (
    MonopoleConvention,
    SkyCoverage,
    SkyRegion,
    SourceSubtractionStatus,
)
from radiosim.core.sky.loaders.synthetic import load_poisson_confusion
from radiosim.core.sky.support.dnds import DNDS_MODELS, DNDSModel, resolve_dn_ds


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


class TestDNDSModels:
    def test_franzen_normalization_matches_published_polynomial_at_1jy(self):
        model = DNDS_MODELS["franzen2019_gleam_154mhz"]
        dn_ds = model.dn_ds(np.asarray([1.0], dtype=np.float64))[0]
        assert dn_ds == pytest.approx(10.0**3.52, rel=1e-12)

    def test_integrated_counts_are_additive_across_subranges(self):
        model = DNDS_MODELS["franzen2019_gleam_154mhz"]
        low = model.integrated_counts(0.01, 0.1)
        high = model.integrated_counts(0.1, 1.0)
        total = model.integrated_counts(0.01, 1.0)
        assert total == pytest.approx(low + high, rel=5e-4)

    def test_sample_respects_requested_range(self):
        model = DNDS_MODELS["franzen2019_gleam_154mhz"]
        rng = np.random.default_rng(1)
        flux = model.sample_flux(5000, rng, 0.1, 1.0)
        assert flux.min() >= 0.1 - 1e-12
        assert flux.max() <= 1.0 + 1e-12

    def test_gervasi_normalization_matches_published_form_at_1jy(self):
        model = DNDS_MODELS["gervasi2008_150mhz"]
        norm_a1, norm_b1 = 1.65e-4, 1.14e-4
        norm_a2, norm_b2 = 0.24 * norm_a1, 1.8e7 * norm_b1
        expected = 1.0 / (norm_a1 + norm_b1) + 1.0 / (norm_a2 + norm_b2)
        dn_ds = model.dn_ds(np.asarray([1.0], dtype=np.float64))[0]
        assert dn_ds == pytest.approx(expected, rel=1e-12)

    def test_mandal_normalization_matches_published_polynomial_at_1mjy(self):
        model = DNDS_MODELS["mandal2021_lotss_150mhz"]
        dn_ds = model.dn_ds(np.asarray([1e-3], dtype=np.float64))[0]
        assert dn_ds == pytest.approx(10.0**1.655 * (1e-3) ** -2.5, rel=1e-12)

    def test_intema_normalization_matches_published_polynomial_at_1jy(self):
        model = DNDS_MODELS["intema2017_tgss_150mhz"]
        dn_ds = model.dn_ds(np.asarray([1.0], dtype=np.float64))[0]
        assert dn_ds == pytest.approx(10.0**3.5142, rel=1e-12)

    def test_gervasi_population_sizes_match_mittal2024(self):
        # Full-sky totals cross-checked against the population sizes the
        # epspy reference implementation of Mittal et al. 2024 documents:
        # ~1.77e6 sources for 10-100 mJy and ~4.4e9 for 1 uJy-100 mJy.
        model = DNDS_MODELS["gervasi2008_150mhz"]
        n_default = 4.0 * np.pi * model.integrated_counts(1e-2, 1e-1)
        n_deep = 4.0 * np.pi * model.integrated_counts(1e-6, 1e-1)
        assert n_default == pytest.approx(1.772e6, rel=1e-2)
        assert n_deep == pytest.approx(4.412e9, rel=1e-2)

    def test_150mhz_class_models_agree_at_bright_end(self):
        s = np.asarray([1.0], dtype=np.float64)
        reference = DNDS_MODELS["franzen2019_gleam_154mhz"].dn_ds(s)[0]
        for name in (
            "gervasi2008_150mhz",
            "mandal2021_lotss_150mhz",
            "intema2017_tgss_150mhz",
        ):
            ratio = DNDS_MODELS[name].dn_ds(s)[0] / reference
            assert 0.5 < ratio < 2.0

    def test_new_models_reject_flux_outside_validated_band(self):
        with pytest.raises(ValueError, match="outside the validated range"):
            DNDS_MODELS["mandal2021_lotss_150mhz"].integrated_counts(1e-4, 1.0)
        with pytest.raises(ValueError, match="outside the validated range"):
            DNDS_MODELS["intema2017_tgss_150mhz"].integrated_counts(1e-3, 1.0)
        with pytest.raises(ValueError, match="outside the validated range"):
            DNDS_MODELS["gervasi2008_150mhz"].integrated_counts(1e-7, 1e-1)

    def test_new_models_are_calibrated_at_150mhz(self):
        for name in (
            "gervasi2008_150mhz",
            "mandal2021_lotss_150mhz",
            "intema2017_tgss_150mhz",
        ):
            model = DNDS_MODELS[name]
            assert model.reference_frequency_hz == pytest.approx(150e6)
            assert model.validated

    def test_resolve_dn_ds_rejects_unknown_preset(self):
        with pytest.raises(KeyError, match="Unknown dN/dS preset"):
            resolve_dn_ds("nonexistent")

    def test_resolve_dn_ds_rejects_nonvalidated_model(self):
        validated = DNDS_MODELS["franzen2019_gleam_154mhz"]
        invalid = DNDSModel(
            name="invalid",
            reference_frequency_hz=validated.reference_frequency_hz,
            flux_valid_range_jy=validated.flux_valid_range_jy,
            dn_ds=validated.dn_ds,
            sample_flux=validated.sample_flux,
            integrated_counts=validated.integrated_counts,
            validated=False,
        )
        with pytest.raises(ValueError, match="not marked as validated"):
            resolve_dn_ds(invalid)

    def test_resolve_dn_ds_rejects_callable(self):
        with pytest.raises(TypeError, match="validated preset name"):
            resolve_dn_ds(lambda s: s)  # type: ignore[arg-type]


class TestPoissonLoader:
    def test_count_matches_seeded_poisson_draw(self, precision):
        model = DNDS_MODELS["franzen2019_gleam_154mhz"]
        flux_range = (0.5, 2.0)
        expected_n = model.integrated_counts(*flux_range) * (4.0 * np.pi)
        seed = 7
        expected_count = int(np.random.default_rng(seed).poisson(expected_n))

        sky = load_poisson_confusion(
            flux_range_jy=flux_range,
            reference_frequency=154e6,
            seed=seed,
            precision=precision,
        )

        assert sky.n_point_sources == expected_count

    def test_respects_region_area_and_partial_coverage(self, precision):
        cone = SkyRegion.cone(ra_deg=180.0, dec_deg=0.0, radius_deg=5.0)
        sky = load_poisson_confusion(
            flux_range_jy=(0.05, 1.0),
            reference_frequency=154e6,
            region=cone,
            seed=1,
            precision=precision,
        )

        inside = cone.contains(sky.point.ra_rad, sky.point.dec_rad)
        assert np.all(inside)
        assert sky.provenance.sky_coverage is SkyCoverage.PARTIAL_SKY
        assert sky.provenance.coverage_fraction is not None
        assert 0.0 < sky.provenance.coverage_fraction < 1.0
        assert sky.provenance.monopole_k is None

    def test_provenance_tagged_absolute_no_cmb(self, precision):
        sky = load_poisson_confusion(
            flux_range_jy=(0.1, 1.0),
            reference_frequency=154e6,
            seed=2,
            precision=precision,
        )
        prov = sky.provenance
        assert prov.monopole_convention is MonopoleConvention.ABSOLUTE_NO_CMB
        assert prov.source_subtraction is SourceSubtractionStatus.NONE
        assert prov.flux_completeness_jy == (0.1, 1.0)
        assert prov.flux_completeness_freq_hz == pytest.approx(154e6)
        assert prov.sky_coverage is SkyCoverage.FULL_SKY
        assert prov.coverage_fraction == pytest.approx(1.0)

    def test_healpix_representation(self, precision):
        sky = load_poisson_confusion(
            flux_range_jy=(0.1, 1.0),
            reference_frequency=154e6,
            representation="healpix_map",
            nside=32,
            frequencies=np.asarray([154e6]),
            seed=3,
            precision=precision,
        )
        assert sky.healpix is not None
        assert sky.healpix.nside == 32
        assert sky.n_frequencies == 1

    def test_zero_draw_preserves_provenance(self, precision):
        cone = SkyRegion.cone(ra_deg=0.0, dec_deg=-89.99, radius_deg=0.001)
        model = DNDS_MODELS["franzen2019_gleam_154mhz"]
        flux_range = (74.0, 75.0)
        expected_n = model.integrated_counts(*flux_range) * cone.area_sr()
        seed = 9
        expected_count = int(np.random.default_rng(seed).poisson(expected_n))
        assert expected_count == 0

        sky = load_poisson_confusion(
            flux_range_jy=flux_range,
            reference_frequency=154e6,
            region=cone,
            seed=seed,
            precision=precision,
        )

        assert sky.n_point_sources == 0
        assert sky.provenance.flux_completeness_jy == flux_range
        assert sky.provenance.sky_coverage is SkyCoverage.PARTIAL_SKY
        assert sky.provenance.monopole_k is None

    def test_zero_area_rejected(self, precision):
        with pytest.raises(ValueError, match="Effective area must be positive"):
            load_poisson_confusion(
                flux_range_jy=(0.1, 1.0),
                reference_frequency=154e6,
                area_sr=0.0,
                region=SkyRegion.cone(0.0, 0.0, 1.0),
                precision=precision,
            )

    def test_bad_flux_range_rejected(self, precision):
        with pytest.raises(ValueError, match="flux_range_jy"):
            load_poisson_confusion(
                flux_range_jy=(1.0, 0.1),
                reference_frequency=154e6,
                precision=precision,
            )

    def test_reference_frequency_must_match_model(self, precision):
        with pytest.raises(ValueError, match="calibration frequency exactly"):
            load_poisson_confusion(
                flux_range_jy=(0.1, 1.0),
                reference_frequency=150e6,
                precision=precision,
            )

    def test_flux_range_must_stay_within_validated_band(self, precision):
        with pytest.raises(ValueError, match="outside the validated range"):
            load_poisson_confusion(
                flux_range_jy=(1e-4, 1e-3),
                reference_frequency=154e6,
                precision=precision,
            )

    def test_area_sr_without_region_is_rejected(self, precision):
        with pytest.raises(ValueError, match="scientifically ambiguous"):
            load_poisson_confusion(
                flux_range_jy=(0.1, 1.0),
                reference_frequency=154e6,
                area_sr=1.0,
                precision=precision,
            )
