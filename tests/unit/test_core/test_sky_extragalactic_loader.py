"""Tests for the extragalactic point-source loader (Mittal et al. 2024)."""

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
from radiosim.core.sky.loaders.extragalactic import load_extragalactic_point_sources
from radiosim.core.sky.registry import loader_registry
from radiosim.core.sky.support.dnds import DNDS_MODELS


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


def _flux_moments(model_name: str, s_min: float, s_max: float) -> tuple[float, float]:
    """Analytic first/second flux moments of the dN/dS PDF on a band."""
    model = DNDS_MODELS[model_name]
    grid = np.logspace(np.log10(s_min), np.log10(s_max), 20001)
    pdf = model.dn_ds(grid)
    norm = np.trapezoid(pdf, grid)
    m1 = np.trapezoid(grid * pdf, grid) / norm
    m2 = np.trapezoid(grid**2 * pdf, grid) / norm
    return float(m1), float(m2)


class TestExtragalacticLoader:
    def test_count_matches_seeded_poisson_draw(self, precision):
        model = DNDS_MODELS["gervasi2008_150mhz"]
        cone = SkyRegion.cone(ra_deg=180.0, dec_deg=-30.0, radius_deg=10.0)
        flux_range = (1e-2, 1e-1)
        expected_n = model.integrated_counts(*flux_range) * cone.area_sr()
        seed = 7
        expected_count = int(np.random.default_rng(seed).poisson(expected_n))

        sky = load_extragalactic_point_sources(
            flux_range_jy=flux_range,
            region=cone,
            seed=seed,
            precision=precision,
        )

        assert sky.n_point_sources == expected_count

    def test_default_population_statistics(self, precision):
        cone = SkyRegion.cone(ra_deg=180.0, dec_deg=-30.0, radius_deg=10.0)
        sky = load_extragalactic_point_sources(
            region=cone,
            seed=11,
            precision=precision,
        )
        n = sky.n_point_sources
        assert n > 5_000

        flux = np.asarray(sky.point.flux, dtype=np.float64)
        assert flux.min() >= 1e-2 - 1e-12
        assert flux.max() <= 1e-1 + 1e-12

        # Sample mean flux must sit on the analytic dN/dS mean: this is the
        # measure-correct sampling contract (epspy's grid-weight draw would
        # land ~32% low on this band and fail here).
        m1, m2 = _flux_moments("gervasi2008_150mhz", 1e-2, 1e-1)
        standard_error = np.sqrt((m2 - m1**2) / n)
        assert abs(flux.mean() - m1) < 6.0 * standard_error

        alpha = np.asarray(sky.point.spectral_index, dtype=np.float64)
        assert abs(alpha.mean() - (-0.681)) < 5.0 * 0.5 / np.sqrt(n)
        assert alpha.std() == pytest.approx(0.5, rel=0.1)

        ref_freq = np.asarray(sky.point.ref_freq, dtype=np.float64)
        assert np.all(ref_freq == pytest.approx(150e6))

    def test_spectral_indices_are_not_clipped(self, precision):
        cone = SkyRegion.cone(ra_deg=10.0, dec_deg=-45.0, radius_deg=1.0)
        sky = load_extragalactic_point_sources(
            region=cone,
            seed=3,
            spectral_index_dist=(5.0, 0.0),
            precision=precision,
        )
        assert sky.n_point_sources > 0
        # A mean far outside the confusion loader's clip band survives intact.
        assert np.all(np.asarray(sky.point.spectral_index) == pytest.approx(5.0))

    def test_respects_region_and_partial_coverage(self, precision):
        cone = SkyRegion.cone(ra_deg=180.0, dec_deg=0.0, radius_deg=5.0)
        sky = load_extragalactic_point_sources(
            region=cone,
            seed=1,
            precision=precision,
        )

        inside = cone.contains(sky.point.ra_rad, sky.point.dec_rad)
        assert np.all(inside)
        assert sky.provenance.sky_coverage is SkyCoverage.PARTIAL_SKY
        assert sky.provenance.coverage_fraction is not None
        assert 0.0 < sky.provenance.coverage_fraction < 1.0

    def test_full_sky_provenance(self, precision):
        sky = load_extragalactic_point_sources(
            flux_range_jy=(1.0, 10.0),
            seed=2,
            precision=precision,
        )
        prov = sky.provenance
        assert prov.monopole_convention is MonopoleConvention.ABSOLUTE_NO_CMB
        assert prov.source_subtraction is SourceSubtractionStatus.NONE
        assert prov.flux_completeness_jy == (1.0, 10.0)
        assert prov.flux_completeness_freq_hz == pytest.approx(150e6)
        assert prov.sky_coverage is SkyCoverage.FULL_SKY
        assert prov.coverage_fraction == pytest.approx(1.0)
        assert prov.rng_seed == 2
        assert "Mittal" in prov.notes

    def test_healpix_representation(self, precision):
        cone = SkyRegion.cone(ra_deg=180.0, dec_deg=-30.0, radius_deg=5.0)
        sky = load_extragalactic_point_sources(
            flux_range_jy=(5e-2, 1e-1),
            region=cone,
            representation="healpix_map",
            nside=32,
            frequencies=np.asarray([150e6, 160e6]),
            seed=4,
            precision=precision,
        )
        assert sky.healpix is not None
        assert sky.healpix.nside == 32
        assert sky.n_frequencies == 2

    def test_zero_draw_preserves_provenance(self, precision):
        cone = SkyRegion.cone(ra_deg=0.0, dec_deg=-89.99, radius_deg=0.001)
        model = DNDS_MODELS["gervasi2008_150mhz"]
        flux_range = (10.0, 100.0)
        expected_n = model.integrated_counts(*flux_range) * cone.area_sr()
        seed = 9
        expected_count = int(np.random.default_rng(seed).poisson(expected_n))
        assert expected_count == 0

        sky = load_extragalactic_point_sources(
            flux_range_jy=flux_range,
            region=cone,
            seed=seed,
            precision=precision,
        )

        assert sky.n_point_sources == 0
        assert sky.provenance.flux_completeness_jy == flux_range
        assert sky.provenance.sky_coverage is SkyCoverage.PARTIAL_SKY
        assert sky.provenance.rng_seed == seed

    def test_expected_count_above_max_sources_rejected(self, precision):
        with pytest.raises(ValueError, match="max_sources"):
            load_extragalactic_point_sources(
                max_sources=100_000,
                precision=precision,
            )

    def test_reference_frequency_must_match_model(self, precision):
        with pytest.raises(ValueError, match="calibration frequency exactly"):
            load_extragalactic_point_sources(
                reference_frequency=154e6,
                precision=precision,
            )

    def test_flux_range_must_stay_within_validated_band(self, precision):
        with pytest.raises(ValueError, match="outside the validated range"):
            load_extragalactic_point_sources(
                flux_range_jy=(1e-7, 1e-1),
                precision=precision,
            )

    def test_bad_flux_range_rejected(self, precision):
        with pytest.raises(ValueError, match="flux_range_jy"):
            load_extragalactic_point_sources(
                flux_range_jy=(1.0, 0.1),
                precision=precision,
            )

    def test_bad_spectral_index_dist_rejected(self, precision):
        with pytest.raises(ValueError, match="spectral_index_dist"):
            load_extragalactic_point_sources(
                spectral_index_dist=(-0.681, -0.5),
                precision=precision,
            )

    def test_healpix_requires_frequencies(self, precision):
        with pytest.raises(ValueError, match="frequencies"):
            load_extragalactic_point_sources(
                flux_range_jy=(1.0, 10.0),
                representation="healpix_map",
                seed=5,
                precision=precision,
            )

    def test_alternative_dnds_presets_load(self, precision):
        cone = SkyRegion.cone(ra_deg=90.0, dec_deg=20.0, radius_deg=3.0)
        for preset in ("mandal2021_lotss_150mhz", "intema2017_tgss_150mhz"):
            sky = load_extragalactic_point_sources(
                flux_range_jy=(1e-2, 1e-1),
                dn_ds=preset,
                region=cone,
                seed=6,
                precision=precision,
            )
            assert sky.n_point_sources > 0
            assert preset in sky.provenance.notes


class TestRegistryIntegration:
    def test_loader_is_registered_with_aliases(self):
        definition = loader_registry.definition("extragalactic_point_sources")
        assert definition.category == "synthetic"
        assert definition.supports_point_sources
        assert definition.supports_healpix_map
        for alias in ("eps", "mittal2024"):
            resolved = loader_registry.resolve_callable(alias)
            assert resolved.canonical_name == "extragalactic_point_sources"

    def test_config_envelope_parses_options(self):
        from radiosim.io.config import parse_sky_source_config

        cfg = parse_sky_source_config(
            {
                "kind": "extragalactic_point_sources",
                "options": {"flux_range_jy": [0.01, 0.1], "seed": 5},
            }
        )
        assert cfg.kind == "extragalactic_point_sources"
        assert cfg.options["seed"] == 5
        assert tuple(cfg.options["flux_range_jy"]) == (0.01, 0.1)
