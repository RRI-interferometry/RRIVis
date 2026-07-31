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
from radiosim.core.sky.containers.constants import C_LIGHT, K_BOLTZMANN
from radiosim.core.sky.loaders import extragalactic as extragalactic_mod
from radiosim.core.sky.loaders.extragalactic import load_extragalactic_point_sources
from radiosim.core.sky.registry import loader_registry
from radiosim.core.sky.support.clustering import (
    clustered_pixel_rates,
    dither_positions_in_pixels,
    gaussian_overdensity_map,
    power_law_acf_to_cl,
)
from radiosim.core.sky.support.dnds import DNDS_MODELS
from radiosim.core.sky.support.healpy import lazy_healpy as hp


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
        # The isotropic mode (clustering_amp=0) draws one top-level Poisson
        # count; replicate the RNG stream independently.
        model = DNDS_MODELS["gervasi2008_150mhz"]
        cone = SkyRegion.cone(ra_deg=180.0, dec_deg=-30.0, radius_deg=10.0)
        flux_range = (1e-2, 1e-1)
        expected_n = model.integrated_counts(*flux_range) * cone.area_sr()
        seed = 7
        expected_count = int(np.random.default_rng(seed).poisson(expected_n))

        sky = load_extragalactic_point_sources(
            flux_range_jy=flux_range,
            clustering_amp=0.0,
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


class TestClusteringSupport:
    def test_acf_to_cl_roundtrip(self):
        # Reconstructing the 2PACF from the band-limited spectrum must
        # recover the input power law at angles the band limit resolves.
        amp, gamma, lmax = 7.8e-3, 0.821, 191
        cl = power_law_acf_to_cl(amp, gamma, lmax, zero_monopole=False)
        coeffs = (2.0 * np.arange(lmax + 1) + 1.0) * cl / (4.0 * np.pi)
        for chi_deg in (2.0, 5.0, 10.0):
            x = np.cos(np.radians(chi_deg))
            reconstructed = np.polynomial.legendre.legval(x, coeffs)
            expected = amp * chi_deg**-gamma
            assert reconstructed == pytest.approx(expected, rel=0.1)

    def test_cl_nonnegative_with_zeroed_monopole(self):
        cl = power_law_acf_to_cl(7.8e-3, 0.821, 191)
        assert cl[0] == 0.0
        assert np.all(cl >= 0.0)
        assert np.all(np.isfinite(cl))

    def test_acf_to_cl_rejects_bad_parameters(self):
        with pytest.raises(ValueError, match="amplitude"):
            power_law_acf_to_cl(0.0, 0.821, 63)
        with pytest.raises(ValueError, match="gamma"):
            power_law_acf_to_cl(7.8e-3, 2.5, 63)

    def test_overdensity_map_statistics_match_spectrum(self):
        nside, lmax = 32, 95
        cl = power_law_acf_to_cl(7.8e-3, 0.821, lmax)
        rng = np.random.default_rng(42)
        delta = gaussian_overdensity_map(cl, nside, rng)
        assert delta.size == hp.nside2npix(nside)
        expected_var = float(
            np.sum((2.0 * np.arange(lmax + 1) + 1.0) * cl) / (4.0 * np.pi)
        )
        assert abs(float(delta.mean())) < 0.02
        assert float(delta.var()) == pytest.approx(expected_var, rel=0.25)

    def test_clustered_pixel_rates_clip_and_error(self, caplog):
        delta = np.array([-1.5, -0.2, 0.0, 0.4, 1.0])
        with caplog.at_level("WARNING"):
            rates = clustered_pixel_rates(10.0, delta)
        assert rates[0] == 0.0
        assert np.all(rates[1:] > 0.0)
        assert any("clipped" in record.message for record in caplog.records)

        mostly_negative = np.full(100, -2.0)
        mostly_negative[:30] = 0.0
        with pytest.raises(ValueError, match="not valid"):
            clustered_pixel_rates(10.0, mostly_negative)

    def test_dither_positions_stay_in_parent_pixel(self):
        nside = 64
        pixels = np.repeat(np.array([7, 1000, 40000], dtype=np.int64), 50)
        rng = np.random.default_rng(5)
        ra_rad, dec_rad = dither_positions_in_pixels(pixels, nside, rng)
        recovered = hp.ang2pix(nside, np.pi / 2.0 - dec_rad, ra_rad)
        assert np.array_equal(recovered, pixels)
        # Positions must not all sit at the pixel centers.
        center_theta, center_phi = hp.pix2ang(nside, pixels)
        assert not np.allclose(ra_rad, center_phi)


class TestClusteredLoader:
    def test_clustered_realization_is_seed_reproducible(self, precision):
        cone = SkyRegion.cone(ra_deg=180.0, dec_deg=-30.0, radius_deg=5.0)
        kwargs = {"region": cone, "seed": 21, "precision": precision}
        sky_a = load_extragalactic_point_sources(**kwargs)
        sky_b = load_extragalactic_point_sources(**kwargs)
        assert sky_a.n_point_sources == sky_b.n_point_sources
        assert np.array_equal(sky_a.point.ra_rad, sky_b.point.ra_rad)
        assert np.array_equal(sky_a.point.flux, sky_b.point.flux)

    def test_clustered_counts_show_excess_variance(self, precision):
        nside = 32
        npix = hp.nside2npix(nside)
        nbar = (
            DNDS_MODELS["gervasi2008_150mhz"].integrated_counts(1e-2, 1e-1)
            * 4.0
            * np.pi
            / npix
        )

        def per_pixel_variance(sky):
            pix = hp.ang2pix(
                nside,
                np.pi / 2.0 - np.asarray(sky.point.dec_rad, dtype=np.float64),
                np.asarray(sky.point.ra_rad, dtype=np.float64),
            )
            counts = np.bincount(pix, minlength=npix)
            return float(np.var(counts))

        clustered = load_extragalactic_point_sources(
            nside=nside, seed=17, precision=precision
        )
        isotropic = load_extragalactic_point_sources(
            nside=nside, clustering_amp=0.0, seed=17, precision=precision
        )

        var_clustered = per_pixel_variance(clustered)
        var_isotropic = per_pixel_variance(isotropic)
        assert var_isotropic < 1.2 * nbar
        assert var_clustered > 1.4 * nbar
        assert var_clustered > 1.3 * var_isotropic

    def test_clustered_notes_and_provenance(self, precision):
        cone = SkyRegion.cone(ra_deg=180.0, dec_deg=-30.0, radius_deg=5.0)
        sky = load_extragalactic_point_sources(region=cone, seed=8, precision=precision)
        assert "clustered 2PACF" in sky.provenance.notes
        assert "Mittal" in sky.provenance.notes
        assert sky.provenance.rng_seed == 8


class TestStreamedHealpix:
    def test_streamed_isotropic_map_mean_matches_analytic(self, precision):
        nside = 32
        sky = load_extragalactic_point_sources(
            clustering_amp=0.0,
            representation="healpix_map",
            nside=nside,
            frequencies=np.asarray([150e6]),
            seed=23,
            precision=precision,
        )
        assert sky.healpix is not None
        maps = np.asarray(sky.healpix.maps, dtype=np.float64)
        assert maps.shape == (1, hp.nside2npix(nside))

        model = DNDS_MODELS["gervasi2008_150mhz"]
        grid = np.logspace(-2, -1, 20001)
        flux_per_sr = np.trapezoid(grid * model.dn_ds(grid), grid)
        expected_mean_k = (
            flux_per_sr * 1e-26 * C_LIGHT**2 / (2.0 * K_BOLTZMANN * 150e6**2)
        )
        assert float(maps.mean()) == pytest.approx(expected_mean_k, rel=0.01)

    def test_streamed_chunking_is_consistent(self, precision, monkeypatch):
        # Force many small chunks; the accumulated mean must stay on the
        # analytic value, exercising the chunk-boundary logic.
        monkeypatch.setattr(extragalactic_mod, "_STREAM_CHUNK_SOURCES", 1000)
        cone = SkyRegion.cone(ra_deg=180.0, dec_deg=-30.0, radius_deg=10.0)
        sky = load_extragalactic_point_sources(
            clustering_amp=0.0,
            representation="healpix_map",
            nside=32,
            frequencies=np.asarray([150e6]),
            region=cone,
            seed=29,
            precision=precision,
        )
        maps = np.asarray(sky.healpix.maps, dtype=np.float64)
        assert sky.healpix.hpx_inds is not None

        model = DNDS_MODELS["gervasi2008_150mhz"]
        grid = np.logspace(-2, -1, 20001)
        flux_per_sr = np.trapezoid(grid * model.dn_ds(grid), grid)
        expected_mean_k = (
            flux_per_sr * 1e-26 * C_LIGHT**2 / (2.0 * K_BOLTZMANN * 150e6**2)
        )
        assert float(maps.mean()) == pytest.approx(expected_mean_k, rel=0.06)

    def test_streamed_clustered_frequency_scaling(self, precision):
        cone = SkyRegion.cone(ra_deg=180.0, dec_deg=-30.0, radius_deg=5.0)
        sky = load_extragalactic_point_sources(
            representation="healpix_map",
            nside=32,
            frequencies=np.asarray([150e6, 180e6]),
            region=cone,
            seed=31,
            precision=precision,
        )
        assert sky.healpix is not None
        assert sky.healpix.hpx_inds is not None
        assert sky.n_frequencies == 2
        maps = np.asarray(sky.healpix.maps, dtype=np.float64)
        assert np.all(np.isfinite(maps))
        assert np.all(maps >= 0.0)
        assert "streamed HEALPix" in sky.provenance.notes

        # Brightness temperature scales as the mean flux power law times
        # the Rayleigh-Jeans nu^-2: (180/150)^alpha * (150/180)^2.
        ratio = float(maps[1].mean() / maps[0].mean())
        expected = (180.0 / 150.0) ** (-0.681) * (150.0 / 180.0) ** 2
        assert ratio == pytest.approx(expected, rel=0.05)


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
