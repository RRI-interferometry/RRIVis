"""Tests for the strict scientific ``realistic_foreground`` recipe."""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import (
    MonopoleConvention,
    SkyCoverage,
    SkyRegion,
    realistic_foreground_sky,
)
from radiosim.core.sky.model import SkyFormat


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


@pytest.fixture
def fake_pygdsm(monkeypatch):
    """Install a deterministic fake pygdsm model so recipe tests stay local."""
    import radiosim.core.sky._loaders_diffuse as diffuse_mod

    class _FakePyGDSM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, freq):
            base = 10.0
            if self.kwargs.get("include_cmb", False):
                base += 2.7255
            return np.full(hp.nside2npix(32), base, dtype=np.float64)

    monkeypatch.setattr(diffuse_mod, "_resolve_model_class", lambda _p: _FakePyGDSM)


def _catalog_kwargs(**overrides):
    defaults = {
        "num_sources": 10,
        "flux_max": 20.0,
        "dec_deg": -30.0,
        "dec_range_deg": 30.0,
        "reference_frequency": 200e6,
        "seed": 11,
        "distribution": "random",
    }
    defaults.update(overrides)
    return defaults


class TestRecipeHappyPath:
    def test_haslam_plus_test_sources_region_is_partial_sky(
        self, precision, fake_pygdsm
    ):
        sky = realistic_foreground_sky(
            diffuse="haslam",
            bright_catalogs="test_sources",
            bright_catalog_kwargs=_catalog_kwargs(),
            bright_catalog_flux_min_jy=3.5,
            frequencies=np.array([150e6, 160e6]),
            nside=32,
            region=SkyRegion.cone(180.0, -30.0, 10.0),
            seed=42,
            precision=precision,
        )

        assert SkyFormat.HEALPIX in sky.formats
        assert sky.healpix is not None
        assert sky.healpix.nside == 32
        assert sky.n_frequencies == 2
        assert sky.provenance.monopole_convention is MonopoleConvention.ABSOLUTE_NO_CMB
        assert sky.provenance.sky_coverage is SkyCoverage.PARTIAL_SKY
        assert sky.provenance.coverage_fraction is not None
        assert 0.0 < sky.provenance.coverage_fraction < 1.0
        assert sky.provenance.monopole_k is None

    def test_include_cmb_promotes_convention_without_double_add(
        self, precision, fake_pygdsm
    ):
        base_kwargs = {
            "diffuse": "haslam",
            "bright_catalogs": "test_sources",
            "bright_catalog_kwargs": _catalog_kwargs(seed=3),
            "bright_catalog_flux_min_jy": 3.5,
            "frequencies": np.array([150e6]),
            "nside": 16,
            "precision": precision,
        }
        sky_no_cmb = realistic_foreground_sky(include_cmb=False, **base_kwargs)
        sky_with_cmb = realistic_foreground_sky(include_cmb=True, **base_kwargs)

        assert (
            sky_with_cmb.provenance.monopole_convention
            is MonopoleConvention.ABSOLUTE_WITH_CMB
        )
        assert sky_no_cmb.provenance.sky_coverage is SkyCoverage.FULL_SKY
        assert sky_with_cmb.provenance.sky_coverage is SkyCoverage.FULL_SKY
        assert sky_with_cmb.healpix.maps[0].mean() == pytest.approx(
            sky_no_cmb.healpix.maps[0].mean() + 2.7255,
            rel=1e-4,
        )
        assert sky_with_cmb.provenance.monopole_k == pytest.approx(
            sky_no_cmb.provenance.monopole_k + 2.7255,
            rel=1e-4,
        )

    def test_gsm2016_rejected_in_scientific_mode(self, precision, fake_pygdsm):
        with pytest.raises(ValueError, match="only accepts pre-subtracted diffuse"):
            realistic_foreground_sky(
                diffuse="gsm2016",
                bright_catalogs="test_sources",
                bright_catalog_kwargs=_catalog_kwargs(seed=5),
                bright_catalog_flux_min_jy=3.5,
                frequencies=np.array([150e6]),
                nside=16,
                precision=precision,
            )


class TestThresholdChain:
    def test_catalog_floor_below_scaled_haslam_cut_raises(self, precision, fake_pygdsm):
        with pytest.raises(ValueError, match="Threshold-chain violation"):
            realistic_foreground_sky(
                diffuse="haslam",
                bright_catalogs="test_sources",
                bright_catalog_kwargs=_catalog_kwargs(seed=7),
                bright_catalog_flux_min_jy=2.0,
                frequencies=np.array([150e6]),
                nside=16,
                precision=precision,
            )

    def test_confusion_requires_fully_subtracted_diffuse(self, precision, fake_pygdsm):
        with pytest.raises(ValueError, match="source_subtraction=ALL"):
            realistic_foreground_sky(
                diffuse="haslam",
                bright_catalogs="test_sources",
                bright_catalog_kwargs=_catalog_kwargs(seed=7),
                bright_catalog_flux_min_jy=4.0,
                confusion_flux_range_jy=(0.1, 3.0),
                frequencies=np.array([150e6]),
                nside=16,
                precision=precision,
            )


class TestRegistryIntegration:
    def test_recipe_is_a_registered_loader(self):
        from radiosim.core.sky._registry import get_loader, get_loader_definition

        loader = get_loader("realistic_foreground")
        assert loader.__name__ == "realistic_foreground_sky"
        definition = get_loader_definition("realistic_foreground")
        assert definition.category == "synthetic"
        assert definition.representations == ("healpix_map",)

    def test_yaml_config_parses(self):
        from radiosim.io.config import parse_sky_source_config

        spec = parse_sky_source_config(
            {
                "kind": "realistic_foreground",
                "diffuse": "haslam",
                "bright_catalogs": "gleam",
                "bright_catalog_flux_min_jy": 3.5,
                "nside": 128,
            }
        )
        assert spec.kind == "realistic_foreground"
        assert spec.bright_catalog_flux_min_jy == pytest.approx(3.5)
        assert spec.confusion_dn_ds == "franzen2019_gleam_154mhz"

    def test_provenance_override_in_yaml(self, precision):
        """A provenance_override dict on any SkySourceConfig is forwarded."""
        from radiosim.core.sky._registry import get_loader
        from radiosim.io.config import parse_sky_source_config

        spec = parse_sky_source_config(
            {
                "kind": "test_sources",
                "num_sources": 3,
                "flux_min": 1.0,
                "flux_max": 5.0,
                "provenance_override": {
                    "monopole_convention": "absolute_no_cmb",
                    "sky_coverage": "full_sky",
                    "coverage_fraction": 1.0,
                    "notes": "from-yaml",
                },
            }
        )
        name, kwargs = spec.to_loader_request()
        loader = get_loader(name)
        sky = loader(precision=precision, **kwargs)
        assert sky.provenance.notes == "from-yaml"
        assert sky.provenance.monopole_convention is MonopoleConvention.ABSOLUTE_NO_CMB
        assert sky.provenance.sky_coverage is SkyCoverage.FULL_SKY
