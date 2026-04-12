"""Tests for the public sky loader registry surface."""

import numpy as np
import pytest

import rrivis.core.sky as sky_public
from rrivis.core.sky._registry import _LOADER_META, _LOADERS, _REGISTRY
from rrivis.core.sky.registry import (
    build_alias_map,
    build_network_services_map,
    build_sky_model_map,
    get_loader,
    get_loader_definition,
    get_loader_meta,
    list_loaders,
    register_loader,
)
from rrivis.io.config import SkyModelConfig, SkySourceConfig, VisibilityConfig


class TestRegistry:
    def test_register_and_get_loader(self):
        """A registered loader is retrievable from the public registry surface."""

        @register_loader("_test_dummy_loader")
        def _dummy_loader(**kwargs):
            return "dummy"

        try:
            retrieved = get_loader("_test_dummy_loader")
            assert retrieved is _dummy_loader
            assert retrieved() == "dummy"
        finally:
            _LOADERS.pop("_test_dummy_loader", None)
            _LOADER_META.pop("_test_dummy_loader", None)
            _REGISTRY._definitions.pop("_test_dummy_loader", None)
            _REGISTRY._aliases = {
                alias: canonical
                for alias, canonical in _REGISTRY._aliases.items()
                if canonical != "_test_dummy_loader"
            }

    def test_get_unknown_loader_raises(self):
        with pytest.raises(ValueError, match="Unknown sky model loader"):
            get_loader("__nonexistent_loader_xyz__")

    def test_list_loaders_sorted(self):
        names = list_loaders()
        assert names == sorted(names)
        assert names

    def test_all_expected_loaders_registered(self):
        expected = {
            "gleam",
            "mals",
            "vlssr",
            "tgss",
            "wenss",
            "sumss",
            "nvss",
            "lotss",
            "3c",
            "vlass",
            "racs",
            "diffuse_sky",
            "pysm3",
            "pyradiosky_file",
            "bbs",
            "fits_image",
            "test_sources",
        }
        assert expected <= set(list_loaders())

    def test_all_loaders_have_metadata(self):
        for name in list_loaders():
            meta = get_loader_meta(name)
            assert "config_section" in meta
            assert "use_flag" in meta
            assert "is_healpix" in meta
            assert "requires_file" in meta
            assert get_loader_definition(name).name == name

    def test_build_sky_model_map_covers_non_file_loaders(self):
        sky_map = build_sky_model_map()
        for name, meta in _LOADER_META.items():
            if meta.get("requires_file", False):
                assert name not in sky_map
                continue
            section, flag, is_hp = sky_map[name]
            assert section == meta["config_section"]
            assert flag == meta["use_flag"]
            assert is_hp == meta["is_healpix"]

    def test_build_sky_model_map_excludes_file_loaders(self):
        sky_map = build_sky_model_map()
        for file_loader in ("bbs", "fits_image", "pyradiosky_file"):
            assert file_loader not in sky_map


class TestRegistryMetadata:
    def test_network_services_map(self):
        svc_map = build_network_services_map()
        assert svc_map["gleam"] == "vizier"
        assert svc_map["racs"] == "casda"
        assert svc_map["diffuse_sky"] == "pygdsm_data"
        assert svc_map["pysm3"] == "pysm3_data"
        assert "bbs" not in svc_map
        assert "fits_image" not in svc_map

    def test_alias_map(self):
        alias_map = build_alias_map()
        for alias in ("gsm", "gsm2008", "gsm2016", "lfsm", "haslam"):
            assert alias_map[alias] == "diffuse_sky"
        for name in list_loaders():
            assert name not in alias_map


class TestSourceSpecs:
    def test_simple_catalog_request(self):
        kind, kwargs = SkySourceConfig(kind="vlssr", flux_limit=2.0).to_loader_request(
            flux_multiplier=1e-3,
            region="mock_region",
            brightness_conversion="rayleigh-jeans",
        )
        assert kind == "vlssr"
        assert kwargs["flux_limit"] == pytest.approx(0.002)
        assert kwargs["region"] == "mock_region"
        assert kwargs["brightness_conversion"] == "rayleigh-jeans"

    def test_diffuse_request_uses_explicit_frequencies(self):
        freqs = np.array([100e6, 101e6])
        kind, kwargs = SkySourceConfig(kind="gsm").to_loader_request(
            frequencies=freqs,
            obs_frequency_config={"starting_frequency": 100.0},
        )
        assert kind == "diffuse_sky"
        assert kwargs["model"] == "gsm"
        np.testing.assert_array_equal(kwargs["frequencies"], freqs)
        assert "obs_frequency_config" not in kwargs

    def test_test_sources_request_preserves_representation(self):
        freqs = np.array([150e6, 151e6])
        kind, kwargs = SkySourceConfig(
            kind="test_healpix",
            nside=32,
            flux_min=3.0,
            flux_max=7.0,
        ).to_loader_request(
            flux_multiplier=1e-3,
            frequencies=freqs,
        )
        assert kind == "test_sources"
        assert kwargs["representation"] == "healpix_map"
        assert kwargs["nside"] == 32
        assert kwargs["flux_min"] == pytest.approx(0.003)
        assert kwargs["flux_max"] == pytest.approx(0.007)
        np.testing.assert_array_equal(kwargs["frequencies"], freqs)

    def test_file_loader_request_is_explicit(self):
        kind, kwargs = SkySourceConfig(
            kind="pyradiosky",
            filename="mock.skyh5",
            flux_limit=5.0,
            reference_frequency_hz=150e6,
        ).to_loader_request(
            flux_multiplier=1e-3,
            obs_frequency_config={"starting_frequency": 100.0},
        )
        assert kind == "pyradiosky_file"
        assert kwargs["filename"] == "mock.skyh5"
        assert kwargs["flux_limit"] == pytest.approx(0.005)
        assert kwargs["reference_frequency_hz"] == 150e6
        assert "obs_frequency_config" in kwargs


class TestPublicBoundary:
    def test_registry_helpers_not_reexported_from_sky_root(self):
        assert not hasattr(sky_public, "register_loader")
        assert not hasattr(sky_public, "build_loader_kwargs")
        assert not hasattr(sky_public, "list_loaders")

    def test_config_models_expose_new_policy_fields(self):
        assert "mixed_model_policy" in SkyModelConfig.model_fields
        assert "allow_lossy_point_materialization" in VisibilityConfig.model_fields


class TestLoaderSignatureConsistency:
    def test_all_loaders_accept_brightness_conversion(self):
        import inspect

        for name in list_loaders():
            if name.startswith("_test_"):
                continue
            assert (
                "brightness_conversion"
                in inspect.signature(get_loader(name)).parameters
            )

    def test_all_loaders_accept_precision(self):
        import inspect

        for name in list_loaders():
            if name.startswith("_test_"):
                continue
            assert "precision" in inspect.signature(get_loader(name)).parameters

    def test_all_loaders_accept_region(self):
        import inspect

        for name in list_loaders():
            if name.startswith("_test_"):
                continue
            assert "region" in inspect.signature(get_loader(name)).parameters
