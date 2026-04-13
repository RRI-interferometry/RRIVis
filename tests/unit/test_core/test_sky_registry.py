"""Tests for the public sky loader registry surface."""

import numpy as np
import pytest
from pydantic import ValidationError

import rrivis.core.sky as sky_public
import rrivis.core.sky.registry as registry_public
from rrivis.core.sky._registry import _LOADER_META, _LOADERS, _REGISTRY
from rrivis.core.sky.discovery import get_catalog_info
from rrivis.core.sky.registry import loader_registry
from rrivis.io.config import (
    DiffuseSkySourceConfig,
    GleamSourceConfig,
    PyradioskyFileSourceConfig,
    SkyModelConfig,
    TestSourcesConfig,
    VisibilityConfig,
    VlssrSourceConfig,
    parse_sky_source_config,
)


class TestRegistry:
    def test_register_and_get_loader(self):
        """A registered loader is retrievable from the public registry surface."""

        @loader_registry.register("_test_dummy_loader")
        def _dummy_loader(**kwargs):
            return "dummy"

        try:
            retrieved = loader_registry.loader("_test_dummy_loader")
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
            _REGISTRY._alias_defaults = {
                alias: defaults
                for alias, defaults in _REGISTRY._alias_defaults.items()
                if _REGISTRY._aliases.get(alias) != "_test_dummy_loader"
            }

    def test_get_unknown_loader_raises(self):
        with pytest.raises(ValueError, match="Unknown sky model loader"):
            loader_registry.loader("__nonexistent_loader_xyz__")

    def test_list_loaders_sorted(self):
        names = loader_registry.names()
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
        assert expected <= set(loader_registry.names())

    def test_all_loaders_have_metadata(self):
        for name in loader_registry.names():
            meta = loader_registry.meta(name)
            assert "config_section" in meta
            assert "use_flag" in meta
            assert "representations" in meta
            assert "output_mode" in meta
            assert "supports_point_sources" in meta
            assert "supports_healpix_map" in meta
            assert "requires_file" in meta
            assert loader_registry.definition(name).name == name

    def test_definitions_cover_non_file_loaders(self):
        definitions = {
            definition.name: definition for definition in loader_registry.definitions()
        }
        for name, meta in _LOADER_META.items():
            definition = definitions[name]
            if definition.requires_file:
                continue
            assert definition.config_section == meta["config_section"]
            assert (definition.use_flag or f"use_{definition.name}") == meta["use_flag"]
            assert list(definition.representations) == meta["representations"]
            assert definition.output_mode == meta["output_mode"]
            assert definition.supports_point_sources == meta["supports_point_sources"]
            assert definition.supports_healpix_map == meta["supports_healpix_map"]

    def test_file_loaders_are_explicitly_marked(self):
        definitions = {
            definition.name: definition for definition in loader_registry.definitions()
        }
        for file_loader in ("bbs", "fits_image", "pyradiosky_file"):
            assert definitions[file_loader].requires_file


class TestRegistryMetadata:
    def test_network_services_map(self):
        svc_map = loader_registry.network_services()
        assert svc_map["gleam"] == "vizier"
        assert svc_map["racs"] == "casda"
        assert svc_map["diffuse_sky"] == "pygdsm_data"
        assert svc_map["pysm3"] == "pysm3_data"
        assert "bbs" not in svc_map
        assert "fits_image" not in svc_map

    def test_alias_map(self):
        alias_map = loader_registry.aliases()
        for alias in ("gsm", "gsm2008", "gsm2016", "lfsm", "haslam"):
            assert alias_map[alias] == "diffuse_sky"
        for name in loader_registry.names():
            assert name not in alias_map

    def test_alias_defaults_map(self):
        defaults = loader_registry.alias_defaults()
        assert defaults["gsm"] == {"model": "gsm2008"}
        assert defaults["gsm2016"] == {"model": "gsm2016"}
        assert defaults["test_healpix"] == {"representation": "healpix_map"}

    def test_canonical_loader_metadata_reports_all_capabilities(self):
        meta = loader_registry.meta("test_sources")
        assert meta["representations"] == ["point_sources", "healpix_map"]
        assert meta["output_mode"] == "polymorphic"
        assert meta["supports_point_sources"] is True
        assert meta["supports_healpix_map"] is True

    def test_alias_metadata_reflects_aliased_representation(self):
        meta = loader_registry.meta("test_healpix")
        assert meta["representations"] == ["healpix_map"]
        assert meta["output_mode"] == "healpix_only"
        assert meta["supports_point_sources"] is False
        assert meta["supports_healpix_map"] is True

    def test_discovery_catalog_info_exposes_capabilities(self):
        info = get_catalog_info("test_healpix")
        assert info["representations"] == ["healpix_map"]
        assert info["output_mode"] == "healpix_only"
        assert info["primary_representation"] == "healpix_map"
        assert info["supports_point_sources"] is False
        assert info["supports_healpix_map"] is True
        assert info["resolved_loader"] == "test_sources"
        assert info["resolved_kwargs"] == {"representation": "healpix_map"}

    def test_discovery_catalog_info_exposes_diffuse_alias_metadata(self):
        info = get_catalog_info("gsm2016")
        assert info["loader"] == "diffuse_sky"
        assert info["resolved_loader"] == "diffuse_sky"
        assert info["resolved_kwargs"] == {"model": "gsm2016"}
        assert info["diffuse_model"] == "gsm2016"
        assert info["diffuse_model_info"]["class_name"] == "GlobalSkyModel16"

    def test_resolve_loader_request_merges_alias_defaults(self):
        kind, kwargs = loader_registry.resolve_request("gsm2016", {"nside": 128})
        assert kind == "diffuse_sky"
        assert kwargs == {"model": "gsm2016", "nside": 128}

        kind, kwargs = loader_registry.resolve_request("gsm2016", {"model": "haslam"})
        assert kind == "diffuse_sky"
        assert kwargs == {"model": "haslam"}


class TestSourceSpecs:
    def test_simple_catalog_request(self):
        spec = parse_sky_source_config(
            {"kind": "vlssr", "flux_limit": 2.0, "max_rows": 1000}
        )
        assert isinstance(spec, VlssrSourceConfig)
        assert spec.kind == "vlssr"

        kind, kwargs = spec.to_loader_request(
            flux_multiplier=1e-3,
            region="mock_region",
            brightness_conversion="rayleigh-jeans",
        )
        assert kind == "vlssr"
        assert kwargs["flux_limit"] == pytest.approx(0.002)
        assert kwargs["max_rows"] == 1000
        assert kwargs["region"] == "mock_region"
        assert kwargs["brightness_conversion"] == "rayleigh-jeans"

    def test_diffuse_request_uses_explicit_frequencies(self):
        freqs = np.array([100e6, 101e6])
        spec = parse_sky_source_config({"kind": "diffuse_sky", "model": "gsm2008"})
        assert isinstance(spec, DiffuseSkySourceConfig)

        kind, kwargs = spec.to_loader_request(
            frequencies=freqs,
            obs_frequency_config={"starting_frequency": 100.0},
        )
        assert kind == "diffuse_sky"
        assert kwargs["model"] == "gsm2008"
        np.testing.assert_array_equal(kwargs["frequencies"], freqs)
        assert "obs_frequency_config" not in kwargs

    def test_diffuse_alias_preserves_selected_model(self):
        spec = parse_sky_source_config({"kind": "diffuse_sky", "model": "gsm2016"})
        kind, kwargs = spec.to_loader_request()
        assert kind == "diffuse_sky"
        assert kwargs["model"] == "gsm2016"

    def test_diffuse_alias_kind_applies_alias_defaults(self):
        spec = parse_sky_source_config({"kind": "gsm2016", "nside": 64})
        kind, kwargs = spec.to_loader_request()
        assert kind == "diffuse_sky"
        assert kwargs["model"] == "gsm2016"
        assert kwargs["nside"] == 64

    def test_diffuse_alias_kind_allows_explicit_override(self):
        spec = parse_sky_source_config(
            {"kind": "gsm2016", "model": "haslam", "nside": 128}
        )
        kind, kwargs = spec.to_loader_request()
        assert kind == "diffuse_sky"
        assert kwargs["model"] == "haslam"
        assert kwargs["nside"] == 128

    def test_test_sources_request_preserves_representation(self):
        freqs = np.array([150e6, 151e6])
        spec = parse_sky_source_config(
            {
                "kind": "test_sources",
                "representation": "healpix_map",
                "nside": 32,
                "flux_min": 3.0,
                "flux_max": 7.0,
            }
        )
        assert isinstance(spec, TestSourcesConfig)

        kind, kwargs = spec.to_loader_request(
            flux_multiplier=1e-3,
            frequencies=freqs,
        )
        assert kind == "test_sources"
        assert kwargs["representation"] == "healpix_map"
        assert kwargs["nside"] == 32
        assert kwargs["flux_min"] == pytest.approx(0.003)
        assert kwargs["flux_max"] == pytest.approx(0.007)
        np.testing.assert_array_equal(kwargs["frequencies"], freqs)

    def test_test_sources_alias_kind_applies_representation_default(self):
        freqs = np.array([150e6, 151e6])
        spec = parse_sky_source_config({"kind": "test_healpix", "nside": 32})
        kind, kwargs = spec.to_loader_request(frequencies=freqs)
        assert kind == "test_sources"
        assert kwargs["representation"] == "healpix_map"
        assert kwargs["nside"] == 32
        np.testing.assert_array_equal(kwargs["frequencies"], freqs)

    def test_file_loader_request_is_explicit(self):
        spec = parse_sky_source_config(
            {
                "kind": "pyradiosky_file",
                "filename": "mock.skyh5",
                "flux_limit": 5.0,
                "reference_frequency_hz": 150e6,
            }
        )
        assert isinstance(spec, PyradioskyFileSourceConfig)

        kind, kwargs = spec.to_loader_request(
            flux_multiplier=1e-3,
            obs_frequency_config={"starting_frequency": 100.0},
        )
        assert kind == "pyradiosky_file"
        assert kwargs["filename"] == "mock.skyh5"
        assert kwargs["flux_limit"] == pytest.approx(0.005)
        assert kwargs["reference_frequency_hz"] == 150e6
        assert "obs_frequency_config" in kwargs

    def test_legacy_nested_sky_model_sections_are_rejected(self):
        with pytest.raises(ValueError, match="sources"):
            SkyModelConfig.model_validate(
                {
                    "gleam": {
                        "use_gleam": True,
                        "flux_limit": 1.0,
                    }
                }
            )

    def test_giant_source_model_replaced_by_discriminated_union(self):
        spec = parse_sky_source_config({"kind": "gleam", "flux_limit": 1.0})
        assert isinstance(spec, GleamSourceConfig)
        assert spec.kind == "gleam"
        assert spec.catalog == "gleam_egc"

    def test_unexpected_fields_are_rejected_by_loader_model(self):
        with pytest.raises(ValidationError):
            parse_sky_source_config({"kind": "gleam", "nside": 64})


class TestPublicBoundary:
    def test_registry_helpers_not_reexported_from_sky_root(self):
        assert not hasattr(sky_public, "register_loader")
        assert not hasattr(sky_public, "build_loader_kwargs")
        assert not hasattr(sky_public, "list_loaders")
        assert not hasattr(sky_public, "loader_registry")
        assert not hasattr(sky_public, "create_from_freq_dict_maps")
        assert not hasattr(sky_public, "load_models_parallel")
        assert not hasattr(sky_public, "prepare_sky_model")
        assert not hasattr(sky_public, "bin_sources_to_flux")
        assert not hasattr(sky_public, "DiffuseModelEntry")
        assert not hasattr(sky_public, "VizierCatalogEntry")
        assert not hasattr(sky_public, "RacsCatalogEntry")
        assert hasattr(sky_public, "write_bbs")

    def test_registry_module_wrappers_are_removed(self):
        assert not hasattr(registry_public, "get_loader")
        assert not hasattr(registry_public, "list_loaders")
        assert not hasattr(registry_public, "resolve_loader_name")
        assert not hasattr(registry_public, "register_loader")

    def test_config_models_expose_new_policy_fields(self):
        assert "mixed_model_policy" in SkyModelConfig.model_fields
        assert "allow_lossy_point_materialization" in VisibilityConfig.model_fields
