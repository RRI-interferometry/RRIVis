"""Tests for the public sky loader registry surface."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import radiosim.core.sky as sky_public
import radiosim.core.sky.registry.core as registry_core
import radiosim.core.sky.registry.facade as registry_public
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.diagnostics.discovery import get_catalog_info
from radiosim.core.sky.loaders.vizier.core import _load_from_vizier_catalog
from radiosim.core.sky.loaders.vizier.point_catalogs import VIZIER_POINT_CATALOGS
from radiosim.core.sky.registry import loader_registry
from radiosim.io.config import (
    CustomRegisteredSourceConfig,
    DiffuseSkySourceConfig,
    GleamSourceConfig,
    PyradioskyFileSourceConfig,
    SkyModelConfig,
    TestSourcesConfig,
    VisibilityConfig,
    parse_sky_source_config,
)


class TestRegistry:
    def test_register_and_get_loader(self):
        """A registered loader is retrievable from the public registry surface."""

        @loader_registry.register_loader("_test_dummy_loader")
        def _dummy_loader(**kwargs):
            return "dummy"

        try:
            retrieved = loader_registry.loader("_test_dummy_loader")
            assert retrieved is _dummy_loader
            assert retrieved() == "dummy"
        finally:
            registry_core._REGISTRY.unregister("_test_dummy_loader")

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
            assert "requires_file" in meta
            assert loader_registry.definition(name).name == name

    def test_definitions_cover_non_file_loaders(self):
        definitions = {
            definition.name: definition for definition in loader_registry.definitions()
        }
        for name, definition in definitions.items():
            if definition.requires_file:
                continue
            meta = loader_registry.meta(name)
            assert definition.config_section == meta["config_section"]
            assert (definition.use_flag or f"use_{definition.name}") == meta["use_flag"]
            assert list(definition.representations) == meta["representations"]
            assert definition.output_mode == meta["output_mode"]
            assert definition.supports_point_sources == (
                "point_sources" in meta["representations"]
            )
            assert definition.supports_healpix_map == (
                "healpix_map" in meta["representations"]
            )

    def test_file_loaders_are_explicitly_marked(self):
        definitions = {
            definition.name: definition for definition in loader_registry.definitions()
        }
        for file_loader in ("bbs", "fits_image", "pyradiosky_file"):
            assert definitions[file_loader].requires_file


class TestRegistryMetadata:
    def test_network_services_map(self):
        svc_map = loader_registry.network_services()
        assert svc_map["gleam"] == ("vizier",)
        assert svc_map["racs"] == ("casda",)
        assert svc_map["diffuse_sky"] == ("pygdsm_data",)
        assert svc_map["pysm3"] == ("pysm3_data",)
        # A composite recipe declares every service it can reach (SKY-002).
        assert svc_map["realistic_foreground"] == ("pygdsm_data", "vizier")
        assert "bbs" not in svc_map
        assert "fits_image" not in svc_map

    def test_every_network_implicated_loader_declares_a_service(self):
        """``SKY-002``'s generalization: no loader reaches the network silently.

        The defect Tier 8D closed was not that one declaration was wrong; it
        was that a *composite* loader had no way to say what it reached, so the
        pre-flight lied about a shipped configuration. This scan is what stops
        the next one repeating it. A loader is network-implicated when the
        module defining it names a network client, or when it resolves other
        loaders dynamically -- ``realistic_foreground`` does the second, which
        is exactly why an import scan alone would have missed it -- and a
        network-implicated loader must declare at least one service.
        """
        import inspect

        network_clients = (
            "astroquery",
            "pygdsm",
            "pysm3",
            "TapPlus",
            "Vizier",
            "require_service(",
            "urllib.request",
            "requests.get",
        )
        dynamic_dispatch = (
            "loader_registry.resolve_callable(",
            "loader_registry.loader(",
        )

        undeclared: list[str] = []
        implicated: set[str] = set()
        for definition in loader_registry.definitions():
            source_file = inspect.getsourcefile(definition.loader)
            assert source_file is not None, definition.name
            text = Path(source_file).read_text(encoding="utf-8")
            reasons = [token for token in network_clients if token in text]
            reasons += [token for token in dynamic_dispatch if token in text]
            if not reasons:
                continue
            implicated.add(definition.name)
            if not definition.network_services:
                undeclared.append(
                    f"{definition.name} ({Path(source_file).name}) names "
                    f"{reasons} but declares network_services=(); add every "
                    "service it can reach, including through loaders it "
                    "dispatches to"
                )
        assert undeclared == []
        assert "realistic_foreground" in implicated
        assert "diffuse_sky" in implicated
        assert "racs" in implicated

    def test_alias_map(self):
        alias_map = loader_registry.aliases()
        for alias in ("gsm", "gsm2008", "gsm2016", "lfsm", "haslam"):
            assert alias_map[alias] == "diffuse_sky"
        for name in loader_registry.names():
            assert name not in alias_map

    def test_alias_defaults_via_definition(self):
        diffuse = loader_registry.definition("diffuse_sky")
        assert diffuse.alias_defaults["gsm"] == {"model": "gsm2008"}
        assert diffuse.alias_defaults["gsm2016"] == {"model": "gsm2016"}
        test_src = loader_registry.definition("test_sources")
        assert test_src.alias_defaults["test_healpix"] == {
            "representation": "healpix_map"
        }

    def test_canonical_loader_metadata_reports_all_capabilities(self):
        meta = loader_registry.meta("test_sources")
        assert meta["representations"] == ["point_sources", "healpix_map"]
        assert meta["output_mode"] == "polymorphic"
        assert "point_sources" in meta["representations"]
        assert "healpix_map" in meta["representations"]

    def test_alias_metadata_reflects_aliased_representation(self):
        meta = loader_registry.meta("test_healpix")
        assert meta["representations"] == ["healpix_map"]
        assert meta["output_mode"] == "healpix_only"
        assert "point_sources" not in meta["representations"]
        assert "healpix_map" in meta["representations"]

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

    def test_discovery_catalog_info_resolves_gleam_subcatalog(self):
        info = get_catalog_info("gleam_egc")
        assert info["loader"] == "gleam"
        assert info["resolved_loader"] == "gleam"
        assert info["resolved_kwargs"] == {"catalog": "gleam_egc"}
        assert info["category"] == "catalog"

    def test_discovery_catalog_info_resolves_racs_band(self):
        info = get_catalog_info("racs_low")
        assert info["loader"] == "racs"
        assert info["resolved_loader"] == "racs"
        assert info["resolved_kwargs"] == {"band": "low"}
        assert info["network_services"] == ["casda"]

    def test_discovery_catalog_info_resolves_lotss_release(self):
        info = get_catalog_info("lotss_dr2")
        assert info["loader"] == "lotss"
        assert info["resolved_kwargs"] == {"release": "dr2"}

    def test_discovery_catalog_info_resolves_mals_release(self):
        info = get_catalog_info("mals_dr1")
        assert info["loader"] == "mals"
        assert info["resolved_kwargs"] == {"release": "dr1"}

    def test_discovery_catalog_info_unknown_key_raises_with_hint(self):
        with pytest.raises(ValueError, match="loader_registry"):
            get_catalog_info("__unknown_catalog_xyz__")

    def test_resolve_loader_request_merges_alias_defaults(self):
        kind, kwargs = loader_registry.resolve_request("gsm2016", {"nside": 128})
        assert kind == "diffuse_sky"
        assert kwargs == {"model": "gsm2016", "nside": 128}

        kind, kwargs = loader_registry.resolve_request("gsm2016", {"model": "haslam"})
        assert kind == "diffuse_sky"
        assert kwargs == {"model": "haslam"}

    def test_alias_inherits_canonical_use_flag(self):
        """Aliases share the canonical loader's metadata. There is no
        per-alias ``use_flag`` override — once an alias resolves, every
        subsequent registry lookup walks via the canonical name and
        returns the canonical's ``use_flag``.

        ``use_flag`` is metadata for catalog browsing (``discovery.py``);
        the activation contract is ``kind`` (or its alias). YAML config
        does not currently consume ``use_flag``; ``simulator.py`` and
        ``io/config.py`` only walk the registry by ``kind``.
        """
        canonical = loader_registry.definition("diffuse_sky")
        # The canonical use_flag for diffuse_sky is "use_gsm" (legacy).
        assert canonical.meta_dict()["use_flag"] == "use_gsm"

        # Resolving any alias hands back the canonical name.  The
        # registry has only one definition per canonical name, so all
        # aliases trivially inherit its use_flag.
        for alias in ("gsm", "gsm2008", "gsm2016", "lfsm", "haslam"):
            kind, _ = loader_registry.resolve_request(alias, {})
            assert kind == "diffuse_sky"
            assert loader_registry.definition(kind).meta_dict()["use_flag"] == "use_gsm"


class TestSourceSpecs:
    def test_simple_catalog_request(self):
        spec = parse_sky_source_config(
            {
                "kind": "vlssr",
                "options": {"flux_limit": 2.0, "max_rows": 1000},
            }
        )
        assert isinstance(spec, CustomRegisteredSourceConfig)
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

    def test_simple_catalog_request_uses_loader_signature_default(self):
        spec = parse_sky_source_config({"kind": "nvss", "options": {"max_rows": 10}})
        assert isinstance(spec, CustomRegisteredSourceConfig)

        kind, kwargs = spec.to_loader_request()

        assert kind == "nvss"
        assert kwargs["max_rows"] == 10
        assert "flux_limit" not in kwargs
        assert VIZIER_POINT_CATALOGS["nvss"].default_flux_limit == pytest.approx(0.0025)

    def test_simple_catalog_loader_requires_download_bound(self):
        with pytest.raises(ValueError, match="region=.*max_rows=.*allow_full_catalog"):
            _load_from_vizier_catalog(
                "nvss",
                precision=PrecisionConfig.standard(),
            )

    def test_diffuse_request_uses_explicit_frequencies(self):
        freqs = np.array([100e6, 101e6])
        spec = parse_sky_source_config({"kind": "diffuse_sky", "model": "gsm2008"})
        assert isinstance(spec, DiffuseSkySourceConfig)

        kind, kwargs = spec.to_loader_request(
            frequencies=freqs,
        )
        freqs[0] = 999e6
        assert kind == "diffuse_sky"
        assert kwargs["model"] == "gsm2008"
        np.testing.assert_array_equal(kwargs["frequencies"], [100e6, 101e6])

    def test_diffuse_alias_preserves_selected_model(self):
        spec = parse_sky_source_config({"kind": "diffuse_sky", "model": "gsm2016"})
        kind, kwargs = spec.to_loader_request()
        assert kind == "diffuse_sky"
        assert kwargs["model"] == "gsm2016"

    def test_diffuse_alias_kind_applies_alias_defaults(self):
        spec = parse_sky_source_config({"kind": "gsm2016", "options": {"nside": 64}})
        kind, kwargs = spec.to_loader_request()
        assert kind == "diffuse_sky"
        assert kwargs["model"] == "gsm2016"
        assert kwargs["nside"] == 64

    def test_diffuse_alias_kind_allows_explicit_override(self):
        spec = parse_sky_source_config(
            {
                "kind": "gsm2016",
                "options": {"model": "haslam", "nside": 128},
            }
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
        spec = parse_sky_source_config(
            {"kind": "test_healpix", "options": {"nside": 32}}
        )
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
            frequencies=np.array([100e6, 101e6]),
        )
        assert kind == "pyradiosky_file"
        assert kwargs["filename"] == "mock.skyh5"
        assert kwargs["flux_limit"] == pytest.approx(0.005)
        assert kwargs["reference_frequency_hz"] == 150e6
        np.testing.assert_array_equal(kwargs["frequencies"], [100e6, 101e6])

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
        assert "assume_disjoint" in SkyModelConfig.model_fields
        assert "allow_lossy_point_materialization" in VisibilityConfig.model_fields

    def test_sky_model_config_accepts_assume_disjoint(self):
        cfg = SkyModelConfig.model_validate(
            {
                "sources": [{"kind": "test_sources"}],
                "assume_disjoint": True,
            }
        )
        assert cfg.assume_disjoint is True


class TestConfigFieldsListShorthand:
    def test_list_shorthand_normalizes_to_identity_mapping(self):
        @loader_registry.register_loader(
            "_test_list_fields",
            config_fields=["filename", "nside"],
        )
        def _dummy(**kwargs):
            return kwargs

        try:
            definition = loader_registry.definition("_test_list_fields")
            assert definition.config_fields == {
                "filename": "filename",
                "nside": "nside",
            }
        finally:
            registry_core._REGISTRY.unregister("_test_list_fields")

    def test_builtin_fits_loader_uses_list_shorthand(self):
        definition = loader_registry.definition("fits_image")
        assert definition.config_fields == {
            "filename": "filename",
            "nside": "nside",
        }


class TestPathOptionMetadata:
    def test_builtin_file_loaders_declare_explicit_path_semantics(self):
        assert loader_registry.definition("bbs").path_options == {"filename": "file"}
        assert loader_registry.definition("pyradiosky_file").path_options == {
            "filename": "file"
        }
        assert loader_registry.definition("fits_image").path_options == {
            "filename": "file"
        }
        assert loader_registry.definition("skyh5_multifile").path_options == {
            "file_glob": "glob",
            "filenames": "file_list",
        }

    @pytest.mark.parametrize(
        "registration, message",
        [
            (
                {
                    "requires_file": True,
                    "config_fields": ["filename"],
                },
                "declares no path_options",
            ),
            (
                {
                    "config_fields": ["filename"],
                    "path_options": {"unknown": "file"},
                },
                "not declared loader arguments",
            ),
            (
                {
                    "config_fields": ["filename"],
                    "path_options": {"filename": "directory"},
                },
                "invalid path_options",
            ),
        ],
    )
    def test_invalid_path_metadata_fails_at_registration(self, registration, message):
        with pytest.raises(ValueError, match=message):

            @loader_registry.register_loader(
                "_test_invalid_path_metadata",
                **registration,
            )
            def _dummy(filename: str):
                return filename

    def test_path_metadata_is_serialized_without_changing_loader_execution(self):
        calls: list[str] = []

        @loader_registry.register_loader(
            "_test_scalar_path_metadata",
            requires_file=True,
            config_fields=["filename"],
            path_options={"filename": "file"},
        )
        def _dummy(filename: str):
            calls.append(filename)
            return filename

        try:
            definition = loader_registry.definition("_test_scalar_path_metadata")
            assert definition.meta_dict()["path_options"] == {"filename": "file"}
            assert loader_registry.loader("_test_scalar_path_metadata")("raw") == "raw"
            assert calls == ["raw"]
        finally:
            registry_core._REGISTRY.unregister("_test_scalar_path_metadata")


class TestExecutorRecommendation:
    def test_thread_for_pure_catalog_loads(self):
        from radiosim.core.sky.operations.parallel import (
            recommend_executor_for_loaders,
        )

        assert recommend_executor_for_loaders([("gleam", {}), ("nvss", {})]) == "thread"

    def test_process_when_any_diffuse_requested(self):
        from radiosim.core.sky.operations.parallel import (
            recommend_executor_for_loaders,
        )

        assert (
            recommend_executor_for_loaders([("gleam", {}), ("diffuse_sky", {})])
            == "process"
        )

    def test_process_for_pyradiosky_file(self):
        from radiosim.core.sky.operations.parallel import (
            recommend_executor_for_loaders,
        )

        assert (
            recommend_executor_for_loaders([("pyradiosky_file", {"filename": "x"})])
            == "process"
        )

    def test_unknown_loader_is_skipped_gracefully(self):
        from radiosim.core.sky.operations.parallel import (
            recommend_executor_for_loaders,
        )

        # Unknown loader should not crash the recommender; the surrounding
        # ``load_models_parallel`` call will raise the actual error later.
        assert (
            recommend_executor_for_loaders([("__no_such_loader__", {}), ("gleam", {})])
            == "thread"
        )
