"""Tests for rrivis.core.sky._registry — loader registration system."""

import pytest

from rrivis.core.sky._registry import (
    _LOADER_META,
    _LOADERS,
    build_alias_map,
    build_loader_kwargs,
    build_network_services_map,
    build_sky_model_map,
    get_loader,
    get_loader_meta,
    list_loaders,
    register_loader,
)

# ---------------------------------------------------------------------------
# Registration and retrieval
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_get_loader(self):
        """Register a dummy function, retrieve it by name."""

        @register_loader("_test_dummy_loader")
        def _dummy_loader(**kwargs):
            return "dummy"

        try:
            retrieved = get_loader("_test_dummy_loader")
            assert retrieved is _dummy_loader
            assert retrieved() == "dummy"
        finally:
            # Clean up so we don't pollute the global registry
            _LOADERS.pop("_test_dummy_loader", None)
            _LOADER_META.pop("_test_dummy_loader", None)

    def test_get_unknown_loader_raises(self):
        """Requesting an unknown loader name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sky model loader"):
            get_loader("__nonexistent_loader_xyz__")

    def test_get_loader_meta_by_name(self):
        """get_loader_meta() returns metadata for a canonical loader name."""
        import rrivis.core.sky  # noqa: F401

        meta = get_loader_meta("gleam")
        assert meta["config_section"] == "gleam"
        assert "use_flag" in meta

    def test_get_loader_meta_resolves_alias(self):
        """get_loader_meta() resolves aliases (e.g. 'gsm' -> 'diffuse_sky')."""
        import rrivis.core.sky  # noqa: F401

        meta_alias = get_loader_meta("gsm")
        meta_canonical = get_loader_meta("diffuse_sky")
        assert meta_alias is meta_canonical

    def test_get_loader_meta_unknown_raises(self):
        """get_loader_meta() raises ValueError for unknown names."""
        with pytest.raises(ValueError, match="No metadata for loader"):
            get_loader_meta("__nonexistent_xyz__")

    def test_list_loaders_sorted(self):
        """list_loaders() returns a sorted list."""
        names = list_loaders()
        assert names == sorted(names)
        assert isinstance(names, list)
        assert len(names) > 0

    def test_all_expected_loaders_registered(self):
        """After importing sky, all expected loaders exist in the registry."""
        # Force import of the sky package which triggers all @register_loader
        import rrivis.core.sky  # noqa: F401

        names = list_loaders()
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
        }
        missing = expected - set(names)
        assert not missing, f"Expected loaders not registered: {missing}"

    def test_all_loaders_have_metadata(self):
        """Every registered loader must have metadata."""
        import rrivis.core.sky  # noqa: F401

        for name in list_loaders():
            assert name in _LOADER_META, f"Loader '{name}' has no metadata"
            meta = _LOADER_META[name]
            assert "config_section" in meta
            assert "use_flag" in meta
            assert "is_healpix" in meta
            assert "requires_file" in meta

    def test_registry_config_coverage(self):
        """Every registered loader's config_section must exist in SkyModelConfig."""
        import rrivis.core.sky  # noqa: F401
        from rrivis.io.config import SkyModelConfig

        config_fields = SkyModelConfig.model_fields
        for name, meta in _LOADER_META.items():
            section = meta["config_section"]
            assert section in config_fields, (
                f"Loader '{name}' references config_section '{section}' "
                f"but SkyModelConfig has no such field. Add it to config.py."
            )
            # Verify the use_flag exists on the config section's model
            section_cls = config_fields[section].annotation
            # Pydantic v2: get the actual class for default_factory fields
            if hasattr(section_cls, "model_fields"):
                assert meta["use_flag"] in section_cls.model_fields, (
                    f"Loader '{name}': config section '{section}' missing "
                    f"field '{meta['use_flag']}'."
                )

    def test_build_sky_model_map_covers_non_file_loaders(self):
        """build_sky_model_map() includes all non-file loaders."""
        import rrivis.core.sky  # noqa: F401

        sky_map = build_sky_model_map()
        # All non-file loaders should be in the map
        for name, meta in _LOADER_META.items():
            if meta.get("requires_file", False):
                assert name not in sky_map, (
                    f"File-based loader '{name}' should not be in sky_model_map"
                )
            else:
                assert name in sky_map, (
                    f"Non-file loader '{name}' missing from sky_model_map"
                )
                section, flag, is_hp = sky_map[name]
                assert section == meta["config_section"]
                assert flag == meta["use_flag"]
                assert is_hp == meta["is_healpix"]

    def test_build_sky_model_map_excludes_file_loaders(self):
        """build_sky_model_map() excludes bbs, fits_image, pyradiosky_file."""
        import rrivis.core.sky  # noqa: F401

        sky_map = build_sky_model_map()
        for file_loader in ("bbs", "fits_image", "pyradiosky_file"):
            assert file_loader not in sky_map


# ---------------------------------------------------------------------------
# Extended metadata (network_service, aliases, config_fields)
# ---------------------------------------------------------------------------


class TestExtendedMetadata:
    def test_all_loaders_have_extended_fields(self):
        """Every loader's metadata includes the extended fields."""
        import rrivis.core.sky  # noqa: F401

        for name, meta in _LOADER_META.items():
            assert "network_service" in meta, f"Loader '{name}' missing network_service"
            assert "aliases" in meta, f"Loader '{name}' missing aliases"
            assert "config_fields" in meta, f"Loader '{name}' missing config_fields"

    def test_network_loaders_have_service(self):
        """Non-file loaders that need network have a network_service set."""
        import rrivis.core.sky  # noqa: F401

        # All VizieR loaders should have "vizier"
        vizier_loaders = {
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
        }
        for name in vizier_loaders:
            assert _LOADER_META[name]["network_service"] == "vizier"

        assert _LOADER_META["racs"]["network_service"] == "casda"
        assert _LOADER_META["diffuse_sky"]["network_service"] == "pygdsm_data"
        assert _LOADER_META["pysm3"]["network_service"] == "pysm3_data"

    def test_file_loaders_have_no_service(self):
        """File-based loaders should have network_service=None."""
        import rrivis.core.sky  # noqa: F401

        for name in ("bbs", "fits_image", "pyradiosky_file"):
            assert _LOADER_META[name]["network_service"] is None


class TestBuildNetworkServicesMap:
    def test_returns_expected_entries(self):
        """build_network_services_map() covers all expected service mappings."""
        import rrivis.core.sky  # noqa: F401

        svc_map = build_network_services_map()
        # Should have entries for vizier, casda, pygdsm_data, pysm3_data
        services_present = set(svc_map.values())
        assert "vizier" in services_present
        assert "casda" in services_present
        assert "pygdsm_data" in services_present
        assert "pysm3_data" in services_present

    def test_excludes_file_loaders(self):
        """File-based loaders should not appear in the network services map."""
        import rrivis.core.sky  # noqa: F401

        svc_map = build_network_services_map()
        sections = {k[0] for k in svc_map}
        assert "bbs" not in sections
        assert "fits_image" not in sections
        assert "pyradiosky" not in sections

    def test_matches_old_hardcoded_map(self):
        """Registry-derived map must cover all entries from the old hardcoded map."""
        import rrivis.core.sky  # noqa: F401

        old_map = {
            ("gleam", "use_gleam"): "vizier",
            ("mals", "use_mals"): "vizier",
            ("vlssr", "use_vlssr"): "vizier",
            ("tgss", "use_tgss"): "vizier",
            ("wenss", "use_wenss"): "vizier",
            ("sumss", "use_sumss"): "vizier",
            ("nvss", "use_nvss"): "vizier",
            ("lotss", "use_lotss"): "vizier",
            ("three_c", "use_3c"): "vizier",
            ("vlass", "use_vlass"): "vizier",
            ("racs", "use_racs"): "casda",
            ("gsm_healpix", "use_gsm"): "pygdsm_data",
            ("pysm3", "use_pysm3"): "pysm3_data",
        }
        new_map = build_network_services_map()
        for key, service in old_map.items():
            assert key in new_map, f"Missing entry: {key}"
            assert new_map[key] == service, (
                f"Service mismatch for {key}: expected {service}, got {new_map[key]}"
            )


class TestBuildAliasMap:
    def test_diffuse_sky_aliases(self):
        """diffuse_sky loader should have GSM variant aliases."""
        import rrivis.core.sky  # noqa: F401

        alias_map = build_alias_map()
        for alias in ("gsm", "gsm2008", "gsm2016", "lfsm", "haslam"):
            assert alias in alias_map
            assert alias_map[alias] == "diffuse_sky"

    def test_no_self_aliases(self):
        """Canonical loader names should not appear as aliases."""
        import rrivis.core.sky  # noqa: F401

        alias_map = build_alias_map()
        for name in list_loaders():
            assert name not in alias_map, (
                f"Canonical name '{name}' should not be an alias"
            )


class TestBuildLoaderKwargs:
    def test_simple_vizier_loader(self):
        """Simple VizieR loader gets flux_limit + region."""
        import rrivis.core.sky  # noqa: F401

        config = {"flux_limit": 2.0}
        kwargs = build_loader_kwargs(
            "vlssr", config, flux_multiplier=1e-3, region="mock_region"
        )
        assert kwargs["flux_limit"] == pytest.approx(0.002)
        assert kwargs["region"] == "mock_region"

    def test_gleam_config_field_rename(self):
        """GLEAM loader renames gleam_catalogue -> catalog."""
        import rrivis.core.sky  # noqa: F401

        config = {"flux_limit": 1.0, "gleam_catalogue": "gleam_x_dr2"}
        kwargs = build_loader_kwargs("gleam", config, flux_multiplier=1.0, region=None)
        assert kwargs["catalog"] == "gleam_x_dr2"
        assert kwargs["flux_limit"] == 1.0
        assert "region" not in kwargs  # region=None is not included

    def test_healpix_loader_gets_obs_freq_config(self):
        """HEALPix loaders receive obs_frequency_config."""
        import rrivis.core.sky  # noqa: F401

        config = {"gsm_catalogue": "gsm2008", "nside": 32}
        obs_freq = {"starting_frequency": 100.0}
        kwargs = build_loader_kwargs(
            "diffuse_sky",
            config,
            flux_multiplier=1.0,
            region=None,
            obs_freq_config=obs_freq,
        )
        assert kwargs["model"] == "gsm2008"
        assert kwargs["nside"] == 32
        assert kwargs["obs_frequency_config"] == obs_freq

    def test_non_healpix_skips_obs_freq_config(self):
        """Non-HEALPix loaders should not receive obs_frequency_config."""
        import rrivis.core.sky  # noqa: F401

        config = {"flux_limit": 1.0}
        kwargs = build_loader_kwargs(
            "nvss",
            config,
            flux_multiplier=1.0,
            region=None,
            obs_freq_config={"starting_frequency": 100.0},
        )
        assert "obs_frequency_config" not in kwargs

    def test_missing_config_field_skipped(self):
        """Config fields not present in the config section are skipped."""
        import rrivis.core.sky  # noqa: F401

        config = {}  # No flux_limit or gleam_catalogue
        kwargs = build_loader_kwargs("gleam", config, flux_multiplier=1.0, region=None)
        assert "flux_limit" not in kwargs
        assert "catalog" not in kwargs

    def test_brightness_conversion_passed_when_set(self):
        """brightness_conversion is included when explicitly provided."""
        import rrivis.core.sky  # noqa: F401

        config = {"flux_limit": 1.0}
        kwargs = build_loader_kwargs(
            "vlssr",
            config,
            flux_multiplier=1.0,
            region=None,
            brightness_conversion="rayleigh-jeans",
        )
        assert kwargs["brightness_conversion"] == "rayleigh-jeans"

    def test_brightness_conversion_omitted_when_none(self):
        """brightness_conversion is omitted when None (loader uses default)."""
        import rrivis.core.sky  # noqa: F401

        config = {"flux_limit": 1.0}
        kwargs = build_loader_kwargs(
            "vlssr",
            config,
            flux_multiplier=1.0,
            region=None,
            brightness_conversion=None,
        )
        assert "brightness_conversion" not in kwargs


class TestLoaderSignatureConsistency:
    """Verify all registered loaders accept the common parameters."""

    def test_all_loaders_accept_brightness_conversion(self):
        """Every registered loader should accept brightness_conversion."""
        import inspect

        import rrivis.core.sky  # noqa: F401

        for name in list_loaders():
            loader = get_loader(name)
            sig = inspect.signature(loader)
            assert "brightness_conversion" in sig.parameters, (
                f"Loader '{name}' missing brightness_conversion parameter"
            )

    def test_all_loaders_accept_precision(self):
        """Every registered loader should accept precision."""
        import inspect

        import rrivis.core.sky  # noqa: F401

        for name in list_loaders():
            loader = get_loader(name)
            sig = inspect.signature(loader)
            assert "precision" in sig.parameters, (
                f"Loader '{name}' missing precision parameter"
            )

    def test_all_loaders_accept_region(self):
        """Every registered loader should accept region."""
        import inspect

        import rrivis.core.sky  # noqa: F401

        for name in list_loaders():
            loader = get_loader(name)
            sig = inspect.signature(loader)
            assert "region" in sig.parameters, (
                f"Loader '{name}' missing region parameter"
            )
