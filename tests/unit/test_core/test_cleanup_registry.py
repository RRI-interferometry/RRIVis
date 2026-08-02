"""Phase-2 cleanup tests for the sky-loader registry (items D1-D4, D6-D8, B10).

These pin the new registry contract:

* D1 - ``meta_dict``/``meta`` expose a single canonical representation set.
* D2 - the facade calls ``core._REGISTRY`` directly (no ``core._get_*`` middle hop).
* D3 - registration rejects ``config_fields`` keys not on the loader signature.
* D4 - every registered catalog identity has a matching ``catalogs.py`` entry
  and vice versa.
* D6 - ``registry/__init__`` re-exports ``loader_registry`` / ``LoaderDefinition``.
* D7 - the core decorator is named ``register_loader`` (matches the facade).
* D8 - footprint filesystem IO lives outside the metadata module.
* B10 - one shared module-level flux-unit validator.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import radiosim.core.sky.registry as registry_pkg
import radiosim.core.sky.registry.core as registry_core
import radiosim.core.sky.registry.facade as registry_facade
from radiosim.core.sky.registry import (
    DIFFUSE_MODELS,
    RACS_CATALOGS,
    VIZIER_POINT_CATALOGS,
    loader_registry,
)
from tests.support.repo_scan import PYTHON_SUFFIXES, iter_tracked_files

# ---------------------------------------------------------------------------
# D6 - package re-exports
# ---------------------------------------------------------------------------


def test_registry_package_reexports_facade_symbols():
    from radiosim.core.sky.registry import (
        LoaderDefinition,
        SkyLoaderRegistry,
        loader_registry,
    )

    assert loader_registry is registry_facade.loader_registry
    assert LoaderDefinition is registry_facade.LoaderDefinition
    assert SkyLoaderRegistry is registry_facade.SkyLoaderRegistry
    assert "loader_registry" in registry_pkg.__all__
    assert "LoaderDefinition" in registry_pkg.__all__
    assert "SkyLoaderRegistry" in registry_pkg.__all__


def test_registry_package_is_canonical_import_surface_for_consumers():
    assert "registry.facade import loader_registry" not in inspect.getsource(
        __import__("radiosim.core.sky.loaders.diffuse", fromlist=["dummy"])
    )
    assert "registry.facade import loader_registry" not in inspect.getsource(
        __import__("radiosim.io.config", fromlist=["dummy"])
    )


def test_registry_package_reexports_catalog_symbols():
    import radiosim.core.sky.registry.catalogs as catalogs_mod
    from radiosim.core.sky.registry import (
        CASDA_TAP_URL,
        DIFFUSE_MODELS,
        RACS_CATALOGS,
        VIZIER_POINT_CATALOGS,
        DiffuseModelEntry,
        RacsCatalogEntry,
        VizierCatalogEntry,
        load_catalog_footprint_asset,
    )

    assert CASDA_TAP_URL is catalogs_mod.CASDA_TAP_URL
    assert DIFFUSE_MODELS is catalogs_mod.DIFFUSE_MODELS
    assert RACS_CATALOGS is catalogs_mod.RACS_CATALOGS
    assert VIZIER_POINT_CATALOGS is catalogs_mod.VIZIER_POINT_CATALOGS
    assert DiffuseModelEntry is catalogs_mod.DiffuseModelEntry
    assert RacsCatalogEntry is catalogs_mod.RacsCatalogEntry
    assert VizierCatalogEntry is catalogs_mod.VizierCatalogEntry
    assert load_catalog_footprint_asset is catalogs_mod.load_catalog_footprint_asset

    catalog_symbols = (
        "CASDA_TAP_URL",
        "DIFFUSE_MODELS",
        "RACS_CATALOGS",
        "VIZIER_POINT_CATALOGS",
        "DiffuseModelEntry",
        "RacsCatalogEntry",
        "VizierCatalogEntry",
        "load_catalog_footprint_asset",
    )
    for name in catalog_symbols:
        assert name in registry_pkg.__all__


def test_no_src_imports_registry_submodule():
    src_root = Path(__file__).resolve().parents[3] / "src" / "radiosim"
    import_pattern = re.compile(
        r"(?:from|import)\s+[\w.]+\.registry\.(?:facade|catalogs|core)\b"
    )
    violations: list[str] = []
    for path in iter_tracked_files(src_root, suffixes=PYTHON_SUFFIXES):
        rel = path.relative_to(src_root)
        if rel.parts[:3] == ("core", "sky", "registry"):
            continue
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if import_pattern.search(line):
                violations.append(f"{rel}:{line_no}: {stripped}")
    assert not violations, "Registry submodule imports in src:\n" + "\n".join(
        violations
    )


# ---------------------------------------------------------------------------
# D7 - decorator naming normalized to register_loader
# ---------------------------------------------------------------------------


def test_core_decorator_named_register_loader():
    assert hasattr(registry_core, "register_loader")
    # The old private name is gone (no shim).
    assert not hasattr(registry_core, "_register_loader")


# ---------------------------------------------------------------------------
# D1 - single canonical representation set
# ---------------------------------------------------------------------------


def test_meta_dict_has_single_representation_key():
    """meta_dict carries exactly one representation set, no denormalized copies."""
    definition = loader_registry.definition("test_sources")
    meta = definition.meta_dict()

    assert meta["representations"] == ["point_sources", "healpix_map"]
    assert meta["output_mode"] == "polymorphic"
    # The denormalized copies have been collapsed away.
    for dropped in (
        "representation",
        "primary_representation",
        "supports_point_sources",
        "supports_healpix_map",
        "capabilities",
    ):
        assert dropped not in meta


def test_representation_views_still_available_as_definition_properties():
    """Booleans/primary are computed from the canonical set, not stored twice."""
    definition = loader_registry.definition("test_sources")
    assert definition.supports_point_sources is True
    assert definition.supports_healpix_map is True
    assert definition.primary_representation == "point_sources"
    assert definition.output_mode == "polymorphic"

    diffuse = loader_registry.definition("diffuse_sky")
    assert diffuse.representations == ("healpix_map",)
    assert diffuse.supports_point_sources is False
    assert diffuse.supports_healpix_map is True
    assert diffuse.primary_representation == "healpix_map"
    assert diffuse.output_mode == "healpix_only"


def test_meta_for_alias_narrows_canonical_representation():
    """An alias that pins a representation narrows the single canonical set."""
    meta = loader_registry.meta("test_healpix")
    assert meta["representations"] == ["healpix_map"]
    assert meta["output_mode"] == "healpix_only"
    # Still no denormalized copies reintroduced on the alias path.
    assert "representation" not in meta
    assert "capabilities" not in meta


def test_meta_keys_are_a_stable_canonical_set():
    """Every loader's meta dict exposes the same canonical key set."""
    expected_keys = {
        "config_section",
        "use_flag",
        "representations",
        "output_mode",
        "category",
        "requires_file",
        "network_service",
        "aliases",
        "alias_defaults",
        "config_fields",
        "path_options",
    }
    for name in loader_registry.names():
        assert set(loader_registry.meta(name)) == expected_keys


# ---------------------------------------------------------------------------
# D2 - facade collapses to core._REGISTRY (no middle-hop wrappers)
# ---------------------------------------------------------------------------


def test_facade_methods_have_at_most_one_indirection():
    """Facade methods use _REGISTRY directly, not core._get_* wrappers."""
    source = inspect.getsource(registry_facade)
    assert "_REGISTRY" in source
    assert "_backend._" not in source
    for stale in (
        "_get_canonical_loader",
        "_get_resolved_loader",
        "_list_loaders",
        "_alias_map",
        "_alias_defaults_map",
        "_unregister_loader",
    ):
        assert stale not in source


def test_unused_facade_methods_removed():
    for name in ("alias_defaults", "ensure_default_loaders_registered", "unregister"):
        assert not hasattr(loader_registry, name)


def test_facade_alias_accessors_delegate():
    """Alias map and per-definition defaults remain correct after collapse."""
    aliases = loader_registry.aliases()
    assert aliases["gsm2016"] == "diffuse_sky"
    diffuse = loader_registry.definition("diffuse_sky")
    assert diffuse.alias_defaults["gsm2016"] == {"model": "gsm2016"}


# ---------------------------------------------------------------------------
# D3 - config_fields keys must be a subset of the loader signature
# ---------------------------------------------------------------------------


def test_registration_rejects_unknown_config_field():
    def _loader(flux_limit: float = 1.0):
        return "ok"

    with pytest.raises(ValueError, match="config_fields"):
        loader_registry.register_loader(
            "_d3_bad_loader",
            config_fields={"flux_limit": "flux_limit", "not_a_param": "x"},
        )(_loader)


def test_registration_accepts_matching_config_fields():
    def _loader(flux_limit: float = 1.0, *, max_rows: int | None = None):
        return "ok"

    decorated = loader_registry.register_loader(
        "_d3_good_loader",
        config_fields={"flux_limit": "flux_limit", "max_rows": "max_rows"},
    )(_loader)
    try:
        assert decorated is _loader
        assert loader_registry.definition("_d3_good_loader").config_fields == {
            "flux_limit": "flux_limit",
            "max_rows": "max_rows",
        }
    finally:
        registry_core._REGISTRY.unregister("_d3_good_loader")


def test_registration_accepts_identity_config_field_shorthand():
    def _loader(flux_limit: float = 1.0, *, max_rows: int | None = None):
        return "ok"

    decorated = loader_registry.register_loader(
        "_d3_shorthand_loader",
        config_fields=("flux_limit", "max_rows"),
    )(_loader)
    try:
        assert decorated is _loader
        assert loader_registry.definition("_d3_shorthand_loader").config_fields == {
            "flux_limit": "flux_limit",
            "max_rows": "max_rows",
        }
        assert loader_registry.meta("_d3_shorthand_loader")["config_fields"] == {
            "flux_limit": "flux_limit",
            "max_rows": "max_rows",
        }
    finally:
        registry_core._REGISTRY.unregister("_d3_shorthand_loader")


def test_registration_allows_var_keyword_loader():
    def _loader(**kwargs):
        return "ok"

    decorated = loader_registry.register_loader(
        "_d3_kwargs_loader",
        config_fields={"anything": "x", "goes": "y"},
    )(_loader)
    try:
        assert decorated is _loader
    finally:
        registry_core._REGISTRY.unregister("_d3_kwargs_loader")


def test_all_builtin_loaders_pass_config_field_check():
    """Every shipped loader already satisfies the D3 invariant."""
    for definition in loader_registry.definitions():
        if not definition.config_fields:
            continue
        try:
            signature = inspect.signature(definition.loader)
        except (TypeError, ValueError):
            continue
        params = signature.parameters.values()
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
            continue
        accepted = {
            p.name
            for p in params
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        unknown = set(definition.config_fields) - accepted
        assert not unknown, f"{definition.name}: {unknown} not in signature"


# ---------------------------------------------------------------------------
# D4 - registry/catalog identity cross-check
# ---------------------------------------------------------------------------


def _registry_catalog_identifiers() -> set[str]:
    """Catalog/model keys reachable through the registry.

    Combines bare loader names that are themselves catalog keys, the
    alias-bound ``catalog`` / ``model`` defaults, and the
    ``<loader>_<release>`` keys reconstructed from release-style aliases.
    """
    identifiers: set[str] = set()
    for definition in loader_registry.definitions():
        if definition.category not in ("catalog", "diffuse"):
            continue
        # Bare loader names (vlssr, tgss, 3c, vlass, ...) are catalog keys.
        identifiers.add(definition.name)
        for alias, defaults in definition.alias_defaults.items():
            identifiers.add(alias)
            for key in ("catalog", "model"):
                if key in defaults:
                    identifiers.add(str(defaults[key]))
            if "release" in defaults:
                identifiers.add(f"{definition.name}_{defaults['release']}")
            if "band" in defaults:
                identifiers.add(str(defaults["band"]))
    return identifiers


def test_every_catalog_entry_is_referenced_by_registry():
    """No catalogs.py entry is orphaned from the loader registry."""
    referenced = _registry_catalog_identifiers()

    for key in VIZIER_POINT_CATALOGS:
        assert key in referenced, f"VizieR catalog '{key}' has no registry reference"
    for key in DIFFUSE_MODELS:
        assert key in referenced, f"diffuse model '{key}' has no registry reference"
    for key in RACS_CATALOGS:
        assert key in referenced, f"RACS catalog '{key}' has no registry reference"


def test_registry_catalog_aliases_resolve_to_real_entries():
    """Every alias-bound catalog/model/band reference points at a real entry."""
    all_keys = set(VIZIER_POINT_CATALOGS) | set(RACS_CATALOGS) | set(DIFFUSE_MODELS)
    for definition in loader_registry.definitions():
        if definition.category not in ("catalog", "diffuse"):
            continue
        for defaults in definition.alias_defaults.values():
            if "catalog" in defaults:
                assert defaults["catalog"] in VIZIER_POINT_CATALOGS
            if "model" in defaults:
                assert defaults["model"] in DIFFUSE_MODELS
            if "band" in defaults:
                assert defaults["band"] in RACS_CATALOGS
            if "release" in defaults:
                assert f"{definition.name}_{defaults['release']}" in all_keys


def test_catalog_entries_carry_registry_identity_metadata():
    gleam = VIZIER_POINT_CATALOGS["gleam_egc"]
    assert gleam.loader_name == "gleam"
    assert gleam.config_section == "gleam"
    assert gleam.use_flag == "use_gleam"
    assert gleam.alias_defaults == {"catalog": "gleam_egc"}
    assert gleam.network_service == "vizier"
    assert gleam.config_fields == {
        "flux_limit": "flux_limit",
        "catalog": "catalog",
        "max_rows": "max_rows",
        "allow_full_catalog": "allow_full_catalog",
    }

    lfsm = DIFFUSE_MODELS["lfsm"]
    assert lfsm.loader_name == "diffuse_sky"
    assert lfsm.alias_defaults == {"model": "lfsm"}
    assert lfsm.network_service == "pygdsm_data"

    racs_low = RACS_CATALOGS["low"]
    assert racs_low.loader_name == "racs"
    assert racs_low.config_section == "racs"
    assert racs_low.use_flag == "use_racs"
    assert racs_low.alias == "racs_low"
    assert racs_low.alias_defaults == {"band": "low"}
    assert racs_low.network_service == "casda"

    three_c = VIZIER_POINT_CATALOGS["3c"]
    assert three_c.config_section == "three_c"
    assert three_c.use_flag == "use_3c"


def test_vizier_catalog_entries_lead_with_vizier_id():
    """Every VizierCatalogEntry literal leads with vizier_id after any ** unpack."""
    import radiosim.core.sky.registry.catalogs as catalogs_mod

    tree = ast.parse(inspect.getsource(catalogs_mod))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", "") != "VizierCatalogEntry":
            continue
        named = [kw for kw in node.keywords if kw.arg is not None]
        assert named, f"VizierCatalogEntry near line {node.lineno} has no named kwargs"
        assert named[0].arg == "vizier_id", (
            f"VizierCatalogEntry near line {node.lineno} must lead with "
            f"vizier_id=, got {named[0].arg!r}"
        )


@pytest.mark.parametrize(
    "loader_name",
    ["gleam", "mals", "lotss"],
)
def test_vizier_family_aliases_are_single_sourced(loader_name: str):
    """Family loader aliases must match catalog entries, not hard-coded decorator lists."""
    from radiosim.core.sky.registry.catalogs import build_vizier_family_aliases

    definition = loader_registry.definition(loader_name)
    expected = build_vizier_family_aliases(loader_name)
    assert set(definition.aliases) == set(expected)
    assert definition.alias_defaults == expected


def test_racs_aliases_are_single_sourced():
    from radiosim.core.sky.registry.catalogs import build_racs_aliases

    definition = loader_registry.definition("racs")
    expected = build_racs_aliases()
    assert set(definition.aliases) == set(expected)
    assert definition.alias_defaults == expected


def test_diffuse_sky_aliases_are_single_sourced():
    from radiosim.core.sky.registry.catalogs import build_diffuse_sky_aliases

    definition = loader_registry.definition("diffuse_sky")
    expected = build_diffuse_sky_aliases()
    assert set(definition.aliases) == set(expected)
    assert definition.alias_defaults == expected


@pytest.mark.parametrize(
    ("loader_name", "catalog_keys"),
    [
        ("gleam", [k for k in VIZIER_POINT_CATALOGS if k.startswith("gleam")]),
        ("mals", [k for k in VIZIER_POINT_CATALOGS if k.startswith("mals_")]),
        ("lotss", [k for k in VIZIER_POINT_CATALOGS if k.startswith("lotss_")]),
        ("racs", list(RACS_CATALOGS)),
        ("diffuse_sky", list(DIFFUSE_MODELS)),
    ],
)
def test_family_loader_registry_metadata_matches_catalog_entries(
    loader_name: str, catalog_keys: list[str]
):
    from radiosim.core.sky.registry.catalogs import (
        diffuse_sky_loader_registration,
        racs_loader_registration,
        vizier_family_loader_registration,
    )

    definition = loader_registry.definition(loader_name)
    if loader_name in {"gleam", "mals", "lotss"}:
        expected = vizier_family_loader_registration(loader_name)
    elif loader_name == "racs":
        expected = racs_loader_registration()
    elif loader_name == "diffuse_sky":
        expected = diffuse_sky_loader_registration()
    else:
        pytest.fail(f"unexpected loader {loader_name!r}")

    for field in ("config_section", "use_flag", "network_service", "category"):
        assert getattr(definition, field) == expected[field], field
    assert set(definition.aliases) == set(expected["aliases"])
    assert definition.alias_defaults == expected["aliases"]

    if loader_name in {"gleam", "mals", "lotss", "racs", "diffuse_sky"}:
        representative = (
            VIZIER_POINT_CATALOGS[catalog_keys[0]]
            if loader_name in {"gleam", "mals", "lotss"}
            else (
                RACS_CATALOGS[catalog_keys[0]]
                if loader_name == "racs"
                else DIFFUSE_MODELS[catalog_keys[0]]
            )
        )
        assert definition.network_service == representative.network_service
        assert definition.category == representative.category


# ---------------------------------------------------------------------------
# D8 - footprint IO relocated out of the metadata module
# ---------------------------------------------------------------------------


def test_footprint_io_lives_in_dedicated_module():
    from radiosim.core.sky.registry import footprint_assets

    assert hasattr(footprint_assets, "load_catalog_footprint_asset")
    # catalogs.py has no filesystem IO of its own anymore.
    catalog_source = inspect.getsource(
        __import__("radiosim.core.sky.registry.catalogs", fromlist=["dummy"])
    )
    assert "resources.files" not in catalog_source
    assert "np.load" not in catalog_source


def test_footprint_asset_loader_is_importable_from_both_paths():
    from radiosim.core.sky.registry.catalogs import (
        load_catalog_footprint_asset as via_catalogs,
    )
    from radiosim.core.sky.registry.footprint_assets import (
        load_catalog_footprint_asset as via_module,
    )

    assert via_catalogs is via_module


# ---------------------------------------------------------------------------
# B10 - one shared flux-unit validator
# ---------------------------------------------------------------------------


def test_shared_flux_unit_validator_used_by_all_entries():
    from radiosim.core.sky.registry import catalogs as catalogs_mod

    assert hasattr(catalogs_mod, "_check_flux_unit")
    # The helper raises with the supplied entry-type label.
    with pytest.raises(ValueError, match="VizierCatalogEntry"):
        catalogs_mod._check_flux_unit("kJy", "VizierCatalogEntry")
    with pytest.raises(ValueError, match="RacsCatalogEntry"):
        catalogs_mod._check_flux_unit("kJy", "RacsCatalogEntry")
    # A known unit passes silently.
    assert catalogs_mod._check_flux_unit("Jy", "VizierCatalogEntry") is None


def test_catalog_entries_still_reject_bad_flux_unit():
    from pydantic import ValidationError

    from radiosim.core.sky.registry.catalogs import (
        RacsCatalogEntry,
        VizierCatalogEntry,
    )

    with pytest.raises(ValidationError, match="flux_unit"):
        VizierCatalogEntry(
            vizier_id="X/1",
            description="bad",
            ra_col="RA",
            dec_col="DEC",
            flux_col="F",
            flux_unit="kJy",
            freq_mhz=100.0,
        )
    with pytest.raises(ValidationError, match="flux_unit"):
        RacsCatalogEntry(
            description="bad",
            freq_mhz=100.0,
            tap_table="casda.x",
            ra_col="ra",
            dec_col="dec",
            flux_col="flux",
            flux_unit="kJy",
        )
