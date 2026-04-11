"""Tests for catalog metadata Pydantic models and entries."""

import pytest
from pydantic import ValidationError

from rrivis.core.sky.catalogs import (
    CASDA_TAP_URL,
    DIFFUSE_MODELS,
    RACS_CATALOGS,
    VIZIER_POINT_CATALOGS,
    DiffuseModelEntry,
    RacsCatalogEntry,
    VizierCatalogEntry,
)

# =========================================================================
# Schema enforcement: missing required fields raise ValidationError
# =========================================================================


class TestSchemaEnforcement:
    """Pydantic models reject entries with missing required fields."""

    def test_vizier_missing_vizier_id(self):
        with pytest.raises(ValidationError):
            VizierCatalogEntry(
                description="test",
                ra_col="RA",
                dec_col="DEC",
                flux_col="Flux",
                freq_mhz=100.0,
                # vizier_id is missing
            )

    def test_vizier_missing_ra_col(self):
        with pytest.raises(ValidationError):
            VizierCatalogEntry(
                vizier_id="VIII/97",
                description="test",
                dec_col="DEC",
                flux_col="Flux",
                freq_mhz=100.0,
                # ra_col is missing
            )

    def test_vizier_missing_freq_mhz(self):
        with pytest.raises(ValidationError):
            VizierCatalogEntry(
                vizier_id="VIII/97",
                description="test",
                ra_col="RA",
                dec_col="DEC",
                flux_col="Flux",
                # freq_mhz is missing
            )

    def test_racs_missing_tap_table(self):
        with pytest.raises(ValidationError):
            RacsCatalogEntry(
                description="test",
                freq_mhz=887.5,
                ra_col="ra",
                dec_col="dec",
                flux_col="flux",
                # tap_table is missing
            )

    def test_diffuse_missing_class_path(self):
        with pytest.raises(ValidationError):
            DiffuseModelEntry(
                description="test",
                freq_range=(10e6, 94e9),
                # class_path is missing
            )

    def test_diffuse_missing_freq_range(self):
        with pytest.raises(ValidationError):
            DiffuseModelEntry(
                description="test",
                class_path="pygdsm.GlobalSkyModel",
                # freq_range is missing
            )


# =========================================================================
# Immutability: frozen models reject attribute assignment
# =========================================================================


class TestImmutability:
    def test_vizier_frozen(self):
        entry = VIZIER_POINT_CATALOGS["vlssr"]
        with pytest.raises(ValidationError):
            entry.ra_col = "NEW_RA"

    def test_racs_frozen(self):
        entry = RACS_CATALOGS["low"]
        with pytest.raises(ValidationError):
            entry.freq_mhz = 999.0

    def test_diffuse_frozen(self):
        entry = DIFFUSE_MODELS["gsm2008"]
        with pytest.raises(ValidationError):
            entry.class_path = "different.Class"


# =========================================================================
# Catalog entry counts
# =========================================================================


class TestCatalogCounts:
    def test_vizier_count(self):
        assert len(VIZIER_POINT_CATALOGS) == 15

    def test_racs_count(self):
        assert len(RACS_CATALOGS) == 3

    def test_diffuse_count(self):
        assert len(DIFFUSE_MODELS) == 4


# =========================================================================
# Type correctness: all entries are proper model instances
# =========================================================================


class TestTypeCorrectness:
    @pytest.mark.parametrize("name", list(VIZIER_POINT_CATALOGS.keys()))
    def test_vizier_entry_type(self, name):
        assert isinstance(VIZIER_POINT_CATALOGS[name], VizierCatalogEntry)

    @pytest.mark.parametrize("name", list(RACS_CATALOGS.keys()))
    def test_racs_entry_type(self, name):
        assert isinstance(RACS_CATALOGS[name], RacsCatalogEntry)

    @pytest.mark.parametrize("name", list(DIFFUSE_MODELS.keys()))
    def test_diffuse_entry_type(self, name):
        assert isinstance(DIFFUSE_MODELS[name], DiffuseModelEntry)


# =========================================================================
# Field constraints
# =========================================================================


class TestFieldConstraints:
    @pytest.mark.parametrize("name", list(VIZIER_POINT_CATALOGS.keys()))
    def test_vizier_coord_frame_valid(self, name):
        entry = VIZIER_POINT_CATALOGS[name]
        assert entry.coord_frame in ("icrs", "fk4")

    @pytest.mark.parametrize("name", list(VIZIER_POINT_CATALOGS.keys()))
    def test_vizier_flux_unit_valid(self, name):
        entry = VIZIER_POINT_CATALOGS[name]
        assert entry.flux_unit in ("Jy", "mJy", "Jy/beam")

    @pytest.mark.parametrize("name", list(VIZIER_POINT_CATALOGS.keys()))
    def test_vizier_description_nonempty(self, name):
        assert len(VIZIER_POINT_CATALOGS[name].description) > 0

    @pytest.mark.parametrize("name", list(VIZIER_POINT_CATALOGS.keys()))
    def test_vizier_reference_url_nonempty(self, name):
        assert len(VIZIER_POINT_CATALOGS[name].reference_url) > 0

    @pytest.mark.parametrize("name", list(RACS_CATALOGS.keys()))
    def test_racs_tap_table_prefix(self, name):
        assert RACS_CATALOGS[name].tap_table.startswith("casda.")

    @pytest.mark.parametrize("name", list(RACS_CATALOGS.keys()))
    def test_racs_reference_url_nonempty(self, name):
        assert len(RACS_CATALOGS[name].reference_url) > 0

    @pytest.mark.parametrize("name", list(DIFFUSE_MODELS.keys()))
    def test_diffuse_class_path_dotted(self, name):
        assert "." in DIFFUSE_MODELS[name].class_path

    @pytest.mark.parametrize("name", list(DIFFUSE_MODELS.keys()))
    def test_diffuse_reference_url_nonempty(self, name):
        assert len(DIFFUSE_MODELS[name].reference_url) > 0

    @pytest.mark.parametrize("name", list(DIFFUSE_MODELS.keys()))
    def test_diffuse_freq_range_ordered(self, name):
        entry = DIFFUSE_MODELS[name]
        assert entry.freq_range[0] < entry.freq_range[1]


# =========================================================================
# Defaults
# =========================================================================


class TestDefaults:
    def test_vizier_defaults(self):
        entry = VizierCatalogEntry(
            vizier_id="TEST/1",
            description="test",
            ra_col="RA",
            dec_col="DEC",
            flux_col="Flux",
            freq_mhz=100.0,
        )
        assert entry.table is None
        assert entry.flux_unit == "Jy"
        assert entry.spindex_col is None
        assert entry.default_spindex == -0.7
        assert entry.coords_sexagesimal is False
        assert entry.coord_frame == "icrs"
        assert entry.major_col is None
        assert entry.minor_col is None
        assert entry.pa_col is None
        assert entry.reference_url == ""

    def test_racs_defaults(self):
        entry = RacsCatalogEntry(
            description="test",
            freq_mhz=100.0,
            tap_table="casda.test",
            ra_col="ra",
            dec_col="dec",
            flux_col="flux",
        )
        assert entry.flux_unit == "mJy"
        assert entry.reference_url == ""

    def test_diffuse_defaults(self):
        entry = DiffuseModelEntry(
            description="test",
            class_path="some.Module",
            freq_range=(1e6, 1e9),
        )
        assert entry.init_kwargs == {}
        assert entry.reference_url == ""


# =========================================================================
# CASDA TAP URL
# =========================================================================


def test_casda_tap_url():
    assert CASDA_TAP_URL.startswith("https://")
