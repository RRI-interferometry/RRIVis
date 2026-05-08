"""Tests for the SkyProvenance metadata plumbing.

Covers:
- SkyProvenance dataclass coercion and validation
- Default (UNKNOWN) provenance on manual SkyModel construction
- Provenance preservation through ``SkyModel.replace``
- Loader-side population: synthetic test sources, (mocked) vizier & racs
  provenance helpers, and diffuse loaders via catalog metadata lookup.

These tests do NOT hit the network — they exercise the local code paths only.
"""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import (
    MonopoleConvention,
    PointSourceData,
    SkyCoverage,
    SkyModel,
    SkyProvenance,
    SkyRegion,
    SourceSubtractionStatus,
    create_empty,
    create_from_arrays,
    create_test_sources,
)


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


class TestSkyProvenanceDataclass:
    def test_defaults_are_unknown(self):
        prov = SkyProvenance()
        assert prov.monopole_convention is MonopoleConvention.UNKNOWN
        assert prov.sky_coverage is SkyCoverage.UNKNOWN
        assert prov.coverage_fraction is None
        assert prov.coverage_footprint is None
        assert prov.source_subtraction is SourceSubtractionStatus.UNKNOWN
        assert prov.flux_completeness_jy is None
        assert prov.angular_resolution_rad is None
        assert prov.monopole_k is None
        assert not prov.has_flux_completeness
        assert not prov.has_angular_resolution
        assert not prov.is_source_subtracted

    def test_string_coercion(self):
        prov = SkyProvenance(
            sky_coverage="full_sky",
            monopole_convention="absolute_no_cmb",
            source_subtraction="above_threshold",
            source_subtraction_threshold_jy=2.0,
        )
        assert prov.sky_coverage is SkyCoverage.FULL_SKY
        assert prov.monopole_convention is MonopoleConvention.ABSOLUTE_NO_CMB
        assert prov.source_subtraction is SourceSubtractionStatus.ABOVE_THRESHOLD
        assert prov.is_source_subtracted

    def test_threshold_without_status_rejected(self):
        with pytest.raises(ValueError, match="source_subtraction is NONE"):
            SkyProvenance(
                monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
                source_subtraction=SourceSubtractionStatus.NONE,
                source_subtraction_threshold_jy=1.0,
            )

    def test_status_above_threshold_requires_threshold(self):
        with pytest.raises(ValueError, match="ABOVE_THRESHOLD requires"):
            SkyProvenance(
                source_subtraction=SourceSubtractionStatus.ABOVE_THRESHOLD,
                source_subtraction_threshold_jy=None,
            )

    def test_partial_sky_rejects_monopole(self):
        with pytest.raises(ValueError, match="must be None"):
            SkyProvenance(
                sky_coverage=SkyCoverage.PARTIAL_SKY,
                coverage_fraction=0.25,
                monopole_k=10.0,
            )


class TestSkyModelProvenanceField:
    def test_default_provenance_is_unknown(self, precision):
        sky = create_empty("empty", precision=precision)
        assert sky.provenance.monopole_convention is MonopoleConvention.UNKNOWN

    def test_dict_coercion_at_construction(self, precision):
        sky = SkyModel(
            point=PointSourceData.empty(),
            _precision=precision,
            provenance={
                "monopole_convention": "absolute_no_cmb",
                "source_subtraction": "none",
                "notes": "from-dict",
            },
        )
        assert sky.provenance.monopole_convention is MonopoleConvention.ABSOLUTE_NO_CMB
        assert sky.provenance.notes == "from-dict"

    def test_bad_provenance_type_rejected(self, precision):
        with pytest.raises(TypeError, match="must be a SkyProvenance or a dict"):
            SkyModel(
                point=PointSourceData.empty(),
                _precision=precision,
                provenance="bogus",
            )

    def test_provenance_preserved_through_replace(self, precision):
        prov = SkyProvenance(
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            source_subtraction=SourceSubtractionStatus.NONE,
            notes="original",
        )
        sky = create_empty("empty", precision=precision, provenance=prov)
        # Replacing an unrelated field must not clobber provenance.
        replaced = sky.replace(model_name="renamed")
        assert replaced.provenance == prov
        # Explicitly replacing provenance overrides.
        new_prov = SkyProvenance(notes="updated")
        updated = sky.replace(provenance=new_prov)
        assert updated.provenance.notes == "updated"

    def test_equality_distinguishes_provenance(self, precision):
        a = create_empty(
            "same",
            precision=precision,
            provenance=SkyProvenance(notes="a"),
        )
        b = create_empty(
            "same",
            precision=precision,
            provenance=SkyProvenance(notes="b"),
        )
        assert a != b
        c = create_empty(
            "same",
            precision=precision,
            provenance=SkyProvenance(notes="a"),
        )
        assert a == c


class TestLoaderPopulation:
    def test_create_test_sources_populates_provenance(self, precision):
        sky = create_test_sources(
            num_sources=5,
            flux_range=(2.0, 8.0),
            precision=precision,
            reference_frequency=150e6,
        )
        prov = sky.provenance
        assert prov.monopole_convention is MonopoleConvention.ABSOLUTE_NO_CMB
        assert prov.source_subtraction is SourceSubtractionStatus.NONE
        assert prov.flux_completeness_jy == (2.0, 8.0)
        assert prov.flux_completeness_freq_hz == pytest.approx(150e6)
        assert prov.angular_resolution_rad == (0.0, pytest.approx(np.pi))
        assert prov.sky_coverage is SkyCoverage.FULL_SKY
        assert prov.coverage_fraction == pytest.approx(1.0)

    def test_create_from_arrays_accepts_provenance(self, precision):
        prov = SkyProvenance(
            flux_completeness_jy=(0.1, 100.0),
            flux_completeness_freq_hz=200e6,
            angular_resolution_rad=(1e-4, np.pi),
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            source_subtraction=SourceSubtractionStatus.NONE,
            notes="custom",
        )
        sky = create_from_arrays(
            ra_rad=np.array([0.1, 0.2]),
            dec_rad=np.array([-0.5, -0.4]),
            flux=np.array([1.0, 2.0]),
            reference_frequency=200e6,
            precision=precision,
            provenance=prov,
        )
        assert sky.provenance == prov

    def test_vizier_helper_builds_catalog_provenance(self):
        # Exercise the helper directly so we don't hit the VizieR network.
        from radiosim.core.sky._loaders_vizier import (
            _build_point_catalog_provenance,
        )
        from radiosim.core.sky.catalogs import VIZIER_POINT_CATALOGS

        info = VIZIER_POINT_CATALOGS["nvss"]
        flux = np.array([2.0, 5.0, 10.0])
        prov = _build_point_catalog_provenance(
            info=info,
            flux_limit_jy=1.0,
            flux_jy=flux,
            catalog_key="nvss",
        )
        assert prov.monopole_convention is MonopoleConvention.ABSOLUTE_NO_CMB
        assert prov.sky_coverage is SkyCoverage.PARTIAL_SKY
        assert prov.coverage_footprint is not None
        assert prov.coverage_fraction is not None
        assert 0.0 < prov.coverage_fraction < 1.0
        assert prov.source_subtraction is SourceSubtractionStatus.NONE
        # Upper bound is the catalog's saturation limit (None for NVSS in the
        # current metadata) — we encode "no upper limit known" as +inf rather
        # than max(loaded sample), which would conflate a sample statistic
        # with intrinsic catalog metadata.
        assert prov.flux_completeness_jy == (1.0, float("inf"))
        assert prov.flux_completeness_freq_hz == pytest.approx(1400e6)
        # Beam FWHM 45″ → radians.
        assert prov.angular_resolution_rad is not None
        beam_rad_lo, beam_rad_hi = prov.angular_resolution_rad
        assert beam_rad_lo == pytest.approx(45 * np.pi / 180.0 / 3600.0)
        assert beam_rad_hi == pytest.approx(np.pi)

    def test_vizier_helper_without_curated_footprint_uses_unknown_fraction(self):
        from radiosim.core.sky._loaders_vizier import (
            _build_point_catalog_provenance,
        )
        from radiosim.core.sky.catalogs import VIZIER_POINT_CATALOGS

        prov = _build_point_catalog_provenance(
            info=VIZIER_POINT_CATALOGS["gleam_egc"],
            flux_limit_jy=1.0,
            flux_jy=np.array([2.0, 5.0]),
            catalog_key="gleam_egc",
        )

        assert prov.sky_coverage is SkyCoverage.PARTIAL_SKY
        assert prov.coverage_fraction is None
        assert prov.coverage_footprint is None

    def test_racs_helper_builds_catalog_provenance(self):
        from radiosim.core.sky._loaders_vizier import (
            _build_point_catalog_provenance,
        )
        from radiosim.core.sky.catalogs import RACS_CATALOGS

        prov = _build_point_catalog_provenance(
            info=RACS_CATALOGS["mid"],
            flux_limit_jy=0.01,
            flux_jy=np.array([0.05, 0.5]),
            catalog_key="racs_mid",
        )
        assert prov.sky_coverage is SkyCoverage.PARTIAL_SKY
        assert prov.coverage_footprint is not None
        assert prov.coverage_fraction is not None
        assert prov.flux_completeness_freq_hz == pytest.approx(1367.5e6)
        assert prov.notes == "racs/racs_mid"

    def test_diffuse_catalog_metadata_is_complete(self):
        """Every registered diffuse model carries the new metadata fields."""
        from radiosim.core.sky.catalogs import DIFFUSE_MODELS

        for name, entry in DIFFUSE_MODELS.items():
            assert entry.native_resolution_arcmin is not None, name
            assert entry.default_monopole_convention is not None, name

    def test_haslam_is_tagged_source_subtracted(self):
        from radiosim.core.sky.catalogs import DIFFUSE_MODELS

        haslam = DIFFUSE_MODELS["haslam"]
        assert haslam.source_subtracted_above_jy == pytest.approx(2.0)
        assert haslam.source_subtraction_freq_hz == pytest.approx(408e6)

    def test_diffuse_loader_provenance_via_fake_pygdsm(self, precision, monkeypatch):
        """Drive load_diffuse_sky against a fake pygdsm class to exercise the
        provenance-population branch without a network fetch."""
        from radiosim.core.sky import _loaders_diffuse as diffuse_mod
        from radiosim.core.sky._loaders_diffuse import load_diffuse_sky

        nside = 4
        npix = hp.nside2npix(nside)

        class FakePyGDSM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def generate(self, freq):
                # pygdsm returns maps in Galactic coords; simulate that.
                return np.full(npix, 250.0, dtype=np.float64)

        monkeypatch.setattr(
            diffuse_mod, "_resolve_model_class", lambda _path: FakePyGDSM
        )

        sky = load_diffuse_sky(
            model="gsm2008",
            nside=nside,
            frequencies=np.array([100e6, 150e6]),
            precision=precision,
        )
        prov = sky.provenance
        assert prov.monopole_convention is MonopoleConvention.ABSOLUTE_NO_CMB
        assert prov.source_subtraction is SourceSubtractionStatus.NONE
        assert prov.angular_resolution_rad is not None
        # Haslam basemap → 60 arcmin lower bound.
        assert prov.angular_resolution_rad[0] == pytest.approx(60.0 * np.pi / 10800.0)
        assert prov.monopole_k == pytest.approx(250.0)
        assert prov.sky_coverage is SkyCoverage.FULL_SKY
        assert prov.coverage_fraction == pytest.approx(1.0)
        assert prov.notes == "pygdsm/gsm2008"

        # include_cmb=True must flip the convention to ABSOLUTE_WITH_CMB.
        sky_cmb = load_diffuse_sky(
            model="gsm2008",
            nside=nside,
            frequencies=np.array([100e6]),
            include_cmb=True,
            precision=precision,
        )
        assert (
            sky_cmb.provenance.monopole_convention
            is MonopoleConvention.ABSOLUTE_WITH_CMB
        )

    def test_diffuse_loader_region_clears_monopole(self, precision, monkeypatch):
        from radiosim.core.sky import _loaders_diffuse as diffuse_mod
        from radiosim.core.sky._loaders_diffuse import load_diffuse_sky

        nside = 8
        npix = hp.nside2npix(nside)

        class FakePyGDSM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def generate(self, freq):
                return np.full(npix, 30.0, dtype=np.float64)

        monkeypatch.setattr(
            diffuse_mod, "_resolve_model_class", lambda _path: FakePyGDSM
        )

        sky = load_diffuse_sky(
            model="gsm2008",
            nside=nside,
            frequencies=np.array([150e6]),
            region=SkyRegion.cone(180.0, -30.0, 10.0),
            precision=precision,
        )
        assert sky.provenance.sky_coverage is SkyCoverage.PARTIAL_SKY
        assert sky.provenance.coverage_fraction is not None
        assert sky.provenance.coverage_footprint is not None
        assert 0.0 < sky.provenance.coverage_fraction < 1.0
        assert sky.provenance.monopole_k is None

    def test_haslam_loader_provenance_tagged_source_subtracted(
        self, precision, monkeypatch
    ):
        from radiosim.core.sky import _loaders_diffuse as diffuse_mod
        from radiosim.core.sky._loaders_diffuse import load_diffuse_sky

        nside = 4
        npix = hp.nside2npix(nside)

        class FakePyGDSM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def generate(self, freq):
                return np.full(npix, 30.0, dtype=np.float64)

        monkeypatch.setattr(
            diffuse_mod, "_resolve_model_class", lambda _path: FakePyGDSM
        )

        sky = load_diffuse_sky(
            model="haslam",
            nside=nside,
            frequencies=np.array([408e6]),
            precision=precision,
        )
        prov = sky.provenance
        assert prov.source_subtraction is SourceSubtractionStatus.ABOVE_THRESHOLD
        assert prov.source_subtraction_threshold_jy == pytest.approx(2.0)
        assert prov.source_subtraction_freq_hz == pytest.approx(408e6)
        assert prov.source_subtraction_method == "gaussian_fit_inpaint"


class TestSkyH5ProvenanceRoundTrip:
    """SkyProvenance must survive a save_skyh5 / load_skyh5 round-trip.

    Without this guarantee, every disjointness check that runs after a
    SkyH5 save would silently see UNKNOWN provenance and either
    fail-closed (error policy) or admit double-counting (warn/allow).
    """

    def test_full_provenance_round_trip(self, precision, tmp_path):
        from radiosim.core.sky import load_skyh5, save_skyh5

        n = 5
        provenance = SkyProvenance(
            flux_completeness_jy=(0.05, 50.0),
            flux_completeness_freq_hz=200e6,
            angular_resolution_rad=(1e-4, 0.05),
            sky_coverage=SkyCoverage.FULL_SKY,
            coverage_fraction=1.0,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            monopole_k=42.0,
            source_subtraction=SourceSubtractionStatus.ABOVE_THRESHOLD,
            source_subtraction_threshold_jy=1.5,
            source_subtraction_freq_hz=200e6,
            source_subtraction_method="gaussian_fit_inpaint",
            notes="round-trip-test",
        )
        sky = create_from_arrays(
            ra_rad=np.linspace(0.0, 1.0, n),
            dec_rad=np.linspace(-0.4, 0.4, n),
            flux=np.linspace(1.0, 5.0, n),
            spectral_index=np.full(n, -0.7),
            reference_frequency=200e6,
            precision=precision,
            provenance=provenance,
        )

        out = tmp_path / "round_trip.skyh5"
        save_skyh5(sky, str(out))
        round_tripped = load_skyh5(str(out), precision=precision)

        rt_prov = round_tripped.provenance
        assert rt_prov.flux_completeness_jy == pytest.approx(
            provenance.flux_completeness_jy
        )
        assert rt_prov.flux_completeness_freq_hz == pytest.approx(
            provenance.flux_completeness_freq_hz
        )
        assert rt_prov.angular_resolution_rad == pytest.approx(
            provenance.angular_resolution_rad
        )
        assert rt_prov.sky_coverage is provenance.sky_coverage
        assert rt_prov.coverage_fraction == pytest.approx(provenance.coverage_fraction)
        assert rt_prov.monopole_convention is provenance.monopole_convention
        assert rt_prov.monopole_k == pytest.approx(provenance.monopole_k)
        assert rt_prov.source_subtraction is provenance.source_subtraction
        assert rt_prov.source_subtraction_threshold_jy == pytest.approx(
            provenance.source_subtraction_threshold_jy
        )
        assert rt_prov.source_subtraction_freq_hz == pytest.approx(
            provenance.source_subtraction_freq_hz
        )
        assert rt_prov.source_subtraction_method == provenance.source_subtraction_method
        assert rt_prov.notes == provenance.notes
