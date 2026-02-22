"""
Tests for newly added sky model catalogs and diffuse model backends.

Test categories
---------------
- No-network (always run): catalog metadata validation, config class defaults,
  SkyModelConfig field presence, ImportError messages, FileNotFoundError,
  invalid argument validation.
- Network (``@pytest.mark.slow``): loading each catalog from VizieR/CASDA and
  verifying the returned SkyModel structure.
- Optional-dep (skip if dep missing): pysm3, pyradiosky, ulsa.
"""

import sys
import types

import numpy as np
import pytest

from rrivis.core.sky_model import (
    CASDA_TAP_URL,
    RACS_CATALOGS,
    VIZIER_POINT_CATALOGS,
    SkyModel,
)
from rrivis.io.config import (
    AT20GConfig,
    FIRSTConfig,
    GB6Config,
    LoTSSConfig,
    NVSSConfig,
    PyRadioSkyConfig,
    PySM3Config,
    RACSConfig,
    SkyModelConfig,
    SUMSSConfig,
    TGSSConfig,
    ThreeCConfig,
    ULSAConfig,
    VLSSrConfig,
    WENSSConfig,
)

# =============================================================================
# No-network tests
# =============================================================================


class TestVizierCatalogMetadata:
    """Validate that VIZIER_POINT_CATALOGS has all required schema fields."""

    REQUIRED_KEYS = {
        "vizier_id",
        "table",
        "description",
        "ra_col",
        "dec_col",
        "flux_col",
        "flux_unit",
        "spindex_col",
        "default_spindex",
        "freq_mhz",
        "coords_sexagesimal",
        "coord_frame",
    }

    def test_all_expected_catalogs_present(self):
        expected = {
            "vlssr", "tgss", "wenss", "sumss", "nvss", "first",
            "lotss_dr1", "lotss_dr2", "at20g", "3c", "gb6",
        }
        assert expected.issubset(set(VIZIER_POINT_CATALOGS.keys()))

    @pytest.mark.parametrize("key", list(VIZIER_POINT_CATALOGS.keys()))
    def test_required_fields_present(self, key):
        entry = VIZIER_POINT_CATALOGS[key]
        missing = self.REQUIRED_KEYS - set(entry.keys())
        assert not missing, f"Catalog '{key}' missing fields: {missing}"

    @pytest.mark.parametrize("key", list(VIZIER_POINT_CATALOGS.keys()))
    def test_flux_unit_valid(self, key):
        flux_unit = VIZIER_POINT_CATALOGS[key]["flux_unit"]
        assert flux_unit in ("Jy", "mJy"), (
            f"Catalog '{key}' has invalid flux_unit '{flux_unit}'"
        )

    @pytest.mark.parametrize("key", list(VIZIER_POINT_CATALOGS.keys()))
    def test_freq_mhz_positive(self, key):
        assert VIZIER_POINT_CATALOGS[key]["freq_mhz"] > 0

    @pytest.mark.parametrize("key", list(VIZIER_POINT_CATALOGS.keys()))
    def test_vizier_id_nonempty(self, key):
        assert VIZIER_POINT_CATALOGS[key]["vizier_id"] != ""

    def test_at20g_has_spindex_from_cols(self):
        at20g = VIZIER_POINT_CATALOGS["at20g"]
        assert "spindex_from_cols" in at20g
        scols = at20g["spindex_from_cols"]
        for field in ("s_low", "s_high", "freq_low_hz", "freq_high_hz"):
            assert field in scols, f"AT20G spindex_from_cols missing '{field}'"
        assert scols["freq_low_hz"] < scols["freq_high_hz"]

    def test_3c_uses_fk4_frame(self):
        assert VIZIER_POINT_CATALOGS["3c"]["coord_frame"] == "fk4"
        assert VIZIER_POINT_CATALOGS["3c"]["coords_sexagesimal"] is True

    def test_lotss_has_spindex_col(self):
        for release in ("lotss_dr1", "lotss_dr2"):
            assert VIZIER_POINT_CATALOGS[release]["spindex_col"] == "SpI"


class TestRacsCatalogMetadata:
    """Validate RACS_CATALOGS and CASDA_TAP_URL."""

    def test_casda_url_is_string(self):
        assert isinstance(CASDA_TAP_URL, str) and CASDA_TAP_URL.startswith("https://")

    def test_all_three_bands_present(self):
        assert {"low", "mid", "high"} == set(RACS_CATALOGS.keys())

    @pytest.mark.parametrize("band", ["low", "mid", "high"])
    def test_racs_band_has_required_fields(self, band):
        entry = RACS_CATALOGS[band]
        for field in ("freq_mhz", "tap_table", "ra_col", "dec_col", "flux_col", "flux_unit"):
            assert field in entry, f"RACS {band} missing '{field}'"

    def test_freq_ordering(self):
        assert RACS_CATALOGS["low"]["freq_mhz"] < RACS_CATALOGS["mid"]["freq_mhz"]
        assert RACS_CATALOGS["mid"]["freq_mhz"] < RACS_CATALOGS["high"]["freq_mhz"]


class TestNewConfigClasses:
    """Test Pydantic config classes for all new catalogs."""

    def test_vlssr_defaults(self):
        cfg = VLSSrConfig()
        assert cfg.use_vlssr is False
        assert cfg.flux_limit >= 0

    def test_tgss_defaults(self):
        cfg = TGSSConfig()
        assert cfg.use_tgss is False
        assert cfg.flux_limit >= 0

    def test_wenss_defaults(self):
        cfg = WENSSConfig()
        assert cfg.use_wenss is False

    def test_sumss_defaults(self):
        cfg = SUMSSConfig()
        assert cfg.use_sumss is False

    def test_nvss_defaults(self):
        cfg = NVSSConfig()
        assert cfg.use_nvss is False

    def test_first_defaults(self):
        cfg = FIRSTConfig()
        assert cfg.use_first is False

    def test_lotss_default_release(self):
        cfg = LoTSSConfig()
        assert cfg.lotss_release == "dr2"

    def test_lotss_valid_releases(self):
        cfg_dr1 = LoTSSConfig(lotss_release="dr1")
        assert cfg_dr1.lotss_release == "dr1"
        cfg_dr2 = LoTSSConfig(lotss_release="dr2")
        assert cfg_dr2.lotss_release == "dr2"

    def test_lotss_invalid_release_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LoTSSConfig(lotss_release="dr99")

    def test_at20g_defaults(self):
        cfg = AT20GConfig()
        assert cfg.use_at20g is False

    def test_three_c_defaults(self):
        cfg = ThreeCConfig()
        assert cfg.use_3c is False

    def test_gb6_defaults(self):
        cfg = GB6Config()
        assert cfg.use_gb6 is False

    def test_racs_default_band(self):
        cfg = RACSConfig()
        assert cfg.racs_band == "low"
        assert cfg.max_rows >= 1

    def test_racs_valid_bands(self):
        for band in ("low", "mid", "high"):
            cfg = RACSConfig(racs_band=band)
            assert cfg.racs_band == band

    def test_racs_invalid_band_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RACSConfig(racs_band="ultra")

    def test_pysm3_defaults(self):
        cfg = PySM3Config()
        assert cfg.use_pysm3 is False
        assert cfg.nside >= 1

    def test_pysm3_list_components(self):
        cfg = PySM3Config(components=["s1", "d1"])
        assert cfg.components == ["s1", "d1"]

    def test_ulsa_defaults(self):
        cfg = ULSAConfig()
        assert cfg.use_ulsa is False
        assert cfg.nside >= 1

    def test_pyradiosky_defaults(self):
        cfg = PyRadioSkyConfig()
        assert cfg.use_pyradiosky is False
        assert cfg.filename == ""
        assert cfg.filetype is None
        assert cfg.reference_frequency_hz is None

    def test_pyradiosky_accepts_filetype(self):
        cfg = PyRadioSkyConfig(filename="test.skyh5", filetype="skyh5")
        assert cfg.filetype == "skyh5"

    def test_flux_limit_must_be_nonnegative(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            VLSSrConfig(flux_limit=-1.0)


class TestSkyModelConfigHasNewFields:
    """Verify SkyModelConfig exposes all 14 new catalog fields."""

    NEW_FIELDS = [
        "vlssr", "tgss", "wenss", "sumss", "nvss", "first",
        "lotss", "at20g", "three_c", "gb6", "racs",
        "pysm3", "ulsa", "pyradiosky",
    ]

    def test_new_fields_present(self):
        cfg = SkyModelConfig()
        for field in self.NEW_FIELDS:
            assert hasattr(cfg, field), f"SkyModelConfig missing field '{field}'"

    def test_existing_fields_still_present(self):
        cfg = SkyModelConfig()
        for field in ("gleam", "mals", "gsm_healpix", "test_sources"):
            assert hasattr(cfg, field), f"SkyModelConfig missing existing field '{field}'"

    def test_default_all_disabled(self):
        cfg = SkyModelConfig()
        assert cfg.vlssr.use_vlssr is False
        assert cfg.tgss.use_tgss is False
        assert cfg.nvss.use_nvss is False
        assert cfg.lotss.use_lotss is False
        assert cfg.racs.use_racs is False
        assert cfg.pysm3.use_pysm3 is False
        assert cfg.ulsa.use_ulsa is False
        assert cfg.pyradiosky.use_pyradiosky is False


class TestImportErrors:
    """Verify helpful ImportError messages when optional deps are missing."""

    def test_from_pysm3_raises_import_error_without_pysm3(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pysm3", None)
        with pytest.raises(ImportError, match="pysm3"):
            SkyModel.from_pysm3(
                frequencies=np.linspace(100e6, 110e6, 2),
            )

    def test_from_ulsa_raises_import_error_without_ulsa(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ulsa", None)
        with pytest.raises(ImportError, match="ulsa"):
            SkyModel.from_ulsa(
                frequencies=np.linspace(10e6, 20e6, 2),
            )

    def test_from_pyradiosky_file_raises_import_error_without_pyradiosky(self, monkeypatch, tmp_path):
        # Create a real file so FileNotFoundError doesn't trigger first
        fake_file = tmp_path / "test.skyh5"
        fake_file.touch()
        monkeypatch.setitem(sys.modules, "pyradiosky", None)
        with pytest.raises(ImportError, match="pyradiosky"):
            SkyModel.from_pyradiosky_file(str(fake_file))


class TestFileNotFound:
    """Verify FileNotFoundError when file doesn't exist."""

    def test_nonexistent_file_raises(self):
        # pyradiosky must be importable for this test to reach the file check
        pytest.importorskip("pyradiosky")
        with pytest.raises(FileNotFoundError):
            SkyModel.from_pyradiosky_file("nonexistent_totally_fake_file.skyh5")


class TestInvalidArguments:
    """Validate argument guard-rails in factory methods."""

    def test_racs_invalid_band_raises(self):
        with pytest.raises(ValueError, match="ultra"):
            SkyModel.from_racs(band="ultra")

    def test_lotss_invalid_release_raises(self):
        with pytest.raises(ValueError, match="dr99"):
            SkyModel.from_lotss(release="dr99")

    def test_unknown_vizier_catalog_key_raises(self):
        with pytest.raises(ValueError, match="unknown_catalog"):
            SkyModel._load_from_vizier_catalog("unknown_catalog")

    def test_from_pysm3_requires_frequencies(self):
        pytest.importorskip("pysm3")
        with pytest.raises(ValueError, match="frequencies"):
            SkyModel.from_pysm3()  # no frequencies, no obs_frequency_config

    def test_from_ulsa_requires_frequencies(self):
        pytest.importorskip("ulsa")
        with pytest.raises(ValueError, match="frequencies"):
            SkyModel.from_ulsa()  # no frequencies, no obs_frequency_config

    def test_from_pyradiosky_rejects_healpix_component(self, tmp_path, monkeypatch):
        """Verify ValueError when pyradiosky file has HEALPix component type."""
        pytest.importorskip("pyradiosky")

        # Build a minimal mock SkyModel with component_type='healpix'
        class MockSky:
            component_type = "healpix"
            def read(self, filename, filetype=None):
                pass

        fake_file = tmp_path / "test.skyh5"
        fake_file.touch()

        # Patch the pyradiosky import inside sky_model module
        mock_module = types.ModuleType("pyradiosky")
        mock_module.SkyModel = MockSky
        monkeypatch.setitem(sys.modules, "pyradiosky", mock_module)

        with pytest.raises(ValueError, match="point"):
            SkyModel.from_pyradiosky_file(str(fake_file))


# =============================================================================
# Optional-dep tests (skip if dep missing)
# =============================================================================


class TestPySM3:
    """PySM3 integration tests — skipped if pysm3 not installed."""

    @staticmethod
    def _ensure_pysm3():
        return pytest.importorskip("pysm3")

    def test_from_pysm3_mode_is_healpix_multifreq(self):
        self._ensure_pysm3()
        freqs = np.linspace(100e6, 102e6, 3)
        sky = SkyModel.from_pysm3(components="s1", nside=16, frequencies=freqs)
        assert sky.mode == "healpix_multifreq"

    def test_from_pysm3_one_map_per_freq(self):
        self._ensure_pysm3()
        freqs = np.linspace(100e6, 102e6, 3)
        sky = SkyModel.from_pysm3(components="s1", nside=16, frequencies=freqs)
        maps, nside, obs_freqs = sky.get_multifreq_maps()
        assert len(maps) == len(freqs)
        assert obs_freqs is not None

    def test_from_pysm3_healpix_shape(self):
        self._ensure_pysm3()
        import healpy as hp
        nside = 16
        freqs = np.linspace(100e6, 101e6, 2)
        sky = SkyModel.from_pysm3(components="s1", nside=nside, frequencies=freqs)
        maps, _, _ = sky.get_multifreq_maps()
        expected_npix = hp.nside2npix(nside)
        for freq, t_map in maps.items():
            assert len(t_map) == expected_npix, (
                f"Map at {freq/1e6:.1f} MHz has {len(t_map)} pixels, "
                f"expected {expected_npix}"
            )

    def test_from_pysm3_raises_without_frequencies(self):
        self._ensure_pysm3()
        with pytest.raises(ValueError, match="frequencies"):
            SkyModel.from_pysm3(components="s1", nside=16)

    def test_from_pysm3_with_obs_frequency_config(self):
        self._ensure_pysm3()
        config = {
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 3.0,
            "frequency_unit": "MHz",
        }
        sky = SkyModel.from_pysm3(components="s1", nside=16, obs_frequency_config=config)
        assert sky.mode == "healpix_multifreq"
        assert sky.n_frequencies == 3

    def test_from_pysm3_multi_component(self):
        self._ensure_pysm3()
        freqs = np.array([100e6, 101e6])
        sky = SkyModel.from_pysm3(components=["s1", "f1"], nside=16, frequencies=freqs)
        assert sky.mode == "healpix_multifreq"
        assert "pysm3:" in (sky.model_name or "")


class TestULSA:
    """ULSA integration tests — skipped if ulsa not installed."""

    @staticmethod
    def _ensure_ulsa():
        return pytest.importorskip("ulsa")

    def test_from_ulsa_mode_is_healpix_multifreq(self):
        self._ensure_ulsa()
        freqs = np.linspace(10e6, 12e6, 3)
        sky = SkyModel.from_ulsa(nside=16, frequencies=freqs)
        assert sky.mode == "healpix_multifreq"

    def test_from_ulsa_one_map_per_freq(self):
        self._ensure_ulsa()
        freqs = np.linspace(10e6, 12e6, 3)
        sky = SkyModel.from_ulsa(nside=16, frequencies=freqs)
        maps, _, _ = sky.get_multifreq_maps()
        assert len(maps) == len(freqs)

    def test_from_ulsa_raises_without_frequencies(self):
        self._ensure_ulsa()
        with pytest.raises(ValueError, match="frequencies"):
            SkyModel.from_ulsa(nside=16)


class TestPyRadioSkyFile:
    """pyradiosky file loader tests — skipped if pyradiosky not installed."""

    def test_load_skyh5_returns_point_sources(self, tmp_path):
        PSky = pytest.importorskip("pyradiosky").SkyModel
        import astropy.units as au
        from astropy.coordinates import SkyCoord

        stokes = np.zeros((4, 1, 3)) * au.Jy
        stokes[0, 0, :] = [1.0, 2.0, 3.0] * au.Jy

        sky = PSky(
            ra=SkyCoord(ra=[10, 20, 30] * au.deg, dec=[-30, -40, -50] * au.deg).ra,
            dec=SkyCoord(ra=[10, 20, 30] * au.deg, dec=[-30, -40, -50] * au.deg).dec,
            stokes=stokes,
            freq_array=np.array([150e6]) * au.Hz,
            spectral_type="flat",
        )

        skyh5_path = tmp_path / "test.skyh5"
        sky.write(str(skyh5_path))

        loaded = SkyModel.from_pyradiosky_file(str(skyh5_path))
        assert loaded.mode == "point_sources"
        sources = loaded.to_point_sources()
        assert len(sources) == 3

    def test_source_structure(self, tmp_path):
        PSky = pytest.importorskip("pyradiosky").SkyModel
        import astropy.units as au
        from astropy.coordinates import SkyCoord

        stokes = np.zeros((4, 1, 2)) * au.Jy
        stokes[0, 0, :] = [5.0, 10.0] * au.Jy

        sky = PSky(
            ra=SkyCoord(ra=[0, 90] * au.deg, dec=[0, 0] * au.deg).ra,
            dec=SkyCoord(ra=[0, 90] * au.deg, dec=[0, 0] * au.deg).dec,
            stokes=stokes,
            freq_array=np.array([100e6]) * au.Hz,
            spectral_type="flat",
        )

        skyh5_path = tmp_path / "test2.skyh5"
        sky.write(str(skyh5_path))

        loaded = SkyModel.from_pyradiosky_file(str(skyh5_path), flux_limit=1.0)
        sources = loaded.to_point_sources()
        assert len(sources) == 2
        for src in sources:
            assert "coords" in src
            assert "flux" in src
            assert src["flux"] >= 1.0
            assert "spectral_index" in src

    def test_flux_limit_applied(self, tmp_path):
        PSky = pytest.importorskip("pyradiosky").SkyModel
        import astropy.units as au
        from astropy.coordinates import SkyCoord

        stokes = np.zeros((4, 1, 4)) * au.Jy
        stokes[0, 0, :] = [0.5, 2.0, 5.0, 0.1] * au.Jy

        sky = PSky(
            ra=SkyCoord(ra=[0, 90, 180, 270] * au.deg, dec=[0, 0, 0, 0] * au.deg).ra,
            dec=SkyCoord(ra=[0, 90, 180, 270] * au.deg, dec=[0, 0, 0, 0] * au.deg).dec,
            stokes=stokes,
            freq_array=np.array([150e6]) * au.Hz,
            spectral_type="flat",
        )
        skyh5_path = tmp_path / "test3.skyh5"
        sky.write(str(skyh5_path))

        loaded = SkyModel.from_pyradiosky_file(str(skyh5_path), flux_limit=1.0)
        sources = loaded.to_point_sources()
        # Only sources with flux >= 1.0 Jy: 2.0 and 5.0
        assert len(sources) == 2
        assert all(s["flux"] >= 1.0 for s in sources)


# =============================================================================
# Network tests
# =============================================================================


@pytest.mark.slow
class TestNetworkVLSSr:
    def test_load_returns_sky_model(self):
        sky = SkyModel.from_vlssr(flux_limit=100.0)
        assert isinstance(sky, SkyModel)
        assert sky.mode == "point_sources"

    def test_source_structure(self):
        sky = SkyModel.from_vlssr(flux_limit=100.0)
        sources = sky.to_point_sources()
        if sources:
            src = sources[0]
            assert "coords" in src
            assert "flux" in src
            assert src["flux"] >= 100.0
            assert "spectral_index" in src


@pytest.mark.slow
class TestNetworkTGSS:
    def test_load_returns_sky_model(self):
        sky = SkyModel.from_tgss(flux_limit=10.0)  # 10 Jy → few bright sources
        assert isinstance(sky, SkyModel)

    def test_flux_in_jy(self):
        """Verify mJy→Jy conversion: loaded sources should meet flux_limit in Jy."""
        flux_limit_jy = 10.0
        sky = SkyModel.from_tgss(flux_limit=flux_limit_jy)
        sources = sky.to_point_sources()
        for src in sources:
            assert src["flux"] >= flux_limit_jy


@pytest.mark.slow
class TestNetworkNVSS:
    def test_load_returns_sky_model(self):
        sky = SkyModel.from_nvss(flux_limit=10.0)  # high limit for speed
        assert isinstance(sky, SkyModel)
        assert sky.mode == "point_sources"

    def test_source_structure(self):
        sky = SkyModel.from_nvss(flux_limit=10.0)
        sources = sky.to_point_sources()
        if sources:
            src = sources[0]
            assert "coords" in src
            assert "flux" in src


@pytest.mark.slow
class TestNetworkAT20G:
    def test_load_returns_sky_model(self):
        sky = SkyModel.from_at20g(flux_limit=0.5)
        assert isinstance(sky, SkyModel)

    def test_spectral_index_finite(self):
        sky = SkyModel.from_at20g(flux_limit=0.5)
        sources = sky.to_point_sources()
        for src in sources:
            assert np.isfinite(src["spectral_index"]), (
                f"Non-finite spectral index: {src['spectral_index']}"
            )


@pytest.mark.slow
class TestNetwork3C:
    def test_load_returns_sky_model(self):
        sky = SkyModel.from_3c(flux_limit=1.0)
        assert isinstance(sky, SkyModel)

    def test_icrs_coords(self):
        """Verify FK4→ICRS conversion produces valid ICRS coordinates."""
        sky = SkyModel.from_3c(flux_limit=1.0)
        sources = sky.to_point_sources()
        for src in sources:
            coords = src["coords"]
            assert coords.frame.name == "icrs", (
                f"Expected ICRS frame, got {coords.frame.name}"
            )
            assert 0 <= coords.ra.deg <= 360
            assert -90 <= coords.dec.deg <= 90


@pytest.mark.slow
class TestNetworkLoTSS:
    def test_dr1_load(self):
        sky = SkyModel.from_lotss(release="dr1", flux_limit=10.0)
        assert isinstance(sky, SkyModel)

    def test_dr2_load(self):
        sky = SkyModel.from_lotss(release="dr2", flux_limit=10.0)
        assert isinstance(sky, SkyModel)


@pytest.mark.slow
class TestNetworkRACS:
    def test_low_band(self):
        sky = SkyModel.from_racs(band="low", flux_limit=1.0, max_rows=1000)
        assert isinstance(sky, SkyModel)

    def test_mid_band(self):
        sky = SkyModel.from_racs(band="mid", flux_limit=1.0, max_rows=1000)
        assert isinstance(sky, SkyModel)

    def test_high_band(self):
        sky = SkyModel.from_racs(band="high", flux_limit=1.0, max_rows=1000)
        assert isinstance(sky, SkyModel)
