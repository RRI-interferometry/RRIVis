"""Tests for the public SkyModel API."""

import dataclasses

import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import HealpixData, PointSourceData, SkyRegion
from rrivis.core.sky.discovery import estimate_healpix_memory
from rrivis.core.sky.model import SkyFormat, SkyModel


@pytest.fixture
def precision():
    return PrecisionConfig.standard()


@pytest.fixture
def obs_freq_config():
    return {
        "starting_frequency": 100.0,
        "frequency_interval": 1.0,
        "frequency_bandwidth": 5.0,
        "frequency_unit": "MHz",
    }


def make_point_model(
    *,
    n: int = 8,
    precision: PrecisionConfig | None = None,
    reference_frequency: float | None = 100e6,
    model_name: str = "point",
    seed: int = 0,
) -> SkyModel:
    rng = np.random.default_rng(seed)
    return SkyModel.from_arrays(
        ra_rad=rng.uniform(0, 2 * np.pi, n),
        dec_rad=rng.uniform(-np.pi / 2, np.pi / 2, n),
        flux=rng.uniform(1.0, 10.0, n),
        spectral_index=np.full(n, -0.7),
        stokes_q=np.zeros(n),
        stokes_u=np.zeros(n),
        stokes_v=np.zeros(n),
        reference_frequency=reference_frequency,
        model_name=model_name,
        precision=precision,
    )


def make_healpix_model(
    *,
    nside: int = 8,
    freqs: np.ndarray | None = None,
    precision: PrecisionConfig | None = None,
    model_name: str = "healpix",
    include_pol: bool = False,
) -> SkyModel:
    if freqs is None:
        freqs = np.array([100e6, 101e6], dtype=np.float64)
    npix = hp.nside2npix(nside)
    maps = np.ones((len(freqs), npix), dtype=np.float32)
    q_maps = np.full((len(freqs), npix), 0.1, dtype=np.float32) if include_pol else None
    u_maps = (
        np.full((len(freqs), npix), 0.05, dtype=np.float32) if include_pol else None
    )
    v_maps = (
        np.full((len(freqs), npix), 0.01, dtype=np.float32) if include_pol else None
    )
    return SkyModel(
        healpix=HealpixData(
            maps=maps,
            nside=nside,
            frequencies=freqs,
            q_maps=q_maps,
            u_maps=u_maps,
            v_maps=v_maps,
        ),
        native_representation=SkyFormat.HEALPIX,
        active_representation=SkyFormat.HEALPIX,
        reference_frequency=float(freqs[0]),
        model_name=model_name,
        _precision=precision,
    )


@pytest.fixture
def test_sky(precision):
    return make_point_model(n=50, precision=precision, model_name="test")


class TestPostInitValidation:
    def test_requires_payload(self):
        with pytest.raises(ValueError, match="requires at least one payload"):
            SkyModel()

    def test_invalid_native_representation_raises(self, precision):
        with pytest.raises(
            TypeError, match="native_representation must be a SkyFormat enum"
        ):
            SkyModel(
                point=PointSourceData.empty(),
                native_representation="invalid",
                _precision=precision,
            )

    def test_invalid_active_representation_raises(self, precision):
        with pytest.raises(ValueError, match="is not available on this model"):
            SkyModel(
                point=PointSourceData.empty(),
                native_representation=SkyFormat.POINT_SOURCES,
                active_representation=SkyFormat.HEALPIX,
                _precision=precision,
            )

    def test_point_payload_validation_is_strict(self):
        with pytest.raises(ValueError, match="expected 5"):
            PointSourceData(
                ra_rad=np.zeros(5),
                dec_rad=np.zeros(4),
                flux=np.ones(5),
                spectral_index=np.zeros(5),
                stokes_q=np.zeros(5),
                stokes_u=np.zeros(5),
                stokes_v=np.zeros(5),
                ref_freq=np.full(5, 100e6),
            )

    def test_morphology_validation_is_strict(self):
        with pytest.raises(ValueError, match="all set or all None"):
            PointSourceData(
                ra_rad=np.zeros(5),
                dec_rad=np.zeros(5),
                flux=np.ones(5),
                spectral_index=np.zeros(5),
                stokes_q=np.zeros(5),
                stokes_u=np.zeros(5),
                stokes_v=np.zeros(5),
                ref_freq=np.full(5, 100e6),
                major_arcsec=np.ones(5),
            )

    def test_healpix_requires_matching_frequency_axis(self):
        with pytest.raises(ValueError, match="channels"):
            HealpixData(
                maps=np.ones((2, hp.nside2npix(8)), dtype=np.float32),
                nside=8,
                frequencies=np.array([100e6]),
            )


class TestBasicBehavior:
    def test_frozen_instance(self, test_sky):
        with pytest.raises(dataclasses.FrozenInstanceError):
            test_sky.reference_frequency = 123.0

    def test_from_test_sources_basic(self, precision):
        sky = SkyModel.from_test_sources(
            num_sources=20,
            flux_range=(1.0, 5.0),
            dec_deg=-30.0,
            spectral_index=-0.8,
            precision=precision,
        )
        assert sky.mode == SkyFormat.POINT_SOURCES
        assert sky.native_format == SkyFormat.POINT_SOURCES
        assert sky.n_point_sources == 20
        assert np.allclose(sky.spectral_index, -0.8)

    def test_from_catalog_uses_public_registry(self, precision):
        sky = SkyModel.from_catalog(
            "test_sources",
            num_sources=4,
            reference_frequency=100e6,
            precision=precision,
        )
        assert sky.n_point_sources == 4

    def test_removed_compatibility_wrappers(self):
        assert not hasattr(SkyModel, "with_healpix_maps")
        assert not hasattr(SkyModel, "with_representation")


class TestMaterialization:
    def test_materialize_healpix_preserves_point_payload(
        self, test_sky, obs_freq_config
    ):
        sky = test_sky.materialize_healpix(
            nside=16, obs_frequency_config=obs_freq_config
        )
        assert sky is not test_sky
        assert sky.point is not None
        assert sky.healpix is not None
        assert sky.mode == SkyFormat.HEALPIX
        assert sky.n_pixels == hp.nside2npix(16)

    def test_materialize_healpix_requires_reference_frequency(self, precision):
        sky = make_point_model(n=5, precision=precision, reference_frequency=None)
        with pytest.raises(ValueError, match="ref_frequency must be provided"):
            sky.materialize_healpix(nside=16, frequencies=np.array([100e6]))

    def test_materialize_healpix_rejects_conflicting_frequency_inputs(self, test_sky):
        with pytest.raises(
            ValueError, match="either 'frequencies' or 'obs_frequency_config'"
        ):
            test_sky.materialize_healpix(
                nside=8,
                frequencies=np.array([100e6]),
                obs_frequency_config={"starting_frequency": 100.0},
            )

    def test_activate_switches_existing_representation(self, test_sky, obs_freq_config):
        hp_sky = test_sky.materialize_healpix(
            nside=8, obs_frequency_config=obs_freq_config
        )
        ps_sky = hp_sky.activate("point_sources")
        assert ps_sky.mode == SkyFormat.POINT_SOURCES
        assert ps_sky.healpix is not None

    def test_materialize_point_sources_requires_explicit_lossy_opt_in(self, precision):
        sky = make_healpix_model(precision=precision)
        with pytest.raises(ValueError, match="lossy"):
            sky.materialize_point_sources(frequency=100e6)

    def test_materialize_point_sources_from_healpix(self, precision):
        sky = make_healpix_model(precision=precision, include_pol=True)
        point = sky.materialize_point_sources(frequency=100e6, lossy=True)
        assert point.mode == SkyFormat.POINT_SOURCES
        assert point.point is not None
        assert point.healpix is not None
        assert point.n_point_sources > 0

    def test_materialize_point_sources_flux_limit_handles_optional_arrays(
        self, precision
    ):
        sky = make_healpix_model(precision=precision)
        point = sky.materialize_point_sources(
            frequency=100e6,
            flux_limit=1e6,
            lossy=True,
        )
        assert point.n_point_sources == 0


class TestFilteringAndAccessors:
    def test_filter_region_filters_both_payloads(self, test_sky, obs_freq_config):
        hp_sky = test_sky.materialize_healpix(
            nside=8, obs_frequency_config=obs_freq_config
        )
        region = SkyRegion.cone(
            np.rad2deg(test_sky.ra_rad[0]),
            np.rad2deg(test_sky.dec_rad[0]),
            2.0,
        )
        filtered = hp_sky.filter_region(region)
        assert filtered.point is not None
        assert filtered.healpix is not None
        assert filtered.n_point_sources <= hp_sky.n_point_sources
        assert np.count_nonzero(filtered.healpix_maps) <= np.count_nonzero(
            hp_sky.healpix_maps
        )

    def test_as_point_source_arrays_requires_point_payload(self, precision):
        sky = make_healpix_model(precision=precision)
        with pytest.raises(ValueError, match="materialize_point_sources"):
            sky.as_point_source_arrays()

    def test_frequency_accessors(self, precision):
        freqs = np.array([100e6, 102e6], dtype=np.float64)
        sky = make_healpix_model(freqs=freqs, precision=precision)
        idx = sky.resolve_frequency_index(101.9e6)
        assert idx == 1
        np.testing.assert_array_equal(
            sky.get_map_at_frequency(100e6), sky.healpix_maps[0]
        )
        maps, nside, returned_freqs = sky.get_multifreq_maps()
        assert nside == 8
        np.testing.assert_array_equal(maps, sky.healpix_maps)
        np.testing.assert_array_equal(returned_freqs, freqs)

    def test_pixel_helpers(self, precision):
        sky = make_healpix_model(nside=4, precision=precision)
        coords = sky.pixel_coords
        assert isinstance(coords, SkyCoord)
        assert len(coords) == hp.nside2npix(4)
        assert sky.pixel_solid_angle == pytest.approx(4 * np.pi / hp.nside2npix(4))

    def test_has_polarized_healpix_maps(self, precision):
        assert not make_healpix_model(precision=precision).has_polarized_healpix_maps
        assert make_healpix_model(
            precision=precision, include_pol=True
        ).has_polarized_healpix_maps


class TestMemmapAndEquality:
    def test_with_memmap_backing(self, precision, tmp_path):
        sky = make_healpix_model(precision=precision)
        mapped = sky.with_memmap_backing(str(tmp_path))
        assert isinstance(mapped.healpix_maps, np.memmap)
        assert (tmp_path / "i_maps.dat").exists()

    def test_repr_and_memory_estimate(self, precision):
        sky = make_healpix_model(precision=precision)
        rep = repr(sky)
        assert "nside=8" in rep
        estimate = estimate_healpix_memory(
            8, len(sky.observation_frequencies), np.float32
        )
        assert estimate["total_mb"] >= 0

    def test_value_equality_and_closeness(self, precision):
        sky_a = make_point_model(n=5, precision=precision, seed=1)
        sky_b = make_point_model(n=5, precision=precision, seed=1)
        sky_c = make_point_model(n=5, precision=precision, seed=2)
        assert sky_a == sky_b
        assert sky_a.is_close(sky_b)
        assert sky_a != sky_c
