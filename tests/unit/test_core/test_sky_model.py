"""Tests for the public SkyModel API."""

import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import (
    HealpixData,
    PointSourceData,
    SkyRegion,
    create_from_arrays,
    create_test_sources,
    materialize_healpix_model,
    materialize_point_sources_model,
    with_memmap_backing,
)
from rrivis.core.sky.discovery import estimate_healpix_memory
from rrivis.core.sky.loaders import load_test_sources
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
    return create_from_arrays(
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
        source_format=SkyFormat.HEALPIX,
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

    def test_invalid_source_format_raises(self, precision):
        with pytest.raises(ValueError, match="Unknown representation"):
            SkyModel(
                point=PointSourceData.empty(),
                source_format="invalid",
                _precision=precision,
            )

    def test_precision_is_required(self):
        with pytest.raises(ValueError, match="explicit PrecisionConfig"):
            SkyModel(point=PointSourceData.empty())

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

    def test_point_metadata_validation_is_strict(self):
        with pytest.raises(ValueError, match="extra column 'catalog' has length 4"):
            PointSourceData(
                ra_rad=np.zeros(5),
                dec_rad=np.zeros(5),
                flux=np.ones(5),
                spectral_index=np.zeros(5),
                stokes_q=np.zeros(5),
                stokes_u=np.zeros(5),
                stokes_v=np.zeros(5),
                ref_freq=np.full(5, 100e6),
                extra_columns={"catalog": np.array(["a", "b", "c", "d"])},
            )

    def test_healpix_requires_matching_frequency_axis(self):
        with pytest.raises(ValueError, match="channels"):
            HealpixData(
                maps=np.ones((2, hp.nside2npix(8)), dtype=np.float32),
                nside=8,
                frequencies=np.array([100e6]),
            )


class TestBasicBehavior:
    def test_no_active_mode_surface(self, test_sky):
        assert not hasattr(test_sky, "mode")
        assert not hasattr(test_sky, "native_format")
        assert not hasattr(test_sky, "active_representation")

    def test_create_test_sources_basic(self, precision):
        sky = create_test_sources(
            num_sources=20,
            flux_range=(1.0, 5.0),
            dec_deg=-30.0,
            spectral_index=-0.8,
            precision=precision,
        )
        assert sky.source_format == SkyFormat.POINT_SOURCES
        assert sky.available_formats == {SkyFormat.POINT_SOURCES}
        assert sky.n_point_sources == 20
        assert np.allclose(sky.point.spectral_index, -0.8)

    def test_healpix_carries_per_stokes_unit_metadata(self, precision):
        sky = make_healpix_model(precision=precision, include_pol=True)
        assert sky.healpix.i_unit == "K"
        assert sky.healpix.q_unit == "K"
        assert sky.healpix.i_brightness_conversion == "planck"
        assert sky.healpix.q_brightness_conversion == "rayleigh-jeans"

    def test_typed_loader_replaces_from_catalog(self, precision):
        sky = load_test_sources(
            num_sources=4,
            reference_frequency=100e6,
            precision=precision,
        )
        assert sky.n_point_sources == 4

    def test_removed_compatibility_wrappers(self):
        assert not hasattr(SkyModel, "with_healpix_maps")
        assert not hasattr(SkyModel, "with_representation")
        assert not hasattr(SkyModel, "materialize_healpix")
        assert not hasattr(SkyModel, "materialize_point_sources")
        assert not hasattr(SkyModel, "with_memmap_backing")
        assert not hasattr(SkyModel, "plot")
        assert not hasattr(SkyModel, "save")

    def test_functional_operations_surface(self, precision, tmp_path):
        point = make_point_model(n=4, precision=precision)
        healpix = materialize_healpix_model(
            point,
            nside=8,
            frequencies=np.array([100e6]),
        )
        recovered = materialize_point_sources_model(
            make_healpix_model(precision=precision),
            frequency=100e6,
            lossy=True,
        )
        mapped = with_memmap_backing(healpix, str(tmp_path))

        assert healpix.healpix is not None
        assert recovered.point is not None
        assert isinstance(mapped.healpix.maps, np.memmap)


class TestMaterialization:
    def test_materialize_healpix_preserves_point_payload(
        self, test_sky, obs_freq_config
    ):
        sky = materialize_healpix_model(
            test_sky, nside=16, obs_frequency_config=obs_freq_config
        )
        assert sky is not test_sky
        assert sky.point is not None
        assert sky.healpix is not None
        assert sky.available_formats == {SkyFormat.POINT_SOURCES, SkyFormat.HEALPIX}
        assert sky.n_pixels == hp.nside2npix(16)

    def test_materialize_healpix_requires_reference_frequency(self, precision):
        sky = make_point_model(n=5, precision=precision, reference_frequency=None)
        with pytest.raises(ValueError, match="ref_frequency must be provided"):
            materialize_healpix_model(sky, nside=16, frequencies=np.array([100e6]))

    def test_materialize_healpix_uses_per_source_reference_frequencies(self, precision):
        sky = create_from_arrays(
            ra_rad=np.array([0.0, 1.0]),
            dec_rad=np.array([0.1, -0.1]),
            flux=np.array([1.0, 2.0]),
            spectral_index=np.array([0.0, -0.7]),
            ref_freq=np.array([100e6, 200e6]),
            reference_frequency=None,
            precision=precision,
        )

        hp_sky = materialize_healpix_model(sky, nside=8, frequencies=np.array([100e6]))

        assert hp_sky.healpix is not None
        assert hp_sky.healpix.maps.shape == (1, hp.nside2npix(8))

    def test_materialize_healpix_rejects_conflicting_frequency_inputs(self, test_sky):
        with pytest.raises(
            ValueError, match="either 'frequencies' or 'obs_frequency_config'"
        ):
            materialize_healpix_model(
                test_sky,
                nside=8,
                frequencies=np.array([100e6]),
                obs_frequency_config={"starting_frequency": 100.0},
            )

    def test_counts_require_explicit_representation(self, test_sky, obs_freq_config):
        hp_sky = materialize_healpix_model(
            test_sky, nside=8, obs_frequency_config=obs_freq_config
        )
        with pytest.raises(ValueError, match="ambiguous"):
            _ = hp_sky.n_sky_elements
        assert hp_sky.n_sky_elements_for("point_sources") == test_sky.n_point_sources
        assert hp_sky.n_sky_elements_for("healpix_map") == hp_sky.n_pixels

    def test_materialize_point_sources_requires_explicit_lossy_opt_in(self, precision):
        sky = make_healpix_model(precision=precision)
        with pytest.raises(ValueError, match="lossy"):
            materialize_point_sources_model(sky, frequency=100e6)

    def test_materialize_point_sources_from_healpix(self, precision):
        sky = make_healpix_model(precision=precision, include_pol=True)
        point = materialize_point_sources_model(sky, frequency=100e6, lossy=True)
        assert point.available_formats == {SkyFormat.POINT_SOURCES, SkyFormat.HEALPIX}
        assert point.point is not None
        assert point.healpix is not None
        assert point.n_point_sources > 0

    def test_materialize_point_sources_flux_limit_handles_optional_arrays(
        self, precision
    ):
        sky = make_healpix_model(precision=precision)
        point = materialize_point_sources_model(
            sky,
            frequency=100e6,
            flux_limit=1e6,
            lossy=True,
        )
        assert point.n_point_sources == 0


class TestFilteringAndAccessors:
    def test_filter_region_filters_both_payloads(self, test_sky, obs_freq_config):
        hp_sky = materialize_healpix_model(
            test_sky, nside=8, obs_frequency_config=obs_freq_config
        )
        region = SkyRegion.cone(
            np.rad2deg(test_sky.point.ra_rad[0]),
            np.rad2deg(test_sky.point.dec_rad[0]),
            2.0,
        )
        filtered = hp_sky.filter_region(region)
        assert filtered.point is not None
        assert filtered.healpix is not None
        assert filtered.n_point_sources <= hp_sky.n_point_sources
        assert np.count_nonzero(filtered.healpix.maps) <= np.count_nonzero(
            hp_sky.healpix.maps
        )

    def test_as_point_source_arrays_requires_point_payload(self, precision):
        sky = make_healpix_model(precision=precision)
        with pytest.raises(ValueError, match="materialize_point_sources_model"):
            sky.as_point_source_arrays()

    def test_frequency_accessors(self, precision):
        freqs = np.array([100e6, 102e6], dtype=np.float64)
        sky = make_healpix_model(freqs=freqs, precision=precision)
        idx = sky.resolve_frequency_index(101.9e6)
        assert idx == 1
        np.testing.assert_array_equal(
            sky.get_map_at_frequency(100e6), sky.healpix.maps[0]
        )
        maps, nside, returned_freqs = sky.get_multifreq_maps()
        assert nside == 8
        np.testing.assert_array_equal(maps, sky.healpix.maps)
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
        mapped = with_memmap_backing(sky, str(tmp_path))
        assert isinstance(mapped.healpix.maps, np.memmap)
        assert (tmp_path / "i_maps.dat").exists()

    def test_repr_and_memory_estimate(self, precision):
        sky = make_healpix_model(precision=precision)
        rep = repr(sky)
        assert "nside=8" in rep
        estimate = estimate_healpix_memory(8, len(sky.healpix.frequencies), np.float32)
        assert estimate["total_mb"] >= 0

    def test_value_equality_and_closeness(self, precision):
        sky_a = make_point_model(n=5, precision=precision, seed=1)
        sky_b = make_point_model(n=5, precision=precision, seed=1)
        sky_c = make_point_model(n=5, precision=precision, seed=2)
        assert sky_a == sky_b
        assert sky_a.is_close(sky_b)
        assert sky_a != sky_c
