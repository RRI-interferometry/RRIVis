"""Tests for sky-model combination helpers."""

import healpy as hp
import numpy as np
import pytest

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import HealpixData, create_from_arrays
from rrivis.core.sky.combine import (
    combine_models,
    concat_point_sources,
    regrid_healpix_model,
)
from rrivis.core.sky.constants import BrightnessConversion
from rrivis.core.sky.model import SkyFormat, SkyModel


@pytest.fixture
def precision():
    return PrecisionConfig.standard()


def make_point_model(
    n: int,
    *,
    precision: PrecisionConfig,
    seed: int = 0,
    include_rm: bool = False,
) -> SkyModel:
    rng = np.random.default_rng(seed)
    return create_from_arrays(
        ra_rad=rng.uniform(0, 2 * np.pi, n),
        dec_rad=rng.uniform(-np.pi / 2, np.pi / 2, n),
        flux=rng.uniform(0.1, 10.0, n),
        spectral_index=np.full(n, -0.7),
        rotation_measure=rng.uniform(-10.0, 10.0, n) if include_rm else None,
        reference_frequency=150e6,
        model_name=f"point_{n}_{seed}",
        precision=precision,
    )


def make_healpix_model(
    *,
    nside: int = 8,
    freqs: np.ndarray | None = None,
    precision: PrecisionConfig,
    value: float = 100.0,
    coordinate_frame: str = "icrs",
) -> SkyModel:
    if freqs is None:
        freqs = np.array([100e6, 101e6], dtype=np.float64)
    npix = hp.nside2npix(nside)
    return SkyModel(
        healpix=HealpixData(
            maps=np.full((len(freqs), npix), value, dtype=np.float32),
            nside=nside,
            frequencies=freqs,
            coordinate_frame=coordinate_frame,
        ),
        source_format=SkyFormat.HEALPIX,
        reference_frequency=float(freqs[0]),
        model_name="diffuse",
        _precision=precision,
    )


def make_sparse_healpix_model(
    *,
    nside: int = 8,
    freqs: np.ndarray | None = None,
    precision: PrecisionConfig,
    pixels: np.ndarray | None = None,
    value: float = 100.0,
    coordinate_frame: str = "icrs",
) -> SkyModel:
    if freqs is None:
        freqs = np.array([100e6, 101e6], dtype=np.float64)
    if pixels is None:
        pixels = np.array([1, 9, 27], dtype=np.int64)
    maps = np.full((len(freqs), len(pixels)), value, dtype=np.float32)
    return SkyModel(
        healpix=HealpixData(
            maps=maps,
            nside=nside,
            frequencies=freqs,
            coordinate_frame=coordinate_frame,
            hpx_inds=pixels,
        ),
        source_format=SkyFormat.HEALPIX,
        reference_frequency=float(freqs[0]),
        model_name="sparse-diffuse",
        _precision=precision,
    )


class TestConcatPointSources:
    def test_concat_two_models(self, precision):
        sky_a = make_point_model(20, precision=precision, seed=1)
        sky_b = make_point_model(30, precision=precision, seed=2)
        data = concat_point_sources([sky_a, sky_b])
        assert len(data["ra_rad"]) == 50
        assert len(data["flux"]) == 50

    def test_concat_preserves_optional_fields(self, precision):
        sky_a = make_point_model(10, precision=precision, seed=1, include_rm=True)
        sky_b = make_point_model(15, precision=precision, seed=2, include_rm=True)
        data = concat_point_sources([sky_a, sky_b])
        assert data["rotation_measure"] is not None
        assert len(data["rotation_measure"]) == 25

    def test_concat_mixed_optional_fields_zero_fills(self, precision):
        sky_a = make_point_model(10, precision=precision, seed=1, include_rm=True)
        sky_b = make_point_model(5, precision=precision, seed=2, include_rm=False)
        data = concat_point_sources([sky_a, sky_b])
        np.testing.assert_array_equal(data["rotation_measure"][10:], 0.0)

    def test_concat_healpix_requires_explicit_lossy_opt_in(self, precision):
        sky = make_healpix_model(precision=precision)
        with pytest.raises(ValueError, match="allow_lossy_point_materialization=True"):
            concat_point_sources([sky], reference_frequency=100e6)

    def test_concat_healpix_allows_explicit_lossy_conversion(self, precision):
        sky = make_healpix_model(precision=precision)
        data = concat_point_sources(
            [sky],
            reference_frequency=100e6,
            allow_lossy_point_materialization=True,
        )
        assert len(data["ra_rad"]) > 0

    def test_concat_preserves_source_metadata(self, precision):
        sky_a = create_from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([0.2]),
            flux=np.array([1.0]),
            spectral_index=np.array([-0.7]),
            ref_freq=np.array([150e6]),
            source_name=np.array(["src-a"]),
            source_id=np.array(["A"]),
            extra_columns={"catalog": np.array(["gleam"])},
            precision=precision,
        )
        sky_b = create_from_arrays(
            ra_rad=np.array([0.3]),
            dec_rad=np.array([0.4]),
            flux=np.array([2.0]),
            spectral_index=np.array([-0.5]),
            ref_freq=np.array([150e6]),
            source_name=np.array(["src-b"]),
            precision=precision,
        )

        data = concat_point_sources([sky_a, sky_b])

        np.testing.assert_array_equal(data["source_name"], np.array(["src-a", "src-b"]))
        assert data["source_id"][0] == "A"
        assert data["source_id"][1] is None
        np.testing.assert_array_equal(
            data["extra_columns"]["catalog"],
            np.array(["gleam", None], dtype=object),
        )


class TestCombineModels:
    def test_empty_list_returns_empty_sky(self, precision):
        sky = combine_models([], precision=precision)
        assert sky.n_point_sources == 0

    def test_point_models_combine_as_point_sources(self, precision):
        sky_a = make_point_model(12, precision=precision, seed=10)
        sky_b = make_point_model(18, precision=precision, seed=20)
        result = combine_models([sky_a, sky_b], precision=precision)
        assert result.available_formats == {SkyFormat.POINT_SOURCES}
        assert result.n_point_sources == 30

    def test_point_models_can_materialize_healpix_with_frequencies(self, precision):
        sky_a = make_point_model(10, precision=precision, seed=1)
        sky_b = make_point_model(10, precision=precision, seed=2)
        freqs = np.array([100e6, 101e6], dtype=np.float64)
        result = combine_models(
            [sky_a, sky_b],
            representation=SkyFormat.HEALPIX,
            nside=8,
            frequencies=freqs,
            precision=precision,
        )
        assert result.healpix is not None
        np.testing.assert_array_equal(result.healpix.frequencies, freqs)

    def test_existing_healpix_nside_override_is_rejected(self, precision):
        sky = make_healpix_model(nside=8, precision=precision)
        with pytest.raises(ValueError, match="regrid_healpix_model"):
            combine_models(
                [sky],
                representation=SkyFormat.HEALPIX,
                nside=16,
                precision=precision,
            )

    def test_existing_healpix_frequency_override_is_rejected(self, precision):
        sky = make_healpix_model(
            freqs=np.array([100e6, 101e6], dtype=np.float64),
            precision=precision,
        )
        with pytest.raises(ValueError, match="frequency grid does not match"):
            combine_models(
                [sky],
                representation=SkyFormat.HEALPIX,
                frequencies=np.array([100e6, 102e6], dtype=np.float64),
                precision=precision,
            )

    def test_regrid_healpix_model_changes_nside_without_frequency_interpolation(
        self,
        precision,
    ):
        sky = make_healpix_model(
            nside=8,
            freqs=np.array([100e6, 101e6], dtype=np.float64),
            precision=precision,
            coordinate_frame="galactic",
        )
        regridded = regrid_healpix_model(
            sky,
            nside=4,
            frequencies=np.array([100e6, 101e6], dtype=np.float64),
        )
        assert regridded.healpix is not None
        assert regridded.healpix.nside == 4
        assert regridded.healpix.coordinate_frame == "galactic"
        assert regridded.healpix.maps.shape == (2, hp.nside2npix(4))
        np.testing.assert_array_equal(
            regridded.healpix.frequencies,
            np.array([100e6, 101e6], dtype=np.float64),
        )
        np.testing.assert_allclose(regridded.healpix.maps, 100.0)
        with pytest.raises(ValueError, match="Exact frequency regridding"):
            regrid_healpix_model(
                sky,
                nside=4,
                frequencies=np.array([100e6, 102e6], dtype=np.float64),
            )

    def test_regrid_sparse_healpix_model_densifies_explicitly(self, precision):
        sparse = make_sparse_healpix_model(precision=precision)

        regridded = regrid_healpix_model(sparse, nside=4)

        assert regridded.healpix is not None
        assert regridded.healpix.nside == 4
        assert not regridded.healpix.is_sparse
        assert regridded.healpix.maps.shape == (2, hp.nside2npix(4))

    def test_healpix_only_to_point_sources_is_blocked_by_default(self, precision):
        sky = make_healpix_model(precision=precision)
        with pytest.raises(ValueError, match="allow_lossy_point_materialization=True"):
            combine_models(
                [sky],
                representation=SkyFormat.POINT_SOURCES,
                frequency=100e6,
                precision=precision,
            )

    def test_healpix_only_to_point_sources_requires_opt_in(self, precision):
        sky = make_healpix_model(precision=precision)
        result = combine_models(
            [sky],
            representation=SkyFormat.POINT_SOURCES,
            frequency=100e6,
            allow_lossy_point_materialization=True,
            precision=precision,
        )
        assert result.point is not None
        assert result.n_point_sources > 0

    def test_mixed_catalog_and_diffuse_defaults_to_error(self, precision):
        point = make_point_model(10, precision=precision, seed=1)
        diffuse = make_healpix_model(precision=precision)
        with pytest.raises(ValueError, match="double-counting"):
            combine_models(
                [point, diffuse],
                representation=SkyFormat.HEALPIX,
                precision=precision,
            )

    def test_mixed_catalog_and_diffuse_can_warn(self, precision):
        point = make_point_model(10, precision=precision, seed=1)
        diffuse = make_healpix_model(precision=precision)
        with pytest.warns(UserWarning, match="double-counting"):
            combine_models(
                [point, diffuse],
                representation=SkyFormat.HEALPIX,
                mixed_model_policy="warn",
                precision=precision,
            )

    def test_healpix_nside_mismatch_raises(self, precision):
        sky_a = make_healpix_model(nside=8, precision=precision, value=100.0)
        sky_b = make_healpix_model(nside=16, precision=precision, value=50.0)
        with pytest.raises(ValueError, match="different nside"):
            combine_models(
                [sky_a, sky_b],
                representation=SkyFormat.HEALPIX,
                precision=precision,
            )

    def test_healpix_frequency_mismatch_raises(self, precision):
        sky_a = make_healpix_model(
            freqs=np.array([100e6, 101e6]),
            precision=precision,
        )
        sky_b = make_healpix_model(
            freqs=np.array([100e6, 102e6]),
            precision=precision,
        )
        with pytest.raises(ValueError, match="different frequency grids"):
            combine_models(
                [sky_a, sky_b],
                representation=SkyFormat.HEALPIX,
                precision=precision,
            )

    def test_healpix_coordinate_frame_mismatch_raises(self, precision):
        sky_a = make_healpix_model(precision=precision, coordinate_frame="icrs")
        sky_b = make_healpix_model(precision=precision, coordinate_frame="galactic")
        with pytest.raises(ValueError, match="coordinate_frame"):
            combine_models(
                [sky_a, sky_b],
                representation=SkyFormat.HEALPIX,
                precision=precision,
            )

    def test_per_source_reference_frequencies_are_preserved(self, precision):
        sky_a = create_from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([0.2]),
            flux=np.array([1.0]),
            spectral_index=np.array([-0.7]),
            ref_freq=np.array([200e6]),
            reference_frequency=200e6,
            precision=precision,
        )
        sky_b = create_from_arrays(
            ra_rad=np.array([0.3]),
            dec_rad=np.array([0.4]),
            flux=np.array([2.0]),
            spectral_index=np.array([-0.5]),
            ref_freq=np.array([1400e6]),
            reference_frequency=1400e6,
            precision=precision,
        )
        combined = combine_models([sky_a, sky_b], precision=precision)
        np.testing.assert_array_equal(
            combined.point.ref_freq, np.array([200e6, 1400e6])
        )

    def test_mixed_brightness_conversions_require_explicit_target(self, precision):
        sky_a = create_from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([0.2]),
            flux=np.array([1.0]),
            spectral_index=np.array([-0.7]),
            reference_frequency=150e6,
            brightness_conversion=BrightnessConversion.PLANCK,
            precision=precision,
        )
        sky_b = create_from_arrays(
            ra_rad=np.array([0.3]),
            dec_rad=np.array([0.4]),
            flux=np.array([2.0]),
            spectral_index=np.array([-0.5]),
            reference_frequency=150e6,
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
            precision=precision,
        )

        with pytest.raises(ValueError, match="brightness_conversion"):
            combine_models([sky_a, sky_b], precision=precision)

        combined = combine_models(
            [sky_a, sky_b],
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
            precision=precision,
        )
        assert combined.brightness_conversion == BrightnessConversion.RAYLEIGH_JEANS

    def test_sparse_healpix_combination_accumulates_on_full_grid(self, precision):
        sparse = make_sparse_healpix_model(precision=precision, value=3.0)

        combined = combine_models(
            [sparse],
            representation=SkyFormat.HEALPIX,
            precision=precision,
        )

        assert combined.healpix is not None
        assert combined.healpix.maps.shape == (2, hp.nside2npix(8))
        np.testing.assert_array_equal(
            combined.healpix.maps[:, sparse.healpix.hpx_inds],
            np.full((2, len(sparse.healpix.hpx_inds)), 3.0, dtype=np.float32),
        )
        assert combined.healpix.coordinate_frame == sparse.healpix.coordinate_frame
