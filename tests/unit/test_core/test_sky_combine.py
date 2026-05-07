"""Tests for sky-model combination helpers."""

import healpy as hp
import numpy as np
import pytest

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import HealpixData, create_from_arrays
from rrivis.core.sky.combine import (
    _combine_models,
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

    def test_extra_columns_preserve_numeric_dtype(self, precision):
        sky_a = create_from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([0.2]),
            flux=np.array([1.0]),
            extra_columns={"snr": np.array([7.5], dtype=np.float64)},
            precision=precision,
        )
        sky_b = create_from_arrays(
            ra_rad=np.array([0.3]),
            dec_rad=np.array([0.4]),
            flux=np.array([2.0]),
            extra_columns={"snr": np.array([3.25], dtype=np.float64)},
            precision=precision,
        )

        data = concat_point_sources([sky_a, sky_b])
        snr = data["extra_columns"]["snr"]

        assert snr.dtype == np.float64
        np.testing.assert_array_equal(snr, np.array([7.5, 3.25]))

    def test_extra_columns_float_column_uses_nan_for_missing(self, precision):
        sky_a = create_from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([0.2]),
            flux=np.array([1.0]),
            extra_columns={"snr": np.array([7.5], dtype=np.float64)},
            precision=precision,
        )
        sky_b = create_from_arrays(
            ra_rad=np.array([0.3]),
            dec_rad=np.array([0.4]),
            flux=np.array([2.0]),
            precision=precision,
        )

        data = concat_point_sources([sky_a, sky_b])
        snr = data["extra_columns"]["snr"]

        assert np.issubdtype(snr.dtype, np.floating)
        assert snr[0] == 7.5
        assert np.isnan(snr[1])

    def test_extra_columns_integer_with_missing_falls_back_to_object(self, precision):
        sky_a = create_from_arrays(
            ra_rad=np.array([0.1]),
            dec_rad=np.array([0.2]),
            flux=np.array([1.0]),
            extra_columns={"catalog_id": np.array([42], dtype=np.int64)},
            precision=precision,
        )
        sky_b = create_from_arrays(
            ra_rad=np.array([0.3]),
            dec_rad=np.array([0.4]),
            flux=np.array([2.0]),
            precision=precision,
        )

        data = concat_point_sources([sky_a, sky_b])
        catalog_id = data["extra_columns"]["catalog_id"]

        assert catalog_id.dtype == object
        assert catalog_id[0] == 42
        assert catalog_id[1] is None


class TestCombineModels:
    def test_empty_list_returns_empty_sky(self, precision):
        sky = _combine_models([], precision=precision)
        assert sky.n_point_sources == 0

    def test_point_models_combine_as_point_sources(self, precision):
        sky_a = make_point_model(12, precision=precision, seed=10)
        sky_b = make_point_model(18, precision=precision, seed=20)
        result = _combine_models([sky_a, sky_b], precision=precision)
        assert result.formats == {SkyFormat.POINT_SOURCES}
        assert result.n_point_sources == 30

    def test_point_models_can_materialize_healpix_with_frequencies(self, precision):
        sky_a = make_point_model(10, precision=precision, seed=1)
        sky_b = make_point_model(10, precision=precision, seed=2)
        freqs = np.array([100e6, 101e6], dtype=np.float64)
        result = _combine_models(
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
            _combine_models(
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
            _combine_models(
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

    def test_regrid_sparse_healpix_model_requires_explicit_densify(self, precision):
        """Per the sparse-HEALPix doctrine, regrid raises on sparse input
        and the user must densify themselves; a follow-up dense call then
        succeeds and produces the expected grid."""
        sparse = make_sparse_healpix_model(precision=precision)
        with pytest.raises(ValueError, match="regrid_healpix_model"):
            regrid_healpix_model(sparse, nside=4)

        densified = sparse.replace(healpix=sparse.healpix.to_dense())
        regridded = regrid_healpix_model(densified, nside=4)
        assert regridded.healpix is not None
        assert regridded.healpix.nside == 4
        assert not regridded.healpix.is_sparse
        assert regridded.healpix.maps.shape == (2, hp.nside2npix(4))

    def test_healpix_only_to_point_sources_is_blocked_by_default(self, precision):
        sky = make_healpix_model(precision=precision)
        with pytest.raises(ValueError, match="allow_lossy_point_materialization=True"):
            _combine_models(
                [sky],
                representation=SkyFormat.POINT_SOURCES,
                frequency=100e6,
                precision=precision,
            )

    def test_healpix_only_to_point_sources_requires_opt_in(self, precision):
        sky = make_healpix_model(precision=precision)
        result = _combine_models(
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
            _combine_models(
                [point, diffuse],
                representation=SkyFormat.HEALPIX,
                precision=precision,
            )

    def test_mixed_catalog_and_diffuse_can_warn(self, precision):
        point = make_point_model(10, precision=precision, seed=1)
        diffuse = make_healpix_model(precision=precision)
        with pytest.warns(UserWarning, match="double-counting"):
            _combine_models(
                [point, diffuse],
                representation=SkyFormat.HEALPIX,
                mixed_model_policy="warn",
                precision=precision,
            )

    def test_healpix_nside_mismatch_raises(self, precision):
        sky_a = make_healpix_model(nside=8, precision=precision, value=100.0)
        sky_b = make_healpix_model(nside=16, precision=precision, value=50.0)
        with pytest.raises(ValueError, match="different nside"):
            _combine_models(
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
            _combine_models(
                [sky_a, sky_b],
                representation=SkyFormat.HEALPIX,
                precision=precision,
            )

    def test_healpix_coordinate_frame_mismatch_raises(self, precision):
        sky_a = make_healpix_model(precision=precision, coordinate_frame="icrs")
        sky_b = make_healpix_model(precision=precision, coordinate_frame="galactic")
        with pytest.raises(ValueError, match="coordinate_frame"):
            _combine_models(
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
        combined = _combine_models([sky_a, sky_b], precision=precision)
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
            _combine_models([sky_a, sky_b], precision=precision)

        combined = _combine_models(
            [sky_a, sky_b],
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
            precision=precision,
        )
        assert combined.brightness_conversion == BrightnessConversion.RAYLEIGH_JEANS

    def test_sparse_healpix_combination_accumulates_on_full_grid(self, precision):
        sparse = make_sparse_healpix_model(precision=precision, value=3.0)

        combined = _combine_models(
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

    def test_combine_two_sparse_healpix_overlapping_indices(self, precision):
        """Two sparse-HEALPix payloads with overlapping pixel sets accumulate
        on the full grid (RJ path so addition is exact)."""
        a = make_sparse_healpix_model(
            precision=precision,
            value=2.0,
            pixels=np.array([1, 5, 9]),
        )
        b = make_sparse_healpix_model(
            precision=precision,
            value=3.0,
            pixels=np.array([5, 9, 27]),
        )

        combined = _combine_models(
            [a, b],
            representation=SkyFormat.HEALPIX,
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
            precision=precision,
            mixed_model_policy="allow",
        )

        assert combined.healpix is not None
        full = combined.healpix.maps
        assert full.shape == (2, hp.nside2npix(8))
        # Index 1: only in a; index 5 and 9: a + b; index 27: only in b.
        np.testing.assert_allclose(full[:, 1], 2.0, atol=1e-6)
        np.testing.assert_allclose(full[:, 5], 5.0, atol=1e-6)
        np.testing.assert_allclose(full[:, 9], 5.0, atol=1e-6)
        np.testing.assert_allclose(full[:, 27], 3.0, atol=1e-6)

    def test_combine_sparse_with_dense_returns_dense(self, precision):
        """Combining a sparse and a dense payload returns a dense full-grid map."""
        sparse = make_sparse_healpix_model(
            precision=precision,
            value=2.0,
            pixels=np.array([1, 5]),
        )
        dense = make_healpix_model(precision=precision, value=4.0)

        combined = _combine_models(
            [sparse, dense],
            representation=SkyFormat.HEALPIX,
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
            precision=precision,
            mixed_model_policy="allow",
        )

        assert combined.healpix is not None
        assert combined.healpix.maps.shape == (2, hp.nside2npix(8))
        # Sparse pixels: dense (4) + sparse (2) = 6.
        np.testing.assert_allclose(combined.healpix.maps[:, 1], 6.0, atol=1e-6)
        np.testing.assert_allclose(combined.healpix.maps[:, 5], 6.0, atol=1e-6)
        # Non-sparse pixels: just dense.
        np.testing.assert_allclose(combined.healpix.maps[:, 0], 4.0, atol=1e-6)


class TestCombineMaterializeCommutativity:
    """combine→materialize should commute with materialize→combine on the
    Rayleigh-Jeans path, where brightness temperature is linearly additive.

    These act as integration smoke tests against future refactors of the
    combine arithmetic; if the RJ fast path drifts, this catches it before
    physics regressions show up downstream.
    """

    @pytest.mark.parametrize("seed_pair", [(1, 2), (7, 11), (42, 137)])
    def test_rj_combine_commutes_with_materialize(self, precision, seed_pair):
        seed_a, seed_b = seed_pair
        nside = 8
        freqs = np.array([100e6, 110e6], dtype=np.float64)

        sky_a = make_point_model(15, precision=precision, seed=seed_a)
        sky_b = make_point_model(20, precision=precision, seed=seed_b)

        # Path A: combine point sources, then materialize HEALPix once.
        combined_first = _combine_models(
            [sky_a, sky_b],
            representation=SkyFormat.HEALPIX,
            nside=nside,
            frequencies=freqs,
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
            precision=precision,
            mixed_model_policy="allow",
        )

        # Path B: materialize each model independently, then combine in HEALPix.
        materialized_a = _combine_models(
            [sky_a],
            representation=SkyFormat.HEALPIX,
            nside=nside,
            frequencies=freqs,
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
            precision=precision,
        )
        materialized_b = _combine_models(
            [sky_b],
            representation=SkyFormat.HEALPIX,
            nside=nside,
            frequencies=freqs,
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
            precision=precision,
        )
        combined_after = _combine_models(
            [materialized_a, materialized_b],
            representation=SkyFormat.HEALPIX,
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
            precision=precision,
            mixed_model_policy="allow",
        )

        assert combined_first.healpix is not None
        assert combined_after.healpix is not None
        np.testing.assert_allclose(
            combined_first.healpix.maps,
            combined_after.healpix.maps,
            rtol=1e-5,
            atol=1e-7,
            err_msg=("RJ combine→materialize should commute with materialize→combine"),
        )
