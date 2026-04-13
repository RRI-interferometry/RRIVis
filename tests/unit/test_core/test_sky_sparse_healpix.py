"""Focused tests for sparse HEALPix support."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import astropy.units as u
import healpy as hp
import numpy as np
import pytest
from astropy.constants import c
from astropy.coordinates import EarthLocation
from astropy.time import Time
from matplotlib.figure import Figure
from pyradiosky import SkyModel as PyRadioSkyModel

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import HealpixData, SkyFormat, SkyModel, SkyPlotter
from rrivis.core.sky._loaders_pyradiosky import _load_pyradiosky_healpix
from rrivis.core.sky._serialization import to_pyradiosky
from rrivis.core.sky.operations import materialize_point_sources_model
from rrivis.core.visibility_healpix import calculate_visibility_healpix


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


def make_sparse_healpix_model(
    precision: PrecisionConfig,
) -> tuple[SkyModel, np.ndarray, np.ndarray]:
    nside = 8
    freqs = np.array([100e6], dtype=np.float64)
    hpx_inds = np.array([2, 17, 123], dtype=np.int64)
    maps = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    sky = SkyModel(
        healpix=HealpixData(
            maps=maps,
            nside=nside,
            frequencies=freqs,
            hpx_inds=hpx_inds,
        ),
        source_format=SkyFormat.HEALPIX,
        reference_frequency=float(freqs[0]),
        model_name="sparse-healpix",
        _precision=precision,
    )
    return sky, hpx_inds, freqs


def make_dense_equivalent(
    precision: PrecisionConfig,
) -> tuple[SkyModel, np.ndarray, np.ndarray]:
    sky, hpx_inds, freqs = make_sparse_healpix_model(precision)
    npix = hp.nside2npix(sky.healpix.nside)
    dense_maps = np.zeros((1, npix), dtype=np.float32)
    dense_maps[0, hpx_inds] = sky.healpix.maps[0]
    dense = SkyModel(
        healpix=HealpixData(
            maps=dense_maps,
            nside=sky.healpix.nside,
            frequencies=freqs,
        ),
        source_format=SkyFormat.HEALPIX,
        reference_frequency=float(freqs[0]),
        model_name="dense-healpix",
        _precision=precision,
    )
    return dense, hpx_inds, freqs


class TestSparseHealpixData:
    def test_masked_region_drops_sparse_pixels(self, precision):
        sky, hpx_inds, freqs = make_sparse_healpix_model(precision)
        mask = np.zeros(hp.nside2npix(sky.healpix.nside), dtype=bool)
        mask[hpx_inds[:2]] = True

        masked = sky.healpix.masked_region(mask)

        assert masked.is_sparse
        np.testing.assert_array_equal(masked.hpx_inds, hpx_inds[:2])
        assert masked.maps.shape == (len(freqs), 2)
        np.testing.assert_array_equal(masked.maps[0], sky.healpix.maps[0, :2])

    def test_to_dense_expands_sparse_maps(self, precision):
        sky, hpx_inds, _ = make_sparse_healpix_model(precision)
        dense = sky.healpix.to_dense()

        assert not dense.is_sparse
        assert dense.maps.shape[1] == hp.nside2npix(sky.healpix.nside)
        np.testing.assert_array_equal(dense.maps[0, hpx_inds], sky.healpix.maps[0])


class TestSparsePyradioskyLoader:
    def test_serialization_preserves_sparse_hpx_inds(self, precision):
        sky, hpx_inds, _ = make_sparse_healpix_model(precision)
        psky = to_pyradiosky(sky, representation=SkyFormat.HEALPIX)

        np.testing.assert_array_equal(psky.hpx_inds, hpx_inds)
        assert psky.stokes.shape == (4, 1, len(hpx_inds))
        np.testing.assert_array_equal(psky.stokes.value[0, 0], sky.healpix.maps[0])

    def test_loader_preserves_sparse_hpx_inds(self, precision):
        sky, hpx_inds, freqs = make_sparse_healpix_model(precision)
        stokes = np.zeros((4, 1, len(hpx_inds)), dtype=np.float64)
        stokes[0, 0] = sky.healpix.maps[0]
        psky = PyRadioSkyModel(
            nside=sky.healpix.nside,
            hpx_inds=hpx_inds,
            hpx_order="ring",
            stokes=stokes * u.K,
            spectral_type="flat",
            freq_array=freqs * u.Hz,
            component_type="healpix",
            frame="icrs",
            run_check=False,
            check_extra=False,
            run_check_acceptability=False,
        )

        loaded = _load_pyradiosky_healpix(
            psky,
            filename="sparse.skyh5",
            frequencies=freqs,
            obs_frequency_config=None,
            brightness_conversion="rayleigh-jeans",
            precision=precision,
        )

        assert loaded.healpix is not None
        assert loaded.healpix.is_sparse
        np.testing.assert_array_equal(loaded.healpix.hpx_inds, hpx_inds)
        assert loaded.healpix.maps.shape == (1, len(hpx_inds))
        np.testing.assert_array_equal(loaded.healpix.maps[0], sky.healpix.maps[0])


class TestSparseSkyModelBehavior:
    def test_sparse_pixel_coords_and_point_materialization(self, precision):
        sky, hpx_inds, freqs = make_sparse_healpix_model(precision)

        coords = sky.pixel_coords
        assert len(coords) == len(hpx_inds)

        point = materialize_point_sources_model(
            sky,
            frequency=float(freqs[0]),
            lossy=True,
        )
        assert point.point is not None
        assert point.n_point_sources > 0

    def test_sparse_plotter_smoke(self, precision):
        sky, _, _ = make_sparse_healpix_model(precision)
        assert isinstance(SkyPlotter(sky).healpix_map(), Figure)


class TestSparseVisibility:
    def test_sparse_visibility_matches_dense_equivalent(self, precision):
        sparse, _, freqs = make_sparse_healpix_model(precision)
        dense, _, _ = make_dense_equivalent(precision)

        antennas = {
            1: {"diameter": 14.0},
            2: {"diameter": 14.0},
        }
        baselines = {
            (1, 2): {"BaselineVector": np.array([0.0, 0.0, 0.0], dtype=np.float64)}
        }
        location = EarthLocation.from_geodetic(0.0 * u.deg, 0.0 * u.deg, 0.0 * u.m)
        obstime = Time("2024-01-01T00:00:00")
        wavelengths = np.array([c.value / freqs[0]], dtype=np.float64) * u.m

        sparse_vis = calculate_visibility_healpix(
            sparse,
            antennas=antennas,
            baselines=baselines,
            location=location,
            obstime=obstime,
            wavelengths=wavelengths,
            freqs=freqs,
            duration_seconds=1.0,
            time_step_seconds=1.0,
        )
        dense_vis = calculate_visibility_healpix(
            dense,
            antennas=antennas,
            baselines=baselines,
            location=location,
            obstime=obstime,
            wavelengths=wavelengths,
            freqs=freqs,
            duration_seconds=1.0,
            time_step_seconds=1.0,
        )

        np.testing.assert_allclose(
            sparse_vis["visibilities"], dense_vis["visibilities"]
        )
