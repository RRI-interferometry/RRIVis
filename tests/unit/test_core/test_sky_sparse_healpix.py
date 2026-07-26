"""Focused tests for sparse HEALPix support."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import astropy.units as u
import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time
from matplotlib.figure import Figure
from pyradiosky import SkyModel as PyRadioSkyModel

from radiosim.api import Simulator
from radiosim.core.instrument_adapters import SolverInstrumentView
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import HealpixData, SkyFormat, SkyModel
from radiosim.core.sky.io.serialization import to_pyradiosky
from radiosim.core.sky.loaders.pyradiosky import _load_pyradiosky_healpix
from radiosim.core.sky.operations.operations import materialize_point_sources_model
from radiosim.core.time_grid import build_observation_time_grid
from radiosim.core.visibility_healpix import calculate_visibility_healpix
from radiosim.visualization import plot_healpix_map
from tests.fixtures.configs import valid_config_mapping


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


def make_sparse_healpix_model(
    precision: PrecisionConfig,
    *,
    coordinate_frame: str = "icrs",
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
            coordinate_frame=coordinate_frame,
            hpx_inds=hpx_inds,
        ),
        reference_frequency=float(freqs[0]),
        model_name="sparse-healpix",
        precision=precision,
    )
    return sky, hpx_inds, freqs


def make_dense_equivalent(
    precision: PrecisionConfig,
    *,
    coordinate_frame: str = "icrs",
) -> tuple[SkyModel, np.ndarray, np.ndarray]:
    sky, hpx_inds, freqs = make_sparse_healpix_model(
        precision,
        coordinate_frame=coordinate_frame,
    )
    npix = hp.nside2npix(sky.healpix.nside)
    dense_maps = np.zeros((1, npix), dtype=np.float32)
    dense_maps[0, hpx_inds] = sky.healpix.maps[0]
    dense = SkyModel(
        healpix=HealpixData(
            maps=dense_maps,
            nside=sky.healpix.nside,
            frequencies=freqs,
            coordinate_frame=coordinate_frame,
        ),
        reference_frequency=float(freqs[0]),
        model_name="dense-healpix",
        precision=precision,
    )
    return dense, hpx_inds, freqs


class TestSparseHealpixData:
    def test_cropped_to_mask_drops_sparse_pixels(self, precision):
        sky, hpx_inds, freqs = make_sparse_healpix_model(precision)
        mask = np.zeros(hp.nside2npix(sky.healpix.nside), dtype=bool)
        mask[hpx_inds[:2]] = True

        cropped = sky.healpix.cropped_to_mask(mask)

        assert cropped.is_sparse
        np.testing.assert_array_equal(cropped.hpx_inds, hpx_inds[:2])
        assert cropped.maps.shape == (len(freqs), 2)
        np.testing.assert_array_equal(cropped.maps[0], sky.healpix.maps[0, :2])

    def test_cropped_to_mask_on_dense_returns_sparse(self, precision):
        nside = 8
        npix = hp.nside2npix(nside)
        freqs = np.array([100e6, 101e6])
        from radiosim.core.sky import HealpixData

        dense = HealpixData(
            maps=np.arange(len(freqs) * npix, dtype=np.float32).reshape(
                len(freqs), npix
            ),
            nside=nside,
            frequencies=freqs,
        )
        mask = np.zeros(npix, dtype=bool)
        kept = np.array([0, 5, 11], dtype=np.int64)
        mask[kept] = True

        cropped = dense.cropped_to_mask(mask)
        assert cropped.is_sparse
        np.testing.assert_array_equal(cropped.hpx_inds, kept)
        assert cropped.maps.shape == (len(freqs), kept.size)
        np.testing.assert_array_equal(cropped.maps[0], dense.maps[0, kept])

    def test_zero_outside_mask_requires_dense(self, precision):
        sky, hpx_inds, _ = make_sparse_healpix_model(precision)
        mask = np.zeros(hp.nside2npix(sky.healpix.nside), dtype=bool)
        mask[hpx_inds[:2]] = True
        with pytest.raises(ValueError, match="dense HEALPix cube"):
            sky.healpix.zero_outside_mask(mask)

    def test_zero_outside_mask_zeroes_pixels(self, precision):
        nside = 8
        npix = hp.nside2npix(nside)
        freqs = np.array([100e6])
        from radiosim.core.sky import HealpixData

        dense = HealpixData(
            maps=np.full((1, npix), 7.0, dtype=np.float32),
            nside=nside,
            frequencies=freqs,
        )
        mask = np.zeros(npix, dtype=bool)
        mask[[3, 4]] = True

        zeroed = dense.zero_outside_mask(mask)
        assert not zeroed.is_sparse
        assert zeroed.maps.shape == dense.maps.shape
        assert zeroed.maps[0, 3] == 7.0
        assert zeroed.maps[0, 4] == 7.0
        assert zeroed.maps[0, 0] == 0.0

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
            brightness_conversion="rayleigh-jeans",
            precision=precision,
        )

        assert loaded.healpix is not None
        assert loaded.healpix.is_sparse
        np.testing.assert_array_equal(loaded.healpix.hpx_inds, hpx_inds)
        assert loaded.healpix.maps.shape == (1, len(hpx_inds))
        np.testing.assert_array_equal(loaded.healpix.maps[0], sky.healpix.maps[0])

    def test_galactic_sparse_round_trip_preserves_frame(self, precision):
        sky, hpx_inds, freqs = make_sparse_healpix_model(
            precision,
            coordinate_frame="galactic",
        )
        psky = to_pyradiosky(sky, representation=SkyFormat.HEALPIX)

        assert "galactic" in str(psky.frame).lower()
        np.testing.assert_array_equal(psky.hpx_inds, hpx_inds)

        loaded = _load_pyradiosky_healpix(
            psky,
            filename="galactic_sparse.skyh5",
            frequencies=freqs,
            brightness_conversion="rayleigh-jeans",
            precision=precision,
        )

        assert loaded.healpix is not None
        assert loaded.healpix.coordinate_frame == "galactic"
        np.testing.assert_array_equal(loaded.healpix.hpx_inds, hpx_inds)
        np.testing.assert_array_equal(loaded.healpix.maps, sky.healpix.maps)


class TestSparseSkyModelBehavior:
    def test_sparse_pixel_coords_and_point_materialization(self, precision):
        sky, hpx_inds, freqs = make_sparse_healpix_model(
            precision,
            coordinate_frame="galactic",
        )

        coords = sky.healpix.pixel_coords
        assert len(coords) == len(hpx_inds)
        assert coords.frame.name == "galactic"

        point = materialize_point_sources_model(
            sky,
            frequency=float(freqs[0]),
            lossy=True,
        )
        assert point.point is not None
        assert point.n_point_sources > 0

    def test_sparse_plotter_requires_explicit_densify(self, precision):
        """Per the sparse-HEALPix doctrine, plotting raises on sparse input
        and the user must densify themselves; a dense follow-up succeeds."""
        sky, _, _ = make_sparse_healpix_model(precision)
        with pytest.raises(ValueError, match="plotter"):
            plot_healpix_map(sky)

        dense_sky = sky.replace(healpix=sky.healpix.to_dense())
        assert isinstance(plot_healpix_map(dense_sky), Figure)


class TestSparseVisibility:
    def test_sparse_visibility_matches_dense_equivalent(self, precision, tmp_path):
        sparse, _, freqs = make_sparse_healpix_model(precision)
        dense, _, _ = make_dense_equivalent(precision)

        data = valid_config_mapping(
            tmp_path,
            baseline_selection={"correlations": "cross"},
        )
        simulator = Simulator.from_mapping(data, base_dir=tmp_path)
        simulator._ensure_instrument_state()
        simulator._ensure_beam_system()
        instrument = SolverInstrumentView.from_state(simulator._instrument_state)
        location = EarthLocation.from_geodetic(0.0 * u.deg, 0.0 * u.deg, 0.0 * u.m)
        obstime = Time("2024-01-01T00:00:00")
        time_grid = build_observation_time_grid(
            start_time=obstime.isot,
            duration_seconds=1.0,
            cadence_seconds=1.0,
        )

        sparse_vis = calculate_visibility_healpix(
            sparse,
            instrument=instrument,
            beam_system=simulator.beam_system,
            location=location,
            time_grid=time_grid,
            frequencies=freqs,
        )
        dense_vis = calculate_visibility_healpix(
            dense,
            instrument=instrument,
            beam_system=simulator.beam_system,
            location=location,
            time_grid=time_grid,
            frequencies=freqs,
        )

        np.testing.assert_allclose(
            sparse_vis,
            dense_vis,
        )
