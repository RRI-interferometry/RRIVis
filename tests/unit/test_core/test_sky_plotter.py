"""Smoke tests for sky-model plotting helpers."""

import matplotlib

matplotlib.use("Agg")

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import HealpixData
from rrivis.core.sky.model import SkyFormat, SkyModel


@pytest.fixture
def precision():
    return PrecisionConfig.standard()


@pytest.fixture
def test_sky(precision):
    return SkyModel.from_test_sources(
        num_sources=50,
        flux_range=(0.1, 10.0),
        dec_deg=-30.0,
        spectral_index=-0.7,
        precision=precision,
    )


@pytest.fixture
def healpix_sky(precision):
    nside = 8
    npix = hp.nside2npix(nside)
    freqs = np.array([100e6, 101e6], dtype=np.float64)
    maps = np.random.default_rng(42).uniform(10, 1000, (2, npix)).astype(np.float32)
    return SkyModel(
        healpix=HealpixData(maps=maps, nside=nside, frequencies=freqs),
        native_representation=SkyFormat.HEALPIX,
        active_representation=SkyFormat.HEALPIX,
        reference_frequency=100e6,
        model_name="healpix_test",
        _precision=precision,
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestPointPlots:
    def test_source_positions_returns_figure(self, test_sky):
        assert isinstance(test_sky.plot.source_positions(), Figure)

    @pytest.mark.parametrize(
        "projection", ["mollweide", "aitoff", "hammer", "cartesian"]
    )
    def test_source_positions_projection_variants(self, test_sky, projection):
        assert isinstance(test_sky.plot.source_positions(projection=projection), Figure)

    def test_flux_histogram_returns_figure(self, test_sky):
        assert isinstance(test_sky.plot.flux_histogram(), Figure)

    def test_spectral_index_sky_map_returns_figure(self, test_sky):
        assert isinstance(test_sky.plot.spectral_index(plot_type="sky_map"), Figure)


class TestHealpixPlots:
    def test_healpix_map_returns_figure(self, healpix_sky):
        assert isinstance(healpix_sky.plot.healpix_map(), Figure)

    def test_auto_dispatch(self, test_sky, healpix_sky):
        assert isinstance(test_sky.plot("auto"), Figure)
        assert isinstance(healpix_sky.plot("auto"), Figure)


class TestPlotterErrors:
    def test_invalid_color_by_raises(self, test_sky):
        with pytest.raises(ValueError, match="Unknown color_by"):
            test_sky.plot.source_positions(color_by="nonexistent")

    def test_invalid_projection_raises(self, test_sky):
        with pytest.raises(ValueError, match="Unknown projection"):
            test_sky.plot.source_positions(projection="lambert")

    def test_point_plot_on_healpix_raises(self, healpix_sky):
        with pytest.raises(ValueError, match="point-source data"):
            healpix_sky.plot.source_positions()

    def test_healpix_plot_on_point_model_raises(self, test_sky):
        with pytest.raises(ValueError, match="HEALPix maps"):
            test_sky.plot.healpix_map()

    def test_invalid_plot_type_raises(self, test_sky):
        with pytest.raises(ValueError, match="Unknown plot_type"):
            test_sky.plot("nonexistent_type")
