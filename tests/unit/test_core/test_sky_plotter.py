"""Smoke tests for sky-model plotting helpers."""

import matplotlib

matplotlib.use("Agg")

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import HealpixData, SkyPlotter, create_test_sources
from radiosim.core.sky.containers.model import SkyModel


@pytest.fixture
def precision():
    return PrecisionConfig.standard()


@pytest.fixture
def test_sky(precision):
    return create_test_sources(
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
        reference_frequency=100e6,
        model_name="healpix_test",
        precision=precision,
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestPointPlots:
    def test_source_positions_returns_figure(self, test_sky):
        assert isinstance(SkyPlotter(test_sky).point.source_positions(), Figure)

    @pytest.mark.parametrize(
        "projection", ["mollweide", "aitoff", "hammer", "cartesian"]
    )
    def test_source_positions_projection_variants(self, test_sky, projection):
        assert isinstance(
            SkyPlotter(test_sky).point.source_positions(projection=projection),
            Figure,
        )

    def test_flux_histogram_returns_figure(self, test_sky):
        assert isinstance(SkyPlotter(test_sky).point.flux_histogram(), Figure)

    def test_spectral_index_sky_map_returns_figure(self, test_sky):
        assert isinstance(
            SkyPlotter(test_sky).point.spectral_index(plot_type="sky_map"),
            Figure,
        )


class TestHealpixPlots:
    def test_healpix_map_returns_figure(self, healpix_sky):
        assert isinstance(SkyPlotter(healpix_sky).healpix.healpix_map(), Figure)


class TestPlotterErrors:
    def test_invalid_color_by_raises(self, test_sky):
        with pytest.raises(ValueError, match="Unknown color_by"):
            SkyPlotter(test_sky).point.source_positions(color_by="nonexistent")

    def test_invalid_projection_raises(self, test_sky):
        with pytest.raises(ValueError, match="Unknown projection"):
            SkyPlotter(test_sky).point.source_positions(projection="lambert")

    def test_point_plot_on_healpix_raises(self, healpix_sky):
        with pytest.raises(ValueError, match="point-source data"):
            SkyPlotter(healpix_sky).point.source_positions()

    def test_healpix_plot_on_point_model_raises(self, test_sky):
        with pytest.raises(ValueError, match="HEALPix maps"):
            SkyPlotter(test_sky).healpix.healpix_map()

    def test_dispatcher_no_longer_callable(self, test_sky):
        """The string-dispatch __call__ has been removed in favour of
        explicit ``plotter.point.X`` / ``plotter.healpix.X`` / etc."""
        with pytest.raises(TypeError):
            SkyPlotter(test_sky)("auto")
