"""Smoke tests for rrivis.core.sky.plotter — SkyPlotter visualization."""

import matplotlib

matplotlib.use("Agg")

import healpy as hp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from rrivis.core.precision import PrecisionConfig  # noqa: E402
from rrivis.core.sky import SkyModel  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def precision():
    return PrecisionConfig.standard()


@pytest.fixture
def test_sky(precision):
    """Point-source model with 50 sources."""
    return SkyModel.from_test_sources(
        num_sources=50,
        flux_range=(0.1, 10.0),
        dec_deg=-30.0,
        spectral_index=-0.7,
        precision=precision,
    )


@pytest.fixture
def healpix_sky(precision):
    """Minimal HEALPix model (nside=8, 2 frequency channels)."""
    nside = 8
    npix = hp.nside2npix(nside)
    freqs = np.array([100e6, 101e6])
    maps = np.random.default_rng(42).uniform(10, 1000, (2, npix)).astype(np.float32)
    return SkyModel(
        _healpix_maps=maps,
        _healpix_nside=nside,
        _observation_frequencies=freqs,
        _native_format="healpix",
        frequency=100e6,
        model_name="healpix_test",
        _precision=precision,
    )


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to free memory."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# source_positions
# ---------------------------------------------------------------------------


class TestSourcePositions:
    def test_source_positions_returns_figure(self, test_sky):
        """source_positions() returns a matplotlib Figure."""
        fig = test_sky.plot.source_positions()
        assert isinstance(fig, Figure)

    @pytest.mark.parametrize(
        "projection", ["mollweide", "aitoff", "hammer", "cartesian"]
    )
    def test_source_positions_projections(self, test_sky, projection):
        """source_positions works with all supported projections."""
        fig = test_sky.plot.source_positions(projection=projection)
        assert isinstance(fig, Figure)

    @pytest.mark.parametrize("color_by", ["flux", "spectral_index"])
    def test_source_positions_color_by(self, test_sky, color_by):
        """source_positions works with flux and spectral_index color modes."""
        fig = test_sky.plot.source_positions(color_by=color_by)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------


class TestHistograms:
    def test_flux_histogram_returns_figure(self, test_sky):
        """flux_histogram() returns a matplotlib Figure."""
        fig = test_sky.plot.flux_histogram()
        assert isinstance(fig, Figure)

    def test_spectral_index_histogram(self, test_sky):
        """spectral_index(plot_type='histogram') returns a Figure."""
        fig = test_sky.plot.spectral_index(plot_type="histogram")
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# HEALPix map
# ---------------------------------------------------------------------------


class TestHealpixMap:
    def test_healpix_map_returns_figure(self, healpix_sky):
        """healpix_map() returns a matplotlib Figure."""
        fig = healpix_sky.plot.healpix_map()
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Dispatcher (__call__)
# ---------------------------------------------------------------------------


class TestCallableDispatcher:
    def test_auto_dispatches_for_point_sources(self, test_sky):
        """sky.plot('auto') dispatches to source_positions for point sources."""
        fig = test_sky.plot("auto")
        assert isinstance(fig, Figure)

    def test_sources_alias(self, test_sky):
        """sky.plot('sources') dispatches to source_positions."""
        fig = test_sky.plot("sources")
        assert isinstance(fig, Figure)

    def test_auto_dispatches_for_healpix(self, healpix_sky):
        """sky.plot('auto') dispatches to healpix_map for healpix models."""
        fig = healpix_sky.plot("auto")
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestPlotterErrors:
    def test_invalid_color_by_raises(self, test_sky):
        """Unknown color_by value raises ValueError."""
        with pytest.raises(ValueError, match="Unknown color_by"):
            test_sky.plot.source_positions(color_by="nonexistent")

    def test_invalid_projection_raises(self, test_sky):
        """Unknown projection value raises ValueError."""
        with pytest.raises(ValueError, match="Unknown projection"):
            test_sky.plot.source_positions(projection="lambert")

    def test_point_source_plot_on_healpix_model_raises(self, healpix_sky):
        """Calling source_positions on a healpix-only model raises ValueError."""
        with pytest.raises(ValueError, match="point-source data"):
            healpix_sky.plot.source_positions()

    def test_healpix_plot_on_point_model_raises(self, test_sky):
        """Calling healpix_map on a point-source model raises ValueError."""
        with pytest.raises(ValueError, match="HEALPix maps"):
            test_sky.plot.healpix_map()

    def test_callable_invalid_raises(self, test_sky):
        """sky.plot('nonexistent_type') raises ValueError."""
        with pytest.raises(ValueError, match="Unknown plot_type"):
            test_sky.plot("nonexistent_type")


# ---------------------------------------------------------------------------
# Frequency label helper
# ---------------------------------------------------------------------------


class TestFreqLabel:
    def test_freq_label_ghz(self):
        """_freq_label(1.5e9) returns '1.5 GHz'."""
        from rrivis.core.sky.plotter import _freq_label

        assert _freq_label(1.5e9) == "1.5 GHz"


# ---------------------------------------------------------------------------
# Spectral index sky map
# ---------------------------------------------------------------------------


class TestSpectralIndexSkyMap:
    def test_spectral_index_sky_map(self, test_sky):
        """spectral_index(plot_type='sky_map') returns a Figure."""
        fig = test_sky.plot.spectral_index(plot_type="sky_map")
        assert isinstance(fig, Figure)

    def test_spectral_index_sky_map_cartesian(self, test_sky):
        """spectral_index(plot_type='sky_map', projection='cartesian') returns a Figure."""
        fig = test_sky.plot.spectral_index(plot_type="sky_map", projection="cartesian")
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# HEALPix map variants
# ---------------------------------------------------------------------------


class TestHealpixMapVariants:
    def test_healpix_map_linear_scale(self, healpix_sky):
        """healpix_map(log_scale=False) returns a Figure."""
        fig = healpix_sky.plot.healpix_map(log_scale=False)
        assert isinstance(fig, Figure)

    def test_healpix_map_stokes_q(self, precision):
        """healpix_map(stokes='Q') on a polarized model returns a Figure."""
        nside = 8
        npix = hp.nside2npix(nside)
        freqs = np.array([100e6, 101e6])
        rng = np.random.default_rng(42)
        i_maps = rng.uniform(10, 1000, (2, npix)).astype(np.float32)
        q_maps = rng.uniform(0, 1, (2, npix)).astype(np.float32) * 5 - 2.5
        u_maps = rng.uniform(0, 1, (2, npix)).astype(np.float32) * 5 - 2.5
        sky = SkyModel(
            _healpix_maps=i_maps,
            _healpix_q_maps=q_maps,
            _healpix_u_maps=u_maps,
            _healpix_nside=nside,
            _observation_frequencies=freqs,
            _native_format="healpix",
            frequency=100e6,
            model_name="test_pol",
            _precision=precision,
        )
        fig = sky.plot.healpix_map(stokes="Q")
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Multi-frequency grid
# ---------------------------------------------------------------------------


class TestMultifreqGrid:
    def test_multifreq_grid(self, healpix_sky):
        """multifreq_grid() returns a Figure with multiple panels."""
        fig = healpix_sky.plot.multifreq_grid()
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Stokes panel (2x2 IQUV)
# ---------------------------------------------------------------------------


class TestStokesPanel:
    def test_stokes_panel(self, precision):
        """stokes() returns a Figure (2x2 IQUV panel)."""
        nside = 8
        npix = hp.nside2npix(nside)
        freqs = np.array([100e6, 101e6])
        rng = np.random.default_rng(42)
        i_maps = rng.uniform(10, 1000, (2, npix)).astype(np.float32)
        q_maps = rng.uniform(0, 1, (2, npix)).astype(np.float32) * 5 - 2.5
        u_maps = rng.uniform(0, 1, (2, npix)).astype(np.float32) * 5 - 2.5
        sky = SkyModel(
            _healpix_maps=i_maps,
            _healpix_q_maps=q_maps,
            _healpix_u_maps=u_maps,
            _healpix_nside=nside,
            _observation_frequencies=freqs,
            _native_format="healpix",
            frequency=100e6,
            model_name="test_pol",
            _precision=precision,
        )
        fig = sky.plot.stokes()
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Coordinate rotation and color options
# ---------------------------------------------------------------------------


class TestCoordinateAndColorOptions:
    def test_coordinate_rotation_galactic(self, test_sky):
        """source_positions(coord='G') returns a Figure (Galactic rotation)."""
        fig = test_sky.plot.source_positions(coord="G")
        assert isinstance(fig, Figure)

    def test_source_positions_log_color_false(self, test_sky):
        """source_positions(log_color=False) returns a Figure (linear color scale)."""
        fig = test_sky.plot.source_positions(log_color=False)
        assert isinstance(fig, Figure)
