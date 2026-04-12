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
from rrivis.core.sky.model import SkyFormat  # noqa: E402

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
        _native_format=SkyFormat.HEALPIX,
        reference_frequency=100e6,
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
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
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
            _native_format=SkyFormat.HEALPIX,
            reference_frequency=100e6,
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


# ---------------------------------------------------------------------------
# Sky-cube analysis fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analysis_sky(precision):
    """HEALPix model with enough channels for spectral analysis (20 freqs)."""
    nside = 16
    npix = hp.nside2npix(nside)
    n_freq = 20
    freqs = np.linspace(75e6, 85e6, n_freq)
    rng = np.random.default_rng(7)
    # Near-Gaussian cube with small monopole
    maps = rng.normal(100.0, 10.0, (n_freq, npix)).astype(np.float32)
    return SkyModel(
        _healpix_maps=maps,
        _healpix_nside=nside,
        _observation_frequencies=freqs,
        _native_format=SkyFormat.HEALPIX,
        reference_frequency=80e6,
        model_name="analysis_test",
        _precision=precision,
    )


# ---------------------------------------------------------------------------
# pixel_histogram
# ---------------------------------------------------------------------------


class TestPixelHistogram:
    def test_returns_figure(self, analysis_sky):
        fig = analysis_sky.plot.pixel_histogram()
        assert isinstance(fig, Figure)

    def test_custom_bins(self, analysis_sky):
        fig = analysis_sky.plot.pixel_histogram(bins=50)
        assert isinstance(fig, Figure)

    def test_no_stats_no_fit(self, analysis_sky):
        fig = analysis_sky.plot.pixel_histogram(
            show_gaussian_fit=False, annotate_stats=False
        )
        assert isinstance(fig, Figure)

    def test_raises_for_point_sources(self, test_sky):
        with pytest.raises(ValueError, match="HEALPix"):
            test_sky.plot.pixel_histogram()


# ---------------------------------------------------------------------------
# variance_spectrum
# ---------------------------------------------------------------------------


class TestVarianceSpectrum:
    def test_returns_figure(self, analysis_sky):
        fig = analysis_sky.plot.variance_spectrum()
        assert isinstance(fig, Figure)

    def test_raises_for_single_frequency(self, healpix_sky):
        # healpix_sky has 2 frequencies — OK. Build one with 1 freq.
        nside = 8
        npix = hp.nside2npix(nside)
        sky = SkyModel(
            _healpix_maps=np.ones((1, npix), dtype=np.float32),
            _healpix_nside=nside,
            _observation_frequencies=np.array([100e6]),
            _native_format=SkyFormat.HEALPIX,
            model_name="single",
        )
        with pytest.raises(ValueError, match="at least 2"):
            sky.plot.variance_spectrum()


# ---------------------------------------------------------------------------
# frequency_spectra
# ---------------------------------------------------------------------------


class TestFrequencySpectra:
    def test_returns_figure(self, analysis_sky):
        fig = analysis_sky.plot.frequency_spectra(n_pixels=4)
        assert isinstance(fig, Figure)

    def test_custom_pixel_indices(self, analysis_sky):
        fig = analysis_sky.plot.frequency_spectra(pixel_indices=[0, 5, 10])
        assert isinstance(fig, Figure)

    def test_no_mean_panel(self, analysis_sky):
        fig = analysis_sky.plot.frequency_spectra(n_pixels=4, show_mean_spectrum=False)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# frequency_waterfall
# ---------------------------------------------------------------------------


class TestFrequencyWaterfall:
    def test_returns_figure(self, analysis_sky):
        fig = analysis_sky.plot.frequency_waterfall()
        assert isinstance(fig, Figure)

    def test_stride(self, analysis_sky):
        fig = analysis_sky.plot.frequency_waterfall(pixel_stride=10)
        assert isinstance(fig, Figure)

    def test_no_detrend(self, analysis_sky):
        fig = analysis_sky.plot.frequency_waterfall(detrend=False)
        assert isinstance(fig, Figure)

    def test_invalid_stride_raises(self, analysis_sky):
        with pytest.raises(ValueError, match="pixel_stride"):
            analysis_sky.plot.frequency_waterfall(pixel_stride=0)


# ---------------------------------------------------------------------------
# angular_power_spectrum
# ---------------------------------------------------------------------------


class TestAngularPowerSpectrum:
    @pytest.mark.parametrize("rep", ["c_ell", "d_ell", "both"])
    def test_returns_figure(self, analysis_sky, rep):
        fig = analysis_sky.plot.angular_power_spectrum(representation=rep)
        assert isinstance(fig, Figure)

    def test_explicit_frequencies(self, analysis_sky):
        fig = analysis_sky.plot.angular_power_spectrum(frequencies=[80e6, 82e6])
        assert isinstance(fig, Figure)

    def test_ell_max_reference(self, analysis_sky):
        fig = analysis_sky.plot.angular_power_spectrum(ell_max_input=30)
        assert isinstance(fig, Figure)

    def test_invalid_representation_raises(self, analysis_sky):
        with pytest.raises(ValueError, match="representation"):
            analysis_sky.plot.angular_power_spectrum(representation="bad")


# ---------------------------------------------------------------------------
# cross_frequency_cell
# ---------------------------------------------------------------------------


class TestCrossFrequencyCell:
    def test_returns_figure(self, analysis_sky):
        fig = analysis_sky.plot.cross_frequency_cell()
        assert isinstance(fig, Figure)

    def test_with_target_frequencies(self, analysis_sky):
        fig = analysis_sky.plot.cross_frequency_cell(
            ref_frequency=80e6,
            target_frequencies=[81e6, 83e6],
        )
        assert isinstance(fig, Figure)

    def test_separations(self, analysis_sky):
        fig = analysis_sky.plot.cross_frequency_cell(separations=[1, 2, 5])
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# multipole_bands
# ---------------------------------------------------------------------------


class TestMultipoleBands:
    def test_returns_figure(self, analysis_sky):
        fig = analysis_sky.plot.multipole_bands()
        assert isinstance(fig, Figure)

    def test_custom_bands(self, analysis_sky):
        fig = analysis_sky.plot.multipole_bands(bands=[(2, 10), (10, 20)])
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# frequency_correlation
# ---------------------------------------------------------------------------


class TestFrequencyCorrelation:
    def test_returns_figure(self, analysis_sky):
        fig = analysis_sky.plot.frequency_correlation()
        assert isinstance(fig, Figure)

    def test_no_decorrelation_panel(self, analysis_sky):
        fig = analysis_sky.plot.frequency_correlation(show_decorrelation=False)
        assert isinstance(fig, Figure)

    def test_invalid_stride_raises(self, analysis_sky):
        with pytest.raises(ValueError, match="pixel_stride"):
            analysis_sky.plot.frequency_correlation(pixel_stride=0)


# ---------------------------------------------------------------------------
# delay_spectrum
# ---------------------------------------------------------------------------


class TestDelaySpectrum:
    @pytest.mark.parametrize(
        "window", ["blackmanharris", "blackman", "hann", "hamming", "none"]
    )
    def test_windows(self, analysis_sky, window):
        fig = analysis_sky.plot.delay_spectrum(window=window)
        assert isinstance(fig, Figure)

    def test_no_kparallel(self, analysis_sky):
        fig = analysis_sky.plot.delay_spectrum(show_kparallel=False)
        assert isinstance(fig, Figure)

    def test_invalid_window_raises(self, analysis_sky):
        with pytest.raises(ValueError, match="window"):
            analysis_sky.plot.delay_spectrum(window="bogus")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestAnalysisDispatcher:
    @pytest.mark.parametrize(
        "plot_type",
        [
            "pdf",
            "histogram",
            "pixel_histogram",
            "cell",
            "power_spectrum",
            "angular_power_spectrum",
            "cross_cell",
            "cross_frequency_cell",
            "bands",
            "multipole_bands",
            "corr",
            "correlation",
            "frequency_correlation",
            "spectra",
            "frequency_spectra",
            "delay",
            "delay_spectrum",
            "variance",
            "variance_spectrum",
            "waterfall",
            "frequency_waterfall",
        ],
    )
    def test_dispatcher(self, analysis_sky, plot_type):
        fig = analysis_sky.plot(plot_type)
        assert isinstance(fig, Figure)
