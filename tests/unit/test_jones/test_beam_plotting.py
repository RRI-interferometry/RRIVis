"""Tests for analytic beam pattern plotting functions.

Uses the Agg backend to avoid display requirements.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


class TestPlotBeamPattern:
    """Tests for plot_beam_pattern."""

    def test_returns_figure(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_pattern

        fig = plot_beam_pattern(diameter=14.0, frequency=150e6)
        assert isinstance(fig, Figure)

    def test_all_tapers(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_pattern

        tapers = ["uniform", "gaussian", "parabolic", "parabolic_squared", "cosine"]
        for taper in tapers:
            fig = plot_beam_pattern(
                diameter=14.0, frequency=150e6, taper=taper, n_points=100
            )
            assert isinstance(fig, Figure)

    def test_all_aperture_shapes(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_pattern

        # Circular
        fig = plot_beam_pattern(
            diameter=14.0,
            frequency=150e6,
            aperture_shape="circular",
            n_points=100,
        )
        assert isinstance(fig, Figure)

        # Rectangular
        fig = plot_beam_pattern(
            diameter=14.0,
            frequency=150e6,
            aperture_shape="rectangular",
            aperture_params={"length_x": 14.0, "length_y": 10.0},
            n_points=100,
        )
        assert isinstance(fig, Figure)

        # Elliptical
        fig = plot_beam_pattern(
            diameter=14.0,
            frequency=150e6,
            aperture_shape="elliptical",
            aperture_params={"diameter_x": 14.0, "diameter_y": 10.0},
            n_points=100,
        )
        assert isinstance(fig, Figure)

    def test_show_voltage(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_pattern

        fig = plot_beam_pattern(
            diameter=14.0, frequency=150e6, show_voltage=True, n_points=100
        )
        assert isinstance(fig, Figure)
        # Should have two y-axes (twinx creates a second)
        axes = fig.get_axes()
        assert len(axes) == 2

    def test_hpbw_annotation(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_pattern

        fig = plot_beam_pattern(
            diameter=14.0, frequency=150e6, show_hpbw=True, n_points=200
        )
        ax = fig.get_axes()[0]
        # Check that an annotation containing "HPBW" exists
        texts = [t.get_text() for t in ax.texts]
        assert any("HPBW" in t for t in texts)

    def test_custom_ax(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_pattern

        fig_ext, ax_ext = plt.subplots()
        fig = plot_beam_pattern(diameter=14.0, frequency=150e6, ax=ax_ext, n_points=100)
        # Should return the same figure that owns ax_ext
        assert fig is fig_ext

    def test_custom_title(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_pattern

        fig = plot_beam_pattern(
            diameter=14.0, frequency=150e6, title="Custom Title", n_points=100
        )
        ax = fig.get_axes()[0]
        assert ax.get_title() == "Custom Title"

    def test_no_hpbw_no_null_no_sidelobe(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_pattern

        fig = plot_beam_pattern(
            diameter=14.0,
            frequency=150e6,
            show_hpbw=False,
            show_first_null=False,
            show_sidelobe_level=False,
            n_points=100,
        )
        assert isinstance(fig, Figure)


class TestPlotBeamComparison:
    """Tests for plot_beam_comparison."""

    def test_returns_figure(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_comparison

        configs = [
            {"taper": "uniform", "label": "Uniform"},
            {"taper": "gaussian", "label": "Gaussian"},
            {"taper": "parabolic", "label": "Parabolic"},
        ]
        fig = plot_beam_comparison(14.0, 150e6, configs, n_points=100)
        assert isinstance(fig, Figure)

    def test_legend_present(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_comparison

        configs = [
            {"taper": "uniform", "label": "Uniform"},
            {"taper": "gaussian", "label": "Gaussian"},
        ]
        fig = plot_beam_comparison(14.0, 150e6, configs, n_points=100)
        ax = fig.get_axes()[0]
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "Uniform" in legend_texts
        assert "Gaussian" in legend_texts

    def test_per_config_overrides(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_comparison

        configs = [
            {"taper": "gaussian", "edge_taper_dB": 5.0, "label": "5 dB"},
            {"taper": "gaussian", "edge_taper_dB": 15.0, "label": "15 dB"},
            {"diameter": 25.0, "label": "25m dish"},
        ]
        fig = plot_beam_comparison(14.0, 150e6, configs, n_points=100)
        assert isinstance(fig, Figure)


class TestPlotBeam2D:
    """Tests for plot_beam_2d."""

    def test_returns_figure(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_2d

        fig = plot_beam_2d(
            diameter=14.0, frequency=150e6, n_points=51, max_angle_deg=3.0
        )
        assert isinstance(fig, Figure)

    def test_elliptical(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_2d

        fig = plot_beam_2d(
            diameter=14.0,
            frequency=150e6,
            aperture_shape="elliptical",
            aperture_params={"diameter_x": 14.0, "diameter_y": 8.0},
            n_points=51,
            max_angle_deg=3.0,
        )
        assert isinstance(fig, Figure)

    def test_has_colorbar(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_2d

        fig = plot_beam_2d(
            diameter=14.0, frequency=150e6, n_points=51, max_angle_deg=3.0
        )
        # pcolormesh + colorbar creates 2 axes
        assert len(fig.get_axes()) == 2


class TestPlotFeedIllumination:
    """Tests for plot_feed_illumination."""

    def test_corrugated_horn(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_feed_illumination

        fig = plot_feed_illumination(feed_model="corrugated_horn")
        assert isinstance(fig, Figure)

    def test_open_waveguide(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_feed_illumination

        fig = plot_feed_illumination(feed_model="open_waveguide")
        assert isinstance(fig, Figure)

    def test_dipole_ground_plane(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_feed_illumination

        fig = plot_feed_illumination(feed_model="dipole_ground_plane")
        assert isinstance(fig, Figure)

    def test_cassegrain(self):
        from rrivis.core.jones.beam.analytic.plotting import plot_feed_illumination

        fig = plot_feed_illumination(
            feed_model="corrugated_horn",
            reflector_type="cassegrain",
            magnification=5.0,
        )
        assert isinstance(fig, Figure)


class TestAnalyticBeamJonesPlot:
    """Tests for AnalyticBeamJones.plot() method."""

    def test_plot_method_returns_figure(self):
        from rrivis.core.jones.beam.analytic import AnalyticBeamJones

        source_altaz = np.array([[np.pi / 2, 0.0]])
        frequencies = np.array([150e6])

        jones = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
            taper="gaussian",
        )

        fig = jones.plot(n_points=100)
        assert isinstance(fig, Figure)

    def test_plot_method_custom_frequency(self):
        from rrivis.core.jones.beam.analytic import AnalyticBeamJones

        source_altaz = np.array([[np.pi / 2, 0.0]])
        frequencies = np.array([150e6, 200e6])

        jones = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
        )

        fig = jones.plot(frequency=200e6, n_points=100)
        assert isinstance(fig, Figure)

    def test_plot_method_forwards_kwargs(self):
        from rrivis.core.jones.beam.analytic import AnalyticBeamJones

        source_altaz = np.array([[np.pi / 2, 0.0]])
        frequencies = np.array([150e6])

        jones = AnalyticBeamJones(
            source_altaz=source_altaz,
            frequencies=frequencies,
            diameter=14.0,
        )

        fig = jones.plot(
            title="Test Title",
            show_hpbw=False,
            show_first_null=False,
            show_sidelobe_level=False,
            n_points=100,
        )
        ax = fig.get_axes()[0]
        assert ax.get_title() == "Test Title"


class TestPrivateHelpers:
    """Tests for private helper functions."""

    def test_find_first_null(self):
        from rrivis.core.jones.beam.analytic.plotting import _find_first_null

        # Create a simple pattern with a zero crossing
        theta = np.linspace(0, 5, 100)
        voltage = np.cos(np.pi * theta / 2)  # crosses zero at theta=1
        result = _find_first_null(voltage, theta)
        assert result is not None
        np.testing.assert_allclose(result, 1.0, atol=0.1)

    def test_find_first_null_no_null(self):
        from rrivis.core.jones.beam.analytic.plotting import _find_first_null

        theta = np.linspace(0, 1, 100)
        voltage = np.ones_like(theta)
        result = _find_first_null(voltage, theta)
        assert result is None

    def test_find_first_sidelobe(self):
        from rrivis.core.jones.beam.analytic.plotting import _find_first_sidelobe

        # Synthetic pattern: main lobe, null at index 10, sidelobe peak at ~15
        n = 30
        theta = np.linspace(0, 10, n)
        power_db = np.zeros(n)
        power_db[:10] = np.linspace(0, -80, 10)
        power_db[10:20] = np.linspace(-80, -20, 10)
        power_db[15] = -13.0  # sidelobe peak
        power_db[20:] = np.linspace(-20, -60, 10)

        result = _find_first_sidelobe(power_db, theta, first_null_idx=10)
        assert result is not None
        angle, level = result
        assert level == -13.0

    def test_format_angle_arcmin(self):
        from rrivis.core.jones.beam.analytic.plotting import _format_angle

        result = _format_angle(0.5)
        assert "arcmin" in result

    def test_format_angle_degrees(self):
        from rrivis.core.jones.beam.analytic.plotting import _format_angle

        result = _format_angle(2.5)
        assert "deg" in result
