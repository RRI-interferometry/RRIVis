"""Tests for rrivis.utils.diagnostics.strip_plotter — Bokeh visualisation."""

import numpy as np

from rrivis.utils.diagnostics.planner import ObservableStrip
from rrivis.utils.diagnostics.strip_plotter import StripPlotter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_strip(**overrides) -> ObservableStrip:
    """Build a minimal ObservableStrip with sensible defaults."""
    defaults = {
        "ra_start_deg": 15.0,
        "ra_end_deg": 75.0,
        "dec_lower_deg": -35.0,
        "dec_upper_deg": -25.0,
        "latitude_deg": -30.0,
        "fov_radius_deg": 5.0,
        "frequency_hz": 150e6,
        "projected_map": None,
        "source_ra_deg": None,
        "source_dec_deg": None,
        "source_flux_jy": None,
        "in_strip_mask": None,
        "top_n_indices": None,
        "beam_projection": None,
        "beam_rgba": None,
        "beam_contours": None,
        "obstime_start_iso": None,
        "obstime_end_iso": None,
        "lst_start_hours": 1.0,
        "lst_end_hours": 5.0,
        "background_mode": "none",
    }
    defaults.update(overrides)
    return ObservableStrip(**defaults)


def _strip_with_sources() -> ObservableStrip:
    """Strip with a few synthetic point sources."""
    ra = np.array([30.0, 45.0, 60.0, 120.0])
    dec = np.array([-30.0, -28.0, -32.0, -30.0])
    flux = np.array([10.0, 5.0, 8.0, 1.0])
    mask = np.array([True, True, True, False])
    top = np.array([0, 2, 1])  # sorted by flux: 10, 8, 5

    return _minimal_strip(
        source_ra_deg=ra,
        source_dec_deg=dec,
        source_flux_jy=flux,
        in_strip_mask=mask,
        top_n_indices=top,
        background_mode="none",
    )


# ---------------------------------------------------------------------------
# Basic plot creation
# ---------------------------------------------------------------------------


class TestCreatePlot:
    def test_returns_bokeh_object(self):
        """create_plot() should return a Bokeh model."""
        from bokeh.model import Model

        strip = _minimal_strip()
        plotter = StripPlotter(strip)
        layout = plotter.create_plot()
        assert isinstance(layout, Model)

    def test_with_sources(self):
        """Plot with sources should not raise."""
        from bokeh.model import Model

        strip = _strip_with_sources()
        plotter = StripPlotter(strip, color_scale="log")
        layout = plotter.create_plot()
        assert isinstance(layout, Model)

    def test_with_sources_reference_mode(self):
        """Reference background mode with sources produces a layout."""
        from bokeh.model import Model

        strip = _strip_with_sources()
        strip = _minimal_strip(
            source_ra_deg=strip.source_ra_deg,
            source_dec_deg=strip.source_dec_deg,
            source_flux_jy=strip.source_flux_jy,
            in_strip_mask=strip.in_strip_mask,
            top_n_indices=strip.top_n_indices,
            background_mode="reference",
        )
        plotter = StripPlotter(strip)
        layout = plotter.create_plot()
        assert isinstance(layout, Model)


# ---------------------------------------------------------------------------
# LST axis mode
# ---------------------------------------------------------------------------


class TestLSTAxis:
    def test_lst_axis_x_range(self):
        """With use_lst_axis, x-range should be [-12, 12]."""
        strip = _minimal_strip()
        plotter = StripPlotter(strip, use_lst_axis=True)
        layout = plotter.create_plot()

        # The layout is a figure (no sources → no tables)

        # Navigate to the actual figure
        fig = layout
        assert fig.x_range.start == -12
        assert fig.x_range.end == 12


# ---------------------------------------------------------------------------
# Background modes
# ---------------------------------------------------------------------------


class TestBackgroundModes:
    def test_none_mode(self):
        """'none' mode should produce a plot with white background."""
        strip = _minimal_strip(background_mode="none")
        plotter = StripPlotter(strip)
        layout = plotter.create_plot()
        assert layout is not None

    def test_gsm_mode_no_map(self):
        """'gsm' mode without a projected map should not crash."""
        strip = _minimal_strip(background_mode="gsm", projected_map=None)
        plotter = StripPlotter(strip)
        layout = plotter.create_plot()
        assert layout is not None

    def test_gsm_mode_with_map(self):
        """'gsm' mode with a projected map renders background."""
        fake_map = np.random.uniform(1.0, 100.0, (500, 1000))
        strip = _minimal_strip(background_mode="gsm", projected_map=fake_map)
        plotter = StripPlotter(strip)
        layout = plotter.create_plot()
        assert layout is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_wrapping_strip(self):
        """Strip that wraps around RA ±180 should not raise."""
        strip = _minimal_strip(ra_start_deg=170.0, ra_end_deg=-170.0)
        plotter = StripPlotter(strip)
        layout = plotter.create_plot()
        assert layout is not None

    def test_linear_color_scale(self):
        """Linear colour scale should work."""
        strip = _strip_with_sources()
        plotter = StripPlotter(strip, color_scale="linear")
        layout = plotter.create_plot()
        assert layout is not None
