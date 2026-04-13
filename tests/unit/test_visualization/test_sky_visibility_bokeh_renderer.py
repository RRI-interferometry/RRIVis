"""Tests for Bokeh sky-visibility rendering."""

import numpy as np
from bokeh.layouts import Column
from bokeh.models import DataTable, Div, GridPlot

from rrivis.core.precision import PrecisionConfig
from rrivis.core.sky import create_from_arrays
from rrivis.visualization.sky_visibility import (
    SkyVisibilityBokehRenderer,
    SkyVisibilityPlanner,
)


def _point_sky():
    zeros = np.zeros(4)
    return create_from_arrays(
        ra_rad=np.deg2rad([30.0, 45.0, 60.0, 120.0]),
        dec_rad=np.deg2rad([-30.0, -28.0, -32.0, -30.0]),
        flux=np.array([10.0, 5.0, 8.0, 1.0]),
        spectral_index=zeros,
        stokes_q=zeros,
        stokes_u=zeros,
        stokes_v=zeros,
        model_name="test_points",
        brightness_conversion="planck",
        precision=PrecisionConfig.standard(),
    )


class TestSkyVisibilityBokehRenderer:
    def test_summary_layout_contains_tables(self):
        plan = SkyVisibilityPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=1.0,
            lst_end_hours=5.0,
            frequency_mhz=150.0,
            field_radius_deg=5.0,
            sky_model=_point_sky(),
            footprint_step_seconds=3600.0,
        ).build()

        layout = SkyVisibilityBokehRenderer(
            plan,
            show_source_colorbar=True,
        ).create_plot()

        assert isinstance(layout, Column)
        assert len(list(layout.select({"type": DataTable}))) >= 1
        assert len(list(layout.select({"type": Div}))) >= 1

    def test_lst_axis_uses_wrapped_hour_range(self):
        plan = SkyVisibilityPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=1.0,
            lst_end_hours=3.0,
            frequency_mhz=150.0,
            field_radius_deg=5.0,
            sky_model=_point_sky(),
            x_axis="lst",
            footprint_step_seconds=3600.0,
        ).build()

        layout = SkyVisibilityBokehRenderer(plan).create_plot()
        figure = layout.children[0]

        assert figure.x_range.start == -12
        assert figure.x_range.end == 12

    def test_snapshot_mode_returns_gridplot(self):
        plan = SkyVisibilityPlanner(
            latitude_deg=-30.0,
            longitude_deg=21.0,
            lst_start_hours=1.0,
            lst_end_hours=5.0,
            frequency_mhz=150.0,
            field_radius_deg=5.0,
            sky_model=_point_sky(),
            mode="snapshots",
            snapshot_step_seconds=7200.0,
        ).build()

        layout = SkyVisibilityBokehRenderer(plan).create_plot()

        assert isinstance(layout, GridPlot)
        assert len(layout.children) == 3
