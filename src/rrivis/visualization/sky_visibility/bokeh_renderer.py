"""Bokeh renderer for sky-visibility plans."""

from __future__ import annotations

import os
import tempfile

import numpy as np
from bokeh.io import save, show
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    DataTable,
    Div,
    FixedTicker,
    HoverTool,
    LabelSet,
    LinearColorMapper,
    LogColorMapper,
    NumberFormatter,
    TableColumn,
)
from bokeh.plotting import figure
from bokeh.resources import CDN

from rrivis.core.jones.beam.projection import create_rgba_overlay

from .geometry import axis_from_ra_deg, split_wrapped_path
from .planner import SkyVisibilityPlan, VisibilitySnapshot


def _create_blue_to_white_palette(n: int = 256) -> list[str]:
    palette = []
    for idx in range(n):
        t = idx / (n - 1)
        r = int(t * 255)
        g = int(120 + t * (255 - 120))
        b = int(200 + t * (255 - 200))
        palette.append(f"#{r:02x}{g:02x}{b:02x}")
    return palette


_SOURCE_PALETTE = _create_blue_to_white_palette(256)


class SkyVisibilityBokehRenderer:
    """Render a :class:`SkyVisibilityPlan` with Bokeh."""

    def __init__(
        self,
        plan: SkyVisibilityPlan,
        *,
        show_source_colorbar: bool = False,
        color_scale: str = "log",
    ):
        self.plan = plan
        self.show_source_colorbar = show_source_colorbar
        self.color_scale = color_scale

    def create_plot(self):
        """Create a Bokeh layout."""
        if self.plan.mode == "snapshots":
            return self._create_snapshot_grid()

        plot = self._create_summary_figure(self.plan.title)
        self._add_background(plot)
        self._add_footprint(
            plot, self.plan.footprint_mask, legend_label=self.plan.footprint_model
        )
        self._add_track(plot)
        self._add_beam(plot)
        self._add_sources(plot, visible_source_mask=None)
        self._add_top_visible_labels(plot)
        self._configure_axes(plot)

        tables = self._build_source_tables()
        if tables is not None:
            return column(plot, tables)
        return plot

    def save(
        self,
        layout,
        filename: str = "sky_visibility.html",
        title: str = "Sky Visibility",
        folder_path: str | None = None,
        open_in_browser: bool = False,
    ) -> str | None:
        """Persist the Bokeh layout to HTML."""
        try:
            from rrivis.visualization.bokeh_plots import _persist_bokeh_document

            return _persist_bokeh_document(
                layout,
                filename,
                title,
                save_flag=folder_path is not None,
                folder_path=folder_path,
                open_flag=open_in_browser,
            )
        except ImportError:
            pass

        target = os.path.join(folder_path or tempfile.mkdtemp(), filename)
        save(layout, filename=target, resources=CDN, title=title)
        if open_in_browser:
            import webbrowser
            from pathlib import Path

            webbrowser.open(Path(target).resolve().as_uri())
        return target

    @staticmethod
    def show_plot(layout):
        """Display the plot."""
        show(layout)

    def _create_summary_figure(self, title: str):
        x_range, x_label = self._x_range_and_label()
        return figure(
            title=title,
            x_range=x_range,
            y_range=(-90, 90),
            x_axis_label=x_label,
            y_axis_label="Dec (deg)",
            aspect_ratio=2,
            width=1100,
            height=520,
            background_fill_color="white",
        )

    def _create_snapshot_grid(self):
        plots = []
        for snapshot in self.plan.snapshots:
            plot = self._create_summary_figure(snapshot.label)
            self._add_background(plot)
            self._add_footprint(
                plot, snapshot.footprint_mask, legend_label=snapshot.label
            )
            self._add_snapshot_center(plot, snapshot)
            self._add_sources(plot, visible_source_mask=snapshot.visible_source_mask)
            self._configure_axes(plot)
            plots.append(plot)
        return gridplot(children=plots, ncols=2, merge_tools=False)

    def _x_range_and_label(self) -> tuple[tuple[float, float], str]:
        if self.plan.x_axis == "lst":
            return (-12, 12), "Sidereal Hours (wrapped)"
        return (-180, 180), "RA (wrapped deg)"

    def _full_image_extent(self) -> tuple[float, float]:
        if self.plan.x_axis == "lst":
            return -12.0, 24.0
        return -180.0, 360.0

    def _to_x(self, ra_deg):
        return axis_from_ra_deg(ra_deg, self.plan.x_axis)

    def _add_background(self, plot) -> None:
        if self.plan.projected_background is None:
            return

        valid = self.plan.projected_background[
            ~np.isnan(self.plan.projected_background)
        ]
        if len(valid) == 0:
            return

        x, dw = self._full_image_extent()
        mapper = LogColorMapper(
            palette="Inferno256",
            low=float(np.percentile(valid, 40)),
            high=float(np.nanmax(self.plan.projected_background)),
        )
        renderer = plot.image(
            image=[self.plan.projected_background],
            x=x,
            y=-90,
            dw=dw,
            dh=180,
            color_mapper=mapper,
        )
        plot.add_tools(
            HoverTool(
                tooltips=[("x", "$x"), ("Dec", "$y"), ("Brightness", "@image")],
                renderers=[renderer],
                mode="mouse",
            )
        )
        plot.add_layout(
            ColorBar(
                color_mapper=mapper,
                label_standoff=12,
                border_line_color=None,
                location=(0, 0),
                title="Diffuse",
            ),
            "right",
        )

    def _add_footprint(self, plot, mask: np.ndarray, legend_label: str) -> None:
        x, dw = self._full_image_extent()
        plot.image_rgba(
            image=[self._mask_to_rgba(mask)],
            x=x,
            y=-90,
            dw=dw,
            dh=180,
        )

        for segment in self.plan.footprint_contours[0]:
            x_values = np.asarray(self._to_x(segment[:, 0]), dtype=float)
            y_values = np.asarray(segment[:, 1], dtype=float)
            boundary = 12 if self.plan.x_axis == "lst" else 180
            for xs, ys in split_wrapped_path(x_values, y_values, boundary):
                plot.line(xs, ys, line_width=2, color="#4a90d9", alpha=0.9)

    def _add_track(self, plot) -> None:
        x_values = np.asarray(self._to_x(self.plan.track_ra_deg), dtype=float)
        y_values = np.full_like(x_values, self.plan.latitude_deg, dtype=float)
        plot.line(
            x_values,
            y_values,
            line_dash="dashed",
            line_color="#333333",
            line_width=2,
            alpha=0.7,
        )

    def _add_snapshot_center(self, plot, snapshot: VisibilitySnapshot) -> None:
        plot.scatter(
            [float(self._to_x(snapshot.zenith_ra_deg))],
            [snapshot.zenith_dec_deg],
            size=8,
            color="#111111",
        )

    def _add_beam(self, plot) -> None:
        if self.plan.beam_projection is None:
            return

        rgba = create_rgba_overlay(
            self.plan.beam_projection,
            cmap="RdBu_r",
            vmin_db=-40.0,
            vmax_db=0.0,
            alpha_scale=0.7,
        )
        x, dw = self._full_image_extent()
        plot.image_rgba(
            image=[rgba["image"]],
            x=x,
            y=-90,
            dw=dw,
            dh=180,
        )

        contour_styles = [
            {"color": "white", "dash": "solid", "label": "-3 dB"},
            {"color": "yellow", "dash": "dashed", "label": "-10 dB"},
        ]
        if self.plan.beam_contours:
            boundary = 12 if self.plan.x_axis == "lst" else 180
            for (segments, _level), style in zip(
                self.plan.beam_contours,
                contour_styles,
                strict=False,
            ):
                for segment in segments:
                    xs = np.asarray(self._to_x(segment[:, 0]), dtype=float)
                    ys = np.asarray(segment[:, 1], dtype=float)
                    for seg_x, seg_y in split_wrapped_path(xs, ys, boundary):
                        plot.line(
                            seg_x,
                            seg_y,
                            line_color=style["color"],
                            line_width=2,
                            line_dash=style["dash"],
                        )

        finite = self.plan.beam_projection.power_db[
            np.isfinite(self.plan.beam_projection.power_db)
        ]
        low = float(finite.min()) if len(finite) else -40.0
        import matplotlib.pyplot as plt

        cmap_fn = plt.get_cmap("RdBu_r")
        cmap = [
            f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            for r, g, b, _ in [cmap_fn(i / 255) for i in range(256)]
        ]
        plot.add_layout(
            ColorBar(
                color_mapper=LinearColorMapper(palette=cmap, low=low, high=0.0),
                label_standoff=12,
                border_line_color=None,
                location=(0, 0),
                title="Beam (dB)",
            ),
            "left",
        )

    def _add_sources(self, plot, visible_source_mask: np.ndarray | None) -> None:
        metrics = self.plan.source_metrics
        if metrics is None:
            return

        alpha = np.full(len(metrics.ra_deg), 0.85)
        if visible_source_mask is not None:
            alpha = np.where(visible_source_mask, 0.95, 0.18)

        data = {
            "x": metrics.x_coord,
            "ra": np.asarray(axis_from_ra_deg(metrics.ra_deg, "ra"), dtype=float),
            "dec": metrics.dec_deg,
            "flux": metrics.flux_jy,
            "visible_fraction": 100.0 * metrics.visible_fraction,
            "min_sep_deg": metrics.min_separation_deg,
            "alpha": alpha,
            "name": (
                metrics.source_name
                if metrics.source_name is not None
                else np.full(len(metrics.ra_deg), "", dtype=object)
            ),
        }
        flux = np.asarray(metrics.flux_jy, dtype=float)
        low = max(float(np.nanmin(flux[flux > 0])), 1e-6) if np.any(flux > 0) else 1e-6
        high = float(np.nanmax(flux)) if len(flux) else 1.0
        mapper = (
            LogColorMapper(palette=_SOURCE_PALETTE, low=low, high=high)
            if self.color_scale == "log"
            else LinearColorMapper(palette=_SOURCE_PALETTE, low=low, high=high)
        )
        source = ColumnDataSource(data=data)
        renderer = plot.scatter(
            x="x",
            y="dec",
            size=4,
            source=source,
            color={"field": "flux", "transform": mapper},
            line_color=None,
            alpha="alpha",
        )
        plot.add_tools(
            HoverTool(
                tooltips=[
                    ("Name", "@name"),
                    ("RA", "@ra{0.00}"),
                    ("Dec", "@dec{0.00}"),
                    ("Flux", "@flux{0.000} Jy"),
                    ("Visible", "@visible_fraction{0.0}%"),
                    ("Min sep", "@min_sep_deg{0.00} deg"),
                ],
                renderers=[renderer],
                mode="mouse",
            )
        )
        if self.show_source_colorbar:
            plot.add_layout(
                ColorBar(
                    color_mapper=mapper,
                    label_standoff=12,
                    border_line_color=None,
                    location=(0, 0),
                    title="Flux (Jy)",
                ),
                "right",
            )

    def _add_top_visible_labels(self, plot) -> None:
        metrics = self.plan.source_metrics
        if metrics is None or len(metrics.top_visible_indices) == 0:
            return

        data = {"x": [], "dec": [], "rank": []}
        for rank, idx in enumerate(metrics.top_visible_indices, start=1):
            data["x"].append(float(metrics.x_coord[idx]))
            data["dec"].append(float(metrics.dec_deg[idx]))
            data["rank"].append(str(rank))

        source = ColumnDataSource(data=data)
        plot.scatter(x="x", y="dec", size=7, source=source, color="yellow")
        plot.add_layout(
            LabelSet(
                x="x",
                y="dec",
                text="rank",
                source=source,
                text_color="yellow",
                text_font_style="bold",
                x_offset=5,
                y_offset=3,
            )
        )

    def _configure_axes(self, plot) -> None:
        if self.plan.x_axis == "lst":
            ticks = list(range(-12, 13))
            plot.xaxis.ticker = FixedTicker(ticks=ticks)
            plot.xaxis.major_label_overrides = {tick: f"{tick % 24}h" for tick in ticks}
        else:
            ticks = list(range(-180, 181, 30))
            plot.xaxis.ticker = FixedTicker(ticks=ticks)
            plot.xaxis.major_label_overrides = {tick: f"{tick} deg" for tick in ticks}

        dec_ticks = list(range(-90, 91, 15))
        plot.yaxis.ticker = FixedTicker(ticks=dec_ticks)
        plot.yaxis.major_label_overrides = {tick: f"{tick} deg" for tick in dec_ticks}

    def _build_source_tables(self):
        metrics = self.plan.source_metrics
        if metrics is None or len(metrics.top_visible_indices) == 0:
            return None

        top_visible = self._table_rows(metrics.top_visible_indices)
        nearby = self._table_rows(metrics.nearby_indices)
        return row(
            _source_table(top_visible, "Top Visible Sources"),
            _source_table(nearby, "Nearby Sources", empty_msg="No nearby sources"),
        )

    def _table_rows(self, indices: np.ndarray) -> list[dict]:
        metrics = self.plan.source_metrics
        assert metrics is not None
        rows = []
        for idx in indices:
            first_idx = metrics.first_visible_index[idx]
            last_idx = metrics.last_visible_index[idx]
            rows.append(
                {
                    "name": (
                        metrics.source_name[idx]
                        if metrics.source_name is not None
                        else ""
                    ),
                    "ra": float(axis_from_ra_deg(metrics.ra_deg[idx], "ra")),
                    "dec": float(metrics.dec_deg[idx]),
                    "flux": float(metrics.flux_jy[idx]),
                    "visible_fraction": float(100.0 * metrics.visible_fraction[idx]),
                    "min_sep_deg": float(metrics.min_separation_deg[idx]),
                    "first": (
                        self.plan.track_labels[first_idx] if first_idx >= 0 else "never"
                    ),
                    "last": (
                        self.plan.track_labels[last_idx] if last_idx >= 0 else "never"
                    ),
                }
            )
        return rows

    @staticmethod
    def _mask_to_rgba(mask: np.ndarray) -> np.ndarray:
        rgba = np.zeros(mask.shape, dtype=np.uint32)
        view = rgba.view(dtype=np.uint8).reshape(*mask.shape, 4)
        view[:, :, 0] = 74
        view[:, :, 1] = 144
        view[:, :, 2] = 217
        view[:, :, 3] = np.where(mask, 50, 0).astype(np.uint8)
        return rgba


def _source_table(
    rows: list[dict],
    title: str,
    empty_msg: str = "No sources found",
):
    if not rows:
        return Div(text=f"<p><i>{empty_msg}</i></p>", width=520)

    data = {
        "rank": list(range(1, len(rows) + 1)),
        "name": [row["name"] for row in rows],
        "ra": [f"{row['ra']:.3f}" for row in rows],
        "dec": [f"{row['dec']:.3f}" for row in rows],
        "flux": [row["flux"] for row in rows],
        "visible_fraction": [row["visible_fraction"] for row in rows],
        "min_sep_deg": [row["min_sep_deg"] for row in rows],
        "first": [row["first"] for row in rows],
        "last": [row["last"] for row in rows],
    }
    source = ColumnDataSource(data=data)
    columns = [
        TableColumn(field="rank", title="Rank", width=40),
        TableColumn(field="name", title="Name", width=110),
        TableColumn(field="ra", title="RA", width=80),
        TableColumn(field="dec", title="Dec", width=80),
        TableColumn(
            field="flux",
            title="Flux (Jy)",
            width=90,
            formatter=NumberFormatter(format="0.000"),
        ),
        TableColumn(
            field="visible_fraction",
            title="Visible %",
            width=80,
            formatter=NumberFormatter(format="0.0"),
        ),
        TableColumn(
            field="min_sep_deg",
            title="Min Sep",
            width=80,
            formatter=NumberFormatter(format="0.00"),
        ),
        TableColumn(field="first", title="First", width=150),
        TableColumn(field="last", title="Last", width=150),
    ]
    header = Div(text=f"<h3 style='margin: 10px 0 5px 0;'>{title}</h3>", width=900)
    table = DataTable(
        source=source,
        columns=columns,
        width=900,
        height=30 + len(rows) * 28,
        index_position=None,
    )
    return column(header, table)
