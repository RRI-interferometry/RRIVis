"""Interactive Bokeh visualisation of the observable sky strip.

Takes a computed :class:`ObservableStrip` and produces an interactive Bokeh
layout with background map, observable region, point sources, beam overlay,
and source tables.
"""

from __future__ import annotations

import logging

import numpy as np
from bokeh.io import save
from bokeh.layouts import column
from bokeh.layouts import row as bokeh_row
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    DataTable,
    Div,
    FixedTicker,
    HoverTool,
    Label,
    LabelSet,
    LinearColorMapper,
    LogColorMapper,
    NumberFormatter,
    TableColumn,
)
from bokeh.plotting import figure, show
from bokeh.resources import CDN

from .planner import ObservableStrip

logger = logging.getLogger(__name__)


def _create_blue_to_white_palette(n: int = 256) -> list[str]:
    """Gradient from sky-blue ``(0, 120, 200)`` to white."""
    palette = []
    for i in range(n):
        t = i / (n - 1)
        r = int(0 + t * 255)
        g = int(120 + t * (255 - 120))
        b = int(200 + t * (255 - 200))
        palette.append(f"#{r:02x}{g:02x}{b:02x}")
    return palette


_BLUE_WHITE_PALETTE = _create_blue_to_white_palette(256)


class StripPlotter:
    """Create interactive Bokeh plots of the observable sky strip.

    Parameters
    ----------
    strip : ObservableStrip
        Pre-computed strip data from :meth:`DiagnosticsPlanner.compute`.
    use_lst_axis : bool
        Display the x-axis as LST (hours) instead of RA (degrees).
    color_scale : str
        ``"log"`` or ``"linear"`` colour mapping for point-source flux.
    """

    def __init__(
        self,
        strip: ObservableStrip,
        *,
        use_lst_axis: bool = False,
        color_scale: str = "log",
    ):
        self.strip = strip
        self.use_lst = use_lst_axis
        self.color_scale = color_scale

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def create_plot(self):
        """Build the complete Bokeh layout.

        Returns
        -------
        bokeh.layouts.column or bokeh.plotting.figure
        """
        s = self.strip
        show_bg = (
            s.background_mode in ("gsm", "reference") and s.projected_map is not None
        )
        show_src_cbar = s.background_mode in ("none", "reference")
        light = s.background_mode == "none"

        # --- colour mapper for background ---
        bg_mapper = None
        if show_bg:
            pm = s.projected_map
            valid = pm.compressed() if hasattr(pm, "compressed") else pm[~np.isnan(pm)]
            lo = float(np.percentile(valid, 40))
            hi = float(np.nanmax(pm))
            bg_mapper = LogColorMapper(palette="Inferno256", low=lo, high=hi)

        # --- title ---
        if s.obstime_start_iso and s.obstime_end_iso:
            title = f"Observable Strip: {s.obstime_start_iso} to {s.obstime_end_iso}"
        elif s.lst_start_hours is not None and s.lst_end_hours is not None:
            ls = s.lst_start_hours % 24
            le = s.lst_end_hours % 24
            title = f"Observable Strip: LST {ls:.2f}h to {le:.2f}h"
        else:
            title = "Observable Strip"

        # --- axes ---
        if self.use_lst:
            x_range = (-12, 12)
            x_label = "LST (hours)"
        else:
            x_range = (-180, 180)
            x_label = "RA (\u00b0)"

        p = figure(
            title=title,
            x_range=x_range,
            y_range=(-90, 90),
            x_axis_label=x_label,
            y_axis_label="Dec (\u00b0)",
            aspect_ratio=2,
            width=1000,
            height=500,
            background_fill_color="white" if light else None,
        )

        # 1. Background image
        bg_image = None
        if show_bg:
            if self.use_lst:
                bg_image = p.image(
                    image=[s.projected_map],
                    x=-12,
                    y=-90,
                    dw=24,
                    dh=180,
                    color_mapper=bg_mapper,
                )
            else:
                bg_image = p.image(
                    image=[s.projected_map],
                    x=-180,
                    y=-90,
                    dw=360,
                    dh=180,
                    color_mapper=bg_mapper,
                )
            if s.background_mode == "gsm" and not self.use_lst:
                p.add_tools(
                    HoverTool(
                        tooltips=[
                            ("RA", "$x\u00b0"),
                            ("Dec", "$y\u00b0"),
                            ("Brightness", "@image"),
                        ],
                        formatters={"$x": "printf", "$y": "printf"},
                        mode="mouse",
                        renderers=[bg_image],
                        attachment="right",
                    )
                )

        # 2. Declination lines
        self._add_dec_lines(p, light)

        # 3. Observable strip patch
        self._add_strip_patch(p, light)

        # 4. Strip boundary lines
        self._add_boundary_lines(p, light)

        # 5. Beam overlay
        self._add_beam_overlay(p, light)

        # 6. Point sources + 7. Top-N highlighting
        self._add_point_sources(p, add_colorbar=show_src_cbar)
        top_sources = self._highlight_top_sources(p, light)

        # 8. Background colorbar
        if s.background_mode == "gsm" and show_bg:
            p.add_layout(
                ColorBar(
                    color_mapper=bg_mapper,
                    label_standoff=12,
                    border_line_color=None,
                    location=(0, 0),
                ),
                "right",
            )

        # 9. Axis ticks
        self._configure_axes(p)

        # 10. Source tables
        if show_src_cbar and s.source_ra_deg is not None:
            tables = self._build_source_tables(top_sources)
            if tables is not None:
                return column(p, tables)

        return p

    def save(
        self,
        layout,
        filename: str = "sky_strip.html",
        title: str = "Observable Sky Strip",
        folder_path: str | None = None,
        open_in_browser: bool = False,
    ) -> str | None:
        """Persist the Bokeh layout to an HTML file.

        Delegates to ``_persist_bokeh_document`` when available; falls
        back to ``bokeh.io.save`` + ``webbrowser.open`` otherwise.
        """
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

        import os
        import tempfile

        target = os.path.join(folder_path or tempfile.mkdtemp(), filename)
        save(layout, filename=target, resources=CDN, title=title)
        if open_in_browser:
            import webbrowser
            from pathlib import Path

            webbrowser.open(Path(target).resolve().as_uri())
        return target

    def show_plot(self, layout):
        """Display a plot inline (notebook) or in browser."""
        show(layout)

    # ------------------------------------------------------------------
    # Private rendering helpers
    # ------------------------------------------------------------------

    def _ra_to_lst(self, ra_deg: float) -> float:
        """Convert RA (degrees) to LST (hours) in [-12, 12)."""
        ra_n = ra_deg % 360
        if ra_n > 180:
            ra_n -= 360
        return ra_n / 15.0

    def _normalize_ra(self, ra: float) -> float:
        """Normalise RA to [-180, 180]."""
        ra = ra % 360
        if ra > 180:
            ra -= 360
        return ra

    def _to_x(self, ra_deg: float) -> float:
        """RA (degrees) → plot x-coordinate."""
        if self.use_lst:
            return self._ra_to_lst(ra_deg)
        return self._normalize_ra(ra_deg)

    def _add_dec_lines(self, p, light: bool):
        s = self.strip
        col = "#4a90d9" if light else "white"
        tcol = "#333333" if light else "white"
        x_line = [-12, 12] if self.use_lst else [-180, 180]

        for dec, y_off in [(s.dec_upper_deg, -12), (s.dec_lower_deg, 2)]:
            p.line(
                x=x_line,
                y=[dec, dec],
                line_dash="dotted",
                color=col,
                alpha=0.7,
                line_width=2,
            )
            p.add_layout(
                Label(
                    x=x_line[0] + 0.2,
                    y=dec,
                    text=f"{dec:.1f}\u00b0",
                    text_font_size="9pt",
                    text_color=tcol,
                    text_alpha=0.9,
                    x_offset=2,
                    y_offset=y_off,
                )
            )

    def _add_strip_patch(self, p, light: bool):
        s = self.strip
        col = "#4a90d9" if light else "red"
        alpha = 0.2 if light else 0.15

        x_start = self._to_x(s.ra_start_deg)
        x_end = self._to_x(s.ra_end_deg)
        boundary = 12 if self.use_lst else 180

        if x_start <= x_end:
            xs = np.linspace(x_start, x_end, 200)
            p.patch(
                x=list(xs) + list(xs[::-1]),
                y=[s.dec_lower_deg] * len(xs) + [s.dec_upper_deg] * len(xs),
                color=col,
                fill_alpha=alpha,
                line_alpha=1.0,
            )
        else:
            # Wrapping
            xs1 = np.linspace(x_start, boundary, 100)
            xs2 = np.linspace(-boundary, x_end, 100)
            for xs in (xs1, xs2):
                p.patch(
                    x=list(xs) + list(xs[::-1]),
                    y=[s.dec_lower_deg] * len(xs) + [s.dec_upper_deg] * len(xs),
                    color=col,
                    fill_alpha=alpha,
                    line_alpha=1.0,
                )

    def _add_boundary_lines(self, p, light: bool):
        s = self.strip
        col = "#4a90d9" if light else "yellow"
        tcol = "#333333" if light else "yellow"

        for ra in (s.ra_start_deg, s.ra_end_deg):
            x = self._to_x(ra)
            if self.use_lst:
                lbl = f"{x % 24:.2f}h"
            else:
                lbl = f"{x:.1f}\u00b0"

            p.line(
                x=[x, x],
                y=[s.dec_lower_deg, s.dec_upper_deg],
                line_dash="dashed",
                color=col,
                alpha=0.8,
                line_width=2,
            )
            p.add_layout(
                Label(
                    x=x,
                    y=s.dec_lower_deg,
                    text=lbl,
                    text_font_size="9pt",
                    text_color=tcol,
                    text_alpha=0.9,
                    x_offset=-15,
                    y_offset=-18,
                )
            )

    def _add_beam_overlay(self, p, light: bool):
        s = self.strip
        if s.beam_rgba is None or s.beam_projection is None:
            return

        import matplotlib.pyplot as plt

        rgba = s.beam_rgba
        proj = s.beam_projection

        # Compute image position in plot coordinates
        if self.use_lst:
            ra_grid_x = np.array([self._ra_to_lst(r) for r in proj.ra_grid_deg])
            dw = rgba["dw"] / 15.0
            cx = self._ra_to_lst(proj.zenith_ra_deg)
            x = cx - dw / 2
        else:
            dw = rgba["dw"]
            cx = self._normalize_ra(proj.zenith_ra_deg)
            x = cx - dw / 2
            ra_grid_x = np.array([self._normalize_ra(r) for r in proj.ra_grid_deg])

        p.image_rgba(
            image=[rgba["image"]],
            x=x,
            y=rgba["y"],
            dw=dw,
            dh=rgba["dh"],
        )

        # Contour lines
        contour_styles = [
            {"color": "white", "dash": "solid", "label": "-3 dB (HPBW)", "pos": 0.25},
            {"color": "yellow", "dash": "dashed", "label": "-10 dB", "pos": 0.75},
        ]
        if s.beam_contours:
            X, Y = np.meshgrid(ra_grid_x, proj.dec_grid_deg)
            for (segments, _level), style in zip(
                s.beam_contours, contour_styles, strict=False
            ):
                labeled = False
                for verts in segments:
                    # Remap vertices from RA-deg to plot x
                    xs_raw = verts[:, 0]
                    if self.use_lst:
                        xs = np.array([self._ra_to_lst(r) for r in xs_raw])
                    else:
                        xs = np.array([self._normalize_ra(r) for r in xs_raw])
                    ys = verts[:, 1]

                    p.line(
                        x=xs,
                        y=ys,
                        line_color=style["color"],
                        line_width=2,
                        line_dash=style["dash"],
                    )

                    if not labeled and len(xs) > 2:
                        idx = int(len(xs) * style["pos"])
                        p.add_layout(
                            Label(
                                x=xs[idx],
                                y=ys[idx],
                                text=style["label"],
                                text_font_size="8pt",
                                text_color=style["color"],
                                text_font_style="bold",
                                background_fill_color="black",
                                background_fill_alpha=0.5,
                                x_offset=5,
                                y_offset=5,
                            )
                        )
                        labeled = True

        # Beam colorbar
        cmap = plt.get_cmap("RdBu_r")
        beam_pal = [
            f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            for r, g, b, _ in [cmap(i / 255) for i in range(256)]
        ]

        p.add_layout(
            ColorBar(
                color_mapper=LinearColorMapper(
                    palette=beam_pal,
                    low=float(
                        s.beam_projection.power_db[
                            np.isfinite(s.beam_projection.power_db)
                        ].min()
                    )
                    if np.any(np.isfinite(s.beam_projection.power_db))
                    else -40.0,
                    high=0.0,
                ),
                label_standoff=12,
                border_line_color=None,
                location=(0, 0),
                title="Beam (dB)",
            ),
            "left",
        )

    def _add_point_sources(self, p, add_colorbar: bool = False):
        s = self.strip
        if s.source_ra_deg is None:
            return

        src = {"x": [], "dec": [], "flux": [], "ra": [], "lst": []}
        for i in range(len(s.source_ra_deg)):
            ra = s.source_ra_deg[i]
            dec = s.source_dec_deg[i]
            flux = s.source_flux_jy[i]

            src["ra"].append(float(ra if ra <= 180 else ra - 360))
            x = self._to_x(ra)
            src["x"].append(float(x))
            src["dec"].append(float(dec))
            src["flux"].append(float(flux))
            src["lst"].append(float(x % 24) if self.use_lst else 0.0)

        flux_arr = np.array(src["flux"])
        lo = float(
            max(flux_arr.min(), 1e-6) if self.color_scale == "log" else flux_arr.min()
        )
        hi = float(flux_arr.max())

        if self.color_scale == "log":
            mapper = LogColorMapper(palette=_BLUE_WHITE_PALETTE, low=lo, high=hi)
        else:
            mapper = LinearColorMapper(palette=_BLUE_WHITE_PALETTE, low=lo, high=hi)

        cds = ColumnDataSource(data=src)
        scatter = p.scatter(
            x="x",
            y="dec",
            size=3,
            source=cds,
            color={"field": "flux", "transform": mapper},
            line_color=None,
        )

        if self.use_lst:
            tips = [
                ("LST", "@lst{0.2f}h"),
                ("RA", "@ra{0.2f}\u00b0"),
                ("Dec", "@dec{0.2f}\u00b0"),
                ("Flux", "@flux Jy"),
            ]
        else:
            tips = [
                ("RA", "@ra{0.2f}\u00b0"),
                ("Dec", "@dec{0.2f}\u00b0"),
                ("Flux", "@flux Jy"),
            ]

        p.add_tools(
            HoverTool(
                tooltips=tips,
                mode="mouse",
                renderers=[scatter],
                attachment="left",
            )
        )

        if add_colorbar:
            p.add_layout(
                ColorBar(
                    color_mapper=mapper,
                    label_standoff=12,
                    border_line_color=None,
                    location=(0, 0),
                    title="Flux (Jy)",
                ),
                "right",
            )

    def _highlight_top_sources(self, p, light: bool) -> list[dict]:
        """Draw yellow markers + rank labels for the top-N in-strip sources.

        Returns the top-source list (for tables).
        """
        s = self.strip
        if s.top_n_indices is None or s.source_ra_deg is None:
            return []

        top: list[dict] = []
        data = {"x": [], "dec": [], "rank": []}

        for rank, idx in enumerate(s.top_n_indices):
            ra = s.source_ra_deg[idx]
            dec = s.source_dec_deg[idx]
            flux = s.source_flux_jy[idx]
            x = self._to_x(ra)

            data["x"].append(float(x))
            data["dec"].append(float(dec))
            data["rank"].append(str(rank + 1))
            top.append(
                {
                    "ra": float(self._normalize_ra(ra)),
                    "dec": float(dec),
                    "flux": float(flux),
                }
            )

        cds = ColumnDataSource(data=data)
        p.scatter(x="x", y="dec", size=3, source=cds, color="yellow", line_color=None)
        p.add_layout(
            LabelSet(
                x="x",
                y="dec",
                text="rank",
                source=cds,
                text_font_size="9pt",
                text_color="yellow",
                text_font_style="bold",
                x_offset=5,
                y_offset=3,
            )
        )
        return top

    def _configure_axes(self, p):
        if self.use_lst:
            ticks = list(range(-12, 13, 1))
            p.xaxis.ticker = FixedTicker(ticks=ticks)
            p.xaxis.major_label_overrides = {t: f"{t % 24}h" for t in ticks}
        else:
            ticks = list(range(-180, 181, 30))
            p.xaxis.ticker = FixedTicker(ticks=ticks)
            p.xaxis.major_label_overrides = {t: f"{t}\u00b0" for t in ticks}

        dec_ticks = list(range(-90, 91, 15))
        p.yaxis.ticker = FixedTicker(ticks=dec_ticks)
        p.yaxis.major_label_overrides = {t: f"{t}\u00b0" for t in dec_ticks}

    def _build_source_tables(self, top_sources: list[dict]):
        """Build DataTable widgets for top in-strip and nearby sources."""
        if not top_sources:
            return None

        strip_table = _sources_table(
            top_sources,
            "Top Brightest Sources in Observable Strip",
        )

        # Nearby: sources not in strip, within 10 deg buffer
        nearby = self._get_nearby_sources(buffer_deg=10.0, n=3)
        nearby_table = _sources_table(
            nearby,
            "Top Brightest Sources Nearby (10\u00b0 buffer)",
            empty_msg="No nearby sources in buffer zone",
        )

        return bokeh_row(strip_table, nearby_table)

    def _get_nearby_sources(self, buffer_deg: float = 10.0, n: int = 3) -> list[dict]:
        """Top-N brightest sources in the buffer zone around the strip."""
        s = self.strip
        if s.source_ra_deg is None or s.in_strip_mask is None:
            return []

        from rrivis.core.sky.region import SkyRegion

        # Expanded region
        ra_start = s.ra_start_deg
        ra_end = s.ra_end_deg
        width = ra_end - ra_start
        if width < 0:
            width += 360.0
        width_exp = width + 2 * buffer_deg
        height_exp = (s.dec_upper_deg - s.dec_lower_deg) + 2 * buffer_deg

        ra_center = (ra_start + ra_end) / 2.0
        if ra_start > ra_end:
            st = ra_start if ra_start >= 0 else ra_start + 360
            en = ra_end if ra_end >= 0 else ra_end + 360
            if en < st:
                en += 360
            ra_center = ((st + en) / 2.0) % 360

        dec_center = (s.dec_lower_deg + s.dec_upper_deg) / 2.0

        region_exp = SkyRegion.box(
            ra_deg=ra_center % 360,
            dec_deg=dec_center,
            width_deg=min(width_exp, 360.0),
            height_deg=min(height_exp, 180.0),
        )

        ra_rad = np.deg2rad(s.source_ra_deg % 360)
        dec_rad = np.deg2rad(s.source_dec_deg)
        in_buffer = region_exp.contains(ra_rad, dec_rad)

        # Nearby = in buffer but NOT in strip
        nearby_mask = in_buffer & ~s.in_strip_mask
        nearby_flux = np.where(nearby_mask, s.source_flux_jy, -np.inf)
        top_idx = np.argsort(nearby_flux)[::-1][:n]

        result = []
        for idx in top_idx:
            if nearby_flux[idx] <= 0:
                break
            result.append(
                {
                    "ra": float(self._normalize_ra(s.source_ra_deg[idx])),
                    "dec": float(s.source_dec_deg[idx]),
                    "flux": float(s.source_flux_jy[idx]),
                }
            )
        return result


def _sources_table(
    sources: list[dict],
    title: str,
    empty_msg: str = "No sources found",
):
    """Build a Bokeh DataTable widget for a list of sources."""
    if not sources:
        return Div(text=f"<p><i>{empty_msg}</i></p>")

    data = {
        "rank": list(range(1, len(sources) + 1)),
        "ra": [f"{s['ra']:.4f}" for s in sources],
        "dec": [f"{s['dec']:.4f}" for s in sources],
        "flux": [s["flux"] for s in sources],
    }
    cds = ColumnDataSource(data=data)

    cols = [
        TableColumn(field="rank", title="Rank", width=50),
        TableColumn(field="ra", title="RA (\u00b0)", width=100),
        TableColumn(field="dec", title="Dec (\u00b0)", width=100),
        TableColumn(
            field="flux",
            title="Flux (Jy)",
            width=120,
            formatter=NumberFormatter(format="0.000"),
        ),
    ]

    header = Div(
        text=f"<h3 style='margin: 10px 0 5px 0;'>{title}</h3>",
        width=450,
    )
    table = DataTable(
        source=cds,
        columns=cols,
        width=450,
        height=30 + len(sources) * 28,
        index_position=None,
    )
    return column(header, table)
