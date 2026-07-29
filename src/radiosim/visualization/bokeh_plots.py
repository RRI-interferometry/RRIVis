# radiosim/visualization/bokeh_plots.py
"""Canonical-result Bokeh renderers and antenna layout helpers.

The visibility renderers consume an exact :class:`SimulationResult`.  They read
the published coordinate arrays directly — canonical UTC time centers, channel
centers, baseline order, and correlation labels — and never reconstruct an axis
from durations, cadences, or scalar start times.  Stokes I is derived
explicitly as ``XX + YY`` through :meth:`SimulationResult.stokes_i`.

Renderers never open a browser.  Browser presentation belongs to the caller and
always follows publication of the rendered files.
"""

from __future__ import annotations

import logging
import os
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Note: avoid Matplotlib here to keep plotting browser-native via Bokeh/Plotly
import numpy as np
from bokeh.io import save
from bokeh.layouts import column
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LabelSet,
    Legend,
    LinearColorMapper,
)
from bokeh.palettes import Inferno256, Turbo256
from bokeh.plotting import figure
from bokeh.resources import CDN

from radiosim.visualization.errors import ResultPlotContractError

if TYPE_CHECKING:
    from bokeh.models import UIElement

    from radiosim.core.result import SimulationResult

logger = logging.getLogger(__name__)

PHASE_UNITS: tuple[str, ...] = ("radians", "degrees")
TIME_AXIS_LABEL = "Time (MJD, UTC)"
FREQUENCY_AXIS_LABEL = "Frequency (Hz)"

VISIBILITY_PLOT_FILENAME = "visibility-phase-lsts.html"
HEATMAP_PLOT_FILENAME = "heatmaps-freq-time.html"
FREQUENCY_PLOT_FILENAME = "modulus-phase-freq.html"
ANTENNA_LAYOUT_FILENAME = "antenna_layout.html"


def _require_result(result: object) -> SimulationResult:
    """Reject every input that is not an exact published simulation result."""
    from radiosim.core.result import SimulationResult

    if type(result) is not SimulationResult:
        raise ResultPlotContractError(
            "visibility renderers require an exact SimulationResult; "
            f"received {type(result).__name__}"
        )
    return result


def _require_phase_unit(value: object) -> str:
    if value not in PHASE_UNITS:
        raise ResultPlotContractError(
            f"visibility_phase_unit must be 'radians' or 'degrees'; received {value!r}"
        )
    return str(value)


def _require_output_path(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ResultPlotContractError("output_path must be a string or Path")
    return Path(value)


def _baseline_labels(result: SimulationResult) -> tuple[tuple[int, int], ...]:
    """Return selected numeric pairs in the exact published baseline order."""
    return tuple(
        (baseline.ant1.number, baseline.ant2.number)
        for baseline in result.selection.baselines
    )


def _stokes_i(result: SimulationResult) -> np.ndarray:
    """Return the explicit ``XX + YY`` Stokes I cube with shape ``(T, B, F)``."""
    stokes = result.stokes_i()
    expected = (
        len(result.time_grid),
        len(result.selection.baselines),
        int(result.frequencies_hz.size),
    )
    if stokes.shape != expected:
        raise ResultPlotContractError(
            f"canonical Stokes I shape {stokes.shape} does not match the "
            f"published coordinate shape {expected}"
        )
    return stokes


def _phase(values: np.ndarray, unit: str) -> np.ndarray:
    """Unwrap in radians, then convert once into the requested display unit."""
    unwrapped = np.unwrap(np.angle(values), axis=0)
    if unit == "degrees":
        return np.degrees(unwrapped)
    return unwrapped


def _phase_axis_label(unit: str) -> str:
    return f"Phase of Visibility ({unit})"


def _baseline_color(index: int, total: int) -> str:
    return Turbo256[int((index / max(total, 1)) * 255)]


def _persist_bokeh_document(
    doc: UIElement,
    filename: str,
    title: str,
    save_flag: bool,
    folder_path: str | Path | None,
    open_flag: bool,
    save_message: str | None = None,
) -> str | None:
    """Save a Bokeh document to an explicit folder; never invent a temporary one."""
    if open_flag and not (save_flag and folder_path):
        raise ResultPlotContractError(
            "opening a plot in a browser requires an explicit output folder"
        )
    if not (save_flag and folder_path):
        return None

    target_path = os.path.join(str(folder_path), filename)
    save(doc, filename=target_path, resources=CDN, title=title)
    logger.debug(f"{save_message or f'Saved {title} to'} {target_path}")

    if open_flag:
        webbrowser.open(Path(target_path).resolve().as_uri())

    return target_path


def save_plot_document(
    doc: UIElement,
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    """Write one standalone Bokeh document to an exact declared path."""
    target = Path(output_path)
    save(doc, filename=str(target), resources=CDN, title=title)
    return target


def plot_visibility(
    result: SimulationResult,
    *,
    output_path: str | Path | None = None,
    visibility_phase_unit: Literal["radians", "degrees"] = "radians",
) -> UIElement:
    """Plot Stokes I modulus and phase against the canonical UTC time centers.

    Parameters
    ----------
    result
        The exact published :class:`SimulationResult` to render.
    output_path
        Optional exact file path receiving the standalone HTML document.
    visibility_phase_unit
        Display unit for phase; the canonical values remain radians.
    """
    canonical = _require_result(result)
    unit = _require_phase_unit(visibility_phase_unit)
    target = _require_output_path(output_path)

    times = canonical.time_grid.to_mjd()
    stokes = _stokes_i(canonical)
    labels = _baseline_labels(canonical)
    phase_label = _phase_axis_label(unit)

    plots: list[Any] = []
    for index, label in enumerate(labels):
        channel = stokes[:, index, 0]
        modulus = np.abs(channel)
        phase = _phase(channel, unit)

        p_mod = figure(
            width=800,
            height=300,
            title=f"Modulus of Visibility vs Time for Baseline {label}",
        )
        p_mod.line(times, modulus, line_width=2, legend_label=f"Baseline {label}")
        p_mod.xaxis.axis_label = TIME_AXIS_LABEL
        p_mod.yaxis.axis_label = "Modulus of Visibility (Stokes I)"
        p_mod.legend.location = "top_left"

        p_phase = figure(
            width=800,
            height=300,
            title=f"Phase of Visibility vs Time for Baseline {label}",
        )
        p_phase.line(times, phase, line_width=2, legend_label=f"Baseline {label}")
        p_phase.xaxis.axis_label = TIME_AXIS_LABEL
        p_phase.yaxis.axis_label = phase_label
        p_phase.legend.location = "top_left"

        plots.append(p_mod)
        plots.append(p_phase)

    combined_mod = figure(
        width=1400,
        height=1400,
        title="Modulus of Visibility vs Time for All Baselines",
    )
    combined_phase = figure(
        width=1400,
        height=1400,
        title="Combined Phase of Visibility vs Time for All Baselines",
    )
    modulus_items: list[tuple[str, list[Any]]] = []
    phase_items: list[tuple[str, list[Any]]] = []
    for index, label in enumerate(labels):
        channel = stokes[:, index, 0]
        color = _baseline_color(index, len(labels))
        modulus_line = combined_mod.line(
            times,
            np.abs(channel),
            line_width=2,
            color=color,
            name=str(label),
        )
        modulus_items.append((f"Baseline {label}", [modulus_line]))
        combined_mod.add_tools(
            HoverTool(
                renderers=[modulus_line],
                tooltips=[
                    ("Time (MJD)", "@x"),
                    ("Value", "@y"),
                    ("Baseline", str(label)),
                ],
                mode="mouse",
            )
        )
        phase_line = combined_phase.line(
            times,
            _phase(channel, unit),
            line_width=2,
            color=color,
            name=str(label),
        )
        phase_items.append((f"Baseline {label}", [phase_line]))
        combined_phase.add_tools(
            HoverTool(
                renderers=[phase_line],
                tooltips=[
                    ("Time (MJD)", "@x"),
                    ("Value", "@y"),
                    ("Baseline", str(label)),
                ],
                mode="mouse",
            )
        )

    combined_mod.xaxis.axis_label = TIME_AXIS_LABEL
    combined_mod.yaxis.axis_label = "Modulus of Visibility (Stokes I)"
    combined_phase.xaxis.axis_label = TIME_AXIS_LABEL
    combined_phase.yaxis.axis_label = phase_label
    for target_figure, items in (
        (combined_mod, modulus_items),
        (combined_phase, phase_items),
    ):
        legend = Legend(
            items=items,
            location="center",
            click_policy="hide",
            title="Baselines",
        )
        legend.ncols = 10
        target_figure.add_layout(legend, "below")

    plots.append(combined_mod)
    plots.append(combined_phase)
    document = column(*plots)

    if target is not None:
        save_plot_document(document, target, title="Visibility/Phase Plots")
    return document


def plot_heatmaps(
    result: SimulationResult,
    *,
    output_path: str | Path | None = None,
    visibility_phase_unit: Literal["radians", "degrees"] = "radians",
) -> UIElement:
    """Render Stokes I modulus and phase over the canonical time/frequency grid."""
    canonical = _require_result(result)
    unit = _require_phase_unit(visibility_phase_unit)
    target = _require_output_path(output_path)

    times = canonical.time_grid.to_mjd()
    frequencies = canonical.frequencies_hz
    stokes = _stokes_i(canonical)
    labels = _baseline_labels(canonical)
    phase_label = _phase_axis_label(unit)

    extent = {
        "x": [float(times[0])],
        "y": [float(frequencies[0])],
        "dw": [float(times[-1] - times[0])],
        "dh": [float(frequencies[-1] - frequencies[0])],
    }

    plots: list[Any] = []
    for index, label in enumerate(labels):
        moduli_total = np.abs(stokes[:, index, :])
        phases_total = _phase(stokes[:, index, :], unit)

        for values, title, axis_label in (
            (
                moduli_total,
                f"Modulus of Visibility Heatmap for Baseline {label}",
                "Modulus of Visibility (Stokes I)",
            ),
            (
                phases_total,
                f"Phase of Visibility Heatmap for Baseline {label}",
                phase_label,
            ),
        ):
            mapper = LinearColorMapper(
                palette=Inferno256,
                low=float(values.min()),
                high=float(values.max()),
            )
            source = ColumnDataSource({"image": [values.T], **extent})
            panel = figure(width=800, height=300, title=title)
            panel.image(
                image="image",
                x="x",
                y="y",
                dw="dw",
                dh="dh",
                color_mapper=mapper,
                source=source,
            )
            panel.xaxis.axis_label = TIME_AXIS_LABEL
            panel.yaxis.axis_label = FREQUENCY_AXIS_LABEL
            color_bar = ColorBar(
                color_mapper=mapper,
                location=(0, 0),
                title=axis_label,
            )
            panel.add_layout(color_bar, "right")
            plots.append(panel)

    document = column(*plots, sizing_mode="stretch_both")
    if target is not None:
        save_plot_document(document, target, title="Visibility Heatmaps")
    return document


def plot_modulus_vs_frequency(
    result: SimulationResult,
    *,
    output_path: str | Path | None = None,
    visibility_phase_unit: Literal["radians", "degrees"] = "radians",
) -> UIElement:
    """Plot Stokes I against the canonical channel centers at peak modulus."""
    canonical = _require_result(result)
    unit = _require_phase_unit(visibility_phase_unit)
    target = _require_output_path(output_path)

    times = canonical.time_grid.to_mjd()
    frequencies = canonical.frequencies_hz
    stokes = _stokes_i(canonical)
    labels = _baseline_labels(canonical)
    phase_label = _phase_axis_label(unit)

    moduli = np.abs(stokes)
    peak_indices = tuple(
        int(np.argmax(moduli[:, index, :].max(axis=1))) for index in range(len(labels))
    )

    plots: list[Any] = []
    for index, label in enumerate(labels):
        peak = peak_indices[index]
        peak_mjd = float(times[peak])

        p_mod = figure(
            width=800,
            height=300,
            title=(
                f"Modulus of Visibility vs Frequency for Baseline {label} "
                f"at MJD {peak_mjd:.8f}"
            ),
        )
        p_mod.line(
            frequencies,
            moduli[peak, index, :],
            line_width=2,
            legend_label=f"Baseline {label}",
        )
        p_mod.xaxis.axis_label = FREQUENCY_AXIS_LABEL
        p_mod.yaxis.axis_label = "Modulus of Visibility (Stokes I)"
        p_mod.legend.location = "top_left"

        p_phase = figure(
            width=800,
            height=300,
            title=(
                f"Phase of Visibility vs Frequency for Baseline {label} "
                f"at MJD {peak_mjd:.8f}"
            ),
        )
        p_phase.line(
            frequencies,
            _phase(stokes[peak, index, :], unit),
            line_width=2,
            legend_label=f"Baseline {label}",
        )
        p_phase.xaxis.axis_label = FREQUENCY_AXIS_LABEL
        p_phase.yaxis.axis_label = phase_label
        p_phase.legend.location = "top_left"

        plots.append(p_mod)
        plots.append(p_phase)

    combined_mod = figure(
        width=1400,
        height=1400,
        title="Modulus of Visibility vs Frequency for All Baselines at Peak Modulus",
    )
    combined_phase = figure(
        width=1400,
        height=1400,
        title="Phase of Visibility vs Frequency for All Baselines at Peak Modulus",
    )
    modulus_items: list[tuple[str, list[Any]]] = []
    phase_items: list[tuple[str, list[Any]]] = []
    for index, label in enumerate(labels):
        peak = peak_indices[index]
        color = _baseline_color(index, len(labels))
        modulus_line = combined_mod.line(
            frequencies,
            moduli[peak, index, :],
            line_width=2,
            color=color,
            name=str(label),
        )
        modulus_items.append((f"Baseline {label}", [modulus_line]))
        combined_mod.add_tools(
            HoverTool(
                renderers=[modulus_line],
                tooltips=[
                    ("Frequency (Hz)", "@x"),
                    ("Value", "@y"),
                    ("Baseline", str(label)),
                ],
                mode="mouse",
            )
        )
        phase_line = combined_phase.line(
            frequencies,
            _phase(stokes[peak, index, :], unit),
            line_width=2,
            color=color,
            name=str(label),
        )
        phase_items.append((f"Baseline {label}", [phase_line]))
        combined_phase.add_tools(
            HoverTool(
                renderers=[phase_line],
                tooltips=[
                    ("Frequency (Hz)", "@x"),
                    ("Value", "@y"),
                    ("Baseline", str(label)),
                ],
                mode="mouse",
            )
        )

    combined_mod.xaxis.axis_label = FREQUENCY_AXIS_LABEL
    combined_mod.yaxis.axis_label = "Modulus of Visibility (Stokes I)"
    combined_phase.xaxis.axis_label = FREQUENCY_AXIS_LABEL
    combined_phase.yaxis.axis_label = phase_label
    for target_figure, items in (
        (combined_mod, modulus_items),
        (combined_phase, phase_items),
    ):
        legend = Legend(
            items=items,
            location="center",
            click_policy="hide",
            title="Baselines",
        )
        legend.ncols = 10
        target_figure.add_layout(legend, "below")

    plots.append(combined_mod)
    plots.append(combined_phase)
    document = column(*plots, sizing_mode="stretch_both")

    if target is not None:
        save_plot_document(
            document,
            target,
            title="Visibility Modulus/Phase vs Frequency",
        )
    return document


def plot_antenna_layout(
    antennas,
    plotting="bokeh",
    save_simulation_data=False,
    folder_path=None,
    open_in_browser=True,
):
    """
    Plot antenna positions (E vs N) with hover labels.

    Parameters:
    - antennas: canonical resolved antenna tuple.
    - plotting (str): currently only 'bokeh' supported.
    - save_simulation_data (bool): save HTML when True and folder_path provided.
    - folder_path (str or None): directory to save HTML.
    - open_in_browser (bool): whether to open in browser when using bokeh.

    Returns:
    - Bokeh figure when plotting='bokeh', else None.
    """
    if plotting != "bokeh":
        return None

    # Extract E & N positions and labels
    numbers = []
    names = []
    e_list = []
    n_list = []
    for ant in antennas:
        numbers.append(str(ant.id.number))
        names.append(ant.id.name)
        e, n, _u = ant.position_enu_m
        e_list.append(float(e))
        n_list.append(float(n))

    source = ColumnDataSource(
        {"E": e_list, "N": n_list, "Number": numbers, "Name": names}
    )

    p = figure(
        width=800,
        height=700,
        title="Antenna Layout (E vs N)",
        x_axis_label="E (m)",
        y_axis_label="N (m)",
        match_aspect=True,
    )
    r = p.scatter(x="E", y="N", size=6, source=source, alpha=0.8)

    # Add light labels for small arrays (kept off for large arrays to avoid clutter)
    if len(e_list) <= 100:
        labels = LabelSet(
            x="E",
            y="N",
            text="Number",
            x_offset=4,
            y_offset=4,
            text_font_size="8pt",
            text_alpha=0.8,
            source=source,
        )
        p.add_layout(labels)

    p.add_tools(
        HoverTool(
            renderers=[r],
            tooltips=[
                ("Number", "@Number"),
                ("Name", "@Name"),
                ("E", "@E"),
                ("N", "@N"),
            ],
        )
    )

    _persist_bokeh_document(
        p,
        filename=ANTENNA_LAYOUT_FILENAME,
        title="Antenna Layout (2D)",
        save_flag=save_simulation_data,
        folder_path=folder_path,
        open_flag=open_in_browser,
        save_message="Saved antenna layout plot to",
    )

    return p


def plot_antenna_layout_3d_plotly(
    antennas,
    save_simulation_data=False,
    folder_path=None,
    open_in_browser=True,
):
    """
    Create an interactive 3D scatter (E,N,U) using Plotly and save as standalone HTML.

    Parameters:
    - antennas: canonical resolved antenna tuple.
    - save_simulation_data (bool): if True and folder_path provided, saves HTML there.
    - folder_path (str or None): directory to save HTML when saving.
    - open_in_browser (bool): open the HTML in a browser.

    Returns:
    - The output HTML file path (str) or None on failure.
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception as exc:
        logger.warning(
            f"Plotly not available for 3D antenna layout ({exc}); skipping 3D plot."
        )
        return None

    if not (save_simulation_data and folder_path):
        raise ResultPlotContractError(
            "the 3D antenna layout requires an explicit output folder"
        )

    e, n, u, hover = [], [], [], []
    for ant in antennas:
        ee, nn, uu = ant.position_enu_m
        e.append(float(ee))
        n.append(float(nn))
        u.append(float(uu))
        num = ant.id.number
        name = ant.id.name
        hover.append(f"{num} {name}")

    # Use antenna numbers as labels on points
    labels = [str(ant.id.number) for ant in antennas]
    fig = go.Figure()
    # Compute symmetric, equal ranges around origin for all axes
    u_scaled = list(u)
    max_abs = 1.0
    if e or n or u_scaled:
        max_abs = max(
            max((abs(val) for val in e), default=0.0),
            max((abs(val) for val in n), default=0.0),
            max((abs(val) for val in u_scaled), default=0.0),
        )
        if max_abs <= 0:
            max_abs = 1.0
    pad_factor = 0.1
    L = max_abs * (1.0 + pad_factor)
    xr = [-L, L]
    yr = [-L, L]
    zr = [-L, L]

    fig.update_layout(
        title="Antenna Layout (3D: E, N, U)",
        scene={
            "xaxis": {
                "title": "E (m)",
                "range": xr,
                "zeroline": True,
                "showgrid": True,
            },
            "yaxis": {
                "title": "N (m)",
                "range": yr,
                "zeroline": True,
                "showgrid": True,
            },
            "zaxis": {
                "title": "U (m)",
                "range": zr,
                "zeroline": True,
                "showgrid": True,
            },
            "aspectmode": "cube",
        },
        margin={"l": 20, "r": 20, "b": 20, "t": 40},
    )

    # Add origin and axis markers/labels (+E/-E, +N/-N, +U/-U)
    try:
        fig.add_trace(
            go.Scatter3d(
                x=[xr[0], xr[1]],
                y=[0, 0],
                z=[0, 0],
                mode="lines",
                line={"color": "red", "width": 4},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[yr[0], yr[1]],
                z=[0, 0],
                mode="lines",
                line={"color": "green", "width": 4},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[zr[0], zr[1]],
                mode="lines",
                line={"color": "blue", "width": 4},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Endpoints with labels
        fig.add_trace(
            go.Scatter3d(
                x=[xr[1], xr[0]],
                y=[0, 0],
                z=[0, 0],
                mode="markers+text",
                marker={"size": 3, "color": "red"},
                text=["+E", "-E"],
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[yr[1], yr[0]],
                z=[0, 0],
                mode="markers+text",
                marker={"size": 3, "color": "green"},
                text=["+N", "-N"],
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[zr[1], zr[0]],
                mode="markers+text",
                marker={"size": 3, "color": "blue"},
                text=["+U", "-U"],
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Origin marker
        fig.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers+text",
                marker={"size": 4, "color": "black"},
                text=["Origin"],
                textposition="bottom center",
                hoverinfo="skip",
                showlegend=False,
            )
        )
    except Exception:
        # If any of these fail, continue without axis adornments
        pass

    # Add antenna diameter disks (EN plane at each U).
    try:
        diameters = [float(ant.diameter_m) for ant in antennas]
        # Use fewer segments for performance, and batch all circles in one trace
        nseg = 8
        batch_x: list = []
        batch_y: list = []
        batch_z: list = []
        for xi, yi, zi, di in zip(e, n, u, diameters, strict=False):
            if di is None or not np.isfinite(di) or di <= 0:
                continue
            r = di / 2.0
            theta = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
            ring_x = xi + r * np.cos(theta)
            ring_y = yi + r * np.sin(theta)
            ring_z = np.full_like(theta, zi)
            # close circle
            ring_x = np.append(ring_x, ring_x[0])
            ring_y = np.append(ring_y, ring_y[0])
            ring_z = np.append(ring_z, ring_z[0])
            # append to batch with None separator
            batch_x.extend(ring_x.tolist() + [None])
            batch_y.extend(ring_y.tolist() + [None])
            batch_z.extend(ring_z.tolist() + [None])
        if batch_x:
            fig.add_trace(
                go.Scatter3d(
                    x=batch_x,
                    y=batch_y,
                    z=batch_z,
                    mode="lines",
                    line={"color": "#1f2a44", "width": 3},  # darker lines
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    except Exception:
        pass

    # Finally add the antenna points with number labels on top
    fig.add_trace(
        go.Scatter3d(
            x=e,
            y=n,
            z=u_scaled,
            mode="markers+text",
            marker={
                "size": 4,
                "opacity": 0.85,
                "color": u,
                "colorscale": "Viridis",
                "showscale": False,
            },
            text=labels,
            textposition="top center",
            textfont={"size": 8, "color": "#222"},
            hovertemplate=(
                "Ant %{text}<br>E=%{x:.2f} m<br>N=%{y:.2f} m<br>U=%{z:.2f} m"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    html_path = os.path.join(str(folder_path), "antenna_layout_3d.html")

    try:
        # Build centered HTML wrapper with fixed figure size
        inner = pio.to_html(
            fig,
            include_plotlyjs="cdn",
            full_html=False,
            default_width="1200px",
            default_height="800px",
        )
        centered = f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Antenna Layout (3D)</title>
  </head>
  <body style="margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:white;">{inner}</body>
  </html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(centered)
        if open_in_browser:
            webbrowser.open(Path(html_path).resolve().as_uri())
        logger.debug(f"Saved antenna 3D layout (Plotly) to {html_path}")
        return html_path
    except Exception as exc:
        logger.warning(f"Failed to save 3D Plotly layout ({exc})")
        return None
