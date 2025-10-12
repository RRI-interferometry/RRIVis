# src/plot.py

from bokeh.plotting import figure
from bokeh.palettes import Turbo256, Inferno256
from bokeh.layouts import column
from bokeh.models import (
    DatetimeTicker,
    HoverTool,
    Legend,
    ColorBar,
    LinearColorMapper,
    ColumnDataSource,
    LabelSet,
)
from bokeh.io import save

# Note: avoid Matplotlib here to keep plotting browser-native via Bokeh/Plotly
import numpy as np
import os
from bokeh.resources import CDN
from astropy.time import Time
import tempfile
import webbrowser
from pathlib import Path


def plot_visibility(
    moduli_over_time,
    phases_over_time,
    baselines,
    mjd_time_points,
    freqs,
    total_seconds,
    plotting="bokeh",
    save_simulation_data=False,
    folder_path=None,
    angle_unit="radians",
    open_in_browser=False,
):
    """
    Create plots of the modulus and phase of visibility versus time for each baseline.

    Parameters:
    moduli_over_time (dict): Dictionary of visibility moduli over time for each baseline.
    phases_over_time (dict): Dictionary of visibility phases over time for each baseline.
    baselines (dict): Dictionary of baselines between antennas.
    time_points (ndarray): Array of time points corresponding to the observations.
    freqs (ndarray): Array of frequencies used in the observations.
    total_seconds (float): Total duration of the observation in seconds.
    plotting (str): Plotting library to use ('matplotlib' or 'bokeh').
    angle_unit (str): Unit for displaying angles ('degrees' or 'radians').
    open_in_browser (bool): Open the rendered HTML in the default browser.

    Returns:
    Figure object: The generated plot figure(s) based on the specified plotting library.
    """

    # Convert MJD to human-readable datetime
    time_points_datetime = Time(mjd_time_points, format="mjd").to_datetime()

    baseline_keys = list(baselines.keys())
    colors = Turbo256

    # Convert phase values based on angle_unit
    phases_converted = {}
    for key in phases_over_time:
        if angle_unit == "degrees":
            phases_converted[key] = np.degrees(phases_over_time[key])
        else:
            phases_converted[key] = phases_over_time[key]

    # Determine y-axis label for phase plots
    phase_label = f"Phase of Visibility ({angle_unit})"

    if plotting == "bokeh":
        # Bokeh plotting
        plots = []
        for key in baseline_keys:
            # # Check if polarization is included by examining the shape of the data
            # if moduli_over_time[key].shape[-1] == 2:
            #     # Ex component
            #     p_mod_ex = figure(
            #         width=800,
            #         height=300,
            #         title=f"Modulus of Visibility (Ex) vs Time for Baseline {key}",
            #     )
            #     p_mod_ex.line(
            #         time_points,
            #         moduli_over_time[key][:, 0, 0],
            #         line_width=2,
            #         legend_label=f"Ex Baseline {key}",
            #     )
            #     p_mod_ex.xaxis.axis_label = "Time (seconds)"
            #     p_mod_ex.yaxis.axis_label = "Modulus of Visibility"
            #     p_mod_ex.legend.location = "top_left"

            #     # Ey component
            #     p_mod_ey = figure(
            #         width=800,
            #         height=300,
            #         title=f"Modulus of Visibility (Ey) vs Time for Baseline {key}",
            #     )
            #     p_mod_ey.line(
            #         time_points,
            #         moduli_over_time[key][:, 0, 1],
            #         line_width=2,
            #         legend_label=f"Ey Baseline {key}",
            #     )
            #     p_mod_ey.xaxis.axis_label = "Time (seconds)"
            #     p_mod_ey.yaxis.axis_label = "Modulus of Visibility"
            #     p_mod_ey.legend.location = "top_left"

            #     # Phase components for Ex and Ey
            #     p_phase_ex = figure(
            #         width=800,
            #         height=300,
            #         title=f"Phase of Visibility (Ex) vs Time for Baseline {key}",
            #     )
            #     p_phase_ex.line(
            #         time_points,
            #         phases_over_time[key][:, 0, 0],
            #         line_width=2,
            #         legend_label=f"Ex Baseline {key}",
            #     )
            #     p_phase_ex.xaxis.axis_label = "Time (seconds)"
            #     p_phase_ex.yaxis.axis_label = "Phase of Visibility (radians)"
            #     p_phase_ex.legend.location = "top_left"

            #     p_phase_ey = figure(
            #         width=800,
            #         height=300,
            #         title=f"Phase of Visibility (Ey) vs Time for Baseline {key}",
            #     )
            #     p_phase_ey.line(
            #         time_points,
            #         phases_over_time[key][:, 0, 1],
            #         line_width=2,
            #         legend_label=f"Ey Baseline {key}",
            #     )
            #     p_phase_ey.xaxis.axis_label = "Time (seconds)"
            #     p_phase_ey.yaxis.axis_label = "Phase of Visibility (radians)"
            #     p_phase_ey.legend.location = "top_left"

            #     plots.extend([p_mod_ex, p_mod_ey, p_phase_ex, p_phase_ey])
            # else:

            # Define a ticker with finer granularity (e.g., hourly ticks)
            datetime_ticker = DatetimeTicker(
                desired_num_ticks=12
            )  # Adjust `desired_num_ticks` as needed

            # No polarization, single component
            p_mod = figure(
                width=800,
                height=300,
                title=f"Modulus of Visibility vs Time for Baseline {key}",
                x_axis_type="datetime",
            )
            p_mod.line(
                time_points_datetime,
                moduli_over_time[key][:, 0],
                line_width=2,
                legend_label=f"Baseline {key}",
            )
            p_mod.xaxis.axis_label = "Time"
            p_mod.yaxis.axis_label = "Modulus of Visibility"
            p_mod.legend.location = "top_left"
            p_mod.xaxis.ticker = datetime_ticker

            p_phase = figure(
                width=800,
                height=300,
                title=f"Phase of Visibility vs Time for Baseline {key}",
                x_axis_type="datetime",
            )
            p_phase.line(
                time_points_datetime,
                np.unwrap(phases_converted[key][:, 0]),
                line_width=2,
                legend_label=f"Baseline {key}",
            )
            p_phase.xaxis.axis_label = "Time"
            p_phase.yaxis.axis_label = phase_label
            p_phase.legend.location = "top_left"
            p_phase.xaxis.ticker = datetime_ticker

            plots.append(p_mod)
            plots.append(p_phase)

        combined_mod = figure(
            width=1400,
            height=1400,
            title="Modulus of Visibility vs Time for All Baselines",
            x_axis_type="datetime",
        )

        lines = []
        for idx, key in enumerate(baseline_keys):
            color = colors[int((idx / len(baseline_keys)) * 255)]
            line = combined_mod.line(
                time_points_datetime,
                moduli_over_time[key][:, 0],
                line_width=2,
                color=color,
                name=str(key),
            )
            lines.append((f"Baseline {key}", [line]))

            hover = HoverTool(
                renderers=[line],
                tooltips=[
                    ("Time", "@x{%F %T}"),
                    ("Value", "@y"),
                    ("Baseline", str(key)),
                ],
                formatters={"@x": "datetime"},
                mode="mouse",
            )
            combined_mod.add_tools(hover)

        combined_mod.xaxis.axis_label = "Time"
        combined_mod.yaxis.axis_label = "Modulus of Visibility"
        combined_mod.xaxis.ticker = DatetimeTicker(desired_num_ticks=12)

        legend = Legend(
            items=lines,
            location="center",
            click_policy="hide",
            title="Baselines",
        )
        legend.ncols = 10
        combined_mod.add_layout(legend, "below")

        # combined_mod.xaxis.axis_label = "Time"
        # combined_mod.yaxis.axis_label = "Modulus of Visibility"
        # combined_mod.legend.label_text_font_size = "8pt"
        # combined_mod.legend.spacing = 1
        # combined_mod.legend.location = "top_left"
        # combined_mod.legend.click_policy = "hide"  # Allow toggling baselines on/off
        # combined_mod.xaxis.ticker = DatetimeTicker(desired_num_ticks=12)

        # Combined Phase vs Time
        combined_phase = figure(
            width=1400,
            height=1400,
            title="Combined Phase of Visibility vs Time for All Baselines",
            x_axis_type="datetime",
        )
        lines = []
        for idx, key in enumerate(baseline_keys):
            color = colors[int((idx / len(baseline_keys)) * 255)]
            line = combined_phase.line(
                time_points_datetime,
                np.unwrap(phases_converted[key][:, 0]),
                line_width=2,
                color=color,
                # legend_label=f"Baseline {key}",
                name=str(key),  # Convert key to string
            )
            lines.append((f"Baseline {key}", [line]))

            # Add hover tool for this line
            hover = HoverTool(
                renderers=[line],
                tooltips=[
                    ("Time", "@x{%F %T}"),  # Time in human-readable format
                    ("Value", "@y"),  # Value (phase)
                    ("Baseline", str(key)),  # Convert key to string for tooltip
                ],
                formatters={"@x": "datetime"},  # Formatter for time
                mode="mouse",
            )
            combined_phase.add_tools(hover)

        combined_phase.xaxis.axis_label = "Time"
        combined_phase.yaxis.axis_label = phase_label
        # combined_phase.legend.location = "top_left"
        # combined_phase.legend.label_text_font_size = "8pt"
        # combined_phase.legend.spacing = 1
        # combined_phase.legend.click_policy = "hide"  # Allow toggling baselines on/off
        combined_phase.xaxis.ticker = DatetimeTicker(desired_num_ticks=12)

        # Create a scrollable legend
        legend = Legend(
            items=lines,  # Use the collected line renderers
            location="center",
            click_policy="hide",  # Allow clicking to hide/show lines
            title="Baselines",
        )

        legend.ncols = 10

        # Restrict legend height to make it scrollable
        combined_phase.add_layout(legend, "below")

        plots.append(combined_mod)
        plots.append(combined_phase)

        # Combine plots into a column
        plot_column = column(*plots)

        _persist_bokeh_document(
            plot_column,
            filename="visibility-phase-lsts.html",
            title="Visibility/Phase Plots",
            save_flag=save_simulation_data,
            folder_path=folder_path,
            open_flag=open_in_browser,
            save_message="Saved visibility plots column to",
        )

        return plot_column

    else:
        return None
    #     # Matplotlib plotting
    #     num_baselines = len(baselines)
    #     fig1, ax1 = plt.subplots(num_baselines, 2, figsize=(15, 5 * num_baselines))

    #     for i, key in enumerate(baseline_keys):
    #         # # If polarization is included, plot both Ex and Ey components
    #         # if moduli_over_time[key].shape[-1] == 2:
    #         #     ax1[i, 0].plot(
    #         #         time_points,
    #         #         moduli_over_time[key][:, 0, 0],
    #         #         marker="o",
    #         #         label=f"Ex Baseline {key}",
    #         #     )
    #         #     ax1[i, 0].plot(
    #         #         time_points,
    #         #         moduli_over_time[key][:, 0, 1],
    #         #         marker="o",
    #         #         label=f"Ey Baseline {key}",
    #         #     )
    #         #     ax1[i, 0].set_xlabel("Time (seconds)")
    #         #     ax1[i, 0].set_ylabel("Modulus of Visibility")
    #         #     ax1[i, 0].set_title(
    #         #         f"Modulus of Visibility (Ex, Ey) vs Time for Baseline {key}"
    #         #     )
    #         #     ax1[i, 0].legend()
    #         #     ax1[i, 0].grid(True)

    #         #     ax1[i, 1].plot(
    #         #         time_points,
    #         #         phases_over_time[key][:, 0, 0],
    #         #         marker="o",
    #         #         label=f"Ex Baseline {key}",
    #         #     )
    #         #     ax1[i, 1].plot(
    #         #         time_points,
    #         #         phases_over_time[key][:, 0, 1],
    #         #         marker="o",
    #         #         label=f"Ey Baseline {key}",
    #         #     )
    #         #     ax1[i, 1].set_xlabel("Time (seconds)")
    #         #     ax1[i, 1].set_ylabel("Phase of Visibility (radians)")
    #         #     ax1[i, 1].set_title(
    #         #         f"Phase of Visibility (Ex, Ey) vs Time for Baseline {key}"
    #         #     )
    #         #     ax1[i, 1].legend()
    #         #     ax1[i, 1].grid(True)
    #         # else:
    #         # No polarization, single component
    #         # ax1[i, 0].plot(
    #         #     time_points,
    #         #     moduli_over_time[key][:, 0],
    #         #     marker="o",
    #         #     label=f"Baseline {key}",
    #         # )
    #         # ax1[i, 0].set_xlabel("Time")
    #         # ax1[i, 0].set_ylabel("Modulus of Visibility")
    #         # ax1[i, 0].set_title(f"Modulus of Visibility vs Time for Baseline {key}")
    #         # ax1[i, 0].legend()
    #         # ax1[i, 0].grid(True)

    #         # ax1[i, 1].plot(
    #         #     time_points,
    #         #     np.unwrap(phases_over_time[key][:, 0]),
    #         #     marker="o",
    #         #     label=f"Baseline {key}",
    #         # )
    #         # ax1[i, 1].set_xlabel("Time")
    #         # ax1[i, 1].set_ylabel("Phase of Visibility (radians)")
    #         # ax1[i, 1].set_title(f"Phase of Visibility vs Time for Baseline {key}")
    #         # ax1[i, 1].legend()
    #         # ax1[i, 1].grid(True)

    #     plt.tight_layout()
    #     return fig1


def plot_heatmaps(
    moduli_over_time,
    phases_over_time,
    baselines,
    freqs,
    total_seconds,
    mjd_time_points,
    plotting="bokeh",
    save_simulation_data=False,
    folder_path=None,
    open_in_browser=False,
):
    """
    Create heatmaps for visibility modulus and phase over time and frequency.

    Parameters:
    moduli_over_time (dict): Dictionary of visibility moduli over time for each baseline.
    phases_over_time (dict): Dictionary of visibility phases over time for each baseline.
    baselines (dict): Dictionary of baselines between antennas.
    freqs (ndarray): Array of frequencies used in the observations.
    total_seconds (float): Total duration of the observation in seconds.
    plotting (str): Plotting library to use ('matplotlib' or 'bokeh').
    open_in_browser (bool): Open the rendered HTML in the default browser.

    Returns:
    Figure object: The generated heatmap figure(s) based on the specified plotting library.
    """

    # Convert MJD to human-readable datetime
    time_points_datetime = Time(mjd_time_points, format="mjd").to_datetime()
    baseline_keys = list(baselines.keys())

    if plotting == "bokeh":
        plots = []
        for key in baseline_keys:
            # if moduli_over_time[key].shape[-1] == 2:
            #     # Combine Ex and Ey components
            #     moduli_total = np.sqrt(
            #         moduli_over_time[key][:, :, 0] ** 2
            #         + moduli_over_time[key][:, :, 1] ** 2
            #     )
            #     phases_total = np.sqrt(
            #         phases_over_time[key][:, :, 0] ** 2
            #         + phases_over_time[key][:, :, 1] ** 2
            #     )
            # else:
            moduli_total = moduli_over_time[key]
            phases_total = np.unwrap(phases_over_time[key], axis=0)

            # Create a LinearColorMapper for the heatmap
            modulus_mapper = LinearColorMapper(
                palette=Inferno256, low=moduli_total.min(), high=moduli_total.max()
            )
            phase_mapper = LinearColorMapper(
                palette=Inferno256, low=phases_total.min(), high=phases_total.max()
            )
            # Ensure data is in the correct format (list of 2D arrays)
            moduli_image = [
                moduli_total.T
            ]  # Transpose to match Bokeh's image orientation
            phases_image = [phases_total.T]

            # Modulus heatmap
            p_mod = figure(
                width=800,
                height=300,
                title=f"Modulus of Visibility Heatmap for Baseline {key}",
                x_axis_type="datetime",
            )
            p_mod.image(
                image=moduli_image,
                x=time_points_datetime[0],  # Start of the datetime range
                y=freqs[0] / 1e6,  # Start of frequency in MHz
                dw=(time_points_datetime[-1] - time_points_datetime[0]).total_seconds()
                * 1e3,  # Time duration in ms
                dh=(freqs[-1] - freqs[0]) / 1e6,  # Frequency range in MHz
                color_mapper=modulus_mapper,
            )
            p_mod.xaxis.axis_label = "Time"
            p_mod.yaxis.axis_label = "Frequency (MHz)"
            p_mod.xaxis.ticker = DatetimeTicker(desired_num_ticks=12)  # Add finer ticks

            # Add color bar for modulus heatmap
            color_bar_mod = ColorBar(color_mapper=modulus_mapper, location=(0, 0))
            p_mod.add_layout(color_bar_mod, "right")

            # Phase heatmap
            p_phase = figure(
                width=800,
                height=300,
                title=f"Phase of Visibility Heatmap for Baseline {key}",
                x_axis_type="datetime",
            )
            p_phase.image(
                image=phases_image,
                x=time_points_datetime[0],  # Start of the datetime range
                y=freqs[0] / 1e6,  # Start of frequency in MHz
                dw=(time_points_datetime[-1] - time_points_datetime[0]).total_seconds()
                * 1e3,  # Time duration in ms
                dh=(freqs[-1] - freqs[0]) / 1e6,  # Frequency range in MHz
                color_mapper=phase_mapper,
            )
            p_phase.xaxis.axis_label = "Time"
            p_phase.yaxis.axis_label = "Frequency (MHz)"
            p_phase.xaxis.ticker = DatetimeTicker(
                desired_num_ticks=12
            )  # Add finer ticks

            # Add color bar for phase heatmap
            color_bar_phase = ColorBar(color_mapper=phase_mapper, location=(0, 0))
            p_phase.add_layout(color_bar_phase, "right")

            plots.append(p_mod)
            plots.append(p_phase)

        # Combine plots into a column
        plot_column = column(*plots, sizing_mode="stretch_both")

        _persist_bokeh_document(
            plot_column,
            filename="heatmaps-freq-time.html",
            title="Visibility Heatmaps",
            save_flag=save_simulation_data,
            folder_path=folder_path,
            open_flag=open_in_browser,
            save_message="Saved heatmaps column to",
        )

        return plot_column

    else:
        return None
        # # Matplotlib plotting
        # num_baselines = len(baselines)
        # fig2, ax2 = plt.subplots(num_baselines, 2, figsize=(15, 5 * num_baselines))

        # for i, key in enumerate(baseline_keys):
        #     # if moduli_over_time[key].shape[-1] == 2:
        #     #     # Combine Ex and Ey components
        #     #     moduli_total = np.sqrt(
        #     #         moduli_over_time[key][:, :, 0] ** 2
        #     #         + moduli_over_time[key][:, :, 1] ** 2
        #     #     )
        #     #     phases_total = np.sqrt(
        #     #         phases_over_time[key][:, :, 0] ** 2
        #     #         + phases_over_time[key][:, :, 1] ** 2
        #     #     )
        #     # else:
        #     moduli_total = moduli_over_time[key]
        #     phases_total = phases_over_time[key]

        #     # Modulus heatmap
        #     im0 = ax2[i, 0].imshow(
        #         moduli_total,
        #         aspect="auto",
        #         origin="lower",
        #         extent=[freqs[0] / 1e6, freqs[-1] / 1e6, 0, total_seconds],
        #         cmap="twilight",  # Updated color scheme to 'twilight'
        #     )
        #     ax2[i, 0].set_xlabel("Frequency (MHz)")
        #     ax2[i, 0].set_ylabel("Time (seconds)")
        #     ax2[i, 0].set_title(f"Modulus of Visibility for Baseline {key}")
        #     fig2.colorbar(im0, ax=ax2[i, 0])

        #     # Phase heatmap
        #     im1 = ax2[i, 1].imshow(
        #         np.unwrap(phases_total),
        #         aspect="auto",
        #         origin="lower",
        #         extent=[freqs[0] / 1e6, freqs[-1] / 1e6, 0, total_seconds],
        #         cmap="twilight",  # Updated color scheme to 'twilight'
        #     )
        #     ax2[i, 1].set_xlabel("Frequency (MHz)")
        #     ax2[i, 1].set_ylabel("Time (seconds)")
        #     ax2[i, 1].set_title(f"Phase of Visibility for Baseline {key}")
        #     fig2.colorbar(im1, ax=ax2[i, 1])

        # plt.tight_layout()
        # return fig2


def _persist_bokeh_document(
    doc,
    filename,
    title,
    save_flag,
    folder_path,
    open_flag,
    save_message=None,
):
    """
    Save a Bokeh document when persistence or browser viewing is requested.
    Returns the path if a file was written, else None.
    """
    needs_file = (save_flag and folder_path) or open_flag
    target_path = None

    if needs_file:
        if save_flag and folder_path:
            target_dir = folder_path
        else:
            target_dir = tempfile.mkdtemp(prefix="rrivis_")
        target_path = os.path.join(target_dir, filename)

        save(doc, filename=target_path, resources=CDN, title=title)

        if save_flag and folder_path:
            message = save_message or f"Saved {title} to"
            print(f"{message} {target_path}")

        if open_flag:
            webbrowser.open(Path(target_path).resolve().as_uri())

    return target_path


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
    - antennas (dict): antenna metadata dict with 'Position' (E,N,U), 'Number', 'Name'.
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
    for ant in antennas.values():
        numbers.append(str(ant.get("Number", "?")))
        names.append(str(ant.get("Name", "")))
        e, n, _u = ant.get("Position", (0.0, 0.0, 0.0))
        e_list.append(float(e))
        n_list.append(float(n))

    source = ColumnDataSource(dict(E=e_list, N=n_list, Number=numbers, Name=names))

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
        filename="antenna_layout.html",
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
    - antennas (dict): antenna metadata dict with 'Position' (E,N,U), 'Number', 'Name'.
    - save_simulation_data (bool): if True and folder_path provided, saves HTML there; else uses a temp dir.
    - folder_path (str or None): directory to save HTML when saving.
    - open_in_browser (bool): open the HTML in a browser.

    Returns:
    - The output HTML file path (str) or None on failure.
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception as exc:
        print(
            f"Warning: Plotly not available for 3D antenna layout ({exc}); skipping 3D plot."
        )
        return None

    e, n, u, hover = [], [], [], []
    scale_z = 1.0  # no vertical exaggeration
    for ant in antennas.values():
        ee, nn, uu = ant.get("Position", (0.0, 0.0, 0.0))
        e.append(float(ee))
        n.append(float(nn))
        u.append(float(uu))
        num = ant.get("Number", "?")
        name = ant.get("Name", "")
        hover.append(f"{num} {name}")

    # Use antenna numbers as labels on points
    labels = [str(ant.get("Number", "?")) for ant in antennas.values()]
    fig = go.Figure()
    # Compute symmetric, equal ranges around origin for all axes
    u_scaled = [val for val in u]
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
        scene=dict(
            xaxis=dict(title="E (m)", range=xr, zeroline=True, showgrid=True),
            yaxis=dict(title="N (m)", range=yr, zeroline=True, showgrid=True),
            zaxis=dict(title="U (m)", range=zr, zeroline=True, showgrid=True),
            aspectmode="cube",
        ),
        margin=dict(l=20, r=20, b=20, t=40),
    )

    # Add origin and axis markers/labels (+E/-E, +N/-N, +U/-U)
    try:
        fig.add_trace(
            go.Scatter3d(
                x=[xr[0], xr[1]],
                y=[0, 0],
                z=[0, 0],
                mode="lines",
                line=dict(color="red", width=4),
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
                line=dict(color="green", width=4),
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
                line=dict(color="blue", width=4),
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
                marker=dict(size=3, color="red"),
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
                marker=dict(size=3, color="green"),
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
                marker=dict(size=3, color="blue"),
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
                marker=dict(size=4, color="black"),
                text=["Origin"],
                textposition="bottom center",
                hoverinfo="skip",
                showlegend=False,
            )
        )
    except Exception:
        # If any of these fail (e.g., textposition in some plotly versions), continue without axis adornments
        pass

    # Add antenna diameter disks (EN plane at each U). Use filled Mesh3d for up to 150 ants; else draw perimeters.
    try:
        diameters = [
            float(ant.get("diameter")) if ant.get("diameter") is not None else None
            for ant in antennas.values()
        ]
        # Use fewer segments for performance, and batch all circles in one trace
        nseg = 8
        batch_x: list = []
        batch_y: list = []
        batch_z: list = []
        for xi, yi, zi, di in zip(e, n, u, diameters):
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
                    line=dict(color="#1f2a44", width=3),  # darker lines
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
            marker=dict(
                size=4, opacity=0.85, color=u, colorscale="Viridis", showscale=False
            ),
            text=labels,
            textposition="top center",
            textfont=dict(size=8, color="#222"),
            hovertemplate="Ant %{text}<br>E=%{x:.2f} m<br>N=%{y:.2f} m<br>U=%{z:.2f} m<extra></extra>",
            showlegend=False,
        )
    )

    # Decide output path
    if save_simulation_data and folder_path:
        out_dir = folder_path
    else:
        out_dir = tempfile.mkdtemp(prefix="rrivis_")
    html_path = os.path.join(out_dir, "antenna_layout_3d.html")

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
        print(f"Saved antenna 3D layout (Plotly) to {html_path}")
        return html_path
    except Exception as exc:
        print(f"Warning: Failed to save 3D Plotly layout ({exc})")
        return None


def plot_modulus_vs_frequency(
    moduli_over_time,
    phases_over_time,
    baselines,
    freqs,
    mjd_time_points,
    plotting="bokeh",
    save_simulation_data=False,
    folder_path=None,
    open_in_browser=False,
):
    """
    Create plots of the modulus of visibility versus frequency at a specific time point.

    Parameters:
    moduli_over_time (dict): Dictionary of visibility moduli over time for each baseline.
    phases_over_time (dict): Dictionary of visibility phases over time for each baseline.
    baselines (dict): Dictionary of baselines between antennas.
    freqs (ndarray): Array of frequencies used in the observations.
    time_index (int): Index of the time point at which to plot the modulus.
    plotting (str): Plotting library to use ('matplotlib' or 'bokeh').
    open_in_browser (bool): Open the rendered HTML in the default browser.

    Returns:
    Figure object: The generated plot figure(s) based on the specified plotting library.
    """
    baseline_keys = list(baselines.keys())
    colors = Turbo256

    # Convert MJD to human-readable datetime
    time_points_datetime = Time(mjd_time_points, format="mjd").to_datetime()

    # Find the time index of maximum modulus for each baseline
    max_time_indices = {
        key: np.argmax(moduli_over_time[key].max(axis=1)) for key in baseline_keys
    }

    if plotting == "bokeh":
        plots = []
        for key in baseline_keys:
            max_time_index = max_time_indices[key]
            max_time_utc = time_points_datetime[max_time_index]
            # Modulus
            # if moduli_over_time[key].shape[-1] == 2:
            #     # Combine Ex and Ey components at the specified time index
            #     moduli_total = np.sqrt(
            #         moduli_over_time[key][time_index, :, 0] ** 2
            #         + moduli_over_time[key][time_index, :, 1] ** 2
            #     )
            # else:
            moduli_total = moduli_over_time[key][max_time_index, :]

            p_mod = figure(
                width=800,
                height=300,
                title=f"Modulus of Visibility vs Frequency for Baseline {key} at {max_time_utc}",
            )
            p_mod.line(
                freqs / 1e6, moduli_total, line_width=2, legend_label=f"Baseline {key}"
            )
            p_mod.xaxis.axis_label = "Frequency (MHz)"
            p_mod.yaxis.axis_label = "Modulus of Visibility"
            p_mod.legend.location = "top_left"

            # Phase
            phases_total = np.unwrap(phases_over_time[key][max_time_index, :])
            p_phase = figure(
                width=800,
                height=300,
                title=f"Phase of Visibility vs Frequency for Baseline {key} at {max_time_utc}",
            )
            p_phase.line(
                freqs / 1e6, phases_total, line_width=2, legend_label=f"Baseline {key}"
            )
            p_phase.xaxis.axis_label = "Frequency (MHz)"
            p_phase.yaxis.axis_label = "Phase of Visibility (radians)"
            p_phase.legend.location = "top_left"

            plots.append(p_mod)
            plots.append(p_phase)

        # Find the global maximum modulus time index across all baselines
        global_max_time_index = max(max_time_indices.values())
        global_max_time_utc = time_points_datetime[global_max_time_index]

        # Combined Modulus vs Frequency
        combined_mod = figure(
            width=1400,
            height=1400,
            title=f"Modulus of Visibility vs Frequency for All Baselines at {global_max_time_utc}",
        )

        lines = []
        for idx, key in enumerate(baseline_keys):
            max_time_index = max_time_indices[key]
            color = colors[int((idx / len(baseline_keys)) * 255)]
            line = combined_mod.line(
                freqs / 1e6,
                moduli_over_time[key][max_time_index, :],
                line_width=2,
                color=color,
                name=str(key),
            )
            lines.append((f"Baseline {key}", [line]))

            hover = HoverTool(
                renderers=[line],
                tooltips=[
                    ("Frequency", "@x MHz"),
                    ("Value", "@y"),
                    ("Baseline", str(key)),
                ],
                mode="mouse",
            )
            combined_mod.add_tools(hover)

        combined_mod.xaxis.axis_label = "Frequency (MHz)"
        combined_mod.yaxis.axis_label = "Modulus of Visibility"

        legend = Legend(
            items=lines,
            location="center",
            click_policy="hide",
            title="Baselines",
        )
        legend.ncols = 10
        combined_mod.add_layout(legend, "below")

        # Combined Phase vs Frequency
        combined_phase = figure(
            width=1400,
            height=1400,
            title=f"Phase of Visibility vs Frequency for All Baselines at {global_max_time_utc}",
        )

        lines = []
        for idx, key in enumerate(baseline_keys):
            max_time_index = max_time_indices[key]
            color = colors[int((idx / len(baseline_keys)) * 255)]
            line = combined_phase.line(
                freqs / 1e6,
                np.unwrap(phases_over_time[key][max_time_index, :]),
                line_width=2,
                color=color,
                name=str(key),
            )
            lines.append((f"Baseline {key}", [line]))

            hover = HoverTool(
                renderers=[line],
                tooltips=[
                    ("Frequency", "@x MHz"),
                    ("Value", "@y radians"),
                    ("Baseline", str(key)),
                ],
                mode="mouse",
            )
            combined_phase.add_tools(hover)

        combined_phase.xaxis.axis_label = "Frequency (MHz)"
        combined_phase.yaxis.axis_label = "Phase of Visibility (radians)"

        legend = Legend(
            items=lines,
            location="center",
            click_policy="hide",
            title="Baselines",
        )
        legend.ncols = 10
        combined_phase.add_layout(legend, "below")

        plots.append(combined_mod)
        plots.append(combined_phase)

        plot_column = column(*plots, sizing_mode="stretch_both")

        _persist_bokeh_document(
            plot_column,
            filename="modulus-phase-freq.html",
            title="Visibility Modulus/Phase vs Frequency",
            save_flag=save_simulation_data,
            folder_path=folder_path,
            open_flag=open_in_browser,
            save_message="Saved Modulus vs Frequency column to",
        )

        return plot_column

    else:
        return None
        # # Matplotlib plotting
        # num_baselines = len(baselines)
        # fig3, ax3 = plt.subplots(num_baselines, 1, figsize=(15, 5 * num_baselines))

        # for i, key in enumerate(baseline_keys):
        #     # if moduli_over_time[key].shape[-1] == 2:
        #     #     # Combine Ex and Ey components at the specified time index
        #     #     moduli_total = np.sqrt(
        #     #         moduli_over_time[key][time_index, :, 0] ** 2
        #     #         + moduli_over_time[key][time_index, :, 1] ** 2
        #     #     )
        #     # else:
        #     moduli_total = moduli_over_time[key][time_index, :]

        #     ax3[i].plot(freqs / 1e6, moduli_total, marker="o", label=f"Baseline {key}")
        #     ax3[i].set_xlabel("Frequency (MHz)")
        #     ax3[i].set_ylabel("Modulus of Visibility")
        #     ax3[i].set_title(
        #         f"Modulus of Visibility vs Frequency for Baseline {key} at Time {time_index} seconds"
        #     )
        #     ax3[i].legend()
        #     ax3[i].grid(True)

        # plt.tight_layout()
        # return fig3
